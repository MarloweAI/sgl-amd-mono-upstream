"""Check one sequence's DSpark draft head and Markov block in one launch (``gfx950/markov``'s draft_head, torchrun
TP4 or TP8) against the two launches it replaces: the head projection (torch's bf16 GEMM) then markov_block.
- the base logits: the launch's (its tagged words) against an f32 reference, rounded once (within a bf16 ulp),
  and how many equal the bf16 GEMM's bits;
- the Markov block on the launch's own base logits: markov_block on the same values gives the same tokens, verify
  ids, last tokens, step logits and noise counter (bitwise), greedy and sampled, eager and under graph replay;
- the tokens against the two launches on the GEMM's base logits, over many seeds;
then the time of each (a graph of chained calls, max over ranks)."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

VOCAB, DIM, K = 129280, 256, 5120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--chain", type=int, default=10)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.gfx950 import markov

    gamma, part = args.gamma, VOCAB // tp
    vocab_start = rank * part
    g = torch.Generator(device=dev).manual_seed(7)  # every rank the same tables
    w1 = (torch.randn(VOCAB, DIM, generator=g, device=dev) * 0.05).bfloat16()
    w2 = (torch.randn(VOCAB, DIM, generator=g, device=dev) * 0.5).bfloat16()
    lm = (torch.randn(VOCAB, K, generator=g, device=dev) * 0.02).bfloat16()
    w2s, lms = (
        w2[vocab_start : vocab_start + part],
        lm[vocab_start : vocab_start + part].contiguous(),
    )
    key = 0x1234_5678_9ABC
    ops = markov.ops()
    xa = markov.Exchange(cpu, rank, tp, dev, gamma, part)  # the two launches'
    xb = markov.Exchange(
        cpu, rank, tp, dev, gamma, part, control=markov.MAX_BLOCKS
    )  # the fused launch's
    base_words = torch.zeros(gamma * part, dtype=torch.int64, device=dev)

    def gathered(
        xch, temps, logits
    ):  # every rank's shard written first, and read before the next launch
        torch.cuda.synchronize()
        dist.barrier()
        ops.gather_step_logits(temps, xch.peers, logits, part)
        torch.cuda.synchronize()
        dist.barrier()

    def fused(x, anchor, temps, offset, out, logits, verify, gather=True):
        ops.draft_head(
            x,
            lms,
            base_words,
            w1,
            w2s,
            anchor,
            temps,
            offset,
            key,
            vocab_start,
            out,
            gamma,
            xb.tok,
            xb.control,
            xb.peers,
            rank,
            logits,
            part,
            20.0,
            verify,
        )
        if gather:
            gathered(xb, temps, logits)

    def two(base, anchor, temps, offset, out, logits, verify, gather=True):
        ops.markov_block(
            base,
            w1,
            w2s,
            anchor,
            temps,
            offset,
            key,
            vocab_start,
            out,
            gamma,
            xa.tok,
            xa.control,
            xa.peers,
            rank,
            logits,
            part,
            20.0,
            verify,
        )
        if gather:
            gathered(xa, temps, logits)

    def bufs():
        return (
            torch.zeros(gamma, dtype=torch.int64, device=dev),
            torch.zeros(1, gamma, VOCAB, dtype=torch.float32, device=dev),
            torch.full((1, gamma + 3), -1, dtype=torch.int64, device=dev)[
                :, : gamma + 1
            ],
        )

    fails, ulp_bad, same_bits, total, tok_agree, tok_total = [], 0, 0, 0, 0, 0
    for seed in range(args.seeds):
        gs = torch.Generator(device=dev).manual_seed(
            1000 + seed
        )  # the same on every rank
        x = (torch.randn(gamma, K, generator=gs, device=dev)).bfloat16()
        anchor = torch.randint(0, VOCAB, (1,), generator=gs, device=dev)
        temps = torch.tensor([1.0 if seed % 2 else 0.0], device=dev)
        offset0 = 31 * seed
        # the launch
        offset = torch.tensor([offset0], dtype=torch.int64, device=dev)
        out, logits, verify = bufs()
        fused(x, anchor, temps, offset, out, logits, verify)
        torch.cuda.synchronize()
        ours = (
            (base_words.view(gamma, part) & 0xFFFF)
            .to(torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        # the base logits against f32, rounded once
        ref = x.float() @ lms.float().t()
        hip = F.linear(x, lms)
        err = (ours.float() - ref).abs()
        ulp = (ref.abs().clamp_min(1e-30).log2().floor() - 7).exp2()
        ulp_bad += int((err > ulp).sum())
        same_bits += int((ours.view(torch.int16) == hip.view(torch.int16)).sum())
        total += ours.numel()
        # the Markov block on the launch's own base logits: bitwise
        offset_b = torch.tensor([offset0], dtype=torch.int64, device=dev)
        out_b, logits_b, verify_b = bufs()
        two(
            ours.view(1, gamma, part).contiguous(),
            anchor,
            temps,
            offset_b,
            out_b,
            logits_b,
            verify_b,
        )
        torch.cuda.synchronize()
        ok = (
            torch.equal(out, out_b)
            and torch.equal(verify, verify_b)
            and torch.equal(xa.tok[:1], xb.tok[:1])
            and torch.equal(logits.view(torch.int32), logits_b.view(torch.int32))
            and int(offset.item()) == int(offset_b.item()) == offset0 + 1
        )
        # the tokens of the two launches on the GEMM's base logits
        offset_c = torch.tensor([offset0], dtype=torch.int64, device=dev)
        out_c, logits_c, verify_c = bufs()
        two(
            hip.view(1, gamma, part).contiguous(),
            anchor,
            temps,
            offset_c,
            out_c,
            logits_c,
            verify_c,
        )
        torch.cuda.synchronize()
        tok_agree += int((out == out_c).sum())
        tok_total += gamma
        if seed < 2:  # graph replays with the counter put back
            graph = torch.cuda.CUDAGraph()
            offset.fill_(offset0)
            with torch.cuda.graph(graph):
                fused(x, anchor, temps, offset, out, logits, verify, gather=False)
            for _ in range(5):
                offset.fill_(offset0)
                out.zero_()
                graph.replay()
                torch.cuda.synchronize()
                ok &= torch.equal(out, out_b)
        ok_all = torch.tensor([int(ok)], device=dev)
        dist.all_reduce(ok_all, op=dist.ReduceOp.MIN)
        if not ok_all.item():
            fails.append(seed)
    stats = torch.tensor(
        [ulp_bad, same_bits, total, tok_agree, tok_total],
        dtype=torch.float64,
        device=dev,
    )
    dist.all_reduce(stats)
    if rank == 0:
        print(
            f"base logits past a bf16 ulp of f32: {int(stats[0])} of {int(stats[2])} | equal to the bf16 GEMM's: "
            f"{stats[1] / stats[2]:.5f}"
        )
        print(
            f"Markov on the launch's base logits bitwise: {'PASS' if not fails else f'FAIL seeds {fails}'}"
        )
        print(
            f"tokens equal to the two launches on the GEMM's base logits: {int(stats[3])} of {int(stats[4])} "
            f"(summed over ranks)"
        )

    # timing: greedy, a graph of ``chain`` calls each, max over ranks
    gs = torch.Generator(device=dev).manual_seed(5)
    x = (torch.randn(gamma, K, generator=gs, device=dev)).bfloat16()
    anchor = torch.randint(0, VOCAB, (1,), generator=gs, device=dev)
    temps = torch.zeros(1, device=dev)
    offset = torch.zeros(1, dtype=torch.int64, device=dev)
    out, logits, verify = bufs()
    hip = torch.empty(gamma, part, dtype=torch.bfloat16, device=dev)

    def timed(fn):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.chain):
                fn()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        best = float("inf")
        for _ in range(10):
            a.record()
            graph.replay()
            b.record()
            torch.cuda.synchronize()
            best = min(best, a.elapsed_time(b) * 1e3 / args.chain)
        t = torch.tensor([best], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    t_one = timed(
        lambda: fused(x, anchor, temps, offset, out, logits, verify, gather=False)
    )
    t_two = timed(
        lambda: (
            torch.mm(x, lms.t(), out=hip),
            two(
                hip.view(1, gamma, part),
                anchor,
                temps,
                offset,
                out,
                logits,
                verify,
                gather=False,
            ),
        )
    )
    t_gemm = timed(lambda: torch.mm(x, lms.t(), out=hip))
    temps.fill_(1.0)
    t_one_s = timed(
        lambda: fused(x, anchor, temps, offset, out, logits, verify, gather=False)
    )
    if rank == 0:
        print(
            f"time us: one launch {t_one:.2f} (sampled {t_one_s:.2f}) | bf16 GEMM + markov_block {t_two:.2f} "
            f"(GEMM alone {t_gemm:.2f})"
        )
        print("ALL PASS" if not fails else f"FAILED {fails}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
