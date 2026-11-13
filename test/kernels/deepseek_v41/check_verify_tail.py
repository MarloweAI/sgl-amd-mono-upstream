"""Check the static verify's accept from the ranks' bf16 logits shards (gfx950/vtail, one launch) against SGLang's
chain on the gathered f32 logits (torchrun, TP2 to TP8): the row argmax (``sampling.argmax``), the simulated bonus
(``simulated_bonus_partials_graph``) and the static accept tail (``static_accept_tail``), bitwise, at 1 to 8
requests of 6 rows, greedy, a simulated length with argmax bonus, and a simulated length with temperature-sampled
bonus; ties within and across ranks. SGLang's own shard path (``softcap_row_argmax`` on bf16,
``simulated_bonus_shard_keys``, the keys gathered) is checked the same way. The launch's noise against SGLang's
Triton over every Philox offset a verify uses; graph replays with fresh inputs; then each path's time in a graph.

    torchrun --nproc-per-node 4 check_verify_tail.py [--replays 10] [--reps 200]
"""

import argparse
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

VOCAB, T = 129280, 6


@triton.jit
def _noisy_kernel(x_ptr, out_ptr, n, t, seed, start, BLOCK: tl.constexpr):
    # _simulated_bonus_row_partial_kernel's arithmetic on a flat range of offsets
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m, other=0.0)
    u = tl.rand(seed, start + i)
    tl.store(out_ptr + i, x / tl.maximum(t, 1e-5) - tl.log(-tl.log(u)), mask=m)


def check_noise(ops, dev) -> int:
    """Mismatched noisy values over offsets [0, 8 requests x vocab) at 3 seeds x 3 temperatures."""
    n = 8 * VOCAB
    bad = 0
    g = torch.Generator(device=dev).manual_seed(5)
    for seed in (0, 12345, 2**31 - 1):
        for t in (0.3, 1.0, 1.7):
            x = (torch.randn(n, generator=g, device=dev) * 6).bfloat16().float()
            want, got = torch.empty_like(x), torch.empty_like(x)
            _noisy_kernel[(triton.cdiv(n, 1024),)](x, want, n, t, seed, 0, BLOCK=1024)
            ops.noisy_values(x, got, t, seed, 0)
            bad += int((want.view(torch.int32) != got.view(torch.int32)).sum())
    return bad


class Case:
    def __init__(self, bs, mode):
        self.bs, self.mode = (
            bs,
            mode,
        )  # mode: greedy, sim0 (simulated length, argmax), simt (sampled)


def inputs(case, rank, tp, part, dev, seed):
    """Every rank's shard (each rank makes all of them from one seed: the gathered rows are known everywhere) with
    ties, candidates matching a random prefix of the argmax, lengths, temperatures."""
    bs, rows = case.bs, case.bs * T
    g = torch.Generator(device=dev).manual_seed(seed)
    full = (torch.randn(rows, tp * part, generator=g, device=dev) * 4).bfloat16()
    hi = torch.tensor(30.0, dtype=torch.bfloat16)
    for r in range(rows):
        kind = r % 3
        if kind == 0:  # a tie within one rank's shard
            c = int(torch.randint(0, VOCAB - 9, (1,), generator=g, device=dev))
            full[r, c + 8] = full[r, c] = hi
        elif (
            kind == 1 and tp > 1
        ):  # a tie across ranks: the lower index (lower rank) wins
            c = int(torch.randint(0, part, (1,), generator=g, device=dev))
            full[r, (tp - 1) * part + c] = full[r, c + part // 2] = hi
        if r % 4 == 3:  # -0 and +0 tie at the maximum
            full[r] = -1.0
            full[r, 5] = -0.0
            full[r, 3 * part // 2] = 0.0
    full[:, VOCAB:] = 0  # past the vocab: never read
    local = full[:, rank * part : (rank + 1) * part].contiguous()
    logits = full[:, :VOCAB].float()  # SGLang's gathered f32 logits
    from sglang.kernels.ops.sampling import argmax as row_argmax

    target = row_argmax(logits).view(bs, T)
    cand = torch.empty(bs, T, dtype=torch.int64, device=dev)
    cand[:, 0] = torch.randint(0, VOCAB, (bs,), generator=g, device=dev)
    for r in range(bs):
        m = int(torch.randint(0, T, (1,), generator=g, device=dev))
        cand[r, 1:] = target[r, : T - 1]
        if m < T - 1:
            cand[r, 1 + m] = (target[r, m] + 1) % VOCAB
    seq_lens = torch.randint(100, 9000, (bs,), generator=g, device=dev)
    verify_lens = torch.full((bs,), T, dtype=torch.int64, device=dev)
    verify_lens[-1] = 3
    sim = (
        torch.randint(0, T, (1,), generator=g, device=dev)
        if case.mode != "greedy"
        else None
    )
    temps = torch.zeros(bs, device=dev)
    if case.mode == "simt":
        temps = torch.tensor([0.0, 1.0, 0.6, 1.3, 0.0, 2.0, 1.0, 0.7][:bs], device=dev)
    seed_t = torch.randint(0, 2**31 - 1, (1,), generator=g, device=dev).int()
    gate = torch.tensor([1], dtype=torch.int32, device=dev)
    return dict(
        local=local,
        logits=logits,
        cand=cand,
        seq_lens=seq_lens,
        verify_lens=verify_lens,
        sim=sim,
        temps=temps,
        seed=seed_t,
        gate=gate,
    )


def outs(bs, dev):
    return dict(
        correct_len_out=torch.zeros(8, dtype=torch.int64, device=dev),
        bonus_out=torch.zeros(8, dtype=torch.int64, device=dev),
        cap_trim_lens_out=torch.full((8,), 7, dtype=torch.int32, device=dev),
        commit_lens_out=torch.zeros(8, dtype=torch.int32, device=dev),
        new_seq_lens_out=torch.zeros(8, dtype=torch.int64, device=dev),
        out_tokens_out=torch.full((8, T), -5, dtype=torch.int64, device=dev),
        gated_commit_lens_out=torch.zeros(8, dtype=torch.int32, device=dev),
    )


def sglang_chain(x, bs, o):
    """The static epilogue's tail on the gathered logits (dspark_verify._static_accept_tail)."""
    from sglang.kernels.ops.sampling import argmax as row_argmax
    from sglang.kernels.ops.speculative.dspark.simulated_bonus import (
        simulated_bonus_partials_graph,
    )
    from sglang.kernels.ops.speculative.dspark.static_accept_tail import (
        static_accept_tail,
    )

    target = parts = None
    if x["mode"] == "simt":
        parts = (
            *simulated_bonus_partials_graph(
                target_logits=x["logits"],
                correct_len=x["sim"].expand(bs),
                temperatures=x["temps"],
                seed=x["seed"],
                bs=bs,
                rows_per_request=T,
            ),
            x["logits"].shape[1],
        )
    else:
        target = row_argmax(x["logits"]).view(-1)
    static_accept_tail(
        candidates=x["cand"],
        target_predict=target,
        seq_lens=x["seq_lens"],
        verify_lens=x["verify_lens"],
        gate=x["gate"],
        simulated_correct_len=x["sim"],
        bonus_parts=parts,
        **o,
    )
    return target


def select(local_keys):
    """deferred_select: every rank's [rows, 2] keys, the largest value, ties to the lower rank."""
    tp = dist.get_world_size()
    g = torch.empty(
        (tp, *local_keys.shape), dtype=local_keys.dtype, device=local_keys.device
    )
    dist.all_gather_into_tensor(g.view(-1, 2), local_keys)
    best = g[..., 0].argmax(dim=0, keepdim=True)
    return g[..., 1].gather(0, best).squeeze(0).long()


def sglang_shard(x, bs, o, rank, part):
    """SGLang's shard path (dspark_verify._static_accept_tail with a vocab shard)."""
    from sglang.kernels.ops.sampling.softcap_row_argmax import softcap_row_argmax
    from sglang.kernels.ops.speculative.dspark.simulated_bonus import (
        simulated_bonus_shard_keys,
    )
    from sglang.kernels.ops.speculative.dspark.static_accept_tail import (
        static_accept_tail,
    )

    start = rank * part
    width = max(0, min(part, VOCAB - start))
    target = bonus = None
    if x["mode"] == "simt":
        bonus = select(
            simulated_bonus_shard_keys(
                target_logits=x["local"],
                col_offset=start,
                valid_width=width,
                vocab=VOCAB,
                correct_len=x["sim"].expand(bs),
                temperatures=x["temps"],
                seed=x["seed"],
                bs=bs,
                rows_per_request=T,
            )
        )
    else:
        target = select(softcap_row_argmax(x["local"], None, width, start)).view(-1)
    static_accept_tail(
        candidates=x["cand"],
        target_predict=target,
        seq_lens=x["seq_lens"],
        verify_lens=x["verify_lens"],
        gate=x["gate"],
        simulated_correct_len=x["sim"],
        bonus_in=bonus,
        **o,
    )
    return target


def ours(tail, x, bs, o, target_out=None):
    tail(
        x["local"],
        x["cand"],
        x["seq_lens"],
        x["verify_lens"],
        x["gate"],
        sim=x["sim"],
        temps=x["temps"] if x["sim"] is not None else None,
        seed=x["seed"] if x["sim"] is not None else None,
        correct_out=o["correct_len_out"],
        bonus_out=o["bonus_out"],
        cap_trim_out=o["cap_trim_lens_out"],
        commit_out=o["commit_lens_out"],
        new_seq_out=o["new_seq_lens_out"],
        out_tokens=o["out_tokens_out"],
        gated_out=o["gated_commit_lens_out"],
        target_out=target_out,
    )


def same(a, b, bs):
    return all(torch.equal(a[k][:bs], b[k][:bs]) for k in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.gfx950 import vtail

    part = -(-VOCAB // tp)
    part += -part % 8
    start = rank * part
    width = max(0, min(part, VOCAB - start))
    tail = vtail.VerifyTail(
        cpu,
        rank,
        tp,
        dev,
        part=part,
        width=width,
        vocab_start=start,
        vocab=VOCAB,
        deadline_s=30.0,
    )
    say = (lambda s: print(s, flush=True)) if rank == 0 else (lambda s: None)
    fails = 0
    bad = check_noise(tail.ops, dev)
    say(f"noise: {bad} mismatches over 8 x {VOCAB} offsets, 3 seeds, 3 temperatures")
    fails += bad != 0
    seed = 100
    for mode in ("greedy", "sim0", "simt"):
        for bs in (1, 2, 4, 8):
            for rep in range(3):
                seed += 1
                x = inputs(Case(bs, mode), rank, tp, part, dev, seed)
                x["mode"] = mode
                want, shard, got = outs(bs, dev), outs(bs, dev), outs(bs, dev)
                t_want = sglang_chain(x, bs, want)
                t_shard = sglang_shard(x, bs, shard, rank, part)
                t_got = torch.full((vtail.MAX_ROWS,), -1, dtype=torch.int64, device=dev)
                ours(tail, x, bs, got, t_got)
                torch.cuda.synchronize()
                ok_shard = same(want, shard, bs) and (
                    t_want is None or torch.equal(t_want, t_shard)
                )
                ok = same(want, got, bs) and (
                    mode != "greedy" or torch.equal(t_want, t_got[: bs * T])
                )
                agree = torch.tensor([int(ok and ok_shard)], device=dev)
                dist.all_reduce(agree, op=dist.ReduceOp.MIN)
                if not agree.item() or rep == 0:
                    say(
                        f"{mode} bs {bs} seed {seed}: sglang shard {'PASS' if ok_shard else 'FAIL'}, "
                        f"one launch {'PASS' if ok else 'FAIL'} (all ranks {'PASS' if agree.item() else 'FAIL'})"
                    )
                if not ok:
                    print(
                        f"rank {rank} {mode} bs {bs}: want {[(k, v[:bs].tolist()) for k, v in want.items()]} "
                        f"got {[(k, v[:bs].tolist()) for k, v in got.items()]}",
                        flush=True,
                    )
                fails += not agree.item()
    # graph replays: inputs copied into the captured buffers, a fresh case each replay
    for mode in ("greedy", "simt"):
        bs = 2
        x = inputs(Case(bs, mode), rank, tp, part, dev, 7)
        got = outs(bs, dev)
        ours(tail, x, bs, got)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ours(tail, x, bs, got)
        nbad = 0
        for r in range(a.replays):
            y = inputs(Case(bs, mode), rank, tp, part, dev, 1000 + r)
            for k in ("local", "cand", "seq_lens", "verify_lens", "temps", "seed"):
                x[k].copy_(y[k])
            if x["sim"] is not None:
                x["sim"].copy_(y["sim"])
            y["mode"] = mode
            want = outs(bs, dev)
            sglang_chain(y, bs, want)
            graph.replay()
            torch.cuda.synchronize()
            nbad += not same(want, got, bs)
        agree = torch.tensor([int(nbad == 0)], device=dev)
        dist.all_reduce(agree, op=dist.ReduceOp.MIN)
        say(f"graph replays {mode}: {'PASS' if agree.item() else 'FAIL'} ({a.replays})")
        fails += not agree.item()
    # time a unit in a graph of 10, the maximum over ranks
    for mode in ("greedy", "sim0", "simt"):
        bs = 1
        x = inputs(Case(bs, mode), rank, tp, part, dev, 3)
        x["mode"] = mode
        o = outs(bs, dev)
        for name, unit in (
            ("one launch", lambda: ours(tail, x, bs, o)),
            ("sglang shard path", lambda: sglang_shard(x, bs, o, rank, part)),
        ):
            for _ in range(3):
                unit()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    unit()
            for _ in range(5):
                graph.replay()
            dist.barrier()
            torch.cuda.synchronize()
            t0, t1 = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            t0.record()
            for _ in range(a.reps):
                graph.replay()
            t1.record()
            torch.cuda.synchronize()
            us = torch.tensor([t0.elapsed_time(t1) * 1e3 / (a.reps * 10)], device=dev)
            dist.all_reduce(us, op=dist.ReduceOp.MAX)
            say(
                f"time {mode} bs {bs} {name}: {us.item():.2f} us a unit (max over ranks)"
            )
    say("ALL PASS" if fails == 0 else f"{fails} FAIL")
    torch.cuda.synchronize()
    dist.barrier(group=cpu)
    os._exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
