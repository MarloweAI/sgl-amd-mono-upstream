"""Check Kimi-K3's sharded sampled Markov walk (gfx950/markov, one launch) against SGLang's arithmetic on one GPU.

``shard``: the loopback build at K3's TP8 shapes (vocab 163,840, a 20,480-row shard): each rank's launch alone
against SGLang's walk on that shard (``MarkovSampleStep`` with ``vocab_start``, no gather), every batch size 1..8,
sampled and temperature-0 rows: tokens, bf16 step logits and the noise counter bitwise, eager and under graph replay;
and whether SGLang's shard GEMV of the bias is bitwise its whole-vocab GEMV.

``ranks``: TP ranks as processes on the one GPU (real peer buffers, no loopback) on a smaller vocab (every rank's grid
co-resident), against SGLang's served walk on the whole vocab (``draft_markov_block`` on an unsharded
``VanillaMarkov``): tokens, the whole step logits gathered from the shards, and the counter bitwise.

    python3 check_markov.py shard [--trials 20]
    python3 check_markov.py ranks [--tp 8 --part 2048 --trials 20]
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

VOCAB, PART, DIM, GAMMA = 163840, 20480, 256, 5
KEY = 0x2D2A9C317E44B1F3 ^ 0x1234
TEMPS = [1.0, 0.0, 0.7, 1.3, 1.0, 1.0, 0.5, 1.0]


def stock_step(head_w2, w1, prev, base_step, temps, offset, step, start, logits_out):
    """SGLang's step on a shard; ``head_w2``: the shard's rows (its shard GEMV), or (the whole markov_w2, start,
    width): the served whole-vocab GEMV's columns of the shard."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    if isinstance(head_w2, tuple):
        w2, a, n = head_w2
        bias = F.linear(w1[prev], w2)[:, a : a + n]
    else:
        bias = F.linear(w1[prev], head_w2)
    return MarkovSampleStep.execute(
        base_logits=base_step, bias=bias, temperatures=temps, key=KEY, offset=offset, step=step,
        step_dtype=torch.bfloat16, vocab_start=start, logits_out=logits_out,
    )  # fmt: skip


def stock_shard_walk(w1, w2s, base, anchor, temps, offset, start):
    bs = base.shape[0]
    logits = torch.empty(bs, GAMMA, base.shape[-1], dtype=torch.bfloat16, device=base.device)
    prev, toks = anchor, []
    for step in range(GAMMA):
        prev = stock_step(w2s, w1, prev, base[:, step], temps, offset, step, start, logits[:, step])
        toks.append(prev)
    offset.add_(1)
    return torch.stack(toks, 1), logits


def check_shard(args) -> int:
    from sgl_amd_mono.kimi_k3.gfx950 import markov

    dev = torch.device("cuda")
    ops = markov.ops(("-DMARKOV_LOOPBACK",))
    g = torch.Generator(device=dev).manual_seed(7)
    w1 = (torch.randn(VOCAB, DIM, generator=g, device=dev) * 0.05).bfloat16()
    w2 = (torch.randn(VOCAB, DIM, generator=g, device=dev) * 0.05).bfloat16()
    fails, bias_diff, n_bias = [], 0, 0
    served_n = served_tok = served_lg = served_vals = 0
    for rank in (0, 3, 7):
        start = rank * PART
        w2s = w2[start : start + PART]
        x = markov.Loopback(8, dev, GAMMA, PART)
        for trial in range(args.trials):
            for bs in range(1, markov.MAX_SEQUENCES + 1):
                base = (torch.randn(bs, GAMMA, PART, generator=g, device=dev) * 3).bfloat16()
                anchor = torch.randint(0, VOCAB, (bs,), generator=g, device=dev)
                temps = torch.tensor(TEMPS[:bs], device=dev)
                if trial % 2:
                    temps = torch.ones(bs, device=dev)
                off0 = torch.randint(0, 1 << 30, (1,), generator=g, device=dev)
                # SGLang's whole-vocab bias GEMV against its shard GEMV
                lat = w1[anchor]
                full = F.linear(lat, w2)[:, start : start + PART]
                bias_diff += int((full != F.linear(lat, w2s)).sum())
                n_bias += full.numel()
                served_off = off0.clone()
                served, served_logits = stock_shard_walk(w1, (w2, start, PART), base, anchor, temps, served_off, start)
                want_off = off0.clone()
                want, want_logits = stock_shard_walk(w1, w2s, base, anchor, temps, want_off, start)
                got = torch.empty(bs * GAMMA, dtype=torch.int64, device=dev)
                got_logits = torch.zeros(bs, GAMMA, PART, dtype=torch.bfloat16, device=dev)
                off = off0.clone()
                ops.markov_block(base, w1, w2s, anchor, temps, off, KEY, start, got, GAMMA, x.tok, x.control,
                                 x.peers, 0, got_logits, PART, 10.0, None)  # fmt: skip
                torch.cuda.synchronize()
                slab = got_logits
                if trial == 0:  # whole rows: the walk keeps its shard in the slab, the gather copies it (loopback: to
                    # every rank's columns)
                    whole = torch.zeros(bs, GAMMA, VOCAB, dtype=torch.bfloat16, device=dev)
                    off = off0.clone()
                    ops.markov_block(base, w1, w2s, anchor, temps, off, KEY, start, got, GAMMA, x.tok, x.control,
                                     x.peers, 0, whole, PART, 10.0, None)  # fmt: skip
                    ops.gather_step_logits(x.peers, whole, PART)
                    torch.cuda.synchronize()
                    for r in range(8):
                        cols = whole[..., r * PART : (r + 1) * PART]
                        if not torch.equal(cols, got_logits):
                            where = (cols != got_logits).sum(-1).nonzero().tolist()
                            fails.append(f"rank {rank} bs {bs}: whole rows' columns of rank {r} differ from the "
                                         f"shard launch's at (seq, step) {where[:6]}")
                    if not torch.equal(got.view(bs, GAMMA), want):
                        fails.append(f"rank {rank} bs {bs}: tokens with whole rows")
                served_n += 1
                served_tok += int(torch.equal(got.view(bs, GAMMA), served))
                served_lg += int(torch.equal(slab, served_logits))
                served_vals += int((slab != served_logits).sum())
                tok_ok = torch.equal(got.view(bs, GAMMA), want)
                lg_ok = torch.equal(slab.view(torch.int16), want_logits.view(torch.int16))
                off_ok = torch.equal(off, want_off)
                if not (tok_ok and lg_ok and off_ok):
                    nd = int((slab != want_logits).sum())
                    fails.append(f"rank {rank} trial {trial} bs {bs}: tokens {tok_ok} logits {lg_ok} ({nd} differ) "
                                 f"offset {off_ok}")
        # graph replay: two replays, the counter advancing, each against SGLang
        bs = 4
        base = (torch.randn(bs, GAMMA, PART, generator=g, device=dev) * 3).bfloat16()
        anchor = torch.randint(0, VOCAB, (bs,), generator=g, device=dev)
        temps = torch.ones(bs, device=dev)
        off = torch.zeros(1, dtype=torch.int64, device=dev)
        got = torch.empty(bs * GAMMA, dtype=torch.int64, device=dev)
        logits = torch.zeros(bs, GAMMA, PART, dtype=torch.bfloat16, device=dev)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            ops.markov_block(base, w1, w2s, anchor, temps, off, KEY, start, got, GAMMA, x.tok, x.control, x.peers, 0,
                             logits, PART, 10.0, None)  # warm-up  # fmt: skip
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ops.markov_block(base, w1, w2s, anchor, temps, off, KEY, start, got, GAMMA, x.tok, x.control, x.peers, 0,
                             logits, PART, 10.0, None)  # fmt: skip
        for rep in range(3):
            ref_off = off.clone()
            want, want_logits = stock_shard_walk(w1, w2s, base, anchor, temps, ref_off, start)
            graph.replay()
            torch.cuda.synchronize()
            if not (torch.equal(got.view(bs, GAMMA), want) and torch.equal(off, ref_off)
                    and torch.equal(logits, want_logits)):  # fmt: skip
                fails.append(f"rank {rank} graph replay {rep}")
        x.close()
    print(f"SHARD bias: SGLang's shard GEMV differs from its whole-vocab GEMV at {bias_diff} of {n_bias} values")
    print(f"SHARD against the served whole-vocab GEMV: tokens bitwise in {served_tok} of {served_n} launches, step logits "
          f"bitwise in {served_lg} ({served_vals} values differ in all)")
    for f in fails:
        print("FAIL", f)
    print(f"SHARD {'PASS' if not fails else 'FAIL'} ({3 * args.trials * 8} launches + 9 graph replays)")
    return 1 if fails else 0


def _rank_main(rank, args, port, q):
    import torch.distributed as dist

    from sgl_amd_mono.kimi_k3.gfx950 import markov

    tp, part = args.tp, args.part
    vocab = tp * part
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=tp)
    group = dist.new_group(backend="gloo")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    from sglang.srt.speculative.dspark_components.dspark_draft import DraftNoise, draft_markov_block
    from sglang.srt.models.dspark import VanillaMarkov

    class _NoSync:
        def sync(self, site, value):
            return value

    g = torch.Generator(device=dev).manual_seed(11)  # alike on every rank
    head = VanillaMarkov(vocab_size=vocab, markov_rank=DIM).to(dev, torch.bfloat16)
    with torch.no_grad():
        head.markov_w1.weight.copy_(torch.randn(vocab, DIM, generator=g, device=dev) * 0.05)
        head.markov_w2.weight.copy_(torch.randn(vocab, DIM, generator=g, device=dev) * 0.05)
    w1, w2 = head.markov_w1.weight.detach(), head.markov_w2.weight.detach()
    start = rank * part
    w2s = w2[start : start + part].contiguous()
    ops = markov.ops()
    x = markov.Exchange(group, rank, tp, dev, GAMMA, part)
    fails = []
    for trial in range(args.trials):
        for bs in range(1, markov.MAX_SEQUENCES + 1):
            base = (torch.randn(bs, GAMMA, vocab, generator=g, device=dev) * 3).bfloat16()
            anchor = torch.randint(0, vocab, (bs,), generator=g, device=dev)
            temps = torch.tensor(TEMPS[:bs], device=dev) if trial % 2 == 0 else torch.ones(bs, device=dev)
            off0 = torch.randint(0, 1 << 30, (1,), generator=g, device=dev)
            noise = DraftNoise(key=KEY, offset=off0.clone())
            want_logits = torch.empty(bs, GAMMA, vocab, dtype=torch.bfloat16, device=dev)
            with torch.no_grad():
                want, _ = draft_markov_block(
                    markov_head=head, base_logits=base, anchor_tokens=anchor, draft_hidden=None,
                    step_temperatures=temps, noise=noise, tp_sync=_NoSync(), logits_out=want_logits,
                )  # fmt: skip
            got = torch.empty(bs * GAMMA, dtype=torch.int64, device=dev)
            whole = torch.zeros(bs, GAMMA, vocab, dtype=torch.bfloat16, device=dev)
            off = off0.clone()
            torch.cuda.synchronize()
            dist.barrier(group)  # every rank's previous gather read before this launch writes
            ops.markov_block(base[..., start : start + part].contiguous(), w1, w2s, anchor, temps, off, KEY, start,
                             got, GAMMA, x.tok, x.control, x.peers, rank, whole, part, 20.0, None)  # fmt: skip
            torch.cuda.synchronize()
            dist.barrier(group)  # every rank's shard written
            ops.gather_step_logits(x.peers, whole, part)
            torch.cuda.synchronize()
            ok = (torch.equal(got.view(bs, GAMMA), want), torch.equal(whole, want_logits),
                  torch.equal(off, noise.offset))  # fmt: skip
            if not all(ok):
                fails.append(f"rank {rank} trial {trial} bs {bs}: tokens/logits/offset {ok}, "
                             f"{int((whole != want_logits).sum())} logits differ")  # fmt: skip
    dist.barrier(group)
    x.close()
    q.put((rank, fails))
    dist.destroy_process_group()


def check_ranks(args) -> int:
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + os.getpid() % 1000
    procs = [ctx.Process(target=_rank_main, args=(r, args, port, q)) for r in range(args.tp)]
    for p in procs:
        p.start()
    res = []
    for _ in procs:
        try:
            res.append(q.get(timeout=900))
        except Exception:  # a rank died: the others wait in a collective
            for p in procs:
                p.kill()
            print("RANKS FAIL: a rank did not report", [p.exitcode for p in procs])
            return 1
    for p in procs:
        p.join()
    fails = [f for _, fs in sorted(res) for f in fs]
    for f in fails[:40]:
        print("FAIL", f)
    n = args.trials * 8
    print(f"RANKS TP{args.tp} part {args.part} {'PASS' if not fails else 'FAIL'} ({n} blocks a rank, "
          f"vocab {args.tp * args.part})")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("shard", "ranks"))
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--part", type=int, default=2048)
    args = ap.parse_args()
    sys.exit(check_shard(args) if args.what == "shard" else check_ranks(args))


if __name__ == "__main__":
    main()
