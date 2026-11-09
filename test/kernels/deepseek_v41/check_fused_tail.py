"""Check the greedy verify's head projection and accept in one launch (gfx950/vtail/fused) against the launches it
replaces (torchrun, TP4): the logits shard bitwise ``gfx950/headgemm``'s variant 13 on the head's normed rows (from
``gfx950/norm``), and the accept's outputs bitwise ``gfx950/vtail``'s (itself bitwise SGLang's chain) on that shard:
greedy, a simulated length with argmax bonus and with temperature-sampled bonus, a request of 6 rows; graph
replays with fresh inputs. Then the time in a graph of 10 (the maximum over ranks) of the head's tail as served
(combine and norm, then the one launch), of the launches it replaces (combine and norm, SGLang's head GEMM, the
verify tail), of the one launch alone and of the head GEMMs alone.

With ``--rows`` (recorded head inputs, ``head_in_*.pt``) and ``--weights`` (the checkpoint's head): each 6 recorded
rows through the launch, its full-vocab argmax (the ranks' exchange) against SGLang's (torch's bf16 linear, gathered)
over every row, ties counted.

    torchrun --nproc-per-node 4 check_fused_tail.py [--weights MODEL_DIR] [--rows DUMP_DIR] [--replays 10]
"""

import argparse
import glob
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

VOCAB, T, K, HC, EPS = 129280, 6, 5120, 4, 1e-6


def head_weights(model, rank, part):
    from safetensors import safe_open

    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))[
        "weight_map"
    ]
    out = {}
    for name in ("head.weight", "norm.weight"):
        with safe_open(
            os.path.join(model, index[name]), framework="pt", device="cpu"
        ) as f:
            t = f.get_slice(name)
            out[name] = (
                (t[rank * part : (rank + 1) * part] if name == "head.weight" else t[:])
                .cuda()
                .contiguous()
            )
    return out["head.weight"], out["norm.weight"].bfloat16()


def norm_inputs(rows, g, dev, wide=False):
    bf = torch.bfloat16
    x = torch.randn(rows, K, generator=g, device=dev).to(bf)
    res = torch.randn(rows, HC, K, generator=g, device=dev).to(bf)
    if wide:
        x[:, ::7] *= 300.0
        res[:, :, ::5] *= 1e-3
    post = torch.rand(rows, HC, generator=g, device=dev) * 2
    comb = torch.rand(rows, HC, HC, generator=g, device=dev)
    pre = torch.rand(rows, HC, generator=g, device=dev)
    return x, res.contiguous(), post, comb, pre


def accept_inputs(bs, mode, g, dev, target=None):
    """Candidates matching a random prefix of ``target`` (key rows' argmax) where given, lengths, temperatures."""
    cand = torch.randint(0, VOCAB, (bs, T), generator=g, device=dev)
    if target is not None:
        tg = target[: bs * T].view(bs, T)
        for r in range(bs):
            m = int(torch.randint(0, T, (1,), generator=g, device=dev))
            cand[r, 1:] = tg[r, : T - 1]
            if m < T - 1:
                cand[r, 1 + m] = (tg[r, m] + 1) % VOCAB
    seq_lens = torch.randint(100, 9000, (bs,), generator=g, device=dev)
    verify_lens = torch.full((bs,), T, dtype=torch.int64, device=dev)
    verify_lens[-1] = 3
    sim = (
        torch.randint(0, T, (1,), generator=g, device=dev) if mode != "greedy" else None
    )
    temps = torch.zeros(bs, device=dev)
    if mode == "simt":
        temps = torch.tensor([1.0, 0.6][:bs], device=dev)
    seed = torch.randint(0, 2**31 - 1, (1,), generator=g, device=dev).int()
    gate = torch.tensor([1], dtype=torch.int32, device=dev)
    return dict(
        cand=cand,
        seq_lens=seq_lens,
        verify_lens=verify_lens,
        sim=sim,
        temps=temps,
        seed=seed,
        gate=gate,
    )


def outs(dev):
    return dict(
        correct_out=torch.zeros(8, dtype=torch.int64, device=dev),
        bonus_out=torch.zeros(8, dtype=torch.int64, device=dev),
        cap_trim_out=torch.full((8,), 7, dtype=torch.int32, device=dev),
        commit_out=torch.zeros(8, dtype=torch.int32, device=dev),
        new_seq_out=torch.zeros(8, dtype=torch.int64, device=dev),
        out_tokens=torch.full((8, T), -5, dtype=torch.int64, device=dev),
        gated_out=torch.zeros(8, dtype=torch.int32, device=dev),
    )


def acc_args(a):
    sim = a["sim"]
    return dict(
        sim=sim,
        temps=a["temps"] if sim is not None else None,
        seed=a["seed"] if sim is not None else None,
    )


def normed_rows(n, nw):
    from sgl_amd_mono.deepseek_v41.gfx950.norm import ops as norm_ops

    x, res, post, comb, pre = n
    streams, normed = torch.empty_like(res), torch.empty_like(x)
    norm_ops().combine_norm(x, res, post, comb, pre, nw, EPS, streams, normed)
    return normed


def run_fused(ft, x, w, a, o, target_out=None):
    return ft(
        x,
        w,
        a["cand"],
        a["seq_lens"],
        a["verify_lens"],
        a["gate"],
        **acc_args(a),
        **o,
        target_out=target_out,
    )


def run_reference(x, w, a, o, tail, target_out=None):
    """headgemm variant 13, then vtail: the logits."""
    from sgl_amd_mono.deepseek_v41.gfx950.headgemm import ops as hg_ops

    logits = torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)
    for r in range(
        0, x.shape[0], T
    ):  # headgemm stages x in at most 64 KB of LDS; rows are independent
        hg_ops().head_gemm(x[r : r + T], w, logits[r : r + T], 13)
    tail(
        logits,
        a["cand"],
        a["seq_lens"],
        a["verify_lens"],
        a["gate"],
        **acc_args(a),
        correct_out=o["correct_out"],
        bonus_out=o["bonus_out"],
        cap_trim_out=o["cap_trim_out"],
        commit_out=o["commit_out"],
        new_seq_out=o["new_seq_out"],
        out_tokens=o["out_tokens"],
        gated_out=o["gated_out"],
        target_out=target_out,
    )
    return logits


def bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def same_outs(a, b, bs):
    return all(torch.equal(a[k][:bs], b[k][:bs]) for k in a)


def graph_us(fn, reps, rank_sync):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            fn()
    for _ in range(5):
        graph.replay()
    rank_sync()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        graph.replay()
    t1.record()
    torch.cuda.synchronize()
    us = torch.tensor([t0.elapsed_time(t1) * 1e3 / (reps * 10)], device="cuda")
    dist.all_reduce(us, op=dist.ReduceOp.MAX)
    return us.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--max-rows", type=int, default=200000)
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.gfx950 import vtail
    from sgl_amd_mono.deepseek_v41.gfx950.vtail import fused

    part = VOCAB // tp
    start, width = rank * part, part
    say = (lambda s: print(s, flush=True)) if rank == 0 else (lambda s: None)
    kw = dict(part=part, width=width, vocab_start=start, vocab=VOCAB, deadline_s=30.0)
    ft = fused.FusedTail(cpu, rank, tp, dev, **kw)
    tail = vtail.VerifyTail(cpu, rank, tp, dev, **kw)
    if args.weights:
        w, nw = head_weights(args.weights, rank, part)
        say(f"head shard {tuple(w.shape)} {w.dtype} from {args.weights}")
    else:
        g = torch.Generator(device=dev).manual_seed(11 + rank)
        w = (torch.randn(part, K, generator=g, device=dev) * 0.02).bfloat16()
        nw = (
            1
            + 0.1
            * torch.randn(
                K, generator=torch.Generator(device=dev).manual_seed(3), device=dev
            )
        ).bfloat16()
    fails = 0

    def all_ok(ok):
        t = torch.tensor([int(ok)], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    seed = 100
    for mode in ("greedy", "sim0", "simt"):
        for bs in (1,):
            for rep in range(4):
                seed += 1
                g = torch.Generator(device=dev).manual_seed(
                    seed
                )  # the same norm inputs on every rank
                x = normed_rows(norm_inputs(bs * T, g, dev, wide=rep % 2 == 1), nw)
                # first a launch for the key rows' argmax, then candidates matching a prefix of it
                a = accept_inputs(bs, "greedy", g, dev)
                tgt = torch.full((fused.MAX_TP * 2,), -1, dtype=torch.int64, device=dev)
                run_fused(ft, x, w, a, outs(dev), target_out=tgt)
                a = accept_inputs(bs, mode, g, dev, target=tgt)
                want, got = outs(dev), outs(dev)
                t_want = torch.full((12,), -1, dtype=torch.int64, device=dev)
                t_got = torch.full((12,), -1, dtype=torch.int64, device=dev)
                rl = run_reference(x, w, a, want, tail, target_out=t_want)
                fl = run_fused(ft, x, w, a, got, target_out=t_got)
                torch.cuda.synchronize()
                checks = {
                    "logits": bits_equal(rl, fl),
                    "accept": same_outs(want, got, bs) and torch.equal(t_want, t_got),
                }
                ok = all(checks.values())
                agree = all_ok(ok)
                if not agree or rep == 0:
                    say(
                        f"{mode} bs {bs} seed {seed}{' wide' if rep % 2 else ''}: "
                        + ", ".join(
                            f"{k} {'same' if v else 'DIFFER'}"
                            for k, v in checks.items()
                        )
                        + f" (all ranks {'PASS' if agree else 'FAIL'}; accept lengths {got['commit_out'][:bs].tolist()})"
                    )
                if not ok:
                    print(
                        f"rank {rank} {mode} bs {bs}: {checks}; want {[(k, v[:bs].tolist()) for k, v in want.items()]}"
                        f" t {t_want.tolist()} got {[(k, v[:bs].tolist()) for k, v in got.items()]} t {t_got.tolist()}",
                        flush=True,
                    )
                fails += not agree

    # graph replays: fresh inputs copied into the captured buffers each replay
    for mode in ("greedy", "simt"):
        bs = 1
        g = torch.Generator(device=dev).manual_seed(7)
        x = normed_rows(norm_inputs(bs * T, g, dev), nw)
        a = accept_inputs(bs, mode, g, dev)
        got = outs(dev)
        run_fused(ft, x, w, a, got)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gl = run_fused(ft, x, w, a, got)
        nbad = 0
        for r in range(args.replays):
            g = torch.Generator(device=dev).manual_seed(1000 + r)
            x.copy_(normed_rows(norm_inputs(bs * T, g, dev), nw))
            a2 = accept_inputs(bs, mode, g, dev)
            for k in ("cand", "seq_lens", "verify_lens", "temps", "seed"):
                a[k].copy_(a2[k])
            if a["sim"] is not None:
                a["sim"].copy_(a2["sim"])
            want = outs(dev)
            rl = run_reference(x, w, a, want, tail)
            graph.replay()
            torch.cuda.synchronize()
            nbad += not (same_outs(want, got, bs) and bits_equal(rl, gl))
        agree = all_ok(nbad == 0)
        say(f"graph replays {mode}: {'PASS' if agree else 'FAIL'} ({args.replays})")
        fails += not agree

    # time: the head's tail as served, the launches it replaces, the one launch alone, the head GEMMs alone
    from sgl_amd_mono.deepseek_v41.gfx950.headgemm import ops as hg_ops
    from sgl_amd_mono.deepseek_v41.gfx950.norm import ops as norm_ops

    for mode in ("greedy", "sim0"):
        bs = 1
        g = torch.Generator(device=dev).manual_seed(3)
        n = norm_inputs(bs * T, g, dev)
        a = accept_inputs(bs, mode, g, dev)
        o = outs(dev)
        x, res, post, comb, pre = n
        streams, normed = torch.empty_like(res), torch.empty_like(x)
        out = torch.empty(T, part, dtype=torch.bfloat16, device=dev)

        def norm():
            norm_ops().combine_norm(x, res, post, comb, pre, nw, EPS, streams, normed)

        def ours():
            norm()
            run_fused(ft, normed, w, a, o)

        def separate():
            norm()
            logits = F.linear(normed, w)
            tail(
                logits,
                a["cand"],
                a["seq_lens"],
                a["verify_lens"],
                a["gate"],
                **acc_args(a),
                correct_out=o["correct_out"],
                bonus_out=o["bonus_out"],
                cap_trim_out=o["cap_trim_out"],
                commit_out=o["commit_out"],
                new_seq_out=o["new_seq_out"],
                out_tokens=o["out_tokens"],
                gated_out=o["gated_out"],
            )

        sync = lambda: dist.barrier()  # noqa: E731
        norm()
        t = {
            name: graph_us(fn, args.reps, sync)
            for name, fn in (
                ("norm + one launch", ours),
                ("norm + hipBLASLt + vtail", separate),
                ("one launch", lambda: run_fused(ft, normed, w, a, o)),
                ("norm", norm),
                ("hipBLASLt", lambda: F.linear(normed, w)),
                ("headgemm v13", lambda: hg_ops().head_gemm(normed, w, out, 13)),
            )
        }
        say(
            f"time {mode} bs {bs}: "
            + ", ".join(f"{k} {v:.2f} us" for k, v in t.items())
            + " (max over ranks)"
        )

    if args.rows:
        fails += real_rows(args, ft, w, dev, rank, tp, part, say, all_ok)
    say("ALL PASS" if fails == 0 else f"{fails} FAIL")
    torch.cuda.synchronize()
    dist.barrier(group=cpu)
    os._exit(0 if fails == 0 else 1)


def real_rows(args, ft, w, dev, rank, tp, part, say, all_ok):
    """Recorded head inputs through the launch, its argmax (the ranks' exchange) against SGLang's (torch's bf16
    linear, gathered) on every row."""
    files = sorted(glob.glob(os.path.join(args.rows, "head_in_*.pt")))
    xs = torch.cat([torch.load(f) for f in files])[: args.max_rows]
    rows = xs.shape[0] // T * T
    xs = xs[:rows].to(dev)
    say(f"{rows} recorded rows from {len(files)} files")
    M = T
    a = accept_inputs(1, "greedy", torch.Generator(device=dev).manual_seed(1), dev)
    o = outs(dev)
    tgt = torch.zeros(M, dtype=torch.int64, device=dev)
    agree = ties_s = ties_o = tie_agree = 0
    dis = []
    for i in range(0, rows, M):
        x = xs[i : i + M].contiguous()
        ft(
            x,
            w,
            a["cand"],
            a["seq_lens"],
            a["verify_lens"],
            a["gate"],
            sim=None,
            temps=None,
            seed=None,
            **o,
            target_out=tgt,
        )
        ls = F.linear(x, w).float()
        full = torch.empty(tp, M, part, device=dev)
        dist.all_gather_into_tensor(full, ls)
        full = full.permute(1, 0, 2).reshape(M, tp * part)
        mx = full.max(dim=1, keepdim=True).values
        hit = full == mx
        a_s = hit.float().argmax(dim=1)
        ts = hit.sum(dim=1) > 1
        same = a_s == tgt
        agree += int(same.sum())
        ties_s += int(ts.sum())
        tie_agree += int((same & ts).sum())
        for b in (~same).nonzero().flatten()[:2].tolist():
            top = torch.topk(full[b], 3)
            dis.append(
                (
                    i + b,
                    a_s[b].item(),
                    tgt[b].item(),
                    [round(v, 3) for v in top.values.tolist()],
                )
            )
    say(
        f"argmax agreement (one launch vs hipBLASLt): {agree} / {rows} rows "
        f"({100 * agree / rows:.4f}%); rows with a tied maximum (hipBLASLt) {ties_s}, of them agreeing {tie_agree}"
    )
    for d in dis[:12]:
        say("  disagree row %d: sglang %d ours %d, sglang top-3 %s" % d)
    return 0 if all_ok(True) else 1


if __name__ == "__main__":
    main()
