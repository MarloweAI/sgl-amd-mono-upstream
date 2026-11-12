"""The verify's vocab projection (gfx950/headgemm) against SGLang's (torch's bf16 linear, hipBLASLt) on the real
head weight and recorded head inputs, one GPU, all four TP4 shards in turn: each row's logits error against an
fp64 reference (subset), the full-vocab argmax agreement with SGLang's over every recorded row (ties counted), and
both projections' time a shard in a graph.

    python3 check_head_gemm.py --weights MODEL_DIR --rows DUMP_DIR [--ref-rows 3000] [--depth 4]
"""

import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F

TP, M = 4, 6


def head_weight(model):
    from safetensors import safe_open

    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))[
        "weight_map"
    ]
    with safe_open(
        os.path.join(model, index["head.weight"]), framework="pt", device="cuda"
    ) as f:
        return f.get_tensor("head.weight")


def graph_us(fn, reps=200):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / (reps * 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--ref-rows", type=int, default=3000)
    ap.add_argument("--max-rows", type=int, default=200000)
    ap.add_argument("--depth", type=int, default=13)
    a = ap.parse_args()
    torch.cuda.set_device(0)
    from sgl_amd_mono.deepseek_v41.gfx950.headgemm import ops

    ops_ = ops()
    w = head_weight(a.weights)
    vocab, hidden = w.shape
    part = vocab // TP
    shards = [w[r * part : (r + 1) * part].contiguous() for r in range(TP)]
    del w
    print(f"head {vocab} x {hidden} {shards[0].dtype}, shard {part}", flush=True)
    files = sorted(glob.glob(os.path.join(a.rows, "head_in_*.pt")))
    if files:
        x_all = torch.cat([torch.load(f) for f in files])[: a.max_rows]
    else:  # no recording: normal rows, for the time and the fp64 errors only
        x_all = torch.randn(
            a.ref_rows, hidden, generator=torch.Generator().manual_seed(1)
        ).bfloat16()
    rows = x_all.shape[0] // M * M
    x_all = x_all[:rows].cuda()
    print(
        f"{rows} recorded rows from {len(files)} files, |x| mean {x_all.float().abs().mean():.3f}",
        flush=True,
    )

    def sglang(x, r):
        return F.linear(x, shards[r])

    def ours(x, r, out):
        ops_.head_gemm(x, shards[r], out, a.depth)
        return out

    # time a shard: 6 rows, the served verify
    x6 = x_all[:M].contiguous()
    out = torch.empty(M, part, dtype=torch.bfloat16, device="cuda")
    for depth in (0, 10, 11, 12, 13, 14):
        t = graph_us(lambda: ops_.head_gemm(x6, shards[0], out, depth))
        print(
            f"time ours depth x waves {depth}: {t:.2f} us ({shards[0].numel() * 2 / t / 1e6:.2f} TB/s)",
            flush=True,
        )
    flat = shards[0].view(-1)
    red = torch.empty((), dtype=torch.bfloat16, device="cuda")
    t = graph_us(lambda: torch.amax(flat, dim=0, out=red))
    print(
        f"time read ceiling (torch.amax of the shard): {t:.2f} us ({shards[0].numel() * 2 / t / 1e6:.2f} TB/s)",
        flush=True,
    )
    t = graph_us(lambda: sglang(x6, 0))
    print(
        f"time sglang (hipBLASLt): {t:.2f} us ({shards[0].numel() * 2 / t / 1e6:.2f} TB/s)",
        flush=True,
    )

    # errors against fp64 on a subset (6-row groups as served)
    ref_rows = min(a.ref_rows, rows) // M * M
    err = {"sglang": [0.0, 0.0, 0], "ours": [0.0, 0.0, 0], "ours_fp32": [0.0, 0.0, 0]}
    for r in range(TP):
        wd = shards[r].double()
        for i in range(0, ref_rows, 600):
            xs = x_all[i : i + 600]
            ref = xs.double() @ wd.T
            got_s = torch.cat(
                [sglang(xs[j : j + M], r) for j in range(0, xs.shape[0], M)]
            ).double()
            got_o = torch.cat(
                [
                    ours(xs[j : j + M].contiguous(), r, out).clone()
                    for j in range(0, xs.shape[0], M)
                ]
            ).double()
            for k, g in (("sglang", got_s), ("ours", got_o)):
                d = (g - ref).abs()
                err[k][0] = max(err[k][0], d.max().item())
                err[k][1] += d.sum().item()
                err[k][2] += d.numel()
    for k in ("sglang", "ours"):
        print(
            f"error vs fp64 ({ref_rows} rows x {vocab}): {k} max {err[k][0]:.4g} mean {err[k][1] / err[k][2]:.4g}",
            flush=True,
        )

    # argmax agreement over every row, full vocab (ranks' shards side by side), lowest index on ties
    agree = ties_s = ties_o = 0
    both_tie_agree = 0
    dis = []
    for i in range(0, rows, 1200):
        xs = x_all[i : i + 1200]
        ls = torch.cat(
            [
                torch.cat([sglang(xs[j : j + M], r) for j in range(0, xs.shape[0], M)])
                for r in range(TP)
            ],
            dim=1,
        ).float()
        lo = torch.cat(
            [
                torch.cat(
                    [
                        ours(xs[j : j + M].contiguous(), r, out).clone()
                        for j in range(0, xs.shape[0], M)
                    ]
                )
                for r in range(TP)
            ],
            dim=1,
        ).float()
        ms, mo = ls.max(dim=1, keepdim=True).values, lo.max(dim=1, keepdim=True).values
        a_s, a_o = (ls == ms).float().argmax(dim=1), (lo == mo).float().argmax(dim=1)
        ts, to = (ls == ms).sum(dim=1) > 1, (lo == mo).sum(dim=1) > 1
        same = a_s == a_o
        agree += int(same.sum())
        ties_s += int(ts.sum())
        ties_o += int(to.sum())
        both_tie_agree += int((same & ts).sum())
        bad = (~same).nonzero().flatten()
        for b in bad[:4].tolist():
            top = torch.topk(ls[b], 3)
            dis.append(
                (
                    i + b,
                    a_s[b].item(),
                    a_o[b].item(),
                    [round(v, 3) for v in top.values.tolist()],
                )
            )
    print(
        f"argmax agreement: {agree} / {rows} rows ({100 * agree / rows:.4f}%); rows with a tied maximum: sglang "
        f"{ties_s}, ours {ties_o}; tied rows agreeing {both_tie_agree}",
        flush=True,
    )
    for d in dis[:12]:
        print("  disagree row %d: sglang %d ours %d, sglang top-3 %s" % d, flush=True)
    print("HEAD_GEMM_DONE", flush=True)


if __name__ == "__main__":
    main()
