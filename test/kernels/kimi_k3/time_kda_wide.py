# SPDX-License-Identifier: Apache-2.0
"""Graph time of the wide KDA attention launch against the sum of SGLang's own
kernels for the same region (in_proj GEMM, f_b GEMM, ``kda_verify_gfx950`` on
the ReplaySSM ring, the gated RMSNorm), one GPU, one process.

    python test/kernels/kimi_k3/time_kda_wide.py --copy ~/pe-k3/copy/<key> --reqs 2,4,8

Each arm captures one CUDA graph a side that runs ``--sets`` layers, each with
its own copy of the weights and its own pools (so no weight is read warm from
MALL), warms it ``--warmup`` replays, then times ``--reps`` replays with
events: µs a layer = replay time / sets. The launch's graph also bumps the
step epoch once (its mailbox tags). Arms alternate stock, wide, wide, stock ..;
the spread of a side's arms is its A/A noise. The paper floor is the region's
bytes (``kda_wide_case.floor_bytes``) at ``--bw`` TB/s.
"""

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import kda_wide_case as kc  # noqa: E402


def graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def time_graph(g, warmup, reps):
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def width(R, base_w, args, device):
    S = R * kc.L
    gen = torch.Generator(device=device).manual_seed(2000 + R)
    sets = []
    for i in range(args.sets):
        w = base_w if i == 0 else kc.clone_layer(base_w)
        x = (torch.randn(S, kc.H, generator=gen, device=device)).to(kc.BF)
        sets.append((w, x, kc.Pools(R, gen, device), kc.Pools(R, gen, device)))
    stocks = [kc.Stock(w) for w, *_ in sets]
    wides = [kc.Wide(w, S, device, depth=args.depth, layer=1 + i) for i, (w, *_) in enumerate(sets)]
    epoch = torch.ones(1, dtype=torch.int32, device=device)
    cores = [torch.empty(S, kc.PROJ, dtype=kc.BF, device=device) for _ in sets]

    def run_stock():
        for st, (w, x, ps, _) in zip(stocks, sets):
            st(x, ps)

    def run_wide():
        epoch.add_(1)
        for wd, (w, x, _, pw), core in zip(wides, sets, cores):
            wd(x, pw, epoch, core)

    def run_epoch():
        epoch.add_(1)

    def run_gemm():
        for st, (w, x, ps, _) in zip(stocks, sets):
            st.proj(x)

    gs, gw, ge, gg = graph(run_stock), graph(run_wide), graph(run_epoch), graph(run_gemm)
    arms = {"stock": [], "wide": []}
    for k in range(args.arms):
        order = ("stock", "wide") if k % 2 == 0 else ("wide", "stock")
        for side in order:
            g = gs if side == "stock" else gw
            arms[side].append(time_graph(g, args.warmup, args.reps) / args.sets)
    t_epoch = time_graph(ge, args.warmup, args.reps)
    t_gemm = time_graph(gg, args.warmup, args.reps) / args.sets
    floor = kc.floor_bytes(R) / (args.bw * 1e6)
    out = {"R": R, "rows": S, "sets": args.sets, "warmup": args.warmup, "reps": args.reps,
           "floor_us": round(floor, 2), "epoch_graph_us": round(t_epoch, 2),
           "stock_inproj_us": round(t_gemm, 2)}
    for side, v in arms.items():
        out[side] = {"us": round(statistics.median(v), 2), "arms": [round(a, 2) for a in v],
                     "aa_spread": round(max(v) - min(v), 2)}
    out["speedup"] = round(out["stock"]["us"] / out["wide"]["us"], 3)
    del gs, gw, ge, gg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", default="")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--reqs", default="2,4,8")
    ap.add_argument("--sets", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--arms", type=int, default=4)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--bw", type=float, default=7.4, help="TB/s of the paper floor")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if args.copy:
        w = kc.load_layer(args.copy, args.layer, device)
    else:
        w = kc.random_layer(torch.Generator(device=device).manual_seed(0), device)
    res = []
    for R in map(int, args.reqs.split(",")):
        r = width(R, w, args, device)
        res.append(r)
        print("KDA_WIDE_TIME", json.dumps(r), flush=True)
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
