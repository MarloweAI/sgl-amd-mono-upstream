# SPDX-License-Identifier: Apache-2.0
"""Per-CTA phase stamps of one cold wide KDA launch (``KdaWideBuild.timeline``).

    python test/kernels/kimi_k3/timeline_kda_wide.py --copy ~/pe-k3/copy/<key> --reqs 2,4,8

A 1 GB write before each launch evicts L2 and MALL, as a served step reads a
layer's weights cold. Prints, per point, the min / median / max over the CTAs
that reach it, in µs after the earliest start, median over ``--runs``.
"""

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import kda_wide_case as kc  # noqa: E402

POINTS = ("start", "mfma", "emit", "issued", "polled", "conv", "gates", "recur", "norm", "pub")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", default="")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--reqs", default="2,4,8")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--depth", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    w = kc.load_layer(args.copy, args.layer, device) if args.copy else kc.random_layer(
        torch.Generator(device=device).manual_seed(0), device
    )
    flush = torch.empty(1 << 28, dtype=torch.float32, device=device)
    for R in map(int, args.reqs.split(",")):
        S = R * kc.L
        gen = torch.Generator(device=device).manual_seed(R)
        x = torch.randn(S, kc.H, generator=gen, device=device).to(kc.BF)
        pools = kc.Pools(R, gen, device)
        wide = kc.Wide(w, S, device, depth=args.depth, timeline=True)
        epoch = torch.ones(1, dtype=torch.int32, device=device)
        core = torch.empty(S, kc.PROJ, dtype=kc.BF, device=device)
        rows = {p: [] for p in POINTS}
        for _ in range(args.runs + 1):
            flush.fill_(1.0)
            epoch += 1
            wide.tl.zero_()
            wide(x, pools, epoch, core)
            torch.cuda.synchronize()
            t = wide.tl.cpu()
            t0 = int(t[:, 0][t[:, 0] > 0].min())
            v1 = (t[:, 1] - t0).double() / 100.0
            xcd = [round(float(v1[x::8].median()), 1) for x in range(8)]
            rows.setdefault("xcd", []).append(xcd)
            for i, p in enumerate(POINTS):
                v = t[:, i]
                v = (v[v > 0] - t0).double() / 100.0  # 100 MHz -> µs
                rows[p].append((float(v.min()), float(v.median()), float(v.max())) if len(v) else None)
        out = {"R": R, "rows": S}
        for p in POINTS:
            got = [r for r in rows[p][1:] if r is not None]
            if got:
                out[p] = [round(statistics.median(r[k] for r in got), 2) for k in range(3)]
        out["mfma_by_xcd"] = rows["xcd"][-1]
        print("KDA_WIDE_TL", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
