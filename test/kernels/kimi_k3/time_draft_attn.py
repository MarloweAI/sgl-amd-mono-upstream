# SPDX-License-Identifier: Apache-2.0
"""Kernel time of the Kimi-K3 drafter attention, new (``draft.DraftAttention``)
against stock (SGLang's ``verify_shared_kv_fwd``), one GPU.

    python3 test/kernels/kimi_k3/time_draft_attn.py --lens 250000,10000,10000,10000 --lens 10000

Each implementation's launches are captured in one graph and replayed; each
timed replay follows a ``--flush`` MB write (default 512: L2 and MALL cold, as
served, where a cycle's target verify runs between two drafter passes). The
implementations alternate A B B A within each of ``--arms`` arms. Prints the
median µs a call and the floor: the context's draft KV bytes (K and V, 2 heads
x 64 x 1 B a token) at 7.4 TB/s.
"""

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import check_draft_attn as cda  # noqa: E402

FLOOR_GBPS = 7400.0


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def time_graph(g, reps, flush):
    buf = torch.ones((max(flush, 1) << 20) // 4, dtype=torch.float32, device="cuda")
    ev = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(reps)
    ]
    for _ in range(3):
        g.replay()
    for a, b in ev:
        if flush:
            buf.sum()  # a read: evicts L2 and MALL and leaves no dirty lines to write back
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lens",
        action="append",
        required=True,
        help="comma-separated contexts (repeatable)",
    )
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--arms", type=int, default=3)
    ap.add_argument(
        "--flush", type=int, default=512, help="MB read before each replay, 0 = warm"
    )
    ap.add_argument("--scattered", action="store_true")
    ap.add_argument("--only", choices=("stock", "new"), help="time one implementation")
    ap.add_argument("--out", help="JSON results")
    a = ap.parse_args()
    results = []
    for spec in a.lens:
        lens = [int(t) for t in spec.split(",")]
        x = cda.make_inputs(lens, a.rows, scattered=a.scattered)
        shape = (len(lens) * a.rows, cda.draft.H_Q, cda.draft.HEAD_DIM)
        o = {
            k: torch.empty(shape, dtype=torch.bfloat16, device="cuda")
            for k in ("stock", "new")
        }
        run_new = cda.new_runner(x)
        fns = {
            "stock": lambda: cda.run_stock(x, o["stock"]),
            "new": lambda: run_new(o["new"]),
        }
        names = [a.only] if a.only else ["stock", "new"]
        graphs = {k: graph_of(fns[k]) for k in names}
        us = {k: [] for k in names}
        for _ in range(a.arms):
            for k in names + names[::-1]:
                us[k].append(time_graph(graphs[k], a.reps, a.flush))
        med = {k: round(statistics.median(v), 2) for k, v in us.items()}
        floor = sum(lens) * cda.draft.H_KV * cda.draft.HEAD_DIM * 2 / FLOOR_GBPS / 1e3
        r = {
            "lens": lens
            if len(lens) <= 8
            else f"{len(lens)} reqs, sum {sum(lens)}, max {max(lens)}",
            "us": med,
            "spread": {k: round(max(v) - min(v), 2) for k, v in us.items()},
            "floor_us": round(floor, 2),
            "flush_mb": a.flush,
        }
        if len(names) == 2:
            r["delta_us"] = round(med["new"] - med["stock"], 2)
            r["speedup"] = round(med["stock"] / med["new"], 2)
        results.append(r)
        print("TIME", json.dumps(r), flush=True)
        x = graphs = fns = None
        torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    print("TIME_DONE", flush=True)


if __name__ == "__main__":
    main()
