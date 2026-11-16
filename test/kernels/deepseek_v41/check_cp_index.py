# SPDX-License-Identifier: Apache-2.0
"""The context-parallel index selection of a TP group against each rank's whole-row selection.

One process a rank (gloo over ``--init-file``, rank r on GPU r). Every rank builds the same
synthetic FP4 plane and queries (``test_deepseek_v41_k2i.fp4_case``). A FULL source (ratio 2) and
a candidate producer (ratio 1) select on a two-level step twice: context-parallel (an
``Exchange``: K2i over this rank's runs of chunks, then the ranks' picks merged) and whole-row
(no exchange). Rows below the band and past it, from one row of 8 to six of 6. Checked: per
row, the same raw and page sets and the same producer blocks, eager and over replays of one
graph. ``--time``: µs a source of each, graph replay, every rank's mean.

Prints ``CP_CHECK PASS`` or ``CP_CHECK FAIL`` on rank 0."""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_deepseek_v41_k2i import _outs, _sorted_sets, fp4_case  # noqa: E402

# (kind, ratio, columns of a request's last row); the bands are runner.CP_BAND's
CASES = [
    ("full", 2, 3000),
    ("full", 2, 20000),
    ("full", 2, 65536),
    ("full", 2, 164112),
    ("full", 2, 524288),
    ("producer", 1, 20000),
    ("producer", 1, 131072),
    ("producer", 1, 328224),
    ("producer", 1, 1048576),
]
# the timed points: (kind, ratio, columns)
TIMED = [
    ("full", 2, 16384),
    ("full", 2, 65536),
    ("full", 2, 164112),
    ("full", 2, 524288),
    ("producer", 1, 32768),
    ("producer", 1, 131072),
    ("producer", 1, 328224),
    ("producer", 1, 1048576),
]


def _select(k2i, case, kind, ratio, outs):
    raw, page = outs
    rows = case["q"].shape[0]
    sel = k2i.select(
        kind=kind, ratio=ratio, rows=rows, per_req=case["per_req"],
        query=(case["q"], case["qscale"], case["w"]), plane=(case["payload"], case["kscale"]),
        page_table=case["table"], bounds=case["bounds"], raw_out=raw, page_out=page,
        two_level=True,
    )  # fmt: skip
    return sel.blocks.ids if sel.blocks is not None else None


def _same(a, b):
    return torch.equal(_sorted_sets(a), _sorted_sets(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--init-file", required=True)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    torch.cuda.set_device(a.rank)
    dev = torch.device("cuda", a.rank)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{a.init_file}",
        rank=a.rank,
        world_size=a.world_size,
    )
    group = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.gfx950.indexer import runner as ir
    from sgl_amd_mono.deepseek_v41.gfx950.indexer.topk_one import Exchange

    bound_max = max(c for _, _, c in CASES + TIMED)
    ex = Exchange(rank=a.rank, tp=a.world_size, group=group, device=dev)
    cp = ir.K2iRunner(device=dev, bound_max=bound_max, exchange=ex)
    whole = ir.K2iRunner(device=dev, bound_max=bound_max)
    bad = []

    def check(tag, kind, ratio, case):
        rows = case["q"].shape[0]
        o_cp, o_wh = _outs(rows), _outs(rows)
        b_cp = _select(cp, case, kind, ratio, o_cp)
        b_wh = _select(whole, case, kind, ratio, o_wh)
        torch.cuda.synchronize()
        ok = _same(o_cp[0], o_wh[0]) and _same(o_cp[1], o_wh[1])
        if b_cp is not None:
            ok &= _same(b_cp, b_wh)
        if not ok:
            bad.append(tag)
        return o_cp, b_cp, o_wh, b_wh

    torch.cuda.synchronize()
    for kind, ratio, cols in CASES:
        for per_req, reqs in ((6, 1), (8, 1), (1, 8), (2, 3)):
            ctxs = [max(cols - 97 * r, 600) for r in range(reqs)]
            case = fp4_case(
                ctxs=ctxs, seed=cols % 997 + per_req, per_req=per_req, dev=dev
            )
            check(f"{kind} {cols} {per_req}x{reqs}", kind, ratio, case)
    # graph replay: the cells are zeroed after each read, so every replay exchanges afresh
    for kind, ratio, cols in (("full", 2, 164112), ("producer", 1, 328224)):
        case = fp4_case(ctxs=[cols], seed=7, dev=dev)
        o_cp, b_cp, o_wh, b_wh = check(f"{kind} {cols} eager", kind, ratio, case)
        outs = _outs(case["q"].shape[0])
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ids = _select(cp, case, kind, ratio, outs)
        for i in range(a.replays):
            outs[0].fill_(-5)
            outs[1].fill_(-5)
            g.replay()
            torch.cuda.synchronize()
            ok = _same(outs[0], o_wh[0]) and _same(outs[1], o_wh[1])
            if ids is not None:
                ok &= _same(ids, b_wh)
            if not ok:
                bad.append(f"{kind} {cols} replay {i}")
                break
    flags = [None] * a.world_size
    dist.all_gather_object(flags, bad, group=group)
    if a.rank == 0:
        for r, f in enumerate(flags):
            for tag in f:
                print(f"rank {r}: DIFFERS {tag}", flush=True)
        print("CP_CHECK " + ("PASS" if not any(flags) else "FAIL"), flush=True)
    if a.time:
        _time(a, cp, whole, dev, group)
    dist.barrier(group=group)


def _time(a, cp, whole, dev, group):
    rows = []
    for kind, ratio, cols in TIMED:
        case = fp4_case(ctxs=[cols], seed=5, dev=dev)
        res = {}
        for name, k2i in (("whole", whole), ("cp", cp)):
            outs = _outs(case["q"].shape[0])
            _select(k2i, case, kind, ratio, outs)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _select(k2i, case, kind, ratio, outs)
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            t0 = time.perf_counter()
            for _ in range(a.iters):
                g.replay()
            torch.cuda.synchronize()
            res[name] = (time.perf_counter() - t0) / a.iters * 1e6
        rows.append((kind, cols, res["whole"], res["cp"]))
    every = [None] * a.world_size
    dist.all_gather_object(every, rows, group=group)
    if a.rank == 0:
        print(
            "CP_TIME kind cols | whole mean/max over ranks | cp mean/max | delta (max)",
            flush=True,
        )
        for i, (kind, cols, _, _) in enumerate(every[0]):
            w = [e[i][2] for e in every]
            c = [e[i][3] for e in every]
            print(
                f"CP_TIME {kind} {cols} | {sum(w) / len(w):.2f} / {max(w):.2f} | "
                f"{sum(c) / len(c):.2f} / {max(c):.2f} | {max(c) - max(w):+.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
