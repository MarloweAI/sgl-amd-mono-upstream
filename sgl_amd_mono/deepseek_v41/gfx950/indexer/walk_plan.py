# SPDX-License-Identifier: Apache-2.0
# The plan of sgl-project/sglang#43984 at 134919476c (kernels/ops/mono/gfx950/deepseek_v41/
# indexer/walk_plan.py), computed in K2i
"""The FULL-layer walk plan on the device: ATOM's host ``index_plan.fill_walk_plan``, each CTA
working out its own task in K2i.

A replayed graph lays the plan out from the step's device bounds, since the no-sync step has no
CPU sequence lengths in time. The step's first FULL (or producer) K2i of a ratio computes it, a
row a lane of every wave, and each CTA writes its task's row of the plan, which the step's later
K2i launches of the ratio read; it equals the host plan (``test_deepseek_v41_k2i.py`` checks)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

from sgl_amd_mono.common.ops import ballot, lane_gather, ld_i32, popcount, read_lane_i32
from sgl_amd_mono.common.plan import BLOCKS

from . import index_plan as ip

MAX_ROWS = 64  # a row a lane; a mono step has at most 48 rows


def _cdiv(a, b):
    return (a + b - 1) // b


def _scan(v, op):
    return fx.coop.warp_inclusive_scan(v, op, width=64)


def _sum(v):
    return fx.coop.warp_reduce(v, fx.ReductionOp.ADD, width=64)


@flyc.jit
def walk_task(a, rows: int, bid):
    """CTA ``bid``'s task of the step's walk plan, (lead, span, ch0, ch1) wave-uniform, from
    the bounds ``a["ibound"]`` and request owners ``a["ibatch"]`` of ``rows`` rows."""
    assert 0 < rows <= MAX_ROWS, rows
    t = fx.thread_idx.x % 64
    row = t < rows
    tt = fx.min(t, rows - 1)
    bound = row.select(ld_i32(a["ibound"], tt), fx.Int32(0))
    owner = row.select(ld_i32(a["ibatch"], tt), fx.Int32(-1))
    prev = lane_gather(owner, fx.max(t - 1, 0))
    run_start = row & ((t == 0) | (owner != prev))
    run_first = _scan(run_start.select(t, fx.Int32(0)), fx.ReductionOp.MAX)
    lead = row & (run_start | (((t - run_first) % ip.WALK_ROWS) == 0))
    piece = _scan(lead.select(fx.Int32(1), fx.Int32(0)), fx.ReductionOp.ADD) - 1
    chunks = (row & (bound > ip.TOPK)).select(_cdiv(bound, ip.CHUNK), fx.Int32(0))
    # a piece on its lead's lane: its rows are the lead and the next ones of its piece
    need, span = chunks, fx.Int32(1)
    for j in range_constexpr(1, ip.WALK_ROWS):
        u = fx.min(t + j, 63)
        same = (t + j < rows) & (lane_gather(piece, u) == piece)
        need = fx.max(need, same.select(lane_gather(chunks, u), fx.Int32(0)))
        span = span + same.select(fx.Int32(1), fx.Int32(0))
    need = lead.select(need, fx.Int32(0))
    pieces = popcount(ballot(need > 0))
    total = _sum(need)
    # the smallest width whose tasks fit in BLOCKS (feasibility is monotone)
    lo = fx.max(_cdiv(total, BLOCKS), 1)
    hi = fx.max(_cdiv(total, fx.max(BLOCKS - pieces, 1)), lo)
    while lo < hi:
        mid = (lo + hi) // 2
        fits = _sum(_cdiv(need, mid)) <= BLOCKS
        hi = fits.select(mid, hi)
        lo = fits.select(lo, mid + 1)
    k = _cdiv(need, lo)
    end = _scan(k, fx.ReductionOp.ADD)
    # this CTA's piece: the first whose tasks end past bid
    past = ballot(lead & (end > bid))
    has = past != 0
    at = popcount((past & (fx.Int64(0) - past)) - 1)
    at = has.select(at, fx.Int32(0))
    p_k = fx.max(read_lane_i32(k, at), 1)
    p_need = read_lane_i32(need, at)
    i = bid - (read_lane_i32(end, at) - read_lane_i32(k, at))
    zero = fx.Int32(0)
    return [
        has.select(at, zero),
        has.select(read_lane_i32(span, at), zero),
        has.select(i * p_need // p_k, zero),
        has.select((i + 1) * p_need // p_k, zero),
    ]


def new_plan(device) -> torch.Tensor:
    return torch.zeros(BLOCKS, 4, dtype=torch.int32, device=device)
