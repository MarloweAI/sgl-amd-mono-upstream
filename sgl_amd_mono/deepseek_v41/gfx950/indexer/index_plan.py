# SPDX-License-Identifier: Apache-2.0
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 68506823, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/index_plan.py (the FP4 plane's parts), by way of
# sgl-project/sglang#43984 at 134919476c
"""The FP4 scorer's tasks of a step, defined once.

The kernel calls these on traced Int32 and the CPU test on Python ints, so the test checks what
the kernel runs; only non-negative operands reach ``//`` and ``%``, where Python's floor and the
device's truncation agree. A token's ``bound`` is the columns its scorer may read.
"""

import numpy as np

from sgl_amd_mono.common.plan import cdiv, imin, sel

CHUNK = 128  # columns a score task: 16 tiles of 8 rows
TOPK = 512
TOPK_BLOCKS = 2048
BLOCK_ROWS = 8  # a candidate block, and an index tile (an FP4 plane's page)
STEP = 64  # keys an FP4 walk step: a wave's 4 MFMA tiles, 8 pages
WALK_ROWS = 6  # query rows an FP4 walk scores off each key load
HEADS = 32  # the indexer's heads (``index_n_heads``)
DIM = 128  # an index head's dims (``index_head_dim``)


def scored(bound):
    """Whether the scorer runs for a token.

    At ``bound <= TOPK`` SGLang's top-k takes every column and reads no logits."""
    return bound > TOPK


def score_chunks(bound):
    """A token's scored chunks; chunk ch covers columns [CHUNK ch, CHUNK (ch + 1))."""
    return sel(scored(bound), cdiv(bound, CHUNK), 0)


# a context-parallel band no row reaches: every rank scores and selects whole rows
NO_CP = 1 << 30


def cp_chunks(bound, rank, tp, band):
    """The scored chunks [lo, hi) of a token that rank ``rank`` of ``tp`` scores and selects.

    A token seeing ``band`` columns or more is context-parallel: each rank takes an equal run of
    its chunks, the last rank the rest; below the band every rank takes all of them."""
    n = score_chunks(bound)
    on = bound >= band
    per = sel(on, cdiv(n, tp), n)
    lo = sel(on, imin(rank * per, n), 0)
    return lo, imin(lo + per, n)


def fill_walk_plan(*, bounds, owners, plan, cp=(0, 1, NO_CP)):
    """The FP4 plane's FULL-layer score tasks of a step, on the host.

    ``plan`` [ctas, 4] gets (lead, span, ch0, ch1) a CTA -- chunks [ch0, ch1) of a piece, the
    ``span`` tokens from ``lead``: at most WALK_ROWS adjacent tokens of one owner, scored off each
    key load -- and zeros past the last task. A piece's chunks run from the first of its rows'
    ``cp_chunks`` (``cp``: rank, tp, band) to the last, split evenly at the smallest width that
    fits ``ctas``, as one chunk wider leaves a wave per CTA a step behind the rest."""
    n, ctas = bounds.shape[0], plan.shape[0]
    plan.fill(0)
    t = np.arange(n)
    run_start = np.ones(n, dtype=bool)
    run_start[1:] = owners[1:] != owners[:-1]
    run_first = np.maximum.accumulate(np.where(run_start, t, 0))
    lead = np.flatnonzero(run_start | ((t - run_first) % WALK_ROWS == 0))
    span = np.diff(lead, append=n)
    rows = [cp_chunks(int(b), *cp) for b in bounds.tolist()]
    base, need = [], []
    for a, b in zip(lead.tolist(), (lead + span).tolist()):
        live = [(lo, hi) for lo, hi in rows[a:b] if hi > lo]
        base.append(min(lo for lo, _ in live) if live else 0)
        need.append(max(hi for _, hi in live) - base[-1] if live else 0)
    base, need = np.array(base, dtype=np.int64), np.array(need, dtype=np.int64)
    live = need > 0
    lead, span, base, need = lead[live], span[live], base[live], need[live]
    if not need.size:
        return
    assert need.size < ctas, (need.size, ctas)
    # feasibility is monotone in width, and ceil(total / (ctas - pieces)) fits: each ceil adds < 1
    total = int(need.sum())
    lo = max(cdiv(total, ctas), 1)
    widths = np.arange(lo, max(cdiv(total, ctas - need.size), lo) + 1)[:, None]
    width = lo + int(np.argmax(cdiv(need, widths).sum(1) <= ctas))
    k = cdiv(need, width)
    piece = np.repeat(np.arange(need.size), k)
    i = np.arange(piece.size) - np.repeat(np.cumsum(k) - k, k)
    tasks = plan[: piece.size]
    tasks[:, 0] = lead[piece]
    tasks[:, 1] = span[piece]
    tasks[:, 2] = base[piece] + i * need[piece] // k[piece]
    tasks[:, 3] = base[piece] + (i + 1) * need[piece] // k[piece]


# a REINDEX token's chunks at most: its candidate blocks' rows
REINDEX_CHUNKS = TOPK_BLOCKS * BLOCK_ROWS // CHUNK


def reindex_split(tokens, ctas, least):
    """A REINDEX layer's tasks a token, at least ``least`` chunks each.

    Fixed by the step's width alone, so a CTA finds its task from its index with no plan."""
    return max(1, min(ctas // tokens, REINDEX_CHUNKS // least))


def reindex_chunks(i, per):
    """Chunks [ch0, ch1) of a REINDEX token's task i of ``per``."""
    return i * REINDEX_CHUNKS // per, (i + 1) * REINDEX_CHUNKS // per
