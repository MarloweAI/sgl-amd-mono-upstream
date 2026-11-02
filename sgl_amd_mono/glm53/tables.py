# SPDX-License-Identifier: Apache-2.0
"""A step's row tables (``Glm53Step``) and the NextN draft's top-k seed, each filled
by one Triton launch with the words of the torch ops they stand for.

``step_rows`` fills a width-W step from SGLang's rows: row s < S takes its slot from
``out_cache_loc`` (0, SGLang's padding slot, gives -1), its position clamped to the
context, request row s, and keys ``indptr[s] ..< indptr[s + 1]``: min(position + 1,
TOPK) for a row with a slot, or, with a ``seed`` (a draft decode step that reuses a
top-k: int32 [rows, TOPK] KV slots, -1 unused), the seed row's valid entries,
compacted in order into ``topk``. Rows S ..< W are pad rows. With ``ids_in`` the
step's token ids also go to ``ids`` (-1 for a pad row).

``write_seed`` writes a step's top-k as SGLang's seed: row s's keys, then -1.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

TOPK = 2048


@triton.jit
def _step_rows(
    ids_in,
    ids_stride,
    loc,
    loc_stride,
    pos,
    pos_stride,
    seed,
    seed_stride,
    ids,
    slots,
    positions,
    indptr,
    req_ids,
    topk,
    S,
    max_position,
    W: tl.constexpr,
    K: tl.constexpr,
    IDS: tl.constexpr,
    SEED: tl.constexpr,
    SEED_BY_POSITION: tl.constexpr,
):
    s = tl.program_id(0)
    r = tl.arange(0, W)
    in_step = r < S
    slot = tl.load(loc + r * loc_stride, mask=in_step, other=0).to(tl.int64)
    live = in_step & (slot != 0)
    p = tl.minimum(
        tl.load(pos + r * pos_stride, mask=in_step, other=0).to(tl.int64), max_position
    )
    if SEED and not SEED_BY_POSITION:
        j = tl.arange(0, K)
        rows = tl.load(
            seed + r[:, None] * seed_stride + j[None, :],
            mask=in_step[:, None],
            other=-1,
        )
        valid = (rows >= 0) & live[:, None]
        counts = tl.sum(valid.to(tl.int32), axis=1)
    else:
        counts = tl.where(live, tl.minimum(p + 1, K), 0).to(tl.int32)
    ends = tl.cumsum(counts, axis=0)
    start = tl.sum(tl.where(r < s, counts, 0), axis=0)
    mine = r == s
    tl.store(slots + r, tl.where(live, slot, -1), mask=mine)
    tl.store(positions + r, tl.where(in_step, p, 0), mask=mine)
    tl.store(req_ids + r, tl.where(in_step, r, 0).to(tl.int32), mask=mine)
    if IDS:
        tok = tl.load(ids_in + r * ids_stride, mask=in_step, other=-1)
        tl.store(ids + r, tl.where(in_step, tok, -1).to(tl.int32), mask=mine)
    if s == 0:
        tl.store(indptr, 0)
        tl.store(indptr + 1 + r, ends)
    if SEED:
        j = tl.arange(0, K)
        row_live = tl.sum(tl.where(mine, live.to(tl.int32), 0), axis=0) > 0
        row = tl.load(seed + s * seed_stride + j, mask=row_live, other=-1)
        if SEED_BY_POSITION:
            n = tl.sum(tl.where(mine, counts, 0), axis=0)
            take = j < n
            dst = start + j
        else:
            take = row >= 0
            dst = start + tl.cumsum(take.to(tl.int32), axis=0) - 1
        tl.store(topk + dst, row, mask=take & row_live)


def step_rows(
    st,
    out_cache_loc: torch.Tensor,
    positions: torch.Tensor,
    S: int,
    max_position: int,
    *,
    ids_in: Optional[torch.Tensor] = None,
    ids: Optional[torch.Tensor] = None,
    seed: Optional[torch.Tensor] = None,
    seed_by_position: bool = False,
) -> None:
    """Fill ``st`` (a ``Glm53Step`` of width W) for rows 0 ..< S in one launch. The
    row tensors may be strided (SGLang's per-step draft slots are a view);
    ``seed_by_position`` takes min(position + 1, TOPK) seed entries a row, a test
    fault."""
    W = st.slots.shape[0]
    if seed is not None and (
        seed.dtype != torch.int32
        or seed.dim() != 2
        or seed.shape[1] != TOPK
        or seed.stride(1) != 1
    ):
        raise ValueError(
            f"needs an int32 top-k seed [rows, {TOPK}], got {seed.dtype} "
            f"{tuple(seed.shape)}"
        )
    z = st.slots
    _step_rows[(W,)](
        ids_in if ids_in is not None else z,
        ids_in.stride(0) if ids_in is not None else 0,
        out_cache_loc,
        out_cache_loc.stride(0),
        positions,
        positions.stride(0),
        seed if seed is not None else z,
        seed.stride(0) if seed is not None else 0,
        ids if ids is not None else z,
        st.slots,
        st.positions,
        st.indptr,
        st.req_ids,
        st.topk,
        S,
        max_position,
        W=W,
        K=TOPK,
        IDS=ids is not None,
        SEED=seed is not None,
        SEED_BY_POSITION=seed_by_position,
        num_warps=8 if seed is not None else 4,
    )


@triton.jit
def _write_seed(topk, indptr, seed, seed_stride, rows_of, K: tl.constexpr):
    s = tl.program_id(0)
    j = tl.arange(0, K)
    src = tl.load(rows_of + s)
    a = tl.load(indptr + src)
    n = tl.load(indptr + src + 1) - a
    v = tl.load(topk + a + j, mask=j < n, other=-1)
    tl.store(seed + s * seed_stride + j, v)


def write_seed(st, seed: torch.Tensor, S: int, rows_of: torch.Tensor) -> None:
    """``seed`` row s := the keys of step row ``rows_of[s]``, then -1 (int32 [>= S,
    TOPK])."""
    if seed.dtype != torch.int32 or seed.shape[1] != TOPK or seed.stride(1) != 1:
        raise ValueError(
            f"needs an int32 seed [rows, {TOPK}], got {seed.dtype} {tuple(seed.shape)}"
        )
    _write_seed[(S,)](st.topk, st.indptr, seed, seed.stride(0), rows_of, K=TOPK)
