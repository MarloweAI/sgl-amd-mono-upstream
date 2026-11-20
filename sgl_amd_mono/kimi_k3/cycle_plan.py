# SPDX-License-Identifier: Apache-2.0
"""A DCP target verify's attention metadata for aiter's cp round-robin verify kernel, in one launch
from the device lengths: the tensors that kernel and ``get_mla_metadata`` read, each as SGLang's chain
writes it (``AiterAttnBackend._apply_cuda_graph_metadata``, target verify under DCP with the cprr
kernel: ``create_flashinfer_kv_indices_triton``, ``plan_dcp_decode_metadata``,
``_build_dcp_verify_token_table`` and ``_build_dcp_verify_ragged_indices``, about 40 launches).

Per request b with kv length ``L = seq_len + Q`` (the verify window included), this rank owns the
positions ``p = rank + 8 j`` for ``j < n_b = L // 8 + (L % 8 > rank)``:

- ``qo_indptr[b] = Q b``;
- ``global_kv_indptr[b + 1] = L_0 + ... + L_b``;
- ``cprr_kv_indptr[b + 1] = sum over c <= b of max(n_c, 1)``;
- ``cprr_kv_indices[cprr_kv_indptr[b] + j]`` = the page of ``req_to_token[req, rank + 8 j]`` (``// 8``, then
  the v2p table where the pool has one), and 0 for an empty shard's one slot;
- ``local_kv_lens[Q b + i] = n_b`` and ``cprr_empty_rows[Q b + i] = (n_b == 0)``.

The global token list, its shard and the rectangular token table SGLang builds on the way are not
written: the cprr verify reads none of them."""

from __future__ import annotations

import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _verify_plan_kernel(
    seq_lens_ptr,
    req_pool_ptr,
    req_to_token_ptr,
    v2p_ptr,
    qo_indptr_ptr,
    global_kv_indptr_ptr,
    cprr_kv_indptr_ptr,
    cprr_kv_indices_ptr,
    local_kv_lens_ptr,
    empty_rows_ptr,
    bs,
    req_to_token_stride,
    Q: tl.constexpr,
    Q_POW2: tl.constexpr,
    DCP: tl.constexpr,
    RANK: tl.constexpr,
    BS_POW2: tl.constexpr,
    HAS_V2P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    part = tl.program_id(1)
    parts = tl.num_programs(1)
    c = tl.arange(0, BS_POW2)
    lens = tl.load(seq_lens_ptr + c, mask=c < bs, other=0).to(tl.int64) + Q
    lens = tl.where(c < bs, lens, 0)
    n = lens // DCP + tl.where(lens % DCP > RANK, 1, 0)
    clamped = tl.where(c < bs, tl.maximum(n, 1), 0)
    n_b = tl.sum(tl.where(c == b, n, 0))
    start = tl.sum(tl.where(c < b, clamped, 0))
    if part == 0:
        if b == 0:
            tl.store(qo_indptr_ptr, 0)
            tl.store(global_kv_indptr_ptr, 0)
            tl.store(cprr_kv_indptr_ptr, 0)
        tl.store(qo_indptr_ptr + b + 1, ((b + 1) * Q).to(tl.int32))
        tl.store(global_kv_indptr_ptr + b + 1, tl.sum(tl.where(c <= b, lens, 0)).to(tl.int32))
        tl.store(cprr_kv_indptr_ptr + b + 1, tl.sum(tl.where(c <= b, clamped, 0)).to(tl.int32))
        rows = tl.arange(0, Q_POW2)
        rmask = rows < Q
        tl.store(local_kv_lens_ptr + b * Q + rows, tl.zeros((Q_POW2,), tl.int32) + n_b.to(tl.int32), mask=rmask)
        tl.store(empty_rows_ptr + b * Q + rows, (tl.zeros((Q_POW2,), tl.int64) + n_b) == 0, mask=rmask)
        if n_b == 0:
            tl.store(cprr_kv_indices_ptr + start, 0)
    req = tl.load(req_pool_ptr + b).to(tl.int64)
    row = req_to_token_ptr + req * req_to_token_stride
    for blk in range(part, tl.cdiv(n_b, BLOCK), parts):
        j = blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = j < n_b
        virtual = tl.load(row + RANK + j * DCP, mask=mask, other=0)
        pages = virtual // DCP // 1
        if HAS_V2P:
            pages = tl.where(virtual < 0, 0, pages)
            physical = tl.load(v2p_ptr + pages, mask=mask, other=0)
            pages = tl.maximum(physical, 0)
        tl.store(cprr_kv_indices_ptr + start + j, pages.to(tl.int32), mask=mask)


def verify_plan(
    *,
    seq_lens,
    req_pool_indices,
    req_to_token,
    v2p,
    q_len: int,
    dcp_size: int,
    dcp_rank: int,
    qo_indptr,
    global_kv_indptr,
    cprr_kv_indptr,
    cprr_kv_indices,
    local_kv_lens,
    empty_rows,
    parts: int = 16,
) -> None:
    """Writes qo_indptr[:bs+1], global_kv_indptr[:bs+1], cprr_kv_indptr[:bs+1], the shards' runs of
    cprr_kv_indices, local_kv_lens[:Q bs] and empty_rows[:Q bs] for the bs = len(seq_lens) requests."""
    bs = seq_lens.shape[0]
    _verify_plan_kernel[(bs, parts)](
        seq_lens,
        req_pool_indices,
        req_to_token,
        v2p if v2p is not None else req_to_token,
        qo_indptr,
        global_kv_indptr,
        cprr_kv_indptr,
        cprr_kv_indices,
        local_kv_lens,
        empty_rows,
        bs,
        req_to_token.stride(0),
        Q=q_len,
        Q_POW2=triton.next_power_of_2(q_len),
        DCP=dcp_size,
        RANK=dcp_rank,
        BS_POW2=max(triton.next_power_of_2(bs), 2),
        HAS_V2P=v2p is not None,
        BLOCK=BLOCK,
        num_warps=4,
    )


@triton.jit
def _draft_plan_kernel(
    seq_lens_ptr,
    row_ids_ptr,
    ids_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    mask_indptr_ptr,
    bs,
    ids_stride,
    W: tl.constexpr,
    BS_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    part = tl.program_id(1)
    parts = tl.num_programs(1)
    c = tl.arange(0, BS_POW2)
    lens = tl.load(seq_lens_ptr + c, mask=c < bs, other=0).to(tl.int64)
    lens = tl.where(c < bs, lens, 0)
    len_b = tl.sum(tl.where(c == b, lens, 0))
    start = tl.sum(tl.where(c < b, lens, 0)).to(tl.int32)
    if part == 0:
        if b == 0:
            tl.store(qo_indptr_ptr, 0)
        tl.store(qo_indptr_ptr + b + 1, ((b + 1) * W).to(tl.int32))
        tl.store(kv_indptr_ptr + b + 1, tl.sum(tl.where(c <= b, lens, 0)))
        tl.store(mask_indptr_ptr + b + 1, tl.sum(tl.where(c <= b, W * (lens + W), 0)))
    row = ids_ptr + tl.load(row_ids_ptr + b).to(tl.int64) * ids_stride
    for blk in range(part, tl.cdiv(len_b, BLOCK), parts):
        offset = tl.arange(0, BLOCK).to(tl.int64) + blk * BLOCK
        mask = offset < len_b
        tl.store(kv_indices_ptr + start + offset, tl.load(row + offset, mask=mask), mask=mask)


def draft_plan(
    *,
    seq_lens,
    row_ids,
    ids,
    ids_stride: int,
    width: int,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_indptr,
    parts: int = 16,
) -> None:
    """A draft block's verify metadata over its full context, in one launch, as SGLang's Triton
    backend writes it for a target-verify replay (``_update_target_verify_buffers``: qo_indptr, the
    kv indptr and the context's token ids gathered from the read table ``ids`` rows ``row_ids``,
    the mask indptr), for ``bs = len(seq_lens)`` requests of ``width`` rows. Index 0 of each indptr
    is left as it is (0)."""
    bs = seq_lens.shape[0]
    _draft_plan_kernel[(bs, parts)](
        seq_lens,
        row_ids,
        ids,
        qo_indptr,
        kv_indptr,
        kv_indices,
        mask_indptr,
        bs,
        ids_stride,
        W=width,
        BS_POW2=max(triton.next_power_of_2(bs), 2),
        BLOCK=BLOCK,
        num_warps=4,
    )
