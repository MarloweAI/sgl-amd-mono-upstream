# SPDX-License-Identifier: Apache-2.0
"""A DSpark verify step's attention metadata (``make_forward_metadata_from_raw_verify``,
~40 launches) and a draft block's (~12), with their SWA write slots, in one launch each,
every tensor as SGLang writes it. ``SGLANG_AMD_MONO_DSV41_METADATA=0`` keeps SGLang's.

Inside a captured forward graph the two builds write tensors kept per shape and launch
nothing: the step's verify window launch (SGLang's ``assign_extend_cache_locs_uniform``,
which every DSpark decode step runs before its draft and verify graphs) computes both from
the same request rows, prefix lengths and ``req_to_token``, as extra programs of its own
launch. A step then pays one launch for its window and both metadata builds. The graph
runners pad a batch to the next captured bucket, so a window launch builds the bucket's
tensors: the step's rows from its own inputs, the pad rows from SGLang's pad inputs."""

from __future__ import annotations

import contextvars
import sys

import torch
import triton
import triton.language as tl

from sgl_amd_mono.deepseek_v41.glue import chose, flag, route

ENV = "SGLANG_AMD_MONO_DSV41_METADATA"
HIP = "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix"

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/attention/deepseek_v4_backend_hip_radix.py": (
        "DeepseekV4HipRadixBackend",
        "make_forward_metadata_from_raw_verify(raw_metadata)",
        "init_forward_metadata_in_graph(forward_batch)",
        "init_forward_metadata_dspark_draft_block(max_seq_len)",
        "init_forward_metadata_dspark_draft_block(req_pool_indices)",
        "init_forward_metadata_dspark_draft_block(seq_lens)",
        "init_forward_metadata_dspark_draft_block(out_cache_loc)",
        "init_forward_metadata_dspark_draft_block(block_size)",
        "_uses_dspark_draft_window",
        "token_req_indices",
        "_swa_loc_i32(out_cache_loc)",
        "DSV4AttnMetadata",
        "DSV4Metadata",
        "SWA_WINDOW",
        "PAGE_INDEX_ALIGNED_SIZE",
        "_create_flashmla_metadata",
        "target_verify_num_draft_tokens",
        "MAX_SEQ_LEN_FOR_CAPTURE",
        "need_compress",
        "has_c4",
        "has_c128",
        "present_ratios",
        "low_ratios",
        "index_topk",
        "page_size",
        "req_to_token",
        "token_to_kv_pool",
        "cuda_int32_kwargs",
        "forward_metadata",
    ),
    "srt/arg_groups/fields/exec_.py": ("dsa_topk_backend",),
    "kernels/ops/speculative/cache_locs.py": (
        "assign_extend_cache_locs_uniform",
        "assign_extend_cache_locs_uniform[(batch_size,)]",
        "START_BY_ROW",
        "RING_ROWS",
    ),
    "srt/layers/attention/dsv4/metadata.py": ("PagedIndexerMetadata",),
    "srt/layers/attention/dsv4/unified_kv_gate.py": ("is_unified_kv_triton",),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "DeepSeekV4TokenToKVPool",
        "translate_loc_from_full_to_swa",
        "full_to_swa_index_mapping",
        "get_index_k_page_size",
        "request_window",
        "swa_req_ring_size",
        "swa_valid_from",
    ),
}

# the backend whose init_forward_metadata_in_graph runs
_IN_GRAPH = contextvars.ContextVar("mono_dsv41_in_graph", default=None)


# the kernel's per-row outputs: a row of L32 / L64 each, by these indices
_I32 = (
    "seq_lens_casual",
    "positions_casual",
    "swa_topk_lengths",
    "c4_topk_lengths_raw",
    "c4_topk_lengths_clamp1",
    "c128_topk_lengths_raw",
    "c128_topk_lengths_clamp1",
    "c1_topk_lengths_clamp1",
    "c2_topk_lengths_clamp1",
    "c4_sparse_topk_lengths",
    "c4_sparse_topk_lengths_raw",
    "c1_sparse_topk_lengths",
    "c2_sparse_topk_lengths",
    "swa_slots",
)
_I64 = ("c4_out_loc", "c128_out_loc", "c1_out_loc", "c2_out_loc", "req_rows")
# the -1 filled top-k buffers, [rows, NEG_W] each
_NEG = (
    "c4_sparse_page_indices",
    "c1_sparse_page_indices",
    "c2_sparse_page_indices",
    "c1_sparse_raw_indices",
    "c2_sparse_raw_indices",
)


@triton.jit
def _verify_meta_row(
    row,
    blk,
    req,
    seq,
    LOCB,
    real,
    R2T,
    MAP,
    L32,
    L64,
    NEG,
    SWA,
    PT,
    PT1,
    PT2,
    C128_PI,
    rows,
    r2t_stride,
    map_n,
    num_cols,
    topk,
    QO: tl.constexpr,
    W: tl.constexpr,
    PAGE: tl.constexpr,
    C128P: tl.constexpr,
    W128: tl.constexpr,
    BPP1: tl.constexpr,
    BPP2: tl.constexpr,
    NEG_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program of a verify step's metadata: token row ``row`` (of request ``req`` with
    prefix length ``seq``; ``LOCB`` its request's write locations, read as 0 where not
    ``real``), page-table block ``blk``."""
    req = req.to(tl.int64)
    casual = (seq + QO + (row % QO - QO + 1)).to(tl.int32)
    # page_table_from_req_to_token, its expansions and the c128 page indices
    col = blk * BLOCK + tl.arange(0, BLOCK)
    cm = col < num_cols
    v = tl.load(R2T + req * r2t_stride + col.to(tl.int64) * PAGE, mask=cm, other=0)
    q = v // PAGE
    q = tl.where((v < 0) & (v % PAGE != 0), q - 1, q).to(tl.int32)
    r64 = row.to(tl.int64)
    tl.store(PT + r64 * num_cols + col, q, mask=cm)
    if BPP1 > 1:
        k1 = tl.arange(0, BPP1)
        e1 = q.to(tl.int64)[:, None] * BPP1 + k1[None, :]
        o1 = r64 * (num_cols * BPP1) + col[:, None] * BPP1 + k1[None, :]
        tl.store(PT1 + o1, e1.to(tl.int32), mask=cm[:, None])
    if BPP2 > 1:
        k2 = tl.arange(0, BPP2)
        e2 = q.to(tl.int64)[:, None] * BPP2 + k2[None, :]
        o2 = r64 * (num_cols * BPP2) + col[:, None] * BPP2 + k2[None, :]
        tl.store(PT2 + o2, e2.to(tl.int32), mask=cm[:, None])
    c128_raw = casual // 128
    kc = tl.arange(0, C128P)
    cidx = col[:, None] * C128P + kc[None, :]
    cval = tl.where(cidx < c128_raw, q[:, None] * C128P + kc[None, :], -1)
    tl.store(C128_PI + r64 * W128 + cidx, cval, mask=cm[:, None])
    if blk == 0:
        pad = num_cols * C128P + tl.arange(0, 64)
        tl.store(C128_PI + r64 * W128 + pad, -1, mask=pad < W128)
        loc = tl.load(LOCB + row % QO, mask=real, other=0).to(tl.int64)
        c4_raw = casual // 4
        c4_c1 = tl.maximum(c4_raw, 1)
        c1_c1 = tl.maximum(casual, 1)
        c2_c1 = tl.maximum(casual // 2, 1)
        # the rows, init_compression_metadata's, low_ratio_compression_metadata's and
        # sparse_buffers' lengths, in _I32 order
        tl.store(L32 + row, casual)
        tl.store(L32 + rows + row, casual - 1)
        tl.store(L32 + 2 * rows + row, tl.minimum(casual, W))
        tl.store(L32 + 3 * rows + row, c4_raw)
        tl.store(L32 + 4 * rows + row, c4_c1)
        tl.store(L32 + 5 * rows + row, c128_raw)
        tl.store(L32 + 6 * rows + row, tl.maximum(c128_raw, 1))
        tl.store(L32 + 7 * rows + row, c1_c1)
        tl.store(L32 + 8 * rows + row, c2_c1)
        tl.store(L32 + 9 * rows + row, tl.minimum(c4_c1, topk))
        tl.store(L32 + 10 * rows + row, tl.minimum(c4_raw, topk))
        tl.store(L32 + 11 * rows + row, tl.minimum(c1_c1, topk))
        tl.store(L32 + 12 * rows + row, tl.minimum(c2_c1, topk))
        sl = tl.load(MAP + tl.where(loc < 0, loc + map_n, loc))
        tl.store(L32 + 13 * rows + row, sl.to(tl.int32))
        # the write slots of ratios 4, 128, 1, 2 and the request row, in _I64 order
        tl.store(L64 + row, tl.where((casual % 4) == 0, loc // 4, 0))
        tl.store(L64 + rows + row, tl.where((casual % 128) == 0, loc // 128, 0))
        tl.store(L64 + 2 * rows + row, loc)
        tl.store(L64 + 3 * rows + row, tl.where((casual % 2) == 0, loc >> 1, -1))
        tl.store(L64 + 4 * rows + row, req)
        w = tl.arange(0, NEG_W)
        for j in tl.static_range(5):
            tl.store(NEG + (j * rows + row).to(tl.int64) * NEG_W + w, -1)
        # get_swa_page_indices: a slot before the request reads req_to_token as -1,
        # so the mapping's last entry, as SGLang's indexing wraps it
        k = tl.arange(0, W)
        off = casual - 1 - k
        before = off < 0
        raw = tl.load(R2T + req * r2t_stride + tl.where(before, 0, off)).to(tl.int64)
        raw = tl.where(before, -1, raw)
        sw = tl.load(MAP + tl.where(raw < 0, raw + map_n, raw))
        tl.store(SWA + row * W + k, sw.to(tl.int32))


@triton.jit
def _verify_meta_kernel(
    SEQ,
    REQ,
    LOC,
    R2T,
    MAP,
    L32,
    L64,
    NEG,
    SWA,
    PT,
    PT1,
    PT2,
    C128_PI,
    rows,
    r2t_stride,
    map_n,
    num_cols,
    topk,
    QO: tl.constexpr,
    W: tl.constexpr,
    PAGE: tl.constexpr,
    C128P: tl.constexpr,
    W128: tl.constexpr,
    BPP1: tl.constexpr,
    BPP2: tl.constexpr,
    NEG_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b = row // QO
    _verify_meta_row(
        row,
        blk,
        tl.load(REQ + b),
        tl.load(SEQ + b),
        LOC + b * QO,
        True,
        R2T,
        MAP,
        L32,
        L64,
        NEG,
        SWA,
        PT,
        PT1,
        PT2,
        C128_PI,
        rows,
        r2t_stride,
        map_n,
        num_cols,
        topk,
        QO,
        W,
        PAGE,
        C128P,
        W128,
        BPP1,
        BPP2,
        NEG_W,
        BLOCK,
    )


def _inputs(backend, raw):
    """(req_to_token, mapping, bpp1, bpp2) where the fused build is SGLang's, else None."""
    hip = sys.modules[HIP]
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    pool, r2t = backend.token_to_kv_pool, backend.req_to_token
    seq, req, loc = raw.seq_lens, raw.req_pool_indices, raw.out_cache_loc
    mapping = getattr(pool, "full_to_swa_index_mapping", None)
    rows = seq.shape[0] * backend.target_verify_num_draft_tokens
    if (
        pool.request_window is not None
        or is_unified_kv_triton()
        or getattr(pool, "unified_swa_pages", 0)
        or not backend.need_compress
        or backend.has_c4
        or backend.has_c128
        or tuple(backend.low_ratios) != (1, 2)
        or backend.page_size != 256
        or hip.SWA_WINDOW % hip.PAGE_INDEX_ALIGNED_SIZE
        or backend.index_topk % hip.PAGE_INDEX_ALIGNED_SIZE
        or type(pool).translate_loc_from_full_to_swa
        is not DeepSeekV4TokenToKVPool.translate_loc_from_full_to_swa
        or mapping is None
        or mapping.dim() != 1
        or not mapping.is_contiguous()
        or r2t.dim() != 2
        or r2t.stride(1) != 1
        or seq.dim() != 1
        or not seq.is_cuda
        or not seq.is_contiguous()
        or req.shape != seq.shape
        or not req.is_contiguous()
        or loc is None
        or loc.shape != (rows,)
        or not loc.is_contiguous()
    ):
        return None
    bpp = []
    for r in (1, 2):
        n = (backend.page_size // r) // pool.get_index_k_page_size(r)
        if n < 1 or n & (n - 1) or n * pool.get_index_k_page_size(r) * r != 256:
            return None
        bpp.append(n)
    return r2t, mapping, bpp[0], bpp[1]


def _fused(backend, raw, inputs):
    """make_forward_metadata_from_raw_verify's DSV4Metadata from one launch."""
    hip = sys.modules[HIP]
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata

    r2t, mapping, bpp1, bpp2 = inputs
    qo, page = backend.target_verify_num_draft_tokens, backend.page_size
    bs = raw.seq_lens.shape[0]
    rows, dev, w = bs * qo, raw.seq_lens.device, hip.SWA_WINDOW
    num_cols = max(
        0, (min(backend.MAX_SEQ_LEN_FOR_CAPTURE, r2t.shape[1]) + page - 1) // page
    )
    align = hip.PAGE_INDEX_ALIGNED_SIZE
    c128p = page // 128
    w128 = -(-(num_cols * c128p) // align) * align
    neg_w = -(-backend.index_topk // align) * align
    i32, i64 = dict(dtype=torch.int32, device=dev), dict(dtype=torch.int64, device=dev)

    def make():
        pt = torch.empty((rows, num_cols), **i32)
        return dict(
            L32=torch.empty((len(_I32), rows), **i32),
            L64=torch.empty((len(_I64), rows), **i64),
            NEG=torch.empty((len(_NEG), rows, neg_w), **i32),
            SWA=torch.empty((rows, w), **i32),
            PT=pt,
            PT1=torch.empty((rows, num_cols * bpp1), **i32) if bpp1 > 1 else pt,
            PT2=torch.empty((rows, num_cols * bpp2), **i32) if bpp2 > 1 else pt,
            C128_PI=torch.empty((rows, w128), **i32),
        )

    key = ("verify", rows, num_cols, w, w128, neg_w, bpp1, bpp2, dev)
    out = _kept(key, make) if _IN_GRAPH.get() is not None else make()
    l32, l64, neg, pt, pt1, pt2 = (
        out[k] for k in ("L32", "L64", "NEG", "PT", "PT1", "PT2")
    )
    swa, c128_pi = out["SWA"], out["C128_PI"]
    t = dict(zip(_I32, l32)) | dict(zip(_I64, l64)) | dict(zip(_NEG, neg))
    if rows:
        block = min(1024, triton.next_power_of_2(max(num_cols, 1)))
        spec = dict(
            out,
            R2T=r2t,
            MAP=mapping,
            rows=rows,
            r2t_stride=r2t.stride(0),
            map_n=mapping.shape[0],
            num_cols=num_cols,
            topk=backend.index_topk,
            QO=qo,
            W=w,
            PAGE=page,
            C128P=c128p,
            W128=w128,
            BPP1=bpp1,
            BPP2=bpp2,
            NEG_W=neg_w,
            BLOCK=block,
        )
        if not _fold(key, spec, backend.get_cuda_graph_seq_len_fill_value()):
            grid = (rows, max(1, triton.cdiv(num_cols, block)))
            _verify_meta_kernel[grid](
                raw.seq_lens,
                raw.req_pool_indices,
                raw.out_cache_loc,
                **spec,
                num_warps=4,
            )
    core = hip.DSV4AttnMetadata(
        page_size=page,
        raw_out_loc=raw.out_cache_loc,
        seq_lens_casual=t["seq_lens_casual"],
        cuda_int32_kwargs=backend.cuda_int32_kwargs,
        positions_casual=t["positions_casual"],
        page_table=pt,
        swa_page_indices=swa,
        swa_topk_lengths=t["swa_topk_lengths"],
        index_topk=backend.index_topk,
        present_ratios=backend.present_ratios,
        low_ratios=backend.low_ratios,
        request_window_layout=None,
    )
    # the C4 / C128 fields only where the pool has those ratios, as SGLang's
    # init_compression_metadata and init_flashmla_related leave them
    c4_c128 = core.has_c4 or core.has_c128
    for name in _I32[3:13] + _I64[:4] + _NEG:
        if name.startswith("c4_sparse_"):
            keep = core.has_c4
        else:
            keep = c4_c128 or not name.startswith(("c4_", "c128_"))
        setattr(core, name, t[name] if keep else None)
    core.c128_page_indices = c128_pi if c4_c128 else None
    present = {0: True, 1: 1 in core.low_ratios, 2: 2 in core.low_ratios}
    present |= {4: core.has_c4, 128: core.has_c128}
    for r, on in present.items():
        meta = hip._create_flashmla_metadata() if on else None
        setattr(core, f"c{r}_flashmla_metadata", meta)
    slots = (raw.out_cache_loc, t["swa_slots"], t["req_rows"], raw.req_pool_indices)
    core.__dict__["_mono_slots"] = slots
    clamp1 = (t["c1_topk_lengths_clamp1"], t["c2_topk_lengths_clamp1"])
    indexers = {
        f"c{r}_indexer_metadata": PagedIndexerMetadata(
            page_size=page,
            compressed_page_size=backend.token_to_kv_pool.get_index_k_page_size(r),
            page_table=table,
            compressed_seq_lens=lens,
            use_topk_v2=backend.dsa_topk_backend.should_use_topk_v2(),
            compress_ratio=r,
        )
        for r, table, lens in ((1, pt1, clamp1[0]), (2, pt2, clamp1[1]))
    }
    return hip.DSV4Metadata(
        core, None, c4_compress_metadata=None, c128_compress_metadata=None, **indexers
    )


def _verify_metadata(orig, self, raw_metadata, *args, **kwargs):
    inputs = None if args or kwargs else _inputs(self, raw_metadata)
    if chose("verify_meta", inputs is not None):
        return _fused(self, raw_metadata, inputs)
    return orig(self, raw_metadata, *args, **kwargs)


def _in_graph(orig, self, forward_batch, *args, **kwargs):
    token = _IN_GRAPH.set(self)
    try:
        return orig(self, forward_batch, *args, **kwargs)
    finally:
        _IN_GRAPH.reset(token)


def _slots(kind: int, t: torch.Tensor):
    """The fused build's SWA slots (kind 1, ``t`` the step's out_cache_loc, which the raw
    metadata holds as an int32 copy) or request row per token (kind 2, ``t`` its
    req_pool_indices), else None."""
    backend = _IN_GRAPH.get()
    core = getattr(
        getattr(backend, "forward_metadata", None), "core_attn_metadata", None
    )
    held = None if core is None else core.__dict__.get("_mono_slots")
    ref = None if held is None else held[0 if kind == 1 else 3]
    if (
        ref is None
        or t.shape != ref.shape
        or (kind == 2 and t.data_ptr() != ref.data_ptr())
    ):
        return None
    return held[kind]


def _translate(orig, self, kv_indices, *args, **kwargs):
    got = None if args or kwargs else _slots(1, kv_indices)
    if got is not None and chose("swa_slots", True):
        return got
    return orig(self, kv_indices, *args, **kwargs)


def _swa_loc_i32(orig, self, out_cache_loc):
    """The backend's int32 SWA locations of ``out_cache_loc`` (current SGLang looks them up
    in one launch, past the pool's translate): the fused build's SWA slots, as ``_translate``."""
    got = _slots(1, out_cache_loc)
    if got is not None and chose("swa_slots", True):
        return got if got.dtype == torch.int32 else got.to(torch.int32)
    return orig(self, out_cache_loc)


def _token_req_indices(orig, forward_batch, *args, **kwargs):
    got = None
    if (
        not args
        and not kwargs.get("num_tokens")
        and forward_batch.forward_mode.is_target_verify()
    ):
        got = _slots(2, forward_batch.req_pool_indices)
        n = forward_batch.batch_size * int(forward_batch.spec_info.draft_token_num)
        if got is not None and got.numel() != n:
            got = None
    if got is not None and chose("verify_req_rows", True):
        return got
    return orig(forward_batch, *args, **kwargs)


@triton.jit
def _block_meta_row(
    q,
    blk,
    req,
    seq,
    LOCB,
    real,
    R2T,
    MAP,
    VALID,
    L32,
    SWA,
    PT,
    n,
    r2t_stride,
    num_cols,
    map_n,
    QO: tl.constexpr,
    W: tl.constexpr,
    TW: tl.constexpr,
    TW_BLOCK: tl.constexpr,
    PAGE: tl.constexpr,
    HAS_VALID: tl.constexpr,
    HAS_SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program of a draft block's metadata: token row ``q`` (of request ``req`` with
    prefix length ``seq``; ``LOCB`` its block's write locations, read as 0 where not
    ``real``), page-table block ``blk``."""
    rp = req.to(tl.int64)
    col = blk * BLOCK + tl.arange(0, BLOCK)
    cm = col < num_cols
    v = tl.load(R2T + rp * r2t_stride + col.to(tl.int64) * PAGE, mask=cm, other=0)
    pq = v // PAGE
    pq = tl.where((v < 0) & (v % PAGE != 0), pq - 1, pq)
    tl.store(PT + q.to(tl.int64) * num_cols + col, pq.to(tl.int32), mask=cm)
    if blk == 0:
        casual = (seq + QO + (q % QO - QO + 1)).to(tl.int32)
        tl.store(L32 + q, casual)
        tl.store(L32 + n + q, casual - 1)
        # ComputeDsparkWindowGather, then BuildDsparkSwaPageIndices
        prefix = (seq + 1).to(tl.int32).to(tl.int64) - 1
        cl = tl.minimum(prefix, W)
        if HAS_VALID:
            cl = tl.minimum(cl, tl.maximum(prefix - tl.load(VALID + rp), 0))
        cl = cl.to(tl.int32)
        k = tl.arange(0, TW_BLOCK)
        kmask = k < TW
        in_window = k < cl
        src_col = tl.minimum(tl.maximum((W - cl) + k, 0), W - 1)
        wmask = kmask & in_window
        off = tl.maximum(prefix - W + src_col, 0)
        win_full = tl.load(R2T + rp * r2t_stride + off, mask=wmask, other=0).to(
            tl.int64
        )
        win_swa = tl.load(MAP + win_full, mask=wmask, other=-1).to(tl.int32)
        in_block = (k >= cl) & (k < cl + QO)
        bmask = kmask & in_block
        bcol = tl.maximum(k - cl, 0)
        blk_full = tl.load(LOCB + bcol, mask=bmask & real, other=0).to(tl.int64)
        blk_swa = tl.load(MAP + blk_full, mask=bmask, other=-1).to(tl.int32)
        val = tl.where(in_window, win_swa, tl.where(in_block, blk_swa, -1))
        tl.store(SWA + q * TW + k, val.to(tl.int32), mask=kmask)
        tl.store(L32 + 2 * n + q, (cl + QO).to(tl.int32))
        if HAS_SLOTS:
            loc = tl.load(LOCB + q % QO, mask=real, other=0).to(tl.int64)
            sl = tl.load(MAP + tl.where(loc < 0, loc + map_n, loc))
            tl.store(L32 + 3 * n + q, sl.to(tl.int32))


@triton.jit
def _block_meta_kernel(
    SEQ,
    REQ,
    LOC,
    R2T,
    MAP,
    VALID,
    L32,
    SWA,
    PT,
    n,
    r2t_stride,
    num_cols,
    map_n,
    QO: tl.constexpr,
    W: tl.constexpr,
    TW: tl.constexpr,
    TW_BLOCK: tl.constexpr,
    PAGE: tl.constexpr,
    HAS_VALID: tl.constexpr,
    HAS_SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0)
    blk = tl.program_id(1)
    i = q // QO
    _block_meta_row(
        q,
        blk,
        tl.load(REQ + i),
        tl.load(SEQ + i),
        LOC + i * QO,
        True,
        R2T,
        MAP,
        VALID,
        L32,
        SWA,
        PT,
        n,
        r2t_stride,
        num_cols,
        map_n,
        QO,
        W,
        TW,
        TW_BLOCK,
        PAGE,
        HAS_VALID,
        HAS_SLOTS,
        BLOCK,
    )


# Tensors of the builds made inside a forward graph, kept per shape; and the builds the
# window launch runs, by kind and captured bucket: ("verify" | "draft", bucket) -> (launch
# spec, the pad rows' length).
_KEPT: dict = {}
_FOLDED: dict = {}
_VERIFY_ARGS = (
    "R2T MAP L32 L64 NEG SWA PT PT1 PT2 C128_PI rows r2t_stride map_n num_cols topk "
    "QO W PAGE C128P W128 BPP1 BPP2 NEG_W BLOCK"
).split()
_DRAFT_ARGS = (
    "R2T MAP VALID L32 SWA PT n r2t_stride num_cols map_n "
    "QO W TW TW_BLOCK PAGE HAS_VALID HAS_SLOTS BLOCK"
).split()


def _kept(key, make) -> dict:
    """The tensors a build inside a forward graph writes: one set per shape, so every
    graph captured for it reads what the step's window launch wrote."""
    out = _KEPT.get(key)
    if out is None:
        out = _KEPT[key] = make()
    return out


def _fold(key, spec, seq_pad) -> bool:
    """True when the build is being captured into a forward graph: it launches nothing,
    and every later window launch of a batch the graph runners pad to its bucket runs it
    (``_window_launch``), its pad rows of length ``seq_pad`` (the runners' fill length)."""
    if (
        _sgl_window is None
        or _IN_GRAPH.get() is None
        or not torch.cuda.is_current_stream_capturing()
    ):
        return False
    kind, rows = key[0], key[1]
    bs = rows // spec["QO"]
    held = _FOLDED.get((kind, bs))
    if held is None:
        _FOLDED[(kind, bs)] = (spec, seq_pad)
    elif held[1] != seq_pad or any(
        held[0][k] is not spec[k]
        if isinstance(spec[k], torch.Tensor)
        else held[0][k] != spec[k]
        for k in spec
    ):
        raise RuntimeError(
            f"DSV4.1 metadata fold: two {kind} builds of batch {bs} differ"
        )
    chose(f"{kind}_meta_in_window", True)
    return True


@triton.jit
def _window_row(
    b,
    BS: tl.constexpr,
    PADDED: tl.constexpr,
    req_pool_indices,
    req_to_token,
    start_offset,
    pool_len,
    seq_pad,
    START_BY_ROW: tl.constexpr,
):
    """Request row ``b`` of a bucket: (request, prefix length, write locations, whether it
    is one of the step's BS rows). A step's row reads the window's inputs; a pad row (only
    where PADDED: the bucket is larger than BS) has request 0 and length ``seq_pad`` (its
    locations read as 0), as the graph runners pad their buffers."""
    if PADDED:
        real = b < BS
        req = tl.load(req_pool_indices + b, mask=real, other=0)
        if START_BY_ROW:
            seq = tl.load(start_offset + req, mask=real, other=0)
        else:
            seq = tl.load(start_offset + b, mask=real, other=0)
        seq = tl.where(real, seq, seq_pad)
    else:
        real = True
        req = tl.load(req_pool_indices + b)
        if START_BY_ROW:
            seq = tl.load(start_offset + req)
        else:
            seq = tl.load(start_offset + b)
    locb = req_to_token + req.to(tl.int64) * pool_len + seq
    return req, seq, locb, real


@triton.jit
def _window_meta_kernel(
    req_pool_indices,
    req_to_token,
    start_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    draft_token_num: tl.constexpr,
    full_v2p=None,
    swa_v2p=None,
    full_out=None,
    swa_out=None,
    PAGE_SIZE: tl.constexpr = 1,
    num_v_pages=0,
    num_swa_pages=0,
    positions_out=None,
    anchor_src=None,
    anchor_out=None,
    anchor_stride=0,
    START_BY_ROW: tl.constexpr = False,
    seq_lens_out=None,
    draft_positions_out=None,
    draft_loc_out=None,
    DRAFT_WIDTH: tl.constexpr = 0,
    loc32_out=None,
    draft_loc32_out=None,
    sim_correct_out=None,
    sim_seed_out=None,
    sim_counter=None,
    sim_seed=0,
    sim_acc_len=0.0,
    SIM_METHOD: tl.constexpr = 0,
    SIM_MAX: tl.constexpr = 1,
    pub_tokens=None,
    pub_accept=None,
    ring=None,
    ring_counter=None,
    PUB_BS: tl.constexpr = 0,
    PUB_N: tl.constexpr = 0,
    PUB_BS_P2: tl.constexpr = 1,
    PUB_N_P2: tl.constexpr = 1,
    RING_ROWS: tl.constexpr = 1,
    # the draft block's build (D_) and the verify's (V_), as _block_meta_kernel's and
    # _verify_meta_kernel's arguments
    D_MAP=None,
    D_VALID=None,
    D_L32=None,
    D_SWA=None,
    D_PT=None,
    d_n=0,
    d_num_cols=0,
    d_map_n=0,
    D_W: tl.constexpr = 1,
    D_TW: tl.constexpr = 1,
    D_TW_BLOCK: tl.constexpr = 1,
    D_PAGE: tl.constexpr = 1,
    D_HAS_VALID: tl.constexpr = False,
    D_HAS_SLOTS: tl.constexpr = False,
    D_BLOCK: tl.constexpr = 1,
    D_QO: tl.constexpr = 1,
    D_NB: tl.constexpr = 0,
    V_MAP=None,
    V_L32=None,
    V_L64=None,
    V_NEG=None,
    V_SWA=None,
    V_PT=None,
    V_PT1=None,
    V_PT2=None,
    V_C128_PI=None,
    v_rows=0,
    v_map_n=0,
    v_num_cols=0,
    v_topk=0,
    V_W: tl.constexpr = 1,
    V_PAGE: tl.constexpr = 1,
    V_C128P: tl.constexpr = 1,
    V_W128: tl.constexpr = 1,
    V_BPP1: tl.constexpr = 1,
    V_BPP2: tl.constexpr = 1,
    V_NEG_W: tl.constexpr = 1,
    V_BLOCK: tl.constexpr = 1,
    V_NB: tl.constexpr = 0,
    BS: tl.constexpr = 1,
    D_BS: tl.constexpr = 0,
    V_BS: tl.constexpr = 0,
    d_seq_pad=0,
    v_seq_pad=0,
):
    """SGLang's verify window launch (programs 0 .. BS - 1, its own body) and the draft
    block metadata (D_BS * D_QO rows) and verify metadata (V_BS * draft_token_num rows) of
    the buckets the step's graphs replay, D_NB / V_NB page-table blocks a row. A row of the
    step's BS requests is built from the window's inputs: a request's prefix length is its
    window start, its write locations ``req_to_token[row, start:]``, the block's the first
    D_QO of them. A bucket's pad row has the graph runners' pad inputs: request 0, length
    d_seq_pad / v_seq_pad, write locations 0."""
    pid = tl.program_id(0)
    if pid < BS:
        _sgl_window(
            req_pool_indices,
            req_to_token,
            start_offset,
            out_cache_loc,
            pool_len,
            draft_token_num,
            full_v2p,
            swa_v2p,
            full_out,
            swa_out,
            PAGE_SIZE,
            num_v_pages,
            num_swa_pages,
            positions_out,
            anchor_src,
            anchor_out,
            anchor_stride,
            START_BY_ROW,
            seq_lens_out,
            draft_positions_out,
            draft_loc_out,
            DRAFT_WIDTH,
            loc32_out,
            draft_loc32_out,
            sim_correct_out,
            sim_seed_out,
            sim_counter,
            sim_seed,
            sim_acc_len,
            SIM_METHOD,
            SIM_MAX,
            pub_tokens,
            pub_accept,
            ring,
            ring_counter,
            PUB_BS,
            PUB_N,
            PUB_BS_P2,
            PUB_N_P2,
            RING_ROWS,
        )
    else:
        p = pid - BS
        nd = D_BS * D_QO * D_NB
        if p < nd:
            if D_NB > 0:
                q = p // D_NB
                b = q // D_QO
                req, seq, locb, real = _window_row(
                    b,
                    BS,
                    D_BS > BS,
                    req_pool_indices,
                    req_to_token,
                    start_offset,
                    pool_len,
                    d_seq_pad,
                    START_BY_ROW,
                )
                _block_meta_row(
                    q,
                    p % D_NB,
                    req,
                    seq,
                    locb,
                    real,
                    req_to_token,
                    D_MAP,
                    D_VALID,
                    D_L32,
                    D_SWA,
                    D_PT,
                    d_n,
                    pool_len,
                    d_num_cols,
                    d_map_n,
                    D_QO,
                    D_W,
                    D_TW,
                    D_TW_BLOCK,
                    D_PAGE,
                    D_HAS_VALID,
                    D_HAS_SLOTS,
                    D_BLOCK,
                )
        else:
            if V_NB > 0:
                p = p - nd
                row = p // V_NB
                b = row // draft_token_num
                req, seq, locb, real = _window_row(
                    b,
                    BS,
                    V_BS > BS,
                    req_pool_indices,
                    req_to_token,
                    start_offset,
                    pool_len,
                    v_seq_pad,
                    START_BY_ROW,
                )
                _verify_meta_row(
                    row,
                    p % V_NB,
                    req,
                    seq,
                    locb,
                    real,
                    req_to_token,
                    V_MAP,
                    V_L32,
                    V_L64,
                    V_NEG,
                    V_SWA,
                    V_PT,
                    V_PT1,
                    V_PT2,
                    V_C128_PI,
                    v_rows,
                    pool_len,
                    v_map_n,
                    v_num_cols,
                    v_topk,
                    draft_token_num,
                    V_W,
                    V_PAGE,
                    V_C128P,
                    V_W128,
                    V_BPP1,
                    V_BPP2,
                    V_NEG_W,
                    V_BLOCK,
                )


_sgl_window = (
    None  # SGLang's window kernel, which _window_meta_kernel runs as its first programs
)
_WINDOW_ARGS = (
    "req_pool_indices req_to_token start_offset out_cache_loc pool_len draft_token_num "
    "full_v2p swa_v2p full_out swa_out PAGE_SIZE num_v_pages num_swa_pages positions_out "
    "anchor_src anchor_out anchor_stride START_BY_ROW seq_lens_out draft_positions_out "
    "draft_loc_out DRAFT_WIDTH loc32_out draft_loc32_out sim_correct_out sim_seed_out "
    "sim_counter sim_seed sim_acc_len SIM_METHOD SIM_MAX pub_tokens pub_accept ring "
    "ring_counter PUB_BS PUB_N PUB_BS_P2 PUB_N_P2 RING_ROWS"
).split()


class _WindowLaunch:
    """Stands in for SGLang's window kernel (``kernel[grid](...)``): a launch whose rows
    have folded metadata builds runs them as more programs of the same launch."""

    def __init__(self, kernel) -> None:
        self.kernel = kernel

    def __getitem__(self, grid):
        return lambda *args, **kwargs: _window_launch(self.kernel, grid, args, kwargs)


def _bucket(kind, bs):
    """The folded build of the bucket the graph runners pad batch ``bs`` to (the smallest
    captured one at least ``bs``), as (spec, pad length), or (None, 0)."""
    sizes = [n for k, n in _FOLDED if k == kind and n >= bs]
    return _FOLDED[(kind, min(sizes))] if sizes else (None, 0)


def _window_launch(kernel, grid, args, kwargs):
    bs = grid[0]
    width = args[5] if len(args) > 5 else kwargs["draft_token_num"]
    (d, d_pad), (v, v_pad) = _bucket("draft", bs), _bucket("verify", bs)
    if d is None and v is None:
        return kernel[grid](*args, **kwargs)
    r2t = args[1] if len(args) > 1 else kwargs["req_to_token"]
    for kind, spec in (("draft", d), ("verify", v)):
        if spec is None:
            continue
        if (
            spec["R2T"] is not r2t
            or spec["QO"] > width
            or (kind == "verify" and spec["QO"] != width)
            or spec["r2t_stride"] != r2t.shape[1]
            or r2t.stride(1) != 1
        ):
            raise RuntimeError(
                f"DSV4.1 metadata fold: the {kind} build of batch {bs} does not match "
                "this window launch (req_to_token, width)"
            )
    meta = dict(BS=bs)
    n_prog = bs
    if d is not None:
        meta.update(
            {
                ("D_" if k.isupper() else "d_") + k: d[k]
                for k in _DRAFT_ARGS
                if k not in ("R2T", "r2t_stride", "QO", "n", "num_cols", "map_n")
            }
        )
        meta.update(
            d_n=d["n"],
            d_num_cols=d["num_cols"],
            d_map_n=d["map_n"],
            D_QO=d["QO"],
            D_NB=max(1, triton.cdiv(d["num_cols"], d["BLOCK"])),
            D_BS=d["n"] // d["QO"],
            d_seq_pad=d_pad,
        )
        n_prog += d["n"] * meta["D_NB"]
    if v is not None:
        meta.update(
            {
                ("V_" if k.isupper() else "v_") + k: v[k]
                for k in _VERIFY_ARGS
                if k
                not in ("R2T", "r2t_stride", "QO", "rows", "map_n", "num_cols", "topk")
            }
        )
        meta.update(
            v_rows=v["rows"],
            v_map_n=v["map_n"],
            v_num_cols=v["num_cols"],
            v_topk=v["topk"],
            V_NB=max(1, triton.cdiv(v["num_cols"], v["BLOCK"])),
            V_BS=v["rows"] // v["QO"],
            v_seq_pad=v_pad,
        )
        n_prog += v["rows"] * meta["V_NB"]
    return _window_meta_kernel[(n_prog,)](*args, **kwargs, **meta, num_warps=4)


def _draft_inputs_ok(
    self, *, max_seq_len, req_pool_indices, seq_lens, out_cache_loc, block_size
):
    pool = self.token_to_kv_pool
    mapping = getattr(pool, "full_to_swa_index_mapping", None)
    return bool(
        pool.request_window is None
        and getattr(pool, "swa_req_ring_size", None) is None
        and self._uses_dspark_draft_window()
        and block_size == self.target_verify_num_draft_tokens
        and mapping is not None
        and mapping.dim() == 1
        and mapping.is_contiguous()
        and self.req_to_token.stride(1) == 1
        and seq_lens.is_cuda
        and seq_lens.dim() == 1
        and seq_lens.is_contiguous()
        and req_pool_indices.shape == seq_lens.shape
        and req_pool_indices.is_contiguous()
        and out_cache_loc.dim() == 1
        and out_cache_loc.is_contiguous()
        and out_cache_loc.shape[0] >= block_size * seq_lens.shape[0]
    )


def _draft_fused(
    self, *, max_seq_len, req_pool_indices, seq_lens, out_cache_loc, block_size
):
    hip = sys.modules[HIP]

    pool, r2t, page = self.token_to_kv_pool, self.req_to_token, self.page_size
    mapping = pool.full_to_swa_index_mapping
    valid = getattr(pool, "swa_valid_from", None)
    n, dev, w = seq_lens.shape[0] * block_size, seq_lens.device, hip.SWA_WINDOW
    align = hip.PAGE_INDEX_ALIGNED_SIZE
    tw = -(-(w + block_size) // align) * align
    num_cols = max(0, (min(max_seq_len + block_size, r2t.shape[1]) + page - 1) // page)
    i32 = dict(dtype=torch.int32, device=dev)

    def make():
        return dict(
            L32=torch.empty((4, n), **i32),
            SWA=torch.empty((n, tw), **i32),
            PT=torch.empty((n, num_cols), **i32),
        )

    key = ("draft", n, num_cols, tw, dev)
    out = _kept(key, make) if _IN_GRAPH.get() is not None else make()
    lens, swa, pt = out["L32"], out["SWA"], out["PT"]
    casual, pos, swa_len, swa_loc = lens
    slots = out_cache_loc.shape[0] == n
    if n:
        block = min(1024, triton.next_power_of_2(max(num_cols, 1)))
        spec = dict(
            out,
            R2T=r2t,
            MAP=mapping,
            VALID=valid if valid is not None else seq_lens,
            n=n,
            r2t_stride=r2t.stride(0),
            num_cols=num_cols,
            map_n=mapping.shape[0],
            QO=block_size,
            W=w,
            TW=tw,
            TW_BLOCK=triton.next_power_of_2(tw),
            PAGE=page,
            HAS_VALID=valid is not None,
            HAS_SLOTS=slots,
            BLOCK=block,
        )
        if not _fold(key, spec, self.get_cuda_graph_seq_len_fill_value()):
            _block_meta_kernel[(n, max(1, triton.cdiv(num_cols, block)))](
                seq_lens, req_pool_indices, out_cache_loc, **spec
            )
    core = hip.DSV4AttnMetadata(
        page_size=page,
        raw_out_loc=out_cache_loc,
        seq_lens_casual=casual,
        cuda_int32_kwargs=self.cuda_int32_kwargs,
        positions_casual=pos,
        page_table=pt,
        swa_page_indices=swa,
        swa_topk_lengths=swa_len,
        index_topk=self.index_topk,
        present_ratios=self.present_ratios,
        low_ratios=self.low_ratios,
        request_window_layout=None,
    )
    for name in (
        "c4_sparse_topk_lengths",
        "c4_sparse_topk_lengths_raw",
        "c4_sparse_page_indices",
        "c4_sparse_raw_indices",
        "c4_flashmla_metadata",
        "c128_flashmla_metadata",
    ):
        setattr(core, name, None)
    core.c0_flashmla_metadata = hip._create_flashmla_metadata()
    if slots:
        core.__dict__["_mono_slots"] = (out_cache_loc, swa_loc, None, None)
    return hip.DSV4Metadata(
        core, None, c4_compress_metadata=None, c128_compress_metadata=None
    )


def _draft_block_metadata(orig, self, *args, **kwargs):
    if not chose("draft_block_meta", not args and _draft_inputs_ok(self, **kwargs)):
        return orig(self, *args, **kwargs)
    return _draft_fused(self, **kwargs)


def install() -> None:
    if not flag(ENV):
        return
    import importlib

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    hip = importlib.import_module(HIP)
    cls = hip.DeepseekV4HipRadixBackend
    route(cls, "make_forward_metadata_from_raw_verify", _verify_metadata)
    route(cls, "init_forward_metadata_in_graph", _in_graph)
    route(cls, "init_forward_metadata_dspark_draft_block", _draft_block_metadata)
    route(DeepSeekV4TokenToKVPool, "translate_loc_from_full_to_swa", _translate)
    if hasattr(cls, "_swa_loc_i32"):  # SGLang with the one-launch SWA lookup
        route(cls, "_swa_loc_i32", _swa_loc_i32)
    route(hip, "token_req_indices", _token_req_indices)
    take_window()


def take_window() -> None:
    """SGLang's window kernel becomes ``_WindowLaunch`` (once), where its arguments are the
    ones ``_window_meta_kernel`` passes on; else the builds stay in their graphs."""
    global _sgl_window
    from sglang.kernels.ops.speculative import cache_locs

    kernel = cache_locs.assign_extend_cache_locs_uniform
    if _sgl_window is not None or isinstance(kernel, _WindowLaunch):
        return
    if list(getattr(kernel, "arg_names", ())) == _WINDOW_ARGS:
        _sgl_window = kernel
        cache_locs.assign_extend_cache_locs_uniform = _WindowLaunch(kernel)
    else:
        chose("verify_window_meta", False, "the window kernel's arguments differ")
