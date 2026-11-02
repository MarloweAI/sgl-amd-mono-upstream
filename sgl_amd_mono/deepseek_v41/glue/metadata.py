# SPDX-License-Identifier: Apache-2.0
"""A DSpark verify step's attention metadata (``make_forward_metadata_from_raw_verify``,
~40 launches) and a draft block's (~12), with their SWA write slots, in one launch each,
every tensor as SGLang writes it. ``SGLANG_AMD_MONO_DSV41_METADATA=0`` keeps SGLang's."""

from __future__ import annotations

import contextvars
import dataclasses
import sys

import torch
import triton
import triton.language as tl

from sgl_amd_mono.deepseek_v41.glue import agreed, flag, route

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
        "low_ratios",
        "index_topk",
        "page_size",
        "req_to_token",
        "token_to_kv_pool",
        "cuda_int32_kwargs",
        "forward_metadata",
    ),
    "srt/arg_groups/fields/exec_.py": ("dsa_topk_backend",),
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
# scratch the top-k planner leaves unwritten: dtype and shape only
_UNWRITTEN = ("topk_metadata", "topk_metadata_chunks")


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
    req = tl.load(REQ + b).to(tl.int64)
    casual = (tl.load(SEQ + b) + QO + (row % QO - QO + 1)).to(tl.int32)
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
        loc = tl.load(LOC + row).to(tl.int64)
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
    l32 = torch.empty((len(_I32), rows), **i32)
    l64 = torch.empty((len(_I64), rows), **i64)
    neg = torch.empty((len(_NEG), rows, neg_w), **i32)
    pt = torch.empty((rows, num_cols), **i32)
    pt1 = torch.empty((rows, num_cols * bpp1), **i32) if bpp1 > 1 else pt
    pt2 = torch.empty((rows, num_cols * bpp2), **i32) if bpp2 > 1 else pt
    swa = torch.empty((rows, w), **i32)
    c128_pi = torch.empty((rows, w128), **i32)
    t = dict(zip(_I32, l32)) | dict(zip(_I64, l64)) | dict(zip(_NEG, neg))
    if rows:
        block = min(1024, triton.next_power_of_2(max(num_cols, 1)))
        grid = (rows, max(1, triton.cdiv(num_cols, block)))
        _verify_meta_kernel[grid](
            raw.seq_lens,
            raw.req_pool_indices,
            raw.out_cache_loc,
            r2t,
            mapping,
            l32,
            l64,
            neg,
            swa,
            pt,
            pt1,
            pt2,
            c128_pi,
            rows,
            r2t.stride(0),
            mapping.shape[0],
            num_cols,
            backend.index_topk,
            QO=qo,
            W=w,
            PAGE=page,
            C128P=c128p,
            W128=w128,
            BPP1=bpp1,
            BPP2=bpp2,
            NEG_W=neg_w,
            BLOCK=block,
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
        low_ratios=backend.low_ratios,
        request_window_layout=None,
    )
    for name in _I32[3:13] + _I64[:4] + _NEG:
        setattr(core, name, t[name])
    core.c128_page_indices = c128_pi
    for r in (0, 1, 2, 4, 128):
        setattr(core, f"c{r}_flashmla_metadata", hip._create_flashmla_metadata())
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


def disagrees(want, got, path="metadata") -> str | None:
    """The first field where ``got`` differs from ``want`` (tensors by dtype, shape and
    value; dataclasses, dicts and sequences by their members), or None."""
    if isinstance(want, torch.Tensor) or isinstance(got, torch.Tensor):
        if not (isinstance(want, torch.Tensor) and isinstance(got, torch.Tensor)):
            return f"{path}: {type(got).__name__} vs SGLang's {type(want).__name__}"
        if want.dtype != got.dtype or want.shape != got.shape:
            return f"{path}: {got.dtype}{tuple(got.shape)} vs {want.dtype}{tuple(want.shape)}"
        unwritten = any(f".{n}" in path for n in _UNWRITTEN)
        return None if unwritten or torch.equal(want, got) else f"{path}: values differ"
    if type(want) is not type(got):
        return f"{path}: {type(got).__name__} vs SGLang's {type(want).__name__}"
    if dataclasses.is_dataclass(want):
        pairs = [
            (f.name, getattr(want, f.name), getattr(got, f.name))
            for f in dataclasses.fields(want)
        ]
    elif isinstance(want, dict):
        if want.keys() != got.keys():
            return f"{path}: keys differ"
        pairs = [(f"[{k!r}]", want[k], got[k]) for k in want]
    elif isinstance(want, (list, tuple)):
        if len(want) != len(got):
            return f"{path}: {len(got)} items vs {len(want)}"
        pairs = [(f"[{i}]", a, b) for i, (a, b) in enumerate(zip(want, got))]
    else:
        same = want == got or not isinstance(want, (int, float, str, bool))
        return None if same else f"{path}: {got!r} vs SGLang's {want!r}"
    for name, a, b in pairs:
        sub = f"{path}.{name}" if not name.startswith("[") else path + name
        why = disagrees(a, b, sub)
        if why:
            return why
    return None


def _verify_metadata(orig, self, raw_metadata, *args, **kwargs):
    inputs = None if args or kwargs else _inputs(self, raw_metadata)
    if inputs is not None and agreed(
        "verify_meta",
        lambda: disagrees(orig(self, raw_metadata), _fused(self, raw_metadata, inputs)),
    ):
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
    if got is not None and agreed(
        "swa_slots",
        lambda: disagrees(orig(self, kv_indices).to(torch.int32), got),
    ):
        return got
    return orig(self, kv_indices, *args, **kwargs)


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
    if got is not None and agreed(
        "verify_req_rows", lambda: disagrees(orig(forward_batch), got)
    ):
        return got
    return orig(forward_batch, *args, **kwargs)


@triton.jit
def _block_meta_kernel(
    SEQ,
    REQ,
    R2T,
    MAP,
    LOC,
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
    req = tl.load(REQ + i)
    rp = req.to(tl.int64)
    col = blk * BLOCK + tl.arange(0, BLOCK)
    cm = col < num_cols
    v = tl.load(R2T + rp * r2t_stride + col.to(tl.int64) * PAGE, mask=cm, other=0)
    pq = v // PAGE
    pq = tl.where((v < 0) & (v % PAGE != 0), pq - 1, pq)
    tl.store(PT + q.to(tl.int64) * num_cols + col, pq.to(tl.int32), mask=cm)
    if blk == 0:
        seq = tl.load(SEQ + i)
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
        blk_full = tl.load(LOC + i * QO + bcol, mask=bmask, other=0).to(tl.int64)
        blk_swa = tl.load(MAP + blk_full, mask=bmask, other=-1).to(tl.int32)
        val = tl.where(in_window, win_swa, tl.where(in_block, blk_swa, -1))
        tl.store(SWA + q * TW + k, val.to(tl.int32), mask=kmask)
        tl.store(L32 + 2 * n + q, (cl + QO).to(tl.int32))
        if HAS_SLOTS:
            loc = tl.load(LOC + q).to(tl.int64)
            sl = tl.load(MAP + tl.where(loc < 0, loc + map_n, loc))
            tl.store(L32 + 3 * n + q, sl.to(tl.int32))


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
    lens = torch.empty((4, n), **i32)
    casual, pos, swa_len, swa_loc = lens
    swa = torch.empty((n, tw), **i32)
    pt = torch.empty((n, num_cols), **i32)
    slots = out_cache_loc.shape[0] == n
    if n:
        block = min(1024, triton.next_power_of_2(max(num_cols, 1)))
        _block_meta_kernel[(n, max(1, triton.cdiv(num_cols, block)))](
            seq_lens,
            req_pool_indices,
            r2t,
            mapping,
            out_cache_loc,
            valid if valid is not None else seq_lens,
            lens,
            swa,
            pt,
            n,
            r2t.stride(0),
            num_cols,
            mapping.shape[0],
            QO=block_size,
            W=w,
            TW=tw,
            TW_BLOCK=triton.next_power_of_2(tw),
            PAGE=page,
            HAS_VALID=valid is not None,
            HAS_SLOTS=slots,
            BLOCK=block,
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
    if args or not _draft_inputs_ok(self, **kwargs):
        return orig(self, *args, **kwargs)
    if agreed(
        "draft_block_meta",
        lambda: disagrees(orig(self, **kwargs), _draft_fused(self, **kwargs)),
    ):
        return _draft_fused(self, **kwargs)
    return orig(self, **kwargs)


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
    route(hip, "token_req_indices", _token_req_indices)
