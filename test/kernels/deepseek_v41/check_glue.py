"""Check the DSV4.1 glue's launches against SGLang's on one GPU: every field of a verify step's
and a draft block's metadata, their SWA slots and request rows on random tables, lengths and
slots, the inject's cache bytes (V4, V4.1), the draft head's FP8 row quant, the fused mHC posts
(the head's combine and the Engram gate) and the prefill MQA logits with the block-table stride
an argument, each bitwise."""

import argparse
import dataclasses
import sys
from types import SimpleNamespace

import torch

# scratch the top-k planner leaves unwritten: dtype and shape only
_UNWRITTEN = ("topk_metadata", "topk_metadata_chunks")


def differs(want, got, path="metadata") -> str | None:
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
        why = differs(a, b, sub)
        if why:
            return why
    return None


def _bits_differ(want: torch.Tensor, got: torch.Tensor) -> int:
    """Elements whose bits differ (dtype and shape must match)."""
    if want.dtype != got.dtype or want.shape != got.shape:
        return max(want.numel(), got.numel(), 1)
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    dt = view[want.element_size()]
    return int((want.contiguous().view(dt) != got.contiguous().view(dt)).sum())


def _backend(rng, *, reqs, ctx, qo, draft, device):
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
    )
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    slots = 1 << 22
    pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    pool.full_to_swa_index_mapping = torch.randint(
        0, 1 << 16, (slots + 1,), generator=rng, device=device
    )
    # the allocator's trailing -1: an empty slot translates to -1
    pool.full_to_swa_index_mapping[-1] = -1
    pool.request_window = None
    pool.swa_req_ring_size = None
    pool.unified_swa_pages = 0
    pool.swa_valid_from = (
        torch.randint(0, 64, (reqs,), generator=rng, device=device) if draft else None
    )
    pool.get_index_k_page_size = lambda ratio: 64
    b = DeepseekV4HipRadixBackend.__new__(DeepseekV4HipRadixBackend)
    b.req_to_token = torch.randint(
        0, slots, (reqs, ctx), generator=rng, device=device, dtype=torch.int32
    )
    b.token_to_kv_pool = pool
    b.page_size, b.MAX_SEQ_LEN_FOR_CAPTURE = 256, ctx
    b.target_verify_num_draft_tokens = qo
    b.need_compress, b.has_c4, b.has_c128 = not draft, False, False
    b.present_ratios = b.low_ratios = () if draft else (1, 2)
    b.index_topk = 512
    b.cuda_int32_kwargs = dict(device=device, dtype=torch.int32)
    b.encoder_replay = False
    b.is_dspark_draft, b.is_dsv41 = draft, True

    class _TopK:
        def should_use_topk_v2(self):
            return False

    b.dsa_topk_backend = _TopK()
    return b


def _step(rng, b, bs, device, long_ctx):
    reqs, ctx = b.req_to_token.shape
    qo = b.target_verify_num_draft_tokens
    top = ctx - 2 * qo if long_ctx else 4096
    seq = torch.randint(1, top, (bs,), generator=rng, device=device)
    seq[0] = 1 if bs > 1 else seq[0]  # a padded row's length
    req = torch.randperm(reqs, generator=rng, device=device)[:bs]
    loc = torch.randint(0, 1 << 22, (bs * qo,), generator=rng, device=device)
    return seq, req, loc


def _check_inject(device) -> int:
    """glue/inject's one-launch store vs SGLang's per-stage store, V4 and V4.1 pools."""
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

    from sgl_amd_mono.deepseek_v41.glue import inject

    bad = 0
    g = torch.Generator(device=device).manual_seed(3)
    for layout in (KVLayout.V4, KVLayout.V41):
        st = inject._Stages.__new__(inject._Stages)
        st.weights = [
            torch.randn(512, generator=g, device=device).bfloat16() for _ in range(3)
        ]
        st.freqs_cis = [
            torch.polar(
                torch.ones(4096, 32, device=device),
                torch.rand(4096, 32, generator=g, device=device) * 6,
            )
            for _ in range(3)
        ]
        st.freqs = [
            torch.view_as_real(f).flatten(-2).contiguous() for f in st.freqs_cis
        ]
        st.layers, st.eps, st.v41 = [0, 1, 2], 1e-6, layout.value == "v41"
        buf = torch.zeros(4, layout.page_bytes(256), dtype=torch.uint8, device=device)
        pool = SimpleNamespace(
            kv_layout=layout, swa_page_size=256, get_swa_raw_buffer=lambda i: buf
        )
        why = _inject_differs(inject, st, pool)
        print(f"inject {layout.value}: {why or 'ok'}")
        bad += why is not None
    return bad


def _inject_differs(inject, st, pool) -> str | None:
    """SGLang's per-stage store vs the one launch, on random rows into scratch pages."""
    from sglang.kernels.ops.attention.dsv4.norm_rope import fused_k_norm_rope_flashmla

    dev = st.weights[0].device
    g = torch.Generator(device=dev).manual_seed(11)
    rows, width = 12, 6
    page_bytes = pool.get_swa_raw_buffer(st.layers[0]).stride(0)
    pages, ps = 3, pool.swa_page_size
    n = len(st.layers)
    kv = torch.randn(rows, n * 512, generator=g, device=dev).mul_(3).bfloat16()
    kvs = [kv[:, s * 512 : (s + 1) * 512] for s in range(n)]
    pos = torch.randint(
        0, min(f.shape[0] for f in st.freqs), (rows,), generator=g, device=dev
    )
    full = torch.randperm(pages * ps, generator=g, device=dev)[:rows]
    mapping = torch.arange(pages * ps, device=dev, dtype=torch.int64)
    commit = torch.tensor([4, 6], dtype=torch.int32, device=dev)
    committed = inject._Committed(full, mapping, commit, width)
    slots = committed.slots()
    want = [
        torch.zeros(pages, page_bytes, dtype=torch.uint8, device=dev) for _ in range(n)
    ]
    got = [torch.zeros_like(w) for w in want]
    for s in range(n):
        fused_k_norm_rope_flashmla(
            kv=kvs[s].contiguous(),
            kv_weight=st.weights[s],
            eps=st.eps,
            freqs_cis=st.freqs_cis[s],
            positions=pos,
            out_loc=slots,
            kvcache=want[s],
            page_size=ps,
            layout=pool.kv_layout,
        )
    inject._store(st, kvs, pos, got, ps, committed=committed)
    for s in range(n):
        if not torch.equal(want[s], got[s]):
            return f"stage {s}: {(want[s] != got[s]).sum().item()} bytes differ"
    return None


def _check_draft_head(device) -> int:
    """glue/draft_head's row quant vs SGLang's _fp8_quant_per_row, bitwise."""
    from sglang.srt.models.dspark import _fp8_quant_per_row

    from sgl_amd_mono.deepseek_v41.glue import draft_head

    g = torch.Generator(device=device).manual_seed(9)
    bad = 0
    for m in (5, 10, 40, 48):
        x = (torch.randn(m, 5120, generator=g, device=device) * 4).bfloat16()
        x[0, :7] = 0
        want_q, want_s = _fp8_quant_per_row(x)
        got_q, got_s = draft_head.quant_rows(x)
        why = None
        if not torch.equal(want_s, got_s):
            why = "scales differ"
        elif n := _bits_differ(want_q, got_q):
            why = f"{n} codes differ"
        print(f"draft head quant {m} rows: {why or 'ok'}")
        bad += why is not None
    return bad


def _check_posts(device) -> int:
    """fused's one-launch posts vs SGLang's mHC post, then its mhc_combine (the head) or
    its Engram gate (image-token rows too), bitwise."""
    import sglang.kernels.ops.layernorm.mhc as mhc_ops
    from sglang.kernels.ops.layernorm.mhc import mhc_combine
    from sglang.srt.layers.engram import engram_gate
    from sglang.srt.models import deepseek_v4_mhc as mhc

    from sgl_amd_mono.deepseek_v41 import HF_CONFIG, fused

    # a server without prefill context parallelism (this process publishes no parallel config)
    mhc_ops.is_dsa_prefill_cp_interleave = lambda: False

    hc, h = HF_CONFIG["hc_mult"], HF_CONFIG["hidden_size"]
    cfg = mhc.HcConfig(
        mult=hc,
        sinkhorn_iters=HF_CONFIG["hc_sinkhorn_iters"],
        eps=HF_CONFIG["hc_eps"],
        rms_eps=HF_CONFIG["rms_norm_eps"],
        hidden=h,
        pre_from_prev=HF_CONFIG["hc_pre_from_prev_sublayer"],
        cp_prefill=False,
    )
    g = torch.Generator(device=device).manual_seed(5)
    bf = torch.bfloat16

    def rand(*shape, dtype=torch.float32, normal=False, scale=1.0):
        f = torch.randn if normal else torch.rand
        return (f(shape, generator=g, device=device) * scale).to(dtype)

    engram = SimpleNamespace(
        q_weight=rand(hc, h, dtype=bf, normal=True),
        k_weight=rand(hc, h, dtype=bf, normal=True),
        eps=HF_CONFIG["rms_norm_eps"],
        clamp_value=1e-6,
        image_token_id=None,
    )
    bad = 0
    for rows in (1, 4, 6, 12, 48):
        x, residual = (
            rand(rows, h, dtype=bf, normal=True),
            rand(rows, hc, h, dtype=bf, normal=True),
        )
        post, comb = rand(rows, hc) * 2, rand(rows, hc, hc)
        pre, kv = rand(rows, hc), rand(rows, (hc + 1) * h, dtype=bf, normal=True)
        settled = mhc.post(cfg, x, residual, post, comb)
        want = (settled, mhc_combine(settled.flatten(1), pre, hc, settled.dtype))
        got = fused._post_combine(x, residual, post, comb, pre)
        n = sum(_bits_differ(a, b) for a, b in zip(want, got))
        ids = torch.randint(0, 1000, (rows,), generator=g, device=device)
        ids[::2] = 7
        for image in (None, 7):
            engram.image_token_id = image
            want = engram_gate(
                settled, kv, engram.q_weight, engram.k_weight, engram.eps,
                engram.clamp_value, ids if image is not None else None, image,
            )  # fmt: skip
            got = fused._post_gate(x, residual, post, comb, kv, engram, ids)
            n += _bits_differ(want, got)
        print(f"fused posts {rows} rows: {f'{n} elements differ' if n else 'ok'}")
        bad += n > 0
    return bad


def _check_prefill_logits(device) -> int:
    """glue/prefill's MQA logits (the block-table stride an argument) vs aiter's, through
    SGLang's prefill indexer call, on random FP4 planes at several table widths: every
    row's logits a top-k reads (its first c4_seq_lens), bitwise."""
    import aiter.ops.flydsl as flydsl_ops
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        aiter_fp4_paged_mqa_logits,
        pack_fp4_query_flydsl,
    )

    from sgl_amd_mono.deepseek_v41.glue import prefill

    module, why = prefill._strided_module()
    if module is None:
        print(f"prefill logits: {why}")
        return 1
    aiter_fn = flydsl_ops.flydsl_pa_mqa_logits_fp4_prefill
    g = torch.Generator(device=device).manual_seed(13)
    u8 = dict(generator=g, device=device, dtype=torch.uint8)
    bad = 0
    # (requests, rows a request, compressed rows a request sees at its last row)
    for reqs, per_req, ctx in (
        (1, 64, 200),
        (3, 16, 4096),
        (2, 40, 33 * 64 + 5),
        (1, 8, 81920),
    ):
        pages_per_req = -(-ctx // 64)
        pages = reqs * pages_per_req + 3
        payload = torch.randint(0, 256, (pages, 1, 4, 64, 16), **u8)
        kscale = torch.randint(123, 130, (pages, 1, 4, 64), **u8)
        perm = torch.randperm(pages, generator=g, device=device).to(torch.int32)
        table = perm[: reqs * pages_per_req].view(reqs, pages_per_req)
        table = table.repeat_interleave(per_req, 0).contiguous()
        rows = reqs * per_req
        q = torch.randn(rows, 32, 128, generator=g, device=device).bfloat16()
        q_fp4, q_scale = pack_fp4_query_flydsl(q)
        w = (torch.randn(rows, 32, generator=g, device=device) * 0.05).bfloat16()
        bounds = torch.tensor(
            [ctx - (per_req - 1 - i) for _ in range(reqs) for i in range(per_req)],
            dtype=torch.int32,
            device=device,
        )
        args = dict(
            q_fp4=q_fp4, q_scale=q_scale, k_payload=payload, k_scale=kscale, weights=w,
            page_table=table, c4_seq_lens=bounds, weight_scale=0.5, is_decode=False,
        )  # fmt: skip
        try:
            flydsl_ops.flydsl_pa_mqa_logits_fp4_prefill = aiter_fn
            want = aiter_fp4_paged_mqa_logits(**args).clone()
            flydsl_ops.flydsl_pa_mqa_logits_fp4_prefill = (
                module.flydsl_pa_mqa_logits_fp4_prefill
            )
            got = aiter_fp4_paged_mqa_logits(**args).clone()
        finally:
            flydsl_ops.flydsl_pa_mqa_logits_fp4_prefill = aiter_fn
        n = 0
        for r, b in enumerate(bounds.tolist()):
            n += _bits_differ(want[r, :b], got[r, :b])
        print(
            f"prefill logits {reqs} x {per_req} rows, {ctx} keys, table {tuple(table.shape)}: "
            f"{f'{n} differ' if n else 'ok'}"
        )
        bad += n > 0
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    import sglang.srt.layers.attention.deepseek_v4_backend_hip_radix as hip
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
        DSV4RawVerifyMetadata,
    )
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    from sgl_amd_mono.deepseek_v41.glue import metadata

    device = torch.device("cuda", 0)
    rng = torch.Generator(device=device).manual_seed(7)
    cls = DeepseekV4HipRadixBackend
    orig_verify = getattr(
        cls.make_forward_metadata_from_raw_verify,
        "__wrapped__",
        cls.make_forward_metadata_from_raw_verify,
    )
    orig_block = getattr(
        cls.init_forward_metadata_dspark_draft_block,
        "__wrapped__",
        cls.init_forward_metadata_dspark_draft_block,
    )
    translate = DeepSeekV4TokenToKVPool.translate_loc_from_full_to_swa
    orig_translate = getattr(translate, "__wrapped__", translate)
    orig_req_rows = getattr(hip.token_req_indices, "__wrapped__", hip.token_req_indices)
    bad = 0
    for trial in range(a.trials):
        long_ctx = trial % 2 == 1
        ctx = (1 << 20) + 256 * (trial % 3) if long_ctx else 1 << 16
        bs = 1 + trial % 4
        b = _backend(rng, reqs=8, ctx=ctx, qo=6, draft=False, device=device)
        seq, req, loc = _step(rng, b, bs, device, long_ctx)
        raw = DSV4RawVerifyMetadata(
            req_pool_indices=req, seq_lens=seq, out_cache_loc=loc
        )
        inputs = metadata._inputs(b, raw)
        if inputs is None:
            print(f"trial {trial}: verify inputs refused")
            bad += 1
            continue
        got = metadata._fused(b, raw, inputs)
        why = differs(orig_verify(b, raw), got)
        held = got.core_attn_metadata.__dict__["_mono_slots"]
        # what the in-graph build reads in their place: SGLang's translation of the
        # step's out_cache_loc and its request row per token
        want_loc = orig_translate(b.token_to_kv_pool, loc).to(torch.int32)
        why = why or differs(want_loc, held[1], "swa slots")
        fb = SimpleNamespace(
            req_pool_indices=req,
            forward_mode=SimpleNamespace(
                is_decode=lambda: False, is_target_verify=lambda: True
            ),
            spec_info=SimpleNamespace(draft_token_num=6),
            batch_size=bs,
            ragged_verify_layout=None,
        )
        why = why or differs(orig_req_rows(fb), held[2], "request rows")
        d = _backend(rng, reqs=8, ctx=ctx, qo=5, draft=True, device=device)
        seq, req, loc = _step(rng, d, bs, device, long_ctx)
        kw = dict(
            max_seq_len=ctx,
            req_pool_indices=req,
            seq_lens=seq,
            out_cache_loc=loc,
            block_size=5,
        )
        if not metadata._draft_inputs_ok(d, **kw):
            why = why or "draft inputs refused"
        else:
            got = metadata._draft_fused(d, **kw)
            why = why or differs(orig_block(d, **kw), got)
            held = got.core_attn_metadata.__dict__["_mono_slots"]
            want_loc = orig_translate(d.token_to_kv_pool, loc).to(torch.int32)
            why = why or differs(want_loc, held[1], "draft swa slots")
        print(f"trial {trial} bs {bs} ctx {ctx}: {why or 'ok'}")
        bad += why is not None
    bad += _check_inject(device)
    bad += _check_draft_head(device)
    bad += _check_posts(device)
    bad += _check_prefill_logits(device)
    print("FAILED" if bad else "PASSED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
