"""Check the DSV4.1 glue's launches against SGLang's on one GPU: every field of a verify step's
and a draft block's metadata and their SWA slots on random tables, lengths and slots, the
inject's cache bytes (V4, V4.1) and the draft head's FP8 row quant."""

import argparse
import sys

import torch


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
    b.low_ratios = () if draft else (1, 2)
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
    from types import SimpleNamespace

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
        why = inject._disagrees(st, pool)
        print(f"inject {layout.value}: {why or 'ok'}")
        bad += why is not None
    return bad


def _check_draft_head(device) -> int:
    """glue/draft_head's row quant vs SGLang's _fp8_quant_per_row, bitwise."""
    import sglang.srt.models.dspark  # noqa: F401

    from sgl_amd_mono.deepseek_v41.glue import draft_head

    g = torch.Generator(device=device).manual_seed(9)
    bad = 0
    for m in (5, 10, 40, 48):
        x = (torch.randn(m, 5120, generator=g, device=device) * 4).bfloat16()
        x[0, :7] = 0
        why = draft_head._disagrees(x)
        print(f"draft head quant {m} rows: {why or 'ok'}")
        bad += why is not None
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
        DSV4RawVerifyMetadata,
    )

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
        why = metadata.disagrees(orig_verify(b, raw), got)
        held = got.core_attn_metadata.__dict__["_mono_slots"]
        want_loc = b.token_to_kv_pool.full_to_swa_index_mapping[loc].to(torch.int32)
        why = why or (None if torch.equal(held[1], want_loc) else "swa slots differ")
        want_req = torch.repeat_interleave(req.to(torch.int64), 6)
        why = why or (None if torch.equal(held[2], want_req) else "request rows differ")
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
            why = why or metadata.disagrees(orig_block(d, **kw), got)
            held = got.core_attn_metadata.__dict__["_mono_slots"]
            want_loc = d.token_to_kv_pool.full_to_swa_index_mapping[loc].to(torch.int32)
            why = why or (
                None if torch.equal(held[1], want_loc) else "draft swa slots differ"
            )
        print(f"trial {trial} bs {bs} ctx {ctx}: {why or 'ok'}")
        bad += why is not None
    bad += _check_inject(device)
    bad += _check_draft_head(device)
    print("FAILED" if bad else "PASSED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
