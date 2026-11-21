# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's routed experts at 48-180 rows (a8w4: MXFP8 activations, MXFP4 weights) as one persistent launch: aiter's
stage 1 (gate/up GEMM + SiTU v2, MXFP8 out, inline activation quant) and stage 2 (down GEMM, bf16 atomics into the
latent) run by one grid, a block's stage-2 tiles starting as soon as its six stage-1 tiles have stored their output.

``persist_moe(g1, g2)`` takes the keyword arguments aiter's ``fused_moe`` passes to
``aiter.ops.flydsl.mxfp4_gemm1_kernels.flydsl_mxfp4_gemm1`` and to
``aiter.ops.flydsl.kernels.mxmoe_dispatcher.mxfp4_moe_gemm2`` for one MoE call, and runs both in one launch with the
same tile bodies, so stage 1's output is the stock kernel's bit for bit; stage 2 adds the same products with the
same bf16 atomics (their order, as stock's, varies run to run).

One static configuration a width (``CONFIGS``, by aiter's tuned-table bucket): the stock tiles of that bucket and this
launch's grid and orders. ``check`` rejects a call whose stock kernels differ from the bucket's."""

from __future__ import annotations

import functools

import torch

from .kernel import LINE_B, NXCD, STAMP_WORDS, SYNC_HEAD_B, compile_persist

MIN_ROWS, MAX_ROWS = 48, 180
HIDDEN, INTER, EXPERTS, TOPK = 3584, 384, 896, 16
BM = 16

# aiter's kimik3_a8w4 tuned rows per bucket (get_padded_M: 64, 128, 256): stage-1 flags, stage-2 tile
_STOCK = {
    64: dict(g1=dict(use_nt=True, prefetch_hidden=False, xcd_swizzle=2), g2=dict(BN=128, BK=128, use_nt=True)),
    128: dict(g1=dict(use_nt=True, prefetch_hidden=True, xcd_swizzle=4), g2=dict(BN=128, BK=128, use_nt=True)),
    256: dict(g1=dict(use_nt=True, prefetch_hidden=True, xcd_swizzle=0), g2=dict(BN=256, BK=128, use_nt=True)),
}

# this launch per bucket: CTAs a XCD (co-resident: grid <= CUs x CTAs a CU), stage-1 and stage-2 block grouping
CONFIGS = {
    64: dict(ctas_per_xcd=128, s1_swizzle=2, s2_swizzle=2),
    128: dict(ctas_per_xcd=128, s1_swizzle=4, s2_swizzle=2),
    256: dict(ctas_per_xcd=128, s1_swizzle=1, s2_swizzle=2),
}


def bucket(rows: int) -> int:
    return 64 if rows <= 64 else 128 if rows <= 128 else 256


@functools.cache
def _launcher(b: int, situ_beta: float, situ_linear_beta: float, stamps: bool, overrides: tuple = ()):
    st, cf = _STOCK[b], dict(CONFIGS[b])
    cf.update(dict(overrides))
    return compile_persist(
        s1_use_nt=st["g1"]["use_nt"],
        s1_prefetch_hidden=cf.get("s1_prefetch_hidden", st["g1"]["prefetch_hidden"]),
        s1_swizzle=cf["s1_swizzle"],
        s2_bn=cf.get("s2_bn", st["g2"]["BN"]),
        s2_bk=cf.get("s2_bk", st["g2"]["BK"]),
        s2_use_nt=cf.get("s2_use_nt", st["g2"]["use_nt"]),
        s2_swizzle=cf["s2_swizzle"],
        ctas_per_xcd=cf["ctas_per_xcd"],
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        waves_per_eu=cf.get("waves_per_eu", 4),
        post_misched=cf.get("post_misched", True),
        stamps=stamps,
        only=cf.get("only", ""),
    )


_MAX_BLOCKS = 1152  # sorted M blocks at 192 rows (Opus: 192 x 16 + 896 x 16 - 16 rows)


@functools.cache
def _sync(device: torch.device) -> torch.Tensor:
    # [ticket count, epoch] a XCD, then the stage-1 flags of each M block (-1: no launch's tag). The launch leaves
    # every count at 0. Made outside graph capture (SGLang warms up eagerly first).
    s = torch.full(((SYNC_HEAD_B + _MAX_BLOCKS * LINE_B) // 4,), -1, dtype=torch.int32, device=device)
    s[: SYNC_HEAD_B // 4] = 0
    return s


def check_g1(g1: dict) -> int:
    """The bucket of this stage-1 call; AssertionError if its stock kernel is not the bucket's."""
    rows = g1["n_tokens"]
    assert MIN_ROWS <= rows <= MAX_ROWS, rows
    b = bucket(rows)
    want1 = dict(BM=BM, BN=128, BK=256, inline_quant=True, interleave=True, native_scale_layout=True, a_dtype="fp8",
                 out_dtype="fp8", act="situv2", num_waves=4, k_wave=1, NE=EXPERTS, D_HIDDEN=HIDDEN, D_INTER=INTER,
                 topk=TOPK, **_STOCK[b]["g1"])
    got1 = {k: g1.get(k) for k in want1}
    assert got1 == want1 and g1.get("bias") is None, (got1, want1)
    return b


def check(g1: dict, g2: dict) -> int:
    """The bucket of this call; AssertionError if its stock kernels are not the bucket's."""
    rows = g1["n_tokens"]
    assert MIN_ROWS <= rows <= MAX_ROWS, rows
    b = bucket(rows)
    st = _STOCK[b]
    want1 = dict(BM=BM, BN=128, BK=256, inline_quant=True, interleave=True, native_scale_layout=True, a_dtype="fp8",
                 out_dtype="fp8", act="situv2", num_waves=4, k_wave=1, NE=EXPERTS, D_HIDDEN=HIDDEN, D_INTER=INTER,
                 topk=TOPK, **st["g1"])
    got1 = {k: g1.get(k) for k in want1}
    assert got1 == want1 and g1.get("bias") is None, (got1, want1)
    want2 = dict(BM=BM, SBM=BM, a_dtype="fp8", b_dtype="fp4", epilog="atomic", persist=False, out_dtype="bf16",
                 NE=EXPERTS, D_HIDDEN=HIDDEN, D_INTER=INTER, **st["g2"])
    got2 = {k: g2.get(k) for k in want2}
    assert got2 == want2 and g2.get("bias") is None and not g2.get("is_ep"), (got2, want2)
    assert g2.get("g2_bf16_lds") in (None, True) and g2.get("g2_spart") in (None, 402), g2
    assert not g2.get("use_valid_token_count"), g2
    assert g2["inter_sorted_quant"].data_ptr() == g1["inter_sorted_quant"].data_ptr()
    assert g2["sorted_expert_ids"].data_ptr() == g1["sorted_expert_ids"].data_ptr()
    assert g2["cumsum_tensor"].data_ptr() == g1["cumsum_tensor"].data_ptr()
    assert g2["out"].dtype == torch.bfloat16 and g2["out"].shape[0] == rows
    return b


def _u8(t):
    return t.view(torch.uint8) if t.dtype != torch.uint8 else t


def persist_moe(g1: dict, g2: dict, stamps: torch.Tensor | None = None, overrides: tuple = ()) -> None:
    """Stage 1 + stage 2 of one MoE call in one launch (arguments: see the module doc). ``stamps``: an int64 buffer
    of [grid, STAMP_WORDS] for the stamped build (timing study only)."""
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    b = check(g1, g2)
    launch, grid = _launcher(b, float(g1["situ_beta"]), float(g1["situ_linear_beta"]), stamps is not None,
                             overrides)
    dev = g1["hidden_states"].device
    sync = _sync(dev)
    if stamps is not None:
        assert stamps.dtype == torch.int64 and stamps.numel() >= grid * STAMP_WORDS
    max_sorted = g2["max_sorted"]
    _run_compiled(
        launch,
        _u8(g1["w1_u8"]).data_ptr(),
        _u8(g1["w1_scale_u8"]).data_ptr(),
        g1["sorted_expert_ids"].data_ptr(),
        g1["cumsum_tensor"].data_ptr(),
        g1["m_indices"].data_ptr(),
        g1["inter_sorted_quant"].data_ptr(),
        g1["inter_sorted_shuffled_scale"].data_ptr(),
        g1["hidden_states"].data_ptr(),
        _u8(g2["w2_u8"]).data_ptr(),
        _u8(g2["w2_scale_u8"]).data_ptr(),
        g2["sorted_token_ids"].data_ptr(),
        g2["sorted_weights"].data_ptr(),
        g2["out"].data_ptr(),
        sync.data_ptr(),
        (stamps if stamps is not None else sync).data_ptr(),
        int(g1["n_tokens"]),
        (max_sorted + BM - 1) // BM,
        torch.cuda.current_stream(),
    )


def grid_of(rows: int, overrides: tuple = ()) -> int:
    cf = dict(CONFIGS[bucket(rows)])
    cf.update(dict(overrides))
    return NXCD * cf["ctas_per_xcd"]
