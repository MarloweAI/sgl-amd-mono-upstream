# SPDX-License-Identifier: Apache-2.0
"""A layer's owed mHC post fused into the launch that reads its streams next: the head's
``mhc_combine`` (and the head's RMSNorm after it, ``gfx950/norm``), or the next layer's
Engram gate; aiter's ``mhc_post`` and RMSNorm bits.
``SGLANG_AMD_MONO_DSV41_FUSED_POST=0`` keeps the separate launches."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

ENV = "SGLANG_AMD_MONO_DSV41_FUSED_POST"

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/layernorm/mhc.py": ("mhc_combine",),
    # the head's RMSNorm, which _post_combine_norm reproduces
    "srt/layers/layernorm.py": ("RMSNorm", "variance_epsilon"),
    # the gate _post_gate_kernel reproduces
    "kernels/ops/embeddings/engram_gate.py": ("_engram_gate_kernel",),
    "srt/layers/engram.py": (
        "engram_gate",
        "engram_gate(input_ids)",
        "image_token_id",
        "q_weight",
        "k_weight",
        "eps",
        "clamp_value",
    ),
}


def enabled() -> bool:
    from sgl_amd_mono.deepseek_v41.glue import flag

    return flag(ENV)


@triton.jit
def _post_stream(X, R, POST, COMB, m, i, h, live, H, HC: tl.constexpr):
    """Stream ``i`` of ``mhc_post(x, residual, post, comb)`` at row ``m``, columns ``h``."""
    x = tl.load(X + m * H + h, mask=live, other=0.0).to(tl.float32)
    r0 = tl.load(R + m * HC * H + h, mask=live, other=0.0).to(tl.float32)
    v = tl.fma(x, tl.load(POST + m * HC + i), r0 * tl.load(COMB + m * HC * HC + i))
    for j in tl.static_range(1, HC):
        r = tl.load(R + m * HC * H + j * H + h, mask=live, other=0.0).to(tl.float32)
        v = tl.fma(r, tl.load(COMB + m * HC * HC + j * HC + i), v)
    return v.to(tl.bfloat16)


@triton.jit
def _post_combine_kernel(
    X, R, POST, COMB, PRE, OUT, Y, H, HC: tl.constexpr, BLOCK: tl.constexpr
):
    m = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = h < H
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(HC):
        vb = _post_stream(X, R, POST, COMB, m, i, h, live, H, HC)
        tl.store(OUT + m * HC * H + i * H + h, vb, mask=live)
        # mhc_combine of the rounded stream, as its kernel sums it
        acc += tl.load(PRE + m * HC + i).to(tl.float32) * vb.to(tl.float32)
    tl.store(Y + m * H + h, acc.to(tl.bfloat16), mask=live)


@triton.jit
def _post_gate_kernel(
    XO,
    R,
    POST,
    COMB,
    KV,
    QW,
    KW,
    IDS,
    O,
    image_token_id,
    D: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
    HAS_IDS: tl.constexpr,
):
    # SGLang's _engram_gate_kernel on the stream the owed post makes
    row = tl.program_id(0)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    x = _post_stream(XO, R, POST, COMB, token, hc, col, mask, D, HC).to(tl.float32)
    key = tl.load(KV + token * (HC + 1) * D + hc * D + col, mask, 0).to(tl.float32)
    q_weight = tl.load(QW + hc * D + col, mask, 0).to(tl.float32)
    k_weight = tl.load(KW + hc * D + col, mask, 0).to(tl.float32)
    weight = q_weight * k_weight
    rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS) * tl.rsqrt(
        tl.sum(key * key, 0) / D + EPS
    )
    dot = tl.sum((x * weight) * key, 0) * rstd * (D**-0.5)
    gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot))
    value = tl.load(KV + token * (HC + 1) * D + HC * D + col, mask, 0).to(tl.float32)
    out = x + gate * value
    if HAS_IDS:
        out = tl.where(tl.load(IDS + token) == image_token_id, x, out)
    tl.store(O + row * D + col, out, mask)


def _fits(x, residual, post, comb) -> bool:
    """x [M, H] and residual [M, HC, H] bf16, post [M, HC] and comb [M, HC, HC] fp32,
    all contiguous."""
    return (
        residual.dim() == 3
        and x.dtype == residual.dtype == torch.bfloat16
        and post.dtype == comb.dtype == torch.float32
        and tuple(x.shape) == (residual.shape[0], residual.shape[2])
        and tuple(post.shape) == residual.shape[:2]
        and tuple(comb.shape) == (*residual.shape[:2], residual.shape[1])
        and all(t.is_contiguous() for t in (x, residual, post, comb))
    )


def _post_combine(x, residual, post, comb, pre):
    m, hc, h = residual.shape
    out = torch.empty_like(residual)
    y = torch.empty((m, h), dtype=residual.dtype, device=residual.device)
    if m:
        block = 1024
        _post_combine_kernel[(m, triton.cdiv(h, block))](
            x, residual, post, comb, pre, out, y, h, HC=hc, BLOCK=block
        )
    return out, y


def _post_gate(x, residual, post, comb, kv, engram, input_ids):
    t, hc, d = residual.shape
    out = torch.empty_like(residual)
    image = engram.image_token_id
    if t:
        _post_gate_kernel[(t * hc,)](
            x,
            residual,
            post,
            comb,
            kv,
            engram.q_weight,
            engram.k_weight,
            input_ids,
            out,
            image if image is not None else 0,
            d,
            hc,
            engram.eps,
            engram.clamp_value,
            triton.next_power_of_2(d),
            image is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def post_combine(layer, pending, pre, hc: int):
    """(the streams the owed post (x, residual, post, comb) of ``layer`` settles, their
    ``mhc_combine`` by ``pre``), in one launch where the tensors fit."""
    from sglang.kernels.ops.layernorm.mhc import mhc_combine

    from sgl_amd_mono.deepseek_v41.glue import chose

    x, residual, post, comb = pending
    if chose(
        "post_combine",
        enabled()
        and _fits(x, residual, post, comb)
        and pre.dtype == torch.float32
        and pre.is_contiguous(),
    ):
        return _post_combine(x, residual, post, comb, pre)
    out = layer.hc_post(x, residual, post, comb)
    return out, mhc_combine(out.flatten(1), pre, hc, out.dtype)


def _post_combine_norm(x, residual, post, comb, pre, norm):
    from sgl_amd_mono.deepseek_v41.gfx950.norm import ops

    out = torch.empty_like(residual)
    normed = torch.empty_like(x)
    ops().combine_norm(
        x,
        residual,
        post,
        comb,
        pre,
        norm.weight.data,
        norm.variance_epsilon,
        out,
        normed,
    )
    return out, normed


def post_combine_norm(layer, pending, pre, hc: int, norm):
    """(the streams the owed post (x, residual, post, comb) of ``layer`` settles, SGLang's
    RMSNorm ``norm`` of their ``mhc_combine`` by ``pre``), in one launch where the tensors
    fit (4 streams, rows 4096 to 6144 wide)."""
    from sgl_amd_mono.deepseek_v41.glue import chose

    x, residual, post, comb = pending
    width = residual.shape[-1] if residual.dim() == 3 else 0
    if chose(
        "post_combine_norm",
        enabled()
        and _fits(x, residual, post, comb)
        and residual.shape[1] == 4
        and 4096 < width <= 6144
        and width % 8 == 0
        and pre.dtype == torch.float32
        and pre.is_contiguous()
        and type(norm).__name__ == "RMSNorm"
        and norm.weight.dtype == torch.bfloat16
        and norm.weight.is_contiguous(),
    ):
        return _post_combine_norm(x, residual, post, comb, pre, norm)
    out, combined = post_combine(layer, pending, pre, hc)
    return out, norm(combined)


def post_gate(engram, layer, pending, kv, input_ids):
    """``engram_gate`` on the streams the owed post (x, residual, post, comb) of
    ``layer`` settles (image-token rows kept), in one launch where the tensors fit."""
    from sglang.srt.layers.engram import engram_gate

    from sgl_amd_mono.deepseek_v41.glue import chose

    x, residual, post, comb = pending
    ids = input_ids if engram.image_token_id is not None else None
    if chose(
        "post_gate",
        enabled()
        and _fits(x, residual, post, comb)
        and kv.dtype == torch.bfloat16
        and kv.is_contiguous()
        and engram.q_weight.is_contiguous()
        and engram.k_weight.is_contiguous()
        and (ids is None or (ids.is_contiguous() and ids.shape == (x.shape[0],))),
    ):
        return _post_gate(x, residual, post, comb, kv, engram, input_ids)
    return engram_gate(
        layer.hc_post(x, residual, post, comb),
        kv,
        engram.q_weight,
        engram.k_weight,
        engram.eps,
        engram.clamp_value,
        ids,
        engram.image_token_id,
    )
