# SPDX-License-Identifier: Apache-2.0
"""Target-verify GDN for one Qwen3.5 block-FP8 rank: the draft chain of each request.

A verify step has ``tokens`` rows a request, row ``r * tokens + t``. The block-FP8
``in_proj_qkvz`` (``gemv.block_fp8_gemv_wg``) is followed by two kernels:

- ``_conv_chain_kernel``: per request and conv channel, the causal conv and SiLU
  over the chain from the committed window. It writes the window after every
  token into SGLang's ``intermediate_conv_window`` (row of the request's
  intermediate slot, step ``t``) and the window after the last token into the
  conv state, as ``causal_conv1d_update`` does in a verify step.
- ``_delta_chain_kernel``: per request and V head, the bf16 ``in_proj_ba``
  projection of each token, the gated delta rule on the committed state, the
  gated RMSNorm and the 1x128 FP8 quant of ``out_proj``'s input (one V head is
  one quant group). The state after every token goes to ``intermediate_ssm``;
  the committed state is left for SGLang's accept step to fill from there.

Rounding follows SGLang's verify kernels: each conv tap's product in bf16, the gates in fp32 with a plain
softplus and an unrounded ``sigmoid(b)``, the core rounded to bf16 before the
norm, and a bf16 SSM state re-rounded between tokens. Pad requests (cache index
< 0) write no state and return zeros.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sgl_amd_mono.qwen35_a17b.fp8.gfx950.gdn import _conv_strides
from sgl_amd_mono.qwen35_a17b.fp8.gfx950.gemv import block_fp8_gemv_wg
from sgl_amd_mono.qwen35_a17b.fp8.gfx950.layout import (
    CONV_KERNEL,
    FP8_MAX,
    SOFTPLUS_THRESHOLD,
)
from sgl_amd_mono.qwen35_a17b.fp8.gfx950.norm import quant_buffers

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/attention/linear/utils.py": (
        "select_verify_intermediate_state_indices",
    ),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/mem_cache/memory_pool.py": (
        "intermediate_ssm",
        "intermediate_conv_window",
    ),
}

CONV_BLOCK = 256
# the draft chain the kernels unroll: --speculative-num-draft-tokens
TOKENS = 4


@triton.jit
def _conv_chain_kernel(
    x_ptr,
    stride_x,
    w_ptr,
    bias_ptr,
    state_ptr,
    stride_slot,
    stride_dim,
    stride_win,
    slot_ptr,
    inter_ptr,
    stride_i_slot,
    stride_i_step,
    stride_i_dim,
    stride_i_win,
    inter_slot_ptr,
    out_ptr,
    stride_out,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    req = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < C
    slot = tl.load(slot_ptr + req).to(tl.int64)
    live = slot >= 0
    keep = mask & live
    cell = state_ptr + tl.maximum(slot, 0) * stride_slot + cols * stride_dim
    c0 = tl.load(cell, mask=keep, other=0.0)
    c1 = tl.load(cell + stride_win, mask=keep, other=0.0)
    c2 = tl.load(cell + 2 * stride_win, mask=keep, other=0.0)
    w0 = tl.load(w_ptr + cols * 4, mask=mask, other=0.0).to(c0.dtype)
    w1 = tl.load(w_ptr + cols * 4 + 1, mask=mask, other=0.0).to(c0.dtype)
    w2 = tl.load(w_ptr + cols * 4 + 2, mask=mask, other=0.0).to(c0.dtype)
    w3 = tl.load(w_ptr + cols * 4 + 3, mask=mask, other=0.0).to(c0.dtype)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    islot = tl.load(inter_slot_ptr + req).to(tl.int64)
    save = keep & (islot >= 0)
    ibase = inter_ptr + tl.maximum(islot, 0) * stride_i_slot + cols * stride_i_dim
    for t in tl.static_range(T):
        row = req * T + t
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
        # each tap's product is rounded to bf16, as causal_conv1d_update's are
        acc = (c0 * w0).to(tl.float32) + (c1 * w1).to(tl.float32)
        acc = acc + (c2 * w2).to(tl.float32) + (x * w3).to(tl.float32)
        if HAS_BIAS:
            acc = acc + bias
        y = acc / (1.0 + tl.exp(-acc))
        y = tl.where(live, y, 0.0)
        tl.store(
            out_ptr + row * stride_out + cols,
            y.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
        c0 = c1
        c1 = c2
        c2 = x.to(c2.dtype)
        step = ibase + t * stride_i_step
        tl.store(step, c0.to(inter_ptr.dtype.element_ty), mask=save)
        tl.store(step + stride_i_win, c1.to(inter_ptr.dtype.element_ty), mask=save)
        tl.store(step + 2 * stride_i_win, c2.to(inter_ptr.dtype.element_ty), mask=save)
    tl.store(cell, c0.to(state_ptr.dtype.element_ty), mask=keep)
    tl.store(cell + stride_win, c1.to(state_ptr.dtype.element_ty), mask=keep)
    tl.store(cell + 2 * stride_win, c2.to(state_ptr.dtype.element_ty), mask=keep)


@triton.jit
def _delta_chain_kernel(
    mixed_ptr,
    stride_mixed,
    qkvz_ptr,
    stride_qkvz,
    hidden_ptr,
    stride_hidden,
    ba_w_ptr,
    stride_ba_w,
    state_ptr,
    stride_s_slot,
    stride_s_head,
    stride_s_v,
    stride_s_k,
    slot_ptr,
    inter_ptr,
    stride_i_slot,
    stride_i_step,
    stride_i_head,
    stride_i_v,
    stride_i_k,
    inter_slot_ptr,
    a_log_ptr,
    dt_bias_ptr,
    norm_w_ptr,
    out_ptr,
    stride_out,
    q_ptr,
    stride_q,
    q_scale_ptr,
    rows,
    scale,
    eps,
    fp8_max,
    inv_fp8_max,
    HK: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    K_IN: tl.constexpr,
    T: tl.constexpr,
    THRESHOLD: tl.constexpr,
    ROUND: tl.constexpr,
):
    req = tl.program_id(0)
    hv = tl.program_id(1)
    kh = hv // (HV // HK)
    offs = tl.arange(0, D)
    slot = tl.load(slot_ptr + req).to(tl.int64)
    live = slot >= 0
    cell = (
        state_ptr
        + tl.maximum(slot, 0) * stride_s_slot
        + hv * stride_s_head
        + offs[:, None] * stride_s_v
        + offs[None, :] * stride_s_k
    )
    h = tl.load(cell, mask=live, other=0.0).to(tl.float32)
    islot = tl.load(inter_slot_ptr + req).to(tl.int64)
    save = live & (islot >= 0)
    icell = (
        inter_ptr
        + tl.maximum(islot, 0) * stride_i_slot
        + hv * stride_i_head
        + offs[:, None] * stride_i_v
        + offs[None, :] * stride_i_k
    )
    a_log = tl.load(a_log_ptr + hv).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + hv).to(tl.float32)
    norm_w = tl.load(norm_w_ptr + offs).to(tl.float32)
    cols = tl.arange(0, K_IN)
    w_b = tl.load(ba_w_ptr + hv * stride_ba_w + cols).to(tl.float32)
    w_a = tl.load(ba_w_ptr + (HV + hv) * stride_ba_w + cols).to(tl.float32)
    for t in tl.static_range(T):
        row = req * T + t
        x_in = tl.load(hidden_ptr + row * stride_hidden + cols).to(tl.float32)
        b = tl.sum(x_in * w_b, axis=0).to(tl.bfloat16).to(tl.float32)
        a = tl.sum(x_in * w_a, axis=0).to(tl.bfloat16).to(tl.float32)
        base = mixed_ptr + row * stride_mixed
        q = tl.load(base + kh * D + offs).to(tl.float32)
        k = tl.load(base + HK * D + kh * D + offs).to(tl.float32)
        v = tl.load(base + 2 * HK * D + hv * D + offs).to(tl.float32)
        z = tl.load(qkvz_ptr + row * stride_qkvz + 2 * HK * D + HV * D + hv * D + offs)
        z = z.to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q, axis=0) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k, axis=0) + 1e-6)
        q = q * scale
        x = a + dt_bias
        softplus = tl.where(x <= THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        g = -tl.exp(a_log) * softplus
        beta = 1.0 / (1.0 + tl.exp(-b))
        h = h * tl.exp(g)
        v = (v - tl.sum(h * k[None, :], axis=1)) * beta
        h = h + v[:, None] * k[None, :]
        o = tl.sum(h * q[None, :], axis=1)
        tl.store(icell + t * stride_i_step, h.to(inter_ptr.dtype.element_ty), mask=save)
        if ROUND:
            h = h.to(tl.bfloat16).to(tl.float32)
        o = o.to(out_ptr.dtype.element_ty).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(o * o, axis=0) / D + eps)
        y = o * rstd * norm_w * (z * tl.sigmoid(z))
        y = tl.where(live, y, 0.0).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_out + hv * D + offs, y)
        yf = y.to(tl.float32)
        group_scale = tl.maximum(tl.max(tl.abs(yf), axis=0), 1e-10) * inv_fp8_max
        yq = tl.clamp(yf * (1.0 / group_scale), -fp8_max, fp8_max)
        tl.store(q_ptr + row * stride_q + hv * D + offs, yq.to(q_ptr.dtype.element_ty))
        tl.store(q_scale_ptr + hv * rows + row, group_scale)


def verify_slots(forward_batch, requests: int):
    """``(cache, inter)``: each request's committed state slot and its row of the
    verify scratch, as SGLang's GDN verify selects them."""
    from sglang.srt.layers.attention.linear.utils import (
        select_verify_intermediate_state_indices,
    )
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    lin = getattr(backend, "linear_attn_backend", backend)
    md = lin.forward_metadata
    cache = md.mamba_cache_indices
    inter = select_verify_intermediate_state_indices(
        lin.verify_intermediate_state_indices,
        forward_batch.req_pool_indices,
        cache[:requests] >= 0,
        lin.req_to_token_pool.size,
    )
    return cache[:requests], inter[:requests]


def verify_core(mod, hidden, hidden_q, cache, where, rows, tokens, layout):
    """The local core ``[rows, V]`` and its FP8 ``(q, scale)`` for a verify step.

    ``cache`` is the layer's mamba cache (``conv``, ``temporal``,
    ``intermediate_conv_window``, ``intermediate_ssm``); ``where`` is the step's
    ``verify_slots``, the same for every layer.
    """
    if tokens != TOKENS or rows % tokens:
        raise RuntimeError(
            f"verify GDN unrolls {TOKENS}-token chains, got {rows} rows of {tokens}"
        )
    requests = rows // tokens
    x = hidden[:rows]
    w, w_scale = mod.in_proj_qkvz.weight, mod.in_proj_qkvz.weight_scale_inv
    ba_w = mod.in_proj_ba.weight
    if getattr(mod.in_proj_ba, "bias", None) is not None:
        raise RuntimeError("in_proj_ba has a bias; the fused GDN verify reads none")
    if w.shape[0] != layout.qkvz_n or ba_w.shape != (layout.ba_n, layout.hidden):
        raise RuntimeError(
            f"GDN projections are {tuple(w.shape)} and {tuple(ba_w.shape)}, "
            f"TP{layout.tp} wants [{layout.qkvz_n}, {layout.hidden}] and "
            f"[{layout.ba_n}, {layout.hidden}]"
        )
    slots, inter = where
    conv = cache.conv[0]
    ssm = cache.temporal
    inter_conv = cache.intermediate_conv_window[0]
    inter_ssm = cache.intermediate_ssm
    qkvz = block_fp8_gemv_wg(x, hidden_q, w, w_scale)
    conv_dim = layout.conv_dim
    weight = mod.conv1d.weight.reshape(conv_dim, CONV_KERNEL)
    bias = mod.conv1d.bias
    mixed = torch.empty(rows, conv_dim, dtype=x.dtype, device=x.device)
    slot_stride, dim_stride, win_stride = _conv_strides(conv)
    _conv_chain_kernel[(requests, triton.cdiv(conv_dim, CONV_BLOCK))](
        qkvz,
        qkvz.stride(0),
        weight,
        weight if bias is None else bias,
        conv,
        slot_stride,
        dim_stride,
        win_stride,
        slots,
        inter_conv,
        inter_conv.stride(0),
        inter_conv.stride(1),
        inter_conv.stride(2),
        inter_conv.stride(3),
        inter,
        mixed,
        mixed.stride(0),
        C=conv_dim,
        T=tokens,
        BLOCK=CONV_BLOCK,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    heads, head = layout.gdn_v_heads, layout.gdn_head
    out = torch.empty(rows, heads * head, dtype=x.dtype, device=x.device)
    out_q, out_scale = quant_buffers(rows, heads * head, x.device)
    _delta_chain_kernel[(requests, heads)](
        mixed,
        mixed.stride(0),
        qkvz,
        qkvz.stride(0),
        x,
        x.stride(0),
        ba_w,
        ba_w.stride(0),
        ssm,
        ssm.stride(0),
        ssm.stride(1),
        ssm.stride(2),
        ssm.stride(3),
        slots,
        inter_ssm,
        inter_ssm.stride(0),
        inter_ssm.stride(1),
        inter_ssm.stride(2),
        inter_ssm.stride(3),
        inter_ssm.stride(4),
        inter,
        mod.A_log.reshape(-1),
        mod.dt_bias.reshape(-1),
        mod.norm.weight,
        out,
        out.stride(0),
        out_q,
        out_q.stride(0),
        out_scale,
        rows,
        head**-0.5,
        mod.norm.eps,
        FP8_MAX,
        1.0 / FP8_MAX,
        HK=layout.gdn_k_heads,
        HV=heads,
        D=head,
        K_IN=layout.hidden,
        T=tokens,
        THRESHOLD=SOFTPLUS_THRESHOLD,
        ROUND=ssm.dtype == torch.bfloat16,
        num_warps=8,
    )
    return out, (out_q, out_scale)
