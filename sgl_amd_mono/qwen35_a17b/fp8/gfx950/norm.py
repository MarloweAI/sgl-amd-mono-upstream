# SPDX-License-Identifier: Apache-2.0
"""Residual add, Gemma RMSNorm and the 1x128 FP8 quant of its output, in one kernel.

The norm matches SGLang's ROCm Gemma add-RMSNorm: ``s = x + r`` in fp32, the
residual stores ``s`` in the activation dtype, ``rstd`` comes from the fp32
``s``, and ``y = s * rstd * (1 + w)``. The quant then reads ``y`` rounded to the
activation dtype, as the block-scale GEMM's own 1x128 quant does: per group,
``scale = max(amax, 1e-10) / 448`` and ``q = clamp(y / scale)`` in e4m3. The scale
is written column-major (``[groups, rows]`` bytes behind a ``[rows, groups]``
view), the layout the preshuffled GEMM reads without a copy.

``gate_quant`` is the same quant of the attention output gate
``attn * sigmoid(gate)``, which is taken in fp32 and rounded to the activation
dtype as SGLang's fused sigmoid-mul stores it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sgl_amd_mono.qwen35_a17b.fp8.gfx950.layout import BLOCK, FP8_MAX, HIDDEN


@triton.jit
def _add_norm_quant_kernel(
    x_ptr,
    stride_x,
    res_ptr,
    stride_res,
    w_ptr,
    out_ptr,
    res_out_ptr,
    q_ptr,
    scale_ptr,
    rows,
    eps,
    fp8_max,
    inv_fp8_max,
    HAS_RES: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, GROUPS)[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    s = tl.load(x_ptr + row * stride_x + offs).to(tl.float32)
    if HAS_RES:
        s = s + tl.load(res_ptr + row * stride_res + offs).to(tl.float32)
    tl.store(
        res_out_ptr + row * GROUPS * GROUP + offs, s.to(res_out_ptr.dtype.element_ty)
    )
    var = tl.sum(tl.sum(s * s, axis=1), axis=0) / (GROUPS * GROUP)
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs).to(tl.float32)
    y = (s * rstd * (1.0 + w)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * GROUPS * GROUP + offs, y)
    yf = y.to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(yf), axis=1), 1e-10)
    scale = amax * inv_fp8_max
    q = tl.clamp(yf * (1.0 / scale)[:, None], -fp8_max, fp8_max)
    tl.store(q_ptr + row * GROUPS * GROUP + offs, q.to(q_ptr.dtype.element_ty))
    tl.store(scale_ptr + tl.arange(0, GROUPS) * rows + row, scale)


@triton.jit
def _gate_quant_kernel(
    a_ptr,
    stride_a,
    g_ptr,
    stride_g,
    stride_g_head,
    out_ptr,
    q_ptr,
    scale_ptr,
    rows,
    fp8_max,
    inv_fp8_max,
    GROUPS: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, GROUPS)[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    head = offs // HEAD
    a = tl.load(a_ptr + row * stride_a + offs).to(tl.float32)
    g = tl.load(g_ptr + row * stride_g + head * stride_g_head + offs - head * HEAD)
    y = (a * tl.sigmoid(g.to(tl.float32))).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * GROUPS * GROUP + offs, y)
    yf = y.to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(yf), axis=1), 1e-10)
    scale = amax * inv_fp8_max
    q = tl.clamp(yf * (1.0 / scale)[:, None], -fp8_max, fp8_max)
    tl.store(q_ptr + row * GROUPS * GROUP + offs, q.to(q_ptr.dtype.element_ty))
    tl.store(scale_ptr + tl.arange(0, GROUPS) * rows + row, scale)


def gate_quant(attn: torch.Tensor, gate: torch.Tensor):
    """``(y, (q, scale))`` for ``y = attn * sigmoid(gate)`` and its 1x128 quant.

    ``attn`` is ``[rows, K]``. ``gate`` is ``[rows, K]`` or ``[rows, heads, head_dim]``
    with any row and head strides.
    """
    rows, k = attn.shape
    if gate.ndim == 3:
        head, strides = gate.shape[2], (gate.stride(0), gate.stride(1))
        fits = gate.shape[1] * head == k and gate.stride(2) == 1
    else:
        head, strides = k, (gate.stride(0), 0)
        fits = gate.shape == attn.shape and gate.stride(1) == 1
    if not fits or attn.stride(1) != 1 or k % BLOCK:
        raise RuntimeError(
            f"gate {tuple(gate.shape)} {gate.stride()} does not gate the core "
            f"{tuple(attn.shape)} {attn.stride()} in 1x{BLOCK} groups"
        )
    out = torch.empty(rows, k, dtype=attn.dtype, device=attn.device)
    q, scale = quant_buffers(rows, k, attn.device)
    _gate_quant_kernel[(rows,)](
        attn,
        attn.stride(0),
        gate,
        *strides,
        out,
        q,
        scale,
        rows,
        FP8_MAX,
        1.0 / FP8_MAX,
        GROUPS=k // BLOCK,
        GROUP=BLOCK,
        HEAD=head,
        num_warps=4,
    )
    return out, (q, scale)


def quant_buffers(rows: int, k: int, device):
    """FP8 rows and their column-major 1x128 scale, ``[rows, k / 128]``."""
    q = torch.empty(rows, k, dtype=torch.float8_e4m3fn, device=device)
    scale = torch.empty(k // BLOCK, rows, dtype=torch.float32, device=device).t()
    return q, scale


def add_norm_quant(hidden: torch.Tensor, residual, norm):
    """``(y, residual, (q, scale))`` for a Gemma RMSNorm ``norm`` of ``hidden + residual``."""
    rows, k = hidden.shape
    if k != HIDDEN:
        raise RuntimeError(f"hidden is [*, {k}], the norm wants [*, {HIDDEN}]")
    out = torch.empty_like(hidden)
    res_out = torch.empty_like(hidden)
    q, scale = quant_buffers(rows, k, hidden.device)
    _add_norm_quant_kernel[(rows,)](
        hidden,
        hidden.stride(0),
        hidden if residual is None else residual,
        hidden.stride(0) if residual is None else residual.stride(0),
        norm.weight,
        out,
        res_out,
        q,
        scale,
        rows,
        norm.variance_epsilon,
        FP8_MAX,
        1.0 / FP8_MAX,
        HAS_RES=residual is not None,
        GROUPS=k // BLOCK,
        GROUP=BLOCK,
        num_warps=16,
    )
    return out, res_out, (q, scale)
