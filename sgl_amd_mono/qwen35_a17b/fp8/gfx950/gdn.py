# SPDX-License-Identifier: Apache-2.0
"""Decode projection, conv and gated delta for one Qwen3.5 GDN layer, in two kernels.

The first kernel is the block-FP8 ``in_proj_qkvz`` (``gemv.wg_dot``): each
workgroup's 16 columns are rounded to the activation dtype, as the linear
rounds them, then a conv channel's column goes through the conv and z is
stored for the second kernel, the gated delta. A plain weight takes the
block-FP8 linear and a separate conv kernel.
The conv window matches causal_conv1d_update (kernel width 4): state columns
are oldest to newest, ``acc = c0*w0 + c1*w1 + c2*w2 + x*w3 + bias``, then
SiLU, then the window shifts so the newest column is ``x``. The recurrence
matches the packed decode rule: per-head L2 norm of q and k, ``q *= K**-0.5``,
``g = -exp(A_log) * softplus(a + dt_bias)``, beta rounded through the
projection dtype, then ``h *= exp(g); v -= h k; v *= beta; h += v k``. The
core is rounded to the activation dtype before the gated RMSNorm
``rmsnorm(o) * w * silu(z)``. ``b`` and ``a`` are the bf16 ``in_proj_ba``
projection, rounded to bf16 the way the linear rounds them.
One V head is one 1x128 quant group of ``out_proj``'s input, so the delta kernel
also writes the FP8 core and its column-major scale (see ``norm.py``).
Pad rows (cache index < 0) write no state and return zeros.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sgl_amd_mono.qwen35_a17b.fp8.gfx950.gemv import (
    WG_BN,
    WG_ROWS,
    WG_SPLIT,
    WG_WARPS,
    block_fp8_gemv_q,
    wg_dot,
    wg_fits,
)
from sgl_amd_mono.qwen35_a17b.fp8.gfx950.layout import (
    BA_N,
    BLOCK,
    CONV_DIM,
    CONV_KERNEL,
    FP8_MAX,
    GDN_HEAD,
    GDN_K_HEADS,
    GDN_V_HEADS,
    HIDDEN,
    QKVZ_N,
    SOFTPLUS_THRESHOLD,
)
from sgl_amd_mono.qwen35_a17b.fp8.gfx950.norm import quant_buffers

CONV_BLOCK = 1024


@triton.jit
def _proj_conv_kernel(
    x_ptr,
    stride_x,
    xs_ptr,
    stride_xs_m,
    stride_xs_k,
    w_ptr,
    ws_ptr,
    stride_ws_n,
    stride_ws_k,
    state_ptr,
    stride_slot,
    stride_dim,
    stride_win,
    tap_ptr,
    bias_ptr,
    idx_ptr,
    mixed_ptr,
    stride_mixed,
    z_ptr,
    stride_z,
    M,
    C: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    GROUP: tl.constexpr,
    BN: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.arange(0, ROWS)
    cols = pid * BN + tl.arange(0, BN)
    rows_ok = m < M
    is_conv = pid * BN < C
    # The window and taps load ahead of the product, which hides their latency
    slot = tl.load(idx_ptr + m, mask=rows_ok & is_conv, other=-1).to(tl.int64)
    live = (slot >= 0)[:, None]
    cell = (
        state_ptr
        + tl.maximum(slot, 0)[:, None] * stride_slot
        + cols[None, :] * stride_dim
    )
    s0 = tl.load(cell, mask=live, other=0.0)
    s1 = tl.load(cell + stride_win, mask=live, other=0.0)
    s2 = tl.load(cell + 2 * stride_win, mask=live, other=0.0)
    taps = cols < C
    w0 = tl.load(tap_ptr + cols * 4, mask=taps, other=0.0).to(tl.float32)
    w1 = tl.load(tap_ptr + cols * 4 + 1, mask=taps, other=0.0).to(tl.float32)
    w2 = tl.load(tap_ptr + cols * 4 + 2, mask=taps, other=0.0).to(tl.float32)
    w3 = tl.load(tap_ptr + cols * 4 + 3, mask=taps, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols, mask=taps, other=0.0).to(tl.float32)
    x = wg_dot(
        x_ptr,
        stride_x,
        xs_ptr,
        stride_xs_m,
        stride_xs_k,
        w_ptr,
        ws_ptr,
        stride_ws_n,
        stride_ws_k,
        pid,
        M,
        K=K,
        S=S,
        GROUP=GROUP,
        BN=BN,
        ROWS=ROWS,
    ).to(mixed_ptr.dtype.element_ty)
    if is_conv:
        acc = s0.to(tl.float32) * w0[None, :] + s1.to(tl.float32) * w1[None, :]
        acc = acc + s2.to(tl.float32) * w2[None, :] + x.to(tl.float32) * w3[None, :]
        if HAS_BIAS:
            acc = acc + bias[None, :]
        y = acc / (1.0 + tl.exp(-acc))
        out = mixed_ptr + m[:, None] * stride_mixed + cols[None, :]
        tl.store(out, y.to(mixed_ptr.dtype.element_ty), mask=rows_ok[:, None])
        tl.store(cell, s1, mask=live)
        tl.store(cell + stride_win, s2, mask=live)
        tl.store(cell + 2 * stride_win, x.to(state_ptr.dtype.element_ty), mask=live)
    else:
        tl.store(
            z_ptr + m[:, None] * stride_z + cols[None, :], x, mask=rows_ok[:, None]
        )


@triton.jit
def _conv_kernel(
    x_ptr,
    stride_x,
    state_ptr,
    stride_slot,
    stride_dim,
    stride_win,
    w_ptr,
    bias_ptr,
    idx_ptr,
    out_ptr,
    stride_out,
    C,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < C
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    w0 = tl.load(w_ptr + cols * 4, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + cols * 4 + 1, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + cols * 4 + 2, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + cols * 4 + 3, mask=mask, other=0.0).to(tl.float32)
    slot = tl.load(idx_ptr + row).to(tl.int64)
    live = slot >= 0
    cell = state_ptr + tl.maximum(slot, 0) * stride_slot + cols * stride_dim
    s0 = tl.load(cell, mask=mask & live, other=0.0)
    s1 = tl.load(cell + stride_win, mask=mask & live, other=0.0)
    s2 = tl.load(cell + 2 * stride_win, mask=mask & live, other=0.0)
    acc = s0.to(tl.float32) * w0 + s1.to(tl.float32) * w1
    acc = acc + s2.to(tl.float32) * w2 + x.to(tl.float32) * w3
    if HAS_BIAS:
        acc = acc + tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = acc / (1.0 + tl.exp(-acc))
    tl.store(
        out_ptr + row * stride_out + cols, y.to(out_ptr.dtype.element_ty), mask=mask
    )
    keep = mask & live
    tl.store(cell, s1, mask=keep)
    tl.store(cell + stride_win, s2, mask=keep)
    tl.store(cell + 2 * stride_win, x.to(state_ptr.dtype.element_ty), mask=keep)


@triton.jit
def _delta_kernel(
    conv_ptr,
    stride_conv,
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
    idx_ptr,
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
    THRESHOLD: tl.constexpr,
):
    row = tl.program_id(0)
    hv = tl.program_id(1)
    kh = hv // (HV // HK)
    offs = tl.arange(0, D)
    slot = tl.load(idx_ptr + row).to(tl.int64)
    live = slot >= 0
    cell = (
        state_ptr
        + tl.maximum(slot, 0) * stride_s_slot
        + hv * stride_s_head
        + offs[:, None] * stride_s_v
        + offs[None, :] * stride_s_k
    )
    h = tl.load(cell, mask=live, other=0.0).to(tl.float32)
    base = conv_ptr + row * stride_conv
    q = tl.load(base + kh * D + offs).to(tl.float32)
    k = tl.load(base + HK * D + kh * D + offs).to(tl.float32)
    v = tl.load(base + 2 * HK * D + hv * D + offs).to(tl.float32)
    z = tl.load(qkvz_ptr + row * stride_qkvz + 2 * HK * D + HV * D + hv * D + offs)
    z = z.to(tl.float32)
    # b and a: rows hv and HV + hv of the bf16 in_proj_ba weight
    cols = tl.arange(0, K_IN)
    x_in = tl.load(hidden_ptr + row * stride_hidden + cols).to(tl.float32)
    w_b = tl.load(ba_w_ptr + hv * stride_ba_w + cols).to(tl.float32)
    w_a = tl.load(ba_w_ptr + (HV + hv) * stride_ba_w + cols).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + hv).to(tl.float32)
    a_log = tl.load(a_log_ptr + hv).to(tl.float32)
    norm_w = tl.load(norm_w_ptr + offs).to(tl.float32)

    b = tl.sum(x_in * w_b, axis=0).to(tl.bfloat16).to(tl.float32)
    a = tl.sum(x_in * w_a, axis=0).to(tl.bfloat16).to(tl.float32)
    q = q / tl.sqrt(tl.sum(q * q, axis=0) + 1e-6)
    k = k / tl.sqrt(tl.sum(k * k, axis=0) + 1e-6)
    q = q * scale
    x = a + dt_bias
    e = tl.exp(tl.minimum(x, THRESHOLD))
    u = 1.0 + e
    softplus = tl.where(x <= THRESHOLD, tl.log(u) - ((u - 1.0) - e) / u, x)
    g = -tl.exp(a_log) * softplus
    beta = tl.sigmoid(b).to(tl.bfloat16).to(tl.float32)

    h = h * tl.exp(g)
    v = (v - tl.sum(h * k[None, :], axis=1)) * beta
    h = h + v[:, None] * k[None, :]
    o = tl.sum(h * q[None, :], axis=1)
    tl.store(cell, h.to(state_ptr.dtype.element_ty), mask=live)

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


def _conv_strides(conv: torch.Tensor):
    """``conv`` is ``[slots, dim, 3]`` or ``[slots, 3, dim]``."""
    if conv.shape[-1] == CONV_KERNEL - 1:
        return conv.stride(0), conv.stride(1), conv.stride(2)
    return conv.stride(0), conv.stride(2), conv.stride(1)


def gdn_core(mod, hidden, hidden_q, conv, ssm, idx, rows: int, layout=None):
    """The local core ``[rows, V]`` and its FP8 ``(q, scale)``.

    ``hidden`` is the input RMSNorm's bf16 output and ``hidden_q`` its FP8 pair.
    ``layout`` is the rank specialization; omitted, the TP4 constants are used.
    """
    qkvz_n = QKVZ_N if layout is None else layout.qkvz_n
    ba_n = BA_N if layout is None else layout.ba_n
    conv_dim = CONV_DIM if layout is None else layout.conv_dim
    gdn_k_heads = GDN_K_HEADS if layout is None else layout.gdn_k_heads
    gdn_v_heads = GDN_V_HEADS if layout is None else layout.gdn_v_heads
    gdn_head = GDN_HEAD if layout is None else layout.gdn_head
    hidden_size = HIDDEN if layout is None else layout.hidden
    fp8_max = FP8_MAX if layout is None else layout.fp8_max
    threshold = SOFTPLUS_THRESHOLD if layout is None else layout.softplus_threshold
    tp = 4 if layout is None else layout.tp
    x = hidden[:rows]
    w, w_scale = mod.in_proj_qkvz.weight, mod.in_proj_qkvz.weight_scale_inv
    ba_w = mod.in_proj_ba.weight
    if getattr(mod.in_proj_ba, "bias", None) is not None:
        raise RuntimeError("in_proj_ba has a bias; the fused GDN decode reads none")
    if (
        w.shape[0] != qkvz_n
        or ba_w.shape != (ba_n, hidden_size)
        or x.shape[-1] != hidden_size
    ):
        raise RuntimeError(
            f"GDN projections are {tuple(w.shape)} and {tuple(ba_w.shape)}, "
            f"TP{tp} wants [{qkvz_n}, {hidden_size}] and [{ba_n}, {hidden_size}]"
        )
    cache_idx = idx[:rows]
    weight = mod.conv1d.weight.reshape(conv_dim, CONV_KERNEL)
    bias = mod.conv1d.bias
    mixed = torch.empty(rows, conv_dim, dtype=x.dtype, device=x.device)
    slot_stride, dim_stride, win_stride = _conv_strides(conv)
    if rows <= WG_ROWS and wg_fits(rows, hidden_size, w) and conv_dim % WG_BN == 0:
        # Only z's columns are written: the conv columns go straight to ``mixed``
        qkvz = torch.empty(rows, qkvz_n, dtype=x.dtype, device=x.device)
        q, q_scale = hidden_q
        _proj_conv_kernel[(qkvz_n // WG_BN,)](
            q,
            q.stride(0),
            q_scale,
            q_scale.stride(0),
            q_scale.stride(1),
            w,
            w_scale,
            w_scale.stride(0),
            w_scale.stride(1),
            conv,
            slot_stride,
            dim_stride,
            win_stride,
            weight,
            weight if bias is None else bias,
            cache_idx,
            mixed,
            mixed.stride(0),
            qkvz,
            qkvz.stride(0),
            rows,
            C=conv_dim,
            K=hidden_size,
            S=WG_SPLIT,
            GROUP=BLOCK,
            BN=WG_BN,
            ROWS=WG_ROWS,
            HAS_BIAS=bias is not None,
            num_warps=WG_WARPS,
        )
    else:
        qkvz = block_fp8_gemv_q(x, hidden_q, w, w_scale)
        _conv_kernel[(rows, triton.cdiv(conv_dim, CONV_BLOCK))](
            qkvz,
            qkvz.stride(0),
            conv,
            slot_stride,
            dim_stride,
            win_stride,
            weight,
            weight if bias is None else bias,
            cache_idx,
            mixed,
            mixed.stride(0),
            conv_dim,
            HAS_BIAS=bias is not None,
            BLOCK=CONV_BLOCK,
        )
    out = torch.empty(rows, gdn_v_heads * gdn_head, dtype=x.dtype, device=x.device)
    out_q, out_scale = quant_buffers(rows, gdn_v_heads * gdn_head, x.device)
    _delta_kernel[(rows, gdn_v_heads)](
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
        cache_idx,
        mod.A_log.reshape(-1),
        mod.dt_bias.reshape(-1),
        mod.norm.weight,
        out,
        out.stride(0),
        out_q,
        out_q.stride(0),
        out_scale,
        rows,
        gdn_head**-0.5,
        mod.norm.eps,
        fp8_max,
        1.0 / fp8_max,
        HK=gdn_k_heads,
        HV=gdn_v_heads,
        D=gdn_head,
        K_IN=hidden_size,
        THRESHOLD=threshold,
        num_warps=8,
    )
    return out, (out_q, out_scale)
