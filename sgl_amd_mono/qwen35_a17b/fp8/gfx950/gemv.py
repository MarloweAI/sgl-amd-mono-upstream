# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 GEMV for one gfx950 rank.

Weights are float8_e4m3fn with a fp32 scale per 128x128 block. Dynamic
activations use the same 1x128 group scale (amax / 448). On gfx950, SGLang
stores the weight in aiter's (16, 16) preshuffle; each expert of a 3D MoE
weight is shuffled on its own last two axes. The preshuffled path calls the
same block-scale GEMM those weights were shuffled for. A plain (row-major)
weight is dequantized per block and multiplied in fp32.

At decode widths that GEMM walks all of K in one workgroup, so a long K is
bound by latency, not bandwidth. ``block_fp8_gemv_wg`` reads the same
preshuffled weight but splits K into slices across the warps of a workgroup,
one batched fp8 dot per 128-wide group scaled by its activation and weight
scales as the GEMM does, and adds the slices in fp32 at the end.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from sgl_amd_mono.qwen35_a17b.fp8.gfx950.layout import BLOCK, FP8_MAX

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/quantization/fp8_utils.py": ("aiter_w8a8_block_fp8_linear",),
}

# block_fp8_gemv_wg: one 16-row block of the preshuffle per workgroup, K in 8
# slices over 4 warps; tl.dot takes at least 16 rows. Verify steps (4 rows a
# request, up to 8 requests) take the 32-row tile.
WG_BN = 16
WG_SPLIT = 8
WG_ROWS = 16
WG_MAX_ROWS = 32
WG_WARPS = 4


def _shuffled(weight: torch.Tensor) -> bool:
    if bool(getattr(weight, "is_shuffled", False)):
        return True
    return bool(getattr(weight, "aiter_bpreshuffled", False))


def block_fp8_gemv(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``x @ W.T`` for a block-FP8 weight ``W`` of shape ``[N, K]``.

    ``x`` is bf16 ``[M, K]``. ``scale`` is fp32 ``[N/128, K/128]``.
    """
    if x.numel() == 0:
        return x.new_empty(x.shape[0], weight.shape[0])
    if _shuffled(weight):
        return _aiter_gemv(x, weight, scale)
    return _plain_gemv(x, weight, scale)


def block_fp8_gemv_q(
    x: torch.Tensor, xq, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``block_fp8_gemv`` from ``x``'s FP8 pair ``xq``: e4m3 rows and their
    column-major 1x128 scale (``norm.quant_buffers``). A plain weight reads ``x``."""
    if x.numel() == 0:
        return x.new_empty(x.shape[0], weight.shape[0])
    if not _shuffled(weight):
        return _plain_gemv(x, weight, scale)
    from sglang.srt.layers.quantization.fp8_utils import aiter_w8a8_block_fp8_linear

    _allow_block_fp8_linear()
    q, q_scale = xq
    return aiter_w8a8_block_fp8_linear(
        q, weight, [BLOCK, BLOCK], scale, input_scale=q_scale
    )


@triton.jit
def wg_dot(
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
    K: tl.constexpr,
    S: tl.constexpr,
    GROUP: tl.constexpr,
    BN: tl.constexpr,
    ROWS: tl.constexpr,
):
    """fp32 ``[ROWS, BN]``: rows ``[0, M)`` of the product for weight rows
    ``pid * BN + [0, BN)``, K in ``S`` slices of one batched dot each."""
    s = tl.arange(0, S)
    m = tl.arange(0, ROWS)
    kk = tl.arange(0, GROUP)
    nn = tl.arange(0, BN)
    # (n, k) of the (16, 16) preshuffle sits at
    # [n // 16, k // 32, k // 16 % 2, n % 16, k % 16] of [N/16, K/32, 2, 16, 16]
    w_row = (pid * (BN // 16) + nn // 16) * (K * 16) + (nn % 16) * 16
    acc = tl.zeros([S, ROWS, BN], dtype=tl.float32)
    for step in tl.range(0, K // S // GROUP, num_stages=2):
        kb = s * (K // S // GROUP) + step
        k = kb[:, None] * GROUP + kk[None, :]
        x = tl.load(
            x_ptr + m[None, :, None] * stride_x + k[:, None, :],
            mask=m[None, :, None] < M,
            other=0.0,
        )
        xs = tl.load(
            xs_ptr + m[None, :] * stride_xs_m + kb[:, None] * stride_xs_k,
            mask=m[None, :] < M,
            other=0.0,
        )
        w_col = (k // 32) * 512 + (k // 16 % 2) * 256 + k % 16
        w = tl.load(w_ptr + w_col[:, :, None] + w_row[None, None, :])
        ws = tl.load(ws_ptr + (pid * BN // GROUP) * stride_ws_n + kb * stride_ws_k)
        acc += tl.dot(x, w) * (xs[:, :, None] * ws[:, None, None])
    return tl.sum(acc, axis=0)


@triton.jit
def _wg_kernel(
    x_ptr,
    stride_x,
    xs_ptr,
    stride_xs_m,
    stride_xs_k,
    w_ptr,
    ws_ptr,
    stride_ws_n,
    stride_ws_k,
    out_ptr,
    stride_out,
    M,
    K: tl.constexpr,
    S: tl.constexpr,
    GROUP: tl.constexpr,
    BN: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = wg_dot(
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
    )
    m = tl.arange(0, ROWS)
    n = pid * BN + tl.arange(0, BN)
    out = out_ptr + m[:, None] * stride_out + n[None, :]
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=m[:, None] < M)


def wg_rows(rows: int) -> int:
    """The row tile of ``block_fp8_gemv_wg`` for ``rows`` rows."""
    return WG_ROWS if rows <= WG_ROWS else WG_MAX_ROWS


def wg_fits(rows: int, k: int, weight: torch.Tensor) -> bool:
    """Whether ``block_fp8_gemv_wg`` serves ``rows`` of K ``k`` on ``weight``."""
    return (
        0 < rows <= WG_MAX_ROWS
        and _shuffled(weight)
        and k % (WG_SPLIT * BLOCK) == 0
        and weight.shape[0] % WG_BN == 0
    )


def block_fp8_gemv_wg(
    x: torch.Tensor, xq, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``block_fp8_gemv_q`` with K split across the warps of each workgroup."""
    m, k = x.shape
    if not wg_fits(m, k, weight):
        return block_fp8_gemv_q(x, xq, weight, scale)
    q, q_scale = xq
    out = torch.empty(m, weight.shape[0], dtype=x.dtype, device=x.device)
    _wg_kernel[(weight.shape[0] // WG_BN,)](
        q,
        q.stride(0),
        q_scale,
        q_scale.stride(0),
        q_scale.stride(1),
        weight,
        scale,
        scale.stride(0),
        scale.stride(1),
        out,
        out.stride(0),
        m,
        K=k,
        S=WG_SPLIT,
        GROUP=BLOCK,
        BN=WG_BN,
        ROWS=wg_rows(m),
        num_warps=WG_WARPS,
    )
    return out


def _allow_block_fp8_linear() -> None:
    """The gfx950 small-M helper imports symbols its MoE module no longer defines.

    That import runs before the helper can decline the shape, so every block-FP8
    linear dies with it. Publishing the names lets the import finish; disabling
    the helper then takes the same block-scale GEMM the larger shapes use. A
    module that still defines them is left alone, helper included.
    """
    import importlib

    moe = importlib.import_module("sglang.kernels.ops.moe.smallm_moe_gfx950")
    if hasattr(moe, "_check"):
        return
    os.environ.setdefault("SGLANG_ROCM_SMALLM_FP8_BS", "0")

    class _Kernel:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("small-M block-FP8 helper is unavailable")

    moe._Kernel = _Kernel
    moe._check = lambda *_args, **_kwargs: None
    moe._hip_lib = lambda: None
    moe._hipcc = lambda: "hipcc"


def _aiter_gemv(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    from sglang.srt.layers.quantization.fp8_utils import aiter_w8a8_block_fp8_linear

    _allow_block_fp8_linear()
    return aiter_w8a8_block_fp8_linear(x, weight, [BLOCK, BLOCK], scale)


def _quantize_rows(x: torch.Tensor) -> torch.Tensor:
    """Dynamic 1x128 e4m3, the same group the block-scale GEMM quantizes."""
    m, k = x.shape
    groups = x.float().reshape(m, k // BLOCK, BLOCK)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / FP8_MAX)
    q = (groups / scale).clamp(-FP8_MAX, FP8_MAX)
    q = q.to(torch.float8_e4m3fn).float() * scale
    return q.reshape(m, k)


def _plain_gemv(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Row-major block-FP8. gfx950 serves the preshuffled weight through aiter."""
    w = weight.view(torch.float8_e4m3fn).float()
    ws = scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return (_quantize_rows(x) @ (w * ws).transpose(0, 1)).to(dtype=x.dtype)
