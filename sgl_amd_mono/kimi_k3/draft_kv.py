# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's DSpark draft KV from the stacked K|V projection of a verify window, in one launch: every
draft layer's K norm (fp32), its bf16 cast, the NeoX rotary embedding of the bf16 K, and the FP8 store
of K and V into each layer's pool rows, for the rows each request commits. Bitwise SGLang's chain
(``DSparkDraftMixin._project_ctx_kv_stacked``, ``rotary_embedding_kernel``, then one
``set_kv_buffer_prefix_valid_tiled_fp8`` a layer): the same fp32 operations in the same order, the K
norm's sum of squares in torch's mean-reduction tree over the head (adjacent pairs, then pairs of
pairs), the rotary products rounded to bf16 as c10::BFloat16 arithmetic does."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.kvcache.cache_move import _saturate_to_fp8_range

HEAD_DIM = 64
HALF = HEAD_DIM // 2


@triton.jit
def _tree32(x):
    """Sum over the last axis (32) of [H, 32] in adjacent pairs, then pairs of pairs."""
    h: tl.constexpr = x.shape[0]
    x = tl.sum(tl.reshape(x, (h, 16, 2)), axis=2)
    x = tl.sum(tl.reshape(x, (h, 8, 2)), axis=2)
    x = tl.sum(tl.reshape(x, (h, 4, 2)), axis=2)
    x = tl.sum(tl.reshape(x, (h, 2, 2)), axis=2)
    return tl.sum(x, axis=1)


@triton.jit
def _fp8(val, scale, SCALE_IS_TENSOR: tl.constexpr, FP8_MAX: tl.constexpr, OUT_DTYPE: tl.constexpr):
    """An FP32 value of a bf16 row to the pool's FP8, as set_kv_buffer_prefix_valid_tiled_fp8 converts."""
    if SCALE_IS_TENSOR:
        val = tl.div_rn(val, scale.to(tl.bfloat16).to(tl.float32))
    else:
        val = val * tl.div_rn(1.0, scale)
    val = val.to(tl.bfloat16)
    val = _saturate_to_fp8_range(val, FP8_MAX, tl.bfloat16)
    return val.to(OUT_DTYPE)


@triton.jit
def _draft_kv_store_kernel(
    kv_ptr,  # [rows, L, 2, H * D] bf16: the stacked K|V projection
    kv_stride,
    pos_ptr,  # [rows] int64
    loc_ptr,  # [bs, W] int64 pool rows (W = rows a request)
    commit_ptr,  # [bs] int32
    knw_ptr,  # [L, D] fp32 K norm weights
    cos_sin_ptr,  # [max_pos, D] bf16: cos | sin
    k_ptrs,  # [L] int64 K pool base addresses (FP8)
    v_ptrs,  # [L] int64 V pool base addresses (FP8)
    k_scale,
    v_scale,
    k_row_stride,
    v_row_stride,
    eps,
    W: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    FP8_MAX: tl.constexpr,
    SCALE_IS_TENSOR: tl.constexpr,
):
    row = tl.program_id(0)
    layer = tl.program_id(1)
    req = row // W
    col = row % W
    if col >= tl.load(commit_ptr + req):
        return
    HALF_D: tl.constexpr = D // 2
    h = tl.arange(0, H)[:, None]
    d = tl.arange(0, HALF_D)[None, :]
    base = kv_ptr + row * kv_stride + layer * (2 * H * D)
    kx = tl.load(base + h * D + d).to(tl.float32)
    ky = tl.load(base + h * D + HALF_D + d).to(tl.float32)
    # variance = mean over the head of k * k (pow 2), the tree across the two halves last
    var = (_tree32(kx * kx) + _tree32(ky * ky)) * (1.0 / D)
    r = tl.math.rsqrt(var + eps)[:, None]
    wx = tl.load(knw_ptr + layer * D + d)
    wy = tl.load(knw_ptr + layer * D + HALF_D + d)
    x = ((kx * r) * wx).to(tl.bfloat16).to(tl.float32)
    y = ((ky * r) * wy).to(tl.bfloat16).to(tl.float32)
    pos = tl.load(pos_ptr + row)
    c = tl.load(cos_sin_ptr + pos * D + d).to(tl.float32)
    s = tl.load(cos_sin_ptr + pos * D + HALF_D + d).to(tl.float32)
    xc = (x * c).to(tl.bfloat16).to(tl.float32)
    ys = (y * s).to(tl.bfloat16).to(tl.float32)
    yc = (y * c).to(tl.bfloat16).to(tl.float32)
    xs = (x * s).to(tl.bfloat16).to(tl.float32)
    kx_out = (xc - ys).to(tl.bfloat16).to(tl.float32)
    ky_out = (yc + xs).to(tl.bfloat16).to(tl.float32)
    vx = tl.load(base + H * D + h * D + d).to(tl.float32)
    vy = tl.load(base + H * D + h * D + HALF_D + d).to(tl.float32)
    ks = k_scale
    vs = v_scale
    k_base = tl.load(k_ptrs + layer).to(tl.pointer_type(tl.float8e4nv))
    v_base = tl.load(v_ptrs + layer).to(tl.pointer_type(tl.float8e4nv))
    loc = tl.load(loc_ptr + req * W + col)
    ko = k_base + loc * k_row_stride + h * D + d
    vo = v_base + loc * v_row_stride + h * D + d
    tl.store(ko, _fp8(kx_out, ks, SCALE_IS_TENSOR, FP8_MAX, tl.float8e4nv))
    tl.store(ko + HALF_D, _fp8(ky_out, ks, SCALE_IS_TENSOR, FP8_MAX, tl.float8e4nv))
    tl.store(vo, _fp8(vx, vs, SCALE_IS_TENSOR, FP8_MAX, tl.float8e4nv))
    tl.store(vo + HALF_D, _fp8(vy, vs, SCALE_IS_TENSOR, FP8_MAX, tl.float8e4nv))


class DraftKvStore:
    """One draft's pool rows and constants for ``store``: the layers' K and V pool base addresses
    (FP8, ``[slots, H, D]`` rows), K norm weights, epsilon, the rotary cos|sin cache, and the K/V
    scales as the FP8 store takes them (a float; ``scale_is_tensor``: the layer's scale is a device
    tensor, so the store divides by its bf16 value)."""

    def __init__(self, k_bufs, v_bufs, k_norm_weight, eps, cos_sin_cache, k_scale, v_scale, scale_is_tensor):
        dev = k_norm_weight.device
        self.layers = len(k_bufs)
        self.heads, self.dim = k_bufs[0].shape[1], k_bufs[0].shape[2]
        assert self.dim == HEAD_DIM and all(b.dtype == torch.float8_e4m3fn for b in (*k_bufs, *v_bufs))
        assert all(b.stride(1) == self.dim and b.stride(2) == 1 for b in (*k_bufs, *v_bufs))
        self.k_row_stride = k_bufs[0].stride(0)
        self.v_row_stride = v_bufs[0].stride(0)
        assert all(b.stride(0) == self.k_row_stride for b in k_bufs)
        assert all(b.stride(0) == self.v_row_stride for b in v_bufs)
        self.k_ptrs = torch.tensor([b.data_ptr() for b in k_bufs], dtype=torch.int64, device=dev)
        self.v_ptrs = torch.tensor([b.data_ptr() for b in v_bufs], dtype=torch.int64, device=dev)
        self.knw = k_norm_weight.to(torch.float32).reshape(self.layers, self.dim).contiguous()
        self.eps = float(eps)
        assert cos_sin_cache.shape[1] == self.dim
        # the rotary kernel reads cos and sin as the bf16 K's dtype
        self.cos_sin = cos_sin_cache.to(torch.bfloat16).contiguous()
        self.k_scale, self.v_scale = float(k_scale), float(v_scale)
        self.scale_is_tensor = bool(scale_is_tensor)

    def store(self, kv: torch.Tensor, positions: torch.Tensor, loc_2d: torch.Tensor, commit_lens: torch.Tensor):
        """kv [rows, L * 2 * H * D] bf16 (the stacked projection), positions [rows] int64, loc_2d
        [bs, W] int64 pool rows, commit_lens [bs] int32: rows ``col < commit_lens[req]`` stored."""
        rows = kv.shape[0]
        bs, w = loc_2d.shape
        assert rows == bs * w and kv.shape[1] == self.layers * 2 * self.heads * self.dim and kv.stride(1) == 1
        assert positions.dtype == torch.int64 and loc_2d.dtype == torch.int64 and commit_lens.dtype == torch.int32
        _draft_kv_store_kernel[(rows, self.layers)](
            kv, kv.stride(0), positions, loc_2d, commit_lens, self.knw, self.cos_sin, self.k_ptrs, self.v_ptrs,
            self.k_scale, self.v_scale, self.k_row_stride, self.v_row_stride,
            self.eps, W=w, H=self.heads, D=self.dim, L=self.layers, FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
            SCALE_IS_TENSOR=self.scale_is_tensor, num_warps=1, enable_fp_fusion=False,
        )
