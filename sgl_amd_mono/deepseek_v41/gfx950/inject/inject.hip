// SPDX-License-Identifier: Apache-2.0
// Adapted from AKLR (https://github.com/AMD-AGI/AKLR, MIT, Copyright (C) 2025-2026 Advanced Micro Devices, Inc.).
// Every DSpark draft stage's SWA KV of a verify's committed target hidden in one launch (grid rows x stages), the bytes
// of SGLang's fused_k_norm_rope_flashmla (main_norm_rope.cuh) per stage: the same sums in the same order.
#include <cstdint>
#include <optional>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#ifndef C10_HIP_KERNEL_LAUNCH_CHECK
#define C10_HIP_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

namespace {

using bfloat16 = __bf16;

constexpr int DIM = 512, ROPE = 64, MAX_STAGES = 4;
constexpr int THREADS = DIM / 2; // a thread 2 dims
constexpr int LANES = 32;        // SGLang's warp
constexpr int WARPS = THREADS / LANES;
constexpr int ROPE_WARP = WARPS - 1;

struct Params {
    const bfloat16 *kv; // stage s's row r at kv + r * kv_stride + s * stage_stride
    int64_t kv_stride, stage_stride;
    const bfloat16 *weight[MAX_STAGES];
    const float *freqs[MAX_STAGES]; // [positions, ROPE] cos / sin pairs
    uint8_t *cache[MAX_STAGES];     // [pages, page_bytes]: a page's rows, then their scale rows
    const void *pos;
    bool pos_i32, v41;
    const int32_t *loc; // null: the committed slots below
    const int64_t *cache_loc, *map;
    const int32_t *commit;
    int64_t map_n;
    int width, rows, stages, page_bits;
    int64_t page_bytes;
    float eps;
};

__device__ inline float bf16_round(float v) { return static_cast<float>(static_cast<bfloat16>(v)); }

template <int W> __device__ inline float lanes_sum(float v) {
#pragma unroll
    for (int m = W / 2; m >= 1; m >>= 1) v = v + __shfl_xor(v, m, W);
    return v;
}

template <int W> __device__ inline float lanes_max(float v) {
#pragma unroll
    for (int m = W / 2; m >= 1; m >>= 1) v = fmaxf(v, __shfl_xor(v, m, W));
    return v;
}

__device__ inline uint16_t pack_e4m3(float x, float y) {
    x = fmaxf(fminf(x, 448.0f), -448.0f);
    y = fmaxf(fminf(y, 448.0f), -448.0f);
    return static_cast<uint16_t>(__builtin_amdgcn_cvt_pk_fp8_f32(x, y, 0, false) & 0xffff);
}

__device__ inline float inv_scale_ue8m0(int32_t exponent) {
    return __uint_as_float(static_cast<uint32_t>(127 + 127 - exponent) << 23);
}

template <bool V41> __global__ void __launch_bounds__(THREADS) swa_store_kernel(const Params p) {
    constexpr int64_t ROW = V41 ? 512 : 576, SCALES = V41 ? 16 : 8;
    const int row = blockIdx.x, stage = blockIdx.y, tx = threadIdx.x;
    const int warp = tx / LANES, lane = tx % LANES;
    const bfloat16 *x = p.kv + row * p.kv_stride + stage * p.stage_stride;
    const int64_t position = p.pos_i32 ? static_cast<const int32_t *>(p.pos)[row] : static_cast<const int64_t *>(p.pos)[row];
    int loc = -1;
    if (p.loc != nullptr) {
        loc = p.loc[row];
    } else if (row % p.width < p.commit[row / p.width]) {
        int64_t c = p.cache_loc[row];
        loc = static_cast<int32_t>(p.map[c < 0 ? c + p.map_n : c]);
    }
    __shared__ float partial[WARPS];
    float data[2], freq[2] = {0.0f, 0.0f};
    {
        const float x0 = static_cast<float>(x[2 * tx]), x1 = static_cast<float>(x[2 * tx + 1]);
        const float w0 = static_cast<float>(p.weight[stage][2 * tx]);
        const float w1 = static_cast<float>(p.weight[stage][2 * tx + 1]);
        if (warp == ROPE_WARP) {
            const float *f = p.freqs[stage] + static_cast<int32_t>(position) * ROPE;
            freq[0] = f[2 * lane];
            freq[1] = f[2 * lane + 1];
        }
        float ss = fmaf(x0, x0, 0.0f);
        ss = fmaf(x1, x1, ss);
        const float warp_sum = lanes_sum<LANES>(ss);
        if (lane == 0) partial[warp] = warp_sum;
        __syncthreads();
        ss = lanes_sum<WARPS>(partial[lane % WARPS]);
        const float norm = rsqrtf(ss / DIM + p.eps);
        data[0] = x0 * norm * w0;
        data[1] = x1 * norm * w1;
    }
    if (loc < 0) return;
    const int64_t page = loc >> p.page_bits, slot = loc & ((1 << p.page_bits) - 1);
    uint8_t *base = p.cache[stage] + page * p.page_bytes;
    uint8_t *data_row = base + slot * ROW;
    uint8_t *scale_row = base + (ROW << p.page_bits) + slot * SCALES;
    if constexpr (V41) {
        float v0 = bf16_round(data[0]), v1 = bf16_round(data[1]);
        if (warp == ROPE_WARP) {
            const float re = v0, im = v1;
            v0 = bf16_round(fmaf(re, freq[0], -(im * freq[1])));
            v1 = bf16_round(fmaf(re, freq[1], im * freq[0]));
        }
        constexpr int TILE = 16; // a 32-value tile's lanes
        const float amax = lanes_max<TILE>(fmaxf(fabsf(v0), fabsf(v1)));
        const uint32_t bits = __float_as_uint(amax);
        const int32_t exponent =
            max(static_cast<int32_t>(bits >> 23) - 8 + static_cast<int32_t>((bits & 0x7FFFFFu) > 0x600000u), 114);
        const float inv = inv_scale_ue8m0(exponent);
        reinterpret_cast<uint16_t *>(data_row)[tx] = pack_e4m3(v0 * inv, v1 * inv);
        if (tx % TILE == 0) scale_row[tx / TILE] = static_cast<uint8_t>(exponent);
    } else if (warp == ROPE_WARP) {
        const float re = data[0], im = data[1];
        bfloat16 *rope = reinterpret_cast<bfloat16 *>(data_row + 448);
        rope[2 * lane] = static_cast<bfloat16>(re * freq[0] - im * freq[1]);
        rope[2 * lane + 1] = static_cast<bfloat16>(re * freq[1] + im * freq[0]);
    } else {
        const float amax = lanes_max<LANES>(fmaxf(fabsf(data[0]), fabsf(data[1])));
        const float scale = fmaxf(1e-4f, amax) / 448.0f;
        const uint32_t u = __float_as_uint(scale);
        const int32_t exponent = static_cast<int32_t>((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0);
        const float inv = inv_scale_ue8m0(exponent);
        reinterpret_cast<uint16_t *>(data_row)[tx] = pack_e4m3(data[0] * inv, data[1] * inv);
        if (lane == 0) scale_row[warp] = static_cast<uint8_t>(exponent);
    }
}

// kv [rows, >= (stages - 1) * stage_stride + 512] bf16, weights / freqs / caches a stage each, pos [rows]
// int64 / int32, loc [rows] int32 (< 0: no slot); or cache_loc [rows] int64, mapping int64, commit int32 by width
void swa_store_op(const at::Tensor &kv, int64_t stage_stride, at::TensorList weights, at::TensorList freqs,
                  const at::Tensor &pos, const std::optional<at::Tensor> &loc, at::TensorList caches, double eps,
                  int64_t page_size, bool v41, const std::optional<at::Tensor> &cache_loc,
                  const std::optional<at::Tensor> &mapping, const std::optional<at::Tensor> &commit, int64_t width) {
    const int64_t stages = caches.size(), rows = kv.size(0);
    TORCH_CHECK(0 < stages && stages <= MAX_STAGES && weights.size() == caches.size() && freqs.size() == caches.size());
    TORCH_CHECK(kv.dim() == 2 && kv.scalar_type() == at::kBFloat16 && kv.stride(1) == 1 && stage_stride >= 0);
    TORCH_CHECK(pos.dim() == 1 && pos.size(0) == rows && pos.is_contiguous() &&
                (pos.scalar_type() == at::kLong || pos.scalar_type() == at::kInt));
    TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(kv.device());
    Params p{};
    for (int64_t s = 0; s < stages; ++s) {
        TORCH_CHECK(weights[s].scalar_type() == at::kBFloat16 && weights[s].numel() == DIM && weights[s].is_contiguous());
        TORCH_CHECK(freqs[s].scalar_type() == at::kFloat && freqs[s].dim() == 2 && freqs[s].size(1) == ROPE &&
                    freqs[s].is_contiguous());
        TORCH_CHECK(caches[s].dim() == 2 && caches[s].scalar_type() == at::kByte && caches[s].stride(1) == 1 &&
                    caches[s].stride(0) == caches[0].stride(0));
        p.weight[s] = static_cast<const bfloat16 *>(weights[s].data_ptr());
        p.freqs[s] = freqs[s].data_ptr<float>();
        p.cache[s] = static_cast<uint8_t *>(caches[s].data_ptr());
    }
    if (cache_loc.has_value()) {
        TORCH_CHECK(mapping.has_value() && commit.has_value() && width > 0 && rows % width == 0);
        TORCH_CHECK(cache_loc->scalar_type() == at::kLong && cache_loc->is_contiguous() && cache_loc->numel() == rows);
        TORCH_CHECK(mapping->scalar_type() == at::kLong && mapping->is_contiguous());
        TORCH_CHECK(commit->scalar_type() == at::kInt && commit->is_contiguous() && commit->numel() >= rows / width);
        p.cache_loc = cache_loc->data_ptr<int64_t>();
        p.map = mapping->data_ptr<int64_t>();
        p.map_n = mapping->numel();
        p.commit = commit->data_ptr<int32_t>();
        p.width = static_cast<int>(width);
    } else {
        TORCH_CHECK(loc.has_value() && loc->scalar_type() == at::kInt && loc->is_contiguous() && loc->numel() == rows);
        p.loc = loc->data_ptr<int32_t>();
    }
    p.kv = static_cast<const bfloat16 *>(kv.data_ptr());
    p.kv_stride = kv.stride(0);
    p.stage_stride = stage_stride;
    p.pos = pos.data_ptr();
    p.pos_i32 = pos.scalar_type() == at::kInt;
    p.v41 = v41;
    p.rows = static_cast<int>(rows);
    p.stages = static_cast<int>(stages);
    p.page_bits = __builtin_ctzll(page_size);
    p.page_bytes = caches[0].stride(0);
    p.eps = static_cast<float>(eps);
    if (rows == 0) return;
    const auto kernel = v41 ? swa_store_kernel<true> : swa_store_kernel<false>;
    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(kv.device().index()).stream();
    hipLaunchKernelGGL(kernel, dim3(p.rows, p.stages), dim3(THREADS), 0, stream, p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("swa_store(Tensor kv, int stage_stride, Tensor[] weights, Tensor[] freqs, Tensor pos, Tensor? loc, "
          "Tensor(a!)[] caches, float eps, int page_size, bool v41, Tensor? cache_loc=None, Tensor? mapping=None, "
          "Tensor? commit=None, int width=0) -> ()");
    m.impl("swa_store", at::kCUDA, &swa_store_op);
}
