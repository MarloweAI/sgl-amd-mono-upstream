// SPDX-License-Identifier: Apache-2.0
// The head's input in one launch: a layer's owed mHC post settled into its streams and their combine by the
// pre mix (fused.py's _post_combine_kernel), then the RMSNorm of the combined row (aiter's
// add_rmsnorm_quant_kernel without residual or quant, as SGLang's RMSNorm runs it at widths up to 6144: 256
// threads a row, 24 values a thread in three 16 B chunks 512 values apart, its DPP block sum and rsqrtf; built
// with aiter's -ffast-math and flushed denormals, see __init__.py).
#include <cstdint>

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
using uint32x4 = uint32_t __attribute__((ext_vector_type(4)));

constexpr int THREADS = 256;
constexpr int WAVE = 64;
constexpr int WAVES = THREADS / WAVE;
constexpr int CHUNK = 8;      // a 16 B chunk's values
constexpr int CHUNKS = 3;     // a thread's chunks of a row
constexpr int HC = 4;         // the mHC streams
constexpr int MAX_N = THREADS * CHUNKS * CHUNK;

template <int CTRL>
__device__ inline float dpp(float v) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), CTRL, 0xf, 0xf, false));
}

// aiter's wave_reduce of a sum over 64 lanes (no broadcast): the last lane holds it
__device__ inline float wave_sum64(float v) {
    v = dpp<0xb1>(v) + v;
    v = dpp<0x4e>(v) + v;
    v = dpp<0x141>(v) + v;
    v = dpp<0x140>(v) + v;
    v = dpp<0x142>(v) + v;
    v = dpp<0x143>(v) + v;
    return v;
}

// aiter's wave_reduce of a sum over 4 lanes: every lane of a quad holds it
__device__ inline float quad_sum(float v) {
    v = dpp<0xb1>(v) + v;
    v = dpp<0x4e>(v) + v;
    return v;
}

__device__ inline void unpack(const uint32x4 &w, float (&f)[CHUNK]) {
    const bfloat16 *b = reinterpret_cast<const bfloat16 *>(&w);
#pragma unroll
    for (int k = 0; k < CHUNK; ++k) f[k] = static_cast<float>(b[k]);
}

__device__ inline uint32x4 pack(const float (&f)[CHUNK]) {
    uint32x4 w;
    bfloat16 *b = reinterpret_cast<bfloat16 *>(&w);
#pragma unroll
    for (int k = 0; k < CHUNK; ++k) b[k] = static_cast<bfloat16>(f[k]);
    return w;
}

// a rounded product the backend cannot fuse into a following add (-ffast-math fuses every mul and add)
__device__ inline float mul_rn(float a, float b) {
    float r;
    asm volatile("v_mul_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}

// aiter's two v_pk_mul_f32: (x rcp) w, rounded each
__device__ inline float mul2(float x, float rcp, float w) {
#pragma clang fp contract(off) reassociate(off)
    return (x * rcp) * w;
}

// _post_combine_kernel's arithmetic for 8 values of a row: each stream rounded to bf16 into s, and the
// combine's sum by the pre mix (a multiply, then an add) into acc
__device__ inline void combine8(const float (&xv)[CHUNK], const float (&r)[HC][CHUNK], const float (&pm)[HC],
                                const float (&cm)[HC][HC], const float (&qm)[HC], float (&s)[HC][CHUNK],
                                float (&acc)[CHUNK]) {
#pragma unroll
    for (int i = 0; i < HC; ++i) {
#pragma unroll
        for (int k = 0; k < CHUNK; ++k) {
            float v = __builtin_fmaf(xv[k], pm[i], r[0][k] * cm[0][i]);
#pragma unroll
            for (int j = 1; j < HC; ++j) v = __builtin_fmaf(r[j][k], cm[j][i], v);
            s[i][k] = static_cast<float>(static_cast<bfloat16>(v));
            const float p = mul_rn(qm[i], s[i][k]);
            acc[k] = i == 0 ? p : acc[k] + p;
        }
    }
}

// a row a block: streams [rows, HC, n] and normed [rows, n] out
__global__ __launch_bounds__(THREADS) void combine_norm_kernel(
    const bfloat16 *__restrict__ x, const bfloat16 *__restrict__ residual, const float *__restrict__ post,
    const float *__restrict__ comb, const float *__restrict__ pre, const bfloat16 *__restrict__ weight, double eps,
    int n, bfloat16 *__restrict__ streams, bfloat16 *__restrict__ normed) {
    __shared__ float partial[WAVES];
    const int m = blockIdx.x, lane = threadIdx.x % WAVE, wave = threadIdx.x / WAVE;
    const int64_t row = static_cast<int64_t>(m) * n;
    float pm[HC], cm[HC][HC], qm[HC];
#pragma unroll
    for (int i = 0; i < HC; ++i) {
        pm[i] = post[m * HC + i];
        qm[i] = pre[m * HC + i];
#pragma unroll
        for (int j = 0; j < HC; ++j) cm[j][i] = comb[(m * HC + j) * HC + i];
    }
    float y[CHUNKS * CHUNK];  // aiter's thread_data_float
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int h = wave * WAVE * CHUNKS * CHUNK + c * WAVE * CHUNK + lane * CHUNK;
        if (h < n) {
            float xv[CHUNK], r[HC][CHUNK];
            unpack(*reinterpret_cast<const uint32x4 *>(x + row + h), xv);
#pragma unroll
            for (int j = 0; j < HC; ++j) unpack(*reinterpret_cast<const uint32x4 *>(residual + row * HC + j * n + h), r[j]);
            float s[HC][CHUNK], acc[CHUNK];
            combine8(xv, r, pm, cm, qm, s, acc);
#pragma unroll
            for (int i = 0; i < HC; ++i) *reinterpret_cast<uint32x4 *>(streams + row * HC + i * n + h) = pack(s[i]);
#pragma unroll
            for (int k = 0; k < CHUNK; ++k) y[c * CHUNK + k] = static_cast<float>(static_cast<bfloat16>(acc[k]));
        } else {
#pragma unroll
            for (int k = 0; k < CHUNK; ++k) y[c * CHUNK + k] = 0.0f;
        }
    }
    float sq = 0.0f;
    for (int i = 0; i < CHUNKS * CHUNK; i++) {
        sq += (y[i] * y[i]);
    }
    sq = wave_sum64(sq);
    if (lane == WAVE - 1) partial[wave] = sq;
    __syncthreads();
    sq = quad_sum(partial[lane % WAVES]);
    const float rcp = rsqrtf(sq / n + eps);
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int h = wave * WAVE * CHUNKS * CHUNK + c * WAVE * CHUNK + lane * CHUNK;
        if (h < n) {
            float w[CHUNK], o[CHUNK];
            unpack(*reinterpret_cast<const uint32x4 *>(weight + h), w);
#pragma unroll
            for (int k = 0; k < CHUNK; ++k) o[k] = mul2(y[c * CHUNK + k], rcp, w[k]);
            *reinterpret_cast<uint32x4 *>(normed + row + h) = pack(o);
        }
    }
}

// x [rows, n] and residual [rows, HC, n] bf16, post [rows, HC], comb [rows, HC, HC] and pre [rows, HC] f32,
// weight [n] bf16; streams [rows, HC, n] and normed [rows, n] bf16 out
void combine_norm_op(const at::Tensor &x, const at::Tensor &residual, const at::Tensor &post, const at::Tensor &comb,
                     const at::Tensor &pre, const at::Tensor &weight, double eps, const at::Tensor &streams,
                     const at::Tensor &normed) {
    const int64_t rows = residual.size(0), n = residual.size(2);
    TORCH_CHECK(residual.dim() == 3 && residual.size(1) == HC && 4096 < n && n <= MAX_N && n % CHUNK == 0);
    for (const at::Tensor *t : {&x, &residual, &weight, &streams, &normed}) {
        TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->is_contiguous());
    }
    for (const at::Tensor *t : {&post, &comb, &pre}) {
        TORCH_CHECK(t->scalar_type() == at::kFloat && t->is_contiguous() && t->size(0) == rows);
    }
    TORCH_CHECK(x.numel() == rows * n && weight.numel() == n && streams.numel() == rows * HC * n &&
                normed.numel() == rows * n && post.numel() == rows * HC && pre.numel() == rows * HC &&
                comb.numel() == rows * HC * HC);
    if (rows == 0) return;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x.device());
    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(x.device().index()).stream();
    hipLaunchKernelGGL(combine_norm_kernel, dim3(rows), dim3(THREADS), 0, stream,
                       static_cast<const bfloat16 *>(x.data_ptr()), static_cast<const bfloat16 *>(residual.data_ptr()),
                       post.data_ptr<float>(), comb.data_ptr<float>(), pre.data_ptr<float>(),
                       static_cast<const bfloat16 *>(weight.data_ptr()), eps, static_cast<int>(n),
                       static_cast<bfloat16 *>(streams.data_ptr()), static_cast<bfloat16 *>(normed.data_ptr()));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("combine_norm(Tensor x, Tensor residual, Tensor post, Tensor comb, Tensor pre, Tensor weight, float eps, "
          "Tensor(a!) streams, Tensor(b!) normed) -> ()");
    m.impl("combine_norm", at::kCUDA, &combine_norm_op);
}
