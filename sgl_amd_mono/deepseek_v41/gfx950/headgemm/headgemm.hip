// SPDX-License-Identifier: Apache-2.0
// The target's vocab projection of a verify's few rows on this rank's shard: out [M, N] bf16 = x [M, K] bf16 times
// W [N, K] bf16 transposed, fp32 accumulation in bf16 MFMAs (16x16x32), rounded to bf16 once, as the head GEMM's
// output dtype. A wave a 16-column tile over the whole K: each lane streams 16 V bytes of a weight row an iteration
// (V MFMAs' B operands, the K order permuted within the tile's sum), DEPTH iterations' loads in flight. Not wired
// into serving. Best so far, variant 13 (a row's 4 lanes contiguous in each load, 8 vectors a lane an iteration,
// 2 iterations ahead): 53.8 us a TP4 shard at 6 rows against hipBLASLt's 51.7.
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
using bf16x8   = __bf16 __attribute__((ext_vector_type(8)));
using f32x4    = float __attribute__((ext_vector_type(4)));

constexpr int WAVE    = 64;
constexpr int TILE_N  = 16;
constexpr int MAX_M   = 16;

template <int V> struct alignas(16) Chunk {
    bf16x8 v[V]; // V x 16 B: 8 V consecutive K of one row
};

// vector j of a lane: 16 B at p + j * GAP bf16 (GAP 8: one contiguous run a lane; 32: a row's 4 lanes contiguous
// in each load instruction)
template <int V, bool NT, int GAP> __device__ inline Chunk<V> load_w(const bfloat16 *p) {
    Chunk<V> c;
#pragma unroll
    for (int j = 0; j < V; ++j) {
        if constexpr (NT) {
            c.v[j] = __builtin_nontemporal_load(reinterpret_cast<const bf16x8 *>(p + j * GAP));
        } else {
            c.v[j] = *reinterpret_cast<const bf16x8 *>(p + j * GAP);
        }
    }
    return c;
}

// x staged in LDS once a block ([M][K], 64 B a lane an iteration): the A operands' reads count on lgkmcnt, so the
// weight loads (vmcnt, returning in order) stay DEPTH iterations ahead
template <int DEPTH, int NWAVES, int V, bool NT, bool IL>
__global__ __launch_bounds__(NWAVES * WAVE) void head_gemm_kernel(const bfloat16 *__restrict__ x, int64_t x_stride,
                                                                  const bfloat16 *__restrict__ w, int64_t w_stride,
                                                                  bfloat16 *__restrict__ out, int64_t out_stride,
                                                                  int M, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    bfloat16 *xs = reinterpret_cast<bfloat16 *>(smem);
    const int wave = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
    const int tile = blockIdx.x * NWAVES + wave;
    const int n0   = tile * TILE_N;
    const int col = lane % TILE_N, grp = lane / TILE_N;
    const int n   = min(n0 + col, N - 1);
    constexpr int   STEP  = 4 * 8 * V; // K an iteration: 4 lane groups x 8 V
    constexpr int   GAP   = IL ? 32 : 8;
    const int       koff  = IL ? grp * 8 : grp * 8 * V;
    const bfloat16 *wrow  = w + static_cast<int64_t>(n) * w_stride + koff;
    const int       iters = K / STEP;
    Chunk<V>        buf[DEPTH];
    // the first weights before the x staging
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
        if (d < iters) {
            buf[d] = load_w<V, NT, GAP>(wrow + d * STEP);
        }
    }
    for (int e = threadIdx.x * 8; e < M * K; e += NWAVES * WAVE * 8) {
        const int r = e / K, k = e % K;
        *reinterpret_cast<bf16x8 *>(xs + e) = *reinterpret_cast<const bf16x8 *>(x + static_cast<int64_t>(r) * x_stride + k);
    }
    __syncthreads();
    if (n0 >= N) {
        return;
    }
    const int       m    = col;
    const bool      live = m < M;
    const bfloat16 *xrow = xs + (live ? m : 0) * K + koff;
    f32x4           acc  = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int it = 0; it < iters; it += DEPTH) {
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            const int i = it + d;
            if (i < iters) {
                const Chunk<V> wb = buf[d];
                if (i + DEPTH < iters) {
                    buf[d] = load_w<V, NT, GAP>(wrow + (i + DEPTH) * STEP);
                }
                Chunk<V> xa;
                if (live) {
#pragma unroll
                    for (int j = 0; j < V; ++j) {
                        xa.v[j] = *reinterpret_cast<const bf16x8 *>(xrow + i * STEP + j * GAP);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < V; ++j) {
                        xa.v[j] = bf16x8{};
                    }
                }
#pragma unroll
                for (int j = 0; j < V; ++j) {
                    acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(xa.v[j], wb.v[j], acc, 0, 0, 0);
                }
            }
        }
    }
    // C: lane holds rows 4 * grp .. + 4 of column col
    if (n0 + col < N) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = 4 * grp + r;
            if (row < M) {
                out[static_cast<int64_t>(row) * out_stride + n0 + col] = static_cast<bfloat16>(acc[r]);
            }
        }
    }
}

void check(const at::Tensor &t, const at::Tensor &ref, const char *name) {
    TORCH_CHECK(t.device() == ref.device() && t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.stride(1) == 1,
                name, " must be 2-D bf16 with unit inner stride on ", ref.device());
}

// x [M <= 16, K] bf16, w [N, K] bf16 (rows 16 B aligned), out [M, N] bf16; K a multiple of 256
void head_gemm(const at::Tensor &x, const at::Tensor &w, at::Tensor &out, int64_t depth) {
    check(x, x, "x");
    check(w, x, "w");
    check(out, x, "out");
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(M >= 1 && M <= MAX_M && w.size(1) == K && K % 256 == 0 && out.size(0) >= M && out.size(1) >= N,
                "x [M <= 16, K], w [N, K], out [M, N], K a multiple of 256");
    TORCH_CHECK(x.stride(0) % 8 == 0 && w.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "16 B aligned rows");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(x.device().index()).stream();
    const int tiles = static_cast<int>((N + TILE_N - 1) / TILE_N);
    const auto *xp  = static_cast<const bfloat16 *>(x.data_ptr());
    const auto *wp  = static_cast<const bfloat16 *>(w.data_ptr());
    auto       *op  = static_cast<bfloat16 *>(out.data_ptr());
    const size_t lds = static_cast<size_t>(M) * K * 2;
    TORCH_CHECK(lds <= 64 * 1024 && x.stride(0) == K, "x [M, K] contiguous, at most 64 KB");
#define LAUNCH(D, W, V, NT, IL)                                                                                          \
    head_gemm_kernel<D, W, V, NT, IL><<<(tiles + W - 1) / W, W * WAVE, lds, stream>>>(                                  \
        xp, x.stride(0), wp, w.stride(0), op, out.stride(0), static_cast<int>(M), static_cast<int>(N),             \
        static_cast<int>(K))
    // variant: depth, waves a block, 16 B vectors a lane an iteration, non-temporal weight loads
    switch (depth) {
    case 0: LAUNCH(4, 8, 4, false, false); break;
    case 1: LAUNCH(4, 8, 4, true, false); break;
    case 2: LAUNCH(2, 8, 8, false, false); break;
    case 3: LAUNCH(2, 8, 8, true, false); break;
    case 4: LAUNCH(4, 8, 8, true, false); break;
    case 5: LAUNCH(3, 8, 8, true, false); break;
    case 6: LAUNCH(4, 4, 8, true, false); break;
    case 7: LAUNCH(8, 8, 4, true, false); break;
    case 8: LAUNCH(4, 16, 4, false, false); break;
    case 9: LAUNCH(6, 16, 4, false, false); break;
    case 10: LAUNCH(4, 8, 4, false, true); break;
    case 11: LAUNCH(6, 8, 4, false, true); break;
    case 12: LAUNCH(4, 4, 4, false, true); break;
    case 13: LAUNCH(2, 8, 8, false, true); break;
    case 14: LAUNCH(3, 8, 8, false, true); break;
    default: TORCH_CHECK(false, "variant 0 to 14");
    }
#undef LAUNCH
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("head_gemm(Tensor x, Tensor w, Tensor(a!) out, int depth) -> ()");
    m.impl("head_gemm", at::kCUDA, &head_gemm);
}
