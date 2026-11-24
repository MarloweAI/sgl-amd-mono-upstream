// SPDX-License-Identifier: Apache-2.0
// GLM-5.3-Flash's decode MoE front at TP2 (gfx950) in one launch, 1-256 rows: the router GEMM (bf16 x bf16, f32
// sums in a fixed order), the biased sigmoid top-8 of 288, the BM = 32 sort of the routes (with the shared expert
// as slot 288 when SHARED), the rows' MXFP4 activation quant and the zeroed MoE output.
//
// CTA roles, in id order (a CTA waits only on CTAs with lower ids):
// - router (g, n, s): row tiles g RB .. (RB = ceil(R / 4)), experts 16 n.., K slice s of 4; its 8 waves' MFMA sums
//   added in wave order. The last of the 4 slices of (g, n) (by counter) adds them in slice order -> f32 logits and
//   counts the expert tile done for each of its row tiles. Never waits.
// - quant: x (bf16) -> MXFP4 a1 [M, 2048 B] and its per-token e8m0 scales [M, 128]; zeroes the MoE output. Never
//   waits.
// - top-k: a wave a row, once its row tile's 18 expert tiles are counted; the last top-k CTA to finish sorts every
//   route and publishes the sort.
// - scale (with a1_scale_sorted only): once the sort and the quant are done, the per-token scales in the sorted,
//   shuffled layout aiter's FlyDSL stage 1 reads (aiter::mx_scale_shuffle_idx).
//
// Bit for bit what aiter writes, given the same inputs:
// - top-k: aiter's grouped_topk_kernel (topk_softmax_kernels_group.cu; biased, one group, f32x4 lanes) on the
//   logits rounded to bf16, SGLang's logits dtype: sigmoid rcpf(1 + exp2f(-C_LOG2E x)) with C_LOG2E a double
//   literal, + bias, eight rounds of the same DPP arg-max, the pre-bias sigmoids renormalized x scale;
// - quant: aiter's fused_mx_quant_moe_sort_kernel for fp4 (RCEIL e8m0 of the group amax, cvt_scalef32_pk_fp4).
// The sort is Opus's layout (experts ascending, an expert's routes in token order, pad rows (k << 24) | M with
// weight 0), in a capacity of M k + min(E, M k) (BM - 1) rows.
//
// Hand-offs between CTAs: data stored and loaded at agent scope (sc1), then s_waitcnt vmcnt(0) and a barrier before
// the counter's agent-scope add. Every counter and flag is put back to 0 by its last user, for graph replay.
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

// aiter's literal: the exponent is an fp64 product rounded once to f32
#define C_LOG2E 1.44269504088896340736

namespace {

constexpr int E        = 288;
constexpr int D        = 4096;
constexpr int TOPK     = 8;
constexpr int BM       = 32;
constexpr int NT       = E / 16; // router expert tiles
constexpr int THREADS  = 512;
constexpr int WAVE     = 64;
constexpr int WAVES    = THREADS / WAVE;
constexpr int MAX_ROWS = 256;
constexpr int MW       = MAX_ROWS / 32; // token-mask words an expert
constexpr int SCOLS    = D / 32;        // e8m0 scales a row
constexpr int SHARED_ID = E;            // the shared expert's slot id
// counters, a 128 B line each
constexpr int CL        = 32;
constexpr int C_RN      = 0;                   // [R * NT]: K slices of a router tile done
constexpr int C_R       = C_RN + 16 * NT;       // [R]: expert tiles of a row tile done
constexpr int C_TOP     = C_R + 16;             // row tiles with their top-k done
constexpr int C_SORTED  = C_TOP + 1;            // 1 once the sort is published
constexpr int C_QUANT   = C_SORTED + 1;         // quant CTAs done
constexpr int C_SCALE   = C_QUANT + 1;          // scale CTAs done
constexpr int N_COUNTERS = C_SCALE + 1;

typedef short short8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float floatx4 __attribute__((ext_vector_type(4)));

struct Params {
    const uint16_t *x;      // [M, D] bf16, row stride x_stride
    int64_t         x_stride;
    const uint16_t *w;      // [E, D] bf16 router weight
    const uint16_t *bias;   // [E] bf16
    float           scale;  // routed_scaling_factor, folded into the routed weights
    const int      *valid;  // or null: rows from *valid on are padding, routed nowhere
    float          *partial; // [G * NT * KSPLIT, RB * 256] f32 router partials, G = ceil(R / RB)
    float          *logits;  // [R * 16, E] f32
    int            *topk_ids; // [M, K]
    float          *topk_w;
    int            *sorted_ids; // [cap]
    float          *sorted_w;
    int            *sorted_eids; // [cap / BM]
    int            *num_valid;   // [2]
    uint8_t        *a1;          // [M, D / 2] fp4x2
    uint8_t        *a1_scale;    // [M, SCOLS] e8m0, a token's own
    uint8_t        *a1_scale_sorted; // [cap rounded to 32, SCOLS] shuffled, or null
    uint4          *out;         // the MoE output, zeroed: M x D bf16
    int            *counters;    // [N_COUNTERS * CL], 0 between launches
    int             M, R, RB, router_ctas, quant_ctas, topk_ctas, scale_ctas, pad_id;
};

__device__ __forceinline__ void st_dev(int *p, int v) {
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void st_dev(float *p, float v) {
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ int ld_dev(const int *p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ float ld_dev(const float *p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ uint32_t ld_dev(const uint32_t *p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ float bf16_f32(uint16_t v) { return __builtin_bit_cast(float, uint32_t(v) << 16); }
// round to nearest even, as torch's f32 -> bf16 (the logits are finite)
__device__ __forceinline__ uint16_t f32_bf16(float f) {
    const uint32_t u = __builtin_bit_cast(uint32_t, f);
    return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// This CTA's stores reach every XCD, then it counts itself done; true for the CTA that completes ``target``,
// which also puts the counter back to 0.
__device__ bool last_arrival(int *counter, int target, int &flag_lds) {
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    if (threadIdx.x == 0) {
        const int prev = __hip_atomic_fetch_add(counter, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const bool last = prev == target - 1;
        if (last) __hip_atomic_store(counter, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        flag_lds = last;
    }
    __syncthreads();
    return flag_lds != 0;
}

// ---- top-k: aiter's grouped_topk_kernel, one group, biased, f32x4 lanes, a wave a row ----


// aiter::wave_reduce<kvp, arg_max, 64, false> of warpReduceMax, then lane 63's pair
__device__ __forceinline__ void warp_reduce_max(float &val_o, int &idx) {
    float v = val_o;
    int   k = idx;
    auto step = [&](float rv, int rk) {
        // arg_max(a = the moved pair, b = this lane's): a.value > b.value ? a : b
        if (rv > v) {
            v = rv;
            k = rk;
        }
    };
#define AITER_DPP_STEP(CTRL)                                                                              \
    {                                                                                                     \
        const float rv = __builtin_bit_cast(                                                              \
            float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), CTRL, 0xf, 0xf, false));          \
        const int rk = __builtin_amdgcn_mov_dpp(k, CTRL, 0xf, 0xf, false);                                \
        step(rv, rk);                                                                                     \
    }
    AITER_DPP_STEP(0xb1)  // quad_perm:[1,0,3,2]
    AITER_DPP_STEP(0x4e)  // quad_perm:[2,3,0,1]
    AITER_DPP_STEP(0x141) // row_half_mirror
    AITER_DPP_STEP(0x140) // row_mirror
    AITER_DPP_STEP(0x142) // row_bcast:15
    AITER_DPP_STEP(0x143) // row_bcast:31
#undef AITER_DPP_STEP
    val_o = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), WAVE - 1));
    idx   = __builtin_amdgcn_readlane(k, WAVE - 1);
}

// One row's top 8 by one wave, in registers: lane l holds expert vectors l and l + 64 (4 experts each), the
// vectors aiter's 64-thread block scans in the same order; ``logit(v, i)`` gives the logit of expert 4 v + i as
// aiter reads it (bf16 widened to f32). Lanes 0..7 return their rank's expert and weight.
template <typename Logit>
__device__ void topk_row(Logit logit, const uint16_t *bias, float scale, int &out_id, float &out_w) {
    const int lane = threadIdx.x & (WAVE - 1);
    float sc[2][4], sg[2][4];
#pragma unroll
    for (int v = 0; v < 2; ++v) {
        const int  e    = lane + WAVE * v;
        const bool live = e < E / 4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float g  = live ? logit(v, i) : 0.0f;
            g        = __builtin_amdgcn_rcpf(1.0f + exp2f(-C_LOG2E * g));
            sg[v][i] = g; // pre-bias sigmoid = routing weight
            sc[v][i] = live ? g + bf16_f32(bias[e * 4 + i]) : -INFINITY;
        }
    }
    float sum = 0.0f;
    int   topk_indice = 0;
    float topk_value  = 0.0f;
    for (int k = 0; k < TOPK; ++k) {
        float max_val = -INFINITY;
        int   max_idx = k;
#pragma unroll
        for (int v = 0; v < 2; ++v)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (sc[v][i] > max_val) {
                    max_val = sc[v][i];
                    max_idx = (lane + WAVE * v) * 4 + i;
                }
            }
        warp_reduce_max(max_val, max_idx);
        // the chosen expert's lane, register; its pre-bias sigmoid, and its score out of the race
        const int owner = (max_idx >> 2) & (WAVE - 1), vv = max_idx >> 8, ii = max_idx & 3;
        float mine = 0.0f;
#pragma unroll
        for (int v = 0; v < 2; ++v)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const bool hit = v == vv && i == ii;
                mine           = hit ? sg[v][i] : mine;
                sc[v][i]       = hit && lane == owner ? -INFINITY : sc[v][i];
            }
        max_val     = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, mine), owner));
        topk_indice = lane == k ? max_idx : topk_indice;
        topk_value  = lane == k ? max_val : topk_value;
        sum += max_val;
    }
    sum    = sum > 0.0f ? scale / sum : 0.0f;
    out_id = topk_indice;
    out_w  = topk_value * sum;
}

// ---- the sort ----

constexpr int K_MAX       = TOPK + 1;
constexpr int MAX_PAIRS   = MAX_ROWS * K_MAX;
constexpr int PR          = (MAX_PAIRS + THREADS - 1) / THREADS; // routes a sorting thread
constexpr int MAX_SORTED  = (MAX_PAIRS + (E + 1) * (BM - 1) + BM - 1) / BM * BM;
constexpr int MAX_BLOCKS  = MAX_SORTED / BM;

struct SortLds {
    union {
        uint32_t mask[(E + 1) * MW];  // an expert's tokens
        uint16_t row_route[MAX_SORTED]; // then: a sorted row's route (token x K + slot)
    };
    float   w[MAX_PAIRS];
    int     cnt[E + 1];
    int     base[E + 1];
    int16_t block_e[MAX_BLOCKS];
    int     wsum[WAVES];
};

// Inclusive scan across a wave (row shifts, then row broadcasts).
__device__ __forceinline__ int wave_incl_scan(int x) {
    int t;
    t = __builtin_amdgcn_mov_dpp(x, 0x111, 0xF, 0xF, true);
    x += t;
    t = __builtin_amdgcn_mov_dpp(x, 0x112, 0xF, 0xF, true);
    x += t;
    t = __builtin_amdgcn_mov_dpp(x, 0x114, 0xF, 0xF, true);
    x += t;
    t = __builtin_amdgcn_mov_dpp(x, 0x118, 0xF, 0xF, true);
    x += t;
    t = __builtin_amdgcn_update_dpp(0, x, 0x142, 0xA, 0xF, true);
    x += t;
    t = __builtin_amdgcn_update_dpp(0, x, 0x143, 0xC, 0xF, true);
    x += t;
    return x;
}

__device__ int block_excl_scan(int v, SortLds &s, int &total) {
    const int lane = threadIdx.x & (WAVE - 1), wave = threadIdx.x / WAVE;
    const int x    = wave_incl_scan(v);
    if (lane == WAVE - 1) s.wsum[wave] = x;
    __syncthreads();
    int before = 0, all = 0;
#pragma unroll
    for (int i = 0; i < WAVES; ++i) {
        const int t = s.wsum[i];
        before += i < wave ? t : 0;
        all += t;
    }
    total = all;
    return before + x - v;
}

// Every route (topk_ids, -1: none) into BM-row blocks: experts ascending, an expert's routes in token order.
template <int K, int NE>
__device__ void sort_routes(const Params &p, SortLds &s) {
    const int tid = threadIdx.x;
    const int P   = p.M * K;
    const int mw  = (p.M + 31) / 32;
    for (int i = tid; i < NE * MW; i += THREADS) s.mask[i] = 0;
    __syncthreads();
    // every route's expert and weight, all loads in flight at once
    int   es[PR];
    float ws[PR];
#pragma unroll
    for (int k = 0; k < PR; ++k) {
        const int q = tid + k * THREADS;
        es[k]       = q < P ? ld_dev(p.topk_ids + q) : -1;
        ws[k]       = q < P ? ld_dev(p.topk_w + q) : 0.0f;
    }
#pragma unroll
    for (int k = 0; k < PR; ++k) {
        const int q = tid + k * THREADS;
        if (q < P) s.w[q] = ws[k];
        if (es[k] >= 0) {
            const int t = q / K;
            atomicOr(&s.mask[es[k] * MW + t / 32], 1u << (t % 32));
        }
    }
    __syncthreads();
    static_assert(THREADS >= NE, "an expert a sorting thread");
    const int e   = tid;
    int       cnt = 0;
    if (e < NE)
        for (int i = 0; i < mw; ++i) cnt += __popc(s.mask[e * MW + i]);
    const int blocks = (cnt + BM - 1) / BM;
    int       run;
    const int base = block_excl_scan(blocks, s, run);
    if (e < NE) {
        s.cnt[e]  = cnt;
        s.base[e] = base;
        for (int b = 0; b < blocks; ++b) s.block_e[base + b] = int16_t(e);
    }
    __syncthreads();
    const int nv = run * BM;
    int dest[PR];
#pragma unroll
    for (int k = 0; k < PR; ++k) {
        const int q = tid + k * THREADS;
        dest[k]     = -1;
        if (es[k] >= 0) {
            const int ex = es[k], t = q / K;
            int       rank = 0;
            for (int i = 0; i < t / 32; ++i) rank += __popc(s.mask[ex * MW + i]);
            rank += __popc(s.mask[ex * MW + t / 32] & ((1u << (t % 32)) - 1u));
            dest[k] = s.base[ex] * BM + rank;
        }
    }
    __syncthreads(); // the masks are read; their space takes the row -> route map
#pragma unroll
    for (int k = 0; k < PR; ++k)
        if (dest[k] >= 0) s.row_route[dest[k]] = uint16_t(tid + k * THREADS);
    __syncthreads();
    for (int b = tid; b < run; b += THREADS) p.sorted_eids[b] = s.block_e[b];
    for (int j = tid; j < nv; j += THREADS) {
        const int ex = s.block_e[j / BM];
        if (j - s.base[ex] * BM < s.cnt[ex]) {
            const int q = s.row_route[j], t = q / K;
            st_dev(p.sorted_ids + j, ((q % K) << 24) | t);
            p.sorted_w[j] = s.w[q];
        } else {
            st_dev(p.sorted_ids + j, p.pad_id);
            p.sorted_w[j] = 0.0f;
        }
    }
    if (tid == 0) {
        st_dev(p.num_valid, nv);
        st_dev(p.num_valid + 1, p.M);
    }
}

// ---- the router tiles, their top-k and the sort ----

constexpr int KSPLIT = 4;                    // K slices of the router: a wave takes 4 MFMA steps of 32
constexpr int KWAVE  = D / KSPLIT / WAVES;   // 128
constexpr int RB_MAX = 4;                    // row tiles a router CTA

union FrontLds {
    float   red[WAVES][RB_MAX][256];
    SortLds sort;
};

// 16 B at agent scope (sc1); the value is read only after vm_wait(value), which ties it to the wait
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ u32x4 ld_dev16(const void *ptr) {
    u32x4 v;
    asm volatile("global_load_dwordx4 %0, %1, off sc1" : "=v"(v) : "v"(ptr) : "memory");
    return v;
}
__device__ __forceinline__ void vm_wait(u32x4 &a) { asm volatile("s_waitcnt vmcnt(0)" : "+v"(a)::"memory"); }
__device__ __forceinline__ void vm_wait(u32x4 &a, u32x4 &b) {
    asm volatile("s_waitcnt vmcnt(0)" : "+v"(a), "+v"(b)::"memory");
}

// Router CTA (g, n, s): row tiles g RB .. g RB + RB - 1, experts 16 n .., K slice s.
__device__ void router_cta(const Params &p, int b, FrontLds &lds, int &flag) {
    const int tid = threadIdx.x, wave = tid / WAVE, lane = tid & (WAVE - 1);
    const int RB = p.RB;
    const int s = b % KSPLIT, gn = b / KSPLIT, n = gn % NT, g = gn / NT;
    const int k0 = s * (D / KSPLIT) + wave * KWAVE + (lane >> 4) * 8;
    const uint16_t *wb = p.w + int64_t(n * 16 + (lane & 15)) * D + k0;
    short8 bv[KWAVE / 32];
#pragma unroll
    for (int q = 0; q < KWAVE / 32; ++q) bv[q] = *reinterpret_cast<const short8 *>(wb + 32 * q);
    short8 av[RB_MAX][KWAVE / 32];
#pragma unroll
    for (int j = 0; j < RB_MAX; ++j) {
        const int  row  = (g * RB + j) * 16 + (lane & 15);
        const bool live = j < RB && row < p.M;
        const uint16_t *xa = p.x + int64_t(live ? row : 0) * p.x_stride + k0;
#pragma unroll
        for (int q = 0; q < KWAVE / 32; ++q)
            av[j][q] = live ? *reinterpret_cast<const short8 *>(xa + 32 * q) : short8{0, 0, 0, 0, 0, 0, 0, 0};
    }
#pragma unroll
    for (int j = 0; j < RB_MAX; ++j) {
        if (j >= RB) break;
        floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int q = 0; q < KWAVE / 32; ++q)
            acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(bf16x8, av[j][q]),
                                                          __builtin_bit_cast(bf16x8, bv[q]), acc, 0, 0, 0);
        // C[4 (lane / 16) + i][lane % 16]: row, expert
#pragma unroll
        for (int i = 0; i < 4; ++i) lds.red[wave][j][((lane >> 4) * 4 + i) * 16 + (lane & 15)] = acc[i];
    }
    __syncthreads();
    // the 8 waves' sums in wave order: element e = j 256 + row 16 + expert
    const int nel = RB * 256;
    float *part = p.partial + int64_t(gn * KSPLIT) * nel;
    for (int e = tid; e < nel; e += THREADS) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < WAVES; ++w) v += lds.red[w][e >> 8][e & 255];
        st_dev(part + s * nel + e, v);
    }
    if (!last_arrival(p.counters + (C_RN + gn) * CL, KSPLIT, flag)) return;
    // the last of the K slices: the slices in order -> logits
    for (int e = tid; e < nel; e += THREADS) {
        float q[KSPLIT];
#pragma unroll
        for (int k = 0; k < KSPLIT; ++k) q[k] = ld_dev(part + k * nel + e);
        float v = 0.0f;
#pragma unroll
        for (int k = 0; k < KSPLIT; ++k) v += q[k];
        const int r = g * RB + (e >> 8);
        st_dev(p.logits + int64_t(r * 16 + ((e >> 4) & 15)) * E + n * 16 + (e & 15), v);
    }
    // the row tiles of the group have this expert tile's logits
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    if (tid < RB && g * RB + tid < p.R)
        __hip_atomic_fetch_add(p.counters + (C_R + g * RB + tid) * CL, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Top-k CTA tk: rows 8 tk .. 8 tk + 7, a wave a row, once their row tile's 18 expert tiles are in. The last top-k
// CTA to finish sorts every route and publishes the sort.
template <int K, int NE>
__device__ void topk_cta(const Params &p, int tk, FrontLds &lds, int &flag) {
    const int tid = threadIdx.x, wave = tid / WAVE, lane = tid & (WAVE - 1);
    const int r = tk / 2;
    if (tid == 0)
        while (ld_dev(p.counters + (C_R + r) * CL) < NT) __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    const int valid = p.valid ? min(max(ld_dev(p.valid), 0), p.M) : p.M;
    const int t     = tk * WAVES + wave;
    if (t < p.M) {
        int   id = -1;
        float wt = 0.0f;
        if (t < valid) {
            // the row's 288 logits: 72 x 16 B, lane l holds vectors l and l + 64
            const float *lg = p.logits + int64_t(t) * E;
            // unconditional loads (lanes past vector 71 reread it), so no register merges an in-flight load
            u32x4 v0 = ld_dev16(lg + lane * 4);
            u32x4 v1 = ld_dev16(lg + min(lane + WAVE, E / 4 - 1) * 4);
            vm_wait(v0, v1);
            const uint32_t w[2][4] = {{v0[0], v0[1], v0[2], v0[3]}, {v1[0], v1[1], v1[2], v1[3]}};
            topk_row([&](int v, int i) { return bf16_f32(f32_bf16(__builtin_bit_cast(float, w[v][i]))); }, p.bias,
                     p.scale, id, wt);
            if (K > TOPK && lane == TOPK) {
                id = SHARED_ID;
                wt = 1.0f;
            }
        }
        if (lane < K) {
            st_dev(p.topk_ids + t * K + lane, id);
            st_dev(p.topk_w + t * K + lane, wt);
        }
    }
    if (!last_arrival(p.counters + C_TOP * CL, p.topk_ctas, flag)) return;
    // every top-k CTA is past its wait: the row-tile counts go back to 0
    if (tid < p.R) st_dev(p.counters + (C_R + tid) * CL, 0);
    sort_routes<K, NE>(p, lds.sort);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    if (tid == 0) st_dev(p.counters + C_SORTED * CL, 1);
}

// ---- the activation quant and the zeroed output ----

__device__ void quant_cta(const Params &p, int qb, uint8_t *scale_lds) {
    const int tid = threadIdx.x, half = tid >> 8, t = tid & 255; // 2 rows a pass, 16 values a thread
    for (int row0 = qb * 2; row0 < p.M; row0 += p.quant_ctas * 2) {
        const int row = row0 + half;
        if (row < p.M) {
            const uint16_t *xr = p.x + int64_t(row) * p.x_stride + t * 16;
            const uint4 lo = *reinterpret_cast<const uint4 *>(xr);
            const uint4 hi = *reinterpret_cast<const uint4 *>(xr + 8);
            const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            float f[16];
            float amax = 1e-10f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                f[2 * i]     = __builtin_bit_cast(float, words[i] << 16);
                f[2 * i + 1] = __builtin_bit_cast(float, words[i] & 0xFFFF0000u);
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) amax = max(amax, abs(f[i]));
            // multithread_reduce over the group's 2 threads: quad_perm [1,0,3,2]
            const float other = __builtin_bit_cast(
                float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, amax), 0xb1, 0xf, 0xf, false));
            amax = other > amax ? other : amax;
            // RCEIL: ceil_pow2(amax / 6)
            const uint32_t u   = __builtin_bit_cast(uint32_t, amax * (1.0f / 6.0f));
            uint32_t       exp = (u >> 23) & 0xFFu;
            if (exp < 0xFFu && (u & 0x7FFFFFu)) exp += 1;
            const float scale = __builtin_bit_cast(float, exp << 23);
            if ((t & 1) == 0) scale_lds[half * SCOLS + t / 2] = uint8_t(exp);
            uint32_t q0 = 0, q1 = 0;
            q0 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q0, f[0], f[1], scale, 0);
            q0 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q0, f[2], f[3], scale, 1);
            q0 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q0, f[4], f[5], scale, 2);
            q0 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q0, f[6], f[7], scale, 3);
            q1 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q1, f[8], f[9], scale, 0);
            q1 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q1, f[10], f[11], scale, 1);
            q1 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q1, f[12], f[13], scale, 2);
            q1 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q1, f[14], f[15], scale, 3);
            *reinterpret_cast<uint2 *>(p.a1 + int64_t(row) * (D / 2) + t * 8) = make_uint2(q0, q1);
            p.out[int64_t(row) * (D * 2 / 16) + t]       = make_uint4(0, 0, 0, 0);
            p.out[int64_t(row) * (D * 2 / 16) + 256 + t] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        if (t < SCOLS / 4 && row < p.M)
            __hip_atomic_store(reinterpret_cast<uint32_t *>(p.a1_scale + int64_t(row) * SCOLS) + t,
                               reinterpret_cast<const uint32_t *>(scale_lds + half * SCOLS)[t], __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
        __syncthreads();
    }
    if (p.a1_scale_sorted) { // counted for the scale CTAs, which put the count back to 0
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        if (threadIdx.x == 0)
            __hip_atomic_fetch_add(p.counters + C_QUANT * CL, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// ---- the sorted, shuffled scales (aiter::mx_scale_shuffle_idx with 128 scale columns) ----

// A 32-row chunk at a time: its token ids and the tokens' 128 scale bytes into LDS, then 1024 dwords out, dword d
// holding rows (d % 16, + 16) x scale columns (8 (d / 64) + (d / 16) % 4, + 4).
__device__ void scale_cta(const Params &p, int sb, uint8_t *rows_lds, int &flag) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        while (ld_dev(p.counters + C_SORTED * CL) == 0 || ld_dev(p.counters + C_QUANT * CL) < p.quant_ctas)
            __builtin_amdgcn_s_sleep(2);
    }
    __syncthreads();
    const int chunks = ld_dev(p.num_valid) / 32;
    for (int c = sb; c < chunks; c += p.scale_ctas) {
        if (tid < 256) { // row tid / 8, 16 B part tid % 8
            const int t = ld_dev(p.sorted_ids + c * 32 + tid / 8) & 0xFFFFFF;
            u32x4 v = {0, 0, 0, 0};
            if (t < p.M) {
                v = ld_dev16(p.a1_scale + int64_t(t) * SCOLS + (tid % 8) * 16);
                vm_wait(v);
            }
            reinterpret_cast<u32x4 *>(rows_lds)[tid] = v;
        }
        __syncthreads();
        for (int d = tid; d < 1024; d += THREADS) {
            const int j16 = d & 15, kk = (d >> 6) * 8 + ((d >> 4) & 3);
            const uint8_t *a = rows_lds + j16 * SCOLS, *b = rows_lds + (j16 + 16) * SCOLS;
            const uint32_t w = uint32_t(a[kk]) | (uint32_t(b[kk]) << 8) | (uint32_t(a[kk + 4]) << 16) |
                               (uint32_t(b[kk + 4]) << 24);
            reinterpret_cast<uint32_t *>(p.a1_scale_sorted)[c * 1024 + d] = w;
        }
        __syncthreads();
    }
    if (last_arrival(p.counters + C_SCALE * CL, p.scale_ctas, flag) && tid == 0) {
        st_dev(p.counters + C_SORTED * CL, 0);
        st_dev(p.counters + C_QUANT * CL, 0);
    }
}

template <int K, int NE>
__global__ __launch_bounds__(THREADS) void front_kernel(Params p) {
    __shared__ FrontLds  lds;
    __shared__ uint8_t   scale_lds[32 * SCOLS];
    __shared__ int       flag;
    const int b = blockIdx.x;
    const int q0 = p.router_ctas, t0 = q0 + p.quant_ctas, s0 = t0 + p.topk_ctas;
    if (b < q0)
        router_cta(p, b, lds, flag);
    else if (b < t0)
        quant_cta(p, b - q0, scale_lds);
    else if (b < s0)
        topk_cta<K, NE>(p, b - t0, lds, flag);
    else
        scale_cta(p, b - s0, scale_lds, flag);
}

// The top-k alone on given bf16 logits (the check against aiter's): a wave a row.
__global__ __launch_bounds__(THREADS) void topk_kernel(const uint16_t *logits, int64_t stride, const uint16_t *bias,
                                                       float scale, int *ids, float *w, int M) {
    const int wave = threadIdx.x / WAVE, lane = threadIdx.x & (WAVE - 1);
    const int t = blockIdx.x * WAVES + wave;
    if (t >= M) return;
    const uint16_t *lg = logits + int64_t(t) * stride;
    int   id;
    float wt;
    topk_row([&](int v, int i) { return bf16_f32(lg[(lane + WAVE * v) * 4 + i]); }, bias, scale, id, wt);
    if (lane < TOPK) {
        ids[t * TOPK + lane] = id;
        w[t * TOPK + lane]   = wt;
    }
}

void check(const at::Tensor &t, at::ScalarType dt, int64_t numel, const char *name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous() && t.numel() >= numel, name,
                ": a contiguous tensor of >= ", numel, " elements of the right dtype on the GPU");
}

// Row tiles a router CTA takes: at most 4 row groups, so 72-288 router CTAs.
int row_tiles_per_cta(int R) { return (R + 3) / 4; }

void front(const at::Tensor &x, const at::Tensor &w, const at::Tensor &bias, double scale,
           const c10::optional<at::Tensor> &valid, at::Tensor &partial, at::Tensor &logits, at::Tensor &topk_ids,
           at::Tensor &topk_w, at::Tensor &sorted_ids, at::Tensor &sorted_w, at::Tensor &sorted_eids,
           at::Tensor &num_valid, at::Tensor &a1, at::Tensor &a1_scale,
           const c10::optional<at::Tensor> &a1_scale_sorted, at::Tensor &out, at::Tensor &counters,
           int64_t topk_total) {
    const int64_t M = x.size(0);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == D && x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                    x.scalar_type() == at::kBFloat16 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "x: [M, 4096] bf16, unit column stride, 16 B aligned rows");
    TORCH_CHECK(M >= 1 && M <= MAX_ROWS, "1 to 256 rows");
    TORCH_CHECK(topk_total == TOPK || topk_total == TOPK + 1, "top-k 8, or 9 with the shared slot");
    const int K = int(topk_total), NE = E + (K - TOPK);
    const int R = int((M + 15) / 16), RB = row_tiles_per_cta(R), G = (R + RB - 1) / RB;
    check(w, at::kBFloat16, int64_t(E) * D, "w");
    check(bias, at::kBFloat16, E, "bias");
    check(partial, at::kFloat, int64_t(G) * NT * KSPLIT * RB * 256, "partial");
    check(logits, at::kFloat, int64_t(R) * 16 * E, "logits");
    check(topk_ids, at::kInt, M * K, "topk_ids");
    check(topk_w, at::kFloat, M * K, "topk_w");
    const int64_t cap = sorted_ids.numel();
    const int64_t P   = M * K;
    TORCH_CHECK(cap % BM == 0 && cap >= P + std::min<int64_t>(NE, P) * (BM - 1), "sorted rows");
    check(sorted_w, at::kFloat, cap, "sorted_w");
    check(sorted_eids, at::kInt, cap / BM, "sorted_eids");
    check(num_valid, at::kInt, 2, "num_valid");
    TORCH_CHECK(a1.is_cuda() && a1.is_contiguous() && a1.numel() * a1.element_size() == M * D / 2, "a1");
    TORCH_CHECK(a1_scale.is_cuda() && a1_scale.is_contiguous() && a1_scale.numel() * a1_scale.element_size() == M * SCOLS,
                "a1_scale");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.numel() == M * D &&
                    reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                "out: [M, 4096] bf16");
    check(counters, at::kInt, N_COUNTERS * CL, "counters");
    Params p{};
    p.x         = reinterpret_cast<const uint16_t *>(x.data_ptr());
    p.x_stride  = x.stride(0);
    p.w         = reinterpret_cast<const uint16_t *>(w.data_ptr());
    p.bias      = reinterpret_cast<const uint16_t *>(bias.data_ptr());
    p.scale     = float(scale);
    p.partial   = partial.data_ptr<float>();
    p.logits    = logits.data_ptr<float>();
    p.topk_ids  = topk_ids.data_ptr<int>();
    p.topk_w    = topk_w.data_ptr<float>();
    p.sorted_ids  = sorted_ids.data_ptr<int>();
    p.sorted_w    = sorted_w.data_ptr<float>();
    p.sorted_eids = sorted_eids.data_ptr<int>();
    p.num_valid   = num_valid.data_ptr<int>();
    p.a1          = reinterpret_cast<uint8_t *>(a1.data_ptr());
    p.a1_scale    = reinterpret_cast<uint8_t *>(a1_scale.data_ptr());
    p.out         = reinterpret_cast<uint4 *>(out.data_ptr());
    p.counters    = counters.data_ptr<int>();
    p.M           = int(M);
    p.R           = R;
    p.RB          = RB;
    p.router_ctas = G * NT * KSPLIT;
    p.topk_ctas   = int((M + WAVES - 1) / WAVES);
    p.quant_ctas  = int(std::min<int64_t>((M + 1) / 2, 64));
    p.scale_ctas  = 0;
    p.pad_id      = (K << 24) | int(M);
    if (valid.has_value()) {
        TORCH_CHECK(valid->is_cuda() && valid->numel() == 1 && valid->scalar_type() == at::kInt,
                    "valid: one int32 on the GPU");
        p.valid = valid->data_ptr<int>();
    }
    if (a1_scale_sorted.has_value()) {
        const int64_t rows = (cap + 31) / 32 * 32;
        TORCH_CHECK(a1_scale_sorted->is_cuda() && a1_scale_sorted->is_contiguous() &&
                        a1_scale_sorted->numel() * a1_scale_sorted->element_size() >= rows * SCOLS,
                    "a1_scale_sorted: [cap rounded to 32, 128] e8m0");
        p.a1_scale_sorted = reinterpret_cast<uint8_t *>(a1_scale_sorted->data_ptr());
        p.scale_ctas      = int(std::min<int64_t>(rows / 32, 192));
    }
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(x.device().index()).stream();
    const int grid = p.router_ctas + p.quant_ctas + p.topk_ctas + p.scale_ctas;
    if (K == TOPK)
        front_kernel<TOPK, E><<<grid, THREADS, 0, stream>>>(p);
    else
        front_kernel<TOPK + 1, E + 1><<<grid, THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void topk(const at::Tensor &logits, const at::Tensor &bias, double scale, at::Tensor &ids, at::Tensor &w) {
    const int64_t M = logits.size(0);
    TORCH_CHECK(logits.dim() == 2 && logits.size(1) == E && logits.stride(1) == 1 &&
                    logits.scalar_type() == at::kBFloat16 && logits.stride(0) % 4 == 0,
                "logits: [M, 288] bf16");
    check(bias, at::kBFloat16, E, "bias");
    check(ids, at::kInt, M * TOPK, "ids");
    check(w, at::kFloat, M * TOPK, "w");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(logits.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(logits.device().index()).stream();
    topk_kernel<<<(M + WAVES - 1) / WAVES, THREADS, 0, stream>>>(
        reinterpret_cast<const uint16_t *>(logits.data_ptr()), logits.stride(0),
        reinterpret_cast<const uint16_t *>(bias.data_ptr()), float(scale), ids.data_ptr<int>(), w.data_ptr<float>(),
        int(M));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("front(Tensor x, Tensor w, Tensor bias, float scale, Tensor? valid, Tensor(a!) partial, "
          "Tensor(b!) logits, Tensor(c!) topk_ids, Tensor(d!) topk_w, Tensor(e!) sorted_ids, Tensor(f!) sorted_w, "
          "Tensor(g!) sorted_eids, Tensor(h!) num_valid, Tensor(i!) a1, Tensor(j!) a1_scale, "
          "Tensor(k!)? a1_scale_sorted, Tensor(l!) out, Tensor(m!) counters, int topk_total) -> ()");
    m.impl("front", at::kCUDA, &front);
    m.def("topk(Tensor logits, Tensor bias, float scale, Tensor(a!) ids, Tensor(b!) w) -> ()");
    m.impl("topk", at::kCUDA, &topk);
}
