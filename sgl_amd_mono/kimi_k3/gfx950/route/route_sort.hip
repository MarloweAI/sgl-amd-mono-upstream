// SPDX-License-Identifier: Apache-2.0
// Kimi-K3's MoE routing in one launch at 48-180 rows (TP8, gfx950): the top-16 of 896 sigmoid + bias scores of each
// row (noaux_tc, renormalized), aiter's BM = 16 aux sort of the routes for the a4w4-port stage 1 / stage 2, and the
// zeroed stage-2 output. Replaces aiter's topk_reg_kernel and its moe sort (the one-CTA sort_quant_kernel_impl below
// 75 rows, the two Opus multi-phase launches from 75).
//
// Bit for bit what those launches write:
// - top-k as topk_reg_kernel (csrc/kernels/topk_softmax_kernels_group.cu): sigmoid rcpf(1 + exp2f(-C_LOG2E x)) with
//   C_LOG2E a double literal, the routes ranked by (sigmoid + bias, then legacy_topk_tie_rank), the weights the
//   ranked pre-bias sigmoids times scale / their serial sum;
// - the sort as Opus's MoeSortingMultiPhaseKernel_P23 (csrc/include/moe_sorting_opus.h): experts ascending, an
//   expert's routes in token order, BM = 16 blocks, experts with no route skipped, num_valid = [blocks x 16, M], pad
//   rows pad_id / weight 0 / m_index M. Below 75 rows aiter's one-CTA sort places an expert's routes in LDS-atomic
//   order instead (it varies run to run) and pads with M; pad_id is the caller's.
//
// One launch, no wait on another CTA: route CTAs (8 rows each, a wave a row) write their routes, count themselves
// done, and the last one to finish sorts all of them; the zero CTAs clear the stage-2 output meanwhile.
//
// route_sort_shared also runs, on its own CTAs, what SGLang's situ_and_mul launch does for the shared experts: their
// SiTU of the front's fp32 gate_up into the bf16 shared-down input, and the fp32 routed input's bf16 cast (bit for
// bit: sgl_kernel's ROCm sigmoid __frcp_rn(1 + exp2(-log2e x)), tanhf, __float2bfloat16).
#include <cstdint>
#include <hip/hip_bf16.h>

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

constexpr int E          = 896;
constexpr int TOPK       = 16;
constexpr int LAT        = 3584;
constexpr int BM         = 16;
constexpr int WAVE       = 64;
constexpr int PAIRS      = E / (2 * WAVE); // 7 float2 a lane: experts 2 (lane + 64 j) + i
constexpr int THREADS    = 1024;           // the sort: an expert a thread
constexpr int RW         = 4;              // rows a route CTA: a wave a row, a SIMD a wave
constexpr int MAX_ROWS   = 192;
constexpr int MW         = MAX_ROWS / 32; // token-mask words an expert
constexpr int MAX_PAIRS  = MAX_ROWS * TOPK;
constexpr int PR         = MAX_PAIRS / THREADS; // routes a sorting thread
// sorted rows at most (Opus's capacity, the larger one) and their blocks
constexpr int MAX_SORTED = (MAX_PAIRS + E * BM - TOPK + BM - 1) / BM * BM;
constexpr int MAX_BLOCKS = MAX_SORTED / BM;
constexpr int ZERO_CTAS  = 32;
constexpr int AUX_CTAS   = 32;  // route_sort_shared: SiTU + routed cast
constexpr int SH         = 768; // the shared experts' intermediate a rank

struct Params {
    const float *logits; // [M, E] f32, row stride `stride`
    int64_t      stride;
    const float *bias;   // [E] f32
    float        scale;  // routed_scaling_factor
    int         *topk_ids;  // [M, 16]
    float       *topk_w;    // [M, 16]
    int         *sorted_ids; // [max_sorted]
    float       *sorted_w;
    int         *sorted_eids; // [max_sorted / 16]
    int         *num_valid;   // [2]
    int         *m_indices;   // [max_sorted]
    int         *reverse;     // [M * 16]
    uint4       *zero_out;    // stage 2's output, zero_vecs x 16 B
    int64_t      zero_vecs;
    unsigned    *counter;     // route CTAs done; 0 between launches
    const int   *valid;       // or null: rows from *valid on are padding (ids -1, weights 0, no route)
    int          M, route_ctas, pad_id;
    // route_sort_shared only (aux_ctas > 0)
    const float   *gate_up;    // [M, 2 x SH] f32 (gate | up), row stride gu_stride
    int64_t        gu_stride;
    __hip_bfloat16 *act;       // [M, SH] bf16
    const float   *routed_in;  // [M, LAT] f32, row stride ri_stride
    int64_t        ri_stride;
    __hip_bfloat16 *routed_out; // [M, LAT] bf16
    float          beta, inv_beta, lbeta, inv_lbeta;
    int            aux_ctas;
};

__device__ __forceinline__ uint32_t float_order_key(float v) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, v + 0.0f);
    return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}

// aiter's legacy_topk_tie_rank: the order topk_reg_kernel breaks an exact score tie in, lower first
__device__ __forceinline__ int tie_rank(int e) {
    const int v4    = e >> 2;
    const int lane  = v4 & (WAVE - 1);
    const int local = ((v4 >> 6) << 2) | (e & 3);
    const int low2  = (lane & 3) ^ ((lane & 4) ? 3 : 0);
    const int bit2  = (((lane >> 2) ^ (lane >> 3)) & 1) << 2;
    return ((((~lane) & 0x38) | bit2 | low2) << 5) | local;
}

// the full route order: the score, then the lower tie rank (a larger key ranks first)
__device__ __forceinline__ uint64_t route_key(uint32_t k, int e) {
    return (uint64_t(k) << 32) | uint32_t(~tie_rank(e));
}

__device__ __forceinline__ int expert_of(int lane, int j, int i) { return 2 * (lane + WAVE * j) + i; }


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

struct WaveLds {
    uint32_t key[WAVE];
    int      id[WAVE];
    float    sig[WAVE];
    uint64_t rkey[WAVE];
    int      out_id[TOPK];
    float    out_sig[TOPK];
};

struct SortLds {
    union {
        uint32_t mask[E * MW];        // an expert's tokens (counting, ranking); zero when the sort starts
        uint16_t row_route[MAX_SORTED]; // then: a sorted row's route (token x 16 + slot)
    };
    float    w[MAX_PAIRS];       // the routes' weights
    int      cnt[E];             // an expert's routes
    int      base[E];            // an expert's first block
    int16_t  block_e[MAX_BLOCKS]; // a block's expert
    int      wsum[THREADS / WAVE];
};

// One row's top 16, ranked, into topk_ids / topk_w (lanes 0..15), by one wave.
__device__ void route_row(const Params &p, int row, WaveLds &w) {
    const int    lane = threadIdx.x & (WAVE - 1);
    const float *x    = p.logits + row * p.stride;
    float        sig[PAIRS][2];
    uint32_t     key[PAIRS][2];
#pragma unroll
    for (int j = 0; j < PAIRS; ++j) {
        const float2 g = reinterpret_cast<const float2 *>(x)[lane + WAVE * j];
        const float2 b = reinterpret_cast<const float2 *>(p.bias)[lane + WAVE * j];
        const float  gv[2] = {g.x, g.y}, bv[2] = {b.x, b.y};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float s = __builtin_amdgcn_rcpf(1.0f + exp2f(-C_LOG2E * gv[i]));
            sig[j][i]     = s;
            key[j][i]     = float_order_key(s + bv[i]);
        }
    }
    // pivot: the 16th largest lane maximum; every route of the top 16 scores at least that
    uint32_t lmax = 0;
#pragma unroll
    for (int j = 0; j < PAIRS; ++j) lmax = max(lmax, max(key[j][0], key[j][1]));
    uint32_t pivot = 0;
#pragma unroll
    for (int b = 30; b >= 0; b -= 2) {
        const uint32_t c1 = pivot | (1u << b), c2 = pivot | (2u << b), c3 = pivot | (3u << b);
        const int      n1 = __popcll(__ballot(lmax >= c1)), n2 = __popcll(__ballot(lmax >= c2)),
                       n3 = __popcll(__ballot(lmax >= c3));
        pivot = n3 >= TOPK ? c3 : n2 >= TOPK ? c2 : n1 >= TOPK ? c1 : pivot;
    }
    // candidates (score >= pivot) to LDS, slot-major
    auto collect = [&](auto take) {
        int n = 0;
#pragma unroll
        for (int j = 0; j < PAIRS; ++j) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const bool     c   = take(j, i);
                const uint64_t bal = __ballot(c);
                const int      at  = n + __builtin_amdgcn_mbcnt_hi(uint32_t(bal >> 32),
                                                                    __builtin_amdgcn_mbcnt_lo(uint32_t(bal), 0));
                if (c && at < WAVE) {
                    w.key[at] = key[j][i];
                    w.id[at]  = expert_of(lane, j, i);
                    w.sig[at] = sig[j][i];
                }
                n += __popcll(bal);
            }
        }
        return n;
    };
    int n = collect([&](int j, int i) { return key[j][i] >= pivot; });
    if (n > WAVE) {
        // ties at the pivot score: the pivot on the full route key (distinct for every expert) instead
        uint64_t lmax64 = 0;
#pragma unroll
        for (int j = 0; j < PAIRS; ++j)
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint64_t K = route_key(key[j][i], expert_of(lane, j, i));
                lmax64           = K > lmax64 ? K : lmax64;
            }
        uint64_t pv = 0;
        for (int b = 62; b >= 0; b -= 2) {
            const uint64_t c1 = pv | (1ull << b), c2 = pv | (2ull << b), c3 = pv | (3ull << b);
            const int      n1 = __popcll(__ballot(lmax64 >= c1)), n2 = __popcll(__ballot(lmax64 >= c2)),
                           n3 = __popcll(__ballot(lmax64 >= c3));
            pv = n3 >= TOPK ? c3 : n2 >= TOPK ? c2 : n1 >= TOPK ? c1 : pv;
        }
        __builtin_amdgcn_wave_barrier();
        n = collect([&](int j, int i) { return route_key(key[j][i], expert_of(lane, j, i)) >= pv; });
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    int   rid = 0;
    float rsig = 0.0f;
    if (n <= WAVE) {
        const bool     live = lane < n;
        const uint32_t k    = live ? w.key[lane] : 0u;
        const int      id   = live ? w.id[lane] : 0;
        const float    s    = live ? w.sig[lane] : 0.0f;
        const uint64_t K    = live ? route_key(k, id) : 0ull;
        w.rkey[lane]        = K;
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        int rank = 0;
#pragma unroll 8
        for (int j = 0; j < n; ++j) rank += w.rkey[j] > K ? 1 : 0;
        __builtin_amdgcn_wave_barrier();
        if (live && rank < TOPK) {
            w.out_id[rank]  = id;
            w.out_sig[rank] = s;
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        rid  = lane < TOPK ? w.out_id[lane] : 0;
        rsig = lane < TOPK ? w.out_sig[lane] : 0.0f;
    } else {
        // more than a wave of candidates even so (one lane holds many of the top routes): 16 rounds of the wave's
        // best remaining route
        uint64_t K[PAIRS][2];
#pragma unroll
        for (int j = 0; j < PAIRS; ++j)
#pragma unroll
            for (int i = 0; i < 2; ++i) K[j][i] = route_key(key[j][i], expert_of(lane, j, i));
        for (int t = 0; t < TOPK; ++t) {
            uint64_t best = 0;
#pragma unroll
            for (int j = 0; j < PAIRS; ++j)
#pragma unroll
                for (int i = 0; i < 2; ++i) best = K[j][i] > best ? K[j][i] : best;
            uint64_t m = best;
            for (int d = 1; d < WAVE; d <<= 1) {
                const uint64_t o = (uint64_t(uint32_t(__shfl_xor(int(m >> 32), d))) << 32) |
                                   uint32_t(__shfl_xor(int(uint32_t(m)), d));
                m = o > m ? o : m;
            }
            int   wid = 0;
            float ws  = 0.0f;
#pragma unroll
            for (int j = 0; j < PAIRS; ++j)
#pragma unroll
                for (int i = 0; i < 2; ++i)
                    if (K[j][i] == m) {
                        wid     = expert_of(lane, j, i);
                        ws      = sig[j][i];
                        K[j][i] = 0;
                    }
            const int owner = __ffsll((long long)__ballot(best == m)) - 1;
            wid = __shfl(wid, owner);
            ws  = __shfl(ws, owner);
            if (lane == t) {
                rid  = wid;
                rsig = ws;
            }
        }
    }
    // aiter's serial renormalization, in rank order
    if (lane < TOPK) w.out_sig[lane] = rsig;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    float sv[TOPK];
#pragma unroll
    for (int i = 0; i < TOPK; ++i) sv[i] = w.out_sig[i];
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < TOPK; ++i) total += sv[i];
    float wt = rsig;
    if (total > 0.0f)
        wt *= p.scale / total;
    else
        wt = 0.0f;
    if (lane < TOPK) {
        st_dev(p.topk_ids + row * TOPK + lane, rid);
        st_dev(p.topk_w + row * TOPK + lane, wt);
    }
}

// Inclusive scan across a wave (aiter's dpp_inclusive_scan_wave: row shifts, then row broadcasts).
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

// Block-wide exclusive scan of one value a thread; the block's total through ``total``.
__device__ int block_excl_scan(int v, SortLds &s, int &total) {
    const int lane = threadIdx.x & (WAVE - 1), wave = threadIdx.x / WAVE;
    const int x    = wave_incl_scan(v);
    if (lane == WAVE - 1) s.wsum[wave] = x;
    __syncthreads();
    int before = 0, all = 0;
#pragma unroll
    for (int i = 0; i < THREADS / WAVE; ++i) {
        const int t = s.wsum[i];
        before += i < wave ? t : 0;
        all += t;
    }
    total = all;
    return before + x - v;
}

// The last route CTA: the BM = 16 aux sort of every route. Routes counted into token masks (LDS), experts
// scanned in order, each route's row from its rank among its expert's tokens; then every sorted array is written
// in one pass over the sorted rows, so the global stores are contiguous.
__device__ void sort_routes(const Params &p, SortLds &s) {
    const int tid = threadIdx.x;
    const int P   = p.M * TOPK;
    const int mw  = (p.M + 31) / 32;
    for (int i = tid; i < E * MW; i += THREADS) s.mask[i] = 0;
    __syncthreads();
    int es[PR];
#pragma unroll
    for (int k = 0; k < PR; ++k) {
        const int q = tid + k * THREADS;
        es[k]       = q < P ? ld_dev(p.topk_ids + q) : -1;
        if (q < P) s.w[q] = ld_dev(p.topk_w + q);
        if (es[k] >= 0) {
            const int t = q / TOPK;
            atomicOr(&s.mask[es[k] * MW + t / 32], 1u << (t % 32));
        }
    }
    __syncthreads();
    // experts ascending, an expert a thread
    static_assert(THREADS >= E, "an expert a sorting thread");
    const int e   = tid;
    int       cnt = 0;
    if (e < E)
        for (int i = 0; i < mw; ++i) cnt += __popc(s.mask[e * MW + i]);
    const int blocks = (cnt + BM - 1) / BM;
    int       run;
    const int base = block_excl_scan(blocks, s, run);
    if (e < E) {
        s.cnt[e]  = cnt;
        s.base[e] = base;
        for (int b = 0; b < blocks; ++b) s.block_e[base + b] = int16_t(e);
    }
    __syncthreads();
    const int nv = run * BM;
    // each route's sorted row: its expert's first row + the expert's tokens below it
    int dest[PR];
#pragma unroll
    for (int k = 0; k < PR; ++k) {
        const int q = tid + k * THREADS;
        dest[k]     = -1;
        if (es[k] >= 0) {
            const int e = es[k], t = q / TOPK;
            int       rank = 0;
            for (int i = 0; i < t / 32; ++i) rank += __popc(s.mask[e * MW + i]);
            rank += __popc(s.mask[e * MW + t / 32] & ((1u << (t % 32)) - 1u));
            dest[k] = s.base[e] * BM + rank;
        }
        if (q < P) p.reverse[q] = dest[k];
    }
    __syncthreads(); // the masks are read; their space takes the row -> route map
#pragma unroll
    for (int k = 0; k < PR; ++k)
        if (dest[k] >= 0) s.row_route[dest[k]] = uint16_t(tid + k * THREADS);
    __syncthreads();
    for (int b = tid; b < run; b += THREADS) p.sorted_eids[b] = s.block_e[b];
    for (int j = tid; j < nv; j += THREADS) {
        const int e = s.block_e[j / BM];
        if (j - s.base[e] * BM < s.cnt[e]) {
            const int q = s.row_route[j], t = q / TOPK;
            p.sorted_ids[j] = ((q % TOPK) << 24) | t;
            p.sorted_w[j]   = s.w[q];
            p.m_indices[j]  = t;
        } else {
            p.sorted_ids[j] = p.pad_id;
            p.sorted_w[j]   = 0.0f;
            p.m_indices[j]  = p.M;
        }
    }
    if (tid == 0) {
        p.num_valid[0] = nv;
        p.num_valid[1] = p.M;
    }
}

__device__ __forceinline__ float situ_sigmoid(float x) {
    return __frcp_rn(1.0f + __builtin_amdgcn_exp2f(x * -1.44269504088896340736f));
}

// SiTU (8 outputs an item) then the routed cast (8 an item), over the aux CTAs
__device__ void aux_items(const Params &p, int cta) {
    const int64_t span  = int64_t(p.aux_ctas) * THREADS;
    const int64_t n_act = int64_t(p.M) * (SH / 8);
    const int64_t n_all = n_act + int64_t(p.M) * (LAT / 8);
    for (int64_t i = int64_t(cta) * THREADS + threadIdx.x; i < n_all; i += span) {
        if (i < n_act) {
            const int64_t r = i / (SH / 8), c = (i % (SH / 8)) * 8;
            const float *g = p.gate_up + r * p.gu_stride + c;
            const float4 g0 = *reinterpret_cast<const float4 *>(g), g1 = *reinterpret_cast<const float4 *>(g + 4);
            const float4 u0 = *reinterpret_cast<const float4 *>(g + SH), u1 = *reinterpret_cast<const float4 *>(g + SH + 4);
            const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
            const float uv[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
            __hip_bfloat16 o[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float gate_out = p.beta * tanhf(gv[j] * p.inv_beta) * situ_sigmoid(gv[j]);
                const float up_out   = p.lbeta * tanhf(uv[j] * p.inv_lbeta);
                o[j] = __float2bfloat16(gate_out * up_out);
            }
            *reinterpret_cast<uint4 *>(p.act + r * SH + c) = *reinterpret_cast<const uint4 *>(o);
        } else {
            const int64_t k = i - n_act, r = k / (LAT / 8), c = (k % (LAT / 8)) * 8;
            const float *x = p.routed_in + r * p.ri_stride + c;
            const float4 x0 = *reinterpret_cast<const float4 *>(x), x1 = *reinterpret_cast<const float4 *>(x + 4);
            const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
            __hip_bfloat16 o[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = __float2bfloat16(xv[j]);
            *reinterpret_cast<uint4 *>(p.routed_out + r * LAT + c) = *reinterpret_cast<const uint4 *>(o);
        }
    }
}

__global__ __launch_bounds__(THREADS) void route_sort_kernel(Params p) {
    // the route rows' scratch, then (the last route CTA only) the sort's
    __shared__ union {
        WaveLds wl[RW];
        SortLds sl;
    } lds;
    __shared__ int last;
    if (int(blockIdx.x) >= p.route_ctas + ZERO_CTAS) {
        aux_items(p, int(blockIdx.x) - p.route_ctas - ZERO_CTAS);
        return;
    }
    if (int(blockIdx.x) >= p.route_ctas) {
        const int64_t span = int64_t(ZERO_CTAS) * THREADS;
        for (int64_t i = int64_t(blockIdx.x - p.route_ctas) * THREADS + threadIdx.x; i < p.zero_vecs; i += span)
            p.zero_out[i] = make_uint4(0, 0, 0, 0);
        return;
    }
    const int wave  = threadIdx.x / WAVE;
    const int row   = blockIdx.x * RW + wave;
    const int lane  = threadIdx.x & (WAVE - 1);
    const int valid = p.valid ? min(max(*p.valid, 0), p.M) : p.M;
    if (wave >= RW) {
    } else if (row < valid)
        route_row(p, row, lds.wl[wave]);
    else if (row < p.M && lane < TOPK) { // as SGLang masks a padded row's routes
        st_dev(p.topk_ids + row * TOPK + lane, -1);
        st_dev(p.topk_w + row * TOPK + lane, 0.0f);
    }
    // this CTA's routes stored (device scope) before it counts itself done
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    if (threadIdx.x == 0)
        last = __hip_atomic_fetch_add(p.counter, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) ==
               unsigned(p.route_ctas - 1);
    __syncthreads();
    if (!last) return;
    sort_routes(p, lds.sl);
    if (threadIdx.x == 0) __hip_atomic_store(p.counter, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

void check(const at::Tensor &t, at::ScalarType dt, int64_t numel, const char *name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous() && t.numel() == numel, name,
                ": a contiguous ", numel, "-element tensor of the right dtype on the GPU");
}

// logits [M, 896] f32 (row stride even, 8 B aligned), bias [896] f32; out [M, 3584] bf16 (zeroed)
Params make_params(const at::Tensor &logits, const at::Tensor &bias, double scale, at::Tensor &topk_ids,
                   at::Tensor &topk_w, at::Tensor &sorted_ids, at::Tensor &sorted_w, at::Tensor &sorted_eids,
                   at::Tensor &num_valid, at::Tensor &m_indices, at::Tensor &reverse, at::Tensor &out,
                   at::Tensor &counter, int64_t pad_id, const c10::optional<at::Tensor> &valid) {
    const int64_t M = logits.size(0);
    TORCH_CHECK(logits.dim() == 2 && logits.size(1) == E && logits.stride(1) == 1 && logits.stride(0) % 2 == 0 &&
                    logits.scalar_type() == at::kFloat && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 8 == 0,
                "logits: [M, 896] f32, unit column stride, even row stride, 8 B aligned");
    TORCH_CHECK(M >= 1 && M <= MAX_ROWS, "1 to 192 rows");
    check(bias, at::kFloat, E, "bias");
    check(topk_ids, at::kInt, M * TOPK, "topk_ids");
    check(topk_w, at::kFloat, M * TOPK, "topk_w");
    const int64_t ms = sorted_ids.numel();
    TORCH_CHECK(ms % BM == 0 && ms >= M * TOPK + std::min<int64_t>(E, M * TOPK) * (BM - 1), "sorted rows");
    check(sorted_ids, at::kInt, ms, "sorted_ids");
    check(sorted_w, at::kFloat, ms, "sorted_w");
    check(m_indices, at::kInt, ms, "m_indices");
    check(sorted_eids, at::kInt, ms / BM, "sorted_eids");
    check(num_valid, at::kInt, 2, "num_valid");
    check(reverse, at::kInt, M * TOPK, "reverse");
    check(out, at::kBFloat16, M * LAT, "out");
    check(counter, at::kInt, 1, "counter");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "out 16 B aligned");
    Params p{};
    p.logits      = logits.data_ptr<float>();
    p.stride      = logits.stride(0);
    p.bias        = bias.data_ptr<float>();
    p.scale       = static_cast<float>(scale);
    p.topk_ids    = topk_ids.data_ptr<int>();
    p.topk_w      = topk_w.data_ptr<float>();
    p.sorted_ids  = sorted_ids.data_ptr<int>();
    p.sorted_w    = sorted_w.data_ptr<float>();
    p.sorted_eids = sorted_eids.data_ptr<int>();
    p.num_valid   = num_valid.data_ptr<int>();
    p.m_indices   = m_indices.data_ptr<int>();
    p.reverse     = reverse.data_ptr<int>();
    p.zero_out    = reinterpret_cast<uint4 *>(out.data_ptr());
    p.zero_vecs   = M * LAT * 2 / 16;
    p.counter     = reinterpret_cast<unsigned *>(counter.data_ptr<int>());
    p.M           = static_cast<int>(M);
    p.route_ctas  = static_cast<int>((M + RW - 1) / RW);
    p.pad_id      = static_cast<int>(pad_id);
    if (valid.has_value()) {
        TORCH_CHECK(valid->is_cuda() && valid->numel() == 1 && valid->scalar_type() == at::kInt,
                    "valid: one int32 on the GPU");
        p.valid = valid->data_ptr<int>();
    }
    return p;
}

void launch(const Params &p, const at::Tensor &logits) {
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(logits.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(logits.device().index()).stream();
    route_sort_kernel<<<p.route_ctas + ZERO_CTAS + p.aux_ctas, THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void route_sort(const at::Tensor &logits, const at::Tensor &bias, double scale, at::Tensor &topk_ids,
                at::Tensor &topk_w, at::Tensor &sorted_ids, at::Tensor &sorted_w, at::Tensor &sorted_eids,
                at::Tensor &num_valid, at::Tensor &m_indices, at::Tensor &reverse, at::Tensor &out,
                at::Tensor &counter, int64_t pad_id, const c10::optional<at::Tensor> &valid) {
    launch(make_params(logits, bias, scale, topk_ids, topk_w, sorted_ids, sorted_w, sorted_eids, num_valid,
                       m_indices, reverse, out, counter, pad_id, valid),
           logits);
}

// route_sort, plus: gate_up [M, 1536] f32 (unit column stride, row stride a multiple of 4, 16 B aligned) -> act
// [M, 768] bf16 (the shared experts' SiTU), routed_in [M, 3584] f32 (same alignment) -> routed_out [M, 3584] bf16
void route_sort_shared(const at::Tensor &logits, const at::Tensor &bias, double scale, at::Tensor &topk_ids,
                       at::Tensor &topk_w, at::Tensor &sorted_ids, at::Tensor &sorted_w, at::Tensor &sorted_eids,
                       at::Tensor &num_valid, at::Tensor &m_indices, at::Tensor &reverse, at::Tensor &out,
                       at::Tensor &counter, int64_t pad_id, const c10::optional<at::Tensor> &valid,
                       const at::Tensor &gate_up, at::Tensor &act, const at::Tensor &routed_in,
                       at::Tensor &routed_out, double beta, double linear_beta) {
    Params p = make_params(logits, bias, scale, topk_ids, topk_w, sorted_ids, sorted_w, sorted_eids, num_valid,
                           m_indices, reverse, out, counter, pad_id, valid);
    const int64_t M = logits.size(0);
    auto strided = [&](const at::Tensor &t, int64_t cols, const char *name) {
        TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.size(0) == M && t.size(1) == cols && t.stride(1) == 1 &&
                        t.stride(0) % 4 == 0 && t.scalar_type() == at::kFloat &&
                        reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                    name, ": [M, ", cols, "] f32, unit column stride, row stride a multiple of 4, 16 B aligned");
    };
    strided(gate_up, 2 * SH, "gate_up");
    strided(routed_in, LAT, "routed_in");
    check(act, at::kBFloat16, M * SH, "act");
    check(routed_out, at::kBFloat16, M * LAT, "routed_out");
    p.gate_up    = gate_up.data_ptr<float>();
    p.gu_stride  = gate_up.stride(0);
    p.act        = reinterpret_cast<__hip_bfloat16 *>(act.data_ptr());
    p.routed_in  = routed_in.data_ptr<float>();
    p.ri_stride  = routed_in.stride(0);
    p.routed_out = reinterpret_cast<__hip_bfloat16 *>(routed_out.data_ptr());
    const float b = static_cast<float>(beta), lb = static_cast<float>(linear_beta);
    p.beta = b, p.inv_beta = 1.0f / b, p.lbeta = lb, p.inv_lbeta = 1.0f / lb;  // as situ_and_mul's host
    p.aux_ctas = AUX_CTAS;
    launch(p, logits);
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("route_sort(Tensor logits, Tensor bias, float scale, Tensor(a!) topk_ids, Tensor(b!) topk_w, "
          "Tensor(c!) sorted_ids, Tensor(d!) sorted_w, Tensor(e!) sorted_eids, Tensor(f!) num_valid, "
          "Tensor(g!) m_indices, Tensor(h!) reverse, Tensor(i!) out, Tensor(j!) counter, int pad_id, "
          "Tensor? valid=None) -> ()");
    m.impl("route_sort", at::kCUDA, &route_sort);
    m.def("route_sort_shared(Tensor logits, Tensor bias, float scale, Tensor(a!) topk_ids, Tensor(b!) topk_w, "
          "Tensor(c!) sorted_ids, Tensor(d!) sorted_w, Tensor(e!) sorted_eids, Tensor(f!) num_valid, "
          "Tensor(g!) m_indices, Tensor(h!) reverse, Tensor(i!) out, Tensor(j!) counter, int pad_id, "
          "Tensor? valid, Tensor gate_up, Tensor(k!) act, Tensor routed_in, Tensor(l!) routed_out, float beta, "
          "float linear_beta) -> ()");
    m.impl("route_sort_shared", at::kCUDA, &route_sort_shared);
}
