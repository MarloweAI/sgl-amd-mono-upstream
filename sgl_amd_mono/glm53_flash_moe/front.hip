// SPDX-License-Identifier: Apache-2.0
// GLM-5.3-Flash's decode MoE front at TP2 (gfx950) in one launch, 1-256 rows: the router GEMM (bf16 x bf16, f32
// sums in a fixed order), the biased sigmoid top-8 of 288, the BM = 32 sort of the routes (with the shared expert
// as slot 288 when SHARED), the rows' MXFP4 activation quant and the zeroed MoE output.
//
// CTA roles, in id order (a CTA waits only on CTAs with lower ids):
// - router (g, n, s): row tiles g RB .. (RB = ceil(R / 3) up to 12 row tiles, else 4), experts 16 n.., K slice s
//   of 4: its 8 waves' MFMA sums added in wave order -> f32 partial logits; counts itself on each of its row
//   tiles. Never waits.
// - top-k (8 rows each): quantizes its rows (x bf16 -> MXFP4 a1 [M, 2048 B] and per-token e8m0 scales [M, 128]) and
//   zeroes their MoE output while the router runs; then, once its row tile's 72 router tiles are counted, sums the
//   K slices in order and picks each row's top 8, a wave a row. The last top-k CTA to finish sorts every route.
// - scale (with a1_scale_sorted only): once the sort and the quant are done, the per-token scales in the sorted,
//   shuffled layout aiter's FlyDSL stage 1 reads (aiter::mx_scale_shuffle_idx).
//
// Bit for bit what aiter writes, given the same inputs:
// - top-k: aiter's grouped_topk_kernel (topk_softmax_kernels_group.cu; biased, one group, f32x4 lanes) on the
//   logits rounded to bf16, SGLang's logits dtype: sigmoid rcpf(1 + exp2f(-C_LOG2E x)) with C_LOG2E a double
//   literal, + bias, the eight its DPP arg-max rounds pick (in their order, by an exact key), the pre-bias sigmoids
//   renormalized x scale;
// - quant: aiter's fused_mx_quant_moe_sort_kernel for fp4 (RCEIL e8m0 of the group amax, cvt_scalef32_pk_fp4).
// The sort is Opus's layout (experts ascending, an expert's routes in token order, pad rows (k << 24) | M with
// weight 0), in a capacity of M k + min(E, M k) (BM - 1) rows.
//
// Hand-offs between CTAs: data stored and loaded at agent scope (sc1), then s_waitcnt vmcnt(0) and a barrier before
// the counter's agent-scope add. Every counter and flag is put back to 0 by its last user, for graph replay.
//
// Sequence-parallel build (front_sp, TP2): x is this rank's own M rows of a 2 M-row batch (rank 0 rows [0, M), rank
// 1 rows [M, 2 M)). The router, top-k and quant CTAs run on those rows as above and write the batch's arrays at
// their own rows, write-through (sc0 sc1 / sc1); each top-k CTA then flags its quant rows in the peer's memory.
// Mover CTAs (one per peer top-k CTA) wait for that flag and pull the peer's quant rows into the peer's half of this
// rank's arrays (and zero the MoE output there). The sorting CTA flags its routes to the peer, waits for the peer's,
// pulls them, and sorts all 2 M rows. Flags are the TP group's tp2_xchg epochs (gen + 1, never reset); the launch
// advances gen once. Every value is the one the full-batch front writes: each op above is per row.
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
    const int      *given_ids; // [M, 8], with GIVEN: the routes the top-k CTAs write once their own top-k is done
    const float    *given_w;
    int             given_stride; // their row stride (8 or more: a wider top-k's first 8 columns)
    float          *partial; // [KSPLIT, R * 16, E] f32 router partials
    float          *logits;  // [R * 16, E] f32
    int            *topk_ids; // [M, K]
    float          *topk_w;
    int            *sorted_ids; // [cap]
    float          *sorted_w;
    int            *sorted_eids; // [cap / BM]
    int            *num_valid;   // [2]
    int            *tile_target; // [16]: 2 x the sorted blocks holding a route of a token of each 16-token tile
    uint8_t        *a1;          // [M, D / 2] fp4x2
    uint8_t        *a1_scale;    // [M, SCOLS] e8m0, a token's own
    uint8_t        *a1_scale_sorted; // [cap rounded to 32, SCOLS] shuffled, or null
    uint8_t        *a1_even;         // [M, D / 2] fp4x2 with the shared expert's scales (even rounding), or null
    uint8_t        *a1_even_scale;   // [M, SCOLS]
    uint4          *out;         // the MoE output, zeroed: M x D bf16
    int            *counters;    // [N_COUNTERS * CL], 0 between launches
    int             M, R, RB, router_ctas, topk_ctas, scale_ctas, pad_id;
    // the sequence-parallel build only
    int             row_off, Mf, mover_ctas; // this rank's first row of the batch, the batch's rows, movers
    int            *topk_ids_full;            // the batch's arrays, from row 0
    float          *topk_w_full;
    uint8_t        *a1_full, *a1_scale_full, *a1_even_full, *a1_even_scale_full;
    uint4          *out_full;
    const int      *peer_topk_ids;            // the peer's batch arrays (IPC-mapped), from row 0
    const float    *peer_topk_w;
    const uint8_t  *peer_a1, *peer_a1_scale, *peer_a1_even, *peer_a1_even_scale;
    uint32_t       *peer_flags;               // tp2_xchg: the peer's flags, this rank's, the launch counter
    const uint32_t *flags;
    uint32_t       *gen, *ticket;
};

constexpr int SP_QF = 3072; // flags: a top-k CTA's quant rows are out (+ the CTA index)
constexpr int SP_RF = 3200; // flags: the routes are out

__device__ __forceinline__ uint32_t flag_ld(const uint32_t *p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void flag_st(uint32_t *p, uint32_t v) {
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
// write-through stores (sc0 sc1): the peer reads them over the link
__device__ __forceinline__ void st_wt8(void *p, uint2 v) {
    __hip_atomic_store(reinterpret_cast<uint64_t *>(p), (uint64_t(v.y) << 32) | v.x, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_wt1(uint8_t *p, uint8_t v) {
    asm volatile("global_store_byte %0, %1, off sc0 sc1\n\ts_nop 1" ::"v"(p), "v"(uint32_t(v)) : "memory");
}
// 16 B of the peer's memory at system scope, as two 8 B loads the compiler waits for
__device__ __forceinline__ uint4 ld_sys16(const void *ptr) {
    const uint64_t *q  = reinterpret_cast<const uint64_t *>(ptr);
    const uint64_t  lo = __hip_atomic_load(q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    const uint64_t  hi = __hip_atomic_load(q + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return make_uint4(uint32_t(lo), uint32_t(lo >> 32), uint32_t(hi), uint32_t(hi >> 32));
}

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

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
constexpr int CPOL_SC1 = 16; // gfx950 cache policy: sc1, the device-coherent level
// A raw buffer over p (4 GB range, gfx9 data format)
__device__ __forceinline__ __amdgpu_buffer_rsrc_t rsrc(const void *p) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(p), 0, 0x7fffffff, 0x00020000);
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



// A row's scores in registers: lane l holds expert vectors l and l + 64 (4 experts each), the vectors aiter's
// 64-thread block scans in the same order.
struct RowScores {
    float sc[2][4]; // sigmoid + bias; -inf past expert 287
    float sg[2][4]; // pre-bias sigmoid = routing weight
};

// ``logit(v, i)`` gives the logit of expert 4 (lane + 64 v) + i as aiter reads it (bf16 widened to f32)
template <typename Logit>
__device__ __forceinline__ RowScores row_scores(Logit logit, const float (&bias)[2][4]) {
    const int lane = threadIdx.x & (WAVE - 1);
    RowScores r;
#pragma unroll
    for (int v = 0; v < 2; ++v) {
        const int  e    = lane + WAVE * v;
        const bool live = e < E / 4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float g     = live ? logit(v, i) : 0.0f;
            g           = __builtin_amdgcn_rcpf(1.0f + exp2f(-C_LOG2E * g));
            r.sg[v][i]  = g;
            r.sc[v][i]  = live ? g + bias[v][i] : -INFINITY;
        }
    }
    return r;
}

// The bias of this lane's experts (vectors lane and lane + 64), widened to f32
__device__ __forceinline__ void lane_bias(const uint16_t *bias, float (&b)[2][4]) {
    const int lane = threadIdx.x & (WAVE - 1);
#pragma unroll
    for (int v = 0; v < 2; ++v) {
        const uint2 w = *reinterpret_cast<const uint2 *>(bias + min(lane + WAVE * v, E / 4 - 1) * 4);
        b[v][0]       = __builtin_bit_cast(float, w.x << 16);
        b[v][1]       = __builtin_bit_cast(float, w.x & 0xFFFF0000u);
        b[v][2]       = __builtin_bit_cast(float, w.y << 16);
        b[v][3]       = __builtin_bit_cast(float, w.y & 0xFFFF0000u);
    }
}

// The order aiter's eight rounds pick in (each round: a lane's first best in scan order, then aiter's warpReduceMax,
// a DPP tree read at lane 63 in which a tie keeps the receiving lane's value): the score (larger first), then the
// lane's rank in that tree (lane 63 first), then the lane's scan order. A larger key ranks first.
__device__ __forceinline__ uint32_t float_order_key(float v) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, v + 0.0f);
    return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}
__device__ __forceinline__ int tie_rank(int e) {
    const int v4    = e >> 2;
    const int lane  = v4 & (WAVE - 1);
    const int local = ((v4 >> 6) << 2) | (e & 3);
    const int low2  = (lane & 3) ^ ((lane & 4) ? 3 : 0);
    const int bit2  = (((lane >> 2) ^ (lane >> 3)) & 1) << 2;
    return ((((~lane) & 0x38) | bit2 | low2) << 5) | local;
}
__device__ __forceinline__ uint64_t route_key(uint32_t k, int e) { return (uint64_t(k) << 32) | uint32_t(~tie_rank(e)); }

struct WaveLds {
    uint64_t rkey[WAVE];
    int      id[WAVE];
    float    sig[WAVE];
    int      out_id[TOPK];
    float    out_sig[TOPK];
};

__device__ __forceinline__ void wave_sync_lds() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// One row's top 8 by one wave, as aiter's rounds pick them (by the full key above): each lane's best full key is
// ranked among the 64 through LDS; the 8th of them is the pivot, every expert of the top 8 has a key at least that,
// and those candidates sit in the 8 best lanes (8 experts a lane at most), so they fit a wave. The candidates are
// ranked by their full keys. Lanes 0..7 return their rank's expert and weight (the ranked sigmoids renormalized x
// scale, summed in rank order).
__device__ void topk_row(const RowScores &r, float scale, WaveLds &w, int &out_id, float &out_w) {
    const int lane = threadIdx.x & (WAVE - 1);
    auto expert = [&](int v, int i) { return (lane + WAVE * v) * 4 + i; };
    uint64_t key[2][4];
    uint64_t lmax = 0;
#pragma unroll
    for (int v = 0; v < 2; ++v)
#pragma unroll
        for (int i = 0; i < 4; i++) {
            key[v][i] = lane + WAVE * v < E / 4 ? route_key(float_order_key(r.sc[v][i]), expert(v, i)) : 0ull;
            lmax      = key[v][i] > lmax ? key[v][i] : lmax;
        }
    w.rkey[lane] = lmax;
    wave_sync_lds();
    int lrank = 0;
#pragma unroll 8
    for (int j = 0; j < WAVE; ++j) lrank += w.rkey[j] > lmax ? 1 : 0;
    const uint64_t at  = __ballot(lrank == TOPK - 1);
    const int      src = __builtin_ctzll(at);
    const uint64_t pivot = (uint64_t(uint32_t(__builtin_amdgcn_readlane(int(lmax >> 32), src))) << 32) |
                           uint32_t(__builtin_amdgcn_readlane(int(uint32_t(lmax)), src));
    wave_sync_lds();
    int n = 0;
#pragma unroll
    for (int v = 0; v < 2; ++v)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const bool     c   = key[v][i] >= pivot;
            const uint64_t bal = __ballot(c);
            const int      pos = n + __builtin_amdgcn_mbcnt_hi(uint32_t(bal >> 32),
                                                               __builtin_amdgcn_mbcnt_lo(uint32_t(bal), 0));
            if (c) {
                w.rkey[pos] = key[v][i];
                w.id[pos]   = expert(v, i);
                w.sig[pos]  = r.sg[v][i];
            }
            n += __popcll(bal);
        }
    wave_sync_lds();
    const bool     live = lane < n;
    const uint64_t K    = live ? w.rkey[lane] : 0ull;
    const int      id   = live ? w.id[lane] : 0;
    const float    sgv  = live ? w.sig[lane] : 0.0f;
    int rank = 0;
#pragma unroll 1
    for (int j = 0; j < n; ++j) rank += w.rkey[j] > K ? 1 : 0;
    wave_sync_lds();
    if (live && rank < TOPK) {
        w.out_id[rank]  = id;
        w.out_sig[rank] = sgv;
    }
    wave_sync_lds();
    const int   rid  = lane < TOPK ? w.out_id[lane] : 0;
    const float rsig = lane < TOPK ? w.out_sig[lane] : 0.0f;
    // aiter's renormalization: the sigmoids summed in rank order, then sigmoid x (scale / sum)
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
        sum += __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, rsig), k));
    sum    = sum > 0.0f ? scale / sum : 0.0f;
    out_id = rid;
    out_w  = rsig * sum;
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
    uint32_t tile_blocks[MAX_ROWS / 16][(MAX_BLOCKS + 31) / 32]; // the sorted blocks holding a token of each tile
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
    for (int i = tid; i < (MAX_ROWS / 16) * ((MAX_BLOCKS + 31) / 32); i += THREADS) (&s.tile_blocks[0][0])[i] = 0;
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
            const int b = dest[k] / BM;
            atomicOr(&s.tile_blocks[t / 16][b / 32], 1u << (b % 32));
        }
    }
    __syncthreads(); // the masks are read; their space takes the row -> route map
#pragma unroll
    for (int k = 0; k < PR; ++k)
        if (dest[k] >= 0) s.row_route[dest[k]] = uint16_t(tid + k * THREADS);
    __syncthreads();
    for (int b = tid; b < run; b += THREADS) p.sorted_eids[b] = s.block_e[b];
    // 4 sorted rows a thread, one 16 B write-through store (sc1) of their ids and one of their weights
    const __amdgpu_buffer_rsrc_t ids_r = rsrc(p.sorted_ids), w_r = rsrc(p.sorted_w);
    for (int j0 = tid * 4; j0 < nv; j0 += THREADS * 4) {
        const int ex = s.block_e[j0 / BM], first = j0 - s.base[ex] * BM;
        int   id[4];
        float wt[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (first + i < s.cnt[ex]) {
                const int q = s.row_route[j0 + i];
                id[i]       = ((q % K) << 24) | (q / K);
                wt[i]       = s.w[q];
            } else {
                id[i] = p.pad_id;
                wt[i] = 0.0f;
            }
        }
        __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(u32x4, make_int4(id[0], id[1], id[2], id[3])), ids_r,
                                               j0 * 4, 0, CPOL_SC1);
        __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(u32x4, make_float4(wt[0], wt[1], wt[2], wt[3])), w_r,
                                               j0 * 4, 0, CPOL_SC1);
    }
    if (tid == 0) {
        st_dev(p.num_valid, nv);
        st_dev(p.num_valid + 1, p.M);
    }
    if (tid < MAX_ROWS / 16) {
        int n = 0;
        for (int w = 0; w < (run + 31) / 32; ++w) n += __popc(s.tile_blocks[tid][w]);
        p.tile_target[tid] = 2 * n;
    }
}

// ---- the router tiles, their top-k and the sort ----

constexpr int KSPLIT = 4;                    // K slices of the router: a wave takes 4 MFMA steps of 32
constexpr int KWAVE  = D / KSPLIT / WAVES;   // 128
constexpr int RB_MAX = 4;                    // row tiles a router CTA

union FrontLds {
    float   red[WAVES][RB_MAX][256];
    WaveLds wave[WAVES];
    SortLds sort;
};

// 16 B at agent scope (sc1), as two 8 B loads the compiler waits for
__device__ __forceinline__ uint4 ld_dev16(const void *ptr) {
    const uint64_t *q = reinterpret_cast<const uint64_t *>(ptr);
    const uint64_t  lo = __hip_atomic_load(q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    const uint64_t  hi = __hip_atomic_load(q + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    return make_uint4(uint32_t(lo), uint32_t(lo >> 32), uint32_t(hi), uint32_t(hi >> 32));
}
__device__ __forceinline__ void to_f32(uint4 v, float (&f)[4]) {
    f[0] = __builtin_bit_cast(float, v.x);
    f[1] = __builtin_bit_cast(float, v.y);
    f[2] = __builtin_bit_cast(float, v.z);
    f[3] = __builtin_bit_cast(float, v.w);
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
    // the 8 waves' sums in wave order -> this K slice's partial logits [s][row][expert]
    const int nel = RB * 256;
    for (int e = tid; e < nel; e += THREADS) {
        if (g * RB + (e >> 8) >= p.R) break; // the last group's row tiles past R
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < WAVES; ++w) v += lds.red[w][e >> 8][e & 255];
        const int row = (g * RB + (e >> 8)) * 16 + ((e >> 4) & 15);
        st_dev(p.partial + (int64_t(s) * p.R * 16 + row) * E + n * 16 + (e & 15), v);
    }
    // each row tile of the group counts this (expert tile, K slice)
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    if (tid < RB && g * RB + tid < p.R)
        __hip_atomic_fetch_add(p.counters + (C_R + g * RB + tid) * CL, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// ---- the activation quant and the zeroed output ----

// Rows first .. last - 1: x (bf16) -> MXFP4 a1 and its per-token e8m0 scales; the MoE output rows zeroed. WT (the
// sequence-parallel build): the quant stores are write-through, for the peer to pull.
template <bool WT>
__device__ void quant_rows(const Params &p, int first, int last, uint8_t *scale_lds) {
    const int tid = threadIdx.x, half = tid >> 8, t = tid & 255; // 2 rows a pass, 16 values a thread
    for (int row0 = first; row0 < last; row0 += 2) {
        const int row = row0 + half;
        if (row < last) {
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
            if (p.a1_even) {
                // the shared expert's quant (aiter's dynamic_mxfp4_quant, "even"): amax of |x| from 0, + half an
                // fp4 mantissa step, the exponent less 2 (at least 0)
                float am = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) am = max(am, abs(f[i]));
                const float am2 = __builtin_bit_cast(
                    float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, am), 0xb1, 0xf, 0xf, false));
                am = am2 > am ? am2 : am;
                const uint32_t raw = ((__builtin_bit_cast(uint32_t, am) + 0x200000u) & 0xFF800000u) >> 23;
                const uint32_t ex  = raw >= 2 ? raw - 2 : 0;
                const float    sc  = __builtin_bit_cast(float, ex << 23);
                uint32_t e0 = 0, e1 = 0;
                e0 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(e0, f[0], f[1], sc, 0);
                e0 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(e0, f[2], f[3], sc, 1);
                e0 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(e0, f[4], f[5], sc, 2);
                e0 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(e0, f[6], f[7], sc, 3);
                e1 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(e1, f[8], f[9], sc, 0);
                e1 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(e1, f[10], f[11], sc, 1);
                e1 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(e1, f[12], f[13], sc, 2);
                e1 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(e1, f[14], f[15], sc, 3);
                if constexpr (WT) {
                    st_wt8(p.a1_even + int64_t(row) * (D / 2) + t * 8, make_uint2(e0, e1));
                    if ((t & 1) == 0) st_wt1(p.a1_even_scale + int64_t(row) * SCOLS + t / 2, uint8_t(ex));
                } else {
                    *reinterpret_cast<uint2 *>(p.a1_even + int64_t(row) * (D / 2) + t * 8) = make_uint2(e0, e1);
                    if ((t & 1) == 0) p.a1_even_scale[int64_t(row) * SCOLS + t / 2] = uint8_t(ex);
                }
            }
            uint32_t q0 = 0, q1 = 0;
            q0 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q0, f[0], f[1], scale, 0);
            q0 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q0, f[2], f[3], scale, 1);
            q0 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q0, f[4], f[5], scale, 2);
            q0 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q0, f[6], f[7], scale, 3);
            q1 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q1, f[8], f[9], scale, 0);
            q1 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q1, f[10], f[11], scale, 1);
            q1 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q1, f[12], f[13], scale, 2);
            q1 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(q1, f[14], f[15], scale, 3);
            if constexpr (WT)
                st_wt8(p.a1 + int64_t(row) * (D / 2) + t * 8, make_uint2(q0, q1));
            else
                *reinterpret_cast<uint2 *>(p.a1 + int64_t(row) * (D / 2) + t * 8) = make_uint2(q0, q1);
            p.out[int64_t(row) * (D * 2 / 16) + t]       = make_uint4(0, 0, 0, 0);
            p.out[int64_t(row) * (D * 2 / 16) + 256 + t] = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        if (t < SCOLS / 4 && row < last)
            __hip_atomic_store(reinterpret_cast<uint32_t *>(p.a1_scale + int64_t(row) * SCOLS) + t,
                               reinterpret_cast<const uint32_t *>(scale_lds + half * SCOLS)[t], __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
        __syncthreads();
    }
    if (p.a1_scale_sorted) { // counted for the scale CTAs (a top-k CTA's rows), which put the count back to 0
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        if (threadIdx.x == 0)
            __hip_atomic_fetch_add(p.counters + C_QUANT * CL, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// Top-k CTA tk: rows 8 tk .. 8 tk + 7. While the router runs it quantizes those rows (and zeroes their output);
// then, a wave a row, once their row tile's 18 x 4 partial tiles are in: the K slices summed in slice order are the
// logits. The last top-k CTA to finish sorts every route and publishes the sort.
// The sequence-parallel launch's CTAs that read gen count out here; the last one advances it.
__device__ void sp_count_out(const Params &p, uint32_t gen) {
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    if (threadIdx.x == 0 &&
        __hip_atomic_fetch_add(p.ticket, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) ==
            uint32_t(p.topk_ctas + p.mover_ctas - 1)) {
        __hip_atomic_store(p.ticket, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(p.gen, gen + 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}

template <int K, int NE, bool GIVEN, bool SP>
__device__ void topk_cta(const Params &p, int tk, FrontLds &lds, uint8_t *scale_lds, int &flag) {
    const int tid = threadIdx.x, wave = tid / WAVE, lane = tid & (WAVE - 1);
    const int r = tk / 2;
    float bias[2][4];
    lane_bias(p.bias, bias); // in flight while the router runs
    uint32_t gen = 0;
    if constexpr (SP) gen = __hip_atomic_load(p.gen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    quant_rows<SP>(p, tk * WAVES, min(tk * WAVES + WAVES, p.M), scale_lds);
    if constexpr (SP) { // this CTA's quant rows are out: the peer's mover tk may pull them
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        if (tid == 0) flag_st(p.peer_flags + SP_QF + tk, gen + 1u);
    }
    if (tid == 0)
        while (ld_dev(p.counters + (C_R + r) * CL) < NT * KSPLIT) __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    // valid counts rows of the batch; this launch's rows start at row_off (0 but in the sequence-parallel build)
    const int valid = p.valid ? min(max(ld_dev(p.valid) - p.row_off, 0), p.M) : p.M;
    const int t     = tk * WAVES + wave;
    if (t < p.M) {
        int   id = -1;
        float wt = 0.0f;
        if (t < valid) {
            // lane l: logit vectors l and l + 64 (lanes past vector 71 reread it) of each K slice
            const int vi[2] = {lane, min(lane + WAVE, E / 4 - 1)};
            float q[KSPLIT][2][4];
#pragma unroll
            for (int k = 0; k < KSPLIT; ++k)
#pragma unroll
                for (int v = 0; v < 2; ++v)
                    to_f32(ld_dev16(p.partial + (int64_t(k) * p.R * 16 + t) * E + vi[v] * 4), q[k][v]);
            float lg[2][4];
#pragma unroll
            for (int v = 0; v < 2; ++v)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float x = 0.0f;
#pragma unroll
                    for (int k = 0; k < KSPLIT; ++k) x += q[k][v][i];
                    lg[v][i] = x;
                }
            float *lrow = p.logits + int64_t(t) * E;
            *reinterpret_cast<floatx4 *>(lrow + lane * 4) = floatx4{lg[0][0], lg[0][1], lg[0][2], lg[0][3]};
            if (lane + WAVE < E / 4)
                *reinterpret_cast<floatx4 *>(lrow + (lane + WAVE) * 4) = floatx4{lg[1][0], lg[1][1], lg[1][2], lg[1][3]};
            const RowScores rs = row_scores([&](int v, int i) { return bf16_f32(f32_bf16(lg[v][i])); }, bias);
            topk_row(rs, p.scale, lds.wave[wave], id, wt);
            if (GIVEN && lane < TOPK) { // a measurement's fixed routes, after the same top-k work
                id = p.given_ids[t * p.given_stride + lane];
                wt = p.given_w[t * p.given_stride + lane];
            }
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
    if (!last_arrival(p.counters + C_TOP * CL, p.topk_ctas, flag)) {
        if constexpr (SP) sp_count_out(p, gen);
        return;
    }
    // every top-k CTA is past its wait: the row-tile counts go back to 0
    if (tid < p.R) st_dev(p.counters + (C_R + tid) * CL, 0);
    if constexpr (SP) {
        // this rank's routes are out (every top-k CTA drained its sc1 stores before counting): flag them, wait for
        // the peer's, pull them into the peer's rows, then sort the whole batch
        if (tid == 0) {
            flag_st(p.peer_flags + SP_RF, gen + 1u);
            while (flag_ld(p.flags + SP_RF) < gen + 1u) __builtin_amdgcn_s_sleep(1);
        }
        __syncthreads();
        const int peer_off = p.Mf - p.M - p.row_off; // h - row_off
        const int n = p.M * K;
        for (int i = tid; i < n; i += THREADS) {
            const int64_t q = int64_t(peer_off) * K + i;
            st_dev(p.topk_ids_full + q, int(__hip_atomic_load(p.peer_topk_ids + q, __ATOMIC_RELAXED,
                                                              __HIP_MEMORY_SCOPE_SYSTEM)));
            st_dev(p.topk_w_full + q, __hip_atomic_load(p.peer_topk_w + q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
        }
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        Params ps   = p;
        ps.M        = p.Mf;
        ps.topk_ids = p.topk_ids_full;
        ps.topk_w   = p.topk_w_full;
        ps.pad_id   = (K << 24) | p.Mf;
        sort_routes<K, NE>(ps, lds.sort);
        sp_count_out(p, gen);
        return;
    }
    sort_routes<K, NE>(p, lds.sort);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    // published for the scale CTAs only (they put it back to 0), so launches without them leave it 0
    if (tid == 0 && p.a1_scale_sorted) st_dev(p.counters + C_SORTED * CL, 1);
}

// ---- the sorted, shuffled scales (aiter::mx_scale_shuffle_idx with 128 scale columns) ----

// A 32-row chunk at a time: its token ids and the tokens' 128 scale bytes into LDS, then 1024 dwords out, dword d
// holding rows (d % 16, + 16) x scale columns (8 (d / 64) + (d / 16) % 4, + 4).
__device__ void scale_cta(const Params &p, int sb, uint8_t *rows_lds, int &flag) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        while (ld_dev(p.counters + C_SORTED * CL) == 0 || ld_dev(p.counters + C_QUANT * CL) < p.topk_ctas)
            __builtin_amdgcn_s_sleep(2);
    }
    __syncthreads();
    const int chunks = ld_dev(p.num_valid) / 32;
    for (int c = sb; c < chunks; c += p.scale_ctas) {
        if (tid < 256) { // row tid / 8, 16 B part tid % 8
            const int t = ld_dev(p.sorted_ids + c * 32 + tid / 8) & 0xFFFFFF;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (t < p.M) v = ld_dev16(p.a1_scale + int64_t(t) * SCOLS + (tid % 8) * 16);
            reinterpret_cast<uint4 *>(rows_lds)[tid] = v;
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

// Mover CTA j (sequence-parallel build): once the peer's top-k CTA j has flagged its quant rows, pull those 8 rows
// of the peer's a1 / a1_scale / a1_even / a1_even_scale into the same rows here, and zero their MoE output rows.
__device__ void mover_cta(const Params &p, int j) {
    const int tid = threadIdx.x;
    const uint32_t gen = __hip_atomic_load(p.gen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (tid == 0)
        while (flag_ld(p.flags + SP_QF + j) < gen + 1u) __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    const int peer_off = p.Mf - p.M - p.row_off;
    const int r0 = peer_off + j * WAVES, r1 = peer_off + min(j * WAVES + WAVES, p.M);
    // a row: a1 128 vectors of 16 B, a1_even 128, a1_scale 8, a1_even_scale 8
    constexpr int VR = (D / 2) / 16 * 2 + SCOLS / 16 * 2;
    for (int i = tid; i < (r1 - r0) * VR; i += THREADS) {
        const int64_t row = r0 + i / VR;
        int v = i % VR;
        const uint8_t *src;
        uint8_t       *dst;
        if (v < 128) {
            src = p.peer_a1 + row * (D / 2) + v * 16, dst = p.a1_full + row * (D / 2) + v * 16;
        } else if ((v -= 128) < 128) {
            src = p.peer_a1_even + row * (D / 2) + v * 16, dst = p.a1_even_full + row * (D / 2) + v * 16;
        } else if ((v -= 128) < 8) {
            src = p.peer_a1_scale + row * SCOLS + v * 16, dst = p.a1_scale_full + row * SCOLS + v * 16;
        } else {
            v -= 8;
            src = p.peer_a1_even_scale + row * SCOLS + v * 16, dst = p.a1_even_scale_full + row * SCOLS + v * 16;
        }
        *reinterpret_cast<uint4 *>(dst) = ld_sys16(src);
    }
    for (int i = tid; i < (r1 - r0) * (D * 2 / 16); i += THREADS)
        p.out_full[int64_t(r0) * (D * 2 / 16) + i] = make_uint4(0, 0, 0, 0);
    sp_count_out(p, gen);
}

template <int K, int NE, bool GIVEN, bool SP>
__global__ __launch_bounds__(THREADS) void front_kernel(Params p) {
    __shared__ FrontLds  lds;
    __shared__ uint8_t   scale_lds[32 * SCOLS];
    __shared__ int       flag;
    const int b = blockIdx.x;
    const int t0 = p.router_ctas, s0 = t0 + p.topk_ctas;
    if (b < t0)
        router_cta(p, b, lds, flag);
    else if (b < s0)
        topk_cta<K, NE, GIVEN, SP>(p, b - t0, lds, scale_lds, flag);
    else if constexpr (SP)
        mover_cta(p, b - s0);
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
    __shared__ WaveLds lds[WAVES];
    float b[2][4];
    lane_bias(bias, b);
    const RowScores rs = row_scores([&](int v, int i) { return bf16_f32(lg[min((lane + WAVE * v) * 4 + i, E - 1)]); }, b);
    topk_row(rs, scale, lds[wave], id, wt);
    if (lane < TOPK) {
        ids[t * TOPK + lane] = id;
        w[t * TOPK + lane]   = wt;
    }
}

void check(const at::Tensor &t, at::ScalarType dt, int64_t numel, const char *name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous() && t.numel() >= numel, name,
                ": a contiguous tensor of >= ", numel, " elements of the right dtype on the GPU");
}

// Row tiles a router CTA takes: at most 3 row groups up to 192 rows (at most 216 router CTAs, so they and the top-k
// CTAs each have a CU), 4 above.
int row_tiles_per_cta(int R) { return R <= 12 ? (R + 2) / 3 : RB_MAX; }

void front(const at::Tensor &x, const at::Tensor &w, const at::Tensor &bias, double scale,
           const c10::optional<at::Tensor> &valid, at::Tensor &partial, at::Tensor &logits, at::Tensor &topk_ids,
           at::Tensor &topk_w, at::Tensor &sorted_ids, at::Tensor &sorted_w, at::Tensor &sorted_eids,
           at::Tensor &num_valid, at::Tensor &tile_target, at::Tensor &a1, at::Tensor &a1_scale,
           const c10::optional<at::Tensor> &a1_scale_sorted, const c10::optional<at::Tensor> &a1_even,
           const c10::optional<at::Tensor> &a1_even_scale, at::Tensor &out, at::Tensor &counters,
           int64_t topk_total, const c10::optional<at::Tensor> &given_ids, const c10::optional<at::Tensor> &given_w) {
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
    check(partial, at::kFloat, int64_t(KSPLIT) * R * 16 * E, "partial");
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
    check(tile_target, at::kInt, MAX_ROWS / 16, "tile_target");
    p.tile_target = tile_target.data_ptr<int>();
    p.a1          = reinterpret_cast<uint8_t *>(a1.data_ptr());
    p.a1_scale    = reinterpret_cast<uint8_t *>(a1_scale.data_ptr());
    p.out         = reinterpret_cast<uint4 *>(out.data_ptr());
    p.counters    = counters.data_ptr<int>();
    p.M           = int(M);
    p.R           = R;
    p.RB          = RB;
    p.router_ctas = G * NT * KSPLIT;
    p.topk_ctas   = int((M + WAVES - 1) / WAVES);
    p.scale_ctas  = 0;
    p.pad_id      = (K << 24) | int(M);
    if (a1_even.has_value()) {
        TORCH_CHECK(a1_even_scale.has_value(), "a1_even and a1_even_scale together");
        TORCH_CHECK(a1_even->is_cuda() && a1_even->is_contiguous() && a1_even->numel() * a1_even->element_size() == M * D / 2 &&
                        a1_even_scale->is_cuda() && a1_even_scale->is_contiguous() &&
                        a1_even_scale->numel() * a1_even_scale->element_size() == M * SCOLS,
                    "a1_even [M, 2048] fp4x2 and a1_even_scale [M, 128] e8m0");
        p.a1_even       = reinterpret_cast<uint8_t *>(a1_even->data_ptr());
        p.a1_even_scale = reinterpret_cast<uint8_t *>(a1_even_scale->data_ptr());
    }
    if (given_ids.has_value()) {
        TORCH_CHECK(given_w.has_value(), "given_ids and given_w together");
        TORCH_CHECK(given_ids->is_cuda() && given_ids->scalar_type() == at::kInt && given_w->scalar_type() == at::kFloat &&
                        given_ids->dim() == 2 && given_ids->size(0) == M && given_ids->size(1) >= TOPK &&
                        given_ids->stride(1) == 1 && given_w->sizes() == given_ids->sizes() &&
                        given_w->strides() == given_ids->strides(),
                    "given routes: [M, >= 8] i32 ids and f32 weights, unit column stride, the same strides");
        p.given_ids    = given_ids->data_ptr<int>();
        p.given_w      = given_w->data_ptr<float>();
        p.given_stride = int(given_ids->stride(0));
    }
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
    const int grid = p.router_ctas + p.topk_ctas + p.scale_ctas;
    const bool given = p.given_ids != nullptr;
    if (K == TOPK && !given)
        front_kernel<TOPK, E, false, false><<<grid, THREADS, 0, stream>>>(p);
    else if (K == TOPK)
        front_kernel<TOPK, E, true, false><<<grid, THREADS, 0, stream>>>(p);
    else if (!given)
        front_kernel<TOPK + 1, E + 1, false, false><<<grid, THREADS, 0, stream>>>(p);
    else
        front_kernel<TOPK + 1, E + 1, true, false><<<grid, THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

// The sequence-parallel front at TP2 (moe_fused's build: the shared slot, no sorted scales, no given routes): x [M,
// 4096] this rank's rows of a 2 M-row batch starting at row_off; the batch arrays (topk, a1, a1_scale, a1_even,
// a1_even_scale, out) hold 2 M rows and are written whole on both ranks; ``peer``: the peer's batch arrays' device
// addresses (topk_ids, topk_w, a1, a1_scale, a1_even, a1_even_scale) and tp2_xchg's (peer_flags, flags, gen, ticket).
void front_sp(const at::Tensor &x, const at::Tensor &w, const at::Tensor &bias, double scale,
              const c10::optional<at::Tensor> &valid, at::Tensor &partial, at::Tensor &logits, at::Tensor &topk_ids,
              at::Tensor &topk_w, at::Tensor &sorted_ids, at::Tensor &sorted_w, at::Tensor &sorted_eids,
              at::Tensor &num_valid, at::Tensor &tile_target, at::Tensor &a1, at::Tensor &a1_scale, at::Tensor &a1_even,
              at::Tensor &a1_even_scale, at::Tensor &out, at::Tensor &counters, int64_t row_off,
              at::IntArrayRef peer) {
    const int64_t M = x.size(0), Mf = 2 * M;
    constexpr int K = TOPK + 1, NE = E + 1;
    TORCH_CHECK(x.dim() == 2 && x.size(1) == D && x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                    x.scalar_type() == at::kBFloat16 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "x: [M, 4096] bf16, unit column stride, 16 B aligned rows");
    TORCH_CHECK(M >= 1 && Mf <= MAX_ROWS && (row_off == 0 || row_off == M), "1 to 128 rows a rank, row_off 0 or M");
    TORCH_CHECK(peer.size() == 10, "peer: 6 array and 4 exchange addresses");
    const int R = int((M + 15) / 16), RB = row_tiles_per_cta(R), G = (R + RB - 1) / RB;
    check(w, at::kBFloat16, int64_t(E) * D, "w");
    check(bias, at::kBFloat16, E, "bias");
    check(partial, at::kFloat, int64_t(KSPLIT) * R * 16 * E, "partial");
    check(logits, at::kFloat, int64_t(R) * 16 * E, "logits");
    check(topk_ids, at::kInt, Mf * K, "topk_ids");
    check(topk_w, at::kFloat, Mf * K, "topk_w");
    const int64_t cap = sorted_ids.numel();
    const int64_t P   = Mf * K;
    TORCH_CHECK(cap % BM == 0 && cap >= P + std::min<int64_t>(NE, P) * (BM - 1), "sorted rows");
    check(sorted_w, at::kFloat, cap, "sorted_w");
    check(sorted_eids, at::kInt, cap / BM, "sorted_eids");
    check(num_valid, at::kInt, 2, "num_valid");
    check(tile_target, at::kInt, MAX_ROWS / 16, "tile_target");
    for (const at::Tensor *t : {&a1, &a1_even})
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->numel() * t->element_size() == Mf * D / 2, "a1 / a1_even");
    for (const at::Tensor *t : {&a1_scale, &a1_even_scale})
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->numel() * t->element_size() == Mf * SCOLS,
                    "a1_scale / a1_even_scale");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.numel() == Mf * D,
                "out: [2 M, 4096] bf16");
    check(counters, at::kInt, N_COUNTERS * CL, "counters");
    auto u8 = [](const at::Tensor &t) { return reinterpret_cast<uint8_t *>(t.data_ptr()); };
    Params p{};
    p.x          = reinterpret_cast<const uint16_t *>(x.data_ptr());
    p.x_stride   = x.stride(0);
    p.w          = reinterpret_cast<const uint16_t *>(w.data_ptr());
    p.bias       = reinterpret_cast<const uint16_t *>(bias.data_ptr());
    p.scale      = float(scale);
    p.partial    = partial.data_ptr<float>();
    p.logits     = logits.data_ptr<float>();
    p.topk_ids_full = topk_ids.data_ptr<int>();
    p.topk_w_full   = topk_w.data_ptr<float>();
    p.topk_ids   = p.topk_ids_full + row_off * K;
    p.topk_w     = p.topk_w_full + row_off * K;
    p.sorted_ids  = sorted_ids.data_ptr<int>();
    p.sorted_w    = sorted_w.data_ptr<float>();
    p.sorted_eids = sorted_eids.data_ptr<int>();
    p.num_valid   = num_valid.data_ptr<int>();
    p.tile_target = tile_target.data_ptr<int>();
    p.a1_full            = u8(a1);
    p.a1_scale_full      = u8(a1_scale);
    p.a1_even_full       = u8(a1_even);
    p.a1_even_scale_full = u8(a1_even_scale);
    p.a1            = p.a1_full + row_off * (D / 2);
    p.a1_scale      = p.a1_scale_full + row_off * SCOLS;
    p.a1_even       = p.a1_even_full + row_off * (D / 2);
    p.a1_even_scale = p.a1_even_scale_full + row_off * SCOLS;
    p.out_full    = reinterpret_cast<uint4 *>(out.data_ptr());
    p.out         = p.out_full + row_off * (D * 2 / 16);
    p.counters    = counters.data_ptr<int>();
    p.M           = int(M);
    p.Mf          = int(Mf);
    p.row_off     = int(row_off);
    p.R           = R;
    p.RB          = RB;
    p.router_ctas = G * NT * KSPLIT;
    p.topk_ctas   = int((M + WAVES - 1) / WAVES);
    p.mover_ctas  = p.topk_ctas;
    p.scale_ctas  = 0;
    p.pad_id      = (K << 24) | int(M); // the sort uses the batch's, (K << 24) | 2 M
    p.peer_topk_ids      = reinterpret_cast<const int *>(peer[0]);
    p.peer_topk_w        = reinterpret_cast<const float *>(peer[1]);
    p.peer_a1            = reinterpret_cast<const uint8_t *>(peer[2]);
    p.peer_a1_scale      = reinterpret_cast<const uint8_t *>(peer[3]);
    p.peer_a1_even       = reinterpret_cast<const uint8_t *>(peer[4]);
    p.peer_a1_even_scale = reinterpret_cast<const uint8_t *>(peer[5]);
    p.peer_flags = reinterpret_cast<uint32_t *>(peer[6]);
    p.flags      = reinterpret_cast<const uint32_t *>(peer[7]);
    p.gen        = reinterpret_cast<uint32_t *>(peer[8]);
    p.ticket     = reinterpret_cast<uint32_t *>(peer[9]);
    if (valid.has_value()) {
        TORCH_CHECK(valid->is_cuda() && valid->numel() == 1 && valid->scalar_type() == at::kInt,
                    "valid: one int32 on the GPU");
        p.valid = valid->data_ptr<int>();
    }
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(x.device().index()).stream();
    const int grid = p.router_ctas + p.topk_ctas + p.mover_ctas;
    front_kernel<K, NE, false, true><<<grid, THREADS, 0, stream>>>(p);
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
          "Tensor(g!) sorted_eids, Tensor(h!) num_valid, Tensor(n!) tile_target, Tensor(i!) a1, Tensor(j!) a1_scale, "
          "Tensor(k!)? a1_scale_sorted, Tensor(o!)? a1_even, Tensor(p!)? a1_even_scale, Tensor(l!) out, "
          "Tensor(m!) counters, int topk_total, "
          "Tensor? given_ids=None, Tensor? given_w=None) -> ()");
    m.impl("front", at::kCUDA, &front);
    m.def("front_sp(Tensor x, Tensor w, Tensor bias, float scale, Tensor? valid, Tensor(a!) partial, "
          "Tensor(b!) logits, Tensor(c!) topk_ids, Tensor(d!) topk_w, Tensor(e!) sorted_ids, Tensor(f!) sorted_w, "
          "Tensor(g!) sorted_eids, Tensor(h!) num_valid, Tensor(n!) tile_target, Tensor(i!) a1, Tensor(j!) a1_scale, "
          "Tensor(o!) a1_even, Tensor(p!) a1_even_scale, Tensor(l!) out, Tensor(m!) counters, int row_off, "
          "int[] peer) -> ()");
    m.impl("front_sp", at::kCUDA, &front_sp);
    m.def("topk(Tensor logits, Tensor bias, float scale, Tensor(a!) ids, Tensor(b!) w) -> ()");
    m.impl("topk", at::kCUDA, &topk);
}
