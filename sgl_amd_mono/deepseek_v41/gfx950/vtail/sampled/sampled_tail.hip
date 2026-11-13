// SPDX-License-Identifier: Apache-2.0
// A temperature-sampled static verify's accept from the ranks' bf16 logits shards and the draft's f32 step logits
// shards, in one launch: SGLang's block-wise rejection (blockwise_rejection.py) and simulated bonus
// (simulated_bonus.py), the static accept tail (static_accept_tail.py) and the Engram history commit.
// - each block takes 512 columns of every row of every request: the target's and the draft's max and sum of exp at
//   the request's temperature into this rank's partials, the drafted tokens' logits (from the block that holds the
//   token) to every rank, and under a simulated accept length that row's Gumbel-noised key (the simulated bonus) to
//   every rank;
// - the rank's last block to arrive reduces the partials and sends the rank's statistics to every rank;
// - every block forms each row's log-sum-exp from every rank's statistics and runs the chain accept (the same coins
//   on every rank and block), then keys its columns of the rejected row's residual max(p - q, 0) (or of p at the
//   bonus row) with the resample noise and sends that key to every rank;
// - each rank's block 0 takes the best key a request and runs the accept tail. Under a simulated length the chain
//   and the resample still run; the length and the bonus are then the simulated ones, as SGLang's region does.
// The draws are SGLang's: the coins tl.rand(key + counter * mult, row) and the resample noise its murmur3 uniforms
// of (seed, request, global vocab index); block 0 advances the counter.
#include <climits>
#include <cstdint>
#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#include "main_proj.h"

#ifndef C10_HIP_KERNEL_LAUNCH_CHECK
#define C10_HIP_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

namespace {

using bfloat16 = __bf16;

constexpr int THREADS = 256;
constexpr int WAVE    = 64;
constexpr int WAVES   = THREADS / WAVE;
constexpr int COLS    = 2;              // a thread's columns of a row
constexpr int CHUNK   = THREADS * COLS; // a block's columns
constexpr int MAX_TP  = 8;
constexpr int MAX_NB  = 128; // blocks a rank
constexpr int MAX_SEQ = 4;   // requests
constexpr int MAX_T   = 8;   // verify rows a request
constexpr int MAX_G   = MAX_T - 1;
constexpr int STATS   = 4; // target max, target sum, draft max, draft sum
constexpr int POLLS   = 4; // records a thread keeps in flight
// records, 8 B each, a launch parity apart: [stats: request][draft row][stat][source rank], [tokens: request][draft
// row][target, draft], [keys: request][residual, simulated][source rank][block]
constexpr int     STAT_RECORDS   = MAX_SEQ * MAX_G * STATS * MAX_TP;
constexpr int     TOK_RECORDS    = MAX_SEQ * MAX_G * 2;
constexpr int     KEY_RECORDS    = MAX_SEQ * 2 * MAX_TP * MAX_NB;
constexpr int     PARITY_RECORDS = STAT_RECORDS + TOK_RECORDS + KEY_RECORDS;
constexpr int64_t RECORD_BYTES   = 2ll * PARITY_RECORDS * 8;
// this rank's scratch: each block's partials [block][request][draft row][stat] f32, then the arrival count (u64)
constexpr int64_t PARTIALS      = int64_t{MAX_NB} * MAX_SEQ * MAX_G * STATS;
constexpr int64_t SCRATCH_BYTES = PARTIALS * 4 + 8;
// a value record: the f32 bits above a 32-bit launch tag; a key record: the candidate key above a 15-bit tag
constexpr int      TAG_BITS    = 15;
constexpr uint64_t TAG_MASK    = (1ull << TAG_BITS) - 1;
constexpr int      IDX_BITS    = 17; // the vocab (129280) below 2^17
constexpr uint32_t IDX_MASK    = (1u << IDX_BITS) - 1;
constexpr int      MAX_HISTORY = 8;
// blockwise_rejection.py's constants
constexpr uint64_t STEP_SEED_MULT = 0x632BE59BD9B4E019ull;
constexpr float    TL_RAND_MIN    = 4.6566127342e-10f;

struct Params {
    const bfloat16 *local; // this rank's target shard: request r's verify row j at local + (r * T + j) * local_stride
    int64_t         local_stride;
    const float    *draft; // this rank's draft step logits: request r's step j at draft + r * dseq + j * dstep
    int64_t         dseq, dstep;
    const float    *temps;   // [bs] each request's temperature (> 0)
    int64_t         key;     // the rejection noise's key
    int64_t        *counter; // its step counter [1]
    const int64_t  *sim;     // the simulated correct length [1], or null
    const int32_t  *sim_seed; // [1] (with sim)
    int             width, vocab_start, vocab, blocks, rank, tp, bs, T;
    const int64_t  *candidates; // request r's anchor and drafts at candidates + r * cand_stride, [T]
    int64_t         cand_stride;
    const int64_t  *seq_lens, *verify_lens;
    const int32_t  *gate;
    int64_t        *correct_out, *bonus_out, *new_seq_out, *out_tokens;
    int32_t        *cap_trim_out, *commit_out, *gated_out;
    int64_t         out_tokens_stride;
    int32_t        *history; // the Engram history commit (or null), as vtail.hip's
    int64_t         history_stride;
    int             history_width;
    const int64_t  *slots;
    int64_t        *relay_lens, *relay_tokens;
    float          *partials; // this rank's scratch
    uint64_t       *arrived;
    uint32_t       *counts; // [blocks] each block's launches
    void           *recv[MAX_TP];
    uint64_t        deadline_ticks;
};

// ---- SGLang's arithmetic and draws ----

// Philox4x32-10 (tl.philox's), the first word
__device__ inline uint32_t philox_c0(uint32_t k0, uint32_t k1, uint32_t c0) {
    uint32_t c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const uint32_t a0 = c0, a2 = c2;
        c0 = __umulhi(0xCD9E8D57u, a2) ^ c1 ^ k0;
        c2 = __umulhi(0xD2511F53u, a0) ^ c3 ^ k1;
        c1 = 0xCD9E8D57u * a2;
        c3 = 0xD2511F53u * a0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c0;
}

// tl.rand(seed, offset) of a 64-bit seed and a 32-bit offset
__device__ inline float uniform(int64_t seed, int32_t offset) {
    const uint64_t s = static_cast<uint64_t>(seed);
    int32_t        x = static_cast<int32_t>(
        philox_c0(static_cast<uint32_t>(s), static_cast<uint32_t>(s >> 32), static_cast<uint32_t>(offset)));
    x = x < 0 ? -x - 1 : x;
    return static_cast<float>(x) * 4.6566127342e-10f;
}

__device__ inline uint32_t rotl32(uint32_t v, int s) { return (v << s) | (v >> (32 - s)); }

__device__ inline uint32_t murmur3_mix(uint32_t h, uint32_t k) {
    k *= 0xCC9E2D51u;
    k = rotl32(k, 15);
    k *= 0x1B873593u;
    h ^= k;
    h = rotl32(h, 13);
    return h * 5u + 0xE6546B64u;
}

// _murmur3_uniform32(seed, pos, offset)
__device__ inline float murmur3_uniform(int64_t seed, int64_t pos, uint32_t offset) {
    const uint64_t s = static_cast<uint64_t>(seed), q = static_cast<uint64_t>(pos);
    uint32_t       h = 0;
    h = murmur3_mix(h, static_cast<uint32_t>(s));
    h = murmur3_mix(h, static_cast<uint32_t>(s >> 32));
    h = murmur3_mix(h, static_cast<uint32_t>(q));
    h = murmur3_mix(h, offset);
    h ^= 16u;
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    const int32_t hi16 = static_cast<int32_t>(h >> 16), lo16 = static_cast<int32_t>(h & 0xFFFFu);
    return static_cast<float>(hi16) * 1.52587890625e-05f + (static_cast<float>(lo16) + 0.5f) * 2.3283064365386963e-10f;
}

// _log1p_neg_stable: log(1 - v)
__device__ inline float log1p_neg_stable(float v) {
    float poly = 1.0f / 8.0f;
    poly       = 1.0f / 7.0f + v * poly;
    poly       = 1.0f / 6.0f + v * poly;
    poly       = 1.0f / 5.0f + v * poly;
    poly       = 1.0f / 4.0f + v * poly;
    poly       = 1.0f / 3.0f + v * poly;
    poly       = 1.0f / 2.0f + v * poly;
    poly       = 1.0f + v * poly;
    const float series = -v * poly;
    const float direct = logf(fmaxf(1.0f - v, 5.960464477539063e-08f));
    return v < 0.25f ? series : direct;
}

// ---- keys, records, reductions ----

// a candidate as one ordered word: the value's bits in order above the inverted global index (0: none); NaN is never
// one, as the float compare never keeps it
__device__ inline uint64_t candidate_key(float value, int index) {
    value               = value == 0.0f ? 0.0f : value;
    const uint32_t bits = __builtin_bit_cast(uint32_t, value);
    const uint32_t ord  = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    return (static_cast<uint64_t>(ord) << IDX_BITS) | static_cast<uint64_t>(IDX_MASK - static_cast<uint32_t>(index));
}

__device__ inline int64_t key_index(uint64_t key) {
    return static_cast<int64_t>(IDX_MASK - static_cast<uint32_t>(key & IDX_MASK));
}

__device__ inline uint64_t umax64(uint64_t a, uint64_t b) { return a > b ? a : b; }

// wave reductions without LDS: DPP within each row of 16 lanes, then the four rows' results by readlane
template <int CTRL> __device__ inline uint32_t dpp(uint32_t v) {
    return static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(v), CTRL, 0xF, 0xF, false));
}
constexpr int ROR8 = 0x128, ROR4 = 0x124, XOR2 = 0x4E, XOR1 = 0xB1;

template <typename Op> __device__ inline float wave_reduce(float v, Op op) {
    v = op(v, __builtin_bit_cast(float, dpp<ROR8>(__builtin_bit_cast(uint32_t, v))));
    v = op(v, __builtin_bit_cast(float, dpp<ROR4>(__builtin_bit_cast(uint32_t, v))));
    v = op(v, __builtin_bit_cast(float, dpp<XOR2>(__builtin_bit_cast(uint32_t, v))));
    v = op(v, __builtin_bit_cast(float, dpp<XOR1>(__builtin_bit_cast(uint32_t, v))));
    const auto lane = [&](int l) {
        return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), l));
    };
    return op(op(lane(0), lane(16)), op(lane(32), lane(48)));
}

__device__ inline uint64_t wave_umax64(uint64_t x) {
#define ST_DPP64(C) (static_cast<uint64_t>(dpp<C>(static_cast<uint32_t>(x >> 32))) << 32 | dpp<C>(static_cast<uint32_t>(x)))
    x = umax64(x, ST_DPP64(ROR8));
    x = umax64(x, ST_DPP64(ROR4));
    x = umax64(x, ST_DPP64(XOR2));
    x = umax64(x, ST_DPP64(XOR1));
#undef ST_DPP64
    const auto lane = [&](int l) {
        const uint32_t lo = __builtin_amdgcn_readlane(static_cast<int>(static_cast<uint32_t>(x)), l);
        const uint32_t hi = __builtin_amdgcn_readlane(static_cast<int>(static_cast<uint32_t>(x >> 32)), l);
        return (static_cast<uint64_t>(hi) << 32) | lo;
    };
    return umax64(umax64(lane(0), lane(16)), umax64(lane(32), lane(48)));
}

struct MaxOp {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct SumOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};

// the block's barrier for LDS alone: the records' stores to the peers stay in flight across it (__syncthreads
// would wait for them)
__device__ inline void lds_sync() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

// block-wide reductions of N values a thread: each wave's, then the waves' through LDS (red: [N][WAVES])
template <int N, typename Op> __device__ inline void block_reduce(float (&v)[N], float (*red)[WAVES], Op op) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = wave_reduce(v[i], op);
    if (threadIdx.x % WAVE == 0) {
#pragma unroll
        for (int i = 0; i < N; ++i) red[i][threadIdx.x / WAVE] = v[i];
    }
    lds_sync();
#pragma unroll
    for (int i = 0; i < N; ++i) {
        float m = red[i][0];
#pragma unroll
        for (int w = 1; w < WAVES; ++w) m = op(m, red[i][w]);
        v[i] = m;
    }
    lds_sync();
}

__device__ inline uint64_t block_key_max(uint64_t k, uint64_t *red) {
    k = wave_umax64(k);
    if (threadIdx.x % WAVE == 0) red[threadIdx.x / WAVE] = k;
    lds_sync();
    uint64_t m = red[0];
#pragma unroll
    for (int w = 1; w < WAVES; ++w) m = umax64(m, red[w]);
    lds_sync();
    return m;
}

__device__ inline int64_t stat_slot(int parity, int r, int j, int s, int src) {
    return static_cast<int64_t>(parity) * PARITY_RECORDS + ((r * MAX_G + j) * STATS + s) * MAX_TP + src;
}
__device__ inline int64_t tok_slot(int parity, int r, int j, int which) {
    return static_cast<int64_t>(parity) * PARITY_RECORDS + STAT_RECORDS + (r * MAX_G + j) * 2 + which;
}
__device__ inline int64_t key_slot(int parity, int r, int which, int src, int block) {
    return static_cast<int64_t>(parity) * PARITY_RECORDS + STAT_RECORDS + TOK_RECORDS +
           ((r * 2 + which) * MAX_TP + src) * MAX_NB + block;
}
__device__ inline int64_t partial_slot(int block, int r, int j, int s) {
    return ((static_cast<int64_t>(block) * MAX_SEQ + r) * MAX_G + j) * STATS + s;
}

// another block's partial, past this CU's cache
__device__ inline float partial(const Params &p, int64_t slot) {
    return __builtin_bit_cast(float, __hip_atomic_load(reinterpret_cast<uint32_t *>(p.partials) + slot,
                                                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
}

__device__ inline void push(const Params &p, int64_t slot, uint64_t record) {
    for (int dst = 0; dst < p.tp; ++dst) {
        __hip_atomic_store(static_cast<uint64_t *>(p.recv[dst]) + slot, record, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

__device__ inline uint64_t value_record(float v, uint32_t tag) {
    return (static_cast<uint64_t>(__builtin_bit_cast(uint32_t, v)) << 32) | tag;
}

__device__ inline float record_value(uint64_t v) { return __builtin_bit_cast(float, static_cast<uint32_t>(v >> 32)); }

// polls the n records at slots slot(i) of this rank's buffer until each carries its tag (match(v)), POLLS a thread in
// flight, handing each to take(i, v)
template <typename Slot, typename Match, typename Take>
__device__ inline void poll(const Params &p, int n, Slot slot, Match match, Take take) {
    const uint64_t *mine = static_cast<const uint64_t *>(p.recv[p.rank]);
    for (int base = 0; base < n; base += THREADS * POLLS) {
        uint32_t pending = 0;
        int64_t  at[POLLS];
#pragma unroll
        for (int k = 0; k < POLLS; ++k) {
            const int i = base + k * THREADS + static_cast<int>(threadIdx.x);
            if (i < n) {
                pending |= 1u << k;
                at[k] = slot(i);
            }
        }
        uint64_t t0 = 0;
        for (int it = 1; pending; ++it) {
            uint64_t v[POLLS];
#pragma unroll
            for (int k = 0; k < POLLS; ++k) {
                if ((pending >> k) & 1u) {
                    v[k] = __hip_atomic_load(const_cast<uint64_t *>(mine) + at[k], __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_SYSTEM);
                }
            }
#pragma unroll
            for (int k = 0; k < POLLS; ++k) {
                if (((pending >> k) & 1u) && match(v[k])) {
                    take(base + k * THREADS + static_cast<int>(threadIdx.x), v[k]);
                    pending &= ~(1u << k);
                }
            }
            if (pending && it % 1024 == 0) { // past the deadline the launch traps, stopping this rank's step
                const uint64_t now = __builtin_amdgcn_s_memrealtime();
                if (t0 == 0) {
                    t0 = now;
                } else if (now - t0 > p.deadline_ticks) {
                    __builtin_trap();
                }
            }
        }
    }
}

// a thread's columns of every one of N rows from c (-inf past width), all loads issued before any is used
template <int N>
__device__ inline void load_targets(const bfloat16 *base, int64_t stride, int c, int width, float (&x)[N][COLS]) {
    uint32_t   raw[N];
    const bool whole = c + COLS <= width;
#pragma unroll
    for (int j = 0; j < N; ++j) {
        if (whole) raw[j] = *reinterpret_cast<const uint32_t *>(base + j * stride + c);
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
        if (whole) {
            const bfloat16 *b = reinterpret_cast<const bfloat16 *>(&raw[j]);
#pragma unroll
            for (int i = 0; i < COLS; ++i) x[j][i] = static_cast<float>(b[i]);
        } else {
#pragma unroll
            for (int i = 0; i < COLS; ++i) {
                x[j][i] = c + i < width ? static_cast<float>(base[j * stride + c + i]) : -INFINITY;
            }
        }
    }
}

template <int N>
__device__ inline void load_drafts(const float *base, int64_t stride, int c, int width, float (&x)[N][COLS]) {
    float2     a[N];
    const bool whole = c + COLS <= width;
#pragma unroll
    for (int j = 0; j < N; ++j) {
        if (whole) a[j] = *reinterpret_cast<const float2 *>(base + j * stride + c);
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
        if (whole) {
            x[j][0] = a[j].x, x[j][1] = a[j].y;
        } else {
#pragma unroll
            for (int i = 0; i < COLS; ++i) x[j][i] = c + i < width ? base[j * stride + c + i] : -INFINITY;
        }
    }
}

// row k of N register rows (k runtime, below N)
template <int N> __device__ inline void pick(const float (&x)[N][COLS], int k, float (&out)[COLS]) {
#pragma unroll
    for (int i = 0; i < COLS; ++i) out[i] = x[0][i];
#pragma unroll
    for (int j = 1; j < N; ++j) {
        if (j == k) {
#pragma unroll
            for (int i = 0; i < COLS; ++i) out[i] = x[j][i];
        }
    }
}

// T verify rows a request (G = T - 1 drafts)
template <int T> __device__ inline void sampled_tail_body(const Params &p) {
    constexpr int G = T - 1;
    __shared__ float    s_stat[MAX_SEQ][G][STATS][MAX_TP];
    __shared__ float    s_tok[MAX_SEQ][G][2];
    __shared__ float    s_lse[MAX_SEQ][G][2];
    __shared__ float    s_logu[MAX_SEQ][G]; // the chain's coins, log u
    __shared__ int64_t  s_cand[MAX_SEQ][T];
    __shared__ int64_t  s_seq[MAX_SEQ], s_vlen[MAX_SEQ];
    __shared__ float    s_temp[MAX_SEQ];
    __shared__ int      s_acc[MAX_SEQ];
    __shared__ uint64_t s_key[MAX_SEQ][2];
    __shared__ float    s_red[2 * G][WAVES];
    __shared__ uint64_t s_kred[WAVES];
    __shared__ uint32_t s_count;
    __shared__ int      s_last;
    __shared__ int64_t  s_counter, s_sim;
    __shared__ int32_t  s_sim_seed;
    // every small input at once
    if (threadIdx.x == 0) {
        s_count    = p.counts[blockIdx.x];
        s_counter  = p.counter[0];
        s_sim      = p.sim ? p.sim[0] : 0;
        s_sim_seed = p.sim ? p.sim_seed[0] : 0;
    }
    if (static_cast<int>(threadIdx.x) < p.bs * T) {
        const int r = threadIdx.x / T, j = threadIdx.x % T;
        s_cand[r][j] = p.candidates[r * p.cand_stride + j];
    }
    if (static_cast<int>(threadIdx.x) < p.bs) {
        const int r = threadIdx.x;
        s_temp[r]   = p.temps[r];
        s_seq[r]    = p.seq_lens[r];
        s_vlen[r]   = p.verify_lens[r];
    }
    if (threadIdx.x < MAX_SEQ * 2) s_key[threadIdx.x / 2][threadIdx.x % 2] = 0;
    lds_sync();
    const uint32_t count  = s_count;
    const uint32_t vtag   = count + 1; // never 0 in 2^32 launches; the launch two back differs
    const uint64_t ktag   = 1 + count % static_cast<uint32_t>(TAG_MASK);
    const int      parity = static_cast<int>(count & 1u);
    const int64_t  seed = static_cast<int64_t>(static_cast<uint64_t>(p.key) +
                                              static_cast<uint64_t>(s_counter) * STEP_SEED_MULT);
    const int c    = blockIdx.x * CHUNK + threadIdx.x * COLS; // this thread's first column (local)
    const int sim  = static_cast<int>(s_sim);
    const int wave = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;

    // the block's statistics of every draft row (to this rank's partials), the drafted tokens' logits and the
    // simulated row's key (to every rank); the last request's rows stay in registers (at temperature)
    float x[T][COLS], d[G][COLS];
    for (int r = 0; r < p.bs; ++r) {
        const float t = s_temp[r];
        float       tv = -INFINITY, dv = -INFINITY;
        bool        owner = false, tagged = false;
        if (static_cast<int>(threadIdx.x) < G) {
            const int     j   = threadIdx.x;
            const int64_t tok = s_cand[r][1 + j];
            const int64_t loc = tok - p.vocab_start;
            owner  = tok >= 0 && tok < p.vocab && loc >= 0 && loc < p.width && loc / CHUNK == blockIdx.x;
            tagged = (tok < 0 || tok >= p.vocab) && p.rank == 0 && blockIdx.x == 0; // no logits: rank 0 tags them
            if (owner) {
                tv = static_cast<float>(p.local[(r * T + j) * p.local_stride + loc]);
                dv = p.draft[r * p.dseq + j * p.dstep + loc];
            }
        }
        load_targets<T>(p.local + r * T * p.local_stride, p.local_stride, c, p.width, x);
        load_drafts<G>(p.draft + r * p.dseq, p.dstep, c, p.width, d);
        if (static_cast<int>(threadIdx.x) >= WAVE && static_cast<int>(threadIdx.x) < WAVE + G) {
            const int j  = threadIdx.x - WAVE;
            s_logu[r][j] = logf(fmaxf(uniform(seed, r * T + j), TL_RAND_MIN));
        }
        float raw_sim[COLS];
        pick<T>(x, sim, raw_sim);
#pragma unroll
        for (int j = 0; j < T; ++j) {
#pragma unroll
            for (int i = 0; i < COLS; ++i) x[j][i] = x[j][i] / t;
        }
        float m[2 * G], sum[2 * G];
#pragma unroll
        for (int j = 0; j < G; ++j) {
            m[2 * j] = m[2 * j + 1] = -INFINITY;
#pragma unroll
            for (int i = 0; i < COLS; ++i) {
                d[j][i]      = d[j][i] / t;
                m[2 * j]     = fmaxf(m[2 * j], x[j][i]);
                m[2 * j + 1] = fmaxf(m[2 * j + 1], d[j][i]);
            }
        }
        block_reduce<2 * G>(m, s_red, MaxOp{});
#pragma unroll
        for (int j = 0; j < G; ++j) {
            sum[2 * j] = sum[2 * j + 1] = 0.0f;
#pragma unroll
            for (int i = 0; i < COLS; ++i) {
                if (m[2 * j] > -INFINITY) sum[2 * j] += expf(x[j][i] - m[2 * j]);
                if (m[2 * j + 1] > -INFINITY) sum[2 * j + 1] += expf(d[j][i] - m[2 * j + 1]);
            }
        }
        block_reduce<2 * G>(sum, s_red, SumOp{});
        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < G; ++j) {
                float *at = p.partials + partial_slot(blockIdx.x, r, j, 0);
                at[0] = m[2 * j], at[1] = sum[2 * j], at[2] = m[2 * j + 1], at[3] = sum[2 * j + 1];
            }
        }
        if (owner || tagged) {
            const int j = threadIdx.x;
            push(p, tok_slot(parity, r, j, 0), value_record(tv, vtag));
            push(p, tok_slot(parity, r, j, 1), value_record(dv, vtag));
        }
        if (p.sim) { // _simulated_bonus_row_partial_kernel's noisy value at the simulated row
            uint64_t best = 0;
#pragma unroll
            for (int i = 0; i < COLS; ++i) {
                const int g = p.vocab_start + c + i;
                if (c + i < p.width) {
                    const float u = uniform(static_cast<int64_t>(s_sim_seed), r * p.vocab + g);
                    best = umax64(best, candidate_key(raw_sim[i] / fmaxf(t, 1e-5f) - logf(-logf(u)), g));
                }
            }
            best = block_key_max(best, s_kred);
            if (threadIdx.x == 0) push(p, key_slot(parity, r, 1, p.rank, blockIdx.x), (best << TAG_BITS) | ktag);
        }
    }
    // this block's partials are this rank's once it arrives; the last block to arrive reduces them
    if (threadIdx.x == 0) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        const uint64_t before = __hip_atomic_fetch_add(p.arrived, 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        s_last = (before + 1) % static_cast<uint64_t>(p.blocks) == 0;
    }
    lds_sync();
    if (s_last) {
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        // a wave a (request, draft row, target or draft), the blocks across the lanes; every one of the wave's
        // partials loaded before any is reduced
        constexpr int ROUNDS = (MAX_SEQ * G * 2 + WAVES - 1) / WAVES, NQ = MAX_NB / WAVE;
        const int     n      = p.bs * G * 2;
        float         sm[ROUNDS][NQ], mb[ROUNDS][NQ];
#pragma unroll
        for (int u = 0; u < ROUNDS; ++u) {
            const int k = wave + u * WAVES;
            const int r = k / (2 * G), j = (k / 2) % G, w = k % 2;
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const int  b  = lane + q * WAVE;
                const bool on = k < n && b < p.blocks;
                mb[u][q]      = on ? partial(p, partial_slot(b, r, j, 2 * w)) : -INFINITY;
                sm[u][q]      = on ? partial(p, partial_slot(b, r, j, 2 * w + 1)) : 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < ROUNDS; ++u) {
            const int k = wave + u * WAVES;
            if (k >= n) break;
            const int r = k / (2 * G), j = (k / 2) % G, w = k % 2;
            float     mx = -INFINITY;
#pragma unroll
            for (int q = 0; q < NQ; ++q) mx = fmaxf(mx, mb[u][q]);
            const float gmax = wave_reduce(mx, MaxOp{});
            float       part = 0.0f;
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                if (mb[u][q] > -INFINITY) part += sm[u][q] * expf(mb[u][q] - gmax);
            }
            part = wave_reduce(part, SumOp{});
            if (lane < 2) push(p, stat_slot(parity, r, j, 2 * w + lane, p.rank), value_record(lane ? part : gmax, vtag));
        }
    }

    // every rank's statistics and the drafted tokens' logits of every draft row, in one poll
    const int per_req = G * STATS * p.tp, nstat = p.bs * per_req;
    poll(
        p, nstat + p.bs * G * 2,
        [&](int i) {
            if (i >= nstat) {
                i -= nstat;
                return tok_slot(parity, i / (2 * G), (i / 2) % G, i % 2);
            }
            const int r = i / per_req, q = i % per_req;
            return stat_slot(parity, r, q / (STATS * p.tp), (q / p.tp) % STATS, q % p.tp);
        },
        [&](uint64_t v) { return static_cast<uint32_t>(v) == vtag; },
        [&](int i, uint64_t v) {
            if (i >= nstat) {
                i -= nstat;
                s_tok[i / (2 * G)][(i / 2) % G][i % 2] = record_value(v);
                return;
            }
            const int r = i / per_req, q = i % per_req;
            s_stat[r][q / (STATS * p.tp)][(q / p.tp) % STATS][q % p.tp] = record_value(v);
        });
    lds_sync();
    // each row's log-sum-exp (_global_lse over the ranks), MAX_TP lanes a row, a lane a rank: every row in one round
    // at bs 1 (the sums in wave_reduce's order: lanes 4 apart, then 2, then 1)
    static_assert(MAX_TP == 8 && WAVE % MAX_TP == 0);
    for (int k = threadIdx.x / MAX_TP; k < p.bs * G * 2; k += THREADS / MAX_TP) {
        const int   r = k / (2 * G), j = (k / 2) % G, w = k % 2, src = lane % MAX_TP;
        const float mx = src < p.tp ? s_stat[r][j][2 * w][src] : -INFINITY;
        const float sm = src < p.tp ? s_stat[r][j][2 * w + 1][src] : 0.0f;
        float       gmax = mx;
#pragma unroll
        for (int d = 4; d >= 1; d /= 2) gmax = fmaxf(gmax, __shfl_xor(gmax, d, MAX_TP));
        float part = src < p.tp && mx > -INFINITY ? sm * expf(mx - gmax) : 0.0f;
#pragma unroll
        for (int d = 4; d >= 1; d /= 2) part += __shfl_xor(part, d, MAX_TP);
        if (src == 0) s_lse[r][j][w] = gmax + logf(part);
    }
    lds_sync();
    // the chain accept, a request a thread (every block alike)
    if (static_cast<int>(threadIdx.x) < p.bs) {
        const int   r        = threadIdx.x;
        const float t        = s_temp[r];
        int         accepted = 0;
        bool        on       = true;
#pragma unroll
        for (int j = 0; j < G; ++j) {
            const int64_t tok   = s_cand[r][1 + j];
            const bool    valid = tok >= 0 && tok < p.vocab;
            const float   log_p = valid ? s_tok[r][j][0] / t - s_lse[r][j][0] : -INFINITY;
            const float   log_q = valid ? s_tok[r][j][1] / t - s_lse[r][j][1] : -INFINITY;
            on = on && valid && log_q > -INFINITY && log_p > s_logu[r][j] + log_q;
            accepted += on ? 1 : 0;
        }
        s_acc[r] = accepted;
    }
    lds_sync();

    // the block's key of each request's resample row (the last request's rows from registers)
    for (int r = p.bs - 1; r >= 0; --r) {
        const float t = s_temp[r];
        const int   a = s_acc[r];
        if (r != p.bs - 1) {
            load_targets<T>(p.local + r * T * p.local_stride, p.local_stride, c, p.width, x);
            load_drafts<G>(p.draft + r * p.dseq, p.dstep, c, p.width, d);
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int i = 0; i < COLS; ++i) x[j][i] = x[j][i] / t;
            }
#pragma unroll
            for (int j = 0; j < G; ++j) {
#pragma unroll
                for (int i = 0; i < COLS; ++i) d[j][i] = d[j][i] / t;
            }
        }
        float xa[COLS], da[COLS];
        pick<T>(x, a, xa);
        pick<G>(d, a < G ? a : 0, da);
        const float lt = a < G ? s_lse[r][a][0] : 0.0f, ld = a < G ? s_lse[r][a][1] : 0.0f;
        uint64_t    best = 0;
#pragma unroll
        for (int i = 0; i < COLS; ++i) {
            const int g = p.vocab_start + c + i;
            if (c + i < p.width) {
                float residual = xa[i];
                if (a < G) { // log max(p - q, 0) = log p + log1p(-q / p), -inf where q >= p
                    const float log_p = residual - lt, log_q = da[i] - ld;
                    const float ratio = expf(log_q - log_p);
                    residual          = ratio < 1.0f ? log_p + log1pf(-ratio) : -INFINITY;
                }
                const float u = murmur3_uniform(seed, r, static_cast<uint32_t>(g));
                const float v = residual - logf(-log1p_neg_stable(u));
                if (v > -INFINITY) best = umax64(best, candidate_key(v, g));
            }
        }
        best = block_key_max(best, s_kred);
        if (threadIdx.x == 0) push(p, key_slot(parity, r, 0, p.rank, blockIdx.x), (best << TAG_BITS) | ktag);
    }
    if (threadIdx.x == 0) p.counts[blockIdx.x] = count + 1;
    if (blockIdx.x != 0) return;

    // block 0: every rank's blocks' keys, then the accept tail
    const int nkeys = p.sim ? 2 : 1, nsrc = p.tp * p.blocks;
    poll(
        p, p.bs * nkeys * nsrc,
        [&](int i) {
            const int q = i % nsrc;
            return key_slot(parity, i / (nkeys * nsrc), (i / nsrc) % nkeys, q / p.blocks, q % p.blocks);
        },
        [&](uint64_t v) { return (v & TAG_MASK) == ktag; },
        [&](int i, uint64_t v) {
            atomicMax(reinterpret_cast<unsigned long long *>(&s_key[i / (nkeys * nsrc)][(i / nsrc) % nkeys]),
                      static_cast<unsigned long long>(v >> TAG_BITS));
        });
    lds_sync();
    if (threadIdx.x == 0) p.counter[0] = s_counter + 1; // every block read the counter before its first record
    const int r = threadIdx.x;
    if (r >= p.bs) return;
    const int     correct = p.sim ? sim : s_acc[r];
    const int64_t bonus   = key_index(s_key[r][p.sim ? 1 : 0]);
    const int     commit  = correct + 1;
    p.correct_out[r]      = correct;
    p.bonus_out[r]        = bonus;
    p.cap_trim_out[r]     = 0;
    p.commit_out[r]       = commit;
    p.new_seq_out[r]      = s_seq[r] + commit;
    int64_t *tokens       = p.out_tokens + r * p.out_tokens_stride;
#pragma unroll
    for (int j = 0; j < T; ++j) {
        tokens[j] = j == correct ? bonus : (j < correct ? s_cand[r][1 + j] : 0); // rejected drafts zeroed, as SGLang
    }
    if (p.gated_out != nullptr) {
        p.gated_out[r] = min(commit, static_cast<int>(s_vlen[r])) * p.gate[0];
    }
    if (p.history != nullptr) { // the Engram history commit of the request's slot (SGLang's engram_commit_history)
        const int64_t slot = p.slots[r];
        if (p.relay_lens != nullptr) {
            p.relay_lens[slot]   = s_seq[r] + commit;
            p.relay_tokens[slot] = bonus;
        }
        if (s_vlen[r] != 0 && p.gate[0] != 0) {
            int32_t *row = p.history + slot * p.history_stride;
            int32_t  old[MAX_HISTORY];
            for (int k = 0; k < p.history_width; ++k) old[k] = row[k];
            for (int k = 0; k < p.history_width; ++k) {
                const int from = commit + k;
                row[k] = from < p.history_width ? old[from] : static_cast<int32_t>(s_cand[r][from - p.history_width]);
            }
        }
    }
}

template <int T> __global__ __launch_bounds__(THREADS) void sampled_tail_kernel(const Params p) {
    sampled_tail_body<T>(p);
}

// the accept's blocks first, then the draft's main_proj CTAs (main_proj.h), whose input rows are ready before the
// head: they run while the accept's blocks wait on their records
static_assert(THREADS == main_proj::THREADS);
template <int T> __global__ __launch_bounds__(THREADS) void sampled_tail_proj_kernel(const Params p,
                                                                                    const main_proj::Params q) {
    if (static_cast<int>(blockIdx.x) < p.blocks) {
        sampled_tail_body<T>(p);
    } else {
        main_proj::run(q, static_cast<int>(blockIdx.x) - p.blocks, static_cast<int>(gridDim.x) - p.blocks);
    }
}

void check(const at::Tensor &t, const at::Tensor &ref, at::ScalarType dtype, const char *name) {
    TORCH_CHECK(t.device() == ref.device() && t.scalar_type() == dtype, name, " must be ", dtype, " on ",
                ref.device());
    TORCH_CHECK(t.numel() == 0 || t.stride(-1) == 1 || t.numel() == 1, name, " must have unit inner stride");
}

template <typename T> T *ptr(const c10::optional<at::Tensor> &t) {
    return t.has_value() ? static_cast<T *>(t->data_ptr()) : nullptr;
}

int64_t record_bytes() { return RECORD_BYTES; }

int64_t scratch_bytes() { return SCRATCH_BYTES; }

int64_t blocks_for(int64_t width) { return (width + CHUNK - 1) / CHUNK; }

// local [bs * T, part] bf16 (4 B aligned rows, the first width columns valid); draft [>= bs, >= T - 1, >= width] f32
// (this rank's columns of the step logits, 8 B aligned rows); temps [bs] f32 (> 0); counter [1] int64; sim [1]
// int64 with sim_seed [1] int32, or none; candidates [bs, T] int64; the outputs as static_accept_tail's (rows
// [:bs]); scratch [SCRATCH_BYTES] u8 and counts [blocks] int32 (each zeroed once), peers [tp] int64 on the host:
// every rank's zeroed records
Params make_params(const at::Tensor &local, const at::Tensor &draft, const at::Tensor &temps, int64_t key,
                  at::Tensor &counter, const c10::optional<at::Tensor> &sim,
                  const c10::optional<at::Tensor> &sim_seed, const at::Tensor &candidates,
                  const at::Tensor &seq_lens, const at::Tensor &verify_lens, const at::Tensor &gate,
                  at::Tensor &correct_out, at::Tensor &bonus_out, at::Tensor &cap_trim_out, at::Tensor &commit_out,
                  at::Tensor &new_seq_out, at::Tensor &out_tokens, const c10::optional<at::Tensor> &gated_out,
                  at::Tensor &scratch, at::Tensor &counts, const at::Tensor &peers, int64_t rank,
                  int64_t vocab_start, int64_t width, int64_t vocab, double deadline_s,
                  const c10::optional<at::Tensor> &history, const c10::optional<at::Tensor> &slots,
                  const c10::optional<at::Tensor> &relay_lens, const c10::optional<at::Tensor> &relay_tokens) {
    const int64_t bs = candidates.size(0), T = candidates.size(1), tp = peers.numel();
    const int64_t blocks = blocks_for(width);
    check(local, local, at::kBFloat16, "local");
    TORCH_CHECK(local.dim() == 2 && local.size(0) == bs * T && local.size(1) >= width && local.stride(0) % 2 == 0 &&
                    reinterpret_cast<uintptr_t>(local.data_ptr()) % 4 == 0,
                "local [bs * T, part] bf16, 4 B aligned rows");
    check(draft, local, at::kFloat, "draft");
    TORCH_CHECK(draft.dim() == 3 && draft.size(0) >= bs && draft.size(1) >= T - 1 && draft.size(2) >= width &&
                    draft.stride(1) % 2 == 0 && draft.stride(0) % 2 == 0 &&
                    reinterpret_cast<uintptr_t>(draft.data_ptr()) % 8 == 0,
                "draft [bs, T - 1, width] f32, 8 B aligned rows");
    TORCH_CHECK(bs >= 1 && bs <= MAX_SEQ && T >= 2 && T <= MAX_T, "at most ", MAX_SEQ, " requests of ", MAX_T,
                " rows");
    TORCH_CHECK(tp >= 1 && tp <= MAX_TP && blocks <= MAX_NB && rank >= 0 && rank < tp, "TP ", tp, " with ", blocks,
                " blocks a rank: at most ", MAX_NB);
    TORCH_CHECK(vocab < (1 << IDX_BITS) && vocab_start + width <= vocab, "the vocab below 2^17");
    check(temps, local, at::kFloat, "temps");
    check(counter, local, at::kLong, "counter");
    check(candidates, local, at::kLong, "candidates");
    check(seq_lens, local, at::kLong, "seq_lens");
    check(verify_lens, local, at::kLong, "verify_lens");
    check(gate, local, at::kInt, "gate");
    check(correct_out, local, at::kLong, "correct_out");
    check(bonus_out, local, at::kLong, "bonus_out");
    check(cap_trim_out, local, at::kInt, "cap_trim_out");
    check(commit_out, local, at::kInt, "commit_out");
    check(new_seq_out, local, at::kLong, "new_seq_out");
    check(out_tokens, local, at::kLong, "out_tokens");
    TORCH_CHECK(out_tokens.dim() == 2 && out_tokens.size(0) >= bs && out_tokens.size(1) >= T, "out_tokens [bs, T]");
    TORCH_CHECK(sim.has_value() == sim_seed.has_value(), "a simulated length with its seed");
    if (sim.has_value()) {
        check(*sim, local, at::kLong, "sim");
        check(*sim_seed, local, at::kInt, "sim_seed");
    }
    if (gated_out.has_value()) check(*gated_out, local, at::kInt, "gated_out");
    check(scratch, local, at::kByte, "scratch");
    TORCH_CHECK(scratch.numel() >= SCRATCH_BYTES && reinterpret_cast<uintptr_t>(scratch.data_ptr()) % 8 == 0,
                "scratch [", SCRATCH_BYTES, "] u8");
    check(counts, local, at::kInt, "counts");
    TORCH_CHECK(counts.numel() >= blocks, "counts [blocks] int32");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong, "peers: every rank's records");
    TORCH_CHECK(history.has_value() == slots.has_value() && relay_lens.has_value() == relay_tokens.has_value() &&
                    (history.has_value() || !relay_lens.has_value()),
                "history and slots together, the relay rows with them");
    Params p{};
    p.local             = static_cast<const bfloat16 *>(local.data_ptr());
    p.local_stride      = local.stride(0);
    p.draft             = draft.data_ptr<float>();
    p.dseq              = draft.stride(0);
    p.dstep             = draft.stride(1);
    p.temps             = temps.data_ptr<float>();
    p.key               = key;
    p.counter           = counter.data_ptr<int64_t>();
    p.sim               = ptr<const int64_t>(sim);
    p.sim_seed          = ptr<const int32_t>(sim_seed);
    p.width             = static_cast<int>(width);
    p.vocab_start       = static_cast<int>(vocab_start);
    p.vocab             = static_cast<int>(vocab);
    p.blocks            = static_cast<int>(blocks);
    p.rank              = static_cast<int>(rank);
    p.tp                = static_cast<int>(tp);
    p.bs                = static_cast<int>(bs);
    p.T                 = static_cast<int>(T);
    p.candidates        = candidates.data_ptr<int64_t>();
    p.cand_stride       = candidates.stride(0);
    p.seq_lens          = seq_lens.data_ptr<int64_t>();
    p.verify_lens       = verify_lens.data_ptr<int64_t>();
    p.gate              = gate.data_ptr<int32_t>();
    p.correct_out       = correct_out.data_ptr<int64_t>();
    p.bonus_out         = bonus_out.data_ptr<int64_t>();
    p.cap_trim_out      = cap_trim_out.data_ptr<int32_t>();
    p.commit_out        = commit_out.data_ptr<int32_t>();
    p.new_seq_out       = new_seq_out.data_ptr<int64_t>();
    p.out_tokens        = out_tokens.data_ptr<int64_t>();
    p.out_tokens_stride = out_tokens.stride(0);
    p.gated_out         = ptr<int32_t>(gated_out);
    if (history.has_value()) {
        TORCH_CHECK(history->scalar_type() == at::kInt && history->dim() == 2 && history->stride(1) == 1 &&
                        history->size(1) <= MAX_HISTORY,
                    "history [slots, <= 8] int32");
        TORCH_CHECK(slots->scalar_type() == at::kLong && slots->numel() >= bs && slots->stride(0) == 1,
                    "slots [bs] int64");
        p.history        = history->data_ptr<int32_t>();
        p.history_stride = history->stride(0);
        p.history_width  = static_cast<int>(history->size(1));
        p.slots          = slots->data_ptr<int64_t>();
        if (relay_lens.has_value()) {
            TORCH_CHECK(relay_lens->scalar_type() == at::kLong && relay_tokens->scalar_type() == at::kLong &&
                            relay_lens->is_contiguous() && relay_tokens->is_contiguous(),
                        "relay rows int64");
            p.relay_lens   = relay_lens->data_ptr<int64_t>();
            p.relay_tokens = relay_tokens->data_ptr<int64_t>();
        }
    }
    p.partials       = reinterpret_cast<float *>(scratch.data_ptr());
    p.arrived        = reinterpret_cast<uint64_t *>(static_cast<char *>(scratch.data_ptr()) + PARTIALS * 4);
    p.counts         = reinterpret_cast<uint32_t *>(counts.data_ptr<int>());
    p.deadline_ticks = static_cast<uint64_t>(deadline_s * 1e8); // the 100 MHz realtime clock
    for (int r = 0; r < p.tp; ++r) {
        p.recv[r] = reinterpret_cast<void *>(peers.data_ptr<int64_t>()[r]);
    }
    return p;
}

#define SAMPLED_ARGS                                                                                               \
    const at::Tensor &local, const at::Tensor &draft, const at::Tensor &temps, int64_t key, at::Tensor &counter,  \
        const c10::optional<at::Tensor> &sim, const c10::optional<at::Tensor> &sim_seed,                          \
        const at::Tensor &candidates, const at::Tensor &seq_lens, const at::Tensor &verify_lens,                  \
        const at::Tensor &gate, at::Tensor &correct_out, at::Tensor &bonus_out, at::Tensor &cap_trim_out,         \
        at::Tensor &commit_out, at::Tensor &new_seq_out, at::Tensor &out_tokens,                                  \
        const c10::optional<at::Tensor> &gated_out, at::Tensor &scratch, at::Tensor &counts,                      \
        const at::Tensor &peers, int64_t rank, int64_t vocab_start, int64_t width, int64_t vocab, double deadline_s
#define SAMPLED_PASS                                                                                               \
    local, draft, temps, key, counter, sim, sim_seed, candidates, seq_lens, verify_lens, gate, correct_out,       \
        bonus_out, cap_trim_out, commit_out, new_seq_out, out_tokens, gated_out, scratch, counts, peers, rank,    \
        vocab_start, width, vocab, deadline_s
#define COMMIT_ARGS                                                                                                \
    const c10::optional<at::Tensor> &history, const c10::optional<at::Tensor> &slots,                             \
        const c10::optional<at::Tensor> &relay_lens, const c10::optional<at::Tensor> &relay_tokens

template <typename Launch> void launch_rows(int64_t T, Launch launch) {
    switch (T) {
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 3: launch(std::integral_constant<int, 3>{}); break;
    case 4: launch(std::integral_constant<int, 4>{}); break;
    case 5: launch(std::integral_constant<int, 5>{}); break;
    case 6: launch(std::integral_constant<int, 6>{}); break;
    case 7: launch(std::integral_constant<int, 7>{}); break;
    default: launch(std::integral_constant<int, 8>{}); break;
    }
}

void sampled_tail(SAMPLED_ARGS, COMMIT_ARGS) {
    const Params p = make_params(SAMPLED_PASS, history, slots, relay_lens, relay_tokens);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(local.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(local.device().index()).stream();
    launch_rows(candidates.size(1), [&](auto t) {
        sampled_tail_kernel<decltype(t)::value><<<p.blocks, THREADS, 0, stream>>>(p);
    });
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

// sampled_tail's accept and, in the same launch, main_proj of x [rows, 15360] bf16 on this rank's columns w, ws into
// proj_out (main_proj.h's arguments: proj_state, proj_work and proj_peers its exchange's)
void sampled_tail_proj(SAMPLED_ARGS, const at::Tensor &x, const at::Tensor &w, const at::Tensor &ws,
                       at::Tensor &proj_out, at::Tensor &proj_state, at::Tensor &proj_work,
                       const at::Tensor &proj_peers, double proj_deadline_s, COMMIT_ARGS) {
    const Params p = make_params(SAMPLED_PASS, history, slots, relay_lens, relay_tokens);
    TORCH_CHECK(x.device() == local.device() && proj_peers.numel() == peers.numel(), "main_proj on the same group");
    const main_proj::Params q =
        main_proj::params(x, w, ws, proj_out, proj_state, proj_work, proj_peers, rank, proj_deadline_s);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(local.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(local.device().index()).stream();
    launch_rows(candidates.size(1), [&](auto t) {
        sampled_tail_proj_kernel<decltype(t)::value><<<p.blocks + main_proj::grid(q), THREADS, 0, stream>>>(p, q);
    });
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("sampled_tail(Tensor local, Tensor draft, Tensor temps, int key, Tensor(a!) counter, Tensor? sim, "
          "Tensor? sim_seed, Tensor candidates, Tensor seq_lens, Tensor verify_lens, Tensor gate, "
          "Tensor(b!) correct_out, Tensor(c!) bonus_out, Tensor(d!) cap_trim_out, Tensor(e!) commit_out, "
          "Tensor(f!) new_seq_out, Tensor(g!) out_tokens, Tensor(h!)? gated_out, Tensor(i!) scratch, "
          "Tensor(j!) counts, Tensor peers, int rank, int vocab_start, int width, int vocab, float deadline_s, "
          "Tensor(x!)? history=None, Tensor? slots=None, Tensor(y!)? relay_lens=None, "
          "Tensor(z!)? relay_tokens=None) -> ()");
    m.def("sampled_tail_proj(Tensor local, Tensor draft, Tensor temps, int key, Tensor(a!) counter, Tensor? sim, "
          "Tensor? sim_seed, Tensor candidates, Tensor seq_lens, Tensor verify_lens, Tensor gate, "
          "Tensor(b!) correct_out, Tensor(c!) bonus_out, Tensor(d!) cap_trim_out, Tensor(e!) commit_out, "
          "Tensor(f!) new_seq_out, Tensor(g!) out_tokens, Tensor(h!)? gated_out, Tensor(i!) scratch, "
          "Tensor(j!) counts, Tensor peers, int rank, int vocab_start, int width, int vocab, float deadline_s, "
          "Tensor x, Tensor w, Tensor ws, Tensor(k!) proj_out, Tensor(l!) proj_state, Tensor(m!) proj_work, "
          "Tensor proj_peers, float proj_deadline_s, Tensor(x!)? history=None, Tensor? slots=None, "
          "Tensor(y!)? relay_lens=None, Tensor(z!)? relay_tokens=None) -> ()");
    m.impl("sampled_tail_proj", at::kCUDA, &sampled_tail_proj);
    m.def("record_bytes() -> int");
    m.def("scratch_bytes() -> int");
    m.def("blocks_for(int width) -> int");
    m.impl("sampled_tail", at::kCUDA, &sampled_tail);
    m.impl("record_bytes", c10::DispatchKey::CompositeExplicitAutograd, &record_bytes);
    m.impl("scratch_bytes", c10::DispatchKey::CompositeExplicitAutograd, &scratch_bytes);
    m.impl("blocks_for", c10::DispatchKey::CompositeExplicitAutograd, &blocks_for);
}
