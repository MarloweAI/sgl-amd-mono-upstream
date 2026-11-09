// SPDX-License-Identifier: Apache-2.0
// Adapted from AKLR (https://github.com/AMD-AGI/AKLR, MIT, Copyright (C) 2025-2026 Advanced Micro Devices, Inc.).
// A sampled DSpark draft's Markov block on this rank's vocab shard as SGLang's MarkovSampleStep, every step in one
// launch: each step's bias GEMV, Philox Gumbel-max scores and the shard's bests, the ranks' bests swapped through
// peer buffers, and after the last step the sampled sequences' step logits of every rank's shard gathered whole.
#include <climits>
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

constexpr int DIM           = 256;
constexpr int THREADS       = 512;
constexpr int ROW_LANES     = 16; // a vocab row's DIM by 16 lanes, 16 values each
constexpr int ROWS_PER_PASS = THREADS / ROW_LANES;
constexpr int PASSES        = 8;
constexpr int BLOCK_ROWS    = ROWS_PER_PASS * PASSES; // 256: a block's rows, a row a thread when scored
constexpr int WAVE          = 64;
constexpr int MAX_SEQ       = 4;
constexpr int MAX_TP        = 8;
constexpr int MAX_STEPS     = 8;
constexpr int MAX_BLOCKS    = 256; // a block a CU at most: a TP2 shard's 64640 rows are 253
constexpr int POLL_ONE      = 8;   // a step's records a lane of wave 0, one sequence (every rank's blocks: at most 512)
// the exchange buffer: every block's step record, 8 B each [launch parity 2][step][sequence][source rank][block]
// (markov.py's EXCHANGE_BYTES), then the rank's step logits [MAX_SEQ, gamma, part] f32
constexpr int EXCHANGE_BYTES = 2 * MAX_STEPS * MAX_SEQ * MAX_TP * MAX_BLOCKS * 8;
// a record: the candidate key (ordered score bits, then the inverted global index) above a 15-bit launch tag
constexpr int      TAG_BITS = 15;
constexpr uint64_t TAG_MASK = (1ull << TAG_BITS) - 1;
constexpr int      IDX_BITS = 17; // the vocab (129280) below 2^17
constexpr uint32_t IDX_MASK = (1u << IDX_BITS) - 1;

struct Params {
    const bfloat16 *logits; // base logits, sequence s's step t row at logits + s * logits_stride + t * step_stride
    int64_t         logits_stride, step_stride;
    const bfloat16 *embed; // markov_w1 [vocab, DIM]
    const bfloat16 *head;  // this rank's markov_w2 rows [width, DIM]
    const int64_t  *anchor; // sequence s's at anchor[s * anchor_stride]
    int64_t         anchor_stride;
    const float    *temps; // [nseq]
    int64_t        *offset;
    uint32_t        key_lo, key_hi;
    int             vocab_start, width, padded, gamma, nseq, blocks, rank, tp, part, vocab;
    int64_t        *out; // sequence s's token of step t at out[s * out_stride + t]
    int64_t         out_stride;
    int64_t        *tok;      // [nseq] the last step's tokens
    uint32_t       *counts; // [blocks] each block's launches
    float          *whole; // sequence s's step t row at whole + s * whole_seq_stride + t * whole_step_stride, [vocab]
    int64_t         whole_seq_stride, whole_step_stride;
    void           *recv[MAX_TP]; // every rank's exchange buffer
    uint64_t        deadline_ticks;
};

__device__ inline void keep_best(float &v, int &i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) {
        v = v2, i = i2;
    }
}

__device__ inline float round_bf16(float x) { return static_cast<float>(static_cast<bfloat16>(x)); }

__device__ inline void load16f(const bfloat16 *src, float (&v)[16]) {
    bfloat16 b[16];
    *reinterpret_cast<uint4 *>(b)     = *reinterpret_cast<const uint4 *>(src);
    *reinterpret_cast<uint4 *>(b + 8) = *reinterpret_cast<const uint4 *>(src + 8);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] = static_cast<float>(b[i]);
    }
}

// Philox4x32-10 (tl.philox's): the word k of the four
__device__ inline uint32_t philox_word(uint32_t k0, uint32_t k1, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                                       int k) {
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
    return k == 0 ? c0 : k == 1 ? c1 : k == 2 ? c2 : c3;
}

// the Gumbel noise of a sampled row (it does not depend on the tokens: computed ahead of its step)
__device__ inline float gumbel(const Params &p, int seq, int step, int64_t offset, int v) {
    const uint32_t g = static_cast<uint32_t>(p.vocab_start + v);
    const uint32_t w = philox_word(p.key_lo, p.key_hi, g >> 2, static_cast<uint32_t>(seq), static_cast<uint32_t>(step),
                                   static_cast<uint32_t>(offset), static_cast<int>(g & 3u));
    const float u = (static_cast<float>(w) + 0.5f) * 2.3283064365386963e-10f;
    return logf(-log1pf(-u));
}

// the row's selection key: its value at temperature 0, else value / t minus its noise
__device__ inline float select_key(float t, float noise, float value) {
    const float key = t != 0.0f ? value / t - noise : value;
    return key != key ? INFINITY : key;
}

__device__ inline void store_nt(uint32x4 *ptr, uint32x4 val) {
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1 nt\n\ts_nop 1" ::"v"(ptr), "v"(val) : "memory");
}

__device__ inline uint32x4 load_nt(const uint32x4 *ptr) {
    uint32x4 val;
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1 nt\n\ts_waitcnt vmcnt(0)" : "=v"(val) : "v"(ptr) : "memory");
    return val;
}

// a peer's packet of this call; past the deadline the launch traps, stopping this rank's step
__device__ inline uint32x4 poll_tag(const uint32x4 *src, uint32_t tag, uint64_t deadline) {
    uint64_t t0 = 0;
    for (int n = 1;; ++n) {
        const uint32x4 pk = load_nt(src);
        if (pk.w == tag) {
            return pk;
        }
        __builtin_amdgcn_s_sleep(2);
        if (n % 1024 == 0) {
            const uint64_t now = __builtin_amdgcn_s_memrealtime();
            if (t0 == 0) {
                t0 = now;
            } else if (now - t0 > deadline) {
                __builtin_trap();
            }
        }
    }
}

// a step's candidate as one ordered word: the score's bits in order above the inverted global index, so the
// larger key is the better score, then the lower index (the ranks' shards ascend: the lower rank); 0 is none
__device__ inline uint64_t candidate_key(float score, int index) {
    score               = score == 0.0f ? 0.0f : score; // -0 ties +0, as the float compare does
    const uint32_t bits = __builtin_bit_cast(uint32_t, score);
    const uint32_t ord  = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    return (static_cast<uint64_t>(ord) << IDX_BITS) | static_cast<uint64_t>(IDX_MASK - static_cast<uint32_t>(index));
}

__device__ inline int64_t key_index(uint64_t key) {
    return static_cast<int64_t>(IDX_MASK - static_cast<uint32_t>(key & IDX_MASK));
}

__device__ inline int64_t record_slot(int parity, int step, int seq, int src, int block) {
    return ((((static_cast<int64_t>(parity) * MAX_STEPS + step) * MAX_SEQ + seq) * MAX_TP + src) * MAX_BLOCKS) + block;
}

#ifdef MARKOV_STAMPS // a timing build (tests only): thread 0's realtime ticks per block, past the launch counts
constexpr int STAMPS = 2 + 5 * MAX_STEPS;
#define MARKOV_STAMP(k)                                                                                               \
    if (threadIdx.x == 0)                                                                                              \
    reinterpret_cast<uint64_t *>(p.counts + MAX_BLOCKS)[blockIdx.x * STAMPS + (k)] = __builtin_amdgcn_s_memrealtime()
#else
#define MARKOV_STAMP(k)
#endif

// DPP moves within a 16-lane row (no LDS): row_ror 8 is lane ^ 8; row_ror 4 is lane ^ 4 once lanes l and l ^ 8 hold
// equal values; the quad permutations are lane ^ 2 and lane ^ 1
template <int CTRL> __device__ inline uint32_t dpp(uint32_t v) {
    return static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(v), CTRL, 0xF, 0xF, false));
}
constexpr int ROR8 = 0x128, ROR4 = 0x124, XOR2 = 0x4E, XOR1 = 0xB1;

// lane_sum<16>'s butterfly (xor 8, 4, 2, 1), the same pairs added in the same order: bitwise equal
__device__ inline float row_sum16(float v) {
    v += __builtin_bit_cast(float, dpp<ROR8>(__builtin_bit_cast(uint32_t, v)));
    v += __builtin_bit_cast(float, dpp<ROR4>(__builtin_bit_cast(uint32_t, v)));
    v += __builtin_bit_cast(float, dpp<XOR2>(__builtin_bit_cast(uint32_t, v)));
    v += __builtin_bit_cast(float, dpp<XOR1>(__builtin_bit_cast(uint32_t, v)));
    return v;
}

// the wave's best (score, index) by keep_best's order: within each row by DPP, then across the four rows
__device__ inline void wave_best(float &v, int &i) {
    keep_best(v, i, __builtin_bit_cast(float, dpp<ROR8>(__builtin_bit_cast(uint32_t, v))),
              static_cast<int>(dpp<ROR8>(static_cast<uint32_t>(i))));
    keep_best(v, i, __builtin_bit_cast(float, dpp<ROR4>(__builtin_bit_cast(uint32_t, v))),
              static_cast<int>(dpp<ROR4>(static_cast<uint32_t>(i))));
    keep_best(v, i, __builtin_bit_cast(float, dpp<XOR2>(__builtin_bit_cast(uint32_t, v))),
              static_cast<int>(dpp<XOR2>(static_cast<uint32_t>(i))));
    keep_best(v, i, __builtin_bit_cast(float, dpp<XOR1>(__builtin_bit_cast(uint32_t, v))),
              static_cast<int>(dpp<XOR1>(static_cast<uint32_t>(i))));
    keep_best(v, i, __shfl_xor(v, 16, WAVE), __shfl_xor(i, 16, WAVE));
    keep_best(v, i, __shfl_xor(v, 32, WAVE), __shfl_xor(i, 32, WAVE));
}

// the wave's largest key: within each row by DPP, then across the four rows
__device__ inline uint64_t wave_max(uint64_t x) {
    const auto step = [&](uint64_t y) { return y > x ? y : x; };
#define MK_DPP64(C) (static_cast<uint64_t>(dpp<C>(static_cast<uint32_t>(x >> 32))) << 32 | dpp<C>(static_cast<uint32_t>(x)))
    x = step(MK_DPP64(ROR8));
    x = step(MK_DPP64(ROR4));
    x = step(MK_DPP64(XOR2));
    x = step(MK_DPP64(XOR1));
#undef MK_DPP64
    x = step(__shfl_xor(x, 16, WAVE));
    x = step(__shfl_xor(x, 32, WAVE));
    return x;
}

__device__ inline float bf16_lo(uint32_t w) { return __builtin_bit_cast(float, w << 16); }
__device__ inline float bf16_hi(uint32_t w) { return __builtin_bit_cast(float, w & 0xFFFF0000u); }

// the whole block in one launch, the grid co-resident (a block a CU at most, two waves an EU):
// a block keeps BLOCK_ROWS rows of the shard in registers (bf16) for every step; each step every block stores its
// best key into its slot of every rank's records and reads every rank's blocks' records from its own buffer: the
// grid barrier and the ranks' swap at once, one cross-GPU trip a step, no atomics. ONE: one sequence, its own build
// (the batched sequences' registers are not held through its steps): each step's dots and scores without a barrier
// between them, the token's embedding row read by wave 0 and handed to the block through LDS
template <bool ONE>
__global__ __launch_bounds__(THREADS) __attribute__((amdgpu_waves_per_eu(2))) void
markov_block_kernel(const Params p) {
    const int nseq = ONE ? 1 : p.nseq;
    __shared__ float    sv[MAX_SEQ][BLOCK_ROWS]; // the bias GEMV of each sequence
    __shared__ float    sbase[MAX_STEPS][MAX_SEQ][BLOCK_ROWS]; // the block's rows' base logits of every step
    __shared__ float    snoise[MAX_SEQ][BLOCK_ROWS]; // the sampled sequences' noise of this step
    __shared__ float    wave_bestv[MAX_SEQ][BLOCK_ROWS / WAVE];
    __shared__ int      wave_bi[MAX_SEQ][BLOCK_ROWS / WAVE];
    __shared__ uint64_t wave_key[THREADS / WAVE]; // one sequence: each wave's best candidate key
    __shared__ __attribute__((aligned(16))) bfloat16 erow[DIM]; // one sequence: the step's token's embedding row
    __shared__ uint64_t top_key[MAX_SEQ][THREADS / WAVE];
    __shared__ int64_t  prev[MAX_SEQ];
    __shared__ float    stemp[MAX_SEQ]; // the sequences' temperatures, read once
    __shared__ uint32_t s_count;
    const int           lane = threadIdx.x % ROW_LANES, grp = threadIdx.x / ROW_LANES;
    const int           row  = blockIdx.x * BLOCK_ROWS + threadIdx.x; // scoring: a row a thread of the first 256
    const bool          scorer = threadIdx.x < BLOCK_ROWS;
    float               e[16];
    uint32_t            h[PASSES][8]; // 16 bf16 a pass, exact as f32
    MARKOV_STAMP(0);
#pragma unroll
    for (int ps = 0; ps < PASSES; ++ps) {
        const int v = blockIdx.x * BLOCK_ROWS + ps * ROWS_PER_PASS + grp;
        if (v < p.width) {
            const uint4 *src = reinterpret_cast<const uint4 *>(p.head + static_cast<int64_t>(v) * DIM + lane * 16);
            const uint4  lo = src[0], hi = src[1];
            h[ps][0] = lo.x, h[ps][1] = lo.y, h[ps][2] = lo.z, h[ps][3] = lo.w;
            h[ps][4] = hi.x, h[ps][5] = hi.y, h[ps][6] = hi.z, h[ps][7] = hi.w;
        }
    }
    // every step's base logits at the start: no step waits for a load of them
    if (scorer && row < p.padded) {
        for (int step = 0; step < p.gamma; ++step) {
            for (int seq = 0; seq < nseq; ++seq) {
                sbase[step][seq][threadIdx.x] =
                    static_cast<float>(p.logits[seq * p.logits_stride + step * p.step_stride + row]);
            }
        }
    }
    if (threadIdx.x == 0) {
        s_count = p.counts[blockIdx.x];
    }
    if (threadIdx.x < nseq) {
        prev[threadIdx.x]  = p.anchor[threadIdx.x * p.anchor_stride];
        stemp[threadIdx.x] = p.temps[threadIdx.x];
    }
    if (ONE && threadIdx.x < WAVE) { // a lane 4 values of the anchor's embedding row
        reinterpret_cast<uint2 *>(erow)[threadIdx.x] =
            *reinterpret_cast<const uint2 *>(p.embed + p.anchor[0] * DIM + threadIdx.x * 4);
    }
    const int64_t offset = p.offset[0];
    // a step's noise: every row of every sampled sequence, two a thread at most (temps: the step's reads of the
    // temperatures, p.temps before the first barrier, stemp after it)
    const auto noise_of = [&](int step, const float *temps) {
        for (int i = threadIdx.x; i < nseq * BLOCK_ROWS; i += THREADS) {
            const int seq = i / BLOCK_ROWS, r = i % BLOCK_ROWS, v = blockIdx.x * BLOCK_ROWS + r;
            if (temps[seq] != 0.0f && v < p.width) {
                snoise[seq][r] = gumbel(p, seq, step, offset, v);
            }
        }
    };
    noise_of(0, p.temps);
    __syncthreads();
    const uint32_t count  = s_count;
    const uint64_t tag    = 1 + count % static_cast<uint32_t>(TAG_MASK); // never 0; the launch two back differs
    const int      parity = static_cast<int>(count & 1u);
    float         *slab   = reinterpret_cast<float *>(static_cast<char *>(p.recv[p.rank]) + EXCHANGE_BYTES);
    uint64_t      *mine   = static_cast<uint64_t *>(p.recv[p.rank]);
    const int      polls  = nseq * p.tp * p.blocks; // records a step, at most 4 a thread
    MARKOV_STAMP(1);
    for (int step = 0; step < p.gamma; ++step) {
        if constexpr (ONE) {
            // the dots, then the scores without a barrier: a pass's row dot lands in all 16 lanes of its group,
            // lane ps of the group keeps pass ps's, and lanes 0-7 then score their rows (row lane * 32 + group)
            load16f(erow + lane * 16, e);
            float own = 0.0f;
#pragma unroll
            for (int ps = 0; ps < PASSES; ++ps) {
                const int v   = blockIdx.x * BLOCK_ROWS + ps * ROWS_PER_PASS + grp;
                float     dot = 0.0f;
                if (v < p.width) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        dot += e[2 * i] * bf16_lo(h[ps][i]);
                        dot += e[2 * i + 1] * bf16_hi(h[ps][i]);
                    }
                }
                dot  = row_sum16(dot);
                own = lane == ps ? dot : own;
            }
            float best = -INFINITY;
            int   bi   = INT_MAX;
            if (lane < PASSES) {
                const int r = lane * ROWS_PER_PASS + grp, v = blockIdx.x * BLOCK_ROWS + r;
                if (v < p.padded) {
                    const float value = sbase[step][0][r] + (v < p.width ? round_bf16(own) : 0.0f);
                    if (stemp[0] != 0.0f) { // a sampled sequence's step logits, for the gather
                        slab[static_cast<int64_t>(step) * p.part + v] = value;
                    }
                    if (v < p.width) {
                        best = select_key(stemp[0], snoise[0][r], value);
                        bi   = v;
                    }
                }
            }
            wave_best(best, bi);
            if (threadIdx.x % WAVE == 0) {
                wave_key[threadIdx.x / WAVE] = bi == INT_MAX ? 0ull : candidate_key(best, bi + p.vocab_start);
            }
            MARKOV_STAMP(5 + 5 * step);
        } else {
            // every sequence's bias rows, then one barrier
            for (int seq = 0; seq < nseq; ++seq) {
                load16f(p.embed + prev[seq] * DIM + lane * 16, e);
#pragma unroll
                for (int ps = 0; ps < PASSES; ++ps) {
                    const int v   = blockIdx.x * BLOCK_ROWS + ps * ROWS_PER_PASS + grp;
                    float     dot = 0.0f;
                    if (v < p.width) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            dot += e[2 * i] * bf16_lo(h[ps][i]);
                            dot += e[2 * i + 1] * bf16_hi(h[ps][i]);
                        }
                    }
                    dot = row_sum16(dot);
                    if (lane == 0) {
                        sv[seq][ps * ROWS_PER_PASS + grp] = dot;
                    }
                }
            }
            MARKOV_STAMP(5 + 5 * step);
            __syncthreads();
            // the scores: two sequences at a time, the first four waves one and the last four the next
            for (int pair = 0; pair < nseq; pair += THREADS / BLOCK_ROWS) {
                const int seq = pair + threadIdx.x / BLOCK_ROWS, r = threadIdx.x % BLOCK_ROWS;
                const int v   = blockIdx.x * BLOCK_ROWS + r;
                if (seq < nseq) {
                    float best = -INFINITY;
                    int   bi   = INT_MAX;
                    if (v < p.padded) {
                        const float bias  = v < p.width ? round_bf16(sv[seq][r]) : 0.0f;
                        const float value = sbase[step][seq][r] + bias;
                        if (stemp[seq] != 0.0f) { // a sampled sequence's step logits, for the gather
                            slab[(static_cast<int64_t>(seq) * p.gamma + step) * p.part + v] = value;
                        }
                        if (v < p.width) {
                            best = select_key(stemp[seq], snoise[seq][r], value);
                            bi   = v;
                        }
                    }
                    wave_best(best, bi);
                    if (r % WAVE == 0) {
                        wave_bestv[seq][r / WAVE] = best, wave_bi[seq][r / WAVE] = bi;
                    }
                }
            }
        }
        if (step + 1 == p.gamma) { // every step's logits stored before the last step's records: the gather reads them
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
        __syncthreads();
        MARKOV_STAMP(2 + 5 * step);
        if constexpr (ONE) {
            // one sequence: the exchange on wave 0 alone; waves 1-4 the next step's noise meanwhile
            if (step + 1 < p.gamma && threadIdx.x >= WAVE && stemp[0] != 0.0f) {
                const int r = threadIdx.x - WAVE, v = blockIdx.x * BLOCK_ROWS + r;
                if (r < BLOCK_ROWS && v < p.width) {
                    snoise[0][r] = gumbel(p, 0, step + 1, offset, v);
                }
            }
            if (threadIdx.x < WAVE) {
                const int l = threadIdx.x;
                // the block's key: the waves' keys, lane l holding wave l % 8's, maximum over each 16-lane row
                uint64_t key = wave_key[l % (THREADS / WAVE)];
                const auto take = [&](uint64_t y) { key = y > key ? y : key; };
#define MK_DPP64(C) (static_cast<uint64_t>(dpp<C>(static_cast<uint32_t>(key >> 32))) << 32 | dpp<C>(static_cast<uint32_t>(key)))
                take(MK_DPP64(ROR4));
                take(MK_DPP64(XOR2));
                take(MK_DPP64(XOR1));
#undef MK_DPP64
                // the block's record into every rank's slot, then every rank's blocks' records (one trip), up to
                // POLL_ONE a lane in flight together until each carries the tag
                if (l < p.tp) {
                    __hip_atomic_store(static_cast<uint64_t *>(p.recv[l]) + record_slot(parity, step, 0, p.rank, blockIdx.x),
                                       (key << TAG_BITS) | tag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                }
                MARKOV_STAMP(3 + 5 * step);
                uint64_t top = 0, t0 = 0;
                uint32_t pending = 0;
#pragma unroll
                for (int k = 0; k < POLL_ONE; ++k) {
                    pending |= (l + k * WAVE < p.tp * p.blocks ? 1u : 0u) << k;
                }
                for (int n = 1; pending; ++n) {
                    uint64_t v[POLL_ONE];
#pragma unroll
                    for (int k = 0; k < POLL_ONE; ++k) {
                        const int i = l + k * WAVE;
                        if ((pending >> k) & 1u) {
                            v[k] = __hip_atomic_load(mine + record_slot(parity, step, 0, i / p.blocks, i % p.blocks),
                                                     __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                        }
                    }
#pragma unroll
                    for (int k = 0; k < POLL_ONE; ++k) {
                        if (((pending >> k) & 1u) && (v[k] & TAG_MASK) == tag) {
                            top = v[k] >> TAG_BITS > top ? v[k] >> TAG_BITS : top;
                            pending &= ~(1u << k);
                        }
                    }
                    if (pending && n % 1024 == 0) { // past the deadline the launch traps, stopping this rank's step
                        const uint64_t now = __builtin_amdgcn_s_memrealtime();
                        if (t0 == 0) {
                            t0 = now;
                        } else if (now - t0 > p.deadline_ticks) {
                            __builtin_trap();
                        }
                    }
                }
                top = wave_max(top);
                MARKOV_STAMP(6 + 5 * step);
                const int64_t tok = key_index(top);
                if (step + 1 < p.gamma) {
                    reinterpret_cast<uint2 *>(erow)[l] = *reinterpret_cast<const uint2 *>(p.embed + tok * DIM + l * 4);
                }
                if (l == 0) {
                    prev[0]           = tok;
                    if (blockIdx.x == 0) {
                        p.out[step] = tok;
                    }
                }
            }
            __syncthreads();
        } else {
            // this block's record of each sequence into every rank's slot
            if (threadIdx.x < nseq * p.tp) {
                const int seq = threadIdx.x / p.tp, r = threadIdx.x % p.tp;
                float     best = wave_bestv[seq][0];
                int       bi   = wave_bi[seq][0];
    #pragma unroll
                for (int w = 1; w < BLOCK_ROWS / WAVE; ++w) {
                    keep_best(best, bi, wave_bestv[seq][w], wave_bi[seq][w]);
                }
                const uint64_t key = bi == INT_MAX ? 0ull : candidate_key(best, bi + p.vocab_start);
                __hip_atomic_store(static_cast<uint64_t *>(p.recv[r]) + record_slot(parity, step, seq, p.rank, blockIdx.x),
                                   (key << TAG_BITS) | tag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            }
            if (step + 1 < p.gamma) { // the next step's noise while the records cross (snoise's reads are behind a barrier)
                noise_of(step + 1, stemp);
            }
            MARKOV_STAMP(3 + 5 * step);
            // every rank's blocks' records of the step: each thread's (at most 4) in flight together until all carry the tag
            uint64_t got[4] = {0, 0, 0, 0};
            uint32_t pending = 0;
    #pragma unroll
            for (int k = 0; k < 4; ++k) {
                pending |= (threadIdx.x + k * THREADS < polls ? 1u : 0u) << k;
            }
            const auto slot_of = [&](int i) {
                const int b = i % p.blocks, src = (i / p.blocks) % p.tp, seq = i / (p.blocks * p.tp);
                return mine + record_slot(parity, step, seq, src, b);
            };
            uint64_t t0 = 0;
            for (int n = 1; pending; ++n) {
                uint64_t v[4];
    #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if ((pending >> k) & 1u) {
                        v[k] = __hip_atomic_load(slot_of(threadIdx.x + k * THREADS), __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_SYSTEM);
                    }
                }
    #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (((pending >> k) & 1u) && (v[k] & TAG_MASK) == tag) {
                        got[k] = v[k] >> TAG_BITS;
                        pending &= ~(1u << k);
                    }
                }
                if (pending && n % 1024 == 0) { // past the deadline the launch traps, stopping this rank's step
                    const uint64_t now = __builtin_amdgcn_s_memrealtime();
                    if (t0 == 0) {
                        t0 = now;
                    } else if (now - t0 > p.deadline_ticks) {
                        __builtin_trap();
                    }
                }
            }
            MARKOV_STAMP(6 + 5 * step);
            // the best record of each sequence: a wave's maximum, then the waves'
            for (int s = 0; s < nseq; ++s) {
                uint64_t x = 0;
    #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int i = threadIdx.x + k * THREADS;
                    if (i < polls && i / (p.blocks * p.tp) == s) {
                        x = got[k] > x ? got[k] : x;
                    }
                }
                x = wave_max(x);
                if (threadIdx.x % WAVE == 0) {
                    top_key[s][threadIdx.x / WAVE] = x;
                }
            }
            __syncthreads();
            if (threadIdx.x < nseq) {
                uint64_t top = 0;
    #pragma unroll
                for (int w = 0; w < THREADS / WAVE; ++w) {
                    top = top_key[threadIdx.x][w] > top ? top_key[threadIdx.x][w] : top;
                }
                const int64_t tok = key_index(top);
                prev[threadIdx.x] = tok;
                if (blockIdx.x == 0) {
                    p.out[threadIdx.x * p.out_stride + step] = tok;
                }
            }
            __syncthreads();
        }
        MARKOV_STAMP(4 + 5 * step);
    }
    if (threadIdx.x == 0) {
        p.counts[blockIdx.x] = count + 1;
    }
    if (blockIdx.x == 0 && threadIdx.x < nseq) {
        p.tok[threadIdx.x] = prev[threadIdx.x];
        if (threadIdx.x == 0) { // every block read the counter before its first record
            p.offset[0] = offset + 1;
        }
    }
    // the last step's records came after every rank's step logits: gather the sampled sequences' rows
    for (int seq = 0; seq < nseq; ++seq) {
        if (p.temps[seq] == 0.0f) {
            continue;
        }
        for (int rs = 0; rs < p.gamma * p.tp; ++rs) {
            const int       step = rs / p.tp, src = rs % p.tp;
            const int       start = src * p.part;
            const int       quads = min(p.part, p.vocab - start) / 4;
            const uint32x4 *from  = reinterpret_cast<const uint32x4 *>(
                                       static_cast<const char *>(p.recv[src]) + EXCHANGE_BYTES) +
                                   (static_cast<int64_t>(seq) * p.gamma + step) * (p.part / 4);
            uint32x4 *to = reinterpret_cast<uint32x4 *>(p.whole + seq * p.whole_seq_stride +
                                                        step * p.whole_step_stride + start);
            for (int i = blockIdx.x * THREADS + threadIdx.x; i < quads; i += p.blocks * THREADS) {
                to[i] = load_nt(from + i);
            }
        }
    }
}

void check(const at::Tensor &t, const at::Tensor &ref, at::ScalarType dtype, const char *name) {
    TORCH_CHECK(t.device() == ref.device() && t.scalar_type() == dtype, name, " must be ", dtype, " on ",
                ref.device());
    TORCH_CHECK((t.stride(-1) == 1 || t.numel() == 1) && reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0, name,
                " must have unit inner stride");
}

// base [nseq, gamma, padded] bf16 (unit inner stride), embed [vocab, 256], head [width, 256] bf16, anchor / tok
// [nseq] int64, temps [nseq] f32, offset [1] int64 (advanced once), out int64 (sequence s's step t token at
// s * out_stride + t), control int32 [blocks] (zeroed once), peers [tp] int64 on the host: every
// rank's exchange buffer; whole [nseq, gamma, vocab] f32 (any outer strides): a sampled sequence's step logits,
// every rank's shard of part columns
void markov_block(const at::Tensor &base, const at::Tensor &embed, const at::Tensor &head, const at::Tensor &anchor,
                  const at::Tensor &temps, at::Tensor &offset, int64_t key, int64_t vocab_start, at::Tensor &out,
                  int64_t out_stride, at::Tensor &tok, at::Tensor &control, const at::Tensor &peers, int64_t rank,
                  at::Tensor &whole, int64_t part, double deadline_s) {
    const int64_t nseq  = base.size(0);
    const int64_t width = head.size(0);
    TORCH_CHECK(base.dim() == 3 && nseq >= 1 && nseq <= MAX_SEQ && base.size(1) >= 1 && base.size(1) <= MAX_STEPS,
                "base [1 .. ", MAX_SEQ, " sequences, 1 .. ", MAX_STEPS, " steps, padded]");
    TORCH_CHECK(head.dim() == 2 && head.size(1) == DIM && head.is_contiguous() && embed.dim() == 2 &&
                    embed.size(1) == DIM && embed.is_contiguous() && width <= base.size(2),
                "head [width, 256], embed [vocab, 256] contiguous");
    check(base, base, at::kBFloat16, "base");
    check(embed, base, at::kBFloat16, "embed");
    check(head, base, at::kBFloat16, "head");
    TORCH_CHECK(anchor.device() == base.device() && anchor.scalar_type() == at::kLong && anchor.dim() == 1 &&
                    anchor.size(0) >= nseq,
                "anchor [nseq] int64 (any stride)");
    check(temps, base, at::kFloat, "temps");
    check(offset, base, at::kLong, "offset");
    check(out, base, at::kLong, "out");
    check(tok, base, at::kLong, "tok");
    check(control, base, at::kInt, "control");
    check(whole, base, at::kFloat, "whole");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong && peers.numel() >= 1 &&
                    peers.numel() <= MAX_TP,
                "peers: every rank's exchange buffer (int64, host)");
    const int64_t gamma = base.size(1), vocab = whole.size(2);
    TORCH_CHECK(whole.dim() == 3 && whole.size(0) >= nseq && whole.size(1) == gamma, "whole [nseq, gamma, vocab]");
    TORCH_CHECK(part % 4 == 0 && vocab % 4 == 0 && whole.stride(0) % 4 == 0 && whole.stride(1) % 4 == 0 &&
                    reinterpret_cast<uintptr_t>(whole.data_ptr()) % 16 == 0 && base.size(2) <= part,
                "16 B aligned rows and shards, the padded shard within part");
    TORCH_CHECK(peers.numel() * part >= vocab && (peers.numel() - 1) * part < vocab, "the shards tile the vocab");
    Params p{};
    p.logits        = static_cast<const bfloat16 *>(base.data_ptr());
    p.logits_stride = base.stride(0);
    p.step_stride   = base.stride(1);
    p.embed         = static_cast<const bfloat16 *>(embed.data_ptr());
    p.head          = static_cast<const bfloat16 *>(head.data_ptr());
    p.anchor        = anchor.data_ptr<int64_t>();
    p.anchor_stride = anchor.stride(0);
    p.temps         = temps.data_ptr<float>();
    p.offset        = offset.data_ptr<int64_t>();
    p.key_lo        = static_cast<uint32_t>(static_cast<uint64_t>(key) & 0xFFFFFFFFull);
    p.key_hi        = static_cast<uint32_t>(static_cast<uint64_t>(key) >> 32);
    p.vocab_start   = static_cast<int>(vocab_start);
    p.width         = static_cast<int>(width);
    p.padded        = static_cast<int>(base.size(2));
    p.gamma         = static_cast<int>(gamma);
    p.nseq          = static_cast<int>(nseq);
    p.blocks        = static_cast<int>((base.size(2) + BLOCK_ROWS - 1) / BLOCK_ROWS);
    TORCH_CHECK(p.blocks <= MAX_BLOCKS && control.numel() >= p.blocks, "control [blocks] int32, at most ",
                MAX_BLOCKS, " blocks");
    TORCH_CHECK(nseq * peers.numel() * p.blocks <= 4 * THREADS, "a step's records at most 4 a thread");
    TORCH_CHECK(nseq > 1 || peers.numel() * p.blocks <= POLL_ONE * WAVE, "one sequence's records on wave 0");
    p.rank             = static_cast<int>(rank);
    p.tp               = static_cast<int>(peers.numel());
    p.part             = static_cast<int>(part);
    p.vocab            = static_cast<int>(vocab);
    p.out              = out.data_ptr<int64_t>();
    p.out_stride       = out_stride;
    p.tok              = tok.data_ptr<int64_t>();
    p.counts           = reinterpret_cast<uint32_t *>(control.data_ptr<int>());
    p.whole            = whole.data_ptr<float>();
    p.whole_seq_stride = whole.stride(0);
    p.whole_step_stride = whole.stride(1);
    p.deadline_ticks   = static_cast<uint64_t>(deadline_s * 1e8); // the 100 MHz realtime clock
    for (int r = 0; r < p.tp; ++r) {
        p.recv[r] = reinterpret_cast<void *>(peers.data_ptr<int64_t>()[r]);
    }
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(base.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(base.device().index()).stream();
    if (nseq == 1) {
        markov_block_kernel<true><<<p.blocks, THREADS, 0, stream>>>(p);
    } else {
        markov_block_kernel<false><<<p.blocks, THREADS, 0, stream>>>(p);
    }
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("markov_block(Tensor base, Tensor embed, Tensor head, Tensor anchor, Tensor temps, Tensor(a!) offset, "
          "int key, int vocab_start, Tensor(b!) out, int out_stride, Tensor(c!) tok, Tensor(d!) control, "
          "Tensor peers, int rank, Tensor(e!) whole, int part, float deadline_s) -> ()");
    m.impl("markov_block", at::kCUDA, &markov_block);
}
