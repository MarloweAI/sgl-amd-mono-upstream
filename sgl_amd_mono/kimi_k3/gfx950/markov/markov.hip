// SPDX-License-Identifier: Apache-2.0
// Adapted from sgl_amd_mono/deepseek_v41/gfx950/markov (itself from AKLR, https://github.com/AMD-AGI/AKLR, MIT,
// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.).
// Kimi-K3's sampled DSpark draft Markov walk (a vanilla Markov head of rank 256, bf16 step logits) on this rank's
// vocab shard, as SGLang's MarkovSampleStep computes it, every step in one launch: each step's bias GEMV, the bf16
// step logits, the Philox Gumbel-max keys and the shard's bests, the ranks' bests swapped through peer buffers. The
// tokens, step logits and noise counter are bitwise those of SGLang's walk on the shard. Each rank writes its shard
// of every sequence's step logits into a shard-wide buffer (a sharded accept's), or into its exchange buffer, from
// which gather_step_logits copies every rank's shard into the whole rows.
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
using bf16x8   = __bf16 __attribute__((ext_vector_type(8)));
using f32x4    = float __attribute__((ext_vector_type(4)));

constexpr int DIM           = 256;
constexpr int THREADS       = 512;
constexpr int WAVE          = 64;
constexpr int TILES         = 2;                             // a wave's 16-row MFMA tiles of the shard
constexpr int BLOCK_ROWS    = THREADS / WAVE * TILES * 16;   // 256: a block's rows, a row a thread when scored
constexpr int MAX_SEQ       = 8;
constexpr int MAX_TP        = 8;
constexpr int MAX_STEPS     = 8;
constexpr int MAX_BLOCKS    = 256; // a block a CU at most: a TP8 shard's 20480 rows are 80
constexpr int POLL_ONE      = 12;  // a sequence's records a lane of its wave (8 ranks x 80 blocks: 10)
// the exchange buffer: every block's step record, 8 B each [launch parity 2][step][sequence][source rank][block]
// (markov.py's EXCHANGE_BYTES), then the rank's step logits [MAX_SEQ, gamma, part] bf16
constexpr int EXCHANGE_BYTES = 2 * MAX_STEPS * MAX_SEQ * MAX_TP * MAX_BLOCKS * 8;
// a record: the candidate key (ordered score bits, then the inverted global index) above a 14-bit launch tag
constexpr int      TAG_BITS = 14;
constexpr uint64_t TAG_MASK = (1ull << TAG_BITS) - 1;
constexpr int      IDX_BITS = 18; // the vocab (163840) below 2^18
constexpr uint32_t IDX_MASK = (1u << IDX_BITS) - 1;

struct Params {
    const bfloat16 *logits; // base logits, sequence s's step t row at logits + s * logits_stride + t * step_stride
    int64_t         logits_stride, step_stride;
    const bfloat16 *embed;  // markov_w1 [vocab, DIM]
    const bfloat16 *head;   // this rank's markov_w2 rows [width, DIM]
    const int64_t  *anchor; // sequence s's at anchor[s * anchor_stride]
    int64_t         anchor_stride;
    const float    *temps; // [nseq]
    int64_t        *offset;
    uint32_t        key_lo, key_hi;
    int             vocab_start, width, padded, gamma, nseq, blocks, rank, tp, part, vocab;
    int64_t        *out; // sequence s's token of step t at out[s * out_stride + t]
    int64_t         out_stride;
    int64_t        *verify; // or null: sequence s's [anchor | tokens] at verify[s * verify_stride ...] (the verify's ids)
    int64_t         verify_stride;
    int64_t        *tok;    // [nseq] the last step's tokens
    uint32_t       *counts; // [blocks] each block's launches
    bfloat16       *whole;  // sequence s's step t row at whole + s * whole_seq_stride + t * whole_step_stride, [vocab]
    int64_t         whole_seq_stride, whole_step_stride;
    bfloat16       *dst; // the walk's step logits of this rank's shard: whole where it is the shard, else the slab
    int64_t         dst_seq_stride, dst_step_stride;
    void           *recv[MAX_TP]; // every rank's exchange buffer
    uint64_t        deadline_ticks;
};

__device__ inline void keep_best(float &v, int &i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) {
        v = v2, i = i2;
    }
}

__device__ inline float round_bf16(float x) { return static_cast<float>(static_cast<bfloat16>(x)); }

// Philox4x32-10 (tl.philox's): all four words
__device__ inline void philox4(uint32_t k0, uint32_t k1, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                               uint32_t (&w)[4]) {
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
    w[0] = c0, w[1] = c1, w[2] = c2, w[3] = c3;
}

// the noise of Philox word w
__device__ inline float gumbel_of(uint32_t w) {
    const float u = (static_cast<float>(w) + 0.5f) * 2.3283064365386963e-10f;
    return logf(-log1pf(-u));
}

// a sampled sequence's 4 rows from v (v + vocab_start a multiple of 4), one Philox call
__device__ inline void gumbel4(const Params &p, int seq, int step, int64_t offset, int v, float *out) {
    const uint32_t g = static_cast<uint32_t>(p.vocab_start + v);
    uint32_t       w[4];
    philox4(p.key_lo, p.key_hi, g >> 2, static_cast<uint32_t>(seq), static_cast<uint32_t>(step),
            static_cast<uint32_t>(offset), w);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        out[k] = gumbel_of(w[k]);
    }
}

// the Gumbel noise of a sampled row; MarkovSampleStep's counters: (global token / 4, row, step, block offset), word
// token % 4
__device__ inline float gumbel(const Params &p, int seq, int step, int64_t offset, int v) {
    const uint32_t g = static_cast<uint32_t>(p.vocab_start + v);
    uint32_t       w[4];
    philox4(p.key_lo, p.key_hi, g >> 2, static_cast<uint32_t>(seq), static_cast<uint32_t>(step),
            static_cast<uint32_t>(offset), w);
    const uint32_t k = g & 3u;
    return gumbel_of(k == 0 ? w[0] : k == 1 ? w[1] : k == 2 ? w[2] : w[3]);
}

// the row's selection key: its value at temperature 0, else value / t minus its noise; a NaN wins (the lowest index)
__device__ inline float select_key(float t, float noise, float value) {
    const float key = t != 0.0f ? value / t - noise : value;
    return key != key ? INFINITY : key;
}

__device__ inline uint32x4 load_nt(const uint32x4 *ptr) {
    uint32x4 val;
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1 nt\n\ts_waitcnt vmcnt(0)" : "=v"(val) : "v"(ptr) : "memory");
    return val;
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

// the source slot this rank's record takes in destination d's buffer: its own rank; the one-GPU loopback build
// (tests and timing only: every peer address is this rank's buffer) fills every source's slot instead
__device__ inline int record_src(const Params &p, int d) {
#ifdef MARKOV_LOOPBACK
    return d;
#else
    (void)d;
    return p.rank;
#endif
}

// DPP moves within a 16-lane row (no LDS): row_ror 8 is lane ^ 8; row_ror 4 is lane ^ 4 once lanes l and l ^ 8 hold
// equal values; the quad permutations are lane ^ 2 and lane ^ 1
template <int CTRL> __device__ inline uint32_t dpp(uint32_t v) {
    return static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(v), CTRL, 0xF, 0xF, false));
}
constexpr int ROR8 = 0x128, ROR4 = 0x124, XOR2 = 0x4E, XOR1 = 0xB1;

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

#define MK_DPP64(X, C)                                                                                                 \
    (static_cast<uint64_t>(dpp<C>(static_cast<uint32_t>((X) >> 32))) << 32 | dpp<C>(static_cast<uint32_t>(X)))

// the wave's largest key: within each row by DPP, then across the four rows
__device__ inline uint64_t wave_max(uint64_t x) {
    const auto step = [&](uint64_t y) { return y > x ? y : x; };
    x = step(MK_DPP64(x, ROR8));
    x = step(MK_DPP64(x, ROR4));
    x = step(MK_DPP64(x, XOR2));
    x = step(MK_DPP64(x, XOR1));
    x = step(__shfl_xor(x, 16, WAVE));
    x = step(__shfl_xor(x, 32, WAVE));
    return x;
}

// a step's value of shard row v: MarkovSampleStep's bf16 round of the base logit plus the bf16 bias (0 past width)
__device__ inline float step_value(bfloat16 base, float dot, bool biased) {
    return round_bf16(static_cast<float>(base) + (biased ? round_bf16(dot) : 0.0f));
}

// poll this thread's records (slot(i) for i = first + k * stride, k < N, i < count) until each carries the tag;
// the largest key of each into got. Past the deadline the launch traps, stopping this rank's step
template <int N, typename Slot>
__device__ inline void poll_records(uint64_t (&got)[N], int first, int stride, int count, uint64_t tag,
                                    uint64_t deadline, Slot slot) {
    uint32_t pending = 0;
#pragma unroll
    for (int k = 0; k < N; ++k) {
        got[k] = 0;
        pending |= (first + k * stride < count ? 1u : 0u) << k;
    }
    uint64_t t0 = 0;
    for (int n = 1; pending; ++n) {
        uint64_t v[N];
#pragma unroll
        for (int k = 0; k < N; ++k) {
            if ((pending >> k) & 1u) {
                v[k] = __hip_atomic_load(slot(first + k * stride), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
#pragma unroll
        for (int k = 0; k < N; ++k) {
            if (((pending >> k) & 1u) && (v[k] & TAG_MASK) == tag) {
                got[k] = v[k] >> TAG_BITS;
                pending &= ~(1u << k);
            }
        }
        if (pending && n % 1024 == 0) {
            const uint64_t now = __builtin_amdgcn_s_memrealtime();
            if (t0 == 0) {
                t0 = now;
            } else if (now - t0 > deadline) {
                __builtin_trap();
            }
        }
    }
}

#ifdef MARKOV_STAMPS // a timing build (tests only): thread 0's realtime ticks per block, past the launch counts
constexpr int STAMPS = 2 + 5 * MAX_STEPS + 8;
#define MARKOV_STAMP(k)                                                                                               \
    if (threadIdx.x == 0)                                                                                              \
    reinterpret_cast<uint64_t *>(p.counts + MAX_BLOCKS)[blockIdx.x * STAMPS + (k)] = __builtin_amdgcn_s_memrealtime()
#else
#define MARKOV_STAMP(k)
#endif

// the whole walk in one launch, the grid co-resident (a block a CU at most, two waves an EU). A block keeps
// BLOCK_ROWS rows of the shard in registers for every step, as a wave's two 16-row tiles of 16x16x32 MFMA B
// operands; a step's bias rows of every sequence are one MFMA chain a tile in K order (the sequences' embedding rows
// the A operand, from LDS). Each step every block stores its best key of each sequence into its slot of every rank's
// records, and wave s reads every rank's blocks' records of sequence s from its own buffer: the grid barrier and the
// ranks' swap at once, one cross-GPU trip a step, no atomics
__global__ __launch_bounds__(THREADS) __attribute__((amdgpu_waves_per_eu(2))) void
markov_block_kernel(const Params p) {
    const int nseq = p.nseq;
    __shared__ float    sv[MAX_SEQ][BLOCK_ROWS];     // the bias GEMV of each sequence
    __shared__ __attribute__((aligned(16))) bfloat16 sbase[MAX_STEPS][MAX_SEQ][BLOCK_ROWS]; // every step's base logits
    __shared__ float    snoise[MAX_SEQ][BLOCK_ROWS]; // the sampled sequences' noise of this step
    __shared__ float    wave_bestv[MAX_SEQ][BLOCK_ROWS / WAVE];
    __shared__ int      wave_bi[MAX_SEQ][BLOCK_ROWS / WAVE];
    __shared__ __attribute__((aligned(16))) bfloat16 erow[MAX_SEQ][DIM]; // the step's tokens' embedding rows
    __shared__ int64_t  prev[MAX_SEQ];
    __shared__ float    stemp[MAX_SEQ]; // the sequences' temperatures, read once
    __shared__ uint32_t s_count;
    const int           wave = threadIdx.x / WAVE, l = threadIdx.x % WAVE, col = l % 16, kgrp = l / 16;
    MARKOV_STAMP(0);
    // the wave's tiles of the shard: B operands, lane (col, kgrp) holding row tile * 16 + col's 8 values at
    // 32 c + 8 kgrp
    bf16x8 wb[TILES][DIM / 32];
#pragma unroll
    for (int t = 0; t < TILES; ++t) {
        const int v = blockIdx.x * BLOCK_ROWS + (wave * TILES + t) * 16 + col;
#pragma unroll
        for (int c = 0; c < DIM / 32; ++c) {
            wb[t][c] = v < p.width ? *reinterpret_cast<const bf16x8 *>(p.head + static_cast<int64_t>(v) * DIM + 32 * c +
                                                                       8 * kgrp)
                                   : bf16x8{};
        }
    }
    // every step's base logits at the start, 8 a thread a load: no step waits for a load of them
    for (int i = threadIdx.x; i < p.gamma * nseq * (BLOCK_ROWS / 8); i += THREADS) {
        const int c = i % (BLOCK_ROWS / 8), seq = (i / (BLOCK_ROWS / 8)) % nseq, step = i / (BLOCK_ROWS / 8 * nseq);
        const int v = blockIdx.x * BLOCK_ROWS + 8 * c;
        if (v < p.padded) { // padded % 8 == 0: the 8 are whole
            *reinterpret_cast<uint4 *>(&sbase[step][seq][8 * c]) =
                *reinterpret_cast<const uint4 *>(p.logits + seq * p.logits_stride + step * p.step_stride + v);
        }
    }
    if (threadIdx.x == 0) {
        s_count = p.counts[blockIdx.x];
    }
    if (threadIdx.x < nseq) {
        prev[threadIdx.x]  = p.anchor[threadIdx.x * p.anchor_stride];
        stemp[threadIdx.x] = p.temps[threadIdx.x];
        if (blockIdx.x == 0 && p.verify != nullptr) p.verify[threadIdx.x * p.verify_stride] = prev[threadIdx.x];
    }
    if (threadIdx.x < nseq * (DIM / 8)) { // the anchors' embedding rows, 8 values a thread
        const int seq = threadIdx.x / (DIM / 8), c = threadIdx.x % (DIM / 8);
        reinterpret_cast<uint4 *>(erow[seq])[c] =
            *reinterpret_cast<const uint4 *>(p.embed + p.anchor[seq * p.anchor_stride] * DIM + c * 8);
    }
    const int64_t offset = p.offset[0];
    // a step's noise: every row of every sampled sequence (it does not depend on the tokens: computed ahead of its
    // step). Up to 4 sequences a row a thread (its quad's Philox call, its word), past 4 a quad of rows a thread (one
    // call, its four words): the same values. temps: the step's reads of the temperatures, p.temps before the first
    // barrier, stemp after it
    const auto noise_of = [&](int step, const float *temps) {
        if (nseq <= 4) {
            for (int i = threadIdx.x; i < nseq * BLOCK_ROWS; i += THREADS) {
                const int seq = i / BLOCK_ROWS, r = i % BLOCK_ROWS, v = blockIdx.x * BLOCK_ROWS + r;
                if (temps[seq] != 0.0f && v < p.width) {
                    snoise[seq][r] = gumbel(p, seq, step, offset, v);
                }
            }
        } else {
            for (int i = threadIdx.x; i < nseq * (BLOCK_ROWS / 4); i += THREADS) {
                const int seq = i / (BLOCK_ROWS / 4), r = 4 * (i % (BLOCK_ROWS / 4)), v = blockIdx.x * BLOCK_ROWS + r;
                if (temps[seq] != 0.0f && v < p.width) { // width % 4 == 0: the quad is whole
                    gumbel4(p, seq, step, offset, v, &snoise[seq][r]);
                }
            }
        }
    };
    noise_of(0, p.temps);
    __syncthreads();
    const uint32_t count  = s_count;
    const uint64_t tag    = 1 + count % static_cast<uint32_t>(TAG_MASK); // never 0; the launch two back differs
    const int      parity = static_cast<int>(count & 1u);
    uint64_t      *mine   = static_cast<uint64_t *>(p.recv[p.rank]);
    MARKOV_STAMP(1);
    for (int step = 0; step < p.gamma; ++step) {
        // every sequence's bias rows, lane (col, kgrp) then holding sequence 4 kgrp + i's bias of row tile * 16 + col:
        // hipBLASLt's order for SGLang's shard GEMV (Tensile MT16x16x128, checked bitwise at 1 to 8 rows): K in two
        // steps of 128, chunk q of 32 of each step into partial q in K order, the 4 partials summed in order
#pragma unroll
        for (int t = 0; t < TILES; ++t) {
            f32x4 part[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                part[q] = f32x4{0.0f, 0.0f, 0.0f, 0.0f};
            }
#pragma unroll
            for (int c = 0; c < DIM / 32; ++c) {
                const bf16x8 a = col < nseq ? *reinterpret_cast<const bf16x8 *>(&erow[col][32 * c + 8 * kgrp]) : bf16x8{};
                part[c % 4]    = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, wb[t][c], part[c % 4], 0, 0, 0);
            }
            const f32x4 acc = ((part[0] + part[1]) + part[2]) + part[3];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (4 * kgrp + i < nseq) sv[4 * kgrp + i][(wave * TILES + t) * 16 + col] = acc[i];
            }
        }
        __syncthreads();
        MARKOV_STAMP(2 + 5 * step);
        // the scores: the first four waves sequences 0, 2, .., the last four 1, 3, ..; two sequences a pass a thread,
        // their chains independent
        {
            const int r = threadIdx.x % BLOCK_ROWS, half = threadIdx.x / BLOCK_ROWS, v = blockIdx.x * BLOCK_ROWS + r;
            for (int pair = half; pair < nseq; pair += 4) {
                float best[2] = {-INFINITY, -INFINITY};
                int   bi[2]   = {INT_MAX, INT_MAX};
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int seq = pair + 2 * j;
                    if (seq < nseq && v < p.padded) {
                        const float value = step_value(sbase[step][seq][r], sv[seq][r], v < p.width);
                        p.dst[seq * p.dst_seq_stride + step * p.dst_step_stride + v] = static_cast<bfloat16>(value);
                        if (v < p.width) {
                            best[j] = select_key(stemp[seq], snoise[seq][r], value);
                            bi[j]   = v;
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int seq = pair + 2 * j; // alike in a wave
                    if (seq < nseq) {
                        wave_best(best[j], bi[j]);
                        if (r % WAVE == 0) {
                            wave_bestv[seq][r / WAVE] = best[j], wave_bi[seq][r / WAVE] = bi[j];
                        }
                    }
                }
            }
        }
        __syncthreads();
        MARKOV_STAMP(3 + 5 * step);
        // this block's record of each sequence into every rank's slot
        if (threadIdx.x < nseq * p.tp) {
            const int seq = threadIdx.x / p.tp, d = threadIdx.x % p.tp;
            float     best = wave_bestv[seq][0];
            int       bi   = wave_bi[seq][0];
#pragma unroll
            for (int w = 1; w < BLOCK_ROWS / WAVE; ++w) {
                keep_best(best, bi, wave_bestv[seq][w], wave_bi[seq][w]);
            }
            const uint64_t key = bi == INT_MAX ? 0ull : candidate_key(best, bi + p.vocab_start);
            uint64_t *const rec = static_cast<uint64_t *>(p.recv[d]) +
                                  record_slot(parity, step, seq, record_src(p, d), blockIdx.x);
            // a record carries its data; only the slab needs ordering: the peers' gather_step_logits read it once
            // their walks saw the last records, so those release the block's slab stores (the barrier does not
            // wait for them). A release every step would write back the L2 every step.
            if (step + 1 == p.gamma && p.dst != p.whole) {
                __hip_atomic_store(rec, (key << TAG_BITS) | tag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            } else {
                __hip_atomic_store(rec, (key << TAG_BITS) | tag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
        if (step + 1 < p.gamma) { // the next step's noise while the records cross (snoise's reads are behind a barrier)
            noise_of(step + 1, stemp);
        }
        MARKOV_STAMP(4 + 5 * step);
        // wave s: every rank's blocks' records of sequence s, then its token and the token's embedding row
        if (wave < nseq) {
            uint64_t got[POLL_ONE];
            poll_records<POLL_ONE>(got, l, WAVE, p.tp * p.blocks, tag, p.deadline_ticks, [&](int i) {
                return mine + record_slot(parity, step, wave, i / p.blocks, i % p.blocks);
            });
            MARKOV_STAMP(5 + 5 * step);
            uint64_t top = 0;
#pragma unroll
            for (int k = 0; k < POLL_ONE; ++k) {
                top = got[k] > top ? got[k] : top;
            }
            top               = wave_max(top);
            const int64_t tok = key_index(top);
            if (step + 1 < p.gamma && l < DIM / 8) {
                reinterpret_cast<uint4 *>(erow[wave])[l] = *reinterpret_cast<const uint4 *>(p.embed + tok * DIM + l * 8);
            }
            if (l == 0) {
                prev[wave] = tok;
                if (blockIdx.x == 0) {
                    p.out[wave * p.out_stride + step] = tok;
                    if (p.verify != nullptr) p.verify[wave * p.verify_stride + 1 + step] = tok;
                }
            }
        }
        __syncthreads();
        MARKOV_STAMP(6 + 5 * step);
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
}


// every rank's shard of the sequences' step logits (each rank's exchange buffer, past EXCHANGE_BYTES, as the
// Markov launch left them) into the whole rows, 8 bf16 a thread
__global__ __launch_bounds__(THREADS) void gather_kernel(const Params p) {
    const int       row = blockIdx.y; // (sequence, step, source rank)
    const int       seq = row / (p.gamma * p.tp), step = (row / p.tp) % p.gamma, src = row % p.tp;
    const int       start = src * p.part;
    const int       octs  = min(p.part, p.vocab - start) / 8;
    const uint32x4 *from  = reinterpret_cast<const uint32x4 *>(static_cast<const char *>(p.recv[src]) + EXCHANGE_BYTES) +
                           (static_cast<int64_t>(seq) * p.gamma + step) * (p.part / 8);
    uint32x4 *to = reinterpret_cast<uint32x4 *>(p.whole + seq * p.whole_seq_stride + step * p.whole_step_stride + start);
    const int i  = blockIdx.x * THREADS + threadIdx.x;
    if (i < octs) {
        to[i] = load_nt(from + i);
    }
}

void check(const at::Tensor &t, const at::Tensor &ref, at::ScalarType dtype, const char *name) {
    TORCH_CHECK(t.device() == ref.device() && t.scalar_type() == dtype, name, " must be ", dtype, " on ",
                ref.device());
    TORCH_CHECK((t.stride(-1) == 1 || t.numel() == 1) && reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0, name,
                " must have unit inner stride");
}

void check_peers(const at::Tensor &peers) {
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong && peers.numel() >= 1 &&
                    peers.numel() <= MAX_TP,
                "peers: every rank's exchange buffer (int64, host)");
}

void check_whole(const at::Tensor &whole, int64_t part, int64_t tp) {
    const int64_t vocab = whole.size(2);
    TORCH_CHECK(part % 8 == 0 && vocab % 8 == 0 && whole.stride(0) % 8 == 0 && whole.stride(1) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(whole.data_ptr()) % 16 == 0,
                "16 B aligned rows and shards");
    // whole: the vocab's rows, or this rank's shard of them where the accept keeps the shard
    TORCH_CHECK(vocab == part || (tp * part >= vocab && (tp - 1) * part < vocab),
                "the shards tile the vocab, or whole is a shard");
}

// the walk's launch parameters: nseq sequences of gamma steps over padded columns; embed [vocab, 256], head
// [width, 256] bf16, anchor / tok [nseq] int64, temps [nseq] f32, offset [1] int64 (advanced once), out int64
// (sequence s's step t token at s * out_stride + t), control int32 [blocks] (zeroed once), peers [tp] int64 on the
// host: every rank's exchange buffer; whole [nseq, gamma, vocab] bf16 (any outer strides): the rows
// gather_step_logits fills, every rank's shard of part columns
Params markov_params(int64_t nseq, int64_t gamma, int64_t padded, const at::Tensor &ref, const at::Tensor &embed,
                     const at::Tensor &head, const at::Tensor &anchor, const at::Tensor &temps, at::Tensor &offset,
                     int64_t key, int64_t vocab_start, at::Tensor &out, int64_t out_stride, at::Tensor &tok,
                     at::Tensor &control, const at::Tensor &peers, int64_t rank, at::Tensor &whole, int64_t part,
                     double deadline_s, const c10::optional<at::Tensor> &verify) {
    const int64_t width = head.size(0);
    TORCH_CHECK(nseq >= 1 && nseq <= MAX_SEQ && gamma >= 1 && gamma <= MAX_STEPS, "1 .. ", MAX_SEQ,
                " sequences, 1 .. ", MAX_STEPS, " steps");
    TORCH_CHECK(head.dim() == 2 && head.size(1) == DIM && head.is_contiguous() && embed.dim() == 2 &&
                    embed.size(1) == DIM && embed.is_contiguous() && width <= padded,
                "head [width, 256], embed [vocab, 256] contiguous");
    check(embed, ref, at::kBFloat16, "embed");
    check(head, ref, at::kBFloat16, "head");
    TORCH_CHECK(anchor.device() == ref.device() && anchor.scalar_type() == at::kLong && anchor.dim() == 1 &&
                    anchor.size(0) >= nseq,
                "anchor [nseq] int64 (any stride)");
    check(temps, ref, at::kFloat, "temps");
    check(offset, ref, at::kLong, "offset");
    check(out, ref, at::kLong, "out");
    check(tok, ref, at::kLong, "tok");
    check(control, ref, at::kInt, "control");
    check(whole, ref, at::kBFloat16, "whole");
    check_peers(peers);
    TORCH_CHECK(whole.dim() == 3 && whole.size(0) >= nseq && whole.size(1) == gamma, "whole [nseq, gamma, vocab]");
    check_whole(whole, part, peers.numel());
    TORCH_CHECK(padded % 8 == 0 && width % 4 == 0, "a shard of whole 8-row chunks");
    TORCH_CHECK(padded <= part && vocab_start % 4 == 0 && vocab_start + width <= (int64_t{1} << IDX_BITS),
                "the padded shard within part; Philox quads of global ids below 2^", IDX_BITS);
    Params p{};
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
    p.padded        = static_cast<int>(padded);
    p.gamma         = static_cast<int>(gamma);
    p.nseq          = static_cast<int>(nseq);
    p.blocks        = static_cast<int>((padded + BLOCK_ROWS - 1) / BLOCK_ROWS);
    TORCH_CHECK(p.blocks <= MAX_BLOCKS && control.numel() >= p.blocks, "control [blocks] int32, at most ",
                MAX_BLOCKS, " blocks");
    TORCH_CHECK(peers.numel() * p.blocks <= POLL_ONE * WAVE, "a sequence's records on its wave");
    p.rank  = static_cast<int>(rank);
    p.tp    = static_cast<int>(peers.numel());
    p.part  = static_cast<int>(part);
    p.vocab = static_cast<int>(whole.size(2));
    p.out        = out.data_ptr<int64_t>();
    p.out_stride = out_stride;
    if (verify.has_value()) {
        check(*verify, ref, at::kLong, "verify");
        TORCH_CHECK(verify->dim() == 2 && verify->size(0) >= nseq && verify->size(1) == gamma + 1,
                    "verify [nseq, gamma + 1] int64: [anchor | tokens]");
        p.verify        = verify->data_ptr<int64_t>();
        p.verify_stride = verify->stride(0);
    }
    p.tok               = tok.data_ptr<int64_t>();
    p.counts            = reinterpret_cast<uint32_t *>(control.data_ptr<int>());
    p.whole             = static_cast<bfloat16 *>(whole.data_ptr());
    p.whole_seq_stride  = whole.stride(0);
    p.whole_step_stride = whole.stride(1);
    p.deadline_ticks    = static_cast<uint64_t>(deadline_s * 1e8); // the 100 MHz realtime clock
    for (int r = 0; r < p.tp; ++r) {
        p.recv[r] = reinterpret_cast<void *>(peers.data_ptr<int64_t>()[r]);
    }
    if (p.vocab == p.part) { // whole is this rank's shard (a sharded accept reads it): the walk writes it there
        p.dst             = p.whole;
        p.dst_seq_stride  = p.whole_seq_stride;
        p.dst_step_stride = p.whole_step_stride;
    } else { // the slab, which gather_step_logits reads on every rank
        p.dst = reinterpret_cast<bfloat16 *>(static_cast<char *>(p.recv[p.rank]) + EXCHANGE_BYTES);
        p.dst_seq_stride  = gamma * part;
        p.dst_step_stride = part;
    }
    return p;
}

// base [nseq, gamma, padded] bf16 (unit inner stride), then markov_params' arguments
void markov_block(const at::Tensor &base, const at::Tensor &embed, const at::Tensor &head, const at::Tensor &anchor,
                  const at::Tensor &temps, at::Tensor &offset, int64_t key, int64_t vocab_start, at::Tensor &out,
                  int64_t out_stride, at::Tensor &tok, at::Tensor &control, const at::Tensor &peers, int64_t rank,
                  at::Tensor &whole, int64_t part, double deadline_s, const c10::optional<at::Tensor> &verify) {
    TORCH_CHECK(base.dim() == 3, "base [sequences, steps, padded]");
    check(base, base, at::kBFloat16, "base");
    Params p = markov_params(base.size(0), base.size(1), base.size(2), base, embed, head, anchor, temps, offset, key,
                             vocab_start, out, out_stride, tok, control, peers, rank, whole, part, deadline_s, verify);
    p.logits        = static_cast<const bfloat16 *>(base.data_ptr());
    p.logits_stride = base.stride(0);
    p.step_stride   = base.stride(1);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(base.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(base.device().index()).stream();
    markov_block_kernel<<<p.blocks, THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

// peers [tp] int64 on the host (every rank's exchange buffer), whole [nseq, gamma, vocab] bf16 (any outer strides):
// every sequence's step logits, every rank's shard of part columns
void gather_step_logits(const at::Tensor &peers, at::Tensor &whole, int64_t part) {
    check(whole, whole, at::kBFloat16, "whole");
    const int64_t nseq = whole.size(0), gamma = whole.size(1);
    TORCH_CHECK(whole.dim() == 3 && nseq >= 1 && nseq <= MAX_SEQ && gamma >= 1 && gamma <= MAX_STEPS,
                "whole [1 .. 8, gamma, vocab]");
    check_peers(peers);
    check_whole(whole, part, peers.numel());
    Params p{};
    p.gamma             = static_cast<int>(gamma);
    p.nseq              = static_cast<int>(nseq);
    p.tp                = static_cast<int>(peers.numel());
    p.part              = static_cast<int>(part);
    p.vocab             = static_cast<int>(whole.size(2));
    p.whole             = static_cast<bfloat16 *>(whole.data_ptr());
    p.whole_seq_stride  = whole.stride(0);
    p.whole_step_stride = whole.stride(1);
    for (int r = 0; r < p.tp; ++r) {
        p.recv[r] = reinterpret_cast<void *>(peers.data_ptr<int64_t>()[r]);
    }
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(whole.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(whole.device().index()).stream();
    gather_kernel<<<dim3((part / 8 + THREADS - 1) / THREADS, nseq * gamma * p.tp), THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("markov_block(Tensor base, Tensor embed, Tensor head, Tensor anchor, Tensor temps, Tensor(a!) offset, "
          "int key, int vocab_start, Tensor(b!) out, int out_stride, Tensor(c!) tok, Tensor(d!) control, "
          "Tensor peers, int rank, Tensor(e!) whole, int part, float deadline_s, Tensor(f!)? verify=None) -> ()");
    m.impl("markov_block", at::kCUDA, &markov_block);
    m.def("gather_step_logits(Tensor peers, Tensor(a!) whole, int part) -> ()");
    m.impl("gather_step_logits", at::kCUDA, &gather_step_logits);
}
