// SPDX-License-Identifier: Apache-2.0
// A greedy-variant static verify's head projection and accept in one launch:
// - every block stages the verify's normed rows in LDS (their loads issued ahead of its weights', so they return
//   first) and streams its 8 tiles of 16 vocab rows of this rank's head shard (headgemm.hip's variant 13: bf16
//   16x16x32 MFMAs, fp32 accumulation, one bf16 rounding) into the bf16 logits shard;
// - each block keys its columns of the key rows (every verify row, or under a simulated accept length each
//   request's row at that length, Gumbel-noised at the request's temperature as SGLang's simulated bonus: vtail.hip's
//   arithmetic) and pushes each row's best (value, global index) to every rank as a launch-tagged record;
// - each rank's last block (the fewest tiles) takes every rank's blocks' records and runs SGLang's static accept
//   tail on the bests (the lowest index among the maxima), as vtail.hip does.
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
using bf16x8   = __bf16 __attribute__((ext_vector_type(8)));
using f32x4    = float __attribute__((ext_vector_type(4)));
using uint32x4 = uint32_t __attribute__((ext_vector_type(4)));

constexpr int WAVE    = 64;
constexpr int NWAVES  = 8;
constexpr int THREADS = NWAVES * WAVE;
// the GEMM: a wave a 16-column tile over the whole K, V 16 B vectors a lane an iteration, DEPTH iterations in flight
constexpr int TILE_N = 16;
constexpr int V      = 8;
constexpr int DEPTH  = 2;
constexpr int GAP    = 32;        // a row's 4 lanes contiguous in each load instruction
constexpr int STEP   = 4 * 8 * V; // K an iteration
constexpr int MAX_K  = 6144;
// the launch
constexpr int MAX_M      = 6; // verify rows (x [M, K] in LDS beside the waves' weight tiles)
constexpr int MAX_TP     = 8;
constexpr int MAX_T      = 8;
constexpr int MAX_BLOCKS = 256;
constexpr int POLLS      = 16; // records a thread keeps in flight
constexpr int X_LOADS    = (MAX_M * MAX_K / 8 + THREADS - 1) / THREADS; // 16 B of x a thread, at most
// uncached scratch, peer-mapped: every rank's blocks' records [launch parity 2][key row][source rank][block]
constexpr int64_t SCRATCH_BYTES = 2ll * MAX_M * MAX_TP * MAX_BLOCKS * 8;
// a record or slot: the candidate key above a 15-bit launch tag
constexpr int      MAX_HISTORY = 8; // the Engram history's width at most
constexpr int      TAG_BITS = 15;
constexpr uint64_t TAG_MASK = (1ull << TAG_BITS) - 1;
constexpr int      IDX_BITS = 17; // the vocab (129280) below 2^17
constexpr uint32_t IDX_MASK = (1u << IDX_BITS) - 1;

struct Params {
    // the GEMM: x [M, K] bf16 (the head's normed rows), this rank's shard [part, K] bf16, the logits [M, part] out
    const bfloat16 *x;
    const bfloat16 *w;
    int64_t         w_stride;
    bfloat16       *logits;
    int64_t         logits_stride;
    int             M, K, part, tiles;
    unsigned char  *scratch; // this rank's uncached scratch
    // the accept (vtail.hip's Params)
    int            width, vocab_start, vocab, blocks, rank, tp;
    int            bs, T, keys;
    const int64_t *sim;
    const float   *temps;
    const int32_t *seed;
    const int64_t *candidates;
    int64_t        cand_stride;
    const int64_t *seq_lens, *verify_lens;
    const int32_t *gate;
    int64_t       *correct_out, *bonus_out, *new_seq_out, *out_tokens, *target_out;
    int32_t       *cap_trim_out, *commit_out, *gated_out;
    int64_t        out_tokens_stride;
    // the Engram history commit in the accept (or null): history [slots, history_width] int32, the requests'
    // slots [bs] int64, the FutureMap relay rows (new seq lens, bonus) by slot (or null)
    int32_t        *history;
    int64_t         history_stride;
    int             history_width;
    const int64_t  *slots;
    int64_t        *relay_lens, *relay_tokens;
    uint32_t      *counts; // [blocks] each block's launches
    void          *recv[MAX_TP];
    uint64_t       deadline_ticks;
};

// ---- vtail.hip's keys ----

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

__device__ inline float uniform(int32_t seed, int32_t offset) {
    const uint64_t s = static_cast<uint64_t>(static_cast<int64_t>(seed));
    int32_t        x = static_cast<int32_t>(
        philox_c0(static_cast<uint32_t>(s), static_cast<uint32_t>(s >> 32), static_cast<uint32_t>(offset)));
    x = x < 0 ? -x - 1 : x;
    return static_cast<float>(x) * 4.6566127342e-10f;
}

// a candidate as one ordered word above the inverted global index (0: none); NaN is never one, as vtail.hip's float
// compare never keeps it
__device__ inline uint64_t candidate_key(float value, int index) {
    if (value != value) {
        return 0;
    }
    value               = value == 0.0f ? 0.0f : value; // -0 ties +0
    const uint32_t bits = __builtin_bit_cast(uint32_t, value);
    const uint32_t ord  = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    return (static_cast<uint64_t>(ord) << IDX_BITS) | static_cast<uint64_t>(IDX_MASK - static_cast<uint32_t>(index));
}

__device__ inline int64_t key_index(uint64_t key) {
    return static_cast<int64_t>(IDX_MASK - static_cast<uint32_t>(key & IDX_MASK));
}

__device__ inline uint64_t umax64(uint64_t a, uint64_t b) { return a > b ? a : b; }

__device__ inline uint64_t shfl_xor64(uint64_t v, int s) {
    const uint32_t lo = __shfl_xor(static_cast<uint32_t>(v), s, WAVE);
    const uint32_t hi = __shfl_xor(static_cast<uint32_t>(v >> 32), s, WAVE);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

// ---- the hand-offs ----

__device__ inline void deadline_check(uint64_t &t0, uint64_t deadline) {
    const uint64_t now = __builtin_amdgcn_s_memrealtime();
    if (t0 == 0) {
        t0 = now;
    } else if (now - t0 > deadline) {
        __builtin_trap(); // past the deadline the launch traps, stopping this rank's step
    }
}

__device__ inline void store_dev(uint64_t *p, uint64_t v) {
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline uint64_t load_dev(const uint64_t *p) {
    return __hip_atomic_load(const_cast<uint64_t *>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

struct Chunk {
    bf16x8 v[V];
};

// A wave's 16 weight rows of an iteration (16 x 512 B), non-temporal (gfx950 nt: streamed once, they are not kept in
// the caches, so the data the next launches read stays there, as the head GEMM's nt weight loads leave it): load j
// takes rows 2j and 2j + 1, 32 lanes a row, whole 128 B lines (a half line a load costs nt its second fetch)
constexpr int NT      = 2;
constexpr int WROW_LD = 512 + 16; // a row's bytes in a wave's LDS tile, padded
__device__ inline Chunk load_w(__amdgpu_buffer_rsrc_t w, const int (&off)[V], int it) {
    Chunk c;
#pragma unroll
    for (int j = 0; j < V; ++j) {
        c.v[j] = __builtin_bit_cast(bf16x8, __builtin_amdgcn_raw_buffer_load_b128(w, off[j] + it * STEP * 2, 0, NT));
    }
    return c;
}

__device__ inline int64_t record_slot(int parity, int k, int src, int block) {
    return ((static_cast<int64_t>(parity) * MAX_M + k) * MAX_TP + src) * MAX_BLOCKS + block;
}

// every rank's blocks' records of key rows 0 .. keys - 1 (uncached) until each carries the tag, POLLS a thread in
// flight; each key row's best into top
__device__ inline void gather_records(const uint64_t *rec, int parity, int keys, int tp, int blocks, uint64_t tag,
                                      uint64_t *top, uint64_t deadline) {
    const int n = keys * tp * blocks;
    for (int base = 0; base < n; base += THREADS * POLLS) {
        uint32_t pending = 0;
#pragma unroll
        for (int j = 0; j < POLLS; ++j) {
            if (base + j * THREADS + static_cast<int>(threadIdx.x) < n) {
                pending |= 1u << j;
            }
        }
        uint64_t t0 = 0;
        for (int it = 1; pending; ++it) {
            uint64_t v[POLLS];
#pragma unroll
            for (int j = 0; j < POLLS; ++j) {
                if ((pending >> j) & 1u) {
                    const int i = base + j * THREADS + threadIdx.x;
                    v[j] = __hip_atomic_load(const_cast<uint64_t *>(rec) +
                                                 record_slot(parity, i / (tp * blocks), (i / blocks) % tp, i % blocks),
                                             __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                }
            }
#pragma unroll
            for (int j = 0; j < POLLS; ++j) {
                if (((pending >> j) & 1u) && (v[j] & TAG_MASK) == tag) {
                    const int k = (base + j * THREADS + static_cast<int>(threadIdx.x)) / (tp * blocks);
                    atomicMax(reinterpret_cast<unsigned long long *>(&top[k]),
                              static_cast<unsigned long long>(v[j] >> TAG_BITS));
                    pending &= ~(1u << j);
                }
            }
            if (pending && it % 1024 == 0) {
                deadline_check(t0, deadline);
            }
        }
    }
}

__global__ __launch_bounds__(THREADS) void fused_tail_kernel(const Params p) {
    extern __shared__ __align__(16) unsigned char smem[];
    bfloat16 *xs = reinterpret_cast<bfloat16 *>(smem);
    __shared__ uint64_t s_key[NWAVES][MAX_M];
    __shared__ uint64_t s_fin[MAX_M];
    __shared__ int64_t  s_cand[MAX_M], s_seq[MAX_M], s_vlen[MAX_M];
    const int wave = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
    const int K = p.K, M = p.M;

    // the rows' loads first, then the weights': loads return in issue order
    const uint32x4 *xsrc   = reinterpret_cast<const uint32x4 *>(p.x);
    const int       xwords = M * K / 8;
    uint32x4        xv[X_LOADS];
#pragma unroll
    for (int j = 0; j < X_LOADS; ++j) {
        const int i = j * THREADS + threadIdx.x;
        if (i < xwords) xv[j] = xsrc[i];
    }
    __builtin_amdgcn_sched_barrier(0);
    // this wave's tile: 16 vocab rows of the shard, each lane 16 B of a row in each of V loads an iteration
    const int       tile  = blockIdx.x * NWAVES + wave;
    const int       n0    = tile * TILE_N;
    const int       col = lane % TILE_N, grp = lane / TILE_N;
    const int       koff  = grp * 8;
    const __amdgpu_buffer_rsrc_t wr = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<bfloat16 *>(p.w), 0, static_cast<int>(int64_t(p.part) * p.w_stride * 2), 0x00020000);
    int woff[V]; // load j's byte offset: row 2j + lane / 32, 16 B (lane % 32) of its 512
#pragma unroll
    for (int j = 0; j < V; ++j) {
        woff[j] = static_cast<int>((int64_t(min(n0 + 2 * j + lane / 32, p.part - 1)) * p.w_stride) * 2 + (lane % 32) * 16);
    }
    unsigned char *wl = smem + size_t(M) * K * 2 + size_t(wave) * TILE_N * WROW_LD; // this wave's weight tile
    const int       iters = K / STEP;
    const bool      live  = tile < p.tiles;
    Chunk           buf[DEPTH];
    if (live) {
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            buf[d] = load_w(wr, woff, d);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    if (static_cast<int>(blockIdx.x) == p.blocks - 1 && static_cast<int>(threadIdx.x) < p.bs * p.T) {
        // the accept's inputs, read long before the accept
        const int r = threadIdx.x / p.T, j = threadIdx.x % p.T;
        s_cand[threadIdx.x] = p.candidates[r * p.cand_stride + j];
        if (j == 0) {
            s_seq[r]  = p.seq_lens[r];
            s_vlen[r] = p.verify_lens[r];
        }
    }
    if (threadIdx.x < NWAVES * MAX_M) {
        s_key[threadIdx.x / MAX_M][threadIdx.x % MAX_M] = 0;
    }
    if (threadIdx.x < MAX_M) {
        s_fin[threadIdx.x] = 0;
    }
    {
        uint32x4 *dst = reinterpret_cast<uint32x4 *>(xs);
#pragma unroll
        for (int j = 0; j < X_LOADS; ++j) {
            const int i = j * THREADS + threadIdx.x;
            if (i < xwords) dst[i] = xv[j];
        }
    }
    __syncthreads();
    const uint32_t count  = p.counts[blockIdx.x];
    const uint64_t tag    = 1 + count % static_cast<uint32_t>(TAG_MASK); // never 0; the launch two back differs
    const int      parity = static_cast<int>(count & 1u);
    const int      s_sim  = p.sim ? static_cast<int>(p.sim[0]) : 0;

    // the projection (headgemm.hip, variant 13)
    f32x4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    if (live) {
        const int       m    = col;
        const bool      mrow = m < M;
        const bfloat16 *xrow = xs + (mrow ? m : 0) * K + koff;
        for (int it = 0; it < iters; it += DEPTH) {
#pragma unroll
            for (int d = 0; d < DEPTH; ++d) {
                const int i = it + d;
                if (i < iters) {
                    // the rows through LDS into the B operands' layout (lane: row col, K 8 grp + 32 j)
#pragma unroll
                    for (int j = 0; j < V; ++j) {
                        *reinterpret_cast<bf16x8 *>(wl + (2 * j + lane / 32) * WROW_LD + (lane % 32) * 16) = buf[d].v[j];
                    }
                    if (i + DEPTH < iters) {
                        buf[d] = load_w(wr, woff, i + DEPTH);
                    }
                    Chunk wb;
#pragma unroll
                    for (int j = 0; j < V; ++j) {
                        wb.v[j] = *reinterpret_cast<const bf16x8 *>(wl + col * WROW_LD + j * 64 + grp * 16);
                    }
                    Chunk xa;
                    if (mrow) {
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
    }

    // the logits (lane: rows 4 grp .. + 4 of column col) and this wave's best a key row
    const int     n    = n0 + col;
    const int     g    = p.vocab_start + n;
    const int32_t seed = p.sim ? p.seed[0] : 0;
    uint64_t      kb[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = 4 * grp + r;
        kb[r]         = 0;
        if (live && row < M && n < p.part) {
            const bfloat16 out = static_cast<bfloat16>(acc[r]);
            p.logits[static_cast<int64_t>(row) * p.logits_stride + n] = out;
            const int k = p.sim ? (row % p.T == s_sim ? row / p.T : -1) : row;
            if (k >= 0 && n < p.width) {
                float       v = static_cast<float>(out);
                const float t = p.sim ? p.temps[k] : 0.0f;
                if (t > 0.0f) { // _simulated_bonus_row_partial_kernel's noisy value
                    const float u = uniform(seed, k * p.vocab + g);
                    v             = v / fmaxf(t, 1e-5f) - logf(-logf(u));
                }
                kb[r] = candidate_key(v, g);
            }
        }
    }
#pragma unroll
    for (int s = 1; s < TILE_N; s *= 2) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            kb[r] = umax64(kb[r], shfl_xor64(kb[r], s));
        }
    }
    if (col == 0) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = 4 * grp + r;
            if (row < M) {
                const int k = p.sim ? (row % p.T == s_sim ? row / p.T : -1) : row;
                if (k >= 0) s_key[wave][k] = kb[r];
            }
        }
    }
    __syncthreads();
    // the block's best a key row to every rank
    for (int i = threadIdx.x; i < p.keys * p.tp; i += THREADS) {
        const int k = i / p.tp, dst = i % p.tp;
        uint64_t  best = 0;
#pragma unroll
        for (int w = 0; w < NWAVES; ++w) best = umax64(best, s_key[w][k]);
        __hip_atomic_store(reinterpret_cast<uint64_t *>(p.recv[dst]) + record_slot(parity, k, p.rank, blockIdx.x),
                           (best << TAG_BITS) | tag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (threadIdx.x == 0) {
        p.counts[blockIdx.x] = count + 1;
    }
    if (static_cast<int>(blockIdx.x) != p.blocks - 1) {
        return;
    }

    // the last block (the fewest tiles): every rank's blocks' bests
    gather_records(reinterpret_cast<const uint64_t *>(p.scratch), parity, p.keys, p.tp, p.blocks, tag, s_fin,
                   p.deadline_ticks);
    __syncthreads();
    if (p.target_out != nullptr) {
        for (int k = threadIdx.x; k < p.keys; k += THREADS) {
            p.target_out[k] = key_index(s_fin[k]);
        }
    }
    // static_accept_tail.py, a request a thread (vtail.hip)
    const int r = threadIdx.x;
    if (r >= p.bs) {
        return;
    }
    const int      gamma = p.T - 1;
    const int64_t *cand  = s_cand + r * p.T;
    int            correct;
    int64_t        bonus;
    if (p.sim) {
        correct = s_sim;
        bonus   = key_index(s_fin[r]);
    } else {
        correct = gamma;
        for (int j = gamma - 1; j >= 0; --j) {
            if (cand[1 + j] != key_index(s_fin[r * p.T + j])) {
                correct = j;
            }
        }
        bonus = key_index(s_fin[r * p.T + correct]);
    }
    const int commit  = correct + 1;
    p.correct_out[r]  = correct;
    p.bonus_out[r]    = bonus;
    p.cap_trim_out[r] = 0;
    p.commit_out[r]   = commit;
    p.new_seq_out[r]  = s_seq[r] + commit;
    int64_t *tokens   = p.out_tokens + r * p.out_tokens_stride;
    for (int j = 0; j < p.T; ++j) {
        tokens[j] = j == correct ? bonus : (j < gamma ? cand[1 + j] : 0);
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
            for (int c = 0; c < p.history_width; ++c) old[c] = row[c];
            for (int c = 0; c < p.history_width; ++c) {
                const int src = commit + c;
                row[c] = src < p.history_width ? old[src] : static_cast<int32_t>(cand[src - p.history_width]);
            }
        }
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

// the Engram history commit's tensors (all or none of history and slots; the relay rows optional)
void commit_params(Params &p, const c10::optional<at::Tensor> &history, const c10::optional<at::Tensor> &slots,
                   const c10::optional<at::Tensor> &relay_lens, const c10::optional<at::Tensor> &relay_tokens,
                   int64_t bs) {
    TORCH_CHECK(history.has_value() == slots.has_value() && relay_lens.has_value() == relay_tokens.has_value() &&
                    (history.has_value() || !relay_lens.has_value()),
                "history and slots together, the relay rows with them");
    if (!history.has_value()) return;
    TORCH_CHECK(history->scalar_type() == at::kInt && history->dim() == 2 && history->stride(1) == 1 &&
                    history->size(1) <= MAX_HISTORY,
                "history [slots, <= 8] int32");
    TORCH_CHECK(slots->scalar_type() == at::kLong && slots->numel() >= bs && slots->stride(0) == 1, "slots [bs] int64");
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

// x [M, K] bf16 contiguous (the head's normed rows); weight [part, K] bf16 (this rank's head shard, 16 B aligned
// rows); logits [M, part] bf16 out; candidates [bs, T] int64 and the accept's tensors as vtail.hip's verify_tail;
// counts [256] int32 (zeroed once), peers [tp] int64 on the host: every rank's uncached scratch, scratch this rank's
void fused_tail(const at::Tensor &x, const at::Tensor &weight, at::Tensor &logits, const at::Tensor &candidates,
                const at::Tensor &seq_lens, const at::Tensor &verify_lens, const at::Tensor &gate,
                const c10::optional<at::Tensor> &sim, const c10::optional<at::Tensor> &temps,
                const c10::optional<at::Tensor> &seed, at::Tensor &correct_out, at::Tensor &bonus_out,
                at::Tensor &cap_trim_out, at::Tensor &commit_out, at::Tensor &new_seq_out, at::Tensor &out_tokens,
                const c10::optional<at::Tensor> &gated_out, const c10::optional<at::Tensor> &target_out,
                at::Tensor &counts, const at::Tensor &peers, int64_t scratch, int64_t rank, int64_t vocab_start,
                int64_t width, int64_t vocab, double deadline_s, const c10::optional<at::Tensor> &history,
                const c10::optional<at::Tensor> &slots, const c10::optional<at::Tensor> &relay_lens,
                const c10::optional<at::Tensor> &relay_tokens) {
    const int64_t M = x.size(0), K = x.size(1), part = weight.size(0);
    const int64_t bs = candidates.size(0), T = candidates.size(1), tp = peers.numel();
    const int64_t tiles  = (part + TILE_N - 1) / TILE_N;
    const int64_t blocks = (tiles + NWAVES - 1) / NWAVES;
    const int64_t keys   = sim.has_value() ? bs : bs * T;
    check(x, x, at::kBFloat16, "x");
    check(weight, x, at::kBFloat16, "weight");
    check(logits, x, at::kBFloat16, "logits");
    TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && M >= 1 && M <= MAX_M && M == bs * T &&
                    reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "x [M <= ", MAX_M, ", K] contiguous, M = bs * T");
    TORCH_CHECK(K % STEP == 0 && K <= MAX_K, "K a multiple of ", STEP, ", at most ", MAX_K);
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == K && weight.stride(0) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0 &&
                    part * weight.stride(0) * 2 < (int64_t(1) << 31),
                "weight [part, K], 16 B aligned rows, under 2 GB");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) >= M && logits.size(1) >= part, "logits [M, part]");
    TORCH_CHECK(blocks >= 2 && blocks <= MAX_BLOCKS, "a shard of ", part, " rows: ", blocks, " blocks");
    TORCH_CHECK(T >= 2 && T <= MAX_T, "T at most ", MAX_T);
    TORCH_CHECK(vocab < (1 << IDX_BITS) && vocab_start + width <= vocab && width <= part, "the vocab below 2^17");
    check(candidates, x, at::kLong, "candidates");
    check(seq_lens, x, at::kLong, "seq_lens");
    check(verify_lens, x, at::kLong, "verify_lens");
    check(gate, x, at::kInt, "gate");
    check(correct_out, x, at::kLong, "correct_out");
    check(bonus_out, x, at::kLong, "bonus_out");
    check(cap_trim_out, x, at::kInt, "cap_trim_out");
    check(commit_out, x, at::kInt, "commit_out");
    check(new_seq_out, x, at::kLong, "new_seq_out");
    check(out_tokens, x, at::kLong, "out_tokens");
    TORCH_CHECK(out_tokens.dim() == 2 && out_tokens.size(0) >= bs && out_tokens.size(1) >= T, "out_tokens [bs, T]");
    if (sim.has_value()) {
        check(*sim, x, at::kLong, "sim");
        TORCH_CHECK(temps.has_value() && seed.has_value(), "a simulated length needs temps and seed");
        check(*temps, x, at::kFloat, "temps");
        check(*seed, x, at::kInt, "seed");
    }
    if (gated_out.has_value()) {
        check(*gated_out, x, at::kInt, "gated_out");
    }
    if (target_out.has_value()) {
        check(*target_out, x, at::kLong, "target_out");
        TORCH_CHECK(target_out->numel() >= keys, "target_out [key rows]");
    }
    check(counts, x, at::kInt, "counts");
    TORCH_CHECK(counts.numel() >= blocks, "counts [blocks] int32");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong && tp >= 1 && tp <= MAX_TP &&
                    rank >= 0 && rank < tp,
                "peers: every rank's scratch (int64, host)");
    Params p{};
    p.x                 = static_cast<const bfloat16 *>(x.data_ptr());
    p.w                 = static_cast<const bfloat16 *>(weight.data_ptr());
    p.w_stride          = weight.stride(0);
    p.logits            = static_cast<bfloat16 *>(logits.data_ptr());
    p.logits_stride     = logits.stride(0);
    p.M                 = static_cast<int>(M);
    p.K                 = static_cast<int>(K);
    p.part              = static_cast<int>(part);
    p.tiles             = static_cast<int>(tiles);
    p.scratch           = reinterpret_cast<unsigned char *>(scratch);
    p.width             = static_cast<int>(width);
    p.vocab_start       = static_cast<int>(vocab_start);
    p.vocab             = static_cast<int>(vocab);
    p.blocks            = static_cast<int>(blocks);
    p.rank              = static_cast<int>(rank);
    p.tp                = static_cast<int>(tp);
    p.bs                = static_cast<int>(bs);
    p.T                 = static_cast<int>(T);
    p.keys              = static_cast<int>(keys);
    p.sim               = ptr<const int64_t>(sim);
    p.temps             = ptr<const float>(temps);
    p.seed              = ptr<const int32_t>(seed);
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
    p.target_out        = ptr<int64_t>(target_out);
    p.counts            = reinterpret_cast<uint32_t *>(counts.data_ptr<int>());
    p.deadline_ticks    = static_cast<uint64_t>(deadline_s * 1e8); // the 100 MHz realtime clock
    commit_params(p, history, slots, relay_lens, relay_tokens, bs);
    for (int r = 0; r < p.tp; ++r) {
        p.recv[r] = reinterpret_cast<void *>(peers.data_ptr<int64_t>()[r]);
    }
    const size_t lds = static_cast<size_t>(M) * K * 2 + NWAVES * TILE_N * WROW_LD;
    TORCH_CHECK(lds <= size_t(MAX_M) * MAX_K * 2 + NWAVES * TILE_N * WROW_LD, "x [M, K] in LDS");
    static bool wide = false; // the LDS past 64 KB, set once a process
    if (!wide) {
        C10_HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void *>(&fused_tail_kernel),
                                          hipFuncAttributeMaxDynamicSharedMemorySize,
                                          static_cast<int>(int64_t(MAX_M) * MAX_K * 2 + NWAVES * TILE_N * WROW_LD)));
        wide = true;
    }
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(x.device().index()).stream();
    fused_tail_kernel<<<p.blocks, THREADS, lds, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

int64_t scratch_bytes() { return SCRATCH_BYTES; }

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tail(Tensor x, Tensor weight, Tensor(a!) logits, Tensor candidates, Tensor seq_lens, "
          "Tensor verify_lens, Tensor gate, Tensor? sim, Tensor? temps, Tensor? seed, Tensor(b!) correct_out, "
          "Tensor(c!) bonus_out, Tensor(d!) cap_trim_out, Tensor(e!) commit_out, Tensor(f!) new_seq_out, "
          "Tensor(g!) out_tokens, Tensor(h!)? gated_out, Tensor(i!)? target_out, Tensor(j!) counts, Tensor peers, "
          "int scratch, int rank, int vocab_start, int width, int vocab, float deadline_s, "
          "Tensor(x!)? history=None, Tensor? slots=None, Tensor(y!)? relay_lens=None, "
          "Tensor(z!)? relay_tokens=None) -> ()");
    m.def("scratch_bytes() -> int", &scratch_bytes);
    m.impl("fused_tail", at::kCUDA, &fused_tail);
}
