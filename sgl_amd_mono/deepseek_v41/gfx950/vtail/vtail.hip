// SPDX-License-Identifier: Apache-2.0
// The static verify's accept from the ranks' bf16 logits shards, in one launch: each block takes its columns of
// the shard's key rows (every verify row, or under a simulated accept length each request's row at that length,
// Gumbel-noised at the request's temperature as SGLang's simulated bonus), keeps each row's best (value, global
// index) and pushes it to every rank's records; each rank's block 0 reads every rank's blocks' records, takes each
// row's best (the lowest index among the maxima) and runs SGLang's static accept tail (static_accept_tail.py).
// The values are SGLang's f32 logits (the bf16 widened); the outputs are its chain's, bit for bit.
#include <climits>
#include <cstdint>

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

constexpr int THREADS    = 256;
constexpr int WAVE       = 64;
constexpr int WAVES      = THREADS / WAVE;
constexpr int MAX_TP     = 8;
constexpr int MAX_ROWS   = 48; // key rows: a verify's rows (8 requests of 6) at most
constexpr int MAX_T      = 8;  // a request's verify rows at most
constexpr int MAX_BLOCKS = 128;
constexpr int POLLS      = 8; // records a thread keeps in flight
// the records: one 8 B record a key row, source rank and block, [launch parity 2][MAX_ROWS][MAX_TP][MAX_BLOCKS]
// (vtail/__init__.py's RECORD_BYTES); a record is the row's key above a 15-bit launch tag
constexpr int      TAG_BITS = 15;
constexpr uint64_t TAG_MASK = (1ull << TAG_BITS) - 1;
constexpr int      IDX_BITS = 17; // the vocab (129280) below 2^17
constexpr uint32_t IDX_MASK = (1u << IDX_BITS) - 1;

struct Params {
    const bfloat16 *local; // this rank's shard, verify row m at local + m * local_stride, [width] valid columns
    int64_t         local_stride;
    int             width, vocab_start, vocab, chunk, blocks, rank, tp;
    int             bs, T, keys; // requests, rows a request, key rows (bs * T, or bs under a simulated length)
    const int64_t  *sim;   // the simulated correct length [1], or null
    const float    *temps; // [bs] (with sim)
    const int32_t  *seed;  // [1] (with sim)
    const int64_t  *candidates; // request r's anchor and drafts at candidates + r * cand_stride, [T]
    int64_t         cand_stride;
    const int64_t  *seq_lens, *verify_lens;
    const int32_t  *gate;
    int64_t        *correct_out, *bonus_out, *new_seq_out, *out_tokens, *target_out;
    int32_t        *cap_trim_out, *commit_out, *gated_out;
    int64_t         out_tokens_stride;
    uint32_t       *counts; // [blocks] each block's launches
    void           *recv[MAX_TP];
    uint64_t        deadline_ticks;
};

__device__ inline void keep_best(float &v, int &i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) {
        v = v2, i = i2;
    }
}

// Philox4x32-10 (tl.philox's), the first word: tl.randint(seed, offset) of a 32-bit offset
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

// tl.rand(seed, offset): the int32 word folded to non-negative, times (2^23 - 1) / 2^23 * 2^-31
__device__ inline float uniform(int32_t seed, int32_t offset) {
    const uint64_t s = static_cast<uint64_t>(static_cast<int64_t>(seed));
    int32_t        x = static_cast<int32_t>(
        philox_c0(static_cast<uint32_t>(s), static_cast<uint32_t>(s >> 32), static_cast<uint32_t>(offset)));
    x = x < 0 ? -x - 1 : x;
    return static_cast<float>(x) * 4.6566127342e-10f;
}

// a candidate as one ordered word: the value's bits in order above the inverted global index, so the larger key is
// the larger value, then the lower index; 0 is none
__device__ inline uint64_t candidate_key(float value, int index) {
    value               = value == 0.0f ? 0.0f : value; // -0 ties +0, as the float compare does
    const uint32_t bits = __builtin_bit_cast(uint32_t, value);
    const uint32_t ord  = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    return (static_cast<uint64_t>(ord) << IDX_BITS) | static_cast<uint64_t>(IDX_MASK - static_cast<uint32_t>(index));
}

__device__ inline int64_t key_index(uint64_t key) {
    return static_cast<int64_t>(IDX_MASK - static_cast<uint32_t>(key & IDX_MASK));
}

__device__ inline int64_t record_slot(int parity, int row, int src, int block) {
    return ((static_cast<int64_t>(parity) * MAX_ROWS + row) * MAX_TP + src) * MAX_BLOCKS + block;
}

// a block a chunk of the shard's columns, a wave a key row at a time; block 0 then reduces and accepts
__device__ inline void verify_tail_body(const Params &p) {
    __shared__ uint32_t s_count;
    __shared__ int64_t  s_sim;
    __shared__ uint64_t s_top[MAX_ROWS];
    if (threadIdx.x == 0) {
        s_count = p.counts[blockIdx.x];
        s_sim   = p.sim ? p.sim[0] : 0;
    }
    if (blockIdx.x == 0 && threadIdx.x < MAX_ROWS) {
        s_top[threadIdx.x] = 0;
    }
    __syncthreads();
    const uint32_t count  = s_count;
    const uint64_t tag    = 1 + count % static_cast<uint32_t>(TAG_MASK); // never 0; the launch two back differs
    const int      parity = static_cast<int>(count & 1u);
    const int      wave = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
    const int      c0 = blockIdx.x * p.chunk, c1 = min(c0 + p.chunk, p.width);
    const int32_t  seed = p.sim ? p.seed[0] : 0;
    for (int k = wave; k < p.keys; k += WAVES) {
        const int     row = p.sim ? k * p.T + static_cast<int>(s_sim) : k;
        const float   t   = p.sim ? p.temps[k] : 0.0f;
        const float   inv = fmaxf(t, 1e-5f);
        const bfloat16 *src = p.local + row * p.local_stride;
        float         best = -INFINITY;
        int           bi   = INT_MAX;
        for (int col = c0 + 8 * lane; col < c1; col += 8 * WAVE) {
            uint4 raw;
            if (col + 8 <= p.width) {
                raw = *reinterpret_cast<const uint4 *>(src + col);
            } else {
                bfloat16 *b = reinterpret_cast<bfloat16 *>(&raw);
                for (int i = 0; i < 8; ++i) {
                    b[i] = col + i < p.width ? src[col + i] : static_cast<bfloat16>(0.0f);
                }
            }
            const bfloat16 *b = reinterpret_cast<const bfloat16 *>(&raw);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (col + i < c1) {
                    const int g = p.vocab_start + col + i;
                    float     v = static_cast<float>(b[i]);
                    if (t > 0.0f) { // _simulated_bonus_row_partial_kernel's noisy value
                        const float u = uniform(seed, k * p.vocab + g);
                        v             = v / inv - logf(-logf(u));
                    }
                    keep_best(best, bi, v, g);
                }
            }
        }
#pragma unroll
        for (int s = WAVE / 2; s > 0; s /= 2) {
            keep_best(best, bi, __shfl_xor(best, s, WAVE), __shfl_xor(bi, s, WAVE));
        }
        if (lane < p.tp) {
            const uint64_t key = bi == INT_MAX ? 0ull : candidate_key(best, bi);
            __hip_atomic_store(static_cast<uint64_t *>(p.recv[lane]) + record_slot(parity, k, p.rank, blockIdx.x),
                               (key << TAG_BITS) | tag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
    if (threadIdx.x == 0) {
        p.counts[blockIdx.x] = count + 1;
    }
    if (blockIdx.x != 0) {
        return;
    }
    // every rank's blocks' records of every key row, POLLS a thread in flight until all carry the tag
    uint64_t      *mine  = static_cast<uint64_t *>(p.recv[p.rank]);
    const int       total = p.keys * p.tp * p.blocks;
    for (int base = 0; base < total; base += THREADS * POLLS) {
        uint32_t pending = 0;
        int64_t  slot[POLLS];
#pragma unroll
        for (int j = 0; j < POLLS; ++j) {
            const int i = base + j * THREADS + threadIdx.x;
            if (i < total) {
                const int b = i % p.blocks, src = (i / p.blocks) % p.tp, k = i / (p.blocks * p.tp);
                slot[j] = record_slot(parity, k, src, b);
                pending |= 1u << j;
            }
        }
        uint64_t t0 = 0;
        for (int n = 1; pending; ++n) {
            uint64_t v[POLLS];
#pragma unroll
            for (int j = 0; j < POLLS; ++j) {
                if ((pending >> j) & 1u) {
                    v[j] = __hip_atomic_load(mine + slot[j], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                }
            }
#pragma unroll
            for (int j = 0; j < POLLS; ++j) {
                if (((pending >> j) & 1u) && (v[j] & TAG_MASK) == tag) {
                    const int k = (base + j * THREADS + threadIdx.x) / (p.blocks * p.tp);
                    atomicMax(reinterpret_cast<unsigned long long *>(&s_top[k]),
                              static_cast<unsigned long long>(v[j] >> TAG_BITS));
                    pending &= ~(1u << j);
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
    }
    __syncthreads();
    if (p.target_out != nullptr) {
        for (int k = threadIdx.x; k < p.keys; k += THREADS) {
            p.target_out[k] = key_index(s_top[k]);
        }
    }
    // static_accept_tail.py, a request a thread
    const int r = threadIdx.x;
    if (r >= p.bs) {
        return;
    }
    const int      gamma = p.T - 1;
    const int64_t *cand  = p.candidates + r * p.cand_stride;
    int            correct;
    int64_t        bonus;
    if (p.sim) {
        correct = static_cast<int>(s_sim);
        bonus   = key_index(s_top[r]);
    } else {
        correct = gamma;
        for (int j = gamma - 1; j >= 0; --j) {
            if (cand[1 + j] != key_index(s_top[r * p.T + j])) {
                correct = j;
            }
        }
        bonus = key_index(s_top[r * p.T + correct]);
    }
    const int commit    = correct + 1;
    p.correct_out[r]    = correct;
    p.bonus_out[r]      = bonus;
    p.cap_trim_out[r]   = 0;
    p.commit_out[r]     = commit;
    p.new_seq_out[r]    = p.seq_lens[r] + commit;
    int64_t *tokens     = p.out_tokens + r * p.out_tokens_stride;
    for (int j = 0; j < p.T; ++j) {
        tokens[j] = j == correct ? bonus : (j < gamma ? cand[1 + j] : 0);
    }
    if (p.gated_out != nullptr) {
        p.gated_out[r] = min(commit, static_cast<int>(p.verify_lens[r])) * p.gate[0];
    }
}

__global__ __launch_bounds__(THREADS) void verify_tail_kernel(const Params p) { verify_tail_body(p); }

// the accept's blocks first, then the draft's main_proj CTAs (main_proj.h), whose input rows are ready before the
// head: its exchange waits overlap the accept's
static_assert(THREADS == main_proj::THREADS);
__global__ __launch_bounds__(THREADS) void verify_tail_proj_kernel(const Params p, const main_proj::Params q) {
    if (static_cast<int>(blockIdx.x) < p.blocks) {
        verify_tail_body(p);
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

// local [bs * T, part] bf16 (16 B aligned rows; the first width columns valid), candidates [bs, T] int64; the
// outputs as static_accept_tail's (rows [:bs]); sim [1] int64 with temps [bs] f32 and seed [1] int32, or none;
// counts [blocks] int32 (zeroed once), peers [tp] int64 on the host: every rank's zeroed records
Params make_params(const at::Tensor &local, const at::Tensor &candidates, const at::Tensor &seq_lens,
                   const at::Tensor &verify_lens, const at::Tensor &gate, const c10::optional<at::Tensor> &sim,
                   const c10::optional<at::Tensor> &temps, const c10::optional<at::Tensor> &seed,
                   at::Tensor &correct_out, at::Tensor &bonus_out, at::Tensor &cap_trim_out, at::Tensor &commit_out,
                   at::Tensor &new_seq_out, at::Tensor &out_tokens, const c10::optional<at::Tensor> &gated_out,
                   const c10::optional<at::Tensor> &target_out, at::Tensor &counts, const at::Tensor &peers,
                   int64_t rank, int64_t vocab_start, int64_t width, int64_t vocab, int64_t chunk, double deadline_s) {
    const int64_t bs = candidates.size(0), T = candidates.size(1), tp = peers.numel();
    // the blocks tile the shard's columns, the same on every rank; the last rank's may run past the vocab
    const int64_t blocks = (local.size(1) + chunk - 1) / chunk;
    const int64_t keys   = sim.has_value() ? bs : bs * T;
    check(local, local, at::kBFloat16, "local");
    TORCH_CHECK(local.dim() == 2 && local.size(0) == bs * T && local.size(1) >= width && local.size(1) % 8 == 0 &&
                    local.stride(0) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(local.data_ptr()) % 16 == 0,
                "local [bs * T, part] bf16, 16 B aligned rows");
    TORCH_CHECK(bs >= 1 && T >= 2 && T <= MAX_T && bs * T <= MAX_ROWS && bs <= THREADS, "bs * T at most ",
                MAX_ROWS, " rows, T at most ", MAX_T);
    TORCH_CHECK(chunk % 8 == 0 && chunk > 0 && blocks <= MAX_BLOCKS, "chunk a multiple of 8, at most ", MAX_BLOCKS,
                " blocks");
    TORCH_CHECK(vocab < (1 << IDX_BITS) && vocab_start + width <= vocab, "the vocab below 2^17");
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
    if (sim.has_value()) {
        check(*sim, local, at::kLong, "sim");
        TORCH_CHECK(temps.has_value() && seed.has_value(), "a simulated length needs temps and seed");
        check(*temps, local, at::kFloat, "temps");
        check(*seed, local, at::kInt, "seed");
    }
    if (gated_out.has_value()) {
        check(*gated_out, local, at::kInt, "gated_out");
    }
    if (target_out.has_value()) {
        check(*target_out, local, at::kLong, "target_out");
        TORCH_CHECK(target_out->numel() >= keys, "target_out [key rows]");
    }
    check(counts, local, at::kInt, "counts");
    TORCH_CHECK(counts.numel() >= blocks, "counts [blocks] int32");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong && tp >= 1 && tp <= MAX_TP &&
                    rank >= 0 && rank < tp,
                "peers: every rank's records (int64, host)");
    Params p{};
    p.local             = static_cast<const bfloat16 *>(local.data_ptr());
    p.local_stride      = local.stride(0);
    p.width             = static_cast<int>(width);
    p.vocab_start       = static_cast<int>(vocab_start);
    p.vocab             = static_cast<int>(vocab);
    p.chunk             = static_cast<int>(chunk);
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
    for (int r = 0; r < p.tp; ++r) {
        p.recv[r] = reinterpret_cast<void *>(peers.data_ptr<int64_t>()[r]);
    }
    return p;
}

void verify_tail(const at::Tensor &local, const at::Tensor &candidates, const at::Tensor &seq_lens,
                 const at::Tensor &verify_lens, const at::Tensor &gate, const c10::optional<at::Tensor> &sim,
                 const c10::optional<at::Tensor> &temps, const c10::optional<at::Tensor> &seed,
                 at::Tensor &correct_out, at::Tensor &bonus_out, at::Tensor &cap_trim_out, at::Tensor &commit_out,
                 at::Tensor &new_seq_out, at::Tensor &out_tokens, const c10::optional<at::Tensor> &gated_out,
                 const c10::optional<at::Tensor> &target_out, at::Tensor &counts, const at::Tensor &peers,
                 int64_t rank, int64_t vocab_start, int64_t width, int64_t vocab, int64_t chunk, double deadline_s) {
    const Params p = make_params(local, candidates, seq_lens, verify_lens, gate, sim, temps, seed, correct_out,
                                 bonus_out, cap_trim_out, commit_out, new_seq_out, out_tokens, gated_out, target_out,
                                 counts, peers, rank, vocab_start, width, vocab, chunk, deadline_s);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(local.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(local.device().index()).stream();
    verify_tail_kernel<<<p.blocks, THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

// verify_tail's accept and, in the same launch, main_proj of x [rows, 15360] bf16 on this rank's columns w, ws into
// proj_out (main_proj.h's arguments: proj_state, proj_work and proj_peers its exchange's)
void verify_tail_proj(const at::Tensor &local, const at::Tensor &candidates, const at::Tensor &seq_lens,
                      const at::Tensor &verify_lens, const at::Tensor &gate, const c10::optional<at::Tensor> &sim,
                      const c10::optional<at::Tensor> &temps, const c10::optional<at::Tensor> &seed,
                      at::Tensor &correct_out, at::Tensor &bonus_out, at::Tensor &cap_trim_out,
                      at::Tensor &commit_out, at::Tensor &new_seq_out, at::Tensor &out_tokens,
                      const c10::optional<at::Tensor> &gated_out, const c10::optional<at::Tensor> &target_out,
                      at::Tensor &counts, const at::Tensor &peers, int64_t rank, int64_t vocab_start, int64_t width,
                      int64_t vocab, int64_t chunk, double deadline_s, const at::Tensor &x, const at::Tensor &w,
                      const at::Tensor &ws, at::Tensor &proj_out, at::Tensor &proj_state, at::Tensor &proj_work,
                      const at::Tensor &proj_peers, double proj_deadline_s) {
    const Params p = make_params(local, candidates, seq_lens, verify_lens, gate, sim, temps, seed, correct_out,
                                 bonus_out, cap_trim_out, commit_out, new_seq_out, out_tokens, gated_out, target_out,
                                 counts, peers, rank, vocab_start, width, vocab, chunk, deadline_s);
    TORCH_CHECK(x.device() == local.device() && proj_peers.numel() == peers.numel(), "main_proj on the same group");
    const main_proj::Params q =
        main_proj::params(x, w, ws, proj_out, proj_state, proj_work, proj_peers, rank, proj_deadline_s);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(local.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(local.device().index()).stream();
    verify_tail_proj_kernel<<<p.blocks + main_proj::grid(q), THREADS, 0, stream>>>(p, q);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

__global__ void noisy_kernel(const float *values, float *out, int n, float t, int32_t seed, int start) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = values[i] / fmaxf(t, 1e-5f) - logf(-logf(uniform(seed, start + i)));
    }
}

// the kernel's noisy value of values[i] at offset start + i (a check against SGLang's Triton, which keys it the
// same way): out [n] f32
void noisy_values(const at::Tensor &values, at::Tensor &out, double t, int64_t seed, int64_t start) {
    check(values, values, at::kFloat, "values");
    check(out, values, at::kFloat, "out");
    TORCH_CHECK(values.dim() == 1 && out.numel() >= values.numel(), "values [n], out [n]");
    const int n = static_cast<int>(values.numel());
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(values.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(values.device().index()).stream();
    noisy_kernel<<<(n + THREADS - 1) / THREADS, THREADS, 0, stream>>>(values.data_ptr<float>(), out.data_ptr<float>(),
                                                                       n, static_cast<float>(t),
                                                                       static_cast<int32_t>(seed),
                                                                       static_cast<int>(start));
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("verify_tail(Tensor local, Tensor candidates, Tensor seq_lens, Tensor verify_lens, Tensor gate, "
          "Tensor? sim, Tensor? temps, Tensor? seed, Tensor(a!) correct_out, Tensor(b!) bonus_out, "
          "Tensor(c!) cap_trim_out, Tensor(d!) commit_out, Tensor(e!) new_seq_out, Tensor(f!) out_tokens, "
          "Tensor(g!)? gated_out, Tensor(h!)? target_out, Tensor(i!) counts, Tensor peers, int rank, "
          "int vocab_start, int width, int vocab, int chunk, float deadline_s) -> ()");
    m.def("verify_tail_proj(Tensor local, Tensor candidates, Tensor seq_lens, Tensor verify_lens, Tensor gate, "
          "Tensor? sim, Tensor? temps, Tensor? seed, Tensor(a!) correct_out, Tensor(b!) bonus_out, "
          "Tensor(c!) cap_trim_out, Tensor(d!) commit_out, Tensor(e!) new_seq_out, Tensor(f!) out_tokens, "
          "Tensor(g!)? gated_out, Tensor(h!)? target_out, Tensor(i!) counts, Tensor peers, int rank, "
          "int vocab_start, int width, int vocab, int chunk, float deadline_s, Tensor x, Tensor w, Tensor ws, "
          "Tensor(j!) proj_out, Tensor(k!) proj_state, Tensor(l!) proj_work, Tensor proj_peers, "
          "float proj_deadline_s) -> ()");
    m.def("noisy_values(Tensor values, Tensor(a!) out, float t, int seed, int start) -> ()");
    m.impl("verify_tail", at::kCUDA, &verify_tail);
    m.impl("verify_tail_proj", at::kCUDA, &verify_tail_proj);
    m.impl("noisy_values", at::kCUDA, &noisy_values);
}
