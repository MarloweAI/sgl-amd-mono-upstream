// SPDX-License-Identifier: Apache-2.0
// A verify step's Engram rows for every Engram layer in one launch: SGLang's n-gram hash (``_engram_hash_kernel``,
// MODE_VERIFY), the row gather of each layer's FP8 table (``_engram_gather_kernel``) and the TP all-reduce that
// reassembles the row-sharded tables, bitwise the three. A CTA owns one (layer, token, hash column) row: it hashes
// the token's window, and the rank whose shard holds the row gathers it and pushes its bf16 pairs, tagged with the
// CTA's launch count, into every rank's exchange buffer; every other rank's CTA polls its row's pairs and copies
// them out. Zeros come out +0, as the all-reduce's sum of the owner's row and the other ranks' zeros does.
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

constexpr int WAVE = 64;
constexpr int N = 4, H = 8, COLS = (N - 1) * H; // n-gram sizes 2..N, hash heads
constexpr int DIM = 256, BLK = 32;               // a row's FP8 values and their E8M0 block
constexpr int PAIRS = DIM / 2;                   // bf16 pairs a row: two a lane
constexpr int MAX_LAYERS = 2, MAX_TOKENS = 48, MAX_TP = 8;
constexpr int MAX_CTAS = MAX_LAYERS * MAX_TOKENS * COLS;
// the exchange buffer a rank: [parity][CTA][pair] tagged pairs
constexpr int64_t EXCHANGE_BYTES = int64_t{2} * MAX_CTAS * PAIRS * 8;
static_assert(DIM == 4 * WAVE && DIM % BLK == 0);

struct Params {
    const void *ids;  // [T] the verify ids
    const void *pos;  // [T] their positions
    const void *slots; // [T / block] the requests' history rows
    int ids_bytes, pos_bytes, slots_bytes;
    const int32_t *hist;     // [slots, N - 1] oldest first
    const int64_t *token_map; // compressed ids
    const int64_t *mult;     // [hash layers, N]
    const int64_t *primes;   // [hash layers, N - 1, H]
    const int64_t *offsets;  // [hash layers, COLS]
    int64_t pad_id, image_token_id, mm_pad_shift; // image_token_id < 0: none
    const uint8_t *w[MAX_LAYERS]; // this rank's E4M3 rows of each layer's table
    const uint8_t *s[MAX_LAYERS]; // their E8M0 scales
    int64_t row_lo[MAX_LAYERS], row_hi[MAX_LAYERS];
    int hash_layer[MAX_LAYERS];
    bfloat16 *out; // [layers][out_tokens][COLS][DIM]
    int64_t out_tokens;
    int *state; // every CTA's launch count (zeroed once)
    uint64_t *recv[MAX_TP];
    uint64_t deadline;
    int tokens, block, rank, tp;
};

__device__ inline int64_t load_index(const void *p, int bytes, int64_t i) {
    return bytes == 8 ? static_cast<const int64_t *>(p)[i] : static_cast<const int32_t *>(p)[i];
}

__device__ inline uint64_t tagged(uint32_t v, int count) {
    return static_cast<uint64_t>(v) | static_cast<uint64_t>(static_cast<uint32_t>(count)) << 32;
}

// E4M3 (OCP, no infinities) to f32, exactly
__device__ inline float fp8_value(uint32_t b) {
    const uint32_t sign = (b & 0x80u) << 24, e = (b >> 3) & 0xfu, m = b & 0x7u;
    if (e == 0) return __uint_as_float(sign | __float_as_uint(static_cast<float>(m) * 0.001953125f)); // m * 2^-9
    if (e == 15 && m == 7) return __uint_as_float(0x7fc00000u);
    return __uint_as_float(sign | (e + 120u) << 23 | m << 20);
}

// a bf16's bits, -0 as +0
__device__ inline uint32_t bf16_bits(float v) {
    const uint32_t b = __builtin_bit_cast(uint16_t, static_cast<bfloat16>(v));
    return b == 0x8000u ? 0u : b;
}

__global__ __launch_bounds__(WAVE) void engram_rows_kernel(const Params p) {
    const int b = blockIdx.x, lane = threadIdx.x;
    const int per_layer = p.tokens * COLS;
    const int k = b / per_layer, t = (b % per_layer) / COLS, c = b % COLS;
    int count = 0;
    if (lane == 0) count = __hip_atomic_fetch_add(p.state + b, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1;
    count = __builtin_amdgcn_readfirstlane(count);

    // the token's window: shift s is an earlier token of its block when s <= its offset, else the history
    const int64_t r = t / p.block;
    const int off = t - static_cast<int>(r) * p.block;
    const int64_t hrow = load_index(p.slots, p.slots_bytes, r);
    const int64_t pos = load_index(p.pos, p.pos_bytes, t);
    const int l = p.hash_layer[k], gram = c / H + 1;
    int64_t rolling = 0;
    bool blocked = false;
#pragma unroll
    for (int s = 0; s < N; ++s) {
        int64_t tok;
        if (s <= off) {
            tok = load_index(p.ids, p.ids_bytes, t - s > 0 ? t - s : 0);
        } else {
            int hcol = N - 2 - (s - off - 1);
            hcol = hcol < 0 ? 0 : (hcol > N - 2 ? N - 2 : hcol);
            tok = p.hist[hrow * (N - 1) + hcol];
        }
        bool blk = pos < s;
        if (p.image_token_id >= 0) {
            if (tok >= p.mm_pad_shift) tok = p.image_token_id;
            blk = blk || tok == p.image_token_id;
        }
        blocked = blocked || blk;
        if (s <= gram) {
            const int64_t comp = blocked ? p.pad_id : p.token_map[tok];
            rolling ^= comp * p.mult[l * N + s];
        }
    }
    const int64_t id = rolling % p.primes[l * COLS + c] + p.offsets[l * COLS + c];

    const int parity = count & 1;
    uint32_t *out = reinterpret_cast<uint32_t *>(p.out + ((k * p.out_tokens + t) * COLS + c) * DIM);
    const int64_t at = (static_cast<int64_t>(parity) * MAX_CTAS + b) * PAIRS + 2 * lane;
    if (p.row_lo[k] <= id && id < p.row_hi[k]) {
        // the owner: the row's 4 values a lane, times its block's 2^(e - 127), pushed to every other rank
        const int64_t row = id - p.row_lo[k];
        const uint32_t v = *reinterpret_cast<const uint32_t *>(p.w[k] + row * DIM + 4 * lane);
        const int e = p.s[k][row * (DIM / BLK) + (4 * lane) / BLK];
        const float scale = e == 0 ? __uint_as_float(0x00400000u) : __uint_as_float(static_cast<uint32_t>(e) << 23);
        uint32_t pk[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float lo = fp8_value((v >> (16 * i)) & 0xffu) * scale;
            const float hi = fp8_value((v >> (16 * i + 8)) & 0xffu) * scale;
            pk[i] = bf16_bits(lo) | bf16_bits(hi) << 16;
        }
        for (int dst = 0; dst < p.tp; ++dst) {
            if (dst == p.rank) continue;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                __hip_atomic_store(p.recv[dst] + at + i, tagged(pk[i], count), __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
        out[lane * 2] = pk[0];
        out[lane * 2 + 1] = pk[1];
        return;
    }
    const uint64_t *mine = p.recv[p.rank] + at;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        uint64_t v;
        uint64_t t0 = 0;
        for (int n = 1;; ++n) {
            v = __hip_atomic_load(mine + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            if (static_cast<int>(v >> 32) == count) break;
            __builtin_amdgcn_s_sleep(1);
            if (n % 1024 == 0) { // a wait past the deadline traps, stopping this rank's step
                const uint64_t now = __builtin_amdgcn_s_memrealtime();
                if (t0 == 0) t0 = now;
                else if (now - t0 > p.deadline) __builtin_trap();
            }
        }
        out[lane * 2 + i] = static_cast<uint32_t>(v);
    }
}

int64_t index_bytes(const at::Tensor &t) {
    TORCH_CHECK(t.is_cuda() && t.dim() == 1 && t.stride(0) == 1 &&
                (t.scalar_type() == at::kLong || t.scalar_type() == at::kInt));
    return t.element_size();
}

// ids, pos [T] and slots [T / block] int32 or int64; hist [slots, N - 1] int32; token_map, mult [L, N], primes
// [L, N - 1, H], offsets [L, COLS] int64 (SGLang's EngramHasher buffers); per Engram layer its hash layer, table
// and scale addresses and global rows [row_lo, row_hi); out [layers, >= T, COLS, DIM] bf16; state int32 (zeroed
// once); peers: every rank's exchange buffer (int64, host), zeroed once
void engram_rows_op(const at::Tensor &ids, const at::Tensor &pos, const at::Tensor &slots, const at::Tensor &hist,
                    const at::Tensor &token_map, const at::Tensor &mult, const at::Tensor &primes,
                    const at::Tensor &offsets, at::IntArrayRef hash_layer, at::IntArrayRef w, at::IntArrayRef s,
                    at::IntArrayRef row_lo, at::IntArrayRef row_hi, int64_t block, int64_t pad_id,
                    int64_t image_token_id, int64_t mm_pad_shift, at::Tensor &out, at::Tensor &state,
                    const at::Tensor &peers, int64_t rank, double deadline_s) {
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(ids.device());
    const int64_t tokens = ids.numel(), layers = static_cast<int64_t>(hash_layer.size()), tp = peers.numel();
    TORCH_CHECK(1 <= tokens && tokens <= MAX_TOKENS && block >= 1 && tokens % block == 0);
    TORCH_CHECK(1 <= layers && layers <= MAX_LAYERS && static_cast<int64_t>(w.size()) == layers &&
                static_cast<int64_t>(s.size()) == layers && static_cast<int64_t>(row_lo.size()) == layers &&
                static_cast<int64_t>(row_hi.size()) == layers);
    TORCH_CHECK(pos.numel() == tokens && slots.numel() == tokens / block);
    TORCH_CHECK(hist.scalar_type() == at::kInt && hist.dim() == 2 && hist.size(1) == N - 1 && hist.is_contiguous());
    const int64_t hl = mult.size(0);
    TORCH_CHECK(token_map.scalar_type() == at::kLong && token_map.is_contiguous());
    TORCH_CHECK(mult.scalar_type() == at::kLong && mult.is_contiguous() && mult.dim() == 2 && mult.size(1) == N);
    TORCH_CHECK(primes.scalar_type() == at::kLong && primes.is_contiguous() && primes.numel() == hl * COLS);
    TORCH_CHECK(offsets.scalar_type() == at::kLong && offsets.is_contiguous() && offsets.numel() == hl * COLS);
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.dim() == 4 && out.size(0) >= layers &&
                out.size(1) >= tokens && out.size(2) == COLS && out.size(3) == DIM);
    TORCH_CHECK(state.scalar_type() == at::kInt && state.is_contiguous() && state.numel() >= MAX_CTAS);
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong && 1 <= tp && tp <= MAX_TP);
    TORCH_CHECK(0 <= rank && rank < tp);
    Params p{};
    p.ids = ids.data_ptr();
    p.pos = pos.data_ptr();
    p.slots = slots.data_ptr();
    p.ids_bytes = static_cast<int>(index_bytes(ids));
    p.pos_bytes = static_cast<int>(index_bytes(pos));
    p.slots_bytes = static_cast<int>(index_bytes(slots));
    p.hist = hist.data_ptr<int32_t>();
    p.token_map = token_map.data_ptr<int64_t>();
    p.mult = mult.data_ptr<int64_t>();
    p.primes = primes.data_ptr<int64_t>();
    p.offsets = offsets.data_ptr<int64_t>();
    p.pad_id = pad_id;
    p.image_token_id = image_token_id;
    p.mm_pad_shift = mm_pad_shift;
    for (int64_t k = 0; k < layers; ++k) {
        TORCH_CHECK(0 <= hash_layer[k] && hash_layer[k] < hl);
        p.hash_layer[k] = static_cast<int>(hash_layer[k]);
        p.w[k] = reinterpret_cast<const uint8_t *>(w[k]);
        p.s[k] = reinterpret_cast<const uint8_t *>(s[k]);
        p.row_lo[k] = row_lo[k];
        p.row_hi[k] = row_hi[k];
    }
    p.out = static_cast<bfloat16 *>(out.data_ptr());
    p.out_tokens = out.size(1);
    p.state = state.data_ptr<int>();
    for (int64_t i = 0; i < tp; ++i) p.recv[i] = reinterpret_cast<uint64_t *>(peers.data_ptr<int64_t>()[i]);
    p.deadline = static_cast<uint64_t>(deadline_s * 1e8); // s_memrealtime: 100 MHz
    p.tokens = static_cast<int>(tokens);
    p.block = static_cast<int>(block);
    p.rank = static_cast<int>(rank);
    p.tp = static_cast<int>(tp);
    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(ids.device().index()).stream();
    hipLaunchKernelGGL(engram_rows_kernel, dim3(static_cast<unsigned>(layers * tokens * COLS)), dim3(WAVE), 0, stream,
                       p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("engram_rows(Tensor ids, Tensor pos, Tensor slots, Tensor hist, Tensor token_map, Tensor mult, "
          "Tensor primes, Tensor offsets, int[] hash_layer, int[] w, int[] s, int[] row_lo, int[] row_hi, int block, "
          "int pad_id, int image_token_id, int mm_pad_shift, Tensor(a!) out, Tensor(b!) state, Tensor peers, "
          "int rank, float deadline_s) -> ()");
    m.impl("engram_rows", at::kCUDA, &engram_rows_op);
}
