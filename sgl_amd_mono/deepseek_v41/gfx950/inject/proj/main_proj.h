// SPDX-License-Identifier: Apache-2.0
// main_proj.h: the device code and launch parameters of main_proj.hip, for any launch that runs its CTAs.
// The DSpark draft's main_proj (K 15360 -> 5120, E4M3 weights in 32 x 32 E8M0 blocks) split over the TP ranks by
// output column, in one launch: each rank quantizes the rows as SGLang's _mxfp8_quant_kernel does, runs its
// 5120 / tp columns with aiter's packed afp8wfp8 arithmetic (BLOCK_M 8, BLOCK_N 16, BLOCK_K 256, K_PACK 2, 3 fused
// splits of K 5120), and pushes its bf16 outputs to every rank; every rank then holds the whole rows, bitwise the
// replicated GEMM's.
//
// aiter's chain for an output (row, col): per split, two accumulators, one a K panel (the first and the second 256
// of every 512), each 10 steps of 2 chained 16x16x128 scaled MFMAs (K 0..127, then 128..255 of the panel); the
// panels added, the splits added to zero in split order, rounded to bf16. Here a CTA runs one (16-column tile,
// split, panel) chain on wave 0, the same MFMA operands lane for lane, while waves 1-3 quantize its K slice of the
// rows into LDS. The tile's first CTA sums the six partials in aiter's order and pushes the bf16 pairs, tagged with
// the launch count, into every rank's exchange buffer; every CTA then copies its share of the incoming pairs into
// the rows. Hand-offs are tagged 8-byte pairs (agent scope on the rank, system scope across): no fence, counter
// reset or barrier launch.
#pragma once

#include <cstdint>

#include <ATen/ATen.h>
#include <hip/hip_runtime.h>

namespace main_proj {

using bfloat16 = __bf16;
using uint32x4 = uint32_t __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef short v2s __attribute__((ext_vector_type(2)));

constexpr int THREADS = 256, WAVE = 64, QWAVES = 3;
constexpr int H = 5120, K = 3 * H, GROUPS = K / 32; // main_proj's output and input widths
constexpr int MAX_ROWS = 8, MAX_TP = 8;
constexpr int TILE_N = 16, SPLITS = 3, SPLIT_K = 5120, STEP = 512, PANEL = 256, PANELS = 2;
constexpr int STEPS = SPLIT_K / STEP, CHUNKS = PANEL / 128, CHAIN = STEPS * CHUNKS; // 10 steps, 20 MFMAs
constexpr int SLICE = STEPS * PANEL;                 // a chain's K values of a row: 2560
constexpr int SLICE_CHUNKS = SLICE / 8;              // 8-value chunks a row: 320
constexpr int UNITS = SPLITS * PANELS;               // CTAs a tile
constexpr int PAIRS = H / 2;                         // bf16 pairs a row
// the exchange buffer a rank: [parity][row][pair] tagged pairs
constexpr int64_t EXCHANGE_BYTES = int64_t{2} * MAX_ROWS * PAIRS * 8;
// the state ints (zeroed once): every CTA's launch count
constexpr int MAX_CTAS = (H / TILE_N) * UNITS, STATE_INTS = 2048;
static_assert(MAX_CTAS <= STATE_INTS);
// the work bytes: the chains' partials, tagged pairs (a lane's 4 floats each)
constexpr int64_t WORK_BYTES = int64_t{MAX_CTAS} * WAVE * 4 * 8;

#ifdef MAIN_PROJ_STAMPS // a timing build (tests only): thread 0's realtime ticks per CTA, past the state ints
#define MAIN_PROJ_STAMP(k) if (tid == 0) reinterpret_cast<uint64_t *>(p.state + STATE_INTS)[b * 8 + (k)] = __builtin_amdgcn_s_memrealtime()
#else
#define MAIN_PROJ_STAMP(k)
#endif

struct Params {
    const bfloat16 *x;   // [rows, K] the target hidden rows
    int64_t x_stride;
    const uint8_t *w;    // this rank's [H / tp, K] E4M3 columns
    const uint8_t *ws;   // their E8M0 scales [H / tp / 32, K / 32]
    bfloat16 *out;       // [rows, H]
    int64_t out_stride;
    int *state;
    uint64_t *part;
    uint64_t *recv[MAX_TP]; // every rank's exchange buffer
    uint64_t deadline;
    int rows, rank, tp, tiles;
};

template <int CTRL> __device__ inline float dpp(float v) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), CTRL, 0xf, 0xf, false));
}

// ---------------------------------------------------------------- the MXFP8 quant (SGLang's _mxfp8_quant_kernel)
// a lane's 8 values of a 32-value group held by the 4 lanes of a quad: their E4M3 bytes; the group's E8M0 scale
__device__ inline uint64_t quant8(const float (&x)[8], uint8_t &scale) {
#pragma float_control(precise, on)
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) amax = __builtin_fmaxf(amax, __builtin_fabsf(x[i]));
    amax = __builtin_fmaxf(amax, dpp<0xb1>(amax));
    amax = __builtin_fmaxf(amax, dpp<0x4e>(amax));
    amax = __builtin_fmaxf(amax, 1e-30f);
    float sb = __builtin_ceilf(__builtin_log2f(amax / 448.0f)) + 127.0f;
    sb = __builtin_fminf(__builtin_fmaxf(sb, 0.0f), 254.0f);
    // x / 2^(sb - 127) as x * 2^(127 - sb): both the correctly rounded value of the same real number
    const int e = 254 - static_cast<int>(sb); // the reciprocal's biased exponent, 0 (2^-127, subnormal) to 254
    const float inv = e > 0 ? __uint_as_float(static_cast<uint32_t>(e) << 23) : __uint_as_float(0x00400000u);
    v2s w[2];
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
        const float a = __builtin_amdgcn_fmed3f(x[i] * inv, -448.0f, 448.0f);
        const float b = __builtin_amdgcn_fmed3f(x[i + 1] * inv, -448.0f, 448.0f);
        const float c = __builtin_amdgcn_fmed3f(x[i + 2] * inv, -448.0f, 448.0f);
        const float d = __builtin_amdgcn_fmed3f(x[i + 3] * inv, -448.0f, 448.0f);
        v2s v = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(v2s{0, 0}, a, b, 1.0f, false);
        w[i / 4] = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(v, c, d, 1.0f, true);
    }
    scale = static_cast<uint8_t>(sb);
    return static_cast<uint64_t>(__builtin_bit_cast(uint32_t, w[0])) | static_cast<uint64_t>(__builtin_bit_cast(uint32_t, w[1])) << 32;
}

// ---------------------------------------------------------------- the operands (aiter's lane layout)
// a lane's 32 bytes of a row for one 128-K MFMA: K (lane / 16) * 16 .. + 16 and 64 + (lane / 16) * 16 .. + 16
__device__ inline v8i load_lds(const uint8_t *row, int k0, int lane) {
    const int g = lane / 16;
    const uint4 lo = *reinterpret_cast<const uint4 *>(row + k0 + g * 16);
    const uint4 hi = *reinterpret_cast<const uint4 *>(row + k0 + 64 + g * 16);
    v8i v;
    v[0] = lo.x, v[1] = lo.y, v[2] = lo.z, v[3] = lo.w, v[4] = hi.x, v[5] = hi.y, v[6] = hi.z, v[7] = hi.w;
    return v;
}

// the weights: plain loads (a lane's 16-byte pieces are half lines; a non-temporal half line refetches its line)
__device__ inline v8i load_weights(const uint8_t *row, int k0, int lane) { return load_lds(row, k0, lane); }

__device__ inline uint64_t tagged(uint32_t v, int count) {
    return static_cast<uint64_t>(v) | static_cast<uint64_t>(static_cast<uint32_t>(count)) << 32;
}

__device__ inline bool tag_is(uint64_t pair, int count) { return static_cast<int>(pair >> 32) == count; }

// a wait past the deadline traps, stopping this rank's step
__device__ inline void check_deadline(const Params &p, int n, uint64_t &t0) {
    if (n % 1024 == 0) {
        const uint64_t now = __builtin_amdgcn_s_memrealtime();
        if (t0 == 0) {
            t0 = now;
        } else if (now - t0 > p.deadline) {
            __builtin_trap();
        }
    }
}

// the K slice of the rows (waves 1-3): chunk c of a row is the chain's K values c * 8 .. + 8, at global K
// split * SPLIT_K + (c / 32) * STEP + panel * PANEL + (c % 32) * 8; a thread's chunks i = j * 192 + t
constexpr int PER = QWAVES * WAVE, BATCH = (MAX_ROWS * SLICE_CHUNKS + PER - 1) / PER;

__device__ inline void load_slice(const Params &p, int tid, int split, int panel, uint32x4 (&v)[BATCH]) {
    const int t = tid - WAVE, total = p.rows * SLICE_CHUNKS;
#pragma unroll
    for (int j = 0; j < BATCH; ++j) {
        const int i = j * PER + t;
        if (i < total) {
            const int row = i / SLICE_CHUNKS, c = i % SLICE_CHUNKS;
            const int k = split * SPLIT_K + (c / 32) * STEP + panel * PANEL + (c % 32) * 8;
            v[j] = *reinterpret_cast<const uint32x4 *>(p.x + static_cast<int64_t>(row) * p.x_stride + k);
        }
    }
}

// the loaded chunks quantized into xq, xs (LDS)
__device__ inline void quant_slice(const Params &p, int tid, const uint32x4 (&v)[BATCH], uint8_t (*xq)[SLICE],
                                   uint8_t (*xs)[SLICE / 32]) {
    const int t = tid - WAVE, total = p.rows * SLICE_CHUNKS;
#pragma unroll
    for (int j = 0; j < BATCH; ++j) {
        const int i = j * PER + t; // quad-uniform: a quad's 4 chunks are one row's group
        if (i < total) {
            const int row = i / SLICE_CHUNKS, c = i % SLICE_CHUNKS;
            const bfloat16 *b = reinterpret_cast<const bfloat16 *>(&v[j]);
            float y[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) y[k] = static_cast<float>(b[k]);
            uint8_t scale;
            *reinterpret_cast<uint64_t *>(&xq[row][c * 8]) = quant8(y, scale);
            if (c % 4 == 0) xs[row][c / 4] = scale;
        }
    }
}

// a CTA's LDS: the rows' quantized K slice and its scales, the CTA's launch count
constexpr int LDS_BYTES = MAX_ROWS * SLICE + MAX_ROWS * (SLICE / 32) + 16;

// CTA b of the grid CTAs (b < tiles * UNITS) on THREADS threads of the caller's block, tid this thread's index among
// them, its LDS xq (16 B aligned), xs and slot; any launch may run it, the grid co-resident. Every thread of the block
// reaches its two barriers: a block of 2 x THREADS runs two CTAs side by side
__device__ inline void run(const Params &p, const int b, const int grid, const int tid, uint8_t (*xq)[SLICE],
                           uint8_t (*xs)[SLICE / 32], int &slot) {
    const int wave = tid / WAVE, lane = tid % WAVE;
    const int tile = b / UNITS, unit = b % UNITS, split = unit / PANELS, panel = unit % PANELS;
    const int n0 = tile * TILE_N, r = lane % 16, g = lane / 16;
    if (tid == 0) slot = __hip_atomic_fetch_add(p.state + b, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1;
    MAIN_PROJ_STAMP(0);
    v8i wa[CHAIN];
    int wsc[CHAIN];
    uint32x4 xv[BATCH];
    // the rows' slice first (waves 1-3), then the weights (wave 0): the slice's loads are ahead in the CU's queue
    if (wave != 0) load_slice(p, tid, split, panel, xv);
    __builtin_amdgcn_s_barrier();
    if (wave == 0) { // the chain's weights and scales, all in flight
        const uint8_t *wrow = p.w + static_cast<int64_t>(n0 + r) * K;
        const uint8_t *srow = p.ws + ((n0 + r) / 32) * GROUPS + g;
#pragma unroll
        for (int i = 0; i < CHAIN; ++i) {
            const int k0 = split * SPLIT_K + (i / CHUNKS) * STEP + panel * PANEL + (i % CHUNKS) * 128;
            wa[i] = load_weights(wrow, k0, lane);
            wsc[i] = srow[k0 / 32];
        }
    } else {
        quant_slice(p, tid, xv, xq, xs);
#ifdef MAIN_PROJ_STAMPS
        if (tid == WAVE) reinterpret_cast<uint64_t *>(p.state + STATE_INTS)[b * 8 + 6] = __builtin_amdgcn_s_memrealtime();
#endif
    }
    __syncthreads();
    const int count = slot, parity = count & 1;
    MAIN_PROJ_STAMP(1);
    uint64_t *part = p.part + static_cast<int64_t>(tile) * UNITS * WAVE * 4;
    if (wave == 0) {
        const int m = min(r, p.rows - 1);
        v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int i = 0; i < CHAIN; ++i) {
            const int kl = (i / CHUNKS) * PANEL + (i % CHUNKS) * 128;
            const v8i xa = load_lds(xq[m], kl, lane);
            const int xsc = r < p.rows ? xs[m][kl / 32 + g] : 0;
            acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(wa[i], xa, acc, 0, 0, 0, wsc[i], 0, xsc);
        }
        MAIN_PROJ_STAMP(2);
        if (unit != 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                __hip_atomic_store(part + (unit * WAVE + lane) * 4 + i, tagged(__float_as_uint(acc[i]), count),
                                   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
        } else { // the tile's sum: its other five chains' partials, then aiter's order
            uint64_t q[UNITS - 1][4];
            uint64_t t0 = 0;
            for (int n = 1;; ++n) {
                bool ready = true;
#pragma unroll
                for (int u = 1; u < UNITS; ++u) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        q[u - 1][i] = __hip_atomic_load(part + (u * WAVE + lane) * 4 + i, __ATOMIC_RELAXED,
                                                        __HIP_MEMORY_SCOPE_AGENT);
                        ready &= tag_is(q[u - 1][i], count);
                    }
                }
                if (ready) break;
                __builtin_amdgcn_s_sleep(1);
                check_deadline(p, n, t0);
            }
            MAIN_PROJ_STAMP(3);
            uint32_t pk[2];
            {
#pragma float_control(precise, on)
                float o[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float total = 0.0f;
#pragma unroll
                    for (int s = 0; s < SPLITS; ++s) {
                        const float a0 = s == 0 ? acc[i] : __uint_as_float(static_cast<uint32_t>(q[2 * s - 1][i]));
                        const float a1 = __uint_as_float(static_cast<uint32_t>(q[2 * s][i]));
                        // the diagonal panels: each summed with its zero partner, then the two added
                        total = total + ((a0 + 0.0f) + (0.0f + a1));
                    }
                    o[i] = total;
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const bfloat16 lo = static_cast<bfloat16>(o[2 * i]), hi = static_cast<bfloat16>(o[2 * i + 1]);
                    pk[i] = static_cast<uint32_t>(__builtin_bit_cast(uint16_t, lo)) |
                            static_cast<uint32_t>(__builtin_bit_cast(uint16_t, hi)) << 16;
                }
            }
            // lane: row r, columns g * 4 .. + 4 of the tile
            if (r < p.rows) {
                const int pair = (p.rank * p.tiles * TILE_N + n0 + g * 4) / 2;
                const int64_t at = (static_cast<int64_t>(parity) * MAX_ROWS + r) * PAIRS + pair;
                for (int dst = 0; dst < p.tp; ++dst) {
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        __hip_atomic_store(p.recv[dst] + at + i, tagged(pk[i], count), __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_SYSTEM);
                    }
                }
            }
            MAIN_PROJ_STAMP(4);
        }
    }
    // every CTA: its share of the rows' pairs, from this rank's exchange buffer
    const uint64_t *mine = p.recv[p.rank] + static_cast<int64_t>(parity) * MAX_ROWS * PAIRS;
    const int total = p.rows * PAIRS, stride = grid * THREADS;
    for (int i = b * THREADS + tid; i < total; i += stride) {
        const int row = i / PAIRS, pair = i % PAIRS;
        uint64_t v;
        uint64_t t0 = 0;
        for (int n = 1;; ++n) {
            v = __hip_atomic_load(mine + static_cast<int64_t>(row) * PAIRS + pair, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
            if (tag_is(v, count)) break;
            __builtin_amdgcn_s_sleep(2);
            check_deadline(p, n, t0);
        }
        reinterpret_cast<uint32_t *>(p.out + static_cast<int64_t>(row) * p.out_stride)[pair] = static_cast<uint32_t>(v);
    }
    MAIN_PROJ_STAMP(5);
}

// CTA b of the grid CTAs on the caller's block of THREADS threads
__device__ inline void run(const Params &p, const int b, const int grid) {
    __shared__ __attribute__((aligned(16))) uint8_t xq[MAX_ROWS][SLICE];
    __shared__ uint8_t xs[MAX_ROWS][SLICE / 32];
    __shared__ int slot;
    run(p, b, grid, threadIdx.x, xq, xs, slot);
}

// x [rows, 15360] bf16; w [5120 / tp, 15360] E4M3 and ws [5120 / tp / 32, 480] E8M0 (this rank's columns of
// main_proj); out [>= rows, 5120] bf16; state [2048] int32 and work (``buffers()``, zeroed once); peers: every
// rank's exchange buffer (int64, host), zeroed once
inline Params params(const at::Tensor &x, const at::Tensor &w, const at::Tensor &ws, at::Tensor &out, at::Tensor &state,
                     at::Tensor &work, const at::Tensor &peers, int64_t rank, double deadline_s) {
    const int64_t rows = x.size(0), tp = peers.numel();
    TORCH_CHECK(1 <= rows && rows <= MAX_ROWS, "1 to 8 rows: aiter's BLOCK_M 8 config");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == K && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 &&
                reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && x.stride(0) % 8 == 0);
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong && 1 <= tp && tp <= MAX_TP && H % (tp * 32) == 0);
    TORCH_CHECK(0 <= rank && rank < tp);
    TORCH_CHECK(w.dim() == 2 && w.size(0) == H / tp && w.size(1) == K && w.stride(1) == 1 && w.stride(0) == K &&
                w.element_size() == 1);
    TORCH_CHECK(ws.dim() == 2 && ws.size(0) == H / tp / 32 && ws.size(1) == GROUPS && ws.stride(1) == 1 &&
                ws.stride(0) == GROUPS && ws.element_size() == 1);
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) >= rows && out.size(1) == H &&
                out.stride(1) == 1 && out.stride(0) % 2 == 0);
    TORCH_CHECK(state.scalar_type() == at::kInt && state.is_contiguous() && state.numel() >= STATE_INTS);
    TORCH_CHECK(work.scalar_type() == at::kByte && work.is_contiguous() && work.numel() >= WORK_BYTES &&
                reinterpret_cast<uintptr_t>(work.data_ptr()) % 16 == 0);
    Params p{};
    p.x = static_cast<const bfloat16 *>(x.data_ptr());
    p.x_stride = x.stride(0);
    p.w = static_cast<const uint8_t *>(w.data_ptr());
    p.ws = static_cast<const uint8_t *>(ws.data_ptr());
    p.out = static_cast<bfloat16 *>(out.data_ptr());
    p.out_stride = out.stride(0);
    p.state = state.data_ptr<int>();
    p.part = reinterpret_cast<uint64_t *>(work.data_ptr<uint8_t>());
    for (int64_t i = 0; i < tp; ++i) p.recv[i] = reinterpret_cast<uint64_t *>(peers.data_ptr<int64_t>()[i]);
    p.deadline = static_cast<uint64_t>(deadline_s * 1e8); // s_memrealtime: 100 MHz
    p.rows = static_cast<int>(rows);
    p.rank = static_cast<int>(rank);
    p.tp = static_cast<int>(tp);
    p.tiles = static_cast<int>(H / tp / TILE_N);
    return p;
}

// the CTAs of a launch of p
inline unsigned grid(const Params &p) { return static_cast<unsigned>(p.tiles * UNITS); }

} // namespace main_proj
