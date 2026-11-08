// SPDX-License-Identifier: Apache-2.0
// The DSpark draft's committed KV after main_proj in one launch: the main norm (aiter's add_rmsnorm_quant_kernel as
// SGLang's RMSNorm runs it at width 5120), the MXFP8 quant (SGLang's _mxfp8_quant_kernel), the stacked wkv GEMM
// (aiter's packed afp8wfp8 at 5 to 8 rows: 5 fused splits of K 1024, each 8 chained 16x16x128 scaled MFMAs, the
// splits summed in order), then every stage's kv_norm, rope and SWA store (inject.hip's). Built with aiter's RMSNorm
// flags (-ffast-math, flushed denormals) for the norm's bits; the rest under float_control(precise).
//
// CTAs by role, the grid fixed: 8 make a row's norm scale (its rsqrt) each and publish it; 8 a stage store a row
// each; the rest are one split of a 16-column wkv tile each, whose weights and K slice of the rows load while the
// scales are made, then norm and quantize the slice (the same values as the whole row's) and publish the split's
// partials. A store CTA sums its row's splits in order and runs inject.hip's kv_norm, rope and store on it. Every
// hand-off is an agent-coherent (sc1) tagged pair whose tag is the CTA's launch count (each CTA counts its own
// launches; with a fixed grid every CTA of a launch holds the same count): no L2 write-back, invalidate or counter.
#include <cstdint>
#include <optional>

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
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef short v2s __attribute__((ext_vector_type(2)));

constexpr int THREADS = 256, WAVE = 64;
constexpr int H = 5120;                   // the main hidden
constexpr int MAX_ROWS = 8;               // aiter's 5-split K 1024 config holds 5 to 8 rows
constexpr int TILE_N = 16;                // a tile's columns of the stacked wkv
constexpr int SPLITS = 5, SPLIT_K = 1024, STEPS = SPLIT_K / 128;
constexpr int GROUPS = H / 32;            // MXFP8 groups a row
constexpr int DIM = 512, ROPE = 64, MAX_STAGES = 4, TILES = DIM / TILE_N;
constexpr int NORM_THREADS = 256, NORM_CHUNK = 8, NORM_CHUNKS = 3; // aiter's 256 threads, 24 values a thread
constexpr int STORE_THREADS = DIM / 2, LANES = 32, SWARPS = STORE_THREADS / LANES, ROPE_WARP = SWARPS - 1;
static_assert(THREADS == NORM_THREADS && THREADS == STORE_THREADS && LANES == 32 && SWARPS == 8);

constexpr int CTAS_A_STAGE = MAX_ROWS + TILES * SPLITS;   // store CTAs (a row each), splits
constexpr int MAX_CTAS = MAX_ROWS + MAX_STAGES * CTAS_A_STAGE;
// the state ints (zeroed once): the rows' tagged scales (rsqrt bits, count), every CTA's launch count
constexpr int RCP = 0, COUNTS = RCP + 2 * MAX_ROWS, STATE_INTS = 1024;
static_assert(COUNTS + MAX_CTAS <= STATE_INTS);
// the work bytes: the splits' partials, tagged pairs (a tile's 5 splits, a lane's 4 floats each)
constexpr int64_t WORK_BYTES = int64_t{MAX_STAGES} * TILES * SPLITS * WAVE * 4 * 8;
// a split CTA's K slice of the rows: 8-value chunks, SLICE_CHUNKS a row, CHUNKS_A_THREAD rows a thread
constexpr int SLICE_CHUNKS = SPLIT_K / NORM_CHUNK, CHUNKS_A_THREAD = MAX_ROWS * SLICE_CHUNKS / THREADS;
constexpr int ROW_STEP = THREADS / SLICE_CHUNKS;

#ifdef INJECT_STAMPS // a timing build (tests only): thread 0's realtime ticks per CTA, past the state ints
#define STAMP(k) if (threadIdx.x == 0) reinterpret_cast<uint64_t *>(p.state + STATE_INTS)[blockIdx.x * 16 + (k)] = __builtin_amdgcn_s_memrealtime()
#else
#define STAMP(k)
#endif

struct Params {
    const bfloat16 *x; // main_proj's output [rows, H]
    int64_t x_stride;
    const bfloat16 *norm_w;
    float norm_eps;
    const uint8_t *w;  // stacked wkv [stages * DIM, H] E4M3
    const uint8_t *ws; // its E8M0 scales [stages * DIM / 32, H / 32]
    bfloat16 *kv;      // [MAX_ROWS, stages * DIM], the kv rows (tests read them)
    int *state;        // [STATE_INTS]
    uint64_t *part;    // the work bytes
    const bfloat16 *weight[MAX_STAGES];
    const float *freqs[MAX_STAGES];
    uint8_t *cache[MAX_STAGES];
    const void *pos;
    bool pos_i32, v41;
    const int32_t *loc;
    const int64_t *cache_loc, *map;
    const int32_t *commit;
    int64_t map_n, positions, page_bytes;
    int width, rows, stages, page_bits;
    float eps;
};

// ---------------------------------------------------------------- agent-coherent hand-offs
template <typename T> __device__ inline void put(T *at, T v) {
    __hip_atomic_store(at, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T> __device__ inline T get(const T *at) {
    return __hip_atomic_load(at, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// ---------------------------------------------------------------- the main norm (aiter's, see norm/combine_norm.hip)
template <int CTRL> __device__ inline float dpp(float v) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), CTRL, 0xf, 0xf, false));
}

__device__ inline float wave_sum64(float v) {
    v = dpp<0xb1>(v) + v;
    v = dpp<0x4e>(v) + v;
    v = dpp<0x141>(v) + v;
    v = dpp<0x140>(v) + v;
    v = dpp<0x142>(v) + v;
    v = dpp<0x143>(v) + v;
    return v;
}

__device__ inline float quad_sum(float v) {
    v = dpp<0xb1>(v) + v;
    v = dpp<0x4e>(v) + v;
    return v;
}

__device__ inline float mul2(float x, float rcp, float w) {
#pragma clang fp contract(off) reassociate(off)
    return (x * rcp) * w;
}

// a norm thread's ``c``-th 8 values of a row: aiter's 256-thread layout, 24 values a thread
__device__ inline int norm_at(int wave, int c, int lane) {
    return wave * WAVE * NORM_CHUNKS * NORM_CHUNK + c * WAVE * NORM_CHUNK + lane * NORM_CHUNK;
}

__device__ inline void unpack8(const uint32x4 &w, float *y) {
    const bfloat16 *b = reinterpret_cast<const bfloat16 *>(&w);
#pragma unroll
    for (int k = 0; k < NORM_CHUNK; ++k) y[k] = static_cast<float>(b[k]);
}

// a row's sum of squares over its 256 threads' wave: each wave's total on its last lane
__device__ inline float row_sq(const uint32x4 (&xr)[NORM_CHUNKS], int wave, int lane) {
    float y[NORM_CHUNKS * NORM_CHUNK];
#pragma unroll
    for (int c = 0; c < NORM_CHUNKS; ++c) {
        if (norm_at(wave, c, lane) < H) {
            unpack8(xr[c], y + c * NORM_CHUNK);
        } else {
#pragma unroll
            for (int k = 0; k < NORM_CHUNK; ++k) y[c * NORM_CHUNK + k] = 0.0f;
        }
    }
    float sq = 0.0f;
    for (int i = 0; i < NORM_CHUNKS * NORM_CHUNK; i++) {
        sq += (y[i] * y[i]);
    }
    return wave_sum64(sq);
}

// ---------------------------------------------------------------- the MXFP8 quant (SGLang's _mxfp8_quant_kernel)
// a lane's 8 normed values of a 32-value group held by 4 lanes of a quad: their E4M3 bytes; the group's biased E8M0
// scale into ``scale``
__device__ inline uint64_t quant8(const float (&x)[NORM_CHUNK], uint8_t &scale) {
#pragma float_control(precise, on)
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < NORM_CHUNK; ++i) amax = __builtin_fmaxf(amax, __builtin_fabsf(x[i]));
    amax = __builtin_fmaxf(amax, dpp<0xb1>(amax));
    amax = __builtin_fmaxf(amax, dpp<0x4e>(amax));
    amax = __builtin_fmaxf(amax, 1e-30f);
    float sb = __builtin_ceilf(__builtin_log2f(amax / 448.0f)) + 127.0f;
    sb = __builtin_fminf(__builtin_fmaxf(sb, 0.0f), 254.0f);
    const float descale = __builtin_amdgcn_exp2f(sb - 127.0f);
    v2s w[2];
#pragma unroll
    for (int i = 0; i < NORM_CHUNK; i += 4) {
        const float a = __builtin_amdgcn_fmed3f(x[i] / descale, -448.0f, 448.0f);
        const float b = __builtin_amdgcn_fmed3f(x[i + 1] / descale, -448.0f, 448.0f);
        const float c = __builtin_amdgcn_fmed3f(x[i + 2] / descale, -448.0f, 448.0f);
        const float d = __builtin_amdgcn_fmed3f(x[i + 3] / descale, -448.0f, 448.0f);
        v2s v = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(v2s{0, 0}, a, b, 1.0f, false);
        w[i / 4] = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(v, c, d, 1.0f, true);
    }
    scale = static_cast<uint8_t>(sb);
    return static_cast<uint64_t>(__builtin_bit_cast(uint32_t, w[0])) | static_cast<uint64_t>(__builtin_bit_cast(uint32_t, w[1])) << 32;
}

// ---------------------------------------------------------------- the wkv GEMM (aiter's packed kernel's chain)
// a lane's 32 bytes of a row for one 128-K step: K (lane / 16) * 16 .. + 16 and 64 + (lane / 16) * 16 .. + 16
__device__ inline v8i load_step(const uint8_t *row, int k0, int lane) {
    const int g = lane / 16;
    const uint4 lo = *reinterpret_cast<const uint4 *>(row + k0 + g * 16);
    const uint4 hi = *reinterpret_cast<const uint4 *>(row + k0 + 64 + g * 16);
    v8i v;
    v[0] = lo.x, v[1] = lo.y, v[2] = lo.z, v[3] = lo.w, v[4] = hi.x, v[5] = hi.y, v[6] = hi.z, v[7] = hi.w;
    return v;
}

// ---------------------------------------------------------------- the SWA store (inject.hip's, a row a half)
__device__ inline float bf16_round(float v) { return static_cast<float>(static_cast<bfloat16>(v)); }

// inject.hip's warp reductions (__shfl_xor butterflies), the partners moved by DPP and one swizzle instead of LDS
// permutes: the same adds in the same order
__device__ inline float xor16(float v) { // within 32 lanes
    return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
}

// the butterfly 16, 8, 4, 2, 1 over a warp of 32 lanes
__device__ inline float sum32(float v) {
    v = v + xor16(v);
    v = v + dpp<0x128>(v);             // xor 8: row_ror 8
    v = v + dpp<0x1B>(dpp<0x141>(v));  // xor 4: row_half_mirror (^7), then quad reverse (^3)
    v = v + dpp<0x4E>(v);              // xor 2
    v = v + dpp<0xB1>(v);              // xor 1
    return v;
}

// the butterfly 4, 2, 1 over values repeating every 8 lanes, where row_ror 4 meets the xor 4 partner's value
__device__ inline float sum8_repeated(float v) {
    v = v + dpp<0x124>(v);
    v = v + dpp<0x4E>(v);
    v = v + dpp<0xB1>(v);
    return v;
}

// the maximum over ``W`` (16 or 32) lanes: exact in any order
template <int W> __device__ inline float lanes_max(float v) {
    if constexpr (W == 32) v = fmaxf(v, xor16(v));
    v = fmaxf(v, dpp<0x128>(v));
    v = fmaxf(v, dpp<0x124>(v));
    v = fmaxf(v, dpp<0x4E>(v));
    v = fmaxf(v, dpp<0xB1>(v));
    return v;
}

__device__ inline uint16_t pack_e4m3(float x, float y) {
    x = fmaxf(fminf(x, 448.0f), -448.0f);
    y = fmaxf(fminf(y, 448.0f), -448.0f);
    return static_cast<uint16_t>(__builtin_amdgcn_cvt_pk_fp8_f32(x, y, 0, false) & 0xffff);
}

__device__ inline float inv_scale_ue8m0(int32_t exponent) {
    return __uint_as_float(static_cast<uint32_t>(127 + 127 - exponent) << 23);
}

// a row's store slot, read ahead: its position, its cache slot (-1 none), the committed path's cache index
struct Slot {
    int64_t position, at;
    int loc;
    bool committed;
};

__device__ inline Slot slot_of(const Params &p, int row) {
    Slot s;
    s.position = p.pos_i32 ? static_cast<const int32_t *>(p.pos)[row] : static_cast<const int64_t *>(p.pos)[row];
    s.loc = -1;
    s.committed = false;
    s.at = 0;
    if (p.loc != nullptr) {
        s.loc = p.loc[row];
    } else {
        s.committed = row % p.width < p.commit[row / p.width];
        const int64_t c = p.cache_loc[row];
        s.at = c < 0 ? c + p.map_n : c;
    }
    return s;
}

// the committed path's slot and the position's check
__device__ inline int slot_loc(const Params &p, const Slot &s) {
    int loc = s.loc;
    if (p.loc == nullptr) {
        const int mapped = static_cast<int32_t>(p.map[s.committed ? s.at : 0]);
        loc = s.committed ? mapped : -1;
    }
    return s.position < 0 || s.position >= p.positions ? -1 : loc;
}

// one row's two columns 2 tx, 2 tx + 1 of stage ``stage`` (a store warp of 32 lanes on the row), its kv pair ``x``
// and its sum of squares ``ss``: kv_norm, rope and the store
template <bool V41>
__device__ inline void store_pair(const Params &p, int stage, int tx, float ss, int loc, float x0, float x1, float w0,
                                  float w1, float f0, float f1) {
#pragma float_control(precise, on)
    constexpr int64_t ROW = V41 ? 512 : 576, SCALES = V41 ? 16 : 8;
    constexpr int TILE = V41 ? 16 : LANES; // the lanes a scale covers
    const int warp = tx / LANES, lane = tx % LANES;
    const float norm = rsqrtf(ss / DIM + p.eps);
    const float d0 = x0 * norm * w0, d1 = x1 * norm * w1;
    float v0 = d0, v1 = d1;
    if constexpr (V41) {
        v0 = bf16_round(d0), v1 = bf16_round(d1);
        if (warp == ROPE_WARP) {
            const float re = v0, im = v1;
            v0 = bf16_round(fmaf(re, f0, -(im * f1)));
            v1 = bf16_round(fmaf(re, f1, im * f0));
        }
    }
    const float amax = lanes_max<TILE>(fmaxf(fabsf(v0), fabsf(v1)));
    if (loc < 0) return;
    const int64_t page = loc >> p.page_bits, slot = loc & ((1 << p.page_bits) - 1);
    uint8_t *base = p.cache[stage] + page * p.page_bytes;
    uint8_t *data_row = base + slot * ROW;
    uint8_t *scale_row = base + (ROW << p.page_bits) + slot * SCALES;
    if constexpr (V41) {
        const uint32_t bits = __float_as_uint(amax);
        const int32_t exponent =
            max(static_cast<int32_t>(bits >> 23) - 8 + static_cast<int32_t>((bits & 0x7FFFFFu) > 0x600000u), 114);
        const float inv = inv_scale_ue8m0(exponent);
        reinterpret_cast<uint16_t *>(data_row)[tx] = pack_e4m3(v0 * inv, v1 * inv);
        if (tx % TILE == 0) scale_row[tx / TILE] = static_cast<uint8_t>(exponent);
    } else if (warp == ROPE_WARP) {
        bfloat16 *rope = reinterpret_cast<bfloat16 *>(data_row + 448);
        rope[2 * lane] = static_cast<bfloat16>(v0 * f0 - v1 * f1);
        rope[2 * lane + 1] = static_cast<bfloat16>(v0 * f1 + v1 * f0);
    } else {
        const float scale = fmaxf(1e-4f, amax) / 448.0f;
        const uint32_t u = __float_as_uint(scale);
        const int32_t exponent = static_cast<int32_t>((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0);
        const float inv = inv_scale_ue8m0(exponent);
        reinterpret_cast<uint16_t *>(data_row)[tx] = pack_e4m3(v0 * inv, v1 * inv);
        if (lane == 0) scale_row[warp] = static_cast<uint8_t>(exponent);
    }
}

__device__ inline float warp_ss(float x0, float x1) {
#pragma float_control(precise, on)
    float ss = fmaf(x0, x0, 0.0f);
    ss = fmaf(x1, x1, ss);
    return sum32(ss);
}

// ---------------------------------------------------------------- the launch
__device__ inline uint64_t tagged(float v, int count) {
    return static_cast<uint64_t>(__float_as_uint(v)) | static_cast<uint64_t>(static_cast<uint32_t>(count)) << 32;
}

__device__ inline bool tag_is(uint64_t pair, int count) { return static_cast<int>(pair >> 32) == count; }

// the CTA's launch count, from thread 0's fetch-add (issued first) to every thread through ``slot`` (LDS) at a barrier
__device__ inline int launch_count(int counted, int *slot) {
    if (threadIdx.x == 0) *slot = counted + 1;
    __syncthreads();
    return *slot;
}

// row ``row``'s norm scale, published tagged ``count``
__device__ inline void make_rcp(const Params &p, int row, int counted, float *npart, int *slot) {
    const int tid = threadIdx.x, lane = tid % WAVE, wave = tid / WAVE;
    uint32x4 xr[NORM_CHUNKS];
#pragma unroll
    for (int c = 0; c < NORM_CHUNKS; ++c) {
        const int h = min(norm_at(wave, c, lane), H - NORM_CHUNK);
        xr[c] = *reinterpret_cast<const uint32x4 *>(p.x + static_cast<int64_t>(row) * p.x_stride + h);
    }
    const float sq = row_sq(xr, wave, lane);
    if (lane == WAVE - 1) npart[wave] = sq;
    const int count = launch_count(counted, slot);
    const float rcp = rsqrtf(quad_sum(npart[lane % (NORM_THREADS / WAVE)]) / H + p.norm_eps);
    if (tid == 0) put(reinterpret_cast<uint64_t *>(p.state + RCP) + row, tagged(rcp, count));
}

// the CTA's K slice of the rows (thread chunk c: row tid / SLICE_CHUNKS + c * ROW_STEP), awaited scales, normed and
// quantized into ``xq``, ``xs`` (LDS)
__device__ inline void make_slice(const Params &p, int count, const uint32x4 (&xc)[CHUNKS_A_THREAD],
                                  const uint32x4 &nwc, uint8_t (*xq)[SPLIT_K], uint8_t (*xs)[SPLIT_K / 32]) {
    const int tid = threadIdx.x, chunk = tid % SLICE_CHUNKS, row0 = tid / SLICE_CHUNKS;
    const uint64_t *tags = reinterpret_cast<const uint64_t *>(p.state + RCP);
    uint64_t rcp[CHUNKS_A_THREAD];
    for (;;) {
        bool ready = true;
#pragma unroll
        for (int c = 0; c < CHUNKS_A_THREAD; ++c) {
            rcp[c] = get(tags + min(row0 + c * ROW_STEP, p.rows - 1));
            ready &= tag_is(rcp[c], count);
        }
        if (ready) break;
        __builtin_amdgcn_s_sleep(1);
    }
#pragma unroll
    for (int c = 0; c < CHUNKS_A_THREAD; ++c) {
        const int row = row0 + c * ROW_STEP;
        if (row < p.rows) {
            float y[NORM_CHUNK], w[NORM_CHUNK], v[NORM_CHUNK];
            unpack8(xc[c], y);
            unpack8(nwc, w);
            const float r = __uint_as_float(static_cast<uint32_t>(rcp[c]));
#pragma unroll
            for (int k = 0; k < NORM_CHUNK; ++k) v[k] = static_cast<float>(static_cast<bfloat16>(mul2(y[k], r, w[k])));
            uint8_t scale;
            *reinterpret_cast<uint64_t *>(&xq[row][chunk * NORM_CHUNK]) = quant8(v, scale);
            if (chunk % 4 == 0) xs[row][chunk / 4] = scale;
        }
    }
}

// the split on wave 0 from the slice: the 8-MFMA chain, the partial published tagged ``count``
__device__ inline void split_gemm(const Params &p, int tile, int split, int count, const v8i (&wa)[STEPS],
                                  const int (&wsc)[STEPS], const uint8_t (*xq)[SPLIT_K], const uint8_t (*xs)[SPLIT_K / 32]) {
    const int lane = threadIdx.x % WAVE, r = lane % 16, g = lane / 16, m = min(r, p.rows - 1);
    v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int st = 0; st < STEPS; ++st) {
        const v8i xa = load_step(xq[m], st * 128, lane);
        const int xsc = r < p.rows ? xs[m][st * 4 + g] : 0;
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(wa[st], xa, acc, 0, 0, 0, wsc[st], 0, xsc);
    }
    uint64_t *part = p.part + ((static_cast<int64_t>(tile) * SPLITS + split) * WAVE + lane) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) put(part + i, tagged(acc[i], count));
}

template <bool V41> __device__ inline void split_cta(const Params &p, int unit, int counted, int *slot) {
    __shared__ __attribute__((aligned(16))) uint8_t xq[MAX_ROWS][SPLIT_K];
    __shared__ uint8_t xs[MAX_ROWS][SPLIT_K / 32];
    const int tid = threadIdx.x, wave = tid / WAVE, lane = tid % WAVE;
    const int stage = unit / (TILES * SPLITS), tile = stage * TILES + unit % (TILES * SPLITS) / SPLITS;
    const int split = unit % SPLITS, n0 = tile * TILE_N;
    // the K slice of the rows and the norm weight; wave 0 the tile's weights
    const int col = split * SPLIT_K + (tid % SLICE_CHUNKS) * NORM_CHUNK;
    uint32x4 xc[CHUNKS_A_THREAD];
#pragma unroll
    for (int c = 0; c < CHUNKS_A_THREAD; ++c) {
        const int row = tid / SLICE_CHUNKS + c * ROW_STEP;
        if (row < p.rows) xc[c] = *reinterpret_cast<const uint32x4 *>(p.x + static_cast<int64_t>(row) * p.x_stride + col);
    }
    const uint32x4 nwc = *reinterpret_cast<const uint32x4 *>(p.norm_w + col);
    v8i wa[STEPS];
    int wsc[STEPS];
    if (wave == 0) {
        const int r = lane % 16, g = lane / 16;
        const uint8_t *wrow = p.w + static_cast<int64_t>(n0 + r) * H + split * SPLIT_K;
        const uint8_t *srow = p.ws + ((n0 + r) / 32) * GROUPS + split * SPLIT_K / 32 + g;
#pragma unroll
        for (int st = 0; st < STEPS; ++st) {
            wa[st] = load_step(wrow, st * 128, lane);
            wsc[st] = srow[st * 4];
        }
    }
    const int count = launch_count(counted, slot);
    make_slice(p, count, xc, nwc, xq, xs);
    __syncthreads();
    if (wave == 0) {
        STAMP(5);
        split_gemm(p, tile, split, count, wa, wsc, xq, xs);
        STAMP(6);
    }
}

// stage ``stage``'s row ``row`` (live): thread tx on columns 2 tx, 2 tx + 1 (inject.hip's store layout)
template <bool V41> __device__ inline void store_cta(const Params &p, int stage, int row, int counted, int *slot) {
#pragma float_control(precise, on)
    __shared__ float spart[SWARPS];
    const int tx = threadIdx.x, lane = tx % LANES, col = 2 * tx;
    // the side reads (earlier kernels' tensors): the row's slot, the kv_norm weights, the rope table row
    const Slot sl = slot_of(p, row);
    const uint32_t wpair = reinterpret_cast<const uint32_t *>(p.weight[stage])[tx];
    float f0 = 0.0f, f1 = 0.0f;
    if (tx / LANES == ROPE_WARP) {
        const int64_t at = sl.position < 0 || sl.position >= p.positions ? 0 : sl.position;
        const float2 f = reinterpret_cast<const float2 *>(p.freqs[stage] + at * ROPE)[lane];
        f0 = f.x, f1 = f.y;
    }
    const int loc = slot_loc(p, sl);
    const int count = launch_count(counted, slot);
    // the columns' partials, split by split: tile, the split's output lane (n / 4) * 16 + row and float n % 4
    const int tile = stage * TILES + col / TILE_N, n = col % TILE_N;
    const uint64_t *part = p.part + (static_cast<int64_t>(tile) * SPLITS * WAVE + (n / 4) * 16 + row) * 4 + n % 4;
    uint64_t q[SPLITS][2];
    for (;;) {
        bool ready = true;
#pragma unroll
        for (int s = 0; s < SPLITS; ++s) {
            q[s][0] = get(part + s * WAVE * 4);
            q[s][1] = get(part + s * WAVE * 4 + 1);
            ready &= tag_is(q[s][0], count) && tag_is(q[s][1], count);
        }
        if (ready) break;
        __builtin_amdgcn_s_sleep(8); // light polling: the splits stream their weights meanwhile
    }
    STAMP(2);
    float t0 = 0.0f, t1 = 0.0f; // aiter's: the splits' partials added to zero in split order
#pragma unroll
    for (int s = 0; s < SPLITS; ++s) {
        t0 = t0 + __uint_as_float(static_cast<uint32_t>(q[s][0]));
        t1 = t1 + __uint_as_float(static_cast<uint32_t>(q[s][1]));
    }
    const bfloat16 k0 = static_cast<bfloat16>(t0), k1 = static_cast<bfloat16>(t1);
    reinterpret_cast<uint32_t *>(p.kv + static_cast<int64_t>(row) * p.stages * DIM + stage * DIM)[tx] =
        static_cast<uint32_t>(__builtin_bit_cast(uint16_t, k0)) | static_cast<uint32_t>(__builtin_bit_cast(uint16_t, k1)) << 16;
    const float x0 = static_cast<float>(k0), x1 = static_cast<float>(k1);
    const float mine = warp_ss(x0, x1);
    if (lane == 0) spart[tx / LANES] = mine;
    __syncthreads();
    STAMP(3);
    const float sum = sum8_repeated(spart[lane % SWARPS]);
    const float w0 = static_cast<float>(__builtin_bit_cast(bfloat16, static_cast<uint16_t>(wpair & 0xffff)));
    const float w1 = static_cast<float>(__builtin_bit_cast(bfloat16, static_cast<uint16_t>(wpair >> 16)));
    store_pair<V41>(p, stage, tx, sum, loc, x0, x1, w0, w1, f0, f1);
    STAMP(4);
}

template <bool V41> __global__ __launch_bounds__(THREADS) void inject_kv_kernel(const Params p) {
    __shared__ float npart[NORM_THREADS / WAVE];
    __shared__ int slot;
    const int b = blockIdx.x;
    int counted = 0;
    if (threadIdx.x == 0) counted = __hip_atomic_fetch_add(p.state + COUNTS + b, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    STAMP(0);
    if (b < MAX_ROWS) {
        if (b < p.rows) make_rcp(p, b, counted, npart, &slot);
        STAMP(1);
        return;
    }
    const int sb = b - MAX_ROWS, stores = p.stages * MAX_ROWS;
    if (sb < stores) {
        if (sb % MAX_ROWS < p.rows) store_cta<V41>(p, sb / MAX_ROWS, sb % MAX_ROWS, counted, &slot);
    } else {
        split_cta<V41>(p, sb - stores, counted, &slot);
        STAMP(1);
    }
}

// x [rows, H] bf16 (main_proj's output), norm_w [H] bf16, w [stages * 512, H] E4M3 and ws [stages * 16, H / 32] E8M0
// (the stacked wkv); scratch (``buffers()``, zeroed once): kv [8, stages * 512] bf16, state [1024] int32, work bytes;
// then swa_store's arguments
void inject_kv_op(const at::Tensor &x, const at::Tensor &norm_w, double norm_eps, const at::Tensor &w,
                  const at::Tensor &ws, const at::Tensor &kv, const at::Tensor &state, const at::Tensor &work,
                  at::TensorList weights, at::TensorList freqs, const at::Tensor &pos,
                  const std::optional<at::Tensor> &loc, at::TensorList caches, double eps, int64_t page_size, bool v41,
                  const std::optional<at::Tensor> &cache_loc, const std::optional<at::Tensor> &mapping,
                  const std::optional<at::Tensor> &commit, int64_t width) {
    const int64_t stages = caches.size(), rows = x.size(0);
    TORCH_CHECK(0 < stages && stages <= MAX_STAGES && weights.size() == caches.size() && freqs.size() == caches.size());
    TORCH_CHECK(x.dim() == 2 && x.size(1) == H && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 &&
                reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && x.stride(0) % 8 == 0);
    TORCH_CHECK(1 <= rows && rows <= MAX_ROWS, "5 to 8 rows: aiter's split-K 1024 config");
    TORCH_CHECK(norm_w.scalar_type() == at::kBFloat16 && norm_w.numel() == H && norm_w.is_contiguous());
    TORCH_CHECK(w.dim() == 2 && w.size(0) == stages * DIM && w.size(1) == H && w.is_contiguous() && w.element_size() == 1);
    TORCH_CHECK(ws.dim() == 2 && ws.size(0) == stages * DIM / 32 && ws.size(1) == H / 32 && ws.is_contiguous() &&
                ws.element_size() == 1);
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16 && kv.is_contiguous() && kv.numel() >= MAX_ROWS * stages * DIM);
    TORCH_CHECK(state.scalar_type() == at::kInt && state.is_contiguous() && state.numel() >= STATE_INTS);
    TORCH_CHECK(work.scalar_type() == at::kByte && work.is_contiguous() && work.numel() >= WORK_BYTES &&
                reinterpret_cast<uintptr_t>(work.data_ptr()) % 16 == 0);
    TORCH_CHECK(pos.dim() == 1 && pos.size(0) == rows && pos.is_contiguous() &&
                (pos.scalar_type() == at::kLong || pos.scalar_type() == at::kInt));
    TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x.device());
    Params p{};
    p.positions = INT64_MAX;
    for (int64_t s = 0; s < stages; ++s) {
        TORCH_CHECK(weights[s].scalar_type() == at::kBFloat16 && weights[s].numel() == DIM && weights[s].is_contiguous());
        TORCH_CHECK(freqs[s].scalar_type() == at::kFloat && freqs[s].dim() == 2 && freqs[s].size(1) == ROPE &&
                    freqs[s].is_contiguous());
        TORCH_CHECK(caches[s].dim() == 2 && caches[s].scalar_type() == at::kByte && caches[s].stride(1) == 1 &&
                    caches[s].stride(0) == caches[0].stride(0));
        p.weight[s] = static_cast<const bfloat16 *>(weights[s].data_ptr());
        p.freqs[s] = freqs[s].data_ptr<float>();
        p.cache[s] = static_cast<uint8_t *>(caches[s].data_ptr());
        p.positions = std::min(p.positions, freqs[s].size(0));
    }
    if (cache_loc.has_value()) {
        TORCH_CHECK(mapping.has_value() && commit.has_value() && width > 0 && rows % width == 0);
        TORCH_CHECK(cache_loc->scalar_type() == at::kLong && cache_loc->is_contiguous() && cache_loc->numel() == rows);
        TORCH_CHECK(mapping->scalar_type() == at::kLong && mapping->is_contiguous());
        TORCH_CHECK(commit->scalar_type() == at::kInt && commit->is_contiguous() && commit->numel() >= rows / width);
        p.cache_loc = cache_loc->data_ptr<int64_t>();
        p.map = mapping->data_ptr<int64_t>();
        p.map_n = mapping->numel();
        p.commit = commit->data_ptr<int32_t>();
        p.width = static_cast<int>(width);
    } else {
        TORCH_CHECK(loc.has_value() && loc->scalar_type() == at::kInt && loc->is_contiguous() && loc->numel() == rows);
        p.loc = loc->data_ptr<int32_t>();
    }
    p.x = static_cast<const bfloat16 *>(x.data_ptr());
    p.x_stride = x.stride(0);
    p.norm_w = static_cast<const bfloat16 *>(norm_w.data_ptr());
    p.norm_eps = static_cast<float>(norm_eps);
    p.w = static_cast<const uint8_t *>(w.data_ptr());
    p.ws = static_cast<const uint8_t *>(ws.data_ptr());
    p.kv = static_cast<bfloat16 *>(kv.data_ptr());
    p.state = state.data_ptr<int>();
    uint8_t *bytes = work.data_ptr<uint8_t>();
    p.part = reinterpret_cast<uint64_t *>(bytes);
    p.pos = pos.data_ptr();
    p.pos_i32 = pos.scalar_type() == at::kInt;
    p.v41 = v41;
    p.rows = static_cast<int>(rows);
    p.stages = static_cast<int>(stages);
    p.page_bits = __builtin_ctzll(page_size);
    TORCH_CHECK(caches[0].size(0) * page_size <= INT32_MAX, "SWA slots past int32");
    p.page_bytes = caches[0].stride(0);
    p.eps = static_cast<float>(eps);
    const auto kernel = v41 ? inject_kv_kernel<true> : inject_kv_kernel<false>;
    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(x.device().index()).stream();
    const unsigned grid = static_cast<unsigned>(MAX_ROWS + stages * CTAS_A_STAGE); // fixed: the launch counts agree
    hipLaunchKernelGGL(kernel, dim3(grid), dim3(THREADS), 0, stream, p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("inject_kv(Tensor x, Tensor norm_w, float norm_eps, Tensor w, Tensor ws, Tensor(a!) kv, Tensor(b!) state, "
          "Tensor(d!) work, Tensor[] weights, Tensor[] freqs, Tensor pos, Tensor? loc, Tensor(c!)[] caches, float eps, "
          "int page_size, bool v41, Tensor? cache_loc=None, Tensor? mapping=None, Tensor? commit=None, int width=0) "
          "-> ()");
    m.impl("inject_kv", at::kCUDA, &inject_kv_op);
}
