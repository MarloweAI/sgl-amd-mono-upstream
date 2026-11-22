// SPDX-License-Identifier: Apache-2.0
// Kimi-K3's o_proj all-reduce and the FFN-side attention-residual read in one launch (TP8, gfx950).
//
// Blocks [0, n_ar) run aiter's own all-reduce for the message (custom_all_reduce.cuh in the image:
// lamportAllReduceTwoShotWaveSplit up to 1 MiB, cross_device_reduce_2stage above), the same exchange
// and the same sums, except that the result goes to a rank-local hand-off buffer, at device scope, in
// the lamport sentinel format instead of the output tensor. Block n_ar + t then reads row t: it loads and
// scores its bank rows while the all-reduce runs, polls its row of the hand-off and re-arms it, and
// runs SGLang's attn_res_hip._agg_kernel (HAS_ADD, APPLY_OUT_NORM, optional bank snapshot) with
// Triton's operation order, reduction trees, DPP reductions and FMA contractions, so prefix_out,
// the bank snapshot and out are bitwise the stock all-reduce + _agg_kernel's.
//
// The hand-off needs no epochs: the next launch's all-reduce writes a row only after this launch,
// whose reader re-armed it, has ended.

#include <cstdint>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#include <custom_all_reduce.cuh>  // aiter csrc/include of the image

namespace k3_ar_seam {

using aiter::RankData;
using aiter::RankSignals;
using aiter::Signal;
using T = opus::bf16_t;
using P = opus::vector_t<T, 8>;
using A = opus::vector_t<opus::fp32_t, 8>;
using ui32x4_t = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;

constexpr int kWorld = 8;
constexpr int kHidden = 7168;
constexpr int kRowPacks = kHidden / 8;
constexpr int kThreads = 512;
constexpr int kMaxRows = 192;
constexpr int64_t kHandoffBytes = int64_t{kMaxRows} * kHidden * 2;
constexpr float kLog2e = 0x1.715476p+0f;

enum Algo : int { kTwoShot = 0, kTwoStage = 1, kTwoStageTail = 2 };

struct Exchange {
  // two-shot: this rank's input, every rank's rs inbox (its ag inbox follows) and the round counters
  const T* input;
  aiter::LamportPeers peers;
  uint32_t* round;
  // 2-stage: every rank's input and signal, this rank's signal
  RankData* inputs;
  RankSignals sg;
  Signal* self_sg;
  int rank;
  int count;  // two-shot: packs a chunk; 2-stage: packs in the message
  int n_ar;
};

struct Read {
  const uint16_t* res;  // [T, H] the pending residual, or null
  uint16_t* bank;       // [T, NB, H]
  const float* cw;      // [H] score_norm weight * score_proj weight
  const uint16_t* ow;   // [H] bf16 output RMSNorm weight
  uint16_t* prefix_out;
  uint16_t* out;
  ui32x4_t* handoff;  // [kMaxRows, H] bf16, rank-local
  int64_t bank_stride_t;
  int64_t bank_stride_b;
  float score_eps;
  float out_eps;
  int write_bank;
#ifdef AR_SEAM_STAMPS
  // test builds: s_memrealtime a row (block start, bank scored, row landed, end), then each AR block's end
  uint64_t* stamps;
#endif
};

#ifdef AR_SEAM_STAMPS
#define K3_STAMP(rd, i) \
  if (threadIdx.x == 0) (rd).stamps[i] = __builtin_amdgcn_s_memrealtime()
#else
#define K3_STAMP(rd, i)
#endif

// ---- the hand-off's 16 B accesses, at device scope ----
//
// Its writers (the all-reduce blocks) and its readers run on different XCDs, whose L2s are not coherent with each
// other. sc1 is the cache policy gfx950 gives an agent-scope relaxed atomic, here on one 16 B access (a 4 B atomic
// would split each in four); nt is aiter's streaming hint. So the hand-off is coherent on any device memory.

using v4u = uint32_t __attribute__((ext_vector_type(4)));
constexpr int kDeviceScope = 16 | 2;  // the buffer intrinsics' cache policy on gfx950: sc1 | nt

DINLINE __amdgpu_buffer_rsrc_t handoff_rsrc(ui32x4_t* handoff) {
  return __builtin_amdgcn_make_buffer_rsrc(handoff, 0, static_cast<int>(kHandoffBytes), 0x00020000);
}

DINLINE void handoff_store(__amdgpu_buffer_rsrc_t h, int64_t pack, ui32x4_t v) {
  __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(v4u, v), h, static_cast<int>(pack * 16), 0, kDeviceScope);
}

DINLINE ui32x4_t handoff_load(__amdgpu_buffer_rsrc_t h, int64_t pack) {
  return __builtin_bit_cast(ui32x4_t,
                            __builtin_amdgcn_raw_buffer_load_b128(h, static_cast<int>(pack * 16), 0, kDeviceScope));
}

// ---- aiter's all-reduce blocks (custom_all_reduce.cuh, the image's aiter), result -> hand-off ----

DINLINE ui32x4_t canonical(ui32x4_t v) {
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = aiter::lamportCanonicalize(v[j]);
  return v;
}

// lamportAllReduceTwoShotWaveSplit<bf16, 8>, part = ex.count
DINLINE void two_shot(const Exchange& ex, __amdgpu_buffer_rsrc_t result) {
  constexpr int ngpus = kWorld;
  constexpr int64_t max_part = aiter::lamportTwoShotMaxPart(ngpus);
  constexpr int64_t buffer_packs = ngpus * max_part;
  constexpr int64_t inbox_packs = aiter::kLamportBuffers * buffer_packs;
  union pack {
    P data;
    ui32x4_t i;
  };
  const int part = ex.count;
  const int rank = ex.rank;

  __shared__ uint32_t s_round;
  __shared__ P s_data[ngpus][aiter::kLamportWaveSize];
  if (threadIdx.x == 0) s_round = ex.round[blockIdx.x];
  __syncthreads();
  const uint32_t round = s_round;
  const uint32_t rearm = (round + 2) % aiter::kLamportBuffers;

  const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / aiter::kLamportWaveSize);
  const int lane_id = threadIdx.x % aiter::kLamportWaveSize;
  const int peer = (rank + wave_id) % ngpus;
  const int pos = blockIdx.x * aiter::kLamportWaveSize + lane_id;
  const bool is_active = pos < part;

  ui32x4_t* self_inbox = reinterpret_cast<ui32x4_t*>(ex.peers.ptrs[rank]);
  ui32x4_t* peer_inbox = reinterpret_cast<ui32x4_t*>(ex.peers.ptrs[peer]);
  const int64_t send_off = round * buffer_packs + rank * max_part + pos;
  const int64_t recv_off = round * buffer_packs + peer * max_part + pos;

  if (is_active) {
    pack pack_val;
    pack_val.data = *(reinterpret_cast<const P*>(ex.input) + (int64_t)peer * part + pos);
#pragma unroll
    for (int j = 0; j < 4; ++j) pack_val.i[j] = aiter::lamportCanonicalize(pack_val.i[j]);
    if (wave_id != 0) {
      __builtin_nontemporal_store(pack_val.i, peer_inbox + send_off);
      pack_val.i = aiter::lamportPollSlot(self_inbox + recv_off);
    }
    s_data[peer][lane_id] = pack_val.data;
  }
  __syncthreads();

  if (is_active) {
    A acc;
#pragma unroll
    for (int i = 0; i < ngpus; ++i) {
      const P rank_data = s_data[i][lane_id];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        acc[j] = i == 0 ? aiter::upcast_s(rank_data[j]) : acc[j] + aiter::upcast_s(rank_data[j]);
    }
    pack sum_val;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum_val.data[j] = aiter::downcast_s<T>(acc[j]);
#pragma unroll
    for (int j = 0; j < 4; ++j) sum_val.i[j] = aiter::lamportCanonicalize(sum_val.i[j]);
    if (wave_id == 0) {
      handoff_store(result, (int64_t)rank * part + pos, sum_val.i);
    } else {
      __builtin_nontemporal_store(sum_val.i, peer_inbox + inbox_packs + send_off);
      sum_val.i = aiter::lamportPollSlot(self_inbox + inbox_packs + recv_off);
      handoff_store(result, (int64_t)peer * part + pos, sum_val.i);
    }
  }

  if (wave_id != 0) {
    const ui32x4_t sentinel{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
    const int64_t rearm_off = rearm * buffer_packs + peer * max_part + pos;
    __builtin_nontemporal_store(sentinel, self_inbox + rearm_off);
    __builtin_nontemporal_store(sentinel, self_inbox + inbox_packs + rearm_off);
  }

  if (threadIdx.x == 0) ex.round[blockIdx.x] = (round + 1) % aiter::kLamportBuffers;
}

// cross_device_reduce_2stage<bf16, 8, *, has_tail, false>, size = ex.count, gridDim.x -> ex.n_ar
template <bool has_tail>
DINLINE void two_stage(const Exchange& ex, __amdgpu_buffer_rsrc_t result) {
  constexpr int ngpus = kWorld;
  constexpr int pack_size = 8;
  constexpr int tnum_gpu = THREAD_NUM / ngpus;
  // warp 0 alone stores the reduced part, and end_sync's release is issued from its first ngpus lanes: one wave
  static_assert(tnum_gpu == 64, "the 2-stage's reducing warp must be one wave");
  const int rank = ex.rank;
  const int size = ex.count;
  const RankSignals& sg = ex.sg;
  Signal* self_sg = ex.self_sg;
  int warp_id = threadIdx.x / tnum_gpu;
  int lane_id = threadIdx.x % tnum_gpu;
  int tid = blockIdx.x * tnum_gpu + lane_id;
  int stride = ex.n_ar * tnum_gpu;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  __shared__ T tmp_smem[tnum_gpu * ngpus * pack_size];
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)ex.inputs->ptrs[i];
    tmps[i] = aiter::get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  aiter::start_sync<ngpus>(sg, self_sg, rank);
  int loop_end = end;
  if constexpr (has_tail) loop_end -= (end - start) % tnum_gpu;
  int idx = start + tid;
  for (; idx < loop_end; idx += stride) {
    *(reinterpret_cast<P*>(&tmp_smem[0]) + threadIdx.x) = ptrs[warp_id][idx];
    __syncthreads();
    if (warp_id == 0) {
      A add_reg;
#pragma unroll
      for (int i = 0; i < pack_size; ++i) add_reg[i] = aiter::upcast_s(tmp_smem[pack_size * threadIdx.x + i]);
      constexpr int smem_gpu_loop_stride = tnum_gpu * pack_size;
#pragma unroll
      for (int i = 1; i < ngpus; ++i) {
#pragma unroll
        for (int j = 0; j < pack_size; ++j)
          add_reg[j] += aiter::upcast_s(tmp_smem[i * smem_gpu_loop_stride + pack_size * threadIdx.x + j]);
      }
      P write_reg;
#pragma unroll
      for (int i = 0; i < pack_size; ++i) write_reg[i] = aiter::downcast_s<T>(add_reg[i]);
      tmp_out[idx - start] = write_reg;
    }
    __syncthreads();
  }
  if constexpr (has_tail) {
    if (warp_id == 0 && idx < end) tmp_out[idx - start] = aiter::packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  aiter::end_sync<ngpus, false>(sg, self_sg, rank);  // final_sync false: a release, which publishes tmp_out

  for (int idx = tid; idx < largest_part; idx += stride) {
    int dst_idx = (warp_id + rank) % ngpus * part + idx;
    union {
      P data;
      ui32x4_t i;
    } reduced;
    reduced.data = tmps[warp_id][idx];
    handoff_store(result, dst_idx, canonical(reduced.i));
  }
}

// ---- the reader: attn_res_hip._agg_kernel as Triton 3.7 compiles it for gfx950 ----
//
// Triton's layout: thread tid < 64 * kNw holds columns tid * 8 + rep * 512 * kNw + [0, 8), reps
// 8192 / (512 * kNw); columns past 7168 are zeros. A reduction over a row: per thread a balanced tree
// over its 2-wide vectors in register order (the first level's left product contracted into an FMA),
// the vector's two lanes, the DPP scan of each wave read from lane 63, then the waves' values through
// LDS with DPP quad / half-row permutes.

DINLINE float bf(uint16_t h) { return __builtin_bit_cast(float, uint32_t{h} << 16); }

DINLINE uint16_t to_bf16(float f) { return __builtin_bit_cast(uint16_t, static_cast<__bf16>(f)); }

#define K3_DPP(ctrl, row_mask, bound, old, src)                                                         \
  __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(__builtin_bit_cast(int, old),                   \
                                                        __builtin_bit_cast(int, src), ctrl, row_mask, 0xf, \
                                                        bound))

// row_shr 8 / 4 / 2 / 1, row_bcast 15 / 31, read lane 63
DINLINE float wave_sum(float x) {
  x = K3_DPP(0x118, 0xf, true, 0.f, x) + x;
  x = K3_DPP(0x114, 0xf, true, 0.f, x) + x;
  x = K3_DPP(0x112, 0xf, true, 0.f, x) + x;
  x = K3_DPP(0x111, 0xf, true, 0.f, x) + x;
  x = K3_DPP(0x142, 0xa, true, x, x) + x;
  x = K3_DPP(0x143, 0xf, true, 0.f, x) + x;
  return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, x), 63));
}

// the waves' totals of one value, each lane holding wave (lane % kNw)'s
template <int kNw>
DINLINE float warps_sum(float x) {
  if constexpr (kNw == 8) {
    float y = K3_DPP(0x141, 0xf, false, x, x);
    y = K3_DPP(0x1b, 0xf, false, y, y);
    x = x + y;
  }
  x = x + K3_DPP(0x4e, 0xf, false, x, x);
  x = x + K3_DPP(0xb1, 0xf, false, x, x);
  return x;
}

// in-thread tree of sum(a * b) over kE values (register order), then the vector's two lanes
template <int kE>
DINLINE float thread_dot(const float* a, const float* b) {
  constexpr int kL = kE / 4;
  float x[kL], y[kL];
#pragma unroll
  for (int m = 0; m < kL; ++m) {
    x[m] = __builtin_fmaf(a[4 * m], b[4 * m], a[4 * m + 2] * b[4 * m + 2]);
    y[m] = __builtin_fmaf(a[4 * m + 1], b[4 * m + 1], a[4 * m + 3] * b[4 * m + 3]);
  }
#pragma unroll
  for (int w = kL / 2; w >= 1; w /= 2) {
#pragma unroll
    for (int m = 0; m < w; ++m) {
      x[m] = x[2 * m] + x[2 * m + 1];
      y[m] = y[2 * m] + y[2 * m + 1];
    }
  }
  return x[0] + y[0];
}

template <int kNw, int kNvb>
struct Reader {
  static constexpr int kRPad = kNvb == 1 ? 1 : kNvb == 2 ? 2 : kNvb <= 4 ? 4 : 8;
  static constexpr int kReps = 8192 / (512 * kNw);
  static constexpr int kE = kReps * 8;
  static constexpr int kSlots = 2 * kRPad + 2;

  struct Shared {
    float part[kNw][kSlots];
    float sq[kNw];
  };

  // the columns of rep k for thread tid, and whether they exist
  DINLINE static int col(int tid, int k) { return tid * 8 + k * 512 * kNw; }

  DINLINE static void run(const Read& rd, int t, Shared& s) {
    const int tid = threadIdx.x;
    const bool math = tid < 64 * kNw;
    const int wave = tid / 64;
    const int lane = tid % 64;

    K3_STAMP(rd, 4 * t);
    float res[kE] = {}, cw[kE] = {}, ow[kE] = {}, tile[kRPad][kE] = {};
    if (math) {
#pragma unroll
      for (int k = 0; k < kReps; ++k) {
        const int c = col(tid, k);
        if (c >= kHidden) continue;
        uint4 v;
        if (rd.res != nullptr) {
          v = *reinterpret_cast<const uint4*>(rd.res + int64_t{t} * kHidden + c);
          unpack(v, res + 8 * k);
        }
        const float4 c0 = *reinterpret_cast<const float4*>(rd.cw + c);
        const float4 c1 = *reinterpret_cast<const float4*>(rd.cw + c + 4);
        cw[8 * k + 0] = c0.x, cw[8 * k + 1] = c0.y, cw[8 * k + 2] = c0.z, cw[8 * k + 3] = c0.w;
        cw[8 * k + 4] = c1.x, cw[8 * k + 5] = c1.y, cw[8 * k + 6] = c1.z, cw[8 * k + 7] = c1.w;
        v = *reinterpret_cast<const uint4*>(rd.ow + c);
        unpack(v, ow + 8 * k);
#pragma unroll
        for (int r = 0; r < kNvb; ++r) {
          v = *reinterpret_cast<const uint4*>(rd.bank + t * rd.bank_stride_t + r * rd.bank_stride_b + c);
          unpack(v, tile[r] + 8 * k);
        }
      }
    }

    // the bank rows' scores, while the all-reduce runs
    if (math) {
#pragma unroll
      for (int r = 0; r < kRPad; ++r) {
        const float dot = wave_sum(thread_dot<kE>(tile[r], cw));
        const float sq = wave_sum(thread_dot<kE>(tile[r], tile[r]));
        if (lane == 0) s.part[wave][2 * r] = dot, s.part[wave][2 * r + 1] = sq;
      }
    }

    K3_STAMP(rd, 4 * t + 1);
    // this row of the sum: every slot's poll in flight at once, re-arm, prefix
    float pv[kE] = {};
    if (math) {
      ui32x4_t got[kReps];
      poll_row(handoff_rsrc(rd.handoff), int64_t{t} * kRowPacks, tid, got);
#pragma unroll
      for (int k = 0; k < kReps; ++k) {
        const int c = col(tid, k);
        if (c >= kHidden) continue;
        union {
          ui32x4_t i;
          uint4 u;
        } a;
        a.i = got[k];
        float sum[8];
        unpack(a.u, sum);
        uint16_t p[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          p[e] = rd.res != nullptr ? to_bf16(res[8 * k + e] + sum[e]) : to_bf16(sum[e]);
          pv[8 * k + e] = bf(p[e]);
        }
        const uint4 pk = pack(p);
        *reinterpret_cast<uint4*>(rd.prefix_out + int64_t{t} * kHidden + c) = pk;
        if (rd.write_bank)
          *reinterpret_cast<uint4*>(rd.bank + t * rd.bank_stride_t + kNvb * rd.bank_stride_b + c) = pk;
      }
      const float dot = wave_sum(thread_dot<kE>(pv, cw));
      const float sq = wave_sum(thread_dot<kE>(pv, pv));
      if (lane == 0) s.part[wave][2 * kRPad] = dot, s.part[wave][2 * kRPad + 1] = sq;
    }
    __syncthreads();
    K3_STAMP(rd, 4 * t + 2);

    const int w = lane & (kNw - 1);
    const float p_dot = warps_sum<kNw>(s.part[w][2 * kRPad]);
    const float p_sq = warps_sum<kNw>(s.part[w][2 * kRPad + 1]);
    const float p_score = p_dot / __builtin_amdgcn_sqrtf(rd.score_eps + p_sq / 7168.0f);
    float b_score[kRPad];
    float m = -INFINITY;
#pragma unroll
    for (int r = 0; r < kRPad; ++r) {
      const float dot = warps_sum<kNw>(s.part[w][2 * r]);
      const float sq = warps_sum<kNw>(s.part[w][2 * r + 1]);
      b_score[r] = dot / __builtin_amdgcn_sqrtf(rd.score_eps + sq / 7168.0f);
      if (r < kNvb) m = __builtin_fmaxf(m, b_score[r]);
    }
    m = __builtin_fmaxf(m, p_score);
    float b_w[kRPad];
#pragma unroll
    for (int r = 0; r < kRPad; ++r) b_w[r] = r < kNvb ? __builtin_exp2f((b_score[r] - m) * kLog2e) : 0.f;
    const float p_w = __builtin_exp2f((p_score - m) * kLog2e);
    float bsum;
    if constexpr (kRPad == 1) {
      bsum = b_w[0];
    } else if constexpr (kRPad == 2) {
      bsum = b_w[0] + b_w[1];
    } else if constexpr (kRPad == 4) {
      bsum = (b_w[0] + b_w[2]) + (b_w[1] + b_w[3]);
    } else {
      bsum = ((b_w[0] + b_w[4]) + (b_w[2] + b_w[6])) + ((b_w[1] + b_w[5]) + (b_w[3] + b_w[7]));
    }
    const float inv = 1.0f / (p_w + bsum);
    const float acc_s = p_w * inv;
    float cr[kRPad];
#pragma unroll
    for (int r = 0; r < kRPad; ++r) cr[r] = b_w[r] * inv;

    float acc[kE];
#pragma unroll
    for (int e = 0; e < kE; ++e) {
      float mix;
      if constexpr (kRPad == 1) {
        acc[e] = __builtin_fmaf(acc_s, pv[e], cr[0] * tile[0][e]);
        continue;
      } else if constexpr (kRPad == 2) {
        mix = __builtin_fmaf(cr[0], tile[0][e], cr[1] * tile[1][e]);
      } else {
        float q[kRPad / 2];
#pragma unroll
        for (int r = 0; r < kRPad / 2; ++r)
          q[r] = __builtin_fmaf(cr[2 * r], tile[2 * r][e], cr[2 * r + 1] * tile[2 * r + 1][e]);
#pragma unroll
        for (int n = kRPad / 4; n >= 1; n /= 2) {
#pragma unroll
          for (int r = 0; r < n; ++r) q[r] = q[2 * r] + q[2 * r + 1];
        }
        mix = q[0];
      }
      acc[e] = __builtin_fmaf(acc_s, pv[e], mix);
    }
    if (math) {
      const float sq = wave_sum(thread_dot<kE>(acc, acc));
      if (lane == 0) s.sq[wave] = sq;
    }
    __syncthreads();
    const float total = warps_sum<kNw>(s.sq[w]);
    const float scale = 1.0f / __builtin_amdgcn_sqrtf(rd.out_eps + total / 7168.0f);
    if (math) {
#pragma unroll
      for (int k = 0; k < kReps; ++k) {
        const int c = col(tid, k);
        if (c >= kHidden) continue;
        uint16_t o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] = to_bf16(acc[8 * k + e] * scale * ow[8 * k + e]);
        *reinterpret_cast<uint4*>(rd.out + int64_t{t} * kHidden + c) = pack(o);
      }
    }
    K3_STAMP(rd, 4 * t + 3);
  }

  // this thread's packs of a hand-off row until none holds a sentinel half (lamportPollSlot on all of
  // them at once), then re-armed
  DINLINE static void poll_row(__amdgpu_buffer_rsrc_t h, int64_t row, int tid, ui32x4_t (&got)[kReps]) {
    uint32_t wait = 0;
#pragma unroll
    for (int k = 0; k < kReps; ++k) wait |= col(tid, k) < kHidden ? 1u << k : 0u;
    while (wait != 0) {
      asm volatile("" ::: "memory");
#pragma unroll
      for (int k = 0; k < kReps; ++k)
        if (wait & (1u << k)) got[k] = handoff_load(h, row + col(tid, k) / 8);
#pragma unroll
      for (int k = 0; k < kReps; ++k) {
        if (!(wait & (1u << k))) continue;
        uint32_t m = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) m |= aiter::lamportSentinelMask(got[k][j]);
        if (m == 0) wait &= ~(1u << k);
      }
    }
    const ui32x4_t sentinel{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
#pragma unroll
    for (int k = 0; k < kReps; ++k)
      if (col(tid, k) < kHidden) handoff_store(h, row + col(tid, k) / 8, sentinel);
  }

  DINLINE static void unpack(uint4 v, float* f) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f[2 * i] = __builtin_bit_cast(float, w[i] << 16);
      f[2 * i + 1] = __builtin_bit_cast(float, w[i] & 0xffff0000u);
    }
  }

  DINLINE static uint4 pack(const uint16_t* h) {
    return make_uint4(h[0] | uint32_t{h[1]} << 16, h[2] | uint32_t{h[3]} << 16, h[4] | uint32_t{h[5]} << 16,
                      h[6] | uint32_t{h[7]} << 16);
  }
};

template <int kAlgo, int kNvb>
__global__ __launch_bounds__(kThreads, 1) void seam_kernel(const Exchange ex, const Read rd) {
  if (static_cast<int>(blockIdx.x) < ex.n_ar) {
    if constexpr (kAlgo == kTwoShot) {
      two_shot(ex, handoff_rsrc(rd.handoff));
    } else {
      two_stage<kAlgo == kTwoStageTail>(ex, handoff_rsrc(rd.handoff));
    }
    K3_STAMP(rd, 4 * kMaxRows + blockIdx.x);
    return;
  }
  constexpr int kNw = kNvb >= 5 ? 8 : 4;
  using R = Reader<kNw, kNvb>;
  __shared__ typename R::Shared s;
  R::run(rd, static_cast<int>(blockIdx.x) - ex.n_ar, s);
}

template <int kAlgo>
void launch_nvb(int nvb, int grid, const Exchange& ex, const Read& rd, hipStream_t stream) {
  switch (nvb) {
#define K3_CASE(N)                                                         \
  case N:                                                                  \
    seam_kernel<kAlgo, N><<<grid, kThreads, 0, stream>>>(ex, rd);          \
    break;
    K3_CASE(1) K3_CASE(2) K3_CASE(3) K3_CASE(4) K3_CASE(5) K3_CASE(6) K3_CASE(7) K3_CASE(8)
#undef K3_CASE
    default:
      TORCH_CHECK(false, "ar_seam: 1-8 bank rows, not ", nvb);
  }
}

// aiter's choice for a bf16 message of rows x 7168 at TP8 (CustomAllreduce::lamportKernel, select_ar_stage):
// the two-shot up to 1 MiB, else the 2-stage on min(80, ...) blocks.
struct Plan {
  int algo, n_ar, count;
};

Plan plan(int rows) {
  const int64_t bytes = int64_t{rows} * kHidden * 2;
  const int64_t packs = bytes / 16;
  if (bytes <= (int64_t{1} << 20) && bytes % (16 * kWorld) == 0) {
    const int part = static_cast<int>(packs / kWorld);
    return {kTwoShot, (part + aiter::kLamportWaveSize - 1) / aiter::kLamportWaveSize, part};
  }
  const int size = static_cast<int>(packs);
  const int tnum_gpu = THREAD_NUM / kWorld;
  const int n_ar = std::min(aiter::kMaxBlocks, (size / kWorld + tnum_gpu - 1) / tnum_gpu);
  return {size % THREAD_NUM != 0 ? kTwoStageTail : kTwoStage, n_ar, size};
}

void launch(const Exchange& ex0, const Read& rd, int rows, int nvb, hipStream_t stream) {
  const Plan pl = plan(rows);
  Exchange ex = ex0;
  ex.count = pl.count;
  ex.n_ar = pl.n_ar;
  const int grid = pl.n_ar + rows;
  if (pl.algo == kTwoShot) {
    launch_nvb<kTwoShot>(nvb, grid, ex, rd, stream);
  } else if (pl.algo == kTwoStage) {
    launch_nvb<kTwoStage>(nvb, grid, ex, rd, stream);
  } else {
    launch_nvb<kTwoStageTail>(nvb, grid, ex, rd, stream);
  }
  C10_HIP_CHECK(hipGetLastError());
}

Read make_read(const at::Tensor& res, const at::Tensor& bank, const at::Tensor& cw, const at::Tensor& ow,
               const at::Tensor& prefix_out, const at::Tensor& out, const at::Tensor& handoff, int64_t nvb,
               bool has_res, bool write_bank, double score_eps, double out_eps) {
  const int64_t rows = out.size(0);
  TORCH_CHECK(rows >= 1 && rows <= kMaxRows, "ar_seam: 1-", kMaxRows, " rows");
  TORCH_CHECK(out.size(1) == kHidden && out.is_contiguous() && prefix_out.is_contiguous(), "ar_seam: [T, 7168] out");
  TORCH_CHECK(bank.dim() == 3 && bank.size(0) == rows && bank.size(2) == kHidden && bank.stride(2) == 1, "bank");
  TORCH_CHECK(bank.stride(0) % 8 == 0 && bank.stride(1) % 8 == 0, "bank rows 16-byte aligned");
  TORCH_CHECK(nvb >= 1 && nvb <= 8 && nvb + (write_bank ? 1 : 0) <= bank.size(1), "bank rows");
  TORCH_CHECK(cw.scalar_type() == at::kFloat && ow.scalar_type() == at::kBFloat16, "cw fp32, ow bf16");
  TORCH_CHECK(handoff.numel() >= kHandoffBytes, "hand-off buffer");
  TORCH_CHECK(!has_res || (res.is_contiguous() && res.size(0) == rows), "residual");
  return Read{
      has_res ? static_cast<const uint16_t*>(res.data_ptr()) : nullptr,
      static_cast<uint16_t*>(bank.data_ptr()),
      cw.data_ptr<float>(),
      static_cast<const uint16_t*>(ow.data_ptr()),
      static_cast<uint16_t*>(prefix_out.data_ptr()),
      static_cast<uint16_t*>(out.data_ptr()),
      static_cast<ui32x4_t*>(handoff.data_ptr()),
      bank.stride(0),
      bank.stride(1),
      static_cast<float>(score_eps),
      static_cast<float>(out_eps),
      write_bank ? 1 : 0,
#ifdef AR_SEAM_STAMPS
      static_cast<uint64_t*>(handoff.data_ptr()) + kHandoffBytes / 8,
#endif
  };
}

// The fused launch on aiter's communicator ``comm`` (a CustomAllreduce*): this rank's o_proj partial
// ``inp`` [T, 7168] bf16 is the message. Outside graph capture an input the 2-stage needs registered
// is first copied into aiter's registered pool (reg_inp_ptr, as aiter's all_reduce does); in capture
// reg_inp_ptr is 0 and the input is recorded for aiter's graph-buffer registration.
void seam(int64_t comm, const at::Tensor& inp, const at::Tensor& res, const at::Tensor& bank, const at::Tensor& cw,
          const at::Tensor& ow, at::Tensor& prefix_out, at::Tensor& out, const at::Tensor& handoff, int64_t nvb,
          bool has_res, bool write_bank, double score_eps, double out_eps, int64_t reg_inp_ptr,
          int64_t reg_inp_bytes) {
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(inp.device());
  const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(inp.device().index()).stream();
  auto* fa = reinterpret_cast<aiter::CustomAllreduce*>(comm);
  TORCH_CHECK(fa->world_size_ == kWorld, "ar_seam: TP8");
  TORCH_CHECK(inp.scalar_type() == at::kBFloat16 && inp.is_contiguous() && inp.size(1) == kHidden, "input");
  const int rows = static_cast<int>(inp.size(0));
  const Read rd = make_read(res, bank, cw, ow, prefix_out, out, handoff, nvb, has_res, write_bank, score_eps, out_eps);
  Exchange ex{};
  ex.rank = fa->rank_;
  if (plan(rows).algo == kTwoShot) {
    TORCH_CHECK(fa->lamport_ != nullptr, "ar_seam: aiter's lamport path is off");
    ex.input = static_cast<const T*>(inp.data_ptr());
    ex.peers = fa->lamport_->two_shot_peers;
    ex.round = fa->lamport_->d_two_shot_round;
  } else {
    void* src = inp.data_ptr();
    if (reg_inp_ptr != 0) {
      const int64_t bytes = inp.numel() * 2;
      TORCH_CHECK(bytes <= reg_inp_bytes, "registered buffer is too small to contain the input");
      C10_HIP_CHECK(hipMemcpyAsync(reinterpret_cast<void*>(reg_inp_ptr), src, bytes, hipMemcpyDeviceToDevice, stream));
      src = reinterpret_cast<void*>(reg_inp_ptr);
    }
    ex.inputs = fa->get_buffer_RD(stream, src);
    ex.sg = fa->sg_;
    ex.self_sg = fa->self_sg_;
  }
  launch(ex, rd, rows, static_cast<int>(nvb), stream);
}

// The same launch on explicit exchange state (tests): ``state`` int64 [32] on the host holds
// [0] input, [1..8] two-shot inbox bases, [9] round counters, [10] RankData*, [11..18] signals,
// [19] self signal, [20] rank.
void seam_explicit(const at::Tensor& state, int64_t rows, const at::Tensor& res, const at::Tensor& bank,
                   const at::Tensor& cw, const at::Tensor& ow, at::Tensor& prefix_out, at::Tensor& out,
                   const at::Tensor& handoff, int64_t nvb, bool has_res, bool write_bank, double score_eps,
                   double out_eps) {
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(out.device());
  const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(out.device().index()).stream();
  TORCH_CHECK(state.device().is_cpu() && state.scalar_type() == at::kLong && state.numel() >= 21, "state");
  const int64_t* s = state.data_ptr<int64_t>();
  Exchange ex{};
  ex.input = reinterpret_cast<const T*>(s[0]);
  for (int i = 0; i < kWorld; ++i) ex.peers.ptrs[i] = reinterpret_cast<void*>(s[1 + i]);
  ex.round = reinterpret_cast<uint32_t*>(s[9]);
  ex.inputs = reinterpret_cast<RankData*>(s[10]);
  for (int i = 0; i < kWorld; ++i) ex.sg.signals[i] = reinterpret_cast<Signal*>(s[11 + i]);
  ex.self_sg = reinterpret_cast<Signal*>(s[19]);
  ex.rank = static_cast<int>(s[20]);
  const Read rd = make_read(res, bank, cw, ow, prefix_out, out, handoff, nvb, has_res, write_bank, score_eps, out_eps);
  TORCH_CHECK(out.size(0) == rows, "rows");
  launch(ex, rd, static_cast<int>(rows), static_cast<int>(nvb), stream);
}

// [algo, n_ar, count] of aiter's all-reduce for ``rows``, then sizeof(aiter::Signal) (a 2-stage tmp
// buffer follows its signal)
std::vector<int64_t> plan_of(int64_t rows) {
  const Plan p = plan(static_cast<int>(rows));
  return {p.algo, p.n_ar, p.count, static_cast<int64_t>(sizeof(Signal))};
}

}  // namespace k3_ar_seam

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def(
      "seam(int comm, Tensor inp, Tensor res, Tensor(a!) bank, Tensor cw, Tensor ow, Tensor(b!) prefix_out, "
      "Tensor(c!) out, Tensor(d!) handoff, int nvb, bool has_res, bool write_bank, float score_eps, "
      "float out_eps, int reg_inp_ptr, int reg_inp_bytes) -> ()");
  m.impl("seam", at::kCUDA, &k3_ar_seam::seam);
  m.def(
      "seam_explicit(Tensor state, int rows, Tensor res, Tensor(a!) bank, Tensor cw, Tensor ow, "
      "Tensor(b!) prefix_out, Tensor(c!) out, Tensor(d!) handoff, int nvb, bool has_res, bool write_bank, "
      "float score_eps, float out_eps) -> ()");
  m.impl("seam_explicit", at::kCUDA, &k3_ar_seam::seam_explicit);
  m.def("plan_of(int rows) -> int[]");
  m.impl("plan_of", &k3_ar_seam::plan_of);
}
