// SPDX-License-Identifier: Apache-2.0
// The one-launch top-k (topk_one.cuh) with context-parallel rows, its own module: past a band of
// columns every rank of a TP group selects in its run of a row's chunks, and every rank merges all
// the ranks' picks. The module's other launches are topk_one.cuh's, unchanged.
#include "topk_one.cuh"

namespace sglang {
namespace topk_one {

// ---------------------------------------------------------------------------------------------
// Context-parallel rows. From `band` columns every rank of the TP group scores and selects one
// run of a row's 128-column chunks (`index_plan.cp_chunks`); each rank's top k of its run goes to
// every rank as 8-byte cells, and each rank takes the row's top k of all of them. A cell's high
// word orders the value as SGLang's exact radix key does, its low word the column, lower first,
// so the keys are distinct and their top k is the row's top k by (value, then lower column).
// ---------------------------------------------------------------------------------------------

/// A launch's exchange: this rank's cells [parity][problem][source rank][row][kMaxTopK] (left
/// zeroed: a rank zeroes each cell it reads), every rank's cells by rank, the rows' own picks,
/// and each row's parity, flipped at each of its exchanges. A rank's next exchange of a row
/// writes the other parity's cells, so it cannot reach a peer still reading this one's.
struct Cp {
  uint64_t* __restrict__ cells;
  const uint64_t* __restrict__ peers;  // [tp] addresses of every rank's cells
  int32_t* __restrict__ picks;         // [kMaxProblems][kMaxRows][kMaxTopK]
  uint32_t* __restrict__ parity;       // [kMaxProblems][kMaxRows]
  uint32_t rank, tp, band;
};

constexpr uint32_t kCpChunk = 128;   // index_plan.CHUNK
constexpr uint32_t kCpScored = 512;  // index_plan.TOPK: rows at or below it are not scored
constexpr uint32_t kCpMaxRanks = 8;
constexpr uint32_t kCpSpinLimit = 1u << 24;

/// This rank's columns [lo, hi) of a row seeing `bound` columns; `on` when the row is
/// context-parallel (index_plan.cp_chunks), else the whole row.
struct CpWindow {
  uint32_t lo, hi;
  bool on;
};
SGL_DEVICE CpWindow cp_window(uint32_t bound, const Cp& cp) {
  if (bound < cp.band) return {0, bound, false};
  const uint32_t n = bound > kCpScored ? (bound + kCpChunk - 1) / kCpChunk : 0;
  const uint32_t per = (n + cp.tp - 1) / cp.tp;
  const uint32_t lo = min(cp.rank * per, n);
  return {min(lo * kCpChunk, bound), min(min(lo + per, n) * kCpChunk, bound), true};
}

/// A pick's cell: SGLang's exact key of its value, then its column, lower first. Never zero.
SGL_DEVICE uint64_t cp_cell(float v, uint32_t col) {
  return (static_cast<uint64_t>(impl::extract_exact_bin(v)) << 32) | (0x80000000u | (0x7FFFFFFFu - col));
}
/// A cell for no pick (`i` keeps it distinct): below every pick's.
SGL_DEVICE uint64_t cp_none(uint32_t i) {
  return 0x80000000u | i;
}
SGL_DEVICE int32_t cp_column(uint64_t cell) {
  return (cell >> 32) == 0 ? -1 : static_cast<int32_t>(0x7FFFFFFFu - (static_cast<uint32_t>(cell) & 0x7FFFFFFFu));
}
constexpr uint32_t kCpDigitBits = 11;
constexpr uint32_t kCpBins = 1u << kCpDigitBits;
constexpr uint32_t kCpWaves = kBlockSize / 64;

struct CpSmem {
  uint32_t hist[kCpBins];
  uint32_t wave_sum[kCpWaves];
  uint32_t bin, above, inbin, count;
  unsigned long long lo, hi;
};

/// A context-parallel row's LDS: the selection's, then (once it is dead) the merge's keys and picks.
union CpLds {
  impl::MaxSmem<Register4::Smem, Streaming::Smem, TopKSplit::Smem> select;
  struct {
    CpSmem m;
    uint64_t keys[kCpMaxRanks * kMaxTopK];
    uint64_t taken[kMaxTopK];
  } merge;
};
SGL_DEVICE CpLds& cp_lds() {
  __shared__ CpLds lds;
  return lds;
}

SGL_DEVICE size_t cp_slot(const Cp& cp, uint32_t q, uint32_t p, uint32_t src, uint32_t row) {
  return (((static_cast<size_t>(q) * kMaxProblems + p) * cp.tp + src) * kMaxRows + row) * kMaxTopK;
}

/// SGLang's top k of columns [w.lo, w.hi) of `params`' row into `picks` (as the row's columns, -1
/// past the window's). True on the CTA that ends the row, once every pick is in.
SGL_DEVICE bool cp_select(
    const TopKPagedParams& params, const Workspace& ws, int32_t* picks, CpWindow w, uint32_t row, uint32_t slot,
    uint32_t rank) {
  auto problem = params.problem(row, w.hi - w.lo);
  problem.in += w.lo;
  problem.bias = static_cast<int32_t>(w.lo);
  problem.out = picks;
  auto& smem = cp_lds().select;
  if (problem.seq_len <= ws.floor) {
    if (rank != 0) return false;
    if (problem.seq_len <= problem.topk) {
      for (uint32_t t = threadIdx.x; t < problem.topk; t += kBlockSize)
        picks[t] = t < problem.seq_len ? static_cast<int32_t>(w.lo + t) : -1;
    } else if (problem.seq_len <= kReg4MaxSeqLen) {
      Register4::forward<false>(problem, &smem);
    } else {
      Streaming::forward<false>(problem, &smem);
    }
    drain();
    __syncthreads();
    return true;
  }
  auto* const sm = reinterpret_cast<Split::Smem*>(&smem);
  auto* const ctr = &ws.ctr[slot];
  auto* const ties = ws.ties + static_cast<size_t>(slot) * TopKSplit::kMaxNumTie;
  auto* const row_hist = ws.hist + static_cast<size_t>(slot) * TopKSplit::kHistSize;
  const auto chunk = TopKSplit::chunk_of(problem.seq_len, rank, kSplit);
  if (threadIdx.x == 0) l2_or(&ctr->ranks, 1u << rank);
  Split::histogram(problem, chunk, row_hist, sm);
  Split::barrier(&ctr->arrive_hist);
  Split::select(problem, chunk, row_hist, ctr, ties, sm);
  if (!Split::arrive_last(ctr, sm)) return false;
  Split::clear_row(row_hist);
  Split::finish_ties(problem, ties, sm);
  drain();
  __syncthreads();
  return true;
}

/// A pick, read past this CU's cache: another CTA of the row may have written it.
SGL_DEVICE int32_t cp_pick(const int32_t* picks, uint32_t t) {
  return __hip_atomic_load(const_cast<int32_t*>(picks + t), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

/// Selection `p` of a context-parallel row: this rank's `k` picks (columns of `scores_row`) go
/// to every rank, then the row's top k of every rank's cells is `emit(pos, cell)`, pos 0 .. k - 1.
/// The cells sit in LDS; a key leaves the set (taken or dropped) as 0.
template <typename Emit>
SGL_DEVICE void cp_merge(
    const Cp& cp, const int32_t* picks, const float* scores_row, uint32_t p, uint32_t row, uint32_t k,
    const Emit& emit) {
  auto& lds = cp_lds().merge;
  auto& m = lds.m;
  uint64_t* const keys = lds.keys;
  uint64_t* const taken = lds.taken;
  const auto tx = threadIdx.x;
  uint32_t* const parity = cp.parity + p * kMaxRows + row;
  const uint32_t q = *parity & 1u;
  const size_t mine = cp_slot(cp, q, p, cp.rank, row);
  // push: this rank's picks as cells to every rank, its own included (k <= 2 a thread)
  static_assert(kMaxTopK <= 2 * kBlockSize);
  int32_t pick[2];
  float val[2];
#pragma unroll
  for (uint32_t u = 0; u < 2; ++u) {
    const uint32_t t = tx + u * kBlockSize;
    pick[u] = t < k ? cp_pick(picks, t) : -1;
  }
#pragma unroll
  for (uint32_t u = 0; u < 2; ++u) val[u] = pick[u] >= 0 ? scores_row[pick[u]] : 0.0f;
#pragma unroll 1
  for (uint32_t d = 0; d < cp.tp; ++d) {
    auto* const base = reinterpret_cast<uint64_t*>(cp.peers[d]) + mine;
#pragma unroll
    for (uint32_t u = 0; u < 2; ++u) {
      const uint32_t t = tx + u * kBlockSize;
      if (t >= k) continue;
      const uint64_t cell = pick[u] >= 0 ? cp_cell(val[u], static_cast<uint32_t>(pick[u])) : cp_none(cp.rank * k + t);
      __hip_atomic_store(base + t, cell, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
  // every rank's cells: one load of each in flight together, then a poll of the ones not in yet
  const uint32_t n = cp.tp * k;
  constexpr uint32_t kBatch = 4;
#pragma unroll 1
  for (uint32_t j0 = tx; j0 < n; j0 += kBatch * kBlockSize) {
    uint64_t v[kBatch];
#pragma unroll
    for (uint32_t u = 0; u < kBatch; ++u) {
      const uint32_t j = j0 + u * kBlockSize;
      v[u] = j < n ? __hip_atomic_load(cp.cells + cp_slot(cp, q, p, j / k, row) + j % k, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM)
                   : 0;
    }
#pragma unroll
    for (uint32_t u = 0; u < kBatch; ++u)
      if (j0 + u * kBlockSize < n) keys[j0 + u * kBlockSize] = v[u];
  }
#pragma unroll 1
  for (uint32_t j = tx; j < n; j += kBlockSize) {
    auto* const at = cp.cells + cp_slot(cp, q, p, j / k, row) + j % k;
    uint64_t key = keys[j];
    uint32_t spins = 0;
    while (key == 0) {
      __builtin_amdgcn_s_sleep(1);
      if (++spins == kCpSpinLimit) __builtin_trap();
      key = __hip_atomic_load(at, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    keys[j] = key;
    __hip_atomic_store(at, uint64_t{0}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);  // for the launch after next
  }
  // the top k of the n distinct keys: each round takes kCpDigitBits from the live keys' highest differing bit
  uint32_t remain = k;
  const uint32_t lane = tx % 64, wave = tx / 64;
  const uint64_t below = (uint64_t{1} << lane) - 1;
  if (tx == 0) {
    m.count = 0;
    m.lo = ~0ull;
    m.hi = 0;
  }
  __syncthreads();
  // the live range: here from every key, then from the keys a round keeps
  auto range = [&](unsigned long long lo, unsigned long long hi) {
#pragma unroll
    for (uint32_t o = 32; o > 0; o >>= 1) {
      lo = min(lo, static_cast<unsigned long long>(__shfl_xor(lo, o, 64)));
      hi = max(hi, static_cast<unsigned long long>(__shfl_xor(hi, o, 64)));
    }
    if (lane == 0) {
      atomicMin(&m.lo, lo);
      atomicMax(&m.hi, hi);
    }
  };
  {
    unsigned long long lo = ~0ull, hi = 0;
#pragma unroll 1
    for (uint32_t j = tx; j < n; j += kBlockSize) {
      lo = min(lo, static_cast<unsigned long long>(keys[j]));
      hi = max(hi, static_cast<unsigned long long>(keys[j]));
    }
    range(lo, hi);
  }
#pragma unroll 1
  while (true) {
    for (uint32_t b = tx; b < kCpBins; b += kBlockSize) m.hist[b] = 0;
    __syncthreads();
    const unsigned long long diff = m.hi ^ m.lo;
    const uint32_t top = diff == 0 ? 0 : 63 - __clzll(diff);  // the highest differing bit
    const uint32_t shift = top >= kCpDigitBits - 1 ? top - (kCpDigitBits - 1) : 0;
#pragma unroll 1
    for (uint32_t j = tx; j < n; j += kBlockSize) {
      const uint64_t key = keys[j];
      if (key != 0) atomicAdd(&m.hist[(key >> shift) & (kCpBins - 1)], 1u);
    }
    __syncthreads();
    // bins from the top, two a thread: thread t holds bins kCpBins - 1 - 2t and the one below it
    const uint32_t h0 = m.hist[kCpBins - 1 - 2 * tx], h1 = m.hist[kCpBins - 2 - 2 * tx];
    uint32_t upto = h0 + h1;
#pragma unroll
    for (uint32_t o = 1; o < 64; o <<= 1) {
      const uint32_t v = __shfl_up(upto, o, 64);
      if (lane >= o) upto += v;
    }
    if (lane == 63) m.wave_sum[wave] = upto;
    if (tx == 0) {
      m.lo = ~0ull;
      m.hi = 0;
    }
    __syncthreads();
    for (uint32_t w = 0; w < wave; ++w) upto += m.wave_sum[w];
    // upto counts the keys at or above bin kCpBins - 2 - 2t
    const uint32_t at1 = upto, at0 = upto - h1;
    if (at0 - h0 < remain && at0 >= remain) {
      m.bin = kCpBins - 1 - 2 * tx;
      m.above = at0 - h0;
      m.inbin = h0;
    } else if (at1 - h1 < remain && at1 >= remain) {
      m.bin = kCpBins - 2 - 2 * tx;
      m.above = at1 - h1;
      m.inbin = h1;
    }
    __syncthreads();
    const uint32_t bin = m.bin;
    remain -= m.above;
    const bool all = m.inbin == remain;  // the bin's keys all make it
    unsigned long long lo = ~0ull, hi = 0;
#pragma unroll 1
    for (uint32_t j0 = 0; j0 < n; j0 += kBlockSize) {
      const uint32_t j = j0 + tx;
      const uint64_t key = j < n ? keys[j] : 0;
      const uint32_t d = (key >> shift) & (kCpBins - 1);
      const bool live = key != 0;
      const bool take = live && (d > bin || (d == bin && all));
      // a wave's takes get consecutive places, one LDS atomic a wave
      const uint64_t mask = __ballot(take);
      uint32_t base = 0;
      if (mask != 0 && lane == static_cast<uint32_t>(__ffsll(static_cast<long long>(mask)) - 1))
        base = atomicAdd(&m.count, static_cast<uint32_t>(__popcll(mask)));
      base = __shfl(base, __ffsll(static_cast<long long>(mask)) - 1, 64);
      if (take) taken[base + __popcll(mask & below)] = key;
      if (live && (take || d < bin)) {
        keys[j] = 0;
      } else if (live) {
        lo = min(lo, static_cast<unsigned long long>(key));
        hi = max(hi, static_cast<unsigned long long>(key));
      }
    }
    if (all) break;
    range(lo, hi);
    __syncthreads();
  }
  __syncthreads();
  for (uint32_t pos = tx; pos < k; pos += kBlockSize) emit(pos, taken[pos]);
  if (tx == 0) *parity = q ^ 1u;
}

/// A row's columns: SGLang's paged top-k with both outputs, context-parallel past the band.
SGL_DEVICE void cp_cols(const TopKPagedParams& params, const Workspace& ws, const Cp& cp, uint32_t row, uint32_t rank) {
  const auto w = cp_window(static_cast<uint32_t>(params.seq_lens[row]), cp);
  int32_t* const picks = cp.picks + static_cast<size_t>(row) * kMaxTopK;
  if (!cp_select(params, ws, picks, w, row, row, rank)) return;
  const auto tr = params.template get_transform<TopKMode::DUAL_OUTPUT>(row);
  int32_t* const out = params.get_output_ptr(row);
  if (!w.on) {
    for (uint32_t t = threadIdx.x; t < params.topk; t += kBlockSize) {
      const int32_t pick = cp_pick(picks, t);
      out[t] = pick >= 0 ? tr.page_to_indices(pick) : -1;
      tr.raw_out[t] = pick;
    }
    return;
  }
  cp_merge(cp, picks, params.scores + row * params.score_stride, 0, row, params.topk, [&](uint32_t pos, uint64_t cell) {
    const int32_t col = cp_column(cell);
    out[pos] = col >= 0 ? tr.page_to_indices(col) : -1;
    tr.raw_out[pos] = col;
  });
}

/// A producer row's blocks (unpaged indices), their windows the 8-column blocks of its columns'.
SGL_DEVICE void cp_blocks(
    const TopKPagedParams& cols, const TopKPagedParams& blocks, const Workspace& ws, const Cp& cp, uint32_t row,
    uint32_t rank) {
  const auto c = cp_window(static_cast<uint32_t>(cols.seq_lens[row]), cp);
  // a non-empty run starts on a chunk, so the ranks' blocks do not overlap
  const auto w = !c.on        ? CpWindow{0, static_cast<uint32_t>(blocks.seq_lens[row]), false}
                 : c.hi > c.lo ? CpWindow{c.lo / 8, (c.hi + 7) / 8, true}
                               : CpWindow{0, 0, true};
  int32_t* const picks = cp.picks + static_cast<size_t>(kMaxRows + row) * kMaxTopK;
  if (!cp_select(blocks, ws, picks, w, row, kMaxRows + row, rank)) return;
  int32_t* const out = blocks.get_output_ptr(row);
  if (!w.on) {
    for (uint32_t t = threadIdx.x; t < blocks.topk; t += kBlockSize) out[t] = cp_pick(picks, t);
    return;
  }
  cp_merge(cp, picks, blocks.scores + row * blocks.score_stride, 1, row, blocks.topk, [&](uint32_t pos, uint64_t cell) {
    out[pos] = cp_column(cell);
  });
}

/// `topk_one` with context-parallel rows.
TOPK_KERNEL void topk_one_cp(const __grid_constant__ TopKPagedParams params, const Workspace ws, const Cp cp) {
  device::enable_smem_spilling();
  if (blockIdx.x == kXcc * kSplit) return check_presence(ws.presence, kRanks);
  const uint32_t rank = blockIdx.x / kXcc;
  const uint32_t row = xcc_row(ws, rank);
  if (row >= params.batch_size) return;
  cp_cols(params, ws, cp, row, rank);
}

/// `topk_pair` with context-parallel rows.
TOPK_KERNEL void topk_pair_cp(
    const __grid_constant__ TopKPagedParams cols,
    const __grid_constant__ TopKPagedParams blocks,
    const Workspace ws,
    const Cp cp) {
  device::enable_smem_spilling();
  if (blockIdx.x == kXcc * kSplit * 2) return check_presence(ws.presence, (1u << (2 * kSplit)) - 1u);
  const uint32_t rank = blockIdx.x / kXcc;
  const uint32_t row = xcc_row(ws, rank);
  if (row >= cols.batch_size) return;
  if (rank < kSplit) return cp_cols(cols, ws, cp, row, rank);
  cp_blocks(cols, blocks, ws, cp, row, rank - kSplit);
}

}  // namespace topk_one

struct TopKCp {
  /// The exchange of a context-parallel launch (`topk_one::Cp`): `picks` and `parity` the module's
  /// scratch, `cells` this rank's cells (an address), `peers` [tp] int64 every rank's on the device.
  static topk_one::Cp cp_of(
      const tvm::ffi::TensorView picks,
      const tvm::ffi::TensorView parity,
      const int64_t cells,
      const tvm::ffi::TensorView peers,
      const int64_t rank,
      const int64_t tp,
      const int64_t band) {
    using namespace host;
    RuntimeCheck(parity.size(0) == topk_one::kMaxProblems * topk_one::kMaxRows, "parity shape");
    RuntimeCheck(tp >= 2 && tp <= topk_one::kCpMaxRanks && rank >= 0 && rank < tp, "context-parallel rank of 2 to 8");
    RuntimeCheck(peers.size(0) == tp, "an address a rank");
    RuntimeCheck(picks.size(0) == topk_one::kMaxProblems * topk_one::kMaxRows * kMaxTopK, "picks shape");
    RuntimeCheck(band > 0 && band <= 0x7FFFFFFF, "band");
    return topk_one::Cp{
        .cells = reinterpret_cast<uint64_t*>(cells),
        .peers = static_cast<const uint64_t*>(peers.data_ptr()),
        .picks = static_cast<int32_t*>(picks.data_ptr()),
        .parity = static_cast<uint32_t*>(parity.data_ptr()),
        .rank = static_cast<uint32_t>(rank),
        .tp = static_cast<uint32_t>(tp),
        .band = static_cast<uint32_t>(band),
    };
  }

  /// `transform_paged`, each row seeing `band` columns or more selected context-parallel over
  /// the `tp` ranks (`cp_of`).
  static void transform_paged_cp(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_out,
      const tvm::ffi::TensorView raw_out,
      const int64_t page_size,
      const tvm::ffi::TensorView hist,
      const tvm::ffi::TensorView ctr,
      const tvm::ffi::TensorView ties,
      const tvm::ffi::TensorView picks,
      const tvm::ffi::TensorView parity,
      const int64_t cells,
      const tvm::ffi::TensorView peers,
      const int64_t rank,
      const int64_t tp,
      const int64_t band) {
    using namespace host;
    const auto params = TopKOne::params_of(
        scores, seq_lens, static_cast<const int32_t*>(page_table.data_ptr()), page_table.stride(0), page_out,
        static_cast<int32_t*>(raw_out.data_ptr()), page_size);
    const auto ws = TopKOne::workspace_of(hist, ctr, ties, params.batch_size);
    const auto cp = cp_of(picks, parity, cells, peers, rank, tp, band);
    LaunchKernel(topk_one::kXcc * topk_one::kSplit + 1, kBlockSize, scores.device())
        .config({.use_pdl = false})
        .launch(topk_one::topk_one_cp, params, ws, cp);
  }

  /// `transform_paged_pair`, context-parallel as `transform_paged_cp`: a row's blocks are the
  /// 8-column blocks of its columns'.
  static void transform_paged_pair_cp(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_out,
      const tvm::ffi::TensorView raw_out,
      const int64_t page_size,
      const tvm::ffi::TensorView block_scores,
      const tvm::ffi::TensorView block_lens,
      const tvm::ffi::TensorView block_out,
      const tvm::ffi::TensorView hist,
      const tvm::ffi::TensorView ctr,
      const tvm::ffi::TensorView ties,
      const tvm::ffi::TensorView picks,
      const tvm::ffi::TensorView parity,
      const int64_t cells,
      const tvm::ffi::TensorView peers,
      const int64_t rank,
      const int64_t tp,
      const int64_t band) {
    using namespace host;
    const auto cols = TopKOne::params_of(
        scores, seq_lens, static_cast<const int32_t*>(page_table.data_ptr()), page_table.stride(0), page_out,
        static_cast<int32_t*>(raw_out.data_ptr()), page_size);
    const auto blocks = TopKOne::params_of(block_scores, block_lens, nullptr, 0, block_out, nullptr, 1);
    RuntimeCheck(blocks.batch_size == cols.batch_size, "the pair's selections take the same rows");
    RuntimeCheck(page_size == 64, "a row's runs start on 128 columns, whole pages of 64");
    const auto ws = TopKOne::workspace_of(hist, ctr, ties, cols.batch_size);
    const auto cp = cp_of(picks, parity, cells, peers, rank, tp, band);
    LaunchKernel(topk_one::kXcc * topk_one::kSplit * 2 + 1, kBlockSize, scores.device())
        .config({.use_pdl = false})
        .launch(topk_one::topk_pair_cp, cols, blocks, ws, cp);
  }

};

}  // namespace sglang
