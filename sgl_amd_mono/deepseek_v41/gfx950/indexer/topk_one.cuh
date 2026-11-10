// SPDX-License-Identifier: Apache-2.0
// SGLang's DeepSeek-V4 paged top-k on ROCm (deepseek_v4/topk_v2.cuh, its split path) in one launch.
//
// SGLang splits a long row over CTAs in two launches: a histogram, then a select whose kernel
// boundary is the barrier. Here the row's CTAs meet at an in-launch counter barrier instead, all
// on one XCC: dispatch hands CTAs to the 8 XCCs round-robin, so CTAs b, b + 8, ... share one, and
// the row is that XCC's id. The row's cross-CTA traffic (histogram, counters, ties, its output read
// back) then stays in one L2, at workgroup scope, launch after launch. Every selection step is
// SGLang's code, called as is (histogram bins, threshold, ties, page transform), so the selected
// set is SGLang's; like SGLang's, the order within a row follows its atomics. Rows at or below the
// split floor take SGLang's one-block path. The row placement relies on that round-robin dispatch,
// and the launch checks it: every CTA reports its rank on its XCC, and one more CTA traps unless
// every XCC holds each rank exactly once; a barrier that never completes traps too.
//
// A candidate producer selects twice from one scoring pass: its top 512 columns and its top 2048
// 8-column blocks. The two are independent, so `topk_pair` runs both in one launch, each row's
// columns on ranks 0-9 of its XCC and its blocks on ranks 10-19, with workspace rows of their own.
#include <deepseek_v4/topk_v2.cuh>

// SGLang's top-k header takes an exactness mode (SGL_TOPK_MODE) since sglang#47960, and its
// zero-bin shortcut then applies in exact mode only; older headers have is_zero_bin.
#ifdef SGL_TOPK_MODE
#define TOPK_ONE_EXACT impl::kExact
#define TOPK_ONE_ZERO_BIN(v) (impl::kExact && zero_bin_shortcut(v))
#else
#define TOPK_ONE_EXACT true
#define TOPK_ONE_ZERO_BIN(v) is_zero_bin(v)
#endif

namespace sglang {
namespace topk_one {

constexpr uint32_t kSplit = 10;  // CTAs a long row; kXcc of them a dispatch round
constexpr uint32_t kXcc = 8;
constexpr uint32_t kMaxRows = kXcc;
constexpr uint32_t kMaxProblems = 2;  // topk_pair's columns and blocks
constexpr uint32_t kRanks = (1u << kSplit) - 1u;
constexpr uint32_t kSpinLimit = 1u << 20;

/// A row's counters, a cache line each. The row's last workgroup to arrive resets every one of
/// them, so each launch starts from zero.
struct alignas(128) Counters {
  uint32_t count_gt, count_eq, tie_nonzero, ranks, arrive_hist, arrive_sel;
  uint32_t pad[26];
};
static_assert(sizeof(Counters) == 128);

/// An XCC's ranks this launch, a cache line an XCC: each CTA adds its rank's bit, so the word is
/// kRanks only when the XCC holds every rank once. Left zeroed.
struct alignas(128) Presence {
  uint32_t ranks;
  uint32_t pad[31];
};
static_assert(sizeof(Presence) == 128);

struct Workspace {
  uint32_t* __restrict__ hist;        // [kMaxProblems * kMaxRows][kHistSize], left zeroed
  Counters* __restrict__ ctr;         // [kMaxProblems * kMaxRows]
  Presence* __restrict__ presence;    // [kXcc], after ctr's rows
  impl::TieValue* __restrict__ ties;  // [kMaxProblems * kMaxRows][kMaxNumTie]
  uint32_t floor;
};

// At workgroup scope an RMW is done in this XCC's L2, which all the row's CTAs share.
SGL_DEVICE uint32_t l2_add(uint32_t* p, uint32_t v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}
SGL_DEVICE uint32_t l2_or(uint32_t* p, uint32_t v) {
  return __hip_atomic_fetch_or(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}
SGL_DEVICE uint32_t l2_read(uint32_t* p) {
  return l2_add(p, 0u);
}
SGL_DEVICE unsigned long long l2_read64(unsigned long long* p) {
  return __hip_atomic_fetch_add(p, 0ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}
SGL_DEVICE void l2_zero(uint32_t* p) {
  __hip_atomic_store(p, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}
/// Report this CTA's rank on its XCC, device-wide; nothing waits on it.
SGL_DEVICE void present(Presence* p, uint32_t xcc, uint32_t rank) {
  __hip_atomic_fetch_add(&p[xcc].ranks, 1u << rank, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
/// The checker CTA, a lane an XCC: wait until every XCC holds each of the launch's ranks once
/// (`all`, a bit a rank), then zero the words for the next launch. The CTAs report as they start,
/// so this ends before the rows do.
SGL_DEVICE void check_presence(Presence* p, uint32_t all) {
  const auto tx = threadIdx.x;
  if (tx < kXcc) {
    uint32_t spins = 0;
    while (__hip_atomic_load(&p[tx].ranks, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != all) {
      __builtin_amdgcn_s_sleep(1);
      if (++spins == kSpinLimit) __builtin_trap();
    }
  }
  __syncthreads();
  if (tx < kXcc) __hip_atomic_store(&p[tx].ranks, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

/// This thread's stores and atomics acknowledged by the L2.
SGL_DEVICE void drain() {
  __builtin_amdgcn_s_waitcnt(0);
}

struct Split : TopKSplit {
  /// histogram_chunk, flushed into the XCC's L2.
  SGL_DEVICE static void histogram(const TopKProblem& problem, Chunk chunk, uint32_t* row_hist, Smem* smem) {
    const auto tx = threadIdx.x;
    init_histogram(smem->histogram, tx);
    __syncthreads();
    for_each_input(problem.in + chunk.start, chunk.len, [&](float val, uint32_t) {
      atomicAdd(&smem->histogram[impl::extract_coarse_bin<kHistBits>(val)], 1);
    });
    __syncthreads();
#pragma unroll
    for (uint32_t i = 0; i < kHistItems; ++i) {
      const auto bin = tx * kHistItems + i;
      if (const auto n = smem->histogram[bin]; n != 0) l2_add(&row_hist[bin], n);
    }
  }

  /// Wait for the row's kSplit CTAs. The poll is an RMW that changes the word (it sets kMark):
  /// one that does not would compile to a load, and a load spins on this CU's own cached line.
  SGL_DEVICE static void barrier(uint32_t* arrive) {
    constexpr uint32_t kMark = 0x80000000u;
    drain();
    __syncthreads();
    if (threadIdx.x == 0) {
      l2_add(arrive, 1u);
      uint32_t spins = 0;
      while ((l2_or(arrive, kMark) & ~kMark) < kSplit) {
        __builtin_amdgcn_s_sleep(1);
        if (++spins == kSpinLimit) __builtin_trap();
      }
    }
    __syncthreads();
  }

  /// select_chunk: the row histogram and the bases from the XCC's L2.
  SGL_DEVICE static void
  select(const TopKProblem& problem, Chunk chunk, uint32_t* row_hist, Counters* ctr, impl::TieValue* ties, Smem* smem) {
    const auto tx = threadIdx.x;
#pragma unroll
    for (uint32_t i = 0; i < kHistItems; ++i) {
      const auto bin = tx * kHistItems + i;
      smem->histogram[bin] = l2_read(&row_hist[bin]);
    }
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
      smem->v_hi = impl::padding_value();
      smem->v_lo = impl::padding_value();
    }
    __syncthreads();
    find_threshold(problem.topk, problem.seq_len, smem, [&](uint32_t threshold_bin) {
      const auto [v_lo, v_hi] = impl::coarse_bin_bounds<kHistBits>(threshold_bin);
      smem->v_hi = v_hi;
      smem->v_lo = v_lo;
    });
    const auto topk = problem.topk;
    const auto v_hi = smem->v_hi;
    const auto v_lo = smem->v_lo;
    const bool zero_bin = TOPK_ONE_ZERO_BIN(v_lo);
    if (zero_bin && tx == 0) smem->tie_nonzero = 0;
    __syncthreads();
    for_each_input(problem.in + chunk.start, chunk.len, [&](float val, uint32_t local) {
      const auto idx = chunk.start + local;
      if (val >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1u);
        if (pos < topk) [[likely]]
          smem->staged[pos] = static_cast<int32_t>(idx);
      } else if (val >= v_lo) {
        const auto slot = atomicAdd(&smem->count_eq, 1u);
        if (zero_bin && __float_as_uint(val) != 0u) atomicOr(&smem->tie_nonzero, 1u);
        if (slot < kMaxNumTie) [[likely]]
          smem->tie_values[slot] = {val, idx};
      }
    });
    __syncthreads();
    const auto n_gt = min(smem->count_gt, topk);
    const auto n_eq = min(smem->count_eq, kMaxNumTie);
    if (tx == 0) {
      smem->base_gt = l2_add(&ctr->count_gt, n_gt);
      smem->base_eq = l2_add(
          &ctr->count_eq,
          TOPK_ONE_EXACT ? smem->count_eq : n_eq);
      if (zero_bin && smem->tie_nonzero != 0) l2_or(&ctr->tie_nonzero, 1u);
    }
    __syncthreads();
    const auto base_gt = smem->base_gt;
    const auto base_eq = smem->base_eq;
    for (uint32_t t = tx; t < n_gt; t += kBlockSize) {
      if (base_gt + t < topk) problem.emit(base_gt + t, static_cast<uint32_t>(smem->staged[t]));
    }
    for (uint32_t t = tx; t < n_eq; t += kBlockSize) {
      if (base_eq + t < kMaxNumTie) ties[base_eq + t] = smem->tie_values[t];
    }
  }

  /// arrive_last on a ticket. The last CTA takes the totals, checks every rank came, and resets
  /// the row's counters for the next launch: by now every CTA of the row is past the barrier and
  /// has arrived here, and the next launch on the stream starts after this one ends.
  SGL_DEVICE static bool arrive_last(Counters* ctr, Smem* smem) {
    drain();
    __syncthreads();
    if (threadIdx.x == 0) {
      const bool last = l2_add(&ctr->arrive_sel, 1u) == kSplit - 1;
      smem->is_last = last ? 1u : 0u;
      if (last) {
        if (l2_read(&ctr->ranks) != kRanks) __builtin_trap();
        smem->total_gt = l2_read(&ctr->count_gt);
        smem->total_eq = l2_read(&ctr->count_eq);
        if (TOPK_ONE_ZERO_BIN(smem->v_lo)) smem->tie_nonzero = l2_read(&ctr->tie_nonzero);
        l2_zero(&ctr->count_gt);
        l2_zero(&ctr->count_eq);
        l2_zero(&ctr->tie_nonzero);
        l2_zero(&ctr->ranks);
        l2_zero(&ctr->arrive_hist);
        l2_zero(&ctr->arrive_sel);
      }
    }
    __syncthreads();
    return smem->is_last != 0;
  }

  SGL_DEVICE static void clear_row(uint32_t* row_hist) {
#pragma unroll
    for (uint32_t i = 0; i < kHistItems; ++i) l2_zero(&row_hist[threadIdx.x * kHistItems + i]);
  }

  /// finish_ties, the row's tie list from the XCC's L2.
  SGL_DEVICE static void finish_ties(const TopKProblem& problem, impl::TieValue* ties, Smem* smem) {
    const auto tx = threadIdx.x;
    uint32_t above_count, tie_count;
    const bool zero_ties = TOPK_ONE_ZERO_BIN(smem->v_lo) && smem->tie_nonzero == 0;
    if (TOPK_ONE_EXACT && smem->total_eq > kMaxNumTie && !zero_ties) [[unlikely]] {
      if (tx == 0) smem->count_gt = smem->total_gt;
      __syncthreads();
      refine_ties(problem, smem, smem->v_lo, smem->v_hi, smem->total_eq);
      above_count = smem->count_gt;
      tie_count = min(smem->count_eq, kMaxNumTie);
    } else {
      above_count = smem->total_gt;
      tie_count = min(smem->total_eq, kMaxNumTie);
      for (uint32_t t = tx; t < tie_count; t += kBlockSize) {
        const auto v = l2_read64(reinterpret_cast<unsigned long long*>(ties + t));
        smem->tie_values[t] = *reinterpret_cast<const impl::TieValue*>(&v);
      }
      __syncthreads();
    }
    const auto remain_topk = above_count < problem.topk ? problem.topk - above_count : 0;
    handle_tie(smem->tie_values, problem, above_count, tie_count, remain_topk, &smem->tie_handle);
  }

  /// paged_transform (both outputs), the row's raw indices read back from the XCC's L2.
  SGL_DEVICE static void transform(const TopKProblem& problem, const PageTransform& t) {
    constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
    int32_t indices[kNumElems];
    for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) {
      indices[i] = static_cast<int32_t>(l2_read(reinterpret_cast<uint32_t*>(problem.out + tx)));
    });
    for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) {
      problem.out[tx] = indices[i] >= 0 ? t.page_to_indices(indices[i]) : -1;
      t.raw_out[tx] = indices[i];
    });
  }
};

/// One row's top-k on its XCC: `slot` is its workspace row, `rank` its CTA among the row's kSplit.
template <TopKMode kMode>
SGL_DEVICE void run(const TopKPagedParams& params, const Workspace& ws, uint32_t row, uint32_t slot, uint32_t rank) {
  constexpr bool kStaged = kMode != TopKMode::INDICES;
  auto problem = params.problem(row);
  __shared__ impl::MaxSmem<Register4::Smem, Streaming::Smem, TopKSplit::Smem> smem;

  if (problem.seq_len <= ws.floor) {  // topk_split_select's one-block path
    if (rank != 0) return;
    __shared__ int32_t s_topk_indices[kStaged ? kMaxTopK : 1];
    if (problem.seq_len <= problem.topk) {
      return trivial_transform<false, kMode>(problem, params.template get_transform<kMode>(row));
    }
    if constexpr (kStaged) problem.out = s_topk_indices;
    if (problem.seq_len <= kReg4MaxSeqLen) {
      Register4::forward<false>(problem, &smem);
    } else {
      Streaming::forward<false>(problem, &smem);
    }
    if constexpr (kStaged) {
      __syncthreads();
      paged_transform<kMode>(problem, params.get_output_ptr(row), params.template get_transform<kMode>(row));
    }
    return;
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
  if (!Split::arrive_last(ctr, sm)) return;
  Split::clear_row(row_hist);
  Split::finish_ties(problem, ties, sm);
  if constexpr (kStaged) {
    drain();
    __syncthreads();
    Split::transform(problem, params.template get_transform<kMode>(row));
  }
}

/// This CTA's XCC (its row) and its rank on that XCC, reported for the checker.
SGL_DEVICE uint32_t xcc_row(const Workspace& ws, uint32_t rank) {
  uint32_t xcc;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc));
  const uint32_t row = xcc & 15u;
  if (row >= kXcc) __builtin_trap();
  if (threadIdx.x == 0) present(ws.presence, row, rank);
  return row;
}

TOPK_KERNEL void topk_one(const __grid_constant__ TopKPagedParams params, const Workspace ws) {
  device::enable_smem_spilling();
  if (blockIdx.x == kXcc * kSplit) return check_presence(ws.presence, kRanks);
  const uint32_t rank = blockIdx.x / kXcc;
  const uint32_t row = xcc_row(ws, rank);
  if (row >= params.batch_size) return;
  run<TopKMode::DUAL_OUTPUT>(params, ws, row, row, rank);
}

/// `cols` (both outputs) on each XCC's ranks 0 to kSplit - 1, `blocks` (indices only) on the next kSplit.
TOPK_KERNEL void topk_pair(
    const __grid_constant__ TopKPagedParams cols, const __grid_constant__ TopKPagedParams blocks, const Workspace ws) {
  device::enable_smem_spilling();
  if (blockIdx.x == kXcc * kSplit * 2) return check_presence(ws.presence, (1u << (2 * kSplit)) - 1u);
  const uint32_t rank = blockIdx.x / kXcc;
  const uint32_t row = xcc_row(ws, rank);
  if (row >= cols.batch_size) return;
  if (rank < kSplit) return run<TopKMode::DUAL_OUTPUT>(cols, ws, row, row, rank);
  run<TopKMode::INDICES>(blocks, ws, row, kMaxRows + row, rank - kSplit);
}

}  // namespace topk_one

struct TopKOne {
  /// SGLang's paged top-k parameters for `scores` [rows, width] into `page_out` [rows, k] (and
  /// `raw_out` when given); no page table when `page_table` is empty.
  static TopKPagedParams params_of(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const int32_t* page_table,
      int64_t page_table_stride,
      const tvm::ffi::TensorView page_out,
      int32_t* raw_out,
      const int64_t page_size) {
    using namespace host;
    const auto B = static_cast<uint32_t>(scores.size(0));
    const auto topk = static_cast<uint32_t>(page_out.size(1));
    RuntimeCheck(B > 0 && B <= topk_one::kMaxRows, "the one-launch top-k takes 1 to 8 rows");
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");
    RuntimeCheck(scores.stride(0) % 4 == 0, "score rows must be 16-byte aligned");
    RuntimeCheck(page_size > 0 && std::has_single_bit(static_cast<uint64_t>(page_size)), "page_size must be a power of 2");
    return TopKPagedParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = page_table,
        .page_indices = static_cast<int32_t*>(page_out.data_ptr()),
        .raw_indices = raw_out,
        .metadata = nullptr,
        .score_stride = scores.stride(0),
        .page_table_stride = page_table_stride,
        .topk = topk,
        .page_bits = static_cast<uint32_t>(std::countr_zero(static_cast<uint64_t>(page_size))),
        .static_cluster_floor = 0,
        .batch_size = B,
        .out_stride = static_cast<int64_t>(topk),
        .token_seq_lens = nullptr,
        .out_offsets = nullptr,
        .pool_bits = 0,
        .out_cols = topk,
    };
  }

  static topk_one::Workspace workspace_of(
      const tvm::ffi::TensorView hist, const tvm::ffi::TensorView ctr, const tvm::ffi::TensorView ties, uint32_t rows) {
    using namespace host;
    constexpr auto kSlots = topk_one::kMaxProblems * topk_one::kMaxRows;
    RuntimeCheck(
        hist.size(0) == kSlots * TopKSplit::kHistSize && ctr.size(0) == (kSlots + topk_one::kXcc) * 32 &&
            ties.size(0) == kSlots * TopKSplit::kMaxNumTie * 2,
        "workspace shape");
    return topk_one::Workspace{
        .hist = static_cast<uint32_t*>(hist.data_ptr()),
        .ctr = static_cast<topk_one::Counters*>(ctr.data_ptr()),
        .presence = reinterpret_cast<topk_one::Presence*>(static_cast<topk_one::Counters*>(ctr.data_ptr()) + kSlots),
        .ties = static_cast<impl::TieValue*>(ties.data_ptr()),
        .floor = split_floor(rows),
    };
  }

  /// `page_out` / `raw_out` [rows, k] of SGLang's `topk_transform_paged` (both outputs) for at most
  /// kMaxRows rows; `hist`, `ctr`, `ties`: the module's zeroed workspace.
  static void transform_paged(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_out,
      const tvm::ffi::TensorView raw_out,
      const int64_t page_size,
      const tvm::ffi::TensorView hist,
      const tvm::ffi::TensorView ctr,
      const tvm::ffi::TensorView ties) {
    using namespace host;
    const auto params = params_of(
        scores, seq_lens, static_cast<const int32_t*>(page_table.data_ptr()), page_table.stride(0), page_out,
        static_cast<int32_t*>(raw_out.data_ptr()), page_size);
    const auto ws = workspace_of(hist, ctr, ties, params.batch_size);
    // a dispatch round of kXcc CTAs a rank, and the checker
    LaunchKernel(topk_one::kXcc * topk_one::kSplit + 1, kBlockSize, scores.device())
        .config({.use_pdl = false})
        .launch(topk_one::topk_one, params, ws);
  }

  /// `transform_paged` of the columns, and in the same launch the top-k indices of `block_scores`
  /// [rows, blocks] (bounds `block_lens`) into `block_out` [rows, k2]: SGLang's
  /// `topk_transform_paged` without a page table.
  static void transform_paged_pair(
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
      const tvm::ffi::TensorView ties) {
    using namespace host;
    const auto cols = params_of(
        scores, seq_lens, static_cast<const int32_t*>(page_table.data_ptr()), page_table.stride(0), page_out,
        static_cast<int32_t*>(raw_out.data_ptr()), page_size);
    const auto blocks = params_of(block_scores, block_lens, nullptr, 0, block_out, nullptr, 1);
    RuntimeCheck(blocks.batch_size == cols.batch_size, "the pair's selections take the same rows");
    const auto ws = workspace_of(hist, ctr, ties, cols.batch_size);
    LaunchKernel(topk_one::kXcc * topk_one::kSplit * 2 + 1, kBlockSize, scores.device())
        .config({.use_pdl = false})
        .launch(topk_one::topk_pair, cols, blocks, ws);
  }
};

}  // namespace sglang
