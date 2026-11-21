# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's routed stage 1 and the shared experts' down GEMM in one launch (48-180 rows, a8w4 stage 1).

The grid's first ``S`` workgroups each compute one [bm, bn] tile of shared_out [M, 7168] = act [M, 768] @
W_down[7168, 768]^T (bf16 in and out, fp32 accumulation); the rest are aiter's stage-1 workgroups, with stock's tile
order (the XCD swizzle of ``compile_gemm1_a4w4_port``; ``S`` is a multiple of 8, so a workgroup's XCD class is
stock's). The dispatcher starts the low ids first, so the shared tiles run on CUs the stage-1 workgroups have not yet
filled, and nothing in the launch waits on anything else in it: act comes from the routing launch before it.

Device code:
- stage 1: ``aiter.ops.flydsl.kernels.mxfp4_gemm1._gemm1_body`` unchanged (output bit for bit stock's);
- the shared tile: the body of aiter's ``gemm_a16w16_gfx950_kernel`` (no split-K, no slice-K, no bias, no K
  stagger) with the tile index an argument. Its MFMA sequence over K is the same for any tile shape, so the shared
  output matches SGLang's own hgemm launch bit for bit at a tile shape whose LDS fits stage 1's occupancy.
"""

import contextlib
import functools
import threading

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T  # no "from __future__ import annotations": FlyDSL reads the annotations

from aiter.ops.flydsl.kernels import gemm_a16w16_gfx950 as hg
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950_utils import GFX950_WAVE_SIZE
from aiter.ops.flydsl.kernels.mxfp4_gemm1 import (
    _bm_constants,
    _gemm1_body,
    default_epi_splits,
    default_k_stages,
)
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _udiv, _umod, global_typed_ptr

BM1, BN1, BK1 = 16, 128, 256
NXCD = 8


@flyc.jit
def _shared_tile(out, a, b, m, n, k, a_ld, b_ld, tiled_mma, param, bid_m, bid_n, smem_a, smem_b, smem_c, tid):
    """One [block_m, block_n] tile of out = a @ b (aiter's gemm_a16w16_gfx950_kernel body, ks1 / k_waves 1 / no
    bias / no stagger)."""
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    block_threads = param.block_threads
    elem_dtype = fx.BFloat16
    k_mma_iters = block_k // param.mma_k
    k_tiles = k // block_k
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    ctx = hg.make_gemm_ab_load_context(elem_dtype, load_tid=tid, ks_begin=fx.Int32(0), param=param)
    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]
    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(ctx.a_tiled_copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(ctx.b_tiled_copy_atom, tiled_mma).get_slice(tid)
    a_lds_layout, b_lds_layout = hg.make_gemm_ab_lds_layouts(
        block_m, block_n, block_k, param.a_is_transposed, param.b_is_transposed
    )
    c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))
    sA = fx.make_view(smem_a, a_lds_layout)
    sB = fx.make_view(smem_b, b_lds_layout)
    sC_write = fx.make_view(smem_c, c_lds_layout)
    frag_A = thr_mma.make_fragment_A(sA)
    frag_B = thr_mma.make_fragment_B(sB)
    frag_C = thr_mma.make_fragment_C(gC)
    frag_A_retile = thr_copy_A.retile(frag_A)
    frag_B_retile = thr_copy_B.retile(frag_B)
    row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
    thr_mma_cRow = thr_mma.partition_C(row_coords)
    thr_mma_cCol = thr_mma.partition_C(col_coords)
    frag_C.fill(0.0)

    def load_a(k_tile, stage):
        hg.async_load_to_lds(
            tile=hg.AsyncLoadTile(
                lds_base=smem_a + stage * block_m * block_k,
                src_base=fx.get_iter(a_buf),
                lds_layout=a_lds_layout,
                outer_tile_size=block_m,
                outer_bound=m,
                global_outer_offset=block_m_offset,
                leading_stride=a_ld,
                k_tile=k_tile,
            ),
            context=ctx,
            load_iters=param.ldg_a_iters,
            is_k_major=param.a_is_transposed,
        )

    def load_b(k_tile, stage):
        hg.async_load_to_lds(
            tile=hg.AsyncLoadTile(
                lds_base=smem_b + stage * block_n * block_k,
                src_base=fx.get_iter(b_buf),
                non_temporal=param.b_non_temporal,
                lds_layout=b_lds_layout,
                outer_tile_size=block_n,
                outer_bound=n,
                global_outer_offset=block_n_offset,
                leading_stride=b_ld,
                k_tile=k_tile,
            ),
            context=ctx,
            load_iters=param.ldg_b_iters,
            is_k_major=not param.b_is_transposed,
        )

    def compute_stage(read_stage):
        thr_sA = thr_copy_A.partition_S(fx.make_view(smem_a + read_stage * block_m * block_k, a_lds_layout))
        thr_sB = thr_copy_B.partition_S(fx.make_view(smem_b + read_stage * block_n * block_k, b_lds_layout))
        for it in range_constexpr(k_mma_iters):
            fx.copy(ctx.b_s2r_copy_atom, thr_sB[None, None, it], frag_B_retile[None, None, it])
            fx.copy(ctx.a_s2r_copy_atom, thr_sA[None, None, it], frag_A_retile[None, None, it])
            fx.gemm(tiled_mma, frag_C, frag_A[None, None, it], frag_B[None, None, it], frag_C,
                    traversal_order=fx.GemmTraversalOrder.KNM)

    for stage in range_constexpr(stages - 1):
        load_b(stage, stage)
        load_a(stage, stage)
        rocdl.asyncmark()
    rocdl.sched_barrier(0)
    main_loop_end = k_tiles - (stages - 1)
    for k_tile in range(0, main_loop_end, 1):
        current_stage = k_tile % stages
        write_stage = (current_stage + stages - 1) % stages
        rocdl.wait_asyncmark(stages - 2)
        rocdl.s_barrier()
        load_b(k_tile + (stages - 1), write_stage)
        load_a(k_tile + (stages - 1), write_stage)
        rocdl.asyncmark()
        compute_stage(current_stage)
        rocdl.sched_barrier(0)
    current_stage = main_loop_end % stages
    for s in range_constexpr(0, stages - 1):
        rocdl.wait_asyncmark(stages - 2 - s)
        rocdl.s_barrier()
        compute_stage(current_stage)
        current_stage = (current_stage + 1) % stages

    frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
    frag_C_out.store(frag_C.load().to(elem_dtype))
    gpu.barrier()
    for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
        sC_write[fx.get_scalar(thr_mma_cRow[i]), fx.get_scalar(thr_mma_cCol[i])] = frag_C_out[i]
    gpu.barrier()
    vec = param.cshuffle_r2g_vec_size
    x_threads = block_n // vec
    vectors = block_m * block_n // vec
    for i in range_constexpr((vectors + block_threads - 1) // block_threads):
        vector_idx = block_threads * i + tid
        if vector_idx < vectors:
            local_row = vector_idx // x_threads
            local_col = vector_idx % x_threads * vec
            global_row = block_m_offset + local_row
            global_col = block_n_offset + local_col
            if (global_row < m) and (global_col < n):
                c_vec = fx.ptr_load(smem_c + local_row * block_n + local_col,
                                    result_type=fx.Vector.make_type(vec, elem_dtype))
                hg.write_cshuffle_vec_to_global(out, out_buf, global_row * n + global_col, c_vec, False, False)


@functools.cache
def compile_stage1_shared(*, s1_use_nt, s1_prefetch_hidden, s1_xcd_swizzle, situ_beta, situ_linear_beta,
                          sh_block_m, sh_block_n, sh_block_k, sh_stages, sh_m_waves, sh_n_waves, sh_last=0,
                          push=None, hidden=3584, inter=384, experts=896):
    """(launcher, shared tile param) of one static configuration (``SHARED_CONFIGS``).

    ``push``: None, or the number of MoE layers a step (``launches``) of the shared partial's C18 push: every
    shared workgroup then pushes its 64 x 64 tile into its slot on the other ranks and flags it
    (``stages.shared_push``) once the tile is in memory, while the stage-1 workgroups run. The launch's push
    arguments are (shared_out address, peer table address, rank, epoch buffer address, layer slot)."""
    N_OUT = 2 * inter
    NUM_N1 = N_OUT // BN1
    KH_TILE = BK1
    K_TILES = hidden // BK1
    epi_splits = default_epi_splits(BM1, BN1, 1, 4)
    k_stages = default_k_stages(BM1, BN1, KH_TILE, K_TILES, N_OUT, 1, epi_splits)
    _, _, _, lds1 = _bm_constants(BM1, BN1, KH_TILE, K_TILES, 1, epi_splits, k_stages)
    hparam = hg.make_gemm_a16w16_gfx950_param(
        in_dtype_id=hg.GEMM_A16W16_DTYPE_BF16, out_dtype_id=hg.GEMM_A16W16_DTYPE_BF16, block_m=sh_block_m,
        block_n=sh_block_n, block_k=sh_block_k, stages=sh_stages, split_k=1, m_waves=sh_m_waves,
        n_waves=sh_n_waves, k_waves=1, group_m=0, a_is_transposed=False, b_is_transposed=True)
    st, bm, bn, bk = sh_stages, sh_block_m, sh_block_n, sh_block_k
    tag = (f"h{hidden}_i{inter}_g1{'nt' if s1_use_nt else 'c'}{'_hpf' if s1_prefetch_hidden else ''}"
           f"_xcd{s1_xcd_swizzle}_sh{bm}x{bn}x{bk}x{st}w{sh_m_waves}x{sh_n_waves}{'_last' if sh_last else ''}"
           f"{'' if push is None else f'_push{push}'}")
    if push is not None:
        from sgl_amd_mono.kimi_k3.gfx950.stages import shared_push as S

    @fx.struct
    class SharedAB:
        a: fx.Array[fx.BFloat16, st * bm * bk, 16]
        b: fx.Array[fx.BFloat16, st * bn * bk, 16]

    @fx.union
    class Smem:
        ab: SharedAB
        c: fx.Array[fx.BFloat16, bm * bn, 16]
        s1: fx.Array[fx.Uint8, lds1, 16]

    @flyc.kernel(name=f"k3_stage1_shared_{tag}", known_block_size=[256, 1, 1])
    def kernel(
        sh_out: fx.Tensor,
        sh_a: fx.Tensor,
        sh_b: fx.Tensor,
        sh_m: fx.Int32,
        sh_n: fx.Int32,
        sh_k: fx.Int32,
        sh_a_ld: fx.Int32,
        sh_b_ld: fx.Int32,
        sh_tiles_m: fx.Int32,
        sh_ctas: fx.Int32,
        s1_grid: fx.Int32,
        tiled_mma: fx.TiledMma,
        param: hg.GemmA16W16Gfx950Param,
        arg_w1: fx.Int64,
        arg_w1s: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        arg_inter: fx.Int64,
        arg_inter_s: fx.Int64,
        arg_hidden: fx.Int64,
        i32_ntok: fx.Int32,
        push_xs: fx.Int64,
        push_peers: fx.Int64,
        push_rank: fx.Int32,
        push_epoch: fx.Int64,
        push_slot: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        storage = fx.SharedAllocator().allocate(Smem)
        # the shared tiles at the grid's front (dispatched first) or, sh_last, at its end (into stage 1's tail)
        sh_id = bx
        pid = bx - sh_ctas
        is_shared = bx < sh_ctas
        if const_expr(sh_last):
            sh_id = bx - s1_grid
            pid = bx
            is_shared = bx >= s1_grid
        if is_shared:
            # tile (m, n): the M blocks of an N tile share a XCD class (its weights read once through L2)
            x = sh_id % fx.Int32(NXCD)
            r = sh_id // fx.Int32(NXCD)
            bid_m = r % sh_tiles_m
            bid_n = (r // sh_tiles_m) * fx.Int32(NXCD) + x
            _shared_tile(sh_out, sh_a, sh_b, sh_m, sh_n, sh_k, sh_a_ld, sh_b_ld, tiled_mma, param, bid_m, bid_n,
                         storage.ab.a.peek().ptr, storage.ab.b.peek().ptr, storage.c.peek().ptr, tid)
            if const_expr(push is not None):
                # the tile's stores in memory, then its data into each other rank's slot (a wave a peer, peers w
                # and w + 4), drained, then the flags
                c = S.context(push_peers, push_rank, push_epoch, push_slot, tid, bx, push)
                wave = rocdl.readfirstlane(T.i32, tid // 64)
                lane = tid % 64
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                for j in range_constexpr(2):
                    S.push_tile(c, push_xs, hidden * 2, bid_m, bid_n, sh_m, wave + fx.Int32(4 * j), lane, S.MAX_ROWS)
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                for j in range_constexpr(2):
                    S.flag_tile(c, bid_m, bid_n, wave + fx.Int32(4 * j), lane)
        else:
            lane = tid % 64
            wave = rocdl.readfirstlane(T.i32, tid // 64)
            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = fx.Int32(cumsum0) // fx.Int32(BM1)
            bound = total_m_blocks * fx.Int32(NUM_N1)
            xq = _udiv(bound, NXCD)
            xr = _umod(bound, NXCD)
            if pid < bound:
                tile = pid
                if const_expr(s1_xcd_swizzle > 0):
                    xc = _umod(pid, NXCD)
                    wgid = xc * xq + (xc < xr).select(xc, xr) + _udiv(pid, NXCD)
                    ng = fx.Int32(s1_xcd_swizzle * NUM_N1)
                    group_id = wgid // ng
                    first_pid_m = group_id * fx.Int32(s1_xcd_swizzle)
                    remaining_m = total_m_blocks - first_pid_m
                    sw = fx.Int32(s1_xcd_swizzle)
                    group_size_m = (remaining_m < sw).select(remaining_m, sw)
                    wig = wgid % ng
                    tile = (first_pid_m + (wig % group_size_m)) * fx.Int32(NUM_N1) + wig // group_size_m
                _gemm1_body(
                    storage.s1.peek().ptr,
                    arg_hidden,  # A quant / scale: unused with inline quantization
                    arg_hidden,
                    arg_w1,
                    arg_w1s,
                    arg_eids,
                    arg_mind,
                    arg_inter,
                    arg_inter_s,
                    arg_hidden,
                    arg_hidden,  # bias: unused
                    tile,
                    lane,
                    wave,
                    s1_use_nt,
                    i32_ntok,
                    total_m_blocks,
                    BM=BM1,
                    BN=BN1,
                    BK=BK1,
                    inline_quant=True,
                    prefetch_hidden=s1_prefetch_hidden,
                    a_dtype="fp8",
                    out_dtype="fp8",
                    act="situv2",
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                    swiglu_limit=7.0,
                    enable_bias=False,
                    K=hidden,
                    N_OUT=N_OUT,
                    NE=experts,
                    interleave=True,
                    native_scale_layout=True,
                    num_waves=4,
                    k_wave=1,
                    epi_splits=epi_splits,
                    k_stages=k_stages,
                )

    @flyc.jit
    def launch(
        sh_out: fx.Tensor,
        sh_a: fx.Tensor,
        sh_b: fx.Tensor,
        param: hg.GemmA16W16Gfx950Param,
        arg_w1: fx.Int64,
        arg_w1s: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        arg_inter: fx.Int64,
        arg_inter_s: fx.Int64,
        arg_hidden: fx.Int64,
        i32_ntok: fx.Int32,
        i32_s1_grid: fx.Int32,
        push_xs: fx.Int64,
        push_peers: fx.Int64,
        push_rank: fx.Int32,
        push_epoch: fx.Int64,
        push_slot: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        m = fx.Int32(fx.get_scalar(sh_a.shape[0]))
        n = fx.Int32(fx.get_scalar(sh_b.shape[1]))
        k = fx.Int32(fx.get_scalar(sh_a.shape[1]))
        a_ld = fx.Int32(fx.get_scalar(sh_a.stride[0]))
        b_ld = fx.Int32(fx.get_scalar(sh_b.stride[1]))
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, fx.BFloat16))
        kpg = param.mma_k // 4
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((param.m_waves, param.n_waves, 1), (param.n_waves, 1, 0)),
            fx.make_tile(None, None, fx.make_layout((kpg, 4), (1, kpg))),
        )
        tiles_m = (m + param.block_m - 1) // param.block_m
        sh_ctas = tiles_m * (n // param.block_n)
        kernel(sh_out, sh_a, sh_b, m, n, k, a_ld, b_ld, tiles_m, sh_ctas, i32_s1_grid, tiled_mma, param, arg_w1, arg_w1s,
               arg_eids, arg_cumsum, arg_mind, arg_inter, arg_inter_s, arg_hidden, i32_ntok, push_xs, push_peers,
               push_rank, push_epoch, push_slot).launch(
            grid=(sh_ctas + i32_s1_grid, 1, 1), block=(256, 1, 1), stream=stream)

    return launch, hparam


# the shared tile a stock bucket (get_padded_M 64 / 128 / 256): LDS <= 40 KB keeps stage 1's 4 workgroups a CU;
# 64 x 64 is the C18 push's tile (shared_push.TM / TN), 4 waves the launch's 256 threads.
# block_k 64 keeps the K order of SGLang's own hgemm (bit for bit); block_k 32 does not.
SHARED_CONFIGS = {
    b: dict(sh_block_m=64, sh_block_n=64, sh_block_k=64, sh_stages=2, sh_m_waves=1, sh_n_waves=4)
    for b in (64, 128, 256)
}


def stage1_shared(g1: dict, act, w_down, shared_out, overrides: tuple = (), push=None) -> None:
    """aiter's stage 1 of one MoE call (``g1``: the keyword arguments fused_moe passes to flydsl_mxfp4_gemm1, its
    a8w4 BM16 stage 1; the launch takes that stage 1's own flags) and shared_out [M, 7168] bf16 = act [M, 768] bf16
    @ w_down [7168, 768]^T in one launch. ``push``: None, or (launches, peers, rank, epoch, slot) of the shared
    partial's C18 push (``SharedPushAr.launches`` and ``SharedPushAr.push_args(layer)``): its tiles also go to
    the other ranks' slots."""
    import torch
    from aiter.ops.flydsl.kernels.mxfp4_gemm1 import gemm1_grid

    from . import bucket

    m = g1["n_tokens"]
    cf = dict(SHARED_CONFIGS[bucket(m)])
    cf.update(dict(overrides))
    launch, hparam = compile_stage1_shared(
        s1_use_nt=bool(g1["use_nt"]), s1_prefetch_hidden=bool(g1["prefetch_hidden"]),
        s1_xcd_swizzle=int(g1["xcd_swizzle"]), situ_beta=float(g1["situ_beta"]),
        situ_linear_beta=float(g1["situ_linear_beta"]), push=None if push is None else int(push[0]), **cf)
    ptr = lambda v: v if isinstance(v, int) else v.data_ptr()
    pa = (0, 0, 0, 0, 0) if push is None else (shared_out.data_ptr(), ptr(push[1]), int(push[2]), ptr(push[3]),
                                               int(push[4]))
    u8 = lambda t: t.view(torch.uint8) if t.dtype != torch.uint8 else t
    args = (
        hg._dynamic_tensor_arg(shared_out, 1),
        hg._dynamic_tensor_arg(act, 1),
        hg._dynamic_tensor_arg(w_down.t(), 0),
        hparam,
        u8(g1["w1_u8"]).data_ptr(),
        u8(g1["w1_scale_u8"]).data_ptr(),
        g1["sorted_expert_ids"].data_ptr(),
        g1["cumsum_tensor"].data_ptr(),
        g1["m_indices"].data_ptr(),
        g1["inter_sorted_quant"].data_ptr(),
        g1["inter_sorted_shuffled_scale"].data_ptr(),
        g1["hidden_states"].data_ptr(),
        int(m),
        gemm1_grid(m, BM1, NE=896, TOPK=16, INTER=384, BN=BN1),
        *pa,
        torch.cuda.current_stream(),
    )
    from aiter.ops.flydsl.kernels.kernels_common import run_cached

    run_cached(launch, *args, constexpr_param=hparam, compiler=flyc.compile, dispatch_args=args)


_PENDING = threading.local()


def _install_hook():
    """Once: aiter's flydsl_mxfp4_gemm1 (looked up by fused_moe at call time) runs stage1_shared when a shared
    down is registered for this thread's next MoE call, else itself."""
    import aiter.ops.flydsl.mxfp4_gemm1_kernels as g1m

    if getattr(g1m.flydsl_mxfp4_gemm1, "_k3_stage1_shared", False):
        return
    orig = g1m.flydsl_mxfp4_gemm1

    def flydsl_mxfp4_gemm1(**kw):
        held = getattr(_PENDING, "held", None)
        if held is None:
            return orig(**kw)
        _PENDING.held = None
        stage1_shared(kw, *held[:3], push=held[3])
        return None

    flydsl_mxfp4_gemm1._k3_stage1_shared = True
    g1m.flydsl_mxfp4_gemm1 = flydsl_mxfp4_gemm1


@contextlib.contextmanager
def shared_down_in_stage1(act, w_down, shared_out, push=None):
    """The next aiter MoE call on this thread runs its stage 1 as ``stage1_shared`` with this shared down
    (shared_out = act @ w_down^T) and ``push``. For a call whose stage 1 is aiter's a8w4 BM16 flydsl stage 1:
    route_sort_shared's presort at ``route.FUSED_SHARED_ROWS``."""
    _install_hook()
    _PENDING.held = (act, w_down, shared_out, push)
    try:
        yield
    finally:
        _PENDING.held = None
