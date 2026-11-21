# SPDX-License-Identifier: Apache-2.0
"""FlyDSL device code of the persistent routed-MoE launch: aiter's a8w4 stage-1 tile body (``_gemm1_body``) and
stage-2 tile (``compile_gemm2_a4w4_port``'s composition hook) run by one grid of persistent CTAs.

Work split:
- The sorted M blocks (``num_valid_ids[0] // 16``) are cut into 8 contiguous ranges, one per XCD; CTA ``bx`` works
  for XCD ``bx % 8``, the XCD the dispatcher places it on. A block's stage-1 producers and stage-2 consumers are on
  one XCD, so stage 1's plain stores reach stage 2's plain loads through that XCD's L2, with no cache maintenance.
- Each XCD's items are its stage-1 tiles (block, N tile) followed by its stage-2 tiles, in a group-swizzled block
  order. CTA rank ``bx // 8`` starts at item rank; later items come from the XCD's ticket counter, fetched one item
  ahead. Every stage-1 item precedes every stage-2 item, so a CTA that waits holds no stage-1 work.
- A stage-1 tile drains its stores (``vmcnt(0)``), then stores its flag word ``tag``. A stage-2 tile waits until its
  block's six flags read ``tag``. ``tag = epoch << 3 | xcd``: the XCD's epoch word advances once a launch, by the
  CTA that takes the XCD's last ticket (which also resets the ticket count), so stale flags never match.

Sync buffer (int32, 128 B lines): line x < 8 = [ticket count, epoch] of XCD x; line 8 + b = the six stage-1 flags
of M block b (initialized to -1)."""


import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T  # no "from __future__ import annotations": FlyDSL reads the annotations

from aiter.ops.flydsl.kernels import communication_ops_utils as cops
from aiter.ops.flydsl.kernels.mxfp4_gemm1 import (
    _bm_constants,
    _gemm1_body,
    default_epi_splits,
    default_k_stages,
)
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr
from aiter.ops.flydsl.kernels.mxmoe_dispatcher import compile_gemm2_a4w4_port

BM = 16
NXCD = 8
LINE_B = 128
SYNC_HEAD_B = NXCD * LINE_B
STAMP_WORDS = 64  # i64 stamps a CTA (stamped build only): start, XCC id, (ticket, begin, end) an item, item count
STAMP_ITEMS = (STAMP_WORDS - 3) // 3


def _swizzle(l, nb, sw, nn):
    """Item l of a range of nb blocks x nn tiles -> (block, tile): groups of sw blocks, tile-major inside a group
    (aiter's stage-1 XCD swizzle)."""
    if const_expr(sw == 1):
        return l // fx.Int32(nn), l % fx.Int32(nn)
    ng = fx.Int32(sw * nn)
    g = l // ng
    wig = l - g * ng
    first = g * fx.Int32(sw)
    rem = nb - first
    gs = (rem < fx.Int32(sw)).select(rem, fx.Int32(sw))
    return first + wig % gs, wig // gs


def _flags_l2(addr_i64):
    """The 8 flag words at ``addr_i64`` (uniform, 32 B aligned), read past the scalar cache from the XCD's L2."""
    v = llvm.InlineAsmOp(T.vec(8, T.i32), [arith.unwrap(fx.Int64(addr_i64))],
                         "s_load_dwordx8 $0, $1, 0x0 glc\ns_waitcnt lgkmcnt(0)", "=s,s", has_side_effects=True).result
    return fx.Vector(v)


def _pending(addr_i64, want):
    v = _flags_l2(addr_i64)
    late = v[0] != want
    for i in range_constexpr(1, 6):
        late = late | (v[i] != want)
    return late


@flyc.jit
def _wait_block(addr_i64, want):
    """Every wave: until the block's six stage-1 flags read ``want`` (scalar loads; the producers are on this XCD)."""
    late = _pending(addr_i64, want)
    while late:
        fx.rocdl.s_sleep(1)
        late = _pending(addr_i64, want)


def _store_agent(addr_i64, value):
    fx.generic_store(cops._to_ptr_global(addr_i64), fx.Int32(value), memory_order=fx.AtomicOrdering.Monotonic,
                     syncscope=fx.rocdl.SyncScope.AgentOneAs)


def _stamp():
    return fx.Int64(
        llvm.InlineAsmOp(T.i64, [], "s_memrealtime $0\ns_waitcnt lgkmcnt(0)", "=s", has_side_effects=True).result
    )


def compile_persist(
    *,
    s1_use_nt: bool,
    s1_prefetch_hidden: bool,
    s1_swizzle: int,
    s2_bn: int,
    s2_bk: int,
    s2_use_nt: bool,
    s2_swizzle: int,
    ctas_per_xcd: int,
    situ_beta: float,
    situ_linear_beta: float,
    hidden: int = 3584,
    inter: int = 384,
    experts: int = 896,
    waves_per_eu: int = 4,
    post_misched: bool = True,
    stamps: bool = False,
    only: str = "",
):
    """The launcher of one static configuration (see ``moe_persist.CONFIGS``). ``only`` ("s1" / "s2", timing
    diagnostics; the output is not the MoE's): run one stage's items alone (``s2`` skips the flag waits)."""
    BN1, BK1 = 128, 256
    N_OUT = 2 * inter
    N1 = N_OUT // BN1
    N2 = hidden // s2_bn
    KH_TILE = BK1  # fp8 activations
    K_TILES = hidden // BK1
    epi_splits = default_epi_splits(BM, BN1, 1, 4)
    k_stages = default_k_stages(BM, BN1, KH_TILE, K_TILES, N_OUT, 1, epi_splits)
    _, _, _, lds1 = _bm_constants(BM, BN1, KH_TILE, K_TILES, 1, epi_splits, k_stages)
    # stage 2's own LDS (compile_gemm2_a4w4_port, bf16 C staging): max(C, aStages x A slot)
    c2 = BM * s2_bn * 2
    slot2 = BM * s2_bk
    lds2 = max(c2, (3 if 3 * slot2 <= c2 else 2) * slot2)
    tile_lds = (max(lds1, lds2) + 15) // 16 * 16
    lds_total = tile_lds + 16
    R = ctas_per_xcd
    grid = NXCD * R
    # a CTA's first item is static (its rank), so every CTA must be resident at once: waves_per_eu CTAs a CU
    # (4 waves each, one a SIMD) on all 256 CUs
    assert grid <= 256 * waves_per_eu, (grid, waves_per_eu)
    tag = (f"h{hidden}_i{inter}_g1{'nt' if s1_use_nt else 'c'}{'_hpf' if s1_prefetch_hidden else ''}_sw{s1_swizzle}"
           f"_g2bn{s2_bn}bk{s2_bk}{'nt' if s2_use_nt else 'c'}_sw{s2_swizzle}_r{R}"
           f"_w{waves_per_eu}{'' if post_misched else '_npm'}{'_st' if stamps else ''}{'_' + only if only else ''}")

    def compose(*, module_name, emit_gemm2_tile, shared_storage):
        @fx.struct
        class Smem:
            buf: fx.Array[fx.Int8, lds_total, 16]

        @flyc.kernel(name=f"k3_moe_persist_{tag}", known_block_size=[256, 1, 1])
        def kernel(
            arg_w1: fx.Int64,
            arg_w1s: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_mind: fx.Int64,
            arg_inter: fx.Int64,
            arg_inter_s: fx.Int64,
            arg_hidden: fx.Int64,
            arg_w2: fx.Int64,
            arg_w2s: fx.Int64,
            arg_stids: fx.Int64,
            arg_sw: fx.Int64,
            arg_out: fx.Int64,
            arg_sync: fx.Int64,
            arg_stamps: fx.Int64,
            i32_ntok: fx.Int32,
            i32_max_m_blocks: fx.Int32,
        ):
            tid = fx.Int32(gpu.thread_id("x"))
            bx = fx.Int32(gpu.block_id("x"))
            lane = tid % 64
            wave = rocdl.readfirstlane(T.i32, tid // 64)
            lds = fx.SharedAllocator().allocate(Smem).peek()
            lds_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))
            tk_lds = lds_typed_ptr(lds_i32, T.i32, byte_offset=fx.Int32(tile_lds))
            ep_lds = lds_typed_ptr(lds_i32, T.i32, byte_offset=fx.Int32(tile_lds + 4))
            xcd = bx % fx.Int32(NXCD)
            rank = bx // fx.Int32(NXCD)
            head = fx.Int64(arg_sync) + fx.Int64(xcd) * fx.Int64(LINE_B)
            flags = fx.Int64(arg_sync) + fx.Int64(SYNC_HEAD_B)
            stamp_base = fx.Int64(arg_stamps) + fx.Int64(bx) * fx.Int64(STAMP_WORDS * 8)

            epoch = fx.Int32(0)
            if tid == fx.Int32(0):
                epoch = fx.Int32(cops.load_i32_global_agent(head + fx.Int64(4)))
            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = fx.Int32(cumsum0) // fx.Int32(BM)
            qb = total_m_blocks // fx.Int32(NXCD)
            rb = total_m_blocks % fx.Int32(NXCD)
            nb = qb + (xcd < rb).select(fx.Int32(1), fx.Int32(0))
            b0 = xcd * qb + (xcd < rb).select(xcd, rb)
            n1 = nb * fx.Int32(N1)
            n_items = n1 + nb * fx.Int32(N2)
            if const_expr(only == "s1"):
                n_items = n1
            if const_expr(only == "s2"):
                n1 = fx.Int32(0)
                n_items = nb * fx.Int32(N2)
            # the flag value of this launch, from thread 0's epoch (thread 0 waits for its load here, which orders
            # the load before this CTA's first ticket fetch)
            if tid == fx.Int32(0):
                fx.ptr_store((epoch << fx.Int32(3)) | xcd, ep_lds)
            gpu.barrier()
            want = rocdl.readfirstlane(T.i32, fx.Int32(fx.ptr_load(ep_lds)))
            if const_expr(stamps):
                if tid == fx.Int32(0):
                    fx.ptr_store(_stamp(), global_typed_ptr(stamp_base, T.i64, align=8))
                    fx.ptr_store(fx.Int64(cops.hw_xcc_id()), global_typed_ptr(stamp_base, T.i64, align=8, byte_offset=8))

            t = rank
            k = fx.Int32(0)
            fetched = fx.Int32(0)
            blk1 = fx.Int32(0)
            nt1 = fx.Int32(0)
            blk2 = fx.Int32(0)
            nt2 = fx.Int32(0)
            while t < n_items:
                if tid == fx.Int32(0):
                    fetched = fx.Int32(cops.atomic_add_agent(head, fx.Int32(1)))
                if const_expr(stamps):
                    if tid == fx.Int32(0):
                        if k < fx.Int32(STAMP_ITEMS):
                            fx.ptr_store(fx.Int64(t), global_typed_ptr(
                                stamp_base, T.i64, align=8, byte_offset=fx.Int64(16) + fx.Int64(k) * fx.Int64(24)))
                            fx.ptr_store(_stamp(), global_typed_ptr(
                                stamp_base, T.i64, align=8, byte_offset=fx.Int64(24) + fx.Int64(k) * fx.Int64(24)))
                if t < n1:
                    blk1, nt1 = _swizzle(t, nb, s1_swizzle, N1)
                    _gemm1_body(
                        lds.buf.ptr,
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
                        (b0 + blk1) * fx.Int32(N1) + nt1,
                        lane,
                        wave,
                        s1_use_nt,
                        i32_ntok,
                        total_m_blocks,
                        BM=BM,
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
                    # every wave's stage-1 stores are in the XCD's L2 before the flag goes out
                    rocdl.s_waitcnt(vmcnt=0)
                    gpu.barrier()
                    if tid == fx.Int32(0):
                        fx.ptr_store(want, global_typed_ptr(
                            flags + fx.Int64(b0 + blk1) * fx.Int64(LINE_B) + fx.Int64(nt1) * fx.Int64(4), T.i32))
                else:
                    blk2, nt2 = _swizzle(t - n1, nb, s2_swizzle, N2)
                    if const_expr(only != "s2"):
                        _wait_block(flags + fx.Int64(b0 + blk2) * fx.Int64(LINE_B), want)
                    emit_gemm2_tile(
                        arg_inter,
                        arg_inter_s,
                        arg_w2,
                        arg_w2s,
                        arg_eids,
                        arg_stids,
                        arg_sw,
                        arg_w2,  # bias: unused
                        arg_out,
                        b0 + blk2,
                        nt2,
                        lane,
                        wave,
                        i32_ntok,
                        i32_max_m_blocks,
                        fx.Int32(inter),
                        fx.Int32(hidden),
                        lds,
                    )
                if const_expr(stamps):
                    if tid == fx.Int32(0):
                        if k < fx.Int32(STAMP_ITEMS):
                            fx.ptr_store(_stamp(), global_typed_ptr(
                                stamp_base, T.i64, align=8, byte_offset=fx.Int64(32) + fx.Int64(k) * fx.Int64(24)))
                k = k + fx.Int32(1)
                if tid == fx.Int32(0):
                    fx.ptr_store(fetched, tk_lds)
                gpu.barrier()
                t = rocdl.readfirstlane(T.i32, fx.Int32(R) + fx.Int32(fx.ptr_load(tk_lds)))
                if tid == fx.Int32(0):
                    if t == n_items + fx.Int32(R - 1):
                        # this XCD's last ticket of the launch: reset the count, advance the epoch
                        _store_agent(head, fx.Int32(0))
                        _store_agent(head + fx.Int64(4), epoch + fx.Int32(1))
            if const_expr(stamps):
                if tid == fx.Int32(0):
                    fx.ptr_store(fx.Int64(k), global_typed_ptr(
                        stamp_base, T.i64, align=8, byte_offset=fx.Int64(STAMP_WORDS * 8 - 8)))

        @flyc.jit
        def launch(
            arg_w1: fx.Int64,
            arg_w1s: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_mind: fx.Int64,
            arg_inter: fx.Int64,
            arg_inter_s: fx.Int64,
            arg_hidden: fx.Int64,
            arg_w2: fx.Int64,
            arg_w2s: fx.Int64,
            arg_stids: fx.Int64,
            arg_sw: fx.Int64,
            arg_out: fx.Int64,
            arg_sync: fx.Int64,
            arg_stamps: fx.Int64,
            i32_ntok: fx.Int32,
            i32_max_m_blocks: fx.Int32,
            stream: fx.Stream,
        ):
            kernel(
                arg_w1, arg_w1s, arg_eids, arg_cumsum, arg_mind, arg_inter, arg_inter_s, arg_hidden,
                arg_w2, arg_w2s, arg_stids, arg_sw, arg_out, arg_sync, arg_stamps, i32_ntok, i32_max_m_blocks,
                value_attrs={"rocdl.waves_per_eu": waves_per_eu},
            ).launch(grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)

        return launch

    launcher = compile_gemm2_a4w4_port(
        BM=BM,
        BN=s2_bn,
        BK=s2_bk,
        use_nt=s2_use_nt,
        HIDDEN_MAX=8192,
        epilog="atomic",
        INTER_MAX=inter,
        a_dtype="fp8",
        b_dtype="fp4",
        topk=1,
        SBM=BM,
        persist=False,
        g2_bhoist=True,
        g2_ascale_pf=True,
        g2_spart=402,
        g2_bf16_lds=True,
        out_dtype="bf16",
        _composition=compose,
    )
    if not post_misched:
        launcher.compile_hints["llvm_options"] = {"enable-post-misched": False}
    else:
        launcher.compile_hints.pop("llvm_options", None)
    return launcher, grid
