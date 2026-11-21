# SPDX-License-Identifier: Apache-2.0
"""K2b ``moe``: one Kimi-K3 latent-MoE layer, from the MoE input row x to the
layer's output, every TP reduction inside the kernel.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``, S <= 8 rows (a
spec-verify step). The original path (vLLM ROCm) replicates the
router and the latent down projection on every rank; here each rank computes
its eighth and the rows are all-gathered over the peer buffer.

    early (CTAs 0 .. 130, one task each, x from the input row; weights first):
      router  (7 x 16 rows of this rank's 112 experts): fp32 logits -> every
              rank's RLOG
      latent  (28 x 16 rows of this rank's 448): bf16 -> every rank's LAT
      sgu     (96 x 16 rows of the shared gate | up): fp32 -> SGU
    route   (S, CTAs 131 ..): sigmoid (v_rcp), + bias, top-16, the pre-bias
            sigmoids renormalized (IEEE 1 / sum) -> ROUTE (every rank alike)
    sdown   (448 x 16 rows, K 768, CTAs 131 ..): SiTU of SGU (vLLM's kernel) ->
            bf16 -> GEMV -> bf16 partial; this rank's up columns -> SHP (plain,
            SHRDY flags), the rest -> every rank's FINAL
    ug      (|U| x 12, every CTA): the union U of the step's picks; an expert's
            32 gate + 32 up rows against the MXFP8 latent (aiter's A8W4 inline
            quant), SiTUv2 (aiter's fp32 epilogue), bf16, MXFP8 -> INTER / ISC a
            pick
    down    (224 x 16 latent rows, every CTA): w2 of every expert of U against
            its picks' INTER, bf16(w acc) folded in each token's top-k order with
            a bf16 rounding an add -> every rank's LATR
    lnorm   (28 x 128 columns): the ranks' LATR in rank order, bf16, RMSNorm
            (the fused AR + RMSNorm form) -> LN (plain, LNRDY flags)
    up      (56 x 16 rows of this rank's 896): LN GEMV + SHP -> bf16 -> every
            rank's FINAL
    reduce  (56 x 128 columns, ``reduce``): FINAL in rank order -> bf16 out

Peer regions are double-buffered by the epoch's parity.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from sgl_amd_mono.common.mx import (
    FP4,
    FP8,
    FP8_MAX,
    UNIT_SCALE,
    fp4_tile_load,
    mfma_scaled,
    scale_index,
)
from sgl_amd_mono.common.ops import (
    CM_DEV,
    _xpartner,
    ballot,
    bf2_f32,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    block_excl_scan,
    butterfly,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    lanes_below,
    ld_i32,
    lds_barrier,
    load_ptr64,
    peer_bases,
    popcount,
    read_lane_i32,
    row_sum,
    rsrc,
    sum_partials,
    uniform,
    wave_incl_scan,
    wave_sum,
    wave_umax,
    write_lane_i32,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    kernel_sources,
    key_tuple,
    pair_layout,
)
from sgl_amd_mono.common.sync import (
    POLL_MAX,
    Mailbox,
    poll_sent,
    preg,
    publish,
    put_sent,
    rearm_sent,
    sent_item,
    sent_ix,
    sent_sums,
    sent_word,
    shift,
    sreg,
)
from sgl_amd_mono.kimi_k3.gfx950.stages import gemv as gemv
from sgl_amd_mono.kimi_k3.gfx950.stages.dims import NA
from sgl_amd_mono.kimi_k3.gfx950.stamps import (
    stamp,
    stamp_begin,
    stamp_flush,
)
from sgl_amd_mono.kimi_k3.gfx950.tag import step_tag

SOURCES = kernel_sources(__name__)

REGIONS = (
    "sgu",
    "h",
    "rpart",
    "lpart",
    "route",
    "inter",
    "isc",
    "lmsq",
    "lnrdy",
    "shrdy",
    "rlog",
    "lat",
    "latq",
    "latr",
    "final",
    "lqr",
    "ugd",
    "dnd",
)


TP = 8
HIDDEN = 7168
LAT = 3584
E = 896
TOPK = 16
RI = 384  # routed intermediate a rank
SI = 768  # shared intermediate a rank
ROWS = 16
LOG2E = 1.0 / math.log(2.0)
BETA, LBETA = 4.0, 25.0

R_GROUPS = E // TP // ROWS  # 7 router row groups a rank
R_PARTS = 4  # K quarters: the router is on the path to the route
L_GROUPS = LAT // TP // ROWS  # 28 latent row groups a rank
L_PARTS = 2  # K halves: the latent is on the path to ug
# latq_push: the latent as 32-row groups (one MX block of columns) in K
# quarters, so part 0 holds whole blocks and quantizes them itself
LQ_GROUPS = LAT // TP // 32  # 14
LQ_PARTS = 4
assert LQ_GROUPS * LQ_PARTS == L_GROUPS * L_PARTS
LQ_WORDS = LAT // 4 + LAT // 32  # a token's fp8 words, then its scale codes
R_TASKS = R_GROUPS * R_PARTS  # 28
L_TASKS = L_GROUPS * L_PARTS  # 56
G_TASKS = 2 * SI // ROWS  # 96
EARLY = R_TASKS + L_TASKS + G_TASKS  # 180
LATE0 = EARLY  # route from here
H_CTA = LATE0 + 8  # the shared experts' SiTU
SPARE0 = H_CTA + 8  # 196: CTAs with no early task run the shared down first
SD_TASKS = HIDDEN // ROWS  # 448
UP_N = HIDDEN // TP  # 896 columns a rank up-projects
UP_TASKS = UP_N // ROWS  # 56
LN_TASKS = LAT // 128  # 28
LN0 = 56  # lnorm's CTAs
RED_TASKS = HIDDEN // 128  # 56
RED0 = 200
UG_GROUPS = RI // 32  # 12
IW = RI // 4  # a pick's INTER words (MXFP8)
W13_K = LAT  # 3584
W13_STEPS = W13_K // 128  # 28
W2_STEPS = RI // 128  # 3
DOWN_TASKS = LAT // ROWS  # 224
DOWN_BATCH = 5  # a wave's experts whose w2 is in flight together
# the wide shared gate/up stages its K three rounds of chunks at a time
SGU_SEGMENTS = ((0, 3), (3, 6), (6, 7))


WIDE_ABOVE = 8  # wider steps take the wide form (``wide``)


def wide(s):
    """Whether an s-row step takes the wide form (two DSpark requests on): GEMVs
    a 16-row token tile at a time, the latent and INTER from L2, the route table
    in units."""
    return s > WIDE_ABOVE


def max_u(s):
    """The step's distinct experts at most (the route table's slots)."""
    return min(E, s * TOPK)


def xl_words(s):
    """The x rows' LDS: a narrow step's whole rows; a wide step's 16-row tile
    of 3584 + pad (or every row of the shared h)."""
    if wide(s):
        return max(ROWS * (LAT + gemv.LDS_PAD), s * (SI + gemv.LDS_PAD)) // 2
    return s * (HIDDEN + gemv.LDS_PAD) // 2


TL_POINTS = 8  # start, early, sdown, ug, down, lnorm, up, reduce

_STREAM = fx.Stream(None)


class MoeBuild(msgspec.Struct, frozen=True):
    tokens: int
    eps: float = 1e-5  # the latent RMSNorm
    reduce: bool = True  # the final reduce in this launch (standalone)
    timeline: bool = False
    # timing experiments (narrow steps): 1 ug without its epilogue, 2 ug loads only
    xp: int = 0
    # debug: run phases < stop only (1 early, 2 route/sdown, 3 ug, 4 down,
    # 5 lnorm, 6 up)
    stop: int = 99
    # w13's AITER layout: False is gate/up separated (``gate_up=False``, what
    # the kernel was written for), True is gate/up interleaved, which is what
    # SGLang's Mxfp4MoEMethod produces for Kimi-K3 under AITER_SITUV2_A8W4.
    # Only the 16-row block index differs, so it costs one remap in ug_loads.
    guint: bool = False


def peer_pairs(s):
    """One parity's peer regions, pairs: name -> count."""
    return {
        "rlog": TP * s * (E // TP),
        "lat": s * LAT // 2,
        # latq_push: every rank's MXFP8 latent words and e8m0 codes, a token a row
        "latq": s * LQ_WORDS,
        "latr": TP * s * LAT // 2,
        "final": TP * s * HIDDEN // 2,
        # every router K part's partial, [src rank][part][token][router row]
        "rlogp": TP * R_PARTS * s * (E // TP),
    }


def peer_layout(s):
    return pair_layout(peer_pairs(s).items())


def peer_bytes(s):
    lay = peer_layout(s)
    return 2 * max(o + n for o, n in lay.values())


def scratch_layout(key: MoeBuild) -> dict:
    s = key.tokens
    pairs = {
        "sgu": s * 2 * SI,
        "h": s * SI // 2,
        "rpart": R_GROUPS * R_PARTS * s * ROWS,
        # sized for latq_push's 32-row groups in quarters (twice the halves')
        "lpart": LQ_GROUPS * LQ_PARTS * s * 2 * ROWS,
        "route": s * 2 * TOPK,
        "inter": s * TOPK * IW,
        "isc": s * TOPK * UG_GROUPS,
        "lmsq": LN_TASKS * s,
        "lnrdy": LN_TASKS,
        "shrdy": UP_TASKS,
        # wide steps: the compact latent copy's (token, quarter) flags and the
        # CTAs' ug-done flags
        "lqr": 4 * s,
        "ugd": BLOCKS,
        # wide steps: the down CTAs' done flags
        "dnd": BLOCKS,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    lay["ln"] = ((end + 15) // 16 * 16, s * LAT * 2)
    end = lay["ln"][0] + lay["ln"][1]
    lay["shp"] = ((end + 15) // 16 * 16, s * UP_N * 2)
    end = lay["shp"][0] + lay["shp"][1]
    lay["lqc"] = ((end + 15) // 16 * 16, s * LQ_WORDS * 4)
    end = lay["lqc"][0] + lay["lqc"][1]
    # wide steps: each pick's bf16(w acc) of every latent row (``fold_wide``)
    lay["contrib"] = ((end + 15) // 16 * 16, s * TOPK * LAT * 2 if wide(s) else 0)
    return lay


# the task queues, their own buffer: [slot 256][epoch parity][ug, down] counters
# (persistent words: a shared scratch's other launches would overwrite them)
QUEUE_BYTES = 256 * 2 * 2 * 4


def scratch_bytes(key: MoeBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def f32_of(w):
    return w.bitcast(fx.Float32)


def fp8_operand(lds, row, st, kb):
    """A lane's MXFP8 MFMA operand of K step ``st`` from a natural-order row at
    LDS word ``row``: lane group ``kb`` takes the step's 16-byte pieces kb and
    kb + 4 (the f8f6f4 MFMA's 8-bit operand layout; its scale is block kb)."""
    halves = [
        fx.Vector(
            fx.ptr_load(
                lds + (row + st * 32 + 4 * kb + 16 * h),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        for h in range(2)
    ]
    return fx.Vector.from_elements([h[d] for h in halves for d in range(4)], fx.Int32)


def tanh_libm(x):
    """The ``tanhf`` of x to a few ulp: (1 - e) / (1 + e), e = exp(-2|x|), x's sign."""
    e = hw_exp2(abs(x) * fx.Float32(-2.0 * LOG2E))
    t = (fx.Float32(1.0) - e) / (fx.Float32(1.0) + e)
    return (x < fx.Float32(0.0)).select(-t, t)


def situ_vllm(g, u):
    """The ``situ_and_mul`` of vLLM on bf16 g, u:
    ((2 tanh(g/4)) (1 + tanh(g/2))) (25 tanh(u/25))."""
    a = fx.Float32(0.5 * BETA) * tanh_libm(g * fx.Float32(1.0 / BETA))
    a = a * (fx.Float32(1.0) + tanh_libm(g * fx.Float32(0.5)))
    return a * (fx.Float32(LBETA) * tanh_libm(u * fx.Float32(1.0 / LBETA)))


def tanh_aiter(x):
    """The ``act.tanh_f32`` of aiter: (1 - e) rcp(1 + e), e = exp2(|x| (-2 log2 e))."""
    e = hw_exp2(abs(x) * fx.Float32(-2.0 * LOG2E))
    t = (fx.Float32(1.0) - e) * hw_rcp(fx.Float32(1.0) + e)
    return (x > fx.Float32(0.0)).select(t, -t)


def situ_aiter(g, u):
    """The situv2 epilogue of aiter on fp32 accumulators."""
    sg = hw_rcp(fx.Float32(1.0) + hw_exp2(g * fx.Float32(-LOG2E)))
    a = (fx.Float32(BETA) * tanh_aiter(g * fx.Float32(1.0 / BETA))) * sg
    return (a * fx.Float32(LBETA)) * tanh_aiter(u * fx.Float32(1.0 / LBETA))


def e8m0_roundup(amax):
    """The ``_e8m0_roundup(amax, 448)`` of aiter's A8W4 MoE (its MXFP8 inline
    quant and fp8out epilogue): ceil code of amax * f32(1/448), at most 254."""
    wi = (amax * fx.Float32(1.0 / FP8_MAX)).bitcast(fx.Int32)
    b = fx.Int32(fx.Uint32(wi + fx.Int32(0x7FFFFF)) >> fx.Uint32(23)) & 0xFF
    return fx.min(b, fx.Int32(254))


def _pk16():
    return fx.Vector.make_type(2, fx.Int16)


def _pk_word(pk):
    return fx.Vector(pk).bitcast(fx.Int32)[0]


def pk_fp8_f32(vals, scale):
    """Four f32 / scale -> one word of four E4M3 bytes (the scaled convert)."""
    sc = as_ir_value(fx.Float32(scale))
    pk = fx.full(2, 0, fx.Int16)
    for h in range(2):
        pk = rocdl.cvt_scalef32_pk_fp8_f32(
            _pk16(),
            pk,
            as_ir_value(fx.Float32(vals[2 * h])),
            as_ir_value(fx.Float32(vals[2 * h + 1])),
            sc,
            h,
        )
    return _pk_word(pk)


def pk_fp8_bf16(words, scale):
    """Two packed bf16 pair words / scale -> one word of four E4M3 bytes."""
    sc = as_ir_value(fx.Float32(scale))
    pk = fx.full(2, 0, fx.Int16)
    for h in range(2):
        v = fx.Vector.from_elements([fx.Int32(words[h])], fx.Int32).bitcast(fx.BFloat16)
        pk = rocdl.cvt_scalef32_pk_fp8_bf16(_pk16(), pk, as_ir_value(v), sc, h)
    return _pk_word(pk)


def quant_block_bf16(ws):
    """One 32-column block (16 packed bf16 pair words) -> (8 MXFP8 words, its
    e8m0 code), as aiter's A8W4 inline quant: the amax of the bf16 bits."""
    am = fx.Int32(0)
    for w in ws:
        am = fx.max(am, w & 0x7FFF)
        am = fx.max(am, (w >> 16) & 0x7FFF)
    code = e8m0_roundup((am << 16).bitcast(fx.Float32))
    scale = (code << 23).bitcast(fx.Float32)
    return [pk_fp8_bf16(ws[2 * q : 2 * q + 2], scale) for q in range(8)], code


# ---------------------------------------------------------------- early
@flyc.jit
def early_out(c, kind, i, q, red, t0=0, sj=None):
    """Task (i, q) of ``kind`` (0 router, 1 latent, 2 sgu): its 16 rows. A K-split
    kind's parts q > 0 publish fp32 partials; part 0 adds them in part order and
    publishes the rows (router: fp32 to every rank's RLOG, latent: bf16 to every
    rank's LAT); sgu bf16 to SGU. Token rows t0 .. t0 + sj - 1 (all S by
    default) in ``red``'s columns 0 ...."""
    s, tid = c["S"], c["tid"]
    sj = s if sj is None else sj
    if const_expr((kind == 0) and c.get("rlog_parts", False)):
        # every part's fp32 partial to every rank's RLOGP: the route sums the
        # parts in part order (as part 0 would), one hop fewer
        if tid < ROWS * sj:
            r = tid % ROWS
            tl = tid // ROWS
            t = t0 + tl
            v = row_sum(red, r, tl)
            for p in range_constexpr(TP):
                c["put"](
                    c["peer"]["rlogp"][p],
                    rlogp_pair(c["rank"], q, s, t, i * ROWS + r),
                    v,
                )
        gpu.barrier()
        stamp(c["tl_on"], c["tls"], c["tid"], 23)
    elif const_expr(kind == 2):
        if tid < (ROWS // 2) * sj:
            rp = tid % (ROWS // 2)
            tl = tid // (ROWS // 2)
            t = t0 + tl
            # fp32, as SGLang's shared gate/up output: SiTU rounds once, to bf16 h
            at = t * 2 * SI + i * ROWS + 2 * rp
            c["put"](c["sgu"], at, row_sum(red, 2 * rp, tl))
            c["put"](c["sgu"], at + 1, row_sum(red, 2 * rp + 1, tl))
    else:
        parts = R_PARTS if kind == 0 else L_PARTS
        region = c["rpart"] if kind == 0 else c["lpart"]
        if tid < ROWS * sj:
            r = tid % ROWS
            tl = tid // ROWS
            t = t0 + tl
            v = row_sum(red, r, tl)
            at = lambda qq: ((i * parts + qq) * s + t) * ROWS + r  # noqa: E731
            if q > 0:
                c["put"](region, at(q), v)
            if q == 0:
                got = c["poll"]([(region, at(qq), 1) for qq in range(1, parts)])
                for qq in range_constexpr(parts - 1):
                    v = v + f32_of(got[qq][0])
                fx.ptr_store(v, c["av"] + tid)
        gpu.barrier()
        if const_expr(kind == 0):
            stamp(c["tl_on"], c["tls"], c["tid"], 23)  # router: parts summed
        if const_expr(kind == 0):
            if (q == 0) & (tid < ROWS * sj):
                r = tid % ROWS
                tl = tid // ROWS
                t = t0 + tl
                v = fx.ptr_load(c["av"] + tid)
                for p in range_constexpr(TP):
                    c["put"](
                        c["peer"]["rlog"][p],
                        (c["rank"] * s + t) * (E // TP) + i * ROWS + r,
                        v,
                    )
        else:
            if (q == 0) & (tid < (ROWS // 2) * sj):
                rp = tid % (ROWS // 2)
                tl = tid // (ROWS // 2)
                t = t0 + tl
                v0 = bf16_round(fx.ptr_load(c["av"] + (tl * ROWS + 2 * rp)))
                v1 = bf16_round(fx.ptr_load(c["av"] + (tl * ROWS + 2 * rp + 1)))
                col = c["rank"] * (LAT // TP) + i * ROWS + 2 * rp
                for p in range_constexpr(TP):
                    c["put_bf"](c["peer"]["lat"][p], t * LAT + col, [v0, v1])
    gpu.barrier()


@flyc.jit
def early_out_lq(c, i, q, accs, t0=0, sj=None):
    """latq_push latent task (i, q): its two 16-row tiles' fp32 partials; part 0
    adds the parts in part order, rounds each row to bf16 (what LAT would hold),
    and quantizes the group's 32 columns -- one MX block -- as load_latq does,
    then pushes the block's 8 words and its code to every rank's LATQ. Token
    rows t0 .. t0 + sj - 1 (all S by default)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    sj = s if sj is None else sj
    region = c["lpart"]
    for h in range_constexpr(2):
        fx.ptr_store(accs[h], red + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < ROWS * sj:
            r = tid % ROWS
            tl = tid // ROWS
            t = t0 + tl
            v = row_sum(red, r, tl)
            at = lambda qq: ((i * LQ_PARTS + qq) * s + t) * 2 * ROWS + h * ROWS + r  # noqa: E731
            if q > 0:
                c["put"](region, at(q), v)
            if q == 0:
                got = c["poll"]([(region, at(qq), 1) for qq in range(1, LQ_PARTS)])
                for qq in range_constexpr(LQ_PARTS - 1):
                    v = v + f32_of(got[qq][0])
                fx.ptr_store(bf16_round(v), c["av"] + (tl * 2 * ROWS + h * ROWS + r))
        gpu.barrier()
    if (q == 0) & (tid < sj):
        t = t0 + tid
        ws = [
            bf16_pair(
                fx.ptr_load(c["av"] + (tid * 2 * ROWS + 2 * j)),
                fx.ptr_load(c["av"] + (tid * 2 * ROWS + 2 * j + 1)),
            ).bitcast(fx.Int32)
            for j in range(16)
        ]
        pks, code = quant_block_bf16(ws)
        blk = c["rank"] * LQ_GROUPS + i
        for p in range_constexpr(TP):
            # two pairs a store: a buffer store moves at most 128 bits
            for q in range_constexpr(4):
                c["put_words"](
                    c["peer"]["latq"][p],
                    t * LQ_WORDS + blk * 8 + 2 * q,
                    pks[2 * q : 2 * q + 2],
                )
            c["put_words"](c["peer"]["latq"][p], t * LQ_WORDS + LAT // 4 + blk, [code])
    gpu.barrier()


@flyc.jit
def early_lq_wide(c, i, q, ops):
    """latq_push latent task (i, q) past one token tile: each tile's two 16-row
    halves' partials (``early_out_lq``'s); part q > 0 pushes them a tile at a
    time, part 0 keeps its own and polls every tile's other parts once, after
    its last tile, then sums each in part order, rounds, quantizes and pushes
    as ``early_out_lq``."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    klen, _ = early_shape(3)
    xrow = klen + gemv.LDS_PAD
    region = c["lpart"]
    tiles = gemv.tiles(s)
    at = lambda qq, t, h, r: ((i * LQ_PARTS + qq) * s + t) * 2 * ROWS + h * ROWS + r  # noqa: E731
    own = []
    for ti in range_constexpr(len(tiles)):
        t0, sj = tiles[ti]
        poll_rows_to_lds(c, c["xt"], klen, sj, c["xl"], xrow, HIDDEN, q * klen, t0)
        if const_expr(t0 == 0):
            stamp(c["tl_on"], c["tls"], c["tid"], 33)  # latent: x tile 0 in
        accs = [gemv.mfmas(lane, wave, c["xl"], xrow, klen, sj, o) for o in ops]
        for h in range_constexpr(2):
            fx.ptr_store(accs[h], red + (wave * 64 + lane) * 4)
            lds_barrier()
            r = tid % ROWS
            t = t0 + fx.min(tid // ROWS, sj - 1)
            v = row_sum(red, r, fx.min(tid // ROWS, sj - 1))
            if (q > 0) & (tid < ROWS * sj):
                c["put"](region, at(q, t, h, r), v)
            own.append(v)
            lds_barrier()
    if q == 0:
        # thread (row, token) of every tile: its other parts, one batch a tile
        for ti in range_constexpr(len(tiles)):
            t0, sj = tiles[ti]
            r = tid % ROWS
            tl = fx.min(tid // ROWS, sj - 1)
            if tid < ROWS * sj:
                specs = [
                    (region, at(qq, t0 + tl, h, r), 1)
                    for h in range(2)
                    for qq in range(1, LQ_PARTS)
                ]
                got = c["poll"](specs, batch=len(specs))
                for h in range_constexpr(2):
                    v = own[2 * ti + h]
                    for qq in range_constexpr(LQ_PARTS - 1):
                        v = v + f32_of(got[h * (LQ_PARTS - 1) + qq][0])
                    fx.ptr_store(bf16_round(v), c["av"] + (tl * 2 * ROWS + h * ROWS + r))
            lds_barrier()
            if tid < sj:
                t = t0 + tid
                ws = [
                    bf16_pair(
                        fx.ptr_load(c["av"] + (tid * 2 * ROWS + 2 * j)),
                        fx.ptr_load(c["av"] + (tid * 2 * ROWS + 2 * j + 1)),
                    ).bitcast(fx.Int32)
                    for j in range(16)
                ]
                pks, code = quant_block_bf16(ws)
                blk = c["rank"] * LQ_GROUPS + i
                for p in range_constexpr(TP):
                    for qw in range_constexpr(4):
                        c["put_words"](
                            c["peer"]["latq"][p],
                            t * LQ_WORDS + blk * 8 + 2 * qw,
                            pks[2 * qw : 2 * qw + 2],
                        )
                    c["put_words"](c["peer"]["latq"][p], t * LQ_WORDS + LAT // 4 + blk, [code])
            lds_barrier()
    gpu.barrier()
    stamp(c["tl_on"], c["tls"], c["tid"], 34)  # latent: parts summed, LATQ pushed


def early_shape(kind):
    """(K a part, parts) of an early kind: router a quarter, latent a half, sgu all."""
    return {
        0: (HIDDEN // R_PARTS, R_PARTS),
        1: (HIDDEN // L_PARTS, L_PARTS),
        2: (HIDDEN, 1),
        3: (HIDDEN // LQ_PARTS, LQ_PARTS),
    }[kind]


def early_loads(c, kind, i, q):
    """Early task (i, q) of ``kind``'s weights for this wave: issued as early as
    the CTA can, then ``early_compute`` takes them."""
    a = c["args"]
    klen, _ = early_shape(kind)
    if kind == 3:
        # latq_push: rows 32 i .. of this rank's latent, as two 16-row tiles
        row0 = c["rank"] * (LAT // TP) + i * 2 * ROWS
        return [
            gemv.loads(c["lane"], c["wave"], a["w_ld"], row0 + h * ROWS, klen,
                       ldk=HIDDEN, k0=q * klen)
            for h in range(2)
        ]  # fmt: skip
    if kind == 0:
        w, row0 = a["w_gate"], c["rank"] * (E // TP) + i * ROWS
    elif kind == 1:
        w, row0 = a["w_ld"], c["rank"] * (LAT // TP) + i * ROWS
    else:
        w, row0 = a["w_sgu"], i * ROWS
    return gemv.loads(c["lane"], c["wave"], w, row0, klen, ldk=HIDDEN, k0=q * klen)


def early_compute(c, kind, i, q, ops):
    """Early task (i, q) of ``kind`` against ``ops`` (``early_loads``)."""
    if wide(c["S"]):
        return _early_wide(c, kind, i, q, ops)
    return _early_narrow(c, kind, i, q, ops)


@flyc.jit
def _early_wide(c, kind, i, q, ops):
    """A wide step's early task: the latent's (``early_lq_wide``), the router's
    or the shared gate/up's (``_early_wide_gemv``)."""
    if const_expr(kind == 3):
        return early_lq_wide(c, i, q, ops)
    return _early_wide_gemv(c, kind, i, q, ops)


@flyc.jit
def _early_wide_gemv(c, kind, i, q, ops):
    """``_early_narrow`` past one MFMA token tile: x's rows 16 at a time into
    LDS (the router from the mix's tagged pairs, the shared gate/up from the
    plain rows a K segment at a time), each tile's GEMV and ``early_out`` in
    turn; the weights stay in registers."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    klen, _ = early_shape(kind)
    xrow = klen + gemv.LDS_PAD
    if const_expr(kind == 2):
        if tid < c["xflags"]:
            c["poll"]([(c["xrdy"], tid, 1)])
        gpu.barrier()
    tiles = gemv.tiles(s)
    for ti in range_constexpr(len(tiles)):
        t0, sj = tiles[ti]
        if const_expr(kind == 2):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for si in range_constexpr(len(SGU_SEGMENTS)):
                r0, r1 = SGU_SEGMENTS[si]
                k0, k1 = r0 * WAVES * gemv.KCH, r1 * WAVES * gemv.KCH
                srow = k1 - k0 + gemv.LDS_PAD
                gemv.rows_to_lds(
                    tid, c["xbuf"], k1 - k0, sj, c["xl"], srow, CM_DEV, ldk=HIDDEN,
                    k0=k0, t0=t0,
                )  # fmt: skip
                acc = gemv.mfmas_rounds(
                    lane, wave, c["xl"], srow, k0, sj, ops, acc, r0, r1
                )
                gpu.barrier()
        else:
            poll_rows_to_lds(c, c["xt"], klen, sj, c["xl"], xrow, HIDDEN, q * klen, t0)
            if const_expr(kind == 0 and t0 == 0):
                stamp(c["tl_on"], c["tls"], c["tid"], 20)  # router: x published
        if const_expr(kind != 2):
            acc = gemv.mfmas(lane, wave, c["xl"], xrow, klen, sj, ops)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        early_out(c, kind, i, q, red, t0, sj)
    if const_expr(kind == 0):
        stamp(c["tl_on"], c["tls"], c["tid"], 24)  # router: logits pushed


@flyc.jit
def _early_narrow(c, kind, i, q, ops):
    """Early task (i, q): x's K slice (the input row, or -- ``xflags`` -- the
    row this launch's AttnRes stage publishes) against ``ops`` -> ``early_out``."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    klen, _ = early_shape(kind)
    xrow = klen + gemv.LDS_PAD
    if const_expr(c.get("tag_x", False) and kind != 2):
        # the rows as the mix's tagged pairs: no wait on every slice's XRDY
        poll_rows_to_lds(c, c["xt"], klen, s, c["xl"], xrow, HIDDEN, q * klen)
        if const_expr(kind == 0):
            stamp(c["tl_on"], c["tls"], c["tid"], 20)  # router: x published
    elif const_expr(c.get("xflags") is not None):
        if tid < c["xflags"]:
            c["poll"]([(c["xrdy"], tid, 1)])
        gpu.barrier()
        if const_expr(kind == 0):
            stamp(c["tl_on"], c["tls"], c["tid"], 20)  # router: x published
        gemv.rows_to_lds(
            tid, c["xbuf"], klen, s, c["xl"], xrow, CM_DEV, ldk=HIDDEN, k0=q * klen
        )
    else:
        gemv.rows_to_lds(
            tid, c["args"]["x"], klen, s, c["xl"], xrow, ldk=HIDDEN, k0=q * klen
        )
    if const_expr(kind == 3):
        accs = [gemv.mfmas(lane, wave, c["xl"], xrow, klen, s, o) for o in ops]
        early_out_lq(c, i, q, accs)
    else:
        if const_expr(kind == 0):
            gpu.barrier()
            stamp(c["tl_on"], c["tls"], c["tid"], 21)  # router: x in LDS
        acc = gemv.mfmas(lane, wave, c["xl"], xrow, klen, s, ops)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        if const_expr(kind == 0):
            stamp(c["tl_on"], c["tls"], c["tid"], 22)  # router: GEMV done
        early_out(c, kind, i, q, red)
        if const_expr(kind == 0):
            stamp(c["tl_on"], c["tls"], c["tid"], 24)  # router: logits pushed


TAG_BATCH = 14  # tagged-row specs a poll batch (2 pairs a spec: 56 registers)


@flyc.jit
def poll_rows_to_lds(c, region, k, s, xl, xrow, ldk, k0, t0=0):
    """``gemv.rows_to_lds`` from tagged pairs: columns k0 .. k0 + k of rows t0 ..
    t0 + s - 1 of the [., ldk] bf16 rows a producer put as (word, tag) pairs
    into ``region``, polled -> LDS rows 0 .., ``xrow`` a row."""
    tid = c["tid"]
    quads = s * k // 4  # 4 bf16 = 2 pairs a spec
    per = (quads + THREADS - 1) // THREADS
    for b0 in range_constexpr(0, per, TAG_BATCH):
        ii = list(range(b0, min(per, b0 + TAG_BATCH)))
        es = [fx.min(tid + THREADS * i, quads - 1) for i in ii]
        tk = [(e // (k // 4), e % (k // 4) * 4) for e in es]
        got = c["poll"](
            [(region, ((t0 + t) * ldk + k0 + kk) // 2, 2) for t, kk in tk],
            batch=len(ii),
        )
        for j in range_constexpr(len(ii)):
            t, kk = tk[j]
            if tid + THREADS * ii[j] < quads:
                fx.ptr_store(
                    fx.Vector.from_elements(got[j], fx.Int32), xl + (t * xrow + kk) // 2
                )
    gpu.barrier()


def rlogp_pair(src, q, s, t, r):
    """RLOGP's pair of (source rank, router K part q, token t, the rank's row r)."""
    return ((src * R_PARTS + q) * s + t) * (E // TP) + r


@flyc.jit
def route_parts(c, t):
    """Token t's router partials (every rank's R_PARTS K parts, RLOGP) -> LDS
    ``rlp`` [part][expert]: wave q polls part q, the 14 experts of its lane in
    one batch."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    if wave < R_PARTS:
        per = E // 64
        specs = []
        for i in range_constexpr(per):
            e = lane + 64 * i
            specs.append(
                (
                    c["peer"]["rlogp_own"],
                    rlogp_pair(e // (E // TP), wave, s, t, e % (E // TP)),
                    1,
                )
            )
        got = c["poll"](specs, batch=per)
        for i in range_constexpr(per):
            fx.ptr_store(f32_of(got[i][0]), c["rlp"] + (wave * E + lane + 64 * i))
    gpu.barrier()


@flyc.jit
def stage_early(c, bid):
    """K2b alone: CTA bid's early task (router parts, latent parts, sgu)."""
    lb = bid - R_TASKS
    gb = bid - R_TASKS - L_TASKS
    if bid < R_TASKS:
        early_compute(
            c,
            0,
            bid // R_PARTS,
            bid % R_PARTS,
            early_loads(c, 0, bid // R_PARTS, bid % R_PARTS),
        )
    if (bid >= R_TASKS) & (bid < R_TASKS + L_TASKS):
        early_compute(
            c,
            1,
            lb // L_PARTS,
            lb % L_PARTS,
            early_loads(c, 1, lb // L_PARTS, lb % L_PARTS),
        )
    if bid >= R_TASKS + L_TASKS:
        early_compute(c, 2, gb, 0, early_loads(c, 2, gb, 0))


# ---------------------------------------------------------------- route
def _batcher(n):
    """Batcher's odd-even merge sort network on n (a power of two) slots: (a, b) pairs."""
    pairs = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return pairs


ROUTE_SLOTS = 16  # a lane's 14 candidates (896 / 64) padded to the network's width
ROUTE_NET = _batcher(ROUTE_SLOTS)


def _route_extract2(lane, keys, sigs, lds):
    """The top-16 by max-extraction, ``_route_rank``'s fallback: each lane sorts
    its candidates as packed (key, 15 - slot) u64s, then 16 rounds of a wave max
    on the heads pop the winner's head. A pick finds its lane by a ballot of the
    max key (the lowest expert among equal keys by a second max only when keys
    tie), the lanes' slot lists shift as one packed u64 of nibbles, and the
    sigmoids wait in ``lds`` (an f32 an expert) for the picks' lanes to read once.
    Pick k lands in lane k already ranked: the radix select's picks, order and
    sigmoids, bit for bit."""
    per = len(keys)
    for i in range_constexpr(per):
        fx.ptr_store(sigs[i], lds + (lane + 64 * i))
    pk = [
        (fx.Uint64(fx.Uint32(keys[i])) << fx.Uint64(32))
        | fx.Uint64(ROUTE_SLOTS - 1 - i)
        for i in range(per)
    ] + [fx.Uint64(0)] * (ROUTE_SLOTS - per)
    for a, b in ROUTE_NET:
        first = pk[a] > pk[b]
        pa, pb = pk[a], pk[b]
        pk[a], pk[b] = first.select(pa, pb), first.select(pb, pa)
    ks = [fx.Int32(fx.Uint32(p >> fx.Uint64(32))) for p in pk] + [fx.Int32(0)]
    nib = fx.Uint64(0)
    for j in range_constexpr(ROUTE_SLOTS):
        slot = fx.Uint64(ROUTE_SLOTS - 1) - (pk[j] & fx.Uint64(ROUTE_SLOTS - 1))
        nib = nib | (slot << fx.Uint64(4 * j))
    ev = fx.Int32(0)
    for k in range_constexpr(TOPK):
        head = ks[0]
        m = wave_umax(head)
        top = head == m
        bal = ballot(top)
        islot = fx.Int32(fx.Uint32(nib & fx.Uint64(15)))
        wl = fx.Int32(fx.cttz(bal))
        expert = wl + 64 * read_lane_i32(islot, wl)
        tie = fx.Int32(1023) - wave_umax(
            top.select(fx.Int32(1023) - (lane + 64 * islot), fx.Int32(0))
        )
        expert = (popcount(bal) == 1).select(expert, tie)
        hit = lane == (expert & 63)
        ks = [hit.select(ks[i + 1], ks[i]) for i in range(ROUTE_SLOTS)] + [ks[-1]]
        nib = hit.select(nib >> fx.Uint64(4), nib)
        ev = write_lane_i32(expert, k, ev)
    sv = fx.ptr_load(lds + fx.min(fx.max(ev, 0), fx.Int32(E - 1))).bitcast(fx.Int32)
    return ev, sv


# a 16-lane row's bitonic network: (span, distance) a stage
_ROW_NET = [(sp, sp >> k) for sp in (2, 4, 8, 16) for k in range(1, sp.bit_length())]
CAND0 = E  # route 2: the candidates' u64s after the experts' sigmoids in ``lds``


@flyc.jit
def _route_rank(lane, keys, sigs, lds):
    """The top-16 with no serial pick rounds: the same picks, order and
    sigmoids as the radix select, bit for bit. T is the least of
    the 16-lane rows' 4th largest lane maxima (a bitonic sort a row, DPP
    only): 16 keys are >= T, so the top-16 are among the candidates, the keys
    >= T. The candidates go to ``lds`` as distinct u64s, (key << 32) | (1023 -
    expert), larger exactly when they rank first, a ballot a slot placing
    them; each lane takes one and counts the larger ones, its rank. More than
    64 candidates (lanes holding many) falls back to ``_route_extract2``."""
    per = len(keys)
    for i in range_constexpr(per):
        fx.ptr_store(sigs[i], lds + (lane + 64 * i))
    mx = keys[0]
    for i in range_constexpr(1, per):
        mx = fx.max(mx, keys[i])
    fx.ptr_store(fx.Float32(0.0), lds + (CAND0 + 2 * lane))
    fx.ptr_store(fx.Float32(0.0), lds + (CAND0 + 2 * lane + 1))
    # each 16-lane row's maxima sorted (DPP only); T, the least of the rows'
    # 4th largest: 4 maxima a row at or above it, 16 in all
    for q in range_constexpr(len(_ROW_NET)):
        span, d = _ROW_NET[q]
        p = fx.Uint32(_xpartner(fx.Int32(mx), d, lane))
        if const_expr(span < 16):
            keep_max = ((lane & d) == 0) == ((lane & span) == 0)
        else:
            keep_max = (lane & d) == 0
        mx = keep_max.select(fx.max(mx, p), fx.min(mx, p))
    th = None
    for r in range_constexpr(4):
        v = fx.Uint32(read_lane_i32(fx.Int32(mx), 16 * r + 3))
        th = v if th is None else fx.min(th, v)
    # the candidates, slot by slot: a slot's lanes in lane order after the
    # earlier slots' (a ballot a slot; the running base is wave-uniform)
    base = fx.Int32(0)
    for i in range_constexpr(per):
        hit = keys[i] >= th
        bal = ballot(hit)
        at = base + lanes_below(bal, lane)
        if hit & (at < 64):
            fx.ptr_store(
                (fx.Int32(1023) - (lane + 64 * i)).bitcast(fx.Float32),
                lds + (CAND0 + 2 * at),
            )
            fx.ptr_store(
                fx.Int32(keys[i]).bitcast(fx.Float32), lds + (CAND0 + 2 * at + 1)
            )
        base = base + popcount(bal)
    total = base
    j = fx.min(lane, fx.Int32(63))
    mine = (
        fx.Uint64(fx.Uint32(fx.ptr_load(lds + (CAND0 + 2 * j + 1)).bitcast(fx.Int32)))
        << fx.Uint64(32)
    ) | fx.Uint64(fx.Uint32(fx.ptr_load(lds + (CAND0 + 2 * j)).bitcast(fx.Int32)))
    lim = fx.min(total, fx.Int32(64))
    m_lo = fx.Int32(fx.Uint32(mine & fx.Uint64(0xFFFFFFFF)))
    m_hi = fx.Int32(fx.Uint32(mine >> fx.Uint64(32)))
    # each candidate's rank: the larger candidates, read lane by lane (8 a
    # round, as far as the candidates go: lanes past them hold 0)
    rank = fx.Int32(0)
    i = fx.Int32(0)
    while i < lim:
        for u in range_constexpr(8):
            o_lo = read_lane_i32(m_lo, i + u)
            o_hi = read_lane_i32(m_hi, i + u)
            o = (fx.Uint64(fx.Uint32(o_hi)) << fx.Uint64(32)) | fx.Uint64(
                fx.Uint32(o_lo)
            )
            rank = rank + (o > mine).select(fx.Int32(1), fx.Int32(0))
        i = i + 8
    ex = fx.Int32(1023) - fx.Int32(fx.Uint32(mine & fx.Uint64(0x3FF)))
    if (lane < lim) & (rank < TOPK):
        fx.ptr_store(ex.bitcast(fx.Float32), lds + (CAND0 + rank))
    ev = fx.ptr_load(lds + (CAND0 + fx.min(lane, fx.Int32(TOPK - 1)))).bitcast(fx.Int32)
    if total > 64:
        ev, _ = _route_extract2(lane, keys, sigs, lds)
    sv = fx.ptr_load(lds + fx.min(fx.max(ev, 0), fx.Int32(E - 1))).bitcast(fx.Int32)
    return ev, sv


def _okey(v):
    """An order-preserving uint32 key of an f32 (larger float, larger key)."""
    b = v.bitcast(fx.Int32)
    neg = b < fx.Int32(0)
    return fx.Uint32(neg.select(~b, b ^ fx.Int32(-2147483648)))


@flyc.jit
def stage_route(c, t):
    """Token t (wave 0): 896 logits, 14 a lane (expert lane + 64 i) -> sigmoid
    (v_rcp) + bias -> the top-16 (equal keys: the lower experts) ranked by
    score, descending (aiter's slot order) -> the pre-bias sigmoids renormalized
    in that order (IEEE 1 / sum). ``route_v`` 2 (the layer launch) ranks the
    candidates (``_route_rank``); 0 (the MoE launch) runs the radix select (the
    16th largest key by a 32-step bit search of ballot counts), the same picks
    bit for bit."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    a = c["args"]
    # read before the traced branch (a dict lookup inside it is carried state)
    route_v = c.get("route_v", 0)
    rlog_parts = c.get("rlog_parts", False)
    batch = c.get("batch", False)
    per = E // 64
    # the bias first: its loads land while the logits' parts are awaited
    bias = []
    for i in range_constexpr(per):
        e = lane + 64 * i
        bias.append(bo_ld_f32(a["bias"], e))
    if const_expr(rlog_parts):
        route_parts(c, t)
    if wave == 0:
        s = c["S"]
        if const_expr(rlog_parts):
            # the parts summed in part order (``route_parts`` put them in LDS)
            lgs = []
            for i in range_constexpr(per):
                e = lane + 64 * i
                v = fx.ptr_load(c["rlp"] + e)
                for q in range_constexpr(1, R_PARTS):
                    v = v + fx.ptr_load(c["rlp"] + (q * E + e))
                lgs.append(v)
        else:
            specs = []
            for i in range_constexpr(per):
                e = lane + 64 * i
                src = e // (E // TP)
                specs.append(
                    (
                        c["peer"]["rlog_own"],
                        (src * s + t) * (E // TP) + e % (E // TP),
                        1,
                    )
                )
            # batch: the 14 in one round trip (POLL_MAX splits them in two)
            got = c["poll"](specs, batch=per) if batch else c["poll"](specs)
            lgs = [f32_of(got[i][0]) for i in range(per)]
        # 14: every rank's router logits landed (the route's wait, apart from its select)
        stamp(c["tl_on"], c["tls"], c["tid"], 14)
        keys, sigs = [], []
        for i in range_constexpr(per):
            e = lane + 64 * i
            lg = lgs[i]
            sg = hw_rcp(fx.Float32(1.0) + hw_exp2(lg * fx.Float32(-LOG2E)))
            sc = sg + bias[i]
            # a NaN logit (a warmup's garbage row) ranks last
            keys.append(_okey((sc == sc).select(sc, fx.Float32(-3.0e38))))
            sigs.append((sg == sg).select(sg, fx.Float32(0.0)))
        stamp(c["tl_on"], c["tls"], c["tid"], 17)  # keys (sigmoid + bias)
        if const_expr(route_v == 2):
            ev, sv = _route_rank(lane, keys, sigs, red)
            stamp(c["tl_on"], c["tls"], c["tid"], 19)  # selected and ranked
            total = fx.Float32(0.0)
            for j in range_constexpr(TOPK):
                # summed serially in rank order (aiter's renorm)
                total = total + read_lane_i32(sv, j).bitcast(fx.Float32)
            inv = fx.Float32(1.0) / total
            if lane < TOPK:
                c["put"](
                    c["route"],
                    t * 2 * TOPK + lane,
                    fx.min(fx.max(ev, 0), fx.Int32(E - 1)),
                )
                c["put"](
                    c["route"], t * 2 * TOPK + TOPK + lane, sv.bitcast(fx.Float32) * inv
                )
        else:
            _route_radix(c, t, lane, red, keys, sigs, per)
    gpu.barrier()


@flyc.jit
def _route_radix(c, t, lane, red, keys, sigs, per):
    """The radix-select route (``route_v`` 0): the 16th largest key by a 32-step
    bit search of ballot counts, then the 16 ranked by score."""

    def count_ge(th):
        n = fx.Int32(0)
        for i in range_constexpr(per):
            n = n + popcount(ballot(keys[i] >= th))
        return n

    th = fx.Uint32(0)
    for bit in range_constexpr(31, -1, -1):
        cand = th | fx.Uint32(1 << bit)
        th = (count_ge(cand) >= TOPK).select(cand, th)
    stamp(c["tl_on"], c["tls"], c["tid"], 18)  # the 16th key
    # selected: key > th, then the lowest experts with key == th
    n_gt = fx.Int32(0)
    for i in range_constexpr(per):
        n_gt = n_gt + popcount(ballot(keys[i] > th))
    need = fx.Int32(TOPK) - n_gt
    eq_before = fx.Int32(0)
    sel_before = fx.Int32(0)
    for i in range_constexpr(per):
        eqb = ballot(keys[i] == th)
        eq_rank = eq_before + lanes_below(eqb, lane)
        sel = (keys[i] > th) | ((keys[i] == th) & (eq_rank < need))
        selb = ballot(sel)
        pos = sel_before + lanes_below(selb, lane)
        if sel:
            fx.ptr_store(keys[i].bitcast(fx.Float32), red + pos * 3)
            fx.ptr_store(
                fx.Int32(lane + 64 * i).bitcast(fx.Float32), red + (pos * 3 + 1)
            )
            fx.ptr_store(sigs[i], red + (pos * 3 + 2))
        eq_before = eq_before + popcount(eqb)
        sel_before = sel_before + popcount(selb)
    # rank the 16 by key descending, the lower expert first on a tie
    q = fx.min(lane, TOPK - 1)
    mk = fx.Uint32(fx.ptr_load(red + q * 3).bitcast(fx.Int32))
    me = fx.ptr_load(red + (q * 3 + 1)).bitcast(fx.Int32)
    ms = fx.ptr_load(red + (q * 3 + 2))
    rank = fx.Int32(0)
    for j in range_constexpr(TOPK):
        ok = fx.Uint32(fx.ptr_load(red + j * 3).bitcast(fx.Int32))
        oe = fx.ptr_load(red + (j * 3 + 1)).bitcast(fx.Int32)
        before = (ok > mk) | ((ok == mk) & (oe < me))
        rank = rank + before.select(fx.Int32(1), fx.Int32(0))
    stamp(c["tl_on"], c["tls"], c["tid"], 19)  # selected and ranked
    # the sigmoids in rank order, summed serially (aiter's renorm)
    if lane < TOPK:
        fx.ptr_store(ms, red + (64 + rank))
    total = fx.Float32(0.0)
    for j in range_constexpr(TOPK):
        total = total + fx.ptr_load(red + (64 + j))
    inv = fx.Float32(1.0) / total
    if lane < TOPK:
        c["put"](
            c["route"], t * 2 * TOPK + rank, fx.min(fx.max(me, 0), fx.Int32(E - 1))
        )
        c["put"](c["route"], t * 2 * TOPK + TOPK + rank, ms * inv)


def bo_ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


@flyc.jit
def load_route2(c):
    """``load_route`` with fewer hand-offs (``rtab`` 2): the same LDS tables,
    bit for bit. The union is a 28-word bitmask of the picked experts (an LDS
    atomic OR a pick); one wave prefix-sums the words' popcounts, and a pick's
    slot is its word's prefix plus the set bits below it -- the experts in
    ascending order, as the block scan placed them. ``flag`` holds the mask
    (words 0 .. 27) and the prefixes (32 .., the total at 32 + 28)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    n = s * 2 * TOPK
    nw = E // 32
    for i in range_constexpr((max_u(s) * s + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < max_u(s) * s:
            fx.ptr_store(fx.Int32(-1), rt["kof"] + j)
    if tid < nw:
        fx.ptr_store(fx.Int32(0), rt["flag"] + tid)
    if tid < n:
        v = c["poll"]([(c["route"], tid, 1)])[0][0]
        fx.ptr_store(v, rt["route"] + tid)
    gpu.barrier()
    pick = tid < s * TOPK
    t = tid // TOPK
    k = tid % TOPK
    e = fx.max(
        fx.min(fx.ptr_load(rt["route"] + fx.min(t * 2 * TOPK + k, n - 1)), E - 1), 0
    )
    if pick:
        fx.llvm.atomic_or(
            rt["flag"] + (e >> 5), fx.Int32(1) << (e & 31), syncscope="workgroup"
        )
    gpu.barrier()
    if wave == 0:
        w = (lane < nw).select(
            fx.ptr_load(rt["flag"] + fx.min(lane, nw - 1)), fx.Int32(0)
        )
        cnt = popcount(fx.Uint32(w))
        incl = wave_incl_scan(cnt, lane)
        if lane < nw:
            fx.ptr_store(incl - cnt, rt["flag"] + (32 + lane))
        if lane == nw - 1:
            fx.ptr_store(incl, rt["flag"] + (32 + nw))
    gpu.barrier()
    if pick:
        w = fx.ptr_load(rt["flag"] + (e >> 5))
        below = w & ((fx.Int32(1) << (e & 31)) - 1)
        slot = fx.ptr_load(rt["flag"] + (32 + (e >> 5))) + popcount(fx.Uint32(below))
        fx.ptr_store(e, rt["expert"] + slot)
        fx.ptr_store(k, rt["kof"] + (slot * s + t))
    tot = fx.ptr_load(rt["flag"] + (32 + nw))
    gpu.barrier()
    return uniform(tot)


@flyc.jit
def load_route(c):
    """ROUTE -> LDS (ids, weights); the union U: slot_of[e], expert_of[slot],
    kof[slot][t] (the token's top-k index, -1 if not picked); returns |U|."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    n = s * 2 * TOPK
    if tid < n:
        v = c["poll"]([(c["route"], tid, 1)])[0][0]
        fx.ptr_store(v, rt["route"] + tid)
    for i in range_constexpr((E + THREADS - 1) // THREADS):
        e = tid + THREADS * i
        if e < E:
            fx.ptr_store(fx.Int32(0), rt["flag"] + e)
    for i in range_constexpr((max_u(s) * s + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < max_u(s) * s:
            fx.ptr_store(fx.Int32(-1), rt["kof"] + j)
    gpu.barrier()
    if tid < s * TOPK:
        t = tid // TOPK
        k = tid % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * TOPK + k)), E - 1), 0)
        fx.ptr_store(fx.Int32(1), rt["flag"] + e)
    gpu.barrier()
    e0 = fx.min(2 * tid, E - 1)
    f0 = (2 * tid < E).select(fx.ptr_load(rt["flag"] + e0), fx.Int32(0))
    f1 = (2 * tid + 1 < E).select(
        fx.ptr_load(rt["flag"] + fx.min(2 * tid + 1, E - 1)), fx.Int32(0)
    )
    off, tot = block_excl_scan(f0 + f1, lane, wave, c["scan"])
    gpu.barrier()
    if f0 == 1:
        fx.ptr_store(off, rt["flag"] + 2 * tid)
        fx.ptr_store(fx.Int32(2 * tid), rt["expert"] + off)
    if f1 == 1:
        fx.ptr_store(off + f0, rt["flag"] + 2 * tid + 1)
        fx.ptr_store(fx.Int32(2 * tid + 1), rt["expert"] + off + f0)
    gpu.barrier()
    if tid < s * TOPK:
        t = tid // TOPK
        k = tid % TOPK
        e = fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * TOPK + k)), E - 1), 0)
        slot = fx.ptr_load(rt["flag"] + e)
        fx.ptr_store(fx.Int32(k), rt["kof"] + (slot * s + t))
    gpu.barrier()
    return uniform(tot)


@flyc.jit
def load_latq(c):
    """LAT (every rank's 448 columns) -> MXFP8 as aiter's A8W4 inline quant: a
    thread a (token, 32-block): amax of |bf16| bits, the ceil code of amax /
    448, v_cvt_scalef32_pk_fp8_bf16 -> latq words, latsc codes. Under ``latq_push``
    the producers quantized it: every rank's words and codes are polled in."""
    s, tid = c["S"], c["tid"]
    push = c.get("latq_push", False)
    if const_expr(push):
        n = s * LQ_WORDS
        per = (n + THREADS - 1) // THREADS
        got = c["poll"](
            [
                (c["peer"]["latq_own"], fx.min(tid + THREADS * i, n - 1), 1)
                for i in range(per)
            ]
        )
        for i in range_constexpr(per):
            e = tid + THREADS * i
            ec = fx.min(e, n - 1)
            t = ec // LQ_WORDS
            w = ec % LQ_WORDS
            if e < n:
                if w < LAT // 4:
                    fx.ptr_store(got[i][0], c["latq"] + (t * (LAT // 4) + w))
                else:
                    fx.ptr_store(
                        got[i][0], c["latsc"] + (t * (LAT // 32) + w - LAT // 4)
                    )
        gpu.barrier()
        return
    nblk = s * (LAT // 32)
    for i in range_constexpr((nblk + THREADS - 1) // THREADS):
        b = fx.min(tid + THREADS * i, nblk - 1)
        t = b // (LAT // 32)
        kb = b % (LAT // 32)
        base = (t * LAT + kb * 32) // 2
        got = c["poll"]([(c["peer"]["lat_own"], base + j, 1) for j in range(16)])
        ws = [got[j][0] for j in range(16)]
        pks, code = quant_block_bf16(ws)
        if tid + THREADS * i < nblk:
            for q in range_constexpr(8):
                fx.ptr_store(pks[q], c["latq"] + (t * (LAT // 4) + kb * 8 + q))
            fx.ptr_store(code, c["latsc"] + (t * (LAT // 32) + kb))
    gpu.barrier()


# ---------------------------------------------------------------- ug
def w13_scale_word(e, rg, st, lane):
    """The dword of aiter's gate_up=False ``shuffle_scale`` holding (row rg 16 +
    lane % 16, K block 4 st + lane / 16) for st's pair and rg's pair: byte
    (st % 2) 2 + rg % 2."""
    n1 = rg // 2
    k1 = st // 2
    return (
        e * (2 * RI * (W13_K // 32) // 4)
        + ((n1 * (W13_STEPS // 2) + k1) * 4 + lane // 16) * 16
        + lane % 16
    )


def ug_row_group(c, g, wave):
    """This wave's 16-row block of w13 for 32-column group ``g``.

    ``rgl`` picks gate row group 0/1 (waves 0, 1) or up row group 0/1 (2, 3).
    AITER's gate_up=False layout keeps the 384 gate rows before the 384 up
    rows; gate_up=True alternates the two 16-row blocks. The scale word packs
    an adjacent pair of blocks and selects the byte by the low bit, so both the
    load and the byte select have to use the same number -- hence one helper.
    """
    rgl = wave % 4
    j, up = 2 * g + rgl % 2, rgl // 2
    return 2 * j + up if c["guint"] else up * (RI // ROWS) + j


def ug_loads(c, task):
    """Task ``task``'s w13 operands for this wave (its row group, its K half)
    and their scale words: issued a task ahead (``stage_ug`` takes them)."""
    lane, wave = c["lane"], c["wave"]
    a, rt = c["args"], c["rt"]
    slot = task // UG_GROUPS
    g = task % UG_GROUPS
    e = uniform(fx.ptr_load(rt["expert"] + slot))
    half = wave // 4
    rg = ug_row_group(c, g, wave)
    r_w = rsrc(a["w13"])
    base = e * (2 * RI * W13_K // 8)
    steps = [half * (W13_STEPS // 2) + i for i in range(W13_STEPS // 2)]
    av = [fp4_tile_load(r_w, base, rg, W13_K, st, lane * 4) for st in steps]
    r_s = rsrc(a["w13s"])
    sw = [
        fx.Int32(
            bo.buffer_load(
                r_s, w13_scale_word(e, rg, st, lane), vec_width=1, dtype=T.i32
            )
        )
        for st in steps[::2]
    ]
    return av + sw


@flyc.jit
def stage_ug(c, task, ops):
    """Slot task / 12, 32-column group task % 12: gate rows 32 g .. and up rows
    384 + 32 g .. of the slot's expert (``ops``: ``ug_loads``) against the MXFP8
    latent; SiTUv2, rounded to bf16; MXFP8 a 32-column group -> INTER / ISC of
    each picking token."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt = c["rt"]
    slot = task // UG_GROUPS
    g = task % UG_GROUPS
    half = wave // 4
    rg = ug_row_group(c, g, wave)
    steps = [half * (W13_STEPS // 2) + i for i in range(W13_STEPS // 2)]
    av = ops[: len(steps)]
    sw = ops[len(steps) :]
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(len(steps) if c["xp"] != 2 else 0):
        st = steps[i]
        bw = fp8_operand(c["latq"], t * (LAT // 4), st, kb)
        sb = fx.ptr_load(c["latsc"] + (t * (LAT // 32) + st * 4 + kb))
        # the code's byte (st % 2) 2 + rg % 2 of the word; rg % 2 is the wave's
        sa = (sw[i // 2] >> ((st % 2) * 16 + (rg % 2) * 8)) & 0xFF
        acc = mfma_scaled(av[i], bw, acc, sa, sb, FP4, FP8, 0, 0)
    if const_expr(c["xp"] == 2):
        # loads only: keep them live
        acc = fx.Vector.from_elements(
            [
                acc[d] + av[-1][d].bitcast(fx.Float32) * fx.Float32(0.0)
                for d in range(4)
            ],
            fx.Float32,
        )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    lds_barrier()
    if const_expr(c["xp"] != 0):
        return
    # thread (token, column): g / u summed over the K halves, SiTUv2
    tt = tid // 32
    col = tid % 32
    live = tt < s
    tt = fx.min(tt, s - 1)
    first = col < 16  # gate / up row group 0 (waves 0, 2) or 1 (1, 3); +4: K's 2nd half
    r = col % 16
    gv = first.select(
        row_sum(red, r, tt, waves=(0, 4)), row_sum(red, r, tt, waves=(1, 5))
    )
    uv = first.select(
        row_sum(red, r, tt, waves=(2, 6)), row_sum(red, r, tt, waves=(3, 7))
    )
    # aiter's fp8out epilogue: the activation is materialized as bf16 first
    v = bf16_round(situ_aiter(gv, uv))
    amax = butterfly(abs(v), (16, 8, 4, 2, 1), fx.max)
    code = e8m0_roundup(amax)
    lds_barrier()
    fx.ptr_store(v, c["av"] + tid)
    lds_barrier()
    k = fx.ptr_load(rt["kof"] + (slot * s + tt))
    if live & (col < 8) & (k >= 0):
        scale = (code << 23).bitcast(fx.Float32)
        xs = [fx.ptr_load(c["av"] + (tt * 32 + col * 4 + j)) for j in range(4)]
        pick = tt * TOPK + k
        c["put_words"](c["inter"], pick * IW + g * 8 + col, [pk_fp8_f32(xs, scale)])
        if col == 0:
            c["put"](c["isc"], pick * UG_GROUPS + g, code)
    lds_barrier()


# ---------------------------------------------------------------- queues
@flyc.jit
def take(c, which):
    """The launch's next task of queue ``which`` (0 ug, 1 down) past the first
    static round: an agent-scope atomic on this slot's counter of this epoch's
    parity (the other parity's is reset by this launch for the next step)."""
    if c["tid"] == 0:
        at = c["queue"] + fx.Int64(((c["slot"] * 2 + c["par"]) * 2 + which) * 4)
        ptr = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), at)
        task = fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")
        fx.ptr_store(fx.Int32(task), c["qslot"])
    gpu.barrier()
    v = fx.ptr_load(c["qslot"])
    gpu.barrier()
    return BLOCKS + v


@flyc.jit
def reset_queues(c):
    """The next step's counters of this slot (the other parity) to zero."""
    if (c["bid"] == 0) & (c["tid"] < 2):
        bo.buffer_store(
            fx.Int32(0),
            rsrc(c["queue"]),
            (c["slot"] * 2 + 1 - c["par"]) * 2 + c["tid"],
            cache_modifier=CM_DEV,
        )


# ---------------------------------------------------------------- down
@flyc.jit
def load_inter(c):
    """Every pick's INTER words and ISC codes -> LDS."""
    s, tid = c["S"], c["tid"]
    nw = s * TOPK * IW
    per = (nw + THREADS - 1) // THREADS
    nc = s * TOPK * UG_GROUPS
    perc = (nc + THREADS - 1) // THREADS
    ispecs = [(c["inter"], fx.min(tid + THREADS * i, nw - 1), 1) for i in range(per)]
    cspecs = [(c["isc"], fx.min(tid + THREADS * i, nc - 1), 1) for i in range(perc)]
    if const_expr(c.get("batch", False) and per + perc <= 2 * POLL_MAX):
        # one round trip once the last ug task landed, not two
        both = c["poll"](ispecs + cspecs, batch=per + perc)
        got, gc = both[:per], both[per:]
    else:
        got, gc = c["poll"](ispecs), None
    for i in range_constexpr(per):
        if tid + THREADS * i < nw:
            fx.ptr_store(got[i][0], c["interq"] + (tid + THREADS * i))
    if const_expr(gc is None):
        gc = c["poll"](cspecs)
    for i in range_constexpr(perc):
        if tid + THREADS * i < nc:
            fx.ptr_store(gc[i][0], c["intsc"] + (tid + THREADS * i))
    gpu.barrier()


def w2_scale(c, e, rgd, st, lane):
    """The e8m0 code of w2 (row e 3584 + rgd 16 + lane % 16, K block 4 st +
    lane / 16) in ``e8m0_shuffle``'s layout (16 codes a row)."""
    row = e * LAT + rgd * ROWS + lane % 16
    idx = scale_index(row, st * 4 + lane // 16, 16)
    w = fx.Int32(
        bo.buffer_load(rsrc(c["args"]["w2s"]), idx // 4, vec_width=1, dtype=T.i32)
    )
    return (w >> ((idx % 4) * 8)) & 0xFF


def down_loads(c, rgd, nu, b0, nbatch=DOWN_BATCH):
    """The w2 tiles and scale codes of a wave's batch of ``nbatch`` from slot b0
    (slots b0, b0 + 8, ...; past |U| the last slot's): flat, ``stage_down``
    unpacks."""
    lane = c["lane"]
    r_w = rsrc(c["args"]["w2"])
    rt = c["rt"]
    slots = [fx.min(b0 + WAVES * j, nu - 1) for j in range(nbatch)]
    es = [uniform(fx.ptr_load(rt["expert"] + sl)) for sl in slots]
    av = [
        fp4_tile_load(r_w, es[j] * (LAT * RI // 8), rgd, RI, st, lane * 4)
        for j in range(nbatch)
        for st in range(W2_STEPS)
    ]
    sa = [
        w2_scale(c, es[j], rgd, st, lane)
        for j in range(nbatch)
        for st in range(W2_STEPS)
    ]
    return av + sa


@flyc.jit
def fold_latr(c, rgd, sent_latr):
    """Latent rows 16 rgd ..: each token's 16 contributions (``contrib``) folded
    in top-k order with a bf16 rounding an add -> every rank's LATR."""
    s, tid = c["S"], c["tid"]
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        y0 = fx.Float32(0.0)
        y1 = fx.Float32(0.0)
        for k in range_constexpr(TOPK):
            y0 = bf16_round(
                y0 + fx.ptr_load(c["contrib"] + ((tt * TOPK + k) * ROWS + 2 * rp))
            )
            y1 = bf16_round(
                y1 + fx.ptr_load(c["contrib"] + ((tt * TOPK + k) * ROWS + 2 * rp + 1))
            )
        if const_expr(sent_latr):
            si = sent_ix(c["rank"], rgd * ROWS + 2 * rp, tt, s, LAT)
            word = sent_word(bf16_pair(y0, y1))
            for p in range_constexpr(TP):
                put_sent(c["latrs_peer"][p], si, word)
        else:
            for p in range_constexpr(TP):
                c["put_bf"](
                    c["peer"]["latr"][p],
                    (c["rank"] * s + tt) * LAT + rgd * ROWS + 2 * rp,
                    [y0, y1],
                )
    gpu.barrier()


@flyc.jit
def stage_down(c, rgd, nu, ops0):
    """Latent rows 16 rgd ..: every slot's w2 against its picks (wave w the slots
    w, w + 8, ...; a batch of DOWN_BATCH in flight, the next issued under this
    one's MFMAs; ``ops0`` the first, issued before INTER was waited for) ->
    bf16(w acc) a (token, k) in LDS -> folded in top-k order with a bf16
    rounding an add -> every rank's LATR."""
    # read before the traced branches (a dict lookup inside one is carried state)
    sent_latr = c.get("sent_latr", False)
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    t = fx.min(lane % 16, s - 1)
    kb = lane // 16
    nbs = DOWN_BATCH * W2_STEPS
    b0 = wave
    ops = ops0
    while b0 < nu:
        nb = b0 + WAVES * DOWN_BATCH
        ops_n = down_loads(c, rgd, nu, fx.min(nb, nu - 1))
        slots = [fx.min(b0 + WAVES * j, nu - 1) for j in range(DOWN_BATCH)]
        for j in range_constexpr(DOWN_BATCH):
            live = (b0 + WAVES * j) < nu
            k = fx.ptr_load(rt["kof"] + (slots[j] * s + t))
            picked = (k >= 0) & (lane % 16 < s)
            pick = t * TOPK + fx.max(k, 0)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            # this slot's picks' INTER, polled here (not all of U first): a slot
            # whose ug tasks are done is multiplied while later ones still run
            for st in range_constexpr(W2_STEPS):
                bw = fp8_operand(c["interq"], pick * IW, st, kb)
                scode = fx.ptr_load(c["intsc"] + (pick * UG_GROUPS + st * 4 + kb))
                bw = fx.Vector.from_elements(
                    [picked.select(bw[d], fx.Int32(0)) for d in range(8)], fx.Int32
                )
                sb = picked.select(scode, fx.Int32(UNIT_SCALE))
                acc = mfma_scaled(
                    ops[j * W2_STEPS + st],
                    bw,
                    acc,
                    ops[nbs + j * W2_STEPS + st],
                    sb,
                    FP4,
                    FP8,
                    0,
                    0,
                )
            # lane: rows 4 (lane / 16) + i of token lane % 16
            if live & picked:
                wt = fx.ptr_load(rt["route"] + (t * 2 * TOPK + TOPK + k)).bitcast(
                    fx.Float32
                )
                for i in range_constexpr(4):
                    row = 4 * (lane // 16) + i
                    fx.ptr_store(
                        bf16_round(acc[i] * wt),
                        c["contrib"] + ((t * TOPK + k) * ROWS + row),
                    )
        b0 = nb
        ops = ops_n
    gpu.barrier()
    fold_latr(c, rgd, sent_latr)



# ---------------------------------------------------------------- wide steps
# Past one MFMA token tile (S > 16) the latent, INTER and the route table do
# not fit a CTA's LDS: the latent is copied once into a compact scratch copy
# (LQC), ug and down read their operands from L2 a unit at a time, and the
# route table lists each expert's picking tokens, cut into units of <= 16.


def max_units(s):
    """The step's units (an expert's <= 16 picking tokens) at most."""
    return max_u(s) + s * TOPK // ROWS


# the latent copy's CTAs: the top ones, past the shared gate/up's (which end last)
COPY_CTAS = BLOCKS - (NA + G_TASKS)


@flyc.jit
def copy_latq(c):
    """Every rank's MXFP8 latent words and codes (LATQ) -> the compact LQC, a
    (token, quarter) unit at a time on the top COPY_CTAS CTAs (their early work
    ends before the shared gate/up's), each unit's LQR flag (``wait_latq`` polls
    them)."""
    s, tid, bid = c["S"], c["tid"], c["bid"]
    qw = LQ_WORDS // 4
    v = BLOCKS - 1 - bid
    for h in range_constexpr((4 * s + COPY_CTAS - 1) // COPY_CTAS):
        u = v + h * COPY_CTAS
        if (v < COPY_CTAS) & (u < 4 * s):
            w = u * qw + fx.min(tid, qw - 1)
            got = c["poll"]([(c["peer"]["latq_own"], w, 1)])[0][0]
            if tid < qw:
                bo.buffer_store(got, rsrc(c["lqc"]), w, cache_modifier=CM_DEV)
            publish(c["put"], c["lqr"], u, fx.Int32(1), tid == 0)
    if v < COPY_CTAS:
        stamp(c["tl_on"], c["tls"], c["tid"], 35)  # latent copied


@flyc.jit
def wait_latq(c):
    """Every LQR flag: the compact latent copy is complete."""
    if c["tid"] < 4 * c["S"]:
        c["poll"]([(c["lqr"], c["tid"], 1)])
    lds_barrier()


@flyc.jit
def load_route_w(c):
    """The route table past one token tile: the experts in ascending order (as
    ``load_route2``), each one's picking tokens in token order (``toks``: the
    pick t TOPK + k), cut into units of <= 16 (``expert`` and ``ustart``: first
    entry | count << 16, a unit). Returns the units."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    n = s * 2 * TOPK
    npk = s * TOPK
    nw = E // 32
    mu = max_u(s)
    for i in range_constexpr((2 * E + THREADS - 1) // THREADS):
        j = tid + THREADS * i
        if j < 2 * E:
            fx.ptr_store(fx.Int32(0), rt["mask"] + j)
    if tid < nw:
        fx.ptr_store(fx.Int32(0), rt["flag"] + tid)
    per = (n + THREADS - 1) // THREADS
    got = c["poll"](
        [(c["route"], fx.min(tid + THREADS * i, n - 1), 1) for i in range(per)]
    )
    for i in range_constexpr(per):
        if tid + THREADS * i < n:
            fx.ptr_store(got[i][0], rt["route"] + (tid + THREADS * i))
    gpu.barrier()
    picks = []
    for i in range_constexpr((npk + THREADS - 1) // THREADS):
        live = (tid + THREADS * i) < npk
        pc = fx.min(tid + THREADS * i, npk - 1)
        t = pc // TOPK
        e = fx.max(
            fx.min(fx.ptr_load(rt["route"] + (t * 2 * TOPK + pc % TOPK)), E - 1), 0
        )
        if live:
            fx.llvm.atomic_or(
                rt["flag"] + (e >> 5), fx.Int32(1) << (e & 31), syncscope="workgroup"
            )
            fx.llvm.atomic_or(
                rt["mask"] + (e * 2 + (t >> 5)),
                fx.Int32(1) << (t & 31),
                syncscope="workgroup",
            )
        picks.append((live, pc, t, e))
    gpu.barrier()
    if wave == 0:
        w = (lane < nw).select(
            fx.ptr_load(rt["flag"] + fx.min(lane, nw - 1)), fx.Int32(0)
        )
        cnt = popcount(fx.Uint32(w))
        incl = wave_incl_scan(cnt, lane)
        if lane < nw:
            fx.ptr_store(incl - cnt, rt["flag"] + (32 + lane))
        if lane == nw - 1:
            fx.ptr_store(incl, rt["flag"] + (32 + nw))
    gpu.barrier()
    ranked = []
    for i in range_constexpr(len(picks)):
        live, pc, t, e = picks[i]
        w = fx.ptr_load(rt["flag"] + (e >> 5))
        below = w & ((fx.Int32(1) << (e & 31)) - 1)
        slot = fx.ptr_load(rt["flag"] + (32 + (e >> 5))) + popcount(fx.Uint32(below))
        m0 = fx.ptr_load(rt["mask"] + e * 2)
        m1 = fx.ptr_load(rt["mask"] + (e * 2 + 1))
        under = (fx.Int32(1) << (t & 31)) - 1
        lo = (t < 32).select(m0 & under, m0)
        hi = (t < 32).select(fx.Int32(0), m1 & under)
        rank = popcount(fx.Uint32(lo)) + popcount(fx.Uint32(hi))
        if live:
            fx.ptr_store(e, rt["sexp"] + slot)
            fx.ptr_store(
                popcount(fx.Uint32(m0)) + popcount(fx.Uint32(m1)), rt["cnt"] + slot
            )
        ranked.append((live, pc, slot, rank))
    tot = uniform(fx.ptr_load(rt["flag"] + (32 + nw)))
    gpu.barrier()
    # two slots a thread: their list and unit offsets
    sls = [2 * tid, 2 * tid + 1]
    cs = [(sl < tot).select(fx.ptr_load(rt["cnt"] + fx.min(sl, mu - 1)), 0) for sl in sls]
    us = [(cn + ROWS - 1) // ROWS for cn in cs]
    es = [fx.ptr_load(rt["sexp"] + fx.min(sl, mu - 1)) for sl in sls]
    off, _ = block_excl_scan(cs[0] + cs[1], lane, wave, c["scan"])
    uoff, utot = block_excl_scan(us[0] + us[1], lane, wave, c["scan"])
    los = [off, off + cs[0]]
    ulos = [uoff, uoff + us[0]]
    for h in range_constexpr(2):
        if sls[h] < tot:
            fx.ptr_store(los[h], rt["toff"] + sls[h])
        for jj in range_constexpr((s + ROWS - 1) // ROWS):
            if (sls[h] < tot) & (jj * ROWS < cs[h]):
                fx.ptr_store(es[h], rt["expert"] + (ulos[h] + jj))
                fx.ptr_store(
                    (los[h] + jj * ROWS) | (fx.min(cs[h] - jj * ROWS, ROWS) << 16),
                    rt["ustart"] + (ulos[h] + jj),
                )
    gpu.barrier()
    for i in range_constexpr(len(ranked)):
        live, pc, slot, rank = ranked[i]
        if live:
            fx.ptr_store(pc, rt["toks"] + (fx.ptr_load(rt["toff"] + slot) + rank))
    gpu.barrier()
    return uniform(utot)


def unit_of(c, u):
    """Unit u's (first ``toks`` entry, tokens)."""
    w = uniform(fx.ptr_load(c["rt"]["ustart"] + u))
    return w & 0xFFFF, w >> 16


UG_BQ = (ROWS * (LQ_WORDS // 4) + THREADS - 1) // THREADS  # 16 B pieces a thread


def ug_b_loads(c, task):
    """Task's latent rows (its unit's tokens' LQC rows), this thread's share in
    16 B pieces; a row past the unit reads zeros without touching memory.
    Issued with the task's weights, a task ahead."""
    s, tid = c["S"], c["tid"]
    st0, n = unit_of(c, task // UG_GROUPS)
    qw = LQ_WORDS // 4
    nb = s * LQ_WORDS * 4
    r = rsrc(c["lqc"], nb)
    out = []
    for i in range(UG_BQ):
        qd = tid + THREADS * i
        j = qd // qw
        ok = (qd < ROWS * qw) & (j < n)
        p = fx.ptr_load(c["rt"]["toks"] + (st0 + fx.min(j, n - 1)))
        at = ok.select((p // TOPK) * LQ_WORDS + (qd % qw) * 4, fx.Int32(nb // 4))
        out.append(
            fx.Vector(
                bo.buffer_load(r, at, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)
            )
        )
    return out


@flyc.jit
def stage_ug_w(c, task, ops, bq):
    """``stage_ug`` for unit task / 12: its tokens' latent rows (``bq``) -> LDS,
    the same MFMAs against them, the same epilogue; INTER words and ISC codes
    stored plain (``wait_ug`` orders them)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt = c["rt"]
    g = task % UG_GROUPS
    st0, n = unit_of(c, task // UG_GROUPS)
    qw = LQ_WORDS // 4
    # the unit's rows only; a column past them reads the last row
    for i in range_constexpr(len(bq)):
        if tid + THREADS * i < n * qw:
            fx.ptr_store(bq[i], c["latq"] + (tid + THREADS * i) * 4)
    lds_barrier()
    half = wave // 4
    rg = ug_row_group(c, g, wave)
    steps = [half * (W13_STEPS // 2) + i for i in range(W13_STEPS // 2)]
    av = ops[: len(steps)]
    sw = ops[len(steps) :]
    t = lane % 16
    tr = fx.min(t, n - 1)
    kb = lane // 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(len(steps)):
        st = steps[i]
        bw = fp8_operand(c["latq"], tr * LQ_WORDS, st, kb)
        sb = fx.ptr_load(c["latq"] + (tr * LQ_WORDS + LAT // 4 + st * 4 + kb))
        sa = (sw[i // 2] >> ((st % 2) * 16 + (rg % 2) * 8)) & 0xFF
        acc = mfma_scaled(av[i], bw, acc, sa, sb, FP4, FP8, 0, 0)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    lds_barrier()
    tt = tid // 32
    col = tid % 32
    live = tt < n
    first = col < 16
    r = col % 16
    gv = first.select(
        row_sum(red, r, tt, waves=(0, 4)), row_sum(red, r, tt, waves=(1, 5))
    )
    uv = first.select(
        row_sum(red, r, tt, waves=(2, 6)), row_sum(red, r, tt, waves=(3, 7))
    )
    v = bf16_round(situ_aiter(gv, uv))
    amax = butterfly(abs(v), (16, 8, 4, 2, 1), fx.max)
    code = e8m0_roundup(amax)
    lds_barrier()
    fx.ptr_store(v, c["av"] + tid)
    lds_barrier()
    if live & (col < 8):
        p = fx.ptr_load(rt["toks"] + (st0 + fx.min(tt, n - 1)))
        scale = (code << 23).bitcast(fx.Float32)
        xs = [fx.ptr_load(c["av"] + (tt * 32 + col * 4 + j)) for j in range(4)]
        bo.buffer_store(
            pk_fp8_f32(xs, scale),
            rsrc(c["inter"].value),
            p * IW + g * 8 + col,
            cache_modifier=CM_DEV,
        )
        if col == 0:
            bo.buffer_store(
                code,
                rsrc(c["isc"].value),
                p * UG_GROUPS + g,
                cache_modifier=CM_DEV,
            )
    lds_barrier()


def ug_late_rounds(s):
    """The ug rounds the shared gate/up CTAs sit out: from three token tiles
    their GEMV (x streamed a K segment a tile) ends ~10 us after the route
    table, about two ug tasks."""
    return 2 if len(gemv.tiles(s)) >= 3 else 0


def ug_task(c, i):
    """This CTA's i-th ug task. The first ``ug_late_rounds`` rounds go to the
    CTAs that are not the shared gate/up's (G_TASKS from NA), then every CTA
    takes BLOCKS - 1 - bid + BLOCKS j: rising in i."""
    bid = c["bid"]
    rev = BLOCKS - 1 - bid
    r0 = ug_late_rounds(c["S"])
    if r0 == 0:
        return rev + BLOCKS * i
    ng = BLOCKS - G_TASKS
    is_g = (bid >= NA) & (bid < NA + G_TASKS)
    # its place among the others, the high CTAs first
    pos = (bid >= NA + G_TASKS).select(rev, ng - 1 - bid)
    late = is_g.select(rev + BLOCKS * i, rev + BLOCKS * (i - r0))
    return ((~is_g) & (i < r0)).select(pos + ng * i, r0 * ng + late)


@flyc.jit
def run_ug_wide(c, total):
    """This CTA's ug tasks (``ug_task``), a task's weights and latent rows a
    task ahead (the first task's weights in flight while the latent copy
    lands)."""
    last = total - 1
    i = fx.Int32(0)
    task = ug_task(c, i)
    ops = ug_loads(c, fx.min(task, last))
    wait_latq(c)
    stamp(c["tl_on"], c["tls"], c["tid"], 12)
    bq = ug_b_loads(c, fx.min(task, last))
    while task < total:
        i = i + 1
        nxt = ug_task(c, i)
        ops_n = ug_loads(c, fx.min(nxt, last))
        bq_n = ug_b_loads(c, fx.min(nxt, last))
        stage_ug_w(c, task, ops, bq)
        task = nxt
        ops = ops_n
        bq = bq_n


@flyc.jit
def ug_done(c):
    """This CTA's ug tasks stored: its UGD flag (after every plain store)."""
    publish(c["put"], c["ugd"], c["bid"], fx.Int32(1), c["tid"] == 0)


@flyc.jit
def wait_ug(c):
    """Every CTA's UGD flag: every pick's INTER and ISC are stored."""
    if c["tid"] < BLOCKS:
        c["poll"]([(c["ugd"], c["tid"], 1)])
    gpu.barrier()


# down past one token tile: tasks (unit, 256 latent rows), unit-major, on
# every CTA; a wave takes DOWN_NB 16-row blocks of the chunk. The unit's INTER
# rows and ISC codes go through LDS once a task (every wave reads them), the
# weights of DOWN_P tasks are in flight, and each pick's bf16(w acc) goes to
# the CONTRIB scratch; ``fold_wide`` folds them once every down CTA is done.
DOWN_NB = 2  # a wave's 16-row blocks a task
DOWN_CHUNK = WAVES * DOWN_NB * ROWS  # 256 latent rows a task
DOWN_CHUNKS = LAT // DOWN_CHUNK  # 14 tasks a unit
DOWN_P = 4  # tasks in flight a CTA
TILE_W = IW + UG_GROUPS  # a pick's tile row: INTER words, then ISC codes
TILE_I = ROWS * (IW // 4)  # the INTER pieces of a tile (16 B a thread)
TILE_C = ROWS * (UG_GROUPS // 4)  # the ISC pieces


def down_task(c, d):
    """Down task d: (unit, its first 16-row block for this wave)."""
    u = d // DOWN_CHUNKS
    rg0 = (d % DOWN_CHUNKS) * (WAVES * DOWN_NB) + c["wave"] * DOWN_NB
    return u, rg0


def down_w_loads(c, d, tile=True):
    """Task d's w2 tiles and scale codes for this wave's blocks, then (``tile``)
    ``down_tile_loads``: flat, ``stage_down_wide`` unpacks."""
    lane = c["lane"]
    u, rg0 = down_task(c, d)
    e = uniform(fx.ptr_load(c["rt"]["expert"] + u))
    r_w = rsrc(c["args"]["w2"])
    av = [
        fp4_tile_load(r_w, e * (LAT * RI // 8), rg0 + i, RI, st, lane * 4)
        for i in range(DOWN_NB)
        for st in range(W2_STEPS)
    ]
    sa = [
        w2_scale(c, e, rg0 + i, st, lane)
        for i in range(DOWN_NB)
        for st in range(W2_STEPS)
    ]
    return av + sa + (down_tile_loads(c, d) if tile else [])


def down_tile_loads(c, d):
    """This thread's INTER / ISC piece of task d's unit tile (a pick past the
    unit reads zeros without touching memory)."""
    s, tid = c["S"], c["tid"]
    st0, n = unit_of(c, down_task(c, d)[0])
    nbi = s * TOPK * IW * 4
    nbc = s * TOPK * UG_GROUPS * 4
    ji, ri = tid // (IW // 4), tid % (IW // 4)
    tc = tid - TILE_I
    jc, rc = tc // (UG_GROUPS // 4), tc % (UG_GROUPS // 4)
    pi = fx.ptr_load(c["rt"]["toks"] + (st0 + fx.min(ji, n - 1)))
    pc = fx.ptr_load(c["rt"]["toks"] + (st0 + fx.max(fx.min(jc, n - 1), 0)))
    oki = (tid < TILE_I) & (ji < n)
    okc = (tid >= TILE_I) & (tc < TILE_C) & (jc < n)
    at_i = oki.select(pi * IW + ri * 4, fx.Int32(nbi // 4))
    at_c = okc.select(pc * UG_GROUPS + rc * 4, fx.Int32(nbc // 4))
    vi = fx.Vector(
        bo.buffer_load(
            rsrc(c["inter"].value, nbi), at_i, vec_width=4, dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )  # fmt: skip
    vc = fx.Vector(
        bo.buffer_load(
            rsrc(c["isc"].value, nbc), at_c, vec_width=4, dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )  # fmt: skip
    return [vi, vc]


@flyc.jit
def stage_down_wide(c, d, ops, buf, live, nxt):
    """Task d (``ops``: ``down_w_loads``): the unit's tile -> LDS ``buf``, this
    wave's blocks of w2 against it (the narrow path's MFMAs and K order), the
    next loads (``nxt``) issued behind the MFMAs, then bf16(w acc) a pick and
    row -> CONTRIB. ``live``: d is a task (else nothing is stored). Returns the
    next loads."""
    s, tid, lane = c["S"], c["tid"], c["lane"]
    nab = DOWN_NB * W2_STEPS
    vi, vc = ops[2 * nab], ops[2 * nab + 1]
    tile = c["dtile"] + buf * (ROWS * TILE_W)
    tc = tid - TILE_I
    if tid < TILE_I:
        fx.ptr_store(vi, tile + ((tid // (IW // 4)) * TILE_W + (tid % (IW // 4)) * 4))
    if (tid >= TILE_I) & (tc < TILE_C):
        fx.ptr_store(
            vc,
            tile
            + ((tc // (UG_GROUPS // 4)) * TILE_W + IW + (tc % (UG_GROUPS // 4)) * 4),
        )
    lds_barrier()
    t = lane % 16
    kb = lane // 16
    accs = []
    for i in range_constexpr(DOWN_NB):
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for st in range_constexpr(W2_STEPS):
            bw = fp8_operand(tile, t * TILE_W, st, kb)
            sb = fx.ptr_load(tile + (t * TILE_W + IW + st * 4 + kb))
            acc = mfma_scaled(
                ops[i * W2_STEPS + st], bw, acc, ops[nab + i * W2_STEPS + st], sb,
                FP4, FP8, 0, 0,
            )  # fmt: skip
        accs.append(acc)
    out = down_w_loads(c, nxt)
    u, rg0 = down_task(c, d)
    st0, n = unit_of(c, u)
    if live & (t < n):
        p = fx.ptr_load(c["rt"]["toks"] + (st0 + t))
        wt = fx.ptr_load(
            c["rt"]["route"] + ((p // TOPK) * 2 * TOPK + TOPK + p % TOPK)
        ).bitcast(fx.Float32)
        r = rsrc(c["contrib"], s * TOPK * LAT * 2)
        for i in range_constexpr(DOWN_NB):
            a = accs[i]
            at = (p * LAT + (rg0 + i) * ROWS + 4 * kb) // 2
            for h in range_constexpr(2):
                w = bf16_pair(a[2 * h] * wt, a[2 * h + 1] * wt)
                bo.buffer_store(w, r, at + h, cache_modifier=CM_DEV)
    return out


@flyc.jit
def run_down_wide(c, nu):
    """Down tasks bid, bid + BLOCKS, ... on every CTA: DOWN_P in flight (the
    first ones' weights issued before INTER is waited for), then DND."""
    bid = c["bid"]
    dt = BLOCKS
    total = nu * DOWN_CHUNKS
    last = total - 1
    d = bid
    ds = [fx.min(d + j * dt, last) for j in range(DOWN_P)]
    a0 = down_w_loads(c, ds[0], False)
    a1 = down_w_loads(c, ds[1], False)
    a2 = down_w_loads(c, ds[2], False)
    a3 = down_w_loads(c, ds[3], False)
    wait_ug(c)
    a0 = a0 + down_tile_loads(c, ds[0])
    a1 = a1 + down_tile_loads(c, ds[1])
    a2 = a2 + down_tile_loads(c, ds[2])
    a3 = a3 + down_tile_loads(c, ds[3])
    stamp(c["tl_on"], c["tls"], c["tid"], 15)
    while d < total:
        d1, d2, d3 = d + dt, d + 2 * dt, d + 3 * dt
        nx = [fx.min(x + DOWN_P * dt, last) for x in (d, d1, d2, d3)]
        a0 = stage_down_wide(c, d, a0, 0, d < total, nx[0])
        a1 = stage_down_wide(c, fx.min(d1, last), a1, 1, d1 < total, nx[1])
        a2 = stage_down_wide(c, fx.min(d2, last), a2, 0, d2 < total, nx[2])
        a3 = stage_down_wide(c, fx.min(d3, last), a3, 1, d3 < total, nx[3])
        d = d + DOWN_P * dt
    publish(c["put"], c["dnd"], bid, fx.Int32(1), c["tid"] == 0)


@flyc.jit
def fold_wide(c, rgd):
    """Latent rows 16 rgd ..: every CTA's DND, then each token's 16 CONTRIB
    words folded in top-k order with a bf16 rounding an add (the narrow fold)
    -> every rank's LATR."""
    sent_latr = c.get("sent_latr", False)
    s, tid = c["S"], c["tid"]
    if tid < BLOCKS:
        c["poll"]([(c["dnd"], tid, 1)])
    gpu.barrier()
    r = rsrc(c["contrib"], s * TOPK * LAT * 2)
    for j in range_constexpr((s * (ROWS // 2) + THREADS - 1) // THREADS):
        v = tid + THREADS * j
        if v < (ROWS // 2) * s:
            rp = v % (ROWS // 2)
            tt = v // (ROWS // 2)
            ws = [
                fx.Int32(
                    bo.buffer_load(
                        r, ((tt * TOPK + k) * LAT + rgd * ROWS + 2 * rp) // 2,
                        vec_width=1, dtype=T.i32, cache_modifier=CM_DEV,
                    )
                )
                for k in range(TOPK)
            ]  # fmt: skip
            y0 = fx.Float32(0.0)
            y1 = fx.Float32(0.0)
            for k in range_constexpr(TOPK):
                y0 = bf16_round(y0 + bf_lo(ws[k]))
                y1 = bf16_round(y1 + bf_hi(ws[k]))
            if const_expr(sent_latr):
                si = sent_ix(c["rank"], rgd * ROWS + 2 * rp, tt, s, LAT)
                word = sent_word(bf16_pair(y0, y1))
                for p in range_constexpr(TP):
                    put_sent(c["latrs_peer"][p], si, word)
            else:
                for p in range_constexpr(TP):
                    c["put_bf"](
                        c["peer"]["latr"][p],
                        (c["rank"] * s + tt) * LAT + rgd * ROWS + 2 * rp,
                        [y0, y1],
                    )
    gpu.barrier()
    # K2's point (the standalone MoE launch has TL_POINTS)
    stamp(c["tl_on"] and c.get("tl_k2", False), c["tls"], tid, 30)  # TL_FOLD



# ---------------------------------------------------------------- shared down
@flyc.jit
def stage_h(c, t):
    """Token t's shared h = SiTU(g, u) (vLLM's situ_and_mul on fp32 g, u) of SGU, once
    -> H; a thread a column pair, its polls in one batch."""
    tid = c["tid"]
    n = SI // 2
    if tid < n:
        got = c["poll"](
            [
                (c["sgu"], t * 2 * SI + 2 * tid, 2),
                (c["sgu"], t * 2 * SI + SI + 2 * tid, 2),
            ]
        )
        g0, g1 = f32_of(got[0][0]), f32_of(got[0][1])
        u0, u1 = f32_of(got[1][0]), f32_of(got[1][1])
        c["put_bf"](c["h"], t * SI + 2 * tid, [situ_vllm(g0, u0), situ_vllm(g1, u1)])
    gpu.barrier()


@flyc.jit
def load_h(c):
    """H -> LDS rows (for the shared down GEMV), once a CTA."""
    s, tid = c["S"], c["tid"]
    hrow = SI + gemv.LDS_PAD
    n = s * SI // 2
    per = (n + THREADS - 1) // THREADS
    got = c["poll"]([(c["h"], fx.min(tid + THREADS * i, n - 1), 1) for i in range(per)])
    for i in range_constexpr(per):
        pidx = tid + THREADS * i
        t = fx.min(pidx, n - 1) // (SI // 2)
        cp = fx.min(pidx, n - 1) % (SI // 2)
        if pidx < n:
            fx.ptr_store(got[i][0], c["xl"] + (t * hrow + 2 * cp) // 2)
    gpu.barrier()


@flyc.jit
def stage_sdown(c, j, ops, flag=True):
    """Shared down rows 16 j .. (K 768) against h in LDS (``load_h``): bf16
    partial -> SHP (this rank's up columns) or every rank's FINAL. ``flag``
    False: a task never in this rank's up columns (no SHRDY flag, its stores
    left in flight)."""
    sent_final = c.get("sent_final", False)
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    hrow = SI + gemv.LDS_PAD
    lo = c["rank"] * UP_TASKS
    mine = (j >= lo) & (j < lo + UP_TASKS)
    tiles = gemv.tiles(s)
    for ti in range_constexpr(len(tiles)):
        t0, sj = tiles[ti]
        acc = gemv.mfmas(lane, wave, c["xl"], hrow, SI, sj, ops, t0)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        lds_barrier()
        if tid < (ROWS // 2) * sj:
            rp = tid % (ROWS // 2)
            tl = tid // (ROWS // 2)
            tt = t0 + tl
            v0 = bf16_round(row_sum(red, 2 * rp, tl))
            v1 = bf16_round(row_sum(red, 2 * rp + 1, tl))
            if mine:
                w = (
                    fx.Vector.from_elements([v0, v1], fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0]
                )
                bo.buffer_store(
                    w,
                    rsrc(c["shp"]),
                    (tt * UP_N + (j - lo) * ROWS + 2 * rp) // 2,
                    cache_modifier=CM_DEV,
                )
            else:
                put_final(c, j * ROWS + 2 * rp, tt, v0, v1, sent_final)
        if const_expr(ti + 1 < len(tiles)):
            lds_barrier()
    if const_expr(flag):
        publish(
            c["put"],
            c["shrdy"],
            fx.max(fx.min(j - lo, UP_TASKS - 1), 0),
            fx.Int32(1),
            (tid == 0) & mine,
        )
    lds_barrier()


@flyc.jit
def run_sdown(c, bid, w_sd):
    """This rank's 56 shared-down tasks whose columns the up stage adds to (k ->
    row group (k + 56 rank) % 448), on the CTAs past the down tasks: up waits
    on them."""
    load_h(c)
    lo = c["rank"] * UP_TASKS
    if bid >= DOWN_TASKS:
        for k in range(bid - DOWN_TASKS, UP_TASKS, BLOCKS - DOWN_TASKS):
            j = (k + lo) % SD_TASKS
            stage_sdown(c, j, gemv.loads(c["lane"], c["wave"], w_sd, j * ROWS, SI))


@flyc.jit
def run_sdown_first(c, bid, w_sd):
    """``run_sdown``'s tasks (up waits on their SHRDY) on the CTAs past the down
    tasks, before their ``run_sdown_rest`` tasks (only the final reduce waits on
    those), both tasks' weights in flight before H is polled. The same tasks and
    arithmetic as ``run_sdown``: placement only."""
    lo = c["rank"] * UP_TASKS
    nsd = BLOCKS - DOWN_TASKS  # 32
    assert UP_TASKS <= 2 * nsd
    if bid >= DOWN_TASKS:
        k0 = bid - DOWN_TASKS
        k1 = fx.min(k0 + nsd, UP_TASKS - 1)
        j0 = (k0 + lo) % SD_TASKS
        j1 = (k1 + lo) % SD_TASKS
        ops0 = gemv.loads(c["lane"], c["wave"], w_sd, j0 * ROWS, SI)
        ops1 = gemv.loads(c["lane"], c["wave"], w_sd, j1 * ROWS, SI)
        load_h(c)
        stage_sdown(c, j0, ops0)
        if k0 + nsd < UP_TASKS:
            stage_sdown(c, j1, ops1)


# a wide step's shared-down CTAs, the latent norm's and up's excluded: the
# CTAs without a fold first (they are free first), then the others
SD_LATE = tuple(range(DOWN_TASKS, BLOCKS)) + tuple(range(LN0 + LN_TASKS, DOWN_TASKS))


@flyc.jit
def run_sdown_late(c, bid, w_sd):
    """A wide step's shared down after the fold: tasks k (row group
    (k + 56 rank) % 448; up's 56 are k < 56) round robin over SD_LATE, so up's
    land first on the CTAs that finish first; every task's weights in flight
    before H is polled. The same tasks and arithmetic as ``run_sdown``:
    placement only."""
    n = len(SD_LATE)
    lo = c["rank"] * UP_TASKS
    # this CTA's place in SD_LATE (the CTAs past the down tasks come first)
    pos = (bid >= DOWN_TASKS).select(bid - DOWN_TASKS, bid - (LN0 + LN_TASKS) + (BLOCKS - DOWN_TASKS))
    if bid >= LN0 + LN_TASKS:
        rounds = (SD_TASKS + n - 1) // n
        ks = [pos + n * r for r in range(rounds)]
        ops = [
            gemv.loads(c["lane"], c["wave"], w_sd, ((fx.min(k, SD_TASKS - 1) + lo) % SD_TASKS) * ROWS, SI)
            for k in ks
        ]
        load_h(c)
        for r in range_constexpr(rounds):
            if ks[r] < SD_TASKS:
                stage_sdown(c, (ks[r] + lo) % SD_TASKS, ops[r])


@flyc.jit
def run_sdown_rest(c, bid, w_sd):
    """The other 392 shared-down tasks (only the final reduce waits on them), on
    the CTAs past lnorm / up's (84 ..), after their down: off the latent-norm ->
    up critical path."""
    r0 = LN0 + LN_TASKS  # 84: past the lnorm / up CTAs
    if bid >= r0:
        load_h(c)
        lo = c["rank"] * UP_TASKS
        for k in range(UP_TASKS + bid - r0, SD_TASKS, BLOCKS - r0):
            j = (k + lo) % SD_TASKS
            stage_sdown(c, j, gemv.loads(c["lane"], c["wave"], w_sd, j * ROWS, SI))


# ---------------------------------------------------------------- lnorm / up / reduce
@flyc.jit
def stage_lnorm(c, j):
    """Columns 128 j ..: the ranks' LATR (rank order, fp32 -> bf16), every
    task's sum of squares -> rstd -> bf16(acc rstd w) -> LN, LNRDY. A thread a
    (token, column pair), 8 tokens a round."""
    sent_latr = c.get("sent_latr", False)
    s, tid, lane, red = c["S"], c["tid"], c["lane"], c["red"]
    a = c["args"]
    cp = tid % 64
    col = j * 128 + 2 * cp
    rounds = (s + 7) // 8
    if const_expr(sent_latr):
        # every round's words in one poll; a dead thread holds a clamped copy
        # of a live one's words: it must not take them; re-armed once the LN
        # rows are out
        tts = [g * 8 + tid // 64 for g in range(rounds)]
        sidxs = [
            [sent_ix(src, col, fx.min(tt, s - 1), s, LAT) for src in range(TP)]
            for tt in tts
        ]
        wall = poll_sent(
            c["latrs_own"],
            [i for sx in sidxs for i in sx],
            [tt < s for tt in tts for _ in range(TP)],
            rearm=False,
        )
    items = []
    for g in range_constexpr(rounds):
        tt = g * 8 + tid // 64
        mine = tt < s
        t = fx.min(tt, s - 1)
        sidx = None
        if const_expr(sent_latr):
            sidx = sidxs[g]
            ws = [wall[g * TP + src] for src in range(TP)]
            lo, hi = bf2_f32(ws[0])
            for src in range_constexpr(1, TP):
                l2, h2 = bf2_f32(ws[src])
                lo, hi = lo + l2, hi + h2
        else:
            lo, hi = sum_partials(
                c["poll"],
                c["peer"]["latr_own"],
                lambda src: ((src * s + t) * LAT + col) // 2,
                TP,
            )
        lo, hi = bf16_round(lo), bf16_round(hi)
        sq = wave_sum(lo * lo + hi * hi)
        if mine & (lane == 0):
            c["put"](c["lmsq"], j * s + t, sq)
        items.append((mine, t, lo, hi, sidx))
    stamp(c["tl_on"] and c.get("tl_k2", False), c["tls"], tid, 31)  # TL_LN_IN
    gpu.barrier()
    for g in range_constexpr(len(items)):
        mine, t = items[g][0], items[g][1]
        if mine & (lane < LN_TASKS):
            v = c["poll"]([(c["lmsq"], lane * s + t, 1)])[0][0]
            fx.ptr_store(f32_of(v), red + (t * 32 + lane))
    gpu.barrier()
    stamp(c["tl_on"] and c.get("tl_k2", False), c["tls"], tid, 32)  # TL_LN_SQ
    w0 = fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(a["ln_w"]), col, vec_width=1, dtype=T.bf16))
    )
    w1 = fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(a["ln_w"]), col + 1, vec_width=1, dtype=T.bf16))
    )
    for g in range_constexpr(len(items)):
        mine, t, lo, hi, _ = items[g]
        tot = fx.ptr_load(red + t * 32)
        for q in range_constexpr(1, LN_TASKS):
            tot = tot + fx.ptr_load(red + (t * 32 + q))
        r = hw_rsq(tot * fx.Float32(1.0 / LAT) + fx.Float32(c["ln_eps"]))
        y = (
            fx.Vector.from_elements([lo * r * w0, hi * r * w1], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        if mine:
            bo.buffer_store(
                y, rsrc(c["ln"]), (t * LAT + col) // 2, cache_modifier=CM_DEV
            )
    publish(c["put"], c["lnrdy"], j, fx.Int32(1), tid == 0)
    if const_expr(sent_latr):
        for g in range_constexpr(len(items)):
            rearm_sent(c["latrs_own"], items[g][4], [items[g][0]] * TP)
    gpu.barrier()


@flyc.jit
def stage_up(c, j):
    """This rank's up rows 16 j .. of its shard (hidden rows 896 rank ..): LN GEMV
    + SHP -> bf16 -> FINAL."""
    sent_final = c.get("sent_final", False)
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    row0 = c["rank"] * UP_N + j * ROWS  # the hidden column; w_up is the shard
    ops = gemv.loads(lane, wave, a["w_up"], j * ROWS, LAT)
    if const_expr(c.get("batch", False)):
        # thread 0's two flags in one round trip
        if tid == 0:
            c["poll"]([(c["lnrdy"], 0, 1), (c["shrdy"], j, 1)])
        if (tid > 0) & (tid < LN_TASKS):
            c["poll"]([(c["lnrdy"], tid, 1)])
    else:
        if tid < LN_TASKS:
            c["poll"]([(c["lnrdy"], tid, 1)])
        if tid == 0:
            c["poll"]([(c["shrdy"], j, 1)])
    gpu.barrier()
    lrow = LAT + gemv.LDS_PAD
    tiles = gemv.tiles(s)
    for ti in range_constexpr(len(tiles)):
        t0, sj = tiles[ti]
        gemv.dev_rows_to_lds(tid, c["ln"], LAT, sj, c["xl"], lrow, t0)
        acc = gemv.mfmas(lane, wave, c["xl"], lrow, LAT, sj, ops)
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < (ROWS // 2) * sj:
            rp = tid % (ROWS // 2)
            tl = tid // (ROWS // 2)
            tt = t0 + tl
            w = fx.Int32(
                bo.buffer_load(
                    rsrc(c["shp"]),
                    (tt * UP_N + j * ROWS + 2 * rp) // 2,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            v0 = bf16_round(bf_lo(w) + row_sum(red, 2 * rp, tl))
            v1 = bf16_round(bf_hi(w) + row_sum(red, 2 * rp + 1, tl))
            put_final(c, row0 + 2 * rp, tt, v0, v1, sent_final)
        gpu.barrier()


def put_final(c, col, t, v0, v1, sent_final):
    """This rank's bf16 pair of FINAL (col, t) to every rank: a tagged pair, or
    (``sent_final``) a sentinel word."""
    if const_expr(sent_final):
        si = sent_ix(c["rank"], col, t, c["S"], HIDDEN)
        word = sent_word(bf16_pair(v0, v1))
        for p in range_constexpr(TP):
            put_sent(c["finals_peer"][p], si, word)
    else:
        at = (c["rank"] * c["S"] + t) * HIDDEN + col
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["final"][p], at, [v0, v1])


@flyc.jit
def stage_reduce(c, j):
    """Columns 128 j ..: FINAL in rank order -> bf16 out (+ the updated prefix
    under ``fold``)."""
    sent_final = c.get("sent_final", False)
    s, tid = c["S"], c["tid"]
    # read before the traced branch: a dict lookup inside it becomes carried state
    fold = c.get("fold", False)
    # the MoE launch has no prefix: only the layer launch's fold reads it
    prefix = c["args"]["prefix"] if fold else None
    # K2's tail (``seam.tail_slice``): the stream also into LDS, and to the row
    # CTAs as tagged pairs
    tpre = c.get("tpre")
    n = s * 64
    rounds = (n + THREADS - 1) // THREADS
    if const_expr(sent_final):
        # every round's words in one poll; consecutive threads read
        # consecutive words
        its = [sent_item(fx.min(tid + THREADS * i, n - 1), s) for i in range(rounds)]
        sums = sent_sums(
            c["finals_own"],
            [(j * 128 + 2 * cp, t) for t, cp in its],
            s,
            HIDDEN,
            TP,
            [tid + THREADS * i < n for i in range(rounds)],
        )
    for i in range_constexpr(rounds):
        idx = tid + THREADS * i
        if idx < n:
            if const_expr(sent_final):
                t, cp = its[i]
            else:
                t, cp = idx // 64, idx % 64
            col = j * 128 + 2 * cp
            if const_expr(sent_final):
                lo, hi = sums[i]
            else:
                lo, hi = sum_partials(
                    c["poll"],
                    c["peer"]["final_own"],
                    lambda src: ((src * s + t) * HIDDEN + col) // 2,
                    TP,
                )
            o_lo, o_hi = lo, hi
            if const_expr(fold):
                # out + the updated prefix, as torch's bf16 add: each operand
                # rounded to bf16, the sum in fp32, rounded once more. The seam
                # stored the prefix device-coherent, causally before FINAL.
                pw = ld_i32(prefix, (t * HIDDEN + col) // 2, CM_DEV)
                o_lo = bf16_round(bf16_round(lo) + bf_lo(pw))
                o_hi = bf16_round(bf16_round(hi) + bf_hi(pw))
            w = (
                fx.Vector.from_elements([o_lo, o_hi], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0]
            )
            bo.buffer_store(w, rsrc(c["args"]["out"]), (t * HIDDEN + col) // 2)
            if const_expr(tpre is not None):
                fx.ptr_store(w, tpre + (t * 64 + cp))
                c["put_words"](c["tstream"], (t * HIDDEN + col) // 2, [w])
            if const_expr(sent_final):
                rearm_sent(
                    c["finals_own"],
                    [sent_ix(src, col, t, s, HIDDEN) for src in range(TP)],
                )
    gpu.barrier()


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: MoeBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    assert not wide(s)  # the standalone MoE launch is narrow only
    lay = scratch_layout(key)
    play = peer_layout(s)
    half = peer_bytes(s) // 2

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * TOPK, 16]
        flag: fx.Array[fx.Int32, E, 16]
        expert: fx.Array[fx.Int32, max_u(s), 16]
        kof: fx.Array[fx.Int32, max_u(s) * s, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class EarlyLds:
        xl: fx.Array[fx.Int32, xl_words(s), 16]

    @fx.struct
    class UgLds:
        latq: fx.Array[fx.Int32, s * LAT // 4, 16]
        latsc: fx.Array[fx.Int32, s * LAT // 32, 16]
        av: fx.Array[fx.Float32, THREADS, 16]

    @fx.struct
    class DownLds:
        interq: fx.Array[fx.Int32, s * TOPK * IW, 16]
        intsc: fx.Array[fx.Int32, s * TOPK * UG_GROUPS, 16]
        contrib: fx.Array[fx.Float32, s * TOPK * ROWS, 16]

    @fx.union
    class StageLds:
        early: EarlyLds
        ug: UgLds
        down: DownLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]
        qslot: fx.Array[fx.Int32, 4, 16]

    keyed = key_tuple(key, SOURCES)
    name = f"k3_mono_moe_s{s}_r{int(key.reduce)}_t{int(key.timeline)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def moe(
        x: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        queue: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        st_lds = alloc.allocate(StageLds)
        el, ul, dl = st_lds.early.peek(), st_lds.ug.peek(), st_lds.down.peek()
        mb = Mailbox(step_tag(epoch, layer))
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, TP)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in ("rlog", "lat", "latq", "latr", "final"):
            off = play[region][0]
            peer[region] = [
                shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
            ]
            peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
        c = {
            "S": s,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "ln_eps": key.eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "peer": peer,
            "xl": el.xl.ptr,
            "latq": ul.latq.ptr,
            "latsc": ul.latsc.ptr,
            "av": ul.av.ptr,
            "interq": dl.interq.ptr,
            "intsc": dl.intsc.ptr,
            "contrib": dl.contrib.ptr,
            "scan": rl.scan.ptr,
            "rt": {
                "route": rl.route.ptr,
                "flag": rl.flag.ptr,
                "expert": rl.expert.ptr,
                "kof": rl.kof.ptr,
            },
            "ln": scratch + fx.Int64(lay["ln"][0]),
            "shp": scratch + fx.Int64(lay["shp"][0]),
            "queue": queue,
            "qslot": lds.qslot.ptr,
            "slot": layer,
            "par": par,
            "xp": key.xp,
            "guint": key.guint,
            "lazy_inter": True,
            # the shared stages' timeline points (the route's, the router's)
            "tl_on": key.timeline,
            "tls": lds.tls.ptr,
            "args": {
                "x": x,
                "w_gate": w_gate,
                "bias": bias,
                "w_ld": w_ld,
                "w_sgu": w_sgu,
                "w_sd": w_sd,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "ln_w": ln_w,
                "w_up": w_up,
                "out": out,
            },
        }
        for region in (
            "sgu",
            "h",
            "rpart",
            "lpart",
            "route",
            "inter",
            "isc",
            "lmsq",
            "lnrdy",
            "shrdy",
        ):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        stamp_begin(on, tls, tid, TL_POINTS)
        if const_expr(key.stop > 1):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if bid < EARLY:
                stage_early(c, bid)
        stamp(on, tls, tid, 1)
        if const_expr(key.stop <= 2):
            stamp_flush(on, tls, tl, tid, bid, TL_POINTS)
            return
        if (bid >= LATE0) & (bid < LATE0 + s):
            stage_route(c, bid - LATE0)
        if (bid >= H_CTA) & (bid < H_CTA + s):
            stage_h(c, bid - H_CTA)
        stamp(on, tls, tid, 2)
        if const_expr(key.stop <= 3):
            return
        nu = load_route(c)
        if const_expr(key.stop == 35):
            # debug: the route table, |U| and U into ``out`` (as int32), CTA 0
            if bid == 0:
                n = s * 2 * TOPK
                if tid < n:
                    bo.buffer_store(fx.ptr_load(c["rt"]["route"] + tid), rsrc(out), tid)
                if tid < max_u(s):
                    bo.buffer_store(
                        fx.ptr_load(c["rt"]["expert"] + tid), rsrc(out), n + 1 + tid
                    )
                if tid == 0:
                    bo.buffer_store(nu, rsrc(out), n)
            return
        load_latq(c)
        if const_expr(key.stop == 36):
            return
        # static round-robin: deterministic, no state across launches (a
        # counter carried between steps went stale when a step skipped K2b)
        total = nu * UG_GROUPS
        last = total - 1
        # reversed placement: the round's leftover tasks go to the high CTAs,
        # whose early work ends first (the AttnRes / route CTAs start ug last)
        task = BLOCKS - 1 - bid
        ops = ug_loads(c, fx.min(task, last))
        while task < total:
            nxt = task + BLOCKS
            # the next task's weights in flight under this one's MFMAs and
            # epilogue (past the end: the last task's, cache hits)
            ops_n = ug_loads(c, fx.min(nxt, last))
            stage_ug(c, task, ops)
            task = nxt
            ops = ops_n
        stamp(on, tls, tid, 3)
        if const_expr(key.stop <= 4):
            stamp_flush(on, tls, tl, tid, bid, TL_POINTS)
            return
        if bid < DOWN_TASKS:
            # the first w2 batch in flight before INTER (every ug) is waited for
            dops = down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
            load_inter(c)
            stage_down(c, bid, nu, dops)
        run_sdown_rest(c, bid, w_sd)
        # the shared down (needed only by up) on the CTAs with no down task:
        # after their ug, so every CTA's ug starts as the route lands
        run_sdown(c, bid, w_sd)
        stamp(on, tls, tid, 4)
        if const_expr(key.stop <= 5):
            return
        if (bid >= LN0) & (bid < LN0 + LN_TASKS):
            stage_lnorm(c, bid - LN0)
        stamp(on, tls, tid, 5)
        if const_expr(key.stop <= 6):
            return
        if bid < UP_TASKS:
            stage_up(c, bid)
        stamp(on, tls, tid, 6)
        if const_expr(key.reduce):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if (bid >= RED0) & (bid < RED0 + RED_TASKS):
                stage_reduce(c, bid - RED0)
        stamp(on, tls, tid, 7)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        x: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        queue: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        moe(
            x,
            w_gate,
            bias,
            w_ld,
            w_sgu,
            w_sd,
            w13,
            w13s,
            w2,
            w2s,
            ln_w,
            w_up,
            out,
            scratch,
            queue,
            peers,
            rank,
            epoch,
            layer,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch


def moe(
    key: MoeBuild,
    *,
    x,
    w_gate,
    bias,
    w_ld,
    w_sgu,
    w_sd,
    w13,
    w13s,
    w2,
    w2s,
    ln_w,
    w_up,
    out,
    scratch,
    queue,
    peers,
    rank,
    epoch,
    layer,
    tl=None,
):
    """Host launcher; every weight as the model holds it after loading, the
    experts' shuffled. ``queue``: QUEUE_BYTES of zeros, this launch's
    alone (no other kernel writes it)."""
    assert queue.numel() * queue.element_size() >= QUEUE_BYTES
    assert w_gate.shape == (E, HIDDEN) and w_ld.shape == (LAT, HIDDEN)
    assert w_sgu.shape == (2 * SI, HIDDEN) and w_sd.shape == (HIDDEN, SI)
    assert w_up.shape == (UP_N, LAT) and w_up.is_contiguous()
    f = build(key)
    f(
        x.data_ptr(),
        w_gate.data_ptr(),
        bias.data_ptr(),
        w_ld.data_ptr(),
        w_sgu.data_ptr(),
        w_sd.data_ptr(),
        w13.data_ptr(),
        w13s.data_ptr(),
        w2.data_ptr(),
        w2s.data_ptr(),
        ln_w.data_ptr(),
        w_up.data_ptr(),
        out.data_ptr(),
        scratch.data_ptr(),
        queue.data_ptr(),
        peers.data_ptr(),
        rank,
        epoch.data_ptr(),
        layer,
        0 if tl is None else tl.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
