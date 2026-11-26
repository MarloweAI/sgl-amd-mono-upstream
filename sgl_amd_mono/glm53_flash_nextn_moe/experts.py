# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's NEXTN draft MoE experts at TP2 (gfx950) in one FlyDSL launch, on a sort of the routes in blocks of
BM = 32 or 64 rows (aiter's ``moe_sorting`` layout: ``sorted_ids`` (slot << 24) | token, pad rows token M;
``sorted_expert_ids`` a block; ``num_valid_ids[0]`` the sorted rows in use): the gate+up GEMM with the clamped SiLU
and the FP8 intermediate (G1), then the down GEMM with the route weight and the bf16 atomic combine into the zeroed
output (G2). Expert 288 is the shared expert (its own weights, in the same layout), routed with weight 1.

Numerics (the checkpoint's FP8 block-scale experts): weights E4M3 with an f32 scale per 128 x 128 block,
activations E4M3 with an f32 scale per token and 128 columns (amax x f32(1 / 448)). Each 128-deep K block is one 16x16x128
E4M3 MFMA accumulating in place in f32: the accumulator is held in units of the current block's scale product
(activation scale x weight scale) and rescaled by old / new at each block, then by the last scale. G1's epilogue:
gate <= limit, |up| <= limit, gate * sigmoid(gate) * up in f32, then the intermediate's E4M3 with an f32 scale per
token and 128 columns (amax x f32(1 / 448), 1 where amax is 0; v / scale), as aiter quantizes the activations.

Workgroups (4 waves a 32-row half of the block: 256 threads at BM = 32, 512 at 64), in id order:

- G1 (block b, tile n) for n < 8: the block's BM sorted rows x intermediate columns 128 n .. 128 n + 127 (w13 gate
  rows 128 n .., up rows 1024 + 128 n ..) over K = 4096: one 128 x 128 scale block a K step. Wave w takes 32
  columns. The tile's 128 columns are one quant group of the intermediate: the 4 waves' amax meet in LDS. It stores
  the E4M3 rows [cap, 1024 B] and their f32 scales [cap, 8] written through (sc0 sc1), waits for the stores and adds
  1 to block b's counter.
- G2 (block b, slab q) for q < 16: output columns 256 q .. 256 q + 255 (two 128-column tiles in turn) over K = 1024.
  Its first w2 loads go out, then thread 0 polls until block b's counter counts the block's 8 G1 workgroups (each
  has a lower id and never waits); the workgroup reads the block's intermediate once into LDS (sc1 loads) and adds
  weight x sum as bf16 pairs into the output (buffer atomics; rows whose token is past M fall outside the output's
  buffer). Its departure counts on the same counter; the block's last zeroes it.
- Blocks from num_valid_ids[0] / BM on exit at once and touch no counter.

Weights (aiter's (16, 16) shuffle) are read nontemporal. Every counter is 0 between launches."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float32, Int32, Int64, T

from sgl_amd_mono.common.mx import FP8, UNIT_SCALE, mfma_scaled
from sgl_amd_mono.common.ops import CM_DEV, CM_NT, CM_SYS, global_ptr, hw_exp2, hw_rcp, ld_i32, rsrc, uniform, xred
from sgl_amd_mono.common.plan import kernel_sources

from . import D, E, FP8_MAX, I, MAX_ROWS, TOPK

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)

WAVE_THREADS = 256  # a row half's 4 waves
SHARED = E  # the shared expert's id in the sort
G1_N = 128  # intermediate columns of a G1 workgroup (one quant group)
G1_WGS = I // G1_N
G1_STEPS = D // 128
G2_N = 128  # output columns of a G2 tile
G2_SPAN = 2  # G2 tiles of a G2 workgroup
G2_WGS = D // (G2_N * G2_SPAN)
G2_STEPS = I // 128
LOG2E = 1.4426950408889634
# an expert's E4M3 weights and f32 block scales
W13_BYTES = 2 * I * D
W2_BYTES = D * I
S13_WORDS = (2 * I // 128) * (D // 128)
S2_WORDS = (D // 128) * (I // 128)
# the intermediate: E4M3 rows, then their f32 scales, for the largest sort (aiter's: routes + experts x BM)
MAX_BM = 64
MAX_CAP = MAX_ROWS * (TOPK + 1) + (E + 1) * MAX_BM
MID_SCALE_AT = MAX_CAP * I
MID_BYTES = MAX_CAP * (I + (I // 128) * 4)
XL_ROW = I // 4 + 4  # LDS dwords of a staged intermediate row (padded)
LINE = 128
COUNTER_WORDS = MAX_CAP // 32 * LINE // 4


def _ld_agent(addr):
    return fx.generic_load(
        global_ptr(addr), dtype=fx.Int32, memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent"
    )


def _st_agent(addr, v):
    fx.generic_store(global_ptr(addr), fx.Int32(v), memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent")


def _add_agent(addr, v=1):
    return fx.Int32(fx.llvm.atomic_add(global_ptr(addr), fx.Int32(v), syncscope="agent"))


def _load(r, i, cm=0):
    return fx.Vector(bo.buffer_load(r, i, vec_width=4, dtype=T.i32, cache_modifier=cm))


def _load_f32(r, i, cm=0):
    return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32, cache_modifier=cm))


def _cat8(a, b):
    """Two 16 B operand halves -> the 32 B E4M3 operand of a 16x16x128 MFMA."""
    return fx.Vector.from_elements([a[i] for i in range(4)] + [b[i] for i in range(4)], fx.Int32)


def _mfma(w, x, acc):
    """One 16x16x128 E4M3 MFMA accumulating into ``acc`` (w: 16 rows of output columns, x: 16 tokens)."""
    one = fx.Int32(UNIT_SCALE)
    return mfma_scaled(w, x, acc, one, one, FP8, FP8, 0, 0)


def _scaled(v, s):
    return v * fx.Vector.filled(4, s, fx.Float32)


def _block_scale(s, prev, accs):
    """Moves ``accs`` (tiles in units of the scale ``prev``, or None at the first block) to the units of this
    block's scale ``s`` (at least 1e-30: pad rows read 0)."""
    s = fx.max(s, fx.Float32(1e-30))
    if const_expr(prev is not None):
        r = prev * hw_rcp(s)
        for t in range_constexpr(len(accs)):
            accs[t] = _scaled(accs[t], r)
    return s


def _store_done():
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()


def _silu(gate, up, neg_limit):
    """The clamped SiLU: gate <= limit, |up| <= limit, gate * sigmoid(gate) * up (exp2 / rcp)."""
    g = -fx.max(-gate, neg_limit)
    u = fx.max(-fx.max(-up, neg_limit), neg_limit)
    s = hw_rcp(fx.Float32(1.0) + hw_exp2(g * fx.Float32(-LOG2E)))
    return g * s * u


def _fp8x4(v, scale):
    """Four f32 over ``scale`` -> four E4M3 bytes (RNE) in a dword."""
    q = rocdl.cvt_pk_fp8_f32(T.i32, v[0] / scale, v[1] / scale, fx.Int32(0), False)
    q = rocdl.cvt_pk_fp8_f32(T.i32, v[2] / scale, v[3] / scale, fx.Int32(q), True)
    return fx.Int32(q)


def _w_load(c, rg, kc, k):
    """This lane's 32 B of row group rg (16 rows), K step kc of a (16, 16)-shuffled E4M3 [N, k] matrix: K bytes
    32 g .. 32 g + 31 of row lane % 16 (g = lane // 16), as two 16 B halves 256 B apart."""
    lane = c["lane"]
    base = rg * (4 * k) + kc * 512 + (lane // 16) * 128 + (lane % 16) * 4
    return _cat8(_load(c["rw"], base, cm=CM_NT), _load(c["rw"], base + 64, cm=CM_NT))


def _srcs(c, w, s, w_sh, s_sh, e, w_bytes, s_words):
    """A task's weight and scale sources: the routed expert's, or the shared expert's."""
    if const_expr(c["sh"]):
        return {"rw": rsrc(w_sh), "rs": rsrc(s_sh)}
    return {"rw": rsrc(w + e * w_bytes), "rs": rsrc(s + e * (s_words * 4))}


# ---- G1 ----


def _g1_loads(c, step):
    """One wave's operands of G1 K step ``step``: gate / up weights (j = 0, 1), the rows' activations, their scales
    and the step's gate / up weight scales (wave-uniform)."""
    lane = c["lane"]
    w = [_w_load(c, c["rg"] + u * (I // 16) + j, step, D) for u in range(2) for j in range(2)]
    x = [
        _cat8(_load(c["ra"], t * (D // 4) + step * 32 + (lane // 16) * 8),
              _load(c["ra"], t * (D // 4) + step * 32 + (lane // 16) * 8 + 4))
        for t in c["toks"]
    ]  # fmt: skip
    xs = [_load_f32(c["ras"], t * (D // 128) + step) for t in c["toks"]]
    ws = [_load_f32(c["rs"], (c["nt"] + u * (I // 128)) * (D // 128) + step) for u in range(2)]
    return w, x, xs, ws


def _g1_epilogue(c, blk, acc):
    """Rows of the block x this wave's 32 columns (acc [gate / up][j][row tile][4]): the clamped SiLU, then E4M3 of
    the tile's 128-column group (amax over the 4 waves in LDS), written through with its f32 scale."""
    lane, w, lds, rts, bm = c["lane"], c["wave"], c["lds"], c["rts"], c["bm"]
    neg = -c["limit"]
    rm = rsrc(c["args"]["mid"], MID_BYTES)
    vals = []
    for rt in range_constexpr(rts):
        v = [[_silu(acc[0][j][rt][i], acc[1][j][rt][i], neg) for i in range(4)] for j in range(2)]
        a = fx.Float32(0.0)
        for j in range_constexpr(2):
            for i in range_constexpr(4):
                a = fx.max(a, fx.max(v[j][i], -v[j][i]))
        a = xred(xred(a, 16, fx.max), 32, fx.max)
        fx.ptr_store(a.bitcast(fx.Int32), lds + (((c["half"] * 4 + w) * rts + rt) * 16 + lane % 16))
        vals.append(v)
    gpu.barrier()
    for rt in range_constexpr(rts):
        row = blk * bm + c["r0"] + rt * 16 + lane % 16
        a = fx.Float32(0.0)
        for ww in range_constexpr(4):
            a = fx.max(a, fx.ptr_load(lds + (((c["half"] * 4 + ww) * rts + rt) * 16 + lane % 16)).bitcast(fx.Float32))
        scale = (a == fx.Float32(0.0)).select(fx.Float32(1.0), a * fx.Float32(1.0 / FP8_MAX))
        for j in range_constexpr(2):
            off = row * I + c["nt"] * G1_N + w * 32 + j * 16 + (lane // 16) * 4
            bo.buffer_store(_fp8x4(vals[rt][j], scale), rm, off, cache_modifier=CM_SYS, offset_is_bytes=True)
        bo.buffer_store(
            scale, rm, MID_SCALE_AT + (row * (I // 128) + c["nt"]) * 4,
            mask=(lane < 16) & (w == 0), cache_modifier=CM_SYS, offset_is_bytes=True,
        )  # fmt: skip


@flyc.jit
def _count_block(c, cnt):
    if c["tid"] == 0:
        _add_agent(cnt)


def _g1_tile(c, blk, nt, e, sh):
    a, lane, w, rts, bm = c["args"], c["lane"], c["wave"], c["rts"], c["bm"]
    k = {**c, "sh": sh}
    k = {
        **k,
        **_srcs(k, a["w13"], a["s13"], a["w13sh"], a["s13sh"], e, W13_BYTES, S13_WORDS),
        "ra": rsrc(a["a1"], fx.Int64(c["m"]) * D),
        "ras": rsrc(a["a1s"], fx.Int64(c["m"]) * (D // 128) * 4),
        "toks": [ld_i32(a["sids"], blk * bm + c["r0"] + rt * 16 + lane % 16) & 0xFFFFFF for rt in range(rts)],
        "rg": nt * (G1_N // 16) + 2 * w,  # the wave's first gate row group
        "nt": nt,
    }
    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    acc = [[[zero] * rts for _ in range(2)] for _ in range(2)]  # [gate / up][j][row tile]
    prev = [[None] * rts for _ in range(2)]  # [gate / up][row tile]: the scale the tiles are held in
    pf = c["pf"][0]
    pend = [_g1_loads(k, s) for s in range(pf)]
    for step in range_constexpr(G1_STEPS):
        wv, x, xs, ws = pend.pop(0)
        if const_expr(step + pf < G1_STEPS):
            pend.append(_g1_loads(k, step + pf))
        for u in range_constexpr(2):
            for rt in range_constexpr(rts):
                tiles = [acc[u][0][rt], acc[u][1][rt]]
                prev[u][rt] = _block_scale(xs[rt] * ws[u], prev[u][rt], tiles)
                for j in range_constexpr(2):
                    acc[u][j][rt] = _mfma(wv[u * 2 + j], x[rt], tiles[j])
        rocdl.sched_barrier(0)
    for u in range_constexpr(2):
        for j in range_constexpr(2):
            for rt in range_constexpr(rts):
                acc[u][j][rt] = _scaled(acc[u][j][rt], prev[u][rt])
    _g1_epilogue(k, blk, [[[[v[i] for i in range(4)] for v in a2] for a2 in a1] for a1 in acc])
    _store_done()
    _count_block(c, a["ws"] + fx.Int64(blk * LINE))


@flyc.jit
def gemm1(c, blk, nt):
    if blk * c["bm"] < c["nv"]:
        ei = uniform(ld_i32(c["args"]["seids"], blk))
        if ei == SHARED:
            _g1_tile(c, blk, nt, fx.Int64(ei), True)
        else:
            _g1_tile(c, blk, nt, fx.Int64(ei), False)


# ---- G2 ----


@flyc.jit
def _await_block(c, cnt):
    """Thread 0 polls the block's counter until its G1 tiles are in, then counts this workgroup's departure (the
    block's last zeroes it); the workgroup then meets."""
    if c["tid"] == 0:
        v = _ld_agent(cnt)
        while v < G1_WGS:
            rocdl.s_sleep(1)
            v = _ld_agent(cnt)
        if _add_agent(cnt) == G1_WGS + G2_WGS - 1:
            _st_agent(cnt, 0)
    gpu.barrier()


def _g2_loads(c, step):
    """One wave's w2 operands of G2 step ``step`` (tile step // 8, K step step % 8): rows j = 0, 1 and the step's
    weight scale (wave-uniform)."""
    n, kc = step // G2_STEPS, step % G2_STEPS
    rg = c["rg"] + n * (G2_N // 16)
    w = [_w_load(c, rg + j, kc, I) for j in range(2)]
    s = _load_f32(c["rs"], (c["q"] * G2_SPAN + n) * (I // 128) + kc)
    return w, s


def _pieces(c):
    """16 B staging pieces a thread: a block's intermediate rows of 1 KB."""
    return c["bm"] * I // 16 // c["threads"]


def _live_rows(c, blk):
    """Whether each of this thread's staging pieces' row holds a route (pad rows: token M)."""
    a, tid, bm = c["args"], c["tid"], c["bm"]
    return [(ld_i32(a["sids"], blk * bm + (tid + c["threads"] * i) // (I // 16)) & 0xFFFFFF) < c["m"]
            for i in range(_pieces(c))]  # fmt: skip


def _g2_stage(c, blk, live):
    """The block's intermediate rows (1 KB each) and their scales, read once (sc1) into LDS; pad rows are not read
    (their outputs fall outside every buffer)."""
    tid, lds, bm = c["tid"], c["lds"], c["bm"]
    rm = rsrc(c["args"]["mid"], MID_BYTES)
    got = []
    for i in range_constexpr(_pieces(c)):
        u = tid + c["threads"] * i
        got.append(_load(rm, live[i].select(blk * (bm * I // 4) + u * 4, fx.Int32(MID_BYTES)), cm=CM_DEV))
    nsc = bm * (I // 128) // c["threads"]
    sw = [_load_f32(rm, MID_SCALE_AT // 4 + blk * (bm * I // 128) + tid + c["threads"] * i, cm=CM_DEV)
          for i in range(nsc)]  # fmt: skip
    for i in range_constexpr(_pieces(c)):
        u = tid + c["threads"] * i
        fx.ptr_store(got[i], lds + ((u // (I // 16)) * XL_ROW + (u % (I // 16)) * 4))
    for i in range_constexpr(nsc):
        fx.ptr_store(sw[i].bitcast(fx.Int32), lds + (bm * XL_ROW + tid + c["threads"] * i))
    gpu.barrier()


def _g2_x(c, rt, kc):
    """This lane's intermediate operand (row tile rt, K step kc) and its row's scale, from LDS."""
    lane, lds = c["lane"], c["lds"]
    r = c["r0"] + rt * 16 + lane % 16
    v4i = fx.Vector.make_type(4, fx.Int32)
    at = r * XL_ROW + kc * 32 + (lane // 16) * 8
    x = _cat8(fx.Vector(fx.ptr_load(lds + at, result_type=v4i)), fx.Vector(fx.ptr_load(lds + (at + 4), result_type=v4i)))
    return x, fx.ptr_load(lds + (c["bm"] * XL_ROW + r * (I // 128) + kc)).bitcast(fx.Float32)


def _g2_combine(c, acc, toks, tws, ro, n):
    """weight x sum of tile ``n`` as bf16 pairs added into the output."""
    for rt in range_constexpr(c["rts"]):
        for j in range_constexpr(2):
            col = (c["rg"] + n * (G2_N // 16) + j) * 16 + (c["lane"] // 16) * 4
            off = toks[rt] * (D * 2) + col * 2
            for p in range_constexpr(2):
                pair = fx.Vector.from_elements(
                    [acc[j][rt][2 * p] * tws[rt], acc[j][rt][2 * p + 1] * tws[rt]], fx.Float32
                ).to(fx.BFloat16)
                rocdl.raw_ptr_buffer_atomic_fadd(pair, ro, off + 4 * p, 0)


def _g2_tile(c, blk, q, e, sh):
    a, lane, wave, rts, bm = c["args"], c["lane"], c["wave"], c["rts"], c["bm"]
    k = {**c, "sh": sh}
    k = {
        **k,
        **_srcs(k, a["w2"], a["s2"], a["w2sh"], a["s2sh"], e, W2_BYTES, S2_WORDS),
        "rg": q * (G2_SPAN * G2_N // 16) + 2 * wave,  # the wave's first w2 row group (16 output columns)
        "q": q,
    }
    steps = G2_SPAN * G2_STEPS
    pf = c["pf"][1]
    pend = [_g2_loads(k, s) for s in range(pf)]
    rows = [blk * bm + c["r0"] + rt * 16 + lane % 16 for rt in range(rts)]
    toks = [ld_i32(a["sids"], r) & 0xFFFFFF for r in rows]
    tws = [fx.Float32(bo.buffer_load(rsrc(a["sw"]), r, 1, T.f32)) for r in rows]
    live = _live_rows(c, blk)
    _await_block(c, a["ws"] + fx.Int64(blk * LINE))
    _g2_stage(c, blk, live)
    ro = rsrc(a["out"], fx.Int64(c["m"]) * (D * 2))
    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    acc = [[zero] * rts for _ in range(2)]  # [j][row tile]
    prev = [None] * rts
    for step in range_constexpr(steps):
        w, ws = pend.pop(0)
        if const_expr(step + pf < steps):
            pend.append(_g2_loads(k, step + pf))
        kc = step % G2_STEPS
        for rt in range_constexpr(rts):
            xv, xsc = _g2_x(c, rt, kc)
            tiles = [acc[0][rt], acc[1][rt]]
            prev[rt] = _block_scale(xsc * ws, prev[rt], tiles)
            for j in range_constexpr(2):
                acc[j][rt] = _mfma(w[j], xv, tiles[j])
        if const_expr(kc == G2_STEPS - 1):
            for rt in range_constexpr(rts):
                for j in range_constexpr(2):
                    acc[j][rt] = _scaled(acc[j][rt], prev[rt])
            _g2_combine(k, acc, toks, tws, ro, step // G2_STEPS)
            acc = [[zero] * rts for _ in range(2)]
            prev = [None] * rts
        rocdl.sched_barrier(0)


@flyc.jit
def gemm2(c, blk, q):
    if blk * c["bm"] < c["nv"]:
        ei = uniform(ld_i32(c["args"]["seids"], blk))
        if ei == SHARED:
            _g2_tile(c, blk, q, fx.Int64(ei), True)
        else:
            _g2_tile(c, blk, q, fx.Int64(ei), False)


@flyc.jit
def _run(c, bid):
    g1 = c["blocks"] * G1_WGS
    if bid < g1:
        gemm1(c, bid // G1_WGS, bid % G1_WGS)
    else:
        b2 = bid - g1
        gemm2(c, b2 // G2_WGS, b2 % G2_WGS)


@functools.cache
def build(bm: int = 32, pf1: int = 2, pf2: int = 4, wpe: int = 2):
    """The launch on BM = ``bm`` blocks with ``pf1`` / ``pf2`` K steps of loads a wave keeps in flight ahead in G1 /
    G2, at least ``wpe`` waves a SIMD (the register budget)."""
    assert bm in (32, 64)
    threads = WAVE_THREADS * (bm // 32)
    keyed = (("sources", SOURCES), ("bm", bm), ("prefetch", (pf1, pf2)), ("wpe", wpe))
    lds_words = bm * XL_ROW + bm * (I // 128)

    @fx.struct
    class Lds:
        words: fx.Array[fx.Int32, lds_words, 16]

    name = f"glm53f_nextn_moe_experts_m{bm}_p{pf1}{pf2}_w{wpe}"

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kexperts(
        a1: Int64, a1s: Int64, sids: Int64, sw: Int64, seids: Int64, nvalid: Int64,
        w13: Int64, w2: Int64, s13: Int64, s2: Int64, w13sh: Int64, w2sh: Int64, s13sh: Int64, s2sh: Int64,
        out: Int64, mid: Int64, ws: Int64, m: Int32, blocks: Int32, limit: Float32,
    ):  # fmt: skip
        _ = keyed
        tid = fx.Int32(fx.thread_idx.x)
        lds = fx.SharedAllocator().allocate(Lds).peek()
        args = {"a1": a1, "a1s": a1s, "sids": sids, "sw": sw, "seids": seids, "w13": w13, "w2": w2, "s13": s13,
                "s2": s2, "w13sh": w13sh, "w2sh": w2sh, "s13sh": s13sh, "s2sh": s2sh, "out": out, "mid": mid,
                "ws": ws}  # fmt: skip
        c = {
            "args": args, "tid": tid, "lane": tid % 64, "wave": uniform((tid // 64) % 4), "m": m,
            "blocks": blocks, "limit": limit, "nv": uniform(ld_i32(nvalid, 0)), "lds": lds.words.ptr,
            "pf": (pf1, pf2), "bm": bm, "rts": 2, "threads": threads, "half": uniform(tid // WAVE_THREADS),
            "r0": uniform(tid // WAVE_THREADS) * 32,
        }  # fmt: skip
        _run(c, fx.Int32(fx.block_idx.x))

    @flyc.jit
    def launch(
        a1: Int64, a1s: Int64, sids: Int64, sw: Int64, seids: Int64, nvalid: Int64,
        w13: Int64, w2: Int64, s13: Int64, s2: Int64, w13sh: Int64, w2sh: Int64, s13sh: Int64, s2sh: Int64,
        out: Int64, mid: Int64, ws: Int64, m: Int32, blocks: Int32, limit: Float32, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        kexperts(
            a1, a1s, sids, sw, seids, nvalid, w13, w2, s13, s2, w13sh, w2sh, s13sh, s2sh, out, mid, ws, m, blocks,
            limit, value_attrs={"rocdl.waves_per_eu": wpe},
        ).launch(grid=(blocks * (G1_WGS + G2_WGS),), block=(threads,), stream=stream)

    return launch


@functools.cache
def workspace(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(counters, zeroed and left zeroed by every launch; the intermediate). Made once a device, outside graph
    capture."""
    return (
        torch.zeros(COUNTER_WORDS, dtype=torch.int32, device=device),
        torch.empty(MID_BYTES, dtype=torch.uint8, device=device),
    )


def experts(
    a1: torch.Tensor,
    a1_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    moe_buf: torch.Tensor,
    swiglu_limit: float | None,
    bm: int,
    variant: tuple = (2, 4, 2),
) -> torch.Tensor:
    """The experts on a sort in blocks of ``bm``: a1 [M, 4096] E4M3 and its f32 scales [M, 32] (amax / 448 a token
    and 128 columns), the checkpoint's FP8 routed weights as aiter shuffles them (w13 [288, 2048, 4096],
    w2 [288, 4096, 1024] E4M3, (16, 16)) and their f32 block scales (w13 [288, 16, 32], w2 [288, 32, 8]); ``shared``:
    the shared expert's (w13 [2048, 4096], w2 [4096, 1024] shuffled the same way, their scales) for routes to expert
    288; ``moe_buf`` [M, 4096] bf16 zeroed (added into). Returns ``moe_buf``."""
    m = a1.shape[0]
    blocks = sorted_expert_ids.shape[0]
    assert 1 <= m <= MAX_ROWS and blocks * bm <= MAX_CAP and sorted_ids.shape[0] >= blocks * bm
    assert w13.shape[0] == E and w13.numel() * w13.element_size() == E * W13_BYTES
    assert w2.numel() * w2.element_size() == E * W2_BYTES and moe_buf.shape == (m, D)
    assert w13_scale.numel() == E * S13_WORDS and w2_scale.numel() == E * S2_WORDS
    assert w13_scale.dtype == torch.float32 and w2_scale.dtype == torch.float32 and a1_scale.dtype == torch.float32
    assert a1_scale.shape == (m, D // 128) and a1_scale.is_contiguous() and a1.is_contiguous()
    sh = shared if shared is not None else (w13, w2, w13_scale, w2_scale)
    assert sh[0].numel() * sh[0].element_size() >= W13_BYTES and sh[1].numel() * sh[1].element_size() >= W2_BYTES
    assert sh[2].numel() >= S13_WORDS and sh[3].numel() >= S2_WORDS
    counters, mid = workspace(a1.device)
    build(bm, *variant)(
        a1.data_ptr(), a1_scale.data_ptr(), sorted_ids.data_ptr(), sorted_weights.data_ptr(),
        sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w13.data_ptr(), w2.data_ptr(), w13_scale.data_ptr(),
        w2_scale.data_ptr(), sh[0].data_ptr(), sh[1].data_ptr(), sh[2].data_ptr(), sh[3].data_ptr(),
        moe_buf.data_ptr(), mid.data_ptr(), counters.data_ptr(), m, blocks,
        float(swiglu_limit) if swiglu_limit else float("inf"), torch.cuda.current_stream(),
    )  # fmt: skip
    return moe_buf
