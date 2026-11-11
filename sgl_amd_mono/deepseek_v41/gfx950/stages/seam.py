# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/kernels/attn_pre.py (slice, gate, norm) and
# atom/models/deepseek_v41/mono/kernels/attn_post.py (the attention's TP reduce).
"""The mHC seam stages (slice, gate, norm, reduce, push), handing off through the tagged mailbox."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.math import fma
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_SYS,
    batched_rounds,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    first_rounds,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    rsrc,
    sum_partials,
)
from sgl_amd_mono.common.plan import (
    THREADS,
    WAVES,
    cdiv,
    pair_layout,
)
from sgl_amd_mono.common.sync import _spin, preg
from sgl_amd_mono.deepseek_v41.gfx950.stages.gemv import (
    ROWS,
    TILE,
    ld_bf,
    ld_f32,
    plus,
    token_tiles,
)

HC = 4
HIDDEN = 5120
K_MIX = HC * HIDDEN
MIX = HC * (HC + 2)  # 24 projections: pre 4 | post 4 | comb 16
EPS = 1e-20
HC_EPS = 1e-6
SINKHORN = 20
POST_MULT = 2.0
LOG2E = 1.0 / math.log(2.0)

COLS = 32  # hidden columns a slice task
SLICES = HIDDEN // COLS
KT = HC * COLS  # projection columns a slice task
PART = MIX + 1  # a slice's partials a token: the mixes, then the sum of squares
GATE_BLOCK = 8  # slices a gate thread sums


def scratch_layout(tokens: int) -> dict[str, tuple[int, int]]:
    """Mailbox region -> (byte offset, bytes) of a seam; 8 B a pair."""
    pairs = {
        "lin": tokens * HIDDEN // 2,  # bf16 pairs
        "pmix": SLICES * tokens * PART,
    }
    return pair_layout(pairs.items())


# ---------------------------------------------------------------- seam state
# A launch can run the next layer's attention seam slice past its MoE reduce
# (``MonoBuild.seam_next``): the next launch then reads that seam's R_new, its layer input
# (LIN, bf16) and the mixes' partials (f32) as plain rows (``MonoBuild.seamed``).
SEAM_MAX_TOKENS = (
    8  # widths the slice's first loads fit in registers beside the MoE reduce's
)


def seam_state_layout(tokens: int) -> dict[str, tuple[int, int]]:
    """A seam state's regions -> (byte offset, bytes)."""
    out, off = {}, 0
    for name, n in (
        ("res", tokens * HC * HIDDEN * 2),
        ("lin", tokens * HIDDEN * 2),
        ("pmix", SLICES * tokens * PART * 4),
    ):
        out[name] = (off, n)
        off = -(-(off + n) // 256) * 256
    return out


def seam_state_bytes(tokens: int) -> int:
    return max(o + n for o, n in seam_state_layout(tokens).values())


def plain_put(base, i, v):
    """Word i := ``v`` (f32 or i32), a plain store the next launch reads."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    bo.buffer_store(bits, rsrc(base), i)


def plain_put_bf(base, i, vs):
    """Elements i, i + 1 (i even) as one bf16 pair, a plain store the next launch reads."""
    bo.buffer_store(bf16_pair(vs[0], vs[1]).bitcast(fx.Int32), rsrc(base), i // 2)


def plain_loads(specs):
    """``Mailbox.poll``'s result for words the previous launch stored (``plain_put``)."""
    return [
        [fx.Int32(bo.buffer_load(rsrc(b), fx.Int32(i), vec_width=1, dtype=T.i32))]
        for b, i, _ in specs
    ]


def attn_peer_bytes(tokens: int, tp: int) -> int:
    """ATTN: ``[source rank][token][hidden / 2]`` bf16 pairs."""
    return tp * tokens * HIDDEN // 2 * 8


def attn_region_pair(src, t, s, col):
    return (src * s + t) * (HIDDEN // 2) + col // 2


# ---------------------------------------------------------------- ATTN cells
# K2's wo_b partials (not token parallel) cross as cells: 8 B, four bf16 values and no
# tag, half a tagged pair's bytes. Each 4-B word is its own flag: a word not yet written
# holds CELL_EMPTY (all ones), and a writer never stores it (a bf16 pair of 0xFFFF NaNs
# goes out as 0xFFFE, 0xFFFF), so a cell is read only once both its words have landed,
# whether its 8 B arrive at once or as two halves. The reduce stores both words back to
# empty once read; the runner fills the region with it. Every cell a launch writes, its
# reduce reads; every width's cells lie on one 8-B grid of a region of their own
# (``token_par.acell_region``), so no other word reaches a cell.
CELL_VALUES = 4
CELL_EMPTY = -1  # each of a cell's two words
WOB_TASKS = HIDDEN // ROWS


def cells_tile(n: int) -> int:
    """A wo_b task's cells over a token tile of n rows: [row][16 columns], 4 a cell."""
    return cdiv(ROWS * n, CELL_VALUES)


def cells_task(s: int) -> int:
    """A wo_b task's cells over an s-row step, a token tile after another."""
    return sum(cells_tile(n) for _, n in token_tiles(s))


def acell_cells(s: int, tp: int) -> int:
    """ACELL: ``[source rank][wo_b task][token tile][cell]``."""
    return tp * WOB_TASKS * cells_task(s)


def attn_cell(src, task, s, t0, u):
    """Cell u of wo_b ``task`` on the token tile from row t0, from rank ``src``."""
    return (src * WOB_TASKS + task) * cells_task(s) + t0 // TILE * cells_tile(TILE) + u


# ---------------------------------------------------------------- FFN cells
# The per-row MoE's down sums cross as cells too: [source rank][row][hidden / 4], four bf16
# values a cell, empty as ATTN's (the reader stores them back to empty).
def ffn_cells(s: int, tp: int) -> int:
    """FCELL's cells."""
    return tp * s * HIDDEN // CELL_VALUES


def ffn_cell(src, t, s, col):
    """The cell of rank ``src``'s down sums of row t at hidden column ``col`` (a multiple of 4)."""
    return (src * s + t) * (HIDDEN // CELL_VALUES) + col // CELL_VALUES


def cell_words(vs):
    """Four values as a cell's two words, never an empty word: a pair of bf16 0xFFFF NaNs
    goes out as 0xFFFE, 0xFFFF."""
    ws = []
    for k in range_constexpr(2):
        w = bf16_pair(vs[2 * k], vs[2 * k + 1]).bitcast(fx.Int32)
        ws.append((w == CELL_EMPTY).select(fx.Int32(CELL_EMPTY - 1), w))
    return fx.Vector.from_elements(ws, fx.Int32)


def sum_cells(addr, cells):
    """Poll the TP ranks' cells at ``addr`` and sum each of the four values in fp32 in rank
    order."""
    ws = poll_cells(addr, cells)
    out = []
    for i in range_constexpr(CELL_VALUES):
        acc = cell_value(ws[0], i)
        for src in range_constexpr(1, len(cells)):
            acc = acc + cell_value(ws[src], i)
        out.append(acc)
    return out


def sigmoid(z):
    """Triton's ``tl.sigmoid``: 1 / (1 + exp2(-z log2(e)))."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def pend_value(c, t, col, cc):
    """Return the owed post's sublayer output at (token t, column col)."""
    if const_expr(c.get("pend_lds") is None):
        return ld_bf(c["args"]["pend"], t * HIDDEN + col)
    return fx.ptr_load(c["pend_lds"] + (t * COLS + cc))


# ---------------------------------------------------------------- slice
def slice_pre_loads(c):
    """The pre mixes of this thread's token in each pass of ``slice_input``, loaded."""
    s, tid = c["S"], c["tid"]
    out = []
    for pas in range(0, s * COLS // 2, THREADS):
        t = fx.min(plus(pas, tid) // (COLS // 2), s - 1)
        out.append([ld_f32(c["args"]["pre_in"], t * HC + h) for h in range(HC)])
    return out


@flyc.jit
def slice_input(c, task, idx, pre):
    """Write thread ``idx``'s pair of the layer input to LIN (``pre``: its token's pre
    mixes, ``slice_pre_loads``); with ``aux_on``, also store the pair's mean over the
    R_new streams, as DSpark captures a target layer."""
    s, rl = c["S"], c["rl"]
    a = c["args"]
    # read before the dynamic branch: a method call on c inside one carries c
    aux_on = c.get("aux_on")
    c0 = task * COLS
    if idx < s * COLS // 2:
        t = idx // (COLS // 2)
        cc = 2 * (idx % (COLS // 2))
        outs, means = [], []
        for d in range_constexpr(2):
            acc = fx.Float32(0.0)
            total = fx.Float32(0.0)
            for h in range_constexpr(HC):
                rb = fx.ptr_load(rl + (t * KT + h * COLS + cc + d))
                acc = acc + pre[h] * rb
                # torch's mean over the streams: summed in order, then times 1/HC
                total = rb if h == 0 else total + rb
            outs.append(acc)
            means.append(total * fx.Float32(1.0 / HC))
        c["put_bf"](c["lin"], t * HIDDEN + c0 + cc, outs)
        if const_expr(aux_on is not None):
            if aux_on:
                for k in range_constexpr(2):
                    bo.buffer_store(
                        means[k].to(fx.BFloat16),
                        rsrc(a["aux_out"]),
                        t * a["aux_stride"] + c0 + cc + k,
                    )


def slice_at(c0, e):
    """Element e of a slice task at column c0: (token, stream, hidden column)."""
    return e // KT, (e % KT) // COLS, c0 + e % COLS


def slice_loads(c, c0, e):
    """Element e's loads of ``stage_slice`` (e clamped into the step's)."""
    a = c["args"]
    t, o, col = slice_at(c0, e)
    if const_expr(c.get("embed", False)):
        # the first layer: every stream is the token's embedding row (SGLang's hc_broadcast)
        row = fx.Int32(
            bo.buffer_load(rsrc(a["ids"]), a["ids_words"] * t, vec_width=1, dtype=T.i32)
        )
        return (ld_bf(a["emb"], row * HIDDEN + col),)
    if const_expr(not c.get("fold", True)):
        return (ld_bf(a["res_in"], (t * HC + o) * HIDDEN + col),)
    r = [ld_bf(a["res_in"], (t * HC + h) * HIDDEN + col) for h in range(HC)]
    cm = [ld_f32(a["comb_in"], (t * HC + h) * HC + o) for h in range(HC)]
    return r, cm, ld_f32(a["post_in"], t * HC + o)


def slice_row(c, c0, e, got):
    """Element e of ``stage_slice``'s R_new from ``slice_loads``' values."""
    a = c["args"]
    t, o, col = slice_at(c0, e)
    if const_expr(not c.get("fold", True)):
        # no post owed (first layer, after Engram): stream o as is, as SGLang's x-less boundary
        v = got[0]
    else:
        r, cm, post = got
        # aiter's mhc_fused_post_pre_delayed_rmsnorm order: ((post x + c0 r0) + c1 r1) + c2 r2, then fma(c3, r3, ..)
        v = post * pend_value(c, t, col, e % COLS) + (r[0] * cm[0])
        for h in range_constexpr(1, HC - 1):
            v = v + cm[h] * r[h]
        # bf16 R_new feeds the collapse, the sums of squares and the mix projection (no fma, as Triton)
        if const_expr(c.get("sgl", False)):
            v = bf16_round(v + cm[HC - 1] * r[HC - 1])
        else:
            v = bf16_round(fma(cm[HC - 1], r[HC - 1], v))
    bo.buffer_store(v.to(fx.BFloat16), rsrc(a["res_out"]), (t * HC + o) * HIDDEN + col)
    fx.ptr_store(v, c["rl"] + (t * KT + o * COLS + e % COLS))


def slice_fn_loads(c, task):
    """Slice ``task``'s f32 mix weights (they wait on nothing), fn in full fp32 as
    SGLang's boundary kernel (an IEEE fp32 dot) applies it."""
    tid, a, c0 = c["tid"], c["args"], task * COLS
    out = []
    for i in range_constexpr((MIX * KT + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, MIX * KT - 1)
        j = e // KT
        o = (e % KT) // COLS
        out.append(ld_f32(a["hc_fn"], j * K_MIX + o * HIDDEN + c0 + e % COLS))
    return out


def slice_prefetch_fits(s: int) -> bool:
    """Whether a slice's first loads fit in registers through the FFN seam's reduce: up to
    32 rows, where the reduce takes one pass."""
    return s * COLS // 2 <= THREADS


def slice_prefetch(c, task):
    """Issue slice ``task``'s first loads, its pre mixes, its rows' first batch and its mix
    weights, ahead of a wait they do not depend on (the FFN seam's attention partials)."""
    c0 = task * COLS
    pm = slice_pre_loads(c)
    first = first_rounds(c["tid"], c["S"] * KT, lambda e: slice_loads(c, c0, e))
    return first, slice_fn_loads(c, task), pm


@flyc.jit
def stage_slice(c, task, pre=None):
    """Compute R_new, the layer input and the partial projections of columns 32 task ..
    +32 (``pre``: ``slice_prefetch``'s loads)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rl, fl = c["rl"], c["fl"]
    c0 = task * COLS
    if const_expr(pre is None):
        # the pre mixes first: they return ahead of the rows (in issue order)
        pm = slice_pre_loads(c)
        first, fn = None, slice_fn_loads(c, task)
    else:
        first, fn, pm = pre
    batched_rounds(
        tid,
        s * KT,
        lambda e: slice_loads(c, c0, e),
        lambda e, got: slice_row(c, c0, e, got),
        first=first,
    )
    for i in range_constexpr(len(fn)):
        e = tid + THREADS * i
        if e < MIX * KT:
            fx.ptr_store(fn[i], fl + e)
    gpu.barrier()
    # layer input: bf16 R_new times pre, products rounded, summed in order (as the Triton collapse)
    for pas in range_constexpr(0, s * COLS // 2, THREADS):
        slice_input(c, task, plus(pas, tid), pm[pas // THREADS])
    # sum of squares: token t on wave t % WAVES, two columns a lane
    for t in range_constexpr(s):
        if wave == t % WAVES:
            x0 = fx.ptr_load(rl + (t * KT + 2 * lane))
            x1 = fx.ptr_load(rl + (t * KT + 2 * lane + 1))
            sq = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
            if lane == 0:
                c["put"](c["pmix"], (task * s + t) * PART + MIX, sq)
    # mixes: fp32 MFMA 16x16x4, A = fn rows, B = R_new^T, a token tile at a time
    for t0, n in token_tiles(s):
        slice_mix(c, task, t0, n)


@flyc.jit
def slice_mix(c, task, t0, n):
    """The mix projection partials of tokens t0 .. t0 + n (``stage_slice``)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rl, fl, red = c["rl"], c["fl"], c["red"]
    tok = plus(t0, fx.min(lane % 16, n - 1))
    kw = KT // WAVES
    cs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for kk in range_constexpr(kw // 4):
        colk = wave * kw + lane // 16 + 4 * kk
        b = fx.ptr_load(rl + (tok * KT + colk))
        for h in range_constexpr(2):
            row = h * 16 + lane % 16
            av = fx.ptr_load(fl + (fx.min(row, MIX - 1) * KT + colk))
            av = (row < MIX).select(av, fx.Float32(0.0))
            ops = rocdl._split_mfma_operands([av, b, cs[h], 0, 0, 0])
            cs[h] = fx.Vector(rocdl.mfma_f32_16x16x4f32(T.vec(4, T.f32), *ops).result)
    for h in range_constexpr(2):
        fx.ptr_store(cs[h], red + ((wave * 2 + h) * 64 + lane) * 4)
    gpu.barrier()
    if tid < MIX * n:
        tl = tid // MIX
        t = plus(t0, tl)
        j = tid % MIX
        h = j // 16
        ln = 16 * ((j % 16) // 4) + tl
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(red + (((w * 2 + h) * 64 + ln) * 4 + j % 4))
        c["put"](c["pmix"], (task * s + t) * PART + j, tot)
    gpu.barrier()


# ---------------------------------------------------------------- gate
# comb[r][c] on lane COMB0 + 4 r + c, one DPP row: xor-butterfly sums, (e0 + e1) + (e2 + e3)
COMB0 = 16


def _row_sum(v):
    return butterfly(v, (1, 2))


def _col_sum(v):
    return butterfly(v, (4, 8))


def _row_max(v):
    return butterfly(v, (1, 2), fx.max)


@flyc.jit
def stage_gate(c, t):
    """Token t's slices, in slice order, -> post / comb / next pre (``dev``: device
    scope, for mixes this launch reads)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red = c["red"]
    a = c["args"]
    cm = CM_DEV if const_expr(c.get("dev", False)) else 0
    groups = SLICES // GATE_BLOCK
    if tid < groups * PART:
        g = tid // PART
        j = tid % PART
        vals = c["poll"](
            [
                (c["pmix"], ((g * GATE_BLOCK + k) * s + t) * PART + j, 1)
                for k in range(GATE_BLOCK)
            ]
        )
        acc = vals[0][0].bitcast(fx.Float32)
        for k in range_constexpr(1, GATE_BLOCK):
            acc = acc + vals[k][0].bitcast(fx.Float32)
        fx.ptr_store(acc, red + (g * PART + j))
    gpu.barrier()
    if wave == 0:
        # lanes 0..7 pre and post, COMB0.. comb (the lanes between repeat 0..7)
        j = (lane < COMB0).select(
            lane % (2 * HC), fx.min(lane - (COMB0 - 2 * HC), MIX - 1)
        )
        mix = fx.ptr_load(red + j)
        sq = fx.ptr_load(red + MIX)
        for g in range_constexpr(1, groups):
            mix = mix + fx.ptr_load(red + (g * PART + j))
            sq = sq + fx.ptr_load(red + (g * PART + MIX))
        sc = [ld_f32(a["hc_scale"], i) for i in range(3)]
        base = ld_f32(a["hc_base"], j)
        # aiter's reduce: rstd = rsq(fma(sum, 1/K, eps)), then each gate fma(mix rstd, scale, base)
        v = mix * hw_rsq(fma(sq, fx.Float32(1.0 / K_MIX), fx.Float32(EPS)))
        if lane < HC:
            pre = sigmoid(fma(v, sc[0], base))
            bo.buffer_store(
                pre + fx.Float32(HC_EPS),
                rsrc(a["pre_out"]),
                t * HC + lane,
                cache_modifier=cm,
            )
        if (lane >= HC) & (lane < 2 * HC):
            post = sigmoid(fma(v, sc[1], base))
            bo.buffer_store(
                post * fx.Float32(POST_MULT),
                rsrc(a["post_out"]),
                t * HC + lane - HC,
                cache_modifier=cm,
            )
        cv = fma(v, sc[2], base)
        # Triton's exp: exp2 of (x - max) log2(e)
        cv = hw_exp2((cv - _row_max(cv)) * fx.Float32(LOG2E))
        # fast_dividef is x rcp(y); its first "+ eps" contracts into the fma
        cv = fma(cv, hw_rcp(_row_sum(cv)), fx.Float32(HC_EPS))
        cv = cv * hw_rcp(_col_sum(cv) + fx.Float32(HC_EPS))
        for _ in range_constexpr(SINKHORN - 1):
            cv = cv * hw_rcp(_row_sum(cv) + fx.Float32(HC_EPS))
            cv = cv * hw_rcp(_col_sum(cv) + fx.Float32(HC_EPS))
        if (lane >= COMB0) & (lane < COMB0 + HC * HC):
            bo.buffer_store(
                cv, rsrc(a["comb_out"]), t * HC * HC + lane - COMB0, cache_modifier=cm
            )
    gpu.barrier()


# ---------------------------------------------------------------- norm
@flyc.jit
def stage_norm(c, t, store):
    """Token t's RMSNorm: rstd = rsq(fma(sum, 1/H, eps)), bf16((x rstd) w), handed to ``store``.

    Threads 0..255 poll and sum three 8-column chunks each (columns 8 u + 2048 ch); the
    outputs are spread over every wave: column 8 tid for each thread (past 255, chunk 1
    of thread tid - 256, handed over in LDS), and 4096 + 8 tid for threads 0..127."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    tt = fx.min(tid, 255)
    mine = tid < 256
    cols = [tt * 8 + 2048 * ch for ch in range(3)]
    at = [fx.min(cols[ch], HIDDEN - 8) for ch in range(3)]
    own = tid * 8
    # the outputs: column 8 tid (chunk 0 below 256, chunk 1's column past it), then
    # 4096 + 8 tid below 128 (chunk 2)
    out_cols = [own, fx.min(4096 + tid * 8, HIDDEN - 8)]
    # the norm weights first: they do not wait on the LIN rows; a chunk's 8 in one 16 B load
    wv = [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["norm_w"]), out_cols[k] // 2, vec_width=4, dtype=T.i32
            )
        )
        for k in range(2)
    ]
    wts = [[f(wv[k][q]) for q in range(4) for f in (bf_lo, bf_hi)] for k in range(2)]
    words = c["poll"](
        [
            (c["lin"], (t * HIDDEN + at[ch]) // 2 + k, 1)
            for ch in range(3)
            for k in range(4)
        ]
    )
    # chunk 1's words for thread tid + 256
    fl = c["fl"]
    if mine:
        for k in range_constexpr(4):
            fx.ptr_store(words[4 + k][0].bitcast(fx.Float32), fl + (tt * 4 + k))
    xs = []
    for ch in range_constexpr(3):
        v = []
        for k in range_constexpr(4):
            v += [bf_lo(words[4 * ch + k][0]), bf_hi(words[4 * ch + k][0])]
        xs.append([(cols[ch] < HIDDEN).select(x, fx.Float32(0.0)) for x in v])
    sgl = c.get("sgl", False)
    if mine:
        if const_expr(sgl):
            # SGLang's rmsnorm_fake_quant_row reduction tree, step for step
            e = [
                fma(xs[2][j], xs[2][j], fma(xs[0][j], xs[0][j], xs[1][j] * xs[1][j]))
                for j in range(8)
            ]
            acc = ((e[0] + e[2]) + (e[4] + e[6])) + ((e[1] + e[3]) + (e[5] + e[7]))
            acc = butterfly(acc, (8, 4, 2, 1, 16, 32))
        else:
            acc = fx.Float32(0.0)
            for ch in range_constexpr(3):
                for x in xs[ch]:
                    acc = acc + x * x
            acc = butterfly(acc, (1, 2, 4, 8, 16, 32))
        if lane == 0:
            fx.ptr_store(acc, red + wave)
    gpu.barrier()
    if const_expr(sgl):
        # rstd = 1 / sqrt_rn(sum / K + eps), IEEE division and sqrt
        tot = butterfly(fx.ptr_load(red + lane % 4), (2, 1))
        var = tot / fx.Float32(float(HIDDEN)) + fx.Float32(EPS)
        r = fx.Float32(1.0) / fmath.sqrt(var)
    else:
        tot = butterfly(fx.ptr_load(red + lane % 4), (1, 2))
        r = hw_rsq(fma(tot, fx.Float32(1.0 / HIDDEN), fx.Float32(EPS)))
    # column 8 tid: chunk 0 below 256, else thread tid - 256's chunk 1
    up = fx.max(tid - 256, 0) * 4
    x1 = []
    for k in range_constexpr(4):
        w = fx.ptr_load(fl + (up + k)).bitcast(fx.Int32)
        x1 += [bf_lo(w), bf_hi(w)]
    x0 = [mine.select(xs[0][j], x1[j]) for j in range(8)]
    ys = [bf16_round((x0[j] * r) * wts[0][j]) for j in range(8)]
    store(t, out_cols[0], ys, own < HIDDEN)
    if tid < 128:
        y2 = [bf16_round((xs[2][j] * r) * wts[1][j]) for j in range(8)]
        store(t, out_cols[1], y2, cols[2] < HIDDEN)
    gpu.barrier()


# ---------------------------------------------------------------- reduce
@flyc.jit
def stage_reduce(c, task):
    """Sum the TP ranks' attention partials at this slice's 32 columns in rank order into
    ``pend_lds``: K2's cells (``c["acell"]``) or the tagged pairs. ``c["cell0"]``: the cells'
    first thread, a wave that pushed no wo_b cells (their reset then waits for
    ``reset_attn_cells``)."""
    s, tid = c["S"], c["tid"]
    first = c.get("cell0", 0)
    if const_expr(c.get("acell") is not None):
        units, reduce_unit = 2 * cells_task(s), reduce_attn_cell
    else:
        units, reduce_unit = s * COLS // 2, reduce_attn_pair
    if const_expr(units <= THREADS - first):
        reduce_unit(c, task, tid - first)
    else:
        # past one pass a traced loop: one pass's polls in flight at a time, not every pass's
        for pas in range(0, units, THREADS):
            reduce_unit(c, task, pas + tid)
    gpu.barrier()


@flyc.jit
def reduce_attn_pair(c, task, idx):
    """Thread ``idx``'s column pair of token idx / 16 (``stage_reduce``)."""
    s = c["S"]
    c0 = task * COLS
    if idx < s * COLS // 2:
        t = idx // (COLS // 2)
        col = c0 + 2 * (idx % (COLS // 2))
        own = preg(c["sym"], 0, "attn")
        acc0, acc1 = sum_partials(
            c["poll"],
            own,
            lambda src: attn_region_pair(src, t, s, col),
            c["d"].tp,
        )
        lds = c["pend_lds"] + (t * COLS + col - c0)
        fx.ptr_store(bf16_round(acc0), lds)
        fx.ptr_store(bf16_round(acc1), lds + 1)


@flyc.jit
def reduce_attn_cell(c, task, idx):
    """Thread ``idx``'s cell of the slice's two wo_b tasks: every rank's cell, once written,
    summed value by value in rank order into ``pend_lds``; then the cells emptied (with
    ``c["cell0"]``, by ``reset_attn_cells``)."""
    s, tp, pend, acell = c["S"], c["d"].tp, c["pend_lds"], c["acell"]
    per = cells_task(s)
    full = cells_tile(TILE)
    first = c.get("cell0", 0)
    live = idx < 2 * per
    if const_expr(first > 0):
        live = (idx >= 0) & live
    if live:
        j = idx // per
        at = idx % per  # the cell in the task's block: its tile, then the tile's cell
        tile = at // full
        u = at - tile * full
        cells = slice_cells(c, task, idx)
        ws = poll_cells(acell, cells)
        for i in range_constexpr(CELL_VALUES):
            acc = cell_value(ws[0], i)
            for src in range_constexpr(1, tp):
                acc = acc + cell_value(ws[src], i)
            v = CELL_VALUES * u + i
            at_lds = pend + ((tile * TILE + v // ROWS) * COLS + j * ROWS + v % ROWS)
            fx.ptr_store(bf16_round(acc), at_lds)
        if const_expr(first == 0):
            empty_cells(acell, cells)


def slice_cells(c, task, idx):
    """Thread ``idx``'s cell of the slice's two wo_b tasks, from every rank."""
    per = cells_task(c["S"])
    return [
        attn_cell(src, COLS // ROWS * task + idx // per, c["S"], 0, 0) + idx % per
        for src in range(c["d"].tp)
    ]


def empty_cells(addr, cells):
    """Store the cells at ``addr`` back to empty, once read."""
    empty = fx.Vector.from_elements([fx.Int32(CELL_EMPTY)] * 2, fx.Int32)
    for at in cells:
        bo.buffer_store(empty, rsrc(addr), at * 2, cache_modifier=CM_SYS)


@flyc.jit
def reset_attn_cells(c, task):
    """``stage_reduce``'s cells back to empty, past the slice (``c["cell0"]``)."""
    idx = c["tid"] - c["cell0"]
    if (idx >= 0) & (idx < 2 * cells_task(c["S"])):
        empty_cells(c["acell"], slice_cells(c, task, idx))


def cell_value(w, i):
    """Value i of a cell's words (w0, w1): bf16 low then high, a word after the other."""
    return (bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1]))[i]


def poll_cells(addr, cells):
    """Load the cells at ``addr`` (system scope) until none is empty: [(w0, w1)]."""

    def load_all():
        words = []
        for at in cells:
            w = fx.Vector(
                bo.buffer_load(
                    rsrc(addr), at * 2, vec_width=2, dtype=T.i32, cache_modifier=CM_SYS
                )
            )
            words += [w[0], w[1]]
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        # word by word: a cell whose 8 B land as two 4-B halves reads as pending until both do
        bad = v[0] == CELL_EMPTY
        for e in range_constexpr(1, 2 * len(cells)):
            bad = bad | (v[e] == CELL_EMPTY)
        return bad

    v = _spin(load_all, pending)
    return [(v[2 * k], v[2 * k + 1]) for k in range(len(cells))]


@flyc.jit
def stage_push(c, task, part):
    """Push this rank's attention partial at this slice's 32 columns to every rank's ATTN region."""
    s, tid, tp = c["S"], c["tid"], c["d"].tp
    for pas in range_constexpr(0, s * COLS // 2, THREADS):
        idx = plus(pas, tid)
        if idx < s * COLS // 2:
            t = idx // (COLS // 2)
            col = task * COLS + 2 * (idx % (COLS // 2))
            w = bo.buffer_load(
                rsrc(part), (t * HIDDEN + col) // 2, vec_width=1, dtype=T.i32
            )
            pair = attn_region_pair(c["rank"], t, s, col)
            for p in range_constexpr(tp):
                c["put_words"](preg(c["peer_addr"](p), 0, "attn"), pair, [w])
