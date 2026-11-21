# SPDX-License-Identifier: Apache-2.0
"""The MLP AttnRes seam, on the attn CTAs: slice, gate, mix.

These three stages run in this order wherever the seam is reached. In this
package that is the layer launch (``layer.py``) and the o_proj launch the
kernel checks use as its reference (``oproj.py``). The seam belongs to
neither, which is why it lives here.

A stage reads its state from the ``c`` dict its launch built.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    butterfly,
    hw_exp2,
    hw_rsq,
    load_ptr64,
    rsrc,
    wave_max,
)
from sgl_amd_mono.common.plan import THREADS
from sgl_amd_mono.common.sync import publish
from sgl_amd_mono.kimi_k3.gfx950.stages.dims import (
    COLS,
    GSL,
    HIDDEN,
    LOG2E,
    NA,
    QC,
)


def ld_bf4(ptr, i):
    """bf16 elements i .. i + 4 (i a multiple of 4) as four f32."""
    w = fx.Vector(bo.buffer_load(rsrc(ptr), i // 2, vec_width=2, dtype=T.i32))
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


def bf4_words(v):
    """Four f32 -> two words of packed bf16."""
    return fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)


def bf16r(v):
    return fx.Float32(fx.Float32(v).to(fx.BFloat16))


def f32_of(w):
    return w.bitcast(fx.Float32)


def sum_polled(vals):
    acc = f32_of(vals[0][0])
    for k in range(1, len(vals)):
        acc = acc + f32_of(vals[k][0])
    return acc


# ---------------------------------------------------------------- attn CTAs
def _raw4(ptr, i):
    """bf16 elements i .. i + 4 (i a multiple of 4) as two raw words."""
    return fx.Vector(bo.buffer_load(rsrc(ptr), i // 2, vec_width=2, dtype=T.i32))


def _f4(w):
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


def lds_words2(ptr, i):
    """Two LDS words at word i (packed bf16 pairs)."""
    return fx.Vector(
        fx.ptr_load(ptr + i, result_type=fx.Vector.make_type(2, fx.Int32))
    )


def _slice_item(c, task, i):
    """Round i's work item of this thread: (live, row, token, source, column)."""
    s, ns, tid = c["S"], c["NS"], c["tid"]
    n = s * ns * QC
    e = fx.min(tid + THREADS * i, n - 1)
    row = e // QC  # (token, source)
    t = row // ns
    return (tid + THREADS * i) < n, e, row, t, row % ns, task * COLS + (e % QC) * 4


@flyc.jit
def stage_slice(c, task):
    """Columns 128 task ..: every token's sources into LDS ``vals`` (4 columns a
    thread), the prefix (+ delta) rounded to bf16 and stored back, per-(token,
    source) partial sum of squares and qk dot -> PART."""
    s, ns, nb, lane = c["S"], c["NS"], c["NB"], c["lane"]
    a = c["args"]
    vals_bf16 = c.get("vals_bf16", False)
    # the incoming prefix; K2 under ``fold`` reads it where the stream left it
    pre_in = a["prefix_in"] if "prefix_in" in a else a["prefix"]
    # under ``fold`` the reduce CTAs read the updated prefix back: store it
    # device-coherent (they may sit on another XCD)
    pre_cm = CM_DEV if c.get("fold", False) else 0
    n = s * ns * QC  # threads' work items
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        live, e, row, t, src, col = _slice_item(c, task, i)
        is_pre = src == nb
        blk = ld_bf4(
            a["blocks"], t * a["blk_sm"] + fx.min(src, nb - 1) * a["blk_sr"] + col
        )
        pre = ld_bf4(pre_in, t * HIDDEN + col)
        if const_expr(c["delta"]):
            if const_expr(c.get("delta_lds") is not None and vals_bf16):
                # the delta reduced into LDS first, as packed bf16 pairs
                dl = _f4(lds_words2(c["delta_lds"], (t * COLS + (e % QC) * 4) // 2))
            elif const_expr(c.get("delta_lds") is not None):
                # the delta reduced into LDS first (K2a: the all-reduced o_proj)
                dl = [
                    fx.ptr_load(c["delta_lds"] + (t * COLS + (e % QC) * 4 + j))
                    for j in range(4)
                ]
            else:
                dl = ld_bf4(a["delta"], t * HIDDEN + col)
            if const_expr(c.get("reset", False)):
                # a block-write layer's MLP seam: the prefix restarts at the
                # attention's output (the input prefix is not read)
                pre = dl
            else:
                pre = [bf16r(pre[j] + dl[j]) for j in range(4)]
            if live & is_pre:
                bo.buffer_store(
                    bf4_words(pre),
                    rsrc(a["prefix"]),
                    (t * HIDDEN + col) // 2,
                    cache_modifier=pre_cm,
                )
        if const_expr(c.get("write_idx", -1) >= 0):  # noqa: SIM102 (traced: `and` does not combine traced values)
            # the block-write layer's attention seam: the updated prefix is
            # block write_idx
            if live & is_pre:
                bo.buffer_store(
                    bf4_words(pre),
                    rsrc(a["blocks"]),
                    (t * a["blk_sm"] + c["write_idx"] * a["blk_sr"] + col) // 2,
                )
        v = [
            live.select(is_pre.select(pre[j], blk[j]), fx.Float32(0.0))
            for j in range(4)
        ]
        w = ld_bf4(a["ares_nw"], col)
        q = ld_bf4(a["ares_qk"], col)
        # a thread past the end holds a clamped copy of the last element: it
        # must not store (it raced the owner with zeros)
        if const_expr(vals_bf16):
            # every source is bf16: packed pairs hold it exactly
            if live:
                fx.ptr_store(
                    bf4_words(v), c["vals"] + (row * COLS + (e % QC) * 4) // 2
                )
        elif live:
            for j in range_constexpr(4):
                fx.ptr_store(v[j], c["vals"] + (row * COLS + (e % QC) * 4 + j))
        sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
        dt = (
            v[0] * (w[0] * q[0])
            + v[1] * (w[1] * q[1])
            + v[2] * (w[2] * q[2])
            + v[3] * (w[3] * q[3])
        )
        sq = butterfly(sq, (16, 8, 4, 2, 1))
        dt = butterfly(dt, (16, 8, 4, 2, 1))
        if live & (lane % 32 == 0):
            at = ((task * s + t) * ns + src) * 2
            c["put"](c["part"], at, sq)
            c["put"](c["part"], at + 1, dt)
    gpu.barrier()


# ---------------------------------------------------------------- split slice
# ``stage_slice`` in two halves, the same values and stores: the block sources
# (which wait on nothing in this launch) before the o_proj poll, the prefix
# source (prefix + delta) after it.
def _block_item(c, task, i):
    """Round i's block-source item of this thread: (live, e, token, source, column)."""
    s, nb, tid = c["S"], c["NB"], c["tid"]
    n = s * nb * QC
    e = fx.min(tid + THREADS * i, n - 1)
    row = e // QC  # (token, block)
    return (tid + THREADS * i) < n, e, row // nb, row % nb, task * COLS + (e % QC) * 4


def _prefix_item(c, task, i):
    """Round i's prefix-source item of this thread: (live, e, token, column)."""
    s, tid = c["S"], c["tid"]
    n = s * QC
    e = fx.min(tid + THREADS * i, n - 1)
    return (tid + THREADS * i) < n, e, e // QC, task * COLS + (e % QC) * 4


def split_slice_loads(c, task):
    """The split slice's loads: every block item's words, every prefix item's
    incoming prefix words, the norm / qk words (the same columns every item)."""
    s, nb = c["S"], c["NB"]
    a = c["args"]
    pre_in = a["prefix_in"] if "prefix_in" in a else a["prefix"]
    blk = []
    for i in range((s * nb * QC + THREADS - 1) // THREADS):
        _, _, t, src, col = _block_item(c, task, i)
        blk.append(_raw4(a["blocks"], t * a["blk_sm"] + src * a["blk_sr"] + col))
    pre = []
    for i in range((s * QC + THREADS - 1) // THREADS):
        _, _, t, col = _prefix_item(c, task, i)
        pre.append(_raw4(pre_in, t * HIDDEN + col))
    # every item's columns are its thread's (THREADS % QC == 0)
    col = task * COLS + (c["tid"] % QC) * 4
    return {
        "blk": blk,
        "pre": pre,
        "w": _raw4(a["ares_nw"], col),
        "q": _raw4(a["ares_qk"], col),
    }


@flyc.jit
def _slice_out(c, task, live, e, t, src, v, w, q):
    """An item's values -> LDS ``vals``, its (token, source) sum of squares and
    qk dot -> PART (``stage_slice``'s)."""
    s, ns, lane = c["S"], c["NS"], c["lane"]
    row = t * ns + src
    if const_expr(c.get("vals_bf16", False)):
        if live:
            fx.ptr_store(bf4_words(v), c["vals"] + (row * COLS + (e % QC) * 4) // 2)
    elif live:
        for j in range_constexpr(4):
            fx.ptr_store(v[j], c["vals"] + (row * COLS + (e % QC) * 4 + j))
    sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    dt = (
        v[0] * (w[0] * q[0])
        + v[1] * (w[1] * q[1])
        + v[2] * (w[2] * q[2])
        + v[3] * (w[3] * q[3])
    )
    sq = butterfly(sq, (16, 8, 4, 2, 1))
    dt = butterfly(dt, (16, 8, 4, 2, 1))
    if live & (lane % 32 == 0):
        at = ((task * s + t) * ns + src) * 2
        c["put"](c["part"], at, sq)
        c["put"](c["part"], at + 1, dt)


@flyc.jit
def slice_blocks(c, task, ld):
    """The split slice's block sources (``split_slice_loads``' words)."""
    s, nb = c["S"], c["NB"]
    w, q = _f4(ld["w"]), _f4(ld["q"])
    for i in range_constexpr((s * nb * QC + THREADS - 1) // THREADS):
        live, e, t, src, _ = _block_item(c, task, i)
        blk = _f4(ld["blk"][i])
        v = [live.select(blk[j], fx.Float32(0.0)) for j in range(4)]
        _slice_out(c, task, live, e, t, src, v, w, q)


@flyc.jit
def slice_prefix(c, task, ld):
    """The split slice's prefix source: prefix + the reduced delta (LDS),
    rounded to bf16 and stored back (device-coherent under ``fold``)."""
    s, nb = c["S"], c["NB"]
    a = c["args"]
    vals_bf16 = c.get("vals_bf16", False)
    pre_cm = CM_DEV if c.get("fold", False) else 0
    w, q = _f4(ld["w"]), _f4(ld["q"])
    for i in range_constexpr((s * QC + THREADS - 1) // THREADS):
        live, e, t, col = _prefix_item(c, task, i)
        pre = _f4(ld["pre"][i])
        at = t * COLS + (e % QC) * 4
        if const_expr(vals_bf16):
            dl = _f4(lds_words2(c["delta_lds"], at // 2))
        else:
            dl = [fx.ptr_load(c["delta_lds"] + (at + j)) for j in range(4)]
        if const_expr(c.get("reset", False)):
            pre = dl
        else:
            pre = [bf16r(pre[j] + dl[j]) for j in range(4)]
        if live:
            bo.buffer_store(
                bf4_words(pre),
                rsrc(a["prefix"]),
                (t * HIDDEN + col) // 2,
                cache_modifier=pre_cm,
            )
        v = [live.select(pre[j], fx.Float32(0.0)) for j in range(4)]
        _slice_out(c, task, live, e, t, nb, v, w, q)
    gpu.barrier()


@flyc.jit
def stage_gate(c, t):
    """Token t: the 56 slices' partials -> source weights e_s and their sum."""
    s, ns, tid, lane, wave, red = (
        c["S"],
        c["NS"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )
    nq = ns * 2
    if tid < 8 * nq:
        g = tid // nq
        q = tid % nq
        vals = c["poll"](
            [(c["part"], (((g * GSL + k) * s + t) * ns) * 2 + q, 1) for k in range(GSL)]
        )
        fx.ptr_store(sum_polled(vals), red + (g * nq + q))
    gpu.barrier()
    if wave == 0:
        src = fx.min(lane, ns - 1)
        sq = fx.ptr_load(red + src * 2)
        dt = fx.ptr_load(red + src * 2 + 1)
        for g in range_constexpr(1, 8):
            sq = sq + fx.ptr_load(red + (g * nq + src * 2))
            dt = dt + fx.ptr_load(red + (g * nq + src * 2 + 1))
        rstd = hw_rsq(sq * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        logit = dt * rstd * fx.Float32(LOG2E)
        logit = (lane < ns).select(logit, fx.Float32(-3.0e38))
        m = wave_max(logit)
        ex = (lane < ns).select(hw_exp2(logit - m), fx.Float32(0.0))
        den = butterfly(ex, (32, 16, 8, 4, 2, 1))
        if lane < ns:
            c["put"](c["wgt"], t * (ns + 1) + lane, ex)
        if lane == 0:
            c["put"](c["wgt"], t * (ns + 1) + ns, den)
    gpu.barrier()


GATE_OFF = 512  # red words of stage_gate_local's group sums


@flyc.jit
def stage_gate_local(c):
    """Every token's source weights computed in this CTA (``stage_gate``'s sums
    in its order, so every CTA's are bit-identical): the 56 slices' partials
    polled in one round trip (no gate CTA, no WGT hand-off) -> red[t (ns + 1)
    + src], the sum at + ns."""
    s, ns, tid, lane, wave, red = (
        c["S"],
        c["NS"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )
    nq = ns * 2
    n = 8 * s * nq  # (group, token, partial) items
    per = (n + THREADS - 1) // THREADS
    specs = []
    for r in range_constexpr(per):
        item = fx.min(tid + THREADS * r, n - 1)
        g = item // (s * nq)
        t = (item // nq) % s
        q = item % nq
        specs += [
            (c["part"], (((g * GSL + k) * s + t) * ns) * 2 + q, 1) for k in range(GSL)
        ]
    got = c["poll"](specs, per * GSL)
    for r in range_constexpr(per):
        acc = f32_of(got[r * GSL][0])
        for k in range_constexpr(1, GSL):
            acc = acc + f32_of(got[r * GSL + k][0])
        if tid + THREADS * r < n:
            fx.ptr_store(acc, red + (GATE_OFF + tid + THREADS * r))
    gpu.barrier()
    # wave t: token t's weights (stage_gate's math)
    if wave < s:
        t = wave
        src = fx.min(lane, ns - 1)
        base = GATE_OFF + t * nq
        sq = fx.ptr_load(red + (base + src * 2))
        dt = fx.ptr_load(red + (base + src * 2 + 1))
        for g in range_constexpr(1, 8):
            sq = sq + fx.ptr_load(red + (base + g * s * nq + src * 2))
            dt = dt + fx.ptr_load(red + (base + g * s * nq + src * 2 + 1))
        rstd = hw_rsq(sq * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        logit = dt * rstd * fx.Float32(LOG2E)
        logit = (lane < ns).select(logit, fx.Float32(-3.0e38))
        m = wave_max(logit)
        ex = (lane < ns).select(hw_exp2(logit - m), fx.Float32(0.0))
        den = butterfly(ex, (32, 16, 8, 4, 2, 1))
        if lane < ns:
            fx.ptr_store(ex, red + (t * (ns + 1) + lane))
        if lane == 0:
            fx.ptr_store(den, red + (t * (ns + 1) + ns))
    gpu.barrier()


@flyc.jit
def stage_mix(c, task):
    """Columns 128 task ..: the softmax mix of ``vals`` (4 columns a thread), its
    sum of squares (MSQ), every slice's -> rstd, then bf16 x = mixed rstd w ->
    the plain x rows, and the slice's XRDY flag."""
    s, ns, tid, lane, red = c["S"], c["NS"], c["tid"], c["lane"], c["red"]
    a = c["args"]
    # read before the traced branches (a dict lookup inside one is carried state)
    tag_x = c.get("tag_x", False)
    vals_bf16 = c.get("vals_bf16", False)
    aux = c.get("aux")
    c0 = task * COLS
    nw = s * (ns + 1)
    if const_expr(c.get("gate_local", False)):
        stage_gate_local(c)
    else:
        if tid < nw:
            v = c["poll"]([(c["wgt"], tid, 1)])[0][0]
            fx.ptr_store(f32_of(v), red + tid)
        gpu.barrier()
    # the slice's group sums past the weights
    moff = 128 if nw <= 128 else 512
    cc = (tid % QC) * 4
    # a thread a (token, 4 columns), 16 tokens a round
    items = []
    for gi in range_constexpr((s * QC + THREADS - 1) // THREADS):
        tt = gi * (THREADS // QC) + tid // QC
        mine = tt < s
        t = fx.min(tt, s - 1)
        mixed = []
        if const_expr(vals_bf16):
            vs = [
                _f4(lds_words2(c["vals"], ((t * ns + src) * COLS + cc) // 2))
                for src in range(ns)
            ]
        for j in range_constexpr(4):
            acc = fx.Float32(0.0)
            for src in range_constexpr(ns):
                if const_expr(vals_bf16):
                    v = vs[src][j]
                else:
                    v = fx.ptr_load(c["vals"] + ((t * ns + src) * COLS + cc + j))
                acc = acc + fx.ptr_load(red + (t * (ns + 1) + src)) * v
            mixed.append(acc / fx.ptr_load(red + (t * (ns + 1) + ns)))
        sq = (
            mixed[0] * mixed[0]
            + mixed[1] * mixed[1]
            + mixed[2] * mixed[2]
            + mixed[3] * mixed[3]
        )
        sq = butterfly(sq, (16, 8, 4, 2, 1))
        if mine & (lane % 32 == 0):
            c["put"](c["msq"], task * s + t, sq)
        items.append((mine, t, mixed))
    gpu.barrier()
    if tid < 8 * s:
        g = tid // s
        tt = tid % s
        vals = c["poll"]([(c["msq"], (g * GSL + k) * s + tt, 1) for k in range(GSL)])
        fx.ptr_store(sum_polled(vals), red + (moff + g * s + tt))
    gpu.barrier()
    wn = ld_bf4(a["in_nw"], c0 + cc)
    for gi in range_constexpr(len(items)):
        mine, t, mixed = items[gi]
        tot = fx.ptr_load(red + (moff + t))
        for g in range_constexpr(1, 8):
            tot = tot + fx.ptr_load(red + (moff + g * s + t))
        r = hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["out_eps"]))
        xw = bf4_words([mixed[j] * r * wn[j] for j in range(4)])
        if mine:
            at = (t * HIDDEN + c0 + cc) // 2
            if const_expr(tag_x):
                # also as tagged pairs: the router and latent GEMVs poll these, not XRDY
                c["put_words"](c["xt"], at, [xw[0], xw[1]])
            bo.buffer_store(xw, rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
            if const_expr(c["debug"]):
                bo.buffer_store(xw, rsrc(a["x_dbg"]), at)
            if const_expr(aux is not None):
                # the drafter's aux hidden: the pre-norm mixture, row t's slot
                ptr, slot = aux
                bo.buffer_store(
                    bf4_words(mixed), rsrc(ptr), (t * AUX_ROW + slot * HIDDEN + c0 + cc) // 2
                )
    publish(c["put"], c["xrdy"], task, fx.Int32(1), tid == 0)
    gpu.barrier()


# ------------------------------------------- the wide KDA launch's attention seam
# The same seam, shaped for up to 48 rows and one hand-off fewer: a thread a
# (token, 4 columns) loads every source of its row at once, and the slices'
# partials carry the sources' Gram terms, so the gate CTA also gives each token
# the input_layernorm rstd (sum of squares of the mix = w^T G w / den^2) and the
# mix needs no MSQ round. The PART sums of squares and qk dots, the gate's
# weights and the mix are those of the stages above, bit for bit; the rstd is
# summed in another order.
def front_pairs(ns):
    """(s, s2) with s < s2: the Gram cross terms, in PART order past the 2 ns
    sums of squares and qk dots."""
    return [(a, b) for a in range(ns) for b in range(a + 1, ns)]


def front_nq(ns):
    """PART values a (slice, token)."""
    return 2 * ns + len(front_pairs(ns))


def x_packed_word(t, col, tiles):
    """The word of element (t, col) of x (col a multiple of 2) in the wide KDA
    launch's ``pack_x`` layout over ``tiles`` 16-row tiles: K step of 64, token
    tile, K half, K group of 8, row, 8 K."""
    i = (((((col // 64) * tiles + t // 16) * 2 + (col % 64) // 32) * 4 + (col % 32) // 8) * 16 + t % 16) * 8
    return (i + col % 8) // 2


@flyc.jit
def front_slice(c, task):
    """Columns 128 task ..: every source of a (token, 4 columns) loaded at once
    into LDS ``vals`` (packed bf16 pairs), per-(token, source) sum of squares and
    qk dot and per-(token, source pair) dot -> PART; a block-write layer's prefix
    stored at block ``write_idx``."""
    s, ns, nb, tid, lane = c["S"], c["NS"], c["NB"], c["tid"], c["lane"]
    a = c["args"]
    nq = front_nq(ns)
    pairs = front_pairs(ns)
    write_idx = c["write_idx"]
    cc = (tid % QC) * 4
    col = task * COLS + cc
    w = ld_bf4(a["ares_nw"], col)
    q = ld_bf4(a["ares_qk"], col)
    wq = [w[j] * q[j] for j in range(4)]
    for i in range_constexpr((s * QC + THREADS - 1) // THREADS):
        tt = i * (THREADS // QC) + tid // QC
        live = tt < s
        t = fx.min(tt, s - 1)
        raw = [
            _raw4(a["blocks"], t * a["blk_sm"] + src * a["blk_sr"] + col) for src in range(nb)
        ] + [_raw4(a["prefix"], t * HIDDEN + col)]
        if const_expr(write_idx >= 0):  # noqa: SIM102
            if live:
                bo.buffer_store(
                    raw[nb], rsrc(a["blocks"]), (t * a["blk_sm"] + write_idx * a["blk_sr"] + col) // 2
                )
        vs = [_f4(r) for r in raw]
        if live:
            for src in range_constexpr(ns):
                fx.ptr_store(raw[src], c["vals"] + ((t * ns + src) * COLS + cc) // 2)
        sums = []
        for src in range_constexpr(ns):
            v = vs[src]
            sums.append(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
            sums.append(v[0] * wq[0] + v[1] * wq[1] + v[2] * wq[2] + v[3] * wq[3])
        for x, y in pairs:
            u, v = vs[x], vs[y]
            sums.append(u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3])
        sums = [butterfly(x, (16, 8, 4, 2, 1)) for x in sums]
        if live & (lane % 32 == 0):
            at = (task * s + t) * nq
            for k in range_constexpr(nq):
                c["put"](c["part"], at + k, sums[k])
    gpu.barrier()


EX_OFF = 1024  # red words of front_gate's source weights


@flyc.jit
def front_gate(c, t):
    """Token t: the 56 slices' partials -> source weights e_s, their sum, and
    the input_layernorm rstd of the mix -> WGT (ns + 2 values)."""
    ns = c["NS"]
    wgt = c["wgt"]
    gate_sums(c, c["part"], t, front_nq(ns), True, lambda i, v: c["put"](wgt, t * (ns + 2) + i, v))


@flyc.jit
def gate_sums(c, part, t, nq, gram, emit):
    """Token t's gate from the 56 slices' partials in region ``part`` (``nq`` a
    (slice, token): each source's sum of squares and qk dot, then under
    ``gram`` the sources' Gram terms): ``emit(i, v)`` the source weights e_s
    (i < ns), their sum (i = ns) and, under ``gram``, the input_layernorm rstd
    of the mix (i = ns + 1)."""
    s, ns, tid, lane, wave, red = c["S"], c["NS"], c["tid"], c["lane"], c["wave"], c["red"]
    pairs = front_pairs(ns)
    if tid < 8 * nq:
        g = tid // nq
        q = tid % nq
        vals = c["poll"]([(part, ((g * GSL + k) * s + t) * nq + q, 1) for k in range(GSL)])
        fx.ptr_store(sum_polled(vals), red + (g * nq + q))
    gpu.barrier()
    if wave == 0:
        src = fx.min(lane, ns - 1)
        sq = fx.ptr_load(red + src * 2)
        dt = fx.ptr_load(red + src * 2 + 1)
        for g in range_constexpr(1, 8):
            sq = sq + fx.ptr_load(red + (g * nq + src * 2))
            dt = dt + fx.ptr_load(red + (g * nq + src * 2 + 1))
        rstd = hw_rsq(sq * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        logit = dt * rstd * fx.Float32(LOG2E)
        logit = (lane < ns).select(logit, fx.Float32(-3.0e38))
        m = wave_max(logit)
        ex = (lane < ns).select(hw_exp2(logit - m), fx.Float32(0.0))
        den = butterfly(ex, (32, 16, 8, 4, 2, 1))
        if const_expr(gram):
            if lane < ns:
                fx.ptr_store(ex, red + (EX_OFF + lane))
            # lane p: Gram entry p (the diagonal, then the pairs), weighted by its sources
            diag = [(x, x, 2 * x, 1.0) for x in range(ns)]
            cross = [(x, y, 2 * ns + k, 2.0) for k, (x, y) in enumerate(pairs)]
            sa, sb, gi, mult = fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Float32(0.0)
            for p, (x, y, k, f) in enumerate(diag + cross):
                hit = lane == p
                sa, sb = hit.select(fx.Int32(x), sa), hit.select(fx.Int32(y), sb)
                gi, mult = hit.select(fx.Int32(k), gi), hit.select(fx.Float32(f), mult)
            gsum = fx.ptr_load(red + gi)
            for g in range_constexpr(1, 8):
                gsum = gsum + fx.ptr_load(red + (g * nq + gi))
            term = mult * fx.ptr_load(red + (EX_OFF + sa)) * fx.ptr_load(red + (EX_OFF + sb)) * gsum
            tot = butterfly(term, (32, 16, 8, 4, 2, 1))
            r = hw_rsq(tot / (den * den) * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["out_eps"]))
        if lane < ns:
            emit(lane, ex)
        if lane == 0:
            emit(ns, den)
            if const_expr(gram):
                emit(ns + 1, r)
    gpu.barrier()


AUX_SLOTS = 5  # DSpark's captured target layers
AUX_ROW = AUX_SLOTS * HIDDEN  # bf16 a row of aux_hidden_states_buffer


@flyc.jit
def front_mix(c, task):
    """Columns 128 task ..: every token's softmax mix of ``vals`` (4 columns a
    thread, ``stage_mix``'s sums), times its rstd and the norm weight -> bf16 x
    in ``pack_x``'s layout over ``x_tiles`` tiles, then the slice's XRDY flag."""
    s, ns, tid, red = c["S"], c["NS"], c["tid"], c["red"]
    a = c["args"]
    tiles = c["x_tiles"]
    # read before the traced branches (a dict lookup inside one is carried state)
    aux = c.get("aux")
    c0 = task * COLS
    cc = (tid % QC) * 4
    nw = s * (ns + 2)
    # issued before the poll: it waits on nothing in this launch
    wn = ld_bf4(a["in_nw"], c0 + cc)
    for k in range_constexpr((nw + THREADS - 1) // THREADS):
        e = tid + THREADS * k
        if e < nw:
            v = c["poll"]([(c["wgt"], e, 1)])[0][0]
            fx.ptr_store(f32_of(v), red + e)
    gpu.barrier()
    for gi in range_constexpr((s * QC + THREADS - 1) // THREADS):
        tt = gi * (THREADS // QC) + tid // QC
        mine = tt < s
        t = fx.min(tt, s - 1)
        base = t * (ns + 2)
        vs = [
            _f4(lds_words2(c["vals"], ((t * ns + src) * COLS + cc) // 2))
            for src in range(ns)
        ]
        r = fx.ptr_load(red + (base + ns + 1))
        mixed = []
        for j in range_constexpr(4):
            acc = fx.Float32(0.0)
            for src in range_constexpr(ns):
                acc = acc + fx.ptr_load(red + (base + src)) * vs[src][j]
            mixed.append(acc / fx.ptr_load(red + (base + ns)))
        xw = bf4_words([mixed[j] * r * wn[j] for j in range(4)])
        if mine:
            bo.buffer_store(xw, rsrc(c["xbuf"]), x_packed_word(t, c0 + cc, tiles), cache_modifier=CM_DEV)
            if const_expr(aux is not None):
                # the drafter's aux hidden: the pre-norm mixture, row t's slot
                ptr, slot = aux
                bo.buffer_store(
                    bf4_words(mixed), rsrc(ptr), (t * AUX_ROW + slot * HIDDEN + c0 + cc) // 2
                )
    publish(c["put"], c["xrdy"], task, fx.Int32(1), tid == 0)
    gpu.barrier()


# ------------------------------- the next layer's attention seam, in K2's tail
# The previous layer's K2 holds the stream (its final reduce writes it) and the
# bank, so a KDA layer's attention-side seam and input_layernorm can run there,
# one hand-off after the reduce instead of four in the layer's own front: the
# reduce CTAs (task j: columns 128 j ..) slice the stream they just reduced
# (``front_slice``'s sums, the stream from LDS) -> TPART, and push the stream
# as tagged pairs; a row CTA a token gathers its 56 slices' partials and its
# stream row, mixes the whole row and norms it -> the next layer's x in
# ``pack_x``'s layout, which that layer's K1 (the wide form) reads with no
# front. Bit for bit that front: the same sums, gate (``gate_sums``) and mix.
TAIL_ROW0 = 84  # the row CTAs, past lnorm's: no K2 work after their down
# The widest step that takes a tail: a reduce CTA's slice round holds 16
# tokens (THREADS // QC), and the move is measured a gain at 12 rows only.
TAIL_MAX_ROWS = 12
ROW_SL = THREADS // QC  # slices a row CTA's round: 32 threads a slice
ROW_ROUNDS = (NA + ROW_SL - 1) // ROW_SL
WOFF = 1536  # red words of the row CTA's weights, their sum and the rstd

# the tail's device table (K2's ``tail`` argument): the next layer's AttnRes norm
# and qk, its input_layernorm and its x, int64 addresses in this order
TAIL_ARGS = ("ares_nw", "ares_qk", "in_nw", "x")


def tail_arg(c, name):
    """An address of the tail's device table, loaded where it is used."""
    return load_ptr64(c["tail_tab"], TAIL_ARGS.index(name))


def tail_loads(c, task):
    """A reduce CTA's loads for its slice that wait on nothing in this launch:
    its tokens' bank blocks and the next layer's AttnRes norm / qk words,
    issued before the final reduce's poll."""
    s, nb, tid = c["S"], c["NB"], c["tid"]
    a = c["args"]
    assert s <= THREADS // QC, "a slice's tokens in one round"
    col = task * COLS + (tid % QC) * 4
    t = fx.min(tid // QC, s - 1)
    return {
        "blk": [_raw4(a["blocks"], t * a["blk_sm"] + src * a["blk_sr"] + col) for src in range(nb)],
        "w": _raw4(tail_arg(c, "ares_nw"), col),
        "q": _raw4(tail_arg(c, "ares_qk"), col),
    }


@flyc.jit
def tail_slice(c, task, ld):
    """Columns 128 task ..: ``front_slice``'s per-(token, source) sums of squares
    and qk dots and per-(token, source pair) dots of the bank blocks and the
    stream the reduce left in LDS ``tpre`` -> TPART."""
    s, ns, tid, lane = c["S"], c["NS"], c["tid"], c["lane"]
    nq = front_nq(ns)
    gpu.barrier()  # the stream's LDS copy
    cc = (tid % QC) * 4
    tt = tid // QC
    live = tt < s
    t = fx.min(tt, s - 1)
    w, q = _f4(ld["w"]), _f4(ld["q"])
    wq = [w[j] * q[j] for j in range(4)]
    raw = ld["blk"] + [lds_words2(c["tpre"], (t * COLS + cc) // 2)]
    vs = [_f4(r) for r in raw]
    sums = []
    for src in range_constexpr(ns):
        v = vs[src]
        sums.append(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
        sums.append(v[0] * wq[0] + v[1] * wq[1] + v[2] * wq[2] + v[3] * wq[3])
    for x, y in front_pairs(ns):
        u, v = vs[x], vs[y]
        sums.append(u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3])
    sums = [butterfly(x, (16, 8, 4, 2, 1)) for x in sums]
    if live & (lane % 32 == 0):
        at = (task * s + t) * nq
        for k in range_constexpr(nq):
            c["put"](c["tpart"], at + k, sums[k])


def _row_col(c, r):
    """Round r of a row CTA's thread: (live, column)."""
    sl = r * ROW_SL + c["tid"] // QC
    return sl < NA, fx.min(sl, NA - 1) * COLS + (c["tid"] % QC) * 4


def row_loads(c, t):
    """A row CTA's loads that wait on nothing in this launch: token t's bank
    blocks and the next layer's input_layernorm words, every round."""
    nb = c["NB"]
    a = c["args"]
    in_nw = tail_arg(c, "in_nw")
    blk, wn = [], []
    for r in range(ROW_ROUNDS):
        col = _row_col(c, r)[1]
        blk.append(
            [_raw4(a["blocks"], t * a["blk_sm"] + src * a["blk_sr"] + col) for src in range(nb)]
        )
        wn.append(_raw4(in_nw, col))
    return {"blk": blk, "wn": wn}


@flyc.jit
def tail_row(c, t, ld, mark):
    """Token t of the next layer: its 56 slices' TPART -> the source weights,
    their sum and the rstd (``gate_sums``, as ``front_gate``); the stream row
    the reduce CTAs pushed; the softmax mix of every column (``front_mix``'s
    sums); bf16 x = mixed rstd w -> the next layer's x, ``pack_x``'s layout over
    ``tail.tiles`` tiles."""
    ns, red = c["NS"], c["red"]
    tiles = c["tail"].tiles
    xbuf = tail_arg(c, "x")
    # the stream row (the reduce CTAs' tagged pairs) in flight under TPART's poll
    specs = [(c["tstream"], (t * HIDDEN + _row_col(c, r)[1]) // 2, 2) for r in range(ROW_ROUNDS)]
    first = c["poll_issue"](specs)
    gate_sums(c, c["tpart"], t, front_nq(ns), True, lambda i, v: fx.ptr_store(v, red + (WOFF + i)))
    mark()
    pre = c["poll_wait"](specs, first)  # a spec's two words: _f4 reads [0], [1]
    rs = fx.ptr_load(red + (WOFF + ns + 1))
    for r in range_constexpr(ROW_ROUNDS):
        live, col = _row_col(c, r)
        vs = [_f4(b) for b in ld["blk"][r]] + [_f4(pre[r])]
        wn = _f4(ld["wn"][r])
        mixed = []
        for j in range_constexpr(4):
            acc = fx.Float32(0.0)
            for src in range_constexpr(ns):
                acc = acc + fx.ptr_load(red + (WOFF + src)) * vs[src][j]
            mixed.append(acc / fx.ptr_load(red + (WOFF + ns)))
        xw = bf4_words([mixed[j] * rs * wn[j] for j in range(4)])
        if live:
            bo.buffer_store(xw, rsrc(xbuf), x_packed_word(t, col, tiles), cache_modifier=CM_DEV)
