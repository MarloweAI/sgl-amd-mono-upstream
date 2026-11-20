# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 KDA attention half of a verify step at R requests x L tokens
(S = R L rows, R >= 1): the layer input (after ``input_layernorm``) to the gated
core output, on the ReplaySSM ring. One launch, ``BLOCKS`` CTAs x ``THREADS``.

    gemv    in_proj: 197 pairs of 16-row groups x 14 K steps of 64 a wave
            (wave w streams K 896 w .. 896 (w + 1)), 2758 units in equal runs
            of 10 or 11 over the CTAs; the two groups share every x load. A
            run's pieces are summed in wave order; a pair's head CTA adds the
            pieces of the next one or two CTAs in order -> bf16 PROJ pairs
    kda     a task a (request, head) on CTA 128 + t (at most 10 requests): its
            state, conv window, f_b rows and constants loaded first, then
            conv + SiLU of the head's q|k|v (verify windows to SGLang's
            ``intermediate_conv_window``, k / v to the ring), q / k l2norm,
            f_b MFMA + lower-bound gate (ring g, beta), the delta-rule
            recurrence with the 128 x 128 state in registers (read only), the
            gated RMSNorm -> core_out

The conv, gate, l2norm and recurrence follow SGLang's ``kda_verify_gfx950``
operation by operation: conv products and sums rounded one at a time, its
SiLU and sigmoid forms, its l2norm and recurrence partial sums (lane order,
then the lane butterfly 1, 2, 4 (, 8)), and its state split of 16 K columns a lane
(SGLang's from three requests up; it takes 8 at one or two). The GEMMs and
the output norm reduce in an order of their own.

Batch invariant: a row's in_proj sum (MFMA shape, chunk order, wave order) and
a request's task do not depend on the width or on the other requests, so a
request's outputs and ring rows are the same bits in any batch.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.ops import (
    CM_NT,
    butterfly,
    hw_exp2,
    ld_i32,
    mfma_bf16,
    rsrc,
    uniform,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    kernel_sources,
    key_tuple,
    pair_layout,
)
from sgl_amd_mono.common.sync import Mailbox, sreg
from sgl_amd_mono.kimi_k3.gfx950.attention.kda import (
    BETA_0,
    CONV_W,
    FA_0,
    G2_0,
    NPROJ,
    ld_bf,
    ld_f32,
)
from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HD, HIDDEN, LOG2E, NH, PROJ, ROWS
from sgl_amd_mono.kimi_k3.gfx950.stages.seam import bf16r, f32_of
from sgl_amd_mono.kimi_k3.gfx950.stamps import stamp, stamp_begin, stamp_flush
from sgl_amd_mono.kimi_k3.gfx950.tag import step_tag

SOURCES = kernel_sources(__name__)

GROUPS = NPROJ // ROWS  # 393
PAIRS = (GROUPS + 1) // 2  # 197; the last pair's second group is past the rows
KSTEP = 64  # K a wave's step
HSTEPS = HIDDEN // KSTEP // WAVES  # 14 steps a wave a pair
UNITS = PAIRS * HSTEPS  # 2758 (pair, step) units
NMAX = (UNITS + BLOCKS - 1) // BLOCKS  # 11: a CTA's run is 10 or 11 units
TASK0 = 128  # the first KDA task's CTA
SCALE = HD**-0.5
# timeline points: start, gemv MFMAs done (every wave), gemv emitted, task inputs issued,
# conv inputs polled, conv done, gates + l2norm done, recurrence done, norm done
TL_POINTS = 9

assert GROUPS * ROWS == NPROJ and NMAX < HSTEPS  # a pair spans at most 3 CTAs

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class KdaWideBuild:
    tokens: int  # S
    qlen: int  # L, tokens a request
    onorm_eps: float = 1e-5
    lower_bound: float = -5.0
    depth: int = 0  # K chunks of loads in flight ahead of the MFMAs (0: by width)
    debug: bool = False  # also the bf16 in_proj rows and the pre-norm core
    timeline: bool = False  # per-CTA phase stamps (``stamps.py``)


def tiles(key):
    return (key.tokens + 15) // 16


def depth(key):
    return key.depth or (5, 3, 2)[tiles(key) - 1]


KPT = 16  # K columns a lane holds of a state row, at every width


def scratch_layout(key: KdaWideBuild) -> dict:
    return pair_layout(
        [("proj", key.tokens * NPROJ), ("part", BLOCKS * 2 * ROWS * key.tokens)]
    )


def scratch_bytes(key: KdaWideBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def lds_barrier():
    """``s_barrier`` for an LDS hand-off between waves: fenced, so no LDS access
    moves across it, and the waves' global loads and stores stay in flight (a
    plain barrier waits for every store's acknowledgement)."""
    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="workgroup")
    rocdl.s_barrier()
    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="workgroup")


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def exp_(z):
    return hw_exp2(z * fx.Float32(LOG2E))


# ---------------------------------------------------------------- gemv
def _u0(c_idx):
    """First unit of CTA ``c_idx`` (traced): CTAs take equal runs of units."""
    return UNITS * c_idx // BLOCKS


def _soff(c, hs):
    """Byte offset (wave-uniform) of K step ``hs``: wave w streams K 896 w .."""
    return uniform((c["wave"] * HSTEPS + hs) * KSTEP * 2)


def _loads(c, p, hs, live):
    """One step of this wave: rows 16 (2 p + g) .. of both groups and every token
    tile's x, 2 x 16 B a lane each (zeros, with no memory read, unless ``live``;
    the rows past in_proj's last are zeros too)."""
    lane, s = c["lane"], c["S"]
    soff = _soff(c, hs)
    wr = rsrc(c["args"]["w_in"], live.select(fx.Int64(NPROJ * HIDDEN * 2), fx.Int64(0)))
    xr = rsrc(c["args"]["x"], live.select(fx.Int64(s * HIDDEN * 2), fx.Int64(0)))
    k8 = (lane // ROWS) * 8

    def ld(r, row, cm):
        base = (row * HIDDEN + k8) // 2
        return [
            fx.Vector(
                bo.buffer_load(
                    r, base + 16 * q, vec_width=4, dtype=T.i32, cache_modifier=cm,
                    soffset_bytes=soff,
                )
            )
            for q in range(KSTEP // 32)
        ]  # fmt: skip

    ws = [ld(wr, (2 * p + g) * ROWS + lane % ROWS, CM_NT) for g in range(2)]
    xs = [ld(xr, fx.min(tt * 16 + lane % 16, s - 1), 0) for tt in range(c["T"])]
    return ws, xs


def _slot_w(c):
    return 2 * c["T"] * WAVES * 64 * 4  # LDS floats a piece


@flyc.jit
def gemv_loop(c):
    """This CTA's run of units (pair of groups, K step): the waves' partial
    sums of its one or two pieces -> LDS."""
    tid, lane, wave, nt, d, bid = c["tid"], c["lane"], c["wave"], c["T"], c["D"], c["bid"]
    u0, u1 = uniform(_u0(bid)), uniform(_u0(bid + 1))
    red = c["red"]
    slot_w = _slot_w(c)

    def unit(i):
        u = u0 + i
        return u // HSTEPS, u % HSTEPS, u < u1

    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nt)] for _ in range(2)]

    def spill(slot):
        for gs in range_constexpr(2):
            for tt in range_constexpr(nt):
                fx.ptr_store(
                    acc[gs][tt],
                    red + slot * slot_w + (((gs * nt + tt) * WAVES + wave) * 64 + lane) * 4,
                )

    pend = {}
    for i in range_constexpr(min(d, NMAX)):
        p, hs, live = unit(i)
        pend[i] = _loads(c, p, hs, live)
    # the barriers keep each step's loads ``d`` steps ahead of its MFMAs (the
    # scheduler would otherwise sink them to their use)
    rocdl.sched_barrier(0)
    for i in range_constexpr(NMAX):
        if const_expr(i + d < NMAX):
            p, hs, live = unit(i + d)
            pend[i + d] = _loads(c, p, hs, live)
        rocdl.sched_barrier(0)
        ws, xs = pend.pop(i)
        if const_expr(i > 0):
            # the run crosses into the next pair here: the first piece is done
            p, hs, live = unit(i)
            cross = (hs == 0) & live
            if cross:
                spill(0)
            for gs in range_constexpr(2):
                for tt in range_constexpr(nt):
                    acc[gs][tt] = cross.select(fx.Vector.filled(4, 0.0, fx.Float32), acc[gs][tt])
        for q in range_constexpr(KSTEP // 32):
            for tt in range_constexpr(nt):
                xb = xs[tt][q].bitcast(fx.BFloat16)
                for gs in range_constexpr(2):
                    acc[gs][tt] = mfma_bf16(ws[gs][q].bitcast(fx.BFloat16), xb, acc[gs][tt])
        rocdl.sched_barrier(0)
    spill(1)
    lds_barrier()
    stamp(c["on"], c["tls"], tid, 1)


@flyc.jit
def gemv_emit(c):
    """The run's pieces: a piece that starts its pair's K is the pair's head,
    and its CTA adds the pair's later pieces (published by the next one or two
    CTAs) in order -> bf16 PROJ pairs. The split depends on the CTA alone,
    never on the width."""
    s, tid, nt, bid = c["S"], c["tid"], c["T"], c["bid"]
    u0, u1 = uniform(_u0(bid)), uniform(_u0(bid + 1))
    red = c["red"]
    slot_w = _slot_w(c)
    pa, hsa = u0 // HSTEPS, u0 % HSTEPS
    crosses = (u1 - 1) // HSTEPS != pa
    # slot 0: the first piece when the run crosses; slot 1: the last piece
    n = 2 * ROWS * s
    per = (n + THREADS - 1) // THREADS

    def piece_sum(slot, e):
        gs, r, tok = e // (ROWS * s), e % ROWS, (e // ROWS) % s
        base = slot * (slot_w // 256) + (gs * nt + tok // 16) * WAVES
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(red + (((base + w) * 64 + 16 * (r // 4) + tok % 16) * 4 + r % 4))
        return tot

    es = [fx.min(tid + THREADS * it, n - 1) for it in range(per)]
    # a piece that is not its pair's head: published for the head's CTA
    pub_slot = crosses.select(0, 1)
    if crosses | (hsa != 0):
        for it in range_constexpr(per):
            v = piece_sum(pub_slot, es[it])
            if tid + THREADS * it < n:
                c["put"](c["part"], bid * n + es[it], v)
    # the head this CTA owns (the last piece), and its pair's later pieces
    ph = crosses.select(pa + 1, pa)
    if crosses | (hsa == 0):
        nb1, nb2 = fx.min(bid + 1, BLOCKS - 1), fx.min(bid + 2, BLOCKS - 1)
        two = (bid + 2 < BLOCKS) & (_u0(bid + 2) < (ph + 1) * HSTEPS)
        got1 = c["poll"]([(c["part"], nb1 * n + e, 1) for e in es])
        got2 = c["poll"]([(c["part"], two.select(nb2, nb1) * n + e, 1) for e in es])
        for it in range_constexpr(per):
            e = es[it]
            tot = piece_sum(1, e) + f32_of(got1[it][0])
            tot = two.select(tot + f32_of(got2[it][0]), tot)
            v = bf16r(tot)
            gs, r, tok = e // (ROWS * s), e % ROWS, (e // ROWS) % s
            row = (2 * ph + gs) * ROWS + r
            if (tid + THREADS * it < n) & (row < NPROJ):
                c["put"](c["proj"], tok * NPROJ + row, v)
                if const_expr(c["debug"]):
                    bo.buffer_store(v.to(fx.BFloat16), rsrc(c["args"]["proj_dbg"]), tok * NPROJ + row)
    lds_barrier()
    stamp(c["on"], c["tls"], tid, 2)


# ---------------------------------------------------------------- kda
def _conv_silu(seq, cw, j):
    """SGLang's bias-free conv of token j (products and sums rounded one by one,
    from 0), its SiLU, rounded to bf16."""
    acc = fx.Float32(0.0)
    for q in range(CONV_W):
        acc = acc + cw[q] * seq[j + q]
    return bf16r(acc / (fx.Float32(1.0) + exp_(-acc)))


def _dpp_add(v, ctrl):
    moved = fx.Int32(update_dpp_i32(fx.Int32(0), v.bitcast(fx.Int32), ctrl, 0xF, 0xF, False))
    return v + moved.bitcast(fx.Float32)


def group_sum(v, lanes):
    """SGLang's sum over aligned groups of ``lanes`` (8 or 16) lanes of a DPP
    row: quad_perm [1, 0, 3, 2], [2, 3, 0, 1], row_half_mirror (, row_mirror)."""
    v = _dpp_add(_dpp_add(v, 0xB1), 0x4E)
    v = _dpp_add(v, 0x141)
    return _dpp_add(v, 0x140) if lanes == 16 else v


def kda_pre(c, req, h, has):
    """The (request, head) task's inputs that need no hand-off, issued under
    the CTA's in_proj emit: f_b rows, constants, the state, the conv window
    (no memory read unless ``has``)."""
    L, tid, kp = c["L"], c["tid"], c["KPT"]
    lane, wave = tid % 64, tid // 64
    a = c["args"]
    fa_l, qkl, decay, vloc, ol, beta = (
        c["fa"], c["qkl"], c["decay"], c["vloc"], c["ol"], c["beta"],
    )  # fmt: skip
    slot = has.select(uniform(ld_i32(a["st_idx"], req)), fx.Int32(-1))
    live = slot >= 0
    sl = fx.max(slot, 0)
    w_row = uniform(ld_i32(a["win_row"], req))

    fr = h * HD + wave * 16 + lane % 16
    fbw = (fr * HD + (lane // 16) * 32) // 2
    fb = [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["w_fb"], has.select(fx.Int64(PROJ * HD * 2), fx.Int64(0))),
                fbw + 4 * q, vec_width=4, dtype=T.i32,
            )
        )
        for q in range(4)
    ]
    aa = exp_(ld_f32(a["a_log"], h))
    dtb = [ld_f32(a["dt_bias"], h * HD + wave * 16 + 4 * (lane // 16) + i) for i in range(4)]
    lpr = HD // kp  # lanes a state row
    rpt = HD * HD // (THREADS * kp)  # state rows a thread
    grp, k0 = tid // lpr, (tid % lpr) * kp
    base0 = rsrc(
        a["rstate"] + fx.Int64(sl) * fx.Int64(a["rs_seq"]) * 4,
        live.select(fx.Int64(NH * HD * HD * 4), fx.Int64(0)),
    )
    st = []
    for i in range_constexpr(rpt):
        row = []
        for q in range_constexpr(kp // 4):
            v = fx.Vector(
                bo.buffer_load(
                    base0, (h * HD + grp * rpt + i) * HD + k0 + 4 * q, vec_width=4, dtype=T.f32
                )
            )
            row += [live.select(v[e], fx.Float32(0.0)) for e in range(4)]
        st.append(row)
    ct = fx.min(tid, 3 * HD - 1)
    part, d = ct // HD, ct % HD
    chan = part * PROJ + h * HD + d
    cs = a["conv_st"] + fx.Int64(sl) * fx.Int64(a["cs_seq"]) * 2
    co = chan * a["cs_dim"]
    cw = fx.Vector(bo.buffer_load(rsrc(a["conv_w"]), chan * CONV_W, vec_width=4, dtype=T.f32))
    win = [ld_bf(cs, co + j * a["cs_tok"]) for j in range(CONV_W - 1)]
    on_w = [ld_bf(a["on_w"], lane + 64 * u) for u in range(2)]
    stamp(c["on"], c["tls"], tid, 3)
    return dict(
        slot=slot, live=live, sl=sl, w_row=w_row, fb=fb, aa=aa, dtb=dtb, st=st,
        part=part, d=d, chan=chan, cw=cw, win=win, on_w=on_w,
    )  # fmt: skip


@flyc.jit
def kda_task(c, req, h, k):
    """One (request, head) task."""
    L, tid, kp = c["L"], c["tid"], c["KPT"]
    lane, wave = tid % 64, tid // 64
    a = c["args"]
    fa_l, qkl, decay, vloc, ol, beta = (
        c["fa"], c["qkl"], c["decay"], c["vloc"], c["ol"], c["beta"],
    )  # fmt: skip
    live, sl, w_row, fb, aa, dtb, st = (
        k["live"], k["sl"], k["w_row"], k["fb"], k["aa"], k["dtb"], list(k["st"]),
    )  # fmt: skip
    part, d, chan, cw, win, on_w = k["part"], k["d"], k["chan"], k["cw"], k["win"], k["on_w"]
    lpr = HD // kp  # lanes a state row
    rpt = HD * HD // (THREADS * kp)  # state rows a thread
    grp, k0 = tid // lpr, (tid % lpr) * kp

    # every in_proj value the task reads, one poll batch a thread: its conv
    # channel's L inputs, two of f_a [L][128], beta of token tid, two of its
    # wave's token's output gate (clamped indices poll a pair that is written)
    row0 = req * L * NPROJ
    n_fa = L * HD
    assert n_fa <= 2 * THREADS
    fa_e = [fx.min(tid + THREADS * u, n_fa - 1) for u in range(2)]
    jn = fx.min(wave, L - 1)
    specs = [(c["proj"], row0 + j * NPROJ + chan, 1) for j in range(L)]
    specs += [(c["proj"], row0 + (e // HD) * NPROJ + FA_0 + e % HD, 1) for e in fa_e]
    specs += [(c["proj"], row0 + fx.min(tid, L - 1) * NPROJ + BETA_0 + h, 1)]
    specs += [
        (c["proj"], row0 + jn * NPROJ + G2_0 + h * HD + lane + 64 * u, 1) for u in range(2)
    ]
    got = [f32_of(g[0]) for g in c["poll"](specs)]
    xv, fav, bv, g2 = got[:L], got[L : L + 2], got[L + 2], got[L + 3 :]
    stamp(c["on"], c["tls"], tid, 4)

    # conv: thread < 384 owns channel (part, d) of the head
    # the task's global loads are all issued by now: a load behind a store
    # would wait for its acknowledgement
    seq = win + list(xv)
    outs = [_conv_silu(seq, cw, j) for j in range(L)]
    ring_len = a["ring_len"]
    if live & (tid < 3 * HD):
        if w_row >= 0:
            # token j's window: the CONV_W - 1 inputs after its first
            wb = rsrc(a["conv_win"] + fx.Int64(fx.max(w_row, 0)) * fx.Int64(a["cw_seq"]) * 2)
            for j in range_constexpr(L):
                for q in range_constexpr(CONV_W - 1):
                    bo.buffer_store(
                        fx.Float32(seq[j + 1 + q]).to(fx.BFloat16),
                        wb,
                        j * a["cw_tok"] + q * a["cw_q"] + chan,
                    )
        for u, (rk, seq_) in ((1, ("ring_k", "rk_seq")), (2, ("ring_v", "rv_seq"))):
            if part == u:
                rres = rsrc(a[rk] + fx.Int64(sl) * fx.Int64(a[seq_]) * 2)
                for j in range_constexpr(L):
                    bo.buffer_store(outs[j].to(fx.BFloat16), rres, (h * ring_len + j) * HD + d)
    if tid < 3 * HD:
        for j in range_constexpr(L):
            if part < 2:
                fx.ptr_store(outs[j], qkl + ((2 * j + part) * HD + d))
            if part == 2:
                fx.ptr_store(outs[j], vloc + (j * HD + d))
    for u in range_constexpr(2):
        if tid + THREADS * u < n_fa:
            fx.ptr_store(fav[u], fa_l + fa_e[u])
    b = sigmoid(bv)
    if tid < L:
        fx.ptr_store(b, beta + tid)
        if live:
            bo.buffer_store(
                b, rsrc(a["ring_b"] + fx.Int64(sl) * fx.Int64(a["rb_seq"]) * 4), h * ring_len + tid
            )
    lds_barrier()
    stamp(c["on"], c["tls"], tid, 5)
    # gate rows 16 wave .. x tokens: A = f_b rows, B = f_a [token][K]
    t = fx.min(lane % 16, L - 1)
    gacc = fx.Vector.filled(4, 0.0, fx.Float32)
    for q in range_constexpr(4):
        kb = (lane // 16) * 32 + 8 * q
        xb = fx.Vector.from_elements(
            [fx.ptr_load(fa_l + (t * HD + kb + e)) for e in range(8)], fx.Float32
        ).to(fx.BFloat16)
        gacc = mfma_bf16(fb[q].bitcast(fx.BFloat16), xb, gacc)
    rg = rsrc(a["ring_g"] + fx.Int64(sl) * fx.Int64(a["rg_seq"]) * 4)
    for i in range_constexpr(4):
        kk = wave * 16 + 4 * (lane // 16) + i
        gate = fx.Float32(c["lb"]) * sigmoid(aa * (bf16r(gacc[i]) + dtb[i]))
        if lane % 16 < L:
            fx.ptr_store(exp_(gate), decay + (t * HD + kk))
            if live:
                bo.buffer_store(gate, rg, (h * ring_len + t) * HD + kk)
    # l2norm: (token, q / k) row r = tid / 16 of the 2 L rows, 8 channels a lane
    if tid < 2 * L * 16:
        r = tid // 16
        c8 = (tid % 16) * 8
        ys = [fx.ptr_load(qkl + (r * HD + c8 + e)) for e in range(8)]
        ss = fx.Float32(0.0)
        for e in range_constexpr(8):
            ss = fmath.fma(ys[e], ys[e], ss)
        ss = group_sum(ss, 16)
        rn = fx.Float32(1.0) / fx.Float32(fmath.sqrt(ss + fx.Float32(1e-6)))
        rn = (r % 2 == 0).select(rn * fx.Float32(SCALE), rn)
        for e in range_constexpr(8):
            fx.ptr_store(ys[e] * rn, qkl + (r * HD + c8 + e))
    lds_barrier()
    stamp(c["on"], c["tls"], tid, 6)

    # recurrence: rows grp * rpt .., K columns k0 .. k0 + kp
    def lds_vec(at):
        return [
            x
            for q in range(kp // 4)
            for x in fx.Vector(
                fx.ptr_load(at + 4 * q, result_type=fx.Vector.make_type(4, fx.Float32))
            )
        ]

    def dot2(x, y):
        """SGLang's lane sum: even and odd columns apart, then added."""
        u0, u1 = fx.Float32(0.0), fx.Float32(0.0)
        for e in range_constexpr(0, kp, 2):
            u0 = fmath.fma(x[e], y[e], u0)
            u1 = fmath.fma(x[e + 1], y[e + 1], u1)
        return u0 + u1

    os_ = []
    for j in range_constexpr(L):
        qv = lds_vec(qkl + ((2 * j) * HD + k0))
        kv = lds_vec(qkl + ((2 * j + 1) * HD + k0))
        dk = lds_vec(decay + (j * HD + k0))
        bj = fx.ptr_load(beta + j)
        vs = [fx.ptr_load(vloc + (j * HD + grp * rpt + i)) for i in range(rpt)]
        for i in range_constexpr(rpt):
            s_ = [st[i][e] * dk[e] for e in range(kp)]
            u = group_sum(dot2(s_, kv), lpr)
            vn = (vs[i] - u) * bj
            s_ = [fmath.fma(kv[e], vn, s_[e]) for e in range(kp)]
            os_.append(live.select(bf16r(group_sum(dot2(s_, qv), lpr)), fx.Float32(0.0)))
            st[i] = s_
    if tid % lpr == 0:
        for j in range_constexpr(L):
            for i in range_constexpr(rpt):
                fx.ptr_store(os_[j * rpt + i], ol + (j * HD + grp * rpt + i))
    lds_barrier()
    stamp(c["on"], c["tls"], tid, 7)
    if const_expr(c["debug"]):
        for i in range_constexpr((L * HD + THREADS - 1) // THREADS):
            e = fx.min(tid + THREADS * i, L * HD - 1)
            if tid + THREADS * i < L * HD:
                bo.buffer_store(
                    fx.ptr_load(ol + e).to(fx.BFloat16),
                    rsrc(a["o_dbg"]),
                    (req * L + e // HD) * PROJ + h * HD + e % HD,
                )

    # gated RMSNorm: wave j = token j
    assert L <= WAVES
    x0 = fx.ptr_load(ol + (jn * HD + lane))
    x1 = fx.ptr_load(ol + (jn * HD + 64 + lane))
    ss = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
    rr = fx.Float32(1.0) / fx.Float32(
        fmath.sqrt(ss * fx.Float32(1.0 / HD) + fx.Float32(c["onorm_eps"]))
    )
    if wave < L:
        for u, x in ((0, x0), (1, x1)):
            e = lane + 64 * u
            y = x * rr * on_w[u] * sigmoid(g2[u])
            bo.buffer_store(
                y.to(fx.BFloat16), rsrc(a["core_out"]), (req * L + jn) * PROJ + h * HD + e
            )
    lds_barrier()
    stamp(c["on"], c["tls"], tid, 8)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}



def build(key: KdaWideBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, L = key.tokens, key.qlen
    assert s % L == 0 and 1 <= L <= 8, (s, L)
    ntasks = s // L * NH
    assert ntasks <= BLOCKS - TASK0, "one KDA task a CTA"
    nt = tiles(key)
    lay = scratch_layout(key)

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, 2 * 2 * nt * WAVES * 64 * 4, 16]
        fa: fx.Array[fx.Float32, L * HD, 16]
        qkl: fx.Array[fx.Float32, 2 * L * HD, 16]
        decay: fx.Array[fx.Float32, L * HD, 16]
        vloc: fx.Array[fx.Float32, L * HD, 16]
        ol: fx.Array[fx.Float32, L * HD, 16]
        beta: fx.Array[fx.Float32, 16, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, SOURCES)
    name = f"k3_mono_kda_wide_s{s}_l{L}_d{depth(key)}_g{int(key.debug)}_t{int(key.timeline)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kda_wide(
        x: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        st_idx: Int64,
        win_row: Int64,
        conv_win: Int64,
        ring_k: Int64,
        ring_v: Int64,
        ring_g: Int64,
        ring_b: Int64,
        core_out: Int64,
        proj_dbg: Int64,
        o_dbg: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        rs_seq: Int32,
        cw_seq: Int32,
        cw_tok: Int32,
        cw_q: Int32,
        rk_seq: Int32,
        rv_seq: Int32,
        rg_seq: Int32,
        rb_seq: Int32,
        ring_len: Int32,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        c = {
            "S": s,
            "L": L,
            "T": nt,
            "D": depth(key),
            "KPT": KPT,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": uniform(tid // 64),
            "debug": key.debug,
            "lb": key.lower_bound,
            "onorm_eps": key.onorm_eps,
            "put": mb.put,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "fa": lds.fa.ptr,
            "qkl": lds.qkl.ptr,
            "decay": lds.decay.ptr,
            "vloc": lds.vloc.ptr,
            "ol": lds.ol.ptr,
            "beta": lds.beta.ptr,
            "on": key.timeline,
            "tls": lds.tls.ptr,
            "proj": sreg(scratch, lay["proj"][0], "proj"),
            "part": sreg(scratch, lay["part"][0], "part"),
            "args": {
                "x": x,
                "w_in": w_in,
                "w_fb": w_fb,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "on_w": on_w,
                "rstate": rstate,
                "st_idx": st_idx,
                "win_row": win_row,
                "conv_win": conv_win,
                "ring_k": ring_k,
                "ring_v": ring_v,
                "ring_g": ring_g,
                "ring_b": ring_b,
                "core_out": core_out,
                "proj_dbg": proj_dbg,
                "o_dbg": o_dbg,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "rs_seq": rs_seq,
                "cw_seq": cw_seq,
                "cw_tok": cw_tok,
                "cw_q": cw_q,
                "rk_seq": rk_seq,
                "rv_seq": rv_seq,
                "rg_seq": rg_seq,
                "rb_seq": rb_seq,
                "ring_len": ring_len,
            },
        }
        stamp_begin(key.timeline, lds.tls.ptr, tid, TL_POINTS)
        gemv_loop(c)
        # task t = (request, head) on CTA TASK0 + t, its inputs in flight under the emit
        t = bid - TASK0
        has = (t >= 0) & (t < ntasks)
        tc = fx.max(fx.min(t, ntasks - 1), 0)
        k = kda_pre(c, tc // NH, tc % NH, has)
        gemv_emit(c)
        if has:
            kda_task(c, tc // NH, tc % NH, k)
        stamp_flush(key.timeline, lds.tls.ptr, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        x: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        st_idx: Int64,
        win_row: Int64,
        conv_win: Int64,
        ring_k: Int64,
        ring_v: Int64,
        ring_g: Int64,
        ring_b: Int64,
        core_out: Int64,
        proj_dbg: Int64,
        o_dbg: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        rs_seq: Int32,
        cw_seq: Int32,
        cw_tok: Int32,
        cw_q: Int32,
        rk_seq: Int32,
        rv_seq: Int32,
        rg_seq: Int32,
        rb_seq: Int32,
        ring_len: Int32,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kda_wide(
            x, w_in, w_fb, conv_w, conv_st, a_log, dt_bias, on_w, rstate, st_idx,
            win_row, conv_win, ring_k, ring_v, ring_g, ring_b, core_out, proj_dbg,
            o_dbg, cs_seq, cs_dim, cs_tok, rs_seq, cw_seq, cw_tok, cw_q, rk_seq,
            rv_seq, rg_seq, rb_seq, ring_len, scratch, layer, epoch, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    _BUILDS[key] = launch
    return launch


def ptr(t):
    return 0 if t is None else t.data_ptr()


def kda_wide(
    key: KdaWideBuild,
    *,
    x,
    w_in,
    w_fb,
    conv_w,
    conv_state,
    a_log,
    dt_bias,
    on_w,
    rstate,
    st_idx,
    conv_win,
    win_row,
    ring,
    core_out,
    scratch,
    layer,
    epoch,
    proj_dbg=None,
    o_dbg=None,
    tl=None,
):
    """Host launcher. ``x`` bf16 [S, 7168] (the input-normed layer input);
    ``conv_state`` [slot, channel, entry] by its strides (SGLang's conv cache
    [slot, CONV_W - 1, C] is its ``.mT``), read only; ``rstate`` [slots, 12, 128,
    128] fp32, read only; ``st_idx`` int32 [R]: each request's slot (< 0: none);
    ``conv_win`` [row, >= L, CONV_W - 1, C] gets request r's verify windows at
    ``win_row[r]`` (int32 [R], < 0: none); ``ring`` = (k, v, g, beta) [slots, 12,
    ring_len, (128)]; ``scratch`` (``scratch_bytes``) zeroed once; ``epoch`` a
    device int32 bumped once a step, ``layer`` this launch's slot."""
    s, L = key.tokens, key.qlen
    R = s // L
    assert x.shape == (s, HIDDEN) and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert w_in.shape == (NPROJ, HIDDEN) and w_in.is_contiguous()
    assert rstate.dtype == torch.float32 and rstate[0].numel() == NH * HD * HD
    assert rstate[0].is_contiguous()
    assert conv_state.dtype == torch.bfloat16 and conv_state.shape[2] >= CONV_W - 1
    assert conv_win.dim() == 4 and conv_win.stride(3) == 1 and conv_win.shape[1] >= L
    assert st_idx.dtype == win_row.dtype == torch.int32
    assert st_idx.numel() == win_row.numel() == R
    assert scratch.numel() >= scratch_bytes(key)
    rk, rv, rg, rb = ring
    ring_len = rk.shape[2]
    assert ring_len >= L
    assert rk.dtype == rv.dtype == torch.bfloat16 and rg.dtype == rb.dtype == torch.float32
    for t in (rk, rv, rg):
        assert t.shape[1:] == (NH, ring_len, HD) and t[0].is_contiguous()
    assert rb.shape[1:] == (NH, ring_len) and rb[0].is_contiguous()
    assert core_out.shape == (s, PROJ) and core_out.is_contiguous()
    assert key.debug == (proj_dbg is not None) == (o_dbg is not None)
    assert key.timeline == (tl is not None)
    f = build(key)
    f(
        x.data_ptr(),
        w_in.data_ptr(),
        w_fb.data_ptr(),
        conv_w.data_ptr(),
        conv_state.data_ptr(),
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        on_w.data_ptr(),
        rstate.data_ptr(),
        st_idx.data_ptr(),
        win_row.data_ptr(),
        conv_win.data_ptr(),
        rk.data_ptr(),
        rv.data_ptr(),
        rg.data_ptr(),
        rb.data_ptr(),
        core_out.data_ptr(),
        ptr(proj_dbg),
        ptr(o_dbg),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        rstate.stride(0),
        conv_win.stride(0),
        conv_win.stride(1),
        conv_win.stride(2),
        rk.stride(0),
        rv.stride(0),
        rg.stride(0),
        rb.stride(0),
        ring_len,
        scratch.data_ptr(),
        layer,
        epoch.data_ptr(),
        ptr(tl),
        stream=torch.cuda.current_stream(),
    )
