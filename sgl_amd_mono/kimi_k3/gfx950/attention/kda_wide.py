# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 KDA attention half of a verify step at R requests x L tokens
(S = R L rows, R >= 1): the layer input (after ``input_layernorm``) to the gated
core output, on the ReplaySSM ring. One launch, ``BLOCKS`` CTAs x ``THREADS``.

    gemv    in_proj (6288 rows in 393 groups of 16): CTAs 0 .. 136 take two
            groups, the rest one. A CTA's groups share every x load; wave w
            streams K 896 w .. 896 (w + 1); the waves' partial sums are added
            in wave order -> bf16 PROJ pairs
    kda     a task a (request, head), from CTA 137 (the one-group CTAs): its
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
    first_task,
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
    KCH,
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
PAIRED = GROUPS - BLOCKS  # 137 CTAs take two groups
CHUNKS = HIDDEN // KCH // WAVES  # 7 K chunks a wave
SCALE = HD**-0.5
# timeline points: start, gemv MFMAs done, gemv emitted, task inputs issued,
# conv inputs polled, conv done, gates + l2norm done, recurrence done, norm done
TL_POINTS = 9

assert GROUPS * ROWS == NPROJ and 0 < PAIRED <= BLOCKS

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
    return key.depth or max(1, 3 - tiles(key))


KPT = 16  # K columns a lane holds of a state row, at every width


def scratch_layout(key: KdaWideBuild) -> dict:
    return pair_layout([("proj", key.tokens * NPROJ)])


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
def _chunk(c, i):
    """Byte offset (wave-uniform) of this wave's i-th K chunk: wave w streams the
    contiguous K range w * 896 .., in an order rotated by the CTA's index (fixed
    per group, so every width reduces a row the same way)."""
    return uniform((c["wave"] * CHUNKS + (i + c["rot"]) % CHUNKS) * KCH * 2)


def _w_loads(c, g, i, live=None):
    """Chunk i of this wave, rows 16 g ..: a lane 4 x 16 B of its row (zeros,
    with no memory read, unless ``live``)."""
    lane = c["lane"]
    base = ((g * ROWS + lane % ROWS) * HIDDEN + (lane // ROWS) * 8) // 2
    nb = None if live is None else live.select(fx.Int64(NPROJ * HIDDEN * 2), fx.Int64(0))
    r = rsrc(c["args"]["w_in"], nb)
    soff = _chunk(c, i)
    return [
        fx.Vector(
            bo.buffer_load(
                r, base + 16 * q, vec_width=4, dtype=T.i32, cache_modifier=CM_NT,
                soffset_bytes=soff,
            )
        )
        for q in range(4)
    ]  # fmt: skip


def _x_loads(c, i):
    """Chunk i of this wave: every token tile's x operands (rows past S clamped)."""
    lane, s = c["lane"], c["S"]
    r = rsrc(c["args"]["x"])
    soff = _chunk(c, i)
    out = []
    for tt in range(c["T"]):
        tok = fx.min(tt * 16 + lane % 16, s - 1)
        base = (tok * HIDDEN + (lane // ROWS) * 8) // 2
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(r, base + 16 * q, vec_width=4, dtype=T.i32, soffset_bytes=soff)
                )
                for q in range(4)
            ]
        )
    return out


@flyc.jit
def stage_gemv(c, g0, g1, two):
    """Groups g0 and g1 (g1 emitted only when ``two``) -> bf16 PROJ pairs."""
    s, tid, lane, wave, nt, d = c["S"], c["tid"], c["lane"], c["wave"], c["T"], c["D"]
    groups = ((g0, None), (g1, two))
    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nt)] for _ in range(2)]
    pend = {}
    for i in range_constexpr(min(d, CHUNKS)):
        pend[i] = ([_w_loads(c, g, i, lv) for g, lv in groups], _x_loads(c, i))
    # the barriers keep each chunk's loads ``d`` chunks ahead of its MFMAs (the
    # scheduler would otherwise sink them to their use)
    rocdl.sched_barrier(0)
    for i in range_constexpr(CHUNKS):
        if const_expr(i + d < CHUNKS):
            pend[i + d] = ([_w_loads(c, g, i + d, lv) for g, lv in groups], _x_loads(c, i + d))
        rocdl.sched_barrier(0)
        ws, xs = pend.pop(i)
        for q in range_constexpr(4):
            for tt in range_constexpr(nt):
                xb = xs[tt][q].bitcast(fx.BFloat16)
                for gs in range_constexpr(2):
                    acc[gs][tt] = mfma_bf16(ws[gs][q].bitcast(fx.BFloat16), xb, acc[gs][tt])
        rocdl.sched_barrier(0)
    stamp(c["on"], c["tls"], tid, 1)
    red = c["red"]
    for gs in range_constexpr(2):
        for tt in range_constexpr(nt):
            fx.ptr_store(acc[gs][tt], red + (((gs * nt + tt) * WAVES + wave) * 64 + lane) * 4)
    lds_barrier()
    n = 2 * ROWS * s
    for it in range_constexpr((n + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * it, n - 1)
        gs = e // (ROWS * s)
        r = e % ROWS
        tok = (e // ROWS) % s
        base = (gs * nt + tok // 16) * WAVES
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(red + (((base + w) * 64 + 16 * (r // 4) + tok % 16) * 4 + r % 4))
        v = bf16r(tot)
        g = (gs == 0).select(g0, g1)
        if (tid + THREADS * it < n) & ((gs == 0) | two):
            c["put"](c["proj"], tok * NPROJ + g * ROWS + r, v)
            if const_expr(c["debug"]):
                bo.buffer_store(
                    v.to(fx.BFloat16), rsrc(c["args"]["proj_dbg"]), tok * NPROJ + g * ROWS + r
                )
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


@flyc.jit
def kda_task(c, req, h):
    """One (request, head) task."""
    L, tid, kp = c["L"], c["tid"], c["KPT"]
    lane, wave = tid % 64, tid // 64
    a = c["args"]
    fa_l, qkl, decay, vloc, ol, beta = (
        c["fa"], c["qkl"], c["decay"], c["vloc"], c["ol"], c["beta"],
    )  # fmt: skip
    slot = uniform(ld_i32(a["st_idx"], req))
    live = slot >= 0
    sl = fx.max(slot, 0)
    w_row = uniform(ld_i32(a["win_row"], req))

    # inputs that need no hand-off: f_b rows, constants, the state, conv window
    fr = h * HD + wave * 16 + lane % 16
    fbw = (fr * HD + (lane // 16) * 32) // 2
    fb = [
        fx.Vector(bo.buffer_load(rsrc(a["w_fb"]), fbw + 4 * q, vec_width=4, dtype=T.i32))
        for q in range(4)
    ]
    aa = exp_(ld_f32(a["a_log"], h))
    dtb = [ld_f32(a["dt_bias"], h * HD + wave * 16 + 4 * (lane // 16) + i) for i in range(4)]
    lpr = HD // kp  # lanes a state row
    rpt = HD * HD // (THREADS * kp)  # state rows a thread
    grp, k0 = tid // lpr, (tid % lpr) * kp
    base0 = rsrc(a["rstate"] + fx.Int64(sl) * fx.Int64(a["rs_seq"]) * 4)
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
    got = [g[0] for g in c["poll"](specs)]
    xv, fav, bv, g2 = got[:L], got[L : L + 2], got[L + 2], got[L + 3 :]
    stamp(c["on"], c["tls"], tid, 4)

    # conv: thread < 384 owns channel (part, d) of the head
    # the task's global loads are all issued by now: a load behind a store
    # would wait for its acknowledgement
    seq = win + [f32_of(xv[j]) for j in range(L)]
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
            fx.ptr_store(f32_of(fav[u]), fa_l + fa_e[u])
    b = sigmoid(f32_of(bv))
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
            y = x * rr * on_w[u] * sigmoid(f32_of(g2[u]))
            bo.buffer_store(
                y.to(fx.BFloat16), rsrc(a["core_out"]), (req * L + jn) * PROJ + h * HD + e
            )
    lds_barrier()
    stamp(c["on"], c["tls"], tid, 8)


@flyc.jit
def kda_tasks(c):
    """Every (request, head) task, task t on CTA (t + PAIRED) % BLOCKS."""
    for task in range(first_task(c["bid"], PAIRED), (c["S"] // c["L"]) * NH, BLOCKS):
        kda_task(c, task // NH, task % NH)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}



def build(key: KdaWideBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, L = key.tokens, key.qlen
    assert s % L == 0 and 1 <= L <= 8, (s, L)
    nt = tiles(key)
    lay = scratch_layout(key)

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, 2 * nt * WAVES * 64 * 4, 16]
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
            "rot": uniform(bid % CHUNKS),
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
        two = bid < PAIRED
        g0 = two.select(2 * bid, bid + PAIRED)
        g1 = two.select(2 * bid + 1, bid + PAIRED)
        stamp_begin(key.timeline, lds.tls.ptr, tid, TL_POINTS)
        stage_gemv(c, g0, g1, two)
        kda_tasks(c)
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
