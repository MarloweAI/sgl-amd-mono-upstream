# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 KDA attention half of a verify step at R requests x L tokens
(S = R L rows, R >= 1): the layer input (after ``input_layernorm``) to the gated
core output, on the ReplaySSM ring. One launch, ``BLOCKS`` CTAs x ``THREADS``.

    front   (``front`` builds) the attention-side AttnRes seam and
            ``input_layernorm`` on CTAs 0 .. NA - 1, as K1 runs them (slice,
            gate, mix: ``stages/seam.py``), x written in ``pack_x``'s layout
            with an XRDY flag a slice; every CTA issues its first weight loads,
            then waits for the flags before its first x load
    gemv    in_proj: 131 sets of three 16-row groups x 14 K steps of 64 a
            wave (wave w streams K 896 w .. 896 (w + 1)), 1834 units in equal
            runs of 7 or 8 over the CTAs, the weights read in ``pack_w_in``'s
            layout (1 KB an instruction) and x in ``pack_x``'s (1 KB an
            instruction); a set's groups share every x load. A run's pieces
            are summed in wave order; a set's head CTA adds the pieces of the
            next one or two CTAs in order -> bf16 PROJ words (16-B tagged
            words, a request's L rows of a channel)
    kda     a task a (request, head), in two halves on CTAs TASK0 + 16 (t // 8)
            + t % 8 and 8 past it (at most 8 requests), each with 64 rows of
            the state: its state rows, conv window, f_b rows and constants
            loaded first, then conv + SiLU of the head's q|k|v (verify windows
            to SGLang's ``intermediate_conv_window``, k / v to the ring), q / k
            l2norm, f_b MFMA + lower-bound gate (ring g, beta), the delta-rule
            recurrence of its rows with the state in registers (read only), the
            halves' outputs swapped (16-B tagged words), the gated RMSNorm of
            its 64 channels -> core_out

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
from typing import Any, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_NT,
    bf2_f32,
    bf16_pair,
    butterfly,
    hw_exp2,
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
from sgl_amd_mono.common.sync import _SCOPE_CM, Mailbox, _spin, sreg
from sgl_amd_mono.kimi_k3.gfx950.attention import kda as _kda
from sgl_amd_mono.kimi_k3.gfx950.attention.kda import (
    BETA_0,
    CONV_W,
    FA_0,
    G2_0,
    NPROJ,
    ld_f32,
)
from sgl_amd_mono.kimi_k3.gfx950.stages.dims import COLS, HD, HIDDEN, LOG2E, NA, NH, PROJ, ROWS
from sgl_amd_mono.kimi_k3.gfx950.stages.seam import (
    AUX_ROW,
    bf16r,
    f32_of,
    front_gate,
    front_mix,
    front_nq,
    front_slice,
)
from sgl_amd_mono.kimi_k3.gfx950.stamps import stamp, stamp_begin, stamp_flush
from sgl_amd_mono.kimi_k3.gfx950.tag import step_tag

SOURCES = kernel_sources(__name__)

GROUPS = NPROJ // ROWS  # 393
G = 3  # groups a unit: their rows share every x load
SETS = GROUPS // G  # 131 sets of three groups
KSTEP = 64  # K a wave's step
HSTEPS = HIDDEN // KSTEP // WAVES  # 14 steps a wave a set
UNITS = SETS * HSTEPS  # 1834 (set, step) units
NMAX = (UNITS + BLOCKS - 1) // BLOCKS  # 8: a CTA's run is 7 or 8 units
WT = 6  # tokens a hand-off word: a request's verify rows (L <= WT)
TASK0 = 64  # the first KDA task CTA: a task's two halves on CTAs b, b + 8 (one XCD)
HALF = HD // 2  # state rows a task half
OXW = WT * HALF // 3  # 16-B words of a half's outputs
SCALE = HD**-0.5
# timeline points: start, gemv MFMAs done (every wave), gemv emitted, task inputs issued,
# conv inputs polled, conv done, gates + l2norm done, recurrence done, norm done, the run's
# non-head piece published (thread 0); the front's slice, gate and mix done, x ready
# (the GEMM form, which has no front: the last fold's sums in LDS, its pieces
# published, the head's poll returned, the first task round done)
TL_POINTS = 14

# a set spans at most 3 CTAs; a run at most 2 sets
assert GROUPS * ROWS == NPROJ and SETS * G == GROUPS
assert NMAX < HSTEPS <= 2 * (UNITS // BLOCKS)

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
    # the AttnRes seam and input_layernorm in front: x from the prefix and the
    # bank's ``nblocks`` blocks (a block-write layer stores the prefix at block
    # ``write_idx``) instead of given
    front: bool = False
    nblocks: int = 0
    eps: float = 1e-5  # AttnRes norm
    out_eps: float = 1e-5  # input_layernorm
    write_idx: int = -1
    # the drafter's aux slot j (0 .. 4) at a DSpark capture layer's next one: the
    # seam's pre-norm mixture also into aux[row, j * 7168 ..] (bf16)
    aux_slot: int = -1


def tiles(key):
    return (key.tokens + 15) // 16


# Past WIDE_T token tiles the launch is its GEMM form (``gemm_pass``): the same
# units on the same CTAs over every token tile, PASS_T tiles a pass (their
# accumulators in registers; a pass streams the CTA's run), the pieces folded
# CHUNK_T tiles at a time through LDS (a run's first piece parked in LDS where
# the run crosses into the next set if it fits, else folded there); then the
# (request, head) tasks, TASK_SLOTS a round, the request count a launch argument
# (at most ``gemm_reqs``).
WIDE_T = 3
WIDE_REQS = 8  # requests the launch takes in its first form
PASS_T = 12
CHUNK_T = 6  # a fold's partials in LDS (G x CHUNK_T x WAVES KB)
TASK_SLOTS = BLOCKS // 2


def xla(npt):
    """x loads in flight ahead of the GEMM form's MFMAs: what the registers
    beside ``npt`` tiles' accumulators hold without spilling."""
    return 16 if npt <= CHUNK_T else (8 if npt <= 10 else 4)


def is_gemm(key):
    return tiles(key) > WIDE_T


def gemm_reqs(key):
    """Requests a GEMM-form build takes (its tiles' worth)."""
    return 16 * tiles(key) // key.qlen if is_gemm(key) else key.tokens // key.qlen


def gemm_key(reqs, qlen, **kw):
    """The GEMM-form build for ``reqs`` requests of ``qlen`` rows."""
    nt = max((reqs * qlen + 15) // 16, WIDE_T + 1)
    return KdaWideBuild(tokens=16 * nt // qlen * qlen, qlen=qlen, **kw)


def depth(key):
    return key.depth or ((3, 2, 1)[tiles(key) - 1] if tiles(key) <= WIDE_T else 1)


KPT = 16  # K columns a lane holds of a state row, at every width
# q, k and the decay in LDS: a row's 16-column pieces 20 floats apart, so the
# recurrence's eight K pieces a wave reads sit on eight disjoint bank quads
HDP = HD // KPT * (KPT + 4)


def kp_(c):
    """LDS index of column ``c`` of a q / k / decay row."""
    return c + c // KPT * 4


def scratch_layout(key: KdaWideBuild) -> dict:
    """16-B tagged words: ``proj`` a (request, in_proj row) (its L bf16 rows),
    ``part`` two a (CTA, group row, request) (its L f32 piece sums), ``ox`` a
    task half's outputs."""
    r = gemm_reqs(key)
    s, ns = key.tokens, key.nblocks + 1
    front = (
        [("fpart", NA * s * front_nq(ns)), ("fwgt", s * (ns + 2)), ("fxrdy", NA)]
        if key.front
        else []
    )
    return pair_layout(
        [
            ("proj", r * NPROJ * 2),
            ("part", BLOCKS * G * ROWS * r * 2 * 2),
            ("ox", r * NH * 2 * OXW * 2),
        ]
        + front,
        align=16,
    )


def put_word(a, i, ws, tag):
    """16-B word ``i`` of region ``a`` := (ws, tag) in one store."""
    bo.buffer_store(
        fx.Vector.from_elements([fx.Int32(w) for w in ws] + [tag], fx.Int32),
        rsrc(a.value),
        fx.Int32(i) * 4,
        cache_modifier=_SCOPE_CM[a.space],
    )


def poll_words(a, idxs, tag, first=None):
    """Words ``idxs`` of region ``a`` once each carries ``tag`` (a 16-B load sees
    a word whole); their three payload words each. ``first``: the first load
    (``load_words``), issued already."""
    load_all = load_words(a, idxs)

    def pending(v):
        bad = v[3] != tag
        for e in range_constexpr(7, 4 * len(idxs), 4):
            bad = bad | (v[e] != tag)
        return bad

    v = _spin(load_all, pending, first=first)
    return [[v[4 * k + e] for e in range(3)] for k in range(len(idxs))]


def load_words(a, idxs):
    """The 16-B loads of words ``idxs`` of region ``a`` (one vector)."""
    cm = _SCOPE_CM[a.space]

    def load_all():
        ws = []
        for i in idxs:
            v = fx.Vector(
                bo.buffer_load(
                    rsrc(a.value), fx.Int32(i) * 4, vec_width=4, dtype=T.i32, cache_modifier=cm
                )
            )
            ws += [v[e] for e in range(4)]
        return fx.Vector.from_elements(ws, fx.Int32)

    return load_all


def bf16_words(word3):
    """A proj word's WT bf16 rows as f32."""
    return [x for w in word3 for x in bf2_f32(w)]


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


def _wloads(c, u, hs, live):
    """Unit ``u``'s weights of this wave: 2 x 16 B a lane of each group, 1 KB an
    instruction at lane * 16 (``pack_w_in``; zeros, with no memory read, unless
    ``live``)."""
    lane = c["lane"]
    wr = rsrc(c["args"]["w_in"], live.select(fx.Int64(NPROJ * HIDDEN * 2), fx.Int64(0)))
    k32 = (c["wave"] * HSTEPS + hs) * (KSTEP // 32)
    wo = uniform(((u // HSTEPS) * G * (HIDDEN // 32) + k32) * 1024)

    def wld(g, q):
        return fx.Vector(
            bo.buffer_load(
                wr, lane * 4, vec_width=4, dtype=T.i32, cache_modifier=CM_NT,
                soffset_bytes=wo + (g * (HIDDEN // 32) + q) * 1024,
            )
        )  # fmt: skip

    return [[wld(g, q) for q in range(KSTEP // 32)] for g in range(G)]


def _xloads(c, hs, live):
    """K step ``hs`` of every token tile's x for this wave (``pack_x``: 1 KB an
    instruction; zeros, with no memory read, unless ``live``)."""
    lane = c["lane"]
    soff = _soff(c, hs)
    nt = c["T"]
    xr = rsrc(c["args"]["x"], live.select(fx.Int64(nt * 16 * HIDDEN * 2), fx.Int64(0)))

    def xld(tt, q):
        return fx.Vector(
            bo.buffer_load(
                xr, lane * 4 + (tt * 2 + q) * 256, vec_width=4, dtype=T.i32,
                soffset_bytes=uniform(soff * (nt * 16)),
            )
        )  # fmt: skip

    return [[xld(tt, q) for q in range(KSTEP // 32)] for tt in range(nt)]


def _loads(c, u, hs, live):
    """One step (unit ``u``, K step ``hs``) of this wave: its weights, then x."""
    return _wloads(c, u, hs, live), _xloads(c, hs, live)


def pack_w_in(w_in):
    """in_proj [NPROJ, HIDDEN] in AITER's 16x16 preshuffle
    (``aiter.ops.shuffle.shuffle_weight``): 1 KB pieces of 16 rows x 32 K in
    (row group, K) order, a lane's 16 B at lane * 16 (row lane % 16, K
    8 (lane // 16) ..). AITER's preshuffled-weight GEMMs read it as it is."""
    assert w_in.shape == (NPROJ, HIDDEN)
    v = w_in.view(GROUPS, ROWS, HIDDEN // 32, 4, 8)
    return v.permute(0, 2, 3, 1, 4).contiguous().view(NPROJ, HIDDEN)


def unpack_w_in(packed):
    """``pack_w_in``'s inverse."""
    v = packed.view(GROUPS, HIDDEN // 32, 4, ROWS, 8)
    return v.permute(0, 3, 1, 2, 4).contiguous().view(NPROJ, HIDDEN)


def pack_x(x, nt):
    """x [S, HIDDEN] in the launch's stream order, its rows padded to ``nt``
    token tiles: K step (wave, step), token tile, K half, lane (K group, row),
    8 K a lane; [16 nt, HIDDEN]."""
    s = x.shape[0]
    assert x.shape == (s, HIDDEN) and s <= 16 * nt
    xp = torch.zeros(16 * nt, HIDDEN, dtype=x.dtype, device=x.device)
    xp[:s] = x
    v = xp.view(nt, 16, HIDDEN // KSTEP, KSTEP // 32, 4, 8)
    return v.permute(2, 0, 3, 4, 1, 5).contiguous().view(16 * nt, HIDDEN)


@flyc.jit
def wait_x(c):
    """Every slice's XRDY flag, then this XCD's L2 made current (x is cached: its
    lines from an earlier launch must not be read)."""
    poll, xrdy, tid = c["fc"]["poll"], c["fc"]["xrdy"], c["tid"]
    if tid < NA:
        poll([(xrdy, tid, 1)])
    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")
    gpu.barrier()


def _slot_w(c):
    return G * c["T"] * WAVES * 64 * 4  # LDS floats a piece


@flyc.jit
def gemv_loop(c):
    """This CTA's run of units (set of groups, K step): the waves' partial
    sums of its one or two pieces -> LDS."""
    tid, lane, wave, nt, d, bid = c["tid"], c["lane"], c["wave"], c["T"], c["D"], c["bid"]
    u0, u1 = uniform(_u0(bid)), uniform(_u0(bid + 1))
    red = c["red"]
    slot_w = _slot_w(c)

    def unit(i):
        u = u0 + i
        return u, u % HSTEPS, u < u1

    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nt)] for _ in range(G)]

    def spill(slot):
        for gs in range_constexpr(G):
            for tt in range_constexpr(nt):
                fx.ptr_store(
                    acc[gs][tt],
                    red + slot * slot_w + (((gs * nt + tt) * WAVES + wave) * 64 + lane) * 4,
                )

    # the KDA task's slot and window row: loaded ahead of the stream, read once
    # the first steps' loads are out
    a, req = c["args"], c["req"]
    idx = [
        fx.Int32(bo.buffer_load(rsrc(a[n]), req, vec_width=1, dtype=T.i32))
        for n in ("st_idx", "win_row")
    ]
    pend = {}
    if const_expr(c["front"]):
        # the first steps' weights go out before x is ready, their x after
        ws0 = [_wloads(c, *unit(i)) for i in range(min(d, NMAX))]
        wait_x(c)
        stamp(c["on"], c["tls"], c["tid"], 13)
        for i in range_constexpr(min(d, NMAX)):
            _, hs, live = unit(i)
            pend[i] = (ws0[i], _xloads(c, hs, live))
    else:
        for i in range_constexpr(min(d, NMAX)):
            p, hs, live = unit(i)
            pend[i] = _loads(c, p, hs, live)
    c["slot0"], c["w_row"] = (uniform(v) for v in idx)
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
            # the run crosses into the next set here: the first piece is done
            _, hs, live = unit(i)
            cross = (hs == 0) & live
            if cross:
                spill(0)
            for gs in range_constexpr(G):
                for tt in range_constexpr(nt):
                    acc[gs][tt] = cross.select(fx.Vector.filled(4, 0.0, fx.Float32), acc[gs][tt])
        for q in range_constexpr(KSTEP // 32):
            for tt in range_constexpr(nt):
                xb = xs[tt][q].bitcast(fx.BFloat16)
                for gs in range_constexpr(G):
                    acc[gs][tt] = mfma_bf16(ws[gs][q].bitcast(fx.BFloat16), xb, acc[gs][tt])
        rocdl.sched_barrier(0)
    spill(1)
    lds_barrier()
    stamp(c["on"], c["tls"], tid, 1)


@flyc.jit
def gemv_emit(c):
    """The run's pieces: a piece that starts its set's K is the set's head,
    and its CTA adds the set's later pieces (published by the next one or two
    CTAs) in order -> bf16 PROJ words. A thread a (group row, request), its L
    tokens. The split depends on the CTA alone, never on the width."""
    s, L, tid, nt, bid = c["S"], c["L"], c["tid"], c["T"], c["bid"]
    tag = c["tag"]
    R = s // L
    u0, u1 = uniform(_u0(bid)), uniform(_u0(bid + 1))
    red = c["red"]
    slot_w = _slot_w(c)
    pa, hsa = u0 // HSTEPS, u0 % HSTEPS
    crosses = (u1 - 1) // HSTEPS != pa
    # slot 0: the first piece when the run crosses; slot 1: the last piece
    n = G * ROWS * R
    per = (n + THREADS - 1) // THREADS

    def piece_sum(slot, gs, r, tok):
        base = slot * (slot_w // 256) + (gs * nt + tok // 16) * WAVES
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(red + (((base + w) * 64 + 16 * (r // 4) + tok % 16) * 4 + r % 4))
        return tot

    es = [fx.min(tid + THREADS * it, n - 1) for it in range(per)]
    cs = [(e // (ROWS * R), e % ROWS, (e // ROWS) % R) for e in es]

    def pieces(slot, it):
        gs, r, q = cs[it]
        return [piece_sum(slot, gs, r, q * L + j) for j in range(L)] + [fx.Float32(0.0)] * (WT - L)

    # the head's first poll of the set's later pieces goes out before the
    # task's inputs (loads return in order)
    ph = crosses.select(pa + 1, pa)
    nb1, nb2 = fx.min(bid + 1, BLOCKS - 1), fx.min(bid + 2, BLOCKS - 1)
    two = (bid + 2 < BLOCKS) & (_u0(bid + 2) < (ph + 1) * HSTEPS)
    nb2 = two.select(nb2, nb1)
    pidx = [(nb * n + e) * 2 + hh for nb in (nb1, nb2) for e in es for hh in (0, 1)]
    first = load_words(c["part"], pidx)()
    # a piece that is not its set's head: published for the head's CTA
    pub_slot = crosses.select(0, 1)
    if crosses | (hsa != 0):
        for it in range_constexpr(per):
            v = pieces(pub_slot, it)
            if tid + THREADS * it < n:
                for hh in range_constexpr(2):
                    put_word(
                        c["part"], (bid * n + es[it]) * 2 + hh,
                        [x.bitcast(fx.Int32) for x in v[3 * hh : 3 * hh + 3]], tag,
                    )  # fmt: skip
    stamp(c["on"], c["tls"], tid, 9)
    # the task's inputs go out once the piece is published
    c["k"] = c["pre"]()
    # the head this CTA owns (the last piece), and its set's later pieces
    if crosses | (hsa == 0):
        got = poll_words(c["part"], pidx, tag, first)
        got1, got2 = got[: 2 * per], got[2 * per :]
        for it in range_constexpr(per):
            own = pieces(1, it)
            g1 = [f32_of(w) for w in got1[2 * it] + got1[2 * it + 1]]
            g2 = [f32_of(w) for w in got2[2 * it] + got2[2 * it + 1]]
            vs = []
            for j in range_constexpr(WT):
                tot = own[j] + g1[j]
                vs.append(bf16r(two.select(tot + g2[j], tot)))
            gs, r, q = cs[it]
            row = (G * ph + gs) * ROWS + r
            if tid + THREADS * it < n:
                put_word(
                    c["proj"], q * NPROJ + row,
                    [bf16_pair(vs[2 * k], vs[2 * k + 1]).bitcast(fx.Int32) for k in range(WT // 2)],
                    tag,
                )  # fmt: skip
                if const_expr(c["debug"]):
                    for j in range_constexpr(L):
                        bo.buffer_store(
                            vs[j].to(fx.BFloat16), rsrc(c["args"]["proj_dbg"]),
                            (q * L + j) * NPROJ + row,
                        )  # fmt: skip
    lds_barrier()
    stamp(c["on"], c["tls"], tid, 2)


# ---------------------------------------------------------------- the GEMM form
def _xload1(c, xr, hs, tt, q):
    """Half ``q`` of K step ``hs`` of token tile ``tt``'s x for this wave (``pack_x``)."""
    return fx.Vector(
        bo.buffer_load(
            xr, c["lane"] * 4 + (tt * 2 + q) * 256, vec_width=4, dtype=T.i32,
            soffset_bytes=uniform(_soff(c, hs) * (c["T"] * 16)),
        )
    )  # fmt: skip


@flyc.jit
def gemm_pass(c, p):
    """Pass ``p``: token tiles PASS_T p .. of this CTA's run of units, each
    (group, tile) chained over the run's K steps as ``gemv_loop`` chains it; a
    piece folded (``gemm_fold``) once its last unit is in."""
    bid, nt, d = c["bid"], c["T"], c["D"]
    t0 = p * PASS_T
    npt = min(PASS_T, nt - t0)
    u0, u1 = uniform(_u0(bid)), uniform(_u0(bid + 1))

    def unit(i):
        u = u0 + i
        return u, u % HSTEPS, u < u1

    def xres(i):
        live = unit(i)[2]
        return rsrc(c["args"]["x"], live.select(fx.Int64(nt * 16 * HIDDEN * 2), fx.Int64(0)))

    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    acc = [[zero for _ in range(npt)] for _ in range(G)]
    la = xla(npt)
    park = npt <= CHUNK_T  # a crossing run's first piece waits in LDS
    wq = {i: _wloads(c, *unit(i)) for i in range(min(d, NMAX))}
    seq = [(i, q, tt) for i in range(NMAX) for q in range(KSTEP // 32) for tt in range(npt)]
    xr = [xres(i) for i in range(NMAX)]
    xq = {}

    def issue(k):
        i, q, tt = seq[k]
        xq[k] = _xload1(c, xr[i], unit(i)[1], t0 + tt, q)

    for k in range_constexpr(min(la, len(seq))):
        issue(k)
    rocdl.sched_barrier(0)
    ws = None
    for k in range_constexpr(len(seq)):
        i, q, tt = seq[k]
        if const_expr(q == 0 and tt == 0):
            if const_expr(i + d < NMAX):
                wq[i + d] = _wloads(c, *unit(i + d))
            ws = wq.pop(i)
            if const_expr(i > 0):
                # the run crosses into the next set here: the first piece is done
                _, hs, live = unit(i)
                cross = (hs == 0) & live
                if cross:
                    if const_expr(park):
                        gemm_store(c, acc, 0, npt)
                    else:
                        gemm_fold(c, acc, p, False)
                for gs in range_constexpr(G):
                    for t in range_constexpr(npt):
                        acc[gs][t] = cross.select(zero, acc[gs][t])
        if const_expr(k + la < len(seq)):
            issue(k + la)
        xb = xq.pop(k).bitcast(fx.BFloat16)
        for gs in range_constexpr(G):
            acc[gs][tt] = mfma_bf16(ws[gs][q].bitcast(fx.BFloat16), xb, acc[gs][tt])
        rocdl.sched_barrier(0)
    stamp(c["on"], c["tls"], c["tid"], 1)
    if const_expr(park):
        if (u1 - 1) // HSTEPS != u0 // HSTEPS:
            gemm_fold(c, None, p, False)
    gemm_fold(c, acc, p, True)
    stamp(c["on"], c["tls"], c["tid"], 2)


def gemm_store(c, acc, t0, nt):
    """This wave's partial sums of tiles t0 .. t0 + nt - 1 into LDS."""
    red, wave, lane = c["red"], c["wave"], c["lane"]
    for gs in range_constexpr(G):
        for tl in range_constexpr(nt):
            fx.ptr_store(
                acc[gs][t0 + tl], red + (((gs * CHUNK_T + tl) * WAVES + wave) * 64 + lane) * 4
            )


SP = 16 * CHUNK_T + 4  # LDS floats a (group, row) of a fold's summed tokens


@flyc.jit
def gemm_fold(c, acc, p, last):
    """A piece of pass ``p`` (``acc``: this wave's partial sums; None: parked in
    LDS already), CHUNK_T tiles at a time, summed in wave order a (group row,
    token) through LDS: a thread a (group, tile, MFMA lane) adds the waves' 4
    rows of one token, then a thread a (group row, request) takes its L tokens.
    The first piece of a run that crosses into the next set (not ``last``), or
    a last piece that does not start its set: published for the set's head CTA.
    The head piece (``last``, starting its set): plus the set's later pieces
    (the next one or two CTAs') in order -> bf16 PROJ words. The same sums, in
    the same order, as ``gemv_emit``'s."""
    tid, bid, L, nt = c["tid"], c["bid"], c["L"], c["T"]
    tag, red, R, rmax = c["tag"], c["red"], c["R"], c["RMAX"]
    npt = min(PASS_T, nt - p * PASS_T)
    rqc = CHUNK_T * 16 // L  # requests a chunk
    u0, u1 = uniform(_u0(bid)), uniform(_u0(bid + 1))
    pa, hsa = u0 // HSTEPS, u0 % HSTEPS
    crosses = (u1 - 1) // HSTEPS != pa
    head = crosses | (hsa == 0)
    n = G * ROWS * rmax
    held = []
    for ch in range_constexpr((npt + CHUNK_T - 1) // CHUNK_T):
        t0 = ch * CHUNK_T
        ntc = min(CHUNK_T, npt - t0)
        if const_expr(acc is not None):
            lds_barrier()
            gemm_store(c, acc, t0, ntc)
            if const_expr(last and t0 + ntc == npt and p == c["NPASS"] - 1):
                # the first task round's inputs, in flight under the fold
                c["nxt"] = c["task_pre"](0)
        lds_barrier()
        n1 = G * ntc * 64
        s1 = []
        for it in range_constexpr((n1 + THREADS - 1) // THREADS):
            e = fx.min(tid + THREADS * it, n1 - 1)
            blk, ln = e // 64, e % 64
            base = ((blk // ntc) * CHUNK_T + blk % ntc) * WAVES
            tot = fx.Vector.filled(4, 0.0, fx.Float32)
            for w in range_constexpr(WAVES):
                tot = tot + fx.Vector(
                    fx.ptr_load(
                        red + ((base + w) * 64 + ln) * 4,
                        result_type=fx.Vector.make_type(4, fx.Float32),
                    )
                )
            s1.append((tot, blk, ln, tid + THREADS * it < n1))
        lds_barrier()
        for tot, blk, ln, live in s1:
            if live:
                gs, tl = blk // ntc, blk % ntc
                for k in range_constexpr(4):
                    fx.ptr_store(
                        tot[k], red + ((gs * ROWS + 4 * (ln // 16) + k) * SP + 16 * tl + ln % 16)
                    )
        lds_barrier()
        if const_expr(last and t0 + ntc == npt):
            stamp(c["on"], c["tls"], tid, 10)
        items = G * ROWS * rqc
        for it in range_constexpr((items + THREADS - 1) // THREADS):
            e = fx.min(tid + THREADS * it, items - 1)
            gs, r, rq = e // (ROWS * rqc), e % ROWS, (e // ROWS) % rqc
            q = p * (PASS_T * 16 // L) + ch * rqc + rq
            ok = (tid + THREADS * it < items) & (q < R)
            v = [fx.ptr_load(red + ((gs * ROWS + r) * SP + rq * L + j)) for j in range(L)]
            v += [fx.Float32(0.0)] * (WT - L)
            ge = gs * (ROWS * rmax) + fx.min(q, R - 1) * ROWS + r
            pub = (ok & ~head) if last else ok
            if pub:
                for hh in range_constexpr(2):
                    put_word(
                        c["part"], (bid * n + ge) * 2 + hh,
                        [x.bitcast(fx.Int32) for x in v[3 * hh : 3 * hh + 3]], tag,
                    )  # fmt: skip
            held.append((v, ok, ge, gs, r, q))
    lds_barrier()
    if const_expr(last):
        stamp(c["on"], c["tls"], tid, 11)
        gemm_head(c, held, head, crosses, pa, n)


@flyc.jit
def gemm_head(c, held, head, crosses, pa, n):
    """The head's pieces ``held`` (every round's) plus its set's later pieces,
    polled at once, in order -> bf16 PROJ words."""
    L, tag, bid = c["L"], c["tag"], c["bid"]
    ph = crosses.select(pa + 1, pa)
    nb1, nb2 = fx.min(bid + 1, BLOCKS - 1), fx.min(bid + 2, BLOCKS - 1)
    two = (bid + 2 < BLOCKS) & (_u0(bid + 2) < (ph + 1) * HSTEPS)
    nb2 = two.select(nb2, nb1)
    if head:
        pidx = [(nb * n + h[2]) * 2 + hh for h in held for nb in (nb1, nb2) for hh in (0, 1)]
        got = poll_words(c["part"], pidx, tag)
        stamp(c["on"], c["tls"], c["tid"], 12)
        for k in range_constexpr(len(held)):
            v, ok, ge, gs, r, q = held[k]
            g1 = [f32_of(w) for w in got[4 * k] + got[4 * k + 1]]
            g2 = [f32_of(w) for w in got[4 * k + 2] + got[4 * k + 3]]
            vs = []
            for j in range_constexpr(WT):
                tot = v[j] + g1[j]
                vs.append(bf16r(two.select(tot + g2[j], tot)))
            row = (G * ph + gs) * ROWS + r
            if ok:
                put_word(
                    c["proj"], q * NPROJ + row,
                    [bf16_pair(vs[2 * m], vs[2 * m + 1]).bitcast(fx.Int32) for m in range(WT // 2)],
                    tag,
                )  # fmt: skip
                if const_expr(c["debug"]):
                    for j in range_constexpr(L):
                        bo.buffer_store(
                            vs[j].to(fx.BFloat16), rsrc(c["args"]["proj_dbg"]),
                            (q * L + j) * NPROJ + row,
                        )  # fmt: skip


def task_pre(c):
    """Round k's task inputs on this CTA (``kda_pre``): (has, task, request, inputs)."""
    bid, a = c["bid"], c["args"]
    ntasks = c["R"] * NH
    if const_expr(c["SPLIT"]):
        slot, hf, per = (bid // 16) * 8 + bid % 8, (bid % 16) // 8, TASK_SLOTS
    else:
        slot, hf, per = bid, fx.Int32(0) * bid, BLOCKS

    def pre(k):
        t = k * per + slot
        has = t < ntasks
        tc = uniform(fx.min(t, ntasks - 1))
        req = tc // NH
        idx = [
            uniform(fx.Int32(bo.buffer_load(rsrc(a[nm]), req, vec_width=1, dtype=T.i32)))
            for nm in ("st_idx", "win_row")
        ]
        return has, tc, req, kda_pre(c, req, tc % NH, hf, has, idx[0], idx[1])

    return pre


@flyc.jit
def gemm_tasks(c):
    """Every (request, head) task: its two halves on CTAs b and b + 8 (a block of
    16 holds 8 tasks), TASK_SLOTS a round, while that is one round; else whole
    tasks (the same bits), BLOCKS a round. The first round's inputs are issued in
    the last fold (``c["nxt"]``), each next round's under the round before."""
    split = c["SPLIT"]
    hf = (c["bid"] % 16) // 8 if split else fx.Int32(0) * c["bid"]
    per = TASK_SLOTS if split else BLOCKS
    rounds = (c["RMAX"] * NH + per - 1) // per
    pre = c["task_pre"]
    nxt = c["nxt"]
    for k in range_constexpr(rounds):
        has, tc, req, kk = nxt
        if const_expr(k + 1 < rounds):
            nxt = pre(k + 1)
        if has:
            kda_task(c, tc, req, tc % NH, hf, kk)
        if const_expr(k == 0):
            stamp(c["on"], c["tls"], c["tid"], 13)


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


def kda_pre(c, req, h, hf, has, slot0, w_row):
    """The (request, head) task's inputs that need no hand-off, issued under
    the CTA's in_proj emit and not waited on there: f_b rows, constants, the
    state, the conv window (the state and f_b with no memory read unless
    ``has``; zeros then). ``slot0`` / ``w_row``: the request's slot and window
    row."""
    L, tid, kp = c["L"], c["tid"], c["KPT"]
    lane, wave = tid % 64, tid // 64
    a = c["args"]
    fa_l, qkl, decay, vloc, ol, beta = (
        c["fa"], c["qkl"], c["decay"], c["vloc"], c["ol"], c["beta"],
    )  # fmt: skip
    slot = has.select(slot0, fx.Int32(-1))
    live = slot >= 0
    sl = fx.max(slot, 0)

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
    a_log = ld_f32(a["a_log"], h)
    dtb = [ld_f32(a["dt_bias"], h * HD + wave * 16 + 4 * (lane // 16) + i) for i in range(4)]
    lpr = HD // kp  # lanes a state row
    rpt = (HALF if c["SPLIT"] else HD) * HD // (THREADS * kp)  # state rows a thread
    grp, k0 = tid // lpr, (tid % lpr) * kp
    r0 = hf * HALF + grp * rpt  # the thread's first state row
    base0 = rsrc(
        a["rstate"] + fx.Int64(sl) * fx.Int64(a["rs_seq"]) * 4,
        live.select(fx.Int64(NH * HD * HD * 4), fx.Int64(0)),
    )
    st = [
        [
            fx.Vector(
                bo.buffer_load(
                    base0, (h * HD + r0 + i) * HD + k0 + 4 * q, vec_width=4, dtype=T.f32
                )
            )
            for q in range(kp // 4)
        ]
        for i in range(rpt)
    ]
    ct = fx.min(tid, 3 * HD - 1)
    part, d = ct // HD, ct % HD
    chan = part * PROJ + h * HD + d
    cs = a["conv_st"] + fx.Int64(sl) * fx.Int64(a["cs_seq"]) * 2
    co = chan * a["cs_dim"]
    cw = fx.Vector(bo.buffer_load(rsrc(a["conv_w"]), chan * CONV_W, vec_width=4, dtype=T.f32))
    win = [
        bo.buffer_load(rsrc(cs), co + j * a["cs_tok"], vec_width=1, dtype=T.bf16)
        for j in range(CONV_W - 1)
    ]
    on_w = [
        bo.buffer_load(rsrc(a["on_w"]), lane + 64 * u, vec_width=1, dtype=T.bf16) for u in range(2)
    ]
    stamp(c["on"], c["tls"], tid, 3)
    return dict(
        slot=slot, live=live, sl=sl, w_row=w_row, fb=fb, a_log=a_log, dtb=dtb, st=st,
        part=part, d=d, chan=chan, cw=cw, win=win, on_w=on_w,
    )  # fmt: skip


@flyc.jit
def kda_task(c, task, req, h, hf, k):
    """Half ``hf`` of the (request, head) task ``task``: state rows 64 hf .. 64 hf + 63
    (``c["SPLIT"]``), or the whole task (hf 0: every state row, no swap)."""
    L, tid, kp = c["L"], c["tid"], c["KPT"]
    lane, wave = tid % 64, tid // 64
    a = c["args"]
    # read before the traced branches (a dict lookup inside one is carried state)
    box = c.get("core_box")
    fa_l, qkl, decay, vloc, ol, beta = (
        c["fa"], c["qkl"], c["decay"], c["vloc"], c["ol"], c["beta"],
    )  # fmt: skip
    live, sl, w_row, fb, dtb = k["live"], k["sl"], k["w_row"], k["fb"], k["dtb"]
    part, d, chan, cw = k["part"], k["d"], k["chan"], k["cw"]
    aa = exp_(k["a_log"])
    st = [[v[e] for v in row for e in range(4)] for row in k["st"]]
    win = [fx.Float32(fx.BFloat16(v)) for v in k["win"]]
    on_w = [fx.Float32(fx.BFloat16(v)) for v in k["on_w"]]
    lpr = HD // kp  # lanes a state row
    rpt = (HALF if c["SPLIT"] else HD) * HD // (THREADS * kp)  # state rows a thread
    grp, k0 = tid // lpr, (tid % lpr) * kp
    r0 = hf * HALF + grp * rpt  # the thread's first state row
    # the half's share of the global writes: q / k windows and ring k on half 0,
    # v windows and ring v of the half's rows, ring g and beta on half 0
    mine = ((part < 2) & (hf == 0)) | ((part == 2) & (d // HALF == hf))
    if const_expr(not c["SPLIT"]):
        mine = part < 3
    first = hf == 0

    # every in_proj word the task reads, one poll batch a thread: its conv
    # channel's, f_a channel tid % 128's, the head's beta and output gate
    # channel tid % 128's (each word a channel's L tokens)
    pr = req * NPROJ
    jn = fx.min(wave, L - 1)
    got = poll_words(
        c["proj"],
        [pr + chan, pr + FA_0 + tid % HD, pr + BETA_0 + h, pr + G2_0 + h * HD + tid % HD],
        c["tag"],
    )
    xv, fav, bvs, g2v = (bf16_words(g) for g in got)
    bv = bvs[0]
    for j in range_constexpr(1, L):
        bv = (tid == j).select(bvs[j], bv)
    stamp(c["on"], c["tls"], tid, 4)

    # conv: thread < 384 owns channel (part, d) of the head
    # the task's global loads are all issued by now: a load behind a store
    # would wait for its acknowledgement
    seq = win + list(xv)
    outs = [_conv_silu(seq, cw, j) for j in range(L)]
    ring_len = a["ring_len"]
    if live & (tid < 3 * HD) & mine:
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
                fx.ptr_store(outs[j], qkl + ((2 * j + part) * HDP + kp_(d)))
            if part == 2:
                fx.ptr_store(outs[j], vloc + (j * HD + d))
    if tid < HD:
        for j in range_constexpr(L):
            fx.ptr_store(fav[j], fa_l + (j * HD + tid))
            fx.ptr_store(g2v[j], c["g2l"] + (j * HD + tid))
    b = sigmoid(bv)
    if tid < L:
        fx.ptr_store(b, beta + tid)
        if live & first:
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
            fx.ptr_store(exp_(gate), decay + (t * HDP + kp_(kk)))
            if live & first:
                bo.buffer_store(gate, rg, (h * ring_len + t) * HD + kk)
    # l2norm: (token, q / k) row r = tid / 16 of the 2 L rows, 8 channels a lane
    if tid < 2 * L * 16:
        r = tid // 16
        c8 = (tid % 16) * 8
        ys = [fx.ptr_load(qkl + (r * HDP + kp_(c8) + e)) for e in range(8)]
        ss = fx.Float32(0.0)
        for e in range_constexpr(8):
            ss = fmath.fma(ys[e], ys[e], ss)
        ss = group_sum(ss, 16)
        rn = fx.Float32(1.0) / fx.Float32(fmath.sqrt(ss + fx.Float32(1e-6)))
        rn = (r % 2 == 0).select(rn * fx.Float32(SCALE), rn)
        for e in range_constexpr(8):
            fx.ptr_store(ys[e] * rn, qkl + (r * HDP + kp_(c8) + e))
    lds_barrier()
    stamp(c["on"], c["tls"], tid, 6)

    # recurrence: rows grp * rpt .., K columns k0 .. k0 + kp, as column pairs
    # (packed f32 ops; a pair's lanes are SGLang's even and odd partial sums)
    def v2(a, b):
        return fx.Vector.from_elements([a, b], fx.Float32)

    def lds_pairs(at):
        out = []
        for q in range_constexpr(kp // 4):
            v = fx.Vector(fx.ptr_load(at + 4 * q, result_type=fx.Vector.make_type(4, fx.Float32)))
            out += [v2(v[0], v[1]), v2(v[2], v[3])]
        return out

    def dot2(x, y):
        """SGLang's lane sum: even and odd columns apart, then added."""
        acc = v2(fx.Float32(0.0), fx.Float32(0.0))
        for e in range_constexpr(kp // 2):
            acc = fmath.fma(x[e], y[e], acc)
        return acc[0] + acc[1]

    sp = [[v2(row[2 * e], row[2 * e + 1]) for e in range(kp // 2)] for row in st]
    os_ = []
    for j in range_constexpr(L):
        qv = lds_pairs(qkl + ((2 * j) * HDP + kp_(k0)))
        kv = lds_pairs(qkl + ((2 * j + 1) * HDP + kp_(k0)))
        dk = lds_pairs(decay + (j * HDP + kp_(k0)))
        bj = fx.ptr_load(beta + j)
        vs = [fx.ptr_load(vloc + (j * HD + r0 + i)) for i in range(rpt)]
        for i in range_constexpr(rpt):
            s_ = [sp[i][e] * dk[e] for e in range(kp // 2)]
            u = group_sum(dot2(s_, kv), lpr)
            vn = (vs[i] - u) * bj
            vn2 = v2(vn, vn)
            s_ = [fmath.fma(kv[e], vn2, s_[e]) for e in range(kp // 2)]
            os_.append(live.select(bf16r(group_sum(dot2(s_, qv), lpr)), fx.Float32(0.0)))
            sp[i] = s_
    if tid % lpr == 0:
        for j in range_constexpr(L):
            for i in range_constexpr(rpt):
                fx.ptr_store(os_[j * rpt + i], ol + (j * HD + r0 + i))
    lds_barrier()
    stamp(c["on"], c["tls"], tid, 7)
    # the other half's outputs: 16-B tagged words, three
    # f32 of the half's (token, row) a word
    if const_expr(c["SPLIT"]):
        if tid < OXW:
            fs = [3 * tid + e for e in range(3)]
            mine_o = [
                fx.ptr_load(ol + (fx.min(f // HALF, L - 1) * HD + hf * HALF + f % HALF)) for f in fs
            ]
            put_word(
                c["ox"], (task * 2 + hf) * OXW + tid, [v.bitcast(fx.Int32) for v in mine_o], c["tag"]
            )
            gox = poll_words(c["ox"], [(task * 2 + 1 - hf) * OXW + tid], c["tag"])[0]
            for e in range_constexpr(3):
                f = fs[e]
                if f // HALF < L:
                    fx.ptr_store(
                        gox[e].bitcast(fx.Float32),
                        ol + (fx.min(f // HALF, L - 1) * HD + (1 - hf) * HALF + f % HALF),
                    )
    lds_barrier()
    if const_expr(c["debug"]):
        for i in range_constexpr((L * HD + THREADS - 1) // THREADS):
            e = fx.min(tid + THREADS * i, L * HD - 1)
            if first & (tid + THREADS * i < L * HD):
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
    for u, x in ((0, x0), (1, x1)):
        if const_expr(box is None):
            own = (hf == u) if c["SPLIT"] else (wave >= 0)
        else:
            # half 0 stores both halves' channels (the same values), so one
            # CTA's stores precede the box
            own = hf == 0
        if (wave < L) & own:
            e = lane + 64 * u
            y = x * rr * on_w[u] * sigmoid(fx.ptr_load(c["g2l"] + (jn * HD + e)))
            bo.buffer_store(
                y.to(fx.BFloat16),
                rsrc(a["core_out"]),
                (req * L + jn) * PROJ + h * HD + e,
                cache_modifier=0 if box is None else CM_DEV,
            )
    lds_barrier()
    if const_expr(box is not None):
        # every wave's core stores acknowledged, then the task's box
        gpu.barrier()
        if (tid == 0) & (hf == 0):
            Mailbox(box[1]).put(box[0], task, fx.Int32(1))
    stamp(c["on"], c["tls"], tid, 8)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}



class KdaLdsTypes(NamedTuple):
    smem: Any
    role: Any


def lds_types(key: KdaWideBuild) -> KdaLdsTypes:
    """The LDS structs ``kda_body`` takes at this build: allocate each from the
    enclosing kernel's allocator (``role`` may sit in a wider union)."""
    s, L = key.tokens, key.qlen
    gemm = is_gemm(key)
    nt = tiles(key)
    ns = key.nblocks + 1

    @fx.struct
    class GemvLds:
        red: fx.Array[fx.Float32, (G * CHUNK_T if gemm else 2 * G * nt) * WAVES * 64 * 4, 16]

    # the seam's sources as packed bf16 pairs (exact: every source is bf16)
    @fx.struct
    class FrontLds:
        vals: fx.Array[fx.Int32, (s * ns * COLS // 2) if key.front else 4, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]

    @fx.struct
    class KdaLds:
        fa: fx.Array[fx.Float32, L * HD, 16]
        qkl: fx.Array[fx.Float32, 2 * L * HDP, 16]
        decay: fx.Array[fx.Float32, L * HDP, 16]
        vloc: fx.Array[fx.Float32, L * HD, 16]
        ol: fx.Array[fx.Float32, L * HD, 16]
        g2l: fx.Array[fx.Float32, L * HD, 16]
        beta: fx.Array[fx.Float32, 16, 16]

    # the KDA task's LDS reuses the in_proj partials' once they are emitted
    @fx.union
    class RoleLds:
        gemv: GemvLds
        kda: KdaLds
        front: FrontLds

    @fx.struct
    class Smem:
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    return KdaLdsTypes(smem=Smem, role=RoleLds)


KDA_ARGS = (
    "x", "w_in", "w_fb", "conv_w", "conv_st", "a_log", "dt_bias", "on_w", "rstate",
    "st_idx", "win_row", "conv_win", "ring_k", "ring_v", "ring_g", "ring_b",
    "core_out", "proj_dbg", "o_dbg", "cs_seq", "cs_dim", "cs_tok", "rs_seq",
    "cw_seq", "cw_tok", "cw_q", "rk_seq", "rv_seq", "rg_seq", "rb_seq", "ring_len",
    "layer", "epoch", "tl", "prefix", "blocks", "blk_sm", "blk_sr", "ares_nw",
    "ares_qk", "in_nw", "reqs", "aux",
)  # kda_body's ``args``: the kernel's parameters but its scratch


@flyc.jit
def kda_body(key: KdaWideBuild, smem, role, scratch, args: dict) -> None:
    """The launch's whole body, traced into the calling kernel: LDS from the
    ``lds_types`` structs (``smem`` peeked, ``role`` as allocated), the scratch
    layout (``scratch_layout``) from ``scratch``; ``args`` the kernel
    parameters by ``KDA_ARGS`` name. With ``args["core_box"]`` = (box region,
    tag) the core is stored device scope and each (request, head) task's box
    then holds the tag (a consumer in the same launch)."""
    s, L = key.tokens, key.qlen
    assert s % L == 0 and 1 <= L <= WT, (s, L)
    ntasks = s // L * NH
    gemm = is_gemm(key)
    nt = tiles(key)
    if gemm:
        assert not key.front and (16 * CHUNK_T) % L == 0, "the GEMM form: no front, whole requests a fold"
    else:
        assert 2 * 8 * ((ntasks + 7) // 8) <= BLOCKS - TASK0, "a KDA task half a CTA"
    lay = scratch_layout(key)
    ns = key.nblocks + 1
    assert not key.front or (key.nblocks >= 1 and s <= NA), "the front's gate runs token t on CTA t"

    a_ = args
    x, w_in, w_fb, conv_w, conv_st = a_["x"], a_["w_in"], a_["w_fb"], a_["conv_w"], a_["conv_st"]
    a_log, dt_bias, on_w, rstate = a_["a_log"], a_["dt_bias"], a_["on_w"], a_["rstate"]
    st_idx, win_row, conv_win = a_["st_idx"], a_["win_row"], a_["conv_win"]
    ring_k, ring_v, ring_g, ring_b = a_["ring_k"], a_["ring_v"], a_["ring_g"], a_["ring_b"]
    core_out, proj_dbg, o_dbg = a_["core_out"], a_["proj_dbg"], a_["o_dbg"]
    cs_seq, cs_dim, cs_tok, rs_seq = a_["cs_seq"], a_["cs_dim"], a_["cs_tok"], a_["rs_seq"]
    cw_seq, cw_tok, cw_q = a_["cw_seq"], a_["cw_tok"], a_["cw_q"]
    rk_seq, rv_seq, rg_seq, rb_seq = a_["rk_seq"], a_["rv_seq"], a_["rg_seq"], a_["rb_seq"]
    ring_len, layer, epoch, tl = a_["ring_len"], a_["layer"], a_["epoch"], a_["tl"]
    prefix, blocks, blk_sm, blk_sr = a_["prefix"], a_["blocks"], a_["blk_sm"], a_["blk_sr"]
    ares_nw, ares_qk, in_nw, reqs = a_["ares_nw"], a_["ares_qk"], a_["in_nw"], a_["reqs"]
    aux = a_["aux"]
    core_box = args.get("core_box")
    assert core_box is None or not is_gemm(key), "a core box: the first form"
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    gl, kl, fl = role.gemv.peek(), role.kda.peek(), role.front.peek()
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
        "front": key.front,
        "lb": key.lower_bound,
        "onorm_eps": key.onorm_eps,
        "tag": mb.tag,
        "SPLIT": True,
        "red": gl.red.ptr,
        "fa": kl.fa.ptr,
        "qkl": kl.qkl.ptr,
        "decay": kl.decay.ptr,
        "vloc": kl.vloc.ptr,
        "ol": kl.ol.ptr,
        "g2l": kl.g2l.ptr,
        "beta": kl.beta.ptr,
        "on": key.timeline,
        "tls": smem.tls.ptr,
        "proj": sreg(scratch, lay["proj"][0], "proj"),
        "part": sreg(scratch, lay["part"][0], "part"),
        "ox": sreg(scratch, lay["ox"][0], "ox"),
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
    if const_expr(key.front):
        # the seam's state, as K1 builds it (attention side: no delta)
        fc = {
            "S": s,
            "NS": ns,
            "NB": key.nblocks,
            "tid": tid,
            "lane": tid % 64,
            "wave": tid // 64,
            "delta": False,
            "debug": False,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "write_idx": key.write_idx,
            "x_tiles": nt,
            "put": mb.put,
            "poll": mb.poll,
            "red": fl.red.ptr,
            "vals": fl.vals.ptr,
            "xbuf": x,
            "args": {
                "prefix": prefix,
                "blocks": blocks,
                "blk_sm": blk_sm,
                "blk_sr": blk_sr,
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
            },
        }
        for region in ("part", "wgt", "xrdy"):
            fc[region] = sreg(scratch, lay["f" + region][0], "f" + region)
        if const_expr(key.aux_slot >= 0):
            fc["aux"] = (aux, key.aux_slot)
        c["fc"] = fc
    if const_expr(core_box is not None):
        c["core_box"] = core_box
    stamp_begin(key.timeline, smem.tls.ptr, tid, TL_POINTS)
    if const_expr(gemm):
        c["R"] = reqs
        c["RMAX"] = gemm_reqs(key)
        # whole tasks, BLOCKS a round, once halves need more than one round
        c["SPLIT"] = gemm_reqs(key) * NH <= TASK_SLOTS
        c["NPASS"] = (nt + PASS_T - 1) // PASS_T
        c["task_pre"] = task_pre(c)
        for p in range_constexpr(c["NPASS"]):
            gemm_pass(c, p)
        gemm_tasks(c)
    if const_expr(key.front):
        if bid < NA:
            front_slice(fc, bid)
            stamp(key.timeline, smem.tls.ptr, tid, 10)
            if bid < s:
                front_gate(fc, bid)
            stamp(key.timeline, smem.tls.ptr, tid, 11)
            front_mix(fc, bid)
            stamp(key.timeline, smem.tls.ptr, tid, 12)
    if const_expr(not gemm):
        # task t = (request, head), half hf on CTA TASK0 + 16 (t // 8) + 8 hf + t % 8,
        # its inputs in flight under the emit
        o = fx.max(bid - TASK0, 0)
        hf = (o % 16) // 8
        t = o // 16 * 8 + o % 8
        has = (bid >= TASK0) & (t < ntasks)
        tc = uniform(fx.min(t, ntasks - 1))
        c["req"] = tc // NH
        gemv_loop(c)
        c["pre"] = lambda: kda_pre(c, tc // NH, tc % NH, hf, has, c["slot0"], c["w_row"])
        gemv_emit(c)
        if has:
            kda_task(c, tc, tc // NH, tc % NH, hf, c["k"])
    stamp_flush(key.timeline, smem.tls.ptr, tl, tid, bid, TL_POINTS)



def build(key: KdaWideBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    types = lds_types(key)
    s, L = key.tokens, key.qlen
    gemm = is_gemm(key)

    keyed = key_tuple(key, SOURCES)
    name = f"k3_mono_kda_wide_s{s}_l{L}_d{depth(key)}_g{int(key.debug)}_t{int(key.timeline)}"
    if gemm:
        name += "_gemm"
    if key.front:
        name += f"_f{key.nblocks}" + (f"w{key.write_idx}" if key.write_idx >= 0 else "")
        name += f"a{key.aux_slot}" if key.aux_slot >= 0 else ""

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
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        reqs: Int32,
        aux: Int64,
    ):
        _ = keyed
        alloc = fx.SharedAllocator()
        smem = alloc.allocate(types.smem).peek()
        role = alloc.allocate(types.role)
        params = (
            x, w_in, w_fb, conv_w, conv_st, a_log, dt_bias, on_w, rstate, st_idx,
            win_row, conv_win, ring_k, ring_v, ring_g, ring_b, core_out, proj_dbg,
            o_dbg, cs_seq, cs_dim, cs_tok, rs_seq, cw_seq, cw_tok, cw_q, rk_seq,
            rv_seq, rg_seq, rb_seq, ring_len, layer, epoch, tl, prefix, blocks,
            blk_sm, blk_sr, ares_nw, ares_qk, in_nw, reqs, aux,
        )  # fmt: skip
        kda_body(key, smem, role, scratch, dict(zip(KDA_ARGS, params)))

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
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        reqs: Int32,
        aux: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kda_wide(
            x, w_in, w_fb, conv_w, conv_st, a_log, dt_bias, on_w, rstate, st_idx,
            win_row, conv_win, ring_k, ring_v, ring_g, ring_b, core_out, proj_dbg,
            o_dbg, cs_seq, cs_dim, cs_tok, rs_seq, cw_seq, cw_tok, cw_q, rk_seq,
            rv_seq, rg_seq, rb_seq, ring_len, scratch, layer, epoch, tl, prefix,
            blocks, blk_sm, blk_sr, ares_nw, ares_qk, in_nw, reqs, aux,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    _BUILDS[key] = launch
    return launch


def ptr(t):
    return 0 if t is None else t.data_ptr()


def kda_wide(key: KdaWideBuild, **kw):
    """Host launcher; the arguments as ``launch_values`` takes them."""
    build(key)(*launch_values(key, **kw), stream=torch.cuda.current_stream())


def launch_values(
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
    prefix=None,
    blocks=None,
    ares_nw=None,
    ares_qk=None,
    in_nw=None,
    aux=None,
):
    """The launch's kernel arguments (``KDA_ARGS`` with the scratch, in the
    kernel's order), checked. ``x`` the input-normed layer input [S, 7168] bf16 as
    ``pack_x`` lays it out (a ``front`` build writes it: [16 tiles, 7168], its
    rows past S zero or anything; then ``prefix`` [S, 7168] bf16 is the stream,
    ``blocks`` the AttnRes bank [S, >= nblocks (+ 1 for a write), 7168] bf16 and
    ``ares_nw`` / ``ares_qk`` / ``in_nw`` the seam's and input_layernorm's
    weights);
    ``w_in`` in_proj as ``pack_w_in`` lays it out;
    ``conv_state`` [slot, channel, entry] by its strides (SGLang's conv cache
    [slot, CONV_W - 1, C] is its ``.mT``), read only; ``rstate`` [slots, 12, 128,
    128] fp32, read only; ``st_idx`` int32 [R]: each request's slot (< 0: none);
    ``conv_win`` [row, >= L, CONV_W - 1, C] gets request r's verify windows at
    ``win_row[r]`` (int32 [R], < 0: none); ``ring`` = (k, v, g, beta) [slots, 12,
    ring_len, (128)]; ``scratch`` (``scratch_bytes``) zeroed once; ``epoch`` a
    device int32 bumped once a step, ``layer`` this launch's slot; ``aux`` the
    drafter's aux_hidden_states_buffer [>= S, 5 x 7168] bf16 (``aux_slot``)."""
    s, L = key.tokens, key.qlen
    R = st_idx.numel()
    assert R <= gemm_reqs(key) and (is_gemm(key) or R == s // L)
    assert x.shape == (16 * tiles(key), HIDDEN) and x.dtype == torch.bfloat16 and x.is_contiguous()
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
    assert core_out.shape == (R * L, PROJ) and core_out.is_contiguous()
    assert key.debug == (proj_dbg is not None) == (o_dbg is not None)
    assert key.timeline == (tl is not None)
    blk_sm = blk_sr = 0
    if key.front:
        assert prefix.shape == (s, HIDDEN) and prefix.dtype == torch.bfloat16
        assert prefix.is_contiguous()
        assert blocks.dtype == torch.bfloat16 and blocks.size(0) >= s
        assert blocks.size(2) == HIDDEN and blocks.stride(2) == 1
        assert blocks.size(1) > max(key.nblocks - 1, key.write_idx)
        blk_sm, blk_sr = blocks.stride(0), blocks.stride(1)
    else:
        assert prefix is None and blocks is None
    assert (key.aux_slot >= 0) == (aux is not None) and (aux is None or key.front)
    if aux is not None:
        assert aux.dtype == torch.bfloat16 and aux.is_contiguous()
        assert aux.shape[0] >= s and aux.shape[1] == AUX_ROW
    return (
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
        ptr(prefix),
        ptr(blocks),
        blk_sm,
        blk_sr,
        ptr(ares_nw),
        ptr(ares_qk),
        ptr(in_nw),
        R,
        ptr(aux),
    )
