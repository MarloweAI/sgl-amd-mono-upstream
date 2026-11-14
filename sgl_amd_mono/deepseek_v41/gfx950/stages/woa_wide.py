# SPDX-License-Identifier: Apache-2.0
"""wo_a at wide row counts (49 .. 384 rows), bit for bit SGLang's gfx950 route.

Inputs, for the rank's ``g`` o-groups:
- x: the attention output after the inverse RoPE, bf16 [m, g, 4096] (row stride g * 4096);
- W: the checkpoint's e4m3 wo_a [g * 1024, 4096] (``weights.mxfp8_exact``: the weight SGLang
  dequantizes to bf16) in ``pack_weight``'s fragment order, with its E8M0 32 x 32 block
  codes [g * 32, 128] (row-major).

Output: wo_b's MXFP8 input, e4m3 [m, g * 1024] and one ue8m0 byte a 32 columns [m, g * 32],
out[t, 1024 j + r] = x[t, j] . W[1024 j + r].

A task is a [bm token rows x bn columns] tile of one group over all of K, on wm x wn waves
of (bm / wm) x (bn / wn) each. A wave loads its operands a 64-wide K chunk at a time,
``depth`` chunks ahead: its x rows as whole 128 B row pieces, staged through its own LDS
slots into MFMA fragments, and its W fragments straight into registers (one 1 KB run a
column tile), widened to bf16 at their block code (exact: a power-of-two scale), so the
MFMAs see SGLang's bf16 weight. Each 16 x 16 accumulator chains one
``mfma_f32_16x16x32_bf16`` a 32-wide K step, in K order from zero, lane group l / 16
holding K 8 (l / 16) .. + 8 of the step: the order of the Triton batched GEMM SGLang runs
(``gfx95_batched_gemm_bf16_fp8_grid``, one tile walk over K), so the f32 sums are its sums. The wave then applies that kernel's output quant to its 32-column
groups: round to bf16, scale by the smallest power of two >= amax / 448 (amax floored at
1e-10), e4m3 RNE. No K is split and the waves never wait on each other; waves of a task
that read the same W rows meet in L1.

With no K split, a CU reads K (2 bm + bn) bytes for its bm x bn outputs: at these widths
the kernel runs at the rate a CU takes in that stream from L2, not at its MFMA rate.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T

from sgl_amd_mono.common.mx import clamp_fp8, code_ceil, rcp_pow2
from sgl_amd_mono.common.ops import (
    CM_DEV,
    bf16_round,
    butterfly,
    fp8_pack4,
    mfma_bf16,
    rsrc,
)
from sgl_amd_mono.common.plan import kernel_sources, key_tuple
from sgl_amd_mono.common.sync import _spin
from sgl_amd_mono.deepseek_v41.gfx950.attention.device import fp8x8_bf16, pow2

SOURCES = kernel_sources(__name__)

R, D = 1024, 4096  # a group's wo_a rows and K
KB = D // 32  # K steps, and E8M0 codes a W row block
AMAX_FLOOR = 1e-10
CH = 64  # K a chunk: two MFMA steps
XW = CH // 2 + 4  # i32 words an x row of a chunk in LDS: 64 bf16 + 16 B of pad
WT = 512  # i32 words a W column tile's chunk in LDS, as bf16: [step][lane][16 B]
FLAG = 16  # i32 words a split flag: a line of its own


class Tile(msgspec.Struct, frozen=True):
    """A width class's static task shape: bm x bn tiles on wm x wn waves, loads ``depth``
    K chunks (64 wide) ahead."""

    bm: int
    bn: int
    wm: int
    wn: int
    depth: int = 4
    # ``woa_task_shared``: the CTA stages x and W once, in LDS
    shared: bool = False
    # K split (``shared`` or ``wring``): f32 partials, reduced in split order
    ks: int = 1
    # ``woa_task_ws``: wprod W waves stream W into an LDS ring of this many chunks,
    # wdepth groups (wring / 2 chunks each) of loads ahead
    wring: int = 0
    wprod: int = 1
    wdepth: int = 2

    def __post_init__(self):
        assert self.bm % (16 * self.wm) == 0 and self.bn % (32 * self.wn) == 0
        assert R % self.bn == 0 and (self.shared or self.wring or self.ks == 1)
        assert not (self.shared and self.wring) and self.wring % 2 == 0
        assert (self.bn // 16) % self.wprod == 0
        assert (D // CH // self.ks) % 2 == 0 and self.bm % self.ks == 0

    @property
    def threads(self) -> int:
        return 64 * (self.wm * self.wn + (self.wprod if self.wring else 0))

    @property
    def lds_words(self) -> int:
        """i32 words of LDS: each MFMA wave's two x chunk slots, or (shared) the CTA's
        two x and two bf16 W chunk slots and its W block codes."""
        if self.shared:
            return (
                2 * self.bm * XW
                + 2 * (self.bn // 16) * WT
                + (self.bn // 32) * (KB // 4)
            )
        # and (wring) the W ring: wring chunks of every column tile, 1 KB each
        ring = self.wring * (self.bn // 16) * (WT // 2)
        return self.wm * self.wn * 2 * (self.bm // self.wm) * XW + ring


# the width classes: up to 96, 192 and 384 rows, at 4 groups (TP2) and 2 (TP4)
TILES = {
    (4, 96): Tile(48, 32, 3, 1, 4),
    (4, 192): Tile(48, 64, 3, 1, 4),
    (4, 384): Tile(96, 64, 3, 1, 2),
    (2, 96): Tile(32, 32, 2, 1, 4),
    (2, 192): Tile(48, 32, 3, 1, 4),
    (2, 384): Tile(48, 64, 3, 1, 4),
}


def width_class(m: int) -> int:
    for w in (96, 192, 384):
        if m <= w:
            return w
    raise ValueError(f"wo_a wide: {m} rows past 384")


def _cdiv(a, b):
    return (a + b - 1) // b


def pack_weight(w):
    """e4m3 [n, 4096] -> fragment order: per 16-row tile and 64-wide K chunk, lane l's
    16 bytes at 16 l: row l % 16, K 8 (l / 16) .. of the chunk's first step, then of its
    second; a wave's chunk of a tile is one 1 KB run."""
    n, k = w.shape
    return (
        w.view(n // 16, 16, k // CH, 2, 4, 8)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(n, k)
    )


def _chunk_loads(lane, mt, nt, xr, xrows, wr, sr, wtiles, cblocks, q):
    """K chunk ``q``'s loads: x rows 16 t + 8 h + l / 8, 16 B each at K 8 (l % 8) (whole
    128 B rows), W's fragments (one 16 B piece a lane a column tile) and, every other
    chunk, the E8M0 word of each 32-row block (a vector load: loads return in order, where
    a scalar load's wait would drain every one in flight)."""
    xs = [
        fx.Vector(bo.buffer_load(xr, xrows[i] + 32 * q, vec_width=4, dtype=T.i32))
        for i in range(2 * mt)
    ]
    ws = [
        fx.Vector(bo.buffer_load(wr, wtiles[b] + 256 * q, vec_width=4, dtype=T.i32))
        for b in range(nt)
    ]
    codes = None
    if q % 2 == 0:
        codes = [
            fx.Int32(
                bo.buffer_load(sr, cb * (KB // 4) + q // 2, vec_width=1, dtype=T.i32)
            )
            for cb in cblocks
        ]
    return xs, ws, codes


def woa_task(tid, tl, x, ldx, m, m0, w, ws, n0, out, osc, ncols, lds):
    """Task (m0, n0): rows m0 .. m0 + bm (masked to m) of columns n0 .. n0 + bn, a
    group's columns, into ``out`` (e4m3 [m, ncols]) and ``osc`` (ue8m0 [m, ncols / 32]).
    ``lds``: ``tl.lds_words`` i32 words. W is in ``pack_weight``'s order. A wave stages
    its own x rows through its LDS slots (coalesced loads in, MFMA fragments out), so the
    waves never wait on each other."""
    lane, wave = tid % 64, tid // 64
    tm, tn = tl.bm // tl.wm, tl.bn // tl.wn
    mt, nt = tm // 16, tn // 16
    wy, wx = wave // tl.wn, wave % tl.wn
    r0, c0 = m0 + wy * tm, n0 + wx * tn
    xr, wr, sr = rsrc(x), rsrc(w), rsrc(ws)
    xcol = (n0 // R) * D
    g = lane // 16
    # i32 word offsets at K 0: x row 8 i + l / 8 of the wave's, piece l % 8; W lane l's piece
    xrows = [
        (fx.min(r0 + 8 * i + lane // 8, m - 1) * ldx + xcol + 8 * (lane % 8)) // 2
        for i in range(2 * mt)
    ]
    wtiles = [((c0 + 16 * b) * D + 16 * lane) // 4 for b in range(nt)]
    cblocks = [(c0 + 32 * c) // 32 for c in range(nt // 2)]
    slot = 2 * tm * XW
    xl = lds + wave * slot
    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nt)] for _ in range(mt)]
    nq = D // CH

    def loads(q):
        return _chunk_loads(lane, mt, nt, xr, xrows, wr, sr, wtiles, cblocks, q)

    def stage(xs, q):
        base = xl + (q % 2) * tm * XW
        for i in range(2 * mt):
            fx.ptr_store(xs[i], base + (8 * i + lane // 8) * XW + 4 * (lane % 8))

    def frags(q):
        base = xl + (q % 2) * tm * XW
        return [
            [
                fx.Vector(
                    fx.ptr_load(
                        base + (16 * t + lane % 16) * XW + 16 * j + 4 * g,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                ).bitcast(fx.BFloat16)
                for t in range(mt)
            ]
            for j in range(2)
        ]

    pend = [loads(q) for q in range(min(tl.depth, nq))]
    words = None
    rocdl.sched_barrier(0)
    cur = pend.pop(0)
    stage(cur[0], 0)
    if tl.depth < nq:
        pend.append(loads(tl.depth))
    for q in range_constexpr(nq):
        xf = frags(q)
        wv, codes = cur[1], cur[2]
        if q + 1 < nq:
            cur = pend.pop(0)
            stage(cur[0], q + 1)
            if q + 1 + tl.depth < nq:
                pend.append(loads(q + 1 + tl.depth))
        rocdl.sched_barrier(0)
        if codes is not None:
            words = codes
        for j in range_constexpr(2):
            kk = 2 * q + j
            a = []
            for b in range_constexpr(nt):
                s = pow2((words[b // 2] >> (8 * (kk % 4))) & 0xFF)
                a.append(
                    fp8x8_bf16(wv[b][2 * j], wv[b][2 * j + 1], s).bitcast(fx.BFloat16)
                )
            for t in range_constexpr(mt):
                for b in range_constexpr(nt):
                    acc[t][b] = mfma_bf16(a[b], xf[j][t], acc[t][b])
        rocdl.sched_barrier(0)
    _quant(lane, acc, mt, nt, r0, c0, m, out, osc, ncols)


def _shared_loads(tid, tl, xr, ldx, xcol, m, m0, wr, n0, q):
    """K chunk ``q``'s 16 B pieces for the whole CTA: x rows m0 + e / 8 (clamped to m), piece
    e % 8; W column tile e / 64, lane piece e % 64 (``pack_weight``'s 1 KB runs)."""
    nt = tl.threads
    xs = []
    for i in range_constexpr(_cdiv(tl.bm * 8, nt)):
        e = fx.min(tid + nt * i, tl.bm * 8 - 1)
        row = fx.min(m0 + e // 8, m - 1)
        word = (row * ldx + xcol + CH * q + 8 * (e % 8)) // 2
        xs.append((e, fx.Vector(bo.buffer_load(xr, word, vec_width=4, dtype=T.i32))))
    ws = []
    tot = (tl.bn // 16) * 64
    for i in range_constexpr(_cdiv(tot, nt)):
        e = fx.min(tid + nt * i, tot - 1)
        word = ((n0 + 16 * (e // 64)) * D) // 4 + 256 * q + 4 * (e % 64)
        ws.append((e, fx.Vector(bo.buffer_load(wr, word, vec_width=4, dtype=T.i32))))
    return xs, ws


def woa_task_shared(tid, tl, x, ldx, m, m0, w, ws, n0, out, osc, ncols, lds, split):
    """Task (m0, n0) over K split ``split`` (``tl.ks`` splits) with both operands staged once
    for the CTA: each chunk's x rows and W tiles go through two LDS slots (W widened to bf16
    on the way in), one barrier a chunk; the wm x wn waves read their fragments from there.
    With ks = 1 the result is ``woa_task``'s, bit for bit (the same chains). With ks > 1
    the caller reduces the partials (``woa_reduce``); returns the accumulators."""
    lane, wave = tid % 64, tid // 64
    tm, tn = tl.bm // tl.wm, tl.bn // tl.wn
    mt, nt = tm // 16, tn // 16
    nb = tl.bn // 16
    wy, wx = wave // tl.wn, wave % tl.wn
    xr, wr, sr = rsrc(x), rsrc(w), rsrc(ws)
    xcol = (n0 // R) * D
    g = lane // 16
    xl = lds
    wl = lds + 2 * tl.bm * XW
    sl = wl + 2 * nb * WT
    nqs = D // CH // tl.ks
    q0 = split * nqs
    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nt)] for _ in range(mt)]

    def loads(q):
        return _shared_loads(tid, tl, xr, ldx, xcol, m, m0, wr, n0, q0 + q)

    def stage(pieces, q):
        xs_, ws_ = pieces
        xb = xl + (q % 2) * tl.bm * XW
        for e, v in xs_:
            fx.ptr_store(v, xb + (e // 8) * XW + 4 * (e % 8))
        wb = wl + (q % 2) * nb * WT
        for e, v in ws_:
            b, l = e // 64, e % 64
            # q0 is even (nqs is), so the step's byte in its code word is static
            word = fx.ptr_load(sl + (16 * b // 32) * (KB // 4) + q0 // 2 + q // 2)
            for j in range_constexpr(2):
                s = pow2((word >> (8 * ((2 * q + j) % 4))) & 0xFF)
                fx.ptr_store(
                    fp8x8_bf16(v[2 * j], v[2 * j + 1], s), wb + b * WT + 256 * j + 4 * l
                )

    def frags(q):
        xb = xl + (q % 2) * tl.bm * XW
        wb = wl + (q % 2) * nb * WT
        v4 = fx.Vector.make_type(4, fx.Int32)
        xf = [
            [
                fx.Vector(
                    fx.ptr_load(
                        xb + (wy * tm + 16 * t + lane % 16) * XW + 16 * j + 4 * g,
                        result_type=v4,
                    )
                ).bitcast(fx.BFloat16)
                for t in range(mt)
            ]
            for j in range(2)
        ]
        wf = [
            [
                fx.Vector(
                    fx.ptr_load(
                        wb + (wx * nt + b) * WT + 256 * j + 4 * lane, result_type=v4
                    )
                ).bitcast(fx.BFloat16)
                for b in range(nt)
            ]
            for j in range(2)
        ]
        return xf, wf

    pend = [loads(q) for q in range(min(tl.depth, nqs))]
    for e in _pieces(tid, tl.threads, (tl.bn // 32) * (KB // 4)):
        cw = fx.Int32(
            bo.buffer_load(sr, (n0 // 32) * (KB // 4) + e, vec_width=1, dtype=T.i32)
        )
        fx.ptr_store(cw, sl + e)
    rocdl.sched_barrier(0)
    gpu.barrier()
    stage(pend.pop(0), 0)
    if tl.depth < nqs:
        pend.append(loads(tl.depth))
    gpu.barrier()
    for q in range_constexpr(nqs):
        xf, wf = frags(q)
        if q + 1 < nqs:
            stage(pend.pop(0), q + 1)
            if q + 1 + tl.depth < nqs:
                pend.append(loads(q + 1 + tl.depth))
        rocdl.sched_barrier(0)
        for j in range_constexpr(2):
            for t in range_constexpr(mt):
                for b in range_constexpr(nt):
                    acc[t][b] = mfma_bf16(wf[j][b], xf[j][t], acc[t][b])
        rocdl.sched_barrier(0)
        if q + 1 < nqs:
            gpu.barrier()
    return acc


def lds_barrier():
    """``s_barrier`` for an LDS hand-off between waves (as ``stages.moe``'s): fenced, so no
    LDS access moves across it, and the waves' loads in flight stay in flight."""
    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="workgroup")
    rocdl.s_barrier()
    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="workgroup")


def _w_stream(lane, tl, w, n0, ring, q0, nqs, p):
    """W wave ``p`` (of ``wprod``): its column tiles' chunks (``pack_weight``'s 1 KB runs,
    one 16 B piece a lane) into the LDS ring, a group of wring / 2 chunks a barrier, with
    ``wdepth`` groups of loads in flight in registers. On their own load queues, so the
    MFMA waves' x loads never wait behind W's HBM latency, and each W byte is read once."""
    nb = tl.bn // 16
    gsz = tl.wring // 2
    ngr = nqs // gsz
    wr = rsrc(w)
    # this wave's column tiles: p, p + wprod, ...
    mine = nb // tl.wprod
    voff = [((n0 + 16 * (p + tl.wprod * i)) * D + 16 * lane) // 4 for i in range(mine)]

    def load(gr):
        return [
            fx.Vector(
                bo.buffer_load(
                    wr, voff[i] + 256 * (q0 + gr * gsz + c), vec_width=4, dtype=T.i32
                )
            )
            for c in range(gsz)
            for i in range(mine)
        ]

    def write(vs, gr):
        for c in range_constexpr(gsz):
            q = gr * gsz + c
            for i in range_constexpr(mine):
                at = (
                    ring
                    + ((q % tl.wring) * nb + p + tl.wprod * i) * (WT // 2)
                    + 4 * lane
                )
                fx.ptr_store(vs[c * mine + i], at)

    pend = [load(gr) for gr in range(min(tl.wdepth, ngr))]
    for k in range_constexpr(ngr):
        # group k's slots held group k - 2, consumed before barrier k - 1
        write(pend.pop(0), k)
        if k + tl.wdepth < ngr:
            pend.append(load(k + tl.wdepth))
        lds_barrier()


def _consume(tid, tl, x, ldx, m, m0, ws, n0, lds, ring, q0, nqs):
    """The MFMA waves of ``woa_task_ws``: each stages its own x rows (as ``woa_task``) and
    takes W's fragments from the ring, a barrier a group; returns the accumulators."""
    lane, wave = tid % 64, tid // 64
    tm, tn = tl.bm // tl.wm, tl.bn // tl.wn
    mt, nt = tm // 16, tn // 16
    nb = tl.bn // 16
    wy, wx = wave // tl.wn, wave % tl.wn
    r0, c0 = m0 + wy * tm, n0 + wx * tn
    xr, sr = rsrc(x), rsrc(ws)
    xcol = (n0 // R) * D
    g = lane // 16
    gsz = tl.wring // 2
    xrows = [
        (fx.min(r0 + 8 * i + lane // 8, m - 1) * ldx + xcol + 8 * (lane % 8)) // 2
        for i in range(2 * mt)
    ]
    cblocks = [(c0 + 32 * c) // 32 for c in range(nt // 2)]
    xl = lds + wave * 2 * tm * XW
    v4 = fx.Vector.make_type(4, fx.Int32)
    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nt)] for _ in range(mt)]

    def loads(q):
        xs = [
            fx.Vector(
                bo.buffer_load(xr, xrows[i] + 32 * (q0 + q), vec_width=4, dtype=T.i32)
            )
            for i in range(2 * mt)
        ]
        codes = None
        if q % 2 == 0:
            codes = [
                fx.Int32(
                    bo.buffer_load(
                        sr, cb * (KB // 4) + (q0 + q) // 2, vec_width=1, dtype=T.i32
                    )
                )
                for cb in cblocks
            ]
        return xs, codes

    def stage(xs, q):
        base = xl + (q % 2) * tm * XW
        for i in range(2 * mt):
            fx.ptr_store(xs[i], base + (8 * i + lane // 8) * XW + 4 * (lane % 8))

    def frags(q):
        base = xl + (q % 2) * tm * XW
        xf = [
            [
                fx.Vector(
                    fx.ptr_load(
                        base + (16 * t + lane % 16) * XW + 16 * j + 4 * g,
                        result_type=v4,
                    )
                ).bitcast(fx.BFloat16)
                for t in range(mt)
            ]
            for j in range(2)
        ]
        wv = [
            fx.Vector(
                fx.ptr_load(
                    ring + ((q % tl.wring) * nb + wx * nt + b) * (WT // 2) + 4 * lane,
                    result_type=v4,
                )
            )
            for b in range(nt)
        ]
        return xf, wv

    pend = [loads(q) for q in range(min(tl.depth, nqs))]
    words = None
    rocdl.sched_barrier(0)
    cur = pend.pop(0)
    stage(cur[0], 0)
    if tl.depth < nqs:
        pend.append(loads(tl.depth))
    for q in range_constexpr(nqs):
        if q % gsz == 0:
            lds_barrier()  # this group's W landed
        xf, wv = frags(q)
        codes = cur[1]
        if q + 1 < nqs:
            cur = pend.pop(0)
            stage(cur[0], q + 1)
            if q + 1 + tl.depth < nqs:
                pend.append(loads(q + 1 + tl.depth))
        rocdl.sched_barrier(0)
        if codes is not None:
            words = codes
        for j in range_constexpr(2):
            kk = 2 * q + j  # q0 is even: the step's byte in its code word is static
            a = []
            for b in range_constexpr(nt):
                s = pow2((words[b // 2] >> (8 * (kk % 4))) & 0xFF)
                a.append(
                    fp8x8_bf16(wv[b][2 * j], wv[b][2 * j + 1], s).bitcast(fx.BFloat16)
                )
            for t in range_constexpr(mt):
                for b in range_constexpr(nt):
                    acc[t][b] = mfma_bf16(a[b], xf[j][t], acc[t][b])
        rocdl.sched_barrier(0)
    return acc


@flyc.jit
def woa_task_ws(
    tid, tl, x, ldx, m, m0, w, ws, n0, out, osc, ncols, lds, part, flags, tile, split
):
    """``woa_task``'s sums (bit for bit at ks = 1) with W off the MFMA waves: wave
    wm * wn streams W into an LDS ring (``_w_stream``), the others consume it
    (``_consume``); with ks > 1 the K split and its reduce as ``woa_reduce``."""
    nw = tl.wm * tl.wn
    wave = tid // 64
    nqs = D // CH // tl.ks
    q0 = split * nqs
    ring = lds + nw * 2 * (tl.bm // tl.wm) * XW
    if wave >= nw:
        _w_stream(tid % 64, tl, w, n0, ring, q0, nqs, wave - nw)
    else:
        acc = _consume(tid, tl, x, ldx, m, m0, ws, n0, lds, ring, q0, nqs)
        if const_expr(tl.ks == 1):
            tm, tn = tl.bm // tl.wm, tl.bn // tl.wn
            _quant(
                tid % 64, acc, tm // 16, tn // 16, m0 + (wave // tl.wn) * tm,
                n0 + (wave % tl.wn) * tn, m, out, osc, ncols,
            )  # fmt: skip
        else:
            store_partial(tid, tl, acc, part, tile, split)
    if const_expr(tl.ks > 1):
        woa_reduce(tid, tl, None, m, m0, n0, out, osc, ncols, part, flags, tile, split)


def _pieces(tid, nt, tot):
    for i in range_constexpr(_cdiv(tot, nt)):
        yield fx.min(tid + nt * i, tot - 1) if tot % nt else tid + nt * i


def store_partial(tid, tl, acc, part, tile, split):
    """A wave's accumulators as rows of the tile's f32 partial [bm][bn] (device scope)."""
    lane, wave = tid % 64, tid // 64
    tm, tn = tl.bm // tl.wm, tl.bn // tl.wn
    wy, wx = wave // tl.wn, wave % tl.wn
    mine = (tile * tl.ks + split) * tl.bm * tl.bn
    for t in range_constexpr(tm // 16):
        for b in range_constexpr(tn // 16):
            row = wy * tm + 16 * t + lane % 16
            col = wx * tn + 16 * b + 4 * (lane // 16)
            bo.buffer_store(
                acc[t][b], rsrc(part), mine + row * tl.bn + col, cache_modifier=CM_DEV
            )


@flyc.jit
def woa_reduce(tid, tl, acc, m, m0, n0, out, osc, ncols, part, flags, tile, split):
    """K-split epilogue: this CTA's f32 partial to ``part`` (device scope), then its flag
    (a phase bit: the opposite of what it held, so no reset is needed); once every split's
    flag of the tile flipped, this CTA reduces rows split * bm / ks .. of the tile: the
    partials summed in split order (the same bits whichever CTA comes last), rounded to
    bf16 and quantized as ``_quant`` does."""
    lane, wave = tid % 64, tid // 64
    tm, tn = tl.bm // tl.wm, tl.bn // tl.wn
    mt, nt = tm // 16, tn // 16
    wy, wx = wave // tl.wn, wave % tl.wn
    g = lane // 16
    size = tl.bm * tl.bn
    r_p, r_f = rsrc(part), rsrc(flags)
    mine = (tile * tl.ks + split) * size
    old = fx.Int32(
        bo.buffer_load(
            r_f,
            (tile * tl.ks + split) * FLAG,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )
    if const_expr(acc is not None):
        store_partial(tid, tl, acc, part, tile, split)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    phase = 1 - old
    if tid == 0:
        bo.buffer_store(
            phase, r_f, (tile * tl.ks + split) * FLAG, cache_modifier=CM_DEV
        )
    if tid < tl.ks:

        def load_all():
            return fx.Int32(
                bo.buffer_load(
                    r_f,
                    (tile * tl.ks + tid) * FLAG,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )

        _spin(load_all, lambda v: v != phase, guarded=False)
    gpu.barrier()
    rows = tl.bm // tl.ks
    per_row = tl.bn // 4
    tot = rows * per_row
    r_o, r_s = rsrc(out, m * ncols), rsrc(osc, m * ncols // 32)
    v4 = fx.Vector.make_type(4, fx.Float32)
    for i in range_constexpr(_cdiv(tot, tl.threads)):
        idx = tid + tl.threads * i
        live = None if tot % tl.threads == 0 else idx < tot
        idx = idx if live is None else fx.min(idx, tot - 1)
        row, c4 = split * rows + idx // per_row, idx % per_row
        off = row * tl.bn + 4 * c4
        v = fx.Vector(
            bo.buffer_load(
                r_p,
                tile * tl.ks * size + off,
                vec_width=4,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )
        for u in range_constexpr(1, tl.ks):
            v = v + fx.Vector(
                bo.buffer_load(
                    r_p,
                    (tile * tl.ks + u) * size + off,
                    vec_width=4,
                    dtype=T.f32,
                    cache_modifier=CM_DEV,
                )
            )
        ys = [bf16_round(v[e]) for e in range(4)]
        amax = fx.max(fx.max(abs(ys[0]), abs(ys[1])), fx.max(abs(ys[2]), abs(ys[3])))
        amax = butterfly(amax, (1, 2, 4), fx.max)
        code = code_ceil(fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / 448.0))
        mul = rcp_pow2(code)
        word = fp8_pack4(*[clamp_fp8(y * mul) for y in ys])
        grow = m0 + row
        ok = grow < m
        ok = ok if live is None else ok & live
        col = n0 + 4 * c4
        bo.buffer_store(word, r_o, (grow * ncols + col) // 4, mask=ok)
        bo.buffer_store(
            fx.Int8(code),
            r_s,
            grow * (ncols // 32) + col // 32,
            mask=ok & (lane % 8 == 0),
        )


def _quant(lane, acc, mt, nt, r0, c0, m, out, osc, ncols):
    """The fp8-grid quant of the wave's accumulators, a 32-column group per column-tile
    pair: accumulator (t, b) lane l element e is column 16 b + 4 (l / 16) + e of token
    row 16 t + l % 16."""
    # sized: a masked store goes to an out-of-range offset
    r_o, r_s = rsrc(out, m * ncols), rsrc(osc, m * ncols // 32)
    g = lane // 16
    for t in range_constexpr(mt):
        row = r0 + 16 * t + lane % 16
        ok = row < m
        for c in range_constexpr(nt // 2):
            ys = [
                [bf16_round(acc[t][2 * c + h][e]) for e in range(4)] for h in range(2)
            ]
            amax = fx.Float32(0.0)
            for v in ys[0] + ys[1]:
                amax = fx.max(amax, abs(v))
            amax = butterfly(amax, (16, 32), fx.max)
            code = code_ceil(
                fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / 448.0)
            )
            mul = rcp_pow2(code)
            for h in range_constexpr(2):
                word = fp8_pack4(*[clamp_fp8(v * mul) for v in ys[h]])
                col = c0 + 32 * c + 16 * h + 4 * g
                bo.buffer_store(word, r_o, (row * ncols + col) // 4, mask=ok)
            bo.buffer_store(
                fx.Int8(code),
                r_s,
                row * (ncols // 32) + (c0 + 32 * c) // 32,
                mask=ok & (g == 0),
            )


def task_origin(bid, tl, mtiles, nblk):
    """CTA ``bid``'s (n0, m0, tile, split). CTAs go round-robin over the 8 XCDs, so
    consecutive tasks are dealt to one XCD: a tile's K splits, and a group's row tile
    (its x rows), reread from that XCD's L2 by its column tiles."""
    task = (bid % 8) * (nblk // 8) + bid // 8 if nblk % 8 == 0 else bid
    tile, split = task // tl.ks, task % tl.ks
    ntiles = R // tl.bn
    grp, rem = tile // (mtiles * ntiles), tile % (mtiles * ntiles)
    return grp * R + (rem % ntiles) * tl.bn, (rem // ntiles) * tl.bm, tile, split


def workspace(m, g, tl):
    """(f32 words of K-split partials, i32 flag words) a launch needs (zeros once)."""
    tiles = g * _cdiv(m, tl.bm) * (R // tl.bn)
    if tl.ks == 1:
        return 1, 1
    return tiles * tl.ks * tl.bm * tl.bn, tiles * tl.ks * FLAG


_BUILDS: dict = {}


def standalone(m, g, tile=None, ldx=None):
    """One task a CTA (``task_origin``): ``launch(x, w, ws, out, osc, part, flags)`` on
    device addresses (see the module docstring for the layouts; ``ldx``: x's row stride in
    elements, g * 4096 when its rows are contiguous; ``part`` and ``flags`` per
    ``workspace``, flags zeroed once; unused when ks = 1)."""
    tl = tile if tile is not None else TILES[(g, width_class(m))]
    ldx = g * D if ldx is None else ldx
    keyed = key_tuple({"m": m, "g": g, "tile": tl, "ldx": ldx}, SOURCES)
    key = (m, g, tl, ldx)
    if key in _BUILDS:
        return _BUILDS[key]
    ncols = g * R
    mtiles = _cdiv(m, tl.bm)
    nblk = g * mtiles * (R // tl.bn) * tl.ks
    # split tasks wait on each other: every CTA must be resident at once
    assert tl.ks == 1 or nblk <= 256, nblk
    name = (
        f"ds41_woa_wide_m{m}_g{g}_t{tl.bm}x{tl.bn}_w{tl.wm}x{tl.wn}_d{tl.depth}"
        f"{'_s' if tl.shared else ''}_k{tl.ks}_r{tl.wring}p{tl.wprod}d{tl.wdepth}"
    )

    @fx.struct
    class Smem:
        x: fx.Array[fx.Int32, tl.lds_words, 16]

    @flyc.kernel(name=name, known_block_size=[tl.threads, 1, 1])
    def woa_kernel(
        x: Int64, w: Int64, ws: Int64, out: Int64, osc: Int64, part: Int64, flags: Int64
    ):
        tid = fx.thread_idx.x
        n0, m0, tile, split = task_origin(fx.block_idx.x, tl, mtiles, nblk)
        lds = fx.SharedAllocator().allocate(Smem).peek().x.ptr
        if const_expr(tl.wring):
            woa_task_ws(
                tid, tl, x, ldx, m, m0, w, ws, n0, out, osc, ncols, lds, part, flags,
                tile, split,
            )  # fmt: skip
        elif const_expr(not tl.shared):
            woa_task(tid, tl, x, ldx, m, m0, w, ws, n0, out, osc, ncols, lds)
        else:
            acc = woa_task_shared(
                tid, tl, x, ldx, m, m0, w, ws, n0, out, osc, ncols, lds, split
            )
            if const_expr(tl.ks == 1):
                tm, tn = tl.bm // tl.wm, tl.bn // tl.wn
                wave = tid // 64
                _quant(
                    tid % 64, acc, tm // 16, tn // 16, m0 + (wave // tl.wn) * tm,
                    n0 + (wave % tl.wn) * tn, m, out, osc, ncols,
                )  # fmt: skip
            else:
                woa_reduce(
                    tid, tl, acc, m, m0, n0, out, osc, ncols, part, flags, tile, split
                )

    @flyc.jit
    def launch(
        x: Int64,
        w: Int64,
        ws: Int64,
        out: Int64,
        osc: Int64,
        part: Int64,
        flags: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = keyed
        woa_kernel(x, w, ws, out, osc, part, flags).launch(
            grid=(nblk,), block=(tl.threads,), stream=stream
        )

    _BUILDS[key] = launch
    return launch
