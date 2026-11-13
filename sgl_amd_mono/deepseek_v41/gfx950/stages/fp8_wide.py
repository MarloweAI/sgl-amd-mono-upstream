# SPDX-License-Identifier: Apache-2.0
"""The attention's dense FP8 projections (wqkv_a, wq_b, wo_b) at wide row counts (60 .. 384
rows): out[m, N] bf16 = x[m, K] . W[N, K]^T on the gfx950 scaled MFMA.

Operands, as SGLang's gfx950 route holds them:
- x: e4m3 [m, K] row-major and its ue8m0 codes [m, K / 32] (one a 32 K), the MXFP8
  activation the fused norms and wo_a emit;
- W: the checkpoint's e4m3 [N, K] row-major, with its E8M0 32 x 32 block codes laid out by
  ``pack_scales`` (one byte a lane a K step, see there).

A CTA computes a bm x bn tile of out over K / ks of K (ks splits), on wm x wn waves of
(bm / wm) x (bn / wn) each. K goes in 128-wide steps: each step's x rows and W rows land in
LDS by DMA (16 B a lane, 1 KB an instruction, the 16 B pieces of a 128 B row XORed by
(row / 2) % 8, so the 16 rows of an MFMA operand read different banks), ``depth`` steps in
flight; the step's codes land beside them. Every wave issues the same loads a step, so one
``s_waitcnt vmcnt`` counts a step in, and one barrier a step hands it over. A wave then
reads its operands: lane (r, g) of a 16-row tile holds row r, K 16 g .. 16 g + 16 and
64 + 16 g .. (the scaled MFMA's layout), and supplies the code of row r, K block g. Each
16 x 16 accumulator chains one ``mfma_scale_f32_16x16x128_f8f6f4`` a step (W first: lane
(r, g) holds out rows r, columns 4 g .. 4 g + 3).

With ks > 1 each CTA stores its f32 partial, flips its flag (a phase bit), waits for the
tile's other splits and then sums rows split * bm / ks .. of the tile in split order, from
zero, to bf16: the same bits whichever CTA arrives last. The splits wait on each other, so
a split launch needs every CTA resident at once.

On SGLang's served operands every tile and split here gives the bits of SGLang's aiter route
(its tuned Triton ``gemm_afp8wfp8``): the scaled MFMA sums these blocks exactly, so the
result does not depend on the K order (checked bit for bit, ``check_fp8_wide.py``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T, as_ir_value

from sgl_amd_mono.common.mx import mfma_scaled
from sgl_amd_mono.common.ops import CM_DEV, rsrc, uniform
from sgl_amd_mono.common.plan import kernel_sources, key_tuple
from sgl_amd_mono.common.sync import _spin

SOURCES = kernel_sources(__name__)

STEP = 128  # K a step: one scaled MFMA
ROWB = STEP  # bytes an operand row a step (e4m3)
FLAG = 16  # i32 words a split flag: a line of its own


class Tile(msgspec.Struct, frozen=True):
    """A width class's static task shape: bm x bn tiles on wm x wn waves, ``depth`` K steps
    in flight, K split ``ks`` ways."""

    bm: int
    bn: int
    wm: int
    wn: int
    ks: int = 1
    depth: int = 2
    # waves 0 .. xw - 1 land x, the others W (each on its own load queue, so x's L2 hits
    # never wait behind W's HBM misses); 0: every wave lands both
    xw: int = 0
    # W straight into the MFMA waves' registers, this many K steps ahead (each wave owns
    # its W columns, wm = 1), and xw more waves land x: W's HBM latency is covered by
    # registers, not LDS; 0: W through LDS
    wreg: int = 0

    def __post_init__(self):
        w = self.wm * self.wn
        assert self.bm % (16 * self.wm) == 0 and self.bn % (16 * self.wn) == 0
        # the waves of a role issue the same DMAs: whole 8-row instructions
        if self.wreg:
            assert self.wm == 1 and self.xw >= 1 and (self.bm // 8) % self.xw == 0, self
        elif self.xw:
            assert 0 < self.xw < w and (self.bm // 8) % self.xw == 0, self
            assert (self.bn // 8) % (w - self.xw) == 0, self
        else:
            assert (self.bm // 8) % w == 0 and (self.bn // 8) % w == 0, self
        assert self.bn <= 256 and self.depth >= 2 and self.bm % self.ks == 0
        # the epilogue's bf16 tile fits the stages' LDS
        assert self.bm * self.bn * 2 <= self.depth * self.stage_bytes

    @property
    def waves(self) -> int:
        return self.wm * self.wn

    @property
    def cthreads(self) -> int:
        """The MFMA waves' threads."""
        return 64 * self.waves

    @property
    def threads(self) -> int:
        return 64 * (self.waves + (self.xw if self.wreg else 0))

    @property
    def xs_rows(self) -> int:
        """x code rows a step lands: whole 64-row instructions."""
        return 64 * _cdiv(self.bm, 64)

    @property
    def stage_bytes(self) -> int:
        """A step's LDS: x rows, W rows, x codes (4 B a row), W codes (1 KB); x alone
        with ``wreg``."""
        if self.wreg:
            return self.bm * ROWB + 4 * self.xs_rows
        return (self.bm + self.bn) * ROWB + 4 * self.xs_rows + 1024

    @property
    def lds_words(self) -> int:
        return self.depth * self.stage_bytes // 4


def _cdiv(a, b):
    return (a + b - 1) // b


def pack_scales(codes: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """E8M0 32 x 32 block codes u8 [n / 32, k / 32] -> the kernel's: u8 [k / 128, n / 16, 64],
    step s, 16-row tile t, lane 16 g + r: the code of row 16 t + r, K block 4 s + g (one
    1 KB run per 16 tiles of a step)."""
    c = codes.view(torch.uint8).repeat_interleave(32, 0)[:n]  # [n, k / 32]
    return (
        c.view(n // 16, 16, k // STEP, 4)
        .permute(2, 0, 3, 1)
        .contiguous()
        .view(k // STEP, n // 16, 64)
    )


def _dma(rsrc_v, lds_ptr, voffset, wide=True):
    """``buffer_load_dword{x4,} ... lds``: 16 (``wide``) or 4 B a lane from ``rsrc_v`` at
    byte ``voffset`` into LDS at ``lds_ptr`` (wave-uniform) + 16 (4) lane, by inline asm, so
    the compiler's wait counting does not see it; the kernel counts them (loads retire in
    order)."""
    r = as_ir_value(rsrc_v)
    if "ptr<8>" in str(r.type):
        r = bo._ptr8_to_v4i32(r)
    rv = fx.Vector(r, (4,), fx.Int32)
    rs = fx.Vector.from_elements([uniform(fx.Int32(rv[i])) for i in range(4)], fx.Int32)
    base = uniform(fx.Int32(llvm.ptrtoint(T.i32, lds_ptr.llvm_ptr)))
    op = "buffer_load_dwordx4" if wide else "buffer_load_dword"
    llvm.inline_asm(
        None,
        [
            as_ir_value(rs),
            as_ir_value(base),
            as_ir_value(fx.Int32(voffset)),
            as_ir_value(uniform(fx.Int32(0))),
        ],
        f"s_mov_b32 m0, $1\n\ts_nop 0\n\t{op} $2, $0, $3 offen lds",  # codespell:ignore offen
        "s,s,v,s,~{m0},~{memory}",
        has_side_effects=True,
    )


def _step_barrier(vmcnt):
    """This wave's loads but the newest ``vmcnt`` and its LDS reads retired, then
    ``s_barrier``; no memory access moves across it."""
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({vmcnt}) lgkmcnt(0)\n\ts_barrier",
        constraints="~{memory}",
        has_side_effects=True,
    )


def _swz(row):
    """The XOR on a 128 B row's 16 B piece index."""
    return (row // 2) % 8


def task_origin(bid, tl, mtiles, ntiles):
    """CTA ``bid``'s (m0, n0, tile, split). CTAs go round-robin over the 8 XCDs; a tile's
    splits and a column tile's row tiles are dealt to one XCD (they share partials and W
    rows in its L2)."""
    nblk = mtiles * ntiles * tl.ks
    task = ((nblk % 8) == 0).select((bid % 8) * (nblk // 8) + bid // 8, bid)
    tile, split = task // tl.ks, task % tl.ks
    return (tile % mtiles) * tl.bm, (tile // mtiles) * tl.bn, tile, split


@flyc.jit
def gemm_task(tid, tl, n, k, m, m0, n0, split, x, xs, w, wsc, lds):
    """One CTA's f32 accumulators of its bm x bn tile over its K split (the module docstring)."""
    lane, wave = tid % 64, tid // 64
    nw = tl.waves
    tm, tn = tl.bm // tl.wm, tl.bn // tl.wn
    mt, nt = tm // 16, tn // 16
    wy, wx = wave // tl.wn, wave % tl.wn
    steps = k // STEP // tl.ks
    s0 = split * steps
    k32 = k // 32
    r_x, r_xs = rsrc(x, m * k), rsrc(xs, m * k32)
    r_w, r_ws = rsrc(w, n * k), rsrc(wsc, n * k32)
    sb = tl.stage_bytes // 4  # words a stage
    xo, wo = 0, tl.bm * ROWB // 4  # words of a stage: x rows, W rows,
    xso = (tl.bm + tl.bn) * ROWB // 4  # x codes,
    wso = xso + tl.xs_rows  # W codes
    # x's and W's landing waves: (count, this wave's index among them)
    if const_expr(tl.xw):
        xr, wr = (tl.xw, wave), (nw - tl.xw, wave - tl.xw)
    else:
        xr, wr = (nw, wave), (nw, wave)
    na, nb = tl.bm // 8 // xr[0], tl.bn // 8 // wr[0]
    # a DMA lane's row of an 8-row instruction and the piece it fetches for its LDS slot
    drow, dpiece = lane // 8, lane % 8
    xins = [xr[1] + xr[0] * j for j in range(na)]  # this wave's 8-row instructions
    wins = [wr[1] + wr[0] * j for j in range(nb)]
    x_src = [(m0 + 8 * i + drow) * k + 16 * (dpiece ^ _swz(8 * i + drow)) for i in xins]
    w_src = [(n0 + 8 * i + drow) * k + 16 * (dpiece ^ _swz(8 * i + drow)) for i in wins]
    # the x codes: 4 B (the step's 4 blocks) a row, 64 rows an instruction, x's waves
    # sharing the instructions (a duplicate writes the same bytes)
    nxs = _cdiv(tl.bm, 64)
    ncx = _cdiv(nxs, xr[0])  # code instructions an x wave
    xs_ins = [(xr[1] + xr[0] * j) % nxs for j in range(ncx)]
    xs_src = [fx.min(m0 + 64 * i + lane, m - 1) * k32 for i in xs_ins]
    # the W codes of the CTA's bn / 16 tiles, a 4 bn B run (lanes past it re-read it)
    ws_src = (n0 // 16) * 64 + (lane * 16) % (tl.bn * 4)

    def issue_x(st):
        slot = lds + (st % tl.depth) * sb
        for j in range_constexpr(na):
            _dma(r_x, slot + xo + xins[j] * 256, x_src[j] + (s0 + st) * STEP)
        for j in range_constexpr(ncx):
            _dma(r_xs, slot + xso + 64 * xs_ins[j], xs_src[j] + 4 * (s0 + st), wide=False)

    def issue_w(st):
        slot = lds + (st % tl.depth) * sb
        for j in range_constexpr(nb):
            _dma(r_w, slot + wo + wins[j] * 256, w_src[j] + (s0 + st) * STEP)
        _dma(r_ws, slot + wso, (s0 + st) * (n // 16) * 64 + ws_src)

    def issue(st):
        if const_expr(tl.xw):
            if wave < tl.xw:
                issue_x(st)
            else:
                issue_w(st)
        else:
            issue_x(st)
            issue_w(st)

    def land(ahead):
        """Wait until the step ``ahead`` steps behind the newest issued landed (every
        wave's part), then the barrier."""
        if const_expr(tl.xw):
            if wave < tl.xw:
                _step_barrier(ahead * (na + ncx))
            else:
                _step_barrier(ahead * (nb + 1))
        else:
            _step_barrier(ahead * (na + ncx + nb + 1))
    r, g = lane % 16, lane // 16
    v4 = fx.Vector.make_type(4, fx.Int32)

    def frag(base_words, row):
        p = base_words + row * (ROWB // 4)
        lo = fx.Vector(fx.ptr_load(p + 4 * (g ^ _swz(row)), result_type=v4))
        hi = fx.Vector(fx.ptr_load(p + 4 * ((4 + g) ^ _swz(row)), result_type=v4))
        return fx.Vector.from_elements(
            [lo[d] for d in range(4)] + [hi[d] for d in range(4)], fx.Int32
        )

    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nt)] for _ in range(mt)]
    for st in range_constexpr(min(tl.depth - 1, steps)):
        issue(st)
    for st in range_constexpr(steps):
        # step st landed for every wave, and step st - 1's slot is free
        land(min(tl.depth - 2, steps - 1 - st))
        if const_expr(st + tl.depth - 1 < steps):
            issue(st + tl.depth - 1)
        slot = lds + (st % tl.depth) * sb
        xf = [frag(slot + xo, wy * tm + 16 * a + r) for a in range(mt)]
        wf = [frag(slot + wo, wx * tn + 16 * b + r) for b in range(nt)]
        xsc = [
            (fx.Int32(fx.ptr_load(slot + xso + wy * tm + 16 * a + r)) >> (8 * g)) & 0xFF
            for a in range(mt)
        ]
        wsw = [
            fx.Int32(fx.ptr_load(slot + wso + ((wx * tn + 16 * b) // 16) * 16 + lane // 4))
            for b in range(nt)
        ]
        wsc_ = [(wsw[b] >> (8 * (lane % 4))) & 0xFF for b in range(nt)]
        for a in range_constexpr(mt):
            for b in range_constexpr(nt):
                acc[a][b] = mfma_scaled(wf[b], xf[a], acc[a][b], wsc_[b], xsc[a])
    return acc


def _x_lander(tid, tl, k, m, m0, s0, x, xs, lds, nw, wi):
    """(issue(st), per-step load count) of x's landing for landing wave ``wi`` of ``nw``:
    its 8-row instructions of the step's x rows and its share of their codes."""
    lane = tid % 64
    k32 = k // 32
    r_x, r_xs = rsrc(x, m * k), rsrc(xs, m * k32)
    sb = tl.stage_bytes // 4
    xso = tl.bm * ROWB // 4
    na = tl.bm // 8 // nw
    drow, dpiece = lane // 8, lane % 8
    ins = [wi + nw * j for j in range(na)]
    src = [(m0 + 8 * i + drow) * k + 16 * (dpiece ^ _swz(8 * i + drow)) for i in ins]
    nxs = _cdiv(tl.bm, 64)
    ncx = _cdiv(nxs, nw)
    xs_ins = [(wi + nw * j) % nxs for j in range(ncx)]
    xs_src = [fx.min(m0 + 64 * i + lane, m - 1) * k32 for i in xs_ins]

    def issue(st):
        slot = lds + (st % tl.depth) * sb
        for j in range_constexpr(na):
            _dma(r_x, slot + ins[j] * 256, src[j] + (s0 + st) * STEP)
        for j in range_constexpr(ncx):
            _dma(r_xs, slot + xso + 64 * xs_ins[j], xs_src[j] + 4 * (s0 + st), wide=False)

    return issue, na + ncx


@flyc.jit
def wreg_land(tid, tl, k, m, m0, split, x, xs, lds, n_epi):
    """``wreg``'s x waves: each step's x rows and codes into the LDS ring (the MFMA waves
    read them after the step's barrier), then the epilogue's ``n_epi`` barriers."""
    steps = k // STEP // tl.ks
    issue, per = _x_lander(
        tid, tl, k, m, m0, split * steps, x, xs, lds, tl.xw, tid // 64 - tl.waves
    )
    for st in range_constexpr(min(tl.depth - 1, steps)):
        issue(st)
    for st in range_constexpr(steps):
        _step_barrier(min(tl.depth - 2, steps - 1 - st) * per)
        if const_expr(st + tl.depth - 1 < steps):
            issue(st + tl.depth - 1)
    for _ in range_constexpr(n_epi):
        _step_barrier(0)


def _lds_barrier():
    """This wave's LDS reads retired, then ``s_barrier`` (its own loads stay in flight)."""
    llvm.inline_asm(
        res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)\n\ts_barrier",
        constraints="~{memory}", has_side_effects=True,
    )  # fmt: skip


@flyc.jit
def wreg_task(tid, tl, n, k, m, m0, n0, split, w, wsc, lds):
    """``wreg``'s MFMA waves: their W columns' fragments straight from the row-major weight
    into registers ``wreg`` steps ahead, x's from the LDS ring; the f32 accumulators."""
    lane, wave = tid % 64, tid // 64
    tn = tl.bn // tl.wn
    mt, nt = tl.bm // 16, tn // 16
    steps = k // STEP // tl.ks
    s0 = split * steps
    r_w, r_ws = rsrc(w, n * k), rsrc(wsc, n * (k // 32))
    sb = tl.stage_bytes // 4
    xso = tl.bm * ROWB // 4
    r, g = lane % 16, lane // 16
    c0 = n0 + wave * tn  # this wave's first column
    v4 = fx.Vector.make_type(4, fx.Int32)

    def wload(st):
        """Step st's W fragments (lane (r, g): row r, K 16 g .. and 64 + 16 g ..) and codes."""
        frags, codes = [], []
        for b in range_constexpr(nt):
            base = (c0 + 16 * b + r) * k + (s0 + st) * STEP + 16 * g
            lo = fx.Vector(bo.buffer_load(r_w, base // 4, vec_width=4, dtype=T.i32))
            hi = fx.Vector(bo.buffer_load(r_w, (base + 64) // 4, vec_width=4, dtype=T.i32))
            frags.append((lo, hi))
            ci = ((s0 + st) * (n // 16) + (c0 + 16 * b) // 16) * 64 + lane
            codes.append(fx.Int32(bo.buffer_load(r_ws, ci // 4, vec_width=1, dtype=T.i32)))
        return frags, codes

    def frag(base_words, row):
        p = base_words + row * (ROWB // 4)
        lo = fx.Vector(fx.ptr_load(p + 4 * (g ^ _swz(row)), result_type=v4))
        hi = fx.Vector(fx.ptr_load(p + 4 * ((4 + g) ^ _swz(row)), result_type=v4))
        return fx.Vector.from_elements(
            [lo[d] for d in range(4)] + [hi[d] for d in range(4)], fx.Int32
        )

    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nt)] for _ in range(mt)]
    pend = [wload(st) for st in range(min(tl.wreg, steps))]
    for st in range_constexpr(steps):
        frags, codes = pend.pop(0)
        if const_expr(st + tl.wreg < steps):
            pend.append(wload(st + tl.wreg))
        _lds_barrier()  # step st's x landed (its landing waves waited for it)
        slot = lds + (st % tl.depth) * sb
        xf = [frag(slot, 16 * a + r) for a in range(mt)]
        xsc = [
            (fx.Int32(fx.ptr_load(slot + xso + 16 * a + r)) >> (8 * g)) & 0xFF
            for a in range(mt)
        ]
        wf = [
            fx.Vector.from_elements(
                [lo[d] for d in range(4)] + [hi[d] for d in range(4)], fx.Int32
            )
            for lo, hi in frags
        ]
        wsc_ = [(c >> (8 * (lane % 4))) & 0xFF for c in codes]
        for a in range_constexpr(mt):
            for b in range_constexpr(nt):
                acc[a][b] = mfma_scaled(wf[b], xf[a], acc[a][b], wsc_[b], xsc[a])
    return acc


@flyc.jit
def store_out(tid, tl, acc, n, m, m0, n0, out, lds):
    """The tile to out (bf16 [m, n]; rows past m are dropped) through LDS: each wave's
    accumulators as bf16 into a [bm][bn] tile (16 B pieces XORed by the row), then whole
    16 B pieces of its rows, a row's bn * 2 B contiguous."""
    lane, wave = tid % 64, tid // 64
    tm, tn = tl.bm // tl.wm, tl.bn // tl.wn
    wy, wx = wave // tl.wn, wave % tl.wn
    pieces = tl.bn // 8  # 16 B pieces a row
    sw = min(pieces, 16)
    _step_barrier(0)  # every wave is past its last step's operand reads
    for a in range_constexpr(tm // 16):
        for b in range_constexpr(tn // 16):
            row = wy * tm + 16 * a + lane % 16
            col = wx * tn + 16 * b + 4 * (lane // 16)
            v = acc[a][b]
            pk = fx.Vector.from_elements([v[0], v[1], v[2], v[3]], fx.Float32).to(
                fx.BFloat16
            )
            piece = (col // 8) ^ (row % sw)
            fx.ptr_store(
                pk.bitcast(fx.Int32), lds + row * (pieces * 4) + piece * 4 + (col % 8) // 2
            )
    _step_barrier(0)
    r_o = rsrc(out, m * n * 2)
    v4 = fx.Vector.make_type(4, fx.Int32)
    tot = tl.bm * pieces
    for i in range_constexpr(_cdiv(tot, tl.cthreads)):
        idx = tid + tl.cthreads * i
        if const_expr(tot % tl.cthreads):
            idx = fx.min(idx, tot - 1)
        row, c = idx // pieces, idx % pieces
        v = fx.Vector(
            fx.ptr_load(lds + row * (pieces * 4) + (c ^ (row % sw)) * 4, result_type=v4)
        )
        bo.buffer_store(v, r_o, ((m0 + row) * n + n0 + 8 * c) // 2)


@flyc.jit
def split_reduce(tid, tl, acc, n, m, m0, n0, out, part, flags, tile, split):
    """K-split epilogue: this CTA's f32 partial to ``part`` (device scope), then its flag (a
    phase bit: the opposite of what it held, so no reset is needed); once every split's flag
    of the tile flipped, rows split * bm / ks .. of the tile, the partials summed in split
    order from zero, to bf16."""
    lane, wave = tid % 64, tid // 64
    tm, tn = tl.bm // tl.wm, tl.bn // tl.wn
    wy, wx = wave // tl.wn, wave % tl.wn
    size = tl.bm * tl.bn
    r_p, r_f = rsrc(part), rsrc(flags)
    mine = (tile * tl.ks + split) * size
    old = fx.Int32(
        bo.buffer_load(
            r_f, (tile * tl.ks + split) * FLAG, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
        )
    )
    for a in range_constexpr(tm // 16):
        for b in range_constexpr(tn // 16):
            row = wy * tm + 16 * a + lane % 16
            col = wx * tn + 16 * b + 4 * (lane // 16)
            bo.buffer_store(acc[a][b], r_p, mine + row * tl.bn + col, cache_modifier=CM_DEV)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    phase = 1 - old
    if tid == 0:
        bo.buffer_store(phase, r_f, (tile * tl.ks + split) * FLAG, cache_modifier=CM_DEV)
    if tid < tl.ks:

        def load_all():
            return fx.Int32(
                bo.buffer_load(
                    r_f, (tile * tl.ks + tid) * FLAG, vec_width=1, dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )  # fmt: skip

        _spin(load_all, lambda v: v != phase, guarded=False)
    gpu.barrier()
    rows = tl.bm // tl.ks
    per_row = tl.bn // 4
    tot = rows * per_row
    r_o = rsrc(out, m * n * 2)
    for i in range_constexpr(_cdiv(tot, tl.cthreads)):
        idx = tid + tl.cthreads * i
        if const_expr(tot % tl.cthreads):
            idx = fx.min(idx, tot - 1)
        row, c4 = split * rows + idx // per_row, idx % per_row
        off = row * tl.bn + 4 * c4
        v = fx.Vector.filled(4, 0.0, fx.Float32)
        for u in range_constexpr(tl.ks):
            v = v + fx.Vector(
                bo.buffer_load(
                    r_p, tile * tl.ks * size + u * size + off, vec_width=4,
                    dtype=T.f32, cache_modifier=CM_DEV,
                )
            )  # fmt: skip
        pk = fx.Vector.from_elements([v[0], v[1], v[2], v[3]], fx.Float32).to(fx.BFloat16)
        bo.buffer_store(pk.bitcast(fx.Int32), r_o, ((m0 + row) * n + n0 + 4 * c4) // 2)


def workspace(m, n, tl):
    """(f32 words of K-split partials, i32 flag words) a launch at up to m rows needs
    (flags zeroed once; unused at ks = 1)."""
    if tl.ks == 1:
        return 1, 1
    tiles = _cdiv(m, tl.bm) * (n // tl.bn)
    return tiles * tl.ks * tl.bm * tl.bn, tiles * tl.ks * FLAG


def grid(m, n, tl) -> int:
    return _cdiv(m, tl.bm) * (n // tl.bn) * tl.ks


_BUILDS: dict = {}


def standalone(n, k, tl):
    """``launch(x, xs, w, wsc, out, part, flags, m, stream=)`` on device addresses (the module
    docstring for the layouts; ``wsc`` from ``pack_scales``; ``part`` and ``flags`` per
    ``workspace``); m, the rows, is a launch argument, the grid ``grid(m, n, tl)``."""
    keyed = key_tuple({"n": n, "k": k, "tile": tl}, SOURCES)
    key = (n, k, tl)
    if key in _BUILDS:
        return _BUILDS[key]
    assert n % tl.bn == 0 and (k // STEP) % tl.ks == 0 and k % STEP == 0
    ntiles = n // tl.bn
    name = f"ds41_fp8_wide_n{n}_k{k}_t{tl.bm}x{tl.bn}_w{tl.wm}x{tl.wn}_k{tl.ks}_d{tl.depth}_x{tl.xw}_r{tl.wreg}"

    @fx.struct
    class Smem:
        x: fx.Array[fx.Int32, tl.lds_words, 16]

    @flyc.kernel(name=name, known_block_size=[tl.threads, 1, 1])
    def fp8_kernel(
        x: Int64, xs: Int64, w: Int64, wsc: Int64, out: Int64, part: Int64, flags: Int64,
        m: fx.Int32,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        mtiles = (m + tl.bm - 1) // tl.bm
        m0, n0, tile, split = task_origin(fx.block_idx.x, tl, mtiles, ntiles)
        lds = fx.SharedAllocator().allocate(Smem).peek().x.ptr
        if const_expr(tl.wreg):
            if tid >= tl.cthreads:
                wreg_land(tid, tl, k, m, m0, split, x, xs, lds, 2)
            else:
                acc = wreg_task(tid, tl, n, k, m, m0, n0, split, w, wsc, lds)
                if const_expr(tl.ks == 1):
                    store_out(tid, tl, acc, n, m, m0, n0, out, lds)
                else:
                    split_reduce(tid, tl, acc, n, m, m0, n0, out, part, flags, tile, split)
        else:
            acc = gemm_task(tid, tl, n, k, m, m0, n0, split, x, xs, w, wsc, lds)
            if const_expr(tl.ks == 1):
                store_out(tid, tl, acc, n, m, m0, n0, out, lds)
            else:
                split_reduce(tid, tl, acc, n, m, m0, n0, out, part, flags, tile, split)

    @flyc.jit
    def launch(
        x: Int64, xs: Int64, w: Int64, wsc: Int64, out: Int64, part: Int64, flags: Int64,
        m: fx.Int32, nblk: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):  # fmt: skip
        _ = keyed
        fp8_kernel(x, xs, w, wsc, out, part, flags, m).launch(
            grid=(nblk,), block=(tl.threads,), stream=stream
        )

    _BUILDS[key] = launch
    return launch


# per (N, K, width class): the tile of each served projection (wqkv_a 1792 x 5120 at every
# TP, wq_b 16384 / 8192 x 1280 at TP 2 / 4, wo_b 5120 x 4096 / 2048), from a sweep over
# tile shapes, K splits and depths at 96, 192 and 384 rows
TILES: dict = {
    (1792, 5120, 96): Tile(96, 64, 2, 2, 8, 4),
    (1792, 5120, 192): Tile(96, 128, 2, 2, 8, 4),
    (1792, 5120, 384): Tile(96, 128, 2, 2, 4, 4),
    (8192, 1280, 96): Tile(64, 64, 2, 2, 1, 4),
    (8192, 1280, 192): Tile(96, 64, 2, 2, 1, 4),
    (8192, 1280, 384): Tile(96, 128, 2, 2, 1, 4),
    (16384, 1280, 96): Tile(96, 64, 2, 2, 1, 4),
    (16384, 1280, 192): Tile(96, 128, 2, 2, 1, 4),
    (16384, 1280, 384): Tile(192, 128, 2, 2, 1, 3, 3),
    (5120, 2048, 96): Tile(64, 64, 2, 2, 1, 4),
    (5120, 2048, 192): Tile(64, 64, 2, 2, 1, 4),
    (5120, 2048, 384): Tile(64, 128, 2, 2, 1, 4),
    (5120, 4096, 96): Tile(32, 128, 1, 4, 2, 6),
    (5120, 4096, 192): Tile(64, 128, 2, 2, 2, 4),
    (5120, 4096, 384): Tile(64, 128, 2, 2, 1, 4),
}


def width_class(m: int) -> int:
    for c in (96, 192, 384):
        if m <= c:
            return c
    raise ValueError(f"fp8 wide: {m} rows past 384")


def served(n: int, k: int, m: int) -> bool:
    """Whether this kernel serves an n x k projection at m rows (a tile for m's class)."""
    return (n, k, width_class(m)) in TILES


class Projection:
    """One dense projection on this kernel: its codes in ``pack_scales``' layout, a launch
    and a K-split workspace for each width class it serves (``TILES``; each class's launches
    run in stream order, so the layers share them)."""

    _ws: dict = {}  # (n, k, tile, class) -> (part, flags), shared by every layer

    def __init__(self, weight: torch.Tensor, codes: torch.Tensor):
        n, k = weight.shape
        assert weight.dtype == torch.float8_e4m3fn and weight.is_contiguous()
        self.weight, self.n, self.k = weight, n, k
        self.scales = pack_scales(codes, n, k)
        self.tiles = {c: TILES[(n, k, c)] for c in (96, 192, 384) if (n, k, c) in TILES}
        for c, tl in self.tiles.items():
            key = (n, k, tl, c)
            if key not in Projection._ws:
                pw, fw = workspace(c, n, tl)
                Projection._ws[key] = (
                    torch.empty(pw, dtype=torch.float32, device=weight.device),
                    torch.zeros(fw, dtype=torch.int32, device=weight.device),
                )

    def __call__(self, xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """bf16 [m, n] = MXFP8 x [m, k] (e4m3, ue8m0 [m, k / 32]) . W^T."""
        m = xq.shape[0]
        c = width_class(m)
        tl = self.tiles[c]
        part, flags = Projection._ws[(self.n, self.k, tl, c)]
        out = torch.empty(m, self.n, dtype=torch.bfloat16, device=xq.device)
        standalone(self.n, self.k, tl)(
            xq.data_ptr(), xs.data_ptr(), self.weight.data_ptr(), self.scales.data_ptr(),
            out.data_ptr(), part.data_ptr(), flags.data_ptr(), m, grid(m, self.n, tl),
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        return out
