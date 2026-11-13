# SPDX-License-Identifier: Apache-2.0
"""The attention's front half at wide row counts (60 .. 384 rows) as one launch a layer: the
previous MoE's sum over the TP ranks, the attention-side mHC seam and its input norm
(``seam_wide``), and optionally the attention front up to the attention core: wqkv_a, the q
norm and its MXFP8 quant, the KV row's norm, RoPE and store into the V4-layout SWA cache, and
wq_b with the query RoPE in its epilogue. Each stage's tasks are a range of the launch's
CTAs, and each stage gives SGLang's bits: the sum is its custom all-reduce's, the seam its
seam pair's, the GEMMs its aiter MXFP8 GEMM's (``fp8_wide``), the q norm its
``rmsnorm_fake_quant_fp8``'s and the KV row and the query RoPE its K store launch's
(``fused_k_norm_rope_flashmla``, V4 layout, the query roped).

Inputs: the previous layer's MoE output on this rank, bf16 [m, 5120] (this rank's partial of
the sum), the seam's inputs (the residual streams, the owed post's post and comb, the previous
boundary's pre, the layer's mix weights and norm weight) and the front's (``build``).

Outputs: the seam's (the new streams, the normed rows, their MXFP8 operand, the mixes) and
the front's (the roped query, the cache rows, the q norm's operand the indexer reads), for
every row or, on a token-parallel step (``tpar``), for the rows this rank owns (rank r owns
rows r m / tp ..). With ``gather`` a token-parallel step's owners run the front on their rows
and send every rank the compact rows wq_b and the cache need, the q norm's operand (1,320 B a
row) and the KV record (584 B), instead of the input norm's bf16 rows (10,240 B); every rank
then stores every row's record and runs wq_b on every row.

The sum's tasks (16 rows by ``ac`` columns, rows in order) come first:
- ``oneshot`` (TP 2): every row to the other rank, then every row summed here;
- ``rsag`` (TP 4): each row to its owner, summed there, the sums to every other rank;
- ``rs``: each row to its owner, summed there (the token-parallel steps);
- None: no sum (the MoE output is summed already, or the layer has no owed post).
An owner sums in fp32 from its own partial, then ranks r + 1, r + 2, ... modulo tp, and
rounds once to bf16: the order of the custom all-reduce SGLang runs (one-shot at TP 2, its
two-stage form at TP 4), so the sums are its bits. They land in the TP group's peer buffer.

A hand-off is a flag a CTA flips once a launch (a phase bit), after its outputs are stored;
a CTA reads the launch's phase from its own flag first. Waits are only on CTAs earlier in
the grid or on peers' CTAs of the same or an earlier stage, so they always finish: the GPU
dispatches a launch's CTAs in order. The scratch and the peer buffer are uncached, so a
hand-off between CTAs on different XCDs or GPUs misses no cache.
"""

import json
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.math import fma
from flydsl.expr.typing import Int64, T

from sgl_amd_mono.common.mx import clamp_fp8, code_ceil, rcp_pow2
from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_SYS,
    bf16_pair,
    bf2_f32,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    hw_rsq,
    peer_bases,
    rsrc,
    xshfl,
)
from sgl_amd_mono.common.plan import kernel_sources, key_tuple
from sgl_amd_mono.common.stamps import TL_POINTS, timeline_row
from sgl_amd_mono.common.sync import _spin

from . import fp8_wide as f8
from . import seam_wide as sw

SOURCES = kernel_sources(__name__)

H = 5120
THREADS = 256
MAXROWS = 384  # the widest step: the peer buffer's row slots
MODES = (None, "oneshot", "rs", "rsag")
# stage marks of a CTA's row (``stamp``), by the stage's tag: base + 0 its start, + 1 its
# inputs ready, + 2 its work done, + 3 its outputs published
MARKS = {"ar": 0, "seama": 4, "seama_row": 8, "qkv": 12, "row": 16, "wqb": 20, "rec": 24}
LINE = 256  # bytes: scratch regions start on their own lines


def _cdiv(a, b):
    return (a + b - 1) // b


@flyc.jit
def stamp(c, point: int):
    """Mark ``point`` on this CTA's timeline row (``c["tl"]``), if the build has one: the
    realtime clock, read by an opaque ``s_memrealtime`` so no two marks merge."""
    if const_expr(c.get("tl") is not None):  # noqa: SIM102
        if c["tid"] == 0:
            t = fx.Int64(
                llvm.inline_asm(
                    T.i64, [], "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s",
                    has_side_effects=True,
                )
            )  # fmt: skip
            ptr = fx.inttoptr(
                fx.Pointer[fx.Int64, fx.AddressSpace.Global, 8].ir_type,
                c["tl"] + fx.Int64(point * 8),
            )
            fx.generic_store(ptr, t)


@flyc.jit
def _wait(c, load_all, pending, site: int):
    """Re-issue ``load_all`` until ``pending`` clears. A debug build (``Plan.bound``) gives up
    after that many polls and leaves ``site`` in its CTA's error word."""
    if const_expr(c.get("bound", 0) == 0):
        _spin(load_all, pending, guarded=False)
    else:
        v = load_all()
        n = fx.Int32(0)
        while pending(v) & (n < c["bound"]):
            rocdl.s_sleep(1)
            v = load_all()
            n = n + 1
        if pending(v):
            bo.buffer_store(fx.Int32(site), rsrc(c["err"]), 0, cache_modifier=CM_DEV)


class PeerAt(msgspec.Struct, frozen=True):
    """A width's regions of the TP group's peer buffer (``peer_layout``), byte offsets: the
    ranks' partials (slot [src][MAXROWS][H] bf16), the sums (x [MAXROWS][H]) and this
    width's flags, rsf and agf [src][sum task] (i32)."""

    slot: int
    x: int
    rsf: int
    agf: int
    size: int  # the buffer's bytes: a masked store lands past them and is dropped
    # the compact gather's rows (q norm operand codes and scales, KV records) and this
    # width's flags [src][row-stage CTA]
    gq: int = 0
    gqs: int = 0
    gkv: int = 0
    gflag: int = 0


QR = 1280  # q_lora_rank: wq_b's K
KVD = 512  # the KV row (head_dim)
QKV = QR + KVD  # wqkv_a's rows
HD = 512  # a query head
ROPE = 64  # the rotated dims at the end of a head and of the KV row
NOPE = HD - ROPE
PAGE_BITS = 8  # the SWA cache's 256-token pages (V4 layout)
KV_DATA = NOPE + 2 * ROPE  # bytes a token's data row: e4m3 nope, bf16 rope
KV_SCALE = 8  # bytes a token's scale row: 7 ue8m0, one pad
KV_PAGE = _cdiv((KV_DATA + KV_SCALE) << PAGE_BITS, KV_DATA) * KV_DATA
ROWS_A_CTA = 4  # the row stage's rows: a wave a row
REC = 640  # bytes a gathered KV record: its data row, then its 7 scale bytes
GROW_CTAS = 64  # row-stage flags a rank in the peer buffer (gather)


class Front(msgspec.Struct, frozen=True):
    """The attention front after the seam: wqkv_a (``qkv``, on the norm's MXFP8 operand), the
    q norm and its MXFP8 quant, the KV row's norm, RoPE and store into the V4-layout SWA cache
    (the row stage), wq_b (``wqb``, ``heads`` heads of 512, the query RoPE in its epilogue)."""

    qkv: f8.Tile
    wqb: f8.Tile
    heads: int
    q_eps: float
    kv_eps: float
    variant: int = 0  # rounding variants of the KV row's arithmetic (``krope_row``)

    @property
    def n_wqb(self) -> int:
        return self.heads * HD


class Plan(msgspec.Struct, frozen=True):
    """A launch's static shape: m rows over tp ranks, the sum's ``mode`` (``MODES``; ``peer``
    its regions) at ``ac`` columns a task, the seam's form and the front (or None: the launch
    ends at the seam)."""

    m: int
    seam: sw.Form
    tp: int = 1
    mode: str | None = None
    peer: PeerAt | None = None
    ac: int = 512
    # the sum's pacing: a task pushes once the task this many before it has (0: no pacing)
    window: int = 0
    front: Front | None = None
    # a token-parallel step: the seam (and the front) on this rank's rows only; its sum, if
    # any, the reduce-scatter
    tpar: bool = False
    # a token-parallel step's compact gather: the owners' q norm operands and KV records to
    # every rank, wq_b and the cache store on every row (needs the front)
    gather: bool = False
    # debug builds: a wait gives up after this many polls (0: never)
    bound: int = 0

    def __post_init__(self):
        assert self.seam.stage == "seama"
        assert self.mode in MODES and (self.peer is None or self.mode is not None or self.gather)
        assert (self.mode == "rs") <= self.tpar and self.mode in ((None, "rs") if self.tpar else MODES)
        assert self.gather <= (self.tpar and self.front is not None)
        assert not self.gather or _cdiv(self.own, ROWS_A_CTA) <= GROW_CTAS
        assert self.mode != "oneshot" or self.tp == 2
        assert self.m <= MAXROWS and self.m % self.tp == 0
        assert H % self.ac == 0 and self.ac % 64 == 0 and (16 * self.ac // 8) % THREADS == 0

    @property
    def own(self) -> int:
        """Rows a rank owns."""
        return self.m // self.tp

    @property
    def seam_rows(self) -> int:
        """The seam's rows: every row, or a token-parallel width's own."""
        return self.own if self.tpar else self.m

    @property
    def sum_tasks(self) -> int:
        """Sum tasks: (unit of 16 rows of an owner, owner, column chunk)."""
        if self.mode is None:
            return 0
        owners, rows = (1, self.m) if self.mode == "oneshot" else (self.tp, self.own)
        return owners * _cdiv(rows, 16) * (H // self.ac)

    @property
    def sum_flags(self) -> int:
        """Sum flags a rank: one a task."""
        return self.sum_tasks

    @property
    def seam_tasks(self) -> tuple[int, int]:
        """(slice CTAs, row CTAs) of the seam (``seam_wide.grid``)."""
        return sw.grid(self.seam_rows)

    @property
    def bases(self) -> dict[str, int]:
        """Each stage's first CTA, and ``end``: the grid. A GEMM stage starts on a multiple of
        8 CTAs, so its tasks land on the XCDs ``fp8_wide.task_origin`` deals them to."""
        out = {"ar": 0, "seama": self.sum_tasks}
        end = self.sum_tasks + sum(self.seam_tasks)
        f = self.front
        if f is not None:
            ms = self.seam_rows
            out["qkv"] = _cdiv(end, 8) * 8
            out["row"] = out["qkv"] + f8.grid(ms, QKV, f.qkv)
            nxt = out["row"] + _cdiv(ms, ROWS_A_CTA)
            if self.gather:
                out["rec"] = nxt
                nxt = out["rec"] + _cdiv(self.m, ROWS_A_CTA)
            out["wqb"] = _cdiv(nxt, 8) * 8
            end = out["wqb"] + f8.grid(self.q_rows, f.n_wqb, f.wqb)
        out["end"] = end
        return out

    @property
    def q_rows(self) -> int:
        """wq_b's rows: the seam's, or with ``gather`` every row of the step."""
        return self.m if self.gather else self.seam_rows

    @property
    def grid(self) -> int:
        return self.bases["end"]

    @property
    def lds_words(self) -> int:
        seam = sw.FN_WORDS + sw.WAVES * (sw.HC + 1) * sw.TILE
        if self.front is None:
            return seam
        return max(seam, self.front.qkv.lds_words, self.front.wqb.lds_words)


def scratch_layout(p: Plan) -> dict[str, tuple[int, int]]:
    """Scratch regions -> (byte offset, bytes): the flags (every CTA's own, then the seam's,
    i32 each), the seam's scratch and the stage marks."""
    ms = p.seam_rows
    items = [
        ("own", 4 * p.grid),
        ("sflags", 4 * sw.flag_words(ms)),
        ("pushed", 4 * max(p.sum_tasks, 1)),
    ]
    if p.front is not None:
        pw, fw = f8.workspace(ms, QKV, p.front.qkv)
        items += [("pflags", 4 * fw), ("part", 4 * pw), ("xq", ms * H), ("xs", ms * H // 32),
                  ("qkv", ms * QKV * 2), ("qq", ms * QR), ("qs", ms * QR // 32)]
    items += [
        ("err", 4 * p.grid),
        ("seam", sw.scratch_bytes(ms)),
        ("tl", 8 * TL_POINTS * p.grid),
    ]
    out, off = {}, 0
    for name, n in items:
        out[name] = (off, n)
        off = _cdiv(off + n, LINE) * LINE
    return out


def scratch_bytes(p: Plan) -> int:
    return max(o + n for o, n in scratch_layout(p).values())


def flag_bytes(p: Plan) -> tuple[int, int]:
    """(byte offset, bytes) of the scratch words that start as zeros: the flags."""
    lay = scratch_layout(p)
    last = "pflags" if p.front is not None else "pushed"
    return 0, lay[last][0] + lay[last][1]


# ---------------------------------------------------------------- the MoE's sum over the ranks
# A sum task is (unit u, owner d, column chunk): rows d own + 16 u .. of owner d (with
# ``oneshot`` one owner, every rank, of all the rows), ``ac`` columns. Tasks go unit-major, so
# every owner's link carries data from the start; with ``window`` w a task pushes only once
# the task w places before it has pushed, so the early rows' sums land first and the seam
# starts on them while the link still carries the later rows.

SU = 16  # rows a sum task


def _owners(p: Plan) -> tuple[int, int]:
    """(owners, rows an owner) of the sum."""
    return (1, p.m) if p.mode == "oneshot" else (p.tp, p.own)


def _units(p: Plan) -> int:
    return _cdiv(_owners(p)[1], SU)


def _task_of(p: Plan, row, col):
    """The sum task holding row ``row`` of the step at column ``col``."""
    D, rows = _owners(p)
    nc = H // p.ac
    if const_expr(p.mode == "oneshot"):
        d, r = 0, row
    else:
        d, r = row // rows, row % rows
    return ((r // SU) * D + d) * nc + col // p.ac


def _await_ranks(c, lane, p, r_flags, fidx, srcs, phase, site):
    """Wait (one wave) until flag ``fidx`` of ``r_flags`` ([src][task] words) of each rank for
    which ``srcs(src)`` holds reads ``phase``: lane l polls rank l."""
    src = fx.min(lane, p.tp - 1)

    def load_all():
        return fx.Int32(
            bo.buffer_load(
                r_flags, src * p.sum_flags + fidx, vec_width=1, dtype=T.i32,
                cache_modifier=CM_SYS,
            )
        )  # fmt: skip

    _wait(c, load_all, lambda v: (v != phase) & srcs(src), site)


@flyc.jit
def sum_task(tid, p, j, x_in, bases, local, pushed, rank, phase, c):
    """Sum task j of the MoE output over the ranks as ``Plan.mode`` says (the module
    docstring): its owner's rows of it land in the peer buffer's x of every rank that reads
    them (every rank, or with ``rs`` the owner). ``pushed``: the tasks' push flags here."""
    lane = tid % 64
    lay = p.peer
    nc = H // p.ac
    D, orows = _owners(p)
    u = j // (D * nc)
    d = (j // nc) % D
    c0 = (j % nc) * p.ac
    pieces = p.ac // 8  # 16-byte pieces a row
    per = SU * pieces // THREADS
    row_bytes = H * 2
    oneshot = p.mode == "oneshot"

    def word(region, src, grow, pc):
        base = getattr(lay, region) // 4
        if const_expr(src is not None):
            base = base + src * (MAXROWS * row_bytes // 4)
        return base + (grow * row_bytes + (c0 + 8 * pc) * 2) // 4

    pcs = []
    for i in range_constexpr(per):
        idx = tid + THREADS * i
        r = u * SU + idx // pieces
        pcs.append((d * orows + fx.min(r, orows - 1), idx % pieces, r < orows))
    r_in = rsrc(x_in)
    vals = [
        fx.Vector(bo.buffer_load(r_in, (grow * row_bytes + (c0 + 8 * pc) * 2) // 4, vec_width=4, dtype=T.i32))
        for grow, pc, _ in pcs
    ]  # fmt: skip
    r_l = rsrc(local, lay.size)
    r_p = rsrc(pushed)
    if const_expr(p.window):
        if j >= p.window:
            if tid == 0:

                def load_prev():
                    return fx.Int32(bo.buffer_load(r_p, j - p.window, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))

                _wait(c, load_prev, lambda v: v != phase, 9)
            gpu.barrier()
    # 1. this rank's rows of the task to the ranks that sum them
    for i in range_constexpr(per):
        grow, pc, ok = pcs[i]
        for dd in range_constexpr(p.tp):
            if const_expr(oneshot):
                okd = ok & (rank != dd)
            else:
                okd = ok & (d == dd) & (rank != dd)
            bo.buffer_store(vals[i], rsrc(bases[dd], lay.size), word("slot", rank, grow, pc), mask=okd, cache_modifier=CM_SYS)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        if const_expr(p.window):
            bo.buffer_store(phase, r_p, j, cache_modifier=CM_DEV)
        for dd in range_constexpr(p.tp):
            if const_expr(oneshot):
                if rank != dd:
                    bo.buffer_store(phase, rsrc(bases[dd]), lay.rsf // 4 + rank * p.sum_flags + j, cache_modifier=CM_SYS)
            else:
                if (rank != dd) & (d == dd):
                    bo.buffer_store(phase, rsrc(bases[dd]), lay.rsf // 4 + rank * p.sum_flags + j, cache_modifier=CM_SYS)
    mine = True if oneshot else d == rank
    if mine:
        # every other rank's rows of this task here, then the sums
        if tid < 64:
            _await_ranks(c, lane, p, rsrc(local + fx.Int64(lay.rsf)), j, lambda s: s != rank, phase, 1)
        gpu.barrier()
        for i in range_constexpr(per):
            grow, pc, ok = pcs[i]
            acc = []
            for w in range_constexpr(4):
                lo, hi = bf2_f32(fx.Int32(vals[i][w]))
                acc += [lo, hi]
            for k in range_constexpr(1, p.tp):
                src = (rank + k) % p.tp
                pv = fx.Vector(bo.buffer_load(r_l, word("slot", src, grow, pc), vec_width=4, dtype=T.i32, cache_modifier=CM_SYS))
                for w in range_constexpr(4):
                    lo, hi = bf2_f32(fx.Int32(pv[w]))
                    acc[2 * w] = acc[2 * w] + lo
                    acc[2 * w + 1] = acc[2 * w + 1] + hi
            sv = fx.Vector.from_elements(acc, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
            bo.buffer_store(sv, r_l, word("x", None, grow, pc), mask=ok, cache_modifier=CM_DEV)
            if const_expr(p.mode == "rsag"):
                for dd in range_constexpr(p.tp):
                    bo.buffer_store(sv, rsrc(bases[dd], lay.size), word("x", None, grow, pc), mask=ok & (rank != dd), cache_modifier=CM_SYS)
        if const_expr(p.mode == "rsag"):
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if tid == 0:
                for dd in range_constexpr(p.tp):
                    if rank != dd:
                        bo.buffer_store(phase, rsrc(bases[dd]), lay.agf // 4 + rank * p.sum_flags + j, cache_modifier=CM_SYS)
    else:
        if const_expr(p.mode == "rsag"):
            # the owner's sums of these rows here
            if tid < 64:
                _await_ranks(c, lane, p, rsrc(local + fx.Int64(lay.agf)), j, lambda s: s == d, phase, 2)
            gpu.barrier()


def _await_sums(c, lane, p, own, row0, rows, col, phase, last):
    """Wait (one wave) until the sum tasks of rows row0 .. row0 + rows (of the step, at most
    row ``last``) at column ``col`` have flipped their own flags to ``phase``: lane l polls
    row row0 + l's."""
    assert rows <= 64
    row = fx.min(row0 + fx.min(lane, rows - 1), last)
    cta = _task_of(p, row, col)
    r_o = rsrc(own)

    def load_all():
        return fx.Int32(bo.buffer_load(r_o, cta, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))

    _wait(c, load_all, lambda v: v != phase, 3)


# ---------------------------------------------------------------- the seam
def _await_slices(c, a, lane, grp, phase):
    """``seam_wide.await_slices`` through ``_wait``."""
    r_f = rsrc(a["flags"])
    s2 = fx.min(64 + lane, sw.SLICES - 1)

    def load_all():
        return fx.Vector.from_elements(
            [
                fx.Int32(bo.buffer_load(r_f, (grp * sw.SLICES + sl) * sw.FLAG, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
                for sl in (lane, s2)
            ],
            fx.Int32,
        )

    _wait(c, load_all, lambda v: (v[0] != phase) | (v[1] != phase), 4)


@flyc.jit
def seam_cta(tid, k, p, wl, own, phase, rank, c, a):
    """Seam CTA k (``seam_wide``'s launch body, its grid after the sum's): a slice CTA
    prefetches its mix weights, waits for the sums its rows and columns read, then runs the
    slice; a row CTA waits for its row group's slices and runs the reduce + sinkhorn and the
    norm. ``a``: ``seam_wide``'s arguments, x at the seam's first row."""
    lane, wave = tid % 64, tid // 64
    form, m = p.seam, p.seam_rows
    a = {**a, "y_off": a.pop("y"), "mix_off": a.pop("mix"), "sq_off": a.pop("sq")}
    r_f = rsrc(a["flags"])
    p1, _ = p.seam_tasks
    split = sw.SPLIT[sw.width_class(m)]
    gr = sw.group_rows(m)
    row0 = rank * m if p.tpar else 0  # the seam's first row among the step's
    if k < p1:
        stamp(c, MARKS["seama"])
        s, grp = k % sw.SLICES, k // sw.SLICES
        old = fx.Int32(
            bo.buffer_load(r_f, k * sw.FLAG, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)
        )
        if const_expr(split == 1):
            blk, nts, main = grp * sw.WAVES + wave, (0, 1), True
        else:
            blk, nts = grp * (sw.WAVES // 2) + wave // 2, (wave % 2,)
            main = wave % 2 == 0
        fnl = sw._fn_loads(tid, a, s)
        if const_expr(p.mode is not None):
            if wave == 0:
                _await_sums(c, lane, p, own, row0 + grp * gr, gr, s * sw.BK, phase, row0 + m - 1)
            gpu.barrier()
        stamp(c, MARKS["seama"] + 1)
        ops = sw._block_loads(form, lane, a, s, blk, m)
        sw._fn_table(fnl, wl)
        gpu.barrier()
        if blk * sw.BM < m:
            sw.slice_block(
                form, lane, a, s, blk, m, wl, wl + sw.FN_WORDS + wave * (sw.HC + 1) * sw.TILE,
                ops, nts, main,
            )  # fmt: skip
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        stamp(c, MARKS["seama"] + 2)
        if tid == 0:
            bo.buffer_store(1 - old, r_f, k * sw.FLAG, cache_modifier=CM_DEV)
        stamp(c, MARKS["seama"] + 3)
    else:
        stamp(c, MARKS["seama_row"])
        row = k - p1
        slot = p1 + row * sw.WAVES + wave
        old = fx.Int32(
            bo.buffer_load(r_f, slot * sw.FLAG, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)
        )
        ph = 1 - old
        if tid < 8:
            fx.ptr_store(fx.Float32(0.0), wl + tid)
        gpu.barrier()
        _await_slices(c, a, lane, row // gr, ph)
        stamp(c, MARKS["seama_row"] + 1)
        if wave == 0:
            gates, combs, sq = sw.row_mixes(lane, a, row, m)
            sw.row_gates(form, lane, a, row, gates, combs, sq, row < m)
        elif wave == 1:
            sw.row_norm(form, lane, 1, a, row, row < m, wl)
        elif wave == 2:
            sw.row_norm(form, lane, 2, a, row, row < m, wl)
        else:
            sw.row_norm(form, lane, 3, a, row, row < m, wl)
        rocdl.s_waitcnt(vmcnt=0)
        stamp(c, MARKS["seama_row"] + 2)
        if lane == 0:
            bo.buffer_store(ph, r_f, slot * sw.FLAG, cache_modifier=CM_DEV)
        stamp(c, MARKS["seama_row"] + 3)


# ---------------------------------------------------------------- the front
def _await_ctas(c, lane, own, ctas, phase, site=5):
    """Wait (one wave) until every CTA of ``ctas`` (lane l's list: the CTAs it polls) has
    flipped its own flag to ``phase``."""
    r_o = rsrc(own)

    def load_all():
        return fx.Vector.from_elements(
            [fx.Int32(bo.buffer_load(r_o, c, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)) for c in ctas],
            fx.Int32,
        )

    def pending(v):
        out = v[0] != phase
        for i in range(1, len(ctas)):
            out = out | (v[i] != phase)
        return out

    _wait(c, load_all, pending, site)


def _task_origin(j, tl, mtiles, ntiles):
    """``fp8_wide.task_origin`` at a static row count: GEMM task j's (m0, n0, tile, split)."""
    nblk = mtiles * ntiles * tl.ks
    task = (j % 8) * (nblk // 8) + j // 8 if nblk % 8 == 0 else j
    tile, split = task // tl.ks, task % tl.ks
    return (tile % mtiles) * tl.bm, (tile // mtiles) * tl.bn, tile, split


def _gemm_cta(tl, m, n, tile, split):
    """The stage-relative CTA of a GEMM task (tile, split): ``fp8_wide.task_origin`` inverted."""
    nblk = f8.grid(m, n, tl)
    task = tile * tl.ks + split
    if nblk % 8 == 0:
        return (task % (nblk // 8)) * 8 + task // (nblk // 8)
    return task


def _lanes(lane, lo, count, most):
    """Lane l's share of the indices lo .. lo + count (at most ``most``): ceil(most / 64)
    of them, clamped to the last."""
    per = _cdiv(most, 64)
    return [lo + fx.min(lane * per + i, count - 1) for i in range(per)]


@flyc.jit
def qkv_cta(tid, j, p, a, lds, own, phase, c):
    """wqkv_a task j: waits for the seam's row CTAs of its rows, then the tile (its K split
    reduced in the launch) into the scratch's qkv rows."""
    f, ms = p.front, p.seam_rows
    tl = f.qkv
    b = p.bases
    lane, wave = tid % 64, tid // 64
    mtiles = _cdiv(ms, tl.bm)
    m0, n0, tile, split = _task_origin(j, tl, mtiles, QKV // tl.bn)
    stamp(c, MARKS["qkv"])
    if wave == 0:
        first = b["seama"] + p.seam_tasks[0] + m0
        _await_ctas(c, lane, own, _lanes(lane, first, fx.min(ms - m0, tl.bm), tl.bm), phase, 5)
    gpu.barrier()
    stamp(c, MARKS["qkv"] + 1)
    acc = f8.gemm_task(tid, tl, QKV, H, ms, m0, n0, split, a["xq"], a["xs"], a["wqkv"], a["wqkv_s"], lds)
    stamp(c, MARKS["qkv"] + 2)
    if const_expr(tl.ks == 1):
        f8.store_out(tid, tl, acc, QKV, ms, m0, n0, a["qkv"], lds)
    else:
        f8.split_reduce(tid, tl, acc, QKV, ms, m0, n0, a["qkv"], a["part"], a["pflags"], tile, split)


def _f32x8(w4):
    """Four bf16-pair words as eight f32 (low first)."""
    out = []
    for q in range_constexpr(4):
        out += [bf_lo(w4[q]), bf_hi(w4[q])]
    return out


@flyc.jit
def qnorm_row(f, lane, a, row, valid, dests):
    """``rmsnorm_fake_quant_row`` of qkv_a's first 1280 columns of ``row`` (K = 1280: one
    2048-wide chunk, 4 warps of 64 threads holding 8 columns each, thread t at columns
    8 t), on one wave: lane l plays threads l, l + 64 and (below 32) l + 128; the warps' sums
    meet as (W0 + W2) + (W1 + W3) with W3 = 0. The normed row's MXFP8 quant (e4m3 + ue8m0)
    to each of ``dests``: (codes, scales, the row there, cache policy)."""
    r_x = rsrc(a["qkv"])
    r_w = rsrc(a["qn_w"])
    parts = []
    for w in range_constexpr(3):
        col = 8 * (lane + 64 * w)
        live = col < QR
        cc = fx.min(col, QR - 8)
        xv = fx.Vector(bo.buffer_load(r_x, (row * QKV + cc) // 2, vec_width=4, dtype=T.i32))
        wv = fx.Vector(bo.buffer_load(r_w, cc // 2, vec_width=4, dtype=T.i32))
        xs = [live.select(v, fx.Float32(0.0)) for v in _f32x8([xv[q] for q in range(4)])]
        parts.append((col, live, xs, _f32x8([wv[q] for q in range(4)])))
    sums = []
    for col, live, xs, _ in parts:
        e = [v * v for v in xs]
        acc = ((e[0] + e[2]) + (e[4] + e[6])) + ((e[1] + e[3]) + (e[5] + e[7]))
        sums.append(butterfly(acc, (8, 4, 2, 1, 16, 32)))
    total = (sums[0] + sums[2]) + sums[1]
    var = total / fx.Float32(float(QR)) + fx.Float32(f.q_eps)
    rstd = fx.Float32(1.0) / fmath.sqrt(var)
    for col, live, xs, ws in parts:
        ys = [(xs[i] * rstd) * ws[i] for i in range(8)]
        words = fx.Vector.from_elements([bf16_pair(ys[2 * q], ys[2 * q + 1]) for q in range(4)], fx.Float32).bitcast(fx.Int32)
        yb = [bf_lo(words[q]) if h == 0 else bf_hi(words[q]) for q in range(4) for h in range(2)]
        amax = fx.Float32(0.0)
        for v in yb:
            amax = fx.max(amax, abs(v))
        amax = butterfly(amax, (1, 2), lambda u, v: fx.max(u, v))
        code = code_ceil(fx.max(amax, fx.Float32(1e-10)) * fx.Float32(1.0 / 448.0))
        mul = rcp_pow2(code)
        q8 = [fp8_pack4(*[clamp_fp8(v * mul) for v in yb[4 * h : 4 * h + 4]]) for h in range(2)]
        ok = valid & live
        for r_q, r_s, drow, cm in dests:
            if ok:
                bo.buffer_store(fx.Vector.from_elements(q8, fx.Int32), r_q, (drow * QR + col) // 4, cache_modifier=cm)
            if ok & (lane % 4 == 0):
                bo.buffer_store(fx.Int8(code), r_s, drow * (QR // 32) + col // 32, cache_modifier=cm)


def _rsqrt(f, v):
    """``rsqrtf`` as HIP compiles it (variant bit 0: the hardware v_rsq_f32, else 1 / sqrt)."""
    if f.variant & 1:
        return fx.Float32(1.0) / fmath.sqrt(v)
    return hw_rsq(v)


def _cache_row(a, grow):
    """(data row, scale row, has a slot) of the step's row ``grow`` in the V4 SWA cache."""
    loc = fx.Int32(bo.buffer_load(rsrc(a["out_loc"]), grow, vec_width=1, dtype=T.i32))
    page = fx.Int64(fx.max(loc, fx.Int32(0)) >> PAGE_BITS)
    slot = fx.Int64(fx.max(loc, fx.Int32(0)) & ((1 << PAGE_BITS) - 1))
    base = a["kvcache"] + page * fx.Int64(KV_PAGE)
    r_d = rsrc(base + slot * fx.Int64(KV_DATA))
    r_sc = rsrc(base + fx.Int64(KV_DATA << PAGE_BITS) + slot * fx.Int64(KV_SCALE))
    return r_d, r_sc, loc >= 0


@flyc.jit
def krope_row(f, lane, a, row, grow, valid, dests):
    """``fused_k_norm_rope_flashmla`` (V4 layout) of qkv_a's KV columns of ``row`` on one
    wave: its 256 threads hold a pair each (thread t, columns 2 t, 2 t + 1), in 8 warps of
    32; lane l plays threads l + 64 k (warp 2 k + l / 32). The sum of squares: each pair's
    fma pair, the warps' xor butterfly (16 .. 1), then the 8 warp sums as lanes j % 8 reduce
    them (xor 4, 2, 1). Warp 7 (the last 64 dims) ropes and stores bf16; warps 0 .. 6 each
    quantize their 64 values to e4m3 with one ue8m0 scale. ``row``: the row of qkv_a;
    ``grow``: the step's row (its position and cache slot). The row lands at its cache slot
    (none for a negative slot), or with ``dests`` [(data, scales, cache policy)] there."""
    r_x = rsrc(a["qkv"])
    r_w = rsrc(a["kvn_w"])
    xs, ws = [], []
    for k in range_constexpr(4):
        t = lane + 64 * k
        xv = fx.Int32(bo.buffer_load(r_x, (row * QKV + QR) // 2 + t, vec_width=1, dtype=T.i32))
        wv = fx.Int32(bo.buffer_load(r_w, t, vec_width=1, dtype=T.i32))
        xs.append((bf_lo(xv), bf_hi(xv)))
        ws.append((bf_lo(wv), bf_hi(wv)))
    add = lambda u, v: u + v  # noqa: E731
    mx = lambda u, v: fx.max(u, v)  # noqa: E731
    warp = []
    for k in range_constexpr(4):
        x0, x1 = xs[k]
        sq = fma(x1, x1, x0 * x0)
        warp.append(butterfly(sq, (16, 8, 4, 2, 1), add))  # lanes < 32: warp 2k, else 2k + 1
    # every lane the 8 warp sums
    ps = []
    for k in range_constexpr(4):
        other = xshfl(warp[k], 32)
        lo_half = lane < 32
        ps.append(lo_half.select(warp[k], other))  # warp 2k
        ps.append(lo_half.select(other, warp[k]))  # warp 2k + 1
    total = ((ps[0] + ps[4]) + (ps[2] + ps[6])) + ((ps[1] + ps[5]) + (ps[3] + ps[7]))
    nf = _rsqrt(f, total / fx.Float32(float(KVD)) + fx.Float32(f.kv_eps))
    data = [((xs[k][0] * nf) * ws[k][0], (xs[k][1] * nf) * ws[k][1]) for k in range(4)]
    pos = fx.Int32(bo.buffer_load(rsrc(a["pos"]), 2 * grow, vec_width=1, dtype=T.i32))
    if const_expr(dests is None):
        r_d, r_sc, ok = _cache_row(a, grow)
        dests = [(r_d, r_sc, 0)]
        ok = valid & ok
    else:
        ok = valid
    # warp 7: lanes 32 .. 63 of k = 3, pair lane - 32 of the rotated dims
    pr = fx.min(fx.max(lane - 32, fx.Int32(0)), fx.Int32(31))
    fr = fx.Vector(bo.buffer_load(rsrc(a["freqs"]), pos * ROPE + 2 * pr, vec_width=2, dtype=T.f32))
    xr, xi = data[3]
    if const_expr(f.variant & 2):
        d0 = fma(-xi, fr[1], xr * fr[0])
        d1 = fma(xi, fr[0], xr * fr[1])
    else:
        d0 = fma(xr, fr[0], -(xi * fr[1]))
        d1 = fma(xr, fr[1], xi * fr[0])
    rope_w = bf16_pair(d0, d1).bitcast(fx.Int32)
    for r_d, _, cm in dests:
        if ok & (lane >= 32):
            bo.buffer_store(rope_w, r_d, NOPE // 4 + (lane - 32), cache_modifier=cm)
    for k in range_constexpr(4):
        x, y = data[k]
        amax = butterfly(fx.max(abs(x), abs(y)), (16, 8, 4, 2, 1), mx)
        scale_raw = fx.max(fx.Float32(1e-4), amax) / fx.Float32(448.0)
        bits = scale_raw.bitcast(fx.Int32)
        e8 = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
        inv = ((254 - e8) << 23).bitcast(fx.Float32)
        w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, clamp_fp8(x * inv), clamp_fp8(y * inv), fx.Int32(0), False))
        live = ok if k < 3 else ok & (lane < 32)
        t = lane + 64 * k
        for r_d, r_sc, cm in dests:
            if live:
                bo.buffer_store(fx.Int16(w8), r_d, t, cache_modifier=cm)
            if live & ((lane % 32) == 0):
                bo.buffer_store(fx.Int8(e8), r_sc, 2 * k + lane // 32, cache_modifier=cm)


@flyc.jit
def row_cta(tid, j, p, a, own, phase, c, bases, rank):
    """Row stage CTA j: rows 4 j .. 4 j + 3 of the seam's, a wave a row; each waits for the
    wqkv_a CTAs that reduce its row (every column tile, the split holding the row), then runs
    the q norm and quant and the KV row's norm, RoPE and store. With ``gather`` the q norm's
    operand and the KV record go to every rank's peer buffer at the step's row, and the CTA's
    flag to every rank once they are stored."""
    f, ms = p.front, p.seam_rows
    tl = f.qkv
    lane, wave = tid % 64, tid // 64
    row = j * ROWS_A_CTA + wave
    rc = fx.min(row, ms - 1)
    grow = rc + rank * ms if p.tpar else rc
    stamp(c, MARKS["row"])
    mtiles, ntiles = _cdiv(ms, tl.bm), QKV // tl.bn
    mt = rc // tl.bm
    split = (rc % tl.bm) // (tl.bm // tl.ks)
    nt = fx.min(lane, ntiles - 1)
    cta = p.bases["qkv"] + _gemm_cta(tl, ms, QKV, nt * mtiles + mt, split)
    _await_ctas(c, lane, own, [cta], phase, 6)
    stamp(c, MARKS["row"] + 1)
    valid = row < ms
    if const_expr(p.gather):
        g = p.peer
        qd = [(rsrc(bases[d] + fx.Int64(g.gq)), rsrc(bases[d] + fx.Int64(g.gqs)), grow, CM_SYS) for d in range(p.tp)]
        kd = []
        for d in range_constexpr(p.tp):
            rec = bases[d] + fx.Int64(g.gkv) + fx.Int64(grow) * fx.Int64(REC)
            kd.append((rsrc(rec), rsrc(rec + fx.Int64(KV_DATA)), CM_SYS))
    else:
        qd = [(rsrc(a["qq"]), rsrc(a["qs"]), rc, 0)]
        kd = None
    qnorm_row(f, lane, a, rc, valid, qd)
    krope_row(f, lane, a, rc, grow, valid, kd)
    stamp(c, MARKS["row"] + 2)
    if const_expr(p.gather):
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid == 0:
            for d in range_constexpr(p.tp):
                bo.buffer_store(phase, rsrc(bases[d]), p.peer.gflag // 4 + rank * GROW_CTAS + j, cache_modifier=CM_SYS)
    stamp(c, MARKS["row"] + 3)


def _await_rows(c, lane, p, own, local, lo, count, most, phase):
    """Wait (one wave) until the q norms of the step's rows lo .. lo + count (at most
    ``most``) are in: the row stage's CTAs, here, or with ``gather`` their owners' flags in
    this rank's peer buffer."""
    per = _cdiv(most, 64)
    rows = [lo + fx.min(lane * per + i, count - 1) for i in range(per)]
    if const_expr(not p.gather):
        _await_ctas(c, lane, own, [p.bases["row"] + r // ROWS_A_CTA for r in rows], phase, 7)
        return
    n = p.seam_rows
    r_g = rsrc(local + fx.Int64(p.peer.gflag))
    idx = [(r // n) * GROW_CTAS + (r % n) // ROWS_A_CTA for r in rows]

    def load_all():
        return fx.Vector.from_elements(
            [fx.Int32(bo.buffer_load(r_g, i, vec_width=1, dtype=T.i32, cache_modifier=CM_SYS)) for i in idx],
            fx.Int32,
        )

    def pending(v):
        out = v[0] != phase
        for i in range(1, len(idx)):
            out = out | (v[i] != phase)
        return out

    _wait(c, load_all, pending, 8)


@flyc.jit
def rec_cta(tid, j, p, a, own, local, phase, c):
    """Record stage CTA j (``gather``): rows 4 j .. of the step, a wave a row: once the row's
    owner has published its KV record here, the record into this rank's cache slot."""
    lane, wave = tid % 64, tid // 64
    row = fx.min(j * ROWS_A_CTA + wave, p.m - 1)
    stamp(c, MARKS["rec"])
    _await_rows(c, lane, p, own, local, row, 1, 1, phase)
    stamp(c, MARKS["rec"] + 1)
    r_d, r_sc, ok = _cache_row(a, row)
    ok = ok & (j * ROWS_A_CTA + wave < p.m)
    rec = rsrc(local + fx.Int64(p.peer.gkv) + fx.Int64(row) * fx.Int64(REC))
    pc = fx.min(lane, KV_DATA // 16 - 1)
    v = fx.Vector(bo.buffer_load(rec, 4 * pc, vec_width=4, dtype=T.i32, cache_modifier=CM_SYS))
    sv = fx.Vector(bo.buffer_load(rec, KV_DATA // 4, vec_width=2, dtype=T.i32, cache_modifier=CM_SYS))
    if ok & (lane < KV_DATA // 16):
        bo.buffer_store(v, r_d, 4 * lane)
    if ok & (lane == 0):
        bo.buffer_store(fx.Int32(sv[0]), r_sc, 0)
        bo.buffer_store(fx.Int16(fx.Int32(sv[1])), r_sc, 2)
        bo.buffer_store(fx.Int8(fx.Int32(sv[1]) >> 16), r_sc, 6)
    stamp(c, MARKS["rec"] + 2)


@flyc.jit
def store_out_rope(tid, tl, acc, n, m, m0, n0, out, lds, freqs, pos):
    """``fp8_wide.store_out`` with the query RoPE: the tile's bf16 rows, whose columns 448 ..
    511 of each 512-wide head rotated by pair (the cross product rounded, then one fma with
    the cosine; bf16 in, bf16 out), as the K launch ropes the query in place."""
    lane, wave = tid % 64, tid // 64
    tm, tn = tl.bm // tl.wm, tl.bn // tl.wn
    wy, wx = wave // tl.wn, wave % tl.wn
    pieces = tl.bn // 8
    swz = min(pieces, 16)
    f8._step_barrier(0)
    for a_ in range_constexpr(tm // 16):
        for b_ in range_constexpr(tn // 16):
            row = wy * tm + 16 * a_ + lane % 16
            col = wx * tn + 16 * b_ + 4 * (lane // 16)
            v = acc[a_][b_]
            pk = fx.Vector.from_elements([v[0], v[1], v[2], v[3]], fx.Float32).to(fx.BFloat16)
            piece = (col // 8) ^ (row % swz)
            fx.ptr_store(pk.bitcast(fx.Int32), lds + row * (pieces * 4) + piece * 4 + (col % 8) // 2)
    f8._step_barrier(0)
    r_o = rsrc(out, m * n * 2)
    r_f, r_p = rsrc(freqs), rsrc(pos)
    v4 = fx.Vector.make_type(4, fx.Int32)
    tot = tl.bm * pieces
    for i in range_constexpr(_cdiv(tot, tl.cthreads)):
        idx = tid + tl.cthreads * i
        if const_expr(tot % tl.cthreads):
            idx = fx.min(idx, tot - 1)
        row, cp = idx // pieces, idx % pieces
        v = fx.Vector(fx.ptr_load(lds + row * (pieces * 4) + (cp ^ (row % swz)) * 4, result_type=v4))
        hc = (n0 + 8 * cp) % HD
        rot = hc >= NOPE
        pr = fx.max(hc - NOPE, fx.Int32(0)) // 2  # the piece's first pair
        ps = fx.Int32(bo.buffer_load(r_p, 2 * fx.min(m0 + row, m - 1), vec_width=1, dtype=T.i32))
        cs = [fx.Vector(bo.buffer_load(r_f, ps * ROPE + 2 * pr + 4 * h, vec_width=4, dtype=T.f32)) for h in range(2)]
        outw = []
        for q in range_constexpr(4):
            qx, qy = bf_lo(v[q]), bf_hi(v[q])
            cos_v, sin_v = cs[q // 2][2 * (q % 2)], cs[q // 2][2 * (q % 2) + 1]
            rr = -(qy * sin_v)
            ri = qx * sin_v
            w = bf16_pair(fma(qx, cos_v, rr), fma(qy, cos_v, ri)).bitcast(fx.Int32)
            outw.append(rot.select(w, fx.Int32(v[q])))
        bo.buffer_store(fx.Vector.from_elements(outw, fx.Int32), r_o, ((m0 + row) * n + n0 + 8 * cp) // 2)


@flyc.jit
def wqb_cta(tid, j, p, a, lds, own, local, phase, c):
    """wq_b task j: waits for the row stage's CTAs of its rows, then the tile with the query
    RoPE into the query (bf16 [rows, heads * 512])."""
    f, ms = p.front, p.q_rows
    tl = f.wqb
    n = f.n_wqb
    lane, wave = tid % 64, tid // 64
    mtiles = _cdiv(ms, tl.bm)
    m0, n0, tile, split = _task_origin(j, tl, mtiles, n // tl.bn)
    stamp(c, MARKS["wqb"])
    if wave == 0:
        _await_rows(c, lane, p, own, local, m0, fx.min(ms - m0, tl.bm), tl.bm, phase)
    gpu.barrier()
    stamp(c, MARKS["wqb"] + 1)
    if const_expr(p.gather):
        xq, xs = local + fx.Int64(p.peer.gq), local + fx.Int64(p.peer.gqs)
    else:
        xq, xs = a["qq"], a["qs"]
    acc = f8.gemm_task(tid, tl, n, QR, ms, m0, n0, split, xq, xs, a["wqb"], a["wqb_s"], lds)
    stamp(c, MARKS["wqb"] + 2)
    store_out_rope(tid, tl, acc, n, ms, m0, n0, a["q_out"], lds, a["freqs"], a["pos"])


# ---------------------------------------------------------------- the launch
def stages(p: Plan, name: str) -> dict:
    """The launch's stage table: each stage's tag, its CTA range [lo, hi) and its marks'
    base (the reading tools split a dispatch into its stages by it); a GEMM stage's leading
    idle CTAs (its 8-CTA alignment) count in the stage before it. ``ar_mode``: "sum", the
    launch sums the MoE output over the ranks, or "nosum", it reads a sum SGLang made. The
    row stage (the q norm, then the KV row's norm, RoPE and store) is tagged ``qnorm``, the
    gathered KV records' store into the cache ``krope``; the query RoPE runs in wq_b's
    epilogue."""
    b = p.bases
    s1, _ = p.seam_tasks
    ranges = []
    if p.sum_tasks:
        ranges.append(("ar", 0, MARKS["ar"]))
    ranges += [("seama", b["seama"], MARKS["seama"]), ("seama", b["seama"] + s1, MARKS["seama_row"])]
    if p.front is not None:
        ranges += [("qkv", b["qkv"], MARKS["qkv"]), ("qnorm", b["row"], MARKS["row"])]
        if p.gather:
            ranges.append(("krope", b["rec"], MARKS["rec"]))
        ranges.append(("wqb", b["wqb"], MARKS["wqb"]))
    out = []
    for i, (tag, lo, mark) in enumerate(ranges):
        hi = ranges[i + 1][1] if i + 1 < len(ranges) else b["end"]
        out.append({"stage": tag, "ctas": [lo, hi], "mark_base": mark})
    return {
        "kernel": name,
        "tp": p.tp,
        "width": p.m,
        "ar_mode": "sum" if p.mode is not None else "nosum",
        "stages": out,
    }


def _write_stages(p: Plan, name: str) -> None:
    """With FlyDSL's IR dump on, the stage table beside the kernel's dump."""
    root = os.environ.get("FLYDSL_DUMP_DIR")
    if root and os.environ.get("FLYDSL_DUMP_IR") == "1":
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.stages.json"), "w") as f:
            json.dump(stages(p, name), f)


def kernel_name(p: Plan, timeline: bool = False) -> str:
    f = p.seam
    fr = p.front
    front = (
        f"_q{fr.qkv.bm}x{fr.qkv.bn}k{fr.qkv.ks}d{fr.qkv.depth}_b{fr.wqb.bm}x{fr.wqb.bn}d{fr.wqb.depth}"
        f"_h{fr.heads}v{fr.variant}"
        if fr is not None
        else ""
    )
    return (
        f"ds41_spana_wide_m{p.m}_tp{p.tp}_ar{'sum' if p.mode else 'nosum'}_{p.mode or 'none'}"
        f"{'_own' if p.tpar else ''}{'_gather' if p.gather else ''}"
        f"_a{p.ac}w{p.window}_p{int(f.post)}c{int(f.combine)}_{f.quant or 'none'}{front}"
        f"{'_b' + str(p.bound) if p.bound else ''}"
        f"{'_tl' if timeline else ''}"
    )


ARGS = (
    "x_in", "scratch", "peers", "local", "rank", "res", "post_in", "comb_in", "pre_in", "fn",
    "hc_scale", "hc_base", "norm_w", "res_out", "normed", "q", "qs", "pre_out", "post_out",
    "comb_out", "wqkv", "wqkv_s", "qn_w", "kvn_w", "wqb", "wqb_s", "freqs", "pos", "out_loc",
    "kvcache", "q_out",
)  # fmt: skip

_BUILDS: dict = {}


def build(p: Plan, timeline: bool = False):
    """``launch(*ARGS, stream=)`` on device addresses: ``x_in`` the MoE output on this rank,
    bf16 [m, 5120] (with mode None: the seam's x, summed already), ``scratch`` of
    ``scratch_bytes(p)`` (uncached; its flags zeros before the first launch, ``flag_bytes``),
    ``peers`` the TP group's peer buffer addresses (``PeerBuffer.addresses``; its flags
    zeros; unused without a mode), ``local`` this rank's address of it, ``rank`` this rank,
    then the seam's tensors (``seam_wide.ARGS``, for the seam's rows; with a front, q and qs
    are the scratch's xq and xs), then the front's (unused without one): wqkv_a's e4m3
    [1792, 5120] and codes (``fp8_wide.pack_scales``), the q and kv norm weights (bf16),
    wq_b's e4m3 [heads * 512, 1280] and codes, the RoPE table (f32 [positions, 64]: cos, sin
    a pair), the rows' positions (int64), cache slots (int32) and the layer's SWA cache, and
    the query out (bf16 [rows, heads * 512]). ``timeline``: each CTA marks its stage
    (``MARKS``)."""
    key = (p, timeline)
    if key in _BUILDS:
        return _BUILDS[key]
    assert (p.mode is not None or p.gather) == (p.peer is not None), p
    keyed = key_tuple({"p": p, "timeline": timeline}, SOURCES)
    lay = scratch_layout(p)
    ns = p.sum_tasks
    b = p.bases
    seam_end = ns + sum(p.seam_tasks)
    name = kernel_name(p, timeline)
    _write_stages(p, name)
    seam_lay = sw.scratch_layout(p.seam_rows)
    fr = p.front

    @fx.struct
    class Smem:
        x: fx.Array[fx.Int32, p.lds_words, 16]

    def off(region):
        return fx.Int64(lay[region][0])

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def spana_kernel(
        x_in: Int64, scratch: Int64, peers: Int64, local: Int64, rank: fx.Int32,
        res: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64, fn: Int64, hc_scale: Int64,
        hc_base: Int64, norm_w: Int64, res_out: Int64, normed: Int64, q: Int64, qs: Int64,
        pre_out: Int64, post_out: Int64, comb_out: Int64,
        wqkv: Int64, wqkv_s: Int64, qn_w: Int64, kvn_w: Int64, wqb: Int64, wqb_s: Int64,
        freqs: Int64, pos: Int64, out_loc: Int64, kvcache: Int64, q_out: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        wl = fx.SharedAllocator().allocate(Smem).peek().x.ptr
        own = scratch + off("own")
        r_own = rsrc(own)
        c = {"tid": tid}
        if const_expr(p.bound):
            c["bound"] = p.bound
            c["err"] = scratch + off("err") + fx.Int64(bid) * fx.Int64(4)
        if const_expr(timeline):
            c["tl"] = timeline_row(scratch, lay["tl"][0], bid)
        old = fx.Int32(
            bo.buffer_load(r_own, bid, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)
        )
        phase = 1 - old
        if const_expr(fr is not None):
            q, qs = scratch + off("xq"), scratch + off("xs")
            fa = {
                "xq": q, "xs": qs, "qkv": scratch + off("qkv"), "part": scratch + off("part"),
                "pflags": scratch + off("pflags"), "qq": scratch + off("qq"),
                "qs": scratch + off("qs"), "wqkv": wqkv, "wqkv_s": wqkv_s, "qn_w": qn_w,
                "kvn_w": kvn_w, "wqb": wqb, "wqb_s": wqb_s, "freqs": freqs, "pos": pos,
                "out_loc": out_loc, "kvcache": kvcache, "q_out": q_out,
            }  # fmt: skip
        if const_expr(ns > 0):
            if bid < ns:
                stamp(c, MARKS["ar"])
                stamp(c, MARKS["ar"] + 1)
                sum_task(tid, p, bid, x_in, peer_bases(peers, p.tp), local, scratch + off("pushed"), rank, phase, c)
                stamp(c, MARKS["ar"] + 2)
                stamp(c, MARKS["ar"] + 3)
        if (bid >= ns) & (bid < seam_end):
            if const_expr(p.mode is None):
                xa = x_in
            else:
                xa = local + fx.Int64(p.peer.x)
                if const_expr(p.tpar):
                    xa = xa + fx.Int64(rank * p.own * H * 2)
            # the seam's LDS words are f32 (its loads take the pointer's element type)
            seam_cta(
                tid, bid - ns, p, fx.recast_iter(fx.Float32, wl), own, phase, rank, c, {
                    "x": xa, "res": res, "post_in": post_in, "comb_in": comb_in,
                    "pre_in": pre_in, "fn": fn, "hc_scale": hc_scale, "hc_base": hc_base,
                    "norm_w": norm_w, "res_out": res_out, "normed": normed, "q": q, "qs": qs,
                    "pre_out": pre_out, "post_out": post_out, "comb_out": comb_out,
                    "scratch": scratch + off("seam"), "flags": scratch + off("sflags"),
                    **{k: v[0] for k, v in seam_lay.items()},
                },
            )  # fmt: skip
        if const_expr(fr is not None):
            if (bid >= b["qkv"]) & (bid < b["row"]):
                qkv_cta(tid, bid - b["qkv"], p, fa, wl, own, phase, c)
            if (bid >= b["row"]) & (bid < b["row"] + _cdiv(p.seam_rows, ROWS_A_CTA)):
                row_cta(tid, bid - b["row"], p, fa, own, phase, c, peer_bases(peers, p.tp) if p.gather else None, rank)
            if const_expr(p.gather):
                if (bid >= b["rec"]) & (bid < b["rec"] + _cdiv(p.m, ROWS_A_CTA)):
                    rec_cta(tid, bid - b["rec"], p, fa, own, local, phase, c)
            if bid >= b["wqb"]:
                wqb_cta(tid, bid - b["wqb"], p, fa, wl, own, local, phase, c)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid == 0:
            bo.buffer_store(phase, r_own, bid, cache_modifier=CM_DEV)

    @flyc.jit
    def launch(
        x_in: Int64, scratch: Int64, peers: Int64, local: Int64, rank: fx.Int32,
        res: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64, fn: Int64, hc_scale: Int64,
        hc_base: Int64, norm_w: Int64, res_out: Int64, normed: Int64, q: Int64, qs: Int64,
        pre_out: Int64, post_out: Int64, comb_out: Int64,
        wqkv: Int64, wqkv_s: Int64, qn_w: Int64, kvn_w: Int64, wqb: Int64, wqb_s: Int64,
        freqs: Int64, pos: Int64, out_loc: Int64, kvcache: Int64, q_out: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):  # fmt: skip
        _ = keyed
        spana_kernel(
            x_in, scratch, peers, local, rank, res, post_in, comb_in, pre_in, fn, hc_scale,
            hc_base, norm_w, res_out, normed, q, qs, pre_out, post_out, comb_out, wqkv,
            wqkv_s, qn_w, kvn_w, wqb, wqb_s, freqs, pos, out_loc, kvcache, q_out,
        ).launch(grid=(p.grid,), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch


def front_tiles(m: int, tp: int) -> tuple:
    """(wqkv_a's, wq_b's) tile at m rows (``fp8_wide.TILES`` at most as deep as two CTAs a
    CU allow beside the seam's LDS)."""
    budget = 4 * (sw.FN_WORDS + sw.WAVES * (sw.HC + 1) * sw.TILE)
    c = f8.width_class(m)
    out = []
    for n, k in ((QKV, H), (64 * HD // tp, QR)):
        tl = f8.TILES[(n, k, c)]
        depth = min(tl.depth, budget // tl.stage_bytes)
        if depth < 2 or tl.wreg:
            tl = f8.Tile(96, 128, 2, 2, 1, 2)
            depth = 2
        out.append(msgspec.structs.replace(tl, depth=depth))
    return tuple(out)


def peer_layout(tp: int, plans: dict) -> tuple[int, dict]:
    """The TP group's peer buffer for the widths of ``plans`` (key -> ``Plan`` without its
    peer regions): (bytes, key -> its ``Plan`` with them). The row regions are shared
    (launches run in stream order, and a collective of the layer's attention back sits
    between two of them); the flags are a key's own (each flips once a launch of its
    plan); every byte from ``flag_region(tp)`` on starts as zero."""
    rows = MAXROWS * H * 2
    slot, x = 0, tp * rows
    gq = x + rows
    gqs = gq + MAXROWS * QR
    gkv = gqs + MAXROWS * QR // 32
    off = flag_region(tp)
    ats = {}
    for key in sorted(plans):
        n = plans[key].sum_flags
        ats[key] = (off, off + 4 * tp * n, off + 8 * tp * n)
        off = _cdiv(off + 8 * tp * n + 4 * tp * GROW_CTAS, LINE) * LINE
    out = {
        key: msgspec.structs.replace(
            p, peer=PeerAt(slot, x, ats[key][0], ats[key][1], off, gq, gqs, gkv, ats[key][2])
        )
        if p.mode is not None or p.gather
        else p
        for key, p in plans.items()
    }
    return off, out


def flag_region(tp: int) -> int:
    """Where the peer buffer's flags start (``peer_layout``): every byte from here starts as
    zero."""
    rows = (tp + 1) * MAXROWS * H * 2 + MAXROWS * (QR + QR // 32 + REC)
    return _cdiv(rows, LINE) * LINE
