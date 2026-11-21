# SPDX-License-Identifier: Apache-2.0
"""``mla_decode_b``: one Kimi-K3 MLA layer's DCP8 verify decode at the tail's
widths, e4m3 Q K and e4m3 P V, with the O accumulator split along the latent.
Same inputs as ``mla_decode``; its output is every row's merged o (bf16) and
LSE (fp32), the split partials merged in the launch itself after a grid
barrier, so the tail's reduce has nothing left to do.

One launch, ``BLOCKS`` CTAs of 4 waves (one a SIMD, a CTA a CU), persistent.

    plan      (``build_plan``, once a step) a request's key splits are equal
              64-key-block runs of about a CTA's share of the batch's work; a
              split's 576 (row, head) columns are 6 items of 16 heads x 6 rows
              (96 columns); items are dealt longest first in snake rounds over
              XCD-major places, so one split's items share an XCD's L2
    a block   64 keys of the item's split, row-major in an LDS ring of two
              slots (a 592-byte padded row a key); every thread loads its 9
              16-byte pieces of block b + 1 (4 threads a key) into registers
              during block b and writes them to the ring between the block's
              two barriers

A block, the 4 waves together:

    QK        S [96 columns x 64 keys] once: wave w takes key half w % 2 and
              column half w / 2 (48 columns = 3 rows x 16 heads), its q in
              registers; v_mfma_scale_f32_16x16x128 e4m3 (M = 16 keys, N = 16
              columns, K = 576 = 4.5 steps, the last half zero)
    softmax   a column's max over the wave's 32 keys, exchanged with the key
              half partner through LDS (barrier A); the exact running max m;
              P = e4m3(2^(s c - m + PRE)) to LDS as [column][key]; the
              rescale alpha of each column to LDS (barrier B)
    PV        O^T [512 x 96] split along the latent: wave w owns dims 128 w ..
              + 128 for all 96 columns (12 tiles of 32x32 fp32); O^T += V^T
              P^T as v_mfma_scale_f32_32x32x64 e4m3, V^T read transposed from
              the ring by ds_read_b64_tr_b8, P^T straight from LDS

The row sum is the fp32 sum of the unrounded P (a key half's on its QK wave,
both halves added at the end); a split's LSE is natural-log fp32. Numerics
against B300's FlashInfer FP8 MLA decode: the same e4m3 x e4m3 products into
fp32, the exact running max of every 64-key block, and P carrying a static
2^PRE so that values B300 flushes below e4m3's normal range stay normal here.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from sgl_amd_mono.common.ops import CM_DEV, ballot, hw_exp2, popcount, rsrc, uniform, xred
from sgl_amd_mono.common.plan import BLOCKS, kernel_sources, key_tuple
from sgl_amd_mono.kimi_k3.gfx950.attention import mla_decode as md

SOURCES = kernel_sources(__name__)

NHA, KV, DQK, G, DCP = md.NHA, md.KV, md.DQK, md.G, md.DCP
THREADS = 256
WAVES = THREADS // 64
BK = 64  # keys a block
ROWW = (DQK // 16 + 1) * 4  # 148 words a key row in LDS: 36 pieces of 16 bytes, then a pad
SLOT_W = BK * ROWW
NSLOT = 2
PIECES = DQK // 16  # 36 pieces a key
LD_N = BK * PIECES // THREADS  # 9 pieces a thread a block
HG = 16  # heads an item
NG = NHA // HG  # 6 items a split
COLS = G * HG  # 96 columns an item: column c = row c / 16, head c % 16
PROW = 20  # words a column of P in LDS: 64 keys, then a pad
KSTEPS = 5  # QK K steps of 128 (the last: dims 512 .. 575 on lane groups 0, 1)
PRE = 8.0
NEG = -1e30  # the running max before any visible key
NINF = float("-inf")  # a masked key's score
LN2 = 0.6931471805599453
LOG2E = 1.4426950408889634
MAX_REQS = md.MAX_REQS
MAX_SPLITS = md.MAX_SPLITS
W_NUM, W_DEN = 11, 10  # a split: W_NUM / W_DEN of a CTA's share of the batch's work (/ k)
ITEM_OVH = 5  # an item's setup and partial stores, in blocks (measured ~12 us an item at ~2.4 us a block)
MAX_PART = md.MAX_PART
TBL, TBL_WORDS = md.TBL, md.TBL_WORDS
T_R, T_KIND, T_N, T_NB, T_WR, T_IPRE, T_L, T_P, T_PB, T_KVP = range(10)
MERGE_K = 8  # a merge's split loads in flight
T_NTILE = 2 * MAX_REQS * TBL + 1  # the plan table's word: the requests with more than one split


class DecodeBuffers(md.DecodeBuffers):
    """``mla_decode.DecodeBuffers`` (the split partials and the reduce's maps)
    and the merge's arrival counter (never reset: a launch waits for the next
    multiple of BLOCKS)."""

    def __init__(self, device):
        super().__init__(device)
        self.sync = torch.zeros(4, dtype=torch.int32, device=device)


@dataclass(frozen=True)
class MlaDecodeBuild:
    scale: float  # the softmax scale
    # timing probes only (checks): parts of a block's work left out
    probe: str = ""


_BUILDS: dict = {}
_STREAM = fx.Stream(None)

_ld, _ld4, _lds16, _tr8, _cat8, _e4m3x4, _memtime, _realtime, _bar = (
    md._ld, md._ld4, md._lds16, md._tr8, md._cat8, md._e4m3x4, md._memtime, md._realtime, md._bar,
)  # fmt: skip


def _wave_sum(v):
    """The wave's i32 sum (permlane swaps and DPP, no LDS)."""
    for off in [32, 16, 8, 4, 2, 1]:
        v = xred(fx.Int32(v), off, lambda x, y: x + y)
    return v


def _wave_max(v):
    for off in [32, 16, 8, 4, 2, 1]:
        v = xred(fx.Int32(v), off, fx.max)
    return v


def _atomic_add(addr, val):
    """Device-scope atomic add of ``val`` on the int32 at global address ``addr``; its old value."""
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(addr)).result
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, ptr, as_ir_value(fx.Int32(val)), llvm.AtomicOrdering.monotonic, syncscope="agent"
    )  # fmt: skip
    return fx.Int32(op.result)


@flyc.jit
def _grid_barrier(sync, tid):
    """Every CTA of the launch past this point (all BLOCKS resident, one a CU), its device-
    coherent stores done: arrivals count up from the last launch's, never reset."""
    rocdl.s_waitcnt(vmcnt=0)
    _bar()
    if tid == 0:
        old = _atomic_add(fx.Int64(sync), 1)
        target = (old & fx.Int32(-BLOCKS)) + BLOCKS
        r = rsrc(sync)
        v = fx.Int32(bo.buffer_load(r, 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
        while v - target < 0:
            rocdl.s_sleep(1)
            v = fx.Int32(bo.buffer_load(r, 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
    _bar()


@flyc.jit
def _merge(o, lse, logits, plse, indptr, fmap, pmap, ntile, bid, wave, lane):
    """The split requests' (row, head) units, one a wave in turn over the launch's waves:
    lane s takes split s's LSE (natural log, fp32); the max, the weights e^(lse - m) and
    their sum; then the splits' fp32 partials (lane l: dims 8 l .. + 8) weighted in split
    order; o in bf16 and lse = m + log(sum) to the request's rows."""
    units = ntile * (G * NHA)
    for u in range(bid * WAVES + wave, units, BLOCKS * WAVES):
        t = u // (G * NHA)
        j = (u % (G * NHA)) // NHA
        h = u % NHA
        p0 = uniform(_ld(indptr, t))
        ns = uniform(_ld(indptr, t + 1)) - p0
        row = uniform(_ld(fmap, 2 * t)) + j
        prow = _ld(pmap, p0 + fx.min(lane, ns - 1)) + j
        ls = (lane < ns).select(
            fx.Float32(bo.buffer_load(rsrc(plse), prow * NHA + h, vec_width=1, dtype=T.f32, cache_modifier=CM_DEV)), fx.Float32(NEG)
        )
        m = _wave_fmax(ls)
        w = (lane < ns).select(hw_exp2((ls - m) * LOG2E), fx.Float32(0.0))
        ws = _wave_fsum(w)
        acc = [fx.Float32(0.0)] * 8
        # MERGE_K splits' loads in flight, then their weighted sums in split order
        for c0 in range(0, ns, MERGE_K):
            parts = []
            for q in range_constexpr(MERGE_K):
                sp = fx.min(c0 + q, ns - 1)
                wsp = fx.Int32(rocdl.readlane(T.i32, w.bitcast(fx.Int32), sp)).bitcast(fx.Float32)
                pr = fx.Int32(rocdl.readlane(T.i32, prow, sp))
                base = (pr * NHA + h) * KV + 8 * lane
                lo = fx.Vector(bo.buffer_load(rsrc(logits), base, vec_width=4, dtype=T.f32, cache_modifier=CM_DEV))
                hi = fx.Vector(bo.buffer_load(rsrc(logits), base + 4, vec_width=4, dtype=T.f32, cache_modifier=CM_DEV))
                parts.append(((c0 + q < ns).select(wsp, fx.Float32(0.0)), lo, hi))
            for wsp, lo, hi in parts:
                acc = [acc[e] + wsp * fx.Float32((lo if e < 4 else hi)[e % 4]) for e in range(8)]
        inv = fx.Float32(1.0) / ws
        ob = fx.Vector.from_elements([acc[e] * inv for e in range(8)], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        bo.buffer_store(ob, rsrc(o), ((row * NHA + h) * KV + 8 * lane) // 2)
        if lane == 0:
            bo.buffer_store(m + fx.Float32(rocdl.log(T.f32, ws.ir_value())) * LN2, rsrc(lse), row * NHA + h)


def _wave_fmax(v):
    for off in [32, 16, 8, 4, 2, 1]:
        v = xred(fx.Float32(v), off, fx.max)
    return v


def _wave_fsum(v):
    for off in [32, 16, 8, 4, 2, 1]:
        v = xred(fx.Float32(v), off, lambda x, y: x + y)
    return v


def _st16(ptr, v):
    llvm.store(as_ir_value(v), ptr.llvm_ptr, alignment=16)


def _mfma32(a, b, c):
    one = fx.Int32(127)
    return fx.Vector(rocdl.mfma_scale_f32_32x32x64_f8f6f4(T.vec(16, T.f32), [a, b, c, 0, 0, 0, one, 0, one]))


def _mfma16(a, b, c):
    one = fx.Int32(127)
    return fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one]))


def _kdim(s, g):
    """First dim of QK K step s's 32 bytes on lane group g (step 4: groups 0, 1
    real, groups 2, 3 a copy of them against a zero q)."""
    return 128 * s + 32 * g if s < KSTEPS - 1 else 512 + 32 * (g % 2)


@flyc.jit
def _rows(c, bi):
    """Lane l's key of the item's block bi: its record's row and whether it is
    a key of the split (``_load`` takes row 0 for the others, so this load's
    wait stays at its use, a block later)."""
    blk = c["b0"] + bi
    kk = blk * BK + c["lane"]
    ok = (blk < c["b1"]) & (kk < c["L"])
    idx = _ld(c["kv_idx"], c["kvp"] + fx.max(fx.Int32(0), fx.min(kk, c["L"] - 1)))
    return [idx, ok.select(fx.Int32(1), fx.Int32(0))]


def _row_of(c, rows):
    """This thread's key row of a block (``rows``: ``_rows`` of it): thread t takes key t / 4's
    16-byte pieces t % 4 + 4 k; a key past the split reads row 0."""
    rv = (rows[1] != 0).select(rows[0], fx.Int32(0))
    row = fx.Int32(rocdl.ds_bpermute(T.i32, (c["tid"] // 4) * 4, rv))
    return row * (DQK // 4) + (c["tid"] % 4) * 4


def _load1(c, base, k):
    return fx.Vector(bo.buffer_load(c["r_kv"], base + 16 * k, vec_width=4, dtype=T.i32))


def _store1(c, v, slot, k):
    _st16(c["ring"] + slot * SLOT_W + (c["tid"] // 4) * ROWW + (c["tid"] % 4) * 4 + 16 * k, v)


@flyc.jit
def _load(c, rows):
    """This thread's 9 pieces of a block into registers."""
    base = _row_of(c, rows)
    return [_load1(c, base, k) for k in range_constexpr(LD_N)]


@flyc.jit
def _store(c, stg, slot):
    for k in range_constexpr(LD_N):
        _store1(c, stg[k], slot, k)


PLAN_CANDS = ((1, True), (1, False), (2, True), (3, True), (4, True))  # (k, equal runs)
PLAN_WAVES = len(PLAN_CANDS)
PLAN_SCR = PLAN_WAVES * 768 + 8


def _layout(nb, work, kk, equal):
    """A request's split count and run (blocks) for candidate (kk, equal)."""
    t = fx.max(fx.Int32(1), (work * W_NUM + BLOCKS * W_DEN * kk - 1) // (BLOCKS * W_DEN * kk))
    nk = fx.min(fx.Int32(MAX_SPLITS), fx.max(fx.Int32(1), (nb + t - 1) // t))
    if equal:
        wk = fx.max(fx.Int32(1), (nb + nk - 1) // nk)
    else:
        wk = (nk >= MAX_SPLITS).select(fx.max(fx.Int32(1), (nb + MAX_SPLITS - 1) // MAX_SPLITS), t)
    return fx.max(fx.Int32(1), (nb + wk - 1) // wk), wk


def _estimate(nb, live, nk, wk, scr, lane):
    """A lower bound of the snake deal's busiest CTA for a layout (lane r: request r; its items'
    costs blocks + ITEM_OVH), or 2^26 past the partials' room."""
    ca = wk + ITEM_OVH
    cb = nb - (nk - 1) * wk + ITEM_OVH
    na = live.select(NG * (nk - 1), fx.Int32(0))
    nbb = live.select(fx.Int32(NG), fx.Int32(0))
    parts = _wave_sum((live & (nk > 1)).select(nk, fx.Int32(0)))
    total = _wave_sum(na * ca + nbb * cb)
    est = fx.max((total + BLOCKS - 1) // BLOCKS, _wave_max(live.select(fx.max(ca, cb), fx.Int32(0))))
    # each class's place in the costs sorted down: [items costlier, items at least as costly)
    _st16(scr + 4 * lane, fx.Vector.from_elements([ca, na, cb, nbb], fx.Int32))
    ea, eb, sa, sb = fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0)
    for r2 in range(MAX_REQS):  # lanes past the batch count no items
        v = _lds16(scr + 4 * r2)
        ca2, na2, cb2, nb2 = fx.Int32(v[0]), fx.Int32(v[1]), fx.Int32(v[2]), fx.Int32(v[3])
        ea = ea + (ca2 >= ca).select(na2, fx.Int32(0)) + (cb2 >= ca).select(nb2, fx.Int32(0))
        eb = eb + (ca2 >= cb).select(na2, fx.Int32(0)) + (cb2 >= cb).select(nb2, fx.Int32(0))
        sa = sa + (ca2 > ca).select(na2, fx.Int32(0)) + (cb2 > ca).select(nb2, fx.Int32(0))
        sb = sb + (ca2 > cb).select(na2, fx.Int32(0)) + (cb2 > cb).select(nb2, fx.Int32(0))
    ea, sa = (na > 0).select(ea, fx.Int32(0)), (na > 0).select(sa, fx.Int32(0))
    eb, sb = (nbb > 0).select(eb, fx.Int32(0)), (nbb > 0).select(sb, fx.Int32(0))
    # some CTA takes j + 1 of the 256 j + 1 costliest items: (j + 1) x the (256 j + 1)-th cost
    for j in range(1, 4):
        q = BLOCKS * j
        cq = _wave_max(fx.max((ea > q).select(ca, fx.Int32(0)), (eb > q).select(cb, fx.Int32(0))))
        est = fx.max(est, (j + 1) * cq)
    # the snake's first two rounds give place p the items p and 511 - p: classes X (this lane's)
    # and Y pair when some p < 256 in X has 511 - p in Y
    _st16(scr + 256 + 8 * lane, fx.Vector.from_elements([ca, sa, ea, fx.Int32(0)], fx.Int32))
    _st16(scr + 260 + 8 * lane, fx.Vector.from_elements([cb, sb, eb, fx.Int32(0)], fx.Int32))
    t2 = fx.Int32(0)
    for r2 in range(MAX_REQS):
        for h in range(2):
            v = _lds16(scr + 256 + 8 * r2 + 4 * h)
            cy, sy, ey = fx.Int32(v[0]), fx.Int32(v[1]), fx.Int32(v[2])
            for cx, sx, ex in ((ca, sa, ea), (cb, sb, eb)):
                lo = fx.max(sx, 2 * BLOCKS - ey)
                hi = fx.min(fx.min(ex, fx.Int32(BLOCKS)), 2 * BLOCKS - sy)
                t2 = fx.max(t2, (lo < hi).select(cx + cy, fx.Int32(0)))
    est = fx.max(est, _wave_max(t2))
    return (parts <= MAX_PART).select(est, fx.Int32(1 << 26))


@flyc.jit
def plan(c, tbl, scr, kv_ptr, g_ptr, nreq, indptr, fmap, pmap, bid, tid):
    """The step's split plan in LDS ``tbl``, by a CTA of 5 waves (``scr``:
    PLAN_SCR words of LDS for broadcasts between lanes). A request's splits
    are runs of at most t blocks, t = 1.1 x the batch's work a CTA / k (at
    most 64 splits): equal runs for k = 1 .. 4, or (k = 1) runs of t and the
    rest. Wave w estimates candidate w (``_estimate``); the least estimate
    wins, a later candidate only when 5% lower. Its items fall in two
    classes: A, the 6 items of each whole run, cost wr; B, the last run's 6,
    cost the rest. The classes go to ``tbl`` a row each, longest first: the
    deal order. CTA 0 also writes the CSR maps. Returns the item count."""
    lane = tid % 64
    wave = uniform(tid // 64)
    # lane r: request r (every wave)
    r = fx.min(lane, nreq - 1)
    live = lane < nreq
    L = live.select(_ld(kv_ptr, r + 1) - _ld(kv_ptr, r), fx.Int32(0))
    P = _ld(g_ptr, r + 1) - _ld(g_ptr, r) - G
    nb = (L + BK - 1) // BK
    work = _wave_sum(NG * nb)
    ests = scr + PLAN_WAVES * 768
    for ci in range_constexpr(PLAN_WAVES):
        if wave == ci:
            nk, wk = _layout(nb, work, *PLAN_CANDS[ci])
            est = _estimate(nb, live, nk, wk, scr + 768 * ci, lane)
            if lane == 0:
                fx.ptr_store(est, ests + ci)
    _bar()
    best = fx.Int32(1 << 26)
    n = fx.Int32(1)
    wr = fx.Int32(1)
    for ci in range_constexpr(PLAN_WAVES):
        nk, wk = _layout(nb, work, *PLAN_CANDS[ci])
        est = fx.Int32(fx.ptr_load(ests + ci))
        better = est * 20 < best * 19
        best = better.select(est, best)
        n = better.select(nk, n)
        wr = better.select(wk, wr)
    if tid < 64:
        rho = nb - (n - 1) * wr
        multi = live & (n > 1)
        msplits = multi.select(n, fx.Int32(0))
        pbase = fx.Int32(0)
        tile = fx.Int32(0)
        fx.ptr_store(msplits, scr + lane)
        for r2 in range_constexpr(MAX_REQS):
            m2 = fx.Int32(fx.ptr_load(scr + r2))
            pbase = (fx.Int32(r2) < r).select(pbase + m2, pbase)
            tile = ((fx.Int32(r2) < r) & (m2 > 0)).select(tile + 1, tile)
        if bid == 0:
            nsplit = _wave_sum(msplits)
            ntile = _wave_sum(multi.select(fx.Int32(1), fx.Int32(0)))
            if multi:
                bo.buffer_store(pbase, rsrc(indptr), tile)
                bo.buffer_store(fx.Int32(G) * r, rsrc(fmap), 2 * tile)
                bo.buffer_store(fx.Int32(G) * r + G, rsrc(fmap), 2 * tile + 1)
                for sp in range(n):
                    bo.buffer_store(G * (pbase + sp), rsrc(pmap), pbase + sp)
            if lane >= ntile:
                if lane <= MAX_REQS:
                    bo.buffer_store(nsplit, rsrc(indptr), lane)
        # lane c: class c % 2 of request c / 2
        cr = fx.min(lane // 2, nreq - 1)
        kind = lane % 2
        clive = lane < 2 * nreq

        def gather(v):
            return fx.Int32(rocdl.ds_bpermute(T.i32, cr * 4, fx.Int32(v)))

        n_c, nb_c, wr_c, rho_c = gather(n), gather(nb), gather(wr), gather(rho)
        L_c, P_c, pb_c = gather(L), gather(P), gather(pbase)
        kvp_c = gather(_ld(kv_ptr, r))
        cnt = (kind == 0).select(NG * (n_c - 1), fx.Int32(NG))
        cnt = clive.select(cnt, fx.Int32(0))
        cost = (cnt > 0).select((kind == 0).select(wr_c, rho_c), fx.Int32(-1))
        rk = fx.Int32(0)
        ipre = fx.Int32(0)
        fx.ptr_store(cost, scr + 64 + 2 * lane)
        fx.ptr_store(cnt, scr + 65 + 2 * lane)
        for c2 in range_constexpr(2 * MAX_REQS):  # classes past the batch: cost -1, no items
            k2 = fx.Int32(fx.ptr_load(scr + 64 + 2 * c2))
            i2 = fx.Int32(fx.ptr_load(scr + 65 + 2 * c2))
            before = (k2 > cost) | ((k2 == cost) & (fx.Int32(c2) < lane))
            rk = before.select(rk + 1, rk)
            ipre = before.select(ipre + i2, ipre)
        total = _wave_sum(cnt)
        if clive:
            row = tbl + rk * TBL
            for f, v in (
                (T_R, cr), (T_KIND, kind), (T_N, n_c), (T_NB, nb_c), (T_WR, wr_c), (T_IPRE, ipre),
                (T_L, L_c), (T_P, P_c), (T_PB, G * pb_c), (T_KVP, kvp_c),
            ):  # fmt: skip
                fx.ptr_store(fx.Int32(v), row + f)
        ntile_all = _wave_sum((live & (n > 1)).select(fx.Int32(1), fx.Int32(0)))
        if lane == 0:
            fx.ptr_store(total, tbl + 2 * MAX_REQS * TBL)
            fx.ptr_store(ntile_all, tbl + T_NTILE)
    _bar()
    return fx.Int32(fx.ptr_load(tbl + 2 * MAX_REQS * TBL))


def build(key: MlaDecodeBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    keyed = key_tuple(key, SOURCES)
    qk_c = key.scale * LOG2E
    probe = key.probe
    PBW = COLS * PROW

    @fx.struct
    class Smem:
        kv: fx.Array[fx.Int32, NSLOT * SLOT_W, 16]
        p: fx.Array[fx.Int32, PBW, 16]
        red: fx.Array[fx.Float32, 2 * COLS, 16]  # the key halves' column maxes
        alpha: fx.Array[fx.Float32, COLS, 16]
        lsum: fx.Array[fx.Float32, 2 * COLS, 16]
        mfin: fx.Array[fx.Float32, COLS, 16]
        tbl: fx.Array[fx.Int32, TBL_WORDS, 16]

    @flyc.kernel(name="k3_mono_mla_decode_b", known_block_size=[THREADS, 1, 1])
    def mla_decode_b(
        q8: Int64, kv: Int64, kv_idx: Int64, kv_ptr: Int64, g_ptr: Int64, nreq: Int32, rank: Int32,
        o: Int64, lse: Int64, logits: Int64, plse: Int64, ptbl: Int64, tl: Int64,
        indptr: Int64, fmap: Int64, pmap: Int64, sync: Int64,
    ):  # fmt: skip
        _ = keyed
        tlon = "tl" in probe
        z = fx.Int64(0)
        tk0 = _memtime() if const_expr(tlon) else z
        rk0 = _realtime() if const_expr(tlon) else z
        t_set, t_loop, t_epi, n_it, n_bl = z, z, z, z, z
        phon = "ph" in probe  # a block's phases (with "tl")
        ph = [z] * 6
        tb = z
        bid, tid = fx.block_idx.x, fx.thread_idx.x
        lane = tid % 64
        wave = uniform(tid // 64)
        kh = wave % 2  # QK key half
        ch = wave // 2  # QK column half
        c16 = lane % 16
        g = lane // 16
        n32 = lane % 32
        hb = lane // 32
        lds = fx.SharedAllocator().allocate(Smem).peek()
        ring, pbuf, red, alph, lsum, mfin, tbl = (
            lds.kv.ptr, lds.p.ptr, lds.red.ptr, lds.alpha.ptr, lds.lsum.ptr, lds.mfin.ptr, lds.tbl.ptr,
        )  # fmt: skip

        def kop(kbp, u):
            """QK MFMA u's A operand (K step u / 2, key tile u % 2) from the slot at ``kbp``."""
            kp = kbp + (32 * kh + 16 * (u % 2) + c16) * ROWW + _kdim(u // 2, g) // 4
            return _cat8(_lds16(kp), _lds16(kp + 4))

        def vop(kbp, d):
            """P V MFMA d's A operand (V^T, dims 128 wave + 32 d + n32; byte e = key 32 hb + e): four
            transposed reads; read rho's source lane grp addresses key 32 hb + 8 rho + grp / 2 at
            dims + 16 (lane / 16 % 2) + 8 (grp % 2)."""
            grp = lane % 16
            vs = []
            for rho in range_constexpr(4):
                key = 32 * hb + 8 * rho + grp // 2
                byte = key * (ROWW * 4) + 128 * wave + 32 * d + 16 * ((lane // 16) % 2) + 8 * (grp % 2)
                vs.append(_tr8(kbp + byte // 4))
            return fx.Vector.from_elements([vs[r_][w_] for r_ in range(4) for w_ in range(2)], fx.Int32)

        def pops_of(pbp):
            """P V's B operands (P^T, columns 32 ct + n32, keys 32 hb ..) from the P buffer at ``pbp``."""
            out = []
            for ct in range_constexpr(3):
                pp = pbp + (32 * ct + n32) * PROW + 8 * hb
                out.append(_cat8(_lds16(pp), _lds16(pp + 4)))
            return out

        def soft1(sacc, i, lims, ct, masked=True):
            """Column tile ct's raw scores of block i (masked keys -inf; ``masked`` False: a block every
            row sees whole) and their scaled max over the wave's 32 keys (to ``red``)."""
            kbase = i * BK + 32 * kh + 4 * g
            room = lims[ct] - kbase
            xc = []
            for kt in range_constexpr(2):
                for e in range_constexpr(4):
                    if const_expr(masked):
                        xc.append((fx.Int32(16 * kt + e) < room).select(fx.Float32(sacc[3 * kt + ct][e]), fx.Float32(NINF)))
                    else:
                        xc.append(fx.Float32(sacc[3 * kt + ct][e]))
            mb = xc[0]
            for e in range_constexpr(1, 8):
                mb = fx.max(mb, xc[e])
            mb = xred(fx.Float32(mb), 16, fx.max)
            mb = xred(fx.Float32(mb), 32, fx.max)
            mb = fx.max(fx.Float32(mb) * qk_c, fx.Float32(NEG))
            fx.ptr_store(fx.Float32(mb), red + kh * COLS + 48 * ch + 16 * ct + c16)
            return xc, mb

        def soft2(xc, mb, m_old, l_old, ct, pbp, alp):
            """Column tile ct after barrier A: the block max (both key halves), the running max, P =
            2^(s c - m + PRE) (e4m3, to the P buffer at ``pbp``; a masked key's -inf gives 0) and
            alpha (to ``alp``, the same value from both halves). Returns the new max and row sum."""
            col = 48 * ch + 16 * ct + c16
            pm = fx.Float32(fx.ptr_load(red + (1 - kh) * COLS + col))
            mn = fx.max(m_old, fx.max(mb, pm))
            al = hw_exp2(m_old - mn)
            off = fx.Float32(PRE) - mn
            ps = []
            for e in range_constexpr(8):
                if const_expr("nosm" in probe):
                    ps.append(xc[e] + off)
                else:
                    ps.append(hw_exp2(fmath.fma(xc[e], fx.Float32(qk_c), off)))
            tot = ps[0]
            for e in range_constexpr(1, 8):
                tot = tot + ps[e]
            for kt in range_constexpr(2):
                fx.ptr_store(_e4m3x4(ps[4 * kt : 4 * kt + 4]), pbp + col * PROW + 8 * kh + 4 * kt + g)
            fx.ptr_store(fx.Float32(al), alp + col)
            return mn, l_old * al + tot

        for k_ in range_constexpr(-(-TBL_WORDS // THREADS)):
            w_ = tid + THREADS * k_
            if w_ < TBL_WORDS:
                fx.ptr_store(_ld(ptbl, w_), tbl + w_)
        _bar()
        nitems = fx.Int32(fx.ptr_load(tbl + 2 * MAX_REQS * TBL))
        t_plan = (_memtime() - tk0) if const_expr(tlon) else z
        r_kv = rsrc(kv)
        place = (bid % 8) * (BLOCKS // 8) + bid // 8
        full_rounds = nitems // BLOCKS
        last_place = (full_rounds % 2 == 0).select(place, BLOCKS - 1 - place)
        nrounds = full_rounds + (last_place < nitems % BLOCKS).select(fx.Int32(1), fx.Int32(0))
        for rnd in range(nrounds):
            ta = _memtime() if const_expr(tlon) else z
            k = rnd * BLOCKS + (rnd % 2 == 0).select(place, BLOCKS - 1 - place)
            # the item's class: the last table row whose first item is <= k (rows ascend; lane r: row r)
            ip = (lane < 2 * nreq).select(fx.Int32(fx.ptr_load(tbl + fx.min(lane, 2 * MAX_REQS - 1) * TBL + T_IPRE)), fx.Int32(1 << 30))
            rk = uniform(popcount(ballot(ip <= k)) - 1)
            row = tbl + rk * TBL
            rq = uniform(fx.ptr_load(row + T_R))
            kind = uniform(fx.ptr_load(row + T_KIND))
            n = uniform(fx.ptr_load(row + T_N))
            nb = uniform(fx.ptr_load(row + T_NB))
            wr = uniform(fx.ptr_load(row + T_WR))
            L = uniform(fx.ptr_load(row + T_L))
            P = uniform(fx.ptr_load(row + T_P))
            pb = uniform(fx.ptr_load(row + T_PB))
            q = k - uniform(fx.ptr_load(row + T_IPRE))
            sp = (kind == 0).select(q // NG, n - 1)  # the item's split
            hg = (kind == 0).select(q % NG, q)  # its head group
            b0 = sp * wr
            b1 = fx.min(b0 + wr, nb)
            nblk = b1 - b0
            kvp = uniform(fx.ptr_load(row + T_KVP))
            sc = {"b0": b0, "b1": b1, "L": L, "kvp": kvp, "kv_idx": kv_idx, "r_kv": r_kv, "ring": ring,
                  "lane": lane, "tid": tid}  # fmt: skip
            # keys visible to the wave's QK rows 3 ch + ct (local key index bound)
            lims = []
            for ct in range_constexpr(3):
                j = 3 * ch + ct
                lims.append(fx.max(fx.Int32(0), fx.min(L, (P + j - rank + 8 * DCP) // 8 - DCP + 1)) - b0 * BK)
            lmin = uniform(fx.min(lims[0], fx.min(lims[1], lims[2])))
            # every wave done with the ring and the item before's buffers
            _bar()
            noload = "noload" in probe
            rows0 = _rows(sc, fx.Int32(0))
            rows1 = _rows(sc, fx.Int32(1))
            # the wave's q: 3 column tiles (rows 3 ch + ct, heads 16 hg + c16) x 5 K steps
            qops = []
            for ct in range_constexpr(3):
                qrow = ((G * rq + 3 * ch + ct) * NHA + HG * hg + c16) * DQK
                for s in range_constexpr(KSTEPS):
                    w = (qrow + _kdim(s, g)) // 4
                    v = _cat8(_ld4(q8, w), _ld4(q8, w + 4))
                    if const_expr(s == KSTEPS - 1):
                        v = fx.Vector.from_elements([(g < 2).select(fx.Int32(v[e]), fx.Int32(0)) for e in range(8)], fx.Int32)
                    qops.append(v)
            # block 0 into slot 0, block 1 into registers
            stg0 = _load(sc, rows0)
            stg = _load(sc, rows1)
            if const_expr(not noload):
                _store(sc, stg0, fx.Int32(0))
            acc = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(12)]  # acc[3 d + ct]
            ms = [fx.Float32(NEG)] * 3
            ls = [fx.Float32(0.0)] * 3
            one = fx.Float32(1.0)
            _bar()
            if const_expr(tlon):
                tb = _memtime()
                t_set = t_set + (tb - ta)
                n_it = n_it + 1
                n_bl = n_bl + fx.Int64(nblk)
            for i in range(nblk):
                p0 = _memtime() if const_expr(phon) else z
                # block i + 2's rows, used before barrier B (no load result crosses the loop's back edge)
                ridx = _rows(sc, i + 2)
                kb = ring + (i % NSLOT) * SLOT_W
                # QK: sacc[3 kt + ct], element e = key 32 kh + 16 kt + 4 g + e, column 48 ch + 16 ct + c16
                sacc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(6)]
                if const_expr("noqk" not in probe):
                    aop = kop(kb, 0)
                    for u in range_constexpr(2 * KSTEPS):
                        cur = aop
                        if const_expr(u + 1 < 2 * KSTEPS):
                            aop = kop(kb, u + 1)
                        for ct in range_constexpr(3):
                            rocdl.sched_barrier(0)
                            sacc[3 * (u % 2) + ct] = _mfma16(cur, qops[KSTEPS * ct + u // 2], sacc[3 * (u % 2) + ct])
                            rocdl.sched_barrier(0)
                # the mask only where a row's visible keys end inside the block (wave-uniform)
                xs = [[fx.Float32(0.0)] * 8 for _ in range(3)]
                mx = [fx.Float32(0.0)] * 3
                if (i + 1) * BK <= lmin:
                    for ct in range_constexpr(3):
                        xc, mb = soft1(sacc, i, lims, ct, False)
                        xs[ct] = xc
                        mx[ct] = mb
                else:
                    for ct in range_constexpr(3):
                        xc, mb = soft1(sacc, i, lims, ct)
                        xs[ct] = xc
                        mx[ct] = mb
                p1 = _memtime() if const_expr(phon) else z
                _bar()
                p2 = _memtime() if const_expr(phon) else z
                # softmax 2, and block i + 1 into the other slot (its readers, block i - 1, are past
                # barrier A) with block i + 2's loads, three pieces after each column tile
                kvb = _row_of(sc, ridx)
                for ct in range_constexpr(3):
                    mn, l2 = soft2(xs[ct], mx[ct], ms[ct], ls[ct], ct, pbuf, alph)
                    ms[ct] = mn
                    ls[ct] = l2
                    if const_expr(not noload):
                        for k in range_constexpr(3 * ct, 3 * ct + 3):
                            _store1(sc, stg[k], (i + 1) % 2, k)
                            stg[k] = _load1(sc, kvb, k)
                p3 = _memtime() if const_expr(phon) else z
                _bar()
                p4 = _memtime() if const_expr(phon) else z
                # PV: this wave's dims 128 wave + 32 d + .., columns 32 ct + n32
                als = [fx.Float32(fx.ptr_load(alph + 32 * ct + n32)) for ct in range(3)]
                resc = ballot((als[0] != one) | (als[1] != one) | (als[2] != one)) != fx.Int64(0)
                if resc:
                    acc = [
                        fx.Vector.from_elements([fx.Float32(acc[t][e]) * als[t % 3] for e in range(16)], fx.Float32)
                        for t in range(12)
                    ]
                if const_expr("nopv" not in probe):
                    pops = pops_of(pbuf)
                    vt = vop(kb, 0)
                    for d in range_constexpr(4):
                        cur = vt
                        if const_expr(d + 1 < 4):
                            vt = vop(kb, d + 1)
                        for ct in range_constexpr(3):
                            rocdl.sched_barrier(0)
                            acc[3 * d + ct] = _mfma32(cur, pops[ct], acc[3 * d + ct])
                            rocdl.sched_barrier(0)
                if const_expr(phon):
                    p5 = _memtime()
                    ph = [ph[0] + (p1 - p0), ph[1] + (p2 - p1), ph[2] + (p3 - p2), ph[3] + (p4 - p3), ph[4] + (p5 - p4), ph[5]]
            tc = _memtime() if const_expr(tlon) else z
            # the split's output: both key halves' row sums, O / l
            for ct in range_constexpr(3):
                col = 48 * ch + 16 * ct + c16
                lt = xred(fx.Float32(ls[ct]), 16, lambda x, y: x + y)
                lt = xred(fx.Float32(lt), 32, lambda x, y: x + y)
                fx.ptr_store(fx.Float32(lt), lsum + kh * COLS + col)
                if kh == 0:
                    fx.ptr_store(fx.Float32(ms[ct]), mfin + col)
            _bar()
            direct = n == 1
            invs, lvals, heads, prows = [], [], [], []
            for ct in range_constexpr(3):
                col = 32 * ct + n32
                lt = fx.Float32(fx.ptr_load(lsum + col)) + fx.Float32(fx.ptr_load(lsum + COLS + col))
                mm = fx.Float32(fx.ptr_load(mfin + col))
                okl = lt > fx.Float32(0.0)
                invs.append(okl.select(fx.Float32(1.0) / lt, fx.Float32(0.0)))
                lvals.append(okl.select((mm - PRE + fx.Float32(rocdl.log(T.f32, fx.Float32(lt).ir_value()))) * LN2, fx.Float32(NEG)))
                j = 2 * ct + n32 // 16
                heads.append(HG * hg + n32 % 16)
                prows.append(direct.select(G * rq + j, pb + G * sp + j))
            if const_expr("nostore" in probe):
                for ct in range_constexpr(3):
                    ssum = lvals[ct]
                    for d in range_constexpr(4):
                        for e in range_constexpr(16):
                            ssum = ssum + fx.Float32(acc[3 * d + ct][e])
                    bo.buffer_store(ssum, rsrc(lse), prows[ct] * NHA + heads[ct])
            else:
                if direct:
                    for ct in range_constexpr(3):
                        for d in range_constexpr(4):
                            for qd in range_constexpr(4):
                                dim = 128 * wave + 32 * d + 8 * qd + 4 * hb
                                vals = [fx.Float32(acc[3 * d + ct][4 * qd + e]) * invs[ct] for e in range(4)]
                                ob = fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
                                bo.buffer_store(ob, rsrc(o), ((prows[ct] * NHA + heads[ct]) * KV + dim) // 2)
                        if (wave == 0) & (hb == 0):
                            bo.buffer_store(lvals[ct], rsrc(lse), prows[ct] * NHA + heads[ct])
                else:
                    for ct in range_constexpr(3):
                        for d in range_constexpr(4):
                            for qd in range_constexpr(4):
                                dim = 128 * wave + 32 * d + 8 * qd + 4 * hb
                                vals = [fx.Float32(acc[3 * d + ct][4 * qd + e]) * invs[ct] for e in range(4)]
                                bo.buffer_store(
                                    fx.Vector.from_elements(vals, fx.Float32), rsrc(logits), (prows[ct] * NHA + heads[ct]) * KV + dim,
                                    cache_modifier=CM_DEV,
                                )  # fmt: skip
                        if (wave == 0) & (hb == 0):
                            bo.buffer_store(lvals[ct], rsrc(plse), prows[ct] * NHA + heads[ct], cache_modifier=CM_DEV)
            if const_expr(tlon):
                td = _memtime()
                t_loop = t_loop + (tc - tb)
                t_epi = t_epi + (td - tc)
        # every CTA's split partials written: then the requests' splits merged here
        tm0 = _memtime() if const_expr(tlon) else z
        if const_expr("nobar" not in probe):
            _grid_barrier(sync, tid)
        if const_expr("nomerge" not in probe):
            _merge(o, lse, logits, plse, indptr, fmap, pmap, uniform(fx.ptr_load(tbl + T_NTILE)), bid, wave, lane)
        if const_expr(tlon):
            t_epi = t_epi + (_memtime() - tm0)
        if const_expr(tlon):
            if tid == 0:
                vals = [t_plan, t_set, t_loop, t_epi, _memtime() - tk0, _realtime() - rk0, n_it, n_bl]
                if const_expr(phon):
                    vals = ph + [n_it, n_bl]
                for f in range_constexpr(8):
                    bo.buffer_store(fx.Int32(vals[f]), rsrc(tl), bid * 8 + f)

    @flyc.jit
    def launch(
        q8: Int64, kv: Int64, kv_idx: Int64, kv_ptr: Int64, g_ptr: Int64, nreq: Int32, rank: Int32,
        o: Int64, lse: Int64, logits: Int64, plse: Int64, ptbl: Int64, tl: Int64,
        indptr: Int64, fmap: Int64, pmap: Int64, sync: Int64,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        mla_decode_b(q8, kv, kv_idx, kv_ptr, g_ptr, nreq, rank, o, lse, logits, plse, ptbl, tl, indptr, fmap, pmap, sync).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    _BUILDS[key] = launch
    return launch


def build_plan():
    """The plan kernel: one CTA of 5 waves, the step's split plan to ``ptbl`` and the CSR
    maps of the split reduce (one launch a step: every MLA layer's decode
    reads the same plan)."""
    if "plan" in _BUILDS:
        return _BUILDS["plan"]
    keyed = key_tuple({"plan": 1}, SOURCES)

    @fx.struct
    class Smem:
        tbl: fx.Array[fx.Int32, TBL_WORDS, 16]
        scr: fx.Array[fx.Int32, PLAN_SCR, 16]

    @flyc.kernel(name="k3_mono_mla_decode_b_plan", known_block_size=[64 * PLAN_WAVES, 1, 1])
    def mla_decode_b_plan(kv_ptr: Int64, g_ptr: Int64, nreq: Int32, indptr: Int64, fmap: Int64, pmap: Int64, ptbl: Int64):
        _ = keyed
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        tbl = lds.tbl.ptr
        plan(None, tbl, lds.scr.ptr, kv_ptr, g_ptr, nreq, indptr, fmap, pmap, fx.Int32(0), tid)
        for k_ in range_constexpr(-(-TBL_WORDS // (64 * PLAN_WAVES))):
            w_ = tid + 64 * PLAN_WAVES * k_
            if w_ < TBL_WORDS:
                bo.buffer_store(fx.Int32(fx.ptr_load(tbl + w_)), rsrc(ptbl), w_)

    @flyc.jit
    def launch(kv_ptr: Int64, g_ptr: Int64, nreq: Int32, indptr: Int64, fmap: Int64, pmap: Int64, ptbl: Int64,
               stream: fx.Stream = _STREAM):  # fmt: skip
        _ = keyed
        mla_decode_b_plan(kv_ptr, g_ptr, nreq, indptr, fmap, pmap, ptbl).launch(grid=(1,), block=(64 * PLAN_WAVES,), stream=stream)

    _BUILDS["plan"] = launch
    return launch


def mla_decode_plan_launch(*, kv_ptr, g_ptr, nreq, bufs):
    """The step's plan (``build_plan``): once a step, before its layers' decodes."""
    assert 1 <= nreq <= MAX_REQS and kv_ptr.dtype == g_ptr.dtype == torch.int32
    build_plan()(
        kv_ptr.data_ptr(), g_ptr.data_ptr(), nreq, bufs.indptr.data_ptr(), bufs.fmap.data_ptr(),
        bufs.pmap.data_ptr(), bufs.tbl.data_ptr(), stream=torch.cuda.current_stream(),
    )  # fmt: skip


def mla_decode_launch(key: MlaDecodeBuild, *, q8, kv, kv_idx, kv_ptr, g_ptr, rank, o, lse, bufs, tl=None):
    """Host launcher, as ``mla_decode.mla_decode_launch``: ``q8`` e4m3 [S, 96,
    576]; ``kv`` the rank's pool rows [., 576] e4m3; ``kv_idx`` / ``kv_ptr`` /
    ``g_ptr`` int32; ``o`` bf16 [S, 96, 512] and ``lse`` fp32 [S, 96] (the
    single-split requests' rows); ``bufs``: ``DecodeBuffers``."""
    s = q8.shape[0]
    nreq = s // G
    assert s % G == 0 and 1 <= nreq <= MAX_REQS, s
    assert q8.element_size() == 1 and q8.is_contiguous() and q8.numel() == s * NHA * DQK
    assert kv.element_size() == 1 and kv.stride(-2) == DQK
    for t in (kv_idx, kv_ptr, g_ptr):
        assert t.dtype == torch.int32 and t.is_contiguous()
    assert kv_ptr.numel() >= nreq + 1 and g_ptr.numel() >= nreq + 1
    assert o.dtype == torch.bfloat16 and o.is_contiguous() and o.numel() == s * NHA * KV
    assert lse.dtype == torch.float32 and lse.is_contiguous() and lse.numel() == s * NHA
    build(key)(
        q8.data_ptr(), kv.data_ptr(), kv_idx.data_ptr(), kv_ptr.data_ptr(), g_ptr.data_ptr(), nreq, rank,
        o.data_ptr(), lse.data_ptr(), bufs.logits.data_ptr(), bufs.plse.data_ptr(), bufs.tbl.data_ptr(),
        0 if tl is None else tl.data_ptr(), bufs.indptr.data_ptr(), bufs.fmap.data_ptr(), bufs.pmap.data_ptr(),
        bufs.sync.data_ptr(),
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
