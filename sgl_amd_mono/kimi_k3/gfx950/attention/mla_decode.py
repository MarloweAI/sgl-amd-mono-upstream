# SPDX-License-Identifier: Apache-2.0
"""``mla_decode``: one Kimi-K3 MLA layer's DCP8 verify decode at the tail's
widths (``mla_tail.MIN_ROWS`` .. ``MAX_ROWS`` rows, 6 a request), e4m3 Q K and
e4m3 P V, in place of aiter's cp round-robin decode (``mla_decode_fwd``). Its
outputs are the tail's inputs: the split partials, their LSEs, the CSR maps of
the split reduce, and the rows of single-split requests finished directly.

Every rank attends all 96 heads of each row against its own share of the
request's keys (local key k at global position 8 k + rank; row j of a request
of committed length P sees the keys at positions <= P + j).

One launch, ``BLOCKS`` CTAs of 4 waves (one a SIMD, a CTA a CU), persistent:

    plan      every CTA, from the requests' local and global lengths: a
              request's key splits (equal 64-key-block runs; the count from
              its blocks and the batch's work a CTA), its items, the items'
              order (longest split first) and the deal (round t of 256
              places, every other round reversed; places XCD-major, so one
              split's items share an XCD's L2); CTA 0 writes the CSR maps
    items     a request's 576 (row, head) columns are 18 wave tiles of one
              row x 32 heads; an item is 4 wave tiles of one split (full), or
              the last 2 wave tiles of two splits side by side (pair, waves
              0-1 and 2-3 on their own split), or of one split (lone)
    a step    two 64-key KV slots into LDS (a full item's next two blocks,
              or one block of each paired split) while the step before's
              two are computed; a ring of two steps

A wave's block: S^T = K Q^T as v_mfma_scale_f32_32x32x64 e4m3 (M = 32 keys,
N = the wave's 32 columns, K = 576: 18 MFMAs), the online softmax in the lane
that holds the column (32 scores a lane, the max over the lane pair l, l ^
32), P' = e4m3(2^(s c - m + PRE)) packed straight into the P V operand, then
O^T += V^T P^T (M = 32 value dims, N = 32 columns, K = 64 keys: 16 MFMAs),
V^T read transposed from the row-major block by ds_read_b64_tr_b8. The
running max ``m`` moves only when a block's max passes it by ``TAU`` (log2
units), so P' <= 2^(TAU + PRE) = 256 is in e4m3's range and the 512
accumulators of a lane pair are rescaled rarely. The row sum is the fp32 sum
of the unrounded P'; a split's LSE is natural-log fp32.

Numerics against B300's FlashInfer FP8 MLA decode (q and the KV as given,
e4m3 x e4m3 into fp32, P cast to e4m3 with P <= 1): the same products and
precisions, with P carrying a static 2^PRE (and up to 2^TAU of lag) so that
values B300 flushes below e4m3's normal range stay normal here.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from sgl_amd_mono.common.ops import ballot, hw_exp2, read_lane_i32, rsrc, uniform, xred
from sgl_amd_mono.common.plan import BLOCKS, kernel_sources, key_tuple
from sgl_amd_mono.kimi_k3.gfx950.attention import mla_tail as mt

SOURCES = kernel_sources(__name__)

NHA = mt.NHA  # 96
KV = mt.KV  # 512
DQK = KV + 64  # 576
G = 6  # verify rows a request
DCP = mt.DCP
THREADS = 256
WAVES = THREADS // 64
BK = 64  # keys a block
PIECES = DQK // 16 + 1  # 16-byte pieces a key row in LDS: its 36, then a pad
ROWW = PIECES * 4  # 148 words a key row
SLOT_W = BK * ROWW  # a block's words
DMA_N = BK * PIECES // 64  # 37 wave-wide 1 KB DMAs a block
WT = G * NHA // 32  # 18 wave tiles a request
MAX_REQS = mt.MAX_ROWS // G  # 30
MAX_SPLITS = mt.LANE_SPLITS  # 64: the tail's reduce takes at most 64 splits a row
MAX_PART = 256  # splits with partials, all requests together
TAU = 0.0
PRE = 8.0
NEG = -1e30
LN2 = 0.6931471805599453
LOG2E = 1.4426950408889634
# a request's split: a run of W_NUM / W_DEN of a CTA's share of the batch's work
W_NUM, W_DEN = 11, 10
TBL = 10  # plan table words a class
T_R, T_KIND, T_N, T_NB, T_WR, T_IPRE, T_L, T_P, T_PB = range(9)


def partial_rows() -> int:
    """Rows of the split partials (``logits`` [rows, 96, 512], ``plse`` [rows, 96])."""
    return MAX_PART * G


@dataclass(frozen=True)
class MlaDecodeBuild:
    scale: float  # the softmax scale
    # timing probes only (checks): parts of a block's work left out
    probe: str = ""


_BUILDS: dict = {}
_STREAM = fx.Stream(None)


def _ld(ptr, i):
    return fx.Int32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32))


def _ld4(ptr, i):
    return fx.Vector(bo.buffer_load(rsrc(ptr), i, vec_width=4, dtype=T.i32))


def _lds16(ptr):
    return fx.Vector(llvm.load(T.vec(4, T.i32), ptr.llvm_ptr, alignment=16))


def _tr8(ptr):
    """ds_read_b64_tr_b8 at a lane's 8-byte aligned LDS ``ptr``."""
    return fx.Vector(rocdl.ds_read_tr8_b64(fx.Vector.make_type(2, fx.Int32), ptr.llvm_ptr).result)


def _cat8(a, b):
    return fx.Vector.from_elements([a[e] for e in range(len(a))] + [b[e] for e in range(len(b))], fx.Int32)


def _mfma(a, b, c):
    one = fx.Int32(127)
    return fx.Vector(rocdl.mfma_scale_f32_32x32x64_f8f6f4(T.vec(16, T.f32), [a, b, c, 0, 0, 0, one, 0, one]))


def _e4m3x4(v):
    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, v[0], v[1], fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, v[2], v[3], lo, True))


def _dma1(rs, lds_ptr, voff):
    """One wave-wide 16-byte-a-lane LDS DMA (lane i's 16 bytes at ``lds_ptr``
    + 16 i), async: its LDS writes are waited for by ``wait_asyncmark`` only."""
    z = as_ir_value(fx.Int32(0))
    rocdl.raw_ptr_buffer_load_async_lds(rs, lds_ptr.llvm_ptr, as_ir_value(fx.Int32(16)), as_ir_value(fx.Int32(voff)), z, z)


def _memtime():
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memtime", [], [], []))


def _bar():
    llvm.inline_asm(
        res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)\n\ts_barrier",
        constraints="~{memory}", has_side_effects=True,
    )  # fmt: skip


def _wave_sum(v):
    for off in [32, 16, 8, 4, 2, 1]:
        v = v + v.shuffle_xor(fx.Int32(off), fx.Int32(64))
    return v


def _key(t, i, hb):
    """Key (in the block) of a score tile's element: tile t (32 keys), element i
    of lane half hb (the 32x32 MFMA result's row)."""
    return 32 * t + 8 * (i // 4) + 4 * hb + i % 4


def _vt_key(rho, e, hb):
    """Key of byte e of transposed read rho (0 .. 3) of lane half hb: the key
    the P operand holds at byte 8 rho + e (tile (8 rho + e) / 16's element)."""
    j = 8 * rho + e
    return _key(j // 16, j % 16, hb)


def _k_op(kb, n32, hb, u):
    """QK MFMA u's A operand (K step u / 2, key tile u % 2): key 32 (u % 2) +
    n32's 32 bytes at dims 64 (u / 2) + 32 hb."""
    kp = kb + (32 * (u % 2) + n32) * ROWW + 16 * (u // 2) + 8 * hb
    return _cat8(_lds16(kp), _lds16(kp + 4))


def _v_op(kb, lane, hb, d):
    """P V MFMA d's A operand (V^T, dims 32 d + lane % 32): four transposed
    reads; read rho's byte e is key _vt_key(rho, e, hb). ds_read_b64_tr_b8
    gives lane i of a 16-lane group byte (i % 8) of source lane 2 e + i / 8's
    8 bytes as its byte e: source lane grp addresses key _vt_key(rho, grp / 2,
    hb) at dims 32 d + 16 (lane / 16 % 2) + 8 (grp % 2)."""
    grp = lane % 16
    vs = []
    for rho in range_constexpr(4):
        ke = 16 * rho + 8 * (grp // 8) + (grp // 2) % 4 + 4 * hb
        byte = ke * (ROWW * 4) + 32 * d + 16 * ((lane // 16) % 2) + 8 * (grp % 2)
        vs.append(_tr8(kb + byte // 4))
    return fx.Vector.from_elements([vs[r_][w_] for r_ in range(4) for w_ in range(2)], fx.Int32)


@flyc.jit
def _rows(c, i):
    """Lane l's key of each slot's block at step i: its record's row (past the
    keys: row 0)."""
    lane, L = c["lane"], c["L"]
    out = []
    for sl in range_constexpr(2):
        blk = c["st"][sl] + c["stride"] * i
        kk = blk * BK + lane
        ok = (blk < c["en"][sl]) & (kk < L)
        idx = _ld(c["kv_idx"], c["kvp"] + fx.max(fx.Int32(0), fx.min(kk, L - 1)))
        if const_expr("l2" in c["probe"]):
            idx = idx % BK
        out.append(ok.select(idx, fx.Int32(0)))
    return out


@flyc.jit
def _dma(c, i, rows):
    """Step i's slots straight into ring buffer i % 2, a padded 592-byte row a
    key (``rows``: ``_rows`` of step i): wave w issues the DMAs w, w + 4 ..
    (DMA 36 on wave 0 only); a slot's row offsets gathered first, so one LDS
    wait covers them."""
    lane, wave = c["lane"], c["wave"]
    nd = -(-DMA_N // WAVES)
    for sl in range_constexpr(2):
        blk = c["st"][sl] + c["stride"] * i
        base = c["ring"] + ((i % 2) * 2 + sl) * SLOT_W
        if (i < c["nsteps"]) & (blk < c["en"][sl]):
            voffs = []
            for dj in range_constexpr(nd):
                pc = (wave + WAVES * dj) * 64 + lane
                kq = fx.min(pc // PIECES, BK - 1)
                part = pc % PIECES
                off = fx.Int32(rocdl.ds_bpermute(T.i32, kq * 4, rows[sl]))
                voffs.append((part < PIECES - 1).select(off * DQK + part * 16, fx.Int32(0)))
            for dj in range_constexpr(nd):
                d = wave + WAVES * dj
                if const_expr(WAVES * (dj + 1) <= DMA_N):
                    _dma1(c["r_kv"], (base + d * 256), voffs[dj])
                else:
                    if d < DMA_N:
                        _dma1(c["r_kv"], (base + d * 256), voffs[dj])


@flyc.jit
def plan(c, tbl, kv_ptr, g_ptr, nreq, rank, indptr, fmap, pmap, bid, tid):
    """The step's split plan in LDS ``tbl``, by wave 0. A request's splits are
    runs of ``w`` blocks (the batch's work a CTA, x 1.1), the last the rest (at
    most 64 splits: then runs of ceil(blocks / 64)). Its items fall in two
    classes: A, the full items of its whole runs and the pairs of the halves
    of runs (2 p, 2 p + 1), cost w; B, the last run's full items (and its lone
    half when the count is odd), cost the rest. The classes go to ``tbl`` a row
    each, longest first: the deal order. CTA 0 also writes the CSR maps.
    Returns the item count."""
    lane = tid % 64
    if tid < 64:
        # lane r: request r
        r = fx.min(lane, nreq - 1)
        live = lane < nreq
        L = live.select(_ld(kv_ptr, r + 1) - _ld(kv_ptr, r), fx.Int32(0))
        P = _ld(g_ptr, r + 1) - _ld(g_ptr, r) - G
        nb = (L + BK - 1) // BK
        w9 = _wave_sum(9 * nb)
        w = fx.max(fx.Int32(1), (w9 * W_NUM + 2 * BLOCKS * W_DEN - 1) // (2 * BLOCKS * W_DEN))
        n = fx.max(fx.Int32(1), (nb + w - 1) // w)
        wr = (n > MAX_SPLITS).select((nb + MAX_SPLITS - 1) // MAX_SPLITS, w)
        n = fx.max(fx.Int32(1), (nb + wr - 1) // wr)
        rho = nb - (n - 1) * wr
        multi = live & (n > 1)
        msplits = multi.select(n, fx.Int32(0))
        pbase = fx.Int32(0)
        tile = fx.Int32(0)
        for r2 in range(nreq):
            m2 = read_lane_i32(msplits, r2)
            pbase = (r2 < r).select(pbase + m2, pbase)
            tile = ((r2 < r) & (m2 > 0)).select(tile + 1, tile)
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
        cnt = (kind == 0).select(4 * (n_c - 1) + n_c // 2, 4 + n_c % 2)
        cnt = clive.select(cnt, fx.Int32(0))
        cost = (cnt > 0).select((kind == 0).select(wr_c, rho_c), fx.Int32(-1))
        rk = fx.Int32(0)
        ipre = fx.Int32(0)
        for c2 in range(2 * nreq):
            k2 = read_lane_i32(cost, c2)
            i2 = read_lane_i32(cnt, c2)
            before = (k2 > cost) | ((k2 == cost) & (c2 < lane))
            rk = before.select(rk + 1, rk)
            ipre = before.select(ipre + i2, ipre)
        total = _wave_sum(cnt)
        if clive:
            row = tbl + rk * TBL
            for f, v in (
                (T_R, cr), (T_KIND, kind), (T_N, n_c), (T_NB, nb_c), (T_WR, wr_c), (T_IPRE, ipre),
                (T_L, L_c), (T_P, P_c), (T_PB, G * pb_c),
            ):  # fmt: skip
                fx.ptr_store(fx.Int32(v), row + f)
        if lane == 0:
            fx.ptr_store(total, tbl + 2 * MAX_REQS * TBL)
    _bar()
    return fx.Int32(fx.ptr_load(tbl + 2 * MAX_REQS * TBL))


def build(key: MlaDecodeBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    keyed = key_tuple(key, SOURCES)
    qk_c = key.scale * LOG2E
    probe = key.probe

    @fx.struct
    class Smem:
        kv: fx.Array[fx.Int32, 4 * SLOT_W, 16]
        tbl: fx.Array[fx.Int32, 2 * MAX_REQS * TBL + 4, 16]

    @flyc.kernel(name="k3_mono_mla_decode", known_block_size=[THREADS, 1, 1])
    def mla_decode(
        q8: Int64, kv: Int64, kv_idx: Int64, kv_ptr: Int64, g_ptr: Int64, nreq: Int32, rank: Int32,
        o: Int64, lse: Int64, logits: Int64, plse: Int64, indptr: Int64, fmap: Int64, pmap: Int64,
        tl: Int64,
    ):  # fmt: skip
        _ = keyed
        tlon = "tl" in probe
        z = fx.Int64(0)
        tk0 = _memtime() if const_expr(tlon) else z
        t_set, t_loop, t_epi, t_bar, t_dma, n_it, n_st = z, z, z, z, z, z, z
        bid, tid = fx.block_idx.x, fx.thread_idx.x
        lane = tid % 64
        wave = uniform(tid // 64)
        n32 = lane % 32
        hb = lane // 32
        lds = fx.SharedAllocator().allocate(Smem).peek()
        ring, tbl = lds.kv.ptr, lds.tbl.ptr
        nitems = plan(None, tbl, kv_ptr, g_ptr, nreq, rank, indptr, fmap, pmap, bid, tid)
        t_plan = (_memtime() - tk0) if const_expr(tlon) else z
        r_kv = rsrc(kv)
        # this CTA's place: XCD-major (CTA c runs on XCD c % 8)
        place = (bid % 8) * (BLOCKS // 8) + bid // 8
        full_rounds = nitems // BLOCKS
        last_place = (full_rounds % 2 == 0).select(place, BLOCKS - 1 - place)
        nrounds = full_rounds + (last_place < nitems % BLOCKS).select(fx.Int32(1), fx.Int32(0))
        for rnd in range(nrounds):
            ta = _memtime() if const_expr(tlon) else z
            k = rnd * BLOCKS + (rnd % 2 == 0).select(place, BLOCKS - 1 - place)
            # the item's class: the last table row whose first item is <= k
            rk = fx.Int32(0)
            for r2 in range(2 * nreq):
                rk = (fx.Int32(fx.ptr_load(tbl + r2 * TBL + T_IPRE)) <= k).select(r2, rk)
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
            # item kind and its splits: s_a (waves 0-1, every wave of a full
            # item), s_b (waves 2-3 of a pair)
            nf = 4 * (n - 1)
            in_a = kind == 0
            full = in_a.select(q < nf, q < 4)
            pair = in_a & (q >= nf)
            lone = (~in_a) & (q == 4)
            s_a = in_a.select((q < nf).select(q // 4, 2 * (q - nf)), n - 1)
            s_b = in_a.select(2 * (q - nf) + 1, n - 1)
            g = q % 4
            _ = pair
            # this wave's split, wave tile and slots
            mine_b = (~full) & (wave >= 2)
            s_w = mine_b.select(s_b, s_a)
            wt = full.select(4 * g + wave, WT - 2 + wave % 2)
            active = ~(lone & (wave >= 2))
            slo = full.select(fx.Int32(0), wave // 2)
            shi = full.select(fx.Int32(2), active.select(wave // 2 + 1, wave // 2))
            j = wt // 3
            head = 32 * (wt % 3) + n32
            # the slots' blocks: slot 0 split s_a, slot 1 s_b (s_a of a full item)
            a0, a1 = s_a * wr, fx.min((s_a + 1) * wr, nb)
            b0, b1 = s_b * wr, fx.min((s_b + 1) * wr, nb)
            st0 = a0
            st1 = full.select(a0 + 1, b0)
            en0 = a1
            en1 = full.select(a1, lone.select(b0, b1))
            stride = full.select(fx.Int32(2), fx.Int32(1))
            nsteps = full.select((a1 - a0 + 1) // 2, fx.max(a1 - a0, lone.select(fx.Int32(0), b1 - b0)))
            kvp = _ld(kv_ptr, rq)
            # the keys this wave's row sees in its split
            vis = fx.max(fx.Int32(0), fx.min(L, (P + j - rank + 8 * DCP) // 8 - DCP + 1))
            sw_end = fx.min((s_w + 1) * wr, nb)
            lim = fx.min(vis, fx.min(L, sw_end * BK))

            sc = {
                "st": (st0, st1), "en": (en0, en1), "stride": stride, "nsteps": nsteps, "L": L,
                "kvp": kvp, "kv_idx": kv_idx, "r_kv": r_kv, "ring": ring, "wave": wave, "lane": lane,
                "probe": probe,
            }  # fmt: skip
            # every wave done with the ring before this item's first DMA
            _bar()
            _dma(sc, fx.Int32(0), _rows(sc, fx.Int32(0)))
            rocdl.asyncmark()
            nxt = _rows(sc, fx.Int32(1))
            # the wave's q: 9 K steps of 32 bytes a lane (dims 64 kc + 32 hb ..)
            qrow = ((G * rq + j) * NHA + head) * DQK
            qops = []
            for kc in range_constexpr(DQK // 64):
                w = (qrow + 64 * kc + 32 * hb) // 4
                qops.append(_cat8(_ld4(q8, w), _ld4(q8, w + 4)))
            acc = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(KV // 32)]
            m = fx.Float32(NEG)
            l = fx.Float32(0.0)
            rocdl.s_waitcnt(vmcnt=0)
            if const_expr(tlon):
                tb = _memtime()
                t_set = t_set + (tb - ta)
                n_it = n_it + 1
            # a step: its DMAs landed (every wave's), then the next step's DMAs
            # out (every wave a share of both slots) under this step's blocks
            nxt0, nxt1 = nxt[0], nxt[1]
            for i in range(nsteps):
                t0 = _memtime() if const_expr(tlon) else z
                rocdl.wait_asyncmark(0)
                _bar()
                if const_expr(tlon):
                    t1 = _memtime()
                    t_bar = t_bar + (t1 - t0)
                    n_st = n_st + 1
                if const_expr("nodma" not in probe):
                    _dma(sc, i + 1, [nxt0, nxt1])
                rocdl.asyncmark()
                r20 = _rows(sc, i + 2)
                nxt0, nxt1 = r20[0], r20[1]
                for sl in range(slo, shi):
                    blk = (sl == 0).select(st0, st1) + stride * i
                    en = (sl == 0).select(en0, en1)
                    if blk < en:
                        kb = ring + ((i % 2) * 2 + sl) * SLOT_W
                        # S^T: element i of tile t = key _key(t, i, hb), column n32
                        sacc = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(2)]
                        # the next MFMA's K operand read under each MFMA
                        kops = _k_op(kb, n32, hb, 0)
                        for u in range_constexpr(2 * (DQK // 64)):
                            kc, t = u // 2, u % 2
                            cur = kops
                            if const_expr(u + 1 < 2 * (DQK // 64)):
                                kops = _k_op(kb, n32, hb, u + 1)
                            rocdl.sched_barrier(0)
                            if const_expr("noqk" not in probe):
                                sacc[t] = _mfma(cur, qops[kc], sacc[t])
                            rocdl.sched_barrier(0)
                        lim_b = lim - blk * BK
                        xs = []
                        for t in range_constexpr(2):
                            for e in range_constexpr(16):
                                ok = fx.Int32(_key(t, e, 0)) + 4 * hb < lim_b
                                xs.append(ok.select(fx.Float32(sacc[t][e]) * qk_c, fx.Float32(NEG)))
                        mb = xs[0]
                        for e in range_constexpr(1, 32):
                            mb = fx.max(mb, xs[e])
                        mb = xred(fx.Float32(mb), 32, fx.max)
                        need = mb > m + TAU
                        started = m > fx.Float32(NEG * 0.5)
                        resc = ballot(need & started) != fx.Int64(0)
                        if const_expr("noresc" in probe):
                            resc = fx.Int32(0) != fx.Int32(0)
                        if resc:
                            a = need.select(hw_exp2(m - mb), fx.Float32(1.0))
                            acc = [fx.Vector.from_elements([fx.Float32(v[e]) * a for e in range(16)], fx.Float32) for v in acc]
                            l = l * a
                        m = need.select(mb, m)
                        ps = []
                        for e in range_constexpr(32):
                            ps.append((xs[e] > fx.Float32(NEG * 0.5)).select(hw_exp2(xs[e] - m + PRE), fx.Float32(0.0)))
                        tot = ps[0]
                        for e in range_constexpr(1, 32):
                            tot = tot + ps[e]
                        l = l + tot
                        pw = fx.Vector.from_elements([_e4m3x4(ps[4 * w_ : 4 * w_ + 4]) for w_ in range(8)], fx.Int32)
                        # O^T += V^T P^T: dim tile d, the lanes' dims 32 d + n32;
                        # read rho's bytes = keys _vt_key(rho, e, hb)
                        # the next dim tile's V^T read under each MFMA
                        vt = _v_op(kb, lane, hb, 0)
                        for d in range_constexpr(KV // 32):
                            cur = vt
                            if const_expr(d + 1 < KV // 32):
                                vt = _v_op(kb, lane, hb, d + 1)
                            rocdl.sched_barrier(0)
                            if const_expr("nopv" not in probe):
                                acc[d] = _mfma(cur, pw, acc[d])
                            rocdl.sched_barrier(0)
            tc = _memtime() if const_expr(tlon) else z
            # the split's output: the lane pair's sums, O / l
            lt = xred(fx.Float32(l), 32, lambda x, y: x + y)
            okl = lt > fx.Float32(0.0)
            inv = okl.select(fx.Float32(1.0) / lt, fx.Float32(0.0))
            lval = okl.select((m - PRE + fx.Float32(rocdl.log(T.f32, fx.Float32(lt).ir_value()))) * LN2, fx.Float32(NEG))
            direct = n == 1
            prow = direct.select(G * rq + j, pb + G * s_w + j)
            if const_expr("nostore" in probe):
                ssum = lval
                for d in range_constexpr(KV // 32):
                    for e in range_constexpr(16):
                        ssum = ssum + fx.Float32(acc[d][e])
                if active:
                    bo.buffer_store(ssum, rsrc(lse), (G * rq + j) * NHA + head)
            if active & const_expr("nostore" not in probe):
                for d in range_constexpr(KV // 32):
                    for qd in range_constexpr(4):
                        dim = 32 * d + 8 * qd + 4 * hb
                        vals = [fx.Float32(acc[d][4 * qd + e]) * inv for e in range(4)]
                        if direct:
                            ob = fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
                            bo.buffer_store(ob, rsrc(o), ((prow * NHA + head) * KV + dim) // 2)
                        else:
                            bo.buffer_store(fx.Vector.from_elements(vals, fx.Float32), rsrc(logits), (prow * NHA + head) * KV + dim)
                if hb == 0:
                    if direct:
                        bo.buffer_store(lval, rsrc(lse), prow * NHA + head)
                    else:
                        bo.buffer_store(lval, rsrc(plse), prow * NHA + head)
            if const_expr(tlon):
                td = _memtime()
                t_loop = t_loop + (tc - tb)
                t_epi = t_epi + (td - tc)
        if const_expr(tlon):
            if tid == 0:
                vals = [t_plan, t_set, t_loop, t_epi, t_bar, t_dma, n_it, n_st]
                for f in range_constexpr(8):
                    bo.buffer_store(fx.Int32(vals[f]), rsrc(tl), bid * 8 + f)

    @flyc.jit
    def launch(
        q8: Int64, kv: Int64, kv_idx: Int64, kv_ptr: Int64, g_ptr: Int64, nreq: Int32, rank: Int32,
        o: Int64, lse: Int64, logits: Int64, plse: Int64, indptr: Int64, fmap: Int64, pmap: Int64,
        tl: Int64, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        mla_decode(q8, kv, kv_idx, kv_ptr, g_ptr, nreq, rank, o, lse, logits, plse, indptr, fmap, pmap, tl).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    _BUILDS[key] = launch
    return launch


class DecodeBuffers:
    """The decode's outputs past ``o`` / ``lse``: the split partials and the
    CSR maps (one set, reused by every layer)."""

    def __init__(self, device):
        f32, i32 = torch.float32, torch.int32
        self.logits = torch.empty(partial_rows(), NHA, KV, dtype=f32, device=device)
        self.plse = torch.empty(partial_rows(), NHA, dtype=f32, device=device)
        self.indptr = torch.zeros(MAX_REQS + 1, dtype=i32, device=device)
        self.fmap = torch.zeros(2 * MAX_REQS, dtype=i32, device=device)
        self.pmap = torch.zeros(MAX_PART, dtype=i32, device=device)


def mla_decode_launch(key: MlaDecodeBuild, *, q8, kv, kv_idx, kv_ptr, g_ptr, rank, o, lse, bufs, tl=None):
    """Host launcher. ``q8`` e4m3 [S, 96, 576] (S = 6 a request); ``kv`` this
    rank's pool rows [., 576] e4m3; ``kv_idx`` int32 the requests' local key
    rows, ``kv_ptr`` int32 [R + 1] their offsets; ``g_ptr`` int32 [R + 1] the
    global lengths' offsets (committed length + 6 a request); ``o`` bf16 [S,
    96, 512] and ``lse`` fp32 [S, 96]: the single-split requests' rows;
    ``bufs``: ``DecodeBuffers`` (partials and the reduce's maps)."""
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
        o.data_ptr(), lse.data_ptr(), bufs.logits.data_ptr(), bufs.plse.data_ptr(), bufs.indptr.data_ptr(),
        bufs.fmap.data_ptr(), bufs.pmap.data_ptr(), 0 if tl is None else tl.data_ptr(),
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
