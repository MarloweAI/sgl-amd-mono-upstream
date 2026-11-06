# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 ``back``: one DeepSeek-V4.1 attention layer from the rotated query to the
rank's wo_b partial, adapted from vLLM's mono decode (``attention/back.py``).

One launch a layer and rank runs split (flash-decoding over a chunk of a token's
keys from K1's key table, every head of the rank), combine (the chunks merged with
the sink, inverse GPT-J RoPE -> bf16 XO), wo_a (-> MXFP8 X8B) and wo_b (-> the
caller's hook); the stages hand off device-scope data behind tagged flags.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from sgl_amd_mono.common.mx import clamp_fp8
from sgl_amd_mono.common.ops import (
    CM_DEV,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    fresh,
    global_load,
    hw_exp2,
    lane_gather,
    mfma_bf16,
    row_sum,
    rsrc,
    uniform,
    xshfl,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
)
from sgl_amd_mono.common.stamps import stamp
from sgl_amd_mono.common.sync import _spin
from sgl_amd_mono.deepseek_v41.gfx950.ids import bid as bid_
from sgl_amd_mono.deepseek_v41.gfx950.ids import tid as tid_
from sgl_amd_mono.deepseek_v41.gfx950.stages.gemv import (
    TILE,
    ld_f32,
    lds_row,
)
from sgl_amd_mono.deepseek_v41.gfx950.stages.seam import CELL_VALUES, cells_tile

from .device import Mailbox, fp8x8_bf16, ld_i32, mx_code, mx_mul, pow2
from .front import kt_entry, kt_entry_loads, kt_inputs, kt_lens
from .gemv import (
    RED,
    bf16_gemv_loads,
    bf16_gemv_mfmas,
    gemv_loads,
    gemv_mfmas,
    rows_loads,
    rows_store,
    words_loads,
    words_store,
)
from .plan import (
    COMB_PARTS,
    DATA,
    FLAG_STRIDE,
    HALF,
    HEAD_DIM,
    HEAD_TILE,
    KEYS,
    NOPE,
    ROPE,
    ROWS,
    WOA_ROWS,
    Dims,
    cdiv,
    split_keys,
    tile_rows,
    token_tiles,
)

_STREAM = fx.Stream(None)
NEG = -3.4028234663852886e38
LOG2E = 1.4426950408889634
REC = DATA + 8  # a raw record row in LDS: 576 data bytes + 8 scale bytes
# a q row in LDS, i32 words: 512 bf16 + 16 B of pad, so rows do not alias banks
Q_ROW = HEAD_DIM // 2 + 4
P_PAD = 2  # i32 words past a P row: 16 head rows read together hit distinct banks
# wo_a's K order: waves 0..3 / 4..7 take the two 16-row tiles, a K range each, summed in wave order
WOA_SPLIT_K = WAVES * ROWS // WOA_ROWS


class BackBuild(msgspec.Struct, frozen=True):
    tokens: int
    tp: int
    ratio: int  # 0: window only; 1 / 2: top-k over a compressed cache
    timeline: bool = False
    shuffled: bool = False  # wo_b in aiter's (16, 16) preshuffle (wo_a stays row-major)


def plus(t0, v):
    return v if isinstance(t0, int) and t0 == 0 else v + t0


def head_groups(d: Dims, s: int) -> int:
    """Wave groups a split unit's head tiles are spread over: two (a tile each) on
    64-key chunks, else one (a wave's keys loaded and dequantized once for every head
    tile)."""
    return 2 if d.head_tiles == 2 and split_keys(d, s) == 64 else 1


def chunk_keys(d: Dims, s: int) -> int:
    """A split unit's keys (``plan.split_keys``)."""
    return split_keys(d, s)


def key_tiles(d: Dims, s: int) -> int:
    """A split unit's 16-key tiles: one a wave of a head group; on one head tile's
    64-key chunks, waves 4 .. load no keys and score none."""
    return chunk_keys(d, s) // 16


def p_row(d: Dims, s: int) -> int:
    """A P row (one head's chunk of bf16 p) in LDS, i32 words."""
    return chunk_keys(d, s) // 2 + P_PAD


@flyc.jit
def publish(c, region, i):
    """This CTA's stores visible device-wide, then the flag pair (region, i)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] == 0:
        c["mb"].put(region, i, fx.Int32(1))


@flyc.jit
def store_partial(part, words, off):
    """A lane's 32 B of a head's chunk partial: ``words`` (8 i32) at i32 ``off``."""
    for hlf in range_constexpr(2):
        bo.buffer_store(
            fx.Vector.from_elements(words[4 * hlf : 4 * hlf + 4], fx.Int32),
            rsrc(part),
            off + 4 * hlf,
            cache_modifier=CM_DEV,
        )


@flyc.jit
def await_flags(c, region, n, index):
    """Thread u < n polls flag ``index(u)``; then the CTA proceeds."""
    tid = c["tid"]
    if tid < n:
        c["mb"].poll([(region, index(fx.min(tid, n - 1)), 1)])
    gpu.barrier()


# ---------------------------------------------------------------- keys


def key_rec(c, t, k):
    """(data address, scale address, valid) of token t's key k from K1's key table
    (``key_addr``)."""
    # a table this launch wrote (``krdy``: an index source's) is read past the CU's cache
    cm = CM_DEV if const_expr("krdy" in c) else 0
    val = ld_i32(c["args"]["kt"], t * KEYS + fx.min(fx.max(k, 0), KEYS - 1), cm)
    return key_addr(c["args"], c["ratio"], val)


def key_addr(a, ratio, val):
    """(data address, scale address, valid) of a key-table entry: slot | 1 << 31 for a
    compressed row, a window slot otherwise, -1 for none (it reads a zero record: its
    scores are masked, its values zeros)."""
    ok = val != fx.Int32(-1)
    slot = ok.select(val & fx.Int32(0x7FFFFFFF), fx.Int32(0))
    if const_expr(ratio > 0):
        is_c = ok & (val < 0)
        blk = is_c.select(a["comp_block"], a["swa_block"])
        stride = is_c.select(a["comp_stride"], a["swa_stride"])
        base = is_c.select(a["comp"], a["swa"])
    else:
        blk, stride, base = a["swa_block"], a["swa_stride"], a["swa"]
    b = slot // blk
    off = slot % blk
    page = base + fx.Int64(b) * stride
    z = a["zrec"]
    data = ok.select(page + fx.Int64(off * DATA), z)
    sc = ok.select(page + fx.Int64(blk * DATA + off * 8), z + fx.Int64(DATA))
    return data, sc, ok


def key_loads(data, sc, g):
    """A split lane's record loads (lane g's 16 B slices): its scale words, e4m3 dims
    64 c + 16 g .., bf16 rope dims 32 c + 8 g .."""
    scw = global_load(sc, words=2)
    nope = [
        global_load(data + fx.Int64(64 * c4 + 16 * g), words=4)
        for c4 in range(NOPE // 64)
    ]
    rope = [
        global_load(data + fx.Int64(NOPE + (cc * 32 + 8 * g) * 2), words=4)
        for cc in range(ROPE // 32)
    ]
    return scw, nope, rope


def coherent_load(addr, words):
    """``words`` (2 or 4) i32 at a 64-bit global address as device-coherent loads (8 B
    agent-scope monotonic atomics: sc1), past any line of it this CU cached before."""
    ptr_t = ir.Type.parse("!llvm.ptr<1>")
    pair = ir.VectorType.get([2], T.i32)
    out = []
    for i in range(words // 2):
        ptr = llvm.IntToPtrOp(ptr_t, as_ir_value(addr + fx.Int64(8 * i))).result
        v = llvm.LoadOp(
            T.i64,
            ptr,
            alignment=8,
            ordering=llvm.AtomicOrdering.monotonic,
            syncscope="agent",
        ).result
        w = fx.Vector(llvm.BitcastOp(pair, v).result)
        out += [w[0], w[1]]
    return fx.Vector.from_elements(out, fx.Int32)


def coherent_key_loads(data, sc, g):
    """``key_loads`` device-coherent, for a record this launch wrote."""
    scw = coherent_load(sc, 2)
    nope = [
        coherent_load(data + fx.Int64(64 * c4 + 16 * g), 4) for c4 in range(NOPE // 64)
    ]
    rope = [
        coherent_load(data + fx.Int64(NOPE + (cc * 32 + 8 * g) * 2), 4)
        for cc in range(ROPE // 32)
    ]
    return scw, nope, rope


def _zero_keys():
    z = fx.Vector.filled(4, 0, fx.Int32)
    return fx.Vector.filled(2, 0, fx.Int32), [z] * (NOPE // 64), [z] * (ROPE // 32)


def _flat(keys):
    scw, nope, rope = keys
    return (scw, *nope, *rope)


def _scale_byte(scw, blk64):
    return (scw[blk64 // 4] >> (blk64 % 4 * 8)) & 0xFF


def _fp8_quad_bf16(word, scale):
    """A dword's four E4M3 times a power-of-two ``scale`` -> four bf16 (exact)."""
    out = []
    for hi in (False, True):
        v = fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp8(
                T.bf16x2,
                as_ir_value(fx.Int32(word)),
                as_ir_value(fx.Float32(scale)),
                hi,
            )
        )
        out += [v[0], v[1]]
    return out


# ---------------------------------------------------------------- split


def k1_flag_load(krdy, words: dict, tid, t, s: int, pred=None):
    """A load of this thread's flag pair of ``await_k1``: CTA tid's q flag of token t's
    tile, past them t's key-table flag or, token parallel (``vrdy`` in ``words``), row
    tid - BLOCKS - 1's record flag. ``words``: each region's i32 offset from ``krdy``'s.
    ``pred`` false: the load touches no memory."""
    w = 2 * t
    if const_expr("vrdy" in words):
        u = tid - (BLOCKS + 1)
        rec = words["vrdy"] + 2 * fx.min(fx.max(u, 0), s - 1) * FLAG_STRIDE
        w = ((u >= 0) & (u < s)).select(rec, w)
    q = (t // TILE) * BLOCKS + fx.min(tid, BLOCKS - 1)
    w = (tid < BLOCKS).select(words["qb"] + 2 * q * FLAG_STRIDE, w)
    base = krdy
    nb = None if pred is None else pred.select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
    return lambda: fx.Vector(
        bo.buffer_load(
            rsrc(base, nb), w, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV
        )
    )


@flyc.jit
def await_k1(c, unit, t):
    """One launch per layer: wait for every CTA's q rows of token t's tile (``qb``; a CTA
    flags them past its records, key table rows and mixes), t's key table (``krdy``)
    and, token parallel, every row's record (``vrdy``: a row's keys hold the step's
    earlier rows, the other ranks' copied late), each published by K1's half of the
    launch; the first loads went out as K1's half ended (``c["k1_first"]``).
    ``c["k1_seen"](unit)`` runs once they are: a flag of every CTA was seen."""
    tag = c["mb"].tag
    load = k1_flag_load(c["krdy"], c["k1_words"], c["tid"], t, c["S"])
    _spin(load, lambda v: v[1] != tag, first=c["k1_first"])
    gpu.barrier()
    c["k1_seen"](unit)


def split_lane(d, s, nchunk, wave, lane, unit):
    """Split unit ``unit``: its token, its first key, this lane's key and whether this
    wave loads keys (``key_tiles``)."""
    nkt = key_tiles(d, s)
    k0 = (unit % nchunk) * chunk_keys(d, s)
    return (
        unit // nchunk,
        k0,
        k0 + (wave % nkt) * 16 + lane % 16,
        wave < (head_groups(d, s) * nkt),
    )


class EarlyKeys:
    """One launch per layer: a split CTA's keys from the launch's inputs, in K1's half
    (``key_table``'s entries, not its table): the inputs past the CTA's wqkv weights,
    the entry words past its wq_b weights, and at K1's end the old keys' record loads,
    in flight through the wait on q. A key of this step's rows (a record this launch
    writes) is read past the q flags (``stage_split``)."""

    def __init__(self, key, sw, a, unit, has):
        d = Dims(key.tp)
        self.s, self.a, self.has, self.sw, self.ratio = (
            key.tokens,
            a,
            has,
            sw,
            key.ratio,
        )
        self.nchunk = cdiv(KEYS, chunk_keys(d, self.s))
        tid = tid_()
        self.lane, self.wave = tid % 64, tid // 64
        self.t, _, self.key, self.keyed = split_lane(
            d, self.s, self.nchunk, self.wave, self.lane, unit
        )

    def inputs(self):
        # the token's counts and row ``lane``'s slot word (for the step's rows)
        a = self.a
        self.ins = kt_inputs(a, self.t)
        self.mine = ld_i32(a["slot"], a["slot_words"] * fx.min(self.lane, self.s - 1))

    def entries(self):
        self.nswa, self.ntopk = kt_lens(self.ins, self.sw)
        self.words = kt_entry_loads(self.a, self.t, self.key, self.ntopk, self.sw)

    def loads(self):
        """The entry, the token's key count, 1 for a key of this step's rows, and the
        record loads (a new key's: the zero record's)."""
        val = kt_entry(self.words, self.key, self.nswa, self.ntopk)
        hit = None
        for j in range_constexpr(self.s):
            sj = lane_gather(self.mine, j)
            h = (sj >= 0) & (sj == val)
            hit = h if hit is None else hit | h
        new = hit.select(fx.Int32(1), fx.Int32(0))
        data, sc, _ = key_addr(self.a, self.ratio, val)
        z, old = self.a["zrec"], new == 0
        data, sc = old.select(data, z), old.select(sc, z + fx.Int64(DATA))
        loads = early_key_loads(self.has & self.keyed, data, sc, self.lane // 16)
        return {
            "val": val,
            "kv_len": self.nswa + self.ntopk,
            "new": new,
            "loads": (loads[0], list(loads[1:8]), list(loads[8:])),
        }


@flyc.jit
def early_key_loads(on, data, sc, g):
    """``key_loads`` where ``on``, else zeros (no load)."""
    assert NOPE // 64 == 7 and ROPE // 32 == 2
    sw, n0, n1, n2, n3, n4, n5, n6, r0, r1 = _flat(_zero_keys())
    # a traced branch yields named values: the ten registers by name
    if on:
        sw, n0, n1, n2, n3, n4, n5, n6, r0, r1 = _flat(key_loads(data, sc, g))
    return sw, n0, n1, n2, n3, n4, n5, n6, r0, r1


@flyc.jit
def stage_split(c, unit):
    """Flash-decoding over chunk ``unit % nchunk`` of token ``unit // nchunk``,
    every head of the rank -> the chunk's partial and (max, sum) per head.

    Wave (head group hg, key tile kt) takes keys 16 kt .. of the chunk, loaded and
    dequantized once, against the group's head tiles; then wave w takes dims
    64 w .. of the PV over every key and head."""
    tid, lane, wave, d, s = c["tid"], c["lane"], c["wave"], c["d"], c["S"]
    a, sl = c["args"], c["sl"]
    nchunk = c["nchunk"]
    t, k0, key, _ = split_lane(d, s, nchunk, wave, lane, unit)
    ch = unit % nchunk
    ngrp = head_groups(d, s)
    nkt = key_tiles(d, s)
    nht = d.head_tiles // ngrp  # head tiles a wave
    # waves past the chunk's key tiles (one head tile's 64-key chunks) load and score no keys
    idle = ngrp * nkt < WAVES
    hgi = fx.min(wave // nkt, ngrp - 1) if idle else wave // nkt
    kt = wave % nkt

    def keyed(cond):
        # ``cond`` on a wave that loads keys
        return cond & (wave < ngrp * nkt) if idle else cond

    g = lane // 16
    r16 = lane % 16
    # one launch per layer: K1's half of this launch wrote q and the records; the old
    # keys' records are in flight since K1's end (``EarlyKeys``)
    one = "qb" in c
    if const_expr(one):
        pre = c["pre"]
        await_k1(c, unit, t)
        kv_len = pre["kv_len"]
        data, sc, okv = key_addr(a, c["ratio"], pre["val"])
        assert NOPE // 64 == 7 and ROPE // 32 == 2
        sw, n0, n1, n2, n3, n4, n5, n6, r0, r1 = _flat(pre["loads"])
        # a key of this step's rows, device-coherent: a line of it may sit in this CU's
        # cache from another key's load
        if keyed(pre["new"] != 0):
            sw, n0, n1, n2, n3, n4, n5, n6, r0, r1 = _flat(
                coherent_key_loads(data, sc, g)
            )
        keys = (sw, [n0, n1, n2, n3, n4, n5, n6], [r0, r1])
    # q does not depend on the key table: in flight before an index source's wait
    qrow = d.heads * HEAD_DIM // 8
    qpad = d.tile_heads * HEAD_DIM // 8  # with the head tile's padding rows
    qv = []
    for i in range_constexpr(cdiv(qrow, THREADS)):
        u = fresh(tid) + THREADS * i
        qv.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["q"]),
                    (t * d.q_rows) // 2 + fx.min(u, qrow - 1) * 4,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV if const_expr(one) else 0,
                )
            )
        )
    if const_expr(not one):
        if const_expr("krdy" in c):
            # an index source's table: this launch builds it after the indexer ran
            await_flags(c, c["krdy"], 1, lambda u: t)
        # kv_len and this lane's key load together: the table holds -1 past a token's keys
        kv_len = ld_i32(a["klen"], t, CM_DEV if const_expr("krdy" in c) else 0)
        data, sc, okv = key_rec(c, t, key)
    if k0 < kv_len:
        if const_expr(one):
            scw, nope, rope = keys
        elif const_expr(idle):
            got = early_key_loads(wave < ngrp * nkt, data, sc, g)
            scw, nope, rope = got[0], list(got[1:8]), list(got[8:])
        else:
            scw, nope, rope = key_loads(data, sc, g)
        for i in range_constexpr(cdiv(qrow, THREADS)):
            u = fresh(tid) + THREADS * i
            if u < qrow:
                fx.ptr_store(
                    qv[i],
                    sl["q"]
                    + (u // (HEAD_DIM // 8)) * Q_ROW
                    + (u % (HEAD_DIM // 8)) * 4,
                )
        if const_expr(qpad > qrow):
            # the tile's padding heads score zeros; their columns are never stored
            zero = fx.Vector.filled(4, 0.0, fx.Float32).bitcast(fx.Int32)
            for i in range_constexpr(cdiv(qpad - qrow, THREADS)):
                u = fresh(tid) + THREADS * i + qrow
                if u < qpad:
                    fx.ptr_store(
                        zero,
                        sl["q"]
                        + (u // (HEAD_DIM // 8)) * Q_ROW
                        + (u % (HEAD_DIM // 8)) * 4,
                    )
        # the raw rows to LDS for the PV: every head group loaded the same keys, group 0 stores
        if keyed(hgi == 0):
            rawb = sl["raw"] + (kt * 16 + r16) * (REC // 4)
            for c4 in range_constexpr(NOPE // 64):
                fx.ptr_store(nope[c4], rawb + (64 * c4 + 16 * g) // 4)
            for cc in range_constexpr(ROPE // 32):
                fx.ptr_store(rope[cc], rawb + (NOPE + (cc * 32 + 8 * g) * 2) // 4)
            if g == 0:
                fx.ptr_store(scw, rawb + DATA // 4)
        gpu.barrier()
        stamp(c, 5)
        # QK: MFMA K chunk 2 c + h is nope dims 64 c + 16 g + 8 h .., q read in the same order
        scale = a["qk_scale"].bitcast(fx.Float32)
        st = sl["st"]
        accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nht)]
        for c4 in range_constexpr(NOPE // 64):
            ksc = pow2(_scale_byte(scw, c4))
            for h2 in range_constexpr(2):
                kw = fp8x8_bf16(nope[c4][2 * h2], nope[c4][2 * h2 + 1], ksc).bitcast(
                    fx.BFloat16
                )
                for ht in range_constexpr(nht):
                    qb = sl["q"] + ((hgi * nht + ht) * HEAD_TILE + r16) * Q_ROW
                    qw = fx.ptr_load(
                        qb + 32 * c4 + 8 * g + 4 * h2,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                    accs[ht] = mfma_bf16(kw, qw.bitcast(fx.BFloat16), accs[ht])
        for cc in range_constexpr(ROPE // 32):
            kw = rope[cc].bitcast(fx.BFloat16)
            for ht in range_constexpr(nht):
                qb = sl["q"] + ((hgi * nht + ht) * HEAD_TILE + r16) * Q_ROW
                qw = fx.ptr_load(
                    qb + (NOPE + cc * 32 + 8 * g) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
                accs[ht] = mfma_bf16(kw, qw.bitcast(fx.BFloat16), accs[ht])
        # scores of keys 4 g + i (C rows), head r16 of a tile (C column)
        ok = (key < kv_len) & okv
        okw = ok.select(fx.Int32(1), fx.Int32(0))
        valid = [lane_gather(okw, 4 * g + i) != 0 for i in range(4)]
        svs = []
        for ht in range_constexpr(nht):
            sv = [
                valid[i].select(accs[ht][i] * scale, fx.Float32(NEG)) for i in range(4)
            ]
            mw = butterfly(
                fx.max(fx.max(sv[0], sv[1]), fx.max(sv[2], sv[3])), (32, 16), fx.max
            )
            if keyed(g == 0):
                fx.ptr_store(mw, st + ((hgi * nht + ht) * nkt + kt) * HEAD_TILE + r16)
            svs.append(sv)
        gpu.barrier()
        lst = st + d.tile_heads * nkt
        for ht in range_constexpr(nht):
            hti = hgi * nht + ht
            m = fx.ptr_load(st + (hti * nkt) * HEAD_TILE + r16)
            for j in range_constexpr(1, nkt):
                m = fx.max(m, fx.ptr_load(st + (hti * nkt + j) * HEAD_TILE + r16))
            sv = svs[ht]
            p = [valid[i].select(hw_exp2(sv[i] - m), fx.Float32(0.0)) for i in range(4)]
            lw = (p[0] + p[2]) + (p[1] + p[3])
            lw = lw + xshfl(lw, 32)
            lw = lw + xshfl(lw, 16)
            pw = (
                fx.Vector.from_elements(p, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
            )
            pbase = (
                sl["p"] + (hti * HEAD_TILE + r16) * p_row(d, s) + (kt * 16 + 4 * g) // 2
            )
            pv2 = fx.Vector.from_elements([pw[0], pw[1]], fx.Int32)
            if const_expr(idle):
                if wave < ngrp * nkt:
                    fx.ptr_store(pv2, pbase)
            else:
                fx.ptr_store(pv2, pbase)
            if keyed(g == 0):
                fx.ptr_store(lw, lst + (hti * nkt + kt) * HEAD_TILE + r16)
        gpu.barrier()
        if tid < d.heads:
            hts = tid // HEAD_TILE
            hr = tid % HEAD_TILE
            mm = fx.ptr_load(st + (hts * nkt) * HEAD_TILE + hr)
            ll = fx.ptr_load(lst + (hts * nkt) * HEAD_TILE + hr)
            for j in range_constexpr(1, nkt):
                mm = fx.max(mm, fx.ptr_load(st + (hts * nkt + j) * HEAD_TILE + hr))
                ll = ll + fx.ptr_load(lst + (hts * nkt + j) * HEAD_TILE + hr)
            bo.buffer_store(
                fx.Vector.from_elements([mm, ll], fx.Float32),
                rsrc(c["pstat"]),
                ((t * d.heads + tid) * nchunk + ch) * 2,
                cache_modifier=CM_DEV,
            )
        # PV: wave w dims 64 w + 4 r16 + dt (an LDS word holds a key's four), lane g keys 4 g + j.
        # Every tile's LDS reads go out before the first MFMA and the rope wave branches once:
        # a read, convert, MFMA chain a tile is LDS-latency bound.
        is_rope = wave == WAVES - 1
        dbase = 64 * wave
        sh = (wave % 4) * 8
        pbs = [
            [
                fx.ptr_load(
                    sl["p"]
                    + (ht2 * HEAD_TILE + r16) * p_row(d, s)
                    + (kt2 * 16 + 4 * g) // 2,
                    result_type=fx.Vector.make_type(2, fx.Int32),
                ).bitcast(fx.Int16)
                for kt2 in range(nkt)
            ]
            for ht2 in range(d.head_tiles)
        ]
        rows = [
            [sl["raw"] + (kt2 * 16 + 4 * g + j) * (REC // 4) for j in range(4)]
            for kt2 in range(nkt)
        ]
        bz = fx.Vector.filled(4, 0.0, fx.Float32).to(fx.BFloat16)
        avs = [[bz for _ in range(4)] for _ in range(nkt)]  # [kt2][dt]: keys 4 g + j
        if is_rope:
            rws = [
                [
                    fx.ptr_load(
                        rb + (NOPE + 8 * r16) // 4,
                        result_type=fx.Vector.make_type(2, fx.Int32),
                    ).bitcast(fx.BFloat16)
                    for rb in rows[kt2]
                ]
                for kt2 in range(nkt)
            ]
            avs = [
                [
                    fx.Vector.from_elements(
                        [rws[kt2][j][dt] for j in range(4)], fx.BFloat16
                    )
                    for dt in range(4)
                ]
                for kt2 in range(nkt)
            ]
        else:
            ws = [
                [fx.ptr_load(rb + (dbase + 4 * r16) // 4) for rb in rows[kt2]]
                for kt2 in range(nkt)
            ]
            scbs = [
                [fx.ptr_load(rb + DATA // 4 + wave // 4) for rb in rows[kt2]]
                for kt2 in range(nkt)
            ]
            avs = []
            for kt2 in range_constexpr(nkt):
                pk = []
                for j in range_constexpr(4):
                    scl = pow2((scbs[kt2][j] >> sh) & 0xFF)
                    pk.append(_fp8_quad_bf16(ws[kt2][j], scl))
                avs.append(
                    [
                        fx.Vector.from_elements(
                            [pk[j][dt] for j in range(4)], fx.BFloat16
                        )
                        for dt in range(4)
                    ]
                )
        accs = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
            for _ in range(d.head_tiles)
        ]
        for kt2 in range_constexpr(nkt):
            for ht2 in range_constexpr(d.head_tiles):
                for dt in range_constexpr(4):
                    accs[ht2][dt] = fx.Vector(
                        rocdl.mfma_f32_16x16x16bf16_1k(
                            T.vec(4, T.f32),
                            [
                                avs[kt2][dt].bitcast(fx.Int16),
                                pbs[ht2][kt2],
                                accs[ht2][dt],
                            ],
                        )
                    )
        stamp(c, 6)
        # partial[t][ch][head][dim] bf16: lane (r16, g) owns 32 contiguous B, dims 64 w + 16 g ..
        for ht2 in range_constexpr(d.head_tiles):
            hh = ht2 * HEAD_TILE + r16
            words = []
            for i in range_constexpr(4):
                v4 = fx.Vector.from_elements(
                    [accs[ht2][dt][i] for dt in range(4)], fx.Float32
                )
                w2 = v4.to(fx.BFloat16).bitcast(fx.Int32)
                words += [w2[0], w2[1]]
            off = (((t * nchunk + ch) * d.heads + hh) * HEAD_DIM + dbase + 16 * g) // 2
            if const_expr(d.tile_heads > d.heads):
                if r16 < d.heads:
                    store_partial(c["part"], words, off)
            else:
                store_partial(c["part"], words, off)
        publish(c, c["srdy"], (t * nchunk + ch) * FLAG_STRIDE)
        stamp(c, 7)


# ---------------------------------------------------------------- combine


def scalar_words(addr, n):
    """``n`` words at a wave-uniform address of uncached memory, scalar glc loads in one
    batch (one wait): they count in lgkmcnt, behind none of the CU's vector loads."""
    sizes, left = [], n
    for k in (16, 8, 4, 2, 1):
        while left >= k:
            sizes.append(k)
            left -= k
    asm, off = [], 0
    for i, k in enumerate(sizes):
        op = "s_load_dword" if k == 1 else f"s_load_dwordx{k}"
        asm.append(f"{op} ${i}, ${len(sizes)}, {off:#x} glc")
        off += 4 * k
    asm.append("s_waitcnt lgkmcnt(0)")
    tys = [T.i32 if k == 1 else T.vec(k, T.i32) for k in sizes]
    st_t = llvm.StructType.get_literal(tys)
    res = llvm.inline_asm(
        st_t,
        [as_ir_value(fx.Int64(addr))],
        "s_nop 4\n" + "\n".join(asm),
        ",".join(["=&s"] * len(sizes)) + ",s",
        has_side_effects=True,
    )
    out = []
    for i, (k, ty) in enumerate(zip(sizes, tys)):
        v = llvm.extractvalue(ty, res, [i])
        if k == 1:
            out.append(fx.Int32(v))
        else:
            vv = fx.Vector(v)
            out += [vv[e] for e in range(k)]
    return out


COMB_HEADS = WAVES  # heads a combine unit: a wave each
COMB_DIMS = HEAD_DIM // (64 * COMB_PARTS)  # dims a lane


def comb_flag(t, hg, part, hg_n):
    """The combine flag (a pair index) of token t's head group hg, dims part ``part``."""
    return ((t * hg_n + hg) * COMB_PARTS + part) * FLAG_STRIDE


@flyc.jit
def stage_combine(c, unit):
    """(token, 8 heads, part of the dims) ``unit``, a wave a head, ``COMB_DIMS`` dims a
    lane: the chunks merged with the sink folded in, bf16, inverse RoPE, bf16 -> XO (the
    normal path's wo_a reads it unquantized), published (a flag a unit)."""
    lane, wave, d, s = c["lane"], c["wave"], c["d"], c["S"]
    a = c["args"]
    hg_n = d.heads // COMB_HEADS
    t = unit // (hg_n * COMB_PARTS)
    hg = unit // COMB_PARTS % hg_n
    part = unit % COMB_PARTS
    h = hg * COMB_HEADS + wave
    nchunk = c["nchunk"]
    ck = chunk_keys(d, s)
    npair = COMB_DIMS // 2
    pos = ld_i32(a["pos"], 2 * t)
    sink = ld_f32(a["sink"], h) * fx.Float32(LOG2E)
    d0 = part * (HEAD_DIM // COMB_PARTS) + lane * COMB_DIMS
    rope_lane = d0 >= NOPE
    jj = fx.max((d0 - NOPE) // 2, 0)  # npair pairs from here
    cs = [
        rope_lane.select(ld_f32(a["cos_sin"], pos * ROPE + jj + i), fx.Float32(1.0))
        for i in range(npair)
    ]
    sn = [
        rope_lane.select(
            ld_f32(a["cos_sin"], pos * ROPE + HALF + jj + i), fx.Float32(0.0)
        )
        for i in range(npair)
    ]
    # the position, sink and RoPE above do not depend on the key table
    if const_expr("krdy" in c):
        # an index source's key count: written by this launch, past the CU's cache
        await_flags(c, c["krdy"], 1, lambda u: t)
    kv_len = ld_i32(a["klen"], t, CM_DEV if const_expr("krdy" in c) else 0)
    nlive = (kv_len + ck - 1) // ck
    await_flags(c, c["srdy"], nlive, lambda u: (t * nchunk + u) * FLAG_STRIDE)
    m = fx.Float32(NEG)
    ms, ls, vs = [], [], []
    pvs = [
        fx.Vector(
            bo.buffer_load(
                rsrc(c["part"]),
                (((t * nchunk + cc) * d.heads + h) * HEAD_DIM + d0) // 2,
                vec_width=npair,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        for cc in range(nchunk)
    ]
    # the head's (max, sum) of every chunk, contiguous: scalar loads past the partials'
    sto = uniform((t * d.heads + h) * nchunk * 8)
    stw = scalar_words(c["pstat"] + fx.Int64(sto), 2 * nchunk)
    for cc in range_constexpr(nchunk):
        live = cc < nlive
        stv = [stw[2 * cc].bitcast(fx.Float32), stw[2 * cc + 1].bitcast(fx.Float32)]
        pv = pvs[cc]
        mc = live.select(stv[0], fx.Float32(NEG))
        ms.append(mc)
        ls.append(live.select(stv[1], fx.Float32(0.0)))
        vals = []
        for e in range_constexpr(npair):
            vals += [bf_lo(pv[e]), bf_hi(pv[e])]
        vs.append([live.select(v, fx.Float32(0.0)) for v in vals])
        m = fx.max(m, mc)
    m_final = fx.max(m, sink)
    lsum = hw_exp2(sink - m_final)
    tot = [fx.Float32(0.0)] * COMB_DIMS
    for cc in range_constexpr(nchunk):
        wgt = hw_exp2(ms[cc] - m_final)
        lsum = lsum + ls[cc] * wgt
        tot = [tot[e] + vs[cc][e] * wgt for e in range(COMB_DIMS)]
    inv = fx.Float32(1.0) / lsum
    o = [bf16_round(tot[e] * inv) for e in range(COMB_DIMS)]
    # inverse GPT-J RoPE of each pair: e' = e cos + o sin, o' = o cos - e sin
    y = []
    for i in range_constexpr(npair):
        e, od = o[2 * i], o[2 * i + 1]
        y += [e * cs[i] + od * sn[i], od * cs[i] - e * sn[i]]
    words = [
        fx.Vector.from_elements([y[2 * i], y[2 * i + 1]], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
        for i in range(npair)
    ]
    bo.buffer_store(
        fx.Vector.from_elements(words, fx.Int32),
        rsrc(c["xo"]),
        t * (d.q_rows // 2) + (h * HEAD_DIM + d0) // 2,
        cache_modifier=CM_DEV,
    )
    publish(c, c["crdy"], comb_flag(t, hg, part, hg_n))


# ---------------------------------------------------------------- wo_a / wo_b


@flyc.jit
def stage_woa(c, task, acc, t0, n):
    """Rows 32 task .. of wo_a from the waves' accumulators (``bf16_gemv_mfmas``)
    -> bf16 -> vLLM's MXFP8 of the 32 -> X8B, wo_b's input."""
    tid, red, d, lane = c["tid"], c["red"], c["d"], c["lane"]
    split = WOA_SPLIT_K
    fx.ptr_store(acc, red + (c["wave"] * 64 + lane) * 4)
    gpu.barrier()
    mine = tid < WOA_ROWS * n
    tl = fx.min(tid // WOA_ROWS, n - 1)
    t = plus(t0, tl)
    row = tid % WOA_ROWS
    tile = row // ROWS
    y = bf16_round(
        row_sum(red, row % ROWS, tl, [tile * split + w for w in range(split)])
    )
    amax = butterfly(abs(y), (1, 2, 4, 8, 16), fx.max)
    code = mx_code(amax)
    q = clamp_fp8(y * mx_mul(code))
    w = fp8_pack4(
        q,
        *[
            lane_gather(q.bitcast(fx.Int32), fx.min(lane + k, 63)).bitcast(fx.Float32)
            for k in (1, 2, 3)
        ],
    )
    if mine & (row % 4 == 0):
        bo.buffer_store(
            w,
            rsrc(c["x8b"]),
            t * (d.o_rows // 4) + (task * WOA_ROWS + row) // 4,
            cache_modifier=CM_DEV,
        )
    if mine & (row == 0):
        bo.buffer_store(
            code, rsrc(c["x8bs"]), t * (d.o_rows // 32) + task, cache_modifier=CM_DEV
        )
    gpu.barrier()


def wob_loads(c, task, pred=None):
    a = c["args"]
    return gemv_loads(
        c,
        a["wob"],
        a["wob_s"],
        c["d"].o_rows,
        task * ROWS,
        pred=pred,
        shuffled=c["shuffled"],
    )


def _wob_store(a, t, col, v0, v1):
    word = (
        fx.Vector.from_elements([v0, v1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    bo.buffer_store(word, rsrc(a["out"]), (t * a["out_stride"] + col) // 2)


@flyc.jit
def stage_wob(c, task, ops, task2, ops2, has2, t0, n, mark=False):
    """Rows 16 task .. (and 16 task2 .., ``has2``) of wo_b -> bf16 -> out, or
    (``c["wob_out"]``, the mono layer) the caller's hook with each bf16 pair."""
    tid, red = c["tid"], c["red"]
    o_rows = c["d"].o_rows
    a = c["args"]
    gemv_mfmas(c, o_rows, c["xl"], c["xsl"], red, ops, rows=n)
    gemv_mfmas(c, o_rows, c["xl"], c["xsl"], red + RED, ops2, rows=n)
    gpu.barrier()
    if const_expr(mark):
        stamp(c, 15)
    cell = c.get("wob_cell")
    if const_expr(cell is not None):
        # the mono layer's cells (``seam.attn_cell``): 4 values a thread, row-major
        per = cells_tile(n)
        k = tid // per
        if (tid < 2 * per) & ((k == 0) | has2):
            u = tid % per
            rk = red + k * RED
            vs = []
            for i in range_constexpr(CELL_VALUES):
                v = CELL_VALUES * u + i
                vs.append(row_sum(rk, v % ROWS, v // ROWS))
            cell((k == 0).select(task, task2), t0, u, vs)
    else:
        per = ROWS * n // 2  # bf16 pairs a task
        k = tid // per
        emit = c.get("wob_out") or (
            lambda t, col, v0, v1: _wob_store(a, t, col, v0, v1)
        )
        if (tid < 2 * per) & ((k == 0) | has2):
            u = tid % per
            rp = u % (ROWS // 2)
            tl = u // (ROWS // 2)
            t = plus(t0, tl)
            rk = red + k * RED
            col = (k == 0).select(task, task2) * ROWS + 2 * rp
            v0 = row_sum(rk, 2 * rp, tl)
            v1 = row_sum(rk, 2 * rp + 1, tl)
            emit(t, col, v0, v1)
    gpu.barrier()


# ---------------------------------------------------------------- kernel


def woa_units(s: int, d: Dims) -> int:
    """wo_a's units: past one token tile, a (task, tile) each where they fit the
    CTAs, else a task each.

    ROCm/ATOM's tile tasks: a task's tiles in series on one CTA would leave most
    CTAs idle through wo_a, and a tile's reread of its task's weights hits L2."""
    nt = len(token_tiles(s))
    return d.woa_tasks * nt if nt > 1 and d.woa_tasks * nt <= BLOCKS else d.woa_tasks


def back_plan(key):
    """(nchunk, n_split, n_comb, COMB0, WOA0, WOB0, early_wob) of a build."""
    s, d = key.tokens, Dims(key.tp)
    ck = chunk_keys(d, s)
    nchunk = cdiv(KEYS, ck)
    n_split = s * nchunk
    n_comb = s * (d.heads // COMB_HEADS) * COMB_PARTS
    COMB0 = n_split % BLOCKS
    assert d.woa_tasks <= BLOCKS
    assert BLOCKS < d.wob_tasks <= 2 * BLOCKS
    # GEMV units past the attention units, so CTAs without attention load weights first
    WOA0 = (n_split + n_comb) % BLOCKS
    WOB0 = (WOA0 + woa_units(s, d)) % BLOCKS
    # loads complete in issue order, so wo_b's weights go early only if wo_a CTAs do no attention
    early_wob = n_split + n_comb + woa_units(s, d) <= BLOCKS
    return nchunk, n_split, n_comb, COMB0, WOA0, WOB0, early_wob


def back_lds_members(s: int, d: Dims) -> dict:
    """K2's attention LDS structs: the split stage's and the GEMVs' (a union's
    members: one at a time)."""
    rows = tile_rows(s)
    ck = chunk_keys(d, s)

    @fx.struct
    class SplitLds:
        q: fx.Array[fx.Int32, d.tile_heads * Q_ROW, 16]
        raw: fx.Array[fx.Int32, ck * REC // 4, 16]
        p: fx.Array[fx.Int32, d.tile_heads * p_row(d, s), 16]
        st: fx.Array[fx.Float32, 2 * d.tile_heads * key_tiles(d, s), 16]

    @fx.struct
    class GemvLds:
        red: fx.Array[fx.Float32, 2 * RED, 16]
        xl: fx.Array[fx.Int32, rows * lds_row(d.group_k // 4), 16]
        xsl: fx.Array[fx.Int32, rows * lds_row(d.group_k // 32), 16]

    return {"split": SplitLds, "gemv": GemvLds}


def back_lds(s: int, d: Dims):
    """K2's attention LDS: the union of ``back_lds_members``."""
    m = back_lds_members(s, d)
    SplitLds, GemvLds = m["split"], m["gemv"]

    @fx.union
    class BackLds:
        split: SplitLds  # type: ignore[valid-type]
        gemv: GemvLds  # type: ignore[valid-type]

    return BackLds


def back_context(key, lds, args: dict, scratch, tag, layout: dict) -> dict:
    """The attention back's context: LDS views (``lds``: an allocated
    ``back_lds``, or a dict of the members' views), mailbox (tag ``tag``),
    scratch regions, arguments by name."""
    s, d = key.tokens, Dims(key.tp)
    tid = tid_()
    if isinstance(lds, dict):
        spl, gml = lds["split"], lds["gemv"]
    else:
        spl, gml = lds.split.peek(), lds.gemv.peek()
    c = {
        "S": s,
        "tid": tid,
        "bid": bid_(),
        "lane": tid % 64,
        "wave": tid // 64,
        "red": gml.red.ptr,
        "xl": gml.xl.ptr,
        "xsl": gml.xsl.ptr,
        "d": d,
        "ratio": key.ratio,
        "shuffled": key.shuffled,
        "mb": Mailbox(tag),
        "nchunk": back_plan(key)[0],
        "sl": {"q": spl.q.ptr, "raw": spl.raw.ptr, "p": spl.p.ptr, "st": spl.st.ptr},
        "args": args,
    }
    for region, (off, _) in layout.items():
        c[region] = scratch + fx.Int64(off)
    return c


@flyc.jit
def run_back(c, key, bid, idle=None):
    """The attention back's stages on this CTA: split, combine, wo_a, wo_b.
    ``idle``: work for the CTAs past their attention units, after the attention work and
    before their GEMV weights load."""
    s = key.tokens
    dd = Dims(key.tp)
    # plain ints: a Python object read inside traced control flow is carried
    woa_tasks, groups, heads, group_k = dd.woa_tasks, dd.groups, dd.heads, dd.group_k
    q_rows, o_rows, wob_tasks = dd.q_rows, dd.o_rows, dd.wob_tasks
    a = c["args"]
    tiles = token_tiles(s)
    _, n_split, n_comb, COMB0, WOA0, WOB0, early_wob = back_plan(key)
    stamp(c, 0)
    if const_expr(n_split <= BLOCKS):
        # a split unit a CTA at most: CTA u runs unit u
        if bid < n_split:
            stage_split(c, bid)
    else:
        for unit in range(first_task(bid, 0), n_split, BLOCKS):
            stage_split(c, unit)
    stamp(c, 1)
    for unit in range(first_task(bid, COMB0), n_comb, BLOCKS):
        stage_combine(c, unit)
    gpu.barrier()
    stamp(c, 2)
    if const_expr(idle is not None):
        # before any GEMV weights load, so none are live under it
        idle()
    # the GEMV weights right after the attention work: the wo_a unit's (if any), then wo_b's
    nt = len(tiles)
    units = woa_units(s, dd)
    woa_spread = units > woa_tasks
    woa_unit = first_task(bid, WOA0)
    has_woa = woa_unit < units
    woa_t = fx.min(woa_unit % woa_tasks, woa_tasks - 1)
    ops = bf16_gemv_loads(
        c, a["woa"], a["woa_s"], group_k, woa_t * WOA_ROWS, WOA_SPLIT_K, pred=has_woa
    )
    wob_unit = first_task(bid, WOB0)
    if const_expr(nt == 1):
        # one or two wo_b tasks a CTA, every one on the step's rows
        wob_tile = 0
        wob_ts = [wob_unit, wob_unit + BLOCKS]
    else:
        # a tile a CTA (ROCm/ATOM's tile tasks): rows read once, every task's weights in flight
        wob_tile = wob_unit % nt
        per_tile = (BLOCKS - wob_tile + nt - 1) // nt
        most = -(-wob_tasks // (BLOCKS // nt))
        wob_ts = [wob_unit // nt + k * per_tile for k in range(most)]

    def wob_weights():
        # one tile: the first task is every CTA's (BLOCKS < wob_tasks), unmasked
        return [
            wob_loads(c, t)
            if nt == 1 and k == 0
            else wob_loads(c, fx.min(t, wob_tasks - 1), pred=t < wob_tasks)
            for k, t in enumerate(wob_ts)
        ]

    if const_expr(early_wob):
        wops = wob_weights()
    if const_expr(c.get("tl") is not None):
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
    stamp(c, 8)
    half_w = group_k // 4

    def woa_tile(ti, n_tokens):
        # wo_a on token tile ``ti`` (its first ``n_tokens`` rows' flags awaited)
        t0, n = tiles[ti]
        grp = woa_t // (woa_tasks // groups)
        await_flags(
            c,
            c["crdy"],
            n_tokens * COMB_PARTS,
            lambda u: comb_flag(
                t0 + u // COMB_PARTS, grp, u % COMB_PARTS, heads // COMB_HEADS
            ),
        )
        stamp(c, 9)
        # the group's XO rows a K half at a time, the next half's loads in flight
        off = [(grp * group_k + h * group_k // 2) // 2 for h in range(2)]
        regs = words_loads(c, c["xo"], q_rows // 2, off[0], half_w, t0, n)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for h in range_constexpr(2):
            words_store(c, regs, half_w, c["xl"], n)
            if const_expr(h == 0):
                regs = words_loads(c, c["xo"], q_rows // 2, off[1], half_w, t0, n)
            acc = bf16_gemv_mfmas(c, group_k, c["xl"], ops, acc, h, n)
            gpu.barrier()  # the next half's rows overwrite the LDS
        stage_woa(c, woa_t, acc, t0, n)

    if has_woa:
        if const_expr(woa_spread):
            # this unit's tile of its task
            for ti in range_constexpr(nt):
                if woa_unit // woa_tasks == ti:
                    woa_tile(ti, tiles[ti][1])
            stamp(c, 10)
            publish(c, c["ardy"], woa_unit * FLAG_STRIDE)
        else:
            # every token tile in turn, the first waiting on every row's flags
            for ti in range_constexpr(nt):
                woa_tile(ti, s if ti == 0 else 0)
            stamp(c, 10)
            publish(c, c["ardy"], woa_t * FLAG_STRIDE)
    stamp(c, 3)
    if const_expr(not early_wob):
        wops = wob_weights()
    stamp(c, 11)
    if const_expr(woa_spread):
        # this tile's wo_a units
        await_flags(
            c, c["ardy"], woa_tasks, lambda u: (wob_tile * woa_tasks + u) * FLAG_STRIDE
        )
    else:
        await_flags(c, c["ardy"], woa_tasks, lambda u: u * FLAG_STRIDE)
    stamp(c, 12)
    x8b = (
        c["x8b"],
        c["x8bs"],
        o_rows // 4,
        0,
        o_rows // 4,
        o_rows // 32,
        0,
        o_rows // 32,
    )

    def wob_tile_run(ti):
        # every wo_b task of this CTA on token tile ``ti``, two at a time
        t0, n = tiles[ti]
        rows_store(
            c,
            rows_loads(c, *x8b, t0, n),
            o_rows // 4,
            o_rows // 32,
            c["xl"],
            c["xsl"],
            n,
        )
        stamp(c, 14)
        for k in range_constexpr(0, len(wob_ts), 2):
            k2 = min(k + 1, len(wob_ts) - 1)
            # an odd last task runs alone: its pair is itself, masked off
            has2 = wob_ts[k2] < (wob_tasks if k2 > k else 0)
            run = (
                c,
                wob_ts[k],
                wops[k],
                fx.min(wob_ts[k2], wob_tasks - 1),
                wops[k2],
                has2,
                t0,
                n,
            )
            if const_expr(nt == 1 and k == 0):
                stage_wob(*run, mark=True)
            else:
                if wob_ts[k] < wob_tasks:
                    stage_wob(*run, mark=k == 0)
        stamp(c, 13)

    for ti in range_constexpr(nt):
        if wob_tile == ti:
            wob_tile_run(ti)
    stamp(c, 4)
