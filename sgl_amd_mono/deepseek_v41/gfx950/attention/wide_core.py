# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1's sparse attention core at wide row counts (60 .. 384 rows): a request's
six verify rows read the request's keys once.

SGLang's core runs a CTA a (row, 16-head group) and reads every row's 128 window and 512
top-k records itself. Rows of one request share most of them: the windows overlap in all
but five positions and the top-k sets in most of their keys. Here a request's keys are
the union of its rows' lists, read once, and its rows and heads share the MFMA's M:

- ``union``: one CTA a request turns its rows' index lists (one per cache) into the
  union of their slots with a row mask a slot (bit r: row r reads it), grouped by mask
  (the most rows first), each group in first-appearance order: blocks that only some
  rows read then skip the others;
- ``core``: one CTA a (request, 16-head group, key split) walks every S-th 32-key block
  of the request's union: a block's records are gathered into registers two blocks
  ahead and dequantized into LDS once; waves 0 .. 5 score them against row 0 .. 5's 16
  heads (keys outside a row's mask score nothing; a row with no key in the block skips
  it) and run the online softmax; every wave adds the block's P V for 64 of the 512 dims
  and every row. The splits of a request (on one XCD) exchange f16 partials, each over
  its own sum, through its L2 and merge them in split order with the sink and the inverse
  RoPE of the last 64 dims, as SGLang's split reduce does.

The records keep the plugin's layout (V4's fp8_ds_mla pages: 576 data bytes a token,
448 e4m3 then 64 bf16 rope dims, then 8 bytes of E8M0 codes a token after the page's data
rows). A key's dequantized values are SGLang's; the sums run in another order (the union's
blocks, not each row's own), so the outputs match SGLang's to rounding, not bit for bit.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    ballot,
    lane_gather,
    fresh,
    global_load,
    hw_exp2,
    mfma_bf16,
    popcount,
    rsrc,
    uniform,
    xred,
)
from sgl_amd_mono.common.plan import kernel_sources, key_tuple
from sgl_amd_mono.common.sync import _spin

from .device import fp8x8_bf16, pow2

SOURCES = kernel_sources(__name__)

G = 6  # verify rows a request (DSpark block 5)
HT = 16  # heads a CTA
D, NOPE, ROPE, DATA = 512, 448, 64, 576
BK = 32  # keys a block
THREADS = 512
WAVES = THREADS // 64
RW = 260  # i32 words a key in LDS: its 512 dims as bf16 + 16 bytes of pad
PW = 20  # i32 words a (row, head)'s P in LDS: 32 bf16 + 8 of pad (16-byte rows)
CHUNKS = BK * DATA // 16  # 16-byte data pieces a block
NEG = -3.4028234663852886e38
LOG2E = 1.4426950408889634
FLAG = 16  # i32 words a split flag (a line of its own)
QK_AHEAD = 2  # QK steps whose K reads are in flight under a step's MFMAs
HASH = -1640531535  # 0x9E3779B1 as an i32


def cdiv(a, b):
    return -(-a // b)


def _ld(ptr, i, cm=0):
    return fx.Int32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32, cache_modifier=cm))


# ---------------------------------------------------------------- union


# the union's order of row masks: the most rows first, then the mask
MASK_ORDER = sorted(range(1, 64), key=lambda m: (-bin(m).count("1"), m))


def table_size(width: int) -> int:
    """The union's hash table: a power of two at least twice a request's entries."""
    n = 1024
    while n < 2 * G * width:
        n *= 2
    return n


_UNIONS: dict = {}


def union_build(width: int, stride: int, limit: int):
    """``launch(lists, lens, slots, masks, counts, reqs)``: per request (a CTA each) the
    union of its rows' slots. ``lists``: i32 [reqs * 6, stride], row t's first ``width``
    entries live up to ``lens[t]`` (entries outside [0, ``limit``) none); ``slots``,
    ``masks``: i32 [reqs, 6 * width], the union's slots and their row masks, grouped by
    mask (``MASK_ORDER``; past the union slot 0, mask 0); ``counts``: i32 [reqs]."""
    keyed = key_tuple({"width": width, "stride": stride, "limit": limit}, SOURCES)
    key = (width, stride, limit)
    if key in _UNIONS:
        return _UNIONS[key]
    assert width & (width - 1) == 0, width
    n = G * width
    ht = table_size(width)
    rounds = cdiv(n, THREADS)
    wshift = width.bit_length() - 1
    hshift = 32 - (ht.bit_length() - 1)

    @fx.struct
    class Smem:
        keys: fx.Array[fx.Int32, ht, 16]
        masks: fx.Array[fx.Int32, ht, 16]
        first: fx.Array[fx.Int32, ht, 16]
        cnt: fx.Array[fx.Int32, rounds * WAVES * 64, 16]
        tot: fx.Array[fx.Int32, 136, 16]

    @flyc.kernel(name=f"ds41_union_wide_w{width}", known_block_size=[THREADS, 1, 1])
    def union_kernel(lists: Int64, lens: Int64, slots: Int64, masks: Int64, counts: Int64):
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        req = fx.block_idx.x
        sm = fx.SharedAllocator().allocate(Smem).peek()
        keys, msk, fst, cnt, tot = sm.keys.ptr, sm.masks.ptr, sm.first.ptr, sm.cnt.ptr, sm.tot.ptr
        for i in range_constexpr(ht // THREADS):
            fx.ptr_store(fx.Int32(-1), keys + (tid + THREADS * i))
            fx.ptr_store(fx.Int32(0), msk + (tid + THREADS * i))
            fx.ptr_store(fx.Int32(0x7FFFFFFF), fst + (tid + THREADS * i))
        # this thread's entries: entry tid + 512 e of the request's rows, row by row
        ents = []
        for e in range_constexpr(rounds):
            idx = tid + THREADS * e
            row = fx.min(idx, n - 1) >> wshift
            col = idx & (width - 1)
            t = req * G + row
            slot = _ld(lists, t * stride + col)
            ok = (slot >= 0) & (slot < limit) & (col < _ld(lens, t))
            if const_expr(n % THREADS != 0):
                ok = ok & (idx < n)
            ents.append((idx, row, slot, ok))
        gpu.barrier()
        found = []
        for e in range_constexpr(rounds):
            idx, row, slot, ok = ents[e]
            pos = ((slot * HASH) >> hshift) & (ht - 1)
            if ok:
                old = fx.llvm.atomic_cas(keys + pos, fx.Int32(-1), slot, syncscope="workgroup")[0]
                while (old != fx.Int32(-1)) & (old != slot):
                    pos = (pos + 1) & (ht - 1)
                    old = fx.llvm.atomic_cas(keys + pos, fx.Int32(-1), slot, syncscope="workgroup")[0]
                fx.llvm.atomic_or(msk + pos, fx.Int32(1) << row, syncscope="workgroup")
                fx.llvm.atomic_min(fst + pos, idx, syncscope="workgroup")
            found.append(pos)
        gpu.barrier()
        # an entry is the union's where its slot first appears (lowest row, then column);
        # the union groups slots by row mask (the most rows first, then the mask), each group
        # in that first-appearance order: per (round, wave) a count and a rank a mask
        for i in range_constexpr(rounds * WAVES * 64 // THREADS):
            fx.ptr_store(fx.Int32(0), cnt + (tid + THREADS * i))
        gpu.barrier()
        incs, rmasks, ranks = [], [], []
        for e in range_constexpr(rounds):
            idx, row, slot, ok = ents[e]
            pos = found[e]
            inc = ok & (fx.ptr_load(fst + pos) == idx)
            mk = inc.select(fx.ptr_load(msk + pos), fx.Int32(0))
            # the wave's lanes with this lane's mask: six ballots, one a mask bit
            same = ballot(inc)
            for bit in range_constexpr(6):
                on = ((mk >> bit) & 1) != 0
                word = ballot(inc & on)
                same = same & on.select(word, word ^ fx.Int64(-1))
            below = same & ((fx.Int64(1) << fx.Int64(lane)) - 1)
            rank = popcount(below)
            if inc & (below == fx.Int64(0)):
                fx.ptr_store(popcount(same), cnt + ((e * WAVES + wave) * 64 + mk))
            incs.append(inc)
            rmasks.append(mk)
            ranks.append(rank)
        gpu.barrier()
        # mask m: its (round, wave) counts to exclusive offsets (every count read first),
        # its total to ``tot``
        if tid < 64:
            cs = [fx.ptr_load(cnt + (i * 64 + tid)) for i in range(rounds * WAVES)]
            run = fx.Int32(0)
            for i in range_constexpr(rounds * WAVES):
                fx.ptr_store(run, cnt + (i * 64 + tid))
                run = run + cs[i]
            fx.ptr_store(run, tot + tid)
        gpu.barrier()
        # the masks' offsets in the union's order: lane l of wave 0 takes the l-th mask,
        # a scan over the lanes
        if wave == 0:
            om = fx.Int32(0)
            for i in range_constexpr(len(MASK_ORDER)):
                om = (lane == i).select(fx.Int32(MASK_ORDER[i]), om)
            v = (lane < len(MASK_ORDER)).select(fx.ptr_load(tot + om), fx.Int32(0))
            inc = v
            for off in range_constexpr(6):
                prev = lane_gather(inc, fx.max(lane - (1 << off), 0))
                inc = inc + (lane >= (1 << off)).select(prev, fx.Int32(0))
            if lane < len(MASK_ORDER):
                fx.ptr_store(inc - v, tot + 64 + om)
            if lane == 63:
                fx.ptr_store(inc, tot + 128)
        gpu.barrier()
        base = fx.ptr_load(tot + 128)
        for e in range_constexpr(rounds):
            mk = rmasks[e]
            at = fx.ptr_load(tot + 64 + mk) + fx.ptr_load(cnt + ((e * WAVES + wave) * 64 + mk)) + ranks[e]
            if incs[e]:
                bo.buffer_store(ents[e][2], rsrc(slots), req * n + at)
                bo.buffer_store(mk, rsrc(masks), req * n + at)
        # past the union: slot 0 under an empty mask, so a block read past it gathers a real
        # record and scores nothing
        for e in range_constexpr(rounds):
            idx = tid + THREADS * e
            if (idx >= base) & (idx < n):
                bo.buffer_store(fx.Int32(0), rsrc(slots), req * n + idx)
                bo.buffer_store(fx.Int32(0), rsrc(masks), req * n + idx)
        if tid == 0:
            bo.buffer_store(base, rsrc(counts), req)

    @flyc.jit
    def launch(
        lists: Int64,
        lens: Int64,
        slots: Int64,
        masks: Int64,
        counts: Int64,
        reqs: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = keyed
        union_kernel(lists, lens, slots, masks, counts).launch(
            grid=(reqs,), block=(THREADS,), stream=stream
        )

    _UNIONS[key] = launch
    return launch




# ---------------------------------------------------------------- core

WB = cdiv(128 + 8, BK)  # a request's window blocks: 128 + 5 keys (5 of 32)
TOPB = G * 512 // BK  # a request's top-k blocks at most
ROUNDS = cdiv(CHUNKS, THREADS)  # a thread's data pieces a block (the last round's tail: none)
QLOADS = D // 32  # a thread's q loads (a 16-byte piece a QK step)
VM_BLOCK = 2 * ROUNDS + ROUNDS + 1  # a thread's vector-memory ops a block: pieces, scale words, slots, mask


class Cache(msgspec.Struct, frozen=True):
    """A paged record cache: tokens a page (a power of two) and the page stride in bytes."""

    page: int
    stride: int

    def __post_init__(self):
        assert self.page & (self.page - 1) == 0, self.page


class CoreBuild(msgspec.Struct, frozen=True):
    reqs: int  # the most requests (rows / 6) a launch takes
    heads: int  # this rank's heads, 16 a CTA
    splits: int  # key splits a (request, head group)
    window: Cache  # the SWA cache
    extra: Cache  # the top-k cache
    qs: int  # q's row stride (elements)
    qh: int  # q's head stride (elements)
    scale: float  # the softmax scale
    rope_stride: int  # the inverse-RoPE table's position stride (f32)
    pos_words: int  # i32 words a position (2: int64 positions)

    @property
    def groups(self) -> int:
        return self.heads // HT


def groups_per_xcd(reqs: int, heads: int) -> int:
    """The core's (request, head group)s an XCD (its grid is 8 * splits * this)."""
    return cdiv(reqs * (heads // HT), 8)


def splits_for(reqs: int, heads: int) -> int:
    """Key splits a (request, head group): the most, a power of two (they share the
    merge's 96 pieces evenly), at most 16, that keep every CTA resident, 32 an XCD (a
    group's splits share an XCD and wait on each other)."""
    most = max(1, min(16, 32 // groups_per_xcd(reqs, heads)))
    return 1 << (most.bit_length() - 1)


def workspace(b: CoreBuild):
    """(f16 elements of partial outputs, f32 of partial (max, sum), i32 of flags) for
    ``b.reqs`` requests."""
    ctas = b.reqs * b.groups * b.splits
    return ctas * G * HT * D, ctas * G * HT * 2, ctas * FLAG


class _Seg(msgspec.Struct):
    """A block's segment: its cache (base, page shift, page size, page stride), its union
    (slots, masks, the request's offset in them) and its first key."""

    base: object
    shift: object
    page: object
    stride: object
    slots: object
    masks: object
    off: object
    k0: object


def _piece(c):
    """Data piece c of a block: (key, piece of the key's 36)."""
    key = (c * 1821) >> 16  # c // 36 for c < 2304
    return key, c - key * 36


def _slots(tid, sg):
    """A thread's slots of a block: its data pieces' (tid + 512 j) keys' (past the block:
    the last key's) and key tid's row mask (0 past the block's 32). A union's tail holds
    slot 0 under mask 0."""
    out = []
    for j in range_constexpr(ROUNDS):
        key, _ = _piece(fx.min(tid + THREADS * j, CHUNKS - 1))
        out.append(_ld(sg.slots, sg.off + sg.k0 + key))
    m = _ld(sg.masks, sg.off + sg.k0 + tid % BK)
    out.append((tid < BK).select(m, fx.Int32(0)))
    return out


def _loads(tid, sg, slots):
    """A block's record loads from its slots into registers, the same count every thread
    (a piece past the block reloads the last): each data piece and the scale word holding
    its 64-dim group's code; streamed past the L2 (read once)."""
    out = []
    for j in range_constexpr(ROUNDS):
        _, piece = _piece(fx.min(tid + THREADS * j, CHUNKS - 1))
        slot = slots[j]
        page, off = slot >> sg.shift, slot & (sg.page - 1)
        row = sg.base + fx.Int64(page) * sg.stride
        data = global_load(row + fx.Int64(off * DATA + piece * 16), 4, nt=True)
        sw = global_load(row + fx.Int64(sg.page * DATA + off * 8 + (fx.min(piece, 27) >> 4) * 4), 1, nt=True)
        out.append((data, fx.Int32(sw)))
    return out


@flyc.jit
def _dequant_round(tid, j, loads, kv):
    """Round j of a block's loads to bf16 rows of ``kv`` (e4m3 times its 64-dim group's
    power of two: exact; rope dims as they are)."""
    c = tid + THREADS * j
    key, piece = _piece(fx.min(c, CHUNKS - 1))
    data, sw = loads[j]
    code = (sw >> (8 * ((fx.min(piece, 27) >> 2) & 3))) & 0xFF
    base = kv + key * RW
    live = True if (j + 1) * THREADS <= CHUNKS else c < CHUNKS
    if live:
        if piece < NOPE // 16:
            fx.ptr_store(fp8x8_bf16(data[0], data[1], pow2(code)), base + 8 * piece)
            fx.ptr_store(fp8x8_bf16(data[2], data[3], pow2(code)), base + 8 * piece + 4)
        else:
            fx.ptr_store(data, base + (NOPE // 2 + 4 * (piece - NOPE // 16)))


@flyc.jit
def _dequant_masks(tid, wave, lane, mask, kmask, rows):
    """Key tid's row mask to ``kmask``, the rows any key of the block has to ``rows``."""
    if tid < BK:
        fx.ptr_store(mask, kmask + tid)
    if wave == 0:
        anyr = mask
        for off in range_constexpr(6):
            anyr = xred(anyr, 1 << off, lambda x, y: x | y)
        if lane == 0:
            fx.ptr_store(anyr, rows)


def _dequant(tid, wave, lane, loads, mask, kv, kmask, rows):
    """A block's loads to bf16 rows of ``kv``, its masks to ``kmask`` and ``rows``."""
    for j in range_constexpr(ROUNDS):
        _dequant_round(tid, j, loads, kv)
    _dequant_masks(tid, wave, lane, mask, kmask, rows)


def _qk(lane, kv, qf):
    """The block's scores against one row's 16 heads: two 16-key tiles of S^T, element i
    of tile kt key 16 kt + 4 (lane / 16) + i, head lane % 16."""
    g, r16 = lane // 16, lane % 16
    v4 = fx.Vector.make_type(4, fx.Int32)
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]

    def step(s):
        return [fx.ptr_load(kv + ((16 * kt + r16) * RW + 16 * s + 4 * g), result_type=v4) for kt in range(2)]

    # step s + QK_AHEAD's reads go out before step s's MFMAs
    ahead = [step(s) for s in range(QK_AHEAD)]
    for s in range_constexpr(16):
        if const_expr(s + QK_AHEAD < 16):
            ahead.append(step(s + QK_AHEAD))
        cur = ahead.pop(0)
        for kt in range_constexpr(2):
            acc[kt] = mfma_bf16(fx.Vector(cur[kt]).bitcast(fx.BFloat16), qf[s], acc[kt])
        rocdl.sched_barrier(0)
    return acc


def _v_operands(lane, wave, kv):
    """PV's A operands of the block for the wave's 64 dims: dt = 0 .. 3 holds keys
    8 (lane / 16) .. + 8 at dim 64 wave + 4 (lane % 16) + dt."""
    g, r16 = lane // 16, lane % 16
    v2 = fx.Vector.make_type(2, fx.Int32)
    w = [fx.ptr_load(kv + ((8 * g + k) * RW + 32 * wave + 2 * r16), result_type=v2) for k in range(8)]
    ops = []
    for dt in range_constexpr(4):
        e, sh = dt // 2, 16 * (dt % 2)
        half = [(w[k][e] >> sh) & 0xFFFF for k in range(8)]
        ops.append(fx.Vector.from_elements([half[2 * i] | (half[2 * i + 1] << 16) for i in range(4)], fx.Int32).bitcast(fx.BFloat16))
    return ops


def _softmax(lane, r, sacc, kmask, m, l, qk_scale, prow):
    """Row r's online softmax over the block: the new (max, sum, factor of the earlier
    sums), its P (bf16; keys outside the row's mask 0) to the row's LDS."""
    g = lane // 16
    kms = [fx.ptr_load(kmask + (16 * kt + 4 * g), result_type=fx.Vector.make_type(4, fx.Int32)) for kt in range(2)]
    valid = [[((kms[kt][i] >> r) & 1) != 0 for i in range(4)] for kt in range(2)]
    sv = [[valid[kt][i].select(sacc[kt][i] * qk_scale, fx.Float32(NEG)) for i in range(4)] for kt in range(2)]
    mb = sv[0][0]
    for kt in range_constexpr(2):
        for i in range_constexpr(4):
            if const_expr(kt + i > 0):
                mb = fx.max(mb, sv[kt][i])
    mb = xred(xred(mb, 16, fx.max), 32, fx.max)
    mn = fx.max(m, mb)
    a = hw_exp2(m - mn)
    ps = [[valid[kt][i].select(hw_exp2(sv[kt][i] - mn), fx.Float32(0.0)) for i in range(4)] for kt in range(2)]
    ls = (ps[0][0] + ps[0][1]) + (ps[0][2] + ps[0][3])
    ls = ls + ((ps[1][0] + ps[1][1]) + (ps[1][2] + ps[1][3]))
    ls = xred(xred(ls, 16, lambda x, y: x + y), 32, lambda x, y: x + y)
    for kt in range_constexpr(2):
        pw = fx.Vector.from_elements(ps[kt], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        fx.ptr_store(pw, prow + (8 * kt + 2 * g))
    return mn, l * a + ls, a


def _scaled(v, a):
    return fx.Vector.from_elements([v[i] * a for i in range(4)], fx.Float32)


CM_L2 = 1  # SC0: past the CU's cache, from the XCD's L2


@flyc.jit
def _merge(lane, wave, b, req, hg, sp, first, sink, freqs, positions, out, part, pml):
    """Split sp's share of the group's output: the (core wave w, row r, element pair ip)
    pieces sp * 96 / S .., a wave a piece at a time, each split's 16 bytes a lane (8 f16 of
    its partial over its own sum) in one read; merged in split order with the sink, then
    the inverse RoPE of the last 64 dims. A wave's loads, all its pieces', go out before
    any of its math."""
    S = b.splits
    pairs = G * HT
    share = WAVES * G * 2 // S
    nk = cdiv(share, WAVES)
    g, hl = lane // 16, lane % 16
    h = hg * HT + hl
    snk = fx.Float32(bo.buffer_load(rsrc(sink), h, vec_width=1, dtype=T.f32)) * LOG2E
    # the request's row positions, a row a lane (read across lanes below)
    pos_all = _ld(positions, (req * G + fx.min(lane, G - 1)) * b.pos_words)
    grp = first // S
    nml = 2 * S  # f32 a (row, head): its splits' (max, sum)
    got = []
    for k in range_constexpr(nk):
        kk = fx.min(k * WAVES + wave, share - 1)
        it = sp * share + kk
        w, r, ip = it // (G * 2), (it // 2) % G, it % 2
        t = req * G + r
        mlw = [
            fx.Vector(bo.buffer_load(rsrc(pml), (grp * pairs + r * HT + hl) * nml + 4 * e, vec_width=min(4, nml), dtype=T.f32, cache_modifier=CM_L2))
            for e in range(cdiv(nml, 4))
        ]
        mls = [(mlw[(2 * u) // 4][(2 * u) % 4], mlw[(2 * u + 1) // 4][(2 * u + 1) % 4]) for u in range(S)]
        vs = [
            fx.Vector(bo.buffer_load(rsrc(part), (((((first + u) * WAVES + w) * G + r) * 2 + ip) * 64 + lane) * 4, vec_width=4, dtype=T.i32, cache_modifier=CM_L2))
            for u in range(S)
        ]
        d0 = 64 * w + 16 * g + 8 * ip
        pos = lane_gather(pos_all, r)
        fw = pos * b.rope_stride + fx.max(d0 - NOPE, 0)
        # the RoPE dims' (cos, sin), the last wave's pieces only
        f0 = fx.Vector.filled(4, 0.0, fx.Float32)
        f1 = fx.Vector.filled(4, 0.0, fx.Float32)
        if w == WAVES - 1:
            f0 = fx.Vector(bo.buffer_load(rsrc(freqs), fw, vec_width=4, dtype=T.f32))
            f1 = fx.Vector(bo.buffer_load(rsrc(freqs), fw + 4, vec_width=4, dtype=T.f32))
        got.append((w, t, d0, mls, vs, [f0, f1]))
    for k in range_constexpr(nk):
        w, t, d0, mls, vs, frs = got[k]
        mf = snk
        for u in range_constexpr(S):
            mf = fx.max(mf, mls[u][0])
        lf = fx.Float32(0.0)
        tot = [fx.Float32(0.0)] * 8
        for u in range_constexpr(S):
            wgt = hw_exp2(mls[u][0] - mf) * mls[u][1]
            lf = lf + wgt
            vals = fx.Vector(vs[u]).bitcast(fx.Float16).to(fx.Float32)
            tot = [tot[d] + wgt * vals[d] for d in range(8)]
        inv = fx.Float32(1.0) / (lf + hw_exp2(snk - mf))
        o = [fx.Float32(fx.Float32(tot[d] * inv).to(fx.BFloat16)) for d in range(8)]
        # the inverse GPT-J RoPE of the last 64 dims: e' = e cos + o sin, o' = o cos - e sin
        rope = w == WAVES - 1
        y = []
        for e in range_constexpr(4):
            cs = rope.select(frs[e // 2][2 * (e % 2)], fx.Float32(1.0))
            sn = rope.select(frs[e // 2][2 * (e % 2) + 1], fx.Float32(0.0))
            e_, o_ = o[2 * e], o[2 * e + 1]
            y += [e_ * cs + o_ * sn, o_ * cs - e_ * sn]
        wd = fx.Vector.from_elements(y, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        if k * WAVES + wave < share:
            bo.buffer_store(wd, rsrc(out), ((t * b.heads + h) * D + d0) // 2)


_CORES: dict = {}


def core_build(b: CoreBuild):
    """``launch(q, wcache, ecache, wslots, wmasks, wcounts, eslots, emasks, ecounts, sink,
    freqs, positions, out, part, pml, flags, reqs, per)`` (device addresses, then the
    launch's requests and ``groups_per_xcd`` of them): every row's attention over its
    request's union (``union_build`` of the window lists into ``w*``, of the top-k lists
    into ``e*``), out bf16 [rows, heads, 512], its last 64 dims inverse-rotated at the
    row's position (``freqs`` f32 [positions, 64]: (cos, sin) pairs). ``part``, ``pml``,
    ``flags`` as ``workspace`` sizes them for ``b.reqs``; flags zeroed once and kept.

    A CTA a (request, 16-head group, key split) takes blocks sp, sp + S, .. of the
    request's union (``WB`` window blocks of 32 keys, then the top-k union's). Every thread
    gathers its pieces of a block's records into registers two blocks ahead and, between
    the rows' P V of the block before, widens them to bf16 in LDS; waves 0 .. 5 score row
    0 .. 5 (its q in registers; a row with no key in the block skips it) and run its
    online softmax; then every wave adds the block's P V for its 64 dims of every row. A (request, head
    group)'s splits share an XCD: each stores its partial (f16, over its own sum), waits
    for the others' flags and merges its share of the output (``_merge``)."""
    keyed = key_tuple(b, SOURCES)
    if b in _CORES:
        return _CORES[b]
    S, NG = b.splits, b.groups
    assert b.heads % HT == 0 and (WAVES * G * 2) % S == 0, b
    assert S * groups_per_xcd(b.reqs, b.heads) <= 32, b
    wcap, ecap = G * 128, G * 512
    pairs = G * HT
    qk_scale = b.scale * LOG2E

    @fx.struct
    class Smem:
        kv: fx.Array[fx.Int32, 2 * BK * RW, 16]
        kmask: fx.Array[fx.Int32, 2 * BK, 16]
        rows: fx.Array[fx.Int32, 2 * 4, 16]
        p: fx.Array[fx.Int32, G * HT * PW, 16]
        alpha: fx.Array[fx.Int32, G * HT, 16]
        ml: fx.Array[fx.Int32, 2 * G * HT, 16]

    name = f"ds41_core_wide_h{b.heads}_s{S}_p{b.window.page}_{b.extra.page}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def core_kernel(
        q: Int64, wcache: Int64, ecache: Int64, wslots: Int64, wmasks: Int64, wcounts: Int64,
        eslots: Int64, emasks: Int64, ecounts: Int64, sink: Int64, freqs: Int64,
        positions: Int64, out: Int64, part: Int64, pml: Int64, flags: Int64, reqs: Int32,
        per: Int32,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        g, r16 = lane // 16, lane % 16
        # a (request, head group)'s splits on one XCD (CTAs go round-robin over the 8),
        # so their partials meet in its L2: XCD x takes groups x * per .. + per
        bid = fx.block_idx.x
        xcd, j = bid % 8, bid // 8
        grp = xcd * per + j // S
        sp = j % S
        live = grp < reqs * NG
        grp = fx.min(grp, reqs * NG - 1)
        task = grp * S + sp
        req = grp // NG
        hg = grp % NG
        sm = fx.SharedAllocator().allocate(Smem).peek()
        kv, kmask, rl = sm.kv.ptr, sm.kmask.ptr, sm.rows.ptr
        pl, al, mll = sm.p.ptr, sm.alpha.ptr, sm.ml.ptr

        def seg(blk):
            w = blk < WB
            kb = fx.min(w.select(blk, blk - WB), TOPB - 1)
            return _Seg(
                base=w.select(fx.Int64(wcache), fx.Int64(ecache)),
                shift=w.select(fx.Int32(b.window.page.bit_length() - 1), fx.Int32(b.extra.page.bit_length() - 1)),
                page=w.select(fx.Int32(b.window.page), fx.Int32(b.extra.page)),
                stride=w.select(fx.Int64(b.window.stride), fx.Int64(b.extra.stride)),
                slots=w.select(fx.Int64(wslots), fx.Int64(eslots)),
                masks=w.select(fx.Int64(wmasks), fx.Int64(emasks)),
                off=w.select(req * wcap, req * ecap),
                k0=kb * BK,
            )

        def blkof(t):
            return sp + t * S

        # row (wave)'s q, waves 0 .. 5: head 16 hg + lane % 16, dims 32 s + 8 (lane / 16) ..;
        # older than every record load (the waits below count those)
        ne = uniform(_ld(ecounts, req))
        nb = WB + (ne + BK - 1) // BK
        nj = live.select((nb - sp + S - 1) // S, fx.Int32(0))
        qbase = (req * G + fx.min(wave, G - 1)) * b.qs + (hg * HT + r16) * b.qh

        def q_loads():
            return [
                fx.Vector(bo.buffer_load(rsrc(q), (qbase + 32 * s + 8 * g) // 2, vec_width=4, dtype=T.i32)).bitcast(fx.BFloat16)
                for s in range(QLOADS)
            ]

        # in flight when block t + 1's loads are waited for: block t + 2's slots, then its
        # loads and block t + 3's slots (``VM_BLOCK`` ops)
        # the record chain first (slots, then records), row q behind block 0's records
        s0 = _slots(tid, seg(blkof(0)))
        s1 = _slots(tid, seg(blkof(1)))
        l0 = _loads(tid, seg(blkof(0)), s0)
        qf = q_loads()
        s2 = _slots(tid, seg(blkof(2)))
        pend = _loads(tid, seg(blkof(1)), s1)
        mask1 = s1[-1]
        rocdl.s_waitcnt(vmcnt=VM_BLOCK + QLOADS)
        _dequant(tid, wave, lane, l0, s0[-1], kv, kmask, rl)
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(G * 4)]
        for t in range(0, nj):
            x = t % 2
            kx = kv + x * (BK * RW)
            gpu.barrier()
            act = uniform(fx.ptr_load(rl + 4 * x))
            # block t + 2's records in flight under this block (block t + 3's slots too)
            s3 = _slots(fresh(tid), seg(blkof(t + 3)))
            nxt = _loads(fresh(tid), seg(blkof(t + 2)), s2)
            mask2 = s2[-1]
            s2 = s3
            if wave < G:
                if ((act >> wave) & 1) != 0:
                    sacc = _qk(lane, kx, qf)
                    m, l, a = _softmax(lane, wave, sacc, kmask + x * BK, m, l, qk_scale, pl + (wave * HT + r16) * PW)
                    if g == 0:
                        fx.ptr_store(a.bitcast(fx.Int32), al + (wave * HT + r16))
            gpu.barrier()
            # block t + 1 into the other buffer (block t - 1's, read last block), its
            # rounds between the rows' P V
            y = 1 - x
            ky = kv + y * (BK * RW)
            rocdl.s_waitcnt(vmcnt=VM_BLOCK)
            dq = fresh(tid)
            av = _v_operands(lane, wave, kx)

            def p_of(r):
                # row r's factor and P, read a row ahead of its MFMAs
                return (
                    fx.ptr_load(al + (r * HT + r16)),
                    fx.ptr_load(pl + ((r * HT + r16) * PW + 4 * g), result_type=fx.Vector.make_type(4, fx.Int32)),
                )

            pn = p_of(0)
            for r in range_constexpr(G):
                # a row the block has keys of: its sums rescaled, then its P V
                pc = pn
                if const_expr(r + 1 < G):
                    pn = p_of(r + 1)
                mine = [acc[r * 4 + dt] for dt in range(4)]
                if ((act >> r) & 1) != 0:
                    alpha = pc[0].bitcast(fx.Float32)
                    pb = fx.Vector(pc[1]).bitcast(fx.BFloat16)
                    mine = [mfma_bf16(av[dt], pb, _scaled(mine[dt], alpha)) for dt in range(4)]
                acc = acc[: r * 4] + mine + acc[r * 4 + 4 :]
                if const_expr(r % 2 == 1):
                    _dequant_round(dq, r // 2, pend, ky)
            _dequant_masks(dq, wave, lane, mask1, kmask + y * BK, rl + 4 * y)
            pend = nxt
            mask1 = mask2
        # every row's (max, sum) to LDS: each wave's partial is over its rows' own sums
        if (wave < G) & (g == 0):
            fx.ptr_store(m.bitcast(fx.Int32), mll + 2 * (wave * HT + r16))
            fx.ptr_store(l.bitcast(fx.Int32), mll + 2 * (wave * HT + r16) + 1)
        gpu.barrier()
        # the partials as the lanes hold them: (row r, elements 2 ip, 2 ip + 1) of wave
        # w are dims 64 w + 16 (lane / 16) + 8 ip .. + 8 of head lane % 16, a lane's 8 f16
        # at part[task][w][r][ip][lane]; a CTA past the groups stores nothing
        if live:
            for r in range_constexpr(G):
                ll = fx.ptr_load(mll + 2 * (r * HT + r16) + 1).bitcast(fx.Float32)
                inv = (ll > fx.Float32(0.0)).select(fx.Float32(1.0) / ll, fx.Float32(0.0))
                for ip in range_constexpr(2):
                    v = fx.Vector.from_elements(
                        [acc[r * 4 + dt][2 * ip + e] * inv for e in range(2) for dt in range(4)], fx.Float32
                    )
                    o0 = ((((task * WAVES + wave) * G + r) * 2 + ip) * 64 + lane) * 4
                    bo.buffer_store(v.to(fx.Float16).bitcast(fx.Int32), rsrc(part), o0)
        if (wave < G) & (g == 0) & live:
            bo.buffer_store(fx.Vector.from_elements([m, l], fx.Float32), rsrc(pml), ((grp * pairs + wave * HT + r16) * S + sp) * 2)
        # the group's splits stored (a flag a split, flipped a launch, device scope);
        # their partials stay in the XCD's L2, read past the CU's cache
        old = _ld(flags, task * FLAG, CM_DEV)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        phase = 1 - old
        if (tid == 0) & live:
            bo.buffer_store(phase, rsrc(flags), task * FLAG, cache_modifier=CM_DEV)
        first = task - sp
        if (tid < S) & live:

            def load_all():
                return _ld(flags, (first + tid) * FLAG, CM_DEV)

            _spin(load_all, lambda v: v != phase, guarded=False)
        gpu.barrier()
        if live:
            _merge(lane, wave, b, req, hg, sp, first, sink, freqs, positions, out, part, pml)

    @flyc.jit
    def launch(
        q: Int64, wcache: Int64, ecache: Int64, wslots: Int64, wmasks: Int64, wcounts: Int64,
        eslots: Int64, emasks: Int64, ecounts: Int64, sink: Int64, freqs: Int64,
        positions: Int64, out: Int64, part: Int64, pml: Int64, flags: Int64, reqs: Int32,
        per: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):  # fmt: skip
        _ = keyed
        core_kernel(
            q, wcache, ecache, wslots, wmasks, wcounts, eslots, emasks, ecounts, sink,
            freqs, positions, out, part, pml, flags, reqs, per,
        ).launch(grid=(8 * S * per,), block=(THREADS,), stream=stream)  # fmt: skip

    _CORES[b] = launch
    return launch
