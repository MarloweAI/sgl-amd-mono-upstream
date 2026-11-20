# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's o_proj TP all-reduce with the FFN read of the attention-residual bank, in one launch.

SGLang runs the two as the all-reduce (aiter lamport two-shot or ``cross_device_reduce_2stage``)
and ``attn_res_hip._agg_kernel``. Here a CTA owns a token row (rows bid, bid + blocks, ...) and,
for that row:

1. pushes the row's 8 column slices of this rank's partial to their owners (rank j owns columns
   [j SL, (j + 1) SL));
2. while the slices travel, loads the row's banked snapshots (rows of the bank past ``nvb`` read
   as zeros, unloaded) and their score sums: the dot with cw and the sum of squares;
3. sums its own slice's 8 partials in f32 in rank order (the bf16 sum is kimi_k31 ``coll.py``'s
   all-reduce) and pushes it into every rank's gathered row;
4. reads the gathered row and runs ``_agg_kernel``'s math with the pending add:
   prefix = bf16(residual + sum) -> ``pre_out`` (without a residual the sum itself), the softmax
   mix of the snapshots and the prefix by their RMS-normed scores, and the output RMSNorm -> ``out``.

The regions are sentinel words (``sync.put_sent``): a 16 B system-scope store of data that is never
the all-ones word, which its reader polls off the sentinel and re-arms, so no flag follows the data.
They alternate halves by the launch parity (``coll.region_context``): a rank a launch ahead writes the
other half, and none gets two ahead, since every launch reads every rank's data.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import (
    CM_SYS,
    bf2_f32,
    bf16_pair,
    bf16_round,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    load_ptr64,
    rsrc,
    uniform,
    wave_sum,
)
from sgl_amd_mono.common.plan import BLOCKS, kernel_sources, key_tuple
from sgl_amd_mono.common.sync import SENT, SENT_NAN, _spin
from sgl_amd_mono.kimi_k31.gfx950.stages import coll

SOURCES = kernel_sources(__name__)

TP = 8
H = 7168
SL = H // TP  # a rank's columns
Q8 = SL // 8  # 16 B units of a slice row
U = H // 8  # 16 B units of a row
NT = U  # threads a CTA: a 16 B unit of the row each
NW = NT // 64 + (NT % 64 > 0)  # its waves
# a CTA's waves fill 4 a SIMD: the budget of 128 VGPRs that allows (the default allows 64, and spills)
WAVES_PER_EU = 4
NB_MAX = 8  # banked snapshots K3 reads at most (93 layers, a block every 12)
LOG2E = 1.4426950408889634
# reduction slots a wave: the snapshots' dots with cw and squares (2 NB_MAX), the snapshot mix's
# square, then the prefix's dot with cw, its square and its dot with the snapshot mix
S_MIX = 2 * NB_MAX
S_PRE = S_MIX + 1
NS = S_PRE + 3
# waves that move the snapshots into LDS; the first ones keep their loads free for the reduce's polls
DMA_WAVES = 8
assert (NB_MAX * NT) % (64 * DMA_WAVES) == 0
MAX_ROWS = 192


def layout(max_rows=MAX_ROWS):
    """Byte offsets of one parity half ({"rs", "ag"}) and the half's size.
    rs: [src][row][SL] bf16 partials of this rank's slice; ag: [row][H] the gathered sums."""
    rs = TP * max_rows * SL * 2
    ag = max_rows * H * 2
    half = (rs + ag + 255) // 256 * 256
    return {"rs": 0, "ag": rs}, half


def peer_bytes(max_rows=MAX_ROWS):
    """The peer buffer a rank needs: two halves + the guard flags."""
    return 2 * layout(max_rows)[1] + guard.FLAG_BYTES


def arm(buf, max_rows=MAX_ROWS):
    """A peer buffer's initial contents: the regions at the sentinel, the guard flags clear."""
    buf.zero_()
    buf[: 2 * layout(max_rows)[1]].fill_(0xFF)


def _ld4(addr, word, cm=0, nbytes=None):
    return fx.Vector(
        bo.buffer_load(
            rsrc(addr, nbytes), word, vec_width=4, dtype=T.i32, cache_modifier=cm
        )
    )


def _st4(vec, addr, word, cm=0):
    bo.buffer_store(vec, rsrc(addr), word, cache_modifier=cm)


def st_sent(a, word, vec):
    """A 16 B system-scope store of four words into sentinel region ``a`` (a ``sync.Addr``)."""
    v = fx.Vector.from_elements(
        [(vec[e] == fx.Int32(SENT)).select(fx.Int32(SENT_NAN), vec[e]) for e in range(4)],
        fx.Int32,
    )
    _st4(v, a.value, word, CM_SYS)


def poll_sent4(base, words):
    """The 16 B vectors at ``words`` of this rank's sentinel region ``base`` once no word is the
    sentinel, then re-armed."""
    n = len(words)

    def load_all():
        got = []
        for w in words:
            v = _ld4(base, w, CM_SYS)
            got += [v[e] for e in range(4)]
        return fx.Vector.from_elements(got, fx.Int32)

    def pending(v):
        bad = v[0] == fx.Int32(SENT)
        for e in range_constexpr(1, 4 * n):
            bad = bad | (v[e] == fx.Int32(SENT))
        return bad

    v = _spin(load_all, pending)
    sent = fx.Vector.from_elements([fx.Int32(SENT)] * 4, fx.Int32)
    for w in words:
        _st4(sent, base, w, CM_SYS)
    return [fx.Vector.from_elements([v[4 * i + e] for e in range(4)], fx.Int32) for i in range(n)]


def f8(vec):
    """A 16 B vector's 8 bf16 as f32."""
    return [x for e in range(4) for x in bf2_f32(vec[e])]


def words8(vals):
    """8 f32 -> a 16 B vector of packed bf16."""
    return fx.Vector.from_elements(
        [bf16_pair(vals[2 * k], vals[2 * k + 1]).bitcast(fx.Int32) for k in range(4)],
        fx.Int32,
    )


@flyc.jit
def block_sums(vals, red, slot0, tid, lane, wave):
    """Every thread's ``vals`` summed over the CTA: lane order in a wave, then wave order;
    the totals land at red[NW * NS + slot0 + j] (read after the next barrier)."""
    n = len(vals)
    for j in range_constexpr(n):
        tot = wave_sum(vals[j])
        if lane == 0:
            fx.ptr_store(tot, red + wave * NS + slot0 + j)
    gpu.barrier()
    if tid < n:
        acc = fx.ptr_load(red + slot0 + tid)
        for w in range_constexpr(1, NW):
            acc = acc + fx.ptr_load(red + w * NS + slot0 + tid)
        fx.ptr_store(acc, red + NW * NS + slot0 + tid)


def total(red, j):
    return fx.ptr_load(red + NW * NS + j)


def lds4(lt, r, tid):
    """Snapshot r's 16 B unit of thread ``tid`` from the LDS tile [r][NT][4 words]."""
    return fx.Vector(
        fx.ptr_load(lt + (r * NT + tid) * 4, result_type=fx.Vector.make_type(4, fx.Int32))
    )


def cw8(cw, tid):
    """cw's 8 f32 of thread ``tid``'s unit."""
    v = [
        fx.Vector(bo.buffer_load(rsrc(cw), 8 * tid + 4 * h, vec_width=4, dtype=T.f32))
        for h in range(2)
    ]
    return [v[h][e] for h in range(2) for e in range(4)]


def dot8(xs, ys):
    """sum x y over 8 elements, in element order."""
    d = xs[0] * ys[0]
    for e in range_constexpr(1, 8):
        d = d + xs[e] * ys[e]
    return d


@flyc.jit
def ar_read_row(c, t, a, red, lt, max_rows, eps_s, eps_o):
    """Row ``t``: the all-reduce of ``a["x"]``'s row and the FFN read (module docstring).
    ``a``: the launch's addresses and runtime scalars by name. Thread u holds 16 B unit u of
    the row (8 columns); in the push, unit q of slice j is thread j Q8 + q."""
    tid = c["tid"]
    lane = tid % 64
    wave = uniform(tid // 64)
    x, res, bank, cw, w_o = a["x"], a["res"], a["bank"], a["cw"], a["w_o"]
    nvb, has_res, nbk = a["nvb"], a["has_res"], a["nbk"]
    rs = [c["rs_of"](p) for p in range(TP)]
    ag = [c["ag_of"](p) for p in range(TP)]
    row_w = t * (H // 2)  # the row's first word

    # 1. this rank's partial of the row, slice j to rank j
    xv = _ld4(x, row_w + 4 * tid)
    j, q = tid // Q8, tid % Q8
    for p in range_constexpr(TP):
        if j == p:
            st_sent(rs[p], (c["rank"] * max_rows + t) * (SL // 2) + 4 * q, xv)

    # 2. this row's snapshots into LDS while the slices travel (bank rows past nvb are out of the
    # resource: zeros, no traffic), by the last DMA_WAVES waves
    brow = bank + fx.Int64(t) * fx.Int64(nbk) * fx.Int64(H * 2)
    rb = rsrc(brow, fx.Int64(nvb) * fx.Int64(H * 2))
    if wave >= NW - DMA_WAVES:
        dw = wave - (NW - DMA_WAVES)
        for i in range_constexpr(NB_MAX * NT // (64 * DMA_WAVES)):
            u0 = (i * DMA_WAVES + dw) * 64  # the wave's first unit of [r][NT]
            rocdl.raw_ptr_buffer_load_lds(
                rb, (lt + u0 * 4).llvm_ptr, 16, (u0 + lane) * 16, fx.Int32(0), 0, 0
            )

    # 3. own slice: the 8 ranks' partials in rank order -> every rank's gathered row
    if tid < Q8:
        got = poll_sent4(
            c["rs_own"].value,
            [(src * max_rows + t) * (SL // 2) + 4 * tid for src in range(TP)],
        )
        acc = f8(got[0])
        for src in range_constexpr(1, TP):
            nxt = f8(got[src])
            acc = [acc[e] + nxt[e] for e in range(8)]
        summed = words8(acc)
        for p in range_constexpr(TP):
            st_sent(ag[p], row_w + c["rank"] * (SL // 2) + 4 * tid, summed)

    # the snapshots' score sums, while the sums travel
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    cws = cw8(cw, tid)
    sums = []
    for r in range_constexpr(NB_MAX):
        b8 = f8(lds4(lt, r, tid))
        sums += [dot8(b8, cws), dot8(b8, b8)]
    block_sums(sums, red, 0, tid, lane, wave)
    gpu.barrier()
    # the snapshots' part of the mix, by their own scores (bmax: their largest): bmix = sum_r e_r b_r,
    # e_r = exp(score_r - bmax); the prefix's score then rescales it
    inv_h = 1.0 / H
    zero = fx.Float32(0.0)
    neg = fx.Float32(float("-inf"))
    scores = []
    for r in range_constexpr(NB_MAX):
        sc = total(red, 2 * r) * hw_rsq(total(red, 2 * r + 1) * inv_h + eps_s)
        scores.append((nvb > r).select(sc, neg))
    bmax = scores[0]
    for r in range_constexpr(1, NB_MAX):
        bmax = fx.max(bmax, scores[r])
    ex = [(nvb > r).select(hw_exp2((scores[r] - bmax) * LOG2E), zero) for r in range(NB_MAX)]
    esum = ex[0]
    for r in range_constexpr(1, NB_MAX):
        esum = esum + ex[r]
    bmix = [ex[0] * b for b in f8(lds4(lt, 0, tid))]
    for r in range_constexpr(1, NB_MAX):
        b8 = f8(lds4(lt, r, tid))
        bmix = [bmix[e] + ex[r] * b8[e] for e in range(8)]
    block_sums([dot8(bmix, bmix)], red, S_MIX, tid, lane, wave)
    res_bytes = fx.Int64(has_res) * fx.Int64(H * 2)
    resv = _ld4(res + fx.Int64(row_w) * 4, 4 * tid, nbytes=res_bytes)
    wov = fx.Vector(bo.buffer_load(rsrc(w_o), 4 * tid, vec_width=4, dtype=T.i32))

    # 4. the gathered row, the prefix and the read: one reduction round once the row is here; the
    # mix's square follows from (wp prefix + fb bmix)^2's three dot products
    ag_v = poll_sent4(c["ag_own"].value, [row_w + 4 * tid])[0]
    s8, r8 = f8(ag_v), f8(resv)
    with_res = has_res != 0
    pv = [with_res.select(bf16_round(r8[e] + s8[e]), s8[e]) for e in range(8)]
    _st4(words8(pv), a["pre_out"] + fx.Int64(row_w) * 4, 4 * tid)
    block_sums([dot8(pv, cws), dot8(pv, pv), dot8(pv, bmix)], red, S_PRE, tid, lane, wave)
    gpu.barrier()
    pp = total(red, S_PRE + 1)
    p_score = total(red, S_PRE) * hw_rsq(pp * inv_h + eps_s)
    mx = fx.max(p_score, bmax)
    ex_p = hw_exp2((p_score - mx) * LOG2E)
    fb = hw_exp2((bmax - mx) * LOG2E)
    inv = hw_rcp(ex_p + fb * esum)
    wp = ex_p * inv
    fb = fb * inv
    ssq = wp * wp * pp + 2.0 * wp * fb * total(red, S_PRE + 2) + fb * fb * total(red, S_MIX)
    scale = hw_rsq(ssq * inv_h + eps_o)
    ws = f8(wov)
    _st4(
        words8([(pv[e] * wp + fb * bmix[e]) * scale * ws[e] for e in range(8)]),
        a["out"] + fx.Int64(row_w) * 4,
        4 * tid,
    )

_BUILDS: dict = {}


def build(eps_s, eps_o, launches, max_rows=MAX_ROWS, blocks=BLOCKS, emulate=False):
    """The launch: ``launch(x, res, bank, cw, w_o, pre_out, out, m, nvb, nbk, has_res, peers,
    rank, epoch, slot, stream=)``.

    x: this rank's o_proj partial [m, H] bf16; res: the pending residual [m, H] (unread when
    ``has_res`` is 0); bank: the snapshots [m, nbk, H] (the first ``nvb`` <= NB_MAX read); cw: f32 [H],
    score_norm.weight x score_proj.weight; w_o: bf16 [H], the output RMSNorm weight;
    pre_out / out: [m, H] bf16. ``slot``: the launch's index in its step of ``launches``.
    ``emulate``: all TP ranks in one launch on one GPU (a check): CTA b is rank b // blocks;
    x, pre_out and out are [TP, max_rows, H], and ``rank`` is unused."""
    keyed = key_tuple(locals(), SOURCES)
    key = (eps_s, eps_o, launches, max_rows, blocks, emulate)
    if key in _BUILDS:
        return _BUILDS[key]
    assert blocks % TP == 0
    offs, half = layout(max_rows)
    fo = 2 * half
    grid = blocks * TP if emulate else blocks
    rank_stride = max_rows * H * 2
    name = f"k3_ar_read_r{max_rows}_b{blocks}_l{launches}_e{int(emulate)}_w{WAVES_PER_EU}"

    @fx.struct
    class Red:
        v: fx.Array[fx.Float32, (NW + 1) * NS, 16]

    @fx.struct
    class Tile:  # the row's snapshots, [r][thread][4 words]
        v: fx.Array[fx.Int32, NB_MAX * NT * 4, 16]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def ar_read_kernel(
        x: Int64,
        res: Int64,
        bank: Int64,
        cw: Int64,
        w_o: Int64,
        pre_out: Int64,
        out: Int64,
        m: Int32,
        nvb: Int32,
        nbk: Int32,
        has_res: Int32,
        peers: Int64,
        rank_arg: Int32,
        epoch: Int64,
        slot: Int32,
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        if const_expr(emulate):
            rank = uniform(bid // blocks)
            bid = bid % blocks
            shift_r = fx.Int64(rank) * fx.Int64(rank_stride)
            x = x + shift_r
            pre_out = pre_out + shift_r
            out = out + shift_r
        else:
            rank = rank_arg
        own = load_ptr64(peers, rank)
        with guard.scope(
            guard.Guard(epoch, own=own, peers=peers, rank=rank, npes=TP, flags_offset=fo)
        ):
            alloc = fx.SharedAllocator()
            red = alloc.allocate(Red).peek().v.ptr
            lt = alloc.allocate(Tile).peek().v.ptr
            c = coll.region_context(
                peers, rank, epoch, slot, offs, half, tid, bid, launches
            )
            a = {
                "x": x,
                "res": res,
                "bank": bank,
                "cw": cw,
                "w_o": w_o,
                "pre_out": pre_out,
                "out": out,
                "nvb": nvb,
                "has_res": has_res,
                "nbk": nbk,
            }
            for t in range(bid, m, blocks):
                ar_read_row(c, t, a, red, lt, max_rows, eps_s, eps_o)

    @flyc.jit
    def launch(
        x: Int64,
        res: Int64,
        bank: Int64,
        cw: Int64,
        w_o: Int64,
        pre_out: Int64,
        out: Int64,
        m: Int32,
        nvb: Int32,
        nbk: Int32,
        has_res: Int32,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008
        _ = keyed
        ar_read_kernel(
            x, res, bank, cw, w_o, pre_out, out, m, nvb, nbk, has_res, peers, rank, epoch, slot
        ).launch(grid=(grid,), block=(NT,), stream=stream)

    def run(*args, **kwargs):
        with CompilationContext.compile_hints({"waves_per_eu": WAVES_PER_EU}):
            return launch(*args, **kwargs)

    _BUILDS[key] = run
    return run
