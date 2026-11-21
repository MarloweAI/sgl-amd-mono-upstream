# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's MoE all-reduces at the widths SGLang's own layers run (TP8, gfx950): a row's two-shot
all-reduce (``ar_read.py``'s protocol: sentinel words in launch-parity halves, f32 rank-order sums,
a CTA a row, one thread a 16 B unit), optionally with the RMSNorm of the sum in the same launch.

The latent half ([m, 3584]) takes the norm: ``routed_expert_norm``'s RMSNorm of the bf16 sum, the
latent up-projection's input. The shared half ([m, 7168]) is the plain all-reduce. Neither copies its
input into a registered buffer: the launch reads it with plain loads.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import hw_rsq, load_ptr64, rsrc, uniform, wave_sum
from sgl_amd_mono.common.plan import BLOCKS, kernel_sources, key_tuple
from sgl_amd_mono.kimi_k3.gfx950.stages.ar_read import (
    MAX_ROWS,
    TP,
    _ld4,
    _st4,
    dot8,
    f8,
    poll_sent4,
    st_sent,
    words8,
)
from sgl_amd_mono.kimi_k31.gfx950.stages import coll

SOURCES = kernel_sources(__name__)

WIDTHS = (3584, 7168)


def layout(h, max_rows=MAX_ROWS):
    """Byte offsets of one parity half ({"rs", "ag"}) and the half's size."""
    rs = TP * max_rows * (h // TP) * 2
    ag = max_rows * h * 2
    half = (rs + ag + 255) // 256 * 256
    return {"rs": 0, "ag": rs}, half


def peer_bytes(h, max_rows=MAX_ROWS):
    return 2 * layout(h, max_rows)[1] + guard.FLAG_BYTES


def arm(h, max_rows=MAX_ROWS):
    """A peer buffer's initial contents for width ``h``: the regions at the sentinel, the flags clear."""

    def init(buf):
        buf.zero_()
        buf[: 2 * layout(h, max_rows)[1]].fill_(0xFF)

    return init


@flyc.jit
def ar_norm_row(c, t, a, red, h, nw, max_rows, norm, eps):
    """Row ``t`` of ``a["x"]`` all-reduced into ``a["out"]`` (RMSNorm'd with ``a["w"]`` under ``norm``)."""
    sl = h // TP
    q8 = sl // 8
    tid = c["tid"]
    lane = tid % 64
    wave = uniform(tid // 64)
    rs = [c["rs_of"](p) for p in range(TP)]
    ag = [c["ag_of"](p) for p in range(TP)]
    row_w = t * (h // 2)
    xv = _ld4(a["x"], row_w + 4 * tid)
    j, q = tid // q8, tid % q8
    for p in range_constexpr(TP):
        if j == p:
            st_sent(rs[p], (c["rank"] * max_rows + t) * (sl // 2) + 4 * q, xv)
    if tid < q8:
        got = poll_sent4(
            c["rs_own"].value,
            [(src * max_rows + t) * (sl // 2) + 4 * tid for src in range(TP)],
        )
        acc = f8(got[0])
        for src in range_constexpr(1, TP):
            nxt = f8(got[src])
            acc = [acc[e] + nxt[e] for e in range(8)]
        summed = words8(acc)
        for p in range_constexpr(TP):
            st_sent(ag[p], row_w + c["rank"] * (sl // 2) + 4 * tid, summed)
    orow = a["out"] + fx.Int64(row_w) * 4
    if const_expr(norm):
        wv = fx.Vector(bo.buffer_load(rsrc(a["w"]), 4 * tid, vec_width=4, dtype=T.i32))
        s8 = f8(poll_sent4(c["ag_own"].value, [row_w + 4 * tid])[0])
        tot = wave_sum(dot8(s8, s8))
        if lane == 0:
            fx.ptr_store(tot, red + wave)
        gpu.barrier()
        ss = fx.ptr_load(red)
        for w in range_constexpr(1, nw):
            ss = ss + fx.ptr_load(red + w)
        scale = hw_rsq(ss * (1.0 / h) + eps)
        ws = f8(wv)
        _st4(words8([s8[e] * scale * ws[e] for e in range(8)]), orow, 4 * tid)
        gpu.barrier()  # red is read before the next row's stores
    else:
        _st4(poll_sent4(c["ag_own"].value, [row_w + 4 * tid])[0], orow, 4 * tid)


_BUILDS: dict = {}


def build(h, norm, eps, launches, max_rows=MAX_ROWS, blocks=BLOCKS):
    """The launch: ``launch(x, w, out, m, peers, rank, epoch, slot, stream=)``: this rank's partial x
    [m, h] bf16 -> out [m, h], the sum (RMSNorm'd with the bf16 weight w [h] under ``norm``)."""
    keyed = key_tuple(locals(), SOURCES)
    key = (h, norm, eps, launches, max_rows, blocks)
    if key in _BUILDS:
        return _BUILDS[key]
    assert h in WIDTHS
    nt = h // 8
    nw = nt // 64
    offs, half = layout(h, max_rows)
    fo = 2 * half
    name = f"k3_ar_norm_h{h}_n{int(norm)}_r{max_rows}_b{blocks}_l{launches}"

    @fx.struct
    class Red:
        v: fx.Array[fx.Float32, 16, 16]

    @flyc.kernel(name=name, known_block_size=[nt, 1, 1])
    def ar_norm_kernel(
        x: Int64, w: Int64, out: Int64, m: Int32, peers: Int64, rank: Int32, epoch: Int64, slot: Int32
    ):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        own = load_ptr64(peers, rank)
        with guard.scope(
            guard.Guard(epoch, own=own, peers=peers, rank=rank, npes=TP, flags_offset=fo)
        ):
            red = fx.SharedAllocator().allocate(Red).peek().v.ptr
            c = coll.region_context(peers, rank, epoch, slot, offs, half, tid, bid, launches)
            a = {"x": x, "w": w, "out": out}
            for t in range(bid, m, blocks):
                ar_norm_row(c, t, a, red, h, nw, max_rows, norm, eps)

    @flyc.jit
    def launch(
        x: Int64,
        w: Int64,
        out: Int64,
        m: Int32,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008
        _ = keyed
        ar_norm_kernel(x, w, out, m, peers, rank, epoch, slot).launch(
            grid=(blocks,), block=(nt,), stream=stream
        )

    _BUILDS[key] = launch
    return launch


# ---------------------------------------------------------------- the flat MoE all-reduce
# SGLang's latent MoE reduces one flat buffer [latent rows | shared rows] in one collective. Here a CTA
# owns row t of both halves (a latent row of LATENT columns and a shared row of SHARED columns): one
# two-shot exchange each, in one chain, and the latent's RMSNorm in the same launch.
LATENT, SHARED = 3584, 7168
MOE_NT = SHARED // 8  # threads: a shared unit each, the first LATENT / 8 a latent unit too
# the halves' reduces run on different waves: wave 0 the latent slice's units, waves 1-2 the shared's
SHARED_REDUCE0 = 64


def moe_layout(max_rows=MAX_ROWS):
    """Byte offsets of one parity half ({"rs_l", "ag_l", "rs_s", "ag_s"}) and its size."""
    offs, off = {}, 0
    for half_name, h in (("l", LATENT), ("s", SHARED)):
        offs["rs_" + half_name] = off
        off += TP * max_rows * (h // TP) * 2
        offs["ag_" + half_name] = off
        off += max_rows * h * 2
    return offs, (off + 255) // 256 * 256


def moe_peer_bytes(max_rows=MAX_ROWS):
    return 2 * moe_layout(max_rows)[1] + guard.FLAG_BYTES


def moe_arm(buf, max_rows=MAX_ROWS):
    buf.zero_()
    buf[: 2 * moe_layout(max_rows)[1]].fill_(0xFF)


@flyc.jit
def _push(c, rs, x, t, h, unit, max_rows):
    """This rank's 16 B unit ``unit`` of row t of the [m, h] partial x to its column slice's owner."""
    sl = h // TP
    q8 = sl // 8
    row_w = t * (h // 2)
    xv = _ld4(x, row_w + 4 * unit)
    j, q = unit // q8, unit % q8
    for p in range_constexpr(TP):
        if j == p:
            st_sent(rs[p], (c["rank"] * max_rows + t) * (sl // 2) + 4 * q, xv)


@flyc.jit
def _reduce(c, own, ag, t, h, q, max_rows):
    """Unit q of this rank's slice of row t: the 8 ranks' partials in rank order, to every rank."""
    sl = h // TP
    got = poll_sent4(own, [(src * max_rows + t) * (sl // 2) + 4 * q for src in range(TP)])
    acc = f8(got[0])
    for src in range_constexpr(1, TP):
        nxt = f8(got[src])
        acc = [acc[e] + nxt[e] for e in range(8)]
    summed = words8(acc)
    for p in range_constexpr(TP):
        st_sent(ag[p], t * (h // 2) + c["rank"] * (sl // 2) + 4 * q, summed)


@flyc.jit
def moe_row(c, t, a, red, max_rows, eps):
    tid = c["tid"]
    lane = tid % 64
    wave = uniform(tid // 64)
    lat = tid < LATENT // 8
    rs_l = [c["rs_l_of"](p) for p in range(TP)]
    ag_l = [c["ag_l_of"](p) for p in range(TP)]
    rs_s = [c["rs_s_of"](p) for p in range(TP)]
    ag_s = [c["ag_s_of"](p) for p in range(TP)]
    if lat:
        _push(c, rs_l, a["xl"], t, LATENT, tid, max_rows)
    _push(c, rs_s, a["xs"], t, SHARED, tid, max_rows)
    if tid < LATENT // TP // 8:
        _reduce(c, c["rs_l_own"].value, ag_l, t, LATENT, tid, max_rows)
    q = tid - SHARED_REDUCE0
    if (q >= 0) & (q < SHARED // TP // 8):
        _reduce(c, c["rs_s_own"].value, ag_s, t, SHARED, q, max_rows)
    wv = fx.Vector(bo.buffer_load(rsrc(a["wl"]), 4 * fx.min(tid, LATENT // 8 - 1), vec_width=4, dtype=T.i32))
    lrow, srow = t * (LATENT // 2), t * (SHARED // 2)
    _st4(poll_sent4(c["ag_s_own"].value, [srow + 4 * tid])[0], a["out_s"] + fx.Int64(srow) * 4, 4 * tid)
    zero = fx.Float32(0.0)
    if lat:
        lv = poll_sent4(c["ag_l_own"].value, [lrow + 4 * tid])[0]
        _st4(lv, a["out_l"] + fx.Int64(lrow) * 4, 4 * tid)
        lf = f8(lv)
        for e in range_constexpr(8):
            fx.ptr_store(lf[e], red + 16 + tid * 8 + e)
    gpu.barrier()
    # the latent row's square, wave order over the latent units (threads past them add zeros)
    lt = fx.min(tid, LATENT // 8 - 1)
    s8 = [lat.select(fx.ptr_load(red + 16 + lt * 8 + e), zero) for e in range(8)]
    tot = wave_sum(dot8(s8, s8))
    if lane == 0:
        fx.ptr_store(tot, red + wave)
    gpu.barrier()
    ss = fx.ptr_load(red)
    for w in range_constexpr(1, LATENT // 8 // 64):
        ss = ss + fx.ptr_load(red + w)
    scale = hw_rsq(ss * (1.0 / LATENT) + eps)
    if lat:
        ws = f8(wv)
        _st4(words8([s8[e] * scale * ws[e] for e in range(8)]), a["out_n"] + fx.Int64(lrow) * 4, 4 * tid)
    gpu.barrier()


_MOE_BUILDS: dict = {}


def build_moe(eps, launches, max_rows=MAX_ROWS, blocks=BLOCKS, emulate=False):
    """The launch: ``launch(xl, xs, wl, out_l, out_s, out_n, m, peers, rank, epoch, slot, stream=)``:
    this rank's latent [m, LATENT] and shared [m, SHARED] partials (bf16, contiguous rows) -> their
    sums out_l / out_s, and out_n = the RMSNorm of out_l with the bf16 weight wl. ``emulate``: all TP
    ranks in one launch on one GPU (a check): CTA b is rank b // blocks, the partials and outputs are
    [TP, max_rows, width], and ``rank`` is unused."""
    keyed = key_tuple(locals(), SOURCES)
    key = (eps, launches, max_rows, blocks, emulate)
    if key in _MOE_BUILDS:
        return _MOE_BUILDS[key]
    offs, half = moe_layout(max_rows)
    fo = 2 * half
    name = f"k3_moe_ar_r{max_rows}_b{blocks}_l{launches}_e{int(emulate)}"
    grid = blocks * TP if emulate else blocks

    @fx.struct
    class Red:
        v: fx.Array[fx.Float32, 16 + LATENT, 16]

    @flyc.kernel(name=name, known_block_size=[MOE_NT, 1, 1])
    def moe_ar_kernel(
        xl: Int64, xs: Int64, wl: Int64, out_l: Int64, out_s: Int64, out_n: Int64, m: Int32, peers: Int64,
        rank_arg: Int32, epoch: Int64, slot: Int32,
    ):  # fmt: skip
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        if const_expr(emulate):
            rank = uniform(bid // blocks)
            bid = bid % blocks
            sh_l = fx.Int64(rank) * fx.Int64(max_rows * LATENT * 2)
            sh_s = fx.Int64(rank) * fx.Int64(max_rows * SHARED * 2)
            xl, out_l, out_n = xl + sh_l, out_l + sh_l, out_n + sh_l
            xs, out_s = xs + sh_s, out_s + sh_s
        else:
            rank = rank_arg
        own = load_ptr64(peers, rank)
        with guard.scope(
            guard.Guard(epoch, own=own, peers=peers, rank=rank, npes=TP, flags_offset=fo)
        ):
            red = fx.SharedAllocator().allocate(Red).peek().v.ptr
            c = coll.region_context(peers, rank, epoch, slot, offs, half, tid, bid, launches)
            a = {"xl": xl, "xs": xs, "wl": wl, "out_l": out_l, "out_s": out_s, "out_n": out_n}
            for t in range(bid, m, blocks):
                moe_row(c, t, a, red, max_rows, eps)

    @flyc.jit
    def launch(
        xl: Int64, xs: Int64, wl: Int64, out_l: Int64, out_s: Int64, out_n: Int64, m: Int32, peers: Int64,
        rank: Int32, epoch: Int64, slot: Int32, stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008  # fmt: skip
        _ = keyed
        moe_ar_kernel(xl, xs, wl, out_l, out_s, out_n, m, peers, rank, epoch, slot).launch(
            grid=(grid,), block=(MOE_NT,), stream=stream
        )

    _MOE_BUILDS[key] = launch
    return launch
