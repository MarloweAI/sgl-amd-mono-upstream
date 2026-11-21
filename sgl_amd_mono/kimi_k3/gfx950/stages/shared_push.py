# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's latent MoE all-reduce with the shared experts' half reduce-scattered ahead (TP8, gfx950).

SGLang reduces the MoE's flat [latent | shared] partials in one all-reduce after the routed experts.
The shared half is final once the shared down projection has run, long before the routed experts end.
Here its reduce-scatter travels on its own:

* ``push``: once a [64, 64] tile of the shared down projection's bf16 output [m, 7168] is stored, the
  rank sends it to the tile's column owner (rank j owns columns [896 j, 896 (j + 1))), then raises one
  tagged flag on it. Nothing in it waits, so it runs inside a launch that keeps going (the routed
  experts' stage 1, whose grid also computes the shared down tiles).
* ``split``: after the routed experts, a CTA a row runs the latent half's two-shot all-reduce with
  ``routed_expert_norm``'s RMSNorm (``ar_norm.py``'s flat launch, latent part). Once the row's tiles
  are flagged, the owner sums its slice's 8 partials (its own from where the shared down wrote it) in
  f32 in rank order, the flat all-reduce's sum and rounding, and sends it to every rank in the same
  round as the latent's gathered slice.

Regions alternate halves by the launch parity (``coll.region_context``). A rank cannot push a layer's
partial into a half a peer still reads: between two MoE layers of one half every rank takes part in the
latent all-reduce, which no rank leaves before every owner has read its slots. Flags carry the launch
tag (step epoch x 256 + layer); the gathered rows are sentinel words their reader re-arms, so no region
is ever cleared.

``context`` / ``push_tile`` / ``flag_tile`` are the push's device functions, for a GEMM tile epilogue;
``build_push`` runs them on their own.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import CM_SYS, hw_rsq, load_ptr64, rsrc, uniform, wave_sum
from sgl_amd_mono.common.plan import kernel_sources, key_tuple
from sgl_amd_mono.kimi_k3.gfx950.stages.ar_norm import LATENT, SHARED, _push, _reduce
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

SL = SHARED // TP  # a rank's shared columns
TM, TN = 64, 64  # a pushed tile: rows x columns of the shared output, one flag
TILES_M = (MAX_ROWS + TM - 1) // TM
OT = SL // TN  # tiles an owner's slice a tile row
TU = TN // 8  # 16 B units a tile row
PUSH_NT = 512  # the standalone push: a tile's 16 B unit a thread
PUSH_BLOCKS = 32
SPLIT_NT = SHARED // 8  # split: a gathered shared unit a thread, the first LATENT / 8 a latent unit too
SPLIT_BLOCKS = 256
FLAG_T0 = 64  # split: the threads that poll the row's 7 x OT tile flags (wave 0 runs the latent reduce)
SUM_T0 = 192  # split: the threads that sum this rank's slice, a 16 B unit each
assert FLAG_T0 + (TP - 1) * OT <= SUM_T0 and SUM_T0 + SL // 8 <= SPLIT_NT


def layout(max_rows=MAX_ROWS):
    """Byte offsets of one parity half and its size.
    rs_l / ag_l: the latent two-shot's sentinel regions ([src][row][LATENT / TP], [row][LATENT]);
    ag_s: the gathered shared rows, sentinel words [row][SHARED]; rs_s: this rank's slice from every rank,
    [src][row][SL] bf16; fl: their flags, tagged pairs [src][tile row][OT]."""
    offs, off = {}, 0
    for name, size in (
        ("rs_l", TP * max_rows * (LATENT // TP) * 2),
        ("ag_l", max_rows * LATENT * 2),
        ("ag_s", max_rows * SHARED * 2),
        ("rs_s", TP * max_rows * SL * 2),
        ("fl", TP * TILES_M * OT * 8),
    ):
        offs[name] = off
        off += (size + 255) // 256 * 256
    return offs, off


def peer_bytes(max_rows=MAX_ROWS):
    return 2 * layout(max_rows)[1] + guard.FLAG_BYTES


def arm(buf, max_rows=MAX_ROWS):
    """The sentinel regions at the sentinel; slots, flags and the guard flags zero."""
    offs, half = layout(max_rows)
    buf.zero_()
    for p in range(2):
        buf[p * half + offs["rs_l"] : p * half + offs["rs_s"]].fill_(0xFF)


def slot_word(src, t, w, max_rows=MAX_ROWS):
    """Word offset of word w of row t of ``src``'s partial of the owner's slice in its rs_s region."""
    return (src * max_rows + t) * (SL // 2) + w


def tile_flag(src, tm, k):
    """Pair index of the flag of ``src``'s tile (tm, k-th of the owner's slice) in the owner's flag region."""
    return (src * TILES_M + tm) * OT + k


def context(peers, rank, epoch, slot, tid, bid, launches, max_rows=MAX_ROWS):
    """The push's region context in any launch: ``peers`` the peer buffer's address table, ``epoch`` the
    step epoch buffer, ``slot`` the MoE layer's launch slot, ``launches`` the slots a step."""
    offs, half = layout(max_rows)
    return coll.region_context(peers, rank, epoch, slot, offs, half, tid, bid, launches)


@flyc.jit
def push_tile(c, out, ld, tm, tn, m, tid, nthreads, max_rows):
    """The workgroup (``nthreads`` threads) stores tile (tm, tn) of out [m, ld] bf16 (rows < m) into its
    column owner's rs_s region (nothing when this rank owns it). The tile must be in memory: after its
    stores, the workgroup's ``s_waitcnt vmcnt(0)`` and a barrier. tm, tn: workgroup-uniform."""
    owner = uniform(tn // OT)
    dst = c["rs_s_of"](owner).value
    for i in range_constexpr((TM * TU + nthreads - 1) // nthreads):
        j = tid + nthreads * i
        t = tm * TM + j // TU
        u = j % TU
        if (j < TM * TU) & (t < m) & (owner != c["rank"]):
            v = _ld4(out, t * (ld // 2) + tn * (TN // 2) + 4 * u)
            _st4(v, dst, slot_word(c["rank"], t, (tn % OT) * (TN // 2) + 4 * u, max_rows), CM_SYS)


@flyc.jit
def flag_tile(c, tm, tn, tid):
    """Publishes tile (tm, tn) on its column owner, from thread 0. Only after the tile's stores drained:
    ``s_waitcnt vmcnt(0)`` on the threads that stored them, then a workgroup barrier."""
    owner = uniform(tn // OT)
    if (tid == 0) & (owner != c["rank"]):
        c["put"](c["fl_of"](owner), tile_flag(c["rank"], tm, tn % OT), fx.Int32(1))


_PUSH_BUILDS: dict = {}


def build_push(launches, max_rows=MAX_ROWS, blocks=PUSH_BLOCKS, emulate=False):
    """The launch: ``launch(xs, m, peers, rank, epoch, slot, stream=)``: this rank's shared partial xs
    [m, SHARED] (bf16, contiguous rows) to its column owners, a tile at a time. ``emulate``: all TP ranks in one
    launch on one GPU (a check): CTA b is rank b // blocks, xs is [TP, max_rows, SHARED]."""
    keyed = key_tuple(locals(), SOURCES)
    key = (launches, max_rows, blocks, emulate)
    if key in _PUSH_BUILDS:
        return _PUSH_BUILDS[key]
    name = f"k3_shared_push_r{max_rows}_b{blocks}_l{launches}_e{int(emulate)}"
    grid = blocks * TP if emulate else blocks

    @flyc.kernel(name=name, known_block_size=[PUSH_NT, 1, 1])
    def push_kernel(xs: Int64, m: Int32, peers: Int64, rank_arg: Int32, epoch: Int64, slot: Int32):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        if const_expr(emulate):
            rank = uniform(bid // blocks)
            bid = bid % blocks
            xs = xs + fx.Int64(rank) * fx.Int64(max_rows * SHARED * 2)
        else:
            rank = rank_arg
        c = context(peers, rank, epoch, slot, tid, bid, launches, max_rows)
        tiles_n = SHARED // TN
        tiles = (m + TM - 1) // TM * tiles_n
        for it in range(bid, tiles, blocks):
            push_tile(c, xs, SHARED, it // tiles_n, it % tiles_n, m, tid, PUSH_NT, max_rows)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        for it in range(bid, tiles, blocks):
            flag_tile(c, it // tiles_n, it % tiles_n, tid)

    @flyc.jit
    def launch(
        xs: Int64, m: Int32, peers: Int64, rank: Int32, epoch: Int64, slot: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008  # fmt: skip
        _ = keyed
        push_kernel(xs, m, peers, rank, epoch, slot).launch(grid=(grid,), block=(PUSH_NT,), stream=stream)

    _PUSH_BUILDS[key] = launch
    return launch


@flyc.jit
def slice_sum(c, t, q, xs, ag_s, max_rows):
    """Unit q of this rank's slice of row t of the shared sum: the 8 ranks' partials in rank order (its own
    from xs, the others' from rs_s), in f32, rounded once, then sent into every rank's gathered row."""
    rank = c["rank"]
    own = _ld4(xs, t * (SHARED // 2) + rank * (SL // 2) + 4 * q)
    rs = c["rs_s_own"].value
    parts = []
    for src in range_constexpr(TP):
        v = _ld4(rs, slot_word(src, t, 4 * q, max_rows), CM_SYS)
        mine = rank == src
        parts.append(fx.Vector.from_elements([mine.select(own[e], v[e]) for e in range(4)], fx.Int32))
    acc = f8(parts[0])
    for src in range_constexpr(1, TP):
        nxt = f8(parts[src])
        acc = [acc[e] + nxt[e] for e in range(8)]
    summed = words8(acc)
    for p in range_constexpr(TP):
        st_sent(ag_s[p], t * (SHARED // 2) + rank * (SL // 2) + 4 * q, summed)


@flyc.jit
def split_row(c, t, a, red, max_rows, eps):
    """Row t: the latent two-shot with its RMSNorm (``ar_norm.moe_row``'s latent part) and the shared
    sum: this rank's slice once the row's tiles are flagged, gathered in the latent's round."""
    tid = c["tid"]
    lane = tid % 64
    wave = uniform(tid // 64)
    lat = tid < LATENT // 8
    rs_l = [c["rs_l_of"](p) for p in range(TP)]
    ag_l = [c["ag_l_of"](p) for p in range(TP)]
    ag_s = [c["ag_s_of"](p) for p in range(TP)]
    if lat:
        _push(c, rs_l, a["xl"], t, LATENT, tid, max_rows)
    if tid < LATENT // TP // 8:
        _reduce(c, c["rs_l_own"].value, ag_l, t, LATENT, tid, max_rows)
    # this rank's slice of the row from the 7 peers: their tile flags (s7 skips this rank)
    f = tid - FLAG_T0
    s7 = f // OT
    src = s7 + (s7 >= c["rank"]).select(fx.Int32(1), fx.Int32(0))
    if (f >= 0) & (f < (TP - 1) * OT):
        c["poll"]([(c["fl_own"], tile_flag(src, t // TM, f % OT), 1)])
    gpu.barrier()
    q = tid - SUM_T0
    if (q >= 0) & (q < SL // 8):
        slice_sum(c, t, q, a["xs"], ag_s, max_rows)
    srow = t * (SHARED // 2)
    _st4(poll_sent4(c["ag_s_own"].value, [srow + 4 * tid])[0], a["out_s"] + fx.Int64(srow) * 4, 4 * tid)
    wv = fx.Vector(bo.buffer_load(rsrc(a["wl"]), 4 * fx.min(tid, LATENT // 8 - 1), vec_width=4, dtype=T.i32))
    lrow = t * (LATENT // 2)
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


_SPLIT_BUILDS: dict = {}


def build_split(eps, launches, max_rows=MAX_ROWS, blocks=SPLIT_BLOCKS, emulate=False):
    """The launch: ``launch(xl, xs, wl, out_l, out_s, out_n, m, peers, rank, epoch, slot, stream=)``:
    this rank's latent [m, LATENT] and shared [m, SHARED] partials (bf16, contiguous rows; the shared
    one pushed by ``push_tile`` / ``flag_tile`` with the same slot) -> the sums out_l / out_s and out_n = the
    RMSNorm of out_l with the bf16 weight wl. ``emulate`` as ``build_push``'s."""
    keyed = key_tuple(locals(), SOURCES)
    key = (eps, launches, max_rows, blocks, emulate)
    if key in _SPLIT_BUILDS:
        return _SPLIT_BUILDS[key]
    offs, half = layout(max_rows)
    fo = 2 * half
    name = f"k3_moe_split_r{max_rows}_b{blocks}_l{launches}_e{int(emulate)}"
    grid = blocks * TP if emulate else blocks

    @fx.struct
    class Red:
        v: fx.Array[fx.Float32, 16 + LATENT, 16]

    @flyc.kernel(name=name, known_block_size=[SPLIT_NT, 1, 1])
    def split_kernel(
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
                split_row(c, t, a, red, max_rows, eps)

    @flyc.jit
    def launch(
        xl: Int64, xs: Int64, wl: Int64, out_l: Int64, out_s: Int64, out_n: Int64, m: Int32, peers: Int64,
        rank: Int32, epoch: Int64, slot: Int32, stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008  # fmt: skip
        _ = keyed
        split_kernel(xl, xs, wl, out_l, out_s, out_n, m, peers, rank, epoch, slot).launch(
            grid=(grid,), block=(SPLIT_NT,), stream=stream
        )

    _SPLIT_BUILDS[key] = launch
    return launch
