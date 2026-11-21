# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's latent MoE all-reduce with the shared experts' half pushed ahead (TP8, gfx950).

SGLang reduces the MoE's flat [latent | shared] partials in one all-reduce after the routed experts.
The shared half is final once the shared down projection has run, long before the routed experts end.
Here it travels on its own:

* ``push``: right after the shared down projection, every rank stores its bf16 shared partial
  [m, 7168] into its own slot of every peer's slot region (one-shot), then raises a tagged flag a
  (row, 896-column chunk) on each peer. The launch waits for nothing, so the routed experts can run
  beside it.
* ``split``: after the routed experts, a CTA a row runs the latent half's two-shot all-reduce with
  ``routed_expert_norm``'s RMSNorm (``ar_norm.py``'s flat launch, latent part), and sums the row's 8
  shared partials (its own from where the shared down wrote it, the peers' from their slots) in f32 in
  rank order, the same sum and rounding as the flat all-reduce.

Slot regions and flags alternate halves by the launch parity (``coll.region_context``). A rank cannot
push a layer's partial into a half a peer still reads: between two MoE layers of one half every rank
takes part in the latent all-reduce, which no rank leaves before every peer's ``split`` has read its
slots. Flags carry the launch tag (step epoch x 256 + layer), so no region is ever cleared.

``push_chunk`` / ``flag_chunk`` are the push's device functions, for a GEMM epilogue that pushes its
own output tile: a (row, chunk) is published once its 896 columns are stored on every peer and drained.
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
    words8,
)
from sgl_amd_mono.kimi_k31.gfx950.stages import coll

SOURCES = kernel_sources(__name__)

CHUNK = 896  # columns a flag publishes
NCH = SHARED // CHUNK  # chunks a row
CU = CHUNK // 8  # 16 B units a chunk
PUSH_NT = 64 * TP  # the push: wave p stores to rank p
PUSH_BLOCKS = 32
SPLIT_NT = SHARED // 8  # split: a shared unit a thread, the first LATENT / 8 a latent unit too
SPLIT_BLOCKS = 256
FLAG_T0 = 64  # split: the threads that poll the shared flags (wave 1; wave 0 runs the latent reduce)


def layout(max_rows=MAX_ROWS):
    """Byte offsets of one parity half and its size.
    rs_l / ag_l: the latent two-shot's sentinel regions ([src][row][LATENT / TP], [row][LATENT]);
    sh: the shared slots [src][row][SHARED] bf16; fl: their flags, tagged pairs [src][row][NCH]."""
    offs, off = {}, 0
    for name, size in (
        ("rs_l", TP * max_rows * (LATENT // TP) * 2),
        ("ag_l", max_rows * LATENT * 2),
        ("sh", TP * max_rows * SHARED * 2),
        ("fl", TP * max_rows * NCH * 8),
    ):
        offs[name] = off
        off += (size + 255) // 256 * 256
    return offs, off


def peer_bytes(max_rows=MAX_ROWS):
    return 2 * layout(max_rows)[1] + guard.FLAG_BYTES


def arm(buf, max_rows=MAX_ROWS):
    """The latent regions at the sentinel; slots, flags and the guard flags zero."""
    offs, half = layout(max_rows)
    buf.zero_()
    for p in range(2):
        buf[p * half + offs["rs_l"] : p * half + offs["sh"]].fill_(0xFF)


def slot_word(rank, t, k, u, max_rows=MAX_ROWS):
    """Word offset of unit u of chunk k of row t of ``rank``'s slot in a slot region."""
    return (rank * max_rows + t) * (SHARED // 2) + k * (CHUNK // 2) + 4 * u


def flag_pair(rank, t, k, max_rows=MAX_ROWS):
    """Pair index of the flag of (``rank``'s slot, row t, chunk k) in a flag region."""
    return (rank * max_rows + t) * NCH + k


@flyc.jit
def push_chunk(c, xs, ldx, t, k, wave, lane, max_rows):
    """Stores chunk k of row t of the partial xs (row stride ldx bf16) into this rank's slot on rank
    ``wave`` (wave-uniform, this rank's own wave stores nothing)."""
    dst = c["sh_of"](wave).value
    for i in range_constexpr((CU + 63) // 64):
        u = lane + 64 * i
        if (u < CU) & (wave != c["rank"]):
            v = _ld4(xs, t * (ldx // 2) + k * (CHUNK // 2) + 4 * u)
            _st4(v, dst, slot_word(c["rank"], t, k, u, max_rows), CM_SYS)


@flyc.jit
def flag_chunk(c, t, k, wave, lane, max_rows):
    """Publishes (row t, chunk k) of this rank's slot on rank ``wave``; after the chunk's stores drained
    (``s_waitcnt vmcnt(0)`` on every wave that stored, then a CTA barrier)."""
    if (lane == 0) & (wave != c["rank"]):
        c["put"](c["fl_of"](wave), flag_pair(c["rank"], t, k, max_rows), fx.Int32(1))


_PUSH_BUILDS: dict = {}


def build_push(launches, max_rows=MAX_ROWS, blocks=PUSH_BLOCKS, emulate=False):
    """The launch: ``launch(xs, m, peers, rank, epoch, slot, stream=)``: this rank's shared partial xs
    [m, SHARED] (bf16, contiguous rows) into its slot on every peer. ``emulate``: all TP ranks in one
    launch on one GPU (a check): CTA b is rank b // blocks, xs is [TP, max_rows, SHARED]."""
    keyed = key_tuple(locals(), SOURCES)
    key = (launches, max_rows, blocks, emulate)
    if key in _PUSH_BUILDS:
        return _PUSH_BUILDS[key]
    offs, half = layout(max_rows)
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
        c = coll.region_context(peers, rank, epoch, slot, offs, half, tid, bid, launches)
        wave = uniform(tid // 64)
        lane = tid % 64
        items = m * NCH
        for it in range(bid, items, blocks):
            push_chunk(c, xs, SHARED, it // NCH, it % NCH, wave, lane, max_rows)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        for it in range(bid, items, blocks):
            flag_chunk(c, it // NCH, it % NCH, wave, lane, max_rows)

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
def shared_sum(c, t, xs, out_s, max_rows):
    """Unit tid of row t of the shared sum: the 8 ranks' partials in rank order (this rank's from xs,
    the peers' from their slots, once flagged), in f32, rounded once."""
    tid, rank = c["tid"], c["rank"]
    srow = t * (SHARED // 2)
    own = _ld4(xs, srow + 4 * tid)
    sh = c["sh_own"].value
    k = tid // CU
    u = tid % CU
    parts = []
    for src in range_constexpr(TP):
        v = _ld4(sh, slot_word(src, t, k, u, max_rows), CM_SYS)
        mine = rank == src
        parts.append(fx.Vector.from_elements([mine.select(own[e], v[e]) for e in range(4)], fx.Int32))
    acc = f8(parts[0])
    for src in range_constexpr(1, TP):
        nxt = f8(parts[src])
        acc = [acc[e] + nxt[e] for e in range(8)]
    _st4(words8(acc), out_s + fx.Int64(srow) * 4, 4 * tid)


@flyc.jit
def split_row(c, t, a, red, max_rows, eps):
    """Row t: the latent two-shot with its RMSNorm (``ar_norm.moe_row``'s latent part) and the shared
    sum from the slots, the latter while the latent's gathered slices travel."""
    tid = c["tid"]
    lane = tid % 64
    wave = uniform(tid // 64)
    lat = tid < LATENT // 8
    rs_l = [c["rs_l_of"](p) for p in range(TP)]
    ag_l = [c["ag_l_of"](p) for p in range(TP)]
    if lat:
        _push(c, rs_l, a["xl"], t, LATENT, tid, max_rows)
    if tid < LATENT // TP // 8:
        _reduce(c, c["rs_l_own"].value, ag_l, t, LATENT, tid, max_rows)
    f = tid - FLAG_T0
    src = f // NCH
    if (f >= 0) & (f < TP * NCH):
        if src != c["rank"]:
            c["poll"]([(c["fl_own"], flag_pair(src, t, f % NCH, max_rows), 1)])
    gpu.barrier()
    shared_sum(c, t, a["xs"], a["out_s"], max_rows)
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
    one pushed by ``build_push``'s launch of the same slot) -> the sums out_l / out_s and out_n = the
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
