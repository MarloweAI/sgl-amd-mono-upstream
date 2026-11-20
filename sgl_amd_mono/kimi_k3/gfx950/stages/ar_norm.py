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
