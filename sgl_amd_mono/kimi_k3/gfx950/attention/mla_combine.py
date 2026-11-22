# SPDX-License-Identifier: Apache-2.0
"""The DCP8 exchange of the MLA front's head partials, their LSE combine and
the output gate, inside the front's launch (``mla_front``).

Every rank attends all 96 heads over its own share of the keys; head h belongs
to rank ``h // 12`` after the exchange. A W_UV task pushes its (row, head)
record to the owner only, in system-scope 8-B cells of the owner's A2A region:

    A2A [source rank][row][local head][REC_CELLS cells]
        cells 0 .. 31    the V-projected partial, four bf16 values a cell
        cell 32          the natural-log LSE (fp32 word) and a zero word
        cell 33          padding (never written)

Every 4-B word is its own flag: an unwritten word holds SENT (all ones), which
no writer stores (``sent_word``), so a cell is read once both of its words have
landed. The owner's reader stores SENT back after reading. The region has two
copies, alternating by the layer's MLA ordinal: the KDA layers in between
each run an all-reduce, so no rank pushes into a copy its owner still reads.

The owner's combine task, a (request, local head), polls the 8 sources' cells
of its rows (one thread a (row, 4 values)), and writes ``core`` [S, 1536] bf16,
row-major (row t, column local head x 128 + v), o_proj's input:

    M = max over sources of the LSE (NaN or +inf as -inf; -inf as 0)
    w_i = exp(lse_i - M), summed in source order; acc = sum of p_i w_i (each
    product rounded, then added), in source order; x = bf16(acc / max(sum w, 1))
    core = bf16(x * bf16(sigmoid(g)))

which is SGLang's ``_dcp_lse_combine_kernel`` and ``mla_output_gate_kernel``
bit for bit. g is this rank's g_proj row from the front's stage 1 (GRDY).

The record layout is the same at every width: a row's cells depend only on its
index and the build's row count.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import CM_DEV, CM_SYS, bf16_pair, bf_hi, bf_lo, rsrc, sfresh64, sload64
from sgl_amd_mono.common.sync import SENT, Mailbox, sent_word
from sgl_amd_mono.kimi_k3.gfx950.stamps import stamp

DCP = 8
NHL = 12  # local heads after the exchange
VD = 128  # v_head_dim
GATE = NHL * VD  # 1536: core's and g_proj's columns a rank
CELL_VALUES = 4
VAL_CELLS = VD // CELL_VALUES  # 32
LSE_CELL = VAL_CELLS  # 32
REC_CELLS = VAL_CELLS + 2  # 34: a record stays 16-B aligned
REQ_ROWS = 6  # a request's verify rows
MAX_A2A_ROWS = 48  # the widest build's rows: one copy of the region fits it
GATE_GROUPS = GATE // 16  # 96 g_proj groups of 16 rows
LOG2E = 1.4426950408889634
NEG_INF = float("-inf")


def a2a_cells(s: int) -> int:
    """One copy of a rank's A2A region at s rows: its 8-B cells."""
    return DCP * s * NHL * REC_CELLS


def a2a_copy_bytes() -> int:
    return -(-a2a_cells(MAX_A2A_ROWS) * 8 // 4096) * 4096


def peer_offset() -> int:
    """Where the A2A region starts in a rank's peer buffer: past K2's (the
    room the front's q all-gather leaves before its QG copies)."""
    from sgl_amd_mono.kimi_k3.gfx950 import layer

    return -(-layer.peer_alloc_bytes() // 4096) * 4096


def peer_end() -> int:
    """A rank's peer buffer with the A2A region's two copies."""
    return peer_offset() + 2 * a2a_copy_bytes()


def init_peer(buf: torch.Tensor) -> None:
    """Arm the A2A region of a rank's peer buffer (``peer_end`` bytes)."""
    buf[peer_offset() : peer_end()].fill_(0xFF)


def a2a_cell(src, t, hl, u, s):
    """Cell u of the record (source rank ``src``, row t of s, local head hl)."""
    return ((src * s + t) * NHL + hl) * REC_CELLS + u


def combine_tasks(s: int) -> int:
    """(request, local head) tasks at s rows."""
    return (s // REQ_ROWS) * NHL


class PeerRegions:
    """Every rank's A2A copy of MLA parity ``par``, indexed by rank: each base is
    loaded (a scalar load of the ``peers`` table) where it is used, so no rank's
    base stays in SGPRs across the launch."""

    def __init__(self, peers, par, offset=None, copy_bytes=None):
        self.peers = peers
        offset = peer_offset() if offset is None else offset
        copy_bytes = a2a_copy_bytes() if copy_bytes is None else copy_bytes
        self.off = fx.Int64(offset) + fx.Int64(par) * fx.Int64(copy_bytes)

    def __getitem__(self, rank):
        return sload64(sfresh64(self.peers) + fx.Int64(rank) * fx.Int64(8)) + self.off


# ---------------------------------------------------------------- the push


def _cell_words(vals):
    """bf16 values (an even count) -> their words, none of them SENT."""
    return [
        sent_word(bf16_pair(vals[2 * k], vals[2 * k + 1]))
        for k in range(len(vals) // 2)
    ]


@flyc.jit
def push_values(c, owner, hl, t, col, vals):
    """Eight bf16 values of (row t, head 12 owner + hl) from value column ``col``
    (a multiple of 8): two cells of the owner's region, this rank's records."""
    words = fx.Vector.from_elements(_cell_words(vals), fx.Int32)
    peer = c["a2a_peer"]
    at = a2a_cell(c["rank"], t, hl, col // CELL_VALUES, c["S"])
    for p in range_constexpr(DCP):
        if owner == p:
            put_cells(peer[p], at, words)


@flyc.jit
def push_lse(c, owner, hl, t, lse):
    """The (row t, head 12 owner + hl) natural-log LSE: the record's LSE cell."""
    words = fx.Vector.from_elements([sent_word(lse), fx.Int32(0)], fx.Int32)
    peer = c["a2a_peer"]
    at = a2a_cell(c["rank"], t, hl, LSE_CELL, c["S"])
    for p in range_constexpr(DCP):
        if owner == p:
            put_cells(peer[p], at, words)


def put_cells(base, cell, words):
    """Cells from ``cell`` := ``words`` (two a cell), one system-scope store."""
    bo.buffer_store(words, rsrc(base), cell * 2, cache_modifier=CM_SYS)


# ---------------------------------------------------------------- the owner


@flyc.jit
def gate_ready(c, g):
    """Stage-1 group g's stores have landed (``stage1`` drains them): a g_proj
    group publishes GRDY."""
    if (g >= c["gate0"]) & (c["tid"] == 0):
        c["put"](c["grdy"], g - c["gate0"], fx.Int32(1))


def _sigmoid_bf16(g):
    """bf16(1 / (1 + e^-g)) in fp32, as ``mla_output_gate_kernel``."""
    s = fx.Float32(1.0) / (fx.Float32(1.0) + fmath.exp(-g))
    return fx.Float32(s.to(fx.BFloat16))


def _lse_of(w):
    """A received LSE word as the combine reads it: NaN or +inf -> -inf."""
    mag = w & fx.Int32(0x7FFFFFFF)
    bad = (mag > fx.Int32(0x7F800000)) | (w == fx.Int32(0x7F800000))
    return bad.select(fx.Float32(NEG_INF), w.bitcast(fx.Float32))


def combine4(cells, lse_words, gate_words):
    """One (row, 4 values) of the combine and gate: ``cells`` the 8 sources'
    (w0, w1) value words in source order, ``lse_words`` their LSE words,
    ``gate_words`` the 4 g values' bf16 pairs -> the gated core's two words."""
    ls = [_lse_of(w) for w in lse_words]
    m = ls[0]
    for src in range(1, DCP):
        m = (ls[src] > m).select(ls[src], m)
    m = (m == fx.Float32(NEG_INF)).select(fx.Float32(0.0), m)
    wsum = fx.Float32(0.0)
    acc = [fx.Float32(0.0)] * CELL_VALUES
    for src in range(DCP):
        wt = fmath.exp2((ls[src] - m) * fx.Float32(LOG2E))
        wsum = wsum + wt
        w0, w1 = cells[src]
        vals = [bf_lo(w0), bf_hi(w0), bf_lo(w1), bf_hi(w1)]
        acc = [acc[e] + vals[e] * wt for e in range(CELL_VALUES)]
    den = fx.max(wsum, fx.Float32(1.0))
    g0, g1 = fx.Int32(gate_words[0]), fx.Int32(gate_words[1])
    gs = [bf_lo(g0), bf_hi(g0), bf_lo(g1), bf_hi(g1)]
    outs = []
    for e in range(CELL_VALUES):
        x = fx.Float32((acc[e] / den).to(fx.BFloat16))
        outs.append(x * _sigmoid_bf16(gs[e]))
    return fx.Vector.from_elements(
        [bf16_pair(outs[0], outs[1]).bitcast(fx.Int32), bf16_pair(outs[2], outs[3]).bitcast(fx.Int32)],
        fx.Int32,
    )


def _spin_cells(own, cells):
    """Poll this rank's cells (system scope) until none has a SENT word: their
    words, a cell after another."""
    from sgl_amd_mono.common.sync import _spin

    def load_all():
        ws = []
        for at in cells:
            v = fx.Vector(
                bo.buffer_load(rsrc(own), at * 2, vec_width=2, dtype=T.i32, cache_modifier=CM_SYS)
            )
            ws += [v[0], v[1]]
        return fx.Vector.from_elements(ws, fx.Int32)

    def pending(v):
        bad = v[0] == fx.Int32(SENT)
        for e in range_constexpr(1, 2 * len(cells)):
            bad = bad | (v[e] == fx.Int32(SENT))
        return bad

    v = _spin(load_all, pending)
    return [(v[2 * k], v[2 * k + 1]) for k in range(len(cells))]


def _rearm(own, cells):
    empty = fx.Vector.from_elements([fx.Int32(SENT), fx.Int32(SENT)], fx.Int32)
    for at in cells:
        bo.buffer_store(empty, rsrc(own), at * 2, cache_modifier=CM_SYS)


@flyc.jit
def stage_combine(c, task):
    """Task (request task // 12, local head task % 12): the 8 sources' records of
    the request's rows -> gated ``core`` rows. A thread a (row, source) polls the
    LSE cells first (into LDS); then a thread a (row t, values 4 u ..) polls its
    8 value cells, so the wait polls 48 words, not every record."""
    s, tid = c["S"], c["tid"]
    a = c["args"]
    red = c["red"]
    # read before the traced branches (a dict lookup inside one is carried state)
    box = c.get("core_box")
    g_rs = c.get("g_rs", GATE)  # g's row stride (bf16)
    pre_wait = c.get("pre_wait")
    # the launch's stamp points (g seen, records seen, core stored), if any
    tlp = c.get("comb_tl")
    b = task // NHL
    hl = task % NHL
    gpr = VD // 16  # g_proj groups a local head
    # the head's g rows, from this rank's stage 1 (none: g given at launch)
    if const_expr(c.get("grdy") is not None):
        if tid < gpr:
            c["poll"]([(c["grdy"], hl * gpr + tid, 1)])
        gpu.barrier()
    if const_expr(tlp is not None):
        stamp(c["on"], c["tls"], tid, tlp[0])
    # the launch's own work for the CTA's idle waves (3 ..) during the wait
    if const_expr(pre_wait is not None):
        pre_wait(c)
    own = c["a2a_peer"][c["rank"]]
    lt = b * REQ_ROWS + fx.min(tid // DCP, REQ_ROWS - 1)
    lcell = a2a_cell(tid % DCP, lt, hl, LSE_CELL, s)
    if tid < REQ_ROWS * DCP:
        lw = _spin_cells(own, [lcell])
        _rearm(own, [lcell])
        fx.ptr_store(fx.Int32(lw[0][0]).bitcast(fx.Float32), red + tid)
    gpu.barrier()
    live = tid < REQ_ROWS * VAL_CELLS
    tr = fx.min(tid // VAL_CELLS, REQ_ROWS - 1)
    t = b * REQ_ROWS + tr
    u = tid % VAL_CELLS
    col = hl * VD + CELL_VALUES * u
    gw = fx.Vector(
        bo.buffer_load(rsrc(c["gbuf"]), (t * g_rs + col) // 2, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV)
    )
    vcells = [a2a_cell(src, t, hl, u, s) for src in range(DCP)]
    if live:
        got = _spin_cells(own, vcells)
        if const_expr(tlp is not None):
            stamp(c["on"], c["tls"], tid, tlp[1])
        _rearm(own, vcells)
        lws = [fx.Int32(fx.ptr_load(red + (tr * DCP + src)).bitcast(fx.Int32)) for src in range(DCP)]
        ow = combine4(got, lws, gw)
        bo.buffer_store(
            ow, rsrc(a["core"]), (t * GATE + col) // 2,
            cache_modifier=0 if box is None else CM_DEV,
        )  # fmt: skip
    if const_expr(box is not None):
        # every wave drains its core stores before the box (gpu.barrier waits only lgkmcnt)
        rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if const_expr(tlp is not None):
        stamp(c["on"], c["tls"], tid, tlp[2])
    # o_proj in the same launch: the (request, local head) box, after the stores
    if const_expr(box is not None):
        if tid == 0:
            Mailbox(box[1]).put(box[0], task, fx.Int32(1))
