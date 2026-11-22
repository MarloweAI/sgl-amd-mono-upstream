# SPDX-License-Identifier: Apache-2.0
"""``mla_tail``: the tail of one Kimi-K3 MLA layer's DCP8 verify attention at
``MIN_ROWS`` .. ``MAX_ROWS`` rows (6 .. 30 requests), from the split partials
of SGLang's decode kernel to the gated core o_proj takes. One launch a layer,
``BLOCKS`` x ``THREADS``, after aiter's cp round-robin decode
(``mla_decode_fwd``) in place of its split reduce, SGLang's fills, the DCP
all-to-all, the LSE combine, W_UV and the output gate.

The decode leaves, for every rank, all 96 heads over its own share of the keys:

    partials  ``logits`` fp32 [P, 96, 512] and ``attn_lse`` fp32 [P, 96]: a
              split's rows from ``reduce_partial_map[split]`` on
    o, lse    bf16 [S, 96, 512] and fp32 [S, 96]: the rows no reduce tile
              covers, finished by the decode itself

    reduce, every wave
              a (reduce tile, head) unit, tile-major, every 2048th unit a
              wave, up to the first tile at the CSR sentinel: each row of the
              tile from its split partials (8 values a lane, a split's LSE
              and weight a lane), as aiter's ``kn_mla_reduce_v1_ps`` computes
              it -> o, lse; then TRDY a CTA
    W_UV, CTAs 0 .. 191
              a (head, 64 value columns) task, once every CTA's units are in:
              the head's rows 32 at a time (a row with no key on this rank:
              zeros and LSE -1e30, SGLang's fills), the next 32's loads in
              flight, W_UV -> the head's owner rank (``mla_combine``), the
              first half with the LSE
    combine, every CTA
              a (request, local head) task: ``mla_combine``'s LSE combine and
              output gate of the 8 ranks' records -> ``core`` [S, 1536] bf16

The reduce is aiter's: below ``MASSIVE`` splits an online max and sum in
split order, the sum's division; from ``MASSIVE`` on one wave's LSE max and
sum (a butterfly), each split's weight exp(lse - LSE), and the weighted sum in
split order. A row's bits depend on its tile's partials only, as aiter's
do; W_UV, the combine and the gate are a row's own.

Unlike SGLang (combine, then W_UV on the 12 local heads), W_UV is applied to
each head's partial before the exchange: 128-dim records, a quarter of the
bytes. The two orders are equal in exact arithmetic.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import CM_DEV, load_ptr64, read_lane_i32, rsrc, uniform
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, kernel_sources, key_tuple, pair_layout
from sgl_amd_mono.common.sync import Mailbox, publish, sreg
from sgl_amd_mono.kimi_k3.gfx950.attention import mla_combine as mc
from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf
from sgl_amd_mono.kimi_k3.gfx950.stages import gemv
from sgl_amd_mono.kimi_k3.gfx950.tag import step_tag

SOURCES = kernel_sources(__name__)

DCP = mc.DCP
NHA = mf.NHA  # 96 heads every rank attends
NHL = mc.NHL  # 12 a rank owns after the exchange
KV = mf.KV  # 512: the latent output a head
VD = mc.VD  # 128: v_head_dim
GATE = mc.GATE
G_ROWS = mf.G_ROWS
MIN_ROWS = 36  # 6 requests: past the front's widths (FRONT_MAX_ROWS)
MAX_ROWS = 180
RV = 4  # values a thread of aiter's reduce (2 waves a row)
MASSIVE = 4  # aiter's kMassiveThreshold
LANE_SPLITS = 64  # splits a massive tile at most (one LSE a lane of a wave)
CHUNK = 8  # a massive row's partials in flight at once
UV_TASKS = 2 * NHA  # 192: a (head, value half) a CTA
EMPTY_LSE = mf.EMPTY_LSE
INF = float("inf")
NEG_INF = float("-inf")


def peer_offset() -> int:
    """Where the tail's A2A copies start in a rank's peer buffer: past the
    front's QG copies."""
    return -(-mf.peer_end() // 4096) * 4096


def a2a_copy_bytes() -> int:
    return -(-mc.a2a_cells(MAX_ROWS) * 8 // 4096) * 4096


def peer_end() -> int:
    """A rank's peer buffer with the tail's two A2A copies."""
    return peer_offset() + 2 * a2a_copy_bytes()


def init_peer(buf: torch.Tensor) -> None:
    """Arm the tail's A2A copies of a rank's peer buffer (``peer_end`` bytes)."""
    buf[peer_offset() : peer_end()].fill_(0xFF)


@dataclass(frozen=True)
class MlaTailBuild:
    """One build serves every width (S, MIN_ROWS .. MAX_ROWS, is a launch
    argument)."""

    # the A2A copy: the parity of the layer's ordinal among the step's MLA layers
    par: int = 0
    # checks only: the W_UV records also to ``send`` in SGLang's all-to-all
    # layout [8, S, 12, 130]
    records: bool = False


def scratch_layout(key: MlaTailBuild) -> dict:
    return pair_layout([("trdy", BLOCKS)])


def scratch_bytes(key: MlaTailBuild = None) -> int:
    return max(o + n for o, n in scratch_layout(key or MlaTailBuild()).values())


def _ld(ptr, i, cm=0):
    return fx.Int32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32, cache_modifier=cm))


def _ldf(ptr, i, cm=0):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32, cache_modifier=cm))


def _ld4f(base, i):
    v = fx.Vector(bo.buffer_load(rsrc(base), i, vec_width=RV, dtype=T.f32))
    return [v[e] for e in range(RV)]


def _bfly(v, op):
    """``op`` over a wave's 64 lanes, strides 32, 16 .. 1 (aiter's warpReduce)."""
    for off in [32, 16, 8, 4, 2, 1]:
        v = op(v, v.shuffle_xor(fx.Int32(off), fx.Int32(64)))
    return v


def _row_base(c, slot, ls):
    """Split partial row (``slot`` + ``ls``)'s base address: 64-bit, uniform."""
    row = fx.Int64(uniform(slot) + ls)
    return c["args"]["logits"] + row * fx.Int64(NHA * KV * 4)


LN2_HI = float.fromhex("0x1.62e42ep-1")  # with LN2_LO, ln 2 to more than 49 bits
LN2_LO = float.fromhex("0x1.efa39ep-25")


def _logf(x):
    """ln x as aiter's reduce computes it (HIP's logf on gfx950, its last add
    contracted): v_log_f32, then log2 x * ln 2 in extended precision."""
    y = fx.Float32(rocdl.log(T.f32, fx.Float32(x).ir_value()))
    c, cc = fx.Float32(LN2_HI), fx.Float32(LN2_LO)
    r = fmath.fma(y, c, fmath.fma(y, cc, fmath.fma(y, c, -(y * c))))
    return (fmath.absf(y) < fx.Float32(INF)).select(r, y)


def _lse_bad(sm):
    return (sm == fx.Float32(0.0)) | (sm != sm)


def _store_row(c, seq, h, lane, acc):
    ob = fx.Vector.from_elements(acc, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    bo.buffer_store(ob, rsrc(c["args"]["o"]), ((seq * NHA + h) * KV + RV * lane) // 2, cache_modifier=CM_DEV)


def _store_lse(c, seq, h, v):
    bo.buffer_store(fx.Float32(v), rsrc(c["args"]["lse"]), seq * NHA + h, cache_modifier=CM_DEV)


def _lane_f32(v, i):
    return read_lane_i32(fx.Float32(v).bitcast(fx.Int32), i).bitcast(fx.Float32)


def _row_vals(c, pm, ls, h, lane):
    """A split partial row's 8 values of this lane (values 8 lane ..): two
    aiter threads' four each."""
    base = _row_base(c, pm, ls)
    return _ld4f(base, h * KV + RV * (2 * lane)) + _ld4f(base, h * KV + RV * (2 * lane + 1))


def _quad(e):
    """A value's place among an aiter thread's four."""
    return e % RV


@flyc.jit
def _simple(c, h, n, ls, seq, pm_l, lse_l):
    """A row of a tile of n < MASSIVE splits: aiter's online max and sum.
    Lane k holds split k's partial row slot and LSE."""
    lane = c["lane"]
    pm = [read_lane_i32(pm_l, fx.min(fx.Int32(sp), n - 1)) for sp in range(MASSIVE - 1)]
    lses = [_lane_f32(lse_l, fx.min(fx.Int32(sp), n - 1)) for sp in range(MASSIVE - 1)]
    vals = [_row_vals(c, pm[sp], ls, h, lane) for sp in range(MASSIVE - 1)]
    acc = vals[0]
    m = lses[0]
    sm = fx.Float32(1.0)
    for sp in range_constexpr(1, MASSIVE - 1):
        live = sp < n
        nm = fx.max(m, lses[sp])
        old = fmath.exp(m - nm)
        new = fmath.exp(lses[sp] - nm)
        # aiter's build fuses values 0, 1 of a thread's four (a packed FMA)
        # and rounds the two products of values 2, 3
        upd = [
            fmath.fma(acc[e], old, new * vals[sp][e]) if _quad(e) < 2
            else acc[e] * old + vals[sp][e] * new
            for e in range(2 * RV)
        ]  # fmt: skip
        acc = [live.select(upd[e], acc[e]) for e in range(2 * RV)]
        sm = live.select(fmath.fma(sm, old, new), sm)
        m = live.select(nm, m)
    _store_row(c, seq, h, 2 * lane, [acc[e] / sm for e in range(RV)])
    _store_row(c, seq, h, 2 * lane + 1, [acc[e] / sm for e in range(RV, 2 * RV)])
    if lane == 0:
        _store_lse(c, seq, h, _lse_bad(sm).select(fx.Float32(INF), _logf(sm) + m))


@flyc.jit
def _massive(c, h, t0, t1, n, ls, seq, pm_l, lse_l):
    """A row of a tile of n >= MASSIVE splits (t0 .. t1) on one wave: the LSE
    (a split a lane), each split's weight on its lane, then the weighted sum in
    split order."""
    lane = c["lane"]
    mx = _bfly(lse_l, lambda x, y: fx.max(x, y))
    sm = _bfly(fmath.exp(lse_l - mx), lambda x, y: x + y)
    glse = _lse_bad(sm).select(fx.Float32(INF), _logf(sm) + mx)
    if lane == 0:
        _store_lse(c, seq, h, glse)
    sc = fmath.exp(lse_l - glse)
    # the splits aiter's two-split loop takes: pairs from t0 while below
    # t1 - t1 % 2 - 2 (absolute split indices)
    n_pair = fx.max(fx.Int32(0), (t1 - t1 % 2 - 2 - t0 + 1) // 2) * 2
    zero = fx.Float32(0.0)
    init = [zero] * (2 * RV)
    for ck, acc in range(fx.Int32(0), (n + CHUNK - 1) // CHUNK, fx.Int32(1), init=init):
        s0 = fx.Int32(ck) * CHUNK
        sps = [fx.min(s0 + sp, n - 1) for sp in range(CHUNK)]
        pm = [read_lane_i32(pm_l, sps[sp]) for sp in range(CHUNK)]
        vals = [_row_vals(c, pm[sp], ls, h, lane) for sp in range(CHUNK)]
        acc = [fx.Float32(x) for x in acc]
        for sp in range_constexpr(CHUNK):
            live = s0 + sp < n
            wt = _lane_f32(sc, sps[sp])
            # aiter's build: values 0, 1 fused; 2, 3 with the product rounded
            # in its two-split loop, fused in the splits after it
            paired = s0 + sp < n_pair
            upd = [
                fmath.fma(wt, vals[sp][e], acc[e]) if _quad(e) < 2
                else paired.select(acc[e] + wt * vals[sp][e], fmath.fma(wt, vals[sp][e], acc[e]))
                for e in range(2 * RV)
            ]  # fmt: skip
            acc = [live.select(upd[e], acc[e]) for e in range(2 * RV)]
        acc = yield acc
    acc = [fx.Float32(x) for x in acc]
    _store_row(c, seq, h, 2 * lane, acc[:RV])
    _store_row(c, seq, h, 2 * lane + 1, acc[RV:])


@flyc.jit
def stage_reduce(c, bid):
    """The (tile, head) units of CTA bid's waves (unit 96 i + h; wave w of
    the launch's 2048 takes w, w + 2048 .. up to the first tile at the CSR
    sentinel), a row at a time on one wave (8 values a lane); then TRDY."""
    a = c["args"]
    tid, lane = c["tid"], c["lane"]
    nt = c["NT"]
    last = uniform(_ld(a["indptr"], nt))
    nw = BLOCKS * WAVES
    lim = NHA * nt
    u = bid * WAVES + c["wave"]
    while u < lim:
        i = u // NHA
        h = u % NHA
        t0 = uniform(_ld(a["indptr"], i))
        t1 = uniform(_ld(a["indptr"], i + 1))
        q0 = uniform(_ld(a["fmap"], 2 * i))
        q1 = uniform(_ld(a["fmap"], 2 * i + 1))
        n = t1 - t0
        # aiter's planner packs the live tiles first: the first at the
        # sentinel ends the wave's units
        dead = t0 == last
        rows = ((n > 0) & ~dead).select(q1 - q0, fx.Int32(0))
        # lane k: split k's partial row slot (n <= 64)
        pm_l = _ld(a["pmap"], t0 + fx.min(lane, fx.max(n - 1, 0)))
        for rr in range(rows):
            seq = q0 + rr
            lse_l = (lane < n).select(_ldf(a["plse"], (pm_l + rr) * NHA + h), fx.Float32(NEG_INF))
            if n >= MASSIVE:
                _massive(c, h, t0, t1, n, rr, seq, pm_l, lse_l)
            else:
                _simple(c, h, n, rr, seq, pm_l, lse_l)
        u = dead.select(lim, u + nw)
    publish(c["put"], c["trdy"], bid, fx.Int32(1), tid == 0)


def _empty(c, t):
    """Whether row t has no key on this rank (SGLang's ``cprr_empty_rows``)."""
    w = _ld(c["args"]["empty"], t // 4)
    return ((w >> ((t % 4) * 8)) & 0xFF) != 0


UV_ROWS = 2 * gemv.ROWS  # rows a W_UV round: two 16-row GEMV tiles
PIECES = UV_ROWS * (KV // 8) // THREADS  # 16-byte pieces of a round a thread: 4


def _round_loads(c, h, t0):
    """A thread's pieces of rows t0 .. t0 + 31 (clamped to the step) and their
    rows' empty flags, and the LSE and flag of the row its lane pushes (wave
    2 b + w, lane r: row t0 + 16 b + r): all in flight."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    out = []
    for i in range_constexpr(PIECES):
        e = tid + THREADS * i
        t = fx.min(t0 + e // (KV // 8), s - 1)
        v = mf._ld4(a["o"], ((t * NHA + h) * KV + 8 * (e % (KV // 8))) // 2, CM_DEV)
        out += [v[q] for q in range(4)] + [_empty(c, t).select(fx.Int32(1), fx.Int32(0))]
    t = fx.min(t0 + (wave // 2 % 2) * gemv.ROWS + lane % 16, s - 1)
    out += [_ld(a["lse"], t * NHA + h, CM_DEV), _empty(c, t).select(fx.Int32(1), fx.Int32(0))]
    return out


@flyc.jit
def stage_wuv(c, task):
    """W_UV for head task // 2, value columns 64 (task % 2) ..: the head's
    reduced rows (every CTA's TRDY) 32 at a time through LDS (two GEMV tiles a
    round), the next round's loads in flight under each, the W_UV tile already
    there -> the head's owner (``mla_combine``), and ``send`` (records)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    h = task // 2
    half = task % 2
    if tid < BLOCKS:
        c["poll"]([(c["trdy"], tid, 1)])
    mf._bar()
    xl, wl = c["xl"], c["wl"]
    owner = h // NHL
    hl = h % NHL
    vq = wave % 2
    kq = wave // 2
    g, r16 = lane // 16, lane % 16
    # the push: waves 0, 1 the round's first tile, waves 2, 3 its second
    pt, pw = wave // 2, wave % 2
    first = _round_loads(c, h, fx.Int32(0))
    for ri, cur in range(fx.Int32(0), (s + UV_ROWS - 1) // UV_ROWS, fx.Int32(1), init=first):
        r0 = fx.Int32(ri) * UV_ROWS
        cur = [fx.Int32(x) for x in cur]
        nxt = _round_loads(c, h, r0 + UV_ROWS)
        for i in range_constexpr(PIECES):
            e = tid + THREADS * i
            em = cur[5 * i + 4] != 0
            v = fx.Vector.from_elements([em.select(fx.Int32(0), cur[5 * i + q]) for q in range(4)], fx.Int32)
            fx.ptr_store(v, xl + ((e // (KV // 8)) * mf.OL_ROW + 8 * (e % (KV // 8))) // 2)
        mf._bar()
        for b in range_constexpr(2):
            sj = fx.min(fx.Int32(gemv.ROWS), s - r0 - b * gemv.ROWS)
            acc = mf._tgemv(
                lane, wl, mf.UV_ROW, 32 * vq, 128 * kq, 4,
                xl + b * gemv.ROWS * (mf.OL_ROW // 2), mf.OL_ROW, fx.max(sj, 1),
            )  # fmt: skip
            rb = red + ((b * WAVES + wave) * 64 + lane) * 8
            for dt in range_constexpr(2):
                fx.ptr_store(acc[dt], rb + 4 * dt)
        mf._bar()
        t = r0 + pt * gemv.ROWS + r16
        if (wave < 4) & (t < s):
            vals = [None] * 8
            for dt in range_constexpr(2):
                for i in range_constexpr(4):
                    tot = fx.Float32(0.0)
                    for q in range_constexpr(4):
                        tot = tot + fx.ptr_load(red + (((pt * WAVES + 2 * q + pw) * 64 + lane) * 8 + 4 * dt + i))
                    vals[2 * i + dt] = tot
            col = half * mf.UV_HALF + 32 * pw + 8 * g
            mc.push_values(c, owner, hl, t, col, vals)
            if (pw == 0) & (g == 0) & (half == 0):
                lse_w = cur[5 * PIECES]
                em_w = cur[5 * PIECES + 1]
                lse = (em_w != 0).select(fx.Float32(EMPTY_LSE), lse_w.bitcast(fx.Float32))
                mc.push_lse(c, owner, hl, t, lse)
                if const_expr(c["records"]):
                    bo.buffer_store(lse, rsrc(a["send"]), ((owner * s + t) * NHL + hl) * (mf.REC // 2) + VD // 2)
            if const_expr(c["records"]):
                ob = fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
                bo.buffer_store(ob, rsrc(a["send"]), (((owner * s + t) * NHL + hl) * mf.REC + col) // 2)
        _ = yield nxt


ARGS = ("logits", "plse", "indptr", "fmap", "pmap", "o", "lse", "empty", "w_vc", "gate", "core", "send")


def build(key: MlaTailBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    rounds = -(-mc.combine_tasks(MAX_ROWS) // BLOCKS)

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, 2 * WAVES * 64 * 8, 16]
        xl: fx.Array[fx.Int32, UV_ROWS * mf.OL_ROW // 2, 16]
        wl: fx.Array[fx.Int32, KV * mf.UV_ROW // 2, 16]

    keyed = key_tuple(key, SOURCES)
    lay = scratch_layout(key)
    name = f"k3_mono_mla_tail_r{int(key.records)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mla_tail(
        logits: Int64, plse: Int64, indptr: Int64, fmap: Int64, pmap: Int64, o: Int64,
        lse: Int64, empty: Int64, w_vc: Int64, gate: Int64, g_rs: Int32, core: Int64, send: Int64,
        s: Int32, nt: Int32, scratch: Int64, peers: Int64, rank: Int32, layer: Int32,
        epoch: Int64,
    ):  # fmt: skip
        _ = keyed
        with guard.scope(
            guard.Guard(
                epoch, own=load_ptr64(peers, rank), peers=peers, rank=rank, npes=DCP,
                flags_offset=mf._flags_offset(),
            )
        ):  # fmt: skip
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lds = fx.SharedAllocator().allocate(Smem).peek()
            mb = Mailbox(step_tag(epoch, layer) - 1)
            c = {
                "S": s, "NT": nt, "tid": tid, "bid": bid,
                "lane": tid % 64, "wave": tid // 64, "rank": rank,
                "put": mb.put, "poll": mb.poll, "records": key.records,
                "red": lds.red.ptr, "xl": lds.xl.ptr, "wl": lds.wl.ptr,
                "trdy": sreg(scratch, lay["trdy"][0], "trdy"),
                "gbuf": gate, "g_rs": g_rs, "on": False, "tls": None,
                "a2a_peer": mc.PeerRegions(peers, key.par, peer_offset(), a2a_copy_bytes()),
                "args": {
                    "logits": logits, "plse": plse, "indptr": indptr, "fmap": fmap,
                    "pmap": pmap, "o": o, "lse": lse, "empty": empty, "w_vc": w_vc,
                    "core": core, "send": send, "peers": peers,
                },
            }  # fmt: skip
            # the W_UV tile streams into LDS (waves 6, 7) under the reduce
            if bid < UV_TASKS:
                mf._uv_dma(c, bid)
            stage_reduce(c, bid)
            if bid < UV_TASKS:
                stage_wuv(c, bid)
            # the combine tasks from the last CTA down: the W_UV-free CTAs take
            # the second round's
            ntask = (s // G_ROWS) * NHL
            for k in range_constexpr(rounds):
                task = k * BLOCKS + (BLOCKS - 1 - bid)
                if task < ntask:
                    mc.stage_combine(c, task)

    @flyc.jit
    def launch(
        logits: Int64, plse: Int64, indptr: Int64, fmap: Int64, pmap: Int64, o: Int64,
        lse: Int64, empty: Int64, w_vc: Int64, gate: Int64, g_rs: Int32, core: Int64, send: Int64,
        s: Int32, nt: Int32, scratch: Int64, peers: Int64, rank: Int32, layer: Int32,
        epoch: Int64, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        mla_tail(
            logits, plse, indptr, fmap, pmap, o, lse, empty, w_vc, gate, g_rs, core, send,
            s, nt, scratch, peers, rank, layer, epoch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    _BUILDS[key] = launch
    return launch


_BUILDS: dict = {}
_STREAM = fx.Stream(None)


def mla_tail_launch(
    key: MlaTailBuild, *, s, logits, plse, indptr, fmap, pmap, o, lse, empty, w_vc,
    gate, core, scratch, peers, rank, layer, epoch, send=None,
):  # fmt: skip
    """Host launcher. ``logits`` / ``plse``: the decode's split partials (fp32,
    [P, 96, 512] / [P, 96], any leading shape); ``indptr`` / ``fmap`` /
    ``pmap``: SGLang's reduce_indptr / reduce_final_map / reduce_partial_map;
    ``o`` / ``lse``: the decode's output and LSE ([S, 96, 512] bf16 / [S, 96]
    fp32), the reduced rows written into them; ``empty``: SGLang's
    cprr_empty_rows (bool [S]); ``w_vc``: W_UV of all 96 heads (bf16 [96, 512,
    128]); ``gate``: g_proj's rows (bf16 [S, 1536], any row stride); ``core`` [S, 1536] bf16
    out; ``peers``: the DCP ranks' peer buffers (``peer_end`` bytes each);
    ``send`` [8, S, 12, 130] bf16 out, a ``records`` build only. ``s``: the
    step's verify rows."""
    assert s % G_ROWS == 0 and MIN_ROWS <= s <= MAX_ROWS, s
    f32, i32, bf = torch.float32, torch.int32, torch.bfloat16
    assert logits.dtype == f32 and logits.is_contiguous() and logits.shape[-2:] == (NHA, KV)
    assert plse.dtype == f32 and plse.is_contiguous() and plse.numel() * KV == logits.numel()
    nt = indptr.numel() - 1
    assert indptr.dtype == i32 and fmap.dtype == i32 and fmap.numel() >= 2 * nt and pmap.dtype == i32
    assert o.dtype == bf and o.is_contiguous() and tuple(o.shape) == (s, NHA, KV)
    assert lse.dtype == f32 and lse.is_contiguous() and lse.numel() == s * NHA
    assert empty.dtype == torch.bool and empty.is_contiguous() and empty.numel() >= s
    assert tuple(w_vc.shape) == (NHA, KV, VD) and w_vc.dtype == bf and w_vc.is_contiguous()
    assert tuple(gate.shape) == (s, GATE) and gate.dtype == bf and gate.stride(1) == 1 and gate.stride(0) % 8 == 0
    assert tuple(core.shape) == (s, GATE) and core.dtype == bf and core.is_contiguous()
    assert peers.dtype == torch.int64 and peers.numel() == DCP and key.par in (0, 1)
    assert (send is not None) == key.records
    if send is not None:
        assert tuple(send.shape) == (DCP, s, NHL, mf.REC) and send.dtype == bf
    assert scratch.numel() >= scratch_bytes(key)
    build(key)(
        logits.data_ptr(), plse.data_ptr(), indptr.data_ptr(), fmap.data_ptr(),
        pmap.data_ptr(), o.data_ptr(), lse.data_ptr(), empty.data_ptr(), w_vc.data_ptr(),
        gate.data_ptr(), gate.stride(0), core.data_ptr(), mf.ptr(send), s, nt, scratch.data_ptr(),
        peers.data_ptr(), rank, layer, epoch.data_ptr(), stream=torch.cuda.current_stream(),
    )  # fmt: skip
