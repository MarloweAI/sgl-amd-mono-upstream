# SPDX-License-Identifier: Apache-2.0
"""An Engram layer's projection inside the reuse layers' launches before it (``MonoBuild.eg``).

SGLang's ``Engram.forward`` projects the step's retrieved n-gram rows with ``wkv``
(replicated FP8 [25600, 6144], 32 x 32 E8M0 blocks): ``kv = wkv(emb.flatten(-2))``, which
the hidden state does not enter. The reuse layers' launches before the Engram layer (since
the Engram layer before it) compute it where their CTAs idle: once the attention back's
weights are in flight, on the CTAs without attention or wo_a units, until wo_a's wait. Each
rank takes 1 / tp of wkv's rows with all of K (an N shard), each launch a range of them:

- a CTA quantizes the rows' token tile into LDS as the MXFP8 linear quantizes its input
  (a ue8m0 code a 32-group, ``vllm_mx_code``), then runs its 16-row tasks (``gemv_fp8``,
  K in the GEMV's step order, the waves summed in order), each K step's registers taking
  the next task's weights once its MFMA has them, and rounds each output to bf16;
- each task's outputs go to every rank's ENG region (tagged bf16 pairs, at the columns of
  this rank's shard), so every rank ends with all of kv;
- at the launch's end the same CTAs copy its range's ENG pairs into ``eg_kv`` [rows,
  25600] for the Engram layer's gate (``eg_row0`` on, ``MonoBuild.eg`` 16-row tasks).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from sgl_amd_mono.common.mx import clamp_fp8
from sgl_amd_mono.common.ops import bf16_round, butterfly, fp8_pack4, row_sum, rsrc
from sgl_amd_mono.common.plan import BLOCKS, THREADS, first_task, kernel_sources
from sgl_amd_mono.common.stamps import stamp
from sgl_amd_mono.common.sync import preg
from sgl_amd_mono.deepseek_v41.gfx950.stages.gemv import (
    ROWS,
    _fp8_step_load,
    _fp8_step_mfma,
    _gemv_steps,
    lds_row,
    token_tiles,
)
from sgl_amd_mono.deepseek_v41.gfx950.stages.moe import vllm_mx_code
from sgl_amd_mono.deepseek_v41.gfx950.stages.seam import HC, HIDDEN

SOURCES = kernel_sources(__name__)

K = 6144  # the n-gram rows' width: hash columns x head_dim
N = (HC + 1) * HIDDEN  # kv: hc_mult keys, then the value
GROUPS = K // 32
G8 = K // 8  # a row's 8-element chunks, a thread's quant unit


def rank_rows(tp: int) -> int:
    """A rank's wkv rows (its N shard)."""
    assert N % (tp * 32) == 0, tp
    return N // tp


def eg_lds(rows: int):
    """The projection's LDS: the token tile's X8 rows and codes, the waves' partials,
    a task's bf16 outputs."""

    @fx.struct
    class EgLds:
        xl: fx.Array[fx.Int32, rows * lds_row(K // 4), 16]
        xsl: fx.Array[fx.Int32, rows * lds_row(GROUPS), 16]
        red: fx.Array[fx.Float32, 64 * 4 * 8, 16]
        out: fx.Array[fx.Int32, ROWS * rows, 16]

    return EgLds


def eg_plan(woa0: int, woa_units: int, n_split: int, n_comb: int) -> tuple[int, int]:
    """(first CTA, CTAs): from the one past the wo_a units, the CTAs without attention
    units where they are a quarter of the grid or more; else with the split's, then the
    combine's too; every CTA where wo_a's units leave under a quarter."""
    free = BLOCKS - woa_units
    if free < BLOCKS // 4:
        return 0, BLOCKS
    first = (woa0 + woa_units) % BLOCKS
    for ctas in (free - n_comb - n_split, free - n_comb):
        if ctas >= BLOCKS // 4:
            return first, ctas
    return first, free


X_BATCH = 6  # a thread's x chunks in flight at a time
# a wave's K steps of the next task whose weights load during a task (the rest as it starts)
AHEAD = 5


def x_loads(c, x, t0, n, k0):
    """This thread's 8-element chunks k0 .. of rows t0 .. t0 + n of ``x`` (bf16 [rows, K]),
    up to X_BATCH, issued together (past the tile's end: its last chunk again, unused)."""
    items = n * G8
    out = []
    for k in range(k0, min(k0 + X_BATCH, -(-items // THREADS))):
        i = fx.min(c["tid"] + THREADS * k, items - 1)
        at = (t0 + i // G8) * (K // 2) + i % G8 * 4
        out.append(fx.Vector(bo.buffer_load(rsrc(x), at, vec_width=4, dtype=T.i32)))
    return out


@flyc.jit
def quant_tile(c, x, t0, n, xl, xsl):
    """Rows t0 .. t0 + n of ``x`` as MXFP8 into LDS, as the MXFP8 linear quantizes its
    input: a 32-group's code from its 4 threads' maxima (``vllm_mx_code``)."""
    for k0 in range_constexpr(0, -(-n * G8 // THREADS), X_BATCH):
        _quant_chunks(c, x_loads(c, x, t0, n, k0), k0, n, xl, xsl)
    gpu.barrier()


@flyc.jit
def _quant_chunks(c, xs, k0, n, xl, xsl):
    tid = c["tid"]
    x_row, s_row = lds_row(K // 4), lds_row(GROUPS)
    for k in range_constexpr(len(xs)):
        w = xs[k]
        i = tid + THREADS * (k0 + k)
        row, g8 = i // G8, i % G8
        ys = []
        for d in range_constexpr(4):
            hi = w[d] & fx.Int32(-65536)
            ys += [(w[d] << 16).bitcast(fx.Float32), hi.bitcast(fx.Float32)]
        amax = abs(ys[0])
        for y in ys[1:]:
            amax = fx.max(amax, abs(y))
        code = vllm_mx_code(butterfly(amax, (1, 2), fx.max))
        mul = ((254 - code) << 23).bitcast(fx.Float32)
        w0 = fp8_pack4(*[clamp_fp8(ys[j] * mul) for j in range(4)])
        w1 = fp8_pack4(*[clamp_fp8(ys[4 + j] * mul) for j in range(4)])
        if i < n * G8:
            fx.ptr_store(w0, xl + (row * x_row + g8 * 2))
            fx.ptr_store(w1, xl + (row * x_row + g8 * 2 + 1))
            if g8 % 4 == 0:
                fx.ptr_store(code, xsl + (row * s_row + g8 // 4))


def tile_order(w):
    """A rank's wkv rows [n, K] in the GEMV's tile order (``gemv_fp8_loads``): 1 KB a
    (16-row group, 64-byte K chunk), lane l's 16 bytes row l % 16's at K byte 16 (l / 16),
    so a wave's loads are contiguous."""
    n = w.shape[0]
    t = w.view(n // ROWS, ROWS, K // 64, 4, 16).permute(0, 2, 3, 1, 4)
    return t.reshape(n, K).contiguous()


def _step_load(c, e, task, st):
    """Task ``task``'s weights of K step st, as ``gemv_fp8_loads`` (past the last task: the
    last one's, unused)."""
    return _fp8_step_load(
        c, rsrc(e["w"]), rsrc(e["ws"]), K, fx.min(task, e["tasks"] - 1), st
    )


def task_loads(c, e, task):
    """Task ``task``'s weight loads, a K step a wave's GEMV step."""
    return [_step_load(c, e, task, st) for st, _ in _gemv_steps(c, K, None)]


@flyc.jit
def stage_task(c, e, task, ops, t0, n, nxt=None):
    """Task ``task``'s 16 wkv rows (its weights ``ops``) on the tile's n tokens -> every
    rank's ENG pairs; nothing past the last task. With ``nxt``, each K step's weights of
    task ``nxt`` load into its step's registers once its MFMA has them, so one task's
    weights are in flight at a time: -> task ``nxt``'s first AHEAD steps' weights (``ops``
    may hold just those: the rest load first)."""
    tid, tp = c["tid"], e["tp"]
    live = task < e["tasks"]
    steps = _gemv_steps(c, K, None)
    # the steps the task before did not load ahead
    ops = list(ops) + [_step_load(c, e, task, st) for st, _ in steps[len(ops) :]]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    nxt_ops = []
    for i, (st, _) in enumerate(steps):
        acc = _fp8_step_mfma(c, K, e["xl"], e["xsl"], st, ops[i], acc, rows=n)
        if const_expr(nxt is not None and i < AHEAD):
            nxt_ops.append(_step_load(c, e, nxt, st))
    fx.ptr_store(acc, e["red"] + (c["wave"] * 64 + c["lane"]) * 4)
    gpu.barrier()
    if tid < ROWS * n:
        v = bf16_round(row_sum(e["red"], tid % ROWS, tid // ROWS))
        fx.ptr_store(v.bitcast(fx.Int32), e["out"] + tid)
    gpu.barrier()
    if live & (tid < ROWS // 2 * n):
        rp, tl = tid % (ROWS // 2), tid // (ROWS // 2)
        v0 = fx.ptr_load(e["out"] + (tl * ROWS + 2 * rp)).bitcast(fx.Float32)
        v1 = fx.ptr_load(e["out"] + (tl * ROWS + 2 * rp + 1)).bitcast(fx.Float32)
        t = t0 + tl
        for p in range_constexpr(tp):
            col = e["rank"] * rank_rows(tp) + e["row0"] + task * ROWS + 2 * rp
            e["put_bf"](e["dst"][p], t * N + col, [v0, v1])
    gpu.barrier()
    return nxt_ops


@flyc.jit
def run_eg(c, e, s):
    """This CTA's share of the projection (``eg_plan``'s CTAs), tile by tile: task k
    of CTA r is r + k CTAs, the next task's weights in flight through each."""
    rank = first_task(c["bid"], e["first"])
    mine = rank < e["ctas"]
    if const_expr(not isinstance(e["tasks"], int)):
        # a multi-layer launch's layer: its own task count, none where it projects nothing
        mine = mine & (e["tasks"] > 0)
    if mine:
        stamp(c, 41)
        tiles = token_tiles(s)
        for k in range_constexpr(len(tiles)):
            _eg_tile(c, e, rank, tiles[k][0], tiles[k][1])
        stamp(c, 42)


@flyc.jit
def _eg_tile(c, e, rank, t0, n):
    per = -(-e["most"] // e["ctas"])
    # the tile quantized before any weights load, so none are live under the quant
    quant_tile(c, e["x"], t0, n, e["xl"], e["xsl"])
    # as a later task's: its first AHEAD steps (the rest load as it starts)
    ops = task_loads(c, e, rank)[:AHEAD]
    if const_expr(per > 1):
        # a loop, not unrolled: its live registers stay one task's weights however many it
        # runs; the last task apart, loading none after it
        for k, state in range(0, fx.Int64(per - 1), 1, init=_flat(ops)):
            task = rank + fx.Int32(k) * e["ctas"]
            got = stage_task(c, e, task, _steps(state), t0, n, task + e["ctas"])
            carried = yield _flat(got)
        ops = _steps(carried)
    stage_task(c, e, rank + (per - 1) * e["ctas"], ops, t0, n)


def _flat(ops):
    """A task's step operands (lo, hi, scale) as a flat list, a loop's state."""
    return [v for op in ops for v in op]


def _steps(flat):
    return [tuple(flat[i : i + 3]) for i in range(0, len(flat), 3)]


@flyc.jit
def unpack(c, e, s):
    """At the launch's end: the ENG pairs of the launch's rows of every rank's shard ->
    ``eg_kv``, a share a CTA of ``eg_plan``'s."""
    rank = first_task(c["bid"], e["first"])
    tp, half = e["tp"], e["tasks"] * ROWS // 2  # a rank's pairs a token
    if rank < e["ctas"]:
        for i in range(rank * THREADS + c["tid"], s * tp * half, e["ctas"] * THREADS):
            t, r = i // (tp * half), i // half % tp
            at = (t * N + r * rank_rows(tp) + e["row0"]) // 2 + i % half
            word = e["poll"]([(e["own"], at, 1)])[0][0]
            bo.buffer_store(word, rsrc(e["kv"]), at)


def eg_context(args, tasks, bases, own, off, tp, rank, mb, lds, most=None):
    """``run_eg`` / ``unpack``'s context: the arguments, the launch's tasks (a multi-layer
    launch: the layer's, traced, at most ``most``), ENG on every rank, the LDS view."""
    return {
        "x": args["eg_x"],
        "w": args["eg_w"],
        "ws": args["eg_ws"],
        "kv": args["eg_kv"],
        "row0": args["eg_row0"],
        "tasks": tasks,
        # the most tasks a launch's layer runs (``tasks`` may be traced, at most this)
        "most": tasks if most is None else most,
        "dst": [preg(b, off, "eng") for b in bases],
        "own": preg(own, off, "eng"),
        "tp": tp,
        "rank": rank,
        "put_bf": mb.put_bf,
        "poll": mb.poll,
        "xl": lds.xl.ptr,
        "xsl": lds.xsl.ptr,
        "red": lds.red.ptr,
        "out": lds.out.ptr,
    }
