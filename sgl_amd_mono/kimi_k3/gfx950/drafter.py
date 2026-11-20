# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 DSpark drafter's layer around its attention, as three launches on a
rank of TP 8, for S = ``tokens`` rows (5 a request, up to ``MAX_TOKENS``)::

  D-pre : h    = RMSNorm(res) w_in                  (bf16; res: the residual stream after the
                                                      previous layer, the embedded rows at layer 0)
          qkv  = h W_qkv^T                          (768 rows a rank: 8 q heads, 2 k, 2 v of 64; bf16)
          q, k : per-head RMSNorm (weights) -> bf16 -> neox RoPE from the rotary table -> bf16
          -> ``qkv`` bf16 [S, 768] (q | k | v), the layout the draft attention reads
  D-post: d    = o W_o^T                            (this rank's partial, 7168)
          attn_out = sum over the ranks (bf16 pairs through the peer regions, rank order; bf16)
          res' = bf16(res + attn_out)               (post_attention_layernorm's fused add)
  D-MLP : h    = RMSNorm(res') w_post               (bf16)
          g, u = h W_gu^T (1792 gate rows, 1792 up rows a rank; bf16), a = bf16(silu(g)) u (bf16)
          d    = a W_down^T                          (this rank's partial)
          out  = sum over the ranks (as D-post's), res'' = bf16(res' + out) (the next layer's
                 input_layernorm's fused add)

An all-reduce stage writes the residual stream and each token's sums of squares over
its columns (``ss`` [token][SS_PARTS], fp32), so the next launch's norm reads its
statistics instead of waiting on other CTAs; only layer 0's D-pre exchanges them.

A GEMV over the hidden size splits K in 14 parts of 512 (a task a CTA, fp32 partials
through the tagged mailbox, summed in part order); the down GEMV splits its 1792 in 7
parts of 256. The stage structure is the Muse Spark DFlash2 draft's
(``muse_spark/gfx950/draft_mlp.py``, ``draft_attn_half.py``) at Kimi-K3's shapes,
without the grouped convolutions.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_NT,
    bf2_f32,
    bf16_round,
    butterfly,
    hw_exp2,
    hw_rsq,
    mfma_bf16,
    peer_bases,
    row_sum,
    rsrc,
    sum_partials,
    uniform,
    xshfl,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    kernel_sources,
    key_tuple,
    pair_layout,
)
from sgl_amd_mono.common.sync import FLAG_STRIDE, Mailbox, preg, publish, sreg
from sgl_amd_mono.kimi_k3.gfx950.stamps import stamp, stamp_begin, stamp_flush
from sgl_amd_mono.kimi_k3.gfx950.tag import step_tag

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)

# ---------------------------------------------------------------- the drafter's shapes (a rank of TP 8)
TP = 8
H = 7168  # hidden
I = 1792  # intermediate a rank (14336 / 8)
GU_ROWS = 2 * I  # the merged gate (first) and up rows
HEAD_DIM = 64
Q_HEADS, KV_HEADS = 8, 2
Q_K = Q_HEADS * HEAD_DIM  # 512: q and the attention output
QKV_ROWS = Q_K + 2 * KV_HEADS * HEAD_DIM  # 768: q | k | v
QKV_HEADS = Q_HEADS + 2 * KV_HEADS  # 12 head-rows a token
BLOCK = 5  # rows a request: the anchor and 4 masks
MAX_REQUESTS = 8
MAX_TOKENS = BLOCK * MAX_REQUESTS  # 40
EPS = 1e-5
LOG2E = 1.4426950408889634

ROWS = 16  # an MFMA tile's weight rows
TILE = 16  # an MFMA tile's token rows
K_STEP = 32  # the bf16 MFMA's K
RED_FLOATS = WAVES * 64 * 4  # one tile's MFMA accumulators in LDS
# K over the hidden size: 14 parts of 512, 64 a wave, 2 MFMA steps
K_PARTS, K_PART, K_PER_WAVE = 14, 512, 64
K_STEPS = K_PER_WAVE // K_STEP
# D-MLP gate / up: 32 row groups of 112 x 7 K parts of 1024 (128 a wave, 4 MFMA steps), in rounds of 4 + 3 tiles
G_KPARTS, G_KPART, G_KPW = 7, 1024, 128
G_KSTEPS = G_KPW // K_STEP  # 4
G_RGS, G_ROWS = 32, 112
G_TILES = G_ROWS // ROWS  # 7
G_ROUNDS = (4, 3)
G_TASKS = G_RGS * G_KPARTS  # 224
ACT_CTAS, ACT_DIMS = 224, 8
# down: 224 tasks of 32 rows over the whole K (224 a wave)
DN_ROWS = 32
DN_TILES = DN_ROWS // ROWS
DN_TASKS = H // DN_ROWS  # 224
DN_KW = I // WAVES  # 224
DN_STEPS = DN_KW // K_STEP  # 7
SS_PARTS = BLOCKS  # a token's sums-of-squares slots (an all-reduce CTA's)
# D-pre: qkv (16 row groups of 48 x 14 K parts)
A_RGS, A_ROWS = 16, 48
A_TILES = A_ROWS // ROWS  # 3
A_TASKS = A_RGS * K_PARTS  # 224
# D-post: o (224 tasks of 32 rows, K 512 whole)
O_ROWS = 32
O_TILES = O_ROWS // ROWS
O_TASKS = H // O_ROWS  # 224
O_STEPS = Q_K // WAVES // K_STEP  # 2

assert K_PARTS * K_PART == H and K_PART == WAVES * K_PER_WAVE
assert G_RGS * G_ROWS == GU_ROWS and sum(G_ROUNDS) == G_TILES and G_TASKS <= BLOCKS
assert G_KPARTS * G_KPART == H and G_KPART == WAVES * G_KPW
assert ACT_CTAS * ACT_DIMS == I and ACT_DIMS % 2 == 0
assert DN_TASKS == G_TASKS == ACT_CTAS and DN_STEPS * K_STEP * WAVES == I
assert A_RGS * A_ROWS == QKV_ROWS and A_TASKS <= BLOCKS
assert O_TASKS * O_ROWS == H and O_TASKS <= BLOCKS and O_STEPS * K_STEP * WAVES == Q_K
assert THREADS == 64 * WAVES and HEAD_DIM == 64


def token_tiles(s: int) -> int:
    return -(-s // TILE)


def scratch_bytes(layout: dict) -> int:
    return max(off + nbytes for off, nbytes in layout.values())


def dm_scratch_layout() -> dict:
    """D-MLP's scratch regions (pairs) at MAX_TOKENS: the K parts' sums of squares,
    the gate / up partials (plain fp32) and their tasks' flags, the activations (bf16 [token][I], plain words) and the
    activation CTAs' flags (a 128-byte line each)."""
    return pair_layout(
        [
            ("dm_ss", K_PARTS * MAX_TOKENS),
            ("dm_gpart", G_KPARTS * MAX_TOKENS * GU_ROWS // 2),
            ("dm_gflags", G_TASKS * FLAG_STRIDE),
            ("dm_act", MAX_TOKENS * I // 4),
            ("dm_flags", ACT_CTAS * FLAG_STRIDE),
        ],
        align=256,
    )


def dp_scratch_layout() -> dict:
    """D-pre's scratch regions (pairs): the K parts' sums of squares, the qkv partials."""
    return pair_layout(
        [
            ("dp_ss", K_PARTS * MAX_TOKENS),
            ("dp_apart", K_PARTS * MAX_TOKENS * QKV_ROWS),
        ],
        align=256,
    )


def sym_layout() -> dict:
    """A rank's peer-symmetric region of D-post or D-MLP: [source rank][token][H / 2]
    bf16 pairs."""
    return pair_layout([("ar", TP * MAX_TOKENS * H // 2)], align=256)


def sym_bytes() -> int:
    return scratch_bytes(sym_layout())


# ---------------------------------------------------------------- shared pieces
def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def w_loads(c, w, width, row0, tiles, k0, kpw, ksteps):
    """Weight tiles ``row0 + 16 it + l16`` of a row-major bf16 [*, width] matrix over
    this wave's K steps from ``k0``: ``[tile][step]`` 16 B a lane, in flight."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(w)
    return [
        [
            fx.Vector(
                bo.buffer_load(
                    r_w,
                    ((row0 + it * ROWS + lane % 16) * width + k0 + wave * kpw + kc * K_STEP + 8 * (lane // 16)) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT,
                )
            )
            for kc in range(ksteps)
        ]
        for it in range(tiles)
    ]  # fmt: skip


@flyc.jit
def gemv_tiles(c, w, xb_of, tiles, ksteps, publish):
    """A round's MFMA tiles: ``w[tile][step]`` weights against ``xb_of(t)`` (token
    tile t's B operands a step); the waves' accumulators summed through ``red`` (wave
    order) and ``publish(r, tk, v)`` called for every (row, token) of the round."""
    S, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    for t in range_constexpr(token_tiles(S)):
        xb = xb_of(t)
        for it in range_constexpr(tiles):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(ksteps):
                acc = mfma_bf16(w[it][kc].bitcast(fx.BFloat16), xb[kc], acc)
            fx.ptr_store(acc, red + (it * RED_FLOATS + (wave * 64 + lane) * 4))
        gpu.barrier()
        nt = min(TILE, S - t * TILE)
        n_out = tiles * ROWS * nt
        for p in range_constexpr(-(-n_out // THREADS)):
            idx = tid + p * THREADS
            if idx < n_out:
                r = idx % (tiles * ROWS)
                v = row_sum(red + (r // ROWS) * RED_FLOATS, r % ROWS, idx // (tiles * ROWS))
                publish(r, t * TILE + idx // (tiles * ROWS), v)
        gpu.barrier()  # ``red`` is reused


def lds_b(c, base, row, kpw=K_PER_WAVE, ksteps=K_STEPS):
    """B operands from the LDS bf16 rows ``base`` (``row`` elements a row): token
    tile t's lane token, this wave's ``ksteps`` K steps (``kpw`` a wave) of the part."""
    V8B = fx.Vector.make_type(8, fx.BFloat16)
    S, lane, wave = c["S"], c["lane"], c["wave"]

    def xb_of(t):
        tok = fx.min(t * TILE + lane % 16, S - 1)
        return [
            fx.Vector(
                fx.ptr_load(
                    base + (tok * row + wave * kpw + kc * K_STEP + 8 * (lane // 16)),
                    result_type=V8B,
                )
            )
            for kc in range(ksteps)
        ]  # fmt: skip

    return xb_of


@flyc.jit
def stage_norm_in(c, kp, nparts, prefetch, kw=K_PART):
    """h = RMSNorm(res) w on K part ``kp`` (``kw`` wide) -> ``hbuf`` [token][kw] bf16, the token's
    sum of squares from the producer's ``nparts`` slots of ``ss`` (lane partials in
    slot order, then the wave's butterfly: every CTA the same). ``prefetch()``: the
    next stage's weight loads, issued after this stage's own; returned."""
    S, lane, wave = c["S"], c["lane"], c["wave"]
    hbuf = c["hbuf"]
    a = c["args"]
    chs = [kp * kw + 512 * q + 8 * lane for q in range(kw // 512)]  # this lane's channels, 8 a chunk
    r_x, r_ss = rsrc(a["x"]), rsrc(a["ss"])
    wvs = [fx.Vector(bo.buffer_load(rsrc(a["w_norm"]), ch // 2, vec_width=4, dtype=T.i32)).bitcast(fx.BFloat16)
           for ch in chs]  # fmt: skip
    TPW = -(-S // WAVES)
    rows = []
    for j in range_constexpr(TPW):
        t = fx.min(wave + WAVES * j, S - 1)
        xvs = [fx.Vector(bo.buffer_load(r_x, (t * H + ch) // 2, vec_width=4, dtype=T.i32)).bitcast(fx.BFloat16)
               for ch in chs]  # fmt: skip
        sv = fx.Vector(bo.buffer_load(r_ss, t * SS_PARTS + 4 * fx.min(lane, nparts // 4 - 1), vec_width=4, dtype=T.f32))
        rows.append((t, xvs, sv))
    # the scheduler would sink the loads to their uses: pin the issue order
    rocdl.sched_barrier(0)
    ahead = prefetch()
    rocdl.sched_barrier(0)
    for j in range_constexpr(TPW):
        t, xvs, sv = rows[j]
        part = (sv[0] + sv[1]) + (sv[2] + sv[3])
        tot = butterfly((lane < nparts // 4).select(part, fx.Float32(0.0)), (32, 16, 8, 4, 2, 1))
        f = hw_rsq(tot / float(H) + EPS)
        for q in range_constexpr(kw // 512):
            xv, wv = xvs[q], wvs[q]
            hv = fx.Vector.from_elements([fx.Float32(xv[e]) * f * fx.Float32(wv[e]) for e in range(8)], fx.Float32).to(
                fx.BFloat16
            )
            fx.ptr_store(hv, hbuf + (t * kw + 512 * q + 8 * lane))
    gpu.barrier()
    return ahead


@flyc.jit
def stage_norm(c, kp, owner, prefetch):
    """Layer 0's: h = RMSNorm(x) w on K part ``kp`` (x: the embedded rows; an
    ``owner`` CTA writes them to ``res_out``, the residual stream, and publishes the
    part's sums of squares), every part's sums polled -> ``hbuf`` [token][512] bf16.
    ``prefetch()``: the next stage's weight loads, issued after this stage's own;
    returned."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    put, poll, hbuf, ssl, nf = c["put"], c["poll"], c["hbuf"], c["ssl"], c["nf"]
    a = c["args"]
    ch = kp * K_PART + 8 * lane  # this lane's eight channels
    r_x, r_out = rsrc(a["x"]), rsrc(a["res_out"])
    wv = fx.Vector(bo.buffer_load(rsrc(a["w_norm"]), ch // 2, vec_width=4, dtype=T.i32)).bitcast(fx.BFloat16)
    TPW = -(-S // WAVES)  # tokens a wave (the last ones repeat token S - 1)
    rows = []
    for j in range_constexpr(TPW):
        t = fx.min(wave + WAVES * j, S - 1)
        xv = fx.Vector(bo.buffer_load(r_x, (t * H + ch) // 2, vec_width=4, dtype=T.i32)).bitcast(fx.BFloat16)
        rows.append((t, xv))
    rocdl.sched_barrier(0)
    ahead = prefetch()
    rocdl.sched_barrier(0)
    rp = []
    for j in range_constexpr(TPW):
        t, xv = rows[j]
        r8 = [fx.Float32(xv[e]) for e in range(8)]
        ss = r8[0] * r8[0]
        for e in range_constexpr(1, 8):
            ss = ss + r8[e] * r8[e]
        ss = butterfly(ss, (1, 2, 4, 8, 16, 32))
        if owner:
            bo.buffer_store(
                fx.Vector.from_elements(r8, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32), r_out, (t * H + ch) // 2
            )
            if lane == 0:
                put(c["ss"], kp * MAX_TOKENS + t, ss)
        rp.append(r8)
    N = S * K_PARTS
    for p in range_constexpr(-(-N // THREADS)):
        idx = fx.min(tid + p * THREADS, N - 1)
        got = poll([(c["ss"], (idx % K_PARTS) * MAX_TOKENS + idx // K_PARTS, 1)])
        if tid + p * THREADS < N:
            fx.ptr_store(got[0][0].bitcast(fx.Float32), ssl + idx)
    gpu.barrier()
    if tid < S:
        tot = fx.ptr_load(ssl + tid * K_PARTS)
        for q in range_constexpr(1, K_PARTS):
            tot = tot + fx.ptr_load(ssl + (tid * K_PARTS + q))
        fx.ptr_store(hw_rsq(tot / float(H) + EPS), nf + tid)
    gpu.barrier()
    for j in range_constexpr(TPW):
        t = fx.min(wave + WAVES * j, S - 1)
        f = fx.ptr_load(nf + t)
        hv = fx.Vector.from_elements([rp[j][e] * f * fx.Float32(wv[e]) for e in range(8)], fx.Float32).to(
            fx.BFloat16
        )
        fx.ptr_store(hv, hbuf + (t * K_PART + 8 * lane))
    gpu.barrier()
    return ahead


@flyc.jit
def stage_split_gemv(c, w, rows0, tiles, rounds, kp, part, width, cur, tail=None):
    """Rows ``rows0`` .. + tiles x rounds x 16 of the bf16 [width, H] weights ``w``
    over K part ``kp`` against ``hbuf``, a round's tiles at a time (``cur``: the first
    round's weights, issued by the caller; the next round's in flight while one
    computes); fp32 partials to ``part`` [kp][token][row]. ``tail()``: the next
    stage's loads, issued with the last round's; returned."""
    put = c["put"]
    xb_of = lds_b(c, c["hbuf"], K_PART)
    k0 = kp * K_PART
    extra = None
    for rd in range_constexpr(rounds):
        rocdl.sched_barrier(0)
        if const_expr(rd + 1 < rounds):
            nxt = w_loads(c, w, H, rows0 + (rd + 1) * tiles * ROWS, tiles, k0, K_PER_WAVE, K_STEPS)
        elif const_expr(tail is not None):
            extra = tail()
        rocdl.sched_barrier(0)

        def publish(r, tk, v, rd=rd):
            put(part, (kp * MAX_TOKENS + tk) * width + rows0 + rd * tiles * ROWS + r, v)

        gemv_tiles(c, cur, xb_of, tiles, K_STEPS, publish)
        if const_expr(rd + 1 < rounds):
            cur = nxt
    return extra


@flyc.jit
def stage_gu(c, rg, kp, cur, tail):
    """Gate / up task (``rg``, ``kp``): rows 112 rg .. of W_gu over K part kp (1024)
    against ``hbuf``, in rounds of 4 and 3 tiles (``cur``: the first round's weights;
    the next in flight while one computes; ``tail()``, the down weights, with the
    last); fp32 partials (device scope) to ``gpart`` [kp][token][row], then the
    task's flag. Returns the tail's loads."""
    xb_of = lds_b(c, c["hbuf"], G_KPART, G_KPW, G_KSTEPS)
    r_p = rsrc(c["gpart"].value)
    k0 = kp * G_KPART
    rows0 = rg * G_ROWS
    extra = None
    for rd in range_constexpr(len(G_ROUNDS)):
        t0 = sum(G_ROUNDS[:rd])
        rocdl.sched_barrier(0)
        if const_expr(rd + 1 < len(G_ROUNDS)):
            nxt = w_loads(c, c["args"]["w_gu"], H, rows0 + (t0 + G_ROUNDS[rd]) * ROWS, G_ROUNDS[rd + 1], k0, G_KPW, G_KSTEPS)
        else:
            extra = tail()
        rocdl.sched_barrier(0)

        def emit(r, tk, v, t0=t0):
            bo.buffer_store(v, r_p, (kp * MAX_TOKENS + tk) * GU_ROWS + rows0 + t0 * ROWS + r, cache_modifier=CM_DEV)

        gemv_tiles(c, cur, xb_of, G_ROUNDS[rd], G_KSTEPS, emit)
        if const_expr(rd + 1 < len(G_ROUNDS)):
            cur = nxt
    publish(c["put"], c["gflags"], (rg * G_KPARTS + kp) * FLAG_STRIDE, fx.Int32(1), c["tid"] == 0)
    return extra


@flyc.jit
def stage_ar(c, bid, nrows, vals_of):
    """Rows ``nrows`` bid .. for every token: this rank's values (``vals_of(t, j)``:
    the pair at rows 2j, 2j + 1 of the CTA's) pushed as bf16 pairs to every rank's
    ``ar`` region, the ranks' summed in fp32 in rank order, bf16 (the all-reduce's
    result), added to the residual stream ``res`` -> ``res_out`` (bf16); the sums
    of squares of the CTA's columns -> ``ss_out`` [token][bid]."""
    S, tid, poll, put_bf = c["S"], c["tid"], c["poll"], c["put_bf"]
    ssp = c["ssp"]
    a = c["args"]
    PAIRS = nrows // 2
    N = PAIRS * S
    r_res, r_out = rsrc(a["res"]), rsrc(a["res_out"])
    for p in range_constexpr(-(-N // THREADS)):
        idx = fx.min(tid + p * THREADS, N - 1)
        j, t = idx % PAIRS, idx // PAIRS
        col = bid * nrows + 2 * j
        v0, v1 = vals_of(t, j)
        live = tid + p * THREADS < N
        if live:
            for pe in range_constexpr(TP):
                put_bf(c["peer_ar"][pe], (c["rank"] * MAX_TOKENS + t) * H + col, [v0, v1])

        def pair_at(src):
            return ((src * MAX_TOKENS + t) * H + col) // 2

        d0, d1 = sum_partials(poll, c["own_ar"], pair_at, TP)
        rr = bf2_f32(fx.Int32(bo.buffer_load(r_res, (t * H + col) // 2, vec_width=1, dtype=T.i32)))
        x0 = bf16_round(rr[0] + bf16_round(d0))
        x1 = bf16_round(rr[1] + bf16_round(d1))
        if live:
            bo.buffer_store(
                fx.Vector.from_elements([x0, x1], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0],
                r_out,
                (t * H + col) // 2,
            )
            fx.ptr_store(x0 * x0 + x1 * x1, ssp + idx)
    gpu.barrier()
    if tid < S:
        tot = fx.ptr_load(ssp + tid * PAIRS)
        for q in range_constexpr(1, PAIRS):
            tot = tot + fx.ptr_load(ssp + (tid * PAIRS + q))
        bo.buffer_store(tot, rsrc(a["ss_out"]), tid * SS_PARTS + bid)


# ---------------------------------------------------------------- D-MLP stages
@flyc.jit
def stage_act(c, bid):
    """Dims 8 bid .. of the intermediate for every token: the K parts' gate and up
    partials summed in part order, bf16, a = bf16(silu(g)) u -> ``act`` [token][I]
    bf16 (two dims a thread, device scope), then this CTA's flag."""
    S, tid, poll = c["S"], c["tid"], c["poll"]
    r_g = rsrc(c["gpart"].value)
    r_act = rsrc(c["act"].value)
    # the 14 gate / up tasks this CTA's dims come from: (the gate rows' group, the
    # up rows' group) x 7 K parts, one thread a flag
    d_lo = bid * ACT_DIMS
    if tid < 2 * G_KPARTS:
        grp = (tid < G_KPARTS).select(d_lo // G_ROWS, (I + d_lo) // G_ROWS)
        poll([(c["gflags"], (grp * G_KPARTS + tid % G_KPARTS) * FLAG_STRIDE, 1)])
    gpu.barrier()
    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="workgroup")
    PAIRS = ACT_DIMS // 2
    N = PAIRS * S
    for p in range_constexpr(-(-N // THREADS)):
        idx = fx.min(tid + p * THREADS, N - 1)
        j, t = idx % PAIRS, idx // PAIRS
        d0 = bid * ACT_DIMS + 2 * j

        def pair(off):
            return fx.Vector(bo.buffer_load(r_g, off, vec_width=2, dtype=T.f32, cache_modifier=CM_DEV))

        gs = [pair((kp * MAX_TOKENS + t) * GU_ROWS + d0) for kp in range(G_KPARTS)]
        us = [pair((kp * MAX_TOKENS + t) * GU_ROWS + I + d0) for kp in range(G_KPARTS)]
        g0, g1, u0, u1 = gs[0][0], gs[0][1], us[0][0], us[0][1]
        for kp in range_constexpr(1, G_KPARTS):
            g0 = g0 + gs[kp][0]
            g1 = g1 + gs[kp][1]
            u0 = u0 + us[kp][0]
            u1 = u1 + us[kp][1]
        g0, g1, u0, u1 = bf16_round(g0), bf16_round(g1), bf16_round(u0), bf16_round(u1)
        a0 = bf16_round(g0 * sigmoid(g0)) * u0
        a1 = bf16_round(g1 * sigmoid(g1)) * u1
        if tid + p * THREADS < N:
            bo.buffer_store(
                fx.Vector.from_elements([a0, a1], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0],
                r_act, (t * I + d0) // 2, cache_modifier=CM_DEV,
            )  # fmt: skip
    publish(c["put"], c["flags"], bid * FLAG_STRIDE, fx.Int32(1), tid == 0)


def dn_loads(c, bid):
    """The down task's weights: rows 32 bid .. (2 tiles) x this wave's 7 K steps
    (224 of the 1792), in flight."""
    return w_loads(c, c["args"]["w_down"], I, bid * DN_ROWS, DN_TILES, 0, DN_KW, DN_STEPS)


@flyc.jit
def stage_down_ar(c, bid, w):
    """The down task: every activation CTA's flag, then rows 32 bid .. of W_down
    over the whole K (``w``: its weights, issued earlier) against the activations
    (device-scope loads), the waves' sums in wave order -> the all-reduce. An
    agent-scope acquire here (an L2 invalidate) costs about 12 us a launch."""
    S, tid, lane, wave, red, dbuf = c["S"], c["tid"], c["lane"], c["wave"], c["red"], c["dbuf"]
    if tid < ACT_CTAS:
        c["poll"]([(c["flags"], tid * FLAG_STRIDE, 1)])
    gpu.barrier()
    # orders the loads below after the polls (device-scope loads: L2 holds no stale copy)
    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="workgroup")
    r_act = rsrc(c["act"].value)
    for t in range_constexpr(token_tiles(S)):
        tok = fx.min(t * TILE + lane % 16, S - 1)
        xb = [
            fx.Vector(
                bo.buffer_load(r_act, (tok * I + wave * DN_KW + i * K_STEP + 8 * (lane // 16)) // 2, vec_width=4,
                               dtype=T.i32, cache_modifier=CM_DEV)
            )
            for i in range(DN_STEPS)
        ]  # fmt: skip
        rocdl.sched_barrier(0)
        for it in range_constexpr(DN_TILES):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for i in range_constexpr(DN_STEPS):
                acc = mfma_bf16(w[it][i].bitcast(fx.BFloat16), xb[i].bitcast(fx.BFloat16), acc)
            fx.ptr_store(acc, red + (it * RED_FLOATS + (wave * 64 + lane) * 4))
        gpu.barrier()
        n_out = DN_ROWS * min(TILE, S - t * TILE)
        for p in range_constexpr(-(-n_out // THREADS)):
            idx = tid + p * THREADS
            if idx < n_out:
                r = idx % DN_ROWS
                tk = t * TILE + idx // DN_ROWS
                fx.ptr_store(row_sum(red + (r // ROWS) * RED_FLOATS, r % ROWS, idx // DN_ROWS), dbuf + (tk * DN_ROWS + r))
        gpu.barrier()

    def vals_of(t, j):
        return fx.ptr_load(dbuf + (t * DN_ROWS + 2 * j)), fx.ptr_load(dbuf + (t * DN_ROWS + 2 * j + 1))

    stage_ar(c, bid, DN_ROWS, vals_of)


@flyc.jit
def run_dmlp(c):
    """Every stage's weights are issued before the wait it follows: gate / up's first
    round with the norm's loads, down's (a task's 114 KB) with gate / up's last."""
    bid, on, tls, tid = c["bid"], c["tl_on"], c["tls"], c["tid"]
    a = c["args"]
    rg, kp = bid // G_KPARTS, bid % G_KPARTS
    if bid < G_TASKS:
        gw = stage_norm_in(
            c, kp, O_TASKS, lambda: w_loads(c, a["w_gu"], H, rg * G_ROWS, G_ROUNDS[0], kp * G_KPART, G_KPW, G_KSTEPS),
            kw=G_KPART,
        )  # fmt: skip
        stamp(on, tls, tid, 1)
        dw = stage_gu(c, rg, kp, gw, lambda: dn_loads(c, bid))
        stamp(on, tls, tid, 2)
        stage_act(c, bid)
        stamp(on, tls, tid, 3)
        stage_down_ar(c, bid, dw)
        stamp(on, tls, tid, 4)


# ---------------------------------------------------------------- D-pre stages
@flyc.jit
def stage_heads(c, bid):
    """Head-rows (token, head) 12 S in all, a wave each: the K parts' qkv partials
    summed in part order (a dim a lane), bf16; q and k heads: RMSNorm with the head's
    weight -> bf16 -> neox RoPE at the row's position -> bf16; v as is -> ``qkv``."""
    S, lane, poll = c["S"], c["lane"], c["poll"]
    wave = uniform(c["wave"])  # a job a wave: its head and token are wave-uniform
    apart = c["apart"]
    a = c["args"]
    r_cs, r_out = rsrc(a["cos_sin"]), rsrc(a["qkv"])
    N = QKV_HEADS * S
    per = BLOCKS * WAVES
    for p in range_constexpr(-(-N // per)):
        job = wave * BLOCKS + bid + p * per  # spread over the CTAs
        live = job < N
        jb = fx.min(job, N - 1)
        t, hd = jb // QKV_HEADS, jb % QKV_HEADS  # token, head-row: 0..7 q, 8..9 k, 10..11 v
        row = hd * HEAD_DIM + lane
        got = poll([(apart, (kp * MAX_TOKENS + t) * QKV_ROWS + row, 1) for kp in range(K_PARTS)], batch=K_PARTS)
        v = got[0][0].bitcast(fx.Float32)
        for kp in range_constexpr(1, K_PARTS):
            v = v + got[kp][0].bitcast(fx.Float32)
        v = bf16_round(v)  # the GEMM's bf16 output
        is_v = hd >= Q_HEADS + KV_HEADS
        is_k = (hd >= Q_HEADS) & ~is_v
        ss = butterfly(v * v, (32, 16, 8, 4, 2, 1))
        inv = hw_rsq(ss / float(HEAD_DIM) + EPS)
        r_w = rsrc(is_k.select(a["k_norm"], a["q_norm"]))
        wp = bf2_f32(fx.Int32(bo.buffer_load(r_w, lane // 2, vec_width=1, dtype=T.i32)))
        wn = ((lane % 2) == 0).select(wp[0], wp[1])
        xn = bf16_round(v * inv * wn)
        pw = xshfl(xn, 32)  # the RoPE partner dim (d +- 32)
        pos = fx.Int32(
            bo.buffer_load(rsrc(a["positions"]), (a["pos64"] != 0).select(2 * t, t), vec_width=1, dtype=T.i32)
        )
        half = lane % 32
        cs = fx.Float32(bo.buffer_load(r_cs, pos * HEAD_DIM + half, vec_width=1, dtype=T.f32))
        sn = fx.Float32(bo.buffer_load(r_cs, pos * HEAD_DIM + 32 + half, vec_width=1, dtype=T.f32))
        rot = (lane < 32).select(xn * cs - pw * sn, xn * cs + pw * sn)
        out = is_v.select(v, bf16_round(rot))
        nb = xshfl(out, 1)  # the odd lane's value: even lanes store the pair
        if live & ((lane % 2) == 0):
            bo.buffer_store(
                fx.Vector.from_elements([out, nb], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0],
                r_out,
                (t * QKV_ROWS + row) // 2,
            )


@flyc.jit
def run_dpre(c):
    bid = c["bid"]
    a = c["args"]
    on, tls, tid = c["tl_on"], c["tls"], c["tid"]
    rg, kp = bid // K_PARTS, bid % K_PARTS
    if bid < A_TASKS:
        def qkv_loads():
            return w_loads(c, a["w_qkv"], H, rg * A_ROWS, A_TILES, kp * K_PART, K_PER_WAVE, K_STEPS)

        if const_expr(c["first"]):
            aw = stage_norm(c, kp, rg == 0, qkv_loads)
        else:
            aw = stage_norm_in(c, kp, DN_TASKS, qkv_loads)
        stamp(on, tls, tid, 1)
        stage_split_gemv(c, a["w_qkv"], rg * A_ROWS, A_TILES, 1, kp, c["apart"], QKV_ROWS, aw)
        stamp(on, tls, tid, 2)
    stage_heads(c, bid)
    stamp(on, tls, tid, 3)


# ---------------------------------------------------------------- D-post stages
@flyc.jit
def stage_o(c, bid):
    """The o GEMV task: rows 32 bid .. of W_o over the whole K (512) against the
    attention output (bf16 [S, 512], the previous launch's) -> ``dbuf`` [token][32]."""
    S, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    dbuf = c["dbuf"]
    a = c["args"]
    r_o = rsrc(a["o"])
    w = w_loads(c, a["w_o"], Q_K, bid * O_ROWS, O_TILES, 0, O_STEPS * K_STEP, O_STEPS)
    xbs = []
    for t in range_constexpr(token_tiles(S)):
        tok = fx.min(t * TILE + lane % 16, S - 1)
        xbs.append([
            fx.Vector(
                bo.buffer_load(
                    r_o, (tok * Q_K + wave * O_STEPS * K_STEP + kc * K_STEP + 8 * (lane // 16)) // 2,
                    vec_width=4, dtype=T.i32, cache_modifier=CM_DEV,
                )
            )
            for kc in range(O_STEPS)
        ])  # fmt: skip
    rocdl.sched_barrier(0)
    for t in range_constexpr(token_tiles(S)):
        xb = [v.bitcast(fx.BFloat16) for v in xbs[t]]
        for it in range_constexpr(O_TILES):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(O_STEPS):
                acc = mfma_bf16(w[it][kc].bitcast(fx.BFloat16), xb[kc], acc)
            fx.ptr_store(acc, red + (it * RED_FLOATS + (wave * 64 + lane) * 4))
        gpu.barrier()
        nt = min(TILE, S - t * TILE)
        n_out = O_ROWS * nt
        for p in range_constexpr(-(-n_out // THREADS)):
            idx = tid + p * THREADS
            if idx < n_out:
                r = idx % O_ROWS
                tk = t * TILE + idx // O_ROWS
                fx.ptr_store(row_sum(red + (r // ROWS) * RED_FLOATS, r % ROWS, idx // O_ROWS), dbuf + (tk * O_ROWS + r))
        gpu.barrier()


@flyc.jit
def run_dpost(c):
    bid = c["bid"]
    dbuf = c["dbuf"]
    on, tls, tid = c["tl_on"], c["tls"], c["tid"]
    if bid < O_TASKS:
        stage_o(c, bid)
        stamp(on, tls, tid, 1)

        def vals_of(t, j):
            return fx.ptr_load(dbuf + (t * O_ROWS + 2 * j)), fx.ptr_load(dbuf + (t * O_ROWS + 2 * j + 1))

        stage_ar(c, bid, O_ROWS, vals_of)
    stamp(on, tls, tid, 2)


# ---------------------------------------------------------------- the launches
class DrafterBuild(msgspec.Struct, frozen=True):
    kind: str  # "pre", "post" or "mlp"
    tokens: int
    first: bool = False  # D-pre of layer 0: x is the embedded rows (its norm exchanges its statistics)
    timeline: bool = False  # per-CTA stage stamps into ``tl`` (debugging)


TL_POINTS = {"pre": 4, "post": 3, "mlp": 5}  # start, then each stage's end


def _lds(key: DrafterBuild):
    S = key.tokens

    if key.kind == "post":

        @fx.struct
        class Lds:
            red: fx.Array[fx.Float32, O_TILES * RED_FLOATS, 16]
            dbuf: fx.Array[fx.Float32, S * O_ROWS, 16]
            ssp: fx.Array[fx.Float32, S * O_ROWS // 2, 16]
            tls: fx.Array[fx.Int64, 8, 16]

        return Lds
    tiles = max(G_ROUNDS) if key.kind == "mlp" else A_TILES

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, tiles * RED_FLOATS, 16]
        hbuf: fx.Array[fx.BFloat16, S * (G_KPART if key.kind == "mlp" else K_PART), 16]
        ssl: fx.Array[fx.Float32, S * K_PARTS, 16]
        nf: fx.Array[fx.Float32, S, 16]
        ssp: fx.Array[fx.Float32, S * DN_ROWS // 2, 16]
        dbuf: fx.Array[fx.Float32, S * DN_ROWS, 16]
        tls: fx.Array[fx.Int64, 8, 16]

    return Lds


_ARGS = {
    "pre": ("x", "ss", "res_out", "w_norm", "w_qkv", "q_norm", "k_norm", "cos_sin", "positions", "pos64", "qkv",
            "scratch", "epoch", "layer", "tl"),
    "post": ("o", "w_o", "res", "res_out", "ss_out", "sym", "peers", "rank", "epoch", "layer", "tl"),
    "mlp": ("x", "ss", "w_norm", "w_gu", "w_down", "res", "res_out", "ss_out", "scratch", "sym", "peers", "rank",
            "epoch", "layer", "tl"),
}  # fmt: skip

_BUILDS: dict = {}


def build(key: DrafterBuild):
    """The launcher of one drafter launch for ``key.tokens`` rows; its arguments are
    ``_ARGS[key.kind]`` in order (device pointers as ints, the flags as ints), then the
    stream. ``positions``: int32 / int64 [S] (``pos64``); ``cos_sin`` fp32 [*, 64]."""
    if key in _BUILDS:
        return _BUILDS[key]
    S = key.tokens
    assert key.kind in _ARGS and 1 <= S <= MAX_TOKENS and (key.kind == "pre" or not key.first)
    name = kernel_symbol(f"k3_drafter_{key.kind}", s=S, f=int(key.first), t=int(key.timeline))
    keyed = key_tuple(key, SOURCES)
    Lds = _lds(key)
    layout = {"pre": dp_scratch_layout(), "mlp": dm_scratch_layout(), "post": {}}[key.kind]
    ar_off = sym_layout()["ar"][0]
    kind = key.kind
    points, on = TL_POINTS[kind], key.timeline

    def body(**kw):
        epoch, layer = kw["epoch"], kw["layer"]
        with guard.scope(guard.Guard(epoch)):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            tid = fx.thread_idx.x
            mb = Mailbox(step_tag(epoch, layer))
            c = {
                "S": S, "tid": tid, "bid": fx.block_idx.x, "lane": tid % 64, "wave": tid // 64,
                "put": mb.put, "put_bf": mb.put_bf, "poll": mb.poll, "args": kw, "red": lds.red.ptr,
                "tl_on": on, "tls": lds.tls.ptr, "first": key.first,
            }  # fmt: skip
            stamp_begin(on, c["tls"], tid, points)
            c["ssp"] = lds.ssp.ptr
            c["dbuf"] = lds.dbuf.ptr
            if kind != "post":
                c.update(hbuf=lds.hbuf.ptr, ssl=lds.ssl.ptr, nf=lds.nf.ptr)
            if kind != "pre":
                c["rank"] = kw["rank"]
                c["own_ar"] = preg(kw["sym"], ar_off, "ar")
                c["peer_ar"] = [preg(b, ar_off, "ar") for b in peer_bases(kw["peers"], TP)]
            for region, (off, _) in layout.items():
                c[region[3:]] = sreg(kw["scratch"], off, region)  # "dm_gpart" -> c["gpart"]
            {"pre": run_dpre, "post": run_dpost, "mlp": run_dmlp}[kind](c)
            stamp_flush(on, c["tls"], kw["tl"], tid, c["bid"], points)

    names = _ARGS[kind]
    if kind == "pre":

        @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
        def kern(
            x: Int64, ss: Int64, res_out: Int64, w_norm: Int64, w_qkv: Int64, q_norm: Int64, k_norm: Int64,
            cos_sin: Int64, positions: Int64, pos64: Int32, qkv: Int64, scratch: Int64, epoch: Int64, layer: Int32,
            tl: Int64,
        ):  # fmt: skip
            body(**dict(zip(names, (x, ss, res_out, w_norm, w_qkv, q_norm, k_norm, cos_sin, positions, pos64, qkv,
                                    scratch, epoch, layer, tl))))  # fmt: skip

        @flyc.jit
        def launch(
            x: Int64, ss: Int64, res_out: Int64, w_norm: Int64, w_qkv: Int64, q_norm: Int64, k_norm: Int64,
            cos_sin: Int64, positions: Int64, pos64: Int32, qkv: Int64, scratch: Int64, epoch: Int64, layer: Int32,
            tl: Int64, stream: fx.Stream = _STREAM,
        ):  # fmt: skip
            _ = keyed
            kern(x, ss, res_out, w_norm, w_qkv, q_norm, k_norm, cos_sin, positions, pos64, qkv, scratch, epoch,
                 layer, tl).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    elif kind == "post":

        @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
        def kern(
            o: Int64, w_o: Int64, res: Int64, res_out: Int64, ss_out: Int64, sym: Int64, peers: Int64, rank: Int32,
            epoch: Int64, layer: Int32, tl: Int64,
        ):  # fmt: skip
            body(**dict(zip(names, (o, w_o, res, res_out, ss_out, sym, peers, rank, epoch, layer, tl))))

        @flyc.jit
        def launch(
            o: Int64, w_o: Int64, res: Int64, res_out: Int64, ss_out: Int64, sym: Int64, peers: Int64, rank: Int32,
            epoch: Int64, layer: Int32, tl: Int64, stream: fx.Stream = _STREAM,
        ):  # fmt: skip
            _ = keyed
            kern(o, w_o, res, res_out, ss_out, sym, peers, rank, epoch, layer, tl).launch(
                grid=(BLOCKS,), block=(THREADS,), stream=stream
            )

    else:

        @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
        def kern(
            x: Int64, ss: Int64, w_norm: Int64, w_gu: Int64, w_down: Int64, res: Int64, res_out: Int64,
            ss_out: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32, epoch: Int64, layer: Int32,
            tl: Int64,
        ):  # fmt: skip
            body(**dict(zip(names, (x, ss, w_norm, w_gu, w_down, res, res_out, ss_out, scratch, sym, peers, rank,
                                    epoch, layer, tl))))  # fmt: skip

        @flyc.jit
        def launch(
            x: Int64, ss: Int64, w_norm: Int64, w_gu: Int64, w_down: Int64, res: Int64, res_out: Int64,
            ss_out: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32, epoch: Int64, layer: Int32,
            tl: Int64, stream: fx.Stream = _STREAM,
        ):  # fmt: skip
            _ = keyed
            kern(x, ss, w_norm, w_gu, w_down, res, res_out, ss_out, scratch, sym, peers, rank, epoch, layer,
                 tl).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    _BUILDS[key] = launch
    return launch
