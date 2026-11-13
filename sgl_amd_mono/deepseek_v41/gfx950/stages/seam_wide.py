# SPDX-License-Identifier: Apache-2.0
"""The mHC seam at wide row counts (60 .. 384 rows) as one launch, bit for bit SGLang's
ROCm seam pair: ``hc_boundary_fused_deferred`` (the Triton ``_hc_boundary_partial_kernel``)
and the norm launch that hosts its reduce + sinkhorn (``rmsnorm_with_sinkhorn``).

A seam of ``m`` rows, from the sublayer output x [m, H] bf16 (the owed post's), the
residual streams [m, 4, H] bf16 and the previous boundary's post [m, 4], comb [m, 4, 4]
and pre [m, 4] (f32), writes:
- the new streams R = hc_post(x, residual, post, comb) [m, 4, H] bf16 (``Form.post``);
- the layer input y = sum_k pre[k] R_k (``Form.combine``; else stream 0), normed with the
  layer's RMSNorm weight to bf16, and its fp8 quant: e4m3 + one ue8m0 byte a 32 columns
  (``quant`` "mxfp8"), the same rounded back to bf16 ("grid"), or none;
- this boundary's mixes of R: pre, post [m, 4] and comb [m, 4, 4] (f32).

The launch has two kinds of CTAs, 256 threads each. A slice CTA owns 64 hidden columns
(one of 80 slices) of 64 rows (32 at 96 and 192 rows, where two waves split a 16-row
block's two MFMA tiles). It loads x's and the streams' tiles as whole 128-byte rows
through LDS into the layout of the Triton kernel's tile, computes R, the collapse and
the row sums of squares in that kernel's order (as the gfx950 prefill kernel
``mhc_boundary_gfx95.cuh`` replays it) and the 24 mix partials as one chain of
``mfma_f32_16x16x4f32`` over the slice's 256 columns, A = the mix weights (an LDS table),
B = R, in column order from zero. It writes y and the partials to device-scope scratch
and flips its flag. A row CTA takes a row: once the 80 slice CTAs of the row's group have
flipped, wave 0 sums the partials over the slices in the reduction tree of SGLang's
reduce + sinkhorn row (``_hc_mix_reduce_sinkhorn_row``: its DPP and LDS steps at 4 warps,
read off the compiled kernel) and runs the gates and the sinkhorn rounds as that
kernel's code does (its contracted multiply-adds, IEEE divisions, the hardware square
root of ``tl.sqrt`` and the range-reduced exp2 of ``tl.exp``); waves 1 .. 3 normalize y as
``rmsnorm_fake_quant_row`` does at H = 5120 (its threads' columns and sum tree, the warps'
sums meeting in LDS) and quantize.

Row CTAs follow every slice CTA in the grid, so a slice CTA never waits and the waits
always finish. A flag is a phase bit, flipped once by its owner in every launch (zeros at
the start): a waiter compares the slice flags with the flipped value of its own flag, so
no reset is needed. Each width has its own flag words.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.math import fma
from flydsl.expr.typing import Int64, T, as_ir_value

from sgl_amd_mono.common.mx import clamp_fp8, code_ceil, rcp_pow2
from sgl_amd_mono.common.ops import (
    CM_DEV,
    bf16_pair,
    bf_hi,
    bf_lo,
    fp8_pack4,
    hw_exp2,
    permlane_swap,
    rsrc,
    xred,
)
from sgl_amd_mono.common.plan import kernel_sources, key_tuple
from sgl_amd_mono.common.sync import _spin

SOURCES = kernel_sources(__name__)

HC = 4
H = 5120
K_MIX = HC * H
MIX = HC * (HC + 2)  # pre 4 | post 4 | comb 16
BK = 64  # hidden columns a slice
SLICES = H // BK
BM = 16  # rows a wave of a slice CTA (one MFMA block)
THREADS = 256
WAVES = THREADS // 64
INV_K = 1.0 / K_MIX
CHUNK = 2048  # rmsnorm_fake_quant_row's chunk at H = 5120
FLAG = 32  # i32 words a flag: a 128-byte line of its own
L2E = 1.4426950408889634
QUANTS = ("mxfp8", "grid", None)


class Form(msgspec.Struct, frozen=True):
    """A seam's static shape: ``post`` (an owed post onto the streams), ``combine`` (the
    layer input from the previous pre; else stream 0), ``quant`` (``QUANTS``)."""

    post: bool = True
    combine: bool = True
    quant: str | None = "mxfp8"
    stage: str = "seama"  # the kernel's stage tag: seama (before attention) or seamb (before the FFN)
    rms_eps: float = 1e-20  # the mixes' RMS (the layer's rms_norm_eps)
    norm_eps: float = 1e-20  # the RMSNorm's
    hc_eps: float = 1e-6
    iters: int = 20  # sinkhorn rounds

    def __post_init__(self):
        assert self.combine or not self.post
        assert self.quant in QUANTS
        assert self.stage in ("seama", "seamb")


def _cdiv(a, b):
    return (a + b - 1) // b


# waves a 16-row block of a slice CTA takes: 1 (a wave computes both 16-mix MFMA tiles)
# or 2 (a wave a tile; both compute the block's streams, the first stores them)
SPLIT = {96: 2, 192: 2, 384: 1}
ROWS_A_CTA = 1  # a row CTA's rows: a wave for the mixes, three for the norm


def width_class(m: int) -> int:
    for c in sorted(SPLIT):
        if m <= c:
            return c
    raise ValueError(f"seam wide: {m} rows past {max(SPLIT)}")


def group_rows(m: int) -> int:
    """Rows a slice CTA takes."""
    return BM * WAVES // SPLIT[width_class(m)]


def grid(m: int) -> tuple[int, int]:
    """(slice CTAs, row CTAs) of an m-row seam."""
    return SLICES * _cdiv(m, group_rows(m)), _cdiv(m, ROWS_A_CTA)


def scratch_bytes(m: int) -> int:
    """Device-scope scratch an m-row seam writes and reads back (``scratch_layout``)."""
    return sum(n for _, n in scratch_layout(m).values())


def scratch_layout(m: int) -> dict[str, tuple[int, int]]:
    """Scratch regions -> (byte offset, bytes): y [m, H] bf16, the mix partials
    [slice, m, 24] and the row sums of squares [slice, m] (f32)."""
    out, off = {}, 0
    for name, n in (("y", m * H * 2), ("mix", SLICES * m * MIX * 4), ("sq", SLICES * m * 4)):
        out[name] = (off, n)
        off = -(-(off + n) // 256) * 256
    return out


def flag_words(m: int) -> int:
    """i32 words of an m-row seam's flags (zeros once, then left to the launches): one a
    slice CTA, one a row CTA's wave."""
    p1, p2 = grid(m)
    return (p1 + p2 * WAVES) * FLAG


# ---------------------------------------------------------------- arithmetic
def amd_sqrt(x):
    """``tl.sqrt``: the hardware v_sqrt_f32."""
    return fx.Float32(
        llvm.call_intrinsic(T.f32, "llvm.amdgcn.sqrt.f32", [as_ir_value(fx.Float32(x))], [], [])
    )


def exp2_scaled(c, z):
    """exp2(c z) as Triton's ``tl.exp`` compiles (``c`` = +-log2(e) as f32): the product
    fused into the range reduction's add (+64 under -126), v_exp_f32, then 2^-64."""
    t = z * fx.Float32(c)
    small = t < fx.Float32(-126.0)
    arg = fma(fx.Float32(c), z, small.select(fx.Float32(64.0), fx.Float32(0.0)))
    e = hw_exp2(arg)
    return small.select(e * fx.Float32(2.0**-64), e)


def sigmoid(z):
    """``tl.sigmoid``: 1 / (1 + exp(-z))."""
    return fx.Float32(1.0) / (exp2_scaled(-L2E, z) + fx.Float32(1.0))


def quad_sq(u0, u1, u2, u3):
    """Triton's sum of four squares: (u0^2 + u1^2) + (u2^2 + u3^2), the second square of
    each pair fused."""
    return fma(u1, u1, u0 * u0) + fma(u3, u3, u2 * u2)


def pair2(w):
    """A bf16 pair word as f32 x 2 (low, high)."""
    return fx.Vector.from_elements([bf_lo(w), bf_hi(w)], fx.Float32)


def transpose4(a):
    """Lane group g (lane / 16) holds R[g][j] in a[j]; afterwards it holds R[j][g]."""
    u = [x.bitcast(fx.Int32) for x in a]
    u[0], u[2] = permlane_swap(32, u[0], u[2])
    u[1], u[3] = permlane_swap(32, u[1], u[3])
    u[0], u[1] = permlane_swap(16, u[0], u[1])
    u[2], u[3] = permlane_swap(16, u[2], u[3])
    return [x.bitcast(fx.Float32) for x in u]


def mfma4(a, b, c):
    ops = rocdl._split_mfma_operands([a, b, c, 0, 0, 0])
    return fx.Vector(rocdl.mfma_f32_16x16x4f32(T.vec(4, T.f32), *ops).result)


def butterfly(v, offsets, op=None):
    for off in offsets:
        v = xred(v, off, op if op is not None else (lambda a, b: a + b))
    return v


# ---------------------------------------------------------------- slice CTA
def _fn_loads(tid, a, s):
    """Slice s's mix weights, 4 columns a load (``_fn_table``)."""
    r_fn = rsrc(a["fn"])
    loads = []
    for i in range_constexpr(MIX * HC * BK // 4 // THREADS):
        e = tid + THREADS * i  # a 4-column piece: (n, copy, piece)
        n, rest = e // (HC * BK // 4), e % (HC * BK // 4)
        copy, p = rest // (BK // 4), rest % (BK // 4)
        off = n * K_MIX + copy * H + s * BK + 4 * p
        loads.append((n, copy, p, fx.Vector(bo.buffer_load(r_fn, off, vec_width=4, dtype=T.f32))))
    return loads


def _fn_table(loads, wl):
    """Stage slice s's mix weights in LDS in MFMA A order, [copy][ntile][chunk][lane] x
    4: element q of (copy, nt, chunk, lane) is fn[n][copy H + 64 s + 4 (4 chunk + q) + g],
    n = lane % 16 + 16 nt, g = lane / 16. Rows past the 24 mixes are left as they are:
    an MFMA output row reads only its own A row."""
    for n, copy, p, v in loads:
        # column 4 p + g of MFMA p is lane (n % 16) + 16 g, chunk p / 4, element p % 4
        for g in range_constexpr(4):
            lane = n % 16 + 16 * g
            at = (((copy * 2 + n // 16) * 4 + p // 4) * 64 + lane) * 4 + p % 4
            fx.ptr_store(v[g], wl + at)


# a wave's staging tiles in LDS: 16 rows of 64 bf16 columns (32 words), rows padded
TILE_ROW = 36  # words
TILE = BM * TILE_ROW
FN_WORDS = HC * 2 * 4 * 64 * 4  # the mix-weight table


def _block_loads(form, lane, a, s, blk, m):
    """A wave's operands of its 16-row block: x's and the streams' tiles as whole 128-byte
    rows (lane l: row l / 8 of each 8-row half, 16 bytes at column 8 (l % 8)), and the
    post, comb and pre of row l % 16."""
    c0 = s * BK
    rowc = [fx.min(blk * BM + 8 * i + lane // 8, m - 1) for i in range(2)]
    tiles = []
    if const_expr(form.post):
        r_x = rsrc(a["x"])
        tiles.append([fx.Vector(bo.buffer_load(r_x, (rowc[i] * H + c0 + 8 * (lane % 8)) // 2, vec_width=4, dtype=T.i32)) for i in range(2)])
    r_res = rsrc(a["res"])
    for k in range_constexpr(HC):
        tiles.append(
            [fx.Vector(bo.buffer_load(r_res, ((rowc[i] * HC + k) * H + c0 + 8 * (lane % 8)) // 2, vec_width=4, dtype=T.i32)) for i in range(2)]
        )
    rc = fx.min(blk * BM + lane % 16, m - 1)
    coefs = None
    if const_expr(form.post):
        post = fx.Vector(bo.buffer_load(rsrc(a["post_in"]), rc * HC, vec_width=4, dtype=T.f32))
        comb = [
            fx.Vector(bo.buffer_load(rsrc(a["comb_in"]), rc * HC * HC + 4 * j, vec_width=4, dtype=T.f32))
            for j in range(HC)
        ]
        coefs = (post, comb)
    pre = None
    if const_expr(form.combine):
        pre = fx.Vector(bo.buffer_load(rsrc(a["pre_in"]), rc * HC, vec_width=4, dtype=T.f32))
    return tiles, coefs, pre


def _stage(tl, lane, vs):
    """Write a tile's two row-piece loads (``_block_loads``) to its LDS tile ``tl``."""
    for i in range_constexpr(2):
        fx.ptr_store(vs[i].bitcast(fx.Float32), tl + (8 * i + lane // 8) * TILE_ROW + 4 * (lane % 8))


def _unstage(tl, lane, form):
    """A tile's words in the Triton layout of the form: for lane (row r = l % 16, group
    g = l / 16), the post form's four pieces (columns 16 mp + 4 g .. + 4, two words
    each), else columns 16 g .. + 16 (two pieces of four words)."""
    r, g = lane % 16, lane // 16
    v2 = fx.Vector.make_type(2, fx.Float32)
    v4 = fx.Vector.make_type(4, fx.Float32)
    if const_expr(form.post):
        return [fx.Vector(fx.ptr_load(tl + r * TILE_ROW + 8 * mp + 2 * g, result_type=v2)).bitcast(fx.Int32) for mp in range(4)]
    return [fx.Vector(fx.ptr_load(tl + r * TILE_ROW + 8 * g + 4 * h, result_type=v4)).bitcast(fx.Int32) for h in range(2)]


@flyc.jit
def _store_rows(tl, lane, pieces, r_out, word_of, ok_row, post_layout, cm=0):
    """Store a tile held in the Triton layout (``pieces``: per lane, as ``_unstage``
    returns them) as whole 128-byte rows: through the wave's LDS tile ``tl``.
    ``word_of(i)``: the global word of lane l's 16 bytes of block row 8 i + l / 8;
    ``ok_row(i)``: whether that row is stored."""
    r, g = lane % 16, lane // 16
    if const_expr(post_layout):
        for mp in range_constexpr(4):
            fx.ptr_store(pieces[mp].bitcast(fx.Float32), tl + r * TILE_ROW + 8 * mp + 2 * g)
    else:
        for h in range_constexpr(2):
            fx.ptr_store(pieces[h].bitcast(fx.Float32), tl + r * TILE_ROW + 8 * g + 4 * h)
    v4 = fx.Vector.make_type(4, fx.Float32)
    for i in range_constexpr(2):
        w = fx.Vector(fx.ptr_load(tl + (8 * i + lane // 8) * TILE_ROW + 4 * (lane % 8), result_type=v4)).bitcast(fx.Int32)
        if ok_row(i):
            bo.buffer_store(w, r_out, word_of(i), cache_modifier=cm)


def _words(vs):
    """The 8 bf16-pair words of a lane's tile piece list, in order."""
    out = []
    for v in vs:
        for i in range_constexpr(len(v)):
            out.append(v[i])
    return out


@flyc.jit
def slice_block(form, lane, a, s, blk, m, wl, tl, ops, nts, main):
    """Rows 16 blk .. + 16 of slice s on this wave (``mhc_boundary_gfx95.cuh``'s
    ``compute_block``): the MFMA tiles ``nts`` (16 mixes each); ``main``: this wave
    stores the streams, y and the sums of squares. ``ops``: ``_block_loads``."""
    g, r = lane // 16, lane % 16
    row = blk * BM + r
    valid = row < m
    vm = valid if main is True else valid & main
    c0 = s * BK
    tiles, coefs, pre = ops
    v4 = fx.Vector.make_type(4, fx.Float32)
    # through the wave's LDS tiles into the Triton layout
    for t in range_constexpr(len(tiles)):
        _stage(tl + t * TILE, lane, tiles[t])
    lay = [_unstage(tl + t * TILE, lane, form) for t in range(len(tiles))]
    xs = lay[0] if form.post else None
    rs = lay[1:] if form.post else lay
    rows = [blk * BM + 8 * i + lane // 8 for i in range(2)]

    def a_ops(k):
        """Copy k's A operands of each tile, [tile][chunk] f32 x 4."""
        return [
            [fx.Vector(fx.ptr_load(wl + (((k * 2 + nt) * 4 + ch) * 64 + lane) * 4, result_type=v4)) for ch in range(4)]
            for nt in nts
        ]
    xw = _words(xs) if xs is not None else None
    rw = [_words(rs[k]) for k in range(HC)]
    # column pairs as f32 x 2: the elementwise math runs as packed f32 ops
    if const_expr(form.post):
        xf = [pair2(w) for w in xw]
        rf = [[pair2(w) for w in rw[k]] for k in range(HC)]
        post, comb = coefs
    if const_expr(form.combine):
        y = [fx.Vector.filled(2, 0.0, fx.Float32) for _ in range(8)]
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in nts]
    sq = fx.Float32(0.0)
    r_out = rsrc(a["res_out"])
    for k in range_constexpr(HC):
        wa = a_ops(k)
        if const_expr(form.post):
            # Triton extracts each coefficient with a one-hot masked sum: c + 0.0
            pk = post[k] + fx.Float32(0.0)
            cs = [comb[j][k] + fx.Float32(0.0) for j in range(HC)]
            xt, vb = [], []
            for d in range_constexpr(8):
                v = xf[d] * pk
                for j in range_constexpr(HC):
                    v = v + rf[j][d] * cs[j]
                w = v.to(fx.BFloat16).bitcast(fx.Float32)[0]
                vb.append(w)
                xt.append(pair2(w.bitcast(fx.Int32)))
            pieces = [fx.Vector.from_elements([vb[2 * mp], vb[2 * mp + 1]], fx.Float32).bitcast(fx.Int32) for mp in range(4)]
            # the tile of stream k (read already) holds the new stream on the way out
            _store_rows(
                tl + (1 + k) * TILE, lane, pieces, r_out,
                lambda i: ((rows[i] * HC + k) * H + c0 + 8 * (lane % 8)) // 2,
                lambda i: (rows[i] < m) & main if main is not True else rows[i] < m,
                True,
            )  # fmt: skip
        else:
            xt = [pair2(w) for w in rw[k]]
        if const_expr(form.combine):
            pp = pre[k] + fx.Float32(0.0)
            for d in range_constexpr(8):
                y[d] = y[d] + xt[d] * pp
        # sum of squares in Triton's tree; Triton lane c = 2 g + hh
        part = []
        for hh in range_constexpr(2):
            if const_expr(form.post):
                # Triton lane c holds columns 2 c + {0, 1} + 16 mp: words 2 mp + hh
                eo = quad_sq(xt[hh], xt[2 + hh], xt[4 + hh], xt[6 + hh])
            else:
                # Triton lane c holds columns 8 c + 0 .. 7: words 4 hh + u
                b = 4 * hh
                eo = quad_sq(xt[b], xt[b + 1], xt[b + 2], xt[b + 3])
            # (even columns' sum, odd columns' sum)
            p = eo[0] + eo[1]
            p = xred(p, 32, lambda u, v: u + v)  # Triton lane c ^ 4
            p = xred(p, 16, lambda u, v: u + v)  # Triton lane c ^ 2
            part.append(p)
        sq = sq + (part[0] + part[1])
        # the MFMA B operands: MFMA i (columns 4 i .. + 4 of the slice) slot g
        bop = [None] * 16
        for blk4 in range_constexpr(4):
            q4 = transpose4([xt[2 * blk4][0], xt[2 * blk4][1], xt[2 * blk4 + 1][0], xt[2 * blk4 + 1][1]])
            for j in range_constexpr(4):
                if const_expr(form.post):
                    bop[4 * blk4 + j] = q4[j]  # columns 16 mp + 4 j + g
                else:
                    bop[4 * j + blk4] = q4[j]  # columns 16 j + 4 c + g
        for i in range_constexpr(16):
            for t in range_constexpr(len(nts)):
                acc[t] = mfma4(wa[t][i // 4][i % 4], bop[i], acc[t])
    r_s = a["scratch"]
    r_y = rsrc(r_s + a["y_off"])
    yw = lambda d: y[d].to(fx.BFloat16).bitcast(fx.Int32)[0]  # noqa: E731
    if const_expr(form.combine and form.post):
        pieces = [fx.Vector.from_elements([yw(2 * mp), yw(2 * mp + 1)], fx.Int32) for mp in range(4)]
    elif const_expr(form.combine):
        pieces = [fx.Vector.from_elements([yw(4 * h + u) for u in range(4)], fx.Int32) for h in range(2)]
    else:
        # the layer input is stream 0 as it stands
        pieces = rs[0]
    # through tile 0 (x's, or stream 0's: read already)
    _store_rows(
        tl, lane, pieces, r_y, lambda i: (rows[i] * H + c0 + 8 * (lane % 8)) // 2,
        lambda i: (rows[i] < m) & main if main is not True else rows[i] < m,
        form.combine and form.post, CM_DEV,
    )  # fmt: skip
    r_mix = rsrc(r_s + a["mix_off"])
    # acc[t][q] = D[mix 16 nts[t] + 4 g + q][row r]
    for t in range_constexpr(len(nts)):
        nt = nts[t]
        if const_expr(isinstance(nt, int)):
            ok = valid if nt == 0 else valid & (g < (MIX - 16) // 4)
        else:
            ok = valid & ((nt == 0) | (g < (MIX - 16) // 4))
        if ok:
            bo.buffer_store(acc[t], r_mix, (s * m + row) * MIX + 16 * nt + 4 * g, cache_modifier=CM_DEV)
    if vm & (g == 0):
        bo.buffer_store(sq, rsrc(r_s + a["sq_off"]), s * m + row, cache_modifier=CM_DEV)


# ---------------------------------------------------------------- row CTA
def await_slices(a, lane, grp, phase):
    """Wait until the 80 slice CTAs of row group ``grp`` have flipped their flags to
    ``phase`` (lane l polls slice l, and slice 64 + l)."""
    r_f = rsrc(a["flags"])
    s2 = fx.min(64 + lane, SLICES - 1)

    def load_all():
        return fx.Vector.from_elements(
            [
                fx.Int32(bo.buffer_load(r_f, (grp * SLICES + sl) * FLAG, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
                for sl in (lane, s2)
            ],
            fx.Int32,
        )

    _spin(load_all, lambda v: (v[0] != phase) | (v[1] != phase), guarded=False)


def _select16(vals, idx):
    """vals[idx] for a lane index in 0 .. len(vals)."""
    v = vals[0]
    for n in range_constexpr(1, len(vals)):
        v = (idx == n).select(vals[n], v)
    return v


def row_mixes(lane, a, row, m):
    """Row ``row``'s 24 mixes and sum of squares over the 80 slices, in the tree of
    ``_hc_mix_reduce_sinkhorn_row`` (4 warps): lane l holds slice l, and slice 64 + l on
    lanes below 16. Returns (pre/post sums [8], comb sums [16], sq), each on every lane."""
    r_s = a["scratch"]
    r_mix, r_sq = rsrc(r_s + a["mix_off"]), rsrc(r_s + a["sq_off"])
    lo = lane < SLICES - 64
    s2 = fx.min(64 + lane, SLICES - 1)
    v1 = [fx.Vector(bo.buffer_load(r_mix, (lane * m + row) * MIX + 4 * i, vec_width=4, dtype=T.f32, cache_modifier=CM_DEV)) for i in range(MIX // 4)]
    v2 = [fx.Vector(bo.buffer_load(r_mix, (s2 * m + row) * MIX + 4 * i, vec_width=4, dtype=T.f32, cache_modifier=CM_DEV)) for i in range(MIX // 4)]
    q1 = fx.Float32(bo.buffer_load(r_sq, lane * m + row, vec_width=1, dtype=T.f32, cache_modifier=CM_DEV))
    q2 = fx.Float32(bo.buffer_load(r_sq, s2 * m + row, vec_width=1, dtype=T.f32, cache_modifier=CM_DEV))
    zero = fx.Float32(0.0)
    add = lambda u, v: u + v  # noqa: E731
    gates = []
    for n in range_constexpr(2 * HC):
        # warps of 32 slices (slice xor 16, 8, 4, 2, 1), then (W0 + W2) + (W1 + W3)
        x1 = v1[n // 4][n % 4]
        x2 = lo.select(v2[n // 4][n % 4], zero)
        w1 = butterfly(x1, (16, 8, 4, 2, 1))
        w2 = butterfly(x2, (16, 8, 4, 2, 1))
        gates.append(xred(w1 + w2, 32, add))
    combs = []
    for n in range_constexpr(2 * HC, MIX):
        # a thread's two slices (s, s + 64) first, warps of 16 (slice xor 8, 4, 2, 1),
        # then (W0 + W2) + (W1 + W3)
        p = v1[n // 4][n % 4] + lo.select(v2[n // 4][n % 4], zero)
        p = butterfly(p, (8, 4, 2, 1))
        combs.append(xred(xred(p, 32, add), 16, add))
    # warps of 64 slices (DPP: row xor 8, 4, 2, 1, then rows), W0 + W1
    sq = butterfly(q1, (8, 4, 2, 1, 16, 32)) + butterfly(lo.select(q2, zero), (8, 4, 2, 1, 16, 32))
    return gates, combs, sq


@flyc.jit
def row_gates(form, lane, a, row, gates, combs, sq, valid):
    """The reduce + sinkhorn row's gates: pre and post on lanes 0 .. 3 (j = lane % 4), comb
    (j, k) on lane 4 j + k of each 16 (the kernel's [4, 4] layout), stored for ``row``."""
    sc = [fx.Float32(bo.buffer_load(rsrc(a["hc_scale"]), i, vec_width=1, dtype=T.f32)) for i in range(3)]
    l16 = lane % 16
    j4 = lane % 4
    base_g = fx.Float32(bo.buffer_load(rsrc(a["hc_base"]), j4, vec_width=1, dtype=T.f32))
    base_p = fx.Float32(bo.buffer_load(rsrc(a["hc_base"]), HC + j4, vec_width=1, dtype=T.f32))
    base_c = fx.Float32(bo.buffer_load(rsrc(a["hc_base"]), 2 * HC + l16, vec_width=1, dtype=T.f32))
    rsqrt = fx.Float32(1.0) / amd_sqrt(fma(fx.Float32(INV_K), sq, fx.Float32(form.rms_eps)))
    a_pre = _select16(gates[:HC], j4)
    a_post = _select16(gates[HC:], j4)
    a_comb = _select16(combs, l16)
    pre = sigmoid(fma(rsqrt * a_pre, sc[0], base_g)) + fx.Float32(form.hc_eps)
    p = sigmoid(fma(rsqrt * a_post, sc[1], base_p))
    post = p + p
    c = fma(rsqrt * a_comb, sc[2], base_c)
    mx = lambda u, v: fx.max(u, v)  # noqa: E731
    c = exp2_scaled(L2E, c - butterfly(c, (2, 1), mx))
    c = c / butterfly(c, (2, 1)) + fx.Float32(form.hc_eps)
    c = c / (butterfly(c, (8, 4)) + fx.Float32(form.hc_eps))
    for _ in range_constexpr(form.iters - 1):
        c = c / (butterfly(c, (2, 1)) + fx.Float32(form.hc_eps))
        c = c / (butterfly(c, (8, 4)) + fx.Float32(form.hc_eps))
    if valid & (lane < HC):
        bo.buffer_store(pre, rsrc(a["pre_out"]), row * HC + lane)
        bo.buffer_store(post, rsrc(a["post_out"]), row * HC + lane)
    if valid & (lane < HC * HC):
        bo.buffer_store(c, rsrc(a["comb_out"]), row * HC * HC + lane)


# rmsnorm_fake_quant_row's threads (t, 4 warps of 64) of each norm wave: wave 1 plays
# warp 0, wave 2 warp 1, wave 3 warps 2 and 3
NORM_WARPS = {1: (0,), 2: (1,), 3: (2, 3)}


@flyc.jit
def row_norm(form, lane, wave, a, row, valid, red):
    """``rmsnorm_fake_quant_row`` of y's row at H = 5120 over three waves: this wave plays
    the kernel's warps ``NORM_WARPS[wave]``, thread t = l + 64 w holding eight columns
    8 t + 2048 c of each chunk c (the third only below thread 128). The warps' sums meet
    in LDS (``red``: four sums, then a flag a wave)."""
    r_y = rsrc(a["scratch"] + a["y_off"])
    r_w = rsrc(a["norm_w"])
    warps = NORM_WARPS[wave]
    cols = [(w, c, 8 * (lane + 64 * w) + CHUNK * c) for w in warps for c in range(3) if c < 2 or w < 2]
    wvs = [fx.Vector(bo.buffer_load(r_w, col // 2, vec_width=4, dtype=T.i32)) for _, _, col in cols]
    parts = []  # (w, c, column, y words, weight words)
    for i, (w, c, col) in enumerate(cols):
        v = fx.Vector(bo.buffer_load(r_y, (row * H + col) // 2, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV))
        parts.append((w, c, col, [v[q] for q in range(4)], wvs[i]))
    for w in warps:
        xs = {c: [f(x[q]) for q in range(4) for f in (bf_lo, bf_hi)] for ww, c, _, x, _ in parts if ww == w}
        zero = [fx.Float32(0.0)] * 8
        x0, x1, x2 = xs[0], xs[1], xs.get(2, zero)
        e = [fma(x2[j], x2[j], fma(x0[j], x0[j], x1[j] * x1[j])) for j in range(8)]
        acc = ((e[0] + e[2]) + (e[4] + e[6])) + ((e[1] + e[3]) + (e[5] + e[7]))
        fx.ptr_store(butterfly(acc, (8, 4, 2, 1, 16, 32)), red + w)
    fx.ptr_store(fx.Float32(1.0), red + 4 + wave)

    def load_flags():
        return fx.Vector.from_elements([fx.ptr_load(red + 4 + v) for v in NORM_WARPS], fx.Float32)

    one = fx.Float32(1.0)
    _spin(load_flags, lambda v: (v[0] != one) | (v[1] != one) | (v[2] != one), guarded=False)
    tot = [fx.ptr_load(red + w) for w in range(4)]
    total = (tot[0] + tot[2]) + (tot[1] + tot[3])
    var = total / fx.Float32(float(H)) + fx.Float32(form.norm_eps)
    rstd = fx.Float32(1.0) / fmath.sqrt(var)
    r_n = rsrc(a["normed"])
    for w, c, col, xw, wv in parts:
        x = [f(xw[q]) for q in range(4) for f in (bf_lo, bf_hi)]
        ws = [f(wv[q]) for q in range(4) for f in (bf_lo, bf_hi)]
        ys = [(x[j] * rstd) * ws[j] for j in range(8)]
        words = fx.Vector.from_elements([bf16_pair(ys[2 * q], ys[2 * q + 1]) for q in range(4)], fx.Float32).bitcast(fx.Int32)
        if valid:
            bo.buffer_store(words, r_n, (row * H + col) // 2)
        if const_expr(form.quant is not None):
            yb = [bf_lo(words[q]) if h == 0 else bf_hi(words[q]) for q in range(4) for h in range(2)]
            amax = fx.Float32(0.0)
            for v in yb:
                amax = fx.max(amax, abs(v))
            amax = butterfly(amax, (1, 2), lambda u, v: fx.max(u, v))
            code = code_ceil(fx.max(amax, fx.Float32(1e-10)) * fx.Float32(1.0 / 448.0))
            mul = rcp_pow2(code)
            q8 = [fp8_pack4(*[clamp_fp8(v * mul) for v in yb[4 * h : 4 * h + 4]]) for h in range(2)]
            if const_expr(form.quant == "mxfp8"):
                if valid:
                    bo.buffer_store(fx.Vector.from_elements(q8, fx.Int32), rsrc(a["q"]), (row * H + col) // 4)
                if valid & (lane % 4 == 0):
                    bo.buffer_store(fx.Int8(code), rsrc(a["qs"]), row * (H // 32) + col // 32)
            else:
                scale = ((code) << 23).bitcast(fx.Float32)
                back = []
                for h in range_constexpr(2):
                    for half in range_constexpr(2):
                        f8 = fx.Vector(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), q8[h], half == 1))
                        back += [f8[0] * scale, f8[1] * scale]
                dq = fx.Vector.from_elements([bf16_pair(back[2 * q], back[2 * q + 1]) for q in range(4)], fx.Float32).bitcast(fx.Int32)
                if valid:
                    bo.buffer_store(dq, rsrc(a["q"]), (row * H + col) // 2)


# ---------------------------------------------------------------- the launch
_BUILDS: dict = {}

ARGS = (
    "x", "res", "post_in", "comb_in", "pre_in", "fn", "hc_scale", "hc_base", "norm_w",
    "res_out", "normed", "q", "qs", "pre_out", "post_out", "comb_out", "scratch", "flags",
)  # fmt: skip


def standalone(m: int, form: Form):
    """``launch(*addresses in ARGS order)``: device addresses of the module docstring's
    tensors (``q`` / ``qs`` unused without quant; ``scratch`` of ``scratch_bytes(m)``;
    ``flags`` of ``flag_words(m)`` i32, zeros before the first launch)."""
    key = (m, form)
    if key in _BUILDS:
        return _BUILDS[key]
    keyed = key_tuple({"m": m, "form": form}, SOURCES)
    p1, p2 = grid(m)
    split = SPLIT[width_class(m)]
    lay = scratch_layout(m)
    name = (
        f"ds41_{form.stage}_wide_m{m}_p{int(form.post)}c{int(form.combine)}_{form.quant or 'none'}"
    )

    @fx.struct
    class Smem:
        w: fx.Array[fx.Float32, FN_WORDS + WAVES * (HC + 1) * TILE, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def seam_kernel(
        x: Int64, res: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64, fn: Int64,
        hc_scale: Int64, hc_base: Int64, norm_w: Int64, res_out: Int64, normed: Int64,
        q: Int64, qs: Int64, pre_out: Int64, post_out: Int64, comb_out: Int64,
        scratch: Int64, flags: Int64,
    ):  # fmt: skip
        a = {
            "x": x, "res": res, "post_in": post_in, "comb_in": comb_in, "pre_in": pre_in,
            "fn": fn, "hc_scale": hc_scale, "hc_base": hc_base, "norm_w": norm_w,
            "res_out": res_out, "normed": normed, "q": q, "qs": qs, "pre_out": pre_out,
            "post_out": post_out, "comb_out": comb_out, "scratch": scratch,
            "y_off": lay["y"][0], "mix_off": lay["mix"][0], "sq_off": lay["sq"][0], "flags": flags,
        }  # fmt: skip
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        bid = fx.block_idx.x
        r_f = rsrc(flags)
        wl = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        if bid < p1:
            s, grp = bid % SLICES, bid // SLICES
            old = fx.Int32(bo.buffer_load(r_f, bid * FLAG, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
            if const_expr(split == 1):
                blk, nts, main = grp * WAVES + wave, (0, 1), True
            else:
                blk, nts = grp * (WAVES // 2) + wave // 2, (wave % 2,)
                main = wave % 2 == 0
            # the mix weights first (the LDS table and its barrier wait on them), then the
            # block's operands
            fnl = _fn_loads(tid, a, s)
            ops = _block_loads(form, lane, a, s, blk, m)
            _fn_table(fnl, wl)
            gpu.barrier()
            if blk * BM < m:
                slice_block(form, lane, a, s, blk, m, wl, wl + FN_WORDS + wave * (HC + 1) * TILE, ops, nts, main)
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if tid == 0:
                bo.buffer_store(1 - old, r_f, bid * FLAG, cache_modifier=CM_DEV)
        else:
            k = bid - p1
            row = k
            slot = p1 + k * WAVES + wave
            old = fx.Int32(bo.buffer_load(r_f, slot * FLAG, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
            phase = 1 - old
            # the norm waves' LDS meeting point: zeros first
            if tid < 8:
                fx.ptr_store(fx.Float32(0.0), wl + tid)
            gpu.barrier()
            if row < m:
                await_slices(a, lane, row // group_rows(m), phase)
                if wave == 0:
                    gates, combs, sq = row_mixes(lane, a, row, m)
                    row_gates(form, lane, a, row, gates, combs, sq, row < m)
                elif wave == 1:
                    row_norm(form, lane, 1, a, row, row < m, wl)
                elif wave == 2:
                    row_norm(form, lane, 2, a, row, row < m, wl)
                else:
                    row_norm(form, lane, 3, a, row, row < m, wl)
            if lane == 0:
                bo.buffer_store(phase, r_f, slot * FLAG, cache_modifier=CM_DEV)

    @flyc.jit
    def launch(
        x: Int64, res: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64, fn: Int64,
        hc_scale: Int64, hc_base: Int64, norm_w: Int64, res_out: Int64, normed: Int64,
        q: Int64, qs: Int64, pre_out: Int64, post_out: Int64, comb_out: Int64,
        scratch: Int64, flags: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):  # fmt: skip
        _ = keyed
        seam_kernel(
            x, res, post_in, comb_in, pre_in, fn, hc_scale, hc_base, norm_w, res_out,
            normed, q, qs, pre_out, post_out, comb_out, scratch, flags,
        ).launch(grid=(p1 + p2,), block=(THREADS,), stream=stream)  # fmt: skip

    _BUILDS[key] = launch
    return launch
