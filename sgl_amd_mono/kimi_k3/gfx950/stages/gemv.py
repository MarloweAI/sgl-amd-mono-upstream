# SPDX-License-Identifier: Apache-2.0
"""bf16 GEMV pieces the K3 mono kernels share: 16 weight rows a task against up
to 16 token rows in LDS, one ``mfma_f32_16x16x32_bf16`` per 32 K.

K is walked in 128-wide chunks, wave w taking chunks w, w + 8, ...; a lane reads
64 contiguous bytes of its row a chunk (4 dwordx4 at q 32 + g 8 elements: lane
group g = lane / 16), and the token rows are read at the same K. The K order
inside a chunk is a permutation both operands share, so the dot is unchanged.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_NT,
    mfma_bf16,
    rsrc,
)
from sgl_amd_mono.common.plan import THREADS, WAVES

ROWS = 16
KCH = 128
LDS_PAD = 8  # bf16 a padded token row


def chunks(k):
    """A wave's K chunks for K = ``k``: (count a wave, total)."""
    n = k // KCH
    return (n + WAVES - 1) // WAVES, n


def loads(lane, wave, w, row0, k, nt=True, ldk=None, k0=0, nbytes=None, cm=CM_NT):
    """This wave's operands of weight rows ``row0`` .. + 16 (row-major, ``ldk``
    bf16 a row at ``w``, K columns ``k0`` .. ``k0 + k``): a list a chunk
    (clamped into K: ``mfmas`` zeroes the extra). ``nbytes`` bounds the buffer:
    0 makes every load return zero without touching memory."""
    per, n = chunks(k)
    ldk = k if ldk is None else ldk
    r_w = rsrc(w, nbytes)
    row = row0 + lane % ROWS
    g = lane // ROWS
    out = []
    for i in range(per):
        ch = fx.min(wave + WAVES * i, n - 1)
        base = (row * ldk + k0 + ch * KCH + g * 8) // 2
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        base + 16 * q,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=cm if nt else 0,
                    )
                )
                for q in range(4)
            ]
        )
    return out


def mfmas(lane, wave, xl, xrow, k, s, ops, t0=0):
    """``loads``' operands against token rows t0 .. t0 + s - 1 (s <= 16) of the
    LDS rows ``xl`` (``xrow`` bf16 a row) -> this wave's 16x16 accumulator."""
    per, n = chunks(k)
    g = lane // ROWS
    t = t0 + fx.min(lane % ROWS, s - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(per):
        ch = wave + WAVES * i
        live = ch < n
        kk = fx.min(ch, n - 1) * KCH + g * 8
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    xl + (t * xrow + kk + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            if const_live(per, n):
                xb = fx.Vector.from_elements(
                    [live.select(xb[d], fx.Int32(0)) for d in range(4)], fx.Int32
                )
            acc = mfma_bf16(
                ops[i][q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    return acc


def mfmas_rounds(lane, wave, xl, xrow, k0, s, ops, acc, r0, r1):
    """``mfmas``' rounds r0 .. r1 (chunks wave + 8 i) chained onto ``acc``, the
    token rows in LDS holding columns k0 .. : the same MFMAs in the same order,
    so a K staged a segment at a time sums bit for bit as ``mfmas``."""
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    for i in range_constexpr(r0, r1):
        kk = (wave + WAVES * i) * KCH + g * 8 - k0
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    xl + (t * xrow + kk + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            acc = mfma_bf16(
                ops[i][q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    return acc


def const_live(per, n):
    """Whether some wave's last chunk is past K (its MFMA must add zero)."""
    return per * WAVES != n


def tiles(s):
    """The 16-row token tiles of an s-row step: (first row, rows) each."""
    return [(t0, min(ROWS, s - t0)) for t0 in range(0, s, ROWS)]


@flyc.jit
def rows_to_lds(tid, src, k, s, xl, xrow, cm=0, ldk=None, k0=0, t0=0):
    """Columns k0 .. k0 + k of token rows t0 .. t0 + s - 1 of the bf16 [., ldk]
    at ``src`` -> LDS rows 0 .. s - 1 (``xrow`` a row), 16 B a lane."""
    ldk = k if ldk is None else ldk
    words = s * k // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        v = fx.Vector(
            bo.buffer_load(
                rsrc(src),
                ((t0 + t) * ldk + k0 + kk) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cm,
            )
        )
        if tid + THREADS * i < words:
            fx.ptr_store(v, xl + (t * xrow + kk) // 2)
    gpu.barrier()


def rows_load(tid, src, k, s, cm=0, ldk=None, k0=0, t0=0):
    """``rows_to_lds``' loads (16 B a lane), for ``rows_store``."""
    ldk = k if ldk is None else ldk
    words = s * k // 8
    out = []
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        out.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(src),
                    ((t0 + t) * ldk + k0 + kk) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=cm,
                )
            )
        )
    return out


@flyc.jit
def rows_store(tid, vals, k, s, xl, xrow):
    """``rows_load``'s words -> LDS rows 0 .. s - 1 (``xrow`` a row), no barrier."""
    words = s * k // 8
    for i in range_constexpr(len(vals)):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        if tid + THREADS * i < words:
            fx.ptr_store(vals[i], xl + (t * xrow + kk) // 2)


def dev_rows_to_lds(tid, src, k, s, xl, xrow, t0=0):
    """``rows_to_lds`` at device scope: rows another CTA of this launch wrote."""
    rows_to_lds(tid, src, k, s, xl, xrow, CM_DEV, t0=t0)
