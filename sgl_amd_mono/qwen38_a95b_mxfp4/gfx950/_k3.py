# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ops.py, atom/mono/device/sync.py, atom/mono/device/ranks.py,
# atom/mono/runtime/abi.py
"""The Kimi-K3 mono framework names the Qwen3.8 kernels use, copied unchanged
from sgl-project/sglang#43481 (``kernels/ops/moe/k3_mono_flydsl`` at 1ebfeb251c):
``common/ops.py`` (wave reductions, the CTA scan), ``common/plan.py``
(``KernelAbi``), ``common/sync.py`` (``shift``), ``common/ranks.py`` and
``stages/gemv.py`` (the bf16 GEMV stage, imported as ``gemv``).

``sgl_amd_mono.common`` has ``peer_bases`` and ``sum_partials`` with the same
bodies, and a ``wave_max`` with another body. sgl-project/sgl-amd-mono#13 adds
``wave_sum``, the scan, ``shift`` and ``KernelAbi`` to ``common``, and keeps the
GEMV stage in ``kimi_k3``.

Every other name the kernels use comes from ``sgl_amd_mono.common``: #43481's,
except that every wait runs under ``common/guard.py`` and ``Mailbox`` stores the
tag as given (#43481 stores tag + 1).
"""

import inspect
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_NT,
    bf2_f32,
    butterfly,
    load_ptr64,
    mfma_bf16,
    permlane_swap,
    rsrc,
    xshfl,
)
from sgl_amd_mono.common.plan import THREADS, WAVES
from sgl_amd_mono.common.sync import Addr

# ---- common/ops.py ----------------------------------------------------------


def wave_sum(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1))


def wave_max(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1), fx.max)


def _xpartner(v, off, lane):
    """Lane ``lane ^ off``'s v: DPP within a row, v_permlane*_swap across rows
    (``xshfl`` takes ``shuffle_xor`` there)."""
    if off < 16:
        return xshfl(v, off)
    a, b = permlane_swap(off, v, v)
    return ((lane & off) != 0).select(a, b)


def wave_incl_scan(v, lane):
    """Inclusive prefix sum of an Int32 over the wave, in lane order."""
    blk = v
    for off in (1, 2, 4, 8, 16, 32):
        o = _xpartner(blk, off, lane)
        v = v + ((lane & off) != 0).select(o, 0)
        blk = blk + o
    return v


@flyc.jit
def block_excl_scan(v, lane, wave, buf):
    """(exclusive prefix of v over the CTA in thread order, CTA total); ``buf``
    needs WAVES words."""
    incl = wave_incl_scan(v, lane)
    gpu.barrier()
    if lane == 63:
        fx.ptr_store(incl, buf + wave)
    gpu.barrier()
    off = fx.Int32(0)
    tot = fx.Int32(0)
    for w in range_constexpr(WAVES):
        cnt = fx.ptr_load(buf + w)
        off = off + (w < wave).select(cnt, 0)
        tot = tot + cnt
    return incl - v + off, tot


# ---- common/plan.py ---------------------------------------------------------


def _python_params(fn) -> tuple[str, ...]:
    """Parameter names of a ``@flyc.kernel`` / ``@flyc.jit`` object's Python
    function (FlyDSL keeps it as ``_func`` / ``func``)."""
    py = getattr(fn, "func", None) or getattr(fn, "_func", None)
    if py is None:
        raise TypeError(f"no Python function behind {fn!r}")
    return tuple(inspect.signature(py).parameters)


@dataclass(frozen=True)
class KernelAbi:
    names: tuple[str, ...]

    def check(self, kernel, launcher) -> None:
        """The kernel takes exactly ``names``; the launcher too, then ``stream``."""
        got = _python_params(kernel)
        if got != self.names:
            raise TypeError(f"kernel parameters {got} != ABI {self.names}")
        got = _python_params(launcher)
        if got != (*self.names, "stream"):
            raise TypeError(f"launcher parameters {got} != ABI {self.names} + stream")

    def pack(self, values: dict) -> list:
        """``values`` (every name, nothing else) in ABI order."""
        try:
            if len(values) == len(self.names):
                return [values[n] for n in self.names]
        except KeyError:
            pass
        extra = sorted(set(values) - set(self.names))
        missing = sorted(set(self.names) - set(values))
        raise TypeError(f"ABI mismatch: missing {missing}, unexpected {extra}")

    def zeros(self) -> list:
        """Every argument 0: what a compile-only call passes."""
        return [0] * len(self.names)


# ---- common/sync.py ---------------------------------------------------------


def shift(a, delta):
    """``a`` moved by ``delta`` bytes (a traced or Python integer), same region."""
    return Addr(a.region, a.space, a.value + delta)


# ---- common/ranks.py --------------------------------------------------------


def peer_bases(peers, tp):
    """Every rank's peer buffer base from the ``peers`` table (an i64 a rank),
    loaded once: a put's address must not wait on a load of its own."""
    return [load_ptr64(peers, p) for p in range(tp)]


def sum_partials(poll, own, pair_of, tp):
    """The ``tp`` ranks' bf16 pair partials at ``pair_of(src)`` of the mailbox
    region ``own``, summed in fp32 in rank order 0 .. tp - 1: one poll batch ->
    (lo sum, hi sum). Every rank sums in the same order, so they agree."""
    ws = poll([(own, pair_of(src), 1) for src in range(tp)])
    acc0, acc1 = bf2_f32(ws[0][0])
    for src in range_constexpr(1, tp):
        lo, hi = bf2_f32(ws[src][0])
        acc0 = acc0 + lo
        acc1 = acc1 + hi
    return acc0, acc1


# ---- stages/gemv.py ---------------------------------------------------------
# bf16 GEMV pieces: 16 weight rows a task against up to 16 token rows in LDS, one
# ``mfma_f32_16x16x32_bf16`` per 32 K. K is walked in 128-wide chunks, wave w
# taking chunks w, w + 8, ...; a lane reads 64 contiguous bytes of its row a chunk
# (4 dwordx4 at q 32 + g 8 elements: lane group g = lane / 16), and the token rows
# are read at the same K. The K order inside a chunk is a permutation both
# operands share, so the dot is unchanged.

ROWS = 16
KCH = 128
LDS_PAD = 8  # bf16 a padded token row


def chunks(k):
    """A wave's K chunks for K = ``k``: (count a wave, total)."""
    n = k // KCH
    return (n + WAVES - 1) // WAVES, n


def loads(lane, wave, w, row0, k, nt=True, ldk=None, k0=0):
    """This wave's operands of weight rows ``row0`` .. + 16 (row-major, ``ldk``
    bf16 a row at ``w``, K columns ``k0`` .. ``k0 + k``): a list a chunk
    (clamped into K: ``mfmas`` zeroes the extra)."""
    per, n = chunks(k)
    ldk = k if ldk is None else ldk
    r_w = rsrc(w)
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
                        cache_modifier=CM_NT if nt else 0,
                    )
                )
                for q in range(4)
            ]
        )
    return out


def mfmas(lane, wave, xl, xrow, k, s, ops):
    """``loads``' operands against token rows 0 .. s - 1 of the LDS rows ``xl``
    (``xrow`` bf16 a row) -> this wave's 16x16 accumulator."""
    per, n = chunks(k)
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
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


def const_live(per, n):
    """Whether some wave's last chunk is past K (its MFMA must add zero)."""
    return per * WAVES != n


@flyc.jit
def rows_to_lds(tid, src, k, s, xl, xrow, cm=0, ldk=None, k0=0):
    """Columns k0 .. k0 + k of token rows 0 .. s - 1 of the bf16 [s, ldk] at
    ``src`` -> LDS (``xrow`` a row), 16 B a lane."""
    ldk = k if ldk is None else ldk
    words = s * k // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        v = fx.Vector(
            bo.buffer_load(
                rsrc(src),
                (t * ldk + k0 + kk) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cm,
            )
        )
        if tid + THREADS * i < words:
            fx.ptr_store(v, xl + (t * xrow + kk) // 2)
    gpu.barrier()


def dev_rows_to_lds(tid, src, k, s, xl, xrow):
    """``rows_to_lds`` at device scope: rows another CTA of this launch wrote."""
    rows_to_lds(tid, src, k, s, xl, xrow, CM_DEV)
