# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next's device helpers that ``sgl_amd_mono.common`` does not hold:
the guarded wait of a one-GPU launch."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import ballot, rsrc
from sgl_amd_mono.common.sync import CHECK_POLLS, _spin


def _deadline(epoch):
    """The guard's deadline in clock ticks (words 1-2 of the epoch buffer), loaded
    into scalar registers: a wait's check takes no vector register."""
    w = fx.Vector(
        bo.buffer_load(rsrc(epoch), 0, vec_width=4, dtype=T.i32, is_scalar=True)
    )
    lo, hi = w[guard.GUARD_DEADLINE], w[guard.GUARD_DEADLINE + 1]
    return (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))


@flyc.jit
def spin(load_all, pending):
    """``load_all()`` again while ``pending(value)``; the value.

    The wait of a launch that runs on one GPU (TP 1, no peers): a pending wait
    re-polls with a single ``s_nop`` between loads and checks its deadline every
    ``CHECK_POLLS`` polls. Past it, it sets the error word and stops the launch
    (``guard.Guard.fail``): with no peers there is no failure elsewhere to watch,
    and a failed wait on this GPU has already stopped it. The fault injection's
    stall runs ``common.sync._spin``."""
    g = guard.current()
    assert g is None or g.peers is None, "a wait with peers takes common.sync._spin"
    if const_expr(g is None or g.fault_site is not None):
        return _spin(load_all, pending)
    site = g.next_site()
    epoch, fail = g.epoch, g.fail
    v = load_all()
    t0 = fx.Int64(0)
    polls = fx.Int32(0)
    late = fx.Int32(0)
    # a wave-uniform loop (until no lane is pending), so its poll count and clock
    # stay in scalar registers; a satisfied lane reloads words that no longer change
    while (ballot(pending(v)) != 0) & (late == 0):
        rocdl.s_nop(0)
        v = load_all()
        polls = polls + 1
        if (polls & (CHECK_POLLS - 1)) == 0:
            now = guard.clock()
            t0 = (polls == CHECK_POLLS).select(now, t0)
            late = ((now - t0) > _deadline(epoch)).select(fx.Int32(1), fx.Int32(0))
    if late != 0:
        fail(site)
    return v
