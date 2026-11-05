# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The launch epoch: every CTA marks it, and CTA 0 moves it on once all have.

The epoch buffer starts with the guard block (``guard.py``): the epoch, the
wait deadline and the error words, then a mark per CTA."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import T

from sgl_amd_mono.common.guard import GUARD_WORDS
from sgl_amd_mono.common.ops import (
    CM_DEV,
    global_ptr,
    rsrc,
)
from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.common.sync import _spin


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``aiter::<stem>_<params>``."""
    name = f"{stem}_{kernel_signature(**params)}"
    return f"_ZN5aiter{len(name)}{name}E"


def gstore(addr, value, words=4):
    """Store ``words`` i32 (an Int32 or a Vector) at a 64-bit global address."""
    fx.generic_store(global_ptr(addr, words), value)


def spin_until(load, pending):
    """Poll ``load()``, sleeping between, until ``pending`` clears; return the value."""
    return _spin(load, pending)


EPOCH_MARKS = GUARD_WORDS  # word offset of the per-CTA marks, past the guard block


@flyc.jit
def epoch_begin(c, epoch, ep):
    """Mark that this CTA read epoch ``ep``."""
    if c["tid"] == 0:
        bo.buffer_store(
            ep + 1, rsrc(epoch), EPOCH_MARKS + c["bid"], cache_modifier=CM_DEV
        )


@flyc.jit
def epoch_end(c, epoch, ep, reset=None, who=0, marks=True):
    """CTA ``who``: once every CTA marked the epoch, run ``reset()`` and move the epoch on.

    ``marks=False``: the caller already saw a flag from every CTA of this launch,
    stored after the CTA read the epoch, so no CTA can read the moved epoch."""
    if c["bid"] == who:
        tid = c["tid"]
        if const_expr(marks):
            if tid < BLOCKS:
                r = rsrc(epoch)

                def load_mark():
                    return fx.Int32(
                        bo.buffer_load(
                            r,
                            EPOCH_MARKS + tid,
                            vec_width=1,
                            dtype=T.i32,
                            cache_modifier=CM_DEV,
                        )
                    )

                _ = spin_until(load_mark, lambda v: v != ep + 1)
            gpu.barrier()
        if tid == 0:
            if const_expr(reset is not None):
                reset()
            gstore(epoch, ep + 1, words=1)
