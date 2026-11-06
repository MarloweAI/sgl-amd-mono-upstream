# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The thread and block ids the layer stages read: the hardware's, or inside ``layer_ids``
copies of them through opaque moves, masked to their ranges. A layer of a multi-layer
launch (``layer.build_mono_layers``) reads its own copies, so LLVM cannot merge its lane
math with another layer's and hold it live from one layer into the next; the masks keep
the ids' ranges, so their divisions stay unsigned."""

import contextlib

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T

from sgl_amd_mono.common.plan import BLOCKS, THREADS

_IDS: list = []  # the (thread, block) ids of the layer being traced, innermost last


def tid():
    return _IDS[-1][0] if _IDS else fx.thread_idx.x


def bid():
    return _IDS[-1][1] if _IDS else fx.block_idx.x


def _opaque(v, asm, cons):
    return fx.Int32(
        llvm.inline_asm(
            T.i32, [fx.Int32(v).ir_value()], asm, cons, has_side_effects=True
        )
    )


@contextlib.contextmanager
def layer_ids():
    """Trace the stages inside on ids of their own (at the scope's start)."""
    assert THREADS & (THREADS - 1) == 0 and BLOCKS & (BLOCKS - 1) == 0
    t = _opaque(fx.thread_idx.x, "v_mov_b32 $0, $1", "=v,v") & (THREADS - 1)
    b = _opaque(fx.block_idx.x, "s_mov_b32 $0, $1", "=s,s") & (BLOCKS - 1)
    _IDS.append((t, b))
    try:
        yield
    finally:
        _IDS.pop()
