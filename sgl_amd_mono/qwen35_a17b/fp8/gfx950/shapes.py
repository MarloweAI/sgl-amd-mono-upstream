# SPDX-License-Identifier: Apache-2.0
"""Per-rank block-FP8 shapes.

TP4 is the ``layout`` module's constants, unchanged. TP8 is a separate object
so a TP4 launch still specializes on those constants.
"""

from __future__ import annotations

from types import SimpleNamespace

from sgl_amd_mono.qwen35_a17b.fp8.gfx950 import layout as tp4
from sgl_amd_mono.qwen35_a17b.layout import (
    GDN_HEAD,
    GDN_K_HEADS,
    GDN_V_HEADS,
    HEAD_DIM,
    HIDDEN,
    MOE_INTERMEDIATE,
    QUERY_HEADS,
)

# Two KV heads do not divide TP4 or TP8, so each rank keeps one replicated head.
_LOCAL_KV_HEADS = 1


def _qkv_n(q_heads: int, kv_heads: int) -> int:
    # q, the attention output gate, k, and v.
    return (q_heads * 2 + kv_heads * 2) * HEAD_DIM


def _from_counts(tp: int, q_heads: int, k_heads: int, v_heads: int, intermediate: int):
    return SimpleNamespace(
        tp=tp,
        q_heads=q_heads,
        kv_heads=_LOCAL_KV_HEADS,
        gdn_k_heads=k_heads,
        gdn_v_heads=v_heads,
        gdn_head=GDN_HEAD,
        hidden=HIDDEN,
        intermediate=intermediate,
        w13_n=intermediate * 2,
        qkvz_n=k_heads * GDN_HEAD * 2 + v_heads * GDN_HEAD * 2,
        ba_n=v_heads * 2,
        conv_dim=k_heads * GDN_HEAD * 2 + v_heads * GDN_HEAD,
        o_k=v_heads * GDN_HEAD,
        qkv_n=_qkv_n(q_heads, _LOCAL_KV_HEADS),
        block=tp4.BLOCK,
        experts=tp4.EXPERTS,
        conv_kernel=tp4.CONV_KERNEL,
        fp8_max=tp4.FP8_MAX,
        softplus_threshold=tp4.SOFTPLUS_THRESHOLD,
    )


def for_tp(tp: int):
    """The rank layout for ``tp``. TP4 reads ``layout.py``; it is not recomputed."""
    if tp == 4:
        return _from_counts(
            tp4.TP,
            tp4.Q_HEADS,
            tp4.GDN_K_HEADS,
            tp4.GDN_V_HEADS,
            tp4.INTERMEDIATE,
        )
    if tp == 8:
        return _from_counts(
            8,
            QUERY_HEADS // 8,
            GDN_K_HEADS // 8,
            GDN_V_HEADS // 8,
            MOE_INTERMEDIATE // 8,
        )
    raise ValueError(f"Qwen3.5 block-FP8 monokernel supports TP4 and TP8, got TP{tp}")


TP4 = for_tp(4)
TP8 = for_tp(8)
BY_O_K = {TP4.o_k: TP4, TP8.o_k: TP8}
TP_SIZES = (TP4.tp, TP8.tp)
