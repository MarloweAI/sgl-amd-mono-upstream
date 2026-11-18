# SPDX-License-Identifier: Apache-2.0
"""Eager per-layer comparison against the standard path.

Enabled by ``SGLANG_AMD_MONO_CHECK=1``. Each layer's K1 and K2 log
``monokernel check layer <i> (<k1|k2>, <rows> rows):`` and, when a spec bound
is missed, ``monokernel check violation:``.
"""

from __future__ import annotations

import logging

from sgl_amd_mono.qwen35_a17b.check import _metrics

logger = logging.getLogger(__name__)

__all__ = ["_metrics", "log_check"]

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
}


def log_check(layer: int, kind: str, rows: int, values: dict[str, float]) -> None:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.qwen35_a17b.fp8.spec import NUMERICS

    if get_parallel().tp_rank != 0:
        return
    logger.info(
        "monokernel check layer %d (%s, %d rows): %s",
        layer,
        kind,
        rows,
        " ".join(f"{k}={v:.6g}" for k, v in values.items()),
    )
    for name, bound in NUMERICS.bounds.items():
        if not name.startswith(kind + "."):
            continue
        got = values.get(name)
        if got is None:
            continue
        ok = got >= bound if name.endswith(".cosine") else got <= bound
        if not ok:
            logger.error(
                "monokernel check violation: layer %d %s = %.6g, bound %g",
                layer,
                name,
                got,
                bound,
            )
