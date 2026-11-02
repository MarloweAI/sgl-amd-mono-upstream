# SPDX-License-Identifier: Apache-2.0
"""Every DSpark draft stage's SWA KV of a verify's committed target hidden in one
launch (``inject.hip``), SGLang's ``fused_k_norm_rope_flashmla`` bytes per stage."""

from __future__ import annotations

import functools
from pathlib import Path

_HERE = Path(__file__).resolve().parent
MAX_STAGES = 4


@functools.cache
def ops():
    """``torch.ops`` of ``inject.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    return load("sgl_amd_mono_dsv41_inject", sources, "gfx950")
