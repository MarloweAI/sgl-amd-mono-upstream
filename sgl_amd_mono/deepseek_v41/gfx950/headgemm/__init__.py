# SPDX-License-Identifier: Apache-2.0
"""The target's vocab projection of a verify's rows on gfx950 (``headgemm.hip``): bf16 MFMAs,
fp32 accumulation, the shard's bf16 logits."""

from __future__ import annotations

import functools
from pathlib import Path

_HERE = Path(__file__).resolve().parent


@functools.cache
def ops():
    """``torch.ops`` of ``headgemm.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    return load("sgl_amd_mono_dsv41_headgemm", sources, "gfx950")
