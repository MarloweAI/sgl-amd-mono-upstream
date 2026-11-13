# SPDX-License-Identifier: Apache-2.0
"""The head's input on gfx950 (``combine_norm.hip``): a layer's owed mHC post, its combine
and SGLang's RMSNorm in one launch (``fused.post_combine_norm``). Built with the flags aiter
builds its RMSNorm module with, so the norm's sum and rsqrt are its bits."""

from __future__ import annotations

import functools
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# aiter's module_rmsnorm_quant (optCompilerConfig.json and jit/core.py)
FLAGS = ("-ffast-math", "-fgpu-flush-denormals-to-zero")


@functools.cache
def ops():
    """``torch.ops`` of ``combine_norm.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    return load("sgl_amd_mono_dsv41_norm", sources, "gfx950", FLAGS)
