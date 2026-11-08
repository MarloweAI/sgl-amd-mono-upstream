# SPDX-License-Identifier: Apache-2.0
"""The DSpark draft's committed KV after main_proj in one launch (``inject_kv.hip``): the main
norm, the MXFP8 quant, the stacked wkv GEMM and every stage's SWA store, bitwise SGLang's
chain where aiter runs the stacked wkv as 5 fused splits of K 1024 (5 to 8 rows). Built with
the flags aiter builds its RMSNorm with, so the norm's sum and rsqrt are its bits."""

from __future__ import annotations

import functools
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# aiter's module_rmsnorm_quant (optCompilerConfig.json and jit/core.py)
FLAGS = ("-ffast-math", "-fgpu-flush-denormals-to-zero")
MAX_ROWS, MIN_ROWS = 8, 5  # aiter's 5-split K 1024 packed config (M_LEQ_8)
MAX_STAGES = 4
STATE_INTS = 1024  # inject_kv.hip's tagged scales and the CTAs' launch counts
WORK_BYTES = MAX_STAGES * 32 * 5 * 64 * 4 * 8  # the splits' tagged partials


def buffers(device, stages: int = MAX_STAGES):
    """The launch's scratch, zeroed once and kept: the kv rows, the state ints, the work bytes."""
    import torch

    return (
        torch.zeros(MAX_ROWS, stages * 512, dtype=torch.bfloat16, device=device),
        torch.zeros(STATE_INTS, dtype=torch.int32, device=device),
        torch.zeros(WORK_BYTES, dtype=torch.uint8, device=device),
    )


@functools.cache
def ops():
    """``torch.ops`` of ``inject_kv.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    return load("sgl_amd_mono_dsv41_inject_kv", sources, "gfx950", FLAGS)
