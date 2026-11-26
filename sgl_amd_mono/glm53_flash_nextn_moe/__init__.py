# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's NEXTN draft MoE at TP2 (E 288, top-8, D 4096, per-rank I 1024; FP8 E4M3 weights with an f32
scale per 128 x 128 block, activations FP8 with an f32 scale per token and 128 columns) in few launches.

``front`` (``front.hip``, one launch): the router GEMM, the biased sigmoid top-8, the BM = 32 sort of the routes with
the shared expert as slot 288, the FP8 per-128 activation quant and the zeroed output (glm53_flash_moe's front with
its own quant). ``experts`` (``experts.py``, one FlyDSL launch): on a BM = 32 sort of the routes (aiter's ``moe_sorting`` layout), the
gate+up GEMM with the clamped SiLU and the FP8 per-128 intermediate (G1), then the down GEMM with the route weight
and the bf16 atomic combine into the zeroed output (G2)."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
E, TOPK, D, I, BM = 288, 8, 4096, 1024, 32
MAX_ROWS = 256
FP8_MAX = 448.0


def capacity(m: int, k: int = TOPK, ne: int = E) -> int:
    """Sorted rows for m rows of k routes over ne experts: every used expert pads its last block by fewer than BM
    rows."""
    p = m * k
    return (p + min(ne, p) * (BM - 1) + BM - 1) // BM * BM


@functools.cache
def ops():
    """``torch.ops`` of ``front.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    # aiter's JIT flushes f32 denormals; the sigmoid matches its top-k only with the same flag
    return load("sgl_amd_mono_g53f_nextn_front", sources, "gfx950", ("-fgpu-flush-denormals-to-zero",))


_COUNTER_WORDS = 32 * 512  # a 128 B line per counter, more than front.hip uses


@functools.cache
def _counters(device: torch.device) -> torch.Tensor:
    # every counter is 0 between launches. Made outside graph capture (SGLang warms up eagerly first).
    return torch.zeros(_COUNTER_WORDS, dtype=torch.int32, device=device)


def front(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    valid: torch.Tensor | None = None,
    given: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """x [m, 4096] bf16, gate_w [288, 4096] bf16, bias [288] bf16, ``scale`` folded into the routed weights,
    ``valid`` (or None) int32 [1]: rows from it on are padding and routed nowhere. Every valid row also routes to
    the shared expert (288, weight 1). ``given`` (ids [m, 8] i32, weights [m, 8] f32), for measurements with fixed
    routes: the top-k runs as always, then the routes written are these. Returns the outputs by name."""
    m = x.shape[0]
    k = TOPK + 1
    dev = x.device
    cap = capacity(m, k, E + 1)
    r = (m + 15) // 16
    i32, f32 = torch.int32, torch.float32
    out = dict(
        topk_ids=torch.empty(m, k, dtype=i32, device=dev),
        topk_weights=torch.empty(m, k, dtype=f32, device=dev),
        sorted_ids=torch.empty(cap, dtype=i32, device=dev),
        sorted_weights=torch.empty(cap, dtype=f32, device=dev),
        sorted_expert_ids=torch.empty(cap // BM, dtype=i32, device=dev),
        num_valid_ids=torch.empty(2, dtype=i32, device=dev),
        tile_target=torch.empty(MAX_ROWS // 16, dtype=i32, device=dev),
        a1=torch.empty(m, D, dtype=torch.float8_e4m3fn, device=dev),
        a1_scale=torch.empty(m, D // 128, dtype=f32, device=dev),
        moe_buf=torch.empty(m, D, dtype=torch.bfloat16, device=dev),
        logits=torch.empty(r * 16, E, dtype=f32, device=dev),
    )
    partial = torch.empty(4 * r * 16 * E, dtype=f32, device=dev)
    ops().front(
        x, gate_w, bias, float(scale), valid, partial, out["logits"], out["topk_ids"], out["topk_weights"],
        out["sorted_ids"], out["sorted_weights"], out["sorted_expert_ids"], out["num_valid_ids"], out["tile_target"],
        out["a1"], out["a1_scale"], out["moe_buf"], _counters(dev), k,
        given[0] if given is not None else None, given[1] if given is not None else None,
    )  # fmt: skip
    return out


def moe(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    swiglu_limit: float,
    valid: torch.Tensor | None = None,
    given: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """The routed and shared experts' sum [m, 4096] bf16 (before the TP reduction) in two launches: ``front``, then
    ``experts.experts`` with the shared expert as route 288 (``shared``: its w13 / w2 E4M3 shuffled as the routed
    ones, and their f32 block scales)."""
    from .experts import experts

    f = front(x, gate_w, bias, scale, valid=valid, given=given)
    return experts(
        f["a1"], f["a1_scale"], f["sorted_ids"], f["sorted_weights"], f["sorted_expert_ids"], f["num_valid_ids"],
        w13, w2, w13_scale, w2_scale, shared, f["moe_buf"], swiglu_limit, BM,
    )  # fmt: skip
