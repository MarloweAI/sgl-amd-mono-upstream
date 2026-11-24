# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's decode MoE at TP2 (E 288, top-8, D 4096, per-rank I 1024, MXFP4 a4w4 per 1x32) in few launches.

``front`` (``front.hip``, one launch): the router GEMM, the biased sigmoid top-8, the BM = 32 sort of the routes
(optionally with the shared expert as slot 288), the MXFP4 activation quant and the zeroed output. ``moe_front_aiter``
runs it ahead of aiter's FlyDSL stage 1 / stage 2 on the same sorted metadata."""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
E, TOPK, D, BM = 288, 8, 4096, 32
MAX_ROWS = 256
SCALE_COLS = D // 32
_COUNTER_WORDS = 32 * 512  # a 128 B line per counter, more than front.hip uses


@functools.cache
def ops():
    """``torch.ops`` of ``front.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    # aiter's JIT flushes f32 denormals; the sigmoid matches its top-k only with the same flag
    return load(
        "sgl_amd_mono_g53f_moe_front",
        sources,
        "gfx950",
        ("-fgpu-flush-denormals-to-zero",),
    )


def capacity(m: int, k: int = TOPK) -> int:
    """Sorted rows for m rows of k routes: every used expert pads its last block by fewer than BM rows."""
    p = m * k
    ne = E + (k - TOPK)
    return (p + min(ne, p) * (BM - 1) + BM - 1) // BM * BM


def partial_floats(m: int) -> int:
    """The router's f32 partials: 4 K slices x the rows rounded to 16 x 288."""
    return 4 * ((m + 15) // 16) * 16 * E


@functools.cache
def ops():
    """``torch.ops`` of ``front.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    # aiter's JIT flushes f32 denormals; the sigmoid matches its top-k only with the same flag
    return load(
        "sgl_amd_mono_g53f_moe_front",
        sources,
        "gfx950",
        ("-fgpu-flush-denormals-to-zero",),
    )


def capacity(m: int, k: int = TOPK) -> int:
    """Sorted rows for m rows of k routes: every used expert pads its last block by fewer than BM rows."""
    p = m * k
    ne = E + (k - TOPK)
    return (p + min(ne, p) * (BM - 1) + BM - 1) // BM * BM


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
    shared: bool = False,
    sorted_scale: bool = True,
    given: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """x [m, 4096] bf16, gate_w [288, 4096] bf16, bias [288] bf16, ``scale`` folded into the routed weights,
    ``valid`` (or None) int32 [1]: rows from it on are padding and routed nowhere. ``shared`` appends expert 288
    with weight 1.0 to every valid row and also quantizes x as the shared expert's own quant does (``a1_even``,
    ``a1_even_scale``); ``sorted_scale`` also writes the scales in the sorted, shuffled layout of
    aiter's FlyDSL stage 1. ``given`` (ids [m, 8] i32, weights [m, 8] f32), for measurements with fixed routes: the
    top-k runs as always, then the routes written are these. Returns the outputs by name."""
    m = x.shape[0]
    k = TOPK + int(shared)
    dev = x.device
    cap = capacity(m, k)
    r = (m + 15) // 16
    i32, f32, u8 = torch.int32, torch.float32, torch.uint8
    out = dict(
        topk_ids=torch.empty(m, k, dtype=i32, device=dev),
        topk_weights=torch.empty(m, k, dtype=f32, device=dev),
        sorted_ids=torch.empty(cap, dtype=i32, device=dev),
        sorted_weights=torch.empty(cap, dtype=f32, device=dev),
        sorted_expert_ids=torch.empty(cap // BM, dtype=i32, device=dev),
        num_valid_ids=torch.empty(2, dtype=i32, device=dev),
        tile_target=torch.empty(MAX_ROWS // 16, dtype=i32, device=dev),
        a1=torch.empty(m, D // 2, dtype=u8, device=dev),
        a1_scale=torch.empty(m, SCALE_COLS, dtype=u8, device=dev),
        a1_scale_sorted=(
            torch.empty((cap + 31) // 32 * 32, SCALE_COLS, dtype=u8, device=dev) if sorted_scale else None
        ),
        # the shared expert's activations with its own quant (aiter's dynamic_mxfp4_quant, "even" scales)
        a1_even=torch.empty(m, D // 2, dtype=u8, device=dev) if shared else None,
        a1_even_scale=torch.empty(m, SCALE_COLS, dtype=u8, device=dev) if shared else None,
        moe_buf=torch.empty(m, D, dtype=torch.bfloat16, device=dev),
        logits=torch.empty(r * 16, E, dtype=f32, device=dev),
    )
    partial = torch.empty(partial_floats(m), dtype=f32, device=dev)
    ops().front(
        x,
        gate_w,
        bias,
        float(scale),
        valid,
        partial,
        out["logits"],
        out["topk_ids"],
        out["topk_weights"],
        out["sorted_ids"],
        out["sorted_weights"],
        out["sorted_expert_ids"],
        out["num_valid_ids"],
        out["tile_target"],
        out["a1"],
        out["a1_scale"],
        out["a1_scale_sorted"],
        out["a1_even"],
        out["a1_even_scale"],
        out["moe_buf"],
        _counters(dev),
        k,
        given[0] if given is not None else None,
        given[1] if given is not None else None,
    )
    out["partial"] = partial
    return out


def topk(logits: torch.Tensor, bias: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The front's top-k alone on bf16 logits [m, 288]: (weights [m, 8] f32, ids [m, 8] i32)."""
    m = logits.shape[0]
    ids = torch.empty(m, TOPK, dtype=torch.int32, device=logits.device)
    w = torch.empty(m, TOPK, dtype=torch.float32, device=logits.device)
    ops().topk(logits, bias, float(scale), ids, w)
    return w, ids


def moe_front_aiter(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    swiglu_limit: float,
    valid: torch.Tensor | None = None,
    shared: bool = False,
    fp4_inter: bool = False,
    given: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """The routed MoE (and the shared expert, as slot 288, when ``shared``): ``front`` then aiter's FlyDSL
    stage 1 / stage 2 on its sorted rows and MXFP4 activations. ``fp4_inter``: stage 1 writes the MXFP4
    intermediate itself (its ``_fp4`` variant) in place of the separate quant launch. Returns [m, 4096] bf16."""
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe_2stages
    from aiter.ops.flydsl.moe_common import GateMode

    f = front(x, gate_w, bias, scale, valid=valid, shared=shared, sorted_scale=True, given=given)
    k = TOPK + int(shared)
    return fused_moe_2stages(
        x,
        w13,
        w2,
        k,
        f["sorted_ids"],
        f["sorted_weights"],
        f["sorted_expert_ids"],
        f["num_valid_ids"],
        f["moe_buf"],
        True,
        BM,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        q_dtype_a=dtypes.fp4x2,
        q_dtype_w=w13.dtype,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        topk_ids=f["topk_ids"],
        topk_weights=f["topk_weights"],
        swiglu_limit=swiglu_limit,
        gate_mode=GateMode.SEPARATED.value,
        routing_num_experts=E + int(shared),
        prequantized_a1=f["a1"].view(dtypes.fp4x2),
        prequantized_a1_scale=f["a1_scale_sorted"].view(dtypes.fp8_e8m0),
        _metadata_transform=_fp4_intermediate if fp4_inter else None,
    )


def _fp4_intermediate(metadata):
    """aiter's heuristic FlyDSL metadata with stage 1's MXFP4-output variant of the same tiles."""
    s1 = metadata.stage1
    name = s1.keywords["kernelName"]
    stage1 = functools.partial(s1.func, *s1.args, **{**s1.keywords, "kernelName": name + "_fp4"})
    return dataclasses.replace(metadata, stage1=stage1, fuse_quant="fp4")


def moe_fused(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    swiglu_limit: float,
    shared_w13: torch.Tensor,
    shared_w13_scale: torch.Tensor,
    shared_w2: torch.Tensor,
    shared_w2_scale: torch.Tensor,
    valid: torch.Tensor | None = None,
    given: tuple[torch.Tensor, torch.Tensor] | None = None,
    push: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """The routed experts (weights [288, ...], aiter-shuffled) and the shared expert (its dense layers' own row-major
    weights, uint8 as SGLang holds them: ``shared_*``, as slot 288 of the sort) in two launches:
    ``front`` with the shared slot, then ``experts.experts`` (G1 + G2 in one launch; ``push``: its peer output,
    flags and epoch). Returns [m, 4096] bf16."""
    from .experts import experts

    f = front(x, gate_w, bias, scale, valid=valid, shared=True, sorted_scale=False, given=given)
    return experts(
        f["a1"],
        f["a1_scale"],
        f["a1_even"],
        f["a1_even_scale"],
        f["sorted_ids"],
        f["sorted_weights"],
        f["sorted_expert_ids"],
        f["num_valid_ids"],
        w13,
        w2,
        w13_scale,
        w2_scale,
        shared_w13,
        shared_w13_scale,
        shared_w2,
        shared_w2_scale,
        f["moe_buf"],
        swiglu_limit,
        f["tile_target"],
        push=push,
    )
