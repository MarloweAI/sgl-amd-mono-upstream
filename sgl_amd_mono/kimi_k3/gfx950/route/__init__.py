# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's MoE routing at 48-180 rows in one launch (``route_sort.hip``): the top-16 of the router's sigmoid +
bias scores, aiter's BM = 16 aux sort of the routes and the zeroed stage-2 output, for aiter's fused_moe to take
through ``aiter.presorted_moe.register_presorted_moe`` in place of its own top-k and sort launches.

The outputs are bit for bit those of the launches it replaces (aiter's ``topk_reg_kernel``, then the sort
``fused_moe`` picks: the one-CTA ``sort_quant_kernel_impl`` below ``OPUS_FROM_ROWS`` rows, the two Opus multi-phase
launches from there), with their buffer sizes and pad words; below ``OPUS_FROM_ROWS`` an expert's routes are in
token order where aiter's one-CTA sort leaves them in an order that varies run to run."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
E, TOPK, LAT, BM = 896, 16, 3584, 16
MIN_ROWS, MAX_ROWS = 48, 180
# aiter's _aux_uses_opus at block_m 16: Opus once 3 x rows x 16 >= 4 x 896
OPUS_FROM_ROWS = 75


@functools.cache
def ops():
    """``torch.ops`` of ``route_sort.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    # aiter's JIT flushes f32 denormals; the sigmoid matches its top-k only with the same flag
    return load(
        "sgl_amd_mono_k3_route", sources, "gfx950", ("-fgpu-flush-denormals-to-zero",)
    )


def sorted_rows(m: int) -> int:
    """The sorted-row capacity aiter's own sort allocates for m rows."""
    p = m * TOPK
    if m >= OPUS_FROM_ROWS:  # _moe_sorting_impl: max_num_tokens_padded in whole blocks
        return (p + E * BM - TOPK + BM - 1) // BM * BM
    return (p + min(E, p) * (BM - 1) + BM - 1) // BM * BM  # aux_sort_capacity


def pad_id(m: int) -> int:
    """The sorted-id word of a pad row: Opus's (topk << 24) | m, the one-CTA sort's m."""
    return (TOPK << 24) | m if m >= OPUS_FROM_ROWS else m


@functools.cache
def _counter(device: torch.device) -> torch.Tensor:
    # the launch's done-count; it leaves it 0. Made outside graph capture (SGLang warms up eagerly first).
    return torch.zeros(1, dtype=torch.int32, device=device)


def route_sort(
    logits: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    scale: float = 1.0,
    valid: torch.Tensor | None = None,
):
    """``logits`` [m, 896] f32 (unit column stride), ``bias`` [896] f32, ``out`` [m, 3584] bf16 (stage 2's
    output; zeroed here), ``valid`` (or None) int32 [1] on the GPU: the rows from it on are padding, routed nowhere
    (ids -1, weights 0), as SGLang masks them. Returns (topk_weights [m, 16] f32, topk_ids [m, 16] i32, the presort dict for
    ``register_presorted_moe``)."""
    m = logits.shape[0]
    dev = logits.device
    ms = sorted_rows(m)
    i32 = torch.int32
    ids = torch.empty(m, TOPK, dtype=i32, device=dev)
    w = torch.empty(m, TOPK, dtype=torch.float32, device=dev)
    aux = dict(
        sorted_token_ids=torch.empty(ms, dtype=i32, device=dev),
        sorted_weights=torch.empty(ms, dtype=torch.float32, device=dev),
        sorted_expert_ids=torch.empty(ms // BM, dtype=i32, device=dev),
        num_valid_ids=torch.empty(2, dtype=i32, device=dev),
        m_indices=torch.empty(ms, dtype=i32, device=dev),
        reverse_sorted=torch.empty(m * TOPK, dtype=i32, device=dev),
    )
    ops().route_sort(
        logits,
        bias,
        float(scale),
        ids,
        w,
        aux["sorted_token_ids"],
        aux["sorted_weights"],
        aux["sorted_expert_ids"],
        aux["num_valid_ids"],
        aux["m_indices"],
        aux["reverse_sorted"],
        out,
        _counter(dev),
        pad_id(m),
        valid,
    )
    aux.update(
        topk_ids_ptr=ids.data_ptr(),
        topk_weights_ptr=w.data_ptr(),
        M=m,
        block_m=BM,
        num_experts=E,
        topk=TOPK,
        moe_buf_zeroed=True,
        moe_buf_ptr=out.data_ptr(),
    )
    return w, ids, aux


SH = 768  # the shared experts' intermediate a rank


def route_sort_shared(
    logits: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    gate_up: torch.Tensor,
    routed_in: torch.Tensor,
    beta: float,
    linear_beta: float,
    scale: float = 1.0,
    valid: torch.Tensor | None = None,
    act: torch.Tensor | None = None,
    routed: torch.Tensor | None = None,
):
    """``route_sort`` plus, on its own CTAs, the shared experts' SiTU of ``gate_up`` [m, 1536] f32 (the front's
    gate | up columns) into ``act`` [m, 768] bf16, and ``routed_in`` [m, 3584] f32 cast to bf16: SGLang's
    situ_and_mul launch with its cast, bit for bit. Returns (topk_weights, topk_ids, presort dict, act, routed
    bf16; ``act`` / ``routed`` when given)."""
    m = logits.shape[0]
    dev = logits.device
    ms = sorted_rows(m)
    i32 = torch.int32
    ids = torch.empty(m, TOPK, dtype=i32, device=dev)
    w = torch.empty(m, TOPK, dtype=torch.float32, device=dev)
    if act is None:
        act = torch.empty(m, SH, dtype=torch.bfloat16, device=dev)
    if routed is None:
        routed = torch.empty(m, LAT, dtype=torch.bfloat16, device=dev)
    aux = dict(
        sorted_token_ids=torch.empty(ms, dtype=i32, device=dev),
        sorted_weights=torch.empty(ms, dtype=torch.float32, device=dev),
        sorted_expert_ids=torch.empty(ms // BM, dtype=i32, device=dev),
        num_valid_ids=torch.empty(2, dtype=i32, device=dev),
        m_indices=torch.empty(ms, dtype=i32, device=dev),
        reverse_sorted=torch.empty(m * TOPK, dtype=i32, device=dev),
    )
    ops().route_sort_shared(
        logits,
        bias,
        float(scale),
        ids,
        w,
        aux["sorted_token_ids"],
        aux["sorted_weights"],
        aux["sorted_expert_ids"],
        aux["num_valid_ids"],
        aux["m_indices"],
        aux["reverse_sorted"],
        out,
        _counter(dev),
        pad_id(m),
        valid,
        gate_up,
        act,
        routed_in,
        routed,
        float(beta),
        float(linear_beta),
    )
    aux.update(
        topk_ids_ptr=ids.data_ptr(),
        topk_weights_ptr=w.data_ptr(),
        M=m,
        block_m=BM,
        num_experts=E,
        topk=TOPK,
        moe_buf_zeroed=True,
        moe_buf_ptr=out.data_ptr(),
    )
    return w, ids, aux, act, routed
