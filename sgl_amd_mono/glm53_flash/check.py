# SPDX-License-Identifier: Apache-2.0
"""SGLANG_AMD_MONO_CHECK=1 (eager only): each layer's launch also runs the normal
layer on its input (layer 0: SGLang's embedding of the step's tokens), on the caches
as they were, and checks the kernel's outputs against it: the written streams (the
last layer's contracted), and the state or cache rows the layer writes; a dense
layer's MLP output also against an f32 MLP on the launch's own FFN input. Each value
against its ``spec.NUMERICS`` bound. Every check is logged; a miss is logged as a
violation."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/layernorm/mhc.py": ("hc_contract",),
    "srt/distributed/parallel_state.py": ("device_group",),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "state_layer",
        "replayssm_ring_indices",
        "linear_attn_backend",
        "state_layer_cache",
    ),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "take_output",
        "set_written",
    ),
    # a KDA layer's ring index (its state view's layer)
    "srt/mem_cache/kv/router.py": ("view_layer_id",),
    "srt/mem_cache/memory_pool.py": (
        "conv",
        "temporal",
        "intermediate_ssm",
        "intermediate_conv_window",
        "get_key_buffer",
        "get_index_k_with_scale_buffer",
        "get_compress_tail_buffers",
    ),
    "srt/layers/attention/linear/kda_fold_state.py": (
        "ring_x",
        "ring_a",
        "ring_b",
        "ring_half",
    ),
    "srt/layers/attention/linear/kda_backend.py": ("kda_fold",),
    "srt/model_executor/forward_batch_info.py": ("out_cache_loc", "req_pool_indices"),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_token_to_kv_pool",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
    "srt/utils/common.py": ("BumpAllocator",),
}

PAGE_SIZE = 256  # tokens an index-cache page covers
INDEX_POOLS = 64  # kpool pools an index-cache page holds
INDEX_VALUE_BYTES = INDEX_POOLS * 128  # their FP8 values; one f32 scale a pool follows


def _index_keys(page: torch.Tensor) -> torch.Tensor:
    """An index-cache page's keys as the indexer reads them: each pool's FP8 values
    (AITER's 16-pool x 16-byte tiles) times the pool's power-of-two scale."""
    j = torch.arange(INDEX_VALUE_BYTES, device=page.device)
    pool = (j // 2048) * 16 + (j % 256) // 16
    values = page[:, :INDEX_VALUE_BYTES].contiguous().view(torch.float8_e4m3fn)
    scales = page[:, INDEX_VALUE_BYTES:].contiguous().view(torch.float32)
    return values.float() * scales[:, pool]


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    if a.numel() == 0:
        return {}
    return {
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
        "rel": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "max_abs": (a - b).abs().max().item(),
    }


def _outputs(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, dict]:
    if name != "index":
        return {name: _metrics(a, b)}
    # the dequantized keys: a pool whose amax sits at a power of two may take the
    # next scale with its values requantized, the same keys to FP8 precision
    return {"index": _metrics(_index_keys(a), _index_keys(b))}


def _regions(L, forward_batch, S: int, state_idx, verify: int = 1) -> dict:
    """The cache entries the layer writes this step: name -> (tensor, index). A KDA
    verify step writes its requests' per-draft-row states (SGLang's intermediate
    caches, request r at row r), not the states."""
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )

    if L.kda:
        kda = get_attn_backend().linear_attn_backend
        cache = kda.state_layer_cache(L.i)
        fs = getattr(kda, "kda_fold", None)
        if verify > 1 and fs is not None:
            # fold-in-verify: the folded state and conv window by slot, and the
            # step's ring rows (the verify's raw inputs) by request
            sel = state_idx[state_idx > 0].long().unique()
            i = kda.state_layer(L.i).view_layer_id
            rr = kda.forward_metadata.replayssm_ring_indices[: S // verify].long()
            rr = rr[rr >= 0]
            return {
                "conv": (cache.conv[0], sel),
                "state": (cache.temporal, sel),
                "ring_x": (fs.ring_x[i], rr),
                "ring_a": (fs.ring_a[i], rr),
                "ring_b": (fs.ring_b[i], rr),
                "ring_half": (fs.ring_half[i], rr),
            }
        if verify > 1:
            req = torch.arange(S // verify, device=state_idx.device)
            return {
                "conv": (cache.intermediate_conv_window[0], req),
                "state": (cache.intermediate_ssm, req),
            }
        sel = state_idx[state_idx > 0].long()
        return {"conv": (cache.conv[0], sel), "state": (cache.temporal, sel)}
    pool = get_token_to_kv_pool()
    loc = forward_batch.out_cache_loc[:S]
    valid = loc != 0
    slots = loc[valid].long()
    kv = pool.get_key_buffer(L.i)
    tail_k, tail_g = pool.get_compress_tail_buffers(L.i)
    # a verify step's request rows are its requests' (one a request), the rows' each
    req = forward_batch.req_pool_indices[: S // verify].repeat_interleave(verify)
    req = req[valid].long()
    return {
        "kv": (kv.view(kv.shape[0], -1), slots),
        "index": (
            pool.get_index_k_with_scale_buffer(L.i),
            (slots // PAGE_SIZE).unique(),
        ),
        "tail_k": (tail_k, req),
        "tail_g": (tail_g, req),
    }


def reference(
    L, positions, hidden_in, forward_batch, S: int, state_idx, verify: int = 1
) -> dict:
    """The normal layer's outputs on ``hidden_in`` (the written streams, or layer 0's
    embedded tokens); the caches are restored after."""
    from sglang.srt.layers.layer_boundary.residual.batch import (
        set_written,
        start,
        take_output,
    )
    from sglang.srt.utils.common import BumpAllocator

    regions = _regions(L, forward_batch, S, state_idx, verify)
    saved = {name: t[idx].clone() for name, (t, idx) in regions.items()}
    if L.i == 0:
        start(forward_batch)
        hidden = hidden_in
    else:
        hidden = set_written(hidden_in, forward_batch)
    zero = BumpAllocator(buffer_size=4, dtype=torch.float32, device=hidden_in.device)
    mlp_in = []
    hook = None
    if L.kind == "dense":
        hook = L.module.mlp.register_forward_pre_hook(
            lambda module, args: mlp_in.append(args[0].clone())
        )
    try:
        out, _ = L.module(
            positions, hidden, forward_batch, zero, None, prev_topk_indices=None
        )
    finally:
        if hook is not None:
            hook.remove()
    ref = {"streams": take_output(out, forward_batch).clone()}
    if mlp_in:
        ref["mlp_in"] = mlp_in[0][:S]
    for name, (t, idx) in regions.items():
        ref[name] = t[idx].clone()
        t[idx] = saved[name]
    return ref


def contract(streams: torch.Tensor) -> torch.Tensor:
    """The mean of the four streams, as the last layer's FFN exit hands it on."""
    from sglang.kernels.ops.layernorm.mhc import hc_contract

    from sgl_amd_mono.glm53_flash.gfx950.config import HC_MULT

    return hc_contract(streams, HC_MULT)


def _all_reduce(t: torch.Tensor) -> torch.Tensor:
    from sglang.srt.runtime_context import get_parallel

    torch.distributed.all_reduce(t, group=get_parallel().tp_group.device_group)
    return t


def _dequant(w: torch.Tensor, s: torch.Tensor, shuffled: bool) -> torch.Tensor:
    """A block-FP8 weight (row-major or AITER (16, 16)-shuffled) in f32."""
    n, k = w.shape
    if shuffled:
        w = w.view(torch.uint8).reshape(n // 16, k // 32, 2, 16, 16)
        w = w.permute(0, 3, 1, 2, 4).reshape(n, k).view(torch.float8_e4m3fn)
    scale = s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return w.float() * scale


def ffn_input(L, W: int, S: int) -> torch.Tensor:
    """The launch's RMSNorm'd FFN input rows (bf16), from its scratch."""
    from sgl_amd_mono.glm53_flash.gfx950.config import HIDDEN

    rt = L.ops[W].rt
    a = rt.scratch[rt.sc["a"] :].view(torch.int32)[: S * HIDDEN : 2]
    return a.contiguous().view(torch.bfloat16).view(S, HIDDEN)


def dense_mlp_fp32(L, W: int, S: int) -> torch.Tensor:
    """The dense MLP in f32 on dequantized weights, on the launch's own FFN input,
    summed over the ranks (collective)."""
    from sgl_amd_mono.glm53_flash.gfx950.config import SWIGLU_LIMIT

    w = L.ops[W].w
    a = ffn_input(L, W, S).float()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        gu = a @ _dequant(w.w_ug, w.s_ug, w.shuffled).T
        gate, up = gu.chunk(2, dim=-1)
        gate = gate.clamp(max=SWIGLU_LIMIT)
        up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        x = (F.silu(gate) * up) @ _dequant(w.w_dn, w.s_dn, w.shuffled).T
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return _all_reduce(x)


# the outputs each kind's check reports (``<kind>.<output>.<metric>``)
REPORTED = {
    "kda": ("streams", "conv", "state", "ring_x", "ring_a", "ring_b", "ring_half"),
    "dense": (
        "streams",
        "conv",
        "state",
        "ring_x",
        "ring_a",
        "ring_b",
        "ring_half",
        "x_vs_fp32",
        "mlp_in",
    ),
    "dsa": ("streams", "kv", "index", "tail_k", "tail_g"),
}
METRICS = ("cosine", "rel", "max_abs")


def _report(what: str, values: dict) -> None:
    from sgl_amd_mono.glm53_flash.spec import NUMERICS

    for name in values:
        kind, out, metric = name.split(".")
        if out not in REPORTED[kind] or metric not in METRICS:
            raise ValueError(f"the check reports {name}, which REPORTED does not list")

    logger.info(
        "monokernel check %s: %s",
        what,
        " ".join(f"{k}={v:.6g}" for k, v in values.items()),
    )
    for name, bound in NUMERICS.bounds.items():
        v = values.get(name)
        if v is None:
            continue
        ok = v >= bound if name.endswith(".cosine") else v <= bound
        if not ok:
            logger.error(
                "monokernel check violation: %s %s = %.6g, bound %g",
                what.split(" (")[0],
                name,
                v,
                bound,
            )


def compare(
    L,
    ref: dict,
    streams: torch.Tensor,
    out,
    positions,
    forward_batch,
    S: int,
    state_idx,
    verify: int = 1,
) -> None:
    from sglang.srt.runtime_context import get_parallel

    x32 = dense_mlp_fp32(L, out[0].shape[0], S) if L.kind == "dense" else None
    if get_parallel().tp_rank != 0:
        return
    outputs = _outputs("streams", streams, ref["streams"])
    for name, (t, idx) in _regions(L, forward_batch, S, state_idx, verify).items():
        outputs.update(_outputs(name, t[idx], ref[name]))
    if x32 is not None:
        outputs["x_vs_fp32"] = _metrics(out[0][:S], x32)
        outputs["mlp_in"] = _metrics(ffn_input(L, out[0].shape[0], S), ref["mlp_in"])
    values = {
        f"{L.kind}.{out}.{metric}": v
        for out, ms in outputs.items()
        for metric, v in ms.items()
    }
    pos = positions[:S]
    _report(
        f"layer {L.i} ({L.kind}, {S} rows, positions {pos.min().item()}.."
        f"{pos.max().item()})",
        values,
    )

