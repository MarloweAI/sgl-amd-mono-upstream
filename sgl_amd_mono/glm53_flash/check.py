# SPDX-License-Identifier: Apache-2.0
"""SGLANG_AMD_MONO_CHECK=1 (eager only): each covered layer also runs the normal
layer on its input, on the caches as they were, and checks the kernel's outputs
against it: the written streams, and the state or cache rows the layer writes, each
against its ``spec.NUMERICS`` bound. Every check is logged; a miss is logged as a
violation."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/layer_boundary/residual/batch.py": ("take_output", "set_written"),
    "srt/mem_cache/memory_pool.py": (
        "mamba2_layer_cache",
        "conv",
        "temporal",
        "get_key_buffer",
        "get_index_k_with_scale_buffer",
        "get_compress_tail_buffers",
    ),
    "srt/model_executor/forward_batch_info.py": ("out_cache_loc", "req_pool_indices"),
    "srt/model_executor/forward_context.py": (
        "get_req_to_token_pool",
        "get_token_to_kv_pool",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"'),
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


def _regions(L, forward_batch, S: int, state_idx) -> dict:
    """The cache entries the layer writes this step: name -> (tensor, index)."""
    from sglang.srt.model_executor.forward_context import (
        get_req_to_token_pool,
        get_token_to_kv_pool,
    )

    if L.kda:
        cache = get_req_to_token_pool().mamba2_layer_cache(L.i)
        sel = state_idx[state_idx > 0].long()
        return {"conv": (cache.conv[0], sel), "state": (cache.temporal, sel)}
    pool = get_token_to_kv_pool()
    loc = forward_batch.out_cache_loc[:S]
    valid = loc != 0
    slots = loc[valid].long()
    kv = pool.get_key_buffer(L.i)
    tail_k, tail_g = pool.get_compress_tail_buffers(L.i)
    req = forward_batch.req_pool_indices[:S][valid].long()
    return {
        "kv": (kv.view(kv.shape[0], -1), slots),
        "index": (
            pool.get_index_k_with_scale_buffer(L.i),
            (slots // PAGE_SIZE).unique(),
        ),
        "tail_k": (tail_k, req),
        "tail_g": (tail_g, req),
    }


def reference(L, positions, streams_in, forward_batch, S: int, state_idx) -> dict:
    """The normal layer's outputs on ``streams_in``; the caches are restored after."""
    from sglang.srt.layers.layer_boundary.residual.batch import set_written, take_output
    from sglang.srt.utils.common import BumpAllocator

    regions = _regions(L, forward_batch, S, state_idx)
    saved = {name: t[idx].clone() for name, (t, idx) in regions.items()}
    hidden = set_written(streams_in, forward_batch)
    zero = BumpAllocator(buffer_size=4, dtype=torch.float32, device=streams_in.device)
    out, _ = L.module(
        positions, hidden, forward_batch, zero, None, prev_topk_indices=None
    )
    ref = {"streams": take_output(out, forward_batch).clone()}
    for name, (t, idx) in regions.items():
        ref[name] = t[idx].clone()
        t[idx] = saved[name]
    return ref


def compare(
    L, ref: dict, streams: torch.Tensor, positions, forward_batch, S: int, state_idx
) -> None:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.glm53_flash.spec import NUMERICS

    if get_parallel().tp_rank != 0:
        return
    kind = "kda" if L.kda else "dsa"
    outputs = _outputs("streams", streams, ref["streams"])
    for name, (t, idx) in _regions(L, forward_batch, S, state_idx).items():
        outputs.update(_outputs(name, t[idx], ref[name]))
    values = {
        f"{kind}.{out}.{metric}": v
        for out, ms in outputs.items()
        for metric, v in ms.items()
    }
    pos = positions[:S]
    logger.info(
        "monokernel check layer %d (%s, %d rows, positions %d..%d): %s",
        L.i,
        kind,
        S,
        pos.min().item(),
        pos.max().item(),
        " ".join(f"{k}={v:.6g}" for k, v in values.items()),
    )
    for name, bound in NUMERICS.bounds.items():
        v = values.get(name)
        if v is None:
            continue
        ok = v >= bound if name.endswith(".cosine") else v <= bound
        if not ok:
            logger.error(
                "monokernel check violation: layer %d %s = %.6g, bound %g",
                L.i,
                name,
                v,
                bound,
            )
