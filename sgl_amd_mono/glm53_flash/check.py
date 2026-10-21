# SPDX-License-Identifier: Apache-2.0
"""SGLANG_AMD_MONO_CHECK=1 (eager only): each layer's launch also runs the normal
layer on its input (layer 0: SGLang's embedding of the step's tokens), on the caches
as they were, and checks the kernel's outputs against it: the written streams (the
last layer's contracted), and the state or cache rows the layer writes; a dense
layer's MLP output also against an f32 MLP on the launch's own FFN input. The tail's
logits are checked against SGLang's final norm and logits, and against an f32 tail.
Each value against its ``spec.NUMERICS`` bound. Every check is logged; a miss is
logged as a violation."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/layernorm/mhc.py": ("hc_contract",),
    "srt/distributed/parallel_state.py": ("device_group",),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "take_output",
        "set_written",
    ),
    "srt/layers/logits_processor.py": ("LogitsMetadata", "from_forward_batch"),
    "srt/layers/layernorm.py": ("variance_epsilon",),
    "srt/models/glm5_next.py": ("model", "norm", "lm_head"),
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
    "srt/runtime_context.py": (
        "get_parallel",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
    "srt/utils/common.py": ("BumpAllocator",),
}

PAGE_SIZE = 256  # tokens an index-cache page covers
INDEX_VALUE_BYTES = 64 * 128  # an index page's FP8 values; its f32 scales follow


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
    # both pages in one layout: compare the decoded FP8 values and the f32 scales
    fp8 = torch.float8_e4m3fn
    return {
        "index": _metrics(
            a[:, :INDEX_VALUE_BYTES].contiguous().view(fp8),
            b[:, :INDEX_VALUE_BYTES].contiguous().view(fp8),
        ),
        "index_scale": _metrics(
            a[:, INDEX_VALUE_BYTES:].contiguous().view(torch.float32),
            b[:, INDEX_VALUE_BYTES:].contiguous().view(torch.float32),
        ),
    }


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


def reference(L, positions, hidden_in, forward_batch, S: int, state_idx) -> dict:
    """The normal layer's outputs on ``hidden_in`` (the written streams, or layer 0's
    embedded tokens); the caches are restored after."""
    from sglang.srt.layers.layer_boundary.residual.batch import (
        set_written,
        start,
        take_output,
    )
    from sglang.srt.utils.common import BumpAllocator

    regions = _regions(L, forward_batch, S, state_idx)
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
    "kda": ("streams", "conv", "state"),
    "dense": ("streams", "conv", "state", "x_vs_fp32", "mlp_in"),
    "dsa": ("streams", "kv", "index", "index_scale", "tail_k", "tail_g"),
    "head": ("logits", "logits_vs_fp32"),
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
) -> None:
    from sglang.srt.runtime_context import get_parallel

    x32 = dense_mlp_fp32(L, out[0].shape[0], S) if L.kind == "dense" else None
    if get_parallel().tp_rank != 0:
        return
    outputs = _outputs("streams", streams, ref["streams"])
    for name, (t, idx) in _regions(L, forward_batch, S, state_idx).items():
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


def tail_fp32(owed, g_norm, eps: float, w_head) -> torch.Tensor:
    """The tail in f32 on the last layer's owed (x, residual, post, comb): the four
    streams (post_j x + sum_i comb_ij R_i), their mean, RMSNorm and the logits of
    this rank's LM-head rows."""
    s = owed.x.shape[0]
    x, r, post, comb = (
        t.float() for t in (owed.x, owed.residual, owed.post, owed.comb)
    )
    streams = post.reshape(s, -1, 1) * x[:, None] + torch.einsum(
        "sij,sih->sjh", comb, r
    )
    h = streams.mean(1)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * g_norm.float()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return h @ w_head.float().T
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def tail(lm, own_get_logits, owed, logits, forward_batch, S: int) -> None:
    """The tail's logits against SGLang's final norm and logits on the last layer's
    contracted output, and against an f32 tail: per row, the launch's distance to it
    in excess of SGLang's, relative to SGLang's (collective)."""
    from sglang.srt.layers.logits_processor import LogitsMetadata
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.glm53_flash.spec import normed_rows

    p = get_parallel()
    normed = normed_rows(lm, owed, S)
    owed = type(owed)(*(t[:S] for t in (owed.x, owed.residual, owed.post, owed.comb)))
    meta = LogitsMetadata.from_forward_batch(forward_batch)
    ref = own_get_logits(normed, lm.lm_head, meta).float()
    norm = lm.model.norm
    shard = tail_fp32(owed, norm.weight, norm.variance_epsilon, lm.lm_head.weight)
    parts = [torch.empty_like(shard) for _ in range(p.tp_size)]
    torch.distributed.all_gather(parts, shard, group=p.tp_group.device_group)
    f32 = torch.cat(parts, -1)
    if p.tp_rank != 0:
        return
    got = logits[:S].float()
    values = {f"head.logits.{k}": v for k, v in _metrics(got, ref).items()}
    nf = f32.norm(dim=-1)
    eg, es = (got - f32).norm(dim=-1) / nf, (ref - f32).norm(dim=-1) / nf
    ratio = eg / es.clamp_min(torch.finfo(torch.float32).tiny)
    finite = bool(torch.isfinite(got).all())
    values["head.logits_vs_fp32.rel"] = (
        ratio.max().item() - 1 if finite else float("inf")
    )
    _report(f"tail ({S} rows)", values)
