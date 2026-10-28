# SPDX-License-Identifier: Apache-2.0
"""SGLANG_AMD_MONO_CHECK=1 (eager only): each covered layer also runs SGLang's
layer on its input, on the caches as they were, and checks the kernel's outputs
against it: the hidden stream it writes, the cache rows the layer writes, and on
an indexer layer the top-k it selects, each against its ``spec.NUMERICS`` bound.
Every check is logged; a miss is logged as a violation.

A dense layer's MLP is also checked against an f32 MLP on the launch's own FFN input
(``dense``), and the tail's logits against SGLang's final norm, LM head and
all-gather and against an f32 tail on the same rows (``tail``). SGLang's own
distance to the f32 references is logged alongside.

On an indexer layer the index keys the step writes stay in the cache as SGLang
writes them, with their scales set to POISON_SCALE. A launch must score those keys
from its rows' pairs; one that reads the cache drops each of them that SGLang
selects with a positive score from its top-k (``layer.topk_own``)."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/layer_boundary/residual/batch.py": (
        "fold",
        "take_output",
        "set_written",
    ),
    "srt/layers/moe/topk.py": ("topk_ids", "topk_weights"),
    "srt/mem_cache/memory_pool.py": ("get_key_buffer", "get_index_k_with_scale_buffer"),
    "srt/model_executor/forward_batch_info.py": ("out_cache_loc",),
    "srt/model_executor/forward_context.py": ("get_token_to_kv_pool",),
    "srt/models/deepseek_v2.py": (
        "mlp",
        "gate",
        "topk",
        "e_score_correction_bias",
        "self_attn",
        "o_proj",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"'),
    "srt/distributed/communication_op.py": (
        "tensor_model_parallel_all_reduce",
        "tensor_model_parallel_all_gather",
    ),
    "srt/layers/linear.py": ("reduce_results",),
    "srt/utils/common.py": ("BumpAllocator",),
}

PAGE_TOKENS = 64  # tokens an index-cache page covers
INDEX_VALUE_BYTES = PAGE_TOKENS * 128  # an index page's FP8 values; f32 scales follow
# the kernel scales a key's head-weighted ReLU sum: a positive sum then scores below
# any top-k threshold; finite, so a zero sum scores 0, not NaN
POISON_SCALE = -1e30
SHARED_EXPERT = 256  # SGLang's id for the fused shared expert
TOP_K, ROUTE_SCALE = 8, 2.5


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    if a.numel() == 0:
        return {}
    return {
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
        "rel": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "max_abs": (a - b).abs().max().item(),
    }


def _index_keys(page: torch.Tensor):
    """An index page's keys as scored (FP8 values times their token's power-of-two
    scale) and its scales. A value byte b of the 16 x 16 tiles is token
    (b // 2048) * 16 + (b % 256) // 16."""
    b = torch.arange(INDEX_VALUE_BYTES, device=page.device)
    token = (b // 2048) * 16 + (b % 256) // 16
    values = page[:, :INDEX_VALUE_BYTES].contiguous().view(torch.float8_e4m3fn).float()
    scales = page[:, INDEX_VALUE_BYTES:].contiguous().view(torch.float32)
    return values * scales[:, token], scales


def _outputs(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, dict]:
    if name != "index":
        return {name: _metrics(a, b)}
    # a key whose amax sits at a power of two may take the other scale and values
    (ka, sa), (kb, sb) = _index_keys(a), _index_keys(b)
    return {"index": _metrics(ka, kb), "index_scale": _metrics(sa, sb)}


def poison_new_index_scales(pages: torch.Tensor, slots: torch.Tensor) -> None:
    """Set the cached scales of the index keys at ``slots`` (this step's, 0 = pad)
    to POISON_SCALE in the uint8 index pages."""
    slots = slots[slots != 0].long()
    scales = pages.view(pages.shape[0], -1)[:, INDEX_VALUE_BYTES:].view(torch.float32)
    scales[slots // PAGE_TOKENS, slots % PAGE_TOKENS] = POISON_SCALE


def launch_scores(runtime, rows: int) -> torch.Tensor:
    """The launch's index scores as f32 bits, int32 [rows, index_max_seq]: the value
    words of the (value, tag) pairs in the runtime's scratch."""
    n = runtime.key.index_max_seq
    at = runtime.sc["index_scores"]
    pairs = runtime.scratch[at : at + rows * n * 8].view(torch.int32)
    return pairs.view(rows, n, 2)[..., 0]


def _radix_keys(bits: torch.Tensor) -> torch.Tensor:
    """The kernel's order of f32 score bits (``radix_key``): larger keys rank first."""
    b = bits.to(torch.int64)
    return torch.where(b >= 0, b + (1 << 31), ~b & 0xFFFFFFFF)


def _exact_topk_outputs(step, scores: torch.Tensor, S: int) -> dict:
    """Per row past TOPK keys, how many keys of the exact top-k of the launch's own
    scores (score descending, then position) its selection (``step.topk``) misses."""
    from sgl_amd_mono.glm53.gfx950.runner import TOPK

    bounds = step.indptr[: S + 1].tolist()
    picked = step.topk.tolist()
    missed = 0
    for s in range(S):
        n = int(step.positions[s]) + 1
        if n <= TOPK:
            continue
        keys = _radix_keys(scores[s, :n])
        pos = torch.sort(keys, descending=True, stable=True).indices[:TOPK]
        pages = step.page_table[int(step.req_ids[s]), pos // PAGE_TOKENS].long()
        want = set((pages * PAGE_TOKENS + pos % PAGE_TOKENS).tolist())
        missed = max(missed, len(want - set(picked[bounds[s] : bounds[s + 1]])))
    return {"topk_exact": {"max_abs": float(missed)}}


def _topk_outputs(ref_topk: torch.Tensor, step, slots: torch.Tensor, S: int) -> dict:
    """Per row, SGLang's selected KV slots against the kernel's (``step.topk`` at
    the row's ``indptr``): the largest share of SGLang's a row misses, and the most
    of this step's own slots (a verify row's earlier drafts and itself) one misses."""
    bounds = step.indptr[: S + 1].tolist()
    picked = step.topk.tolist()
    own = set(slots[slots != 0].tolist())
    rel, own_missed = 0.0, 0
    for s, row in enumerate(ref_topk[:S].tolist()):
        want = {k for k in row if k >= 0}
        if not want:
            continue
        missed = want - set(picked[bounds[s] : bounds[s + 1]])
        rel = max(rel, len(missed) / len(want))
        own_missed = max(own_missed, len(missed & own))
    return {"topk": {"rel": rel}, "topk_own": {"max_abs": float(own_missed)}}


def _regions(L, forward_batch, S: int) -> dict:
    """The cache entries the layer writes this step: name -> (tensor, index)."""
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    pool = get_token_to_kv_pool()
    loc = forward_batch.out_cache_loc[:S]
    slots = loc[loc != 0].long()
    kv = pool.get_key_buffer(L.i)
    regions = {"kv": (kv.view(kv.shape[0], -1), slots)}
    if L.w.index is not None:
        regions["index"] = (
            pool.get_index_k_with_scale_buffer(L.i),
            (slots // PAGE_TOKENS).unique(),
        )
    return regions


def _f32_routing(mlp):
    """A top-k hook that routes in f32 as the model defines it and the kernel does
    (SGLang's AITER top-k adds the ~34 correction bias in bf16): the 8 experts of
    the largest sigmoid + bias, weighted by their sigmoid over its sum, times 2.5."""

    seen = {}

    def gate_in(mod, args):
        seen["x"] = args[0]

    def route(mod, args, out):
        x = seen.pop("x")
        scores = torch.sigmoid(x.float() @ mlp.gate.weight.float().t())
        ids = (
            (scores + mlp.gate.e_score_correction_bias.float()).topk(TOP_K, -1).indices
        )
        raw = scores.gather(1, ids)
        routed = out.topk_ids != SHARED_EXPERT
        new_ids, new_w = out.topk_ids.clone(), out.topk_weights.clone()
        new_ids[routed] = ids.to(new_ids.dtype).reshape(-1)
        new_w[routed] = (
            (raw / raw.sum(-1, keepdim=True) * ROUTE_SCALE).to(new_w.dtype).reshape(-1)
        )
        return out._replace(topk_weights=new_w, topk_ids=new_ids)

    return mlp.register_forward_pre_hook(gate_in), mlp.topk.register_forward_hook(route)


def reference(L, positions, h_in, forward_batch, S: int, prev_topk) -> dict:
    """SGLang's layer on ``h_in`` [S, HIDDEN] (a copy), routed in f32: its output
    stream, its top-k and the cache rows it writes. The KV cache is restored after;
    the index keys keep SGLang's values with POISON_SCALE scales."""
    from sglang.srt.layers.layer_boundary.residual.batch import (
        fold,
        set_written,
        take_output,
    )
    from sglang.srt.utils.common import BumpAllocator

    regions = _regions(L, forward_batch, S)
    kv, kv_slots = regions["kv"]
    saved = kv[kv_slots].clone()
    hidden = set_written(h_in, forward_batch)
    zero = BumpAllocator(buffer_size=4, dtype=torch.float32, device=h_in.device)
    mlp_in, o_proj = [], []
    if L.w.dense:
        hooks = (
            L.module.mlp.register_forward_pre_hook(
                lambda module, args: mlp_in.append(args[0].clone())
            ),
            L.module.self_attn.o_proj.register_forward_hook(
                lambda module, args, out: o_proj.append(out[0].clone())
            ),
        )
    else:
        hooks = _f32_routing(L.module.mlp)
    try:
        out, topk = L.module(
            positions[:S],
            hidden,
            forward_batch,
            zero,
            None,
            None,
            prev_topk_indices=prev_topk,
        )
    finally:
        for hook in hooks:
            hook.remove()
    ref = {
        "streams": take_output(fold(out, forward_batch), forward_batch).clone(),
        "topk": topk,
    }
    if mlp_in:
        ref["mlp_in"], ref["o_proj"] = mlp_in[0][:S], o_proj[0][:S]
    for name, (t, idx) in regions.items():
        ref[name] = t[idx].clone()
    kv[kv_slots] = saved
    if "index" in regions:
        poison_new_index_scales(regions["index"][0], forward_batch.out_cache_loc[:S])
    return ref


def compare(
    L,
    ref: dict,
    streams: torch.Tensor,
    positions,
    forward_batch,
    S: int,
    *,
    step,
    scores: torch.Tensor | None,
) -> None:
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().tp_rank != 0:
        return
    outputs = _outputs("streams", streams, ref["streams"])
    for name, (t, idx) in _regions(L, forward_batch, S).items():
        outputs.update(_outputs(name, t[idx], ref[name]))
    if scores is not None:
        outputs.update(_exact_topk_outputs(step, scores, S))
    # SGLang's layer returns no top-k when no later layer reuses it
    if L.w.index is not None and ref["topk"] is not None:
        slots = forward_batch.out_cache_loc[:S]
        outputs.update(_topk_outputs(ref["topk"], step, slots, S))
    values = {
        f"layer.{out}.{metric}": v
        for out, ms in outputs.items()
        for metric, v in ms.items()
    }
    pos = positions[:S]
    logger.info(
        "monokernel check layer %d (%s, %d rows, positions %d..%d): %s",
        L.i,
        "dense" if L.w.dense else "indexer" if L.w.index is not None else "shared",
        S,
        pos.min().item(),
        pos.max().item(),
        " ".join(f"{k}={v:.6g}" for k, v in values.items()),
    )
    _bounds(f"layer {L.i}", values)


# the outputs each check reports (``<check>.<output>.<metric>``)
REPORTED = {
    "layer": (
        "streams",
        "kv",
        "index",
        "index_scale",
        "topk",
        "topk_own",
        "topk_exact",
    ),
    "dense": ("attn", "mlp_in", "mid", "x", "x_vs_fp32", "stock_vs_fp32", "excess"),
    "head": ("logits", "logits_vs_fp32", "stock_vs_fp32", "normed", "top1"),
}
METRICS = ("cosine", "rel", "max_abs", "agree")


def _bounds(where: str, values: dict) -> None:
    from sgl_amd_mono.glm53.spec import NUMERICS

    for name in values:
        kind, out, metric = name.split(".")
        if out not in REPORTED[kind] or metric not in METRICS:
            raise ValueError(f"the check reports {name}, which REPORTED does not list")
    for name, bound in NUMERICS.bounds.items():
        v = values.get(name)
        if v is None:
            continue
        ok = v >= bound if name.endswith(".cosine") else v <= bound
        if not ok:
            logger.error(
                "monokernel check violation: %s %s = %.6g, bound %g",
                where,
                name,
                v,
                bound,
            )


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def unshuffle(w: torch.Tensor) -> torch.Tensor:
    """A [rows, K] FP8 weight from AITER's (16, 16) layout ([rows / 16, K / 32, 2,
    16 rows, 16 bytes]) back to row-major."""
    n, k = w.shape
    b = w.view(torch.uint8).view(n // 16, k // 32, 2, 16, 16)
    return b.permute(0, 3, 1, 2, 4).reshape(n, k).view(w.dtype)


def _rms(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g.float()


def ffn_input(runtime, rows: int) -> torch.Tensor:
    """The launch's FFN input (its residual after attention, mailbox ``a``), bf16
    [rows, HIDDEN]: the value words of the (packed bf16 pair, tag) pairs."""
    from sgl_amd_mono.glm53.gfx950.config import HIDDEN

    at = runtime.sc["a"]
    pairs = runtime.scratch[at : at + rows * HIDDEN * 4].view(torch.int32)
    return pairs.view(-1, 2)[:, 0].contiguous().view(torch.bfloat16).view(rows, HIDDEN)


def _pairs_f32(runtime, name: str, n: int) -> torch.Tensor:
    """The first ``n`` value words of mailbox ``name``'s (f32, tag) pairs."""
    at = runtime.sc[name]
    return (
        runtime.scratch[at : at + n * 8]
        .view(torch.int32)
        .view(n, 2)[:, 0]
        .view(torch.float32)
    )


def _q128(x: torch.Tensor) -> torch.Tensor:
    """Per-128 FP8 E4M3 (amax / 448) of the last dimension, dequantized, as the
    launch quantizes its MLP's mid."""
    shp = x.shape
    x = x.float().reshape(*shp[:-1], -1, 128)
    sc = x.abs().amax(-1, keepdim=True) / 448.0
    sc = torch.where(sc > 0, sc, torch.ones_like(sc))
    return ((x / sc).to(torch.float8_e4m3fn).float() * sc).reshape(shp)


def slices_f32(w: torch.Tensor, s: torch.Tensor, shuffled: bool) -> torch.Tensor:
    """f32 of FP8 expert-shaped slices [n, rows, K] with their 128 x 128 scales."""
    if shuffled:
        w = torch.stack([unshuffle(t) for t in w])
    s = s.float().repeat_interleave(128, -2).repeat_interleave(128, -1)
    return w.float() * s[:, : w.shape[1], : w.shape[2]]


def dense_parts(runtime, w, S: int) -> dict:
    """A dense launch's MLP from its own mailboxes against f32 on the same inputs, this
    rank's part: its FP8 activations (``xqd``), SGLang-equal per-128 quantization of
    ``a``'s RMSNorm; its mid (``mid``, f32 SiLU(gate) * up) against an f32 gate | up on
    those activations; and the f32 down projection of its mid quantized per 128 as the
    launch does (``down``, to be summed over the ranks)."""
    from sgl_amd_mono.glm53.gfx950.config import HIDDEN, MOE_SLOTS

    n, two_i = w.w_ug.shape[:2]
    inter = two_i // 2
    xq = _pairs_raw_f32(runtime, "xqd", S * HIDDEN).view(S, HIDDEN)
    wug = slices_f32(w.w_ug, w.s_ug, w.shuffled)
    wdn = slices_f32(w.w_dn, w.s_dn, w.shuffled)
    gu = torch.einsum("sk,njk->snj", xq, wug)
    mid_ref = torch.nn.functional.silu(gu[..., :inter]) * gu[..., inter:]
    mid = _pairs_f32(runtime, "mid", S * MOE_SLOTS * inter).view(S, MOE_SLOTS, inter)[
        :, :n
    ]
    down = torch.einsum("sni,nhi->sh", _q128(mid), wdn)
    return {"mid": mid, "mid_ref": mid_ref, "down": down}


def _pairs_raw_f32(runtime, name: str, n: int) -> torch.Tensor:
    """The first ``n`` f32 words of plain mailbox ``name``."""
    at = runtime.sc[name]
    return runtime.scratch[at : at + n * 4].view(torch.float32)


def launch_o_proj(op, S: int) -> torch.Tensor:
    """A launch's o_proj output, f32 [S, HIDDEN]: every rank's bf16 partial in its
    symmetric buffer (the slot of this launch's parity), summed in rank order as the
    launch sums them. Read before the step advances."""
    from sgl_amd_mono.glm53.gfx950.config import HIDDEN

    rt = op.rt
    slot = (int(rt.epoch[0]) * rt.key.launches_per_step + op.layer) & 1
    at = rt.sy["attn"] + slot * rt.sy["_part_stride"]
    n = rt.key.tp * rt.width * HIDDEN
    pairs = rt.peer.bytes[at : at + n * 4].view(torch.int32).view(-1, 2)[:, 0]
    parts = pairs.contiguous().view(torch.bfloat16).view(rt.key.tp, rt.width, HIDDEN)
    total = torch.zeros(S, HIDDEN, device=parts.device)
    for part in parts[:, :S].float():
        total += part
    return total


def dense(L, op, ref: dict, out: torch.Tensor, S: int) -> None:
    """A dense launch's attention: its o_proj output against SGLang's
    (``dense.attn.rel``). Its MLP: its input (the post-attention norm of its FFN input)
    against SGLang's MLP's input in ``ref`` (``dense.mlp_in.rel``), its mid against an
    f32 gate | up on its own FP8 activations (``dense.mid.rel``), its output (output
    minus FFN input) against the f32 down projection of its own quantized mid
    (``dense.x.rel``), and both its output and SGLang's MLP on the same FFN input
    against an f32 MLP (``dense.x_vs_fp32.rel``, ``dense.stock_vs_fp32.rel``; their
    ratio less one is ``dense.excess.rel``). Every rank its shard, summed over the TP
    group."""
    from sglang.srt.distributed.communication_op import (
        tensor_model_parallel_all_reduce,
    )
    from sglang.srt.runtime_context import get_parallel

    runtime, mlp = op.rt, L.module.mlp
    o_ref = ref["o_proj"]
    if not L.module.self_attn.o_proj.reduce_results:
        o_ref = tensor_model_parallel_all_reduce(o_ref.contiguous())
    a = ffn_input(runtime, S)
    parts = dense_parts(runtime, L.w, S)
    x = _rms(a, L.module.post_attention_layernorm.weight)
    wug = slices_f32(L.w.w_ug, L.w.s_ug, L.w.shuffled)
    wdn = slices_f32(L.w.w_dn, L.w.s_dn, L.w.shuffled)
    inter = wug.shape[1] // 2
    gu = torch.einsum("sk,njk->snj", x, wug)
    mid32 = torch.nn.functional.silu(gu[..., :inter]) * gu[..., inter:]
    both = torch.cat(
        [parts["down"], torch.einsum("sni,nhi->sh", mid32, wdn)], 0
    ).contiguous()
    both = tensor_model_parallel_all_reduce(both)
    down, f32 = both[:S], both[S:]
    stock = mlp(x.to(torch.bfloat16))
    # SGLang fuses the dense MLP's TP reduce into the next residual add and norm
    if not mlp.down_proj.reduce_results:
        stock = tensor_model_parallel_all_reduce(stock.contiguous())
    mine = out.float() - a.float()
    values = {
        "dense.attn.rel": _rel(launch_o_proj(op, S), o_ref),
        "dense.mlp_in.rel": _rel(x.to(torch.bfloat16), ref["mlp_in"]),
        "dense.mid.rel": _rel(parts["mid"], parts["mid_ref"]),
        "dense.x.rel": _rel(mine, down),
        "dense.x_vs_fp32.rel": _rel(mine, f32),
        "dense.stock_vs_fp32.rel": _rel(stock, f32),
    }
    values["dense.excess.rel"] = (
        values["dense.x_vs_fp32.rel"] / max(values["dense.stock_vs_fp32.rel"], 1e-30)
        - 1
    )
    if get_parallel().tp_rank != 0:
        return
    logger.info(
        "monokernel check dense MLP layer %d (%d rows): %s",
        L.i,
        S,
        " ".join(f"{k}={v:.6g}" for k, v in values.items()),
    )
    _bounds(f"layer {L.i}", values)


def tail(lm, x: torch.Tensor, logits: torch.Tensor, normed: torch.Tensor) -> None:
    """The tail's logits (and normed rows) against SGLang's final norm, LM head and
    all-gather on the same rows, and against an f32 tail."""
    from sglang.srt.distributed.communication_op import (
        tensor_model_parallel_all_gather,
    )
    from sglang.srt.runtime_context import get_parallel

    w = lm.lm_head.weight
    ref_normed = lm.model.norm(x.clone())
    if isinstance(ref_normed, tuple):
        ref_normed = ref_normed[0]
    ref = tensor_model_parallel_all_gather((ref_normed @ w.t()).contiguous()).float()
    f32 = tensor_model_parallel_all_gather(
        (_rms(x, lm.model.norm.weight) @ w.float().t()).contiguous()
    )
    vocab = logits.shape[-1]
    values = {
        "head.logits.rel": _rel(logits, ref[:, :vocab]),
        "head.logits_vs_fp32.rel": _rel(logits, f32[:, :vocab]),
        "head.stock_vs_fp32.rel": _rel(ref[:, :vocab], f32[:, :vocab]),
        "head.normed.rel": _rel(normed, ref_normed),
        "head.top1.agree": (logits.argmax(-1) == ref[:, :vocab].argmax(-1))
        .float()
        .mean()
        .item(),
    }
    if get_parallel().tp_rank != 0:
        return
    logger.info(
        "monokernel check tail (%d rows): %s",
        x.shape[0],
        " ".join(f"{k}={v:.6g}" for k, v in values.items()),
    )
    _bounds("tail", values)
