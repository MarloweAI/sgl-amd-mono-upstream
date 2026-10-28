# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from ROCm/aiter#6173 at 56d6390def (Apache-2.0):
# aiter/ops/flydsl/kernels/glm5_mono/weights.py
"""GLM-5.3's layer tensors as the kernel reads them, from SGLang's loaded modules
(``Glm53Weights``), and the tail's (``Glm53HeadWeights``). Projections, experts,
the embedding and the LM head are SGLang's tensors in place; W_UK and W_UV, the
router, the indexer's weights_proj and its k LayerNorm, and the dense layers' MLPs
in the experts' layout are copies, recorded in ``copies`` for the startup log."""

from __future__ import annotations

from typing import Optional

import torch

from .config import (
    GLM5_GLOBAL_HEADS,
    GLM5_GLOBAL_INTER,
    HIDDEN,
    KV_LORA,
    MOE_SLOTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    V_DIM,
)
from .layout import INDEX_DIM, INDEX_HEADS
from .packing import pack_bf16, pack_fp8

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/deepseek_v2.py": (
        "self_attn",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
        "fused_qkv_a_proj_with_mqa",
        "q_a_layernorm",
        "kv_a_layernorm",
        "q_b_proj",
        "o_proj",
        "w_kc",
        "w_vc",
        "w_scale",
        "indexer",
        "rotary_emb",
        "gate",
        "e_score_correction_bias",
        "experts",
    ),
    "srt/layers/attention/dsa/dsa_indexer.py": (
        "wq_b",
        "wk",
        "weights_proj",
        "k_norm",
        "rotary_emb",
    ),
    "srt/layers/rotary_embedding/base.py": ("cos_sin_cache",),
    "srt/layers/quantization/fp8.py": (
        "weight_scale_inv",
        "aiter_bpreshuffled",
        "is_shuffled",
        "w13_weight",
        "w13_weight_scale_inv",
        "w2_weight",
        "w2_weight_scale_inv",
        "runner",
    ),
    "srt/models/deepseek_v2.py|srt/layers/linear.py": ("gate_up_proj", "down_proj"),
    "srt/layers/vocab_parallel_embedding.py": (
        "shard_indices",
        "org_vocab_start_index",
        "org_vocab_end_index",
        "added_vocab_start_index",
        "added_vocab_end_index",
        "num_org_vocab_padding",
        "enable_tp",
        "use_attn_tp_group",
        "tp_size",
        "output_dtype",
        "quant_method",
        "UnquantizedEmbeddingMethod",
    ),
    "srt/layers/moe/fused_moe_triton/layer.py": ("quant_method",),
    "srt/layers/moe/moe_runner/runner.py": ("runner_backend",),
    "srt/layers/moe/utils.py": ("is_aiter",),
}

N_EXPERTS = 256
# SGLang appends the shared expert to the routed ones
PHYSICAL_EXPERTS = N_EXPERTS + 1
QKV_A = Q_LORA + KV_LORA + PE_DIM


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _need(name: str, t: torch.Tensor, shape: tuple, dtypes: tuple) -> torch.Tensor:
    if tuple(t.shape) != shape or t.dtype not in dtypes or not t.is_contiguous():
        raise ValueError(
            f"needs {name} as contiguous {'/'.join(map(str, dtypes))} {shape}, got "
            f"{t.dtype} {tuple(t.shape)} {tuple(t.stride())}"
        )
    return t


def _fp8(name: str, weight, scale, rows: int, cols: int):
    """An FP8 E4M3 [rows, cols] weight with its f32 128 x 128 block scales."""
    _need(f"{name} weight", weight, (rows, cols), (torch.float8_e4m3fn,))
    if scale is None:
        raise ValueError(f"needs {name} as block FP8; it has no weight_scale_inv")
    _need(
        f"{name} scale", scale, (_cdiv(rows, 128), _cdiv(cols, 128)), (torch.float32,)
    )
    return weight, scale


def _shuffled(linear) -> bool:
    # SGLang sets these only on a linear it rewrote into AITER's (16, 16) layout
    return bool(
        getattr(linear, "aiter_bpreshuffled", False)
        or getattr(linear.weight, "is_shuffled", False)
    )


def _linear(name: str, linear, rows: int, cols: int):
    """A block-FP8 linear as SGLang keeps it after loading: (weight, scales,
    shuffled), shuffled meaning AITER's (16, 16) layout in place."""
    w, sc = _fp8(
        name, linear.weight, getattr(linear, "weight_scale_inv", None), rows, cols
    )
    return w, sc, _shuffled(linear)


def unshuffled_projections(layer) -> list[str]:
    """The layer's FP8 projections that SGLang left row-major: the kernel reads
    every one in place in AITER's (16, 16) layout."""
    attn = layer.self_attn
    linears = {
        "fused_qkv_a_proj_with_mqa": attn.fused_qkv_a_proj_with_mqa,
        "q_b_proj": attn.q_b_proj,
        "o_proj": attn.o_proj,
    }
    if attn.indexer is not None:
        linears["indexer.wq_b"] = attn.indexer.wq_b
        linears["indexer.wk"] = attn.indexer.wk
    if is_dense(layer):
        linears["mlp.gate_up_proj"] = layer.mlp.gate_up_proj
        linears["mlp.down_proj"] = layer.mlp.down_proj
    return [name for name, linear in linears.items() if not _shuffled(linear)]


def dense_reason(layer, tp: int) -> Optional[str]:
    """Why the kernel cannot run this dense layer's MLP as expert slices, or None."""
    mlp, inter = layer.mlp, GLM5_GLOBAL_INTER // tp
    try:
        rows = mlp.gate_up_proj.weight.shape[0]
        _linear("gate_up_proj", mlp.gate_up_proj, rows, HIDDEN)
        _linear("down_proj", mlp.down_proj, HIDDEN, rows // 2)
    except ValueError as e:
        return f"dense MLP: {e}"
    if rows % (2 * inter) or rows // (2 * inter) > MOE_SLOTS:
        return (
            f"dense MLP: needs its {rows // 2} intermediates a rank as at most "
            f"{MOE_SLOTS} slices of {inter}"
        )
    if layer.self_attn.indexer is None:
        return "a dense layer without its indexer"
    return None


def experts_reason(moe) -> Optional[str]:
    """Why the kernel cannot read this MoE's experts in place, or None."""
    experts = getattr(moe, "experts", None)
    w13 = getattr(experts, "w13_weight", None)
    if w13 is None:
        return "no fused experts (w13_weight)"
    if w13.dtype != torch.float8_e4m3fn:
        return f"needs block-FP8 experts, got {w13.dtype}"
    if w13.shape[0] != PHYSICAL_EXPERTS:
        return (
            f"needs the shared expert fused as expert {N_EXPERTS} "
            f"({PHYSICAL_EXPERTS} experts), got {w13.shape[0]}"
        )
    runner = getattr(experts.quant_method, "runner", None)
    if runner is None or not runner.runner_backend.is_aiter():
        return "needs the experts on SGLang's AITER MoE runner, which shuffles them"
    return None


def _uk_uv(attn, heads: int, idx: int, copies: dict):
    """W_UK [H * 512, 192] and W_UV [H * 256, 512] packed from SGLang's per-tensor
    FP8 ``w_kc`` / ``w_vc``, every block scale ``w_scale``: the values SGLang's
    own absorbed attention multiplies by."""
    w_kc, w_vc, w_scale = attn.w_kc, attn.w_vc, attn.w_scale
    if w_kc.dtype != torch.float8_e4m3fn or w_scale is None or w_scale.numel() != 1:
        raise ValueError(
            f"needs w_kc / w_vc as per-tensor FP8 E4M3 with a scalar w_scale, got "
            f"{w_kc.dtype} and {None if w_scale is None else tuple(w_scale.shape)}"
        )
    if tuple(w_kc.shape) != (heads, NOPE_DIM, KV_LORA) or tuple(w_vc.shape) != (
        heads,
        KV_LORA,
        V_DIM,
    ):
        raise ValueError(
            f"needs w_kc [{heads}, {NOPE_DIM}, {KV_LORA}] and w_vc [{heads}, "
            f"{KV_LORA}, {V_DIM}], got {tuple(w_kc.shape)} and {tuple(w_vc.shape)}"
        )
    uk = w_kc.transpose(1, 2).reshape(heads * KV_LORA, NOPE_DIM)
    uv = w_vc.transpose(1, 2).reshape(heads * V_DIM, KV_LORA)
    scale = float(w_scale.float().item())
    dev = uk.device
    w_uk, w_uv = pack_fp8(uk.contiguous()), pack_fp8(uv.contiguous())
    s_uk = torch.full(
        (heads * KV_LORA // 128, NOPE_DIM // 64), scale, dtype=torch.float32, device=dev
    )
    s_uv = torch.full(
        (heads * V_DIM // 128, KV_LORA // 128), scale, dtype=torch.float32, device=dev
    )
    for name, t in (("w_uk", w_uk), ("w_uv", w_uv), ("s_uk", s_uk), ("s_uv", s_uv)):
        copies["W_UK / W_UV packed", idx, name] = t
    return w_uk, s_uk, w_uv, s_uv


def _indexer(attn, idx: int, copies: dict):
    """An indexer layer's pointer table (wk, its scales, weights_proj packed, wq_b,
    its scales, k LayerNorm weight and bias in f32, no timeline) and what it
    points at."""
    ind = attn.indexer
    w_k, s_k, sh_k = _linear("indexer.wk", ind.wk, INDEX_DIM, HIDDEN)
    w_q, s_q, sh_q = _linear("indexer.wq_b", ind.wq_b, INDEX_HEADS * INDEX_DIM, Q_LORA)
    w_w = _need(
        "indexer.weights_proj",
        ind.weights_proj.weight,
        (INDEX_HEADS, HIDDEN),
        (torch.bfloat16,),
    )
    w_w = pack_bf16(w_w)
    ln = ind.k_norm
    g_k = ln.weight.detach().float().contiguous()
    b_k = ln.bias.detach().float().contiguous()
    copies["indexer weights_proj packed", idx] = w_w
    copies["indexer k LayerNorm f32", idx, "weight"] = g_k
    copies["indexer k LayerNorm f32", idx, "bias"] = b_k
    keep = (w_k, s_k, w_w, w_q, s_q, g_k, b_k)
    table = torch.tensor(
        [t.data_ptr() for t in keep] + [0], dtype=torch.int64, device=w_k.device
    )
    return table, keep, (sh_k, sh_q)


def _slices(rows: torch.Tensor, n: int) -> torch.Tensor:
    """``n`` equal row slices stacked: [n, rows / n, ...]."""
    return torch.stack(list(rows.view(torch.uint8).chunk(n))).view(rows.dtype)


def _dense_mlp(mlp, idx: int, inter: int, copies: dict):
    """A dense SwiGLU MLP's FP8 weights as ``n`` expert-shaped slices of ``inter``
    intermediates (gate rows, then up rows; down's matching K columns), copied:
    (w_ug, s_ug, w_dn, s_dn, n, shuffled). A shuffled weight is AITER's (16, 16)
    layout, [rows / 16, K / 32, 512 bytes], whose 16-row groups and 32-K tiles the
    slices keep whole."""
    gu, s_gu, sh_gu = _linear(
        "gate_up_proj", mlp.gate_up_proj, mlp.gate_up_proj.weight.shape[0], HIDDEN
    )
    I = gu.shape[0] // 2
    if I % inter or I // inter > MOE_SLOTS or gu.shape[0] != 2 * I:
        raise ValueError(
            f"needs the dense MLP's {I} intermediates a rank as at most {MOE_SLOTS} "
            f"slices of {inter}"
        )
    n = I // inter
    dn, s_dn, sh_dn = _linear("down_proj", mlp.down_proj, HIDDEN, I)
    if sh_gu != sh_dn:
        raise ValueError("needs gate_up_proj and down_proj in one FP8 layout")
    gate, up = _slices(gu[:I], n), _slices(gu[I:], n)
    w_ug = torch.cat([gate.view(torch.uint8), up.view(torch.uint8)], 1).view(gu.dtype)
    sg, su = _slices(s_gu[: I // 128], n), _slices(s_gu[I // 128 :], n)
    s_ug = torch.cat([sg, su], 1).contiguous()
    if sh_dn:
        tiles = dn.view(torch.uint8).view(HIDDEN // 16, I // 32, 512)
        w_dn = torch.stack(
            [t.reshape(HIDDEN, inter) for t in tiles.chunk(n, dim=1)]
        ).view(dn.dtype)
    else:
        w_dn = torch.stack(
            [t.contiguous() for t in dn.view(torch.uint8).chunk(n, dim=1)]
        ).view(dn.dtype)
    s_dn = torch.stack([t.contiguous() for t in s_dn.chunk(n, dim=1)])
    for name, t in (("w_ug", w_ug), ("s_ug", s_ug), ("w_dn", w_dn), ("s_dn", s_dn)):
        copies["dense MLP as expert slices", idx, name] = t
    return w_ug.contiguous(), s_ug, w_dn.contiguous(), s_dn, n, sh_dn


def is_dense(layer) -> bool:
    """A dense layer: its MLP is SGLang's DeepseekV2MLP, not the MoE."""
    return hasattr(layer.mlp, "gate_up_proj")


def layer_weights(layer, idx: int, tp: int, copies: dict):
    """A covered layer's tensors (``Glm53Weights``); a dense layer's MLP as expert
    slices, with no router."""
    from .runner import Glm53Weights

    attn, moe = layer.self_attn, layer.mlp
    heads = GLM5_GLOBAL_HEADS // tp
    bf16 = (torch.bfloat16,)
    g_in = _need("input_layernorm", layer.input_layernorm.weight, (HIDDEN,), bf16)
    g_post = _need(
        "post_attention_layernorm",
        layer.post_attention_layernorm.weight,
        (HIDDEN,),
        bf16,
    )
    g_q = _need("q_a_layernorm", attn.q_a_layernorm.weight, (Q_LORA,), bf16)
    g_kv = _need("kv_a_layernorm", attn.kv_a_layernorm.weight, (KV_LORA,), bf16)
    w_qkv, s_qkv, sh_qkv = _linear(
        "fused_qkv_a_proj_with_mqa", attn.fused_qkv_a_proj_with_mqa, QKV_A, HIDDEN
    )
    w_qb, s_qb, sh_qb = _linear(
        "q_b_proj", attn.q_b_proj, heads * (NOPE_DIM + PE_DIM), Q_LORA
    )
    w_o, s_o, sh_o = _linear("o_proj", attn.o_proj, HIDDEN, heads * V_DIM)
    w_uk, s_uk, w_uv, s_uv = _uk_uv(attn, heads, idx, copies)
    index, keep, shuffles = None, (), (sh_qkv, sh_qb, sh_o)
    if attn.indexer is not None:
        index, keep, ind_shuffles = _indexer(attn, idx, copies)
        shuffles += ind_shuffles
    if is_dense(layer):
        w_ug, s_ug, w_dn, s_dn, n, sh_mlp = _dense_mlp(
            moe, idx, GLM5_GLOBAL_INTER // tp, copies
        )
        shuffles += (sh_mlp,)
        if len(set(shuffles)) != 1:
            raise ValueError(
                f"needs every block-FP8 projection in one layout, got shuffled {shuffles}"
            )
        # no router: its tensors are never read
        return Glm53Weights(
            g_in=g_in,
            g_q=g_q,
            g_kv=g_kv,
            g_post=g_post,
            w_qkv_a=w_qkv,
            s_qkv_a=s_qkv,
            w_q_b=w_qb,
            s_q_b=s_qb,
            w_uk=w_uk,
            s_uk=s_uk,
            w_uv=w_uv,
            s_uv=s_uv,
            w_o=w_o,
            s_o=s_o,
            w_r=g_post,
            bias=g_post,
            w_ug=w_ug,
            s_ug=s_ug,
            w_dn=w_dn,
            s_dn=s_dn,
            shuffled=shuffles[0],
            index=index,
            keep=keep,
            dense=n,
        )
    why = experts_reason(moe)
    if why is not None:
        raise ValueError(why)
    experts = moe.experts
    inter = experts.w13_weight.shape[1] // 2
    w_ug = _need(
        "w13_weight",
        experts.w13_weight,
        (PHYSICAL_EXPERTS, 2 * inter, HIDDEN),
        (torch.float8_e4m3fn,),
    )
    s_ug = _need(
        "w13_weight_scale_inv",
        experts.w13_weight_scale_inv,
        (PHYSICAL_EXPERTS, 2 * inter // 128, HIDDEN // 128),
        (torch.float32,),
    )
    w_dn = _need(
        "w2_weight",
        experts.w2_weight,
        (PHYSICAL_EXPERTS, HIDDEN, inter),
        (torch.float8_e4m3fn,),
    )
    s_dn = _need(
        "w2_weight_scale_inv",
        experts.w2_weight_scale_inv,
        (PHYSICAL_EXPERTS, HIDDEN // 128, inter // 128),
        (torch.float32,),
    )
    gate = moe.gate
    router = _need(
        "gate", gate.weight, (N_EXPERTS, HIDDEN), (torch.bfloat16, torch.float32)
    )
    if router.dtype != torch.bfloat16 and not torch.equal(
        router.to(torch.bfloat16).float(), router
    ):
        # SGLang routes an f32 gate (router_fp32) in f32; the kernel's router is bf16
        raise ValueError("needs router weights exactly representable in bf16")
    w_r = pack_bf16(router.to(torch.bfloat16).contiguous())
    copies["router packed", idx] = w_r
    bias = _need(
        "e_score_correction_bias",
        gate.e_score_correction_bias,
        (N_EXPERTS,),
        (torch.float32,),
    )
    # the kernel reads every in-place FP8 weight of a launch in one layout
    if len(set(shuffles)) != 1:
        raise ValueError(
            f"needs every block-FP8 projection in one layout, got shuffled {shuffles}"
        )
    return Glm53Weights(
        g_in=g_in,
        g_q=g_q,
        g_kv=g_kv,
        g_post=g_post,
        w_qkv_a=w_qkv,
        s_qkv_a=s_qkv,
        w_q_b=w_qb,
        s_q_b=s_qb,
        w_uk=w_uk,
        s_uk=s_uk,
        w_uv=w_uv,
        s_uv=s_uv,
        w_o=w_o,
        s_o=s_o,
        w_r=w_r,
        bias=bias,
        w_ug=w_ug,
        s_ug=s_ug,
        w_dn=w_dn,
        s_dn=s_dn,
        shuffled=shuffles[0],
        index=index,
        keep=keep,
    )


def _cos_sin(rotary):
    """A rotary's (cos, sin) [positions, 32]: SGLang's f32 ``cos_sin_cache``
    [positions, 64] (cos, then sin), or AITER's ``cos_cache`` and ``sin_cache``."""
    cache = getattr(rotary, "cos_sin_cache", None)
    if cache is not None:
        return cache[:, : PE_DIM // 2], cache[:, PE_DIM // 2 :]
    cos, sin = rotary.cos_cache, rotary.sin_cache
    return cos.reshape(cos.shape[0], -1), sin.reshape(sin.shape[0], -1)


def rope_tables(layers: list, copies: dict):
    """The bf16 cos and sin tables [positions, 32] both rotaries of every covered
    layer read, from the first layer's MLA rotary; each indexer's must equal it."""
    cos, sin = _cos_sin(layers[0].self_attn.rotary_emb)
    for layer in layers:
        indexer = layer.self_attn.indexer
        if indexer is None:
            continue
        ic, is_ = _cos_sin(indexer.rotary_emb)
        if ic.shape != cos.shape or not (
            torch.equal(ic, cos) and torch.equal(is_, sin)
        ):
            raise ValueError("needs the indexer's rotary table to equal the MLA one")
    if cos.dim() != 2 or cos.shape[1] != PE_DIM // 2:
        raise ValueError(
            f"needs rotary tables [positions, {PE_DIM // 2}], got {tuple(cos.shape)}"
        )
    cos = cos.to(torch.bfloat16).contiguous()
    sin = sin.to(torch.bfloat16).contiguous()
    copies["rotary cos / sin bf16", "cos"] = cos
    copies["rotary cos / sin bf16", "sin"] = sin
    return cos, sin


def _vocab_shard(name: str, module, tp: int, rank: int) -> int:
    """This rank's rows of a plain vocab-parallel shard over the TP group (equal
    shards in rank order, no added or padded rows): their count."""
    method = type(module.quant_method).__name__
    if method != "UnquantizedEmbeddingMethod":
        raise ValueError(f"needs an unquantized {name}, got {method}")
    if not module.enable_tp or module.use_attn_tp_group or module.tp_size != tp:
        raise ValueError(f"needs the {name} sharded over the TP group")
    si = module.shard_indices
    rows = si.org_vocab_end_index - si.org_vocab_start_index
    if (
        si.num_org_vocab_padding
        or si.added_vocab_end_index != si.added_vocab_start_index
        or si.org_vocab_start_index != rank * rows
    ):
        raise ValueError(
            f"needs the {name} as equal vocabulary shards in rank order, no added or "
            "padded rows"
        )
    _need(f"{name} weight", module.weight, (rows, HIDDEN), (torch.bfloat16,))
    return rows


def embedding(model, tp: int, rank: int) -> torch.Tensor:
    """This rank's rows of the input embedding (bf16 [rows, HIDDEN], in place)."""
    emb = model.model.embed_tokens
    if emb.output_dtype is not None:
        raise ValueError("needs the input embedding in its weight's dtype")
    _vocab_shard("input embedding", emb, tp, rank)
    return emb.weight


def head_weights(model, tp: int, rank: int):
    """The tail's tensors (``Glm53HeadWeights``): the final norm and this rank's
    LM-head rows, in place."""
    from .runner import Glm53HeadWeights

    _vocab_shard("LM head", model.lm_head, tp, rank)
    return Glm53HeadWeights(
        g_norm=_need(
            "model.norm", model.model.norm.weight, (HIDDEN,), (torch.bfloat16,)
        ),
        w_head=model.lm_head.weight,
    )
