# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm at ca328e297b (Apache-2.0):
# vllm/models/glm5next/amd/mono/weights.py
"""One GLM-5.3-Flash layer's tensors as the launch reads them, and the embedding
and LM head the first and the tail launches read: views of SGLang's loaded
parameters, except the copies each function names (each records them in
``copies``, so attach logs their sizes)."""

from __future__ import annotations

from typing import Optional

import torch

from .config import (
    HC_MULT,
    HIDDEN,
    INDEX_DIM,
    INDEX_HEADS,
    KDA_DIM,
    KPOOL,
    KV_LORA,
    N_EXPERTS,
    NOPE_DIM,
    Q_LORA,
    V_DIM,
    VOCAB,
    FlashShard,
)

N_MIX = HC_MULT * (2 + HC_MULT)

# The SGLang names these functions read
SGLANG_READS = {
    "srt/models/glm5_next.py": (
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "do_fuse_qkvbfg",
        "fused_qkvbfg_a_proj",
        "fused_fg_b_proj",
        "qkv_conv1d",
        "A_log",
        "dt_bias",
        "o_norm",
        "num_fused_shared_experts",
        "self_attn",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
        "o_proj",
        "model",
        "embed_tokens",
        "norm",
        "lm_head",
    ),
    "srt/models/deepseek_v2.py": (
        "fused_qkv_a_proj_with_mqa",
        "q_a_layernorm",
        "kv_a_layernorm",
        "q_b_proj",
        "kv_b_proj",
        "shared_experts",
        "gate_up_proj",
        "down_proj",
        "gate",
        "experts",
        "e_score_correction_bias",
        "indexer",
        "swiglu_limit",
    ),
    "srt/layers/vocab_parallel_embedding.py": (
        "enable_tp",
        "use_attn_tp_group",
        "tp_size",
        "shard_indices",
        "org_vocab_start_index",
        "org_vocab_end_index",
        "added_vocab_start_index",
        "added_vocab_end_index",
        "num_org_vocab_padding",
        "output_dtype",
        "quant_method",
        "UnquantizedEmbeddingMethod",
    ),
    "srt/layers/attention/dsa/dsa_indexer.py": ("wk", "wq_b", "weights_proj"),
    "srt/layers/attention/dsa/dsa_indexer_kpool.py": (
        "index_kpool_compress_gate",
        "index_kpool_compress_ape",
        "k_norm",
    ),
    "srt/layers/quantization/fp8.py": (
        "aiter_bpreshuffled",
        "is_shuffled",
        "weight_scale_inv",
    ),
    "srt/layers/quantization/fp8_hip.py": ("mxfp8_native_ready",),
    "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py": (
        "is_shuffled",
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ),
}


def _need(
    name: str, t: torch.Tensor, shape: tuple, dtypes: tuple[torch.dtype, ...]
) -> torch.Tensor:
    if tuple(t.shape) != shape or t.dtype not in dtypes or not t.is_contiguous():
        raise ValueError(
            f"needs {name} as contiguous {'/'.join(map(str, dtypes))} {shape}, "
            f"got {t.dtype} {tuple(t.shape)}"
        )
    return t


def _bytes(name: str, t: torch.Tensor, shape: tuple) -> torch.Tensor:
    """An MXFP4 / E8M0 tensor viewed as raw bytes, checked."""
    if t.element_size() != 1 or tuple(t.shape) != shape or not t.is_contiguous():
        raise ValueError(
            f"needs {name} as contiguous 1-byte {shape}, got {t.dtype} {tuple(t.shape)}"
        )
    return t.view(torch.uint8)


def experts_reason(layer) -> Optional[str]:
    """Why the layer's routed experts are not in the layout the launch reads, or None.
    It reads AITER's A4W4 layout: (16, 16)-shuffled MXFP4 gate rows, then up rows,
    and 3-D e8m0_shuffle scales (the GU-interleaved A16W4 layout keeps them 2-D)."""
    experts = layer.mlp.experts
    w13, s13 = experts.w13_weight, experts.w13_weight_scale
    if not getattr(w13, "is_shuffled", False):
        return "needs the MXFP4 experts in AITER's shuffled layout"
    if s13.dim() != 3:
        return (
            "needs the A4W4 expert layout (gate rows, then up rows), not GU-interleaved"
        )
    return None


def _mxfp4_rows(name: str, linear, rows: int, k: int):
    """A row-major Quark MXFP4 linear as SGLang keeps it after loading: weight
    [rows, K / 2] and one E8M0 per 32 inputs, [rows, K / 32] (or [K / 32, rows])."""
    w, sc = linear.weight, getattr(linear, "weight_scale", None)
    if sc is not None and tuple(sc.shape) == (k // 32, rows) and rows != k // 32:
        sc = sc.T
    if (
        sc is None
        or w.element_size() != 1
        or sc.element_size() != 1
        or tuple(w.shape) != (rows, k // 2)
        or tuple(sc.shape) != (rows, k // 32)
    ):
        got = None if sc is None else tuple(sc.shape)
        raise ValueError(
            f"needs shared_experts.{name} as row-major MXFP4 {(rows, k // 2)} with a "
            f"{(rows, k // 32)} scale, got {w.dtype} {tuple(w.shape)} and scale {got}"
        )
    return w.view(torch.uint8).contiguous(), sc.view(torch.uint8).contiguous()


def _shared_expert(layer, i: int, inter: int, copies: dict):
    """The shared expert of an unfused layer (SGLang fuses it only at TP4 and TP8),
    packed once as one expert of AITER's MoE layout."""
    from .packing import pack_shared_expert

    shared = layer.mlp.shared_experts
    gate_up = _mxfp4_rows("gate_up_proj", shared.gate_up_proj, 2 * inter, HIDDEN)
    down = _mxfp4_rows("down_proj", shared.down_proj, HIDDEN, inter)
    packed = pack_shared_expert(*gate_up, *down)
    for name, t in zip(("w_ug", "s_ug", "w_dn", "s_dn"), packed):
        copies["shared expert packed", i, name] = t
    return packed


def layer_weights(layer, i: int, inter: int, copies: dict):
    """The MoE side: the routed experts as stored, the router packed once and its
    bias in f32, and the shared expert as the fused expert slot or a packed copy."""
    from .packing import pack_bf16
    from .runner import FlashFfnWeights

    why = experts_reason(layer)
    if why is not None:
        raise ValueError(why)
    mlp, experts = layer.mlp, layer.mlp.experts
    fused = mlp.num_fused_shared_experts > 0
    E = N_EXPERTS + 1 if fused else N_EXPERTS
    w_ug = _bytes("w13_weight", experts.w13_weight, (E, 2 * inter, HIDDEN // 2))
    s_ug = _bytes(
        "w13_weight_scale", experts.w13_weight_scale, (E, 2 * inter, HIDDEN // 32)
    )
    w_dn = _bytes("w2_weight", experts.w2_weight, (E, HIDDEN, inter // 2))
    s_dn = _bytes("w2_weight_scale", experts.w2_weight_scale, (E, HIDDEN, inter // 32))
    if fused:
        shared = (w_ug[N_EXPERTS], s_ug[N_EXPERTS], w_dn[N_EXPERTS], s_dn[N_EXPERTS])
    else:
        shared = _shared_expert(layer, i, inter, copies)
    gate = _need(
        "gate.weight",
        mlp.gate.weight,
        (N_EXPERTS, HIDDEN),
        (torch.float32, torch.bfloat16),
    )
    router = gate.to(torch.bfloat16)
    if gate.dtype != torch.bfloat16 and not torch.equal(router.float(), gate):
        raise ValueError("needs router weights exactly representable in bf16")
    w_r = pack_bf16(router)
    copies["router packed", i] = w_r
    bias = _need(
        "gate.e_score_correction_bias",
        mlp.gate.e_score_correction_bias,
        (N_EXPERTS,),
        (torch.float32, torch.bfloat16),
    )
    if bias.dtype != torch.float32:
        bias = bias.float()
        copies["f32 router bias", i] = bias
    return FlashFfnWeights(
        w_r=w_r,
        bias=bias,
        w_ug=w_ug,
        s_ug=s_ug,
        w_dn=w_dn,
        s_dn=s_dn,
        w_sh_ug=shared[0],
        s_sh_ug=shared[1],
        w_sh_dn=shared[2],
        s_sh_dn=shared[3],
        hc_fn=_need(
            "hc_ffn_fn", layer.hc_ffn_fn, (N_MIX, HC_MULT * HIDDEN), (torch.float32,)
        ),
        hc_scale=_need("hc_ffn_scale", layer.hc_ffn_scale, (3,), (torch.float32,)),
        hc_base=_need("hc_ffn_base", layer.hc_ffn_base, (N_MIX,), (torch.float32,)),
        g_norm=_need(
            "post_attention_layernorm.weight",
            layer.post_attention_layernorm.weight,
            (HIDDEN,),
            (torch.bfloat16,),
        ),
    )


def _proj(name: str, linear, rows: int, cols: int):
    """A projection as SGLang keeps it after loading: (weight, scales, shuffled).
    FP8 E4M3 with f32 128 x 128 block scales, row-major or AITER-shuffled in place,
    or row-major bf16 (scales None)."""
    w = linear.weight
    if w.dtype == torch.bfloat16:
        return _need(f"{name}.weight", w, (rows, cols), (torch.bfloat16,)), None, False
    if getattr(linear, "mxfp8_native_ready", False):
        raise ValueError(f"needs {name} as block FP8; it was requantized to MXFP8")
    sc = getattr(linear, "weight_scale_inv", None)
    if (
        w.dtype != torch.float8_e4m3fn
        or tuple(w.shape) != (rows, cols)
        or not w.is_contiguous()
        or sc is None
        or sc.dtype != torch.float32
        or tuple(sc.shape) != (rows // 128, cols // 128)
        or not sc.is_contiguous()
    ):
        got = None if sc is None else f"{sc.dtype} {tuple(sc.shape)}"
        raise ValueError(
            f"needs {name} as row-major bf16 {(rows, cols)}, or FP8 E4M3 with f32 "
            f"{(rows // 128, cols // 128)} block scales; got {w.dtype} "
            f"{tuple(w.shape)} {tuple(w.stride())} and scale {got}"
        )
    shuffled = getattr(linear, "aiter_bpreshuffled", False) or getattr(
        w, "is_shuffled", False
    )
    return w, sc, bool(shuffled)


def dsa_weights(layer, i: int, tp: int, copies: dict):
    """A sparse-attention layer's attention side: views of SGLang's parameters, plus
    W_UK^T packed once from kv_b_proj into [H, 512, 256] bf16 and the indexer's
    wk | weights_proj in one bf16 matrix."""
    from .runner import FlashDsaWeights

    attn = layer.self_attn
    indexer = attn.indexer
    H = FlashShard(tp=tp).mla_heads
    bf16, f32 = (torch.bfloat16,), (torch.float32,)
    w_qkv, s_qkv, qkv_sh = _proj(
        "fused_qkv_a_proj_with_mqa",
        attn.fused_qkv_a_proj_with_mqa,
        Q_LORA + KV_LORA,
        HIDDEN,
    )
    w_qb, s_qb, qb_sh = _proj("q_b_proj", attn.q_b_proj, H * NOPE_DIM, Q_LORA)
    w_o, s_o, o_sh = _proj("o_proj", attn.o_proj, HIDDEN, H * V_DIM)
    per = NOPE_DIM + V_DIM
    kv_b = _need("kv_b_proj.weight", attn.kv_b_proj.weight, (H * per, KV_LORA), bf16)
    w_uk = kv_b.view(H, per, KV_LORA)[:, :NOPE_DIM].transpose(1, 2).contiguous()
    copies["W_UK^T bf16", i] = w_uk
    wk = _need("indexer.wk.weight", indexer.wk.weight, (INDEX_DIM, HIDDEN), bf16)
    wp = _need(
        "indexer.weights_proj.weight",
        indexer.weights_proj.weight,
        (INDEX_HEADS, HIDDEN),
        (torch.bfloat16, torch.float32),
    )
    wp16 = wp.to(torch.bfloat16)
    if wp.dtype != torch.bfloat16 and not torch.equal(wp16.float(), wp):
        raise ValueError("needs indexer weights_proj exactly representable in bf16")
    w_idx = torch.cat([wk, wp16]).contiguous()
    copies["indexer wk | weights_proj bf16", i] = w_idx
    return FlashDsaWeights(
        hc_fn=_need("hc_attn_fn", layer.hc_attn_fn, (N_MIX, HC_MULT * HIDDEN), f32),
        hc_scale=_need("hc_attn_scale", layer.hc_attn_scale, (3,), f32),
        hc_base=_need("hc_attn_base", layer.hc_attn_base, (N_MIX,), f32),
        g_norm=_need(
            "input_layernorm.weight", layer.input_layernorm.weight, (HIDDEN,), bf16
        ),
        w_qkv=w_qkv,
        s_qkv=s_qkv,
        w_idx=w_idx,
        w_gate=_need(
            "indexer.index_kpool_compress_gate",
            indexer.index_kpool_compress_gate,
            (INDEX_DIM, HIDDEN),
            bf16,
        ),
        g_qa=_need("q_a_layernorm.weight", attn.q_a_layernorm.weight, (Q_LORA,), bf16),
        g_kv=_need(
            "kv_a_layernorm.weight", attn.kv_a_layernorm.weight, (KV_LORA,), bf16
        ),
        ln_w=_need("indexer.k_norm.weight", indexer.k_norm.weight, (INDEX_DIM,), f32),
        ln_b=_need("indexer.k_norm.bias", indexer.k_norm.bias, (INDEX_DIM,), f32),
        ape=_need(
            "indexer.index_kpool_compress_ape",
            indexer.index_kpool_compress_ape,
            (KPOOL, INDEX_DIM),
            f32,
        ),
        w_qb=w_qb,
        s_qb=s_qb,
        w_iq=_need(
            "indexer.wq_b.weight",
            indexer.wq_b.weight,
            (INDEX_HEADS * INDEX_DIM, Q_LORA),
            bf16,
        ),
        w_uk=w_uk,
        w_kvb=kv_b,
        w_o=w_o,
        s_o=s_o,
        shuffled=(qkv_sh, qb_sh, o_sh),
    )


def kda_weights(layer, i: int, tp: int, copies: dict):
    """A KDA layer's attention side: views of SGLang's fused bf16 projections; the
    o_norm weight as f32 (a copy when SGLang keeps it in bf16)."""
    from .runner import FlashKdaWeights

    attn = layer.self_attn
    shard = FlashShard(tp=tp)
    P, H, D = shard.kda_proj, shard.kda_heads, KDA_DIM
    bf16, f32 = (torch.bfloat16,), (torch.float32,)
    if not attn.do_fuse_qkvbfg:
        raise ValueError("needs the fused q | k | v | beta | f_a | g_a projection")
    fg_b = _need("fused_fg_b_proj.weight", attn.fused_fg_b_proj.weight, (2, P, D), bf16)
    conv = _need("qkv_conv1d.weight", attn.qkv_conv1d.weight, (3 * P, 1, 4), f32)
    if attn.qkv_conv1d.bias is not None:
        raise ValueError("needs the short conv without bias")
    w_on = attn.o_norm.weight
    if w_on is None or tuple(w_on.shape) != (D,):
        got = None if w_on is None else tuple(w_on.shape)
        raise ValueError(f"needs o_norm.weight [{D}], got {got}")
    if w_on.dtype != torch.float32:
        w_on = w_on.float().contiguous()
        copies["o_norm f32", i] = w_on
    return FlashKdaWeights(
        hc_fn=_need("hc_attn_fn", layer.hc_attn_fn, (N_MIX, HC_MULT * HIDDEN), f32),
        hc_scale=_need("hc_attn_scale", layer.hc_attn_scale, (3,), f32),
        hc_base=_need("hc_attn_base", layer.hc_attn_base, (N_MIX,), f32),
        g_norm=_need(
            "input_layernorm.weight", layer.input_layernorm.weight, (HIDDEN,), bf16
        ),
        w_in=_need(
            "fused_qkvbfg_a_proj.weight",
            attn.fused_qkvbfg_a_proj.weight,
            (shard.kda_in_rows, HIDDEN),
            bf16,
        ),
        w_fb=fg_b[0],
        w_gb=fg_b[1],
        conv_w=conv.view(3 * P, 4),
        a_log=_need("A_log", attn.A_log, (1, 1, H, 1), f32).view(H),
        dt_bias=_need("dt_bias", attn.dt_bias, (P,), f32),
        w_on=w_on,
        w_o=_need("o_proj.weight", attn.o_proj.weight, (HIDDEN, P), bf16),
    )


def dense_weights(layer, i: int, tp: int, copies: dict):
    """A dense layer's FFN side: its MLP's gate | up and down as SGLang keeps them,
    block FP8 (row-major or AITER-shuffled, the same for both), and the FFN mHC."""
    from .runner import FlashDenseWeights

    mlp = layer.mlp
    inter = FlashShard(tp=tp).dense_inter
    if mlp.swiglu_limit != 10.0:
        raise ValueError(
            f"needs the dense MLP's SwiGLU limit 10, got {mlp.swiglu_limit}"
        )
    w_ug, s_ug, ug_sh = _proj("mlp.gate_up_proj", mlp.gate_up_proj, 2 * inter, HIDDEN)
    w_dn, s_dn, dn_sh = _proj("mlp.down_proj", mlp.down_proj, HIDDEN, inter)
    if s_ug is None or s_dn is None:
        raise ValueError("needs the dense MLP as block FP8, got bf16")
    if ug_sh != dn_sh:
        raise ValueError("needs the dense MLP's gate | up and down in one FP8 layout")
    return FlashDenseWeights(
        w_ug=w_ug,
        s_ug=s_ug,
        w_dn=w_dn,
        s_dn=s_dn,
        hc_fn=_need(
            "hc_ffn_fn", layer.hc_ffn_fn, (N_MIX, HC_MULT * HIDDEN), (torch.float32,)
        ),
        hc_scale=_need("hc_ffn_scale", layer.hc_ffn_scale, (3,), (torch.float32,)),
        hc_base=_need("hc_ffn_base", layer.hc_ffn_base, (N_MIX,), (torch.float32,)),
        g_norm=_need(
            "post_attention_layernorm.weight",
            layer.post_attention_layernorm.weight,
            (HIDDEN,),
            (torch.bfloat16,),
        ),
        shuffled=ug_sh,
    )


def _unquantized(name: str, module) -> None:
    method = type(module.quant_method).__name__
    if method != "UnquantizedEmbeddingMethod":
        raise ValueError(f"needs an unquantized {name}, got {method}")


def _vocab_shard(name: str, module, tp: int) -> None:
    """A plain vocab-parallel shard over the TP group: VOCAB / TP rows a rank, no
    added or padded rows."""
    if not module.enable_tp or module.use_attn_tp_group or module.tp_size != tp:
        raise ValueError(f"needs the {name} sharded over the TP group")
    si = module.shard_indices
    rows = VOCAB // tp
    if (
        si.num_org_vocab_padding
        or si.added_vocab_end_index != si.added_vocab_start_index
        or si.org_vocab_end_index - si.org_vocab_start_index != rows
    ):
        raise ValueError(
            f"needs the {name} as {rows} vocabulary rows a rank, no added or padded rows"
        )


def head_weights(model, tp: int):
    """The tail: the final RMSNorm weight and this rank's rows of the vocab-parallel
    LM head, both views."""
    from .runner import FlashHeadWeights

    head = model.lm_head
    _unquantized("LM head", head)
    _vocab_shard("LM head", head, tp)
    return FlashHeadWeights(
        g_norm=_need(
            "model.norm.weight", model.model.norm.weight, (HIDDEN,), (torch.bfloat16,)
        ),
        w_head=_need(
            "lm_head.weight", head.weight, (VOCAB // tp, HIDDEN), (torch.bfloat16,)
        ),
    )


def embed_weights(model, tp: int):
    """The input embedding as the step's first launch reads it: this rank's rows of
    the vocab-parallel table (a view) and the vocabulary ids they hold."""
    from .runner import FlashEmbedding

    emb = model.model.embed_tokens
    _unquantized("input embedding", emb)
    if emb.output_dtype is not None:
        raise ValueError("needs the input embedding in its weight's dtype")
    _vocab_shard("input embedding", emb, tp)
    si = emb.shard_indices
    return FlashEmbedding(
        weight=_need(
            "embed_tokens.weight", emb.weight, (VOCAB // tp, HIDDEN), (torch.bfloat16,)
        ),
        start=int(si.org_vocab_start_index),
        end=int(si.org_vocab_end_index),
    )
