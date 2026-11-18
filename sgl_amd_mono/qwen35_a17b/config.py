# SPDX-License-Identifier: Apache-2.0
"""Checkpoint helpers shared by the Qwen3.5 precision paths."""

from __future__ import annotations

from sgl_amd_mono.qwen35_a17b.layout import (
    BLOCK,
    CONV_KERNEL,
    EXPERTS,
    FULL_ATTENTION_INTERVAL,
    GDN_HEAD,
    GDN_K_HEADS,
    GDN_V_HEADS,
    HEAD_DIM,
    HIDDEN,
    KV_HEADS,
    LAYERS,
    MOE_INTERMEDIATE,
    QUERY_HEADS,
    RMS_NORM_EPS,
    SHARED_INTERMEDIATE,
    TOPK,
    VOCAB_SIZE,
)

TEXT = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": HIDDEN,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": QUERY_HEADS,
    "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "linear_num_key_heads": GDN_K_HEADS,
    "linear_num_value_heads": GDN_V_HEADS,
    "linear_key_head_dim": GDN_HEAD,
    "linear_value_head_dim": GDN_HEAD,
    "linear_conv_kernel_dim": CONV_KERNEL,
    "num_experts": EXPERTS,
    "num_experts_per_tok": TOPK,
    "moe_intermediate_size": MOE_INTERMEDIATE,
    "shared_expert_intermediate_size": SHARED_INTERMEDIATE,
    "rms_norm_eps": RMS_NORM_EPS,
    "vocab_size": VOCAB_SIZE,
    "hidden_act": "silu",
    "attn_output_gate": True,
    "full_attention_interval": FULL_ATTENTION_INTERVAL,
}


def text_of(config):
    return getattr(config, "text_config", config)


def quant_of(config) -> dict:
    raw = getattr(config, "quantization_config", None) or getattr(
        text_of(config), "quantization_config", None
    )
    return dict(raw or {})


def is_block_fp8(config) -> bool:
    quant = quant_of(config)
    return (
        quant.get("quant_method") == "fp8"
        and quant.get("activation_scheme") == "dynamic"
        and list(quant.get("weight_block_size") or []) == [BLOCK, BLOCK]
    )
