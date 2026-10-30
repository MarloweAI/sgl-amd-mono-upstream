# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next monokernels (RFC #43949; SGLang's Qwen4-Exp): the spec and
its SGLang glue in ``spec`` (target verify) and ``draft`` (the MTP draft), and the
gfx950 kernel and host runner in ``gfx950``.
"""

# Qwen3.8-Flash-Next's text config, which the kernel builds in
# (amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8 config.json, text_config)
HF_CONFIG = {
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "full_attention_interval": 4,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "output_gate_type": "sigmoid",
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "hc_count": 4,
    "hc_lowrank": 320,
    "ple_layer_ids": [2],
    "ple_embed_dim": 2560,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}
# Its QSA indexer: 4 index heads of 128, 4-token compressed blocks, top-2048
# tokens (512 blocks)
INDEXER_CONFIG = {
    "indexer_n_heads": 4,
    "indexer_head_dim": 128,
    "indexer_compress_ratio": 4,
    "indexer_budget": 2048,
}
# 3 Gated DeltaNet layers, then one QSA layer, repeated over the 48 layers
LAYER_PATTERN = ("linear_attention",) * 3 + ("full_attention",)
# the tokens a request's context holds at most, verify tokens included: 9 index
# segments of 8192 compressed blocks of 4 (``gfx950/layout.MAX_CTX``)
MAX_CTX = 294912
