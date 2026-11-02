# SPDX-License-Identifier: Apache-2.0
"""Muse Spark monokernels (RFC #43949): the spec and its SGLang glue in ``spec``,
the per-layer checks in ``weights``, and the gfx950 kernels and host runner in
``gfx950``.

Stage 0: the plugin takes every routed step (the TEP8 verify steps of 8, 16, 32
and 64 rows) and runs SGLang's own ROCm decoder layers inside its forward, so the
routing, the rank agreement, the graph coverage and the DFlash aux capture are
exercised with outputs identical to the normal path before a kernel exists.
"""

# Muse Spark's text config the kernels build in (config.json, text_config)
HF_CONFIG = dict(
    hidden_size=8192,
    num_hidden_layers=62,
    num_attention_heads=128,
    num_key_value_heads=16,
    head_dim=64,
    num_local_experts=256,
    num_experts_per_tok=8,
    num_shared_experts=0,
    moe_hidden_size=4096,
    moe_intermediate_size=4096,
    routed_scaling_factor=1.0,
    output_multiplier=0.17677669529663687,
    rms_norm_eps=1e-5,
    post_norm_eps=1e-8,
    residual_gate_temperature=0.3,
    qk_scale_factor=5.4730064863875,
    sliding_window=2048,
)
TP = 8  # attention TP and expert EP: the ROCm fast paths are TP8 / EP8-shaped
DRAFT_TOKENS = 8  # DFlash2 verifies 8 rows a request; the widths are its multiples
