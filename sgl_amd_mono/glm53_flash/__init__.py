# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash monokernels (RFC #43949): the spec and its SGLang glue in
``spec``, and the gfx950 kernel and host runner in ``gfx950``.

Adapted from vLLM's GLM-5.3-Flash MonoKernel decode (vllm-project/vllm at
ca328e297b, Apache-2.0), whose kernel comes from ROCm/aiter#6173 at c39b56c36
(Apache-2.0, Copyright (c) 2025 FlyDSL Project Contributors); each file names its
origin.
"""

# GLM-5.3-Flash's text-model shapes, routing and activation, which the kernel builds in
HF_CONFIG = dict(
    model_type="glm5_next_text",
    hidden_size=4096,
    vocab_size=154880,
    intermediate_size=12288,
    n_routed_experts=288,
    num_experts_per_tok=8,
    n_shared_experts=1,
    moe_layer_freq=1,
    moe_intermediate_size=2048,
    routed_scaling_factor=2.5,
    scoring_func="sigmoid",
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    swiglu_limit=10.0,
    mhc=True,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rms_norm_eps=1e-5,
    num_attention_heads=64,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=256,
    qk_rope_head_dim=0,
    v_head_dim=256,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=2048,
    index_kpool=4,
    index_kpool_compress=True,
    index_kpool_always_select_tail=True,
)
# Its KDA linear attention (``linear_attn_config``)
LINEAR_ATTN_CONFIG = dict(
    num_heads=64,
    head_dim=128,
    short_conv_kernel_size=4,
    gate_lower_bound=-5.0,
)
