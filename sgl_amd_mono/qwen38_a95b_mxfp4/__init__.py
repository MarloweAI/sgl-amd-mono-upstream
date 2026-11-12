# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 MoE monokernels (RFC #43949): the spec and its forward in ``spec.py``,
the gfx950 kernels in ``gfx950``. Ported from sgl-project/sglang#43516, whose
kernels come from vLLM's Qwen3.8 mono decode (vllm-project/vllm#60833 at
3346929b27, Apache-2.0).
"""

# The Qwen3.8-2.4T text config the kernels are built for (gfx950/layout.py)
HF_CONFIG = {
    "hidden_size": 8192,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 128,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 2048,
    "shared_expert_intermediate_size": 2048,
    "hidden_act": "silu",
}
