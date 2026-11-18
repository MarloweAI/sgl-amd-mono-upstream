"""The Qwen3.8 MoE servers that the monokernel serving test launches: MI355X, TP8,
decode steps, and NEXTN verify steps."""

import os

MODEL = os.environ.get(
    "QWEN38_A95B_MXFP4_MODEL_PATH", "amd/Qwen3.8-2.4T-A95B-Quark-MXFP4"
)
ENV = {
    "SGLANG_USE_AITER": "1",
    # bf16 GDN projections: SGLang quantizes them to MXFP4 by default
    "SGLANG_ROCM_GDN_ONLINE_MXFP4": "0",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION": "1",
    "AITER_USE_FLYDSL_MOE_SORTING": "1",
    "ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
    "SAFETENSORS_FAST_GPU": "1",
}
# The benchmark server of the PR, with up to 16 running requests and graphs to 16.
# Without --monokernel-backend, so a test can launch the normal path too.
ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "8",
    "--attention-backend",
    "aiter",
    "--page-size",
    "16",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--chunked-prefill-size",
    "16384",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.9",
    "--max-running-requests",
    "16",
    "--cuda-graph-config",
    '{"decode":{"backend":"full","max_bs":16,"bs":[1,2,4,8,16]}}',
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--reasoning-parser",
    "qwen3",
]
# NEXTN as benchmarked: verify steps of up to 2 requests (8 rows) run K2
NEXTN = [
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-draft-model-path",
    MODEL,
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--enable-linear-replayssm-spec",
]
