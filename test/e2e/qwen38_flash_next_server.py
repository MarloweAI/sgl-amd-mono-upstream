"""The Qwen3.8-Flash-Next server the serving tests launch: one MI355X, TP1, the
MXFP4 / FP8-attention checkpoint, NEXTN with 3 draft steps (verify steps of 4 rows
a request, 8 requests at the largest width: 32 rows).

The cookbook's AMD recipe serves the BF16 / FP8 checkpoints at TP8; the launches
run the whole model on one GPU, so these servers diverge from it in the TP size
and the checkpoint (amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8)."""

import os

MODEL = os.environ.get(
    "QWEN38_FLASH_NEXT_MODEL_PATH", "amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8"
)
ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "AITER_FLYDSL_FORCE": "1",
    "SGLANG_MAMBA_SSM_DTYPE": "bfloat16",
    "SGLANG_USE_AITER_FP8_PER_TOKEN": "1",
}
# Without --monokernel-backend, so a test can launch the normal path too.
ARGS = [
    "--tp",
    "1",
    "--attention-backend",
    "triton",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--cuda-graph-max-bs-decode",
    "8",
    "--max-running-requests",
    "16",
]
