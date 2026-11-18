"""Qwen3.5-397B-A17B block-FP8 TP4 server with EAGLE MTP for the verify monokernel.

The flags follow InferenceX's ``qwen3.5/sglang/mi355x-fp8-mtp/8k1k.yaml``: aiter
attention, the aiter all-reduce fusion, page size 16 and a 3-step, topk-1,
4-token draft chain. A verify step is 4 rows a request, so decode graphs at
batch sizes 1 to 8 are the widths 4 to 32 the kernels serve. The arms differ
only by the monokernel flag that bench_server.py adds.
"""

import os

MODEL = os.environ.get("QWEN35_FP8_MODEL_PATH", "/data/Qwen3.5-397B-A17B-FP8")
ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "AITER_ONLINE_TUNE": "0",
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "SAFETENSORS_FAST_GPU": "1",
}
ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--attention-backend",
    "aiter",
    "--enable-aiter-allreduce-fusion",
    "--mem-fraction-static",
    "0.8",
    "--max-running-requests",
    "8",
    "--disable-radix-cache",
    "--chunked-prefill-size",
    "32768",
    "--page-size",
    "16",
    "--context-length",
    "9236",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--cuda-graph-config",
    '{"decode":{"backend":"full","max_bs":8,"bs":[1,2,3,4,5,6,7,8]}}',
    "--watchdog-timeout",
    "1200",
]
