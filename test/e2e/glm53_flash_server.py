"""The GLM-5.3-Flash servers that the monokernel serving test launches: MI35x, TP4
(``GLM_53_FLASH_TP``: 2, 4 or 8), decode steps (no speculative decoding).
``GLM_53_FLASH_MODEL_PATH`` may name a checkpoint whose dense MLP, shared expert and
attention projections are bf16."""

import os

MODEL = os.environ.get("GLM_53_FLASH_MODEL_PATH", "amd/GLM-5.3-Flash-Quark-MXFP4")
TP = int(os.environ.get("GLM_53_FLASH_TP", "4"))
ENV = {"SGLANG_USE_AITER": "1", "AITER_ONLINE_TUNE": "0"}
# The cookbook's MI355X high-throughput MXFP4 cell, except: a checkpoint whose shared
# expert is MXFP4 (SGLang fuses it into the routed experts at TP4 and TP8), EP 1 (the
# kernels shard every expert by TP) and a bf16 KV cache. Without --monokernel-backend,
# so a test can launch the normal path too.
ARGS = [
    "--trust-remote-code",
    "--quantization",
    "quark",
    "--tp-size",
    str(TP),
    "--ep-size",
    "1",
    "--mem-fraction-static",
    "0.82",
    "--max-running-requests",
    "64",
    "--cuda-graph-max-bs-decode",
    "64",
    "--chunked-prefill-size",
    "8192",
    "--max-prefill-tokens",
    "8192",
    "--dsa-prefill-backend",
    "tilelang",
    "--dsa-decode-backend",
    "tilelang",
    "--kv-cache-dtype",
    "bfloat16",
    "--moe-runner-backend",
    "aiter",
    "--reasoning-parser",
    "glm45",
    "--mm-feature-transport",
    "cpu",
]
