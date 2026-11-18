"""The models the AMD PR test and nightly run, one entry a served model.

A PR runs the entries whose ``packages`` it touches; a change to ``common/``,
``sgl_amd_mono/__init__.py`` or ``pyproject.toml`` runs all of them. The nightly
runs the entries marked ``nightly`` on main. Each entry
names the server its accuracy and speed tests launch (a recipe module under
``test/e2e`` and its argument list, or ``env`` / ``args`` given here), the
checkpoints on the CI nodes, its kernel and serving tests, its kernel-only
timing scripts and the client concurrencies of the speed test.

An entry without a checkpoint on the CI nodes, or on a GPU the nodes do not
have, still runs its unit and kernel tests; the report names what it skipped.
"""

import os

# The checkpoint root on the cluster's shared storage (the runner's environment
# sets it). A job mirrors its model's checkpoints to the node's local disk first
# (slurm_job.sh) and serves them from there, at the same paths.
HF = os.environ.get("CI_MODEL_ROOT", "/models")

# Every checkpoint the library supports, by Hugging Face ID under HF, kept on the
# CI nodes' local disk (`python3 models.py checkpoints`; a staging job with
# CI_MODEL=all mirrors them), including the models CI cannot serve yet.
# moonshotai/Kimi-K3.1 is on neither Hugging Face nor the cluster yet.
CHECKPOINTS = [
    f"{HF}/{repo}"
    for repo in (
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "deepseek-ai/DeepSeek-V4-Pro",
        "deepseek-ai/DeepSeek-V4-Pro-DSpark",
        "zai-org/GLM-5.3",
        "amd/GLM-5.2-MXFP4",
        "amd/GLM-5.3-Quark-MXFP4-AttnFP8",
        "amd/GLM-5.3-Flash-Quark-MXFP4",
        "moonshotai/Kimi-K3",
        "RadixArk/Kimi-K3-DSpark",
        "moonshotai/Kimi-K3.1",
        "RadixArk/Kimi-K3.1-DFlash2",
        "amd/MiniMax-M3-MXFP4",
        "Inferact/MiniMax-M3-EAGLE3-GQA",
        "MiniMaxAI/MiniMax-M3.1-Flash",
        "amd/MiniMax-M3.1-Flash-Quark-MXFP4",
        "amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8",
        "amd/Qwen3.8-2.4T-A95B-Quark-MXFP4",
        "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4",
        "RadixArk/Muse-Spark-Open-DFlash2",
    )
]

# The CI nodes' GPU.
GPU = "mi355x"

# Tests every PR runs, on any change.
COMMON_KERNEL_TESTS = [
    "test/kernels/test_guard_hip.py",
    "test/kernels/test_guard_multi_rank.py",
    "test/kernels/test_compiled_dispatch.py",
]

# Paths whose change runs every model.
SHARED_PATHS = ("sgl_amd_mono/common/", "sgl_amd_mono/__init__.py", "pyproject.toml")

# The fixed-length speed test (InferenceX's): random prompts of 8k +/- 20% tokens
# in, 1k +/- 20% out, 10 prompts a concurrency slot.
ISL, OSL, RANGE_RATIO, PROMPTS_PER_SLOT = 8192, 1024, 0.8, 10

# Accuracy, sgl-eval benchmarks, each scored with the monokernels on and off on
# the same build and server recipe. GSM8K (all 1319 questions) gates: the two runs
# are compared question by question (exact McNemar test), and a significant loss
# with the monokernels on fails. The others have too few questions to gate on;
# a score more than ``warn`` below the off run's is a warning. A PR runs GSM8K
# (and GPQA with the `run-ci-extra` label); the nightly runs the model's benchmarks but the ``weekly`` ones, and the
# weekly run (Saturdays) runs all of them. AIME26 x4 is the costliest (~23K tokens
# an answer), and GPQA covers much the same ground every night.
EVALS = {
    # max tokens: room for a thinking model's whole answer. Repeats are explicit:
    # sgl-eval's own defaults are 8 for GPQA and 16 for AIME.
    "gsm8k": {"args": ["--max-tokens", "16384", "--n-repeats", "1"], "gate": True},
    "gpqa": {"args": ["--max-tokens", "65536", "--n-repeats", "1"], "warn": 0.05},
    "aime26": {
        "args": ["--max-tokens", "65536", "--n-repeats", "4"],
        "warn": 0.03,
        "weekly": True,
    },
    # vision models only
    "mmmu_pro": {"args": ["--max-tokens", "8192", "--n-repeats", "1"], "warn": 0.03},
}
PR_EVALS = ["gsm8k"]
# Added to a PR run by the `run-ci-extra` label: GSM8K's short answers cannot show
# a loss at long context, GPQA's long ones can. It gates like GSM8K (paired against
# the monokernel-off reference) in a PR run, and only warns in the nightly.
EXTRA_EVALS = ["gpqa"]
# The nightly runs no off side. It holds each on score to the model's fixed
# ``baseline`` ({benchmark: minimum score}): below it fails for GSM8K and warns
# for the rest. A benchmark without one is compared with the monokernel-off
# reference (the latest weekly run's off result) instead. Each weekly report
# suggests baselines (the off score less the benchmark's margin,
# ``BASELINE_MARGIN``) to copy into the model's entry.
BASELINE_MARGIN = {"gsm8k": 0.015, "gpqa": 0.05, "aime26": 0.03, "mmmu_pro": 0.03}
TEXT_EVALS = ["gsm8k", "gpqa", "aime26"]
# One-sided significance level of the GSM8K test.
ALPHA = 0.05

# The speed gate, base side (the PR's merge-base) to PR. Every PR fails on a
# median-TPOT loss of more than PERF_LOSS at a gate (covered) concurrency; at the
# other measured concurrencies such a loss warns. A [Perf] or [Feat] PR also needs
# a gain of at least PERF_GAIN at a gate concurrency of one of its models.
PERF_GAIN, PERF_LOSS = 0.05, 0.02
GAIN_TYPES = ("Perf", "Feat")

MODELS = {
    "deepseek_v41": {
        "name": "DeepSeek-V4.1-Flash",
        # Phase 1 of the nightly
        "nightly": True,
        # set from the first weekly report
        "baseline": {},
        "packages": ["deepseek_v41"],
        "tp": 4,
        "paths": {
            "DEEPSEEK_V41_FLASH_MODEL_PATH": f"{HF}/deepseek-ai/DeepSeek-V4.1-Flash"
        },
        "recipe": ("deepseek_v41_server", "ARGS"),
        "kernel_tests": [
            "test/kernels/test_deepseek_v41_kernels.py",
            "test/kernels/test_deepseek_v41_k2i.py",
        ],
        "e2e_tests": ["test/e2e/test_deepseek_v41_server.py"],
        "microbench": [
            "torchrun --nproc-per-node 4 test/kernels/deepseek_v41/time_layer.py --widths 6,12"
        ],
        "evals": TEXT_EVALS,
        # DSpark verify steps of 6 rows a request: 8 requests fill the 48-row width
        "concurrency": [1, 2, 4, 8, 16],
        "gate_concurrency": [1, 8],
    },
    "deepseek_v4": {
        "name": "DeepSeek-V4-Pro",
        "packages": ["deepseek_v4"],
        "tp": 8,
        "paths": {"DEEPSEEK_V4_PRO_MODEL_PATH": f"{HF}/deepseek-ai/DeepSeek-V4-Pro"},
        # the README's server: SGLang's two-pool fp8 unified KV
        "model_env": "DEEPSEEK_V4_PRO_MODEL_PATH",
        "env": {
            "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
            "SGLANG_DSV4_UNIFIED_KV_FP8": "1",
            "AITER_BF16_FP8_MOE_BOUND": "0",
        },
        "args": [
            "--tp",
            "8",
            "--attention-backend",
            "dsv4",
            "--enable-deepseek-v4-fp4-indexer",
            "--page-size",
            "256",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--disable-shared-experts-fusion",
            "--cuda-graph-max-bs-decode",
            "8",
            "--max-running-requests",
            "16",
        ],
        "kernel_tests": ["test/kernels/test_deepseek_v4_compile.py"],
        "e2e_tests": [],
        "microbench": [],
        "evals": TEXT_EVALS,
        # decode steps of up to 8 rows
        "concurrency": [1, 2, 4, 8, 16],
        "gate_concurrency": [1, 8],
    },
    "glm53": {
        "name": "GLM-5.3",
        # Phase 1 of the nightly
        "nightly": True,
        # set from the first weekly report
        "baseline": {},
        "packages": ["glm53"],
        "tp": 8,
        "paths": {"GLM_53_MODEL_PATH": f"{HF}/zai-org/GLM-5.3"},
        "recipe": ("glm53_server", "ARGS"),
        "kernel_tests": ["test/kernels/test_glm53_kernels.py"],
        "e2e_tests": ["test/e2e/test_glm53_server.py"],
        "microbench": [
            "torchrun --nproc-per-node 8 test/kernels/glm53/time_layers.py --kind model --rows 2,4,8"
        ],
        "evals": TEXT_EVALS,
        # decode steps of up to 8 rows
        "concurrency": [1, 2, 4, 8, 16],
        "gate_concurrency": [1, 8],
    },
    "glm52_mxfp4": {
        "name": "GLM-5.2 (MXFP4)",
        "packages": ["glm53"],
        "tp": 8,
        # GLM-5.3's recipe: the spec tells the checkpoints apart
        "paths": {"GLM_53_MODEL_PATH": f"{HF}/amd/GLM-5.2-MXFP4"},
        "recipe": ("glm53_server", "ARGS"),
        # test_glm53_kernels.py, its MXFP4 cases included, runs under glm53
        "kernel_tests": [],
        "e2e_tests": ["test/e2e/test_glm53_server.py"],
        "microbench": [
            "torchrun --nproc-per-node 8 test/kernels/glm53/time_layers.py --kind model --rows 2,4,8 --mxfp4 --native-fp4"
        ],
        "evals": TEXT_EVALS,
        # decode steps of up to 8 rows
        "concurrency": [1, 2, 4, 8, 16],
        "gate_concurrency": [1, 8],
    },
    "glm53_mxfp4": {
        "name": "GLM-5.3 (MXFP4)",
        "packages": ["glm53"],
        "tp": 8,
        "paths": {"GLM_53_MODEL_PATH": f"{HF}/amd/GLM-5.3-Quark-MXFP4-AttnFP8"},
        "recipe": ("glm53_server", "ARGS"),
        # test_glm53_kernels.py, its MXFP4 cases included, runs under glm53
        "kernel_tests": [],
        "e2e_tests": ["test/e2e/test_glm53_server.py"],
        "microbench": [
            "torchrun --nproc-per-node 8 test/kernels/glm53/time_layers.py --kind model --rows 2,4,8 --mxfp4"
        ],
        "evals": TEXT_EVALS,
        # decode steps of up to 8 rows
        "concurrency": [1, 2, 4, 8, 16],
        "gate_concurrency": [1, 8],
    },
    "glm53_flash": {
        "name": "GLM-5.3-Flash",
        # Phase 1 of the nightly
        "nightly": True,
        # set from the first weekly report
        "baseline": {},
        "packages": ["glm53_flash"],
        "tp": 4,
        # amd/GLM-5.3-Flash-Quark-MXFP4: not on shared storage; downloaded by hand
        # into both nodes' mirror
        "paths": {"GLM_53_FLASH_MODEL_PATH": f"{HF}/amd/GLM-5.3-Flash-Quark-MXFP4"},
        "recipe": ("glm53_flash_server", "ARGS"),
        "kernel_tests": ["test/kernels/test_glm53_flash_kernels.py"],
        "e2e_tests": ["test/e2e/test_glm53_flash_server.py"],
        "microbench": [
            "torchrun --nproc-per-node 4 test/kernels/glm53_flash/time_layers.py --kind model --rows 2,4,8"
        ],
        "evals": TEXT_EVALS,
        "concurrency": [1, 2, 4, 8, 16],
        "gate_concurrency": [1, 8],
    },
    "kimi_k3": {
        "name": "Kimi-K3",
        # Phase 1 of the nightly
        "nightly": True,
        # set from the first weekly report
        "baseline": {},
        "packages": ["kimi_k3"],
        "tp": 8,
        "paths": {
            "KIMI_K3_MODEL_PATH": f"{HF}/moonshotai/Kimi-K3",
            "KIMI_K3_DSPARK_MODEL_PATH": f"{HF}/RadixArk/Kimi-K3-DSpark",
        },
        "recipe": ("kimi_k3_server", "ARGS"),
        "kernel_tests": [
            "test/kernels/test_kimi_k3_kernels.py",
            "test/kernels/test_kimi_k3_k1.py",
            "test/kernels/test_kimi_k3_k1_conv.py",
            "test/kernels/test_kimi_k3_mla_pre.py",
        ],
        "e2e_tests": ["test/e2e/test_kimi_k3_server.py"],
        "microbench": [
            "torchrun --nproc-per-node 8 test/kernels/kimi_k3/time_k2.py --rows 8 --flush 1024"
        ],
        "evals": TEXT_EVALS + ["mmmu_pro"],
        # the recipe runs one request a step (DSpark verify steps of 8 rows)
        "concurrency": [1],
        "gate_concurrency": [1],
    },
    "kimi_k31": {
        "name": "Kimi-K3.1",
        "packages": ["kimi_k31", "kimi_k3"],
        "tp": 8,
        # moonshotai/Kimi-K3.1; not on the CI nodes yet, and no serving recipe
        "paths": {},
        "kernel_tests": ["test/kernels/test_kimi_k31_k1_ring.py"],
        "e2e_tests": [],
        "microbench": [
            "torchrun --nproc-per-node 8 test/kernels/kimi_k31/layer_bench.py --rows 16,128"
        ],
        "evals": [],
        "concurrency": [],
        "gate_concurrency": [],
    },
    "minimax_m3": {
        "name": "MiniMax-M3 (MXFP4)",
        # SGLang refuses --enable-amd-monokernel for it until both merge
        "blocked": "needs SGLang's page16 SHUFFLE K/V (sgl-project/sglang#41998, #41708)",
        "packages": ["minimax_m3", "minimax"],
        "tp": 4,
        "paths": {
            "MINIMAX_M3_MODEL_PATH": f"{HF}/amd/MiniMax-M3-MXFP4",
            "MINIMAX_M3_EAGLE3_PATH": f"{HF}/Inferact/MiniMax-M3-EAGLE3-GQA",
        },
        "recipe": ("minimax_m3_server", "ARGS"),
        "kernel_tests": [
            "test/kernels/test_minimax_m3_compile.py",
            "test/kernels/test_minimax_m3_rows.py",
        ],
        "e2e_tests": ["test/e2e/test_minimax_m3_server.py"],
        "microbench": [],
        "evals": TEXT_EVALS,
        # EAGLE3 verify steps of 4 rows a request: 4 requests fill the 16-row width
        "concurrency": [1, 2, 4, 8],
        "gate_concurrency": [1, 4],
    },
    "minimax_m31": {
        "name": "MiniMax-M3.1-Flash",
        "packages": ["minimax_m31", "minimax"],
        "tp": 4,
        # the launches cover MI350X only; the CI nodes are MI355X
        "gpus": ["mi350x"],
        "paths": {},
        "kernel_tests": ["test/kernels/test_minimax_m31_ffn.py"],
        "e2e_tests": ["test/e2e/test_minimax_m31_server.py"],
        "microbench": [],
        "evals": [],
        "concurrency": [],
        "gate_concurrency": [],
    },
    "qwen38_flash_next": {
        "name": "Qwen3.8-Flash-Next (MXFP4)",
        # Phase 1 of the nightly
        "nightly": True,
        # set from the first weekly report
        "baseline": {},
        "packages": ["qwen38_flash_next"],
        "tp": 1,
        # amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8: not on shared storage;
        # downloaded by hand into both nodes' mirror
        "paths": {
            "QWEN38_FLASH_NEXT_MODEL_PATH": f"{HF}/amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8"
        },
        "recipe": ("qwen38_flash_next_server", "ARGS"),
        "kernel_tests": ["test/kernels/test_qwen38_flash_next_launches.py"],
        "e2e_tests": ["test/e2e/test_qwen38_flash_next_server.py"],
        "microbench": [],
        "evals": TEXT_EVALS,
        # NEXTN verify steps of 4 rows a request, up to 8 requests
        "concurrency": [1, 2, 4, 8, 16],
        "gate_concurrency": [1, 8],
    },
    "muse_spark": {
        "name": "Muse-Spark",
        "packages": ["muse_spark"],
        "tp": 8,
        # no serving recipe under test/e2e yet
        "paths": {},
        "kernel_tests": [
            "test/kernels/test_muse_spark_k1.py",
            "test/kernels/test_muse_spark_k2.py",
            "test/kernels/test_muse_spark_k3.py",
            "test/kernels/test_muse_spark_k3_tp8.py",
            "test/kernels/test_muse_spark_kattn.py",
            "test/kernels/test_muse_spark_layer.py",
            "test/kernels/test_muse_spark_dattn.py",
            "test/kernels/test_muse_spark_dhalf.py",
            "test/kernels/test_muse_spark_dmlp.py",
        ],
        "e2e_tests": [],
        "microbench": [],
        "evals": [],
        "concurrency": [],
        "gate_concurrency": [],
    },
}


if __name__ == "__main__":
    import sys

    # `python3 models.py paths MODEL`: the checkpoints a job mirrors, one a line
    if len(sys.argv) == 3 and sys.argv[1] == "paths":
        print("\n".join(MODELS.get(sys.argv[2], {}).get("paths", {}).values()))
    # `python3 models.py checkpoints`: every supported checkpoint, one a line
    elif sys.argv[1:] == ["checkpoints"]:
        print("\n".join(CHECKPOINTS))
    # `python3 models.py nightly` (the weekly run's too): the nightly's matrix, as JSON
    elif sys.argv[1:] == ["nightly"]:
        import json

        keys = ["common", *(k for k, m in MODELS.items() if m.get("nightly"))]
        print(json.dumps([{"model": k, "tier": "full"} for k in keys]))
    else:
        sys.exit("usage: models.py paths MODEL | checkpoints | nightly")
