# SPDX-License-Identifier: Apache-2.0
"""The monokernel spec of Qwen3.5-397B-A17B (``Qwen3_5MoeForConditionalGeneration``).

SGLang registers one spec per architecture. Block-FP8 is served by ``fp8``.
The MXFP4 verify path is a separate package and is refused here until it moves
under ``qwen35_a17b/mxfp4``.
"""

from __future__ import annotations

try:
    from sglang.kernels.plugin_kernels.amd.spec import MonoSpec, Requirements
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import MonoSpec, Requirements

from sgl_amd_mono.qwen35_a17b.config import is_block_fp8, quant_of
from sgl_amd_mono.qwen35_a17b.fp8 import spec as fp8

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "MonoSpec",
        "Requirements",
    ),
}


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    if not is_block_fp8(hf_config):
        method = quant_of(hf_config).get("quant_method")
        return [
            "Qwen3.5 family serves dynamic 128x128 block-FP8; "
            f"quant_method {method!r} is not in this package (MXFP4 stays on its own path)"
        ]
    fp8._install_block_fp8_linear()
    return fp8.SPEC.config_disable_reasons(cfg, hf_config)


def _model_disable_reasons(model, hf_config, copies) -> list[str]:
    if not is_block_fp8(hf_config):
        return _config_disable_reasons(None, hf_config)
    return fp8.SPEC.model_disable_reasons(model, hf_config, copies)


def _build(model, hf_config, ctx):
    if not is_block_fp8(hf_config):
        method = quant_of(hf_config).get("quant_method")
        raise ValueError(
            f"Qwen3.5 family has no monokernel for quant_method {method!r}"
        )
    fp8._install_block_fp8_linear()
    return fp8.SPEC.build(model, hf_config, ctx)


SPEC = MonoSpec(
    name="qwen35_a17b",
    devices=fp8.SPEC.devices,
    features=fp8.SPEC.features,
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={"moe_runner_backend": frozenset({"auto", "aiter"})},
    ),
    numerics=fp8.SPEC.numerics,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,
    config_disable_reasons=_config_disable_reasons,
)
