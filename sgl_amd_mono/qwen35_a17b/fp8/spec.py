# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-397B-A17B block-FP8 on gfx950, TP4 and TP8.

Decode steps of 1 to 8 rows, and with EAGLE/MTP the target-verify steps of 1 to
8 requests x 4 draft tokens (4 to 32 rows), open with two guarded peer barriers, then run each
layer as K1 and K2. A GDN layer's K1 is the previous layer's MoE all-reduce,
fused with the residual add, the input norm and the FP8 quant of its output,
then the block-FP8 projections, the conv and the gated delta, whose gated norm
also quantizes the core. K2 is the output projection, its all-reduce fused with
the post-norm, and the block-FP8 MoE. A full-attention layer takes the same
fused input into its block-FP8 ``qkv_proj``, runs SGLang's q/k norm, rope and
aiter attention, which writes the KV cache, and then the same K2 with the output
gate and its FP8 quant ahead of ``o_proj``. A verify step's GDN K1 runs each
request's draft chain and writes SGLang's per-token conv and SSM scratch, which
SGLang's accept step commits from. Wider steps, and anything this spec does not
declare, keep the standard forward.
"""

from __future__ import annotations

import importlib.util
import os

import torch

try:
    from sglang.kernels.plugin_kernels.amd.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.qwen35_a17b.fp8.gfx950.layout import (
    BLOCK,
    EXPERTS,
    HIDDEN,
    LAYERS,
    ROWS,
    VERIFY_TOKENS,
    WIDTHS,
)
from sgl_amd_mono.qwen35_a17b.fp8.gfx950.shapes import BY_O_K, TP_SIZES, for_tp

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "enable_linear_replayssm",
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
    ),
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/runtime_context.py": ("get_parallel",),
}

NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "k1.conv.cosine": 0.999,
        "k1.iconv.cosine": 0.999,
        "k1.state.cosine": 0.999,
        "k1.proj.cosine": 0.99,
        "k1.proj.rel": 0.05,
        "k2.proj.cosine": 0.99,
        "k2.proj.rel": 0.05,
        "k2.moe.cosine": 0.99,
        "k2.moe.rel": 0.05,
        "k1.core.cosine": 0.999,
        "k2.moe_nd.cosine": 0.9,
        "k2.moe_nd.rel": 0.5,
    },
)

_TEXT = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": HIDDEN,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": EXPERTS,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "rms_norm_eps": 1e-6,
    "vocab_size": 248320,
    "hidden_act": "silu",
    "attn_output_gate": True,
    "full_attention_interval": 4,
}
_FULL = frozenset({"full_attention", "attention"})
_LINEAR = frozenset({"linear_attention"})
_FP8 = torch.float8_e4m3fn


def _text(config):
    return getattr(config, "text_config", config)


def _quant(config) -> dict:
    raw = getattr(config, "quantization_config", None) or getattr(
        _text(config), "quantization_config", None
    )
    return dict(raw or {})


def _speculative_reasons(cfg) -> list[str]:
    if getattr(cfg, "speculative_algorithm", None) is None:
        return []
    reasons = []
    topk = cfg.speculative_eagle_topk or 1
    tokens, steps = cfg.speculative_num_draft_tokens, cfg.speculative_num_steps
    if topk != 1 or tokens != VERIFY_TOKENS or (steps or 0) + 1 != tokens:
        reasons.append(
            "the verify step runs one draft chain of "
            f"{VERIFY_TOKENS} tokens: needs --speculative-eagle-topk 1, "
            f"--speculative-num-steps {VERIFY_TOKENS - 1} and "
            f"--speculative-num-draft-tokens {VERIFY_TOKENS}, got topk {topk}, "
            f"{steps} steps, {tokens} tokens"
        )
    if getattr(cfg, "enable_linear_replayssm", False):
        reasons.append(
            "--enable-linear-replayssm: the verify step writes SGLang's per-token "
            "SSM scratch, which ReplaySSM replaces with a ring"
        )
    return reasons


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    _install_block_fp8_linear()
    reasons = _speculative_reasons(cfg) if cfg is not None else []
    text = _text(hf_config)
    reasons += [
        f"needs Qwen3.5-397B-A17B's {key} = {want!r}, got {getattr(text, key, None)!r}"
        for key, want in _TEXT.items()
        if getattr(text, key, None) != want
    ]
    # Some loaders leave layer_types empty until the modules exist. The model
    # check then reads the real GDN and full-attention layers.
    kinds = list(getattr(text, "layer_types", None) or [])
    if kinds and (
        len(kinds) != LAYERS
        or any(
            (kinds[i] not in _FULL) if i % 4 == 3 else (kinds[i] not in _LINEAR)
            for i in range(len(kinds))
        )
    ):
        reasons.append(
            "needs 60 layers, linear attention except every 4th layer "
            f"(full attention), got {kinds[:8]}"
        )
    rope = (
        getattr(text, "rope_parameters", None)
        or getattr(text, "rope_scaling", None)
        or {}
    )
    section = list(rope.get("mrope_section") or [])
    if section != [11, 11, 10] or rope.get("partial_rotary_factor") != 0.25:
        reasons.append(
            "needs mRoPE sections [11, 11, 10] and partial_rotary_factor 0.25 "
            f"(rotary dim 64), got {section} and {rope.get('partial_rotary_factor')!r}"
        )
    if rope.get("rope_theta") != 10000000:
        reasons.append(f"needs rope_theta 10000000, got {rope.get('rope_theta')!r}")
    quant = _quant(hf_config)
    method = quant.get("quant_method")
    block = list(quant.get("weight_block_size") or [])
    if (
        method != "fp8"
        or quant.get("activation_scheme") != "dynamic"
        or block != [BLOCK, BLOCK]
    ):
        reasons.append(
            "needs dynamic block-FP8 weights with weight_block_size [128, 128], "
            f"got quant_method {method!r}, activation_scheme "
            f"{quant.get('activation_scheme')!r}, weight_block_size {block}"
        )
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _shape(tensor):
    return tuple(getattr(tensor, "shape", ()))


def _dense_block(module, n, k, what) -> list[str]:
    weight = getattr(module, "weight", None)
    scale = getattr(module, "weight_scale_inv", None)
    if (
        weight is None
        or getattr(weight, "dtype", None) != _FP8
        or _shape(weight) != (n, k)
    ):
        return [f"{what} weight is not block-FP8 [{n}, {k}]"]
    if (
        scale is None
        or getattr(scale, "dtype", None) != torch.float32
        or _shape(scale) != (n // BLOCK, k // BLOCK)
    ):
        return [f"{what} scale is not a fp32 {BLOCK}x{BLOCK} block scale"]
    if getattr(module, "bias", None) is not None:
        return [f"{what} has a bias, the block-FP8 GEMV adds none"]
    return []


def _floating(module, shape, what) -> list[str]:
    weight = getattr(module, "weight", None)
    dtype = getattr(weight, "dtype", None)
    if weight is None or dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return [f"{what} is not an unquantized floating weight"]
    if shape is not None and _shape(weight) != shape:
        return [f"{what} weight is {_shape(weight)}, wants {shape}"]
    return []


def _expert_block(weight, scale, what, *, k_last: int) -> list[str]:
    shape = _shape(weight)
    if (
        getattr(weight, "dtype", None) != _FP8
        or len(shape) != 3
        or shape[2] != k_last
        or shape[1] % BLOCK
        or shape[0] < EXPERTS
    ):
        return [f"{what} weight is not block-FP8 [E, N, {k_last}]"]
    want = (shape[0], shape[1] // BLOCK, shape[2] // BLOCK)
    if (
        scale is None
        or getattr(scale, "dtype", None) != torch.float32
        or _shape(scale) != want
    ):
        return [f"{what} scale is not a fp32 {BLOCK}x{BLOCK} block scale {want}"]
    return []


def _projection_k(layers) -> int | None:
    for layer in layers:
        attn = getattr(layer, "linear_attn", None)
        weight = getattr(getattr(attn, "out_proj", None), "weight", None)
        shape = _shape(weight)
        if len(shape) == 2:
            return shape[1]
        weight = getattr(getattr(layer, "o_proj", None), "weight", None)
        shape = _shape(weight)
        if len(shape) == 2:
            return shape[1]
    return None


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    del hf_config, copies
    inner = getattr(model, "model", None)
    layers = list(getattr(inner, "layers", []) or [])
    if len(layers) != LAYERS:
        return [f"needs {LAYERS} decoder layers, got {len(layers)}"]
    layout = BY_O_K.get(_projection_k(layers))
    if layout is None:
        return [
            "needs a TP4 shard (projection K 2048) or a TP8 shard "
            f"(projection K 1024), got K {_projection_k(layers)}"
        ]
    reasons = []
    # The fused norms take the weight as Gemma's 1 + w
    norms = [
        (f"layer {i} {name}", getattr(layer, name, None))
        for i, layer in enumerate(layers)
        for name in ("input_layernorm", "post_attention_layernorm")
    ]
    norms.append(("the final norm", getattr(inner, "norm", None)))
    reasons += [
        f"{what} is {type(norm).__name__}, not a Gemma RMSNorm"
        for what, norm in norms
        if type(norm).__name__ != "GemmaRMSNorm"
    ]
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not getattr(mlp, "enable_shared_expert_fusion", False):
            reasons.append(
                f"layer {i}: the shared expert is not fused into the block-FP8 MoE"
            )
            continue
        if getattr(mlp, "reduce_results", True):
            reasons.append(
                f"layer {i}: the MoE all-reduce is inside the block, K2 owns it"
            )
        experts = getattr(mlp, "experts", None)
        w13 = getattr(experts, "w13_weight", None)
        reasons += _expert_block(
            w13,
            getattr(experts, "w13_weight_scale_inv", None),
            f"layer {i} w13",
            k_last=HIDDEN,
        )
        if _shape(w13)[1:2] != (layout.w13_n,):
            reasons.append(
                f"layer {i} w13 N is {_shape(w13)[1:2]}, TP{layout.tp} wants {(layout.w13_n,)}"
            )
        w2 = getattr(experts, "w2_weight", None)
        w2_scale = getattr(experts, "w2_weight_scale_inv", None)
        w2_shape = _shape(w2)
        if (
            getattr(w2, "dtype", None) != _FP8
            or len(w2_shape) != 3
            or w2_shape[1] != HIDDEN
            or w2_shape[2] != layout.intermediate
        ):
            reasons.append(
                f"layer {i} w2 weight is not block-FP8 [E, {HIDDEN}, {layout.intermediate}]"
            )
        elif (
            w2_scale is None
            or getattr(w2_scale, "dtype", None) != torch.float32
            or _shape(w2_scale) != (w2_shape[0], HIDDEN // BLOCK, w2_shape[2] // BLOCK)
        ):
            reasons.append(
                f"layer {i} w2 scale is not a fp32 {BLOCK}x{BLOCK} block scale"
            )
        reasons += _floating(
            getattr(mlp, "gate", None), (EXPERTS, HIDDEN), f"layer {i} router"
        )
        if hasattr(layer, "linear_attn"):
            if i % 4 == 3:
                reasons.append(
                    f"layer {i}: a full-attention layer is registered as GDN"
                )
            attn = layer.linear_attn
            reasons += _dense_block(
                attn.in_proj_qkvz, layout.qkvz_n, HIDDEN, f"layer {i} in_proj_qkvz"
            )
            reasons += _floating(
                attn.in_proj_ba, (layout.ba_n, HIDDEN), f"layer {i} in_proj_ba"
            )
            conv = _shape(getattr(getattr(attn, "conv1d", None), "weight", None))
            if conv not in ((layout.conv_dim, 1, 4), (layout.conv_dim, 4)):
                reasons.append(
                    f"layer {i} conv1d weight is {conv}, "
                    f"TP{layout.tp} wants {(layout.conv_dim, 1, 4)}"
                )
            reasons += _dense_block(
                attn.out_proj, HIDDEN, layout.o_k, f"layer {i} out_proj"
            )
        elif hasattr(layer, "o_proj"):
            if i % 4 != 3:
                reasons.append(
                    f"layer {i}: a GDN layer is registered as full attention"
                )
            qkv = getattr(layer, "qkv_proj", None)
            qkv_weight = getattr(qkv, "weight", None)
            qkv_n = _shape(qkv_weight)[:1]
            if len(qkv_n) != 1 or qkv_n[0] != layout.qkv_n:
                reasons.append(
                    f"layer {i} qkv N is {qkv_n[:1]}, TP{layout.tp} wants {layout.qkv_n}"
                )
            else:
                reasons += _dense_block(qkv, qkv_n[0], HIDDEN, f"layer {i} qkv")
            reasons += _dense_block(
                layer.o_proj, HIDDEN, layout.o_k, f"layer {i} o_proj"
            )
        else:
            reasons.append(f"layer {i}: neither a GDN layer nor a full-attention layer")
    return reasons


class _Built:
    """One rank's barriers and the decode forward."""

    def __init__(self, model, runtime, layout):
        self.model = model
        self.runtime = runtime
        self.layout = layout
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        """``Qwen3_5MoeForConditionalGeneration.forward`` for a routed decode or
        target-verify step."""
        from sgl_amd_mono.qwen35_a17b.fp8.gfx950.forward import (
            final_norm,
            run_attn,
            run_k1,
            run_k1_verify,
            run_k2,
        )
        from sgl_amd_mono.qwen35_a17b.fp8.gfx950.verify import verify_slots

        lm = self.model
        contains = getattr(forward_batch, "contains_image_inputs", None)
        key = plan.key
        verify = key.forward_mode == "target_verify"
        if verify:
            served = key.tokens_per_request == VERIFY_TOKENS and key.rows <= WIDTHS[-1]
        else:
            served = key.forward_mode == "decode" and key.rows <= ROWS[-1]
        if (
            get_embedding
            or pp_proxy_tensors is not None
            or not served
            or (contains is not None and contains())
        ):
            return type(lm).forward(
                lm, input_ids, positions, forward_batch, get_embedding, pp_proxy_tensors
            )
        if getattr(lm, "is_mrope_enabled", False):
            positions = forward_batch.mrope_positions
        inner = lm.model
        self.runtime.advance()
        self.runtime.enter(0)
        self.runtime.enter(1)
        hidden = inner.embed_tokens(input_ids)
        rows = plan.key.rows
        if hidden.shape[0] != rows:
            raise RuntimeError(
                f"decode step has {hidden.shape[0]} rows, the routed width is {rows}"
            )
        step_slots = (
            verify_slots(forward_batch, rows // key.tokens_per_request)
            if verify
            else None
        )
        residual = None
        start = getattr(inner, "start_layer", 0)
        end = getattr(inner, "end_layer", len(inner.layers))
        for i in range(start, end):
            layer = inner.layers[i]
            if hasattr(layer, "linear_attn"):
                if verify:
                    core, residual = run_k1_verify(
                        layer,
                        hidden,
                        residual,
                        forward_batch,
                        rows,
                        key.tokens_per_request,
                        step_slots,
                        self.check,
                        self.layout,
                    )
                else:
                    core, residual = run_k1(
                        layer,
                        hidden,
                        residual,
                        forward_batch,
                        rows,
                        self.check,
                        self.layout,
                    )
                proj = layer.linear_attn.out_proj
            else:
                core, residual = run_attn(
                    layer, hidden, residual, positions, forward_batch
                )
                proj = layer.o_proj
            hidden, residual = run_k2(
                layer, core, residual, forward_batch, rows, proj, self.check
            )
        if residual is None:
            hidden = inner.norm(hidden)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        else:
            hidden = final_norm(hidden, residual, inner.norm)
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, None)


def _build(model, hf_config, ctx):
    del hf_config
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.qwen35_a17b.fp8.gfx950.barrier import PhaseRuntime

    if ctx.isa != "gfx950":
        raise ValueError(f"Qwen3.5 block-FP8 monokernel is gfx950 only, got {ctx.isa}")
    parallel = get_parallel()
    tp = int(parallel.tp_size)
    if tp not in TP_SIZES:
        raise ValueError(
            f"Qwen3.5 block-FP8 monokernel supports TP{list(TP_SIZES)}, got TP{tp}"
        )
    _install_block_fp8_linear()
    layout = for_tp(tp)
    device = model.model.embed_tokens.weight.device
    runtime = PhaseRuntime(
        int(parallel.tp_rank),
        parallel.tp_group.cpu_group,
        device,
        ctx.deadline_s,
        tp,
    )
    return _Built(model, runtime, layout)


SPEC = MonoSpec(
    name="qwen35_a17b_fp8",
    devices={
        "gfx950": DeviceLimits(widths=WIDTHS, tp_sizes=TP_SIZES, resident_blocks=BLOCKS)
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={"moe_runner_backend": frozenset({"auto", "aiter"})},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,
    config_disable_reasons=_config_disable_reasons,
)


def _install_block_fp8_linear() -> None:
    # Prefill uses SGLang's block-FP8 linear, whose gfx950 small-M helper imports
    # symbols that module no longer defines. Install the stand-in before that import.
    from sgl_amd_mono.qwen35_a17b.fp8.gfx950.gemv import _allow_block_fp8_linear

    _allow_block_fp8_linear()
