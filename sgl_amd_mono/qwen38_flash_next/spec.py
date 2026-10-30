# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next (SGLang's Qwen4-Exp) on gfx950: the monokernel spec and its
forward (RFC #43949).

SGLang's runtime switches the model's forward at attach: a routed NEXTN verify
step (1 to 8 requests x 4 tokens, one draft chain) runs ``_Built.forward`` here,
which takes and returns what ``Qwen4ExpForConditionalGeneration.forward`` does.
The whole model runs in 13 launches (``gfx950/runner.py``): layer 0 behind the
prelude (the token embedding and the PLE n-gram history), layers 1-3 behind layer
1's PLE, then 11 launches of 3 Gated DeltaNet + 1 QSA layers, the last followed by
the model-level hyper-connection mixer. SGLang's logits processor closes the step,
and the residual streams after the last layer are the hidden states the MTP draft
reads, as SGLang's forward hands them on.

The kernels read SGLang's weights in place, except the copies logged at startup:
the hyper-connection mixers and the router (with the shared expert's gate) packed
for the MFMA tiles, the GDN layers' A (-exp(A_log)), dt_bias and gated-norm weight
in f32, the FP8 projections SGLang keeps row-major (in_proj_ba: N is not a
multiple of 64) zero-padded in AITER's shuffle, the PLE projections packed, and
the PLE n-gram metadata as one table. The bf16 RoPE table is SGLang's own cast.

A request's verify rows write its draft tokens' KV, index keys and per-token
GDN, PLE and n-gram states, as SGLang's verify does: request slot r owns row r of
every verify scratch. A graph's pad rows write KV slot 0, the reserved dummy slot.

``SGLANG_AMD_MONO_CHECK=1`` checks the weights the launches read of three layers
against the checkpoint at build, then runs SGLang's own forward of every eager
routed step first and logs the differences of the launches' logits and hidden
states (``check.py``); the bounds in ``NUMERICS`` come from these runs.
"""

from __future__ import annotations

import importlib.util
import logging
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
from sgl_amd_mono.qwen38_flash_next import (
    HF_CONFIG,
    INDEXER_CONFIG,
    LAYER_PATTERN,
    MAX_CTX,
)

logger = logging.getLogger(__name__)

MAX_R = 8  # requests a launch takes at most (``gfx950/config.MAX_R``)
TOKENS = 4  # rows a request: NEXTN's 3 draft steps + 1
TP = 1
# the verify widths: whole requests of 4 rows, up to 8 (32 rows)
WIDTHS = tuple(range(TOKENS, MAX_R * TOKENS + 1, TOKENS))
# the kernel's GDN SSM state dtypes (``gfx950/config.MonoBuild.ssm_bf16``)
SSM_DTYPES = (torch.float32, torch.bfloat16)

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "MONOKERNEL_API_VERSION",
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "batch_size",
        "tokens_per_request",
        "forward_mode",
        "widths",
        "deadline_s",
        "copies",
    ),
    "srt/kernel_plugins/amd/binding.py|srt/monokernels/runtime.py": (
        '"target_verify"',
    ),
    "srt/arg_groups/fields/*.py": (
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
        "enable_linear_replayssm",
        "enable_linear_replayssm_spec",
        "ple_offload_embedding",
        "disable_shared_experts_fusion",
        "moe_runner_backend",
        "enable_paged_experts",
        "enable_w4a4_mxfp4_megamoe",
        "expert_distribution_recorder_mode",
        "kv_cache_dtype",
        "enable_hisparse",
        "context_length",
    ),
    # NEXTN resolves to EAGLE
    "srt/arg_groups/speculative_hook.py": ('"EAGLE"', '"NEXTN"'),
    "srt/configs/qwen4_exp.py": (
        "text_config",
        "hc_count",
        "hc_lowrank",
        "layer_types",
        "ple_layer_ids",
        "ple_embed_dim",
        "ple_conv_kernel_size",
        "ngram_size",
        "heads_per_ngram",
    ),
    "srt/configs/qwen3_next.py": (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "partial_rotary_factor",
        "rms_norm_eps",
        "hidden_act",
        "output_gate_type",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "max_position_embeddings",
    ),
    "srt/configs/mamba_utils.py": ("mamba2_state_dtype", "conv", "temporal"),
    "srt/configs/model_config.py": ("context_len",),
    "srt/environ.py": (
        "SGLANG_USE_AITER",
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
        "SGLANG_ENABLE_PP_SPEC",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        "process_model_config",
        '"tp_size"',
        '"pp_size"',
        '"attn_cp_size"',
        '"moe_ep_size"',
    ),
    "srt/layers/rotary_embedding/base.py": ("cos_sin_cache_as",),
    "srt/layers/vocab_parallel_embedding.py": ("weight",),
    "srt/models/qwen3_5.py": (
        "layers",
        "embed_tokens",
        "linear_attn",
        "rotary_emb",
        "start_layer",
        "end_layer",
    ),
    "srt/models/qwen3_vl.py": (
        "model",
        "lm_head",
        "logits_processor",
        "pooler",
        "capture_aux_hidden_states",
        "forward(get_embedding)",
        "forward(pp_proxy_tensors)",
    ),
    "srt/models/qwen4_exp.py": (
        "Qwen4ExpForConditionalGeneration",
        "hyper_connection_mixer",
        "last_hc_hidden_states",
        "is_mrope_enabled",
        "ple",
        "is_qsa",
    ),
    "srt/layers/logits_processor.py": ("hidden_states",),
    "srt/model_executor/forward_batch_info.py": ("mrope_positions",),
}

# The launches' logits and hidden states (the residual streams after the last
# layer) against SGLang's forward of the same eager verify step (check mode), on
# the step's worst row. The launches matched a reference implementation bitwise
# over 534 steps (every launch's stream), so the gap is SGLang's own noise: its
# AITER MXFP4 MoE differs by about 2.6e-3 between two identical forwards, and 48
# layers grow that to a per-row logits rel of about 0.18 between two SGLang
# forwards of one step (0.156 for the launches). The worst row of a step spreads
# wide, so the bounds catch row, slot or request mix-ups, garbage layouts and NaN,
# not a subtly wrong layer. Worst row per step, median / p1 or p99 of 364 steps
# (bs 1-8, triton), worst of 11,824 steps (triton and aiter), and of SGLang
# against itself (an input 1 bf16 ulp off moves SGLang's own hidden rel up to 3.1):
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "verify.logits.cosine": 0.1,  # 0.972 / p1 0.70 / worst 0.28; itself 0.33
        "verify.logits.rel": 2.5,  # 0.25 / p99 0.81 / worst 1.64; itself 1.51
        "verify.hidden.cosine": 0.3,  # 0.977 / p1 0.75 / worst 0.47; itself 0.54
        "verify.hidden.rel": 4.0,  # 0.22 / p99 0.74 / worst 2.19; itself 1.11
    },
)


def _text(hf_config):
    """The text model's config: the VL checkpoint nests it."""
    return getattr(hf_config, "text_config", None) or hf_config


def _has_flydsl() -> bool:
    return importlib.util.find_spec("flydsl") is not None


def _kinds(text) -> list[str]:
    """The kernel's layer kinds: ``gdn`` (linear attention) or ``qsa``."""
    return ["qsa" if t == "full_attention" else "gdn" for t in text.layer_types]


def _context_reason(context_len) -> list[str]:
    if context_len is not None and context_len + 64 > MAX_CTX:
        return [
            f"context length {context_len}: the index selection covers {MAX_CTX - 64}"
            " tokens"
        ]
    return []


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """What ``SPEC.requires`` cannot state: the nested text config (its shapes, layer
    pattern and PLE layer), the checkpoint's quantization, the GDN state dtypes,
    the context length, and FlyDSL."""
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype

    t = _text(hf_config)
    want = {**HF_CONFIG, **INDEXER_CONFIG}
    reasons = [
        f"config {k}={getattr(t, k, None)!r}, the kernel is built for {v!r}"
        for k, v in want.items()
        if getattr(t, k, None) != v
    ]
    layer_types = list(getattr(t, "layer_types", None) or ())
    pattern = list(LAYER_PATTERN) * (HF_CONFIG["num_hidden_layers"] // 4)
    if layer_types != pattern:
        reasons.append(
            "layer_types: the kernel runs 3 linear-attention layers, then a "
            "full-attention one, over 48 layers"
        )
    quant = getattr(hf_config, "quantization_config", None) or {}
    if quant.get("quant_method") != "quark":
        reasons.append(
            f"needs the Quark checkpoint (MXFP4 experts, FP8 attention and GDN "
            f"projections, FP8 PLE table), got {quant.get('quant_method')!r}"
        )
    dtypes = mamba2_state_dtype(hf_config)
    if dtypes.conv != torch.bfloat16 or dtypes.temporal not in SSM_DTYPES:
        reasons.append(
            f"GDN states conv {dtypes.conv}, SSM {dtypes.temporal}: the kernel "
            "reads a bf16 conv state and an f32 or bf16 SSM state"
        )
    reasons += _context_reason(
        getattr(cfg, "context_length", None)
        or getattr(t, "max_position_embeddings", None)  # fmt: skip
    )
    if not _has_flydsl():
        reasons.append("needs FlyDSL")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel, process_model_config

    from sgl_amd_mono.qwen38_flash_next.gfx950 import weights

    m, p = model.model, get_parallel()
    reasons = []
    widths = (p.tp_size, p.pp_size, p.attn_cp_size, p.moe_ep_size)
    if widths != (1, 1, 1, 1):
        reasons.append(
            f"TP {p.tp_size} / PP {p.pp_size} / CP {p.attn_cp_size} / EP "
            f"{p.moe_ep_size}: the launches run the whole model on one GPU"
        )
    if getattr(model, "capture_aux_hidden_states", False):
        reasons.append("auxiliary hidden states (EAGLE3 / DFLASH): none are captured")
    kinds = _kinds(_text(hf_config))
    if m.start_layer != 0 or m.end_layer != len(m.layers) or len(m.layers) != len(kinds):  # fmt: skip
        reasons.append(f"layers {m.start_layer}..{m.end_layer} of {len(m.layers)}")
        return reasons
    for i, (layer, kind) in enumerate(zip(m.layers, kinds)):
        is_gdn = getattr(layer, "linear_attn", None) is not None
        if is_gdn != (kind == "gdn") or (kind == "qsa" and not getattr(layer, "is_qsa", False)):  # fmt: skip
            reasons.append(f"layer {i}: not the {kind} layer the config declares")
            continue
        has_ple = getattr(layer, "ple", None) is not None
        if has_ple != (i == 1):
            reasons.append(f"layer {i}: the kernel runs PLE on layer 1 only")
        reasons += weights.layer_reasons(layer, i, kind)
    if getattr(m.layers[1], "ple", None) is not None:
        reasons += weights.ngram_reasons(m.layers[1].ple)
    reasons += weights.mixer_reasons(getattr(m, "hyper_connection_mixer", None))
    emb = getattr(getattr(m, "embed_tokens", None), "weight", None)
    if (
        not isinstance(emb, torch.Tensor)
        or emb.dtype != torch.bfloat16
        or emb.dim() != 2
        or emb.shape[1] != HF_CONFIG["hidden_size"]
        or not emb.is_contiguous()
    ):
        reasons.append("the token embedding must be one contiguous bf16 table")
    if not hasattr(getattr(m.layers[3], "rotary_emb", None), "cos_sin_cache_as"):
        reasons.append("layer 3: no rotary embedding with a bf16 cos / sin table")
    return reasons + _context_reason(process_model_config().context_len)


class _Built:
    """The monokernels: the runner and the forward."""

    def __init__(self, lm, runner):
        self.lm, self.runner = lm, runner
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1"

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        get_embedding=False,
        pp_proxy_tensors=None,
        *_,
        **__,
    ):
        """``Qwen4ExpForConditionalGeneration.forward`` for a routed verify step."""
        lm, key = self.lm, plan.key
        if (
            key.forward_mode != "target_verify"
            or key.tokens_per_request != TOKENS
            or key.batch_size > MAX_R
        ):
            # an eager step the launches do not take: SGLang's forward
            return type(lm).forward(
                lm, input_ids, positions, forward_batch, get_embedding, pp_proxy_tensors
            )
        checking = self.check and not torch.cuda.is_current_stream_capturing()
        ref = None
        if checking:
            from sgl_amd_mono.qwen38_flash_next import check

            ref = check.reference(
                lm, input_ids, positions, forward_batch, self.runner.kinds
            )
        # the RoPE positions SGLang's forward applies: MRoPE positions [3, rows]
        # where enabled (they differ from the sequence positions with images or
        # video)
        pos = forward_batch.mrope_positions if lm.is_mrope_enabled else positions
        x, hidden = self.runner.launch(forward_batch, pos)
        # the VL model's forward hands the streams on, as SGLang's does
        lm.model.last_hc_hidden_states = hidden
        if get_embedding:
            return lm.pooler(x, forward_batch)
        out = lm.logits_processor(input_ids, x, lm.lm_head, forward_batch, None)
        out.hidden_states = hidden
        if ref is not None:
            check.compare(ref, out, forward_batch)
        return out


def _build(model, hf_config, ctx) -> _Built:
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.qwen38_flash_next.gfx950 import weights
    from sgl_amd_mono.qwen38_flash_next.gfx950.runner import VerifyRunner

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Qwen3.8-Flash-Next monokernel: build inside a capture")
    t, m = _text(hf_config), model.model
    kinds = _kinds(t)
    params = weights.param_ptrs(model)
    layers = {
        i: weights.layer_views(m.layers[i], i, kind, ctx.copies, params)
        for i, kind in enumerate(kinds)
    }
    final = weights.mixer_views(m.hyper_connection_mixer, ctx.copies, params)
    ngram = weights.ngram_views(m.layers[1].ple, ctx.copies, params)
    rope = m.layers[3].rotary_emb
    runner = VerifyRunner(
        kinds=kinds,
        layers=layers,
        final=final,
        ngram=ngram,
        embed_w=m.embed_tokens.weight,
        # SGLang's bf16 cast of its RoPE table, kept while the table is
        cos_sin=lambda: rope.cos_sin_cache_as(torch.bfloat16),
        ntok=TOKENS,
        max_reqs=MAX_R,
        eos_token_id=int(t.eos_token_id),
        device=m.embed_tokens.weight.device,
        fault=guard.fault_injection(0),
    )
    runner.set_deadline(ctx.deadline_s)
    built = _Built(model, runner)
    if built.check:
        from sgl_amd_mono.qwen38_flash_next import check

        check.weights(model, kinds)
    # The launches compile the cache strides in, and SGLang sizes the caches after
    # the build: each batch size's launches compile at its first forward (the
    # decode graph runner's eager warmup), then come from FlyDSL's disk cache.
    logger.info(
        "Qwen3.8-Flash-Next monokernel: %d launches a verify step; each batch size "
        "of %s compiles at its first forward (minutes on a cold FlyDSL cache)",
        len(runner.plans),
        sorted(w // TOKENS for w in ctx.widths),
    )
    return built


SPEC = MonoSpec(
    name="qwen38_flash_next_hip",
    devices={
        "gfx950": DeviceLimits(widths=WIDTHS, tp_sizes=(TP,), resident_blocks=BLOCKS)
    },
    # the launches serve NEXTN target verify steps only
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        # the backends measured on and off (GSM8K, check mode); the launches
        # replace every attention of a routed step, and SGLang runs its GDN + QSA
        # hybrid for the rest
        attention_backends=frozenset({"triton", "aiter"}),
        server_args={
            # NEXTN (SGLang resolves it to EAGLE): verify steps of one draft chain
            # of 3 steps, 4 rows a request
            "speculative_algorithm": frozenset({"EAGLE"}),
            "speculative_eagle_topk": frozenset({1}),
            "speculative_num_steps": frozenset({3}),
            "speculative_num_draft_tokens": frozenset({4}),
            # the launches read the committed GDN state and write every draft
            # token's state to the verify scratch; ReplaySSM keeps neither
            "enable_linear_replayssm": frozenset({False}),
            "enable_linear_replayssm_spec": frozenset({False}),
            # the prelude gathers the PLE n-gram table on the GPU
            "ple_offload_embedding": frozenset({None, False}),
            # the shared expert runs as expert 512 of the fused MXFP4 experts
            "disable_shared_experts_fusion": frozenset({False}),
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # both lay the experts out otherwise
            "enable_paged_experts": frozenset({False}),
            "enable_w4a4_mxfp4_megamoe": frozenset({False}),
            # the launches record no expert distribution
            "expert_distribution_recorder_mode": frozenset({None}),
            # the launches write bf16 K / V
            "kv_cache_dtype": frozenset({"auto", "bfloat16"}),
            # HiSparse maps the logical KV slots to device rows; the launches
            # write raw slots
            "enable_hisparse": frozenset({False}),
        },
        env={
            # AITER's shuffled MXFP4 experts and FP8 projections, read in place
            "SGLANG_USE_AITER": True,
            # benchmark routing overrides: the launches route with the model's router
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
            # PP spec remaps verify scratch rows; the launches index request r's
            # intermediate SSM / conv / PLE rows as row r
            "SGLANG_ENABLE_PP_SPEC": False,
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
