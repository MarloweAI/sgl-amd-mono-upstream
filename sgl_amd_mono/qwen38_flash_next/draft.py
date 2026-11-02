# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next's MTP draft on gfx950: its spec and forward (RFC #43949).

The NEXTN draft runs its own steps: a draft extend of R requests x 4 rows
(DRAFT_EXTEND_V2) and the chain's draft decode steps of R x 1 (DECODE on the
draft's runner). Monokernel API version 1 routes a draft model by the target's
key, ("target_verify", R, draft tokens), which the draft never runs, so the spec
routes the draft's own steps itself: at build, ``_route_draft_steps`` puts a
wrapper in front of the model's forward, before the runtime installs its own
over it. The wrapper routes a step by its forward mode, batch size, rows and
hidden-state buffer alone, so a captured graph replays the decision made at
capture, as a routed step does. It relies on every draft call site calling the
instance attribute ``model.forward`` (pinned by a unit test), and does itself
what the runtime does for a routed step, checking the draft's attention backend
("qsa") once per step mode. It can move to SGLang if a later monokernel API
routes a draft's own steps.

Every draft extend and draft decode step of 1 to 8 requests runs the launch: one
the launch cannot read raises instead of running SGLang's forward; larger steps
run SGLang's forward. ``_build`` turns SGLang's QSA MTP index sharing off for the
draft: with it on, SGLang's draft decode reuses the selection its own draft extend
stored, which a launched extend never writes, so a batch growing past 8 requests
would decode with a stale selection. SGLang's capture check never sees a draft
step, so this module logs each draft key the first time it launches. Check mode
(``SGLANG_AMD_MONO_CHECK``) does not cover the draft yet, and fault injection arms
the target's launches only.

A draft step is one launch (``gfx950/runner.DraftRunner``): the MTP input fusion
(the token embedding and the target's hyper-connection hidden states, each
normalized and projected), the QSA decoder layer and the draft's mixer;
``_Draft.forward`` returns what ``Qwen4ExpForCausalLMMTP.forward`` does. The
weights are SGLang's, read in place, except the copies logged at startup: the
hyper-connection mixers, router and input-fusion projections packed for the MFMA
tiles, and the FP8 projections SGLang keeps row-major, zero-padded in AITER's
shuffle.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

try:
    from sglang.kernels.plugin_kernels.amd.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Requirements,
    )

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.qwen38_flash_next import spec as target

logger = logging.getLogger(__name__)

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Requirements",
    ),
    "srt/models/qwen4_exp_mtp.py": (
        "Qwen4ExpForCausalLMMTP",
        "model",
        "lm_head",
        "logits_processor",
        "_set_hc_logits_hidden_states",
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
    ),
    "srt/models/qwen3_5.py": ("layers", "embed_tokens", "rotary_emb"),
    "srt/layers/radix_attention.py": ("layer_id",),
    "srt/model_executor/forward_batch_info.py": (
        "spec_info",
        "forward_mode",
        "batch_size",
        "is_draft_extend_v2",
        "is_decode",
    ),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/base_attn_backend.py": ("decode_attention_backend_str",),
    "srt/layers/attention/qsa/glue.py": ("resolve_qsa_sparse_backend",),
    "srt/layers/attention/qwen_sparse_attn_backend.py": ("_mtp_shared_sparse_indices",),
    "srt/configs/qwen4_exp.py": ("text_config", "index_share_for_mtp_iteration"),
    # the draft worker reads the flag from the draft's config, text_config first,
    # when it sets up the draft's attention backends, after the draft's build
    "srt/speculative/eagle_worker_v2.py": (
        '"index_share_for_mtp_iteration"',
        "_configure_qsa_mtp_index_share",
    ),
    "srt/speculative/draft_utils.py": ('"qsa"',),
    "srt/speculative/*.py": ("hidden_states",),
    "srt/runtime_context.py": ("get_parallel", '"tp_size"'),
}

# The draft runs on its own QSA backends whatever --speculative-draft-attention-backend
# says (speculative/draft_utils.py stamps them "qsa"); the launch writes their caches.
DRAFT_ATTENTION_BACKENDS = frozenset({"qsa"})


def _tokens_per_request(model) -> int:
    """A draft extend's rows a request: the target's verify rows."""
    return target.TOKENS


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.qwen38_flash_next.gfx950 import weights

    reasons = []
    if get_parallel().tp_size != 1:
        reasons.append("the draft launch runs the MTP layer on one GPU")
    return reasons + weights.mtp_reasons(model)


def _step_backend() -> Optional[str]:
    """The decode attention backend the current step runs on."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    return get_attn_backend().decode_attention_backend_str


def _shares_mtp_sparse_indices() -> bool:
    """Whether the current step's QSA backend holds SGLang's shared MTP selection
    (index sharing on)."""
    from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = resolve_qsa_sparse_backend(get_attn_backend())
    return getattr(backend, "_mtp_shared_sparse_indices", None) is not None


def _turn_off_mtp_index_share(hf_config) -> None:
    """Turn SGLang's QSA MTP index sharing off for the draft (see the module
    docstring). The draft worker reads the flag from the draft's own config,
    text_config first, after the build and before it sets up the draft's
    attention backends, so no shared selection is ever created."""
    text = target._text(hf_config)
    was = getattr(text, "index_share_for_mtp_iteration", None)
    for config in (text, hf_config):
        config.index_share_for_mtp_iteration = False
    logger.info(
        "monokernel qwen38_flash_next MTP draft: SGLang's QSA MTP index sharing off "
        "(index_share_for_mtp_iteration %r -> False), so every draft decode "
        "selects its own blocks",
        was,
    )


class _Draft:
    """The draft's launch and its forward."""

    def __init__(self, lm, runner):
        self.lm, self.runner = lm, runner
        self.seen: set = set()
        self.backend_checked: set = set()

    def takes(self, forward_batch, input_embeds=None) -> bool:
        """Whether the launch runs this step: static properties of the step only
        (mode, batch size, rows, the hidden-state buffer), so graph-capture safe.

        A draft extend or draft decode of 1 to ``max_reqs`` requests the launch
        cannot read raises, eager or in capture: SGLang's forward must not run a
        draft step the launch serves (see the module docstring)."""
        from sgl_amd_mono.qwen38_flash_next.gfx950.runner import (
            HC_DIM,
            draft_eligible,
        )

        r, fb = self.runner, forward_batch
        hidden = None if fb.spec_info is None else fb.spec_info.hidden_states
        if input_embeds is None and draft_eligible(fb, hidden, r.L_ext, r.max_reqs):
            return True
        mode, R = fb.forward_mode, int(fb.batch_size)
        is_extend = mode.is_draft_extend_v2()
        if not (is_extend or mode.is_decode()) or not 1 <= R <= r.max_reqs:
            return False
        if input_embeds is not None:
            what = "input embeddings"
        elif hidden is None:
            what = "no hidden states"
        else:
            strided = "" if hidden.is_contiguous() else " (strided)"
            what = f"hidden states {hidden.dtype} {tuple(hidden.shape)}{strided}"
        key = (mode.name, R, int(fb.input_ids.numel()))
        rows = R * r.L_ext if is_extend else R
        raise ValueError(
            f"Qwen3.8-Flash-Next MTP draft step {key} has {what}: the launch takes "
            f"every draft step of 1 to {r.max_reqs} requests, {rows} rows of "
            f"contiguous bf16 [rows, {HC_DIM}] hidden states and no input "
            "embeddings; refusing to run SGLang's draft forward"
        )

    def check_backend(self, forward_batch) -> None:
        """Refuse a draft step on an attention backend whose caches the launch does
        not write, or that shares one QSA selection across the draft steps, once
        per step mode (what the runtime does for routed steps)."""
        mode = forward_batch.forward_mode.name
        if mode in self.backend_checked:
            return
        backend = _step_backend()
        if backend not in DRAFT_ATTENTION_BACKENDS:
            raise ValueError(
                f"--monokernel-backend amd: {SPEC.name} needs a draft attention "
                f"backend in {sorted(DRAFT_ATTENTION_BACKENDS)}, this {mode} step's "
                f"is {backend!r}"
            )
        if _shares_mtp_sparse_indices():
            raise ValueError(
                f"--monokernel-backend amd: {SPEC.name}: this {mode} step's QSA "
                "backend shares one sparse selection across the MTP draft steps "
                "(index_share_for_mtp_iteration on); SGLang's draft decode would "
                "read a selection the launched draft extend never writes"
            )
        self.backend_checked.add(mode)

    def step(self, input_ids, forward_batch):
        """``Qwen4ExpForCausalLMMTP.forward`` for a step the launch takes."""
        lm, fb = self.lm, forward_batch
        key = (fb.forward_mode.name, int(fb.batch_size), int(input_ids.numel()))
        if key not in self.seen:
            self.seen.add(key)
            logger.info("monokernel qwen38_flash_next MTP draft runs %s", key)
        x, hc = self.runner.launch(fb, fb.spec_info.hidden_states)
        out = lm.logits_processor(input_ids, x, lm.lm_head, fb)
        lm._set_hc_logits_hidden_states(out, hc, fb)
        return out

    def forward(
        self, plan, input_ids, positions, forward_batch, input_embeds=None, *_, **kw
    ):
        """A step the runtime routes: the launch, or SGLang's forward. The draft's
        own steps reach the launch through ``_route_draft_steps``."""
        if self.takes(forward_batch, input_embeds):
            return self.step(input_ids, forward_batch)
        lm = self.lm
        return type(lm).forward(
            lm, input_ids, positions, forward_batch, input_embeds, **kw
        )


def _route_draft_steps(model, draft: _Draft) -> None:
    """The draft-step wrapper in front of the model's forward (see the module
    docstring); every other step runs the forward it wraps. Runs in build, so the
    runtime's forward wraps this one."""
    standard = model.forward

    # Graph runners inspect the original parameters to select forward arguments.
    @functools.wraps(standard)
    def forward(*args, **kwargs):
        # (input_ids, positions, forward_batch, input_embeds=None, ...)
        fb = kwargs["forward_batch"] if "forward_batch" in kwargs else args[2]
        embeds = kwargs.get("input_embeds", args[3] if len(args) > 3 else None)
        if draft.takes(fb, embeds):
            draft.check_backend(fb)
            ids = kwargs["input_ids"] if "input_ids" in kwargs else args[0]
            return draft.step(ids, fb)
        return standard(*args, **kwargs)

    model.forward = forward


def _build(model, hf_config, ctx) -> _Draft:
    import torch

    from sgl_amd_mono.qwen38_flash_next.gfx950 import weights
    from sgl_amd_mono.qwen38_flash_next.gfx950.runner import DraftRunner

    _turn_off_mtp_index_share(hf_config)
    params = weights.param_ptrs(model)
    views = weights.mtp_views(model, ctx.copies, params)
    layer = model.model.layers[0]
    rope = layer.rotary_emb
    runner = DraftRunner(
        views=views,
        embed_w=model.model.embed_tokens.weight,
        cos_sin=lambda: rope.cos_sin_cache_as(torch.bfloat16),
        layer_id=layer.attn.layer_id,
        ntok_extend=target.TOKENS,
        max_reqs=target.MAX_R,
        eos_token_id=int(target._text(hf_config).eos_token_id),
        device=model.model.embed_tokens.weight.device,
        # SGLANG_AMD_MONO_FAULT_INJECT stalls the target's launches only, so the
        # fault-injection test knows which launch stopped
        fault=None,
    )
    runner.set_deadline(ctx.deadline_s)
    draft = _Draft(model, runner)
    _route_draft_steps(model, draft)
    logger.info(
        "monokernel qwen38_flash_next MTP draft: draft extend steps of up to %d x %d "
        "rows and draft decode steps of up to %d x 1 are routed by the spec's "
        "wrapper of the draft's forward, and one it cannot launch raises; each key "
        "logs 'MTP draft runs' at its first launch",
        runner.max_reqs,
        runner.L_ext,
        runner.max_reqs,
    )
    return draft


SPEC = MonoSpec(
    name="qwen38_flash_next_mtp_draft_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, target.MAX_R * target.TOKENS + 1)),
            tp_sizes=(target.TP,),
            resident_blocks=BLOCKS,
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    # the launcher checks the target's requirements; a draft step is checked
    # against these by the draft-step wrapper
    requires=Requirements(attention_backends=DRAFT_ATTENTION_BACKENDS),
    numerics=target.NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    tokens_per_request=_tokens_per_request,
)
