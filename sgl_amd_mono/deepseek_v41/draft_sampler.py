# SPDX-License-Identifier: Apache-2.0
"""A sampled DSpark draft's Markov block, its step logits included, in one launch
(``gfx950/markov``), offered through SGLang's draft sampler entry where the
draft's Markov head and step logits have the layout the kernels read. Each rank keeps its
shard of a sampled sequence's step logits (``step_logits_shard``, what the verify's sampled
accept reads); SGLang's whole rows are gathered from every rank's shard only for a reader
that needs them (``gather_step_logits``). A temperature-0 sequence's stay as SGLang's
buffer holds them.
``SGLANG_AMD_MONO_DSV41_MARKOV=0`` keeps SGLang's steps."""

from __future__ import annotations

import functools
import logging
import sys

import torch

_LOGGER = logging.getLogger(__name__)

ENV = "SGLANG_AMD_MONO_DSV41_MARKOV"
DRAFT = "sglang.srt.models.deepseek_v4_dspark"

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": ("draft_sampler", "graph_variants"),
    "srt/kernel_plugins/nvidia/binding.py": ("built_megakernel", '"_megakernel"'),
    "srt/models/deepseek_v4_dspark.py": (
        "DSparkV4MarkovHead",
        "compute_base_logits",
        "markov_head",
        "markov_w1",
        "markov_w2",
        "_tp_shard",
        "_shard_group",
        "org_vocab_start",
        "org_vocab_end",
        "_opt_markov_w2_bf16",
        "step_logits_dtype",
        "vocab_size",
        "lm_head",
        "num_embeddings_per_partition",
        "project_through_lm_head",
        "hc_pre_from_prev_sublayer",
        "_opt_markov_w2_tp_shard",
        "_use_fp32_lm_head",
        "_logits_from_x_post_hc",
    ),
    "srt/models/dspark.py": ("vocab_shard", "vocab_size"),
    "srt/speculative/dspark_components/dspark_draft.py": ("key", "offset"),
    "srt/speculative/dspark_components/dspark_draft_sampler.py": (
        '"verify_ids"',
        '"writes_verify_ids"',
        '"gather_step_logits"',
        "whole_step_logits",
    ),
}


# per SGLang step-logits buffer (its address): the sampler whose launches keep their shards
_SAMPLERS: dict = {}


def step_logits_shard(step_logits: torch.Tensor, bs: int):
    """This rank's columns of the last block's sampled step logits [bs, gamma, width] f32, where the
    kernels' Markov block ran it (SGLang's whole ``step_logits`` buffer then holds no rows until
    gathered), else None (SGLang's block wrote the whole rows)."""
    sampler = _SAMPLERS.get(step_logits.data_ptr())
    if sampler is None or bs not in sampler.ran:
        return None
    return sampler.ran[bs].slab[:bs, :, : sampler.width]


# the draft forward's last (combined rows, their norm by the last stage's RMSNorm) from one launch
# (``fused.combine_and_norm``): the sampler's base logits skip the norm for those rows. The reference
# keeps the rows' memory from being reused by another tensor while it is held.
_HEAD_INPUT: list = []


def keep_head_input(combined: torch.Tensor, normed: torch.Tensor) -> None:
    _HEAD_INPUT[:] = [combined, normed]


def head_input_folds(model) -> bool:
    """Whether the draft's base logits are the last stage's norm then the head on the forward's
    output (V4.1's collapsed hidden, a TP-sharded bf16 head), so the norm can come with the combine."""
    from sgl_amd_mono.deepseek_v41.glue import flag

    return (
        flag(ENV)
        and getattr(model, "hc_pre_from_prev_sublayer", False)
        and getattr(model, "_opt_markov_w2_tp_shard", False)
        and not getattr(model, "_use_fp32_lm_head", True)
    )


class _Entries:
    """What SGLang reads of a built draft through ``built_megakernel``: the draft
    sampler only (no covered step, no graph variants, no EAGLE entries)."""

    widths: tuple = ()
    draft_loop = draft_extend = begin_draft_round = graph_variants = None
    # current SGLang reads .built and .prepare of the entry, older SGLang the entry itself
    prepare = None

    def __init__(self, draft_sampler):
        self.draft_sampler = draft_sampler

    @property
    def built(self):
        return self

    def summary(self) -> str:
        return "the sampled DSpark draft's Markov block"


def install(model, deadline_s: float) -> None:
    """Offer the kernels' Markov block to the DSpark draft ``model``'s sampler."""
    from sgl_amd_mono.deepseek_v41.glue import flag

    if not flag(ENV) or "_megakernel" in model.__dict__:
        return
    model.__dict__["_megakernel"] = _Entries(
        functools.partial(_factory, model, deadline_s)
    )


def _why_not(model, step_logits) -> str | None:
    from sgl_amd_mono.deepseek_v41.gfx950 import markov

    head = getattr(model, "markov_head", None)
    if type(head).__name__ != "DSparkV4MarkovHead":
        return f"a {type(head).__name__}"
    shard, group = getattr(head, "_tp_shard", None), getattr(head, "_shard_group", None)
    if shard is None or group is None or head.vocab_shard is None:
        return "markov_w2 not TP-sharded"
    if not 1 < group.world_size <= 8:
        return f"TP {group.world_size}"
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return f"markov weights {w1.dtype} / {w2.dtype}, not bf16"
    if w1.shape[1] != markov.DIM or w2.shape[1] != markov.DIM:
        return f"markov rank {w1.shape[1]}, the kernels' is {markov.DIM}"
    if head.step_logits_dtype(torch.bfloat16) != torch.float32:
        return "step logits not fp32"
    part = int(model.lm_head.num_embeddings_per_partition)
    if step_logits is None or step_logits.dtype != torch.float32:
        return "no fp32 step logits"
    if step_logits.shape[1] > markov.MAX_STEPS:
        return f"{step_logits.shape[1]} draft steps, the kernel's at most {markov.MAX_STEPS}"
    if step_logits.shape[-1] != head.vocab_size:
        return f"step logits {step_logits.shape[-1]} wide, not the vocab's {head.vocab_size}"
    if shard.org_vocab_start != group.rank_in_group * part or part % 4:
        return f"vocab shards of {part} not at the ranks' multiples of the partition"
    return None


class _Sampler:
    def __init__(
        self, model, deadline_s, gamma, noise, draft_tokens, step_logits, temps,
        verify_ids=None,
    ):  # fmt: skip
        from sgl_amd_mono.deepseek_v41.gfx950 import markov

        head = model.markov_head
        shard, group = head._tp_shard, head._shard_group
        self.model, self.gamma, self.noise = model, gamma, noise
        self.out, self.step_logits, self.temps = draft_tokens, step_logits, temps
        # [max_bs, gamma + 1] the verify's ids ([anchor | tokens]), written by the launch, or None
        self.verify_ids = verify_ids
        self.deadline_s = float(deadline_s)
        self.vocab_start = int(shard.org_vocab_start)
        self.embed = head.markov_w1.weight.contiguous()
        self.head = head.markov_w2.weight[shard.org_vocab_start : shard.org_vocab_end]
        assert self.head.is_contiguous()
        device = self.head.device
        self.width = int(model.lm_head.num_embeddings_per_partition)
        self.exchange = markov.Exchange(
            group.cpu_group,
            group.rank_in_group,
            group.world_size,
            device,
            gamma,
            self.width,
        )
        self.ops = markov.ops()
        # per batch size: the exchange whose shard holds that block's step logits
        self.ran: dict = {}
        # one sequence's head and Markov block in one launch: its own exchange, launch counts and base logits
        self.fused = None
        if _head_folds(model, gamma, group.world_size):
            self.fused = markov.Exchange(
                group.cpu_group, group.rank_in_group, group.world_size, device, gamma, self.width,
                control=markov.MAX_BLOCKS,
            )  # fmt: skip
            self.base = torch.zeros(
                gamma * self.width, dtype=torch.int64, device=device
            )

    def block(
        self, base_logits, anchor, temps, noise, out, step_logits, verify_ids=None
    ):
        """Every step of a [bs, gamma, padded] base logits block: tokens to ``out``
        ([bs * gamma]), the sampled sequences' step logits to ``step_logits``
        ([bs, gamma, vocab])."""
        x = self.exchange
        self.ops.markov_block(
            base_logits, self.embed, self.head, anchor, temps, noise.offset,
            int(noise.key), self.vocab_start, out, self.gamma, x.tok, x.control, x.peers,
            x.rank, step_logits, self.width, self.deadline_s, verify_ids,
        )  # fmt: skip

    def step(self, hidden_states, anchor_tokens):
        """SGLang's base logits and Markov block of a bs-row draft block."""
        from sgl_amd_mono.deepseek_v41.glue import chose

        held = _HEAD_INPUT
        normed = (
            held
            and held[0].data_ptr() == hidden_states.data_ptr()
            and held[0].shape == hidden_states.shape
            and head_input_folds(self.model)
        )
        bs = anchor_tokens.shape[0]
        if chose(
            "draft_head",
            self.fused is not None and normed and bs == 1 and held[1].is_contiguous(),
        ):
            # the head's projection of the normed rows and the Markov block in one launch
            x = self.fused
            self._ran(bs, x)
            out, verify = (
                self.out[: self.gamma],
                None if self.verify_ids is None else self.verify_ids[:1],
            )
            self.ops.draft_head(
                held[1], self.model.lm_head.weight, self.base, self.embed, self.head, anchor_tokens.long(),
                self.temps[:1], self.noise.offset, int(self.noise.key), self.vocab_start, out, self.gamma,
                x.tok, x.control, x.peers, x.rank, self.step_logits[:1], self.width, self.deadline_s, verify,
            )  # fmt: skip
            return out.view(1, self.gamma), hidden_states
        if normed:
            # _logits_from_x_post_hc on the rows normed in the combine's launch
            project = sys.modules[DRAFT].project_through_lm_head
            base_logits, confidence_features = (
                project(held[1], self.model.lm_head),
                hidden_states,
            )
        else:
            base_logits, confidence_features = self.model.compute_base_logits(
                hidden_states
            )
        base = base_logits.view(bs, self.gamma, -1)
        self._ran(bs, self.exchange)
        out = self.out[: bs * self.gamma]
        verify = None if self.verify_ids is None else self.verify_ids[:bs]
        self.block(
            base, anchor_tokens.long(), self.temps[:bs], self.noise, out,
            self.step_logits[:bs], verify,
        )  # fmt: skip
        return out.view(bs, self.gamma), confidence_features

    def _ran(self, bs, exchange):
        held = self.ran.setdefault(bs, exchange)
        if held is not exchange:
            raise RuntimeError(
                f"DSV4.1 Markov block: batch size {bs} ran on two exchanges"
            )

    def gather_step_logits(self, bs):
        """SGLang's whole step-logits rows [:bs] of the last block, from every rank's shard."""
        x = self.ran[bs]
        self.ops.gather_step_logits(
            self.temps[:bs], x.peers, self.step_logits[:bs], self.width
        )


def _head_folds(model, gamma: int, tp: int) -> bool:
    """Whether one sequence's base logits run in the Markov block's launch: TP8 (at TP4 the
    launch measured level with hipBLASLt + the Markov block), the head input folds
    (``head_input_folds``), a bf16 head shard [width, 5120] of the kernel's shapes, at most 6 steps."""
    w = getattr(model.lm_head, "weight", None)
    return (
        tp == 8
        and head_input_folds(model)
        and w is not None
        and w.dtype == torch.bfloat16
        and w.dim() == 2
        and w.shape[1] == 5120
        and w.stride(1) == 1
        and w.shape[0] == int(model.lm_head.num_embeddings_per_partition)
        and gamma <= 6
    )


def _factory(
    model, deadline_s, *, gamma, max_bs, greedy_draft, noise, draft_tokens,
    step_logits, step_temperatures, verify_ids=None,
):  # fmt: skip
    """SGLang's draft sampler factory: bs -> the kernels' step, or None (SGLang's).
    ``verify_ids`` ([max_bs, gamma + 1] int64): the step also writes the verify's ids
    ([anchor | tokens]) there, and the steps say so (``writes_verify_ids``)."""
    from sgl_amd_mono.deepseek_v41.gfx950 import markov

    why = "a greedy draft" if greedy_draft else _why_not(model, step_logits)
    if why is not None:
        _LOGGER.info("DSpark draft: SGLang's Markov block runs (%s)", why)
        return lambda bs: None
    sampler = _Sampler(
        model, deadline_s, gamma, noise, draft_tokens, step_logits, step_temperatures,
        verify_ids,
    )  # fmt: skip
    top = min(max_bs, markov.MAX_SEQUENCES)
    if model.markov_head._shard_group.rank_in_group == 0:
        _LOGGER.info(
            "DSpark draft: the sampled Markov block in the gfx950 kernels at batch "
            "sizes 1 to %d",
            top,
        )

    def steps(bs):
        return sampler.step if 1 <= bs <= top else None

    steps.writes_verify_ids = verify_ids is not None
    steps.gather_step_logits = sampler.gather_step_logits
    _SAMPLERS[step_logits.data_ptr()] = sampler
    return steps
