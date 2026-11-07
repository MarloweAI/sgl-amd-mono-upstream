# SPDX-License-Identifier: Apache-2.0
"""A sampled DSpark draft's Markov block in two launches a step and one more for its step
logits (``gfx950/markov``), offered through SGLang's draft sampler entry where the
draft's Markov head and step logits have the layout the kernels read. The step logits are
SGLang's whole rows: a sampled sequence's are gathered from every rank's shard, a
temperature-0 sequence's stay as SGLang's buffer holds them.
``SGLANG_AMD_MONO_DSV41_MARKOV=0`` keeps SGLang's steps."""

from __future__ import annotations

import functools
import logging

import torch

_LOGGER = logging.getLogger(__name__)

ENV = "SGLANG_AMD_MONO_DSV41_MARKOV"

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
    ),
    "srt/models/dspark.py": ("vocab_shard", "vocab_size"),
    "srt/speculative/dspark_components/dspark_draft.py": ("key", "offset"),
}


class _Entries:
    """What SGLang reads of a built draft through ``built_megakernel``: the draft
    sampler only (no covered step, no graph variants, no EAGLE entries)."""

    widths: tuple = ()
    draft_loop = draft_extend = begin_draft_round = graph_variants = None

    def __init__(self, draft_sampler):
        self.draft_sampler = draft_sampler

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
    if step_logits.shape[-1] != head.vocab_size:
        return f"step logits {step_logits.shape[-1]} wide, not the vocab's {head.vocab_size}"
    if shard.org_vocab_start != group.rank_in_group * part or part % 4:
        return f"vocab shards of {part} not at the ranks' multiples of the partition"
    return None


class _Sampler:
    def __init__(
        self, model, deadline_s, gamma, noise, draft_tokens, step_logits, temps
    ):
        from sgl_amd_mono.deepseek_v41.gfx950 import markov

        head = model.markov_head
        shard, group = head._tp_shard, head._shard_group
        self.model, self.gamma, self.noise = model, gamma, noise
        self.out, self.step_logits, self.temps = draft_tokens, step_logits, temps
        self.deadline_s = float(deadline_s)
        self.vocab_start = int(shard.org_vocab_start)
        self.embed = head.markov_w1.weight.contiguous()
        self.head = head.markov_w2.weight[shard.org_vocab_start : shard.org_vocab_end]
        assert self.head.is_contiguous()
        device = self.head.device
        self.width = int(model.lm_head.num_embeddings_per_partition)
        self.part = torch.empty(
            markov.MAX_SEQUENCES,
            markov.blocks(self.width),
            2,
            dtype=torch.float32,
            device=device,
        )
        self.exchange = markov.Exchange(
            group.cpu_group,
            group.rank_in_group,
            group.world_size,
            device,
            gamma,
            self.width,
        )
        self.ops = markov.ops()

    def block(self, base_logits, anchor, temps, noise, out, step_logits):
        """Every step of a [bs, gamma, padded] base logits block: tokens to ``out``
        ([bs * gamma]), the sampled sequences' step logits to ``step_logits``
        ([bs, gamma, vocab])."""
        from sgl_amd_mono.deepseek_v41.gfx950 import markov

        x, bs = self.exchange, base_logits.shape[0]
        for s in range(self.gamma):
            self.ops.markov_sample_step(
                base_logits[:, s], self.embed, self.head, anchor, x.slab[:bs, s], temps,
                noise.offset, int(noise.key), self.vocab_start, s, self.part, out,
                self.gamma, x.tok, x.calls, x.peers, x.rank, s == self.gamma - 1,
                self.deadline_s,
            )  # fmt: skip
        # the last step's swap saw every rank's packet: their step logits are written
        self.ops.markov_gather(
            step_logits, temps, x.peers, markov.EXCHANGE_BYTES, self.width
        )

    def step(self, hidden_states, anchor_tokens):
        """SGLang's base logits and Markov block of a bs-row draft block."""
        base_logits, confidence_features = self.model.compute_base_logits(hidden_states)
        bs = anchor_tokens.shape[0]
        base = base_logits.view(bs, self.gamma, -1)
        out = self.out[: bs * self.gamma]
        self.block(
            base, anchor_tokens.long(), self.temps[:bs], self.noise, out,
            self.step_logits[:bs],
        )  # fmt: skip
        return out.view(bs, self.gamma), confidence_features


def _factory(
    model, deadline_s, *, gamma, max_bs, greedy_draft, noise, draft_tokens,
    step_logits, step_temperatures,
):  # fmt: skip
    """SGLang's draft sampler factory: bs -> the kernels' step, or None (SGLang's)."""
    from sgl_amd_mono.deepseek_v41.gfx950 import markov

    why = "a greedy draft" if greedy_draft else _why_not(model, step_logits)
    if why is not None:
        _LOGGER.info("DSpark draft: SGLang's Markov block runs (%s)", why)
        return lambda bs: None
    sampler = _Sampler(
        model, deadline_s, gamma, noise, draft_tokens, step_logits, step_temperatures
    )
    top = min(max_bs, markov.MAX_SEQUENCES)
    if model.markov_head._shard_group.rank_in_group == 0:
        _LOGGER.info(
            "DSpark draft: the sampled Markov block in the gfx950 kernels at batch "
            "sizes 1 to %d",
            top,
        )
    return lambda bs: sampler.step if 1 <= bs <= top else None
