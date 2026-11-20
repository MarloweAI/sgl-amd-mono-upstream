# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's sampled DSpark draft tail on vocab shards, offered through SGLang's draft sampler
entry: each rank projects its lm_head shard (no base-logits all-gather), and one launch
(``gfx950/markov``) walks every Markov step on the rank's ``markov_w2`` shard, the ranks' bests
swapped through peer buffers. The tokens, the bf16 step logits and the noise counter are those of
SGLang's ``draft_markov_block`` on the gathered vocab. Where SGLang's step-logits buffer holds whole
rows, they are gathered from every rank's shard after the walk."""

from __future__ import annotations

import functools
import logging
import sys

import torch

_LOGGER = logging.getLogger(__name__)

DRAFT = "sglang.srt.models.dspark"

# by file under sglang/ (the attributes this module reads)
SGLANG_READS = {
    "srt/kernel_plugins/nvidia/binding.py": ("built_megakernel", '"_megakernel"'),
    "kernels/plugin_kernels/nvidia/spec.py": ("draft_sampler",),
    "srt/models/dspark.py": (
        "VanillaMarkov",
        "markov_head",
        "markov_w1",
        "markov_w2",
        "step_logits_dtype",
        "vocab_shard",
        "vocab_size",
        "lm_head",
        "project_through_lm_head",
        "logits_mup_width_multiplier",
    ),
    "srt/layers/vocab_parallel_embedding.py": (
        "num_embeddings_per_partition",
        "shard_indices",
        "org_vocab_start_index",
        "org_vocab_end_index",
        "org_vocab_size",
        "tp_size",
    ),
    "srt/speculative/dspark_components/dspark_draft.py": ("key", "offset"),
}


class _Entries:
    """What SGLang reads of a built draft through ``built_megakernel``: the draft sampler only."""

    widths: tuple = ()
    draft_loop = draft_extend = begin_draft_round = graph_variants = None
    prepare = tensor_entry = binding = None

    def __init__(self, draft_sampler):
        self.draft_sampler = draft_sampler

    @property
    def built(self):
        return self

    def summary(self) -> str:
        return "Kimi-K3's sampled DSpark draft tail on vocab shards"


def install(model, group, deadline_s: float = 20.0) -> None:
    """Offer the sharded tail to the DSpark draft ``model``'s sampler; ``group`` is the lm_head's
    TP GroupCoordinator (its ``cpu_group``, ``rank_in_group`` and ``world_size``)."""
    if "_megakernel" in model.__dict__:
        return
    model.__dict__["_megakernel"] = _Entries(functools.partial(_factory, model, group, deadline_s))


def why_not(model, group, step_logits) -> str | None:
    """Why the sharded tail does not apply to this draft, or None."""
    from sgl_amd_mono.kimi_k3.gfx950 import markov

    head = getattr(model, "markov_head", None)
    if type(head).__name__ != "VanillaMarkov":
        return f"a {type(head).__name__}"
    lm = model.lm_head
    tp = int(getattr(lm, "tp_size", 1))
    if not 1 < tp <= markov.MAX_TP or group.world_size != tp:
        return f"lm_head TP {tp}, group {group.world_size}"
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return f"markov weights {w1.dtype} / {w2.dtype}, not bf16"
    if w1.shape[1] != markov.DIM or w2.shape[1] != markov.DIM:
        return f"markov rank {w1.shape[1]}, the kernels' is {markov.DIM}"
    if head.step_logits_dtype(torch.bfloat16) != torch.bfloat16:
        return "step logits not bf16"
    w = getattr(lm, "weight", None)
    if w is None or w.dtype != torch.bfloat16 or getattr(lm, "quant_method", None) is not None and type(
        lm.quant_method
    ).__name__ != "UnquantizedEmbeddingMethod":
        return "lm_head not an unquantized bf16 weight"
    part = int(lm.num_embeddings_per_partition)
    start, end = int(lm.shard_indices.org_vocab_start_index), int(lm.shard_indices.org_vocab_end_index)
    vocab = int(lm.org_vocab_size)
    if vocab != head.vocab_size or start != group.rank_in_group * part or part % 8 or vocab % 8:
        return f"vocab {vocab} / {head.vocab_size} in shards of {part} at {start}"
    if end - start > part or markov.blocks(part) > markov.MAX_BLOCKS:
        return f"shard [{start}, {end}) of {part}"
    vs = head.vocab_shard
    if vs is not None and (vs.start != start or vs.width != end - start):
        return f"SGLang's markov_w2 shard [{vs.start}, +{vs.width}) is not the lm_head's"
    if step_logits is None or step_logits.dtype != torch.bfloat16:
        return "no bf16 step logits"
    if step_logits.shape[1] > markov.MAX_STEPS:
        return f"{step_logits.shape[1]} draft steps, the kernel's at most {markov.MAX_STEPS}"
    if step_logits.shape[-1] not in (vocab, part):
        return f"step logits {step_logits.shape[-1]} wide, not the vocab's {vocab} or a shard's {part}"
    return None


class Sampler:
    """The sharded tail of one draft: ``step(hidden, anchor)`` is SGLang's base logits and Markov
    block of a bs-row draft block. ``exchange`` builds the TP group's ``markov.Exchange``."""

    def __init__(self, model, group, deadline_s, gamma, noise, draft_tokens, step_logits, temps, ops, exchange):
        from sgl_amd_mono.kimi_k3.gfx950 import markov

        head, lm = model.markov_head, model.lm_head
        self.model, self.gamma, self.noise = model, gamma, noise
        self.out, self.step_logits, self.temps = draft_tokens, step_logits, temps
        self.deadline_s = float(deadline_s)
        self.rank = group.rank_in_group
        self.vocab_start = int(lm.shard_indices.org_vocab_start_index)
        end = int(lm.shard_indices.org_vocab_end_index)
        self.width = int(lm.num_embeddings_per_partition)
        self.embed = head.markov_w1.weight.contiguous()
        self.head = head.markov_w2.weight[self.vocab_start : end]
        assert self.head.is_contiguous()
        self.whole_rows = step_logits.shape[-1] != self.width
        assert gamma <= markov.MAX_STEPS
        self.exchange = exchange(group, self.head.device, gamma, self.width)
        self.ops = ops

    def base_logits(self, hidden):
        """This rank's lm_head shard of the base logits (SGLang's ``compute_base_logits`` before the
        gather)."""
        mup = self.model.logits_mup_width_multiplier
        if mup:
            hidden = hidden / mup
        return sys.modules[DRAFT].project_through_lm_head(hidden, self.model.lm_head)

    def walk(self, base, anchor, temps, out, step_logits):
        x = self.exchange
        self.ops.markov_block(
            base, self.embed, self.head, anchor, temps, self.noise.offset, int(self.noise.key),
            self.vocab_start, out, self.gamma, x.tok, x.control, x.peers, self.rank, step_logits,
            self.width, self.deadline_s, None,
        )  # fmt: skip

    def gather(self, step_logits):
        x = self.exchange
        self.ops.gather_step_logits(x.peers, step_logits, self.width)

    def step(self, hidden_states, anchor_tokens):
        bs = anchor_tokens.shape[0]
        base = self.base_logits(hidden_states).view(bs, self.gamma, -1)
        out = self.out[: bs * self.gamma]
        logits = self.step_logits[:bs]
        self.walk(base, anchor_tokens.long(), self.temps[:bs], out, logits)
        if self.whole_rows:  # the walk wrote the slab; a shard-wide buffer it wrote itself
            self.gather(logits)
        return out.view(bs, self.gamma), hidden_states


def peer_exchange(group, device, gamma, part):
    from sgl_amd_mono.kimi_k3.gfx950 import markov

    return markov.Exchange(group.cpu_group, group.rank_in_group, group.world_size, device, gamma, part)


def _factory(model, group, deadline_s, *, gamma, max_bs, greedy_draft, noise, draft_tokens, step_logits,
             step_temperatures, **_):  # fmt: skip
    """SGLang's draft sampler factory: bs -> the sharded tail's step, or None (SGLang's)."""
    from sgl_amd_mono.kimi_k3.gfx950 import markov

    why = "a greedy draft" if greedy_draft else why_not(model, group, step_logits)
    if why is not None:
        _LOGGER.info("DSpark draft: SGLang's Markov block runs (%s)", why)
        return lambda bs: None
    sampler = Sampler(model, group, deadline_s, gamma, noise, draft_tokens, step_logits, step_temperatures,
                      markov.ops(), peer_exchange)  # fmt: skip
    top = min(max_bs, markov.MAX_SEQUENCES)
    if group.rank_in_group == 0:
        _LOGGER.info("DSpark draft: the sharded sampled tail at batch sizes 1 to %d", top)

    def steps(bs):
        return sampler.step if 1 <= bs <= top else None

    steps.sampler = sampler
    return steps
