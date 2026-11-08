# SPDX-License-Identifier: Apache-2.0
"""The static verify graph's accept where its logits are the rank's bf16 vocab shard
(SGLang's "greedy" verify variant with ``defer_greedy_verify_gather``): the ranks' row keys,
their exchange and the accept tail in one launch (``gfx950/vtail``), in place of SGLang's
per-rank keys, keys all-gather, selection and accept tail. Bitwise SGLang's.
``SGLANG_AMD_MONO_DSV41_VERIFY_TAIL=0`` keeps SGLang's."""

from __future__ import annotations

import torch

from sgl_amd_mono.deepseek_v41.glue import chose, flag, route

ENV = "SGLANG_AMD_MONO_DSV41_VERIFY_TAIL"
VERIFY = "sglang.srt.speculative.dspark_components.dspark_verify"
DEADLINE_S = 60.0

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "srt/speculative/dspark_components/dspark_verify.py": (
        "DsparkVerifyEpilogue",
        "_static_accept_tail",
        "verify_shard",
        "simulate_fold",
        "_tail_broadcasts",
        "simulated_correct_len",
        "sim_temperatures_buf",
        "sim_seed_buf",
        "inject_gate_buf",
        "correct_len_buf",
        "bonus_buf",
        "cap_trim_lens_buf",
        "commit_lens_buf",
        "new_seq_lens_buf",
        "out_tokens_buf",
        "gated_commit_lens_buf",
        "_log_accept_tail(shard)",
        "stride",
    ),
    "srt/layers/logits_processor.py": (
        "deferred_shard_offset",
        "vocab_size",
    ),
    "srt/runtime_context.py": ("get_parallel",),
}

_TAILS: dict = {}  # per epilogue: the launch and a zero temperature row


def _why_not(epi, shard, logits, candidates) -> str | None:
    from sgl_amd_mono.deepseek_v41.gfx950 import vtail

    if shard is None:
        return "whole logits"
    if epi._tail_broadcasts:
        return "the accept-site TP broadcast"
    bs, t = candidates.shape
    if logits.dtype != torch.bfloat16 or logits.stride(1) != 1 or logits.dim() != 2:
        return f"{logits.dtype} logits"
    if logits.shape[0] != bs * t or t > vtail.MAX_T or bs * t > vtail.MAX_ROWS:
        return f"{bs} x {t} verify rows"
    if logits.shape[1] % 8 or logits.stride(0) % 8 or logits.data_ptr() % 16:
        return "unaligned shard rows"
    if vtail.blocks(logits.shape[1]) > vtail.MAX_BLOCKS or shard.vocab_size >= 1 << 17:
        return f"a {logits.shape[1]}-column shard"
    return None


def _tail(epi, shard, logits):
    got = _TAILS.get(id(epi))
    if got is None:
        from sglang.srt.runtime_context import get_parallel

        from sgl_amd_mono.deepseek_v41.gfx950 import vtail

        assert not torch.cuda.is_current_stream_capturing(), (
            "the verify tail's peer buffers are built at a warmup, before capture"
        )
        group = get_parallel().tp_group
        assert group.world_size <= vtail.MAX_TP
        vocab_start, width = shard.deferred_shard_offset(logits)
        launch = vtail.VerifyTail(
            group.cpu_group, group.rank_in_group, group.world_size, logits.device,
            part=logits.shape[1], width=width, vocab_start=vocab_start,
            vocab=shard.vocab_size, deadline_s=DEADLINE_S,
        )  # fmt: skip
        zeros = torch.zeros_like(epi.sim_temperatures_buf)
        got = _TAILS[id(epi)] = (launch, zeros)
    return got


def _static_accept_tail(
    orig, self, *, candidates, logits, seq_lens, verify_lens, gated
):
    shard = self.verify_shard(logits)
    why = _why_not(self, shard, logits, candidates)
    if not chose("verify_tail", why is None, why):
        return orig(
            self, candidates=candidates, logits=logits, seq_lens=seq_lens,
            verify_lens=verify_lens, gated=gated,
        )  # fmt: skip
    launch, zeros = _tail(self, shard, logits)
    bs = candidates.shape[0]
    sim = self.simulated_correct_len
    # SGLang's tail draws a simulated length's bonus at the request's temperature under
    # simulate_fold, else takes that row's argmax
    temps = (self.sim_temperatures_buf if self.simulate_fold else zeros)[:bs]
    self._log_accept_tail(bs, bonus_in=None, shard=shard)
    launch(
        logits, candidates, seq_lens, verify_lens, self.inject_gate_buf,
        sim=sim, temps=temps if sim is not None else None,
        seed=self.sim_seed_buf if sim is not None else None,
        correct_out=self.correct_len_buf, bonus_out=self.bonus_buf,
        cap_trim_out=self.cap_trim_lens_buf, commit_out=self.commit_lens_buf,
        new_seq_out=self.new_seq_lens_buf, out_tokens=self.out_tokens_buf,
        gated_out=self.gated_commit_lens_buf if gated else None,
    )  # fmt: skip
    return self.gated_commit_lens_buf[:bs] if gated else None


def install() -> None:
    if not flag(ENV) or torch.version.hip is None:
        return
    import importlib

    verify = importlib.import_module(VERIFY)
    if not hasattr(verify.DsparkVerifyEpilogue, "verify_shard"):
        return
    route(verify.DsparkVerifyEpilogue, "_static_accept_tail", _static_accept_tail)
