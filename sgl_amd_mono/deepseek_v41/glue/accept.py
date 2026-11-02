# SPDX-License-Identifier: Apache-2.0
"""The static verify graph's accept: SGLang's one-launch tail without the accept-site TP
broadcasts, and under simulated acceptance a temperature-only sampled batch accepted and
committed in the graph as a greedy one is. ``SGLANG_AMD_MONO_DSV41_ACCEPT=0`` keeps SGLang's."""

from __future__ import annotations

import os
import random
import sys

import torch

from sgl_amd_mono.deepseek_v41.glue import flag, route

ENV = "SGLANG_AMD_MONO_DSV41_ACCEPT"
VERIFY = "sglang.srt.speculative.dspark_components.dspark_verify"
DRAFT = "sglang.srt.speculative.dspark_components.dspark_draft"

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "srt/speculative/dspark_components/dspark_verify.py": (
        "DsparkVerifyEpilogue",
        "__init__(tp_sync)",
        "__init__(fused_accept_tail)",
        "begin_static_step(bs)",
        "begin_static_step(armed)",
        "begin_sim_bonus(bs)",
        "begin_sim_bonus(temperatures)",
        "sim_temperatures_buf",
        "sim_seed_buf",
        "_sim_greedy_bs",
        "_static_accept_tail(candidates)",
        "_static_accept_tail(logits)",
        "_static_accept_tail(seq_lens)",
        "_static_accept_tail(verify_lens)",
        "_static_accept_tail(gated)",
        "_sim_bonus_args",
        "_log_accept_tail",
        "_static_step_state",
        "simulate_fold",
        "simulated_correct_len",
        "folds_commit",
        "folds_engram_commit",
        "correct_len_buf",
        "bonus_buf",
        "cap_trim_lens_buf",
        "commit_lens_buf",
        "new_seq_lens_buf",
        "out_tokens_buf",
        "gated_commit_lens_buf",
        "inject_gate_buf",
        "read_accept",
        "note_static_verify",
        "static_accept_tail",
        "simulated_bonus_partials_graph",
        "TargetVerifyExecutor",
        "begin_simulated_step",
        "run_non_compact(batch)",
        "run_non_compact(sampling_info)",
        "accept_and_finalize(folded_accept)",
        "accept_and_finalize(bs)",
        "commit_hidden(bs)",
        "commit_hidden(run_compact)",
        "verify_epilogue",
        "verify_logits_adjustments_are_noop",
        "_simulate_acc_len",
        "_simulated_correct_len",
        "gamma",
        "target_worker",
        "SIMULATE_ACC_METHOD",
        "sample_simulated_acc_len",
    ),
    "srt/speculative/dspark_components/dspark_draft.py": (
        "DraftBlockProposer",
        "propose",
        "folded",
    ),
    "srt/speculative/spec_tp_sync.py": ("SpecTpSyncSite", "DSPARK_ACCEPT_GRAPH"),
    "srt/managers/tp_worker.py": ("random_seed",),
    "srt/environ.py": (
        "SGLANG_DSPARK_SIMULATE_FOLD",
        "SGLANG_DSPARK_FUSED_ACCEPT_TAIL",
    ),
}

# the simulated lengths and bonus seeds: one stream a process, drawn once a step on
# every rank alike, so the graph's accept needs no broadcast to agree
_RNG: dict = {}
_FOLDED = [False]  # the last proposal ran the draft graph's sampler
_ACCEPT = [None]  # (executor, bs) whose accept the verify graph made
_COMMIT = [None]  # (executor, bs) whose commit it made


def _seeded(seed: int):
    if not _RNG:
        _RNG["torch"] = torch.Generator(device="cpu").manual_seed(int(seed))
        _RNG["py"] = random.Random(int(seed) + 1)
    return _RNG["torch"], _RNG["py"]


class _WithoutAcceptSite:
    """A SpecTpSync without the accept-site broadcast."""

    def __init__(self, sync, site):
        self._sync, self._site = sync, site

    def enabled(self, site) -> bool:
        return site != self._site and self._sync.enabled(site)

    def sync(self, site, values):
        return values if site == self._site else self._sync.sync(site, values)

    def __getattr__(self, name):
        return getattr(self._sync, name)


def _epilogue_init(orig, self, *args, **kwargs):
    sync = kwargs.get("tp_sync")
    if not args and sync is not None:
        site = sys.modules[VERIFY].SpecTpSyncSite.DSPARK_ACCEPT_GRAPH
        kwargs["tp_sync"] = _WithoutAcceptSite(sync, site)
        if os.environ.get("SGLANG_DSPARK_FUSED_ACCEPT_TAIL") is None:
            kwargs["fused_accept_tail"] = True
    return orig(self, *args, **kwargs)


def _begin_simulated_step(orig, self):
    if self._simulated_correct_len is None:
        return orig(self)
    verify = sys.modules[VERIFY]
    gen, _ = _seeded(self.target_worker.random_seed)
    n = verify.sample_simulated_acc_len(
        self._simulate_acc_len,
        verify.SIMULATE_ACC_METHOD,
        self.gamma + 1,
        generator=gen,
    )
    self._simulated_correct_len.fill_(n - 1)


def _begin_sim_bonus(orig, self, bs, temperatures):
    if temperatures is None:
        return orig(self, bs, temperatures)
    _, py = _seeded(0)  # seeded by begin_simulated_step, which runs first
    self._sim_greedy_bs = None
    self.sim_temperatures_buf[:bs].copy_(temperatures.view(-1))
    self.sim_seed_buf.fill_(py.getrandbits(31))


def _static_accept_tail(
    orig, self, *, candidates, logits, seq_lens, verify_lens, gated
):
    """SGLang's tail, without the argmax it does not read under a simulated length."""
    if not (self.simulate_fold and self.simulated_correct_len is not None):
        return orig(
            self,
            candidates=candidates,
            logits=logits,
            seq_lens=seq_lens,
            verify_lens=verify_lens,
            gated=gated,
        )
    verify = sys.modules[VERIFY]
    bs = candidates.shape[0]
    parts = verify.simulated_bonus_partials_graph(**self._sim_bonus_args(logits, bs))
    self._log_accept_tail(bs)
    verify.static_accept_tail(
        candidates=candidates,
        target_predict=candidates.view(-1),  # unread
        seq_lens=seq_lens,
        verify_lens=verify_lens,
        gate=self.inject_gate_buf,
        correct_len_out=self.correct_len_buf,
        bonus_out=self.bonus_buf,
        cap_trim_lens_out=self.cap_trim_lens_buf,
        commit_lens_out=self.commit_lens_buf,
        new_seq_lens_out=self.new_seq_lens_buf,
        out_tokens_out=self.out_tokens_buf,
        simulated_correct_len=self.simulated_correct_len,
        bonus_parts=(*parts, logits.shape[1]),
        gated_commit_lens_out=self.gated_commit_lens_buf if gated else None,
    )
    return self.gated_commit_lens_buf[:bs] if gated else None


def _foldable(executor, batch, info, bs: int) -> bool:
    """A simulated, temperature-only sampled static step the graph can accept."""
    ep = executor.verify_epilogue
    return bool(
        ep is not None
        and ep.simulate_fold
        and executor._simulated_correct_len is not None
        and _FOLDED[0]
        and info is not None
        and not info.is_all_greedy
        and not (info.need_top_k_sampling or info.need_top_p_sampling)
        and not info.need_min_p_sampling
        and sys.modules[VERIFY].verify_logits_adjustments_are_noop(info)
        and not batch.has_grammar
        and not batch.return_logprob
        and ep.folds_commit
        and not ep.folds_engram_commit
        and ep._static_step_state in ((bs, False), (bs, True))
    )


def _begin_static_step(orig, self, bs, armed):
    # between two folded steps of one batch size the commit stays armed
    if not armed and self.__dict__.get("_mono_armed") == bs:
        if self._static_step_state == (bs, True):
            self.__dict__["_mono_pending"] = bs
            return None
    self.__dict__["_mono_pending"] = self.__dict__["_mono_armed"] = None
    return orig(self, bs, armed)


def _run_non_compact(orig, self, *args, **kwargs):
    _ACCEPT[0] = _COMMIT[0] = None
    ep, ids = self.verify_epilogue, kwargs.get("verify_ids_2d")
    info = kwargs.get("sampling_info")
    pending = None if ep is None else ep.__dict__.get("_mono_pending")
    bs = None if ids is None else ids.shape[0]
    if args or bs is None or not _foldable(self, kwargs.get("batch"), info, bs):
        if pending is not None:  # the gating off it deferred
            ep.__dict__["_mono_pending"] = ep.__dict__["_mono_armed"] = None
            type(ep).begin_static_step.__wrapped__(ep, pending, False)
        return orig(self, *args, **kwargs)
    ep.__dict__["_mono_pending"] = None
    if pending != bs:
        ep.inject_gate_buf.fill_(1)
    ep._static_step_state = (bs, True)
    ep.__dict__["_mono_armed"] = bs
    ep.begin_sim_bonus(bs, info.temperatures[:bs])
    result = orig(self, *args, **kwargs)
    if result.can_run_cuda_graph:
        _ACCEPT[0] = (self, bs)
    else:
        ep.__dict__["_mono_armed"] = None
    return result


def _accept_and_finalize(orig, self, *args, **kwargs):
    mine, _ACCEPT[0] = _ACCEPT[0], None
    bs = kwargs.get("bs")
    if not kwargs.get("folded_accept", True) and mine == (self, bs):
        _COMMIT[0] = mine
        self.verify_epilogue.note_static_verify(bs)
        return self.verify_epilogue.read_accept(bs)
    return orig(self, *args, **kwargs)


def _commit_hidden(orig, self, *args, **kwargs):
    mine, _COMMIT[0] = _COMMIT[0], None
    if mine == (self, kwargs.get("bs")) and kwargs.get("run_compact") is False:
        return None  # the verify graph committed these rows
    return orig(self, *args, **kwargs)


def _propose(orig, self, *args, **kwargs):
    proposal = orig(self, *args, **kwargs)
    _FOLDED[0] = bool(proposal.folded)
    return proposal


def install() -> None:
    if not flag(ENV):
        return
    import importlib

    verify, draft = importlib.import_module(VERIFY), importlib.import_module(DRAFT)
    ep, ex = verify.DsparkVerifyEpilogue, verify.TargetVerifyExecutor
    route(ep, "__init__", _epilogue_init)
    route(ep, "begin_sim_bonus", _begin_sim_bonus)
    route(ep, "_static_accept_tail", _static_accept_tail)
    route(ex, "begin_simulated_step", _begin_simulated_step)
    if float(os.environ.get("SGLANG_SIMULATE_ACC_LEN", "-1")) > 0:
        # the epilogue draws the simulated bonus in the graph (read at its build)
        os.environ.setdefault("SGLANG_DSPARK_SIMULATE_FOLD", "1")
        route(ep, "begin_static_step", _begin_static_step)
        route(ex, "run_non_compact", _run_non_compact)
        route(ex, "accept_and_finalize", _accept_and_finalize)
        route(ex, "commit_hidden", _commit_hidden)
        route(draft.DraftBlockProposer, "propose", _propose)
