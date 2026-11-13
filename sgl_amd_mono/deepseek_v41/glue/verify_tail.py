# SPDX-License-Identifier: Apache-2.0
"""The static verify graph's accept where its logits are the rank's bf16 vocab shard
(SGLang's "greedy" verify variant with ``defer_greedy_verify_gather``): the ranks' row keys,
their exchange and the accept tail in one launch (``gfx950/vtail``), in place of SGLang's
per-rank keys, keys all-gather, selection and accept tail. Bitwise SGLang's.

Where the verify's rows fit (``gfx950/vtail/fused``), the vocab projection and that accept
are one launch after the head's combine and norm (``fused_head``, from the monokernel
forward): the logits processor returns its shard in place of its head GEMM, and the
epilogue's accept takes the outputs it wrote. Its logits are not hipBLASLt's bits.
An all-sampled static step's accept (SGLang's block-wise rejection, its simulated bonus and
accept tail) runs on the same shard and the draft's step logits' columns of this rank
(``gfx950/vtail/sampled``), in the sampling-accept region after the "sampled" variant, which
then keeps the shard (``sampled_accept_on_shard``). Its outcomes are SGLang's draws.

``SGLANG_AMD_MONO_DSV41_VERIFY_TAIL=0`` keeps SGLang's."""

from __future__ import annotations

import importlib
import weakref

import torch

from sgl_amd_mono.deepseek_v41.glue import chose, flag, inject, route

ENV = "SGLANG_AMD_MONO_DSV41_VERIFY_TAIL"
VERIFY = "sglang.srt.speculative.dspark_components.dspark_verify"
LOGITS = "sglang.srt.layers.logits_processor"
DEADLINE_S = 60.0

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "srt/speculative/dspark_components/dspark_verify.py": (
        "DsparkVerifyEpilogue",
        "_static_accept_tail",
        "verify_shard",
        "simulate_fold",
        "_tp_sync",
        "SpecTpSyncSite",
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
        "target_logits_processor",
        "_serves_accept_tail",
        "folds_commit",
        "verify_lens_buf",
        "run_static",
        "run_static(kv_projected)",
        "hidden",
        # the engram commit in the accept's launch
        "_commit_engram_history",
        "_commit_engram_history(candidates)",
        "_commit_engram_history(req_pool_indices)",
        "_commit_engram_history(commit_lens)",
        "_commit_engram_history(verify_lens)",
        "_commit_engram_history(gate)",
        "_commit_engram_history(relay)",
        "engram_history",
        "_future_relay_rows",
        "_post_fold",
        "req_pool_indices",
        "deferred_draft_kv",
        # the sampled accept on the shard
        "sampled_accept_on_shard",
        "_static_sampled_tail",
        "_static_sampled_tail(draft_logits)",
        "_static_sampled_tail(temperatures)",
        "_static_sampled_tail(rejection_noise)",
        "sampled_shard_serves",
    ),
    "srt/speculative/spec_tp_sync.py": ("DSPARK_ACCEPT_GRAPH",),
    "kernels/ops/speculative/dspark/blockwise_rejection.py": (
        "RejectionNoise",
        "key",
        "counter",
    ),
    "kernels/ops/embeddings/engram_hash.py": ("_engram_commit_history_kernel",),
    "srt/layers/logits_processor.py": (
        "LogitsProcessor",
        "deferred_shard_offset",
        "vocab_size",
        "defer_greedy_verify_gather",
        "_compute_lm_head",
        "_compute_lm_head(hidden_states)",
        "_compute_lm_head(lm_head)",
        "_compute_lm_head(embedding_bias)",
        "should_apply_lm_head_quant_method",
        "logit_scale",
        "do_tensor_parallel_all_gather",
        "do_tensor_parallel_all_gather_dp_attn",
        "use_attn_tp_group",
        "use_fp32_lm_head",
        "use_fp32_lm_head_output",
        "rl_on_policy_target",
    ),
    "srt/model_executor/runner_utils/capture_mode.py": ("get_capture_verify_variant",),
    "srt/runtime_context.py": ("get_parallel",),
}

# per epilogue: (the launch, a zero temperature row); per ("fused", epilogue): the fused launch
_TAILS: dict = {}
_EPILOGUES: weakref.WeakSet = weakref.WeakSet()  # every DsparkVerifyEpilogue built
# what the fused launch made in the forward being run or captured: the logits processor's
# head output (per processor) and the epilogue's accept (per epilogue)
_HEADS: dict = {}
_ACCEPTS: dict = {}
# per epilogue: the commit rows of the static verify being run or captured (``run_static``)
_COMMITS: dict = {}
# per epilogue: its requests' slots in the static verify being run or captured (``run_static``), and
# the engram commit the accept's launch ran for this step: (history, slots, candidates) addresses
_SLOTS: dict = {}
_COMMITTED: dict = {}


def _commit_args(epi, slots, candidates, bs, gate=None):
    """(history, slots, relay lens, relay bonus) of SGLang's engram commit that follows the accept
    (``run_static``'s), for the accept's launch to run, or None; noted for ``_commit_engram_history``."""
    _COMMITTED.pop(id(epi), None)
    history = epi.engram_history() if slots is not None else None
    if (
        history is None
        or history.dtype != torch.int32
        or history.stride(1) != 1
        or history.shape[1] > 8
    ):
        return None
    relay = epi._future_relay_rows(bs) if epi._post_fold else None
    if relay is not None and (relay[2].data_ptr(), relay[3].data_ptr()) != (
        epi.new_seq_lens_buf.data_ptr(),
        epi.bonus_buf.data_ptr(),
    ):
        return None
    gate = epi.inject_gate_buf if gate is None else gate
    _COMMITTED[id(epi)] = (
        history.data_ptr(),
        slots.data_ptr(),
        candidates.data_ptr(),
        relay is not None,
        gate.data_ptr(),
    )
    return history, slots, *(relay[:2] if relay is not None else (None, None))


def _why_not(epi, shard, logits, candidates) -> str | None:
    from sgl_amd_mono.deepseek_v41.gfx950 import vtail

    if shard is None:
        return "whole logits"
    if _tail_broadcasts(epi):
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


def _tail(epi, shard, part: int):
    """The epilogue's launches (built at a warmup, before capture): the accept from the
    shard, and the fused tail where the shard fits it."""
    got = _TAILS.get(id(epi))
    if got is None:
        from sglang.srt.runtime_context import get_parallel

        from sgl_amd_mono.deepseek_v41.gfx950 import vtail
        from sgl_amd_mono.deepseek_v41.gfx950.vtail import fused

        assert not torch.cuda.is_current_stream_capturing(), (
            "the verify tail's peer buffers are built at a warmup, before capture"
        )
        group = get_parallel().tp_group
        assert group.world_size <= vtail.MAX_TP
        vocab_start, width = shard.deferred_shard_offset(torch.empty(0, part))
        device = epi.inject_gate_buf.device
        args = dict(part=part, width=width, vocab_start=vocab_start,
                    vocab=shard.vocab_size, deadline_s=DEADLINE_S)  # fmt: skip
        launch = vtail.VerifyTail(
            group.cpu_group, group.rank_in_group, group.world_size, device, **args
        )
        zeros = torch.zeros_like(epi.sim_temperatures_buf)
        got = _TAILS[id(epi)] = (launch, zeros)
        if fused.MIN_BLOCKS <= fused.blocks(part) <= fused.MAX_BLOCKS:
            _TAILS["fused", id(epi)] = fused.FusedTail(
                group.cpu_group, group.rank_in_group, group.world_size, device, **args
            )
    return got


def _tail_broadcasts(epi) -> bool:
    """Whether the epilogue's accept-site TP broadcast runs."""
    site = importlib.import_module(VERIFY).SpecTpSyncSite.DSPARK_ACCEPT_GRAPH
    return epi._tp_sync.enabled(site)


def _epilogue(lp):
    """The verify epilogue whose accept reads ``lp``'s shard (defer_greedy_verify_gather)."""
    found = [e for e in _EPILOGUES if e.target_logits_processor is lp]
    return found[0] if len(found) == 1 else None


def _fused_why_not(lm, forward_batch, x) -> str | None:
    """Why the forward's head cannot be the fused launch (static: the capture's variant,
    shapes and the logits processor's settings), or None."""
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_verify_variant,
    )

    from sgl_amd_mono.deepseek_v41.gfx950.vtail import fused

    if not forward_batch.forward_mode.is_target_verify():
        return "not a target verify"
    if get_capture_verify_variant() != "greedy":
        return "not the greedy verify variant"
    lp, head = lm.logits_processor, lm.lm_head
    if not lp.defer_greedy_verify_gather:
        return "the logits processor gathers the greedy verify's logits"
    epi = _epilogue(lp)
    if epi is None or _tail_broadcasts(epi):
        return "no epilogue accept on the shard"
    lpm = importlib.import_module(LOGITS)
    weight = getattr(head, "weight", None)
    if (
        weight is None
        or hasattr(head, "set_lora")
        or lpm.should_apply_lm_head_quant_method(
            head, getattr(head, "quant_method", None)
        )
        or lp.use_fp32_lm_head
        or lp.use_fp32_lm_head_output
        or lp.rl_on_policy_target is not None
        or lp.logit_scale is not None
        or not lp.do_tensor_parallel_all_gather
        or lp.use_attn_tp_group
        or lp.do_tensor_parallel_all_gather_dp_attn
    ):
        return "the logits processor's head"
    part, hidden = weight.shape
    if (
        weight.dtype != torch.bfloat16
        or weight.stride(1) != 1
        or weight.stride(0) % 8
        or weight.data_ptr() % 16
        or hidden != x.shape[1]
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.data_ptr() % 16
        or not fused.fits(part=part, hidden=hidden, vocab=lp.vocab_size)
    ):
        return f"a {tuple(weight.shape)} {weight.dtype} head shard"
    bs, t = forward_batch.batch_size, epi.stride
    if not epi._serves_accept_tail(bs) or forward_batch.input_ids.numel() != bs * t:
        return f"{bs} x {t} verify rows"
    if x.shape[0] != bs * t or t > fused.MAX_T:
        return f"{bs} x {t} verify rows"
    return None


def fused_head(lm, forward_batch, x, commit_rows=None):
    """The head's projection of a greedy-variant target verify's normed rows ``x`` and the
    epilogue's accept in one launch: its logits shard handed to the logits processor and its
    accept's outputs to the epilogue; with the commit's main_proj of ``commit_rows`` (the
    forward's DSpark captures, the epilogue's commit rows) where the commit takes it. Returns
    whether the forward takes it."""
    if not flag(ENV) or torch.version.hip is None:
        return False
    from sgl_amd_mono.deepseek_v41.gfx950.vtail import fused

    why = _fused_why_not(lm, forward_batch, x)
    if why is None:
        lp = lm.logits_processor
        epi = _epilogue(lp)
        _tail(epi, lp, lm.lm_head.weight.shape[0])
        bs = forward_batch.batch_size
        if bs * epi.stride > fused.MAX_ROWS:
            why = f"{bs} x {epi.stride} verify rows"
    if not chose("fused_tail", why is None, why):
        return False
    launch = _TAILS["fused", id(epi)]
    bs = forward_batch.batch_size
    candidates = forward_batch.input_ids.view(bs, epi.stride)
    verify_lens = epi.verify_lens_buf[:bs]
    gated = epi.folds_commit
    sim = epi.simulated_correct_len
    temps = (epi.sim_temperatures_buf if epi.simulate_fold else _TAILS[id(epi)][1])[:bs]
    # the commit that follows in this forward: its main_proj in this launch
    proj = (
        inject.projection(epi, commit_rows)
        if gated and commit_rows is not None
        else None
    )
    logits = launch(
        x, lm.lm_head.weight, candidates, forward_batch.seq_lens, verify_lens,
        epi.inject_gate_buf, sim=sim, temps=temps if sim is not None else None,
        seed=epi.sim_seed_buf if sim is not None else None,
        correct_out=epi.correct_len_buf, bonus_out=epi.bonus_buf,
        cap_trim_out=epi.cap_trim_lens_buf, commit_out=epi.commit_lens_buf,
        new_seq_out=epi.new_seq_lens_buf, out_tokens=epi.out_tokens_buf,
        gated_out=epi.gated_commit_lens_buf if gated else None, proj=proj,
        commit=_commit_args(epi, forward_batch.req_pool_indices, candidates, bs),
    )  # fmt: skip
    _HEADS[id(lp)] = (x, logits)
    _ACCEPTS[id(epi)] = (logits, candidates.data_ptr(), forward_batch.seq_lens.data_ptr(),
                         verify_lens.data_ptr(), gated)  # fmt: skip
    return True


def _compute_lm_head(orig, self, hidden_states, lm_head, embedding_bias=None, **kwargs):
    got = _HEADS.pop(id(self), None)
    if got is None:
        return orig(self, hidden_states, lm_head, embedding_bias, **kwargs)
    hidden, logits = got
    if (
        hidden_states.data_ptr() != hidden.data_ptr()
        or hidden_states.shape != hidden.shape
        or embedding_bias is not None
    ):
        raise RuntimeError(
            "DSV4.1 fused verify tail: the logits processor's head input is not the "
            f"launch's rows ({tuple(hidden_states.shape)} vs {tuple(hidden.shape)})"
        )
    return logits


def _static_accept_tail(
    orig, self, *, candidates, logits, seq_lens, verify_lens, gated
):
    shard = self.verify_shard(logits)
    why = _why_not(self, shard, logits, candidates)
    done = _ACCEPTS.pop(id(self), None)
    if done is not None:
        # the fused launch wrote this accept's outputs
        same = (
            logits.data_ptr(), candidates.data_ptr(), seq_lens.data_ptr(),
            verify_lens.data_ptr(), gated,
        ) == (done[0].data_ptr(), *done[1:])  # fmt: skip
        if why is not None or not same:
            raise RuntimeError(
                "DSV4.1 fused verify tail: the epilogue's accept is not the launch's "
                f"({why or 'other tensors'})"
            )
        bs = candidates.shape[0]
        self._log_accept_tail(bs, shard=shard)
        return self.gated_commit_lens_buf[:bs] if gated else None
    _COMMITTED.pop(id(self), None)
    if not chose("verify_tail", why is None, why):
        return orig(
            self, candidates=candidates, logits=logits, seq_lens=seq_lens,
            verify_lens=verify_lens, gated=gated,
        )  # fmt: skip
    launch, zeros = _tail(self, shard, logits.shape[1])
    bs = candidates.shape[0]
    commit = _commit_args(self, _SLOTS.get(id(self)), candidates, bs)
    sim = self.simulated_correct_len
    # SGLang's tail draws a simulated length's bonus at the request's temperature under
    # simulate_fold, else takes that row's argmax
    temps = (self.sim_temperatures_buf if self.simulate_fold else zeros)[:bs]
    self._log_accept_tail(bs, shard=shard)
    # the commit that follows in this forward: its main_proj in this launch
    hidden = _COMMITS.get(id(self))
    proj = inject.projection(self, hidden) if gated and hidden is not None else None
    launch(
        logits, candidates, seq_lens, verify_lens, self.inject_gate_buf,
        sim=sim, temps=temps if sim is not None else None,
        seed=self.sim_seed_buf if sim is not None else None,
        correct_out=self.correct_len_buf, bonus_out=self.bonus_buf,
        cap_trim_out=self.cap_trim_lens_buf, commit_out=self.commit_lens_buf,
        new_seq_out=self.new_seq_lens_buf, out_tokens=self.out_tokens_buf,
        gated_out=self.gated_commit_lens_buf if gated else None, proj=proj, commit=commit,
    )  # fmt: skip
    return self.gated_commit_lens_buf[:bs] if gated else None


def _sampled_why_not(epi, shard, logits, candidates, draft_logits, noise) -> str | None:
    from sgl_amd_mono.deepseek_v41.gfx950.vtail import sampled

    if shard is None:
        return "whole logits"
    bs, t = candidates.shape
    if logits.dtype != torch.bfloat16 or logits.stride(1) != 1 or logits.dim() != 2:
        return f"{logits.dtype} logits"
    if logits.shape[0] != bs * t or t > sampled.MAX_T or bs > sampled.MAX_SEQ:
        return f"{bs} x {t} verify rows"
    if logits.shape[1] % 8 or logits.stride(0) % 8 or logits.data_ptr() % 16:
        return "unaligned shard rows"
    if noise is None:
        return "no block-wise rejection noise"
    if (
        draft_logits is None
        or draft_logits.dtype != torch.float32
        or draft_logits.stride(2) != 1
    ):
        return "no f32 draft step logits"
    # the draft's step logits: whole rows, or this rank's vocab shard as the verify's
    # (SGLang's sharded accept)
    if (
        draft_logits.shape[-1] not in (shard.vocab_size, logits.shape[1])
        or draft_logits.stride(1) % 4
        or draft_logits.data_ptr() % 16
    ):
        return f"draft step logits {tuple(draft_logits.shape)}"
    return None


def _sampled_shard_serves(orig, self, bs: int, rows_per_request: int) -> bool:
    """The steps the sampled accept's launch takes (its request and row bounds); SGLang
    gathers the whole logits for the others."""
    from sgl_amd_mono.deepseek_v41.gfx950.vtail import sampled

    return bs <= sampled.MAX_SEQ and rows_per_request <= sampled.MAX_T


def _sampled(epi, shard, part: int):
    """The epilogue's sampled accept launch (built at the region's eager first step)."""
    got = _TAILS.get(("sampled", id(epi)))
    if got is None:
        from sglang.srt.runtime_context import get_parallel

        from sgl_amd_mono.deepseek_v41.gfx950.vtail import sampled

        assert not torch.cuda.is_current_stream_capturing(), (
            "the sampled accept's peer buffers are built before capture"
        )
        group = get_parallel().tp_group
        vocab_start, width = shard.deferred_shard_offset(torch.empty(0, part))
        if vocab_start % 4 or not sampled.fits(
            tp=group.world_size, width=width, vocab=shard.vocab_size
        ):
            raise RuntimeError(
                f"DSV4.1 sampled verify tail: a {width}-column shard at TP {group.world_size}"
            )
        got = _TAILS["sampled", id(epi)] = sampled.SampledTail(
            group.cpu_group, group.rank_in_group, group.world_size, epi.inject_gate_buf.device,
            width=width, vocab_start=vocab_start, vocab=shard.vocab_size, deadline_s=DEADLINE_S,
        )  # fmt: skip
    return got


def _static_sampled_tail(orig, self, *, candidates, logits, draft_logits, temperatures, rejection_noise,
                         seq_lens, verify_lens, gate, gated):  # fmt: skip
    shard = self.verify_shard(logits)
    why = _sampled_why_not(
        self, shard, logits, candidates, draft_logits, rejection_noise
    )
    if why is not None:
        raise RuntimeError(f"DSV4.1 sampled verify tail: {why}")
    from sgl_amd_mono.deepseek_v41.draft_sampler import step_logits_shard

    launch = _sampled(self, shard, logits.shape[1])
    bs = candidates.shape[0]
    start = launch.vocab_start
    # the kernels' Markov block keeps this rank's columns; SGLang's writes the whole rows or,
    # with its sharded accept, this rank's shard of them
    draft = step_logits_shard(draft_logits, bs)
    if draft is None:
        first = start if draft_logits.shape[-1] == shard.vocab_size else 0
        draft = draft_logits[:bs, :, first : first + launch.width]
    commit = _commit_args(self, _SLOTS.get(id(self)), candidates, bs, gate)
    sim = self.simulated_correct_len
    # the commit that follows in this forward: its main_proj in this launch
    hidden = _COMMITS.get(id(self))
    proj = inject.projection(self, hidden) if gated and hidden is not None else None
    launch(
        logits, draft, temperatures,
        key=rejection_noise.key, counter=rejection_noise.counter, candidates=candidates,
        seq_lens=seq_lens, verify_lens=verify_lens, gate=gate, sim=sim,
        sim_seed=self.sim_seed_buf if sim is not None else None,
        correct_out=self.correct_len_buf, bonus_out=self.bonus_buf,
        cap_trim_out=self.cap_trim_lens_buf, commit_out=self.commit_lens_buf,
        new_seq_out=self.new_seq_lens_buf, out_tokens=self.out_tokens_buf,
        gated_out=self.gated_commit_lens_buf if gated else None, proj=proj, commit=commit,
    )  # fmt: skip
    return self.gated_commit_lens_buf[:bs] if gated else None


def _run_static(orig, self, io, *, accept, gate, kv_projected=False):
    # kv_projected: a deferred commit's projection already joined (without one, the commit projects here)
    _COMMITS[id(self)] = (
        None if kv_projected and self.deferred_draft_kv is not None else io.hidden
    )
    _SLOTS[id(self)] = io.req_pool_indices
    if (
        id(self) not in _ACCEPTS
    ):  # a commit claim stands only with the fused launch's accept of this forward
        _COMMITTED.pop(id(self), None)
    try:
        return orig(self, io, accept=accept, gate=gate, kv_projected=kv_projected)
    finally:
        _COMMITS.pop(id(self), None)
        _SLOTS.pop(id(self), None)
        _COMMITTED.pop(id(self), None)
        inject.check_projected()


def _commit_engram_history(
    orig, self, candidates, req_pool_indices, commit_lens, verify_lens, gate, *, relay
):
    """SGLang's engram commit, or nothing where the accept's launch ran it (``_commit_args``)."""
    done = _COMMITTED.pop(id(self), None)
    if done is None:
        return orig(
            self,
            candidates,
            req_pool_indices,
            commit_lens,
            verify_lens,
            gate,
            relay=relay,
        )
    history = self.engram_history()
    want = (
        history.data_ptr(),
        req_pool_indices.data_ptr(),
        candidates.data_ptr(),
        relay is not None,
        gate.data_ptr(),
    )
    same = (commit_lens.data_ptr(), verify_lens.data_ptr()) == (
        self.commit_lens_buf.data_ptr(),
        self.verify_lens_buf.data_ptr(),
    )
    if done != want or not same:
        raise RuntimeError(
            "DSV4.1 accept commit: the epilogue's engram commit is not the launch's"
        )
    return True


def _epilogue_init(orig, self, *args, **kwargs):
    orig(self, *args, **kwargs)
    _EPILOGUES.add(self)


def install() -> None:
    if not flag(ENV) or torch.version.hip is None:
        return
    verify = importlib.import_module(VERIFY)
    if not hasattr(verify.DsparkVerifyEpilogue, "verify_shard"):
        return
    route(verify.DsparkVerifyEpilogue, "__init__", _epilogue_init)
    route(verify.DsparkVerifyEpilogue, "_static_accept_tail", _static_accept_tail)
    route(verify.DsparkVerifyEpilogue, "run_static", _run_static)
    route(verify.DsparkVerifyEpilogue, "_commit_engram_history", _commit_engram_history)
    if hasattr(verify.DsparkVerifyEpilogue, "_static_sampled_tail"):
        route(verify.DsparkVerifyEpilogue, "_static_sampled_tail", _static_sampled_tail)
        route(
            verify.DsparkVerifyEpilogue, "sampled_shard_serves", _sampled_shard_serves
        )
        verify.DsparkVerifyEpilogue.sampled_accept_on_shard = True
    route(
        importlib.import_module(LOGITS).LogitsProcessor,
        "_compute_lm_head",
        _compute_lm_head,
    )
