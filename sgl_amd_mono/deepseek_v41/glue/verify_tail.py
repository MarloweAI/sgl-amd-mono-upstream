# SPDX-License-Identifier: Apache-2.0
"""The static verify graph's accept where its logits are the rank's bf16 vocab shard
(SGLang's "greedy" verify variant with ``defer_greedy_verify_gather``): the ranks' row keys,
their exchange and the accept tail in one launch (``gfx950/vtail``), in place of SGLang's
per-rank keys, keys all-gather, selection and accept tail. Bitwise SGLang's.

Where the verify's rows fit (``gfx950/vtail/fused``), the vocab projection and that accept
are one launch after the head's combine and norm (``fused_head``, from the monokernel
forward): the logits processor returns its shard in place of its head GEMM, and the
epilogue's accept takes the outputs it wrote. Its logits are not hipBLASLt's bits.
``SGLANG_AMD_MONO_DSV41_VERIFY_TAIL=0`` keeps SGLang's."""

from __future__ import annotations

import importlib
import weakref

import torch

from sgl_amd_mono.deepseek_v41.glue import chose, flag, route

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
        "target_logits_processor",
        "_serves_accept_tail",
        "folds_commit",
        "verify_lens_buf",
    ),
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
    if epi is None or epi._tail_broadcasts:
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


def fused_head(lm, forward_batch, x):
    """The head's projection of a greedy-variant target verify's normed rows ``x`` and the
    epilogue's accept in one launch: its logits shard handed to the logits processor and its
    accept's outputs to the epilogue. Returns whether the forward takes it."""
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
    logits = launch(
        x, lm.lm_head.weight, candidates, forward_batch.seq_lens, verify_lens,
        epi.inject_gate_buf, sim=sim, temps=temps if sim is not None else None,
        seed=epi.sim_seed_buf if sim is not None else None,
        correct_out=epi.correct_len_buf, bonus_out=epi.bonus_buf,
        cap_trim_out=epi.cap_trim_lens_buf, commit_out=epi.commit_lens_buf,
        new_seq_out=epi.new_seq_lens_buf, out_tokens=epi.out_tokens_buf,
        gated_out=epi.gated_commit_lens_buf if gated else None,
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
        self._log_accept_tail(bs, bonus_in=None, shard=shard)
        return self.gated_commit_lens_buf[:bs] if gated else None
    if not chose("verify_tail", why is None, why):
        return orig(
            self, candidates=candidates, logits=logits, seq_lens=seq_lens,
            verify_lens=verify_lens, gated=gated,
        )  # fmt: skip
    launch, zeros = _tail(self, shard, logits.shape[1])
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
    route(
        importlib.import_module(LOGITS).LogitsProcessor,
        "_compute_lm_head",
        _compute_lm_head,
    )
