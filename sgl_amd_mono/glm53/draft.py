# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3's NextN (MTP) draft on gfx950: its spec and forward (RFC
sgl-project/sglang#43949).

EAGLE's NextN draft runs its own steps: the draft extend over each request's verify
rows (DRAFT_EXTEND_V2, R requests x the draft tokens) and the chain's draft decode
steps (DECODE, R x 1), which reuse the top-k the draft extend selected (SGLang's MTP
index share). Monokernel API version 1 routes a draft model by the target's verify
key, which the draft never runs, so at build the spec puts a wrapper in front of the
model's forward (before the runtime installs its own over it) that routes the
draft's own steps by forward mode, batch size, rows and the presence of the reused
top-k alone: a captured graph replays the decision made at capture, as a routed step
does. It relies on every draft call site calling the instance attribute
``model.forward``, and checks the draft's attention backend once per step mode, as
the runtime does for a routed step. SGLang's capture check never sees a draft step,
so the wrapper logs each draft key at its first launch and refuses, during graph
capture, a draft-shaped step it cannot take.

A routed step is two launches: the NextN layer from the step's tokens and its entry,
eh_proj(enorm(e) | hnorm(h)) (``gfx950/kernel.py``, ``eh_entry``: e is the tokens'
embedding rows, pushed by the rank that holds each), and the tail with the NextN's
norm (shared_head.norm, the shared LM head, bf16 logits gathered as f32). A draft
extend runs the indexer kind over all its rows, writes the NextN MLA and index-K rows
of each, and hands its top-k to SGLang as the seed (``dsa_seed_topk_capture``); a
draft decode step runs the shared kind on the seed's valid keys. The draft's top-1
stays SGLang's: the logits processor returns the tail's logits on a routed step
(``_get_logits``), the draft extend's selected rows in a graph. The weights are
SGLang's, read in place (eh_proj as this rank's row view; the embedding and LM head,
which SGLang shares with the target after the draft is built, as bound at a step),
except the copies logged at startup, as for the target layers.

A draft decode step that samples its proposal (under rejection sampling, a batch that
is not all greedy: the draft graph runner's sampled variant) runs SGLang's path. The
logits are the tail's (bf16, gathered as f32, as SGLang's), so the draft extend's
proposal, greedy or sampled, is SGLang's on them. A speculative token map, no MTP
index share or another seed width leave the whole draft on SGLang's path, as the
startup log says. Check mode (``SGLANG_AMD_MONO_CHECK``, eager) compares each routed
step with SGLang's NextN on the same inputs (``check.draft``).
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional

import msgspec
import torch

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
from sgl_amd_mono.glm53 import spec as target
from sgl_amd_mono.glm53.gfx950.config import HIDDEN

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Requirements",
        "widths",
        "deadline_s",
        "copies",
    ),
    "srt/models/deepseek_nextn.py": (
        "model",
        "embed_tokens",
        "enorm",
        "hnorm",
        "eh_proj",
        "decoder",
        "shared_head",
        "lm_head",
        "logits_processor",
    ),
    "srt/models/deepseek_v2.py": (
        "self_attn",
        "attn_mqa",
        "layer_id",
        "indexer",
        "input_layernorm",
    ),
    "srt/layers/attention/dsa/dsa_indexer.py": ("use_gfx950_fused_indexer",),
    "srt/model_executor/forward_batch_info.py": (
        "spec_info",
        "forward_mode",
        "batch_size",
        "input_ids",
        "is_draft_extend_v2",
        "is_decode",
        "reuse_dsa_topk_indices",
        "out_cache_loc",
        "sampling_info",
    ),
    "srt/sampling/sampling_batch_info.py": ("is_all_greedy",),
    "srt/speculative/*.py": (
        "hidden_states",
        "dsa_topk_indices",
        "dsa_seed_topk_capture",
    ),
    "srt/layers/layer_boundary/residual/batch.py": ("start", "clear"),
    "srt/layers/logits_processor.py": ("_get_logits", "draft_extend_select_index"),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/base_attn_backend.py": ("decode_attention_backend_str",),
    "srt/layers/attention/dsa_backend.py": ("forward_metadata", "real_page_table"),
    "srt/configs/model_config.py": ("get_dsa_mtp_topk_width", "context_len"),
    "srt/arg_groups/fields/*.py": (
        "kv_cache_dtype",
        "speculative_token_map",
        "speculative_use_rejection_sampling",
        "speculative_num_draft_tokens",
    ),
    "srt/runtime_context.py": (
        "get_model",
        "get_parallel",
        "get_spec",
        "process_model_config",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
    "srt/distributed/parallel_state.py": ("cpu_group",),
}

# the draft's steps run on the DSA backend, whose caches the launches write
DRAFT_ATTENTION_BACKENDS = frozenset({"dsa"})
# a test fault for the check's negative controls (each must fail a check):
# count_by_position (a draft decode row's reused top-k sized by its position),
# swap_norms (the entry's enorm and hnorm weights swapped), seed_row0 (every seed row
# row 0's), rope_shift (the NextN layer rotated 4096 positions off)
FAULT_ENV = "SGLANG_AMD_MONO_DRAFT_FAULT"


def _tokens_per_request(model) -> int:
    """A draft extend's rows a request: the target's verify rows."""
    from sglang.srt.runtime_context import get_spec

    return int(get_spec().speculative_num_draft_tokens)


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    # what the launches cannot serve leaves the draft on SGLang's path (``_build``)
    return []


def _standard_reasons(model, hf_config) -> list[str]:
    """Why the draft stays on SGLang's path."""
    from sglang.srt.configs.model_config import get_dsa_mtp_topk_width
    from sglang.srt.runtime_context import get_parallel, get_spec

    reasons = []
    if get_spec().speculative_token_map is not None:
        reasons.append("a speculative token map")
    # an optional config field: SGLang reads it the same way
    if not getattr(hf_config, "index_share_for_mtp_iteration", False):
        reasons.append("the draft decode steps need SGLang's MTP index share")
    elif get_dsa_mtp_topk_width(hf_config) != target.TOPK:
        reasons.append(f"a seed of {get_dsa_mtp_topk_width(hf_config)} keys")
    indexer = model.model.decoder.self_attn.indexer
    if indexer is None or not indexer.use_gfx950_fused_indexer:
        reasons.append("needs SGLang's gfx950 fused DSA indexer on the NextN layer")
    if get_parallel().tp_size not in target.TP_SIZES:
        reasons.append(f"TP{get_parallel().tp_size}")
    if not isinstance(model.model.eh_proj, torch.nn.Linear):
        reasons.append(f"eh_proj is a {type(model.model.eh_proj).__name__}")
    return reasons


def _is_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class _Standard:
    """A draft on SGLang's path."""

    def __init__(self, lm):
        self.lm = lm

    def forward(self, plan, *args, **kwargs):
        return type(self.lm).forward(self.lm, *args, **kwargs)


class _Draft:
    """The NextN launches (per (kind, width)), the tails (per width), their step
    tables, and the draft's forward."""

    def __init__(self, lm, layer, ops, heads, epochs, cos, sin, max_position):
        from sglang.srt.runtime_context import get_spec

        self.lm, self.L, self.ops, self.heads = lm, layer, ops, heads
        # per width, the guard blocks of its NextN and tail runtimes: each launches at
        # most once a draft step, so one advance a step keeps their epochs apart
        self.epochs = epochs
        self.cos, self.sin = cos, sin
        self.max_position = max_position
        # a draft extend's rows at most (whole requests), a draft decode step's
        self.max_extend = max(w for k, w in ops if k == "index")
        self.max_decode = max(w for k, w in ops if k == "shared")
        self.extend_rows = _tokens_per_request(lm)
        self.rejection_sampling = bool(get_spec().speculative_use_rejection_sampling)
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"
        self.fault = os.environ.get(FAULT_ENV, "")
        if self.fault:
            logger.warning("monokernel %s: test fault %r is on", SPEC.name, self.fault)
        self.seen, self.refused, self.backend_checked = set(), set(), set()
        self.decode_step = 0  # the draft decode steps since the last draft extend
        device = cos.device
        self.steps, self.ids = {}, {}
        for key in ops:
            self.steps[key] = _new_step(key[1], device)
            self.ids[key] = torch.full((key[1],), -1, dtype=torch.int32, device=device)
        self.emb_ptr = None
        # the step row each seed row takes its keys from
        self.rows_of = {
            w: (
                torch.zeros(w, dtype=torch.int32, device=device)
                if self.fault == "seed_row0"
                else torch.arange(w, dtype=torch.int32, device=device)
            )
            for w in {w for _, w in ops}
        }
        lp = lm.logits_processor
        self._own_logits, self._logits = lp._get_logits, None
        lp._get_logits = self._get_logits

    def forward(self, plan, *args, **kwargs):
        """A step the runtime routes by the target's key, which the draft never
        runs: SGLang's forward."""
        return type(self.lm).forward(self.lm, *args, **kwargs)

    def _get_logits(self, hidden_states, lm_head, logits_metadata, *args, **kwargs):
        """The logits processor's logits: the tail's on a routed step (the draft
        extend graph's selected rows), else its own."""
        if self._logits is None:
            return self._own_logits(
                hidden_states, lm_head, logits_metadata, *args, **kwargs
            )
        sel = logits_metadata.draft_extend_select_index
        if sel is not None:
            return self._logits.index_select(0, sel)
        return self._logits[: hidden_states.shape[0]]

    # ---------------------------------------------------------------- routing
    def _kind(self, fb) -> Optional[tuple[str, int]]:
        """(kind, width) of a draft-shaped step: a draft extend of whole requests,
        or a draft decode step that reuses a top-k."""
        R, n = int(fb.batch_size), int(fb.input_ids.shape[0])
        mode = fb.forward_mode
        if R < 1:
            return None
        if mode.is_draft_extend_v2():
            if n == R * self.extend_rows and n <= self.max_extend:
                return "index", n
            return None
        if mode.is_decode() and n == R and n <= self.max_decode:
            # a draft decode that samples its proposal (under rejection sampling, a
            # batch not all greedy) runs SGLang's path
            info = fb.sampling_info
            if self.rejection_sampling and info is not None and not info.is_all_greedy:
                return None
            return "shared", target._width(n)
        return None

    def _why_not(self, fb, kind: str) -> Optional[str]:
        """Why the launches cannot take a draft-shaped step, or None."""
        spec_info, n = fb.spec_info, int(fb.input_ids.shape[0])
        hs = None if spec_info is None else spec_info.hidden_states
        if hs is None:
            return "no hidden states"
        if (
            hs.dtype != torch.bfloat16
            or tuple(hs.shape) != (n, HIDDEN)
            or not hs.is_contiguous()
        ):
            return f"hidden states {hs.dtype} {tuple(hs.shape)}"
        if kind == "shared":
            seed = spec_info.dsa_topk_indices
            if not fb.reuse_dsa_topk_indices or seed is None:
                return "no reused top-k (the draft decode would run the indexer)"
            if seed.dtype != torch.int32 or seed.dim() != 2 or seed.shape[0] < n:
                return f"a reused top-k {seed.dtype} {tuple(seed.shape)}"
        return None

    def takes(self, fb) -> bool:
        """Whether the launches run this step: static properties of the step only,
        so graph-capture safe. A draft-shaped step they cannot take is refused during
        graph capture and logged once eagerly."""
        kw = self._kind(fb)
        if kw is None:
            return False
        why = self._why_not(fb, kw[0])
        if why is None:
            return True
        key = (fb.forward_mode.name, int(fb.batch_size), int(fb.input_ids.shape[0]))
        msg = f"GLM-5.3 MTP draft step {key}: {why}"
        if _is_capturing():
            raise ValueError(msg + ": refusing to capture the standard path")
        if key not in self.refused:
            self.refused.add(key)
            logger.warning("%s: this step runs the standard path", msg)
        return False

    def check_backend(self, fb) -> None:
        """Refuse a draft step on an attention backend whose caches the launches do
        not write, once per step mode."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        mode = fb.forward_mode.name
        if mode in self.backend_checked:
            return
        backend = get_attn_backend().decode_attention_backend_str
        if backend not in DRAFT_ATTENTION_BACKENDS:
            raise ValueError(
                f"--monokernel-backend: {SPEC.name} needs a draft attention backend "
                f"in {sorted(DRAFT_ATTENTION_BACKENDS)}, this {mode} step's is "
                f"{backend!r}"
            )
        self.backend_checked.add(mode)

    # ------------------------------------------------------------------ steps
    def _head(self, W: int):
        """The tail at width W on the LM head bound now (SGLang shares the
        target's after the draft is built)."""
        from sglang.srt.runtime_context import get_parallel

        from sgl_amd_mono.glm53.gfx950.runner import Glm53HeadOp
        from sgl_amd_mono.glm53.gfx950.weights import head_weights

        op = self.heads[W]
        lm = self.lm
        if op.w.w_head.data_ptr() != lm.lm_head.weight.data_ptr():
            p = get_parallel()
            w = head_weights(lm, p.tp_size, p.tp_rank, lm.model.shared_head.norm)
            op = self.heads[W] = Glm53HeadOp(op.rt, w)
        return op

    def _bind_embedding(self) -> None:
        """The launches read the embedding bound now (SGLang shares the target's after
        the draft is built)."""
        from sgl_amd_mono.glm53.gfx950.layout import EH_ARGS

        ptr = self.lm.model.embed_tokens.weight.data_ptr()
        if ptr == self.emb_ptr:
            return
        from sglang.srt.runtime_context import get_parallel

        from sgl_amd_mono.glm53.gfx950.weights import embedding

        p = get_parallel()
        embedding(self.lm, p.tp_size, p.tp_rank)  # the shard checks of the build
        for op in self.ops.values():
            op.index[EH_ARGS[3]].fill_(ptr)
        self.emb_ptr = ptr

    def step(self, input_ids, positions, fb):
        """``GlmMoeDsaForCausalLMNextN.forward`` for a step the launches take."""
        from sglang.srt.layers.layer_boundary.residual.batch import clear, start

        from sgl_amd_mono.glm53 import tables

        lm, L = self.lm, self.L
        kind, W = self._kind(fb)
        S = int(input_ids.shape[0])
        key = (fb.forward_mode.name, int(fb.batch_size), S)
        if key not in self.seen:
            self.seen.add(key)
            logger.info("monokernel %s runs the MTP draft step %s", SPEC.name, key)
        op = self.ops[kind, W]
        target._Built._bind(L, kv_dtype=op.rt.kv_dtype)
        self._bind_embedding()
        spec_info = fb.spec_info
        st, ids = self.steps[kind, W], self.ids[kind, W]
        tables.step_rows(
            st,
            fb.out_cache_loc,
            positions,
            S,
            self.max_position,
            ids_in=input_ids,
            ids=ids,
            seed=spec_info.dsa_topk_indices if kind == "shared" else None,
            seed_by_position=self.fault == "count_by_position",
        )
        if kind == "index":
            from sglang.srt.model_executor.forward_context import get_attn_backend

            md = get_attn_backend().forward_metadata
            # the DSA backend gives each draft extend row its request's page-table row
            st = msgspec.structs.replace(st, page_table=md.real_page_table[:S])
        start(fb)
        h = spec_info.hidden_states
        checking = self.check and not _is_capturing()
        # the previous draft step's h rows are the tail's output, which this step's
        # tail overwrites
        h_in = h.clone() if checking else None
        x = op.launch(ids, L.caches, self.cos, self.sin, st, h_prev=h)
        seed_buf = spec_info.dsa_seed_topk_capture
        if kind == "index" and seed_buf is not None:
            tables.write_seed(st, seed_buf, S, self.rows_of[W])
        logits, normed = self._head(W).launch(x)
        self.epochs[W].advance()
        if checking:
            from sgl_amd_mono.glm53 import check

            draft = 1 if kind == "index" else 2 + self.decode_step
            e = lm.model.embed_tokens(input_ids)
            check.draft(
                self,
                L,
                op,
                kind,
                draft,
                positions,
                fb,
                S,
                st,
                e,
                h_in,
                x,
                logits,
                normed,
                seed_buf,
            )
        clear(fb)
        self.decode_step = 0 if kind == "index" else self.decode_step + 1
        self._logits = logits
        try:
            return lm.logits_processor(input_ids, normed[:S], lm.lm_head, fb)
        finally:
            self._logits = None


def _new_step(W: int, device):
    from sgl_amd_mono.glm53.gfx950.runner import Glm53Step

    i32, i64 = torch.int32, torch.int64
    return Glm53Step(
        positions=torch.zeros(W, dtype=i64, device=device),
        slots=torch.full((W,), -1, dtype=i64, device=device),
        indptr=torch.zeros(W + 1, dtype=i32, device=device),
        req_ids=torch.zeros(W, dtype=i32, device=device),
        page_table=torch.zeros(W, 1, dtype=i32, device=device),
        topk=torch.zeros(W * target.TOPK + 1, dtype=i32, device=device),
    )


def _route_draft_steps(model, draft: _Draft) -> None:
    """The draft-step wrapper in front of the model's forward (see the module
    docstring); every other step runs the forward it wraps. Runs in build, so the
    runtime's forward wraps this one."""
    standard = model.forward

    # graph runners inspect the original parameters to select forward arguments
    @functools.wraps(standard)
    def forward(*args, **kwargs):
        # (input_ids, positions, forward_batch, ...)
        fb = kwargs["forward_batch"] if "forward_batch" in kwargs else args[2]
        if draft.takes(fb):
            draft.check_backend(fb)
            ids = kwargs["input_ids"] if "input_ids" in kwargs else args[0]
            pos = kwargs["positions"] if "positions" in kwargs else args[1]
            return draft.step(ids, pos, fb)
        return standard(*args, **kwargs)

    model.forward = forward


def _build(model, hf_config, ctx):
    from sglang.srt.runtime_context import get_model, get_parallel, process_model_config

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.glm53.gfx950.layout import INDEX_KEYS_PER_TASK
    from sgl_amd_mono.glm53.gfx950.runner import (
        Glm53Build,
        Glm53EhWeights,
        Glm53Epochs,
        Glm53HeadOp,
        Glm53LayerOp,
        Glm53Runtime,
    )
    from sgl_amd_mono.glm53.gfx950.weights import (
        embedding,
        head_weights,
        layer_weights,
        rope_tables,
    )

    reasons = _standard_reasons(model, hf_config)
    if reasons:
        logger.warning(
            "monokernel %s: the MTP draft runs SGLang's path: %s",
            SPEC.name,
            "; ".join(reasons),
        )
        return _Standard(model)
    p = get_parallel()
    tp, rank = p.tp_size, p.tp_rank
    m = model.model
    decoder = m.decoder
    layer_id = decoder.self_attn.attn_mqa.layer_id
    weights = layer_weights(decoder, layer_id, tp, ctx.copies)
    rows = HIDDEN // tp
    eh_w = m.eh_proj.weight
    if eh_w.dtype != torch.bfloat16 or tuple(eh_w.shape) != (HIDDEN, 2 * HIDDEN):
        raise ValueError(
            f"GLM-5.3 MTP draft: needs a bf16 eh_proj [{HIDDEN}, {2 * HIDDEN}], got "
            f"{eh_w.dtype} {tuple(eh_w.shape)}"
        )
    g_e, g_h = m.enorm.weight, m.hnorm.weight
    if os.environ.get(FAULT_ENV) == "swap_norms":
        g_e, g_h = g_h, g_e
    eh = Glm53EhWeights(
        w_eh=eh_w[rank * rows : (rank + 1) * rows],
        g_e=g_e,
        g_h=g_h,
        emb=embedding(model, tp, rank),
    )
    cos, sin = rope_tables([decoder], ctx.copies)
    if os.environ.get(FAULT_ENV) == "rope_shift":
        cos, sin = cos.roll(-4096, 0), sin.roll(-4096, 0)
    context_len = process_model_config().context_len
    if cos.shape[0] < context_len:
        raise ValueError(
            f"GLM-5.3 MTP draft: the rotary table holds {cos.shape[0]} positions, "
            f"the server's context is {context_len}"
        )
    max_seq = -(-context_len // INDEX_KEYS_PER_TASK) * INDEX_KEYS_PER_TASK
    kv_fp8 = get_model().kv_cache_dtype == "fp8_e4m3"
    extend_widths = sorted(ctx.widths)
    # a draft decode step of 5 to 8 rows runs SGLang's path: the NextN shared launch
    # at 8 rows faulted in a kernel check (a process of several launches, not alone)
    decode_widths = [w for w in target.WIDTHS if w <= 4]
    layer = target._Layer(layer_id, decoder, weights)
    shared_w = msgspec.structs.replace(weights, index=None)
    widths = sorted(set(extend_widths) | set(decode_widths))
    device = decoder.input_layernorm.weight.device
    # a width's NextN index, NextN shared and tail runtimes
    epochs = {w: Glm53Epochs(3, device) for w in widths}
    ops, heads, runtimes = {}, {}, []

    def runtime(key):
        rt = Glm53Runtime(key, rank, p.tp_group.cpu_group, device, epochs[key.rows])
        # a NextN launch is a draft step's first, as layer 0's is a target step's
        rt.set_deadline(
            target._deadline(
                key, ctx.deadline_s, guard.MAX_DEADLINE_TICKS // guard.CLOCK_HZ
            )
        )
        runtimes.append(rt)
        return rt

    def nextn(W: int, indexer: bool):
        return Glm53Build(
            rows=W,
            tp=tp,
            launches_per_step=1,
            with_indexer=indexer,
            index_max_seq=max_seq if indexer else 0,
            fp8_shuffled=weights.shuffled,
            kv_fp8=kv_fp8,
            eh=True,
            tokens=True,
            vocab_rows=eh.emb.shape[0],
        )

    head = head_weights(model, tp, rank, m.shared_head.norm)
    # each runtime's peer handshake is collective: every rank builds them in one
    # order; a draft step's first launch is where the ranks' hosts meet
    for W in extend_widths:
        ops["index", W] = Glm53LayerOp(runtime(nextn(W, True)), weights, 0, eh)
    for W in decode_widths:
        ops["shared", W] = Glm53LayerOp(runtime(nextn(W, False)), shared_w, 0, eh)
    for W in widths:
        key = Glm53Build(
            rows=W,
            tp=tp,
            launches_per_step=1,
            with_indexer=False,
            index_max_seq=0,
            fp8_shuffled=weights.shuffled,
            head=True,
            vocab_rows=head.w_head.shape[0],
        )
        heads[W] = Glm53HeadOp(runtime(key), head)
    # FlyDSL's disk cache hands a rank the kernels another rank built (a lock per
    # kernel), so each rank starts at its own share
    start = len(runtimes) * rank // tp
    for rt in runtimes[start:] + runtimes[:start]:
        rt.precompile()
    draft = _Draft(
        model, layer, ops, heads, epochs, cos, sin, max_position=context_len - 1
    )
    _route_draft_steps(model, draft)
    logger.info(
        "monokernel %s: draft extend steps of %s rows and draft decode steps of up "
        "to %d rows run the NextN launches, routed by the spec's wrapper of the "
        "draft's forward; each key logs 'runs the MTP draft step' at its first launch",
        SPEC.name,
        extend_widths,
        max(decode_widths),
    )
    return draft


SPEC = MonoSpec(
    name="glm53_nextn_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=target.ROWS, tp_sizes=target.TP_SIZES, resident_blocks=BLOCKS
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    # the launcher checks the target's requirements; the draft-step wrapper checks
    # a draft step's attention backend against these
    requires=Requirements(attention_backends=DRAFT_ATTENTION_BACKENDS),
    numerics=target.NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    tokens_per_request=_tokens_per_request,
)
