# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 on gfx950: the monokernel spec and its forward (RFC
sgl-project/sglang#43949).

Every covered layer is one launch for the whole decoder layer (``gfx950/kernel.py``):
the input norm, q_a | kv_a, the q and kv norms, the cache writes, q_b, the indexer
(on the layers that run it) and its exact top-2048, the sparse MLA, o_proj and its
TP reduce, the post-attention norm, the router, the routed and shared FP8 experts
and the MoE TP reduce, adding both outputs to the hidden stream.

The dense layers 0 to 2 run the same launch with their FP8 SwiGLU MLP in place
of the MoE, on the experts' W8A8 path (per-128 FP8 activations, as SGLang's dense
FP8 linears). Layer 0 starts from the step's tokens: SGLang's embedding is not run.
After layer 77 the tail launch runs the final norm, the LM head and the logits'
all-gather; SGLang's final norm is not run, and its logits processor returns the
tail's logits (``_get_logits``) and stores the final norm's rows.

The forward is ``DeepseekV2ForCausalLM.forward`` for a routed decode or MTP verify
step (a verify step's rows are its requests' draft chains). A pad row (SGLang's
padding slot 0, or a width's row past the step) writes no cache and attends no key.

The kernels read SGLang's weights in place, except the copies logged at startup:
W_UK and W_UV packed from SGLang's per-tensor FP8 w_kc and w_vc, the router and the
indexer's weights_proj packed for the MFMA tiles, the indexer's k LayerNorm in f32,
one bf16 rotary table, and the dense layers' MLPs as expert-shaped slices.
"""

from __future__ import annotations

import importlib.util
import os

import msgspec
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
from sgl_amd_mono.glm53 import HF_CONFIG

# kernel widths; a step of fewer rows runs at the next width, the rest pad rows (a
# one-row launch never completes on gfx950)
WIDTHS = (2, 4, 8)
ROWS = tuple(range(1, max(WIDTHS) + 1))
TP_SIZES = (4, 8)
# the rows a step runs on the kernels at most at this TP size: on MI355X an 8-row
# launch at TP4 is slower than the model's forward, so wider steps run that
TP_MAX_ROWS = {4: 4}
# SGLang's DSA cache on ROCm: 64-token pages
PAGE_SIZE = 64
TOPK = 2048


def _width(rows: int) -> int:
    return next(w for w in WIDTHS if w >= rows)


# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "page_size",
        "enable_hisparse",
        "expert_distribution_recorder_mode",
        "kv_cache_dtype",
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
    ),
    "srt/distributed/parallel_state.py": ("cpu_group",),
    "srt/environ.py": (
        "SGLANG_DSA_FUSE_TOPK",
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
    ),
    "srt/configs/model_config.py": ("context_len",),
    "srt/layers/attention/dsa/dsa_indexer.py": ("use_gfx950_fused_indexer",),
    "srt/layers/attention/dsa_backend.py": ("forward_metadata", "real_page_table"),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": ("maybe_input_scattered",),
    "srt/layers/layer_boundary/residual/batch.py": ("start", "clear"),
    # the logits, replaced on a routed step by the tail's
    "srt/layers/logits_processor.py": (
        "_get_logits",
        "_get_logits(hidden_states)",
        "_get_logits(lm_head)",
        "_get_logits(logits_metadata)",
        "do_tensor_parallel_all_gather",
        "do_tensor_parallel_all_gather_dp_attn",
        "use_attn_tp_group",
        "use_fp32_lm_head",
        "use_fp32_lm_head_output",
        "logit_scale",
        "final_logit_softcapping",
    ),
    "srt/mem_cache/memory_pool.py": ("get_key_buffer", "get_index_k_with_scale_buffer"),
    "srt/model_executor/forward_batch_info.py": (
        "out_cache_loc",
        "seq_lens",
        "forward_mode",
        "is_idle",
    ),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_token_to_kv_pool",
    ),
    "srt/models/deepseek_v2.py": (
        "model",
        "embed_tokens",
        "layers",
        "start_layer",
        "end_layer",
        "norm",
        "self_attn",
        "input_layernorm",
        "indexer",
        "mlp",
        "logits_processor",
        "lm_head",
        "forward(prev_topk_indices)",
    ),
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "widths",
        "deadline_s",
        "copies",
    ),
    "srt/runtime_context.py": (
        "get_model",
        "get_parallel",
        "process_model_config",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
}

# Each covered layer against SGLang's normal layer on the same input and caches
# (SGLANG_AMD_MONO_CHECK): the stream it writes and the cache rows.
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "layer.streams.cosine": 0.95,
        "layer.kv.cosine": 0.995,
        "layer.index.cosine": 0.999,
        # an indexer layer selects every key of a row's own draft chain SGLang selects
        "layer.topk_own.max_abs": 0.0,
        # and exactly the top-k of its own scores (score descending, then position)
        "layer.topk_exact.max_abs": 0.0,
        # A dense layer's MLP (check.dense), every rank's shard summed: its mid against
        # an f32 gate | up on its own FP8 activations, its output against the f32 down
        # projection of its own per-128 FP8 mid, and its distance to an f32 MLP on its
        # FFN input over SGLang's FP8 MLP's on the same input, less one. 3x the largest
        # the serving check measured (TP8, FP8 KV, 1-8 rows, positions to 8.2K: 5.5e-5,
        # 9.2e-3, 0.036); a down or gate | up scale x2 on one rank fails them.
        # Its input, the post-attention norm of its FFN input, against SGLang's MLP's
        # input (both bf16): the dense launch's attention, o_proj and residual. 3x
        # the largest the serving check measured (0.029, layer 0 at 108K keys).
        "dense.mlp_in.rel": 0.09,
        "dense.mid.rel": 1.7e-4,
        "dense.x.rel": 2.8e-2,
        "dense.excess.rel": 0.11,
        # The tail's logits against SGLang's final norm, LM head and all-gather on the
        # same rows (serving max 5.6e-4), and against an f32 tail (serving max 2.5e-3,
        # SGLang's own distance alike); 3x those. The final norm x1.01 gives 9.2e-3.
        "head.logits.rel": 1.7e-3,
        "head.logits_vs_fp32.rel": 7.5e-3,
    },
)


def _covered_layers(config) -> list[int]:
    """Every layer: the dense ones below ``first_k_dense_replace``, then the MoE ones."""
    return list(range(config.num_hidden_layers))


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    reasons = [
        f"needs GLM-5.3's {key} = {want!r}, got {getattr(hf_config, key, None)!r}"
        for key, want in HF_CONFIG.items()
        if getattr(hf_config, key, None) != want
    ]
    if getattr(hf_config, "moe_layer_freq", 1) != 1:
        reasons.append("needs every layer after the dense ones to be MoE")
    # SGLang scales the softmax by YaRN's mscale; the kernel's scale is fixed
    rope = (
        getattr(hf_config, "rope_parameters", None)
        or getattr(hf_config, "rope_scaling", None)
        or {}
    )
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type != "default":
        reasons.append(f"RoPE {rope_type!r}: the kernel's softmax scale is unscaled")
    if cfg.page_size not in (None, PAGE_SIZE):
        reasons.append(f"--page-size {cfg.page_size}: the DSA caches need {PAGE_SIZE}")
    # a verify row sees the keys up to its own position: one draft chain per request
    if cfg.speculative_algorithm is not None and (
        cfg.speculative_eagle_topk != 1
        or cfg.speculative_num_draft_tokens != cfg.speculative_num_steps + 1
    ):
        reasons.append("MTP verifies one draft chain: topk 1, draft tokens = steps + 1")
    quant = getattr(hf_config, "quantization_config", None) or {}
    if quant.get("quant_method") != "fp8" or list(
        quant.get("weight_block_size") or ()
    ) != [128, 128]:
        reasons.append(
            "needs the FP8 block-128 checkpoint, got "
            f"{quant.get('quant_method')!r} {quant.get('weight_block_size')!r}"
        )
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _where(ids: list) -> str:
    return f"layer {ids[0]}" if len(ids) == 1 else f"{len(ids)} layers from {ids[0]}"


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.glm53.gfx950.weights import (
        dense_reason,
        embedding,
        experts_reason,
        head_weights,
        is_dense,
        unshuffled_projections,
    )

    p = get_parallel()
    reasons = []
    covered = _covered_layers(hf_config)
    row_major, unrotated = {}, []
    for i in covered:
        layer = model.model.layers[i]
        if is_dense(layer):
            why = dense_reason(layer, p.tp_size)
        else:
            why = experts_reason(layer.mlp)
        if why is not None:
            reasons.append(f"layer {i}: {why}")
        for name in unshuffled_projections(layer):
            row_major.setdefault(name, []).append(i)
        indexer = layer.self_attn.indexer
        if indexer is not None and not indexer.use_gfx950_fused_indexer:
            unrotated.append(i)
    reasons += [
        f"{name} of {_where(ids)}: needs AITER's (16, 16) FP8 layout, which SGLang "
        "applies for its AITER block-FP8 GEMM"
        for name, ids in row_major.items()
    ]
    if unrotated:
        reasons.append(
            f"the indexer of {_where(unrotated)}: needs SGLang's gfx950 fused DSA "
            "indexer, which caches rotated index keys"
        )
    if model.model.layers[covered[0]].self_attn.indexer is None:
        reasons.append(f"layer {covered[0]} reuses a top-k no layer computed")
    lp = model.logits_processor
    if (
        not lp.do_tensor_parallel_all_gather
        or lp.do_tensor_parallel_all_gather_dp_attn
        or lp.use_attn_tp_group
    ):
        reasons.append(
            "the tail all-gathers the logits over the TP group, as the logits "
            "processor does without DP attention or a DP LM head"
        )
    if lp.use_fp32_lm_head or lp.use_fp32_lm_head_output:
        reasons.append("the tail runs the LM head in bf16, not f32")
    for name, value in (
        ("logit_scale", lp.logit_scale),
        ("final_logit_softcapping", lp.final_logit_softcapping),
    ):
        if value:
            reasons.append(f"the tail does not apply the logits processor's {name}")
    for name, weights in (("LM head", head_weights), ("embedding", embedding)):
        try:
            weights(model, p.tp_size, p.tp_rank)
        except ValueError as e:
            reasons.append(f"{name}: {e}")
    return reasons


class _Layer:
    """One covered layer: its module, weights and per-width ops; its caches bind at
    its first launch and rebind if they move."""

    def __init__(self, i: int, module, weights):
        self.i, self.module, self.w = i, module, weights
        self.ops: dict = {}  # width -> Glm53LayerOp
        self.caches = None  # Glm53Caches


class _Built:
    """One rank's monokernels: the runtimes per (width, kind), each covered layer's
    ops, the tail's op per width, the step's rows per width, and the forward."""

    def __init__(
        self,
        lm,
        layers: dict,
        runtimes: dict,
        heads: dict,
        cos,
        sin,
        tp: int,
        device,
        context_len,
    ):
        from sgl_amd_mono.glm53.gfx950.runner import Glm53Step

        self.lm, self.model = lm, lm.model
        self._layers, self.runtimes, self.heads = layers, runtimes, heads
        self.max_rows = TP_MAX_ROWS.get(tp, max(WIDTHS))
        self.max_position = context_len - 1
        self.check_layers = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"
        self.cos, self.sin = cos, sin
        self.steps, self.ids = {}, {}
        i32, i64 = torch.int32, torch.int64
        for w in {w for w, _ in runtimes}:
            self.steps[w] = Glm53Step(
                positions=torch.zeros(w, dtype=i64, device=device),
                slots=torch.full((w,), -1, dtype=i64, device=device),
                indptr=torch.zeros(w + 1, dtype=i32, device=device),
                req_ids=torch.zeros(w, dtype=i32, device=device),
                page_table=torch.zeros(w, 1, dtype=i32, device=device),
                # the last entry takes the copies of a row's unused keys
                topk=torch.zeros(w * TOPK + 1, dtype=i32, device=device),
            )
            # layer 0's input: the step's token ids, -1 for a pad row
            self.ids[w] = torch.full((w,), -1, dtype=i32, device=device)
        # the logits processor returns the tail's logits on a routed step
        lp = lm.logits_processor
        self._own_logits, self._logits = lp._get_logits, None
        lp._get_logits = self._get_logits

    def _get_logits(self, hidden_states, lm_head, logits_metadata, *args, **kwargs):
        """The logits processor's logits: the tail's on a routed step (f32 [rows,
        vocab], as SGLang's LM head, all-gather and logits buffer leave them), else
        its own."""
        if self._logits is None:
            return self._own_logits(
                hidden_states, lm_head, logits_metadata, *args, **kwargs
            )
        return self._logits[: hidden_states.shape[0]]

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        """``DeepseekV2ForCausalLM.forward`` for a routed decode or verify step."""
        lm = self.lm
        if plan.key.rows > self.max_rows or input_embeds is not None:
            return type(lm).forward(
                lm, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            normed, self._logits = self._backbone(
                plan, input_ids, positions, forward_batch
            )
        try:
            # the logits processor stores the final norm's rows as SGLang's forward
            # hands them over
            return lm.logits_processor(
                input_ids, normed, lm.lm_head, forward_batch, None
            )
        finally:
            self._logits = None

    def _backbone(self, plan, input_ids, positions, forward_batch):
        """``DeepseekV2Model.forward`` with every layer a launch, layer 0 from the
        tokens, then the tail: the final norm's rows and the gathered logits."""
        from sglang.srt.layers.layer_boundary.residual.batch import clear, start

        start(forward_batch)
        S = plan.key.rows
        W = _width(S)
        step = self._step(plan, input_ids, positions, forward_batch, S, W)
        h = self.ids[W]
        for i in sorted(self._layers):
            h = self._layer(
                self._layers[i], plan, input_ids, positions, h, forward_batch, step
            )
        logits, normed = self.heads[W].launch(h)
        if self.check_layers and not torch.cuda.is_current_stream_capturing():
            from sgl_amd_mono.glm53 import check

            check.tail(self.lm, h[:S], logits[:S], normed[:S])
        for (w, _), rt in self.runtimes.items():
            if w == W:
                rt.advance_step()
        clear(forward_batch)
        return normed[:S], logits

    def _layer(self, L: _Layer, plan, input_ids, positions, h, forward_batch, step):
        """The previous launch's output (layer 0: the token ids) in; the launch's
        output out."""
        S = plan.key.rows
        W = _width(S)
        self._bind(L, kv_dtype=L.ops[W].rt.kv_dtype)
        checking = self.check_layers and not torch.cuda.is_current_stream_capturing()
        if checking:
            from sgl_amd_mono.glm53 import check

            x = self.model.embed_tokens(input_ids[:S]) if L.i == 0 else h[:S].clone()
            prev = None if L.w.index is not None else plan.state.get("ref_topk")
            ref = check.reference(L, positions, x, forward_batch, S, prev)
            if ref["topk"] is not None:
                plan.state["ref_topk"] = ref["topk"]
        out = L.ops[W].launch(h, L.caches, self.cos, self.sin, step)
        if checking:
            rt = L.ops[W].rt
            scores = check.launch_scores(rt, S) if L.w.index is not None else None
            check.compare(
                L, ref, out[:S], positions, forward_batch, S, step=step, scores=scores
            )
            if L.w.dense:
                check.dense(L, rt, ref, out[:S], S)
        return out

    @staticmethod
    def _bind(L: _Layer, kv_dtype: torch.dtype) -> None:
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        from sgl_amd_mono.glm53.gfx950.runner import (
            INDEX_PAGE_TOKENS,
            KV_ROW,
            Glm53Caches,
        )

        pool = get_token_to_kv_pool()
        kv = pool.get_key_buffer(L.i)
        # a layer that reuses the previous top-k keeps no index cache
        idx = pool.get_index_k_with_scale_buffer(L.i) if L.w.index is not None else None
        if (
            L.caches is not None
            and L.caches.kv.data_ptr() == kv.data_ptr()
            and (idx is None or L.caches.idx.data_ptr() == idx.data_ptr())
        ):
            return
        # an E4M3 row is SGLang's HIP DSA layout: 576 values, no block scales
        if kv.dtype != kv_dtype or kv.shape[-1] != KV_ROW or not kv.is_contiguous():
            raise RuntimeError(
                f"GLM-5.3 monokernel: layer {L.i}: MLA cache {kv.dtype} "
                f"{tuple(kv.shape)}; needs contiguous {kv_dtype} [slots, 1, {KV_ROW}]"
            )
        if idx is not None and (idx.dtype != torch.uint8 or not idx.is_contiguous()):
            raise RuntimeError(
                f"GLM-5.3 monokernel: layer {L.i}: index cache {idx.dtype} "
                f"{tuple(idx.shape)}; needs contiguous uint8"
            )
        L.caches = Glm53Caches(
            kv=kv.view(kv.shape[0], KV_ROW),
            idx=None if idx is None else idx.view(idx.shape[0], INDEX_PAGE_TOKENS, -1),
        )

    def _step(self, plan, input_ids, positions, forward_batch, S: int, W: int):
        """The step's rows for every launch (``Glm53Step``) and layer 0's token ids:
        built once a step, in the graph when captured."""
        if "step" not in plan.state:
            from sglang.srt.model_executor.forward_context import get_attn_backend

            st = self.steps[W]
            ids = self.ids[W]
            ids[:S] = input_ids[:S]
            ids[S:] = -1
            loc = forward_batch.out_cache_loc[:S].to(torch.int64)
            valid = loc != 0
            st.slots[:S] = torch.where(valid, loc, -1)
            st.slots[S:] = -1
            # verify rows past the context are lookahead the scheduler discards; the
            # rotary table and the indexer's score rows end at the context
            pos = positions[:S].to(torch.int64).clamp(max=self.max_position)
            st.positions[:S] = pos
            st.positions[S:] = 0
            # a row sees keys 0 ..= its position (in TARGET_VERIFY seq_lens holds
            # each request's prefix, not each row's keys)
            count = torch.where(valid, (pos + 1).clamp(max=TOPK), 0)
            # indptr[0] stays 0 from its allocation
            st.indptr[1 : S + 1] = torch.cumsum(count, 0, dtype=torch.int32)
            st.indptr[S + 1 :] = st.indptr[S]
            # the DSA backend repeats a verify request's page-table row for each row
            st.req_ids[:S] = torch.arange(S, dtype=torch.int32, device=loc.device)
            st.req_ids[S:] = 0
            md = get_attn_backend().forward_metadata
            plan.state["step"] = msgspec.structs.replace(
                st, page_table=md.real_page_table[:S]
            )
        return plan.state["step"]


def _build(model, hf_config, ctx) -> _Built:
    from sglang.srt.runtime_context import get_model, get_parallel, process_model_config

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.glm53.gfx950.layout import INDEX_KEYS_PER_TASK
    from sgl_amd_mono.glm53.gfx950.runner import (
        Glm53Build,
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

    p = get_parallel()
    tp, rank = p.tp_size, p.tp_rank
    device = model.model.layers[0].input_layernorm.weight.device
    copies = ctx.copies
    covered = _covered_layers(hf_config)
    layers = {
        i: _Layer(
            i,
            model.model.layers[i],
            layer_weights(model.model.layers[i], i, tp, copies),
        )
        for i in covered
    }
    emb = embedding(model, tp, rank)
    vocab_rows = emb.shape[0]
    head = head_weights(model, tp, rank)
    if head.w_head.shape[0] != vocab_rows:
        raise ValueError("GLM-5.3 monokernel: the LM head and embedding shards differ")
    # layer 0 reads its embedding rows where a layer reads its router
    layers[0].w = msgspec.structs.replace(layers[0].w, w_r=emb)
    cos, sin = rope_tables([model.model.layers[i] for i in covered], copies)
    shuffled = {L.w.shuffled for L in layers.values()}
    if len(shuffled) != 1:
        raise ValueError("GLM-5.3 monokernel: the layers' FP8 layouts differ")
    fp8_shuffled = shuffled.pop()
    dense = {L.w.dense for L in layers.values() if L.w.dense}
    if len(dense) > 1 or not layers[0].w.dense or layers[0].w.index is None:
        raise ValueError(
            "GLM-5.3 monokernel: needs layer 0 a dense indexer layer, every dense MLP "
            "in one slicing"
        )
    # layer 0 (from the tokens) has its own runtime; the other dense layers share one
    kinds = {
        "embed": [0],
        "dense": [i for i in covered if i and layers[i].w.dense],
        "index": [
            i
            for i in covered
            if not layers[i].w.dense and layers[i].w.index is not None
        ],
        "shared": [i for i in covered if layers[i].w.index is None],
    }
    if any(layers[i].w.index is None for i in kinds["dense"]):
        raise ValueError(
            "GLM-5.3 monokernel: needs the dense layers to run the indexer"
        )
    fault_site, fault_after = guard.fault_injection(rank) or (0, 0)
    kv_fp8 = get_model().kv_cache_dtype == "fp8_e4m3"
    context_len = process_model_config().context_len
    # the copied table is SGLang's at build; a longer served context reads past it
    if cos.shape[0] < context_len:
        raise ValueError(
            f"GLM-5.3 monokernel: the rotary table holds {cos.shape[0]} positions, "
            f"the server's context is {context_len}"
        )
    # the keys an indexer row scores at most: the served context's, in whole tasks
    tasks = -(-context_len // INDEX_KEYS_PER_TASK)
    max_seq = tasks * INDEX_KEYS_PER_TASK
    cap = TP_MAX_ROWS.get(tp, max(WIDTHS))
    widths = sorted({_width(r) for r in ctx.widths if r <= cap})
    if not widths:
        raise ValueError(
            f"GLM-5.3 monokernel: TP{tp} runs at most {cap} rows on the kernels, and "
            f"this server's decode graphs route {sorted(ctx.widths)}"
        )
    runtimes, heads = {}, {}

    def runtime(key):
        rt = Glm53Runtime(key, rank, p.tp_group.cpu_group, device)
        rt.set_deadline(ctx.deadline_s)
        return rt

    # each runtime's peer handshake is collective: every rank builds them in one order
    for w in widths:
        for kind, ids in kinds.items():
            if not ids:
                continue
            indexer = layers[ids[0]].w.index is not None
            key = Glm53Build(
                rows=w,
                tp=tp,
                launches_per_step=len(ids),
                with_indexer=indexer,
                # an unpaged launch keeps index_max_seq words in LDS
                index_max_seq=max_seq if indexer else 0,
                fp8_shuffled=fp8_shuffled,
                kv_fp8=kv_fp8,
                fault_site=fault_site,
                fault_after=fault_after,
                dense=layers[ids[0]].w.dense,
                tokens=kind == "embed",
                vocab_rows=vocab_rows if kind == "embed" else 0,
            )
            rt = runtime(key)
            runtimes[w, kind] = rt
            for n, i in enumerate(ids):
                layers[i].ops[w] = Glm53LayerOp(rt, layers[i].w, n)
        key = Glm53Build(
            rows=w,
            tp=tp,
            launches_per_step=1,
            with_indexer=False,
            index_max_seq=0,
            fp8_shuffled=fp8_shuffled,
            fault_site=fault_site,
            fault_after=fault_after,
            head=True,
            vocab_rows=vocab_rows,
        )
        runtimes[w, "head"] = rt = runtime(key)
        heads[w] = Glm53HeadOp(rt, head)
    # Every rank builds every width. FlyDSL's disk cache hands a rank the kernels
    # another rank built (a lock per kernel), so each rank starts at its own share.
    start = len(widths) * rank // tp
    for w in widths[start:] + widths[:start]:
        for (rw, _), rt in runtimes.items():
            if rw == w:
                rt.precompile()
    return _Built(
        model, layers, runtimes, heads, cos, sin, tp, device, context_len=context_len
    )


SPEC = MonoSpec(
    name="glm53_hip",
    devices={
        "gfx950": DeviceLimits(widths=ROWS, tp_sizes=TP_SIZES, resident_blocks=BLOCKS)
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"dsa"}),
        server_args={
            "kv_cache_dtype": frozenset({"auto", "bfloat16", "fp8_e4m3"}),
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # MTP: the NextN draft's verify steps (NEXTN resolves to EAGLE)
            "speculative_algorithm": frozenset({None, "EAGLE"}),
            # HiSparse maps the logical KV slots to device rows; the kernels write raw slots
            "enable_hisparse": frozenset({False}),
            # the launches record no expert distribution (SGLang's forward sets it per layer)
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        env={
            # layer 3 reads layer 2's top-k as KV slots; unfused top-k hands out positions
            "SGLANG_DSA_FUSE_TOPK": True,
            # benchmark routing overrides: the kernels route with the model's own router
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
