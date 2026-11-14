# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash's wide verify steps: ``DeepseekV4ForCausalLM.forward`` at 60 to 384
rows (``WIDTHS``), each layer in stages, every stage SGLang's own code in SGLang's order.

The forward is SGLang's ROCm path for a decode or verify step (``_forward_impl``,
``_forward_layers_hc_pre_from_prev``, ``forward_layer_fused_boundary``), with the attention
(``MQALayer.forward``) and the routed MoE (``DeepseekV2MoE.forward_normal``) opened into
their stages:

- ``seam_attn``: the previous layer's pending mHC post, this layer's attention pre and
  its input norm (``hc_boundary``, ``_input_norm``);
- ``attn_front``: the projections, norms, RoPE, the KV and compressor writes and the
  indexer (``_forward_prepare*``); ``attn_core``: the sparse attention;
- ``wo_a`` (the inverse RoPE and the grouped projection), ``wo_b`` (and the o-proj
  all-reduce, or its deferral into the mHC post);
- ``seam_ffn``: the attention's mHC post, the FFN pre and the post-attention norm;
- ``route`` (the router GEMM), ``moe_prep`` (the top-k), ``experts`` (aiter's sorted
  MoE), ``moe_out`` (the routed scale, the shared add and the MoE all-reduce);
- ``post``: the FFN mHC post, run, deferred to the next layer's seam, or fused with the
  all-reduce.

A step's outputs are bit for bit SGLang's forward on the same inputs. A kernel of this
package replaces one stage (or ``layer``, the whole layer) of ``Stages`` for the width
classes (``CLASSES``) it serves; every row count takes one class's stages, so each
(TP size, width) runs one path.

The classes of ``TOKEN_PARALLEL`` run token parallel (``TokenParallelStages``): rank r
owns rows r S / TP .. (r + 1) S / TP of an S-row step and keeps only their residual
streams. Each sublayer's all-reduce becomes a reduce-scatter to the rows' owners, the
row-local work (the mHC seams and their norms, the posts, Engram's gate, DSpark's
captures, the head's combine and norm) runs on the owner alone, and an all-gather brings
every rank the rows the next projection reads. The reduce-scatter sums an owner's rows
in the all-reduce's order (``Exchange``), and every row-local kernel computes a row the
same way at any row count, so each output is the same to the bit.

At TP 2 and TP 4 each layer's span A runs as one launch (``SpanAStages``, ``stages.spana``): a
layer whose post is deferred leaves its MoE output unsummed, and the next layer's launch sums
it over the ranks (the all-reduce's bits, or the reduce-scatter's on a token-parallel step),
runs the attention seam and the input norm and, on the classes of ``SPAN_A_FRONT``, the
attention front up to the attention core.

The forward reads what SGLang's backend builds for the step, except the low-ratio
indexers' decode schedules: the spec declares them unread (the monokernel scores every
index source itself), so ``prepare_step`` builds them here, in the graph, as the backend
would (``init_forward_metadata_in_graph``).
"""

from __future__ import annotations

import functools
from contextlib import contextmanager, nullcontext
from typing import Any, NamedTuple, Optional

import torch

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    # the step's model and its ROCm glue, read through their module namespaces
    "srt/models/deepseek_v4.py": (
        "_hip",
        "wo_b",
        "_is_gfx95_supported",
        "_FUSED_WO_A_MAX_TOKENS",
        "_wo_a_fp8_mxscale",
        "_wo_a_fp8_mxscale_fused_invrope",
        "_freqs_cis_to_cos_sin",
        "_apply_wo_a_bf16_matmul",
        "fused_rope_inplace",
        "fused_rope_wo_a_bf16",
        "Mxfp8SwizzledInput",
        "envs",
        "get_is_capture_mode",
        "is_in_breakable_cuda_graph",
        "is_cp_active",
        "get_forward",
        "get_token_to_kv_pool",
        "get_global_expert_distribution_recorder",
        "check_cuda_graph_backend",
        "bcg_deepseek_v4_attention_with_output",
        "bcg_deepseek_v4_engram_hash_ids",
        "layer_id",
        "Phase",
        "Backend",
        "AuxHiddenStatePacker",
        "mhc",
        "embed_tokens",
        "hc_mult",
        "engram_hasher",
        "engram",
        "dspark_layers_to_capture",
        "start_layer",
        "end_layer",
        "norm",
        "capture_aux_hidden_states",
        "prepare_language_model_inputs",
        "logits_processor",
        "lm_head",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "hc_cfg",
        "hc_post",
        "_input_norm",
        "maybe_use_decode_attn_tp",
        "alt_streams",
        "_multi_stream_bs_limit",
        "compressor",
        "compress_ratio",
        "use_mega_attention",
        "is_dsv41",
        "q_head_norm",
        "n_local_heads",
        "n_local_groups",
        "o_lora_rank",
        "head_dim",
        "qk_rope_head_dim",
        "freqs_cis",
        "attn_mqa",
        "wo_a",
        "wo_a_fp8",
        "use_fused_wo_a",
        "use_flashinfer_mxfp8_wo_b",
        "_kernel_num_heads",
        "_zero_padded_q",
        "_local_attn_sink",
        "_forward_prepare",
        "_forward_prepare_low_ratio_multi_stream",
        "_forward_prepare_multi_stream_hip",
        "_project_wo_b",
        "hc_sinkhorn_iters",
        "rms_norm_eps",
        "hc_eps",
        "input_layernorm",
        "post_attention_layernorm",
        "fuse_wqa_wkv",
        "wqkv_a",
        "wq_b",
        "q_norm",
        "kv_norm",
        "eps",
        "indexer",
        "_low_ratio_sources_break",
        "bcg_deepseek_v4_low_ratio_sources",
    ),
    "srt/models/deepseek_common/amd/deepseek_v4_hip.py": (
        "skip_head_pad",
        "attention_inv_rope",
        "attention_inv_rope(wo_a_applies_inv_rope)",
        "wo_a_emits_fp8_grid",
        "wo_b_emits_mxfp8",
        "fused_rmsnorm_fake_quant",
        "_wqkv_a_native_consumer_checked",
    ),
    "srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py": (
        "hc_boundary",
        "attention_mhc_fusion",
        "moe_mhc_fusion",
        "apply_attention_mhc",
        "apply_moe_mhc",
        "_gfx95_dense_post_attention_norm",
    ),
    "srt/models/deepseek_v4_mhc.py": (
        "HcState",
        "HcPending",
        "AttnOutput",
        "materialized",
        "with_residual",
        "residual",
        "streams",
        "pre",
        "post",
        "comb",
    ),
    "srt/layers/engram.py": (
        "new_hash_ids",
        "layer_hash_index",
        # a token-parallel step's gate on the owner's rows (``TokenParallelStages.engram``)
        "embed_rows",
        "wkv",
        "engram_gate",
        "q_weight",
        "k_weight",
        "eps",
        "clamp_value",
        "image_token_id",
    ),
    # the token-parallel exchange
    "srt/runtime_context.py": (
        "get_parallel",
        '"tp_size"',
        '"tp_group"',
        '"attn_tp_size"',
        '"attn_tp_group"',
        '"attn_dp_size"',
    ),
    "srt/distributed/parallel_state.py": ("ca_comm", "world_size", "rank_in_group"),
    "srt/layers/linear.py": ("RowParallelLinear.forward(skip_all_reduce)", "use_decode_attn_tp"),
    # the operand the next projection takes, rebuilt from the gathered norm
    "kernels/ops/quantization/mxfp8_amd_gfx95.py": (
        "mxfp8_operand",
        "Fp8GridActivation",
        "Mxfp8Activation",
        "fake_quant_fp8_activation",
        "fp8_grid_quantize",
    ),
    "srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py": (
        "_hip_mxfp8_operand",
        "_ffn_norm_emits_mxfp8",
        "_mxfp8_consumer",
        "_wqkv_a_native_consumer",
        "Mxfp8Activation",
        "Fp8GridActivation",
    ),
    "srt/layers/radix_attention.py": ("v_head_dim", "layer_id"),
    # the fused boundary's coefficients
    "kernels/ops/layernorm/mhc_boundary_hip.py": ("tensors",),
    "srt/models/deepseek_v2.py": (
        "_hip_moe",
        "_use_aiter",
        "_is_hip",
        "MoEOutput",
        "KTEPWrapperMethod",
        "vision_topk",
        "maybe_fuse_routed_scale_and_shared_add",
        "post_experts_all_reduce",
        "get_is_capture_mode",
        "is_in_breakable_cuda_graph",
        "_enable_a2a_moe",
        "_can_dual_stream_graph",
        "forward_normal",
        "alt_stream",
        "num_fused_shared_experts",
        "shared_experts",
        "experts",
        "topk",
        "gate",
        "e_score_correction_bias_vl",
        "_fuse_shared_experts_inside_sbo",
        "_maybe_quant_moe_input_once",
        "_forward_shared_experts",
        "routed_scaling_factor",
        "routed_scaling_factor_fused_in_topk",
        "_shared_expert_tp1",
        "_adds_replicated_output",
        "reduce_results",
        "is_hash",
    ),
    "srt/models/deepseek_common/amd/deepseek_v2_hip_moe.py": (
        "forward_gate",
        "forward_gate(fused_gate)",
        "batch_has_images",
    ),
    "srt/models/deepseek_common/moe_transport.py": (
        "MoeRowTransport",
        "MoeRowTransport.__init__(cp_return_contiguous)",
        "scope",
        "return_output",
        "do_shared_local",
        "_use_cp",
        "_use_tp_moe_gather",
        "_use_tp_attn_a2a_scatter",
    ),
    "srt/layers/moe/mega_moe.py": ("should_use_mega_moe",),
    "kernels/ops/layernorm/mhc.py": ("hc_broadcast", "mhc_combine", "mhc_stream_mean"),
    "srt/layers/layer_boundary/adapters/attention.py": (
        "get_attn_tp_context",
        "maybe_input_scattered",
    ),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    # the low-ratio indexers' decode schedules (``prepare_step``)
    "srt/layers/attention/dsv4/low_ratio_backend_hip.py": (
        "build_low_ratio_decode_workspaces",
        "build_low_ratio_decode_workspaces(max_context_len)",
        "build_low_ratio_decode_workspaces(num_requests)",
        "build_low_ratio_decode_workspaces(page8_ratios)",
        "low_ratio_decode_rows_are_identity",
    ),
    "srt/layers/attention/deepseek_v4_backend_hip_radix.py": (
        "DSV4Metadata",
        "forward_metadata",
        "core_metadata",
        "low_ratios",
        "low_ratio_indexer_metadata_by_ratio",
        "fp4_low_ratio_decode_workspaces",
        "max_context_len",
        "token_to_kv_pool",
        # the union core (``UnionStages``)
        "softmax_scale",
        "swa_page_indices",
        "swa_topk_lengths",
        "sparse_page_indices",
        "sparse_topk_lengths",
        "get_swa_out_cache_loc",
        "forward_low_ratio_sources",
    ),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "low_ratio_index_k_page8_ratios",
        # the union core's caches (``UnionStages``)
        "get_swa_key_buffer_radix",
        "get_extra_key_buffer",
        "get_extra_key_page_size",
        # the span A launch's KV store (``SpanA``)
        "get_swa_raw_buffer",
        "uniform_fp8",
        "kv_layout",
        "swa_page_size",
    ),
    # the span A launch: its seam, its front's routes and the sources after it
    "srt/layers/layernorm.py": ("variance_epsilon",),
    "srt/layers/quantization/fp8.py": (
        "Fp8LinearMethod",
        "w8a8_mxfp8_linear",
        "block_fp8_as_mxfp8",
        "mxfp8_dense_backend",
        "block_fp8_mxfp8_ready",
    ),
    "srt/layers/quantization/fp8_utils.py": ("is_gfx95_mxfp8_aiter",),
    "srt/layers/quantization/fp8_hip.py": (
        "apply_dense",
        '"weight_scale_mx_e8m0"',
        "mxfp8_aiter_preshuffled",
    ),
    # the dense FP8 projections' route GEMM (``WideFp8``)
    "kernels/ops/gemm/mxfp8_aiter_gfx95.py": (
        "aiter_mxfp8_blockscaled_linear(input)",
        "aiter_mxfp8_blockscaled_linear(weight_scale_ue8m0)",
        "aiter_mxfp8_blockscaled_linear(input_scale)",
        "aiter_mxfp8_blockscaled_linear(weight_preshuffled)",
    ),
    "srt/layers/attention/dsv4/unified_kv_gate.py": ("is_unified_kv_triton",),
    "srt/environ.py": ("SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE",),
}

# the wide steps' rows: SGLang's speculative decode graph sizes from 10 requests on
# (``generate_decode_cuda_graph_batch_sizes``) at DSpark's 6 rows a request, so a wide step
# pads to the graph stock SGLang runs it in
WIDTHS = tuple(6 * b for b in (*range(10, 33, 2), *range(40, 65, 4)))
# the width classes a kernel's stages are built for: each serves the rows up to it
CLASSES = (96, 192, 384)


def width_class(rows: int) -> int:
    """The class of a wide step's rows (``CLASSES``)."""
    return next(c for c in CLASSES if rows <= c)


class Front(NamedTuple):
    """The attention front's outputs ``attn_core`` and ``wo_a`` read
    (``MQALayer.forward``'s locals)."""

    q: torch.Tensor  # the query the core reads (padded heads where the kernel needs them)
    k: torch.Tensor  # the KV the core stores (the query itself where the front stored it)
    save_kv_cache: bool
    sink: torch.Tensor
    tp_slice: slice  # this rank's heads of the core's output
    inv_rope: Optional[tuple]  # the core applies the inverse RoPE itself
    dtype: torch.dtype  # the core's output, the layer input's


class Shared:
    """What the stages of every width class share, built once a process before any graph
    capture: the model, the TP size, the spec's layers (``spec._Layer`` by layer) and each
    kernel's resources (``once``)."""

    def __init__(self, lm, layers: Optional[dict] = None):
        from sglang.srt.runtime_context import get_parallel

        self.model, self.layers = lm.model, layers or {}
        self.tp, self.device = get_parallel().tp_size, next(lm.parameters()).device
        self.heads = self.model.layers[self.model.start_layer].self_attn.n_local_heads
        self._made: dict = {}

    def once(self, key, make):
        """The resource ``key``, made by ``make()`` at its first use (in class order: a
        collective one is made in the same order on every rank)."""
        got = self._made.get(key)
        if got is None:
            got = self._made[key] = make()
        return got


class Stages:
    """One layer's stages, SGLang's own code in SGLang's order (the module docstring). A
    kernel's stages are a subclass whose ``__init__(shared, c)`` (``Shared``, the width class)
    calls its base's; ``mixins`` composes them per (TP size, class)."""

    def __init__(self, shared: Optional[Shared] = None, c: int = 0):
        pass

    # ---- the attention side ----

    def seam_attn(self, layer, state, positions, forward_batch):
        """(residual, x, x_quant, coefficients): the pending post (or the settled
        residual) and this layer's attention pre, then the input norm."""
        from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
            hc_boundary,
        )
        from sglang.srt.models.deepseek_v4_mhc import HcPending

        if isinstance(state.streams, HcPending):
            pending = tuple(state.streams)
        else:
            pending = (None, state.streams, None, None)
        residual, x, co = hc_boundary(
            layer, *pending, state.pre, layer.hc_attn_fn, layer.hc_attn_scale,
            layer.hc_attn_base,
        )  # fmt: skip
        # the boundary's reduce + sinkhorn rides in the norm launch
        x, x_quant = layer._input_norm(x, allow_aiter_quant=False, coefficients=co)
        return residual, x, x_quant, co

    def attn_front(self, attn, x, positions, forward_batch, x_quant) -> Front:
        """``MQALayer.forward`` up to its attention core, on ROCm's paged SWA cache."""
        import sglang.srt.models.deepseek_v4 as dv4
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        hip, M = dv4._hip, x.shape[0]
        multi_stream = (
            dv4.envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and attn.alt_streams is not None
            and dv4.get_is_capture_mode()
            and (dv4.is_in_breakable_cuda_graph() or M <= attn._multi_stream_bs_limit)
            and not dv4.is_cp_active(forward_batch)
            and attn.compressor is not None
            and attn.compress_ratio not in (1, 2)
        )
        mode = forward_batch.forward_mode
        low_ratio_multi_stream = (
            dv4._is_gfx95_supported
            and attn.compress_ratio in (1, 2)
            and attn.alt_streams is not None
            and (mode.is_decode() or mode.is_target_verify())
        )
        tp_slice, q_padded, q_out = slice(None), None, None
        if (
            attn.is_dsv41
            and not attn.q_head_norm
            and dv4.get_token_to_kv_pool().uniform_fp8
            and attn.n_local_heads in (8, 16, 32, 64, 128)
        ):
            heads = attn.n_local_heads
            q_out = torch.empty(
                (M, attn.n_local_heads, attn.head_dim),
                dtype=torch.float8_e4m3fn,
                device=x.device,
            )
        else:
            heads = (
                attn.n_local_heads
                if hip.skip_head_pad(attn)
                else attn._kernel_num_heads(M)
            )
            if heads != attn.n_local_heads:
                # V4.1's kernels read every padded head: the padding is zero
                q_padded = attn._zero_padded_q(x, heads, backend)
                tp_slice = slice(0, attn.n_local_heads)
                q_out = q_padded[:, tp_slice, :]
        sink = attn._local_attn_sink(heads)
        if multi_stream:
            q = attn._forward_prepare_multi_stream_hip(
                x, positions, forward_batch, backend, q_out, x_quant=x_quant
            )
            kv = None
        elif low_ratio_multi_stream:
            q = attn._forward_prepare_low_ratio_multi_stream(
                x, positions, forward_batch, backend, q_out, x_quant=x_quant
            )
            kv = None
        else:
            q, kv = attn._forward_prepare(
                x, positions, forward_batch, backend, q_out, x_quant=x_quant,
                q_rope_out=None, k_nope_out=None, k_rope_out=None,
            )  # fmt: skip
        inv_rope = hip.attention_inv_rope(
            attn,
            positions,
            forward_batch,
            False,
            wo_a_applies_inv_rope=attn.wo_a_fp8
            and dv4._wo_a_fp8_mxscale_fused_invrope is not None,
        )
        # the front stored the KV where it returned none: the core's k is a placeholder
        return Front(
            q=q_padded if q_padded is not None else q,
            k=kv if kv is not None else q,
            save_kv_cache=False,
            sink=sink,
            tp_slice=tp_slice,
            inv_rope=inv_rope,
            dtype=x.dtype,
        )

    def attn_core(self, attn, front: Front, forward_batch) -> torch.Tensor:
        """The sparse attention over the window and the top-k keys: this rank's heads."""
        import sglang.srt.models.deepseek_v4 as dv4
        from sglang.srt.model_executor.forward_context import get_attn_backend

        if forward_batch.forward_mode.is_extend() and dv4.is_in_breakable_cuda_graph():
            # a breakable graph runs the core eagerly at replay
            o = front.q.new_empty(
                (*front.q.shape[:-1], attn.attn_mqa.v_head_dim), dtype=front.dtype
            )
            dv4.bcg_deepseek_v4_attention_with_output(
                front.q, front.k, o, attn.attn_mqa.layer_id, attn.compress_ratio,
                front.sink, front.save_kv_cache,
            )  # fmt: skip
            return o[:, front.tp_slice, :]
        o = get_attn_backend().forward(
            q=front.q,
            k=front.k,
            v=front.k,
            layer=attn.attn_mqa,
            forward_batch=forward_batch,
            compress_ratio=attn.compress_ratio,
            attn_sink=front.sink,
            save_kv_cache=front.save_kv_cache,
            **({} if front.inv_rope is None else {"inv_rope": front.inv_rope}),
        )
        return o[:, front.tp_slice, :]

    def wo_a(self, attn, o, positions, forward_batch, front: Front):
        """The inverse RoPE and the grouped wo_a: wo_b's input."""
        import sglang.srt.models.deepseek_v4 as dv4

        hip, G = dv4._hip, attn.n_local_groups
        if attn.wo_a_fp8 and dv4._wo_a_fp8_mxscale_fused_invrope is not None:
            # the inverse RoPE and the per-group MXFP8 quant in one aiter launch, then the
            # a8w8 mxscale GEMM
            cos_c, sin_c = dv4._freqs_cis_to_cos_sin(attn.freqs_cis, o.dtype, o.device)
            return dv4._wo_a_fp8_mxscale_fused_invrope(
                o, positions, cos_c, sin_c, G,
                attn.wo_a.weight.view(G, attn.o_lora_rank, -1),
                attn.wo_a.weight_scale_inv.data,
            )  # fmt: skip
        fuse_mxfp8_quant = (
            attn.use_flashinfer_mxfp8_wo_b and not dv4.get_forward().sp_active
        )
        fuse_rope_wo_a = (
            attn.use_fused_wo_a and 0 < o.shape[0] <= dv4._FUSED_WO_A_MAX_TOKENS
        )
        if front.inv_rope is None and not fuse_rope_wo_a:
            dv4.fused_rope_inplace(
                o[..., -attn.qk_rope_head_dim :],
                None,
                attn.freqs_cis,
                positions=positions,
                inverse=True,
            )
        o = o.view(o.shape[0], G, -1)
        if fuse_rope_wo_a:
            out = dv4.fused_rope_wo_a_bf16(
                o,
                attn.wo_a.weight.view(G, attn.o_lora_rank, -1),
                torch.view_as_real(attn.freqs_cis).flatten(1),
                positions,
                out_mxfp8=fuse_mxfp8_quant,
            )
            return dv4.Mxfp8SwizzledInput(*out) if fuse_mxfp8_quant else out[0]
        if attn.wo_a_fp8:
            # gfx950's aiter e8m0 block-scale batched GEMM, quantizing per token group
            T, G, D = o.shape
            return dv4._wo_a_fp8_mxscale(
                o,
                attn.wo_a.weight.view(G, attn.o_lora_rank, D),
                attn.wo_a.weight_scale_inv.data,
            )
        mode = forward_batch.forward_mode
        return dv4._apply_wo_a_bf16_matmul(
            o,
            attn.wo_a.weight.view(G, attn.o_lora_rank, -1),
            is_decode=mode.is_decode(),
            is_target_verify=mode.is_target_verify(),
            is_prefill=mode.is_extend_without_speculative(),
            fast_path=attn.is_dsv41,
            fuse_mxfp8_quant=fuse_mxfp8_quant,
            fp8_grid=hip.wo_a_emits_fp8_grid(attn),
            emit_fp8=hip.wo_b_emits_mxfp8(attn, o.shape[0]),
        )

    def wo_b(self, attn, o, defer_all_reduce: bool):
        """wo_b and its all-reduce; deferred (an ``AttnOutput`` partial) where the mHC post
        takes it."""
        return attn._project_wo_b(o, defer_all_reduce)

    # ---- the FFN side ----

    def seam_ffn(self, layer, x, residual, co, fusion, forward_batch):
        """(residual, x, coefficients): the attention's mHC post (with its all-reduce
        where ``fusion``), the FFN pre and the post-attention norm."""
        from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
            _gfx95_dense_post_attention_norm,
            apply_attention_mhc,
            hc_boundary,
        )

        if fusion is not None:
            residual, x = apply_attention_mhc(x.partial, fusion), None
        residual, x, ffn_co = hc_boundary(
            layer, x, residual,
            co.post if x is not None else None, co.comb if x is not None else None,
            co.pre, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
        )  # fmt: skip
        x = _gfx95_dense_post_attention_norm(layer, x, ffn_co)
        return residual, x, ffn_co

    def route(self, moe, x, zero_allocator, use_vision_topk):
        """(router logits, the split-K router's partials or None)."""
        import sglang.srt.models.deepseek_v2 as dv2

        if dv2._is_hip:
            return dv2._hip_moe.forward_gate(
                moe, x, zero_allocator, fused_gate=not use_vision_topk
            )
        return moe.gate(x, zero_allocator), None

    def moe_prep(self, moe, x, logits, partials, input_ids_global, use_vision_topk):
        """The top-k experts and weights of each row (the fused shared expert among them)."""
        import sglang.srt.models.deepseek_v2 as dv2

        kwargs = {"input_ids": input_ids_global} if getattr(moe, "is_hash", False) else {}
        if partials is not None:
            kwargs["router_logits_partials"] = partials
        if use_vision_topk:
            return dv2.vision_topk(moe, logits, input_ids_global)
        return moe.topk(x, logits, **kwargs)

    def experts(self, moe, x, topk_output, pre_quant_input):
        """The routed experts: aiter's sort, quant and MoE GEMMs."""
        if pre_quant_input is not None:
            return moe.experts(x, topk_output, pre_quant_input=pre_quant_input)
        return moe.experts(x, topk_output)

    def moe_out(self, moe, routed, shared, return_moe_output):
        """The routed scale, the shared add and the MoE all-reduce; the pieces unmerged
        (``MoEOutput``) where the mHC post takes the all-reduce."""
        import sglang.srt.models.deepseek_v2 as dv2

        if return_moe_output:
            return dv2.MoEOutput(
                routed=routed,
                shared=shared,
                experts=moe.experts,
                routed_scaling_factor=moe.routed_scaling_factor,
                shared_is_replicated=moe._shared_expert_tp1,
            )
        out = dv2.maybe_fuse_routed_scale_and_shared_add(
            moe.experts,
            routed,
            None if moe._shared_expert_tp1 else shared,
            moe.routed_scaling_factor,
        )
        if moe.reduce_results:
            out = dv2.post_experts_all_reduce(out)
        if shared is not None and moe._shared_expert_tp1 and moe._adds_replicated_output():
            out += shared
        return out

    def post(self, layer, x, residual, ffn_co, fusion, defer_post):
        """The layer's (residual or pending post, pre)."""
        from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
            apply_moe_mhc,
        )
        from sglang.srt.models.deepseek_v4_mhc import HcPending, HcState

        pre, post, comb = ffn_co.tensors()
        if fusion is not None:
            # the post carries the reduce: the merged rows without one
            return HcState(apply_moe_mhc(x.get_merged(), fusion), pre)
        if defer_post:
            return HcState(HcPending(x, residual, post, comb), pre)
        return HcState(layer.hc_post(x, residual, post, comb), pre)

    # ---- one layer ----

    def layer(self, wide, i, state, positions, input_ids, forward_batch, input_ids_global,
              defer_post):  # fmt: skip
        """Layer i through the fused mHC boundary
        (``forward_hc_pre_from_prev_fused_boundary``)."""
        from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
            attention_mhc_fusion,
            moe_mhc_fusion,
        )

        layer = wide.model.layers[i]
        attn = layer.self_attn
        residual, x, x_quant, co = self.seam_attn(layer, state, positions, forward_batch)
        with attn.maybe_use_decode_attn_tp(forward_batch):
            fusion = attention_mhc_fusion(layer, residual, co, forward_batch)
            front = self.attn_front(attn, x, positions, forward_batch, x_quant)
            o = self.attn_core(attn, front, forward_batch)
            o = self.wo_a(attn, o, positions, forward_batch, front)
            x = self.wo_b(attn, o, fusion is not None)
        residual, x, ffn_co = self.seam_ffn(layer, x, residual, co, fusion, forward_batch)
        fusion = moe_mhc_fusion(layer, residual, ffn_co, forward_batch)
        x = self.ffn(layer, x, forward_batch, input_ids, input_ids_global, fusion is not None)
        return self.post(layer, x, residual, ffn_co, fusion, defer_post)

    def ffn(self, layer, x, forward_batch, input_ids, input_ids_global, return_moe_output):
        """``_run_moe_ffn_dp_sync``: the rows' transport around the MoE."""
        from sglang.srt.models.deepseek_common.moe_transport import MoeRowTransport

        prepared = MoeRowTransport(
            layer.mlp, x, forward_batch, input_ids, input_ids_global,
            cp_return_contiguous=layer.config.model_type == "deepseek_v41",
        )  # fmt: skip
        with prepared.scope():
            return_moe_output = (
                return_moe_output
                and not prepared._use_cp
                and not prepared._use_tp_moe_gather
                and not prepared._use_tp_attn_a2a_scatter
            )
            out = self.moe(
                layer.mlp, prepared.input, forward_batch,
                input_ids_global=prepared.input_ids_global,
                skip_shared_experts=prepared.do_shared_local,
                return_moe_output=return_moe_output,
                input_ids=prepared.input_ids,
            )  # fmt: skip
        prepared.input = None
        return prepared.return_output(out)

    def moe(self, moe, x, forward_batch, *, input_ids_global, skip_shared_experts,
            return_moe_output, input_ids):  # fmt: skip
        """``DeepseekV2MoE.forward``: its ``forward_normal`` in stages; the other routes
        (mega MoE, a2a, the dual-stream graph) SGLang's whole."""
        import sglang.srt.models.deepseek_v2 as dv2

        use_vision_topk = moe.gate.e_score_correction_bias_vl is not None
        if use_vision_topk and dv2._is_hip:
            use_vision_topk = dv2._hip_moe.batch_has_images(forward_batch)
        if self.whole_moe(moe, x):
            return moe.forward(
                x, forward_batch, input_ids=input_ids,
                input_ids_global=input_ids_global,
                skip_shared_experts=skip_shared_experts,
                return_moe_output=return_moe_output,
            )  # fmt: skip
        defer_shared = not moe.experts.moe_runner_config.inplace
        pre_quant = None if skip_shared_experts else moe._maybe_quant_moe_input_once(x)
        shared = None
        if not defer_shared and not skip_shared_experts:
            shared = moe._forward_shared_experts(x, None, pre_quant_input=pre_quant)
        logits, partials = self.route(moe, x, None, use_vision_topk)
        topk_output = self.moe_prep(
            moe, x, logits, partials, input_ids_global, use_vision_topk
        )
        routed = self.experts(moe, x, topk_output, pre_quant)
        if not (dv2._use_aiter and moe.routed_scaling_factor_fused_in_topk) or (
            isinstance(moe.experts.quant_method, dv2.KTEPWrapperMethod)
        ):
            routed *= moe.routed_scaling_factor
        if defer_shared and not skip_shared_experts:
            shared = moe._forward_shared_experts(x, None, pre_quant_input=pre_quant)
        return self.moe_out(moe, routed, shared, return_moe_output)

    def whole_moe(self, moe, x) -> bool:
        """Whether ``moe`` takes one of the routes SGLang runs whole."""
        import sglang.srt.models.deepseek_v2 as dv2
        from sglang.srt.layers.moe.mega_moe import should_use_mega_moe

        return (
            should_use_mega_moe(moe, x)
            or moe._enable_a2a_moe
            or moe._can_dual_stream_graph(x)
            or (
                moe.alt_stream is not None
                and moe.num_fused_shared_experts == 0
                and x.shape[0] > 0
                and dv2.get_is_capture_mode()
                and not dv2.is_in_breakable_cuda_graph()
            )
            or moe._fuse_shared_experts_inside_sbo
            or x.shape[0] == 0
        )

    # ---- the step's rows (all of them on every rank) ----

    def own(self, rows: int) -> slice:
        """The rows of an ``rows``-row step whose residual streams this rank keeps."""
        return slice(0, rows)

    def engram(self, engram, residual, hash_ids, forward_batch, input_ids):
        """Engram's gated value added to the residual streams."""
        return engram(
            residual, hash_ids, forward_batch, input_ids=input_ids, cp_all_tokens=False
        )

    def gather(self, x):
        """Every row of a row-local result (each rank holds them all)."""
        return x


class Exchange:
    """A TP group's token-parallel collectives, on its custom all-reduce's registered
    buffers (aiter's ``CustomAllreduce``). Rank r owns rows r S / TP .. (r + 1) S / TP.

    The reduce-scatter is the custom all-reduce's own first half: each owner sums its rows
    over the ranks' partials in fp32, in the order the all-reduce sums them (its own
    partial first, then ranks r + 1, r + 2, ... modulo TP; at TP 2 the order of two terms
    does not change an fp32 sum), and rounds once to bf16. So an owner's rows are bit for
    bit the all-reduce's. The all-gather copies each owner's rows to every rank."""

    def __init__(self, group):
        self.ca, self.world, self.rank = group.ca_comm, group.world_size, group.rank_in_group

    def own(self, rows: int) -> slice:
        n = rows // self.world
        return slice(self.rank * n, (self.rank + 1) * n)

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's rows of the sum of every rank's ``x``."""
        assert self.ca.should_custom_rs(x, 0), (x.shape, x.dtype)
        out = x.new_empty((x.shape[0] // self.world, *x.shape[1:]))
        self.ca.custom_reduce_scatter(x, out)
        return out

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Every owner's rows of ``x``, in row order."""
        out = self.ca.custom_all_gather(x.contiguous())
        assert out is not None, (x.shape, x.dtype)
        return out


def regrid(like, x: torch.Tensor):
    """The fp8-grid operand ``like`` (a norm launch's, ``Fp8GridActivation`` or
    ``Mxfp8Activation``) rebuilt for the rows ``x`` of that norm: the norm quantizes its
    own bf16 output, per 32 values, by the rule of ``fake_quant_fp8_activation``, so the
    same rule on the bf16 rows gives the same bits."""
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        Fp8GridActivation,
        Mxfp8Activation,
        fake_quant_fp8_activation,
        fp8_grid_quantize,
    )

    if like is None:
        return None
    if isinstance(like, Fp8GridActivation):
        return Fp8GridActivation(fake_quant_fp8_activation(x))
    assert isinstance(like, Mxfp8Activation), type(like)
    return Mxfp8Activation(*fp8_grid_quantize(x))


class TokenParallelStages(Stages):
    """``Stages`` token parallel (the module docstring): the residual streams of this rank's
    rows only. Per layer:

    - ``seam_attn`` on the owner's rows, then the all-gather of its input norm (bf16; the
      projections' fp8-grid operand rebuilt from it on every rank);
    - the attention on every row, as ``Stages``; ``wo_b`` leaves its sum to the exchange;
    - the reduce-scatter of wo_b's partials, ``seam_ffn`` on the owner's rows, the
      all-gather of the MoE input;
    - the MoE on every row, as ``Stages``, but for its all-reduce: the reduce-scatter of
      its merged output, and ``post`` on the owner's rows.
    """

    def __init__(self, shared: Shared, c: int):
        from sglang.srt.runtime_context import get_parallel

        super().__init__(shared, c)
        self.x = shared.once("exchange", lambda: Exchange(get_parallel().tp_group))

    def own(self, rows: int) -> slice:
        return self.x.own(rows)

    def gather(self, x):
        return self.x.all_gather(x)

    def layer(self, wide, i, state, positions, input_ids, forward_batch, input_ids_global,
              defer_post):  # fmt: skip
        layer = wide.model.layers[i]
        attn = layer.self_attn
        residual, x, x_quant, co = self.seam_attn(layer, state, positions, forward_batch)
        x, x_quant = self.gather_norm(x, x_quant)
        with attn.maybe_use_decode_attn_tp(forward_batch):
            front = self.attn_front(attn, x, positions, forward_batch, x_quant)
            o = self.attn_core(attn, front, forward_batch)
            o = self.wo_a(attn, o, positions, forward_batch, front)
            x = self.wo_b(attn, o, False)
        residual, x, ffn_co = self.seam_ffn(layer, x, residual, co, None, forward_batch)
        x = self.ffn(layer, x, forward_batch, input_ids, input_ids_global, False)
        return self.post(layer, x, residual, ffn_co, None, defer_post)

    def gather_norm(self, x, x_quant):
        """Every row of the input norm and of the projections' operand made with it."""
        x = self.x.all_gather(x)
        return x, regrid(x_quant, x)

    def wo_b(self, attn, o, defer_all_reduce: bool):
        """wo_b's partial, reduce-scattered: this rank's rows of the attention output."""
        assert not defer_all_reduce and not attn.wo_b.use_decode_attn_tp
        partial, _ = attn.wo_b(
            o.flatten(1) if isinstance(o, torch.Tensor) else o, skip_all_reduce=True
        )
        return self.x.reduce_scatter(partial)

    def ffn(self, layer, x, forward_batch, input_ids, input_ids_global, return_moe_output):
        """Every row of the MoE input to the MoE; this rank's rows of its output."""
        from sglang.srt.models.deepseek_common.moe_transport import MoeRowTransport

        rows = self.x.all_gather(x)
        operand = getattr(x, "_hip_mxfp8_operand", None)
        if operand is not None:
            rows._hip_mxfp8_operand = regrid(operand, rows)
        prepared = MoeRowTransport(
            layer.mlp, rows, forward_batch, input_ids, input_ids_global,
            cp_return_contiguous=layer.config.model_type == "deepseek_v41",
        )  # fmt: skip
        assert not (
            prepared._use_cp
            or prepared._use_tp_moe_gather
            or prepared._use_tp_attn_a2a_scatter
        ), "a token-parallel step moves no MoE rows of its own"
        with prepared.scope():
            out = self.moe(
                layer.mlp, prepared.input, forward_batch,
                input_ids_global=prepared.input_ids_global,
                skip_shared_experts=prepared.do_shared_local,
                return_moe_output=False,
                input_ids=prepared.input_ids,
            )  # fmt: skip
        prepared.input = None
        return self.x.reduce_scatter(prepared.return_output(out))

    def whole_moe(self, moe, x) -> bool:
        # SGLang's whole routes sum over the ranks themselves
        assert not super().whole_moe(moe, x), "a token-parallel MoE runs in stages"
        return False

    def moe_out(self, moe, routed, shared, return_moe_output):
        """The routed scale and the shared add; the sum over the ranks is ``ffn``'s."""
        import sglang.srt.models.deepseek_v2 as dv2

        assert not return_moe_output and moe.reduce_results and not moe._shared_expert_tp1
        return dv2.maybe_fuse_routed_scale_and_shared_add(
            moe.experts, routed, shared, moe.routed_scaling_factor
        )

    def engram(self, engram, residual, hash_ids, forward_batch, input_ids):
        """``Engram.forward`` with the gate on this rank's rows: the table lookup is a
        collective over every row, and the projection runs on every row too, so the
        owner's rows of it are the replicated forward's."""
        from sglang.srt.layers.engram import engram_gate

        emb = engram.embed_rows(hash_ids, forward_batch, False)
        kv, _ = engram.wkv(emb.flatten(-2))
        own = self.own(kv.shape[0])
        image_ids = input_ids[own] if engram.image_token_id is not None else None
        return engram_gate(
            residual, kv[own], engram.q_weight, engram.k_weight, engram.eps,
            engram.clamp_value, image_ids, engram.image_token_id,
        )  # fmt: skip


# the TP sizes and width classes whose steps run token parallel (``TokenParallelStages``):
# from 240 rows the owners' share of the seams saves more than the second collective of each
# sublayer and the operand rebuild cost; at 96 and 192 rows it does not (served, both TPs)
TOKEN_PARALLEL = {2: (384,), 4: (384,)}


# ---------------------------------------------------------------- span A in one launch
# the TP sizes the span A launch serves, and per TP the width classes whose launch also runs
# the attention front (wqkv_a .. wq_b); the others end it at the seam
SPAN_A_TPS = (2, 4)
SPAN_A_FRONT = {2: (), 4: ()}
# per TP the sum's pacing window (``stages.spana.Plan.window``): the early rows' sums land
# first, so the seam overlaps the link's later rows
SPAN_A_WINDOW = {2: 40, 4: 80}
# the token-parallel classes whose launch runs the front on the owners' rows and gathers its
# compact outputs (the q norm's operand, the KV record) to every rank
SPAN_A_GATHER = {2: (384,), 4: (384,)}
# per TP the width classes whose launch sums the previous MoE's output; the others take
# SGLang's all-reduce or reduce-scatter (at TP 4 the launch's sum is no faster served)
SPAN_A_SUM = {2: (96, 192, 384), 4: ()}


class _Owed(NamedTuple):
    """A sublayer output owed its sum over the ranks: this rank's partial, which the next
    layer's span launch sums (``SpanA``)."""

    x: torch.Tensor


def _fp8_route_why(linear) -> Optional[str]:
    """Why ``linear`` is not on SGLang's gfx950 aiter MXFP8 route over its plain [N, K] e4m3
    weight (the route whose GEMM the span launch runs), or None."""
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    method = getattr(linear, "quant_method", None)
    if not (
        isinstance(method, Fp8LinearMethod)
        and method.block_fp8_as_mxfp8
        and getattr(linear, "block_fp8_mxfp8_ready", False)
        and method.mxfp8_dense_backend.is_gfx95_mxfp8_aiter()
    ):
        return "SGLang's gfx950 aiter MXFP8 route"
    if getattr(linear, "mxfp8_aiter_preshuffled", False) or linear.bias is not None:
        return "the plain [N, K] e4m3 weight and no bias"
    return None


def _front_why(layer, pool) -> Optional[str]:
    """Why the span launch cannot run ``layer``'s attention front as SGLang's route does, or
    None: the fused wqkv_a and wq_b on aiter's MXFP8 GEMM fed by the fake-quant norms, the
    V4-layout SWA cache with 256-token pages, the query roped by the K store launch."""
    import sglang.srt.models.deepseek_common.amd.deepseek_v4_gfx95_dense as dense
    import sglang.srt.models.deepseek_common.amd.deepseek_v4_hip as hip
    import sglang.srt.models.deepseek_v4 as dv4
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton

    from sgl_amd_mono.deepseek_v41.gfx950.stages import spana

    attn = layer.self_attn
    if not (attn.fuse_wqa_wkv and attn.fused_rmsnorm_fake_quant):
        return "the fused wqkv_a and the fake-quant q norm"
    for lin in (attn.wqkv_a, attn.wq_b):
        why = _fp8_route_why(lin)
        if why is not None:
            return why
    if not (dense._mxfp8_consumer(attn.wqkv_a) and dense._mxfp8_consumer(attn.wq_b)):
        return "MXFP8 operands into wqkv_a and wq_b"
    if attn.q_head_norm or attn.use_mega_attention or not attn.is_dsv41:
        return "V4.1's query without a head norm, outside the mega attention"
    if not hip.skip_head_pad(attn):
        return "the real head count in the attention core"
    if attn.compress_ratio not in (0, 1, 2) or attn.head_dim != spana.HD:
        return "a V4.1 layer (ratio 0, 1 or 2; 512-wide heads)"
    if is_unified_kv_triton() or pool.uniform_fp8 or dv4.envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.get():
        return "the plain V4 SWA store"
    if str(pool.kv_layout).split(".")[-1] not in ("V4", "0") or pool.swa_page_size != 1 << spana.PAGE_BITS:
        return "the V4 layout on 256-token pages"
    return None


class Mixes(NamedTuple):
    """A boundary's mixes as computed tensors (``HcCoefficients``' readers)."""

    pre: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor

    def tensors(self):
        return self.pre, self.post, self.comb


class _FrontOut(NamedTuple):
    """What the span launch's front left for ``attn_front``: the roped query and the q norm's
    MXFP8 operand (the indexer reads it)."""

    q: torch.Tensor
    q_quant: Any


class SpanA:
    """The span launch's resources (``stages.spana``), shared by the width classes: a width's
    plans (by the seam's form) and scratch, each layer's constants as the kernel reads them,
    and the TP group's peer buffer, where each rank's MoE partial reaches the ranks that sum
    it. A width's sum is the all-reduce's (one-shot at TP 2, reduce-scatter then all-gather at
    TP 4), or, for the token-parallel classes (``TOKEN_PARALLEL``), the reduce-scatter's."""

    def __init__(self, model, device):
        from sglang.srt.runtime_context import get_parallel

        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.deepseek_v41.gfx950.stages import seam_wide, spana

        parallel = get_parallel()
        group = parallel.tp_group
        self.tp, self.rank, self.device = group.world_size, group.rank_in_group, device
        assert self.tp in SPAN_A_TPS, self.tp
        token_parallel = TOKEN_PARALLEL.get(self.tp, ())
        self.modes = {
            m: "rs" if width_class(m) in token_parallel else ("oneshot" if self.tp == 2 else "rsag")
            for m in WIDTHS
        }
        self.gathers = {
            m: self.modes[m] == "rs" and width_class(m) in SPAN_A_GATHER.get(self.tp, ())
            for m in WIDTHS
        }
        self.sums = {m: width_class(m) in SPAN_A_SUM.get(self.tp, ()) for m in WIDTHS}
        # the peer buffer's flag regions: a width's sum, and a gather's for each shape a width's
        # plans take (after a deferred post, with or without the sum; without one, after a
        # settled post or at the first layer)
        keyed = {}
        for m in WIDTHS:
            for key in self._shapes(m):
                _, owed, post, combine = key
                keyed[key] = spana.Plan(
                    m=m, seam=seam_wide.Form(stage="seama", post=post, combine=combine), tp=self.tp,
                    mode=self.modes[m] if owed else None, tpar=self.modes[m] == "rs",
                    front=self._stand_in_front(m) if self.gathers[m] else None,
                    gather=self.gathers[m],
                )  # fmt: skip
        nbytes, plans = spana.peer_layout(self.tp, keyed)
        self.peers = {key: p.peer for key, p in plans.items()}
        # collective: every rank builds it here, in the same order
        self.peer = PeerBuffer(nbytes, group.cpu_group, self.rank, self.tp, device)
        self.peer.bytes[spana.flag_region(self.tp) :].zero_()
        self.scratch: dict = {}
        self.consts: dict = {}
        self.front_consts: dict = {}
        self._gather_plan: dict = {}

    def _shapes(self, m):
        """The flag-region keys of width m: (m, with a sum, the seam's post, its combine)."""
        if not self.gathers[m]:
            return [(m, True, True, True)] if self.sums[m] else []
        return [(m, self.sums[m], True, True), (m, False, False, True), (m, False, False, False)]

    def _stand_in_front(self, m):
        """A front of the width's shape (the peer layout reads only its presence)."""
        from sgl_amd_mono.deepseek_v41.gfx950.stages import spana

        tq, tb = spana.front_tiles(m // self.tp if self.modes[m] == "rs" else m, self.tp)
        return spana.Front(qkv=tq, wqb=tb, heads=64 // self.tp, q_eps=0.0, kv_eps=0.0)

    def _layer(self, layer):
        i = layer.self_attn.layer_id
        got = self.consts.get(i)
        if got is None:
            norm = layer.input_layernorm
            got = self.consts[i] = (
                layer.hc_attn_fn.data.float().contiguous(),
                layer.hc_attn_scale.data.float().contiguous(),
                layer.hc_attn_base.data.float().contiguous(),
                norm.weight.data.contiguous(),
            )
        return got

    def _front(self, layer, pool):
        from sgl_amd_mono.deepseek_v41.gfx950.stages import fp8_wide, spana

        i = layer.self_attn.layer_id
        got = self.front_consts.get(i)
        if got is None:
            why = _front_why(layer, pool)
            if why is not None:
                raise RuntimeError(f"layer {i}: the wide span A launch needs {why}")
            attn = layer.self_attn
            ws = []
            for lin in (attn.wqkv_a, attn.wq_b):
                n, k = lin.weight.shape
                assert lin.weight.dtype == torch.float8_e4m3fn and lin.weight.is_contiguous()
                ws += [lin.weight, fp8_wide.pack_scales(lin.weight_scale_mx_e8m0, n, k)]
            assert ws[0].shape == (spana.QKV, spana.H) and ws[2].shape[1] == spana.QR
            freqs = torch.view_as_real(attn.freqs_cis).flatten(-2).float().contiguous()
            got = self.front_consts[i] = (
                *ws,
                attn.q_norm.weight.data.contiguous(),
                attn.kv_norm.weight.data.contiguous(),
                freqs,
                (attn.q_norm.variance_epsilon, attn.eps, attn.n_local_heads),
            )
        return got

    def plan(self, layer, m, x, pre, front):
        """The launch's plan for layer ``layer`` at m rows: the seam's form (an owed post or
        none, a previous pre or none; the norm's quant as SGLang's route picks it), the sum
        (with an owed partial) and the front (``front``: its constants)."""
        import sglang.srt.models.deepseek_common.amd.deepseek_v4_gfx95_dense as dense
        import msgspec

        from sgl_amd_mono.deepseek_v41.gfx950.stages import seam_wide, spana

        if not layer._wqkv_a_native_consumer_checked:
            layer._wqkv_a_native_consumer = dense._mxfp8_consumer(
                layer.self_attn.wqkv_a if layer.self_attn.fuse_wqa_wkv else None
            )
            layer._wqkv_a_native_consumer_checked = True
        owed = isinstance(x, _Owed)
        form = seam_wide.Form(
            post=x is not None,
            combine=pre is not None,
            quant="mxfp8" if layer._wqkv_a_native_consumer else "grid",
            stage="seama",
            rms_eps=layer.rms_norm_eps,
            norm_eps=layer.input_layernorm.variance_epsilon,
            hc_eps=layer.hc_eps,
            iters=layer.hc_sinkhorn_iters,
        )
        mode = self.modes[m] if owed else None
        fr = None
        tpar = self.modes[m] == "rs"
        if front is not None:
            q_eps, kv_eps, heads = front[-1]
            tq, tb = spana.front_tiles(m // self.tp if tpar else m, self.tp)
            fr = spana.Front(qkv=tq, wqb=tb, heads=heads, q_eps=q_eps, kv_eps=kv_eps, variant=FRONT_VARIANT)
        gather = fr is not None and self.gathers[m]
        p = spana.Plan(
            m=m, seam=form, tp=self.tp, mode=mode, front=fr, tpar=tpar, gather=gather,
            window=SPAN_A_WINDOW[self.tp] if mode is not None else 0,
        )  # fmt: skip
        if mode is not None or gather:
            key = (m, mode is not None, form.post, form.combine) if gather else (m, True, True, True)
            p = msgspec.structs.replace(p, peer=self.peers[key])
            # a gather's flags in the peer buffer flip once a launch of their one plan, a region a
            # plan shape (the sum's carry the width's own phase, ``Plan.peer.sown``)
            if gather and self._gather_plan.setdefault(key, p) != p:
                raise RuntimeError(
                    f"layer {layer.self_attn.layer_id}: the span A launch's gather needs one plan a width: "
                    f"{self._gather_plan[key]} and {p}"
                )
        return p

    def __call__(self, layer, state, positions, forward_batch, with_front):
        """Layer ``layer``'s span A for the step's rows: the owed MoE partial summed over the
        ranks (an ``_Owed`` post), its mHC seam and input norm, and with ``with_front`` the
        attention front. Returns (new residual streams, normed rows, their operand, ``Mixes``,
        the ``_FrontOut`` or None)."""
        import sglang.srt.models.deepseek_common.amd.deepseek_v4_gfx95_dense as dense
        import sglang.srt.models.deepseek_v4 as dv4
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.models.deepseek_v4_mhc import HcPending

        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.deepseek_v41.gfx950.stages import seam_wide, spana

        if isinstance(state.streams, HcPending):
            x, residual, post, comb = state.streams
        else:
            x, residual, post, comb = None, state.streams, None, None
        m = positions.shape[0]
        attn = layer.self_attn
        pool = dv4.get_token_to_kv_pool() if with_front else None
        front = self._front(layer, pool) if with_front else None
        p = self.plan(layer, m, x, state.pre, front)
        n, dev = p.seam_rows, residual.device
        fn, scale, base, norm_w = self._layer(layer)
        residual = residual.contiguous()
        assert residual.shape == (n, seam_wide.HC, seam_wide.H), (residual.shape, p)
        got = self.scratch.get(p)
        if got is None:
            scr = PeerBuffer(spana.scratch_bytes(p), None, self.rank, 1, dev)
            lo, nb = spana.flag_bytes(p)
            scr.bytes[lo : lo + nb].zero_()
            got = self.scratch[p] = (scr, torch.zeros(spana.work_bytes(p), dtype=torch.uint8, device=dev))
        scr, work = got
        lay = spana.scratch_layout(p)

        def view(name, dtype, shape):
            o, nb = lay[name]
            return scr.bytes[o : o + nb].view(dtype)[: torch.Size(shape).numel()].view(shape)

        res_out = torch.empty_like(residual) if x is not None else residual
        normed = torch.empty((n, seam_wide.H), dtype=residual.dtype, device=dev)
        q = qs = None
        if front is not None:
            q, qs = view("xq", torch.float8_e4m3fn, (n, seam_wide.H)), view("xs", torch.uint8, (n, seam_wide.H // 32))
        elif p.seam.quant == "mxfp8":
            q = torch.empty((n, seam_wide.H), dtype=torch.float8_e4m3fn, device=dev)
            qs = torch.empty((n, seam_wide.H // 32), dtype=torch.uint8, device=dev)
        else:
            q = torch.empty((n, seam_wide.H), dtype=residual.dtype, device=dev)
        mixes = Mixes(
            torch.empty((n, seam_wide.HC), dtype=torch.float32, device=dev),
            torch.empty((n, seam_wide.HC), dtype=torch.float32, device=dev),
            torch.empty((n, seam_wide.HC, seam_wide.HC), dtype=torch.float32, device=dev),
        )
        x_in = x.x if isinstance(x, _Owed) else x
        if x_in is not None:
            x_in = x_in.contiguous()
            assert x_in.shape == (m if p.mode is not None else n, seam_wide.H), (x_in.shape, p)

        def ptr(t):
            return 0 if t is None else t.data_ptr()

        fargs = (0,) * 11 + (work.data_ptr(),)
        out = None
        if front is not None:
            wqkv, wqkv_s, wqb, wqb_s, qn_w, kvn_w, freqs, _ = front
            backend = get_attn_backend()
            loc = backend.get_swa_out_cache_loc(forward_batch)
            cache = pool.get_swa_raw_buffer(attn.layer_id)
            assert loc.dtype == torch.int32 and positions.dtype == torch.int64, (loc.dtype, positions.dtype)
            assert cache.dtype == torch.uint8 and cache.stride(0) == spana.KV_PAGE, (cache.shape, cache.stride())
            q_out = torch.empty((p.q_rows, p.front.heads, spana.HD), dtype=residual.dtype, device=dev)
            fargs = (
                wqkv.data_ptr(), wqkv_s.data_ptr(), qn_w.data_ptr(), kvn_w.data_ptr(),
                wqb.data_ptr(), wqb_s.data_ptr(), freqs.data_ptr(), positions.contiguous().data_ptr(),
                loc.contiguous().data_ptr(), cache.data_ptr(), q_out.data_ptr(), work.data_ptr(),
            )  # fmt: skip
            if p.gather:
                g, pb = p.peer, self.peer.bytes
                qq = pb[g.gq : g.gq + m * spana.QR].view(torch.float8_e4m3fn).view(m, spana.QR)
                qqs = pb[g.gqs : g.gqs + m * spana.QR // 32].view(m, spana.QR // 32)
            else:
                qq = view("qq", torch.float8_e4m3fn, (n, spana.QR))
                qqs = view("qs", torch.uint8, (n, spana.QR // 32))
            out = _FrontOut(q_out, dense.Mxfp8Activation(qq, qqs))
        spana.build(p)(
            ptr(x_in), scr.bytes.data_ptr(), self.peer.addresses.data_ptr(), self.peer.local,
            self.rank, residual.data_ptr(), ptr(post), ptr(comb), ptr(state.pre), fn.data_ptr(),
            scale.data_ptr(), base.data_ptr(), norm_w.data_ptr(), res_out.data_ptr(),
            normed.data_ptr(), ptr(q), ptr(qs), mixes.pre.data_ptr(), mixes.post.data_ptr(),
            mixes.comb.data_ptr(), *fargs, stream=torch.cuda.current_stream(),
        )  # fmt: skip
        if p.seam.quant == "mxfp8":
            x_quant = dense.Mxfp8Activation(q, qs)
        else:
            x_quant = dense.Fp8GridActivation(q)
        return res_out, normed, x_quant, mixes, out


# the rounding variant of the KV row's norm and RoPE that gives the K store launch's bits
FRONT_VARIANT = 0


class SpanAStages(Stages):
    """``Stages`` with span A in one launch a layer (``SpanA``): a layer whose post is
    deferred leaves its MoE output unsummed (``_Owed``), and the next layer's launch sums it
    over the ranks, runs its mHC seam and input norm, and on the classes of ``SPAN_A_FRONT``
    its attention front up to the attention core. Every stage gives SGLang's bits."""

    def __init__(self, shared: Shared, c: int):
        super().__init__(shared, c)
        self.span = shared.once("span", lambda: SpanA(shared.model, shared.device))
        self.with_front = c in SPAN_A_FRONT.get(shared.tp, ())
        self.fold = c in SPAN_A_SUM.get(shared.tp, ())
        self._front_out = None

    def seam_attn(self, layer, state, positions, forward_batch):
        residual, x, x_quant, mixes, out = self.span(
            layer, state, positions, forward_batch, self.with_front
        )
        self._front_out = out
        return residual, x, x_quant, mixes

    def attn_front(self, attn, x, positions, forward_batch, x_quant) -> Front:
        import sglang.srt.models.deepseek_common.amd.deepseek_v4_hip as hip
        import sglang.srt.models.deepseek_v4 as dv4
        from sglang.srt.model_executor.forward_context import get_attn_backend

        out, self._front_out = self._front_out, None
        if out is None:
            return super().attn_front(attn, x, positions, forward_batch, x_quant)
        backend = get_attn_backend()
        if attn.compressor is not None or attn.indexer is not None:
            if dv4._low_ratio_sources_break(backend, forward_batch):
                dv4.bcg_deepseek_v4_low_ratio_sources(attn, x, out.q_quant, positions)
            else:
                backend.forward_low_ratio_sources(
                    layer=attn, x=x, q_lora=out.q_quant, positions=positions,
                    forward_batch=forward_batch,
                )  # fmt: skip
        inv_rope = hip.attention_inv_rope(
            attn,
            positions,
            forward_batch,
            False,
            wo_a_applies_inv_rope=attn.wo_a_fp8
            and dv4._wo_a_fp8_mxscale_fused_invrope is not None,
        )
        return Front(
            q=out.q,
            k=out.q,
            save_kv_cache=False,
            sink=attn._local_attn_sink(attn.n_local_heads),
            tp_slice=slice(None),
            inv_rope=inv_rope,
            dtype=x.dtype,
        )

    def moe_out(self, moe, routed, shared, return_moe_output):
        import sglang.srt.models.deepseek_v2 as dv2

        if not self._owe:
            return super().moe_out(moe, routed, shared, return_moe_output)
        # the routed scale and the shared add; the sum over the ranks is the next launch's
        assert not return_moe_output and moe.reduce_results and not moe._shared_expert_tp1
        return dv2.maybe_fuse_routed_scale_and_shared_add(
            moe.experts, routed, shared, moe.routed_scaling_factor
        )

    def ffn(self, layer, x, forward_batch, input_ids, input_ids_global, return_moe_output):
        out = super().ffn(layer, x, forward_batch, input_ids, input_ids_global, return_moe_output)
        return _Owed(out) if self._owe else out

    def whole_moe(self, moe, x) -> bool:
        # SGLang's whole routes sum over the ranks themselves
        assert not (self._owe and super().whole_moe(moe, x)), "an owed MoE runs in stages"
        return super().whole_moe(moe, x)

    _owe = False

    def layer(self, wide, i, state, positions, input_ids, forward_batch, input_ids_global,
              defer_post):  # fmt: skip
        # a deferred post leaves the MoE's sum to the next layer's launch
        self._owe = defer_post and self.fold
        try:
            return super().layer(
                wide, i, state, positions, input_ids, forward_batch, input_ids_global,
                defer_post,
            )  # fmt: skip
        finally:
            self._owe = False


class TokenParallelSpanA(SpanAStages, TokenParallelStages):
    """``TokenParallelStages`` with ``SpanAStages``' launch: the launch reduce-scatters the
    owed MoE partial to the rows' owners and runs the seam on them; the input norm's rows
    are gathered as before."""

    def __init__(self, shared: Shared, c: int):
        super().__init__(shared, c)
        # the launch's front runs on the owners' rows of the gathering classes
        self.with_front = c in SPAN_A_GATHER.get(shared.tp, ())
        self._needs_x = False

    def seam_attn(self, layer, state, positions, forward_batch):
        attn = layer.self_attn
        self._needs_x = attn.compressor is not None or attn.indexer is not None
        return SpanAStages.seam_attn(self, layer, state, positions, forward_batch)

    def gather_norm(self, x, x_quant):
        """With the launch's gather the front is done on every row: only an index or KV
        source layer's compressor and indexer still read the bf16 input norm of every row."""
        if self._front_out is None:
            return TokenParallelStages.gather_norm(self, x, x_quant)
        return (self.x.all_gather(x) if self._needs_x else x), None

    def layer(self, wide, i, state, positions, input_ids, forward_batch, input_ids_global,
              defer_post):  # fmt: skip
        self._owe = defer_post and self.fold
        try:
            return TokenParallelStages.layer(
                self, wide, i, state, positions, input_ids, forward_batch, input_ids_global,
                defer_post,
            )  # fmt: skip
        finally:
            self._owe = False

    def ffn(self, layer, x, forward_batch, input_ids, input_ids_global, return_moe_output):
        """Every row of the MoE input to the MoE; this rank's rows of its output, or with a
        deferred post every row of its partial, owed (the next launch reduce-scatters it)."""
        from sglang.srt.models.deepseek_common.moe_transport import MoeRowTransport

        rows = self.x.all_gather(x)
        operand = getattr(x, "_hip_mxfp8_operand", None)
        if operand is not None:
            rows._hip_mxfp8_operand = regrid(operand, rows)
        prepared = MoeRowTransport(
            layer.mlp, rows, forward_batch, input_ids, input_ids_global,
            cp_return_contiguous=layer.config.model_type == "deepseek_v41",
        )  # fmt: skip
        assert not (
            prepared._use_cp
            or prepared._use_tp_moe_gather
            or prepared._use_tp_attn_a2a_scatter
        ), "a token-parallel step moves no MoE rows of its own"
        with prepared.scope():
            out = self.moe(
                layer.mlp, prepared.input, forward_batch,
                input_ids_global=prepared.input_ids_global,
                skip_shared_experts=prepared.do_shared_local,
                return_moe_output=False,
                input_ids=prepared.input_ids,
            )  # fmt: skip
        prepared.input = None
        out = prepared.return_output(out)
        return _Owed(out) if self._owe else self.x.reduce_scatter(out)

    def moe_out(self, moe, routed, shared, return_moe_output):
        return TokenParallelStages.moe_out(self, moe, routed, shared, return_moe_output)

    def whole_moe(self, moe, x) -> bool:
        return TokenParallelStages.whole_moe(self, moe, x)


class SeamStages(Stages):
    """``Stages`` with both mHC seams as one FlyDSL launch each (``stages.seam_wide``):
    SGLang's boundary and norm launches, bit for bit, the norm's quant as SGLang's route
    picks it for the layer. Where the span A launch runs the attention seam
    (``SpanAStages``, ahead of this class) the FFN seam is this class's."""

    def __init__(self, shared: Shared, c: int):
        super().__init__(shared, c)
        self.seam = shared.once("seam", lambda: _SeamResources(shared))

    def _seam(self, layer, kind, x, residual, post, comb, pre, quant):
        """One launch: (new streams, normed rows, quant (q, scale) or (q,) or None, Mixes)."""
        r = self.seam
        sw = r.sw
        fn, scale, base, w, norm_eps = r.consts(layer, kind)
        m = residual.shape[0]
        form = sw.Form(
            stage="seama" if kind == "attn" else "seamb",
            post=x is not None,
            combine=pre is not None,
            quant=quant,
            rms_eps=layer.rms_norm_eps,
            norm_eps=norm_eps,
            hc_eps=layer.hc_eps,
            iters=layer.hc_sinkhorn_iters,
        )
        dev = residual.device
        res_out = torch.empty_like(residual) if x is not None else residual
        normed = torch.empty((m, sw.H), dtype=residual.dtype, device=dev)
        q = qs = None
        if quant == "mxfp8":
            q = torch.empty((m, sw.H), dtype=torch.float8_e4m3fn, device=dev)
            qs = torch.empty((m, sw.H // 32), dtype=torch.uint8, device=dev)
        elif quant == "grid":
            q = torch.empty((m, sw.H), dtype=residual.dtype, device=dev)
        mixes = Mixes(
            torch.empty((m, sw.HC), dtype=torch.float32, device=dev),
            torch.empty((m, sw.HC), dtype=torch.float32, device=dev),
            torch.empty((m, sw.HC, sw.HC), dtype=torch.float32, device=dev),
        )
        if x is not None:
            x = x.contiguous()

        def ptr(t):
            return 0 if t is None else t.data_ptr()

        sw.standalone(m, form)(
            ptr(x), ptr(residual), ptr(post), ptr(comb), ptr(pre), ptr(fn), ptr(scale),
            ptr(base), ptr(w), ptr(res_out if x is not None else None), ptr(normed), ptr(q),
            ptr(qs), ptr(mixes.pre), ptr(mixes.post), ptr(mixes.comb), ptr(r.scratch),
            ptr(r.flags[m]), stream=torch.cuda.current_stream(),
        )  # fmt: skip
        return res_out, normed, (q, qs), mixes

    def seam_attn(self, layer, state, positions, forward_batch):
        import sglang.srt.models.deepseek_common.amd.deepseek_v4_gfx95_dense as dense
        from sglang.srt.models.deepseek_v4_mhc import HcPending

        if isinstance(state.streams, HcPending):
            x, residual, post, comb = state.streams
        else:
            x, residual, post, comb = None, state.streams, None, None
        # the gfx950 fake-quant input norm (``input_norm_fake_quant``): its operand fp8 +
        # ue8m0 where wqkv_a consumes them, else on the fp8 grid as bf16
        if not layer._wqkv_a_native_consumer_checked:
            layer._wqkv_a_native_consumer = dense._mxfp8_consumer(
                layer.self_attn.wqkv_a if layer.self_attn.fuse_wqa_wkv else None
            )
            layer._wqkv_a_native_consumer_checked = True
        quant = "mxfp8" if layer._wqkv_a_native_consumer else "grid"
        residual, x, (q, qs), mixes = self._seam(
            layer, "attn", x, residual, post, comb, state.pre, quant
        )
        x_quant = dense.Mxfp8Activation(q, qs) if quant == "mxfp8" else dense.Fp8GridActivation(q)
        return residual, x, x_quant, mixes

    def seam_ffn(self, layer, x, residual, co, fusion, forward_batch):
        import sglang.srt.models.deepseek_common.amd.deepseek_v4_gfx95_dense as dense

        assert fusion is None, "the all-reduce + post fusion runs at 8 rows at most"
        mx = dense._ffn_norm_emits_mxfp8(layer, x.shape[0])
        residual, x, (q, qs), mixes = self._seam(
            layer, "ffn", x, residual, co.post, co.comb, co.pre, "mxfp8" if mx else None
        )
        if mx:
            # the shared expert's gate_up takes it in place of its own quant
            x._hip_mxfp8_operand = dense.Mxfp8Activation(q, qs)
        return residual, x, mixes


class _SeamResources:
    """The seam launches' scratch (one for every launch: they run in stream order), their
    flags a row count (every wide width and a token-parallel class's rows a rank), zeros once,
    and each layer's constants as the kernel reads them."""

    def __init__(self, shared: Shared):
        from sgl_amd_mono.deepseek_v41.gfx950.stages import seam_wide as sw

        token_parallel = TOKEN_PARALLEL.get(shared.tp, ())
        rows = set(WIDTHS) | {m // shared.tp for m in WIDTHS if width_class(m) in token_parallel}
        dev = shared.device
        self.sw = sw
        self.scratch = torch.zeros(sw.scratch_bytes(max(rows)), dtype=torch.uint8, device=dev)
        self.flags = {m: torch.zeros(sw.flag_words(m), dtype=torch.int32, device=dev) for m in sorted(rows)}
        self._consts: dict = {}

    def consts(self, layer, kind):
        """(fn, scale, base, norm weight, norm eps) of the layer's ``kind`` boundary, as the
        kernel reads them (f32, f32, f32, bf16; contiguous)."""
        key = (id(layer), kind)
        got = self._consts.get(key)
        if got is None:
            fn, scale, base = (
                (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base)
                if kind == "attn"
                else (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base)
            )
            norm = layer.input_layernorm if kind == "attn" else layer.post_attention_layernorm
            got = self._consts[key] = (
                fn.data.float().contiguous(),
                scale.data.float().contiguous(),
                base.data.float().contiguous(),
                norm.weight.data.contiguous(),
                norm.variance_epsilon,
            )
        return got


class WideWoa(Stages):
    """``Stages`` with wo_a on ``stages.woa_wide`` for the covered layers: SGLang's inverse
    RoPE, then one launch from the attention output to wo_b's MXFP8 input (the codes and
    scales SGLang's gfx950 route emits, bit for bit from 65 rows on at TP 2; at TP 4 up to
    192 rows a deterministic K split, within an e4m3 ulp of SGLang's bmm; ``woa_wide``). Layers
    without the plugin's wo_a copy (the spec layer's ``woa_wide``: fragment-order e4m3, with
    its E8M0 codes) keep SGLang's."""

    def __init__(self, shared: Shared, c: int):
        super().__init__(shared, c)
        self._woa = shared.once("woa", lambda: _woa_weights(shared))
        self._ws = shared.once("woa workspace", lambda: _woa_workspaces(shared, self._woa))

    def wo_a(self, attn, o, positions, forward_batch, front: Front):
        import sglang.srt.models.deepseek_v4 as dv4
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Mxfp8Activation

        from sgl_amd_mono.deepseek_v41.gfx950.stages import woa_wide

        w = self._woa.get(attn.layer_id)
        if w is None:
            return super().wo_a(attn, o, positions, forward_batch, front)
        if front.inv_rope is None:
            dv4.fused_rope_inplace(
                o[..., -attn.qk_rope_head_dim :],
                None,
                attn.freqs_cis,
                positions=positions,
                inverse=True,
            )
        T, G = o.shape[0], attn.n_local_groups
        o = o.view(T, G, -1)
        assert o.stride(2) == 1 and o.stride(1) == o.shape[2], o.stride()
        part, flags = self._ws[width_class(T)]
        q = torch.empty(T, G * 1024, dtype=torch.float8_e4m3fn, device=o.device)
        s = torch.empty(T, G * 32, dtype=torch.uint8, device=o.device)
        woa_wide.standalone(T, G, ldx=o.stride(0))(
            o.data_ptr(), w[0].data_ptr(), w[1].data_ptr(), q.data_ptr(), s.data_ptr(),
            part.data_ptr(), flags.data_ptr(),
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        return Mxfp8Activation(q, s)


def _woa_weights(shared: Shared) -> dict:
    """layer -> (wo_a's fragment-order e4m3, its E8M0 codes) where the spec keeps the copy;
    each such layer on SGLang's bf16 wo_a route with wo_b on an MXFP8 input."""
    import sglang.srt.models.deepseek_v4 as dv4

    woa = {i: (L.woa_wide, L.k2.woa_s) for i, L in shared.layers.items() if L.woa_wide is not None}
    for i in woa:
        a = shared.model.layers[i].self_attn
        if a.wo_a_fp8 or a.use_fused_wo_a or a.use_flashinfer_mxfp8_wo_b:
            raise RuntimeError(f"layer {i}: the wide wo_a needs SGLang's bf16 wo_a route")
        if not dv4._hip.wo_b_emits_mxfp8(a, CLASSES[0]):
            raise RuntimeError(f"layer {i}: the wide wo_a needs wo_b on an MXFP8 input")
    return woa


def _woa_workspaces(shared: Shared, woa: dict) -> dict:
    """Each class's K-split workspace (f32 partials, phase-bit flags zeroed once), at its
    widest row count; a class's launches run in stream order, so they share it."""
    from sgl_amd_mono.deepseek_v41.gfx950.stages import woa_wide

    if not woa:
        return {}
    a = shared.model.layers[min(woa)].self_attn
    g, dev = a.n_local_groups, a.wo_a.weight.device
    out = {}
    for c in CLASSES:
        pw, fw = woa_wide.workspace(c, g, woa_wide.TILES[(g, woa_wide.width_class(c))])
        out[c] = (
            torch.empty(pw, dtype=torch.float32, device=dev),
            torch.zeros(fw, dtype=torch.int32, device=dev),
        )
    return out


# the width classes whose core runs on the union kernel, by the rank's heads: where it is
# measured faster than SGLang's (the 384-row class at TP4 and TP2)
UNION_CLASSES = {16: (384,), 32: (384,)}


class UnionStages(Stages):
    """``Stages`` with the attention core on the plugin's union kernel
    (``gfx950/attention/wide_core.py``): a request's six rows read its keys once.

    The unions follow the step's index lists: the window union once a step (every layer
    reads the same window), the top-k union at each index source (the layers after it
    read its top-k until the next source). A window-only layer keeps SGLang's core (the
    union kernel is slower there). A width class's launches take the class's buffers,
    sized for its widest step."""

    def __init__(self, shared: Shared, c: int):
        super().__init__(shared, c)
        self.rows = c
        self.reqs = c // 6
        self._bufs = None

    def _buffers(self, dev, heads, b):
        from sgl_amd_mono.deepseek_v41.gfx950.attention import wide_core as wc

        if self._bufs is None:
            i32 = dict(dtype=torch.int32, device=dev)
            npart, nml, nflag = wc.workspace(b)
            self._bufs = dict(
                ws=torch.empty(self.reqs, 6 * 128, **i32),
                wm=torch.empty(self.reqs, 6 * 128, **i32),
                wn=torch.empty(self.reqs, **i32),
                es=torch.empty(self.reqs, 6 * 512, **i32),
                em=torch.empty(self.reqs, 6 * 512, **i32),
                en=torch.empty(self.reqs, **i32),
                part=torch.empty(npart, dtype=torch.float16, device=dev),
                pml=torch.empty(nml, dtype=torch.float32, device=dev),
                flags=torch.zeros(nflag, **i32),
            )
        return self._bufs

    def attn_core(self, attn, front, forward_batch):
        from sgl_amd_mono.deepseek_v41.gfx950.attention import wide_core as wc
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        md = backend.forward_metadata.core_attn_metadata
        pool = backend.token_to_kv_pool
        layer = attn.attn_mqa.layer_id
        ratio = attn.compress_ratio
        q = front.q
        T, H = q.shape[0], q.shape[1]
        reqs = T // 6
        assert T % 6 == 0 and T <= self.rows and front.inv_rope is not None, (T, self.rows)
        assert q.dtype == torch.bfloat16 and H % 16 == 0, (q.dtype, H)
        freqs, positions = front.inv_rope
        swa = pool.get_swa_key_buffer_radix(layer)
        wcache = wc.Cache(page=pool.swa_page_size, stride=swa.stride(0) * swa.element_size())
        ext = pool.get_extra_key_buffer(layer) if ratio else swa
        ecache = (
            wc.Cache(page=pool.get_extra_key_page_size(layer), stride=ext.stride(0) * ext.element_size())
            if ratio
            else wcache
        )
        b = wc.CoreBuild(
            reqs=self.reqs, heads=H, splits=wc.splits_for(self.reqs, H), window=wcache,
            extra=ecache, qs=q.stride(0), qh=q.stride(1), scale=backend.softmax_scale,
            rope_stride=freqs.stride(0), pos_words=positions.element_size() // 4,
        )  # fmt: skip
        buf = self._buffers(q.device, H, b)
        s = torch.cuda.current_stream()
        if layer == 0:
            win, win_n = md.swa_page_indices, md.swa_topk_lengths
            wc.union_build(128, win.stride(0), swa.shape[0] * wcache.page)(
                win.data_ptr(), win_n.data_ptr(), buf["ws"].data_ptr(), buf["wm"].data_ptr(),
                buf["wn"].data_ptr(), reqs, stream=s,
            )  # fmt: skip
        if not ratio:
            return super().attn_core(attn, front, forward_batch)
        if getattr(attn, "indexer", None) is not None:
            top, top_n = md.sparse_page_indices(ratio), md.sparse_topk_lengths(ratio)
            wc.union_build(512, top.stride(0), ext.shape[0] * ecache.page)(
                top.data_ptr(), top_n.data_ptr(), buf["es"].data_ptr(), buf["em"].data_ptr(),
                buf["en"].data_ptr(), reqs, stream=s,
            )  # fmt: skip
        out = q.new_empty((T, H, 512))
        wc.core_build(b)(
            q.data_ptr(), swa.data_ptr(), ext.data_ptr(), buf["ws"].data_ptr(), buf["wm"].data_ptr(),
            buf["wn"].data_ptr(), buf["es"].data_ptr(), buf["em"].data_ptr(),
            buf["en"].data_ptr(), front.sink.data_ptr(), freqs.data_ptr(),
            positions.data_ptr(), out.data_ptr(), buf["part"].data_ptr(), buf["pml"].data_ptr(),
            buf["flags"].data_ptr(), reqs, wc.groups_per_xcd(reqs, H), stream=s,
        )  # fmt: skip
        return out[:, front.tp_slice, :]


FP8_PROJECTIONS = ("wqkv_a", "wq_b", "wo_b")


class WideFp8(Stages):
    """``Stages`` with the attention's dense FP8 projections (wqkv_a, wq_b, wo_b) on
    ``stages.fp8_dense``: SGLang's front and wo_b run as they are, with the GEMM of their
    route (``Fp8LinearMethod.w8a8_mxfp8_linear``, after its operand is unwrapped or
    quantized) bound to aiter's FlyDSL GEMM, at the classes it is tuned for, while the stage
    runs (one bf16 ulp at about 1e-8 of the outputs: ``fp8_dense``)."""

    def __init__(self, shared: Shared, c: int):
        super().__init__(shared, c)
        self._gemm = shared.once("fp8 projections", lambda: _fp8_projections(shared.model))

    @contextmanager
    def _ours(self, rows, *linears):
        """The linears' route GEMM on the FlyDSL GEMM inside the block, those it serves at
        ``rows`` (``fp8_dense.served``)."""
        from sgl_amd_mono.deepseek_v41.gfx950.stages import fp8_dense

        linears = [lin for lin in linears if fp8_dense.served(*lin.weight.shape, rows)]
        methods = [lin.quant_method for lin in linears]
        saved = [m.w8a8_mxfp8_linear for m in methods]
        for lin, m in zip(linears, methods):
            m.w8a8_mxfp8_linear = self._gemm[lin]
        try:
            yield
        finally:
            for m, f in zip(methods, saved):
                m.w8a8_mxfp8_linear = f

    def attn_front(self, attn, x, positions, forward_batch, x_quant) -> Front:
        with self._ours(x.shape[0], attn.wqkv_a, attn.wq_b):
            return super().attn_front(attn, x, positions, forward_batch, x_quant)

    def wo_b(self, attn, o, defer_all_reduce: bool):
        # wo_a's output: bf16 rows, or wo_b's MXFP8 input (``Mxfp8Activation``)
        rows = (o.q if hasattr(o, "q") else o).shape[0]
        with self._ours(rows, attn.wo_b):
            return super().wo_b(attn, o, defer_all_reduce)


class _Fp8Linear:
    """The route GEMM's call (``aiter_mxfp8_blockscaled_linear``'s arguments) on a
    ``fp8_dense.Projection``: the operand as SGLang's route makes it (``mxfp8_operand``: an
    fp8 x with its scale as is, a bf16 x quantized), then the FlyDSL GEMM."""

    def __init__(self, proj):
        self.proj = proj

    def __call__(
        self, input, weight, weight_scale_ue8m0, input_scale=None, bias=None,
        weight_preshuffled=False,
    ):  # fmt: skip
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import mxfp8_operand

        assert weight is self.proj.weight and bias is None and not weight_preshuffled
        xq, xs = mxfp8_operand(input, input_scale)
        out = self.proj(xq, xs.view(torch.uint8))
        return out.view(*input.shape[:-1], self.proj.n)


def _fp8_projections(model) -> dict:
    """linear -> its route's GEMM on ``fp8_dense`` (``_Fp8Linear``), for the attention's
    dense FP8 projections at the shapes a class serves (each over a preshuffled weight copy)."""
    from sgl_amd_mono.deepseek_v41.gfx950.stages import fp8_dense

    gemm = {}
    for i in range(model.start_layer, model.end_layer):
        a = model.layers[i].self_attn
        if not a.fuse_wqa_wkv:
            raise RuntimeError(f"layer {i}: the wide FP8 projections need wqkv_a")
        for name in FP8_PROJECTIONS:
            lin = getattr(a, name)
            why = _fp8_route_why(lin)
            if why is not None:
                raise RuntimeError(f"layer {i} {name}: the wide FP8 projections need {why}")
            if any(fp8_dense.served(*lin.weight.shape, c) for c in CLASSES):
                gemm[lin] = _Fp8Linear(
                    fp8_dense.Projection(lin.weight, lin.weight_scale_mx_e8m0)
                )
    return gemm


def mixins(tp: int, heads: int, c: int) -> tuple:
    """The stages of width class c at TP size ``tp`` (``heads`` attention heads a rank), in
    method resolution order: each kernel's stages ahead of SGLang's (``Stages``)."""
    token_parallel = c in TOKEN_PARALLEL.get(tp, ())
    # wrapping SGLang's projections, ahead of every class that runs one (token parallel: wo_b)
    out: list = [WideFp8]
    if tp in SPAN_A_TPS:
        out.append(TokenParallelSpanA if token_parallel else SpanAStages)
    elif token_parallel:
        out.append(TokenParallelStages)
    out.append(SeamStages)
    out.append(WideWoa)
    if c in UNION_CLASSES.get(heads, ()):
        out.append(UnionStages)
    return (*out, Stages)


@functools.cache
def _composed(bases: tuple) -> type:
    """One class of ``bases``' stages."""
    return type("Wide" + "".join(b.__name__ for b in bases), bases, {})


class WideForward:
    """``DeepseekV4ForCausalLM.forward`` for a wide step, its layers in ``Stages``."""

    def __init__(self, lm, layers: Optional[dict] = None):
        self.lm, self.model = lm, lm.model
        # one set of stages a width class: SGLang's but for the kernels that replace one
        shared = Shared(lm, layers)
        self.stages: dict[int, Any] = {
            c: _composed(mixins(shared.tp, shared.heads, c))(shared, c) for c in CLASSES
        }

    def forward(self, plan, input_ids, positions, forward_batch, input_embeds=None):
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        lm = self.lm
        input_ids, input_embeds = lm.prepare_language_model_inputs(
            input_ids, forward_batch, input_embeds
        )
        stages = self.stages[width_class(plan.key.rows)]
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            out = self._backbone(stages, input_ids, positions, forward_batch, input_embeds)
        aux = None
        if lm.capture_aux_hidden_states:
            out, aux = out
        hidden, pre_hc_head = out
        return lm.logits_processor(
            input_ids,
            hidden,
            lm.lm_head,
            forward_batch,
            aux,
            hidden_states_before_norm=None if aux is not None else pre_hc_head,
        )

    def prepare_step(self, forward_batch) -> None:
        """The low-ratio indexers' decode schedules of the step, as the backend's
        ``init_forward_metadata_in_graph`` builds them for a step whose forward reads them."""
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DSV4Metadata,
        )
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            build_low_ratio_decode_workspaces,
            low_ratio_decode_rows_are_identity,
        )
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        metadata = backend.forward_metadata
        mode = forward_batch.forward_mode
        if not (
            isinstance(metadata, DSV4Metadata)
            and metadata.core_metadata.low_ratios
            and (mode.is_decode() or mode.is_target_verify())
        ):
            return
        metadata.fp4_low_ratio_decode_workspaces = build_low_ratio_decode_workspaces(
            {
                ratio: indexer_metadata
                for ratio, indexer_metadata in (
                    metadata.low_ratio_indexer_metadata_by_ratio().items()
                )
                # identity rows are not scored: they need no schedule
                if not low_ratio_decode_rows_are_identity(backend, forward_batch, ratio)
            },
            max_context_len=backend.max_context_len,
            num_requests=forward_batch.batch_size,
            page8_ratios=backend.token_to_kv_pool.low_ratio_index_k_page8_ratios(),
        )

    @torch.no_grad()
    def _backbone(self, stages, input_ids, positions, forward_batch, input_embeds):
        """``DeepseekV4Model._forward_impl`` with ``_forward_layers_hc_pre_from_prev`` for a
        decode or verify step: the embedding, Engram, each layer, the head's combine and
        norm."""
        import sglang.srt.models.deepseek_v4 as dv4
        from sglang.kernels.ops.layernorm.mhc import (
            hc_broadcast,
            mhc_combine,
            mhc_stream_mean,
        )

        m = self.model
        hidden = m.embed_tokens(input_ids) if input_embeds is None else input_embeds
        rows = hidden.shape[0]
        # this rank's rows: their streams and every row-local result of them
        own = stages.own(rows)
        like = hidden[own]  # a row of the captures' shape
        hidden = hc_broadcast(like, m.hc_mult)
        input_ids_global = getattr(forward_batch, "input_ids_global", input_ids)
        targets = m.dspark_layers_to_capture
        aux = None if targets is None else dv4.AuxHiddenStatePacker(len(targets))
        # the compressor caches each step's rotary tables on the batch
        for attr in ("freqs_cis_c4", "freqs_cis_c128"):
            if hasattr(forward_batch, attr):
                delattr(forward_batch, attr)
        self.prepare_step(forward_batch)
        hash_ids = None
        if m.engram_hasher is not None:
            if forward_batch.forward_mode.is_extend() and dv4.is_in_breakable_cuda_graph():
                hash_ids = m.engram_hasher.new_hash_ids(input_ids)
                dv4.bcg_deepseek_v4_engram_hash_ids(m.engram_hasher, input_ids, hash_ids)
            else:
                hash_ids = m.engram_hasher(input_ids, forward_batch)
        assert not dv4.get_forward().sp_active
        state = dv4.mhc.HcState(hidden)
        del hidden
        piecewise = dv4.check_cuda_graph_backend(dv4.Phase.PREFILL, dv4.Backend.TC_PIECEWISE)
        for i in range(m.start_layer, m.end_layer):
            engram = m.layers[i].engram
            if engram is not None:
                state = state.with_residual(
                    stages.engram(
                        engram,
                        state.residual,
                        hash_ids[:, engram.layer_hash_index],
                        forward_batch,
                        input_ids,
                    )
                )
            if aux is not None and i in targets:
                # the draft head reads the attention input of its target layers
                mhc_stream_mean(state.residual, out=aux.slot(like))
            nxt = i + 1
            # the next layer's Engram gate or DSpark capture reads the settled residual
            defer = (
                nxt < m.end_layer
                and m.layers[nxt].engram is None
                and not (aux is not None and nxt in targets)
            )
            ctx = (
                nullcontext()
                if piecewise
                else dv4.get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:
                state = stages.layer(
                    self, i, state, positions, input_ids, forward_batch,
                    input_ids_global, defer,
                )  # fmt: skip
        state = state.materialized(m.layers[m.end_layer - 1].hc_cfg)
        residual, pre = state.residual, state.pre
        pre_hc_head = residual.flatten(1)
        # the kernel widens each element to f32 as it loads it
        hidden = m.norm(mhc_combine(pre_hc_head, pre, m.hc_mult, residual.dtype))
        # the head reads every row (and the streams without the captures)
        hidden = stages.gather(hidden)
        if aux is not None:
            return (hidden, pre_hc_head), stages.gather(aux.finalize())
        return hidden, stages.gather(pre_hc_head)


def reasons(model, layers) -> list[str]:
    """What the wide forward needs of ``layers`` past the spec's other reasons: the
    attention this module opens (not the mega one; an FP8 wo_a only on aiter's mxscale
    GEMM). A late-layer tail is a prefill's: decode and verify steps run every row.
    Token parallel (``TOKEN_PARALLEL``) needs plain TP (the attention's TP group is the
    MoE's), aiter's custom all-reduce on it, and MoEs whose only sum over the ranks is
    their all-reduce."""
    from sglang.srt.runtime_context import get_parallel

    m = model.model
    out = []
    parallel = get_parallel()
    if TOKEN_PARALLEL.get(parallel.tp_size):
        ca = parallel.tp_group.ca_comm
        if parallel.attn_tp_size != parallel.tp_size:
            out.append("token-parallel wide steps need the attention TP group to be the TP group")
        if ca is None or ca.disabled or not hasattr(ca, "custom_reduce_scatter"):
            out.append("token-parallel wide steps need aiter's custom all-reduce on the TP group")
        for i in layers:
            moe = m.layers[i].mlp
            if not moe.reduce_results or moe._shared_expert_tp1 or moe._enable_a2a_moe:
                out.append(
                    f"layer {i}: token-parallel wide steps need the MoE's sum to be its "
                    "all-reduce (no TP1 shared expert, no all-to-all)"
                )
    if parallel.tp_size in SPAN_A_TPS:
        for i in layers:
            moe = m.layers[i].mlp
            if not moe.reduce_results or moe._shared_expert_tp1 or moe._enable_a2a_moe:
                out.append(
                    f"layer {i}: the wide span A launch needs the MoE's sum to be its "
                    "all-reduce (no TP1 shared expert, no all-to-all)"
                )
            if not getattr(m.layers[i], "fused_rmsnorm_fake_quant", False):
                out.append(f"layer {i}: the wide span A launch runs gfx950's fake-quant input norm")
    for i in layers:
        a = m.layers[i].self_attn
        if not getattr(m.layers[i], "fused_rmsnorm_fake_quant", False):
            out.append(f"layer {i}: wide seams run gfx950's fake-quant input norm")
        if getattr(a, "use_mega_attention", False):
            out.append(f"layer {i}: wide steps open MQALayer's attention, not the mega one")
        if getattr(a, "wo_a_fp8", False):
            import sglang.srt.models.deepseek_v4 as dv4

            if dv4._wo_a_fp8_mxscale is None:
                out.append(f"layer {i}: wide steps run an FP8 wo_a on aiter's mxscale GEMM")
    return out
