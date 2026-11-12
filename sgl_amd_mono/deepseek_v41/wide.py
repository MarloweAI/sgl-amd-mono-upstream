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

The forward reads what SGLang's backend builds for the step, except the low-ratio
indexers' decode schedules: the spec declares them unread (the monokernel scores every
index source itself), so ``prepare_step`` builds them here, in the graph, as the backend
would (``init_forward_metadata_in_graph``).
"""

from __future__ import annotations

from contextlib import nullcontext
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
    ),
    "srt/models/deepseek_common/amd/deepseek_v4_hip.py": (
        "skip_head_pad",
        "attention_inv_rope",
        "attention_inv_rope(wo_a_applies_inv_rope)",
        "wo_a_emits_fp8_grid",
        "wo_b_emits_mxfp8",
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
        "Fp8GridActivation",
        "Mxfp8Activation",
        "fake_quant_fp8_activation",
        "fp8_grid_quantize",
    ),
    "srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py": ("_hip_mxfp8_operand",),
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
    ),
    "srt/mem_cache/deepseek_v4_memory_pool.py": ("low_ratio_index_k_page8_ratios",),
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


class Stages:
    """One layer's stages, SGLang's own code in SGLang's order (the module docstring)."""

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

    def __init__(self, exchange: Exchange):
        self.x = exchange

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


class WideForward:
    """``DeepseekV4ForCausalLM.forward`` for a wide step, its layers in ``Stages``."""

    def __init__(self, lm):
        from sglang.srt.runtime_context import get_parallel

        self.lm, self.model = lm, lm.model
        # one set of stages a width class: SGLang's until a kernel replaces one
        parallel = get_parallel()
        token_parallel = TOKEN_PARALLEL.get(parallel.tp_size, ())
        exchange = Exchange(parallel.tp_group) if token_parallel else None
        self.stages: dict[int, Any] = {
            c: TokenParallelStages(exchange) if c in token_parallel else Stages()
            for c in CLASSES
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
    for i in layers:
        a = m.layers[i].self_attn
        if getattr(a, "use_mega_attention", False):
            out.append(f"layer {i}: wide steps open MQALayer's attention, not the mega one")
        if getattr(a, "wo_a_fp8", False):
            import sglang.srt.models.deepseek_v4 as dv4

            if dv4._wo_a_fp8_mxscale is None:
                out.append(f"layer {i}: wide steps run an FP8 wo_a on aiter's mxscale GEMM")
    return out
