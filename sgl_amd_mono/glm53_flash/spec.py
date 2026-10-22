# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash on gfx950: the monokernel spec and its forward (RFC
sgl-project/sglang#43949).

Every covered layer is one launch for the whole decoder layer (``gfx950/kernel.py``):
the attention-side mHC, the attention (KDA, or sparse MLA with its kpool indexer
and cache writes), o_proj and its TP reduce, the FFN mHC seam and norm, the
router, the shared and routed MXFP4 experts and the MoE TP reduce.

The forward is ``Glm5NextForConditionalGeneration.forward`` for a routed decode
step: SGLang's embedding, dense layers 0 to 2, last layer, final norm and logits
processor around the covered layers' launches. The kernel owes each layer's FFN
post to the next layer's attention seam, so between covered layers the forward
hands on ``_Owed`` (x, residual, post, comb). The first covered layer takes the
streams layer 2 wrote with an identity mix (x 0, post 0, comb I: exact), and the
last applies the owed post with SGLang's hc_post and writes the streams again.

Covered: the MoE layers but the last (3 to 43: 30 KDA, 11 sparse attention); the
dense layers 0 to 2 and the last layer, whose FFN exit contracts the streams,
keep the normal path. A KDA pad row (state index not positive) keeps every state;
a sparse-attention pad row (SGLang's padding slot 0) writes no cache.

The kernels read SGLang's weights in place, except the copies logged at startup:
the router packed for the MFMA tiles and its bias in f32, W_UK^T from kv_b_proj, the indexer's
wk | weights_proj in bf16, the o_norm weight in f32, and the shared expert packed
as one AITER expert where SGLang keeps it apart (TP2).
"""

from __future__ import annotations

import importlib.util
import os

import msgspec
import torch
from sglang.srt.monokernels.spec import (
    DeviceLimits,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.glm53_flash import (
    HF_CONFIG,
    LINEAR_ATTN_CONFIG,
)

# kernel widths; a step of fewer rows runs at the next width, the rest pad rows
WIDTHS = (2, 4, 8)
ROWS = tuple(range(1, max(WIDTHS) + 1))
TP_SIZES = (2, 4, 8)
# the rows a step runs on the kernels at most at this TP size: at TP2 a sparse-
# attention launch of 8 rows does not fit the LDS, so wider steps run the model's
# forward
TP_MAX_ROWS = {2: 4}
# SGLang's DSA cache: 256-token KV pages (an index page holds 64 kpool pools)
PAGE_SIZE = 256


def _width(rows: int) -> int:
    return next(w for w in WIDTHS if w >= rows)


# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "page_size",
        "enable_linear_replayssm",
        "enable_hisparse",
        "expert_distribution_recorder_mode",
    ),
    "srt/configs/glm5_next.py": (
        "num_hidden_layers",
        "first_k_dense_replace",
        "moe_layer_freq",
        "linear_attn_config",
    ),
    "srt/configs/mamba_utils.py": ("mamba2_state_dtype",),
    "srt/environ.py": (
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
        "SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK",
    ),
    "srt/configs/model_config.py": ("context_len",),
    "srt/distributed/parallel_state.py": ("cpu_group",),
    "srt/layers/attention/dsa/utils.py": ("aiter_can_use_preshuffle_paged_mqa",),
    "srt/layers/attention/dsa_backend.py": ("forward_metadata", "real_page_table"),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "full_attn_backend",
        "linear_attn_backend",
    ),
    "srt/layers/attention/mamba/mamba2_metadata.py": ("mamba_cache_indices",),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": ("maybe_input_scattered",),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "take_output",
        "set_written",
        "complete_output",
        "final_norm",
    ),
    "srt/mem_cache/memory_pool.py": (
        "mamba2_layer_cache",
        "conv",
        "temporal",
        "get_key_buffer",
        "get_index_k_with_scale_buffer",
        "get_compress_tail_buffers",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "out_cache_loc",
        "req_pool_indices",
        "forward_mode",
        "is_idle",
    ),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_req_to_token_pool",
        "get_token_to_kv_pool",
    ),
    "srt/models/deepseek_v2.py": ("indexer",),
    "srt/models/glm5_next.py": (
        "model",
        "embed_tokens",
        "layers",
        "input_layernorm",
        "do_fuse_qkvbfg",
        "start_layer",
        "end_layer",
        "gemm_output_zero_allocator_size",
        "norm",
        "self_attn",
        "hc_post",
        "is_linear_attn",
        "is_mrope_enabled",
        "logits_processor",
        "lm_head",
        "forward(prev_topk_indices)",
    ),
    "srt/monokernels/spec.py": (
        "DeviceLimits",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "widths",
        "deadline_s",
        "copies",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        "process_model_config",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
    "srt/utils/common.py": ("BumpAllocator",),
}

# Each covered layer against SGLang's normal layer on the same input and caches
# (SGLANG_AMD_MONO_CHECK, the serving check test): the streams it writes and the
# state or cache rows. The kernel quantizes the expert inputs to MXFP8 and SGLang's
# AITER MoE its own way, so the streams differ by about the MoE's error.
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "kda.streams.cosine": 0.95,
        "kda.conv.cosine": 0.9999,
        "kda.state.cosine": 0.9999,
        "dsa.streams.cosine": 0.95,
        "dsa.kv.cosine": 0.995,
        "dsa.index.cosine": 0.999,
        "dsa.index_scale.rel": 0.01,
        "dsa.tail_k.cosine": 0.9999,
        "dsa.tail_g.cosine": 0.9999,
    },
)


def _text(config):
    return getattr(config, "text_config", config)


def _covered_layers(config) -> list[int]:
    """The MoE layers but the last, whose FFN exit contracts the streams."""
    text = _text(config)
    last = text.num_hidden_layers - 1
    return [
        i
        for i in range(text.first_k_dense_replace, last)
        if i % text.moe_layer_freq == 0
    ]


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype
    from sglang.srt.layers.attention.dsa.utils import (
        aiter_can_use_preshuffle_paged_mqa,
    )

    text = _text(hf_config)
    reasons = [
        f"needs GLM-5.3-Flash's {key} = {want!r}, got {getattr(text, key, None)!r}"
        for key, want in HF_CONFIG.items()
        if getattr(text, key, None) != want
    ]
    linear = getattr(text, "linear_attn_config", None) or {}
    reasons += [
        f"needs GLM-5.3-Flash's linear_attn_config {key} = {want!r}"
        for key, want in LINEAR_ATTN_CONFIG.items()
        if linear.get(key) != want
    ]
    kda_layers = linear.get("kda_layers") or ()
    if kda_layers and max(kda_layers) in _covered_layers(hf_config):
        reasons.append(
            f"layer {max(kda_layers)}: the last KDA layer keeps the normal path, where "
            "SGLang copies every KDA layer's decode state for the radix cache"
        )
    dtypes = mamba2_state_dtype(hf_config)
    if dtypes.temporal != torch.float32 or dtypes.conv != torch.bfloat16:
        reasons.append(
            f"needs the KDA state in f32 and its conv state in bf16, got "
            f"{dtypes.temporal} and {dtypes.conv}"
        )
    if cfg.page_size not in (None, PAGE_SIZE):
        reasons.append(
            f"--page-size {cfg.page_size}: the kpool caches need {PAGE_SIZE}"
        )
    if not aiter_can_use_preshuffle_paged_mqa():
        reasons.append(
            "needs the index cache in AITER's preshuffled layout, which the kernels "
            "write; without AITER's preshuffle paged MQA, SGLang keeps it row-major"
        )
    quant = (getattr(hf_config, "quantization_config", None) or {}).get("quant_method")
    if quant != "quark":
        reasons.append(
            f"needs a Quark checkpoint (MXFP4 experts, FP8 block-128 attention), got "
            f"{quant!r}"
        )
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sgl_amd_mono.glm53_flash.gfx950.weights import (
        experts_reason,
    )

    reasons = []
    if model.is_mrope_enabled:
        reasons.append("multimodal rotary positions: the kernels take one per row")
    for i in _covered_layers(hf_config):
        layer = model.model.layers[i]
        why = experts_reason(layer)
        if why is not None:
            reasons.append(f"layer {i}: {why}")
        attn = layer.self_attn
        if layer.is_linear_attn and not attn.do_fuse_qkvbfg:
            reasons.append(
                f"layer {i}: needs the fused q | k | v | beta | f_a | g_a projection"
            )
        if not layer.is_linear_attn and attn.indexer is None:
            reasons.append(f"layer {i}: sparse attention without its indexer")
    return reasons


class _Owed(msgspec.Struct, frozen=True):
    """A covered layer's output with its FFN post still owed: x [S, HIDDEN], the
    streams after its attention [S, HC, HIDDEN], post [S, HC, 1], comb [S, HC, HC]."""

    x: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor


def _written(module, x, residual, post, comb) -> torch.Tensor:
    """The streams with the owed FFN post applied, as SGLang's FFN exit writes them."""
    s = x.shape[0]
    return module.hc_post(x, residual.reshape(s, -1), comb, post.reshape(s, -1))


class _Layer:
    """One covered layer: its module, kind and per-width ops; its state or caches
    bind at its first launch and rebind if they move."""

    def __init__(self, i: int, module, kda: bool, attn):
        self.i, self.module, self.kda, self.attn = i, module, kda, attn
        self.ops: dict = {}  # width -> FlashLayerOp
        self.state = None  # FlashKdaState
        self.caches = None  # FlashDsaCaches


class _Built:
    """One rank's monokernels: the runtimes per (width, kind), each covered layer's
    ops, and the forward."""

    def __init__(self, lm, layers: dict, runtimes: dict, tp: int, device):
        from sgl_amd_mono.glm53_flash.gfx950.config import (
            HC_MULT,
            HIDDEN,
        )
        from sgl_amd_mono.glm53_flash.gfx950.layout import (
            ROW_BT,
            ROW_WORDS,
        )

        self.lm, self.model = lm, lm.model
        self._layers, self.runtimes = layers, runtimes
        self.first, self.last = min(layers), max(layers)
        self.max_rows = TP_MAX_ROWS.get(tp, max(WIDTHS))
        self.check_layers = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"
        # the first covered layer's identity mix: the written streams pass unchanged
        n = max(WIDTHS)
        self.zero_x = torch.zeros(n, HIDDEN, dtype=torch.bfloat16, device=device)
        self.zero_post = torch.zeros(n, HC_MULT, 1, dtype=torch.float32, device=device)
        self.eye_comb = (
            torch.eye(HC_MULT, dtype=torch.float32, device=device)
            .expand(n, HC_MULT, HC_MULT)
            .contiguous()
        )
        # per width: the step's sparse-attention row table, KDA state indices and
        # the first covered layer's streams with zero pad rows
        self.rows, self.kda_idx, self.pad_res = {}, {}, {}
        self.row_ids = torch.arange(n, dtype=torch.int32, device=device)
        for w in {w for w, _ in runtimes}:
            rows = torch.full((w, ROW_WORDS), -1, dtype=torch.int32, device=device)
            rows[:, ROW_BT] = self.row_ids[:w]
            self.rows[w] = rows
            self.kda_idx[w] = torch.zeros(w, dtype=torch.int32, device=device)
            self.pad_res[w] = torch.zeros(
                w, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=device
            )

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        """``Glm5NextForConditionalGeneration.forward`` for a routed decode step."""
        lm = self.lm
        if plan.key.rows > self.max_rows:
            return type(lm).forward(
                lm, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden = self._backbone(
                plan, input_ids, positions, forward_batch, input_embeds
            )
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, None)

    def _backbone(self, plan, input_ids, positions, forward_batch, input_embeds):
        """``Glm5NextModel.forward``: the embedding, the dense and last layers on
        SGLang's path and the covered ones as launches, then the final norm."""
        from sglang.srt.layers.layer_boundary.residual.batch import (
            complete_output,
            final_norm,
            start,
        )
        from sglang.srt.utils.common import BumpAllocator

        m = self.model
        hidden = m.embed_tokens(input_ids) if input_embeds is None else input_embeds
        start(forward_batch)
        zero = BumpAllocator(
            buffer_size=(m.end_layer - m.start_layer) * 2,
            dtype=torch.float32,
            device=hidden.device,
        )
        gemm_zero = None
        if m.gemm_output_zero_allocator_size > 0:
            gemm_zero = BumpAllocator(
                buffer_size=m.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=hidden.device,
            )
        topk_indices = None
        for i in range(m.start_layer, m.end_layer):
            L = self._layers.get(i)
            if L is not None:
                hidden = self._layer(L, plan, positions, hidden, forward_batch)
                topk_indices = None
                continue
            hidden, topk_indices = m.layers[i](
                positions,
                hidden,
                forward_batch,
                zero,
                gemm_zero,
                prev_topk_indices=topk_indices,
            )
        hidden = complete_output(hidden, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden = final_norm(hidden, forward_batch, m.norm)
        return hidden

    def _layer(self, L: _Layer, plan, positions, hidden_states, forward_batch):
        """The streams the previous layer wrote (the first covered layer) or the
        previous launch's ``_Owed`` in; ``_Owed`` out, or the written streams after
        the last covered layer."""
        from sglang.srt.layers.layer_boundary.residual.batch import (
            set_written,
            take_output,
        )

        from sgl_amd_mono.glm53_flash import check

        S = plan.key.rows
        W = _width(S)
        checking = self.check_layers and not torch.cuda.is_current_stream_capturing()
        if L.i == self.first:
            streams = take_output(hidden_states, forward_batch)
            res = streams.view(S, self.eye_comb.shape[1], -1)
            if S < W:
                pad = self.pad_res[W]
                pad[:S].copy_(res)
                pad[S:].zero_()
                res = pad
            ins = (self.zero_x[:W], res, self.zero_post[:W], self.eye_comb[:W])
        else:
            ins = (
                hidden_states.x,
                hidden_states.residual,
                hidden_states.post,
                hidden_states.comb,
            )
            streams = _written(L.module, *(t[:S] for t in ins)) if checking else None
        idx = self._kda_state_idx(plan, S, W) if L.kda else None
        if checking:
            ref = check.reference(
                L, positions, streams, forward_batch, S, idx[:S] if L.kda else None
            )
        op = L.ops[W]
        if L.kda:
            self._bind_kda(L)
            out = op.launch(*ins, state=L.state, state_idx=idx)
        else:
            self._bind_dsa(L)
            rows, page_table = self._dsa_rows(plan, positions, forward_batch, S, W)
            out = op.launch(*ins, caches=L.caches, rows=rows, page_table=page_table)
        if checking:
            written = _written(L.module, *(t[:S] for t in out))
            check.compare(
                L, ref, written, positions, forward_batch, S, idx[:S] if L.kda else None
            )
        if L.i != self.last:
            return _Owed(*out)
        for (w, _), rt in self.runtimes.items():
            if w == W:
                rt.advance_step()
        return set_written(_written(L.module, *(t[:S] for t in out)), forward_batch)

    @staticmethod
    def _bind_kda(L: _Layer) -> None:
        from sglang.srt.model_executor.forward_context import get_req_to_token_pool

        from sgl_amd_mono.glm53_flash.gfx950.runner import (
            FlashKdaState,
        )

        cache = get_req_to_token_pool().mamba2_layer_cache(L.i)
        conv, rec = cache.conv[0], cache.temporal
        if L.state is not None and (
            L.state.conv_state.data_ptr() == conv.data_ptr()
            and L.state.rec_state.data_ptr() == rec.data_ptr()
        ):
            return
        P3, H, D = L.attn.conv_w.shape[0], L.attn.a_log.shape[0], 128
        if not (
            conv.dtype == torch.bfloat16
            and tuple(conv.shape[1:]) == (3, P3)
            and conv.stride()[1:] == (P3, 1)
            and rec.dtype == torch.float32
            and tuple(rec.shape[1:]) == (H, D, D)
            and rec.stride()[1:] == (D * D, D, 1)
            and conv.shape[0] == rec.shape[0]
        ):
            raise RuntimeError(
                f"GLM-5.3-Flash monokernel: layer {L.i}: KDA state conv "
                f"{conv.dtype} {tuple(conv.shape)} {conv.stride()}, recurrent "
                f"{rec.dtype} {tuple(rec.shape)} {rec.stride()}; needs bf16 "
                f"[slots, 3, {P3}] and f32 [slots, {H}, {D}, {D}]"
            )
        L.state = FlashKdaState(conv_state=conv, rec_state=rec)

    @staticmethod
    def _bind_dsa(L: _Layer) -> None:
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        from sgl_amd_mono.glm53_flash.gfx950.runner import (
            INDEX_BLOCK_POOLS,
            FlashDsaCaches,
        )

        pool = get_token_to_kv_pool()
        kv = pool.get_key_buffer(L.i)
        kv = kv.view(kv.shape[0], -1)
        idx = pool.get_index_k_with_scale_buffer(L.i)
        tail_k, tail_g = pool.get_compress_tail_buffers(L.i)
        if L.caches is not None and all(
            a.data_ptr() == b.data_ptr()
            for a, b in zip(
                (
                    L.caches.kv_cache,
                    L.caches.idx_cache,
                    L.caches.tail_k,
                    L.caches.tail_g,
                ),
                (kv, idx, tail_k, tail_g),
            )
        ):
            return
        reasons = []
        if kv.dtype != torch.bfloat16 or kv.shape[1] < 512 or kv.stride(1) != 1:
            reasons.append(
                f"MLA cache {kv.dtype} {tuple(kv.shape)} {kv.stride()}, needs bf16 "
                "[slots, >= 512] with unit stride"
            )
        if (
            idx.dtype != torch.uint8
            or idx.dim() != 2
            or idx.shape[1] != INDEX_BLOCK_POOLS * 132
            or not idx.is_contiguous()
        ):
            reasons.append(
                f"index cache {idx.dtype} {tuple(idx.shape)}, needs contiguous uint8 "
                f"[pages, {INDEX_BLOCK_POOLS * 132}]"
            )
        for name, t in (("k", tail_k), ("gate", tail_g)):
            if (
                t.dtype != torch.bfloat16
                or tuple(t.shape[1:]) != (4, 128)
                or not t.is_contiguous()
            ):
                reasons.append(
                    f"kpool {name} ring {t.dtype} {tuple(t.shape)}, needs contiguous "
                    "bf16 [slots, 4, 128]"
                )
        if reasons:
            raise RuntimeError(
                f"GLM-5.3-Flash monokernel: layer {L.i}: " + "; ".join(reasons)
            )
        L.caches = FlashDsaCaches(
            kv_cache=kv, idx_cache=idx, tail_k=tail_k, tail_g=tail_g
        )

    def _kda_state_idx(self, plan, S: int, W: int) -> torch.Tensor:
        """The step's KDA state slots as int32, 0 on the width's pad rows (the
        forward's first KDA layer converts them; a graph records the copy)."""
        if "kda_idx" not in plan.state:
            from sglang.srt.model_executor.forward_context import get_attn_backend

            md = get_attn_backend().linear_attn_backend.forward_metadata
            buf = self.kda_idx[W]
            buf[:S].copy_(md.mamba_cache_indices[:S])
            buf[S:].zero_()
            plan.state["kda_idx"] = buf
        return plan.state["kda_idx"]

    def _dsa_rows(self, plan, positions, forward_batch, S: int, W: int):
        """The step's (position, MLA slot, index pool slot, ring slot) per row, -1 on
        pad rows (SGLang's padding slot 0, and the width's rows past S); and
        SGLang's 64-token page table, of S rows (the rows past S read row 0)."""
        if "dsa_rows" not in plan.state:
            from sglang.srt.model_executor.forward_context import get_attn_backend

            from sgl_amd_mono.glm53_flash.gfx950.config import (
                KPOOL,
            )
            from sgl_amd_mono.glm53_flash.gfx950.layout import (
                ROW_BT,
                ROW_POOL_SLOT,
                ROW_POS,
                ROW_SLOT,
                ROW_TAIL_SLOT,
            )

            loc = forward_batch.out_cache_loc[:S]
            valid = loc != 0
            none = torch.full_like(loc, -1)
            rows = self.rows[W]
            rows[:S, ROW_POS] = torch.where(valid, positions[:S].to(loc.dtype), none)
            rows[:S, ROW_SLOT] = torch.where(valid, loc, none)
            rows[:S, ROW_POOL_SLOT] = torch.where(valid, loc // KPOOL, none)
            req = forward_batch.req_pool_indices[:S].to(loc.dtype)
            rows[:S, ROW_TAIL_SLOT] = torch.where(valid, req * KPOOL, none)
            rows[:S, ROW_BT] = self.row_ids[:S]
            # an empty row's split still reads its first page-table entry
            rows[S:, ROW_POS : ROW_TAIL_SLOT + 1] = -1
            rows[S:, ROW_BT] = 0
            md = get_attn_backend().full_attn_backend.forward_metadata
            plan.state["dsa_rows"] = (rows, md.real_page_table[:S])
        return plan.state["dsa_rows"]


def _build(model, hf_config, ctx) -> _Built:
    from sglang.srt.runtime_context import get_parallel, process_model_config

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.glm53_flash.gfx950.config import (
        GLOBAL_INTER,
        KPOOL,
    )
    from sgl_amd_mono.glm53_flash.gfx950.kernel import (
        FlashBuild,
    )
    from sgl_amd_mono.glm53_flash.gfx950.layout import (
        POOLS_PER_TASK,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashLayerOp,
        FlashRuntime,
    )
    from sgl_amd_mono.glm53_flash.gfx950.weights import (
        dsa_weights,
        kda_weights,
        layer_weights,
    )

    p = get_parallel()
    tp, rank = p.tp_size, p.tp_rank
    inter = GLOBAL_INTER // tp
    device = model.model.layers[0].input_layernorm.weight.device
    copies = ctx.copies
    layers, weights = {}, {}
    for i in _covered_layers(hf_config):
        module = model.model.layers[i]
        ffn = layer_weights(module, i, inter, copies)
        if module.is_linear_attn:
            attn = kda_weights(module, i, tp, copies)
        else:
            attn = dsa_weights(module, i, tp, copies)
        layers[i] = _Layer(i, module, module.is_linear_attn, attn)
        weights[i] = (ffn, attn)
    dsa_formats = {
        (weights[i][1].fp8, weights[i][1].shuffled) for i in layers if not layers[i].kda
    }
    if len(dsa_formats) > 1:
        raise ValueError(
            f"GLM-5.3-Flash monokernel: sparse-attention projections differ between "
            f"layers (fp8, shuffled): {sorted(dsa_formats)}"
        )
    (qkv8, qb8, o8), (qkv_sh, qb_sh, o_sh) = (
        dsa_formats.pop() if dsa_formats else ((True, True, True), (False,) * 3)
    )
    kinds = {"kda": sorted(i for i in layers if layers[i].kda)}
    kinds["dsa"] = sorted(i for i in layers if not layers[i].kda)
    fault_site, fault_after = guard.fault_injection(rank) or (0, 0)
    # the pools a sparse-attention row scores at most: the served context's, whole
    # score tasks of 256 pools
    pools = -(-process_model_config().context_len // KPOOL)
    max_pools = -(-pools // POOLS_PER_TASK) * POOLS_PER_TASK
    runtimes = {}
    cap = TP_MAX_ROWS.get(tp, max(WIDTHS))
    widths = sorted({_width(r) for r in ctx.widths if r <= cap})
    if not widths:
        raise ValueError(
            f"GLM-5.3-Flash monokernel: TP{tp} runs at most {cap} rows on the kernels, "
            f"and this server's decode graphs route {sorted(ctx.widths)}"
        )
    # each runtime's peer handshake is collective: every rank builds them in one order
    for w in widths:
        for kind, ids in kinds.items():
            key = FlashBuild(
                rows=w,
                tp=tp,
                launches_per_step=len(ids),
                inter=inter,
                kda=kind == "kda",
                dsa=kind == "dsa",
                max_pools=max_pools if kind == "dsa" else 0,
                qkv_fp8=qkv8,
                qb_fp8=qb8,
                o_fp8=o8,
                qkv_shuffled=qkv_sh,
                qb_shuffled=qb_sh,
                o_shuffled=o_sh,
                fault_site=fault_site,
                fault_after=fault_after,
            )
            rt = FlashRuntime(key, rank, p.tp_group.cpu_group, device)
            rt.set_deadline(ctx.deadline_s)
            runtimes[w, kind] = rt
            for n, i in enumerate(ids):
                ffn, attn = weights[i]
                layers[i].ops[w] = FlashLayerOp(
                    rt,
                    ffn,
                    n,
                    kda=attn if kind == "kda" else None,
                    dsa=attn if kind == "dsa" else None,
                )
    # Every rank builds every width. FlyDSL's disk cache hands a rank the kernels
    # another rank built (a lock per kernel), so each rank starts at its own share.
    start = len(widths) * rank // tp
    for w in widths[start:] + widths[:start]:
        for kind in kinds:
            runtimes[w, kind].precompile()
    return _Built(model, layers, runtimes, tp, device)


SPEC = MonoSpec(
    name="glm53_flash_hip",
    devices={
        "gfx950": DeviceLimits(widths=ROWS, tp_sizes=TP_SIZES, resident_blocks=BLOCKS)
    },
    features=frozenset(),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"dsa"}),
        server_args={
            "kv_cache_dtype": frozenset({"auto", "bfloat16"}),
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # the launch updates the KDA state in place; ReplaySSM defers it to a ring
            "enable_linear_replayssm": frozenset({False}),
            # HiSparse maps the logical KV slots to device rows; the kernels write raw slots
            "enable_hisparse": frozenset({False}),
            # the launches record no expert distribution, and layer 44 would record
            # outside its layer (SGLang's forward sets it per layer)
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        env={
            # benchmark routing overrides: the kernel routes with the model's own router
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
            # the kernel breaks router ties in AITER's grouped top-k order, the path
            # SGLang takes with this off; on, its Triton gate keeps the lower expert id
            "SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK": False,
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
