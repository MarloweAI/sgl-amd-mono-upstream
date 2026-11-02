# SPDX-License-Identifier: Apache-2.0
"""Muse Spark's DFlash2 draft on gfx950: its spec and forward (RFC #43949).

SGLang attaches the draft model as it does the target: a routed draft step (a
block of DRAFT_TOKENS rows a request, run as a TARGET_VERIFY batch) runs
``_Draft.forward``, which takes and returns what ``DFlash2DraftModel.forward``
does: the final-normed hidden states of the block, which the worker's selector
turns into the draft tokens. The draft is a dense Qwen3-style stack (5 sliding
layers, hidden 8192, 64 / 8 heads of 128, intermediate 28672, a grouped two-tap
convolution around each sublayer) with no head of its own.

Stage 2 (the default): each layer runs as four launches, D-pre (the input norm's
fused add, the attention convolution's coefficients and conv0, the qkv projection,
the per-head q / k norm + table RoPE, q out and k / v into the fp8 cache), D-attn
(the block attention over the compact windowed cache), D-post (the o projection,
the TP sum, conv1) and D-MLP (the post norm's fused add, the convolutions, gate /
up / down, the TP sum) — ``gfx950/draft_attn_half.py``, ``draft_attn.py``,
``draft_mlp.py``. ``SGLANG_AMD_MONO_DRAFT_STAGE=1`` keeps the attention half on
SGLang's modules; ``SGLANG_AMD_MONO_DRAFT_KERNELS=0`` (stage 0) runs the model's
own forward, so the wiring (attach, widths, block rows, graph capture) can be
exercised with an output identical to the normal path. Check mode
(``SGLANG_AMD_MONO_CHECK=1``, graphs off) runs SGLang's halves on copies and logs
each layer's cosines (attn_out, the residual, the MLP output).
"""

from __future__ import annotations

import logging
import os

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
from sgl_amd_mono.muse_spark import DRAFT_TOKENS, TP
from sgl_amd_mono.muse_spark import spec as target
from sgl_amd_mono.muse_spark.gfx950.config import WIDTHS

logger = logging.getLogger(__name__)

# the draft the kernels are written for (RadixArk/Muse-Spark-Open-DFlash2)
DRAFT_CONFIG = dict(
    hidden_size=8192,
    num_hidden_layers=5,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=28672,
    sliding_window=2048,
    rms_norm_eps=1e-5,
)
DFLASH_CONFIG = dict(
    block_size=DRAFT_TOKENS,
    conv_kernel_size=2,
    conv_group_size=16,
    attention_mode="gqa",
)

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Requirements",
    ),
    "srt/models/dflash.py": (
        "DFlash2DraftModel",
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
        "block_size",
        "layers",
        "candidate_selector",
        "draft_config",
        "norm",
        "input_layernorm",
        "post_attention_layernorm",
        "attention_conv",
        "mlp_conv",
        "self_attn",
        "mlp",
        "kernel_projection",
        "base_kernel",
        "gate_up_proj",
        "down_proj",
    ),
    "srt/layers/logits_processor.py": (
        "LogitsProcessorOutput",
        "next_token_logits",
        "hidden_states",
    ),
    "srt/runtime_context.py": ("get_parallel", "get_spec"),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/triton_backend.py": ("token_to_kv_pool", "req_to_token_pool"),
    "srt/mem_cache/memory_pool.py": ("get_kv_buffer", "req_to_token"),
    "srt/layers/radix_attention.py": ("layer_id", "scaling", "k_scale", "v_scale"),
    "srt/model_executor/forward_batch_info.py": (
        "out_cache_loc",
        "req_pool_indices",
        "seq_lens",
    ),
}

# on by default; SGLANG_AMD_MONO_DRAFT_KERNELS=0 keeps the draft on SGLang's modules (stage 0)
USE_KERNELS = os.environ.get("SGLANG_AMD_MONO_DRAFT_KERNELS", "1") == "1"
# the kernels' reach: 1 = the MLP half as D-MLP launches (the attention half on SGLang's
# modules); 2 (default) = the attention half too (D-pre, D-attn, D-post: four launches a layer)
STAGE = int(os.environ.get("SGLANG_AMD_MONO_DRAFT_STAGE", "2")) if USE_KERNELS else 0
# a timing diagnostic: the routed draft forward returns the normed embeddings (no layer runs):
# the step time's drop is the whole draft forward's cost (the draft tokens are then garbage)
SKIP_LAYERS = os.environ.get("SGLANG_AMD_MONO_DRAFT_SKIP", "0") == "1"
# the context projection (``fc``: the target layers' hidden states -> hidden, 671 MB bf16 replicated,
# one GEMM a step) as each rank's 1 / TP of the output rows and the group's all-gather (0: replicated)
SHARD_FC = os.environ.get("SGLANG_AMD_MONO_DRAFT_FC_SHARD", "1") == "1"


def _mlp_views(layer) -> dict:
    """A draft layer's MLP-half weights as the D-MLP launch reads them, in place."""
    conv = layer.mlp_conv
    return dict(
        w_post=layer.post_attention_layernorm.weight,
        w_kp=conv.kernel_projection.weight,
        base=conv.base_kernel,
        w_gu=layer.mlp.gate_up_proj.weight,
        w_down=layer.mlp.down_proj.weight,
    )


def _attn_views(layer) -> dict:
    """A draft layer's attention-half weights as D-pre / D-post read them, in place."""
    conv, attn = layer.attention_conv, layer.self_attn
    return dict(
        w_in=layer.input_layernorm.weight,
        w_kpa=conv.kernel_projection.weight,
        base_a=conv.base_kernel,
        w_qkv=attn.qkv_proj.weight,
        q_norm=attn.q_norm.weight,
        k_norm=attn.k_norm.weight,
        cos_sin=attn.rotary_emb.cos_sin_cache,
        w_o=attn.o_proj.weight,
    )


def _attn_view_reasons(i: int, views: dict) -> list[str]:
    from sgl_amd_mono.muse_spark.gfx950 import draft_attn_half as DH
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D

    shapes = {
        "w_in": (D.H,), "w_kpa": (D.COEF_ROWS, D.H), "base_a": (2, D.TAPS, D.H), "w_qkv": (DH.QKV_ROWS, D.H),
        "q_norm": (DH.HEAD_DIM,), "k_norm": (DH.HEAD_DIM,), "w_o": (D.H, DH.Q_K),
    }  # fmt: skip
    reasons = []
    for name, want in shapes.items():
        t = views[name]
        if (
            not isinstance(t, torch.Tensor)
            or t.dtype != torch.bfloat16
            or not t.is_contiguous()
            or tuple(t.shape) != want
        ):
            reasons.append(f"draft layer {i} {name}: {getattr(t, 'dtype', None)} {tuple(getattr(t, 'shape', ()))}, want contiguous bf16 {want}")  # fmt: skip
    cs = views["cos_sin"]
    if (
        not isinstance(cs, torch.Tensor)
        or cs.dtype != torch.float32
        or not cs.is_contiguous()
        or cs.dim() != 2
        or cs.shape[1] != DH.HEAD_DIM
    ):
        reasons.append(f"draft layer {i} rotary table: {getattr(cs, 'dtype', None)} {tuple(getattr(cs, 'shape', ()))}, D-pre reads fp32 [*, {DH.HEAD_DIM}]")  # fmt: skip
    return reasons


def _mlp_view_reasons(i: int, views: dict) -> list[str]:
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D

    shapes = {
        "w_post": (D.H,), "w_kp": (D.COEF_ROWS, D.H), "base": (2, D.TAPS, D.H), "w_gu": (D.GU_ROWS, D.H), "w_down": (D.H, D.I),
    }  # fmt: skip
    reasons = []
    for name, t in views.items():
        if (
            not isinstance(t, torch.Tensor)
            or t.dtype != torch.bfloat16
            or not t.is_contiguous()
            or tuple(t.shape) != shapes[name]
        ):
            reasons.append(f"draft layer {i} {name}: {getattr(t, 'dtype', None)} {tuple(getattr(t, 'shape', ()))}, want contiguous bf16 {shapes[name]}")  # fmt: skip
    return reasons


class DraftRunner:
    """One rank's D-MLP launches: the layers' weights, the scratch, the peer
    buffer, the step epoch, one launch a (layer, rows)."""

    def __init__(self, layers, rank: int, npes: int, cpu_group, device):
        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D
        from sgl_amd_mono.muse_spark.gfx950.config import MAX_TOKENS
        from sgl_amd_mono.muse_spark.gfx950.layout import scratch_bytes
        from sgl_amd_mono.muse_spark.gfx950.runner import _ensure_writable_flydsl_cache

        _ensure_writable_flydsl_cache()
        self.views = [_mlp_views(layer) for layer in layers]
        self.rank, self.npes, self.device = rank, npes, torch.device(device)
        dev = self.device
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
        self.scratch = torch.zeros(
            scratch_bytes(D.dm_scratch_layout()), dtype=torch.uint8, device=dev
        )
        # the peer handshake is collective: every rank gets here (attach agreed first)
        self.peers = PeerBuffer(
            D.dm_sym_bytes(npes) + guard.FLAG_BYTES, cpu_group, rank, npes, dev
        )
        self.peers.bytes.zero_()
        self.res_out = torch.empty(MAX_TOKENS, D.H, dtype=torch.bfloat16, device=dev)
        self.out = torch.empty(MAX_TOKENS, D.H, dtype=torch.bfloat16, device=dev)
        self._launch: dict = {}
        self.layers = list(layers)
        if STAGE >= 2:
            from sgl_amd_mono.muse_spark.gfx950 import draft_attn as DA
            from sgl_amd_mono.muse_spark.gfx950 import draft_attn_half as DH

            self.aviews = [_attn_views(layer) for layer in layers]
            self.scratch_pre = torch.zeros(
                scratch_bytes(DH.dp_scratch_layout()), dtype=torch.uint8, device=dev
            )
            self.scratch_attn = torch.zeros(scratch_bytes(DA.dattn_scratch_layout(MAX_TOKENS, DRAFT_TOKENS)), dtype=torch.uint8, device=dev)  # fmt: skip
            self.scratch_post = torch.zeros(
                scratch_bytes(DH.do_scratch_layout()), dtype=torch.uint8, device=dev
            )
            self.peers_post = PeerBuffer(
                DH.do_sym_bytes(npes) + guard.FLAG_BYTES, cpu_group, rank, npes, dev
            )
            self.peers_post.bytes.zero_()
            self.q = torch.empty(MAX_TOKENS, DH.Q_K, dtype=torch.bfloat16, device=dev)
            self.o = torch.empty(MAX_TOKENS, DH.Q_K, dtype=torch.bfloat16, device=dev)
            self.coef = torch.empty(
                MAX_TOKENS, D.COEF_ROWS, dtype=torch.float32, device=dev
            )
            self.attn_out = torch.empty(
                MAX_TOKENS, D.H, dtype=torch.bfloat16, device=dev
            )
            self.res_pre = torch.empty(
                MAX_TOKENS, D.H, dtype=torch.bfloat16, device=dev
            )
            self._pre: dict = {}
            self._attn: dict = {}
            self._post: dict = {}
            self._cache_checked = False

    def set_deadline(self, seconds: float) -> None:
        from sgl_amd_mono.common import guard

        ticks = guard.deadline_ticks(float(seconds))
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def kernel(self, rows: int):
        if rows not in self._launch:
            from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D

            self._launch[rows] = D.build_dmlp(D.DMLPBuild(tokens=rows, npes=self.npes))
        return self._launch[rows]

    def attn_kernels(self, rows: int):
        from sgl_amd_mono.muse_spark.gfx950 import draft_attn as DA
        from sgl_amd_mono.muse_spark.gfx950 import draft_attn_half as DH

        if rows not in self._pre:
            self._pre[rows] = DH.build_dpre(DH.DPreBuild(tokens=rows))
            self._attn[rows] = DA.build_dattn(
                DA.DAttnBuild(tokens=rows, q_len=DRAFT_TOKENS)
            )
            self._post[rows] = DH.build_dpost(
                DH.DPostBuild(tokens=rows, npes=self.npes)
            )
        return self._pre[rows], self._attn[rows], self._post[rows]

    def precompile(self, rows: int) -> None:
        from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

        compile_only(self.kernel(rows))
        if STAGE >= 2:
            for k in self.attn_kernels(rows):
                compile_only(k)

    def _backend(self):
        """The draft's attention backend (the forward context's: the draft runner is forwarding),
        which holds its KV pool and its request-to-token table."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        return get_attn_backend()

    def _check_cache(self, backend) -> None:
        """D-pre / D-attn read the draft's fp8 [slots, 1, 128] caches and its int32 page table once."""
        from sgl_amd_mono.muse_spark.gfx950 import draft_attn as DA

        pool = backend.token_to_kv_pool
        for i, layer in enumerate(self.layers):
            k, v = pool.get_kv_buffer(layer.self_attn.attn.layer_id)
            for t, what in ((k, "K"), (v, "V")):
                if not (t.dtype == torch.float8_e4m3fn and t.is_contiguous() and t.dim() == 3 and tuple(t.shape[-2:]) == (DA.LOCAL_KV_HEADS, DA.HEAD_DIM)):  # fmt: skip
                    raise RuntimeError(f"Muse Spark draft layer {i} {what} cache: {t.dtype} {tuple(t.shape)}, the kernels take contiguous fp8 [*, {DA.LOCAL_KV_HEADS}, {DA.HEAD_DIM}]")  # fmt: skip
            a = layer.self_attn.attn
            if a.k_scale is not None or a.v_scale is not None:
                raise RuntimeError(
                    f"Muse Spark draft layer {i}: a K / V cache scale; the kernels write fp8 at unit scale"
                )
        table = backend.req_to_token_pool.req_to_token
        if table.dtype != torch.int32 or table.dim() != 2 or not table.is_contiguous():
            raise RuntimeError(
                f"Muse Spark draft: page table {table.dtype} {tuple(table.shape)}, the kernels take a contiguous int32 [pool, ctx]"
            )
        self._cache_checked = True

    def launch_attn(
        self,
        i: int,
        rows: int,
        hidden,
        residual,
        first: bool,
        positions,
        forward_batch,
        stream,
        trace: bool = False,
    ):
        """Layer ``i``'s attention half on ``rows`` rows (D-pre, D-attn, D-post): (attn_out,
        the residual after the input norm's fused add), bf16. ``trace`` (check mode):
        every launch synchronized and logged, so a hang names its launch."""
        from sgl_amd_mono.muse_spark.gfx950.runner import _f32_bits, _ptr

        def mark(what):
            if trace:
                torch.cuda.synchronize()
                logger.info("monokernel check: draft layer %d %s done (seq_lens %s, slots %s, table %s)", i, what, lens.tolist()[:4], slots.tolist()[:4], tuple(table.shape))  # fmt: skip

        backend = self._backend()
        if not self._cache_checked:
            self._check_cache(backend)
        v = self.aviews[i]
        attn = self.layers[i].self_attn.attn
        k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(attn.layer_id)
        slots = forward_batch.out_cache_loc
        table = backend.req_to_token_pool.req_to_token
        rpi = forward_batch.req_pool_indices
        lens = forward_batch.seq_lens
        if (
            slots.numel() < rows
            or positions.numel() < rows
            or lens.numel() * DRAFT_TOKENS < rows
        ):
            raise RuntimeError(f"Muse Spark draft: slots {tuple(slots.shape)} positions {tuple(positions.shape)} seq_lens {tuple(lens.shape)} for {rows} rows")  # fmt: skip
        pre, dattn, post = self.attn_kernels(rows)
        res_pre, q, o, coef, attn_out = (
            self.res_pre[:rows],
            self.q[:rows],
            self.o[:rows],
            self.coef[:rows],
            self.attn_out[:rows],
        )
        pre(
            _ptr(hidden), _ptr(residual), int(first), _ptr(v["w_in"]), _ptr(v["w_kpa"]), _ptr(v["base_a"]), _ptr(v["w_qkv"]),
            _ptr(v["q_norm"]), _ptr(v["k_norm"]), _ptr(v["cos_sin"]), _ptr(positions), int(positions.dtype == torch.int64),
            _ptr(slots), int(slots.dtype == torch.int64), _ptr(k_cache), _ptr(v_cache), _ptr(res_pre), _ptr(q), _ptr(coef),
            _ptr(self.scratch_pre), _ptr(self.epoch), i, stream=stream,
        )  # fmt: skip
        mark("D-pre")
        # the draft batch is a TARGET_VERIFY batch: seq_lens is the (compact) prefix, the
        # block's DRAFT_TOKENS keys follow it; the compact cache holds the window, so no
        # window mask (the window is the cache)
        dattn(
            _ptr(q), _ptr(o), _ptr(k_cache), _ptr(v_cache), _ptr(table), table.shape[1], table.numel() * 4, _ptr(rpi),
            int(rpi.dtype == torch.int64), _ptr(lens), int(lens.dtype == torch.int64), DRAFT_TOKENS, 0, 0,
            _f32_bits(float(attn.scaling)), _ptr(self.scratch_attn), _ptr(self.epoch), i, stream=stream,
        )  # fmt: skip
        mark("D-attn")
        post(
            _ptr(o), _ptr(v["w_o"]), _ptr(coef), _ptr(v["base_a"]), _ptr(attn_out), _ptr(self.scratch_post),
            self.peers_post.local, _ptr(self.peers_post.addresses), self.rank, _ptr(self.epoch), i, stream=stream,
        )  # fmt: skip
        mark("D-post")
        return attn_out, res_pre

    def launch(
        self, i: int, rows: int, attn_out: torch.Tensor, residual: torch.Tensor, stream
    ):
        """Layer ``i``'s MLP half on ``rows`` rows: (out, the updated residual), bf16."""
        from sgl_amd_mono.muse_spark.gfx950.runner import _ptr

        v = self.views[i]
        res_out, out = self.res_out[:rows], self.out[:rows]
        self.kernel(rows)(
            _ptr(residual), _ptr(attn_out), _ptr(v["w_post"]), _ptr(v["w_kp"]), _ptr(v["base"]), _ptr(v["w_gu"]), _ptr(v["w_down"]),
            _ptr(res_out), _ptr(out), _ptr(self.scratch), self.peers.local, _ptr(self.peers.addresses), self.rank,
            _ptr(self.epoch), i, stream=stream,
        )  # fmt: skip
        return out, res_out

    def end_step(self) -> None:
        self.epoch[0].add_(1)


def _tokens_per_request(model) -> int:
    """A draft block's rows a request: the block size (the anchor and the drafts)."""
    return int(model.block_size)


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    reasons = []
    for k, want in DRAFT_CONFIG.items():
        got = getattr(hf_config, k, None)
        if got != want:
            reasons.append(f"draft {k} {got}: the kernels take {want}")
    dc = getattr(hf_config, "dflash_config", None) or {}
    for k, want in DFLASH_CONFIG.items():
        got = dc.get(k) if isinstance(dc, dict) else getattr(dc, k, None)
        if got != want:
            reasons.append(f"draft dflash_config.{k} {got}: the kernels take {want}")
    types = getattr(hf_config, "layer_types", None)
    if types is not None and any(t != "sliding_attention" for t in types):
        reasons.append(
            f"draft layer_types {types}: the kernels take sliding layers only"
        )
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel

    reasons = []
    if get_parallel().tp_size != TP:
        reasons.append(f"draft TP {get_parallel().tp_size}: the kernels take TP {TP}")
    if USE_KERNELS:
        for i, layer in enumerate(model.layers):
            if layer.mlp_conv is not None and layer.attention_conv is not None:
                reasons += _mlp_view_reasons(i, _mlp_views(layer))
                if STAGE >= 2:
                    reasons += _attn_view_reasons(i, _attn_views(layer))
    if STAGE >= 2:
        from sglang.srt.runtime_context import get_spec

        if get_spec().speculative_draft_window_size is None:
            reasons.append(
                "draft stage 2 takes the compact (windowed) draft cache: set --speculative-draft-window-size"
            )
    if int(model.block_size) != DRAFT_TOKENS:
        reasons.append(
            f"draft block {model.block_size}: the kernels take {DRAFT_TOKENS}"
        )
    if getattr(model, "candidate_selector", None) is None:
        reasons.append("the draft has no candidate selector (not a DFlash2 draft)")
    if len(model.layers) != DRAFT_CONFIG["num_hidden_layers"]:
        reasons.append(f"draft layers {len(model.layers)}")
    for i, layer in enumerate(model.layers):
        if layer.attention_conv is None or layer.mlp_conv is None:
            reasons.append(f"draft layer {i}: no grouped convolutions")
        a = layer.self_attn
        if a.num_heads * TP != DRAFT_CONFIG["num_attention_heads"] or a.num_kv_heads * TP != DRAFT_CONFIG["num_key_value_heads"]:  # fmt: skip
            reasons.append(
                f"draft layer {i}: heads {a.num_heads} / {a.num_kv_heads} a rank"
            )
        if a.v_scale is not None or a.attention_sink_bias is not None:
            reasons.append(f"draft layer {i}: a V scale or attention sinks")
    return reasons


class _Draft:
    """One rank's draft: stage 0 runs the model's own forward; stage 1 runs each
    layer's attention half on SGLang's modules and its MLP half as the D-MLP launch."""

    def __init__(self, model, original, runner=None):
        self.model, self.original, self.runner = model, original, runner
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1"

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        *args,
        **kwargs,
    ):
        """``DFlash2DraftModel.forward`` for a routed draft step."""
        if self.runner is None:
            return self.original(
                input_ids,
                positions,
                forward_batch,
                input_embeds=input_embeds,
                *args,
                **kwargs,
            )
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        m = self.model
        if input_embeds is None:
            # the worker embeds the block with the target's table (a DFlash2 draft has none)
            raise ValueError("the Muse Spark DFlash2 draft forward needs input_embeds")
        rows = int(plan.key.rows)
        hidden, residual = input_embeds, None
        if SKIP_LAYERS:
            return LogitsProcessorOutput(
                next_token_logits=None, hidden_states=m.norm(hidden)
            )
        stream = torch.cuda.current_stream()
        check = self.check and not torch.cuda.is_current_stream_capturing()
        for i, layer in enumerate(m.layers):
            if STAGE >= 2:
                first = residual is None
                h_in, r_in = hidden, (hidden if first else residual)
                attn_out, residual = self.runner.launch_attn(
                    i,
                    rows,
                    h_in.contiguous(),
                    r_in.contiguous(),
                    first,
                    positions,
                    forward_batch,
                    stream,
                    trace=check,
                )
                if check:
                    _log_check_pair(i, "attn", (attn_out, residual), self._attention_reference(layer, h_in, None if first else r_in, positions, forward_batch))  # fmt: skip
            else:
                # the attention half, as DFlashDecoderLayer.forward runs it
                if residual is None:
                    residual = hidden
                    hidden = layer.input_layernorm(hidden)
                else:
                    hidden, residual = layer.input_layernorm(hidden, residual)
                hidden, kernel = layer.attention_conv.prepare(hidden)
                attn_out = layer.self_attn(
                    positions=positions,
                    hidden_states=hidden,
                    forward_batch=forward_batch,
                )
                attn_out = layer.attention_conv.finish(attn_out, kernel)
            ref = self._reference(layer, attn_out, residual) if check else None
            hidden, residual = self.runner.launch(
                i, rows, attn_out.contiguous(), residual.contiguous(), stream
            )
            if ref is not None:
                _log_check(i, (hidden, residual), ref)
        self.runner.end_step()
        hidden, _ = m.norm(hidden, residual)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)

    @staticmethod
    def _attention_reference(layer, hidden, residual, positions, forward_batch):
        """Check mode: SGLang's attention half on copies of the launch's inputs (its
        attention also rewrites the block's K / V cache rows with the same values)."""
        h = hidden.clone()
        if residual is None:
            res = h
            h = layer.input_layernorm(h)
        else:
            h, res = layer.input_layernorm(h, residual.clone())
        h, kernel = layer.attention_conv.prepare(h)
        ao = layer.self_attn(
            positions=positions, hidden_states=h, forward_batch=forward_batch
        )
        return layer.attention_conv.finish(ao, kernel), res

    @staticmethod
    def _reference(layer, attn_out, residual):
        """Check mode: SGLang's MLP half on copies of the launch's inputs."""
        h, res = layer.post_attention_layernorm(attn_out.clone(), residual.clone())
        h, kernel = layer.mlp_conv.prepare(h)
        h = layer.mlp(h)
        return layer.mlp_conv.finish(h, kernel), res


def _log_check_pair(i: int, what: str, got, ref) -> None:
    for name, g, r in ((what, got[0], ref[0]), ("residual'", got[1], ref[1])):
        g, r = g.float(), r.float()
        cos = torch.nn.functional.cosine_similarity(g, r, dim=-1)
        rel = ((g - r).norm() / r.norm().clamp_min(1e-6)).item()
        logger.info("monokernel check: draft layer %d %s: rel %.4f, cosine min %.5f mean %.5f", i, name, rel, cos.min().item(), cos.mean().item())  # fmt: skip


def _log_check(i: int, got, ref) -> None:
    for name, g, r in (("out", got[0], ref[0]), ("residual", got[1], ref[1])):
        g, r = g.float(), r.float()
        cos = torch.nn.functional.cosine_similarity(g, r, dim=-1)
        rel = ((g - r).norm() / r.norm().clamp_min(1e-6)).item()
        logger.info("monokernel check: draft layer %d %s: rel %.4f, cosine min %.5f mean %.5f", i, name, rel, cos.min().item(), cos.mean().item())  # fmt: skip


def _shard_fc(model) -> bool:
    """Replace ``model.project_target_hidden`` with the TP-sharded projection: the rank's
    rows of ``fc.weight`` (a view: the loader's storage) against the replicated input,
    the ranks' parts all-gathered in rank order, then ``hidden_norm`` as the model does.
    The draft's context rows only (the target's distribution is the verify's); a call
    that does not match (another TP size or shape) runs the model's own."""
    fc = getattr(model, "fc", None)
    w = getattr(fc, "weight", None)
    if (
        w is None
        or w.dtype != torch.bfloat16
        or w.dim() != 2
        or w.shape[0] % TP
        or getattr(fc, "bias", None) is not None
    ):
        return False
    original = model.project_target_hidden

    def project_target_hidden(target_hidden: torch.Tensor) -> torch.Tensor:
        from sglang.srt.runtime_context import get_parallel

        p = get_parallel()
        if (
            p.tp_size != TP
            or target_hidden.ndim != 2
            or target_hidden.shape[-1] != w.shape[1]
        ):
            return original(target_hidden)
        n = w.shape[0] // TP
        part = torch.nn.functional.linear(
            target_hidden.to(w.dtype), w.narrow(0, p.tp_rank * n, n)
        )
        return model.hidden_norm(p.tp_group.all_gather(part, dim=-1))

    model.project_target_hidden = project_target_hidden
    return True


def _build(model, hf_config, ctx) -> _Draft:
    original = type(model).forward.__get__(model, type(model))
    if not USE_KERNELS:
        logger.info(
            "monokernel: muse_spark DFlash2 draft attached at stage 0 (the model's own forward)"
        )
        return _Draft(model, original)
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    runner = DraftRunner(
        list(model.layers),
        p.tp_rank,
        p.tp_size,
        p.tp_group.cpu_group,
        model.norm.weight.device,
    )
    runner.set_deadline(ctx.deadline_s)
    widths = list(ctx.widths)
    start = len(widths) * p.tp_rank // max(p.tp_size, 1)
    for rows in widths[start:] + widths[:start]:
        runner.precompile(rows)
    if SHARD_FC and p.tp_size == TP and _shard_fc(model):
        logger.info("monokernel: muse_spark DFlash2 draft: the context projection TP-sharded (%s rows a rank)", model.fc.weight.shape[0] // TP)  # fmt: skip
    logger.info("monokernel: muse_spark DFlash2 draft attached at stage %d (%s)", STAGE, "the MLP half as D-MLP launches" if STAGE < 2 else "D-pre, D-attn, D-post and D-MLP launches a layer")  # fmt: skip
    return _Draft(model, original, runner)


SPEC = MonoSpec(
    name="muse_spark_dflash2_draft_hip",
    devices={
        "gfx950": DeviceLimits(widths=WIDTHS, tp_sizes=(TP,), resident_blocks=BLOCKS)
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    # the launcher checks the target's requirements; the draft's attention backend is its own
    requires=Requirements(
        attention_backends=frozenset({"triton"}), hf_config={"model_type": "qwen3"}
    ),
    numerics=target.NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
    tokens_per_request=_tokens_per_request,
)
