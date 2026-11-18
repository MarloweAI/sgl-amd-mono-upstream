# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 MoE on gfx950: the monokernel spec and its forward (RFC
sgl-project/sglang#43949).

A decode step runs each decoder layer as at most two launches:

- K1 (``gfx950/gdn.py``, GDN layers): input norm, in_proj_qkvz and in_proj_ba,
  the conv and the gated delta rule with their state updates, and the gated norm.
  It writes the core output (o_proj's input) and the residual.
- K2 (``gfx950/layer.py``, every layer): o_proj and its TP reduce, the
  post-attention norm, the router, the shared and routed MXFP4 experts and the MoE
  TP reduce. It writes the layer output and the residual.

A full-attention layer runs SGLang's attention up to o_proj's input, then K2. A
target-verify step (speculative decoding) runs K2 only: K1 keeps no state per
draft token for the rollback, so a GDN layer runs SGLang's GDN up to out_proj's
input there. SGLang's embedding, final norm and logits processor run around the
layers.

The kernels read SGLang's weights in place at each launch (SGLang repacks
in_proj_qkvz after the build). The fused shared expert
(``--enforce-shared-experts-fusion``, MXFP4 with sgl-project/sglang#41899) runs K2
``fse``: the shared expert is routed expert 512. The spec refuses the MXFP4 GDN
projections (``SGLANG_ROCM_GDN_ONLINE_MXFP4``) at startup.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

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

from sgl_amd_mono.qwen38_a95b_mxfp4 import HF_CONFIG
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layout as L

logger = logging.getLogger(__name__)

# a launch takes 1 to MAX_TOKENS rows: decode requests, or verify tokens
WIDTHS = tuple(range(1, L.MAX_TOKENS + 1))
# the mailbox tags a launch writes: K1 at slot i, K2 at TAG_SLOTS // 2 + i
TAG_SLOTS = 256
MAX_LAYERS = TAG_SLOTS // 2

# The variant the kernels do not run yet, refused once a model
PENDING_MXFP4 = (
    "the kernels do not run the MXFP4 GDN projections yet; "
    "start with SGLANG_ROCM_GDN_ONLINE_MXFP4=0"
)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "enable_linear_replayssm",
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
    ),
    "srt/configs/mamba_utils.py": ("mamba2_state_dtype",),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "linear_attn_backend",
        "forward_metadata",
        "state_layer_cache",
    ),
    "srt/layers/attention/mamba/mamba2_metadata.py": ("mamba_cache_indices",),
    "kernels/ops/attention/fla/layernorm_gated.py": (
        "eps",
        "group_size",
        "norm_before_gate",
        "activation",
    ),
    "srt/layers/layernorm.py": ("variance_epsilon",),
    "srt/mem_cache/memory_pool.py": (
        "conv",
        "temporal",
        "replayssm_d",
    ),
    "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py": (
        "is_shuffled",
        '"w13_weight"',
        '"w2_weight"',
        '"w13_weight_scale"',
        '"w2_weight_scale"',
    ),
    "srt/model_executor/forward_batch_info.py": ("mrope_positions",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/models/qwen2_moe.py": (
        "Qwen2MoeSparseMoeBlock",
        "gate",
        "experts",
        "shared_expert",
        "shared_expert_gate",
        "num_fused_shared_experts",
        "gate_up_proj",
        "down_proj",
    ),
    "srt/models/qwen3_5.py": (
        "embed_tokens",
        "layers",
        "start_layer",
        "end_layer",
        "norm",
        "layers_to_capture",
        "input_layernorm",
        "post_attention_layernorm",
        "mlp",
        "linear_attn",
        "in_proj_qkvz",
        "in_proj_ba",
        "conv1d",
        "A_log",
        "dt_bias",
        "out_proj",
        "activation",
        "layer_id",
        "self_attention",
        "o_proj",
        "q_size",
        "attn_output_gate",
    ),
    "srt/models/qwen3_5_text.py": (
        "model",
        "lm_head",
        "logits_processor",
        "is_mrope_enabled",
    ),
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        "get_spec",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
}

# K1 and K2 against SGLang's layer on the same input and state (check.py,
# SGLANG_AMD_MONO_CHECK). K2 quantizes the expert inputs to MXFP4 as SGLang's
# AITER MoE does but sums in another order, so the layer output differs by about
# the MoE's rounding error. The k2.out bound holds on the rows that K2 and
# SGLang route alike (check.k2_alike).
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "k1.core.cosine": 0.9999,
        "k1.conv.cosine": 0.9999,
        "k1.state.cosine": 0.9999,
        "k2.out.cosine": 0.99,
    },
)


def _text(config):
    return getattr(config, "text_config", config)


def _is_gdn(layer) -> bool:
    return hasattr(layer, "linear_attn")


def _mxfp4(linear) -> bool:
    """An MXFP4 projection: fp4x2 bytes and their e8m0 scales."""
    return (
        linear.weight.dtype == torch.uint8
        and getattr(linear, "weight_scale", None) is not None
    )


def _fse(mlp) -> bool:
    return bool(getattr(mlp, "num_fused_shared_experts", 0))


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype

    text = _text(hf_config)
    reasons = [
        f"needs Qwen3.8's {key} = {want!r}, got {getattr(text, key, None)!r}"
        for key, want in HF_CONFIG.items()
        if getattr(text, key, None) != want
    ]
    if not getattr(text, "norm_topk_prob", True):
        reasons.append("needs norm_topk_prob: the router renormalizes the top-k")
    layers = getattr(text, "num_hidden_layers", 0)
    if layers > MAX_LAYERS:
        reasons.append(f"{layers} layers: the mailbox tags hold {MAX_LAYERS}")
    dtypes = mamba2_state_dtype(hf_config)
    if dtypes.temporal not in (torch.float32, torch.bfloat16):
        reasons.append(f"needs the SSM state in f32 or bf16, got {dtypes.temporal}")
    if dtypes.conv != torch.bfloat16:
        reasons.append(f"needs the conv state in bf16, got {dtypes.conv}")
    if cfg.speculative_algorithm is None:
        if cfg.enable_linear_replayssm:
            reasons.append(
                "--enable-linear-replayssm: K1 updates the SSM state in place; "
                "ReplaySSM defers it to a ring"
            )
    else:
        topk = cfg.speculative_eagle_topk or 1
        tokens, steps = cfg.speculative_num_draft_tokens, cfg.speculative_num_steps
        if topk != 1 or tokens != (steps or 0) + 1:
            reasons.append(
                "the verify step takes one draft chain: needs "
                "--speculative-eagle-topk 1 and --speculative-num-draft-tokens = "
                f"--speculative-num-steps + 1, got topk {topk}, {tokens} tokens, "
                f"{steps} steps"
            )
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _bf16(*tensors) -> bool:
    return all(t.dtype == torch.bfloat16 for t in tensors)


def _layer_reasons(layer, eps: tuple) -> list[str]:
    """Why the kernels cannot run one decoder layer as SGLang loaded it."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    mlp = layer.mlp
    if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return [f"needs the sparse MoE block, got {type(mlp).__name__}"]
    reasons = []
    se = mlp.shared_expert
    if se is None and not _fse(mlp):
        reasons.append("needs the shared expert")
    elif se is not None and not _bf16(se.gate_up_proj.weight, se.down_proj.weight):
        reasons.append("needs the shared expert in bf16")
    ex = mlp.experts
    e = L.E + _fse(mlp)
    if not getattr(ex.w13_weight, "is_shuffled", False):
        reasons.append("needs the routed experts in AITER's shuffled layout")
    if ex.w13_weight.dtype != torch.uint8 or ex.w2_weight.dtype != torch.uint8:
        reasons.append("needs MXFP4 routed experts")
    if tuple(ex.w13_weight.shape) != (e, 2 * L.RI, L.HIDDEN // 2):
        reasons.append(
            f"routed w13 shape {tuple(ex.w13_weight.shape)}: the kernels take the "
            f"experts TP-sharded, unpadded"
        )
    if not _bf16(
        layer.input_layernorm.weight,
        layer.post_attention_layernorm.weight,
        mlp.gate.weight,
        mlp.shared_expert_gate.weight,
    ):
        reasons.append("needs the norms, router and shared-expert gate in bf16")
    if _is_gdn(layer):
        a = layer.linear_attn
        o = a.out_proj
        if _mxfp4(a.in_proj_qkvz) or _mxfp4(o):
            reasons.append(PENDING_MXFP4)
        elif not _bf16(a.in_proj_qkvz.weight, o.weight):
            reasons.append("needs in_proj_qkvz and out_proj in bf16")
        if not _bf16(a.in_proj_ba.weight, a.conv1d.weight, a.dt_bias, a.norm.weight):
            reasons.append("needs the GDN in_proj_ba, conv, dt_bias and norm in bf16")
        if a.A_log.dtype != torch.float32:
            reasons.append("needs A_log in f32")
        if a.activation != "silu" or a.conv1d.bias is not None:
            reasons.append("needs the GDN conv without bias and with silu")
        norm = a.norm
        if norm.activation not in ("silu", "swish") or norm.group_size is not None:
            reasons.append("needs the GDN gated norm over the head with silu")
        if not norm.norm_before_gate:
            reasons.append("needs the GDN norm before the gate")
        layer_eps = (layer.input_layernorm.variance_epsilon, norm.eps)
    else:
        o = layer.o_proj
        if not layer.attn_output_gate or layer.q_size != L.CORE:
            reasons.append("needs the gated attention output with one rank's heads")
        if not _bf16(o.weight):
            reasons.append("needs o_proj in bf16")
        layer_eps = (layer.input_layernorm.variance_epsilon, None)
    if not _mxfp4(o) and tuple(o.weight.shape) != (L.HIDDEN, L.CORE):
        reasons.append(f"o_proj shape {tuple(o.weight.shape)}")
    want = (eps[0], eps[1] if _is_gdn(layer) else None)
    if layer_eps != want or layer.post_attention_layernorm.variance_epsilon != eps[2]:
        reasons.append("its norm eps differ from the first layer's")
    return reasons


def _eps(layers) -> tuple:
    """(input norm, GDN gated norm, post-attention norm) eps: one for all layers."""
    gdn = next((x.linear_attn for x in layers if _is_gdn(x)), None)
    return (
        layers[0].input_layernorm.variance_epsilon,
        None if gdn is None else gdn.norm.eps,
        layers[0].post_attention_layernorm.variance_epsilon,
    )


def _layers(model) -> list:
    m = model.model
    return [m.layers[i] for i in range(m.start_layer, m.end_layer)]


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    reasons = []
    layers = _layers(model)
    eps = _eps(layers)
    pending = []
    for layer in layers:
        for why in _layer_reasons(layer, eps):
            if why == PENDING_MXFP4:
                pending.append(why)
            else:
                reasons.append(f"layer {layer.layer_id}: {why}")
    return list(dict.fromkeys(pending)) + reasons


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x, None


_IDENTITY = _Identity()


@contextmanager
def _proj_input(module, name: str):
    """``module.<name>`` returns its input, so SGLang's attention or GDN returns
    o_proj's input. ponytail: a module swap around SGLang's call; the upgrade is a
    core / projection split in SGLang's Qwen3.5 layers."""
    proj = getattr(module, name)
    setattr(module, name, _IDENTITY)
    try:
        yield
    finally:
        setattr(module, name, proj)


class _Built:
    """One rank's launches: the build keys per (width, variant), the scratch per
    width, the guard block (word 0 the step epoch) and the peer buffer."""

    def __init__(
        self,
        lm,
        rank,
        state_bf16,
        k1_keys,
        k2_keys,
        k1_scratch,
        k2_scratch,
        epoch,
        peer,
    ):
        self.lm, self.rank = lm, rank
        self.state_dtype = torch.bfloat16 if state_bf16 else torch.float32
        self.k1_keys, self.k2_keys = k1_keys, k2_keys
        self.k1_scratch, self.k2_scratch = k1_scratch, k2_scratch
        self.epoch, self.peer = epoch, peer
        self.check_layers = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
        **kwargs,
    ):
        lm = self.lm
        # SGLang sets layers_to_capture (EAGLE3 / DFlash targets) after the build,
        # and the launches keep no auxiliary hidden states
        if (
            input_embeds is not None
            or pp_proxy_tensors is not None
            or kwargs
            or lm.model.layers_to_capture
        ):
            return type(lm).forward(
                lm,
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                pp_proxy_tensors,
                **kwargs,
            )
        if lm.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        m = lm.model
        s = plan.key.rows
        verify = plan.key.forward_mode == "target_verify"
        fb = (
            forward_batch
            if self.check_layers and not torch.cuda.is_current_stream_capturing()
            else None
        )
        self.epoch[:1].add_(1)
        h = m.embed_tokens(input_ids)
        residual = None
        idx = None if verify else self._state_indices(s)
        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            if _is_gdn(layer) and not verify:
                core, res_attn = self._k1(i, layer, h, residual, idx, s, fb)
            else:
                if residual is None:
                    res_attn, x = h, layer.input_layernorm(h)
                else:
                    x, res_attn = layer.input_layernorm(h, residual)
                core = self._core(layer, positions, x, forward_batch)
            h, residual = self._k2(i, layer, core, res_attn, s, fb)
        h, _ = m.norm(h, residual)
        return lm.logits_processor(input_ids, h, lm.lm_head, forward_batch, None)

    @staticmethod
    def _core(layer, positions, x, forward_batch):
        if _is_gdn(layer):
            with _proj_input(layer.linear_attn, "out_proj"):
                return layer.linear_attn(x, forward_batch).contiguous()
        with _proj_input(layer, "o_proj"):
            return layer.self_attention(positions, x, forward_batch).contiguous()

    @staticmethod
    def _state_indices(s):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        md = get_attn_backend().linear_attn_backend.forward_metadata
        idx = md.mamba_cache_indices
        if idx.dtype != torch.int32 or not idx.is_contiguous() or idx.numel() < s:
            raise RuntimeError(
                f"Qwen3.8 monokernel: K1 needs {s} contiguous int32 state indices, "
                f"got {idx.dtype} {tuple(idx.shape)}"
            )
        return idx

    def _state(self, a):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        st = get_attn_backend().linear_attn_backend.state_layer_cache(a.layer_id)
        conv, temporal = st.conv[0], st.temporal
        want = self.state_dtype
        if (
            getattr(st, "replayssm_d", None) is not None
            or temporal.dtype != want
            or conv.dtype != torch.bfloat16
            or tuple(conv.shape[1:]) != (L.CONV, L.CONV_W - 1)
            or not temporal[0].is_contiguous()
        ):
            raise RuntimeError(
                f"Qwen3.8 monokernel: layer {a.layer_id}: K1 needs the {want} SSM "
                f"state and the bf16 (slots, {L.CONV}, {L.CONV_W - 1}) conv state, "
                f"got {temporal.dtype} {tuple(temporal.shape)} and {conv.dtype} "
                f"{tuple(conv.shape)}"
            )
        return conv, temporal

    def _k1(self, i, layer, h, residual, idx, s, check_batch=None):
        """``check_batch``: the forward batch in check mode, else None."""
        from sgl_amd_mono.qwen38_a95b_mxfp4 import check
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import gdn

        a = layer.linear_attn
        conv, temporal = self._state(a)
        if check_batch is not None:
            ref = check.k1_reference(
                layer, h, residual, conv, temporal, idx[:s], check_batch
            )
        core = torch.empty(s, L.CORE, dtype=h.dtype, device=h.device)
        res_out = torch.empty_like(h)
        gdn.gdn_pre(
            self.k1_keys[s, residual is None],
            hidden=h,
            residual=residual,
            res_out=res_out,
            ln_w=layer.input_layernorm.weight,
            w_qkvz=a.in_proj_qkvz.weight,
            w_ba=a.in_proj_ba.weight,
            conv_w=a.conv1d.weight,
            conv_state=conv,
            a_log=a.A_log,
            dt_bias=a.dt_bias,
            norm_w=a.norm.weight,
            rstate=temporal,
            st_idx=idx,
            core=core,
            scratch=self.k1_scratch[s],
            epoch=self.epoch,
            layer=i,
        )
        if check_batch is not None:
            got = check.k1_outputs(core, res_out, conv, temporal, idx[:s])
            check.compare("k1", layer.layer_id, s, ref, got)
        return core, res_out

    def _k2(self, i, layer, core, residual, s, check_batch=None):
        from sgl_amd_mono.qwen38_a95b_mxfp4 import check
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer as k2

        mlp = layer.mlp
        o = layer.linear_attn.out_proj if _is_gdn(layer) else layer.o_proj
        if check_batch is not None:
            ref, ids, logits = check.k2_reference(layer, o, core, residual, check_batch)
        fse = _fse(mlp)
        se, ex = mlp.shared_expert, mlp.experts
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        k2.layer_post(
            self.k2_keys[s, fse],
            core=core,
            residual=residual,
            w_o=o.weight,
            ln_w=layer.post_attention_layernorm.weight,
            w_gate=mlp.gate.weight,
            w_sg=mlp.shared_expert_gate.weight,
            w13=ex.w13_weight,
            w13s=ex.w13_weight_scale,
            w2=ex.w2_weight,
            w2s=ex.w2_weight_scale,
            out=out,
            res_out=res_out,
            scratch=self.k2_scratch[s],
            peers=self.peer.addresses,
            rank=self.rank,
            epoch=self.epoch,
            layer=TAG_SLOTS // 2 + i,
            w_sgu=None if se is None else se.gate_up_proj.weight,
            w_sd=None if se is None else se.down_proj.weight,
        )
        if check_batch is not None:
            # ROUTE: (value, tag) pairs; token t's experts at 2 k t, then its weights
            off, n = k2.scratch_layout(self.k2_keys[s, fse])["route"]
            route = (
                self.k2_scratch[s][off : off + n].view(torch.int32).view(s, 2, -1, 2)
            )
            alike = check.k2_alike(layer.layer_id, ids, logits, route[:, 0, :, 0])
            got = {"out": out, "res_out": res_out}
            check.compare("k2", layer.layer_id, s, ref, got, alike)
        return out, res_out


def _compile_only(launch) -> None:
    """Compile a launch for this process without running it (FlyDSL's compile-only
    mode, for the one call); the first real call then only loads the code."""
    from flydsl.utils.env import CompileEnvManager

    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    var = CompileEnvManager.options["compile_only"].env_var
    old = os.environ.get(var)
    os.environ[var] = "1"
    try:
        launch(*([0] * len(names)), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old


def _writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache (aiter sets a read-only one)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def _scratch(keys: dict, size, device) -> dict:
    """One zeroed scratch a width, as large as its largest variant's."""
    out = {}
    for key in keys.values():
        out[key.tokens] = max(out.get(key.tokens, 0), size(key))
    return {w: torch.zeros(n, dtype=torch.uint8, device=device) for w, n in out.items()}


def _build(model, hf_config, ctx) -> _Built:
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype
    from sglang.srt.runtime_context import get_parallel, get_spec

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import gdn
    from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer as k2

    _writable_flydsl_cache()
    p = get_parallel()
    tp, rank = p.tp_size, p.tp_rank
    layers = _layers(model)
    device = layers[0].input_layernorm.weight.device
    e1, e_norm, e2 = _eps(layers)
    fault_site, fault_after = guard.fault_injection(rank) or (0, 0)
    # the target model runs no plain decode step under speculative decoding
    verify = get_spec().speculative_algorithm is not None
    gdns = [x for x in layers if _is_gdn(x)]
    bf16 = mamba2_state_dtype(hf_config).temporal == torch.bfloat16
    k1_keys = {}
    if not verify:
        for w in ctx.widths:
            for first in sorted({x is layers[0] for x in gdns}):
                k1_keys[w, first] = gdn.K1Build(
                    tokens=w,
                    first=first,
                    eps=e1,
                    norm_eps=e_norm,
                    state_bf16=bf16,
                    fault_site=fault_site,
                    fault_after=fault_after,
                )
    k2_keys = {
        (w, fse): k2.K2Build(
            tokens=w,
            eps=e2,
            fse=fse,
            fault_site=fault_site,
            fault_after=fault_after,
        )
        for w in ctx.widths
        for fse in sorted({_fse(x.mlp) for x in layers})
    }
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(ctx.deadline_s)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    # collective: every rank builds it at the same point
    group = p.tp_group.cpu_group
    peer = PeerBuffer(k2.peer_buffer_bytes(), group, rank, tp, device)
    peer.bytes.zero_()
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)  # no peer writes before its owner zeroed
    built = _Built(
        model,
        rank,
        bf16,
        k1_keys,
        k2_keys,
        _scratch(k1_keys, gdn.scratch_bytes, device),
        _scratch(k2_keys, k2.scratch_bytes, device),
        epoch,
        peer,
    )
    # Every rank builds every key. FlyDSL's disk cache hands a rank the kernels
    # another rank built (a lock per kernel), so each rank starts at its own share.
    launches = [gdn.build(k) for k in k1_keys.values()]
    launches += [k2.build(k) for k in k2_keys.values()]
    start = len(launches) * rank // tp
    t0 = time.monotonic()
    for launch in launches[start:] + launches[:start]:
        _compile_only(launch)
    torch.cuda.synchronize()
    logger.info(
        "Qwen3.8 monokernel: %d K1 and %d K2 launches (widths %s, %s) in %.0f s",
        len(k1_keys),
        len(k2_keys),
        list(ctx.widths),
        "verify" if verify else "decode",
        time.monotonic() - t0,
    )
    return built


SPEC = MonoSpec(
    name="qwen38_a95b_mxfp4_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=WIDTHS, tp_sizes=(L.TP,), resident_blocks=L.BLOCKS
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={"moe_runner_backend": frozenset({"auto", "aiter"})},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
