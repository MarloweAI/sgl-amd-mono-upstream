# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 MoE on gfx950: the monokernel spec and its forward (RFC
sgl-project/sglang#43949).

A decode step runs each decoder layer's tail as one launch, K2
(``gfx950/layer.py``): o_proj (out_proj on a GDN layer) and its TP reduce, the
post-attention norm, the router, the shared and routed MXFP4 experts and the MoE
TP reduce. It writes the layer output and the residual.

In front of K2, each layer runs SGLang's input norm and its attention or GDN up to
the projection's input. SGLang's embedding, final norm and logits processor run
around the layers.

The kernels read SGLang's weights in place at each launch. The spec refuses the
fused shared expert (``--enforce-shared-experts-fusion``) and the MXFP4 GDN
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
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        DeviceLimits,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_amd_mono.qwen38_a95b_mxfp4 import HF_CONFIG
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layout as L

logger = logging.getLogger(__name__)

# a launch takes 1 to MAX_TOKENS rows: decode requests
WIDTHS = tuple(range(1, L.MAX_TOKENS + 1))
# the mailbox tags K2 writes: TAG_SLOTS // 2 + i, as in sgl-project/sglang#43516
# (its K1 has i)
TAG_SLOTS = 256
MAX_LAYERS = TAG_SLOTS // 2

# The variants the kernels do not run yet, refused once a model
PENDING_FSE = (
    "the kernels do not run the fused shared expert yet; "
    "start without --enforce-shared-experts-fusion"
)
PENDING_MXFP4 = (
    "the kernels do not run the MXFP4 GDN projections yet; "
    "start with SGLANG_ROCM_GDN_ONLINE_MXFP4=0"
)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": ("enforce_shared_experts_fusion",),
    "srt/layers/layernorm.py": ("variance_epsilon",),
    "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py": (
        "is_shuffled",
        '"w13_weight"',
        '"w2_weight"',
        '"w13_weight_scale"',
        '"w2_weight_scale"',
    ),
    "srt/model_executor/forward_batch_info.py": ("mrope_positions",),
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
        "out_proj",
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
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
}

# K2 against SGLang's layer on the same input (check.py, SGLANG_AMD_MONO_CHECK).
# K2 quantizes the expert inputs to MXFP4 as SGLang's AITER MoE does but sums in
# another order, so the layer output differs by about the MoE's rounding error.
# The bound holds on the rows that K2 and SGLang route alike (check.k2_alike).
NUMERICS = Numerics(NumericsKind.TOLERANCE, {"k2.out.cosine": 0.99})


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
    if cfg.enforce_shared_experts_fusion:
        reasons.append(PENDING_FSE)
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _bf16(*tensors) -> bool:
    return all(t.dtype == torch.bfloat16 for t in tensors)


def _layer_reasons(layer, eps: float) -> list[str]:
    """Why the kernels cannot run one decoder layer as SGLang loaded it."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    mlp = layer.mlp
    if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
        return [f"needs the sparse MoE block, got {type(mlp).__name__}"]
    reasons = []
    pending = _fse(mlp) or mlp.shared_expert is None
    if pending:
        reasons.append(PENDING_FSE)
    else:
        se = mlp.shared_expert
        if not _bf16(se.gate_up_proj.weight, se.down_proj.weight):
            reasons.append("needs the shared expert in bf16")
    ex = mlp.experts
    if not getattr(ex.w13_weight, "is_shuffled", False):
        reasons.append("needs the routed experts in AITER's shuffled layout")
    if ex.w13_weight.dtype != torch.uint8 or ex.w2_weight.dtype != torch.uint8:
        reasons.append("needs MXFP4 routed experts")
    if not pending and tuple(ex.w13_weight.shape) != (L.E, 2 * L.RI, L.HIDDEN // 2):
        reasons.append(
            f"routed w13 shape {tuple(ex.w13_weight.shape)}: the kernels take the "
            f"experts TP-sharded, unpadded"
        )
    if not _bf16(
        layer.post_attention_layernorm.weight,
        mlp.gate.weight,
        mlp.shared_expert_gate.weight,
    ):
        reasons.append("needs the norm, router and shared-expert gate in bf16")
    if _is_gdn(layer):
        o = layer.linear_attn.out_proj
        if _mxfp4(o):
            reasons.append(PENDING_MXFP4)
        elif not _bf16(o.weight):
            reasons.append("needs out_proj in bf16")
    else:
        o = layer.o_proj
        if not layer.attn_output_gate or layer.q_size != L.CORE:
            reasons.append("needs the gated attention output with one rank's heads")
        if not _bf16(o.weight):
            reasons.append("needs o_proj in bf16")
    if not _mxfp4(o) and tuple(o.weight.shape) != (L.HIDDEN, L.CORE):
        reasons.append(f"o_proj shape {tuple(o.weight.shape)}")
    if layer.post_attention_layernorm.variance_epsilon != eps:
        reasons.append("its norm eps differ from the first layer's")
    return reasons


def _eps(layers) -> float:
    """The post-attention norm eps: one for all layers."""
    return layers[0].post_attention_layernorm.variance_epsilon


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
            if why in (PENDING_FSE, PENDING_MXFP4):
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
    """One rank's launches: the build key and the scratch per width, the guard
    block (word 0 the step epoch) and the peer buffer."""

    def __init__(self, lm, rank, k2_keys, k2_scratch, epoch, peer):
        self.lm, self.rank = lm, rank
        self.k2_keys, self.k2_scratch = k2_keys, k2_scratch
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
        fb = (
            forward_batch
            if self.check_layers and not torch.cuda.is_current_stream_capturing()
            else None
        )
        self.epoch[:1].add_(1)
        h = m.embed_tokens(input_ids)
        residual = None
        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
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

    def _k2(self, i, layer, core, residual, s, check_batch=None):
        """``check_batch``: the forward batch in check mode, else None."""
        from sgl_amd_mono.qwen38_a95b_mxfp4 import check
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer as k2

        mlp = layer.mlp
        o = layer.linear_attn.out_proj if _is_gdn(layer) else layer.o_proj
        if check_batch is not None:
            ref, ids, logits = check.k2_reference(layer, o, core, residual, check_batch)
        se, ex = mlp.shared_expert, mlp.experts
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        k2.layer_post(
            self.k2_keys[s],
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
            w_sgu=se.gate_up_proj.weight,
            w_sd=se.down_proj.weight,
        )
        if check_batch is not None:
            # ROUTE: (value, tag) pairs; token t's experts at 2 k t, then its weights
            off, n = k2.scratch_layout(self.k2_keys[s])["route"]
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
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer as k2

    _writable_flydsl_cache()
    p = get_parallel()
    tp, rank = p.tp_size, p.tp_rank
    layers = _layers(model)
    device = layers[0].input_layernorm.weight.device
    fault_site, fault_after = guard.fault_injection(rank) or (0, 0)
    k2_keys = {
        w: k2.K2Build(
            tokens=w,
            eps=_eps(layers),
            fault_site=fault_site,
            fault_after=fault_after,
        )
        for w in ctx.widths
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
        k2_keys,
        _scratch(k2_keys, k2.scratch_bytes, device),
        epoch,
        peer,
    )
    # Every rank builds every key. FlyDSL's disk cache hands a rank the kernels
    # another rank built (a lock per kernel), so each rank starts at its own share.
    launches = [k2.build(k) for k in k2_keys.values()]
    start = len(launches) * rank // tp
    t0 = time.monotonic()
    for launch in launches[start:] + launches[:start]:
        _compile_only(launch)
    torch.cuda.synchronize()
    logger.info(
        "Qwen3.8 monokernel: %d K2 launches (widths %s) in %.0f s",
        len(k2_keys),
        list(ctx.widths),
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
    features=frozenset(),
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
