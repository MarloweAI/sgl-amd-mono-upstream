"""The Muse Spark gfx950 monokernel spec on CPU (RFC #43949): its registry entry,
the configurations it refuses before load, the layers it refuses after load (meta
tensors of the real shapes at TP8 / EP8) and the verify widths it serves on the
DFlash recipe. SGLang checks the declared requirements, the same way for every spec."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry
try:
    from sglang.srt.kernel_plugins.amd.binding import covered_capture_sizes
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.guards import covered_capture_sizes

import sgl_amd_mono
from sgl_amd_mono.muse_spark import HF_CONFIG
from sgl_amd_mono.muse_spark import spec as muse
from sgl_amd_mono.muse_spark import weights as muse_weights
from sgl_amd_mono.muse_spark.gfx950.config import (
    EXPERTS,
    GATE_TEMPERATURE,
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_EXPERTS,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MOE_HIDDEN,
    POST_EPS,
    Q_K,
    QK_SCALE,
    QKVG_ROWS,
    RMS_EPS,
    ROUTE_EPS,
    ROUTE_SCALE,
    ROUTER_MULT,
    SOFTMAX_SCALE,
    TOP_K,
    WINDOW,
)
from sgl_amd_mono.muse_spark.weights import LayerViews, expert_kind, layer_reasons

# Muse Spark's text config (config.json, text_config): 46 sliding + 16 full layers
LAYERS = HF_CONFIG["num_hidden_layers"]
TEXT = SimpleNamespace(
    **HF_CONFIG,
    layer_types=[
        "full_attention" if i % 4 == 1 else "sliding_attention" for i in range(LAYERS)
    ],
    mlp_layer_types=["sparse"] * LAYERS,
)
OMNI = SimpleNamespace(
    model_type="muse_spark",
    text_config=TEXT,
    architectures=["MuseSparkForConditionalGeneration"],
)
# the AgentX recipe's server arguments the spec reads
CFG = SimpleNamespace(speculative_algorithm="DFLASH", speculative_num_draft_tokens=8)


def _replace(ns, **changes):
    return SimpleNamespace(**{**vars(ns), **changes})


def _config_reasons(cfg=CFG, hf_config=OMNI):
    with mock.patch.object(muse, "_has_flydsl", return_value=True):
        return muse._config_disable_reasons(cfg, hf_config)


def test_the_plugin_registers_the_architecture():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND,
        architectures=["MuseSparkForConditionalGeneration"],
        isa="gfx950",
    )
    assert found is muse.SPEC, why
    # the DFlash2 draft's spec, by its architecture
    from sgl_amd_mono.muse_spark import draft

    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=["DFlash2DraftModel"], isa="gfx950"
    )
    assert found is draft.SPEC, why
    assert draft.SPEC.tokens_per_request is not None and draft.SPEC.widths(
        "gfx950"
    ) == muse.SPEC.widths("gfx950")


def test_the_draft_config_is_accepted():
    from types import SimpleNamespace

    from sgl_amd_mono.muse_spark import draft

    hf = SimpleNamespace(
        model_type="qwen3",
        architectures=["DFlash2DraftModel"],
        dflash_config={
            "block_size": 8,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "attention_mode": "gqa",
        },
        layer_types=["sliding_attention"] * 5,
        **draft.DRAFT_CONFIG,
    )
    assert draft._config_disable_reasons(None, hf) == []
    hf.head_dim = 64
    hf.dflash_config["block_size"] = 16
    reasons = draft._config_disable_reasons(None, hf)
    assert len(reasons) == 2 and "head_dim" in reasons[0] and "block_size" in reasons[1]


def test_the_recipe_is_accepted():
    assert _config_reasons() == []
    assert _config_reasons(cfg=_replace(CFG, speculative_algorithm=None)) == []
    # a config.json that rounds the multiplier is the same model
    rounded = _replace(TEXT, output_multiplier=0.1767766953)
    assert _config_reasons(hf_config=_replace(OMNI, text_config=rounded)) == []


@pytest.mark.parametrize(
    "cfg, text",
    [
        (_replace(CFG, speculative_num_draft_tokens=4), TEXT),
        (CFG, _replace(TEXT, hidden_size=4096)),
        (CFG, _replace(TEXT, num_experts_per_tok=4)),
        (CFG, _replace(TEXT, qk_scale_factor=1.0)),
        (CFG, _replace(TEXT, mlp_layer_types=["dense"] + ["sparse"] * (LAYERS - 1))),
    ],
    ids=["draft tokens", "hidden size", "top-k", "q scale", "dense layer"],
)
def test_each_unsupported_setting_is_one_reason(cfg, text):
    assert (
        len(_config_reasons(cfg=cfg, hf_config=_replace(OMNI, text_config=text))) == 1
    )


def test_the_recipe_routes_verify_batches_of_one_to_eight():
    # the served widths come from SGLANG_AMD_MONO_MUSE_WIDTHS (default 8: the
    # verify batch of one request; wider launches lose to SGLang's kernels)
    from sgl_amd_mono.muse_spark.gfx950.config import WIDTHS

    assert muse.SPEC.widths("gfx950") == WIDTHS and 8 in WIDTHS
    # 8 verify rows a request: with every width served, the 8-, 16-, 32- and 64-row
    # launches route batch sizes 1, 2, 4 and 8, captured besides the configured sizes
    assert covered_capture_sizes(
        widths=(8, 16, 32, 64), tokens_per_request=8, sizes=[1, 2, 4, 8, 16, 32]
    ) == [1, 2, 4, 8, 16, 32]
    assert covered_capture_sizes(
        widths=(8,), tokens_per_request=8, sizes=[1, 2, 4, 8, 16, 32]
    ) == [1, 2, 4, 8, 16, 32]
    # without speculation the same launches serve decode batches of 8 to 64
    assert covered_capture_sizes(
        widths=(8, 16, 32, 64), tokens_per_request=1, sizes=[1, 8, 128]
    ) == [8, 16, 32, 64, 128]


class MuseSpark1xRMSNorm(torch.nn.Module):
    """Named as SGLang's: the checks read the norm's type by name."""

    def __init__(self, n, eps, weighted=True):
        super().__init__()
        self.eps, self.has_weight = eps, weighted
        self.weight = (
            torch.nn.Parameter(torch.empty(n, dtype=torch.bfloat16, device="meta"))
            if weighted
            else None
        )
        self._effective_weight_cache = None

    def _get_effective_weight(self):
        return None if self.weight is None else self.weight + 1


class MuseSpark1xRMSNormWithInputScale(MuseSpark1xRMSNorm):
    pass


class MuseSpark1xMxfp4Experts(SimpleNamespace):
    pass


class MuseSpark1xNativeExperts(SimpleNamespace):
    pass


def _meta(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


class _Gate:
    """A norm-preserving residual gate as the views read it (coefficients not precomputed)."""

    def __init__(self, gate=None):
        self.gate = _meta(HIDDEN) if gate is None else gate
        self.temperature = GATE_TEMPERATURE
        self._alpha = self._beta = None

    def _coefficients(self, w):
        beta = torch.sigmoid(w)
        return torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)), beta


def _gate():
    return _Gate()


def _mxfp4_experts():
    E = LOCAL_EXPERTS
    return MuseSpark1xMxfp4Experts(
        w13_weight=_meta(E, 2 * INTER, MOE_HIDDEN // 2, dtype=torch.uint8),
        w2_weight=_meta(E, MOE_HIDDEN, INTER // 2, dtype=torch.uint8),
        w13_weight_scale=_meta(E, 2 * INTER, MOE_HIDDEN // 32, dtype=torch.uint8),
        w2_weight_scale=_meta(E, MOE_HIDDEN, INTER // 32, dtype=torch.uint8),
    )


def _bf16_experts():
    E = LOCAL_EXPERTS
    return MuseSpark1xNativeExperts(
        w13_weight=_meta(E, 2 * INTER, MOE_HIDDEN),
        w2_weight=_meta(E, MOE_HIDDEN, INTER),
    )


def _layer(sliding=True, experts=None, **changes):
    """A Muse Spark decoder layer at TP8 / EP8, its tensors on the meta device."""
    attn = SimpleNamespace(
        num_heads=LOCAL_Q_HEADS,
        num_kv_heads=LOCAL_KV_HEADS,
        head_dim=HEAD_DIM,
        use_qk_norm=True,
        use_o_gate=True,
        use_o_norm=True,
        scaling=SOFTMAX_SCALE,
        scale_query_by=QK_SCALE,
        norm_eps=RMS_EPS,
        qkv_proj=SimpleNamespace(weight=_meta(QKVG_ROWS, HIDDEN)),
        o_proj=SimpleNamespace(weight=_meta(HIDDEN, Q_K)),
        rotary_emb=SimpleNamespace(
            cos_sin_cache=_meta(1 << 20, HEAD_DIM, dtype=torch.float32),
            is_neox_style=False,
            rotary_dim=HEAD_DIM,
        )
        if sliding
        else None,
        attn=SimpleNamespace(
            sliding_window_size=WINDOW if sliding else -1, k_scale=None, v_scale=None
        ),
        pre_attn_norm=MuseSpark1xRMSNorm(HIDDEN, RMS_EPS),
    )
    mlp = SimpleNamespace(
        tp_size=8,
        ep_size=8,
        num_local_experts=LOCAL_EXPERTS,
        top_k=TOP_K,
        expert_hidden_size=MOE_HIDDEN,
        intermediate_size=INTER,
        route_scale=ROUTE_SCALE,
        route_eps=ROUTE_EPS,
        router_multiplier=ROUTER_MULT,
        router=SimpleNamespace(weight=_meta(HIDDEN, EXPERTS, dtype=torch.float32)),
        expert_bias=_meta(EXPERTS, dtype=torch.float32),
        pre_expert_proj=SimpleNamespace(weight=_meta(MOE_HIDDEN, HIDDEN)),
        post_expert_proj=SimpleNamespace(weight=_meta(HIDDEN, MOE_HIDDEN)),
        pre_expert_norm=MuseSpark1xRMSNorm(MOE_HIDDEN, RMS_EPS),
        post_expert_norm=MuseSpark1xRMSNormWithInputScale(MOE_HIDDEN, POST_EPS),
        experts=experts or _mxfp4_experts(),
    )
    layer = SimpleNamespace(
        is_moe_layer=True,
        self_attn=attn,
        mlp=mlp,
        post_attn_norm=MuseSpark1xRMSNorm(HIDDEN, POST_EPS, weighted=False),
        pre_feedforward_layernorm=MuseSpark1xRMSNorm(HIDDEN, RMS_EPS),
        post_feedforward_layernorm=MuseSpark1xRMSNormWithInputScale(HIDDEN, POST_EPS),
        post_attention_residual_gate=_gate(),
        post_feedforward_residual_gate=_gate(),
    )
    for path, value in changes.items():
        owner, name = layer, path
        while "." in name:
            head, name = name.split(".", 1)
            owner = getattr(owner, head)
        setattr(owner, name, value)
    return layer


def test_the_checks_accept_every_layer_kind():
    assert layer_reasons(_layer(), 3) == []
    assert layer_reasons(_layer(sliding=False), 1) == []
    bf16 = _layer(experts=_bf16_experts())
    assert layer_reasons(bf16, 0) == []
    assert (expert_kind(_layer()), expert_kind(bf16)) == ("mxfp4", "bf16")


@pytest.mark.parametrize(
    "changes",
    [
        {"is_moe_layer": False},
        {"self_attn.num_kv_heads": 4},  # TP4
        {"self_attn.use_qk_norm": False},
        {"self_attn.scale_query_by": 1.0},
        {
            "self_attn.qkv_proj": SimpleNamespace(
                weight=_meta(QKVG_ROWS, HIDDEN, dtype=torch.float8_e4m3fn)
            )
        },
        {
            "self_attn.attn": SimpleNamespace(
                sliding_window_size=4095, k_scale=None, v_scale=None
            )
        },
        {
            "self_attn.attn": SimpleNamespace(
                sliding_window_size=WINDOW,
                k_scale=_meta(2, dtype=torch.float32),
                v_scale=None,
            )
        },
        {"post_attn_norm": MuseSpark1xRMSNorm(HIDDEN, POST_EPS)},  # weighted post norm
        {
            "post_feedforward_layernorm": SimpleNamespace(weight=_meta(HIDDEN))
        },  # not a Muse norm
        {"post_attention_residual_gate": _Gate(_meta(1))},
        {"mlp.ep_size": 4},
        {"mlp.top_k": 4},
        {"mlp.router_multiplier": 1.0},
        {
            "mlp.router": SimpleNamespace(
                weight=_meta(EXPERTS, HIDDEN, dtype=torch.float32)
            )
        },  # transposed
        {"mlp.expert_bias": None},
        {"mlp.pre_expert_proj": None},
        {"mlp.experts": SimpleNamespace(w13_weight=None)},  # NVFP4 (another class)
        {
            "mlp.experts": MuseSpark1xMxfp4Experts(
                **{**vars(_mxfp4_experts()), "w2_weight_scale": None}
            )
        },
    ],
    ids=[
        "dense layer",
        "tp4 heads",
        "no qk norm",
        "q scale",
        "fp8 qkv",
        "window",
        "per-head k scale",
        "weighted post norm",
        "norm type",
        "scalar gate",
        "ep4",
        "top-k",
        "router multiplier",
        "transposed router",
        "no routing bias",
        "no pooler",
        "nvfp4 experts",
        "no block scales",
    ],  # fmt: skip
)
def test_each_unsupported_layer_is_refused(changes):
    reasons = layer_reasons(_layer(**changes), 3)
    assert len(reasons) == 1, reasons


def test_the_views_read_in_place_and_log_the_copies():
    layer = _layer()
    copies = {}
    v = LayerViews.build(layer, 3, MuseSpark1xRMSNorm(HIDDEN, RMS_EPS), copies)
    assert v.kind == "mxfp4" and v.is_swa and v.use_rope
    assert v.w_qkvg.data_ptr() == layer.self_attn.qkv_proj.weight.data_ptr()
    assert tuple(v.gate_hi.shape) == tuple(v.gate_lo.shape) == (EXPERTS, HIDDEN)
    assert (
        tuple(v.alpha.shape) == tuple(v.beta2.shape) == (HIDDEN,)
        and v.alpha.dtype == torch.float32
    )
    assert (
        v.pre_w.dtype == v.npw.dtype == torch.bfloat16 and v.pen_w.numel() == MOE_HIDDEN
    )
    # no packed router and no cached norms on this layer: the split and the effective weights are copies
    kinds = {k[0] for k in copies}
    from sgl_amd_mono.muse_spark import weights as W

    want = {"router hi/lo", "effective norm weight", "residual gate"}
    if W.FP8_DENSE:
        want.add("fp8 dense")  # the pooler / unpooler FP8 copies (off by default)
    assert kinds == want
    # a packed router is read in place
    packed = _meta(MOE_HIDDEN + 2 * EXPERTS, HIDDEN)
    layer.mlp.router_pooler_packed_weight = packed
    v2 = LayerViews.build(layer, 3, MuseSpark1xRMSNorm(HIDDEN, RMS_EPS), {})
    assert v2.gate_hi.data_ptr() == packed[MOE_HIDDEN].data_ptr()
    # bf16 experts: quantized online into Quark-layout copies with the switch on (the layer
    # then takes the kernels), else left to the modules (the default)
    bf16 = _layer(experts=_bf16_experts())
    copies = {}
    with mock.patch.object(muse_weights, "QUANT_BF16", True):
        q = LayerViews.build(bf16, 0, MuseSpark1xRMSNorm(HIDDEN, RMS_EPS), copies)
    assert q.quantized and q.kind == "mxfp4"
    assert (
        tuple(q.w13.shape) == (LOCAL_EXPERTS, 2 * INTER, MOE_HIDDEN // 2)
        and q.w13.dtype == torch.uint8
    )
    assert (
        tuple(q.s2.shape) == (LOCAL_EXPERTS, MOE_HIDDEN, INTER // 32)
        and q.s2.dtype == torch.uint8
    )
    assert {k[0] for k in copies} >= {"online MXFP4 experts"} and all(
        torch.is_tensor(v) for v in copies.values()
    )
    with mock.patch.object(muse_weights, "QUANT_BF16", False):
        plain = LayerViews.build(bf16, 0, MuseSpark1xRMSNorm(HIDDEN, RMS_EPS), {})
    assert plain.kind == "bf16" and plain.w13 is None and not plain.quantized
