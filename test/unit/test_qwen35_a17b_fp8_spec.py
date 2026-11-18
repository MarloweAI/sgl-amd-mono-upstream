"""Qwen3.5-397B-A17B block-FP8 spec on CPU: family registration, TP4 and TP8
shapes, the MTP verify widths and their refusals, the fallback routing, and the
refusals for MXFP4, per-channel FP8, a plain RMSNorm and a projection with a
bias."""

from types import SimpleNamespace

import torch

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.qwen35_a17b import spec as family
from sgl_amd_mono.qwen35_a17b.fp8 import spec as fp8
from sgl_amd_mono.qwen35_a17b.fp8.gfx950.layout import (
    GDN_K_HEADS,
    GDN_V_HEADS,
    INTERMEDIATE,
    O_K,
    QKVZ_N,
    TP,
)
from sgl_amd_mono.qwen35_a17b.fp8.gfx950.shapes import TP4, TP8, for_tp
from sgl_amd_mono.qwen35_a17b.layout import LAYERS

ARCH = ["Qwen3_5MoeForConditionalGeneration"]


def _text(**overrides):
    values = dict(fp8._TEXT)
    values["layer_types"] = [
        "full_attention" if i % 4 == 3 else "linear_attention" for i in range(LAYERS)
    ]
    values["rope_parameters"] = {
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _hf(**quant):
    block = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    block.update(quant)
    return SimpleNamespace(
        text_config=_text(),
        architectures=ARCH,
        quantization_config=block,
    )


def _cfg(**overrides):
    return SimpleNamespace(**overrides)


def _mtp(**overrides):
    values = dict(
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        enable_linear_replayssm=False,
    )
    values.update(overrides)
    return _cfg(**values)


class _Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def _fp8(shape):
    return SimpleNamespace(
        weight=_Tensor(shape, torch.float8_e4m3fn),
        weight_scale_inv=_Tensor(tuple(s // 128 for s in shape[-2:]), torch.float32),
    )


def _bf16(shape):
    return SimpleNamespace(weight=_Tensor(shape, torch.bfloat16))


def _experts(layout, e=513):
    w13_n = layout.w13_n
    intermediate = layout.intermediate
    return SimpleNamespace(
        w13_weight=_Tensor((e, w13_n, 4096), torch.float8_e4m3fn),
        w13_weight_scale_inv=_Tensor((e, w13_n // 128, 32), torch.float32),
        w2_weight=_Tensor((e, 4096, intermediate), torch.float8_e4m3fn),
        w2_weight_scale_inv=_Tensor((e, 32, intermediate // 128), torch.float32),
    )


def _mlp(layout):
    return SimpleNamespace(
        enable_shared_expert_fusion=True,
        reduce_results=False,
        experts=_experts(layout),
        gate=_bf16((512, 4096)),
    )


class GemmaRMSNorm:
    """Stands in for SGLang's, which the spec knows by name."""


class RMSNorm:
    pass


def _norms():
    return dict(input_layernorm=GemmaRMSNorm(), post_attention_layernorm=GemmaRMSNorm())


def _layer(i, layout):
    if i % 4 == 3:
        return SimpleNamespace(
            mlp=_mlp(layout),
            qkv_proj=_fp8((layout.qkv_n, 4096)),
            o_proj=_fp8((4096, layout.o_k)),
            **_norms(),
        )
    return SimpleNamespace(
        mlp=_mlp(layout),
        linear_attn=SimpleNamespace(
            in_proj_qkvz=_fp8((layout.qkvz_n, 4096)),
            in_proj_ba=_bf16((layout.ba_n, 4096)),
            conv1d=_bf16((layout.conv_dim, 1, 4)),
            out_proj=_fp8((4096, layout.o_k)),
        ),
        **_norms(),
    )


def _model(layout=TP4):
    return SimpleNamespace(
        model=SimpleNamespace(
            layers=[_layer(i, layout) for i in range(LAYERS)], norm=GemmaRMSNorm()
        )
    )


def test_tp4_specialization_is_the_layout_module():
    assert (TP4.tp, TP4.gdn_k_heads, TP4.gdn_v_heads) == (TP, GDN_K_HEADS, GDN_V_HEADS)
    assert (TP4.qkvz_n, TP4.o_k, TP4.intermediate) == (QKVZ_N, O_K, INTERMEDIATE)
    assert for_tp(4) is not for_tp(8)
    assert (TP8.gdn_k_heads, TP8.gdn_v_heads, TP8.qkvz_n) == (2, 8, 2560)
    assert (TP8.ba_n, TP8.conv_dim, TP8.o_k, TP8.intermediate, TP8.w13_n) == (
        16,
        1536,
        1024,
        128,
        256,
    )
    assert TP8.qkv_n == 2560


def test_the_plugin_registers_the_family_on_gfx950():
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=ARCH, isa="gfx950"
    )
    assert found is family.SPEC, why
    limits = found.devices["gfx950"]
    assert limits.tp_sizes == (4, 8)
    assert fp8.SPEC.devices["gfx950"].tp_sizes == (4, 8)
    assert limits.widths == (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32)
    assert registry.Feature.SPECULATIVE_DECODING in found.features
    missing, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=ARCH, isa="gfx942"
    )
    assert missing is None and "gfx942" in why


def test_config_accepts_block_fp8_and_refuses_mxfp4():
    assert fp8.SPEC.config_disable_reasons(_cfg(), _hf()) == []
    quark = _hf(quant_method="quark", activation_scheme=None, weight_block_size=None)
    joined = " | ".join(fp8.SPEC.config_disable_reasons(_cfg(), quark))
    assert "block-FP8" in joined, joined
    channel = _hf(weight_block_size=[1, 4096])
    assert any(
        "block-FP8" in r for r in fp8.SPEC.config_disable_reasons(_cfg(), channel)
    )
    mx = _hf(quant_method="mxfp4")
    assert any("block-FP8" in r for r in fp8.SPEC.config_disable_reasons(_cfg(), mx))
    family_mx = " | ".join(family.SPEC.config_disable_reasons(_cfg(), mx))
    assert "not in this package" in family_mx, family_mx
    assert family.SPEC.config_disable_reasons(_cfg(), _hf()) == []


def test_model_accepts_tp4_and_tp8_and_refuses_per_channel():
    assert fp8.SPEC.model_disable_reasons(_model(TP4), _hf(), {}) == []
    assert fp8.SPEC.model_disable_reasons(_model(TP8), _hf(), {}) == []
    assert family.SPEC.model_disable_reasons(_model(TP8), _hf(), {}) == []
    odd = _model(TP4)
    odd.model.layers[0].linear_attn.out_proj = _fp8((4096, 4096))
    refused = fp8.SPEC.model_disable_reasons(odd, _hf(), {})
    assert refused and "projection K" in refused[0], refused
    model = _model()
    # per-channel AttnFP8: a 1-D scale instead of a 128x128 block scale
    model.model.layers[3].o_proj = SimpleNamespace(
        weight=_Tensor((4096, 2048), torch.float8_e4m3fn),
        weight_scale_inv=_Tensor((4096,), torch.float32),
    )
    reasons = fp8.SPEC.model_disable_reasons(model, _hf(), {})
    assert reasons and "layer 3 o_proj" in reasons[0], reasons


def test_model_refuses_a_plain_rmsnorm():
    model = _model()
    model.model.layers[5].post_attention_layernorm = RMSNorm()
    reasons = fp8.SPEC.model_disable_reasons(model, _hf(), {})
    assert reasons == [
        "layer 5 post_attention_layernorm is RMSNorm, not a Gemma RMSNorm"
    ], reasons


def test_config_accepts_the_mtp_chain_and_refuses_other_drafts():
    assert fp8.SPEC.config_disable_reasons(_mtp(), _hf()) == []
    assert family.SPEC.config_disable_reasons(_mtp(), _hf()) == []
    for bad in (
        _mtp(speculative_eagle_topk=4),
        _mtp(speculative_num_steps=5, speculative_num_draft_tokens=6),
        _mtp(speculative_num_steps=2, speculative_num_draft_tokens=4),
    ):
        joined = " | ".join(fp8.SPEC.config_disable_reasons(bad, _hf()))
        assert "one draft chain of 4 tokens" in joined, joined
    replay = fp8.SPEC.config_disable_reasons(_mtp(enable_linear_replayssm=True), _hf())
    assert any("--enable-linear-replayssm" in r for r in replay), replay
    # ReplaySSM alone, without a draft, keeps the decode kernels' in-place update
    assert (
        fp8.SPEC.config_disable_reasons(
            _cfg(speculative_algorithm=None, enable_linear_replayssm=True), _hf()
        )
        == []
    )


class _Fallback:
    """A model whose own forward records the steps the monokernel hands back."""

    def __init__(self):
        self.calls = []

    def forward(self, input_ids, positions, forward_batch, *rest):
        self.calls.append(forward_batch)
        return "standard"


def test_forward_hands_unserved_steps_to_the_standard_forward():
    model = _Fallback()
    built = fp8._Built(model, runtime=None, layout=TP4)
    batch = SimpleNamespace(contains_image_inputs=lambda: False)
    for mode, bs, tokens in (
        ("decode", 9, 1),  # wider than the decode kernels
        ("target_verify", 9, 4),  # more than 8 requests
        ("target_verify", 4, 2),  # another draft chain
        ("extend", 1, 1),
    ):
        key = registry.ShapeKey(
            forward_mode=mode, batch_size=bs, tokens_per_request=tokens
        )
        out = built.forward(registry.Plan(key=key), None, None, batch)
        assert out == "standard", (mode, bs, tokens)
    assert len(model.calls) == 4
    image = SimpleNamespace(contains_image_inputs=lambda: True)
    key = registry.ShapeKey(
        forward_mode="target_verify", batch_size=2, tokens_per_request=4
    )
    assert built.forward(registry.Plan(key=key), None, None, image) == "standard"


def test_model_refuses_a_projection_with_a_bias():
    model = _model()
    model.model.layers[3].qkv_proj.bias = _Tensor((4608,), torch.bfloat16)
    reasons = fp8.SPEC.model_disable_reasons(model, _hf(), {})
    assert reasons == ["layer 3 qkv has a bias, the block-FP8 GEMV adds none"], reasons
