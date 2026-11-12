"""The Qwen3.8 MoE gfx950 monokernel spec on CPU (RFC #43949): its registry entry
and widths, and its config and model refusal reasons. The fused shared expert and
the MXFP4 GDN projections refuse at startup, once a model."""

import sys
from types import SimpleNamespace

import pytest
import torch

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.qwen38_a95b_mxfp4 import HF_CONFIG
from sgl_amd_mono.qwen38_a95b_mxfp4 import spec as qwen

try:
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
except RuntimeError:  # the module needs a GPU at import; the spec checks only the class
    Qwen2MoeSparseMoeBlock = type("Qwen2MoeSparseMoeBlock", (torch.nn.Module,), {})
    sys.modules["sglang.srt.models.qwen2_moe"] = SimpleNamespace(
        Qwen2MoeSparseMoeBlock=Qwen2MoeSparseMoeBlock
    )

QWEN = SimpleNamespace(**HF_CONFIG, num_hidden_layers=8)


def test_the_plugin_registers_the_spec():
    sgl_amd_mono.register_specs()
    arch = ["Qwen3_5MoeForCausalLM"]
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=arch, isa="gfx950"
    )
    assert found is qwen.SPEC, why
    assert found.devices["gfx950"].widths == tuple(range(1, 9))
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=arch, isa="gfx942"
    )
    assert found is None and "gfx942" in why


def _cfg(**kw):
    return SimpleNamespace(
        **{
            "speculative_algorithm": None,
            "speculative_eagle_topk": None,
            "speculative_num_steps": None,
            "speculative_num_draft_tokens": None,
            "enable_linear_replayssm": False,
            "enforce_shared_experts_fusion": False,
            **kw,
        }
    )


def test_config_reasons():
    reasons = qwen.SPEC.config_disable_reasons
    assert reasons(_cfg(), QWEN) == []
    assert reasons(_cfg(enable_linear_replayssm=True), QWEN) == []
    assert reasons(_cfg(enforce_shared_experts_fusion=True), QWEN) == [qwen.PENDING_FSE]
    other = SimpleNamespace(**{**vars(QWEN), "hidden_size": 4096})
    joined = " | ".join(reasons(_cfg(), other))
    assert "hidden_size = 8192" in joined, joined


def _meta(*shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


def _w(*shape, dtype=torch.bfloat16):
    return SimpleNamespace(weight=_meta(*shape, dtype=dtype))


def _norm(eps=1e-6):
    return SimpleNamespace(weight=_meta(8192), variance_epsilon=eps)


def _mlp(fse=False):
    mlp = Qwen2MoeSparseMoeBlock.__new__(Qwen2MoeSparseMoeBlock)
    torch.nn.Module.__init__(mlp)
    w13 = _meta(512 + fse, 512, 4096, dtype=torch.uint8)
    w13.is_shuffled = True
    mlp.experts = SimpleNamespace(
        w13_weight=w13, w2_weight=_meta(512 + fse, 8192, 128, dtype=torch.uint8)
    )
    mlp.num_fused_shared_experts = int(fse)
    mlp.shared_expert = (
        None
        if fse
        else SimpleNamespace(gate_up_proj=_w(512, 8192), down_proj=_w(8192, 256))
    )
    mlp.gate, mlp.shared_expert_gate = _w(512, 8192), _w(1, 8192)
    return mlp


def _layer(i, gdn=True, fse=False, mxfp4=False, eps=1e-6):
    layer = SimpleNamespace(
        layer_id=i,
        input_layernorm=_norm(eps),
        post_attention_layernorm=_norm(),
        mlp=_mlp(fse),
    )
    if not gdn:
        layer.o_proj, layer.attn_output_gate, layer.q_size = _w(8192, 2048), True, 2048
        return layer
    out = _w(8192, 2048)
    if mxfp4:
        out = SimpleNamespace(
            weight=_meta(8192, 1024, dtype=torch.uint8),
            weight_scale=_meta(8192, 64, dtype=torch.uint8),
        )
    layer.linear_attn = SimpleNamespace(
        layer_id=i,
        in_proj_qkvz=_w(4608, 8192),
        in_proj_ba=_w(32, 8192),
        conv1d=SimpleNamespace(weight=_meta(2560, 1, 4), bias=None),
        A_log=_meta(16, dtype=torch.float32),
        dt_bias=_meta(16),
        out_proj=out,
        activation="silu",
        norm=SimpleNamespace(
            weight=_meta(128),
            eps=1e-6,
            activation="silu",
            group_size=None,
            norm_before_gate=True,
        ),
    )
    return layer


def _model(layers):
    body = SimpleNamespace(
        layers=layers, start_layer=0, end_layer=len(layers), layers_to_capture=[]
    )
    return SimpleNamespace(model=body)


@pytest.mark.parametrize("kw", [{}, {"fse": True}, {"mxfp4": True}])
def test_model_reasons(kw):
    # Qwen3.8's pattern: three GDN layers, then a full-attention layer
    layers = [_layer(i, gdn=i % 4 != 3, **kw) for i in range(8)]
    reasons = qwen.SPEC.model_disable_reasons(_model(layers), QWEN, {})
    want = {
        "fse": [qwen.PENDING_FSE],
        "mxfp4": [qwen.PENDING_MXFP4],
    }
    # each pending variant refuses once, not once a layer
    assert reasons == want.get(next(iter(kw), None), []), reasons


def test_check_compares_the_rows_routed_alike():
    from sgl_amd_mono.qwen38_a95b_mxfp4 import check

    k2 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    # SGLang's order differs
    ids = torch.tensor([[3, 1, 2], [4, 5, 7]], dtype=torch.int32)
    assert check.routed_alike(ids, k2).tolist() == [True, False]


def test_model_reasons_name_the_layer():
    layers = [_layer(i, gdn=i % 4 != 3) for i in range(8)]
    layers[1].post_attention_layernorm = _norm(1e-5)
    layers[2].linear_attn.out_proj = _w(8192, 2048, dtype=torch.float8_e4m3fn)
    layers[3].mlp.experts.w13_weight.is_shuffled = False
    layers[7].o_proj = _w(8192, 2048, dtype=torch.float8_e4m3fn)
    reasons = qwen.SPEC.model_disable_reasons(_model(layers), QWEN, {})
    assert [r.split(":")[0] for r in reasons] == [
        "layer 1",
        "layer 2",
        "layer 3",
        "layer 7",
    ]
    assert "eps" in reasons[0] and "out_proj in bf16" in reasons[1]
    assert "shuffled" in reasons[2] and "o_proj in bf16" in reasons[3]


def test_aux_hidden_states_run_sglangs_forward():
    class LM:
        def forward(self, *args):
            return "sglang"

    built = qwen._Built.__new__(qwen._Built)
    built.lm = LM()
    built.lm.model = SimpleNamespace(layers_to_capture=[2])
    assert built.forward(None, None, None, None) == "sglang"
