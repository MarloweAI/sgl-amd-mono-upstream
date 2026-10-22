"""The GLM-5.3-Flash gfx950 monokernel spec on CPU (RFC #43949): the layers its
forward covers on the GLM-5.3-Flash layer structure, its row routing and TP2 row
cap, its registry entry, and its config and model refusal reasons. SGLang checks its
declared requirements, the same way for every spec."""

from types import SimpleNamespace

import torch
from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.glm53_flash import HF_CONFIG, LINEAR_ATTN_CONFIG
from sgl_amd_mono.glm53_flash import spec as flash

# GLM-5.3-Flash's text-model layer structure (its config.json)
TEXT = SimpleNamespace(
    **HF_CONFIG,
    num_hidden_layers=45,
    first_k_dense_replace=3,
    linear_attn_config={
        **LINEAR_ATTN_CONFIG,
        "kda_layers": [i for i in range(45) if i % 4 != 3],
        "full_attn_layers": [i for i in range(45) if i % 4 == 3],
    },
)
FLASH = SimpleNamespace(
    text_config=TEXT,
    architectures=["Glm5NextForConditionalGeneration"],
    quantization_config={"quant_method": "quark"},
)


def test_the_forward_covers_the_moe_layers_but_the_last():
    # the dense layers 0 to 2 and layer 44, whose FFN exit contracts the streams,
    # run SGLang's path inside the forward
    assert flash._covered_layers(FLASH) == list(range(3, 44))


def test_the_plugin_registers_the_spec():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    arch = ["Glm5NextForConditionalGeneration"]
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=arch, isa="gfx950"
    )
    assert found is flash.SPEC, why
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=arch, isa="gfx942"
    )
    assert found is None and "gfx942" in why


class _Lm:
    """A model whose own forward records that it ran."""

    def __init__(self):
        self.model, self.ran = SimpleNamespace(), []

    def forward(self, *args):
        self.ran.append(args)
        return "the model's forward"


def test_a_step_runs_at_the_next_width_and_tp2_takes_four_rows():
    # a step of fewer rows than a width runs at the next width, the rest pad rows
    assert [flash._width(r) for r in flash.ROWS] == [2, 2, 4, 4, 8, 8, 8, 8]
    # TP2 builds no 8-row launch: a wider step runs the model's own forward
    runtimes = {(w, k): None for w in (2, 4) for k in ("kda", "dsa")}
    layers = {i: SimpleNamespace(i=i) for i in range(3, 44)}
    lm = _Lm()
    built = flash._Built(lm, layers, runtimes, tp=2, device="cpu")
    key = registry.ShapeKey(forward_mode="decode", batch_size=8, tokens_per_request=1)
    out = built.forward(registry.Plan(key=key), "ids", "positions", "batch")
    assert out == "the model's forward"
    assert lm.ran == [("ids", "positions", "batch", None, None)]
    # the first covered layer's identity mix leaves the written streams unchanged
    assert torch.equal(built.eye_comb[1], torch.eye(4)) and not built.zero_x.any()


def test_the_spec_needs_aiters_grouped_topk(monkeypatch):
    # the kernel's router ties follow AITER's grouped top-k; SGLang's JIT grouped
    # top-k (a Triton gate) keeps the lower expert id, so the spec requires it off.
    # SGLang refuses a spec whose required env value differs from envs' value.
    from sglang.srt.environ import envs

    def unmet():
        req = flash.SPEC.requires.env
        return [name for name, want in req.items() if getattr(envs, name).get() != want]

    name = "SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK"
    assert flash.SPEC.requires.env[name] is False
    monkeypatch.delenv(name, raising=False)
    assert unmet() == []
    monkeypatch.setenv(name, "1")
    assert unmet() == [name]


def _cfg(**kw):
    return SimpleNamespace(**{"page_size": 256, **kw})


def _dtypes(temporal=torch.float32, conv=torch.bfloat16):
    return lambda config: SimpleNamespace(temporal=temporal, conv=conv)


def test_config_reasons_name_each_mismatch(monkeypatch):
    import sglang.srt.configs.mamba_utils as mamba_utils
    import sglang.srt.layers.attention.dsa.utils as dsa_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes())
    monkeypatch.setattr(dsa_utils, "aiter_can_use_preshuffle_paged_mqa", lambda: True)
    assert flash.SPEC.config_disable_reasons(_cfg(), FLASH) == []
    # the last KDA layer (42 here) covered: its normal path tracks every layer's state
    kda = [i for i in TEXT.linear_attn_config["kda_layers"] if i < 44]
    text = SimpleNamespace(
        **{
            **vars(TEXT),
            "hidden_size": 6144,
            "moe_layer_freq": 2,
            "linear_attn_config": {
                **TEXT.linear_attn_config,
                "num_heads": 32,
                "kda_layers": kda,
            },
        }
    )
    monkeypatch.setattr(
        mamba_utils, "mamba2_state_dtype", _dtypes(temporal=torch.bfloat16)
    )
    # SGLang's k-pool writer keeps the index cache row-major; the kernels write tiles
    monkeypatch.setattr(dsa_utils, "aiter_can_use_preshuffle_paged_mqa", lambda: False)
    other = SimpleNamespace(
        text_config=text,
        architectures=FLASH.architectures,
        quantization_config={"quant_method": "fp8"},
    )
    reasons = flash.SPEC.config_disable_reasons(_cfg(page_size=64), other)
    joined = " | ".join(reasons)
    for want in (
        "hidden_size = 4096",
        "moe_layer_freq = 1",
        "num_heads = 64",
        "layer 42: the last KDA layer",
        "KDA state in f32",
        "--page-size 64",
        "preshuffled layout",
        "Quark checkpoint",
    ):
        assert want in joined, joined
    assert len(reasons) == 8, reasons


def _layer(kda=True, shuffled=True, scale_dims=3, fused=True, indexer=True):
    w13 = SimpleNamespace(is_shuffled=shuffled)
    experts = SimpleNamespace(
        w13_weight=w13, w13_weight_scale=torch.zeros((1,) * scale_dims)
    )
    attn = SimpleNamespace(
        do_fuse_qkvbfg=fused, indexer=SimpleNamespace() if indexer else None
    )
    return SimpleNamespace(
        is_linear_attn=kda, self_attn=attn, mlp=SimpleNamespace(experts=experts)
    )


def test_model_reasons():
    kda = TEXT.linear_attn_config["kda_layers"]
    layers = [_layer(kda=i in kda) for i in range(45)]
    model = SimpleNamespace(
        model=SimpleNamespace(layers=layers), is_mrope_enabled=False
    )
    assert flash.SPEC.model_disable_reasons(model, FLASH, {}) == []
    # layer 4's experts are not shuffled, layer 5's are GU-interleaved (2-D
    # scales), layer 6 (KDA) runs unfused projections, layer 7 has no indexer
    layers[4] = _layer(shuffled=False)
    layers[5] = _layer(scale_dims=2)
    layers[6] = _layer(fused=False)
    layers[7] = _layer(kda=False, indexer=False)
    model.is_mrope_enabled = True
    reasons = flash.SPEC.model_disable_reasons(model, FLASH, {})
    assert [r.split(":")[0] for r in reasons] == [
        "multimodal rotary positions",
        "layer 4",
        "layer 5",
        "layer 6",
        "layer 7",
    ], reasons
    assert "shuffled" in reasons[1] and "GU-interleaved" in reasons[2]
