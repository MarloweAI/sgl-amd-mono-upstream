"""The DeepSeek-V4.1 gfx950 monokernel spec on CPU (RFC #43949).

The layers its forward covers on DeepSeek-V4.1-Flash, its registry entry, its per-output bounds
and its model refusal reasons; SGLang checks its declared requirements, as for every spec."""

from types import SimpleNamespace

import torch

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.deepseek_v41 import CANDIDATES, HF_CONFIG, draft, wide
from sgl_amd_mono.deepseek_v41 import spec as dsv41

# DeepSeek-V4.1-Flash's layer structure (its config.json)
FLASH = SimpleNamespace(
    num_hidden_layers=40,
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    engram_layer_ids=[1, 14],
    **HF_CONFIG,
    model_type="deepseek_v41",
    architectures=["DeepseekV4ForCausalLM"],
)


def test_the_forward_covers_every_flash_layer():
    # the first layer, the Engram layers, the index sources and DSpark's target layers (37-39) too
    assert dsv41._covered_layers(FLASH) == list(range(40))


def test_other_layers_have_no_kernels():
    hashed = SimpleNamespace(**{**vars(FLASH), "num_hash_layers": 4})
    assert dsv41._covered_layers(hashed) == list(range(4, 40))  # hash-routed MoE
    ratio4 = SimpleNamespace(**{**vars(FLASH), "compress_ratios": [4] * 43})
    assert dsv41._covered_layers(ratio4) == []  # another compress ratio


def test_the_plugin_registers_both_specs():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    for arch, spec in (
        ("DeepseekV4ForCausalLM", dsv41.SPEC),
        ("DeepseekV4ForCausalLMDSpark", draft.SPEC),
    ):
        found, why = registry.find_spec(
            backend=sgl_amd_mono.BACKEND,
            architectures=[arch],
            isa="gfx950",
            model_type="deepseek_v41",
        )
        assert found is spec, why


def test_wide_steps_take_stock_graphs():
    from sglang.srt.kernel_plugins.amd.binding import covered_capture_sizes

    assert dsv41.SPEC.widths("gfx950") == dsv41.MONO_WIDTHS == draft.SPEC.widths("gfx950")
    assert sorted(wide.TP_WIDTHS) == [2, 4] and max(wide.WIDTHS) == 384
    for tp in (2, 4):
        widths = dsv41.SPEC.widths_at("gfx950", tp)
        assert widths == dsv41.MONO_WIDTHS + wide.WIDTHS
        # SGLang's speculative decode graphs to 64 requests: a DSpark server keeps them all
        stock = [*range(1, 9), *range(10, 33, 2), *range(40, 65, 4)]
        for top in (16, 32, 64):
            sizes = [b for b in stock if b <= top]
            assert covered_capture_sizes(widths=widths, tokens_per_request=6, sizes=sizes) == sizes
    rows = (60, 96, 108, 192, 240, 384)
    assert [wide.width_class(r) for r in rows] == [96, 96, 192, 192, 384, 384]


def test_tp8_steps_stay_within_the_monokernels_rows():
    # no wide width at TP8: the wide forward, its workspaces and peer buffers are never built
    assert dsv41.SPEC.widths_at("gfx950", 8) == dsv41.MONO_WIDTHS
    assert max(dsv41.SPEC.widths_at("gfx950", 8)) == dsv41.MAX_ROWS


def test_token_parallel_rows_split_evenly():
    # every rank owns the same number of rows of every width a token-parallel class serves
    for tp, classes in wide.TOKEN_PARALLEL.items():
        for rows in wide.WIDTHS:
            if wide.width_class(rows) in classes:
                assert rows % tp == 0 and (rows // tp) * 5120 * 2 % (tp * 16) == 0


def test_kv_records_are_bit_exact():
    # the normal path (a prefill, a batch past the largest width) reads the KV records K1 writes
    for spec in (dsv41.SPEC, draft.SPEC):
        assert spec.numerics.bound("k1.kv.max_abs") == 0.0


def _layer(i, **attn):
    attn = {
        "compressor": None,
        "indexer": None,
        "q_head_norm": False,
        "fuse_wqa_wkv": True,
        "n_local_heads": 16,
        "compress_ratio": 0 if i < 2 else 2 if i < 20 else 1,
        **{lin: SimpleNamespace() for lin in ("wqkv_a", "wq_b", "wo_a", "wo_b")},
        **attn,
    }
    experts = SimpleNamespace(w13_weight=SimpleNamespace(is_shuffled=True))
    return SimpleNamespace(
        hc_boundary_fused=True,
        fused_rmsnorm_fake_quant=True,
        engram=None,
        self_attn=SimpleNamespace(**attn),
        mlp=SimpleNamespace(
            is_nextn=False,
            is_hash=False,
            experts=experts,
            reduce_results=True,
            _shared_expert_tp1=False,
            _enable_a2a_moe=False,
        ),
    )


def _parallel(tp_size=4, attn_tp_size=4, ca=True):
    """get_parallel() of a plain-TP server, its TP group's custom all-reduce aiter's."""
    comm = SimpleNamespace(disabled=False, custom_reduce_scatter=None) if ca else None
    return SimpleNamespace(
        tp_size=tp_size, attn_tp_size=attn_tp_size, tp_group=SimpleNamespace(ca_comm=comm)
    )


def test_token_parallel_reasons(monkeypatch):
    import sglang.srt.runtime_context as rc

    layers = [_layer(i) for i in range(40)]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    monkeypatch.setattr(rc, "get_parallel", _parallel)
    assert wide.reasons(model, range(40)) == []
    monkeypatch.setattr(rc, "get_parallel", lambda: _parallel(ca=False))
    assert wide.reasons(model, range(40)) == [
        "token-parallel wide steps need aiter's custom all-reduce on the TP group"
    ]
    monkeypatch.setattr(rc, "get_parallel", lambda: _parallel(attn_tp_size=2))
    assert "attention TP group" in wide.reasons(model, range(40))[0]
    monkeypatch.setattr(rc, "get_parallel", _parallel)
    layers[3].mlp._shared_expert_tp1 = True
    assert wide.reasons(model, range(40))[0].startswith("layer 3: token-parallel")
    layers[3] = _layer(3)
    # the span A launch runs gfx950's fake-quant input norm (``SpanAStages``)
    layers[4].fused_rmsnorm_fake_quant = False
    assert wide.reasons(model, range(40)) == [
        "layer 4: the wide span A launch runs gfx950's fake-quant input norm",
        "layer 4: wide seams run gfx950's fake-quant input norm",
    ]
    layers[4] = _layer(4)
    # TP8 has no token-parallel class
    monkeypatch.setattr(rc, "get_parallel", lambda: _parallel(tp_size=8, ca=False))
    assert wide.reasons(model, range(40)) == []


def test_model_reasons(monkeypatch):
    import sglang.srt.runtime_context as rc

    monkeypatch.setattr(rc, "get_parallel", _parallel)
    layers = [_layer(i) for i in range(40)]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    seen = {
        ("shared expert fp8", i, p, k): None
        for i in range(40)
        for p in dsv41._SHARED
        for k in ("weight", "scale")
    }
    assert dsv41.SPEC.model_disable_reasons(model, FLASH, seen) == []
    # wo_b shuffled in place (layer 5), a compressor at ratio 0 (layer 1), no load hook (layer 7)
    layers[5] = _layer(5, wo_b=SimpleNamespace(aiter_bpreshuffled=True))
    layers[1] = _layer(1, compress_ratio=0, compressor=SimpleNamespace())
    seen = {k: v for k, v in seen.items() if k[1] != 7}
    reasons = dsv41.SPEC.model_disable_reasons(model, FLASH, seen)
    assert len(reasons) == 3, reasons
    assert reasons[0].startswith("layer 1: a compressor or indexer at compress ratio 0")
    assert reasons[1].startswith("layer 5: an FP8 linear method reordered")
    assert "(layers [7])" in reasons[2]
    # the forward runs every layer, so one without kernels refuses the model
    hashed = SimpleNamespace(**{**vars(FLASH), "num_hash_layers": 2})
    reasons = dsv41.SPEC.model_disable_reasons(model, hashed, seen)
    assert reasons[0].startswith("layers [0, 1]: no kernels"), reasons
    # the wide seams run gfx950's fake-quant input norm (``SeamStages``)
    layers[3] = SimpleNamespace(**{**vars(_layer(3)), "fused_rmsnorm_fake_quant": False})
    reasons = dsv41.SPEC.model_disable_reasons(model, FLASH, seen)
    assert any(r.startswith("layer 3: wide seams run gfx950's fake-quant input norm") for r in reasons), reasons
    layers[3] = _layer(3)
    # K2i's indexers: the candidate blocks REINDEX layers score, SGLang's decode-row query path
    layers[1], layers[5] = _layer(1), _layer(5)
    seen |= {
        ("shared expert fp8", 7, p, k): None
        for p in dsv41._SHARED
        for k in ("weight", "scale")
    }
    rope = SimpleNamespace(dtype=torch.complex64)
    ix = SimpleNamespace(
        is_candidate_source=True,
        uses_candidates=False,
        weights_proj_hip_max_tokens=64,
        **CANDIDATES,
    )
    layers[20] = _layer(20, indexer=ix, freqs_cis=rope)
    assert dsv41.SPEC.model_disable_reasons(model, FLASH, seen) == []
    other_blocks = SimpleNamespace(**{**vars(ix), "candidate_topk_blocks": 1024})
    layers[20] = _layer(20, indexer=other_blocks, freqs_cis=rope)
    no_gemv = SimpleNamespace(**{**vars(ix), "weights_proj_hip_max_tokens": -1})
    layers[24] = _layer(24, indexer=no_gemv, freqs_cis=rope)
    reasons = dsv41.SPEC.model_disable_reasons(model, FLASH, seen)
    assert [r.split(":")[0] for r in reasons] == ["layer 20", "layer 24"], reasons


def test_pd_runs_on_a_mori_decode_server_only():
    def reasons(mode, backend="mori"):
        cfg = SimpleNamespace(
            disaggregation_mode=mode, disaggregation_transfer_backend=backend
        )
        return [
            r for r in dsv41._config_disable_reasons(cfg, FLASH) if "FlyDSL" not in r
        ]

    pd = registry.Feature.PD_DISAGGREGATION
    assert pd in dsv41.SPEC.features and pd in draft.SPEC.features
    assert reasons("null") == reasons(None) == []
    assert reasons("decode") == []
    assert reasons("prefill") == ["PD monokernels support the decode role only"]
    assert reasons("decode", "mooncake") == [
        "PD monokernels require the mori transfer backend"
    ]
