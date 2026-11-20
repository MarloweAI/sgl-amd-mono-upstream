"""The Kimi-K3 gfx950 monokernel spec on CPU (RFC #43949): the layers its forward
covers on the Kimi-K3 layer structure, its registry entry, its per-output bounds
and its config refusal reasons. SGLang checks its declared requirements, the same
way for every spec."""

from types import SimpleNamespace

import pytest

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.kimi_k3 import (
    HF_CONFIG,
    K2_MAX_ROWS,
    LINEAR_ATTN_CONFIG,
    MAX_BLOCKS,
    MAX_ROWS,
)
from sgl_amd_mono.kimi_k3 import spec as k3

# Kimi-K3's layer structure (its config.json): 93 layers, a dense layer 0, then
# the KDA / full-attention cadence. Every layer past the dense one runs a MoE.
FULL_ATTN = list(range(4, 93, 4))
KDA = [i for i in range(1, 93) if i not in FULL_ATTN]
TEXT = SimpleNamespace(
    num_hidden_layers=93,
    attn_res_block_size=12,
    linear_attn_config={**LINEAR_ATTN_CONFIG, "full_attn_layers": FULL_ATTN + [93],
                        "kda_layers": KDA},
    **HF_CONFIG,
)  # fmt: skip
K3 = SimpleNamespace(
    model_type="kimi_k3",
    architectures=["KimiK3ForConditionalGeneration"],
    text_config=TEXT,
)


def test_the_forward_covers_every_moe_layer():
    # the dense layer 0 keeps the model's own forward; every other layer is a
    # MoE layer and takes the launch
    assert k3._covered_layers(K3) == list(range(1, 93))


def test_coverage_is_not_an_attention_kind():
    # The launch starts after the attention, so both kinds are covered. A spec
    # that declared only the KDA layers would still serve and still be correct,
    # just slower -- which is why this is pinned.
    covered = set(k3._covered_layers(K3))
    assert set(KDA) <= covered and set(FULL_ATTN) <= covered
    assert len(covered) == 92


def test_the_dense_prefix_and_the_moe_cadence_are_read_from_the_config():
    dense = SimpleNamespace(**{**vars(TEXT), "first_k_dense_replace": 4})
    assert k3._covered_layers(SimpleNamespace(text_config=dense)) == list(range(4, 93))
    every_other = SimpleNamespace(**{**vars(TEXT), "moe_layer_freq": 2})
    assert k3._covered_layers(SimpleNamespace(text_config=every_other)) == list(
        range(2, 93, 2)
    )


def test_the_plugin_registers_the_spec():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND,
        architectures=["KimiK3ForConditionalGeneration"],
        isa="gfx950",
    )
    assert found is k3.SPEC, why


def test_the_spec_is_tp8_and_at_most_k2_max_rows():
    limits = k3.SPEC.devices["gfx950"]
    assert limits.tp_sizes == (8,)
    assert limits.widths == tuple(range(1, K2_MAX_ROWS + 1))
    # With DSpark the runtime serves only the widths that hold whole requests:
    # block size 5 (6 tokens a request) gives batch sizes 1 .. K2_MAX_ROWS // 6.
    assert [w // 6 for w in limits.widths if w % 6 == 0] == list(range(1, K2_MAX_ROWS // 6 + 1))


def test_the_shapes_are_checked_on_the_text_config():
    # The architecture names the VLM wrapper, whose top-level config carries
    # none of the shapes, so requires.hf_config cannot reach them.
    assert "hidden_size" not in k3.SPEC.requires.hf_config
    assert k3.SPEC.requires.hf_config == {"model_type": "kimi_k3"}
    wrong = SimpleNamespace(**{**vars(TEXT), "num_experts_per_token": 8})
    reasons = k3._config_disable_reasons(None, SimpleNamespace(text_config=wrong))
    assert any("num_experts_per_token" in r for r in reasons), reasons


def test_a_block_size_past_the_launch_is_refused():
    # The bank rows a layer's seam reads grow with the layer index; at block
    # size 12 the last layer reads exactly MAX_BLOCKS of them.
    tight = SimpleNamespace(**{**vars(TEXT), "attn_res_block_size": 8})
    reasons = k3._config_disable_reasons(None, SimpleNamespace(text_config=tight))
    assert any("attention-residual blocks" in r for r in reasons), reasons
    ok = k3._config_disable_reasons(None, K3)
    assert not any("attention-residual blocks" in r for r in ok), ok


def test_the_bounds_name_the_launches_and_their_outputs():
    for name in ("k2.prefix.cosine", "k2.x.cosine", "k2.out.cosine", "moe.out.cosine"):
        assert 0.0 < k3.SPEC.numerics.bound(name) <= 1.0
    with pytest.raises(KeyError):
        k3.SPEC.numerics.bound("k2.nothing.cosine")


def test_the_build_needs_a_real_model():
    with pytest.raises((AttributeError, TypeError)):
        k3.SPEC.build(None, K3, None)


def test_max_blocks_matches_the_reference_config():
    # layer 92 at block size 12 reads 8 blocks: cdiv(92, 12) = 8, and 92 is not
    # a block-write layer. The launch is built for exactly that many.
    last = max(k3._covered_layers(K3))
    block = TEXT.attn_res_block_size
    assert -(-last // block) + int(last % block == 0) == MAX_BLOCKS


def _fake_layer(*, gate=True, kda=False):
    """A covered layer as model_disable_reasons reads it, everything passing."""
    import torch

    def lin(*shape):
        return SimpleNamespace(
            weight=torch.zeros(*shape, dtype=torch.bfloat16), bias=None
        )

    attn = SimpleNamespace(
        o_proj=lin(HF_CONFIG["hidden_size"], 1536),
        forward_pre_o_proj=lambda **kw: None,
    )
    if not kda:  # only the MLA layers carry the gate
        attn.use_output_gate = gate
    experts = SimpleNamespace(
        w13_weight=SimpleNamespace(is_shuffled=True),
        w2_weight=SimpleNamespace(is_shuffled=True),
    )
    up = SimpleNamespace(
        weight=torch.zeros(HF_CONFIG["hidden_size"], 8, dtype=torch.bfloat16)
    )
    return SimpleNamespace(
        self_attn=attn,
        mlp=SimpleNamespace(
            use_latent_moe=True,
            routed_expert_norm=object(),
            routed_expert_up_proj=up,
            experts=experts,
        ),
        use_attn_residuals=True,
        _sp_moe=False,
        all_reduce_fusion=False,
    )


def _reasons(monkeypatch, layer):
    from sglang.srt.layers import dp_attention
    from sglang.srt.utils import common

    # where spec.py imports it from (common: SGLang before sgl-project/sglang#48781)
    source = dp_attention if hasattr(dp_attention, "require_mlp_sync") else common
    monkeypatch.setattr(source, "require_mlp_sync", lambda: False)
    model = SimpleNamespace(model=SimpleNamespace(layers={i: layer for i in range(93)}))
    return k3._model_disable_reasons(model, K3, {})


def test_a_covered_layer_passes(monkeypatch):
    assert _reasons(monkeypatch, _fake_layer()) == []
    assert _reasons(monkeypatch, _fake_layer(kda=True)) == []


def test_an_mla_layer_without_the_output_gate_is_refused(monkeypatch):
    # hasattr(attn, "forward_pre_o_proj") is true for both attention classes,
    # so it cannot carry this: for MLA the split rides on the gate's wrapper.
    reasons = _reasons(monkeypatch, _fake_layer(gate=False))
    assert any("output gate" in r for r in reasons), reasons


def test_the_guard_flags_sit_past_every_width(monkeypatch):
    """Every width reads the guard flags at the same offset.

    The runner allocates one peer buffer for all the widths and puts the flags
    after the data. A launch that derived the offset from its own width wrote
    its flags inside the data region of a wider launch. Two widths in one run
    then disagreed on where the flags are, and the narrower one read live data
    as an error word.
    """
    from sgl_amd_mono.kimi_k3.gfx950 import layer
    from sgl_amd_mono.kimi_k3.gfx950.stages import moe

    off = layer.flags_offset()
    for width in range(1, K2_MAX_ROWS + 1):
        assert layer.peer_bytes(width) <= off, width
        assert moe.peer_bytes(width) <= off, width
    # the widest launch ends exactly there: a larger offset wastes a peer page
    assert off == max(layer.peer_bytes(K2_MAX_ROWS), moe.peer_bytes(K2_MAX_ROWS))


@pytest.mark.parametrize(
    "rows,front_rows",
    [(K2_MAX_ROWS, K2_MAX_ROWS), (12, 30), (18, 18), (18, 30), (24, 30), (48, 48)],
)
def test_the_scratch_regions_are_disjoint_at_every_width(rows, front_rows):
    """No launch family's scratch overlaps another's at any built width.

    The DCP front's pv rows once covered K2's tagged x rows: a front word left
    there could pass for a pair of K2's tag. Each region holds its family's
    layout at every width and bank depth it is built for.
    """
    from sgl_amd_mono.kimi_k3 import MAX_BLOCKS
    from sgl_amd_mono.kimi_k3.gfx950 import k1, kda_layer, layer
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    reg = layer.scratch_regions(rows, front_rows)
    spans = sorted(reg.values())
    assert all(o % 4096 == 0 for o, _ in spans)
    for (o0, n0), (o1, _) in zip(spans, spans[1:]):
        assert o0 + n0 <= o1, reg
    blocks = range(1, MAX_BLOCKS + 1)
    k2_end = max(
        layer.scratch_bytes(layer.K2Build(tokens=s, nblocks=nb, core_box=cb))
        for s in range(1, rows + 1) for nb in blocks for cb in (False, True)
    )  # fmt: skip
    assert k2_end <= reg["k2"][1]
    assert reg["xt"][1] == rows * HIDDEN // 2 * 8
    assert k1.scratch_bytes(rows) <= reg["k1"][1]
    fronts = [s for s in range(mf.G_ROWS, front_rows + 1, mf.G_ROWS) if s // mf.G_ROWS <= mf.MAX_REQS]
    for s in fronts:
        for nb in blocks:
            key = mf.MlaFrontBuild(tokens=s, nblocks=nb, scale=1.0)
            assert mf.scratch_bytes(key) <= reg["front"][1], (s, nb)
    for s in range(k1.WIDE_FROM_REQS * kw.WT, rows + 1, kw.WT):
        for nb in blocks:
            akey = kw.KdaWideBuild(tokens=s, qlen=kw.WT, front=True, nblocks=nb)
            assert kda_layer.region_bytes(akey) <= reg["attn"][1], (s, nb)
    # K2's tail seam: TPART a pair a (slice, token, partial), TSTREAM a pair a
    # bf16 pair of every stream row, at the widest step and deepest bank
    from sgl_amd_mono.kimi_k3.gfx950.stages import seam
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import NA

    for s in range(1, rows + 1):
        for nb in blocks:
            assert NA * s * seam.front_nq(nb + 1) * 8 <= reg["tpart"][1], (s, nb)
        assert s * HIDDEN // 2 * 8 <= reg["tstream"][1], s
    # and K2's own layout keeps none of them, tail or not
    for s in range(1, rows + 1):
        tail = layer.K2Build(tokens=s, nblocks=MAX_BLOCKS, core_box=True, tail=layer.TailSeam())
        assert layer.scratch_bytes(tail) <= reg["k2"][1]


def test_every_launch_takes_its_region_from_the_one_layout():
    """The offsets the kernels and the runner read are the layout's."""
    from sgl_amd_mono.kimi_k3 import MAX_BLOCKS
    from sgl_amd_mono.kimi_k3.gfx950 import kda_layer, layer
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

    reg = layer.scratch_regions()
    assert layer.tag_offset() == reg["xt"][0]
    assert kda_layer.attn_offset() == reg["attn"][0]
    assert layer.scratch_alloc_bytes() >= max(o + n for o, n in reg.values())
    akey = kw.KdaWideBuild(tokens=K2_MAX_ROWS // kw.WT * kw.WT, qlen=kw.WT, front=True, nblocks=MAX_BLOCKS)
    o, n = reg["attn"]
    assert all(o <= a and a + b <= o + n for a, b in kda_layer.scratch_layout(akey).values())


# A TP8 + DCP8 DSpark server on the ReplaySSM ring, as SGLang's feature probes
# read it.
TP8_DCP8 = SimpleNamespace(
    enable_linear_replayssm_spec=True,
    tp_size=8, dcp_size=8, attn_cp_size=1, enable_prefill_cp=False, pp_size=1,
    nnodes=1, attn_dp_size=1, ep_size=1, enable_eplb=False,
    speculative_algorithm="DSPARK", enable_two_batch_overlap=False,
    enable_single_batch_overlap=False, enable_lora=False, lora_paths=None,
    disaggregation_mode="null", enable_return_routed_experts=False,
    enable_return_indexer_topk=False, enable_hierarchical_cache=False,
    disaggregation_decode_enable_offload_kvcache=False, enable_lmcache=False,
    enable_flexkv=False, enable_memory_saver=False, enable_weights_cpu_backup=False,
    cpu_offload_gb=0, offload_group_size=0, enable_torch_compile=False,
)  # fmt: skip


def _config_reasons(monkeypatch, cfg):
    """The config reasons with the ROCm image's environment: aiter's interleaved
    experts, SGLANG_AITER_K3_OPT and FlyDSL."""
    import importlib.util

    from sglang.srt.layers.quantization import mxfp4

    monkeypatch.setattr(mxfp4, "_aiter_situ_uses_gu_interleaved_weights", lambda: True)
    monkeypatch.setenv("SGLANG_AITER_K3_OPT", "1")
    find = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "flydsl" else find(name, *a),
    )
    return k3._config_disable_reasons(cfg, K3)


def test_dcp8_is_accepted(monkeypatch):
    # SGLang counts --dcp-size > 1 as CONTEXT_PARALLEL and refuses a spec that
    # does not declare it
    binding = pytest.importorskip("sglang.srt.kernel_plugins.amd.binding")
    monkeypatch.setenv("SGLANG_RAGGED_VERIFY_MODE", "static")
    assert binding.features_in_use(TP8_DCP8) - k3.SPEC.features == set()
    assert _config_reasons(monkeypatch, TP8_DCP8) == []
    assert _reasons(monkeypatch, _fake_layer()) == []
    assert _reasons(monkeypatch, _fake_layer(kda=True)) == []


def test_other_context_parallel_shapes_are_refused(monkeypatch):
    dcp4 = SimpleNamespace(**{**vars(TP8_DCP8), "dcp_size": 4})
    assert any("--dcp-size 4" in r for r in _config_reasons(monkeypatch, dcp4))
    cp = SimpleNamespace(**{**vars(TP8_DCP8), "attn_cp_size": 2})
    assert any("attention context" in r for r in _config_reasons(monkeypatch, cp))


def test_the_replayssm_ring_is_required(monkeypatch):
    snapshots = SimpleNamespace(**{**vars(TP8_DCP8), "enable_linear_replayssm_spec": False})
    assert any(
        "--enable-linear-replayssm-spec" in r for r in _config_reasons(monkeypatch, snapshots)
    )
    monkeypatch.setenv("SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT", "1")
    from sglang.srt.mem_cache import kv_cache_configurator as kvc

    monkeypatch.setattr(kvc, "host_only_mamba_checkpoint", lambda: True)
    assert any("by request" in r for r in _config_reasons(monkeypatch, TP8_DCP8))


def test_kda_dwdp_is_refused(monkeypatch):
    dwdp = SimpleNamespace(**{**vars(TP8_DCP8), "kda_dwdp_size": 8})
    assert any("--kda-dwdp-size" in r for r in _config_reasons(monkeypatch, dwdp))


def test_the_mla_front_is_off_under_dcp():
    # the front computes only this rank's 12 heads' q; SGLang's DCP MLA forward
    # needs all 96, so those layers keep SGLang's MLA attention
    reason = k3._mla_front_disable_reason(SimpleNamespace(attn_dcp_size=8))
    assert reason is not None and "DCP" in reason
    if k3.MLA_PRE:
        assert k3._mla_front_disable_reason(SimpleNamespace(attn_dcp_size=1)) is None
