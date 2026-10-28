"""The GLM-5.3 gfx950 monokernel spec on CPU (RFC #43949): the layers its forward
covers on the GLM-5.3 layer structure, its row routing and TP4 row cap, the keys
each row of a decode or MTP verify step attends, the MLA cache layout an FP8 launch
binds, its registry entry, its config and model refusal reasons, a routed step's
launches and what it hands the logits processor, the dense MLP's expert slices and the embedding and LM-head shards, and the layer
check's top-k metric, cache poisoning, tail check and reported outputs.
SGLang checks its declared requirements, the same way for every spec."""

import logging
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from sglang.srt.layers.moe.utils import MoeRunnerBackend

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.glm53 import HF_CONFIG, check
from sgl_amd_mono.glm53 import spec as glm53
from sgl_amd_mono.glm53.gfx950.weights import rope_tables

# GLM-5.3's layer structure and quantization (its config.json)
GLM53 = SimpleNamespace(
    **HF_CONFIG,
    num_hidden_layers=78,
    first_k_dense_replace=3,
    moe_layer_freq=1,
    quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
    rope_parameters={"rope_theta": 8000000, "rope_type": "default"},
    architectures=["GlmMoeDsaForCausalLM"],
)
# the server arguments the spec reads before load: no speculative decoding
CFG = SimpleNamespace(page_size=None, speculative_algorithm=None)
# MTP: SGLang's GLM-5.3 default draft chain (NEXTN resolves to EAGLE)
MTP = SimpleNamespace(
    page_size=None,
    speculative_algorithm="EAGLE",
    speculative_eagle_topk=1,
    speculative_num_steps=3,
    speculative_num_draft_tokens=4,
)
# the graph keys of a verify step of one request (4 rows) and a decode step of 4
VERIFY = registry.ShapeKey(
    forward_mode="target_verify", batch_size=1, tokens_per_request=4
)
DECODE = registry.ShapeKey(forward_mode="decode", batch_size=4, tokens_per_request=1)


def test_the_forward_covers_every_layer():
    # the dense layers 0 to 2, then the MoE layers
    assert glm53._covered_layers(GLM53) == list(range(78))


def test_the_plugin_registers_the_spec():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    arch = ["GlmMoeDsaForCausalLM"]
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=arch, isa="gfx950"
    )
    assert found is glm53.SPEC, why
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=arch, isa="gfx942"
    )
    assert found is None and "gfx942" in why


class _Logits:
    """A logits processor whose own logits are the rows' sums."""

    def _get_logits(self, hidden_states, lm_head, logits_metadata, **kw):
        return hidden_states.sum(-1, keepdim=True)


class _Lm:
    """A model whose own forward records that it ran."""

    def __init__(self):
        self.model, self.ran = SimpleNamespace(), []
        self.logits_processor = _Logits()

    def forward(self, *args):
        self.ran.append(args)
        return "the model's forward"


def _built(lm, layers, runtimes, tp=8, context_len=202752, heads=None):
    return glm53._Built(
        lm,
        layers,
        runtimes,
        heads or {},
        None,
        None,
        tp,
        "cpu",
        context_len=context_len,
    )


KINDS = ("embed", "dense", "index", "shared", "head")


def test_a_step_runs_at_the_next_width_and_tp4_takes_four_rows():
    # a one-row launch never completes on gfx950: one row runs at width 2
    assert [glm53._width(r) for r in glm53.ROWS] == [2, 2, 4, 4, 8, 8, 8, 8]
    # TP4 builds no 8-row launch: a wider step runs the model's own forward
    runtimes = {(w, k): None for w in (2, 4) for k in KINDS}
    layers = {i: SimpleNamespace(i=i) for i in range(78)}
    lm = _Lm()
    built = _built(lm, layers, runtimes, tp=4)
    key = registry.ShapeKey(forward_mode="decode", batch_size=5, tokens_per_request=1)
    out = built.forward(registry.Plan(key=key), "ids", "positions", "batch")
    assert out == "the model's forward"
    assert lm.ran == [("ids", "positions", "batch", None, None)]
    assert _built(lm, layers, runtimes, tp=8).max_rows == 8


def test_a_step_from_embeddings_runs_the_model_s_forward():
    # layer 0's launch reads the step's tokens; given embeddings, SGLang's path runs
    lm = _Lm()
    built = _built(lm, {}, {(2, k): None for k in KINDS})
    key = registry.ShapeKey(forward_mode="decode", batch_size=2, tokens_per_request=1)
    out = built.forward(registry.Plan(key=key), "ids", "positions", "batch", "emb")
    assert out == "the model's forward" and lm.ran[0][3] == "emb"


def test_the_logits_processor_returns_the_tail_s_logits_on_a_routed_step():
    lm = _Lm()
    built = _built(lm, {}, {(2, k): None for k in KINDS})
    lp, h = lm.logits_processor, torch.ones(3, 5)
    # not routed: the processor's own logits
    assert torch.equal(lp._get_logits(h, None, None), torch.full((3, 1), 5.0))
    # routed: the tail's, the step's rows of the width's
    built._logits = torch.arange(8.0).view(4, 2)
    assert torch.equal(lp._get_logits(h, None, None), torch.arange(6.0).view(3, 2))


def _step_rows(key, positions, slots, seq_lens, context_len=202752):
    """``_Built._step`` on CPU for a 4-row step: each row's position and its keys
    (indptr). ``seq_lens`` is what SGLang passes: one length per request."""
    W = 4
    runtimes = {(W, k): None for k in KINDS}
    built = _built(_Lm(), {3: None}, runtimes, context_len=context_len)
    batch = SimpleNamespace(
        out_cache_loc=torch.tensor(slots), seq_lens=torch.tensor(seq_lens)
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(real_page_table=torch.zeros(W, 1))
    )
    with mock.patch(
        "sglang.srt.model_executor.forward_context.get_attn_backend",
        return_value=backend,
    ):
        ids = torch.arange(len(slots), dtype=torch.int32)
        step = built._step(
            registry.Plan(key=key), ids, torch.tensor(positions), batch, len(slots), W
        )
    # layer 0's token ids: the step's, -1 for the width's rows past it
    assert built.ids[W].tolist() == list(range(len(slots))) + [-1] * (W - len(slots))
    return step.positions.tolist(), step.indptr.tolist()


@pytest.mark.parametrize(
    "key, positions, slots, seq_lens, indptr",
    [
        # one request of prefix 10 verifies 4 drafts at positions 10 to 13
        (VERIFY, [10, 11, 12, 13], [100, 101, 102, 103], [10], [0, 11, 23, 36, 50]),
        # one row a request (seq_lens = position + 1); slot 0 is a pad row
        (DECODE, [5, 7, 0, 0], [40, 41, 0, 0], [6, 8, 1, 1], [0, 6, 14, 14, 14]),
        # a row attends at most the indexer's 2048 keys
        (
            VERIFY,
            [4000, 4001, 4002, 4003],
            [7, 8, 9, 10],
            [4000],
            [0, 2048, 4096, 6144, 8192],
        ),
    ],
    ids=["verify step", "decode step and a pad row", "rows past 2048 keys"],
)
def test_each_row_attends_the_keys_up_to_its_position(
    key, positions, slots, seq_lens, indptr
):
    """A verify row attends its request's prefix and the drafts up to itself, not
    the prefix alone: in a verify step SGLang's seq_lens holds only the prefix."""
    assert _step_rows(key, positions, slots, seq_lens)[1] == indptr


def test_verify_lookahead_past_the_context_stays_in_the_tables():
    """Verify rows past the context keep positions inside the rotary table and the
    indexer's score rows (12 tokens here); the scheduler discards their tokens."""
    positions, indptr = _step_rows(
        VERIFY, [10, 11, 12, 13], [100, 101, 102, 103], [10], context_len=12
    )
    assert positions == [10, 11, 11, 11]
    assert indptr == [0, 11, 23, 35, 47]


@pytest.mark.parametrize(
    "dtype, row, bound",
    [
        # SGLang's HIP DSA E4M3 cache: 576 values a row, no block scales
        (torch.float8_e4m3fn, 576, True),
        # the scaled E4M3 row of the other DSA backends: 512 values, 4 f32 scales,
        # then the bf16 rope
        (torch.float8_e4m3fn, 656, False),
        (torch.bfloat16, 576, False),
    ],
    ids=["raw E4M3", "scaled E4M3", "bf16"],
)
def test_an_fp8_launch_binds_only_the_raw_e4m3_mla_cache(dtype, row, bound):
    """The kernels address 576-byte E4M3 rows: another row or dtype would be read
    as wrong keys without an error."""
    kv = torch.zeros(130, 1, row, dtype=dtype)
    pool = SimpleNamespace(get_key_buffer=lambda i: kv)
    layer = glm53._Layer(5, None, SimpleNamespace(index=None))
    with mock.patch(
        "sglang.srt.model_executor.forward_context.get_token_to_kv_pool",
        return_value=pool,
    ):
        if bound:
            glm53._Built._bind(layer, kv_dtype=torch.float8_e4m3fn)
            assert layer.caches.kv.dtype == torch.float8_e4m3fn
            assert layer.caches.kv.shape == (130, 576)
        else:
            with pytest.raises(RuntimeError, match="MLA cache"):
                glm53._Built._bind(layer, kv_dtype=torch.float8_e4m3fn)


@pytest.mark.parametrize(
    "changes, refused",
    [
        ({}, False),
        ({"speculative_eagle_topk": 2}, True),
        ({"speculative_num_draft_tokens": 3}, True),
    ],
    ids=["default chain", "tree draft", "draft tokens"],
)
def test_mtp_runs_one_draft_chain(changes, refused):
    cfg = SimpleNamespace(**{**vars(MTP), **changes})
    reasons = glm53._config_disable_reasons(cfg, GLM53)
    assert len(reasons) == int(refused)
    assert all("one draft chain" in r for r in reasons)


def test_the_layer_check_counts_the_own_chain_keys_a_row_drops():
    # row 0 drops SGLang's 11 and its own slot 100; row 1 drops SGLang's 21 for row
    # 0's slot 100, which is no drop of its own chain; row 2 is a pad row
    ref = torch.full((3, 2048), -1)
    ref[0, :3] = torch.tensor([10, 11, 100])
    ref[1, :2] = torch.tensor([20, 21])
    topk = torch.zeros(2 * 2048 + 1, dtype=torch.int32)
    topk[:5] = torch.tensor([10, 12, 13, 20, 100])
    step = SimpleNamespace(indptr=torch.tensor([0, 3, 5, 5]), topk=topk)
    out = check._topk_outputs(ref, step, torch.tensor([100, 101, 0]), 3)
    assert out == {"topk": {"rel": 2 / 3}, "topk_own": {"max_abs": 1.0}}


def test_the_layer_check_poisons_only_this_steps_index_scales():
    # page 1 token 5 and page 2 token 63 are this step's keys; slot 0 is a pad row.
    # Bytes below 120 keep every f32 scale and FP8 value finite
    pages = torch.randint(0, 120, (3, check.PAGE_TOKENS * 132), dtype=torch.uint8)
    before = pages.clone()
    check.poison_new_index_scales(pages, torch.tensor([0, 64 + 5, 128 + 63]))
    _, scales = check._index_keys(before)
    _, new_scales = check._index_keys(pages)
    poisoned = torch.zeros_like(scales, dtype=torch.bool)
    poisoned[1, 5] = poisoned[2, 63] = True
    assert (new_scales[poisoned] == check.POISON_SCALE).all()
    assert torch.equal(new_scales[~poisoned], scales[~poisoned])
    # the FP8 values stay, so only the scale of a key read from the cache changes
    values = slice(0, check.INDEX_VALUE_BYTES)
    assert torch.equal(pages[:, values], before[:, values])


def test_config_reasons_name_each_mismatch():
    assert glm53._config_disable_reasons(CFG, GLM53) == []
    other = SimpleNamespace(
        **{
            **vars(GLM53),
            "index_topk_freq": 8,
            "moe_layer_freq": 2,
            "quantization_config": {"quant_method": "quark"},
            "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
        }
    )
    cfg = SimpleNamespace(**{**vars(CFG), "page_size": 1})
    reasons = glm53._config_disable_reasons(cfg, other)
    text = " ".join(reasons)
    assert "index_topk_freq" in text
    assert "MoE" in text
    assert "--page-size 1" in text
    assert "FP8 block-128" in text
    assert "RoPE 'yarn'" in text


def _linear(shuffled=True):
    # SGLang marks only the block-FP8 linears it rewrote into AITER's layout
    if not shuffled:
        return SimpleNamespace(weight=SimpleNamespace())
    return SimpleNamespace(
        aiter_bpreshuffled=True, weight=SimpleNamespace(is_shuffled=True)
    )


def _layer(
    experts_dtype=torch.float8_e4m3fn,
    n_experts=257,
    indexer=True,
    shuffled=True,
    runner=MoeRunnerBackend.AITER,
    fused_indexer=True,
):
    w13 = torch.empty(n_experts, 1, 1, dtype=experts_dtype)
    quant = SimpleNamespace(runner=SimpleNamespace(runner_backend=runner))
    ind = SimpleNamespace(
        wq_b=_linear(), wk=_linear(), use_gfx950_fused_indexer=fused_indexer
    )
    attn = SimpleNamespace(
        fused_qkv_a_proj_with_mqa=_linear(shuffled),
        q_b_proj=_linear(),
        o_proj=_linear(),
        indexer=ind if indexer else None,
    )
    experts = SimpleNamespace(w13_weight=w13, quant_method=quant)
    return SimpleNamespace(mlp=SimpleNamespace(experts=experts), self_attn=attn)


def _fp8_linear(rows, cols, dtype=torch.float8_e4m3fn, shuffled=True):
    linear = SimpleNamespace(
        weight=torch.empty(rows, cols, dtype=dtype), aiter_bpreshuffled=shuffled
    )
    linear.weight_scale_inv = torch.ones(-(-rows // 128), -(-cols // 128))
    return linear


# a dense layer's MLP at TP8: 1,536 intermediates a rank (6 expert slices of 256)
DENSE_TP8 = 12288 // 8


def _dense_layer(indexer=True, inter=DENSE_TP8, dtype=torch.float8_e4m3fn):
    from sgl_amd_mono.glm53.gfx950.config import HIDDEN

    layer = _layer(indexer=indexer)
    layer.mlp = SimpleNamespace(
        gate_up_proj=_fp8_linear(2 * inter, HIDDEN, dtype),
        down_proj=_fp8_linear(HIDDEN, inter),
    )
    return layer


def _lp(**kw):
    """SGLang's logits processor without DP attention, a DP LM head, an f32 LM head,
    a logit scale or soft-capping."""
    return SimpleNamespace(
        **{
            "do_tensor_parallel_all_gather": True,
            "do_tensor_parallel_all_gather_dp_attn": False,
            "use_attn_tp_group": False,
            "use_fp32_lm_head": False,
            "use_fp32_lm_head_output": False,
            "logit_scale": None,
            "final_logit_softcapping": None,
            **kw,
        }
    )


def _model(layers, lp=None):
    return SimpleNamespace(
        model=SimpleNamespace(layers=layers), logits_processor=lp or _lp()
    )


def _layers():
    # GLM-5.3's dense layers 0 to 2, each with the indexer; then the MoE layers, the
    # indexer every fourth from layer 6
    return [_dense_layer() for _ in range(3)] + [
        _layer(indexer=i % 4 == 2) for i in range(3, 78)
    ]


@pytest.fixture
def tp8(monkeypatch):
    """TP8 rank 0, the LM head and embedding shards accepted."""
    import sglang.srt.runtime_context as runtime_context

    import sgl_amd_mono.glm53.gfx950.weights as weights

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=8, tp_rank=0),
    )
    monkeypatch.setattr(weights, "head_weights", lambda *args: None)
    monkeypatch.setattr(weights, "embedding", lambda *args: None)
    return weights


def test_model_reasons(tp8):
    good = _layers()
    assert glm53._model_disable_reasons(_model(good), GLM53, {}) == []
    bad = list(good)
    bad[5] = _layer(experts_dtype=torch.bfloat16)
    bad[9] = _layer(n_experts=256)
    bad[0] = _dense_layer(indexer=False)
    reasons = " | ".join(glm53._model_disable_reasons(_model(bad), GLM53, {}))
    assert "layer 5: needs block-FP8 experts" in reasons
    assert "layer 9: needs the shared expert fused" in reasons
    # layer 0 runs the indexer: the MoE layers before the first indexer one read its
    # top-k (and layers 1 and 2 recompute it)
    assert "layer 0: a dense layer without its indexer" in reasons
    assert "layer 0 reuses a top-k no layer computed" in reasons


def test_model_reasons_of_the_tail_the_embedding_and_the_dense_layers(tp8, monkeypatch):
    def refuse(why):
        def f(*args):
            raise ValueError(why)

        return f

    monkeypatch.setattr(tp8, "head_weights", refuse("an fp8 LM head"))
    monkeypatch.setattr(tp8, "embedding", refuse("padded rows"))
    layers = _layers()
    # a bf16 dense MLP; 1,600 intermediates a rank, not slices of the experts' 256
    layers[1] = _dense_layer(dtype=torch.bfloat16)
    layers[2] = _dense_layer(inter=1600)
    lp = _lp(
        use_attn_tp_group=True,
        use_fp32_lm_head_output=True,
        logit_scale=0.5,
        final_logit_softcapping=30.0,
    )
    reasons = glm53._model_disable_reasons(_model(layers, lp), GLM53, {})
    joined = " | ".join(reasons)
    for want in (
        "layer 1: dense MLP: needs gate_up_proj weight as contiguous",
        "layer 2: dense MLP: needs its 1600 intermediates a rank as at most 9 slices "
        "of 256",
        "all-gathers the logits",
        "LM head in bf16",
        "logit_scale",
        "final_logit_softcapping",
        "LM head: an fp8 LM head",
        "embedding: padded rows",
    ):
        assert want in joined, joined
    assert len(reasons) == 8, reasons


def test_model_reasons_name_the_layouts_the_kernel_cannot_read(tp8):
    # These served wrong tokens without a refusal: the kernel read row-major FP8
    # projections (--fp8-gemm-backend triton) and experts off the AITER MoE
    # runner as (16, 16) tiles, and scored rotated index queries against keys that
    # an indexer other than SGLang's gfx950 fused one caches unrotated
    layers = _layers()
    layers[4] = _layer(indexer=False, shuffled=False)
    layers[5] = _layer(indexer=False, shuffled=False)
    layers[7] = _layer(indexer=False, runner=MoeRunnerBackend.TRITON)
    layers[10] = _layer(fused_indexer=False)
    reasons = glm53._model_disable_reasons(_model(layers), GLM53, {})
    assert [r.split(":")[0] for r in reasons] == [
        "layer 7",
        "fused_qkv_a_proj_with_mqa of 2 layers from 4",
        "the indexer of layer 10",
    ], reasons
    assert "AITER MoE runner" in reasons[0] and "(16, 16)" in reasons[1]
    assert "gfx950 fused DSA indexer" in reasons[2]


def test_every_indexer_rotary_must_equal_the_mla_one():
    # The first covered layer reuses layer 2's top-k and has no indexer, so an
    # indexer rotary unlike the MLA one went unchecked
    table = torch.randn(16, 64)

    def layer(indexer_table=None):
        rotary = SimpleNamespace(cos_sin_cache=table)
        indexer = None
        if indexer_table is not None:
            indexer = SimpleNamespace(
                rotary_emb=SimpleNamespace(cos_sin_cache=indexer_table)
            )
        return SimpleNamespace(
            self_attn=SimpleNamespace(rotary_emb=rotary, indexer=indexer)
        )

    cos, _ = rope_tables([layer(), layer(table.clone())], {})
    assert torch.equal(cos, table[:, :32].to(torch.bfloat16))
    with pytest.raises(ValueError, match="indexer's rotary"):
        rope_tables([layer(), layer(table + 1)], {})


def _shuffle(w: torch.Tensor) -> torch.Tensor:
    """A row-major [rows, K] FP8 weight in AITER's (16, 16) layout (check.unshuffle's
    inverse)."""
    n, k = w.shape
    b = w.view(torch.uint8).view(n // 16, 16, k // 32, 2, 16)
    return b.permute(0, 2, 3, 1, 4).reshape(n, k).view(w.dtype)


@pytest.mark.parametrize("shuffled", [False, True], ids=["row-major", "shuffled"])
def test_the_dense_mlp_is_copied_as_expert_slices(shuffled):
    # TP8: 1,536 intermediates a rank as 6 slices of the experts' 256. Slice e holds
    # gate rows e * 256 .. and the matching up rows, then down's matching K columns,
    # each in the layout the experts' path reads
    from sgl_amd_mono.glm53.gfx950.config import HIDDEN
    from sgl_amd_mono.glm53.gfx950.weights import _dense_mlp

    inter, n, I = 256, 6, DENSE_TP8
    g = torch.Generator().manual_seed(0)
    gu = torch.randint(0, 120, (2 * I, HIDDEN), generator=g, dtype=torch.uint8)
    dn = torch.randint(0, 120, (HIDDEN, I), generator=g, dtype=torch.uint8)
    gu, dn = gu.view(torch.float8_e4m3fn), dn.view(torch.float8_e4m3fn)
    mlp = SimpleNamespace(
        gate_up_proj=_fp8_linear(2 * I, HIDDEN, shuffled=shuffled),
        down_proj=_fp8_linear(HIDDEN, I, shuffled=shuffled),
    )
    mlp.gate_up_proj.weight = _shuffle(gu) if shuffled else gu
    mlp.down_proj.weight = _shuffle(dn) if shuffled else dn
    mlp.gate_up_proj.weight_scale_inv = torch.rand(2 * I // 128, HIDDEN // 128)
    mlp.down_proj.weight_scale_inv = torch.rand(HIDDEN // 128, I // 128)
    copies = {}
    w_ug, s_ug, w_dn, s_dn, slices, sh = _dense_mlp(mlp, 0, inter, copies)
    assert (slices, sh) == (n, shuffled)
    assert w_ug.shape == (n, 2 * inter, HIDDEN) and w_dn.shape == (n, HIDDEN, inter)
    assert len(copies) == 4
    u8 = lambda t: t.view(torch.uint8)  # noqa: E731
    for e in range(n):
        rows = slice(e * inter, (e + 1) * inter)
        ug, d = w_ug[e], w_dn[e]
        if shuffled:
            ug, d = check.unshuffle(ug), check.unshuffle(d)
        assert torch.equal(u8(ug[:inter]), u8(gu[rows]))
        assert torch.equal(u8(ug[inter:]), u8(gu[I:][rows]))
        assert torch.equal(u8(d), u8(dn[:, rows]))
        blocks = slice(e * inter // 128, (e + 1) * inter // 128)
        s = mlp.gate_up_proj.weight_scale_inv
        assert torch.equal(s_ug[e], torch.cat([s[blocks], s[I // 128 :][blocks]]))
        assert torch.equal(s_dn[e], mlp.down_proj.weight_scale_inv[:, blocks])
    # gate | up and down in two layouts refuse
    mlp.down_proj.aiter_bpreshuffled = not shuffled
    with pytest.raises(ValueError, match="one FP8 layout"):
        _dense_mlp(mlp, 0, inter, {})


def _vocab_module(tp, rank=1, padding=0, added=0, method="UnquantizedEmbeddingMethod"):
    from sgl_amd_mono.glm53.gfx950.config import HIDDEN

    rows = 64
    si = SimpleNamespace(
        org_vocab_start_index=rank * rows,
        org_vocab_end_index=(rank + 1) * rows,
        added_vocab_start_index=tp * rows,
        added_vocab_end_index=tp * rows + added,
        num_org_vocab_padding=padding,
    )
    return SimpleNamespace(
        quant_method=type(method, (), {})(),
        enable_tp=True,
        use_attn_tp_group=False,
        tp_size=tp,
        shard_indices=si,
        output_dtype=None,
        weight=torch.zeros(rows, HIDDEN, dtype=torch.bfloat16),
    )


def test_the_embedding_and_the_lm_head_are_plain_vocab_shards():
    from sgl_amd_mono.glm53.gfx950.config import HIDDEN
    from sgl_amd_mono.glm53.gfx950.weights import embedding, head_weights

    tp, rank = 8, 1

    def model(emb, head):
        norm = SimpleNamespace(weight=torch.zeros(HIDDEN, dtype=torch.bfloat16))
        return SimpleNamespace(
            model=SimpleNamespace(embed_tokens=emb, norm=norm), lm_head=head
        )

    m = model(_vocab_module(tp), _vocab_module(tp))
    assert embedding(m, tp, rank).data_ptr() == m.model.embed_tokens.weight.data_ptr()
    assert head_weights(m, tp, rank).w_head.data_ptr() == m.lm_head.weight.data_ptr()
    for bad, match in (
        (_vocab_module(tp, padding=64), "no added or padded rows"),
        (_vocab_module(tp, added=8), "no added or padded rows"),
        (_vocab_module(tp, rank=2), "in rank order"),
        (_vocab_module(tp, method="Fp8LinearMethod"), "unquantized"),
        (_vocab_module(4), "sharded over the TP group"),
    ):
        with pytest.raises(ValueError, match=match):
            embedding(model(bad, _vocab_module(tp)), tp, rank)
        with pytest.raises(ValueError, match=match):
            head_weights(model(_vocab_module(tp), bad), tp, rank)
    emb = _vocab_module(tp)
    emb.output_dtype = torch.float32
    with pytest.raises(ValueError, match="weight's dtype"):
        embedding(model(emb, _vocab_module(tp)), tp, rank)


def test_every_bound_names_an_output_the_check_reports():
    # a bound the check never reports would be skipped silently
    for name in glm53.NUMERICS.bounds:
        kind, out, metric = name.split(".")
        assert out in check.REPORTED[kind] and metric in check.METRICS, name


def test_the_check_rejects_an_output_it_does_not_list():
    with pytest.raises(ValueError, match="dense.mlp_out.rel"):
        check._bounds("layer 1", {"dense.mlp_out.rel": 0.0})


def test_the_dense_check_reads_the_launch_s_o_proj_partials():
    # the o_proj partials of every rank, (packed bf16 pair, tag) pairs in the slot of
    # the launch's parity; the other slot holds another launch's
    from sgl_amd_mono.glm53.gfx950.config import HIDDEN

    tp, width, S, n = 2, 2, 1, 2 * 2 * HIDDEN
    parts = torch.arange(n, dtype=torch.float32).div(64).bfloat16().view(torch.int32)
    slot = torch.stack([parts, torch.full_like(parts, 7)], 1).view(torch.uint8).view(-1)
    other = torch.zeros_like(slot)
    stride = slot.numel()
    for epoch, layer, first in ((4, 1, other), (4, 0, slot), (5, 1, slot)):
        rt = SimpleNamespace(
            epoch=torch.tensor([epoch]),
            key=SimpleNamespace(tp=tp, launches_per_step=3),
            width=width,
            sy={"attn": 0, "_part_stride": stride},
            peer=SimpleNamespace(
                bytes=torch.cat([first, other if first is slot else slot])
            ),
        )
        got = check.launch_o_proj(SimpleNamespace(rt=rt, layer=layer), S)
        # (epoch x 3 + layer) & 1 picks slot 0 for (4, 0) and (5, 1), slot 1 for (4, 1)
        want = parts.view(torch.bfloat16).float().view(tp, width, HIDDEN)[:, :S].sum(0)
        assert torch.equal(got, want), (epoch, layer)


def test_the_tail_check_catches_a_wrong_final_norm(monkeypatch, caplog):
    # check.tail on one rank: the tail's logits against SGLang's final norm and LM
    # head, and against an f32 tail. A tail whose norm weight is 1% off fails
    import sglang.srt.distributed.communication_op as comm
    import sglang.srt.runtime_context as runtime_context

    from sgl_amd_mono.glm53.gfx950.config import HIDDEN

    monkeypatch.setattr(comm, "tensor_model_parallel_all_gather", lambda t: t)
    monkeypatch.setattr(
        runtime_context, "get_parallel", lambda: SimpleNamespace(tp_rank=0)
    )
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, HIDDEN, generator=g).bfloat16()
    w_norm = (1 + 0.1 * torch.randn(HIDDEN, generator=g)).bfloat16()
    w_head = (torch.randn(512, HIDDEN, generator=g) * HIDDEN**-0.5).bfloat16()
    lm = SimpleNamespace(
        model=SimpleNamespace(norm=lambda h: check._rms(h, w_norm).bfloat16()),
        lm_head=SimpleNamespace(weight=w_head),
    )
    lm.model.norm.weight = w_norm

    def tail(g_norm):
        normed = check._rms(x, g_norm).bfloat16()
        return (normed @ w_head.t()).float(), normed

    def violations(g_norm):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=check.__name__):
            check.tail(lm, x, *tail(g_norm))
        assert "monokernel check tail (3 rows)" in caplog.text
        return [r for r in caplog.messages if "violation" in r]

    assert violations(w_norm) == []
    bad = violations((w_norm.float() * 1.01).bfloat16())
    assert any("head.logits.rel" in v for v in bad), bad


class _Op:
    """A layer's launch: records its input, returns it plus one (layer 0: its token
    ids as rows)."""

    def __init__(self, i, calls, rt):
        self.i, self.calls, self.rt = i, calls, rt

    def launch(self, h, caches, cos, sin, step):
        from sgl_amd_mono.glm53.gfx950.config import HIDDEN

        self.calls.append((self.i, h.clone(), step))
        if h.dim() == 1:
            h = h.float()[:, None].expand(-1, HIDDEN)
        return (h + 1).bfloat16()


class _Head:
    """The tail: the logits are a row's x and twice it, the normed rows twice x."""

    def __init__(self, calls):
        self.calls = calls

    def launch(self, x):
        self.calls.append(("head", x.clone(), None))
        logits = torch.cat([x[:, :1], x[:, :1] * 2], 1).float()
        return logits, (x.float() * 2).bfloat16()


class _Rt:
    kv_dtype = torch.bfloat16

    def __init__(self):
        self.steps = 0

    def advance_step(self):
        self.steps += 1


class _Processor:
    """SGLang's logits processor's call: records the hidden rows it is handed and
    returns its logits of them."""

    def __init__(self):
        self.hidden = None

    def _get_logits(self, hidden_states, lm_head, logits_metadata, **kw):
        raise AssertionError("a routed step returns the tail's logits")

    def __call__(self, input_ids, hidden, lm_head, forward_batch, aux):
        self.hidden = hidden
        return self._get_logits(hidden, lm_head, forward_batch)


def test_a_routed_step_runs_every_layer_and_the_tail(monkeypatch):
    # one decode row at width 2 on fake launches. SGLang's layer_boundary needs a
    # GPU to import, so its two calls are stand-ins here
    import contextlib

    import sglang.srt.model_executor.forward_context as forward_context

    from sgl_amd_mono.glm53.gfx950.config import HIDDEN

    calls, boundary = [], []
    batch = SimpleNamespace(
        start=lambda fb: boundary.append("start"),
        clear=lambda fb: boundary.append("clear"),
    )
    ctx = SimpleNamespace(maybe_input_scattered=lambda fb: contextlib.nullcontext())
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.layer_boundary",
        SimpleNamespace(get_attn_tp_context=lambda: ctx),
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.layer_boundary.residual.batch", batch
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(real_page_table=torch.zeros(1, 4))
    )
    monkeypatch.setattr(forward_context, "get_attn_backend", lambda: backend)
    monkeypatch.setattr(glm53._Built, "_bind", staticmethod(lambda L, kv_dtype: None))
    runtimes = {(w, k): _Rt() for w in (2, 4, 8) for k in KINDS}
    layers = {}
    for i in range(78):
        layers[i] = glm53._Layer(i, None, SimpleNamespace(index=None, dense=i < 3))
        layers[i].ops[2] = _Op(i, calls, runtimes[2, "shared"])
    lm = _Lm()
    lm.logits_processor, lm.lm_head = _Processor(), "lm_head"
    built = _built(lm, layers, runtimes, heads={2: _Head(calls)})
    key = registry.ShapeKey(forward_mode="decode", batch_size=1, tokens_per_request=1)
    fb = SimpleNamespace(out_cache_loc=torch.tensor([300]))
    out = built.forward(
        registry.Plan(key=key), torch.tensor([11]), torch.tensor([42]), fb
    )
    # every layer in order, then the tail; layer 0 from the step's token ids, the
    # width's row past the step a pad row (-1)
    assert [c[0] for c in calls] == list(range(78)) + ["head"]
    assert calls[0][1].tolist() == [11, -1]
    assert all(c[2] is calls[0][2] for c in calls[:78])
    assert calls[0][2].slots.tolist() == [300, -1]
    # the tail reads the last layer's rows: the token id plus one a layer
    assert calls[-1][1][0, 0].item() == 11 + 78
    # the logits processor returns the tail's logits, the step's row of the width's,
    # and stores the tail's normed rows as the step's hidden states
    assert torch.equal(out, torch.tensor([[89.0, 178.0]]))
    assert lm.logits_processor.hidden.shape == (1, HIDDEN)
    assert lm.logits_processor.hidden[0, 0].item() == 2 * 89
    assert built._logits is None and boundary == ["start", "clear"]
    # this width's runtimes moved to the next step, once
    assert all(rt.steps == (w == 2) for (w, _), rt in runtimes.items())
