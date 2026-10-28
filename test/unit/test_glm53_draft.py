"""The GLM-5.3 NextN (MTP) draft spec on CPU (RFC #43949): its registry entry, the
draft steps its wrapper routes to the launches and the ones it leaves to SGLang's
forward or refuses, the backend check, the settings that keep the whole draft on
SGLang's path, the logits a routed step returns, and the draft check's seed and
reused top-k metrics with their negative controls. The step tables
(``tables.step_rows``, ``tables.write_seed``) run under Triton's interpreter against
the torch ops they stand for."""

import contextlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import msgspec
import pytest
import sglang
import torch
import triton_interpreter
from sglang.srt.model_executor.forward_batch_info import ForwardMode

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.glm53 import check
from sgl_amd_mono.glm53 import draft as mtp
from sgl_amd_mono.glm53 import spec as glm53
from sgl_amd_mono.glm53.gfx950.config import HIDDEN

DRAFT = "GlmMoeDsaForCausalLMNextN"
TOPK = glm53.TOPK
# the server's MTP settings the draft reads: SGLang's GLM-5.3 default chain
SPEC_ARGS = SimpleNamespace(
    speculative_num_draft_tokens=4,
    speculative_use_rejection_sampling=False,
    speculative_token_map=None,
)


def test_the_plugin_registers_the_draft_spec():
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=[DRAFT], isa="gfx950"
    )
    assert found is mtp.SPEC, why
    assert mtp.SPEC.api_version == registry.MONOKERNEL_API_VERSION
    assert mtp.SPEC.requires.attention_backends == {"dsa"}
    assert registry.Feature.SPECULATIVE_DECODING in mtp.SPEC.features


class _Lm(torch.nn.Module):
    """GlmMoeDsaForCausalLMNextN's forward signature; records that it ran."""

    def __init__(self):
        super().__init__()
        self.ran = []
        self.logits_processor = SimpleNamespace(_get_logits=self._own_logits)

    def forward(self, input_ids, positions, forward_batch, pp_proxy_tensors=None):
        self.ran.append(forward_batch.forward_mode.name)
        return "the model's forward"

    @staticmethod
    def _own_logits(hidden_states, lm_head, logits_metadata):
        return "its own logits"


def _draft(lm, rejection_sampling=False, max_extend=8, max_decode=4):
    ops = {("index", w): None for w in (4, 8) if w <= max_extend}
    ops.update({("shared", w): None for w in (2, 4, 8) if w <= max_decode})
    args = SimpleNamespace(
        **{**vars(SPEC_ARGS), "speculative_use_rejection_sampling": rejection_sampling}
    )
    with mock.patch("sglang.srt.runtime_context.get_spec", return_value=args):
        return mtp._Draft(lm, None, ops, {}, {}, torch.zeros(1), torch.zeros(1), 9999)


def _step(mode, R, rows, *, hidden=True, seed=True, greedy=True, dtype=None):
    hs = torch.zeros(rows, HIDDEN, dtype=dtype or torch.bfloat16) if hidden else None
    return SimpleNamespace(
        forward_mode=ForwardMode[mode],
        batch_size=R,
        input_ids=torch.zeros(rows, dtype=torch.int64),
        spec_info=SimpleNamespace(
            hidden_states=hs,
            dsa_topk_indices=(
                torch.full((rows, TOPK), -1, dtype=torch.int32) if seed else None
            ),
        ),
        reuse_dsa_topk_indices=seed,
        sampling_info=SimpleNamespace(is_all_greedy=greedy),
    )


def _wrapped(**kw):
    lm = _Lm()
    draft = _draft(lm, **kw)
    launched = []
    draft.step = lambda ids, pos, fb: (
        launched.append((fb.forward_mode.name, int(fb.batch_size)))
        or ("launched", draft._kind(fb))
    )
    draft.check_backend = lambda fb: None
    mtp._route_draft_steps(lm, draft)
    return lm, draft, launched


TAKEN = [
    # a draft extend of R requests x 4 verify rows: the NextN indexer launch
    ("DRAFT_EXTEND_V2", 1, 4, ("index", 4)),
    ("DRAFT_EXTEND_V2", 2, 8, ("index", 8)),
    # a draft decode step of R rows on the reused top-k: the NextN shared launch at
    # the next width
    ("DECODE", 1, 1, ("shared", 2)),
    ("DECODE", 3, 3, ("shared", 4)),
]
NOT_TAKEN = {
    "three extend requests": _step("DRAFT_EXTEND_V2", 3, 12),
    "an extend of 3 rows a request": _step("DRAFT_EXTEND_V2", 2, 6),
    "five decode rows": _step("DECODE", 5, 5),
    "a verify step": _step("TARGET_VERIFY", 1, 4),
    "an extend": _step("EXTEND", 1, 4),
}
# draft-shaped steps the launches cannot read: the standard path eagerly, refused in
# a graph capture
REFUSED = {
    "no hidden states": _step("DECODE", 2, 2, hidden=False),
    "f32 hidden states": _step("DECODE", 2, 2, dtype=torch.float32),
    "no reused top-k": _step("DECODE", 2, 2, seed=False),
}


def test_the_draft_wrapper_keeps_the_forward_signature():
    lm, _, _ = _wrapped()
    # SGLang's graph runners choose what to pass from the forward's parameters
    assert inspect.signature(lm.forward) == inspect.signature(_Lm().forward)


@pytest.mark.parametrize("mode, R, rows, kind", TAKEN)
@pytest.mark.parametrize("as_kwarg", [False, True], ids=["positional", "kwarg"])
def test_the_draft_wrapper_launches_the_draft_steps(mode, R, rows, kind, as_kwarg):
    lm, _, launched = _wrapped()
    fb = _step(mode, R, rows)
    out = (
        lm.forward(input_ids=fb.input_ids, positions="p", forward_batch=fb)
        if as_kwarg
        else lm.forward(fb.input_ids, "p", fb)
    )
    assert out == ("launched", kind)
    assert launched == [(mode, R)] and lm.ran == []


@pytest.mark.parametrize("fb", NOT_TAKEN.values(), ids=NOT_TAKEN.keys())
def test_the_draft_wrapper_runs_the_forward_it_wraps(fb):
    lm, _, launched = _wrapped()
    assert lm.forward(fb.input_ids, "p", fb) == "the model's forward"
    assert lm.ran == [fb.forward_mode.name] and launched == []


def test_tp4_launches_up_to_four_rows():
    lm, _, launched = _wrapped(max_extend=4, max_decode=4)
    for mode, R, rows in (("DRAFT_EXTEND_V2", 2, 8), ("DECODE", 5, 5)):
        fb = _step(mode, R, rows)
        assert lm.forward(fb.input_ids, "p", fb) == "the model's forward"
    fb = _step("DECODE", 4, 4)
    assert lm.forward(fb.input_ids, "p", fb) == ("launched", ("shared", 4))
    assert launched == [("DECODE", 4)]


def test_a_sampled_draft_decode_runs_the_standard_path():
    # under rejection sampling a batch that is not all greedy samples its draft
    # decode proposal; the draft extend's proposal is SGLang's on the tail's logits
    lm, _, launched = _wrapped(rejection_sampling=True)
    for mode, R, rows in (("DECODE", 2, 2), ("DRAFT_EXTEND_V2", 2, 8)):
        fb = _step(mode, R, rows, greedy=False)
        lm.forward(fb.input_ids, "p", fb)
    fb = _step("DECODE", 2, 2)
    lm.forward(fb.input_ids, "p", fb)
    assert lm.ran == ["DECODE"]
    assert launched == [("DRAFT_EXTEND_V2", 2), ("DECODE", 2)]


@pytest.mark.parametrize("fb", REFUSED.values(), ids=REFUSED.keys())
def test_a_draft_step_the_launches_cannot_read_is_refused_in_capture(fb, caplog):
    lm, _, launched = _wrapped()
    with mock.patch.object(mtp, "_is_capturing", return_value=True):
        with pytest.raises(ValueError, match="refusing to capture the standard path"):
            lm.forward(fb.input_ids, "p", fb)
    # eagerly it runs the standard path, logged once a key
    with caplog.at_level("WARNING"):
        for _ in range(2):
            assert lm.forward(fb.input_ids, "p", fb) == "the model's forward"
    assert caplog.text.count("this step runs the standard path") == 1
    assert launched == []


def test_the_runtimes_forward_wraps_the_draft_wrapper():
    # build installs the draft wrapper, then SGLang wraps it: API v1 routes the
    # draft by the target's verify key, which never matches a draft step
    try:
        from sglang.srt.kernel_plugins.amd import binding as runtime
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.monokernels import runtime

    lm, draft, launched = _wrapped()
    served = frozenset(
        registry.ShapeKey("target_verify", batch_size=r, tokens_per_request=4)
        for r in (1, 2)
    )
    runtime._install_forward(model=lm, built=draft, served=served, spec=mtp.SPEC)
    for mode, R, rows, _ in TAKEN:
        fb = _step(mode, R, rows)
        lm.forward(fb.input_ids, "p", fb)
    assert launched == [(m, R) for m, R, _, _ in TAKEN] and lm.ran == []


def test_the_draft_checks_its_attention_backend_once_a_step_mode():
    lm = _Lm()
    draft = _draft(lm)
    backends = ["dsa", "triton", "triton"]

    def backend():
        return SimpleNamespace(decode_attention_backend_str=backends.pop(0))

    with mock.patch(
        "sglang.srt.model_executor.forward_context.get_attn_backend", backend
    ):
        draft.check_backend(_step("DECODE", 1, 1))
        draft.check_backend(_step("DECODE", 2, 2))
        with pytest.raises(ValueError, match="this DRAFT_EXTEND_V2 step's is 'triton'"):
            draft.check_backend(_step("DRAFT_EXTEND_V2", 1, 4))
    assert backends == ["triton"]


def test_every_draft_call_site_calls_the_model_forward_attribute():
    # the wrapper replaces the instance attribute model.forward: pin that SGLang's
    # draft paths call it, so a refactor fails here, not as a silent fallback
    root = Path(sglang.__file__).resolve().parent / "srt"
    calls = {
        "model_executor/runner/eager_runner.py": "model_runner.model.forward(",
        "speculative/eagle_draft_extend_cuda_graph_runner.py": (
            "self.model_runner.model.forward("
        ),
        "speculative/eagle_worker_v2.py": "self.draft_runner.forward(",
        "speculative/eagle_draft_cuda_graph_runner.py": (
            "self.eagle_worker.draft_forward("
        ),
    }
    for path, call in calls.items():
        assert call in (root / path).read_text(), (path, call)


def test_a_routed_step_returns_the_tails_logits():
    lm = _Lm()
    draft = _draft(lm)
    lp = lm.logits_processor
    rows = torch.zeros(4, HIDDEN)
    meta = SimpleNamespace(draft_extend_select_index=None)
    assert lp._get_logits(rows, None, meta) == "its own logits"
    draft._logits = torch.arange(8.0).view(8, 1)
    assert lp._get_logits(rows, None, meta).flatten().tolist() == [0, 1, 2, 3]
    # the draft extend graph's selected rows
    meta = SimpleNamespace(draft_extend_select_index=torch.tensor([3, 7]))
    assert lp._get_logits(rows[:2], None, meta).flatten().tolist() == [3, 7]


class _ReplicatedLinear(torch.nn.Module):
    """A quantized eh_proj, as SGLang builds it for a Quark checkpoint."""


def _nextn_model(fused=True, eh=None):
    return SimpleNamespace(
        model=SimpleNamespace(
            decoder=SimpleNamespace(
                self_attn=SimpleNamespace(
                    indexer=SimpleNamespace(use_gfx950_fused_indexer=fused)
                )
            ),
            eh_proj=torch.nn.Linear(2, 1, bias=False) if eh is None else eh,
        )
    )


GLM53 = SimpleNamespace(index_topk=TOPK, index_share_for_mtp_iteration=True)


@pytest.mark.parametrize(
    "spec_args, hf_config, model, tp, why",
    [
        (SPEC_ARGS, GLM53, _nextn_model(), 8, None),
        (
            SimpleNamespace(**{**vars(SPEC_ARGS), "speculative_token_map": "map.pt"}),
            GLM53,
            _nextn_model(),
            8,
            "token map",
        ),
        (SPEC_ARGS, SimpleNamespace(index_topk=TOPK), _nextn_model(), 8, "index share"),
        (
            SPEC_ARGS,
            SimpleNamespace(**{**vars(GLM53), "index_kpool": 2}),
            _nextn_model(),
            8,
            "a seed of 2049 keys",
        ),
        (SPEC_ARGS, GLM53, _nextn_model(fused=False), 8, "fused DSA indexer"),
        (SPEC_ARGS, GLM53, _nextn_model(), 2, "TP2"),
        (SPEC_ARGS, GLM53, _nextn_model(eh=_ReplicatedLinear()), 8, "eh_proj is a"),
    ],
    ids=["served", "token map", "no index share", "seed width", "indexer", "tp", "eh"],
)
def test_each_setting_the_launches_cannot_serve_keeps_the_draft_standard(
    spec_args, hf_config, model, tp, why
):
    # SGLang's model_config imports AITER, which needs a GPU: its seed width
    model_config = SimpleNamespace(
        get_dsa_mtp_topk_width=lambda c: c.index_topk + getattr(c, "index_kpool", 1) - 1
    )
    with (
        mock.patch.dict(sys.modules, {"sglang.srt.configs.model_config": model_config}),
        mock.patch("sglang.srt.runtime_context.get_spec", return_value=spec_args),
        mock.patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(tp_size=tp),
        ),
    ):
        reasons = mtp._standard_reasons(model, hf_config)
    if why is None:
        assert reasons == []
    else:
        assert len(reasons) == 1 and why in reasons[0], reasons


class _Weights(msgspec.Struct, frozen=True):
    """A layer's weights as the build reads them."""

    shuffled: bool
    index: object = None


class _Runtime:
    def __init__(self, key, rank, group, device, epochs):
        self.key, self.rows = key, key.rows
        epochs.take()
        self.precompiled = False

    def set_deadline(self, s):
        self.deadline = s

    def precompile(self):
        self.precompiled = True


@pytest.mark.parametrize("tp, extend, decode", [(8, [4, 8], [2, 4]), (4, [4], [2, 4])])
def test_the_build_runs_the_nextn_launches_and_installs_the_wrapper(
    tp, extend, decode, caplog
):
    from sgl_amd_mono.glm53.gfx950 import runner, weights

    built = []
    decoder = SimpleNamespace(
        self_attn=SimpleNamespace(
            attn_mqa=SimpleNamespace(layer_id=78),
            indexer=SimpleNamespace(use_gfx950_fused_indexer=True),
        ),
        input_layernorm=SimpleNamespace(weight=torch.zeros(1)),
    )
    lm = _Lm()
    lm.model = SimpleNamespace(
        decoder=decoder,
        eh_proj=torch.nn.Linear(
            2 * HIDDEN, HIDDEN, bias=False, device="meta", dtype=torch.bfloat16
        ),
        enorm=SimpleNamespace(weight=torch.zeros(HIDDEN)),
        hnorm=SimpleNamespace(weight=torch.zeros(HIDDEN)),
        shared_head=SimpleNamespace(norm=None),
    )
    vocab_rows = 154880 // tp
    # the guard and SGLang's model_config import GPU libraries: their stand-ins, the
    # guard's longest deadline 5497 s
    patches = [
        mock.patch.dict(
            sys.modules,
            {
                "sglang.srt.configs.model_config": SimpleNamespace(
                    get_dsa_mtp_topk_width=lambda c: TOPK
                ),
                "sgl_amd_mono.common.guard": SimpleNamespace(
                    MAX_DEADLINE_TICKS=5497 * 10**8, CLOCK_HZ=10**8, GUARD_WORDS=8
                ),
            },
        ),
        mock.patch("sglang.srt.runtime_context.get_spec", return_value=SPEC_ARGS),
        mock.patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(
                tp_size=tp, tp_rank=0, tp_group=SimpleNamespace(cpu_group=None)
            ),
        ),
        mock.patch(
            "sglang.srt.runtime_context.get_model",
            return_value=SimpleNamespace(kv_cache_dtype="fp8_e4m3"),
        ),
        mock.patch(
            "sglang.srt.runtime_context.process_model_config",
            return_value=SimpleNamespace(context_len=202752),
        ),
        mock.patch(
            "sglang.srt.runtime_context.get_device",
            return_value=SimpleNamespace(watchdog_timeout=1200),
        ),
        mock.patch.object(weights, "layer_weights", lambda *a: _Weights(True, "ix")),
        mock.patch.object(
            weights, "rope_tables", lambda *a: (torch.zeros(202752, 1),) * 2
        ),
        mock.patch.object(weights, "embedding", lambda *a: torch.zeros(vocab_rows, 1)),
        mock.patch.object(
            weights,
            "head_weights",
            lambda *a: SimpleNamespace(w_head=torch.zeros(vocab_rows, 1)),
        ),
        mock.patch.object(runner, "Glm53Runtime", _Runtime),
        mock.patch.object(
            runner,
            "Glm53LayerOp",
            lambda rt, w, n, eh: built.append(("op", rt, w.index, eh)) or rt,
        ),
        mock.patch.object(
            runner, "Glm53HeadOp", lambda rt, w: built.append(("head", rt)) or rt
        ),
    ]
    ctx = SimpleNamespace(widths=(4, 8), deadline_s=60.0, copies={})
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        with caplog.at_level("INFO"):
            draft = mtp._build(
                lm, SimpleNamespace(index_share_for_mtp_iteration=True), ctx
            )
    ops = [(b[1].key, b[2], b[3]) for b in built if b[0] == "op"]
    # the draft extend's NextN indexer launches, then the draft decode's shared ones,
    # every one from the tokens through the entry, a draft step's first launch
    assert [(k.rows, k.with_indexer) for k, _, _ in ops] == [
        (w, True) for w in extend
    ] + [(w, False) for w in decode]
    assert all(k.eh and k.tokens and k.vocab_rows == vocab_rows for k, _, _ in ops)
    assert all(eh.w_eh.shape == (HIDDEN // tp, 2 * HIDDEN) for _, _, eh in ops)
    assert [index for _, index, _ in ops] == ["ix"] * len(extend) + [None] * len(decode)
    heads = [b[1] for b in built if b[0] == "head"]
    assert [rt.key.rows for rt in heads] == sorted(set(extend) | set(decode))
    assert all(rt.key.head and not rt.key.eh for rt in heads)
    # half the watchdog for the NextN launch, the framework's deadline for the tail
    nextn = [b[1] for b in built if b[0] == "op"]
    assert {rt.deadline for rt in nextn} == {600.0} and {
        rt.deadline for rt in heads
    } == {60.0}
    assert all(rt.precompiled for rt in nextn + heads)
    assert "forward" in lm.__dict__ and isinstance(draft, mtp._Draft)
    assert "run the NextN launches, routed by the spec's wrapper" in caplog.text


# ------------------------------------------------------------- draft check metrics
class _RMSNorm(torch.nn.Module):
    """SGLang's RMSNorm on bf16 rows: f32 sums, bf16 out."""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return check._rms(x, self.weight).bfloat16()


def test_the_draft_check_bounds_the_entry_and_catches_swapped_norms():
    g = torch.Generator().manual_seed(3)
    D = 256
    norm_w = lambda: (1 + 0.3 * torch.randn(D, generator=g)).bfloat16()  # noqa: E731
    m = SimpleNamespace(
        enorm=_RMSNorm(norm_w()),
        hnorm=_RMSNorm(norm_w()),
        eh_proj=torch.nn.Linear(2 * D, D, bias=False).bfloat16(),
    )
    e = torch.randn(4, D, generator=g).bfloat16()
    h = (3 * torch.randn(4, D, generator=g)).bfloat16()
    bounds = glm53.NUMERICS.bounds
    # an entry as the launch computes it: f32 sums of bf16 normed rows, bf16 out
    cat = torch.cat((m.enorm(e), m.hnorm(h)), -1).float()
    x = (cat @ m.eh_proj.weight.float().t()).bfloat16()
    values, _ = check.entry_values(m, e, h, x)
    assert values["draft.eh.rel"] <= bounds["draft.eh.rel"]
    assert values["draft.eh_vs_fp32.rel"] <= bounds["draft.eh_vs_fp32.rel"]
    # negative control: the enorm and hnorm weights swapped
    cat = torch.cat((_RMSNorm(m.hnorm.weight)(e), _RMSNorm(m.enorm.weight)(h)), -1)
    bad = (cat.float() @ m.eh_proj.weight.float().t()).bfloat16()
    values, _ = check.entry_values(m, e, h, bad)
    assert values["draft.eh.rel"] > bounds["draft.eh.rel"]
    assert values["draft.eh_vs_fp32.rel"] > bounds["draft.eh_vs_fp32.rel"]


def _seed_step():
    # row 0 keeps 3 keys, row 1 two, row 2 (a pad row) none
    topk = torch.zeros(3 * TOPK + 1, dtype=torch.int32)
    topk[:5] = torch.tensor([10, 11, 12, 20, 21], dtype=torch.int32)
    return SimpleNamespace(
        indptr=torch.tensor([0, 3, 5, 5], dtype=torch.int32), topk=topk
    )


def test_the_draft_check_compares_the_seed_with_the_launchs_top_k():
    step = _seed_step()
    seed = torch.full((3, TOPK), -1, dtype=torch.int32)
    seed[0, :3] = torch.tensor([10, 11, 12], dtype=torch.int32)
    seed[1, :2] = torch.tensor([20, 21], dtype=torch.int32)
    assert check.seed_mismatch(step, seed, 3) == 0
    # negative control: every seed row row 0's
    bad = seed.clone()
    bad[1] = seed[0]
    assert check.seed_mismatch(step, bad, 3) == 3
    # an entry past a row's keys left set
    bad = seed.clone()
    bad[1, 2] = 7
    assert check.seed_mismatch(step, bad, 3) == 1


def test_the_draft_check_counts_a_decode_rows_keys_against_the_seed():
    step = _seed_step()
    seed = torch.full((3, TOPK), -1, dtype=torch.int32)
    seed[0, [0, 2, 5]] = torch.tensor(
        [10, 11, 12], dtype=torch.int32
    )  # -1 entries inside
    seed[1, :2] = torch.tensor([20, 21], dtype=torch.int32)
    seed[2, :4] = 1  # a pad row: slot 0
    loc = torch.tensor([100, 101, 0])
    assert check.topk_extra(step, seed, loc, 3) == 0
    # negative control: the row's count from its position, past its valid entries
    step.indptr = torch.tensor([0, 6, 8, 8], dtype=torch.int32)
    assert check.topk_extra(step, seed, loc, 3) == 3


# ------------------------------------------------------------------ step tables
def _torch_rows(st, loc, pos, S, max_position, ids_in, ids, seed=None):
    """The torch ops the step tables stand for: the target's, and SGLang's
    compaction of a reused top-k's valid entries."""
    W = st.slots.shape[0]
    ids[:S] = ids_in[:S]
    ids[S:] = -1
    loc = loc[:S].to(torch.int64)
    valid = loc != 0
    st.slots[:S] = torch.where(valid, loc, -1)
    st.slots[S:] = -1
    p = pos[:S].to(torch.int64).clamp(max=max_position)
    st.positions[:S] = p
    st.positions[S:] = 0
    if seed is None:
        count = torch.where(valid, (p + 1).clamp(max=TOPK), 0)
    else:
        keep = (seed[:S] >= 0) & valid[:, None]
        count = keep.sum(-1)
    st.indptr[1 : S + 1] = torch.cumsum(count, 0, dtype=torch.int32)
    st.indptr[S + 1 :] = st.indptr[S]
    st.req_ids[:S] = torch.arange(S, dtype=torch.int32)
    st.req_ids[S:] = 0
    if seed is not None:
        start = st.indptr[:S].to(torch.int64)[:, None]
        dst = torch.where(keep, start + keep.cumsum(-1) - 1, W * TOPK)
        st.topk.scatter_(0, dst.reshape(-1), seed[:S].reshape(-1))


def _interpreted_tables(W, S, with_seed):
    """The step tables against the torch ops (run under Triton's interpreter): which
    words differ, and whether the seed sized by position (a negative control) moves
    the rows' keys."""
    from sgl_amd_mono.glm53 import tables

    g = torch.Generator().manual_seed(W * 10 + S)
    # a strided view, as SGLang's per-step draft slots
    loc = torch.randint(1, 1 << 30, (3 * S,), generator=g)[::3]
    if S > 1:
        loc[S - 1] = 0  # SGLang's padding slot
    pos = torch.randint(0, 12000, (S,), generator=g)
    pos[0] = 100
    ids_in = torch.randint(0, 150000, (S,), generator=g)
    seed = None
    if with_seed:
        seed = torch.randint(0, 1 << 30, (S, TOPK), generator=g, dtype=torch.int32)
        for s in range(S):
            seed[s, int(torch.randint(1, TOPK + 1, (1,), generator=g)) :] = -1
        seed[0, 3] = -1
    a, b = mtp._new_step(W, "cpu"), mtp._new_step(W, "cpu")
    ida = torch.zeros(W, dtype=torch.int32)
    idb = torch.zeros(W, dtype=torch.int32)
    _torch_rows(a, loc, pos, S, 9000, ids_in, ida, seed)
    tables.step_rows(b, loc, pos, S, 9000, ids_in=ids_in, ids=idb, seed=seed)
    n = int(a.indptr[W])
    differ = [
        name
        for name in ("slots", "positions", "indptr", "req_ids")
        if not torch.equal(getattr(a, name), getattr(b, name))
    ]
    differ += [] if torch.equal(ida, idb) else ["ids"]
    differ += [] if torch.equal(a.topk[:n], b.topk[:n]) else ["topk"]
    control = None
    if with_seed:
        c = mtp._new_step(W, "cpu")
        tables.step_rows(c, loc, pos, S, 9000, seed=seed, seed_by_position=True)
        control = not torch.equal(a.indptr, c.indptr)
    # the seed SGLang gets: each row's keys, then -1
    out = torch.full((S, TOPK), 7, dtype=torch.int32)
    tables.write_seed(a, out, S, torch.arange(S, dtype=torch.int32))
    return differ, control, check.seed_mismatch(a, out, S)


@pytest.mark.parametrize("W, S", [(2, 1), (4, 3), (4, 4), (8, 5)])
@pytest.mark.parametrize("with_seed", [False, True], ids=["positions", "seed"])
def test_the_step_tables_equal_the_torch_ops(W, S, with_seed):
    differ, control, seed_bad = triton_interpreter.run(
        __file__, "_interpreted_tables", W, S, with_seed
    )
    assert differ == []
    # negative control: the reused top-k sized by position moves the rows' keys
    assert control is (True if with_seed else None)
    assert seed_bad == 0
