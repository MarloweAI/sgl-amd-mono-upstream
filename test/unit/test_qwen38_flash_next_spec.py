"""The Qwen3.8-Flash-Next gfx950 monokernel spec on CPU (RFC #43949): its registry
entries (the target and the MTP draft), the configurations it refuses before
load, the layer views it refuses after load (meta tensors of the real shapes),
the verify widths it serves on the NEXTN recipe, the draft steps the draft's
wrapper routes to its launch (and the guards that keep every draft step of up to 8
requests on it), its build keys and its fallback forward. SGLang checks the
declared requirements, the same way for every spec."""

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import sglang
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry
try:
    from sglang.srt.kernel_plugins.amd.binding import covered_capture_sizes
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.guards import covered_capture_sizes

import sgl_amd_mono
from sgl_amd_mono.common.plan import key_tuple
from sgl_amd_mono.qwen38_flash_next import (
    HF_CONFIG,
    INDEXER_CONFIG,
    LAYER_PATTERN,
    MAX_CTX,
)
from sgl_amd_mono.qwen38_flash_next import draft as mtp
from sgl_amd_mono.qwen38_flash_next import spec as q38
from sgl_amd_mono.qwen38_flash_next.gfx950 import layout as LY
from sgl_amd_mono.qwen38_flash_next.gfx950 import weights
from sgl_amd_mono.qwen38_flash_next.gfx950.config import ConvStrides, MonoBuild
from sgl_amd_mono.qwen38_flash_next.gfx950.runner import (
    HC_DIM,
    VerifyRunner,
    _overlaps,
    launch_plans,
)

TARGET, DRAFT = "Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLMMTP"
# amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8 config.json, text_config
TEXT = SimpleNamespace(
    **HF_CONFIG,
    **INDEXER_CONFIG,
    layer_types=list(LAYER_PATTERN) * 12,
    mamba_ssm_dtype="float32",
    max_position_embeddings=262144,
    eos_token_id=248044,
)
VL = SimpleNamespace(
    text_config=TEXT,
    architectures=[TARGET],
    quantization_config={"quant_method": "quark"},
)
# the NEXTN recipe's server arguments the spec reads
CFG = SimpleNamespace(context_length=None)


def _replace(ns, **changes):
    return SimpleNamespace(**{**vars(ns), **changes})


def _config_reasons(cfg=CFG, hf_config=VL, flydsl=True):
    with mock.patch.object(q38, "_has_flydsl", return_value=flydsl):
        return q38._config_disable_reasons(cfg, hf_config)


def test_the_plugin_registers_the_target_and_the_draft():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=[TARGET], isa="gfx950"
    )
    assert found is q38.SPEC, why
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=[TARGET], isa="gfx942"
    )
    assert found is None and "gfx942" in why
    # the NEXTN draft's own steps reach its launch through the wrapper _build installs
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=[DRAFT], isa="gfx950"
    )
    assert found is mtp.SPEC, why


def test_the_draft_spec_declares_the_drafts_own_backend():
    assert mtp.SPEC.api_version == registry.MONOKERNEL_API_VERSION
    # the draft runs on its QSA backends, stamped "qsa", never the target's
    assert mtp.SPEC.requires.attention_backends == {"qsa"}
    assert (
        not mtp.SPEC.requires.attention_backends & q38.SPEC.requires.attention_backends
    )
    assert mtp._tokens_per_request(None) == q38.TOKENS
    assert registry.Feature.SPECULATIVE_DECODING in mtp.SPEC.features


# ---------------------------------------------------------------- draft steps
SHARES_MTP_SPARSE_INDICES = mtp._shares_mtp_sparse_indices


@pytest.fixture(autouse=True)
def _qsa_steps(monkeypatch):
    """Draft steps run on the draft's QSA backend with index sharing off, as on a
    server."""
    monkeypatch.setattr(mtp, "_step_backend", lambda: "qsa")
    monkeypatch.setattr(mtp, "_shares_mtp_sparse_indices", lambda: False)


class _DraftRunner:
    """A DraftRunner whose launch records the steps it took."""

    L_ext, max_reqs = q38.TOKENS, q38.MAX_R

    def __init__(self):
        self.launched = []

    def launch(self, fb, hidden_states):
        self.launched.append((fb.forward_mode.name, int(fb.batch_size)))
        return "x", "hc"


class _MtpLm(torch.nn.Module):
    """Qwen4ExpForCausalLMMTP's forward signature; records that it ran."""

    def __init__(self):
        super().__init__()
        self.ran = []
        self.lm_head = "lm_head"

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        self.ran.append(forward_batch.forward_mode.name)
        return "the model's forward"

    def logits_processor(self, input_ids, x, lm_head, fb):
        return SimpleNamespace(x=x)

    def _set_hc_logits_hidden_states(self, out, hc, fb):
        out.hidden_states = hc


def _step(mode, R, rows, dtype=torch.bfloat16, hc_rows=None):
    hidden = torch.zeros(rows if hc_rows is None else hc_rows, HC_DIM, dtype=dtype)
    return SimpleNamespace(
        forward_mode=ForwardMode[mode],
        batch_size=R,
        input_ids=torch.zeros(rows, dtype=torch.int64),
        spec_info=SimpleNamespace(hidden_states=hidden),
    )


TAKEN = [
    ("DRAFT_EXTEND_V2", 1, 4),
    ("DRAFT_EXTEND_V2", 8, 32),
    ("DECODE", 1, 1),
    ("DECODE", 8, 8),
]
NOT_TAKEN = {
    "nine requests": _step("DECODE", 9, 9),
    "nine requests extend": _step("DRAFT_EXTEND_V2", 9, 36),
    "verify": _step("TARGET_VERIFY", 2, 8),
    "extend": _step("EXTEND", 2, 8),
}
# draft steps of up to 8 requests the launch cannot read: refused, never SGLang's
REFUSED = {
    "three rows a request": _step("DRAFT_EXTEND_V2", 2, 6),
    "fp32 hidden": _step("DECODE", 2, 2, dtype=torch.float32),
    "hidden of other rows": _step("DECODE", 2, 2, hc_rows=4),
    "strided hidden": SimpleNamespace(
        **{
            **vars(_step("DECODE", 2, 2)),
            "spec_info": SimpleNamespace(
                hidden_states=torch.zeros(HC_DIM, 2, dtype=torch.bfloat16).t()
            ),
        }
    ),
    "no hidden states": SimpleNamespace(
        **{**vars(_step("DRAFT_EXTEND_V2", 8, 32)), "spec_info": None}
    ),
}


def _wrapped():
    lm, runner = _MtpLm(), _DraftRunner()
    mtp._route_draft_steps(lm, mtp._Draft(lm, runner))
    return lm, runner


def test_the_draft_wrapper_keeps_the_forward_signature():
    lm, _ = _wrapped()
    # SGLang's graph runners choose what to pass from the forward's parameters
    assert inspect.signature(lm.forward) == inspect.signature(_MtpLm().forward)


@pytest.mark.parametrize("mode, R, rows", TAKEN)
@pytest.mark.parametrize("as_kwarg", [False, True], ids=["positional", "kwarg"])
def test_the_draft_wrapper_launches_the_draft_steps(mode, R, rows, as_kwarg):
    lm, runner = _wrapped()
    fb = _step(mode, R, rows)
    out = (
        lm.forward(fb.input_ids, "positions", forward_batch=fb)
        if as_kwarg
        else lm.forward(fb.input_ids, "positions", fb)
    )
    assert (out.x, out.hidden_states) == ("x", "hc")
    assert runner.launched == [(mode, R)] and lm.ran == []


@pytest.mark.parametrize("fb", NOT_TAKEN.values(), ids=NOT_TAKEN.keys())
@pytest.mark.parametrize("as_kwarg", [False, True], ids=["positional", "kwarg"])
def test_the_draft_wrapper_runs_the_forward_it_wraps(fb, as_kwarg):
    lm, runner = _wrapped()
    out = (
        lm.forward(input_ids=fb.input_ids, positions="p", forward_batch=fb)
        if as_kwarg
        else lm.forward(fb.input_ids, "p", fb)
    )
    assert out == "the model's forward"
    assert lm.ran == [fb.forward_mode.name] and runner.launched == []


def test_a_draft_step_with_input_embeds_is_refused():
    lm, runner = _wrapped()
    fb = _step("DECODE", 2, 2)
    with pytest.raises(ValueError, match="has input embeddings"):
        lm.forward(fb.input_ids, "p", fb, torch.zeros(2))
    with pytest.raises(ValueError, match="has input embeddings"):
        lm.forward(fb.input_ids, "p", fb, input_embeds=torch.zeros(2))
    # above 8 requests the wrapped forward handles them
    fb = NOT_TAKEN["nine requests"]
    assert lm.forward(fb.input_ids, "p", fb, input_embeds=torch.zeros(9)) == (
        "the model's forward"
    )
    assert runner.launched == [] and lm.ran == ["DECODE"]


def test_the_runtimes_forward_wraps_the_draft_wrapper():
    # build installs the draft wrapper, then SGLang wraps it: API v1 routes the
    # draft by the target's verify key, which never matches a draft step, so every
    # draft step reaches the draft wrapper
    try:
        from sglang.srt.kernel_plugins.amd import binding as runtime
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.monokernels import runtime

    lm, runner = _wrapped()
    served = frozenset(
        registry.ShapeKey("target_verify", batch_size=r, tokens_per_request=4)
        for r in range(1, 9)
    )
    built = mtp._Draft(lm, runner)
    kw = dict(model=lm, built=built, served=served, spec=mtp.SPEC)
    # SGLang before sgl-project/sglang#45389 also takes the backend name
    if "backend" in inspect.signature(runtime._install_forward).parameters:
        kw["backend"] = sgl_amd_mono.BACKEND
    runtime._install_forward(**kw)
    for mode, R, rows in TAKEN:
        fb = _step(mode, R, rows)
        lm.forward(fb.input_ids, "p", fb)
    assert runner.launched == [(m, R) for m, R, _ in TAKEN] and lm.ran == []


def test_the_wrapper_checks_the_backend_once_a_step_mode(monkeypatch):
    # what the runtime does for the steps it routes
    backends = ["qsa", "triton", "triton"]
    monkeypatch.setattr(mtp, "_step_backend", lambda: backends.pop(0))
    lm, runner = _wrapped()
    for mode, R, rows in (("DECODE", 1, 1), ("DECODE", 2, 2)):
        fb = _step(mode, R, rows)
        lm.forward(fb.input_ids, "p", fb)
    fb = _step("DRAFT_EXTEND_V2", 1, 4)
    with pytest.raises(ValueError, match="this DRAFT_EXTEND_V2 step's is 'triton'"):
        lm.forward(fb.input_ids, "p", fb)
    assert runner.launched == [("DECODE", 1), ("DECODE", 2)]


@pytest.mark.parametrize("fb", REFUSED.values(), ids=REFUSED.keys())
def test_a_draft_step_the_launch_cannot_read_is_refused(fb):
    # eager as in capture: SGLang's draft forward would run a draft step the
    # launch serves, and with a launched extend before it, read a stale selection
    lm, runner = _wrapped()
    with pytest.raises(ValueError, match="refusing to run SGLang's draft forward"):
        lm.forward(fb.input_ids, "p", fb)
    # the runtime-routed forward refuses it too
    with pytest.raises(ValueError, match="refusing to run SGLang's draft forward"):
        mtp._Draft(lm, runner).forward(None, fb.input_ids, "p", fb)
    assert runner.launched == [] and lm.ran == []


class _QsaBackend:
    """SGLang's QSA backend: the MTP sparse-selection hooks and their state."""

    def __init__(self, shared):
        self._mtp_shared_sparse_indices = shared

    def set_mtp_shared_sparse_indices(self, state):
        self._mtp_shared_sparse_indices = state


@pytest.mark.parametrize("hybrid", [False, True], ids=["qsa", "hybrid"])
def test_a_draft_step_on_shared_sparse_indices_is_refused(monkeypatch, hybrid):
    from sglang.srt.model_executor import forward_context

    monkeypatch.setattr(mtp, "_shares_mtp_sparse_indices", SHARES_MTP_SPARSE_INDICES)
    backend = _QsaBackend(shared=None)
    step_backend = SimpleNamespace(full_attn_backend=backend) if hybrid else backend
    monkeypatch.setattr(forward_context, "get_attn_backend", lambda: step_backend)
    lm, runner = _wrapped()
    fb = _step("DECODE", 2, 2)
    lm.forward(fb.input_ids, "p", fb)
    assert runner.launched == [("DECODE", 2)]
    # index sharing on: SGLang's draft decode would reuse a selection the launched
    # draft extend never writes
    backend._mtp_shared_sparse_indices = object()
    fb = _step("DRAFT_EXTEND_V2", 2, 8)
    with pytest.raises(ValueError, match="shares one sparse selection"):
        lm.forward(fb.input_ids, "p", fb)
    assert runner.launched == [("DECODE", 2)]


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


def test_the_draft_writes_the_buffer_its_input_does_not_overlap():
    a, b = torch.zeros(8, 4), torch.zeros(8, 4)
    assert _overlaps(a, a) and _overlaps(a, a[2:3]) and _overlaps(a[2:], a[:3])
    assert not _overlaps(a, b) and not _overlaps(a[:2], a[2:])


def test_the_draft_reads_new_hidden_states_from_a_fixed_address(monkeypatch):
    """A chain step's hidden states under rejection sampling are a new tensor each
    step (a capture allocates them elsewhere than its warmup): the launch reads a
    copy at a fixed address; a previous step's output is read in place."""
    from sgl_amd_mono.qwen38_flash_next.gfx950 import runner as R

    S, D = 4, R.HC_DIM
    r = object.__new__(R.DraftRunner)
    r.hc_bufs = [torch.zeros(S, D, dtype=torch.bfloat16) for _ in range(2)]
    r.hin_buf = torch.zeros(S, D, dtype=torch.bfloat16)
    r.ids_buf = torch.zeros(S, dtype=torch.int64)
    r.x_buf = torch.zeros(S, 8, dtype=torch.bfloat16)
    r.scratch = r.epoch = torch.zeros(4, dtype=torch.int32)
    read = []
    r.prepare_step = lambda fb, hs: (read.append(hs.data_ptr()), (r.scratch, None))[1]
    r.kernel = lambda build: lambda *a, **kw: None
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    fb = SimpleNamespace(input_ids=torch.arange(2))
    for _ in range(2):  # a new tensor each step: one table address
        new = torch.ones(2, D, dtype=torch.bfloat16)
        _, hc = r.launch(fb, new)
        assert read[-1] == r.hin_buf.data_ptr()
        assert torch.equal(r.hin_buf[:2], new)
        assert hc.data_ptr() == r.hc_bufs[0].data_ptr()
    _, hc2 = r.launch(fb, hc)  # the previous step's output: in place
    assert read[-1] == hc.data_ptr() and hc2.data_ptr() == r.hc_bufs[1].data_ptr()


def test_build_installs_the_wrapper(monkeypatch, caplog):
    from sgl_amd_mono.qwen38_flash_next.gfx950 import runner as R

    class _Runner(_DraftRunner):
        def __init__(self, **kw):
            super().__init__()
            # fault injection arms the target's launches only
            assert kw["fault"] is None

        def set_deadline(self, s):
            pass

    monkeypatch.setattr(weights, "param_ptrs", lambda model: {})
    monkeypatch.setattr(weights, "mtp_views", lambda model, copies, params: {})
    monkeypatch.setattr(R, "DraftRunner", _Runner)
    layer = SimpleNamespace(rotary_emb=None, attn=SimpleNamespace(layer_id=48))
    lm = _MtpLm()
    lm.model = SimpleNamespace(
        layers=[layer], embed_tokens=SimpleNamespace(weight=torch.zeros(1))
    )
    # the checkpoint's config turns SGLang's QSA MTP index sharing on
    hf_config = _replace(VL, text_config=_replace(TEXT))
    hf_config.text_config.index_share_for_mtp_iteration = True
    with caplog.at_level("INFO"):
        draft = mtp._build(lm, hf_config, SimpleNamespace(copies={}, deadline_s=1.0))
    assert "forward" in lm.__dict__
    assert "are routed by the spec's wrapper of the draft's forward" in caplog.text
    fb = _step("DECODE", 1, 1)
    lm.forward(fb.input_ids, "p", fb)
    assert draft.runner.launched == [("DECODE", 1)] and lm.ran == []
    # the draft worker reads the flag after the build: sharing off, on both levels
    assert hf_config.text_config.index_share_for_mtp_iteration is False
    assert hf_config.index_share_for_mtp_iteration is False
    assert "QSA MTP index sharing off (index_share_for_mtp_iteration True" in (
        caplog.text
    )


def test_the_draft_worker_reads_index_sharing_off_after_the_build():
    from sglang.srt.speculative.eagle_worker_v2 import _qsa_index_share_requested

    hf_config = _replace(VL, text_config=_replace(TEXT))
    hf_config.text_config.index_share_for_mtp_iteration = True
    assert _qsa_index_share_requested(hf_config)
    mtp._turn_off_mtp_index_share(hf_config)
    assert not _qsa_index_share_requested(hf_config)


def test_the_recipe_is_accepted():
    assert _config_reasons() == []
    assert _config_reasons(cfg=_replace(CFG, context_length=131072)) == []


@pytest.mark.parametrize(
    "cfg, hf_config, flydsl",
    [
        (CFG, _replace(VL, text_config=_replace(TEXT, hidden_size=4096)), True),
        (CFG, _replace(VL, text_config=_replace(TEXT, num_experts=256)), True),
        (CFG, _replace(VL, text_config=_replace(TEXT, indexer_budget=1024)), True),
        (
            CFG,
            _replace(
                VL,
                text_config=_replace(TEXT, layer_types=["linear_attention"] * 48),
            ),
            True,
        ),
        (CFG, _replace(VL, quantization_config={"quant_method": "fp8"}), True),
        (
            CFG,
            _replace(VL, text_config=_replace(TEXT, mamba_ssm_dtype="float16")),
            True,
        ),
        (_replace(CFG, context_length=MAX_CTX), VL, True),
        (CFG, VL, False),
    ],
    ids=[
        "hidden size",
        "experts",
        "index budget",
        "layer pattern",
        "quantization",
        "ssm dtype",
        "context",
        "no flydsl",
    ],
)
def test_each_unsupported_setting_is_one_reason(cfg, hf_config, flydsl):
    assert len(_config_reasons(cfg=cfg, hf_config=hf_config, flydsl=flydsl)) == 1


def test_the_ple_layer_is_one_reason():
    # PLE on layer 1 (ple_layer_ids counts from 1) is built into the second launch
    text = _replace(TEXT, ple_layer_ids=[2, 6])
    assert len(_config_reasons(hf_config=_replace(VL, text_config=text))) == 1


def test_the_recipe_routes_verify_batches_of_one_to_eight():
    # NEXTN, 3 steps: 4 verify rows a request, the 4- to 32-row launches
    assert q38.SPEC.widths("gfx950") == tuple(range(4, 33, 4))
    assert q38.SPEC.devices["gfx950"].tp_sizes == (1,)
    assert covered_capture_sizes(
        widths=q38.SPEC.widths("gfx950"), tokens_per_request=4, sizes=[1, 2, 4, 8]
    ) == list(range(1, 9))
    # above 8 requests the normal path, at the configured sizes
    assert covered_capture_sizes(
        widths=q38.SPEC.widths("gfx950"), tokens_per_request=4, sizes=[1, 2, 4, 16]
    ) == [*range(1, 9), 16]


def test_the_requirements_are_declared_sglang_reads():
    # SGLang checks each by name: test_contract.py pins the names it reads
    reads = q38.SGLANG_READS
    assert set(q38.SPEC.requires.server_args) <= set(
        reads["srt/arg_groups/fields/*.py"]
    )
    assert set(q38.SPEC.requires.env) <= set(reads["srt/environ.py"])
    # NEXTN resolves to EAGLE; the kernel verifies 4 rows a request of one chain
    args = q38.SPEC.requires.server_args
    assert args["speculative_algorithm"] == {"EAGLE"}
    assert args["speculative_num_draft_tokens"] == {q38.TOKENS}
    assert args["enable_linear_replayssm_spec"] == {False}


class _Lm:
    """A model whose own forward records that it ran."""

    def __init__(self):
        self.model, self.ran = SimpleNamespace(), []

    def forward(self, *args):
        self.ran.append(args)
        return "the model's forward"


@pytest.mark.parametrize(
    "key",
    [
        registry.ShapeKey("target_verify", batch_size=2, tokens_per_request=2),
        registry.ShapeKey("target_verify", batch_size=9, tokens_per_request=4),
        registry.ShapeKey("decode", batch_size=4, tokens_per_request=1),
    ],
    ids=["two tokens", "nine requests", "decode"],
)
def test_a_step_the_launches_do_not_take_runs_the_models_forward(key):
    lm = _Lm()
    built = q38._Built(lm, runner=None)
    out = built.forward(registry.Plan(key=key), "ids", "positions", "batch")
    assert out == "the model's forward"
    assert lm.ran == [("ids", "positions", "batch", False, None)]


@pytest.mark.parametrize("mrope", [True, False], ids=["mrope", "text positions"])
def test_a_routed_verify_rotates_by_the_positions_sglang_applies(mrope):
    # with images or video, MRoPE's positions differ from the sequence positions;
    # the kernel takes each request's sequence start from seq_lens
    lm = _Lm()
    lm.is_mrope_enabled = mrope
    lm.lm_head = None
    lm.logits_processor = lambda ids, x, head, fb, aux: SimpleNamespace()
    launched = []
    runner = SimpleNamespace(launch=lambda fb, pos: launched.append(pos) or ("x", "h"))
    fb = SimpleNamespace(mrope_positions="mrope positions")
    key = registry.ShapeKey("target_verify", batch_size=2, tokens_per_request=4)
    q38._Built(lm, runner).forward(registry.Plan(key=key), "ids", "positions", fb)
    assert launched == ["mrope positions" if mrope else "positions"]
    verify = VerifyRunner.__new__(VerifyRunner)
    verify.plans, verify.L = launch_plans(["gdn"] * 3 + ["qsa"]), 4
    verify.eos_token_id, verify.fault_site, verify.fault_after = 1, 0, 0
    assert all(k.seq_p0 for k in verify.keys(2, {}))


# ---------------------------------------------------------------- build keys
def test_a_build_key_reaches_the_jit_cache_key():
    key = MonoBuild(plan="gdn,gdn,gdn,qsa", nreq=8, ntok=4, rtt_stride=262208)
    pairs = dict(key_tuple(key, "digest"))
    assert pairs["plan"] == "gdn,gdn,gdn,qsa" and pairs["nreq"] == 8
    assert pairs["iconv.step"] == ConvStrides().step and pairs["sources"] == "digest"
    assert key.layers == ("gdn", "gdn", "gdn", "qsa")
    # a frozen Struct: the runner caches its launchers by key
    assert {key: 1}[MonoBuild(plan="gdn,gdn,gdn,qsa", nreq=8, ntok=4, rtt_stride=262208)]  # fmt: skip


@pytest.mark.parametrize(
    "kw",
    [
        dict(plan="gdn,mla", nreq=1, ntok=4),
        dict(plan="", nreq=1, ntok=4),
        dict(plan="qsa", nreq=9, ntok=1),
        dict(plan="qsa", nreq=1, ntok=5),
        dict(plan="qsa", nreq=8, ntok=4, first_src="hidden"),
    ],
    ids=["kind", "empty", "requests", "tokens", "first source"],
)
def test_a_build_key_out_of_the_kernels_range_is_refused(kw):
    with pytest.raises(ValueError):
        MonoBuild(**kw)


def test_a_verify_is_thirteen_launches_of_the_48_layers():
    kinds = ["qsa" if t == "full_attention" else "gdn" for t in TEXT.layer_types]
    plans = launch_plans(kinds)
    assert len(plans) == 13
    assert plans[0] == ("gdn", [0]) and plans[1] == ("gdn,gdn,qsa", [1, 2, 3])
    assert all(p == "gdn,gdn,gdn,qsa" for p, _ in plans[2:])
    assert [i for _, ids in plans for i in ids] == list(range(48))


# ---------------------------------------------------------------- layout
@pytest.mark.parametrize("R", range(1, 9))
def test_the_scratch_regions_do_not_overlap(R):
    regions, total = LY.scratch_layout(6, 4 * R, R)
    offs = sorted(regions.values())
    assert offs == sorted(set(offs)) and offs[0] == 0 and total > offs[-1]
    assert all(off % 256 == 0 for off in offs)


def test_the_layout_matches_the_spec():
    assert LY.MAX_CTX == MAX_CTX
    assert LY.QP_RG_QKV * 16 + 640 == LY.QP_ROWS
    assert len(LAYER_PATTERN) == HF_CONFIG["full_attention_interval"]


# ---------------------------------------------------------------- layer views
def _meta(*shape, dtype=torch.bfloat16, **attrs):
    t = torch.empty(*shape, dtype=dtype, device="meta")
    for k, v in attrs.items():
        setattr(t, k, v)
    return t


def _fp8(n, k, scale=True):
    """A Quark W8A8 FP8 linear as SGLang keeps it: the [N, K] weight behind a
    transposed view, f32 per-channel scales."""
    return SimpleNamespace(
        weight=_meta(n, k, dtype=torch.float8_e4m3fn).t(),
        weight_scale=_meta(n, dtype=torch.float32) if scale else None,
    )


def _w(*shape, dtype=torch.bfloat16):
    return SimpleNamespace(weight=_meta(*shape, dtype=dtype))


def _hc():
    return SimpleNamespace(
        hc_norm=_w(10240),
        input_mix_weight_down=_w(320, 10240),
        input_mix_weight_up=_w(10240, 320),
        block_inject_weight=_w(4, 10240),
    )


def _experts():
    E = 513
    return SimpleNamespace(
        num_fused_shared_experts=1,
        moe_runner_config=SimpleNamespace(activation="silu"),
        w13_weight=_meta(E, 1536, 1280, dtype=torch.uint8, is_shuffled=True),
        w2_weight=_meta(E, 2560, 384, dtype=torch.uint8, is_shuffled=True),
        w13_weight_scale=_meta(E, 1536, 80, dtype=torch.uint8),
        w2_weight_scale=_meta(E, 2560, 24, dtype=torch.uint8),
    )


def _layer(kind, ple=False, **changes):
    """A decoder layer as SGLang builds it, its tensors on the meta device."""
    layer = SimpleNamespace(
        attn_hyper_connection=_hc(),
        mlp_hyper_connection=_hc(),
        mlp=SimpleNamespace(
            experts=_experts(), gate=_w(512, 2560), shared_expert_gate=_w(1, 2560)
        ),
        ple=None,
    )
    if kind == "gdn":
        layer.linear_attn = SimpleNamespace(
            in_proj_qkvz=_fp8(16384, 2560),
            in_proj_ba=_fp8(96, 2560),
            out_proj=_fp8(2560, 6144),
            A_log=_meta(48, dtype=torch.float32),
            dt_bias=_meta(48),
            conv1d=_w(10240, 1, 4),
            norm=_w(128),
        )
    else:
        layer.is_qsa = True
        layer.qkv_proj = _fp8(13312, 2560)
        layer.o_proj = _fp8(2560, 6144)
        layer.q_norm, layer.k_norm = _w(256), _w(256)
        layer.indexer = SimpleNamespace(
            index_qk_proj=_fp8(640, 2560), q_layernorm=_w(128), k_layernorm=_w(128)
        )
        layer.rotary_emb = SimpleNamespace(cos_sin_cache_as=lambda dtype: None)
    if ple:
        layer.ple = SimpleNamespace(
            key_proj=_w(10240, 2560),
            value_proj=_w(2560, 2560),
            norm_key=_w(10240),
            norm_query=_w(10240),
            norm_conv=_w(10240),
            conv1d=_w(10240, 1, 4),
            ple_embedding=SimpleNamespace(
                ngram_embedding=SimpleNamespace(
                    weight=_meta(1000, 160, dtype=torch.float8_e4m3fn),
                    weight_scale=_meta(1),
                )
            ),
        )
    for path, value in changes.items():
        owner, name = layer, path
        while "." in name:
            head, name = name.split(".", 1)
            owner = getattr(owner, head)
        setattr(owner, name, value)
    return layer


def test_the_views_accept_the_layers():
    assert weights.layer_reasons(_layer("gdn"), 0, "gdn") == []
    assert weights.layer_reasons(_layer("gdn", ple=True), 1, "gdn") == []
    assert weights.layer_reasons(_layer("qsa"), 3, "qsa") == []
    assert weights.ngram_reasons(_layer("gdn", ple=True).ple) == []
    assert weights.mixer_reasons(_hc()) == []


@pytest.mark.parametrize(
    "kind, changes",
    [
        ("gdn", {"mlp.experts.num_fused_shared_experts": 0}),
        (
            "gdn",
            {
                "mlp.experts.w13_weight": _meta(
                    513, 1536, 1280, dtype=torch.uint8, is_shuffled=False
                )
            },
        ),
        ("gdn", {"mlp.experts.moe_runner_config": SimpleNamespace(activation="situ")}),
        ("gdn", {"mlp.experts.w2_weight_scale": _meta(513, 2560, 24)}),
        ("gdn", {"linear_attn.out_proj": _fp8(2560, 6144, scale=False)}),
        ("gdn", {"linear_attn.in_proj_ba": _fp8(256, 2560)}),
        ("gdn", {"linear_attn.in_proj_qkvz": _w(16384, 2560)}),
        ("gdn", {"linear_attn.conv1d": _w(10240, 4)}),
        ("gdn", {"attn_hyper_connection.hc_norm": _w(10240, dtype=torch.float32)}),
        ("qsa", {"indexer.index_qk_proj": _fp8(640, 1280)}),
        ("qsa", {"q_norm": _w(128)}),
        ("qsa", {"mlp.gate": _w(256, 2560)}),
    ],
    ids=[
        "no fused shared expert",
        "unshuffled experts",
        "interleaved gate and up",
        "2-byte scales",
        "dequantized projection",
        "wide b / a projection",
        "bf16 projection",
        "conv weight",
        "f32 norm",
        "index projection",
        "q norm",
        "router",
    ],
)
def test_each_unsupported_view_is_refused(kind, changes):
    assert len(weights.layer_reasons(_layer(kind, **changes), 3, kind)) == 1


def test_an_offloaded_ngram_table_is_refused():
    ple = _layer("gdn", ple=True).ple
    ple.ple_embedding.ngram_embedding.weight = torch.empty(
        8, 160, dtype=torch.float8_e4m3fn
    )
    assert len(weights.ngram_reasons(ple)) == 1


def _model(**changes):
    kinds = ["qsa" if t == "full_attention" else "gdn" for t in TEXT.layer_types]
    layers = [_layer(k, ple=i == 1) for i, k in enumerate(kinds)]
    m = SimpleNamespace(
        layers=layers,
        start_layer=0,
        end_layer=len(layers),
        hyper_connection_mixer=_hc(),
        embed_tokens=_w(248320, 2560),
    )
    lm = SimpleNamespace(model=m, capture_aux_hidden_states=False)
    for name, value in changes.items():
        setattr(lm, name, value)
    return lm


def _model_reasons(lm, tp=1, context_len=262144):
    parallel = SimpleNamespace(tp_size=tp, pp_size=1, attn_cp_size=1, moe_ep_size=1)
    with (
        mock.patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
        mock.patch(
            "sglang.srt.runtime_context.process_model_config",
            return_value=SimpleNamespace(context_len=context_len),
        ),
    ):
        return q38._model_disable_reasons(lm, VL, {})


def test_the_model_is_accepted():
    assert _model_reasons(_model()) == []


def test_each_unsupported_model_is_one_reason():
    assert len(_model_reasons(_model(), tp=2)) == 1
    assert len(_model_reasons(_model(), context_len=MAX_CTX)) == 1
    assert len(_model_reasons(_model(capture_aux_hidden_states=True))) == 1
    lm = _model()
    lm.model.layers[5].ple = lm.model.layers[1].ple
    assert len(_model_reasons(lm)) == 1
    lm = _model()
    lm.model.end_layer = 24  # pipeline stage
    assert len(_model_reasons(lm)) == 1


def test_the_mtp_draft_views_are_checked():
    lm = SimpleNamespace(
        model=SimpleNamespace(layers=[_layer("qsa")], hyper_connection_mixer=_hc()),
        pre_fc_norm_embedding=_w(2560),
        pre_fc_norm_hidden=_w(10240),
        fc_embedding=_w(2560, 2560),
        fc_hidden=_w(2560, 2560),
    )
    assert weights.mtp_reasons(lm) == []
    lm.fc_hidden = None
    assert len(weights.mtp_reasons(lm)) == 1
