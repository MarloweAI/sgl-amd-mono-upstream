# SPDX-License-Identifier: Apache-2.0
"""Each Qwen3.8-Flash-Next launch kind on one gfx950 GPU, on random weights in
SGLang's post-load gfx950 layouts (no checkpoint): 12 layers (launches [0], [1-3]
with the PLE, [4-7], [8-11] with the final mixer) and the MTP head, on pools of
SGLang's shapes with random state. Verify steps (prelude, PLE, middle, final-mixer
launches, MRoPE positions) of 1, 8 and 7 + 1 pad requests, MTP draft extend and
decode steps of 1 and 8. Every step leaves the error words zero and its outputs
finite, writes only its own pool rows (a pad request: KV slot 0, request slot 0's
ring, its scratch rows), also from a CUDA graph, which replays an unpadded step
bitwise and a padded one's layers 0-3 states but KV slot 0 (see the README's
Limitations). The prelude's n-gram embedding and context are bitwise a torch
reference's. Every MoE block's router logits match an f32 router, its output an
a16w4 torch MoE on the launch's input and routes. The GDN and QSA layers are checked
against SGLang's forward by check mode (``test/e2e``)."""

import contextlib
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("flydsl")
if (
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
):
    pytest.skip("gfx950 only", allow_module_level=True)

from aiter.utility.fp4_utils import mxfp4_to_f32  # noqa: E402

from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.qwen38_flash_next.gfx950 import layout as LY  # noqa: E402
from sgl_amd_mono.qwen38_flash_next.gfx950 import packing  # noqa: E402
from sgl_amd_mono.qwen38_flash_next.gfx950 import weights as W  # noqa: E402
from sgl_amd_mono.qwen38_flash_next.gfx950.config import (  # noqa: E402
    GDN_CONV_DIM,
    GDN_HV,
    HC_DIM,
    HIDDEN,
    MOE_INTER,
    N_EXPERTS,
    TOPK,
)

DEV = "cuda"
RING = 8  # index-key ring slots a request: two groups, as SGLang's QSA pool keeps them
KINDS = ["qsa" if i % 4 == 3 else "gdn" for i in range(12)]
EOS, VOCAB = 248044, 248320
L, MAX_R = 4, 8  # verify rows a request (3 draft steps + 1), requests
REQUESTS = (1, 8)  # 7 + 1 pad builds the launches of 8
# request r's context: every context % 4, past one 8192-block index segment, and a
# step crossing the first segment boundary
CONTEXTS = (60, 1203, 9000, 70001, 32766, 2048, 140003, 60)
# request r's KV slots start at BASES[r]
BASES = [64 + sum((c + L + 127) // 64 * 64 for c in CONTEXTS[:r]) for r in range(MAX_R)]


def gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEV).manual_seed(seed)


# ---------------------------------------------------------------- random weights
def _random(shapes: dict, g) -> dict:
    """Random bf16 tensors by name: matrices [N, K] at 1 / sqrt(K) (outputs at their
    inputs' scale), Gemma norm offsets near 0, the GDN gated norm near 1, decays
    exp(A_log) from 1, dt_bias near -3."""
    out = {}
    for n, shape in shapes.items():
        x = torch.randn(shape, generator=g, device=DEV)
        if n.endswith("A_log"):
            x = x.abs()
        elif n.endswith("dt_bias"):
            x = x * 0.5 - 3
        elif n.endswith("linear_attn.norm.weight"):
            x = 1 + 0.1 * x
        elif "conv1d" in n:
            x = 0.5 * x
        else:
            x = x * (0.1 if len(shape) == 1 else shape[-1] ** -0.5)
        out[n] = x.bfloat16()
    return out


def _moe(g) -> dict:
    """FusedMoE's MXFP4 params as SGLang keeps them on gfx950 (AITER): the routed
    experts and the shared one (index 512), the intermediate dim zero-padded 640 ->
    768, ``shuffle_weight(w, (16, 16))`` and ``e8m0_shuffle``'d scales. Random codes,
    E8M0 scales 2^-8 .. 2^-5: gate / up / down outputs of a few units."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    E, I, IP = N_EXPERTS + 1, MOE_INTER, LY.MOE_INTER_PAD

    def mxfp4(rows, k):
        u8 = dict(dtype=torch.uint8, generator=g, device=DEV)
        w = torch.randint(0, 256, (E, rows, k // 2), **u8)
        s = torch.randint(119, 123, (E * rows, k // 32), **u8)
        return w, e8m0_shuffle(s).view(E, rows, -1)

    gu, gu_s = mxfp4(2 * IP, HIDDEN)
    gu.view(E, 2, IP, -1)[:, :, I:] = 0  # gate rows 0 .. 639, up rows 768 .. 1407
    dn, dn_s = mxfp4(HIDDEN, IP)
    dn[..., I // 2 :] = 0
    m = {"gu": shuffle_weight(gu, (16, 16)), "gu_s": gu_s,
         "dn": shuffle_weight(dn, (16, 16)), "dn_s": dn_s}  # fmt: skip
    packing.check_moe_params(m)
    return m


def _proj(kind: str, g) -> dict:
    """A layer's merged FP8 projections as SGLang keeps them (``weights.GDN_PROJ`` /
    ``QSA_PROJ``): random codes, (16, 16)-shuffled, f32 per-channel scales 1 / sqrt(K);
    in_proj_ba's pad rows zero, scale 1."""
    from aiter.ops.shuffle import shuffle_weight

    out = {}
    for key, (_, rows, k) in (W.GDN_PROJ if kind == "gdn" else W.QSA_PROJ).items():
        real = 2 * GDN_HV if key == "ba" else rows
        w = torch.randn(rows, k, generator=g, device=DEV)
        s = torch.full((rows,), k**-0.5, device=DEV)
        w[real:], s[real:] = 0, 1
        out[key] = (shuffle_weight(w.to(torch.float8_e4m3fn), (16, 16)), s)
    return out


def _small(kind: str, ple: bool, g) -> dict:
    """A layer's small tensors by their checkpoint names in the layer."""
    return _random({k: s for k, (_, s, _) in W._small(kind, ple).items()}, g)


def _mixer(p: str, g) -> dict:
    return _random({k: s for k, (s, _) in W._hc_tensors(p, inject=False).items()}, g)


def _ngram(g) -> dict:
    """The PLE n-gram tables: odd layer multipliers below 2^63 / VOCAB (as SGLang
    draws them), 16 head vocab sizes, one FP8 table every head shares."""
    mult = torch.randint(2**44, (3,), generator=g, device=DEV) * 2 + 1
    vocab = torch.randint(50000, 100000, (16,), generator=g, device=DEV)
    table = torch.randn(int(vocab.max()), LY.NGRAM_DIM, generator=g, device=DEV)
    return {
        "table": (table * 32).to(torch.float8_e4m3fn),
        "scale": torch.full((1,), 1 / 32, dtype=torch.bfloat16, device=DEV),
        "meta": torch.cat([mult, vocab, torch.zeros_like(vocab)]),
    }


@lru_cache(maxsize=1)
def cos_sin() -> torch.Tensor:
    """SGLang's bf16 RoPE table [262144, 64] (theta 1e7, 64 rotated dims)."""
    inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, device=DEV).float() / 64))
    f = torch.outer(torch.arange(262144, device=DEV, dtype=torch.float), inv)
    return torch.cat([f.cos(), f.sin()], -1).bfloat16().contiguous()


@lru_cache(maxsize=1)
def embedding() -> torch.Tensor:
    return torch.randn(VOCAB, HIDDEN, generator=gen(2), device=DEV).bfloat16()


def _compile(key) -> None:
    from sgl_amd_mono.qwen38_flash_next.gfx950 import kernel, runner

    runner.ensure_writable_flydsl_cache()
    runner.compile_only(kernel.build_qwen38_flash_next_kernel(key))


def precompile(keys) -> None:
    """Compile ``keys`` in 16 processes (FlyDSL's compile-only mode, to its cache)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    keys = list(dict.fromkeys(keys))
    with ProcessPoolExecutor(min(16, len(keys)), mp_context=get_context("spawn")) as ex:
        list(ex.map(_compile, keys))


# ---------------------------------------------------------------- pools and steps
class Pools:
    """SGLang's req-to-token and KV pools in one, for ``MAX_R`` requests over
    ``CONTEXTS``: request r at request slot 2 r + 3, mamba slot 5 MAX_R - 2 r and KV
    slots BASES[r] + position; the verify scratch rows (request r owns row r) and the
    deduplicated intermediate conv window SGLang allocates on ROCm."""

    def __init__(self, kinds, seed: int = 0):
        g = gen(seed)
        R = MAX_R
        self.reqs = [2 * r + 3 for r in range(R)]
        n_req, n_slot, n_scr = max(self.reqs) + 2, 5 * R + 2, R + 1
        off = BASES[-1] + (CONTEXTS[-1] + L + 127) // 64 * 64
        stride = (max(CONTEXTS) + L + 64 + 255) // 256 * 256
        self.req_to_token = torch.zeros(n_req, stride, dtype=torch.int32, device=DEV)
        self.req_index_to_mamba_index_mapping = torch.zeros(n_req, dtype=torch.int32, device=DEV)  # fmt: skip
        for r, c in enumerate(CONTEXTS):
            self.req_to_token[self.reqs[r], : c + L] = BASES[r] + torch.arange(c + L, device=DEV)  # fmt: skip
            self.req_index_to_mamba_index_mapping[self.reqs[r]] = 5 * R - 2 * r

        def rnd(*shape, scale=1.0):
            return (torch.randn(*shape, generator=g, device=DEV) * scale).bfloat16()

        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(*shape, dtype=dtype, device=DEV)

        self.mamba, self.kv, self.iconv_phys = {}, {}, {}
        for i, kind in enumerate(kinds):
            if kind == "gdn":
                # draft token j's conv window: the physical row's columns j .. j + 2
                phys = self.iconv_phys[i] = zeros(n_scr, GDN_CONV_DIM, L + 2)
                s = phys.stride()
                icw = phys.as_strided((n_scr, L, GDN_CONV_DIM, 3), (s[0], s[2], s[1], s[2]))  # fmt: skip
                self.mamba[i] = SimpleNamespace(
                    conv=[rnd(n_slot, GDN_CONV_DIM, 3, scale=0.5)],
                    temporal=rnd(n_slot, 48, 128, 128, scale=0.02),
                    intermediate_ssm=zeros(n_scr, L, 48, 128, 128),
                    intermediate_conv_window=[icw],
                )
                continue
            k, v, comp = zeros(off + 64, 2, 256), zeros(off + 64, 2, 256), zeros(off // 4 + 32, 1, 128)  # fmt: skip
            for b, c in zip(BASES, CONTEXTS):
                k[b : b + c], v[b : b + c] = rnd(c, 2, 256, scale=0.5), rnd(c, 2, 256, scale=0.5)  # fmt: skip
                comp[b // 4 : (b + c) // 4] = rnd((b + c) // 4 - b // 4, 1, 128, scale=0.5)  # fmt: skip
            self.kv[i] = SimpleNamespace(k=k, v=v, comp=comp, ring=rnd(n_req * RING, 1, 128, scale=0.5))  # fmt: skip
        # the RoPE positions of each request's last keys c - RING .. c - 1
        self.qsa_rope_position_buffer = zeros(n_req * RING, 3, dtype=torch.int64)
        self.qsa_num_request_slots = n_req
        for r, c in enumerate(CONTEXTS):
            for q in range(RING):
                self.qsa_rope_position_buffer[self.reqs[r] * RING + q] = c - 1 - (c - 1 - q) % RING  # fmt: skip
        self.short_conv = rnd(n_slot, HC_DIM, LY.PLE_STATE_LEN, scale=0.5)
        self.short_conv_ist = zeros(n_scr, L, HC_DIM, LY.PLE_STATE_LEN)
        ctx = torch.randint(0, 248000, (n_slot, 2), generator=g, device=DEV)
        ictx = torch.full((n_scr, L, 2), EOS, dtype=torch.int64, device=DEV)
        self.ngram_pool = SimpleNamespace(context=ctx, intermediate_context=ictx)
        # the accessors the runners call
        self.mamba2_layer_cache = self.mamba.__getitem__
        self.layer_cache = lambda i: self.short_conv
        self.layer_intermediate_cache = lambda i: self.short_conv_ist
        self.layer_router = SimpleNamespace(read=lambda i, part: SimpleNamespace(view=self, view_layer_id=i))  # fmt: skip
        self.get_key_buffer = lambda i: self.kv[i].k
        self.get_value_buffer = lambda i: self.kv[i].v
        self.get_qsa_key_state_buffer = lambda i: self.kv[i].ring
        self.get_qsa_compressed_k_buffer = lambda i: self.kv[i].comp

    def install(self) -> None:
        """Make these the pools SGLang's forward context hands the runners."""
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            set_forward_context,
        )

        backend = SimpleNamespace(req_to_token_pool=self, token_to_kv_pool=self)
        set_forward_context(ForwardContext(attn_backend=backend))

    def tensors(self) -> dict:
        """Every pool tensor by name (the conv windows' physical rows)."""
        ngram = self.ngram_pool
        out = {"rope_pos": self.qsa_rope_position_buffer, "short_conv": self.short_conv,
               "short_conv_ist": self.short_conv_ist, "ngram_ctx": ngram.context,
               "ngram_ictx": ngram.intermediate_context}  # fmt: skip
        for i, m in self.mamba.items():
            out |= {f"L{i}.conv": m.conv[0], f"L{i}.ssm": m.temporal,
                    f"L{i}.issm": m.intermediate_ssm, f"L{i}.iconv": self.iconv_phys[i]}  # fmt: skip
        for i, kv in self.kv.items():
            out |= {f"L{i}.{n}": getattr(kv, n) for n in ("k", "v", "ring", "comp")}
        return out

    def snapshot(self) -> dict:
        return {k: t.clone() for k, t in self.tensors().items()}

    def restore(self, snap: dict) -> None:
        for k, t in self.tensors().items():
            t.copy_(snap[k])

    def allowed_rows(self, b) -> dict:
        """Pool tensor -> the rows step ``b`` may write (none if absent): its KV slots
        and compressed groups, its requests' ring and RoPE rows, its scratch rows."""
        locs = b.out_cache_loc.unique()
        ring = (b.req_pool_indices[:, None] * RING + torch.arange(RING, device=DEV)).flatten()  # fmt: skip
        scratch = torch.arange(b.batch_size, device=DEV)
        out = {"rope_pos": ring, "short_conv_ist": scratch, "ngram_ictx": scratch}
        for i in self.mamba:
            out |= {f"L{i}.issm": scratch, f"L{i}.iconv": scratch}
        for i in self.kv:
            out |= {f"L{i}.k": locs, f"L{i}.v": locs, f"L{i}.ring": ring, f"L{i}.comp": (locs // 4).unique()}  # fmt: skip
        return out


def step_batch(R: int, rows: int = L, seed: int = 1, pads: int = 0) -> SimpleNamespace:
    """SGLang's forward batch fields of a step of the first R requests, ``rows`` rows
    each at their next positions, and ``pads`` pad requests as SGLang pads a graph's
    batch (request slot 0, KV slot 0, position 0, a sequence of 1)."""
    ctx, pad = CONTEXTS[:R], [0] * pads * rows
    ids = torch.randint(0, 248000, ((R + pads) * rows,), generator=gen(seed), device=DEV)  # fmt: skip
    ids[1:2] = EOS  # an n-gram across an EOS
    pos = [c + j for c in ctx for j in range(rows)]
    locs = [b + c + j for b, c in zip(BASES, ctx) for j in range(rows)]
    reqs = [2 * r + 3 for r in range(R)] + [0] * pads
    pos = torch.tensor(pos + pad, device=DEV)
    # the MRoPE positions [3, S]: a view of SGLang's [3, max tokens] graph buffer
    mrope = torch.zeros(3, MAX_R * L, dtype=torch.int64, device=DEV)[:, : pos.numel()]
    return SimpleNamespace(
        batch_size=R + pads,
        input_ids=ids,
        positions=pos,
        mrope_positions=mrope.copy_(pos.expand(3, -1)),
        out_cache_loc=torch.tensor(locs + pad, device=DEV),
        req_pool_indices=torch.tensor(reqs, device=DEV),
        seq_lens=torch.tensor(list(ctx) + [1] * pads, device=DEV),
    )


def _router(small: dict, p: str = "") -> torch.Tensor:
    """The 512 routed experts' and the shared expert's gate rows, f32."""
    gate = [small[p + "mlp.gate.weight"], small[p + "mlp.shared_expert_gate.weight"]]
    return torch.cat(gate).float()


@pytest.fixture(scope="module")
def target():
    from sgl_amd_mono.qwen38_flash_next.gfx950.runner import VerifyRunner

    g, layers, small = gen(0), {}, {}
    for i, kind in enumerate(KINDS):
        lw = small[i] = _small(kind, i == 1, g)
        pack = packing.pack_gdn_layer if kind == "gdn" else packing.pack_qsa_layer
        layers[i] = pack(lw, moe=_moe(g), proj=_proj(kind, g))
        if i == 1:
            layers[i]["ple"] = packing.pack_ple(lw)
    mix = _mixer("m", g)
    final = {"norm": mix["m.hc_norm.weight"],
             "down": packing.pack_hc_down(mix["m.input_mix_weight_down.weight"]),
             "up": packing.pack_bf16(mix["m.input_mix_weight_up.weight"])}  # fmt: skip
    ngram = _ngram(g)
    runner = VerifyRunner(kinds=KINDS, layers=layers, final=final, ngram=ngram,
                          embed_w=embedding(), cos_sin=cos_sin, ntok=L, max_reqs=MAX_R,
                          eos_token_id=EOS, device=DEV)  # fmt: skip
    runner.set_deadline(30.0)
    t = SimpleNamespace(ngram=ngram, runner=runner, pools=Pools(KINDS))
    t.pools.install()
    steps = [step_batch(R) for R in REQUESTS]
    precompile(k for b in steps for k in runner.prepare_step(b, b.mrope_positions)[1])
    t.snap = t.pools.snapshot()
    t.blocks = {i: (_router(small[i]), layers[i]["moe"]) for i in layers}
    return t


@pytest.fixture(scope="module")
def draft():
    from sgl_amd_mono.qwen38_flash_next.gfx950.runner import DraftRunner

    g = gen(1)
    lw = {f"layers.0.{k}": v for k, v in _small("qsa", False, g).items()}
    lw |= _random(W._MTP_FUSION, g) | _mixer("hyper_connection_mixer", g)
    views = packing.pack_mtp(lw, moe=_moe(g), proj=_proj("qsa", g))
    runner = DraftRunner(views=views, embed_w=embedding(), cos_sin=cos_sin, layer_id=0,
                         ntok_extend=L, max_reqs=MAX_R, eos_token_id=EOS, device=DEV)  # fmt: skip
    runner.set_deadline(30.0)
    d = SimpleNamespace(runner=runner, pools=Pools(["qsa"]))
    d.pools.install()
    steps = [step_batch(R, rows) for R in REQUESTS for rows in (L, 1)]
    hs = torch.zeros(MAX_R * L, HC_DIM, dtype=torch.bfloat16, device=DEV)
    precompile(runner.prepare_step(b, hs[: b.input_ids.numel()])[1] for b in steps)
    d.snap = d.pools.snapshot()
    d.blocks = [(_router(lw, "layers.0."), views["layer"]["moe"])]
    return d


# ---------------------------------------------------------------- checks
def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _mailbox(runner, key, layer: int, name: str, n: int, tag: int, whole=True):
    """The first n (value, tag) pairs of mailbox ``name`` of the launch's layer:
    (values int32 [n], tagged by this launch bool [n])."""
    S = key.nreq * key.ntok
    w0 = LY.scratch_layout(layer + 1, S, key.nreq)[0][f"L{layer}.{name}"] // 4
    pairs = runner.scratch[w0 : w0 + 2 * n].view(n, 2)
    ok = pairs[:, 1] == tag
    assert not whole or bool(ok.all()), f"layer {layer} {name}: stale pairs"
    return pairs[:, 0].contiguous(), ok


@contextlib.contextmanager
def _after_each_launch(runner, check):
    """``check(key, tag)`` after each launch of the runner's eager steps; its first
    failure is raised once the step is done (an unfinished step leaves its tags)."""
    build, failed = runner.kernel, []

    def kernel(key):
        launch = build(key)

        def run(*args, stream):
            launch(*args, stream=stream)
            torch.cuda.synchronize()
            try:
                check(key, int(runner.epoch[0]) * LY.LAUNCH_SLOTS + args[5] + 1)
            except AssertionError as e:
                failed.append(e)

        return run

    runner.kernel = kernel
    try:
        yield
    finally:
        del runner.kernel
    if failed:
        raise failed[0]


def _error_words(runner) -> list:
    return runner.epoch[list(guard.ERROR_WORDS)].tolist()


def _stray_rows(pools, snap: dict, b) -> list:
    """The pool rows the step wrote outside its own, and the states it owns but
    left unwritten."""
    allowed, bad = pools.allowed_rows(b), []
    for name, t in pools.tensors().items():
        ne = (_bits(t) != _bits(snap[name])).view(t.shape[0], -1).any(1)
        rows, ok = ne.nonzero().flatten(), allowed.get(name)
        stray = rows if ok is None else rows[~torch.isin(rows, ok)]
        if stray.numel():
            bad.append(f"{name} rows {stray[:4].tolist()}")
        if name.endswith((".k", ".v", ".issm", ".iconv")) and not rows.numel():
            bad.append(f"{name} unwritten")
    return bad


def _run_step(m, b, run, check, pads: int = 0) -> None:
    """Step ``b`` (``run()`` -> its outputs) eagerly with ``check`` after each launch,
    then from a CUDA graph, each from the same pools."""
    pools, runner = m.pools, m.runner
    pools.install()
    pools.restore(m.snap)
    with _after_each_launch(runner, check):
        eager = [t.clone() for t in run()]
    assert not any(err := _error_words(runner)), err
    assert not (bad := _stray_rows(pools, m.snap, b)), bad
    assert all(e.isfinite().all() for e in eager), "non-finite outputs"
    after = pools.snapshot()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = run()
    pools.restore(m.snap)
    graph.replay()
    torch.cuda.synchronize()
    assert not any(err := _error_words(runner)), err
    assert not (bad := _stray_rows(pools, m.snap, b)), bad
    # the pads' rows race on KV slot 0 from layer 3's QSA on, so the pads' states and
    # routes vary run to run, and a row's MoE sum order follows every row's routes: a
    # padded step replays only layers 0-3's states bitwise, but KV slot 0
    if not pads:
        assert all(_same(o, e) for o, e in zip(outs, eager)), "graph outputs"
    for n, t in pools.tensors().items():
        if pads and n[0] == "L" and int(n[1 : n.index(".")]) > 3:
            continue
        lo = pads if n in ("L3.k", "L3.v") else 0
        assert _same(t[lo:], after[n][lo:]), f"graph pool {n} differs"
    pools.restore(m.snap)


def _check_prelude(t, b) -> None:
    """The n-gram embedding and context as ``Qwen4ExpNGramEmbedding`` computes them,
    bitwise."""
    R, S = b.batch_size, b.batch_size * L
    pools, ngram = t.pools, t.ngram
    slots = pools.req_index_to_mamba_index_mapping[b.req_pool_indices].long()
    seq = torch.cat([pools.ngram_pool.context[slots], b.input_ids.view(R, -1)], 1)
    t0, t1, t2 = seq.unfold(1, 3, 1).reshape(S, 3).unbind(1)
    mult, vocab = ngram["meta"][:3], ngram["meta"][3:19]
    p2 = torch.where((t0 == EOS) | (t1 == EOS), EOS, t0)
    bigram = (t2 * mult[0]) ^ (t1 * mult[1])
    mix = torch.stack([bigram, bigram ^ (p2 * mult[2])], 1).repeat_interleave(8, 1)
    rows = ngram["table"][torch.remainder(mix, vocab).flatten()]
    want = (rows.bfloat16() * ngram["scale"]).view(S, -1)
    assert _same(t.runner.emb_buf[:S], want), "PLE n-gram embedding"
    ictx = pools.ngram_pool.intermediate_context[:R]
    assert torch.equal(ictx, seq.unfold(1, 2, 1)[:, 1:]), "n-gram context"


def _deq(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    v = mxfp4_to_f32(w.contiguous()).unflatten(-1, (-1, 32))
    return (v * torch.exp2(s.float() - 127)[..., None]).flatten(-2)


def _torch_moe(x: torch.Tensor, moe: dict, logits: torch.Tensor) -> torch.Tensor:
    """The MXFP4-weight, bf16-activation MoE on the top-10 of softmax(logits)
    renormalized plus the shared expert (sigmoid gate), f32 [S, HIDDEN]."""
    p = logits[:, :N_EXPERTS].softmax(-1)
    wts, ids = p.topk(TOPK, -1)
    wts = torch.cat([wts / wts.sum(-1, keepdim=True), logits[:, N_EXPERTS:].sigmoid()], 1)  # fmt: skip
    ids = torch.cat([ids, torch.full_like(ids[:, :1], N_EXPERTS)], 1)
    y = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for e in ids.unique().tolist():
        g, u, d, gs, us, ds = packing.unshuffle_moe_expert(moe, e)
        rows, k = (ids == e).nonzero(as_tuple=True)
        xe = x[rows].float()
        mid = (F.silu(xe @ _deq(g, gs).T) * (xe @ _deq(u, us).T)).bfloat16().float()
        y.index_add_(0, rows, wts[rows, k, None] * (mid @ _deq(d, ds).T))
    return y


def _check_moe(runner, key, tag: int, blocks: list) -> None:
    """The launch's MoE blocks (``blocks``: (router, experts) a layer) from its
    mailboxes: the input x, the router logits, the down partials of every group."""
    S = key.nreq * key.ntok
    for li, (router, moe) in enumerate(blocks):
        x = _mailbox(runner, key, li, "m.mixx", S * HIDDEN // 2, tag)[0]
        x = x.view(torch.bfloat16).view(S, -1)
        logits = _mailbox(runner, key, li, "logit", S * LY.ROUTER_ROWS, tag)[0]
        logits = logits.view(torch.float32).view(S, -1)[:, : N_EXPERTS + 1]
        want = x.float() @ router.T
        rel = (logits - want).norm() / want.norm()
        assert rel < 1e-5, f"launch layer {li}: router logits rel {rel}"
        groups = LY.max_groups(S, LY.dn_sg(S))
        words, ok = _mailbox(runner, key, li, "dpart", groups * S * HIDDEN, tag, False)
        live = ok.view(groups, -1).all(1)
        y = words.view(torch.float32).view(groups, S, HIDDEN)[live].sum(0)
        cos = F.cosine_similarity(y, _torch_moe(x, moe, logits), dim=1)
        assert cos.min() > 0.9999, f"launch layer {li}: MoE cosine {cos.min()}"


@pytest.mark.parametrize(
    "requests,pads", [(1, 0), (8, 0), (7, 1)], ids=["R=1", "R=8", "R=7+1 pad"]
)
def test_verify_step(target, requests, pads):
    b = step_batch(requests, seed=requests, pads=pads)
    plans = target.runner.plans

    def check(key, tag):
        i = (tag - 1) % LY.LAUNCH_SLOTS
        _check_moe(target.runner, key, tag, [target.blocks[li] for li in plans[i][1]])
        if i == 0 and not pads:
            _check_prelude(target, b)

    run = lambda: target.runner.launch(b, b.mrope_positions)  # noqa: E731
    _run_step(target, b, run, check, pads)


@pytest.mark.parametrize("rows", [L, 1], ids=["extend", "decode"])
@pytest.mark.parametrize("requests", REQUESTS, ids=["R=1", "R=8"])
def test_draft_step(draft, requests, rows):
    b = step_batch(requests, rows, seed=10 + requests)
    hs = torch.randn(requests * rows, HC_DIM, generator=gen(11), device=DEV).bfloat16()

    def check(key, tag):
        _check_moe(draft.runner, key, tag, draft.blocks)

    _run_step(draft, b, lambda: draft.runner.launch(b, hs), check)
