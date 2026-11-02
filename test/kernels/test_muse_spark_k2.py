"""The Muse Spark K2 launch on one gfx950 GPU (a TP group of one: the peer push
and the rank-ordered sum go to itself) against the math of the ROCm model's ops
it replaces: the fused output norm-gate, o_proj (one bf16 rounding; the
all-reduce sums one partial), the attention boundary (weightless post norm,
gated residual in fp32, pre-FFN norm), the joint router / pooler's top-8 on the
bf16 hi + lo router with the sigmoid scores, and the pooler with the pre-expert
norm; and that it compiles for every served width at TP 8.

The 8-rank test (later): ``torchrun --nproc-per-node 8``, every rank building
``K2Build(tokens, npes=8)``, a gloo group for ``PeerBuffer(sym_bytes(8) +
guard.FLAG_BYTES, group, rank, 8, dev)``, the launch given ``peers.local`` as
``sym`` and ``peers.addresses.data_ptr()`` as ``peers``; each rank's stream and
x against a reference whose o_proj output is ``torch.distributed.all_reduce`` of
the ranks' bf16 partials summed in fp32 and rounded once (the old
``test_muse_spark_peers.py`` pattern), every rank agreeing on ids and scores.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx950 GPU",
)

LAYER = 5
# the routing key keeps 24 bits of the biased score: two candidates closer than
# that (or than the GEMM order's noise) may swap at the top-8 boundary
TIE_TOL = 1e-3  # a near-tie may flip: the kernel's x is rstd2 (b1 + rstd b2) from two bf16 halves (the
# boundary's norms folded in by the GEMV CTAs), a few 1e-4 from the ops' one rounding of the normed row


def _epoch(device, deadline_s=20.0):
    from sgl_amd_mono.common import guard

    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return e


def _bf(t):
    return t.to(torch.bfloat16).float()


def _rms(x, eps, w=None):
    """RMSNorm in fp32, the weight applied after the scale, one bf16 rounding."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if w is not None:
        y = y * w.float()
    return y.to(torch.bfloat16)


def _reference(o, g, stream, w):
    """The ops' math: ``_rms_norm_gate_fwd_kernel`` (norm x sigmoid, one rounding),
    the o_proj GEMM rounded once, ``_fused_norm_residual_triton_kernel`` with a
    weightless post norm and bf16 round-trips, ``_joint_epilogue`` (sigmoid, bias,
    repeated argmax with the pick masked: a tie to the lowest id; the pooled row
    rounded before its norm)."""
    from sgl_amd_mono.muse_spark.gfx950.config import (
        HEAD_DIM,
        POST_EPS,
        RMS_EPS,
        ROUTE_EPS,
        ROUTE_SCALE,
        ROUTER_MULT,
        TOP_K,
    )

    rows = o.shape[0]
    bf = torch.bfloat16
    of = o.float().reshape(rows, -1, HEAD_DIM)
    rstd = torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    og = (of * rstd * torch.sigmoid(g.float().reshape(rows, -1, HEAD_DIM))).to(bf)
    ao = (og.reshape(rows, -1).float() @ w["w_o"].float().T).to(bf)  # TP 1: no sum
    nb = _rms(ao, POST_EPS).float()
    new = w["alpha"] * stream + w["beta"] * nb
    x = _rms(new, RMS_EPS, w["pre_w"])
    logits = x.float() @ w["w_hi"].float().T + x.float() @ w["w_lo"].float().T
    sc = torch.sigmoid(logits * ROUTER_MULT)
    biased = sc + w["bias"]
    order = torch.argsort(
        biased, dim=-1, descending=True, stable=True
    )  # a tie: the lower id first
    ids = order[:, :TOP_K]
    picks = sc.gather(1, ids)
    scores = picks / (picks.sum(-1, keepdim=True) + ROUTE_EPS) * ROUTE_SCALE
    pooled = (x.float() @ w["w_pool"].float().T).to(bf)
    xn = _rms(pooled, RMS_EPS, w["xn_w"])
    return new, ids.to(torch.int32), scores, xn, biased


def _close(got, ref, rtol, atol, what):
    got, ref = got.float(), ref.float()
    bad = ((got - ref).abs() > rtol * ref.abs() + atol).float().mean().item()
    assert bad == 0.0, f"{what}: {bad:.2%} of the elements off by more than {rtol} relative + {atol}"  # fmt: skip


def _check_routing(ids, scores, ids_ref, scores_ref, biased):
    """Every token's picks are the reference's, unless a pick differs from one
    within TIE_TOL of the boundary; the scores of the shared picks agree."""
    from sgl_amd_mono.muse_spark.gfx950.config import EXPERTS, TOP_K

    for t in range(ids.shape[0]):
        got = ids[t].tolist()
        ref = ids_ref[t].tolist()
        assert len(set(got)) == TOP_K and all(0 <= e < EXPERTS for e in got), (t, got)
        extra = sorted(set(got) - set(ref))
        if extra:
            bound = biased[t, ref].min().item()  # the weakest reference pick
            worst = biased[t, extra].min().item()
            assert worst >= bound - TIE_TOL, f"token {t}: picks {got} vs {ref}"
        ref_at = {e: s for e, s in zip(ref, scores_ref[t].tolist())}
        for e, s in zip(got, scores[t].tolist()):
            if e in ref_at:
                assert abs(s - ref_at[e]) <= 1e-3 + 1e-2 * abs(ref_at[e]), (
                    t,
                    e,
                    s,
                    ref_at[e],
                )


@pytest.mark.parametrize(
    "rows,fp8",
    [
        (8, False),
        (16, False),
        (32, False),
        (64, False),
        (24, False),
        (8, True),
        (64, True),
    ],
)
def test_k2_matches_the_ops_math(rows, fp8):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx950.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx950.config import (
        EXPERTS,
        HIDDEN,
        MAX_TOKENS,
        MOE_HIDDEN,
        Q_K,
        TOP_K,
    )
    from sgl_amd_mono.muse_spark.gfx950.layout import scratch_bytes
    from sgl_amd_mono.muse_spark.gfx950.stages.post_attn import (
        k2_scratch_layout,
        sym_bytes,
    )

    torch.manual_seed(100 + rows)
    dev = torch.device("cuda")
    bf = torch.bfloat16
    o = torch.randn(rows, Q_K, device=dev).to(bf)
    g = torch.randn(rows, Q_K, device=dev).to(bf)
    stream = torch.randn(rows, HIDDEN, device=dev)
    router = (
        torch.randn(HIDDEN, EXPERTS, device=dev) * 0.02
    )  # the fp32 router [HIDDEN, EXPERTS]
    w_hi = router.T.to(bf).contiguous()
    w_lo = (
        (router.T - w_hi.float()).to(bf).contiguous()
    )  # pack_joint_router_pooler_weight's split
    w = {
        "w_o": (torch.randn(HIDDEN, Q_K, device=dev) * 0.02).to(bf),
        "alpha": torch.rand(HIDDEN, device=dev) * 0.5 + 0.5,
        "beta": torch.rand(HIDDEN, device=dev) * 0.5,
        "pre_w": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
        "w_hi": w_hi,
        "w_lo": w_lo,
        "bias": torch.randn(EXPERTS, device=dev) * 0.01,
        "w_pool": (torch.randn(MOE_HIDDEN, HIDDEN, device=dev) * 0.02).to(bf),
        "xn_w": (1.0 + 0.1 * torch.randn(MOE_HIDDEN, device=dev)).to(bf),
    }
    # the FP8 pooler: the kernel streams the packed codes + row scales; the reference
    # takes the values they encode (fp32)
    pool_ptr = w["w_pool"].data_ptr()
    if fp8:
        from sgl_amd_mono.muse_spark.fp8_quant import (
            pack_fp8_rows,
            pack_fp8_rows_pooler,
            unpack_fp8_rows,
        )

        packed = pack_fp8_rows_pooler(w["w_pool"])  # the kernel's lane-swizzled layout
        w["w_pool"] = unpack_fp8_rows(
            pack_fp8_rows(w["w_pool"]), MOE_HIDDEN, HIDDEN
        )  # the same codes, row-major
        pool_ptr = packed.data_ptr()
    ids = torch.full((MAX_TOKENS, TOP_K), -7, dtype=torch.int32, device=dev)
    scores = torch.full((MAX_TOKENS, TOP_K), float("nan"), device=dev)
    xn = torch.full((MAX_TOKENS, MOE_HIDDEN), float("nan"), device=dev, dtype=bf)
    scratch = torch.zeros(
        scratch_bytes(k2_scratch_layout()), dtype=torch.uint8, device=dev
    )
    peers = PeerBuffer(sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, dev)
    epoch = _epoch(dev)
    launch = build_k2(K2Build(tokens=rows, npes=1, fp8_pool=fp8))
    new_ref, ids_ref, scores_ref, xn_ref, biased = _reference(o, g, stream, w)
    got_stream = stream.clone()
    try:
        for step in range(2):  # the second step's tags differ from the first's
            got_stream.copy_(stream)
            launch(
                o.data_ptr(), g.data_ptr(), got_stream.data_ptr(), w["w_o"].data_ptr(),
                w["alpha"].data_ptr(), w["beta"].data_ptr(), w["pre_w"].data_ptr(),
                w["w_hi"].data_ptr(), w["w_lo"].data_ptr(), w["bias"].data_ptr(),
                pool_ptr, w["xn_w"].data_ptr(), ids.data_ptr(), scores.data_ptr(),
                xn.data_ptr(), scratch.data_ptr(), peers.local, peers.addresses.data_ptr(), 0,
                epoch.data_ptr(), LAYER,
                stream=torch.cuda.current_stream(),
            )  # fmt: skip
            epoch[0] += 1
        torch.cuda.synchronize()
        _close(got_stream, new_ref, 1 / 64, 1e-2, "stream")
        _close(xn[:rows], xn_ref, 1 / 32, 2e-2, "expert input")
        _check_routing(ids[:rows], scores[:rows], ids_ref, scores_ref, biased)
        assert (
            (ids[rows:] == -7).all()
            and scores[rows:].isnan().all()
            and xn[rows:].isnan().all()
        )
        assert epoch[list(guard.ERROR_WORDS)].eq(0).all(), "a wait failed"
    finally:
        peers.close()


@pytest.mark.parametrize("rows", [8, 16, 32, 64])
def test_every_width_compiles(rows):
    from sgl_amd_mono.muse_spark.gfx950.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

    # every width compiles, served (SGLANG_AMD_MONO_MUSE_WIDTHS, default 8) or not
    compile_only(build_k2(K2Build(tokens=rows)))


def test_a_fault_site_compiles():
    from sgl_amd_mono.muse_spark.gfx950.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

    compile_only(build_k2(K2Build(tokens=8, fault_site=1, fault_after=1)))
