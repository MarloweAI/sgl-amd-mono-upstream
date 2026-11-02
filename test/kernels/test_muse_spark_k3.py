"""The Muse Spark K3 launch on one gfx950 GPU (a TP / EP group of one) against
the math of the ROCm model's compact-A4 verify path it replaces: the experts'
input quantized to MXFP4 (aiter's ``_mxfp4_quant_op`` rule), gate / up on the E2M1
weights in fp32, bf16(silu(gate) up) re-quantized, the down GEMM rounded to bf16 a
(token, slot), the finalize (per-expert post norm with input scale, the
score-weighted slot sum, one rounding), the unpooler (one rounding) and the FFN
boundary, with the last-layer rule (the stream rounded to bf16 before the final
norm) as a case; picks outside the rank's experts are skipped and one expert is
picked by every token, so a 64-row step fills four column tiles of it. Also that
it compiles for every served width and a fault site.

The 8-rank form (not written yet) runs under ``torchrun --nproc-per-node 8``: each
rank builds ``K3Build(tokens, npes=8)``, allocates ``PeerBuffer(k3_sym_layout(8)
bytes + FLAG_BYTES, gloo group, rank, 8, dev)``, passes ``local_start = 32 rank``
with the same ids / scores on every rank and its own Quark-layout experts, and the
reference sums the ranks' finalize partials in rank order in fp32 before one bf16
rounding (``all_reduce`` of the bf16 partials as fp32) and unpools the whole row;
every rank's stream / ``attn_out`` must match and its error words stay clean."""

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

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
EXPERT_ROWS = 8  # the test's expert table: a prefix of the rank's 32; picks past it are other ranks'
LOCAL_START = 64
LAYER = 7


def _epoch(device, deadline_s=20.0):
    from sgl_amd_mono.common import guard

    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return e


def _fp4_weight(experts, rows, k, device, gen):
    """Random packed E2M1 [experts, rows, k / 2] (the low nibble the even k) with
    E8M0 scales [experts, rows, k / 32]: the Quark layout; and the decoded bf16
    values [experts, rows, k]."""
    packed = torch.randint(0, 256, (experts, rows, k // 2), device=device, dtype=torch.uint8, generator=gen)  # fmt: skip
    scales = torch.randint(118, 128, (experts, rows, k // 32), device=device, dtype=torch.uint8, generator=gen)  # fmt: skip
    lo, hi = packed & 0xF, packed >> 4
    codes = torch.stack([lo, hi], dim=-1).reshape(experts, rows, k)
    mag = E2M1.to(device)[(codes & 7).long()]
    val = torch.where(codes & 8 != 0, -mag, mag)
    scale = torch.ldexp(torch.ones(1, device=device), scales.to(torch.int32) - 127)
    dec = (val.reshape(experts, rows, k // 32, 32) * scale[..., None]).reshape(
        experts, rows, k
    )
    return packed, scales, dec.to(torch.bfloat16)


def _mx_dequant(x):
    """x (fp32 [..., N]) through aiter's Triton ``_mxfp4_quant_op`` (32-value
    blocks, the E8M0 scale from the max rounded at mantissa bit 21, E2M1 half to
    even, saturating) and back: the values the scaled FP4 MFMA multiplies."""
    shape = x.shape
    xb = x.float().reshape(*shape[:-1], -1, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    bits = (amax.view(torch.int32) + 0x200000) & -0x800000
    unb = (((bits >> 23) & 0xFF) - 129).clamp(min=-127)
    qx = xb * torch.ldexp(torch.ones_like(amax), -unb)
    a = qx.abs()
    ab = a.view(torch.int32)
    den = ((a + 4194304.0).view(torch.int32) - (149 << 23)) & 0xFF
    nrm = ((ab + ((-126 << 23) + 0x1FFFFF) + ((ab >> 22) & 1)) >> 22) & 0xFF
    code = torch.where(
        a >= 6.0, torch.full_like(den, 7), torch.where(a < 1.0, den, nrm)
    )
    mag = E2M1.to(x.device)[code.long()]
    val = torch.where(qx < 0, -mag, mag)
    return (val * torch.ldexp(torch.ones_like(amax), unb)).reshape(shape)


def _bf(t):
    return t.to(torch.bfloat16).float()


def _reference(xin, ids, scores, local_start, dec13, dec2, w, stream, last):
    """compact_a4_experts + the finalize (one rank: the all-reduce is the identity)
    + unpool_gemm_norm_residual with ``pre_bf16 = last``."""
    from sgl_amd_mono.muse_spark.gfx950.config import (
        INTER,
        LOCAL_EXPERTS,
        POST_EPS,
        RMS_EPS,
        TOP_K,
    )

    rows = xin.shape[0]
    out = torch.zeros(rows, dec2.shape[1], device=xin.device)
    for t in range(rows):
        for s in range(TOP_K):
            e = int(ids[t, s]) - local_start
            if not 0 <= e < LOCAL_EXPERTS:
                continue
            gu = _mx_dequant(xin[t].float()) @ dec13[e].float().T
            gate, up = gu[:INTER], gu[INTER:]
            mid = _mx_dequant(_bf(gate * torch.sigmoid(gate) * up))
            y = _bf(mid @ dec2[e].float().T)
            v = _bf(y * w["pen_w"].float())
            sc = float(scores[t, s]) / torch.sqrt(v.pow(2).mean() + POST_EPS)
            out[t] += sc * v
    part = _bf(out)
    branch = _bf(part @ w["w_unpool"].float().T)
    post_input = _bf(branch * w["pfw"].float())
    nb = _bf(
        post_input * torch.rsqrt(post_input.pow(2).mean(-1, keepdim=True) + POST_EPS)
    )
    new = w["alpha"] * stream + w["beta"] * nb
    pre = _bf(new) if last else new
    attn_in = (pre * torch.rsqrt(pre.pow(2).mean(-1, keepdim=True) + RMS_EPS) * w["npw"].float()).to(torch.bfloat16)  # fmt: skip
    return new, attn_in


def _close(got, ref, rtol, atol, what):
    got, ref = got.float(), ref.float()
    bad = ((got - ref).abs() > rtol * ref.abs() + atol).float().mean().item()
    assert bad == 0.0, f"{what}: {bad:.2%} of the elements off by more than {rtol} relative + {atol}"  # fmt: skip
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    assert cos.min().item() >= 0.996, (
        f"{what}: a row's cosine is {cos.min().item():.4f}"
    )


def _picks(rows, device, gen):
    """ids int32 [rows, TOP_K]: slot 0 is the rank's expert 0 for every token (the
    widest column set), the rest a random choice of the other table experts and of
    ids outside the rank, one of them past the rank's table."""
    from sgl_amd_mono.muse_spark.gfx950.config import LOCAL_EXPERTS, TOP_K

    ids = torch.empty(rows, TOP_K, dtype=torch.int64, device=device)
    for t in range(rows):
        others = (
            torch.randperm(EXPERT_ROWS + 7, device=device, generator=gen)[: TOP_K - 1]
            + 1
        )
        ids[t, 0] = 0
        ids[t, 1:] = others
    ids[0, 1] = LOCAL_EXPERTS + 3  # past the rank's table
    # a pick past the table (the rank's remaining 24 experts) belongs to another rank
    ids = torch.where(ids >= EXPERT_ROWS, ids + LOCAL_EXPERTS, ids)
    return (ids + LOCAL_START).to(torch.int32)


@pytest.mark.parametrize(
    "rows,last,fp8",
    [
        (8, False, False),
        (16, True, False),
        (32, False, False),
        (64, False, False),
        (64, True, False),
        (8, False, True),
        (64, True, True),
    ],
)
def test_k3_matches_the_compact_a4_math(rows, last, fp8):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx950.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx950.config import (
        HIDDEN,
        INTER,
        MAX_TOKENS,
        MOE_HIDDEN,
        TOP_K,
    )
    from sgl_amd_mono.muse_spark.gfx950.layout import scratch_bytes
    from sgl_amd_mono.muse_spark.gfx950.stages.ffn import (
        k3_scratch_layout,
        k3_sym_layout,
    )

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(300 + rows)
    torch.manual_seed(300 + rows)
    bf = torch.bfloat16
    w13, s13, dec13 = _fp4_weight(EXPERT_ROWS, 2 * INTER, MOE_HIDDEN, dev, gen)
    w2, s2, dec2 = _fp4_weight(EXPERT_ROWS, MOE_HIDDEN, INTER, dev, gen)
    xin = (torch.randn(rows, MOE_HIDDEN, device=dev) * 0.5).to(bf)
    ids = _picks(rows, dev, gen)
    scores = torch.rand(rows, TOP_K, device=dev) * 0.3
    stream = torch.randn(rows, HIDDEN, device=dev)
    w = {
        "pen_w": (1.0 + 0.1 * torch.randn(MOE_HIDDEN, device=dev)).to(bf),
        "pfw": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
        "alpha": torch.rand(HIDDEN, device=dev) * 0.5 + 0.5,
        "beta": torch.rand(HIDDEN, device=dev) * 0.5,
        "npw": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
        "w_unpool": (torch.randn(HIDDEN, MOE_HIDDEN, device=dev) * 0.02).to(bf),
    }
    unpool_ptr = w["w_unpool"].data_ptr()
    if (
        fp8
    ):  # the kernel streams the packed codes + row scales; the reference their values
        from sgl_amd_mono.muse_spark.fp8_quant import pack_fp8_rows, unpack_fp8_rows

        packed = pack_fp8_rows(w["w_unpool"])
        w["w_unpool"] = unpack_fp8_rows(packed, HIDDEN, MOE_HIDDEN)
        unpool_ptr = packed.data_ptr()
    attn_out = torch.full((MAX_TOKENS, HIDDEN), float("nan"), device=dev, dtype=bf)
    scratch = torch.zeros(
        scratch_bytes(k3_scratch_layout()), dtype=torch.uint8, device=dev
    )
    peers = PeerBuffer(
        scratch_bytes(k3_sym_layout(1)) + guard.FLAG_BYTES, None, 0, 1, dev
    )
    peers.bytes.zero_()
    epoch = _epoch(dev)
    launch = build_k3(
        K3Build(tokens=rows, npes=1, fp8_unpool=fp8)
    )  # the rule from the launch's ``last``
    new_ref, attn_ref = _reference(
        xin, ids, scores, LOCAL_START, dec13, dec2, w, stream, last
    )
    got = stream.clone()
    try:
        for step in range(2):  # the second step's tags differ from the first's
            got.copy_(stream)
            launch(
                xin.data_ptr(), ids.data_ptr(), scores.data_ptr(), LOCAL_START,
                w13.data_ptr(), s13.data_ptr(), w2.data_ptr(), s2.data_ptr(),
                w["pen_w"].data_ptr(), got.data_ptr(), w["pfw"].data_ptr(),
                w["alpha"].data_ptr(), w["beta"].data_ptr(), w["npw"].data_ptr(),
                unpool_ptr, attn_out.data_ptr(), scratch.data_ptr(),
                peers.local, peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, int(last),
                stream=torch.cuda.current_stream(),
            )  # fmt: skip
            epoch[0] += 1
        torch.cuda.synchronize()
        assert (epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0).all(), (
            "a wait failed"
        )
        _close(got, new_ref, 1 / 32, 2e-2, "stream")
        _close(attn_out[:rows], attn_ref, 1 / 32, 2e-2, "attention input")
        assert attn_out[rows:].isnan().all(), "rows past the step were written"
    finally:
        peers.close()


@pytest.mark.parametrize("rows,last", [(8, False), (16, True), (64, False)])
def test_k3_tpx_matches_the_compact_a4_math(rows, last):
    """TP experts on a group of one (the slices are the whole experts): every pick is
    worked, the plan over the global experts, the mids / values / sums of squares
    through the peer regions, the finalize without an all-reduce."""
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx950.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx950.config import (
        HIDDEN,
        INTER,
        MAX_TOKENS,
        MOE_HIDDEN,
        TOP_K,
    )
    from sgl_amd_mono.muse_spark.gfx950.layout import scratch_bytes
    from sgl_amd_mono.muse_spark.gfx950.stages.ffn import (
        k3_scratch_layout,
        k3_sym_layout,
    )

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(700 + rows)
    torch.manual_seed(700 + rows)
    bf = torch.bfloat16
    w13, s13, dec13 = _fp4_weight(EXPERT_ROWS, 2 * INTER, MOE_HIDDEN, dev, gen)
    w2, s2, dec2 = _fp4_weight(EXPERT_ROWS, MOE_HIDDEN, INTER, dev, gen)
    xin = (torch.randn(rows, MOE_HIDDEN, device=dev) * 0.5).to(bf)
    # slot 0: expert 0 for every token (its widest column set); the others distinct table experts
    ids = torch.empty(rows, TOP_K, dtype=torch.int32, device=dev)
    for t in range(rows):
        ids[t, 0] = 0
        ids[t, 1:] = (
            torch.randperm(EXPERT_ROWS - 1, device=dev, generator=gen)[: TOP_K - 1] + 1
        ).to(torch.int32)
    scores = torch.rand(rows, TOP_K, device=dev) * 0.3
    stream = torch.randn(rows, HIDDEN, device=dev)
    w = {
        "pen_w": (1.0 + 0.1 * torch.randn(MOE_HIDDEN, device=dev)).to(bf),
        "pfw": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
        "alpha": torch.rand(HIDDEN, device=dev) * 0.5 + 0.5,
        "beta": torch.rand(HIDDEN, device=dev) * 0.5,
        "npw": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
        "w_unpool": (torch.randn(HIDDEN, MOE_HIDDEN, device=dev) * 0.02).to(bf),
    }
    attn_out = torch.full((MAX_TOKENS, HIDDEN), float("nan"), device=dev, dtype=bf)
    scratch = torch.zeros(
        scratch_bytes(k3_scratch_layout()), dtype=torch.uint8, device=dev
    )
    peers = PeerBuffer(
        scratch_bytes(k3_sym_layout(1)) + guard.FLAG_BYTES, None, 0, 1, dev
    )
    peers.bytes.zero_()
    epoch = _epoch(dev)
    launch = build_k3(K3Build(tokens=rows, npes=1, tpx=True))
    new_ref, attn_ref = _reference(xin, ids, scores, 0, dec13, dec2, w, stream, last)
    got = stream.clone()
    try:
        for step in range(2):
            got.copy_(stream)
            launch(
                xin.data_ptr(), ids.data_ptr(), scores.data_ptr(), 0,
                w13.data_ptr(), s13.data_ptr(), w2.data_ptr(), s2.data_ptr(),
                w["pen_w"].data_ptr(), got.data_ptr(), w["pfw"].data_ptr(),
                w["alpha"].data_ptr(), w["beta"].data_ptr(), w["npw"].data_ptr(),
                w["w_unpool"].data_ptr(), attn_out.data_ptr(), scratch.data_ptr(),
                peers.local, peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, int(last),
                stream=torch.cuda.current_stream(),
            )  # fmt: skip
            epoch[0] += 1
        torch.cuda.synchronize()
        assert (epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0).all(), (
            "a wait failed"
        )
        _close(got, new_ref, 1 / 32, 2e-2, "stream")
        _close(attn_out[:rows], attn_ref, 1 / 32, 2e-2, "attention input")
        assert attn_out[rows:].isnan().all(), "rows past the step were written"
    finally:
        peers.close()


@pytest.mark.parametrize("rows", [8, 16, 64])
def test_tpx_compiles_at_tp8(rows):
    from sgl_amd_mono.muse_spark.gfx950.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

    compile_only(build_k3(K3Build(tokens=rows, npes=8, tpx=True)))


def test_the_last_rule_compiles_in():
    """``K3Build(last=True)`` compiles the last-layer rule in (its own symbol)."""
    from sgl_amd_mono.muse_spark.gfx950.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

    compile_only(build_k3(K3Build(tokens=8, npes=1, last=True)))


@pytest.mark.parametrize("rows", [8, 16, 32, 64])
def test_every_width_compiles(rows):
    from sgl_amd_mono.muse_spark.gfx950.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

    compile_only(build_k3(K3Build(tokens=rows)))


def test_a_fault_site_compiles():
    from sgl_amd_mono.muse_spark.gfx950.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

    compile_only(build_k3(K3Build(tokens=8, fault_site=1, fault_after=1)))
