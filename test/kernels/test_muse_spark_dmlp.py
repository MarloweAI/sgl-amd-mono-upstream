"""The DFlash2 draft's D-MLP launch on one gfx950 GPU (a TP group of one) against
the SGLang draft layer's MLP half in torch (its bf16 roundings: the fused add +
RMSNorm, the kernel projection, the grouped two-tap convolution within blocks of
8, gate / up, silu(g) u, down, the finish convolution)."""

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

LAYER = 3


def _epoch(device, deadline_s=30.0):
    from sgl_amd_mono.common import guard

    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return e


def _grouped_conv(x, delta, base, block, groups, group, taps):
    """``sglang.srt.models.dflash._grouped_conv`` (bf16 tensors)."""
    blocks = x.unflatten(-1, (groups, group))
    coefficients = base.view(1, taps, groups, group) + delta.unsqueeze(-1)
    out = coefficients[:, 0] * blocks
    position = torch.arange(x.shape[0], device=x.device) & (block - 1)
    for tap in range(1, taps):
        shifted = torch.nn.functional.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        shifted = torch.where((position >= tap).view(-1, 1, 1), shifted, 0)
        out = out + coefficients[:, tap] * shifted
    return out.flatten(-2)


def _conv32(x, delta, base, rows):
    """The grouped two-tap convolution in fp32 (the kernel's arithmetic): ``x`` [S, H]
    fp32, ``delta`` [S, TAPS, GROUPS] fp32, ``base`` [TAPS, H] bf16."""
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D

    xb = x.reshape(rows, D.GROUPS, D.GROUP)
    c = base.float().reshape(1, D.TAPS, D.GROUPS, D.GROUP) + delta.reshape(
        rows, D.TAPS, D.GROUPS, 1
    )
    pos = torch.arange(rows, device=x.device) % D.BLOCK
    prev = torch.cat([torch.zeros_like(xb[:1]), xb[:-1]], 0) * (pos >= 1).view(-1, 1, 1)
    return (c[:, 0] * xb + c[:, 1] * prev).reshape(rows, D.H)


def _reference(res, attn, w, eps=1e-5):
    """(res', out_sglang, out_fp32) of the draft layer's MLP half: ``out_sglang`` with
    the bf16 roundings of SGLang's modules (its grouped convolution rounds every
    op), ``out_fp32`` with the kernel's rounding points only (bf16 at h, hc, g, u,
    a, the rank partial and out)."""
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D

    bf = torch.bfloat16
    rows = res.shape[0]
    res2 = (res.float() + attn.float()).to(bf)  # the fused add's residual
    var = res2.float().pow(2).mean(-1, keepdim=True)
    h = (res2.float() * torch.rsqrt(var + eps) * w["w_post"].float()).to(bf)
    # SGLang's chain
    coef = (h @ w["w_kp"].T).reshape(-1, 2, D.TAPS, D.GROUPS)  # bf16 matmul
    hc = _grouped_conv(h, coef[:, 0], w["base"][0], D.BLOCK, D.GROUPS, D.GROUP, D.TAPS)
    gu = hc @ w["w_gu"].T
    g, u = gu[:, : D.I], gu[:, D.I :]
    a = (torch.nn.functional.silu(g.float()).to(bf).float() * u.float()).to(bf)
    d = a @ w["w_down"].T  # one rank: its partial is the sum
    out_sglang = _grouped_conv(
        d, coef[:, 1], w["base"][1], D.BLOCK, D.GROUPS, D.GROUP, D.TAPS
    )
    # the kernel's chain
    coef32 = (h.float() @ w["w_kp"].float().T).reshape(-1, 2, D.TAPS, D.GROUPS)
    hc32 = _conv32(h.float(), coef32[:, 0], w["base"][0], rows).to(bf)
    gu32 = hc32.float() @ w["w_gu"].float().T
    g32, u32 = gu32[:, : D.I].to(bf).float(), gu32[:, D.I :].to(bf).float()
    a32 = (torch.nn.functional.silu(g32).to(bf).float() * u32).to(bf)
    d32 = (
        (a32.float() @ w["w_down"].float().T).to(bf).float()
    )  # the rank partial, a bf16 pair
    out_fp32 = _conv32(d32, coef32[:, 1], w["base"][1], rows).to(bf)
    return res2, out_sglang, out_fp32


def _close(got, ref, rtol, atol, what, bad_max=0.001, cos_min=0.9999):
    got, ref = got.float(), ref.float()
    bad = ((got - ref).abs() > rtol * ref.abs() + atol).float().mean().item()
    assert bad < bad_max, (
        f"{what}: {bad:.2%} of the elements off by more than {rtol} relative + {atol}"
    )
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    assert cos.min().item() >= cos_min, (
        f"{what}: a row's cosine is {cos.min().item():.5f}"
    )


@pytest.mark.parametrize("rows", [8, 16, 64])
def test_dmlp_matches_the_draft_mlp(rows):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D
    from sgl_amd_mono.muse_spark.gfx950.layout import scratch_bytes

    dev = torch.device("cuda")
    torch.manual_seed(400 + rows)
    bf = torch.bfloat16
    res = torch.randn(rows, D.H, device=dev).to(bf)
    attn = (torch.randn(rows, D.H, device=dev) * 0.5).to(bf)
    w = {
        "w_post": (1.0 + 0.1 * torch.randn(D.H, device=dev)).to(bf),
        "w_kp": (torch.randn(D.COEF_ROWS, D.H, device=dev) * 0.01).to(bf),
        "base": torch.zeros(2, D.TAPS, D.H, device=dev, dtype=bf),
        "w_gu": (torch.randn(D.GU_ROWS, D.H, device=dev) * 0.02).to(bf),
        "w_down": (torch.randn(D.H, D.I, device=dev) * 0.02).to(bf),
    }
    w["base"][:, 0] = 1.0
    w["base"] += (0.05 * torch.randn(2, D.TAPS, D.H, device=dev)).to(bf)
    res_ref, out_sglang, out_fp32 = _reference(res, attn, w)
    res_out = torch.full((rows, D.H), float("nan"), device=dev, dtype=bf)
    out = torch.full((rows, D.H), float("nan"), device=dev, dtype=bf)
    scratch = torch.zeros(
        scratch_bytes(D.dm_scratch_layout()), dtype=torch.uint8, device=dev
    )
    peers = PeerBuffer(D.dm_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, dev)
    peers.bytes.zero_()
    epoch = _epoch(dev)
    import os

    stages = int(
        os.environ.get("DMLP_STAGES", "8")
    )  # a bisection: run the first stages only
    launch = D.build_dmlp(D.DMLPBuild(tokens=rows, npes=1, stages=stages))
    try:
        for step in range(2):  # the second step's tags differ from the first's
            launch(
                res.data_ptr(), attn.data_ptr(), w["w_post"].data_ptr(), w["w_kp"].data_ptr(), w["base"].data_ptr(),
                w["w_gu"].data_ptr(), w["w_down"].data_ptr(), res_out.data_ptr(), out.data_ptr(), scratch.data_ptr(),
                peers.local, peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, stream=torch.cuda.current_stream(),
            )  # fmt: skip
            epoch[0] += 1
        torch.cuda.synchronize()
        assert (epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0).all(), (
            "a wait failed"
        )
        if stages < 8:
            return
        assert not res_out.isnan().any() and not out.isnan().any(), "NaN in the outputs"
        assert torch.equal(res_out, res_ref), "the residual is not bf16(res + attn_out)"
        # the kernel's own rounding points in torch: elementwise (a few ill-conditioned
        # near-cancellations differ by a bf16 ulp of their terms), rows at cosine 0.9999
        _close(out, out_fp32, 1 / 32, 2e-2, "out vs the kernel's arithmetic in torch")
        # SGLang's grouped convolution rounds every op to bf16: rows at cosine 0.999
        _close(
            out,
            out_sglang,
            1 / 8,
            1e-1,
            "out vs SGLang's bf16 chain",
            bad_max=0.02,
            cos_min=0.999,
        )
    finally:
        peers.close()


@pytest.mark.parametrize("rows", [8, 16, 32, 64])
def test_widths_compile(rows):
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D
    from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

    compile_only(D.build_dmlp(D.DMLPBuild(tokens=rows, npes=8)))
