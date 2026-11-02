"""The DFlash2 draft layer's attention half around the attention, D-pre and D-post,
on one gfx950 GPU (a TP group of one) against the SGLang draft layer's math in
torch: D-pre's fused add + input RMSNorm, the kernel projection (both sides'
coefficients), conv0, the qkv projection, the per-head q / k RMSNorm + table neox
RoPE (``table_qk_norm_rope_``), q out and k / v into the fp8 cache at the rows'
slots; D-post's o projection, the (one-rank) all-reduce and conv1 with side 1."""

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

LAYER = 4
SLOTS = 4096
MAX_POS = 8192


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


def _rms(x, w, eps=1e-5):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(
        torch.bfloat16
    )


def _pre_reference(x, res, first, w):
    """(res', q [S, 1024], k [S, 128], v [S, 128], coef [S, 2048]) as SGLang's modules
    compute them (bf16 roundings at each module boundary)."""
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D

    bf = torch.bfloat16
    S = x.shape[0]
    res2 = x if first else (res.float() + x.float()).to(bf)
    h = _rms(res2, w["w_in"])
    coef = (
        h @ w["w_kp"].T
    ).float()  # the kernel projection (bf16 GEMM output, as fp32 for the kernel's compare)
    coef_b = coef.to(bf)
    delta = coef_b.reshape(S, 2, D.TAPS, D.GROUPS)
    hc = _grouped_conv(h, delta[:, 0], w["base"][0], D.BLOCK, D.GROUPS, D.GROUP, D.TAPS)
    qkv = hc @ w["w_qkv"].T  # [S, 1280] bf16
    q, k, v = qkv[:, :1024], qkv[:, 1024:1152], qkv[:, 1152:]

    # table_qk_norm_rope_: per head RMSNorm (weight), bf16, neox rotation from the table
    def norm_rope(t, wn):
        heads = t.reshape(S, -1, 128).float()
        inv = torch.rsqrt(heads.pow(2).mean(-1, keepdim=True) + 1e-5)
        n = (heads * inv * wn.float()).to(bf).float()
        x1, x2 = n[..., :64], n[..., 64:]
        cs = w["cos_sin"][w["positions"].long()].float()  # [S, 128]
        cos, sin = cs[:, None, :64], cs[:, None, 64:]
        return (
            torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
            .to(bf)
            .reshape(S, -1)
        )

    return res2, norm_rope(q, w["q_norm"]), norm_rope(k, w["k_norm"]), v, coef


def _post_reference(o, w, coef):
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D

    bf = torch.bfloat16
    S = o.shape[0]
    d = o @ w["w_o"].T  # one rank: the partial is the sum, bf16
    delta = coef.to(bf).reshape(S, 2, D.TAPS, D.GROUPS)
    return _grouped_conv(
        d, delta[:, 1], w["base"][1], D.BLOCK, D.GROUPS, D.GROUP, D.TAPS
    )


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


@pytest.mark.parametrize("rows,first", [(8, True), (8, False), (16, False)])
def test_dpre_matches_the_attention_input_path(rows, first):
    from sgl_amd_mono.muse_spark.gfx950 import draft_attn_half as DH
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D
    from sgl_amd_mono.muse_spark.gfx950.layout import scratch_bytes

    dev = torch.device("cuda")
    torch.manual_seed(500 + rows + int(first))
    bf = torch.bfloat16
    x = (torch.randn(rows, D.H, device=dev) * 0.5).to(bf)
    res = torch.randn(rows, D.H, device=dev).to(bf)
    cos_sin = torch.empty(MAX_POS, 128, device=dev)
    ang = (
        torch.arange(MAX_POS, device=dev)[:, None]
        * (10000.0 ** (-torch.arange(64, device=dev) / 64))[None, :]
    )
    cos_sin[:, :64], cos_sin[:, 64:] = torch.cos(ang), torch.sin(ang)
    w = {
        "w_in": (1.0 + 0.1 * torch.randn(D.H, device=dev)).to(bf),
        "w_kp": (torch.randn(D.COEF_ROWS, D.H, device=dev) * 0.01).to(bf),
        "base": (torch.randn(2, D.TAPS, D.H, device=dev) * 0.05 + 1).to(bf),
        "w_qkv": (torch.randn(DH.QKV_ROWS, D.H, device=dev) * 0.02).to(bf),
        "q_norm": (1.0 + 0.1 * torch.randn(128, device=dev)).to(bf),
        "k_norm": (1.0 + 0.1 * torch.randn(128, device=dev)).to(bf),
        "cos_sin": cos_sin,  # the served table is fp32
        "positions": torch.randint(0, MAX_POS, (rows,), device=dev, dtype=torch.int32),
    }
    slots = torch.randperm(SLOTS, device=dev)[:rows].to(torch.int32)
    k_cache = torch.zeros(SLOTS, 1, 128, device=dev, dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros(SLOTS, 1, 128, device=dev, dtype=torch.float8_e4m3fn)
    res_out = torch.full((rows, D.H), float("nan"), device=dev, dtype=bf)
    q = torch.full((rows, 1024), float("nan"), device=dev, dtype=bf)
    coef = torch.full((rows, D.COEF_ROWS), float("nan"), device=dev)
    scratch = torch.zeros(
        scratch_bytes(DH.dp_scratch_layout()), dtype=torch.uint8, device=dev
    )
    epoch = _epoch(dev)
    launch = DH.build_dpre(DH.DPreBuild(tokens=rows))
    for step in range(2):
        launch(
            x.data_ptr(), res.data_ptr(), int(first), w["w_in"].data_ptr(), w["w_kp"].data_ptr(), w["base"].data_ptr(),
            w["w_qkv"].data_ptr(), w["q_norm"].data_ptr(), w["k_norm"].data_ptr(), w["cos_sin"].data_ptr(),
            w["positions"].data_ptr(), 0, slots.data_ptr(), 0, k_cache.data_ptr(), v_cache.data_ptr(),
            res_out.data_ptr(), q.data_ptr(), coef.data_ptr(), scratch.data_ptr(), epoch.data_ptr(), LAYER,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        epoch[0] += 1
    torch.cuda.synchronize()
    res_ref, q_ref, k_ref, v_ref, coef_ref = _pre_reference(x, res, first, w)
    assert (
        not res_out.isnan().any() and not q.isnan().any() and not coef.isnan().any()
    ), "NaN in D-pre's outputs"
    assert torch.equal(res_out, res_ref), "the residual is not bf16(res + x) / x"
    _close(
        coef,
        coef_ref,
        1 / 64,
        1e-3,
        "the coefficients vs the fp32 projection",
        bad_max=0.002,
    )
    _close(
        q,
        q_ref,
        1 / 16,
        3e-2,
        "q vs SGLang's qkv + norm + rope",
        bad_max=0.02,
        cos_min=0.999,
    )
    k_got = k_cache[slots.long(), 0].float()
    v_got = v_cache[slots.long(), 0].float()
    # fp8 e4m3 rounding: about 6% relative
    _close(
        k_got,
        k_ref.float(),
        1 / 8,
        5e-2,
        "k in the cache (fp8)",
        bad_max=0.02,
        cos_min=0.995,
    )
    _close(
        v_got,
        v_ref.float(),
        1 / 8,
        5e-2,
        "v in the cache (fp8)",
        bad_max=0.02,
        cos_min=0.995,
    )


@pytest.mark.parametrize("rows", [8, 16])
def test_dpost_matches_the_attention_output_path(rows):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx950 import draft_attn_half as DH
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D
    from sgl_amd_mono.muse_spark.gfx950.layout import scratch_bytes

    dev = torch.device("cuda")
    torch.manual_seed(600 + rows)
    bf = torch.bfloat16
    o = (torch.randn(rows, 1024, device=dev) * 0.5).to(bf)
    w = {
        "w_o": (torch.randn(D.H, 1024, device=dev) * 0.03).to(bf),
        "base": (torch.randn(2, D.TAPS, D.H, device=dev) * 0.05 + 1).to(bf),
    }
    coef = torch.randn(rows, D.COEF_ROWS, device=dev) * 0.1
    attn_out = torch.full((rows, D.H), float("nan"), device=dev, dtype=bf)
    scratch = torch.zeros(
        scratch_bytes(DH.do_scratch_layout()), dtype=torch.uint8, device=dev
    )
    peers = PeerBuffer(DH.do_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, dev)
    peers.bytes.zero_()
    epoch = _epoch(dev)
    launch = DH.build_dpost(DH.DPostBuild(tokens=rows, npes=1))
    try:
        for step in range(2):
            launch(
                o.data_ptr(), w["w_o"].data_ptr(), coef.data_ptr(), w["base"].data_ptr(), attn_out.data_ptr(),
                scratch.data_ptr(), peers.local, peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER,
                stream=torch.cuda.current_stream(),
            )  # fmt: skip
            epoch[0] += 1
        torch.cuda.synchronize()
        ref = _post_reference(o, w, coef)
        assert not attn_out.isnan().any(), "NaN in D-post's output"
        _close(
            attn_out,
            ref,
            1 / 16,
            2e-2,
            "attn_out vs SGLang's o projection + conv1",
            bad_max=0.02,
            cos_min=0.999,
        )
    finally:
        peers.close()


@pytest.mark.parametrize("rows", [8, 16])
def test_widths_compile(rows):
    from sgl_amd_mono.muse_spark.gfx950 import draft_attn_half as DH
    from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

    compile_only(DH.build_dpre(DH.DPreBuild(tokens=rows)))
    compile_only(DH.build_dpost(DH.DPostBuild(tokens=rows, npes=8)))
