"""Qwen3.5 block-FP8 verify kernels on one gfx950 GPU against SGLang's own kernels.

Random weights in SGLang's post-load layouts at the TP4 rank shapes (aiter's
(16, 16) preshuffle for block FP8). The 32-row split-K GEMV is compared with
aiter's block-scale GEMM. The verify GDN chain is compared with SGLang's verify
path (``causal_conv1d_update`` with per-token windows, the delta rule with
``disable_state_update``, the gated RMSNorm) at 1, 3 and 8 requests x 4 draft
tokens, one of them a pad request: the core, the conv window, every per-token
conv and SSM snapshot, the untouched committed SSM state and the FP8 quant of
the core. Then each request accepts a different number of draft tokens, both
sides commit that snapshot as SGLang's accept step does, and the next decode
step must agree.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a gfx950 GPU"
)

from sgl_amd_mono.qwen35_a17b.fp8.gfx950 import verify  # noqa: E402
from sgl_amd_mono.qwen35_a17b.fp8.gfx950.shapes import TP4  # noqa: E402

FP8 = torch.float8_e4m3fn
T = 4
SLOTS = 24


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _block_fp8(n, k, gen, device):
    from aiter.ops.shuffle import shuffle_weight

    w = torch.randn(n, k, generator=gen, device=device) * 0.02
    blocks = w.view(n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-8) / 448.0
    q = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(FP8).view(n, k)
    shuffled = shuffle_weight(q, (16, 16))
    shuffled.is_shuffled = True
    return shuffled, scale.float().contiguous()


def _norm(device):
    return SimpleNamespace(
        weight=torch.zeros(TP4.hidden, dtype=torch.bfloat16, device=device),
        variance_epsilon=1e-6,
    )


def _activations(rows, gen, device):
    from sgl_amd_mono.qwen35_a17b.fp8.gfx950.norm import add_norm_quant

    hidden = torch.randn(rows, TP4.hidden, generator=gen, device=device)
    y, _, yq = add_norm_quant(hidden.to(torch.bfloat16), None, _norm(device))
    return y, yq


@pytest.fixture(scope="module")
def device():
    from sgl_amd_mono.qwen35_a17b.fp8.gfx950.gemv import _allow_block_fp8_linear

    _allow_block_fp8_linear()
    return torch.device("cuda")


@pytest.mark.parametrize("rows", [4, 12, 17, 24, 32])
def test_gemv_32_rows_matches_aiter_block_scale_gemm(device, rows):
    from sgl_amd_mono.qwen35_a17b.fp8.gfx950.gemv import (
        block_fp8_gemv_q,
        block_fp8_gemv_wg,
        wg_fits,
    )

    gen = torch.Generator(device=device).manual_seed(rows)
    for n, k in ((TP4.qkvz_n, TP4.hidden), (TP4.hidden, TP4.o_k)):
        w, s = _block_fp8(n, k, gen, device)
        x = torch.randn(rows, k, generator=gen, device=device).to(torch.bfloat16)
        if k == TP4.hidden:
            x, xq = _activations(rows, gen, device)
        else:
            from sgl_amd_mono.qwen35_a17b.fp8.gfx950.norm import gate_quant

            x, xq = gate_quant(x, torch.zeros_like(x))
        assert wg_fits(rows, k, w)
        got = block_fp8_gemv_wg(x, xq, w, s)
        want = block_fp8_gemv_q(x, xq, w, s)
        assert got.shape == want.shape == (rows, n)
        assert _cos(got, want) > 0.99999, (n, k)
        rel = (got.float() - want.float()).norm() / want.float().norm()
        assert rel < 5e-3, (n, k, rel.item())


def _gdn(gen, device):
    hv, head = TP4.gdn_v_heads, TP4.gdn_head
    w, s = _block_fp8(TP4.qkvz_n, TP4.hidden, gen, device)
    return SimpleNamespace(
        in_proj_qkvz=SimpleNamespace(weight=w, weight_scale_inv=s),
        in_proj_ba=SimpleNamespace(
            weight=(
                torch.randn(TP4.ba_n, TP4.hidden, generator=gen, device=device) * 0.02
            ).to(torch.bfloat16),
            bias=None,
        ),
        conv1d=SimpleNamespace(
            weight=(
                torch.randn(TP4.conv_dim, 1, 4, generator=gen, device=device) * 0.5
            ).to(torch.bfloat16),
            bias=None,
        ),
        A_log=torch.rand(hv, generator=gen, device=device) * 2 - 1,
        dt_bias=torch.rand(hv, generator=gen, device=device),
        norm=SimpleNamespace(
            weight=(1 + 0.1 * torch.randn(head, generator=gen, device=device)).to(
                torch.bfloat16
            ),
            eps=1e-6,
        ),
        out_proj=None,
    )


def _cache(gen, device, transposed_conv):
    hv, head, c = TP4.gdn_v_heads, TP4.gdn_head, TP4.conv_dim
    if transposed_conv:
        conv = torch.randn(SLOTS, 3, c, generator=gen, device=device)
        conv = conv.to(torch.bfloat16).transpose(1, 2)
    else:
        conv = torch.randn(SLOTS, c, 3, generator=gen, device=device)
        conv = conv.to(torch.bfloat16)
    ssm = torch.randn(SLOTS, hv, head, head, generator=gen, device=device) * 0.1
    return SimpleNamespace(
        conv=[conv],
        temporal=ssm,
        intermediate_conv_window=[
            torch.zeros(SLOTS + 1, T, c, 3, dtype=torch.bfloat16, device=device)
        ],
        intermediate_ssm=torch.zeros(SLOTS + 1, T, hv, head, head, device=device),
    )


def _clone(cache):
    return SimpleNamespace(
        conv=[cache.conv[0].clone()],
        temporal=cache.temporal.clone(),
        intermediate_conv_window=[cache.intermediate_conv_window[0].clone()],
        intermediate_ssm=cache.intermediate_ssm.clone(),
    )


def _sglang_verify(mod, y, yq, cache, slots, inter, rows):
    """SGLang's verify GDN core on the same projection, as gdn_backend runs it."""
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    from sgl_amd_mono.qwen35_a17b.fp8.gfx950.gemv import block_fp8_gemv_wg

    hk, hv, d, c = TP4.gdn_k_heads, TP4.gdn_v_heads, TP4.gdn_head, TP4.conv_dim
    qkvz = block_fp8_gemv_wg(
        y, yq, mod.in_proj_qkvz.weight, mod.in_proj_qkvz.weight_scale_inv
    )
    req = rows // T
    x = qkvz[:, :c].reshape(req, T, c).transpose(1, 2)
    mixed = causal_conv1d_update(
        x,
        cache.conv[0],
        mod.conv1d.weight.reshape(c, 4),
        None,
        "silu",
        conv_state_indices=slots,
        intermediate_conv_window=cache.intermediate_conv_window[0],
        intermediate_state_indices=inter,
    )
    mixed = mixed.transpose(1, 2).reshape(rows, c)
    ba = y @ mod.in_proj_ba.weight.t()
    b, a = ba[:, :hv].contiguous(), ba[:, hv:].contiguous()
    q = mixed[:, : hk * d].reshape(1, rows, hk, d)
    k = mixed[:, hk * d : 2 * hk * d].reshape(1, rows, hk, d)
    v = mixed[:, 2 * hk * d :].reshape(1, rows, hv, d)
    o = fused_sigmoid_gating_delta_rule_update(
        A_log=mod.A_log,
        a=a,
        dt_bias=mod.dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=cache.temporal,
        initial_state_indices=slots,
        scale=d**-0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.arange(0, rows + 1, T, dtype=torch.int32, device=y.device),
        disable_state_update=True,
        intermediate_states_buffer=cache.intermediate_ssm,
        intermediate_state_indices=inter,
        cache_steps=T,
    )
    o = o.reshape(rows, hv, d).float()
    z = qkvz[:, 2 * hk * d + hv * d :].reshape(rows, hv, d).float()
    rstd = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + mod.norm.eps)
    core = o * rstd * mod.norm.weight.float() * (z * torch.sigmoid(z))
    return core.reshape(rows, hv * d).to(torch.bfloat16)


def _quant_1x128(y):
    """aiter's per-1x128 quant of ``y``: ``max(amax, 1e-10) / 448`` per group."""
    rows, k = y.shape
    groups = y.float().view(rows, k // 128, 128)
    scale = groups.abs().amax(-1).clamp(min=1e-10) * (1.0 / 448.0)
    q = (groups * (1.0 / scale)[:, :, None]).clamp(-448, 448).to(FP8)
    return q.view(rows, k), scale


def _commit(cache, slots, inter, accepted):
    """SGLang's accept step: the snapshot after the last accepted draft token."""
    for r, n in enumerate(accepted):
        if slots[r] < 0:
            continue
        s, i = int(slots[r]), int(inter[r])
        cache.temporal[s] = cache.intermediate_ssm[i, n - 1]
        cache.conv[0][s] = cache.intermediate_conv_window[0][i, n - 1]


@pytest.mark.parametrize("transposed_conv", [False, True])
@pytest.mark.parametrize("requests", [1, 3, 8])
def test_verify_chain_matches_sglang_and_commits_any_acceptance(
    device, requests, transposed_conv
):
    from sgl_amd_mono.qwen35_a17b.fp8.gfx950.gdn import gdn_core

    gen = torch.Generator(device=device).manual_seed(100 + requests)
    rows = requests * T
    mod = _gdn(gen, device)
    ours = _cache(gen, device, transposed_conv)
    ref = _clone(ours)
    perm = torch.randperm(SLOTS, generator=gen, device=device)[:requests]
    slots = perm.to(torch.int32)
    if requests > 1:
        slots[1] = -1  # a pad request of a CUDA-graph batch
    inter = torch.arange(requests, dtype=torch.int32, device=device)
    y, yq = _activations(rows, gen, device)
    committed = ours.temporal.clone()

    core, (cq, cs) = verify.verify_core(mod, y, yq, ours, (slots, inter), rows, T, TP4)
    want = _sglang_verify(mod, y, yq, ref, slots, inter, rows)

    live = slots >= 0
    live_rows = live.repeat_interleave(T)
    assert _cos(core[live_rows], want[live_rows]) > 0.9999
    assert (core[~live_rows] == 0).all()
    use, rows_i = slots[live].long(), inter[live].long()
    assert torch.equal(ours.conv[0][use], ref.conv[0][use])
    assert torch.equal(
        ours.intermediate_conv_window[0][rows_i],
        ref.intermediate_conv_window[0][rows_i],
    )
    got_ssm = ours.intermediate_ssm[rows_i]
    want_ssm = ref.intermediate_ssm[rows_i]
    assert _cos(got_ssm, want_ssm) > 0.99999
    assert _rel(got_ssm, want_ssm) < 1e-4
    # the committed state is SGLang's accept step's to write
    assert torch.equal(ours.temporal, committed)
    want_q, want_s = _quant_1x128(core)
    assert torch.equal(cs, want_s)
    assert torch.equal(cq.view(torch.uint8), want_q.view(torch.uint8))

    accepted = [1 + (r * 3) % T for r in range(requests)]
    _commit(ours, slots, inter, accepted)
    _commit(ref, slots, inter, accepted)
    y2, yq2 = _activations(requests, gen, device)
    out, _ = gdn_core(mod, y2, yq2, ours.conv[0], ours.temporal, slots, requests)
    out_ref, _ = gdn_core(mod, y2, yq2, ref.conv[0], ref.temporal, slots, requests)
    assert _cos(out[live], out_ref[live]) > 0.9999
    assert _cos(ours.temporal[use], ref.temporal[use]) > 0.99999
    assert torch.equal(ours.conv[0][use], ref.conv[0][use])
