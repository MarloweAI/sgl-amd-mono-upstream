"""The Muse Spark K1 launch (one gfx950 GPU) against the math of the ROCm model's
fused ``qkv_proj_qk_norm_rope_cache`` it replaces: the q | k | v | gate projection
(one bf16 rounding), the per-head RMSNorm, the query scale, interleaved RoPE with
bf16 cos / sin and rounded products, and the fp8 K / V cache write through the
sliding layers' slot table; and that it compiles for every served width."""

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

SLOTS = 4096  # cache slots of the test's K / V planes
POSITIONS = 4096
THETA = 500000.0


def _cos_sin(device):
    from sgl_amd_mono.muse_spark.gfx950.config import HEAD_DIM

    inv = 1.0 / (
        THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    freqs = torch.arange(POSITIONS, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device)


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


def _reference(x, w, cos_sin, positions, rope):
    """The fused kernel's math (``_norm_rope`` of qk_norm_rope_cache.py): one
    rounding after the fp32 GEMM, the per-head norm rounded, q scaled and rounded,
    bf16 cos / sin with each RoPE product rounded, then the sum rounded."""
    from sgl_amd_mono.muse_spark.gfx950.config import (
        G_OFF,
        HEAD_DIM,
        K_OFF,
        LOCAL_KV_HEADS,
        LOCAL_Q_HEADS,
        QK_SCALE,
        RMS_EPS,
        V_OFF,
    )

    y = (x.float() @ w.float().T).to(torch.bfloat16)
    rows = x.shape[0]
    q = y[:, :K_OFF].reshape(rows, LOCAL_Q_HEADS, HEAD_DIM)
    k = y[:, K_OFF:V_OFF].reshape(rows, LOCAL_KV_HEADS, HEAD_DIM)
    v = y[:, V_OFF:G_OFF].reshape(rows, LOCAL_KV_HEADS, HEAD_DIM)
    g = y[:, G_OFF:]

    def norm(t):
        tf = t.float()
        rstd = torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return _bf(tf * rstd)

    def rotate(value):
        if not rope:
            return value.to(torch.bfloat16)
        cs = cos_sin[positions]  # [rows, 64]: cos 32 | sin 32
        d = torch.arange(HEAD_DIM, device=value.device)
        cos = _bf(cs[:, None, d // 2])
        sin = _bf(cs[:, None, HEAD_DIM // 2 + d // 2])
        other = (
            value.reshape(rows, value.shape[1], HEAD_DIM // 2, 2)
            .flip(-1)
            .reshape(rows, value.shape[1], HEAD_DIM)
        )
        rotated = torch.where(d % 2 == 0, -other, other)
        return (_bf(value * cos) + _bf(rotated * sin)).to(torch.bfloat16)

    q = rotate(_bf(norm(q) * QK_SCALE))
    k = rotate(norm(k))
    return q.reshape(rows, -1), k.reshape(rows, -1), v.reshape(rows, -1), g


def _fp8(t):
    return t.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def _close(got, ref, rtol, what, flips=1e-3):
    """Within ``rtol`` relative but for a few bf16 flips (a 1-ulp difference of the
    split-K sum, amplified by the query scale), themselves within 4 ``rtol``."""
    got, ref = got.float(), ref.float()
    err = (got - ref).abs()
    bad = (err > rtol * ref.abs() + 1e-3).float().mean().item()
    worst = (err > 4 * rtol * ref.abs() + 1e-3).float().mean().item()
    assert bad <= flips and worst == 0.0, (
        f"{what}: {bad:.3%} of the elements off by more than {rtol} relative, {worst:.3%} by more than {4 * rtol}"
    )


def _one_ulp_fp8(got, ref, what):
    """fp8 e4m3 values within one code of each other, nearly all equal."""
    g, r = got.float(), ref.float()
    assert ((g - r).abs() <= r.abs() / 8 + 1e-3).all(), f"{what}: a cache value is more than one fp8 code off"  # fmt: skip
    exact = (g == r).float().mean().item()
    assert exact > 0.95, f"{what}: only {exact:.1%} of the cache values match exactly"


@pytest.mark.parametrize(
    "rows,rope,swa,slot64",
    [
        (8, True, True, True),
        (16, True, False, True),
        (32, False, True, False),
        (64, True, True, True),
        (5, True, False, True),
    ],
)
def test_k1_matches_the_fused_kernel_math(rows, rope, swa, slot64):
    from sgl_amd_mono.muse_spark.gfx950.config import (
        HEAD_DIM,
        HIDDEN,
        KV_K,
        LOCAL_KV_HEADS,
        MAX_TOKENS,
        Q_K,
        QKVG_ROWS,
    )
    from sgl_amd_mono.muse_spark.gfx950.layer import K1Build, build_k1
    from sgl_amd_mono.muse_spark.gfx950.layout import k1_scratch_layout, scratch_bytes

    torch.manual_seed(rows)
    dev = torch.device("cuda")
    bf = torch.bfloat16
    x = torch.randn(rows, HIDDEN, device=dev).to(bf)
    w = (torch.randn(QKVG_ROWS, HIDDEN, device=dev) * 0.02).to(bf)
    cos_sin = _cos_sin(dev)
    positions = torch.randint(0, POSITIONS, (rows,), device=dev, dtype=torch.int64)
    full = torch.randperm(SLOTS, device=dev)[:rows]
    slots = full if slot64 else full.to(torch.int32)
    # the sliding layers' table: full slot -> SWA slot, a permutation here
    swa_map = torch.randperm(SLOTS, device=dev, dtype=torch.int64)
    k_cache = torch.zeros(SLOTS, LOCAL_KV_HEADS, HEAD_DIM, device=dev).to(
        torch.float8_e4m3fn
    )
    v_cache = torch.zeros_like(k_cache)
    q = torch.full((MAX_TOKENS, Q_K), float("nan"), device=dev, dtype=bf)
    k = torch.full((MAX_TOKENS, KV_K), float("nan"), device=dev, dtype=bf)
    v = torch.full_like(k, float("nan"))
    g = torch.full_like(q, float("nan"))
    scratch = torch.zeros(
        scratch_bytes(k1_scratch_layout()), dtype=torch.uint8, device=dev
    )
    epoch = _epoch(dev)
    launch = build_k1(K1Build(tokens=rows))
    for step in range(2):  # the second step's tags differ from the first's
        launch(
            x.data_ptr(), w.data_ptr(), cos_sin.data_ptr(), positions.data_ptr(),
            slots.data_ptr(), swa_map.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
            q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), scratch.data_ptr(),
            epoch.data_ptr(), 3, int(rope), int(swa), int(slot64), 1,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        epoch[0] += 1
    torch.cuda.synchronize()
    q_ref, k_ref, v_ref, g_ref = _reference(x, w, cos_sin, positions, rope)
    _close(q[:rows], q_ref, 1 / 64, "q")
    _close(k[:rows], k_ref, 1 / 64, "k")
    _close(v[:rows], v_ref, 1 / 128, "v")
    _close(g[:rows], g_ref, 1 / 128, "gate")
    assert q[rows:].isnan().all(), "rows past the step were written"
    at = swa_map[full] if swa else full
    _one_ulp_fp8(k_cache[at].reshape(rows, -1), _fp8(k_ref), "K cache")
    _one_ulp_fp8(v_cache[at].reshape(rows, -1), _fp8(v_ref), "V cache")
    untouched = torch.ones(SLOTS, dtype=torch.bool, device=dev)
    untouched[at] = False
    assert (k_cache[untouched].float() == 0).all() and (
        v_cache[untouched].float() == 0
    ).all()


@pytest.mark.parametrize("rows", [8, 16, 32, 64])
def test_every_width_compiles(rows):
    from sgl_amd_mono.muse_spark.gfx950.layer import K1Build, build_k1
    from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

    compile_only(build_k1(K1Build(tokens=rows)))


def test_a_fault_site_compiles():
    from sgl_amd_mono.muse_spark.gfx950.layer import K1Build, build_k1
    from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

    compile_only(build_k1(K1Build(tokens=8, fault_site=1, fault_after=1)))
