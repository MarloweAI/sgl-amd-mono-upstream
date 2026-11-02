"""The layout contract between the Qwen3.8-Flash-Next launches and SGLang's AITER
weight layouts, on CPU: the kernel reads SGLang's MXFP4 FusedMoE params and FP8
projection weights in place, so it hard-codes AITER's ``shuffle_weight(w, (16, 16))``
and ``e8m0_shuffle`` storage orders and FusedMoE's padded intermediate width. The
packing's un-shuffles must invert AITER's shuffles exactly, and its layout constants
must match SGLang's AITER MoE weight sizes."""

import pytest
import torch

from sgl_amd_mono.qwen38_flash_next.gfx950 import layout as LY
from sgl_amd_mono.qwen38_flash_next.gfx950 import packing
from sgl_amd_mono.qwen38_flash_next.gfx950.config import HIDDEN, MOE_INTER, N_EXPERTS


def _aiter():
    """AITER's shuffles (CPU torch); skipped where AITER does not import."""
    try:
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility.fp4_utils import e8m0_shuffle
    except Exception as e:  # noqa: BLE001 - no AITER, or no ROCm to import it on
        pytest.skip(f"AITER does not import: {type(e).__name__}: {e}")
    return shuffle_weight, e8m0_shuffle


def _bytes(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)


@pytest.mark.parametrize(
    "rows,kbytes", [(2560, 6144), (16384, 2560), (128, 2560), (640, 2560)]
)
def test_unshuffle16_inverts_aiter_shuffle(rows, kbytes):
    # FP8 projection weights: AITER's (16, 16) shuffle of a 1-byte [N, K] matrix
    shuffle_weight, _ = _aiter()
    w = _bytes(rows, kbytes, seed=rows)
    shuffled = shuffle_weight(w.clone(), (16, 16))
    assert not torch.equal(shuffled.view(torch.uint8), w)
    assert torch.equal(packing.unshuffle16(shuffled), w)


def test_moe_padding_matches_fusedmoe():
    # FusedMoE pads the MXFP4 intermediate width for AITER; the kernel's constants
    # assume the same padding
    from sglang.srt.layers.moe.utils import get_moe_weight_sizes

    w13_dim, w2_dim = get_moe_weight_sizes(
        MOE_INTER, is_concat=True, is_packed=True, is_aiter_moe=True
    )[:2]
    assert w13_dim == 2 * LY.MOE_INTER_PAD
    assert w2_dim == LY.MOE_INTER_PAD // 2
    assert LY.UP_TILE0 * 16 == LY.MOE_INTER_PAD
    assert LY.GU_BYTES == 2 * LY.MOE_INTER_PAD * HIDDEN // 2
    assert LY.DN_BYTES == HIDDEN * LY.MOE_INTER_PAD // 2


def test_unshuffle_moe_expert_inverts_fusedmoe_layout():
    # MXFP4 experts: the checkpoint's gate, up, down and E8M0 scales -> FusedMoE's
    # padded, AITER-shuffled params -> the packing's inverse gives the bytes back
    shuffle_weight, e8m0_shuffle = _aiter()

    def scales(s):
        e, r, _ = s.shape  # flattened [E * rows, groups], as SGLang shuffles them
        return e8m0_shuffle(s.reshape(e * r, -1).contiguous()).view(e, r, -1)

    E, I, IP = 2, MOE_INTER, LY.MOE_INTER_PAD
    gate, up = _bytes(E, I, HIDDEN // 2, seed=1), _bytes(E, I, HIDDEN // 2, seed=2)
    down = _bytes(E, HIDDEN, I // 2, seed=3)
    gate_s = _bytes(E, I, HIDDEN // 32, seed=4)
    up_s = _bytes(E, I, HIDDEN // 32, seed=5)
    down_s = _bytes(E, HIDDEN, I // 32, seed=6)
    w13 = torch.zeros(E, 2 * IP, HIDDEN // 2, dtype=torch.uint8)
    w13[:, :I], w13[:, IP : IP + I] = gate, up
    s13 = torch.zeros(E, 2 * IP, LY.GU_SGROUPS, dtype=torch.uint8)
    s13[:, :I], s13[:, IP : IP + I] = gate_s, up_s
    w2 = torch.zeros(E, HIDDEN, IP // 2, dtype=torch.uint8)
    w2[..., : I // 2] = down
    s2 = torch.zeros(E, HIDDEN, LY.DN_SGROUPS, dtype=torch.uint8)
    s2[..., : I // 32] = down_s
    m = {
        "gu": shuffle_weight(w13.contiguous(), (16, 16)),
        "gu_s": scales(s13),
        "dn": shuffle_weight(w2.contiguous(), (16, 16)),
        "dn_s": scales(s2),
    }
    for e in range(E):
        got = packing.unshuffle_moe_expert(m, e)
        ref = (gate[e], up[e], down[e], gate_s[e], up_s[e], down_s[e])
        for name, g, r in zip(("gate", "up", "down", "gate_s", "up_s", "down_s"), got, ref):  # fmt: skip
            assert torch.equal(g, r), f"expert {e} {name}"


def test_check_moe_params_accepts_fusedmoe_shapes():
    # the 512 routed experts and the fused shared expert
    E = N_EXPERTS + 1
    m = {
        "gu": torch.empty(E, 2 * LY.MOE_INTER_PAD, HIDDEN // 2, dtype=torch.uint8),
        "gu_s": torch.empty(E, 2 * LY.MOE_INTER_PAD, LY.GU_SGROUPS, dtype=torch.uint8),
        "dn": torch.empty(E, HIDDEN, LY.MOE_INTER_PAD // 2, dtype=torch.uint8),
        "dn_s": torch.empty(E, HIDDEN, LY.DN_SGROUPS, dtype=torch.uint8),
    }
    packing.check_moe_params(m)
    m["gu"] = torch.empty(E - 1, 2 * LY.MOE_INTER_PAD, HIDDEN // 2, dtype=torch.uint8)
    with pytest.raises(AssertionError):
        packing.check_moe_params(m)


def test_pack_bf16_tiles_16_rows_by_64_k():
    # the kernel's bf16 MFMA tiles: 16 rows x 64 K, two 32-K halves of 8-value
    # lanes; every element lands once
    w = torch.arange(32 * 128, dtype=torch.int16).view(32, 128).view(torch.bfloat16)
    packed = packing.pack_bf16(w).view(torch.int16)
    assert packed.numel() == w.numel()
    assert torch.equal(packed.sort().values, torch.arange(32 * 128, dtype=torch.int16))
    # tile (row group 0, K chunk 0): lane l of half 0 holds row l % 16, K 8 (l // 16) ..
    tile = packed[: 16 * 64].view(2, 4, 16, 8)
    assert torch.equal(tile[0, 1, 3], w.view(torch.int16)[3, 8:16])
    with pytest.raises(ValueError):
        packing.pack_bf16(torch.zeros(24, 64, dtype=torch.bfloat16))
