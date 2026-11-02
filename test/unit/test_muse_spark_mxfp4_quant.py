"""The online MXFP4 quantization of bf16 expert weights (CPU): its values are the
aiter ``_mxfp4_quant_op`` rule's (the K3 test's reference), the packing is the
Quark layout the kernels decode (the even k in the low nibble, one E8M0 a block
of 32), zero blocks and saturation behave, and the error is an MXFP4's."""

import torch

from sgl_amd_mono.muse_spark.mxfp4_quant import E2M1, dequantize_mxfp4, quantize_mxfp4


def _rule(x):
    """The K3 test's ``_mx_dequant``: aiter's quantize-and-back on fp32."""
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
    mag = torch.tensor(E2M1)[code.long()]
    val = torch.where(qx < 0, -mag, mag)
    return (val * torch.ldexp(torch.ones_like(amax), unb)).reshape(shape)


def _decode_like_the_kernel_test(packed, scales):
    """The K3 test's ``_fp4_weight`` decode of the Quark layout."""
    E, N, K2 = packed.shape
    lo, hi = packed & 0xF, packed >> 4
    codes = torch.stack([lo, hi], dim=-1).reshape(E, N, 2 * K2)
    mag = torch.tensor(E2M1)[(codes & 7).long()]
    val = torch.where(codes & 8 != 0, -mag, mag)
    scale = torch.ldexp(torch.ones(1), scales.to(torch.int32) - 127)
    return (val.reshape(E, N, -1, 32) * scale[..., None]).reshape(E, N, 2 * K2)


def test_the_values_are_the_aiter_rule_and_the_layout_is_quark():
    torch.manual_seed(1)
    w = (torch.randn(3, 64, 256) * 0.02).to(torch.bfloat16)
    w[0, 0, :32] = 0  # a zero block
    w[1, 5, 32:64] = torch.linspace(-6.5, 6.5, 32).to(
        torch.bfloat16
    )  # a block to saturate (6.5 keeps the 2^0 scale)
    packed, scales = quantize_mxfp4(w, chunk=2)
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == (3, 64, 128)
    assert scales.dtype == torch.uint8 and tuple(scales.shape) == (3, 64, 8)
    back = dequantize_mxfp4(packed, scales)
    assert torch.equal(back, _rule(w)), "the values are not aiter's rule"
    assert torch.equal(back, _decode_like_the_kernel_test(packed, scales)), (
        "the layout differs"
    )
    assert (packed[0, 0, :16] == 0).all() and scales[
        0, 0, 0
    ] == 0  # the zero block: codes 0, scale 2^-127
    sat = back[1, 5, 32:64]
    assert (
        int(scales[1, 5, 1]) == 127 and sat.abs().max() == 6.0
    )  # 6.5 at scale 1 saturates to the E2M1 max
    rel = ((back.float() - w.float()).norm() / w.float().norm()).item()
    assert 0.02 < rel < 0.2, rel  # an MXFP4's error, not a layout slip


def test_the_even_k_sits_in_the_low_nibble():
    w = torch.zeros(1, 1, 32)
    w[0, 0, 0] = (
        1.0  # k = 0: the block max, code 2 at the block's scale... 1.0 / 2^(0 - 2) = 4 -> code 6 (4.0)
    )
    w[0, 0, 1] = -0.5  # k = 1: 2.0 -> code 4 with the sign -> 0xC
    packed, scales = quantize_mxfp4(w)
    assert int(scales[0, 0, 0]) == 125  # 2^-2
    assert int(packed[0, 0, 0]) == (0xC << 4) | 0x6
