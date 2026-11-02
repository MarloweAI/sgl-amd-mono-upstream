# SPDX-License-Identifier: Apache-2.0
"""Online MXFP4 quantization of bf16 expert weights into the Quark layout the K3
stage reads: E2M1 codes packed two a byte along K (the low nibble the even k) and
one E8M0 scale a block of 32. The rule is aiter's Triton ``_mxfp4_quant_op``, the
one the kernels apply to the experts' activations: the block scale from the max
rounded at mantissa bit 21, E2M1 half to even, saturating at 6. Host-side torch
only (no aiter / FlyDSL), so it also runs on a CPU."""

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GROUP = 32


def quantize_mxfp4(
    w: torch.Tensor, chunk: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """``w`` [E, N, K] (bf16 or fp32, K a multiple of 32) -> (packed u8 [E, N, K / 2],
    scales u8 [E, N, K / 32]), computed in fp32 ``chunk`` experts at a time."""
    assert w.dim() == 3 and w.shape[-1] % GROUP == 0, tuple(w.shape)
    E, N, K = w.shape
    packed = torch.empty(E, N, K // 2, dtype=torch.uint8, device=w.device)
    scales = torch.empty(E, N, K // GROUP, dtype=torch.uint8, device=w.device)
    for s in range(0, E, chunk):
        p, e = _quantize(w[s : s + chunk].float())
        packed[s : s + chunk] = p
        scales[s : s + chunk] = e
    return packed, scales


def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = x.shape
    xb = x.reshape(*shape[:-1], -1, GROUP)
    amax = xb.abs().amax(-1, keepdim=True)
    # the block exponent: the max's float exponent after rounding at mantissa bit 21, less 2
    bits = (amax.view(torch.int32) + 0x200000) & -0x800000
    unb = (((bits >> 23) & 0xFF) - 129).clamp(min=-127)
    qx = xb * torch.ldexp(torch.ones_like(amax), -unb)
    a = qx.abs()
    ab = a.view(torch.int32)
    # the E2M1 code: below 1 through a float add (half to even at the 0.5 grid), from
    # 1 through the exponent and the top mantissa bit (half to even), 6 and above the max
    den = ((a + 4194304.0).view(torch.int32) - (149 << 23)) & 0xFF
    nrm = ((ab + ((-126 << 23) + 0x1FFFFF) + ((ab >> 22) & 1)) >> 22) & 0xFF
    code = torch.where(
        a >= 6.0, torch.full_like(den, 7), torch.where(a < 1.0, den, nrm)
    )
    nib = (
        code | torch.where(qx < 0, torch.full_like(code, 8), torch.zeros_like(code))
    ).to(torch.uint8)
    nib = nib.reshape(*shape[:-1], -1, 2)
    packed = nib[..., 0] | (nib[..., 1] << 4)
    e8m0 = (unb.squeeze(-1) + 127).to(torch.uint8)
    return packed, e8m0


def dequantize_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The fp32 values [..., K] a Quark-layout (packed, scales) pair encodes."""
    lo, hi = packed & 0xF, packed >> 4
    codes = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)
    mag = torch.tensor(E2M1, device=packed.device)[(codes & 7).long()]
    val = torch.where(codes & 8 != 0, -mag, mag)
    scale = torch.ldexp(
        torch.ones((), device=packed.device), scales.to(torch.int32) - 127
    )
    K = codes.shape[-1]
    return (
        val.reshape(*packed.shape[:-1], K // GROUP, GROUP) * scale[..., None]
    ).reshape(*packed.shape[:-1], K)
