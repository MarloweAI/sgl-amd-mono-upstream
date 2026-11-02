# SPDX-License-Identifier: Apache-2.0
"""Online FP8 (E4M3, a scale a row) quantization of bf16 dense weights the kernels
stream once a layer: the pooler and the unpooler (36 % of a layer's bytes at TP 8).
The kernels decode the codes to bf16 in registers (``fp8_to_bf16x8``), run the bf16
MFMA on the unchanged bf16 activations, and scale each output row once in fp32.
Host-side torch only."""

import torch

FP8_MAX = 448.0  # E4M3's largest finite value


def quantize_fp8_rows(
    w: torch.Tensor, chunk: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """``w`` [N, K] (bf16 or fp32) -> (codes float8_e4m3fn [N, K], scales fp32 [N]):
    row r's codes are ``w[r] / scales[r]`` rounded to E4M3, ``scales[r]`` = the row's
    largest magnitude / 448 (a zero row keeps scale 1). ``chunk`` rows at a time."""
    assert w.dim() == 2, tuple(w.shape)
    N, K = w.shape
    codes = torch.empty(N, K, dtype=torch.float8_e4m3fn, device=w.device)
    scales = torch.empty(N, dtype=torch.float32, device=w.device)
    for r0 in range(0, N, chunk):
        x = w[r0 : r0 + chunk].float()
        amax = x.abs().amax(dim=1)
        sc = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
        codes[r0 : r0 + chunk] = (
            (x / sc[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        )
        scales[r0 : r0 + chunk] = sc
    return codes, scales


def dequantize_fp8_rows(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The fp32 values [N, K] of a (codes, scales) pair."""
    return codes.float() * scales[:, None]


def pack_fp8_rows(w: torch.Tensor) -> torch.Tensor:
    """One uint8 buffer the kernels read through one pointer: the E4M3 codes [N, K]
    (row-major, a byte each) followed by the fp32 row scales [N] (at byte N K)."""
    codes, scales = quantize_fp8_rows(w)
    N, K = codes.shape
    out = torch.empty(N * K + 4 * N, dtype=torch.uint8, device=w.device)
    out[: N * K] = codes.view(torch.uint8).reshape(-1)
    out[N * K :] = scales.view(torch.uint8)
    return out


def pack_fp8_rows_pooler(w: torch.Tensor) -> torch.Tensor:
    """``pack_fp8_rows`` with each row's codes permuted for the pooler GEMV's 16-byte
    lane loads: within every 128-k wave block (k = 32 kc + 8 g4 + j, kc < 4 the MFMA
    steps, g4 the lane group, j < 8), the two steps of a lane group sit together
    (position 64 (kc / 2) + 16 g4 + 8 (kc % 2) + j), so one load a step pair feeds
    both steps (a CU issues VMEM instructions at a fixed rate: 8-byte lane loads
    stream at half the bandwidth of 16-byte ones). The row scales follow as before."""
    codes, scales = quantize_fp8_rows(w)
    N, K = codes.shape
    assert K % 128 == 0, K
    c8 = codes.view(torch.uint8).reshape(
        N, K // 128, 2, 2, 4, 8
    )  # [n][block][kc / 2][kc % 2][g4][j]
    c8 = (
        c8.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(N, K)
    )  # -> [n][block][kc / 2][g4][kc % 2][j]
    out = torch.empty(N * K + 4 * N, dtype=torch.uint8, device=w.device)
    out[: N * K] = c8.reshape(-1)
    out[N * K :] = scales.view(torch.uint8)
    return out


def unpack_fp8_rows(packed: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The fp32 values [n, k] a ``pack_fp8_rows`` buffer encodes."""
    codes = packed[: n * k].view(torch.float8_e4m3fn).reshape(n, k)
    scales = packed[n * k :].view(torch.float32)
    return dequantize_fp8_rows(codes, scales)
