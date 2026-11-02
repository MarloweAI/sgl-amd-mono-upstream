"""The online FP8 row quantization of dense weights (CPU): E4M3 codes with a scale a
row, the largest magnitude of a row landing on 448, zero rows kept, and an FP8's
error."""

import torch

from sgl_amd_mono.muse_spark.fp8_quant import (
    FP8_MAX,
    dequantize_fp8_rows,
    quantize_fp8_rows,
)


def test_rows_scale_to_the_e4m3_range():
    torch.manual_seed(2)
    w = (torch.randn(300, 512) * 0.02).to(torch.bfloat16)
    w[7] = 0  # a zero row
    w[11, 3] = 1.5  # a row with an outlier
    codes, scales = quantize_fp8_rows(w, chunk=128)
    assert codes.dtype == torch.float8_e4m3fn and tuple(codes.shape) == (300, 512)
    assert scales.dtype == torch.float32 and tuple(scales.shape) == (300,)
    back = dequantize_fp8_rows(codes, scales)
    assert torch.isfinite(back).all()
    assert (back[7] == 0).all() and scales[7] == 1.0
    assert (
        codes.float().abs().amax(dim=1)[11] == FP8_MAX
    )  # the outlier lands on the max code
    assert abs(back[11, 3].item() - 1.5) / 1.5 < 1 / 16
    rel = ((back - w.float()).norm() / w.float().norm()).item()
    assert 0.005 < rel < 0.08, (
        rel
    )  # an E4M3's error (3 mantissa bits), not a scale slip
    # every row's largest magnitude is reproduced to E4M3 precision
    amax = w.float().abs().amax(dim=1)
    live = amax > 0
    assert (
        (back.abs().amax(dim=1)[live] - amax[live]).abs() / amax[live] < 1 / 16
    ).all()
