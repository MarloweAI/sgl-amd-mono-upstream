# SPDX-License-Identifier: Apache-2.0
"""The attention's dense FP8 projections (wqkv_a, wq_b, wo_b) at wide row counts (60 .. 384
rows) on aiter's FlyDSL gfx950 MXFP8 GEMMs: out[m, N] bf16 = x[m, K] . W[N, K]^T, x e4m3 with
its ue8m0 codes [m, K / 32], W the checkpoint's e4m3 in aiter's (16, 16) preshuffle with its
E8M0 32 x 32 block codes.

SGLang's route runs aiter's tuned Triton ``gemm_afp8wfp8`` on the plain weight. Here each
(N, K, width class) in ``KERNELS`` runs, on the preshuffled weight, either aiter's own tuned
choice for it (``gemm_a8w8_blockscale``'s preshuffled e8m0 route: None) or a config of its
FlyDSL batched GEMM (``flydsl.batched_gemm_a8w8``, batch 1), whichever was faster on real
weights at the class's top row count; a class without an entry keeps SGLang's GEMM. The sums
are SGLang's but for the K split: where the two split K differently, about 1e-8 of the outputs
differ by one bf16 ulp. The result does not depend on the run.
"""

import torch

# per (N, K, width class): None, aiter's tuned preshuffled route, or a FlyDSL batched GEMM
# config (wqkv_a 1792 x 5120 at every TP, wq_b 16384 / 8192 x 1280 at TP 2 / 4, wo_b
# 5120 x 4096 / 2048)
KERNELS = {
    (1792, 5120, 192): "flydsl_bmm_mxfp8_mfma_t96x64x256_w2x4_nb3_sk4",
    (1792, 5120, 384): "flydsl_bmm_mxfp8_mfma_t96x64x256_w2x4_nb3_sk2",
    (16384, 1280, 96): None,
    (16384, 1280, 192): None,
    (16384, 1280, 384): None,
    (8192, 1280, 96): None,
    (8192, 1280, 192): "flydsl_bmm_mxfp8_mfma_t96x64x256_w1x4_nb4_sk1_bd1",
    (8192, 1280, 384): None,
    (5120, 4096, 96): None,
    (5120, 4096, 192): None,
    (5120, 4096, 384): None,
    (5120, 2048, 384): None,
}
CLASSES = (96, 192, 384)


def width_class(m: int) -> int:
    for c in CLASSES:
        if m <= c:
            return c
    raise ValueError(f"fp8 dense: {m} rows past 384")


def served(n: int, k: int, m: int) -> bool:
    """Whether an n x k projection at m rows runs here (a kernel for m's class)."""
    return (n, k, width_class(m)) in KERNELS


class Projection:
    """One dense projection: its weight in aiter's (16, 16) preshuffle (a copy; SGLang's
    route keeps the plain one) and its codes, as the FlyDSL GEMM reads them."""

    def __init__(self, weight: torch.Tensor, codes: torch.Tensor):
        from aiter.ops.shuffle import shuffle_weight

        n, k = weight.shape
        assert weight.dtype == torch.float8_e4m3fn and codes.shape == (n // 32, k // 32)
        self.weight, self.n, self.k = weight, n, k
        self.shuffled = shuffle_weight(weight.data, (16, 16)).view(1, n, k)
        self.codes = codes.data.view(torch.float8_e8m0fnu).view(1, n // 32, k // 32)
        self.kernels = {c: KERNELS[(n, k, c)] for c in CLASSES if (n, k, c) in KERNELS}

    def __call__(self, xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """bf16 [m, n] = MXFP8 x [m, k] (e4m3, ue8m0 [m, k / 32]) . W^T."""
        from aiter import gemm_a8w8_blockscale
        from aiter.ops.flydsl.batched_gemm_a8w8 import run_bmm_a8w8_mxfp8

        m, n, k = xq.shape[0], self.n, self.k
        name = self.kernels[width_class(m)]
        e8 = torch.float8_e8m0fnu
        if name is None:
            return gemm_a8w8_blockscale(
                xq, self.shuffled.view(n, k), xs.view(e8), self.codes.view(n // 32, k // 32),
                dtype=torch.bfloat16, isBpreshuffled=True,
            )  # fmt: skip
        out = torch.empty(m, n, dtype=torch.bfloat16, device=xq.device)
        run_bmm_a8w8_mxfp8(
            xq.view(m, 1, k),
            self.shuffled,
            xs.view(e8).view(m, 1, k // 32),
            self.codes,
            out.view(m, 1, n),
            kernel_name=name,
        )
        return out
