# SPDX-License-Identifier: Apache-2.0
"""The DSpark draft's main_proj split over the TP ranks by output column (``main_proj.hip``):
each rank runs its 5120 / tp columns of the replicated E4M3 weight with SGLang's MXFP8 quant
and aiter's packed afp8wfp8 arithmetic, and pushes its bf16 outputs to every rank, so each
rank reads 1 / tp of the weight and holds the whole rows, bitwise the replicated GEMM's."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
H, K = 5120, 15360
MIN_ROWS, MAX_ROWS = (
    5,
    8,
)  # aiter's BLOCK_M 8 packed config (4 rows or fewer take another)
TP_SIZES = (4, 8)  # every CTA resident: the copy-out waits on all ranks' tiles
STATE_INTS = 2048
WORK_BYTES = (H // 16) * 6 * 64 * 4 * 8
EXCHANGE_BYTES = 2 * MAX_ROWS * (H // 2) * 8


STAMPS = 8  # a stamped build's realtime ticks a CTA, past the state ints (tests only)


@functools.cache
def ops(stamps: bool = False):
    """``torch.ops`` of ``main_proj.hip``, built at first use (``stamps``: the timing build)."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {
        p.name: p.read_text()
        for p in sorted([*_HERE.glob("*.hip"), *_HERE.glob("*.h")])
    }
    if stamps:
        return load(
            "sgl_amd_mono_dsv41_main_proj_st",
            sources,
            "gfx950",
            ("-DMAIN_PROJ_STAMPS",),
        )
    return load("sgl_amd_mono_dsv41_main_proj", sources, "gfx950")


class Exchange:
    """One TP group's main_proj exchange: every rank's buffer address (int64 on the host),
    the CTAs' launch counts and the chains' partials (zeroed once), the rows out. Collective:
    every rank builds it together, ``group`` the gloo group."""

    def __init__(
        self, group, rank: int, size: int, device: torch.device, deadline_s: float
    ):
        from sgl_amd_mono.common.peer import PeerBuffer

        self.rank, self.size, self.deadline_s = rank, size, float(deadline_s)
        self._buffer = PeerBuffer(EXCHANGE_BYTES, group, rank, size, device)
        self._buffer.bytes.zero_()
        self.peers = self._buffer.addresses.cpu()
        self.state = torch.zeros(STATE_INTS, dtype=torch.int32, device=device)
        self.work = torch.zeros(WORK_BYTES, dtype=torch.uint8, device=device)
        self.out = torch.zeros(MAX_ROWS, H, dtype=torch.bfloat16, device=device)

    def columns(self, weight: torch.Tensor, scale: torch.Tensor):
        """This rank's rows of main_proj's [5120, 15360] E4M3 weight and [160, 480] E8M0 scales."""
        n = H // self.size
        lo = self.rank * n
        return (
            weight.view(torch.uint8)[lo : lo + n],
            scale.view(torch.uint8)[lo // 32 : (lo + n) // 32],
        )

    def run(
        self, x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, state=None
    ) -> torch.Tensor:
        """main_proj of ``x`` [rows, 15360] bf16 -> [rows, 5120] bf16 (a view of ``out``); ``state``: a
        stamped build's (tests)."""
        o = ops(state is not None)
        o.main_proj(
            x,
            w,
            ws,
            self.out,
            self.state if state is None else state,
            self.work,
            self.peers,
            self.rank,
            self.deadline_s,
        )
        return self.out[: x.shape[0]]

    def close(self) -> None:
        self._buffer.close()
