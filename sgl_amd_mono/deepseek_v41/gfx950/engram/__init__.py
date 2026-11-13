# SPDX-License-Identifier: Apache-2.0
"""A verify step's Engram rows in one launch (``engram_rows.hip``): SGLang's n-gram hash, each
Engram layer's row gather from this rank's table shard and the all-reduce that reassembles the
row-sharded tables, bitwise the three: the owner of each row pushes it to every rank."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
N, HEADS, DIM, BLOCK = (
    4,
    8,
    256,
    32,
)  # n-gram sizes 2..N, hash heads, a row's values, its E8M0 block
COLS = (N - 1) * HEADS
MAX_LAYERS, MAX_TOKENS, MAX_TP = 2, 48, 8
MAX_CTAS = MAX_LAYERS * MAX_TOKENS * COLS
EXCHANGE_BYTES = 2 * MAX_CTAS * (DIM // 2) * 8


@functools.cache
def ops():
    """``torch.ops`` of ``engram_rows.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    return load("sgl_amd_mono_dsv41_engram_rows", sources, "gfx950")


class Rows:
    """One TP group's Engram row exchange: every rank's buffer address (int64 on the host), the
    CTAs' launch counts (zeroed once) and the rows out. Collective: every rank builds it together,
    ``group`` the gloo group."""

    def __init__(
        self,
        group,
        rank: int,
        size: int,
        device: torch.device,
        deadline_s: float,
        layers: int,
    ):
        from sgl_amd_mono.common.peer import PeerBuffer

        assert 1 <= layers <= MAX_LAYERS and 1 < size <= MAX_TP
        self.rank, self.size, self.deadline_s = rank, size, float(deadline_s)
        self._buffer = PeerBuffer(EXCHANGE_BYTES, group, rank, size, device)
        self._buffer.bytes.zero_()
        self.peers = self._buffer.addresses.cpu()
        self.state = torch.zeros(MAX_CTAS, dtype=torch.int32, device=device)
        self.out = torch.zeros(
            layers, MAX_TOKENS, COLS, DIM, dtype=torch.bfloat16, device=device
        )

    def run(
        self, ids, positions, slots, hasher, tables, block: int, mm_pad_shift: int
    ) -> list:
        """Each Engram layer's rows [tokens, COLS, DIM] of a verify step's ``ids`` (blocks of
        ``block``): ``hasher`` SGLang's ``EngramHasher``; ``tables`` per layer (its hash layer, the
        table and scale addresses, its global rows [lo, hi) on this rank); ids at or above
        ``mm_pad_shift`` read as the image token."""
        image = hasher.image_token_id
        ops().engram_rows(
            ids, positions, slots, hasher.history, hasher.token_map, hasher.multipliers,
            hasher.primes, hasher.offsets, [t[0] for t in tables], [t[1] for t in tables],
            [t[2] for t in tables], [t[3] for t in tables], [t[4] for t in tables], block,
            hasher.pad_id, image if image is not None else -1, mm_pad_shift, self.out, self.state,
            self.peers, self.rank, self.deadline_s,
        )  # fmt: skip
        return [self.out[k, : ids.shape[0]] for k in range(len(tables))]

    def close(self) -> None:
        self._buffer.close()
