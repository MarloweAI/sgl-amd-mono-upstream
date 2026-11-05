# SPDX-License-Identifier: Apache-2.0
"""The DSpark draft's sampled Markov steps on gfx950 (``markov.hip``): a step's bias
GEMV, scores, Gumbel-max pick and the ranks' bests swapped in two launches, every
rank's token alike without a collective; after a block, the sampled sequences' step
logits gathered whole from every rank's shard in one launch."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
DIM = 256  # the Markov rank the kernels hold in registers
MAX_SEQUENCES = 4
BLOCK_ROWS = 64
# a sequence's 16 packet slots of 16 B (markov.hip's SLOTS); the rank's step logits follow
EXCHANGE_BYTES = MAX_SEQUENCES * 16 * 16


@functools.cache
def ops():
    """``torch.ops`` of ``markov.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    return load("sgl_amd_mono_dsv41_markov", sources, "gfx950")


class Exchange:
    """One TP group's exchange: every rank's buffer address (int64 on the host), the
    per-sequence call counters and last tokens, and this rank's step logits
    (``slab``, [MAX_SEQUENCES, gamma, part] f32, which every rank reads at
    ``EXCHANGE_BYTES`` past its peers' addresses). Collective: every rank builds it
    together, ``group`` the gloo group."""

    def __init__(
        self, group, rank: int, size: int, device: torch.device, gamma: int, part: int
    ):
        from sgl_amd_mono.common.peer import PeerBuffer

        self.rank, self.size = rank, size
        slab = MAX_SEQUENCES * gamma * part
        self._buffer = PeerBuffer(EXCHANGE_BYTES + 4 * slab, group, rank, size, device)
        self._buffer.bytes.zero_()
        self.slab = (
            self._buffer.bytes[EXCHANGE_BYTES:]
            .view(torch.float32)
            .view(MAX_SEQUENCES, gamma, part)
        )
        self.peers = self._buffer.addresses.cpu()
        self.calls = torch.zeros(MAX_SEQUENCES, dtype=torch.int32, device=device)
        self.tok = torch.zeros(MAX_SEQUENCES, dtype=torch.int64, device=device)

    def close(self) -> None:
        self._buffer.close()


def blocks(width: int) -> int:
    return -(-width // BLOCK_ROWS)
