# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's sampled DSpark Markov walk on gfx950 (``markov.hip``) in one launch: each step's bias
GEMV, bf16 step logits, Gumbel-max pick and the ranks' bests swapped through peer buffers, every
rank's token alike without a collective; the step logits stay on each rank's vocab shard, and
``gather_step_logits`` copies every rank's shard into the whole rows. A rank's next walk rewrites its
shard, so the peers' gathers of it must be done by then (in a decode cycle, the target verify's
collectives between two draft blocks order them)."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
DIM = 256  # the Markov rank the kernels hold in registers
MAX_SEQUENCES = 8
MAX_STEPS = 8
MAX_TP = 8
MAX_BLOCKS = 256
BLOCK_ROWS = 256
# every block's step record, 8 B each [launch parity][step][sequence][source rank][block]; the rank's bf16 step
# logits follow
EXCHANGE_BYTES = 2 * MAX_STEPS * MAX_SEQUENCES * MAX_TP * MAX_BLOCKS * 8


@functools.cache
def ops(flags: tuple[str, ...] = ()):
    """``torch.ops`` of ``markov.hip``, built at first use; ``flags`` are a test build's defines
    (``-DMARKOV_LOOPBACK``: every peer address is this rank's buffer, one GPU)."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    return load("sgl_amd_mono_kimi_k3_markov", sources, "gfx950", flags)


def slab_bytes(gamma: int, part: int) -> int:
    return MAX_SEQUENCES * gamma * part * 2


class Exchange:
    """One TP group's exchange: every rank's buffer address (``peers``, int64 on the host), the
    launch count of each block (``control``), the last tokens, and this rank's step logits
    (``slab``, [MAX_SEQUENCES, gamma, part] bf16, which every rank reads at ``EXCHANGE_BYTES``
    past its peers' addresses). Collective: every rank builds it together, ``group`` the gloo
    group."""

    def __init__(self, group, rank: int, size: int, device: torch.device, gamma: int, part: int):
        from sgl_amd_mono.common.peer import PeerBuffer

        self.rank, self.size = rank, size
        self._buffer = PeerBuffer(EXCHANGE_BYTES + slab_bytes(gamma, part), group, rank, size, device)
        self.peers = self._buffer.addresses.cpu()
        self._finish(device, gamma, part)

    def _finish(self, device, gamma, part):
        self.slab = (
            self._buffer.bytes[EXCHANGE_BYTES:].view(torch.bfloat16).view(MAX_SEQUENCES, gamma, part)
        )
        self.control = torch.zeros(blocks(part), dtype=torch.int32, device=device)
        self.tok = torch.zeros(MAX_SEQUENCES, dtype=torch.int64, device=device)

    def close(self) -> None:
        self._buffer.close()


class Loopback(Exchange):
    """One GPU (tests and timing, with ``ops(("-DMARKOV_LOOPBACK",))``): every one of the ``size``
    peer addresses is this rank's own buffer, which then holds every source rank's records."""

    def __init__(self, size: int, device: torch.device, gamma: int, part: int):
        from sgl_amd_mono.common.peer import PeerBuffer

        self.rank, self.size = 0, size
        self._buffer = PeerBuffer(EXCHANGE_BYTES + slab_bytes(gamma, part), None, 0, 1, device)
        self.peers = self._buffer.addresses.cpu().repeat(size)
        self._finish(device, gamma, part)


def blocks(width: int) -> int:
    return -(-width // BLOCK_ROWS)
