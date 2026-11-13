# SPDX-License-Identifier: Apache-2.0
"""The static verify's accept from the ranks' bf16 logits shards on gfx950 (``vtail.hip``)
in one launch: each rank's blocks key their columns of the shard's rows, every rank's
records reach every rank through peer buffers, and each rank takes each row's best (the
lowest vocab index among the maxima) and runs SGLang's static accept tail on it."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
MAX_TP = 8
MAX_ROWS = 48  # key rows: a verify's rows at most
MAX_T = 8
MAX_BLOCKS = 128
CHUNK = 512  # a block's columns of the shard
# one 8 B record a key row, source rank and block, two launch parities
RECORD_BYTES = 2 * MAX_ROWS * MAX_TP * MAX_BLOCKS * 8


@functools.cache
def ops():
    """``torch.ops`` of ``vtail.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    proj = (
        _HERE.parent / "inject" / "proj" / "main_proj.h"
    )  # main_proj's CTAs, for verify_tail_proj
    sources[proj.name] = proj.read_text()
    return load("sgl_amd_mono_dsv41_vtail", sources, "gfx950")


def blocks(part: int) -> int:
    return -(-part // CHUNK)


class VerifyTail:
    """One TP group's records (every rank's peer buffer) and each block's launch count.
    Collective: every rank builds it together, ``group`` the gloo group."""

    def __init__(self, group, rank: int, tp: int, device, *, part: int, width: int,
                 vocab_start: int, vocab: int, deadline_s: float):  # fmt: skip
        """``part``: a rank's shard columns, the same on every rank; ``width``: this
        rank's valid ones, from the global index ``vocab_start``."""
        from sgl_amd_mono.common.peer import PeerBuffer

        assert 1 <= tp <= MAX_TP and blocks(part) <= MAX_BLOCKS and width <= part
        self.rank, self.width, self.vocab_start, self.vocab = (
            rank,
            width,
            vocab_start,
            vocab,
        )
        self.deadline_s = float(deadline_s)
        self._buffer = PeerBuffer(RECORD_BYTES, group, rank, tp, device)
        self._buffer.bytes.zero_()
        self.peers = self._buffer.addresses.cpu()
        self.counts = torch.zeros(MAX_BLOCKS, dtype=torch.int32, device=device)
        self.ops = ops()

    def __call__(self, local, candidates, seq_lens, verify_lens, gate, *, sim, temps,
                 seed, correct_out, bonus_out, cap_trim_out, commit_out, new_seq_out,
                 out_tokens, gated_out=None, target_out=None, proj=None, commit=None) -> None:  # fmt: skip
        """static_accept_tail's outputs (rows [:bs]) from this rank's shard ``local``
        [bs * T, width] bf16 of the verify's logits; ``sim`` the simulated length [1]
        (with ``temps`` [bs] and ``seed`` [1]: the bonus drawn at that row as SGLang's
        simulated bonus), or None (the greedy accept). ``proj``: (x, w, ws, exchange), the
        draft's main_proj of rows ``x`` on this rank's columns ``w``, ``ws`` in the same
        launch, into ``exchange.out`` (``gfx950/inject/proj``'s Exchange of this group).
        ``commit``: (history, slots, relay_lens, relay_tokens), SGLang's engram_commit_history of
        the accept in the same launch (relay rows None: none stored)."""
        args = (
            local, candidates, seq_lens, verify_lens, gate, sim, temps, seed,
            correct_out, bonus_out, cap_trim_out, commit_out, new_seq_out, out_tokens,
            gated_out, target_out, self.counts, self.peers, self.rank,
            self.vocab_start, self.width, self.vocab, CHUNK, self.deadline_s,
        )  # fmt: skip
        tail = commit if commit is not None else (None,) * 4
        if proj is None:
            self.ops.verify_tail(*args, *tail)
            return
        x, w, ws, ex = proj
        assert ex.rank == self.rank
        self.ops.verify_tail_proj(
            *args, x, w, ws, ex.out, ex.state, ex.work, ex.peers, ex.deadline_s, *tail
        )  # fmt: skip

    def close(self) -> None:
        self._buffer.close()
