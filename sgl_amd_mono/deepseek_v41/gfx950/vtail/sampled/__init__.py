# SPDX-License-Identifier: Apache-2.0
"""A temperature-sampled static verify's accept from the ranks' bf16 logits shards and the
draft's f32 step logits shards on gfx950 (``sampled_tail.hip``) in one launch: SGLang's
block-wise rejection with its draws, its simulated bonus and its static accept tail, the
ranks' statistics and keys swapped through peer buffers, no whole-vocab row anywhere."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
MAX_TP = 8
MAX_SEQ = 4  # requests
MAX_T = 8  # verify rows a request
MAX_BLOCKS = 128  # blocks a rank
CHUNK = 512  # a block's columns of the shard


@functools.cache
def ops():
    """``torch.ops`` of ``sampled_tail.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    proj = (
        _HERE.parents[1] / "inject" / "proj" / "main_proj.h"
    )  # main_proj's CTAs, for sampled_tail_proj
    sources[proj.name] = proj.read_text()
    return load("sgl_amd_mono_dsv41_stail", sources, "gfx950")


def blocks(width: int) -> int:
    return -(-width // CHUNK)


def fits(*, tp: int, width: int, vocab: int) -> bool:
    return (
        1 <= tp <= MAX_TP
        and blocks(width) <= MAX_BLOCKS
        and width % 2 == 0
        and vocab < 1 << 17
    )


class SampledTail:
    """One TP group's records (every rank's peer buffer), this rank's scratch (the blocks'
    partials and their arrival count) and each block's launch count. Collective: every rank
    builds it together, ``group`` the gloo group."""

    def __init__(self, group, rank: int, tp: int, device, *, width: int, vocab_start: int,
                 vocab: int, deadline_s: float):  # fmt: skip
        """``width``: this rank's valid shard columns, from the global index ``vocab_start``;
        every rank's shard has as many blocks."""
        from sgl_amd_mono.common.peer import PeerBuffer

        assert fits(tp=tp, width=width, vocab=vocab)
        self.rank, self.width, self.vocab_start, self.vocab = (
            rank,
            width,
            vocab_start,
            vocab,
        )
        self.deadline_s = float(deadline_s)
        self.ops = ops()
        self._buffer = PeerBuffer(int(self.ops.record_bytes()), group, rank, tp, device)
        self._buffer.bytes.zero_()
        self.peers = self._buffer.addresses.cpu()
        self.scratch = torch.zeros(
            int(self.ops.scratch_bytes()), dtype=torch.uint8, device=device
        )
        self.counts = torch.zeros(blocks(width), dtype=torch.int32, device=device)

    def __call__(self, local, draft, temps, *, key, counter, candidates, seq_lens, verify_lens,
                 gate, sim, sim_seed, correct_out, bonus_out, cap_trim_out, commit_out,
                 new_seq_out, out_tokens, gated_out=None, proj=None, commit=None) -> None:  # fmt: skip
        """static_accept_tail's outputs (rows [:bs]) of SGLang's block-wise rejection on this
        rank's shards: ``local`` [bs * T, part] bf16 of the verify's logits, ``draft``
        [>= bs, >= T - 1, width] f32 of the draft's step logits (pre-temperature), ``temps``
        [bs] (> 0) the requests' temperatures, ``key`` and ``counter`` the rejection noise.
        ``sim`` the simulated length [1] (with ``sim_seed`` [1]: the bonus drawn at that row
        as SGLang's simulated bonus), or None. ``proj``: (x, w, ws, exchange), the draft's
        main_proj of rows ``x`` on this rank's columns ``w``, ``ws`` in the same launch, into
        ``exchange.out`` (``gfx950/inject/proj``'s Exchange of this group). ``commit``:
        (history, slots, relay_lens, relay_tokens), SGLang's engram_commit_history of the
        accept in the same launch."""
        tail = commit if commit is not None else (None,) * 4
        args = (
            local, draft, temps, int(key), counter, sim, sim_seed, candidates, seq_lens,
            verify_lens, gate, correct_out, bonus_out, cap_trim_out, commit_out, new_seq_out,
            out_tokens, gated_out, self.scratch, self.counts, self.peers, self.rank, self.vocab_start,
            self.width, self.vocab, self.deadline_s,
        )  # fmt: skip
        if proj is None:
            self.ops.sampled_tail(*args, *tail)
            return
        x, w, ws, ex = proj
        assert ex.rank == self.rank
        self.ops.sampled_tail_proj(
            *args, x, w, ws, ex.out, ex.state, ex.work, ex.peers, ex.deadline_s, *tail
        )  # fmt: skip

    def close(self) -> None:
        self._buffer.close()
