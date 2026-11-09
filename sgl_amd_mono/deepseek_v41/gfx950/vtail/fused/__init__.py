# SPDX-License-Identifier: Apache-2.0
"""A greedy-variant static verify's head and accept on gfx950 in one launch (``fused_tail.hip``):
this rank's vocab projection of the head's normed rows (bf16 MFMAs, the logits shard in the
head's output dtype), the shard's row keys, their exchange between the ranks and SGLang's
static accept tail."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
# verify rows (requests x rows a request) the launch takes; larger verifies keep the
# separate launches
MAX_ROWS = 6
MAX_TP = 8
MAX_T = 8
MAX_BLOCKS = 256  # a block 128 vocab rows of the shard
# fewer blocks leave CUs idle while the shard streams: a TP8 shard (127 blocks) is 39.7 us a launch against 34.0 for
# hipBLASLt's GEMM and the vtail launch (isolated, 6 rows, simulated length), so it keeps those
MIN_BLOCKS = 200
HIDDEN = 6144  # the head's input width at most, a multiple of 256
STEP = 256


@functools.cache
def ops():
    """``torch.ops`` of ``fused_tail.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    return load("sgl_amd_mono_dsv41_ftail", sources, "gfx950")


def blocks(part: int) -> int:
    return -(-part // 128)


def fits(*, part: int, hidden: int, vocab: int) -> bool:
    """A shard of ``part`` vocab rows of ``hidden`` the launch can take."""
    return (
        MIN_BLOCKS <= blocks(part) <= MAX_BLOCKS
        and hidden <= HIDDEN
        and hidden % STEP == 0
        and vocab < 1 << 17
    )


class FusedTail:
    """One TP group's scratch (every rank's uncached buffer: the ranks' records, this rank's
    rows, boxes and slots) and each block's launch count. Collective: every rank builds it
    together, ``group`` the gloo group."""

    def __init__(self, group, rank: int, tp: int, device, *, part: int, width: int,
                 vocab_start: int, vocab: int, deadline_s: float):  # fmt: skip
        """``part``: a rank's shard rows, the same on every rank; ``width``: this rank's
        valid ones, from the global index ``vocab_start``."""
        from sgl_amd_mono.common.peer import PeerBuffer

        assert 1 <= tp <= MAX_TP and blocks(part) <= MAX_BLOCKS and width <= part
        self.rank, self.width, self.vocab_start, self.vocab = (
            rank,
            width,
            vocab_start,
            vocab,
        )
        self.deadline_s = float(deadline_s)
        self.ops = ops()
        self._buffer = PeerBuffer(
            int(self.ops.scratch_bytes()), group, rank, tp, device
        )
        self._buffer.bytes.zero_()
        self.peers = self._buffer.addresses.cpu()
        self.scratch = int(self._buffer.local)
        self.counts = torch.zeros(MAX_BLOCKS, dtype=torch.int32, device=device)

    def __call__(self, x, weight, candidates, seq_lens, verify_lens, gate, *, sim, temps, seed,
                 correct_out, bonus_out, cap_trim_out, commit_out, new_seq_out, out_tokens,
                 gated_out=None, target_out=None):  # fmt: skip
        """The logits shard [M, part] of the verify's normed rows ``x`` [M = bs * T, K] on
        this rank's head shard ``weight``, and static_accept_tail's outputs (rows [:bs]) in
        the out tensors; ``sim`` the simulated length [1] (with ``temps`` [bs] and ``seed``
        [1]), or None."""
        logits = torch.empty(
            (x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device
        )
        self.ops.fused_tail(
            x, weight, logits, candidates, seq_lens, verify_lens, gate, sim, temps, seed,
            correct_out, bonus_out, cap_trim_out, commit_out, new_seq_out, out_tokens,
            gated_out, target_out, self.counts, self.peers, self.scratch, self.rank,
            self.vocab_start, self.width, self.vocab, self.deadline_s,
        )  # fmt: skip
        return logits

    def close(self) -> None:
        self._buffer.close()
