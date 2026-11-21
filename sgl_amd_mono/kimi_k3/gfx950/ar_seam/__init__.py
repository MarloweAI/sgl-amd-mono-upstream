# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's o_proj all-reduce and the FFN-side attention-residual read in one launch on gfx950 at TP8
(``ar_seam.hip``): aiter's own all-reduce blocks (its two-shot up to 1 MiB, its 2-stage above) hand
each summed row to one reader block, which runs SGLang's ``attn_res_hip._agg_kernel`` with Triton's
numerics. Outputs are bitwise the stock all-reduce followed by ``attn_res_hip``."""

from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
HIDDEN = 7168
MAX_ROWS = 192
HANDOFF_BYTES = MAX_ROWS * HIDDEN * 2
# lamportPollSlot's empty word: every byte 0xff
SENTINEL = 0xFF


def aiter_include() -> str:
    spec = importlib.util.find_spec("aiter")
    return str(Path(spec.origin).resolve().parent.parent / "csrc" / "include")


@functools.cache
def ops(flags: tuple[str, ...] = ()):
    """``torch.ops`` of ``ar_seam.hip``, built at first use against the installed aiter's headers; ``flags``
    are a test build's defines (``-DAR_SEAM_STAMPS``: per-row timestamps after the hand-off buffer)."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    base = (f"-I{aiter_include()}", "-ffp-contract=off")
    return load("sgl_amd_mono_kimi_k3_ar_seam", sources, "gfx950", base + tuple(flags))


def handoff(device, extra: int = 0) -> object:
    """A rank-local uncached hand-off buffer, armed, with ``extra`` zero bytes after it (a stamps
    build's); the ``PeerBuffer`` keeps the memory."""
    from sgl_amd_mono.common.peer import PeerBuffer

    def arm(b):
        b[:HANDOFF_BYTES].fill_(SENTINEL)

    return PeerBuffer(HANDOFF_BYTES + extra, None, 0, 1, torch.device(device), init=arm)


def plan(rows: int) -> tuple[int, int, int]:
    """(algorithm, all-reduce blocks, count) of aiter's all-reduce for ``rows`` x 7168 bf16 at TP8:
    algorithm 0 the two-shot, 1 the 2-stage, 2 the 2-stage with a tail."""
    return tuple(ops().plan_of(rows)[:3])


class ArSeam:
    """One rank's fused o_proj all-reduce + FFN read on aiter's communicator ``ca`` (aiter's Python
    ``CustomAllreduce``, the one SGLang's TP group reduces with) and its hand-off buffer. Built once per
    rank, outside graph capture."""

    def __init__(self, ca, device):
        self.ca = ca
        self.buf = handoff(device)

    def read(self, x, res, bank, nvb: int, cw, ow, score_eps: float, out_eps: float, write: bool = False):
        """(out, prefix): the FFN input (the output RMSNorm of the bank's mix) and the written residual, from
        this rank's o_proj partial ``x`` [m, 7168], the pending residual ``res`` (or None) and the bank
        [m, blocks, 7168] (its first ``nvb`` snapshots; ``write`` snapshots the prefix into row ``nvb``).
        The input takes aiter's path: registered in its graph capture, else copied into its pool."""
        ca = self.ca
        registered = ca._IS_CAPTURING and torch.cuda.is_current_stream_capturing() and ca._capture_registered(x)
        pool = ca._pool["input"]
        out = torch.empty_like(x)
        pre = torch.empty_like(x)
        ops().seam(
            ca._ptr, x, res if res is not None else x, bank, cw, ow, pre, out, self.buf.bytes, nvb,
            res is not None, write, score_eps, out_eps, 0 if registered else pool.data_ptr,
            0 if registered else pool.max_size,
        )  # fmt: skip
        return out, pre
