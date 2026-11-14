# SPDX-License-Identifier: Apache-2.0
"""A REINDEX layer's candidate logits at 60-384 rows: K2i's FP4 scorer (``score_fp4``) over each
row's own candidate blocks, in one launch of ``cls * per`` CTAs for the rows of a width class.

SGLang's wide-row consumer scores every compressed key of its request on aiter's row-group
kernel and then gathers the candidate blocks' columns; this launch scores only those columns,
2048 blocks of 8 a row, so its work does not grow with the context. Each CTA scores one row's
chunks ``reindex_chunks(i, per)`` (``index_plan``) through the row's candidate table (a virtual
8-row page an entry, ``candidate_table.reindex_table``) into its compact logits row; the per-key
arithmetic is the row-group kernel's, so a column's logit is SGLang's bit for bit.

A request's rows mostly share candidate blocks, so their CTAs go to one XCD (whose L2 they share):
workgroups reach the XCDs round robin (CTA b on XCD b % 8), and each XCD takes a contiguous eighth
of the step's tasks in row order."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common.plan import THREADS, kernel_sources

from . import index_plan as ip
from .score_fp4 import REINDEX_TASK_CHUNKS, ROWS_LDS_WORDS, _read_all, stage_iscore4

SOURCES = kernel_sources(__name__)

_STREAM = fx.Stream(None)
# CTAs a launch aims at: three a CU of 256
TARGET_CTAS = 768
XCDS = 8


def tasks_per_row(rows: int) -> int:
    """A row's CTAs: as many as keep the grid near TARGET_CTAS, each at least a step a wave."""
    return max(1, min(TARGET_CTAS // rows, ip.REINDEX_CHUNKS // REINDEX_TASK_CHUNKS))


@functools.cache
def build(cls: int):
    """The launch for up to ``cls`` rows (a width class); the step's ``rows`` is an argument.
    One a class: a graph capture must not drop (and so unload) a launch's module."""
    per = tasks_per_row(cls)
    assert cls * per % XCDS == 0, (cls, per)
    name = f"ds41_idx_wide_reindex_r{cls}_p{per}"
    keyed = (("cls", cls), ("per", per), ("sources", SOURCES))

    @fx.struct
    class RowsLds:
        rows: fx.Array[fx.Int32, ROWS_LDS_WORDS, 16]

    @fx.union
    class ReindexLds:
        scorer: RowsLds

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def reindex_wide(
        iq: Int64, iqs: Int64, iw: Int64, plane: Int64, pscale: Int64, itab: Int64,
        ibound: Int64, ilog: Int64, rows: Int32,
    ):  # fmt: skip
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(ReindexLds).scorer.peek()
        # score_fp4's arguments: a candidate table of TOPK_BLOCKS virtual pages a row, the
        # compact logits rows COMPACT columns apart
        args = {
            "iq": iq, "iqs": iqs, "iw": iw, "plane": plane, "pscale": pscale,
            "itab": itab, "itab_stride": fx.Int32(ip.TOPK_BLOCKS),
            "itab_len": fx.Int32(ip.TOPK_BLOCKS), "ibatch": ibound, "ishift": fx.Int32(0),
            "ibound": ibound, "ilog": ilog, "lstride": fx.Int32(ip.TOPK_BLOCKS * ip.BLOCK_ROWS),
        }  # fmt: skip
        c = {
            "tid": tid, "lane": tid % 64, "wave": tid // 64, "args": args,
            "rows_lds": lds.rows.ptr, "blocks": False,
        }  # fmt: skip
        _read_all([iq, iqs, iw, plane, pscale, itab, ibound, ilog, rows])
        # task v of the step's rows * per: XCD bid % 8's share is the eighth from its index
        share = (rows * per + XCDS - 1) // XCDS
        k = bid // XCDS
        v = (bid % XCDS) * share + k
        if (k < share) & (v < rows * per):
            ch0, ch1 = ip.reindex_chunks(v % per, per)
            stage_iscore4(c, v // per, fx.Int32(1), ch0, ch1, 1)

    @flyc.jit
    def launch(
        iq: Int64, iqs: Int64, iw: Int64, plane: Int64, pscale: Int64, itab: Int64,
        ibound: Int64, ilog: Int64, rows: Int32, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        reindex_wide(iq, iqs, iw, plane, pscale, itab, ibound, ilog, rows).launch(
            grid=(cls * per,), block=(THREADS,), stream=stream
        )

    return launch
