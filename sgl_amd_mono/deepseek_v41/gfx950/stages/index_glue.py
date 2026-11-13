# SPDX-License-Identifier: Apache-2.0
"""An index source's selection glue, run in its K2 on CTAs without attention or wo_a work: what
``indexer.candidate_table``'s launches compute, element for element.

- ``TABLE`` (a producer's K2): the REINDEX layers' candidate table from SGLang's candidate
  blocks, as ``reindex_table``: the blocks with the newest moved last, their virtual 8-row pages
  in the request's page table and each row's compact columns. The REINDEX layers' K2i read them.
- ``RAWS`` (a REINDEX layer's K2): SGLang's raw indices of the layer's top-512, whose page
  indices SGLang's top-k wrote through the candidate table: each compact column's position, as
  ``map_compact_selection``. Nothing in the step reads them.

A wave a row: the work is small, and wave-wide reductions need no LDS."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from sgl_amd_mono.common.ops import ballot, global_ptr, ld_i32, popcount, wave_max
from sgl_amd_mono.common.plan import BLOCKS, WAVES, kernel_sources

from ..indexer import index_plan as ip

SOURCES = kernel_sources(__name__)

TABLE, RAWS = 1, 2
ARGS = (
    "gl_ids", "gl_cols", "gl_ids_out", "gl_table", "gl_bound", "gl_ptab", "gl_ptab_stride",
    "gl_bounds", "gl_owners", "gl_raw",
)  # fmt: skip
UNITS = 64 // ip.BLOCK_ROWS  # a 64-row page's 8-row blocks
PER = ip.TOPK_BLOCKS // 64  # a lane's blocks of a row


def _store(addr, v):
    fx.ptr_store(fx.Int32(v), global_ptr(addr))


@flyc.jit
def _table_row(a, lane, row):
    """``reindex_table``'s outputs of one row, from ``gl_ids`` (unordered, -1 after the kept)."""
    js = [lane + 64 * i for i in range(PER)]
    base = fx.Int64(row) * ip.TOPK_BLOCKS
    cand = [ld_i32(a["gl_ids"], row * ip.TOPK_BLOCKS + j) for j in js]
    kept = fx.Int32(0)
    for i in range_constexpr(PER):
        kept = kept + popcount(ballot(cand[i] >= 0))
    seen = ld_i32(a["gl_bounds"], row)
    newest = (seen - 1) // ip.BLOCK_ROWS
    at = fx.Int32(-1)
    for i in range_constexpr(PER):
        at = fx.max(
            at, ((cand[i] >= 0) & (cand[i] == newest)).select(js[i], fx.Int32(-1))
        )
    at = wave_max(at)
    last = kept - 1
    moved = ld_i32(a["gl_ids"], row * ip.TOPK_BLOCKS + fx.max(last, 0))
    swap = at >= 0
    batch = ld_i32(a["gl_owners"], row)
    for i in range_constexpr(PER):
        j, c = js[i], cand[i]
        c = (swap & (j == at)).select(moved, (swap & (j == last)).select(newest, c))
        live = cand[i] >= 0
        page = ld_i32(a["gl_ptab"], batch * a["gl_ptab_stride"] + fx.max(c, 0) // UNITS)
        at4 = (base + j) * 4
        _store(a["gl_ids_out"] + at4, c)
        _store(a["gl_table"] + at4, live.select(page * UNITS + c % UNITS, fx.Int32(0)))
    tail = swap.select(newest, moved)
    span = fx.min(fx.max(seen - tail * ip.BLOCK_ROWS, 0), ip.BLOCK_ROWS)
    if lane == 0:
        _store(
            a["gl_bound"] + fx.Int64(row) * 4,
            (kept > 0).select(last * ip.BLOCK_ROWS + span, fx.Int32(0)),
        )


@flyc.jit
def _raws_row(a, lane, row):
    """``map_compact_selection``'s raw indices of one row, from its compact columns."""
    for i in range_constexpr(ip.TOPK // 64):
        s = row * ip.TOPK + lane + 64 * i
        col = ld_i32(a["gl_cols"], s)
        valid = col >= 0
        c = fx.max(col, 0)
        blk = ld_i32(a["gl_ids_out"], row * ip.TOPK_BLOCKS + c // ip.BLOCK_ROWS)
        pos = blk * ip.BLOCK_ROWS + c % ip.BLOCK_ROWS
        _store(a["gl_raw"] + fx.Int64(s) * 4, valid.select(pos, fx.Int32(-1)))


@flyc.jit
def run_glue(c, a, kind, rows, first, ctas):
    """This CTA's rows of the glue ``kind``: wave w of glue CTA k takes row k * WAVES + w."""
    k = (c["bid"] - first + BLOCKS) % BLOCKS
    row = k * WAVES + c["wave"]
    if (k < ctas) & (row < rows):
        if const_expr(kind == TABLE):
            _table_row(a, c["lane"], row)
        else:
            _raws_row(a, c["lane"], row)


def build_glue(kind: int, rows: int):
    """The glue ``kind`` of ``rows`` rows as its own launch (``K2iRunner``'s tests run it as the
    K2 that would): a wave a row, ARGS in order."""
    from flydsl.expr.typing import Int32, Int64

    from sgl_amd_mono.common.epoch import kernel_symbol
    from sgl_amd_mono.common.plan import THREADS

    ctas = -(-rows // WAVES)
    name = kernel_symbol("dsv41_mono_glue", s=rows, k=kind)
    keyed = (("kind", kind), ("rows", rows), ("sources", SOURCES))
    stream0 = fx.Stream(None)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def glue(
        gl_ids: Int64, gl_cols: Int64, gl_ids_out: Int64, gl_table: Int64, gl_bound: Int64,
        gl_ptab: Int64, gl_ptab_stride: Int32, gl_bounds: Int64, gl_owners: Int64, gl_raw: Int64,
    ):  # fmt: skip
        _ = keyed
        tid = fx.thread_idx.x
        vals = (
            gl_ids, gl_cols, gl_ids_out, gl_table, gl_bound, gl_ptab, gl_ptab_stride, gl_bounds,
            gl_owners, gl_raw,
        )  # fmt: skip
        c = {"bid": fx.Int32(fx.block_idx.x), "lane": tid % 64, "wave": tid // 64}
        run_glue(c, dict(zip(ARGS, vals)), kind, rows, 0, ctas)

    @flyc.jit
    def launch(
        gl_ids: Int64, gl_cols: Int64, gl_ids_out: Int64, gl_table: Int64, gl_bound: Int64,
        gl_ptab: Int64, gl_ptab_stride: Int32, gl_bounds: Int64, gl_owners: Int64, gl_raw: Int64,
        stream: fx.Stream = stream0,
    ):  # fmt: skip
        _ = keyed
        glue(
            gl_ids,
            gl_cols,
            gl_ids_out,
            gl_table,
            gl_bound,
            gl_ptab,
            gl_ptab_stride,
            gl_bounds,
            gl_owners,
            gl_raw,
        ).launch(grid=(ctas,), block=(THREADS,), stream=stream)

    return launch
