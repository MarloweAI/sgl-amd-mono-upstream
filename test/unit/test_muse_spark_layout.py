"""The Muse Spark gfx950 layouts on CPU: the CTA map of K1's stages does not
overlap, the scratch regions do not overlap, and the LDS fits at every width."""

import pytest

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.muse_spark.gfx950 import layout as L
from sgl_amd_mono.muse_spark.gfx950.config import HIDDEN, MAX_TOKENS, QKVG_ROWS, WIDTHS

LDS_BYTES = 160 * 1024


def test_k1_tasks_fit_the_grid():
    # 36 row groups of 64 rows x 8 K parts: a task a CTA, a second on 32 CTAs past the head / gate tasks
    assert (
        L.GEMV_GROUPS * L.G_ROWS == QKVG_ROWS
        and L.GEMV_TASKS == BLOCKS
        and L.SECOND_TASKS == 0
    )
    assert L.GT0 == L.HT0 + L.HEAD_TASKS and L.QT0 == L.GT0 + L.GATE_TASKS
    assert L.K_PARTS * L.K_PER_WAVE * 8 == HIDDEN and L.PARTS == L.K_PARTS == 16


@pytest.mark.parametrize("s", list(WIDTHS) + [1, 5, 24, 63])
def test_k1_lds_fits(s):
    assert L.k1_lds_bytes(s) <= LDS_BYTES
    assert L.token_tiles(s) * L.TILE >= s
    assert L.token_passes(s) * L.TOKENS_PER_PASS >= s


def test_k1_scratch_regions_do_not_overlap():
    lay = L.k1_scratch_layout()
    spans = sorted((off, off + n) for off, n in lay.values())
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0
    assert lay["k1_part"][1] == L.PARTS * MAX_TOKENS * QKVG_ROWS * 8
    assert L.scratch_bytes(lay) == spans[-1][1]


@pytest.mark.parametrize("nt", [1, 2])
def test_kattn_lds_fits(nt):
    assert L.kattn_lds_bytes(nt) <= LDS_BYTES


@pytest.mark.parametrize("s,ql", [(8, 8), (64, 8), (16, 1)])
def test_kattn_scratch_covers_every_partial(s, ql):
    lay = L.kattn_scratch_layout(s, ql)
    rows = L.attn_rows(ql)
    assert (
        lay["attn_part"][1]
        == (s // ql) * 2 * L.ATTN_MAX_PARTS * rows * L.ATTN_PART_ROW * 4
    )  # fp32 words
    tiles = (rows + 15) // 16
    assert (
        lay["attn_pflag"][1] == (s // ql) * 2 * tiles * L.ATTN_MAX_PARTS * 8
    )  # a flag a task
