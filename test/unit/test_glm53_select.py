"""The GLM-5.3 gfx950 sliced top-k plan on CPU (RFC #43949): a row's top-k runs on up
to INDEX_SEL_SLICES slices, runs of its keys in position order, and the kernel calls
the same ``select_slices`` on traced values. On Python ints: the slices tile a row's
keys once, none empty, each within a slice's LDS keys and its 16-bit counts; the slice
tasks of every width are whole rows of the grid, and the stages run in the order the
selection's hand-offs need."""

import pytest

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.glm53.gfx950 import layout as L
from sgl_amd_mono.glm53.spec import WIDTHS

MAX_SEQ = 1 << 20
KEYS = L.INDEX_SEL_SLICE_KEYS
BOUNDS = [1, 2, 2047, 2048, 2049, KEYS, KEYS + 1, 2 * KEYS, 2 * KEYS + 1, 65535, 65536,
          65537, 131072, 262144 + 48, MAX_SEQ - 1, MAX_SEQ]  # fmt: skip


@pytest.mark.parametrize("bound", BOUNDS + list(range(3, MAX_SEQ, 7919)))
def test_slices_tile_a_rows_keys_once_in_position_order(bound):
    n, per = L.select_slices(bound)
    assert 1 <= n <= L.INDEX_SEL_SLICES
    # slice c takes keys [c * per, min((c + 1) * per, bound)): each nonempty, all taken
    assert (n - 1) * per < bound <= n * per
    # a slice per INDEX_SEL_SLICE_KEYS keys until the most slices
    assert n == min(-(-bound // KEYS), L.INDEX_SEL_SLICES)
    assert n == L.INDEX_SEL_SLICES or per <= KEYS
    # a slice's keys fit its share of the longest row, and its counts 16 bits
    assert per <= -(-MAX_SEQ // L.INDEX_SEL_SLICES) < 1 << 16


def test_an_empty_row_has_one_slice():
    assert L.select_slices(0) == (1, 0)


@pytest.mark.parametrize("rows", WIDTHS)
def test_slice_tasks_are_whole_rows_of_the_grid_between_score_and_split(rows):
    stages = L.stage_tasks(rows, 8, 2048, with_indexer=True, index_max_seq=MAX_SEQ)
    tasks = dict(stages)
    assert tasks["index_slice"] == rows * L.INDEX_SEL_SLICES
    assert BLOCKS % L.INDEX_SEL_SLICES == 0
    assert tasks["index_score"] == BLOCKS
    assert tasks["index_key"] == rows
    order = [name for name, _ in stages]
    chain = ["q_norm", "index_key", "index_q", "index_qh", "index_score",
             "index_slice", "split"]  # fmt: skip
    assert [order.index(name) for name in chain] == sorted(
        order.index(name) for name in chain
    )
