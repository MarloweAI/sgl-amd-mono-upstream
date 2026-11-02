"""The token-parallel row gathers' pair layout and line alignment, on CPU."""

import pytest

from sgl_amd_mono.deepseek_v41.gfx950.attention.plan import Q_RANK
from sgl_amd_mono.deepseek_v41.gfx950.stages.seam import HIDDEN
from sgl_amd_mono.deepseek_v41.gfx950.token_par import (
    LINE,
    chunk_pairs,
    half_pair,
    peer_layout,
)
from sgl_amd_mono.deepseek_v41.spec import SPEC

WIDTHS = SPEC.devices["gfx950"].widths


@pytest.mark.parametrize("n", [HIDDEN // 2, Q_RANK // 2])
def test_the_gather_inverts_the_push(n):
    stored = {
        half_pair(n, c, h) + j: 4 * c + 2 * h + j
        for c in range(n // 4)
        for h in range(2)
        for j in range(2)
    }
    assert sorted(stored) == sorted(stored.values()) == list(range(n))
    for q in range(n // 2):
        assert stored[2 * q] == chunk_pairs(n, q)
        assert stored[2 * q + 1] == chunk_pairs(n, q) + 1


@pytest.mark.parametrize("tp", [2, 4])
def test_peer_regions_start_on_a_line(tp):
    for s in WIDTHS:
        assert all(off % LINE == 0 for off, _ in peer_layout(s, tp).values())
    # the other parity's regions start a half (the widest layout) later
    widest = peer_layout(max(WIDTHS), tp)
    assert max(off + size for off, size in widest.values()) % LINE == 0
