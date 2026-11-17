"""K2i's per-step score tasks (``index_plan``, which the kernel traces too) on CPU (RFC #43949).

On random steps: the host walk plan fits the grid and covers every scored row's chunks exactly
once, and a candidate consumer's tasks tile its chunks."""

import numpy as np
import pytest

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.deepseek_v41.gfx950.indexer import index_plan as ip

BOUNDS = [0, 1, 300, 511, 512, 513, 640, 8191, 8192, 8193, 16384, 16385, 20000,
          131071, 131072, 131073, 524288, (1 << 20) + 96]  # fmt: skip


def _step(*, rng, n):
    """A step's rows: requests of 1 (decode), 6 (DSpark verify) or ragged rows,
    each request's rows seeing one more column than the row before."""
    kind = rng.integers(3)
    owners, bounds, row = [], [], 0
    while row < n:
        size = 1 if kind == 0 else 6 if kind == 1 else int(rng.integers(1, 9))
        size = min(size, n - row)
        ctx = int(rng.choice(BOUNDS + [int(rng.integers(0, 1 << 20))]))
        owners += [row] * size
        bounds += [max(ctx - (size - 1 - r), 0) for r in range(size)]
        row += size
    return np.array(bounds, dtype=np.int64), np.array(owners, dtype=np.int64)


@pytest.mark.parametrize("seed", range(3))
def test_walk_plan_covers_each_scored_chunk_once(seed):
    rng = np.random.default_rng(seed)
    for _ in range(300):
        n = int(rng.integers(1, 49))
        bounds, owners = _step(rng=rng, n=n)
        plan = np.full((BLOCKS, 4), -1, dtype=np.int64)
        ip.fill_walk_plan(bounds=bounds, owners=owners, plan=plan)
        live = plan[:, 1] > 0
        k = int(live.sum())
        assert live[:k].all() and (plan[k:] == 0).all(), plan[: k + 2]
        covered = {}
        for lead, span, ch0, ch1 in plan[:k].tolist():
            assert 0 < span <= ip.WALK_ROWS and 0 <= ch0 < ch1, (lead, span, ch0, ch1)
            assert len(set(owners[lead : lead + span].tolist())) == 1  # one table
            covered.setdefault((lead, span), []).append((ch0, ch1))
        pieces = {}
        for (lead, span), ranges in covered.items():
            ranges.sort()
            # the piece's tasks tile its chunks: [0, its rows' most) exactly once
            need = max(ip.score_chunks(int(b)) for b in bounds[lead : lead + span])
            assert ranges[0][0] == 0 and ranges[-1][1] == need, (ranges, need)
            assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])), ranges
            for t in range(lead, lead + span):
                assert t not in pieces, t  # each row in one piece
                pieces[t] = need
        for t, b in enumerate(bounds.tolist()):
            # a scored row's chunks are all walked; a row seeing <= TOPK is not scored
            if ip.scored(b):
                assert pieces.get(t, 0) >= ip.score_chunks(b), (t, b)


@pytest.mark.parametrize("rows", range(1, 49))
def test_a_consumer_tiles_its_candidate_chunks(rows):
    per = ip.reindex_split(rows, BLOCKS, 4)  # a wave's step at least a task
    assert rows * per <= BLOCKS
    spans = [ip.reindex_chunks(i, per) for i in range(per)]
    assert spans[0][0] == 0 and spans[-1][1] == ip.REINDEX_CHUNKS
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert all(ch1 - ch0 >= 4 for ch0, ch1 in spans)


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_cp_chunks_tile_each_row(tp):
    """Past the band the ranks' chunks of a row tile its scored chunks in rank order; below it
    every rank takes them all."""
    for b in BOUNDS + list(range(500, 2000, 37)):
        n = ip.score_chunks(b)
        runs = [ip.cp_chunks(b, r, tp, 513) for r in range(tp)]
        if b >= 513:
            assert runs[0][0] == 0 and runs[-1][1] == n, (b, runs)
            assert all(a[1] == c[0] for a, c in zip(runs, runs[1:])), (b, runs)
        else:
            assert all(run == (0, n) for run in runs), (b, runs)
        assert ip.cp_chunks(b, tp - 1, tp, ip.NO_CP) == (0, n)


@pytest.mark.parametrize("seed", range(3))
def test_walk_plan_covers_each_ranks_chunks_once(seed):
    """A rank's plan walks every row's run of chunks (``cp_chunks``), each chunk of a piece once,
    and nothing outside its rows' runs."""
    rng = np.random.default_rng(100 + seed)
    for _ in range(200):
        n = int(rng.integers(1, 49))
        bounds, owners = _step(rng=rng, n=n)
        tp = int(rng.choice([2, 4, 8]))
        rank = int(rng.integers(tp))
        band = int(rng.choice([513, 4096, 1 << 16, int(rng.integers(0, 1 << 20))]))
        plan = np.full((BLOCKS, 4), -1, dtype=np.int64)
        ip.fill_walk_plan(bounds=bounds, owners=owners, plan=plan, cp=(rank, tp, band))
        k = int((plan[:, 1] > 0).sum())
        assert (plan[k:] == 0).all()
        walked = {}
        for lead, span, ch0, ch1 in plan[:k].tolist():
            assert 0 < span <= ip.WALK_ROWS and 0 <= ch0 < ch1
            for t in range(lead, lead + span):
                walked.setdefault(t, []).append((ch0, ch1))
        for t, b in enumerate(bounds.tolist()):
            lo, hi = ip.cp_chunks(b, rank, tp, band)
            got = sorted(walked.get(t, []))
            if hi > lo:
                assert got and got[0][0] <= lo and got[-1][1] >= hi, (t, b, got, lo, hi)
                assert all(a[1] == c[0] for a, c in zip(got, got[1:])), got
