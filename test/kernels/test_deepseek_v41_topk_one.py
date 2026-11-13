"""The one-launch paged top-k (``indexer/topk_one``) against SGLang's ``topk_transform_paged_hip``
on one gfx950: per row, the same selected set and the same (raw, page) pairs, eager and in a
graph replayed many times with calls of different widths, the launch's counters carried over.

Rows are compared as sets: both leave a row's order to their atomics."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx950 GPU",
)

K = 512
_ONE = []


def _one():
    from sgl_amd_mono.deepseek_v41.gfx950.indexer.topk_one import TopKOne

    if not _ONE:
        _ONE.append(TopKOne(torch.device("cuda")))
    return _ONE[0]


def _case(*, bounds, width, seed, page_size=64, ties=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = len(bounds)
    scores = torch.randn(rows, width, generator=g)
    if ties:  # a coarse grid: many equal values, across the threshold bin
        scores = (scores * 8).round() / 8
    table = torch.randint(
        0, 1 << 18, (rows, -(-width // page_size)), generator=g, dtype=torch.int32
    )
    return {
        "scores": scores.cuda(),
        "seq_lens": torch.tensor(bounds, dtype=torch.int32).cuda(),
        "page_table": table.cuda(),
        "page_size": page_size,
    }


def _outs(rows):
    return (torch.full((rows, K), -7, dtype=torch.int32, device="cuda"),
            torch.full((rows, K), -7, dtype=torch.int32, device="cuda"))  # fmt: skip


def _sglang(case, outs):
    from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
        topk_transform_paged_hip,
    )

    topk_transform_paged_hip(page_indices=outs[0], raw_indices=outs[1], **case)


def _mine(case, outs):
    _one().transform_paged(page_indices=outs[0], raw_indices=outs[1], **case)


def _pairs(outs):
    """Each row's (raw, page) pairs, ascending by raw, -1 padding last."""
    page, raw = outs
    big = torch.iinfo(torch.int32).max
    key = torch.where(raw >= 0, raw, big)
    order = key.argsort(dim=1)
    return torch.stack([raw.gather(1, order), page.gather(1, order)])


def _same(case, got, want, *, exact_ties=True):
    """Same sets of (raw, page) pairs; with ``exact_ties`` False, only rows without values equal
    to the k-th are compared whole (SGLang places equal values by arrival past 128 ties)."""
    a, b = _pairs(got), _pairs(want)
    if exact_ties:
        return torch.equal(a, b)
    ok = True
    for r in range(a.shape[1]):
        if not torch.equal(a[:, r], b[:, r]):
            n = int(case["seq_lens"][r])
            v = case["scores"][r, :n]
            kth = v.topk(min(K, n)).values[-1]
            sel_a = v[a[0, r][a[0, r] >= 0].long()]
            sel_b = v[b[0, r][b[0, r] >= 0].long()]
            # differing only in which equal-to-kth values were taken
            ok &= bool(torch.equal(sel_a.sort().values, sel_b.sort().values)) and bool(
                (sel_a.min() == kth)
            )
    return ok


CASES = {
    "6x320k": ([164112] * 6, 165888),
    "6x8k": ([4331, 4330, 4329, 4328, 4327, 4326], 165888),
    "6x8k_narrow": ([4331] * 6, 8192),
    "mixed": ([164112, 300, 0, 512, 40960, 40961], 165888),
    "8x_long": ([524288, 400000, 300000, 200000, 100000, 50000, 41000, 600000], 600064),
    "1x320k": ([328224], 328256),
    "4x160k": ([163840, 163839, 120000, 90000], 165888),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("seed", range(3))
def test_equals_sglang(name, seed):
    bounds, width = CASES[name]
    case = _case(bounds=bounds, width=width, seed=seed)
    want, got = _outs(len(bounds)), _outs(len(bounds))
    _sglang(case, want)
    for _ in range(3):  # the counters carry over launches
        got[0].fill_(-7)
        got[1].fill_(-7)
        _mine(case, got)
        torch.cuda.synchronize()
        assert _same(case, got, want), name


def test_every_launch_leaves_the_workspace_zeroed():
    """Each row's last workgroup resets its counters and histogram, so no count grows over launches
    (a counter that only grew would wrap on a long-running server)."""
    one = _one()
    for name in ("6x320k", "mixed", "8x_long", "6x8k"):
        bounds, width = CASES[name]
        case = _case(bounds=bounds, width=width, seed=21)
        outs = _outs(len(bounds))
        for _ in range(3):
            _mine(case, outs)
            torch.cuda.synchronize()
            assert not one.ctr.any() and not one.hist.any(), name


@pytest.mark.parametrize("page_size", [8, 64])
def test_ties_and_page_sizes(page_size):
    case = _case(bounds=[164112, 4331, 100000, 90000, 2000, 160000], width=165888, seed=5,
                 page_size=page_size, ties=True)  # fmt: skip
    want, got = _outs(6), _outs(6)
    _sglang(case, want)
    _mine(case, got)
    torch.cuda.synchronize()
    assert _same(case, got, want, exact_ties=False)


def test_graph_replay_carries_counters():
    """One graph of calls of different widths and row counts, replayed 600 times over two data
    sets swapped in between replays, equals SGLang's selection every replay."""
    names = ["6x320k", "6x8k", "mixed", "8x_long", "1x320k", "6x8k_narrow"]
    cases = [_case(bounds=CASES[n][0], width=CASES[n][1], seed=11) for n in names]
    alt = [_case(bounds=CASES[n][0], width=CASES[n][1], seed=12) for n in names]
    data = [[c["scores"].clone() for c in cases], [c["scores"].clone() for c in alt]]
    want = []
    for d in data:
        for c, s in zip(cases, d):
            c["scores"].copy_(s)
        w = [_outs(len(CASES[n][0])) for n in names]
        for c, o in zip(cases, w):
            _sglang(c, o)
        want.append([_pairs(o) for o in w])
    got = [_outs(len(CASES[n][0])) for n in names]

    def step():
        for c, o in zip(cases, got):
            _mine(c, o)

    step()  # build outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    replays = int(os.environ.get("TOPK_ONE_REPLAYS", "600"))
    bad = 0
    for i in range(replays):
        d = i % 2
        for c, s in zip(cases, data[d]):
            c["scores"].copy_(s)
        for o in got:
            o[0].fill_(-7)
            o[1].fill_(-7)
        graph.replay()
        bad += sum(int(not torch.equal(_pairs(o), w)) for o, w in zip(got, want[d]))
    torch.cuda.synchronize()
    assert bad == 0, f"{bad} mismatched calls over {replays} replays"


KB = 2048
# a producer's columns and its 8-column blocks: (column bounds, column width, block bounds, block width)
PAIRS = {
    "6x320k": ([328224] * 6, 328256, [41028] * 6, 41032),
    "6x128k": ([131072] * 6, 328256, [16384] * 6, 41032),
    "6x8k": (
        [8662, 8661, 8660, 8659, 8658, 8657],
        328256,
        [1083, 1083, 1083, 1083, 1083, 1083],
        41032,
    ),
    "mixed": (
        [328224, 300, 0, 4096, 40961, 200000],
        328256,
        [41028, 38, 0, 512, 5121, 25000],
        41032,
    ),
    "8x_long": (
        [600000, 500000, 400000, 328000, 100000, 50000, 41000, 300],
        600064,
        [75000, 62500, 50000, 41000, 12500, 6250, 5125, 38],
        75008,
    ),
}


def _pair_case(name, seed):
    bounds, width, bbounds, bwidth = PAIRS[name]
    case = _case(bounds=bounds, width=width, seed=seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 1000)
    blocks = {
        "scores": torch.randn(len(bounds), bwidth, generator=g).cuda(),
        "seq_lens": torch.tensor(bbounds, dtype=torch.int32).cuda(),
    }
    return case, blocks


def _block_ids(rows):
    return torch.full((rows, KB), -7, dtype=torch.int32, device="cuda")


def _sglang_blocks(blocks, ids):
    from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
        topk_transform_paged_hip,
    )

    topk_transform_paged_hip(
        page_table=None, page_indices=ids, page_size=1, raw_indices=None, **blocks
    )


def _mine_pair(case, blocks, outs, ids):
    _one().transform_paged_pair(
        page_indices=outs[0], raw_indices=outs[1], block_scores=blocks["scores"],
        block_lens=blocks["seq_lens"], block_ids=ids, **case,
    )  # fmt: skip


def _same_ids(got, want):
    big = torch.iinfo(torch.int32).max
    key = lambda t: torch.where(t >= 0, t, big).sort(dim=1).values  # noqa: E731
    return torch.equal(key(got), key(want))


@pytest.mark.parametrize("name", list(PAIRS))
@pytest.mark.parametrize("seed", range(3))
def test_pair_equals_sglangs_two_calls(name, seed):
    """The producer's pair launch: the columns as ``transform_paged``, the blocks as SGLang's
    unpaged top-2048, both as sets; the workspace is zeroed after every launch."""
    case, blocks = _pair_case(name, seed)
    rows = len(PAIRS[name][0])
    want, got = _outs(rows), _outs(rows)
    want_ids, got_ids = _block_ids(rows), _block_ids(rows)
    _sglang(case, want)
    _sglang_blocks(blocks, want_ids)
    for _ in range(3):
        for t in (*got, got_ids):
            t.fill_(-7)
        _mine_pair(case, blocks, got, got_ids)
        torch.cuda.synchronize()
        assert _same(case, got, want), name
        assert _same_ids(got_ids, want_ids), name
        assert not _one().ctr.any() and not _one().hist.any(), name


def test_pair_graph_replay():
    """Pair and single launches in one graph, replayed over two data sets."""
    names = ["6x320k", "6x8k", "mixed", "6x128k"]
    pcs = [_pair_case(n, 31) for n in names]
    alt = [_pair_case(n, 32) for n in names]
    data = [[(c["scores"].clone(), b["scores"].clone()) for c, b in pcs],
            [(c["scores"].clone(), b["scores"].clone()) for c, b in alt]]  # fmt: skip
    rows = [len(PAIRS[n][0]) for n in names]
    want = []
    for d in data:
        for (c, b), (s, bs) in zip(pcs, d):
            c["scores"].copy_(s)
            b["scores"].copy_(bs)
        w = []
        for (c, b), r in zip(pcs, rows):
            o, i = _outs(r), _block_ids(r)
            _sglang(c, o)
            _sglang_blocks(b, i)
            w.append((_pairs(o), i))
        want.append(w)
    got = [(_outs(r), _block_ids(r)) for r in rows]

    def step():
        for (c, b), (o, i) in zip(pcs, got):
            _mine_pair(c, b, o, i)
            _mine(c, o)  # a single launch between pairs shares the workspace

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    replays = int(os.environ.get("TOPK_ONE_REPLAYS", "300"))
    bad = 0
    for k in range(replays):
        d = k % 2
        for (c, b), (s, bs) in zip(pcs, data[d]):
            c["scores"].copy_(s)
            b["scores"].copy_(bs)
        for o, i in got:
            for t in (*o, i):
                t.fill_(-7)
        graph.replay()
        for (o, i), (wp, wi) in zip(got, want[d]):
            bad += int(not torch.equal(_pairs(o), wp)) + int(not _same_ids(i, wi))
    torch.cuda.synchronize()
    assert bad == 0, f"{bad} mismatches over {replays} replays"
