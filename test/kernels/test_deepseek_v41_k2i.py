"""K2i, the index sources' FP4 scorer (RFC #43949), against SGLang's indexer ops on one gfx950.

On synthetic FP4 planes in SGLang's layout: the device walk plan against the host plan, the
logits against SGLang's and fp64, a candidate consumer's selection against SGLang's top-k within
the same candidate blocks, the two-level pipeline against SGLang's by per-row overlap, and a
graph replay against eager runs.

Adapted from sgl-project/sglang#43984's test_dsv41_k2a_indexer.py."""

import numpy as np
import pytest
import torch

from sgl_amd_mono.deepseek_v41 import spec as dsv41
from sgl_amd_mono.deepseek_v41.gfx950.indexer import index_plan as ip

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx950 GPU",
)

BOUND_MAX = 1 << 19  # every case's longest context
PER_REQ = 6  # DSpark verify: 6 rows a request
# per-row Jaccard against SGLang's pipeline: rounding can flip a near-tie at the 512th row
JACCARD_MEDIAN, JACCARD_P1 = 1.0, 0.99

_RUNNER = []


def _k2i():
    from sgl_amd_mono.deepseek_v41.gfx950.indexer.runner import K2iRunner

    if not _RUNNER:
        _RUNNER.append(K2iRunner(device=torch.device("cuda"), bound_max=BOUND_MAX))
    return _RUNNER[0]


# ---------------------------------------------------------------- the walk plan


def _walk_case(*, rng, n):
    """A step's rows: requests of 1 (decode), 6 (verify) or ragged rows, bounds around a context."""
    kind = rng.integers(3)
    sizes = []
    while sum(sizes) < n:
        sizes.append(1 if kind == 0 else 6 if kind == 1 else int(rng.integers(1, 9)))
    owners, bounds, row = [], [], 0
    for size in sizes:
        size = min(size, n - row)
        choices = [0, 300, 512, 513, int(rng.integers(0, 4096))]
        ctx = int(rng.choice(choices + [int(rng.integers(0, 1 << 20))]))
        owners += [row] * size
        bounds += [max(ctx - (size - 1 - r), 0) for r in range(size)]
        row += size
    return np.array(bounds, dtype=np.int32), np.array(owners, dtype=np.int32)


@pytest.mark.parametrize("seed", range(4))
def test_walk_plan_device_equals_host(seed):
    """The plan K2i lays out (``fill_plan``) equals the host plan, on ragged requests."""
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda")
    k2i = _k2i()
    # one page a request: a row's keys past it repeat its last page (the plan is checked)
    case = fp4_case(ctxs=[64] * 8, seed=seed)
    plan = k2i.plans[2]
    for _ in range(60):
        n = int(rng.integers(1, 49))
        bounds, owners = _walk_case(rng=rng, n=n)
        bounds = np.minimum(bounds, 1 << 16)  # bounded scoring time
        want = np.zeros((plan.shape[0], 4), dtype=np.int32)
        ip.fill_walk_plan(bounds=bounds, owners=owners, plan=want)
        plan.fill_(-7)
        q = case["q"][:1].expand(n, -1, -1).contiguous()
        qs = case["qscale"][:1].expand(n, -1, -1, -1, -1).contiguous()
        w = case["w"][:1].expand(n, -1).contiguous()
        table = case["table"][:1].expand(n, -1).contiguous()
        k2i._score(
            rows=n, query=(q, qs, w), plane=(case["payload"], case["kscale"]), plan=plan,
            itab=table, itab_stride=table.stride(0), itab_len=table.shape[1],
            ibatch=torch.from_numpy(owners).to(dev), ishift=3,
            ibound=torch.from_numpy(bounds).to(dev), lstride=64, reindex=0, fill_plan=1,
        )  # fmt: skip
        got = plan.cpu().numpy()
        assert np.array_equal(got, want), (
            n,
            bounds.tolist(),
            owners.tolist(),
            np.argwhere(got != want)[:4].tolist(),
        )


# ---------------------------------------------------------------- the cases

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 += [-v for v in E2M1]


def _cdiv(a, b):
    return -(-a // b)


def fp4_case(*, ctxs, seed, per_req=PER_REQ, pages_per_req=None, dev="cuda"):
    """Synthetic indexer inputs in SGLang's layouts: ``len(ctxs)`` requests of
    ``per_req`` rows, request r seeing ``ctxs[r]`` compressed rows on its last row
    (one fewer each row before); its pages scattered over the plane."""
    g = torch.Generator(device=dev).manual_seed(seed)
    rows = len(ctxs) * per_req
    ppr = pages_per_req or _cdiv(max(ctxs), 64)
    pages = len(ctxs) * ppr + 3
    u8 = dict(generator=g, device=dev, dtype=torch.uint8)
    payload = torch.randint(0, 256, (pages, 1, 4, 64, 16), **u8)
    kscale = torch.randint(123, 130, (pages, 1, 4, 64), **u8)
    perm = torch.randperm(pages, generator=g, device=dev).to(torch.int32)
    table = (
        perm[: len(ctxs) * ppr]
        .view(len(ctxs), ppr)
        .repeat_interleave(per_req, 0)
        .contiguous()
    )
    q = torch.randint(0, 256, (rows, 32, 64), **u8)
    qscale = torch.randint(123, 130, (rows, 1, 4, 16, 4), **u8)
    w = (torch.randn(rows, 32, generator=g, device=dev) * 0.05).to(torch.bfloat16)
    bounds = torch.tensor(
        [max(c - (per_req - 1 - i), 0) for c in ctxs for i in range(per_req)],
        dtype=torch.int32,
        device=dev,
    )
    return dict(q=q, qscale=qscale, w=w, payload=payload, kscale=kscale, table=table,
                bounds=bounds, per_req=per_req)  # fmt: skip


def _other_query(*, case, seed):
    """The same plane and table, another layer's query (a candidate consumer)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = case["q"].shape[0]
    q = torch.randint(
        0, 256, case["q"].shape, generator=g, device="cuda", dtype=torch.uint8
    )
    w = (torch.randn(rows, 32, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    return dict(case, q=q, w=w)


def _unpack(b):
    lut = torch.tensor(E2M1, dtype=torch.float64, device=b.device)
    return torch.stack([lut[(b & 15).long()], lut[(b >> 4).long()]], -1).flatten(-2)


def ref_logits(*, case, t, cols):
    """fp64 logits of row t at columns ``cols``: sum_h w_h relu(sum_c 2^qe 2^ke q.k)."""
    page = case["table"][t][cols // 64].long()
    slot = cols % 64
    k = _unpack(case["payload"][page, 0, :, slot, :])  # [n, 4, 32]
    ke = case["kscale"][
        page[:, None],
        0,
        torch.arange(4, device=cols.device)[None],
        ((slot % 16) * 4 + slot // 16)[:, None],
    ]
    ks = torch.exp2(ke.double() - 127)  # [n, 4]
    q = _unpack(case["q"][t].view(32, 4, 16))  # [32, 4, 32]
    h = torch.arange(32, device=cols.device)
    qs = torch.exp2(
        case["qscale"][t, 0][:, h % 16, h // 16].double().T - 127
    )  # [32, 4]
    s = torch.einsum("hcd,ncd->hnc", q, k) * qs[:, None, :] * ks[None]
    return (case["w"][t].double()[:, None] * s.sum(-1).clamp_min(0)).sum(0)


def _outs(rows):
    raw = torch.full((rows, 512), -9, dtype=torch.int32, device="cuda")
    return raw, torch.full_like(raw, -9)


def _inputs(case):
    rows = case["q"].shape[0]
    return dict(
        rows=rows,
        per_req=case["per_req"],
        query=(case["q"], case["qscale"], case["w"]),
        plane=(case["payload"], case["kscale"]),
        page_table=case["table"],
        bounds=case["bounds"],
    )


def run_k2i(*, k2i, case, kind="full", outs=None, ratio=1, candidates=None):
    """One index source's selection (FULL and producer: laying out the step's walk plan
    first). -> (raw, page, the producer's candidate blocks)."""
    inputs = _inputs(case)
    raw, page = outs if outs is not None else _outs(inputs["rows"])
    sel = k2i.select(
        kind=kind,
        ratio=ratio,
        raw_out=raw,
        page_out=page,
        candidates=candidates,
        **inputs,
    )
    # the glue the source's K2 runs: the producer's candidate table, a consumer's raw indices
    k2i.run_glue(inputs["rows"], sel)
    return raw, page, sel.blocks


def sglang_logits(case):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        aiter_fp4_paged_mqa_logits,
    )

    return aiter_fp4_paged_mqa_logits(
        q_fp4=case["q"], q_scale=case["qscale"], k_payload=case["payload"],
        k_scale=case["kscale"], weights=case["w"], page_table=case["table"],
        c4_seq_lens=case["bounds"], weight_scale=1.0, page_table_bucket=64,
        is_decode=True,
    )  # fmt: skip


def sglang_select(*, case, logits, candidates=None):
    """SGLang's decode selection on ``logits``: the plain paged top-k and the
    producer's candidate blocks, or a consumer's top-k inside ``candidates``.
    -> (raw, page, the candidate blocks)."""
    from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
        select_candidate_blocks_hip,
        topk_transform_paged_hip,
        topk_within_candidate_blocks_hip,
    )

    raw, page = _outs(case["q"].shape[0])
    if candidates is not None:
        topk_within_candidate_blocks_hip(
            logits=logits, seq_lens=case["bounds"], candidates=candidates,
            page_table=case["table"], page_size=64, page_indices=page, raw_indices=raw,
        )  # fmt: skip
        return raw, page, candidates
    published = select_candidate_blocks_hip(
        logits=logits, seq_lens=case["bounds"], topk_blocks=ip.TOPK_BLOCKS,
        block_size=ip.BLOCK_ROWS,
    )  # fmt: skip
    topk_transform_paged_hip(
        scores=logits, seq_lens=case["bounds"], page_table=case["table"],
        page_indices=page, page_size=64, raw_indices=raw,
    )  # fmt: skip
    return raw, page, published


def _sorted_sets(x):
    """Each row's valid ids, ascending, -1 padded (SGLang's top-k leaves rows unordered)."""
    big = torch.iinfo(torch.int32).max
    s = torch.where(x >= 0, x, big).sort(dim=1).values
    return torch.where(s == big, -1, s)


def _jaccard(*, got, want):
    out = []
    for x, y in zip(got.tolist(), want.tolist()):
        x, y = {v for v in x if v >= 0}, {v for v in y if v >= 0}
        out.append(1.0 if not x and not y else len(x & y) / len(x | y))
    return out


# ---------------------------------------------------------------- the launch


@pytest.mark.parametrize(
    "ctxs", [[8192], [300, 8192], [131072], [524288]], ids=["8k", "mix", "128k", "512k"]
)
def test_logits(ctxs):
    """K2i's FULL logits of every scored row stay within ``k2i.logits.rel`` of SGLang's and fp64."""
    k2i = _k2i()
    case = fp4_case(ctxs=ctxs, seed=len(ctxs) + max(ctxs))
    got = k2i.score(ratio=1, **_inputs(case))
    torch.cuda.synchronize()
    want = sglang_logits(case)
    bound = dsv41.NUMERICS.bound("k2i.logits.rel")
    g = torch.Generator(device="cuda").manual_seed(1)
    for t, b in enumerate(case["bounds"].tolist()):
        if b <= ip.TOPK:
            continue
        a, s = got[t, :b], want[t, :b]
        assert torch.isfinite(a).all(), t
        err = float((a - s).abs().max() / s.abs().max().clamp_min(1e-30))
        assert err <= bound, (t, err)
        cols = (
            torch.randperm(b, generator=g, device="cuda")[:4096].int()
            if b > 8192
            else torch.arange(b, device="cuda", dtype=torch.int32)
        )
        ref = ref_logits(case=case, t=t, cols=cols)
        err64 = float((a[cols.long()].double() - ref).abs().max() / ref.abs().max())
        assert err64 <= bound, (t, err64)


@pytest.mark.parametrize("ctxs", [[20000, 9000], [131072]], ids=["20k9k", "128k"])
def test_reindex_selects_sglangs_rows_within_the_candidates(ctxs):
    """A REINDEX consumer, scoring its candidate columns alone, selects the rows SGLang's top-k
    within the candidate blocks selects from the same layer's full logits."""
    k2i = _k2i()
    case = fp4_case(ctxs=ctxs, seed=21 + max(ctxs))
    _, _, cands = run_k2i(k2i=k2i, case=case, kind="producer")
    consumer = _other_query(case=case, seed=6)
    raw, page, _ = run_k2i(k2i=k2i, case=consumer, kind="reindex", candidates=cands)
    full = k2i.score(ratio=1, **_inputs(consumer)).clone()
    want_raw, want_page, _ = sglang_select(case=consumer, logits=full, candidates=cands)
    torch.cuda.synchronize()
    diff = (_sorted_sets(raw) - _sorted_sets(want_raw)).abs().max().item()
    assert diff <= dsv41.NUMERICS.bound("k2i.sel.max_abs"), diff
    assert torch.equal(_sorted_sets(page), _sorted_sets(want_page))


@pytest.mark.parametrize(
    "ctxs", [[8192, 20000], [131072], [300, 524288]], ids=["8k20k", "128k", "512k"]
)
def test_pipeline_against_sglang(ctxs):
    """A two-level step's producer rows, candidate blocks and consumer rows overlap SGLang's
    whole pipeline (on its own logits) per row."""
    k2i = _k2i()
    case = fp4_case(ctxs=ctxs, seed=11 + max(ctxs))
    raw, _, cands = run_k2i(k2i=k2i, case=case, kind="producer")
    want_raw, _, want_cands = sglang_select(case=case, logits=sglang_logits(case))
    consumer = _other_query(case=case, seed=5)
    raw2, _, _ = run_k2i(k2i=k2i, case=consumer, kind="reindex", candidates=cands)
    want_raw2, _, _ = sglang_select(
        case=consumer, logits=sglang_logits(consumer), candidates=want_cands
    )
    torch.cuda.synchronize()
    for name, js in (
        ("full", _jaccard(got=raw, want=want_raw)),
        ("blocks", _jaccard(got=cands.ids, want=want_cands.ids)),
        ("reindex", _jaccard(got=raw2, want=want_raw2)),
    ):
        med, p1 = float(np.median(js)), float(np.percentile(js, 1))
        print(f"{ctxs} {name}: jaccard median {med:.4f} p1 {p1:.4f}")
        assert med >= JACCARD_MEDIAN and p1 >= JACCARD_P1, (name, med, p1)


def test_graph_replay_selects_as_eager():
    """A captured step of the three kinds, replayed on other contexts and tables, selects as eager.

    Compared as sets: SGLang's top-k leaves a row's order to its atomics; eager runs differ too."""
    k2i = _k2i()
    ppr = _cdiv(40000, 64)
    static = fp4_case(ctxs=[20000, 9000], seed=3, pages_per_req=ppr)
    static2 = _other_query(case=static, seed=4)
    rows = static["q"].shape[0]
    outs = [_outs(rows) for _ in range(3)]
    held = {}

    def step():
        run_k2i(k2i=k2i, case=static, kind="full", outs=outs[0], ratio=2)
        _, _, held["cands"] = run_k2i(
            k2i=k2i, case=static, kind="producer", outs=outs[1]
        )
        run_k2i(
            k2i=k2i,
            case=static2,
            kind="reindex",
            outs=outs[2],
            candidates=held["cands"],
        )

    def results():
        return [t.clone() for o in outs for t in o] + [held["cands"].ids.clone()]

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()  # compile and warm up outside the capture
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    captured = held["cands"]
    for seed, ctxs in ((7, [24000, 31000]), (8, [600, 17000]), (9, [40000, 40000])):
        new = fp4_case(ctxs=ctxs, seed=seed, pages_per_req=ppr)
        for name in ("q", "qscale", "w", "payload", "kscale", "table", "bounds"):
            static[name].copy_(new[name])
        other = _other_query(case=new, seed=seed + 100)
        static2["q"].copy_(other["q"])
        static2["w"].copy_(other["w"])
        graph.replay()
        torch.cuda.synchronize()
        held["cands"] = captured
        replayed = results()
        step()
        torch.cuda.synchronize()
        eager = results()
        for i, (a, b) in enumerate(zip(replayed, eager)):
            assert torch.equal(_sorted_sets(a), _sorted_sets(b)), (ctxs, i)


@pytest.mark.parametrize(
    "ctxs", [[20000, 9000], [131072], [300, 524288]], ids=["20k9k", "128k", "512k"]
)
def test_two_level_equals_sglangs_launches(ctxs):
    """A two-level step's selection equals the one through SGLang's level-one launches,
    ``reindex_table`` and ``map_compact_selection`` (``indexer.candidate_table``): the producer's
    blocks as sets, the REINDEX table entry for entry by block, its widths, and the consumer's
    raw and page indices as sets."""
    from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
        select_candidate_blocks_hip,
        topk_transform_paged_hip,
    )

    from sgl_amd_mono.deepseek_v41.gfx950.indexer.candidate_table import (
        map_compact_selection,
        reindex_table,
    )

    k2i = _k2i()
    case = fp4_case(ctxs=ctxs, seed=51 + max(ctxs))
    rows = case["q"].shape[0]
    _, _, cands = run_k2i(k2i=k2i, case=case, kind="producer")
    width = min(case["table"].shape[1] * 64, k2i.bound_max)
    logits = k2i.ilog[: rows * width].view(rows, width).clone()
    want = select_candidate_blocks_hip(
        logits=logits, seq_lens=case["bounds"], topk_blocks=ip.TOPK_BLOCKS,
        block_size=ip.BLOCK_ROWS,
    )  # fmt: skip
    torch.cuda.synchronize()
    assert torch.equal(_sorted_sets(cands.ids), _sorted_sets(want.ids))
    owners = torch.arange(rows, device="cuda", dtype=torch.int32) // PER_REQ * PER_REQ
    ids, table = torch.empty_like(want.ids), torch.empty_like(want.ids)
    lens = torch.empty(rows, dtype=torch.int32, device="cuda")
    reindex_table(ids=want.ids, page_table=case["table"], owners=owners, bounds=case["bounds"],
                  ids_out=ids, table=table, context=lens)  # fmt: skip
    got_ids, got_table = k2i.cand_ids[:rows].clone(), k2i.cand_table[:rows].clone()
    assert torch.equal(k2i.cand_bound[:rows], lens)
    pairs = lambda i, tb: [sorted(zip(x, y)) for x, y in zip(i.tolist(), tb.tolist())]  # noqa: E731
    assert pairs(got_ids, got_table) == pairs(ids, table)
    # the newest block last, where a row keeps every slot
    full = lens > 8 * (ip.TOPK_BLOCKS - 1)
    assert torch.equal(got_ids[full, -1], ids[full, -1])
    consumer = _other_query(case=case, seed=8)
    raw, page, _ = run_k2i(k2i=k2i, case=consumer, kind="reindex", candidates=cands)
    compact = k2i.ilog[: rows * 16384].view(rows, 16384).clone()
    pos = torch.empty(rows, ip.TOPK, dtype=torch.int32, device="cuda")
    topk_transform_paged_hip(
        scores=compact, seq_lens=k2i.cand_bound[:rows], page_table=torch.zeros(rows, 1, dtype=torch.int32, device="cuda"),
        page_indices=pos, page_size=16384, raw_indices=None,
    )  # fmt: skip
    want_raw = torch.empty_like(raw)
    want_page = torch.empty_like(page)
    map_compact_selection(compact_pos=pos, ids=got_ids, page_table=case["table"], raw_out=want_raw,
                          page_out=want_page)  # fmt: skip
    torch.cuda.synchronize()
    assert torch.equal(_sorted_sets(raw), _sorted_sets(want_raw))
    assert torch.equal(_sorted_sets(page), _sorted_sets(want_page))
    # each slot's raw and page index name the same key
    both = lambda r, p: [sorted(zip(x, y)) for x, y in zip(r.tolist(), p.tolist())]  # noqa: E731
    assert both(raw, page) == both(want_raw, want_page)
