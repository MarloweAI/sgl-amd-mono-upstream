"""The wide REINDEX selection (``indexer.reindex_wide`` and ``wide.WideReindex``'s glue) against
SGLang's consumer on one gfx950, synthetic FP4 planes in SGLang's layout.

SGLang's wide-row consumer scores every compressed key of each request on aiter's row-group
kernel and selects the top 512 inside the producer's candidate blocks. Here: each candidate
column's logit equals SGLang's bit for bit; each row's selected slots and positions equal
SGLang's as sets; the rows come out ascending by position and identical over repeated launches.
"""

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

PER_REQ, HEADS, PAGE, MAX_CTX = 6, 32, 64, 1 << 20


def _case(*, requests, ctx, seed, dev="cuda"):
    """``requests`` DSpark requests at about ``ctx`` compressed rows (ratio 1), their pages
    scattered over the plane; a producer's and a consumer's index query."""
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import rowgroup_table_columns

    g = torch.Generator(device=dev).manual_seed(seed)
    rows = requests * PER_REQ
    ppr = -(-(ctx + PER_REQ) // PAGE)
    pages = requests * ppr + 4
    u8 = dict(generator=g, device=dev, dtype=torch.uint8)
    payload = torch.randint(0, 256, (pages, 1, 4, PAGE, 16), **u8)
    scale = torch.randint(123, 130, (pages, 1, 4, PAGE), **u8)
    perm = torch.randperm(pages - 4, generator=g, device=dev).to(torch.int32) + 4
    cols = rowgroup_table_columns(MAX_CTX, 1, 64)
    table = torch.zeros((rows, cols), dtype=torch.int32, device=dev)
    table[:, :ppr] = perm[: requests * ppr].view(requests, ppr).repeat_interleave(PER_REQ, 0)
    # requests at different lengths, a request's rows one position apart
    lens = torch.tensor(
        [max(ctx - 997 * (b % 5) - (PER_REQ - 1 - i), 1) for b in range(requests) for i in range(PER_REQ)],
        dtype=torch.int32, device=dev,
    )  # fmt: skip

    def query():
        q = torch.randint(0, 256, (rows, HEADS, 64), **u8)
        qs = torch.randint(123, 130, (rows, 1, 4, 16, 4), **u8)
        w = (torch.randn(rows, HEADS, generator=g, device=dev) * 0.05).to(torch.bfloat16)
        return q, qs, w

    return dict(plane=(payload.view(torch.float4_e2m1fn_x2), scale), table=table, lens=lens,
                producer=query(), consumer=query(), rows=rows)  # fmt: skip


def _logits(case, query):
    """SGLang's served logits: aiter's row-group kernel, a request's rows one sequence."""
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        aiter_fp4_paged_mqa_logits,
        prepare_fp4_rowgroup_decode_workspace,
        rowgroup_table_columns,
    )

    ws = prepare_fp4_rowgroup_decode_workspace(
        case["table"], PER_REQ, page_table_bucket=64,
        table_columns=rowgroup_table_columns(MAX_CTX, 1, 64),
    )  # fmt: skip
    q, qs, w = query
    return aiter_fp4_paged_mqa_logits(
        q_fp4=q, q_scale=qs, k_payload=case["plane"][0], k_scale=case["plane"][1], weights=w,
        page_table=case["table"], c4_seq_lens=case["lens"], weight_scale=1.0, is_decode=True,
        decode_workspace=ws, page_table_bucket=64, page8=False,
        rowgroup_table_columns=ws.guarded_page_table.shape[1],
    )  # fmt: skip


def _sets(x):
    big = torch.iinfo(torch.int32).max
    return torch.where(x >= 0, x, big).sort(dim=1).values


@pytest.mark.parametrize(
    "requests,ctx", [(10, 20000), (16, 131072), (40, 65536), (64, 131072)],
    ids=["60r20k", "96r128k", "240r64k", "384r128k"],
)  # fmt: skip
def test_reindex_wide_against_sglang(requests, ctx):
    from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
        select_candidate_blocks_hip,
        topk_compact_candidate_rows,
        topk_within_candidate_blocks_hip,
    )

    from sgl_amd_mono.deepseek_v41 import wide
    from sgl_amd_mono.deepseek_v41.gfx950.indexer import reindex_wide
    from sgl_amd_mono.deepseek_v41.gfx950.indexer.candidate_table import reindex_table

    case = _case(requests=requests, ctx=ctx, seed=requests + ctx)
    rows, lens, table = case["rows"], case["lens"], case["table"]
    cands = select_candidate_blocks_hip(
        _logits(case, case["producer"]), lens, topk_blocks=2048, block_size=8
    )
    full = _logits(case, case["consumer"])
    want_page = torch.full((rows, 512), -9, dtype=torch.int32, device="cuda")
    want_raw = torch.full_like(want_page, -9)
    topk_within_candidate_blocks_hip(
        full, lens, cands, page_table=table, page_size=64, page_indices=want_page,
        raw_indices=want_raw,
    )  # fmt: skip
    buf = wide.ReindexBuffers(torch.device("cuda"))
    buf.compact.fill_(float("nan"))
    q, qs, w = case["consumer"]
    got = []
    for _ in range(3):
        page = torch.full_like(want_page, -9)
        raw = torch.full_like(want_page, -9)
        reindex_table(
            ids=cands.ids, page_table=table, owners=buf.owners[:rows], bounds=lens,
            ids_out=buf.ids[:rows], table=buf.table[:rows], context=buf.context[:rows],
        )  # fmt: skip
        reindex_wide.build(wide.width_class(rows))(
            q.data_ptr(), qs.data_ptr(), w.data_ptr(), case["plane"][0].data_ptr(),
            case["plane"][1].data_ptr(), buf.table.data_ptr(), buf.context.data_ptr(),
            buf.compact.data_ptr(), rows, stream=torch.cuda.current_stream(),
        )  # fmt: skip
        topk_compact_candidate_rows(
            buf.compact[:rows], buf.context[:rows], buf.ids[:rows], cands, lens,
            page_table=table, page_size=64, page_indices=page, raw_indices=raw,
            sort_output=True,
        )  # fmt: skip
        got.append((page, raw))
    torch.cuda.synchronize()
    # every candidate column's logit, SGLang's bit for bit
    ids = buf.ids[:rows].long()
    j = torch.arange(2048 * 8, device="cuda")
    blk = ids[:, j // 8]
    pos = blk * 8 + j % 8
    valid = (j[None] < buf.context[:rows, None]) & (blk >= 0)
    ref = torch.gather(full, 1, pos.clamp_min(0))
    compact = buf.compact[:rows]
    assert valid.sum(1).ge(512).all()
    assert torch.equal(compact.view(torch.int32)[valid], ref.view(torch.int32)[valid])
    page, raw = got[0]
    # the same slots and positions as SGLang's, each row ascending, every launch alike
    assert torch.equal(_sets(page), _sets(want_page))
    assert torch.equal(_sets(raw), _sets(want_raw))
    assert torch.equal(raw, _sets(raw).masked_fill(_sets(raw) == torch.iinfo(torch.int32).max, -1))
    for p, r in got[1:]:
        assert torch.equal(p, page) and torch.equal(r, raw)


@pytest.mark.parametrize(
    "requests,ctx", [(10, 20000), (16, 131072), (64, 131072)], ids=["60r20k", "96r128k", "384r128k"]
)
def test_block_maxima_against_sglang(requests, ctx):
    """The wide producer's level one: block maxima bit for bit SGLang's, its lengths SGLang's,
    and SGLang's top 2048 of them the same blocks as SGLang's own level one selects."""
    from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
        candidate_block_scores,
        select_candidate_blocks_hip,
        topk_transform_paged_hip,
    )

    from sgl_amd_mono.deepseek_v41 import wide
    from sgl_amd_mono.deepseek_v41.gfx950.indexer.candidate_table import block_maxima

    case = _case(requests=requests, ctx=ctx, seed=7 + requests + ctx)
    rows, lens = case["rows"], case["lens"]
    logits = _logits(case, case["producer"])
    want = select_candidate_blocks_hip(logits, lens, topk_blocks=2048, block_size=8)
    ref = candidate_block_scores(logits, lens, block_size=8, fill_tail=False)
    buf = wide.ReindexBuffers(torch.device("cuda"))
    scores = buf.block_scores(logits.shape[1])[:rows]
    scores.fill_(float("nan"))
    block_maxima(
        logits=logits, bounds=lens, scores=scores, block_lens=buf.block_lens[:rows],
        compact_lens=buf.compact_lens[:rows], topk_blocks=2048,
    )  # fmt: skip
    ids = buf.blocks[:rows]
    topk_transform_paged_hip(scores, buf.block_lens[:rows], None, ids, 1, None)
    torch.cuda.synchronize()
    nb = (lens + 7) // 8
    m = torch.arange(scores.shape[1], device="cuda")[None] < nb[:, None]
    assert torch.equal(scores.view(torch.int32)[m], ref.view(torch.int32)[m])
    assert torch.equal(buf.block_lens[:rows], nb.int())
    assert torch.equal(buf.compact_lens[:rows], want.compact_lens)
    assert torch.equal(_sets(ids), _sets(want.ids))
