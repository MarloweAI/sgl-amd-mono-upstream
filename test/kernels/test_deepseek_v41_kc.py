"""KC, an index KV source's compressor after K1 (RFC #43949), against SGLang's chain on one gfx950.

On random projections, states and slots: the state ring a ratio-2 step parks bit for bit, and the
compressed records and index keys byte for byte but for the rare element a sum in another order
moves across a rounding edge -- SGLang's ``c2_decode_norm_rope_store`` / ``c1_decode_norm_rope_store``,
the indexer's ``wk`` and ``index_k_norm_rope_pack_store_split``, verify blocks of 6 rows and decode
rows, padded rows included."""

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

PAGE, PAGES, RING, MAX_POS = 64, 64, 8, 4096
# bytes allowed to differ: an element whose value sits on a rounding edge (sums in another order)
RECORD_BYTES, KEY_BYTES = 0.01, 0.03


def _case(*, ratio, per_req, reqs, seed, pad=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    rows = per_req * reqs
    cols = 512 * ratio
    proj = torch.randn(rows, cols, generator=g, device=dev)
    proj = proj if ratio == 2 else proj.bfloat16()
    pos = torch.cat(
        [
            torch.arange(per_req, device=dev)
            + int(torch.randint(8, MAX_POS - 8, (1,), generator=g, device=dev))
            for _ in range(reqs)
        ]
    ).long()
    req = torch.arange(reqs, device=dev).repeat_interleave(per_req).long()
    slots = torch.randperm(PAGES * PAGE * ratio - 2, generator=g, device=dev)[:rows] + 2
    raw = slots.long()
    raw[-pad:] = 0  # padded graph rows
    if ratio == 2:
        iloc = torch.where(
            (pos % 2 == 1) & (raw != 0), raw // 2, torch.full_like(raw, -1)
        )
    else:
        iloc = raw.clone()
    freqs = torch.randn(MAX_POS, 64, generator=g, device=dev)
    if per_req == 6:
        raw = raw.int()  # the scheduler's int32 slots, as a verify step serves them
    return dict(
        proj=proj, pos=pos, req=req, raw=raw, iloc=iloc, freqs=freqs, per_req=per_req,
        norm=(torch.rand(512, generator=g, device=dev) + 0.5).bfloat16(),
        wk=(torch.randn(128, 512, generator=g, device=dev) * 0.05).bfloat16(),
        knorm=(torch.rand(128, generator=g, device=dev) + 0.5).bfloat16(),
        state=torch.randn(reqs * RING, 1024, generator=g, device=dev),
    )  # fmt: skip


def _buffers():
    page_bytes = -(-PAGE * 584 // 576) * 576
    cache = torch.zeros(PAGES, page_bytes, dtype=torch.uint8, device="cuda")
    payload = torch.zeros(PAGES, 1, 4, 64, 16, dtype=torch.uint8, device="cuda")
    scale = torch.zeros(PAGES, 1, 4, 64, dtype=torch.uint8, device="cuda")
    return cache, payload, scale


def _forward_wk(latent, wk):
    """The indexer's ``forward_wk``: SGLang's tiny GEMM where it takes the shape, else the linear."""
    from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import _tiny_gemm_max_m

    if 0 < latent.shape[0] <= _tiny_gemm_max_m(wk.shape[0], latent.shape[-1]):
        return tiny_gemm_bf16(latent, wk)
    return torch.nn.functional.linear(latent, wk)


def _sglang(case, ratio):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope_hip import (
        index_k_norm_rope_pack_store_split,
    )
    from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
        c1_decode_norm_rope_store,
        c2_decode_norm_rope_store,
    )

    cache, payload, scale = _buffers()
    state = case["state"].clone()
    if ratio == 2:
        latent = c2_decode_norm_rope_store(
            case["proj"], state, case["norm"], case["pos"], case["req"], case["raw"], 1e-6,
            case["freqs"], cache, page_size=PAGE, ring_size=RING, draft_len=case["per_req"],
            layout="v4",
        )  # fmt: skip
    else:
        latent = c1_decode_norm_rope_store(
            case["proj"], case["norm"], case["pos"], case["raw"], 1e-6, case["freqs"], cache,
            page_size=PAGE, layout="v4",
        )  # fmt: skip
    key = _forward_wk(latent, case["wk"])
    index_k_norm_rope_pack_store_split(
        key, case["knorm"], 1e-6, case["freqs"], case["pos"], case["iloc"], payload, scale,
        ratio=ratio, page8=False,
    )  # fmt: skip
    return state, cache, payload, scale


def _ours(case, ratio):
    from sgl_amd_mono.deepseek_v41.gfx950.indexer.compress import wk_order
    from sgl_amd_mono.deepseek_v41.gfx950.indexer.runner import K2iRunner

    k2i = K2iRunner(device=torch.device("cuda"), bound_max=1 << 16)
    cache, payload, scale = _buffers()
    state = case["state"].clone()
    kc = k2i.kc_args(
        live=case["proj"].shape[0], per_req=case["per_req"], state=state, ring_size=RING,
        norm=case["norm"], eps=1e-6, freqs=case["freqs"], pos=case["pos"], req=case["req"],
        raw_loc=case["raw"], index_loc=case["iloc"], cache=cache, page_size=PAGE,
        wk=wk_order(case["wk"]), k_norm=case["knorm"], k_eps=1e-6, plane=(payload, scale),
    )  # fmt: skip
    k2i.compress(ratio=ratio, rows=case["proj"].shape[0], projected=case["proj"], kc=kc)
    return state, cache, payload, scale


@pytest.mark.parametrize(
    "ratio,per_req,reqs",
    [(2, 6, 1), (2, 6, 8), (2, 1, 16), (1, 6, 1), (1, 6, 8), (1, 1, 16)],
)
def test_kc_is_sglangs_chain(ratio, per_req, reqs):
    case = _case(
        ratio=ratio, per_req=per_req, reqs=reqs, seed=ratio * 100 + per_req + reqs
    )
    want = _sglang(case, ratio)
    got = _ours(case, ratio)
    torch.cuda.synchronize()
    assert torch.equal(got[0], want[0]), "state ring"
    stats = []
    for name, a, b, bound in (
        ("records", got[1], want[1], RECORD_BYTES),
        ("key payload", got[2], want[2], KEY_BYTES),
        ("key scales", got[3], want[3], KEY_BYTES),
    ):
        written = (a != 0) | (b != 0)
        frac = float(((a != b) & written).sum()) / max(int(written.sum()), 1)
        stats.append((name, int(written.sum()), frac))
        assert written.any() and frac <= bound, (name, frac)
    print(ratio, per_req, reqs, stats)
