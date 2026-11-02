"""The DFlash2 draft's D-attn launch on one gfx950 GPU against the draft layer's
attention in torch: a block of 8 rows a request, 8 query heads of one KV head of
128 dims, every block row sees the prefix's last ``window`` keys and all 8 block
keys (an encoder-only layer), fp8 page-16 cache [slot, 1, 128], page ids through
``req_to_token`` (per-token slots, pages contiguous); and that it compiles for the
served widths."""

import struct

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

PAGES = 4096  # cache pages of the test's K / V planes (65k keys)
MAX_CTX = 8192  # req_to_token's row (tokens)
LAYER = 2


def _epoch(device, deadline_s=20.0):
    from sgl_amd_mono.common import guard

    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return e


def _reference(
    q, k_cache, v_cache, req_to_token, req_pool, prefix_lens, ql, window, scale
):
    """Per request and head: softmax attention of each block row over the prefix's
    last ``window`` keys and the whole block; K / V exact in fp32, P rounded to bf16."""
    from sgl_amd_mono.muse_spark.gfx950.draft_attn import GQA, HEAD_DIM, Q_K

    rows, _ = q.shape
    R = rows // ql
    out = torch.zeros(rows, Q_K, device=q.device)
    for r in range(R):
        ctx = int(prefix_lens[r])
        keys = ctx + ql
        slots = req_to_token[int(req_pool[r]), :keys].long()
        K = k_cache[slots, 0].float()  # [keys, 128]
        V = v_cache[slots, 0].float()
        kidx = torch.arange(keys, device=q.device)
        ok = (
            (kidx >= ctx) | ((ctx - kidx) <= window)
            if window
            else torch.ones(keys, dtype=torch.bool, device=q.device)
        )
        for qpos in range(ql):
            t = r * ql + qpos
            qh = q[t].float().reshape(GQA, HEAD_DIM)  # [8, 128]
            s = scale * (qh @ K.T)
            s = torch.where(ok[None, :], s, torch.full_like(s, float("-inf")))
            p = torch.softmax(s, dim=-1).to(torch.bfloat16).float()
            out[t] = (p @ V).reshape(-1)
    return out


@pytest.mark.parametrize(
    "R,window,part,lens",
    [
        (1, 0, 256, [300]),
        (2, 2048, 256, [3000, 2200]),
        (4, 0, 1024, [5000, 17, 700, 2049]),
        (8, 2048, 0, None),  # the kernel sizes the partition
        (8, 2048, 0, [6000, 300, 8, 7000, 4096, 2048, 2047, 9]),
        (2, 2048, 0, [8000, 6000]),
    ],
)
def test_dattn_matches_the_block_attention(R, window, part, lens):
    from sgl_amd_mono.muse_spark.gfx950 import draft_attn as D
    from sgl_amd_mono.muse_spark.gfx950.layout import scratch_bytes

    ql = 8
    torch.manual_seed(11 * R + window)
    dev = torch.device("cuda")
    bf = torch.bfloat16
    S = ql * R
    if lens is None:
        lens = [int(x) for x in torch.randint(0, 2600, (R,))]  # prefix lengths
    prefix = torch.tensor(lens, dtype=torch.int32, device=dev)
    q = (torch.randn(S, D.Q_K, device=dev) * 0.4).to(bf)
    k_cache = (
        torch.randn(PAGES * 16, D.LOCAL_KV_HEADS, D.HEAD_DIM, device=dev) * 1.5
    ).to(torch.float8_e4m3fn)
    v_cache = torch.randn(PAGES * 16, D.LOCAL_KV_HEADS, D.HEAD_DIM, device=dev).to(
        torch.float8_e4m3fn
    )
    # per-token slots: distinct pages a request, the rows at shuffled pool indices
    perm = torch.randperm(PAGES - 1, device=dev) + 1
    req_to_token = torch.zeros(R + 3, MAX_CTX, dtype=torch.int32, device=dev)
    req_pool = torch.randperm(R + 3, device=dev)[:R].to(torch.int64)
    at = 0
    for r in range(R):
        keys = lens[r] + ql
        n = (keys + 15) // 16
        assert at + n <= PAGES - 1, "the test's pages do not cover these lengths"
        pages = perm[at : at + n]
        slots = (pages[:, None] * 16 + torch.arange(16, device=dev)[None, :]).reshape(
            -1
        )[:keys]
        req_to_token[int(req_pool[r]), :keys] = slots.to(torch.int32)
        at += n
    scale = D.HEAD_DIM**-0.5
    o = torch.full((S, D.Q_K), float("nan"), device=dev, dtype=bf)
    scratch = torch.zeros(
        scratch_bytes(D.dattn_scratch_layout(S, ql)), dtype=torch.uint8, device=dev
    )
    epoch = _epoch(dev)
    launch = D.build_dattn(D.DAttnBuild(tokens=S, q_len=ql))
    scale_bits = struct.unpack("i", struct.pack("f", scale))[0]
    for step in range(2):
        launch(
            q.data_ptr(), o.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
            req_to_token.data_ptr(), MAX_CTX, req_to_token.numel() * 4, req_pool.data_ptr(), 1,
            prefix.data_ptr(), 0, ql, window, part, scale_bits, scratch.data_ptr(), epoch.data_ptr(), LAYER,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        epoch[0] += 1
    torch.cuda.synchronize()
    ref = _reference(
        q, k_cache, v_cache, req_to_token, req_pool, prefix, ql, window, scale
    )
    got = o.float()
    assert not got.isnan().any(), "NaN in the attention output"
    err = (got - ref).abs()
    tol = ref.abs() * (1 / 32) + 2e-2
    bad = (err > tol).float().mean().item()
    assert bad == 0.0, f"{bad:.2%} of the outputs off (max err {err.max().item():.4f})"
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1).min().item()
    assert cos > 0.999, f"least row cosine {cos}"


@pytest.mark.parametrize("rows", [8, 16, 32, 64])
def test_widths_compile(rows):
    from sgl_amd_mono.muse_spark.gfx950 import draft_attn as D
    from sgl_amd_mono.muse_spark.gfx950.runner import compile_only

    compile_only(D.build_dattn(D.DAttnBuild(tokens=rows, q_len=8)))
