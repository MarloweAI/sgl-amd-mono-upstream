"""Run the GLM-5.3-Flash layer launch on random weights in SGLang's layouts under
torchrun (TP 2, 4 or 8): ``torchrun --nproc-per-node 4
test/kernels/glm53_flash/check_layer.py --kind dsa --rows 4``. Every rank must
finish with no error word set and the same output (the TP sums run in rank order);
``--writes``: the launch changes exactly its valid rows' state or cache rows;
``--compare-shuffled``: a sparse-attention launch on AITER-shuffled FP8 projections
equals the row-major one bitwise; ``--check-routing``: the top-8 equals SGLang's on
keys that tie in bf16; ``--check-keys``: rows past 2048 tokens attend to the selected
pools and then their tail, as SGLang's kpool top-k selects."""

import argparse
import os
import sys

import torch
import torch.distributed as dist

from sgl_amd_mono.glm53_flash.gfx950.config import (
    GLOBAL_INTER,
    HC_MULT,
    HIDDEN,
    N_EXPERTS,
)

N_MIX = HC_MULT * (2 + HC_MULT)


def _gen(seed: int, device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def ffn_weights(rank: int, tp: int, device):
    """One MoE layer's random weights (every rank builds the same) as this rank's
    shard in SGLang's post-load AITER A4W4 layout; the shared expert is slot 288."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    from sgl_amd_mono.glm53_flash.gfx950.packing import (
        pack_bf16,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashFfnWeights,
    )

    g = _gen(0, device)
    E, full, inter = N_EXPERTS + 1, GLOBAL_INTER, GLOBAL_INTER // tp
    lo, hi = rank * inter, (rank + 1) * inter

    def mxfp4(rows, k):
        q = torch.randint(0, 256, (E, rows, k // 2), generator=g, device=device)
        s = torch.randint(119, 124, (E, rows, k // 32), generator=g, device=device)
        return q.to(torch.uint8), s.to(torch.uint8)

    def scales(s):
        e, r, c = s.shape
        return e8m0_shuffle(s.contiguous().view(e * r, c)).view(e, r, c)

    q13, s13 = mxfp4(2 * full, HIDDEN)
    q2, s2 = mxfp4(HIDDEN, full)
    fp4 = torch.float4_e2m1fn_x2
    rows13 = (
        lambda t: torch.cat(  # noqa: E731
            [t[:, lo:hi], t[:, full + lo : full + hi]], 1
        ).contiguous()
    )
    w_ug = shuffle_weight(rows13(q13).view(fp4), layout=(16, 16)).view(torch.uint8)
    s_ug = scales(rows13(s13))
    w_dn = shuffle_weight(
        q2[:, :, lo // 2 : hi // 2].contiguous().view(fp4), layout=(16, 16)
    ).view(torch.uint8)
    s_dn = scales(s2[:, :, lo // 32 : hi // 32])
    w_r = (torch.randn(N_EXPERTS, HIDDEN, generator=g, device=device) * 0.02).bfloat16()
    bias = (torch.rand(N_EXPERTS, generator=g, device=device) - 0.5) * 0.2
    return FlashFfnWeights(
        w_r=pack_bf16(w_r),
        bias=bias,
        w_ug=w_ug,
        s_ug=s_ug,
        w_dn=w_dn,
        s_dn=s_dn,
        w_sh_ug=w_ug[N_EXPERTS],
        s_sh_ug=s_ug[N_EXPERTS],
        w_sh_dn=w_dn[N_EXPERTS],
        s_sh_dn=s_dn[N_EXPERTS],
        hc_fn=torch.randn(N_MIX, HC_MULT * HIDDEN, generator=g, device=device) * 0.01,
        hc_scale=torch.rand(3, generator=g, device=device) + 0.5,
        hc_base=torch.randn(N_MIX, generator=g, device=device) * 0.1,
        g_norm=(1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).bfloat16(),
    )


def kda_weights(rank: int, tp: int, device):
    """One KDA layer's random attention weights (every rank builds the same) as this
    rank's shard in SGLang's post-load layout."""
    from sgl_amd_mono.glm53_flash.gfx950.config import (
        KDA_DIM,
        KDA_HEADS,
        KDA_PROJ,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashKdaWeights,
    )

    g = _gen(1, device)
    D, PF, HF = KDA_DIM, KDA_PROJ, KDA_HEADS
    P, H = PF // tp, HF // tp

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=device) * scale

    def sh(t, n):
        return t[rank * n : (rank + 1) * n]

    wq, wk, wv = (rnd(PF, HIDDEN, scale=HIDDEN**-0.5) for _ in range(3))
    wb = rnd(HF, HIDDEN, scale=HIDDEN**-0.5)
    wfa, wga = (rnd(D, HIDDEN, scale=HIDDEN**-0.5) for _ in range(2))
    w_fb, w_gb = (rnd(PF, D, scale=D**-0.5).bfloat16() for _ in range(2))
    w_o = rnd(HIDDEN, PF, scale=PF**-0.5).bfloat16()
    conv = rnd(3, PF, 4, scale=0.5)
    a_log = torch.log(torch.rand(HF, generator=g, device=device) * 8 + 1)
    return FlashKdaWeights(
        hc_fn=rnd(N_MIX, HC_MULT * HIDDEN, scale=0.01),
        hc_scale=torch.rand(3, generator=g, device=device) + 0.5,
        hc_base=rnd(N_MIX, scale=0.1),
        g_norm=(1 + 0.1 * rnd(HIDDEN)).bfloat16(),
        w_in=torch.cat(
            [sh(wq, P), sh(wk, P), sh(wv, P), sh(wb, H), wfa, wga]
        ).bfloat16(),
        w_fb=sh(w_fb, P).contiguous(),
        w_gb=sh(w_gb, P).contiguous(),
        conv_w=torch.cat([sh(conv[i], P) for i in range(3)]).contiguous(),
        a_log=sh(a_log, H).contiguous(),
        dt_bias=sh(rnd(PF, scale=0.5), P).contiguous(),
        w_on=(1 + 0.1 * rnd(D)).float(),
        w_o=w_o[:, rank * P : (rank + 1) * P].contiguous(),
    )


def kda_state(rows: int, tp: int, device):
    """SGLang's KDA state pool: bf16 conv [slots, 3, 3P], f32 recurrent
    [slots, H, 128, 128]; slot 0 is the padding slot."""
    from sgl_amd_mono.glm53_flash.gfx950.config import (
        KDA_DIM,
        KDA_HEADS,
        KDA_PROJ,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashKdaState,
    )

    g = _gen(3, device)
    P, H, slots = KDA_PROJ // tp, KDA_HEADS // tp, rows + 2
    conv = torch.randn(slots, 3, 3 * P, generator=g, device=device).bfloat16()
    rec = torch.randn(slots, H, KDA_DIM, KDA_DIM, generator=g, device=device) * 0.05
    idx = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    return FlashKdaState(conv_state=conv, rec_state=rec), idx


def _fp8_block(w):
    """FP8 E4M3 with f32 128 x 128 block scales: (q, scales)."""
    R, K = w.shape
    wb = w.float().view(R // 128, 128, K // 128, 128)
    s = wb.abs().amax((1, 3)) / 448.0 + 1e-12
    q = (wb / s[:, None, :, None]).to(torch.float8_e4m3fn)
    return q.view(R, K), s.contiguous()


def dsa_weights(rank: int, tp: int, device):
    """One sparse-attention layer's random weights as this rank's shard: q_a | kv_a,
    q_b and o_proj FP8 with block scales, as the amd/ checkpoint loads."""
    from sgl_amd_mono.glm53_flash.gfx950.config import (
        INDEX_DIM,
        INDEX_HEADS,
        KV_LORA,
        MLA_HEADS,
        NOPE_DIM,
        Q_LORA,
        V_DIM,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashDsaWeights,
    )

    g = _gen(2, device)
    H, per = MLA_HEADS // tp, NOPE_DIM + V_DIM

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=device) * scale

    w_qkv, s_qkv = _fp8_block(rnd(Q_LORA + KV_LORA, HIDDEN, scale=HIDDEN**-0.5))
    q_b, s_qb = _fp8_block(rnd(MLA_HEADS * NOPE_DIM, Q_LORA, scale=Q_LORA**-0.5))
    kv_b = rnd(MLA_HEADS * per, KV_LORA, scale=KV_LORA**-0.5)
    w_o, s_o = _fp8_block(
        rnd(HIDDEN, MLA_HEADS * V_DIM, scale=(MLA_HEADS * V_DIM) ** -0.5)
    )
    w_kvb = kv_b[rank * H * per : (rank + 1) * H * per].bfloat16().contiguous()
    return FlashDsaWeights(
        hc_fn=rnd(N_MIX, HC_MULT * HIDDEN, scale=0.01),
        hc_scale=torch.rand(3, generator=g, device=device) + 0.5,
        hc_base=rnd(N_MIX, scale=0.1),
        g_norm=(1 + 0.1 * rnd(HIDDEN)).bfloat16(),
        w_qkv=w_qkv,
        s_qkv=s_qkv,
        w_idx=rnd(INDEX_DIM + INDEX_HEADS, HIDDEN, scale=HIDDEN**-0.5).bfloat16(),
        w_gate=rnd(INDEX_DIM, HIDDEN, scale=HIDDEN**-0.5).bfloat16(),
        g_qa=(1 + 0.1 * rnd(Q_LORA)).bfloat16(),
        g_kv=(1 + 0.1 * rnd(KV_LORA)).bfloat16(),
        ln_w=1 + 0.1 * rnd(INDEX_DIM),
        ln_b=0.1 * rnd(INDEX_DIM),
        ape=0.5 * rnd(4, INDEX_DIM),
        w_qb=q_b[rank * H * NOPE_DIM : (rank + 1) * H * NOPE_DIM].contiguous(),
        s_qb=s_qb[rank * H * 2 : (rank + 1) * H * 2].contiguous(),
        w_iq=rnd(INDEX_HEADS * INDEX_DIM, Q_LORA, scale=Q_LORA**-0.5).bfloat16(),
        w_uk=w_kvb.view(H, per, KV_LORA)[:, :NOPE_DIM].transpose(1, 2).contiguous(),
        w_kvb=w_kvb,
        w_o=w_o[:, rank * H * V_DIM : (rank + 1) * H * V_DIM].contiguous(),
        s_o=s_o[:, rank * H * 2 : (rank + 1) * H * 2].contiguous(),
    )


MAX_LEN = 8192
# tokens of each row including the new one (0: a pad row); 4100, 1500, 7000, 64
# complete a pool, 2050 / 2049 / 3333 keep a tail
LENS = (4100, 1500, 2050, 7000, 2049, 64, 3333)


def dsa_caches(rows: int, device, lens=None):
    """SGLang's DSA caches for ``rows`` requests of up to MAX_LEN tokens (256-token
    pages; page 0 is the padding page), its 64-token page table and the step's row
    table."""
    from sgl_amd_mono.glm53_flash.gfx950.config import (
        INDEX_DIM,
        KPOOL,
        KV_LORA,
    )
    from sgl_amd_mono.glm53_flash.gfx950.layout import (
        ROW_BT,
        ROW_POOL_SLOT,
        ROW_POS,
        ROW_SLOT,
        ROW_TAIL_SLOT,
        ROW_WORDS,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashDsaCaches,
    )

    g = _gen(5, device)
    per = MAX_LEN // 256
    pages = 1 + rows * per
    kv = torch.randn(pages * 256, KV_LORA, generator=g, device=device).bfloat16()
    idx = torch.zeros(pages, 64 * (INDEX_DIM + 4), dtype=torch.uint8, device=device)
    off = torch.arange(64, device=device)[:, None]
    d = torch.arange(INDEX_DIM, device=device)[None, :]
    perm = ((off // 16) * 2048 + (d // 16) * 256 + (off % 16) * 16 + d % 16).reshape(-1)
    vals = (torch.randn(pages, 64 * INDEX_DIM, generator=g, device=device) * 64).clamp(
        -448, 448
    )
    idx[:, perm] = vals.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.exp2(
        -torch.randint(6, 10, (pages, 64), generator=g, device=device).float()
    )
    idx[:, 64 * INDEX_DIM :] = scales.view(torch.uint8)
    tail_k = torch.randn(rows + 1, KPOOL, INDEX_DIM, generator=g, device=device)
    tail_g = torch.randn(rows + 1, KPOOL, INDEX_DIM, generator=g, device=device)
    order = torch.randperm(pages - 1, generator=g, device=device) + 1
    req_pages = order[: rows * per].view(rows, per)
    # 64-token page table: page p of a request is 64-token pages 4p .. 4p + 3
    pt = (req_pages[:, :, None] * 4 + torch.arange(4, device=device)).view(rows, -1)
    table = torch.full((rows, ROW_WORDS), -1, dtype=torch.int32, device=device)
    lens = lens or list(LENS[: rows - 1]) + [0]
    for r, n in enumerate(lens):
        table[r, ROW_BT] = r
        if n == 0:
            continue
        pos = n - 1
        slot = int(req_pages[r, pos // 256]) * 256 + pos % 256
        table[r, ROW_POS], table[r, ROW_SLOT] = pos, slot
        table[r, ROW_POOL_SLOT], table[r, ROW_TAIL_SLOT] = (
            slot // KPOOL,
            (r + 1) * KPOOL,
        )
    caches = FlashDsaCaches(
        kv_cache=kv,
        idx_cache=idx,
        tail_k=tail_k.bfloat16().contiguous(),
        tail_g=tail_g.bfloat16().contiguous(),
    )
    return caches, table, pt.int().contiguous()


def layer_inputs(rows: int, device):
    """The previous layer's (x, residual, post, comb), its FFN post owed."""
    g = _gen(7, device)
    x = torch.randn(rows, HIDDEN, generator=g, device=device).bfloat16()
    res = torch.randn(rows, HC_MULT, HIDDEN, generator=g, device=device).bfloat16()
    post = torch.rand(rows, HC_MULT, 1, generator=g, device=device) * 2
    comb = torch.softmax(
        torch.randn(rows, HC_MULT, HC_MULT, generator=g, device=device), -1
    )
    return x, res, post, comb


def shuffled_copy(w):
    """``w`` with its FP8 projections in AITER's (16, 16) layout, as SGLang's gfx950
    block-FP8 linear method shuffles them in place."""
    import msgspec
    from aiter.ops.shuffle import shuffle_weight

    return msgspec.structs.replace(
        w,
        w_qkv=shuffle_weight(w.w_qkv, (16, 16)),
        w_qb=shuffle_weight(w.w_qb, (16, 16)),
        w_o=shuffle_weight(w.w_o, (16, 16)),
        shuffled=(True, True, True),
    )


def compare_shuffled(args, rank, tp, group, device) -> bool:
    """One sparse-attention launch on row-major and on AITER-shuffled FP8 weights:
    the outputs and every cache write must be bitwise equal."""
    from msgspec.structs import astuple, replace

    from sgl_amd_mono.glm53_flash.gfx950.kernel import (
        FlashBuild,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashLayerOp,
        FlashRuntime,
    )

    key = FlashBuild(
        rows=args.rows,
        tp=tp,
        launches_per_step=1,
        inter=GLOBAL_INTER // tp,
        dsa=True,
        max_pools=args.max_pools,
        stop=False,
    )
    ffn, w = ffn_weights(rank, tp, device), dsa_weights(rank, tp, device)
    x, res, post, comb = layer_inputs(args.rows, device)
    lens = [int(v) for v in args.lens.split(",")] if args.lens else None
    caches, table, pt = dsa_caches(args.rows, device, lens)
    results = []
    for k, weights in (
        (key, w),
        (
            replace(key, qkv_shuffled=True, qb_shuffled=True, o_shuffled=True),
            shuffled_copy(w),
        ),
    ):
        rt = FlashRuntime(k, rank, group, device)
        rt.set_deadline(args.deadline)
        op = FlashLayerOp(rt, ffn, 0, dsa=weights)
        c = type(caches)(*(t.clone() for t in astuple(caches)))
        out = op.launch(x, res, post, comb, caches=c, rows=table, page_table=pt)
        torch.cuda.synchronize()
        results.append((tuple(t.clone() for t in out), c))
        rt.close()
    (out_a, c_a), (out_b, c_b) = results
    same_out = all(torch.equal(a, b) for a, b in zip(out_a, out_b))
    same_cache = all(torch.equal(a, b) for a, b in zip(astuple(c_a), astuple(c_b)))
    wrote = not torch.equal(c_a.kv_cache, caches.kv_cache)
    print(
        f"rank {rank}: shuffled == row-major: outputs {same_out}, caches {same_cache} "
        f"(kv written {wrote})",
        flush=True,
    )
    return same_out and same_cache and wrote


# router weights and biases whose keys tie after bf16 rounding (see tie_router)
TIE_GROUPS, TIE_MEMBERS = 24, 12
TIE_DELTA = 2.0**-12


def tie_router(ffn, device):
    """``ffn`` with router weights and biases whose keys tie after bf16 rounding: 24
    groups of 12 experts at random ids. Group g reads column j_g with weight 1 and has
    bias b_g (bf16); its odd members add +-TIE_DELTA x column j2_g, and every member's
    f32 bias is b_g plus less than a quarter bf16 ulp. The logits are exact in f32 (two
    nonzero products), so the keys tie exactly in bf16 and differ in f32. Returns the
    weights and the plain router [288, HIDDEN] bf16."""
    from msgspec.structs import replace

    from sgl_amd_mono.glm53_flash.gfx950.packing import pack_bf16

    g = _gen(11, device)
    assert TIE_GROUPS * TIE_MEMBERS == N_EXPERTS
    ids = torch.randperm(N_EXPERTS, generator=g, device=device).view(
        TIE_GROUPS, TIE_MEMBERS
    )
    cols = torch.randperm(HIDDEN, generator=g, device=device)[: 2 * TIE_GROUPS]
    cols = cols.view(2, TIE_GROUPS)
    b = ((torch.rand(TIE_GROUPS, generator=g, device=device) - 0.5) * 0.4).bfloat16()
    b = b.float()
    ulp = torch.exp2(torch.floor(torch.log2(b.abs())) - 7)
    w = torch.zeros(N_EXPERTS, HIDDEN, device=device)
    bias = torch.zeros(N_EXPERTS, device=device)
    for grp in range(TIE_GROUPS):
        for m in range(TIE_MEMBERS):
            e = int(ids[grp, m])
            w[e, cols[0, grp]] = 1.0
            if m % 2:
                w[e, cols[1, grp]] = TIE_DELTA if m % 4 == 1 else -TIE_DELTA
            eps = (torch.rand((), generator=g, device=device) - 0.5) * 0.5 * ulp[grp]
            bias[e] = b[grp] + eps
    w = w.bfloat16()
    return replace(ffn, w_r=pack_bf16(w), bias=bias), w


def ffn_input(rt, rows: int) -> torch.Tensor:
    """The launch's own RMSNorm'd FFN input rows (bf16), from its scratch."""
    a = rt.scratch[rt.sc["a"] :].view(torch.int32)[: rows * HIDDEN : 2]
    return a.contiguous().view(torch.bfloat16).view(rows, HIDDEN)


def routed(rt, rows: int) -> torch.Tensor:
    """The launch's 8 routed experts a row, best first (slot 0 is the shared expert)."""
    from sgl_amd_mono.glm53_flash.gfx950.config import MOE_SLOTS

    sel = rt.scratch[rt.sc["sel"] :].view(torch.int32)[: 2 * rows * MOE_SLOTS : 2]
    return sel.view(rows, MOE_SLOTS)[:, 1:].clone()


def stock_topk(x, w_r, bias) -> torch.Tensor:
    """SGLang's top-8 for GLM-5.3-Flash with AITER: bf16 logits and bias into AITER's
    ``biased_grouped_topk`` (one group, renormalized, scale 2.5), best first."""
    from aiter import biased_grouped_topk

    from sgl_amd_mono.glm53_flash.gfx950.config import ROUTE_SCALE, TOP_K

    logits = (x.float() @ w_r.float().T).bfloat16()
    weights = torch.empty(x.shape[0], TOP_K, dtype=torch.float32, device=x.device)
    ids = torch.empty(x.shape[0], TOP_K, dtype=torch.int32, device=x.device)
    biased_grouped_topk(logits, bias.bfloat16(), weights, ids, 1, 1, True, ROUTE_SCALE)
    return ids


def key_order(x, w_r, bias, rounded: bool, tie: str) -> torch.Tensor:
    """Top-8 ids of sigmoid(logits) + bias, best first: logits and bias in bf16
    (``rounded``) or f32; equal keys in AITER's lane order (``tie="aiter"``) or by
    lower id (``tie="id"``)."""
    from sgl_amd_mono.glm53_flash.gfx950.config import TOP_K
    from sgl_amd_mono.glm53_flash.gfx950.kernel import tie_rank

    logits, b = x.float() @ w_r.float().T, bias.float()
    if rounded:
        logits, b = logits.bfloat16().float(), b.bfloat16().float()
    keys = torch.sigmoid(logits) + b
    ids = torch.arange(N_EXPERTS, device=x.device)
    if tie == "aiter":
        rank = torch.tensor([tie_rank(e) for e in range(N_EXPERTS)], device=x.device)
    else:
        rank = ids
    by_rank = torch.argsort(rank)
    order = torch.argsort(-keys[:, by_rank], dim=-1, stable=True)
    return by_rank[order][:, :TOP_K].int()


def check_routing(args, rank, tp, group, device) -> bool:
    """FFN launches on router keys that tie in bf16: the launch's top-8 (order
    included) equals SGLang's (AITER's ``biased_grouped_topk`` on bf16 logits and bias)
    on the launch's own FFN input, every draw; and the inputs tell that apart from
    lower-id ties and from f32 keys (negative controls)."""
    from sgl_amd_mono.glm53_flash.gfx950.kernel import FlashBuild
    from sgl_amd_mono.glm53_flash.gfx950.runner import FlashLayerOp, FlashRuntime

    key = FlashBuild(
        rows=args.rows, tp=tp, launches_per_step=1, inter=GLOBAL_INTER // tp, stop=False
    )
    ffn, w_r = tie_router(ffn_weights(rank, tp, device), device)
    rt = FlashRuntime(key, rank, group, device)
    rt.set_deadline(args.deadline)
    op = FlashLayerOp(rt, ffn, 0)
    ok, by_id, by_f32, rows = True, 0, 0, 0
    for draw in range(3):
        g = _gen(20 + draw, device)
        x = torch.randn(args.rows, HIDDEN, generator=g, device=device).bfloat16()
        _, res, post, comb = layer_inputs(args.rows, device)
        op.launch(x / tp, res, post, comb)
        rt.advance_step()
        torch.cuda.synchronize()
        a, got = ffn_input(rt, args.rows), routed(rt, args.rows)
        same = torch.equal(got, stock_topk(a, w_r, ffn.bias))
        own = torch.equal(got, key_order(a, w_r, ffn.bias, True, "aiter"))
        print(
            f"rank {rank} draw {draw}: top-8 == AITER biased_grouped_topk {same}, "
            f"== bf16 keys in AITER's tie order {own}",
            flush=True,
        )
        ok = ok and same and own
        by_id += int((got != key_order(a, w_r, ffn.bias, True, "id")).any(-1).sum())
        by_f32 += int(
            (got != key_order(a, w_r, ffn.bias, False, "aiter")).any(-1).sum()
        )
        rows += args.rows
    print(
        f"rank {rank}: lower-id ties would differ in {by_id}/{rows} rows, f32 keys "
        f"in {by_f32}/{rows}",
        flush=True,
    )
    rt.close()
    return ok and by_id > 0 and by_f32 > 0


# row lengths for the key check: scored rows with tails of 2, 3 and 1 tokens, and a
# row of 2051 tokens (2048 + a tail of 3), which attends to every token
KEY_LENS = (4102, 5003, 7001, 2051)


def kv_slot(pt: torch.Tensor, row: int, token: int) -> int:
    """MLA cache row of ``token`` of request ``row`` through the 64-token page table."""
    return int(pt[row, token // 64]) * 64 + token % 64


def check_keys(args, rank, tp, group, device) -> bool:
    """Sparse-attention launches on rows past 2048 tokens. A scored row (more than 512
    complete pools) attends to the selected pools' 2048 tokens and then its tail (up to
    2051 keys), as SGLang's kpool top-k appends it: its selection holds 512 whole
    distinct complete pools, then the tail tokens in order. Perturbing the MLA rows of
    the tokens a 2048-key selection gives up for the tail (its last selected pool's last
    positions) changes the row's output (negative control: a 2048-key selection never
    reads them); perturbing an unselected pool changes nothing. A row of 2051 tokens
    attends to every token: perturbing its first token changes its output."""
    from msgspec.structs import replace

    from sgl_amd_mono.glm53_flash.gfx950.config import KPOOL, TOPK_POOLS, TOPK_TOKENS
    from sgl_amd_mono.glm53_flash.gfx950.kernel import FlashBuild
    from sgl_amd_mono.glm53_flash.gfx950.layout import SEL_STRIDE
    from sgl_amd_mono.glm53_flash.gfx950.runner import FlashLayerOp, FlashRuntime

    rows = len(KEY_LENS)
    key = FlashBuild(
        rows=rows,
        tp=tp,
        launches_per_step=1,
        inter=GLOBAL_INTER // tp,
        dsa=True,
        max_pools=args.max_pools,
        stop=False,
    )
    rt = FlashRuntime(key, rank, group, device)
    rt.set_deadline(args.deadline)
    op = FlashLayerOp(
        rt, ffn_weights(rank, tp, device), 0, dsa=dsa_weights(rank, tp, device)
    )
    x, res, post, comb = layer_inputs(rows, device)
    caches, table, pt = dsa_caches(rows, device, list(KEY_LENS))

    def run(kv):
        c = replace(
            caches,
            kv_cache=kv,
            idx_cache=caches.idx_cache.clone(),
            tail_k=caches.tail_k.clone(),
            tail_g=caches.tail_g.clone(),
        )
        out = op.launch(x, res, post, comb, caches=c, rows=table, page_table=pt)
        rt.advance_step()
        torch.cuda.synchronize()
        sel = rt.scratch[rt.sc["tok_sel"] :].view(torch.int32)[: rows * SEL_STRIDE]
        return out[1].clone(), sel.view(rows, SEL_STRIDE).clone()

    base, sel = run(caches.kv_cache.clone())
    g = _gen(30, device)
    ok = True
    for r, n in enumerate(KEY_LENS):
        bound, tail = n // KPOOL, n % KPOOL
        if bound <= TOPK_POOLS:
            victims, spare = [0], None
        else:
            picked = sel[r, :TOPK_TOKENS].long()
            pools = picked.view(TOPK_POOLS, KPOOL)
            whole = bool(
                (pools == pools[:, :1] + torch.arange(KPOOL, device=device)).all()
            )
            whole = whole and bool((pools[:, 0] % KPOOL == 0).all())
            ids = pools[:, 0] // KPOOL
            distinct = ids.unique().numel() == TOPK_POOLS and int(ids.max()) < bound
            tail_ok = sel[r, TOPK_TOKENS : TOPK_TOKENS + tail].tolist() == [
                bound * KPOOL + i for i in range(tail)
            ]
            print(
                f"rank {rank} row {r} ({n} tokens): 512 whole distinct pools "
                f"{whole and distinct}, tail {tail_ok}",
                flush=True,
            )
            ok = ok and whole and distinct and tail_ok
            victims = picked[TOPK_TOKENS - tail : TOPK_TOKENS].tolist()
            unpicked = sorted(set(range(bound)) - set(ids.tolist()))
            spare = unpicked[0] * KPOOL
        kv = caches.kv_cache.clone()
        for t in victims:
            kv[kv_slot(pt, r, t)] = (
                30 * torch.randn(kv.shape[1], generator=g, device=device).bfloat16()
            )
        moved = not torch.equal(run(kv)[0][r], base[r])
        print(
            f"rank {rank} row {r}: perturbing tokens {victims} changes the row {moved}",
            flush=True,
        )
        ok = ok and moved
        if spare is not None:
            kv = caches.kv_cache.clone()
            for t in range(spare, spare + KPOOL):
                kv[kv_slot(pt, r, t)] = (
                    30 * torch.randn(kv.shape[1], generator=g, device=device).bfloat16()
                )
            still = torch.equal(run(kv)[0][r], base[r])
            print(
                f"rank {rank} row {r}: perturbing unselected pool {spare // KPOOL} "
                f"leaves the row unchanged {still}",
                flush=True,
            )
            ok = ok and still
    rt.close()
    return ok


def _changed_rows(before: torch.Tensor, after: torch.Tensor) -> set:
    d = (before != after).reshape(before.shape[0], -1).any(1)
    return set(d.nonzero().flatten().tolist())


def writes_only_its_rows(extra: dict, before: dict) -> bool:
    """After one launch: the state or cache rows that changed are exactly the valid
    rows' (pad rows write nothing)."""
    from msgspec.structs import asdict

    from sgl_amd_mono.glm53_flash.gfx950.layout import (
        ROW_POOL_SLOT,
        ROW_POS,
        ROW_SLOT,
        ROW_TAIL_SLOT,
    )

    if "state" in extra:
        idx = extra["state_idx"].tolist()
        want = {
            "conv_state": {i for i in idx if i > 0},
            "rec_state": {i for i in idx if i > 0},
        }
        now = asdict(extra["state"])
    else:
        rows = [r for r in extra["rows"].tolist() if r[ROW_POS] >= 0]
        want = {
            "kv_cache": {r[ROW_SLOT] for r in rows},
            "idx_cache": {r[ROW_POOL_SLOT] // 64 for r in rows if r[ROW_POS] % 4 == 3},
            "tail_k": {r[ROW_TAIL_SLOT] // 4 for r in rows},
            "tail_g": {r[ROW_TAIL_SLOT] // 4 for r in rows},
        }
        now = asdict(extra["caches"])
    ok = True
    for name, rows_want in want.items():
        got = _changed_rows(before[name], now[name])
        if got != rows_want:
            print(
                f"{name}: changed rows {sorted(got)}, the valid rows' {sorted(rows_want)}"
            )
            ok = False
    return ok


def main() -> int:
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.glm53_flash.gfx950.kernel import (
        FlashBuild,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashLayerOp,
        FlashRuntime,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("ffn", "kda", "dsa"), default="ffn")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--deadline", type=float, default=10.0)
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--max-pools", type=int, default=MAX_LEN // 4)
    ap.add_argument("--lens", type=str, default="", help="comma-separated row lengths")
    ap.add_argument("--compare-shuffled", action="store_true")
    ap.add_argument("--check-routing", action="store_true")
    ap.add_argument("--check-keys", action="store_true")
    ap.add_argument("--pad-rows", type=int, default=0, help="trailing pad rows")
    ap.add_argument(
        "--writes", action="store_true", help="check the rows a launch writes"
    )
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    if args.compare_shuffled:
        return 0 if compare_shuffled(args, rank, tp, group, device) else 1
    if args.check_routing:
        return 0 if check_routing(args, rank, tp, group, device) else 1
    if args.check_keys:
        return 0 if check_keys(args, rank, tp, group, device) else 1
    kda, dsa = args.kind == "kda", args.kind == "dsa"
    key = FlashBuild(
        rows=args.rows,
        tp=tp,
        launches_per_step=1,
        inter=GLOBAL_INTER // tp,
        kda=kda,
        dsa=dsa,
        max_pools=args.max_pools if dsa else 0,
        stop=False,
    )
    rt = FlashRuntime(key, rank, group, device)
    rt.set_deadline(args.deadline)
    op = FlashLayerOp(
        rt,
        ffn_weights(rank, tp, device),
        0,
        kda=kda_weights(rank, tp, device) if kda else None,
        dsa=dsa_weights(rank, tp, device) if dsa else None,
    )
    x, res, post, comb = layer_inputs(args.rows, device)
    extra = {}
    if kda:
        state, idx = kda_state(args.rows, tp, device)
        idx[args.rows - args.pad_rows :] = 0  # not positive: a pad row
        extra = {"state": state, "state_idx": idx}
    elif dsa:
        from sgl_amd_mono.glm53_flash.gfx950.layout import ROW_BT

        lens = [int(v) for v in args.lens.split(",")] if args.lens else None
        lens = lens or list(LENS[: args.rows])
        valid = args.rows - args.pad_rows
        caches, table, pt = dsa_caches(
            args.rows, device, lens[:valid] + [0] * args.pad_rows
        )
        # SGLang's page table has the step's rows; the width's pad rows read row 0
        pt = pt[:valid].clone()
        table[valid:, ROW_BT] = 0
        extra = {"caches": caches, "rows": table, "page_table": pt}
    else:
        x = x / tp  # the FFN launch takes this rank's unreduced attention share
    outs, written = [], True
    for n in range(args.launches):
        if args.writes and n == 0 and extra:
            from msgspec.structs import asdict

            held = asdict(extra.get("state") or extra["caches"])
            before = {k: t.clone() for k, t in held.items()}
        out = op.launch(x, res, post, comb, **extra)
        rt.advance_step()
        torch.cuda.synchronize()
        words = rt.epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist()
        outs.append(tuple(t.clone() for t in out))
        print(f"rank {rank} launch {n}: error words {words}", flush=True)
        if any(words):
            return 1
        if args.writes and n == 0 and extra:
            written = writes_only_its_rows(extra, before)
            print(f"rank {rank}: writes only its rows {written}", flush=True)
    xs = [torch.zeros_like(outs[-1][0]) for _ in range(tp)]
    dist.all_gather_object(xs, outs[-1][0].cpu())
    same = all(torch.equal(xs[0], t) for t in xs)
    finite = bool(torch.isfinite(outs[-1][0].float()).all())
    print(f"rank {rank}: ranks agree {same}, finite {finite}", flush=True)
    rt.close()
    return 0 if same and finite and written else 1


if __name__ == "__main__":
    sys.exit(main())
