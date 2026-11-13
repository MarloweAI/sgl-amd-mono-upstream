"""Check K2, the attention back fused into the FFN launch, stage by stage (torchrun, TP2 or TP4).

XO against a torch attention, X8B against wo_a of K2's own XO; the index-source, window-only and
token-parallel (this rank's rows) K2 bit for bit against the K2 of the same keys; where the width
runs token parallel, the wo_b partial that K2 pushed against wo_b of its X8B, and K2's output bit
for bit against the FFN launch fed that partial; then a graph replay."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from check_ffn import (
    HIDDEN,
    bf16_ulps,
    cos,
    deq_fp8_block,
    finish,
    make_weights,
    mono_weights,
    mxfp8_qdq,
    rel,
    seam_inputs,
)

from sgl_amd_mono.deepseek_v41.spec import NUMERICS, SPEC

HEAD_DIM, NOPE, ROPE = 512, 448, 64
DATA, SCALE_BYTES, REC = 576, 8, 584
WINDOW, TOPK = 128, 512
O_RANK = 1024
SWA_BLOCK, COMP_BLOCK = 64, 128
SWA_PAGES, COMP_PAGES = 48, 32
POSITIONS = 4096
WIDTHS = SPEC.devices["gfx950"].widths


def page_bytes(block):
    """A V4 page: ``block`` data rows, then ``block`` scale rows, padded to 576 B."""
    return -(-block * REC // DATA) * DATA


def make_cache(pages, block, g, dev):
    """A paged V4 cache of random records: (raw bytes [pages, page bytes], nope f32, rope f32)."""
    nope = (torch.randn(pages, block, NOPE, generator=g, device=dev) * 64).clamp(
        -448, 448
    )
    nope = nope.to(torch.float8_e4m3fn)
    rope = torch.randn(pages, block, ROPE, generator=g, device=dev).to(torch.bfloat16)
    # 7 UE8M0 codes (a 64-value tile each) and a pad byte
    codes = torch.randint(
        118, 126, (pages, block, 7), generator=g, device=dev, dtype=torch.uint8
    )
    scales = torch.cat([codes, torch.zeros_like(codes[..., :1])], -1)
    raw = torch.zeros(pages, page_bytes(block), dtype=torch.uint8, device=dev)
    data = torch.cat([nope.view(torch.uint8), rope.view(torch.uint8)], -1)
    raw[:, : block * DATA] = data.reshape(pages, -1)
    raw[:, block * DATA : block * REC] = scales.reshape(pages, -1)
    k_nope = nope.float() * torch.exp2(codes.float() - 127).repeat_interleave(64, -1)
    keys = torch.cat([k_nope, rope.float()], -1).reshape(pages * block, HEAD_DIM)
    return raw, keys


def key_lists(M, g, dev, pad_rows=()):
    """Each token's window slots and top-k compressed slots (-1: an invalid one), and the K2 key table."""
    kt = torch.full((M, WINDOW + TOPK), -1, dtype=torch.int32, device="cpu")
    klen = torch.zeros(2 * M, dtype=torch.int32, device="cpu")
    lists = []
    for t in range(M):
        if t in pad_rows:
            lists.append(([], []))
            continue
        nswa = int(torch.randint(1, WINDOW + 1, (1,), generator=g))
        ntopk = int(torch.randint(0, TOPK + 1, (1,), generator=g))
        swa = torch.randperm(SWA_PAGES * SWA_BLOCK, generator=g)[:nswa].tolist()
        comp = torch.randperm(COMP_PAGES * COMP_BLOCK, generator=g)[:ntopk].tolist()
        if ntopk > 2:
            comp[ntopk // 2] = -1  # an invalid top-k slot reads the zero record, masked
        for k, c in enumerate(comp):
            kt[t, k] = -1 if c < 0 else c | -(2**31)
        for j, s in enumerate(swa):
            kt[t, ntopk + j] = s
        klen[t], klen[M + t] = ntopk + nswa, ntopk
        lists.append((swa, comp))
    return kt.to(dev), klen.to(dev), lists


def source_keys(M, lists, dev):
    """The key table's sources, as K1 and an index source's K2 read them: SWA slots
    (-1 on a pad row), window slots and lengths, top-k slots and lengths."""
    slot = torch.full((M,), -1, dtype=torch.int64)
    swa_idx = torch.full((M, WINDOW), -1, dtype=torch.int32)
    topk = torch.full((M, TOPK), -1, dtype=torch.int32)
    swa_lens = torch.zeros(M, dtype=torch.int32)
    topk_lens = torch.zeros(M, dtype=torch.int32)
    for t, (swa, comp) in enumerate(lists):
        if not swa:
            continue  # a pad row
        slot[t] = swa[0]
        swa_idx[t, : len(swa)] = torch.tensor(swa, dtype=torch.int32)
        topk[t, : len(comp)] = torch.tensor(comp, dtype=torch.int32)
        swa_lens[t], topk_lens[t] = len(swa), len(comp)
    keys = tuple(x.to(dev) for x in (swa_idx, swa_lens, topk, topk_lens))
    return slot.to(dev), keys


def window_only(M, lists, dev):
    """The key table of the same windows without top-k rows (a compress ratio 0 layer)."""
    kt = torch.full((M, WINDOW + TOPK), -1, dtype=torch.int32)
    klen = torch.zeros(2 * M, dtype=torch.int32)
    for t, (swa, _) in enumerate(lists):
        kt[t, : len(swa)] = torch.tensor(swa, dtype=torch.int32)
        klen[t] = len(swa)
    return kt.to(dev), klen.to(dev)


def attention_xo(q, swa_keys, comp_keys, lists, sink, pos, cos_sin):
    """[M, H * 512] XO as K2 computes it: attention with the sink, bf16 o, inverse GPT-J
    RoPE of dims 448.., bf16 (exact softmax: K2 rounds p and partials to bf16)."""
    M, H, _ = q.shape
    o = torch.zeros(M, H, HEAD_DIM, device=q.device)
    for t, (swa, comp) in enumerate(lists):
        rows = [comp_keys[c] for c in comp if c >= 0] + [swa_keys[s] for s in swa]
        if not rows:
            continue
        k = torch.stack(rows).bfloat16().float()
        s = (q[t].float() @ k.T) * HEAD_DIM**-0.5  # [H, keys]
        m = torch.maximum(s.amax(1), sink)
        p = torch.exp(s - m[:, None])
        o[t] = (p @ k) / (p.sum(1) + torch.exp(sink - m))[:, None]
    o = o.bfloat16().float()
    cs = cos_sin[pos]  # [M, 64]: 32 cos, then 32 sin
    c, sn = cs[:, None, :32], cs[:, None, 32:]
    e, od = o[..., NOPE::2], o[..., NOPE + 1 :: 2]
    r = torch.stack([e * c + od * sn, od * c - e * sn], -1).flatten(-2)
    return torch.cat([o[..., :NOPE], r], -1).reshape(M, H * HEAD_DIM).bfloat16().float()


def pushed_partials(runner, rows, tp):
    """[tp, rows, 5120] bf16: every rank's wo_b partial, as the last K2 (token parallel)
    pushed this rank's rows of it into the ATTN region (bf16 pairs next to their tags,
    ``seam.attn_region_pair``), gathered over the ranks. A K2 that is not token parallel
    pushes cells, which its reduce empties once read (``seam.attn_cell``)."""
    from sgl_amd_mono.deepseek_v41.gfx950.layer import peer_half_bytes

    ep = int(runner.epoch[0].item()) - 1  # the K2 launch's epoch (run_ffn moved it on)
    base = (ep & 1) * peer_half_bytes(tp)
    own = rows // tp
    words = runner.peer.bytes[base : base + tp * own * HIDDEN // 2 * 8].view(
        torch.int32
    )
    mine = words.view(tp, own, HIDDEN // 2, 2)[..., 0].contiguous().view(torch.bfloat16)
    gathered = [torch.empty_like(mine) for _ in range(tp)]
    dist.all_gather(gathered, mine)
    return torch.cat(gathered, dim=1)


def mx_region(scratch, layout, name, rows, cols):
    """A scratch region of MXFP8 rows (e4m3 words, then an i32 E8M0 code a 32) dequantized."""
    qo, so = layout[name][0], layout[name + "s"][0]
    q = (
        scratch[qo : qo + rows * cols]
        .view(torch.float8_e4m3fn)
        .view(rows, cols)
        .float()
    )
    code = scratch[so : so + rows * cols // 32 * 4].view(torch.int32).view(rows, -1)
    return (q.view(rows, -1, 32) * torch.exp2(code.float() - 127)[..., None]).view(
        rows, cols
    )


def x8b_tagged(scratch, layout, rows, cols):
    """X8B handed to wo_b as tagged 16-B words (``plan.tagged_x8b``: three a (row, wo_a task), its
    32 e4m3 as 12 + 12 + 8 bytes and the E8M0 code in the last), dequantized as ``mx_region``."""
    tasks = cols // 32
    off = layout["x8t"][0]
    w = (
        scratch[off : off + rows * tasks * 3 * 16]
        .view(torch.int32)
        .view(rows, tasks, 3, 4)
    )
    data = torch.cat(
        [w[:, :, 0, :3], w[:, :, 1, :3], w[:, :, 2, :2]], dim=2
    ).contiguous()
    q = data.view(torch.float8_e4m3fn).view(rows, tasks, 32).float()
    code = w[:, :, 2, 2]
    return (q * torch.exp2(code.float() - 127)[..., None]).view(rows, cols)


def k2_timeline(M, tl) -> str:
    """Every mark K2 passed (the back's 0 .., the FFN's 16 ..) in median order, in µs."""
    marks = [k for k in range(28) if bool((tl[:, k] != 0).all())]
    t0 = tl[:, marks[0]].min()
    rel = (tl[:, marks] - t0).double() / 100.0  # 100 MHz ticks
    order = sorted(range(len(marks)), key=lambda i: rel[:, i].median().item())
    last = int(tl[:, 27].argmax()) if 27 in marks else 0
    lines = [f"M={M} K2 timeline (median CTA / last CTA / CTA {last}), us:"]
    for i in order:
        lines.append(
            f"  mark {marks[i]:2d} {rel[:, i].median().item():8.2f}"
            f" {rel[:, i].max().item():8.2f} {rel[last, i].item():8.2f}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument(
        "--timeline", action="store_true", help="print each width's stage timeline"
    )
    ap.add_argument("--widths", default="", help="comma-separated widths to run")
    # K2 reads wo_b in aiter's (16, 16) preshuffle; the references keep row-major
    ap.add_argument("--shuffled", action="store_true")
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.gfx950.attention.plan import (
        Dims,
    )
    from sgl_amd_mono.deepseek_v41.gfx950.layer import (
        scratch_layout,
    )
    from sgl_amd_mono.deepseek_v41.gfx950.runner import (
        DSV41MonoRunner,
        K2Weights,
    )
    from sgl_amd_mono.deepseek_v41.gfx950.token_par import token_par

    d = Dims(tp)
    w, _, _ = make_weights(tp, rank, dev)
    mw = mono_weights(w)
    mw.check(tp)
    if args.timeline:  # the runner builds its launches with stage marks
        os.environ["SGLANG_AMD_MONO_TIMELINE"] = "1"
    runner = DSV41MonoRunner(tp, rank, cpu, dev, shuffled=args.shuffled)

    # the KV caches are alike on every rank; the heads, wo_a and wo_b are this rank's
    gc = torch.Generator(device=dev).manual_seed(5)
    swa_raw, swa_keys = make_cache(SWA_PAGES, SWA_BLOCK, gc, dev)
    comp_raw, comp_keys = make_cache(COMP_PAGES, COMP_BLOCK, gc, dev)
    gr = torch.Generator(device=dev).manual_seed(77 + rank)
    o_rows = d.groups * O_RANK

    def e8m0(*shape):
        return torch.randint(
            118, 124, shape, generator=gr, device=dev, dtype=torch.uint8
        )

    woa = (torch.randn(o_rows, d.group_k, generator=gr, device=dev) * 4).clamp(
        -448, 448
    )
    wob = (torch.randn(HIDDEN, o_rows, generator=gr, device=dev) * 4).clamp(-448, 448)
    woa = (woa.to(torch.float8_e4m3fn), e8m0(o_rows // 32, d.group_k // 32))
    wob = (wob.to(torch.float8_e4m3fn), e8m0(HIDDEN // 32, o_rows // 32))
    freqs = torch.outer(
        torch.arange(POSITIONS, device=dev).float(),
        1.0 / 10000 ** (torch.arange(32, device=dev).float() / 32),
    )
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], -1).contiguous()
    sink = torch.randn(d.heads, generator=gr, device=dev)
    wob_k = wob[0]
    if args.shuffled:
        from aiter.ops.shuffle import shuffle_weight

        wob_k = shuffle_weight(wob[0], (16, 16))
    aw = K2Weights(
        woa=woa[0], woa_s=woa[1], wob=wob_k, wob_s=wob[1], sink=sink, cos_sin=cos_sin
    )
    aw.check(d)

    fails, steps = [], []
    widths = tuple(int(w) for w in args.widths.split(",")) if args.widths else WIDTHS
    for M in widths:
        # alike on every rank: the step's metadata
        gk = torch.Generator().manual_seed(1000 + M)
        pad = (M - 1,) if M >= 6 else ()
        kt, klen, lists = key_lists(M, gk, dev, pad_rows=pad)
        pos = torch.randint(0, POSITIONS, (M,), generator=gk).to(dev)
        _, res, post, comb, pre = seam_inputs(M, tp, rank, dev)
        q = (
            torch.randn(M, d.heads, HEAD_DIM, generator=gr, device=dev) * 0.05
        ).bfloat16()
        runner.q[:M] = q
        runner.res_mid[:M], runner.post_a[:M] = res, post
        runner.comb_a[:M], runner.pre_a[:M] = comb, pre
        runner.kt[: M * (WINDOW + TOPK)] = kt.reshape(-1)
        runner.klen[: 2 * M] = klen
        out, res_o, post_o, comb_o, pre_o = runner.k2(
            aw, mw, M, pos, swa_raw, SWA_BLOCK, comp_raw, COMP_BLOCK
        )
        torch.cuda.synchronize()
        if args.timeline:
            for _ in range(5):
                runner.k2(aw, mw, M, pos, swa_raw, SWA_BLOCK, comp_raw, COMP_BLOCK)
            torch.cuda.synchronize()
            if rank == 0:
                print(k2_timeline(M, runner.timeline(M).cpu()), flush=True)
        lay, sc = scratch_layout(M, tp, **runner.route_at(M)), runner.scratch(M)
        live = torch.tensor([t not in pad for t in range(M)], device=dev)
        # XO: the attention, combine and inverse RoPE
        o = lay["xo"][0]
        xo = sc[o : o + M * d.q_rows * 2].view(torch.bfloat16).view(M, d.q_rows).float()
        xo_ref = attention_xo(q, swa_keys, comp_keys, lists, sink, pos, cos_sin)
        row_cos = F.cosine_similarity(xo[live], xo_ref[live], dim=1)
        # X8B: wo_a of K2's own XO
        if "x8t" in lay:
            x8b = x8b_tagged(sc, lay, M, o_rows)
        else:
            x8b = mx_region(sc, lay, "x8b", M, o_rows)
        wa = deq_fp8_block(*woa).reshape(d.groups, O_RANK, -1)
        ya = torch.einsum("tgk,grk->tgr", xo.view(M, d.groups, -1), wa)
        x8b_ref = mxfp8_qdq(ya.bfloat16().float().reshape(M, -1), "vllm")
        part_ref = (x8b @ deq_fp8_block(*wob).T).bfloat16()
        checks = {
            "finite": all(
                torch.isfinite(x).all().item()
                for x in (out, res_o, post_o, comb_o, pre_o)
            ),
            # bf16 p / partials against an exact softmax, then MXFP8 rounding flips: ~1% rel
            "xo": cos(xo[live], xo_ref[live]) > NUMERICS.bound("k2.attn_out.cosine")
            and row_cos.min().item() > 0.999,
            "pad rows read nothing": bool((xo[~live] == 0).all()),
            "x8b": rel(x8b, x8b_ref) < NUMERICS.bound("k2.wo_a.rel"),
        }
        part_note = "n/a (not token parallel)"
        outs = (out, res_o, post_o, comb_o, pre_o)
        # an index source's K2 builds the table from K1's slots and top-k: same table and outputs
        slot, keys = source_keys(M, lists, dev)
        runner.kt.fill_(-7)
        runner.klen.fill_(-7)
        got = runner.k2(
            aw, mw, M, pos, swa_raw, SWA_BLOCK, comp_raw, COMP_BLOCK,
            slot=slot, keys=keys, index=True,
        )  # fmt: skip
        torch.cuda.synchronize()
        checks["index source"] = (
            all(torch.equal(a, b) for a, b in zip(got, outs))
            and torch.equal(runner.kt[: M * (WINDOW + TOPK)], kt.reshape(-1))
            and torch.equal(runner.klen[: 2 * M], klen)
        )
        # a window-only K2 (no compressed cache) against the full K2 on a window-keys-only table
        kt0, klen0 = window_only(M, lists, dev)
        pair = []
        for comp in (comp_raw, None):
            runner.kt[: M * (WINDOW + TOPK)] = kt0.reshape(-1)
            runner.klen[: 2 * M] = klen0
            pair.append(runner.k2(aw, mw, M, pos, swa_raw, SWA_BLOCK, comp, COMP_BLOCK))
        torch.cuda.synchronize()
        checks["window only"] = all(torch.equal(a, b) for a, b in zip(*pair))
        if token_par(M, tp):
            # token parallel: this rank's rows of every output, bit for bit
            n = M // tp
            mine = slice(rank * n, (rank + 1) * n)
            for index in (False, True):
                runner.kt[: M * (WINDOW + TOPK)] = kt.reshape(-1)
                runner.klen[: 2 * M] = klen
                extra = dict(slot=slot, keys=keys, index=True) if index else {}
                tpo = runner.k2(
                    aw, mw, M, pos, swa_raw, SWA_BLOCK, comp_raw, COMP_BLOCK,
                    tok_par=True, **extra,
                )  # fmt: skip
                torch.cuda.synchronize()
                checks[f"token parallel{' index' if index else ''}"] = all(
                    torch.equal(a[mine], b[mine]) for a, b in zip(tpo, outs)
                )
                if index:
                    continue
                # wo_b: the partials it pushed, before the FFN launch below overwrites them
                own = pushed_partials(runner, M, tp)[rank]
                part_ulps, part_share = bf16_ulps(own[live], part_ref[live])
                # the GEMV's summation order: few elements apart (more ulps near cancelling)
                checks["wo_b partial"] = (
                    rel(own[live], part_ref[live]) < NUMERICS.bound("k2.wo_b.rel")
                    and part_share <= 0.02
                )
                # K2's output (its cells' all-reduce): bit for bit the FFN launch fed K2's
                # own partial (the tagged pairs' all-reduce)
                ref = runner.ffn(mw, own.contiguous(), res, post, comb, pre)
                checks["ffn stages"] = all(torch.equal(a, b) for a, b in zip(ref, outs))
                part_note = (
                    f"rel {rel(own[live], part_ref[live]):.2e}, {100 * part_share:.3f}% "
                    f"differ (max {part_ulps:.0f} ulp), ffn stages bit for bit "
                    f"{checks['ffn stages']}"
                )
        ref0 = out.clone()
        dist.broadcast(ref0, 0)
        checks["ranks agree"] = torch.equal(ref0, out)
        bad = [k for k, v in checks.items() if not v]
        if rank == 0:
            print(
                f"M={M:2d} {'PASS' if not bad else 'FAIL ' + ','.join(bad)} | "
                f"xo cos {cos(xo[live], xo_ref[live]):.6f} rel {rel(xo[live], xo_ref[live]):.4f} "
                f"min-row {row_cos.min().item():.5f} | x8b rel {rel(x8b, x8b_ref):.5f} | "
                f"wo_b partial {part_note}",
                flush=True,
            )
        fails += [f"M={M} {k}" for k in bad]
        steps.append(
            (M, pos, kt, klen, q, res, post, comb, pre, False, runner._outs(M, res))
        )
        steps.append(
            (
                M,
                pos,
                (slot, keys),
                None,
                q,
                res,
                post,
                comb,
                pre,
                False,
                runner._outs(M, res),
            )
        )
        if token_par(M, tp):
            steps.append(
                (M, pos, kt, klen, q, res, post, comb, pre, True, runner._outs(M, res))
            )

    # the hand-offs under replay: every width's K2 in one graph, bit for bit against eager
    def run_all():
        for M, pos, kt, klen, q, res, post, comb, pre, tok_par, outs in steps:
            runner.q[:M].copy_(q)
            runner.res_mid[:M].copy_(res)
            runner.post_a[:M].copy_(post)
            runner.comb_a[:M].copy_(comb)
            runner.pre_a[:M].copy_(pre)
            if klen is None:  # an index source's K2: it builds the table
                slot, keys = kt
                runner.k2(
                    aw, mw, M, pos, swa_raw, SWA_BLOCK, comp_raw, COMP_BLOCK,
                    outs=outs, slot=slot, keys=keys, index=True,
                )  # fmt: skip
                continue
            runner.kt[: M * (WINDOW + TOPK)].copy_(kt.reshape(-1))
            runner.klen[: 2 * M].copy_(klen)
            runner.k2(
                aw, mw, M, pos, swa_raw, SWA_BLOCK, comp_raw, COMP_BLOCK, outs=outs,
                tok_par=tok_par,
            )  # fmt: skip

    # token-parallel steps write only their rows; a sentinel elsewhere, as NaN never compares equal
    sentinel = -3.25
    for st in steps:
        if st[9]:
            for t in st[-1]:
                t.fill_(sentinel)
    run_all()
    torch.cuda.synchronize()
    eager = [t.clone() for s in steps for t in s[-1]]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    bad_replays = 0
    for i in range(args.replays):
        graph.replay()
        if i % 50 == 49 or i == args.replays - 1:
            torch.cuda.synchronize()
            got = [t for s in steps for t in s[-1]]
            bad_replays += sum(not torch.equal(a, b) for a, b in zip(got, eager))
    for M, *_, tok_par, outs in steps:
        if tok_par:
            others = torch.ones(M, dtype=torch.bool, device=dev)
            others[rank * (M // tp) : (rank + 1) * (M // tp)] = False
            if any((t[others].float() != sentinel).any().item() for t in outs):
                fails.append(f"M={M} token parallel wrote another rank's rows")
    if rank == 0:
        print(f"replay check: {args.replays} replays of widths {WIDTHS} in one graph: "
              f"{bad_replays} mismatching checks", flush=True)  # fmt: skip
    if bad_replays:
        fails.append("replay")
    finish(fails, dev)


if __name__ == "__main__":
    main()
