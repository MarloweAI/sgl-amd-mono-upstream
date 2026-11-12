"""Check one launch per layer (torchrun, TP2 or TP4): a reuse layer's layer launch bit for
bit against its K1 then its K2 (two launches) on the same inputs, every row's window
holding the step's own new records; with and without an owed post, with and without
compressed keys, and token parallel where the width allows; the first layer on the
embedding table bit for bit against the same launch on the broadcast rows; with the next
layer's Engram projection, its outputs bit for bit as without and every rank's kv the same,
within the bf16 rounding of an fp64 product of the MXFP8 operands; then every step in one
graph, bit for bit under replay."""

import argparse
import os

import torch
import torch.distributed as dist
from check_ffn import HIDDEN, finish, make_weights, mono_weights, seam_inputs
from check_k1 import attn_weights, rope_table
from check_k2 import (
    COMP_BLOCK,
    COMP_PAGES,
    O_RANK,
    SWA_BLOCK,
    SWA_PAGES,
    TOPK,
    WINDOW,
    make_cache,
)

POSITIONS = 131072
VOCAB = 4096  # the test's embedding table rows
# in the outputs before a launch: a token-parallel step writes its rows only
SENTINEL = -3.25


def step_inputs(M, tp, rank, dev):
    """One step's K1 inputs and keys, alike on every rank but the seam's partial; row t's
    window ends in the step's rows 0 .. t (the records this launch writes)."""
    gk = torch.Generator().manual_seed(1000 + M)
    _, res, post, comb, pre = seam_inputs(M, tp, rank, dev)
    x = (torch.randn(M, HIDDEN, generator=gk) * 0.1).to(torch.bfloat16).to(dev)
    pos = torch.randint(WINDOW, POSITIONS, (M,), generator=gk).to(dev)
    perm = torch.randperm(SWA_PAGES * SWA_BLOCK, generator=gk)
    slot = perm[:M].int()
    old = perm[M:]
    swa_idx = torch.empty(M, WINDOW, dtype=torch.int32)
    swa_lens = torch.empty(M, dtype=torch.int32)
    for t in range(M):
        n = int(torch.randint(t + 1, WINDOW + 1, (1,), generator=gk))
        swa_idx[t, : n - t - 1] = old[
            torch.randperm(len(old), generator=gk)[: n - t - 1]
        ]
        swa_idx[t, n - t - 1 : n] = slot[: t + 1]
        swa_idx[t, n:] = -1
        swa_lens[t] = n
    topk = torch.stack(
        [torch.randperm(COMP_PAGES * COMP_BLOCK, generator=gk)[:TOPK] for _ in range(M)]
    ).int()
    topk_lens = torch.randint(0, TOPK + 1, (M,), generator=gk).int()
    keys = tuple(v.contiguous().to(dev) for v in (swa_idx, swa_lens, topk, topk_lens))
    return x, res, post, comb, pre, pos, slot.to(dev), keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,2,5,6,8,12,16,17,24,48")
    ap.add_argument("--replays", type=int, default=50)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.gfx950.attention.plan import Dims
    from sgl_amd_mono.deepseek_v41.gfx950.runner import (
        DSV41MonoRunner,
        EngramWeights,
        K1Weights,
        K2Weights,
    )
    from sgl_amd_mono.deepseek_v41.gfx950.stages import engram
    from sgl_amd_mono.deepseek_v41.gfx950.token_par import token_par

    d = Dims(tp)
    w, _, _ = make_weights(tp, rank, dev)
    mw = mono_weights(w)
    del w
    runner = DSV41MonoRunner(tp, rank, cpu, dev)
    g = torch.Generator(device=dev).manual_seed(11)
    a = attn_weights(d.heads, dev, g)
    cos_sin = rope_table(dev)
    k1w = K1Weights(
        hc_fn=mw.hc_ffn_fn.clone(), hc_scale=mw.hc_ffn_scale.clone(),
        hc_base=mw.hc_ffn_base.clone(), norm=mw.ffn_norm.clone(), cos_sin=cos_sin, **a,
    )  # fmt: skip
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
    aw = K2Weights(
        woa=woa.to(torch.float8_e4m3fn), woa_s=e8m0(o_rows // 32, d.group_k // 32),
        wob=wob.to(torch.float8_e4m3fn), wob_s=e8m0(HIDDEN // 32, o_rows // 32),
        sink=torch.randn(d.heads, generator=gr, device=dev), cos_sin=cos_sin,
    )  # fmt: skip
    aw.check(d)
    gc = torch.Generator(device=dev).manual_seed(5)
    swa0, _ = make_cache(SWA_PAGES, SWA_BLOCK, gc, dev)
    comp, _ = make_cache(COMP_PAGES, COMP_BLOCK, gc, dev)
    swa = swa0.clone()

    table = (torch.randn(VOCAB, HIDDEN, generator=g, device=dev) * 0.05).bfloat16()
    # the next layer's Engram projection: one wkv on every rank, each with its rows
    ge = torch.Generator(device=dev).manual_seed(23)
    wkv = (torch.randn(engram.N, engram.K, generator=ge, device=dev) * 64).clamp(
        -448, 448
    )
    wkv = wkv.to(torch.float8_e4m3fn)
    wkv_s = torch.randint(
        112, 120, (engram.N // 32, engram.K // 32), generator=ge, device=dev
    ).to(torch.uint8)
    n_eg = engram.rank_rows(tp)
    ew = EngramWeights(
        w=engram.tile_order(wkv[rank * n_eg : (rank + 1) * n_eg]),
        ws=wkv_s[rank * n_eg // 32 : (rank + 1) * n_eg // 32].contiguous(),
    )
    ew.check(tp)
    # the same projection shared by two launches (the reuse layers before an Engram layer)
    halves = [
        EngramWeights(w=ew.w[a:b], ws=ew.ws[a // 32 : b // 32].contiguous(), row0=a)
        for a, b in ((0, n_eg // 64 * 32), (n_eg // 64 * 32, n_eg))
    ]
    # wkv in fp64: each 32 x 32 block times its scale
    wkv64 = wkv.double() * torch.exp2(wkv_s.double() - 127).repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)

    def mx_rows(x):
        """The MXFP8 linear's input quant of bf16 rows (vLLM's scale rule), in fp64."""
        v = x.float().view(x.shape[0], -1, 32)
        a = (
            v.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny) / 448.0
        ).float()
        code = (torch.ceil(torch.log2(a.double())) + 127).clamp(0, 254)
        q = (v * torch.exp2(127 - code)[..., None].float()).clamp(-448, 448)
        q = q.to(torch.float8_e4m3fn).double()
        return (q * torch.exp2(code - 127)[..., None]).view(x.shape[0], -1)

    def launches(M, step, fold, compressed, tok_par, one, outs, embed=None, eg=None):
        x, res, post, comb, pre, pos, slot, keys = step
        cc = comp if compressed else None
        if embed is not None:  # the first layer: the table at the step's ids
            res = None
        if not compressed:  # a window-only layer has no top-k (as the server passes it)
            keys = (*keys[:2], runner.no_topk, runner.no_topk_lens)
        x = x if fold else None
        if one:
            return runner.layer(
                k1w, aw, mw, x, res, post, comb, pre, pos, slot, swa, SWA_BLOCK, cc,
                COMP_BLOCK, keys, outs=outs, tok_par=tok_par, embed=embed, eg=eg,
            )  # fmt: skip
        runner.k1(
            k1w, x, res, post, comb, pre, pos, slot, swa, SWA_BLOCK, keys,
            tok_par=tok_par,
        )  # fmt: skip
        return runner.k2(
            aw, mw, M, pos, swa, SWA_BLOCK, cc, COMP_BLOCK, outs=outs, tok_par=tok_par
        )

    def state(M, outs):
        """The launch's outputs, K1's outputs K2 read, and the SWA cache."""
        mid = (runner.q, runner.res_mid, runner.post_a, runner.comb_a, runner.pre_a)
        return [t.clone() for t in outs] + [t[:M].clone() for t in mid] + [swa.clone()]

    def check_engram(M, step, fold, compressed, want):
        """The launch with an Engram layer's rows, its projection in one launch, then in
        two halves: its outputs as without; kv."""
        gi = torch.Generator().manual_seed(500 + M)
        rows = (torch.randn(M, engram.K, generator=gi) * 0.3).bfloat16().to(dev)
        kvs, same = [], True
        for shares in ([ew], halves):
            runner.egkv.fill_(SENTINEL)
            for share in shares:
                outs = runner._outs(M, step[1])
                swa.copy_(swa0)
                launches(M, step, fold, compressed, False, True, outs, eg=(rows, share))
                torch.cuda.synchronize()
                same &= all(
                    x.view(torch.uint8).equal(y.view(torch.uint8))
                    for x, y in zip(state(M, outs), want)
                )
            kvs.append(runner.egkv[:M].clone())
        same &= kvs[0].view(torch.uint8).equal(kvs[1].view(torch.uint8))
        kv = kvs[0]
        every = [torch.empty_like(kv) for _ in range(tp)]
        dist.all_gather(every, kv)
        alike = all(e.view(torch.uint8).equal(kv.view(torch.uint8)) for e in every)
        xq = mx_rows(rows)
        ref = xq @ wkv64.T
        mag = xq.abs() @ wkv64.abs().T
        err = (kv.double() - ref).abs()
        # bf16 rounding of the f32 sum, and the f32 sum's own error (K terms)
        ok = bool((err <= ref.abs() * 2.0**-8 + mag * engram.K * 2.0**-24).all())
        rel = (err / mag.clamp_min(1e-30)).max().item()
        name = f"M={M} fold={fold} compressed={compressed} engram"
        if rank == 0:
            print(
                f"{name}: outputs {'bitwise' if same else 'DIFFER'} (kv in halves alike), kv "
                f"{'alike' if alike else 'DIFFERS'} on every rank, "
                f"{'within' if ok else 'PAST'} the bound (max |d| / sum |x w| {rel:.2e}, "
                f"{(kv.double() != ref.bfloat16().double()).float().mean().item():.1%} "
                "not the fp64 product's nearest bf16)",
                flush=True,
            )
        return [] if same and alike and ok else [name]

    fails, steps = [], []
    for M in (int(v) for v in args.widths.split(",")):
        step = step_inputs(M, tp, rank, dev)
        for fold in (True, False):
            for compressed in (True, False):
                for tok_par in sorted({False, token_par(M, tp)}):
                    got = []
                    for one in (False, True):
                        outs = runner._outs(M, step[1])
                        for t in outs:
                            t.fill_(SENTINEL)
                        for t in (
                            runner.q,
                            runner.res_mid,
                            runner.post_a,
                            runner.comb_a,
                            runner.pre_a,
                        ):
                            t.fill_(SENTINEL)
                        swa.copy_(swa0)
                        launches(M, step, fold, compressed, tok_par, one, outs)
                        torch.cuda.synchronize()
                        got.append(state(M, outs))
                    same = all(
                        x.view(torch.uint8).equal(y.view(torch.uint8))
                        for x, y in zip(*got)
                    )
                    name = (
                        f"M={M} fold={fold} compressed={compressed} tok_par={tok_par}"
                    )
                    if rank == 0:
                        print(f"{name}: {'bitwise' if same else 'DIFFERS'}", flush=True)
                    if not same:
                        fails.append(name)
                    outs = runner._outs(M, step[1])
                    steps.append((M, step, fold, compressed, tok_par, outs, got[1]))
                    if not tok_par and runner.eg_fused(M, tok_par):
                        fails += check_engram(M, step, fold, compressed, got[1])
                    if fold:
                        continue
                    # the first layer: on (table, ids) and on the rows hc_broadcast makes
                    gi = torch.Generator().manual_seed(M)
                    ids = torch.randint(0, VOCAB, (M,), generator=gi).to(dev)
                    rows = table[ids].unsqueeze(1).repeat(1, 4, 1).contiguous()
                    st = (step[0], rows, *step[2:])
                    got = []
                    for e in ((table, ids), None):
                        outs = runner._outs(M, rows)
                        for t in outs:
                            t.fill_(SENTINEL)
                        swa.copy_(swa0)
                        launches(M, st, False, compressed, tok_par, True, outs, e)
                        torch.cuda.synchronize()
                        got.append(state(M, outs))
                    same = all(
                        x.view(torch.uint8).equal(y.view(torch.uint8))
                        for x, y in zip(*got)
                    )
                    name = f"M={M} embedding compressed={compressed} tok_par={tok_par}"
                    if rank == 0:
                        print(f"{name}: {'bitwise' if same else 'DIFFERS'}", flush=True)
                    if not same:
                        fails.append(name)

    # the seam move: three layers A, B, C, each layer's attention seam run by the launch
    # before it (``layer(nxt=)``, ``layer(seamed=True)``), against three plain launches each on
    # the previous one's outputs; B's mix weights its own, so a launch that takes its own
    # layer's for the next one's differs
    k1b = K1Weights(
        hc_fn=(mw.hc_ffn_fn * 0.75).contiguous(), hc_scale=(mw.hc_ffn_scale * 1.25).contiguous(),
        hc_base=(mw.hc_ffn_base + 0.125).contiguous(), norm=(mw.ffn_norm * 1.5).contiguous(),
        cos_sin=cos_sin, **a,
    )  # fmt: skip
    chain = (k1w, k1b, k1w)

    def run_chain(M, step, compressed, seam):
        """The three layers on ``step``: their outputs, B's and C's attention seams' (R_new,
        mixes), B's capture and the SWA cache."""
        x, res, post, comb, pre, pos, slot, keys = step
        cc = comp if compressed else None
        if not compressed:
            keys = (*keys[:2], runner.no_topk, runner.no_topk_lens)
        swa.copy_(swa0)
        aux = torch.full((M, HIDDEN), SENTINEL, dtype=torch.bfloat16, device=dev)
        got, ins = [], (x, res, post, comb, pre)
        for i, w in enumerate(chain):
            kw = {}
            if seam and i > 0:
                ins, kw["seamed"] = (None,) * 5, True
            if seam and i + 1 < len(chain):
                kw["nxt"] = chain[i + 1]
                kw["nxt_aux"] = aux if i == 0 else None
            outs = runner._outs(M, step[1])
            for t in outs:
                t.fill_(SENTINEL)
            outs = runner.layer(
                w, aw, mw, *ins, pos, slot, swa, SWA_BLOCK, cc, COMP_BLOCK, keys,
                outs=outs, aux=aux if (i == 1 and not seam) else None, **kw,
            )  # fmt: skip
            mids = [
                runner.seam_res(runner.seam_state(M), M) if seam and i > 0 else runner.res_mid[:M],
                runner.post_a[:M], runner.comb_a[:M], runner.pre_a[:M],
            ]  # fmt: skip
            # a launch that runs the next seam does not write its MoE output, and a seamed
            # one's R_new is gone once its tail wrote the next seam's into the same state
            keep = outs[1:] if seam and i + 1 < len(chain) else outs
            if seam and 0 < i < len(chain) - 1:
                mids = mids[1:]
            got += [t.clone() for t in keep] + [t.clone() for t in mids]
            ins = outs
        return got + [aux.clone(), swa.clone()]

    def chain_names(seam):
        names = []
        for i in range(len(chain)):
            outs = ["res", "post", "comb", "pre"]
            if not (seam and i + 1 < len(chain)):
                outs = ["out"] + outs
            mids = ["res_mid", "post_a", "comb_a", "pre_a"]
            if seam and 0 < i < len(chain) - 1:
                mids = mids[1:]
            names += [f"{'ABC'[i]}.{n}" for n in outs + mids]
        return names + ["aux", "swa"]

    seam_cases = []
    for M in (int(v) for v in args.widths.split(",")):
        if not runner.seam_moves(M):
            continue
        step = step_inputs(M, tp, rank, dev)
        for compressed in (True, False):
            plain = dict(zip(chain_names(False), run_chain(M, step, compressed, False)))
            moved = dict(zip(chain_names(True), run_chain(M, step, compressed, True)))
            bad = [
                n for n, t in moved.items()
                if not t.view(torch.uint8).equal(plain[n].view(torch.uint8))
            ]  # fmt: skip
            name = f"M={M} compressed={compressed} seam move"
            if rank == 0:
                print(
                    f"{name}: {'bitwise' if not bad else 'DIFFERS ' + ' '.join(bad)}",
                    flush=True,
                )
            if bad:
                fails.append(name)
            seam_cases.append((M, step, compressed, moved))

    # the seam move's three launches in one graph: bit for bit against eager under replay
    if seam_cases:

        def seam_all():
            return [run_chain(M, st, cmp, True) for M, st, cmp, _ in seam_cases]

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs_g = seam_all()
        bad = 0
        for i in range(args.replays):
            graph.replay()
            if i % 10 == 9 or i == args.replays - 1:
                torch.cuda.synchronize()
                for got, (_, _, _, want) in zip(outs_g, seam_cases):
                    bad += sum(
                        not x.view(torch.uint8).equal(y.view(torch.uint8))
                        for x, y in zip(got, want.values())
                    )
        if rank == 0:
            print(
                f"seam move replays: {args.replays}, {bad} mismatching checks",
                flush=True,
            )
        if bad:
            fails.append("seam move replay")

    # every step's layer launch in one graph: bit for bit against eager under replay; the
    # steps share the cache their K1s write, so each pass starts from the same one
    def run_all():
        swa.copy_(swa0)
        for M, step, fold, compressed, tok_par, outs, _ in steps:
            launches(M, step, fold, compressed, tok_par, True, outs)

    run_all()
    torch.cuda.synchronize()
    eager = [t.clone() for s in steps for t in s[5]]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    bad = 0
    for i in range(args.replays):
        graph.replay()
        if i % 10 == 9 or i == args.replays - 1:
            torch.cuda.synchronize()
            got = [t for s in steps for t in s[5]]
            bad += sum(
                not x.view(torch.uint8).equal(y.view(torch.uint8))
                for x, y in zip(got, eager)
            )
    if rank == 0:
        print(f"replays: {args.replays}, {bad} mismatching checks", flush=True)
    if bad:
        fails.append("replay")
    finish(fails, dev)


if __name__ == "__main__":
    main()
