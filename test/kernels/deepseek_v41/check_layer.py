"""Check one launch per layer (torchrun, TP2 or TP4): a reuse layer's layer launch bit for
bit against its K1 then its K2 (two launches) on the same inputs, every row's window
holding the step's own new records; with and without an owed post, with and without
compressed keys, and token parallel where the width allows; then every step in one graph,
bit for bit under replay."""

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
SENTINEL = (
    -3.25
)  # in the outputs before a launch: a token-parallel step writes its rows only


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
        K1Weights,
        K2Weights,
    )
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

    def launches(M, step, fold, compressed, tok_par, one, outs):
        x, res, post, comb, pre, pos, slot, keys = step
        cc = comp if compressed else None
        if not compressed:  # a window-only layer has no top-k (as the server passes it)
            keys = (*keys[:2], runner.no_topk, runner.no_topk_lens)
        x = x if fold else None
        if one:
            return runner.layer(
                k1w, aw, mw, x, res, post, comb, pre, pos, slot, swa, SWA_BLOCK, cc,
                COMP_BLOCK, keys, outs=outs, tok_par=tok_par,
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
