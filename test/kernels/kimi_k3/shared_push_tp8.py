# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's split MoE all-reduce (``kimi_k3/gfx950/stages/shared_push.py``) on 8 real ranks: check and time
against the flat all-reduce.

Per width, a graph of ``--layers`` MoE layers runs one of:

* ``stock``: aiter's custom all-reduce of SGLang's flat [latent | shared] buffer, then SGLang's RMSNorm of the
  latent (the stock path);
* ``flat``: C13's flat launch (``ar_norm.build_moe``: both sums and the latent's norm);
* ``push``: the standalone shared push alone (``build_push``: its wire time, which a hosting launch hides);
* ``push_split``: the push, ``--gap-us`` of HBM-bound work standing in for the routed experts, then the split
  launch (``build_split``: the latent half's all-reduce + norm + the pushed shared sum);
* ``gap`` / ``gap_flat``: the stand-in alone and before the flat launch (so push_split - gap is the split's cost
  with the slots landed, and gap_flat - gap the flat launch's).

Check (every rank, all-reduced counts): the split's three outputs against the rank-order f32 sum (bit for bit)
and against the flat launch's (bit for bit). Times: µs a layer, median and slowest rank.

    torchrun --nproc-per-node 8 shared_push_tp8.py --rows 54,96,168
"""

import argparse
import json
import os

import torch
import torch.distributed as dist

TP = 8
LAT, SHR = 3584, 7168
BF = torch.bfloat16


def rank_order(x):
    acc = x[0].float()
    for r in range(1, TP):
        acc = acc + x[r].float()
    return acc.to(BF)


def bad(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def across(v, dev):
    t = torch.tensor([v], dtype=torch.float64, device=dev)
    every = [torch.empty_like(t) for _ in range(TP)]
    dist.all_gather(every, t)
    vals = sorted(float(e.item()) for e in every)
    return (vals[3] + vals[4]) / 2, vals[-1]


def graph_us(body, reps, per, capture=None):
    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    if capture is not None:
        with capture():
            with torch.cuda.graph(g):
                body()
    else:
        with torch.cuda.graph(g):
            body()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / (reps * per)


def epoch_buf(dev, deadline):
    from sgl_amd_mono.common import guard

    ep = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(deadline)
    lo = ticks & 0xFFFFFFFF
    ep[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    ep[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return ep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="54,96,168")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--gap-us", type=float, default=100.0)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--deadline", type=float, default=10.0)
    p.add_argument("--json", default="")
    args = p.parse_args()
    rows = [int(r) for r in args.rows.split(",")]

    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from sglang.srt.layers.layernorm import RMSNorm

    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.stages import ar_norm as N
    from sgl_amd_mono.kimi_k3.gfx950.stages import shared_push as S

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)
    print(f"rank {rank}: process groups up", flush=True)
    nl = args.layers
    push = S.build_push(nl)
    split = S.build_split(args.eps, nl)
    flat = N.build_moe(args.eps, nl)
    sp = PeerBuffer(S.peer_bytes(), cpu, rank, TP, dev, init=S.arm)
    fp = PeerBuffer(N.moe_peer_bytes(), cpu, rank, TP, dev, init=N.moe_arm)
    sep, fep = epoch_buf(dev, args.deadline), epoch_buf(dev, args.deadline)
    ca = CustomAllreduce(cpu, dev)
    w = (1 + 0.1 * torch.randn(LAT, generator=torch.Generator(device=dev).manual_seed(5), device=dev)).to(BF)
    norm = RMSNorm(LAT, eps=args.eps).to(dev)
    norm.weight.data.copy_(w)
    # the routed experts' stand-in: HBM-bound copies of about --gap-us
    gsrc = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    gdst = torch.empty_like(gsrc)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    gdst.copy_(gsrc)
    e0.record()
    gdst.copy_(gsrc)
    e1.record()
    torch.cuda.synchronize()
    gn = max(1, int(len(gsrc) * args.gap_us / (e0.elapsed_time(e1) * 1000)))
    gn = min(gn, len(gsrc))

    def gap():
        gdst[:gn].copy_(gsrc[:gn])

    out = {"rank": rank, "check": [], "time": []}
    ok = True
    for m in rows:
        g = torch.Generator(device=dev).manual_seed(777 + m)
        xl = (torch.randn(TP, m, LAT, generator=g, device=dev) * 0.3).to(BF)
        xs = (torch.randn(TP, m, SHR, generator=g, device=dev) * 0.3).to(BF)
        al, as_ = rank_order(xl), rank_order(xs)
        pl = xl[rank].unsqueeze(0).expand(nl, m, LAT).contiguous()
        ps = xs[rank].unsqueeze(0).expand(nl, m, SHR).contiguous()
        ol, on, os_ = torch.zeros_like(pl), torch.zeros_like(pl), torch.zeros_like(ps)
        fl, fn, fs = torch.zeros_like(pl), torch.zeros_like(pl), torch.zeros_like(ps)
        cur = torch.cuda.current_stream

        def push_layer(i):
            push(ps[i].data_ptr(), m, sp.addresses.data_ptr(), rank, sep.data_ptr(), i, stream=cur())

        def split_layer(i):
            split(pl[i].data_ptr(), ps[i].data_ptr(), w.data_ptr(), ol[i].data_ptr(), os_[i].data_ptr(),
                  on[i].data_ptr(), m, sp.addresses.data_ptr(), rank, sep.data_ptr(), i, stream=cur())  # fmt: skip

        def flat_layer(i):
            flat(pl[i].data_ptr(), ps[i].data_ptr(), w.data_ptr(), fl[i].data_ptr(), fs[i].data_ptr(),
                 fn[i].data_ptr(), m, fp.addresses.data_ptr(), rank, fep.data_ptr(), i, stream=cur())  # fmt: skip

        flats = [torch.cat([pl[i].reshape(-1), ps[i].reshape(-1)]) for i in range(nl)]

        def stock_layer(i):
            r = ca.custom_all_reduce(flats[i])
            norm(r[: m * LAT].view(m, LAT))

        def arm(name):
            def body():
                if name in ("push", "push_split"):
                    sep[0:1].add_(1)
                if name in ("flat", "gap_flat"):
                    fep[0:1].add_(1)
                for i in range(nl):
                    if name in ("push", "push_split"):
                        push_layer(i)
                    if name in ("gap", "gap_flat", "push_split"):
                        gap()
                    if name == "push_split":
                        split_layer(i)
                    if name in ("flat", "gap_flat"):
                        flat_layer(i)
                    if name == "stock":
                        stock_layer(i)

            return body

        times = {}
        for name in ("stock", "flat", "push", "gap", "gap_flat", "push_split"):
            us = graph_us(arm(name), args.reps, nl, ca.capture if name == "stock" else None)
            times[name] = across(us, dev)
        nrm = norm(al.clone())
        bads = torch.tensor([sum(bad(os_[i], as_) for i in range(nl)), sum(bad(ol[i], al) for i in range(nl)),
                             sum(bad(os_[i], fs[i]) for i in range(nl)), sum(bad(ol[i], fl[i]) for i in range(nl)),
                             sum(bad(on[i], fn[i]) for i in range(nl)), sum(bad(on[i], nrm) for i in range(nl))],
                            dtype=torch.int64, device=dev)  # fmt: skip
        dist.all_reduce(bads)
        b = dict(zip(("shared_vs_rankorder", "latent_vs_rankorder", "shared_vs_flat", "latent_vs_flat",
                      "norm_vs_flat", "norm_vs_stock_ulps"), bads.tolist()))  # fmt: skip
        passed = all(v == 0 for k, v in b.items() if k != "norm_vs_stock_ulps")
        ok &= passed
        out["check"].append({"m": m, **b, "pass": passed})
        t = {k: {"median": round(v[0], 2), "slowest": round(v[1], 2)} for k, v in times.items()}
        t["split_landed"] = {k: round(times["push_split"][j] - times["gap"][j] - times["push"][j], 2)
                             for j, k in ((0, "median"), (1, "slowest"))}  # fmt: skip
        t["flat_after_gap"] = {k: round(times["gap_flat"][j] - times["gap"][j], 2)
                               for j, k in ((0, "median"), (1, "slowest"))}  # fmt: skip
        out["time"].append({"m": m, **t})
        if rank == 0:
            print("TP8CHECK", json.dumps({"m": m, **b, "pass": passed}), flush=True)
            print("TP8TIME", json.dumps({"m": m, **t}), flush=True)
    if rank == 0 and args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    if rank == 0:
        print("SHARED_PUSH_TP8", "PASS" if ok else "FAIL", flush=True)
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
