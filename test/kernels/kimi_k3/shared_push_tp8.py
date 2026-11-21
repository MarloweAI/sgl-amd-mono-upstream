# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's MoE layer end to end on 8 real ranks, from the fused MoE call through the all-reduce output: the
split all-reduce (the shared partial pushed by the stage-1 launch's shared tiles) against the flat one.

Per width, a CUDA graph of ``--layers`` MoE layers, each ``check_stage1_shared.run_case`` (routing, the fused
stage 1 with the shared down, stage 2) and then:

* ``stock``: aiter's custom all-reduce of SGLang's flat [latent | shared] buffer and SGLang's RMSNorm of the latent;
* ``flat``: C13's flat launch (``MoeAr.reduce``: both sums and the latent's norm);
* ``push``: the call with ``SharedPushAr.push_args(layer)`` (its shared tiles push), then ``SharedPushAr.reduce``
  (the latent half's all-reduce + norm + the pushed shared sum).

Times: µs a layer, median and slowest rank. Check (every rank, all-reduced counts), on each layer of a replayed
``push`` graph: the three outputs against the rank-order f32 sum of the ranks' own partials, and against
``MoeAr.reduce`` on the same partials, bit for bit.

    torchrun --nproc-per-node 8 shared_push_tp8.py --rows 54,96,168
"""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

TP = 8
LAT, SHR = 3584, 7168
BF = torch.bfloat16


def rank_order(parts):
    acc = parts[0].float()
    for r in range(1, TP):
        acc = acc + parts[r].float()
    return acc.to(BF)


def bad(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def across(v, dev):
    t = torch.tensor([v], dtype=torch.float64, device=dev)
    every = [torch.empty_like(t) for _ in range(TP)]
    dist.all_gather(every, t)
    vals = sorted(float(e.item()) for e in every)
    return (vals[3] + vals[4]) / 2, vals[-1]


def gathered(x):
    every = [torch.empty_like(x) for _ in range(TP)]
    dist.all_gather(every, x.contiguous())
    return every


def capture(body, ctx=None):
    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    if ctx is not None:
        with ctx():
            with torch.cuda.graph(g):
                body()
    else:
        with torch.cuda.graph(g):
            body()
    return g


def graph_us(g, reps, per):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="54,96,168")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--deadline", type=float, default=10.0)
    p.add_argument("--json", default="")
    args = p.parse_args()
    rows = [int(r) for r in args.rows.split(",")]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from check_stage1_shared import make_case, run_case
    from sglang.srt.layers.layernorm import RMSNorm

    from sgl_amd_mono.kimi_k3.ar_read import MoeAr, SharedPushAr

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)
    print(f"rank {rank}: process groups up", flush=True)
    nl = args.layers
    eps = 1e-6
    split = SharedPushAr(cpu, rank, dev, range(nl), eps, deadline_s=args.deadline)
    flat = MoeAr(cpu, rank, dev, range(nl), eps, deadline_s=args.deadline)
    ca = CustomAllreduce(cpu, dev)
    w = (1 + 0.1 * torch.randn(LAT, generator=torch.Generator(device=dev).manual_seed(5), device=dev)).to(BF)
    norm = RMSNorm(LAT, eps=eps).to(dev)
    norm.weight.data.copy_(w)
    out = {"rank": rank, "check": [], "time": []}
    ok = True
    for m in rows:
        assert split.covers(m), m
        case = make_case(m, 1000 * m + rank, dev)  # each rank its own partials
        keep = {}

        def arm(name):
            def body():
                res = []
                for i in range(nl):
                    if name == "push":
                        latent, shared = run_case(case, push=(split.launches, *split.push_args(i)))
                        res.append((latent, shared, *split.reduce(i, latent, shared, w)))
                    else:
                        latent, shared = run_case(case)
                        if name == "flat":
                            flat.reduce(i, latent, shared, w)
                        else:
                            r = ca.custom_all_reduce(torch.cat([latent.reshape(-1), shared.reshape(-1)]))
                            norm(r[: m * LAT].view(m, LAT))
                keep[name] = res

            return body

        times = {}
        graphs = {}
        for name in ("stock", "flat", "push"):
            graphs[name] = capture(arm(name), ca.capture if name == "stock" else None)
            times[name] = across(graph_us(graphs[name], args.reps, nl), dev)
        # check: the last replay of the push graph, layer by layer
        torch.cuda.synchronize()
        bads = torch.zeros(5, dtype=torch.int64, device=dev)
        for i, (latent, shared, sl, ss, sn) in enumerate(keep["push"]):
            fl, fs, fn = flat.reduce(i, latent.clone(), shared.clone(), w)
            al, as_ = rank_order(gathered(latent)), rank_order(gathered(shared))
            torch.cuda.synchronize()
            bads += torch.tensor([bad(ss, as_), bad(sl, al), bad(ss, fs), bad(sl, fl), bad(sn, fn)],
                                 dtype=torch.int64, device=dev)  # fmt: skip
        dist.all_reduce(bads)
        b = dict(zip(("shared_vs_rankorder", "latent_vs_rankorder", "shared_vs_flat", "latent_vs_flat",
                      "norm_vs_flat"), bads.tolist()))  # fmt: skip
        passed = all(v == 0 for v in b.values())
        ok &= passed
        out["check"].append({"m": m, **b, "pass": passed})
        t = {k: {"median": round(v[0], 2), "slowest": round(v[1], 2)} for k, v in times.items()}
        t["push_minus_flat"] = {k: round(times["push"][j] - times["flat"][j], 2)
                                for j, k in ((0, "median"), (1, "slowest"))}  # fmt: skip
        t["push_minus_stock"] = {k: round(times["push"][j] - times["stock"][j], 2)
                                 for j, k in ((0, "median"), (1, "slowest"))}  # fmt: skip
        out["time"].append({"m": m, **t})
        if rank == 0:
            print("TP8CHECK", json.dumps({"m": m, **b, "pass": passed}), flush=True)
            print("TP8TIME", json.dumps({"m": m, **t}), flush=True)
        del graphs, keep, case
        torch.cuda.empty_cache()
    if rank == 0 and args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    if rank == 0:
        print("SHARED_PUSH_TP8", "PASS" if ok else "FAIL", flush=True)
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
