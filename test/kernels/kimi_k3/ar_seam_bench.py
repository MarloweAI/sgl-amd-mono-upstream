# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's fused o_proj all-reduce + FFN read (``kimi_k3/gfx950/ar_seam``) at TP8: bitwise against SGLang's
pair (aiter's custom all-reduce, then ``attn_res_hip``) on the real ranks, and µs a call of both.

Check: every rank's ``out``, ``pre`` and bank snapshot against the stock pair run in the same graph on the same
inputs, and against ``attn_res_hip`` on the rank-order bf16 sum. Three input sets rotate between replays.
Time: CUDA graphs of 20 back-to-back calls (the fused launch; aiter's all-reduce + ``attn_res_hip``; each part),
median and slowest rank.

    torchrun --nproc-per-node 8 ar_seam_bench.py --rows 54,72,96,120,168 --time
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

TP = 8
H = 7168
NB = 9
SETS = 3
BF = torch.bfloat16
EPS_S, EPS_O = 1e-6, 1e-5


def stage(rank, what):
    print(f"STAGE r{rank} {time.strftime('%H:%M:%S')} {what}", flush=True)


def make_inputs(m, s, dev):
    g = torch.Generator(device=dev).manual_seed(1234 + 17 * m + s)

    def rn(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=dev) * scale

    return {"x": rn(TP, m, H, scale=0.3).to(BF), "res": rn(m, H, scale=2.0).to(BF), "bank": rn(m, NB, H, scale=2.0).to(BF)}


def make_weights(dev):
    g = torch.Generator(device=dev).manual_seed(99)
    cw = (torch.randn(H, generator=g, device=dev) * 0.05).float().contiguous()
    ow = (1.0 + 0.1 * torch.randn(H, generator=g, device=dev)).to(BF).contiguous()
    return cw, ow


def stock_read(ar, res, bank, nvb, cw, ow, write):
    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip

    out = torch.empty_like(ar)
    if res is None:
        attn_res_hip(ar, bank, cw, ow, out, nvb, EPS_S, EPS_O, write_prefix=write)
        return out, ar
    pre = torch.empty_like(ar)
    attn_res_hip(res, bank, cw, ow, out, nvb, EPS_S, EPS_O, addend=ar, prefix_out=pre, write_prefix=write)
    return out, pre


def graphed(ca, body):
    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with ca.capture():
        with torch.cuda.graph(g):
            body()
    torch.cuda.synchronize()
    return g


def bad(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def check(ca, seam, rank, dev, m, nvb, has_res, write, cw, ow):
    """Mismatches of the fused launch against the stock pair in one graph, and against the rank-order sum."""
    sets = [make_inputs(m, s, dev) for s in range(SETS)]
    cur = {k: torch.empty_like(v[rank] if k == "x" else v) for k, v in sets[0].items()}
    bank_f = torch.empty_like(cur["bank"])
    bank_s = torch.empty_like(cur["bank"])
    xf = torch.empty(m, H, dtype=BF, device=dev)
    xs = torch.empty_like(xf)
    res = cur["res"] if has_res else None
    got = {}

    def body():
        xf.copy_(cur["x"])
        xs.copy_(cur["x"])
        bank_f.copy_(cur["bank"])
        bank_s.copy_(cur["bank"])
        got["f"] = seam.read(xf, res, bank_f, nvb, cw, ow, EPS_S, EPS_O, write)
        ar = ca.custom_all_reduce(xs)
        got["s"] = stock_read(ar, res, bank_s, nvb, cw, ow, write)

    g = graphed(ca, body)
    n = {"vs_stock": 0, "vs_order": 0, "bank_vs_stock": 0}
    for rep in range(2 * SETS):
        st = sets[rep % SETS]
        cur["x"].copy_(st["x"][rank])
        cur["res"].copy_(st["res"])
        cur["bank"].copy_(st["bank"])
        g.replay()
        torch.cuda.synchronize()
        (fo, fp), (so, sp) = got["f"], got["s"]
        n["vs_stock"] += bad(fo, so) + bad(fp, sp)
        n["bank_vs_stock"] += bad(bank_f, bank_s)
        acc = st["x"][0].float()
        for r in range(1, TP):
            acc = acc + st["x"][r].float()
        bank_o = st["bank"].clone()
        oo, op = stock_read(acc.to(BF), st["res"] if has_res else None, bank_o, nvb, cw, ow, write)
        n["vs_order"] += bad(fo, oo) + bad(fp, op) + bad(bank_f, bank_o)
    return n


def graph_us(ca, body, reps, per):
    g = torch.cuda.CUDAGraph()
    with ca.capture():
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


def across(v, dev):
    t = torch.tensor([v], dtype=torch.float64, device=dev)
    every = [torch.empty_like(t) for _ in range(TP)]
    dist.all_gather(every, t)
    vals = sorted(float(e.item()) for e in every)
    return (vals[3] + vals[4]) / 2, vals[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="54,72,96,120,168", type=lambda s: [int(v) for v in s.split(",")])
    p.add_argument("--nvb", default="2,4,8", type=lambda s: [int(v) for v in s.split(",")])
    p.add_argument("--time", action="store_true")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--json")
    a = p.parse_args()

    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from sgl_amd_mono.kimi_k3.gfx950 import ar_seam as S

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    stage(rank, "init")
    dist.init_process_group("nccl", device_id=dev)
    cpu = dist.new_group(backend="gloo")
    dist.barrier()
    stage(rank, "process groups up")
    ca = CustomAllreduce(cpu, dev)
    assert ca.lamport_enabled, "aiter's lamport path is off"
    seam = S.ArSeam(ca, dev)
    S.ops()
    dist.barrier()
    stage(rank, "comm and build up")
    cw, ow = make_weights(dev)
    out = {"rank": rank, "check": [], "time": []}
    fails = 0
    for m in a.rows:
        for nvb, has_res, write in ((1, False, False), (4, True, False), (8, True, False), (2, True, True)):
            stage(rank, f"check m={m} nvb={nvb} res={int(has_res)} write={int(write)}")
            n = check(ca, seam, rank, dev, m, nvb, has_res, write, cw, ow)
            t = torch.tensor([n["vs_stock"], n["vs_order"], n["bank_vs_stock"]], dtype=torch.int64, device=dev)
            dist.all_reduce(t)
            fails += int(t.sum() != 0)
            out["check"].append({"m": m, "nvb": nvb, "res": has_res, "write": write, "vs_stock": int(t[0]),
                                 "vs_order": int(t[1]), "bank_vs_stock": int(t[2])})
            if rank == 0:
                print(f"CHECK m={m:3d} nvb={nvb} res={int(has_res)} write={int(write)} plan={S.plan(m)} "
                      f"vs_stock={int(t[0])} vs_order={int(t[1])} bank_vs_stock={int(t[2])}", flush=True)

    if a.time:
        from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip

        nl = 20

        def record(name, m, nvb, us):
            med, mx = across(us, dev)
            out["time"].append({"name": name, "m": m, "nvb": nvb, "us_rank": us, "us_med": med, "us_max": mx})
            if rank == 0:
                print(f"TIME {name:14s} m={m:3d} nvb={nvb} med {med:7.2f} us  slowest {mx:7.2f} us", flush=True)

        for m in a.rows:
            st = make_inputs(m, 0, dev)
            ins = [st["x"][rank].clone() for _ in range(nl)]
            pres = [torch.empty(m, H, dtype=BF, device=dev) for _ in range(nl)]
            outs = [torch.empty_like(pres[0]) for _ in range(nl)]
            for nvb in a.nvb:
                stage(rank, f"time m={m} nvb={nvb}")

                def fused():
                    for i in range(nl):
                        seam.read(ins[i], st["res"], st["bank"], nvb, cw, ow, EPS_S, EPS_O)

                def stock(with_ar=True, with_read=True):
                    def body():
                        for i in range(nl):
                            x = ca.custom_all_reduce(ins[i]) if with_ar else ins[i]
                            if with_read:
                                attn_res_hip(st["res"], st["bank"], cw, ow, outs[i], nvb, EPS_S, EPS_O, addend=x,
                                             prefix_out=pres[i])
                    return body

                for name, body in (("fused", fused), ("stock_ar+read", stock()),
                                   ("stock_ar", stock(with_read=False)), ("stock_read", stock(with_ar=False))):
                    body()
                    torch.cuda.synchronize()
                    record(name, m, nvb, graph_us(ca, body, a.reps, nl))
    every = [None] * TP
    dist.all_gather_object(every, out, group=cpu)
    if rank == 0 and a.json:
        with open(a.json, "w") as f:
            json.dump(every, f, indent=1)
    if rank == 0:
        print("TP8", "PASS" if fails == 0 else "FAIL", flush=True)
    dist.destroy_process_group()
    raise SystemExit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
