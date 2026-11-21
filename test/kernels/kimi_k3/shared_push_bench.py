# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's split MoE all-reduce (``kimi_k3/gfx950/stages/shared_push.py``) with the 8 ranks emulated in
one launch on one GPU: check against the flat all-reduce, and time the launches.

Check: a graph of six MoE layers at mixed widths (each the shared push, then the latent all-reduce with
the shared sum and the latent norm), replayed with three input sets rotating through its buffers, so no
layer passes on an earlier replay's data. Every rank's shared and latent sums against the rank-order f32
sum rounded once (bit for bit), and the normed latent against ``ar_norm.build_moe``'s flat launch on the
same partials (bit for bit) and against SGLang's RMSNorm (mismatch count, rel-L2 vs f64).

Time (``--time``): µs a launch of the emulated push, the emulated split launch and the emulated flat
launch at each width, in a graph of ``--layers`` launches. The 8 ranks share the GPU, so these are a
launch's cost on one device, not a TP8 call.

    python shared_push_bench.py --rows 49,54,64,96,120,168,192 --time
"""

import argparse
import json

import torch

TP = 8
LAT, SHR = 3584, 7168
SETS = 3
BF = torch.bfloat16
GRAPH_ROWS = (54, 168, 96, 49, 192, 120)


def rank_order(x):
    acc = x[0].float()
    for r in range(1, TP):
        acc = acc + x[r].float()
    return acc.to(BF)


def stock_rmsnorm(eps, w):
    from sglang.srt.layers.layernorm import RMSNorm

    n = RMSNorm(LAT, eps=eps).to(w.device)
    n.weight.data.copy_(w)
    return n


def rel_l2(a, ref):
    a, ref = a.double(), ref.double()
    return float((a - ref).norm() / ref.norm())


def bad(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


class Emu:
    """The emulated ranks' peer buffers (one a rank, on this GPU), epoch and launches."""

    def __init__(self, eps, launches, dev, deadline):
        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.kimi_k3.gfx950.stages import ar_norm as N
        from sgl_amd_mono.kimi_k3.gfx950.stages import shared_push as S

        self.mr = S.MAX_ROWS
        self.push = S.build_push(launches, blocks=8, emulate=True)
        self.split = S.build_split(eps, launches, blocks=32, emulate=True)
        self.flat = N.build_moe(eps, launches, blocks=32, emulate=True)
        self.bufs = [PeerBuffer(S.peer_bytes(), None, 0, 1, dev, init=S.arm) for _ in range(TP)]
        self.table = torch.tensor([b.local for b in self.bufs], dtype=torch.int64, device=dev)
        self.fbufs = [PeerBuffer(N.moe_peer_bytes(), None, 0, 1, dev, init=N.moe_arm) for _ in range(TP)]
        self.ftable = torch.tensor([b.local for b in self.fbufs], dtype=torch.int64, device=dev)
        self.ep = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
        self.ep[guard.GUARD_DEADLINE] = guard.deadline_ticks(deadline) & 0x7FFFFFFF
        self.fep = self.ep.clone()

    def layer(self, slot, m, io, w):
        s = torch.cuda.current_stream()
        self.push(io["ps"].data_ptr(), m, self.table.data_ptr(), 0, self.ep.data_ptr(), slot, stream=s)
        self.split(io["pl"].data_ptr(), io["ps"].data_ptr(), w.data_ptr(), io["ol"].data_ptr(),
                   io["os"].data_ptr(), io["on"].data_ptr(), m, self.table.data_ptr(), 0, self.ep.data_ptr(),
                   slot, stream=s)  # fmt: skip

    def flat_layer(self, slot, m, io, w):
        self.flat(io["pl"].data_ptr(), io["ps"].data_ptr(), w.data_ptr(), io["fl"].data_ptr(),
                  io["fs"].data_ptr(), io["fn"].data_ptr(), m, self.ftable.data_ptr(), 0, self.fep.data_ptr(),
                  slot, stream=torch.cuda.current_stream())  # fmt: skip


def buffers(mr, dev):
    z = lambda n: torch.zeros(TP, mr, n, dtype=BF, device=dev)  # noqa: E731
    return {"pl": z(LAT), "ps": z(SHR), "ol": z(LAT), "os": z(SHR), "on": z(LAT),
            "fl": z(LAT), "fs": z(SHR), "fn": z(LAT)}  # fmt: skip


def inputs(m, s, li, dev):
    g = torch.Generator(device=dev).manual_seed(4242 + 131 * m + 17 * s + li)
    xl = (torch.randn(TP, m, LAT, generator=g, device=dev) * 0.3).to(BF)
    xs = (torch.randn(TP, m, SHR, generator=g, device=dev) * 0.3).to(BF)
    return xl, xs


def check(args, emu, dev, w):
    """The mixed-width graph, replayed over the input sets; returns (ok, cases)."""
    layers = [(slot, m) for slot, m in enumerate(r for r in GRAPH_ROWS if r in args.rows)]
    ios = [buffers(emu.mr, dev) for _ in layers]

    def body():
        emu.ep[0:1].add_(1)
        for (slot, m), io in zip(layers, ios):
            emu.layer(slot, m, io, w)

    def fill(s):
        for (slot, m), io in zip(layers, ios):
            xl, xs = inputs(m, s, slot, dev)
            io["pl"][:, :m].copy_(xl)
            io["ps"][:, :m].copy_(xs)
            for k in ("ol", "os", "on"):
                io[k].zero_()

    fill(0)
    body()  # eager first
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    ok, cases = True, []
    norm = stock_rmsnorm(args.eps, w)
    for rep in range(2 * SETS):
        s = rep % SETS
        fill(s)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        emu.fep[0:1].add_(1)
        for (slot, m), io in zip(layers, ios):
            emu.flat_layer(slot, m, io, w)
        torch.cuda.synchronize()
        for (slot, m), io in zip(layers, ios):
            xl, xs = inputs(m, s, slot, dev)
            al, as_ = rank_order(xl), rank_order(xs)
            n = norm(al.clone())
            a64 = xl.double().sum(0)
            r64 = a64 / torch.sqrt((a64 * a64).mean(-1, keepdim=True) + args.eps) * w.double()
            c = {
                "rep": rep, "m": m, "slot": slot,
                "shared_bad": bad(io["os"][:, :m], as_.expand(TP, -1, -1)),
                "latent_bad": bad(io["ol"][:, :m], al.expand(TP, -1, -1)),
                "norm_vs_flat_bad": bad(io["on"][:, :m], io["fn"][:, :m]),
                "shared_vs_flat_bad": bad(io["os"][:, :m], io["fs"][:, :m]),
                "norm_vs_stock_bad": bad(io["on"][:, :m], n.expand(TP, -1, -1)),
                "rel_split": max(rel_l2(io["on"][r, :m], r64) for r in range(TP)),
                "rel_stock": rel_l2(n, r64),
            }  # fmt: skip
            c["pass"] = (c["shared_bad"] == 0 and c["latent_bad"] == 0 and c["norm_vs_flat_bad"] == 0
                         and c["shared_vs_flat_bad"] == 0)  # fmt: skip
            ok &= c["pass"]
            cases.append(c)
            print("EMU", json.dumps(c), flush=True)
    print("EMU guard", emu.ep[:8].tolist(), "flat guard", emu.fep[:8].tolist(), flush=True)
    return ok, cases


def graph_us(body, per, reps=20):
    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / (reps * per)


def time_launches(args, emu, dev, w):
    """µs a launch (emulated: all 8 ranks in it) of push, push + split, and the flat launch."""
    out = []
    nl = args.layers
    for m in args.rows:
        io = buffers(emu.mr, dev)
        xl, xs = inputs(m, 0, 0, dev)
        io["pl"][:, :m].copy_(xl)
        io["ps"][:, :m].copy_(xs)
        s = torch.cuda.current_stream

        def push_only():
            emu.ep[0:1].add_(1)
            for slot in range(nl):
                emu.push(io["ps"].data_ptr(), m, emu.table.data_ptr(), 0, emu.ep.data_ptr(), slot, stream=s())

        def both():
            emu.ep[0:1].add_(1)
            for slot in range(nl):
                emu.layer(slot, m, io, w)

        def flat():
            emu.fep[0:1].add_(1)
            for slot in range(nl):
                emu.flat_layer(slot, m, io, w)

        r = {"m": m, "push": graph_us(push_only, nl), "push_split": graph_us(both, nl), "flat": graph_us(flat, nl)}
        r["split"] = r["push_split"] - r["push"]
        print("TIME", json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        out.append(r)
    return out


def push_sweep(args, emu, dev):
    """µs a launch of the emulated push at ``--push-blocks`` CTAs a rank (8 ranks' stores on this GPU)."""
    from sgl_amd_mono.kimi_k3.gfx950.stages import shared_push as S

    nl = args.layers
    for blocks in args.push_blocks:
        launch = S.build_push(nl, blocks=blocks, emulate=True)
        for m in (96, 168):
            ps = torch.zeros(TP, emu.mr, SHR, dtype=BF, device=dev)

            def body():
                emu.ep[0:1].add_(1)
                for slot in range(nl):
                    launch(ps.data_ptr(), m, emu.table.data_ptr(), 0, emu.ep.data_ptr(), slot,
                           stream=torch.cuda.current_stream())  # fmt: skip

            us = graph_us(body, nl)
            gbs = TP * (TP - 1) * m * SHR * 2 / us / 1e3
            print("PUSH", json.dumps({"blocks": blocks, "m": m, "us": round(us, 2), "GBps_total": round(gbs, 1),
                                      "GBps_cta": round(gbs / (TP * blocks), 2)}), flush=True)  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="49,54,64,96,120,168,192")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--deadline", type=float, default=5.0)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--time", action="store_true")
    p.add_argument("--push-blocks", default="")
    p.add_argument("--json", default="")
    args = p.parse_args()
    args.rows = [int(r) for r in args.rows.split(",")]
    args.push_blocks = [int(b) for b in args.push_blocks.split(",") if b]
    dev = torch.device("cuda", 0)
    w = (1 + 0.1 * torch.randn(LAT, generator=torch.Generator(device=dev).manual_seed(5), device=dev)).to(BF)
    emu = Emu(args.eps, max(len(GRAPH_ROWS), args.layers), dev, args.deadline)
    ok, cases = check(args, emu, dev, w)
    times = time_launches(args, emu, dev, w) if args.time else []
    push_sweep(args, emu, dev)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"pass": ok, "cases": cases, "time": times}, f, indent=1)
    print("SHARED_PUSH_CHECK", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
