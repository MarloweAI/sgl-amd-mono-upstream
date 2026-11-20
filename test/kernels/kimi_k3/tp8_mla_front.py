# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 MLA front at TP8 + DCP8 on one node, one process a rank
(``torchrun --nproc-per-node 8``): the cross-rank hop the launch pays.

* ``hop``: the 8-B system-scope cell hop between rank 0 and each peer, as the
  combine polls it (a store to the peer, a poll of one's own word, ``s_sleep
  1`` between polls): a ping-pong, its one-way latency (half the round trip)
  and the peers' clock offsets from rank 0;
* ``front``: one request's MLA attention, x to the gated core, timed per rank
  over CUDA-graph replays: this tree's front (the all-to-all, LSE combine and
  output gate inside the launch), or with ``--m2`` a tree whose front ends at
  the send records, then RCCL's all-to-all, SGLang's LSE combine and output
  gate; then, for this tree's front, its stamps on every rank on rank 0's clock:
  each rank's last record pushed, its combine's records seen and core stored.

Seeded random weights of the served shapes; every rank its own DCP share of
the keys. Results as JSON (``--out``), from rank 0.
"""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_mla_front import (  # noqa: E402
    BF,
    DCP,
    G,
    NHL,
    NVB,
    VD,
    Rank,
    layer_weights,
    make_keys,
    stream_inputs,
)

HOP_ITERS = 2000
HOP_WARM = 200


def _hop_kernel():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr.typing import Int32, Int64, T

    from sgl_amd_mono.common.ops import CM_SYS, clock, load_ptr64, rsrc
    from sgl_amd_mono.common.sync import _spin

    def wait_word(base, i, want):
        def load_all():
            return fx.Vector.from_elements(
                [fx.Int32(bo.buffer_load(rsrc(base), fx.Int32(i), vec_width=1, dtype=T.i32, cache_modifier=CM_SYS))],
                fx.Int32,
            )

        _spin(load_all, lambda v: v[0] != want)

    def put_word(base, i, v):
        bo.buffer_store(fx.Int32(v), rsrc(base), fx.Int32(i), cache_modifier=CM_SYS)

    @flyc.jit
    def ping(own, dst, out, i, v, first):
        if first:
            t0 = clock()
            put_word(dst, 0, v)
            wait_word(own, 1, v)
            t1 = clock()
            bo.buffer_store(fx.Vector.from_elements([t0, t1], fx.Int64).bitcast(fx.Int32), rsrc(out), i * 4)
        else:
            wait_word(own, 0, v)
            t = clock()
            put_word(dst, 1, v)
            bo.buffer_store(fx.Vector.from_elements([t, t], fx.Int64).bitcast(fx.Int32), rsrc(out), i * 4)

    @flyc.kernel(name="k3_hop_pingpong", known_block_size=[64, 1, 1])
    def hop(peers: Int64, rank: Int32, partner: Int32, iters: Int32, base: Int32, out: Int64):
        own = load_ptr64(peers, rank)
        first = rank == 0
        dst = load_ptr64(peers, first.select(partner, fx.Int32(0)))
        if fx.thread_idx.x == 0:
            for i in range(iters):
                ping(own, dst, out, i, base + i + 1, first)

    @flyc.jit
    def launch(peers: Int64, rank: Int32, partner: Int32, iters: Int32, base: Int32, out: Int64,
               stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        hop(peers, rank, partner, iters, base, out).launch(grid=(1,), block=(64,), stream=stream)

    return launch


def run_hop(rank, device, group):
    """Rank 0 against each peer in turn: one-way latency (us) and the peer's
    clock minus rank 0's (ticks of 10 ns)."""
    from sgl_amd_mono.common.peer import PeerBuffer

    buf = PeerBuffer(4096, group, rank, DCP, device)
    launch = _hop_kernel()
    out = torch.zeros(HOP_ITERS, 2, dtype=torch.int64, device=device)
    res = {}
    base = 0
    for partner in range(1, DCP):
        dist.barrier(group)
        if rank in (0, partner):
            launch(buf.addresses.data_ptr(), rank, partner, HOP_ITERS, base, out.data_ptr(),
                   stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        base += HOP_ITERS
        mine = out.clone().cpu()
        allv = [None] * DCP
        dist.all_gather_object(allv, mine, group=group)
        if rank == 0:
            t0, t1 = allv[0][HOP_WARM:, 0].double(), allv[0][HOP_WARM:, 1].double()
            tp = allv[partner][HOP_WARM:, 0].double()
            rtt = (t1 - t0) / 100.0
            off = tp - (t0 + t1) / 2
            res[partner] = {
                "one_way_us_median": rtt.median().item() / 2,
                "one_way_us_p10": rtt.quantile(0.1).item() / 2,
                "one_way_us_p90": rtt.quantile(0.9).item() / 2,
                "clock_offset_ticks": off.median().item(),
                "clock_offset_spread_ticks": (off.quantile(0.9) - off.quantile(0.1)).item(),
            }
    buf.close()
    return res


def graph_time(fn, group, sets, reps, warmup):
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    dist.barrier(group)
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(sets):
        dist.barrier(group)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1000.0 / reps)
    out.sort()
    return out[len(out) // 2]


def set_deadline(epoch, seconds=20.0):
    from sgl_amd_mono.common import guard

    ticks = guard.deadline_ticks(seconds)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[10000])
    ap.add_argument("--m2", action="store_true", help="the tree's front ends at the send records")
    ap.add_argument("--hop", action="store_true")
    ap.add_argument("--sets", type=int, default=9)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.copy, args.layer, args.temp = None, 3, 1.0
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2_layer
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

    result = {"m2": args.m2, "cells": {}}
    if args.hop:
        result["hop"] = run_hop(rank, device, cpu)
        if rank == 0:
            print("HOP", json.dumps(result["hop"]), flush=True)
    w = layer_weights(args, device)
    st = stream_inputs(args.seed, device)
    key = mf.MlaFrontBuild(
        tokens=G, nblocks=NVB, scale=w["scale"], eps=w["eps"], out_eps=w["out_eps"],
        q_eps=w["q_eps"], kv_eps=w["kv_eps"],
    )  # fmt: skip
    scratch = PeerBuffer(mf.scratch_bytes(key), None, 0, 1, device).bytes
    epoch = torch.zeros(8, dtype=torch.int32, device=device)
    set_deadline(epoch)
    if args.m2:
        peers = None
    else:
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_combine as mc

        peers = PeerBuffer(mc.peer_end(), cpu, rank, DCP, device)
        k2_layer.init_peer(peers.bytes)
        mc.init_peer(peers.bytes)
        torch.cuda.synchronize()
        dist.barrier(cpu)
    core = torch.zeros(G, NHL * VD, dtype=BF, device=device)
    for p in args.ctx:
        keys = make_keys(w, p, args.seed + p, device, args.temp)
        rk = Rank(keys, p, rank, device, args.seed + p)
        del keys
        common = dict(
            prefix=st["prefix"], blocks=st["blocks"], w=w, kv=rk.pool, kv_idx=rk.kv_idx,
            kv_ptr=rk.kv_ptr, g_kv_ptr=rk.g_kv_ptr, loc=rk.loc, scratch=scratch, rank=rank,
            layer=args.layer, epoch=epoch,
        )  # fmt: skip
        if args.m2:
            from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
            from sglang.kernels.ops.elementwise import mla_output_gate

            gate = torch.zeros(G, NHL * VD, dtype=BF, device=device)
            send = torch.zeros(DCP, G, NHL, mf.REC, dtype=BF, device=device)
            recv = torch.zeros_like(send)

            def step():
                epoch[0].add_(1)
                mf.mla_front_launch(key, gate=gate, send=send, **common)
                dist.all_to_all_single(recv.view(-1).view(torch.uint8), send.view(-1).view(torch.uint8))
                comb, _ = dcp_lse_combine_triton(recv[..., :VD], recv.view(torch.float32)[..., VD // 2], is_lse_base_on_e=True)
                core.copy_(mla_output_gate.kimi_k3_mla_output_gate(comb.flatten(1, 2), gate))

            def front_only():
                epoch[0].add_(1)
                mf.mla_front_launch(key, gate=gate, send=send, **common)
        else:

            def step():
                epoch[0].add_(1)
                mf.mla_front_launch(key, core=core, peers=peers.addresses, a2a_par=0, **common)

            front_only = None
        t = graph_time(step, cpu, args.sets, args.reps, args.warmup)
        t_front = graph_time(front_only, cpu, args.sets, args.reps, args.warmup) if front_only else None
        cell = {"us": t, "front_only_us": t_front}
        if not args.m2:
            tkey = mf.MlaFrontBuild(**{**{f: getattr(key, f) for f in key.__dataclass_fields__}, "timeline": True})
            tls = []
            for _ in range(5):
                tl = torch.zeros(256, mf.TL_POINTS, dtype=torch.int64, device=device)
                dist.barrier(cpu)
                epoch[0].add_(1)
                mf.mla_front_launch(tkey, core=core, peers=peers.addresses, a2a_par=0, tl=tl, **common)
                torch.cuda.synchronize()
                tls.append(tl.cpu())
            cell["tl"] = [x.tolist() for x in tls]
        allc = [None] * DCP
        dist.all_gather_object(allc, cell, group=cpu)
        if rank == 0:
            result["cells"][p] = allc
            print(f"CELL P={p}: us per rank {[round(c['us'], 2) for c in allc]}"
                  + (f"; front only {[round(c['front_only_us'], 2) for c in allc]}" if args.m2 else ""), flush=True)
    if rank == 0:
        with open(args.out, "w") as fh:
            json.dump(result, fh)
    dist.barrier(cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
