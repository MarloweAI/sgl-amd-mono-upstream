# SPDX-License-Identifier: Apache-2.0
"""The wide MLA tail's exchange at TP8 + DCP8 on one node (``torchrun
--nproc-per-node 8``), or on one GPU with a loopback tree (``--nproc-per-node
1``): ``mla_tail`` with no reduce tile (W_UV, the push to the heads' owners,
the LSE combine and gate) on random reduced rows, graph-timed per rank at
36 / 108 / 180 rows, two launches a replay alternating the A2A copies (as
consecutive MLA layers do). The slowest rank's median is the step's; the
loopback's time taken from it is the exchange's. Results as JSON (``--out``),
from rank 0.
"""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_mla_front import new_epoch  # noqa: E402

BF = torch.bfloat16
NHA, NHL, KV, VD, DCP = 96, 12, 512, 128, 8


def graph_time(fn, sets, reps, warmup, world):
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        for _ in range(reps):
            fn()
    if world > 1:
        dist.barrier()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(sets):
        if world > 1:
            dist.barrier()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1000.0 / reps)
    out.sort()
    return out[len(out) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[36, 108, 180])
    ap.add_argument("--sets", type=int, default=9)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_combine as mc
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_tail as mt

    if world > 1:
        dist.init_process_group("nccl", device_id=device)
        cpu = dist.new_group(backend="gloo")
        peers = PeerBuffer(mt.peer_end(), cpu, rank, DCP, device)
        table = peers.addresses
    else:
        # the loopback tree: every rank's buffer is this one
        peers = PeerBuffer(mt.peer_end(), None, 0, 1, device)
        table = torch.full((DCP,), peers.local, dtype=torch.int64, device=device)
    peers.bytes.zero_()
    mc.init_peer(peers.bytes)
    mt.init_peer(peers.bytes)
    torch.cuda.synchronize()
    if world > 1:
        dist.barrier(cpu)
    epoch = new_epoch(device)
    scratch = PeerBuffer(mt.scratch_bytes(), None, 0, 1, device).bytes
    scratch.zero_()
    g = torch.Generator(device=device).manual_seed(rank + 1)
    w_vc = (torch.randn(NHA, KV, VD, generator=g, device=device) * 0.045).to(BF)
    res = {"world": world, "rows": {}}
    for s in args.rows:
        r = {}
        o = torch.randn(s, NHA, KV, generator=g, device=device).to(BF)
        lse = torch.randn(s, NHA, generator=g, device=device)
        empty = torch.zeros(s, dtype=torch.bool, device=device)
        gate = torch.randn(s, NHL * VD, generator=g, device=device).to(BF)
        core = torch.empty(s, NHL * VD, dtype=BF, device=device)
        f32, i32 = torch.float32, torch.int32
        z = dict(
            logits=torch.zeros(1, NHA, KV, dtype=f32, device=device),
            plse=torch.zeros(1, NHA, dtype=f32, device=device),
            indptr=torch.zeros(2, dtype=i32, device=device),
            fmap=torch.zeros(2, dtype=i32, device=device), pmap=torch.zeros(1, dtype=i32, device=device),
        )  # fmt: skip

        def tail():
            for par in (0, 1):
                epoch[0].add_(1)
                mt.mla_tail_launch(
                    mt.MlaTailBuild(par=par), s=s, o=o, lse=lse, empty=empty, w_vc=w_vc,
                    gate=gate, core=core, scratch=scratch, peers=table, rank=rank, layer=3,
                    epoch=epoch, **z,
                )  # fmt: skip

        for name, fn in (("tail", tail),):
            t = graph_time(fn, args.sets, args.reps, args.warmup, world) / 2
            if world > 1:
                ts = [None] * world
                dist.all_gather_object(ts, t)
            else:
                ts = [t]
            r[name] = {"max": max(ts), "median": sorted(ts)[len(ts) // 2], "ranks": ts}
        res["rows"][s] = r
        if rank == 0:
            print(f"ROWS {s} " + " ".join(f"{k} max {v['max']:.1f} med {v['median']:.1f}" for k, v in r.items()), flush=True)
    if rank == 0:
        json.dump(res, open(args.out, "w"), indent=1)
        print("WROTE", args.out)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
