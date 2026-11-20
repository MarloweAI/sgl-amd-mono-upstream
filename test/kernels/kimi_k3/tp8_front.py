# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 DCP front at TP8 + DCP8 on one node, one process a rank
(``torchrun --nproc-per-node 8``): x to the gated core a step -- the front
launch (its q all-gathered in the launch where the tree's front takes
``peers``), then RCCL's all-to-all of the send records, SGLang's LSE combine
and output gate -- timed per rank over CUDA-graph replays, at 1 / 4 / 8
requests and a mixed batch. The slowest rank's time is the step's. Then each
rank's stamps of one timeline launch: its q chain (x seen to the attention's
q seen, the latest CTA).

Seeded random weights of the served shapes (every rank its own 12 heads'
q_b / W_UK shards where the front takes them); every rank its own DCP share
of every request's keys. Results as JSON (``--out``), from rank 0.
"""

import argparse
import dataclasses
import faulthandler
import inspect
import json
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_mla_front import (
    BF,
    DCP,
    NHL,
    STAMP_NAMES,
    VD,
    Batch,
    batch_stream,
    front_key,
    layer_weights,
    make_keys,
    new_epoch,
    rank_weights,
    scratch_for,
)

CELLS = {
    **{f"{n}x{p}": [p] * n for n in (1, 4, 8) for p in (10000, 100000, 250000)},
    "mixed8": [10000, 250000, 3000, 100000, 37000, 180000, 64, 100000],
}
# the timeline's points: x seen, the attention's q seen
TL_X, TL_Q = 1, 6


def graph_time(fn, group, sets, reps, warmup):
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    # thread-local: the NCCL watchdog's event queries run beside the capture
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
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


def phase(rank, what):
    """A progress line, and every thread's stack on this rank if the phase is still running in 5 minutes."""
    print(f"PHASE {time.strftime('%H:%M:%S')} rank {rank} {what}", file=sys.stderr, flush=True)
    faulthandler.dump_traceback_later(300, exit=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--sets", type=int, default=9)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tl-dir", help="each rank's stamped launch, every CTA's points (us from its first): <cell>-r<rank>.pt")
    args = ap.parse_args()
    args.copy, args.layer = None, 3
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    cpu = dist.new_group(backend="gloo")
    from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
    from sglang.kernels.ops.elementwise import mla_output_gate

    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

    gathered = "peers" in inspect.signature(mf.mla_front_launch).parameters
    w = layer_weights(args, device)
    wr = rank_weights(w, rank) if gathered else w
    epoch = new_epoch(device)
    extra = {}
    if gathered:
        peers = PeerBuffer(mf.peer_end(), cpu, rank, DCP, device)
        peers.bytes.zero_()
        torch.cuda.synchronize()
        dist.barrier(cpu)
        extra = {"peers": peers.addresses}
    need = max(max(CELLS[n]) for n in args.cells) + 8 * 997
    bank = make_keys(w, need, args.seed, device, 1.0)
    result = {"gathered": gathered, "cells": {}}
    for name in args.cells:
        ctxs = CELLS[name]
        b = Batch([(p, bank[997 * i : 997 * i + p], args.seed + 13 * i) for i, p in enumerate(ctxs)], rank, device)
        st = batch_stream(range(len(ctxs)), args.seed, device)
        s = b.s
        key = front_key(w, s)
        # consecutive fronts (MLA layers) take alternate QG copies, where the tree has them
        keys = [dataclasses.replace(key, qg_par=q) for q in (0, 1)] if hasattr(key, "qg_par") else [key]
        scratch = scratch_for(key, device)
        gate = torch.zeros(s, NHL * VD, dtype=BF, device=device)
        send = torch.zeros(DCP, s, NHL, mf.REC, dtype=BF, device=device)
        recv = torch.zeros_like(send)
        core = torch.zeros(s, NHL * VD, dtype=BF, device=device)
        common = dict(
            prefix=st["prefix"], blocks=st["blocks"], w=wr, kv=b.pool, kv_idx=b.kv_idx,
            kv_ptr=b.kv_ptr, g_kv_ptr=b.g_kv_ptr, loc=b.loc, gate=gate, send=send,
            scratch=scratch, rank=rank, layer=args.layer, epoch=epoch, **extra,
        )  # fmt: skip

        nth = [0]

        def front(keys=keys, common=common, nth=nth):
            mf.mla_front_launch(keys[nth[0] % len(keys)], **common)
            nth[0] += 1

        def step(front=front, send=send, recv=recv, gate=gate, core=core):
            epoch[0].add_(1)
            front()
            dist.all_to_all_single(recv.view(-1).view(torch.uint8), send.view(-1).view(torch.uint8))
            comb, _ = dcp_lse_combine_triton(recv[..., :VD], recv.view(torch.float32)[..., VD // 2], is_lse_base_on_e=True)
            core.copy_(mla_output_gate.kimi_k3_mla_output_gate(comb.flatten(1, 2), gate))

        def front_only(front=front):
            epoch[0].add_(1)
            front()

        dist.barrier(cpu)
        phase(rank, f"{name} step")
        t = graph_time(step, cpu, args.sets, args.reps, args.warmup)
        phase(rank, f"{name} front")
        t_front = graph_time(front_only, cpu, args.sets, args.reps, args.warmup)
        tkey = front_key(w, s, timeline=True)
        tl = torch.zeros(256, mf.TL_POINTS, dtype=torch.int64, device=device)
        align = torch.zeros(1, device=device)
        phase(rank, f"{name} stamps")
        for _ in range(3):
            tl.zero_()
            dist.barrier(cpu)
            # the ranks start the stamped front together, as after the step's a2a: the GPU idles
            # while the host enqueues the all-reduce and the launch
            torch.cuda._sleep(4_000_000)
            dist.all_reduce(align)
            epoch[0].add_(1)
            mf.mla_front_launch(tkey, tl=tl, **{**common, "scratch": scratch_for(tkey, device)})
            torch.cuda.synchronize()
        t0 = tl[:, 0][tl[:, 0] > 0].min()

        if args.tl_dir:
            os.makedirs(args.tl_dir, exist_ok=True)
            us = torch.where(tl > 0, (tl - t0).double() / 100.0, torch.full_like(tl, float("nan"), dtype=torch.double))
            torch.save(us.cpu(), os.path.join(args.tl_dir, f"{name}-r{rank}.pt"))

        def lat(k, tl=tl, t0=t0):  # a point no CTA stamped: nan
            v = tl[:, k][tl[:, k] > 0]
            return ((v.max() - t0).double() / 100.0).item() if v.numel() else float("nan")

        cell = {"us": t, "front_us": t_front, "x_seen_us": lat(TL_X), "q_seen_us": lat(TL_Q),
                "stamps_us": [lat(k) for k in range(len(STAMP_NAMES))]}
        allc = [None] * DCP
        phase(rank, f"{name} gather")
        dist.all_gather_object(allc, cell, group=cpu)
        if rank == 0:
            result["cells"][name] = allc
            slow = max(c["us"] for c in allc)
            print(f"CELL {name}: step slowest {slow:.1f} us (ranks {[round(c['us'], 1) for c in allc]}); "
                  f"front slowest {max(c['front_us'] for c in allc):.1f}; q chain (x seen -> q seen, latest CTA) "
                  f"{[round(c['q_seen_us'] - c['x_seen_us'], 1) for c in allc]}", flush=True)
            mid = [sorted(c["stamps_us"][k] for c in allc)[DCP // 2] for k in range(len(STAMP_NAMES))]
            print(f"  STAMPS {name} (latest CTA from the first, the ranks' median, us): "
                  + ", ".join(f"{n} {v:.1f}" for n, v in zip(STAMP_NAMES, mid)), flush=True)
        del b, scratch
    faulthandler.cancel_dump_traceback_later()
    if rank == 0:
        with open(args.out, "w") as fh:
            json.dump(result, fh)
    dist.barrier(cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
