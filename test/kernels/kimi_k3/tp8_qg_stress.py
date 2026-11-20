# SPDX-License-Identifier: Apache-2.0
"""The DCP front's q all-gather under back-to-back fronts on 8 ranks
(``torchrun --nproc-per-node 8``): no all-to-all or all-reduce between them,
and a random GPU-side delay on every rank before every front.

A step runs ``--fronts`` fronts as consecutive MLA layers (their own mailbox
tags; alternate QG copies where the tree's build has them, else its one copy).
Every front's send records and gate are summed on the GPU (integer sums of
their bits) and checked against a front run alone at the end.

A rank whose GPU finishes no front for ``--hang-s`` seconds reads its own QG
flags (both copies) on a side stream while the launch spins, prints each
(source, request) pair's tag against the one the hung front expects, and
exits.
"""

import argparse
import dataclasses
import inspect
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_mla_front import (
    BF,
    DCP,
    NHL,
    VD,
    Batch,
    batch_stream,
    front_key,
    layer_weights,
    make_keys,
    rank_weights,
    scratch_for,
)
from tp8_front import CELLS

TAG_SLOTS = 256


def tag_of(e, layer):
    """The mailbox tag of a front at epoch ``e`` and layer slot ``layer``."""
    return (e % (1 << 22)) * TAG_SLOTS + layer


def new_epoch(device, seconds=30.0):
    """A launch's epoch buffer with every wait's deadline (here, not from
    check_mla_front, so builds before the guarded waits run this test too)."""
    from sgl_amd_mono.common import guard

    epoch = torch.zeros(8, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(seconds)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return epoch


def bits_sum(t):
    return t.view(torch.int32).to(torch.int64).sum()


def dump_flags(mf, peers, rank, nreq, hung, keys, layers):
    """This rank's QG flag pairs, both copies, read while a front spins."""
    side = torch.cuda.Stream()
    out = torch.empty(2, DCP, mf.MAX_REQS, 2, dtype=torch.int32, pin_memory=True)
    with torch.cuda.stream(side):
        for q in range(2):
            o = mf.qg_offset() + q * mf.QG_COPY + mf.QG_DATA
            f = peers.bytes[o : o + DCP * mf.MAX_REQS * 8].view(torch.int32).view(DCP, mf.MAX_REQS, 2)
            out[q].copy_(f, non_blocking=True)
    ev = side.record_event()
    t0 = time.time()
    while not ev.query() and time.time() - t0 < 20:
        time.sleep(0.01)
    step, j, e = hung
    par = getattr(keys[j % len(keys)], "qg_par", 0)
    want = tag_of(e, layers[j])
    lines = [f"HANG rank {rank}: step {step} front {j} (layer {layers[j]}, copy {par}) expects tag {want}"]
    if not ev.query():
        lines.append(f"  rank {rank}: the flags could not be read (side copy did not finish)")
    else:
        for q in range(2):
            for src in range(DCP):
                row = []
                for rq in range(nreq):
                    v, tg = int(out[q, src, rq, 0]), int(out[q, src, rq, 1])
                    mark = "" if tg == want else f" (epoch {tg // TAG_SLOTS} layer {tg % TAG_SLOTS})"
                    row.append(f"{v}/{tg}{mark}")
                lines.append(f"  rank {rank} copy {q} src {src}: " + ", ".join(row))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["8x100000", "mixed8"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--fronts", type=int, default=4, help="fronts a step (consecutive MLA layers)")
    ap.add_argument("--skew-us", type=float, default=150.0, help="the largest random delay before a front")
    ap.add_argument("--hang-s", type=float, default=60.0)
    ap.add_argument("--hog", type=float, default=0.0, help="the chance a front runs beside a GEMM on a side stream")
    ap.add_argument("--graph", action="store_true", help="replay the step's fronts as one CUDA graph (no sums)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.copy, args.layer = None, 3
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    cpu = dist.new_group(backend="gloo")

    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

    assert "peers" in inspect.signature(mf.mla_front_launch).parameters, "a tree without the q all-gather"
    w = layer_weights(args, device)
    wr = rank_weights(w, rank)
    epoch = new_epoch(device)
    peers = PeerBuffer(mf.peer_end(), cpu, rank, DCP, device)
    peers.bytes.zero_()
    torch.cuda.synchronize()
    dist.barrier(cpu)
    rng = np.random.default_rng(args.seed * 1000 + rank)
    # torch.cuda._sleep counts shader clocks: calibrate a microsecond's worth
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    torch.cuda._sleep(1_000_000)
    b.record()
    b.synchronize()
    cyc_us = 1_000_000 / (a.elapsed_time(b) * 1000.0)
    layers = [3 + 4 * j for j in range(args.fronts)]
    need = max(max(CELLS[n]) for n in args.cells) + 8 * 997
    bank = make_keys(w, need, args.seed, device, 1.0)
    fail = 0
    for name in args.cells:
        ctxs = CELLS[name]
        bt = Batch([(p, bank[997 * i : 997 * i + p], args.seed + 13 * i) for i, p in enumerate(ctxs)], rank, device)
        st = batch_stream(range(len(ctxs)), args.seed, device)
        key = front_key(w, bt.s)
        keys = [dataclasses.replace(key, qg_par=q) for q in (0, 1)] if hasattr(key, "qg_par") else [key]
        gate = torch.zeros(bt.s, NHL * VD, dtype=BF, device=device)
        send = torch.zeros(DCP, bt.s, NHL, mf.REC, dtype=BF, device=device)
        common = dict(  # noqa: C408
            prefix=st["prefix"], blocks=st["blocks"], w=wr, kv=bt.pool, kv_idx=bt.kv_idx,
            kv_ptr=bt.kv_ptr, g_kv_ptr=bt.g_kv_ptr, loc=bt.loc, gate=gate, send=send,
            scratch=scratch_for(key, device), rank=rank, epoch=epoch, peers=peers.addresses,
        )  # fmt: skip
        # the reference: one front, every rank started together
        for k in keys:
            dist.barrier(cpu)
            epoch[0].add_(1)
            mf.mla_front_launch(k, layer=layers[0], **common)
            torch.cuda.synchronize()
        ref = (bits_sum(send).item(), bits_sum(gate).item())
        e = int(epoch[0].item())
        n = args.steps * args.fronts
        sums = torch.zeros(n, 2, dtype=torch.int64, device=device)
        side = torch.cuda.Stream()
        hog = torch.randn(8192, 8192, dtype=BF, device=device)
        graph = None
        if args.graph:
            # a step's fronts back to back, as the front-only timing replays them
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                epoch[0].add_(1)
                for j in range(args.fronts):
                    mf.mla_front_launch(keys[j % len(keys)], layer=layers[j], **common)
            sums.fill_(ref[0])[:, 1] = ref[1]
        evs, hung = [], None
        dist.barrier(cpu)
        t0 = time.time()
        for step in range(args.steps):
            e += 1
            if graph is not None:
                time.sleep(rng.uniform(0, args.skew_us) * 1e-6)
                graph.replay()
                evs.append((torch.cuda.current_stream().record_event(), (step, args.fronts - 1, e)))
            else:
                epoch[0].add_(1)
            for j in range(args.fronts if graph is None else 0):
                torch.cuda._sleep(int(rng.uniform(0, args.skew_us) * cyc_us))
                if rng.uniform() < args.hog:
                    # a GEMM beside the front takes CUs from it on this rank only
                    m = int(rng.choice([2048, 4096, 8192]))
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        torch.mm(hog[:m], hog[:, :m])
                mf.mla_front_launch(keys[j % len(keys)], layer=layers[j], **common)
                i = step * args.fronts + j
                sums[i, 0] = bits_sum(send)
                sums[i, 1] = bits_sum(gate)
                evs.append((torch.cuda.current_stream().record_event(), (step, j, e)))
            # keep the host at most 8 fronts ahead; a front that never finishes is a hang
            while len(evs) > 8 and hung is None:
                ev, at = evs[0]
                t1 = time.time()
                while not ev.query():
                    if time.time() - t1 > args.hang_s:
                        hung = at
                        break
                    time.sleep(0.0005)
                if hung is None:
                    evs.pop(0)
            if hung is not None:
                break
        if hung is not None:
            dump_flags(mf, peers, rank, bt.s // mf.G_ROWS, hung, keys, layers)
            # the other ranks' watchdogs fire meanwhile, then torchrun stops them all
            time.sleep(30)
            os._exit(3)
        torch.cuda.synchronize()
        got = sums.cpu()
        bad = int(((got[:, 0] != ref[0]) | (got[:, 1] != ref[1])).sum())
        res = [None] * DCP
        dist.all_gather_object(res, (bad, time.time() - t0), group=cpu)
        if rank == 0:
            print(
                f"STRESS {name}: {n} fronts a rank ({args.fronts} a step, {len(keys)} QG copies, "
                f"skew <= {args.skew_us:.0f} us, hog {args.hog}, graph {args.graph}): fronts whose outputs differ from the front alone, by rank "
                f"{[r[0] for r in res]}; {max(r[1] for r in res):.1f} s",
                flush=True,
            )
        fail += sum(r[0] for r in res)
        del bt
    dist.barrier(cpu)
    if rank == 0:
        print("STRESS", "PASS" if fail == 0 else f"FAIL ({fail} fronts)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
