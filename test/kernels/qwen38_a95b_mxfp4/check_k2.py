"""Check K2 (``layer_post``: o_proj, its TP reduce, the norm, the router and the
experts, the MoE TP reduce) on 8 gfx950 GPUs (torchrun), on random TP8 shards in
SGLang's layouts:

- compile: every build key the spec can make (widths 1 .. 8, fused shared
  expert) compiles on every rank;
- batch invariance: a probe row's out and res_out do not change with the step's
  other rows (other requests, zero rows, copies of the probe) or the width, on
  every rank, and every rank has the same out;
- graph replay: widths 8, 1, 3, 8, 2, 5 in one graph on one peer buffer,
  replayed, bit for bit against eager on every rank;
- fault: rank 3 stalls at its first wait (``stop=False``): rank 3 sets its
  deadline word and its flag reaches every rank.
"""

import argparse
import datetime
import os
import sys
import time

import torch
import torch.distributed as dist
from inputs import BF, CORE, HIDDEN, MAXT, epoch, errors, gen, k2_weights, rnd

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.peer import PeerBuffer
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer as k2
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layout as L
from sgl_amd_mono.qwen38_a95b_mxfp4.spec import _compile_only

REPLAY_WIDTHS = (8, 1, 3, 8, 2, 5)
KINDS = ("rand0", "rand1", "zero", "copy")

rank = int(os.environ["RANK"])
assert int(os.environ["WORLD_SIZE"]) == L.TP, f"needs {L.TP} ranks"
torch.cuda.set_device(rank)
DEV = torch.device("cuda", rank)
dist.init_process_group("gloo", timeout=datetime.timedelta(hours=1))


def log(msg):
    if rank == 0:
        print(msg, flush=True)


def all_ok(flag: bool) -> bool:
    t = torch.tensor([0 if flag else 1])
    dist.all_reduce(t)
    return int(t) == 0


def same_on_every_rank(t: torch.Tensor) -> bool:
    mine = t.cpu()
    ref = mine.clone()
    dist.broadcast(ref, 0)
    return all_ok(torch.equal(ref, mine))


class Side:
    """A peer buffer, an epoch and a scratch a width, as the spec builds them."""

    def __init__(self, seconds=30.0):
        self.peer = PeerBuffer(
            k2.peer_buffer_bytes(), dist.group.WORLD, rank, L.TP, DEV
        )
        self.peer.bytes.zero_()
        torch.cuda.synchronize()
        dist.barrier()  # no peer writes into a buffer before its owner zeroed it
        self.epoch = epoch(DEV, seconds)
        self.scratch = {}

    def run(self, key, w, core, residual, out, res_out):
        if key.tokens not in self.scratch:
            n = max(
                k2.scratch_bytes(k2.K2Build(tokens=key.tokens, fse=f))
                for f in (False, True)
            )
            self.scratch[key.tokens] = torch.zeros(n, dtype=torch.uint8, device=DEV)
        self.epoch[:1].add_(1)
        k2.layer_post(
            key,
            core=core,
            residual=residual,
            **w,
            out=out,
            res_out=res_out,
            scratch=self.scratch[key.tokens],
            peers=self.peer.addresses,
            rank=rank,
            epoch=self.epoch,
            layer=k2.TAG_SLOTS // 2,
        )

    def close(self):
        self.peer.close()


def compile_keys(keys):
    """Every rank compiles every key (compile-only) before any rank launches one:
    a rank still compiling would leave the others waiting in the kernel."""
    keys = list(keys)
    start = len(keys) * rank // L.TP  # each rank at its own share of the disk cache
    ok = True
    try:
        for key in keys[start:] + keys[:start]:
            _compile_only(k2.build(key))
        torch.cuda.synchronize()
    except Exception as e:  # reported, then every rank stops
        print(f"[rank {rank}] compile: {type(e).__name__}: {e}", flush=True)
        ok = False
    return all_ok(ok)


def check_compile():
    keys = [
        k2.K2Build(tokens=s, fse=f) for s in range(1, MAXT + 1) for f in (False, True)
    ]
    t0 = time.time()
    ok = compile_keys(keys)
    log(
        f"compile: {len(keys)} K2 keys in {time.time() - t0:.0f} s: {'OK' if ok else 'FAIL'}"
    )
    if not ok:
        raise SystemExit("a rank failed to compile")
    return []


def _rows(probe, s, pos, kind, make):
    if kind == "copy":
        rows = probe.expand(s, *probe.shape[1:]).clone()
    elif kind == "zero":
        rows = torch.zeros(s, *probe.shape[1:], dtype=probe.dtype, device=DEV)
    else:
        rows = make(s)
    rows[pos] = probe[0]
    return rows


def check_batch_invariance(side):
    w = k2_weights(DEV, rank, 100)
    g = gen(DEV, 0)
    probe_r = rnd(g, 1, HIDDEN)  # replicated: one seed on every rank
    probe_c = rnd(gen(DEV, 200 + rank), 1, CORE)  # this rank's o_proj input
    outs = {}
    for s in range(1, MAXT + 1):
        key = k2.K2Build(tokens=s)
        for pos in sorted({0, s - 1}):
            for kind in KINDS if s > 1 else KINDS[:1]:
                g.manual_seed(1 + KINDS.index(kind))
                residual = _rows(probe_r, s, pos, kind, lambda n: rnd(g, n, HIDDEN))
                g.manual_seed(1000 * rank + 1 + KINDS.index(kind))
                core = _rows(probe_c, s, pos, kind, lambda n: rnd(g, n, CORE))
                out = torch.empty(s, HIDDEN, dtype=BF, device=DEV)
                res_out = torch.empty_like(out)
                side.run(key, w, core, residual, out, res_out)
                torch.cuda.synchronize()
                outs[s, pos, kind] = (out[pos].clone(), res_out[pos].clone())
    ref = outs[1, 0, "rand0"][0]
    with_others = [
        c
        for c, got in outs.items()
        if not all(map(torch.equal, got, outs[c[0], c[1], "rand0"]))
    ]
    with_width = [
        c
        for c, got in outs.items()
        if c[2] == "rand0" and not all(map(torch.equal, got, outs[1, 0, "rand0"]))
    ]
    checks = {
        "finite, nonzero": bool(torch.isfinite(ref).all())
        and ref.abs().sum().item() > 0,
        "the probe does not change with the other rows": not with_others,
        "the probe does not change with the width": not with_width,
        "no error word": not any(errors(side.epoch).values()),
    }
    bad = [k for k, v in checks.items() if not v]
    if bad:
        print(
            f"[rank {rank}] batch invariance: {bad} others {with_others} width {with_width}",
            flush=True,
        )
    agree = same_on_every_rank(
        torch.stack([o for got in outs.values() for o in got[:1]])
    )
    ok = all_ok(not bad)
    log(
        f"batch invariance: {len(outs)} cases: {'OK' if ok else 'FAIL'}, ranks agree {agree}"
    )
    return ([] if ok else ["batch invariance"]) + (
        [] if agree else ["batch invariance: ranks differ"]
    )


def check_graph_replay(side, replays):
    w = k2_weights(DEV, rank, 300)
    steps = []
    for j, s in enumerate(REPLAY_WIDTHS):
        residual = rnd(gen(DEV, 400 + j), s, HIDDEN)
        core = rnd(gen(DEV, 500 + 100 * rank + j), s, CORE)
        out = torch.empty(s, HIDDEN, dtype=BF, device=DEV)
        steps.append((k2.K2Build(tokens=s), core, residual, out, torch.empty_like(out)))

    def run_all():
        for key, core, residual, out, res_out in steps:
            side.run(key, w, core, residual, out, res_out)

    def outputs():
        return [t for st in steps for t in st[3:]]

    run_all()
    torch.cuda.synchronize()
    eager = [t.clone() for t in outputs()]
    agree = same_on_every_rank(torch.cat([t.reshape(-1) for t in eager[::2]]))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    dist.barrier()
    bad = 0
    for i in range(replays):
        graph.replay()
        if i % 50 == 49 or i == replays - 1:
            torch.cuda.synchronize()
            bad += sum(not torch.equal(a, b) for a, b in zip(outputs(), eager))
    ok = all_ok(bad == 0 and not any(errors(side.epoch).values()))
    log(
        f"graph replay: {replays} replays of widths {REPLAY_WIDTHS}: {'OK' if ok else 'FAIL'}, ranks agree {agree}"
    )
    if bad:
        print(f"[rank {rank}] graph replay: {bad} mismatching checks", flush=True)
    return ([] if ok else ["graph replay"]) + (
        [] if agree else ["graph replay: ranks differ"]
    )


def check_fault():
    fault = dict(fault_site=1) if rank == 3 else {}
    key = k2.K2Build(tokens=1, stop=False, **fault)
    if not compile_keys([key]):
        raise SystemExit("a rank failed to compile the fault key")
    w = k2_weights(DEV, rank, 77)
    g = gen(DEV, 78)
    core, residual = rnd(g, 1, CORE), rnd(g, 1, HIDDEN)
    side = Side(seconds=0.5)
    try:
        out = torch.empty(1, HIDDEN, dtype=BF, device=DEV)
        t0 = time.time()
        side.run(key, w, core, residual, out, torch.empty_like(out))
        torch.cuda.synchronize()
        dt = time.time() - t0
        words = errors(side.epoch)
        flags = side.peer.bytes.view(torch.int32)[
            k2.FLAGS_OFF // 4 : k2.FLAGS_OFF // 4 + L.TP
        ].tolist()
    finally:
        side.close()
    bad = []
    if rank == 3 and words[guard.ERR_DEADLINE] != 2:
        bad.append(f"rank 3 error words {words}, want deadline word 2")
    # with stop=False rank 3 sends its data after the deadline, so a peer may
    # finish its waits before it sees the flag; the e2e test stops on it
    if flags[3] != 1:
        bad.append(f"rank 3's flag not set here: {flags}")
    if bad:
        print(f"[rank {rank}] fault: {bad} in {dt:.2f} s", flush=True)
    ok = all_ok(not bad)
    log(f"fault: rank 3 stalled, returned in {dt:.2f} s: {'OK' if ok else 'FAIL'}")
    return [] if ok else ["fault"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=200)
    args = ap.parse_args()
    fails = check_compile()
    side = Side()
    try:
        fails += check_batch_invariance(side)
        fails += check_graph_replay(side, args.replays)
    finally:
        side.close()
    dist.barrier()
    fails += check_fault()
    for f in fails:
        log(f"FAIL {f}")
    log("PASS" if not fails else f"FAIL ({len(fails)})")
    dist.destroy_process_group()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
