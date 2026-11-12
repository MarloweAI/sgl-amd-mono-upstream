"""The refactor proof (RFC #43949): the ported kernels, with every wait guarded,
against the kernels of sgl-project/sglang#43516, bit for bit, on 8 gfx950 GPUs
(torchrun). Needs both packages: ``sglang.kernels.ops.moe.qwen3_5_mono_flydsl``
(the #43516 head, 51b0c592b2) and ``sgl_amd_mono.qwen38_a95b_mxfp4.gfx950``.

1. K2: out and res_out on every rank, for widths 1 .. 8.
2. ``--time``: the time a launch, #43516 against the port, widths 1 .. 8, from a
   CUDA graph of launches.
"""

import argparse
import datetime
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
from inputs import CORE, HIDDEN, MAXT, epoch, errors, gen, k2_weights, rnd

TP = 8
TIME_ITERS = 200
TIME_ROUNDS = 6

rank = int(os.environ["RANK"])
assert int(os.environ["WORLD_SIZE"]) == TP, f"needs {TP} ranks"
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
dist.init_process_group("gloo", timeout=datetime.timedelta(hours=2))  # the compiles

from sglang.kernels.ops.moe.k3_mono_flydsl.common.peer_memory import (  # noqa: E402
    PeerBuffer as OldPeer,
)
from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import layer as ol  # noqa: E402

from sgl_amd_mono.common.peer import PeerBuffer as NewPeer  # noqa: E402
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layer as nl  # noqa: E402

failures: list[str] = []
report: dict = {"rank": rank, "k2": [], "time": {}}


def log(msg):
    print(f"[rank {rank}] {msg}", flush=True)


def all_ok(flag: bool) -> bool:
    t = torch.tensor([0 if flag else 1])
    dist.all_reduce(t)
    return int(t) == 0


def old_epoch():
    return torch.zeros(1, dtype=torch.int32, device=dev)


# ---------------------------------------------------------------- K2


def k2_cases():
    return [dict(tokens=s) for s in range(1, MAXT + 1)]


class K2Side:
    """One implementation's peer buffer, epoch and scratch per key."""

    def __init__(self, mod, peer_cls, nbytes, ep):
        self.mod = mod
        self.peer = peer_cls(nbytes, dist.group.WORLD, rank, TP, dev)
        self.peer.bytes.zero_()
        torch.cuda.synchronize()
        dist.barrier()  # no peer writes into a buffer before its owner zeroed it
        self.epoch = ep
        self.scratch = {}
        self.base = None

    @classmethod
    def on(cls, mod, base):
        """``mod``'s kernel on ``base``'s peer buffer, epoch and scratch: the
        #43516 layouts are a prefix of the port's (it adds the flags past the
        end), so the two kernels then differ only in their code."""
        s = cls.__new__(cls)
        s.mod, s.peer, s.epoch, s.scratch, s.base = mod, base.peer, base.epoch, {}, base
        return s

    def scratch_for(self, key):
        if self.base is not None:
            k = self.base.mod.K2Build(tokens=key.tokens)
            assert self.mod.scratch_bytes(key) <= self.base.mod.scratch_bytes(k)
            return self.base.scratch_for(k)
        if key not in self.scratch:
            self.scratch[key] = torch.zeros(
                self.mod.scratch_bytes(key), dtype=torch.uint8, device=dev
            )
        return self.scratch[key]

    def run(self, case, w, core, residual):
        key = self.mod.K2Build(**case)
        scratch = self.scratch_for(key)
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        self.epoch[:1].add_(1)
        self.mod.layer_post(
            key,
            core=core,
            residual=residual,
            **w,
            out=out,
            res_out=res_out,
            scratch=scratch,
            peers=self.peer.addresses,
            rank=rank,
            epoch=self.epoch,
            layer=self.mod.TAG_SLOTS // 2,
        )
        return out, res_out

    def close(self):
        if self.base is None:
            self.peer.close()


def compile_all(launches):
    """Compile every launch on every rank before any rank runs one: FlyDSL
    compiles at the first call, and a rank still compiling would leave the
    others waiting in the kernel (past the port's deadline)."""
    from flydsl.utils.env import CompileEnvManager

    var = CompileEnvManager.options["compile_only"].env_var
    ok = True
    os.environ[var] = "1"
    try:
        for fn in launches:
            fn()
    except Exception as e:  # reported, then every rank stops
        failures.append(f"compile: {type(e).__name__}: {e}")
        log(failures[-1])
        ok = False
    finally:
        os.environ.pop(var, None)
    torch.cuda.synchronize()
    if not all_ok(ok):
        raise SystemExit("a rank failed to compile")


def check_k2(timing):
    cases = k2_cases()
    weights = k2_weights(dev, rank, 5000)
    old = K2Side(ol, OldPeer, ol.peer_bytes(MAXT), old_epoch())
    new = K2Side(nl, NewPeer, nl.peer_buffer_bytes(), epoch(dev))

    def args(i, case):
        residual = rnd(gen(dev, 9000 + i), case["tokens"], HIDDEN)
        core = rnd(gen(dev, 9500 + 1000 * rank + i), case["tokens"], CORE)
        return weights, core, residual

    t0 = time.time()
    compile_all(
        lambda side=side, i=i, c=c: side.run(c, *args(i, c))
        for i, c in enumerate(cases)
        for side in (old, new)
    )
    log(f"K2 compiled {2 * len(cases)} kernels in {time.time() - t0:.0f} s")
    try:
        for i, case in enumerate(cases):
            w, core, residual = args(i, case)
            a = old.run(case, w, core, residual)
            torch.cuda.synchronize()
            b = new.run(case, w, core, residual)
            torch.cuda.synchronize()
            bad = [
                n for n, x, y in zip(("out", "res_out"), a, b) if not torch.equal(x, y)
            ]
            finite = bool(torch.isfinite(a[0]).all())
            report["k2"].append({**case, "bad": bad, "finite": finite})
            if bad or not finite:
                failures.append(f"K2 {case}: {bad} finite={finite}")
            log(f"K2 {case} {'OK' if not bad else 'DIFF ' + str(bad)} finite={finite}")
        if any(errors(new.epoch).values()):
            failures.append(f"K2 error words {errors(new.epoch)}")
        if timing:
            time_k2(old, new, weights)
    finally:
        old.close()
        new.close()


# ---------------------------------------------------------------- timing


def time_launches(fn, sync=lambda: None):
    """Microseconds a launch (and its epoch bump) in a graph of TIME_ITERS
    launches, as the server replays them: no host time in the number."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(TIME_ITERS):
            fn()
    sync()
    graph.replay()
    torch.cuda.synchronize()
    sync()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    graph.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / TIME_ITERS


def time_arms(name, arms, sync=lambda: None):
    """TIME_ROUNDS rounds of every arm in turn, the order reversed each round."""
    rows = {k: [] for k in arms}
    for i in range(TIME_ROUNDS):
        for k, fn in list(arms.items())[:: 1 if i % 2 == 0 else -1]:
            rows[k].append(round(time_launches(fn, sync), 2))
    t = {k: statistics.median(v) for k, v in rows.items()}
    report["time"][name] = {**t, "rounds": rows}
    log(
        f"{name} us: "
        + " ".join(f"{k} {v:.1f}" for k, v in t.items())
        + f" rounds {rows}"
    )


def time_k2(old, new, w):
    """``old_nb``: the #43516 kernel on the port's buffers. old against old_nb
    is the placement of the buffers; old_nb against new is the code."""
    old_nb = K2Side.on(ol, new)
    for s in range(1, MAXT + 1):
        case = dict(tokens=s)
        residual = rnd(gen(dev, 6100), s, HIDDEN)
        core = rnd(gen(dev, 6200 + rank), s, CORE)
        time_arms(
            f"k2_s{s}",
            {
                "old": lambda: old.run(case, w, core, residual),
                "old_nb": lambda: old_nb.run(case, w, core, residual),
                "new": lambda: new.run(case, w, core, residual),
            },
            dist.barrier,
        )


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true", help="time the launches too")
    ap.add_argument("--out", default=os.environ.get("CHECK_OUT"), help="per-rank JSON")
    args = ap.parse_args()
    t0 = time.time()
    check_k2(args.time)
    report["failures"] = failures
    report["seconds"] = round(time.time() - t0)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(f"{args.out}/rank{rank}.json", "w") as f:
            json.dump(report, f, indent=1, default=str)
    ok = all_ok(not failures)
    log(
        f"{'PASS' if ok else 'FAIL'} ({len(failures)} failures here) in {report['seconds']} s"
    )
    for x in failures:
        log(f"FAIL {x}")
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
