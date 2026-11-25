# SPDX-License-Identifier: Apache-2.0
"""Checks that ``moe_fused`` is bitwise run to run where it combines the routes without atomics (from
``experts.PARTS_ROWS`` rows): one GPU, synthetic MXFP4 weights (``check_experts.weights``), CUDA-graph replay.

Each case captures ``moe_fused`` once and replays it on fixed inputs; every replay's output (the valid rows) must be
bitwise the same as the first's. Cases: exact row counts (``valid`` = rows), and padded buckets as SGLang's decode
graphs run them (captured at the bucket, ``valid`` below it, the pad rows refilled with new random data before every
replay). Counters must be back to 0 after the replays.

Usage: python3 check_determinism.py [--rows 32,36,...] [--padded 48:36,...] [--replays 25]"""

import argparse
import sys

import torch
from check_experts import counters_left, weights
from check_front import LIMIT, SCALE, inputs

import sgl_amd_mono.glm53_flash_moe as fm

dev = torch.device("cuda", torch.cuda.current_device())


def replays(m, valid, n, ws, raw, g):
    """Distinct outputs (valid rows) of ``n`` replays of ``moe_fused`` captured at ``m`` rows with ``valid`` valid."""
    w13, w2, s13, s2 = ws
    x0, gw, bias = inputs(m, g)
    x = x0.clone()
    v = torch.tensor([valid], dtype=torch.int32, device=dev)

    def run():
        return fm.moe_fused(x, gw, bias, SCALE, w13, w2, s13, s2, LIMIT, *raw, valid=v)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    first, distinct = None, 1
    for _ in range(n):
        x.copy_(x0)
        if valid < m:
            x[valid:] = torch.randn(m - valid, fm.D, generator=g, device=dev).to(torch.bfloat16)
        graph.replay()
        torch.cuda.synchronize()
        got = out[:valid].clone()
        if first is None:
            first = got
        elif not torch.equal(got.view(torch.int16), first.view(torch.int16)):
            distinct += 1
    return distinct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="32,36,40,44,48,64")
    ap.add_argument("--padded", default="48:36,48:40,48:44")
    ap.add_argument("--replays", type=int, default=25)
    a = ap.parse_args()
    ws, raw = weights(torch.Generator(device=dev).manual_seed(5))
    cases = [(int(r), int(r)) for r in a.rows.split(",") if r]
    cases += [tuple(int(t) for t in p.split(":")) for p in a.padded.split(",") if p]
    ok = True
    for m, valid in cases:
        n = replays(m, valid, a.replays, ws, raw, torch.Generator(device=dev).manual_seed(m * 1000 + valid))
        left = counters_left()
        good = n == 1 and not left
        ok &= good
        print(f"determinism m={m} valid={valid}: {n} distinct output(s) in {a.replays} replays"
              f"{f', {left} counters not back to 0' if left else ''} {'OK' if good else 'FAILED'}", flush=True)
    print("ALL OK" if ok else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
