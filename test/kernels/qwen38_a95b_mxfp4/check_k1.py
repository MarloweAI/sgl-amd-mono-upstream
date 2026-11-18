"""Check K1 (``gdn_pre``, the GDN layer front) on one gfx950 GPU, on random
weights in SGLang's layouts:

- compile: every build key the spec can make (widths 1 .. 8, first layer, bf16
  state) compiles;
- pad rows: K1 writes its live rows' core and state slots, zeros a pad row's core
  (state index <= 0), and leaves every other slot alone;
- bf16 state: the fp32 recurrence on the widened state, stored rounded: the same
  core and conv state as the fp32 kernel, and its new state in bf16, bit for bit;
- batch invariance: a probe row's outputs do not change with the step's other
  rows (other requests, pad rows, copies of the probe) or the width (ATOM's
  MiniMax-M3 K4 failed this at 11 and 16 rows, ROCm/ATOM#2485);
- graph replay: widths 8, 1, 3, 8, 2, 5 on one state in one graph, replayed,
  bit for bit against eager;
- fault: a stalled first wait sets the deadline error word (``stop=False``).
"""

import argparse
import sys
import time

import torch
from inputs import (
    BF,
    CORE,
    HIDDEN,
    MAXT,
    epoch,
    errors,
    gen,
    k1_state,
    k1_weights,
    rnd,
)

from sgl_amd_mono.common import guard
from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import gdn
from sgl_amd_mono.qwen38_a95b_mxfp4.spec import _compile_only

DEV = torch.device("cuda", 0)
REPLAY_WIDTHS = (8, 1, 3, 8, 2, 5)
KINDS = ("rand0", "rand1", "zero", "copy")


def run(key, w, hidden, residual, conv, state, idx, core, res_out, scratch, ep):
    ep[:1].add_(1)
    gdn.gdn_pre(
        key,
        **w,
        hidden=hidden,
        residual=residual,
        res_out=res_out,
        conv_state=conv,
        rstate=state,
        st_idx=idx,
        core=core,
        scratch=scratch,
        epoch=ep,
        layer=0,
    )


def scratch(key):
    return torch.zeros(gdn.scratch_bytes(key), dtype=torch.uint8, device=DEV)


def check_compile():
    keys = [
        gdn.K1Build(tokens=s, first=f, state_bf16=b)
        for s in range(1, MAXT + 1)
        for f in (False, True)
        for b in (False, True)
    ]
    t0 = time.time()
    for key in keys:
        _compile_only(gdn.build(key))
    torch.cuda.synchronize()
    print(f"compile: {len(keys)} K1 keys in {time.time() - t0:.0f} s", flush=True)
    return []


def check_pad_rows():
    fails = []
    g = gen(DEV, 0)
    slots = 16
    # (rows, live rows, first layer, pad index): -1 as SGLang's graph pads, 0 too
    for s, n_real, first, pad in (
        (1, 1, True, -1),
        (4, 3, False, -1),
        (7, 6, False, 0),
        (8, 8, False, -1),
    ):
        w = k1_weights(g)
        conv, state = k1_state(g, slots)
        conv0, state0 = conv.clone(), state.clone()
        idx = torch.full((s,), pad, dtype=torch.int32, device=DEV)
        idx[:n_real] = torch.arange(3, 3 + n_real, dtype=torch.int32, device=DEV)
        key = gdn.K1Build(tokens=s, first=first)
        core = torch.full((s, CORE), float("nan"), dtype=BF, device=DEV)
        res_out = torch.empty(s, HIDDEN, dtype=BF, device=DEV)
        ep = epoch(DEV)
        hidden = rnd(g, s, HIDDEN, scale=2.0)
        residual = None if first else rnd(g, s, HIDDEN, scale=2.0)
        run(key, w, hidden, residual, conv, state, idx, core, res_out, scratch(key), ep)
        torch.cuda.synchronize()
        live = idx[:n_real].long()
        keep = torch.ones(slots, dtype=torch.bool, device=DEV)
        keep[live] = False
        checks = {
            "live core finite, nonzero": bool(torch.isfinite(core[:n_real]).all())
            and core[:n_real].abs().sum().item() > 0,
            "pad core zero": bool((core[n_real:] == 0).all()),
            "live state written": not torch.equal(state[live], state0[live])
            and not torch.equal(conv[live], conv0[live]),
            "other slots kept": torch.equal(state[keep], state0[keep])
            and torch.equal(conv[keep], conv0[keep]),
            "no error word": not any(errors(ep).values()),
        }
        bad = [k for k, v in checks.items() if not v]
        print(f"pad rows s={s} live={n_real} first={first} pad={pad}: {bad or 'OK'}")
        fails += [f"pad rows s={s} {b}" for b in bad]
    return fails


def check_bf16_state():
    g = gen(DEV, 1)
    s, slots = 4, 8
    w = k1_weights(g)
    hidden, residual = rnd(g, s, HIDDEN, scale=2.0), rnd(g, s, HIDDEN, scale=2.0)
    conv, state = k1_state(g, slots, bf16=True)
    idx = torch.tensor([1, 2, -1, 4], dtype=torch.int32, device=DEV)
    out = {}
    for bf in (False, True):
        key = gdn.K1Build(tokens=s, state_bf16=bf)
        o = out[bf] = {
            "conv": conv.clone(),
            "state": state.clone() if bf else state.float(),
            "core": torch.empty(s, CORE, dtype=BF, device=DEV),
            "res_out": torch.empty(s, HIDDEN, dtype=BF, device=DEV),
        }
        run(
            key,
            w,
            hidden,
            residual,
            o["conv"],
            o["state"],
            idx,
            o["core"],
            o["res_out"],
            scratch(key),
            epoch(DEV),
        )
    torch.cuda.synchronize()
    f32, b16 = out[False], out[True]
    bad = [n for n in ("core", "res_out", "conv") if not torch.equal(f32[n], b16[n])]
    if torch.equal(b16["state"], state):
        bad.append("state not written")
    if not torch.equal(f32["state"].to(BF), b16["state"]):
        bad.append("state is not the fp32 state rounded")
    print(f"bf16 state: {bad or 'OK'}")
    return [f"bf16 state {b}" for b in bad]


def _rows(probe, s, pos, kind, make):
    if kind == "copy":
        rows = probe.expand(s, *probe.shape[1:]).clone()
    elif kind == "zero":
        rows = torch.zeros(s, *probe.shape[1:], dtype=probe.dtype, device=DEV)
    else:
        rows = make(s)
    rows[pos] = probe[0]
    return rows


def check_batch_invariance():
    g = gen(DEV, 0)
    w = k1_weights(g)
    slots, probe_slot = MAXT + 2, 1
    conv0, state0 = k1_state(g, slots)
    probe_h, probe_r = rnd(g, 1, HIDDEN, scale=2.0), rnd(g, 1, HIDDEN, scale=2.0)
    ep = epoch(DEV)
    # one scratch a width, as the spec has (widths that share one race in a graph)
    keys = {s: gdn.K1Build(tokens=s) for s in range(1, MAXT + 1)}
    scr = {s: scratch(k) for s, k in keys.items()}
    outs = {}
    for s in range(1, MAXT + 1):
        for pos in sorted({0, s - 1}):
            for kind in KINDS if s > 1 else KINDS[:1]:
                g.manual_seed(1 + KINDS.index(kind))
                hidden = _rows(
                    probe_h, s, pos, kind, lambda n: rnd(g, n, HIDDEN, scale=2.0)
                )
                residual = _rows(
                    probe_r, s, pos, kind, lambda n: rnd(g, n, HIDDEN, scale=2.0)
                )
                others = list(range(2, slots))[: s - 1]
                idx = torch.tensor(
                    others[:pos] + [probe_slot] + others[pos:],
                    dtype=torch.int32,
                    device=DEV,
                )
                conv, state = conv0.clone(), state0.clone()
                if kind == "zero":
                    idx[torch.arange(s, device=DEV) != pos] = -1
                elif kind == "copy":
                    conv[others] = conv[probe_slot].clone()
                    state[others] = state[probe_slot].clone()
                elif kind == "rand1":
                    conv[others], state[others] = k1_state(g, len(others))
                core = torch.empty(s, CORE, dtype=BF, device=DEV)
                res_out = torch.empty(s, HIDDEN, dtype=BF, device=DEV)
                run(
                    keys[s],
                    w,
                    hidden,
                    residual,
                    conv,
                    state,
                    idx,
                    core,
                    res_out,
                    scr[s],
                    ep,
                )
                torch.cuda.synchronize()
                outs[s, pos, kind] = (
                    core[pos],
                    res_out[pos],
                    conv[probe_slot],
                    state[probe_slot],
                )
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
    fails = []
    if with_others:
        fails.append(
            f"batch invariance: the probe changes with the other rows {with_others}"
        )
    if with_width:
        fails.append(f"batch invariance: the probe changes with the width {with_width}")
    if any(errors(ep).values()):
        fails.append(f"batch invariance: error words {errors(ep)}")
    print(f"batch invariance: {len(outs)} cases, {fails or 'OK'}")
    return fails


def check_graph_replay(replays):
    """The widths of REPLAY_WIDTHS one after another on slots 1 .. 8 (each step
    reads the state the previous wrote), from one saved state, in one graph."""
    g = gen(DEV, 3)
    w = k1_weights(g)
    slots = MAXT + 1
    conv0, state0 = k1_state(g, slots)
    conv, state = conv0.clone(), state0.clone()
    ep = epoch(DEV)
    keys = {s: gdn.K1Build(tokens=s) for s in set(REPLAY_WIDTHS)}
    scr = {s: scratch(k) for s, k in keys.items()}
    steps = []
    for s in REPLAY_WIDTHS:
        idx = torch.arange(1, s + 1, dtype=torch.int32, device=DEV)
        if s >= 3 and s % 2 == 1:
            idx[-1] = -1  # a pad row
        steps.append(
            (
                s,
                rnd(g, s, HIDDEN, scale=2.0),
                rnd(g, s, HIDDEN, scale=2.0),
                idx,
                torch.empty(s, CORE, dtype=BF, device=DEV),
                torch.empty(s, HIDDEN, dtype=BF, device=DEV),
            )
        )

    def run_all():
        conv.copy_(conv0)
        state.copy_(state0)
        for s, hidden, residual, idx, core, res_out in steps:
            run(
                keys[s],
                w,
                hidden,
                residual,
                conv,
                state,
                idx,
                core,
                res_out,
                scr[s],
                ep,
            )

    def outputs():
        return [t for st in steps for t in st[4:]] + [conv, state]

    run_all()
    torch.cuda.synchronize()
    eager = [t.clone() for t in outputs()]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    bad = 0
    for i in range(replays):
        graph.replay()
        if i % 50 == 49 or i == replays - 1:
            torch.cuda.synchronize()
            bad += sum(not torch.equal(a, b) for a, b in zip(outputs(), eager))
    print(
        f"graph replay: {replays} replays of widths {REPLAY_WIDTHS}: {bad} mismatching checks"
    )
    fails = [f"graph replay: {bad} mismatching checks"] if bad else []
    if any(errors(ep).values()):
        fails.append(f"graph replay: error words {errors(ep)}")
    return fails


def check_fault():
    g = gen(DEV, 77)
    w = k1_weights(g)
    conv, state = k1_state(g, 4)
    key = gdn.K1Build(tokens=1, fault_site=1, stop=False)
    ep = epoch(DEV, 0.5)
    idx = torch.ones(1, dtype=torch.int32, device=DEV)
    core = torch.empty(1, CORE, dtype=BF, device=DEV)
    res_out = torch.empty(1, HIDDEN, dtype=BF, device=DEV)
    t0 = time.time()
    run(
        key,
        w,
        rnd(g, 1, HIDDEN),
        rnd(g, 1, HIDDEN),
        conv,
        state,
        idx,
        core,
        res_out,
        scratch(key),
        ep,
    )
    torch.cuda.synchronize()
    words = errors(ep)
    print(f"fault: error words {words} in {time.time() - t0:.2f} s")
    if words[guard.ERR_DEADLINE] != 2:
        return [f"fault: error words {words}, want deadline word 2 (site 1)"]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=200)
    args = ap.parse_args()
    torch.cuda.set_device(DEV)
    fails = check_compile()
    fails += check_pad_rows()
    fails += check_bf16_state()
    fails += check_batch_invariance()
    fails += check_graph_replay(args.replays)
    fails += check_fault()
    for f in fails:
        print(f"FAIL {f}", flush=True)
    print("PASS" if not fails else f"FAIL ({len(fails)})", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
