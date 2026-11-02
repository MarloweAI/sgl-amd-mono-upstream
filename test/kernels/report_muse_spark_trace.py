"""Per-layer stage timeline of a served Muse Spark rank from a stamp snapshot
(``SGLANG_AMD_MONO_MUSE_TRACE=<dir>``, ``<dir>/rank<r>.pt``): for every layer the
merged launch's stages (the attention, K2, K3 and the next layer's K1) on one clock,
then the sliding / full layers' means and the gaps between launches.
``python report_muse_spark_trace.py rank0.pt [--layers 0,5,9] [--all]``."""

import sys

import torch

BLOCKS, SLOTS, TRACE_LAYERS = 256, 16, 64
STAMPS = {
    "attn": [
        "entry",
        "tasks done",
        "merge done",
        "loop begins",
        "first chunk",
        "loop done",
    ],
    "k2": [
        "entry",
        "og",
        "o done",
        "a_part",
        "x arrived",
        "passes",
        "pooled",
        "end",
        "ar polled",
        "ss polled",
    ],
    "k3": [
        "entry",
        "flags",
        "plan",
        "ug done",
        "dn done",
        "fin done",
        "unpool done",
        "end",
        "unp flags",
        "unp dma",
        "unp mfma",
        "unp pushed",
        "dn mids",
        "fin polled",
    ],
    "k1": ["entry", "x ready", "gemv done", "head/gate done"],
}
WINDOW_US = 600.0  # a launch's stamps lie within this of its latest entry (older ones: the previous step)


def stamps(snap, name):
    t = (
        snap[name]
        .view(torch.int64)
        .reshape(-1)[: TRACE_LAYERS * BLOCKS * SLOTS]
        .reshape(TRACE_LAYERS, BLOCKS, SLOTS)
    )
    return t


def clean(t):
    """[BLOCKS, SLOTS] ticks of one layer: the CTAs of the latest launch (entries within
    WINDOW_US of the latest entry), later stamps older than the entry dropped (stale)."""
    e = t[:, 0]
    if (e != 0).sum() == 0:
        return None
    ref = e[e != 0].max()
    keep = (e != 0) & (e > ref - int(WINDOW_US * 100))
    t = t.clone()
    t[~keep] = 0
    for i in range(1, SLOTS):
        bad = (t[:, i] != 0) & (
            (t[:, i] < t[:, 0]) | (t[:, i] > t[:, 0] + int(WINDOW_US * 100))
        )
        t[bad, i] = 0
    return t


def row(t, names, base):
    out = []
    for i, nm in enumerate(names):
        v = t[:, i]
        v = v[v != 0]
        if v.numel() == 0:
            continue
        us = (v.double() - base) / 100.0
        out.append(f"{nm} {us.median().item():6.1f}/{us.max().item():6.1f}")
    return " | ".join(out)


def main():
    path = sys.argv[1]
    snap = torch.load(path, weights_only=False)
    n = len(snap["kinds"])
    swa = snap["swa"]
    print(
        f"rank {snap['rank']} rows {snap['rows']} q_len {snap['q_len']} layers {n} (sliding {sum(swa)}, full {n - sum(swa)}) use_layer {snap['use_layer']}"
    )
    T = {k: stamps(snap, k) for k in ("attn", "k2", "k3", "k1")}
    want = None
    if "--layers" in sys.argv:
        want = {int(x) for x in sys.argv[sys.argv.index("--layers") + 1].split(",")}
    per = []  # (layer, swa, span, attn_end, k2_end, k3_end, k1_end, next_gap)
    rows_out = []
    for l in range(n):
        if snap["kinds"][l] != "mxfp4":
            continue
        a = clean(T["attn"][l])
        if a is None:
            continue
        base = a[:, 0][a[:, 0] != 0].min()
        k2 = clean(T["k2"][l])
        k3 = clean(T["k3"][l])
        k1n = clean(T["k1"][(l + 1) % TRACE_LAYERS]) if l + 1 < n else None

        def end(t):
            if t is None:
                return None
            v = t[t != 0]
            return (v.max().item() - base.item()) / 100.0 if v.numel() else None

        ends = [end(a), end(k2), end(k3), end(k1n)]
        # the next launch's first entry, from this launch's start
        nxt = None
        if l + 1 < n and snap["kinds"][l + 1] == "mxfp4":
            an = clean(T["attn"][l + 1])
            if an is not None:
                nb = an[:, 0][an[:, 0] != 0].min()
                d = (nb.item() - base.item()) / 100.0
                nxt = d if 0 < d < 5000 else None
        per.append((l, swa[l], ends, nxt))
        if want is None or l in want:
            rows_out.append((l, swa[l], base, a, k2, k3, k1n, ends, nxt))

    # the summary
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    for label, sel in (("sliding", True), ("full", False)):
        ps = [p for p in per if p[1] == sel]
        if not ps:
            continue
        print(
            f"== {label} layers (n={len(ps)}): attn end {mean([p[2][0] for p in ps]):6.1f}  K2 end {mean([p[2][1] for p in ps]):6.1f}  "
            f"K3 end {mean([p[2][2] for p in ps]):6.1f}  next K1 end {mean([p[2][3] for p in ps]):6.1f}  next launch entry {mean([p[3] for p in ps]):6.1f} us"
        )
    verbose = "--all" in sys.argv or want is not None
    for l, is_swa, base, a, k2, k3, k1n, ends, nxt in rows_out:
        if not verbose and l not in (0, 1, 2, 5, n - 2, n - 1):
            continue
        print(
            f"-- layer {l} ({'sliding' if is_swa else 'full'}): ends attn {ends[0]} K2 {ends[1]} K3 {ends[2]} K1next {ends[3]}; next launch entry {nxt}"
        )
        print(f"   attn: {row(a, STAMPS['attn'], base)}")
        if k2 is not None:
            print(f"   K2  : {row(k2, STAMPS['k2'], base)}")
        if k3 is not None:
            print(f"   K3  : {row(k3, STAMPS['k3'], base)}")
        if k1n is not None:
            print(f"   K1' : {row(k1n, STAMPS['k1'], base)}")


if __name__ == "__main__":
    main()
