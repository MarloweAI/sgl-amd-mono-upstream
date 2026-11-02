"""Average stage timeline of a served Muse Spark rank over many stamp snapshots
(``SGLANG_AMD_MONO_MUSE_TRACE`` copies, e.g. c1_*/rank0.pt): for each stage stamp,
the median CTA's and the last CTA's time from the launch's first entry, averaged
over the merged-launch layers of every snapshot, sliding and full apart; and the
gap to the next launch. ``python summarize_muse_spark_trace.py rank0.pt [more.pt ...]``"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_muse_spark_trace import STAMPS, TRACE_LAYERS, clean, stamps  # noqa: E402


def main():
    acc = {}  # (kind, stage, stamp) -> [(p50, max)]
    nxt = {True: [], False: []}
    shapes = set()
    for path in sys.argv[1:]:
        try:
            snap = torch.load(path, weights_only=False)
        except Exception as e:  # a snapshot being written
            print(f"skip {path}: {e}")
            continue
        shapes.add((snap["rows"], snap["q_len"]))
        T = {k: stamps(snap, k) for k in ("attn", "k2", "k3", "k1")}
        n = len(snap["kinds"])
        for l in range(n - 1):
            if snap["kinds"][l] != "mxfp4" or snap["kinds"][l + 1] != "mxfp4":
                continue
            a = clean(T["attn"][l])
            if a is None:
                continue
            base = a[:, 0][a[:, 0] != 0].min().item()
            kind = bool(snap["swa"][l])
            parts = {
                "attn": a,
                "k2": clean(T["k2"][l]),
                "k3": clean(T["k3"][l]),
                "k1": clean(T["k1"][(l + 1) % TRACE_LAYERS]),
            }
            ok = True
            for stage, t in parts.items():
                if t is None:
                    ok = False
                    break
            if not ok:
                continue
            for stage, t in parts.items():
                for i, nm in enumerate(STAMPS[stage]):
                    v = t[:, i]
                    v = v[v != 0]
                    if v.numel() == 0:
                        continue
                    us = (v.double() - base) / 100.0
                    if us.max().item() > 2000:
                        continue
                    acc.setdefault((kind, stage, i), []).append(
                        (us.median().item(), us.max().item())
                    )
            an = clean(T["attn"][l + 1])
            if an is not None:
                d = (an[:, 0][an[:, 0] != 0].min().item() - base) / 100.0
                if 0 < d < 2000:
                    nxt[kind].append(d)
    print(f"shapes (rows, q_len): {sorted(shapes)}; snapshots {len(sys.argv) - 1}")
    for kind, label in ((True, "sliding"), (False, "full")):
        if not nxt[kind]:
            continue
        g = torch.tensor(nxt[kind])
        print(
            f"== {label} layers: {len(nxt[kind])} launches; next launch entry p50 {g.median().item():.1f} mean {g.mean().item():.1f} us"
        )
        for stage in ("attn", "k2", "k3", "k1"):
            cells = []
            for i, nm in enumerate(STAMPS[stage]):
                xs = acc.get((kind, stage, i))
                if not xs:
                    continue
                p = sum(x[0] for x in xs) / len(xs)
                m = sum(x[1] for x in xs) / len(xs)
                cells.append(f"{nm} {p:5.1f}/{m:5.1f}")
            print(f"  {stage:4s}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
