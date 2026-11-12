"""Read a server log of ``wide_check`` (its ``sitecustomize.py`` on the server's
``PYTHONPATH``): each rank's last count per width class and output (attention, MoE input,
layer), and each stage's own rerun (``det_<stage>``). Exit 0 when every class asked for ran on every rank and the wide layers differ from
SGLang's nowhere SGLang's own rerun agrees with it (``wide_only`` 0).

    python3 check_wide.py SERVER_LOG [--classes 96,192,384] --ranks 4
"""

import argparse
import re
import sys

LINE = re.compile(
    r"WIDE_CHECK rank=(\d+) class=(\d+) field=(\w+) steps=(\d+) differing=(\d+) "
    r"rerun_differing=(\d+) wide_only=(\d+) of=(\d+)"
)


def last_counts(text: str) -> dict:
    """(rank, class, field) -> (steps, differing, rerun differing, wide only, compared),
    each one's last line."""
    out = {}
    for m in LINE.finditer(text):
        rank, cls, field, *counts = m.groups()
        out[int(rank), int(cls), field] = tuple(map(int, counts))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--classes", default="96,192,384")
    ap.add_argument("--ranks", type=int, required=True)
    a = ap.parse_args()
    counts = last_counts(open(a.log, errors="replace").read())
    ok = True
    for cls in map(int, a.classes.split(",")):
        for rank in range(a.ranks):
            fields = {f: c for (r, k, f), c in counts.items() if (r, k) == (rank, cls)}
            if not fields:
                print(f"class {cls} rank {rank}: no wide step")
                ok = False
            for field, (steps, diff, again, wide_only, of) in sorted(fields.items()):
                if field.startswith("det_"):
                    # a stage run twice on the same inputs: whether its kernels reproduce
                    print(
                        f"class {cls} rank {rank} {field[4:]}: "
                        f"{'reproduces' if again == 0 else 'does not reproduce'} its bits "
                        f"({again} of {of} elements differ between two runs, {steps} steps)"
                    )
                    continue
                good = steps > 0 and of > 0 and wide_only == 0
                ok &= good
                how = "BITWISE" if diff == 0 else "AS SGLANG'S RERUN" if good else "DIFFER"
                print(
                    f"class {cls} rank {rank} {field}: {how}: {steps} layer steps, "
                    f"{diff} of {of} elements differ from SGLang, {again} between SGLang's "
                    f"two runs, {wide_only} where SGLang's runs agree"
                )
    print("ALL BITWISE WHERE SGLANG IS" if ok else "NOT BITWISE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
