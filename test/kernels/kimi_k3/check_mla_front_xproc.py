# SPDX-License-Identifier: Apache-2.0
"""The MLA front bit for bit across processes: ``check_mla_front.py`` with
the given arguments and ``--digest`` in three fresh processes, the third with
every launch's scratch filled with 0xA5 first. Every launch's outputs (every
pass of the 8-rank emulation, not only the step's) must hash the same.

    python3 check_mla_front_xproc.py --ranks 8 --ctx 10000 --skip-units
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ([], [], ["--scratch-fill", "165"])


def main():
    digests = []
    for extra in RUNS:
        cmd = [sys.executable, os.path.join(HERE, "check_mla_front.py"), *sys.argv[1:], "--digest", *extra]
        p = subprocess.run(cmd, capture_output=True, text=True)
        m = re.search(r"^DIGEST (\S+) launches=(\d+)", p.stdout, re.M)
        ok = p.returncode == 0 and m is not None
        print(f"{' '.join(extra) or 'zeroed scratch'}: rc={p.returncode} {m.group(0) if m else 'no digest'}", flush=True)
        if not ok:
            print(p.stdout[-2000:], p.stderr[-2000:], sep="\n")
            sys.exit(1)
        digests.append(m.group(1, 2))
    same = len(set(digests)) == 1
    print("XPROC PASS: every process the same bits" if same else f"XPROC FAIL: {digests}", flush=True)
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
