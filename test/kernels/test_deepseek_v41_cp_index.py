# SPDX-License-Identifier: Apache-2.0
"""The context-parallel index selection (``indexer/topk_one`` ``Exchange``) across 4 and 8 gfx950
ranks: per row, the sets of each rank's whole-row selection (``deepseek_v41/check_cp_index.py``)."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

CHECK = Path(__file__).with_name("deepseek_v41") / "check_cp_index.py"


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


@pytest.mark.parametrize("tp", [4, 8])
def test_cp_selection_equals_whole_rows(tp):
    if _gfx950_gpus() < tp:
        pytest.skip(f"needs {tp} gfx950 GPUs")
    with tempfile.TemporaryDirectory() as d:
        init = Path(d) / "init"
        procs = [
            subprocess.Popen(
                [
                    sys.executable,
                    str(CHECK),
                    "--rank",
                    str(r),
                    "--world-size",
                    str(tp),
                    "--init-file",
                    str(init),
                ],
                stdout=subprocess.PIPE if r == 0 else subprocess.DEVNULL,
                stderr=subprocess.STDOUT if r == 0 else None,
                text=True,
                env={**os.environ},
            )
            for r in range(tp)
        ]
        out, _ = procs[0].communicate(timeout=1800)
        rcs = [p.wait(timeout=600) for p in procs]
    assert rcs == [0] * tp, (rcs, out[-4000:])
    assert "CP_CHECK PASS" in out, out[-4000:]
