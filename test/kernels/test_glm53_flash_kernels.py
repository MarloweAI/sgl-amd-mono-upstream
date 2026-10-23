"""The GLM-5.3-Flash gfx950 layer launch on random weights in SGLang's layouts
(RFC #43949), under torchrun at TP2, TP4 and TP8 (``glm53_flash/check_layer.py``):
each launch kind finishes on every rank with no error word and the same output,
and writes exactly its valid rows' state or cache rows (none of a pad row's); a
sparse-attention launch on AITER-shuffled FP8 projections equals the row-major one
bitwise; the router's top-8 equals SGLang's on keys that tie in bf16; and rows past
2048 tokens attend to their selected pools and then their tail, as SGLang selects; and
a graph of several launches of each kind replays the eager step bitwise at every width
the TP size builds."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch
from sglang.test.test_utils import CustomTestCase


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


CHECK = Path(__file__).with_name("glm53_flash") / "check_layer.py"


@unittest.skipIf(_gfx950_gpus() < 8, "needs 8 gfx950 GPUs")
class TestMonoGlm53FlashKernels(CustomTestCase):
    def _check(self, *args: str, widths=None) -> None:
        for tp in (2, 4, 8):
            if widths is not None:
                # TP2 builds no launch wider than 4 rows
                for w in (w for w in widths if tp > 2 or w <= 4):
                    self._run(tp, *args, "--rows", str(w))
                continue
            self._run(tp, *args)

    def _run(self, tp: int, *args: str) -> None:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={tp}",
            f"--master-port={29610 + tp}",
            str(CHECK),
            *args,
        ]
        result = subprocess.run(cmd, timeout=1800)
        self.assertEqual(result.returncode, 0, f"{' '.join(args)} at TP{tp}")

    def test_each_launch_kind(self):
        # 4 rows (TP2 builds no wider launch), the last a pad row
        for kind in ("ffn", "kda", "dsa"):
            self._check("--kind", kind, "--rows", "4", "--pad-rows", "1", "--writes")

    def test_shuffled_fp8_projections_match_row_major(self):
        self._check("--kind", "dsa", "--rows", "4", "--compare-shuffled")

    def test_routing_matches_sglang_on_ties(self):
        self._check("--kind", "ffn", "--rows", "4", "--check-routing")

    def test_long_rows_attend_their_tail(self):
        self._check("--kind", "dsa", "--check-keys")

    def test_graph_replays_equal_the_eager_step(self):
        for kind in ("kda", "dsa"):
            self._check(
                "--kind", kind, "--layers", "4", "--replays", "20", widths=(2, 4, 8)
            )


if __name__ == "__main__":
    unittest.main()
