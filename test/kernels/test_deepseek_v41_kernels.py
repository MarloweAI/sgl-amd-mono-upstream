"""The DeepSeek-V4.1 gfx950 monokernels against SGLang's ops (RFC #43949).

Each ``deepseek_v41/check_*.py`` (K1, K2, the FFN launch on both routes, the sampled Markov block)
under torchrun at TP2 and TP4: outputs within the spec's bound, bitwise graph replays with every
rank agreeing; and ``check_glue.py`` (the step glue) on one GPU."""

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


CHECKS = Path(__file__).with_name("deepseek_v41")


@unittest.skipIf(_gfx950_gpus() < 4, "needs 4 gfx950 GPUs")
class TestMonoDeepseekV41Kernels(CustomTestCase):
    def _check(self, module: str, *args: str) -> None:
        for tp in (2, 4):
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={tp}",
                f"--master-port={29600 + tp}",
                str(CHECKS / f"{module}.py"),
                *args,
            ]
            result = subprocess.run(cmd, timeout=1800)
            self.assertEqual(result.returncode, 0, f"{module} at TP{tp}")

    def test_ffn_launch(self):
        self._check("check_ffn", "--replays", "200")

    def test_ffn_launch_draft_route(self):
        self._check("check_ffn", "--replays", "100", "--draft")

    def test_k1(self):
        self._check("check_k1", "--trials", "2")

    def test_k1_preshuffled(self):
        self._check("check_k1", "--trials", "2", "--shuffled")

    def test_k2(self):
        self._check("check_k2", "--replays", "100")

    def test_k2_preshuffled(self):
        self._check("check_k2", "--replays", "100", "--shuffled")

    def test_sampled_markov_block(self):
        self._check("check_markov", "--replays", "20")

    def test_step_glue(self):
        cmd = [sys.executable, str(CHECKS / "check_glue.py"), "--trials", "8"]
        self.assertEqual(subprocess.run(cmd, timeout=1800).returncode, 0, "check_glue")


if __name__ == "__main__":
    unittest.main()
