"""The Qwen3.8 MoE gfx950 monokernels (RFC #43949) on random weights in SGLang's
layouts: the routed experts' fp4 tiles and e8m0 scales where K2 reads them (on
the host); K2 on 8 GPUs (``qwen38_a95b_mxfp4/check_k2.py``): every build key
compiles, pad rows, batch invariance, graph replay across widths and an injected
wait fault; and the refactor proof against the sgl-project/sglang#43516 kernels,
bit for bit (``check_port.py``, where that SGLang is installed)."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import torch
from sglang.test.test_utils import CustomTestCase

CHECKS = Path(__file__).with_name("qwen38_a95b_mxfp4")
NE = 2  # experts: enough to catch a wrong expert stride
SGLANG_43516 = "sglang.kernels.ops.moe.qwen3_5_mono_flydsl"


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


@unittest.skipIf(_gfx950_gpus() < 1, "needs aiter on gfx950")
class TestQwen38MoeLayout(CustomTestCase):
    """SGLang's gfx950 MXFP4 MoE layout (quark_w4a4_mxfp4_moe.py:
    ``shuffle_weight(w, (16, 16))`` on the uint8 weight, ``e8m0_shuffle`` over
    (E N, K / 32)): K2 reads the experts in place, so a layout change breaks it
    silently."""

    def _check_weight(self, w, base_of, k):
        from aiter.ops.shuffle import shuffle_weight

        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layout as L

        flat = shuffle_weight(w.contiguous(), (16, 16)).reshape(-1).view(torch.int32)
        lane = torch.arange(64)
        for e in range(NE):
            for rg in range(w.shape[1] // L.ROWS):
                rows = w[e, rg * L.ROWS + lane % 16]  # (64, K / 2)
                for st in range(k // L.FP4_STEP):
                    at = L.fp4_tile_dword(base_of(e), rg, k, st, lane * 0) + lane * 4
                    got = torch.stack([flat[at + q] for q in range(4)], 1)
                    b0 = st * 64 + (lane // 16) * 16  # first byte of the lane's 32 fp4
                    want = torch.stack(
                        [rows[i, b0[i] : b0[i] + 16] for i in range(64)]
                    ).view(torch.int32)
                    self.assertTrue(torch.equal(got, want), (e, rg, st))

    def test_w13_tiles(self):
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layout as L

        g = torch.Generator().manual_seed(0)
        w = torch.randint(
            0, 256, (NE, 2 * L.RI, L.HIDDEN // 2), dtype=torch.uint8, generator=g
        )
        self._check_weight(w, L.w13_base, L.HIDDEN)

    def test_w2_tiles(self):
        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layout as L

        g = torch.Generator().manual_seed(1)
        w = torch.randint(
            0, 256, (NE, L.HIDDEN, L.RI // 2), dtype=torch.uint8, generator=g
        )
        self._check_weight(w, L.w2_base, L.RI)

    def test_scales(self):
        from aiter.utility.fp4_utils import e8m0_shuffle

        from sgl_amd_mono.qwen38_a95b_mxfp4.gfx950 import layout as L

        for rows, k, index in (
            (2 * L.RI, L.HIDDEN, L.w13_scale_index),
            (L.HIDDEN, L.RI, L.w2_scale_index),
        ):
            g = torch.Generator().manual_seed(2)
            s = torch.randint(
                0, 256, (NE, rows, k // 32), dtype=torch.uint8, generator=g
            )
            flat = e8m0_shuffle(s.view(NE * rows, -1)).reshape(-1)
            self.assertEqual(flat.numel(), s.numel())
            n = torch.arange(rows)[:, None]
            kb = torch.arange(k // 32)[None, :]
            for e in range(NE):
                self.assertTrue(torch.equal(flat[index(e, n, kb)], s[e]), (rows, e))


def _run(script: str, *args: str, ranks: int = 0) -> int:
    launcher = [sys.executable]
    if ranks:
        launcher += [
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={ranks}",
            "--master-port=29630",
        ]
    return subprocess.run(
        [*launcher, str(CHECKS / script), *args], timeout=3600
    ).returncode


@unittest.skipIf(_gfx950_gpus() < 8, "needs 8 gfx950 GPUs")
class TestQwen38MoeK2(CustomTestCase):
    def test_k2(self):
        self.assertEqual(_run("check_k2.py", "--replays", "200", ranks=8), 0)


@unittest.skipIf(_gfx950_gpus() < 8, "needs 8 gfx950 GPUs")
@unittest.skipUnless(_has(SGLANG_43516), "needs the sgl-project/sglang#43516 kernels")
class TestQwen38MoePort(CustomTestCase):
    def test_bitwise_against_sglang_43516(self):
        self.assertEqual(_run("check_port.py", ranks=8), 0)


if __name__ == "__main__":
    unittest.main()
