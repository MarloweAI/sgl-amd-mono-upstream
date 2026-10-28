"""The GLM-5.3 gfx950 layer launch on random weights in SGLang's layouts (RFC
#43949), under torchrun at TP4 and TP8 (``glm53/check_layer.py``), at every width the
spec builds at that TP size: each launch kind finishes on every rank with no error
word and the same output, on the bf16 and the E4M3 MLA cache, and writes exactly its
live rows' cache rows (none of a pad row's); a graph of the model order's first
launches, and of layer 0, the dense layers, the first MoE layers and the tail,
replays the eager step bitwise, for one request a row and for verify rows; every
selected token of a row past 2048 keys is a pair tagged with its indexer launch's
tag that holds the slot the launch publishes; a dense layer's MLP and the tail's
logits are within the spec's bounds against f32 references, and fail them with a
scale or the final norm off; layer 0 from the token ids equals the dense launch fed
the gathered embedding rows; and every indexer row's list is the exact top-k of the
launch's scores in the selection's order (``glm53/check_topk.py``), on random, tied,
zero and recency-heavy index caches, at about 8K and 300K keys (and for layer 0 and
a dense layer at 8K and 128K)."""

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


CHECK = Path(__file__).with_name("glm53") / "check_layer.py"
TOPK = Path(__file__).with_name("glm53") / "check_topk.py"
# the widths the spec builds at each TP size (TP4 runs at most 4 rows)
WIDTHS = {4: "2,4", 8: "2,4,8"}


@unittest.skipIf(_gfx950_gpus() < 8, "needs 8 gfx950 GPUs")
class TestMonoGlm53Kernels(CustomTestCase):
    def _check(self, *args: str, script: Path = CHECK) -> None:
        for tp, rows in WIDTHS.items():
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={tp}",
                f"--master-port={29630 + tp}",
                str(script),
                "--rows",
                rows,
                *args,
            ]
            result = subprocess.run(cmd, timeout=3600)
            self.assertEqual(result.returncode, 0, f"{' '.join(args)} at TP{tp}")

    def test_each_launch_kind_writes_its_rows(self):
        # the last row a pad row
        for kind in ("shared", "index", "dense", "embed"):
            for kv in ("bf16", "fp8"):
                self._check("--kind", kind, "--kv", kv, "--rowsets", "p1", "--writes")

    def test_graph_replays_equal_the_eager_step(self):
        # 3 shared launches, an indexer launch, 3 shared, an indexer launch
        self._check(
            "--kind",
            "model",
            "--launches",
            "8",
            "--replays",
            "20",
            "--rowsets",
            "p0,v4",
        )

    def test_graph_replays_with_the_dense_layers_and_the_tail(self):
        # layer 0 from the tokens, 2 dense launches, 3 shared, an indexer launch and
        # the tail
        self._check(
            "--kind", "full", "--launches", "7", "--replays", "20", "--rowsets", "p1,v4"
        )

    def test_dense_mlp_within_its_bounds(self):
        for kv in ("bf16", "fp8"):
            self._check(
                "--kind", "dense", "--kv", kv, "--rowsets", "p0,p1", "--check-dense"
            )

    def test_layer_0_from_the_tokens_equals_the_dense_launch(self):
        self._check(
            "--kind", "embed", "--launches", "1", "--rowsets", "p0,p1", "--check-embed"
        )

    def test_tail_logits_within_their_bounds(self):
        self._check("--kind", "head", "--launches", "0", "--check-head")

    def test_selected_tokens_carry_the_launch_tag(self):
        self._check("--kind", "index", "--check-tags", "--rowsets", "p0,v4")

    def test_indexer_selects_the_exact_top_k(self):
        self._check("--pos", "8447,300000", script=TOPK)
        # layer 0 from the tokens and a dense layer: the same indexer stages
        self._check("--kind", "embed", "--pos", "8447,131071", script=TOPK)


if __name__ == "__main__":
    unittest.main()
