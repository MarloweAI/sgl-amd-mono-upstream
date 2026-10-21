"""The GLM-5.3-Flash launch from the tokens keeps a captured graph's step-input table
apart from eager steps (CPU tensors stand in for the step's)."""

from types import SimpleNamespace

import pytest
import torch

from sgl_amd_mono.glm53_flash.gfx950 import runner
from sgl_amd_mono.glm53_flash.gfx950.kernel import FlashBuild
from sgl_amd_mono.glm53_flash.gfx950.layout import ROW_WORDS


def _op():
    key = FlashBuild(
        rows=4, tp=4, launches_per_step=1, inter=512, kda=True, dense=True, tokens=True
    )
    t = torch.zeros(1)
    dense = runner.FlashDenseWeights(*([t] * 8))
    embedding = runner.FlashEmbedding(weight=torch.zeros(8, 4), start=0, end=8)
    rt = SimpleNamespace(key=key, width=4)
    return runner.FlashLayerOp(rt, dense, 0, kda=object(), embedding=embedding)


def _step(n):
    ids = torch.zeros(n, dtype=torch.int64)
    return {
        "ids": ids,
        "positions": ids.clone(),
        "out_cache_loc": ids.clone(),
        "req_pool_indices": ids.clone(),
        "slots": torch.zeros(4, dtype=torch.int32),
        "rows": torch.zeros(4, ROW_WORDS, dtype=torch.int32),
    }


def test_an_eager_step_never_rewrites_a_captured_table(monkeypatch):
    op, graph_step = _op(), _step(3)
    capturing = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    # the eager step before the capture makes the table; the capture reads it
    eager = op._step_table(graph_step)
    capturing[0] = True
    captured = op._step_table(graph_step)
    assert captured is eager
    held = captured.clone()
    # later eager steps of the same row count, on other tensors, take their own
    # tables; the captured one keeps its values
    capturing[0] = False
    for _ in range(3):
        other = op._step_table(_step(3))
        assert other is not captured
    assert torch.equal(captured, held)
    assert op._step_table(graph_step) is captured
    # only the latest uncaptured table of a row count is kept
    assert len(op.step_tabs) == 2
    # a capture of inputs no eager step used first cannot fill a table
    capturing[0] = True
    with pytest.raises(RuntimeError, match="no eager step used first"):
        op._step_table(_step(3))
