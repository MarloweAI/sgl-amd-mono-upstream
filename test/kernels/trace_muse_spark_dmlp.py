"""Stage timeline of the D-MLP launch on one gfx950 GPU (a TP group of one): built
with ``trace``, the CTAs stamp the 100 MHz clock at each stage boundary; this runs
three steps and prints when the CTAs reached each stamp (median / p90 / max over
the CTAs, in us from the launch's first stamp). ``python trace_muse_spark_dmlp.py [S]``."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_muse_spark_dmlp import LAYER, _epoch  # noqa: E402

STAMPS = [
    "entry",
    "norm done",
    "coef GEMV done",
    "coef reduced",
    "conv0 done",
    "gate/up GEMV done",
    "act done",
    "down GEMV done",
    "out done",
]


def main():
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.common.plan import BLOCKS
    from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D
    from sgl_amd_mono.muse_spark.gfx950.device import TRACE_LAYERS, TRACE_SLOTS
    from sgl_amd_mono.muse_spark.gfx950.layout import scratch_bytes

    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    dev = torch.device("cuda")
    torch.manual_seed(400 + rows)
    bf = torch.bfloat16
    res = torch.randn(rows, D.H, device=dev).to(bf)
    attn = (torch.randn(rows, D.H, device=dev) * 0.5).to(bf)
    w = {
        "w_post": (1.0 + 0.1 * torch.randn(D.H, device=dev)).to(bf),
        "w_kp": (torch.randn(D.COEF_ROWS, D.H, device=dev) * 0.01).to(bf),
        "base": (torch.randn(2, D.TAPS, D.H, device=dev) * 0.05 + 1).to(bf),
        "w_gu": (torch.randn(D.GU_ROWS, D.H, device=dev) * 0.02).to(bf),
        "w_down": (torch.randn(D.H, D.I, device=dev) * 0.02).to(bf),
    }
    res_out = torch.empty(rows, D.H, device=dev, dtype=bf)
    out = torch.empty(rows, D.H, device=dev, dtype=bf)
    layout = D.dm_scratch_layout()
    scratch = torch.zeros(scratch_bytes(layout), dtype=torch.uint8, device=dev)
    peers = PeerBuffer(D.dm_sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, dev)
    peers.bytes.zero_()
    epoch = _epoch(dev)
    launch = D.build_dmlp(D.DMLPBuild(tokens=rows, npes=1, trace=True))
    try:
        for _ in range(3):
            launch(
                res.data_ptr(), attn.data_ptr(), w["w_post"].data_ptr(), w["w_kp"].data_ptr(), w["base"].data_ptr(),
                w["w_gu"].data_ptr(), w["w_down"].data_ptr(), res_out.data_ptr(), out.data_ptr(), scratch.data_ptr(),
                peers.local, peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, stream=torch.cuda.current_stream(),
            )  # fmt: skip
            epoch[0] += 1
            torch.cuda.synchronize()
        off, nbytes = layout["dm_trace"]
        t = (
            scratch[off : off + nbytes]
            .view(torch.int64)
            .reshape(-1)[: TRACE_LAYERS * BLOCKS * TRACE_SLOTS]
        )
        t = t.reshape(TRACE_LAYERS, BLOCKS, TRACE_SLOTS)[LAYER % TRACE_LAYERS].cpu()
        used = [i for i in range(len(STAMPS)) if (t[:, i] != 0).any()]
        t0 = t[:, 0][t[:, 0] != 0].min().item()
        us = (t.double() - t0) / 100.0
        print(f"D-MLP S={rows}: launch span {us[:, used].max().item():7.1f} us")
        for i in used:
            v = us[:, i][t[:, i] != 0]
            print(
                f"  {STAMPS[i]:>18s}: p50 {v.median().item():6.1f} p90 {v.quantile(0.9).item():6.1f} max {v.max().item():6.1f}"
            )
    finally:
        peers.close()


if __name__ == "__main__":
    main()
