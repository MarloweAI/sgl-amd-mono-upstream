"""K-attn step times at the server's shapes on one gfx950 GPU, by graph replay:
the sliding layers (window 2047) and the full layers over the AgentX context, at
the served widths. ``python sweep_attention.py``."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_muse_spark_layer import (  # noqa: E402
    LAYER,
    LENS,
    Buffers,
    Layer,
    _attn_args,
    _epoch,
    _time_graph,
)

CASES = (
    (8, 0, [300000]),
    (8, 2047, [300000]),
    (16, 0, [300000, 200000]),
    (8, 0, [95000]),
    (8, 2047, [95000]),
    (8, 0, [20000]),
    (8, 2047, [2500]),
    (16, 0, [95000, 60000]),
    (16, 2047, [95000, 60000]),
    (32, 0, [95000, 300, 17, 60000]),
    (32, 2047, [95000, 300, 17, 60000]),
    (64, 0, LENS),
    (64, 2047, LENS),
)


def main():
    from sgl_amd_mono.muse_spark.gfx950.layer import KAttnBuild, build_kattn

    dev = torch.device("cuda")
    cur = torch.cuda.current_stream
    for S, window, lens in CASES:
        L = Layer(S, 8, dev, seed=50 + S, lens=lens, window=window)
        ka = build_kattn(KAttnBuild(tokens=S, q_len=8))
        epoch = _epoch(dev, deadline_s=60.0)
        B = Buffers(S, 8, dev)
        try:
            B.q.normal_().mul_(5.47)

            def fn():
                ka(*_attn_args(L, B), epoch.data_ptr(), LAYER, stream=cur())
                epoch[0] += 1

            print(
                f"attn S={S:2d} window={window:4d} lens={lens}: graph {_time_graph(fn) * 1000:7.1f} us"
            )
        finally:
            B.close()


if __name__ == "__main__":
    main()
