"""K-attn at the C8 server shape: eight 95k-key requests (64 rows), full and sliding, by graph replay."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_muse_spark_layer import (  # noqa: E402
    LAYER,
    Buffers,
    Layer,
    _attn_args,
    _epoch,
    _time_graph,
)


def main():
    from sgl_amd_mono.muse_spark.gfx950.layer import KAttnBuild, build_kattn

    dev = torch.device("cuda")
    cur = torch.cuda.current_stream
    for S, window, lens in (
        (64, 0, [95000] * 8),
        (64, 2047, [95000] * 8),
        (32, 0, [95000] * 4),
        (8, 0, [95000]),
    ):
        L = Layer(S, 8, dev, seed=90 + S, lens=lens, window=window)
        ka = build_kattn(KAttnBuild(tokens=S, q_len=8))
        epoch = _epoch(dev, deadline_s=60.0)
        B = Buffers(S, 8, dev)
        try:
            B.q.normal_().mul_(5.47)

            def fn():
                ka(*_attn_args(L, B), epoch.data_ptr(), LAYER, stream=cur())
                epoch[0] += 1

            print(
                f"attn C-shape S={S:2d} window={window:4d} lens={lens[0]}x{len(lens)}: graph {_time_graph(fn) * 1000:7.1f} us"
            )
        finally:
            B.close()


if __name__ == "__main__":
    main()
