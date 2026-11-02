"""Repeat a whole-layer step (the layer's K1, then the merged launch) at a long full-attention
context and compare the attention output with the exact attention every time: the
served 95k full layers showed intermittent wrong heads. Also the standalone K-attn
launch the same number of times. ``python stress_muse_spark_attention.py [iters] [lens...]``"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_muse_spark_kattn import _reference  # noqa: E402
from test_muse_spark_layer import (  # noqa: E402
    LAYER,
    Buffers,
    Layer,
    _epoch,
    _k1,
    _kernels,
    run_merged,
)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    lens = [int(x) for x in sys.argv[2:]] or [95000]
    dev = torch.device("cuda")
    ql = 8
    S = ql * len(lens)
    from sgl_amd_mono.muse_spark.gfx950.config import SOFTMAX_SCALE

    L = Layer(S, ql, dev, seed=5, lens=lens, window=0)
    K = _kernels(S, ql)
    B = Buffers(S, ql, dev)
    epoch = _epoch(dev, deadline_s=60.0)
    stream = torch.cuda.current_stream()
    bad_merged = bad_alone = 0
    worst = 1.0
    try:
        ref = None
        for it in range(iters):
            B.res.copy_(L.res0)
            _k1(
                K["k1"],
                L,
                B,
                L.x0,
                L.w_qkvg,
                L.k_cache,
                L.v_cache,
                epoch,
                LAYER,
                0,
                stream,
            )
            torch.cuda.synchronize()
            q = B.q[:S].clone()
            if ref is None:
                ref = _reference(
                    q, L.k_cache, L.v_cache, L.table, L.seq_lens, ql, 0, SOFTMAX_SCALE
                )
            B.o.fill_(float("nan"))
            run_merged(K, L, B, epoch, stream)
            epoch[0] += 1
            torch.cuda.synchronize()
            o = B.o[:S].float()
            cos = torch.nn.functional.cosine_similarity(
                o.reshape(S, -1, 64), ref.reshape(S, -1, 64), dim=-1
            )
            worst = min(worst, cos.min().item())
            if cos.min().item() < 0.999 or o.isnan().any():
                bad_merged += 1
                if bad_merged <= 5:
                    low = (cos < 0.999).nonzero().tolist()
                    print(
                        f"merged iter {it}: head cosine min {cos.min().item():.5f}, low (row, head) {low[:16]}",
                        flush=True,
                    )
            # the standalone attention on the same q
            B.q[:S].copy_(q)
            B.o.fill_(float("nan"))
            K["ka"](
                *__import__("test_muse_spark_layer")._attn_args(L, B),
                epoch.data_ptr(),
                LAYER,
                stream=stream,
            )
            epoch[0] += 1
            torch.cuda.synchronize()
            o2 = B.o[:S].float()
            cos2 = torch.nn.functional.cosine_similarity(
                o2.reshape(S, -1, 64), ref.reshape(S, -1, 64), dim=-1
            )
            if cos2.min().item() < 0.999 or o2.isnan().any():
                bad_alone += 1
                if bad_alone <= 5:
                    print(
                        f"standalone iter {it}: head cosine min {cos2.min().item():.5f}",
                        flush=True,
                    )
        errs = epoch[1:8].tolist()
        print(
            f"lens {lens}: {iters} iterations; merged bad {bad_merged}, standalone bad {bad_alone}; worst merged cosine {worst:.5f}; epoch words {errs}"
        )
    finally:
        B.close()


if __name__ == "__main__":
    main()
