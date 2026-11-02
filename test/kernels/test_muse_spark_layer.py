"""The Muse Spark whole-layer launch on one gfx950 GPU (a TP / EP group of one)
against the four launches it merges (K1 of the layer first, then the attention,
K2, K3 and the next layer's K1 as one launch, or as four): the same inputs, the
same stage code, so the residual stream, the next attention input, the routing,
the expert input, the attention output and the next layer's q / k / v / gate and
fp8 cache rows must agree to the last bit but for the attention's partition merge
order; and the step time of each form at every served width when run as a script
(``python test_muse_spark_layer.py``)."""

import os
import struct
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx950 GPU",
)

PAGES = (
    65536  # cache pages of the layer's K / V planes (1M keys: eight 95k requests fit)
)
FP8_DENSE = (
    os.environ.get("MUSE_BENCH_FP8", "0") == "1"
)  # the pooler / unpooler as packed FP8 copies
UG_DYN = (
    os.environ.get("MUSE_BENCH_UGDYN", "0") == "1"
)  # K3's ug items from the launch's queue (merged launch)
MAX_BLOCKS = 65536
POSITIONS = 4096
THETA = 500000.0
LAYER = 9
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
LENS = [
    20000,
    300,
    17,
    95000,
    4096,
    2048,
    2047,
    9000,
]  # a request's keys including the step's rows


def _epoch(device, deadline_s=20.0):
    from sgl_amd_mono.common import guard

    e = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(deadline_s)
    lo = ticks & 0xFFFFFFFF
    e[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    e[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return e


def _cos_sin(device):
    from sgl_amd_mono.muse_spark.gfx950.config import HEAD_DIM

    inv = 1.0 / (
        THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    freqs = torch.arange(POSITIONS, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device)


def _fp4_weight(experts, rows, k, device, gen):
    packed = torch.randint(0, 256, (experts, rows, k // 2), device=device, dtype=torch.uint8, generator=gen)  # fmt: skip
    scales = torch.randint(118, 128, (experts, rows, k // 32), device=device, dtype=torch.uint8, generator=gen)  # fmt: skip
    return packed, scales


class Layer:
    """One MXFP4 layer's weights and caches, and a step's inputs, on the device."""

    def __init__(self, S, ql, dev, seed, lens=None, window=0):
        from sgl_amd_mono.muse_spark.gfx950.config import (
            EXPERTS,
            HEAD_DIM,
            HIDDEN,
            INTER,
            LOCAL_EXPERTS,
            LOCAL_KV_HEADS,
            MOE_HIDDEN,
            Q_K,
            QKVG_ROWS,
            SOFTMAX_SCALE,
        )

        gen = torch.Generator(device=dev).manual_seed(seed)
        torch.manual_seed(seed)
        bf = torch.bfloat16
        self.S, self.ql, self.dev, self.window = S, ql, dev, window
        R = S // ql
        self.x0 = torch.randn(S, HIDDEN, device=dev).to(
            bf
        )  # the layer's attention input
        self.res0 = torch.randn(S, HIDDEN, device=dev)  # the residual stream before it
        self.w_qkvg = (torch.randn(QKVG_ROWS, HIDDEN, device=dev) * 0.02).to(bf)
        self.w_qkvg1 = (torch.randn(QKVG_ROWS, HIDDEN, device=dev) * 0.02).to(
            bf
        )  # the next layer's
        self.cos_sin = _cos_sin(dev)
        self.positions = torch.randint(
            0, POSITIONS, (S,), device=dev, dtype=torch.int64
        )
        lens = list(lens) if lens is not None else LENS[:R]
        self.lens = lens
        self.seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
        self.prefix = (self.seq_lens - ql).contiguous()
        # this layer's cache: random keys, distinct pages a request, the step's rows at the end
        self.k_cache = (
            torch.randn(PAGES * 16, LOCAL_KV_HEADS, HEAD_DIM, device=dev) * 2
        ).to(torch.float8_e4m3fn)
        self.v_cache = torch.randn(PAGES * 16, LOCAL_KV_HEADS, HEAD_DIM, device=dev).to(
            torch.float8_e4m3fn
        )
        perm = torch.randperm(PAGES - 1, device=dev) + 1
        self.table = torch.zeros(R, MAX_BLOCKS, dtype=torch.int32, device=dev)
        slots, at = [], 0
        for r in range(R):
            n = (lens[r] + 15) // 16
            assert at + n <= PAGES - 1
            self.table[r, :n] = perm[at : at + n].to(torch.int32)
            at += n
            for j in range(ql):
                p = lens[r] - ql + j
                slots.append(int(self.table[r, p // 16]) * 16 + p % 16)
        self.slots = torch.tensor(slots, dtype=torch.int64, device=dev)
        # the next layer (sliding): its slot table and empty cache
        self.swa_map = torch.randperm(PAGES * 16, device=dev, dtype=torch.int64)
        self.k_cache1 = torch.zeros_like(self.k_cache)
        self.v_cache1 = torch.zeros_like(self.v_cache)
        self.scale_bits = struct.unpack("i", struct.pack("f", SOFTMAX_SCALE))[0]
        router = torch.randn(HIDDEN, EXPERTS, device=dev) * 0.02
        w_hi = router.T.to(bf).contiguous()
        self.w2k = {
            "w_o": (torch.randn(HIDDEN, Q_K, device=dev) * 0.02).to(bf),
            "alpha": torch.rand(HIDDEN, device=dev) * 0.5 + 0.5,
            "beta": torch.rand(HIDDEN, device=dev) * 0.5,
            "pre_w": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
            "w_hi": w_hi,
            "w_lo": (router.T - w_hi.float()).to(bf).contiguous(),
            "bias": torch.randn(EXPERTS, device=dev) * 0.01,
            "w_pool": (torch.randn(MOE_HIDDEN, HIDDEN, device=dev) * 0.02).to(bf),
            "xn_w": (1.0 + 0.1 * torch.randn(MOE_HIDDEN, device=dev)).to(bf),
        }
        self.w13, self.s13 = _fp4_weight(LOCAL_EXPERTS, 2 * INTER, MOE_HIDDEN, dev, gen)
        self.w2, self.s2 = _fp4_weight(LOCAL_EXPERTS, MOE_HIDDEN, INTER, dev, gen)
        self.w3k = {
            "pen_w": (1.0 + 0.1 * torch.randn(MOE_HIDDEN, device=dev)).to(bf),
            "pfw": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
            "alpha": torch.rand(HIDDEN, device=dev) * 0.5 + 0.5,
            "beta": torch.rand(HIDDEN, device=dev) * 0.5,
            "npw": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
            "w_unpool": (torch.randn(HIDDEN, MOE_HIDDEN, device=dev) * 0.02).to(bf),
        }
        if FP8_DENSE:  # the kernels stream the packed copies (the merged / four launch forms agree either way)
            from sgl_amd_mono.muse_spark.fp8_quant import (
                pack_fp8_rows,
                pack_fp8_rows_pooler,
            )

            self.w2k["w_pool"] = pack_fp8_rows_pooler(
                self.w2k["w_pool"]
            )  # the pooler's lane-swizzled layout
            self.w3k["w_unpool"] = pack_fp8_rows(self.w3k["w_unpool"])


class Buffers:
    """A path's step buffers (the runner's), fresh for each path."""

    def __init__(self, S, ql, dev):
        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.muse_spark.gfx950.config import (
            HIDDEN,
            KV_K,
            MAX_TOKENS,
            MOE_HIDDEN,
            Q_K,
            TOP_K,
        )
        from sgl_amd_mono.muse_spark.gfx950.layout import (
            k1_scratch_layout,
            kattn_scratch_layout,
            scratch_bytes,
        )
        from sgl_amd_mono.muse_spark.gfx950.stages.ffn import (
            k3_scratch_layout,
            k3_sym_layout,
        )
        from sgl_amd_mono.muse_spark.gfx950.stages.post_attn import (
            k2_scratch_layout,
            sym_bytes,
        )

        bf = torch.bfloat16
        nan = float("nan")
        self.q = torch.full((MAX_TOKENS, Q_K), nan, device=dev, dtype=bf)
        self.k = torch.full((MAX_TOKENS, KV_K), nan, device=dev, dtype=bf)
        self.v = torch.full_like(self.k, nan)
        self.g = torch.full_like(self.q, nan)
        self.o = torch.full_like(self.q, nan)
        self.ids = torch.full((MAX_TOKENS, TOP_K), -7, dtype=torch.int32, device=dev)
        self.scores = torch.full((MAX_TOKENS, TOP_K), nan, device=dev)
        self.xin = torch.full((MAX_TOKENS, MOE_HIDDEN), nan, device=dev, dtype=bf)
        self.attn_in = torch.full((MAX_TOKENS, HIDDEN), nan, device=dev, dtype=bf)
        self.res = torch.zeros(S, HIDDEN, device=dev)
        self.scratch_k1 = torch.zeros(
            scratch_bytes(k1_scratch_layout()), dtype=torch.uint8, device=dev
        )
        self.scratch_a = torch.zeros(
            scratch_bytes(kattn_scratch_layout(S, ql)), dtype=torch.uint8, device=dev
        )
        self.scratch_k2 = torch.zeros(
            scratch_bytes(k2_scratch_layout()), dtype=torch.uint8, device=dev
        )
        self.scratch_k3 = torch.zeros(
            scratch_bytes(k3_scratch_layout()), dtype=torch.uint8, device=dev
        )
        assert scratch_bytes(k3_sym_layout(1)) <= sym_bytes(1), (
            "K3's symmetric region overlaps the flags"
        )
        self.peers = PeerBuffer(sym_bytes(1) + guard.FLAG_BYTES, None, 0, 1, dev)
        self.peers.bytes.zero_()

    def close(self):
        self.peers.close()


def _k1(launch, L, B, x, w, k_cache, v_cache, epoch, layer, use_swa, stream):
    launch(
        x.data_ptr(), w.data_ptr(), L.cos_sin.data_ptr(), L.positions.data_ptr(), L.slots.data_ptr(),
        L.swa_map.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), B.q.data_ptr(), B.k.data_ptr(),
        B.v.data_ptr(), B.g.data_ptr(), B.scratch_k1.data_ptr(), epoch.data_ptr(), layer, 1, use_swa, 1, 1,
        stream=stream,
    )  # fmt: skip


def _attn_args(L, B):
    return (
        B.q.data_ptr(), B.o.data_ptr(), L.k_cache.data_ptr(), L.v_cache.data_ptr(), L.table.data_ptr(),
        MAX_BLOCKS, L.table.numel() * 4, L.prefix.data_ptr(), 0, L.ql, L.window, 0, L.scale_bits,
        B.scratch_a.data_ptr(),
    )  # fmt: skip


def _k2_args(L, B):
    w = L.w2k
    return (
        B.g.data_ptr(), B.res.data_ptr(), w["w_o"].data_ptr(), w["alpha"].data_ptr(), w["beta"].data_ptr(),
        w["pre_w"].data_ptr(), w["w_hi"].data_ptr(), w["w_lo"].data_ptr(), w["bias"].data_ptr(),
        w["w_pool"].data_ptr(), w["xn_w"].data_ptr(), B.ids.data_ptr(), B.scores.data_ptr(), B.xin.data_ptr(),
        B.scratch_k2.data_ptr(),
    )  # fmt: skip


def _k3_args(L, B):
    w = L.w3k
    return (
        0, L.w13.data_ptr(), L.s13.data_ptr(), L.w2.data_ptr(), L.s2.data_ptr(), w["pen_w"].data_ptr(),
        w["pfw"].data_ptr(), w["alpha"].data_ptr(), w["beta"].data_ptr(), w["npw"].data_ptr(),
        w["w_unpool"].data_ptr(), B.attn_in.data_ptr(), B.scratch_k3.data_ptr(),
    )  # fmt: skip


def _k1_next_args(L, B):
    return (
        L.w_qkvg1.data_ptr(), L.cos_sin.data_ptr(), L.positions.data_ptr(), L.slots.data_ptr(),
        L.swa_map.data_ptr(), L.k_cache1.data_ptr(), L.v_cache1.data_ptr(), B.k.data_ptr(), B.v.data_ptr(),
        B.scratch_k1.data_ptr(), 1, 1, 1, 1,
    )  # fmt: skip


def run_split(K, L, B, epoch, stream):
    """The four launches: the attention, K2, K3 of the layer, the next layer's K1."""
    k1, ka, k2, k3 = K["k1"], K["ka"], K["k2"], K["k3"]
    ka(*_attn_args(L, B), epoch.data_ptr(), LAYER, stream=stream)
    k2(B.o.data_ptr(), *_k2_args(L, B), B.peers.local, B.peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, stream=stream)  # fmt: skip
    a3 = _k3_args(L, B)
    k3(B.xin.data_ptr(), B.ids.data_ptr(), B.scores.data_ptr(), a3[0], *a3[1:5], a3[5], B.res.data_ptr(), *a3[6:],
       B.peers.local, B.peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, 0, stream=stream)  # fmt: skip
    _k1(
        k1,
        L,
        B,
        B.attn_in,
        L.w_qkvg1,
        L.k_cache1,
        L.v_cache1,
        epoch,
        LAYER + 1,
        1,
        stream,
    )


def run_merged(K, L, B, epoch, stream):
    K["layer"](
        *_attn_args(L, B), *_k2_args(L, B), *_k3_args(L, B), *_k1_next_args(L, B),
        B.peers.local, B.peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, 0, stream=stream,
    )  # fmt: skip


def _kernels(S, ql, defer=False):
    from sgl_amd_mono.muse_spark.gfx950.build_k2 import K2Build, build_k2
    from sgl_amd_mono.muse_spark.gfx950.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx950.layer import (
        K1Build,
        KAttnBuild,
        LayerBuild,
        build_k1,
        build_kattn,
        build_layer,
    )

    return {
        "k1": build_k1(K1Build(tokens=S)),
        "ka": build_kattn(KAttnBuild(tokens=S, q_len=ql)),
        "k2": build_k2(K2Build(tokens=S, npes=1, fp8_pool=FP8_DENSE)),
        "k3": build_k3(K3Build(tokens=S, npes=1, fp8_unpool=FP8_DENSE)),
        "layer": build_layer(
            LayerBuild(
                tokens=S,
                q_len=ql,
                npes=1,
                next_k1=True,
                fp8_dense=FP8_DENSE,
                defer_k1=defer,
                ug_dyn=UG_DYN,
            )
        ),
    }


def _step(K, L, B, epoch, stream, merged):
    """One layer step of a path: the layer's K1 on the step's input, then the rest."""
    B.res.copy_(L.res0)
    _k1(K["k1"], L, B, L.x0, L.w_qkvg, L.k_cache, L.v_cache, epoch, LAYER, 0, stream)
    (run_merged if merged else run_split)(K, L, B, epoch, stream)
    epoch[0] += 1


def _same(a, b, what, tol=0.0):
    a, b = a.float(), b.float()
    assert not a.isnan().any() and not b.isnan().any(), f"{what}: NaN"
    err = (a - b).abs()
    exact = (a == b).float().mean().item()
    assert (err <= tol * b.abs() + tol).all(), (
        f"{what}: max difference {err.max().item():.5f}, {exact:.2%} exact"
    )
    return exact


@pytest.mark.parametrize("S,ql", [(8, 8), (16, 8), (32, 8), (64, 8)])
def test_the_merged_launch_matches_the_four(S, ql):
    from sgl_amd_mono.common import guard

    dev = torch.device("cuda")
    L = Layer(S, ql, dev, seed=40 + S)
    K = _kernels(S, ql)
    stream = torch.cuda.current_stream()
    epoch = _epoch(dev)
    outs = {}
    for merged in (False, True):
        B = Buffers(S, ql, dev)
        try:
            for _ in range(2):  # the second step's tags differ from the first's
                _step(K, L, B, epoch, stream, merged)
            torch.cuda.synchronize()
            assert (epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0).all(), (
                f"merged={merged}: a wait failed"
            )
            rows = L.swa_map[L.slots]
            outs[merged] = {
                "o": B.o[:S].clone(), "res": B.res.clone(), "ids": B.ids[:S].clone(), "scores": B.scores[:S].clone(),
                "xin": B.xin[:S].clone(), "attn_in": B.attn_in[:S].clone(), "q1": B.q[:S].clone(),
                "k1": B.k[:S].clone(), "v1": B.v[:S].clone(), "g1": B.g[:S].clone(),
                "K1 cache": L.k_cache1[rows].clone(), "V1 cache": L.v_cache1[rows].clone(),
            }  # fmt: skip
            L.k_cache1.zero_()
            L.v_cache1.zero_()
        finally:
            B.close()
    a, b = outs[False], outs[True]
    assert (a["ids"] == b["ids"]).all(), "the routing differs"
    report = []
    for name in (
        "o",
        "res",
        "scores",
        "xin",
        "attn_in",
        "q1",
        "k1",
        "v1",
        "g1",
        "K1 cache",
        "V1 cache",
    ):
        # the attention's partition merge order may differ by a bf16 ulp; everything after is the same code
        report.append(f"{name} {_same(b[name], a[name], name, tol=1 / 128):.1%}")
    print("merged vs four, exact share:", ", ".join(report))
    assert (b["K1 cache"].float() != 0).any(), (
        "the next layer's cache rows were not written"
    )


@pytest.mark.parametrize("S,ql", [(8, 8), (16, 8), (64, 8)])
def test_the_deferred_k1_matches_the_four(S, ql):
    """The merged launch whose next K1 folds the FFN boundary's norms itself (K2's
    c1 half, the unpooled rows, K3's scalars) against the four launches: the layer's
    outputs alike (the same code up to the K1), the next layer's q / k / v / gate and
    cache rows at row cosine 0.999 (the operand roundings moved: x is no longer one
    bf16 rounding of the normed row)."""
    from sgl_amd_mono.common import guard

    dev = torch.device("cuda")
    L = Layer(S, ql, dev, seed=50 + S)
    stream = torch.cuda.current_stream()
    epoch = _epoch(dev)
    outs = {}
    for merged in (False, True):
        K = _kernels(S, ql, defer=merged)
        B = Buffers(S, ql, dev)
        try:
            for _ in range(2):
                _step(K, L, B, epoch, stream, merged)
            torch.cuda.synchronize()
            assert (epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1] == 0).all(), (
                f"merged={merged}: a wait failed"
            )
            rows = L.swa_map[L.slots]
            outs[merged] = {
                "res": B.res.clone(), "ids": B.ids[:S].clone(), "xin": B.xin[:S].clone(), "q1": B.q[:S].clone(),
                "k1": B.k[:S].clone(), "v1": B.v[:S].clone(), "g1": B.g[:S].clone(),
                "K1 cache": L.k_cache1[rows].float().clone(), "V1 cache": L.v_cache1[rows].float().clone(),
            }  # fmt: skip
            L.k_cache1.zero_()
            L.v_cache1.zero_()
        finally:
            B.close()
    a, b = outs[False], outs[True]
    assert (a["ids"] == b["ids"]).all(), "the routing differs"
    _same(b["res"], a["res"], "res", tol=1 / 128)
    _same(b["xin"], a["xin"], "xin", tol=1 / 32)
    report = []
    for name in ("q1", "k1", "v1", "g1", "K1 cache", "V1 cache"):
        x, y = b[name].float().reshape(S, -1), a[name].float().reshape(S, -1)
        assert not x.isnan().any(), f"{name}: NaN"
        cos = torch.nn.functional.cosine_similarity(x, y, dim=-1).min().item()
        report.append(f"{name} {cos:.5f}")
        assert cos > (0.99 if "cache" in name else 0.999), (
            f"{name}: row cosine {cos:.5f}"
        )
    print("deferred K1 vs four, least row cosine:", ", ".join(report))


def _time(fn, iters=20):
    """Eager: the launches back to back (the CPU launch cost bounds it)."""
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _time_graph(fn, iters=20):
    """Graph replay of ``iters`` launches (the GPU time, as the server runs them)."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    """Step times of each launch and of the two forms at every served width:
    eager (bounded by the CPU launch cost) and graph replay (the GPU time)."""
    dev = torch.device("cuda")
    cur = torch.cuda.current_stream
    for S in (8, 16, 32, 64):
        ql = 8
        L = Layer(S, ql, dev, seed=40 + S)
        K = _kernels(S, ql)
        epoch = _epoch(dev, deadline_s=60.0)
        B = Buffers(S, ql, dev)
        try:

            def k1():
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
                    cur(),
                )
                epoch[0] += 1

            def ka():
                K["ka"](*_attn_args(L, B), epoch.data_ptr(), LAYER, stream=cur())
                epoch[0] += 1

            def k2():
                K["k2"](B.o.data_ptr(), *_k2_args(L, B), B.peers.local, B.peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, stream=cur()); epoch[0] += 1  # fmt: skip

            def k3():
                a3 = _k3_args(L, B)
                K["k3"](B.xin.data_ptr(), B.ids.data_ptr(), B.scores.data_ptr(), a3[0], *a3[1:5], a3[5], B.res.data_ptr(), *a3[6:],
                        B.peers.local, B.peers.addresses.data_ptr(), 0, epoch.data_ptr(), LAYER, 0, stream=cur()); epoch[0] += 1  # fmt: skip

            def split():
                _k1(K["k1"], L, B, L.x0, L.w_qkvg, L.k_cache, L.v_cache, epoch, LAYER, 0, cur()); run_split(K, L, B, epoch, cur()); epoch[0] += 1  # fmt: skip

            def merged():
                _k1(K["k1"], L, B, L.x0, L.w_qkvg, L.k_cache, L.v_cache, epoch, LAYER, 0, cur()); run_merged(K, L, B, epoch, cur()); epoch[0] += 1  # fmt: skip

            # a valid state for the standalone timings: one full step first
            B.res.copy_(L.res0)
            split()
            torch.cuda.synchronize()
            fns = (
                ("k1", k1),
                ("kattn", ka),
                ("k2", k2),
                ("k3", k3),
                ("k1+four", split),
                ("k1+merged", merged),
            )
            for mode, timer in (("eager", _time), ("graph", _time_graph)):
                t = {n: timer(f) for n, f in fns}
                print(
                    f"S={S:2d} {mode:5s}: "
                    + "  ".join(f"{n} {v * 1000:7.1f} us" for n, v in t.items())
                    + f"   (lens {LENS[: S // ql]})"
                )
        finally:
            B.close()
    # the attention at the server's shapes: the sliding layers (window 2047) and the full
    # layers over the AgentX context, graph replay
    from sgl_amd_mono.muse_spark.gfx950.layer import KAttnBuild, build_kattn

    for S, window, lens in (
        (8, 0, [95000]),
        (8, 2047, [95000]),
        (8, 2047, [2500]),
        (8, 0, [20000]),
        (16, 0, [95000, 60000]),
        (16, 2047, [95000, 60000]),
        (32, 0, [95000, 300, 17, 60000]),
        (32, 2047, [95000, 300, 17, 60000]),
        (64, 0, LENS),
        (64, 2047, LENS),
    ):
        ql = 8
        L = Layer(S, ql, dev, seed=50 + S, lens=lens, window=window)
        ka = build_kattn(KAttnBuild(tokens=S, q_len=ql))
        epoch = _epoch(dev, deadline_s=60.0)
        B = Buffers(S, ql, dev)
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
    sys.exit(main())
