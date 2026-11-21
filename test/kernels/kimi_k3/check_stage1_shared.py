# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's MoE with the shared chain fused (48-180 rows, 1 GPU): route_sort_shared + aiter's fused_moe with
stage1_shared, on random weights and routes.

    python3 check_stage1_shared.py OUT.json [--widths 48,96,168] [--push]

- against the separate launches (SGLang's situ_and_mul + its shared down GEMM, sgl-amd-mono's route_sort, aiter's
  stage 1): the SiTU output, the routed cast, the shared output and stage 1's output bit for bit, and the MoE output
  against stock's own A/A (bf16 atomics);
- ``--push``: the C18 push from the shared tiles against shared_push's standalone push of the same shared output,
  into 8 emulated peer buffers on this GPU, every rank: slots and flags bit for bit.

Entry points for a TP8 check (each rank, its own GPU):
- ``run_case(case, push=None)``: one MoE call on ``make_case(m, seed, device)``; ``push`` = (launches, peers, rank,
  epoch, slot), i.e. (``SharedPushAr.launches``, ``*SharedPushAr.push_args(layer)``); returns (latent, shared_out);
- ``bench_case(case, push=None)``: its per-call µs (a CUDA graph of 8 calls, median of 20 replays)."""

import argparse
import json
import os

os.environ.setdefault("AITER_SITUV2_A8W4", "1")
import torch

E, K, LAT, H, SH, RI = 896, 16, 3584, 7168, 768, 384  # RI: a routed expert's intermediate a rank
FRONT = 2 * SH + E + LAT


def make_case(m: int, seed: int, device) -> dict:
    """Random inputs of one MoE call at m rows: the fp32 front [gate_up | router logits | routed input], the shared
    down weight, the routed experts (MXFP4 in SGLang's K3 layout) and a zero bias."""
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    g = torch.Generator(device=device).manual_seed(seed)

    def u8(*shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, device=device, dtype=torch.int32).to(torch.uint8)

    fp4 = torch.float4_e2m1fn_x2
    w13 = shuffle_weight_a16w4(u8(E, 2 * RI, LAT // 2), 16, True).view(fp4)
    w2 = shuffle_weight_a16w4(u8(E, LAT, RI // 2), 16, False).view(fp4)
    w13.is_shuffled = w2.is_shuffled = True
    w13s = shuffle_scale_a16w4(u8(E, 2 * RI, LAT // 32, lo=118, hi=123).view(-1, LAT // 32), E, True)
    w2s = shuffle_scale_a16w4(u8(E, LAT, RI // 32, lo=118, hi=123).view(-1, RI // 32), E, False)
    front = torch.randn(m, FRONT, generator=g, device=device) * 0.5
    return dict(m=m, front=front, w_down=(torch.randn(H, SH, generator=g, device=device) * 0.02).to(torch.bfloat16),
                experts=(w13, w2, w13s, w2s), bias=torch.zeros(E, dtype=torch.float32, device=device))


def _views(case):
    f = case["front"]
    return f[:, : 2 * SH], f[:, 2 * SH: 2 * SH + E], f[:, 2 * SH + E:]


def _fused_moe(case, x, tw, tids, aux, latent):
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType, dtypes
    from aiter.ops.flydsl.moe_common import GateMode
    from aiter.presorted_moe import register_presorted_moe

    register_presorted_moe(aux)
    w13, w2, w13s, w2s = case["experts"]
    fm.fused_moe(x, w13, w2, tw, tids, None, ActivationType.Situv2, QuantType.per_1x32, False, w13s, w2s,
                 dtype=torch.bfloat16, gate_mode=GateMode.INTERLEAVE.value, beta=4.0, linear_beta=25.0,
                 quant_dtype_a=dtypes.fp8, output=latent)


def run_case(case, push=None):
    """One MoE call, the shared chain fused: (latent [m, 3584], shared_out [m, 7168]), both bf16."""
    from sgl_amd_mono.kimi_k3.gfx950 import route
    from sgl_amd_mono.kimi_k3.gfx950.moe_persist.stage1_shared import shared_down_in_stage1

    m = case["m"]
    gate_up, logits, routed = _views(case)
    dev = gate_up.device
    latent = torch.empty(m, LAT, dtype=torch.bfloat16, device=dev)
    shared = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    tw, tids, aux, act, x = route.route_sort_shared(logits, case["bias"], latent, gate_up, routed, 4.0, 25.0)
    with shared_down_in_stage1(act, case["w_down"], shared, push):
        _fused_moe(case, x, tw, tids, aux, latent)
    return latent, shared


def run_separate(case):
    """The same call as today's separate launches: (latent, shared_out, act, routed bf16)."""
    from sglang.kernels.ops.activation import situ_and_mul
    from sglang.srt.models.kimi_k3 import _k3_bf16_gemm

    from sgl_amd_mono.kimi_k3.gfx950 import route

    m = case["m"]
    gate_up, logits, routed = _views(case)
    dev = gate_up.device
    latent = torch.empty(m, LAT, dtype=torch.bfloat16, device=dev)
    shared = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    act = torch.empty(m, SH, dtype=torch.bfloat16, device=dev)
    x = torch.empty(m, LAT, dtype=torch.bfloat16, device=dev)
    situ_and_mul(gate_up, act, 4.0, 25.0, (routed, x))
    _k3_bf16_gemm(act, case["w_down"], out=shared)
    tw, tids, aux = route.route_sort(logits, case["bias"], latent)
    _fused_moe(case, x, tw, tids, aux, latent)
    return latent, shared, act, x


def check_push(case, launches=93, slot=5):
    """The fused push against shared_push's standalone push of the same shared output, every rank."""
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.kimi_k3.gfx950.stages import shared_push as S

    dev = case["front"].device
    bufs = [torch.empty(S.peer_bytes(), dtype=torch.uint8, device=dev) for _ in range(S.TP)]
    table = torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64, device=dev)
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    epoch[0] = 3
    standalone = S.build_push(launches)
    out = {}
    for rank in range(S.TP):
        for b in bufs:
            S.arm(b)
        _, shared = run_case(case, push=(launches, table, rank, epoch, slot))
        torch.cuda.synchronize()
        got = [b.clone() for b in bufs]
        for b in bufs:
            S.arm(b)
        standalone(shared.data_ptr(), case["m"], table.data_ptr(), rank, epoch.data_ptr(), slot,
                   stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        out[rank] = int(sum(int((g != b).sum()) for g, b in zip(got, bufs)))
        written = int(sum(int((g != 0).sum()) for g in got))
        out[f"{rank}.bytes_nonzero"] = written
    return out


def _graph_us(fn, reps=20, calls=8):
    import statistics

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(calls):
            fn()
    gr.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        gr.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / calls)
    return round(statistics.median(ts), 2)


def bench_case(case, push=None, reps=20):
    """Per-call µs of ``run_case(case, push)`` in a CUDA graph."""
    return _graph_us(lambda: run_case(case, push), reps=reps)


def time_push(case, launches=93, slot=5):
    """Per-call µs of the fused MoE call without and with the push (into emulated peers on this GPU: the peer
    writes land in this GPU's HBM, not over XGMI)."""
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.kimi_k3.gfx950.stages import shared_push as S

    dev = case["front"].device
    bufs = [torch.empty(S.peer_bytes(), dtype=torch.uint8, device=dev) for _ in range(S.TP)]
    for b in bufs:
        S.arm(b)
    table = torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64, device=dev)
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    a = _graph_us(lambda: run_case(case))
    b = _graph_us(lambda: run_case(case, push=(launches, table, 1, epoch, slot)))
    a2 = _graph_us(lambda: run_case(case))
    return dict(us_fused=a, us_fused_push=b, us_fused_aa=a2, d_push=round(b - (a + a2) / 2, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--widths", default="48,60,72,96,120,144,168,180")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    rl = lambda u, v: float((u.float() - v.float()).norm() / v.float().norm().clamp_min(1e-30))
    res = {}
    for m in [int(v) for v in args.widths.split(",")]:
        case = make_case(m, m, dev)
        l0, s0, a0, x0 = run_separate(case)
        l0b, _, _, _ = run_separate(case)
        l1, s1 = run_case(case)
        torch.cuda.synchronize()
        from sgl_amd_mono.kimi_k3.gfx950 import route

        gate_up, logits, routed = _views(case)
        _, _, _, a1, x1 = route.route_sort_shared(logits, case["bias"], torch.empty_like(l1), gate_up, routed, 4.0,
                                                  25.0)
        r = dict(act=bool(torch.equal(a0, a1)), cast=bool(torch.equal(x0, x1)), shared=bool(torch.equal(s0, s1)),
                 out_rel=rl(l1, l0), out_rel_aa=rl(l0b, l0))
        if args.push:
            r["push_mismatched_bytes"] = check_push(case)
            r.update(time_push(case))
        res[m] = r
        print("CHECK", m, json.dumps(r), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    ok = all(r["act"] and r["cast"] and r["shared"] for r in res.values()) and all(
        all(v == 0 for k, v in r.get("push_mismatched_bytes", {}).items() if isinstance(k, int))
        for r in res.values())
    print("CHECK_STAGE1_SHARED", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
