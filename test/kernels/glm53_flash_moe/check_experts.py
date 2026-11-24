# SPDX-License-Identifier: Apache-2.0
"""Checks and times the GLM-5.3-Flash MoE experts launch (one GPU, synthetic MXFP4 weights, the shared expert in its dense layout)
against stock's two paths:

- routed: the front without the shared slot, then aiter's ``fused_moe_2stages`` (stage 1, MXFP4 quant, stage 2);
- shared (SGLang's dense MLP at TP2): ``dynamic_mxfp4_quant`` (even scales), ``gemm_afp4wfp4`` -> bf16 gate/up,
  ``fused_clamp_act_mul`` (silu, clamp), ``dynamic_mxfp4_quant``, ``gemm_afp4wfp4`` -> bf16, on the shared expert's
  dense weights (row-major), which ``experts`` reads as slot 288 too;
- final = routed + shared (bf16 add).

``experts`` runs on the front with the shared slot: rel-L2 of its output vs stock's final, next to stock's own
run-to-run floor (the routed atomics' order); the shared rows alone (routes of slot 288 only) vs stock's shared
path; counters back to 0. Times (CUDA graphs) both ways.

Usage: python3 check_experts.py [--rows 1,4,...] [--time-rows ...] [--json OUT]"""

import argparse
import json
import sys

import torch
from check_front import LIMIT, SCALE, graph_time, inputs

import sgl_amd_mono.glm53_flash_moe as fm
from sgl_amd_mono.glm53_flash_moe import experts as ex

I = ex.I
dev = torch.device("cuda")
# bytes of a touched expert: w13 4 MB + w2 2 MB fp4 and their e8m0 scales
EXPERT_BYTES = (2 * I * fm.D + fm.D * I) // 2 + (2 * I * fm.D + fm.D * I) // 32


def weights(g):
    """Synthetic MXFP4 weights: the routed experts [288, ...] in Quark's shuffled layout, and the shared expert's
    dense gate_up / down as SGLang's linear layers hold them (row-major fp4 and e8m0)."""
    from aiter import dtypes

    u8 = torch.uint8
    w13 = torch.randint(0, 256, (fm.E, 2 * I, fm.D // 2), dtype=u8, device=dev, generator=g).view(dtypes.fp4x2)
    w2 = torch.randint(0, 256, (fm.E, fm.D, I // 2), dtype=u8, device=dev, generator=g).view(dtypes.fp4x2)
    s13 = torch.randint(118, 122, (fm.E, 2 * I, fm.D // 32), dtype=u8, device=dev, generator=g).view(dtypes.fp8_e8m0)
    s2 = torch.randint(118, 122, (fm.E, fm.D, I // 32), dtype=u8, device=dev, generator=g).view(dtypes.fp8_e8m0)
    for t in (w13, w2):
        t.is_shuffled = True
    raw = (
        torch.randint(0, 256, (2 * I, fm.D // 2), dtype=u8, device=dev, generator=g),
        torch.randint(118, 122, (2 * I, fm.D // 32), dtype=u8, device=dev, generator=g),
        torch.randint(0, 256, (fm.D, I // 2), dtype=u8, device=dev, generator=g),
        torch.randint(118, 122, (fm.D, I // 32), dtype=u8, device=dev, generator=g),
    )
    return (w13, w2, s13, s2), raw


def stock_routed(x, f, ws):
    """aiter's stage 1 / quant / stage 2 on the front's routed-only sort (``moe_front_aiter``'s call)."""
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe_2stages
    from aiter.ops.flydsl.moe_common import GateMode

    w13, w2, s13, s2 = ws
    return fused_moe_2stages(
        x, w13, w2, fm.TOPK, f["sorted_ids"], f["sorted_weights"], f["sorted_expert_ids"], f["num_valid_ids"],
        torch.zeros_like(f["moe_buf"]), True, fm.BM, activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        q_dtype_a=dtypes.fp4x2, q_dtype_w=w13.dtype, w1_scale=s13, w2_scale=s2, topk_ids=f["topk_ids"],
        topk_weights=f["topk_weights"], swiglu_limit=LIMIT, gate_mode=GateMode.SEPARATED.value,
        routing_num_experts=fm.E, prequantized_a1=f["a1"].view(dtypes.fp4x2),
        prequantized_a1_scale=f["a1_scale_sorted"].view(dtypes.fp8_e8m0),
    )  # fmt: skip


def stock_shared(x, raw):
    """SGLang's shared expert at TP2 (Quark MXFP4): quant, gate_up GEMM, clamped silu x up, quant, down GEMM (uint8
    fp4 / e8m0 as SGLang passes them)."""
    from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    w13, s13, w2, s2 = raw
    xq, xs = dynamic_mxfp4_quant(x)
    gu = gemm_afp4wfp4(xq, w13, xs, s13, torch.bfloat16)
    h = fused_clamp_act_mul(gu, swiglu_limit=LIMIT, activation="silu")
    hq, hs = dynamic_mxfp4_quant(h)
    return gemm_afp4wfp4(hq, w2, hs, s2, torch.bfloat16)


def stock(x, gw, bias, ws, raw, given=None):
    f = fm.front(x, gw, bias, SCALE, shared=False, sorted_scale=True, given=given)
    return (stock_routed(x, f, ws).float() + stock_shared(x, raw).float()).to(torch.bfloat16)


def ours(f, ws, raw, buf):
    w13, w2, s13, s2 = ws
    return ex.experts(
        f["a1"], f["a1_scale"], f["a1_even"], f["a1_even_scale"], f["sorted_ids"], f["sorted_weights"],
        f["sorted_expert_ids"], f["num_valid_ids"], w13, w2, s13, s2, *raw, buf, LIMIT, f["tile_target"],
    )  # fmt: skip


def shared_only(f, m):
    """The front's outputs re-sorted to slot 288 alone (one route a token, weight 1)."""
    blocks = (m + fm.BM - 1) // fm.BM
    cap = blocks * fm.BM
    rows = torch.arange(cap, dtype=torch.int32, device=dev)
    sid = torch.where(rows < m, rows | (fm.TOPK << 24), torch.full_like(rows, (fm.TOPK + 1) << 24 | m))
    g = dict(f)
    g["sorted_ids"] = sid
    g["sorted_weights"] = (rows < m).float()
    g["sorted_expert_ids"] = torch.full((blocks,), ex.SHARED, dtype=torch.int32, device=dev)
    g["num_valid_ids"] = torch.tensor([cap, m], dtype=torch.int32, device=dev)
    g["tile_target"] = ex.tile_targets(sid, g["num_valid_ids"], m)
    return g


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def counters_left():
    return int((ex.workspace(dev)[0] != 0).sum().item())


def check(rows, ws, raw):
    g = torch.Generator(device=dev).manual_seed(6)
    ok = True
    for m in rows:
        x, gw, bias = inputs(m, g)
        ref, ref2 = stock(x, gw, bias, ws, raw), stock(x, gw, bias, ws, raw)
        floor = rel(ref2, ref)
        msgs = []
        outs = []
        for _ in range(2):
            f = fm.front(x, gw, bias, SCALE, shared=True, sorted_scale=False)
            outs.append(ours(f, ws, raw, f["moe_buf"]).clone())
            torch.cuda.synchronize()
            if counters_left():
                msgs.append(f"{counters_left()} counters not back to 0")
        err, err2 = rel(outs[0], ref), rel(outs[1], ref)
        sh_ref = stock_shared(x, raw)
        g1 = shared_only(f, m)
        buf = torch.zeros(m, fm.D, dtype=torch.bfloat16, device=dev)
        sh = ours(g1, ws, raw, buf)
        torch.cuda.synchronize()
        err_sh = rel(sh, sh_ref)
        bits = (sh.view(torch.int16) != sh_ref.view(torch.int16)).float().mean().item()
        if not torch.isfinite(outs[0].float()).all():
            msgs.append("non-finite output")
        bound = max(2 * floor, 5e-3)
        if max(err, err2) > bound:
            msgs.append(f"above {bound:.1e}")
        if err_sh > 5e-3:
            msgs.append("shared rows")
        ok &= not msgs
        print(f"experts m={m}: rel-L2 vs stock {err:.2e} / {err2:.2e} (stock run-to-run {floor:.2e}, ours run-to-run "
              f"{rel(outs[1], outs[0]):.2e}); shared rows alone vs stock's shared path {err_sh:.2e} "
              f"({bits:.1%} of bf16 values differ) {'OK' if not msgs else '; '.join(msgs)}", flush=True)
    return ok


def time_all(rows, ws, raw):
    g = torch.Generator(device=dev).manual_seed(7)
    res = {}
    for m in rows:
        x, gw, bias = inputs(m, g)
        f = fm.front(x, gw, bias, SCALE, shared=True, sorted_scale=False)
        fs = fm.front(x, gw, bias, SCALE, shared=False, sorted_scale=True)
        b0 = torch.zeros(m, fm.D, dtype=torch.bfloat16, device=dev)
        t_ours = graph_time(lambda: ours(f, ws, raw, b0), iters=100)
        t_stock = graph_time(lambda: (stock_routed(x, fs, ws), stock_shared(x, raw)), iters=100)
        blocks = int(f["num_valid_ids"][0].item()) // fm.BM
        touched = int(torch.unique(f["sorted_expert_ids"][:blocks]).numel())
        nbytes = touched * EXPERT_BYTES
        res[m] = dict(experts=t_ours, stock=t_stock, blocks=blocks, touched=touched, counters_left=counters_left(),
                      tbs=nbytes / t_ours / 1e6, stock_tbs=nbytes / t_stock / 1e6)
        print(f"time m={m} ({blocks} blocks, {touched} experts, {nbytes / 1e6:.0f} MB): experts {t_ours:.1f} us "
              f"{res[m]['tbs']:.2f} TB/s, stock routed + shared {t_stock:.1f} us {res[m]['stock_tbs']:.2f} TB/s"
              f"{'' if not res[m]['counters_left'] else ' (COUNTERS LEFT)'}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,4,8,16,32,64,128,256")
    ap.add_argument("--time-rows", default="1,4,8,16,32,64,128,256")
    ap.add_argument("--skip", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    skip = set(a.skip.split(","))
    ws, raw = weights(torch.Generator(device=dev).manual_seed(5))
    ok = check([int(v) for v in a.rows.split(",")], ws, raw) if "check" not in skip else True
    res = time_all([int(v) for v in a.time_rows.split(",")], ws, raw) if "time" not in skip else {}
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(ok=ok, times=res), fh, indent=1)
    print("ALL OK" if ok else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
