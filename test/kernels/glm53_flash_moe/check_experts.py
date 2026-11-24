# SPDX-License-Identifier: Apache-2.0
"""Checks and times the GLM-5.3-Flash MoE experts launch (one GPU, synthetic MXFP4 weights, shared expert as 288):

- the front (``front``, shared slot) gives the sorted rows, a1 and its per-token scales;
- ``experts`` against aiter's ``fused_moe_2stages`` (stage 1, MXFP4 quant, stage 2) on the same front outputs:
  rel-L2 vs stock, next to stock's own run-to-run floor; run twice, and every counter back to 0 after each;
- times (CUDA graphs) of both on the same front outputs.

Usage: python3 check_experts.py [--rows 1,4,...] [--time-rows ...] [--json OUT]"""

import argparse
import json
import sys

import torch
from check_front import LIMIT, SCALE, expert_weights, graph_time, inputs

import sgl_amd_mono.glm53_flash_moe as fm
from sgl_amd_mono.glm53_flash_moe import experts as ex

NE = fm.E + 1
# bytes of a touched expert: w13 4 MB + w2 2 MB fp4 and their e8m0 scales
EXPERT_BYTES = (2 * 1024 * fm.D + fm.D * 1024) // 2 + (2 * 1024 * fm.D + fm.D * 1024) // 32
dev = torch.device("cuda")


def stock(x, f, ws, buf):
    """aiter's stage 1 / quant / stage 2 on the front's sorted rows (``moe_front_aiter``'s call), into ``buf``."""
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe_2stages
    from aiter.ops.flydsl.moe_common import GateMode

    w13, w2, s13, s2 = ws
    return fused_moe_2stages(
        x, w13, w2, fm.TOPK + 1, f["sorted_ids"], f["sorted_weights"], f["sorted_expert_ids"], f["num_valid_ids"],
        buf, True, fm.BM, activation=ActivationType.Silu, quant_type=QuantType.per_1x32, q_dtype_a=dtypes.fp4x2,
        q_dtype_w=w13.dtype, w1_scale=s13, w2_scale=s2, topk_ids=f["topk_ids"], topk_weights=f["topk_weights"],
        swiglu_limit=LIMIT, gate_mode=GateMode.SEPARATED.value, routing_num_experts=NE,
        prequantized_a1=f["a1"].view(dtypes.fp4x2), prequantized_a1_scale=f["a1_scale_sorted"].view(dtypes.fp8_e8m0),
    )  # fmt: skip


def ours(f, ws, buf, target):
    w13, w2, s13, s2 = ws
    return ex.experts(
        f["a1"], f["a1_scale"], f["sorted_ids"], f["sorted_weights"], f["sorted_expert_ids"], f["num_valid_ids"],
        w13, w2, s13, s2, buf, LIMIT, target,
    )  # fmt: skip


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def check(rows, ws):
    g = torch.Generator(device=dev).manual_seed(6)
    ok = True
    for m in rows:
        x, gw, bias = inputs(m, g)
        f = fm.front(x, gw, bias, SCALE, shared=True, sorted_scale=True)
        target = ex.tile_targets(f["sorted_ids"], f["num_valid_ids"], m)
        o_ref = stock(x, f, ws, torch.zeros(m, fm.D, dtype=torch.bfloat16, device=dev))
        o_ref2 = stock(x, f, ws, torch.zeros(m, fm.D, dtype=torch.bfloat16, device=dev))
        floor = rel(o_ref2, o_ref)
        msgs = []
        outs = []
        for _ in range(2):
            outs.append(ours(f, ws, torch.zeros(m, fm.D, dtype=torch.bfloat16, device=dev), target))
            torch.cuda.synchronize()
            left = int((ex.workspace(dev)[0] != 0).sum().item())
            if left:
                msgs.append(f"{left} counters not back to 0")
        err, err2 = rel(outs[0], o_ref), rel(outs[1], o_ref)
        if not torch.isfinite(outs[0].float()).all():
            msgs.append("non-finite output")
        # the G1 sums' order differs from stock's: a rounding may flip a bf16 / fp4 code, so a little above the floor
        bound = max(2 * floor, 5e-3)
        if max(err, err2) > bound:
            msgs.append(f"above {bound:.1e}")
        ok &= not msgs
        print(f"experts m={m}: rel-L2 vs stock {err:.2e} / {err2:.2e} (stock run-to-run {floor:.2e}, ours "
              f"run-to-run {rel(outs[1], outs[0]):.2e}) {'OK' if not msgs else '; '.join(msgs)}", flush=True)
    return ok


def time_all(rows, ws):
    g = torch.Generator(device=dev).manual_seed(7)
    res = {}
    for m in rows:
        x, gw, bias = inputs(m, g)
        f = fm.front(x, gw, bias, SCALE, shared=True, sorted_scale=True)
        target = ex.tile_targets(f["sorted_ids"], f["num_valid_ids"], m)
        b0 = torch.zeros(m, fm.D, dtype=torch.bfloat16, device=dev)
        b1 = torch.zeros_like(b0)
        t_ours = graph_time(lambda: ours(f, ws, b0, target), iters=100)
        t_stock = graph_time(lambda: stock(x, f, ws, b1), iters=100)
        left = int((ex.workspace(dev)[0] != 0).sum().item())
        blocks = int(f["num_valid_ids"][0].item()) // fm.BM
        touched = int(torch.unique(f["sorted_expert_ids"][:blocks]).numel())
        nbytes = touched * EXPERT_BYTES
        res[m] = dict(experts=t_ours, stock=t_stock, blocks=blocks, touched=touched, counters_left=left,
                      tbs=nbytes / t_ours / 1e6, stock_tbs=nbytes / t_stock / 1e6)
        print(f"time m={m} ({blocks} blocks, {touched} experts, {nbytes / 1e6:.0f} MB): experts {t_ours:.1f} us "
              f"{res[m]['tbs']:.2f} TB/s, stock stage 1 + quant + stage 2 {t_stock:.1f} us {res[m]['stock_tbs']:.2f} TB/s"
              f"{'' if not left else f' ({left} COUNTERS LEFT)'}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,4,8,16,32,64,128,256")
    ap.add_argument("--time-rows", default="1,4,8,16,32,64,128,256")
    ap.add_argument("--skip", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    skip = set(a.skip.split(","))
    ws = expert_weights(torch.Generator(device=dev).manual_seed(5), ne=NE)
    ok = check([int(v) for v in a.rows.split(",")], ws) if "check" not in skip else True
    res = time_all([int(v) for v in a.time_rows.split(",")], ws) if "time" not in skip else {}
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(ok=ok, times=res), fh, indent=1)
    print("ALL OK" if ok else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
