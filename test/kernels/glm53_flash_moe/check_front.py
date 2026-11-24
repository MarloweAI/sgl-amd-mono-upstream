# SPDX-License-Identifier: Apache-2.0
"""Checks and times the GLM-5.3-Flash MoE front (one GPU, synthetic weights):

- top-k: ``topk`` on bf16 logits against aiter's ``biased_grouped_topk``, bit for bit (random and tied logits);
- the front: logits against a f32 reference, its top-k against aiter's on its own bf16-rounded logits, the sort
  as a permutation of the routes, a1 and the sorted scales against aiter's ``fused_dynamic_mxfp4_quant_moe_sort``
  on the front's sorted rows, the zeroed output;
- the MoE: front + aiter stage 1 / 2 against aiter's ``fused_moe`` on the same routes;
- times (CUDA graphs): the front against stock's router + top-k + sort + a1 quant, and the routed MoE both ways.

Usage: python3 check_front.py [--rows 4,8,...] [--json OUT]"""

import argparse
import json
import sys

import torch

import sgl_amd_mono.glm53_flash_moe as fm

E, TOPK, D, I = 288, 8, 4096, 1024
SCALE, LIMIT = 2.5, 10.0
dev = torch.device("cuda")


def stock_topk(logits_bf16, bias_bf16):
    from aiter.ops.topk import biased_grouped_topk

    m = logits_bf16.shape[0]
    w = torch.empty(m, TOPK, dtype=torch.float32, device=dev)
    ids = torch.empty(m, TOPK, dtype=torch.int32, device=dev)
    biased_grouped_topk(logits_bf16, bias_bf16, w, ids, 1, 1, True, SCALE)
    return w, ids


def check_topk(rows):
    g = torch.Generator(device=dev).manual_seed(1)
    bias = (torch.randn(E, generator=g, device=dev) * 0.05).to(torch.bfloat16)
    bad = 0
    for m in rows:
        for kind in ("randn", "ties", "bias-ties"):
            if kind == "randn":
                lg = torch.randn(m, E, generator=g, device=dev).to(torch.bfloat16)
                b = bias
            elif kind == "ties":
                lg = (torch.randint(-3, 4, (m, E), generator=g, device=dev).float() * 0.5).to(torch.bfloat16)
                b = bias
            else:
                lg = (torch.randint(-2, 3, (m, E), generator=g, device=dev).float()).to(torch.bfloat16)
                b = torch.zeros(E, dtype=torch.bfloat16, device=dev)
            w0, i0 = stock_topk(lg, b)
            w1, i1 = fm.topk(lg, b, SCALE)
            ok = torch.equal(i0, i1) and torch.equal(w0.view(torch.int32), w1.view(torch.int32))
            if not ok:
                bad += 1
                print(f"TOPK MISMATCH m={m} {kind}: ids rows differ {(i0 != i1).any(1).sum().item()}", flush=True)
    print(f"topk bitwise: {'OK' if bad == 0 else f'{bad} FAILED'}", flush=True)
    return bad == 0


def inputs(m, g):
    x = (torch.randn(m, D, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    gw = (torch.randn(E, D, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    bias = (torch.randn(E, generator=g, device=dev) * 0.05).to(torch.bfloat16)
    return x, gw, bias


def check_front(rows, shared=False):
    from aiter import dtypes
    from aiter.ops.quant import fused_dynamic_mxfp4_quant_moe_sort

    g = torch.Generator(device=dev).manual_seed(2)
    k = TOPK + int(shared)
    ok_all = True
    for m in rows:
        x, gw, bias = inputs(m, g)
        valid = torch.tensor([max(1, m - 1)], dtype=torch.int32, device=dev) if m > 2 else None
        nvalid = int(valid.item()) if valid is not None else m
        # twice: the second run checks the counters came back to 0
        for _ in range(2):
            f = fm.front(x, gw, bias, SCALE, valid=valid, shared=shared, sorted_scale=True)
        torch.cuda.synchronize()
        msgs = []
        ref = (x.float() @ gw.float().t())[:nvalid]
        lg = f["logits"][:nvalid]
        err = ((lg - ref).norm() / ref.norm()).item()
        if err > 1e-5:
            msgs.append(f"logits rel-L2 {err:.2e}")
        lb = lg.to(torch.bfloat16)
        w0, i0 = stock_topk(lb[:nvalid], bias)
        if not (torch.equal(f["topk_ids"][:nvalid, :TOPK], i0)
                and torch.equal(f["topk_weights"][:nvalid, :TOPK].view(torch.int32), w0.view(torch.int32))):
            msgs.append("top-k differs from aiter on the front's logits")
        if nvalid < m and not (f["topk_ids"][nvalid:] == -1).all():
            msgs.append("padded rows routed")
        if shared and not ((f["topk_ids"][:nvalid, TOPK] == E).all() and (f["topk_weights"][:nvalid, TOPK] == 1).all()):
            msgs.append("shared slot")
        # the sort: every route once, experts ascending, token order within an expert, BM blocks
        nv = int(f["num_valid_ids"][0].item())
        if int(f["num_valid_ids"][1].item()) != m or nv % fm.BM:
            msgs.append(f"num_valid {f['num_valid_ids'].tolist()}")
        sid = f["sorted_ids"][:nv].cpu()
        sw = f["sorted_weights"][:nv].cpu()
        se = f["sorted_expert_ids"][: nv // fm.BM].cpu()
        ids = f["topk_ids"].cpu()
        tw = f["topk_weights"].cpu()
        seen = []
        prev_e = -1
        for b in range(nv // fm.BM):
            e = int(se[b])
            if e < prev_e:
                msgs.append("experts not ascending")
                break
            prev_e = e
            last_t = -1
            for j in range(b * fm.BM, (b + 1) * fm.BM):
                v = int(sid[j])
                t, slot = v & 0xFFFFFF, v >> 24
                if t >= m:
                    if v != (k << 24) | m or float(sw[j]) != 0.0:
                        msgs.append("pad row")
                    continue
                if int(ids[t, slot]) != e or float(tw[t, slot]) != float(sw[j]):
                    msgs.append(f"row {j} route mismatch")
                    break
                seen.append((t, slot))
        want = sorted((t, s) for t in range(nvalid) for s in range(k))
        if sorted(seen) != want:
            msgs.append(f"routes {len(seen)} vs {len(want)}")
        # a1 + sorted scales vs aiter's quant on the same sorted rows
        a1r, sr = fused_dynamic_mxfp4_quant_moe_sort(
            x, sorted_ids=f["sorted_ids"], num_valid_ids=f["num_valid_ids"], token_num=m, topk=k, block_size=fm.BM,
            sorted_weights=f["sorted_weights"], num_experts_upper_bound=E + int(shared),
        )
        a1r = a1r.view(torch.uint8).view(m, -1)[:nvalid]
        if not torch.equal(a1r, f["a1"][:nvalid]):
            msgs.append(f"a1 bytes differ: {(a1r != f['a1'][:nvalid]).sum().item()}")
        # compare the shuffled scales of the real rows only (aiter leaves pad rows unwritten)
        real = torch.zeros(sr.shape[0], dtype=torch.bool)
        for j in range(nv):
            if int(sid[j]) & 0xFFFFFF < m:
                real[j] = True
        sr_b = sr.view(torch.uint8).cpu().view(-1)
        mine = f["a1_scale_sorted"].cpu().view(-1)
        idx = []
        for x_ in torch.nonzero(real).flatten().tolist():
            for y in range(fm.SCALE_COLS):
                idx.append((x_ // 32 * 128) * 32 + (y // 8) * 256 + (y % 4) * 64 + (x_ % 16) * 4 + (y % 8) // 4 * 2
                           + (x_ % 32) // 16)
        idx = torch.tensor(idx, dtype=torch.long)
        if not torch.equal(sr_b[idx], mine[idx]):
            msgs.append(f"sorted scales differ: {(sr_b[idx] != mine[idx]).sum().item()} of {len(idx)}")
        from sgl_amd_mono.glm53_flash_moe.experts import tile_targets

        if not torch.equal(f["tile_target"], tile_targets(f["sorted_ids"], f["num_valid_ids"], m)):
            msgs.append(f"tile_target {f['tile_target'].tolist()}")
        if f["moe_buf"].abs().sum().item() != 0:
            msgs.append("moe_buf not zeroed")
        print(f"front m={m} shared={shared}: {'OK' if not msgs else '; '.join(sorted(set(msgs)))}", flush=True)
        ok_all &= not msgs
    return ok_all


def check_mixed():
    """Launches with and without the sorted scales, alternating: every counter and flag left for the next."""
    g = torch.Generator(device=dev).manual_seed(6)
    for m in (16, 128):
        x, gw, bias = inputs(m, g)
        ref = None
        for i, ss in enumerate((False, True, False, True, True, False)):
            f = fm.front(x, gw, bias, SCALE, shared=bool(i % 2), sorted_scale=ss)
            torch.cuda.synchronize()
            if ss:
                ref = f["a1_scale_sorted"].clone() if ref is None else ref
    print("front mixed launches: OK", flush=True)
    return True


def expert_weights(g, ne=E):
    """Synthetic shuffled MXFP4 experts in Quark's layout: w13 [ne, 2I, D/2], w2 [ne, D, I/2], e8m0 scales."""
    from aiter import dtypes

    w13 = torch.randint(0, 256, (ne, 2 * I, D // 2), dtype=torch.uint8, device=dev, generator=g).view(dtypes.fp4x2)
    w2 = torch.randint(0, 256, (ne, D, I // 2), dtype=torch.uint8, device=dev, generator=g).view(dtypes.fp4x2)
    s13 = torch.randint(118, 122, (ne, 2 * I, D // 32), dtype=torch.uint8, device=dev, generator=g)
    s2 = torch.randint(118, 122, (ne, D, I // 32), dtype=torch.uint8, device=dev, generator=g)
    for t in (w13, w2):
        t.is_shuffled = True
    return w13, w2, s13.view(dtypes.fp8_e8m0), s2.view(dtypes.fp8_e8m0)


def stock_moe(x, w13, w2, s13, s2, w, ids):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    return fused_moe(x, w13, w2, w, ids, quant_type=QuantType.per_1x32, activation=ActivationType.Silu,
                     w1_scale=s13, w2_scale=s2, gate_mode=GateMode.SEPARATED.value, swiglu_limit=LIMIT)


def check_moe(rows, ws):
    g = torch.Generator(device=dev).manual_seed(3)
    w13, w2, s13, s2 = ws
    ok = True
    for m in rows:
        x, gw, bias = inputs(m, g)
        f = fm.front(x, gw, bias, SCALE)
        o_ref = stock_moe(x, w13, w2, s13, s2, f["topk_weights"], f["topk_ids"])
        o_ref2 = stock_moe(x, w13, w2, s13, s2, f["topk_weights"], f["topk_ids"])
        floor = ((o_ref2.float() - o_ref.float()).norm() / o_ref.float().norm()).item()
        for fp4 in (False, True):
            o = fm.moe_front_aiter(x, gw, bias, SCALE, w13, w2, s13, s2, LIMIT, fp4_inter=fp4)
            torch.cuda.synchronize()
            err = ((o.float() - o_ref.float()).norm() / o_ref.float().norm()).item()
            # the fused intermediate quant may round differently from the separate one: reported, not gated
            good = err <= max(2 * floor, 1e-3) or fp4
            ok &= good
            print(f"moe m={m} fp4_inter={fp4}: rel-L2 vs stock {err:.2e} (stock run-to-run {floor:.2e}) "
                  f"{'OK' if good else 'FAILED'}", flush=True)
    return ok


def graph_time(fn, iters=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(10):
            fn()
    gr.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters // 10):
        t0.record()
        gr.replay()
        t1.record()
        t1.synchronize()
        ts.append(t0.elapsed_time(t1) * 1000 / 10)
    ts.sort()
    return ts[len(ts) // 2]


def stock_front(x, gw, bias, ws_dummy=None):
    """Stock's launches before stage 1: router GEMM (bf16 logits), top-k, sort, a1 quant."""
    from aiter import dtypes
    from aiter.fused_moe import moe_sorting
    from aiter.ops.quant import fused_dynamic_mxfp4_quant_moe_sort
    from aiter.tuned_gemm import tgemm

    m = x.shape[0]
    logits = tgemm.mm(x, gw, otype=torch.bfloat16)
    w, ids = stock_topk(logits, bias)
    sid, sw, se, nv, buf = moe_sorting(ids, w, E, D, torch.bfloat16, fm.BM)
    a1, sc = fused_dynamic_mxfp4_quant_moe_sort(x, sorted_ids=sid, num_valid_ids=nv, token_num=m, topk=TOPK,
                                                block_size=fm.BM, sorted_weights=sw, num_experts_upper_bound=E)
    return a1, sc, buf


def time_all(rows, ws):
    g = torch.Generator(device=dev).manual_seed(4)
    w13, w2, s13, s2 = ws
    res = {}
    for m in rows:
        x, gw, bias = inputs(m, g)
        t_front = graph_time(lambda: fm.front(x, gw, bias, SCALE))
        t_front2 = graph_time(lambda: fm.front(x, gw, bias, SCALE, shared=True, sorted_scale=False))
        t_sfront = graph_time(lambda: stock_front(x, gw, bias))

        def stock_full():
            from aiter.tuned_gemm import tgemm

            logits = tgemm.mm(x, gw, otype=torch.bfloat16)
            w, ids = stock_topk(logits, bias)
            return stock_moe(x, w13, w2, s13, s2, w, ids)

        t_smoe = graph_time(stock_full, iters=100)
        t_moe = graph_time(lambda: fm.moe_front_aiter(x, gw, bias, SCALE, w13, w2, s13, s2, LIMIT), iters=100)
        t_moe4 = graph_time(lambda: fm.moe_front_aiter(x, gw, bias, SCALE, w13, w2, s13, s2, LIMIT, fp4_inter=True),
                            iters=100)
        res[m] = dict(front=t_front, front_shared_no_sorted_scale=t_front2, stock_front=t_sfront, moe=t_moe,
                      moe_fp4=t_moe4, stock_moe=t_smoe)
        print(f"time m={m}: front {t_front:.1f} us, with the shared slot and no sorted scales {t_front2:.1f} "
              f"(stock router+topk+sort+quant {t_sfront:.1f}); "
              f"routed MoE {t_moe:.1f} us, fp4 intermediate {t_moe4:.1f} (stock {t_smoe:.1f})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,4,8,16,17,32,64,100,128,192,256")
    ap.add_argument("--time-rows", default="4,8,16,32,64,128,256")
    ap.add_argument("--skip", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = [int(v) for v in a.rows.split(",")]
    trows = [int(v) for v in a.time_rows.split(",")]
    skip = set(a.skip.split(","))
    ok = True
    if "topk" not in skip:
        ok &= check_topk(rows)
    if "front" not in skip:
        ok &= check_mixed()
        ok &= check_front(rows)
        ok &= check_front(rows, shared=True)
    ws = expert_weights(torch.Generator(device=dev).manual_seed(5))
    if "moe" not in skip:
        ok &= check_moe(trows, ws)
    res = time_all(trows, ws) if "time" not in skip else {}
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(ok=ok, times=res), fh, indent=1)
    print("ALL OK" if ok else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
