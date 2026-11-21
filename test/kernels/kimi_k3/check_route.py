# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's one-launch routing (``kimi_k3.gfx950.route``) against what it replaces, on one GPU: aiter's top-k
(``biased_grouped_topk``) and the BM = 16 aux sort aiter's fused_moe runs on its routes, on the same logits.

    python3 check_route.py OUT.json [--logits DUMP.pt] [--moe] [--time] [--widths 25,30,48]

- bits: top-k ids / weights bit for bit; the sort's num_valid and expert ids bit for bit, its rows bit for bit from
  75 rows (Opus places an expert's routes in token order) and as a set an expert below (aiter's one-CTA sort places
  them in LDS-atomic order, which varies run to run: counted on two stock runs), reverse_sorted pointing at its
  route; the stage-2 output zeroed. Random logits (normal, and coarse ones with many exact ties), plus ``--logits``
  (a dict {name: [m, 896] f32 logits, "bias:<name>": [896] f32}).
- ``--moe``: aiter's a8w4 2-stage MoE (SiTU v2, MXFP8 activations, random MXFP4 experts) on the stock routing and on
  the launch's: stage 1's output bit for bit by route, stage 2's output against two stock runs (its bf16 atomics).
- ``--time``: µs a call in CUDA graphs, stock's two (three) launches against the one.
"""

import argparse
import json
import os
import time

os.environ.setdefault("AITER_SITUV2_A8W4", "1")

import torch

from sgl_amd_mono.kimi_k3.gfx950 import route

E, K, LAT = route.E, route.TOPK, route.LAT
WIDTHS = [25, 30, 32, 36, 42, 47, 48, 54, 60, 66, 72, 78, 96, 120, 144, 168, 174, 180]


def stock(logits, bias, out, valid=None):
    """aiter's top-k (SGLang's padded-row mask after it, ``valid`` rows), then fused_moe's sort."""
    import aiter
    import aiter.fused_moe as fm

    m = logits.shape[0]
    w = torch.empty(m, K, dtype=torch.float32, device=logits.device)
    ids = torch.empty(m, K, dtype=torch.int32, device=logits.device)
    aiter.biased_grouped_topk(logits, bias, w, ids, 1, 1, True, 1.0)
    if valid is not None:
        ids[valid:] = -1
        w[valid:] = 0.0
    r = fm.moe_sorting(ids, w, E, LAT, torch.bfloat16, 16, None, None, return_local_topk_ids=False,
                       accumulate=True, output_aux=fm.AUX_SORT_OPUS, output=out)
    st, sw, se, nv, buf, mi, rs = r[:7]
    return w, ids, dict(sorted_token_ids=st, sorted_weights=sw, sorted_expert_ids=se, num_valid_ids=nv,
                        m_indices=mi, reverse_sorted=rs), buf


def bits(t):
    return t.contiguous().view(torch.int32)


def rows(aux, nv, blk_e):
    st = aux["sorted_token_ids"][:nv].long() & 0xFFFFFFFF
    sw = bits(aux["sorted_weights"][:nv]).long() & 0xFFFFFFFF
    mi = aux["m_indices"][:nv].long()
    r = torch.stack([blk_e, st, sw, mi], 1)
    return r[torch.argsort(r[:, 0] * (1 << 33) + r[:, 1], stable=True)]


def compare(logits, bias, valid=None):
    """One case: mismatch counts (0 everywhere = bit for bit)."""
    m = logits.shape[0]
    dev = logits.device
    out_s = torch.full((m, LAT), float("nan"), dtype=torch.bfloat16, device=dev)
    out_m = torch.full((m, LAT), float("nan"), dtype=torch.bfloat16, device=dev)
    w_s, ids_s, aux_s, buf = stock(logits, bias, out_s, valid)
    _, _, aux_s2, _ = stock(logits, bias, torch.empty_like(out_s), valid)
    vt = None if valid is None else torch.tensor([valid], dtype=torch.int32, device=dev)
    w_m, ids_m, aux_m = route.route_sort(logits, bias, out_m, valid=vt)
    torch.cuda.synchronize()
    nv = int(aux_s["num_valid_ids"][0])
    nb = nv // 16
    r = dict(m=m, valid=valid, nv=nv, opus=m >= route.OPUS_FROM_ROWS,
             ids=int((ids_m != ids_s).sum()), weights=int((bits(w_m) != bits(w_s)).sum()),
             num_valid=int((aux_m["num_valid_ids"] != aux_s["num_valid_ids"]).sum()),
             eids=int((aux_m["sorted_expert_ids"][:nb] != aux_s["sorted_expert_ids"][:nb]).sum()),
             sizes=int(any(aux_m[k].numel() != aux_s[k].numel() for k in aux_s)),
             zeroed=int((out_m.view(torch.int16) != 0).sum()),
             stock_zeroed=int((buf.view(torch.int16) != 0).sum()))
    exact = {}
    for k in ("sorted_token_ids", "sorted_weights", "m_indices"):
        exact[k] = int((bits(aux_m[k][:nv]) != bits(aux_s[k][:nv])).sum())
        exact[k + "_stock_aa"] = int((bits(aux_s2[k][:nv]) != bits(aux_s[k][:nv])).sum())
    exact["reverse_sorted"] = int((aux_m["reverse_sorted"] != aux_s["reverse_sorted"]).sum())
    exact["reverse_sorted_stock_aa"] = int((aux_s2["reverse_sorted"] != aux_s["reverse_sorted"]).sum())
    r["exact"] = exact
    blk_e = aux_s["sorted_expert_ids"][:nb].long().repeat_interleave(16)
    r["rows_as_set"] = int((rows(aux_m, nv, blk_e) != rows(aux_s, nv, blk_e)).any(1).sum())
    # reverse_sorted: route (t, s) -> its row, which holds it
    rs = aux_m["reverse_sorted"].long()
    p = torch.arange(m * K, device=dev)
    t, s = p // K, p % K
    routed = ids_m.view(-1) >= 0
    ok = (rs >= 0) & (rs < nv)
    rc = rs.clamp(0, nv - 1)
    bad = ~ok | (aux_m["sorted_token_ids"][rc].long() != ((s << 24) | t)) | (aux_m["m_indices"][rc].long() != t) \
        | (bits(aux_m["sorted_weights"][rc]) != bits(w_m).view(-1))
    r["reverse_points"] = int((bad & routed).sum() + ((rs != -1) & ~routed).sum())
    bad = [r[k] for k in ("ids", "weights", "num_valid", "eids", "sizes", "zeroed", "rows_as_set", "reverse_points")]
    if r["opus"]:
        bad += [exact[k] for k in ("sorted_token_ids", "sorted_weights", "m_indices", "reverse_sorted")]
    r["pass"] = not any(bad)
    return r


def cases(gen, dev, m):
    yield "normal", torch.randn(m, E, generator=gen, device=dev) * 2.0
    # coarse: many exact sigmoid + bias ties, so the tie rank decides
    yield "coarse", torch.randint(-4, 5, (m, E), generator=gen, device=dev).float()
    x = torch.randn(m, E, generator=gen, device=dev)
    x[: m // 2] = 0.5  # rows whose scores are the bias alone
    yield "flat", x


def check_bits(args, dev):
    gen = torch.Generator(device=dev).manual_seed(0)
    res = []
    for m in args.widths or WIDTHS:
        for seed in range(args.seeds):
            bias = torch.randn(E, generator=gen, device=dev) * 0.05
            for name, lg in cases(gen, dev, m):
                # a row-strided view, as the merged front hands the router logits over
                full = torch.randn(m, 6016, generator=gen, device=dev)
                full[:, 1536 : 1536 + E] = lg
                for b in (bias, torch.zeros_like(bias)):
                    res.append(dict(compare(full[:, 1536 : 1536 + E], b), case=name, seed=seed,
                                    zero_bias=bool((b == 0).all())))
                # a padded graph batch: the last request's 6 rows masked
                res.append(dict(compare(full[:, 1536 : 1536 + E], bias, valid=m - 6), case=name + "-padded",
                                seed=seed))
    if args.logits:
        d = torch.load(args.logits, map_location=dev)
        for name, lg in d.items():
            if name.startswith("bias:"):
                continue
            res.append(dict(compare(lg.float(), d["bias:" + name].float()), case=name))
    return res


def moe_weights(gen, dev):
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    ri = 384

    def u8(*shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=gen, device=dev, dtype=torch.int32).to(torch.uint8)

    fp4 = torch.float4_e2m1fn_x2
    w13 = shuffle_weight_a16w4(u8(E, 2 * ri, LAT // 2), 16, True).view(fp4)
    w2 = shuffle_weight_a16w4(u8(E, LAT, ri // 2), 16, False).view(fp4)
    w13.is_shuffled = w2.is_shuffled = True
    w13s = shuffle_scale_a16w4(u8(E, 2 * ri, LAT // 32, lo=118, hi=123).view(-1, LAT // 32), E, True)
    w2s = shuffle_scale_a16w4(u8(E, LAT, ri // 32, lo=118, hi=123).view(-1, ri // 32), E, False)
    return w13, w2, w13s, w2s


def run_moe(x, w, ids, wts, out, aux=None, keep=None):
    """aiter's fused_moe as SGLang's a8w4 runner calls it; ``keep`` gets stage 1's output."""
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType, dtypes
    from aiter.ops.flydsl.moe_common import GateMode
    from aiter.presorted_moe import register_presorted_moe

    def over(ordinary_stage2, stage2_args, stage2_kwargs):
        keep["a2"] = stage2_args[0].clone()
        keep["a2_scale"] = None if stage2_kwargs.get("a2_scale") is None else stage2_kwargs["a2_scale"].clone()
        keep["sorted_ids"] = stage2_args[3].clone()
        ordinary_stage2(*stage2_args, **stage2_kwargs)
        return stage2_args[6]

    w13, w2, w13s, w2s = w
    if aux is not None:
        register_presorted_moe(aux)
    return fm._fused_moe_impl(x, w13, w2, wts, ids, None, ActivationType.Situv2.value, QuantType.per_1x32.value,
                              False, w13s, w2s, None, None, dtype=torch.bfloat16,
                              gate_mode=GateMode.INTERLEAVE.value, beta=4.0, linear_beta=25.0, output=out,
                              quant_dtype_a=dtypes.fp8, _stage2_override=over if keep is not None else None)


def by_route(keep, m):
    """Stage 1's output rows keyed by route (token, slot), from the sorted ids the stages ran on."""
    a2 = keep["a2"]
    st = keep["sorted_ids"].long()
    if a2.dim() == 3 and a2.shape[0] == m:  # already [token, slot, inter]
        return a2.reshape(m * K, -1)
    flat = a2.reshape(a2.shape[0], -1) if a2.dim() >= 2 else a2
    n = min(flat.shape[0], st.numel())
    tok, slot = st[:n] & 0xFFFFFF, st[:n] >> 24
    real = tok < m
    outp = torch.zeros(m * K, flat.shape[1], dtype=flat.dtype, device=flat.device)
    outp[(tok * K + slot)[real]] = flat[:n][real]
    return outp


def check_moe(args, dev):
    import aiter.fused_moe as fm

    used = {}
    cur = fm.match_presorted_moe

    def match(aux, *a, **kw):
        got = cur(aux, *a, **kw)
        used["last"] = got is not None
        return got

    fm.match_presorted_moe = match
    gen = torch.Generator(device=dev).manual_seed(1)
    w = moe_weights(gen, dev)
    res = []
    for m in args.widths or [25, 32, 42, 48, 72, 96, 120, 168, 180]:
        x = (torch.randn(m, LAT, generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        logits = torch.randn(m, E, generator=gen, device=dev) * 2.0
        bias = torch.randn(E, generator=gen, device=dev) * 0.05
        outs, keeps = [], []
        for _ in range(2):
            w_s = torch.empty(m, K, dtype=torch.float32, device=dev)
            ids_s = torch.empty(m, K, dtype=torch.int32, device=dev)
            import aiter

            aiter.biased_grouped_topk(logits, bias, w_s, ids_s, 1, 1, True, 1.0)
            o = torch.empty(m, LAT, dtype=torch.bfloat16, device=dev)
            kp = {}
            used["last"] = None
            run_moe(x, w, ids_s, w_s, o, keep=kp)
            outs.append(o)
            keeps.append(kp)
        o_m = torch.full((m, LAT), float("nan"), dtype=torch.bfloat16, device=dev)
        w_m, ids_m, aux = route.route_sort(logits, bias, o_m)
        kp_m = {}
        used["last"] = None
        run_moe(x, w, ids_m, w_m, o_m, aux=aux, keep=kp_m)
        torch.cuda.synchronize()
        a_s, a_s2, a_m = (by_route(k, m) for k in (keeps[0], keeps[1], kp_m))
        d = lambda a, b: float((a.float() - b.float()).abs().max())
        res.append(dict(m=m, presort_used=used["last"], a2_shape=list(kp_m["a2"].shape),
                        stage1_mismatch=int((a_m.view(torch.uint8) != a_s.view(torch.uint8)).sum()),
                        stage1_stock_aa=int((a_s2.view(torch.uint8) != a_s.view(torch.uint8)).sum()),
                        out_maxdiff=d(o_m, outs[0]), out_stock_aa=d(outs[1], outs[0]),
                        out_rel=float((o_m.float() - outs[0].float()).norm() / outs[0].float().norm()),
                        out_rel_aa=float((outs[1].float() - outs[0].float()).norm() / outs[0].float().norm()),
                        finite=bool(torch.isfinite(o_m.float()).all())))
    fm.match_presorted_moe = cur
    return res


def graph_us(fn, reps=200, inner=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps // 20):
        a.record()
        for _ in range(20):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / (20 * inner))
    ts.sort()
    return ts[len(ts) // 2]


def check_time(args, dev):
    gen = torch.Generator(device=dev).manual_seed(2)
    res = []
    for m in args.widths or [25, 30, 36, 42, 48, 60, 72, 96, 120, 144, 168, 180]:
        full = torch.randn(m, 6016, generator=gen, device=dev) * 2.0
        lg = full[:, 1536 : 1536 + E]
        bias = torch.randn(E, generator=gen, device=dev) * 0.05
        out = torch.empty(m, LAT, dtype=torch.bfloat16, device=dev)
        route.route_sort(lg, bias, out)  # build, counter made outside the graph
        # every score tied (one logit, no bias): the tie rank alone orders the routes
        tied = torch.full((m, E), 0.5, device=dev)
        zb = torch.zeros_like(bias)
        res.append(dict(m=m, stock_us=graph_us(lambda: stock(lg, bias, out)),
                        fused_us=graph_us(lambda: route.route_sort(lg, bias, out)),
                        stock_tied_us=graph_us(lambda: stock(tied, zb, out)),
                        fused_tied_us=graph_us(lambda: route.route_sort(tied, zb, out))))
        print(json.dumps(res[-1]), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--logits")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--widths", type=lambda v: [int(x) for x in v.split(",")], help="rows, comma separated")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    t0 = time.time()
    route.ops()
    res = dict(build_s=time.time() - t0)
    res["bits"] = check_bits(args, dev)
    fails = [r for r in res["bits"] if not r["pass"]]
    print(f"bits: {len(res['bits']) - len(fails)}/{len(res['bits'])} pass", flush=True)
    for r in fails[:10]:
        print("FAIL", json.dumps(r), flush=True)
    if args.moe:
        res["moe"] = check_moe(args, dev)
        for r in res["moe"]:
            print("MOE", json.dumps(r), flush=True)
    if args.time:
        res["time"] = check_time(args, dev)
    json.dump(res, open(args.out, "w"), indent=1)
    print("ROUTE_CHECK_DONE", "PASS" if not fails else "FAIL", flush=True)


if __name__ == "__main__":
    main()
