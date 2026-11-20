# SPDX-License-Identifier: Apache-2.0
"""The wide KDA attention launch against SGLang's own ReplaySSM verify kernels,
one GPU, at 1 .. 8 requests of 6 rows (``kda_wide_case.py``).

    python test/kernels/kimi_k3/check_kda_wide.py --copy ~/pe-k3/copy/<key> --layer 4

Per width, three comparisons:

* end to end: the launch's core_out, ring rows and verify windows against
  SGLang's path from the same x and pools;
* past in_proj: SGLang's kernels fed the launch's own bf16 in_proj rows
  (debug build) against the launch: windows and beta bit for bit, the rest
  within ``CLOSE_REL`` (see ``BITWISE``);
* in_proj alone: the launch's bf16 rows against SGLang's GEMM.

The committed state and conv state must be left as they were, and a request's
outputs must not depend on the width or its neighbours (``run_invariance``). Prints one JSON
line a width and exits non-zero when a bitwise expectation fails.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import kda_wide_case as kc  # noqa: E402

# past in_proj: bit for bit (windows, beta), and within a few ulps where a
# handful of elements a width differ: the gate through the f_b GEMM's order,
# k / v through the conv + SiLU (1 - 3 elements in 10^4 to 10^5, one or two
# bf16 ulps), the core through both and the output norm's order
BITWISE = ("ring_b", "window")
CLOSE = ("ring_k", "ring_v", "ring_g", "o", "core")
CLOSE_REL = 1e-4
E2E_REL = 1e-2


def run_width(R, w, args, device):
    S = R * kc.L
    gen = torch.Generator(device=device).manual_seed(1000 + R)
    x = (torch.randn(S, kc.H, generator=gen, device=device) * args.x_scale).to(kc.BF)
    base = kc.Pools(R, gen, device)
    stock = kc.Stock(w)
    wide = kc.Wide(w, S, device, debug=True)
    epoch = torch.ones(1, dtype=torch.int32, device=device)

    # end to end
    ps = base.clone()
    keep = {}
    core_s = stock(x, ps, keep)
    pw = base.clone()
    core_w = torch.empty(S, kc.PROJ, dtype=kc.BF, device=device)
    wide(x, pw, epoch, core_w)
    torch.cuda.synchronize()
    out = {"R": R, "rows": S}
    e2e = {"core": kc.cmp(core_w, core_s)}
    so, wo = ps.outputs(), pw.outputs()
    for k in so:
        e2e[k] = kc.cmp(wo[k], so[k])
    out["e2e"] = e2e
    out["inproj"] = kc.cmp(wide.dbg[0], keep["proj"])
    # both GEMMs against an fp32 product (one rounding to bf16)
    ref = (x.float() @ w["w_in"].float().t()).to(kc.BF)
    out["inproj_ref"] = {
        "wide": kc.cmp(wide.dbg[0], ref),
        "stock": kc.cmp(keep["proj"], ref),
    }

    # past in_proj: SGLang on the launch's rows
    pi = base.clone()
    keep2 = {}
    core_i = stock.attn(wide.dbg[0].clone(), pi, keep2)
    torch.cuda.synchronize()
    past = {"core": kc.cmp(core_w, core_i), "o": kc.cmp(wide.dbg[1], keep2["o"])}
    io = pi.outputs()
    for k in io:
        past[k] = kc.cmp(wo[k], io[k])
    out["past_inproj"] = past
    out["state_untouched"] = bool(
        torch.equal(pw.temporal, base.temporal) and torch.equal(pw.conv, base.conv)
    )
    # a second launch of the same step (next epoch) gives the same bits
    pw2 = base.clone()
    core_w2 = torch.empty_like(core_w)
    epoch += 1
    wide(x, pw2, epoch, core_w2)
    torch.cuda.synchronize()
    out["repeatable"] = bool(torch.equal(core_w2, core_w)) and all(
        torch.equal(a, b) for a, b in zip(pw2.ring, pw.ring)
    )
    ok = (
        out["state_untouched"]
        and out["repeatable"]
        and all(past[k]["equal"] for k in BITWISE)
        and all(past[k]["rel_l2"] <= CLOSE_REL for k in CLOSE)
        and e2e["core"]["rel_l2"] <= E2E_REL
    )
    out["ok"] = ok
    return out


def run_invariance(w, device, reqs=(1, 2, 4, 8)):
    """One request alone and at the first and last place among 1 / 3 / 7 others:
    its core rows, ring rows and verify windows must be the same bits."""
    L = kc.L
    gen = torch.Generator(device=device).manual_seed(77)
    xr = torch.randn(L, kc.H, generator=gen, device=device).to(kc.BF)
    st = torch.randn(kc.NH, kc.HD, kc.HD, generator=gen, device=device) * 0.05
    cv = (torch.randn(3, kc.QKV, generator=gen, device=device) * 0.5).to(kc.BF)
    ref, res = None, []
    for R in reqs:
        for pos in sorted({0, R - 1}):
            S = R * L
            g2 = torch.Generator(device=device).manual_seed(500 + 10 * R + pos)
            x = torch.randn(S, kc.H, generator=g2, device=device).to(kc.BF)
            x[pos * L : (pos + 1) * L] = xr
            pools = kc.Pools(R, g2, device)
            slot, row = int(pools.slots[pos]), int(pools.rows[pos])
            pools.temporal[slot] = st
            pools.conv[slot] = cv
            core = torch.empty(S, kc.PROJ, dtype=kc.BF, device=device)
            wide = kc.Wide(w, S, device)
            wide(x, pools, torch.ones(1, dtype=torch.int32, device=device), core)
            torch.cuda.synchronize()
            got = {
                "core": core[pos * L : (pos + 1) * L].clone(),
                "window": pools.window[row].clone(),
                **{n: t[slot, :, :L].clone() for n, t in zip(("ring_k", "ring_v", "ring_g", "ring_b"), pools.ring)},
            }
            if ref is None:
                ref = got
            same = {k: bool(torch.equal(v, ref[k])) for k, v in got.items()}
            res.append({"R": R, "pos": pos, "equal": all(same.values()), "parts": same})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", default="")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--reqs", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--x-scale", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if args.copy:
        w = kc.load_layer(args.copy, args.layer, device)
    else:
        w = kc.random_layer(torch.Generator(device=device).manual_seed(0), device)
    res, bad = [], 0
    for R in map(int, args.reqs.split(",")):
        r = run_width(R, w, args, device)
        res.append(r)
        bad += not r["ok"]
        print("KDA_WIDE_CHECK", json.dumps(r), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    inv = run_invariance(w, device)
    for r in inv:
        print("KDA_WIDE_INVARIANCE", json.dumps(r), flush=True)
        bad += not r["equal"]
    print("KDA_WIDE_CHECK_DONE", "fail" if bad else "pass", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
