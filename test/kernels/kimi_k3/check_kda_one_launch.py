# SPDX-License-Identifier: Apache-2.0
"""A KDA layer as one launch (``gfx950/kda_layer.py``: K1's wide form, its seam
front included, and K2 in one kernel) against its two launches (``kda_wide``
with the front, then ``layer.k2_launch``), on one GPU, one rank's real weights.

Run on a loopback tree (perf-engine ``plugin_tree.py make --model kimi_k3``),
whose peer pushes write every rank's slot of this rank's own buffer, so one
rank's K2 finds its all-reduces complete.

Per width (1 / 2 / 4 / 8 requests of 6 rows): the layer output, K2's updated
prefix, the core, x, the ring rows and the verify windows bit for bit against
the two launches, and the one launch run to run. Then batch invariance of the
one launch: one request alone and first / last among 1 / 3 / 7 others gives the
same output and prefix rows, core, x, ring rows and windows. ``--time``: each
form as a CUDA graph, distinct inputs a replay set, median us.

    python3 check_kda_one_launch.py --tree TREE --copy ~/pe-k3/copy/<key> --layer 4
"""

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import kda_wide_case as kc  # noqa: E402
from check_k2_loopback import DEADLINE_S, load_tree, weights  # noqa: E402
from check_kda_layer import Front, seam_weights  # noqa: E402

NB = 3
LAYER = 4  # the attention's tag slot; K2's is 128 + it


class Layer:
    """One width's two forms on the same weights and the runner's scratch."""

    def __init__(self, R, w, sw, k2w, scratch, addresses, epoch, device, timeline=False):
        from sgl_amd_mono.kimi_k3.gfx950 import kda_layer as kl
        from sgl_amd_mono.kimi_k3.gfx950 import layer as k2

        self.k2mod, self.kl = k2, kl
        self.S = R * kc.L
        self.front = Front(w, sw, self.S, NB, -1, device, layer=LAYER, timeline=timeline)
        self.w, self.sw, self.k2w = w, sw, k2w
        self.scratch, self.addresses, self.epoch = scratch, addresses, epoch
        self.key = k2.K2Build(
            tokens=self.S, nblocks=NB, eps=k2w["eps"], out_eps=k2w["out_eps"],
            guint=True, opt=k2.K2_OPT, timeline=timeline,
        )  # fmt: skip
        self.k2_tl = (
            torch.zeros(k2.BLOCKS, k2.TL_POINTS, dtype=torch.int64, device=device)
            if timeline
            else None
        )

    def attn_kwargs(self, stream, blocks, pools):
        f, w = self.front, self.w
        return dict(
            x=f.x, w_in=f.w_in, w_fb=w["w_fb"], conv_w=w["conv_w"],
            conv_state=pools.conv.mT, a_log=w["a_log"], dt_bias=w["dt_bias"],
            on_w=w["on_w"], rstate=pools.temporal, st_idx=pools.slots,
            conv_win=pools.window, win_row=pools.rows, ring=pools.ring,
            layer=LAYER, epoch=self.epoch, prefix=stream, blocks=blocks, tl=f.tl, **self.sw,
        )  # fmt: skip

    def run_k2(self, core, kprefix, stream, blocks, out, attn=None):
        w = self.k2w
        self.k2mod.k2_launch(
            self.key, core=core, w_o=w["w_o"], prefix=kprefix, blocks=blocks,
            ares_nw=w["ares_nw"], ares_qk=w["ares_qk"], in_nw=w["in_nw"],
            w_gate=w["w_gate"], bias=w["bias"], w_ld=w["w_ld"], w_sgu=w["w_sgu"],
            w_sd=w["w_sd"], w13=w["w13"], w13s=w["w13s"], w2=w["w2"], w2s=w["w2s"],
            ln_w=w["ln_w"], w_up=w["w_up_shard"], out=out, scratch=self.scratch,
            peers=self.addresses, rank=0, epoch=self.epoch, layer=128 + LAYER,
            prefix_in=stream, tl=self.k2_tl, attn=attn,
        )  # fmt: skip

    def two(self, stream, blocks, pools, kprefix, out, core):
        """kda_wide with its front, then K2; ``core`` the hand-off buffer."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

        self.epoch[0].add_(1)
        kw.kda_wide(
            self.front.key, core_out=core, scratch=self.front.scratch,
            **self.attn_kwargs(stream, blocks, pools),
        )  # fmt: skip
        self.run_k2(core, kprefix, stream, blocks, out)
        return core

    def one(self, stream, blocks, pools, kprefix, out):
        """The one launch; returns its core (in the runner's scratch)."""
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

        kl, akey = self.kl, self.front.key
        self.epoch[0].add_(1)
        core = kl.core_view(akey, self.scratch)
        vals = kw.launch_values(
            akey, core_out=core, scratch=kl.attn_scratch(akey, self.scratch),
            **self.attn_kwargs(stream, blocks, pools),
        )  # fmt: skip
        self.run_k2(core, kprefix, stream, blocks, out, attn=kl.Attn(key=akey, vals=vals))
        return core


def inputs(S, gen, device):
    stream = (torch.randn(S, kc.H, generator=gen, device=device) * 0.3).to(kc.BF)
    blocks = (torch.randn(S, NB + 1, kc.H, generator=gen, device=device) * 0.3).to(kc.BF)
    return stream, blocks


def snap(lay, pools, core, out, kprefix):
    o = {"out": out.clone(), "prefix": kprefix.clone(), "core": core.clone()}
    o["x"] = lay.front.x_rows(lay.S).clone()
    o.update({k: v.clone() for k, v in pools.outputs().items()})
    return o


def run_case(R, w, sw, k2w, env, device):
    S = R * kc.L
    gen = torch.Generator(device=device).manual_seed(4000 + R)
    stream, blocks = inputs(S, gen, device)
    base = kc.Pools(R, gen, device)
    lay = Layer(R, w, sw, k2w, *env, device)
    res = {"R": R, "rows": S}
    got = {}
    for form in ("two", "one", "one_again"):
        pools, kp = base.clone(), torch.zeros(S, kc.H, dtype=kc.BF, device=device)
        out = torch.empty_like(kp)
        if form == "two":
            core = lay.two(stream, blocks, pools, kp, out, torch.empty(S, kc.PROJ, dtype=kc.BF, device=device))
        else:
            core = lay.one(stream, blocks, pools, kp, out)
        torch.cuda.synchronize()
        got[form] = snap(lay, pools, core, out, kp)
    res["vs_two"] = {k: bool(torch.equal(v, got["two"][k])) for k, v in got["one"].items()}
    res["repeatable"] = all(torch.equal(v, got["one"][k]) for k, v in got["one_again"].items())
    res["out_vs_two"] = kc.cmp(got["one"]["out"], got["two"]["out"])
    res["ok"] = all(res["vs_two"].values()) and res["repeatable"]
    return res


def run_invariance(w, sw, k2w, env, device, reqs=(1, 2, 4, 8)):
    L = kc.L
    gen = torch.Generator(device=device).manual_seed(79)
    sr, br = inputs(L, gen, device)
    st = torch.randn(kc.NH, kc.HD, kc.HD, generator=gen, device=device) * 0.05
    cv = (torch.randn(3, kc.QKV, generator=gen, device=device) * 0.5).to(kc.BF)
    ref, res = None, []
    for R in reqs:
        lay = Layer(R, w, sw, k2w, *env, device)
        for pos in sorted({0, R - 1}):
            S = R * L
            g2 = torch.Generator(device=device).manual_seed(700 + 10 * R + pos)
            stream, blocks = inputs(S, g2, device)
            rs = slice(pos * L, (pos + 1) * L)
            stream[rs], blocks[rs] = sr, br
            pools = kc.Pools(R, g2, device)
            slot, row = int(pools.slots[pos]), int(pools.rows[pos])
            pools.temporal[slot] = st
            pools.conv[slot] = cv
            kp = torch.zeros(S, kc.H, dtype=kc.BF, device=device)
            out = torch.empty_like(kp)
            core = lay.one(stream, blocks, pools, kp, out)
            torch.cuda.synchronize()
            g = {
                "out": out[rs].clone(),
                "prefix": kp[rs].clone(),
                "core": core[rs].clone(),
                "x": lay.front.x_rows(S)[rs].clone(),
                "window": pools.window[row].clone(),
                **{n: t[slot, :, :L].clone() for n, t in zip(("ring_k", "ring_v", "ring_g", "ring_b"), pools.ring)},
            }
            if ref is None:
                ref = g
            same = {k: bool(torch.equal(v, ref[k])) for k, v in g.items()}
            res.append({"R": R, "pos": pos, "equal": all(same.values()), "parts": same})
    return res


def time_forms(R, w, sw, k2w, env, device, sets=8, reps=20):
    """Each form as one CUDA graph over ``sets`` distinct input sets (so no
    set's weights or inputs sit in a cache from the last), median us a layer."""
    S = R * kc.L
    lay = Layer(R, w, sw, k2w, *env, device)
    gen = torch.Generator(device=device).manual_seed(5000 + R)
    data = []
    for _ in range(sets):
        stream, blocks = inputs(S, gen, device)
        data.append((stream, blocks, kc.Pools(R, gen, device), torch.zeros(S, kc.H, dtype=kc.BF, device=device)))
    out = torch.empty(S, kc.H, dtype=kc.BF, device=device)
    core = torch.empty(S, kc.PROJ, dtype=kc.BF, device=device)
    us = {}
    for form in ("two", "one", "two", "one"):
        run = (
            (lambda d: lay.two(d[0], d[1], d[2], d[3], out, core))
            if form == "two"
            else (lambda d: lay.one(d[0], d[1], d[2], d[3], out))
        )
        for d in data:
            run(d)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for d in data:
                run(d)
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3 / sets)
        us.setdefault(form, []).append(round(statistics.median(ts), 2))
    return {"R": R, "rows": S, "us": us}


# K2's timeline points reported (name: point); the attention's task end is its 8
K2_POINTS = {"k2 start": 0, "oproj gemv": 25, "oproj pushed/AR": 1, "x": 9, "route": 10,
             "route table": 11, "ug": 3, "down": 4, "up": 6, "reduce": 7}


def timeline(R, w, sw, k2w, env, device, runs=3):
    """Both forms' stamps (us from the attention's first CTA start), median and
    max CTA of each point, the median over ``runs`` launches."""
    S = R * kc.L
    lay = Layer(R, w, sw, k2w, *env, device, timeline=True)
    gen = torch.Generator(device=device).manual_seed(6000 + R)
    stream, blocks = inputs(S, gen, device)
    base = kc.Pools(R, gen, device)
    out = {}
    for form in ("two", "one") * runs:
        pools, kp = base.clone(), torch.zeros(S, kc.H, dtype=kc.BF, device=device)
        o = torch.empty_like(kp)
        lay.front.tl.zero_()
        lay.k2_tl.zero_()
        if form == "two":
            lay.two(stream, blocks, pools, kp, o, torch.empty(S, kc.PROJ, dtype=kc.BF, device=device))
        else:
            lay.one(stream, blocks, pools, kp, o)
        torch.cuda.synchronize()
        a, k = lay.front.tl.double(), lay.k2_tl.double()
        t0 = a[:, 0][a[:, 0] > 0].min()
        pts = {"attn tasks done": a[:, 8]}
        pts.update({n: k[:, p] for n, p in K2_POINTS.items()})
        pts["k2 end"] = k.max(dim=1).values
        # by role: AttnRes CTAs (0 .. 55), KDA task CTAs, the rest
        bid = torch.arange(k.shape[0], device=k.device)
        o_ = (bid - 64).clamp(min=0)
        task = (bid >= 64) & (o_ // 16 * 8 + o_ % 8 < R * 12)
        ar = bid < 56
        free = ~task & ~ar
        for role, m in (("AR", ar), ("task", task), ("free", free)):
            pts[f"{role} k2 start"] = k[m, 0]
            pts[f"{role} oproj gemv"] = k[m, 25]
            pts[f"{role} oproj pushed"] = k[m, 1] if role != "AR" else k[m, 26]
        for n, p in (("AR reduced", 1), ("AR slice", 2), ("AR gate", 8), ("AR mix", 9)):
            pts[n] = k[ar, p]
        row = {}
        for n, v in pts.items():
            v = (v[v > 0] - t0) / 100.0  # the 100 MHz clock, us
            if v.numel():
                row[n] = (float(v.median()), float(v.max()))
        out.setdefault(form, []).append(row)
    res = {}
    for form, rows in out.items():
        res[form] = {
            n: [round(statistics.median(r[n][j] for r in rows), 2) for j in (0, 1)]
            for n in rows[0]
        }
    return {"R": R, "rows": S, "points_us_median_max": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--copy", required=True)
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--reqs", default="1,2,4,8")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    load_tree(args.tree)
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950 import kda_layer as kl
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    peers = PeerBuffer(k2.peer_alloc_bytes(), None, 0, 1, device)
    k2.init_peer(peers.bytes)
    addresses = torch.tensor([peers.local] * k2.TP_, dtype=torch.int64, device=device)
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(DEADLINE_S)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    # the runner's scratch: uncached, every launch's layout in it
    sbuf = PeerBuffer((kl.scratch_end() + 255) // 256 * 256, None, 0, 1, device)
    sbuf.bytes.zero_()
    env = (sbuf.bytes, addresses, epoch)
    gen = torch.Generator(device=device).manual_seed(0)
    w = kc.load_layer(args.copy, args.layer, device)
    sw = seam_weights(args.copy, args.layer, device, gen)
    k2w = weights(args.copy, args.layer, device)
    reqs = list(map(int, args.reqs.split(",")))
    res, bad = {"cases": [], "invariance": [], "time": []}, 0
    for R in [] if args.no_check else reqs:
        r = run_case(R, w, sw, k2w, env, device)
        res["cases"].append(r)
        bad += not r["ok"]
        print("KDA_ONE_CHECK", json.dumps(r), flush=True)
    for r in [] if args.no_check else run_invariance(w, sw, k2w, env, device, tuple(reqs)):
        res["invariance"].append(r)
        bad += not r["equal"]
        print("KDA_ONE_INVARIANCE", json.dumps(r), flush=True)
    if args.timeline:
        res["timeline"] = []
        for R in reqs:
            t = timeline(R, w, sw, k2w, env, device)
            res["timeline"].append(t)
            print("KDA_ONE_TIMELINE", json.dumps(t), flush=True)
    if args.time:
        for R in reqs:
            t = time_forms(R, w, sw, k2w, env, device)
            res["time"].append(t)
            print("KDA_ONE_TIME", json.dumps(t), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print("KDA_ONE_CHECK_DONE", "fail" if bad else "pass", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
