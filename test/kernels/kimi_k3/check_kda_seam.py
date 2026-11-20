# SPDX-License-Identifier: Apache-2.0
"""The KDA attention seam in the previous layer's K2 tail (``layer.TailSeam``)
against the seam in the next layer's own K1 front, on one GPU, one rank's real
weights (a loopback tree: every K2 all-reduce sums 8 copies of this rank's
partial, as ``check_kda_one_launch.py``).

Per form and bank depth, the previous layer's K2 then the next layer's K1:

* base: K2 with no tail, then K1 with its front on the stream K2 wrote;
* tail: K2 with the tail, then K1 with no front on the x the tail wrote.

Checked bit for bit: K2's output (the stream) and updated prefix, x (the
tail's against the front's), and the next layer's core, ring rows and verify
windows, at 12 rows (two requests, the wide form): ``pack_x`` x, the next
layer as its one launch (its K2's output and prefix too). Then the tail's x rows of a request alone are the same bits as first /
last of two (12 rows), and the tail build run to run. ``--time``: the K2
(base / tail) and the next K1 (front / none) apart, CUDA graphs, median us.

    python3 check_kda_seam.py --tree TREE --copy ~/pe-k3/copy/<key> --prev 5 --next 6
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import kda_wide_case as kc  # noqa: E402
from check_k2_loopback import DEADLINE_S, load_tree, weights  # noqa: E402
from check_kda_layer import seam_weights, unpack_x  # noqa: E402


class Env:
    """The runner's peer buffer, epoch and scratch, as ``check_kda_one_launch``."""

    def __init__(self, device):
        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.kimi_k3.gfx950 import kda_layer as kl
        from sgl_amd_mono.kimi_k3.gfx950 import layer as k2

        self.peers = PeerBuffer(k2.peer_alloc_bytes(), None, 0, 1, device)
        k2.init_peer(self.peers.bytes)
        self.addresses = torch.tensor([self.peers.local] * k2.TP_, dtype=torch.int64, device=device)
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        ticks = guard.deadline_ticks(DEADLINE_S)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
        self._sbuf = PeerBuffer(k2.scratch_alloc_bytes(), None, 0, 1, device)
        self._sbuf.bytes.zero_()
        self.scratch = self._sbuf.bytes


class Chain:
    """The previous layer's K2 (core given) and the next layer's KDA layer as
    one launch, at one width and bank depth."""

    def __init__(self, S, nb, k2w, kw_, sw, k2w_next, env, device, prev, nxt):
        from sgl_amd_mono.kimi_k3.gfx950 import layer as k2
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

        self.k2, self.kda, self.kw = k2, kda, kw
        self.S, self.nb, self.env, self.device = S, nb, env, device
        self.prev, self.nxt = prev, nxt
        self.k2w, self.w, self.sw, self.k2w_next = k2w, kw_, sw, k2w_next
        assert S > kc.L, "the tail feeds the wide form"
        self.base_key = k2.K2Build(
            tokens=S, nblocks=nb, eps=k2w["eps"], out_eps=k2w["out_eps"], guint=True, opt=k2.K2_OPT
        )
        tiles = kw.tiles(kw.KdaWideBuild(tokens=S, qlen=kc.L))
        self.tail_key = k2.K2Build(
            tokens=S, nblocks=nb, eps=k2w["eps"], out_eps=k2w["out_eps"], guint=True,
            opt=k2.K2_OPT, tail=k2.TailSeam(eps=kc.EPS, out_eps=kc.EPS, tiles=tiles),
        )  # fmt: skip
        self.w_in = kw.pack_w_in(kw_["w_in"])
        self.akey = kw.KdaWideBuild(
            tokens=S, qlen=kc.L, onorm_eps=kc.EPS, lower_bound=kc.LB, front=True,
            nblocks=nb, eps=kc.EPS, out_eps=kc.EPS,
        )  # fmt: skip
        self.x = torch.zeros(16 * tiles, kc.H, dtype=kc.BF, device=device)
        self.next_key = k2.K2Build(
            tokens=S, nblocks=nb, eps=k2w_next["eps"], out_eps=k2w_next["out_eps"],
            guint=True, opt=k2.K2_OPT,
        )  # fmt: skip
        self.table = torch.tensor(
            [sw["ares_nw"].data_ptr(), sw["ares_qk"].data_ptr(), sw["in_nw"].data_ptr(), self.x.data_ptr()],
            dtype=torch.int64, device=device,
        )  # fmt: skip

    def run_k2(self, key, w, core, kprefix, stream, blocks, out, layer, tail=None, attn=None):
        e = self.env
        self.k2.k2_launch(
            key, core=core, w_o=w["w_o"], prefix=kprefix, blocks=blocks,
            ares_nw=w["ares_nw"], ares_qk=w["ares_qk"], in_nw=w["in_nw"],
            w_gate=w["w_gate"], bias=w["bias"], w_ld=w["w_ld"], w_sgu=w["w_sgu"],
            w_sd=w["w_sd"], w13=w["w13"], w13s=w["w13s"], w2=w["w2"], w2s=w["w2s"],
            ln_w=w["ln_w"], w_up=w["w_up_shard"], out=out, scratch=e.scratch,
            peers=e.addresses, rank=0, epoch=e.epoch, layer=128 + layer,
            prefix_in=stream, tail=tail, attn=attn,
        )  # fmt: skip

    def run_next(self, front, stream, blocks, pools):
        """The next layer's one launch: its K1 front on ``stream``, or none on
        ``self.x``; returns its outputs."""
        import dataclasses

        from sgl_amd_mono.kimi_k3.gfx950 import kda_layer as kl

        e, w, dev = self.env, self.w, self.device
        akey = self.akey if front else dataclasses.replace(self.akey, front=False)
        core = kl.core_view(akey, e.scratch)
        vals = self.kw.launch_values(
            akey, x=self.x, w_in=self.w_in, w_fb=w["w_fb"], conv_w=w["conv_w"],
            conv_state=pools.conv.mT, a_log=w["a_log"], dt_bias=w["dt_bias"], on_w=w["on_w"],
            rstate=pools.temporal, st_idx=pools.slots, conv_win=pools.window,
            win_row=pools.rows, ring=pools.ring, core_out=core,
            scratch=kl.attn_scratch(akey, e.scratch), layer=self.nxt, epoch=e.epoch,
            prefix=stream if front else None, blocks=blocks if front else None, **self.sw,
        )  # fmt: skip
        kp = torch.zeros(self.S, kc.H, dtype=kc.BF, device=dev)
        out = torch.empty_like(kp)
        self.run_k2(
            self.next_key, self.k2w_next, core, kp, stream, blocks, out, self.nxt,
            attn=kl.Attn(key=akey, vals=vals),
        )  # fmt: skip
        torch.cuda.synchronize()
        o = {"core": core.clone(), "next_out": out.clone(), "next_prefix": kp.clone()}
        o.update({k: v.clone() for k, v in pools.outputs().items()})
        if front:
            o["x"] = unpack_x(self.x, self.x.shape[0] // 16)[: self.S].clone()
        return o

    def chain(self, tail, core, stream, blocks, pools):
        """K2 (prev) then K1 (next): (K2's out, its prefix, the next layer's outputs)."""
        S, dev, e = self.S, self.device, self.env
        kp = torch.zeros(S, kc.H, dtype=kc.BF, device=dev)
        out = torch.empty_like(kp)
        e.epoch[0].add_(1)
        if tail:
            self.x.fill_(7.0)  # what the tail must overwrite
            self.x[S:].zero_()
        self.run_k2(
            self.tail_key if tail else self.base_key, self.k2w, core, kp, stream, blocks, out,
            self.prev, tail=self.table if tail else None,
        )  # fmt: skip
        torch.cuda.synchronize()
        x = None
        if tail:
            x = unpack_x(self.x, self.x.shape[0] // 16)[:S].clone()
        e.epoch[0].add_(1)
        nxt = self.run_next(not tail, out, blocks, pools.clone())
        if tail:
            nxt["x"] = x
        return out.clone(), kp.clone(), nxt


def inputs(S, nb, gen, device):
    core = (torch.randn(S, kc.PROJ, generator=gen, device=device) * 0.3).to(kc.BF)
    stream = (torch.randn(S, kc.H, generator=gen, device=device) * 0.3).to(kc.BF)
    blocks = (torch.randn(S, nb + 1, kc.H, generator=gen, device=device) * 0.3).to(kc.BF)
    return core, stream, blocks


def run_case(ch, gen):
    S, nb = ch.S, ch.nb
    core, stream, blocks = inputs(S, nb, gen, ch.device)
    pools = kc.Pools(S // kc.L, gen, ch.device)
    ob, pb, nb_ = ch.chain(False, core, stream, blocks, pools)
    ot, pt, nt = ch.chain(True, core, stream, blocks, pools)
    ot2, _, nt2 = ch.chain(True, core, stream, blocks, pools)
    res = {"rows": S, "nblocks": nb}
    res["k2_out"] = bool(torch.equal(ot, ob))
    res["k2_prefix"] = bool(torch.equal(pt, pb))
    res["next"] = {k: bool(torch.equal(nt[k], nb_[k])) for k in nb_}
    res["x_vs_front"] = kc.cmp(nt["x"], nb_["x"])
    res["repeatable"] = bool(torch.equal(ot2, ot)) and all(torch.equal(nt2[k], nt[k]) for k in nt)
    res["ok"] = res["k2_out"] and res["k2_prefix"] and all(res["next"].values()) and res["repeatable"]
    return res


def run_invariance(ch, gen):
    """12 rows: request A's rows alone in either position of two give the same
    tail x rows."""
    S, L = ch.S, kc.L
    a = inputs(L, ch.nb, gen, ch.device)
    xs = []
    for pos in (0, 1):
        g2 = torch.Generator(device=ch.device).manual_seed(800 + pos)
        core, stream, blocks = inputs(S, ch.nb, g2, ch.device)
        rs = slice(pos * L, (pos + 1) * L)
        core[rs], stream[rs], blocks[rs] = a
        pools = kc.Pools(S // L, g2, ch.device)
        _, _, n = ch.chain(True, core, stream, blocks, pools)
        xs.append(n["x"][rs].clone())
    return {"rows": S, "nblocks": ch.nb, "equal": bool(torch.equal(xs[0], xs[1]))}


def time_parts(ch, gen, sets=8, reps=20):
    """The previous layer's K2 (base / tail) and the next layer's one launch
    (its K1 front / none), each a CUDA graph over ``sets`` input sets,
    median us a launch, in A B B A order."""
    import dataclasses
    import statistics

    from sgl_amd_mono.kimi_k3.gfx950 import kda_layer as kl

    S, dev, e, w = ch.S, ch.device, ch.env, ch.w
    data = []
    for _ in range(sets):
        core, stream, blocks = inputs(S, ch.nb, gen, dev)
        pools = kc.Pools(S // kc.L, gen, dev)
        data.append((core, stream, blocks, pools, None, torch.zeros(S, kc.H, dtype=kc.BF, device=dev)))
    out = torch.empty(S, kc.H, dtype=kc.BF, device=dev)

    def k2(d, tail):
        ch.run_k2(ch.tail_key if tail else ch.base_key, ch.k2w, d[0], d[5], d[1], d[2], out,
                  ch.prev, tail=ch.table if tail else None)

    def k1(d, front):
        akey = ch.akey if front else dataclasses.replace(ch.akey, front=False)
        core = kl.core_view(akey, e.scratch)
        pools = d[3]
        vals = ch.kw.launch_values(
            akey, x=ch.x, w_in=ch.w_in, w_fb=w["w_fb"], conv_w=w["conv_w"],
            conv_state=pools.conv.mT, a_log=w["a_log"], dt_bias=w["dt_bias"], on_w=w["on_w"],
            rstate=pools.temporal, st_idx=pools.slots, conv_win=pools.window,
            win_row=pools.rows, ring=pools.ring, core_out=core,
            scratch=kl.attn_scratch(akey, e.scratch), layer=ch.nxt, epoch=e.epoch,
            prefix=d[1] if front else None, blocks=d[2] if front else None, **ch.sw,
        )  # fmt: skip
        ch.run_k2(ch.next_key, ch.k2w_next, core, d[5], d[1], d[2], out, ch.nxt,
                  attn=kl.Attn(key=akey, vals=vals))

    def graph_us(fn):
        for d in data:
            e.epoch[0].add_(1)
            fn(d)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for d in data:
                e.epoch[0].add_(1)
                fn(d)
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1e3 / sets)
        return round(statistics.median(ts), 2)

    us = {}
    for name, fn in (
        ("k2 base", lambda d: k2(d, False)), ("k2 tail", lambda d: k2(d, True)),
        ("k2 tail", lambda d: k2(d, True)), ("k2 base", lambda d: k2(d, False)),
        ("k1 front", lambda d: k1(d, True)), ("k1 none", lambda d: k1(d, False)),
        ("k1 none", lambda d: k1(d, False)), ("k1 front", lambda d: k1(d, True)),
    ):  # fmt: skip
        us.setdefault(name, []).append(graph_us(fn))
    return {"rows": S, "nblocks": ch.nb, "us": us}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--copy", required=True)
    ap.add_argument("--prev", type=int, default=5)
    ap.add_argument("--next", type=int, default=6)
    ap.add_argument("--nbs", default="1,3,8")
    ap.add_argument("--rows", default="12")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    load_tree(args.tree)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    env = Env(device)
    gen = torch.Generator(device=device).manual_seed(0)
    k2w = weights(args.copy, args.prev, device)
    k2w_next = weights(args.copy, args.next, device)
    w = kc.load_layer(args.copy, args.next, device)
    sw = seam_weights(args.copy, args.next, device, gen)
    res, bad = {"cases": [], "invariance": []}, 0
    for S in map(int, args.rows.split(",")):
        for nb in map(int, args.nbs.split(",")):
            ch = Chain(S, nb, k2w, w, sw, k2w_next, env, device, args.prev, args.next)
            if args.time:
                t = time_parts(ch, torch.Generator(device=device).manual_seed(2000 + nb))
                res.setdefault("time", []).append(t)
                print("KDA_SEAM_TIME", json.dumps(t), flush=True)
            if args.no_check:
                continue
            r = run_case(ch, torch.Generator(device=device).manual_seed(1000 + 10 * S + nb))
            res["cases"].append(r)
            bad += not r["ok"]
            print("KDA_SEAM_CHECK", json.dumps(r), flush=True)
            r = run_invariance(ch, torch.Generator(device=device).manual_seed(77 + nb))
            res["invariance"].append(r)
            bad += not r["equal"]
            print("KDA_SEAM_INVARIANCE", json.dumps(r), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print("KDA_SEAM_CHECK_DONE", "fail" if bad else "pass", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
