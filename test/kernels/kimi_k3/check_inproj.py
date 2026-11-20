# SPDX-License-Identifier: Apache-2.0
"""The KDA in_proj packed in place (``gfx950/inproj.py``), one GPU:

* ``unpack_w_in`` inverts ``pack_w_in`` bit for bit, and ``pack_chunks`` gives
  each chunk ``pack_x``'s layout of its rows;
* SGLang's verify path on a packed layer (``InProj.chunks``, kda_wide CHUNK
  requests a launch) at 1 .. 30 requests: every request's core, ring rows and
  verify windows bitwise equal to the request run alone, and at 8 requests to
  one 48-row launch; the committed state and conv state untouched; against
  SGLang's own ReplaySSM path (rel-L2);
* graph time of the chunked path, SGLang's path, and the unpack.

Prints one JSON line (``--out`` also writes it).

    python test/kernels/kimi_k3/check_inproj.py [--copy DIR --layer N]
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kda_wide_case as kc  # noqa: E402


def pools_ns(p):
    return SimpleNamespace(
        conv_state=p.conv.mT, rstate=p.temporal, conv_win=p.window, ring=p.ring,
        slots=p.slots, rows=p.rows,
    )  # fmt: skip


def graph_us(fn, reps=40):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


class _Attn(torch.nn.Module):
    """SGLang's KDA attention as far as the packing sees it: the merged in_proj
    and its views, and a forward that reads them."""

    def __init__(self, w):
        super().__init__()
        merged = w["w_in"].clone()
        self.merged = merged
        lin = lambda t: SimpleNamespace(weight=torch.nn.Parameter(t, requires_grad=False))  # noqa: E731
        self.fused_qkvg_proj = lin(merged[: kc.QKV + kc.PROJ])
        self.f_a_proj = lin(merged[kc.QKV + kc.PROJ : kc.QKV + kc.PROJ + kc.HD])
        b0 = kc.QKV + kc.PROJ + kc.HD
        self.b_proj = lin(merged[b0 : b0 + kc.NH])
        self._bfa_w = merged[kc.QKV + kc.PROJ :]
        self._qkvgbfa_layer = SimpleNamespace(weight=merged)
        self.f_b_proj = lin(w["w_fb"])
        self.attn = SimpleNamespace(conv_weights=w["conv_w"], lower_bound=kc.LB, layer_id=1)
        self.A_log, self.dt_bias = w["a_log"], w["dt_bias"]
        self.o_norm = SimpleNamespace(weight=w["on_w"], eps=kc.EPS)

    def forward_pre_o_proj(self, *, hidden_states, positions, forward_batch, zero_allocator):
        x = hidden_states
        return torch.cat(
            [
                x @ self.fused_qkvg_proj.weight.t(),
                x @ self._bfa_w.t(),
                x @ self.f_a_proj.weight.t(),
                x @ self.b_proj.weight.t(),
                x @ self._qkvgbfa_layer.weight.t(),
            ],
            dim=1,
        )


def in_place(w, p, gen, dev):
    """Pack a layer in place: the merged buffer holds pack_w_in's layout, and the
    SGLang path (extend) reads every view right through the shared buffer."""
    from sgl_amd_mono.kimi_k3.gfx950 import inproj
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

    attn = _Attn(w)
    layer = SimpleNamespace(self_attn=attn)
    x = torch.randn(40, kc.H, generator=gen, device=dev).to(kc.BF)
    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: False))
    kwargs = dict(hidden_states=x, positions=None, forward_batch=fb, zero_allocator=None)
    want = type(attn).forward_pre_o_proj(attn, **kwargs)
    ptr = attn.merged.data_ptr()
    before = torch.cuda.memory_allocated()
    got_w = p.pack(layer, 1)
    after = torch.cuda.memory_allocated()
    other = _Attn(w)  # a second layer shares the unpack buffer
    p.pack(SimpleNamespace(self_attn=other), 2)
    got = attn.forward_pre_o_proj(**kwargs)
    return {
        "packed_in_place": got_w.data_ptr() == ptr
        and bool(torch.equal(attn.merged, kw.pack_w_in(w["w_in"]))),
        "bytes_kept": after - before,
        "extend_bitwise": bool(torch.equal(got, want)),
        "views_shared": attn.fused_qkvg_proj.weight.data_ptr() == p.plain.data_ptr()
        and other.fused_qkvg_proj.weight.data_ptr() == p.plain.data_ptr(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", default="")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--reqs", default="1,2,5,8,9,12,16,17,24,29,30")
    ap.add_argument("--time-reqs", default="8,9,12,16,24,29,30")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    from sgl_amd_mono.kimi_k3.gfx950 import inproj
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

    w = kc.load_layer(args.copy, args.layer, dev) if args.copy else kc.random_layer(gen, dev)
    out = {"layer": args.layer if args.copy else "random"}
    packed = kw.pack_w_in(w["w_in"])
    p = inproj.InProj(dev)
    inproj.unpack_w_in(packed, p.plain)
    out["unpack_bitwise"] = bool(torch.equal(p.plain, w["w_in"]))
    xs_ok = {}
    for R in (1, 5, 8, 9, 13, 30):
        x = torch.randn(R * kc.L, kc.H, generator=gen, device=dev).to(kc.BF)
        inproj.pack_chunks(x, p.xp, inproj.CHUNK * kc.L)
        ok = True
        for c, r0 in enumerate(range(0, R, inproj.CHUNK)):
            r = min(inproj.CHUNK, R - r0)
            nt = (r * kc.L + 15) // 16
            want = kw.pack_x(x[r0 * kc.L : (r0 + r) * kc.L], nt)
            ok &= bool(torch.equal(p.xp[c, : 16 * nt], want))
        xs_ok[R] = ok
    out["pack_chunks_bitwise"] = xs_ok

    out["in_place"] = in_place(w, p, gen, dev)

    ww = dict(w, w_in=packed)
    entry = inproj.layer_entry(1, ww, kc.EPS, kc.LB)
    stock = kc.Stock(w)
    alone = kc.Wide(w, kc.L, dev)
    ep = torch.zeros(8, dtype=torch.int32, device=dev)

    def run_chunks(x, R, pools):
        p.epochs[:, 0].add_(inproj.EPOCH_STRIDE)
        return p.chunks(entry, x, R, pools_ns(pools))

    cases = []
    for R in [int(v) for v in args.reqs.split(",")]:
        pools = kc.Pools(R, gen, dev)
        x = (torch.randn(R * kc.L, kc.H, generator=gen, device=dev)).to(kc.BF)
        t0, c0 = pools.temporal.clone(), pools.conv.clone()
        pc = pools.clone()
        core = run_chunks(x, R, pc)
        torch.cuda.synchronize()
        got = pc.outputs()
        case = {
            "R": R,
            "state_untouched": bool(torch.equal(pc.temporal, t0) and torch.equal(pc.conv, c0)),
        }
        # each request alone
        inv = True
        for j in range(R):
            pa = pools.clone()
            pa.slots, pa.rows = pools.slots[j : j + 1].clone(), pools.rows[j : j + 1].clone()
            ca = torch.zeros(kc.L, kc.PROJ, dtype=kc.BF, device=dev)
            ep.add_(1)
            alone._xk = None
            alone(x[j * kc.L : (j + 1) * kc.L].contiguous(), pa, ep, ca)
            oa = pa.outputs()
            inv &= bool(torch.equal(ca, core[j * kc.L : (j + 1) * kc.L]))
            for k in oa:
                inv &= bool(torch.equal(oa[k], got[k][j : j + 1]))
        case["alone_bitwise"] = inv
        if R == inproj.CHUNK:
            one = kc.Wide(w, R * kc.L, dev)
            po = pools.clone()
            co = torch.zeros(R * kc.L, kc.PROJ, dtype=kc.BF, device=dev)
            ep.add_(1)
            one._xk = None
            one(x, po, ep, co)
            oo = po.outputs()
            case["one_launch_bitwise"] = bool(torch.equal(co, core)) and all(
                torch.equal(oo[k], got[k]) for k in oo
            )
        ps = pools.clone()
        ref = stock(x, ps)
        case["vs_stock_core"] = kc.cmp(core, ref)["rel_l2"]
        os_ = ps.outputs()
        case["vs_stock_ring"] = {k: kc.cmp(got[k], os_[k])["rel_l2"] for k in got}
        cases.append(case)
    out["cases"] = cases
    ip = out["in_place"]
    out["pass"] = (
        out["unpack_bitwise"]
        and ip["packed_in_place"] and ip["extend_bitwise"] and ip["views_shared"]
        and ip["bytes_kept"] <= 0
        and all(xs_ok.values())
        and all(c["alone_bitwise"] and c["state_untouched"] for c in cases)
        and all(c.get("one_launch_bitwise", True) for c in cases)
        and all(c["vs_stock_core"] < 1e-2 for c in cases)
    )

    times = {"unpack": graph_us(lambda: inproj.unpack_w_in(packed, p.plain))}
    for R in [int(v) for v in args.time_reqs.split(",")]:
        pools = kc.Pools(R, gen, dev)
        x = torch.randn(R * kc.L, kc.H, generator=gen, device=dev).to(kc.BF)
        times[f"chunks_r{R}"] = graph_us(lambda: run_chunks(x, R, pools))
        times[f"stock_r{R}"] = graph_us(lambda: stock(x, pools))
    out["us"] = times
    line = json.dumps(out)
    print(line)
    if args.out:
        with open(args.out, "w") as f:
            f.write(line + "\n")
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
