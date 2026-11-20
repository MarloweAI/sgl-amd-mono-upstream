# SPDX-License-Identifier: Apache-2.0
"""The KDA in_proj packed in place (``gfx950/inproj.py``), one GPU:

* ``kda_wide.unpack_w_in`` inverts ``pack_w_in`` bit for bit, and
  ``pack_chunks`` gives ``pack_x``'s layout;
* the extend path on a packed layer: SGLang's q, k, v, g GEMM through
  ``Preshuffled`` on the packed rows (rel-L2 against the plain GEMM), its f_a | b
  GEMM on the plain tail (bitwise);
* SGLang's verify path on a packed layer (``InProj.launch``: one kda_wide
  launch, its GEMM form past 8 requests) at 1 .. 64 requests: every request's
  core, ring rows and verify windows bitwise equal to the request run alone;
  the committed state and conv state untouched; against SGLang's own ReplaySSM
  path (rel-L2);
* graph time of that path and SGLang's, per width; and of the prefill q, k, v,
  g GEMM, stock (plain weight) and ``Preshuffled`` (packed), per M.

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


class _Plain:
    """SGLang's unquantized GEMM on a plain weight, as the stand-in's method."""

    @staticmethod
    def apply(layer, x, bias=None):
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        return UnquantizedLinearMethod().apply(layer, x, bias)


class _Attn(torch.nn.Module):
    """SGLang's KDA attention as far as the packing sees it: the merged in_proj
    and its views, and an extend forward that reads them as SGLang's does (q, k,
    v, g through ``fused_qkvg_proj``'s method, f_a | b plain)."""

    def __init__(self, w):
        super().__init__()
        merged = w["w_in"].clone()
        self.merged = merged
        lin = lambda t: SimpleNamespace(weight=torch.nn.Parameter(t, requires_grad=False))  # noqa: E731
        self.fused_qkvg_proj = lin(merged[: kc.QKV + kc.PROJ])
        self.fused_qkvg_proj.quant_method = _Plain()
        self.f_a_proj = lin(merged[kc.QKV + kc.PROJ : kc.QKV + kc.PROJ + kc.HD])
        b0 = kc.QKV + kc.PROJ + kc.HD
        self.b_proj = lin(merged[b0 : b0 + kc.NH])
        self._bfa_w = merged[kc.QKV + kc.PROJ :]
        self._qkvgbfa_layer = SimpleNamespace(weight=merged)
        self._qkvgbfa_sizes = [kc.QKV + kc.PROJ, kc.HD, kc.NH]
        self.f_b_proj = lin(w["w_fb"])
        self.attn = SimpleNamespace(conv_weights=w["conv_w"], lower_bound=kc.LB, layer_id=1)
        self.A_log, self.dt_bias = w["a_log"], w["dt_bias"]
        self.o_norm = SimpleNamespace(weight=w["on_w"], eps=kc.EPS)

    def forward_pre_o_proj(self, *, hidden_states, positions, forward_batch, zero_allocator):
        x = hidden_states
        m = self.fused_qkvg_proj
        return torch.cat([m.quant_method.apply(m, x, None), x @ self._bfa_w.t()], dim=1)


def in_place(w, p, gen, dev):
    """Pack a layer in place: the merged buffer holds pack_w_in's layout (and a
    plain copy of the 144-row f_a | b tail is all that is added), and the SGLang
    path (extend) reads it as it is."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

    attn = _Attn(w)
    layer = SimpleNamespace(self_attn=attn)
    x = torch.randn(512, kc.H, generator=gen, device=dev).to(kc.BF)
    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: False))
    kwargs = dict(hidden_states=x, positions=None, forward_batch=fb, zero_allocator=None)
    want = type(attn).forward_pre_o_proj(attn, **kwargs)
    ptr = attn.merged.data_ptr()
    before = torch.cuda.memory_allocated()
    got_w = p.pack(layer, 1)
    after = torch.cuda.memory_allocated()
    got = attn.forward_pre_o_proj(**kwargs)
    q = kc.QKV + kc.PROJ
    return {
        "packed_in_place": got_w.data_ptr() == ptr
        and bool(torch.equal(attn.merged, kw.pack_w_in(w["w_in"]))),
        "bytes_kept": after - before,
        "extend_tail_bitwise": bool(torch.equal(got[:, q:], want[:, q:])),
        "extend_qkvg_rel": kc.cmp(got[:, :q], want[:, :q])["rel_l2"],
        "merged_gemm_off": attn._qkvgbfa_sizes is None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", default="")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--reqs", default="1,2,5,8,9,12,16,17,24,29,30,33,64")
    ap.add_argument("--time-reqs", default="1,2,4,8,9,10,12,14,16,18,20,22,24,26,28,29,30,48,64")
    ap.add_argument("--prefill-m", default="256,512,768,1024,1536,2048,3072,4096,8192,16384")
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
    out["unpack_bitwise"] = bool(torch.equal(kw.unpack_w_in(packed), w["w_in"]))
    xs_ok = {}
    for R in (1, 5, 8, 9, 13, 30, 64):
        x = torch.randn(R * kc.L, kc.H, generator=gen, device=dev).to(kc.BF)
        inproj.pack_chunks(x, p.xp, R * kc.L)
        nt = (R * kc.L + 15) // 16
        xs_ok[R] = bool(torch.equal(p.xp[0, : 16 * nt], kw.pack_x(x, nt)))
    out["pack_chunks_bitwise"] = xs_ok

    out["in_place"] = in_place(w, p, gen, dev)

    ww = dict(w, w_in=packed)
    entry = inproj.layer_entry(1, ww, kc.EPS, kc.LB)
    stock = kc.Stock(w)
    alone = kc.Wide(w, kc.L, dev)
    ep = torch.zeros(8, dtype=torch.int32, device=dev)

    def run_ours(x, R, pools):
        p.epochs[:, 0].add_(1)
        return p.launch(entry, x, R, pools_ns(pools))

    cases = []
    for R in [int(v) for v in args.reqs.split(",")]:
        pools = kc.Pools(R, gen, dev)
        x = (torch.randn(R * kc.L, kc.H, generator=gen, device=dev)).to(kc.BF)
        t0, c0 = pools.temporal.clone(), pools.conv.clone()
        pc = pools.clone()
        core = run_ours(x, R, pc)
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
        if R == kw.WIDE_REQS:
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
        and ip["packed_in_place"] and ip["extend_tail_bitwise"] and ip["merged_gemm_off"]
        and ip["extend_qkvg_rel"] < 5e-3
        and ip["bytes_kept"] <= 3 << 20
        and all(xs_ok.values())
        and all(c["alone_bitwise"] and c["state_untouched"] for c in cases)
        and all(c.get("one_launch_bitwise", True) for c in cases)
        and all(c["vs_stock_core"] < 1e-2 for c in cases)
    )

    times = {}
    for R in [int(v) for v in args.time_reqs.split(",")]:
        pools = kc.Pools(R, gen, dev)
        x = torch.randn(R * kc.L, kc.H, generator=gen, device=dev).to(kc.BF)
        times[f"pack_x_r{R}"] = graph_us(lambda: inproj.pack_chunks(x, p.xp, R * kc.L))
        times[f"ours_r{R}"] = graph_us(lambda: run_ours(x, R, pools))
        times[f"stock_r{R}"] = graph_us(lambda: stock(x, pools))
    # the prefill q, k, v, g GEMM: stock on the plain rows, ours on the packed ones
    qrows = kc.QKV + kc.PROJ
    plain = SimpleNamespace(weight=w["w_in"][:qrows])
    pk = packed[:qrows]
    pk.is_shuffled = True
    shuf = SimpleNamespace(weight=pk)
    pre = {}
    for m in [int(v) for v in args.prefill_m.split(",") if v]:
        x = torch.randn(m, kc.H, generator=gen, device=dev).to(kc.BF)
        ref = _Plain.apply(plain, x)
        got = inproj.PRESHUFFLED.apply(shuf, x)
        pre[m] = {
            "stock": graph_us(lambda: _Plain.apply(plain, x), reps=20),
            "ours": graph_us(lambda: inproj.PRESHUFFLED.apply(shuf, x), reps=20),
            "rel": kc.cmp(got, ref)["rel_l2"],
            "bitwise": bool(torch.equal(got, ref)),
        }
    out["prefill"] = pre
    out["us"] = times
    line = json.dumps(out)
    print(line)
    if args.out:
        with open(args.out, "w") as f:
            f.write(line + "\n")
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
