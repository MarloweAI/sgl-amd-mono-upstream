# SPDX-License-Identifier: Apache-2.0
"""K1's wide form as the KDA layer runs it: the attention-side AttnRes seam and
``input_layernorm`` in front of the wide KDA attention launch
(``kda_wide.KdaWideBuild(front=True)``), one GPU, 1 .. 8 requests of 6 rows.

    python test/kernels/kimi_k3/check_kda_layer.py --copy ~/pe-k3/copy/<key> --layer 4

Per width and bank depth:

* x, the front's output (``pack_x``'s layout), against a torch reference of
  the seam and the norm (``SEAM_REL``);
* the attention past x, bit for bit against the launch without the front fed
  that x (core, ring rows, verify windows);
* end to end against SGLang's ReplaySSM verify path fed that x (as
  ``check_kda_wide.py``);
* a block-write layer's bank row: the prefix, bit for bit, other rows untouched;
* the committed state and conv state untouched, a second launch the same bits.

Then batch invariance: one request alone and first / last among 1 / 3 / 7
others gives the same x rows, core, ring rows and windows. Prints one JSON line
a case; exits non-zero on a failure.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import kda_wide_case as kc  # noqa: E402
from check_k2_loopback import load_layer as load_views  # noqa: E402

SEAM_REL = 5e-3
E2E_REL = 1e-2


def seam_weights(copy, layer, device, gen):
    if not copy:
        return {
            "ares_nw": (1.0 + 0.1 * torch.randn(kc.H, generator=gen, device=device)).to(kc.BF),
            "ares_qk": (0.05 * torch.randn(kc.H, generator=gen, device=device)).to(kc.BF),
            "in_nw": (1.0 + 0.1 * torch.randn(kc.H, generator=gen, device=device)).to(kc.BF),
        }
    t = load_views(copy, layer, device)
    p = f"language_model.model.layers.{layer}."
    return {
        "ares_nw": t[p + "self_attention_res_norm.weight"].contiguous(),
        "ares_qk": t[p + "self_attention_res_proj.weight"].reshape(-1).contiguous(),
        "in_nw": t[p + "input_layernorm.weight"].contiguous(),
    }


def rms(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def seam_reference(prefix, blocks, nb, sw):
    """The bank's nb blocks and the prefix mixed by their AttnRes scores, then
    input_layernorm."""
    rows = torch.cat([blocks[:, :nb], prefix.unsqueeze(1)], dim=1)
    scores = (rms(rows, sw["ares_nw"], kc.EPS) * sw["ares_qk"].float()).sum(-1)
    mixed = (torch.softmax(scores, dim=-1).unsqueeze(-1) * rows.float()).sum(1)
    return rms(mixed, sw["in_nw"], kc.EPS).to(kc.BF)


def unpack_x(xp, nt):
    """``pack_x``'s inverse: [16 nt, H] packed -> [16 nt, H] rows."""
    v = xp.view(kc.H // 64, nt, 2, 4, 16, 8)
    return v.permute(1, 4, 0, 2, 3, 5).reshape(16 * nt, kc.H)


class Front:
    """The front build on the same weights and pools as ``kc.Wide``."""

    def __init__(self, w, sw, S, nb, write_idx, device, layer=1):
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw

        self.kw, self.w, self.sw = kw, w, sw
        self.w_in = kw.pack_w_in(w["w_in"])
        self.key = kw.KdaWideBuild(
            tokens=S, qlen=kc.L, onorm_eps=kc.EPS, lower_bound=kc.LB,
            front=True, nblocks=nb, eps=kc.EPS, out_eps=kc.EPS, write_idx=write_idx,
        )  # fmt: skip
        self.nt = kw.tiles(self.key)
        self._scratch = PeerBuffer(kw.scratch_bytes(self.key), None, 0, 1, device)
        self.scratch = self._scratch.bytes
        self.x = torch.zeros(16 * self.nt, kc.H, dtype=kc.BF, device=device)
        self.layer = layer

    def __call__(self, prefix, blocks, pools, epoch, core):
        w = self.w
        self.kw.kda_wide(
            self.key, x=self.x, w_in=self.w_in, w_fb=w["w_fb"], conv_w=w["conv_w"],
            conv_state=pools.conv.mT, a_log=w["a_log"], dt_bias=w["dt_bias"],
            on_w=w["on_w"], rstate=pools.temporal, st_idx=pools.slots,
            conv_win=pools.window, win_row=pools.rows, ring=pools.ring, core_out=core,
            scratch=self.scratch, layer=self.layer, epoch=epoch, prefix=prefix,
            blocks=blocks, **self.sw,
        )  # fmt: skip

    def x_rows(self, S):
        return unpack_x(self.x, self.nt)[:S]


def inputs(S, nb, gen, device):
    prefix = torch.randn(S, kc.H, generator=gen, device=device).to(kc.BF)
    blocks = torch.randn(S, nb + 1, kc.H, generator=gen, device=device).to(kc.BF)
    return prefix, blocks


def run_case(R, nb, write_idx, w, sw, device):
    S = R * kc.L
    gen = torch.Generator(device=device).manual_seed(3000 + 10 * R + nb)
    prefix, blocks = inputs(S, nb, gen, device)
    base = kc.Pools(R, gen, device)
    epoch = torch.ones(1, dtype=torch.int32, device=device)
    front = Front(w, sw, S, nb, write_idx, device)
    pf, bf = prefix.clone(), blocks.clone()
    pw = base.clone()
    core = torch.empty(S, kc.PROJ, dtype=kc.BF, device=device)
    front(pf, bf, pw, epoch, core)
    torch.cuda.synchronize()
    x = front.x_rows(S).clone()
    out = {"R": R, "rows": S, "nblocks": nb, "write_idx": write_idx}
    out["x"] = kc.cmp(x, seam_reference(prefix, blocks, nb, sw))
    out["x_pad_zero"] = bool(torch.equal(unpack_x(front.x, front.nt)[S:], torch.zeros_like(front.x[S:])))

    # past x: the launch without the front, on the front's x
    plain = kc.Wide(w, S, device)
    pp = base.clone()
    core_p = torch.empty_like(core)
    plain(x, pp, epoch, core_p)
    # end to end: SGLang's path on the front's x
    ps = base.clone()
    core_s = kc.Stock(w)(x, ps)
    torch.cuda.synchronize()
    wo, po, so = pw.outputs(), pp.outputs(), ps.outputs()
    out["past_x_bitwise"] = {"core": bool(torch.equal(core, core_p))}
    out["past_x_bitwise"].update({k: bool(torch.equal(wo[k], po[k])) for k in wo})
    out["e2e"] = {"core": kc.cmp(core, core_s), **{k: kc.cmp(wo[k], so[k]) for k in wo}}
    want = blocks.clone()
    if write_idx >= 0:
        want[:, write_idx] = prefix
    out["bank"] = bool(torch.equal(bf, want)) and bool(torch.equal(pf, prefix))
    out["state_untouched"] = bool(
        torch.equal(pw.temporal, base.temporal) and torch.equal(pw.conv, base.conv)
    )
    pw2 = base.clone()
    core2 = torch.empty_like(core)
    epoch += 1
    front(prefix.clone(), blocks.clone(), pw2, epoch, core2)
    torch.cuda.synchronize()
    out["repeatable"] = (
        bool(torch.equal(core2, core))
        and bool(torch.equal(front.x_rows(S), x))
        and all(torch.equal(a, b) for a, b in zip(pw2.ring, pw.ring))
    )
    out["ok"] = (
        out["x"]["rel_l2"] <= SEAM_REL
        and out["x_pad_zero"]
        and all(out["past_x_bitwise"].values())
        and out["e2e"]["core"]["rel_l2"] <= E2E_REL
        and out["bank"]
        and out["state_untouched"]
        and out["repeatable"]
    )
    return out


def run_invariance(w, sw, device, nb=3, reqs=(1, 2, 4, 8)):
    L = kc.L
    gen = torch.Generator(device=device).manual_seed(78)
    pr, br = inputs(L, nb, gen, device)
    st = torch.randn(kc.NH, kc.HD, kc.HD, generator=gen, device=device) * 0.05
    cv = (torch.randn(3, kc.QKV, generator=gen, device=device) * 0.5).to(kc.BF)
    ref, res = None, []
    for R in reqs:
        for pos in sorted({0, R - 1}):
            S = R * L
            g2 = torch.Generator(device=device).manual_seed(600 + 10 * R + pos)
            prefix, blocks = inputs(S, nb, g2, device)
            prefix[pos * L : (pos + 1) * L] = pr
            blocks[pos * L : (pos + 1) * L] = br
            pools = kc.Pools(R, g2, device)
            slot, row = int(pools.slots[pos]), int(pools.rows[pos])
            pools.temporal[slot] = st
            pools.conv[slot] = cv
            core = torch.empty(S, kc.PROJ, dtype=kc.BF, device=device)
            front = Front(w, sw, S, nb, -1, device)
            front(prefix, blocks, pools, torch.ones(1, dtype=torch.int32, device=device), core)
            torch.cuda.synchronize()
            got = {
                "x": front.x_rows(S)[pos * L : (pos + 1) * L].clone(),
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
    ap.add_argument("--reqs", default="1,2,3,4,8")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    gen = torch.Generator(device=device).manual_seed(0)
    if args.copy:
        w = kc.load_layer(args.copy, args.layer, device)
    else:
        w = kc.random_layer(gen, device)
    sw = seam_weights(args.copy, args.layer, device, gen)
    cases = [(R, 3, -1) for R in map(int, args.reqs.split(","))]
    cases += [(1, 1, -1), (2, 2, 2), (8, 8, 8)]
    res, bad = [], 0
    for R, nb, wi in cases:
        r = run_case(R, nb, wi, w, sw, device)
        res.append(r)
        bad += not r["ok"]
        print("KDA_LAYER_CHECK", json.dumps(r), flush=True)
    inv = run_invariance(w, sw, device)
    for r in inv:
        print("KDA_LAYER_INVARIANCE", json.dumps(r), flush=True)
        bad += not r["equal"]
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"cases": res, "invariance": inv}, f, indent=1)
    print("KDA_LAYER_CHECK_DONE", "fail" if bad else "pass", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
