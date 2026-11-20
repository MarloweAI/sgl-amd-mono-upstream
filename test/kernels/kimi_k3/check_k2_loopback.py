# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 layer launch on one GPU, against stock SGLang / AITER's A8W4
path for the same region, on one rank's real weights.

The launch runs from a loopback tree (perf-engine ``plugin_tree.py make
--model kimi_k3``): every in-kernel push writes this rank's value into every
rank's slot. So each all-reduce sums 8 copies of this rank's partial and each
gather finds this rank's slice in every slice. The reference applies the same
stub to the stock ops:

* o_proj all-reduce, latent all-reduce, final reduce: 8 x this rank's bf16
  partial (exact in bf16);
* router logits: this rank's 112 experts' logits in every rank's slice;
* latent down: this rank's 448 columns in every slice.

The experts are AITER's ``fused_moe`` as SGLang's Mxfp4MoEMethod calls it
(SiTUv2, per_1x32, gate / up interleaved), under ``AITER_SITUV2_A8W4=1``: the
stock MXFP8-activation numerics. Stock's down sums the top-k by atomics, so
the output is compared by rel-L2 and max-abs, not bit for bit. The routes and
the latent's MXFP8 words are compared exactly.

    python3 check_k2_loopback.py --tree TREE --copy ~/pe-k3/copy/<key> --layer 4
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

BF = torch.bfloat16
TP = 8
NVB = 3
MAX_BLOCKS = 8
DEADLINE_S = 30.0


def load_tree(tree):
    """Import sgl_amd_mono from a loopback tree; every PeerBuffer's peers are
    its own buffer."""
    with open(os.path.join(tree, "PLUGIN_REF.json")) as fh:
        if not json.load(fh).get("loopback"):
            raise SystemExit(f"{tree} is not a loopback tree")
    sys.path.insert(0, os.path.abspath(tree))
    import sgl_amd_mono

    if not os.path.abspath(sgl_amd_mono.__file__).startswith(os.path.abspath(tree)):
        raise SystemExit(f"sgl_amd_mono resolved to {sgl_amd_mono.__file__}")


def load_layer(copy, layer, device):
    """One layer's post-load tensors from a perf-engine weight copy (views at
    byte offsets of their storages): name -> tensor, as SGLang holds them."""
    path = os.path.join(copy, f"L{layer}.pt")
    with open(path + ".json") as fh:
        meta = json.load(fh)
    data = torch.load(path, map_location="cpu", weights_only=True)
    out = {}
    for sname, m in meta.items():
        st = data[sname].to(device)
        for name, off, shape, stride, dtype in m["views"]:
            dt = getattr(torch, dtype.split(".")[1])
            size = torch.empty(0, dtype=dt).element_size()
            out[name] = st.view(dt).as_strided(shape, stride, off // size)
    return out


def weights(copy, layer, device):
    p = f"language_model.model.layers.{layer}."
    t = load_layer(copy, layer, device)

    def g(name):
        return t[p + name]

    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import UP_N

    return {
        "w_o": g("self_attn.o_proj.weight"),
        "ares_nw": g("mlp_res_norm.weight"),
        "ares_qk": g("mlp_res_proj.weight").reshape(-1),
        "in_nw": g("post_attention_layernorm.weight"),
        "w_gate": g("mlp.gate.weight"),
        "bias": g("mlp.gate.e_score_correction_bias"),
        "w_ld": g("mlp.routed_expert_down_proj.weight"),
        "w_sgu": g("mlp.shared_experts.gate_up_proj.weight"),
        "w_sd": g("mlp.shared_experts.down_proj.weight"),
        "w13": g("mlp.experts.w13_weight"),
        "w13s": g("mlp.experts.w13_weight_scale"),
        "w2": g("mlp.experts.w2_weight"),
        "w2s": g("mlp.experts.w2_weight_scale"),
        "ln_w": g("mlp.routed_expert_norm.weight"),
        "w_up_shard": g("mlp.routed_expert_up_proj.weight")[:UP_N],
        "eps": 1e-5,
        "out_eps": 1e-5,
    }


def rms(x, weight, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def situ(g, u):
    g, u = g.float(), u.float()
    a = (2.0 * torch.tanh(g * 0.25)) * (1.0 + torch.tanh(g * 0.5))
    return (a * (25.0 * torch.tanh(u * 0.04))).to(BF)


def seam_reference(core, w, prefix_in, blocks):
    """o_proj, its loopback all-reduce (8 x the partial) and the MLP AttnRes
    seam: (the updated prefix, the MoE input)."""
    attn = F.linear(core, w["w_o"]).float() * TP
    prefix = (prefix_in.float() + attn).to(BF)
    rows = torch.cat([blocks[:, :NVB, :], prefix.unsqueeze(1)], dim=1)
    scores = (rms(rows, w["ares_nw"], w["eps"]) * w["ares_qk"].float()).sum(-1)
    probs = torch.softmax(scores, dim=-1)
    mixed = (probs.unsqueeze(-1) * rows.float()).sum(dim=1)
    return prefix, rms(mixed, w["in_nw"], w["out_eps"]).to(BF)


def routes_reference(x, w):
    """noaux_tc on the loopback logits: this rank's 112 router rows in every
    rank's slice; ranked by sigmoid + bias, weights the sigmoids renormalized."""
    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import TOPK, E

    part = x.float() @ w["w_gate"][: E // TP].float().t()
    sig = torch.sigmoid(part.repeat(1, TP))
    ids = torch.topk(sig + w["bias"], TOPK, dim=-1, sorted=True).indices
    wts = sig.gather(-1, ids)
    return ids.to(torch.int32), wts / wts.sum(-1, keepdim=True)


def latent_reference(x, w):
    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import LAT

    return F.linear(x, w["w_ld"][: LAT // TP]).repeat(1, TP)


def mxfp8_rule(lat):
    """aiter's A8W4 inline quant in torch: a 32-block's code is the ceil
    exponent of amax / 448 (at most 254), the bytes E4M3 of x / 2^(code - 127)
    -> (uint8 words [S, LAT], codes [S, LAT / 32])."""
    s, k = lat.shape
    blk = lat.float().view(s, k // 32, 32)
    amax = blk.abs().amax(-1)
    wi = (amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)).view(torch.int32)
    code = (((wi.to(torch.int64) + 0x7FFFFF) >> 23) & 0xFF).clamp(max=254)
    scale = torch.ldexp(torch.ones_like(amax), (code - 127).to(torch.int32))
    q = (blk / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).view(s, k), code.to(torch.uint8)


def experts_reference(lat, ids, wts, w):
    """AITER's fused_moe as SGLang's Mxfp4MoEMethod calls it for Kimi-K3."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    fp4 = torch.float4_e2m1fn_x2
    w13, w2 = w["w13"].view(fp4), w["w2"].view(fp4)
    w13.is_shuffled = w2.is_shuffled = True
    return fused_moe(
        lat, w13, w2, wts.float(), ids, None, ActivationType.Situv2,
        QuantType.per_1x32, False, w["w13s"], w["w2s"], None, None, dtype=BF,
        gate_mode=GateMode.INTERLEAVE.value, beta=4.0, linear_beta=25.0,
    )  # fmt: skip


def stock_noise(x, w, ids, wts):
    """Stock against itself with the latent's bf16 rounded from an fp32
    product instead of the GEMM: the rel-L2 one ulp of input order costs."""
    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import LAT

    a = experts_reference(latent_reference(x, w), ids, wts, w)
    lat = (x.float() @ w["w_ld"][: LAT // TP].float().t()).to(BF).repeat(1, TP)
    b = experts_reference(lat, ids, wts, w)
    return ((a.float() - b.float()).norm() / a.float().norm()).item()


def moe_reference(x, w, ids, wts):
    """The MoE under the loopback stub, from the experts' input routes: (the
    output, the latent RMSNorm's output)."""
    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import SI, UP_N

    y = experts_reference(latent_reference(x, w), ids, wts, w)
    lat = (y.float() * TP).to(BF)  # the latent all-reduce of 8 equal partials
    ln = rms(lat, w["ln_w"], 1e-5).to(BF)
    up = ln.float() @ w["w_up_shard"].float().t()
    gu = F.linear(x, w["w_sgu"])
    shared = F.linear(situ(gu[:, :SI], gu[:, SI:]), w["w_sd"])
    shared[:, :UP_N] = (shared[:, :UP_N].float() + up).to(BF)
    return (shared.float() * TP).to(BF), ln


def pairs(buf, off, n):
    """The values of ``n`` mailbox pairs (value, tag) at byte ``off``."""
    return buf[off : off + 8 * n].view(torch.int32).view(n, 2)[:, 0]


def stats(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return {
        "cos": F.cosine_similarity(a, b, dim=0).item(),
        "rel": ((a - b).norm() / b.norm()).item(),
        "maxabs": (a - b).abs().max().item(),
        "bitwise": bool(torch.equal(a, b)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--copy", required=True)
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--widths", type=int, nargs="+", default=[6])
    ap.add_argument("--json")
    # every width's output, the latent norm and the x rows: written to --save,
    # or required bit for bit equal to an earlier tree's --save (--against)
    ap.add_argument("--save")
    ap.add_argument("--against")
    # every width under one epoch, the warmup's order before it bumped the epoch:
    # a launch's polls can then pass on the pairs an earlier width left
    ap.add_argument("--same-epoch", action="store_true")
    args = ap.parse_args()
    assert os.environ.get("AITER_SITUV2_A8W4") == "1", "the stock A8W4 numerics"
    load_tree(args.tree)

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2
    from sgl_amd_mono.kimi_k3.gfx950.stages import moe as k2b
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN, PROJ

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    peers = PeerBuffer(k2.peer_alloc_bytes(), None, 0, 1, dev)
    addresses = torch.tensor([peers.local] * TP, dtype=torch.int64, device=dev)
    k2.init_peer(peers.bytes)
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(DEADLINE_S)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    scratch = torch.zeros(k2.scratch_alloc_bytes(), dtype=torch.uint8, device=dev)
    w = weights(args.copy, args.layer, dev)

    # One draw at the widest width, every width its first rows: a row's inputs
    # are the same at every width, so its output must be too (batch invariance).
    gen = torch.Generator(device=dev).manual_seed(11)

    def rnd(*shape, scale):
        return (torch.randn(*shape, generator=gen, device=dev) * scale).to(BF)

    top = max(args.widths)
    draw = {
        "core": rnd(top, PROJ, scale=0.3),
        "prefix": rnd(top, HIDDEN, scale=0.3),
        "blocks": rnd(top, MAX_BLOCKS, HIDDEN, scale=0.3),
    }
    results, fails, outs, bits = {}, [], {}, {}
    for rows in args.widths:
        core = draw["core"][:rows].contiguous()
        prefix_in = draw["prefix"][:rows].contiguous()
        blocks = draw["blocks"][:rows].contiguous()
        key = k2.K2Build(
            tokens=rows, nblocks=NVB, eps=w["eps"], out_eps=w["out_eps"],
            fold=False, guint=True, opt=k2.K2_OPT,
        )  # fmt: skip
        prefix, out = prefix_in.clone(), torch.empty_like(prefix_in)
        if not args.same_epoch or rows == args.widths[0]:
            epoch[0].add_(1)
        k2.k2_launch(
            key, core=core, w_o=w["w_o"], prefix=prefix, blocks=blocks,
            ares_nw=w["ares_nw"], ares_qk=w["ares_qk"], in_nw=w["in_nw"],
            w_gate=w["w_gate"], bias=w["bias"], w_ld=w["w_ld"],
            w_sgu=w["w_sgu"], w_sd=w["w_sd"], w13=w["w13"], w13s=w["w13s"],
            w2=w["w2"], w2s=w["w2s"], ln_w=w["ln_w"], w_up=w["w_up_shard"],
            out=out, scratch=scratch, peers=addresses, rank=0, epoch=epoch,
            layer=1,
        )  # fmt: skip
        torch.cuda.synchronize()
        lay = k2.scratch_layout(key)
        off, nbytes = lay["xrow"]
        xrow = scratch[off : off + nbytes].view(BF).view(rows, HIDDEN).clone()
        r = pairs(scratch, lay["route"][0], rows * 2 * k2b.TOPK).view(rows, 2, -1)
        k_ids, k_wts = r[:, 0].clone(), r[:, 1].view(torch.float32).clone()

        ref_prefix, ref_x = seam_reference(core, w, prefix_in, blocks)
        ids, wts = routes_reference(xrow, w)
        same_routes = int((k_ids == ids).all(-1).sum())
        # the experts on the kernel's routes, so a near-tie flip in one row
        # does not decide the output comparison
        ref, ref_ln = moe_reference(xrow, w, k_ids, k_wts)
        ref_own, _ = moe_reference(xrow, w, ids, wts)
        off, nbytes = lay["ln"]
        k_ln = scratch[off : off + nbytes].view(BF).view(rows, k2b.LAT)
        up_n = k2b.UP_N

        # the latent's MXFP8 words (LATQ, this parity) against the rule
        par = int(epoch[0].item()) % 2
        lq = k2.peer_layout(rows)["latq"][0] + par * (k2.peer_bytes(rows) // 2)
        words = pairs(peers.bytes, lq, rows * k2b.LQ_WORDS).view(rows, -1)
        k_q = words[:, : k2b.LAT // 4].contiguous().view(torch.uint8)
        k_c = words[:, k2b.LAT // 4 :].to(torch.uint8)
        q, codes = mxfp8_rule(latent_reference(xrow, w))
        from aiter import dtypes
        from aiter.ops.quant import per_1x32_mx_quant_hip

        aq, asc = per_1x32_mx_quant_hip(
            latent_reference(xrow, w), quant_dtype=dtypes.fp8,
            scale_type=dtypes.fp8_e8m0,
        )  # fmt: skip
        aq = aq.view(torch.uint8).reshape(rows, -1)
        asc = asc.view(torch.uint8).reshape(-1)[: rows * k2b.LAT // 32].view(rows, -1)
        res = {
            "prefix": stats(prefix, ref_prefix),
            "x": stats(xrow, ref_x),
            "routes_identical_rows": f"{same_routes}/{rows}",
            "out_vs_stock_on_kernel_routes": stats(out, ref),
            "out_vs_stock_own_routes": stats(out, ref_own),
            "latent_norm": stats(k_ln, ref_ln),
            "stock_experts_self_rel": stock_noise(xrow, w, k_ids, k_wts),
            "out_shared_only_columns": stats(out[:, up_n:], ref[:, up_n:]),
            "latq_bytes_equal_rule": float((k_q == q).float().mean()),
            "latq_codes_equal_rule": float((k_c == codes).float().mean()),
            "rule_vs_aiter_hip_quant_bytes": float((q == aq).float().mean()),
            "rule_vs_aiter_hip_quant_codes": float((codes == asc).float().mean()),
            "finite": bool(torch.isfinite(out.float()).all()),
        }
        outs[rows] = out.clone()
        bits[rows] = {"out": out.cpu(), "ln": k_ln.cpu().clone(), "x": xrow.cpu()}
        print(f"rows={rows}: {json.dumps(res)}", flush=True)
        results[rows] = res
        if not res["finite"] or res["out_vs_stock_on_kernel_routes"]["rel"] > 0.02:
            fails.append(rows)
    low = min(args.widths)
    for rows in args.widths:
        same = torch.equal(outs[rows][:low], outs[low])
        print(f"batch invariance: rows 0..{low - 1} at width {rows} bit for bit "
              f"width {low}: {same}", flush=True)  # fmt: skip
        results[rows]["batch_invariant"] = same
        if not same:
            fails.append(f"invariance {rows}")
    if args.save:
        torch.save(bits, args.save)
    if args.against:
        ref_bits = torch.load(args.against, weights_only=True)
        for rows in args.widths:
            same = {n: torch.equal(bits[rows][n], ref_bits[rows][n]) for n in bits[rows]}
            print(f"against {args.against} rows={rows}: {same}", flush=True)
            results[rows]["against"] = same
            if not all(same.values()):
                fails.append(f"against {rows}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
    print("ALL PASS" if not fails else f"FAILED {fails}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
