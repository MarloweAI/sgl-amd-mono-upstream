# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 MLA decode (``mla_decode``) on one GPU: numerics against an
fp64 reference and a B300-numerics reference, and time against aiter's decode.

A batch of 6 .. 30 requests (6 rows each) at committed lengths P (tier 0's
log-normal mix or uniform), one rank's share of their keys (random e4m3
records), e4m3 q. For each rank checked:

* the merged (row, head) outputs (direct rows, or the split partials merged
  through the CSR maps in fp64) and LSEs against fp64 attention on the same
  e4m3 q and records: rel-L2, next to the B300-numerics reference's (q K in
  fp32, a 64-key tile's running max, P = e4m3(2^(s c - m)) with P <= 1, the row
  sum from the unrounded P, P V into fp32); the bar is ours <= B300's;
* ``--time``: the graph-timed decode against aiter's (``mla_decode_fwd`` as
  SGLang's DCP verify calls it, the split reduce routed past).
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import check_mla_tail as ct  # noqa: E402

BF, F8 = torch.bfloat16, torch.float8_e4m3fn
NHA, KV, DQK, DCP, G = 96, 512, 576, 8, 6
SCALE = 192**-0.5
LOG2E = 1.4426950408889634


def contexts(n, mode, ctx, seed):
    """n committed lengths: ``ctx`` each, or tier 0's log-normal mix (perf-engine
    tier0/sgl/case.py ``mixed_lens``: p50 10K, sigma 1.3, [512, 250K], one at
    250K)."""
    import random

    if mode == "uniform":
        return [ctx] * n
    rng = random.Random(seed)
    lens = [int(min(250_000, max(512, 10_000 * math.exp(rng.gauss(0.0, 1.3))))) for _ in range(n)]
    lens[rng.randrange(n)] = 250_000
    return lens


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def visible(p, j, r):
    return max(0, (p + j - r) // DCP + 1)


def refs(q8, b):
    """fp64 and B300-numerics (o [S, 96, 512], lse [S, 96]) for rank b.r; rows
    with no key: o 0."""
    dev = q8.device
    s = q8.shape[0]
    o64 = torch.zeros(s, NHA, KV, dtype=torch.float64, device=dev)
    ob3 = torch.zeros(s, NHA, KV, dtype=torch.float64, device=dev)
    l64 = torch.full((s, NHA), -1e30, dtype=torch.float64, device=dev)
    ptr = b.kv_ptr.tolist()
    c = SCALE * LOG2E
    for i, rq in enumerate(b.reqs):
        keys = b.pool[ptr[i] : ptr[i + 1]].double()
        rows = slice(G * i, G * i + G)
        qq = q8[rows].double()  # [6, 96, 576]
        sc = torch.einsum("jhd,kd->jhk", qq, keys)  # [6, 96, L]
        L = keys.shape[0]
        for j in range(G):
            v = min(visible(rq.p, j, b.r), L)
            if v == 0:
                continue
            x = sc[j, :, :v]
            # fp64
            lse = torch.logsumexp(x * SCALE, -1)
            p = torch.exp(x * SCALE - lse[:, None])
            o64[G * i + j] = p @ keys[:v, :KV]
            l64[G * i + j] = lse
            # B300: fp32 scores, a 64-key tile's running max, e4m3 P <= 1
            t = (x.float() * c).double()
            nt = -(-v // 64)
            tm = torch.full((NHA, nt * 64), -float("inf"), dtype=torch.float64, device=dev)
            tm[:, :v] = t
            run = tm.view(NHA, nt, 64).max(-1).values.cummax(-1).values  # [96, nt]
            mk = run.repeat_interleave(64, -1)[:, :v]
            pu = torch.exp2((t - mk).float())  # fp32 P, <= 1
            pq = pu.to(F8).double()
            fin = run[:, -1:]
            w = torch.exp2(mk - fin)
            num = (pq * w) @ keys[:v, :KV]
            den = (pu.double() * w).sum(-1, keepdim=True)
            ob3[G * i + j] = num / den
    return o64, l64, ob3


def merged(out_o, out_lse, bufs, s):
    """Our (o, lse) per (row, head): the direct rows, the partial rows merged
    in fp64 through the CSR maps."""
    o = out_o.double().clone()
    lse = out_lse.double().clone()
    ip, fm, pm = bufs.indptr.tolist(), bufs.fmap.view(-1, 2).tolist(), bufs.pmap.tolist()
    lg, pl = bufs.logits.double(), bufs.plse.double()
    for i in range(len(ip) - 1):
        if ip[i] == ip[-1]:
            break
        slots = pm[ip[i] : ip[i + 1]]
        for rr, seq in enumerate(range(fm[i][0], fm[i][1])):
            rows = [sl + rr for sl in slots]
            ls = pl[rows]
            m = ls.max(0).values
            w = (ls - m).exp()
            o[seq] = (w.unsqueeze(-1) * lg[rows]).sum(0) / w.sum(0).unsqueeze(-1)
            lse[seq] = w.sum(0).log() + m
    return o, lse


def run_ours(q8, b, bufs):
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_decode as md

    s = q8.shape[0]
    o = torch.zeros(s, NHA, KV, dtype=BF, device=q8.device)
    lse = torch.zeros(s, NHA, dtype=torch.float32, device=q8.device)

    def go():
        md.mla_decode_launch(
            md.MlaDecodeBuild(scale=SCALE), q8=q8, kv=b.pool, kv_idx=b.kv_idx, kv_ptr=b.kv_ptr,
            g_ptr=b.g_kv_ptr, rank=b.r, o=o, lse=lse, bufs=bufs,
        )  # fmt: skip

    return o, lse, go


def aiter_decode(q8, b):
    """aiter's decode as SGLang's DCP verify calls it (metadata built once),
    its split reduce routed past: a callable."""
    import types

    import aiter.mla as am
    from sglang.srt.layers.attention import aiter_backend as ab

    dev = q8.device
    nreq = len(b.reqs)
    be = types.SimpleNamespace()
    be.page_size, be.kv_cache_dtype, be.dcp_world_size, be.num_head = 1, F8, DCP, NHA // DCP
    be.attn_dp_enabled, be.use_mla_auto_kv_splits = False, False
    be.max_split_per_batch = be.fix_max_split_per_batch = 32
    be.cprr_kernel_heads = ab._asm_cprr_kernel_heads(NHA, G)
    be.mla_kernel_num_head_padded = be.cprr_kernel_heads
    qo_indptr = torch.arange(nreq + 1, dtype=torch.int32, device=dev) * G
    last = torch.ones(nreq, dtype=torch.int32, device=dev)
    meta = ab.AiterAttnBackend.make_mla_decode_meta_data_buffer(
        be, G, nreq, metadata_fast_mode=True, metadata_intra_batch_mode=False,
        nhead_override=be.cprr_kernel_heads, max_split_per_batch=32,
    )  # fmt: skip
    ab.AiterAttnBackend.make_mla_meta_data(
        be, qo_indptr, b.kv_ptr, last, *meta[:1], meta[2], meta[1], *meta[3:], G,
        fast_mode=True, max_split_per_batch=32, intra_batch_mode=False,
        is_cp_round_robin=True, nhead_override=be.cprr_kernel_heads,
    )  # fmt: skip
    out = torch.empty(q8.shape[0], NHA, KV, dtype=BF, device=dev)
    one = torch.ones((), dtype=torch.float32, device=dev)
    info = {}

    def no_reduce(logits, plse, indptr, fmap, pmap, *a):
        info["partial_rows"] = logits.view(-1, NHA, KV).shape[0]

    def go():
        old = am._mla_decode_reduce_v1_dispatch
        am._mla_decode_reduce_v1_dispatch = no_reduce
        try:
            ab.mla_decode_fwd(
                q8, b.pool.view(-1, 1, 1, DQK), out, qo_indptr, b.kv_ptr, b.kv_idx, last, G,
                sm_scale=SCALE, work_meta_data=meta[0], work_indptr=meta[1], work_info_set=meta[2],
                reduce_indptr=meta[3], reduce_final_map=meta[4], reduce_partial_map=meta[5],
                q_scale=one, kv_scale=one, intra_batch_mode=False, num_kv_splits=32, return_lse=True,
                g_kv_indptr=b.g_kv_ptr, cp_world_size=DCP, cp_rank=b.r, causal=True,
            )  # fmt: skip
        finally:
            am._mla_decode_reduce_v1_dispatch = old

    return go, info


def case(ps, args, dev, bufs):
    s = G * len(ps)
    reqs = [ct.Req(p, args.seed * 1000 + 7 * i, dev) for i, p in enumerate(ps)]
    g = torch.Generator(device=dev).manual_seed(args.seed + 5)
    q = torch.randn(s, NHA, DQK, generator=g, device=dev) * args.qstd
    q8 = q.clamp(-448, 448).to(F8).contiguous()
    worst = 0.0
    out = {}
    for r in args.ranks:
        b = ct.RankBatch(reqs, r, dev)
        o, lse, go = run_ours(q8, b, bufs)
        go()
        torch.cuda.synchronize()
        if args.check:
            mo, ml = merged(o, lse, bufs, s)
            o64, l64, ob3 = refs(q8, b)
            has = l64 > -1e29
            e_ours, e_b3 = rel(mo[has], o64[has]), rel(ob3[has], o64[has])
            e_lse = (ml[has] - l64[has]).abs().max().item()
            ratio = e_ours / max(e_b3, 1e-30)
            worst = max(worst, ratio)
            nsp = int(bufs.indptr[-1])
            print(f"  rank {r}: o rel-L2 to fp64 ours {e_ours:.3e} B300-numerics {e_b3:.3e} (ratio {ratio:.3f}); "
                  f"max |lse - ref| {e_lse:.2e}; splits with partials {nsp}", flush=True)  # fmt: skip
        if args.time and r == args.ranks[0]:
            t_ours = ct.graph_us(go)
            ago, info = aiter_decode(q8, b)
            t_aiter = ct.graph_us(ago)
            out["time"] = (t_ours, t_aiter)
            print(f"  time rank {r}: ours {t_ours:.1f} us, aiter decode {t_aiter:.1f} us "
                  f"(aiter partial rows {info.get('partial_rows')}, ours {int(bufs.indptr[-1]) * G})", flush=True)
    return worst, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", type=int, nargs="+", default=[9, 16, 24, 30])
    ap.add_argument("--mode", default="mixed", choices=["mixed", "uniform"])
    ap.add_argument("--ctx", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranks", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--qstd", type=float, default=0.5, help="q's std (4: a peaky softmax)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--compile", action="store_true", help="compile-only build (no GPU)")
    args = ap.parse_args()
    if args.compile:
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_decode as md

        md.build(md.MlaDecodeBuild(scale=SCALE))(*([0] * 14), stream=0)
        print("COMPILE_OK", flush=True)
        return
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_decode as md

    dev = torch.device("cuda")
    bufs = md.DecodeBuffers(dev)
    worst = 0.0
    for n in args.reqs:
        ps = contexts(n, args.mode, args.ctx, args.seed)
        print(f"case {args.mode} {n} requests (keys a rank ~{sum(math.ceil(p / 8) for p in ps)}), q std {args.qstd}", flush=True)
        w, _ = case(ps, args, dev, bufs)
        worst = max(worst, w)
    if args.check:
        print(f"numerics: worst ours / B300-numerics rel-L2 {worst:.3f} {'OK' if worst <= 1.0 else 'ABOVE'}")


if __name__ == "__main__":
    main()
