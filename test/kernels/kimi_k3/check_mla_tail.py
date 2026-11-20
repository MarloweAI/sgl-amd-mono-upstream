# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 MLA tail (``mla_tail``) against SGLang's DCP8 verify attention
past the decode kernel, on one GPU.

A batch of 6 .. 30 requests (6 rows each) at committed lengths P, every rank's
share of their keys (random e4m3 records) and one q for all ranks. For each
rank, aiter's cp round-robin decode runs as SGLang calls it
(``_forward_verify_dcp`` on a stand-in backend), its split partials taken
before the split reduce. On those partials:

* ``reduce``: the tail's reduced rows and LSEs against aiter's reduce, bit for
  bit, every row a tile covers;
* ``core``: the eight ranks' tails one after another (three passes, so every
  combine reads real records) against SGLang's chain on the same decode
  outputs (fills, all-to-all as a copy, LSE combine, W_UV, output gate), both
  against an fp64 reference of the same partials: ``|tail - ref| <= 1.25
  |stock - ref|``; and the last two passes bit for bit;
* ``--invariance``: each request of a batch alone with others at different
  places: its core rows bit for bit the same.
"""

import argparse
import math
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(__file__))
import check_mla_front as cf  # noqa: E402

BF, F8 = torch.bfloat16, torch.float8_e4m3fn
NHA, NHL, KV, VD, DQK, DCP, G = 96, 12, 512, 128, 576, 8, 6
GATE = NHL * VD
SCALE = 192**-0.5


def contexts(n, mode, ctx, seed):
    """n committed lengths: ``ctx`` each (uniform), or tier 0's log-normal mix
    (p50 10K, sigma 1.3, one request at 250K)."""
    if mode == "uniform":
        return [ctx] * n
    g = torch.Generator().manual_seed(seed)
    ps = (torch.randn(n, generator=g) * 1.3 + math.log(10000)).exp().clamp(64, 250000)
    ps = [int(x) for x in ps]
    ps[0] = 250000
    return ps


class Req:
    """One request's keys on every rank: rank r's local rows (random records)."""

    def __init__(self, p, seed, device):
        self.p, self.seed = p, seed
        g = torch.Generator(device=device).manual_seed(seed)
        self.keys = (torch.randn(p + G, DQK, generator=g, device=device) * 0.6).to(F8)


class RankBatch:
    """Rank r's share of the requests: one pool, SGLang's ragged verify index
    list (the rank's prefix keys, then its own verify row), the indptrs."""

    def __init__(self, reqs, r, device):
        from sglang.srt.layers.dcp.layout import get_dcp_chain_spec_lens

        self.r, self.reqs, self.s = r, reqs, G * len(reqs)
        rows, ptr, gptr = [], [0], [0]
        for q in reqs:
            lr = q.p // DCP + int(r < q.p % DCP)
            own = (r - q.p) % DCP
            nloc = lr + int(own < G)
            rows.append(q.keys[: q.p][r::DCP][:lr])
            if own < G:
                rows.append(q.keys[q.p + own : q.p + own + 1])
            ptr.append(ptr[-1] + nloc)
            gptr.append(gptr[-1] + q.p + G)
        self.pool = torch.cat(rows + [torch.zeros(1, DQK, dtype=F8, device=device)])
        self.kv_idx = torch.arange(ptr[-1], dtype=torch.int32, device=device)
        self.kv_ptr = torch.tensor(ptr, dtype=torch.int32, device=device)
        self.g_kv_ptr = torch.tensor(gptr, dtype=torch.int32, device=device)
        glens = torch.tensor([q.p + G for q in reqs], device=device)
        self.local = get_dcp_chain_spec_lens(glens, G, DCP, r)


_CAPTURE = {}


def _capture_reduce():
    """aiter's reduce dispatch, its inputs taken first (``_CAPTURE``)."""
    import aiter.mla as am

    if "orig" in _CAPTURE:
        return
    orig = am._mla_decode_reduce_v1_dispatch

    def dispatch(logits, plse, indptr, fmap, pmap, max_q, nsplits, o, lse):
        _CAPTURE["args"] = dict(
            logits=logits.clone(), plse=plse.clone(), indptr=indptr.clone(),
            fmap=fmap.clone(), pmap=pmap.clone(), o=o.clone(), lse=lse.clone(),
            o_ptr=o.data_ptr(),
        )  # fmt: skip
        orig(logits, plse, indptr, fmap, pmap, max_q, nsplits, o, lse)
        _CAPTURE["args"].update(o_red=o.clone(), lse_red=lse.clone())

    _CAPTURE["orig"] = orig
    am._mla_decode_reduce_v1_dispatch = dispatch


def stock_decode(q, b):
    """Rank b.r's ``_forward_verify_dcp``: (out, lse) after SGLang's fills and
    the decode's partials, outputs and reduce outputs (``_CAPTURE``)."""
    from sglang.srt.layers.attention import aiter_backend as ab

    _capture_reduce()
    dev = q.device
    nreq = len(b.reqs)
    be = types.SimpleNamespace()
    be.page_size, be.kv_cache_dtype, be.dcp_world_size, be.num_head = 1, F8, DCP, NHL
    be.attn_dp_enabled, be.use_mla_auto_kv_splits = False, False
    be.max_split_per_batch = be.fix_max_split_per_batch = 32
    be.cprr_fast_mode, be.cprr_intra_batch_mode = True, False
    be.cprr_kernel_heads = ab._asm_cprr_kernel_heads(NHA, G)
    assert be.cprr_kernel_heads == NHA, be.cprr_kernel_heads
    be.mla_kernel_num_head_padded = be.cprr_kernel_heads
    be._cprr_launch_logged, be.input_dtype = True, BF
    be.pad_heads = types.MethodType(ab.AiterAttnBackend.pad_heads, be)
    be._forward_verify_asm_cprr = types.MethodType(ab.AiterAttnBackend._forward_verify_asm_cprr, be)
    be._kernel_num_kv_splits = lambda: 32
    be._resolve_fp8_kv_scale_float = lambda layer, k_descale: 1.0
    kv = b.pool
    be.kv_layers = types.SimpleNamespace(read=lambda i: types.SimpleNamespace(key=lambda: kv))
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
    be.forward_metadata = types.SimpleNamespace(
        max_q_len=G, local_kv_lens=b.local, use_asm_cprr_verify=True,
        cprr_kernel_heads=be.cprr_kernel_heads, cprr_kv_indices=b.kv_idx,
        cprr_kv_indptr=b.kv_ptr, cprr_empty_rows=b.local.eq(0), qo_indptr=qo_indptr,
        kv_last_page_len=last, work_metadata=meta[0], work_indptr=meta[1],
        work_info_set=meta[2], reduce_indptr=meta[3], reduce_final_map=meta[4],
        reduce_partial_map=meta[5], global_kv_indptr=b.g_kv_ptr, num_kv_splits=32,
    )  # fmt: skip
    layer = types.SimpleNamespace(tp_q_head_num=NHA, qk_head_dim=DQK, v_head_dim=KV, scaling=SCALE, layer_id=0)
    par = types.SimpleNamespace(attn_dcp_rank=b.r)
    old = ab.get_parallel
    ab.get_parallel = lambda: par
    try:
        out, lse = ab.AiterAttnBackend._forward_verify_dcp(be, q, None, layer, None)
    finally:
        ab.get_parallel = old
    cap = dict(_CAPTURE.pop("args"))
    assert cap["o_ptr"] == out.data_ptr(), "the reduce's output is not the decode's (a head transform)"
    cap.update(out=out.clone(), lse_out=lse.clone(), empty=b.local.eq(0).contiguous())
    return cap


def stock_core(caps, w_vc, gates, s):
    """SGLang's chain past the decode on every rank's (filled) outputs: the
    all-to-all pack (a copy for the exchange), LSE combine, W_UV, output gate."""
    from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton, dcp_pack_a2a_send
    from sglang.kernels.ops.elementwise import mla_output_gate

    sends = []
    for r in range(DCP):
        send = torch.empty(DCP, s, NHL, KV + 2, dtype=BF, device=w_vc.device)
        words = send.view(torch.float32)
        dcp_pack_a2a_send(caps[r]["out"].view(s, NHA, KV), caps[r]["lse_out"].view(s, NHA).contiguous(),
                          send[..., :KV], words[..., KV // 2])  # fmt: skip
        sends.append(send)
    outs = []
    for o in range(DCP):
        recv = torch.stack([sends[src][o] for src in range(DCP)])
        comb, _ = dcp_lse_combine_triton(
            recv[..., :KV].contiguous(), recv.view(torch.float32)[..., KV // 2].contiguous(),
            is_lse_base_on_e=True,
        )
        proj = torch.empty(s, NHL, VD, dtype=BF, device=w_vc.device)
        torch.bmm(comb.view(s, NHL, KV).transpose(0, 1), w_vc[NHL * o : NHL * (o + 1)], out=proj.transpose(0, 1))
        outs.append(mla_output_gate.kimi_k3_mla_output_gate(proj.reshape(s, GATE).contiguous(), gates[o]))
    return outs


def ref_core(caps, w_vc, gates, s):
    """fp64 of the same partials: each rank's split reduce, SGLang's fills, the
    LSE combine over ranks, W_UV and the gate."""
    lat, lses = [], []
    for r in range(DCP):
        c = caps[r]
        o = c["o"].double().view(s, NHA, KV).clone()
        lse = c["lse"].double().view(s, NHA).clone()
        lg = c["logits"].double().view(-1, NHA, KV)
        pl = c["plse"].double().view(-1, NHA)
        ip, fm, pm = c["indptr"].tolist(), c["fmap"].view(-1, 2).tolist(), c["pmap"].tolist()
        lastp = ip[-1]
        for i in range(len(ip) - 1):
            if ip[i] == lastp or ip[i + 1] <= ip[i]:
                continue
            slots = pm[ip[i] : ip[i + 1]]
            for seq in range(fm[i][0], fm[i][1]):
                rows = [sl + seq - fm[i][0] for sl in slots]
                ls = pl[rows]  # [n, 96]
                m = ls.max(0).values
                wgt = (ls - m).exp()
                o[seq] = (wgt.unsqueeze(-1) * lg[rows]).sum(0) / wgt.sum(0).unsqueeze(-1)
                lse[seq] = wgt.sum(0).log() + m
        em = c["empty"]
        o[em] = 0
        lse[em] = -1e30
        lat.append(o)
        lses.append(lse)
    lat = torch.stack(lat)  # [8, S, 96, 512]
    lses = torch.stack(lses)  # [8, S, 96]
    m = lses.max(0).values
    wgt = (lses - m).exp()
    comb = (wgt.unsqueeze(-1) * lat).sum(0) / wgt.sum(0).unsqueeze(-1)  # [S, 96, 512]
    proj = torch.einsum("shk,hkv->shv", comb, w_vc.double())  # [S, 96, 128]
    outs = []
    for o in range(DCP):
        x = proj[:, NHL * o : NHL * (o + 1)].reshape(s, GATE)
        outs.append(x * torch.sigmoid(gates[o].double()))
    return outs


class TailPeers:
    """The eight ranks' peer buffers in one uncached allocation, ``mla_tail``'s
    A2A copies armed."""

    def __init__(self, device):
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_combine as mc
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_tail as mt

        self.mt, self.mc = mt, mc
        self.stride = mt.peer_end()
        self.buf = PeerBuffer(DCP * self.stride, None, 0, 1, device)
        self.buf.bytes.zero_()
        self.rearm()
        self.table = torch.tensor([self.buf.local + o * self.stride for o in range(DCP)], dtype=torch.int64, device=device)

    def a2a(self, o, s, par):
        off = o * self.stride + self.mt.peer_offset() + par * self.mt.a2a_copy_bytes()
        return self.buf.bytes[off : off + self.mc.a2a_cells(s) * 8].view(torch.int32).view(DCP, s, NHL, self.mc.REC_CELLS, 2)

    def stand_in(self, o, s, par, srcs):
        a = self.a2a(o, s, par)
        for src in srcs:
            a[src, ..., : self.mc.LSE_CELL + 1, :] = 0

    def rearm(self):
        for o in range(DCP):
            self.buf.bytes[o * self.stride + self.mt.peer_offset() : o * self.stride + self.mt.peer_end()].fill_(0xFF)


def tail_once(cap, w_vc, gate, r, peers, epoch, par, layer=3, records=False):
    """Rank r's tail on its decode capture: (o, lse, core, send)."""
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_tail as mt

    s = cap["o"].shape[0]
    key = mt.MlaTailBuild(par=par, records=records)
    dev = w_vc.device
    scratch = PeerBuffer(mt.scratch_bytes(key), None, 0, 1, dev).bytes
    scratch.zero_()
    o, lse = cap["o"].clone(), cap["lse"].clone().view(s, NHA)
    core = torch.zeros(s, GATE, dtype=BF, device=dev)
    send = torch.zeros(DCP, s, NHL, VD + 2, dtype=BF, device=dev) if records else None
    epoch[0].add_(1)
    mt.mla_tail_launch(
        key, s=s, logits=cap["logits"], plse=cap["plse"], indptr=cap["indptr"], fmap=cap["fmap"],
        pmap=cap["pmap"], o=o, lse=lse, empty=cap["empty"], w_vc=w_vc, gate=gate, core=core,
        scratch=scratch, peers=peers.table, rank=r, layer=layer, epoch=epoch, send=send,
    )  # fmt: skip
    torch.cuda.synchronize()
    return o, lse, core, send


def run_ranks(caps, w_vc, gates, peers, epoch, par):
    """The eight ranks' tails one after another, three passes (see the front's
    check): the last pass's cores and the second's."""
    s = caps[0]["o"].shape[0]
    for o in range(DCP):
        peers.stand_in(o, s, par, range(o + 1, DCP))
    outs = []
    for _ in range(3):
        outs.append([tail_once(caps[r], w_vc, gates[r], r, peers, epoch, par) for r in range(DCP)])
    peers.rearm()
    return outs[1], outs[2]


def covered_rows(cap):
    ip, fm = cap["indptr"].tolist(), cap["fmap"].view(-1, 2).tolist()
    rows = set()
    for i in range(len(ip) - 1):
        if ip[i] != ip[-1] and ip[i + 1] > ip[i]:
            rows.update(range(fm[i][0], fm[i][1]))
    return rows


def diag_reduce(cap, o, lse, s):
    """Rank 0's mismatches by the tile's split count: rows, LSE ulps."""
    ip, fm = cap["indptr"].tolist(), cap["fmap"].view(-1, 2).tolist()
    ob = o.view(s, NHA, KV).view(torch.int16)
    sb = cap["o_red"].view(s, NHA, KV).view(torch.int16)
    lb = lse.view(s, NHA).view(torch.int32)
    sl = cap["lse_red"].view(s, NHA).view(torch.int32)
    by = {}
    for i in range(len(ip) - 1):
        if ip[i] == ip[-1] or ip[i + 1] <= ip[i]:
            continue
        n = ip[i + 1] - ip[i]
        for seq in range(fm[i][0], fm[i][1]):
            d = by.setdefault(n, [0, 0, 0, 0, 0])
            oe = (ob[seq] == sb[seq]).all(-1)
            le = lb[seq] == sl[seq]
            d[0] += NHA
            d[1] += int(oe.sum())
            d[2] += int(le.sum())
            d[3] = max(d[3], int((lb[seq] - sl[seq]).abs().max()))
            d[4] = max(d[4], int((ob[seq].int() - sb[seq].int()).abs().max()))
    mod = {}
    for i in range(len(ip) - 1):
        if ip[i] == ip[-1] or ip[i + 1] <= ip[i]:
            continue
        n = ip[i + 1] - ip[i]
        for seq in range(fm[i][0], fm[i][1]):
            dd = (ob[seq] != sb[seq]).nonzero()
            for h, k in dd.tolist():
                key = (n, k % 4)
                mod[key] = mod.get(key, 0) + 1
    print(f"    mismatches by (splits, dim % 4): {sorted(mod.items())}")
    for n in sorted(by):
        d = by[n]
        print(f"    splits {n}: {d[0]} (row, head): o equal {d[1]}, lse equal {d[2]}, lse max ulps {d[3]}, o max ulps {d[4]}")


def case(ps, args, device, peers, epoch, w_vc):
    s = G * len(ps)
    reqs = [Req(p, args.seed * 1000 + 7 * i, device) for i, p in enumerate(ps)]
    g = torch.Generator(device=device).manual_seed(args.seed + 5)
    q = torch.randn(s, NHA, DQK, generator=g, device=device).to(BF)
    gates = [torch.randn(s, GATE, generator=g, device=device).to(BF) for _ in range(DCP)]
    caps = [stock_decode(q, RankBatch(reqs, r, device)) for r in range(DCP)]
    fails = []
    # the reduce, bit for bit, every rank's covered rows
    n_eq = n_all = 0
    for r in range(DCP):
        cap = caps[r]
        # the other ranks' records stand in (their tails run later)
        peers.stand_in(r, s, args.par, [x for x in range(DCP) if x != r])
        o, lse, _, _ = tail_once(cap, w_vc, gates[r], r, peers, epoch, args.par)
        cov = sorted(covered_rows(cap))
        idx = torch.tensor(cov, device=device, dtype=torch.long)
        a = o.view(torch.int16)[idx]
        b = cap["o_red"].view(s, NHA, KV).view(torch.int16)[idx]
        la = lse.view(torch.int32)[idx]
        lb = cap["lse_red"].view(s, NHA).view(torch.int32)[idx]
        n_eq += int((a == b).all(-1).sum()) + int((la == lb).sum())
        n_all += a.shape[0] * NHA * 2
        if r == 0:
            print(f"  rank 0: {len(cov)}/{s} rows in reduce tiles, tiles {int((cap['indptr'][1:] > cap['indptr'][:-1]).sum())}"
                  f"/{cap['fmap'].numel() // 2}, partial rows {cap['logits'].shape[0]}")  # fmt: skip
        if r == 0 and args.diag:
            diag_reduce(cap, o, lse, s)
        if not (bool((a == b).all()) and bool((la == lb).all())):
            bad = (a != b).any(-1).nonzero()[:4].tolist()
            print(f"  rank {r}: reduce differs at (row, head) {bad}; lse equal {(la == lb).float().mean().item():.4f}")
    peers.rearm()
    ok_red = n_eq == n_all
    print(f"reduce: {n_eq}/{n_all} (row, head) outputs and LSEs bit for bit {'ok' if ok_red else 'FAIL'}")
    if not ok_red:
        fails.append("reduce")
    # the core on eight ranks
    stock = stock_core(caps, w_vc, gates, s)
    ref = ref_core(caps, w_vc, gates, s)
    p2, p3 = run_ranks(caps, w_vc, gates, peers, epoch, args.par)
    worst = 0.0
    for o in range(DCP):
        e_t, e_s = cf.rel(p3[o][2], ref[o]), cf.rel(stock[o], ref[o])
        worst = max(worst, e_t / max(e_s, 1e-30))
        if o == 0:
            print(f"  rank 0 core rel to fp64: tail {e_t:.3e} stock {e_s:.3e}; tail vs stock {cf.rel(p3[o][2], stock[o]):.3e}")
    same = all(torch.equal(p2[o][2], p3[o][2]) for o in range(DCP))
    ok = worst <= 1.25 and same
    print(f"core: worst rank tail/stock error {worst:.3f} (bound 1.25), passes bit for bit {same} {'ok' if ok else 'FAIL'}")
    if not ok:
        fails.append("core")
    return fails, caps, gates, p3


def graph_us(fn, reps=20):
    """Mean us of ``fn`` replayed in a CUDA graph of ``reps`` calls."""
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (5 * reps)


def time_case(caps, w_vc, gates, s, peers, epoch, par):
    """Rank 0's tail (the other ranks' records stood in as zero words before
    each launch, that store's time taken off) against SGLang's chain past the
    decode on the same outputs: reduce, fills, pack, the all-to-all as a copy,
    LSE combine, W_UV, gate."""
    from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton, dcp_pack_a2a_send
    from sglang.kernels.ops.elementwise import mla_output_gate
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_tail as mt
    import aiter.mla as am

    cap, dev = caps[0], w_vc.device
    key = mt.MlaTailBuild(par=par)
    scratch = PeerBuffer(mt.scratch_bytes(key), None, 0, 1, dev).bytes
    scratch.zero_()
    o, lse = cap["o"].clone(), cap["lse"].clone().view(s, NHA)
    core = torch.zeros(s, GATE, dtype=BF, device=dev)
    own = peers.a2a(0, s, par)

    def stand_in():
        own[1:, ..., : peers.mc.LSE_CELL + 1, :].zero_()

    def tail():
        stand_in()
        epoch[0].add_(1)
        mt.mla_tail_launch(
            key, s=s, logits=cap["logits"], plse=cap["plse"], indptr=cap["indptr"],
            fmap=cap["fmap"], pmap=cap["pmap"], o=o, lse=lse, empty=cap["empty"], w_vc=w_vc,
            gate=gates[0], core=core, scratch=scratch, peers=peers.table, rank=0, layer=3,
            epoch=epoch,
        )  # fmt: skip

    def base():
        stand_in()
        epoch[0].add_(1)

    t_tail = graph_us(tail) - graph_us(base)
    peers.rearm()
    reduce = _CAPTURE["orig"]
    so, sl = cap["o"].clone(), cap["lse"].clone()
    send = torch.empty(DCP, s, NHL, KV + 2, dtype=BF, device=dev)
    recv = torch.empty_like(send)
    em = cap["empty"]
    proj = torch.empty(s, NHL, VD, dtype=BF, device=dev)

    def stock():
        reduce(cap["logits"], cap["plse"], cap["indptr"], cap["fmap"], cap["pmap"], G, 32, so, sl)
        ov, lv = so.view(s, NHA, KV), sl.view(s, NHA)
        ov.masked_fill_(em[:, None, None], 0.0)
        lv.masked_fill_(em[:, None], -1e30)
        dcp_pack_a2a_send(ov, lv, send[..., :KV], send.view(torch.float32)[..., KV // 2])
        recv.copy_(send)
        comb, _ = dcp_lse_combine_triton(recv[..., :KV], recv.view(torch.float32)[..., KV // 2], is_lse_base_on_e=True)
        torch.bmm(comb.view(s, NHL, KV).transpose(0, 1), w_vc[:NHL], out=proj.transpose(0, 1))
        mla_output_gate.kimi_k3_mla_output_gate(proj.view(s, GATE), gates[0])

    def stock_reduce():
        reduce(cap["logits"], cap["plse"], cap["indptr"], cap["fmap"], cap["pmap"], G, 32, so, sl)

    t_stock, t_red = graph_us(stock), graph_us(stock_reduce)
    print(f"time {s // G} requests: tail {t_tail:.1f} us, stock chain {t_stock:.1f} us (its reduce {t_red:.1f})")
    return t_tail, t_stock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", type=int, nargs="+", default=[9])
    ap.add_argument("--mode", choices=("mixed", "uniform"), default="mixed")
    ap.add_argument("--ctx", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--par", type=int, default=0, choices=(0, 1))
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--time", action="store_true", help="time rank 0's tail against SGLang's chain")
    args = ap.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    g = torch.Generator(device=device).manual_seed(args.seed + 11)
    w_vc = (torch.randn(NHA, KV, VD, generator=g, device=device) * 0.045).to(BF)
    peers = TailPeers(device)
    epoch = cf.new_epoch(device)
    fails = []
    for n in args.reqs:
        ps = contexts(n, args.mode, args.ctx, args.seed)
        print(f"== {n} requests ({args.mode}): P = {ps[:8]}{' ...' if n > 8 else ''}")
        f, caps, gates, _ = case(ps, args, device, peers, epoch, w_vc)
        fails += [f"{n}:{x}" for x in f]
        if args.time:
            time_case(caps, w_vc, gates, G * n, peers, epoch, args.par)
    print("FAILS " + " ".join(fails) if fails else "ALL OK")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
