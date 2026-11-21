# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's fused o_proj all-reduce + FFN bank read (``kimi_k3/gfx950/stages/ar_read.py``):
check, and time against SGLang's pair (aiter's custom all-reduce + ``attn_res_hip``).

Check (both modes): every rank's ``pre_out`` against SGLang's prefix from a rank-order f32
all-reduce (bit for bit), and ``out`` against SGLang's ``attn_res_hip`` on that sum (mismatch
count) and against an f64 reference with the sum unrounded (relative L2 of both). Three input
sets rotate between graph replays, so a launch never passes on an earlier launch's data.

    # one GPU: the 8 ranks in one launch
    python ar_read_bench.py --emulate --rows 54,96,168
    # TP8: check + time (fused, stock pair, its parts, kimi_k31 coll.py's AR, the MoE AR,
    # the column tail's LL gather + read)
    torchrun --nproc-per-node 8 ar_read_bench.py --rows 54,96,168 --time
"""

import argparse
import json
import os

import torch
import torch.distributed as dist

TP = 8
H = 7168
NBK = 8
SETS = 3
BF = torch.bfloat16


def make_inputs(m, s, dev):
    g = torch.Generator(device=dev).manual_seed(1234 + 17 * m + s)

    def rn(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=dev) * scale

    return {
        "x": rn(TP, m, H, scale=0.3).to(BF),  # every rank's o_proj partial
        "res": rn(m, H).to(BF),
        "bank": rn(m, NBK, H).to(BF),
    }


def make_weights(dev):
    g = torch.Generator(device=dev).manual_seed(99)
    cw = (torch.randn(H, generator=g, device=dev) * 0.05).float().contiguous()
    w_o = (1.0 + 0.1 * torch.randn(H, generator=g, device=dev)).to(BF).contiguous()
    return cw, w_o


def ar_rank_order(x):
    acc = x[0].float()
    for r in range(1, TP):
        acc = acc + x[r].float()
    return acc.to(BF)


def stock_read(ar, res, bank, nvb, cw, w_o, eps):
    """SGLang's FFN read on the all-reduced ``ar`` (attn_residual._aggregate_hip's call)."""
    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip

    out = torch.empty_like(ar)
    if res is None:
        attn_res_hip(ar, bank, cw, w_o, out, nvb, eps, eps)
        return out, ar
    pre = torch.empty_like(ar)
    attn_res_hip(res, bank, cw, w_o, out, nvb, eps, eps, addend=ar, prefix_out=pre)
    return out, pre


def ref64(x, res, bank, nvb, cw, w_o, eps):
    s = x.double().sum(0)
    pre = s if res is None else res.double() + s
    rows = torch.cat([bank[:, :nvb].double(), pre.unsqueeze(1)], dim=1)
    score = (rows * cw.double()).sum(-1) / torch.sqrt((rows * rows).mean(-1) + eps)
    p = torch.softmax(score, dim=-1)
    mixed = (p.unsqueeze(-1) * rows).sum(1)
    out = mixed / torch.sqrt((mixed * mixed).mean(-1, keepdim=True) + eps) * w_o.double()
    return out


def rel_l2(a, ref):
    a = a.double()
    return float((a - ref).norm() / ref.norm())


def cases(rows):
    """(m, nvb, has_res) per width: a write layer's read and a later one, few and many snapshots."""
    out = []
    for m in rows:
        out += [(m, 1, False), (m, 4, True), (m, 8, True)]
    return out


def check_case(launch_one, m, nvb, has_res, dev, eps, ranks, graph_launches=4):
    """Run the fused launch for ``ranks`` (one or all emulated); returns stats."""
    cw, w_o = make_weights(dev)
    sets = [make_inputs(m, s, dev) for s in range(SETS)]
    refs = []
    for st in sets:
        ar = ar_rank_order(st["x"])
        res = st["res"] if has_res else None
        so, sp = stock_read(ar, res, st["bank"], nvb, cw, w_o, eps)
        refs.append((so, sp, ref64(st["x"], res, st["bank"], nvb, cw, w_o, eps)))
    cur = {k: torch.empty_like(v) for k, v in sets[0].items()}
    outs = launch_one(cur, m, nvb, has_res, cw, w_o, graph_launches)
    stats = {"pre_bad": 0, "out_bad": 0, "rel_fused": 0.0, "rel_stock": 0.0, "launches": 0}
    for rep in range(2 * SETS):
        k = rep % SETS
        for name, v in sets[k].items():
            cur[name].copy_(v)
        outs["replay"]()
        torch.cuda.synchronize()
        so, sp, r64 = refs[k]
        for r in ranks:
            for i in range(graph_launches):
                o, p = outs["out"](r, i), outs["pre"](r, i)
                stats["pre_bad"] += int((p.view(torch.int16) != sp.view(torch.int16)).sum())
                stats["out_bad"] += int((o.view(torch.int16) != so.view(torch.int16)).sum())
                stats["rel_fused"] = max(stats["rel_fused"], rel_l2(o, r64))
                stats["launches"] += 1
        stats["rel_stock"] = max(stats["rel_stock"], rel_l2(so, r64))
    return stats


def emulate_main(args):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.stages import ar_read as K

    dev = torch.device("cuda", 0)
    blocks = 32
    nl = 4
    launch = K.build(args.eps, args.eps, nl, blocks=blocks, emulate=True)
    bufs = [PeerBuffer(K.peer_bytes(), None, 0, 1, dev, init=K.arm) for _ in range(TP)]
    table = torch.tensor([b.local for b in bufs], dtype=torch.int64, device=dev)
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(args.deadline)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    results = []
    for m, nvb, has_res in cases(args.rows):
        def launch_one(cur, m, nvb, has_res, cw, w_o, gl):
            mr = K.MAX_ROWS
            xs = torch.zeros(gl, TP, mr, H, dtype=BF, device=dev)
            pres = torch.zeros(gl, TP, mr, H, dtype=BF, device=dev)
            outs = torch.zeros(gl, TP, mr, H, dtype=BF, device=dev)

            def body():
                epoch[0:1].add_(1)
                for i in range(gl):
                    xs[i, :, :m].copy_(cur["x"])
                    launch(
                        xs[i].data_ptr(), cur["res"].data_ptr(), cur["bank"].data_ptr(),
                        cw.data_ptr(), w_o.data_ptr(), pres[i].data_ptr(), outs[i].data_ptr(),
                        m, nvb, NBK, int(has_res), table.data_ptr(), 0, epoch.data_ptr(), i,
                        stream=torch.cuda.current_stream(),
                    )  # fmt: skip

            body()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            return {
                "replay": g.replay,
                "out": lambda r, i: outs[i, r, :m],
                "pre": lambda r, i: pres[i, r, :m],
            }

        st = check_case(launch_one, m, nvb, has_res, dev, args.eps, range(TP), nl)
        st.update(m=m, nvb=nvb, has_res=has_res)
        results.append(st)
        print(f"EMU m={m:3d} nvb={nvb} res={int(has_res)} pre_bad={st['pre_bad']} "
              f"out_bad={st['out_bad']} rel_fused={st['rel_fused']:.3e} rel_stock={st['rel_stock']:.3e}",
              flush=True)
    err = [int(w) for w in epoch[: guard.GUARD_WORDS].tolist()]
    print("EMU guard words", err, flush=True)
    moe_ok = emulate_moe(args, dev)
    ok = moe_ok and all(r["pre_bad"] == 0 and r["rel_fused"] <= 1.05 * r["rel_stock"] + 1e-6 for r in results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"mode": "emulate", "cases": results}, f, indent=1)
    print("EMU", "PASS" if ok else "FAIL", flush=True)
    return ok


LAT, SHR = 3584, 7168


def moe_inputs(m, s, dev):
    g = torch.Generator(device=dev).manual_seed(777 + 13 * m + s)
    xl = (torch.randn(TP, m, LAT, generator=g, device=dev) * 0.3).to(BF)
    xs = (torch.randn(TP, m, SHR, generator=g, device=dev) * 0.3).to(BF)
    return xl, xs


def moe_refs(xl, xs, w, eps):
    """(latent sum, shared sum) in rank order, SGLang's RMSNorm of the latent sum, and its f64 norm."""
    al, as_ = ar_rank_order(xl), ar_rank_order(xs)
    n = stock_rmsnorm(LAT, eps, w)(al.clone())
    a64 = xl.double().sum(0)
    r64 = a64 / torch.sqrt((a64 * a64).mean(-1, keepdim=True) + eps) * w.double()
    return al, as_, n, r64


def emulate_moe(args, dev):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.stages import ar_norm as N

    blocks, mr = 32, N.MAX_ROWS
    launch = N.build_moe(args.eps, 2, blocks=blocks, emulate=True)
    bufs = [PeerBuffer(N.moe_peer_bytes(), None, 0, 1, dev, init=N.moe_arm) for _ in range(TP)]
    table = torch.tensor([b.local for b in bufs], dtype=torch.int64, device=dev)
    ep = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ep[guard.GUARD_DEADLINE] = guard.deadline_ticks(args.deadline) & 0x7FFFFFFF
    w = (1 + 0.1 * torch.randn(LAT, device=dev)).to(BF)
    ok = True
    for m in args.rows:
        for s in range(SETS):
            xl, xs = moe_inputs(m, s, dev)
            pl = torch.zeros(TP, mr, LAT, dtype=BF, device=dev)
            ps = torch.zeros(TP, mr, SHR, dtype=BF, device=dev)
            pl[:, :m].copy_(xl)
            ps[:, :m].copy_(xs)
            ol, on, os_ = torch.zeros_like(pl), torch.zeros_like(pl), torch.zeros_like(ps)
            ep[0:1].add_(1)
            for slot in range(2):
                launch(pl.data_ptr(), ps.data_ptr(), w.data_ptr(), ol.data_ptr(), os_.data_ptr(), on.data_ptr(), m,
                       table.data_ptr(), 0, ep.data_ptr(), slot, stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            al, as_, n, r64 = moe_refs(xl, xs, w, args.eps)
            bl = int((ol[:, :m].view(torch.int16) != al.view(torch.int16)).sum())
            bs = int((os_[:, :m].view(torch.int16) != as_.view(torch.int16)).sum())
            bn = int((on[:, :m].view(torch.int16) != n.view(torch.int16)).sum())
            rf = max(rel_l2(on[r, :m], r64) for r in range(TP))
            rs_ = rel_l2(n, r64)
            ok &= bl == 0 and bs == 0 and rf <= 1.05 * rs_ + 1e-6
            print(f"EMU moe m={m:3d} set {s} latent_bad={bl} shared_bad={bs} norm_vs_stock_bad={bn} "
                  f"rel_fused={rf:.3e} rel_stock={rs_:.3e}", flush=True)
    print("EMU moe guard", ep[:8].tolist(), flush=True)
    return ok


def graph_time(body, reps, per):
    """µs a call: ``body`` captured once (``per`` calls), replayed ``reps`` times."""
    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / (reps * per)


def across(v, dev):
    """(median, max) of a per-rank float over the TP ranks."""
    t = torch.tensor([v], dtype=torch.float64, device=dev)
    n = dist.get_world_size()
    every = [torch.empty_like(t) for _ in range(n)]
    dist.all_gather(every, t)
    vals = sorted(float(e.item()) for e in every)
    return (vals[(n - 1) // 2] + vals[n // 2]) / 2, vals[-1]


def stage(rank, what):
    """A per-rank progress marker, so a hang shows where each rank stopped."""
    import time

    print(f"STAGE r{rank} {time.strftime('%H:%M:%S')} {what}", flush=True)


def tp8_main(args):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.stages import ar_read as K

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    stage(rank, "init")
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)
    dist.barrier()
    stage(rank, "process groups up")
    nl = 20
    if args.only_ag:
        out = {"rank": rank, "time": []}
        time_fp32_ag(args, cpu, dev, rank, recorder(out, dev, rank), nl)
        dist.destroy_process_group()
        return True
    launch = K.build(args.eps, args.eps, nl)
    peer = PeerBuffer(K.peer_bytes(), cpu, rank, TP, dev, init=K.arm)
    stage(rank, "peer buffer up")
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(args.deadline)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    ca = CustomAllreduce(cpu, dev)
    stage(rank, "aiter custom all-reduce up")
    out = {"rank": rank, "check": [], "time": []}

    def fused_body(cur, m, nvb, has_res, cw, w_o, xs, pres, outs, gl):
        def body():
            epoch[0:1].add_(1)
            for i in range(gl):
                launch(
                    xs[i].data_ptr(), cur["res"].data_ptr(), cur["bank"].data_ptr(), cw.data_ptr(),
                    w_o.data_ptr(), pres[i].data_ptr(), outs[i].data_ptr(), m, nvb, NBK, int(has_res),
                    peer.addresses.data_ptr(), rank, epoch.data_ptr(), i,
                    stream=torch.cuda.current_stream(),
                )  # fmt: skip
        return body

    # checks
    for m, nvb, has_res in cases(args.rows):
        def launch_one(cur, m, nvb, has_res, cw, w_o, gl):
            xs = torch.zeros(gl, m, H, dtype=BF, device=dev)
            pres = torch.zeros(gl, m, H, dtype=BF, device=dev)
            outs = torch.zeros(gl, m, H, dtype=BF, device=dev)
            inner = fused_body(cur, m, nvb, has_res, cw, w_o, xs, pres, outs, gl)

            def body():
                for i in range(gl):
                    xs[i].copy_(cur["x"][rank])
                inner()

            body()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            return {"replay": g.replay, "out": lambda r, i: outs[i], "pre": lambda r, i: pres[i]}

        stage(rank, f"check m={m} nvb={nvb} res={int(has_res)}")
        st = check_case(launch_one, m, nvb, has_res, dev, args.eps, [rank], 4)
        tot = torch.tensor([st["pre_bad"], st["out_bad"]], dtype=torch.int64, device=dev)
        dist.all_reduce(tot)
        rel = torch.tensor([st["rel_fused"]], dtype=torch.float64, device=dev)
        dist.all_reduce(rel, op=dist.ReduceOp.MAX)
        st.update(m=m, nvb=nvb, has_res=has_res, pre_bad_all=int(tot[0]), out_bad_all=int(tot[1]),
                  rel_fused_all=float(rel))
        out["check"].append(st)
        if rank == 0:
            print(f"CHECK m={m:3d} nvb={nvb} res={int(has_res)} pre_bad={int(tot[0])} out_bad={int(tot[1])} "
                  f"rel_fused={float(rel):.3e} rel_stock={st['rel_stock']:.3e}", flush=True)

    if args.time:
        cw, w_o = make_weights(dev)

        record = recorder(out, dev, rank)

        for m in args.rows:
            stage(rank, f"time o_proj m={m}")
            st = make_inputs(m, 0, dev)
            nvb = 8
            cur = {"res": st["res"], "bank": st["bank"]}
            x = st["x"][rank].contiguous()
            xs = x.unsqueeze(0).expand(nl, m, H).contiguous()
            pres = torch.empty(nl, m, H, dtype=BF, device=dev)
            outs = torch.empty_like(pres)
            record("fused", m, graph_time(fused_body(cur, m, nvb, True, cw, w_o, xs, pres, outs, nl),
                                          args.reps, nl))
            # stock: aiter's custom all-reduce (its own size dispatch) + attn_res_hip
            from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip

            ins = [x.clone() for _ in range(nl)]

            def stock_body(with_ar=True, with_read=True):
                def body():
                    for i in range(nl):
                        a = ca.custom_all_reduce(ins[i]) if with_ar else ins[i]
                        if with_read:
                            attn_res_hip(cur["res"], cur["bank"], cw, w_o, outs[i], nvb, args.eps,
                                         args.eps, addend=a, prefix_out=pres[i])
                return body

            for name, kw in (("stock_ar+read", {}), ("stock_ar", {"with_read": False}),
                             ("stock_read", {"with_ar": False})):
                b = stock_body(**kw)
                b()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with ca.capture():
                    with torch.cuda.graph(g):
                        b()
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                dist.barrier()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(args.reps):
                    g.replay()
                e1.record()
                torch.cuda.synchronize()
                record(name, m, e0.elapsed_time(e1) * 1000 / (args.reps * nl))
        # in order of need, each section on its own: a failure leaves the others' numbers
        sections = []
        if args.gather_rows:
            sections.append(("gather", lambda: time_gather(args, cpu, dev, rank, record, nl)))
        if args.ag_rows:
            sections.append(("fp32 ag", lambda: time_fp32_ag(args, cpu, dev, rank, record, nl)))
        for m in args.rows:
            sections.append((f"moe m={m}", lambda m=m: moe_halves(args, cpu, dev, rank, record, nl, m, ca, out)))
        for m in args.rows:
            sections.append((f"coll m={m}", lambda m=m: moe_and_coll(args, cpu, dev, rank, record, nl, m, ca, epoch)))
        for name, run in sections:
            stage(rank, f"section {name}")
            try:
                run()
            except Exception as e:  # noqa: BLE001
                print(f"SECTION {name} rank {rank} failed: {type(e).__name__}: {e}", flush=True)
    every = [None] * TP
    dist.all_gather_object(every, out, group=cpu)
    if rank == 0 and args.json:
        with open(args.json, "w") as f:
            json.dump(every, f, indent=1)
    ok = all(c["pre_bad_all"] == 0 and c["rel_fused_all"] <= 1.05 * c["rel_stock"] + 1e-6
             for c in out["check"] if "pre_bad_all" in c)
    bad = torch.tensor([int(not ok)], device=dev)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if rank == 0:
        print("TP8", "PASS" if not bad.item() else "FAIL", flush=True)
    dist.destroy_process_group()
    return not bad.item()


def moe_and_coll(args, cpu, dev, rank, record, nl, m, ca, epoch):
    """aiter's flat MoE all-reduce and kimi_k31 coll.py's standalone ARs at this width."""
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer

    # the MoE's [latent | shared] all-reduce at this width
    moe_in = [torch.randn(m, 10752, device=dev).to(BF) for _ in range(nl)]

    def moe_body():
        for i in range(nl):
            ca.custom_all_reduce(moe_in[i])

    moe_body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with ca.capture():
        with torch.cuda.graph(g):
            moe_body()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(args.reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    record("stock_moe_ar_10752", m, e0.elapsed_time(e1) * 1000 / (args.reps * nl))
    # kimi_k31 coll.py's standalone two-shot AR
    from sgl_amd_mono.kimi_k31.gfx950.stages import coll

    for h in (7168, 10752):
        for fmt in ("tag", "flag"):
            cl = coll.standalone(m, h, "ar", fmt, False, nl, "launch")
            cp = PeerBuffer(coll.peer_bytes(m, h, fmt), cpu, rank, TP, dev)
            cp.bytes.zero_()
            ep = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
            ep[1:3].copy_(epoch[1:3])
            cx = torch.randn(nl, m, h, device=dev).to(BF)
            cy = torch.empty_like(cx)
            dl = torch.zeros(nl, 2 * 256, dtype=torch.int32, device=dev)

            def cbody():
                ep[0:1].add_(1)
                for i in range(nl):
                    cl(cx[i].data_ptr(), cy[i].data_ptr(), cp.addresses.data_ptr(), rank,
                       ep.data_ptr(), dl[i].data_ptr(), i, stream=torch.cuda.current_stream())

            record(f"coll_ar_{fmt}_{h}", m, graph_time(cbody, args.reps, nl))
            cp.close()


def stock_rmsnorm(h, eps, w):
    """SGLang's RMSNorm module (the latent MoE's routed_expert_norm) with weight ``w``."""
    from sglang.srt.layers.layernorm import RMSNorm

    norm = RMSNorm(h, eps=eps).to(w.device)
    norm.weight.data.copy_(w)
    return norm


def moe_halves(args, cpu, dev, rank, record, nl, m, ca, out):
    """The MoE's two all-reduces once split (latent [m, 3584] + its RMSNorm, shared [m, 7168]):
    sgl-amd-mono's ar_norm launch against aiter's all-reduce (+ SGLang's RMSNorm); check and time."""
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.stages import ar_norm as N

    for h, norm in ((3584, True), (7168, False)):
        launch = N.build(h, norm, args.eps, nl)
        peer = PeerBuffer(N.peer_bytes(h), cpu, rank, TP, dev, init=N.arm(h))
        ep = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
        ticks = guard.deadline_ticks(args.deadline)
        lo = ticks & 0xFFFFFFFF
        ep[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        ep[guard.GUARD_DEADLINE + 1] = ticks >> 32
        g = torch.Generator(device=dev).manual_seed(4321 + m + h)
        xall = (torch.randn(TP, m, h, generator=g, device=dev) * 0.3).to(BF)
        w = (1 + 0.1 * torch.randn(h, generator=g, device=dev)).to(BF)
        xs = xall[rank].unsqueeze(0).expand(nl, m, h).contiguous()
        ys = torch.empty_like(xs)

        def body():
            ep[0:1].add_(1)
            for i in range(nl):
                launch(xs[i].data_ptr(), w.data_ptr(), ys[i].data_ptr(), m, peer.addresses.data_ptr(), rank,
                       ep.data_ptr(), i, stream=torch.cuda.current_stream())

        us = graph_time(body, args.reps, nl)
        ar = ar_rank_order(xall)
        snorm = stock_rmsnorm(h, args.eps, w) if norm else None
        ref = snorm(ar.clone()) if norm else ar
        bad = int(sum((ys[i].view(torch.int16) != ref.view(torch.int16)).sum() for i in range(nl)))
        r64 = None
        if norm:
            a64 = xall.double().sum(0)
            r64 = a64 / torch.sqrt((a64 * a64).mean(-1, keepdim=True) + args.eps) * w.double()
        rel = rel_l2(ys[0], r64) if norm else 0.0
        rel_s = rel_l2(ref, r64) if norm else 0.0
        tot = torch.tensor([bad], dtype=torch.int64, device=dev)
        dist.all_reduce(tot)
        if rank == 0:
            print(f"CHECK moe h={h} norm={int(norm)} m={m:3d} mismatches_vs_stock={int(tot)} rel_fused={rel:.3e} "
                  f"rel_stock={rel_s:.3e}", flush=True)
        out["check"].append({"moe": h, "norm": norm, "m": m, "mismatch_vs_stock_all": int(tot), "rel_fused": rel,
                             "rel_stock": rel_s})
        record(f"fused_moe_{h}{'_norm' if norm else ''}", m, us)
        ins = [xall[rank].clone() for _ in range(nl)]

        def stock_body():
            for i in range(nl):
                r = ca.custom_all_reduce(ins[i])
                if norm:
                    snorm(r)

        stock_body()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with ca.capture():
            with torch.cuda.graph(gr):
                stock_body()
        for _ in range(3):
            gr.replay()
        torch.cuda.synchronize()
        dist.barrier()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(args.reps):
            gr.replay()
        e1.record()
        torch.cuda.synchronize()
        record(f"stock_moe_{h}{'_ar+norm' if norm else '_ar'}", m, e0.elapsed_time(e1) * 1000 / (args.reps * nl))
        peer.close()
    # the flat [latent | shared] all-reduce in one launch, the latent's RMSNorm in it
    launch = N.build_moe(args.eps, nl)
    peer = PeerBuffer(N.moe_peer_bytes(), cpu, rank, TP, dev, init=N.moe_arm)
    ep = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ep[guard.GUARD_DEADLINE] = guard.deadline_ticks(args.deadline) & 0x7FFFFFFF
    xl, xs = moe_inputs(m, 0, dev)
    w = (1 + 0.1 * torch.randn(LAT, device=dev)).to(BF)
    dist.broadcast(w, 0)
    al, as_, n, r64 = moe_refs(xl, xs, w, args.eps)
    pl = xl[rank].unsqueeze(0).expand(nl, m, LAT).contiguous()
    ps = xs[rank].unsqueeze(0).expand(nl, m, SHR).contiguous()
    ol, on, os_ = torch.empty_like(pl), torch.empty_like(pl), torch.empty_like(ps)

    def body():
        ep[0:1].add_(1)
        for i in range(nl):
            launch(pl[i].data_ptr(), ps[i].data_ptr(), w.data_ptr(), ol[i].data_ptr(), os_[i].data_ptr(),
                   on[i].data_ptr(), m, peer.addresses.data_ptr(), rank, ep.data_ptr(), i,
                   stream=torch.cuda.current_stream())

    us = graph_time(body, args.reps, nl)
    bad = torch.tensor([int((ol.view(torch.int16) != al.view(torch.int16)).sum()),
                        int((os_.view(torch.int16) != as_.view(torch.int16)).sum()),
                        int((on.view(torch.int16) != n.view(torch.int16)).sum())], dtype=torch.int64, device=dev)
    dist.all_reduce(bad)
    rel = rel_l2(on[0], r64)
    if rank == 0:
        print(f"CHECK moe flat m={m:3d} latent_bad={int(bad[0])} shared_bad={int(bad[1])} "
              f"norm_vs_stock_bad={int(bad[2])} rel_fused={rel:.3e} rel_stock={rel_l2(n, r64):.3e}", flush=True)
    out["check"].append({"moe": "flat", "m": m, "bad": bad.tolist(), "rel_fused": rel, "rel_stock": rel_l2(n, r64)})
    record("fused_moe_flat+norm", m, us)
    snorm = stock_rmsnorm(LAT, args.eps, w)
    flats = [torch.cat([xl[rank].reshape(-1), xs[rank].reshape(-1)]) for _ in range(nl)]

    def stock_body():
        for i in range(nl):
            r = ca.custom_all_reduce(flats[i])
            snorm(r[: m * LAT].view(m, LAT))

    stock_body()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with ca.capture():
        with torch.cuda.graph(gr):
            stock_body()
    for _ in range(3):
        gr.replay()
    torch.cuda.synchronize()
    dist.barrier()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(args.reps):
        gr.replay()
    e1.record()
    torch.cuda.synchronize()
    record("stock_moe_flat_ar+norm", m, e0.elapsed_time(e1) * 1000 / (args.reps * nl))
    peer.close()


def recorder(out, dev, rank):
    def record(name, m, us):
        med, mx = across(us, dev)
        out["time"].append({"name": name, "m": m, "us_rank": us, "us_med": med, "us_max": mx})
        if rank == 0:
            print(f"TIME {name:24s} m={m:3d} med {med:7.2f} us  slowest {mx:7.2f} us", flush=True)

    return record


AG_COLS = 560  # a rank's slice of a sharded MoE front's [router 896 | latent-down 3584] fp32 columns


def time_fp32_ag(args, cpu, dev, rank, record, nl):
    """The fp32 all-gather a column-sharded MoE front needs: every rank's [rows, 560] block -> [rows, 4480]
    (RCCL through SGLang's PyNcclCommunicator, gathering [8 rows, 560] blocks in rank order, i.e. dim 0; aiter's
    custom all-gather along the last dim, [rows, 4480] directly). CUDA graphs of nl calls, as the other timings."""
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    world = dist.get_world_size()
    pyn = PyNcclCommunicator(cpu, dev)
    ca = CustomAllreduce(cpu, dev) if world > 1 else None
    for m in args.ag_rows:
        ins = [torch.randn(m, AG_COLS, device=dev) for _ in range(nl)]
        outs = [torch.empty(world * m, AG_COLS, device=dev) for _ in range(nl)]
        if pyn.available:

            def rccl():
                for i in range(nl):
                    pyn.all_gather(outs[i], ins[i])

            with pyn.change_state(enable=True):
                us = graph_time(rccl, args.reps, nl)
                want = torch.cat([torch.full((m, AG_COLS), float(r), device=dev) for r in range(world)])
                ins[0].fill_(float(rank))
                pyn.all_gather(outs[0], ins[0])
                torch.cuda.synchronize()
                ok = bool(torch.equal(outs[0], want))
            record(f"fp32_ag_rccl_dim0{'' if ok else '_WRONG'}", m, us)
        if ca is not None and ca.should_custom_ag(ins[0]):

            def custom():
                for i in range(nl):
                    ca.custom_all_gather(ins[i], dim=-1)

            custom()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with ca.capture():
                with torch.cuda.graph(g):
                    custom()
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            dist.barrier()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(args.reps):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000 / (args.reps * nl)
            ins[0].fill_(float(rank))
            got = ca.custom_all_gather(ins[0], dim=-1)
            torch.cuda.synchronize()
            want = torch.cat([torch.full((m, AG_COLS), float(r), device=dev) for r in range(world)], dim=1)
            ok = got is not None and bool(torch.equal(got, want))
            record(f"fp32_ag_aiter_lastdim{'' if ok else '_WRONG'}", m, us)
        elif rank == 0:
            print(f"TIME fp32 ag m={m}: aiter custom all-gather does not take [{m}, {AG_COLS}] fp32", flush=True)


def time_gather(args, cpu, dev, rank, record, nl):
    """The column tail's LL gather + next attention read (SGLang's gather_tail_attn_res_read) at
    rows above its 64-row cap: partials mode (as with _COLUMN_TAIL_PARTIALS_MAX_TOKENS raised) and
    projected mode (the slice's GEMM)."""
    from sglang.kernels.ops.attention import k3_attn_res_read_hip as R
    from sglang.kernels.ops.communication.k3_ll_comm_hip import create_ll_comm
    from sglang.kernels.ops.gemm import kimi_k3_latent_tail_hip as T

    comm = create_ll_comm(cpu, dev)
    assert comm is not None
    cw, _ = make_weights(dev)
    ow = torch.ones(H, dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev).manual_seed(5)
    norm_w = (1 + 0.1 * torch.randn(3584, generator=g, device=dev)).to(BF)
    up_w = (torch.randn(H // TP, 3584, generator=g, device=dev) * 0.02).to(BF)
    for m in args.gather_rows:
        latent = torch.randn(m, 3584, generator=g, device=dev).to(BF)
        shared = torch.randn(m, H, generator=g, device=dev).to(BF)
        prefix = torch.randn(m, H, generator=g, device=dev).to(BF)
        bank = torch.randn(m, NBK, H, generator=g, device=dev).to(BF)
        part, sq = T.k3_latent_tail_partials(latent, norm_w, up_w)
        proj = torch.randn(m, H // TP, generator=g, device=dev).to(BF)

        def gbody(p, s):
            def body():
                for _ in range(nl):
                    R.gather_tail_attn_res_read(p, s, shared, prefix, 1e-6, bank, 8, cw, ow, args.eps,
                                                args.eps, comm)
            return body

        record("ll_gather_partials", m, graph_time(gbody(part, sq), args.reps, nl))
        record("ll_gather_projected", m, graph_time(gbody(proj, None), args.reps, nl))

        def pbody():
            for _ in range(nl):
                T.k3_latent_tail_partials(latent, norm_w, up_w)

        record("latent_tail_partials", m, graph_time(pbody, args.reps, nl))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="54,96,168", type=lambda s: [int(v) for v in s.split(",")])
    p.add_argument("--gather-rows", default="", type=lambda s: [int(v) for v in s.split(",") if v])
    p.add_argument("--ag-rows", default="", type=lambda s: [int(v) for v in s.split(",") if v])
    p.add_argument("--only-ag", action="store_true", help="only the fp32 all-gather timing")
    p.add_argument("--emulate", action="store_true")
    p.add_argument("--time", action="store_true")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--deadline", type=float, default=10.0)
    p.add_argument("--json")
    args = p.parse_args()
    ok = emulate_main(args) if args.emulate else tp8_main(args)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
