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
    ok = all(r["pre_bad"] == 0 and r["rel_fused"] <= 1.05 * r["rel_stock"] + 1e-6 for r in results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"mode": "emulate", "cases": results}, f, indent=1)
    print("EMU", "PASS" if ok else "FAIL", flush=True)
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
    every = [torch.empty_like(t) for _ in range(TP)]
    dist.all_gather(every, t)
    vals = sorted(float(e.item()) for e in every)
    return (vals[TP // 2 - 1] + vals[TP // 2]) / 2, vals[-1]


def tp8_main(args):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.stages import ar_read as K

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)
    nl = 20
    launch = K.build(args.eps, args.eps, nl)
    peer = PeerBuffer(K.peer_bytes(), cpu, rank, TP, dev, init=K.arm)
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(args.deadline)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    ca = CustomAllreduce(cpu, dev)
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

        def record(name, m, us):
            med, mx = across(us, dev)
            out["time"].append({"name": name, "m": m, "us_rank": us, "us_med": med, "us_max": mx})
            if rank == 0:
                print(f"TIME {name:24s} m={m:3d} med {med:7.2f} us  slowest {mx:7.2f} us", flush=True)

        for m in args.rows:
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
        if args.gather_rows:
            time_gather(args, cpu, dev, rank, record, nl)
    every = [None] * TP
    dist.all_gather_object(every, out, group=cpu)
    if rank == 0 and args.json:
        with open(args.json, "w") as f:
            json.dump(every, f, indent=1)
    ok = all(c["pre_bad_all"] == 0 and c["rel_fused_all"] <= 1.05 * c["rel_stock"] + 1e-6 for c in out["check"])
    bad = torch.tensor([int(not ok)], device=dev)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if rank == 0:
        print("TP8", "PASS" if not bad.item() else "FAIL", flush=True)
    dist.destroy_process_group()
    return not bad.item()


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
