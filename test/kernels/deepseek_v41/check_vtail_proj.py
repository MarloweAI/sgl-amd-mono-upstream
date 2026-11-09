"""Check the static verify's accept launch with the draft's main_proj in it (``gfx950/vtail`` ``proj=``, torchrun
TP4 or TP8) against the two launches it replaces (``gfx950/vtail`` alone, then ``gfx950/inject/proj``): the accept's
outputs and every rank's main_proj rows bitwise, greedy and under a simulated length, wide-range rows, eager and
under graph replay with fresh inputs; then each form's time in a graph (max over ranks).

    torchrun --nproc-per-node 8 check_vtail_proj.py [--replays 10] [--chain 20]
"""

import argparse
import os

import torch
import torch.distributed as dist
from check_verify_tail import VOCAB, Case, T, inputs, ours, outs, same

H, K = 5120, 15360


def rows_bitwise(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def hidden_rows(rows, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)  # every rank the same rows
    x = (torch.randn(rows, K, generator=g, device=dev) * 2).bfloat16()
    if seed % 2:  # wide rows: large and tiny values together
        x[:, ::7] *= 300.0
        x[:, 3::11] *= 1e-3
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--chain", type=int, default=20)
    a = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.gfx950 import vtail
    from sgl_amd_mono.deepseek_v41.gfx950.inject import proj

    part = -(-VOCAB // tp)
    part += -part % 8
    start = rank * part
    width = max(0, min(part, VOCAB - start))
    tail = vtail.VerifyTail(
        cpu,
        rank,
        tp,
        dev,
        part=part,
        width=width,
        vocab_start=start,
        vocab=VOCAB,
        deadline_s=30.0,
    )
    mp = proj.Exchange(cpu, rank, tp, dev, deadline_s=30.0)
    g = torch.Generator(device=dev).manual_seed(3)
    w = (
        (torch.randn(H, K, generator=g, device=dev) * 40)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    ws = torch.randint(
        116, 124, (H // 32, K // 32), generator=g, device=dev, dtype=torch.uint8
    )
    cols = mp.columns(w, ws)
    say = (lambda s: print(s, flush=True)) if rank == 0 else (lambda s: None)
    fails = []

    def fused(x, o, hidden):
        tail(
            x["local"],
            x["cand"],
            x["seq_lens"],
            x["verify_lens"],
            x["gate"],
            sim=x["sim"],
            temps=x["temps"] if x["sim"] is not None else None,
            seed=x["seed"] if x["sim"] is not None else None,
            correct_out=o["correct_len_out"],
            bonus_out=o["bonus_out"],
            cap_trim_out=o["cap_trim_lens_out"],
            commit_out=o["commit_lens_out"],
            new_seq_out=o["new_seq_lens_out"],
            out_tokens=o["out_tokens_out"],
            gated_out=o["gated_commit_lens_out"],
            proj=(hidden, *cols, mp),
        )

    def agree(ok):
        t = torch.tensor([int(ok)], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    seed = 300
    for mode in ("greedy", "sim0"):
        for rows in (proj.MIN_ROWS, T, proj.MAX_ROWS):
            for rep in range(2):
                seed += 1
                bs = 1
                x = inputs(Case(bs, mode), rank, tp, part, dev, seed)
                hidden = hidden_rows(rows, dev, seed)
                want = outs(bs, dev)
                ours(tail, x, bs, want)
                want_rows = mp.run(hidden, *cols).clone()
                got = outs(bs, dev)
                mp.out.zero_()
                fused(x, got, hidden)
                torch.cuda.synchronize()
                ok = agree(
                    same(want, got, bs) and rows_bitwise(mp.out[:rows], want_rows)
                )
                say(f"{mode} rows {rows} rep {rep}: {'PASS' if ok else 'FAIL'}")
                if not ok:
                    fails.append(f"{mode} rows {rows} rep {rep}")
    # graph replays with fresh inputs: the static tensors refilled before each replay
    for mode in ("greedy", "sim0"):
        seed += 1
        x = inputs(Case(1, mode), rank, tp, part, dev, seed)
        hidden = hidden_rows(T, dev, seed)
        got = outs(1, dev)
        fused(x, got, hidden)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused(x, got, hidden)
        ok_all = True
        for r in range(a.replays):
            seed += 1
            fresh = inputs(Case(1, mode), rank, tp, part, dev, seed)
            for k in ("local", "cand", "seq_lens", "verify_lens") + (
                ("sim", "temps", "seed") if mode != "greedy" else ()
            ):
                x[k].copy_(fresh[k])
            hidden.copy_(hidden_rows(T, dev, seed))
            want = outs(1, dev)
            ours(tail, x, 1, want)
            want_rows = mp.run(hidden, *cols).clone()
            mp.out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            ok_all &= same(want, got, 1) and rows_bitwise(mp.out[:T], want_rows)
        ok_all = agree(ok_all)
        say(
            f"{mode} {a.replays} graph replays, fresh inputs: {'PASS' if ok_all else 'FAIL'}"
        )
        if not ok_all:
            fails.append(f"{mode} replays")
    # time: a graph of ``chain`` calls, the median of 5 replays, max over ranks
    x = inputs(Case(1, "sim0"), rank, tp, part, dev, 7)
    hidden = hidden_rows(T, dev, 8)
    o = outs(1, dev)

    def timed(fn):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(a.chain):
                fn()
        got = []
        for _ in range(7):
            dist.barrier()
            torch.cuda.synchronize()
            t0, t1 = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            t0.record()
            graph.replay()
            t1.record()
            torch.cuda.synchronize()
            got.append(t0.elapsed_time(t1) * 1000 / a.chain)
        t = torch.tensor([sorted(got)[3]], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    forms = {
        "accept alone": lambda: ours(tail, x, 1, o),
        "main_proj alone": lambda: mp.run(hidden, *cols),
        "accept then main_proj": lambda: (ours(tail, x, 1, o), mp.run(hidden, *cols)),
        "one launch": lambda: fused(x, o, hidden),
    }
    for _ in range(2):
        times = {name: timed(fn) for name, fn in forms.items()}
    say(
        f"time a call, TP{tp}, {T} rows, simulated length (max over ranks): "
        + ", ".join(f"{k} {v:.1f} us" for k, v in times.items())
    )
    say("ALL PASS" if not fails else f"FAIL at {fails}")
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
