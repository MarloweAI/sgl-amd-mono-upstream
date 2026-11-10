"""Check the Engram history commit in the static accept's launch (``gfx950/vtail``'s verify_tail and, where the
shard fits, ``vtail/fused``'s fused_tail; torchrun TP4 or TP8) against SGLang's engram_commit_history after the
accept: the history rows and the FutureMap relay rows bitwise, the accept's outputs unchanged, at 1 and 2 requests
(the second a padded row), the gate open and closed, with and without the relay rows."""

import argparse
import os

import torch
import torch.distributed as dist

VOCAB, K, T, SLOTS, W = 129280, 5120, 6, 32, 3


def outs(dev):
    return dict(
        correct_out=torch.zeros(8, dtype=torch.int64, device=dev),
        bonus_out=torch.zeros(8, dtype=torch.int64, device=dev),
        cap_trim_out=torch.full((8,), 7, dtype=torch.int32, device=dev),
        commit_out=torch.zeros(8, dtype=torch.int32, device=dev),
        new_seq_out=torch.zeros(8, dtype=torch.int64, device=dev),
        out_tokens=torch.full((8, T), -5, dtype=torch.int64, device=dev),
        gated_out=torch.zeros(8, dtype=torch.int32, device=dev),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    cpu = dist.new_group(backend="gloo")
    from sglang.kernels.ops.embeddings.engram_hash import engram_commit_history

    from sgl_amd_mono.deepseek_v41.gfx950 import vtail
    from sgl_amd_mono.deepseek_v41.gfx950.vtail import fused

    part = VOCAB // tp
    kw = dict(
        part=part, width=part, vocab_start=rank * part, vocab=VOCAB, deadline_s=30.0
    )
    tail = vtail.VerifyTail(cpu, rank, tp, dev, **kw)
    ft = (
        fused.FusedTail(cpu, rank, tp, dev, **kw)
        if fused.fits(part=part, hidden=K, vocab=VOCAB)
        else None
    )
    gw = torch.Generator(device=dev).manual_seed(3 + rank)
    w = (torch.randn(part, K, generator=gw, device=dev) * 0.02).bfloat16()
    fails = []
    for seed in range(args.seeds):
        for bs in (1, 2):
            for gate_v in (1, 0):
                for relayed in (True, False):
                    for kind in (
                        ("vtail", "fused") if ft is not None and bs == 1 else ("vtail",)
                    ):
                        g = torch.Generator(device=dev).manual_seed(
                            1000 * seed + 10 * bs + gate_v
                        )
                        cand = torch.randint(0, VOCAB, (bs, T), generator=g, device=dev)
                        seq_lens = torch.randint(
                            100, 9000, (bs,), generator=g, device=dev
                        )
                        verify_lens = torch.full(
                            (bs,), T, dtype=torch.int64, device=dev
                        )
                        if bs == 2:
                            verify_lens[1] = (
                                0  # a padded row: relay stored, history kept
                            )
                        gate = torch.tensor([gate_v], dtype=torch.int32, device=dev)
                        sim = (
                            torch.randint(0, T, (1,), generator=g, device=dev)
                            if seed % 2
                            else None
                        )
                        temps, sd = (
                            torch.zeros(bs, device=dev),
                            torch.randint(0, 2**31 - 1, (1,), device=dev).int(),
                        )
                        slots = torch.randperm(SLOTS, generator=g, device=dev)[:bs]
                        hist0 = torch.randint(
                            0,
                            VOCAB,
                            (SLOTS + 1, W),
                            generator=g,
                            device=dev,
                            dtype=torch.int32,
                        )
                        acc = dict(
                            sim=sim,
                            temps=temps if sim is not None else None,
                            seed=sd if sim is not None else None,
                        )
                        if kind == "vtail":
                            local = (
                                torch.randn(bs * T, part, generator=g, device=dev) * 3
                            ).bfloat16()

                            def run(o, commit=None):
                                tail(
                                    local,
                                    cand,
                                    seq_lens,
                                    verify_lens,
                                    gate,
                                    **acc,
                                    **o,
                                    commit=commit,
                                )
                        else:
                            x = torch.randn(
                                bs * T, K, generator=g, device=dev
                            ).bfloat16()

                            def run(o, commit=None):
                                ft(
                                    x,
                                    w,
                                    cand,
                                    seq_lens,
                                    verify_lens,
                                    gate,
                                    **acc,
                                    **o,
                                    commit=commit,
                                )

                        # the accept, then SGLang's commit on its outputs
                        want, hist_w = outs(dev), hist0.clone()
                        rl_w, rt_w = (
                            torch.full((SLOTS + 1,), -9, dtype=torch.int64, device=dev),
                            torch.full((SLOTS + 1,), -9, dtype=torch.int64, device=dev),
                        )
                        run(want)
                        relay = (
                            (rl_w, rt_w, want["new_seq_out"], want["bonus_out"])
                            if relayed
                            else None
                        )
                        engram_commit_history(
                            hist_w,
                            cand,
                            slots,
                            want["commit_out"][:bs],
                            live_rows=verify_lens,
                            gate=gate,
                            relay=relay,
                        )
                        # the accept with the commit in its launch
                        got, hist_g = outs(dev), hist0.clone()
                        rl_g, rt_g = (
                            torch.full_like(rl_w, -9),
                            torch.full_like(rt_w, -9),
                        )
                        run(
                            got,
                            (
                                hist_g,
                                slots,
                                rl_g if relayed else None,
                                rt_g if relayed else None,
                            ),
                        )
                        torch.cuda.synchronize()
                        ok = (
                            torch.equal(hist_g, hist_w)
                            and torch.equal(rl_g, rl_w)
                            and torch.equal(rt_g, rt_w)
                            and all(torch.equal(got[k][:bs], want[k][:bs]) for k in got)
                        )
                        t = torch.tensor([int(ok)], device=dev)
                        dist.all_reduce(t, op=dist.ReduceOp.MIN)
                        if not t.item():
                            fails.append((kind, seed, bs, gate_v, relayed))
    if rank == 0:
        print(
            f"cases failing: {fails}" if fails else "ALL PASS",
            f"(fused tail {'checked' if ft else 'not at this TP'})",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
