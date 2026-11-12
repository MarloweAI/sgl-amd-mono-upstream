"""Time the sampled accept launch (gfx950/vtail/sampled) against the greedy vtail launch on the same shard, in
graphs of 20 launches, real and simulated lengths (torchrun, TP2 to TP8)."""

import os

import torch
import torch.distributed as dist
from check_sampled_tail import VOCAB, make_inputs, ours, outs, shards


def graph_us(fn, reps=400, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    dist.barrier()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps // n):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / (reps // n * n)


def main():
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda")
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")
    from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
        make_rejection_noise,
    )

    from sgl_amd_mono.deepseek_v41.gfx950 import vtail
    from sgl_amd_mono.deepseek_v41.gfx950.vtail import sampled

    part = VOCAB // tp
    st = sampled.SampledTail(
        gloo,
        rank,
        tp,
        dev,
        width=part,
        vocab_start=rank * part,
        vocab=VOCAB,
        deadline_s=30.0,
    )
    vt = vtail.VerifyTail(
        gloo,
        rank,
        tp,
        dev,
        part=part,
        width=part,
        vocab_start=rank * part,
        vocab=VOCAB,
        deadline_s=30.0,
    )
    x = make_inputs(1, dev, 5)
    local, draft = shards(x, rank, part)
    draft = draft.contiguous()
    noise = make_rejection_noise(1, dev)
    gate = torch.ones(1, dtype=torch.int32, device=dev)
    o = outs(dev)
    sim = torch.tensor([3], device=dev)
    seed = torch.tensor([11], dtype=torch.int32, device=dev)
    zeros = torch.zeros(1, device=dev)
    res = {
        "sampled real": graph_us(
            lambda: ours(st, x, 1, None, None, noise, gate, o, rank, part, local, draft)
        ),
        "sampled sim": graph_us(
            lambda: ours(st, x, 1, sim, seed, noise, gate, o, rank, part, local, draft)
        ),
        "vtail greedy": graph_us(
            lambda: vt(
                local,
                x["cand"],
                x["seq_lens"],
                x["verify_lens"],
                gate,
                sim=None,
                temps=None,
                seed=None,
                correct_out=o["correct_len_out"],
                bonus_out=o["bonus_out"],
                cap_trim_out=o["cap_trim_lens_out"],
                commit_out=o["commit_lens_out"],
                new_seq_out=o["new_seq_lens_out"],
                out_tokens=o["out_tokens_out"],
            )
        ),
        "vtail sim": graph_us(
            lambda: vt(
                local,
                x["cand"],
                x["seq_lens"],
                x["verify_lens"],
                gate,
                sim=sim,
                temps=zeros,
                seed=seed,
                correct_out=o["correct_len_out"],
                bonus_out=o["bonus_out"],
                cap_trim_out=o["cap_trim_lens_out"],
                commit_out=o["commit_lens_out"],
                new_seq_out=o["new_seq_lens_out"],
                out_tokens=o["out_tokens_out"],
            )
        ),
    }
    t = torch.tensor(list(res.values()), device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            f"TP{tp} us a launch (max over ranks): "
            + ", ".join(f"{k} {v:.1f}" for k, v in zip(res, t.tolist()))
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
