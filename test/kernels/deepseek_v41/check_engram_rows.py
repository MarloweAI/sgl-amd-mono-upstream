"""Check a verify step's Engram rows in one launch (``gfx950/engram``, torchrun TP4 or TP8) against SGLang's chain:
the hash (``engram_hash_ids``, MODE_VERIFY), each layer's gather from this rank's table shard (``engram_gather``)
and the all-reduce of the shards' rows. Every rank's rows bitwise, at 1, 2 and 8 requests, with and without image
tokens, eager and under graph replay with fresh inputs; then the time of each (a graph of chained calls, max over
ranks)."""

import argparse
import os
import types

import torch
import torch.distributed as dist

BLOCK, VOCAB, CVOCAB = 6, 129280, 99092


def layout(vocab: int):
    from sglang.srt.layers.engram import EngramLayout

    cfg = types.SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        engram_n_heads=8,
        engram_vocab_size=vocab,
        engram_num_embeddings=[0, 0],
        engram_head_dim=256,
    )
    return EngramLayout.from_config(cfg)


def hasher(lay, dev, g, slots: int, image):
    import numpy as np
    from sglang.srt.layers.engram import compute_hash_multipliers

    flat = [[p for per in layer for p in per] for layer in lay.primes]
    offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat])
    return types.SimpleNamespace(
        history=torch.randint(
            0, VOCAB, (slots + 1, 3), generator=g, device=dev, dtype=torch.int32
        ),
        token_map=torch.randint(0, CVOCAB, (VOCAB,), generator=g, device=dev),
        multipliers=compute_hash_multipliers(lay.layer_ids, 4, CVOCAB).to(dev),
        primes=torch.tensor(lay.primes, device=dev),
        offsets=torch.tensor(offsets, device=dev),
        pad_id=7,
        image_token_id=image,
    ), [sum(f) for f in flat]


def table(g, dev, rows: int):
    w = torch.randint(0, 256, (rows, 256), generator=g, device=dev, dtype=torch.uint8)
    w[(w & 0x7F) == 0x7F] = 0x80  # no NaN; -0 kept
    s = torch.randint(110, 135, (rows, 8), generator=g, device=dev, dtype=torch.uint8)
    s[::97] = 0  # 2^-127 blocks
    return w, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--chain", type=int, default=20)
    ap.add_argument(
        "--vocab",
        type=int,
        default=20011,
        help="the hash vocab: tables of about 24 x this many rows",
    )
    ap.add_argument("--diag", action="store_true")
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    cpu = dist.new_group(backend="gloo")
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather
    from sglang.kernels.ops.embeddings.engram_hash import MODE_VERIFY, engram_hash_ids
    from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE

    from sgl_amd_mono.deepseek_v41.gfx950 import engram as eg

    lay = layout(args.vocab)
    g = torch.Generator(device=dev).manual_seed(11)  # every rank the same tensors
    slots_n = 16
    h, rows_per_layer = hasher(lay, dev, g, slots_n, None)
    tabs, shards = [], []
    for k, n in enumerate(rows_per_layer):
        w, s = table(g, dev, n)
        lo, hi = n * rank // tp, n * (rank + 1) // tp
        tabs.append((w[lo:hi].contiguous(), s[lo:hi].contiguous()))
        shards.append((k, tabs[-1][0].data_ptr(), tabs[-1][1].data_ptr(), lo, hi))
    x = eg.Rows(cpu, rank, tp, dev, 20.0, 2)

    def want(ids, pos, slots):
        hid, _ = engram_hash_ids(
            ids,
            pos,
            mode=MODE_VERIFY,
            history=h.history,
            token_map=h.token_map,
            multipliers=h.multipliers,
            primes=h.primes,
            offsets=h.offsets,
            pad_id=h.pad_id,
            num_real=ids.shape[0],
            req_slots=slots,
            block=BLOCK,
            row=None,
            starts=None,
            image_token_id=h.image_token_id,
            mm_pad_shift=MM_PAD_SHIFT_VALUE,
        )
        by_layer = hid.transpose(0, 1).contiguous()
        out = torch.empty(2, ids.shape[0], 24, 256, dtype=torch.bfloat16, device=dev)
        for k, (_, wp, sp, lo, hi) in enumerate(shards):
            engram_gather(
                wp,
                sp,
                by_layer[k].view(-1),
                out[k].view(-1, 256),
                256,
                32,
                row_lo=lo,
                row_hi=hi,
            )
        dist.all_reduce(out)
        return out

    def inputs(bs):
        ids = torch.randint(0, VOCAB, (bs * BLOCK,), generator=g, device=dev)
        pos = torch.randint(0, 40, (bs * BLOCK,), generator=g, device=dev)
        pos[:BLOCK] = torch.arange(
            BLOCK, device=dev
        )  # the first request at the sequence start: blocked shifts
        slots = torch.randperm(slots_n, generator=g, device=dev)[:bs]
        return ids, pos, slots

    def same(a, b):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))

    fails = []
    for image in (None, 4242):
        h.image_token_id = image
        if image is not None:
            h.history[::3, 1] = (
                MM_PAD_SHIFT_VALUE  # the history's multimodal pad ids read as the image token
            )
        for bs in (1, 2, 8):
            ids, pos, slots = inputs(bs)
            if image is not None:
                ids[2::5] = image
            ref = want(ids, pos, slots)
            got = x.run(ids, pos, slots, h, shards, BLOCK, MM_PAD_SHIFT_VALUE)
            torch.cuda.synchronize()
            ok = all(same(got[k], ref[k]) for k in range(2))
            nd = sum(
                int((got[k].view(torch.int16) != ref[k].view(torch.int16)).sum())
                for k in range(2)
            )
            if nd and rank == 0 and args.diag:
                for k in range(2):
                    bad = (
                        (got[k].view(torch.int16) != ref[k].view(torch.int16))
                        .nonzero()[:6]
                        .tolist()
                    )
                    for t_, c_, d_ in bad:
                        a_, b_ = (
                            int(got[k][t_, c_, d_].view(torch.int16)),
                            int(ref[k][t_, c_, d_].view(torch.int16)),
                        )
                        print(
                            f"  layer {k} token {t_} col {c_} dim {d_}: got {a_ & 0xFFFF:04x} "
                            f"({float(got[k][t_, c_, d_])}) want {b_ & 0xFFFF:04x} ({float(ref[k][t_, c_, d_])})"
                        )
            graph_ok = True
            if bs in (1, 8):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    x.run(ids, pos, slots, h, shards, BLOCK, MM_PAD_SHIFT_VALUE)
                for _ in range(args.replays):
                    i2, p2, s2 = inputs(bs)
                    ids.copy_(i2), pos.copy_(p2), slots.copy_(s2)
                    ref = want(ids, pos, slots)
                    x.out.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    graph_ok &= all(
                        same(x.out[k, : ids.shape[0]], ref[k]) for k in range(2)
                    )
            ok_all = torch.tensor([int(ok and graph_ok)], device=dev)
            dist.all_reduce(ok_all, op=dist.ReduceOp.MIN)
            if rank == 0:
                print(
                    f"image {image} bs {bs}: {'PASS' if ok_all.item() else 'FAIL'} | rank0 differing {nd} of "
                    f"{2 * bs * BLOCK * 6144} | replays {'bitwise' if graph_ok else 'DIFFER'}",
                    flush=True,
                )
            if not ok_all.item():
                fails.append((image, bs))
    # timing at one request: a graph of ``chain`` calls each, max over ranks
    h.image_token_id = None
    ids, pos, slots = inputs(1)

    def timed(fn):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.chain):
                fn()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        best = float("inf")
        for _ in range(10):
            a.record()
            graph.replay()
            b.record()
            torch.cuda.synchronize()
            best = min(best, a.elapsed_time(b) * 1e3 / args.chain)
        t = torch.tensor([best], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    t_one = timed(lambda: x.run(ids, pos, slots, h, shards, BLOCK, MM_PAD_SHIFT_VALUE))
    t_chain = timed(lambda: want(ids, pos, slots))
    if rank == 0:
        print(
            f"time us: one launch {t_one:.2f} | SGLang's hash + copy + 2 gathers + NCCL all-reduce {t_chain:.2f}"
        )
        print("ALL PASS" if not fails else f"FAILED {fails}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
