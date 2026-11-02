"""Time a DeepSeek-V4.1 decode layer's K1 and K2 (torchrun, TP2 or TP4): layers of one kind,
each with its own random weights (experts cold from HBM), chained in one graph."""

import argparse
import os

import torch
import torch.distributed as dist
from check_ffn import HC, HIDDEN, make_weights, mono_weights, seam_inputs
from check_k1 import attn_weights, rope_table
from check_k2 import (
    COMP_BLOCK,
    COMP_PAGES,
    SWA_BLOCK,
    SWA_PAGES,
    key_lists,
    make_cache,
    source_keys,
)


class _Layer:
    def __init__(self, i, k1, k2, ffn):
        self.i, self.k1, self.k2, self.ffn = i, k1, k2, ffn


def build_layer(i, tp, rank, dev, cos_sin):
    from sgl_amd_mono.deepseek_v41.gfx950.attention.plan import Dims
    from sgl_amd_mono.deepseek_v41.gfx950.runner import K1Weights, K2Weights

    d = Dims(tp)
    w, _, _ = make_weights(tp, rank, dev, seed=i)
    ffn = mono_weights(w)
    del w
    g = torch.Generator(device=dev).manual_seed(31 + i)
    a = attn_weights(d.heads, dev, g)
    k1 = K1Weights(
        hc_fn=ffn.hc_ffn_fn, hc_scale=ffn.hc_ffn_scale, hc_base=ffn.hc_ffn_base,
        norm=ffn.ffn_norm, wqkv=a["wqkv"], wqkv_s=a["wqkv_s"], q_norm=a["q_norm"],
        kv_norm=a["kv_norm"], wqb=a["wqb"], wqb_s=a["wqb_s"], cos_sin=cos_sin,
    )  # fmt: skip
    o_rows = d.groups * 1024

    def e8m0(*shape):
        return torch.randint(
            118, 124, shape, generator=g, device=dev, dtype=torch.uint8
        )

    woa = (torch.randn(o_rows, d.group_k, generator=g, device=dev) * 4).clamp(-448, 448)
    wob = (torch.randn(HIDDEN, o_rows, generator=g, device=dev) * 4).clamp(-448, 448)
    k2 = K2Weights(
        woa=woa.to(torch.float8_e4m3fn), woa_s=e8m0(o_rows // 32, d.group_k // 32),
        wob=wob.to(torch.float8_e4m3fn), wob_s=e8m0(HIDDEN // 32, o_rows // 32),
        sink=torch.randn(d.heads, generator=g, device=dev), cos_sin=cos_sin,
    )  # fmt: skip
    k1.check(d)
    k2.check(d)
    ffn.check(tp, 384)
    return _Layer(i, k1, k2, ffn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="6,12")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=2, help="the layers' passes a graph")
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--trials", type=int, default=5, help="timed batches of replays")
    ap.add_argument(
        "--timeline", action="store_true", help="the last layer's stage marks"
    )
    args = ap.parse_args()
    if args.timeline:  # the launches build with stage marks
        os.environ["SGLANG_AMD_MONO_TIMELINE"] = "1"
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    cos_sin = rope_table(dev)
    layers = [build_layer(i, tp, rank, dev, cos_sin) for i in range(args.layers)]
    gc = torch.Generator(device=dev).manual_seed(5)
    swa_raw, _ = make_cache(SWA_PAGES, SWA_BLOCK, gc, dev)
    comp_raw, _ = make_cache(COMP_PAGES, COMP_BLOCK, gc, dev)
    from sgl_amd_mono.deepseek_v41.gfx950.runner import DSV41MonoRunner

    runner = DSV41MonoRunner(tp, rank, cpu, dev)
    for M in (int(m) for m in args.widths.split(",")):
        gk = torch.Generator().manual_seed(1000 + M)
        _, _, lists = key_lists(M, gk, dev)
        slot, keys = source_keys(M, lists, dev)
        slot = slot.clamp(min=0)
        pos = torch.randint(0, 4096, (M,), generator=gk).to(dev)
        x0 = (torch.randn(M, HIDDEN, device=dev) * 0.1).bfloat16()
        _, res0, post0, comb0, pre0 = seam_inputs(M, tp, rank, dev)

        def chain():
            x, res, post, comb, pre = x0, res0, post0, comb0, pre0
            for _ in range(args.repeat):
                for L in layers:
                    runner.k1(
                        L.k1, x, res, post.view(M, HC), comb, pre, pos, slot,
                        swa_raw, SWA_BLOCK, keys,
                    )  # fmt: skip
                    x, res, post, comb, pre = runner.k2(
                        L.k2, L.ffn, M, pos, swa_raw, SWA_BLOCK, comp_raw,
                        COMP_BLOCK, slot=slot, keys=keys,
                    )  # fmt: skip
            return x

        out = chain()  # eager: builds the launches and the width's scratch
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout = chain()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        pers = []
        for _ in range(args.trials):
            dist.barrier()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(args.replays):
                graph.replay()
            t1.record()
            torch.cuda.synchronize()
            per = t0.elapsed_time(t1) * 1e3 / (args.replays * args.repeat * len(layers))
            pers.append(per)
        sums = torch.tensor(pers, device=dev)
        dist.all_reduce(sums)
        sums = (sums / tp).sort().values
        if rank == 0:
            print(
                f"TP{tp} M={M}: {sums[0].item():.2f} min "
                f"{sums[len(sums) // 2].item():.2f} median us a layer (K1 + K2; out "
                f"{out.float().abs().mean().item():.4g}, graph "
                f"{gout.float().abs().mean().item():.4g})",
                flush=True,
            )
        if args.timeline and rank == 0:
            tl = runner.timeline(M).cpu()
            print(timeline(tl.view(1, *tl.shape)), flush=True)
    dist.destroy_process_group()


def timeline(tl) -> str:
    """Every mark any CTA passed, in median order: µs after the first K1 CTA's start
    (median and last CTA), per launch."""
    live = [tl[k][:, tl[k].ne(0).any(0)] for k in range(tl.shape[0])]
    t0 = min(t[t > 0].min() for t in live if t.numel())
    lines = []
    for k in range(tl.shape[0]):
        t = tl[k]
        marks = [j for j in range(t.shape[1]) if bool((t[:, j] != 0).any())]
        rows = []
        for j in marks:
            v = t[:, j][t[:, j] != 0].double()
            rows.append(
                (
                    ((v - t0) / 100.0).median().item(),
                    ((v - t0) / 100.0).max().item(),
                    j,
                    int(v.numel()),
                )
            )
        for med, mx, j, n in sorted(rows):
            lines.append(
                f"  launch {k} mark {j:2d}: median {med:7.2f} last {mx:7.2f} ({n} CTAs)"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
