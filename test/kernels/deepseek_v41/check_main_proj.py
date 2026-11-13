"""Check the DSpark draft's main_proj split over the TP ranks (``gfx950/inject/proj``, torchrun TP4 or TP8) against
SGLang's replicated aiter route (``aiter_mxfp8_blockscaled_linear``: the MXFP8 quant, then aiter's GEMM): every
rank's whole rows bitwise, at 5 to 8 rows (aiter's BLOCK_M 8 packed config), wide-range rows, eager and under graph replay; then the time of each
(a graph of chained calls, max over ranks)."""

import argparse
import os

import torch
import torch.distributed as dist

H, K = 5120, 15360


def same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def flydsl_layer(w, ws, proj, rank) -> list:
    """A main_proj layer as SGLang prepares it under SGLANG_OPT_HIP_FLYDSL_MXFP8: the weight and its E8M0
    scales stay the tensors the split reads, and SGLang's route takes FlyDSL's GEMM only from more rows
    than a commit has (else aiter's, the reference below)."""
    from sglang.srt.layers.quantization import fp8_hip

    from sgl_amd_mono.deepseek_v41.glue import inject

    lin = torch.nn.Module()
    lin.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    lin.weight_scale_mx_e8m0 = torch.nn.Parameter(ws.clone(), requires_grad=False)
    fp8_hip._prepare_flydsl_mxfp8(lin)
    bad = []
    if getattr(lin, "weight_scale_mx_flydsl", None) is None:
        bad.append("no FlyDSL scale copy (FlyDSL unavailable)")
    if not (same(lin.weight, w) and same(lin.weight_scale_mx_e8m0, ws)):
        bad.append("the FlyDSL preparation changed the weight or its E8M0 scales")
    if fp8_hip._FLYDSL_MXFP8_MIN_M <= proj.MAX_ROWS:
        bad.append(
            f"FlyDSL's GEMM from {fp8_hip._FLYDSL_MXFP8_MIN_M} rows, a commit has up to {proj.MAX_ROWS}"
        )
    model = type("M", (), {})()
    stage = torch.nn.Module()
    stage.main_proj = lin
    lin.quant_method = type(
        "Q",
        (),
        {
            "mxfp8_dense_backend": type(
                "B", (), {"is_gfx95_mxfp8_aiter": lambda self: True}
            )()
        },
    )()
    model.stages = [stage]
    inject._EXCHANGE.append(object())
    try:
        why = inject._main_proj_why(model)
    finally:
        inject._EXCHANGE.pop()
    if why is not None:
        bad.append(f"the static rule refuses the FlyDSL-prepared layer: {why}")
    if rank == 0:
        print(
            f"FlyDSL-prepared layer: {'PASS' if not bad else 'FAIL: ' + '; '.join(bad)}",
            flush=True,
        )
    return ["flydsl layer"] if bad else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--chain", type=int, default=20)
    ap.add_argument("--rows", type=int, default=6)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    cpu = dist.new_group(backend="gloo")
    from sglang.kernels.ops.gemm.mxfp8_aiter_gfx95 import aiter_mxfp8_blockscaled_linear

    from sgl_amd_mono.deepseek_v41.gfx950.inject import proj

    x_ = proj.Exchange(cpu, rank, tp, dev, deadline_s=20.0)
    fails = []
    for seed in range(args.seeds):
        g = torch.Generator(device=dev).manual_seed(
            5 + seed
        )  # every rank the same tensors
        w = (
            (torch.randn(H, K, generator=g, device=dev) * 40)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        ws = torch.randint(
            116, 124, (H // 32, K // 32), generator=g, device=dev, dtype=torch.uint8
        )
        if seed == 0:
            fails += flydsl_layer(w, ws, proj, rank)
        cols = x_.columns(w, ws)
        for rows in range(proj.MIN_ROWS, proj.MAX_ROWS + 1):
            x = (torch.randn(rows, K, generator=g, device=dev) * 2).bfloat16()
            if seed % 2:  # wide rows: large and tiny values together
                x[:, ::7] *= 300.0
                x[:, 3::11] *= 1e-3
            if (
                seed % 4 == 2
            ):  # extreme groups: the quant's scales near both ends of bf16's range
                x[:, : K // 2 : 5] *= 1e30
                x[:, K // 2 :] *= 1e-30
                x[:, K // 2 : K // 2 + 64] = 0.0
            want = aiter_mxfp8_blockscaled_linear(x, w, ws)
            got = x_.run(x, *cols)
            torch.cuda.synchronize()
            ok = same(got, want)
            ndiff = int((got.view(torch.int16) != want.view(torch.int16)).sum())
            graph_ok = True
            if seed == 0 and rows in (6, 8):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    x_.run(x, *cols)
                for _ in range(args.replays):
                    x_.out.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    graph_ok &= same(x_.out[:rows], want)
            ok_all = torch.tensor([int(ok and graph_ok)], device=dev)
            dist.all_reduce(ok_all, op=dist.ReduceOp.MIN)
            if rank == 0:
                print(
                    f"seed {seed} rows {rows}: {'PASS' if ok_all.item() else 'FAIL'} | rank0 differing {ndiff} of "
                    f"{rows * H} | replays {'bitwise' if graph_ok else 'DIFFER'}",
                    flush=True,
                )
            if not ok_all.item():
                fails.append((seed, rows))
    # timing: a graph of ``chain`` calls each, max over ranks
    g = torch.Generator(device=dev).manual_seed(1)
    w = (
        (torch.randn(H, K, generator=g, device=dev) * 40)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    ws = torch.randint(
        116, 124, (H // 32, K // 32), generator=g, device=dev, dtype=torch.uint8
    )
    cols = x_.columns(w, ws)
    x = (torch.randn(args.rows, K, generator=g, device=dev) * 2).bfloat16()

    def timed(fn):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.chain):
                fn()
        out = []
        for _ in range(5):
            dist.barrier()
            torch.cuda.synchronize()
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            graph.replay()
            b.record()
            torch.cuda.synchronize()
            out.append(a.elapsed_time(b) * 1000 / args.chain)
        t = torch.tensor([sorted(out)[2]], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    t_ref = timed(lambda: aiter_mxfp8_blockscaled_linear(x, w, ws))
    t_ours = timed(lambda: x_.run(x, *cols))
    # a stamped run: per CTA, ticks (10 ns) from the rank's first CTA start, median / max over CTAs
    st = torch.zeros(
        proj.STATE_INTS + 2 * proj.STAMPS * (H // 16) * 6, dtype=torch.int32, device=dev
    )
    for _ in range(3):
        x_.run(x, *cols, state=st)
    torch.cuda.synchronize()
    ticks = (
        st[proj.STATE_INTS :]
        .view(torch.int64)
        .view(-1, proj.STAMPS)[: (H // tp // 16) * 6]
        .double()
    )
    base = ticks[:, 0].min()
    rel = (ticks - base) / 100.0  # us
    names = [
        "start",
        "slice+weights",
        "chain",
        "partials seen",
        "pushed",
        "copied out",
        "slice (wave 1)",
    ]
    lines = []
    for k, nm in enumerate(names):
        col = rel[:, k]
        col = col[ticks[:, k] > 0]
        if k in (3, 4):
            col = col[col > 0]
        if col.numel():
            lines.append(f"{nm} {col.median().item():.2f}/{col.max().item():.2f}")
    print(
        f"rank {rank} stamps (us, median/max over CTAs): " + " | ".join(lines),
        flush=True,
    )
    dist.barrier()
    if rank == 0:
        print(
            f"time a call, {args.rows} rows, TP{tp} (max over ranks): SGLang replicated {t_ref:.1f} us, "
            f"split {t_ours:.1f} us",
            flush=True,
        )
        print("ALL PASS" if not fails else f"FAIL at {fails}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
