"""Check the DSpark draft's committed KV in one launch (``gfx950/inject/kv``) against SGLang's chain on gfx950:
SGLang's RMSNorm (main_norm), its MXFP8 quant, aiter's stacked wkv GEMM and the per-stage SWA store
(``gfx950/inject``'s ``swa_store``): the stacked kv and every stage's cache bytes bitwise, at 5 to 8 rows, both
cache layouts (V4, V4.1), explicit slots and the committed-slot path, eager and under graph replay."""

import argparse

import torch

H, DIM, ROPE, STAGES, PAGE = 5120, 512, 64, 3, 64
PAGES, POSITIONS = 32, 4096


def same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--replays", type=int, default=10)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8
    from sglang.kernels.ops.quantization.mxfp8_quant import to_mxfp8
    from sglang.srt.layers.layernorm import RMSNorm

    from sgl_amd_mono.deepseek_v41.gfx950 import inject
    from sgl_amd_mono.deepseek_v41.gfx950.inject import kv as inject_kv

    store, fused = inject.ops(), inject_kv.ops()
    norm = RMSNorm(H, eps=1e-6).to(dev, torch.bfloat16)
    kvbuf, state, work = inject_kv.buffers(dev, STAGES)
    fails = []
    for v41 in (True, False):
        fails += check_layout(
            args,
            dev,
            v41,
            norm,
            store,
            fused,
            kvbuf,
            state,
            work,
            gemm_afp8wfp8,
            to_mxfp8,
            inject_kv,
        )
    print("ALL PASS" if not fails else f"FAIL at {fails}", flush=True)
    raise SystemExit(1 if fails else 0)


def check_layout(
    args,
    dev,
    v41,
    norm,
    store,
    fused,
    kvbuf,
    state,
    work,
    gemm_afp8wfp8,
    to_mxfp8,
    inject_kv,
):
    """The checks for one cache layout; the failures."""
    page_bytes = PAGE * (
        (512 + 16) if v41 else (576 + 8)
    )  # a page: the slots' rows, then their scales
    name = "V4.1" if v41 else "V4"
    fails = []
    for seed in range(args.seeds):
        g = torch.Generator(device=dev).manual_seed(11 + seed)
        with torch.no_grad():
            norm.weight.copy_(
                (1 + 0.1 * torch.randn(H, generator=g, device=dev)).bfloat16()
            )
        w = (
            (torch.randn(STAGES * DIM, H, generator=g, device=dev) * 40)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        ws = torch.randint(
            116,
            124,
            (STAGES * DIM // 32, H // 32),
            generator=g,
            device=dev,
            dtype=torch.uint8,
        )
        weights = [
            (1 + 0.1 * torch.randn(DIM, generator=g, device=dev)).bfloat16()
            for _ in range(STAGES)
        ]
        ang = torch.rand(POSITIONS, ROPE // 2, generator=g, device=dev) * 6.283
        freqs = [
            torch.stack([ang.cos(), ang.sin()], -1)
            .reshape(POSITIONS, ROPE)
            .contiguous()
            for _ in range(STAGES)
        ]
        for rows in range(inject_kv.MIN_ROWS, inject_kv.MAX_ROWS + 1):
            x = (torch.randn(rows, H, generator=g, device=dev) * 2).bfloat16()
            if seed % 2:  # wide rows: large and tiny values together
                x[:, ::7] *= 300.0
                x[:, 3::11] *= 1e-3
            pos = torch.randint(0, POSITIONS, (rows,), generator=g, device=dev)
            loc = torch.randperm(PAGES * PAGE, generator=g, device=dev)[:rows].int()
            loc[rows - 1] = -1  # a row without a slot
            want_caches = [
                torch.zeros(PAGES, page_bytes, dtype=torch.uint8, device=dev)
                for _ in range(STAGES)
            ]
            got_caches = [c.clone() for c in want_caches]
            normed = norm(x)
            q = to_mxfp8(normed)
            kv = gemm_afp8wfp8(
                q.data,
                w,
                q.scale,
                ws,
                torch.bfloat16,
                x_scale_group_size=32,
                w_scale_group_size=(32, 32),
            )
            store.swa_store(
                kv, DIM, weights, freqs, pos, loc, want_caches, 1e-6, PAGE, v41
            )

            def run():
                fused.inject_kv(
                    x,
                    norm.weight,
                    1e-6,
                    w.view(torch.uint8),
                    ws,
                    kvbuf,
                    state,
                    work,
                    weights,
                    freqs,
                    pos,
                    loc,
                    got_caches,
                    1e-6,
                    PAGE,
                    v41,
                )

            run()
            torch.cuda.synchronize()
            ok_kv = same(kvbuf[:rows], kv)
            ok_c = all(same(a, b) for a, b in zip(want_caches, got_caches))
            nkv = int((kvbuf[:rows].view(torch.int16) != kv.view(torch.int16)).sum())
            graph_ok = True
            if seed == 0:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for _ in range(args.replays):
                    for c in got_caches:
                        c.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    graph_ok &= all(same(a, b) for a, b in zip(want_caches, got_caches))
            ok = ok_kv and ok_c and graph_ok
            print(
                f"{name} seed {seed} rows {rows}: {'PASS' if ok else 'FAIL'} | kv differing {nkv} of {rows * STAGES * DIM}"
                f" | caches {'bitwise' if ok_c else 'DIFFER'} | replays {'bitwise' if graph_ok else 'DIFFER'}",
                flush=True,
            )
            if not ok:
                fails.append((name, seed, rows))
    # the committed-slot path: rows of a 6-row request, the first ``commit`` stored
    rows, width = 6, 6
    g = torch.Generator(device=dev).manual_seed(99)
    x = (torch.randn(rows, H, generator=g, device=dev) * 2).bfloat16()
    pos = torch.randint(0, POSITIONS, (rows,), generator=g, device=dev)
    cache_loc = torch.randperm(PAGES * PAGE, generator=g, device=dev)[:rows].long()
    mapping = torch.arange(PAGES * PAGE, device=dev).long()
    commit = torch.tensor([4], dtype=torch.int32, device=dev)
    want_caches = [
        torch.zeros(PAGES, page_bytes, dtype=torch.uint8, device=dev)
        for _ in range(STAGES)
    ]
    got_caches = [c.clone() for c in want_caches]
    kv = gemm_afp8wfp8(
        to_mxfp8(norm(x)).data,
        w,
        to_mxfp8(norm(x)).scale,
        ws,
        torch.bfloat16,
        x_scale_group_size=32,
        w_scale_group_size=(32, 32),
    )
    store.swa_store(
        kv,
        DIM,
        weights,
        freqs,
        pos,
        None,
        want_caches,
        1e-6,
        PAGE,
        v41,
        cache_loc,
        mapping,
        commit,
        width,
    )
    fused.inject_kv(
        x,
        norm.weight,
        1e-6,
        w.view(torch.uint8),
        ws,
        kvbuf,
        state,
        work,
        weights,
        freqs,
        pos,
        None,
        got_caches,
        1e-6,
        PAGE,
        v41,
        cache_loc,
        mapping,
        commit,
        width,
    )
    torch.cuda.synchronize()
    ok = all(same(a, b) for a, b in zip(want_caches, got_caches))
    print(f"{name} committed slots: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        fails.append((name, "committed"))
    return fails


if __name__ == "__main__":
    main()
