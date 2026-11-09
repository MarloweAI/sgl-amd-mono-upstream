"""Check the head's combine and RMSNorm in one launch (``fused.post_combine_norm``) against the
separate launches (``fused.post_combine``, then SGLang's RMSNorm) on gfx950: the settled streams
and the normed rows bitwise at every width, eager and under graph replay; and both paths' time.
One GPU (torchrun --nproc-per-node 1)."""

import argparse

import torch

from sgl_amd_mono.deepseek_v41 import fused

HIDDEN, HC, EPS = 5120, 4, 1e-6
WIDTHS = (1, 2, 5, 6, 10, 12, 16, 24, 48)


def inputs(rows, g, dev):
    bf = torch.bfloat16
    return (
        torch.randn(rows, HIDDEN, generator=g, device=dev).to(bf),
        torch.randn(rows, HC, HIDDEN, generator=g, device=dev).to(bf),
        torch.rand(rows, HC, generator=g, device=dev) * 2,
        torch.rand(rows, HC, HC, generator=g, device=dev),
        torch.rand(rows, HC, generator=g, device=dev),
    )


def graph_us(fn, reps):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        graph.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / reps


def same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    from sglang.srt.layers.layernorm import RMSNorm

    g = torch.Generator(device=dev).manual_seed(5)
    norm = RMSNorm(HIDDEN, eps=EPS).to(
        dev, torch.bfloat16
    )  # the model loads it in bf16
    with torch.no_grad():
        norm.weight.copy_(
            (1 + 0.1 * torch.randn(HIDDEN, generator=g, device=dev)).bfloat16()
        )
    fails = []
    for rows in WIDTHS:
        bad = 0
        for n in range(args.replays):
            x, res, post, comb, pre = inputs(rows, g, dev)
            if n % 2:  # wide rows: large and tiny values together
                x[:, ::7] *= 300.0
                res[:, :, ::5] *= 1e-3
            want_s, want_c = fused._post_combine(x, res, post, comb, pre)
            want = norm(want_c)
            got_s, got = fused._post_combine_norm(x, res, post, comb, pre, norm)
            got_c, got_n2 = fused.combine_and_norm(x, res, post, comb, pre, norm)
            torch.cuda.synchronize()
            ok_s, ok_n = (
                same(want_s, got_s),
                same(want, got) and same(want, got_n2) and same(want_c, got_c),
            )
            if rows == 48 and n < 2:
                # the norm alone: an identity combine of Triton's combined rows
                e0 = torch.zeros(rows, HC, device=dev)
                e0[:, 0] = 1.0
                zr = torch.zeros_like(res)
                zc = torch.zeros(rows, HC, HC, device=dev)
                _, alone = fused._post_combine_norm(
                    want_c.contiguous(), zr, e0, zc, e0, norm
                )
                print(
                    f"  the norm alone: {int((alone.view(torch.int16) != want.view(torch.int16)).sum())} values "
                    f"differ from SGLang's",
                    flush=True,
                )
                # which rounding of the combine's sum Triton's kernel made
                sf = want_s.double()
                p = pre.double()
                fma = (p[:, 0, None] * sf[:, 0]).float()
                sep = fma.clone()
                for i in range(1, HC):
                    fma = (p[:, i, None] * sf[:, i] + fma.double()).float()
                    sep = (
                        (p[:, i, None] * sf[:, i]).float().double() + sep.double()
                    ).float()
                for name, acc in (("fma", fma), ("mul+add", sep)):
                    y = acc.to(torch.bfloat16)
                    print(
                        f"  combine as {name}: {int((y.view(torch.int16) != want_c.view(torch.int16)).sum())} values "
                        f"differ from Triton's",
                        flush=True,
                    )
            if not (ok_s and ok_n) and rows in (5, 48) and n < 4:
                d = (want.float() - got.float()).abs()
                rowbad = (d.amax(-1) > 0).sum().item()
                print(
                    f"  rows {rows} replay {n}: streams {'same' if ok_s else 'DIFFER'}, normed differ at "
                    f"{int((d > 0).sum())} values in {rowbad} rows, max |d| {d.max().item():.3g}",
                    flush=True,
                )
            bad += not (ok_s and ok_n)
        # graph replay of the fused launch on fixed buffers
        x, res, post, comb, pre = inputs(rows, g, dev)
        out_s, out = torch.empty_like(res), torch.empty_like(x)

        def fused_run():
            from sgl_amd_mono.deepseek_v41.gfx950.norm import ops

            ops().combine_norm(
                x, res, post, comb, pre, norm.weight.data, EPS, out_s, out
            )

        def separate_run():
            norm(fused._post_combine(x, res, post, comb, pre)[1])

        t_fused, t_sep = graph_us(fused_run, 200), graph_us(separate_run, 200)
        want_s, want_c = fused._post_combine(x, res, post, comb, pre)
        if not (same(want_s, out_s) and same(norm(want_c), out)):
            bad += 1
        status = "PASS" if bad == 0 else f"FAIL ({bad} of {args.replays + 1} differ)"
        print(
            f"rows {rows:2d}: {status}; one launch {t_fused:.1f} us, combine + norm {t_sep:.1f} us",
            flush=True,
        )
        if bad:
            fails.append(rows)
    print("ALL PASS" if not fails else f"FAIL at rows {fails}")


if __name__ == "__main__":
    main()
