# SPDX-License-Identifier: Apache-2.0
"""The wide sparse-attention core (``gfx950/attention/wide_core.py``) against a float32
reference, on one GPU: synthetic requests of six verify rows (windows of 128 consecutive
positions, top-512 sets sharing most of their keys), V4 fp8_ds_mla pages, the sink and the
inverse RoPE of the last 64 dims.

The kernel sums in its union's order, so it matches the reference to bf16 rounding, not
bit for bit; it must reproduce its own bits launch to launch.

    python3 check_wide_core.py [--reqs 1 3 10 16] [--heads 16 32]
"""

import argparse
import sys

import torch

G, D, NOPE, ROPE, DATA, REC = 6, 512, 448, 64, 576, 584
SCALE = 512**-0.5
FP8 = torch.float8_e4m3fn


class Pool:
    """u8 [pages, page * 584]: a page's data rows, then its 8-byte scale rows."""

    def __init__(self, pages, page):
        self.pages, self.page = pages, page
        self.buf = torch.zeros(pages, page * REC, dtype=torch.uint8, device="cuda")

    def _offs(self, slots):
        p, o = (slots // self.page).long(), (slots % self.page).long()
        d = o[:, None] * DATA + torch.arange(DATA, device="cuda")
        s = self.page * DATA + o[:, None] * 8 + torch.arange(8, device="cuda")
        return p[:, None], d, s

    def put(self, slots, rec):
        p, d, s = self._offs(slots)
        self.buf[p, d] = rec[:, :DATA]
        self.buf[p, s] = rec[:, DATA:]

    def get(self, slots):
        p, d, s = self._offs(slots)
        return torch.cat([self.buf[p, d], self.buf[p, s]], 1)


def records(n, gen):
    rec = torch.zeros(n, REC, dtype=torch.uint8, device="cuda")
    rec[:, :NOPE] = (torch.randn(n, NOPE, generator=gen, device="cuda") * 2).to(FP8).view(torch.uint8)
    rope = (torch.randn(n, ROPE, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    rec[:, NOPE:DATA] = rope.view(torch.uint8).view(n, 2 * ROPE)
    rec[:, DATA : DATA + 7] = torch.randint(118, 124, (n, 7), generator=gen, device="cuda").to(torch.uint8)
    return rec


def dequant(rec):
    x = rec[:, :NOPE].view(FP8).float().view(-1, 7, 64)
    x = x * torch.pow(2.0, rec[:, DATA : DATA + 7].float() - 127)[:, :, None]
    return torch.cat([x.view(-1, NOPE), rec[:, NOPE:DATA].contiguous().view(torch.bfloat16).float()], 1)


def inputs(reqs, heads, seed, ctx=8192, share=0.85, page=256):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    T = reqs * G
    wpool = Pool(reqs * 256 // page + 2, page)
    epool = Pool(reqs * (ctx // page + 1) + 2, page)
    win = torch.full((T, 128), -1, dtype=torch.int32, device="cuda")
    top = torch.full((T, 512), -1, dtype=torch.int32, device="cuda")
    pos = torch.zeros(T, dtype=torch.int64, device="cuda")
    for r in range(reqs):
        wperm = torch.randperm(wpool.pages * page - 1, generator=gen, device="cuda")[:256] + 1
        eperm = torch.randperm(epool.pages * page - 1, generator=gen, device="cuda")[:ctx] + 1
        base = torch.randperm(ctx, generator=gen, device="cuda")[:512]
        for j in range(G):
            t, p = r * G + j, ctx + j
            pos[t] = p
            win[t] = wperm[(p - torch.arange(128, device="cuda")) % 256].int()
            keep = torch.rand(512, generator=gen, device="cuda") < share
            sel = torch.unique(torch.where(keep, base, torch.randperm(ctx, generator=gen, device="cuda")[:512]))
            rest = torch.randperm(ctx, generator=gen, device="cuda")
            sel = torch.cat([sel, rest[~torch.isin(rest, sel)][: 512 - sel.numel()]])
            top[t] = eperm[sel[torch.randperm(512, generator=gen, device="cuda")]].int()
    for pool, lst in ((wpool, win), (epool, top)):
        used = lst[lst >= 0].unique()
        pool.put(used, records(used.numel(), gen))
    q = (torch.randn(T, heads, D, generator=gen, device="cuda") * 0.4).to(torch.bfloat16)
    sink = torch.randn(heads, generator=gen, device="cuda") * 0.5
    inv = 1.0 / (160000 ** (torch.arange(0, ROPE, 2, device="cuda").float() / ROPE))
    ang = torch.arange(ctx + 64, device="cuda").float()[:, None] * inv[None, :]
    freqs = torch.view_as_real(torch.polar(torch.ones_like(ang), ang)).flatten(-2).contiguous()
    n = lambda w: torch.full((T,), w, dtype=torch.int32, device="cuda")  # noqa: E731
    return q, win, n(128), top, n(512), wpool, epool, sink, freqs, pos


def reference(q, win, win_n, top, top_n, wpool, epool, sink, freqs, pos):
    T, H = q.shape[:2]
    out = torch.empty(T, H, D, dtype=torch.bfloat16, device="cuda")
    for t in range(T):
        ks = []
        for lst, n, pool in ((win, win_n, wpool), (top, top_n, epool)):
            s = lst[t, : int(n[t])]
            ks.append(dequant(pool.get(s[s >= 0])))
        k = torch.cat(ks)
        sc = torch.cat([(q[t].float() @ k.T) * SCALE, sink[:, None]], 1)
        out[t] = (torch.softmax(sc, 1)[:, :-1] @ k).to(torch.bfloat16)
    x = out.float()
    fr = freqs[pos]
    cos, sin = fr[:, 0::2][:, None, :], fr[:, 1::2][:, None, :]
    e, o = x[..., NOPE::2].clone(), x[..., NOPE + 1 :: 2].clone()
    x[..., NOPE::2], x[..., NOPE + 1 :: 2] = e * cos + o * sin, o * cos - e * sin
    return x.to(torch.bfloat16)


def run(reqs, heads, seed):
    from sgl_amd_mono.deepseek_v41.gfx950.attention import wide_core as wc

    q, win, win_n, top, top_n, wpool, epool, sink, freqs, pos = args = inputs(reqs, heads, seed)
    i32 = dict(dtype=torch.int32, device="cuda")
    b = wc.CoreBuild(
        reqs=reqs, heads=heads, splits=wc.splits_for(reqs, heads),
        window=wc.Cache(page=wpool.page, stride=wpool.buf.stride(0)),
        extra=wc.Cache(page=epool.page, stride=epool.buf.stride(0)),
        qs=q.stride(0), qh=q.stride(1), scale=SCALE, rope_stride=freqs.stride(0), pos_words=2,
    )  # fmt: skip
    bufs = [torch.empty(reqs, G * w, **i32) for w in (128, 128, 512, 512)]
    counts = [torch.empty(reqs, **i32) for _ in range(2)]
    npart, nml, nflag = wc.workspace(b)
    part = torch.empty(npart, dtype=torch.float16, device="cuda")
    pml = torch.empty(nml, dtype=torch.float32, device="cuda")
    flags = torch.zeros(nflag, **i32)
    out = torch.empty(reqs * G, heads, D, dtype=torch.bfloat16, device="cuda")

    def launch():
        wc.union_build(128, win.stride(0), wpool.pages * wpool.page)(
            win.data_ptr(), win_n.data_ptr(), bufs[0].data_ptr(), bufs[1].data_ptr(), counts[0].data_ptr(), reqs
        )
        wc.union_build(512, top.stride(0), epool.pages * epool.page)(
            top.data_ptr(), top_n.data_ptr(), bufs[2].data_ptr(), bufs[3].data_ptr(), counts[1].data_ptr(), reqs
        )
        wc.core_build(b)(
            q.data_ptr(), wpool.buf.data_ptr(), epool.buf.data_ptr(), bufs[0].data_ptr(),
            bufs[1].data_ptr(), counts[0].data_ptr(), bufs[2].data_ptr(), bufs[3].data_ptr(),
            counts[1].data_ptr(), sink.data_ptr(), freqs.data_ptr(), pos.data_ptr(), out.data_ptr(),
            part.data_ptr(), pml.data_ptr(), flags.data_ptr(), reqs, wc.groups_per_xcd(reqs, heads),
        )  # fmt: skip
        return out

    first = launch().clone()
    same = all(torch.equal(first, launch()) for _ in range(20))
    ref = reference(*args)
    d = (first.float() - ref.float()).abs()
    rel = (d.norm() / ref.float().norm()).item()
    ok = same and rel < 1e-2
    print(f"{'PASS' if ok else 'FAIL'} reqs={reqs} heads={heads} rel_l2={rel:.2e} max_abs={d.max().item():.2e} "
          f"deterministic={same}", flush=True)  # fmt: skip
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", type=int, nargs="*", default=[1, 3, 10, 16])
    ap.add_argument("--heads", type=int, nargs="*", default=[16, 32])
    a = ap.parse_args()
    ok = all(run(r, h, 1000 * r + h) for h in a.heads for r in a.reqs)
    print("ALL PASS" if ok else "SOME FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
