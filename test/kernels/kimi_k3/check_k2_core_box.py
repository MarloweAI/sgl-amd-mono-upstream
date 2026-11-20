# SPDX-License-Identifier: Apache-2.0
"""K2 built with ``core_box`` (each core chunk read once its (request, local
head) box holds the launch's tag) against K2 without it, on one GPU.

Run on a loopback tree (perf-engine ``plugin_tree.py make``), whose peer
pushes write every rank's slot of this rank's own buffer, so one rank's K2
finds its all-reduces complete. The boxes are set before the launch, as the
attention front of the same launch sets them: the two builds' outputs must be
bit for bit equal, and each run to run.
"""

import argparse
import sys

import torch
from check_common import BF, ROWS, moe_weights
from check_k2 import DEADLINE_S, layer_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    # rows past check_common's ROWS draw their own core, prefix and bank rows
    ap.add_argument("--widths", type=int, nargs="+", default=[ROWS])
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2
    from sgl_amd_mono.kimi_k3.gfx950.stages import oproj
    from sgl_amd_mono.kimi_k3.gfx950.tag import TAG_SLOTS

    peer = PeerBuffer(k2.peer_alloc_bytes(), None, 0, 1, dev)
    k2.init_peer(peer.bytes)
    peers = torch.tensor([peer.local] * k2.TP_, dtype=torch.int64, device=dev)
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(DEADLINE_S)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    scratch = PeerBuffer(k2.scratch_alloc_bytes(), None, 0, 1, dev).bytes
    scratch.zero_()
    w = moe_weights(0, dev, guint=False)
    w.update(layer_weights(0, dev))

    top = max(args.widths)
    gen = torch.Generator(device=dev).manual_seed(5)

    def rnd(*shape):
        return (torch.randn(*shape, generator=gen, device=dev) * 0.3).to(BF)

    wide = {
        "core": rnd(top, w["core"].shape[-1]),
        "prefix": rnd(top, w["prefix"].shape[-1]),
        "blocks": rnd(top, *w["blocks"].shape[1:]),
    }

    def launch(core_box, rows):
        if rows == ROWS:
            core, prefix0, blocks = w["core"], w["prefix"], w["blocks"]
        else:
            core, prefix0, blocks = (wide[n][:rows].contiguous() for n in ("core", "prefix", "blocks"))
        key = k2.K2Build(
            tokens=rows, nblocks=3, eps=w["eps"], out_eps=w["out_eps"], opt=k2.K2_OPT,
            core_box=core_box,
        )  # fmt: skip
        epoch[0].add_(1)
        if core_box:
            off, nbytes = k2.scratch_layout(key)["cbox"]
            boxes = scratch[off : off + nbytes].view(torch.int32).view(-1, 2)
            assert boxes.shape[0] == oproj.core_boxes(rows)
            ep = int(epoch[0].item())
            boxes[:, 0] = 1
            boxes[:, 1] = (ep % (1 << 22)) * TAG_SLOTS + args.layer + 1
        prefix = prefix0.clone()
        out = torch.empty_like(prefix)
        k2.k2_launch(
            key, core=core, w_o=w["w_o"], prefix=prefix, blocks=blocks,
            ares_nw=w["ares_nw"], ares_qk=w["ares_qk"], in_nw=w["in_nw"],
            w_gate=w["w_gate"], bias=w["bias"], w_ld=w["w_ld"], w_sgu=w["w_sgu"],
            w_sd=w["w_sd"], w13=w["w13"], w13s=w["w13s"], w2=w["w2"], w2s=w["w2s"],
            ln_w=w["ln_w"], w_up=w["w_up_shard"], out=out, scratch=scratch,
            peers=peers, rank=0, epoch=epoch, layer=args.layer,
        )  # fmt: skip
        torch.cuda.synchronize()
        return out.clone(), prefix.clone()

    fails = []
    for rows in args.widths:
        runs = {cb: [launch(cb, rows), launch(cb, rows)] for cb in (False, True)}
        for cb, (a, b) in runs.items():
            same = all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(a, b))
            print(f"rows={rows} core_box={cb}: run to run bitwise {same}", flush=True)
            if not same:
                fails.append(f"rows={rows} core_box={cb} run to run")
        eq = all(
            torch.equal(x.view(torch.int16), y.view(torch.int16))
            for x, y in zip(runs[False][0], runs[True][0])
        )
        print(f"rows={rows} core_box=True vs False: out and prefix bitwise {eq}", flush=True)
        if not eq:
            fails.append(f"rows={rows} core_box vs plain")
    assert runs[False][0][0].dtype == BF
    print("ALL PASS" if not fails else f"FAILED: {fails}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
