# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 layer launch against SGLang's ops, at TP8 (RFC #43949).

Three outputs, each against the spec's bound:

* ``prefix``: o_proj, the attention-TP all-reduce and the residual add.
* ``x``: the MLP attention-residual seam, then the FFN input norm.
* ``out``: the layer output, the MoE on top of the seam.

The MoE is compared on the seam output the kernel itself produced, read back
out of its scratch. A wrong seam then cannot hide behind a MoE that agrees
with whatever it was given.

Both prefix modes run. A plain layer accumulates into the prefix. A block-write
layer restarts it at the attention output. The control at the end of the file
is why that matters. Without it, a kernel that ignored the prefix would match
both references and prove nothing. So would a harness reading a stale
scratch.
"""

import argparse

import torch
from check_common import (
    BF,
    MAX_BLOCKS,
    NVB,
    ROWS,
    TP,
    all_reduce_fp32,
    cos,
    finish,
    moe_reference,
    moe_weights,
    randn,
    rel,
    rms,
    setup,
)

from sgl_amd_mono.kimi_k3.spec import NUMERICS

# A wait's deadline here: long enough that only a real stall trips it.
DEADLINE_S = 30.0


def seam_reference(core, w_o, prefix_in, blocks, nvb, w, *, reset):
    """o_proj, its all-reduce and the MLP attention-residual seam, in torch.

    This mirrors AttnResidual's eager path. The sources are the ``nvb`` banked
    rows and the updated prefix. Each is scored through the score RMSNorm and
    the one-row projection. A softmax over the sources gives the mix, and the
    output RMSNorm follows. Returns (the updated prefix, the MoE input).
    """
    import torch.nn.functional as F

    attn = all_reduce_fp32(F.linear(core, w_o)).float()
    # A block-write layer restarts the prefix at the attention output.
    prefix = (attn if reset else prefix_in.float() + attn).to(BF)
    rows = torch.cat([blocks[:, :nvb, :], prefix.unsqueeze(1)], dim=1)
    scores = (rms(rows, w["ares_nw"], w["eps"]) * w["ares_qk"].float()).sum(-1)
    probs = torch.softmax(scores, dim=-1)
    mixed = (probs.unsqueeze(-1) * rows.float()).sum(dim=1)
    return prefix, rms(mixed, w["in_nw"], w["out_eps"]).to(BF)


def layer_weights(rank, device):
    """The layer launch's own weights, on top of the MoE's.

    o_proj is a TP shard, so it is drawn per rank. The attention-residual
    stream and its two norms are replicated, so they have to match on every
    rank."""
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN, PROJ

    gc = torch.Generator(device=device).manual_seed(23)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    return {
        "w_o": randn(gr, HIDDEN, PROJ, scale=0.02),
        "core": randn(gr, ROWS, PROJ, scale=0.3),
        # The bank is allocated for every block of the model, so only its
        # first nvb rows are live: that is the production stride.
        "blocks": randn(gc, ROWS, MAX_BLOCKS, HIDDEN, scale=0.3),
        "prefix": randn(gc, ROWS, HIDDEN, scale=0.3),
        "ares_nw": (1 + randn(gc, HIDDEN, scale=0.1)).to(BF),
        "ares_qk": randn(gc, HIDDEN, scale=0.02),
        "in_nw": (1 + randn(gc, HIDDEN, scale=0.1)).to(BF),
        "eps": 1e-5,
        "out_eps": 1e-5,
    }


def run(
    k2,
    w,
    scratch,
    *,
    reset,
    peers,
    epoch,
    rank,
    layer,
    guint,
    fold=False,
    tl=None,
    opt=0,
):
    """One layer launch -> (out, the updated prefix, the build key the caller
    reads the seam output back with)."""
    key = k2.K2Build(
        tokens=ROWS,
        nblocks=NVB,
        eps=w["eps"],
        out_eps=w["out_eps"],
        reset=reset,
        guint=guint,
        fold=fold,
        opt=opt,
        timeline=tl is not None,
    )
    # The kernel updates the prefix in place, so hand it a copy and keep the
    # input for the reference. Under ``fold`` it reads the input in place and
    # writes a buffer of its own.
    prefix = torch.empty_like(w["prefix"]) if fold else w["prefix"].clone()
    out = torch.empty_like(prefix)
    k2.k2_launch(
        key,
        core=w["core"],
        w_o=w["w_o"],
        prefix=prefix,
        blocks=w["blocks"],
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        w_gate=w["w_gate"],
        bias=w["bias"],
        w_ld=w["w_ld"],
        w_sgu=w["w_sgu"],
        w_sd=w["w_sd"],
        w13=w["w13"],
        w13s=w["w13s"],
        w2=w["w2"],
        w2s=w["w2s"],
        ln_w=w["ln_w"],
        w_up=w["w_up_shard"],
        out=out,
        scratch=scratch,
        peers=peers.addresses,
        rank=rank,
        epoch=epoch,
        layer=layer,
        prefix_in=w["prefix"] if fold else None,
        tl=tl,
    )
    return out, prefix, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=int, default=50)
    args = ap.parse_args()

    rank, dev, cpu = setup()
    import torch.distributed as dist

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    # The launch runs under the guard, so the peer buffer carries the flag word
    # a rank and the epoch buffer carries the guard block. Without a deadline
    # every wait reads garbage, decides it is late and faults the GPU at 0x0.
    #
    # Size this from peer_alloc_bytes(), the same way the runner does. A buffer
    # sized for this check's own width puts the flags where the launch does not
    # read them, and the check then passes on a layout no server uses.
    peers = PeerBuffer(k2.peer_alloc_bytes(), cpu, rank, TP, dev)
    k2.init_peer(peers.bytes)
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(DEADLINE_S)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    scratch = torch.zeros(k2.scratch_alloc_bytes(), dtype=torch.uint8, device=dev)
    w = moe_weights(rank, dev, guint=False)
    w.update(layer_weights(rank, dev))

    fails, seen, first = [], {}, {}
    for tag, reset in (("accumulate", False), ("reset", True)):
        epoch[0].add_(1)
        out, prefix, key = run(
            k2, w, scratch, reset=reset, peers=peers, epoch=epoch,
            rank=rank, layer=1, guint=False,
        )  # fmt: skip
        torch.cuda.synchronize()
        # the seam output the kernel wrote, out of its own scratch
        off, nbytes = k2.scratch_layout(key)["xrow"]
        xrow = scratch[off : off + nbytes].view(BF).view(ROWS, HIDDEN).clone()
        out, prefix = out.clone(), prefix.clone()
        ref_prefix, ref_x = seam_reference(
            w["core"], w["w_o"], w["prefix"], w["blocks"], NVB, w, reset=reset
        )
        ref_moe = moe_reference(xrow, w, rank)
        gathered = [torch.empty_like(out) for _ in range(TP)]
        dist.all_gather(gathered, out.contiguous())
        checks = {
            "finite": bool(torch.isfinite(out).all()),
            "ranks agree": all(torch.equal(gathered[0], g) for g in gathered[1:]),
            "prefix": cos(prefix, ref_prefix) > NUMERICS.bound("k2.prefix.cosine"),
            "x": cos(xrow, ref_x) > NUMERICS.bound("k2.x.cosine"),
            "out": cos(out, ref_moe) > NUMERICS.bound("k2.out.cosine")
            and rel(out, ref_moe) < NUMERICS.bound("k2.out.rel"),
        }
        if rank == 0:
            print(
                f"{tag}: prefix cos {cos(prefix, ref_prefix):.6f}  "
                f"x cos {cos(xrow, ref_x):.6f}  "
                f"out cos {cos(out, ref_moe):.6f} rel {rel(out, ref_moe):.4g}",
                flush=True,
            )
        fails += [f"{tag}/{n}" for n, ok in checks.items() if not ok]
        seen[tag] = xrow
        first[tag] = out

    # Control: the two modes read the input prefix differently, so they must
    # disagree. Without this the two checks above would pass for a kernel that
    # ignored the prefix entirely.
    if cos(seen["accumulate"], seen["reset"]) >= NUMERICS.bound("k2.x.cosine"):
        fails.append("the two prefix modes agree, so neither check means anything")

    # The folded launch (the serving form): the seam reads the stream in place
    # and the reduce writes out + the updated prefix, which must be bit for bit
    # the unfolded launch's torch add. The input must stay untouched.
    for tag, reset in (("accumulate", False), ("reset", True)):
        before = w["prefix"].clone()
        epoch[0].add_(1)
        plain, plain_prefix, _ = run(
            k2, w, scratch, reset=reset, peers=peers, epoch=epoch,
            rank=rank, layer=1, guint=False,
        )  # fmt: skip
        want, plain_prefix = (plain + plain_prefix).clone(), plain_prefix.clone()
        epoch[0].add_(1)
        folded, folded_prefix, _ = run(
            k2, w, scratch, reset=reset, peers=peers, epoch=epoch,
            rank=rank, layer=1, guint=False, fold=True,
        )  # fmt: skip
        torch.cuda.synchronize()
        ok = {
            "fold out": torch.equal(folded, want),
            "fold prefix": torch.equal(folded_prefix, plain_prefix),
            "fold input untouched": torch.equal(w["prefix"], before),
        }
        if rank == 0:
            print(
                f"fold/{tag}: " + ", ".join(f"{k} {v}" for k, v in ok.items()),
                flush=True,
            )
        fails += [f"{tag}/{n}" for n, v in ok.items() if not v]

    # The OPT_* bits, each alone and all together, against opt=0 on the served
    # form: bit for bit. A sentinel word is pushed again only two launches
    # later, so consecutive launches take layers of alternating parity.
    served = dict(fold=True, guint=False)
    bits = [
        k2.OPT_SENT, k2.OPT_TAG_X, k2.OPT_SENT_LATR,
        k2.OPT_SENT_FINAL, k2.OPT_SDOWN_FIRST, k2.OPT_RLOG_PARTS, k2.OPT_BATCH,
        k2.K2_OPT,
    ]  # fmt: skip
    layer_no = 0
    for tag, reset in (("accumulate", False), ("reset", True)):
        outs = {}
        for opt in [0] + bits:
            epoch[0].add_(1)
            layer_no += 1
            o, p, _ = run(
                k2, w, scratch, reset=reset, peers=peers, epoch=epoch,
                rank=rank, layer=layer_no, opt=opt, **served,
            )  # fmt: skip
            torch.cuda.synchronize()
            outs[opt] = (o.clone(), p.clone())
        for opt in bits:
            same = torch.equal(outs[0][0], outs[opt][0]) and torch.equal(
                outs[0][1], outs[opt][1]
            )
            if rank == 0:
                print(f"opt/{tag}: {opt} == 0 bit for bit: {same}", flush=True)
            if not same:
                fails.append(f"opt{opt}/{tag}")

    # A graph of two launches (as a step's layers alternate) under every bit,
    # replayed: each replay bit for bit the eager pair.
    def pair(opt):
        o1, _, _ = run(
            k2, w, scratch, reset=False, peers=peers, epoch=epoch,
            rank=rank, layer=1, opt=opt, **served,
        )  # fmt: skip
        o2, _, _ = run(
            k2, w, scratch, reset=False, peers=peers, epoch=epoch,
            rank=rank, layer=2, opt=opt, **served,
        )  # fmt: skip
        return o1, o2

    epoch[0].add_(1)
    want = [o.clone() for o in pair(k2.K2_OPT)]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = pair(k2.K2_OPT)
    ok = True
    for i in range(args.replays):
        epoch[0].add_(1)
        graph.replay()
        if i % 10 == 0 or i == args.replays - 1:
            torch.cuda.synchronize()
            ok = ok and all(torch.equal(g, x) for g, x in zip(got, want))
    if rank == 0:
        print(f"opt {k2.K2_OPT}: {args.replays} replays bit for bit: {ok}", flush=True)
    if not ok:
        fails.append("opt replay")

    # The captured graph replays the launch. A replay must be bit for bit what
    # the eager launch produced, which is what makes the decode graph safe.
    epoch[0].add_(1)
    eager_out, eager_prefix, _ = run(
        k2, w, scratch, reset=False, peers=peers, epoch=epoch,
        rank=rank, layer=1, guint=False,
    )  # fmt: skip
    torch.cuda.synchronize()
    eager_out, eager_prefix = eager_out.clone(), eager_prefix.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_prefix, _ = run(
            k2, w, scratch, reset=False, peers=peers, epoch=epoch,
            rank=rank, layer=1, guint=False,
        )  # fmt: skip
    for i in range(args.replays):
        epoch[0].add_(1)
        graph.replay()
        if i % 10 == 0 or i == args.replays - 1:
            torch.cuda.synchronize()
            if not torch.equal(graph_out, eager_out):
                fails.append(f"replay {i}: the output is not the eager one")
                break
    torch.cuda.synchronize()
    if rank == 0:
        print(f"{args.replays} replays bit for bit: {not fails}", flush=True)

    peers.close()
    finish(fails, dev)


if __name__ == "__main__":
    main()
