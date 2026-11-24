"""Kernel-only time of the GLM-5.3-Flash layer launch, under torchrun (TP 2, 4 or 8):

    torchrun --nproc-per-node 4 test/kernels/glm53_flash/time_layers.py \\
        --kind model --rows 2,4,8 [--pos 2047] [--layers 12] [--row-major] [--per-row-moe]

One HIP graph holds a step of launches, inputs chained layer to layer, every row
live. Each layer has its own random weights (``check_layer.py``'s, seeded by layer,
so they stream from HBM as in a served step), its own state or caches, and every
sparse-attention row sits at position ``--pos``; with ``--verify L`` (``kda``,
``dsa`` or ``model``) the launch is an MTP verify step, rows // L requests whose L
rows sit at ``--pos`` .. ``--pos + L - 1`` on the request's page table (a KDA
request's draft rows on one state, with the verify scratch). ``--kv-fp8``: the
sparse-attention launches on an E4M3 MLA cache. ``--kind model`` is the served
step: layers 3 to 43 in model order (30 KDA, 11 sparse attention) on one runtime a
kind; ``--kind full`` the whole served step as the spec builds it: layer 0 from the
step's tokens, the dense layers 1 and 2, layers 3 to 44, then the tail. ``kda``,
``dsa``, ``dense``, ``embed`` (layer 0 from the tokens) and ``head`` (the tail) run
``--layers`` launches of one kind; each ``embed`` and ``head`` launch takes the step's
input, the others the previous launch's output. FP8 projections are read in place (AITER's (16, 16) layout), or row-major with
``--row-major``. After an eager step and 20 warm-up replays, each of ``--trials``
trials replays the graph ``--reps`` times; rank 0 prints per-launch microseconds
(trial time / (reps x launches)): the median, minimum and maximum over trials, the
slowest rank's median, and whether every error word stayed clean."""

import argparse
import gc
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_layer as cl  # noqa: E402

from sgl_amd_mono.glm53_flash.spec import GROUPED_FROM  # noqa: E402

_GEN = cl._gen


def _seed_layer(layer: int) -> None:
    """``check_layer``'s generators seeded for one layer: distinct weights a layer."""
    cl._gen = lambda seed, device: _GEN(seed + 1000 * layer, device)


def _kinds(args) -> list:
    if args.kind == "model":
        return ["dsa" if i % 4 == 3 else "kda" for i in range(3, 44)]
    if args.kind == "full":
        layers = ["dsa" if i % 4 == 3 else "kda" for i in range(3, 45)]
        return ["embed", "dense", "dense", *layers, "head"]
    return [args.kind] * args.layers


def _kda_state(rows: int, tp: int, device, verify: int = 1):
    """``check_layer``'s KDA state; with ``verify`` L > 1, rows // L requests (slots
    1 ..) of L rows each, pads on slot 0, and the verify scratch (a row a request)."""
    from msgspec.structs import replace

    from sgl_amd_mono.glm53_flash.gfx950.config import KDA_DIM, KDA_HEADS, KDA_PROJ

    if verify == 1:
        return cl.kda_state(rows, tp, device)
    R = rows // verify
    state, _ = cl.kda_state(R, tp, device)
    P, H = KDA_PROJ // tp, KDA_HEADS // tp
    state = replace(
        state,
        inter_rec=torch.zeros(
            R + 1, verify, H, KDA_DIM, KDA_DIM, dtype=torch.float32, device=device
        ),
        inter_conv=torch.zeros(
            R + 1, verify, 3, 3 * P, dtype=torch.bfloat16, device=device
        ),
    )
    idx = torch.zeros(rows, dtype=torch.int32, device=device)
    idx[: R * verify] = torch.arange(
        1, R + 1, dtype=torch.int32, device=device
    ).repeat_interleave(verify)
    return state, idx


def _fold(state, rows: int, tp: int, device, kept: int):
    """``state`` on the fold-in-verify contract: a ring of its own a layer, every
    request (ring row r for slot r + 1) with ``kept`` tokens to fold, the kept
    window verified at the stock width of ``rows`` // 4 requests."""
    from msgspec.structs import replace

    from sgl_amd_mono.glm53_flash.gfx950.config import KDA_HEADS, KDA_PROJ
    from sgl_amd_mono.glm53_flash.gfx950.runner import FlashKdaFold

    P, H, R = KDA_PROJ // tp, KDA_HEADS // tp, rows // 4
    g = cl._gen(13, device)
    slots = state.rec_state.shape[0]
    ring = [
        (torch.randn(R, 4, n, generator=g, device=device) * s_).bfloat16()
        for n, s_ in ((3 * P, 1.0), (P, 1.0), (H, 1.0))
    ]
    half = torch.full((R, H), int(R <= 2), dtype=torch.int32, device=device)
    fold_len = torch.full((slots + 1,), kept, dtype=torch.int32, device=device)
    rr = torch.arange(R, dtype=torch.int32, device=device)
    return replace(
        state,
        inter_rec=None,
        inter_conv=None,
        fold=FlashKdaFold.make(*ring, half, fold_len, rr),
    )


def _verify_caches(rows: int, L: int, pos: int, device, kv_fp8: bool = False):
    """``check_layer``'s verify inputs: rows // L requests of pos + 1 tokens, request
    r's row j at pos + j, SGLang's verify page table (the request's row for each of
    its rows), tail rings of 4 + L; rows past the requests are pads."""
    from msgspec.structs import replace

    from sgl_amd_mono.glm53_flash.gfx950.config import INDEX_DIM, KPOOL
    from sgl_amd_mono.glm53_flash.gfx950.layout import (
        ROW_BT,
        ROW_POOL_SLOT,
        ROW_POS,
        ROW_SLOT,
        ROW_TAIL_SLOT,
        ROW_WORDS,
    )

    R = rows // L
    caches, _, pt = cl.dsa_caches(R, device, [pos + 1] * R, kv_fp8)
    g = cl._gen(11, device)
    tails = [
        torch.randn(R + 1, 4 + L, INDEX_DIM, generator=g, device=device).bfloat16()
        for _ in range(2)
    ]
    caches = replace(caches, tail_k=tails[0], tail_g=tails[1])
    table = torch.full((rows, ROW_WORDS), -1, dtype=torch.int32, device=device)
    table[:, ROW_BT] = 0
    for r in range(R):
        for j in range(L):
            s = r * L + j
            slot = cl.kv_slot(pt, r, pos + j)
            table[s, [ROW_POS, ROW_SLOT, ROW_POOL_SLOT, ROW_TAIL_SLOT]] = torch.tensor(
                [pos + j, slot, slot // KPOOL, (r + 1) * KPOOL],
                dtype=torch.int32,
                device=device,
            )
            table[s, ROW_BT] = s
    return caches, table, pt.repeat_interleave(L, 0).contiguous()


def _build(args, kinds: list, rows: int, rank: int, tp: int, group, device):
    """The runtimes (one a kind), each layer's op and launch arguments."""
    from sgl_amd_mono.glm53_flash.gfx950.config import GLOBAL_INTER
    from sgl_amd_mono.glm53_flash.gfx950.kernel import FlashBuild
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashHeadOp,
        FlashLayerOp,
        FlashRuntime,
    )

    shuffled = not args.row_major
    runtimes = {}
    for kind in sorted(set(kinds)):
        dsa, head = kind == "dsa", kind == "head"
        dense = kind in ("dense", "embed")
        key = FlashBuild(
            rows=rows,
            tp=tp,
            launches_per_step=kinds.count(kind),
            inter=GLOBAL_INTER // tp,
            kda=not (dsa or head),
            dsa=dsa,
            max_pools=args.max_pools if dsa else 0,
            qkv_shuffled=shuffled and dsa,
            qb_shuffled=shuffled and dsa,
            o_shuffled=shuffled and dsa,
            dense=dense,
            dense_shuffled=shuffled and dense,
            tokens=kind == "embed",
            head=head,
            grouped=not args.per_row_moe
            and rows >= GROUPED_FROM
            and not (dense or head),
            verify=args.verify if kind in ("kda", "dsa") else 1,
            kv_fp8=args.kv_fp8 and dsa,
            fold=args.fold >= 0 and kind == "kda" and args.verify > 1,
            prof=args.prof,
        )
        runtimes[kind] = FlashRuntime(key, rank, group, device)
        runtimes[kind].set_deadline(60.0)
    ops, extra, ordinal = [], [], {kind: 0 for kind in runtimes}
    for layer, kind in enumerate(kinds):
        _seed_layer(layer)
        n = ordinal[kind]
        ordinal[kind] += 1
        if kind == "head":
            ops.append(
                FlashHeadOp(runtimes[kind], cl.head_weights(rank, tp, device), n)
            )
            extra.append({})
            continue
        if kind in ("dense", "embed"):
            ffn = cl.dense_weights(rank, tp, device, shuffled)
            attn = cl.kda_weights(rank, tp, device)
            state, idx = cl.kda_state(rows, tp, device)
            embedding = cl.embedding(rank, tp, device)[0] if kind == "embed" else None
            op = FlashLayerOp(runtimes[kind], ffn, n, kda=attn, embedding=embedding)
            ops.append(op)
            kw = dict(state=state, state_idx=idx)
            if embedding is not None:
                kw["step"] = cl.step_inputs(rows, rows, tp, device)
            extra.append(kw)
            continue
        ffn = cl.ffn_weights(rank, tp, device)
        if kind == "kda":
            attn = cl.kda_weights(rank, tp, device)
            state, idx = _kda_state(rows, tp, device, args.verify)
            if args.fold >= 0 and args.verify > 1:
                state = _fold(state, rows, tp, device, args.fold)
            ops.append(FlashLayerOp(runtimes[kind], ffn, n, kda=attn))
            extra.append(dict(state=state, state_idx=idx))
        else:
            attn = cl.dsa_weights(rank, tp, device)
            if shuffled:
                attn = cl.shuffled_copy(attn)
            if args.verify > 1:
                caches, table, pt = _verify_caches(
                    rows, args.verify, args.pos, device, args.kv_fp8
                )
            else:
                caches, table, pt = cl.dsa_caches(
                    rows, device, [args.pos + 1] * rows, kv_fp8=args.kv_fp8
                )
            ops.append(FlashLayerOp(runtimes[kind], ffn, n, dsa=attn))
            extra.append(dict(caches=caches, rows=table, page_table=pt))
    cl._gen = _GEN
    return runtimes, ops, extra


def _profile(graph, runtimes, rows: int, rank: int, samples: int = 9) -> None:
    """Each kind's last launch of a replay: per stage, the time (µs from the first
    CTA's start) by which every CTA has left it, median over ``samples`` replays."""
    from sgl_amd_mono.glm53_flash.gfx950.layout import BLOCKS

    got = {kind: [] for kind in runtimes}
    for _ in range(samples):
        graph.replay()
        torch.cuda.synchronize()
        for kind, rt in runtimes.items():
            names = rt.sc.get("_prof_marks", [])
            if not names:
                continue
            n = len(names)
            st = rt.scratch[rt.sc["prof"] :].view(torch.int64)[: BLOCKS * 48]
            st = st.view(BLOCKS, 48)[:, :n].double()
            t0 = st[:, 0].min()
            got[kind].append(((st.max(0).values - t0) / 100.0).tolist())
    if rank != 0:
        return
    for kind, runs in got.items():
        if not runs:
            continue
        names = runtimes[kind].sc["_prof_marks"]
        med = [statistics.median(r[i] for r in runs) for i in range(len(names))]
        parts = " ".join(f"{nm}={v:.1f}" for nm, v in zip(names, med))
        print(f"PROF kind={kind} rows={rows} done_us {parts}", flush=True)


def _time(args, rows: int, rank: int, tp: int, group, device) -> None:
    from sgl_amd_mono.common import guard

    kinds = _kinds(args)
    runtimes, ops, extra = _build(args, kinds, rows, rank, tp, group, device)
    x, res, post, comb = cl.layer_inputs(rows, device)
    # layer 0's input besides its tokens: every stream a copy of the embedding row
    entry = (
        torch.zeros_like(x),
        torch.zeros_like(res),
        torch.ones_like(post),
        torch.zeros_like(comb),
    )

    def step():
        h = (x, res, post, comb)
        for op, kw, kind in zip(ops, extra, kinds):
            if kind == "embed":
                h = op.launch(*entry, **kw)
            elif kind == "head":
                op.launch(*(h if args.kind == "full" else (x, res, post, comb)))
            else:
                h = op.launch(*h, **kw)
        for rt in runtimes.values():
            rt.advance_step()

    step()  # eager first: each launch loads its code outside the capture
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    per_launch = []
    for _ in range(args.trials):
        dist.barrier(group=group)
        begin.record()
        for _ in range(args.reps):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        per_launch.append(begin.elapsed_time(end) * 1e3 / (args.reps * len(kinds)))
    clean = not any(
        int(rt.epoch[i]) for rt in runtimes.values() for i in guard.ERROR_WORDS
    )
    median = statistics.median(per_launch)
    medians = [None] * tp
    dist.all_gather_object(medians, (median, clean), group=group)
    if rank == 0:
        print(
            f"TIME kind={args.kind} rows={rows} verify={args.verify} tp={tp} "
            f"launches={len(kinds)} "
            f"pos={args.pos} fp8={'row-major' if args.row_major else 'in-place'} "
            f"kv={'e4m3' if args.kv_fp8 else 'bf16'} fold={args.fold} "
            f"us_per_launch median={median:.2f} min={min(per_launch):.2f} "
            f"max={max(per_launch):.2f} "
            f"slowest_rank_median={max(m for m, _ in medians):.2f} "
            f"clean={all(c for _, c in medians)}",
            flush=True,
        )
    if args.prof:
        _profile(graph, runtimes, rows, rank)
    del graph
    dist.barrier(group=group)
    for rt in runtimes.values():
        rt.close()
    ops.clear()
    extra.clear()
    runtimes.clear()
    # free this width's code objects now: an unload during the next capture breaks it
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kind",
        choices=("kda", "dsa", "dense", "embed", "head", "model", "full"),
        default="model",
    )
    ap.add_argument("--rows", default="2,4,8", help="comma-separated widths")
    ap.add_argument(
        "--prof", action="store_true", help="stage stamps (a profiling build)"
    )
    ap.add_argument("--layers", type=int, default=12, help="launches of one kind")
    ap.add_argument("--reps", type=int, default=200, help="graph replays a trial")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--pos", type=int, default=2047, help="sparse-attention position")
    ap.add_argument("--max-pools", type=int, default=262144, help="1M-token context")
    ap.add_argument("--row-major", action="store_true", help="row-major FP8 weights")
    ap.add_argument(
        "--per-row-moe",
        action="store_true",
        help="the per-row MoE at every width, the baseline of the grouped one",
    )
    ap.add_argument(
        "--verify",
        type=int,
        default=1,
        help="kda, dsa or model: an MTP verify step's rows a request",
    )
    ap.add_argument("--kv-fp8", action="store_true", help="an E4M3 MLA cache")
    ap.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="kda with --verify: the fold-in-verify contract, this many kept tokens",
    )
    args = ap.parse_args()
    assert args.verify == 1 or args.kind in ("kda", "dsa", "model"), (
        "--verify times kda, dsa or model launches"
    )
    cl.MAX_LEN = max(cl.MAX_LEN, (args.pos + args.verify - 1 + 256) // 256 * 256)
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    for rows in map(int, args.rows.split(",")):
        _time(args, rows, rank, tp, group, device)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
