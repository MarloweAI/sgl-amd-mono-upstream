"""Kernel-only time of the GLM-5.3 layer launch, under torchrun (TP 4 or 8):

    torchrun --nproc-per-node 8 test/kernels/glm53/time_layers.py \\
        --kind model --rows 2,4,8 [--pos 8447] [--kv fp8] [--layers 12]

One HIP graph holds a step of launches, inputs chained layer to layer. Each layer has
its own random weights and caches (``check_layer.py``'s, seeded by layer, so they
stream from HBM as in a served step). ``--kind full`` is the served step: layer 0
from the step's tokens, the dense layers 1 and 2, layers 3 to 77 in model order and
the tail (79 launches); ``model`` its MoE layers 3 to 77 (18 indexer and 57 shared
launches); ``embed`` layer 0 alone and ``head`` the tail alone; ``shared``, ``index``
and ``dense`` run ``--layers`` launches of one kind; ``nextn_index`` and ``nextn_shared``
one NextN launch a step (``check_nextn.py``'s draft layer; the spec serves the first at
4 and 8 rows, the second at 2, 4 and 8). The rows are those of the served decode
points at ``--pos``: one request and a pad row at 2 rows (one request without MTP),
one request's 4 verify rows at 4 (MTP, one request), two requests' at 8 (MTP, two
requests). After an eager step and 20 warm-up replays, each of ``--trials`` trials
replays the graph ``--reps`` times; rank 0 prints per-launch microseconds (trial
time / (reps x launches)): the median, minimum and maximum over trials, the slowest
rank's median, and whether every error word stayed clean."""

import argparse
import gc
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_layer as cl  # noqa: E402
import check_nextn as cn  # noqa: E402


def serving_rows(width: int, pos: int):
    """The served decode point's rows at ``width`` (positions, requests)."""
    if width == 2:
        return [pos, None], [0, 0]
    return [pos + s % 4 for s in range(width)], [s // 4 for s in range(width)]


LAUNCHES = {"model": 75, "full": 78, "embed": 1, "head": 0}


def _model(args, rows: int, rank: int, tp: int, group, device):
    """The launches a step of ``rows`` rows runs, and how many."""
    positions, requests = serving_rows(rows, args.pos)
    if args.kind.startswith("nextn_"):
        kind, at = args.kind.removeprefix("nextn_"), (positions, requests)
        return cn.Draft(kind, rows, None, rank, tp, group, device, args, rows=at), 1
    kinds = cl.kinds_of(args.kind, LAUNCHES.get(args.kind, args.layers))
    # the spec's builds: a failed wait stops the launch
    head = args.kind in ("full", "head")
    model = cl.Model(
        kinds,
        positions,
        requests,
        rank,
        tp,
        group,
        device,
        args,
        stop=True,
        head=head,
        similar=args.similar,
    )
    return model, len(kinds) + head


def _time(args, rows: int, rank: int, tp: int, group, device) -> None:
    model, launches = _model(args, rows, rank, tp, group, device)
    model.step()  # eager first: each launch loads its code outside the capture
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        model.step()
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
        per_launch.append(begin.elapsed_time(end) * 1e3 / (args.reps * launches))
    clean = model.errors() == 0
    median = statistics.median(per_launch)
    medians = [None] * tp
    dist.all_gather_object(medians, (median, clean), group=group)
    if rank == 0:
        print(
            f"TIME kind={args.kind} rows={rows} tp={tp} launches={launches} "
            f"pos={args.pos} kv={args.kv} us_per_launch median={median:.2f} "
            f"min={min(per_launch):.2f} max={max(per_launch):.2f} "
            f"slowest_rank_median={max(m for m, _ in medians):.2f} "
            f"clean={all(c for _, c in medians)}",
            flush=True,
        )
    del graph
    dist.barrier(group=group)
    model.close()
    del model
    # free this width's code objects now: an unload during the next capture breaks it
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kind",
        choices=(
            *("model", "full", "shared", "index", "dense", "embed", "head"),
            *("nextn_index", "nextn_shared"),
        ),
        default="model",
    )
    ap.add_argument("--rows", default="2,4,8", help="comma-separated widths")
    ap.add_argument("--layers", type=int, default=12, help="launches of one kind")
    ap.add_argument("--pos", type=int, default=8447, help="the first row's position")
    ap.add_argument("--kv", choices=("bf16", "fp8"), default="fp8")
    ap.add_argument("--reps", type=int, default=50, help="graph replays a trial")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--deadline", type=float, default=60.0)
    ap.add_argument("--similar", type=float, default=0.0, help="rows routing alike")
    args = ap.parse_args()
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
