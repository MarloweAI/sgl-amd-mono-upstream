"""Exact top-k check of the GLM-5.3 indexer launch at chosen contexts, under torchrun
(TP 4 or 8):

    torchrun --nproc-per-node 8 test/kernels/glm53/check_topk.py --pos 8447,131071 \\
        [--rows 2,4,8] [--caches random,ties,zeros,recent] [--rowsets serving,mixed,chain]

Per index cache, position, row set and width, ``--layers`` launches of ``--kind``
(``index``; ``dense``: the dense layers' launch; ``embed``: layer 0 from the token
ids, then dense launches) run ``--steps`` steps eagerly, with the cached scales of each step's own keys poisoned
first (``check.poison_new_index_scales``): a launch must score the keys its rows write
from those rows, not from the cache. After each launch, for every live row:

- **scores** (``check.launch_scores``) against an f64 reference: the launch's index
  query (its ``index_q`` mailbox, rotated and transformed in f32 as the kernel does,
  then bf16), its head weights (``index_w``) and the row's keys in the index cache as
  the launch leaves it. ``score_err`` is the largest error over the row's largest
  reference score;
- **selection**: the row's list (``step.topk`` at its ``indptr``) against the exact
  top-k of the launch's own scores, k = min(2048, keys). ``missed`` counts the keys of
  the top-k (score descending, then position) it lacks; ``order`` holds when the list
  is, entry by entry, the radix select's: the keys above the k-th key's 12-, 24- or
  32-bit prefix in position order, then the prefix's keys above the k-th, then its
  ties in position order (the prefix is the first whose keys fit the build's
  candidate room).

Negative controls on copies of each row's result: a selected slot replaced by the
best unselected key (``missed``), two entries swapped (``order``) and one score's bits
flipped (``score_err``) must each fail. Index caches: ``random`` (check_layer's),
``ties`` (four keys repeated over every position, so the k-th score falls in a tie),
``zeros`` (every cached key 0) and ``recent`` (each live row's last 4,096 keys scaled
8x over check_layer's scale). Row sets: ``serving`` (time_layers.py's), ``mixed`` (one request a row, at
``--pos`` and below, some under 2,048 keys) and ``chain`` (one request's rows, as a
verify step of a draft chain as long as the width). ``--dump DIR``: every row's
scores and list as SHA-256 digests, in check_layer.py's format for its ``--compare``.
Prints a ``TOPK`` line per step and ``TOPK_RESULT exact=True`` when all pass."""

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_layer as cl  # noqa: E402
import time_layers as tl  # noqa: E402

from sgl_amd_mono.glm53 import check  # noqa: E402
from sgl_amd_mono.glm53.gfx950.layout import (  # noqa: E402
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_Q_ROWS,
)

SCORE_BOUND = 1e-4
CACHES = ("random", "ties", "zeros", "recent")


def _pairs(rt, name: str, n: int) -> torch.Tensor:
    """The value words of a mailbox's first n (value, tag) pairs."""
    at = rt.sc[name]
    return rt.scratch[at : at + n * 8].view(torch.int32)[0::2]


def _bf16_words(words: torch.Tensor) -> torch.Tensor:
    """Words of two bf16 values (the first in the low half) -> f32 [.., 2 * n]."""
    lo = (words << 16).view(torch.float32)
    hi = (words & -65536).view(torch.float32)
    return torch.stack([lo, hi], -1).flatten(-2)


def _hadamard(x: torch.Tensor) -> torch.Tensor:
    """The kernel's orthonormal 128-point Hadamard transform of f32 rows, in f32:
    butterflies of strides 1 to 64, then the scale."""
    x = x.clone()
    n = x.shape[-1]
    h = 1
    while h < n:
        y = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :].clone(), y[..., 1, :].clone()
        y[..., 0, :], y[..., 1, :] = a + b, a - b
        h *= 2
    return x * torch.tensor(128**-0.5, dtype=torch.float32)


def query(rt, model, s: int, W: int) -> torch.Tensor:
    """Row s's rotated, transformed index query as the launch scores it: f64 [32, 128]."""
    q = _bf16_words(_pairs(rt, "index_q", W * INDEX_Q_ROWS // 2))
    q = q.view(W, INDEX_HEADS, INDEX_DIM)[s].clone()
    pos = int(model.step_tables.positions[s])
    cos, sin = model.cos[pos].float(), model.sin[pos].float()
    q0, q1 = q[:, 0:64:2].clone(), q[:, 1:64:2].clone()
    q[:, 0:64:2] = q0 * cos - q1 * sin
    q[:, 1:64:2] = q0 * sin + q1 * cos
    return _hadamard(q).bfloat16().double()


def row_slots(st, s: int) -> torch.Tensor:
    """The cache slots of row s's keys 0 .. its position, in key order."""
    bound = int(st.positions[s]) + 1
    pages = st.page_table[int(st.req_ids[s]), : -(-bound // cl.PAGE)].long()
    keys = torch.arange(bound, device=pages.device)
    return pages[keys // cl.PAGE] * cl.PAGE + keys % cl.PAGE


def reference_scores(rt, model, layer: int, s: int, W: int, slots) -> torch.Tensor:
    entries = cl._index_entries(model.caches[layer].idx, slots)
    keys = entries[:, :INDEX_DIM].contiguous().view(torch.float8_e4m3fn).double()
    scales = entries[:, INDEX_DIM:].contiguous().view(torch.float32).double().flatten()
    w = _pairs(rt, "index_w", W * INDEX_HEADS).view(torch.float32)
    w = w.view(W, INDEX_HEADS)[s].double()
    return (torch.relu(keys @ query(rt, model, s, W).T) @ w) * scales


def radix_lists(bits: torch.Tensor, k: int) -> list:
    """The radix select's lists (key indices) of the top k of f32 score bits, at a
    12-, 24- and 32-bit prefix of the k-th key."""
    keys = check._radix_keys(bits)
    kth = torch.sort(keys, descending=True).values[k - 1]
    lists = []
    for bits_kept in (12, 24, 32):
        shift = 32 - bits_kept
        high = keys >> shift
        above = (high > (kth >> shift)).nonzero().flatten()
        under = ((high == (kth >> shift)) & (keys > kth)).nonzero().flatten()
        ties = (keys == kth).nonzero().flatten()[: k - len(above) - len(under)]
        lists.append(torch.cat([above, under, ties]))
    return lists


def check_row(bits, ref, picked, slots) -> dict:
    k = min(cl.TOPK, len(slots))
    keys = check._radix_keys(bits)
    exact = slots[torch.sort(keys, descending=True, stable=True).indices[:k]]
    out = {
        "missed": len(set(exact.tolist()) - set(picked.tolist())),
        "order": any(
            torch.equal(slots[lst], picked.long()) for lst in radix_lists(bits, k)
        ),
    }
    if ref is not None:
        got = bits.view(torch.float32).double()
        out["score_err"] = float(
            (got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)
        )
    return out


def controls(bits, ref, picked, slots) -> bool:
    """A wrong slot, two entries swapped and a flipped score bit each fail."""
    seen = True
    if len(slots) > cl.TOPK:
        order = torch.sort(check._radix_keys(bits), descending=True, stable=True)
        bad = picked.clone()
        bad[0] = slots[order.indices[cl.TOPK]]
        seen &= check_row(bits, None, bad, slots)["missed"] >= 1
    if len(picked) > 1:
        swapped = picked.clone()
        swapped[0], swapped[1] = picked[1], picked[0]
        seen &= not check_row(bits, None, swapped, slots)["order"]
    flipped = bits.clone()
    flipped[int(ref.abs().argmax())] ^= 1 << 22
    seen &= check_row(flipped, ref, picked, slots)["score_err"] > SCORE_BOUND
    return seen


def fill_index_cache(model, mode: str) -> None:
    """Index keys of ``mode`` (CACHES) in every layer's cache."""
    st = model.step_tables
    for c in model.caches:
        n = c.idx.shape[0]
        values = c.idx.view(n, -1)[:, : cl.PAGE * INDEX_DIM]
        if mode == "ties":
            # [page, 16-token group, 16-dim tile, token, dim]: page 0's first four
            # tokens repeated over every token
            tiles = values.view(n, 4, 8, 16, 16)
            tiles.copy_(tiles[:1, :1, :, :4, :].repeat(n, 4, 1, 4, 1))
        elif mode == "zeros":
            values.zero_()
        elif mode == "recent":
            scales = c.idx.view(n, -1)[:, cl.PAGE * INDEX_DIM :].view(torch.float32)
            for s in range(st.positions.shape[0]):
                if int(st.slots[s]) >= 0:
                    slots = row_slots(st, s)[-4096:]
                    scales[slots // cl.PAGE, slots % cl.PAGE] = 2.0**-3
        else:
            assert mode == "random", mode


def rows_of(rowset: str, width: int, pos: int):
    if rowset == "serving":
        return tl.serving_rows(width, pos)
    if rowset == "chain":
        return [pos + s for s in range(width)], [0] * width
    assert rowset == "mixed", rowset
    mixed = [pos, 1500, pos // 2, 300, pos - 7, 2047, 2048, 9000]
    return [min(p, pos) for p in mixed[:width]], list(range(width))


def run(args, model, name: str, digests: dict) -> list:
    st = model.step_tables
    W = st.positions.shape[0]
    live = st.slots[st.slots >= 0]
    records = []
    for step in range(args.steps):
        for c in model.caches:
            check.poison_new_index_scales(c.idx, live)
        results = []

        def after(i, _, step=step):
            rt = model.ops[i].rt
            bits = check.launch_scores(rt, W)
            indptr = st.indptr.tolist()
            d = {}
            for s in range(W):
                if int(st.slots[s]) < 0:
                    continue
                slots = row_slots(st, s)
                row_bits = bits[s, : len(slots)].clone()
                picked = st.topk[indptr[s] : indptr[s + 1]].clone()
                ref = reference_scores(rt, model, i, s, W, slots)
                res = check_row(row_bits, ref, picked, slots)
                res["controls"] = controls(row_bits, ref, picked, slots)
                results.append(res)
                d[f"scores.r{s}"] = cl._digest(row_bits)
                d[f"list.r{s}"] = cl._digest(picked)
            digests[f"{name}/s{step}/l{i}"] = d

        model.step(after=after)
        torch.cuda.synchronize()
        rec = {
            "name": name,
            "step": step,
            "rows_checked": len(results),
            "missed": max(r["missed"] for r in results),
            "order": all(r["order"] for r in results),
            "score_err": max(r["score_err"] for r in results),
            "controls": all(r["controls"] for r in results),
            "clean": model.errors() == 0,
        }
        rec["ok"] = (
            rec["missed"] == 0
            and rec["order"]
            and rec["score_err"] <= SCORE_BOUND
            and rec["controls"]
            and rec["clean"]
        )
        records.append(rec)
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="8447", help="comma-separated first positions")
    ap.add_argument("--rows", default="2,4,8", help="comma-separated widths")
    ap.add_argument("--caches", default=",".join(CACHES))
    ap.add_argument("--rowsets", default="serving,mixed,chain")
    ap.add_argument("--kind", choices=("index", "dense", "embed"), default="index")
    ap.add_argument("--layers", type=int, default=2, help="launches a step")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--kv", choices=("bf16", "fp8"), default="fp8")
    ap.add_argument("--deadline", type=float, default=60.0)
    ap.add_argument("--dump", default="", help="write the digests into DIR")
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    ok, digests = True, {}
    for mode in args.caches.split(","):
        for pos in map(int, args.pos.split(",")):
            for rowset in args.rowsets.split(","):
                for W in map(int, args.rows.split(",")):
                    positions, requests = rows_of(rowset, W, pos)
                    model = cl.Model(
                        cl.kinds_of(args.kind, args.layers),
                        positions,
                        requests,
                        rank,
                        tp,
                        group,
                        device,
                        args,
                    )
                    fill_index_cache(model, mode)
                    name = f"{args.kv}-{args.kind}-{mode}/p{pos}/{rowset}/w{W}"
                    for rec in run(args, model, name, digests):
                        ok &= rec["ok"]
                        if rank == 0:
                            print(
                                "TOPK " + " ".join(f"{k}={v}" for k, v in rec.items()),
                                flush=True,
                            )
                    model.close()
                    del model
                    torch.cuda.empty_cache()
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        name = f"digests_topk_{args.kind}_{args.kv}_r{rank}.json"
        path = os.path.join(args.dump, name)
        with open(path, "w") as f:
            json.dump(digests, f, indent=0, sort_keys=True)
    oks = [None] * tp
    dist.all_gather_object(oks, ok, group=group)
    if rank == 0:
        print(f"TOPK_RESULT tp={tp} exact={all(oks)}", flush=True)
    dist.barrier(group=group)
    dist.destroy_process_group()
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
