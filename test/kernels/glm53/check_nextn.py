"""Check the GLM-5.3 NextN (MTP draft) launches on random weights under torchrun (TP 4
or 8): ``torchrun --nproc-per-node 8 test/kernels/glm53/check_nextn.py --rows 2,4,8
--kv fp8``. A NextN launch starts from the step's token ids (e: their embedding rows,
pushed by the rank that holds each) and the h rows, through its entry
eh_proj(enorm(e) | hnorm(h)); the indexer kind runs the draft extend (4 rows a
request, widths 4 and 8), the shared kind a draft decode step (widths 2 and 4).
Each width runs with and without a pad row.

- ``entry``: the NextN launch against the plain launch of its kind fed the NextN
  launch's own x (its gathered entry rows): the output rows, the MLA cache rows, the
  index cache rows and the top-k bitwise equal; x against a torch entry (bf16 norms,
  f32 GEMM, bf16: ``draft.eh.rel``) and an f32 entry (``draft.eh_vs_fp32.rel``)
  within ``EH_REL`` and ``EH_F32_REL``; a pad row's x zero. Negative control: the
  enorm and hnorm weights swapped fail both bounds.
- ``tables``: ``tables.step_rows`` and ``tables.write_seed`` against the torch ops
  they stand for, word for word, on the GPU: SGLang's padding slot 0, pad rows,
  positions past the context, a reused top-k with -1 entries inside. Negative
  control: the reused top-k sized by position differs.
- ``replay``: a draft step (the NextN launch and the tail) in a graph, 20 replays
  bitwise equal to the eager step, no error word set.

Prints a ``CHECK`` line a check on rank 0 and ``CHECK_RESULT pass=..``; exits 1 on
any failure."""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_layer as C  # noqa: E402

from sgl_amd_mono.glm53.gfx950.config import HIDDEN  # noqa: E402

# the entry's bounds here: 3x the largest the TP8 runs gave (torch 2.9e-5, f32
# 2.3e-3, the torch entry's own distance to f32 alike); swapped norms give 0.14
EH_REL, EH_F32_REL = 1e-4, 7e-3
VOCAB = C.VOCAB
# the rows a width runs: draft extend rows (requests of 4) or one request a row
ROWS = {"index": {4: ("v4", "v4p"), 8: ("v4", "v4p")}, "shared": {}}
for _w in (2, 4):
    ROWS["shared"][_w] = ("p0", "p1")


def rows_of(width: int, rowset: str):
    """``check_layer.rows_of``, and ``v4p``: verify rows with the last a pad row."""
    if rowset != "v4p":
        return C.rows_of(width, rowset)
    positions, requests = C.rows_of(width, "v4")
    return positions[:-1] + [None], requests[:-1] + [0]


def eh_weights(rank: int, tp: int, device):
    """The entry's random tensors (every rank draws the same): this rank's rows of
    eh_proj, the enorm and hnorm weights, and this rank's embedding rows."""
    from sgl_amd_mono.glm53.gfx950.runner import Glm53EhWeights

    g = C._gen(13, device)
    full = torch.randn(HIDDEN, 2 * HIDDEN, generator=g, device=device)
    full = (full * (2 * HIDDEN) ** -0.5).bfloat16()
    emb = torch.randn(VOCAB, HIDDEN, generator=g, device=device).bfloat16()
    rows, vrows = HIDDEN // tp, VOCAB // tp
    return Glm53EhWeights(
        w_eh=full[rank * rows : (rank + 1) * rows].contiguous(),
        g_e=C._norm(g, HIDDEN, device),
        g_h=C._norm(g, HIDDEN, device),
        emb=emb[rank * vrows : (rank + 1) * vrows].contiguous(),
    ), (full, emb)


def _rms(x, g, eps=1e-5):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g.float()


def _rel(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


class Draft:
    """A NextN layer of one kind at one width: its NextN runtime, the plain runtime
    of its kind on the same layer weights and caches, and the tail. ``rows``:
    (positions, requests) in place of ``rowset``'s."""

    def __init__(self, kind, W, rowset, rank, tp, group, device, args, rows=None):
        from sgl_amd_mono.glm53.gfx950.runner import (
            Glm53Build,
            Glm53HeadOp,
            Glm53LayerOp,
            Glm53Runtime,
        )

        index = kind == "index"
        positions, requests = rows or rows_of(W, rowset)
        live = [p for p in positions if p is not None]
        self.live = len(live)
        pages = C.Pages(max(requests) + 1, max(live) + W + 1, device)
        self.caches = C.layer_caches(78, pages, index, device, args.kv == "fp8")
        self.cos, self.sin = C.rope(max(live) + W + 1, device)
        self.st, self.topk0 = C.make_step(positions, requests, pages, device)
        self.eh, (self.w_eh, self.emb) = eh_weights(rank, tp, device)
        w = C.layer_weights(78, rank, tp, device, kind)

        def build(**kw):
            key = Glm53Build(
                rows=W,
                tp=tp,
                launches_per_step=1,
                with_indexer=index,
                index_max_seq=C.MAX_SEQ if index else 0,
                fp8_shuffled=True,
                kv_fp8=args.kv == "fp8",
                **kw,
            )
            rt = Glm53Runtime(key, rank, group, device)
            rt.set_deadline(args.deadline)
            return rt

        self.nextn = Glm53LayerOp(
            build(eh=True, tokens=True, vocab_rows=VOCAB // tp), w, 0, self.eh
        )
        self.plain = Glm53LayerOp(build(), w, 0)
        head = C.head_weights(rank, tp, device)
        self.head = Glm53HeadOp(build(head=True, vocab_rows=head.w_head.shape[0]), head)
        g = C._gen(19, device)
        ids = torch.randint(0, VOCAB, (W,), generator=g, device=device)
        self.ids = torch.where(
            torch.tensor([p is not None for p in positions], device=device), ids, -1
        ).to(torch.int32)
        self.h = torch.randn(W, HIDDEN, generator=g, device=device).bfloat16()
        self.runtimes = [self.nextn.rt, self.plain.rt, self.head.rt]

    def advance(self):
        for rt in self.runtimes:
            rt.advance_step()

    def step(self):
        """One NextN launch from the step's rows, then the next epoch."""
        self.run_nextn()
        self.advance()

    def run_nextn(self):
        self.st.topk.copy_(self.topk0)
        x = self.nextn.launch(
            self.ids, self.caches, self.cos, self.sin, self.st, h_prev=self.h
        )
        return x

    def run_plain(self, x):
        self.st.topk.copy_(self.topk0)
        return self.plain.launch(x, self.caches, self.cos, self.sin, self.st)

    def outputs(self, out):
        """The launch's live output rows, the MLA and index cache rows it wrote, and
        the top-k."""
        slots = self.st.slots[self.st.slots >= 0]
        d = [
            out[: self.live].clone(),
            self.caches.kv[slots].clone(),
            self.st.topk.clone(),
        ]
        if self.caches.idx is not None:
            d.append(C._index_entries(self.caches.idx, slots).clone())
        return d

    def saved(self):
        c = self.caches
        return c.kv.clone(), None if c.idx is None else c.idx.clone()

    def restore(self, saved):
        kv, idx = saved
        self.caches.kv.copy_(kv)
        if idx is not None:
            self.caches.idx.copy_(idx)

    def errors(self) -> int:
        from sgl_amd_mono.common import guard

        return sum(
            int(rt.epoch[i]) != 0 for rt in self.runtimes for i in guard.ERROR_WORDS
        )

    def close(self):
        for rt in self.runtimes:
            rt.close()


def entry_refs(d: Draft, rank: int, tp: int):
    """(torch entry, f32 entry) of the live rows, every column."""
    live = d.ids[: d.live].long()
    e = d.emb[live]
    h = d.h[: d.live]
    cat = torch.cat((_rms(e, d.eh.g_e).bfloat16(), _rms(h, d.eh.g_h).bfloat16()), -1)
    ref = (cat.float() @ d.w_eh.float().t()).bfloat16()
    f32 = torch.cat((_rms(e, d.eh.g_e), _rms(h, d.eh.g_h)), -1) @ d.w_eh.float().t()
    return ref, f32


def check_entry(d: Draft, name, rank, tp, report):
    from sgl_amd_mono.glm53 import check
    from sgl_amd_mono.glm53.gfx950.layout import EH_ARGS

    saved = d.saved()
    out = d.run_nextn()
    d.advance()
    torch.cuda.synchronize()
    x = check.entry_rows(d.nextn.rt, d.st.slots.shape[0]).clone()
    mine = d.outputs(out)
    d.restore(saved)
    out2 = d.run_plain(x)
    d.advance()
    torch.cuda.synchronize()
    theirs = d.outputs(out2)
    d.restore(saved)
    same = [
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(mine, theirs)
    ]
    ref, f32 = entry_refs(d, rank, tp)
    v = {
        "out": same[0],
        "kv": same[1],
        "topk": same[2],
        "index": same[3] if len(same) > 3 else True,
        "draft.eh.rel": _rel(x[: d.live], ref),
        "draft.eh_vs_fp32.rel": _rel(x[: d.live], f32),
        "torch_vs_fp32.rel": _rel(ref, f32),
        "pads_zero": bool((x[d.live :] == 0).all()),
    }
    ok = (
        all(same)
        and v["pads_zero"]
        and v["draft.eh.rel"] <= EH_REL
        and v["draft.eh_vs_fp32.rel"] <= EH_F32_REL
    )
    report(f"entry {name}", v, ok)
    # negative control: the enorm and hnorm weights swapped
    table = d.nextn.index
    a, b = EH_ARGS[1], EH_ARGS[2]
    table[[a, b]] = table[[b, a]].clone()
    d.run_nextn()
    d.advance()
    torch.cuda.synchronize()
    bad = check.entry_rows(d.nextn.rt, d.st.slots.shape[0])[: d.live].clone()
    table[[a, b]] = table[[b, a]].clone()
    d.restore(saved)
    v = {"draft.eh.rel": _rel(bad, ref), "draft.eh_vs_fp32.rel": _rel(bad, f32)}
    report(
        f"entry-control {name}",
        v,
        v["draft.eh.rel"] > EH_REL and v["draft.eh_vs_fp32.rel"] > EH_F32_REL,
    )


def check_replay(d: Draft, name, replays, report):
    saved = d.saved()

    def step():
        logits, _ = d.head.launch(d.run_nextn())
        d.advance()
        return logits

    want = step().clone()
    torch.cuda.synchronize()
    d.restore(saved)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    same = 0
    for _ in range(replays):
        d.restore(saved)
        graph.replay()
        torch.cuda.synchronize()
        same += torch.equal(out.view(torch.uint8), want.view(torch.uint8))
    del graph
    d.restore(saved)
    errors = d.errors()
    report(
        f"replay {name}",
        {"equal": same, "errors": errors},
        same == replays and errors == 0,
    )


def _torch_rows(st, loc, pos, S, max_position, ids_in, ids, seed=None):
    """The torch ops the step tables stand for: the target's, and SGLang's
    compaction of a reused top-k's valid entries."""
    W = st.slots.shape[0]
    ids[:S] = ids_in[:S]
    ids[S:] = -1
    loc = loc[:S].to(torch.int64)
    valid = loc != 0
    st.slots[:S] = torch.where(valid, loc, -1)
    st.slots[S:] = -1
    p = pos[:S].to(torch.int64).clamp(max=max_position)
    st.positions[:S] = p
    st.positions[S:] = 0
    if seed is None:
        count = torch.where(valid, (p + 1).clamp(max=C.TOPK), 0)
    else:
        keep = (seed[:S] >= 0) & valid[:, None]
        count = keep.sum(-1)
    st.indptr[0] = 0
    st.indptr[1 : S + 1] = torch.cumsum(count, 0, dtype=torch.int32)
    st.indptr[S + 1 :] = st.indptr[S]
    st.req_ids[:S] = torch.arange(S, dtype=torch.int32, device=loc.device)
    st.req_ids[S:] = 0
    if seed is not None:
        start = st.indptr[:S].to(torch.int64)[:, None]
        dst = torch.where(keep, start + keep.cumsum(-1) - 1, W * C.TOPK)
        st.topk.scatter_(0, dst.reshape(-1), seed[:S].reshape(-1))


def check_tables(widths, device, report):
    from sgl_amd_mono.glm53 import check, tables
    from sgl_amd_mono.glm53.draft import _new_step

    g = C._gen(11, device)
    for W in widths:
        for S in sorted({1, W - 1, W} - {0}):
            # a strided view, as SGLang's per-step draft slots
            loc = torch.randint(1, 1 << 30, (3 * S,), generator=g, device=device)[::3]
            if S > 1:
                loc[S - 1] = 0  # SGLang's padding slot
            pos = torch.randint(0, 12000, (S,), generator=g, device=device)
            pos[0] = 100
            ids_in = torch.randint(0, VOCAB, (S,), generator=g, device=device)
            seed = torch.randint(
                0, 1 << 30, (S, C.TOPK), generator=g, device=device, dtype=torch.int32
            )
            for s in range(S):
                n = int(torch.randint(1, C.TOPK + 1, (1,), generator=g, device=device))
                seed[s, n:] = -1
            seed[0, 3] = -1
            for with_seed in (False, True):
                a, b = _new_step(W, device), _new_step(W, device)
                ida = torch.zeros(W, dtype=torch.int32, device=device)
                idb = torch.zeros(W, dtype=torch.int32, device=device)
                sd = seed if with_seed else None
                _torch_rows(a, loc, pos, S, 9000, ids_in, ida, sd)
                tables.step_rows(b, loc, pos, S, 9000, ids_in=ids_in, ids=idb, seed=sd)
                torch.cuda.synchronize()
                n = int(a.indptr[W])
                same = {
                    name: torch.equal(getattr(a, name), getattr(b, name))
                    for name in ("slots", "positions", "indptr", "req_ids")
                }
                same["ids"] = torch.equal(ida, idb)
                same["topk"] = torch.equal(a.topk[:n], b.topk[:n])
                report(
                    f"tables w{W} rows{S} seed{int(with_seed)}",
                    same,
                    all(same.values()),
                )
            # negative control: the reused top-k sized by position
            c = _new_step(W, device)
            tables.step_rows(c, loc, pos, S, 9000, seed=seed, seed_by_position=True)
            torch.cuda.synchronize()
            differs = not torch.equal(a.indptr, c.indptr)
            report(f"tables-control w{W} rows{S}", {"differs": differs}, differs)
            # the seed SGLang gets: each row's keys, then -1
            out = torch.full((S, C.TOPK), 7, dtype=torch.int32, device=device)
            tables.write_seed(
                a, out, S, torch.arange(S, dtype=torch.int32, device=device)
            )
            torch.cuda.synchronize()
            bad = check.seed_mismatch(a, out, S)
            report(f"seed w{W} rows{S}", {"draft.seed.max_abs": bad}, bad == 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="2,4,8", help="comma-separated widths")
    ap.add_argument("--kv", choices=("bf16", "fp8"), default="fp8")
    ap.add_argument("--only", default="tables,entry,replay")
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--deadline", type=float, default=10.0)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    widths = [int(w) for w in args.rows.split(",")]
    only = args.only.split(",")
    results = []

    def report(name, values, ok):
        if rank == 0:
            fields = " ".join(
                f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in values.items()
            )
            print(f"CHECK {args.kv} {name} {fields} ok={ok}", flush=True)
        results.append(bool(ok))

    if "tables" in only:
        check_tables(widths, device, report)
    for kind in ("index", "shared"):
        for W in [w for w in widths if w in ROWS[kind]]:
            for rowset in ROWS[kind][W]:
                d = Draft(kind, W, rowset, rank, tp, group, device, args)
                name = f"nextn_{kind} w{W} {rowset}"
                if "entry" in only:
                    check_entry(d, name, rank, tp, report)
                if "replay" in only:
                    check_replay(d, name, args.replays, report)
                torch.cuda.synchronize()
                if d.errors():
                    report(f"errors {name}", {"error_words": d.errors()}, False)
                d.close()
                del d
                torch.cuda.empty_cache()
    every = [None] * tp
    dist.all_gather_object(every, all(results), group=group)
    passed = all(every)
    if rank == 0:
        print(f"CHECK_RESULT kv={args.kv} tp={tp} pass={passed} checks={len(results)}")
    dist.barrier(group=group)
    dist.destroy_process_group()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
