"""Run the GLM-5.3 layer launch on random weights in SGLang's layouts under torchrun
(TP 4 or 8): ``torchrun --nproc-per-node 8 test/kernels/glm53/check_layer.py --kind
index --rows 2,4,8``. Kinds: ``shared`` (a layer that reuses the previous top-k),
``index`` (a layer that runs the indexer) and ``model`` (the model order's first
``--launches`` layers: 3 shared, then an indexer layer every fourth). Each width runs
``--steps`` steps of ``--launches`` launches, inputs chained, on the rows of each of
``--rowsets`` (one request a row, the same with a pad row, or MTP's verify rows; the
live rows at positions below and past the 2048-key top-k), with the MLA cache in
bf16 or E4M3 (``--kv``). Every rank must finish with no error word set and the same
output (the TP sums run in rank order). ``--writes``: each launch changes exactly
its live rows' MLA and index cache rows, and a shared launch leaves the top-k alone;
``--replays N``: a graph of the step replays the eager step bitwise;
``--check-tags``: every selected token of a scored row is a pair carrying its
indexer launch's tag and the slot the launch publishes, and a cleared tag fails.

``--dump DIR`` writes, per run and rank, the SHA-256 of every launch's output rows,
of the cache rows and top-k entries it writes, and after each width of every cache
whole (``--perturb`` negates one input value, a negative control); ``--compare A B``
matches two dump directories and exits 1 on any difference."""

import argparse
import hashlib
import json
import os
import sys

import torch
import torch.distributed as dist

from sgl_amd_mono.glm53.gfx950.config import (
    GLM5_GLOBAL_HEADS,
    GLM5_GLOBAL_INTER,
    HIDDEN,
    KV_LORA,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    V_DIM,
)
from sgl_amd_mono.glm53.gfx950.layout import INDEX_DIM, INDEX_HEADS

N_EXPERTS = 256
QKV_A = Q_LORA + KV_LORA + PE_DIM
# the served context (GLM-5.3's 1M positions): the keys an indexer row scores at most
MAX_SEQ = 1 << 20
# the live rows' positions, in order: past and below the 2048-key top-k
POSITIONS = [8447, 1500, 5000, 300, 8447, 2047, 2048, 9000]
TOPK = 2048
PAGE = 64


def _gen(seed: int, device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _fp8(g, shape, device) -> torch.Tensor:
    return torch.randn(shape, generator=g, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )


def _scales(g, shape, k: int, device) -> torch.Tensor:
    """Block scales near 1 / sqrt(k): a GEMV of unit-RMS inputs gives unit outputs."""
    return (0.5 + torch.rand(shape, generator=g, device=device)) * k**-0.5


def _norm(g, n: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    return (1 + 0.1 * torch.randn(n, generator=g, device=device)).to(dtype)


def model_kinds(n: int) -> list:
    """The kinds of the first n covered layers (from layer 3): SGLang runs the
    indexer on layer i unless max(i - 2, 0) % 4."""
    return ["index" if (i - 2) % 4 == 0 else "shared" for i in range(3, 3 + n)]


def layer_weights(layer: int, rank: int, tp: int, device, index: bool):
    """Layer ``layer``'s random tensors as this rank's shard (every rank draws the
    same): block-FP8 projections and experts read in place, W_UK / W_UV, the router
    and the indexer's weights_proj packed; the shared expert is slot 256."""
    from sgl_amd_mono.glm53.gfx950.packing import pack_bf16, pack_fp8
    from sgl_amd_mono.glm53.gfx950.runner import Glm53Weights

    g = _gen(1000 * layer + 7, device)
    heads, inter = GLM5_GLOBAL_HEADS // tp, GLM5_GLOBAL_INTER // tp
    E = N_EXPERTS + 1
    qb_rows = heads * (NOPE_DIM + PE_DIM)
    w_qkv = _fp8(g, (QKV_A, HIDDEN), device)
    s_qkv = _scales(g, (-(-QKV_A // 128), HIDDEN // 128), HIDDEN, device)
    w_qb = _fp8(g, (qb_rows, Q_LORA), device)
    s_qb = _scales(g, (qb_rows // 128, Q_LORA // 128), Q_LORA, device)
    w_o = _fp8(g, (HIDDEN, heads * V_DIM), device)
    s_o = _scales(g, (HIDDEN // 128, heads * V_DIM // 128), heads * V_DIM, device)
    uk = _fp8(g, (heads * KV_LORA, NOPE_DIM), device)
    uv = _fp8(g, (heads * V_DIM, KV_LORA), device)
    router = (
        torch.randn(N_EXPERTS, HIDDEN, generator=g, device=device) * 2 / HIDDEN**0.5
    )
    bias = 0.1 * torch.randn(N_EXPERTS, generator=g, device=device)
    w_ug = _fp8(g, (E, 2 * inter, HIDDEN), device)
    s_ug = _scales(g, (E, 2 * inter // 128, HIDDEN // 128), HIDDEN, device)
    w_dn = _fp8(g, (E, HIDDEN, inter), device)
    s_dn = _scales(g, (E, HIDDEN // 128, inter // 128), GLM5_GLOBAL_INTER, device)
    table, keep = None, ()
    if index:
        w_k = _fp8(g, (INDEX_DIM, HIDDEN), device)
        s_k = _scales(g, (1, HIDDEN // 128), HIDDEN, device)
        w_w = torch.randn(INDEX_HEADS, HIDDEN, generator=g, device=device) / HIDDEN**0.5
        w_q = _fp8(g, (INDEX_HEADS * INDEX_DIM, Q_LORA), device)
        s_q = _scales(
            g, (INDEX_HEADS * INDEX_DIM // 128, Q_LORA // 128), Q_LORA, device
        )
        g_k = _norm(g, INDEX_DIM, device, torch.float32)
        b_k = 0.1 * torch.randn(INDEX_DIM, generator=g, device=device)
        keep = (w_k, s_k, pack_bf16(w_w.bfloat16()), w_q, s_q, g_k, b_k)
        ptrs = [t.data_ptr() for t in keep] + [0]
        table = torch.tensor(ptrs, dtype=torch.int64, device=device)
    return Glm53Weights(
        g_in=_norm(g, HIDDEN, device),
        g_q=_norm(g, Q_LORA, device),
        g_kv=_norm(g, KV_LORA, device),
        g_post=_norm(g, HIDDEN, device),
        w_qkv_a=w_qkv,
        s_qkv_a=s_qkv,
        w_q_b=w_qb,
        s_q_b=s_qb,
        w_uk=pack_fp8(uk.contiguous()),
        s_uk=torch.full(
            (heads * KV_LORA // 128, NOPE_DIM // 64), NOPE_DIM**-0.5, device=device
        ),
        w_uv=pack_fp8(uv.contiguous()),
        s_uv=torch.full(
            (heads * V_DIM // 128, KV_LORA // 128), KV_LORA**-0.5, device=device
        ),
        w_o=w_o,
        s_o=s_o,
        w_r=pack_bf16(router.bfloat16().contiguous()),
        bias=bias,
        w_ug=w_ug,
        s_ug=s_ug,
        w_dn=w_dn,
        s_dn=s_dn,
        shuffled=True,
        index=table,
        keep=keep,
    )


class Pages:
    """The page table every layer's caches share: a request's pages are consecutive,
    and page 0 (slot 0, SGLang's pad slot) is never used."""

    def __init__(self, requests: int, max_pos: int, device):
        self.per_req = max_pos // PAGE + 2
        self.n_pages = 1 + requests * self.per_req
        n = requests * self.per_req
        self.table = (1 + torch.arange(n, dtype=torch.int32, device=device)).view(
            requests, self.per_req
        )

    def slot(self, req: int, pos: int) -> int:
        return int(self.table[req, pos // PAGE]) * PAGE + pos % PAGE


def layer_caches(layer: int, pages: Pages, index: bool, device, kv_fp8: bool):
    """SGLang's caches, random: the MLA cache [slots, 576] (bf16, or E4M3 without
    scales) and, on an indexer layer, the uint8 index cache [pages, 64, 132]
    (a page's 64 x 128 E4M3 keys, then their 64 f32 scales)."""
    from sgl_amd_mono.glm53.gfx950.runner import KV_ROW, Glm53Caches

    g = _gen(1000 * layer + 11, device)
    n = pages.n_pages
    kv = torch.randn(n * PAGE, KV_ROW, generator=g, device=device, dtype=torch.bfloat16)
    if kv_fp8:
        kv = kv.to(torch.float8_e4m3fn)
    idx = None
    if index:
        keys = PAGE * INDEX_DIM
        idx = torch.empty(n, PAGE * (INDEX_DIM + 4), dtype=torch.uint8, device=device)
        idx[:, :keys] = (
            torch.randn(n, keys, generator=g, device=device, dtype=torch.bfloat16)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        idx[:, keys:].view(torch.float32).fill_(2.0**-6)
        idx = idx.view(n, PAGE, INDEX_DIM + 4)
    return Glm53Caches(kv=kv, idx=idx)


def make_step(positions: list, requests: list, pages: Pages, device):
    """The step tables of rows at ``positions`` (None: a pad row) in ``requests``,
    and the top-k the shared layers before the first indexer layer read: each live
    row's 2048 KV slots (all its keys when it has fewer), spread over its keys."""
    from sgl_amd_mono.glm53.gfx950.runner import Glm53Step

    g = _gen(5, device)
    W = len(positions)
    counts = [0 if p is None else min(p + 1, TOPK) for p in positions]
    indptr = [0]
    for c in counts:
        indptr.append(indptr[-1] + c)
    topk = torch.zeros(W * TOPK + 1, dtype=torch.int32, device=device)
    for s, p in enumerate(positions):
        if p is None:
            continue
        keys = torch.arange(p + 1, device=device)
        if p + 1 > TOPK:
            pick = torch.randperm(p + 1, generator=g, device=device)[:TOPK]
            keys = keys[pick].sort().values
        page = pages.table[requests[s]].to(torch.int64)[keys // PAGE]
        topk[indptr[s] : indptr[s + 1]] = (page * PAGE + keys % PAGE).to(torch.int32)
    slots = [-1 if p is None else pages.slot(r, p) for p, r in zip(positions, requests)]
    i64, i32 = torch.int64, torch.int32
    step = Glm53Step(
        positions=torch.tensor([p or 0 for p in positions], dtype=i64, device=device),
        slots=torch.tensor(slots, dtype=i64, device=device),
        indptr=torch.tensor(indptr, dtype=i32, device=device),
        req_ids=torch.tensor(requests, dtype=i32, device=device),
        # SGLang's DSA backend gives each row its request's page-table row
        page_table=pages.table[torch.tensor(requests, device=device)].contiguous(),
        topk=topk,
    )
    return step, topk.clone()


def rope(max_pos: int, device):
    """bf16 cos and sin [positions, 32] of a rotary (the base is immaterial)."""
    exps = torch.arange(0, PE_DIM, 2, device=device, dtype=torch.float32) / PE_DIM
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    ang = pos[:, None] / (1e6**exps)[None]
    return ang.cos().bfloat16().contiguous(), ang.sin().bfloat16().contiguous()


class Model:
    """Random layers of ``kinds`` (layer ids from 3), their caches and one runtime
    a kind at one width; ``step()`` runs a step of launches, inputs chained.
    ``stop=False``: a failed wait sets its error word and the launch runs on."""

    def __init__(
        self, kinds, positions, requests, rank, tp, group, device, args, stop=False
    ):
        from sgl_amd_mono.glm53.gfx950.runner import (
            Glm53Build,
            Glm53LayerOp,
            Glm53Runtime,
        )

        self.kinds, W = kinds, len(positions)
        live = [p for p in positions if p is not None]
        pages = Pages(max(requests) + 1, max(live) + W + 1, device)
        self.caches = [
            layer_caches(3 + i, pages, k == "index", device, args.kv == "fp8")
            for i, k in enumerate(kinds)
        ]
        self.cos, self.sin = rope(max(live) + W + 1, device)
        self.step_tables, self.topk0 = make_step(positions, requests, pages, device)
        self.runtimes = {}
        for kind in sorted(set(kinds)):
            key = Glm53Build(
                rows=W,
                tp=tp,
                launches_per_step=kinds.count(kind),
                with_indexer=kind == "index",
                index_max_seq=MAX_SEQ if kind == "index" else 0,
                fp8_shuffled=True,
                kv_fp8=args.kv == "fp8",
                stop=stop,
            )
            self.runtimes[kind] = Glm53Runtime(key, rank, group, device)
            self.runtimes[kind].set_deadline(args.deadline)
        self.ops, ordinal = [], {k: 0 for k in self.runtimes}
        for i, kind in enumerate(kinds):
            w = layer_weights(3 + i, rank, tp, device, kind == "index")
            self.ops.append(Glm53LayerOp(self.runtimes[kind], w, ordinal[kind]))
            ordinal[kind] += 1
        g = _gen(3, device)
        self.h = torch.randn(W, HIDDEN, generator=g, device=device).bfloat16()

    def step(self, after=None) -> torch.Tensor:
        """One step (``after(i, out)`` after launch i); returns the last output."""
        st = self.step_tables
        st.topk.copy_(self.topk0)
        h = self.h
        for i, op in enumerate(self.ops):
            h = op.launch(h, self.caches[i], self.cos, self.sin, st)
            if after is not None:
                after(i, h)
        for rt in self.runtimes.values():
            rt.advance_step()
        return h

    def errors(self) -> int:
        from sgl_amd_mono.common import guard

        return sum(
            int(rt.epoch[i]) != 0
            for rt in self.runtimes.values()
            for i in guard.ERROR_WORDS
        )

    def close(self) -> None:
        for rt in self.runtimes.values():
            rt.close()


def rows_of(width: int, rowset: str):
    """A width's rows (positions, requests): ``p0`` one request a row at
    POSITIONS; ``p1`` the same with the last row a pad row; ``v4`` verify rows,
    4 a request (as MTP's), request r's rows from POSITIONS[r] on."""
    if rowset == "v4":
        reqs = [s // 4 for s in range(width)]
        return [POSITIONS[r] + s % 4 for s, r in enumerate(reqs)], reqs
    pad = int(rowset[1:])
    live = width - pad
    return POSITIONS[:live] + [None] * pad, list(range(live)) + [0] * pad


def _digest(t: torch.Tensor) -> str:
    data = t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _index_at(slots: torch.Tensor):
    """Where the index cache holds each of ``slots``: (page, byte columns) of a token's
    128 keys, in 16-token x 16-byte tiles as SGLang stores them, then its scale."""
    page, t = slots[:, None] // PAGE, slots[:, None] % PAGE
    d = torch.arange(INDEX_DIM).to(t)
    keys = (t // 16) * 16 * INDEX_DIM + (t % 16) * 16 + (d // 16) * 256 + d % 16
    scale = PAGE * INDEX_DIM + t * 4 + torch.arange(4).to(t)
    return page, torch.cat([keys, scale], 1)


def _index_entries(idx: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """The index cache's bytes of ``slots``: each token's 128 keys, then its scale."""
    return idx.view(idx.shape[0], -1)[_index_at(slots)]


def _changed_slots(before: torch.Tensor, after: torch.Tensor, index: bool) -> set:
    if index:
        every = torch.arange(before.shape[0] * PAGE, device=before.device)
        before, after = _index_entries(before, every), _index_entries(after, every)
    a, b = before.view(torch.uint8), after.view(torch.uint8)
    return set((a != b).any(1).nonzero().flatten().tolist())


def writes_only_its_rows(model: Model) -> bool:
    """Each launch of one step writes exactly its live rows' MLA and index cache rows
    (filled with 0xFF bytes first, as earlier steps wrote the same values); a shared
    launch leaves the top-k as it was."""
    st = model.step_tables
    slots = st.slots[st.slots >= 0]
    live = set(slots.tolist())
    for c in model.caches:
        c.kv.view(torch.uint8)[slots] = 0xFF
        if c.idx is not None:
            c.idx.view(c.idx.shape[0], -1)[_index_at(slots)] = 0xFF
    before = [
        (c.kv.clone(), c.idx.clone() if c.idx is not None else None)
        for c in model.caches
    ]
    topk = []
    model.step(after=lambda i, _: topk.append(st.topk.clone()))
    torch.cuda.synchronize()
    ok = True
    for i, (c, (kv, idx)) in enumerate(zip(model.caches, before)):
        wrote = [("MLA", _changed_slots(kv, c.kv, False))]
        if idx is not None:
            wrote.append(("index", _changed_slots(idx, c.idx, True)))
        for name, got in wrote:
            if got != live:
                print(
                    f"launch {i}: wrote {name} slots {sorted(got)}, not {sorted(live)}"
                )
                ok = False
        prev = model.topk0 if i == 0 else topk[i - 1]
        if model.kinds[i] == "shared" and not torch.equal(topk[i], prev):
            print(f"launch {i}: a shared launch wrote the top-k", flush=True)
            ok = False
    return ok


def check_replays(model: Model, replays: int) -> bool:
    """A graph of the step, replayed, gives the eager step's output each time."""
    want = model.step().clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = model.step()
    ok = True
    for n in range(replays):
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(out, want):
            print(f"replay {n} differs from the eager step", flush=True)
            ok = False
    del graph
    return ok


def check_tags(model: Model) -> bool:
    """Every selected token of a scored row (one past 2048 keys) is a pair tagged
    with its launch's tag and holds the slot the launch publishes in the top-k; a
    pair whose tag is cleared fails the check (negative control)."""
    st = model.step_tables
    scored = [
        s for s, p in enumerate(st.positions.tolist()) if st.slots[s] >= 0 and p >= TOPK
    ]
    seen = []

    def after(i, _):
        rt = model.ops[i].rt
        if model.kinds[i] == "index":
            at, n = rt.sc["indices"], rt.width * TOPK * 8
            pairs = rt.scratch[at : at + n].view(torch.int32).view(rt.width, TOPK, 2)
            tag = int(rt.epoch[0]) * rt.key.launches_per_step + model.ops[i].layer + 1
            seen.append((i, pairs.clone(), st.topk.clone(), tag))

    model.step(after=after)
    torch.cuda.synchronize()

    def ok(pairs, topk, tag) -> bool:
        for s in scored:
            begin = int(st.indptr[s])
            if not (pairs[s, :, 1] == tag).all():
                return False
            if not torch.equal(pairs[s, :, 0], topk[begin : begin + TOPK]):
                return False
        return True

    good = True
    for i, pairs, topk, tag in seen:
        if not ok(pairs, topk, tag):
            print(f"launch {i}: a selected token is not the launch's pair", flush=True)
            good = False
        pairs[scored[0], TOPK - 1, 1] = 0
        if ok(pairs, topk, tag):
            print(f"launch {i}: a cleared tag passed the check", flush=True)
            good = False
    return good and bool(seen) and bool(scored)


def dump(model: Model, name: str, steps: int, perturb: bool, out: dict) -> str:
    """The digests of ``steps`` steps of ``model`` into ``out``; returns the last
    output's."""
    st = model.step_tables
    slots = st.slots[st.slots >= 0]
    if perturb:
        model.h[0, 0] = -model.h[0, 0]
    for step in range(steps):

        def after(i, h, step=step):
            d = {f"out.r{s}": _digest(h[s]) for s in range(h.shape[0])}
            c = model.caches[i]
            d["kv_rows"] = _digest(c.kv[slots])
            if c.idx is not None:
                d["index_rows"] = _digest(_index_entries(c.idx, slots))
                d["topk"] = _digest(st.topk)
            out[f"{name}/s{step}/l{i}"] = d

        h = model.step(after=after)
        torch.cuda.synchronize()
    whole = {f"kv.l{i}": _digest(c.kv) for i, c in enumerate(model.caches)}
    whole.update(
        {
            f"idx.l{i}": _digest(c.idx)
            for i, c in enumerate(model.caches)
            if c.idx is not None
        }
    )
    whole["error_words"] = model.errors()
    out[f"{name}/caches"] = whole
    return _digest(h)


def compare(a_dir: str, b_dir: str) -> int:
    equal = differ = one_sided = 0
    names = []
    files = sorted(f for f in os.listdir(a_dir) if f.startswith("digests_"))
    if not files or sorted(os.listdir(b_dir)) != sorted(os.listdir(a_dir)):
        print(f"COMPARE the dump files differ: {a_dir} {b_dir}")
        return 1
    for f in files:
        A = json.load(open(os.path.join(a_dir, f)))
        B = json.load(open(os.path.join(b_dir, f)))
        for key in sorted(set(A) | set(B)):
            a, b = A.get(key, {}), B.get(key, {})
            for k in sorted(set(a) | set(b)):
                if k not in a or k not in b:
                    one_sided += 1
                    names.append(f"{f}:{key}:{k} (one side)")
                elif a[k] == b[k]:
                    equal += 1
                else:
                    differ += 1
                    names.append(f"{f}:{key}:{k}")
    print(f"COMPARE equal={equal} differ={differ} one_sided={one_sided}")
    for n in names[:20]:
        print("  differs:", n)
    return 1 if differ or one_sided else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("shared", "index", "model"), default="shared")
    ap.add_argument("--rows", default="2,4,8", help="comma-separated widths")
    ap.add_argument("--rowsets", default="p0", help="comma-separated: p0, p1, v4")
    ap.add_argument("--launches", type=int, default=3, help="launches a step")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--kv", choices=("bf16", "fp8"), default="bf16")
    ap.add_argument("--deadline", type=float, default=10.0)
    ap.add_argument("--writes", action="store_true")
    ap.add_argument("--replays", type=int, default=0, help="graph replays of a step")
    ap.add_argument("--check-tags", action="store_true")
    ap.add_argument("--dump", default="", help="write the digests into DIR")
    ap.add_argument("--perturb", action="store_true", help="negate one input value")
    ap.add_argument("--compare", nargs=2, metavar=("A_DIR", "B_DIR"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    kinds = (
        model_kinds(args.launches)
        if args.kind == "model"
        else [args.kind] * args.launches
    )
    ok, digests, outs = True, {}, {}
    for W, rowset in [
        (W, r) for W in map(int, args.rows.split(",")) for r in args.rowsets.split(",")
    ]:
        positions, requests = rows_of(W, rowset)
        model = Model(kinds, positions, requests, rank, tp, group, device, args)
        name = f"{args.kv}-{args.kind}/w{W}{rowset}"
        if args.dump:
            outs[name] = dump(model, name, args.steps, args.perturb, digests)
        else:
            for _ in range(args.steps):
                outs[name] = _digest(model.step())
        if args.writes:
            ok &= writes_only_its_rows(model)
        if args.replays:
            ok &= check_replays(model, args.replays)
        if args.check_tags:
            ok &= check_tags(model)
        torch.cuda.synchronize()
        if model.errors():
            print(f"rank {rank} {name}: error words set", flush=True)
            ok = False
        model.close()
        del model
        torch.cuda.empty_cache()
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        run = f"{args.kv}_{args.kind}_{args.rowsets.replace(',', '')}"
        with open(os.path.join(args.dump, f"digests_{run}_r{rank}.json"), "w") as f:
            json.dump(digests, f, indent=0, sort_keys=True)
    every = [None] * tp
    dist.all_gather_object(every, (ok, outs), group=group)
    if rank == 0:
        same = all(o == every[0][1] for _, o in every)
        passed = same and all(k for k, _ in every)
        print(f"CHECK kind={args.kind} tp={tp} ranks_equal={same} passed={passed}")
    else:
        passed = True
    dist.barrier(group=group)
    dist.destroy_process_group()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
