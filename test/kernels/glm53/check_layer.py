"""Run the GLM-5.3 layer launch on random weights in SGLang's layouts under torchrun
(TP 4 or 8): ``torchrun --nproc-per-node 8 test/kernels/glm53/check_layer.py --kind
index --rows 2,4,8``. Kinds: ``shared`` (a layer that reuses the previous top-k),
``index`` (a layer that runs the indexer), ``model`` (the model order's first
``--launches`` layers: 3 shared, then an indexer layer every fourth), ``dense`` (a
dense layer: the indexer layer with its MLP as expert-shaped slices), ``embed``
(layer 0 from the step's token ids, then dense layers), ``full`` (layer 0, the dense
layers 1 and 2, then the model order, and the tail) and ``head`` (the tail alone:
the final norm, the LM head and the logits' all-gather). Each width runs
``--steps`` steps of ``--launches`` launches, inputs chained, on the rows of each of
``--rowsets`` (one request a row, the same with a pad row, or MTP's verify rows; the
live rows at positions below and past the 2048-key top-k), with the MLA cache in
bf16 or E4M3 (``--kv``). Every rank must finish with no error word set and the same
output (the TP sums run in rank order). ``--writes``: each launch changes exactly
its live rows' MLA and index cache rows, and a shared launch leaves the top-k alone;
``--replays N``: a graph of the step replays the eager step bitwise;
``--check-tags``: every selected token of a scored row is a pair carrying its
indexer launch's tag and the slot the launch publishes, and a cleared tag fails;
``--check-dense``: the last dense launch's MLP against f32 references on its own
inputs, within the spec's bounds, and one rank's down or gate | up scales x2 fail
them; ``--check-embed``: layer 0 from the token ids equals the dense launch fed the
gathered embedding rows (outputs, cache rows, top-k), and a changed token id differs;
``--check-head``: the tail's logits against a torch and an f32 tail, within the
spec's bounds, and the final norm x1.01 fails them. ``--similar X``: the live rows of a
request are one random row plus X times their own, so they route alike, as a verify
step's rows do (with ``--rowsets v4``).

``--dump DIR`` writes, per run and rank, the SHA-256 of every launch's output rows
(the tail's: its gathered logits), of the cache rows and top-k entries it writes,
and after each width of every cache whole (``--perturb`` negates one input value, a
negative control); ``--compare A B`` matches two dump directories and exits 1 on
any difference."""

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
# GLM-5.3's vocabulary and dense MLP intermediates (over all ranks)
VOCAB = 154880
DENSE_INTER = 12288
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
    """The kinds of the first n MoE layers (from layer 3): SGLang runs the indexer on
    layer i unless max(i - 2, 0) % 4."""
    return ["index" if (i - 2) % 4 == 0 else "shared" for i in range(3, 3 + n)]


def kinds_of(kind: str, n: int) -> list:
    """The layer kinds of ``n`` launches of ``--kind`` (``head``: none)."""
    if kind == "model":
        return model_kinds(n)
    if kind == "full":
        return ["embed", "dense", "dense"] + model_kinds(n - 3)
    if kind == "embed":
        return ["embed"] + ["dense"] * (n - 1)
    return [] if kind == "head" else [kind] * n


def has_index(kind: str) -> bool:
    return kind in ("index", "dense", "embed")


def embedding(rank: int, tp: int, device) -> torch.Tensor:
    """This rank's rows of a random vocab-parallel embedding."""
    g = _gen(70000 + rank, device)
    rows = VOCAB // tp
    return (0.02 * torch.randn(rows, HIDDEN, generator=g, device=device)).bfloat16()


def head_weights(rank: int, tp: int, device):
    """The tail's random tensors: the final norm (every rank alike) and this rank's
    LM-head rows."""
    from sgl_amd_mono.glm53.gfx950.runner import Glm53HeadWeights

    g = _gen(90000 + rank, device)
    w = torch.randn(VOCAB // tp, HIDDEN, generator=g, device=device) * HIDDEN**-0.5
    g_norm = _norm(_gen(90000, device), HIDDEN, device)
    return Glm53HeadWeights(g_norm=g_norm, w_head=w.bfloat16())


def layer_weights(layer: int, rank: int, tp: int, device, kind: str):
    """Layer ``layer``'s random tensors as this rank's shard (every rank draws the
    same): block-FP8 projections and experts read in place, W_UK / W_UV, the router
    and the indexer's weights_proj packed; the shared expert is slot 256. A dense
    layer's MLP: its intermediates as expert-shaped slices, no router read; layer 0
    (``embed``) reads this rank's embedding rows in the router's place."""
    from sgl_amd_mono.glm53.gfx950.packing import pack_bf16, pack_fp8
    from sgl_amd_mono.glm53.gfx950.runner import Glm53Weights

    g = _gen(1000 * layer + 7, device)
    index = has_index(kind)
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
    w_r, dense = pack_bf16(router.bfloat16().contiguous()), 0
    if kind in ("dense", "embed"):
        dense = DENSE_INTER // tp // inter
        w_ug = _fp8(g, (dense, 2 * inter, HIDDEN), device)
        s_ug = _scales(g, (dense, 2 * inter // 128, HIDDEN // 128), HIDDEN, device)
        w_dn = _fp8(g, (dense, HIDDEN, inter), device)
        s_dn = _scales(g, (dense, HIDDEN // 128, inter // 128), DENSE_INTER, device)
    if kind == "embed":
        w_r = embedding(rank, tp, device)
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
        w_r=w_r,
        bias=bias,
        w_ug=w_ug,
        s_ug=s_ug,
        w_dn=w_dn,
        s_dn=s_dn,
        shuffled=True,
        index=table,
        keep=keep,
        dense=dense,
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
        # SGLang's DSA backend gives each row its request's page-table row, and the
        # spec each row that row
        req_ids=torch.arange(W, dtype=i32, device=device),
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


def token_ids(positions: list, device) -> torch.Tensor:
    """Random token ids of the live rows, over every rank's shard; -1 a pad row."""
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(0, VOCAB, (len(positions),), generator=g, dtype=torch.int32)
    ids[[p is None for p in positions]] = -1
    return ids.to(device)


class Model:
    """Random layers of ``kinds`` (layer ids from 3, or from 0 when the first is
    ``embed``), their caches and one runtime a kind at one width, then with ``head``
    the tail; ``step()`` runs a step of launches, inputs chained. ``stop=False``: a
    failed wait sets its error word and the launch runs on."""

    def __init__(
        self,
        kinds,
        positions,
        requests,
        rank,
        tp,
        group,
        device,
        args,
        stop=False,
        head=False,
        first=None,
        similar=0.0,
    ):
        from sgl_amd_mono.glm53.gfx950.runner import (
            Glm53Build,
            Glm53HeadOp,
            Glm53LayerOp,
            Glm53Runtime,
        )

        self.kinds, W = kinds, len(positions)
        if first is None:
            first = 0 if kinds[:1] == ["embed"] else 3
        live = [p for p in positions if p is not None]
        pages = Pages(max(requests) + 1, max(live) + W + 1, device)
        self.caches = [
            layer_caches(first + i, pages, has_index(k), device, args.kv == "fp8")
            for i, k in enumerate(kinds)
        ]
        self.cos, self.sin = rope(max(live) + W + 1, device)
        self.step_tables, self.topk0 = make_step(positions, requests, pages, device)
        self.weights = [
            layer_weights(first + i, rank, tp, device, k) for i, k in enumerate(kinds)
        ]
        vocab_rows = VOCAB // tp
        self.runtimes = {}
        for kind in sorted(set(kinds)):
            w = self.weights[kinds.index(kind)]
            key = Glm53Build(
                rows=W,
                tp=tp,
                launches_per_step=kinds.count(kind),
                with_indexer=has_index(kind),
                index_max_seq=MAX_SEQ if has_index(kind) else 0,
                fp8_shuffled=True,
                kv_fp8=args.kv == "fp8",
                stop=stop,
                dense=w.dense,
                tokens=kind == "embed",
                vocab_rows=vocab_rows if kind == "embed" else 0,
            )
            self.runtimes[kind] = Glm53Runtime(key, rank, group, device)
            self.runtimes[kind].set_deadline(args.deadline)
        self.ops, ordinal = [], {k: 0 for k in self.runtimes}
        for kind, w in zip(kinds, self.weights):
            self.ops.append(Glm53LayerOp(self.runtimes[kind], w, ordinal[kind]))
            ordinal[kind] += 1
        self.head = None
        if head:
            key = Glm53Build(
                rows=W,
                tp=tp,
                launches_per_step=1,
                with_indexer=False,
                index_max_seq=0,
                fp8_shuffled=True,
                stop=stop,
                head=True,
                vocab_rows=vocab_rows,
            )
            rt = self.runtimes["head"] = Glm53Runtime(key, rank, group, device)
            rt.set_deadline(args.deadline)
            self.head = Glm53HeadOp(rt, head_weights(rank, tp, device))
        g = _gen(3, device)
        h = torch.randn(W, HIDDEN, generator=g, device=device)
        if similar:
            base = h.clone()
            noise = similar * torch.randn(W, HIDDEN, generator=g, device=device)
            for s, (p, r) in enumerate(zip(positions, requests)):
                if p is not None:
                    h[s] = base[r] + noise[s]
        self.h = h.bfloat16()
        if kinds[:1] == ["embed"]:
            self.h = token_ids(positions, device)

    def step(self, after=None, h=None) -> torch.Tensor:
        """One step from ``h`` (default ``self.h``; ``after(i, out)`` after launch i);
        returns the last layer's output, or with the tail its gathered logits."""
        st = self.step_tables
        st.topk.copy_(self.topk0)
        h = self.h if h is None else h
        for i, op in enumerate(self.ops):
            h = op.launch(h, self.caches[i], self.cos, self.sin, st)
            if after is not None:
                after(i, h)
        if self.head is not None:
            h, _ = self.head.launch(h)
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


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _sum_ranks(t: torch.Tensor, group) -> torch.Tensor:
    c = t.float().cpu()
    dist.all_reduce(c, group=group)
    return c.to(t.device)


def _gather_cols(t: torch.Tensor, group, tp: int) -> torch.Tensor:
    parts = [torch.empty_like(t.float().cpu()) for _ in range(tp)]
    dist.all_gather(parts, t.float().cpu().contiguous(), group=group)
    return torch.cat(parts, -1).to(t.device)


def _live(model: Model) -> int:
    """The step's live rows: those before its pad rows."""
    return int((model.step_tables.slots >= 0).sum())


def dense_values(model: Model, out: torch.Tensor, group) -> dict:
    """The last dense launch's MLP (its own mailboxes, ``check.dense_parts``) on the
    live rows: its mid against an f32 gate | up on its own FP8 activations, its output
    against the f32 down of its own per-128 FP8 mid, and its output and the same MLP
    with SGLang's W8A8 rounding (bf16 normed rows, FP8 per 128, bf16 gate | up and
    mid) against an f32 MLP on its FFN input; their distances' ratio less one is
    ``excess``. Every rank's shard summed."""
    from sgl_amd_mono.glm53 import check

    op = model.ops[-1]
    rt, w, S = op.rt, op.w, _live(model)
    a = check.ffn_input(rt, rt.width)[:S]
    parts = check.dense_parts(rt, w, rt.width)
    x = check._rms(a, w.g_post)
    wug = check.slices_f32(w.w_ug, w.s_ug, w.shuffled)
    wdn = check.slices_f32(w.w_dn, w.s_dn, w.shuffled)
    inter = wug.shape[1] // 2

    def mlp(x, rounded):
        gu = torch.einsum("sk,njk->snj", check._q128(x) if rounded else x, wug)
        if rounded:
            gu = gu.bfloat16().float()
        mid = torch.nn.functional.silu(gu[..., :inter]) * gu[..., inter:]
        if rounded:
            mid = check._q128(mid.bfloat16())
        return torch.einsum("sni,nhi->sh", mid, wdn)

    f32 = _sum_ranks(mlp(x, False), group)
    stock = _sum_ranks(mlp(x.bfloat16().float(), True), group)
    down = _sum_ranks(parts["down"][:S], group)
    mine = out[:S].float() - a.float()
    v = {
        "mid": _rel(parts["mid"][:S], parts["mid_ref"][:S]),
        "x": _rel(mine, down),
        "x_vs_fp32": _rel(mine, f32),
        "stock_vs_fp32": _rel(stock, f32),
    }
    v["excess"] = v["x_vs_fp32"] / v["stock_vs_fp32"] - 1
    return v


def check_dense(model: Model, rank: int, group) -> bool:
    """The last dense launch's MLP within the spec's dense bounds; rank 0's down scales
    x2, or its gate | up scales x2, must fail one of them."""
    from sgl_amd_mono.glm53.spec import NUMERICS

    def within(v):
        bounds = NUMERICS.bounds
        return all(v[k] <= bounds[f"dense.{k}.rel"] for k in ("mid", "x", "excess"))

    w = model.ops[-1].w
    out = model.step()
    torch.cuda.synchronize()
    v = dense_values(model, out, group)
    ok = within(v)
    print(f"rank {rank} dense {v} within={ok}", flush=True)
    for name, t in (("down", w.s_dn), ("gate_up", w.s_ug)):
        saved = t.clone()
        if rank == 0:
            t.mul_(2)
        out = model.step()
        torch.cuda.synchronize()
        t.copy_(saved)
        bad = dense_values(model, out, group)
        print(f"rank {rank} dense control {name} x2: {bad}", flush=True)
        ok &= not within(bad)
    return ok


def _writes(model: Model, h=None):
    """One step's output, the MLA and index cache rows of its live rows and its
    top-k, the caches restored after."""
    st = model.step_tables
    saved = [(c.kv.clone(), c.idx.clone()) for c in model.caches]
    out = model.step(h=h)
    torch.cuda.synchronize()
    slots = st.slots[st.slots >= 0]
    c = model.caches[0]
    got = [
        out.clone(),
        c.kv[slots].clone(),
        _index_entries(c.idx, slots),
        st.topk.clone(),
    ]
    for c, (kv, idx) in zip(model.caches, saved):
        c.kv.copy_(kv)
        c.idx.copy_(idx)
    return got


def check_embed(model: Model, dense: Model, rank: int, group) -> bool:
    """Layer 0 from the token ids (``model``, one launch) equals the same layer as a
    dense launch (``dense``) fed the gathered embedding rows: outputs, cache rows and
    top-k bitwise. One live row's token id changed must change the output."""
    ids, emb = model.h, model.ops[0].w.w_r
    rows = emb.shape[0]
    local = torch.zeros(ids.shape[0], HIDDEN, device=ids.device)
    for s, t in enumerate(ids.tolist()):
        if t >= 0 and t // rows == rank:
            local[s] = emb[t - rank * rows].float()
    x = _sum_ranks(local, group).bfloat16()
    a, b = _writes(model), _writes(dense, h=x)
    same = [torch.equal(p.view(torch.uint8), q.view(torch.uint8)) for p, q in zip(a, b)]
    print(f"rank {rank} embed equal (out, kv, index, topk): {same}", flush=True)
    saved = ids.clone()
    ids[0] = (ids[0] + rows // 2) % VOCAB
    changed = _writes(model)[0]
    ids.copy_(saved)
    differs = not torch.equal(changed.view(torch.uint8), a[0].view(torch.uint8))
    print(f"rank {rank} embed control (a changed token id) differs: {differs}")
    return all(same) and differs


def check_head(model: Model, rank: int, tp: int, group) -> bool:
    """The tail alone: its logits against a torch tail (bf16 normed rows, bf16 logits,
    gathered) and an f32 tail within the spec's bounds, its normed rows against
    torch's; the final norm's weight x1.01 must fail ``head.logits.rel``."""
    from sgl_amd_mono.glm53 import check
    from sgl_amd_mono.glm53.spec import NUMERICS

    bounds, w, x = NUMERICS.bounds, model.head.w, model.h
    logits = model.step().clone()
    torch.cuda.synchronize()
    normed = model.head.rt.outs[1].clone()
    want = check._rms(x, w.g_norm).bfloat16()
    ref = _gather_cols((want @ w.w_head.t()).float(), group, tp)
    f32 = _gather_cols(check._rms(x, w.g_norm) @ w.w_head.float().t(), group, tp)
    v = {
        "logits": _rel(logits, ref),
        "logits_vs_fp32": _rel(logits, f32),
        "stock_vs_fp32": _rel(ref, f32),
        "normed": _rel(normed, want),
        "top1": (logits.argmax(-1) == ref.argmax(-1)).float().mean().item(),
    }
    ok = (
        v["logits"] <= bounds["head.logits.rel"]
        and v["logits_vs_fp32"] <= bounds["head.logits_vs_fp32.rel"]
    )
    saved = w.g_norm.clone()
    w.g_norm.mul_(1.01)
    bad = _rel(model.step(), ref)
    torch.cuda.synchronize()
    w.g_norm.copy_(saved)
    print(f"rank {rank} head {v} within={ok}; final norm x1.01: {bad:.4g}", flush=True)
    return ok and bad > bounds["head.logits.rel"]


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
        if model.head is not None:
            out[f"{name}/s{step}/tail"] = {
                f"out.r{s}": _digest(h[s]) for s in range(h.shape[0])
            }
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
    ap.add_argument(
        "--kind",
        choices=("shared", "index", "model", "dense", "embed", "full", "head"),
        default="shared",
    )
    ap.add_argument("--rows", default="2,4,8", help="comma-separated widths")
    ap.add_argument("--rowsets", default="p0", help="comma-separated: p0, p1, v4")
    ap.add_argument("--launches", type=int, default=3, help="launches a step")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--kv", choices=("bf16", "fp8"), default="bf16")
    ap.add_argument("--deadline", type=float, default=10.0)
    ap.add_argument("--writes", action="store_true")
    ap.add_argument("--replays", type=int, default=0, help="graph replays of a step")
    ap.add_argument("--check-tags", action="store_true")
    ap.add_argument("--check-dense", action="store_true")
    ap.add_argument("--check-embed", action="store_true", help="with --launches 1")
    ap.add_argument("--check-head", action="store_true")
    ap.add_argument("--dump", default="", help="write the digests into DIR")
    ap.add_argument("--perturb", action="store_true", help="negate one input value")
    ap.add_argument("--similar", type=float, default=0.0, help="rows routing alike")
    ap.add_argument("--compare", nargs=2, metavar=("A_DIR", "B_DIR"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    kinds = kinds_of(args.kind, args.launches)
    head = args.kind in ("full", "head")
    ok, digests, outs = True, {}, {}
    for W, rowset in [
        (W, r) for W in map(int, args.rows.split(",")) for r in args.rowsets.split(",")
    ]:
        positions, requests = rows_of(W, rowset)
        model = Model(
            kinds,
            positions,
            requests,
            rank,
            tp,
            group,
            device,
            args,
            head=head,
            similar=args.similar,
        )
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
        if args.check_dense:
            ok &= check_dense(model, rank, group)
        if args.check_embed:
            dense = Model(
                ["dense"], positions, requests, rank, tp, group, device, args, first=0
            )
            ok &= check_embed(model, dense, rank, group)
            dense.close()
            del dense
        if args.check_head:
            ok &= check_head(model, rank, tp, group)
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
