# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 MLA front (``mla_front``) against SGLang's DCP8 MLA ops, on one GPU.

One request's verify block (6 rows) at committed length P, the layer's weights
(seeded random, or one rank's real copy with its 12 heads standing for all 96 as
the one-rank harness builds them), and every rank's share of the request's
keys. For a rank r the front and SGLang's ops of rank r take the same x, the
same KV slice and the same rows and positions:

* ``cast``: the front's e4m3 conversion against torch's, on every bf16 value;
* ``frontier``: the front's (local keys, owned row) against SGLang's DCP
  chain lengths, for every P mod 8 and rank;
* ``x``: the seam against the AttnRes math in torch;
* ``kv``: the new record the front writes against SGLang's casts of the front's
  own latent (the normed latent and the rope part), bit for bit;
* ``q``: the front's e4m3 q against SGLang's cast of its own bf16 q;
* ``attn``, ``send``: the attention partial (merged from the front's splits)
  and the V-projected send records against an fp32 reference of the same fp8
  q and KV: ``|front - ref| <= 1.25 |stock - ref|``;
* ``core`` (``--ranks 8``): the eight ranks' launches run one after another,
  then SGLang's all-to-all (a copy here), LSE combine and output gate, against
  SGLang's full DCP8 attention on the same inputs, and run to run bit for bit;
* ``--invariance``: batch invariance -- each request alone (6 rows) and with 1,
  3 and 7 others of mixed contexts (12, 24, 48 rows, at different places in the
  batch): every output of the request bit for bit the same at every width, and
  the widest batch run to run.
"""

import argparse
import math
import sys
import types

import torch
import torch.nn.functional as F

BF = torch.bfloat16
F8 = torch.float8_e4m3fn
HIDDEN, QL, KV, ROPE, DQK, LATW = 7168, 1536, 512, 64, 576, 2112
NHA, NHL, QHD, QN, VD, DCP, G = 96, 12, 192, 128, 128, 8, 6
NVB = 3  # live bank rows
MAX_BLOCKS = 8


def rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def rms(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


# ---------------------------------------------------------------- inputs


def layer_weights(args, device):
    """The front's weights: real from a one-rank copy (``--copy``, the layer's
    file), or seeded random of the served shapes."""
    if args.copy:
        return real_weights(args.copy, args.layer, device)
    g = torch.Generator(device=device).manual_seed(args.seed)

    def rn(*shape, scale):
        return (torch.randn(*shape, generator=g, device=device) * scale).to(BF)

    return {
        "ares_nw": (1 + rn(HIDDEN, scale=0.1)).to(BF),
        "ares_qk": rn(HIDDEN, scale=0.02),
        "in_nw": (1 + rn(HIDDEN, scale=0.1)).to(BF),
        "w_qkv": rn(LATW, HIDDEN, scale=0.02),
        "w_g": rn(NHL * VD, HIDDEN, scale=0.02),
        "q_nw": (1 + rn(QL, scale=0.1)).to(BF),
        "kv_nw": (1 + rn(KV, scale=0.1)).to(BF),
        "w_qb": rn(NHA * QHD, QL, scale=0.035),
        "w_kc": rn(NHA, QN, KV, scale=0.09),
        "w_vc": rn(NHA, KV, VD, scale=0.045),
        "eps": 1e-5, "out_eps": 1e-5, "q_eps": 1e-6, "kv_eps": 1e-6,
        "scale": QHD**-0.5,
    }  # fmt: skip


def real_weights(path, layer, device):
    """Layer ``layer`` of a tier-0 one-rank copy (``L<i>.pt`` + its manifest):
    the rank's 12 heads of q_b / W_UK / W_UV repeated for all 96, as SGLang's
    replicas are on a one-rank run."""
    import json

    man = json.load(open(f"{path}/L{layer}.pt.json"))
    blobs = torch.load(f"{path}/L{layer}.pt", mmap=True, weights_only=True)
    views = {}
    for key, spec in man.items():
        flat = blobs[key]
        for name, off, shape, stride, dt in spec["views"]:
            dt = getattr(torch, dt.split(".")[-1])
            flat_t = flat.view(torch.uint8).view(dt)
            views[name] = flat_t.as_strided(shape, stride, off // flat_t.element_size())
    p = f"language_model.model.layers.{layer}."
    a = p + "self_attn."

    def t(n):
        return views[n].to(device)

    w_qb = t(a + "q_b_proj.weight").contiguous()  # [12 * 192, 1536]
    w_kc = t(a + "w_kc").contiguous()  # [12, 128, 512]
    w_vc = t(a + "w_vc").contiguous()  # [12, 512, 128]
    return {
        "ares_nw": t(p + "self_attention_res_norm.weight").contiguous(),
        "ares_qk": t(p + "self_attention_res_proj.weight").reshape(HIDDEN).contiguous(),
        "in_nw": t(p + "input_layernorm.weight").contiguous(),
        "w_qkv": t(a + "fused_qkv_a_proj_with_mqa.weight").contiguous(),
        "w_g": t(a + "g_proj.weight").contiguous(),
        "q_nw": t(a + "q_a_layernorm.weight").contiguous(),
        "kv_nw": t(a + "kv_a_layernorm.weight").contiguous(),
        "w_qb": w_qb.repeat(DCP, 1).contiguous(),
        "w_kc": w_kc.repeat(DCP, 1, 1).contiguous(),
        "w_vc": w_vc.repeat(DCP, 1, 1).contiguous(),
        "eps": 1e-6, "out_eps": 1e-6, "q_eps": 1e-6, "kv_eps": 1e-6,
        "scale": QHD**-0.5,
    }  # fmt: skip


def stream_inputs(seed, device):
    g = torch.Generator(device=device).manual_seed(seed + 1)
    return {
        "prefix": (torch.randn(G, HIDDEN, generator=g, device=device) * 0.3).to(BF),
        "blocks": (torch.randn(G, MAX_BLOCKS, HIDDEN, generator=g, device=device) * 0.3).to(BF),
    }


def seam_x(st, w):
    """The attention-side AttnRes seam and input_layernorm, in torch."""
    rows = torch.cat([st["blocks"][:, :NVB, :], st["prefix"].unsqueeze(1)], dim=1)
    scores = (rms(rows, w["ares_nw"], w["eps"]) * w["ares_qk"].float()).sum(-1)
    probs = torch.softmax(scores, dim=-1)
    mixed = (probs.unsqueeze(-1) * rows.float()).sum(dim=1)
    return rms(mixed, w["in_nw"], w["out_eps"]).to(BF)


def make_keys(w, p, seed, device, temp):
    """Every position's e4m3 KV record (positions 0 .. p - 1): the layer's own
    fused_qkv_a, kv norm and casts applied to random hidden rows, the latent
    scaled by ``temp`` (it sets how peaked the softmax is)."""
    g = torch.Generator(device=device).manual_seed(seed + 2)
    out = torch.empty(p, DQK, dtype=F8, device=device)
    for s0 in range(0, p, 8192):
        n = min(8192, p - s0)
        h = torch.randn(n, HIDDEN, generator=g, device=device).to(BF)
        lat = F.linear(h, w["w_qkv"])[:, QL:]
        kn = (rms(lat[:, :KV], w["kv_nw"], w["kv_eps"]) * temp).to(BF)
        out[s0 : s0 + n, :KV] = kn.to(F8)
        out[s0 : s0 + n, KV:] = lat[:, KV:].to(F8)
    return out


class Rank:
    """Rank r's share of one request: its pool rows (a random placement), the
    ragged local index list SGLang's DCP verify uses, and the rows' slots."""

    def __init__(self, keys, p, r, device, seed):
        self.p, self.r = p, r
        self.lr = p // DCP + int(r < p % DCP)
        own = (r - p) % DCP
        self.own = own if own < G else DCP
        nloc = self.lr + int(self.own < G)
        g = torch.Generator(device="cpu").manual_seed(seed * 31 + r)
        nrows = nloc + 64
        self.rows = torch.randperm(nrows, generator=g)[: max(nloc, 1)].to(torch.int32)
        self.pool = torch.zeros(nrows, DQK, dtype=F8, device=device)
        prefix_rows = self.rows[: self.lr].long().to(device)
        self.pool[prefix_rows] = keys[r::DCP][: self.lr]
        # SGLang's ragged list: an empty span on a rank with no key of the request
        self.kv_idx = self.rows[:nloc].to(device)
        self.kv_ptr = torch.tensor([0, nloc], dtype=torch.int32, device=device)
        # the verify rows' slots: position q lives at rank q % 8, local row q // 8
        self.loc = torch.zeros(G, dtype=torch.int64, device=device)
        if self.own < G:
            self.loc[self.own] = int(self.rows[self.lr]) * DCP + r
        for i in range(G):
            if i != self.own:
                self.loc[i] = 7 * DCP + (r + 1) % DCP  # another rank's slot
        self.g_kv_ptr = torch.tensor([0, p + G], dtype=torch.int32, device=device)


class Batch:
    """Rank r's share of several requests in one pool: each request's ``Rank``
    layout (its own seed), the pools stacked, the index lists and slots
    shifted, the ragged and global indptrs over the requests."""

    def __init__(self, reqs, r, device):
        self.r, self.args, self.device = r, reqs, device
        self.reqs = [Rank(keys, p, r, device, sd) for p, keys, sd in reqs]
        idx, loc, ptr, gptr, base = [], [], [0], [0], 0
        for rk in self.reqs:
            nloc = int(rk.kv_ptr[1])
            idx.append(rk.kv_idx[:nloc] + base)
            lc = rk.loc.clone()
            if rk.own < G:
                lc[rk.own] = (int(rk.rows[rk.lr]) + base) * DCP + r
            loc.append(lc)
            ptr.append(ptr[-1] + nloc)
            gptr.append(gptr[-1] + rk.p + G)
            rk.base = base
            base += rk.pool.shape[0]
        self.pool = torch.cat([rk.pool for rk in self.reqs])
        self.kv_idx = torch.cat(idx)
        self.loc = torch.cat(loc)
        self.kv_ptr = torch.tensor(ptr, dtype=torch.int32, device=device)
        self.g_kv_ptr = torch.tensor(gptr, dtype=torch.int32, device=device)
        self.s = G * len(self.reqs)

    def own_row(self, j):
        rk = self.reqs[j]
        return rk.base + int(rk.rows[rk.lr]) if rk.own < G else None

    def at_rank(self, o):
        """The same requests as rank o's share."""
        return self if o == self.r else Batch(self.args, o, self.device)


def batch_stream(ids, seed, device):
    """The requests' rows (6 each, seeded by request id), stacked."""
    sts = [stream_inputs(seed + 101 * i, device) for i in ids]
    return {k: torch.cat([x[k] for x in sts]).contiguous() for k in sts[0]}


# the QG copy every launch here takes (--qg-par)
QG_PAR = [0]


def new_epoch(device, seconds=30.0):
    """A launch's epoch buffer: epoch 0 and every wait's deadline."""
    from sgl_amd_mono.common import guard

    epoch = torch.zeros(8, dtype=torch.int32, device=device)
    ticks = guard.deadline_ticks(seconds)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    return epoch


def front_key(w, s, timeline=False, records=True):
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

    return mf.MlaFrontBuild(
        tokens=s, nblocks=NVB, scale=w["scale"], eps=w["eps"], out_eps=w["out_eps"],
        q_eps=w["q_eps"], kv_eps=w["kv_eps"], qg_par=QG_PAR[0], timeline=timeline,
        records=records,
    )  # fmt: skip


_SCRATCH = {}


def scratch_for(key, device):
    """A zeroed scratch a build (its flags never meet another layout's words)."""
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

    if key.tokens not in _SCRATCH:
        _SCRATCH[key.tokens] = PeerBuffer(mf.scratch_bytes(key), None, 0, 1, device).bytes
    return _SCRATCH[key.tokens]


class Peers:
    """The eight DCP ranks' peer buffers in one uncached allocation (the q
    all-gather's QG copies) and the table a launch takes."""

    def __init__(self, device):
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_combine as mc
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

        self.mf, self.mc = mf, mc
        self.stride = mf.peer_end()
        self.buf = PeerBuffer(DCP * self.stride, None, 0, 1, device)
        self.buf.bytes.zero_()
        self.rearm()
        self.table = torch.tensor(
            [self.buf.local + o * self.stride for o in range(DCP)], dtype=torch.int64, device=device
        )

    def _copy(self, o):
        off = o * self.stride + self.mf.qg_offset() + QG_PAR[0] * self.mf.QG_COPY
        return self.buf.bytes[off : off + self.mf.QG_COPY]

    def q_of(self, o, s):
        """Rank o's gathered q of s rows: uint8 [s, 96, 576]."""
        mf = self.mf
        d = self._copy(o)[: mf.QG_DATA].view(DCP, mf.MAX_REQS, NHL, G, DQK)[:, : s // G]
        return d.permute(1, 3, 0, 2, 4).reshape(s, NHA, DQK).clone()

    def flags(self, o):
        mf = self.mf
        return self._copy(o)[mf.QG_DATA : mf.QG_DATA + DCP * mf.MAX_REQS * 8].view(torch.int32).view(DCP, mf.MAX_REQS, 2)

    def a2a(self, o, s):
        """Rank o's all-to-all cells of an s-row step, int32 [src][row][head][cell][2]."""
        mc = self.mc
        off = o * self.stride + mc.peer_offset() + QG_PAR[0] * mc.a2a_copy_bytes()
        return self.buf.bytes[off : off + mc.a2a_cells(s) * 8].view(torch.int32).view(DCP, s, NHL, mc.REC_CELLS, 2)

    def a2a_stand_in(self, o, s, srcs):
        """Rank o's records from ``srcs`` as zero words (sources that run later)."""
        a = self.a2a(o, s)
        for src in srcs:
            a[src, ..., : self.mc.LSE_CELL + 1, :] = 0

    def rearm(self):
        for o in range(DCP):
            self.buf.bytes[o * self.stride + self.mc.peer_offset() : o * self.stride + self.mc.peer_end()].fill_(0xFF)

    def stand_in(self, o, tag):
        """Rank o's (source, request) flags as the launch tagged ``tag`` reads them."""
        f = self.flags(o)
        f[..., 0] = 1
        f[..., 1] = tag


_PEERS = {}


def peers_for(device):
    if "p" not in _PEERS:
        _PEERS["p"] = Peers(device)
    return _PEERS["p"]


def next_tag(epoch, layer):
    """The mailbox tag of the launch after ``epoch[0]``'s next bump."""
    ev = int(epoch[0].item()) + 1
    return (ev % (1 << 22)) * 256 + layer


_WRANK = {}


def rank_weights(w, o):
    """Rank o's launch weights: its 12 heads' q_b / W_UK shards."""
    key = (id(w), o)
    if key not in _WRANK:
        h0, h1 = NHL * o, NHL * (o + 1)
        # W_UK as SGLang holds its shard: [12, 128, 512], n innermost
        w_kc = w["w_kc"][h0:h1].transpose(1, 2).contiguous().transpose(1, 2)
        _WRANK[key] = dict(w, w_qb=w["w_qb"][h0 * QHD : h1 * QHD].contiguous(), w_kc=w_kc)
    return _WRANK[key]


def launch_one(w, st, b, epoch, layer, peers, *, timeline=False):
    """Rank b.r's launch (its peers' q already in its QG copy or arriving): its
    outputs, scratch hand-offs and pool."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

    s = getattr(b, "s", G)
    key = front_key(w, s, timeline)
    dev = b.pool.device
    scratch = scratch_for(key, dev)
    core = torch.zeros(s, NHL * VD, dtype=BF, device=dev)
    send = torch.zeros(DCP, s, NHL, mf.REC, dtype=BF, device=dev)
    tl = torch.zeros(256, mf.TL_POINTS, dtype=torch.int64, device=dev) if timeline else None
    peers.stand_in(b.r, next_tag(epoch, layer))
    epoch[0].add_(1)
    mf.mla_front_launch(
        key, prefix=st["prefix"], blocks=st["blocks"], w=rank_weights(w, b.r), kv=b.pool,
        kv_idx=b.kv_idx, kv_ptr=b.kv_ptr, g_kv_ptr=b.g_kv_ptr, loc=b.loc,
        core=core, send=send, scratch=scratch, peers=peers.table, rank=b.r, layer=layer,
        epoch=epoch, tl=tl,
    )  # fmt: skip
    torch.cuda.synchronize()
    v = lambda n, dt, shp: mf.scratch_view(key, scratch, n, dt, shp).clone()  # noqa: E731
    reqs = getattr(b, "reqs", [b])
    return {
        "key": key, "gate": v("g", BF, (s, NHL * VD)), "send": send, "core": core, "tl": tl,
        "x": v("x", BF, (s, HIDDEN)),
        "lat": v("lat", BF, (s, LATW)),
        "qb": v("qb", BF, (s, NHA, QHD)),
        "q8": peers.q_of(b.r, s),
        "pv": v("pv", torch.float16, (mf.MAX_SPLITS, s, NHA, KV)),
        "ml": v("ml", torch.float32, (mf.MAX_SPLITS, s, NHA, 2)),
        "splits": [mf.splits_of(rk.p) for rk in reqs],
    }  # fmt: skip


def run_ranks(w, st, bs, epoch, layer, *, timeline_rank=None):
    """The eight ranks' launches (``bs``: each rank's share) one after another,
    three times: the first pass lands every rank's q in every QG copy (rank o's
    combine reads zeros for the later ranks' records); the second pushes every
    rank's real records (its combines still read some of the first's); the
    third's outputs are a DCP8 step's, every combine reading real records."""
    peers = peers_for(bs[0].pool.device)
    s = getattr(bs[0], "s", G)
    for o in range(DCP):
        peers.a2a_stand_in(o, s, range(o + 1, DCP))
    for _ in range(2):
        for b in bs:
            launch_one(w, st, b, epoch, layer, peers)
    out = [launch_one(w, st, b, epoch, layer, peers, timeline=b.r == timeline_rank) for b in bs]
    peers.rearm()
    return out


def front_closure(w, st, b, epoch, layer):
    """Rank b.r's launch as a graph can replay it: the epoch bump, the other
    ranks' flags stood in with the launch's tag (their q: what the last
    ``run_ranks`` left in its QG copy), the launch."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

    peers = peers_for(b.pool.device)
    s = getattr(b, "s", G)
    key = front_key(w, s, records=False)
    scratch = scratch_for(key, b.pool.device)
    core = torch.zeros(s, NHL * VD, dtype=BF, device=b.pool.device)
    f = peers.flags(b.r)
    wr = rank_weights(w, b.r)
    # the other ranks' records stand in as zero words, set before each launch
    own = peers.a2a(b.r, s)

    def front():
        epoch[0].add_(1)
        f[..., 0].fill_(1)
        f[..., 1].copy_(((epoch[0] % (1 << 22)) * 256 + layer).expand(f.shape[:2]))
        own[: b.r].zero_()
        own[b.r + 1 :].zero_()
        mf.mla_front_launch(
            key, prefix=st["prefix"], blocks=st["blocks"], w=wr, kv=b.pool,
            kv_idx=b.kv_idx, kv_ptr=b.kv_ptr, g_kv_ptr=b.g_kv_ptr, loc=b.loc,
            core=core, scratch=scratch, peers=peers.table, rank=b.r, layer=layer,
            epoch=epoch,
        )  # fmt: skip

    return front


def run_batch(w, st, b, epoch, layer, *, timeline=False):
    """The front on batch ``b`` (every rank's share run, rank b.r's outputs)."""
    bs = [b.at_rank(o) for o in range(DCP)]
    return run_ranks(w, st, bs, epoch, layer, timeline_rank=b.r if timeline else None)[b.r]


def request_view(fr, b, j):
    """Request j's outputs out of a batch run, as bit patterns."""
    sl = slice(G * j, G * j + G)
    n = fr["splits"][j]
    u8 = lambda t: t.contiguous().view(torch.uint8)  # noqa: E731
    out = {
        "x": u8(fr["x"][sl]), "lat": u8(fr["lat"][sl]),
        "qb": u8(fr["qb"][sl][:, NHL * b.r : NHL * (b.r + 1)]),
        "q8": u8(fr["q8"][sl]), "gate": u8(fr["gate"][sl]), "send": u8(fr["send"][:, sl]),
        "core": u8(fr["core"][sl]),
        "pv": u8(fr["pv"][:n, sl]), "ml": u8(fr["ml"][:n, sl]),
    }  # fmt: skip
    row = b.own_row(j)
    out["rec"] = u8(b.pool[row]) if row is not None else None
    return out


INV_OTHERS = (10000, 3, 250000, 0, 37000, 7, 180000)  # long ones and P < 8
# a width's batch: request ids in batch order (request 0 at the context under test)
INV_BATCHES = {12: (1, 0), 24: (2, 0, 3, 1), 48: (1, 2, 3, 4, 5, 6, 7, 0)}


def invariance(w, args, device, epoch):
    """Each request alone against the same request in wider batches, bit for
    bit, on every rank in ``--rank``; the widest batch twice."""
    fails = []
    ctx_of = lambda i, p: p if i == 0 else INV_OTHERS[i - 1]  # noqa: E731
    bank = make_keys(w, max([*args.ctx, *INV_OTHERS]) + 8 * 997, args.seed, device, args.temp[0])
    for p in args.ctx:
        reqs = {i: (ctx_of(i, p), bank[997 * i : 997 * i + ctx_of(i, p)], args.seed + 13 * i) for i in range(8)}
        alone = {}
        for i in range(8):
            bs = [Batch([reqs[i]], o, device) for o in range(DCP)]
            frs = run_ranks(w, batch_stream([i], args.seed, device), bs, epoch, args.layer)
            for r in args.rank:
                alone[(i, r)] = request_view(frs[r], bs[r], 0)
        for s_, ids in INV_BATCHES.items():
            if s_ > G * args.max_reqs:
                continue
            st = batch_stream(ids, args.seed, device)
            bs = [Batch([reqs[i] for i in ids], o, device) for o in range(DCP)]
            frs = run_ranks(w, st, bs, epoch, args.layer)
            frs2 = None
            if s_ == max(INV_BATCHES):
                bs2 = [Batch([reqs[i] for i in ids], o, device) for o in range(DCP)]
                frs2 = run_ranks(w, st, bs2, epoch, args.layer)
            for r in args.rank:
                fr, b = frs[r], bs[r]
                bad = {}
                for j, i in enumerate(ids):
                    got = request_view(fr, b, j)
                    diff = [
                        k for k in got
                        if (got[k] is None) != (alone[(i, r)][k] is None)
                        or (got[k] is not None and not torch.equal(got[k], alone[(i, r)][k]))
                    ]  # fmt: skip
                    if diff:
                        bad[f"req{i}@{j} P={ctx_of(i, p)}"] = diff
                again = ""
                if frs2 is not None:
                    fr2 = frs2[r]
                    names = ("x", "lat", "q8", "pv", "ml", "gate", "send")
                    dd = [n for n in names if not torch.equal(fr2[n].view(torch.uint8), fr[n].view(torch.uint8))]
                    dd += [] if torch.equal(bs2[r].pool.view(torch.uint8), b.pool.view(torch.uint8)) else ["pool"]
                    again = f"; run to run bitwise {not dd} {dd}"
                    if dd:
                        fails.append(f"P={p} rank={r} S={s_} run-to-run")
                tag = f"P={p} rank={r} S={s_} order {ids} splits {fr['splits']}"
                print(f"INV {tag}: {len(ids) - len(bad)}/{len(ids)} requests bitwise equal to alone"
                      + (f" BAD {bad}" if bad else "") + again, flush=True)
                if bad:
                    fails.append(tag)
    return fails


def time_wide(w, args, device, epoch):
    """The front at 4 and 8 requests (every request at one context, and mixed),
    CUDA-graph timed, then its phase stamps."""
    cells = []
    for nr in (4, 8):
        if nr > args.max_reqs:
            continue
        for p in args.ctx:
            cells.append((f"{nr}x{p}", [p] * nr))
    if args.max_reqs >= 8:
        cells.append(("mixed8", [10000, 250000, 3000, 100000, 37000, 180000, 64, 100000]))
    need = max(max(c) for _, c in cells) + 8 * 997
    bank = make_keys(w, need, args.seed, device, args.temp[0])
    for name, ctxs in cells:
        for r in args.rank:
            b = Batch([(p, bank[997 * i : 997 * i + p], args.seed + 13 * i) for i, p in enumerate(ctxs)], r, device)
            st = batch_stream(range(len(ctxs)), args.seed, device)
            from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

            run_batch(w, st, b, epoch, args.layer)  # every rank's q in the QG copies
            front = front_closure(w, st, b, epoch, args.layer)
            t = graph_time(front, args.sets, args.reps, args.warmup)
            print(f"TIME {name} rank={r} S={b.s} splits={[mf.splits_of(p) for p in ctxs]}: front {t[0]:.1f} us "
                  f"[{t[1]:.1f}, {t[2]:.1f}] (sets {args.sets}, reps {args.reps}, warm-up {args.warmup})", flush=True)
            fr = run_batch(w, st, b, epoch, args.layer, timeline=True)
            print_stamps(fr["tl"], f"{name} rank={r}")


# ---------------------------------------------------------------- the front


def merged(fr):
    """The front's attention output (bf16 [S, 96, 512]) and natural-log LSE from
    its split partials, merged in fp32 as the W_UV tasks merge them."""
    m, l = fr["ml"][..., 0], fr["ml"][..., 1]
    mx = m.max(0).values
    wgt = torch.exp2(m - mx) * l
    lsum = wgt.sum(0)
    o = (wgt.unsqueeze(-1) * fr["pv"].float()).sum(0)
    o = torch.where(lsum.unsqueeze(-1) > 0, o / lsum.clamp_min(1e-30).unsqueeze(-1), 0)
    lse = torch.where(lsum > 0, (mx + torch.log2(lsum.clamp_min(1e-30))) * math.log(2), -1e30)
    return o.to(BF), lse


# ---------------------------------------------------------------- SGLang's ops


def _linear(x, weight):
    """SGLang's UnquantizedLinearMethod on a bf16 weight (fused_qkv_a, g_proj)."""
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    return UnquantizedLinearMethod().apply(layer, x)


def _qk_rmsnorm():
    from sglang.srt.models.deepseek_common.attention_forward_methods import (
        forward_mla_rocm,
    )

    return forward_mla_rocm.fused_qk_rmsnorm_bf16


def stock_attention(q, rk, scale, page_size):
    """SGLang's DCP8 target-verify attention of rank r (``_forward_verify_dcp``
    on the asm cprr kernel), its own methods run on a stand-in backend."""
    from sglang.srt.layers.attention import aiter_backend as ab

    dev = q.device
    be = types.SimpleNamespace()
    be.page_size = page_size
    be.kv_cache_dtype = F8
    be.dcp_world_size = DCP
    be.num_head = NHL
    be.attn_dp_enabled = False
    be.use_mla_auto_kv_splits = False
    be.max_split_per_batch = 32
    be.fix_max_split_per_batch = 32
    be.cprr_fast_mode = True
    be.cprr_intra_batch_mode = False
    be.cprr_kernel_heads = ab._asm_cprr_kernel_heads(NHA, G)
    be.mla_kernel_num_head_padded = be.cprr_kernel_heads
    be._cprr_launch_logged = True
    be.input_dtype = BF
    be.pad_heads = types.MethodType(ab.AiterAttnBackend.pad_heads, be)
    be._forward_verify_asm_cprr = types.MethodType(ab.AiterAttnBackend._forward_verify_asm_cprr, be)
    be._kernel_num_kv_splits = lambda: 32
    be._resolve_fp8_kv_scale_float = lambda layer, k_descale: 1.0
    kv = rk.pool
    be.kv_layers = types.SimpleNamespace(
        read=lambda i: types.SimpleNamespace(key=lambda: kv)
    )
    lens = rk.kv_ptr[1:] - rk.kv_ptr[:1]
    qo_indptr = torch.tensor([0, G], dtype=torch.int32, device=dev)
    g_kv_indptr = torch.tensor([0, rk.p + G], dtype=torch.int32, device=dev)
    last = torch.ones(1, dtype=torch.int32, device=dev)
    meta = ab.AiterAttnBackend.make_mla_decode_meta_data_buffer(
        be, G, 1, metadata_fast_mode=True, metadata_intra_batch_mode=False,
        nhead_override=be.cprr_kernel_heads, max_split_per_batch=32,
    )  # fmt: skip
    ab.AiterAttnBackend.make_mla_meta_data(
        be, qo_indptr, rk.kv_ptr, last, *meta[:1], meta[2], meta[1], *meta[3:], G,
        fast_mode=True, max_split_per_batch=32, intra_batch_mode=False,
        is_cp_round_robin=True, nhead_override=be.cprr_kernel_heads,
    )  # fmt: skip
    from sglang.srt.layers.dcp.layout import get_dcp_chain_spec_lens

    local = get_dcp_chain_spec_lens(torch.tensor([rk.p + G], device=dev), G, DCP, rk.r)
    be.forward_metadata = types.SimpleNamespace(
        max_q_len=G, local_kv_lens=local, use_asm_cprr_verify=True,
        cprr_kernel_heads=be.cprr_kernel_heads, cprr_kv_indices=rk.kv_idx,
        cprr_kv_indptr=rk.kv_ptr, cprr_empty_rows=local.eq(0), qo_indptr=qo_indptr,
        kv_last_page_len=last, work_metadata=meta[0], work_indptr=meta[1],
        work_info_set=meta[2], reduce_indptr=meta[3], reduce_final_map=meta[4],
        reduce_partial_map=meta[5], global_kv_indptr=g_kv_indptr, num_kv_splits=32,
    )  # fmt: skip
    layer = types.SimpleNamespace(
        tp_q_head_num=NHA, qk_head_dim=DQK, v_head_dim=KV, scaling=scale, layer_id=0
    )
    par = types.SimpleNamespace(attn_dcp_rank=rk.r)
    old = ab.get_parallel
    ab.get_parallel = lambda: par
    try:
        out, lse = ab.AiterAttnBackend._forward_verify_dcp(be, q, None, layer, None)
    finally:
        ab.get_parallel = old
    return out, lse, local


def stock_rank(w, x, rk, page_size):
    """SGLang's rank-r ops from x to the a2a send records (ops 1-13 and 16)."""
    from sglang.kernels.ops.attention.dcp_kernels import dcp_pack_a2a_send
    from sglang.srt.layers.quantization.unquant_hip import a16w16_linear

    lat = _linear(x, w["w_qkv"])
    q, k_nope = _qk_rmsnorm()(
        lat[:, :QL], w["q_nw"], w["q_eps"], lat[:, QL : QL + KV], w["kv_nw"], w["kv_eps"]
    )
    qb = a16w16_linear(q, w["w_qb"]).view(G, NHA, QHD)
    q_nope_out = qb.new_empty(G, NHA, KV)
    torch.bmm(qb[..., :QN].transpose(0, 1), w["w_kc"], out=q_nope_out.transpose(0, 1))
    qcat = torch.cat([q_nope_out, qb[..., QN:]], dim=-1)
    q8 = qcat.view(G, -1).to(F8).view(G, NHA, DQK)
    rec = torch.cat([k_nope.to(F8), lat[:, QL + KV :].to(F8)], dim=-1)
    pool = rk.pool.clone()
    if rk.own < G:
        pool[int(rk.rows[rk.lr])] = rec[rk.own]
    rk2 = types.SimpleNamespace(**vars(rk))
    rk2.pool = pool
    out, lse, local = stock_attention(qcat, rk2, w["scale"], page_size)
    proj = out.new_empty(G, NHA, VD)
    torch.bmm(out.transpose(0, 1), w["w_vc"], out=proj.transpose(0, 1))
    send = torch.empty(DCP, G, NHL, VD + 2, dtype=BF, device=x.device)
    words = send.view(torch.float32)
    dcp_pack_a2a_send(proj, lse.contiguous(), send[..., :VD], words[..., VD // 2])
    gate = _linear(x, w["w_g"])
    return {
        "lat": lat, "q8": q8.view(torch.uint8), "qcat": qcat, "rec": rec,
        "k_nope": k_nope, "out": out, "lse": lse, "proj": proj, "send": send,
        "gate": gate, "pool": pool, "local": local,
    }  # fmt: skip


# ---------------------------------------------------------------- the reference


def ref_attention(q8, keys_r, own_rec, rk, scale):
    """fp32 attention of rank r's keys for the 6 rows: (out [S, 96, 512], lse)."""
    q = q8.view(F8).float()  # [S, 96, 576]
    k = keys_r.view(F8).float()  # [L, 576]
    if rk.own < G:
        k = torch.cat([k, own_rec.view(F8).float().unsqueeze(0)], 0)
    s = torch.einsum("thd,kd->thk", q, k) * scale
    nk = k.shape[0]
    j = torch.arange(G, device=q.device).view(G, 1, 1)
    kk = torch.arange(nk, device=q.device).view(1, 1, nk)
    ok = (kk < rk.lr) | ((kk == rk.lr) & (j >= rk.own))
    s = torch.where(ok, s, float("-inf"))
    lse = torch.logsumexp(s, -1)
    p = torch.where(ok, torch.exp(s - lse.unsqueeze(-1)), 0)
    out = torch.einsum("thk,kd->thd", p, k[:, :KV])
    empty = ~ok.any(-1)
    out = torch.where(empty.unsqueeze(-1), 0, out)
    lse = torch.where(empty, -1e30, lse)
    return out, lse


def diag_splits(fr, keys_r, own_rec, rk, scale):
    """Each split's LSE (from the front's (max, sum)) against fp32 over the
    split's blocks, per row: where an attention mismatch starts."""
    q = fr["q8"].view(F8).float()
    k = keys_r.view(F8).float()
    if rk.own < G:
        k = torch.cat([k, own_rec.view(F8).float().unsqueeze(0)], 0)
    n = fr["splits"]
    nbl = -(-(rk.lr + int(rk.own < G)) // 32)
    s_all = torch.einsum("thd,kd->thk", q, k) * scale
    nk = k.shape[0]
    j = torch.arange(G, device=q.device).view(G, 1, 1)
    kk = torch.arange(nk, device=q.device).view(1, 1, nk)
    ok = (kk < rk.lr) | ((kk == rk.lr) & (j >= rk.own))
    for sp in range(n):
        b0, b1 = sp * nbl // n, (sp + 1) * nbl // n
        sel = ok & (kk >= b0 * 32) & (kk < b1 * 32)
        ss = torch.where(sel, s_all, float("-inf"))
        lse = torch.logsumexp(ss, -1)
        m, l = fr["ml"][sp, ..., 0], fr["ml"][sp, ..., 1]
        mine = (m + torch.log2(l)) * math.log(2)
        mref = ss.max(-1).values * math.log2(math.e)
        def lse_of(c0, c1):
            sl = ok & (kk >= c0 * 32) & (kk < c1 * 32)
            return torch.logsumexp(torch.where(sl, s_all, float("-inf")), -1)

        first, last = lse_of(b0, b0 + 1), lse_of(b1 - 1, b1)
        if sp == 0:
            sl = ok & (kk >= b0 * 32) & (kk < (b0 + 1) * 32)
            pe = torch.where(sl, torch.exp2(s_all * math.log2(math.e) - m.unsqueeze(-1)), 0)[..., b0 * 32 : b0 * 32 + 32]
            pe = F.pad(pe, (0, 32 - pe.shape[-1]))
            tot = pe.sum(-1)
            grp = [pe[..., [4 * g + i for i in range(4)] + [16 + 4 * g + i for i in range(4)]].sum(-1) for g in range(4)]
            print(f"  block0 head0 row0: l {l[0, 0].item():.4f} true {tot[0, 0].item():.4f} groups "
                  f"{[round(x[0, 0].item(), 4) for x in grp]} keys {[round(x, 3) for x in pe[0, 0].tolist()]}", flush=True)
        print(f"  split {sp}: signed lse err mean {(mine - lse).mean().item():.3f}; vs first block "
              f"{(mine - first).abs().mean().item():.3f}; vs last block {(mine - last).abs().mean().item():.3f}; "
              f"l mean {l.mean().item():.3e}", flush=True)
        print(f"  split {sp} blocks {b0}..{b1}: lse err per row "
              f"{[round((mine[t] - lse[t]).abs().max().item(), 3) for t in range(G)]} "
              f"max err {[round((m[t] - mref[t]).abs().max().item(), 3) for t in range(G)]}", flush=True)


def emulate_kernel(q8, keys_r, own_rec, rk, scale, n, pscale, pdtype=F8, lq=False):
    """The front's attention in torch: per split, 32-key blocks, the online
    softmax with e4m3 P (times ``pscale``), f16 partials, the split merge."""
    q = q8.view(F8).float()
    k = keys_r.view(F8).float()
    if rk.own < G:
        k = torch.cat([k, own_rec.view(F8).float().unsqueeze(0)], 0)
    nk = k.shape[0]
    nbl = -(-nk // 32)
    k = F.pad(k, (0, 0, 0, nbl * 32 - nk))
    s_all = torch.einsum("thd,kd->thk", q, k) * scale * math.log2(math.e)
    j = torch.arange(G, device=q.device).view(G, 1, 1)
    kk = torch.arange(nbl * 32, device=q.device).view(1, 1, -1)
    ok = (kk < rk.lr) | ((kk == rk.lr) & (j >= rk.own))
    s_all = torch.where(ok, s_all, float("-inf"))
    parts = []
    for sp in range(n):
        b0, b1 = sp * nbl // n, (sp + 1) * nbl // n
        m = torch.full(s_all.shape[:2], -3.4e38, device=q.device)
        l = torch.zeros_like(m)
        acc = torch.zeros(G, NHA, KV, device=q.device)
        for b in range(b0, b1):
            sb = s_all[..., b * 32 : b * 32 + 32]
            mn = torch.maximum(m, sb.max(-1).values)
            a = torch.exp2(m - mn)
            p = torch.exp2(sb - mn.unsqueeze(-1))
            pq = (p * pscale).to(pdtype).float()
            l = l * a + (pq.sum(-1) / pscale if lq else p.sum(-1))
            acc = acc * a.unsqueeze(-1) + torch.einsum("thk,kd->thd", pq, k[b * 32 : b * 32 + 32, :KV])
            m = mn
        o = torch.where(l.unsqueeze(-1) > 0, acc / (l * pscale).clamp_min(1e-30).unsqueeze(-1), 0).half().float()
        parts.append((m, l, o))
    mx = torch.stack([x[0] for x in parts]).max(0).values
    wg = [torch.exp2(x[0] - mx) * x[1] for x in parts]
    ls = sum(wg)
    o = sum(w_.unsqueeze(-1) * x[2] for w_, x in zip(wg, parts)) / ls.unsqueeze(-1)
    lse = (mx + torch.log2(ls)) * math.log(2)
    return o.to(BF), lse


def ref_send(out, lse, w):
    proj = torch.einsum("thk,hkv->thv", out.float(), w["w_vc"].float())
    return proj, lse


# ---------------------------------------------------------------- checks


def check_cast(device):
    """The front's e4m3 conversion against torch's on every bf16 bit pattern."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr import range_constexpr
    from flydsl.expr.typing import Int64, T

    from sgl_amd_mono.common.ops import rsrc
    from sgl_amd_mono.kimi_k3.gfx950.attention.mla_front import e4m3x4

    @flyc.kernel(name="k3_check_e4m3", known_block_size=[256, 1, 1])
    def cast_kernel(src: Int64, dst: Int64):
        i = fx.block_idx.x * 256 + fx.thread_idx.x
        w = fx.Vector(bo.buffer_load(rsrc(src), i * 2, vec_width=2, dtype=T.i32))
        vals = []
        for h in range_constexpr(2):
            x = fx.Int32(w[h])
            vals += [(x << 16).bitcast(fx.Float32), (x & fx.Int32(-65536)).bitcast(fx.Float32)]
        bo.buffer_store(e4m3x4(vals), rsrc(dst), i)

    @flyc.jit
    def launch(src: Int64, dst: Int64, stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        cast_kernel(src, dst).launch(grid=(64,), block=(256,), stream=stream)

    bits = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16)
    src = bits.view(BF).contiguous()
    dst = torch.zeros(65536, dtype=torch.uint8, device=device)
    launch(src.data_ptr(), dst.data_ptr(), stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    ref = src.to(F8).view(torch.uint8)
    nan = torch.isnan(src.float())
    bad = (dst != ref) & ~(nan & torch.isnan(dst.view(F8).float()))
    return int(bad.sum()), src[bad][:8].float().tolist(), dst[bad][:8].tolist(), ref[bad][:8].tolist()


def check_frontier():
    """(L_r, owned row) as the front computes them against SGLang's DCP chain
    lengths, for P in 0 .. 63 and every rank."""
    from sglang.srt.layers.dcp.layout import get_dcp_chain_spec_lens

    bad = 0
    for p in range(64):
        for r in range(DCP):
            lr = p // DCP + int(r < p % DCP)
            own = (r - p) % DCP
            own = own if own < G else DCP
            mine = [lr + int(own <= j) for j in range(G)]
            ref = get_dcp_chain_spec_lens(torch.tensor([p + G]), G, DCP, r).tolist()
            bad += mine != ref
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[10000])
    ap.add_argument("--ranks", type=int, default=1, help="1: rank --rank; 8: emulate DCP8")
    ap.add_argument("--rank", type=int, nargs="+", default=[0])
    ap.add_argument("--copy", default=None, help="a one-rank real-weights copy dir")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, nargs="+", default=[1.0])
    ap.add_argument("--page-size", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--skip-units", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--time", action="store_true", help="time the front and the stock ops only")
    ap.add_argument("--sets", type=int, default=10)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--invariance", action="store_true", help="the batch-invariance suite")
    ap.add_argument("--time-wide", action="store_true", help="time the front at 4 and 8 requests")
    ap.add_argument("--max-reqs", type=int, default=8)
    ap.add_argument("--qg-par", type=int, default=0, choices=(0, 1), help="the QG copy the launches take")
    args = ap.parse_args()
    QG_PAR[0] = args.qg_par
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

    fails = []
    if not args.skip_units:
        nbad, xs, got, want = check_cast(device)
        print(f"cast: {65536 - nbad}/65536 bf16 values bit for bit" + (f" BAD {xs} {got} {want}" if nbad else ""))
        if nbad:
            fails.append("cast")
        fb = check_frontier()
        print(f"frontier: {64 * DCP - fb}/{64 * DCP} (P, rank) pairs equal")
        if fb:
            fails.append("frontier")
    w = layer_weights(args, device)
    if args.invariance or args.time_wide:
        epoch = new_epoch(device)
        if args.time_wide:
            time_wide(w, args, device, epoch)
        else:
            fails += invariance(w, args, device, epoch)
        print("ALL PASS" if not fails else f"FAILED: {fails}", flush=True)
        sys.exit(1 if fails else 0)
    st = stream_inputs(args.seed, device)
    xr = seam_x(st, w)
    key = mf.MlaFrontBuild(tokens=G, nblocks=NVB, scale=w["scale"])
    scratch = PeerBuffer(mf.scratch_bytes(key), None, 0, 1, device).bytes
    epoch = new_epoch(device)
    for p in args.ctx:
        if args.time:
            keys = make_keys(w, p, args.seed + p, device, args.temp[0])
            for r in args.rank:
                time_cells(w, st, keys, p, r, scratch, epoch, args.layer, args)
            continue
        for temp in args.temp:
            keys = make_keys(w, p, args.seed + p, device, temp)
            ranks = list(range(DCP)) if args.ranks == DCP else args.rank
            sends, stock_sends, gates, stock_gates, refs, emus = {}, {}, {}, {}, {}, {}
            cores = {}

            def all_ranks():
                """Every rank's launch (one request); each rank's outputs as the
                one-rank checks read them, q_b assembled from every rank's heads."""
                bs = [Batch([(p, keys, args.seed + p)], o, device) for o in range(DCP)]
                frs = run_ranks(w, st, bs, epoch, args.layer)
                qb = torch.cat([frs[o]["qb"][:, NHL * o : NHL * (o + 1)] for o in range(DCP)], 1)
                out = []
                for o in range(DCP):
                    fr = dict(frs[o])
                    n = fr["splits"][0]
                    fr.update(pv=fr["pv"][:n], ml=fr["ml"][:n], splits=n, qb=qb)
                    rk = bs[o].reqs[0]
                    rk.pool = bs[o].pool  # the launch wrote its new record there
                    out.append((rk, fr))
                return out

            runs = all_ranks()
            again = [all_ranks() for _ in range(args.repeat - 1)]
            for r in ranks:
                rk, fr = runs[r]
                sk = stock_rank(w, fr["x"], rk, args.page_size)
                # the send records in SGLang's a2a layout
                assert fr["send"].shape == sk["send"].shape, (fr["send"].shape, sk["send"].shape)
                tag = f"P={p} temp={temp} rank={r} splits={fr['splits']}"
                # the seam
                rx = rel(fr["x"], xr)
                # the KV record: SGLang's norm + casts of the front's own latent
                lat = fr["lat"]
                _, kn = _qk_rmsnorm()(
                    lat[:, :QL], w["q_nw"], w["q_eps"], lat[:, QL : QL + KV], w["kv_nw"], w["kv_eps"]
                )
                want_rec = torch.cat([kn.to(F8), lat[:, QL + KV :].to(F8)], -1).view(torch.uint8)
                kv_ok = True
                if rk.own < G:
                    got_rec = rk.pool[int(rk.rows[rk.lr])].view(torch.uint8)
                    kv_eq = int((got_rec == want_rec[rk.own]).sum())
                    kv_ok = kv_eq == DQK
                else:
                    kv_eq = -1
                # q: SGLang's cast of the front's bf16 q (its q_b output, then
                # the absorb of SGLang's bmm on it)
                qb = fr["qb"]
                qn = qb.new_empty(G, NHA, KV)
                torch.bmm(qb[..., :QN].transpose(0, 1), w["w_kc"], out=qn.transpose(0, 1))
                q_want = torch.cat([qn, qb[..., QN:]], -1).view(G, -1).to(F8).view(G, NHA, DQK).view(torch.uint8)
                q_rope_eq = bool(torch.equal(fr["q8"][..., KV:], q_want[..., KV:]))
                # both paths' e4m3 q against the cast of the fp32 q chain from the same x
                lat32 = fr["x"].float() @ w["w_qkv"].float().t()
                qa32 = rms(lat32[:, :QL].to(BF), w["q_nw"], w["q_eps"]).to(BF).float()
                qb32 = (qa32 @ w["w_qb"].float().t()).view(G, NHA, QHD)
                qn32 = torch.einsum("thn,hnk->thk", qb32[..., :QN], w["w_kc"].float())
                q32 = torch.cat([qn32, qb32[..., QN:]], -1).to(F8).view(torch.uint8)
                q_fid = ((fr["q8"] == q32).float().mean().item(), (sk["q8"] == q32).float().mean().item())
                q_nope_frac = (fr["q8"][..., :KV] == q_want[..., :KV]).float().mean().item()
                q_ulp = ((fr["q8"][..., :KV].view(F8).float() - q_want[..., :KV].view(F8).float()).abs()
                         / q_want[..., :KV].view(F8).float().abs().clamp_min(2**-9)).max().item()
                # attention partials against fp32 of each path's own fp8 q and KV
                keys_r = keys[rk.r :: DCP][: rk.lr]
                own_rec = rk.pool[int(rk.rows[rk.lr])] if rk.own < G else None
                o_f, lse_f = merged(fr)
                ref_f, rlse_f = ref_attention(fr["q8"], keys_r, own_rec, rk, w["scale"])
                ref_s, rlse_s = ref_attention(sk["q8"], keys_r, sk["rec"][rk.own] if rk.own < G else None, rk, w["scale"])
                e_attn = rel(o_f, ref_f)
                if args.diag:
                    emu, _ = emulate_kernel(fr["q8"], keys_r, own_rec, rk, w["scale"], fr["splits"], 256.0)
                    emu1, _ = emulate_kernel(fr["q8"], keys_r, own_rec, rk, w["scale"], fr["splits"], 1.0)
                    emus[r] = {
                        v: emulate_kernel(fr["q8"], keys_r, own_rec, rk, w["scale"], fr["splits"], *a_)
                        for v, a_ in {
                            "fp8x1": (1.0,), "fp8x256": (256.0,), "fp8x256_lq": (256.0, F8, True),
                            "fp8x1_lq": (1.0, F8, True), "bf16": (1.0, BF),
                        }.items()
                    }
                    sc = lambda a_, b_: ((a_.float() * b_.float()).sum() / (b_.float() ** 2).sum()).item()  # noqa: E731
                    print(f"  scale vs ref: front {sc(o_f, ref_f):.5f} stock {sc(sk['out'], ref_s):.5f} "
                          f"emu256 {sc(emu, ref_f):.5f} emu1 {sc(emu1, ref_f):.5f}; front vs emu256 {rel(o_f, emu):.3e}, "
                          f"emu256 vs ref {rel(emu, ref_f):.3e}, emu1 vs ref {rel(emu1, ref_f):.3e}", flush=True)
                if args.diag:
                    diag_splits(fr, keys_r, own_rec, rk, w["scale"])
                e_attn_st = rel(sk["out"], ref_s)
                e_lse = (lse_f - rlse_f).abs().max().item()
                e_lse_st = (sk["lse"] - rlse_s).abs().max().item()
                proj_f = fr["send"][..., :VD].permute(1, 0, 2, 3).reshape(G, NHA, VD)
                rproj_f, _ = ref_send(o_f, lse_f, w)
                rproj_s, _ = ref_send(sk["out"], sk["lse"], w)
                rproj_fr, _ = ref_send(ref_f, rlse_f, w)
                rproj_sr, _ = ref_send(ref_s, rlse_s, w)
                e_send = rel(proj_f, rproj_fr)
                e_send_st = rel(sk["proj"], rproj_sr)
                lse_send = fr["send"].view(torch.float32)[..., VD // 2].permute(1, 0, 2).reshape(G, NHA)
                lse_eq = bool(torch.equal(lse_send, lse_f)) or (lse_send - lse_f).abs().max().item() < 1e-5
                gref = fr["x"].float() @ w["w_g"].float().t()
                gate_rel = rel(fr["gate"], sk["gate"])
                gate_ref = (rel(fr["gate"], gref), rel(sk["gate"], gref))
                ok_attn = e_attn <= 1.25 * e_attn_st
                ok_send = e_send <= 1.25 * e_send_st
                print(
                    f"{tag}: x rel {rx:.2e} | kv rec {kv_eq}/{DQK} | q rope bitwise {q_rope_eq}, "
                    f"q nope equal {q_nope_frac:.5f} (max rel {q_ulp:.3f}), q = fp32 cast {q_fid[0]:.4f} (stock {q_fid[1]:.4f}) | attn rel {e_attn:.3e} "
                    f"(stock {e_attn_st:.3e}) {'ok' if ok_attn else 'FAIL'} | lse max {e_lse:.2e} "
                    f"(stock {e_lse_st:.2e}) | send rel {e_send:.3e} (stock {e_send_st:.3e}) "
                    f"{'ok' if ok_send else 'FAIL'} | send lse {lse_eq} | gate rel {gate_rel:.2e} (to fp32 {gate_ref[0]:.2e}, stock {gate_ref[1]:.2e})",
                    flush=True,
                )
                if not (kv_ok and q_rope_eq and ok_attn and ok_send and lse_eq):
                    fails.append(tag)
                # run to run
                for rep_ in again:
                    fr2 = rep_[r][1]
                    names = ("x", "lat", "qb", "q8", "pv", "ml", "gate", "send", "core")
                    # bit patterns: the send records' LSE words read as bf16 can be NaNs
                    diff = [n for n in names if not torch.equal(fr2[n].view(torch.uint8), fr[n].view(torch.uint8))]
                    same = not diff
                    if not same:
                        fails.append(tag + " run-to-run")
                    print(f"{tag}: run to run bitwise {same} {diff}", flush=True)
                sends[r], stock_sends[r] = fr["send"], sk["send"]
                gates[r], stock_gates[r] = fr["gate"], sk["gate"]
                cores[r] = fr["core"]
                refs[r] = (rproj_sr, rlse_s, gref, o_f, lse_f, sk["out"], sk["lse"], ref_s, proj_f, sk["proj"],
                           rproj_fr, rlse_f, ref_f)
            if args.ranks == DCP:
                fails += emulate(sends, stock_sends, gates, stock_gates, refs, f"P={p} temp={temp}", emus, cores)
    print("ALL PASS" if not fails else f"FAILED: {fails}", flush=True)
    sys.exit(1 if fails else 0)


def graph_time(fn, sets, reps, warmup):
    """Median and spread over ``sets`` of the mean µs of ``fn`` replayed ``reps``
    times in one CUDA graph, after ``warmup`` replays."""
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(warmup):
        g.replay()
    out = []
    for _ in range(sets):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1000.0 / reps)
    out.sort()
    return out[len(out) // 2], out[0], out[-1]


def time_cells(w, st, keys, p, r, scratch, epoch, layer, args):
    """The front (one launch, the step's epoch bump in the graph) against SGLang's
    rank-r ops 1-13 and 16 on the same inputs (metadata built outside the graph,
    as served), CUDA-graph timed; then the front's phase stamps."""
    from sglang.kernels.ops.attention.dcp_kernels import dcp_pack_a2a_send
    from sglang.srt.layers.quantization.unquant_hip import a16w16_linear

    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

    dev = scratch.device
    b = Batch([(p, keys, args.seed + p)], r, dev)
    rk = b.reqs[0]
    run_batch(w, st, b, epoch, layer)  # every rank's q in the QG copies
    t_front = graph_time(front_closure(w, st, b, epoch, layer), args.sets, args.reps, args.warmup)
    # stock: x from the seam (op A is not here), then ops 1-13 and 16
    x = seam_x(st, w)
    from sglang.srt.layers.attention import aiter_backend as ab

    rk2 = types.SimpleNamespace(**vars(rk))
    rk2.pool = rk.pool.clone()
    qn_out = torch.empty(G, NHA, KV, dtype=BF, device=dev)
    sendb = torch.empty(DCP, G, NHL, VD + 2, dtype=BF, device=dev)
    proj = torch.empty(G, NHA, VD, dtype=BF, device=dev)
    loc_row = int(rk.rows[rk.lr]) if rk.own < G else 0

    # the attention's metadata once, outside the graph (served: once a forward)
    cap = {}
    orig = ab.AiterAttnBackend._forward_verify_dcp

    def grab(be, q, kw, layer_, kd):
        cap["be"], cap["layer"] = be, layer_
        return orig(be, q, kw, layer_, kd)

    ab.AiterAttnBackend._forward_verify_dcp = grab
    try:
        stock_attention(torch.zeros(G, NHA, DQK, dtype=BF, device=dev), rk2, w["scale"], args.page_size)
    finally:
        ab.AiterAttnBackend._forward_verify_dcp = orig
    par = types.SimpleNamespace(attn_dcp_rank=rk.r)

    def stock():
        lat = _linear(x, w["w_qkv"])
        q, k_nope = _qk_rmsnorm()(
            lat[:, :QL], w["q_nw"], w["q_eps"], lat[:, QL : QL + KV], w["kv_nw"], w["kv_eps"]
        )
        qb = a16w16_linear(q, w["w_qb"]).view(G, NHA, QHD)
        torch.bmm(qb[..., :QN].transpose(0, 1), w["w_kc"], out=qn_out.transpose(0, 1))
        qcat = torch.cat([qn_out, qb[..., QN:]], dim=-1)
        if rk.own < G:
            rk2.pool[loc_row, :KV] = k_nope[rk.own].to(F8)
            rk2.pool[loc_row, KV:] = lat[rk.own, QL + KV :].to(F8)
        old = ab.get_parallel
        ab.get_parallel = lambda: par
        try:
            out, lse = orig(cap["be"], qcat, None, cap["layer"], None)
        finally:
            ab.get_parallel = old
        torch.bmm(out.transpose(0, 1), w["w_vc"], out=proj.transpose(0, 1))
        words = sendb.view(torch.float32)
        dcp_pack_a2a_send(proj, lse, sendb[..., :VD], words[..., VD // 2])
        _linear(x, w["w_g"])

    t_stock = graph_time(stock, args.sets, args.reps, args.warmup)
    print(
        f"TIME P={p} rank={r} splits={mf.splits_of(p)}: front {t_front[0]:.1f} us "
        f"[{t_front[1]:.1f}, {t_front[2]:.1f}]; stock ops 1-13+16 {t_stock[0]:.1f} us "
        f"[{t_stock[1]:.1f}, {t_stock[2]:.1f}] (sets {args.sets}, reps {args.reps}, warm-up {args.warmup})",
        flush=True,
    )
    # phase stamps: per point, the latest CTA past it, from the earliest start
    fr = run_batch(w, st, b, epoch, layer, timeline=True)
    print_stamps(fr["tl"], f"P={p} rank={r}")


STAMP_NAMES = ["start", "x seen", "stage 1", "q latent", "q_b", "absorb", "attn q seen", "attn done",
               "W_UV seen", "W_UV done", "blk start", "blk qk", "blk pv", "blk stored", "blk sm", "blk pv start",
               "blk bar1", "blk dma issued", "blk dma waited"]


def print_stamps(tl, tag):
    t0 = tl[:, 0].min()
    pts = []
    for k, n in enumerate(STAMP_NAMES):
        col = tl[:, k]
        live = col > 0
        if live.any():
            us = ((col[live] - t0).double() / 100.0).sort().values
            pts.append(f"{n} {us[len(us) // 2].item():.1f}/{us[-1].item():.1f}")
    print(f"STAMPS {tag} (us from launch, median/latest CTA): " + ", ".join(pts), flush=True)
    groups = {"0-27": (0, 28), "28-55": (28, 56), "56-127": (56, 128), "128-191": (128, 192), "192-255": (192, 256)}
    for gname, (a_, b_) in groups.items():
        cols = []
        for k, n in enumerate(STAMP_NAMES[:10]):
            col = tl[a_:b_, k]
            live = col > 0
            if live.any():
                cols.append(f"{n} {((col[live].max() - t0).double() / 100.0).item():.1f}")
        print(f"  CTAs {gname} latest: " + ", ".join(cols), flush=True)
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf

    steps = tl[:, mf.TL_STEPS].double()
    live = steps > 0
    if live.any():
        # 100 MHz ticks a step, the median over the CTAs that ran attention steps
        per = lambda k: (tl[live, k].double() / steps[live] / 100.0).median().item()  # noqa: E731
        ph = ["qk", "vt", "pv", "dma"]
        cells = [f"w{w} " + "/".join(f"{per(mf.TL_ROLE + mf.TL_PHASES * w + k):.2f}" for k in range(mf.TL_PHASES)) for w in range(8)]
        print(f"  PHASES {tag} (us a step: " + " / ".join(ph) + f"; steps a CTA median {steps[live].median().item():.0f}): "
              + ", ".join(cells), flush=True)


def emulate(sends, stock_sends, gates, stock_gates, refs, tag, emus=None, cores=None):
    """SGLang's a2a (a copy), LSE combine and output gate on the eight ranks'
    records: every owner's core, front and stock, against the fp32 combine of
    the ranks' fp32 partials gated by the fp32 g_proj."""
    from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
    from sglang.kernels.ops.elementwise import mla_output_gate

    def core(snd, gts):
        outs = []
        for o in range(DCP):
            recv = torch.stack([snd[src][o] for src in range(DCP)])  # [8, S, 12, 130]
            comb, _ = dcp_lse_combine_triton(
                recv[..., :VD].contiguous(), recv.view(torch.float32)[..., VD // 2].contiguous(),
                is_lse_base_on_e=True,
            )
            x = comb.reshape(G, NHL * VD).contiguous()
            outs.append(mla_output_gate.kimi_k3_mla_output_gate(x, gts[o]))
        return torch.stack(outs)

    def ref_core(pi, li):
        """fp32 combine of the ranks' fp32 projected partials (``pi``) and LSEs
        (``li``), gated by the fp32 g_proj."""
        outs = []
        for o in range(DCP):
            hs = slice(o * NHL, (o + 1) * NHL)
            pr = torch.stack([refs[r][pi][:, hs] for r in range(DCP)])  # [8, S, 12, 128]
            ls = torch.stack([refs[r][li][:, hs] for r in range(DCP)]).float()  # [8, S, 12]
            m = ls.max(0).values
            wg = torch.exp(ls - m)
            x = (wg.unsqueeze(-1) * pr).sum(0) / wg.sum(0).unsqueeze(-1)
            outs.append(x.reshape(G, NHL * VD) * torch.sigmoid(refs[o][2]))
        return torch.stack(outs)

    # each path against the fp32 layer of its own e4m3 q and KV
    c_f, c_s = core(sends, gates), core(stock_sends, stock_gates)
    c_rf, c_rs = ref_core(10, 11), ref_core(0, 1)
    ones = {o: torch.full_like(gates[o], 30.0) for o in range(DCP)}
    xf, xs = core(sends, ones), core(stock_sends, ones)
    mixed = {r: torch.cat([sends[r][..., :VD], stock_sends[r][..., VD:]], -1) for r in range(DCP)}
    xm = core(mixed, ones)
    def comb(outs, lses):
        o = torch.stack([x.float() for x in outs])
        ls = torch.stack([x.float() for x in lses])
        m = ls.max(0).values
        wg = torch.exp(ls - m)
        return (wg.unsqueeze(-1) * o).sum(0) / wg.sum(0).unsqueeze(-1)

    R = [refs[r] for r in range(DCP)]
    a_ref = comb([x[7] for x in R], [x[1] for x in R])
    a_reff = comb([x[12] for x in R], [x[11] for x in R])
    a_f = comb([x[3] for x in R], [x[4] for x in R])
    a_s = comb([x[5] for x in R], [x[6] for x in R])
    p_ref = comb([x[10] for x in R], [x[11] for x in R])
    p_refs = comb([x[0] for x in R], [x[1] for x in R])
    p_f = comb([x[8] for x in R], [x[4] for x in R])
    p_s = comb([x[9] for x in R], [x[6] for x in R])
    if emus:
        for v in emus[0]:
            a_v = comb([emus[r][v][0] for r in range(DCP)], [emus[r][v][1] for r in range(DCP)])
            print(f"{tag}: emulated {v}: combined rel {rel(a_v, a_reff):.3e}, rank-0 rel {rel(emus[0][v][0], R[0][12]):.3e}", flush=True)
    print(f"{tag}: attention-level combine rel to fp32 of its own q: front {rel(a_f, a_reff):.3e} stock {rel(a_s, a_ref):.3e}; "
          f"the two fp32 references apart {rel(a_reff, a_ref):.3e}; "
          f"projected: front {rel(p_f, p_ref):.3e} stock {rel(p_s, p_refs):.3e}", flush=True)
    print(f"{tag}: pre-gate combine front vs stock {rel(xf, xs):.3e}; front values + stock LSE vs stock {rel(xm, xs):.3e}; "
          f"gated with front gates vs stock gates (stock sends) {rel(core(stock_sends, gates), c_s):.3e}", flush=True)
    e_f, e_s = rel(c_f, c_rf), rel(c_s, c_rs)
    again = torch.equal(c_f.view(torch.int16), core(sends, gates).view(torch.int16))
    ok = e_f <= 1.25 * e_s and again
    print(
        f"{tag}: core (8 ranks) rel to fp32 {e_f:.3e} (stock {e_s:.3e}) "
        f"{'ok' if ok else 'FAIL'}; front vs stock {rel(c_f, c_s):.3e}; combine repeat bitwise {again}",
        flush=True,
    )
    fails = [] if ok else [tag + " core"]
    if cores is not None:
        mine = torch.stack([cores[o] for o in range(DCP)])
        neq = (mine.view(torch.int16) != c_f.view(torch.int16)).sum().item()
        e_m = rel(mine, c_rf)
        ok_m = neq == 0 and e_m <= 1.25 * e_s
        print(
            f"{tag}: in-launch core: {mine.numel() - neq}/{mine.numel()} equal to SGLang's combine + gate "
            f"of the same records; rel to fp32 {e_m:.3e} (stock {e_s:.3e}) {'ok' if ok_m else 'FAIL'}",
            flush=True,
        )
        if not ok_m:
            fails.append(tag + " in-launch core")
    return fails


if __name__ == "__main__":
    main()
