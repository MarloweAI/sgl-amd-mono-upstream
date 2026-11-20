# SPDX-License-Identifier: Apache-2.0
"""K2 ``layer_post``: one Kimi-K3 layer from the attention's core output (the
o_proj input) to the layer's MoE output, in one launch -- K2a's o_proj, its
all-reduce and the MLP AttnRes seam, then K2b's latent MoE (``moe``).

As ATOM's ``layer_post``, the two halves share the launch so that the MoE's
static weights stream while the AttnRes chain (partials -> reduce -> slice ->
gate -> mix) is still settling: the CTAs that will run the router, latent and
shared gate/up GEMVs issue their weights before they first wait.

    CTAs 0 .. 55     K2a's reduce + AttnRes (slice, gate, mix -> x rows + XRDY);
                     then route (0 .. S-1) and the shared SiTU (8 .. 8+S-1)
    every CTA        o_proj (448 x 16 rows: one task a CTA below 64, two
                     above) -> partials to every rank
    CTAs 56 .. 151   the shared gate/up (96 tasks, weights right after o_proj)
    CTAs 160 .. 243  router parts (28) and latent parts (56): weights first
    every CTA        ug; down (0 .. 223) or the shared down (224 .. 255);
                     lnorm (56 .. 83), up (0 .. 55), reduce (200 .. 255)
"""

import functools
from typing import Any, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import (
    ld_i32,
    load_ptr64,
    peer_bases,
    uniform,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    kernel_sources,
    key_tuple,
    pair_layout,
)
from sgl_amd_mono.common.sync import Mailbox, preg, shift, sreg
from sgl_amd_mono.kimi_k3 import CORE, K2_MAX_ROWS, MAX_BLOCKS, MAX_ROWS
from sgl_amd_mono.kimi_k3.gfx950.stages import gemv as gemv
from sgl_amd_mono.kimi_k3.gfx950.stages import moe as k2b
from sgl_amd_mono.kimi_k3.gfx950.stages import oproj as k2a
from sgl_amd_mono.kimi_k3.gfx950.stages.dims import COLS, HIDDEN, NA
from sgl_amd_mono.kimi_k3.gfx950.stages import seam as tail_seam
from sgl_amd_mono.kimi_k3.gfx950.stages.seam import (
    slice_loads,
    stage_gate,
    stage_mix,
    stage_slice,
)
from sgl_amd_mono.kimi_k3.gfx950.stamps import (
    stamp,
    stamp_begin,
    stamp_flush,
)
from sgl_amd_mono.kimi_k3.gfx950.tag import step_tag

SOURCES = kernel_sources(__name__)

O0 = NA  # the CTAs past AttnRes's
OTASKS = k2a.OTASKS  # 448
# o_proj on every CTA: task b on CTA b, then task b + O_SECOND on CTAs O_TWO0 ..
O_SECOND = OTASKS - BLOCKS  # 192
O_TWO0 = BLOCKS - O_SECOND  # 64
E0 = 160  # router / latent CTAs (their weights in flight under their o_proj)
H0 = 8  # the shared SiTU's CTAs (after their AttnRes work)
SD0 = k2b.DOWN_TASKS  # 224: the shared down's CTAs
# timeline points: 0 start, 1 o_proj done / AR reduced, 2 early done / slice,
# 3 ug, 4 down/sdown, 5 lnorm, 6 up, 7 reduce, 8 gate, 9 mix (x published),
# 10 route, 11 route table, 12 latent quant, 13 route starts
TL_POINTS = 30  # + 20-24 router: x / LDS / GEMV / parts / pushed; + 14 route logits in, 15 INTER loaded, 16 down done, 17-19 route keys / select / rank
TL_OPROJ = (
    25  # o_proj CTAs: the last task's GEMV done, before its AR pushes (point 1: pushed)
)
TL_AR_WAIT = 26  # AttnRes CTAs: the AR poll starts (point 1: reduced)
# the tail seam (``K2Build.tail``): reduce CTAs' TPART pushed; row CTAs' gate
# (TPART in), x stored
TL_TAIL_PART = 27
TL_TAIL_GATE = 28
TL_TAIL_X = 29

assert E0 + k2b.R_TASKS + k2b.L_TASKS <= BLOCKS and k2b.G_TASKS <= E0 - O0

REGIONS = (
    ("part", "wgt", "msq", "xrdy", "ar")
    + k2b.REGIONS
    + (
        "ars",
        "latrs",
        "finals",
        "xt",
    )
)

# K2 options (``K2Build.opt``), each bit for bit the launch without it:
OPT_SENT = 1  # o_proj's all-reduce through a sentinel region (sync.put_sent)
OPT_PRELOAD = 2  # the seam's block / prefix / norm loads before that poll
OPT_TAG_X = 8  # x rows also as tagged pairs to the router / latent GEMVs: no XRDY
OPT_SENT_LATR = 16  # the latent all-reduce through a sentinel region
OPT_SENT_FINAL = 32  # the final all-reduce through a sentinel region
OPT_SDOWN_FIRST = 64  # the shared-down tasks up waits on before the others
OPT_RLOG_PARTS = 256  # every router part pushes its partial: no part-0 sum hop
OPT_BATCH = 512  # the route's, down's and up's polls in one batch each
OPT_SENTS = OPT_SENT | OPT_SENT_LATR | OPT_SENT_FINAL
K2_OPT = (
    OPT_SENTS | OPT_PRELOAD | OPT_TAG_X | OPT_SDOWN_FIRST | OPT_RLOG_PARTS | OPT_BATCH
)
NARROW_ROWS = (
    K2_MAX_ROWS  # this launch's widest step: the sentinel and tagged rows' size
)

_STREAM = fx.Stream(None)


class TailSeam(msgspec.Struct, frozen=True):
    """The next layer's attention-side AttnRes seam and input_layernorm, run in
    this launch's tail (``stages/seam.py``: ``tail_slice``, ``tail_row``) on the
    stream its final reduce writes: the next layer's x, for its K1 (the wide
    form) with no front. Its bank blocks are this launch's ``nblocks``; the next layer's
    weights and x come in the ``tail`` argument (``TAIL_ARGS``)."""

    eps: float = 1e-5  # the next layer's self_attention_res_norm
    out_eps: float = 1e-5  # its input_layernorm
    tiles: int = 1  # x in ``kda_wide.pack_x``'s layout over this many tiles


TAIL_ARGS = tail_seam.TAIL_ARGS  # ``tail``'s device table


class K2Build(msgspec.Struct, frozen=True):
    tokens: int
    nblocks: int  # the MLP AttnRes block sources
    # o_proj's input width a rank: the attention's TP-local output (KDA: 12
    # heads x 128). The rows are bf16, row-major, ``core.stride(0)`` apart.
    kin: int = CORE
    eps: float = 1e-5  # mlp_res_norm
    out_eps: float = 1e-5  # post_attention_layernorm
    ln_eps: float = 1e-5  # the latent RMSNorm
    reduce: bool = True  # the final reduce here (else the next layer's K1)
    reset: bool = False  # a block-write layer: the prefix restarts at the o_proj output
    # the reduce writes out + the updated prefix (the layer's residual stream), and
    # the seam reads the incoming prefix from ``prefix_in``: no copy in, no add out.
    # False (the kernel checks): out is the MoE output alone
    fold: bool = True
    timeline: bool = False
    # fault injection (tests only): the wait to stall and how many launches
    # to let through first. Compile-time, so a serving build pays nothing.
    fault_site: int = 0
    fault_after: int = 0
    xp: int = 0  # timing experiments (7: route without its selection; 2-5 stages.moe)
    guint: bool = False  # w13's AITER layout; see stages.moe.MoeBuild
    opt: int = 0  # the OPT_* bits
    # an MLA layer's K2 in the attention front's launch: each core chunk is read
    # once its (request, local head) box (CBOX) holds this launch's tag
    core_box: bool = False
    # o_proj's tasks rotated: CTA b takes CTA b + o_shift's (mod BLOCKS), so CTAs
    # BLOCKS - o_shift .. + O_TWO0 - 1 take one task, the rest two (a launch
    # whose CTAs there carry other work first)
    o_shift: int = 0
    # the next layer's attention seam in this launch's tail (``TailSeam``)
    tail: TailSeam | None = None


def peer_layout(s):
    """K2a's AR region, then K2b's: one parity's (name -> (byte offset, bytes))."""
    pairs = {"ar": k2a.ar_half_pairs(s), **k2b.peer_pairs(s)}
    return pair_layout(pairs.items())


def peer_bytes(s):
    lay = peer_layout(s)
    return 2 * max(o + n for o, n in lay.values())


def flags_offset() -> int:
    """Where every launch reads the guard flags in a rank's peer buffer.

    The runner allocates one peer buffer for all the widths. The flags follow
    the data, so they start after the widest data region of either launch.

    Both the runner and the kernel must use this function. A launch that
    computes the offset from its own width puts the flags inside the data
    region at every width below the widest one. The widths then disagree on
    where the flags are. A launch reads the bytes another width wrote, takes
    live data for an error word, and stops every rank.
    """
    return max(peer_bytes(K2_MAX_ROWS), k2b.peer_bytes(K2_MAX_ROWS))


def sent_layout() -> dict:
    """One layer parity's sentinel regions, name -> (byte offset, bytes): a word
    a (rank, bf16 pair, row), NARROW_ROWS rows."""
    sizes = {
        "ars": TP_ * NARROW_ROWS * HIDDEN // 2 * 4,
        "latrs": TP_ * NARROW_ROWS * k2b.LAT // 2 * 4,
        "finals": TP_ * NARROW_ROWS * HIDDEN // 2 * 4,
    }
    out, off = {}, 0
    for name, n in sizes.items():
        out[name] = (off, n)
        off += -(-n // 256) * 256
    return out


def sent_half_bytes() -> int:
    return max(o + n for o, n in sent_layout().values())


def sent_offset() -> int:
    """Where the sentinel regions start in a rank's peer buffer: past the guard
    flags, two layer parities. A word is pushed again two layers later, after
    a launch in between that its reader's rank finished first."""
    return -(-(flags_offset() + guard.FLAG_BYTES) // 4096) * 4096


def peer_alloc_bytes() -> int:
    """A rank's peer buffer: the parity halves, the guard flags, the sentinels."""
    return sent_offset() + 2 * sent_half_bytes()


# set by init_peer: a launch using the sentinel regions without it would read
# zeros as words that came
SENT_READY = False


def init_peer(buf: torch.Tensor) -> None:
    """Zero a rank's peer buffer (``peer_alloc_bytes``) and arm its sentinels."""
    global SENT_READY
    buf.zero_()
    buf[sent_offset() : peer_alloc_bytes()].fill_(0xFF)
    SENT_READY = True


@functools.cache
def scratch_regions(rows: int = K2_MAX_ROWS, front_rows: int = K2_MAX_ROWS) -> dict:
    """The runner's scratch, one region a launch family, back to back:
    name -> (byte offset, bytes), K2 and K1 built to ``rows`` rows, the DCP
    front to ``front_rows``.

    No region overlaps another at any built width. A launch's polls take a pair
    whose second word equals its tag, so a word another launch left where they
    poll could pass for one; and the one-launch KDA layer's attention runs
    beside its K2.

    - ``k2``: K2 (``scratch_layout``, core boxes included) at every width,
      and the MoE alone (the kernel checks');
    - ``xt``: K2's tagged x rows (OPT_TAG_X);
    - ``k1``: K1's own launches (``k1.scratch_bytes``);
    - ``mla_pre``: the MLA projection front;
    - ``front``: the DCP MLA front (``mla_front.scratch_layout`` inside it);
    - ``attn``: the one-launch KDA layer's attention, then its core;
    - ``tpart``, ``tstream``: K2's tail seam, its slices' partials and the
      stream rows it pushes (``tail_part_bytes``, ``tail_stream_bytes``).

    Every launch takes its offset from here, and the end sizes the runner's
    allocation (``scratch_alloc_bytes``)."""
    from sgl_amd_mono.kimi_k3.gfx950 import k1, kda_layer
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda_wide as kw
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_front as mf
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre

    widths = range(1, rows + 1)
    sizes = {
        "k2": max(
            k2b.scratch_bytes(k2b.MoeBuild(tokens=MAX_ROWS)),
            *(scratch_bytes(K2Build(tokens=s, nblocks=MAX_BLOCKS, core_box=True)) for s in widths),
        ),
        "xt": rows * HIDDEN // 2 * 8,
        "k1": k1.scratch_bytes(rows),
        "mla_pre": mla_pre.scratch_bytes(mla_pre.MlaPreBuild(tokens=MAX_ROWS, nblocks=MAX_BLOCKS)),
        "front": max(
            mf.scratch_bytes(mf.MlaFrontBuild(tokens=s, nblocks=MAX_BLOCKS, scale=1.0))
            for s in range(mf.G_ROWS, min(front_rows, mf.G_ROWS * mf.MAX_REQS) + 1, mf.G_ROWS)
        ),
        "attn": max(
            kda_layer.region_bytes(kw.KdaWideBuild(tokens=s, qlen=kw.WT, front=True, nblocks=MAX_BLOCKS))
            for s in range(k1.WIDE_FROM_REQS * kw.WT, rows + 1, kw.WT)
        ),
        # K2's tail seam (``TailSeam``): the slices' partials and the stream rows
        "tpart": tail_part_bytes(rows),
        "tstream": tail_stream_bytes(rows),
    }
    out, off = {}, 0
    for name, n in sizes.items():
        out[name] = (off, n)
        off = -(-(off + n) // 4096) * 4096
    return out


def tail_part_bytes(rows: int) -> int:
    """TPART at ``rows`` rows: a tagged pair a (slice, token, partial) at the
    deepest bank."""
    return NA * rows * tail_seam.front_nq(MAX_BLOCKS + 1) * 8


def tail_stream_bytes(rows: int) -> int:
    """TSTREAM at ``rows`` rows: a tagged pair a bf16 pair of a stream row."""
    return rows * HIDDEN // 2 * 8


def scratch_region(scratch, name: str):
    """Region ``name`` (``scratch_regions``) of the runner's uint8 ``scratch``."""
    off, n = scratch_regions()[name]
    return scratch[off : off + n]


def tag_offset() -> int:
    """Where the tagged x rows (OPT_TAG_X) start in the scratch."""
    return scratch_regions()["xt"][0]


def scratch_alloc_bytes() -> int:
    """The scratch every Kimi-K3 launch on a rank shares: every region."""
    return -(-max(o + n for o, n in scratch_regions().values()) // 4096) * 4096


def scratch_layout(key: K2Build) -> dict:
    s = key.tokens
    a = k2a.scratch_layout(k2a.AttnPostBuild(tokens=s, nblocks=key.nblocks))
    end = max(o + n for o, n in a.values())
    b = k2b.scratch_layout(k2b.MoeBuild(tokens=s))
    lay = dict(a)
    for name, (o, n) in b.items():
        lay[name] = ((end + 15) // 16 * 16 + o, n)
    end = max(o + n for o, n in lay.values())
    lay["xrow"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    if key.core_box:
        end = lay["xrow"][0] + lay["xrow"][1]
        lay["cbox"] = ((end + 255) // 256 * 256, k2a.core_boxes(s) * 8)
    return lay


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


@flyc.jit
def stage_oproj(c, cta):
    """o_proj task ``bid`` (the CTA rotated by ``o_shift``), and ``bid +
    O_SECOND`` from ``bid`` ``O_TWO0`` on: both tasks' weights in flight at once
    (one memory latency, not two), then the MFMAs, a token tile at a time."""
    bid = cta
    if const_expr(c["o_shift"]):
        bid = (cta + c["o_shift"]) % BLOCKS
    two = bid >= O_TWO0
    t1 = fx.min(bid + O_SECOND, OTASKS - 1)
    ops = [k2a.o_loads(c, bid), k2a.o_loads(c, t1)]
    tiles = gemv.tiles(c["S"])
    for ti in range_constexpr(len(tiles)):
        t0, sj = tiles[ti]
        k2a.load_core(c, t0, sj)
        k2a.stage_o(c, bid, ops[0], t0, sj)
        if two:
            k2a.stage_o(c, t1, ops[1], t0, sj)


_BUILDS: dict = {}

K2_ARGS = (
    "core",
    "core_ld",
    "w_o",
    "prefix",
    "prefix_in",
    "blocks",
    "blk_sm",
    "blk_sr",
    "ares_nw",
    "ares_qk",
    "in_nw",
    "w_gate",
    "bias",
    "w_ld",
    "w_sgu",
    "w_sd",
    "w13",
    "w13s",
    "w2",
    "w2s",
    "ln_w",
    "w_up",
    "out",
    "peers",
    "rank",
    "epoch",
    "layer",
    "tl",
    "tail",
)  # k2_body's ``args``: the K2 kernel's parameters


class K2Lds(NamedTuple):
    """K2's LDS in an enclosing kernel: the ``lds_types`` structs it allocated
    (``smem`` and ``route`` peeked, ``stage`` the union as allocated)."""

    smem: Any
    route: Any
    stage: Any


class K2LdsTypes(NamedTuple):
    smem: Any
    route: Any
    stage: Any


def lds_types(key: K2Build) -> K2LdsTypes:
    """The LDS structs K2's body takes at this build: allocate each from the
    enclosing kernel's allocator (``stage`` may sit in a wider union)."""
    s, ns = key.tokens, key.nblocks + 1
    wide = k2b.wide(s)
    mu, mun = k2b.max_u(s), k2b.max_units(s)

    if wide:

        @fx.struct
        class RouteLds:
            route: fx.Array[fx.Int32, s * 2 * k2b.TOPK, 16]
            flag: fx.Array[fx.Int32, k2b.E, 16]
            expert: fx.Array[fx.Int32, mun, 16]  # a unit's
            ustart: fx.Array[fx.Int32, mun, 16]
            toks: fx.Array[fx.Int32, s * k2b.TOPK, 16]
            scan: fx.Array[fx.Int32, WAVES, 16]

        @fx.struct
        class UgLds:
            latq: fx.Array[fx.Int32, gemv.ROWS * k2b.LQ_WORDS, 16]
            av: fx.Array[fx.Float32, THREADS, 16]

        @fx.struct
        class DownLds:  # two units' INTER / ISC tiles
            tile: fx.Array[fx.Int32, 2 * k2b.ROWS * k2b.TILE_W, 16]

        @fx.struct
        class TableLds:  # the route table's build
            mask: fx.Array[fx.Int32, 2 * k2b.E, 16]
            cnt: fx.Array[fx.Int32, mu, 16]
            toff: fx.Array[fx.Int32, mu, 16]
            sexp: fx.Array[fx.Int32, mu, 16]

        @fx.struct
        class AttnLds:  # packed bf16 pairs
            vals: fx.Array[fx.Int32, s * ns * COLS // 2, 16]
            dl: fx.Array[fx.Int32, s * COLS // 2, 16]

    else:

        @fx.struct
        class RouteLds:
            route: fx.Array[fx.Int32, s * 2 * k2b.TOPK, 16]
            flag: fx.Array[fx.Int32, k2b.E, 16]
            expert: fx.Array[fx.Int32, mu, 16]
            kof: fx.Array[fx.Int32, mu * s, 16]
            scan: fx.Array[fx.Int32, WAVES, 16]

        @fx.struct
        class UgLds:
            latq: fx.Array[fx.Int32, s * k2b.LAT // 4, 16]
            latsc: fx.Array[fx.Int32, s * k2b.LAT // 32, 16]
            av: fx.Array[fx.Float32, THREADS, 16]

        @fx.struct
        class DownLds:
            interq: fx.Array[fx.Int32, s * k2b.TOPK * k2b.IW, 16]
            intsc: fx.Array[fx.Int32, s * k2b.TOPK * k2b.UG_GROUPS, 16]
            contrib: fx.Array[fx.Float32, s * k2b.TOPK * k2b.ROWS, 16]

        @fx.struct
        class TableLds:
            mask: fx.Array[fx.Int32, 16, 16]

        @fx.struct
        class AttnLds:
            vals: fx.Array[fx.Float32, s * ns * COLS, 16]
            dl: fx.Array[fx.Float32, s * COLS, 16]

    @fx.struct
    class EarlyLds:
        xl: fx.Array[fx.Int32, k2b.xl_words(s), 16]

    @fx.struct
    class PartsLds:
        rlp: fx.Array[fx.Float32, k2b.R_PARTS * k2b.E, 16]

    @fx.struct
    class TailLds:  # the tail seam: the stream the reduce CTA reduced, bf16 pairs
        pre: fx.Array[fx.Int32, s * COLS // 2, 16]

    @fx.union
    class StageLds:
        early: EarlyLds
        attn: AttnLds
        ug: UgLds
        down: DownLds
        parts: PartsLds  # the route's router parts (OPT_RLOG_PARTS), after the mix
        table: TableLds
        tail: TailLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    return K2LdsTypes(smem=Smem, route=RouteLds, stage=StageLds)


@flyc.jit
def k2_body(key: K2Build, k2_lds: K2Lds, scratch_base, args: dict) -> None:
    """K2's whole launch body, traced into the calling kernel: LDS from
    ``k2_lds``, the scratch layout (``scratch_layout``) from ``scratch_base``;
    ``args`` the kernel parameters by ``K2_ARGS`` name. Every scratch region
    is an offset from ``scratch_base``: the layout (``scratch_layout``,
    ``scratch_bytes`` a width) and, on a narrow step under OPT_TAG_X, the
    tagged x rows at ``tag_offset`` (``scratch_alloc_bytes`` in all)."""
    core = args["core"]
    core_ld = args["core_ld"]
    w_o = args["w_o"]
    prefix = args["prefix"]
    prefix_in = args["prefix_in"]
    blocks = args["blocks"]
    blk_sm = args["blk_sm"]
    blk_sr = args["blk_sr"]
    ares_nw = args["ares_nw"]
    ares_qk = args["ares_qk"]
    in_nw = args["in_nw"]
    w_gate = args["w_gate"]
    bias = args["bias"]
    w_ld = args["w_ld"]
    w_sgu = args["w_sgu"]
    w_sd = args["w_sd"]
    w13 = args["w13"]
    w13s = args["w13s"]
    w2 = args["w2"]
    w2s = args["w2s"]
    ln_w = args["ln_w"]
    w_up = args["w_up"]
    out = args["out"]
    peers = args["peers"]
    rank = args["rank"]
    epoch = args["epoch"]
    layer = args["layer"]
    tl = args["tl"]
    tail = args["tail"]
    s, ns = key.tokens, key.nblocks + 1
    lay = scratch_layout(key)
    play = peer_layout(s)
    half = peer_bytes(s) // 2
    wide = k2b.wide(s)

    own = load_ptr64(peers, rank)
    # Every wait of this launch runs under the guard. A wait past the
    # deadline sets this GPU's error word. It also flags every peer, so
    # their waits return too. The launch then stops on all of them.
    with guard.scope(
        guard.Guard(
            epoch,
            own=own,
            peers=peers,
            rank=rank,
            npes=TP_,
            # Not key.tokens: the offset is the same at every width. See
            # flags_offset().
            flags_offset=flags_offset(),
            fault_site=key.fault_site or None,
            fault_after=key.fault_after,
        )
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds, rl, st = k2_lds.smem, k2_lds.route, k2_lds.stage
        el, al, ul, dl, pl, tb = (
            st.early.peek(),
            st.attn.peek(),
            st.ug.peek(),
            st.down.peek(),
            st.parts.peek(),
            st.table.peek(),
        )
        mb = Mailbox(step_tag(epoch, layer))
        par = uniform(ld_i32(epoch, 0)) % 2
        bases = peer_bases(peers, TP_)
        peer = {}
        for region in ("rlog", "lat", "latq", "latr", "final", "ar", "rlogp"):
            off = play[region][0]
            peer[region] = [
                shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
            ]
            peer[region + "_own"] = shift(preg(own, off, region), fx.Int64(par) * half)
        c = {
            "S": s,
            "o_shift": key.o_shift,
            "NS": ns,
            "NB": key.nblocks,
            "OK": key.kin,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "delta": True,
            "reset": key.reset,
            "fold": key.fold,
            # the top-16 by candidate rank; the latent quantized to MXFP8 by
            # its producers and pushed to every rank
            "route_v": 2,
            "latq_push": True,
            "gate_local": False,
            "debug": False,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "ln_eps": key.ln_eps,
            "xp": key.xp,
            "guint": key.guint,
            "lazy_inter": True,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "peer": peer,
            "ar_peer": peer["ar"],
            "ar_own": peer["ar_own"],
            "xl": el.xl.ptr,
            "vals": al.vals.ptr,
            "delta_lds": al.dl.ptr,
            "latq": ul.latq.ptr,
            "av": ul.av.ptr,
            "scan": rl.scan.ptr,
            "vals_bf16": wide,
            "rt": {
                "route": rl.route.ptr,
                "flag": rl.flag.ptr,
                "expert": rl.expert.ptr,
            },
            "xbuf": scratch_base + fx.Int64(lay["xrow"][0]),
            "xflags": NA,
            "tl_on": key.timeline,
            "tls": lds.tls.ptr,
            "tl_push": TL_OPROJ,
            "ln": scratch_base + fx.Int64(lay["ln"][0]),
            "shp": scratch_base + fx.Int64(lay["shp"][0]),
            "args": {
                "core": core,
                "core_ld": core_ld,
                "w_o": w_o,
                "prefix": prefix,
                "prefix_in": prefix_in,
                "blocks": blocks,
                "blk_sm": blk_sm,
                "blk_sr": blk_sr,
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
                "x_dbg": 0,
                "w_gate": w_gate,
                "bias": bias,
                "w_ld": w_ld,
                "w_sgu": w_sgu,
                "w_sd": w_sd,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "ln_w": ln_w,
                "w_up": w_up,
                "out": out,
            },
        }
        c["rlp"] = pl.rlp.ptr
        if wide:
            c["rt"]["ustart"] = rl.ustart.ptr
            c["rt"]["toks"] = rl.toks.ptr
            c["rt"]["mask"] = tb.mask.ptr
            c["rt"]["cnt"] = tb.cnt.ptr
            c["rt"]["toff"] = tb.toff.ptr
            c["rt"]["sexp"] = tb.sexp.ptr
            c["lqc"] = scratch_base + fx.Int64(lay["lqc"][0])
            c["contrib"] = scratch_base + fx.Int64(lay["contrib"][0])
            c["dtile"] = dl.tile.ptr
        else:
            c["contrib"] = dl.contrib.ptr
            c["rt"]["kof"] = rl.kof.ptr
            c["latsc"] = ul.latsc.ptr
            c["interq"] = dl.interq.ptr
            c["intsc"] = dl.intsc.ptr
        c["tag_x"] = bool(key.opt & OPT_TAG_X)
        c["rlog_parts"] = bool(key.opt & OPT_RLOG_PARTS)
        c["batch"] = bool(key.opt & OPT_BATCH)
        if const_expr(key.opt & OPT_TAG_X):
            c["xt"] = sreg(scratch_base, tag_offset(), "xt")
        if const_expr(key.core_box):
            c["core_box"] = sreg(scratch_base, lay["cbox"][0], "cbox")
        c["sent_ar"] = bool(key.opt & OPT_SENT)
        c["sent_latr"] = bool(key.opt & OPT_SENT_LATR)
        # the K1 that takes a deferred reduce reads tagged pairs
        c["sent_final"] = bool(key.opt & OPT_SENT_FINAL) and key.reduce
        if const_expr(key.opt & OPT_SENTS):
            # this layer parity's copy
            lpar = fx.Int64(layer % 2) * sent_half_bytes()
            for region, (roff, _) in sent_layout().items():
                off = sent_offset() + roff
                c[region + "_peer"] = [shift(preg(b, off, region), lpar) for b in bases]
                c[region + "_own"] = shift(preg(own, off, region), lpar)
        for region in (
            "part",
            "wgt",
            "msq",
            "xrdy",
            "sgu",
            "h",
            "rpart",
            "lpart",
            "route",
            "inter",
            "isc",
            "lmsq",
            "lnrdy",
            "shrdy",
            "lqr",
            "ugd",
            "dnd",
        ):
            c[region] = sreg(scratch_base, lay[region][0], region)
        if const_expr(key.tail is not None):
            c["tpre"] = st.tail.peek().pre.ptr
            regions = scratch_regions()
            c["tstream"] = sreg(scratch_base, regions["tstream"][0], "tstream")
            # the tail's own copy: the next layer's seam weights and eps
            tc = {
                **c,
                "eps": key.tail.eps,
                "out_eps": key.tail.out_eps,
                "tail": key.tail,
                "tail_tab": tail,
                "tpart": sreg(scratch_base, regions["tpart"][0], "tpart"),
                "poll_issue": mb.poll_issue,
                "poll_wait": mb.poll_wait,
            }
        on, tls = key.timeline, lds.tls.ptr
        stamp_begin(on, tls, tid, TL_POINTS)
        # ---- o_proj on every CTA past AttnRes's
        eb = bid - E0
        lb = eb - k2b.R_TASKS
        is_r = (bid >= E0) & (eb < k2b.R_TASKS)
        is_l = (bid >= E0 + k2b.R_TASKS) & (lb < k2b.L_TASKS)
        # o_proj first (every AttnRes CTA waits on the slowest partial; loads
        # retire in order), the early weights after: they land before x does
        if is_r:
            stage_oproj(c, bid)
            stamp(on, tls, tid, 1)
            rops = k2b.early_loads(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS)
            k2b.early_compute(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS, rops)
        if is_l:
            stage_oproj(c, bid)
            stamp(on, tls, tid, 1)
            lops = k2b.early_loads(c, 3, lb // k2b.LQ_PARTS, lb % k2b.LQ_PARTS)
            k2b.early_compute(c, 3, lb // k2b.LQ_PARTS, lb % k2b.LQ_PARTS, lops)
        if ~is_r & ~is_l:
            stage_oproj(c, bid)
            stamp(on, tls, tid, 1)
        if (bid >= O0) & ~is_r & ~is_l:
            gi = bid - O0
            if gi < k2b.G_TASKS:
                k2b.early_compute(c, 2, gi, 0, k2b.early_loads(c, 2, gi, 0))
        # ---- AttnRes: the reduced o_proj -> x; then route / SiTU
        if bid < NA:
            # PRELOAD: the slice's loads in flight across the o_proj poll
            sld = slice_loads(c, bid) if key.opt & OPT_PRELOAD else None
            stamp(on, tls, tid, TL_AR_WAIT)
            k2a.stage_reduce(c, bid)
            stamp(on, tls, tid, 1)
            stage_slice(c, bid, sld)
            stamp(on, tls, tid, 2)
            if bid < s:
                stage_gate(c, bid)
            stamp(on, tls, tid, 8)
            stage_mix(c, bid)
            if const_expr(key.opt & OPT_SENT):
                # the words stage_reduce took, off the path to x
                k2a.rearm_ar(c, bid)
            stamp(on, tls, tid, 9)
            stamp(on, tls, tid, 13)
            if bid < s:
                k2b.stage_route(c, bid)
            stamp(on, tls, tid, 10)
            if (bid >= H0) & (bid < H0 + s):
                k2b.stage_h(c, bid - H0)
        if bid >= NA:
            stamp(on, tls, tid, 2)
        # ---- the routed experts
        if const_expr(wide):
            k2b.copy_latq(c)
            nu = k2b.load_route_w(c)
            stamp(on, tls, tid, 11)
        else:
            # the latent (ready ~10 us before the route) quantized first
            k2b.load_latq(c)
            stamp(on, tls, tid, 12)
            # the route table from a bitmask's wave scan
            nu = k2b.load_route2(c)
            stamp(on, tls, tid, 11)
        total = nu * k2b.UG_GROUPS
        last = total - 1
        # reversed placement: the round's leftover tasks go to the high CTAs,
        # whose early work ends first (the AttnRes / route CTAs start ug last)
        if const_expr(wide):
            k2b.run_ug_wide(c, total)
        else:
            task = BLOCKS - 1 - bid
            ops = k2b.ug_loads(c, fx.min(task, last))
            while task < total:
                nxt = task + BLOCKS
                ops_n = k2b.ug_loads(c, fx.min(nxt, last))
                k2b.stage_ug(c, task, ops)
                task = nxt
                ops = ops_n
        if const_expr(wide):
            k2b.ug_done(c)
            stamp(on, tls, tid, 3)
            k2b.run_down_wide(c, nu)
            stamp(on, tls, tid, 16)
            # the shared down while the last down tasks land
            k2b.run_sdown_wide(c, bid, w_sd)
            if bid < k2b.DOWN_TASKS:
                k2b.fold_wide(c, bid)
        else:
            stamp(on, tls, tid, 3)
            if bid < k2b.DOWN_TASKS:
                dops = k2b.down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
                k2b.load_inter(c)
                stamp(on, tls, tid, 15)
                k2b.stage_down(c, bid, nu, dops)
                stamp(on, tls, tid, 16)
            if const_expr(key.opt & OPT_SDOWN_FIRST):
                k2b.run_sdown_first(c, bid, w_sd)
                k2b.run_sdown_rest(c, bid, w_sd)
            else:
                k2b.run_sdown_rest(c, bid, w_sd)
                k2b.run_sdown(c, bid, w_sd)
        stamp(on, tls, tid, 4)
        if (bid >= k2b.LN0) & (bid < k2b.LN0 + k2b.LN_TASKS):
            k2b.stage_lnorm(c, bid - k2b.LN0)
        stamp(on, tls, tid, 5)
        if bid < k2b.UP_TASKS:
            k2b.stage_up(c, bid)
        stamp(on, tls, tid, 6)
        if const_expr(key.reduce):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if (bid >= k2b.RED0) & (bid < k2b.RED0 + k2b.RED_TASKS):
                if const_expr(key.tail is not None):
                    # the slice's bank and weight loads in flight across FINAL's poll
                    tld = tail_seam.tail_loads(tc, bid - k2b.RED0)
                k2b.stage_reduce(c, bid - k2b.RED0)
                if const_expr(key.tail is not None):
                    stamp(on, tls, tid, 7)
                    tail_seam.tail_slice(tc, bid - k2b.RED0, tld)
                    stamp(on, tls, tid, TL_TAIL_PART)
        if const_expr(key.tail is not None):
            row = bid - tail_seam.TAIL_ROW0
            if (row >= 0) & (row < s):
                rld = tail_seam.row_loads(tc, row)
                tail_seam.tail_row(tc, row, rld, lambda: stamp(on, tls, tid, TL_TAIL_GATE))
                stamp(on, tls, tid, TL_TAIL_X)
        if const_expr(key.tail is None):
            stamp(on, tls, tid, 7)
        else:
            # a reduce CTA stamped its reduce before its tail work
            if (bid < k2b.RED0) | (bid >= k2b.RED0 + k2b.RED_TASKS):
                stamp(on, tls, tid, 7)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    assert 1 <= s <= K2_MAX_ROWS
    types = lds_types(key)

    keyed = key_tuple(key, SOURCES)
    name = (
        f"k3_mono_k2_s{s}_k{key.kin}_nb{key.nblocks}_x{int(key.reset)}_r{int(key.reduce)}"
        f"_t{int(key.timeline)}"
        + ("_f1" if key.fold else "")
        + (f"_o{key.opt}" if key.opt else "")
        + ("_cb" if key.core_box else "")
        + (f"_os{key.o_shift}" if key.o_shift else "")
        + (f"_tl{key.tail.tiles}" if key.tail is not None else "")
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
        core_ld: Int32,
        w_o: Int64,
        prefix: Int64,
        prefix_in: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        tl: Int64,
        tail: Int64,
    ):
        _ = keyed
        alloc = fx.SharedAllocator()
        k2_lds = K2Lds(
            smem=alloc.allocate(types.smem).peek(),
            route=alloc.allocate(types.route).peek(),
            stage=alloc.allocate(types.stage),
        )
        args = dict(
            zip(
                K2_ARGS,
                (
                    core,
                    core_ld,
                    w_o,
                    prefix,
                    prefix_in,
                    blocks,
                    blk_sm,
                    blk_sr,
                    ares_nw,
                    ares_qk,
                    in_nw,
                    w_gate,
                    bias,
                    w_ld,
                    w_sgu,
                    w_sd,
                    w13,
                    w13s,
                    w2,
                    w2s,
                    ln_w,
                    w_up,
                    out,
                    peers,
                    rank,
                    epoch,
                    layer,
                    tl,
                    tail,
                ),
            )
        )
        k2_body(key, k2_lds, scratch, args)

    @flyc.jit
    def launch(
        core: Int64,
        core_ld: Int32,
        w_o: Int64,
        prefix: Int64,
        prefix_in: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        tl: Int64,
        tail: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
            core_ld,
            w_o,
            prefix,
            prefix_in,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_gate,
            bias,
            w_ld,
            w_sgu,
            w_sd,
            w13,
            w13s,
            w2,
            w2s,
            ln_w,
            w_up,
            out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
            tl,
            tail,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch


TP_ = k2b.TP


def ptr(t):
    return 0 if t is None else t.data_ptr()


def k2_launch(
    key: K2Build,
    *,
    core,
    w_o,
    prefix,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_gate,
    bias,
    w_ld,
    w_sgu,
    w_sd,
    w13,
    w13s,
    w2,
    w2s,
    ln_w,
    w_up,
    out,
    scratch,
    peers,
    rank,
    epoch,
    layer,
    tl=None,
    prefix_in=None,
    attn=None,
    tail=None,
):
    """Host launcher: ``core`` [S, key.kin] bf16 (o_proj's input; rows may be
    strided, 16-B aligned), ``prefix`` updated in
    place (read from ``prefix_in`` when given), ``out`` the MoE output [S, 7168]
    (plus the updated prefix under ``key.fold``); MoE weights as ``moe``.
    ``attn`` (``kda_layer.Attn``): the layer's KDA attention in the same launch,
    its core into ``core`` (``kda_layer.core_view``; ``key.core_box``).
    ``tail``: ``key.tail``'s int64 device table (``TAIL_ARGS``)."""
    assert (
        w_o.shape == (HIDDEN, key.kin)
        and key.kin % 128 == 0
        and core.shape == (key.tokens, key.kin)
        and core.stride(1) == 1
        and core.stride(0) % 8 == 0
        and core.data_ptr() % 16 == 0
        and prefix.is_contiguous()
    )
    assert w_up.shape == (k2b.UP_N, k2b.LAT) and w_up.is_contiguous()
    assert SENT_READY or not key.opt & OPT_SENTS, "OPT_SENTS without init_peer"
    assert not key.opt & OPT_TAG_X or scratch.numel() >= scratch_alloc_bytes(), (
        "OPT_TAG_X without scratch_alloc_bytes of scratch"
    )
    assert (key.tail is None) == (tail is None), "a tail build takes its table, and only it"
    if tail is not None:
        assert tail.dtype == torch.int64 and tail.numel() == len(TAIL_ARGS) and tail.is_cuda
        assert key.reduce and key.fold, "the tail seam slices the stream the fold writes"
    vals = (
        core.data_ptr(),
        core.stride(0),
        w_o.data_ptr(),
        prefix.data_ptr(),
        (prefix if prefix_in is None else prefix_in).data_ptr(),
        blocks.data_ptr(),
        blocks.stride(0),
        blocks.stride(1),
        ares_nw.data_ptr(),
        ares_qk.data_ptr(),
        in_nw.data_ptr(),
        w_gate.data_ptr(),
        bias.data_ptr(),
        w_ld.data_ptr(),
        w_sgu.data_ptr(),
        w_sd.data_ptr(),
        w13.data_ptr(),
        w13s.data_ptr(),
        w2.data_ptr(),
        w2s.data_ptr(),
        ln_w.data_ptr(),
        w_up.data_ptr(),
        out.data_ptr(),
        scratch.data_ptr(),
        peers.data_ptr(),
        rank,
        epoch.data_ptr(),
        layer,
        ptr(tl),
        ptr(tail),
    )
    if attn is not None:
        from sgl_amd_mono.kimi_k3.gfx950 import kda_layer

        kda_layer.run_layer(attn, key, vals)
    else:
        build(key)(*vals, stream=torch.cuda.current_stream())
