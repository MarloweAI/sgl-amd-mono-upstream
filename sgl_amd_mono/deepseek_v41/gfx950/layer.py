# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build K1 (attention seam and front), K2 (attention back, seam, MoE and TP
all-reduces) and the FFN launch (seam, MoE and TP all-reduce)."""

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.epoch import (
    EPOCH_MARKS,
    epoch_begin,
    epoch_end,
    gstore,
    kernel_symbol,
)
from sgl_amd_mono.common.mx import clamp_fp8
from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_SYS,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_pack4,
    peer_bases,
    ptr64,
    rsrc,
    sum_partials,
    uniform,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
    kernel_sources,
    key_tuple,
)
from sgl_amd_mono.common.stamps import TL_POINTS, stamp, timeline_row
from sgl_amd_mono.common.sync import FLAG_STRIDE, Mailbox, _spin, preg, publish, sreg

from .attention import back as mb_back
from .attention import device as mb_device
from .attention import front as mb_front
from .attention import index_query
from .attention.plan import (
    Q_RANK,
    QKV_ROWS,
    ROWS,
    WINDOW,
    WORD,
    WQKV_TASKS,
    Dims,
    back_scratch,
    front_scratch,
    q_words,
    tagged_q_split,
    tagged_x8b,
    wqkv_split,
)
from .indexer import compress
from .stages import engram, index_glue, moe, seam
from .stages.dims import Dims as StageDims
from .stages.gemv import plus, token_tiles
from .stages.moe_shape import (
    EXPERTS,
    SORT_NETS,
    TOPK,
    MoeBuild,
    route_shape,
    row_widths,
)
from .token_par import (
    XG_WORDS,
    acell_region,
    chunk_pairs,
    fcell_region,
    half_pair,
    own_view,
    peer_offsets,
    tagged_rows,
)

SOURCES = kernel_sources(__name__)

_STREAM = fx.Stream(None)
MAX_TOKENS = 48
HIDDEN = seam.HIDDEN
X_WORDS = HIDDEN // 4
X_GROUPS = HIDDEN // 32
SLICES = seam.SLICES
# the MoE's MXFP8 x regions, named apart from K1's: one launch per layer writes both, and
# a CTA's cached lines of K1's would be stale for the MoE's
MOE_X8 = {"x8": "moe_x8", "x8s": "moe_x8s"}
COUNTER_WORDS = 2 * 256  # the MoE's ug queue and down counts, a word a slot
COUNTERS = ("ugq", "dq")
# epoch buffer words: the epoch, one mark per CTA, then the MoE counters on their own lines
EPOCH_COUNTERS = -(-(EPOCH_MARKS + BLOCKS) // 64) * 64
EPOCH_WORDS = EPOCH_COUNTERS + COUNTER_WORDS


class MonoBuild(msgspec.Struct, frozen=True):
    tokens: int
    tp: int
    kt: bool = False  # K1 also writes K2's key table
    # K1 folds the owed post into the residual; off on the first layer and after an Engram write
    fold: bool = True
    # an index source: K1 hands normed and qr to SGLang's indexer, K2 builds the key table
    index: bool = False
    # K2's keys include top-k compressed rows (compress ratio 1 or 2), not only the window
    compressed: bool = True
    # SGLANG_AMD_MONO_FAULT_INJECT: the wait to stall at (0: none) after fault_after launches
    fault_site: int = 0
    fault_after: int = 0
    # the route and window: the target's, or the DSpark draft's (``DSV41MonoRunner``)
    experts: int = EXPERTS
    topk: int = TOPK
    ug_groups: int = 2
    swa_width: int = WINDOW
    # SGLANG_AMD_MONO_TIMELINE: each CTA marks its stages in a scratch timeline
    timeline: bool = False
    # token parallel (``token_par``): each rank's seam, norm and qkv on its own rows
    tok_par: bool = False
    # K1 also computes an index source's index query, where ``index_query.fused``
    iq: bool = False
    # with the index query, an index KV source's compressor projection (its compress ratio)
    cp: int = 0
    # wqkv, wq_b and wo_b in SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE's (16, 16) layout; wo_a row-major
    shuffled: bool = False
    # the first layer: K1 reads each stream from the embedding table at the token's id
    embed: bool = False
    # a reuse layer's launch also projects this many 16-row tasks of a later Engram layer's
    # rows (``stages.engram``)
    eg: int = 0
    # an index source's K2 also runs its selection's glue (``stages.index_glue``: TABLE, RAWS)
    glue: int = 0
    # one launch per layer: the previous launch ran this layer's attention seam slice
    # (``seam_next``); the front reads its seam state, each wqkv CTA norms the rows itself
    seamed: bool = False
    # one launch per layer: the MoE reduce runs the next layer's attention seam slice into
    # the seam state the next launch reads (``seamed``)
    seam_next: bool = False
    # a layer of a multi-layer launch's loop (``build_mono_layers``): its seam slices may run
    # the layer before's MoE all-reduce, its FFN gates flag their mixes, its MoE may skip
    # its all-reduce
    chain: bool = False


ROUTE = ("experts", "topk", "ug_groups", "swa_width")


def route_of(key: MonoBuild) -> dict:
    return {f: getattr(key, f) for f in ROUTE}


def _variant(key: MonoBuild) -> dict:
    """A kernel name's route parameters, where they are not the target's."""
    base = MonoBuild(key.tokens, key.tp)
    short = {"experts": "e", "topk": "k", "ug_groups": "u", "swa_width": "w"}
    out = {short[f]: v for f, v in route_of(key).items() if v != getattr(base, f)}
    return (
        out
        | ({"tl": 1} if key.timeline else {})
        | ({"p": 1} if key.tok_par else {})
        | ({"sh": 1} if key.shuffled else {})
        | ({"em": 1} if key.embed else {})
        | ({"eg": key.eg} if key.eg else {})
        | ({"gl": key.glue} if key.glue else {})
        | ({"sd": 1} if key.seamed else {})
        | ({"sn": 1} if key.seam_next else {})
        | ({"ch": 1} if key.chain else {})
    )


def _moe_key(key: MonoBuild) -> MoeBuild:
    return MoeBuild(key.tokens, key.tp, key.experts, key.topk, key.ug_groups)


def _front_key(key: MonoBuild):
    return mb_front.FrontBuild(
        key.tokens, key.tp, key.kt, key.swa_width, shuffled=key.shuffled
    )


def _back_key(key: MonoBuild):
    # 0: the window only (compress ratio 0); 1: also top-k compressed rows
    return mb_back.BackBuild(
        key.tokens, key.tp, 1 if key.compressed else 0, shuffled=key.shuffled
    )


def scratch_layout(s: int, tp: int, timeline: bool = False, **route) -> dict:
    """Return each scratch region of width s -> (byte offset, bytes), without overlap."""
    d = Dims(tp)
    out: dict = {}
    off = 0

    def place(regions):
        nonlocal off
        base = off
        for name, (o, n) in regions.items():
            assert name not in out, name
            out[name] = (base + o, n)
        off = max(off, base + max(o + n for o, n in regions.values()))
        off = -(-off // 256) * 256

    place(front_scratch(s, d))
    place(seam.scratch_layout(s))
    place(back_scratch(s, d, start=0))
    regions = moe.scratch_layout(_moe_key(MonoBuild(s, tp, **route))).items()
    mr = {MOE_X8.get(n, n): v for n, v in regions if n not in COUNTERS}
    lo = min(o for o, _ in mr.values())
    place({n: (o - lo, n_) for n, (o, n_) in mr.items()})
    place({"normed": (0, s * HIDDEN * 2), "xrdy_moe": (s * HIDDEN * 2 + 256, s * 8)})
    # the key table's flag a token (built by K2 on an index source, else in one launch
    # per layer by K1); one launch per layer's, a line each: a q flag a (token tile, CTA),
    # a record's a token (token parallel) and the gates' mixes' a token
    flag = 8 * FLAG_STRIDE
    q_flags = len(token_tiles(s)) * BLOCKS * flag
    place(
        {
            "krdy": (0, s * 8),
            "qb": (256, q_flags),
            "vrdy": (256 + q_flags, s * flag),
            "grdy": (256 + q_flags + s * flag, s * flag),
            # the FFN gates' mixes a token, for the next layer's seam slice (``seam_next``)
            "frdy": (256 + q_flags + 2 * s * flag, s * flag),
        }
    )
    # one launch per layer with ``tagged_q_split``: q as tagged words (``front.stage_wqb``)
    place({"qt": (0, s * q_words(d) * WORD if tagged_q_split(d, s) else 0)})
    # the router CTAs' x boxes (``moe.signal_xbox``): uncached scratch, scalar-polled
    place({"xbox": (0, moe.xbox_bytes(s))})
    if timeline:
        place({"timeline": (0, BLOCKS * TL_POINTS * 8)})
    return out


def scratch_bytes(
    s: int = MAX_TOKENS, tp: int = 2, timeline: bool = False, **route
) -> int:
    return max(o + n for o, n in scratch_layout(s, tp, timeline, **route).values())


def acell_offset(tp: int) -> int:
    """ACELL's byte offset in a parity's half (``token_par.acell_region``)."""
    return acell_region(tp, range(1, MAX_TOKENS + 1))[0]


def acell_region_bytes(tp: int) -> int:
    """ACELL's bytes in a parity's half (``token_par.acell_region``)."""
    return acell_region(tp, range(1, MAX_TOKENS + 1))[1]


def fcell_offset(tp: int) -> int:
    """FCELL's byte offset in a parity's half (``token_par.fcell_region``)."""
    return fcell_region(tp, range(1, MAX_TOKENS + 1), row_widths(tp))[0]


def fcell_region_bytes(tp: int) -> int:
    """FCELL's bytes in a parity's half (``token_par.fcell_region``)."""
    return fcell_region(tp, range(1, MAX_TOKENS + 1), row_widths(tp))[1]


def peer_half_bytes(tp: int) -> int:
    """One parity's peer regions (``token_par.peer_layout``), then ACELL, then FCELL."""
    return sum(fcell_region(tp, range(1, MAX_TOKENS + 1), row_widths(tp)))


def _epoch(epoch):
    return fx.Int32(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))


# ---------------------------------------------------------------- K1


@flyc.jit
def store_x8(c, t, col, ys, live):
    """Store the seam's norm output at ``col`` as MXFP8 X8 / X8S for the front."""
    amax = abs(ys[0])
    for y in ys[1:]:
        amax = fx.max(amax, abs(y))
    code = moe.vllm_mx_code(butterfly(amax, (1, 2), fx.max))
    mul = ((254 - code) << 23).bitcast(fx.Float32)
    w0 = fp8_pack4(*[clamp_fp8(ys[i] * mul) for i in range(4)])
    w1 = fp8_pack4(*[clamp_fp8(ys[4 + i] * mul) for i in range(4)])
    if live:
        bo.buffer_store(
            fx.Vector.from_elements([w0, w1], fx.Int32),
            rsrc(c["x8"]),
            t * X_WORDS + col // 4,
            cache_modifier=CM_DEV,
        )
        if c["tid"] % 4 == 0:
            bo.buffer_store(
                code, rsrc(c["x8s"]), t * X_GROUPS + col // 32, cache_modifier=CM_DEV
            )


@flyc.jit
def publish_x8(c, t):
    """Token t's two XRDY flags (one a wqkv K half) once its X8 / X8S landed."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] < 2:
        c["mb"].put(c["xrdy"], 2 * t + c["tid"], fx.Int32(1))
    gpu.barrier()


K1_ARGS = (
    "res_in",
    "pend",
    "post_in",
    "comb_in",
    "pre_in",
    "hc_fn",
    "hc_scale",
    "hc_base",
    "attn_w",
    "res_out",
    "aux_out",
    "aux_stride",
    "post_out",
    "comb_out",
    "pre_out",
    "wqkv",
    "wqkv_s",
    "qn_w",
    "kvn_w",
    "wqb",
    "wqb_s",
    "cos_sin",
    "pos",
    "slot",
    "slot_words",
    "swa",
    "swa_stride",
    "swa_block",
    "swa_idx",
    "swa_lens",
    "topk",
    "topk_lens",
    "kt",
    "klen",
    "q",
    "normed_out",
    "qr_out",
    "iq_w",
    "iq_ws",
    "iw_w",
    "iw_scale",
    "num_pos",
    "iq_out",
    "iqs_out",
    "iw_out",
    "scratch",
    "sym",
    "peers",
    "rank",
    "epoch",
    "ids",
    "ids_words",
)
# an index K1's past K1_ARGS: its compressor's projection weights and output, then the
# rest of the compressor's arguments (``compress.KC_ARGS`` past the projection)
K1_CP_ARGS = ("cp_w", "cp_out")
K1_KC_ARGS = tuple(f"kc_{n}" for n in compress.KC_ARGS[1:])


@flyc.jit
def run_k1(key: MonoBuild, sl, front, a: dict, ep, k2=None, early=None, chained=None):
    """K1 on this CTA: the pending post, the attention seam and the attention front
    (``sl``, ``front``: the seam's and the front's LDS views). ``k2``: one launch per
    layer, the attention back's hand-off (``layer_handoff``); ``early``: its split's keys
    from the launch's inputs (``back.EarlyKeys``), loaded past the wqkv and wq_b weights.
    ``chained`` (``key.chain``, traced): the layer before ran in this launch; each seam
    slice then sums that layer's MoE partials of its columns itself (``chained_slice``)."""
    (
        res_in,
        pend,
        post_in,
        comb_in,
        pre_in,
        hc_fn,
        hc_scale,
        hc_base,
        attn_w,
        res_out,
        aux_out,
        aux_stride,
        post_out,
        comb_out,
        pre_out,
        wqkv,
        wqkv_s,
        qn_w,
        kvn_w,
        wqb,
        wqb_s,
        cos_sin,
        pos,
        slot,
        slot_words,
        swa,
        swa_stride,
        swa_block,
        swa_idx,
        swa_lens,
        topk,
        topk_lens,
        kt,
        klen,
        q,
        normed_out,
        qr_out,
        iq_w,
        iq_ws,
        iw_w,
        iw_scale,
        num_pos,
        iq_out,
        iqs_out,
        iw_out,
        scratch,
        sym,
        peers,
        rank,
        epoch,
        ids,
        ids_words,
    ) = (a[n] for n in K1_ARGS)
    s, tp = key.tokens, key.tp
    layout = scratch_layout(s, tp, key.timeline, **route_of(key))
    fkey = _front_key(key)
    # token parallel: the seam, wqkv and qkv on this rank's rows
    own_n = s // tp if key.tok_par else s
    # the norms past the slices' CTAs (the wqkv waits on them), and past a split wqkv's
    NORM0 = 2 * WQKV_TASKS if wqkv_split(s, own_n) else SLICES
    GATE0 = NORM0 + own_n
    # seamed: no slice; the norms and gates run at entry
    assert not key.seamed or not (key.tok_par or key.index or k2 is None), key
    assert GATE0 + own_n <= BLOCKS
    # the index query here (``index_query``) only where it is reproduced
    assert not key.iq or (key.index and index_query.fused(s, key.tok_par)), key
    assert not key.cp or (key.iq and not key.tok_par and key.cp in (1, 2)), key
    QKV0 = GATE0 + own_n if GATE0 + 2 * own_n <= BLOCKS else 0
    # one launch per layer with tagged q to the split: the kv norms apart, on the seam
    # norms' CTAs, off the q path
    KV0 = NORM0 if tagged_q_split(Dims(tp), s, key.tok_par) and k2 is not None else QKV0
    # the head weights' CTAs, past the q / kv norms': they run in q's wait
    IW0 = (QKV0 + own_n) % BLOCKS
    # the compressor projection's CTAs, past the head weights'
    CP0 = (IW0 + index_query.iw_tasks(own_n)) % BLOCKS
    half = peer_half_bytes(tp)
    bid = fx.block_idx.x
    tag = (ep << 1) + 1
    fargs = {
        "x": 0,
        "x_stride": 0,
        "wqkv": wqkv,
        "wqkv_s": wqkv_s,
        "qn_w": qn_w,
        "kvn_w": kvn_w,
        "wqb": wqb,
        "wqb_s": wqb_s,
        "cos_sin": cos_sin,
        "pos": pos,
        "slot": slot,
        "slot_words": slot_words,
        "swa": swa,
        "swa_stride": swa_stride,
        "swa_block": swa_block,
        "swa_idx": swa_idx,
        "swa_lens": swa_lens,
        "topk": topk,
        "topk_lens": topk_lens,
        "kt": kt,
        "klen": klen,
        "q": q,
        "qr_out": qr_out,
        "normed_out": normed_out,
        "iq_w": iq_w,
        "iq_ws": iq_ws,
        "iw_w": iw_w,
        "iw_scale": iw_scale.bitcast(fx.Float32),
        "num_pos": num_pos,
        "iq_out": iq_out,
        "iqs_out": iqs_out,
        "iw_out": iw_out,
        # an index KV source's compressor projection (``key.cp``; K1's own launch only)
        "cp_w": a.get("cp_w", 0),
        "cp_out": a.get("cp_out", 0),
    }
    flayout = {n: layout[n] for n in front_scratch(s, Dims(tp))}
    c = mb_front.front_context(fkey, front, fargs, scratch, tag, flayout)
    c["index"] = key.index
    if const_expr(k2 is not None):
        # one launch per layer: the products the attention back reads, device scope, flagged
        c["k2"] = k2
        c["dev"] = True
        if const_expr(tagged_q_split(Dims(tp), s, key.tok_par)):
            # q to the attention back as tagged words, the records behind flags of their own
            c["qt"] = scratch + fx.Int64(layout["qt"][0])
    if const_expr(key.iq):
        c["iq_chains"] = index_query.iq_chains(s)
    if const_expr(key.timeline):
        c["tl"] = timeline_row(scratch, layout["timeline"][0], bid)
    amb = Mailbox(tag)
    tid = fx.thread_idx.x
    ca = {
        "S": s,
        "tid": tid,
        "bid": bid,
        "lane": tid % 64,
        "wave": tid // 64,
        "rl": sl.rl.ptr,
        "fl": sl.fl.ptr,
        "red": sl.red.ptr,
        "put": amb.put,
        "put_bf": amb.put_bf,
        "put_words": amb.put_words,
        "poll": amb.poll,
        "args": {
            "res_in": res_in,
            "pend": pend,
            "post_in": post_in,
            "comb_in": comb_in,
            "pre_in": pre_in,
            "hc_fn": hc_fn,
            "hc_scale": hc_scale,
            "hc_base": hc_base,
            "norm_w": attn_w,
            "res_out": res_out,
            "post_out": post_out,
            "comb_out": comb_out,
            "pre_out": pre_out,
            # DSpark's capture of this layer: its streams' mean (0: none), rows
            # ``aux_stride`` elements apart (a column block of the packed captures)
            "aux_out": aux_out,
            "aux_stride": aux_stride,
        },
        "d": StageDims(tp),
        # SGLang's Triton boundary + rmsnorm arithmetic (seam.slice_row, stage_norm)
        "sgl": True,
        "fold": key.fold,
        # before own_view moves the row arguments
        "aux_on": aux_out != 0,
    }
    for region in ("lin", "pmix"):
        ca[region] = sreg(scratch, layout[region][0], region)
    if const_expr(key.seamed):
        # the layer input the previous launch's tail stored: plain rows
        st = seam.seam_state_layout(s)
        ca["lin"] = a["seam_in"] + fx.Int64(st["lin"][0])
        ca["poll"] = seam.plain_loads
    if const_expr(k2 is not None):
        ca["dev"] = True
    if const_expr(key.embed):
        # the first layer: ``res_in`` is the embedding table, a row by ``ids``
        assert not key.fold, key
        ca["embed"] = True
        ca["args"] = {**ca["args"], "emb": res_in, "ids": ids, "ids_words": ids_words}
    t0 = 0
    if const_expr(key.tok_par):
        t0 = rank * own_n
        ca = own_view(ca, t0, own_n)
        ca["rank"] = rank
        par = fx.Int64(ep & 1) * fx.Int64(half)
        bases = [b + par for b in peer_bases(peers, tp)]
        pl = peer_offsets(s, tp)
        mb_front.token_par_context(c, own_n, t0, rank, bases, sym + par, pl)

    if const_expr(chained is not None):
        cr = chained_reduce(key, ca, sl, scratch, sym, ep)

    def seam_slice():
        stamp(c, 32)
        # seamed: the previous launch ran the slice
        if not key.seamed:
            # TP4: the partial projections later (``slice_mixes``)
            cs = {**ca, "mixes": seam.MIX_LATE} if tp == 4 else ca
            if const_expr(chained is None):
                for task in range(first_task(bid, 0), SLICES, BLOCKS):
                    seam.stage_slice(cs, task)
            else:
                if chained:
                    for ctask in range(first_task(bid, 0), SLICES, BLOCKS):
                        chained_slice(cr, ctask)
                else:
                    for task in range(first_task(bid, 0), SLICES, BLOCKS):
                        seam.stage_slice(cs, task)
            gpu.barrier()
            stamp(c, 33)

    xr = sl.xr.ptr if tagged_rows(tp) else None

    def slice_mixes():
        # TP4: the slice's partial projections past its CTA's wqkv weights (row halves: a
        # whole-K tile a CTA): R_new and LIN, which the norms wait on, go out before the
        # weights; the mixes only the gates read. TP8 (K halves) keeps them in the slice:
        # there the later mixes cost the attention back more than the weights gain
        for task in range(first_task(bid, 0), SLICES, BLOCKS):
            seam.slice_partials(ca, task)

    def store(t, col, ys, live):
        store_x8(c, t, col, ys, live)
        if const_expr(key.index):
            # the normed input, SGLang's compressor and indexer's input
            store_normed(normed_out, plus(t0, t), col, ys, live)
            if const_expr(key.tok_par and tagged_rows(tp)):
                stage_xg_row(xr, col, ys, live)
            elif const_expr(key.tok_par):
                push_bf_row(ca, bases, pl["nbg"], plus(t0, t), (col, ys, live))

    def seam_norm():
        for t in range(first_task(bid, NORM0), own_n, BLOCKS):
            seam.stage_norm(ca, t, store)
            if const_expr(key.index and key.tok_par and tagged_rows(tp)):
                # the LDS row is whole past the norm's barrier
                push_bf_row(ca, bases, pl["nbg"], plus(t0, t), xr, amb.tag)
            publish_x8(c, t)
        gpu.barrier()
        stamp(c, 34)
        if const_expr(early is not None):
            early.inputs()

    def seam_gate():
        # only K2 reads the mixes, so they fill q's wait; the gate's LDS aliases the front's
        gpu.barrier()
        for t in range(first_task(bid, GATE0), own_n, BLOCKS):
            seam.stage_gate(ca, t)
            if const_expr(k2 is not None):
                mb_front.hand_off(c, "grdy", t * FLAG_STRIDE)
        if const_expr(key.iq):
            # the head weights fill the same wait: the wqkv's LDS is free by now
            push = None
            if const_expr(key.tok_par):
                cp = {"put_bf": amb.put_bf, "rank": rank, "d": ca["d"]}
                push = lambda row, h, w0, w1: push_head_pair(  # noqa: E731
                    cp, bases, pl["iwg"], row, h, [w0, w1]
                )
            index_query.run_iw(c, c["args"], own_n, IW0, c["xl"], t0, push)
        if const_expr(key.cp):
            # the compressor projection fills it too, on the CTAs past them (one token
            # tile: its operands issued past the wqkv GEMV)
            index_query.run_cp(
                c, c["args"], own_n, CP0, c["xl"], key.cp, cp_ops if cp_ops else None
            )
        if const_expr(early is not None):
            early.entries()
            early.touch()

    # one token tile: the compressor projection's operands go out past the wqkv GEMV, ahead
    # of the wq_b weights (they return in issue order), not behind them in q's wait
    cp_ops = {}

    def cp_issue(seen):
        cp_ops.update(index_query.cp_issue(c, c["args"], own_n, CP0, key.cp, seen))

    def early_keys():
        early.entries()
        early.touch()

    if const_expr(key.seamed):
        # LIN is in: the norms run at the launch's entry; the mixes' partials and the
        # gates in the attention back's idle CTAs (``seam_gates``)
        mb_front.run_front(
            c,
            fkey,
            bid,
            x8_given=True,
            before_wqkv=seam_slice,
            after_wqkv=seam_norm,
            before_wqb=early_keys,
            qkv_first=QKV0,
            kv_first=KV0,
            # a row tile's sums each on one CTA: no partner hand-off
            row_halves=True,
        )
    else:
        mb_front.run_front(
            c,
            fkey,
            bid,
            x8_given=True,
            before_wqkv=seam_slice,
            after_wqkv=seam_norm,
            after_weights=slice_mixes if tp == 4 else None,
            before_wqb=seam_gate,
            before_wqb_weights=cp_issue
            if key.cp and len(token_tiles(own_n)) == 1
            else None,
            # past the slices, norms and gates where the grid has room: a slice CTA
            # issues its weights after its slice, and the norms' publish (vmcnt(0))
            # waited for them to land
            qkv_first=QKV0,
            kv_first=KV0,
            # TP4: a row tile's sums each on one CTA, so no partial crosses CTAs and each
            # wqkv CTA's wq_b weights go out right past its own GEMV
            row_halves=tp == 4,
        )
    if const_expr(key.tok_par):
        # K1's outputs hold every row: copy in the rows the other ranks computed
        own = sym + par
        for t in range(first_task(bid, 0), s, BLOCKS):
            if t // own_n != rank:
                mb_front.record_from(c, preg(own, pl["kg"], "kg"), t)
                if const_expr(key.index):
                    if const_expr(tagged_rows(tp)):
                        gather_words(ca, own + fx.Int64(pl["nbg"]), t, normed_out, amb.tag)
                    else:
                        gather_row(ca, preg(own, pl["nbg"], "nbg"), t, normed_out)
                    mb_front.qr_from(c, preg(own, pl["qrg"], "qrg"), t)
                if const_expr(key.iq):
                    index_query.iw_from(c, preg(own, pl["iwg"], "iwg"), t)
            if const_expr(k2 is not None):
                # one launch per layer: every row's record is in (the others' copied
                # past this CTA's q flags)
                mb_front.hand_off(c, "vrdy", t * FLAG_STRIDE)
    if const_expr(key.cp):
        # the rest of the compressor, on the CTAs that finish their front first, in the
        # front's X tile
        xl = mb_front.tile_rows(s) * mb_front.lds_row(mb_front.X_WORDS)
        assert compress.LDS_WORDS <= xl, s
        kc = {n: a[f"kc_{n}"] for n in compress.KC_ARGS[1:]}
        kc.update(kv=a["cp_out"], ratio=fx.Int32(key.cp), cm=CM_DEV)
        index_query.run_kc(c, kc, s, key.cp)
    stamp(c, 40)


def chained_reduce(key: MonoBuild, ca, sl, scratch, sym, ep) -> dict:
    """The context of a seam slice that runs the layer before's MoE all-reduce (that layer
    on epoch ``ep - 1``, in this launch): its gates' flags (``frdy``) and partials (its
    parity's FCELL cells, or its tagged pairs) under its K2 tag, this seam's pend LDS."""
    s, tp = key.tokens, key.tp
    assert key.chain and not (key.tok_par or key.seamed or key.seam_next), key
    assert TAIL_SUM0 + s * seam.COLS // 2 <= THREADS, s
    layout = scratch_layout(s, tp, key.timeline, **route_of(key))
    pep = ep - 1
    own = sym + fx.Int64(pep & 1) * fx.Int64(peer_half_bytes(tp))
    if const_expr(route_shape(_moe_key(key)).row_moe):
        ffn = preg(own, fcell_offset(tp), "fcell")
    else:
        ffn = preg(own, peer_offsets(s, tp)["ffn"], "ffn")
    return {
        # TP4's slice leaves its partial projections to ``slice_mixes``, as an unchained one
        "ca": {
            **ca,
            "pend_lds": sl.pl.ptr,
            "mixes": seam.MIX_LATE if tp == 4 else seam.MIX_HERE,
        },
        "cp": {**ca, "pend_lds": sl.pl.ptr, "poll": Mailbox((pep << 1) + 2).poll},
        "ffn": ffn,
        "frdy": sreg(scratch, layout["frdy"][0], "frdy"),
        # the layer before's MoE output, this layer's ``pend``
        "out": ca["args"]["pend"],
    }


@flyc.jit
def chained_slice(cr, task):
    """Seam slice ``task`` of a layer whose layer before ran in this launch, as that layer's
    ``seam_tail`` would run it: once its gates flagged their mixes, the slice's own loads,
    then its MoE partials of the slice's 32 columns summed in rank order into the pend LDS
    and that layer's ``out`` (its all-reduce, which it skipped), then the slice. The
    residual is that layer's FFN slice ``task``, stored by this CTA."""
    ca, cp = cr["ca"], cr["cp"]
    s, tp = ca["S"], ca["d"].tp
    pre = tail_loads(cp, ca, cr["frdy"], s, task)
    tail_run(cp, ca, cr["ffn"], s, tp, task, pre, cr["out"])


def _k1_lds(s, tp):
    """K1's LDS: its seam's and its front's stages, one at a time."""

    if tagged_rows(tp):

        @fx.struct
        class SeamLds:
            red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
            rl: fx.Array[fx.Float32, s * seam.KT, 16]
            fl: fx.Array[fx.Float32, seam.MIX * seam.KT, 16]
            # an index source's normed row for the other ranks (``push_bf_row``)
            xr: fx.Array[fx.Int32, 3 * XG_WORDS, 16]

    else:

        @fx.struct
        class SeamLds:
            red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
            rl: fx.Array[fx.Float32, s * seam.KT, 16]
            fl: fx.Array[fx.Float32, seam.MIX * seam.KT, 16]

    FrontLds = mb_front.front_lds(s, Dims(tp))

    @fx.union
    class K1Lds:
        seam: SeamLds
        front: FrontLds  # type: ignore[valid-type]

    return K1Lds


def build_mono_k1(key: MonoBuild):
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    K1Lds = _k1_lds(s, tp)
    name = kernel_symbol(
        "dsv41_mono_k1",
        s=s,
        tp=tp,
        kt=key.kt,
        f=key.fold,
        i=key.index,
        **({"q": 1} if key.iq else {}),
        **({"cp": key.cp} if key.cp else {}),
        **_variant(key),
    )
    keyed = key_tuple(key, SOURCES)
    half = peer_half_bytes(tp)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        res_in: Int64,
        pend: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        attn_w: Int64,
        res_out: Int64,
        aux_out: Int64,
        aux_stride: Int32,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        wqkv: Int64,
        wqkv_s: Int64,
        qn_w: Int64,
        kvn_w: Int64,
        wqb: Int64,
        wqb_s: Int64,
        cos_sin: Int64,
        pos: Int64,
        slot: Int64,
        slot_words: Int32,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        swa_idx: Int64,
        swa_lens: Int64,
        topk: Int64,
        topk_lens: Int64,
        kt: Int64,
        klen: Int64,
        q: Int64,
        normed_out: Int64,
        qr_out: Int64,
        iq_w: Int64,
        iq_ws: Int64,
        iw_w: Int64,
        iw_scale: Int32,
        num_pos: Int32,
        iq_out: Int64,
        iqs_out: Int64,
        iw_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        ids: Int64,
        ids_words: Int32,
        cp_w: Int64,
        cp_out: Int64,
        kc_per_req: Int32,
        kc_state: Int64,
        kc_ring: Int32,
        kc_norm: Int64,
        kc_eps: Int32,
        kc_freqs: Int64,
        kc_pos: Int64,
        kc_req: Int64,
        kc_raw_loc: Int64,
        kc_iloc: Int64,
        kc_cache: Int64,
        kc_page_bits: Int32,
        kc_page_bytes: Int32,
        kc_wk: Int64,
        kc_knorm: Int64,
        kc_keps: Int32,
        kc_payload: Int64,
        kc_pscale: Int64,
        kc_wide: Int32,
        kc_live: Int32,
    ):
        _ = keyed
        warm_kernargs(K1_KERNARG_BYTES)
        # every wait runs under the guard, so a timeout flags every peer and stops the launch
        with guard.scope(
            guard.Guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=tp,
                flags_offset=2 * half,
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            )
        ):
            lds = fx.SharedAllocator().allocate(K1Lds)
            a = {
                "res_in": res_in,
                "pend": pend,
                "post_in": post_in,
                "comb_in": comb_in,
                "pre_in": pre_in,
                "hc_fn": hc_fn,
                "hc_scale": hc_scale,
                "hc_base": hc_base,
                "attn_w": attn_w,
                "res_out": res_out,
                "aux_out": aux_out,
                "aux_stride": aux_stride,
                "post_out": post_out,
                "comb_out": comb_out,
                "pre_out": pre_out,
                "wqkv": wqkv,
                "wqkv_s": wqkv_s,
                "qn_w": qn_w,
                "kvn_w": kvn_w,
                "wqb": wqb,
                "wqb_s": wqb_s,
                "cos_sin": cos_sin,
                "pos": pos,
                "slot": slot,
                "slot_words": slot_words,
                "swa": swa,
                "swa_stride": swa_stride,
                "swa_block": swa_block,
                "swa_idx": swa_idx,
                "swa_lens": swa_lens,
                "topk": topk,
                "topk_lens": topk_lens,
                "kt": kt,
                "klen": klen,
                "q": q,
                "normed_out": normed_out,
                "qr_out": qr_out,
                "iq_w": iq_w,
                "iq_ws": iq_ws,
                "iw_w": iw_w,
                "iw_scale": iw_scale,
                "num_pos": num_pos,
                "iq_out": iq_out,
                "iqs_out": iqs_out,
                "iw_out": iw_out,
                "scratch": scratch,
                "sym": sym,
                "peers": peers,
                "rank": rank,
                "epoch": epoch,
                "ids": ids,
                "ids_words": ids_words,
                "cp_w": cp_w,
                "cp_out": cp_out,
                "kc_per_req": kc_per_req,
                "kc_state": kc_state,
                "kc_ring": kc_ring,
                "kc_norm": kc_norm,
                "kc_eps": kc_eps,
                "kc_freqs": kc_freqs,
                "kc_pos": kc_pos,
                "kc_req": kc_req,
                "kc_raw_loc": kc_raw_loc,
                "kc_iloc": kc_iloc,
                "kc_cache": kc_cache,
                "kc_page_bits": kc_page_bits,
                "kc_page_bytes": kc_page_bytes,
                "kc_wk": kc_wk,
                "kc_knorm": kc_knorm,
                "kc_keps": kc_keps,
                "kc_payload": kc_payload,
                "kc_pscale": kc_pscale,
                "kc_wide": kc_wide,
                "kc_live": kc_live,
            }
            run_k1(key, lds.seam.peek(), lds.front.peek(), a, _epoch(epoch))

    @flyc.jit
    def launch(
        res_in: Int64,
        pend: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        attn_w: Int64,
        res_out: Int64,
        aux_out: Int64,
        aux_stride: Int32,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        wqkv: Int64,
        wqkv_s: Int64,
        qn_w: Int64,
        kvn_w: Int64,
        wqb: Int64,
        wqb_s: Int64,
        cos_sin: Int64,
        pos: Int64,
        slot: Int64,
        slot_words: Int32,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        swa_idx: Int64,
        swa_lens: Int64,
        topk: Int64,
        topk_lens: Int64,
        kt: Int64,
        klen: Int64,
        q: Int64,
        normed_out: Int64,
        qr_out: Int64,
        iq_w: Int64,
        iq_ws: Int64,
        iw_w: Int64,
        iw_scale: Int32,
        num_pos: Int32,
        iq_out: Int64,
        iqs_out: Int64,
        iw_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        ids: Int64,
        ids_words: Int32,
        cp_w: Int64,
        cp_out: Int64,
        kc_per_req: Int32,
        kc_state: Int64,
        kc_ring: Int32,
        kc_norm: Int64,
        kc_eps: Int32,
        kc_freqs: Int64,
        kc_pos: Int64,
        kc_req: Int64,
        kc_raw_loc: Int64,
        kc_iloc: Int64,
        kc_cache: Int64,
        kc_page_bits: Int32,
        kc_page_bytes: Int32,
        kc_wk: Int64,
        kc_knorm: Int64,
        kc_keps: Int32,
        kc_payload: Int64,
        kc_pscale: Int64,
        kc_wide: Int32,
        kc_live: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            res_in,
            pend,
            post_in,
            comb_in,
            pre_in,
            hc_fn,
            hc_scale,
            hc_base,
            attn_w,
            res_out,
            aux_out,
            aux_stride,
            post_out,
            comb_out,
            pre_out,
            wqkv,
            wqkv_s,
            qn_w,
            kvn_w,
            wqb,
            wqb_s,
            cos_sin,
            pos,
            slot,
            slot_words,
            swa,
            swa_stride,
            swa_block,
            swa_idx,
            swa_lens,
            topk,
            topk_lens,
            kt,
            klen,
            q,
            normed_out,
            qr_out,
            iq_w,
            iq_ws,
            iw_w,
            iw_scale,
            num_pos,
            iq_out,
            iqs_out,
            iw_out,
            scratch,
            sym,
            peers,
            rank,
            epoch,
            ids,
            ids_words,
            cp_w,
            cp_out,
            kc_per_req,
            kc_state,
            kc_ring,
            kc_norm,
            kc_eps,
            kc_freqs,
            kc_pos,
            kc_req,
            kc_raw_loc,
            kc_iloc,
            kc_cache,
            kc_page_bits,
            kc_page_bytes,
            kc_wk,
            kc_knorm,
            kc_keps,
            kc_payload,
            kc_pscale,
            kc_wide,
            kc_live,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


# ---------------------------------------------------------------- the FFN launch


@flyc.jit
def store_normed(normed, t, col, ys, live):
    """Store the FFN seam's norm output as the MoE input row (bf16, device scope), the 8
    values in one 16 B store: as 8 stores of 2 B they held the norm's publish back."""
    words = [
        fx.Vector.from_elements([ys[2 * i], ys[2 * i + 1]], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
        for i in range(4)
    ]
    if live:
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(normed),
            (t * HIDDEN + col) // 2,
            cache_modifier=CM_DEV,
        )


def row_cells(key: MonoBuild) -> bool:
    """Whether token-parallel wo_b partials reach their rows' owners as ACELL cells (four bf16
    values in 8 B, no tag: ``seam.attn_row_cell``) instead of tagged pairs (two in 8 B):
    TP2, where the partials of the peer's rows all cross one link."""
    return key.tok_par and key.tp == 2


@flyc.jit
def push_attn_pair(c, bases, own, t, col, vs, acell=None):
    """Push this rank's wo_b partial of row t at ``col`` to the row's owner (token
    parallel, ``own`` rows a rank): a bf16 pair at its row of the owner's ATTN region, or
    with ``acell`` (ACELL's byte offset, ``row_cells``) a cell of four values at the
    owner's ``seam.attn_row_cell``."""
    for p in range_constexpr(len(bases)):
        if t // own == p:
            if const_expr(acell is None):
                pair = seam.attn_region_pair(c["rank"], t - p * own, own, col)
                c["put_bf"](preg(bases[p], 0, "attn"), 2 * pair, vs)
            else:
                at = seam.attn_row_cell(c["rank"], t - p * own, own, col)
                bo.buffer_store(
                    seam.cell_words(vs),
                    rsrc(bases[p] + fx.Int64(acell)),
                    at * 2,
                    cache_modifier=CM_SYS,
                )


@flyc.jit
def push_attn_cell(c, bases, off, s, task, t0, u, vs):
    """Push this rank's wo_b cell u of ``task`` (the token tile from row t0) to every
    rank's ACELL region at byte ``off`` (``seam.attn_cell``)."""
    w0 = bf16_pair(vs[0], vs[1]).bitcast(fx.Int32)
    w1 = bf16_pair(vs[2], vs[3]).bitcast(fx.Int32)
    # never an empty word: a pair of bf16 0xFFFF NaNs goes out as 0xFFFE, 0xFFFF
    w0 = (w0 == seam.CELL_EMPTY).select(fx.Int32(seam.CELL_EMPTY - 1), w0)
    w1 = (w1 == seam.CELL_EMPTY).select(fx.Int32(seam.CELL_EMPTY - 1), w1)
    at = seam.attn_cell(c["rank"], task, s, t0, u)
    for p in range_constexpr(len(bases)):
        bo.buffer_store(
            fx.Vector.from_elements([w0, w1], fx.Int32),
            rsrc(bases[p] + fx.Int64(off)),
            at * 2,
            cache_modifier=CM_SYS,
        )


@flyc.jit
def push_head_pair(c, bases, off, t, h, vs):
    """Row t's index head weights h, h + 1 to every other rank's IWG."""
    for p in range_constexpr(c["d"].tp):
        if c["rank"] != p:
            c["put_bf"](preg(bases[p], off, "iwg"), t * index_query.HEADS + h, vs)


@flyc.jit
def stage_xg_row(xr, col, ys, live):
    """A normed row's 8 bf16 values at ``col`` into the LDS row ``xr``, packed as
    ``store_normed`` packs them (``push_bf_row`` sends the row)."""
    words = [
        fx.Vector.from_elements([ys[2 * i], ys[2 * i + 1]], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
        for i in range(4)
    ]
    if live:
        fx.ptr_store(fx.Vector.from_elements(words, fx.Int32), xr + col // 2)


def xg_slots(tid):
    """This thread's tagged words of a gathered row: [(word, live)]."""
    out = []
    for i in range(-(-XG_WORDS // THREADS)):
        w = tid + THREADS * i
        out.append((fx.min(w, XG_WORDS - 1), w < XG_WORDS))
    return out


@flyc.jit
def push_bf_row(c, bases, off, t, row, tag=None):
    """Row t to every other rank's peer region ``off``. With ``tag`` (``tagged_rows``):
    the row's bf16 pairs from the LDS row ``row`` (``stage_xg_row``) as ``XG_WORDS`` tagged
    16-B words, three pairs and ``tag`` each (a 16-B store is seen whole, so a word with the
    tag is this launch's), 3/4 of the bytes on the link where (value, tag) pairs take 1/2.
    Else ``row`` = (col, ys, live): its 8 bf16 values at ``col`` as (pair, tag) words."""
    if const_expr(tag is None):
        col, ys, live = row
        if live:
            for p in range_constexpr(c["d"].tp):
                if c["rank"] != p:
                    dst = preg(bases[p], off, "gather")
                    for h in range_constexpr(2):
                        at = t * HIDDEN + 2 * half_pair(HIDDEN // 2, col // 8, h)
                        c["put_bf"](dst, at, ys[4 * h : 4 * h + 4])
    else:
        for w, live in xg_slots(c["tid"]):
            if live:
                d = [fx.Int32(fx.ptr_load(row + (3 * w + e))) for e in range(3)]
                word = fx.Vector.from_elements(d + [tag], fx.Int32)
                for p in range_constexpr(c["d"].tp):
                    if c["rank"] != p:
                        bo.buffer_store(
                            word,
                            rsrc(bases[p] + fx.Int64(off)),
                            (t * XG_WORDS + w) * 4,
                            cache_modifier=CM_SYS,
                        )


@flyc.jit
def gather_row(c, src, t, dst):
    """Row t of a bf16 [row][HIDDEN] peer region ``src`` -> ``dst`` (device scope)."""
    tid = c["tid"]
    for k in range_constexpr(-(-(HIDDEN // 4) // THREADS)):
        q = tid + THREADS * k
        if q < HIDDEN // 4:
            (w,) = c["poll"]([(src, t * (HIDDEN // 2) + 2 * q, 2)])
            bo.buffer_store(
                fx.Vector.from_elements([w[0], w[1]], fx.Int32),
                rsrc(dst),
                t * (HIDDEN // 2) + chunk_pairs(HIDDEN // 2, q),
                cache_modifier=CM_DEV,
            )


@flyc.jit
def gather_words(c, src, t, dst, tag):
    """Row t of the tagged words (``push_bf_row``) of the peer region at ``src`` -> its
    bf16 [row][HIDDEN] row of ``dst`` (device scope)."""
    slots = xg_slots(c["tid"])

    def load_all():
        out = []
        for w, _ in slots:
            v = fx.Vector(
                bo.buffer_load(
                    rsrc(src),
                    (t * XG_WORDS + w) * 4,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_SYS,
                )
            )
            out += [v[e] for e in range(4)]
        return fx.Vector.from_elements(out, fx.Int32)

    def pending(v):
        bad = None
        for i, (_, live) in enumerate(slots):
            b = live & (v[4 * i + 3] != tag)
            bad = b if bad is None else bad | b
        return bad

    v = _spin(load_all, pending)
    for i, (w, live) in enumerate(slots):
        for e in range_constexpr(3):
            if live & (3 * w + e < HIDDEN // 2):
                bo.buffer_store(
                    v[4 * i + e],
                    rsrc(dst),
                    t * (HIDDEN // 2) + 3 * w + e,
                    cache_modifier=CM_DEV,
                )


def _seam_lds(s, tp):
    if tagged_rows(tp):

        @fx.struct
        class SeamLds:
            red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
            rl: fx.Array[fx.Float32, s * seam.KT, 16]
            fl: fx.Array[fx.Float32, seam.MIX * seam.KT, 16]
            pl: fx.Array[fx.Float32, s * seam.COLS, 16]
            # a normed row for the other ranks (token parallel, ``push_bf_row``)
            xr: fx.Array[fx.Int32, 3 * XG_WORDS, 16]

    else:

        @fx.struct
        class SeamLds:
            red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
            rl: fx.Array[fx.Float32, s * seam.KT, 16]
            fl: fx.Array[fx.Float32, seam.MIX * seam.KT, 16]
            pl: fx.Array[fx.Float32, s * seam.COLS, 16]

    return SeamLds


@flyc.jit
def run_ffn(
    key: MonoBuild,
    lds,
    a: dict,
    amb,
    bases,
    own,
    rank,
    ep,
    part=None,
    sn=None,
    pre=None,
    chain=None,
):
    """Run the FFN seam, the MoE and its all-reduce into ``a["out"]``; ``sn``
    (``key.seam_next``, ``SEAM_ARGS``): the all-reduce into the next layer's attention seam
    slice instead, whose outputs go to the seam state ``sn["seam_out"]``. ``pre``: the
    slice's first loads, issued before wo_b (``early_slice``). ``chain`` (``key.chain``):
    traced (more, last); where a layer follows in this launch, its seam slices run the
    all-reduce (``chained_slice``) and this MoE stops at its down."""
    assert (sn is not None) == key.seam_next, key
    assert (chain is not None) == key.chain, key
    assert not key.seam_next or not key.tok_par, key
    s, tp = key.tokens, key.tp
    layout = scratch_layout(s, tp, key.timeline, **route_of(key))
    mkey = _moe_key(key)
    rs = route_shape(mkey)
    # token parallel: the seam, norm and gates on this rank's rows
    own_n = s // tp if const_expr(key.tok_par) else s
    GATE0 = SLICES
    NORM0 = SLICES + own_n
    GATHER0 = NORM0 + own_n  # the other ranks' MoE input rows (token parallel)
    assert not key.tok_par or GATHER0 + s <= BLOCKS
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    sl = lds.seam.peek()
    scratch = a["scratch"]
    seam_args = (
        "res_in",
        "post_in",
        "comb_in",
        "pre_in",
        "hc_fn",
        "hc_scale",
        "hc_base",
        "res_out",
        "post_out",
        "comb_out",
        "pre_out",
    )
    ca = {
        "S": s,
        "tid": tid,
        "bid": bid,
        "lane": tid % 64,
        "wave": tid // 64,
        "rl": sl.rl.ptr,
        "fl": sl.fl.ptr,
        "red": sl.red.ptr,
        "pend_lds": sl.pl.ptr,
        "put": amb.put,
        "put_bf": amb.put_bf,
        "put_words": amb.put_words,
        "poll": amb.poll,
        "peer_addr": lambda p: bases[p],
        "rank": rank,
        "sym": own,
        "args": {n: a[n] for n in seam_args} | {"norm_w": a["ffn_w"]},
        "d": StageDims(tp),
        # SGLang's Triton boundary + rmsnorm arithmetic (seam.slice_row, stage_norm)
        "sgl": True,
    }
    if const_expr(key.timeline):
        ca["tl"] = timeline_row(scratch, layout["timeline"][0], bid)
    stamp(ca, 16)
    t0 = 0
    if const_expr(key.tok_par):
        t0 = rank * own_n
        ca = own_view(ca, t0, own_n)
        xg = peer_offsets(s, tp)["xg"]
    normed = scratch + fx.Int64(layout["normed"][0])
    for region in ("lin", "pmix"):
        ca[region] = sreg(scratch, layout[region][0], region)
    mlayout = {
        n: layout[MOE_X8.get(n, n)]
        for n in moe.scratch_layout(mkey)
        if n not in COUNTERS
    }
    mlayout["xrdy"] = layout["xrdy_moe"]
    margs = moe.moe_args(
        normed,
        a["gate_w"],
        a["bias"],
        a["w13"],
        a["w13_s"],
        a["w2"],
        a["w2_s"],
        a["sgu"],
        a["sgu_s"],
        a["sw2"],
        a["sw2_s"],
        a["out"],
    )
    cb = moe.moe_context(
        s, rs, lds.moe.peek(), amb, bases, mlayout, scratch, margs, rank, own
    )
    cb["tag"] = ep & 255
    cb["fcell_off"] = fcell_offset(tp)
    if const_expr(key.timeline):
        cb["tl"] = ca["tl"]
    for i, region in enumerate(COUNTERS):
        cb[region] = sreg(a["epoch"], 4 * (EPOCH_COUNTERS + i * moe.TAGS), region)
    # the next layer's seam slice sums the down partials itself (``seam_next``); in a
    # multi-layer launch, where a layer follows (traced)
    cb["seam_next"] = key.seam_next
    if const_expr(key.chain):
        cb["reduce"] = chain[1]
    frdy = sreg(scratch, layout["frdy"][0], "frdy")
    if const_expr(key.tok_par):
        # the down sums reduce-scattered: each rank sums its own rows into ``out``
        cb["own"] = own_n
        out_own = a["out"] + fx.Int64(t0) * fx.Int64(HIDDEN * 2)
        cb["own_view"] = {**cb, "S": own_n, "args": {**margs, "out": out_own}}
    grdy = a.get("grdy")
    if const_expr(grdy is not None and pre is None):
        # one launch per layer: the slices read the mixes K1's gates wrote (the CTAs that
        # waited on wo_a's flags polled them there)
        poll = amb.poll
        if (bid < SLICES) & a["grdy_late"]:
            if tid < own_n:
                poll([(grdy, tid * FLAG_STRIDE, 1)])
            gpu.barrier()
    if const_expr(part is None and row_cells(key)):
        # K2's partials of this rank's rows arrive as cells (``push_attn_pair``)
        ca["acell"] = own + fx.Int64(acell_offset(tp))
        ca["row_cells"] = True
    if const_expr(part is None and not key.tok_par):
        # K2's partials arrive as cells (``push_attn_cell``)
        ca["acell"] = own + fx.Int64(acell_offset(tp))
        if const_expr(early_slice(key)):
            # polled on wave 1: wave 0 pushed wo_b's cells, and a wave's first wait also
            # waits on its own pushes' acks
            ca["cell0"] = 64
    # read before the loop: a method call on ca inside one carries ca
    reset = ca.get("cell0") is not None
    for task in range(first_task(bid, 0), SLICES, BLOCKS):
        if const_expr(pre is None):
            # the slice's own loads go out ahead of the wait on the ranks' partials, where
            # they fit in registers beside the reduce
            got = (
                seam.slice_prefetch(ca, task)
                if seam.slice_prefetch_fits(own_n)
                else None
            )
        else:
            got = pre
        if const_expr(part is not None):
            seam.stage_push(ca, task, part)
        seam.stage_reduce(ca, task)
        seam.stage_slice(ca, task, got)
        if const_expr(reset):
            seam.reset_attn_cells(ca, task)
    stamp(ca, 17)

    xr = sl.xr.ptr if tagged_rows(tp) else None

    def store(t, col, ys, live):
        store_normed(normed, plus(t0, t), col, ys, live)
        if const_expr(key.tok_par and tagged_rows(tp)):
            # the row for every other rank's MoE (``push_bf_row``, ``gather_words``)
            stage_xg_row(xr, col, ys, live)
        elif const_expr(key.tok_par):
            # the row for every other rank's MoE (``gather_row``)
            push_bf_row(ca, bases, xg, plus(t0, t), (col, ys, live))

    xbox = scratch + fx.Int64(layout["xbox"][0])
    cb["xbox"], cb["xtag"] = xbox, amb.tag

    def x_ready(t):
        # the flag (the drain first), then the router CTAs' box word
        publish(cb["put"], cb["xrdy"], t, 1, tid == 0)
        moe.signal_xbox(cb, xbox, t, amb.tag)

    for t in range(first_task(bid, NORM0), own_n, BLOCKS):
        seam.stage_norm(ca, t, store)
        if const_expr(key.tok_par and tagged_rows(tp)):
            # the LDS row is whole past the norm's barrier
            push_bf_row(ca, bases, xg, plus(t0, t), xr, amb.tag)
        x_ready(plus(t0, t))
    if const_expr(key.tok_par):
        for t in range(first_task(bid, GATHER0), s, BLOCKS):
            if t // own_n != rank:
                if const_expr(tagged_rows(tp)):
                    gather_words(ca, own + fx.Int64(xg), t, normed, amb.tag)
                else:
                    gather_row(ca, preg(own, xg, "xg"), t, normed)
                x_ready(t)
    gpu.barrier()
    stamp(ca, 18)

    # the next layer's seam slice reads the mixes in this launch: device scope, flagged
    flagged = key.seam_next or key.chain
    cg = {**ca, "dev": True} if const_expr(flagged) else ca

    def gates():
        # only the next layer's seam reads the mixes; later MoE stages reload the LDS the
        # gate clobbers
        gpu.barrier()
        for t in range(first_task(bid, GATE0), own_n, BLOCKS):
            seam.stage_gate(cg, t)
            if const_expr(flagged):
                publish(amb.put, frdy, t * FLAG_STRIDE, 1, tid == 0)
        stamp(ca, 23)

    moe.run_moe(cb, mkey, bid, before_ug=gates)
    stamp(ca, 27)
    if const_expr(key.seam_next):
        # the next layer's seam slice past the MoE: its loads issued after the down's, which
        # they would delay, and in flight through the wait on the ranks' partials
        loads, run = seam_tail(key, ca, cb, a, sn, frdy)
        gpu.barrier()
        for task in range(first_task(bid, 0), SLICES, BLOCKS):
            run(task, loads(task))


def mix_first(key: MonoBuild) -> int:
    """A seamed launch's first CTA of the mixes' partials (``seam_gates``): the first past the
    attention back's split, combine and wo_a units. The split and combine run before the idle
    hook; a wo_a CTA would issue its weights only past the partials, and wo_a's flags hold
    the attention back's chain."""
    bkey = _back_key(key)
    woa0 = mb_back.back_plan(bkey)[4]
    first = woa0 + mb_back.woa_units(key.tokens, Dims(key.tp))
    assert first + key.tokens < BLOCKS, key
    return first


def seam_gates(key: MonoBuild, lds, a, gin, layout, ep, amb):
    """A seamed launch's attention gates (``key.seamed``), as the idle work of the attention
    back's CTAs past its split, combine and wo_a units: slice task k's mixes' partials
    (``seam.stage_mix``, from the R_new rows the previous launch stored) on CTA
    ``mix_first`` + k, a CTA after another past the grid's last; then token t's gate on CTA
    BLOCKS - s + t, which polls the partials and flags its mixes for the FFN seam's slices
    (``grdy``, K2's tag ``amb``). The partials go out as tagged pairs with K1's tag, device
    scope, in a region the FFN seam rewrites with K2's only past those flags."""
    s = key.tokens
    M0 = mix_first(key)
    ctas = BLOCKS - M0
    sl = lds.seam.peek()
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    k1mb = Mailbox((ep << 1) + 1)
    cg = {
        "S": s,
        "tid": tid,
        "lane": tid % 64,
        "wave": tid // 64,
        "rl": sl.rl.ptr,
        "fl": sl.fl.ptr,
        "red": sl.red.ptr,
        "put": k1mb.put,
        "poll": k1mb.poll,
        "pmix": sreg(a["scratch"], layout["pmix"][0], "pmix"),
        "dev": True,
        "args": {
            "res_in": a["res_in"],
            "hc_fn": gin["hc_fn"],
            "hc_scale": gin["hc_scale"],
            "hc_base": gin["hc_base"],
            "pre_out": a["pre_in"],
            "post_out": a["post_in"],
            "comb_out": a["comb_in"],
        },
    }
    if const_expr(key.timeline):
        cg["tl"] = timeline_row(a["scratch"], layout["timeline"][0], bid)
    grdy = sreg(a["scratch"], layout["grdy"][0], "grdy")
    return lambda: run_seam_gates(cg, bid, M0, ctas, grdy, amb.put)


@flyc.jit
def run_seam_gates(cg, bid, M0, ctas, grdy, put):
    """``seam_gates``' work on CTA ``bid``."""
    s, tid = cg["S"], cg["tid"]
    if bid >= M0:
        for task in range(bid - M0, seam.SLICES, ctas):
            seam.stage_mix(cg, task)
    for t in range(first_task(bid, BLOCKS - s), s, BLOCKS):
        seam.stage_gate(cg, t)
        publish(put, grdy, t * FLAG_STRIDE, 1, tid == 0)


def early_slice(key: MonoBuild) -> bool:
    """Whether the FFN seam slice's first loads go out before wo_b, so a slice CTA's poll on
    the ranks' wo_b cells has no loads ahead of it, and the cells are polled on a wave that
    pushed none: TP4 and TP8, the cells' path, the loads in registers."""
    return key.tp in (4, 8) and not key.tok_par and seam.slice_prefetch_fits(key.tokens)


TAIL_SUM0 = 128  # the seam tail's first summing thread


def split_first(key: MonoBuild) -> int:
    """The first CTA of the attention back's split units in one launch a layer: on TP4,
    where the front's wqkv row tiles run as row halves, the grid's last CTAs, whose K1
    half (no seam slice, the norms', gates', qkv's and the CTAs without wqkv) ends first,
    so their old keys' records go out sooner; else CTA 0."""
    s = key.tokens
    if key.tp == 4 and not (key.tok_par or key.eg or key.glue) and wqkv_split(s, s):
        return BLOCKS - mb_back.back_plan(_back_key(key))[1]
    return 0


def seam_tail(key: MonoBuild, ca, cb, a, sn, frdy):
    """The next layer's attention seam slice, task k on CTA k past its MoE work, as
    (``loads``, ``run``): ``loads(task)`` waits on the FFN gates' mixes and issues the
    slice's own loads; ``run(task, pre)`` sums the ranks' down partials of columns 32 k ..
    in rank order (the MoE reduce's sum, bf16) into the pend LDS, then folds them into the
    next layer's R_new and computes its layer input, both stored plain into the seam state
    ``seam_out``."""
    s, tp = key.tokens, key.tp
    assert TAIL_SUM0 + s * seam.COLS // 2 <= THREADS, s
    st = seam.seam_state_layout(s)
    out = sn["seam_out"]
    cn = {
        **ca,
        "put_bf": seam.plain_put_bf,
        "lin": out + fx.Int64(st["lin"][0]),
        # the next launch computes the mixes' partials from R_new (``seam_gates``)
        "mixes": seam.MIX_NONE,
        "args": {
            "res_in": a["res_out"],
            "post_in": a["post_out"],
            "comb_in": a["comb_out"],
            "pre_in": a["pre_out"],
            "res_out": out + fx.Int64(st["res"][0]),
            "aux_out": sn["nx_aux"],
            "aux_stride": sn["nx_aux_stride"],
        },
        "fold": True,
        "aux_on": sn["nx_aux"] != 0,
    }
    if const_expr(cb["rs"].row_moe):
        ffn = preg(cb["sym"], cb["fcell_off"], "fcell")
    else:
        ffn = preg(cb["sym"], cb["ffn_off"], "ffn")
    return (
        lambda task: tail_loads(ca, cn, frdy, s, task),
        lambda task, pre: tail_run(ca, cn, ffn, s, tp, task, pre),
    )


@flyc.jit
def tail_loads(ca, cn, frdy, s, task):
    """``seam_tail``'s loads: the FFN gates' mixes (post, comb, next pre) waited on first,
    the slice's rows read them; then the slice's own loads. The flags are polled on wave 1:
    wave 0 pushed the down sums, and a wave's first wait also waits on its own pushes' acks."""
    stamp(ca, 29)
    u = ca["tid"] - 64
    if (u >= 0) & (u < s):
        ca["poll"]([(frdy, u * FLAG_STRIDE, 1)])
    gpu.barrier()
    return seam.slice_prefetch(cn, task)


def tail_cells(s, tp, task, u):
    """Summing thread u's FCELL cells: row u // 8, columns 32 task + 4 (u % 8) .., every rank's."""
    per = seam.COLS // seam.CELL_VALUES
    col = task * seam.COLS + seam.CELL_VALUES * (u % per)
    return [seam.ffn_cell(src, u // per, s, col) for src in range(tp)]


@flyc.jit
def tail_run(ca, cn, ffn, s, tp, task, pre, out=None):
    """``seam_tail``'s sums and slice. ``ffn``: FCELL (the per-row MoE: the summed cells
    emptied) or the tagged pairs' region; ``out``: the sums also stored there, bf16, as
    the MoE's own all-reduce stores them."""
    gpu.barrier()
    # the sums on waves 2 .. : wave 0 pushed the down partials, wave 1 polled the gates' flags
    u = ca["tid"] - TAIL_SUM0
    if const_expr(ffn.region == "fcell"):
        per = seam.COLS // seam.CELL_VALUES
        live = (u >= 0) & (u < s * per)
        if live:
            cells = tail_cells(s, tp, task, u)
            accs = seam.sum_cells(ffn.value, cells)
            cc = seam.CELL_VALUES * (u % per)
            for i in range_constexpr(seam.CELL_VALUES):
                at = ca["pend_lds"] + ((u // per) * seam.COLS + cc + i)
                fx.ptr_store(bf16_round(accs[i]), at)
                if const_expr(out is not None):
                    at = (u // per) * HIDDEN + task * seam.COLS + cc + i
                    bo.buffer_store(accs[i].to(fx.BFloat16), rsrc(out), at)
            # emptied here, not past the slice: the launch's end would wait on the stores
            seam.empty_cells(ffn.value, cells)
    else:
        if (u >= 0) & (u < s * seam.COLS // 2):
            t = u // (seam.COLS // 2)
            cc = 2 * (u % (seam.COLS // 2))
            col = task * seam.COLS + cc
            acc0, acc1 = sum_partials(
                ca["poll"], ffn, lambda src: moe.region_pair(src, t, s, col), tp
            )
            fx.ptr_store(bf16_round(acc0), ca["pend_lds"] + (t * seam.COLS + cc))
            fx.ptr_store(bf16_round(acc1), ca["pend_lds"] + (t * seam.COLS + cc + 1))
            if const_expr(out is not None):
                bo.buffer_store(acc0.to(fx.BFloat16), rsrc(out), t * HIDDEN + col)
                bo.buffer_store(acc1.to(fx.BFloat16), rsrc(out), t * HIDDEN + col + 1)
    gpu.barrier()
    stamp(ca, 31)
    seam.stage_slice(cn, task, pre)
    stamp(ca, 124)


def reset_counters(epoch, ep):
    """Clear the MoE counters' slot 128 launches ahead: its last use is long done."""
    slot = (ep + 128) & 255
    for i in range_constexpr(len(COUNTERS)):
        at = 4 * (EPOCH_COUNTERS + i * moe.TAGS + slot)
        gstore(epoch + fx.Int64(at), fx.Int32(0), words=1)


# A seam-move launch's CTAs past the seam tail's slices are idle once their MoE work is
# done: they load the next layer's front weights into the L2, which keeps clean lines
# across the launch boundary. A CTA runs on the same XCD in every launch of the grid, and
# CTAs c and c + 8 on the same XCD, so the front's 16-row tile read on CTA c (wqkv's row
# tile c % 112, wq_b's task c) goes to the L2 of a loading CTA = c (mod 8).
L2_FIRST = SLICES
L2_ROUND = (
    4096  # bytes a round: 16 B a thread of a CTA's half, each half its own rounds
)
L2_HALVES = THREADS * 16 // L2_ROUND
L2_DEPTH = 8  # loads in flight a CTA (64 KB)
L2_ARGS = ("nx_wqkv_s", "nx_wqb_s", "nx_wqkv", "nx_wqb")


def l2_weights(tp: int) -> list:
    """The next layer's front weights in load order, each (argument, its tile's bytes, or 0:
    the whole tensor into every XCD's L2 (the E8M0 blocks), its bytes)."""
    d = Dims(tp)
    return [
        ("nx_wqkv_s", 0, QKV_ROWS // 32 * HIDDEN // 32),
        ("nx_wqb_s", 0, d.q_rows // 32 * Q_RANK // 32),
        ("nx_wqkv", ROWS * HIDDEN, QKV_ROWS * HIDDEN),
        ("nx_wqb", ROWS * Q_RANK, d.q_rows * Q_RANK),
    ]


def l2_plan(tp: int) -> list:
    """Each weight's loads a loading CTA, as (argument, its bytes, the tile's bytes or 0,
    its rounds an XCD, a CTA's loads)."""
    per = (BLOCKS - L2_FIRST) // 8
    out = []
    for arg, tile, nbytes in l2_weights(tp):
        if tile:
            assert tile % L2_ROUND == 0 and nbytes % (8 * tile) == 0, arg
            rounds = nbytes // tile // 8 * (tile // L2_ROUND)
        else:
            rounds = -(-nbytes // L2_ROUND)
        out.append((arg, nbytes, tile, rounds, -(-rounds // (per * L2_HALVES))))
    return out


def keep_vectors(values):
    """``values`` (loads) waited for here, never used."""
    llvm.InlineAsmOp(
        None,
        [as_ir_value(v) for v in values],
        "",
        ",".join("v" * len(values)),
        has_side_effects=True,
    )


def l2_prefetch(key: MonoBuild) -> bool:
    """Whether a seam-move launch's idle CTAs load the next layer's front weights: where its
    MoE leaves them the time, up to 2 rows a token tile or past the per-row MoE's widths. At
    4-7 rows the per-row MoE holds them until the seam tail's end, and the loads would hold
    the launch's end."""
    return key.tokens <= 2 or not route_shape(_moe_key(key)).row_moe


@flyc.jit
def idle_tail(c, tp, bid):
    """A seam-move launch's end on a CTA past the seam tail's slices: ``front_l2``."""
    if bid >= L2_FIRST:
        nx = reload_args(L2_ARGS, kernarg_offsets(LAYER_ARGS, LAYER_I32), LAYER_I32)
        front_l2(c, nx, l2_plan(tp), bid)


def front_l2(c, nx, plan, bid):
    """Load the next layer's front weights (``nx``: ``L2_ARGS``) of this XCD's CTAs into
    its L2, default cache policy: CTA ``bid`` (past ``L2_FIRST``) the rounds j, j + 12 ..
    of each weight's rounds of the tiles = bid (mod 8), a half of the CTA every other one,
    64 KB in flight."""
    per = (BLOCKS - L2_FIRST) // 8
    xcd = bid % 8
    j = (bid - L2_FIRST) // 8
    half = c["tid"] // (L2_ROUND // 16)
    at16 = c["tid"] % (L2_ROUND // 16) * 16
    pending = []
    for arg, nbytes, tile, rounds, loads in plan:
        res = rsrc(nx[arg], fx.Int64(nbytes))
        for i in range_constexpr(loads):
            u = j + per * (L2_HALVES * i + half)
            if const_expr(tile):
                spt = tile // L2_ROUND
                off = (xcd + 8 * (u // spt)) * tile + (u % spt) * L2_ROUND
            else:
                off = u * L2_ROUND
            # past the weight's rounds: out of bounds, no traffic
            off = (u < rounds).select(off + at16, fx.Int32(nbytes))
            pending.append(bo.buffer_load(res, off // 4, vec_width=4, dtype=T.i32))
            if const_expr(len(pending) == L2_DEPTH):
                keep_vectors(pending)
                pending = []
    if const_expr(pending):
        keep_vectors(pending)
    stamp(c, 173)


def build_mono_ffn(key: MonoBuild):
    """Build the FFN launch: push this rank's wo_b partial, then ``run_ffn``."""
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    assert not key.tok_par, "the FFN launch's partials arrive whole"
    rs = route_shape(_moe_key(key))
    assert rs.experts // 64 in SORT_NETS
    MoeLds = moe.moe_smem(s, rs)
    half = peer_half_bytes(tp)
    assert SLICES + 2 * s <= BLOCKS
    SeamLds = _seam_lds(s, tp)

    @fx.union
    class FfnLds:
        seam: SeamLds  # type: ignore[valid-type]
        moe: MoeLds  # type: ignore[valid-type]

    name = kernel_symbol("dsv41_mono_ffn", s=s, tp=tp, **_variant(key))
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kffn(
        part: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        ffn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
    ):
        _ = keyed
        # every wait runs under the guard, so a timeout flags every peer and stops the launch
        with guard.scope(
            guard.Guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=tp,
                flags_offset=2 * half,
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            )
        ):
            lds = fx.SharedAllocator().allocate(FfnLds)
            ep = _epoch(epoch)
            par = fx.Int64(ep & 1) * fx.Int64(half)
            bases = [b + par for b in peer_bases(peers, tp)]
            ctx = {"bid": fx.block_idx.x, "tid": fx.thread_idx.x}
            epoch_begin(ctx, epoch, ep)
            # the last CTA moves the epoch on once every CTA marked it
            epoch_end(ctx, epoch, ep, lambda: reset_counters(epoch, ep), who=BLOCKS - 1)
            a = dict(
                res_in=res_in,
                post_in=post_in,
                comb_in=comb_in,
                pre_in=pre_in,
                hc_fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                ffn_w=ffn_w,
                res_out=res_out,
                post_out=post_out,
                comb_out=comb_out,
                pre_out=pre_out,
                gate_w=gate_w,
                bias=bias,
                w13=w13,
                w13_s=w13_s,
                w2=w2,
                w2_s=w2_s,
                sgu=sgu,
                sgu_s=sgu_s,
                sw2=sw2,
                sw2_s=sw2_s,
                out=out,
                scratch=scratch,
                epoch=epoch,
            )
            amb = Mailbox((ep << 1) + 2)
            run_ffn(key, lds, a, amb, bases, sym + par, rank, ep, part=part)

    @flyc.jit
    def launch(
        part: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        ffn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kffn(
            part,
            res_in,
            post_in,
            comb_in,
            pre_in,
            hc_fn,
            hc_scale,
            hc_base,
            ffn_w,
            res_out,
            post_out,
            comb_out,
            pre_out,
            gate_w,
            bias,
            w13,
            w13_s,
            w2,
            w2_s,
            sgu,
            sgu_s,
            sw2,
            sw2_s,
            out,
            scratch,
            sym,
            peers,
            rank,
            epoch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


# ---------------------------------------------------------------- K2

K2_ARGS = (
    "q",
    "swa",
    "swa_stride",
    "swa_block",
    "comp",
    "comp_stride",
    "comp_block",
    "kt",
    "klen",
    "slot",
    "slot_words",
    "swa_idx",
    "swa_lens",
    "topk",
    "topk_lens",
    "pos",
    "sink",
    "qk_scale",
    "cos_sin",
    "woa",
    "woa_s",
    "wob",
    "wob_s",
    "zrec",
    "res_in",
    "post_in",
    "comb_in",
    "pre_in",
    "hc_fn",
    "hc_scale",
    "hc_base",
    "ffn_w",
    "res_out",
    "post_out",
    "comb_out",
    "pre_out",
    "gate_w",
    "bias",
    "w13",
    "w13_s",
    "w2",
    "w2_s",
    "sgu",
    "sgu_s",
    "sw2",
    "sw2_s",
    "out",
    "scratch",
    "sym",
    "peers",
    "rank",
    "epoch",
)


@flyc.jit
def run_k2(
    key: MonoBuild,
    lds,
    a: dict,
    ep,
    one=False,
    eg=None,
    k1_first=None,
    pre=None,
    pbases=None,
    sn=None,
    gin=None,
    chain=None,
):
    """K2 on this CTA: the attention back on K1's q, its wo_b partial pushed to every
    rank, then ``run_ffn`` (``lds``: a union with K2's members). ``one``: one launch per
    layer, K1's half before it in the same launch (``build_mono_layer``; ``k1_first``: the
    first loads of a split CTA's K1 flags, issued as K1's half ended; ``pre``: its keys,
    ``back.EarlyKeys.loads``; ``pbases``: the peer bases, loaded at the launch's start). ``eg``: a later
    Engram layer's projection's arguments (``EG_ARGS``, ``key.eg``); ``sn``: the next
    layer's attention seam's (``SEAM_ARGS``, ``key.seam_next``); ``gin``: this layer's
    attention mix weights (``hc_fn``, ``hc_scale``, ``hc_base``, ``key.seamed``: its gates
    run here, ``seam_gates``); ``chain``: ``run_ffn``'s."""
    (
        q,
        swa,
        swa_stride,
        swa_block,
        comp,
        comp_stride,
        comp_block,
        kt,
        klen,
        slot,
        slot_words,
        swa_idx,
        swa_lens,
        topk,
        topk_lens,
        pos,
        sink,
        qk_scale,
        cos_sin,
        woa,
        woa_s,
        wob,
        wob_s,
        zrec,
        res_in,
        post_in,
        comb_in,
        pre_in,
        hc_fn,
        hc_scale,
        hc_base,
        ffn_w,
        res_out,
        post_out,
        comb_out,
        pre_out,
        gate_w,
        bias,
        w13,
        w13_s,
        w2,
        w2_s,
        sgu,
        sgu_s,
        sw2,
        sw2_s,
        out,
        scratch,
        sym,
        peers,
        rank,
        epoch,
    ) = (a[n] for n in K2_ARGS)
    s, tp = key.tokens, key.tp
    layout = scratch_layout(s, tp, key.timeline, **route_of(key))
    bkey = _back_key(key)
    half = peer_half_bytes(tp)
    bid = fx.block_idx.x
    tag = (ep << 1) + 2
    par = fx.Int64(ep & 1) * fx.Int64(half)
    pbs = peer_bases(peers, tp) if pbases is None else pbases
    bases = [b + par for b in pbs]
    own = sym + par
    bargs = {
        "q": q,
        "swa": swa,
        "swa_stride": swa_stride,
        "swa_block": swa_block,
        "comp": comp,
        "comp_stride": comp_stride,
        "comp_block": comp_block,
        "kt": kt,
        "klen": klen,
        "pos": pos,
        "sink": sink,
        "qk_scale": qk_scale,
        "cos_sin": cos_sin,
        "woa": woa,
        "woa_s": woa_s,
        "wob": wob,
        "wob_s": wob_s,
        "zrec": zrec,
    }
    blayout = {n: layout[n] for n in back_scratch(s, Dims(tp))}
    views = {"split": lds.split.peek(), "gemv": lds.gemv.peek()}
    c = mb_back.back_context(bkey, views, bargs, scratch, tag, blayout)
    if const_expr(key.timeline):
        c["tl"] = timeline_row(scratch, layout["timeline"][0], bid)
    amb = Mailbox(tag)

    cp = {"put_bf": amb.put_bf, "rank": rank}
    if const_expr(row_cells(key)):
        acell = acell_offset(tp)
        assert s * HIDDEN // seam.CELL_VALUES * 8 <= acell_region_bytes(tp)

        def wob_cell(task, t0, u, vs):
            # cell u of a task's tile: row u // 4, its columns 4 (u % 4) ..
            per_row = seam.ROWS // seam.CELL_VALUES
            t = plus(t0, u // per_row)
            col = task * seam.ROWS + seam.CELL_VALUES * (u % per_row)
            push_attn_pair(cp, bases, s // tp, t, col, vs, acell)

        c["wob_cell"] = wob_cell
    elif const_expr(key.tok_par):
        c["wob_out"] = lambda t, col, v0, v1: push_attn_pair(
            cp, bases, s // tp, t, col, [v0, v1]
        )
    else:
        acell = acell_offset(tp)
        c["wob_cell"] = lambda task, t0, u, vs: push_attn_cell(
            cp, bases, acell, s, task, t0, u, vs
        )
    ctx = {"bid": bid, "tid": fx.thread_idx.x}
    if const_expr(one):
        # K1's half wrote q, the records, the key table and the mixes: the split waits on
        # every CTA's q flags (each past the CTA's records, table and mixes), its token's
        # table flag and, token parallel, the records copied from the other ranks; the
        # combine on the table flag, the FFN seam's slices on the mixes'
        # tagged q (``tagged_q_split``): q in tagged words, each row's record behind its flag
        tq = tagged_q_split(Dims(tp), s, key.tok_par)
        regions = ("krdy", "qb", "vrdy") if key.tok_par else ("krdy", "qb")
        regions += ("vrdy", "qt") if tq else ()
        for region in regions:
            c[region] = scratch + fx.Int64(layout[region][0])
        c["k1_words"] = k1_words(layout, key.tok_par or tq)
        c["k1_first"] = k1_first
        c["pre"] = pre
        split0 = split_first(key)
        c["split0"] = split0

        def k1_seen(unit):
            # every CTA's q flag is in, each stored past its epoch read: move the epoch on
            if unit == 0:
                # unit 0's CTA: ``split_first``
                epoch_end(
                    ctx,
                    epoch,
                    ep,
                    lambda: reset_counters(epoch, ep),
                    who=split0,
                    marks=False,
                )

        c["k1_seen"] = k1_seen
        if const_expr(not early_slice(key)):
            # the slice CTAs that wait on wo_a's flags poll the gates' flags in the same wait:
            # in ``run_ffn`` the poll would queue behind their wo_b push's peer stores
            own_n = s // tp if const_expr(key.tok_par) else s
            if const_expr(tagged_x8b(Dims(tp), s)):
                # with X8B's words (``back.x8b_rows``), from the flags' own region
                grdy = scratch + fx.Int64(layout["grdy"][0])
                c["ardy_also"] = (bid < SLICES, own_n, grdy, lambda u: u * FLAG_STRIDE)
            else:
                gap = layout["grdy"][0] - layout["ardy"][0]
                assert gap >= 0 and gap % 8 == 0, gap
                c["ardy_also"] = (
                    bid < SLICES,
                    own_n,
                    lambda u: gap // 8 + u * FLAG_STRIDE,
                )
    else:
        epoch_begin(ctx, epoch, ep)
    if const_expr(key.index):
        # the indexer's top-k came after K1; each split task waits on its token's krdy
        c["krdy"] = scratch + fx.Int64(layout["krdy"][0])
        ckt = {
            "S": s,
            "tid": c["tid"],
            "dev": True,
            "args": {
                "slot": slot,
                "slot_words": slot_words,
                "swa_idx": swa_idx,
                "swa_lens": swa_lens,
                "topk": topk,
                "topk_lens": topk_lens,
                "kt": kt,
                "klen": klen,
            },
        }
        for t in range(first_task(bid, BLOCKS - s), s, BLOCKS):
            mb_front.stage_kt(ckt, t)
            mb_back.publish(c, c["krdy"], t)
    idle = None
    if const_expr(key.glue):
        # the selection's glue on the CTAs without attention or wo_a work
        assert not key.eg, key
        _, n_split, n_comb, _, woa0, _, _ = mb_back.back_plan(bkey)
        gfirst, gctas = engram.eg_plan(
            woa0, mb_back.woa_units(s, Dims(tp)), n_split, n_comb
        )
        gc = {"bid": bid, "lane": c["lane"], "wave": c["wave"]}

        def idle():
            index_glue.run_glue(gc, a, key.glue, s, gfirst, gctas)

    if const_expr(key.eg):
        # a later Engram layer's projection on the CTAs without a wo_a unit
        _, n_split, n_comb, _, woa0, _, _ = mb_back.back_plan(bkey)
        ec = engram.eg_context(
            eg, key.eg, bases, own, peer_offsets(s, tp)["eng"], tp, rank, amb,
            lds.eg.peek(),
        )  # fmt: skip
        ec["first"], ec["ctas"] = engram.eg_plan(
            woa0, mb_back.woa_units(s, Dims(tp)), n_split, n_comb
        )
        gc = {
            "tid": c["tid"], "bid": bid, "lane": c["lane"], "wave": c["wave"],
            "tl": c.get("tl"),
        }  # fmt: skip

        def idle():
            engram.run_eg(gc, ec, s)

    if const_expr(key.seamed):
        # the attention gates' inputs and the gates, ahead of any other idle work
        gates = seam_gates(key, lds, a, gin, layout, ep, amb)
        eg_idle = idle

        def idle():
            gates()
            if const_expr(eg_idle is not None):
                eg_idle()

    before_wob = None
    if const_expr(early_slice(key)):
        cs = {
            "S": s,
            "tid": c["tid"],
            "args": {
                "res_in": res_in,
                "post_in": post_in,
                "comb_in": comb_in,
                "pre_in": pre_in,
                "hc_fn": hc_fn,
            },
        }
        grdy = sreg(scratch, layout["grdy"][0], "grdy") if const_expr(one) else None
        poll, tid = amb.poll, c["tid"]

        def before_wob():
            # one launch per layer: the slice reads the mixes K1's gates wrote. Every CTA
            # loads a slice's rows, those past the slices a copy of slice bid - SLICES'
            if const_expr(grdy is not None):
                if tid < s:
                    poll([(grdy, tid * FLAG_STRIDE, 1)])
                gpu.barrier()
            return seam.slice_prefetch(cs, bid % SLICES)

    # TP8's seamed launches: wo_b polls the X8B words alone (their idle CTAs run the gates' inputs)
    c["x8_hints"] = not (tp == 8 and key.seamed)
    spre = mb_back.run_back(c, bkey, bid, idle, before_wob)
    if const_expr(not one):
        # the epoch moves on once every CTA marked it, off the launch's end: the last CTA
        # waits on its next stage's inputs (the FFN seam) a while
        epoch_end(ctx, epoch, ep, lambda: reset_counters(epoch, ep), who=BLOCKS - 1)
    gpu.barrier()
    a = dict(
        res_in=res_in,
        post_in=post_in,
        comb_in=comb_in,
        pre_in=pre_in,
        hc_fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        ffn_w=ffn_w,
        res_out=res_out,
        post_out=post_out,
        comb_out=comb_out,
        pre_out=pre_out,
        gate_w=gate_w,
        bias=bias,
        w13=w13,
        w13_s=w13_s,
        w2=w2,
        w2_s=w2_s,
        sgu=sgu,
        sgu_s=sgu_s,
        sw2=sw2,
        sw2_s=sw2_s,
        out=out,
        scratch=scratch,
        epoch=epoch,
    )
    if const_expr(one):
        a["grdy"] = sreg(scratch, layout["grdy"][0], "grdy")
        wob0 = mb_back.back_plan(bkey)[5]
        a["grdy_late"] = first_task(bid, wob0) >= mb_back.ardy_waiters(bkey)
    run_ffn(key, lds, a, amb, bases, own, rank, ep, sn=sn, pre=spre, chain=chain)
    if const_expr(key.eg):
        # every rank's share of the projection in, long since: into ``eg_kv``
        engram.unpack(gc, ec, s)
    if const_expr(key.seam_next and l2_prefetch(key)):
        # the layer launch's arguments (``L2_ARGS``)
        assert one, key
        idle_tail({n: c[n] for n in ("tid", "tl") if n in c}, tp, bid)


def _k2_members(s, tp, rs):
    back_m = mb_back.back_lds_members(s, Dims(tp))
    return back_m["split"], back_m["gemv"], _seam_lds(s, tp), moe.moe_smem(s, rs)


# K2's selection glue (``MonoBuild.glue``), past K2_ARGS in K2's own launch
K2_GLUE_ARGS = index_glue.ARGS


def build_mono_k2(key: MonoBuild):
    """Build K2: the attention back on K1's q, its wo_b partial pushed to every rank,
    then ``run_ffn``."""
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    rs = route_shape(_moe_key(key))
    assert rs.experts // 64 in SORT_NETS
    half = peer_half_bytes(tp)
    assert SLICES + 2 * s <= BLOCKS
    SplitLds, GemvLds, SeamLds, MoeLds = _k2_members(s, tp, rs)

    @fx.union
    class K2Lds:
        split: SplitLds  # type: ignore[valid-type]
        gemv: GemvLds  # type: ignore[valid-type]
        seam: SeamLds  # type: ignore[valid-type]
        moe: MoeLds  # type: ignore[valid-type]

    name = kernel_symbol(
        "dsv41_mono_k2", s=s, tp=tp, c=key.compressed, i=key.index, **_variant(key)
    )
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        q: Int64,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        comp: Int64,
        comp_stride: Int64,
        comp_block: Int32,
        kt: Int64,
        klen: Int64,
        slot: Int64,
        slot_words: Int32,
        swa_idx: Int64,
        swa_lens: Int64,
        topk: Int64,
        topk_lens: Int64,
        pos: Int64,
        sink: Int64,
        qk_scale: Int32,
        cos_sin: Int64,
        woa: Int64,
        woa_s: Int64,
        wob: Int64,
        wob_s: Int64,
        zrec: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        ffn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        gl_ids: Int64,
        gl_cols: Int64,
        gl_ids_out: Int64,
        gl_table: Int64,
        gl_bound: Int64,
        gl_ptab: Int64,
        gl_ptab_stride: Int32,
        gl_bounds: Int64,
        gl_owners: Int64,
        gl_raw: Int64,
    ):
        _ = keyed
        warm_kernargs(K2_KERNARG_BYTES)
        # every wait runs under the guard, so a timeout flags every peer and stops the launch
        with guard.scope(
            guard.Guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=tp,
                flags_offset=2 * half,
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            )
        ):
            lds = fx.SharedAllocator().allocate(K2Lds)
            a = {
                "q": q,
                "swa": swa,
                "swa_stride": swa_stride,
                "swa_block": swa_block,
                "comp": comp,
                "comp_stride": comp_stride,
                "comp_block": comp_block,
                "kt": kt,
                "klen": klen,
                "slot": slot,
                "slot_words": slot_words,
                "swa_idx": swa_idx,
                "swa_lens": swa_lens,
                "topk": topk,
                "topk_lens": topk_lens,
                "pos": pos,
                "sink": sink,
                "qk_scale": qk_scale,
                "cos_sin": cos_sin,
                "woa": woa,
                "woa_s": woa_s,
                "wob": wob,
                "wob_s": wob_s,
                "zrec": zrec,
                "res_in": res_in,
                "post_in": post_in,
                "comb_in": comb_in,
                "pre_in": pre_in,
                "hc_fn": hc_fn,
                "hc_scale": hc_scale,
                "hc_base": hc_base,
                "ffn_w": ffn_w,
                "res_out": res_out,
                "post_out": post_out,
                "comb_out": comb_out,
                "pre_out": pre_out,
                "gate_w": gate_w,
                "bias": bias,
                "w13": w13,
                "w13_s": w13_s,
                "w2": w2,
                "w2_s": w2_s,
                "sgu": sgu,
                "sgu_s": sgu_s,
                "sw2": sw2,
                "sw2_s": sw2_s,
                "out": out,
                "scratch": scratch,
                "sym": sym,
                "peers": peers,
                "rank": rank,
                "epoch": epoch,
                "gl_ids": gl_ids,
                "gl_cols": gl_cols,
                "gl_ids_out": gl_ids_out,
                "gl_table": gl_table,
                "gl_bound": gl_bound,
                "gl_ptab": gl_ptab,
                "gl_ptab_stride": gl_ptab_stride,
                "gl_bounds": gl_bounds,
                "gl_owners": gl_owners,
                "gl_raw": gl_raw,
            }
            run_k2(key, lds, a, _epoch(epoch))

    @flyc.jit
    def launch(
        q: Int64,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        comp: Int64,
        comp_stride: Int64,
        comp_block: Int32,
        kt: Int64,
        klen: Int64,
        slot: Int64,
        slot_words: Int32,
        swa_idx: Int64,
        swa_lens: Int64,
        topk: Int64,
        topk_lens: Int64,
        pos: Int64,
        sink: Int64,
        qk_scale: Int32,
        cos_sin: Int64,
        woa: Int64,
        woa_s: Int64,
        wob: Int64,
        wob_s: Int64,
        zrec: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        ffn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        gl_ids: Int64,
        gl_cols: Int64,
        gl_ids_out: Int64,
        gl_table: Int64,
        gl_bound: Int64,
        gl_ptab: Int64,
        gl_ptab_stride: Int32,
        gl_bounds: Int64,
        gl_owners: Int64,
        gl_raw: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            q,
            swa,
            swa_stride,
            swa_block,
            comp,
            comp_stride,
            comp_block,
            kt,
            klen,
            slot,
            slot_words,
            swa_idx,
            swa_lens,
            topk,
            topk_lens,
            pos,
            sink,
            qk_scale,
            cos_sin,
            woa,
            woa_s,
            wob,
            wob_s,
            zrec,
            res_in,
            post_in,
            comb_in,
            pre_in,
            hc_fn,
            hc_scale,
            hc_base,
            ffn_w,
            res_out,
            post_out,
            comb_out,
            pre_out,
            gate_w,
            bias,
            w13,
            w13_s,
            w2,
            w2_s,
            sgu,
            sgu_s,
            sw2,
            sw2_s,
            out,
            scratch,
            sym,
            peers,
            rank,
            epoch,
            gl_ids,
            gl_cols,
            gl_ids_out,
            gl_table,
            gl_bound,
            gl_ptab,
            gl_ptab_stride,
            gl_bounds,
            gl_owners,
            gl_raw,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


# ---------------------------------------------------------------- one launch per layer

LAYER_ARGS = K1_ARGS + (
    "comp",
    "comp_stride",
    "comp_block",
    "sink",
    "qk_scale",
    "woa",
    "woa_s",
    "wob",
    "wob_s",
    "zrec",
    "ffn_fn",
    "ffn_scale",
    "ffn_base",
    "ffn_w",
    "out_res",
    "out_post",
    "out_comb",
    "out_pre",
    "gate_w",
    "bias",
    "w13",
    "w13_s",
    "w2",
    "w2_s",
    "sgu",
    "sgu_s",
    "sw2",
    "sw2_s",
    "out",
    # a later Engram layer's projection (``MonoBuild.eg``): its rows, this launch's range
    # of this rank's wkv rows and codes, kv out, the range's first row
    "eg_x",
    "eg_w",
    "eg_ws",
    "eg_kv",
    "eg_row0",
    # the next layer's attention seam (``MonoBuild.seam_next``): its DSpark capture (0: none)
    # and the capture's row stride; the seam state this launch reads (``MonoBuild.seamed``)
    # and the one it writes
    "nx_aux",
    "nx_aux_stride",
    "seam_in",
    "seam_out",
    # the next layer's front weights, loaded into the L2 at a seam-move launch's end
    # (``front_l2``)
    "nx_wqkv",
    "nx_wqkv_s",
    "nx_wqb",
    "nx_wqb_s",
)
EG_ARGS = ("eg_x", "eg_w", "eg_ws", "eg_kv", "eg_row0")
SEAM_ARGS = ("nx_aux", "nx_aux_stride", "seam_out")
# the attention gate's mix weights (K1's), read again in K2's half of a seamed launch
GATE_ARGS = ("hc_fn", "hc_scale", "hc_base")
# K2's arguments in the layer launch: its rows in are K1's out, and K1's rope table
K2_OF_LAYER = {
    "res_in": "res_out",
    "post_in": "post_out",
    "comb_in": "comb_out",
    "pre_in": "pre_out",
    "hc_fn": "ffn_fn",
    "hc_scale": "ffn_scale",
    "hc_base": "ffn_base",
    "res_out": "out_res",
    "post_out": "out_post",
    "comb_out": "out_comb",
    "pre_out": "out_pre",
}


# the layer launch's 32-bit arguments (the others are 64-bit)
LAYER_I32 = frozenset(
    ("aux_stride", "slot_words", "swa_block", "iw_scale", "num_pos", "rank")
    + ("ids_words", "comp_block", "qk_scale", "eg_row0", "nx_aux_stride")
)


def kernarg_offsets(names, i32) -> dict:
    """Each argument's byte offset in the kernarg segment: in order, each at its size."""
    out, off = {}, 0
    for n in names:
        size = 4 if n in i32 else 8
        off = -(-off // size) * size
        out[n] = off
        off += size
    return out


def warm_kernargs(nbytes: int, row=None):
    """Load a word of every 64-byte line of the kernarg segment at the launch's start, in
    the entry's wait: the arguments LLVM loads again later (it rematerializes them under
    SGPR pressure) then hit the scalar cache instead of waiting on L2 where they are used.
    ``row``: a table row laid out as the segment, at that address, instead."""
    seg_ptr = ir.Type.parse("!llvm.ptr<4>")
    if row is None:
        seg = llvm.PtrToIntOp(
            T.i64,
            llvm.call_intrinsic(seg_ptr, "llvm.amdgcn.kernarg.segment.ptr", [], [], []),
        ).result
    else:
        seg = as_ir_value(fx.Int64(row))
    words = []
    for off in range(0, nbytes, 64):
        ptr = llvm.IntToPtrOp(
            seg_ptr, as_ir_value(fx.Int64(seg) + fx.Int64(off))
        ).result
        words.append(llvm.LoadOp(T.i32, ptr, alignment=4, invariant=True).result)
    llvm.InlineAsmOp(
        None, words, "", ",".join(["s"] * len(words)), has_side_effects=True
    )


def kernarg_bytes(names, i32) -> int:
    """The kernarg segment's size: its last argument's offset plus its size."""
    off = kernarg_offsets(names, i32)[names[-1]]
    return off + (4 if names[-1] in i32 else 8)


# the index K1's and K2's 32-bit arguments past the layer launch's
K1_KC_I32 = frozenset(
    f"kc_{n}"
    for n in (
        "per_req",
        "ring",
        "eps",
        "page_bits",
        "page_bytes",
        "keps",
        "wide",
        "live",
    )
)
K1_KERNARG_BYTES = kernarg_bytes(
    K1_ARGS + K1_CP_ARGS + K1_KC_ARGS, LAYER_I32 | K1_KC_I32
)
K2_KERNARG_BYTES = kernarg_bytes(K2_ARGS + K2_GLUE_ARGS, LAYER_I32 | {"gl_ptab_stride"})


def reload_args(names, offsets: dict, i32, row=None) -> dict:
    """Arguments ``names`` loaded again from the kernarg segment (``row``: from a table row
    laid out as the segment, at that address), its address through an empty asm first so
    LLVM keeps these loads apart from the entry's: an argument a later half of the launch
    reads then does not stay live (spilled) through the earlier half."""
    seg_ptr = ir.Type.parse("!llvm.ptr<4>")
    if row is None:
        seg = llvm.PtrToIntOp(
            T.i64,
            llvm.call_intrinsic(seg_ptr, "llvm.amdgcn.kernarg.segment.ptr", [], [], []),
        ).result
    else:
        seg = as_ir_value(fx.Int64(row))
    base = fx.Int64(llvm.inline_asm(T.i64, [seg], "", "=s,0", has_side_effects=True))
    out = {}
    for n in names:
        ptr = llvm.IntToPtrOp(seg_ptr, as_ir_value(base + fx.Int64(offsets[n]))).result
        if n in i32:
            out[n] = fx.Int32(
                llvm.LoadOp(T.i32, ptr, alignment=4, invariant=True).result
            )
        else:
            out[n] = fx.Int64(
                llvm.LoadOp(T.i64, ptr, alignment=8, invariant=True).result
            )
    return out


def k1_words(layout: dict, records: bool) -> dict:
    """The split's K1 flag regions (``back.k1_flag_load``, ``back.q_flag_load``) as i32
    offsets from the key-table flags'; ``records``: with the record flags'."""
    regions = ("qb", "vrdy") if records else ("qb",)
    return {r: (layout[r][0] - layout["krdy"][0]) // 4 for r in regions}


def layer_handoff(scratch, layout: dict, ep) -> dict:
    """K1's hand-off to the attention back of the same launch: flags with K2's tag in
    regions of their own (``mb_front.hand_off``)."""
    h = {"mb": mb_device.Mailbox((ep << 1) + 2)}
    for region in ("krdy", "vrdy", "qb", "grdy"):
        h[region] = scratch + fx.Int64(layout[region][0])
    return h


class _Ids:
    def __init__(self, x):
        self.x = x


class layer_ids:
    """Trace the stages inside on opaque copies of the thread and block ids, masked to their
    ranges: a layer of a multi-layer launch's loop then shares no lane math with the code
    outside it, so LLVM hoists none of it out of the loop into long-lived registers."""

    def __enter__(self):
        t = fx.Int32(
            llvm.inline_asm(
                T.i32,
                [fx.Int32(fx.thread_idx.x).ir_value()],
                "v_mov_b32 $0, $1",
                "=v,v",
                has_side_effects=True,
            )
        ) & (THREADS - 1)
        b = fx.Int32(
            llvm.inline_asm(
                T.i32,
                [fx.Int32(fx.block_idx.x).ir_value()],
                "s_mov_b32 $0, $1",
                "=s,s",
                has_side_effects=True,
            )
        ) & (BLOCKS - 1)
        self.hw = fx.thread_idx, fx.block_idx
        fx.thread_idx, fx.block_idx = _Ids(t), _Ids(b)

    def __exit__(self, *exc):
        fx.thread_idx, fx.block_idx = self.hw


# a multi-layer launch (``build_mono_layers``): up to XL_LAYERS layers; a layer's row of its
# table is the layer launch's kernarg segment (``LAYER_ARGS``); the launch's own arguments
# are its step's (every layer's) and its first layer's inputs (the others' are the layer
# before's outputs), so a table stays the same from step to step
XL_LAYERS = 8
# (and the later Engram layer's rows its layers project: one layer's, 0 for none)
XL_STEP = (
    "pos",
    "slot",
    "slot_words",
    "swa_idx",
    "swa_lens",
    "topk",
    "topk_lens",
    "eg_x",
)
XL_FIRST = ("res_in", "pend", "post_in", "comb_in", "pre_in")
XL_ARGS = (
    ("table", "count", "scratch", "sym", "peers", "rank", "epoch") + XL_STEP + XL_FIRST
)
XL_I32 = frozenset(("count", "rank", "slot_words"))
# a layer body's K1 arguments (``layer_body``'s ``a``)
BODY_ARGS = tuple(n for n in LAYER_ARGS if n not in SEAM_ARGS)
XL_ROW = -(-kernarg_bytes(LAYER_ARGS, LAYER_I32) // 64) * 64


def xl_row(values: dict) -> bytes:
    """One layer's row of a multi-layer launch's table, from its ``LAYER_ARGS`` values."""
    row = bytearray(XL_ROW)
    for n, off in kernarg_offsets(LAYER_ARGS, LAYER_I32).items():
        v = int(values[n])
        if n in LAYER_I32:
            struct.pack_into("<I", row, off, v & 0xFFFFFFFF)
        else:
            struct.pack_into("<Q", row, off, v & 0xFFFFFFFFFFFFFFFF)
    return bytes(row)


def _layer_lds(key: MonoBuild, rs):
    """A layer launch's LDS: K1's and K2's stages one at a time."""
    s, tp = key.tokens, key.tp
    SplitLds, GemvLds, SeamLds, MoeLds = _k2_members(s, tp, rs)
    FrontLds = mb_front.front_lds(s, Dims(tp))

    # the Engram projection's (``key.eg``), else one no larger than the others
    EgLds = engram.eg_lds(mb_front.tile_rows(s) if key.eg else 1)

    # K1's seam takes K2's (a prefix of its members)
    @fx.union
    class LayerLds:
        seam: SeamLds  # type: ignore[valid-type]
        front: FrontLds  # type: ignore[valid-type]
        split: SplitLds  # type: ignore[valid-type]
        gemv: GemvLds  # type: ignore[valid-type]
        moe: MoeLds  # type: ignore[valid-type]
        eg: EgLds  # type: ignore[valid-type]

    return LayerLds


def layer_body(key: MonoBuild, lds, a: dict, ep, pbases, load, chain=None):
    """A layer launch's K1, then its K2, on this CTA: ``a`` K1's arguments, ``load(names)``
    loads K2's (past K1's half, so they do not stay live through it), ``pbases`` the peer
    bases. ``chain``, a multi-layer launch's loop (``build_mono_layers``): traced
    (chained, more, last): the layer before ran in this launch, a layer follows, this is
    the launch's last."""
    s, tp = key.tokens, key.tp
    nchunk, n_split = mb_back.back_plan(_back_key(key))[:2]
    SPLIT0 = split_first(key)
    layout = scratch_layout(s, tp, key.timeline, **route_of(key))
    k2 = layer_handoff(a["scratch"], layout, ep)
    bid = fx.block_idx.x
    # this CTA's split unit (``split_first``)
    sunit = bid if SPLIT0 == 0 else first_task(bid, SPLIT0)
    early = mb_back.EarlyKeys(
        _back_key(key), key.swa_width, a, fx.min(sunit, n_split - 1), sunit < n_split
    )
    run_k1(
        key,
        lds.seam.peek(),
        lds.front.peek(),
        a,
        ep,
        k2,
        early,
        None if chain is None else chain[0],
    )
    # the K1 half's LDS is free
    gpu.barrier()
    # a split CTA's K2 inputs in issue order, past the barrier (it would drain them):
    # its K1 flags' first loads, then its old keys' records, in flight through the
    # wait on q
    if const_expr(tagged_q_split(Dims(tp), s, key.tok_par)):
        first = mb_back.q_flag_load(
            a["scratch"] + fx.Int64(layout["qt"][0]),
            k2["krdy"],
            k1_words(layout, True),
            fx.thread_idx.x,
            fx.min(sunit, n_split - 1) // nchunk,
            s,
            Dims(tp),
            # the waves past the poll's (the keys') issue none
            pred=(sunit < n_split) & (fx.thread_idx.x < 256),
        )()
    else:
        first = mb_back.k1_flag_load(
            k2["krdy"],
            k1_words(layout, key.tok_par),
            fx.thread_idx.x,
            fx.min(sunit, n_split - 1) // nchunk,
            s,
            pred=sunit < n_split,
        )()
    pre = early.loads()
    names = [K2_OF_LAYER.get(n, n) for n in K2_ARGS]
    names += list(EG_ARGS) if key.eg else []
    names += list(SEAM_ARGS) if key.seam_next else []
    # a seamed launch's attention gates run in K2's half (``seam_gates``)
    names += list(GATE_ARGS) if key.seamed else []
    a2 = load(names)
    eg = {n: a2[n] for n in EG_ARGS} if key.eg else None
    sn = {n: a2[n] for n in SEAM_ARGS} if key.seam_next else None
    gin = {n: a2[n] for n in GATE_ARGS} if key.seamed else None
    run_k2(
        key,
        lds,
        {n: a2[K2_OF_LAYER.get(n, n)] for n in K2_ARGS},
        ep,
        True,
        eg,
        first,
        pre,
        pbases,
        sn,
        gin,
        None if chain is None else chain[1:],
    )


def build_mono_layer(key: MonoBuild):
    """Build one launch per layer (a reuse layer): K1, then K2 in the same launch.

    K1's products cross the grid without a launch boundary: each is stored device
    scope and flagged with K2's tag once it landed (``mb_front.hand_off``). Every CTA
    flags its q rows past its wq_b, so the flag also tells that the CTA's records, key
    table rows and mixes landed: the attention back's split waits on every CTA's q
    flag and its token's key-table flag, the FFN seam's slices on the gates' flags.
    With ``tagged_q_split`` the q rows come as tagged words instead, each row's record
    behind a flag of its own.
    The epoch moves on once a CTA has seen every CTA's q flag, so no CTA waits at the
    launch's end."""
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    assert key.kt and not key.index, key
    rs = route_shape(_moe_key(key))
    assert rs.experts // 64 in SORT_NETS
    half = peer_half_bytes(tp)
    assert SLICES + 2 * s <= BLOCKS
    LayerLds = _layer_lds(key, rs)
    name = kernel_symbol(
        "dsv41_mono_layer", s=s, tp=tp, c=key.compressed, f=key.fold, **_variant(key)
    )
    keyed = key_tuple(key, SOURCES)
    offsets = kernarg_offsets(LAYER_ARGS, LAYER_I32)
    # a split unit a CTA (``back.run_back``), its keys loaded in K1's half
    assert mb_back.back_plan(_back_key(key))[1] <= BLOCKS

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def layer(
        res_in: Int64,
        pend: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        attn_w: Int64,
        res_out: Int64,
        aux_out: Int64,
        aux_stride: Int32,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        wqkv: Int64,
        wqkv_s: Int64,
        qn_w: Int64,
        kvn_w: Int64,
        wqb: Int64,
        wqb_s: Int64,
        cos_sin: Int64,
        pos: Int64,
        slot: Int64,
        slot_words: Int32,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        swa_idx: Int64,
        swa_lens: Int64,
        topk: Int64,
        topk_lens: Int64,
        kt: Int64,
        klen: Int64,
        q: Int64,
        normed_out: Int64,
        qr_out: Int64,
        iq_w: Int64,
        iq_ws: Int64,
        iw_w: Int64,
        iw_scale: Int32,
        num_pos: Int32,
        iq_out: Int64,
        iqs_out: Int64,
        iw_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        ids: Int64,
        ids_words: Int32,
        comp: Int64,
        comp_stride: Int64,
        comp_block: Int32,
        sink: Int64,
        qk_scale: Int32,
        woa: Int64,
        woa_s: Int64,
        wob: Int64,
        wob_s: Int64,
        zrec: Int64,
        ffn_fn: Int64,
        ffn_scale: Int64,
        ffn_base: Int64,
        ffn_w: Int64,
        out_res: Int64,
        out_post: Int64,
        out_comb: Int64,
        out_pre: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        eg_x: Int64,
        eg_w: Int64,
        eg_ws: Int64,
        eg_kv: Int64,
        eg_row0: Int32,
        nx_aux: Int64,
        nx_aux_stride: Int32,
        seam_in: Int64,
        seam_out: Int64,
        nx_wqkv: Int64,
        nx_wqkv_s: Int64,
        nx_wqb: Int64,
        nx_wqb_s: Int64,
    ):
        _ = keyed
        warm_kernargs(max(offsets.values()) + 8)
        # every wait runs under the guard, so a timeout flags every peer and stops the launch
        with guard.scope(
            guard.Guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=tp,
                flags_offset=2 * half,
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            )
        ):
            lds = fx.SharedAllocator().allocate(LayerLds)
            # the peer table's words and the epoch together; the epoch in a scalar register,
            # it lives through K1 into K2, and the peer bases too (K2 reads them, a load of
            # them there would wait behind the split's early loads)
            peer_words = [
                fx.Vector(bo.buffer_load(rsrc(peers), 2 * p, vec_width=2, dtype=T.i32))
                for p in range(tp)
            ]
            ep = uniform(_epoch(epoch))
            pbases = [ptr64(w) for w in peer_words]

            layer_body(
                key,
                lds,
                {
                    "res_in": res_in,
                    "pend": pend,
                    "post_in": post_in,
                    "comb_in": comb_in,
                    "pre_in": pre_in,
                    "hc_fn": hc_fn,
                    "hc_scale": hc_scale,
                    "hc_base": hc_base,
                    "attn_w": attn_w,
                    "res_out": res_out,
                    "aux_out": aux_out,
                    "aux_stride": aux_stride,
                    "post_out": post_out,
                    "comb_out": comb_out,
                    "pre_out": pre_out,
                    "wqkv": wqkv,
                    "wqkv_s": wqkv_s,
                    "qn_w": qn_w,
                    "kvn_w": kvn_w,
                    "wqb": wqb,
                    "wqb_s": wqb_s,
                    "cos_sin": cos_sin,
                    "pos": pos,
                    "slot": slot,
                    "slot_words": slot_words,
                    "swa": swa,
                    "swa_stride": swa_stride,
                    "swa_block": swa_block,
                    "swa_idx": swa_idx,
                    "swa_lens": swa_lens,
                    "topk": topk,
                    "topk_lens": topk_lens,
                    "kt": kt,
                    "klen": klen,
                    "q": q,
                    "normed_out": normed_out,
                    "qr_out": qr_out,
                    "iq_w": iq_w,
                    "iq_ws": iq_ws,
                    "iw_w": iw_w,
                    "iw_scale": iw_scale,
                    "num_pos": num_pos,
                    "iq_out": iq_out,
                    "iqs_out": iqs_out,
                    "iw_out": iw_out,
                    "scratch": scratch,
                    "sym": sym,
                    "peers": peers,
                    "rank": rank,
                    "epoch": epoch,
                    "ids": ids,
                    "ids_words": ids_words,
                    "comp": comp,
                    "comp_stride": comp_stride,
                    "comp_block": comp_block,
                    "sink": sink,
                    "qk_scale": qk_scale,
                    "woa": woa,
                    "woa_s": woa_s,
                    "wob": wob,
                    "wob_s": wob_s,
                    "zrec": zrec,
                    "ffn_fn": ffn_fn,
                    "ffn_scale": ffn_scale,
                    "ffn_base": ffn_base,
                    "ffn_w": ffn_w,
                    "out_res": out_res,
                    "out_post": out_post,
                    "out_comb": out_comb,
                    "out_pre": out_pre,
                    "gate_w": gate_w,
                    "bias": bias,
                    "w13": w13,
                    "w13_s": w13_s,
                    "w2": w2,
                    "w2_s": w2_s,
                    "sgu": sgu,
                    "sgu_s": sgu_s,
                    "sw2": sw2,
                    "sw2_s": sw2_s,
                    "out": out,
                    "eg_x": eg_x,
                    "eg_w": eg_w,
                    "eg_ws": eg_ws,
                    "eg_kv": eg_kv,
                    "eg_row0": eg_row0,
                    "seam_in": seam_in,
                },
                ep,
                pbases,
                lambda names: reload_args(names, offsets, LAYER_I32),
            )

    @flyc.jit
    def launch(
        res_in: Int64,
        pend: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        attn_w: Int64,
        res_out: Int64,
        aux_out: Int64,
        aux_stride: Int32,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        wqkv: Int64,
        wqkv_s: Int64,
        qn_w: Int64,
        kvn_w: Int64,
        wqb: Int64,
        wqb_s: Int64,
        cos_sin: Int64,
        pos: Int64,
        slot: Int64,
        slot_words: Int32,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        swa_idx: Int64,
        swa_lens: Int64,
        topk: Int64,
        topk_lens: Int64,
        kt: Int64,
        klen: Int64,
        q: Int64,
        normed_out: Int64,
        qr_out: Int64,
        iq_w: Int64,
        iq_ws: Int64,
        iw_w: Int64,
        iw_scale: Int32,
        num_pos: Int32,
        iq_out: Int64,
        iqs_out: Int64,
        iw_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        ids: Int64,
        ids_words: Int32,
        comp: Int64,
        comp_stride: Int64,
        comp_block: Int32,
        sink: Int64,
        qk_scale: Int32,
        woa: Int64,
        woa_s: Int64,
        wob: Int64,
        wob_s: Int64,
        zrec: Int64,
        ffn_fn: Int64,
        ffn_scale: Int64,
        ffn_base: Int64,
        ffn_w: Int64,
        out_res: Int64,
        out_post: Int64,
        out_comb: Int64,
        out_pre: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        eg_x: Int64,
        eg_w: Int64,
        eg_ws: Int64,
        eg_kv: Int64,
        eg_row0: Int32,
        nx_aux: Int64,
        nx_aux_stride: Int32,
        seam_in: Int64,
        seam_out: Int64,
        nx_wqkv: Int64,
        nx_wqkv_s: Int64,
        nx_wqb: Int64,
        nx_wqb_s: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        layer(
            res_in,
            pend,
            post_in,
            comb_in,
            pre_in,
            hc_fn,
            hc_scale,
            hc_base,
            attn_w,
            res_out,
            aux_out,
            aux_stride,
            post_out,
            comb_out,
            pre_out,
            wqkv,
            wqkv_s,
            qn_w,
            kvn_w,
            wqb,
            wqb_s,
            cos_sin,
            pos,
            slot,
            slot_words,
            swa,
            swa_stride,
            swa_block,
            swa_idx,
            swa_lens,
            topk,
            topk_lens,
            kt,
            klen,
            q,
            normed_out,
            qr_out,
            iq_w,
            iq_ws,
            iw_w,
            iw_scale,
            num_pos,
            iq_out,
            iqs_out,
            iw_out,
            scratch,
            sym,
            peers,
            rank,
            epoch,
            ids,
            ids_words,
            comp,
            comp_stride,
            comp_block,
            sink,
            qk_scale,
            woa,
            woa_s,
            wob,
            wob_s,
            zrec,
            ffn_fn,
            ffn_scale,
            ffn_base,
            ffn_w,
            out_res,
            out_post,
            out_comb,
            out_pre,
            gate_w,
            bias,
            w13,
            w13_s,
            w2,
            w2_s,
            sgu,
            sgu_s,
            sw2,
            sw2_s,
            out,
            eg_x,
            eg_w,
            eg_ws,
            eg_kv,
            eg_row0,
            nx_aux,
            nx_aux_stride,
            seam_in,
            seam_out,
            nx_wqkv,
            nx_wqkv_s,
            nx_wqb,
            nx_wqb_s,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


def build_mono_layers(key: MonoBuild):
    """Build up to ``XL_LAYERS`` consecutive reuse layers in one launch: ``layer_body`` in a
    runtime loop over the launch's ``count`` layers.

    Layer l's arguments are row l of a table laid out as the layer launch's kernarg segment
    (``xl_row``), its step's from the launch's own (``XL_STEP``: with the Engram rows each
    layer's share of a later Engram layer's projection reads, ``key.eg`` tasks a layer), and
    the first layer's inputs too (``XL_FIRST``); it runs on epoch ``ep + l``, so every tag, parity and MoE
    counter slot is the one separate launches would use, and on ids of its own
    (``layer_ids``). A layer past the first has no launch boundary before it: its seam slice
    k sums the layer before's MoE partials of its 32 columns on CTA k, where that layer's
    FFN slice k stored the residual (``chained_slice``); that layer skipped its own sum.
    Every write of layer l + 1 follows, on its CTA, that CTA's own layer-l work, which
    waited on layer l's down (the slices, on the down partials of their columns; every other
    CTA, through its down task's wait on the ug), so no single-slot region or runner buffer
    is written while layer l reads it."""
    s, tp = key.tokens, key.tp
    assert key.chain and key.kt and key.fold, key
    assert not (key.index or key.tok_par or key.embed or key.glue), key
    assert not (key.seamed or key.seam_next), key
    rs = route_shape(_moe_key(key))
    assert rs.experts // 64 in SORT_NETS
    half = peer_half_bytes(tp)
    assert SLICES + 2 * s <= BLOCKS
    assert mb_back.back_plan(_back_key(key))[1] <= BLOCKS
    LayerLds = _layer_lds(key, rs)
    name = kernel_symbol(
        "dsv41_mono_layers", s=s, tp=tp, c=key.compressed, f=key.fold, **_variant(key)
    )
    keyed = key_tuple(key, SOURCES)
    offsets = kernarg_offsets(LAYER_ARGS, LAYER_I32)
    xoffsets = kernarg_offsets(XL_ARGS, XL_I32)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def layers(
        table: Int64,
        count: Int32,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        pos: Int64,
        slot: Int64,
        slot_words: Int32,
        swa_idx: Int64,
        swa_lens: Int64,
        topk: Int64,
        topk_lens: Int64,
        eg_x: Int64,
        res_in: Int64,
        pend: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
    ):
        _ = keyed
        warm_kernargs(max(xoffsets.values()) + 8)
        with guard.scope(
            guard.Guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=tp,
                flags_offset=2 * half,
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            )
        ):
            lds = fx.SharedAllocator().allocate(LayerLds)
            peer_words = [
                fx.Vector(bo.buffer_load(rsrc(peers), 2 * p, vec_width=2, dtype=T.i32))
                for p in range(tp)
            ]
            ep0 = uniform(_epoch(epoch))
            pbases = [ptr64(w) for w in peer_words]

            def one_layer(li):
                # a function: the loop carries none of the stages' Python state
                row = table + fx.Int64(li) * fx.Int64(XL_ROW)
                # the row's lines in the scalar cache before its arguments' loads
                warm_kernargs(XL_ROW, row)
                k = reload_args(XL_STEP + XL_FIRST, xoffsets, XL_I32)

                def load(names):
                    out = reload_args(
                        [n for n in names if n not in XL_STEP], offsets, LAYER_I32, row
                    )
                    return out | {n: k[n] for n in names if n in XL_STEP}

                with layer_ids():
                    a = load(BODY_ARGS)
                    a |= {n: (li == 0).select(k[n], a[n]) for n in XL_FIRST}
                    layer_body(
                        key,
                        lds,
                        a,
                        uniform(ep0 + li),
                        pbases,
                        load,
                        (li > 0, li < count - 1, li >= count - 1),
                    )

            for li in range(count):
                # the layer before's LDS is free
                gpu.barrier()
                one_layer(fx.Int32(li))

    @flyc.jit
    def launch(
        table: Int64,
        count: Int32,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        pos: Int64,
        slot: Int64,
        slot_words: Int32,
        swa_idx: Int64,
        swa_lens: Int64,
        topk: Int64,
        topk_lens: Int64,
        eg_x: Int64,
        res_in: Int64,
        pend: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        layers(
            table,
            count,
            scratch,
            sym,
            peers,
            rank,
            epoch,
            pos,
            slot,
            slot_words,
            swa_idx,
            swa_lens,
            topk,
            topk_lens,
            eg_x,
            res_in,
            pend,
            post_in,
            comb_in,
            pre_in,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
