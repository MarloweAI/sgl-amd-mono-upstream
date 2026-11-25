# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's MoE experts at TP2 (gfx950) in one FlyDSL launch, on the front's BM = 32 sort: the gate+up GEMM
with the clamped SiLU and the MXFP4 intermediate (G1), then the down GEMM with the route weight and the bf16 atomic
combine into the zeroed output (G2), for the 288 routed experts and the shared expert as expert 288.

Workgroups (256 threads, 4 waves), in id order:

- G1 (block b, tile n) for b < blocks, n < 1024 / N: the block's 32 sorted rows x intermediate columns N n ..
  N n + N - 1 (w13 gate rows N n .., up rows 1024 + N n ..) over K = 4096, N = 64 below 32 rows (more workgroups),
  else 128. Wave w takes N / 4 columns over all of K: one accumulator an output, K ascending in 128-deep scaled
  MFMAs, as aiter's stage 1 (at N = 64 a 32-column quant group's amax joins two waves' through LDS). Activations: a1
  rows gathered
  by token id with their per-token e8m0 scales. The epilogue is stock's stage 1 then its separate MXFP4 quant:
  gate <= limit, |up| <= limit, gate * sigmoid(gate) * up, rounded to bf16, then a scale per 32 columns
  (RCEIL e8m0 of amax / 6, amax from 1e-10) and v_cvt_scalef32_pk_fp4_f32. It stores the fp4 row-major [cap, 512 B]
  and its e8m0 [cap, 32] written through (sc0 sc1), waits for the stores, and adds 1 to block b's counter.
- G2 (block b, slab q) for q < 16: output columns 256 q .. 256 q + 255 (two 128-column tiles, one after another)
  over K = 1024; each wave takes 32 columns of a tile. Its first w2 loads go out, then thread 0 polls until block
  b's counter counts its G1 workgroups (each has a lower id and never waits); the workgroup reads the block's
  intermediate once into LDS (sc1 loads) and adds weight x sum as bf16 pairs into the output (buffer atomics; rows
  whose token is past M fall outside the output's buffer range). Its departure counts on the same counter; the
  block's last zeroes it.
- Blocks from num_valid_ids[0] / 32 on exit at once and touch no counter.

The shared expert (slot 288 of the sort) follows its own path in stock (SGLang's dense MLP): its weights are the
dense layers' own (row-major fp4 [N, K/2], row-major e8m0 [N, K/32]); its G1 reads the activations quantized
with even-rounded scales (``a1_even``), rounds gate and up to bf16, applies aiter's ``fused_clamp_act_mul`` (silu as
x / (1 + exp2(-x log2 e)), correctly rounded division), rounds to bf16 and quantizes the intermediate with
``dynamic_mxfp4_quant``'s even scales; its G2 rounds to bf16 and stores the shared rows (one route a token) to
``shared_out``. The hook of a completed output group then writes bf16(f32(routed) + f32(shared)) for its rows.
Tasks of the shared expert take their own code path (chosen by the block's expert id) and its blocks are dispatched
first; its dense rows stream slower, so its G1 issues both K steps of a row's 128 B line together and its G2 runs
as 16 workgroups of 128 columns (slots right after the G1 workgroups, for the sort's last blocks).

Routed weights are read nontemporal (streamed once). Output groups: after its atomics land, a G2 workgroup adds 1 on
``count[i][s]`` for every 16-token tile i among its block's rows and every 128-column tile of slab s it covers. The
workgroup that brings it to ``tile_target[i]`` (2 x the sorted blocks holding a token of tile i: the front's sort
gives it) zeroes it and runs ``hook(i, s)`` once rows 16 i .. 16 i + 15, columns 256 s .. + 255 have every route
added (read them with sc1 loads); it adds the shared rows. Every counter is 0 between launches."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float32, Int32, Int64, T

from sgl_amd_mono.common.mx import FP4, mfma_scaled
from sgl_amd_mono.common.ops import CM_DEV, CM_NT, CM_SYS, bf16_round, global_ptr, hw_exp2, hw_rcp, ld_i32, rsrc
from sgl_amd_mono.common.ops import uniform, xred
from sgl_amd_mono.common.plan import kernel_sources

from . import BM, D, MAX_ROWS, TOPK, capacity

SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)

THREADS = 256
NE = 288  # routed experts
SHARED = 288  # the shared expert's id in the sort
I = 1024  # a rank's intermediate
WIDE_ROWS = 32  # from this many rows on, G1 workgroups take 128 columns (32 a wave), else 64 (16 a wave)
G1_STEPS = D // 128  # 128-deep MFMA steps of G1
G2_N = 128  # output columns of a G2 tile
G2_SPAN = 2  # G2 tiles of a G2 workgroup
G2_WGS = D // (G2_N * G2_SPAN)  # G2 workgroups of a block
SH_SPAN = 1  # G2 tiles of a shared-expert G2 workgroup (its dense layout streams slower: more workgroups)
SH_WGS = D // (G2_N * SH_SPAN)  # G2 workgroups of a shared-expert block
SH_SLOTS = MAX_ROWS // BM * SH_WGS  # their slots, after the G1 workgroups (the shared blocks are the sort's last)
G2_STEPS = I // 128  # 128-deep MFMA steps of a G2 tile
# Launch kinds by rows: G1 workgroups of 128 columns from WIDE_ROWS, else 64; below DEEP_ROWS so few blocks stream
# that a block's G1 -> G2 chain sets the time, and each wave keeps more of its loads in flight. G1 / G2 steps of
# loads a wave has in flight, by kind:
DEEP_ROWS = 8
PREFETCH = {"deep": (16, 16), "narrow": (4, 8), "wide": (4, 8)}
LOG2E = 1.4426950408889634
# weights and e8m0 scales of an expert (aiter's (16, 16) shuffle and e8m0_shuffle)
W13_BYTES = 2 * I * D // 2
W2_BYTES = D * I // 2
S13_BYTES = 2 * I * D // 32
S2_BYTES = D * I // 32
# the intermediate: fp4 rows, then their e8m0, for the largest sort
MAX_CAP = capacity(MAX_ROWS, TOPK + 1)
MAX_BLOCKS = MAX_CAP // BM
MID_ROW = I // 2
MID_SCALES = I // 32
MID_SCALE_AT = MAX_CAP * MID_ROW
MID_BYTES = MAX_CAP * (MID_ROW + MID_SCALES)
# LDS (dwords): a G2 block's intermediate (rows padded so 16 rows hit distinct
# banks) and its e8m0
XL_ROW = MID_ROW // 4 + 4
XL_SCALES = BM * XL_ROW
PART_AT = XL_SCALES + BM * MID_SCALES // 4  # the parts variant's G2 staging: 32 rows x 128 columns of bf16
PART_ROW = G2_N // 2 + 4  # its row in dwords, padded
LDS_WORDS = PART_AT
# counters, a 128 B line each: the blocks', then the output groups' (16-token tile x 256 columns)
LINE = 128
TOKEN_TILES = MAX_ROWS // 16
SLABS = D // 256
GROUP_AT = MAX_BLOCKS * LINE
COUNTER_WORDS = (MAX_BLOCKS + TOKEN_TILES * SLABS) * LINE // 4


def _ld_agent(addr):
    return fx.generic_load(
        global_ptr(addr), dtype=fx.Int32, memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent"
    )


def _st_agent(addr, v):
    fx.generic_store(global_ptr(addr), fx.Int32(v), memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent")


def _add_agent(addr, v=1):
    """The old value, after an agent-scope add of ``v``."""
    return fx.Int32(fx.llvm.atomic_add(global_ptr(addr), fx.Int32(v), syncscope="agent"))


def _load(r, i, cm=0):
    return fx.Vector(bo.buffer_load(r, i, vec_width=4, dtype=T.i32, cache_modifier=cm))


def _load_word(r, i, cm=0):
    return fx.Int32(bo.buffer_load(r, i, vec_width=1, dtype=T.i32, cache_modifier=cm))


def _load_byte(r, i):
    return fx.Int32(bo.buffer_load(r, i, vec_width=1, dtype=T.i8))


def _store_done():
    """Every wave's stores and atomics have landed, then the workgroup meets."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()


def _silu(gate, up, neg_limit):
    """Stock's clamped SiLU: gate <= limit, |up| <= limit, gate * sigmoid(gate) * up (exp2 / rcp)."""
    g = -fx.max(-gate, neg_limit)
    u = fx.max(-fx.max(-up, neg_limit), neg_limit)
    s = hw_rcp(fx.Float32(1.0) + hw_exp2(g * fx.Float32(-LOG2E)))
    return g * s * u


def _e8m0_ceil(amax):
    """RCEIL: the biased exponent of the smallest power of two >= amax / 6."""
    u = (amax * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
    e = (u >> 23) & 0xFF
    return e + ((e < 0xFF) & ((u & 0x7FFFFF) != 0)).select(fx.Int32(1), fx.Int32(0))


def _e8m0_even(amax):
    """``dynamic_mxfp4_quant``'s scale: the biased exponent of amax rounded to even, less 2, at least 0."""
    raw = ((amax.bitcast(fx.Int32) + 0x200000) & -8388608) >> 23
    raw = raw & 0xFF
    return (raw >= 2).select(raw - 2, fx.Int32(0))


def _silu_shared(gate, up, limit):
    """aiter's ``fused_clamp_act_mul`` (silu) on bf16 gate / up: up in [-limit, limit], gate <= limit,
    gate / (1 + exp2(-(gate log2 e))) * up."""
    g = fx.min(bf16_round(gate), limit)
    u = fx.max(fx.min(bf16_round(up), limit), -limit)
    return g / (fx.Float32(1.0) + hw_exp2(-(g * fx.Float32(1.44269504089)))) * u


def _fp4x4(v, scale):
    """Four f32 -> four E2M1 (two bytes) at the dequant ``scale``."""
    q = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, fx.Int32(0), v[0], v[1], scale, 0)
    q = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, q, v[2], v[3], scale, 1)
    return fx.Int32(q).to(fx.Int16)


SLOTS = TOPK + 1  # a token's routes in the sort (slot field of its sorted ids): TOPK routed, then the shared expert
# from PARTS_ROWS rows, the atomic-free combine: G2 stores each route's bf16 rows to parts [token][slot][D], and
# reduce workgroups after the G2 slots sum each output group once it is complete (deterministic); below, bf16 atomics
PARTS_ROWS = 32
PART_ROWS = SLOTS + 1  # a token's rows of parts: its slots', then a 256-column slab's routes' expert ids + 1
ID_SPAN = 16  # dwords of a slab's ids (slot s at dword s)


def _bf16_add(a, b):
    """bf16 pairs a + b (packed dwords), each sum rounded to bf16 (RNE)."""
    lo = (a << 16).bitcast(fx.Float32) + (b << 16).bitcast(fx.Float32)
    hi = (a & -65536).bitcast(fx.Float32) + (b & -65536).bitcast(fx.Float32)
    return fx.Vector.from_elements([lo, hi], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]


def _hook_parts(c, token_tile, slab):
    """The parts variant's hook, for rows 16 token_tile .., columns 256 slab ..: the routes present (their expert
    id + 1 in the slab's id row; 0: absent, read as 0) added as bf16 in ascending expert order (the order a sort's
    atomics mostly land in, each add rounded), then the shared rows as bf16(f32(routed) + f32(shared)); the slab's
    ids are put back to 0."""
    a, tid = c["args"], c["tid"]
    ro = rsrc(a["out"], fx.Int64(c["m"]) * (D * 2))
    rp = rsrc(a["shared"], fx.Int64(c["m"]) * (PART_ROWS * D * 2))
    row = token_tile * 16 + tid // 16
    idw = ((row * PART_ROWS + SLOTS) * (D * 2)) // 4 + slab * ID_SPAN
    ids = [_load(rp, idw + h * 4, cm=CM_DEV) for h in range(3)]
    e = [ids[s // 4][s % 4] for s in range(SLOTS)]
    one, nil = fx.Int32(1), fx.Int32(0)
    key = [(v == 0).select(fx.Int32(1 << 30), v) for v in e[:TOPK]]
    rank = []
    for s in range(TOPK):
        r = nil
        for t in range(TOPK):
            if t != s:  # ties (absent routes) broken by slot
                before = (key[t] <= key[s]) if t < s else (key[t] < key[s])
                r = r + before.select(one, nil)
        rank.append(r)
    # the slots in that order, then the shared expert's; an absent route's offset falls outside the buffer
    order = []
    for r in range(TOPK):
        v = nil
        for s in range(TOPK):
            v = (rank[s] == r).select(fx.Int32(s), v)
        order.append(v)
    order.append(fx.Int32(TOPK))
    present = []
    for r in range(TOPK):
        p = nil
        for s in range(TOPK):
            p = (order[r] == s).select(e[s], p)
        present.append(p != 0)
    present.append(e[TOPK] != 0)
    pushed = []
    for i in range_constexpr(2):
        col = slab * 512 + (tid % 16) * 32 + i * 16
        parts = [
            _load(rp, present[r].select(((row * PART_ROWS + order[r]) * (D * 2) + col) // 4, fx.Int32(1 << 29)),
                  cm=CM_DEV)
            for r in range(SLOTS)
        ]  # fmt: skip
        res = []
        for d in range_constexpr(4):
            acc = parts[0][d]
            for r in range_constexpr(1, TOPK):
                acc = _bf16_add(acc, parts[r][d])
            sh = parts[TOPK][d]
            lo = (acc << 16).bitcast(fx.Float32) + (sh << 16).bitcast(fx.Float32)
            hi = (acc & -65536).bitcast(fx.Float32) + (sh & -65536).bitcast(fx.Float32)
            res.append(fx.Vector.from_elements([lo, hi], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0])
        out = fx.Vector.from_elements(res, fx.Int32)
        bo.buffer_store(out, ro, (row * (D * 2) + col) // 4)
        pushed.append((out, (row * (D * 2) + col) // 4))
    zero = fx.Vector.filled(4, 0, fx.Int32)
    for h in range_constexpr(3):
        bo.buffer_store(zero, rp, idw + h * 4, mask=tid % 16 == 0, cache_modifier=CM_DEV)
    return pushed


def _push(c, token_tile, slab, pushed):
    """The push build: this thread's output pieces to the peer's output (written through, sc0 sc1); once every
    thread's stores have landed, thread 0 stores the launch's epoch to the peer's flag of the group."""
    rpeer = rsrc(c["args"]["peer_out"], fx.Int64(c["m"]) * (D * 2))
    for v, w in pushed:
        bo.buffer_store(v, rpeer, w, cache_modifier=CM_SYS)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    _flag(c, token_tile * SLABS + slab)


@flyc.jit
def _flag(c, i):
    """Thread 0 stores the epoch to the peer's flag ``i`` (sc0 sc1)."""
    if c["tid"] == 0:
        rf = rsrc(c["args"]["peer_flags"], TOKEN_TILES * SLABS * 4)
        bo.buffer_store(c["epoch"], rf, i, cache_modifier=CM_SYS)


def hook(c, token_tile, slab):
    """Runs on the workgroup that completed output rows 16 token_tile .., columns 256 slab ..: every route of those
    rows has landed (sc1 loads); adds the shared rows, f32 sum rounded once to bf16."""
    if const_expr(c["parts"]):
        _hook_parts(c, token_tile, slab)
        return
    a, tid = c["args"], c["tid"]
    ro = rsrc(a["out"], fx.Int64(c["m"]) * (D * 2))
    rsh = rsrc(a["shared"], fx.Int64(c["m"]) * (D * 2))
    row = token_tile * 16 + tid // 16
    for i in range_constexpr(2):
        w = (row * (D * 2) + slab * 512 + (tid % 16) * 32 + i * 16) // 4
        o = _load(ro, w, cm=CM_DEV)
        sh = _load(rsh, w, cm=CM_DEV)
        res = []
        for d in range_constexpr(4):
            lo = (o[d] << 16).bitcast(fx.Float32) + (sh[d] << 16).bitcast(fx.Float32)
            hi = (o[d] & -65536).bitcast(fx.Float32) + (sh[d] & -65536).bitcast(fx.Float32)
            res.append(fx.Vector.from_elements([lo, hi], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0])
        bo.buffer_store(fx.Vector.from_elements(res, fx.Int32), ro, w)


# ---- G1 ----


def _srcs(c, w, s, w_sh, s_sh, e, w_bytes, s_bytes):
    """A task's weight and e8m0 sources: the routed expert's, or the shared expert's dense tensors."""
    if const_expr(c["sh"]):
        return {"rw": rsrc(w_sh), "rs": rsrc(s_sh)}
    return {"rw": rsrc(w + e * w_bytes), "rs": rsrc(s + e * s_bytes)}


def _w_load(c, rg, kc, k):
    """This lane's 16 B of row group rg, K step kc (K = k): aiter's shuffle (streamed nontemporal), or the shared
    expert's row-major rows (a row's 128 B line serves two K steps: cached)."""
    lane = c["lane"]
    if const_expr(c["sh"]):
        return _load(c["rw"], (rg * 16 + lane % 16) * (k // 8) + kc * 16 + (lane // 16) * 4)
    return _load(c["rw"], rg * (2 * k) + kc * 256 + lane * 4, cm=CM_NT)


def _s_word(c, rg, kc, k):
    """The e8m0 dword of row groups rg, rg + 1 and K steps kc, kc + 1 (kc, rg even; byte (kc % 2) * 2 + j): one
    e8m0_shuffle dword, or (the shared expert) four bytes of its row-major scales."""
    lane = c["lane"]
    if const_expr(not c["sh"]):
        return _load_word(c["rs"], ((rg // 2) * (k // 256) + kc // 2) * 64 + lane)
    shift = (lane // 16) * 8
    word = fx.Int32(0)
    for j in range_constexpr(2):
        v = fx.Vector(bo.buffer_load(c["rs"], ((rg + j) * 16 + lane % 16) * (k // 128) + kc // 2 * 2, 2, T.i32))
        for h in range_constexpr(2):
            word = word | (((v[h] >> shift) & 0xFF) << (8 * (h * 2 + j)))
    return word


def _s_word1(c, rg, kc, k):
    """The e8m0 of row group rg at K steps kc, kc + 1 (kc even) in bytes 0 and 2 of a dword."""
    if const_expr(not c["sh"]):
        return _s_word(c, rg - rg % 2, kc, k) >> ((rg % 2) * 8)
    lane = c["lane"]
    v = fx.Vector(bo.buffer_load(c["rs"], (rg * 16 + lane % 16) * (k // 128) + kc // 2 * 2, 2, T.i32))
    shift = (lane // 16) * 8
    return ((v[0] >> shift) & 0xFF) | (((v[1] >> shift) & 0xFF) << 16)


def _g1n(c):
    """Intermediate columns of a G1 workgroup."""
    return 128 if c["wide"] else 64


def _g1_wgs(c):
    return I // _g1n(c)


def _g1_loads(c, step):
    """One wave's operands of G1 K step ``step``: gate / up weights, the rows' activations and their scales, and
    (even steps) the gate / up scale dwords of the step pair."""
    lane, kc = c["lane"], step
    nj = 2 if c["wide"] else 1
    w = [_w_load(c, c["rg"] + u * (I // 16) + j, kc, D) for u in range(2) for j in range(nj)]
    x = [_load(c["ra"], t * (D // 8) + kc * 16 + (lane // 16) * 4) for t in c["toks"]]
    xs = [_load_byte(c["ras"], t * (D // 32) + kc * 4 + lane // 16) for t in c["toks"]]
    s = None
    if const_expr(step % 2 == 0):
        word = _s_word if c["wide"] else _s_word1
        s = [word(c, c["rg"] + u * (I // 16), kc, D) for u in range(2)]
    return w, x, xs, s


def _g1_epilogue(c, blk, nt, acc):
    """Rows of the block x this wave's columns (acc [gate / up][j][row tile][4]): activation, bf16, MXFP4 with the
    e8m0 of its 32-column group (at N = 64 shared with the partner wave w ^ 1: every wave meets here), written
    through."""
    lane, w, lds, wide = c["lane"], c["wave"], c["lds"], c["wide"]
    nj = 2 if wide else 1
    neg = -c["limit"]
    rm = rsrc(c["args"]["mid"], MID_BYTES)
    sh = c["sh"]
    act = (lambda g_, u_: _silu_shared(g_, u_, c["limit"])) if sh else (lambda g_, u_: _silu(g_, u_, neg))
    vals, amax = [], []
    for rt in range_constexpr(2):
        v = [[bf16_round(act(acc[0][j][rt][i], acc[1][j][rt][i])) for i in range(4)] for j in range(nj)]
        a = fx.Float32(0.0 if sh else 1e-10)
        for j in range_constexpr(nj):
            for i in range_constexpr(4):
                a = fx.max(a, fx.max(v[j][i], -v[j][i]))
        a = xred(xred(a, 16, fx.max), 32, fx.max)
        if const_expr(not wide):
            fx.ptr_store(a.bitcast(fx.Int32), lds + ((w * 2 + rt) * 16 + lane % 16))
        vals.append(v)
        amax.append(a)
    if const_expr(not wide):
        gpu.barrier()
    for rt in range_constexpr(2):
        row = blk * BM + rt * 16 + lane % 16
        a = amax[rt]
        if const_expr(not wide):
            a = fx.max(a, fx.ptr_load(lds + (((w ^ 1) * 2 + rt) * 16 + lane % 16)).bitcast(fx.Float32))
        e = _e8m0_even(a) if const_expr(sh) else _e8m0_ceil(a)
        scale = (e << 23).bitcast(fx.Float32)
        for j in range_constexpr(nj):
            off = row * MID_ROW + nt * (_g1n(c) // 2) + w * (8 * nj) + j * 8 + (lane // 16) * 2
            bo.buffer_store(_fp4x4(vals[rt][j], scale), rm, off, cache_modifier=CM_SYS, offset_is_bytes=True)
        writer = (lane < 16) if wide else ((lane < 16) & (w % 2 == 0))
        bo.buffer_store(
            e.to(fx.Int8), rm, MID_SCALE_AT + row * MID_SCALES + nt * (_g1n(c) // 32) + (w if wide else w // 2),
            mask=writer, cache_modifier=CM_SYS, offset_is_bytes=True,
        )  # fmt: skip


@flyc.jit
def _count_block(c, cnt):
    """Thread 0 adds the G1 tile to its block's counter."""
    if c["tid"] == 0:
        _add_agent(cnt)


def _g1_tile(c, blk, nt, e, sh):
    """G1 tile (block ``blk``, intermediate columns 64 ``nt`` .., expert ``e``; ``sh``: the shared expert's path),
    then its count on the block's counter."""
    a, lane, w = c["args"], c["lane"], c["wave"]
    k = {**c, "sh": sh}
    k = {
        **k,
        **_srcs(k, a["w13"], a["s13"], a["w13sh"], a["s13sh"], e, W13_BYTES, S13_BYTES),
        # the shared expert's activations: x quantized with even scales
        "ra": rsrc(a["a1e"] if sh else a["a1"], fx.Int64(c["m"]) * (D // 2)),
        "ras": rsrc(a["a1es"] if sh else a["a1s"], fx.Int64(c["m"]) * (D // 32)),
        "toks": [ld_i32(a["sids"], blk * BM + rt * 16 + lane % 16) & 0xFFFFFF for rt in range(2)],
        "rg": nt * (_g1n(c) // 16) + (2 if c["wide"] else 1) * w,  # the wave's first gate row group
    }
    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    nj = 2 if c["wide"] else 1
    acc = [[[zero, zero] for _ in range(nj)] for _ in range(2)]  # [gate / up][j][row tile]
    pf = c["pf"][0]
    pend = [_g1_loads(k, s) for s in range(pf)]
    sc = None
    # the shared expert's dense rows: both K steps of a row's 128 B line go out together
    burst = 2 if sh else 1
    for step in range_constexpr(G1_STEPS):
        wv, x, xs, s = pend.pop(0)
        if const_expr(step % burst == 0):
            for b in range_constexpr(burst):
                if const_expr(step + pf + b < G1_STEPS):
                    pend.append(_g1_loads(k, step + pf + b))
        if const_expr(s is not None):
            sc = s
        for u in range_constexpr(2):
            for j in range_constexpr(nj):
                for rt in range_constexpr(2):
                    acc[u][j][rt] = mfma_scaled(
                        wv[u * nj + j], x[rt], acc[u][j][rt], sc[u], xs[rt], FP4, FP4, (step % 2) * 2 + j, 0
                    )
        rocdl.sched_barrier(0)
    _g1_epilogue(k, blk, nt, [[[[v[i] for i in range(4)] for v in a2] for a2 in a1] for a1 in acc])
    _store_done()
    _count_block(c, a["ws"] + fx.Int64(blk * LINE))


@flyc.jit
def gemm1(c, blk, nt):
    if blk * BM < c["nv"]:
        ei = uniform(ld_i32(c["args"]["seids"], blk))
        if ei == SHARED:
            _g1_tile(c, blk, nt, fx.Int64(ei), True)
        else:
            _g1_tile(c, blk, nt, fx.Int64(ei), False)


# ---- G2 ----


@flyc.jit
def _await_block(c, cnt, g2_wgs):
    """Thread 0 polls the block's counter until its G1 tiles are in, then counts this workgroup's departure (the
    block's last of ``g2_wgs`` G2 workgroups zeroes it); the workgroup then meets."""
    if c["tid"] == 0:
        v = _ld_agent(cnt)
        while v < _g1_wgs(c):
            rocdl.s_sleep(1)
            v = _ld_agent(cnt)
        if _add_agent(cnt) == _g1_wgs(c) + g2_wgs - 1:
            _st_agent(cnt, 0)
    gpu.barrier()


def _group_rows(c, blk):
    """Thread t < 32: (token tile of sorted row t, whether row t starts that tile among the block's live rows, the
    tile's target). Loaded ahead of the wait."""
    a, tid = c["args"], c["tid"]
    row = blk * BM + fx.min(tid, BM - 1)
    tok = ld_i32(a["sids"], row) & 0xFFFFFF
    prev = ld_i32(a["sids"], row - (tid > 0).select(fx.Int32(1), fx.Int32(0))) & 0xFFFFFF
    tile = tok // 16
    # a block's rows are one expert's routes in token order, padding last: a tile's first row starts it
    first = (tid < BM) & (tok < c["m"]) & ((tid == 0) | (prev // 16 != tile))
    return tile, first, ld_i32(a["target"], fx.min(tile, TOKEN_TILES - 1))


@flyc.jit
def _arrive_groups(c, q, span, rows):
    """Counts this G2 workgroup (columns 128 ``span`` ``q`` ..) on the output group of every token tile among the
    block's rows, for each slab it covers (1 a 128-column tile); the group's last arrival zeroes the count and the
    workgroup runs ``hook`` for it."""
    tid, flags = c["tid"], c["lds"]
    tile, first, target = rows
    nsl, add = max(span // 2, 1), min(span, 2)
    s0 = q * span // 2
    if const_expr(c["parts"]):
        # the parts variant: a reduce workgroup waits for the group instead
        for h in range_constexpr(nsl):
            if first:
                _add_agent(c["args"]["ws"] + fx.Int64(GROUP_AT + (tile * SLABS + s0 + h) * LINE), add)
        return
    for h in range_constexpr(nsl):
        done = fx.Int32(-1)
        if first:
            cnt = c["args"]["ws"] + fx.Int64(GROUP_AT + (tile * SLABS + s0 + h) * LINE)
            if _add_agent(cnt, add) == target - add:
                _st_agent(cnt, 0)
                done = tile
        if tid < BM:
            fx.ptr_store(done, flags + (h * BM + tid))
    gpu.barrier()
    for h in range_constexpr(nsl):
        for t in range_constexpr(BM):
            i = fx.ptr_load(flags + (h * BM + t))
            if i >= 0:
                hook(c, i, s0 + h)


def _g2_loads(c, step):
    """One wave's w2 operands of G2 step ``step`` (tile step // 8 of the workgroup's, K step step % 8): rows j = 0, 1, and
    (even K steps) the scale dword of the K step pair."""
    lane, n, kc = c["lane"], step // G2_STEPS, step % G2_STEPS
    rg = c["rg"] + n * (G2_N // 16)
    w = [_w_load(c, rg + j, kc, I) for j in range(2)]
    s = None
    if const_expr(kc % 2 == 0):
        s = _s_word(c, rg, kc, I)
    return w, s


def _live_rows(c, blk):
    """Whether each of this thread's 4 intermediate rows (tid // 32 + 8 i) holds a route (pad rows: token M)."""
    a = c["args"]
    return [(ld_i32(a["sids"], blk * BM + c["tid"] // 32 + 8 * i) & 0xFFFFFF) < c["m"] for i in range(4)]


def _g2_stage(c, blk, live):
    """The block's 32 intermediate rows and their e8m0, read once (sc1) into LDS; pad rows (``live``: this thread's
    rows) are not read (their bytes stay whatever LDS held: their outputs fall outside every buffer)."""
    tid, lds = c["tid"], c["lds"]
    rm = rsrc(c["args"]["mid"], MID_BYTES)
    got = [
        _load(rm, live[i].select(blk * (BM * MID_ROW // 4) + (tid + THREADS * i) * 4, fx.Int32(MID_BYTES)), cm=CM_DEV)
        for i in range(4)
    ]
    sw = _load(rm, MID_SCALE_AT // 4 + blk * (BM * MID_SCALES // 4) + fx.min(tid, 63) * 4, cm=CM_DEV)
    for i in range_constexpr(4):
        u = tid + THREADS * i
        fx.ptr_store(got[i], lds + ((u // 32) * XL_ROW + (u % 32) * 4))
    fx.ptr_store(sw, lds + (XL_SCALES + fx.min(tid, 63) * 4))
    gpu.barrier()


def _g2_x(c, rt, kc):
    """This lane's intermediate operand (row tile rt, K step kc) and its e8m0 (low byte), from LDS."""
    lane, lds = c["lane"], c["lds"]
    r = rt * 16 + lane % 16
    v4i = fx.Vector.make_type(4, fx.Int32)
    x = fx.Vector(fx.ptr_load(lds + (r * XL_ROW + kc * 16 + (lane // 16) * 4), result_type=v4i))
    return x, fx.ptr_load(lds + (XL_SCALES + r * (MID_SCALES // 4) + kc)) >> ((lane // 16) * 8)


def _g2_parts(c, acc, tws, n):
    """The parts variant's combine of tile ``n``: the workgroup's 32 rows x 128 columns of weight x sum (bf16) go
    through LDS so each row's 256 B is stored whole to its route's part (and, at tile 0, the route's expert id)."""
    lane, wave, tid, st = c["lane"], c["wave"], c["tid"], c["lds"] + PART_AT
    rp = rsrc(c["args"]["shared"], fx.Int64(c["m"]) * (PART_ROWS * D * 2))
    gpu.barrier()
    for rt in range_constexpr(2):
        for j in range_constexpr(2):
            pairs = [
                fx.Vector.from_elements(
                    [acc[j][rt][2 * p] * tws[rt], acc[j][rt][2 * p + 1] * tws[rt]], fx.Float32
                ).to(fx.BFloat16).bitcast(fx.Int32)[0]
                for p in range(2)
            ]
            at = (rt * 16 + lane % 16) * PART_ROW + (wave * 32 + j * 16 + (lane // 16) * 4) // 2
            fx.ptr_store(fx.Vector.from_elements(pairs, fx.Int32), st + at)
    gpu.barrier()
    v4i = fx.Vector.make_type(4, fx.Int32)
    col0 = (c["rg"] - 2 * wave) * 16 + n * G2_N
    for h in range_constexpr(2):
        tok, slot = c["wrows"][h]
        r = tid // 16 + 16 * h
        v = fx.Vector(fx.ptr_load(st + (r * PART_ROW + (tid % 16) * 4), result_type=v4i))
        off = (tok * PART_ROWS + slot) * (D * 2) + col0 * 2 + (tid % 16) * 16
        bo.buffer_store(v, rp, off, cache_modifier=CM_DEV, offset_is_bytes=True)
        if const_expr(n == 0):
            at = (tok * PART_ROWS + SLOTS) * (D * 2) + ((col0 // 256) * ID_SPAN + slot) * 4
            bo.buffer_store(c["e32"] + 1, rp, at, mask=tid % 16 == 0, cache_modifier=CM_DEV, offset_is_bytes=True)


def _g2_combine(c, acc, toks, tws, ro, n):
    """weight x sum of tile ``n`` as bf16 pairs: added into the output, or (the shared expert, weight 1) stored to
    its rows of ``shared_out`` (written through); the other access falls outside its buffer. The parts variant
    stores every route's pairs to its row of parts."""
    if const_expr(c["parts"]):
        _g2_parts(c, acc, tws, n)
        return
    rsh = rsrc(c["args"]["shared"], fx.Int64(c["m"]) * (D * 2))
    for rt in range_constexpr(2):
        for j in range_constexpr(2):
            col = (c["rg"] + n * (G2_N // 16) + j) * 16 + (c["lane"] // 16) * 4
            off = toks[rt] * (D * 2) + col * 2
            pairs = [
                fx.Vector.from_elements(
                    [acc[j][rt][2 * p] * tws[rt], acc[j][rt][2 * p + 1] * tws[rt]], fx.Float32
                ).to(fx.BFloat16)
                for p in range(2)
            ]
            if const_expr(c["sh"]):
                both = fx.Vector.from_elements([pairs[p].bitcast(fx.Int32)[0] for p in range(2)], fx.Int32)
                bo.buffer_store(both, rsh, off, cache_modifier=CM_SYS, offset_is_bytes=True)
            else:
                for p in range_constexpr(2):
                    rocdl.raw_ptr_buffer_atomic_fadd(pairs[p], ro, off + 4 * p, 0)


def _g2_tile(c, blk, q, e, sh):
    """G2 workgroup (block ``blk``, output columns 128 span ``q`` .. (span: 4 routed, 1 the shared expert), expert
    ``e``; ``sh``: the shared expert's path) once the block's G1 tiles are in."""
    span = SH_SPAN if sh else G2_SPAN
    a, lane, wave = c["args"], c["lane"], c["wave"]
    k = {**c, "sh": sh}
    k = {
        **k,
        **_srcs(k, a["w2"], a["s2"], a["w2sh"], a["s2sh"], e, W2_BYTES, S2_BYTES),
        "rg": q * (span * G2_N // 16) + 2 * wave,  # the wave's first w2 row group (16 output columns)
    }
    steps = span * G2_STEPS
    pf = c["pf"][1]
    pend = [_g2_loads(k, s) for s in range(pf)]
    rows = [blk * BM + rt * 16 + lane % 16 for rt in range(2)]
    toks = [ld_i32(a["sids"], r) & 0xFFFFFF for r in rows]
    k["e32"] = uniform(ld_i32(a["seids"], blk))
    wid = [ld_i32(a["sids"], blk * BM + c["tid"] // 16 + 16 * h) for h in range(2)]
    k["wrows"] = [(v & 0xFFFFFF, fx.min((v >> 24) & 0xFF, SLOTS - 1)) for v in wid]
    tws = [fx.Float32(bo.buffer_load(rsrc(a["sw"]), r, 1, T.f32)) for r in rows]
    groups = _group_rows(c, blk)
    live = _live_rows(c, blk)
    _await_block(c, a["ws"] + fx.Int64(blk * LINE), SH_WGS if sh else G2_WGS)
    _g2_stage(c, blk, live)
    ro = rsrc(a["out"], fx.Int64(c["m"]) * (D * 2))
    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    acc = [[zero, zero] for _ in range(2)]  # [j][row tile]
    sc = None
    for step in range_constexpr(steps):
        w, s = pend.pop(0)
        if const_expr(step + pf < steps):
            pend.append(_g2_loads(k, step + pf))
        if const_expr(s is not None):
            sc = s
        kc = step % G2_STEPS
        for rt in range_constexpr(2):
            xv, xsc = _g2_x(c, rt, kc)
            for j in range_constexpr(2):
                acc[j][rt] = mfma_scaled(w[j], xv, acc[j][rt], sc, xsc, FP4, FP4, (kc % 2) * 2 + j, 0)
        if const_expr(kc == G2_STEPS - 1):
            _g2_combine(k, acc, toks, tws, ro, step // G2_STEPS)
            acc = [[zero, zero] for _ in range(2)]
        rocdl.sched_barrier(0)
    _store_done()
    _arrive_groups(c, q, span, groups)


@flyc.jit
def gemm2(c, blk, q):
    """A routed block's G2 workgroup q (a shared-expert block's are ``gemm2_shared``'s)."""
    if blk * BM < c["nv"]:
        ei = uniform(ld_i32(c["args"]["seids"], blk))
        if ei != SHARED:
            _g2_tile(c, blk, q, fx.Int64(ei), False)


def _shared_blocks(c):
    """(the sort's block count, its last ns blocks: the shared expert's at most)."""
    nb = c["nv"] // BM
    return nb, fx.min((c["m"] + BM - 1) // BM, nb)


@flyc.jit
def gemm2_shared(c, p):
    """Shared-expert G2 slot p: block (p // SH_WGS) of the sort's last ns, workgroup p % SH_WGS."""
    nb, ns = _shared_blocks(c)
    if p // SH_WGS < ns:
        blk = nb - ns + p // SH_WGS
        ei = uniform(ld_i32(c["args"]["seids"], blk))
        if ei == SHARED:
            _g2_tile(c, blk, p % SH_WGS, fx.Int64(ei), True)


def _block_order(c, b):
    """Workgroup block slot b -> sorted block: the shared expert's blocks (the sort's last, one per 32 rows) first,
    so its tiles, on the dense layout's slower stream, start earliest."""
    nb, ns = _shared_blocks(c)
    return (b < nb).select((b + nb - ns) % fx.max(nb, 1), b)


@flyc.jit
def _reduce(c, r):
    """The parts variant's reduce workgroup r (after every G2 slot): once output group (r // SLABS, r % SLABS) has
    every route (its count at the tile's target), zeroes the count and writes the group's output rows."""
    tile, slab = r // SLABS, r % SLABS
    if tile * 16 < c["m"]:
        if c["tid"] == 0:
            cnt = c["args"]["ws"] + fx.Int64(GROUP_AT + r * LINE)
            want = ld_i32(c["args"]["target"], tile)
            v = _ld_agent(cnt)
            while v < want:
                rocdl.s_sleep(1)
                v = _ld_agent(cnt)
            _st_agent(cnt, 0)
        gpu.barrier()
        pushed = _hook_parts(c, tile, slab)
        if const_expr(c["push"]):
            if slab < c["push"]:  # the first push_slabs slabs only (a workgroup-uniform branch)
                _push(c, tile, slab, pushed)


@flyc.jit
def _run(c, bid):
    g1 = c["blocks"] * _g1_wgs(c)
    if bid < g1:
        gemm1(c, _block_order(c, bid // _g1_wgs(c)), bid % _g1_wgs(c))
    elif bid < g1 + SH_SLOTS:
        gemm2_shared(c, bid - g1)
    else:
        b2 = bid - g1 - SH_SLOTS
        if const_expr(c["parts"] and c["push"] > 0):
            # the push builds go slab by slab (each slab's output groups complete, and are pushed, in turn): every
            # block's G2 workgroup of slab q, then slab q's reduce workgroups
            per = c["blocks"] + TOKEN_TILES
            q, w = b2 // per, b2 % per
            if w < c["blocks"]:
                gemm2(c, _block_order(c, w), q)
            else:
                _reduce(c, (w - c["blocks"]) * SLABS + q)
        elif const_expr(c["parts"]):
            if b2 < c["blocks"] * G2_WGS:
                gemm2(c, _block_order(c, b2 // G2_WGS), b2 % G2_WGS)
            else:
                _reduce(c, b2 - c["blocks"] * G2_WGS)
        else:
            gemm2(c, _block_order(c, b2 // G2_WGS), b2 % G2_WGS)


@functools.cache
def build(kind: str, parts: bool = False, push: int = 0):
    """The launch of ``kind`` (``PREFETCH``'s keys): G1 workgroups of 128 columns ("wide") or 64; ``parts``: the
    atomic-free combine (``shared`` is then the parts buffer); ``push`` (with ``parts``, 12 or 16): each reduce
    workgroup of slab s < push also writes its output group to ``peer_out`` and then the epoch (read once from
    ``epoch_ptr``) to ``peer_flags[16 t + s]`` (0: no push; the other builds ignore those three arguments)."""
    assert parts or not push
    wide = kind == "wide"
    keyed = (("sources", SOURCES), ("kind", kind), ("prefetch", PREFETCH[kind]), ("parts", parts),
             ("push", push))  # fmt: skip

    @fx.struct
    class Lds:
        words: fx.Array[fx.Int32, LDS_WORDS + (BM * PART_ROW if parts else 0), 16]

    name = "glm53f_moe_experts_" + kind + ("_parts" if parts else "") + (f"_push{push}" if push else "")

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kexperts(
        a1: Int64, a1s: Int64, a1e: Int64, a1es: Int64, sids: Int64, sw: Int64, seids: Int64, nvalid: Int64,
        w13: Int64, w2: Int64, s13: Int64, s2: Int64, w13sh: Int64, s13sh: Int64, w2sh: Int64, s2sh: Int64,
        out: Int64, shared: Int64, mid: Int64, ws: Int64, target: Int64, m: Int32, blocks: Int32,
        limit: Float32, epoch_ptr: Int64, peer_out: Int64, peer_flags: Int64,
    ):  # fmt: skip
        _ = keyed
        tid = fx.Int32(fx.thread_idx.x)
        lds = fx.SharedAllocator().allocate(Lds).peek()
        args = {
            "a1": a1, "a1s": a1s, "a1e": a1e, "a1es": a1es, "sids": sids, "sw": sw, "seids": seids, "w13": w13,
            "w2": w2, "s13": s13, "s2": s2, "w13sh": w13sh, "s13sh": s13sh, "w2sh": w2sh, "s2sh": s2sh, "out": out,
            "shared": shared, "mid": mid, "ws": ws, "target": target, "peer_out": peer_out, "peer_flags": peer_flags,
        }  # fmt: skip
        c = {
            "args": args, "tid": tid, "lane": tid % 64, "wave": uniform(tid // 64), "m": m, "blocks": blocks,
            "limit": limit, "nv": uniform(ld_i32(nvalid, 0)), "lds": lds.words.ptr, "wide": wide,
            "pf": PREFETCH[kind], "parts": parts, "push": push,
        }  # fmt: skip
        if const_expr(push):
            c["epoch"] = uniform(ld_i32(epoch_ptr, 0))
        _run(c, fx.Int32(fx.block_idx.x))

    @flyc.jit
    def launch(
        a1: Int64, a1s: Int64, a1e: Int64, a1es: Int64, sids: Int64, sw: Int64, seids: Int64, nvalid: Int64,
        w13: Int64, w2: Int64, s13: Int64, s2: Int64, w13sh: Int64, s13sh: Int64, w2sh: Int64, s2sh: Int64,
        out: Int64, shared: Int64, mid: Int64, ws: Int64, target: Int64, m: Int32, blocks: Int32,
        limit: Float32, epoch_ptr: Int64, peer_out: Int64, peer_flags: Int64, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        kexperts(
            a1, a1s, a1e, a1es, sids, sw, seids, nvalid, w13, w2, s13, s2, w13sh, s13sh, w2sh, s2sh, out, shared, mid,
            ws, target, m, blocks, limit, epoch_ptr, peer_out, peer_flags,
        ).launch(
            grid=(blocks * (I // (128 if wide else 64) + G2_WGS) + SH_SLOTS + (TOKEN_TILES * SLABS if parts else 0),),
            block=(THREADS,), stream=stream,
        )

    return launch


@functools.cache
def workspace(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(counters, zeroed and left zeroed by every launch; the intermediate; the shared expert's rows; the parts
    variant's per-route rows, zeroed and left zeroed). Made once a device, outside graph capture (SGLang warms up
    eagerly first)."""
    return (
        torch.zeros(COUNTER_WORDS, dtype=torch.int32, device=device),
        torch.empty(MID_BYTES, dtype=torch.uint8, device=device),
        torch.empty(MAX_ROWS, D, dtype=torch.bfloat16, device=device),
        torch.zeros(MAX_ROWS, PART_ROWS, D, dtype=torch.bfloat16, device=device),
    )


def tile_targets(sorted_ids: torch.Tensor, num_valid_ids: torch.Tensor, m: int) -> torch.Tensor:
    """int32 [16]: 2 x the sorted blocks holding a route of a token of each 16-token tile (static shapes, so it
    captures in a graph)."""
    rows = sorted_ids.view(-1, BM)
    tok = rows & 0xFFFFFF
    blk = torch.arange(rows.shape[0], device=rows.device).unsqueeze(1)
    live = (tok < m) & (blk * BM < num_valid_ids[0])
    tile = torch.where(live, tok // 16, TOKEN_TILES).long()
    hit = torch.zeros(rows.shape[0], TOKEN_TILES + 1, dtype=torch.int32, device=rows.device).scatter_(1, tile, 1)
    return (hit[:, :TOKEN_TILES].sum(0) * 2).to(torch.int32)


def experts(
    a1: torch.Tensor,
    a1_scale: torch.Tensor,
    a1_even: torch.Tensor,
    a1_even_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_shared: torch.Tensor,
    w13_shared_scale: torch.Tensor,
    w2_shared: torch.Tensor,
    w2_shared_scale: torch.Tensor,
    moe_buf: torch.Tensor,
    swiglu_limit: float | None,
    tile_target: torch.Tensor,
    push: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    push_slabs: int = 16,
) -> torch.Tensor:
    """The experts on the front's outputs (``front``'s names): a1 [M, 2048] fp4x2 and its per-token e8m0 [M, 128],
    the same with even-rounded scales for the shared expert (``a1_even``, ``a1_even_scale``),
    the sort, Quark's w4a4 routed weights [288, ...] as aiter shuffles them, the shared expert's dense weights as
    SGLang's linear layers hold them (gate_up [2048, 2048] fp4x2 / [2048, 128] e8m0, down [4096, 512] / [4096, 32],
    row-major), ``moe_buf`` [M, 4096] bf16 zeroed (added
    into), ``tile_target`` int32 [16] (the front's, or ``tile_targets``). ``push`` = (peer_out [M, 4096] bf16,
    peer_flags uint32 [256], epoch_ptr uint32 [1]), from PARTS_ROWS rows only: the push build also writes each
    finished 16-token x 256-column output group of the first ``push_slabs`` (12 or 16) 256-column slabs to peer_out
    and then the epoch to its flag. Returns ``moe_buf``."""
    m = a1.shape[0]
    blocks = sorted_expert_ids.shape[0]
    assert 1 <= m <= MAX_ROWS and blocks <= MAX_BLOCKS and sorted_ids.shape[0] == blocks * BM
    assert w13.shape[0] == NE and w13.numel() * w13.element_size() == NE * W13_BYTES
    assert w2.numel() * w2.element_size() == NE * W2_BYTES and moe_buf.shape == (m, D)
    assert w13_shared.numel() * w13_shared.element_size() == W13_BYTES and w13_shared_scale.numel() == S13_BYTES
    assert w2_shared.numel() * w2_shared.element_size() == W2_BYTES and w2_shared_scale.numel() == S2_BYTES
    counters, mid, shared_rows, parts = workspace(a1.device)
    use_parts = m >= PARTS_ROWS
    rows_out = parts if use_parts else shared_rows
    assert push is None or use_parts, "the push build covers the atomic-free combine only"
    peer_out, peer_flags, epoch = push if push is not None else (moe_buf, counters, counters)
    assert push_slabs in (12, 16)
    kind = "wide" if m >= WIDE_ROWS else "narrow" if m >= DEEP_ROWS else "deep"
    build(kind, use_parts, push_slabs if push is not None else 0)(
        a1.data_ptr(), a1_scale.data_ptr(), a1_even.data_ptr(), a1_even_scale.data_ptr(), sorted_ids.data_ptr(),
        sorted_weights.data_ptr(), sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), w13.data_ptr(),
        w2.data_ptr(), w13_scale.data_ptr(), w2_scale.data_ptr(), w13_shared.data_ptr(), w13_shared_scale.data_ptr(),
        w2_shared.data_ptr(), w2_shared_scale.data_ptr(), moe_buf.data_ptr(), rows_out.data_ptr(),
        mid.data_ptr(), counters.data_ptr(),
        tile_target.data_ptr(), m, blocks, float(swiglu_limit) if swiglu_limit else float("inf"), epoch.data_ptr(),
        peer_out.data_ptr(), peer_flags.data_ptr(), torch.cuda.current_stream(),
    )  # fmt: skip
    return moe_buf
