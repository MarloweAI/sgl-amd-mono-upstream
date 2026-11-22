# SPDX-License-Identifier: Apache-2.0
"""``mla_decode_c64``: one Kimi-K3 MLA layer's DCP8 verify decode, e4m3 Q K and
e4m3 P V, on 64-column tiles with the QK and P V work on separate waves. Same
inputs and outputs as ``mla_decode_b``: every row's o (bf16) and LSE (fp32).

One launch, ``BLOCKS`` CTAs of 8 waves (two a SIMD, a CTA a CU), persistent.

    tile      a request's 2 verify rows x 32 heads (64 columns, 9 tiles a
              request); column tile ct (16 columns) is row 2 rp + ct / 2, heads
              32 ht + 16 (ct % 2) ..
    piece     a tile's 64-key blocks [b0, b1); a CTA takes the step plan's list
              entries in turn with one atomic. A split tile's last piece waits
              for its other pieces' fp32 partials and merges the tile
    roles     waves 0-3 (S): QK and softmax, wave s the keys 16 s .. 16 s + 15
              of every block for all 64 columns (q in registers); waves 4-7 (O):
              P V, wave w the latent dims 128 w .. for all 64 columns, and the
              latent part of every block's loads; each role has its own loop,
              so its registers are its own
    a block   64 keys in an LDS ring of three slots, a key's record in groups of
              128 bytes (dims 128 m ..; the rope's 64 bytes the fifth),
              group-major: group m of key k at 8 KB m + 128 k, its 16-byte
              chunks XOR-swizzled by (k / 2) % 8 (the rope's by (k / 4) % 4). A
              block is loaded two barrier intervals before it is stored and
              stored the interval before its QK (both by the O waves)

One barrier interval a block. Interval t: the S waves mask block t's scores
(from interval t - 1) and exchange their column maxes through LDS (a counter,
inside the interval), then take block t's softmax (the exact running max m(t),
P = e4m3(2^(s c - m(t) + PRE)) to LDS as [column][key], each wave's row sum
from the unrounded P), all among the QK MFMAs of block t + 1 (S^T [16 keys x 16
columns] tiles: dims 0-511 on v_mfma_scale_f32_16x16x128, the rope on
v_mfma_f32_16x16x32_fp8); the O waves take P V of block t -
1, O^T [128 x 64] += V^T P^T on v_mfma_scale_f32_32x32x64 (V^T read transposed
by ds_read_b64_tr_b8) after the rescale 2^(m(t - 2) - m(t - 1)).

Numerics as ``mla_decode_b``: e4m3 x e4m3 products into fp32, the exact running
max of every 64-key block, P carrying a static 2^PRE, the row sum from the fp32
P, LSE natural-log fp32.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from sgl_amd_mono.common.ops import CM_DEV, ballot, butterfly, hw_exp2, permlane_swap, read_lane_i32, rsrc, uniform, xred
from sgl_amd_mono.common.plan import BLOCKS, kernel_sources, key_tuple
from sgl_amd_mono.kimi_k3.gfx950.attention import mla_decode as md

SOURCES = kernel_sources(__name__)

NHA, KV, DQK, G, DCP = md.NHA, md.KV, md.DQK, md.G, md.DCP
THREADS = 512
BK = 64  # keys a block
GRP_W = BK * 32  # words a 128-byte group of a block's 64 keys
ROPE_W = BK * 16  # words the rope group (64 bytes a key)
SLOT_W = 4 * GRP_W + ROPE_W
NSLOT = 3
TC = 64  # columns a tile
RT, HT = 2, 32  # verify rows, heads a tile
TILES = (G // RT) * (NHA // HT)  # 9 tiles a request
PROW = 20  # words a column of P in LDS: 64 keys, then a pad
PRE = 8.0
NEG = -1e30
NINF = float("-inf")
LN2 = 0.6931471805599453
LOG2E = 1.4426950408889634
MAX_REQS = md.MAX_REQS
# a piece's fixed cost (setup, the pipeline's two extra intervals, the epilogue), a split piece's extra (its
# partial's write, the wait, the slice merge), and the merges' shared traffic (blocks a split piece / C_G);
# blocks of ~2.65K ticks, from the measured timelines
C_P, C_W, C_G = 9, 4, 25
MAX_SLOTS = 1024  # split pieces' fp32 partials, all tiles together
MAX_KP = 32  # pieces a tile at most
STAGE_V = 8192  # 16-byte chunks of partials a merge round stages in LDS (the ring)
MAX_LAYERS = 64
MAX_TILES = TILES * MAX_REQS
CNT_Q = MAX_LAYERS  # a queue counter a layer, then MAX_TILES words a layer:
CNT_H = CNT_Q  # (from CNT_Q + MAX_TILES layer) a counter a head: its W_UV pair's units done
CNT_X = CNT_Q + MAX_LAYERS * MAX_TILES  # then 8 queue counters a layer, one an XCD's list
CNT_G = CNT_X + 8 * MAX_LAYERS  # then 4 counters a layer: the pieces done a head group (32 heads, 3 groups)
CNT_U = CNT_G + 4 * MAX_LAYERS  # then a layer's W_UV unit queue
CNT_F = CNT_U + MAX_LAYERS  # then a layer's group-done flags: 3 groups x 16 copies (a few units poll a copy)
CNT_WORDS = CNT_F + 48 * MAX_LAYERS
NU = 2 * NHA  # W_UV units: (head, latent half)
SM0 = 7  # the S interval's MFMA at which block t's softmax starts (after the column maxes' LDS read)
QSW = KV // 4 + 4  # words a column of a piece's q (dims 0-511, e4m3) staged in ring slot 2: a pad, so b128 reads are conflict free
OBW, RDW = 132, 36  # LDS words a row: a unit's bf16 o (128 + a pad), its k-half products (32 + a pad)
WVW = 80  # LDS words a latent row of a unit's W_UV half (64 + a pad: a transposed read of 16 lanes is conflict free)
HO = 12  # heads a DCP owner (the emit hooks' owner, head = HO owner + hl)
WO = 128  # W_UV's output dims a head
MAX_ENTRIES = 4096
WQ_HDR = 4  # the list: its pieces (twice, then 2 words free); 4 words an entry
# then 8 lists, one an XCD: a key range's 9 tiles (piece q of request r: group G, in the list's order) go to XCD
# G % 8 as entries 9 (G / 8) + tile, so a key range's KV is fetched into one L2 and read there 9 times
XLEN = 9 * (-(-MAX_ENTRIES // 72))
WQ_X = WQ_HDR + 4 * MAX_ENTRIES
WQ_T = WQ_X + 8 * XLEN  # then a tile's split: first slot + 65536 x pieces, or -1 unsplit
WQ_WORDS = WQ_T + MAX_TILES
PLAN_THREADS = 320  # 10 groups of 32 lanes: group c evaluates candidate c, lane r request r
# runs (most blocks a piece) as fractions of the batch's work a CTA; (0, 1): every tile unsplit
PLAN_F = ((0, 1), (1, 2), (7, 10), (17, 20), (1, 1), (23, 20), (13, 10), (8, 5), (2, 1))
NCLS = MAX_REQS  # piece classes: a request's pieces


OVW = KV + 4  # words a column of a piece's fp32 partial in LDS (a pad against bank conflicts)
PART_E = MAX_SLOTS * TC * KV  # the partials' fp32 elements (an offset there reads 0 and fetches nothing)
TL_W = 128  # timeline words a CTA: 32 each from waves 0, 1 (S), 4, 5 (O)
# the W_UV stage's timeline (probe "ut" with "tl"), past the CTAs' words: 16 a CTA, then the groups' flag
# times (16), then 32 a unit
TL_UB = 256 * TL_W
TL_UG = TL_UB + 256 * 16
TL_UU = TL_UG + 16

_BUILDS: dict = {}
_STREAM = fx.Stream(None)

_ld, _ld4, _lds16, _tr8, _cat8, _e4m3x4, _memtime, _realtime, _bar = (
    md._ld, md._ld4, md._lds16, md._tr8, md._cat8, md._e4m3x4, md._memtime, md._realtime, md._bar,
)  # fmt: skip


@dataclass(frozen=True)
class MlaDecodeBuild:
    scale: float  # the softmax scale
    # timing probes only (checks): parts of a block's work left out
    probe: str = ""


_AGPR_ALLOC: dict = {}  # kernel symbol -> its "amdgpu-agpr-alloc"


def _agpr_hint_lowering():
    """The compile hint ``agpr_alloc``: LLVM's "amdgpu-agpr-alloc" on the kernels of a
    build that carries it, no other build touched. FlyDSL's convert-gpu-to-rocdl drops a
    gpu.func's passthrough, so the hinted kernels are named at hint lowering and their
    llvm.func gets the attribute once it exists (a hinted module's pipeline runs a
    fragment at a time)."""
    from flydsl.compiler import jit_function as jf
    from flydsl.compiler.backends import rocm as fr

    if getattr(fr.RocmBackend, "_agpr_alloc_hint", False):
        return
    lower, run_pm, run_pipeline = fr.RocmBackend.lower_compile_hints, jf._run_pass_manager, jf._run_pipeline

    def lower_with_agpr(self, module, *, compile_hints):
        lower(self, module, compile_hints=compile_hints)
        n = compile_hints.get("agpr_alloc")
        if n is not None:
            for func_op in fr._iter_gpu_kernel_funcs(module):
                _AGPR_ALLOC[_str(func_op.attributes["sym_name"])] = str(n)

    def hinted(module):
        for top in module.body.operations:
            if top.operation.name == "gpu.module":
                for op in top.regions[0].blocks[0].operations:
                    if op.operation.name in ("gpu.func", "llvm.func") and "sym_name" in op.attributes:
                        name = _str(op.attributes["sym_name"])
                        if name in _AGPR_ALLOC:
                            yield op, _AGPR_ALLOC[name]

    def run_pm_agpr(pm, module):
        run_pm(pm, module)
        with module.context:
            for op, n in hinted(module):
                if op.operation.name == "llvm.func":
                    fr._set_passthrough(op, "amdgpu-agpr-alloc", n)

    def run_pipeline_agpr(module, fragments, **kw):
        if not any(True for _ in hinted(module)):
            return run_pipeline(module, fragments, **kw)
        for frag in fragments:
            run_pipeline(module, [frag], **kw)

    fr.RocmBackend.lower_compile_hints = lower_with_agpr
    jf._run_pass_manager = run_pm_agpr
    jf._run_pipeline = run_pipeline_agpr
    fr.RocmBackend._agpr_alloc_hint = True


def _str(attr):
    from flydsl._mlir import ir

    return ir.StringAttr(attr).value


def _st16(ptr, v):
    llvm.store(as_ir_value(v), ptr.llvm_ptr, alignment=16)


def _xcc():
    """The CTA's XCD (XCC_ID)."""
    v = llvm.inline_asm(res=T.i32, operands_=[], asm_string="s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)",
                        constraints="=s", has_side_effects=True)  # fmt: skip
    return fx.Int32(v) & 15


def _atomic_add(addr, val):
    """Device-scope atomic add of ``val`` on the int32 at global address ``addr``; its old value."""
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(addr)).result
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, ptr, as_ir_value(fx.Int32(val)), llvm.AtomicOrdering.monotonic, syncscope="agent"
    )  # fmt: skip
    return fx.Int32(op.result)


def _lds_add(ptr, val):
    """Workgroup-scope atomic add (acquire-release) on the LDS int32 at ``ptr``; its old value."""
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, ptr.llvm_ptr, as_ir_value(fx.Int32(val)), llvm.AtomicOrdering.acq_rel, syncscope="workgroup"
    )  # fmt: skip
    return fx.Int32(op.result)


def _lds_acquire(ptr):
    """Workgroup-scope acquire load of the LDS int32 at ``ptr`` (one read for the wave)."""
    return fx.Int32(
        llvm.load(T.i32, ptr.llvm_ptr, alignment=4, ordering=llvm.AtomicOrdering.acquire, syncscope="workgroup")
    )


def _opaque(v):
    """``v`` through a v_mov LLVM cannot see past: an epilogue's lane constants are computed there,
    not hoisted out of the piece loops (where they would be live, and spill, through the loops)."""
    return fx.Int32(
        llvm.inline_asm(res=T.i32, operands_=[as_ir_value(fx.Int32(v))], asm_string="v_mov_b32 $0, $1",
                        constraints="=v,v", has_side_effects=True)
    )  # fmt: skip


def _dma(rs, lds_ptr, voff, nbytes):
    """One wave-wide LDS DMA (``nbytes`` a lane, lane i's at ``lds_ptr`` + nbytes i), device-coherent."""
    rocdl.raw_ptr_buffer_load_lds(rs, lds_ptr.llvm_ptr, nbytes, voff, fx.Int32(0), 0, CM_DEV)


def _words(b):
    """A non-negative byte offset in 32-bit words (a shift, not a signed division)."""
    return fx.Int32(fx.Uint32(b) >> fx.Uint32(2))


def _tr16(ptr):
    """ds_read_b64_tr_b16 at a lane's 8-byte aligned LDS ``ptr``: 4 bf16 as 2 i32."""
    return fx.Vector(rocdl.ds_read_tr16_b64(T.vec(4, T.i16), ptr.llvm_ptr).result).bitcast(fx.Int32)


def _ldsf4(ptr):
    return fx.Vector(llvm.load(T.vec(4, T.f32), ptr.llvm_ptr, alignment=16))


def _mfma32(a, b, c):
    one = fx.Int32(127)
    return fx.Vector(rocdl.mfma_scale_f32_32x32x64_f8f6f4(T.vec(16, T.f32), [a, b, c, 0, 0, 0, one, 0, one]))


def _mfma16(a, b, c):
    one = fx.Int32(127)
    return fx.Vector(rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one]))


def _mfma16r(a, b, c):
    """16x16x32 e4m3 (the rope): a, b one i64 each."""
    return fx.Vector(rocdl.mfma_f32_16x16x32_fp8_fp8(T.vec(4, T.f32), [a, b, c, 0, 0, 0]))


@fx.struct
class Smem:
    kv: fx.Array[fx.Int32, max(NSLOT * SLOT_W, TC * OVW), 16]  # the ring; at a split piece's end its partial
    p: fx.Array[fx.Int32, 2 * TC * PROW, 16]  # P of blocks b % 2, [column][key]
    red: fx.Array[fx.Float32, 2 * TC * 4, 16]  # block b % 2's column maxes, [column][S wave]
    mrun: fx.Array[fx.Float32, 4 * TC, 16]  # the running max after block b % 4
    qr: fx.Array[fx.Int32, TC * 16, 16]  # q's rope (64 bytes a column)
    fin: fx.Array[fx.Float32, TC * 4, 16]  # each S wave's row sums, [column][S wave]
    olse: fx.Array[fx.Float32, TC, 16]  # a split piece's LSE a column
    wt: fx.Array[fx.Float32, (MAX_KP + 1) * TC, 16]  # the merge's weights [piece][column], then 1 / their sum
    nxt: fx.Array[fx.Int32, 8, 16]  # the next entries taken: pieces, W_UV units (by parity); a pair's count
    rcnt: fx.Array[fx.Int32, 4, 16]  # the S waves' block maxes written so far this piece


# The decode + W_UV stage, for a caller's launch of BLOCKS x THREADS threads (every CTA calls it once a layer):
#   key: an MlaDecodeBuild; lds: this module's Smem (``fx.SharedAllocator().allocate(Smem).peek()`` or a caller's
#   union member); bid, tid: the CTA and thread;
#   a: {"kv", "kv_idx", "kv_ptr", "g_ptr", "rank", "o", "lse", "part", "plse", "wq", "cnt", "layer", "wuv", "half",
#       "tl"}: the rank's pool rows, its index and pointer tables (as ``mla_decode_launch``), the rank, scratch (bf16
#       o [S, 96, 512] and fp32 LSE [S, 96] of unsplit tiles, the merged LSE of split ones; the split pieces' fp32
#       partials and LSEs; fp32 [S, 96, 2, 128] for the two latent halves), the step's work list and the counters
#       (``DecodeBuffers``), the MLA layer's index, W_UV (bf16 [96][512 latent][128 out]) and the timeline (0: none);
#   the hooks (static callables, traced into the caller's kernel):
#     q_load(rq, head, t, byte) -> 16 bytes (i32 x 4): q of request rq, verify row t (0 .. 5), at byte offset
#       ``byte`` (a multiple of 16) of its 576-byte e4m3 row;
#     ready(rq, h0, nh): by the S waves (tid < 256) at each piece's start, before its q loads (heads h0 .. h0 + nh);
#       no CTA barrier inside;
#     kv_ready(rq): by the O waves (tid >= 256) before the first record loads that read the request's last block;
#       no CTA barrier inside;
#     emit_values(owner, hl, t, col, vals): by one thread for head HO owner + hl, step row t (0 .. S - 1), out dims
#       col .. col + 7 (col a multiple of 8): vals, 8 fp32 (half 0 + half 1 of the latent);
#     emit_lse(owner, hl, t, lse): the head's merged natural-log fp32 LSE of row t (its key range on this rank),
#       once a (row, head).
@flyc.jit
def decode_stage(key, lds, bid, tid, a, q_load, ready, kv_ready, emit_values, emit_lse):
    qk_c = key.scale * LOG2E
    probe = key.probe
    tlon = "tl" in probe
    # timing-only idealisations (wrong numerics): "nordy" no readiness waits, "capN" every piece at most N blocks, "zq" no
    # q loads, "zw" no W_UV loads
    if const_expr("nordy" in probe):
        ready = lambda rq, h0, nh: None  # noqa: E731
        kv_ready = lambda rq: None  # noqa: E731
    cap = int(probe.split("cap")[1].split(",")[0]) if "cap" in probe else 0
    if const_expr("zq" in probe):
        q_load = lambda rq, head, t, byte: fx.Vector.filled(4, 0, fx.Int32)  # noqa: E731
    phon = "ph" in probe  # phase stamps inside an interval (with "tl")
    uton = tlon and "ut" in probe  # the W_UV stage's stamps
    kv, kv_idx, kv_ptr, g_ptr, rank = a["kv"], a["kv_idx"], a["kv_ptr"], a["g_ptr"], a["rank"]
    o, lse, part, plse, wq, cnt, layer = a["o"], a["lse"], a["part"], a["plse"], a["wq"], a["cnt"], a["layer"]
    wuv, half, tl = a["wuv"], a["half"], a["tl"]
    z = fx.Int64(0)
    tk0 = _memtime() if const_expr(tlon) else z
    rk0 = _realtime() if const_expr(tlon) else z
    tw, tb = z, z  # a wave's work and barrier waits in the loop
    t_set, t_epi, n_bl, n_un, t_pe = z, z, z, z, z
    pe_x, pe_w, pe_m, n_last = z, z, z, z  # the piece end: to its first barrier, the wait / write, the rest
    e_pre, e_post = z, z  # a piece's loop end to its piece end; its piece end to the next piece
    phs = [z] * 4  # phase sums inside the loop's intervals
    t_rdy, r_rdy = z, z  # S: the ready hook's waits; the realtime the first one returned

    def stamp():
        return _memtime() if const_expr(phon) else z
    lane = tid % 64
    wave = uniform(tid // 64)
    role = wave // 4  # 0: S, 1: O
    w = wave % 4  # S: keys 16 w .. of a block; O: dims 128 w ..
    n16, g = lane % 16, lane // 16
    tcol = 2 * (g % 2) + g // 2  # S: the column tile a lane holds after the max transpose
    n32, hb = lane % 32, lane // 32
    ring, pall, red, mrun, fin, qrl = lds.kv.ptr, lds.p.ptr, lds.red.ptr, lds.mrun.ptr, lds.fin.ptr, lds.qr.ptr
    # a split piece's fp32 partial at the piece's end, [column][OVW] over the ring
    ovl, olse = lds.kv.ptr, lds.olse.ptr
    r_kv = rsrc(kv)

    kq = 16 * w + n16  # S: the lane's key of the wave's 16 (QK's A operand row)

    def kop(kbp, st):
        """QK step st's A operand (dims 128 st + 32 g .., two chunks) of key kq, from the slot at
        ``kbp``."""
        base = kbp + st * GRP_W + kq * 32
        sw = (kq // 2) % 8
        return _cat8(_lds16(base + ((2 * g) ^ sw) * 4), _lds16(base + ((2 * g + 1) ^ sw) * 4))

    def kop_rope(kbp):
        """The rope MFMAs' A operands of key kq: its rope chunk g (dims 512 + 16 g ..), half r for
        MFMA r."""
        return _lds16(kbp + 4 * GRP_W + kq * 16 + (g ^ ((kq // 4) % 4)) * 4).bitcast(fx.Int64)

    # P V's V^T reads: transposed read rho of MFMA d, lane l: key 32 hb + 8 rho + grp / 2 (grp =
    # l % 16) at chunk (2 d + (l / 16) % 2) ^ ((key / 2) % 8): one of two per-lane words plus a
    # constant
    v_grp = lane % 16
    v_a0 = ((lane // 16) % 2) ^ ((v_grp // 4) % 2)
    v_a1 = v_grp // 8
    v_base = w * GRP_W + 1024 * hb + 32 * (v_grp // 2) + 2 * (v_grp % 2) + 4 * v_a0
    v_c = [v_base + 8 * v_a1, v_base + 8 * (1 - v_a1)]

    def vop(kbp, d):
        """P V MFMA d's A operand (V^T, dims 128 w + 32 d + n32; byte e = key 32 hb + e)."""
        vs = [_tr8(kbp + v_c[d % 2] + (256 * rho + 16 * ((d // 2) ^ (rho % 2)))) for rho in range(4)]
        return fx.Vector.from_elements([vs[r_][w_] for r_ in range(4) for w_ in range(2)], fx.Int32)

    def kv_rows(bb, kvp, L):
        """Lane l: the byte offset of absolute block bb's key l's record (past the request its
        last key; L >= 1)."""
        kk = fx.max(fx.Int32(0), fx.min(bb * BK + lane, L - 1))
        if const_expr("hot" in probe):
            kk = fx.Int32(0)
        return _ld(kv_idx, kvp + kk) * DQK

    # O wave w (the only waves that feed): group w (latent dims 128 w .., the bytes its V^T reads
    # cover) of keys 8 i + lane / 8, i = 0 .. 7 (8 keys x 128 bytes a load, LDS chunk lane % 8 from
    # the record's chunk (lane % 8) ^ ((4 i + lane / 16) % 8)), and the rope of keys 16 w + lane / 4
    # (16 keys x 64 bytes, chunk (lane % 4) ^ ((lane / 16) % 4))
    NLD = 9

    def kv_offs(rall):
        """This O wave's 9 load offsets (words) of a block (``rall``: lane l key l's record offset)."""
        rows = [fx.Int32(rocdl.ds_bpermute(T.i32, (8 * i + lane // 8) * 4, rall)) for i in range(8)]
        rows.append(fx.Int32(rocdl.ds_bpermute(T.i32, (16 * w + lane // 4) * 4, rall)))
        rocdl.sched_barrier(0)
        out = [_words(rows[i] + 128 * w + ((lane % 8) ^ ((4 * i + lane // 16) % 8)) * 16) for i in range(8)]
        out.append(_words(rows[8] + 512 + ((lane % 4) ^ ((lane // 16) % 4)) * 16))
        return out

    def kv_ld(off, dev):
        """A 16-byte record load; ``dev``: device scope (SC1), for the block ``kv_ready`` guards: its new
        rows were written in this launch from another XCD, and the cache lines they share with rows read
        earlier can sit stale in this XCD's L2 or this CU's L1."""
        if const_expr("noload" in probe):
            return fx.Vector.filled(4, 0, fx.Int32)
        if const_expr(dev):
            return fx.Vector(bo.buffer_load(r_kv, off, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV))
        return fx.Vector(bo.buffer_load(r_kv, off, vec_width=4, dtype=T.i32))

    def kv_st(v, b, i):
        """Load i of ``kv_offs`` into relative block b's slot (each load's 1 KB contiguous)."""
        if const_expr("noload" not in probe):
            sl = ring + (b % NSLOT) * SLOT_W
            if const_expr(i < 8):
                _st16(sl + w * GRP_W + 256 * i + lane * 4, v)
            else:
                _st16(sl + 4 * GRP_W + lane * 4 + w * 256, v)

    def kv_load(rall, dev=False):
        return [kv_ld(off, dev) for off in kv_offs(rall)]

    def kv_store(stg, b):
        for i in range_constexpr(NLD):
            kv_st(stg[i], b, i)

    def feed_prologue(b0, kvp, L, rq):
        """Block 0 into the ring; block 1 into registers; block 2's rows."""
        rs = [kv_rows(b0 + j, kvp, L) for j in range(3)]
        s0 = [fx.Vector.filled(4, 0, fx.Int32)] * NLD
        st = s0
        if (L - 1) // BK - b0 <= 1:
            kv_ready(rq)  # (loads past the request's last key read that key)
            s0 = kv_load(rs[0], dev=True)
            st = kv_load(rs[1], dev=True)
        else:
            s0 = kv_load(rs[0])
            st = kv_load(rs[1])
        kv_store(s0, fx.Int32(0))
        return st, rs[2]

    def feed(t, st, rw, b0, kvp, L, rq):
        """Interval t's feed: block t + 4's rows first (an interval ahead of their use, so waiting
        for them never drains the block loads), block t + 2 (loaded in interval t - 1) into its slot
        (block t - 1's), block t + 3 (``rw``: its rows) into registers."""
        rn = kv_rows(b0 + t + 4, kvp, L)
        kv_store(st, t + 2)
        if b0 + t + 3 == (L - 1) // BK:
            kv_ready(rq)
            st = kv_load(rw, dev=True)
        else:
            st = kv_load(rw)
        return st, rn

    def end_iv(ta, tw, tb):
        t1 = _memtime() if const_expr(tlon) else z
        _bar()
        if const_expr(tlon):
            t2 = _memtime()
            return t2, tw + (t1 - ta), tb + (t2 - t1)
        return ta, tw, tb

    def entry(ent):
        """Work-table entry ent: the piece's tile and blocks, its partial slot, its keys' bounds."""
        ew = WQ_HDR + 4 * ent
        u = uniform(_ld(wq, ew))
        b0 = uniform(_ld(wq, ew + 1))
        b1 = uniform(_ld(wq, ew + 2))
        sk = uniform(_ld(wq, ew + 3))  # < 0: the tile is unsplit; else its partial slot + 4096 x its pieces
        pq = u >> 12  # the piece's index in its tile
        u = u & 4095
        slot = (sk < 0).select(fx.Int32(-1), sk & 4095)
        kp = (sk < 0).select(fx.Int32(1), sk >> 12)
        rq = u // TILES
        rp = (u % TILES) // 3
        ht = u % 3
        kvp = _ld(kv_ptr, rq)
        L = _ld(kv_ptr, rq + 1) - kvp
        P = _ld(g_ptr, rq + 1) - _ld(g_ptr, rq) - G
        lim0 = fx.max(fx.Int32(0), fx.min(L, (P + 2 * rp - rank + 8 * DCP) // 8 - DCP + 1))
        lim1 = fx.max(fx.Int32(0), fx.min(L, (P + 2 * rp + 1 - rank + 8 * DCP) // 8 - DCP + 1))
        L = uniform(L)
        kvp = uniform(kvp)
        b1 = fx.min(b1, (uniform(lim1) + BK - 1) // BK)
        if const_expr(cap > 0):
            b1 = fx.min(b1, b0 + cap)
        nb = b1 - b0
        # intervals -1 .. nb (ni = nb + 2): interval t the softmax of block t and the QK of block t
        # + 1 (S), P V of block t - 1 (O), block t + 2 into its slot
        ni = nb + 2
        kend0 = fx.min(uniform(lim0), b1 * BK)
        kend1 = fx.min(uniform(lim1), b1 * BK)
        return u, rq, rp, ht, b0, b1, slot, kvp, L, nb, ni, kend0, kend1, pq, kp

    r_part = rsrc(part, fx.Int32(PART_E * 4))
    r_plse = rsrc(plse, fx.Int32(MAX_SLOTS * TC * 4))

    gcnt = cnt + fx.Int64(4 * (CNT_G + 4 * layer))

    def piece_end(u, rq, rp, ht, slot, pq, kp, nslot):
        """A piece's end, every wave: a split piece's fp32 partial (in ``ovl`` as [column][OVW], the LSE in
        ``olse``) to its slot (an unsplit piece's bf16 o and LSE are already stored); then, once every
        wave's stores are acknowledged, one count on its head group's counter. No piece waits: the W_UV
        units merge the partials."""
        ta_, tb_, lst = z, z, z  # (timeline)
        if slot >= 0:
            _bar()
            # wave v's store i: 1 KB of column 8 v + i / 2 (chunks 64 (i % 2) + lane)
            lx = _opaque(lane)
            for i0 in range_constexpr(0, 16, 4):
                cs = [8 * wave + i // 2 for i in range(i0, i0 + 4)]
                vs = [_ldsf4(ovl + cs[i] * OVW + 4 * (64 * ((i0 + i) % 2) + lx)) for i in range(4)]
                for i in range_constexpr(4 if "nopw" not in probe else 0):
                    bo.buffer_store(vs[i], r_part, (slot * TC + cs[i]) * KV + 4 * (64 * ((i0 + i) % 2) + lx),
                                    cache_modifier=CM_DEV)  # fmt: skip
                rocdl.sched_barrier(0)
            if wave == 0:
                bo.buffer_store(fx.Float32(fx.ptr_load(olse + lx)), r_plse, slot * TC + lx, cache_modifier=CM_DEV)
        rocdl.s_waitcnt(vmcnt=0)
        _bar()
        if const_expr(tlon):
            ta_ = _memtime()
        if tid == 0:
            # a head group holds npc / 3 pieces (3 tiles a request); its last piece raises the group's flags
            if _atomic_add(gcnt + fx.Int64(4) * fx.Int64(ht), 1) == npc // 3 - 1:
                for f in range_constexpr(16):
                    bo.buffer_store(fx.Int32(1), rsrc(cnt), CNT_F + 48 * layer + 16 * ht + f, cache_modifier=CM_DEV)
                if const_expr(uton):
                    bo.buffer_store(fx.Int32(_realtime()), rsrc(tl), TL_UG + ht)
        return ta_, tb_, lst

    def pe_add(slot, tpe, ta_, tb_, lst, t_pe, pe_x, pe_w, pe_m, n_last):
        te_ = _memtime()
        sp = slot >= 0
        return (t_pe + (te_ - tpe), pe_x + sp.select(ta_ - tpe, z), pe_w + sp.select(tb_ - ta_, z),
                pe_m + sp.select(te_ - lst, z), n_last + sp.select(lst - tb_, z), te_)  # fmt: skip

    # the list: pieces [0, npc); a CTA takes the next entry with one atomic on the layer's queue
    # counter (thread 0, into nxt[npk % 2], read by all after a barrier)
    npc = uniform(_ld(wq, 0))
    ngr = npc // TILES  # key-range groups
    qa = cnt + fx.Int64(4 * layer)
    cx = cnt + fx.Int64(4 * (CNT_X + 8 * layer))
    xcc = _xcc()
    nxt = lds.nxt.ptr

    def xlen(x):
        return TILES * fx.max(fx.Int32(0), (ngr - x + 7) // 8)

    def kfirst(x):
        """List x's entries taken first, without a take: entry j by CTA 8 j + x."""
        return fx.min(fx.Int32(BLOCKS // 8), xlen(x))

    kall = kfirst(0)
    for x_ in range_constexpr(1, 8):
        kall = kall + kfirst(x_)

    def take():
        """The next entry: this XCD's list first, then the other XCDs' in turn (only which CTA runs an
        entry depends on the take order; npc when every list is done). The layer's queue counter, taken
        beside the XCD's, counts takes: past npc every entry has a taker, so an empty list ends the
        search without walking the others (a take below npc walks them all, so none is left). A list's
        counter and the queue's count from past the entries taken first."""
        x = xcc
        gv = _atomic_add(qa, 1) + kall
        v = _atomic_add(cx + fx.Int64(4) * fx.Int64(x), 1) + kfirst(x)
        d = fx.Int32(0)
        while (v >= xlen(x)) & (gv < npc) & (d < 7):
            d = d + 1
            x = (xcc + d) % 8
            v = _atomic_add(cx + fx.Int64(4) * fx.Int64(x), 1) + kfirst(x)
        return (v < xlen(x)).select(_ld(wq, WQ_X + x * XLEN + fx.min(v, fx.Int32(XLEN - 1))), npc)

    if const_expr(uton):
        if tid == 0:
            bo.buffer_store(fx.Int32(_realtime()), rsrc(tl), TL_UB + bid * 16 + 4)
    if tid == 0:
        # the first entry: list bid % 8's entry bid / 8 if there is one (the CTAs of an XCD, when CTAs go to
        # XCDs in turn), else a take
        x0, j0 = bid % 8, bid // 8
        if j0 < xlen(x0):
            fx.ptr_store(_ld(wq, WQ_X + x0 * XLEN + j0), nxt)
        else:
            fx.ptr_store(take(), nxt)
    _bar()
    idx = uniform(fx.Int32(fx.ptr_load(nxt)))
    if const_expr(uton):
        if tid == 0:
            bo.buffer_store(fx.Int32(_realtime()), rsrc(tl), TL_UB + bid * 16 + 5)
    npk = fx.Int32(1)  # entries taken so far (the parity of nxt)
    # each role its own piece loop, so a role's per-lane constants live only in its loop
    if role == 0:
        while idx < npc:
            ta = _memtime() if const_expr(tlon) else z
            u, rq, rp, ht, b0, b1, slot, kvp, L, nb, ni, kend0, kend1, pq, kp = entry(idx)
            te_ = z
            nv = fx.Int32(0)
            if const_expr(tlon):
                n_bl = n_bl + fx.Int64(nb)
                n_un = n_un + 1
            if nb > 0:
                # S: q of column tiles ct (column n16: row 2 rp + ct / 2, head 32 ht + 16 (ct % 2) + n16):
                # 4 steps of dims 128 st + 32 g .., and the rope's dims 512 + 16 g ..
                tr0 = _memtime() if const_expr(tlon) else z
                ready(rq, HT * ht, HT)
                if const_expr(tlon):
                    t_rdy = t_rdy + (_memtime() - tr0)
                    r_rdy = (r_rdy == z).select(_realtime(), r_rdy)
                # the tile's q (dims 0-511 of its 64 columns, 32 KB) read once by the four S waves into ring slot 2
                # (free until interval 0's feed), each thread 8 chunks of 16 bytes (chunk k: column k / 32, bytes
                # 16 (k % 32) ..); column tile w's rope into LDS (column 16 w + n16, its chunk g)
                qsl = ring + 2 * SLOT_W
                qv = []
                for i in range_constexpr(8):
                    k = 256 * i + tid
                    col, ch = k // 32, k % 32
                    qv.append(q_load(rq, HT * ht + 16 * ((col // 16) % 2) + col % 16, RT * rp + col // 32, 16 * ch))
                qrv = q_load(rq, HT * ht + 16 * (w % 2) + n16, RT * rp + w // 2, 512 + 16 * g)
                rocdl.sched_barrier(0)
                for i in range_constexpr(8):
                    k = 256 * i + tid
                    _st16(qsl + (k // 32) * QSW + 4 * (k % 32), qv[i])
                _st16(qrl + (16 * w + n16) * 16 + 4 * g, qrv)
                _bar()
                # q of column tiles ct (column n16: row 2 rp + ct / 2, head 32 ht + 16 (ct % 2) + n16): 4 steps of
                # dims 128 st + 32 g ..
                qops = []
                for ct in range_constexpr(4):
                    qb = qsl + (16 * ct + n16) * QSW + 8 * g
                    qops.append([_cat8(_lds16(qb + 32 * st), _lds16(qb + 32 * st + 4)) for st in range(4)])
                # block -1's column maxes (none: the first interval's softmax is of no keys)
                fx.ptr_store(fx.Float32(NINF), red + TC * 4 + (16 * tcol + n16) * 4 + w)
                _bar()
                if const_expr(tlon):
                    tn = _memtime()
                    t_set = t_set + (tn - ta)
                    ta = tn
                xc = [fx.Float32(NINF)] * 16  # block t's scores, element 4 ct + v: key 16 w + 4 g + v
                mg = [fx.Float32(NEG)] * 4  # the running max (column tile ct's column n16)
                lw = [fx.Float32(0.0)] * 4  # the lane's row sums at mg

                def s_iv(t, xc, mg, lw, phs, tq0, nxt_max=1):
                    """S's interval t: the QK of block t + 1 (step by step over the four column tiles'
                    independent accumulators: dims 0-511 in 4 steps, then the rope's two halves) with, among
                    its MFMAs, block t's softmax (``xc``: its masked scores, from interval t - 1; the four S
                    waves' column maxes of block t, stored before the interval's barrier, then the running
                    max). After the MFMAs, block t + 1's column maxes over the wave's 16 keys to LDS
                    (``nxt_max`` 1; 2: masked to the piece and the rows' bounds; 0: none, the piece's last
                    block)."""
                    kb = ring + ((t + 1) % NSLOT) * SLOT_W
                    rb = red + ((t + 2) % 2) * (TC * 4)
                    rbn = red + ((t + 3) % 2) * (TC * 4)
                    pb = pall + ((t + 2) % 2) * (TC * PROW) + n16 * PROW + 4 * w + g
                    # the first two steps' A operands up front; the later steps' two steps ahead among the
                    # MFMAs (the interval's LDS reads spread over its MFMAs)
                    kops = [kop(kb, st) for st in range(2)] + [None, None]
                    kr = None
                    rocdl.sched_barrier(0)
                    tq1 = stamp()
                    sacc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
                    ps, lw2 = [], []
                    mn, off = [], []
                    qrs = None
                    kbase = (b0 + t) * BK + 16 * w + 4 * g
                    # MFMA i: step i / 4 of column tile i % 4 (steps 4, 5: the rope's halves). A tile's
                    # MFMAs are 4 apart: back to back on one accumulator they read it too early
                    for i in range_constexpr(24):
                        st, ct = i // 4, i % 4
                        if const_expr(i in (0, 4)):
                            kops[st + 2] = kop(kb, st + 2)
                        if const_expr(i == 8):
                            kr = kop_rope(kb)
                        if const_expr(st < 4):
                            if const_expr("noqk" not in probe):
                                sacc[ct] = _mfma16(kops[st], qops[ct][st], sacc[ct])
                        elif const_expr("noqk" not in probe):
                            sacc[ct] = _mfma16r(kr[st - 4], qrs[ct][st - 4], sacc[ct])
                        if const_expr(i == 0):
                            # the four S waves' block maxes of block t, then the running max after it
                            for c4 in range_constexpr(4):
                                r4 = _ldsf4(rb + (16 * c4 + n16) * 4)
                                # two 3-input maxes: (r0, r1, r2), then (that, r3, the running max)
                                b3 = fx.max(fx.max(fx.Float32(r4[0]), fx.Float32(r4[1])), fx.Float32(r4[2]))
                                mn.append(fx.max(fx.max(b3, fx.Float32(r4[3])), mg[c4]))
                                off.append(fx.Float32(PRE) - mn[c4])
                            if (w == 0) & (g == 0):
                                for c4 in range_constexpr(4):
                                    fx.ptr_store(mn[c4], mrun + ((t + 4) % 4) * TC + 16 * c4 + n16)
                        if const_expr(i == 11):
                            # the rope's q of the four column tiles
                            qrs = [_lds16(qrl + (16 * c4 + n16) * 16 + 4 * g).bitcast(fx.Int64) for c4 in range(4)]
                        if const_expr(SM0 <= i < SM0 + 16):
                            # softmax element e of block t: column tile e / 4, key 16 w + 4 g + e % 4
                            e = i - SM0
                            c4 = e // 4
                            if const_expr("nosm" in probe):
                                ps.append(xc[e] + off[c4])
                            else:
                                ps.append(hw_exp2(fmath.fma(xc[e], fx.Float32(qk_c), off[c4])))
                            if const_expr(e % 4 == 3):
                                q4 = ps[e - 3 : e + 1]
                                fx.ptr_store(_e4m3x4(q4), pb + 16 * c4 * PROW)
                                lw2.append(lw[c4] * hw_exp2(mg[c4] - mn[c4]) + ((q4[0] + q4[1]) + (q4[2] + q4[3])))
                        rocdl.sched_barrier(0)
                    tq2 = stamp()
                    xn = [fx.Float32(sacc[c][v]) for c in range(4) for v in range(4)]
                    if const_expr(nxt_max > 0):
                        # block t + 1's column tiles: masked scores (keys past the piece or the row's bound:
                        # -inf) and their column maxes over the wave's 16 keys
                        cm = []
                        for ct in range_constexpr(4):
                            if const_expr(nxt_max == 2):
                                room = kend0 - kbase - BK if ct < 2 else kend1 - kbase - BK
                                for v in range_constexpr(4):
                                    xn[4 * ct + v] = (fx.Int32(v) < room).select(xn[4 * ct + v], fx.Float32(NINF))
                            xs = xn[4 * ct : 4 * ct + 4]
                            cm.append(fx.max(fx.max(fx.max(xs[0], xs[1]), xs[2]), xs[3]))
                        # the four tiles' maxes over the wave's 16 keys (lane bits 4, 5) as one transpose:
                        # 32-swap tiles (0, 1) and (2, 3), then 16-swap the two results; lane l ends with
                        # tile (0, 2, 1, 3)[l / 16], column l % 16
                        a0, a1 = permlane_swap(32, cm[0].bitcast(fx.Int32), cm[1].bitcast(fx.Int32))
                        a2, a3 = permlane_swap(32, cm[2].bitcast(fx.Int32), cm[3].bitcast(fx.Int32))
                        r01 = fx.max(a0.bitcast(fx.Float32), a1.bitcast(fx.Float32))
                        r23 = fx.max(a2.bitcast(fx.Float32), a3.bitcast(fx.Float32))
                        b0_, b1_ = permlane_swap(16, r01.bitcast(fx.Int32), r23.bitcast(fx.Int32))
                        m_ = fx.max(b0_.bitcast(fx.Float32), b1_.bitcast(fx.Float32))
                        # -inf for a fully masked column: the running max (from NEG) absorbs it
                        fx.ptr_store(fx.Float32(m_) * qk_c, rbn + (16 * tcol + n16) * 4 + w)
                    if const_expr(phon):
                        tq3 = stamp()
                        phs = [phs[0] + (tq1 - tq0), phs[1] + (tq2 - tq1), phs[2] + (tq3 - tq2), phs[3]]
                    return xn, mn, lw2, phs

                # block t's softmax needs the mask only in the piece's last block: pieces are whole blocks and
                # the two rows of a tile's row pair see keys up to one apart, so every earlier block is whole
                for tp in range(ni - 3):
                    t = tp - 1
                    tq0 = stamp()
                    xc, mg, lw, phs = s_iv(t, xc, mg, lw, phs, tq0)
                    ta, tw, tb = end_iv(ta, tw, tb)
                tq0 = stamp()
                xc, mg, lw, phs = s_iv(nb - 2, xc, mg, lw, phs, tq0, nxt_max=2)
                ta, tw, tb = end_iv(ta, tw, tb)
                tq0 = stamp()
                xc, mg, lw, phs = s_iv(nb - 1, xc, mg, lw, phs, tq0, nxt_max=0)
                ta, tw, tb = end_iv(ta, tw, tb)
                # the last interval (O's P V of the last block) has no S work: block nb's softmax and
                # block nb + 1's QK would be past the piece. An unsplit piece takes the next entry here
                # (so a CTA that starts early takes no second piece before the others have taken their
                # first); a split piece only once its tile's pieces have all arrived (``piece_end``: a
                # CTA waiting there holds no entry it has not started, so the waits cannot deadlock)
                if tid == 0:
                    nv = take()
                ta, tw, tb = end_iv(ta, tw, tb)
                # the wave's row sums over its 16 keys
                for ct in range_constexpr(4):
                    lt = xred(fx.Float32(xred(fx.Float32(lw[ct]), 16, lambda x, y: x + y)), 32, lambda x, y: x + y)
                    if g == 0:
                        fx.ptr_store(fx.Float32(lt), fin + (16 * ct + n16) * 4 + w)
            else:
                if tid == 0:
                    nv = take()
            if tid == 0:
                fx.ptr_store(nv, nxt + npk % 2)
            _bar()
            tpe = _memtime() if const_expr(tlon) else z
            if const_expr(tlon):
                e_pre = e_pre + (tpe - ta)
            ta_, tb_, lst = piece_end(u, rq, rp, ht, slot, pq, kp, nxt + npk % 2)
            if const_expr(tlon):
                t_pe, pe_x, pe_w, pe_m, n_last, te_ = pe_add(slot, tpe, ta_, tb_, lst, t_pe, pe_x, pe_w, pe_m, n_last)
            if const_expr("nops" in probe):
                # timing probe: 8 KB of straight-line code run once a piece (the cost of cold code)
                llvm.inline_asm(res=None, operands_=[], asm_string="s_nop 0\n" * 2048, constraints="", has_side_effects=True)
            # every wave done with the ring and the LDS tables before the next piece
            _bar()
            idx = uniform(fx.Int32(fx.ptr_load(nxt + npk % 2)))
            npk = npk + 1
            if const_expr(tlon):
                tn = _memtime()
                t_epi = t_epi + (tn - ta)
                e_post = e_post + (tn - te_)
    else:
        while idx < npc:
            ta = _memtime() if const_expr(tlon) else z
            u, rq, rp, ht, b0, b1, slot, kvp, L, nb, ni, kend0, kend1, pq, kp = entry(idx)
            te_ = z
            if const_expr(tlon):
                n_bl = n_bl + fx.Int64(nb)
                n_un = n_un + 1
            if nb > 0:
                st, rw = feed_prologue(b0, kvp, L, rq)
                _bar()  # (the S waves' q into ring slot 2)
                _bar()
                if const_expr(tlon):
                    tn = _memtime()
                    t_set = t_set + (tn - ta)
                    ta = tn
                acc = [[fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(2)] for _ in range(4)]  # O^T tiles (d, ct)
                mp = [fx.Float32(NEG)] * 2  # the running max after block b - 1 (column 32 ct + n32)
                one = fx.Float32(1.0)

                def o_iv(t, st, rw, acc, mp, phs, dev=False):
                    """O's interval t: P V of block t - 1 after the rescale to its running max, among
                    the interval's feed (``dev``: block t + 3 is the request's last, read after
                    ``kv_ready`` at device scope)."""
                    tq0 = stamp()
                    rn = kv_rows(b0 + t + 4, kvp, L)
                    offs = kv_offs(rw)
                    if const_expr(dev):
                        kv_ready(rq)  # before block t + 3's loads
                    bp = t - 1
                    kb = ring + (bp % NSLOT) * SLOT_W
                    pops = []
                    for ct in range_constexpr(2):
                        pp = pall + (bp % 2) * (TC * PROW) + (32 * ct + n32) * PROW + 8 * hb
                        pops.append(_cat8(_lds16(pp), _lds16(pp + 4)))
                    vt = [vop(kb, 0), vop(kb, 1)]
                    mn = [fx.Float32(fx.ptr_load(mrun + (bp % 4) * TC + 32 * ct + n32)) for ct in range(2)]
                    al = [hw_exp2(mp[ct] - mn[ct]) for ct in range(2)]
                    if ballot((al[0] != one) | (al[1] != one)) != fx.Int64(0):
                        # one 16-wide multiply a tile: v_pk_mul_f32 pairs (SLP is off). The packed op issues on the
                        # MFMA pipe, which O's 8 and S's 24 MFMAs leave half idle; 8 of them cost the O wave
                        # less than 16 single v_mul_f32
                        als = [fx.Vector.from_elements([al[ct]] * 16, fx.Float32) for ct in range(2)]
                        acc = [[acc[d][ct] * als[ct] for ct in range(2)] for d in range(4)]
                    tq1 = stamp()
                    # the feed among the 8 MFMAs: block t + 2 into its slot (it overwrites the slot's group w,
                    # which this wave's V^T reads cover: in this wave's LDS order, after all four of them) and
                    # block t + 3 into registers, store j then load j, in load order (each store waits for its
                    # own load alone)
                    nst = [None] * NLD
                    stores = {2: (0, 1), 3: (2,), 4: (3, 4), 5: (5,), 6: (6,), 7: (7, 8)}
                    for i in range_constexpr(8):
                        d, ct = i // 2, i % 2
                        if const_expr(ct == 0 and d < 2):
                            vt.append(vop(kb, d + 2))
                        rocdl.sched_barrier(0)
                        if const_expr("nopv" not in probe):
                            acc[d][ct] = _mfma32(vt[d], pops[ct], acc[d][ct])
                        for j in stores.get(i, ()):
                            kv_st(st[j], t + 2, j)
                            nst[j] = kv_ld(offs[j], dev)
                        rocdl.sched_barrier(0)
                    tq2 = stamp()
                    rw = rn
                    if const_expr(phon):
                        tq3 = stamp()
                        phs = [phs[0] + (tq1 - tq0), phs[1] + (tq2 - tq1), phs[2] + (tq3 - tq2), phs[3]]
                    return nst, rw, acc, mn, phs

                # intervals -1 and 0: the feed alone
                st, rw = feed(fx.Int32(-1), st, rw, b0, kvp, L, rq)
                ta, tw, tb = end_iv(ta, tw, tb)
                st, rw = feed(fx.Int32(0), st, rw, b0, kvp, L, rq)
                ta, tw, tb = end_iv(ta, tw, tb)
                # interval td feeds the request's last block (td <= 0: the prologue's or interval -1's or 0's
                # feed did): intervals 1 .. td - 1, then td alone at device scope, then the rest
                td = (L - 1) // BK - b0 - 3
                tda = fx.max(fx.Int32(0), fx.min(td - 1, nb))
                tdb = fx.max(fx.Int32(0), fx.min(td, nb))
                for tp in range(tda):
                    st, rw, acc, mp, phs = o_iv(tp + 1, st, rw, acc, mp, phs)
                    ta, tw, tb = end_iv(ta, tw, tb)
                for tp in range(tda, tdb):
                    st, rw, acc, mp, phs = o_iv(tp + 1, st, rw, acc, mp, phs, dev=True)
                    ta, tw, tb = end_iv(ta, tw, tb)
                for tp in range(tdb, nb):
                    st, rw, acc, mp, phs = o_iv(tp + 1, st, rw, acc, mp, phs)
                    ta, tw, tb = end_iv(ta, tw, tb)
                _bar()
                # the epilogue's lane constants from an opaque lane id (computed here, not hoisted)
                lx = _opaque(lane)
                n32e, hbe = lx % 32, lx // 32
                # l: the S waves' row sums
                lt, inv, lv = [], [], []
                for ct in range_constexpr(2):
                    f4 = _ldsf4(fin + (32 * ct + n32e) * 4)
                    s_ = (fx.Float32(f4[0]) + fx.Float32(f4[1])) + (fx.Float32(f4[2]) + fx.Float32(f4[3]))
                    okl = s_ > fx.Float32(0.0)
                    inv.append(okl.select(fx.Float32(1.0) / s_, fx.Float32(0.0)))
                    lv.append(okl.select((mp[ct] - PRE + fx.Float32(rocdl.log(T.f32, fx.Float32(s_).ir_value()))) * LN2, fx.Float32(NEG)))
                if slot < 0:
                    # unsplit: bf16 o and the LSE
                    for ct in range_constexpr(2):
                        orow = G * rq + RT * rp + ct
                        head = HT * ht + n32e
                        for d in range_constexpr(4):
                            for qd in range_constexpr(4):
                                vals = [fx.Float32(acc[d][ct][4 * qd + e]) * inv[ct] for e in range(4)]
                                dim = 128 * w + 32 * d + 8 * qd + 4 * hbe
                                obf = fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
                                bo.buffer_store(obf, rsrc(o), ((orow * NHA + head) * KV + dim) // 2, cache_modifier=CM_DEV)
                        if (w == 0) & (hbe == 0):
                            bo.buffer_store(lv[ct], rsrc(lse), orow * NHA + head, cache_modifier=CM_DEV)
                else:
                    # a split piece: its fp32 partial (normalized) and LSE into LDS for the piece's end
                    for ct in range_constexpr(2):
                        col = 32 * ct + n32e
                        for d in range_constexpr(4):
                            for qd in range_constexpr(4):
                                vals = [fx.Float32(acc[d][ct][4 * qd + e]) * inv[ct] for e in range(4)]
                                dim = 128 * w + 32 * d + 8 * qd + 4 * hbe
                                _st16(ovl + col * OVW + dim, fx.Vector.from_elements(vals, fx.Float32))
                        if (w == 0) & (hbe == 0):
                            fx.ptr_store(lv[ct], olse + col)
            else:
                _bar()
                lx = _opaque(lane)
                n32e, hbe = lx % 32, lx // 32
                # no visible key: o 0, LSE NEG (no KV read)
                if slot < 0:
                    for ct in range_constexpr(2):
                        orow = G * rq + RT * rp + ct
                        head = HT * ht + n32e
                        for d in range_constexpr(4):
                            for qd in range_constexpr(4):
                                dim = 128 * w + 32 * d + 8 * qd + 4 * hbe
                                bo.buffer_store(fx.Vector.filled(2, 0, fx.Int32), rsrc(o), ((orow * NHA + head) * KV + dim) // 2, cache_modifier=CM_DEV)
                        if (w == 0) & (hbe == 0):
                            bo.buffer_store(fx.Float32(NEG), rsrc(lse), orow * NHA + head, cache_modifier=CM_DEV)
                else:
                    for ct in range_constexpr(2):
                        col = 32 * ct + n32e
                        for d in range_constexpr(4):
                            for qd in range_constexpr(4):
                                dim = 128 * w + 32 * d + 8 * qd + 4 * hbe
                                _st16(ovl + col * OVW + dim, fx.Vector.filled(4, 0.0, fx.Float32))
                        if (w == 0) & (hbe == 0):
                            fx.ptr_store(fx.Float32(NEG), olse + col)
            tpe = _memtime() if const_expr(tlon) else z
            if const_expr(tlon):
                e_pre = e_pre + (tpe - ta)
            ta_, tb_, lst = piece_end(u, rq, rp, ht, slot, pq, kp, nxt + npk % 2)
            if const_expr(tlon):
                t_pe, pe_x, pe_w, pe_m, n_last, te_ = pe_add(slot, tpe, ta_, tb_, lst, t_pe, pe_x, pe_w, pe_m, n_last)
            # every wave done with the ring and the LDS tables before the next piece
            _bar()
            idx = uniform(fx.Int32(fx.ptr_load(nxt + npk % 2)))
            npk = npk + 1
            if const_expr(tlon):
                tn = _memtime()
                t_epi = t_epi + (tn - ta)
                e_post = e_post + (tn - te_)
    # the W_UV stage: unit (head h, latent half c) merges the head's pieces for every row (fp32, in piece
    # order) and multiplies the bf16 o by W_UV[h]'s half (fp32, to the halves' scratch); the pair's last unit adds
    # the halves and emits. Units come after every piece, so a unit waits only on pieces in flight.
    _bar()
    if const_expr(uton):
        if tid == 0:
            bo.buffer_store(fx.Int32(_realtime()), rsrc(tl), TL_UB + bid * 16 + 1)
            bo.buffer_store(fx.Int32(rk0), rsrc(tl), TL_UB + bid * 16)
    ua = cnt + fx.Int64(4 * (CNT_U + layer))
    nrq = uniform(_ld(wq, 2))
    srows = G * nrq
    ttl = ring + 32 * OBW + 4 * 32 * RDW  # the tile table in LDS (past the unit's o and k-half products), a word a tile
    wvl = ttl + 512  # the unit's W_UV half: [256 latent][128 out] bf16, WVW words a row
    for k0 in range_constexpr(0, MAX_TILES, THREADS):
        if k0 + tid < TILES * nrq:
            fx.ptr_store(_ld(wq, WQ_T + k0 + tid), ttl + k0 + tid)
    if tid == 0:
        fx.ptr_store(_atomic_add(ua, 1), nxt + 2)
    _bar()
    ui = uniform(fx.Int32(fx.ptr_load(nxt + 2)))
    nuk = fx.Int32(1)
    r_o, r_half = rsrc(o, srows * fx.Int32(NHA * KV * 2)), rsrc(half)  # o: a split row's load fetches nothing
    pc = cnt + fx.Int64(4 * (CNT_H + layer * MAX_TILES))  # the head's W_UV pair: its units done
    # the unit's bf16 o, [32 rows][256 dims] (OBW words a row), and the second k half's products,
    # [n tile][32 rows][RDW]: rows padded so a b128 read of 16 lanes (16 rows) is bank-conflict free
    obl = ring
    rdl = ring + 32 * OBW
    lu = _opaque(lane)
    nt, kh = wave % 4, wave // 4  # GEMV: wave's 32 output dims and half of the unit's 256 latent dims
    q16, s16 = lu // 16, lu % 16

    def wop(st):
        """The GEMV's B operand of step st (lane: out dim 32 nt + lane % 32, latent 128 kh + 16 st + 8 (lane / 32)
        .., 8 of them) from the [latent][out] tile: two ds_read_b64_tr_b16. In a 16-lane group, source lane s
        reads latent row 4 r + s / 4 at out dims 4 (s % 4) ..; lane i gets latent 4 r + e of out dim i as its
        element e."""
        vs = []
        for r in range_constexpr(2):
            row = 128 * kh + 16 * st + 8 * (q16 // 2) + 4 * r + s16 // 4
            vs.append(_tr16(wvl + row * WVW + 16 * nt + 8 * (q16 % 2) + 2 * (s16 % 4)))
        return fx.Vector.from_elements([vs[0][0], vs[0][1], vs[1][0], vs[1][1]], fx.Int32)

    nut = fx.Int32(0)
    m0, m1, m2, m3, m4, m5, m6, ur0, ur1, ur6, ph_a, ph_b = z, z, z, z, z, z, z, z, z, z, z, z  # (timeline)
    while ui < (0 if const_expr("nounit" in probe) else NU):
        ph_a, ph_b = z, z
        if const_expr(uton):
            rocdl.sched_barrier(0)
            m0 = _memtime()
            ur0 = _realtime()
            rocdl.sched_barrier(0)
        h = ui // 2
        c = ui % 2
        gh = h // HT
        # W_UV[h]'s latent half c (bf16 [512 latent][128 out]: one contiguous 64 KB) into LDS before the wait
        for st in range_constexpr(8 if "zw" not in probe else 0):
            k = st * THREADS + tid  # 16-byte chunk k: latent row k / 16, out dims 8 (k % 16) ..
            _st16(wvl + (k // 16) * WVW + 4 * (k % 16), _ld4(wuv, ((h * KV + 256 * c) * WO + 8 * k) // 2))
        if const_expr("nowait" not in probe):
            if tid == 0:
                fw = CNT_F + 48 * layer + 16 * gh + ui % 16
                v = fx.Int32(bo.buffer_load(rsrc(cnt), fw, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
                while v == 0:
                    rocdl.s_sleep(4)
                    v = fx.Int32(bo.buffer_load(rsrc(cnt), fw, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
        _bar()
        if const_expr(uton):
            rocdl.sched_barrier(0)
            m1 = _memtime()
            ur1 = _realtime()
            rocdl.sched_barrier(0)
        d4 = 256 * c + 4 * lu  # the lane's 4 latent dims of the half
        for rb in range((srows + 31) // 32):
            # merge: wave w merges rows 32 rb + w + 8 i (i < 4), a lane 4 dims, so a (row, piece) is one 1 KB line
            # a load. An unsplit tile's bf16 o as it is; a split tile's pieces in piece order: piece p's LSE in
            # lane p, the max and the weights' sum over the wave, the weighted partials. Every row's o, LSEs and
            # first 8 pieces in one batch of loads, then a row's pieces 8 .. in one more. Rows past the step's issue
            # no loads and no merge (an out-of-range load fetches nothing but costs its issue)
            rows = []
            for i in range_constexpr(4):
                rl = wave + 8 * i
                r = 32 * rb + rl
                live = r < srows
                rr = fx.min(r, srows - 1)
                rq_ = rr // G
                jr = rr - G * rq_
                col = (jr % 2) * 32 + h % 32
                tv = uniform(fx.Int32(fx.ptr_load(ttl + TILES * rq_ + 3 * (jr // 2) + gh)))
                kq = (live & (tv >= 0)).select(tv >> 16, fx.Int32(0)) * fx.Int32(0 if "nomerge" in probe else 1)
                rows.append((rl, rr, live, tv, kq, tv & 65535, col))

            def part_ld(kq, s0, col, p):
                base = (p < kq).select(((s0 + p) * TC + col) * KV + d4, fx.Int32(PART_E))
                return fx.Vector(bo.buffer_load(r_part, base, vec_width=4, dtype=T.f32, cache_modifier=CM_DEV))

            lds_ = []
            for i in range_constexpr(4):
                rl, rr, live, tv, kq, s0, col = rows[i]
                ub, lp = fx.Vector.filled(2, 0, fx.Int32), fx.Float32(0.0)
                vv = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(8)]
                if live:
                    ub = fx.Vector(bo.buffer_load(r_o, (tv < 0).select(((rr * NHA + h) * KV + d4) // 2, srows * fx.Int32(NHA * KV // 2)),
                                                  vec_width=2, dtype=T.i32, cache_modifier=CM_DEV))  # fmt: skip
                    lp = fx.Float32(bo.buffer_load(r_plse, (lu < kq).select((s0 + lu) * TC + col, fx.Int32(MAX_SLOTS * TC)),
                                                   vec_width=1, dtype=T.f32, cache_modifier=CM_DEV))  # fmt: skip
                    vv = [part_ld(kq, s0, col, fx.Int32(p8)) for p8 in range(8)]
                lds_.append((ub, lp, vv))
            for i in range_constexpr(4):
                rl, rr, live, tv, kq, s0, col = rows[i]
                ub, lp, vv = lds_[i]
                if live:
                    if const_expr(uton):
                        rocdl.sched_barrier(0)
                        tA = _memtime()
                        rocdl.sched_barrier(0)
                    okl = lu < kq
                    lp = okl.select(lp, fx.Float32(NEG))
                    mx = fx.Float32(butterfly(lp, (32, 16, 8, 4, 2, 1), fx.max))
                    wl = okl.select(hw_exp2((lp - mx) * fx.Float32(LOG2E)), fx.Float32(0.0))
                    wsum = fx.Float32(butterfly(wl, (32, 16, 8, 4, 2, 1)))
                    if const_expr(uton):
                        rocdl.sched_barrier(0)
                        tB = _memtime()
                        ph_a = ph_a + (tB - tA)
                        rocdl.sched_barrier(0)
                    acc = [fx.Float32(0.0)] * 4
                    for p8 in range_constexpr(8):
                        wg = read_lane_i32(wl.bitcast(fx.Int32), fx.Int32(p8)).bitcast(fx.Float32)
                        acc = [fmath.fma(wg, fx.Float32(vv[p8][e]), acc[e]) for e in range(4)]
                    for pb in range(8, kq, MAX_KP - 8):  # at most once
                        v2 = [part_ld(kq, s0, col, pb + p) for p in range(MAX_KP - 8)]
                        for p in range_constexpr(MAX_KP - 8):
                            wg = read_lane_i32(wl.bitcast(fx.Int32), pb + p).bitcast(fx.Float32)
                            acc = [fmath.fma(wg, fx.Float32(v2[p][e]), acc[e]) for e in range(4)]
                    inv = (wsum > fx.Float32(0.0)).select(fx.Float32(1.0) / wsum, fx.Float32(0.0))
                    mb = fx.Vector.from_elements([acc[e] * inv for e in range(4)], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
                    o2 = [live.select((tv < 0).select(fx.Int32(ub[e]), fx.Int32(mb[e])), fx.Int32(0)) for e in range(2)]
                    if live & (tv >= 0) & (c == 0) & (lu == 0):
                        bo.buffer_store(mx + fx.Float32(rocdl.log(T.f32, fx.Float32(wsum).ir_value())) * LN2, rsrc(lse), rr * NHA + h,
                                        cache_modifier=CM_DEV)  # fmt: skip
                    fx.ptr_store(o2[0], obl + rl * OBW + 2 * lu)
                    fx.ptr_store(o2[1], obl + rl * OBW + 2 * lu + 1)
                    if const_expr(uton):
                        rocdl.sched_barrier(0)
                        ph_b = ph_b + (_memtime() - tB)
                        rocdl.sched_barrier(0)
            if const_expr(uton):
                rocdl.sched_barrier(0)
                m2 = _memtime()
                rocdl.sched_barrier(0)
            _bar()
            if const_expr(uton):
                rocdl.sched_barrier(0)
                m3 = _memtime()
                rocdl.sched_barrier(0)
            # GEMV: out[32 rows, 32 nt ..] over the wave's 128 dims: 8 steps of 32x32x16 bf16, four accumulators
            # in turn (no MFMA reads its accumulator right after writing it)
            ga = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(4)]
            for st in range_constexpr(8):
                av = _lds16(obl + (lu % 32) * OBW + 64 * kh + 8 * st + 4 * (lu // 32))
                ga[st % 4] = fx.Vector(rocdl.mfma_f32_32x32x16_bf16(
                    T.vec(16, T.f32), [av.bitcast(fx.BFloat16), wop(st).bitcast(fx.BFloat16), ga[st % 4]]))  # fmt: skip
            gs = [(fx.Float32(ga[0][e]) + fx.Float32(ga[1][e])) + (fx.Float32(ga[2][e]) + fx.Float32(ga[3][e])) for e in range(16)]
            if const_expr(uton):
                rocdl.sched_barrier(0)
                m4 = _memtime()
                rocdl.sched_barrier(0)
            if kh == 1:
                for e4 in range_constexpr(4):
                    _st16(rdl + nt * 32 * RDW + (lu % 32) * RDW + 4 * e4 + 16 * (lu // 32),
                          fx.Vector.from_elements(gs[4 * e4 : 4 * e4 + 4], fx.Float32))  # fmt: skip
            _bar()
            if kh == 0:
                # the half's [32 rows][128 out] fp32 over the unit's o (OBW words a row)
                for e4 in range_constexpr(4):
                    o4 = _ldsf4(rdl + nt * 32 * RDW + (lu % 32) * RDW + 4 * e4 + 16 * (lu // 32))
                    for e in range_constexpr(4):
                        # accumulator element 4 e4 + e: row 8 e4 + 4 (lane / 32) + e, column lane % 32
                        rl_ = 8 * e4 + 4 * (lu // 32) + e
                        fx.ptr_store((gs[4 * e4 + e] + fx.Float32(o4[e])).bitcast(fx.Int32), obl + rl_ * OBW + 32 * nt + lu % 32)
            if const_expr(uton):
                rocdl.sched_barrier(0)
                m5 = _memtime()
                rocdl.sched_barrier(0)
            _bar()
            # thread (row tid / 16, out dims 8 (tid % 16) ..): the half to the halves' scratch
            orow = 32 * rb + tid // 16
            if orow < srows:
                for j in range_constexpr(2):
                    bo.buffer_store(_ldsf4(obl + (tid // 16) * OBW + 8 * (tid % 16) + 4 * j), r_half,
                                    ((orow * NHA + h) * 2 + c) * WO + 8 * (tid % 16) + 4 * j, cache_modifier=CM_DEV)  # fmt: skip
            _bar()
        # the two latent halves: the pair's last unit (of (h, 0) and (h, 1)) adds them, half 0 + half 1, and emits
        # each row's out dims 8 at a time and its LSE
        rocdl.s_waitcnt(vmcnt=0)
        _bar()
        if tid == 0:
            fx.ptr_store(_atomic_add(pc + fx.Int64(4) * fx.Int64(h), 1), nxt + 4)
        _bar()
        if uniform(fx.Int32(fx.ptr_load(nxt + 4))) == 1:
            for rb in range((srows + 31) // 32):
                orow = 32 * rb + tid // 16
                if orow < srows:
                    cg = tid % 16
                    hv = [[fx.Vector(bo.buffer_load(r_half, ((orow * NHA + h) * 2 + hf) * WO + 8 * cg + 4 * j, vec_width=4,
                                                    dtype=T.f32, cache_modifier=CM_DEV)) for j in range(2)] for hf in range(2)]  # fmt: skip
                    emit_values(h // HO, h % HO, orow, 8 * cg,
                                [fx.Float32(hv[0][e // 4][e % 4]) + fx.Float32(hv[1][e // 4][e % 4]) for e in range(8)])  # fmt: skip
                    if cg == 0:
                        emit_lse(h // HO, h % HO, orow, fx.Float32(bo.buffer_load(rsrc(lse), orow * NHA + h, vec_width=1, dtype=T.f32,
                                                                                cache_modifier=CM_DEV)))  # fmt: skip
        if tid == 0:
            fx.ptr_store(_atomic_add(ua, 1), nxt + 2 + nuk % 2)
        _bar()
        if const_expr(uton):
            rocdl.sched_barrier(0)
            m6 = _memtime()
            ur6 = _realtime()
            rocdl.sched_barrier(0)
            ub0 = TL_UU + ui * 32
            if lane == 0:
                bo.buffer_store(fx.Int32(ph_a), rsrc(tl), ub0 + 16 + 2 * wave)
                bo.buffer_store(fx.Int32(ph_b), rsrc(tl), ub0 + 17 + 2 * wave)
                bo.buffer_store(fx.Int32(m2 - m1), rsrc(tl), ub0 + 8 + wave)
            if tid == 0:
                uv = [bid, ur0, ur1, ur6, m1 - m0, m3 - m1, m4 - m3, m5 - m4]
                for f in range_constexpr(8):
                    bo.buffer_store(fx.Int32(uv[f]), rsrc(tl), ub0 + f)
            nut = nut + 1
        ui = uniform(fx.Int32(fx.ptr_load(nxt + 2 + nuk % 2)))
        nuk = nuk + 1
    if const_expr(uton):
        if tid == 0:
            bo.buffer_store(fx.Int32(_realtime()), rsrc(tl), TL_UB + bid * 16 + 2)
            bo.buffer_store(nut, rsrc(tl), TL_UB + bid * 16 + 3)
    if const_expr(tlon):
        if lane == 0:
            tm1 = _memtime()
            vals = [t_set, tw, tb, t_epi, tm1 - tk0, _realtime() - rk0, n_bl, n_un] + phs + [t_pe, pe_x, rk0, pe_w, pe_m, n_last, e_pre, e_post]
            vals = vals + [t_rdy, r_rdy]
            if w < 2:
                for f in range_constexpr(len(vals)):
                    bo.buffer_store(fx.Int32(vals[f]), rsrc(tl), bid * TL_W + 32 * (2 * role + w) + f)


def default_hooks(q8, wout, lse):
    """The standalone kernel's hooks: q from q8 (e4m3 [S, 96, 576]); no readiness waits; out to wout (fp32
    [S, 96, 2, 128]: the sum in half 0, half 1 left as it is) and the LSE to lse (fp32 [S, 96])."""

    def q_load(rq, head, t, byte):
        return _ld4(q8, (((G * rq + t) * NHA + head) * DQK + byte) // 4)

    def ready(rq, h0, nh):
        pass

    def kv_ready(rq):
        pass

    def emit_values(owner, hl, t, col, vals):
        for j in range_constexpr(2):
            bo.buffer_store(fx.Vector.from_elements(vals[4 * j : 4 * j + 4], fx.Float32), rsrc(wout),
                            ((t * NHA + HO * owner + hl) * 2) * WO + col + 4 * j)  # fmt: skip

    def emit_lse(owner, hl, t, v):
        bo.buffer_store(v, rsrc(lse), t * NHA + HO * owner + hl)

    return q_load, ready, kv_ready, emit_values, emit_lse


def build(key: MlaDecodeBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name="k3_mono_mla_decode_c64", known_block_size=[THREADS, 1, 1])
    def mla_decode_c64(
        q8: Int64, kv: Int64, kv_idx: Int64, kv_ptr: Int64, g_ptr: Int64, rank: Int32,
        o: Int64, lse: Int64, part: Int64, plse: Int64, wq: Int64, cnt: Int64, layer: Int32, tl: Int64,
        wuv: Int64, half: Int64, wout: Int64,
    ):  # fmt: skip
        _ = keyed
        a = dict(kv=kv, kv_idx=kv_idx, kv_ptr=kv_ptr, g_ptr=g_ptr, rank=rank, o=o, lse=lse, part=part, plse=plse, wq=wq,
                 cnt=cnt, layer=layer, wuv=wuv, half=half, tl=tl)  # fmt: skip
        decode_stage(key, fx.SharedAllocator().allocate(Smem).peek(), fx.block_idx.x, fx.thread_idx.x, a,
                     *default_hooks(q8, wout, lse))  # fmt: skip

    @flyc.jit
    def launch(
        q8: Int64, kv: Int64, kv_idx: Int64, kv_ptr: Int64, g_ptr: Int64, rank: Int32,
        o: Int64, lse: Int64, part: Int64, plse: Int64, wq: Int64, cnt: Int64, layer: Int32, tl: Int64,
        wuv: Int64, half: Int64, wout: Int64, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        mla_decode_c64(q8, kv, kv_idx, kv_ptr, g_ptr, rank, o, lse, part, plse, wq, cnt, layer, tl, wuv, half, wout).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    # two waves a SIMD share 512 registers: no AGPRs, so a wave's q (S) or O and its loads in
    # flight (MFMA accumulators in VGPRs) fit its 256 VGPRs
    # SLP off (its cost threshold out of reach): packed fp32 (v_pk_*) issues on the MFMA pipe and serializes with the
    # MFMAs (lowreq probe), so the softmax, row sums and O rescale stay one fp32 op a value
    _agpr_hint_lowering()
    launch = flyc.compile[{"agpr_alloc": 0, "llvm_options": {"slp-threshold": 100000}}](launch)
    _BUILDS[key] = launch
    return launch


def _layout(nb, m):
    """A request's split for most m blocks a piece: (pieces a tile k, blocks o, r): pieces q < r have
    o + 1 blocks, the others o."""
    k = (nb > m).select((nb + m - 1) // m, fx.Int32(1))
    o = nb // k
    return k, o, nb - k * o


def build_plan():
    if "plan" in _BUILDS:
        return _BUILDS["plan"]
    keyed = key_tuple({"plan": 1}, SOURCES)

    @fx.struct
    class PlanSmem:
        nb: fx.Array[fx.Int32, 32, 16]  # blocks a tile, a request
        ck: fx.Array[fx.Int32, 4 * 32, 16]  # a request's (pieces a tile, blocks, longer pieces, first slot)
        cls: fx.Array[fx.Int32, 4 * NCLS, 16]  # a piece class's (cost, count, first entry, -)
        cand: fx.Array[fx.Int32, 2 * 16, 16]  # a candidate's (estimate, run)

    @flyc.kernel(name="k3_mono_mla_decode_c64_plan", known_block_size=[PLAN_THREADS, 1, 1])
    def mla_decode_c64_plan(kv_ptr: Int64, nreq: Int32, wq: Int64, cnt: Int64):
        """The step's work list: the run that predicts the least time (a request's tiles split in
        equal runs of at most that many blocks), its pieces longest first; and every layer's queue and
        tile counters zeroed."""
        _ = keyed
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(PlanSmem).peek()
        nbl, ck, cls = lds.nb.ptr, lds.ck.ptr, lds.cls.ptr
        for k_ in range_constexpr(-(-CNT_WORDS // PLAN_THREADS)):
            w_ = tid + PLAN_THREADS * k_
            if w_ < CNT_WORDS:
                bo.buffer_store(fx.Int32(0), rsrc(cnt), w_)
        if tid < 32:
            r = fx.min(tid, nreq - 1)
            L = _ld(kv_ptr, r + 1) - _ld(kv_ptr, r)
            fx.ptr_store((tid < nreq).select((L + BK - 1) // BK, fx.Int32(0)), nbl + tid)
        _bar()
        nbs = [fx.Int32(fx.ptr_load(nbl + r)) for r in range(MAX_REQS)]
        live = [fx.Int32(r) < nreq for r in range(MAX_REQS)]
        # group c (32 lanes, lane r: request r) estimates candidate c: max(longest piece, the work a
        # CTA) + the merges a CTA, or 2^30 past the list's or the partials' room
        cg, rl = tid // 32, tid % 32
        nbr = fx.Int32(fx.ptr_load(nbl + rl))
        lr = rl < nreq

        def s32(v):
            for off in (16, 8, 4, 2, 1):
                v = xred(fx.Int32(v), off, lambda x, y: x + y)
            return v

        def m32(v):
            for off in (16, 8, 4, 2, 1):
                v = xred(fx.Int32(v), off, fx.max)
            return v

        tot = s32(lr.select(TILES * (nbr + C_P), fx.Int32(0)))
        fn, fd = fx.Int32(0), fx.Int32(1)
        for ci in range_constexpr(len(PLAN_F)):
            fn = (cg == ci).select(fx.Int32(PLAN_F[ci][0]), fn)
            fd = (cg == ci).select(fx.Int32(PLAN_F[ci][1]), fd)
        m = (fn == 0).select(fx.Int32(1 << 20), fx.max(fx.Int32(4), (tot * fn + BLOCKS * fd - 1) // (BLOCKS * fd)))
        k, base, ext = _layout(nbr, m)
        sp = lr & (k > 1)
        cw = sp.select(fx.Int32(C_W), fx.Int32(0))
        work = s32(lr.select(TILES * (nbr + k * (C_P + cw)), fx.Int32(0)))
        pmax = m32(lr.select(base + (ext > 0).select(fx.Int32(1), fx.Int32(0)) + C_P + cw, fx.Int32(0)))
        npc = s32(lr.select(TILES * k, fx.Int32(0)))
        nsl = s32(sp.select(TILES * k, fx.Int32(0)))
        # the longest piece times the pieces a CTA (whole pieces), the work a CTA, plus the merges' traffic
        est = fx.max(pmax * ((npc + BLOCKS - 1) // BLOCKS), (work + BLOCKS - 1) // BLOCKS) + (nsl + C_G - 1) // C_G
        ok = (nsl <= MAX_SLOTS) & (npc <= MAX_ENTRIES) & (m32(lr.select(k, fx.Int32(0))) <= MAX_KP)
        if (rl == 0) & (cg < len(PLAN_F)):
            fx.ptr_store(ok.select(est, fx.Int32(1 << 30)), lds.cand.ptr + 2 * cg)
            fx.ptr_store(m, lds.cand.ptr + 2 * cg + 1)
        _bar()
        best, bm = fx.Int32(1 << 30), fx.Int32(1 << 20)
        for ci in range_constexpr(len(PLAN_F)):
            e_ = fx.Int32(fx.ptr_load(lds.cand.ptr + 2 * ci))
            better = e_ < best
            best = better.select(e_, best)
            bm = better.select(fx.Int32(fx.ptr_load(lds.cand.ptr + 2 * ci + 1)), bm)
        # the run's layout: a request's (pieces a tile, o, r, first slot)
        if tid < MAX_REQS:
            k, base, ext = _layout(fx.Int32(fx.ptr_load(nbl + tid)), bm)
            sb = fx.Int32(0)
            for r in range_constexpr(MAX_REQS):
                k2, _, _ = _layout(nbs[r], bm)
                sb = ((fx.Int32(r) < tid) & live[r] & (k2 > 1)).select(sb + TILES * k2, sb)
            _st16(ck + 4 * tid, fx.Vector.from_elements([k, base, ext, sb], fx.Int32))
            for tt in range_constexpr(TILES):
                bo.buffer_store((k > 1).select(sb + tt * k + 65536 * k, fx.Int32(-1)), rsrc(wq), WQ_T + TILES * tid + tt)
        _bar()
        # piece class c: request c's pieces, tile-major (a tile's pieces on consecutive entries: they
        # wait for each other at the tile's merge); the class's cost (its longest piece) and count
        if tid < NCLS:
            v = _lds16(ck + 4 * tid)
            k, base, ext = fx.Int32(v[0]), fx.Int32(v[1]), fx.Int32(v[2])
            cw = (k > 1).select(fx.Int32(C_W), fx.Int32(0))
            cost = base + (ext > 0).select(fx.Int32(1), fx.Int32(0)) + C_P + cw
            fx.ptr_store(cost, cls + 4 * tid)
            fx.ptr_store((tid < nreq).select(TILES * k, fx.Int32(0)), cls + 4 * tid + 1)
        _bar()
        # a class's first entry: the classes before it in (cost down, request up) order
        if tid < NCLS:
            cost = fx.Int32(fx.ptr_load(cls + 4 * tid))
            st = fx.Int32(0)
            for c2 in range_constexpr(NCLS):
                v = _lds16(cls + 4 * c2)
                before = (fx.Int32(v[0]) > cost) | ((fx.Int32(v[0]) == cost) & (fx.Int32(c2) < tid))
                st = before.select(st + fx.Int32(v[1]), st)
            fx.ptr_store(st, cls + 4 * tid + 2)
        _bar()
        npc = fx.Int32(0)
        for r in range_constexpr(MAX_REQS):
            v = _lds16(ck + 4 * r)
            npc = npc + live[r].select(TILES * fx.Int32(v[0]), fx.Int32(0))
        if tid == 0:
            bo.buffer_store(npc, rsrc(wq), 0)
            bo.buffer_store(npc, rsrc(wq), 1)
            bo.buffer_store(nreq, rsrc(wq), 2)
        # entry e: its class, then (tile + 4096 x its piece index, b0, b1, -1 unsplit or its partial slot +
        # 4096 x the tile's pieces)
        for j in range((npc + PLAN_THREADS - 1 - tid) // PLAN_THREADS):
            e = tid + PLAN_THREADS * j
            cr, ci, cst = fx.Int32(0), fx.Int32(0), fx.Int32(0)
            for c2 in range_constexpr(NCLS):
                v = _lds16(cls + 4 * c2)
                hit = (fx.Int32(v[2]) <= e) & (e < fx.Int32(v[2]) + fx.Int32(v[1]))
                cr = hit.select(fx.Int32(c2), cr)
                ci = hit.select(e - fx.Int32(v[2]), ci)
                cst = hit.select(fx.Int32(v[2]), cst)
            r = cr
            v = _lds16(ck + 4 * r)
            k, base, ext, sb = fx.Int32(v[0]), fx.Int32(v[1]), fx.Int32(v[2]), fx.Int32(v[3])
            tt = ci // k
            q = ci - tt * k
            b0 = q * base + fx.min(q, ext)
            b1 = b0 + base + (q < ext).select(fx.Int32(1), fx.Int32(0))
            f3 = (k > 1).select(sb + tt * k + q + 4096 * k, fx.Int32(-1))
            bo.buffer_store(fx.Vector.from_elements([TILES * r + tt + 4096 * q, b0, b1, f3], fx.Int32), rsrc(wq), WQ_HDR + 4 * e)
            # its key range's group (the classes before it hold cst / 9 groups) and its place in that XCD's list
            gi = cst // TILES + q
            bo.buffer_store(e, rsrc(wq), WQ_X + (gi % 8) * XLEN + TILES * (gi // 8) + tt)

    @flyc.jit
    def launch(kv_ptr: Int64, nreq: Int32, wq: Int64, cnt: Int64, stream: fx.Stream = _STREAM):
        _ = keyed
        mla_decode_c64_plan(kv_ptr, nreq, wq, cnt).launch(grid=(1,), block=(PLAN_THREADS,), stream=stream)

    _BUILDS["plan"] = launch
    return launch


class DecodeBuffers:
    """The decode's work list, its queue and tile counters, and the split pieces' fp32 partials."""

    def __init__(self, device):
        import torch

        self.wq = torch.zeros(WQ_WORDS, dtype=torch.int32, device=device)
        self.cnt = torch.zeros(CNT_WORDS, dtype=torch.int32, device=device)
        self.part = torch.zeros(MAX_SLOTS * TC * KV, dtype=torch.float32, device=device)
        self.plse = torch.zeros(MAX_SLOTS * TC, dtype=torch.float32, device=device)
        # the W_UV stage's weights (bf16 [96 heads][512 latent][128 out]; zeros unless a caller passes its own),
        # the two latent halves' scratch and the standalone output (fp32 [S, 96, 2 latent halves, 128])
        self.wuv = torch.zeros(NHA * KV * WO, dtype=torch.bfloat16, device=device)
        self.half = torch.zeros(MAX_REQS * G * NHA * 2 * WO, dtype=torch.float32, device=device)
        self.wout = torch.zeros(MAX_REQS * G * NHA * 2 * WO, dtype=torch.float32, device=device)


def _region_words(n: int) -> int:
    """``n`` 4-byte words rounded up to a 256-byte multiple."""
    return -(-n // 64) * 64


class StageBuffers:
    """``decode_stage``'s buffers in a caller's launch of up to ``rows`` rows:
    the work list and counters its own (zeroed once), and the pieces' partials
    and LSEs, the two latent halves and the unsplit tiles' bf16 o and LSE as
    views into ``mla_decode.partials``, every one written in a launch before
    that launch reads it."""

    def __init__(self, device, rows: int):
        import torch

        self.wq = torch.zeros(WQ_WORDS, dtype=torch.int32, device=device)
        self.cnt = torch.zeros(CNT_WORDS, dtype=torch.int32, device=device)
        sizes = {
            "part": PART_E, "plse": MAX_SLOTS * TC, "half": MAX_REQS * G * NHA * 2 * WO,
            "o": rows * NHA * KV // 2, "lse": rows * NHA,
        }  # fmt: skip
        region, off = md.partials(device), 0
        assert off + sum(_region_words(n) for n in sizes.values()) <= region.numel()
        for name, n in sizes.items():
            setattr(self, name, region[off : off + n])
            off += _region_words(n)
        self.o = self.o.view(torch.bfloat16)


def mla_decode_plan_launch(*, kv_ptr, g_ptr, nreq, bufs):
    """The step's work list (``build_plan``): once a step, before its layers' decodes."""
    import torch

    assert 1 <= nreq <= MAX_REQS and kv_ptr.dtype == torch.int32
    build_plan()(kv_ptr.data_ptr(), nreq, bufs.wq.data_ptr(), bufs.cnt.data_ptr(), stream=torch.cuda.current_stream())


def mla_decode_launch(key: MlaDecodeBuild, *, q8, kv, kv_idx, kv_ptr, g_ptr, rank, o, lse, bufs, layer=0, tl=None,
                      wuv=None, wout=None):
    """Host launcher, as ``mla_decode_b.mla_decode_launch``: ``q8`` e4m3 [S, 96, 576];
    ``kv`` the rank's pool rows [., 576] e4m3; ``kv_idx`` / ``kv_ptr`` / ``g_ptr``
    int32; ``o`` bf16 [S, 96, 512] and ``lse`` fp32 [S, 96]; ``bufs``: ``DecodeBuffers``; ``layer``:
    the MLA layer's index (its own queue and tile counters, all zeroed by the step's plan); ``wuv`` bf16
    [96, 512 latent, 128 out]; ``wout`` fp32 [S, 96, 2, 128]: the W_UV output in half 0 (half 1 untouched)."""
    import torch

    s = q8.shape[0]
    nreq = s // G
    assert s % G == 0 and 1 <= nreq <= MAX_REQS, s
    assert q8.element_size() == 1 and q8.is_contiguous() and q8.numel() == s * NHA * DQK
    assert kv.element_size() == 1 and kv.stride(-2) == DQK
    for t in (kv_idx, kv_ptr, g_ptr):
        assert t.dtype == torch.int32 and t.is_contiguous()
    assert o.dtype == torch.bfloat16 and o.is_contiguous() and o.numel() == s * NHA * KV
    assert lse.dtype == torch.float32 and lse.is_contiguous() and lse.numel() == s * NHA
    assert 0 <= layer < MAX_LAYERS
    for t, n, dt in ((wuv, NHA * KV * WO, torch.bfloat16), (wout, s * NHA * 2 * WO, torch.float32)):
        assert t is None or (t.dtype == dt and t.is_contiguous() and t.numel() >= n)
    build(key)(
        q8.data_ptr(), kv.data_ptr(), kv_idx.data_ptr(), kv_ptr.data_ptr(), g_ptr.data_ptr(), rank,
        o.data_ptr(), lse.data_ptr(), bufs.part.data_ptr(), bufs.plse.data_ptr(), bufs.wq.data_ptr(),
        bufs.cnt.data_ptr(), layer, 0 if tl is None else tl.data_ptr(),
        (bufs.wuv if wuv is None else wuv).data_ptr(), bufs.half.data_ptr(), (bufs.wout if wout is None else wout).data_ptr(),
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
