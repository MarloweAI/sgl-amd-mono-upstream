# SPDX-License-Identifier: Apache-2.0
"""``mla_front``: one Kimi-K3 MLA layer's attention under decode context
parallelism (DCP8), from the AttnRes seam to the V-projected head partials in
the DCP all-to-all's send layout, for one request's verify block (6 rows). One
launch a layer, ``BLOCKS`` x ``THREADS``.

Every rank attends all 96 heads over its own share of the request's keys (rank
``pos % 8`` holds position ``pos``), with q computed for every head from the
replicated q_b and W_UK, as SGLang's DCP MLA forward does. The launch ends where
SGLang's all-to-all starts: the 96 heads' partials, each V-projected and with
its natural-log LSE, at ``send[owner][row][local head]`` (``[8, S, 12, 130]``
bf16; the LSE is fp32 word 64 of a record). It also writes the output gate's
g_proj and this rank's new KV record.

    CTAs 0 .. 55     the attention-side seam (K1's): slice -> gate -> mix,
                     input_layernorm -> the bf16 x rows (XRDY a slice)
    stage 1, CTAs 56 .. (and 28 groups on seam CTAs)
                     fused_qkv_a (132 groups of 16 rows) and g_proj (96): the
                     raw latent to LAT, a q / kv group's rows' sum of squares to
                     SS, g_proj to ``gate``
    KV CTA           kv_a_layernorm, e4m3 of the latent and the rope part: the
                     new token this rank owns -> its KV row and NEWKV
    stage 2, every CTA
                     q_b over all 96 heads (1152 groups of 16 rows) from the
                     normed q latent: bf16 to QB; the rope rows as e4m3 into Q8
    stage 3, every CTA
                     the absorb q_nope @ W_UK, a (head, 128 latent columns) a
                     task, e4m3 into Q8 [S, 96, 576]
    attention, CTAs 128 ..
                     an item a (16-head group, key split, dim range): e4m3 QK
                     and bf16 PV (V widened exactly) over 32-key blocks, the
                     online softmax, the new token from NEWKV; a partial (f16,
                     over its own sum) and its (max, sum) to PART / ML
    W_UV, CTAs 0 .. 127
                     a (head, 64 value columns) a task: the head's split
                     partials merged in split order to the bf16 latent output,
                     then W_UV -> ``send``, and the head's LSE

Numerics follow SGLang's hand-offs: bf16 x, latent, q_b and absorb outputs, q
and the KV record in e4m3 at scale 1 (the round-to-nearest-even, non-saturating
cast torch uses), the bf16 attention output before W_UV, bf16 partials. The key
split count is a function of the request's own context, so a row's bits do not
depend on the batch. The GEMV split-K and the attention's summation orders are
this kernel's own.
"""

import math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    fp8_to_bf16x8,
    bf16_round,
    global_load,
    hw_exp2,
    hw_rsq,
    row_sum,
    rsrc,
    uniform,
    xred,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    kernel_sources,
    key_tuple,
    pair_layout,
)
from sgl_amd_mono.common.sync import Mailbox, sreg
from sgl_amd_mono.kimi_k3.gfx950.attention.mla_pre import load_qlora, load_x
from sgl_amd_mono.kimi_k3.gfx950.stages import gemv
from sgl_amd_mono.kimi_k3.gfx950.stages.dims import COLS, HIDDEN, LDS_PAD, NA, ROWS
from sgl_amd_mono.kimi_k3.gfx950.stages.seam import (
    f32_of,
    stage_mix,
    stage_slice,
)
from sgl_amd_mono.kimi_k3.gfx950.stamps import stamp, stamp_begin, stamp_flush
from sgl_amd_mono.kimi_k3.gfx950.tag import step_tag

SOURCES = kernel_sources(__name__)

SGLANG_READS = {
    "srt/models/deepseek_v2.py": (
        "fused_qkv_a_proj_with_mqa",
        "q_a_layernorm",
        "kv_a_layernorm",
        "q_b_proj_qrep_weight",
        "w_kc_qrep",
        "w_vc_qrep",
    ),
    "srt/models/kimi_k3.py": ("g_proj",),
}

G_ROWS = 6  # a request's verify rows (DSpark block 5)
DCP = 8
QL = 1536  # q_lora_rank
KV = 512  # kv_lora_rank
ROPE = 64
DQK = KV + ROPE  # 576: an e4m3 KV record and a q row a head
LATW = QL + KV + ROPE  # 2112: fused_qkv_a's rows
GATE = 1536  # g_proj's rows a rank (12 heads x 128)
NHA = 96  # heads every rank attends
NHL = NHA // DCP  # 12: heads a rank owns after the all-to-all
QHD, QN, VD = 192, 128, 128  # a head's q_b rows, its nope part, v_head_dim
REC = NHL * VD + 2  # 130: bf16 a send record (128 values, the fp32 LSE)
G_LAT = LATW // ROWS  # 132
G_Q = QL // ROWS  # 96
G_KV = KV // ROWS  # 32
S1 = (LATW + GATE) // ROWS  # 228 stage-1 groups
S1_GATE0 = G_LAT
# The CTA roles before the attention (every CTA then takes an attention item):
#   0 .. 55      the seam, then 5 q_b groups
#   56 .. 187    a fused_qkv_a group at launch, then 7 (to 135) or 6 q_b groups
#   188 .. 255   g_proj, off the q path: a group each, 188 .. 215 a second one
QKV0 = NA  # the first fused_qkv_a CTA
GP0 = QKV0 + G_LAT  # 188: the first g_proj CTA
GP2 = GATE // ROWS - (BLOCKS - GP0)  # 28 g_proj CTAs with a second group
QB_SEAM, QB_HI, QB_LO = 5, 7, 6  # q_b groups a seam / early / late qkv_a CTA
QBG = NHA * QHD // ROWS  # 1152 q_b groups
QBG_HEAD = QHD // ROWS  # 12 a head: 8 nope, then 4 rope
QB_ROUNDS = QB_HI  # q_b groups a CTA at most
QB_HI_N = QBG - NA * QB_SEAM - G_LAT * QB_LO  # 80 qkv_a CTAs with 7
ABS_BLK = KV // 128  # 4 absorb tasks a head
ABS = NHA * ABS_BLK  # 384
HQ = 8  # heads an attention item (a row pair x 8 heads fill the MFMA's 16)
NRP = 3  # row pairs (the 6 verify rows)
NHG = NHA // HQ  # 12 head groups
BK = 64  # keys a block
KT = BK // 16  # 16-key QK tiles a block
KC = BK // 32  # 32-key PV steps a block
RWK = DQK // 4 + 4  # i32 words a key in LDS: its record + 16 bytes of pad
PW = BK // 2 + 4  # i32 words a (row, head)'s bf16 P in LDS: its keys + pad
ATT_CTAS = BLOCKS  # every CTA takes an attention item (at most one)
MAX_SPLITS = 21  # the most splits whose items (12 head groups) fit the CTAs
KEYS_PER_SPLIT = 64  # local keys a split aims at (one block)
UV_HALF = 64  # value columns a W_UV task
UV_TASKS = NHA * (VD // UV_HALF)  # 192
UV_CTAS = 192  # a W_UV task a CTA, on CTAs 0 .. 191
UV_ROW = UV_HALF + 8  # bf16 a W_UV tile row in LDS
ABS_ROW = 128 + 8  # bf16 an absorb tile row in LDS
OL_ROW = KV + 8  # bf16 a latent output row in LDS
KV_CTA = NA - 1  # a seam CTA with no stage-1 group
NEG = -3.4028234663852886e38
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
EMPTY_LSE = -1e30  # SGLang's LSE of a row with no key on this rank
# start, x seen, stage 1, q latent normed, q_b, absorb, attention q seen,
# attention done, W_UV partials seen, W_UV done
TL_POINTS = 19

assert GP2 >= 0 and QB_HI_N >= 0 and NA * QB_SEAM + QB_HI_N * QB_HI + (G_LAT - QB_HI_N) * QB_LO == QBG


def splits_of(p: int) -> int:
    """Key splits of a request at committed length ``p``: a function of the
    request's own (global) context only, the same on every rank."""
    lg = -(-(p + G_ROWS) // DCP)
    return max(1, min(MAX_SPLITS, -(-lg // KEYS_PER_SPLIT)))


def dims_of(n: int) -> int:
    """Value-dim ranges an item covers: one (every item all 512 dims)."""
    assert NHG * n <= ATT_CTAS, n
    return 1


@dataclass(frozen=True)
class MlaFrontBuild:
    tokens: int  # S: one request's verify rows
    nblocks: int  # AttnRes block sources (the prefix is one more)
    scale: float  # the softmax scale
    eps: float = 1e-5  # AttnRes norm
    out_eps: float = 1e-5  # input_layernorm
    q_eps: float = 1e-6  # q_a_layernorm
    kv_eps: float = 1e-6  # kv_a_layernorm
    write_idx: int = -1  # a block-write layer's block: the updated prefix stored there
    timeline: bool = False


def scratch_layout(key: MlaFrontBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA,
        "ss": (G_Q + G_KV) * s,
        "qrdy": QBG,
        "absrdy": ABS,
        "kvrdy": 1,
        "itrdy": ATT_CTAS,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())

    def add(name, nbytes):
        nonlocal end
        lay[name] = ((end + 255) // 256 * 256, nbytes)
        end = lay[name][0] + nbytes

    add("x", s * HIDDEN * 2)
    add("lat", s * LATW * 2)
    add("qb", s * NHA * QHD * 2)
    add("q8", s * NHA * DQK)
    add("newkv", DQK)
    add("pv", MAX_SPLITS * s * NHA * KV * 2)
    add("ml", MAX_SPLITS * s * NHA * 2 * 4)
    return lay


def scratch_bytes(key: MlaFrontBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


_BUILDS: dict = {}
_STREAM = fx.Stream(None)


def _bar():
    """``s_barrier`` for an LDS hand-off: this wave's LDS accesses retired, its
    global loads still in flight (``gpu.barrier`` drains them)."""
    llvm.inline_asm(
        res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)\n\ts_barrier",
        constraints="~{memory}", has_side_effects=True,
    )  # fmt: skip


def _published(nwait):
    """Every wave's stores landed (its newest ``nwait`` loads may still be in
    flight), then ``s_barrier``: a hand-off flag may follow."""
    llvm.inline_asm(
        res=None, operands_=[],
        asm_string=f"s_waitcnt vmcnt({nwait}) lgkmcnt(0)\n\ts_barrier",
        constraints="~{memory}", has_side_effects=True,
    )  # fmt: skip


def _st_bf(ptr, i, v, cm=0):
    bo.buffer_store(fx.Float32(v).to(fx.BFloat16), rsrc(ptr), i, cache_modifier=cm)


def e4m3x2(a, b, old, hi):
    """Two f32 -> e4m3 bytes into half ``hi`` of ``old`` (v_cvt_pk_fp8_f32:
    round to nearest even, no saturation), as torch casts bf16 to float8_e4m3fn."""
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, old, hi))


def e4m3x4(v):
    """Four f32 -> one word of four e4m3 bytes, element 0 lowest."""
    lo = e4m3x2(v[0], v[1], fx.Int32(0), False)
    return e4m3x2(v[2], v[3], lo, True)


def _ld(ptr, i, cm=0):
    return fx.Int32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32, cache_modifier=cm)
    )


def _ldf(ptr, i, cm=0):
    return fx.Float32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32, cache_modifier=cm)
    )


def _ld4(ptr, i, cm=0):
    return fx.Vector(
        bo.buffer_load(rsrc(ptr), i, vec_width=4, dtype=T.i32, cache_modifier=cm)
    )


def _mfma16(a, b, c):
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


def _pack_bytes(words, sh):
    """Byte ``sh`` / 8 of each of eight words -> one i64, word j's byte at byte j."""
    lo = fx.Int32(0)
    hi = fx.Int32(0)
    for j in range_constexpr(4):
        lo = lo | (((words[j] >> sh) & 0xFF) << (8 * j))
        hi = hi | (((words[4 + j] >> sh) & 0xFF) << (8 * j))
    return fx.Vector.from_elements([lo, hi], fx.Int32).bitcast(fx.Int64)[0]


@flyc.jit
def _tgemv_ops(lane, wl, wrow, col, k0, step):
    """A transposed GEMV's A operand at one 32-wide K step: weights W[k][col] in
    LDS (``wrow`` bf16 a row, columns contiguous) read down column ``col`` and
    ``col + 1`` at rows k0 + 32 step + 8 (lane / 16) ..: (operand of col,
    operand of col + 1)."""
    g = lane // 16
    kb = k0 + 32 * step + 8 * g
    ws = [fx.ptr_load(wl + ((kb + j) * wrow + col) // 2) for j in range(8)]
    lo = [(ws[j] & 0xFFFF) for j in range(8)]
    hi = [((ws[j] >> 16) & 0xFFFF) for j in range(8)]
    a = fx.Vector.from_elements([lo[2 * i] | (lo[2 * i + 1] << 16) for i in range(4)], fx.Int32)
    b = fx.Vector.from_elements([hi[2 * i] | (hi[2 * i + 1] << 16) for i in range(4)], fx.Int32)
    return a.bitcast(fx.BFloat16), b.bitcast(fx.BFloat16)


def _tgemv(lane, wl, wrow, col0, k0, steps, xl, xrow, s):
    """Columns col0 + 2 (lane % 16) + dt (dt = 0, 1) of X W over K = k0 ..
    k0 + 32 steps, X the token rows in LDS (``xrow`` bf16 a row): two 16x16
    accumulators C[col][token] (lane: columns col0 + 2 (4 (lane / 16) + i) + dt,
    token lane % 16)."""
    g, r16 = lane // 16, lane % 16
    t = fx.min(r16, s - 1)
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for st in range_constexpr(steps):
        a0, a1 = _tgemv_ops(lane, wl, wrow, col0 + 2 * r16, k0, st)
        xb = fx.Vector(
            fx.ptr_load(
                xl + (t * xrow + k0 + 32 * st + 8 * g) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        ).bitcast(fx.BFloat16)
        acc[0] = _mfma16(a0, xb, acc[0])
        acc[1] = _mfma16(a1, xb, acc[1])
    return acc


# ---------------------------------------------------------------- stage 1


def s1_loads(c, g):
    """Stage-1 group ``g``'s weights for this wave (fused_qkv_a, then g_proj)."""
    a = c["args"]
    g = fx.min(g, S1 - 1)
    w = (g < S1_GATE0).select(a["w_qkv"], a["w_g"])
    r0 = (g < S1_GATE0).select(g, g - S1_GATE0)
    return gemv.loads(c["lane"], c["wave"], w, r0 * ROWS, HIDDEN)


@flyc.jit
def stage1(c, g, ops):
    """Rows 16 g .. of stage 1 against the LDS x: a latent group to LAT (and its
    rows' squares to SS for the q / kv norms), a g_proj group to ``gate``."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    acc = gemv.mfmas(lane, wave, c["xl"], HIDDEN + LDS_PAD, HIDDEN, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    mine = tid < ROWS * s
    r = tid % ROWS
    t = fx.min(tid // ROWS, s - 1)
    v = bf16_round(row_sum(red, r, t))
    is_lat = g < S1_GATE0
    if mine & is_lat:
        _st_bf(c["lat"], t * LATW + g * ROWS + r, v, CM_DEV)
    if mine & (g >= S1_GATE0):
        _st_bf(a["gate"], t * GATE + (g - S1_GATE0) * ROWS + r, v)
    gpu.barrier()
    if mine:
        fx.ptr_store(v * v, red + tid)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if (tid < s) & (g < G_Q + G_KV):
        tot = fx.ptr_load(red + tid * ROWS)
        for rr in range_constexpr(1, ROWS):
            tot = tot + fx.ptr_load(red + (tid * ROWS + rr))
        c["put"](c["ss"], g * s + tid, tot)
    gpu.barrier()


# ---------------------------------------------------------------- the request


def request(c):
    """The request's global committed length P (its global KV indptr span less
    the verify rows) and this rank's view of it: (P, local prefix keys L, the
    verify row this rank owns or 8, splits n, dim ranges d)."""
    a = c["args"]
    p = uniform(_ld(a["g_kv_ptr"], 1) - _ld(a["g_kv_ptr"], 0)) - G_ROWS
    rank = c["rank"]
    pm = p & (DCP - 1)
    lr = (p >> 3) + (rank < pm).select(fx.Int32(1), fx.Int32(0))
    own = (rank - pm + DCP) & (DCP - 1)  # the row at position P + own
    own = (own < G_ROWS).select(own, fx.Int32(DCP))
    lg = (p + G_ROWS + DCP - 1) >> 3
    n = fx.max(fx.Int32(1), fx.min(fx.Int32(MAX_SPLITS), (lg + KEYS_PER_SPLIT - 1) // KEYS_PER_SPLIT))
    return p, lr, own, n, fx.Int32(1)


# ---------------------------------------------------------------- the KV record


@flyc.jit
def stage_kv(c):
    """kv_a_layernorm of the latent, then the e4m3 record (normed latent, raw
    rope part) of the new token this rank owns: its KV row and NEWKV."""
    s, tid, red = c["S"], c["tid"], c["red"]
    a = c["args"]
    _, _, own, _, _ = request(c)
    n = G_KV * s
    got = c["poll"]([(c["ss"], G_Q * s + fx.min(tid, n - 1), 1)])
    if tid < n:
        fx.ptr_store(f32_of(got[0][0]), red + tid)
    gpu.barrier()
    if tid < s:
        tot = fx.ptr_load(red + tid)
        for gg in range_constexpr(1, G_KV):
            tot = tot + fx.ptr_load(red + (gg * s + tid))
        fx.ptr_store(
            hw_rsq(tot / fx.Float32(KV) + fx.Float32(c["kv_eps"])),
            red + (2048 - 16 + tid),
        )
    gpu.barrier()
    t = fx.min(own, s - 1)
    owned = own < s
    words = DQK // 4  # 144 record words, 4 dims each
    if owned & (tid < words):
        k = tid * 4
        lw = fx.Vector(
            bo.buffer_load(rsrc(c["lat"]), (t * LATW + QL + k) // 2, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV)
        )
        vals = []
        for h in range_constexpr(2):
            w = fx.Int32(lw[h])
            vals += [(w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)]
        if k < KV:
            nw = fx.Vector(bo.buffer_load(rsrc(a["kv_nw"]), k // 2, vec_width=2, dtype=T.i32))
            rs = fx.ptr_load(red + (2048 - 16 + t))
            ws = []
            for h in range_constexpr(2):
                w = fx.Int32(nw[h])
                ws += [(w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)]
            vals = [bf16_round((vals[j] * rs) * ws[j]) for j in range(4)]
        word = e4m3x4(vals)
        row = fx.Int64(_ld(a["loc"], t * c["loc_words"])) >> 3
        p = fx.inttoptr(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
            a["kv"] + row * fx.Int64(DQK) + fx.Int64(k),
        )
        fx.generic_store(p, word)
        bo.buffer_store(word, rsrc(c["newkv"]), tid, cache_modifier=CM_DEV)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](c["kvrdy"], 0, fx.Int32(1))
    gpu.barrier()


# ---------------------------------------------------------------- stage 2: q_b


def qb_range(bid):
    """(first group, count) of CTA ``bid``'s q_b groups (contiguous)."""
    seam = bid < NA
    hi = (bid >= QKV0) & (bid < QKV0 + QB_HI_N)
    lo = (bid >= QKV0 + QB_HI_N) & (bid < GP0)
    first = seam.select(
        bid * QB_SEAM,
        hi.select(
            NA * QB_SEAM + (bid - QKV0) * QB_HI,
            NA * QB_SEAM + QB_HI_N * QB_HI + (bid - QKV0 - QB_HI_N) * QB_LO,
        ),
    )
    count = seam.select(fx.Int32(QB_SEAM), hi.select(fx.Int32(QB_HI), lo.select(fx.Int32(QB_LO), fx.Int32(0))))
    return first, count


def s2_loads(c, h):
    """q_b group ``h``'s weights (past the last group: zero-range loads)."""
    a = c["args"]
    nbytes = (h < QBG).select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
    h = fx.min(h, QBG - 1)
    return gemv.loads(c["lane"], c["wave"], a["w_qb"], h * ROWS, QL, nbytes=nbytes)


S2_LOADS = 4 * gemv.chunks(QL)[0]  # a wave's loads of one q_b group


@flyc.jit
def stage2(c, h, ops):
    """Rows 16 h .. of q_b against the normed q latent in LDS: bf16 to QB; a rope
    group also as e4m3 into Q8's last 64 dims (``stage2_flag`` publishes)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    acc = gemv.mfmas(lane, wave, c["xl"], QL + LDS_PAD, QL, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    _bar()
    mine = tid < ROWS * s
    r = tid % ROWS
    t = fx.min(tid // ROWS, s - 1)
    v = bf16_round(row_sum(red, r, t))
    row = h * ROWS + r
    if mine:
        _st_bf(c["qb"], t * (NHA * QHD) + row, v, CM_DEV)
    # a rope group: four rows' e4m3 a word into Q8 (rows 4 q .. on lane 4 q)
    head = h // QBG_HEAD
    d = (h % QBG_HEAD) * ROWS + r - QN
    vb = v.bitcast(fx.Int32)
    nb = [
        fx.Int32(rocdl.ds_bpermute(T.i32, ((lane + k) % 64) * 4, vb)).bitcast(fx.Float32)
        for k in range(1, 4)
    ]
    if mine & ((h % QBG_HEAD) >= QN // ROWS) & ((r % 4) == 0):
        bo.buffer_store(
            e4m3x4([v] + nb),
            rsrc(c["q8"]),
            (t * (NHA * DQK) + head * DQK + KV + d) // 4,
            cache_modifier=CM_DEV,
        )


@flyc.jit
def stage2_flag(c, h, nwait):
    """Group h's rows landed (the newest ``nwait`` loads aside), then its QRDY."""
    _published(nwait)
    if c["tid"] == 0:
        c["put"](c["qrdy"], h, fx.Int32(1))


# ---------------------------------------------------------------- stage 3: absorb


def s3_loads(c, task):
    """Absorb task ``task``'s W_UK tile, W[h][n][128 b ..+128], 16 bytes a piece
    (4 pieces a thread, row n = piece / 16)."""
    a = c["args"]
    tid = c["tid"]
    task = fx.min(task, ABS - 1)
    h = task // ABS_BLK
    b = task % ABS_BLK
    out = []
    for i in range_constexpr(4):
        p = tid + THREADS * i
        n = p // 16
        seg = p % 16
        out.append(_ld4(a["w_kc"], (h * (QN * KV) + n * KV + b * 128 + seg * 8) // 2))
    return out


@flyc.jit
def stage3(c, task, wt):
    """q_nope @ W_UK for head task // 4, latent columns 128 (task % 4) ..: the
    head's 8 nope q_b groups (QRDY) into LDS, the tile into LDS, transposed MFMAs
    (wave w: 32 columns, a K half), bf16 -> e4m3 into Q8; then ABSRDY."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    h = task // ABS_BLK
    b = task % ABS_BLK
    if tid < QN // ROWS:
        c["poll"]([(c["qrdy"], h * QBG_HEAD + tid, 1)])
    gpu.barrier()
    xl, wl = c["xl"], c["wl"]
    # q_nope rows: 6 x 128 bf16, 16 bytes a thread
    if tid < s * (QN // 8):
        t = tid // (QN // 8)
        k = (tid % (QN // 8)) * 8
        v = _ld4(c["qb"], (t * (NHA * QHD) + h * QHD + k) // 2, CM_DEV)
        fx.ptr_store(v, xl + (t * (QN + LDS_PAD) + k) // 2)
    for i in range_constexpr(4):
        p = tid + THREADS * i
        n = p // 16
        seg = p % 16
        fx.ptr_store(wt[i], wl + (n * ABS_ROW + seg * 8) // 2)
    gpu.barrier()
    cq = wave % 4
    kh = wave // 4
    acc = _tgemv(lane, wl, ABS_ROW, 32 * cq, 64 * kh, 2, xl, QN + LDS_PAD, s)
    # the two K halves summed in order: wave w + 4 adds onto wave w's
    rb = red + (wave * 64 + lane) * 8
    for dt in range_constexpr(2):
        fx.ptr_store(acc[dt], rb + 4 * dt)
    gpu.barrier()
    g, r16 = lane // 16, lane % 16
    if (wave < 4) & (r16 < s):
        lo = red + (wave * 64 + lane) * 8
        hi = red + ((wave + 4) * 64 + lane) * 8
        vals = [None] * 8
        for dt in range_constexpr(2):
            for i in range_constexpr(4):
                vals[2 * i + dt] = bf16_round(fx.ptr_load(lo + (4 * dt + i)) + fx.ptr_load(hi + (4 * dt + i)))
        w0 = e4m3x4(vals[0:4])
        w1 = e4m3x4(vals[4:8])
        col = b * 128 + 32 * cq + 8 * g
        bo.buffer_store(
            fx.Vector.from_elements([w0, w1], fx.Int32),
            rsrc(c["q8"]),
            (r16 * (NHA * DQK) + h * DQK + col) // 4,
            cache_modifier=CM_DEV,
        )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](c["absrdy"], task, fx.Int32(1))
    gpu.barrier()


# ---------------------------------------------------------------- attention




KV_SLOTS = RWK * 4 // 16  # 37 16-byte LDS slots a key: its 36 pieces, then pad
DMA_N = BK * KV_SLOTS // 64  # 37 wave-wide DMAs a block
DMA_W0 = NRP  # the DMA wave: 3 (QK on waves 0 .. 2, P V on 4 .. 7)
DMA_WAVES = 1
PV_W0 = DMA_W0 + 1  # the P V waves, 128 value dims each
DMA_ROUNDS = -(-DMA_N // DMA_WAVES)  # 8 a wave at most
DMA_FULL = DMA_N - (DMA_ROUNDS - 1) * DMA_WAVES  # waves with DMA_ROUNDS (the rest one fewer)
KV_BUFS = 3  # LDS record buffers: block i in use, i + 1 landed, i + 2 in flight
assert BK * KV_SLOTS % 64 == 0


@flyc.jit
def _kv_dma(c, kvp, lr, b1, blk, kvl):
    """Block ``blk``'s records straight into LDS at ``kvl`` (padded rows), a
    16-byte slot a lane: a wave's DMA fills 64 consecutive slots (DMA wave
    only). Lane l reads key l's row first; the pad slots and the new token's
    (patched after, ``_kv_patch``) take a valid dummy row."""
    a = c["args"]
    lane, wave = c["lane"], c["wave"]
    if wave == DMA_W0:
        r_kv = rsrc(a["kv"])
        kk = blk * BK + lane
        row_l = _ld(a["kv_idx"], kvp + fx.max(fx.min(kk, lr - 1), 0))
        live_l = (kk < lr) & (blk < b1)
        row_l = live_l.select(row_l * DQK, fx.Int32(0))
        for i in range_constexpr(DMA_N):
            q = i * 64 + lane
            key = q // KV_SLOTS
            part = q % KV_SLOTS
            off = fx.Int32(rocdl.ds_bpermute(T.i32, key * 4, row_l))
            voff = (part < KV_SLOTS - 1).select(off + part * 16, fx.Int32(0))
            rocdl.raw_ptr_buffer_load_lds(r_kv, (kvl + i * 256).llvm_ptr, 16, voff, fx.Int32(0), 0, 0)


@flyc.jit
def _kv_patch(c, lr, own, blk, kvl):
    """The new token's record from NEWKV into its slot, where block ``blk``
    holds it."""
    e = c["tid"] - DMA_W0 * 64  # a thread of the DMA waves
    if (own < G_ROWS) & (lr // BK == blk) & (e >= 0) & (e < DQK // 16):
        v = _ld4(c["newkv"], 4 * e, CM_DEV)
        fx.ptr_store(v, kvl + ((lr % BK) * RWK + 4 * e))


QK_STEPS = -(-DQK // 128)  # 5 K = 128 e4m3 MFMAs a key tile (the last half zero)


def _mfma128(a, b, c):
    """C += A B over 128 e4m3 (E8M0 scales 1): A, B 32 bytes a lane, the same K
    to lane mapping on both, so any mapping gives the dot product."""
    one = fx.Int32(127)
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one])
    )


def _k128(lane, words8, kc):
    """Step kc's 32 e4m3 of a lane (dims 128 kc + 32 (lane / 16) ..); past the
    576 dims zero."""
    g = lane // 16
    if const_expr(128 * (kc + 1) <= DQK):
        return words8
    live = g < (DQK - 128 * kc) // 32
    return fx.Vector.from_elements([live.select(words8[e], fx.Int32(0)) for e in range(8)], fx.Int32)


def _qk(lane, kvl, ql):
    """The block's scores against one row pair's 8 heads (``ql``: its e4m3 q in
    LDS, a (row, head) a key-sized row): KT 16-key tiles C[key][rh] (element i of tile kt:
    key 16 kt + 4 (lane / 16) + i; rh = lane % 16: row 2 rp + rh / 8, head rh % 8)."""
    g, r16 = lane // 16, lane % 16
    v4 = fx.Vector.make_type(4, fx.Int32)
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(KT)]

    def row32(base, kc):
        # past the record the reads stay in the row (its pad, then the next row)
        w = base + fx.min(32 * kc + 8 * g, DQK // 4 - 8)
        lo = fx.Vector(fx.ptr_load(w, result_type=v4))
        hi = fx.Vector(fx.ptr_load(w + 4, result_type=v4))
        return _k128(lane, fx.Vector.from_elements([lo[e] for e in range(4)] + [hi[e] for e in range(4)], fx.Int32), kc)

    for kc in range_constexpr(QK_STEPS):
        qo = row32(ql + r16 * RWK, kc)
        for kt in range_constexpr(KT):
            acc[kt] = _mfma128(row32(kvl + (16 * kt + r16) * RWK, kc), qo, acc[kt])
    return acc


def _softmax(lane, t, blk, kend, sacc, lr, own, m, l, qk_scale, prow):
    """(row t, head) lane % 16's online softmax over the block: the new (max,
    sum, factor of the earlier sums); its P (bf16, keys the row does not see 0)
    to the (row, head)'s LDS."""
    g = lane // 16
    sv = []
    for kt in range_constexpr(KT):
        for i in range_constexpr(4):
            kk = blk * BK + 16 * kt + 4 * g + i
            ok = ((kk < lr) | ((kk == lr) & (own <= t))) & (kk < kend)
            sv.append(ok.select(sacc[kt][i] * qk_scale, fx.Float32(NEG)))
    mb = sv[0]
    for e in range_constexpr(1, 4 * KT):
        mb = fx.max(mb, sv[e])
    # xred reduces an f32 as f32 only when it is typed so
    mb = xred(xred(fx.Float32(mb), 16, fx.max), 32, fx.max)
    mn = fx.max(m, mb)
    a = hw_exp2(m - mn)
    ls = fx.Float32(0.0)
    for kt in range_constexpr(KT):
        ps = [(sv[4 * kt + i] > fx.Float32(NEG)).select(hw_exp2(sv[4 * kt + i] - mn), fx.Float32(0.0)) for i in range(4)]
        ls = ls + ((ps[0] + ps[1]) + (ps[2] + ps[3]))
        pw = fx.Vector.from_elements(ps, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        fx.ptr_store(pw, prow + (8 * kt + 2 * g))
    ls = xred(xred(fx.Float32(ls), 16, lambda x, y: x + y), 32, lambda x, y: x + y)
    return mn, l * a + ls, a


def _v_ops(lane, kvl, dbase, kc, h):
    """PV's A operands of 32-key step kc for 16-dim tiles 4 h .. 4 h + 3: tile dt
    holds dim dbase + 8 (lane % 16) + dt at keys 32 kc + 8 (lane / 16) .. + 8,
    widened to bf16 (exact)."""
    g, r16 = lane // 16, lane % 16
    word = (dbase + 8 * r16) // 4 + h
    ws = [fx.ptr_load(kvl + ((32 * kc + 8 * g + j) * RWK + word)) for j in range(8)]
    out = []
    for dt in range_constexpr(4):
        v = fx.Vector.from_elements([_pack_bytes(ws, 8 * dt)], fx.Int64).bitcast(fx.Int32)
        out.append(fp8_to_bf16x8(v[0], v[1]))
    return out


def _scaled(v, a):
    return fx.Vector.from_elements([v[i] * a for i in range(4)], fx.Float32)


def _pv(lane, wave, kx, pl, al, acc):
    """Block (LDS ``kx``, its P at ``pl`` and factors at ``al``) P V onto every
    row pair's accumulators (rescaled first): this wave's 128 dims, 8 tiles x
    3 row pairs."""
    g, r16 = lane // 16, lane % 16
    dbase = 128 * (wave - PV_W0)
    out = list(acc)
    for rp in range_constexpr(NRP):
        alpha = fx.ptr_load(al + (rp * 16 + r16)).bitcast(fx.Float32)
        for dt in range_constexpr(8):
            out[rp * 8 + dt] = _scaled(acc[rp * 8 + dt], alpha)
    for kc in range_constexpr(KC):
        for h in range_constexpr(2):
            vops = _v_ops(lane, kx, dbase, kc, h)
            for rp in range_constexpr(NRP):
                pw = fx.ptr_load(pl + ((rp * 16 + r16) * PW + 16 * kc + 4 * g), result_type=fx.Vector.make_type(4, fx.Int32))
                pb = fx.Vector(pw).bitcast(fx.BFloat16)
                for dt in range_constexpr(4):
                    out[rp * 8 + 4 * h + dt] = _mfma16(vops[dt], pb, out[rp * 8 + 4 * h + dt])
    return out


@flyc.jit
def _partial_out(c, lane, wave, sp, hg, acc):
    """Each (row, head)'s partial over its own sum, f16, to PV [split][row][head][512]:
    lane (g, rh) of row pair rp holds dims 128 (wave - 4) + 32 g + 8 i + dt."""
    s = c["S"]
    g, r16 = lane // 16, lane % 16
    mll = c["mll"]
    if wave >= PV_W0:
        for rp in range_constexpr(NRP):
            ll = fx.ptr_load(mll + 2 * (rp * 16 + r16) + 1).bitcast(fx.Float32)
            inv = (ll > fx.Float32(0.0)).select(fx.Float32(1.0) / ll, fx.Float32(0.0))
            vals = [None] * 32
            for i in range_constexpr(4):
                for dt in range_constexpr(8):
                    vals[8 * i + dt] = acc[rp * 8 + dt][i] * inv
            r = 2 * rp + r16 // HQ
            head = hg * HQ + r16 % HQ
            o0 = (((sp * s + r) * NHA + head) * KV + 128 * (wave - PV_W0) + 32 * g) // 2
            for q in range_constexpr(4):
                hv = fx.Vector.from_elements(vals[8 * q : 8 * q + 8], fx.Float32).to(fx.Float16).bitcast(fx.Int32)
                bo.buffer_store(hv, rsrc(c["pv"]), o0 + 4 * q, cache_modifier=CM_DEV)


@flyc.jit
def _qk_softmax(c, kx, blk, kend, m, l, qf, t, lr, own, qk_scale, pl, al):
    """Block ``blk`` (LDS ``kx``; past the split it scores nothing): QK and the
    online softmax on waves 0 .. 2 (a row pair each): P to ``pl``, the factors
    of the earlier sums to ``al``."""
    lane, wave = c["lane"], c["wave"]
    g, r16 = lane // 16, lane % 16
    if wave < NRP:
        sacc = _qk(lane, kx, qf)
        m, l, al_ = _softmax(lane, t, blk, kend, sacc, lr, own, m, l, qk_scale, pl + (wave * 16 + r16) * PW)
        if g == 0:
            fx.ptr_store(al_.bitcast(fx.Int32), al + (wave * 16 + r16))
    return m, l


@flyc.jit
def attention(c, hg, sp):
    """Item (8-head group hg, key split sp) over the split's blocks of this
    rank's keys; its partial, its (max, sum), ITRDY."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    g, r16 = lane // 16, lane % 16
    _, lr, own, n, _ = request(c)
    item = hg * n + sp
    nbl = (lr + (own < G_ROWS).select(fx.Int32(1), fx.Int32(0)) + BK - 1) // BK
    b0 = sp * nbl // n
    b1 = (sp + 1) * nbl // n
    kvp = uniform(_ld(a["kv_ptr"], 0))
    kvl0 = c["kvl"]
    # the split's rows, then its first two blocks' records in flight before the
    # waits (the keys need no q)
    kv1 = kvl0 + BK * RWK
    kend = b1 * BK
    _kv_dma(c, kvp, lr, b1, b0, kvl0)
    _kv_dma(c, kvp, lr, b1, fx.min(b0 + 1, b1 - 1), kv1)
    # the group's q: its 8 heads' absorb tasks and rope groups; the new token
    if tid < HQ * ABS_BLK:
        c["poll"]([(c["absrdy"], (hg * HQ + tid // ABS_BLK) * ABS_BLK + tid % ABS_BLK, 1)])
    if (tid >= HQ * ABS_BLK) & (tid < 2 * HQ * ABS_BLK):
        e = tid - HQ * ABS_BLK
        c["poll"]([(c["qrdy"], (hg * HQ + e // 4) * QBG_HEAD + QN // ROWS + e % 4, 1)])
    has_new = (own < G_ROWS) & (b1 > lr // BK) & (b0 <= lr // BK)
    if has_new & (tid == 2 * HQ * ABS_BLK):
        c["poll"]([(c["kvrdy"], 0, 1)])
    _bar()
    stamp(c["on"], c["tls"], tid, 6)
    # the group's q into LDS: (row pair rp, rh) a key-sized row, rh = lane % 16 of
    # QK wave rp: row 2 rp + rh / 8, head rh % 8
    rp = fx.min(wave, NRP - 1)
    t = 2 * rp + r16 // HQ
    qrow = DQK // 16  # 16-byte pieces a (row, head)
    for j in range_constexpr(-(-(NRP * 16 * qrow) // THREADS)):
        e = fx.min(tid + THREADS * j, NRP * 16 * qrow - 1)
        rr = e // qrow
        pq = e % qrow
        tq = 2 * (rr // 16) + (rr % 16) // HQ
        hq = hg * HQ + rr % HQ
        v = _ld4(c["q8"], (tq * (NHA * DQK) + hq * DQK) // 4 + 4 * pq, CM_DEV)
        if tid + THREADS * j < NRP * 16 * qrow:
            fx.ptr_store(v, c["qll"] + (rr * RWK + 4 * pq))
    qf = c["qll"] + rp * (16 * RWK)
    qk_scale = c["scale"] * LOG2E
    m = fx.Float32(NEG)
    l = fx.Float32(0.0)
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(NRP * 8)]
    # q and the first two blocks landed: the loop's waits count only its DMAs
    rocdl.s_waitcnt(vmcnt=0)
    _kv_patch(c, lr, own, b0, kvl0)
    _kv_patch(c, lr, own, b0 + 1, kv1)
    _bar()
    pl0, al0 = c["pl"], c["al"]
    # block b0's scores first; then each step overlaps three things, one wave
    # set each: QK + softmax of block blk + 1 (waves 0 .. 2), the DMA of blk + 2
    # (wave 3), P V of blk (waves 4 .. 7). Buffers: records (blk - b0) % 3, P
    # and factors (blk - b0) % 2.
    m, l = _qk_softmax(c, kvl0, b0, kend, m, l, qf, t, lr, own, qk_scale, pl0, al0)
    _bar()
    for blk in range(b0, b1):
        i = blk - b0
        x = i % KV_BUFS
        kx = kvl0 + x * (BK * RWK)
        ky = kvl0 + ((x + 1) % KV_BUFS) * (BK * RWK)
        kz = kvl0 + ((x + 2) % KV_BUFS) * (BK * RWK)
        par = i % 2
        pc, ac = pl0 + par * (NRP * 16 * PW), al0 + par * (NRP * 16)
        pn, an = pl0 + (1 - par) * (NRP * 16 * PW), al0 + (1 - par) * (NRP * 16)
        stamp(c["on"], c["tls"], tid, 10)
        m, l = _qk_softmax(c, ky, blk + 1, kend, m, l, qf, t, lr, own, qk_scale, pn, an)
        if wave == DMA_W0:
            _kv_dma(c, kvp, lr, b1, fx.min(blk + 2, b1 - 1), kz)
            rocdl.s_waitcnt(vmcnt=0)
        _kv_patch(c, lr, own, blk + 2, kz)
        if wave >= PV_W0:
            acc = _pv(lane, wave, kx, pc, ac, acc)
        _bar()
        stamp(c["on"], c["tls"], tid, 13)
    stamp(c["on"], c["tls"], tid, 7)
    # (max, sum) a (row, head) to LDS, then the partial over its own sum
    mll = c["mll"]
    if (wave < NRP) & (g == 0):
        fx.ptr_store(m.bitcast(fx.Int32), mll + 2 * (wave * 16 + r16))
        fx.ptr_store(l.bitcast(fx.Int32), mll + 2 * (wave * 16 + r16) + 1)
    gpu.barrier()
    _partial_out(c, lane, wave, sp, hg, acc)
    if (wave < NRP) & (g == 0):
        bo.buffer_store(
            fx.Vector.from_elements([m, l], fx.Float32),
            rsrc(c["ml"]),
            ((sp * s + t) * NHA + hg * HQ + r16 % HQ) * 2,
            cache_modifier=CM_DEV,
        )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](c["itrdy"], item, fx.Int32(1))
    gpu.barrier()


# ---------------------------------------------------------------- W_UV


UV_SLOTS = UV_ROW * 2 // 16  # 9 16-byte LDS slots a tile row: its 8 pieces, then pad
UV_DMA_N = KV * UV_SLOTS // 64  # 72 wave-wide DMAs a tile
UV_DMA_W0 = 6  # the DMA waves (the merge runs on waves 0 .. 5)
UV_DMA_ROUNDS = UV_DMA_N // (WAVES - UV_DMA_W0)


@flyc.jit
def _uv_dma(c, task):
    """Task ``task``'s W_UV tile straight into LDS (rows padded), on the DMA waves."""
    a = c["args"]
    lane, wave = c["lane"], c["wave"]
    h = task // 2
    half = task % 2
    r_w = rsrc(a["w_vc"])
    for r in range_constexpr(UV_DMA_ROUNDS):
        i = r * (WAVES - UV_DMA_W0) + wave - UV_DMA_W0
        q = i * 64 + lane
        k = q // UV_SLOTS
        seg = q % UV_SLOTS
        voff = (seg < UV_SLOTS - 1).select(((h * KV + k) * VD + half * UV_HALF + 8 * seg) * 2, fx.Int32(0))
        if wave >= UV_DMA_W0:
            rocdl.raw_ptr_buffer_load_lds(r_w, (c["wl"] + i * 256).llvm_ptr, 16, voff, fx.Int32(0), 0, 0)


@flyc.jit
def stage_uv(c, task):
    """Head task // 2's split partials (its group's items) merged in split order
    -> the bf16 latent output in LDS and the head's LSE; then W_UV for value
    columns 64 (task % 2) .. -> ``send`` (bf16), the LSE word with the first half."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    h = task // 2
    half = task % 2
    _, _, _, n, d = request(c)
    nd = n * d
    hg = h // HQ
    # the tile streams into LDS on waves 6, 7 while wave 0 polls and waves
    # 0 .. 5 merge (a wave's loads complete in order)
    _uv_dma(c, task)
    if tid < nd:
        c["poll"]([(c["itrdy"], hg * nd + tid, 1)])
    _bar()
    stamp(c["on"], c["tls"], tid, 8)
    xl, wl = c["xl"], c["wl"]
    owner = h // NHL
    hl = h % NHL
    # merge: thread (row t, 8 dims at 8 o)
    if tid < s * (KV // 8):
        t = tid // (KV // 8)
        o = tid % (KV // 8)
        mx = fx.Float32(NEG)
        mls = []
        for sp in range_constexpr(MAX_SPLITS):
            live = sp < n
            spc = fx.min(fx.Int32(sp), n - 1)
            mw = fx.Vector(
                bo.buffer_load(rsrc(c["ml"]), ((spc * s + t) * NHA + h) * 2, vec_width=2, dtype=T.f32, cache_modifier=CM_DEV)
            )
            mm = live.select(mw[0], fx.Float32(NEG))
            ll = live.select(mw[1], fx.Float32(0.0))
            mls.append((mm, ll, spc))
            mx = fx.max(mx, mm)
        lsum = fx.Float32(0.0)
        tot = [fx.Float32(0.0)] * 8
        for sp in range_constexpr(MAX_SPLITS):
            mm, ll, spc = mls[sp]
            wgt = hw_exp2(mm - mx) * ll
            lsum = lsum + wgt
            pv = _ld4(c["pv"], (((spc * s + t) * NHA + h) * KV + 8 * o) // 2, CM_DEV)
            vals = fx.Vector(pv).bitcast(fx.Float16).to(fx.Float32)
            tot = [tot[e] + wgt * vals[e] for e in range(8)]
        inv = (lsum > fx.Float32(0.0)).select(fx.Float32(1.0) / lsum, fx.Float32(0.0))
        ob = fx.Vector.from_elements([tot[e] * inv for e in range(8)], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        fx.ptr_store(ob, xl + (t * OL_ROW + 8 * o) // 2)
        if (o == 0) & (half == 0):
            lse = (lsum > fx.Float32(0.0)).select(
                (mx + fx.Float32(fmath.log2(lsum))) * fx.Float32(LN2),
                fx.Float32(EMPTY_LSE),
            )
            bo.buffer_store(lse, rsrc(a["send"]), ((owner * s + t) * NHL + hl) * (REC // 2) + VD // 2)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    vq = wave % 2
    kq = wave // 2
    acc = _tgemv(lane, wl, UV_ROW, 32 * vq, 128 * kq, 4, xl, OL_ROW, s)
    rb = red + (wave * 64 + lane) * 8
    for dt in range_constexpr(2):
        fx.ptr_store(acc[dt], rb + 4 * dt)
    gpu.barrier()
    g, r16 = lane // 16, lane % 16
    if (wave < 2) & (r16 < s):
        vals = [None] * 8
        for dt in range_constexpr(2):
            for i in range_constexpr(4):
                tot = fx.Float32(0.0)
                for q in range_constexpr(4):
                    tot = tot + fx.ptr_load(red + (((2 * q + wave) * 64 + lane) * 8 + 4 * dt + i))
                vals[2 * i + dt] = tot
        ob = fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        col = half * UV_HALF + 32 * wave + 8 * g
        bo.buffer_store(ob, rsrc(a["send"]), (((owner * s + r16) * NHL + hl) * REC + col) // 2)
    gpu.barrier()


# ---------------------------------------------------------------- the launch

ARGS = (
    "prefix", "blocks", "ares_nw", "ares_qk", "in_nw", "w_qkv", "w_g", "q_nw",
    "kv_nw", "w_qb", "w_kc", "w_vc", "kv", "kv_idx", "kv_ptr", "g_kv_ptr", "loc",
    "gate", "send",
)  # fmt: skip


def build(key: MlaFrontBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    assert s == G_ROWS, s
    ns = key.nblocks + 1
    lay = scratch_layout(key)
    assert s * (HIDDEN + LDS_PAD) * 2 <= 136 * 1024

    @fx.struct
    class AttnLds:
        red: fx.Array[fx.Float32, WAVES * 64 * 8, 16]
        vals: fx.Array[fx.Float32, s * ns * COLS, 16]

    @fx.struct
    class GemvLds:
        red: fx.Array[fx.Float32, WAVES * 64 * 8, 16]
        xl: fx.Array[fx.Int32, s * (HIDDEN + LDS_PAD) // 2, 16]

    @fx.struct
    class AbsLds:
        red: fx.Array[fx.Float32, WAVES * 64 * 8, 16]
        xl: fx.Array[fx.Int32, s * (QN + LDS_PAD) // 2, 16]
        wl: fx.Array[fx.Int32, QN * ABS_ROW // 2, 16]

    @fx.struct
    class CoreLds:
        kv: fx.Array[fx.Int32, KV_BUFS * BK * RWK, 16]
        p: fx.Array[fx.Int32, 2 * NRP * 16 * PW, 16]
        alpha: fx.Array[fx.Int32, 2 * NRP * 16, 16]
        ml: fx.Array[fx.Int32, 2 * NRP * 16, 16]
        q: fx.Array[fx.Int32, NRP * 16 * RWK, 16]

    @fx.struct
    class UvLds:
        red: fx.Array[fx.Float32, WAVES * 64 * 8, 16]
        xl: fx.Array[fx.Int32, s * OL_ROW // 2, 16]
        wl: fx.Array[fx.Int32, KV * UV_ROW // 2, 16]

    @fx.union
    class RoleLds:
        attn: AttnLds
        gemv: GemvLds
        absorb: AbsLds
        core: CoreLds
        uv: UvLds

    @fx.struct
    class Smem:
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, SOURCES)
    name = (
        f"k3_mono_mla_front_s{s}_nb{key.nblocks}_w{int(key.write_idx >= 0)}"
        f"_t{int(key.timeline)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mla_front(
        prefix: Int64, blocks: Int64, blk_sm: Int32, blk_sr: Int32, ares_nw: Int64,
        ares_qk: Int64, in_nw: Int64, w_qkv: Int64, w_g: Int64, q_nw: Int64,
        kv_nw: Int64, w_qb: Int64, w_kc: Int64, w_vc: Int64, kv: Int64, kv_idx: Int64,
        kv_ptr: Int64, g_kv_ptr: Int64, loc: Int64, gate: Int64, send: Int64,
        scratch: Int64, rank: Int32, layer: Int32, epoch: Int64, tl: Int64,
    ):  # fmt: skip
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        role = alloc.allocate(RoleLds)
        al, gl = role.attn.peek(), role.gemv.peek()
        abl, cl, ul = role.absorb.peek(), role.core.peek(), role.uv.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        c = {
            "S": s, "NS": ns, "NB": key.nblocks, "tid": tid, "bid": bid,
            "lane": tid % 64, "wave": tid // 64, "delta": False, "debug": False,
            "eps": key.eps, "out_eps": key.out_eps, "q_eps": key.q_eps,
            "kv_eps": key.kv_eps, "write_idx": key.write_idx, "gate_local": True,
            "qlora_out": False, "qn": False, "put": mb.put, "poll": mb.poll,
            # every role but the attention's starts with the same red (offset 0)
            "red": al.red.ptr, "vals": al.vals.ptr, "xl": gl.xl.ptr,
            "scale": key.scale, "rank": rank, "loc_words": 2,
            "xbuf": scratch + fx.Int64(lay["x"][0]),
            "lat": scratch + fx.Int64(lay["lat"][0]),
            "qb": scratch + fx.Int64(lay["qb"][0]),
            "q8": scratch + fx.Int64(lay["q8"][0]),
            "newkv": scratch + fx.Int64(lay["newkv"][0]),
            "pv": scratch + fx.Int64(lay["pv"][0]),
            "ml": scratch + fx.Int64(lay["ml"][0]),
            "kvl": cl.kv.ptr, "pl": cl.p.ptr, "al": cl.alpha.ptr, "mll": cl.ml.ptr,
            "qll": cl.q.ptr,
            "args": {
                "prefix": prefix, "delta": 0, "blocks": blocks, "blk_sm": blk_sm,
                "blk_sr": blk_sr, "ares_nw": ares_nw, "ares_qk": ares_qk, "in_nw": in_nw,
                "x_dbg": 0, "w_qkv": w_qkv, "w_g": w_g, "q_nw": q_nw, "kv_nw": kv_nw,
                "w_qb": w_qb, "w_kc": w_kc, "w_vc": w_vc, "kv": kv, "kv_idx": kv_idx,
                "kv_ptr": kv_ptr, "g_kv_ptr": g_kv_ptr, "loc": loc, "gate": gate,
                "send": send, "q_lora": 0,
            },
        }  # fmt: skip
        for region in ("part", "wgt", "msq", "xrdy", "ss", "qrdy", "absrdy", "kvrdy", "itrdy"):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        c["on"], c["tls"] = on, tls
        stamp_begin(on, tls, tid, TL_POINTS)
        if bid < NA:
            stage_slice(c, bid)
            stage_mix(c, bid)
            if bid == KV_CTA:
                stage_kv(c)
        if (bid >= QKV0) & (bid < GP0):
            # fused_qkv_a: the q path's first weights, at launch
            ops = s1_loads(c, bid - QKV0)
            load_x(c)
            stage1(c, bid - QKV0, ops)
        if bid >= GP0:
            # g_proj, read only after the launch
            ops = s1_loads(c, S1_GATE0 + bid - GP0)
            load_x(c)
            stage1(c, S1_GATE0 + bid - GP0, ops)
            if bid < GP0 + GP2:
                ops2 = s1_loads(c, S1_GATE0 + BLOCKS - GP0 + bid - GP0)
                stage1(c, S1_GATE0 + BLOCKS - GP0 + bid - GP0, ops2)
        stamp(on, tls, tid, 2)
        # stage 2: this CTA's q_b groups, two groups of weights in flight
        q0, qn_ = qb_range(bid)
        # (a group index past QBG loads nothing)
        ops = [s2_loads(c, (j < qn_).select(q0 + j, fx.Int32(QBG))) for j in range(2)]
        if bid < GP0:
            load_qlora(c)
        stamp(on, tls, tid, 3)
        for j in range_constexpr(QB_ROUNDS):
            h = q0 + j
            if j < qn_:
                stage2(c, h, ops[j])
            nwait = 0
            if const_expr(j + 2 < QB_ROUNDS):
                ops.append(s2_loads(c, (j + 2 < qn_).select(h + 2, fx.Int32(QBG))))
                nwait = S2_LOADS
            if j < qn_:
                stage2_flag(c, h, nwait)
            _bar()
        stamp(on, tls, tid, 4)
        # stage 3: tasks bid and bid + 256
        c["xl"], c["wl"] = abl.xl.ptr, abl.wl.ptr
        w3 = [s3_loads(c, bid), s3_loads(c, bid + BLOCKS)]
        stage3(c, bid, w3[0])
        if bid < ABS - BLOCKS:
            stage3(c, bid + BLOCKS, w3[1])
        stamp(on, tls, tid, 5)
        _, _, _, n, _ = request(c)
        if bid < NHG * n:
            attention(c, bid // n, bid % n)
        c["xl"], c["wl"] = ul.xl.ptr, ul.wl.ptr
        if bid < UV_CTAS:
            stage_uv(c, bid)
        stamp(on, tls, tid, 9)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        prefix: Int64, blocks: Int64, blk_sm: Int32, blk_sr: Int32, ares_nw: Int64,
        ares_qk: Int64, in_nw: Int64, w_qkv: Int64, w_g: Int64, q_nw: Int64,
        kv_nw: Int64, w_qb: Int64, w_kc: Int64, w_vc: Int64, kv: Int64, kv_idx: Int64,
        kv_ptr: Int64, g_kv_ptr: Int64, loc: Int64, gate: Int64, send: Int64,
        scratch: Int64, rank: Int32, layer: Int32, epoch: Int64, tl: Int64,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        mla_front(
            prefix, blocks, blk_sm, blk_sr, ares_nw, ares_qk, in_nw, w_qkv, w_g, q_nw,
            kv_nw, w_qb, w_kc, w_vc, kv, kv_idx, kv_ptr, g_kv_ptr, loc, gate, send,
            scratch, rank, layer, epoch, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    _BUILDS[key] = launch
    return launch


def ptr(t):
    return 0 if t is None else t.data_ptr()


WEIGHT_SHAPES = {
    "w_qkv": (LATW, HIDDEN), "w_g": (GATE, HIDDEN), "w_qb": (NHA * QHD, QL),
    "w_kc": (NHA, QN, KV), "w_vc": (NHA, KV, VD),
}  # fmt: skip


def mla_front_launch(
    key: MlaFrontBuild, *, prefix, blocks, w, kv, kv_idx, kv_ptr, g_kv_ptr, loc,
    gate, send, scratch, rank, layer, epoch, tl=None,
):  # fmt: skip
    """Host launcher. ``w``: ares_nw, ares_qk, in_nw, q_nw, kv_nw and the
    ``WEIGHT_SHAPES`` weights (bf16, contiguous); ``kv``: this rank's e4m3 KV
    rows [rows, 576] (row-contiguous); ``kv_idx`` / ``kv_ptr``: the request's
    local rows by position (int32, ragged: SGLang's DCP verify indices);
    ``g_kv_ptr``: its global KV indptr (int32, verify rows included);
    ``loc``: its rows' KV slots (int64);
    ``gate`` [S, 1536] bf16 out; ``send`` [8, S, 12, 130] bf16 out."""
    assert key.write_idx < blocks.shape[1]
    for n, shp in WEIGHT_SHAPES.items():
        t = w[n]
        assert tuple(t.shape) == shp and t.is_contiguous() and t.dtype == torch.bfloat16, (
            n, tuple(t.shape), t.dtype,
        )  # fmt: skip
    assert kv.dtype in (torch.float8_e4m3fn, torch.uint8) and kv.stride(-2) == DQK
    assert kv_idx.dtype == torch.int32 and kv_ptr.dtype == torch.int32
    assert loc.dtype == torch.int64 and g_kv_ptr.dtype == torch.int32
    assert tuple(send.shape) == (DCP, key.tokens, NHL, REC) and send.dtype == torch.bfloat16
    assert tuple(gate.shape) == (key.tokens, GATE) and gate.dtype == torch.bfloat16
    f = build(key)
    f(
        prefix.data_ptr(), blocks.data_ptr(), blocks.stride(0), blocks.stride(1),
        w["ares_nw"].data_ptr(), w["ares_qk"].data_ptr(), w["in_nw"].data_ptr(),
        w["w_qkv"].data_ptr(), w["w_g"].data_ptr(), w["q_nw"].data_ptr(),
        w["kv_nw"].data_ptr(), w["w_qb"].data_ptr(), w["w_kc"].data_ptr(),
        w["w_vc"].data_ptr(), kv.data_ptr(), kv_idx.data_ptr(), kv_ptr.data_ptr(),
        g_kv_ptr.data_ptr(), loc.data_ptr(), gate.data_ptr(), send.data_ptr(),
        scratch.data_ptr(), rank, layer, epoch.data_ptr(), ptr(tl),
        stream=torch.cuda.current_stream(),
    )  # fmt: skip


def scratch_view(key: MlaFrontBuild, scratch, name: str, dtype, shape):
    """A scratch region as a tensor, for the checks."""
    off, nbytes = scratch_layout(key)[name]
    return scratch[off : off + nbytes].view(dtype).view(shape)
