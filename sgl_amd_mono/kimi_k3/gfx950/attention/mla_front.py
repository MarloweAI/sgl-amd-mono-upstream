# SPDX-License-Identifier: Apache-2.0
"""``mla_front``: one Kimi-K3 MLA layer's attention under decode context
parallelism (DCP8), from the AttnRes seam to the V-projected head partials in
the DCP all-to-all's send layout, for the verify blocks of 1 .. 8 requests (6
rows each, S = 6 requests). One launch a layer, ``BLOCKS`` x ``THREADS``.

Every rank attends all 96 heads over its own share of each request's keys (rank
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
    attention, every CTA
                     an item a (request, 16-head group, key split): e4m3 QK
                     and bf16 PV (V transposed in LDS, widened exactly) over
                     64-key blocks, the online softmax, the new token from
                     NEWKV; a partial (f16, over its own sum) and its (max,
                     sum) to PV / ML
    W_UV, CTAs 0 .. 191
                     a (head, 64 value columns) a task: each row's split
                     partials merged in split order to the bf16 latent output,
                     then W_UV -> ``send``, and the head's LSE

Past 8 rows (two requests on) the GEMVs run a 16-row token tile at a time: x
staged into LDS a K segment at a time for stage 1, the whole normed q latent in
LDS for q_b. The MFMAs and their order a row are the same at every width.

Numerics follow SGLang's hand-offs: bf16 x, latent, q_b and absorb outputs, q
and the KV record in e4m3 at scale 1 (the round-to-nearest-even, non-saturating
cast torch uses), the bf16 attention output before W_UV, bf16 partials. The key
split count is a function of the request's own context and every reduction a
row's own, so a row's bits do not depend on the batch. The GEMV split-K and the
attention's summation orders are this kernel's own.
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
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_SYS,
    clock,
    bf_hi,
    bf_lo,
    fp8_to_bf16x8,
    bf16_round,
    global_load,
    hw_exp2,
    hw_rsq,
    load_ptr64,
    row_sum,
    rsrc,
    sfresh64,
    sload64,
    uniform,
    wave_max,
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
from sgl_amd_mono.common import guard
from sgl_amd_mono.common.sync import Addr, Mailbox, Space, publish, sreg
from sgl_amd_mono.kimi_k3.gfx950.attention import mla_combine as mc
from sgl_amd_mono.kimi_k3.gfx950.attention.mla_pre import load_qlora, load_x
from sgl_amd_mono.kimi_k3.gfx950.stages import gemv
from sgl_amd_mono.kimi_k3.gfx950.stages.dims import COLS, HIDDEN, LDS_PAD, NA, ROWS
from sgl_amd_mono.kimi_k3.gfx950.stages.seam import (
    f32_of,
    stage_gate,
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
        "q_b_proj",
        "w_kc",
        "w_vc",
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
# a send record, as SGLang's dcp_pack_a2a_send lays out [8, S, 12, REC]: a
# (row, head)'s 128 bf16 values, then its fp32 LSE
REC = VD + 2
G_LAT = LATW // ROWS  # 132
G_Q = QL // ROWS  # 96
G_KV = KV // ROWS  # 32
S1 = (LATW + GATE) // ROWS  # 228 stage-1 groups
S1_GATE0 = G_LAT
# The CTA roles before the attention (every CTA then takes an attention item):
#   0 .. 55      the seam
#   56 .. 187    a fused_qkv_a group at launch
#   188 .. 255   g_proj, off the q path: a group each, 188 .. 215 a second one;
#                254 then the new KV record
#   0 .. 143     a q_b group of this rank's 12 heads; 144 .. 187 and 216 .. 219
#                an absorb task
#   255          the q all-gather's flags
QKV0 = NA  # the first fused_qkv_a CTA
GP0 = QKV0 + G_LAT  # 188: the first g_proj CTA
GP2 = GATE // ROWS - (BLOCKS - GP0)  # 28 g_proj CTAs with a second group
QBG = NHA * QHD // ROWS  # 1152 q_b groups
QBG_HEAD = QHD // ROWS  # 12 a head: 8 nope, then 4 rope
ABS_BLK = KV // 128  # 4 absorb tasks a head
ABS = NHA * ABS_BLK  # 384
# q is computed for this rank's 12 heads only (the local q_b / W_UK shards)
# and all-gathered: every rank's producers store their fp8 q into every rank's
# QG region; one tagged flag a (source, request) follows
QBG_L = NHL * QBG_HEAD  # 144 local q_b groups: CTAs 0 .. 143, one each
ABS_L = NHL * ABS_BLK  # 48 local absorb tasks: CTAs 144 .. 187, then 216 .. 219
ABS0 = QBG_L
ABS_N0 = 188 - ABS0  # 44 tasks on qkv_a CTAs; the rest past the two-group g_proj CTAs
ABS1 = 216
QG_FLAG_CTA = 255  # raises the (source, request) flags once the local q landed
# heads an attention item: 16 (96 (row, head) columns) past one request, 8 (48)
# for one (its 12 head groups x splits fill the CTAs); a column's arithmetic is
# the same either way. An item's 16-column MFMA tiles: tile T = row pair T % 3
# x head octet T // 3.


def heads_of(s: int) -> int:
    return 8 if reqs_of(s) == 1 else 16
BK = 64  # keys a block
KT = BK // 16  # 16-key QK tiles a block
KC = BK // 32  # 32-key PV steps a block
RWK = DQK // 4 + 4  # i32 words a key in LDS: its record + 16 bytes of pad
PW = BK // 2 + 4  # i32 words a (row, head)'s bf16 P in LDS: its keys + pad
ATT_CTAS = BLOCKS  # one request's items fit the CTAs (one each)
MAX_SPLITS = 21  # the most splits of a request (its 12 head groups' items fit the CTAs)
KEYS_PER_SPLIT = 64  # local keys a split aims at (one block)
UV_HALF = 64  # value columns a W_UV task
UV_TASKS = NHA * (VD // UV_HALF)  # 192
UV_CTAS = 192  # a W_UV task a CTA, on CTAs 0 .. 191, after every CTA's share of the merge
UV_ROW = UV_HALF + 8  # bf16 a W_UV tile row in LDS
ABS_ROW = 128 + 8  # bf16 an absorb tile row in LDS
OL_ROW = KV + 8  # bf16 a latent output row in LDS
KV_CTA = QG_FLAG_CTA - 1  # 254: the new KV record, after its g_proj group (no q path work)
NEG = -3.4028234663852886e38
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
EMPTY_LSE = -1e30  # SGLang's LSE of a row with no key on this rank
MAX_REQS = 8
WIDE_ABOVE = 8  # wider steps run their GEMVs a 16-row token tile at a time
RSTD_W = 2048 - 64  # red words of the rows' rstd (q / kv norms)
# start, x seen, stage 1, q latent normed, q_b, absorb, attention q seen,
# attention done, W_UV partials seen, W_UV done
# a timeline build's attention step accounting: wave w's time in phase k of a
# step (QK + softmax, transpose, P V, the DMA wait and the step barrier) at
# point TL_ROLE + TL_PHASES w + k; the steps at TL_STEPS
TL_ROLE = 19
TL_PHASES = 4
TL_STEPS = TL_ROLE + TL_PHASES * WAVES
TL_POINTS = TL_STEPS + 1

assert GP2 >= 0 and ABS1 >= 188 + GP2 and ABS1 + ABS_L - ABS_N0 <= QG_FLAG_CTA


QG_REC = DQK  # bytes a (source, request, local head, row) q record
QG_DATA = DCP * MAX_REQS * NHL * G_ROWS * QG_REC  # one copy's records
QG_COPY = -(-(QG_DATA + DCP * MAX_REQS * 8) // 4096) * 4096  # records, then the flag pairs
QG_RESERVE = 4 << 20  # room past K2's region for the DCP all-to-all's A2A copies (mla_combine)


def qg_offset() -> int:
    """Where a rank's QG region starts in its peer buffer."""
    from sgl_amd_mono.kimi_k3.gfx950 import layer

    return -(-layer.peer_alloc_bytes() // 4096) * 4096 + QG_RESERVE


def _flags_offset() -> int:
    """Where every launch reads the guard flags in a rank's peer buffer (K2's)."""
    from sgl_amd_mono.kimi_k3.gfx950 import layer

    return layer.flags_offset()


assert mc.a2a_copy_bytes() * 2 <= QG_RESERVE


def peer_end() -> int:
    """A rank's peer buffer with the QG region's two copies.

    Consecutive fronts take alternate copies, so a rank writes a copy again two
    fronts later. By then every peer has finished the front that read it: a
    front ends only once every source's flags of its own launch came (CTA
    ``QG_FLAG_CTA``), so each peer had started that front, after the one before.
    That needs an even number of fronts a step (K3: 24 MLA layers)."""
    return qg_offset() + 2 * QG_COPY


def qg_word(src, rq, hl, t, byte):
    """i32 word of (source rank, request, local head, row, byte) in a QG copy."""
    return ((((src * MAX_REQS + rq) * NHL + hl) * G_ROWS + t) * QG_REC + byte) // 4


def splits_of(p: int) -> int:
    """Key splits of a request at committed length ``p``: a function of the
    request's own (global) context only, the same on every rank."""
    lg = -(-(p + G_ROWS) // DCP)
    return max(1, min(MAX_SPLITS, -(-lg // KEYS_PER_SPLIT)))


def wide(s: int) -> bool:
    return s > WIDE_ABOVE


def reqs_of(s: int) -> int:
    assert s % G_ROWS == 0 and 0 < s // G_ROWS <= MAX_REQS, s
    return s // G_ROWS


assert (NHA // 8) * MAX_SPLITS <= ATT_CTAS


@dataclass(frozen=True)
class MlaFrontBuild:
    tokens: int  # S: the requests' verify rows (6 a request)
    nblocks: int  # AttnRes block sources (the prefix is one more)
    scale: float  # the softmax scale
    eps: float = 1e-5  # AttnRes norm
    out_eps: float = 1e-5  # input_layernorm
    q_eps: float = 1e-6  # q_a_layernorm
    kv_eps: float = 1e-6  # kv_a_layernorm
    write_idx: int = -1  # a block-write layer's block: the updated prefix stored there
    # the QG copy: the parity of the layer's ordinal among the step's MLA fronts
    qg_par: int = 0
    # the attention through W_UV: this kernel's own ("f") or the decode core's
    # (``mla_decode_c64``, "c64"), whose counters a layer are the MLA ordinal's
    core: str = "f"
    mla_ord: int = 0
    timeline: bool = False
    # checks only: the W_UV records also to ``send`` in SGLang's all-to-all
    # layout [8, S, 12, 130]
    records: bool = False
    # the x rows given (``prefix``: SGLang's input_layernorm output, its own
    # seam done): no seam roles, stage 1 reads them in place
    x_in: bool = False


def scratch_layout(key: MlaFrontBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    nreq = reqs_of(s)
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA,
        "ss": G_LAT * s,
        "qrdy": QBG,
        "absrdy": ABS,
        "kvrdy": nreq,
        "itrdy": (NHA // heads_of(s)) * MAX_SPLITS * nreq,
    }
    if wide(s):
        pairs["olrdy"] = BLOCKS
        pairs["deal"] = DEAL_WORDS + 2
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
    add("newkv", nreq * DQK)
    add("pv", MAX_SPLITS * s * NHA * KV * 2)
    add("ml", MAX_SPLITS * s * NHA * 2 * 4)
    add("g", s * GATE * 2)
    add("grdy", mc.GATE_GROUPS * 8)
    if wide(s):
        add("ol", s * NHA * KV * 2)
        add("olse", s * NHA * 4)
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


def qg_base(c, p):
    """Rank p's QG copy (the launch's parity): its base, loaded where used."""
    return sload64(sfresh64(c["args"]["peers"]) + fx.Int64(p) * fx.Int64(8)) + c["qg_off"]


class QgCopies:
    """Every rank's QG copy by rank (``[p]``: its base), the records', or with
    ``flags`` its flag pairs' (an ``Addr``)."""

    def __init__(self, c, flags=False):
        self.c, self.flags = c, flags

    def __getitem__(self, p):
        base = qg_base(self.c, p)
        return Addr("qgf", Space.PEER, base + fx.Int64(QG_DATA)) if self.flags else base


def put_q(base, w, words):
    bo.buffer_store(words, rsrc(base), w, cache_modifier=CM_SYS)


def push_q(c, t, head, byte, words):
    """One piece of row t's q of (global) head ``head`` -- ``words`` (i32) at
    ``byte`` of its 576 -- into every rank's QG copy, system scope, at this
    rank's (source) slot."""
    peer = QgCopies(c)
    for p in range_constexpr(DCP):
        put_q(peer[p], qg_word(c["rank"], t // G_ROWS, head % NHL, t % G_ROWS, byte), words)


def _st_bf(ptr, i, v, cm=0):
    bo.buffer_store(fx.Float32(v).to(fx.BFloat16), rsrc(ptr), i, cache_modifier=cm)


E4M3_MAX = 448.0


def _sat(x):
    """x within +-448, e4m3's largest finite (+-inf to it too); NaN stays NaN."""
    x = fx.Float32(x)
    return (x > fx.Float32(E4M3_MAX)).select(
        fx.Float32(E4M3_MAX), (x < fx.Float32(-E4M3_MAX)).select(fx.Float32(-E4M3_MAX), x)
    )


def e4m3x2(a, b, old, hi):
    """Two f32 -> e4m3 bytes into half ``hi`` of ``old``: saturating, round to nearest
    even (v_cvt_pk_fp8_f32 on the value within +-448), as B300's cvt.rn.satfinite;
    torch's float8_e4m3fn cast of the bf16 value wherever |x| <= 448."""
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, _sat(a), _sat(b), old, hi))


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


def _lds16(ptr):
    """16 bytes of LDS at a 16-byte aligned ``ptr`` (the alignment stated: the
    compiler does not always infer it inside nested loops)."""
    return fx.Vector(llvm.load(T.vec(4, T.i32), ptr.llvm_ptr, alignment=16))


def _mfma16(a, b, c):
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


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


def _tgemv_cols(lane, wl, wrow, col, k0, step):
    """``_tgemv_ops`` of a tile held column major (column c a row of ``wrow``
    bf16 at ``wl``, K contiguous): the same operands, a 16-byte read each."""
    g = lane // 16
    kb = k0 + 32 * step + 8 * g
    a = _lds16(wl + (col * wrow + kb) // 2)
    b = _lds16(wl + ((col + 1) * wrow + kb) // 2)
    return a.bitcast(fx.BFloat16), b.bitcast(fx.BFloat16)


def _tgemv(lane, wl, wrow, col0, k0, steps, xl, xrow, s, cols_major=False):
    """Columns col0 + 2 (lane % 16) + dt (dt = 0, 1) of X W over K = k0 ..
    k0 + 32 steps, X the token rows in LDS (``xrow`` bf16 a row): two 16x16
    accumulators C[col][token] (lane: columns col0 + 2 (4 (lane / 16) + i) + dt,
    token lane % 16)."""
    g, r16 = lane // 16, lane % 16
    t = fx.min(r16, s - 1)
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for st in range_constexpr(steps):
        if const_expr(cols_major):
            a0, a1 = _tgemv_cols(lane, wl, wrow, col0 + 2 * r16, k0, st)
        else:
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
    """Rows 16 g .. of stage 1 against the LDS x (every row): ``s1_out``."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    acc = gemv.mfmas(lane, wave, c["xl"], HIDDEN + LDS_PAD, HIDDEN, s, ops)
    s1_out(c, g, acc, 0, s)


@flyc.jit
def stage1_wide(c, g, ops):
    """``stage1`` a 16-row token tile at a time, x into LDS a round (1024
    columns, every wave's next chunk) at a time through two buffers: a round's
    loads are in flight while the round before's MFMAs run (``mfmas_rounds``:
    the same MFMAs in the same order)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rk = WAVES * gemv.KCH
    nr = HIDDEN // rk
    srow = rk + LDS_PAD
    bufs = [c["xl"], c["xl"] + ROWS * srow // 2]
    rounds = [(t0, sj, r) for t0, sj in gemv.tiles(s) for r in range(nr)]

    def load(j):
        t0, sj, r = rounds[j]
        return gemv.rows_load(tid, c["xbuf"], rk, sj, CM_DEV, ldk=HIDDEN, k0=r * rk, t0=t0)

    vals = load(0)
    gemv.rows_store(tid, vals, rk, rounds[0][1], bufs[0], srow)
    gpu.barrier()
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for j in range_constexpr(len(rounds)):
        t0, sj, r = rounds[j]
        if const_expr(j + 1 < len(rounds)):
            vals = load(j + 1)
        acc = gemv.mfmas_rounds(lane, wave, bufs[j % 2], srow, r * rk, sj, ops, acc, r, r + 1)
        if const_expr(r == nr - 1):
            s1_out(c, g, acc, t0, sj)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
        if const_expr(j + 1 < len(rounds)):
            gemv.rows_store(tid, vals, rk, rounds[j + 1][1], bufs[(j + 1) % 2], srow)
            gpu.barrier()


@flyc.jit
def s1_out(c, g, acc, t0, sj):
    """Stage-1 group g's rows for tokens t0 .. t0 + sj: a latent group to LAT
    (and its rows' squares to SS: the q / kv norms read them, and the rope
    groups' flag the KV record's wait), a g_proj group to ``gate``."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    mine = tid < ROWS * sj
    r = tid % ROWS
    t = fx.min(tid // ROWS, sj - 1)
    v = bf16_round(row_sum(red, r, t))
    is_lat = g < S1_GATE0
    if mine & is_lat:
        _st_bf(c["lat"], (t0 + t) * LATW + g * ROWS + r, v, CM_DEV)
    if mine & (g >= S1_GATE0):
        _st_bf(a["gate"], (t0 + t) * GATE + (g - S1_GATE0) * ROWS + r, v, CM_DEV)
    gpu.barrier()
    if mine:
        fx.ptr_store(v * v, red + tid)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if (tid < sj) & (g < G_LAT):
        tot = fx.ptr_load(red + tid * ROWS)
        for rr in range_constexpr(1, ROWS):
            tot = tot + fx.ptr_load(red + (tid * ROWS + rr))
        c["put"](c["ss"], g * s + t0 + tid, tot)
    gpu.barrier()


@flyc.jit
def x_seen(c):
    """Every slice's x ready (the wide stage 1 stages it itself); given x
    rows are ready at launch."""
    tid = c["tid"]
    if const_expr(not c["x_in"]):
        if tid < NA:
            c["poll"]([(c["xrdy"], tid, 1)])
        gpu.barrier()
    stamp(c["on"], c["tls"], tid, 1)


@flyc.jit
def load_qlora_wide(c):
    """``load_qlora`` past 8 rows: a row's 96 group squares summed in group
    order (as ``load_qlora``) 16 rows at a time through LDS, then every row's
    normed q latent into LDS (``xl``, rows padded). The norm weights are read
    before the squares are polled, and the latent rows right after (every group
    landed), so their loads overlap the squares' sums."""
    s, tid, red = c["S"], c["tid"], c["red"]
    a = c["args"]
    tiles = gemv.tiles(s)

    def quads_of(ti):
        t0, sj = tiles[ti]
        quads = sj * QL // 8  # 4 words (8 bf16) a lane
        out = []
        for i in range_constexpr(-(-quads // THREADS)):
            e = fx.min(tid + THREADS * i, quads - 1)
            out.append((t0 + e // (QL // 8), e % (QL // 8) * 8, tid + THREADS * i < quads))
        return out

    qs = [quads_of(ti) for ti in range(len(tiles))]
    nws = [[_ld4(a["q_nw"], k // 2) for _, k, _ in q] for q in qs]
    n = G_Q * s  # (row, group) items, row-major: 16 rows a 3 rounds
    per = -(-n // THREADS)
    es = [fx.min(tid + THREADS * i, n - 1) for i in range(per)]
    got = [w[0] for w in c["poll"]([(c["ss"], (e % G_Q) * s + e // G_Q, 1) for e in es])]
    lws = [[_ld4(c["lat"], (t * LATW + k) // 2, CM_DEV) for t, k, _ in q] for q in qs]
    rpad = G_Q + 1
    for ti in range_constexpr(len(tiles)):
        t0, sj = tiles[ti]
        for i in range_constexpr(3 * ti, min(3 * ti + 3, per)):
            e = tid + THREADS * i
            if e < n:
                fx.ptr_store(f32_of(got[i]), red + ((e // G_Q - t0) * rpad + e % G_Q))
        gpu.barrier()
        if tid < sj:
            tot = fx.ptr_load(red + tid * rpad)
            for gg in range_constexpr(1, G_Q):
                tot = tot + fx.ptr_load(red + (tid * rpad + gg))
            fx.ptr_store(hw_rsq(tot / fx.Float32(QL) + fx.Float32(c["q_eps"])), red + (RSTD_W + t0 + tid))
        gpu.barrier()
    xrow = QL + LDS_PAD
    for ti in range_constexpr(len(tiles)):
        for i in range_constexpr(len(qs[ti])):
            t, k, live = qs[ti][i]
            rs = fx.ptr_load(red + (RSTD_W + t))
            out = []
            for d in range_constexpr(4):
                w, nw = fx.Int32(lws[ti][i][d]), fx.Int32(nws[ti][i][d])
                y0 = bf16_round((bf_lo(w) * rs) * bf_lo(nw))
                y1 = bf16_round((bf_hi(w) * rs) * bf_hi(nw))
                out.append(fx.Vector.from_elements([y0, y1], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0])
            if live:
                fx.ptr_store(fx.Vector.from_elements(out, fx.Int32), c["xl"] + (t * xrow + k) // 2)
    gpu.barrier()


# ---------------------------------------------------------------- the request


def request(c, r=0):
    """Request r's global committed length P (its global KV indptr span less
    the verify rows) and this rank's view of it: (P, local prefix keys L, the
    verify row this rank owns or 8, splits n, its first local index)."""
    a = c["args"]
    p = uniform(_ld(a["g_kv_ptr"], r + 1) - _ld(a["g_kv_ptr"], r)) - G_ROWS
    rank = c["rank"]
    pm = p & (DCP - 1)
    lr = (p >> 3) + (rank < pm).select(fx.Int32(1), fx.Int32(0))
    own = (rank - pm + DCP) & (DCP - 1)  # the row at position P + own
    own = (own < G_ROWS).select(own, fx.Int32(DCP))
    lg = (p + G_ROWS + DCP - 1) >> 3
    n = fx.max(fx.Int32(1), fx.min(fx.Int32(MAX_SPLITS), (lg + KEYS_PER_SPLIT - 1) // KEYS_PER_SPLIT))
    return p, lr, own, n, uniform(_ld(a["kv_ptr"], r))


def blocks_of(lr, own):
    """A request's key blocks on this rank: its local keys and new token, at
    least one (a rank with no key of the request, P < 2, runs one fully masked
    block: an empty partial, LSE -1e30 as SGLang's)."""
    return fx.max((lr + (own < G_ROWS).select(fx.Int32(1), fx.Int32(0)) + BK - 1) // BK, fx.Int32(1))


def item_bases(c):
    """Each request's (first attention item, splits), its items (head group
    hg, split sp) at first + hg n + sp; and the items in all."""
    HQ, NCT, NHG = c["HQ"], c["NCT"], c["NHG"]
    out = []
    base = fx.Int32(0)
    for r in range(c["NREQ"]):
        n = request(c, r)[3]
        out.append((base, n))
        base = base + NHG * n
    return out, base


def pick(vals, r):
    """vals[r] for a traced r (a Python r: vals[r])."""
    if isinstance(r, int):
        return vals[r]
    v = vals[0]
    for j in range(1, len(vals)):
        v = (r == j).select(vals[j], v)
    return v


# ---------------------------------------------------------------- the KV record


@flyc.jit
def stage_kv(c):
    """kv_a_layernorm of the latent, then the e4m3 record (normed latent, raw
    rope part) of each request's new token this rank owns: its KV row and the
    request's NEWKV."""
    s, tid, red = c["S"], c["tid"], c["red"]
    a = c["args"]
    # the kv groups' squares, and the rope groups' flags (the record carries their rows)
    n = (G_LAT - G_Q) * s
    per = -(-n // THREADS)
    got = c["poll"]([(c["ss"], G_Q * s + fx.min(tid + THREADS * i, n - 1), 1) for i in range(per)])
    for i in range_constexpr(per):
        if tid + THREADS * i < G_KV * s:
            fx.ptr_store(f32_of(got[i][0]), red + (tid + THREADS * i))
    gpu.barrier()
    if tid < s:
        tot = fx.ptr_load(red + tid)
        for gg in range_constexpr(1, G_KV):
            tot = tot + fx.ptr_load(red + (gg * s + tid))
        fx.ptr_store(
            hw_rsq(tot / fx.Float32(KV) + fx.Float32(c["kv_eps"])),
            red + (RSTD_W + tid),
        )
    gpu.barrier()
    words = DQK // 4  # 144 record words, 4 dims each
    for r in range_constexpr(c["NREQ"]):
        _, _, own, _, _ = request(c, r)
        t = G_ROWS * r + fx.min(own, G_ROWS - 1)
        owned = own < G_ROWS
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
                rs = fx.ptr_load(red + (RSTD_W + t))
                ws = []
                for h in range_constexpr(2):
                    w = fx.Int32(nw[h])
                    ws += [(w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)]
                vals = [bf16_round((vals[j] * rs) * ws[j]) for j in range(4)]
            word = e4m3x4(vals)
            row = fx.Int32(fx.Int64(_ld(a["loc"], t * c["loc_words"])) >> 3)
            # device scope (as NEWKV): the decode core reads this row in this launch, on any XCD, once KVRDY is up
            bo.buffer_store(word, rsrc(a["kv"]), row * (DQK // 4) + tid, cache_modifier=CM_DEV)
            bo.buffer_store(word, rsrc(c["newkv"]), r * words + tid, cache_modifier=CM_DEV)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid < c["NREQ"]:
        c["put"](c["kvrdy"], tid, fx.Int32(1))
    gpu.barrier()


# ---------------------------------------------------------------- stage 2: q_b


def s2_loads(c, h):
    """q_b group ``h``'s weights (past the last group: zero-range loads)."""
    a = c["args"]
    # the local q_b shard's rows: group h of head 12 rank + hl is its row 16 (h - 144 rank)
    hl = h - QBG_L * c["rank"]
    nbytes = ((hl >= 0) & (hl < QBG_L)).select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
    hl = fx.max(fx.min(hl, QBG_L - 1), 0)
    return gemv.loads(c["lane"], c["wave"], a["w_qb"], hl * ROWS, QL, nbytes=nbytes)


S2_LOADS = 4 * gemv.chunks(QL)[0]  # a wave's loads of one q_b group


@flyc.jit
def stage2(c, h, ops):
    """Rows 16 h .. of q_b against the normed q latent in LDS: bf16 to QB; a rope
    group also as e4m3 into Q8's last 64 dims (``stage2_flag`` publishes), a
    16-row token tile at a time."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    tiles = gemv.tiles(s)
    for ti in range_constexpr(len(tiles)):
        t0, sj = tiles[ti]
        acc = gemv.mfmas(lane, wave, c["xl"], QL + LDS_PAD, QL, sj, ops, t0)
        s2_out(c, h, acc, t0, sj)
        if const_expr(ti + 1 < len(tiles)):
            _bar()


@flyc.jit
def s2_out(c, h, acc, t0, sj):
    """q_b group h's rows for tokens t0 .. t0 + sj: bf16 to QB, a rope group's
    as e4m3 into Q8."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    _bar()
    mine = tid < ROWS * sj
    r = tid % ROWS
    t = t0 + fx.min(tid // ROWS, sj - 1)
    v = bf16_round(row_sum(red, r, fx.min(tid // ROWS, sj - 1)))
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
        word = e4m3x4([v] + nb)
        bo.buffer_store(word, rsrc(c["q8"]), (t * (NHA * DQK) + head * DQK + KV + d) // 4, cache_modifier=CM_DEV)
        push_q(c, t, head, KV + d, word)


@flyc.jit
def stage2_flag(c, h, nwait):
    """Group h's rows landed (the newest ``nwait`` loads aside), then its QRDY."""
    _published(nwait)
    if c["tid"] == 0:
        c["put"](c["qrdy"], h, fx.Int32(1))


# ---------------------------------------------------------------- stage 3: absorb


def s3_loads(c, task):
    """Absorb task ``task``'s W_UK tile, W[h][n][128 b ..+128] of the local shard
    as SGLang holds it (``W_KC_STRIDES``: n innermost), 16 bytes (8 n) a piece
    (4 pieces a thread, column k = piece / 16)."""
    a = c["args"]
    tid = c["tid"]
    task = fx.min(task, ABS - 1)
    h = task // ABS_BLK - NHL * c["rank"]  # the local W_UK shard's head
    h = fx.max(fx.min(h, NHL - 1), 0)
    b = task % ABS_BLK
    out = []
    for i in range_constexpr(4):
        p = tid + THREADS * i
        k = p // 16
        seg = p % 16
        out.append(_ld4(a["w_kc"], (h * (QN * KV) + (b * 128 + k) * QN + seg * 8) // 2))
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
    xrow = QN + LDS_PAD
    # q_nope rows: S x 128 bf16, 16 bytes a thread
    for j in range_constexpr(-(-s * (QN // 8) // THREADS)):
        e = tid + THREADS * j
        if e < s * (QN // 8):
            t = e // (QN // 8)
            k = (e % (QN // 8)) * 8
            v = _ld4(c["qb"], (t * (NHA * QHD) + h * QHD + k) // 2, CM_DEV)
            fx.ptr_store(v, xl + (t * xrow + k) // 2)
    for i in range_constexpr(4):
        p = tid + THREADS * i
        kr = p // 16
        seg = p % 16
        fx.ptr_store(wt[i], wl + (kr * ABS_ROW + seg * 8) // 2)  # tile row k, n 8 seg ..
    gpu.barrier()
    cq = wave % 4
    kh = wave // 4
    g, r16 = lane // 16, lane % 16
    tiles = gemv.tiles(s)
    for ti in range_constexpr(len(tiles)):
        t0, sj = tiles[ti]
        acc = _tgemv(lane, wl, ABS_ROW, 32 * cq, 64 * kh, 2, xl + t0 * (xrow // 2), xrow, sj, cols_major=True)
        # the two K halves summed in order: wave w + 4 adds onto wave w's
        rb = red + (wave * 64 + lane) * 8
        for dt in range_constexpr(2):
            fx.ptr_store(acc[dt], rb + 4 * dt)
        gpu.barrier()
        if (wave < 4) & (r16 < sj):
            lo = red + (wave * 64 + lane) * 8
            hi = red + ((wave + 4) * 64 + lane) * 8
            vals = [None] * 8
            for dt in range_constexpr(2):
                for i in range_constexpr(4):
                    vals[2 * i + dt] = bf16_round(fx.ptr_load(lo + (4 * dt + i)) + fx.ptr_load(hi + (4 * dt + i)))
            w0 = e4m3x4(vals[0:4])
            w1 = e4m3x4(vals[4:8])
            col = b * 128 + 32 * cq + 8 * g
            words = fx.Vector.from_elements([w0, w1], fx.Int32)
            bo.buffer_store(words, rsrc(c["q8"]), ((t0 + r16) * (NHA * DQK) + h * DQK + col) // 4, cache_modifier=CM_DEV)
            push_q(c, t0 + r16, h, col, words)
        if const_expr(ti + 1 < len(tiles)):
            _bar()
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](c["absrdy"], task, fx.Int32(1))
    gpu.barrier()


@flyc.jit
def qg_flags(c):
    """This rank's q landed in every rank's QG copy (its 48 absorb tasks, its
    48 rope groups): the (this source, request) flag pairs into every rank's."""
    tid = c["tid"]
    src = c["rank"]
    if tid < ABS_L:
        c["poll"]([(c["absrdy"], ABS_L * src + tid, 1)])
    if (tid >= ABS_L) & (tid < 2 * ABS_L):
        e = tid - ABS_L
        c["poll"]([(c["qrdy"], (NHL * src + e // 4) * QBG_HEAD + QN // ROWS + e % 4, 1)])
    gpu.barrier()
    if tid < c["NREQ"]:
        flags = QgCopies(c, flags=True)
        for p in range_constexpr(DCP):
            c["put"](flags[p], c["rank"] * MAX_REQS + tid, fx.Int32(1))
    rocdl.s_waitcnt(vmcnt=0)
    # every source's flags of this launch, whichever heads this rank's items read
    if (tid < DCP * MAX_REQS) & (tid % MAX_REQS < c["NREQ"]):
        c["poll"]([(QgCopies(c, flags=True)[c["rank"]], tid, 1)])
    gpu.barrier()


@flyc.jit
def qg_wait(c, rq, h0, nh):
    """Heads h0 .. h0 + nh of request rq: their sources' flags in this rank's
    QG copy (at most two sources)."""
    tid = c["tid"]
    s0 = h0 // NHL
    s1 = (h0 + nh - 1) // NHL
    if tid < 2:
        src = (tid == 0).select(s0, s1)
        c["poll"]([(Addr("qgf", Space.PEER, qg_base(c, c["rank"]) + fx.Int64(QG_DATA)), src * MAX_REQS + rq, 1)])


@flyc.jit
def qg_wait_wave(c, rq, h0, nh):
    """``qg_wait`` for the calling wave alone: heads h0 .. h0 + nh - 1 span
    sources s0 .. s1 (up to 4 at nh = 32), lane i polling source s0 + i, so
    the wave's later loads follow every one of them and no barrier is needed
    (the decode core's hooks run on some of a CTA's waves)."""
    lane = c["tid"] % 64
    s0 = h0 // NHL
    s1 = (h0 + nh - 1) // NHL
    if lane <= s1 - s0:
        src = s0 + lane
        c["poll"]([(Addr("qgf", Space.PEER, qg_base(c, c["rank"]) + fx.Int64(QG_DATA)), src * MAX_REQS + rq, 1)])


@flyc.jit
def kvrdy_wave(c, rq):
    """Request rq's KVRDY for the calling wave alone (its lane 0 polls)."""
    kvrdy = c["kvrdy"]
    if c["tid"] % 64 == 0:
        c["poll"]([(kvrdy, rq, 1)])


def core_hooks(c):
    """The decode core's hooks inside this launch (``mla_decode_c64``): q from
    this rank's QG copy; at a piece's start the QG flags of its heads' sources;
    before its request's last block KVRDY (``stage_kv`` wrote the new tokens'
    pool rows); its records into the owners' all-to-all cells, rounded to bf16
    there once (``mla_combine``). Every wait is the calling wave's own."""

    def q_load(rq, head, t, byte):
        return qg_load4(c, rq, head, t, byte)

    def ready(rq, h0, nh):
        qg_wait_wave(c, rq, h0, nh)

    def kv_ready(rq):
        kvrdy_wave(c, rq)

    def emit_values(owner, hl, t, col, vals):
        mc.push_values(c, owner, hl, t, col, vals)

    def emit_lse(owner, hl, t, lse):
        mc.push_lse(c, owner, hl, t, lse)

    return {"q_load": q_load, "ready": ready, "kv_ready": kv_ready, "emit_values": emit_values, "emit_lse": emit_lse}

def qg_load4(c, rq, head, t, byte):
    """16 bytes of row t's q of head ``head`` (request rq) from this rank's QG copy."""
    w = qg_word(head // NHL, rq, head % NHL, t, byte)
    return fx.Vector(bo.buffer_load(rsrc(qg_base(c, c["rank"])), w, vec_width=4, dtype=T.i32, cache_modifier=CM_SYS))


# ---------------------------------------------------------------- attention




KV_SLOTS = RWK * 4 // 16  # 37 16-byte LDS slots a key: its 36 pieces, then pad
DMA_N = BK * KV_SLOTS // 64  # 37 wave-wide DMAs a block
PATCH_W = WAVES - 1  # the wave that patches a block's new token
PV_DIMS = KV // WAVES  # 64 value dims a wave in P V
KV_BUFS = 2  # LDS record buffers: block i in QK / P V, i + 1 in flight
VTW = BK // 4 + 2  # i32 words a value dim's row of the transposed block: its keys + pad
assert BK * KV_SLOTS % 64 == 0 and KV * VTW <= BK * RWK


def _kv_rows(c, kvp, lr, b1, blk):
    """Lane l's key of block ``blk``: its record's byte offset (a key past the
    request's keys or the run: row 0, a valid dummy)."""
    a = c["args"]
    kk = blk * BK + c["lane"]
    # a request with no local key has no index to read (an empty range)
    r_idx = rsrc(a["kv_idx"], (lr > 0).select(fx.Int64(0xFFFFFFFF), fx.Int64(0)))
    row = fx.Int32(bo.buffer_load(r_idx, kvp + fx.max(fx.min(kk, lr - 1), 0), vec_width=1, dtype=T.i32))
    return ((kk < lr) & (blk < b1)).select(row * DQK, fx.Int32(0))


DMA_SHARE = -(-DMA_N // WAVES)  # 5 of a block's wave-wide DMAs a wave


@flyc.jit
def _kv_dma(c, rows, kvl):
    """A block's records straight into LDS at ``kvl`` (padded rows), a 16-byte
    slot a lane: DMA i fills 64 consecutive slots, wave w issuing DMAs w, w +
    8, .. (``rows``: ``_kv_rows`` of the block). The pad slots and the new
    token's (patched after, ``_kv_patch``) take a valid dummy row."""
    a = c["args"]
    lane, wave = c["lane"], c["wave"]
    r_kv = rsrc(a["kv"])
    for j in range_constexpr(DMA_SHARE):
        i = wave + WAVES * j
        q = i * 64 + lane
        key = q // KV_SLOTS
        part = q % KV_SLOTS
        off = fx.Int32(rocdl.ds_bpermute(T.i32, fx.min(key, BK - 1) * 4, rows))
        voff = (part < KV_SLOTS - 1).select(off + part * 16, fx.Int32(0))
        if i < DMA_N:
            rocdl.raw_ptr_buffer_load_lds(r_kv, (kvl + i * 256).llvm_ptr, 16, voff, fx.Int32(0), 0, 0)


@flyc.jit
def _kv_patch(c, rq, lr, own, blk, kvl):
    """Request rq's new token's record from its NEWKV into its slot, where block
    ``blk`` holds it."""
    e = c["tid"] - PATCH_W * 64  # a thread of the patching wave
    if (own < G_ROWS) & (lr // BK == blk) & (e >= 0) & (e < DQK // 16):
        v = _ld4(c["newkv"], rq * (DQK // 4) + 4 * e, CM_DEV)
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
    """The block's scores against a 16-column tile (``ql``: its e4m3 q in LDS, a
    (row, head) a key-sized row, or its operands a K step in registers): KT
    16-key tiles C[key][rh] (element i of tile kt:
    key 16 kt + 4 (lane / 16) + i; rh = lane % 16: row 2 rp + rh / 8, head rh % 8)."""
    g, r16 = lane // 16, lane % 16
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(KT)]

    def row32(base, kc):
        # past the record the reads stay in the row (its pad, then the next row)
        if const_expr(128 * (kc + 1) <= DQK):
            w = base + (32 * kc + 8 * g)
        else:
            w = base + fx.min(32 * kc + 8 * g, DQK // 4 - 8)
        lo = _lds16(w)
        hi = _lds16(w + 4)
        return _k128(lane, fx.Vector.from_elements([lo[e] for e in range(4)] + [hi[e] for e in range(4)], fx.Int32), kc)

    for kc in range_constexpr(QK_STEPS):
        qo = ql[kc] if isinstance(ql, list) else row32(ql + r16 * RWK, kc)
        for kt in range_constexpr(KT):
            acc[kt] = _mfma128(row32(kvl + (16 * kt + r16) * RWK, kc), qo, acc[kt])
    return acc


def _softmax(lane, t, blk, kend, sacc, lr, own, m, l, qk_scale, prow):
    """(row t, head) lane % 16's online softmax over the block: the new (max,
    sum, factor of the earlier sums); its P (bf16, keys the row does not see 0)
    to the (row, head)'s LDS."""
    g = lane // 16
    # the keys the row sees: below L, the new token (key L) from row own on,
    # none past the run
    lim = fx.min(lr + (own <= t).select(fx.Int32(1), fx.Int32(0)), kend) - (blk * BK + 4 * g)
    sv = []
    for kt in range_constexpr(KT):
        for i in range_constexpr(4):
            sv.append((16 * kt + i < lim).select(sacc[kt][i] * qk_scale, fx.Float32(NEG)))
    mb = sv[0]
    for e in range_constexpr(1, 4 * KT):
        mb = fx.max(mb, sv[e])
    # xred reduces an f32 as f32 only when it is typed so
    mb = xred(xred(fx.Float32(mb), 16, fx.max), 32, fx.max)
    mn = fx.max(m, mb)
    a = hw_exp2(m - mn)
    # a masked key's 2^(NEG - mn) is 0 for any mn >= -1e30; a row with every key
    # masked so far exponentiates against -1e30
    me = fx.max(mn, fx.Float32(-1e30))
    ls = fx.Float32(0.0)
    for kt in range_constexpr(KT):
        ps = [hw_exp2(sv[4 * kt + i] - me) for i in range(4)]
        ls = ls + ((ps[0] + ps[1]) + (ps[2] + ps[3]))
        pw = fx.Vector.from_elements(ps, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        fx.ptr_store(pw, prow + (8 * kt + 2 * g))
    ls = xred(xred(fx.Float32(ls), 16, lambda x, y: x + y), 32, lambda x, y: x + y)
    return mn, l * a + ls, a


def _perm(hi, lo, sel):
    """v_perm_b32: byte k of the result is byte sel_k of (hi:lo) (0 .. 3 lo's)."""
    return fx.Int32(
        llvm.call_intrinsic(
            T.i32, "llvm.amdgcn.perm", [as_ir_value(hi), as_ir_value(lo), as_ir_value(fx.Int32(sel))], [], []
        )
    )


def _byte_q(w, q):
    """Byte q of w[0] .. w[3] -> one word (w[j]'s at byte j)."""
    sel = q | ((4 + q) << 8) | (q << 16) | ((4 + q) << 24)
    x = _perm(w[1], w[0], sel)
    y = _perm(w[3], w[2], sel)
    return _perm(y, x, 0 | (1 << 8) | (4 << 16) | (5 << 24))


VT_THREADS = 2 * 64  # the transposing waves' threads: a 4-dim word each
VT_PASS = 16  # keys a pass


@flyc.jit
def _transpose_v(c, kx, vt):
    """The block's value part (its 512 latent dims) transposed, dim major, into
    ``vt``: row d at ``vt`` + d VTW holds dim d of keys 0 .. 63 (key j at byte
    j). On the two waves the QK leaves idle, thread t a 4-dim word (reads
    across a wave's consecutive words, 16-byte writes)."""
    tt = c["tid"] - (WAVES - 2) * 64
    if tt >= 0:
        for p_ in range_constexpr(BK // VT_PASS):
            ws = [fx.Int32(fx.ptr_load(kx + ((VT_PASS * p_ + j) * RWK + tt))) for j in range(VT_PASS)]
            for i in range_constexpr(4):
                out = [_byte_q(ws[4 * q : 4 * q + 4], i) for q in range(VT_PASS // 4)]
                fx.ptr_store(fx.Vector.from_elements(out, fx.Int32), vt + ((4 * tt + i) * VTW + VT_PASS * p_ // 4))


def _scaled(v, a):
    return fx.Vector.from_elements([v[i] * a for i in range(4)], fx.Float32)


def _pv(c, lane, wave, vt, pl, al, acc):
    """The block's P V (values transposed at ``vt``, P at ``pl``, the factors
    at ``al``) onto this wave's accumulators, rescaled first: dims 64 wave +
    16 dt + .., every tile's 16 columns; acc[4 ct + dt]."""
    HQ, NCT, NHG = c["HQ"], c["NCT"], c["NHG"]
    g, r16 = lane // 16, lane % 16
    out = list(acc)
    for ct in range_constexpr(NCT):
        alpha = fx.ptr_load(al + (ct * 16 + r16)).bitcast(fx.Float32)
        for dt in range_constexpr(4):
            out[4 * ct + dt] = _scaled(acc[4 * ct + dt], alpha)
    for kc in range_constexpr(KC):
        vops = []
        for dt in range_constexpr(4):
            w = fx.Vector(
                fx.ptr_load(
                    vt + ((PV_DIMS * wave + 16 * dt + r16) * VTW + 8 * kc + 2 * g),
                    result_type=fx.Vector.make_type(2, fx.Int32),
                )
            )
            vops.append(fp8_to_bf16x8(fx.Int32(w[0]), fx.Int32(w[1])))
        for ct in range_constexpr(NCT):
            pb = _lds16(pl + ((ct * 16 + r16) * PW + 16 * kc + 4 * g)).bitcast(fx.BFloat16)
            for dt in range_constexpr(4):
                out[4 * ct + dt] = _mfma16(vops[dt], pb, out[4 * ct + dt])
    return out


def _col(ct, r16, row0, hg, hq):
    """Tile ct's column r16 of an hq-head item: (row, head)."""
    return row0 + 2 * (ct % 3) + r16 // 8, hg * hq + (ct // 3) * 8 + r16 % 8


@flyc.jit
def _partial_out(c, lane, wave, sp, hg, row0, acc, mll):
    """Each (row, head)'s partial over its own sum (the (max, sum) pairs at
    ``mll``), f16, to PV [split][row][head][512] (the request's rows from
    ``row0``): this wave's dims 64 wave + 16 dt + 4 g + i; wave 0 the ML."""
    HQ, NCT, NHG = c["HQ"], c["NCT"], c["NHG"]
    s = c["S"]
    g, r16 = lane // 16, lane % 16
    for ct in range_constexpr(NCT):
        ll = fx.ptr_load(mll + 2 * (ct * 16 + r16) + 1).bitcast(fx.Float32)
        inv = (ll > fx.Float32(0.0)).select(fx.Float32(1.0) / ll, fx.Float32(0.0))
        r, head = _col(ct, r16, row0, hg, HQ)
        o0 = ((sp * s + r) * NHA + head) * KV + PV_DIMS * wave + 4 * g
        for dt in range_constexpr(4):
            hv = fx.Vector.from_elements([acc[4 * ct + dt][i] * inv for i in range(4)], fx.Float32)
            bo.buffer_store(hv.to(fx.Float16).bitcast(fx.Int32), rsrc(c["pv"]), (o0 + 16 * dt) // 2, cache_modifier=CM_DEV)
    if (wave == 0) & (g == 0):
        for ct in range_constexpr(NCT):
            r, head = _col(ct, r16, row0, hg, HQ)
            bo.buffer_store(
                fx.Vector.from_elements(
                    [fx.ptr_load(mll + 2 * (ct * 16 + r16)), fx.ptr_load(mll + 2 * (ct * 16 + r16) + 1)], fx.Int32
                ),
                rsrc(c["ml"]),
                ((sp * s + r) * NHA + head) * 2,
                cache_modifier=CM_DEV,
            )


@flyc.jit
def _qk_softmax(c, kx, blk, kend, m, l, qf, t, lr, own, qk_scale, pl, al):
    """Block ``blk`` (LDS ``kx``; past the run it scores nothing): QK and the
    online softmax on waves 0 .. 5 (a 16-column tile each): P to ``pl``, the
    factors of the earlier sums to ``al``."""
    HQ, NCT, NHG = c["HQ"], c["NCT"], c["NHG"]
    lane, wave = c["lane"], c["wave"]
    g, r16 = lane // 16, lane % 16
    if wave < NCT:
        sacc = _qk(lane, kx, qf)
        m, l, al_ = _softmax(lane, t, blk, kend, sacc, lr, own, m, l, qk_scale, pl + (wave * 16 + r16) * PW)
        if g == 0:
            fx.ptr_store(al_.bitcast(fx.Int32), al + (wave * 16 + r16))
    return m, l


@flyc.jit
def _empty_out(c, lane, wave, sp, hg, row0):
    """A split with no block: zero partials, (max, sum) = (NEG, 0)."""
    HQ, NCT, NHG = c["HQ"], c["NCT"], c["NHG"]
    s = c["S"]
    g, r16 = lane // 16, lane % 16
    zero = fx.Vector.filled(2, 0, fx.Int32)
    for ct in range_constexpr(NCT):
        r, head = _col(ct, r16, row0, hg, HQ)
        o0 = ((sp * s + r) * NHA + head) * KV + PV_DIMS * wave + 4 * g
        for dt in range_constexpr(4):
            bo.buffer_store(zero, rsrc(c["pv"]), (o0 + 16 * dt) // 2, cache_modifier=CM_DEV)
    if (wave == 0) & (g == 0):
        for ct in range_constexpr(NCT):
            r, head = _col(ct, r16, row0, hg, HQ)
            bo.buffer_store(
                fx.Vector.from_elements([fx.Float32(NEG), fx.Float32(0.0)], fx.Float32),
                rsrc(c["ml"]),
                ((sp * s + r) * NHA + head) * 2,
                cache_modifier=CM_DEV,
            )


@flyc.jit
def _account(c, ts):
    """Timeline builds: this wave's step phases (clock stamps ``ts``) onto its
    sums; wave 0 counts the steps."""
    lane, wave, tls = c["lane"], c["wave"], c["tls"]
    if lane == 0:
        r = tls + TL_PHASES * wave + TL_ROLE
        for k in range_constexpr(TL_PHASES):
            fx.ptr_store(fx.ptr_load(r + k) + (ts[k + 1] - ts[k]), r + k)
        if wave == 0:
            fx.ptr_store(fx.ptr_load(tls + TL_STEPS) + fx.Int64(1), tls + TL_STEPS)


@flyc.jit
def attention(c, rq, hg, sa, sb):
    """Request rq's 16-head group hg, key splits sa .. sb (items first + hg n +
    sp), over the splits' blocks of this rank's keys as one pipeline: each
    split's online softmax and accumulators start afresh at its first block,
    so a split's bits are those of the split alone. Each split's partial and
    (max, sum); then ITRDY for every item."""
    HQ, NCT, NHG = c["HQ"], c["NCT"], c["NHG"]
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    g, r16 = lane // 16, lane % 16
    _, lr, own, n, kvp = request(c, rq)
    first_item = pick([b for b, _ in item_bases(c)[0]], rq) + hg * n
    row0 = G_ROWS * rq
    nbl = blocks_of(lr, own)
    b0 = sa * nbl // n
    b1 = sb * nbl // n
    kvl0 = c["kvl"]
    kend = b1 * BK
    # the run's first block's records in flight before the waits (the keys
    # need no q); the next block's rows a step ahead
    rows = _kv_rows(c, kvp, lr, b1, b0)
    _kv_dma(c, rows, kvl0)
    rows = _kv_rows(c, kvp, lr, b1, fx.min(b0 + 1, b1 - 1))
    # splits with no block (fewer blocks than splits)
    for sp in range(sa, sb):
        if sp * nbl // n == (sp + 1) * nbl // n:
            _empty_out(c, lane, wave, sp, hg, row0)
    # the group's q: its sources' QG flags; the new token
    qg_wait(c, rq, hg * HQ, HQ)
    has_new = (own < G_ROWS) & (b1 > lr // BK) & (b0 <= lr // BK)
    if has_new & (tid == 64):
        c["poll"]([(c["kvrdy"], rq, 1)])
    _bar()
    stamp(c["on"], c["tls"], tid, 6)
    # the group's q into LDS: tile ct's column rh a key-sized row
    ct_w = fx.min(wave, NCT - 1)
    t = 2 * (ct_w % 3) + r16 // 8
    # the tile's q in registers, a K step's operands (as ``_qk`` reads them)
    tq, hq = _col(ct_w, r16, row0, hg, HQ)
    qops = []
    for kc in range_constexpr(QK_STEPS):
        wd = 32 * kc + 8 * g if 128 * (kc + 1) <= DQK else fx.min(32 * kc + 8 * g, DQK // 4 - 8)
        lo, hi = qg_load4(c, rq, hq, tq - row0, 4 * wd), qg_load4(c, rq, hq, tq - row0, 4 * wd + 16)
        qops.append(_k128(lane, fx.Vector.from_elements([lo[e] for e in range(4)] + [hi[e] for e in range(4)], fx.Int32), kc))
    qk_scale = c["scale"] * LOG2E
    m = fx.Float32(NEG)
    l = fx.Float32(0.0)
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(NCT * 4)]
    # q and the first block landed: the loop's waits count only its DMAs
    rocdl.s_waitcnt(vmcnt=0)
    _kv_patch(c, rq, lr, own, b0, kvl0)
    _bar()
    pl0, al0, mll0, vt = c["pl"], c["al"], c["mll"], c["vt"]
    # a step a block: the next block's records in flight (every wave a share),
    # QK and the online softmax (waves 0 .. 5, a tile each) while waves 6, 7
    # transpose the block's values into ``vt``, then P V (every wave, 64
    # dims). Buffers: records (blk - b0) % 2.
    for sp in range(sa, sb):
        e0 = sp * nbl // n
        e1 = (sp + 1) * nbl // n
        acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(NCT * 4)]
        for blk in range(e0, e1):
            par = (blk - b0) % 2
            kx = kvl0 + par * (BK * RWK)
            kn = kvl0 + (1 - par) * (BK * RWK)
            stamp(c["on"], c["tls"], tid, 10)
            ts = [clock()] if const_expr(c["on"]) else []
            # the next block's records (every wave a share), then the rows of
            # the one after
            _kv_dma(c, rows, kn)
            rows = _kv_rows(c, kvp, lr, b1, fx.min(blk + 2, b1 - 1))
            # the split's first block: its softmax afresh; its last: the
            # (max, sum) to LDS for the partial
            fst = blk == e0
            m = fst.select(fx.Float32(NEG), m)
            l = fst.select(fx.Float32(0.0), l)
            if wave < NCT:
                sacc = _qk(lane, kx, qops)
                m, l, al_ = _softmax(lane, t, blk, kend, sacc, lr, own, m, l, qk_scale, pl0 + (wave * 16 + r16) * PW)
                if g == 0:
                    fx.ptr_store(al_.bitcast(fx.Int32), al0 + (wave * 16 + r16))
            # meanwhile the other two waves transpose the block's values
            _transpose_v(c, kx, vt)
            if (blk + 1 == e1) & (wave < NCT) & (g == 0):
                mb = mll0 + 2 * (wave * 16 + r16)
                fx.ptr_store(m.bitcast(fx.Int32), mb)
                fx.ptr_store(l.bitcast(fx.Int32), mb + 1)
            if const_expr(c["on"]):
                ts.append(clock())
            _bar()
            if const_expr(c["on"]):
                ts.append(clock())
            acc = _pv(c, lane, wave, vt, pl0, al0, acc)
            if const_expr(c["on"]):
                ts.append(clock())
            rocdl.s_waitcnt(vmcnt=0)
            _bar()
            # every wave's DMAs landed: the new token over its slot
            _kv_patch(c, rq, lr, own, blk + 1, kn)
            _bar()
            if const_expr(c["on"]):
                ts.append(clock())
                _account(c, ts)
            stamp(c["on"], c["tls"], tid, 13)
        if e1 > e0:
            _partial_out(c, lane, wave, sp, hg, row0, acc, mll0)
            _bar()
    stamp(c["on"], c["tls"], tid, 7)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid < sb - sa:
        c["put"](c["itrdy"], first_item + sa + tid, fx.Int32(1))
    gpu.barrier()


# ---------------------------------------------------------------- attention, one request
# One request's 8-head items keep a pipeline of three wave sets a 64-key step:
# QK + softmax of block i + 1 (waves 0 .. 2, a row pair each), the DMA of
# i + 2 (wave 3), P V of block i (waves 4 .. 7, 128 dims each); a column's
# arithmetic is that of the 16-head items.

HQ1 = 8
NRP1 = 3  # row pairs an 8-head item: its 3 16-column tiles
DMA_W1 = NRP1  # the DMA wave
PV_W1 = DMA_W1 + 1  # the P V waves, 128 value dims each
KV_BUFS1 = 3  # LDS record buffers: block i in P V, i + 1 in QK, i + 2 in flight


@flyc.jit
def _kv_dma1(c, kvp, lr, b1, blk, kvl):
    """Block ``blk``'s records straight into LDS at ``kvl`` (padded rows), a
    16-byte slot a lane: a wave's DMA fills 64 consecutive slots (DMA wave
    only). Lane l reads key l's row first; the pad slots and the new token's
    (patched after, ``_kv_patch``) take a valid dummy row."""
    a = c["args"]
    lane, wave = c["lane"], c["wave"]
    if wave == DMA_W1:
        r_kv = rsrc(a["kv"])
        kk = blk * BK + lane
        # a request with no local key has no index to read (an empty range)
        r_idx = rsrc(a["kv_idx"], (lr > 0).select(fx.Int64(0xFFFFFFFF), fx.Int64(0)))
        row_l = fx.Int32(
            bo.buffer_load(r_idx, kvp + fx.max(fx.min(kk, lr - 1), 0), vec_width=1, dtype=T.i32)
        )
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
def _kv_patch1(c, rq, lr, own, blk, kvl):
    """Request rq's new token's record from its NEWKV into its slot, where block
    ``blk`` holds it."""
    e = c["tid"] - DMA_W1 * 64  # a thread of the DMA waves
    if (own < G_ROWS) & (lr // BK == blk) & (e >= 0) & (e < DQK // 16):
        v = _ld4(c["newkv"], rq * (DQK // 4) + 4 * e, CM_DEV)
        fx.ptr_store(v, kvl + ((lr % BK) * RWK + 4 * e))


def _pack_bytes1(words, sh):
    """Byte ``sh`` / 8 of each of eight words -> one i64, word j's byte at byte j."""
    lo = fx.Int32(0)
    hi = fx.Int32(0)
    for j in range_constexpr(4):
        lo = lo | (((words[j] >> sh) & 0xFF) << (8 * j))
        hi = hi | (((words[4 + j] >> sh) & 0xFF) << (8 * j))
    return fx.Vector.from_elements([lo, hi], fx.Int32).bitcast(fx.Int64)[0]


def _v_ops1(lane, kvl, dbase, kc, h):
    """PV's A operands of 32-key step kc for 16-dim tiles 4 h .. 4 h + 3: tile dt
    holds dim dbase + 8 (lane % 16) + dt at keys 32 kc + 8 (lane / 16) .. + 8,
    widened to bf16 (exact)."""
    g, r16 = lane // 16, lane % 16
    word = (dbase + 8 * r16) // 4 + h
    ws = [fx.ptr_load(kvl + ((32 * kc + 8 * g + j) * RWK + word)) for j in range(8)]
    out = []
    for dt in range_constexpr(4):
        v = fx.Vector.from_elements([_pack_bytes1(ws, 8 * dt)], fx.Int64).bitcast(fx.Int32)
        out.append(fp8_to_bf16x8(v[0], v[1]))
    return out


def _pv1(lane, wave, kx, pl, al, acc):
    """Block (LDS ``kx``, its P at ``pl`` and factors at ``al``) P V onto every
    row pair's accumulators (rescaled first): this wave's 128 dims, 8 tiles x
    3 row pairs."""
    g, r16 = lane // 16, lane % 16
    dbase = 128 * (wave - PV_W1)
    out = list(acc)
    for rp in range_constexpr(NRP1):
        alpha = fx.ptr_load(al + (rp * 16 + r16)).bitcast(fx.Float32)
        for dt in range_constexpr(8):
            out[rp * 8 + dt] = _scaled(acc[rp * 8 + dt], alpha)
    for kc in range_constexpr(KC):
        for h in range_constexpr(2):
            vops = _v_ops1(lane, kx, dbase, kc, h)
            for rp in range_constexpr(NRP1):
                pb = _lds16(pl + ((rp * 16 + r16) * PW + 16 * kc + 4 * g)).bitcast(fx.BFloat16)
                for dt in range_constexpr(4):
                    out[rp * 8 + 4 * h + dt] = _mfma16(vops[dt], pb, out[rp * 8 + 4 * h + dt])
    return out


@flyc.jit
def _partial_out1(c, lane, wave, sp, hg, row0, acc, mll):
    """Each (row, head)'s partial over its own sum (from the (max, sum) pairs
    at ``mll``), f16, to PV [split][row][head][512] (the request's rows from
    ``row0``): lane (g, rh) of row pair rp holds dims 128 (wave - 4) + 32 g +
    8 i + dt."""
    s = c["S"]
    g, r16 = lane // 16, lane % 16
    if wave >= PV_W1:
        for rp in range_constexpr(NRP1):
            ll = fx.ptr_load(mll + 2 * (rp * 16 + r16) + 1).bitcast(fx.Float32)
            inv = (ll > fx.Float32(0.0)).select(fx.Float32(1.0) / ll, fx.Float32(0.0))
            vals = [None] * 32
            for i in range_constexpr(4):
                for dt in range_constexpr(8):
                    vals[8 * i + dt] = acc[rp * 8 + dt][i] * inv
            r = row0 + 2 * rp + r16 // HQ1
            head = hg * HQ1 + r16 % HQ1
            o0 = (((sp * s + r) * NHA + head) * KV + 128 * (wave - PV_W1) + 32 * g) // 2
            for q in range_constexpr(4):
                hv = fx.Vector.from_elements(vals[8 * q : 8 * q + 8], fx.Float32).to(fx.Float16).bitcast(fx.Int32)
                bo.buffer_store(hv, rsrc(c["pv"]), o0 + 4 * q, cache_modifier=CM_DEV)
    if (wave == PV_W1) & (g == 0):
        for rp in range_constexpr(NRP1):
            mm = fx.ptr_load(mll + 2 * (rp * 16 + r16))
            ll = fx.ptr_load(mll + 2 * (rp * 16 + r16) + 1)
            r = row0 + 2 * rp + r16 // HQ1
            bo.buffer_store(
                fx.Vector.from_elements([mm, ll], fx.Int32),
                rsrc(c["ml"]),
                ((sp * s + r) * NHA + hg * HQ1 + r16 % HQ1) * 2,
                cache_modifier=CM_DEV,
            )


@flyc.jit
def _qk_softmax1(c, kx, blk, kend, m, l, qf, t, lr, own, qk_scale, pl, al):
    """Block ``blk`` (LDS ``kx``; past the run it scores nothing): QK and the
    online softmax on waves 0 .. 2 (a row pair each): P to ``pl``, the factors
    of the earlier sums to ``al``."""
    lane, wave = c["lane"], c["wave"]
    g, r16 = lane // 16, lane % 16
    if wave < NRP1:
        sacc = _qk(lane, kx, qf)
        m, l, al_ = _softmax(lane, t, blk, kend, sacc, lr, own, m, l, qk_scale, pl + (wave * 16 + r16) * PW)
        if g == 0:
            fx.ptr_store(al_.bitcast(fx.Int32), al + (wave * 16 + r16))
    return m, l


@flyc.jit
def _empty_out1(c, lane, wave, sp, hg, row0):
    """A split with no block: zero partials, (max, sum) = (NEG, 0)."""
    s = c["S"]
    g, r16 = lane // 16, lane % 16
    if wave >= PV_W1:
        zero = fx.Vector.filled(4, 0, fx.Int32)
        for rp in range_constexpr(NRP1):
            r = row0 + 2 * rp + r16 // HQ1
            head = hg * HQ1 + r16 % HQ1
            o0 = (((sp * s + r) * NHA + head) * KV + 128 * (wave - PV_W1) + 32 * g) // 2
            for q in range_constexpr(4):
                bo.buffer_store(zero, rsrc(c["pv"]), o0 + 4 * q, cache_modifier=CM_DEV)
    if (wave < NRP1) & (g == 0):
        t = 2 * wave + r16 // HQ1
        bo.buffer_store(
            fx.Vector.from_elements([fx.Float32(NEG), fx.Float32(0.0)], fx.Float32),
            rsrc(c["ml"]),
            ((sp * s + row0 + t) * NHA + hg * HQ1 + r16 % HQ1) * 2,
            cache_modifier=CM_DEV,
        )


@flyc.jit
def _account1(c, ta, tb, tc):
    """Timeline builds: this wave's step, in its role (ta .. tb) and at the
    barrier (tb .. tc), onto its sums; wave 0 counts the steps."""
    lane, wave, tls = c["lane"], c["wave"], c["tls"]
    if lane == 0:
        r = tls + (TL_ROLE + TL_PHASES * wave)
        fx.ptr_store(fx.ptr_load(r) + (tb - ta), r)
        fx.ptr_store(fx.ptr_load(r + 3) + (tc - tb), r + 3)
        if wave == 0:
            fx.ptr_store(fx.ptr_load(tls + TL_STEPS) + fx.Int64(1), tls + TL_STEPS)


@flyc.jit
def attention1(c, rq, hg, sa, sb):
    """Request rq's 8-head group hg, key splits sa .. sb (items first + hg n +
    sp), over the splits' blocks of this rank's keys as one pipeline: each
    split's online softmax and accumulators start afresh at its first block,
    so a split's bits are those of the split alone. Each split's partial and
    (max, sum); then ITRDY for every item."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    g, r16 = lane // 16, lane % 16
    _, lr, own, n, kvp = request(c, rq)
    first_item = pick([b for b, _ in item_bases(c)[0]], rq) + hg * n
    row0 = G_ROWS * rq
    nbl = blocks_of(lr, own)
    b0 = sa * nbl // n
    b1 = sb * nbl // n
    kvl0 = c["kvl"]
    # the run's first two blocks' records in flight before the waits (the keys
    # need no q)
    kv1 = kvl0 + BK * RWK
    kend = b1 * BK
    _kv_dma1(c, kvp, lr, b1, b0, kvl0)
    _kv_dma1(c, kvp, lr, b1, fx.min(b0 + 1, b1 - 1), kv1)
    # splits with no block (fewer blocks than splits)
    for sp in range(sa, sb):
        if sp * nbl // n == (sp + 1) * nbl // n:
            _empty_out1(c, lane, wave, sp, hg, row0)
    # the group's q: its 8 heads' absorb tasks and rope groups; the new token
    qg_wait(c, rq, hg * HQ1, HQ1)
    has_new = (own < G_ROWS) & (b1 > lr // BK) & (b0 <= lr // BK)
    if has_new & (tid == 64):
        c["poll"]([(c["kvrdy"], rq, 1)])
    _bar()
    stamp(c["on"], c["tls"], tid, 6)
    # the group's q into LDS: (row pair rp, rh) a key-sized row, rh = lane % 16 of
    # QK wave rp: row 2 rp + rh / 8, head rh % 8
    rp = fx.min(wave, NRP1 - 1)
    t = 2 * rp + r16 // HQ1
    qrow = DQK // 16  # 16-byte pieces a (row, head)
    for j in range_constexpr(-(-(NRP1 * 16 * qrow) // THREADS)):
        e = fx.min(tid + THREADS * j, NRP1 * 16 * qrow - 1)
        rr = e // qrow
        pq = e % qrow
        tq = row0 + 2 * (rr // 16) + (rr % 16) // HQ1
        hq = hg * HQ1 + rr % HQ1
        v = qg_load4(c, rq, hq, tq - row0, 16 * pq)
        if tid + THREADS * j < NRP1 * 16 * qrow:
            fx.ptr_store(v, c["qll"] + (rr * RWK + 4 * pq))
    qf = c["qll"] + rp * (16 * RWK)
    qk_scale = c["scale"] * LOG2E
    m = fx.Float32(NEG)
    l = fx.Float32(0.0)
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(NRP1 * 8)]
    # q and the first two blocks landed: the loop's waits count only its DMAs
    rocdl.s_waitcnt(vmcnt=0)
    _kv_patch1(c, rq, lr, own, b0, kvl0)
    _kv_patch1(c, rq, lr, own, b0 + 1, kv1)
    _bar()
    pl0, al0, mll0 = c["pl"], c["al"], c["mll"]
    # block b0's scores first; then each step overlaps three things, one wave
    # set each: QK + softmax of block blk + 1 (waves 0 .. 2), the DMA of blk + 2
    # (wave 3), P V of blk (waves 4 .. 7). Buffers: records (blk - b0) % 3, P
    # and factors (blk - b0) % 2.
    m, l = _qk_softmax1(c, kvl0, b0, kend, m, l, qf, t, lr, own, qk_scale, pl0, al0)
    _bar()
    for sp in range(sa, sb):
        e0 = sp * nbl // n
        e1 = (sp + 1) * nbl // n
        acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(NRP1 * 8)]
        for blk in range(e0, e1):
            i = blk - b0
            x = i % KV_BUFS1
            kx = kvl0 + x * (BK * RWK)
            ky = kvl0 + ((x + 1) % KV_BUFS1) * (BK * RWK)
            kz = kvl0 + ((x + 2) % KV_BUFS1) * (BK * RWK)
            par = i % 2
            pc, ac = pl0 + par * (NRP1 * 16 * PW), al0 + par * (NRP1 * 16)
            pn, an = pl0 + (1 - par) * (NRP1 * 16 * PW), al0 + (1 - par) * (NRP1 * 16)
            stamp(c["on"], c["tls"], tid, 10)
            if const_expr(c["on"]):
                ta = clock()
            # the next split's first block: the split's (max, sum) to LDS
            # (buffer e1 % 2), the softmax afresh
            nxt = blk + 1 == e1
            if nxt & (wave < NRP1) & (g == 0):
                mb = mll0 + (e1 % 2) * (2 * NRP1 * 16) + 2 * (wave * 16 + r16)
                fx.ptr_store(m.bitcast(fx.Int32), mb)
                fx.ptr_store(l.bitcast(fx.Int32), mb + 1)
            m = nxt.select(fx.Float32(NEG), m)
            l = nxt.select(fx.Float32(0.0), l)
            m, l = _qk_softmax1(c, ky, blk + 1, kend, m, l, qf, t, lr, own, qk_scale, pn, an)
            if wave == DMA_W1:
                _kv_dma1(c, kvp, lr, b1, fx.min(blk + 2, b1 - 1), kz)
                rocdl.s_waitcnt(vmcnt=0)
            _kv_patch1(c, rq, lr, own, blk + 2, kz)
            if wave >= PV_W1:
                acc = _pv1(lane, wave, kx, pc, ac, acc)
            if const_expr(c["on"]):
                tb = clock()
            _bar()
            if const_expr(c["on"]):
                _account1(c, ta, tb, clock())
            stamp(c["on"], c["tls"], tid, 13)
        # the split's partial (its (max, sum) stay in buffer e1 % 2 until the
        # next split's last step)
        if e1 > e0:
            _partial_out1(c, lane, wave, sp, hg, row0, acc, mll0 + (e1 % 2) * (2 * NRP1 * 16))
    stamp(c["on"], c["tls"], tid, 7)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid < sb - sa:
        c["put"](c["itrdy"], first_item + sa + tid, fx.Int32(1))
    gpu.barrier()


ITEM_COST = 2  # an item's cost past its blocks, in blocks (its run's start, its partial)
NHG_W = NHA // 16  # 6 head groups a request past one request


def dealt_places(c):
    """Every request's items in decreasing cost (blocks + ITEM_COST a split;
    ties by request), dealt to the CTAs as cards: round q gives item q B + i
    to CTA i, every other round in reverse, so a CTA's few large items sit
    beside the smallest. Each request's (splits, place in the deal) and the
    items in all; a function of the batch alone."""
    reqs, tot = [], fx.Int32(0)
    for r in range(c["NREQ"]):
        _, lr, own, n, _ = request(c, r)
        reqs.append((blocks_of(lr, own) + ITEM_COST * n, n))
        tot = tot + NHG_W * n
    out = []
    for r, (w, n) in enumerate(reqs):
        place = fx.Int32(0)
        for q, (wq, nq) in enumerate(reqs):
            if q == r:
                continue
            # request q's items deal before r's: costlier, or as costly and earlier
            ahead = (wq * n > w * nq) | (wq * n == w * nq) if q < r else (wq * n > w * nq)
            place = place + ahead.select(NHG_W * nq, fx.Int32(0))
        out.append((n, place))
    return out, tot


RQT = 6  # LDS words a request: splits, deal place, blocks, first item, blocks before it, cost before it
RQT_TOT = RQT * MAX_REQS  # then the items in all, the cost in all


@flyc.jit
def deal_table(c):
    """Every request's (splits, deal place, blocks) and the items in all into
    LDS (``rqt``), once a launch: the deal's lookups read them there."""
    if c["tid"] == 0:
        reqs, tot = dealt_places(c)
        first = fx.Int32(0)
        before = fx.Int32(0)
        cost = fx.Int32(0)
        for r in range_constexpr(c["NREQ"]):
            _, lr, own, _, _ = request(c, r)
            nbl = blocks_of(lr, own)
            for f, v in enumerate((reqs[r][0], reqs[r][1], nbl, first, before, cost)):
                fx.ptr_store(v, c["rqt"] + (RQT * r + f))
            first = first + NHG_W * reqs[r][0]
            before = before + NHG_W * nbl
            cost = cost + NHG_W * (nbl + ITEM_COST * reqs[r][0])
        fx.ptr_store(tot, c["rqt"] + RQT_TOT)
        fx.ptr_store(cost, c["rqt"] + (RQT_TOT + 1))
    gpu.barrier()


def rqt(c, r, f):
    return fx.ptr_load(c["rqt"] + (RQT * r + f))


def dealt_item(c, j):
    """Deal place j's (request, head group, split)."""
    r = fx.Int32(0)
    k = fx.Int32(0)
    for q in range(c["NREQ"]):
        n, place = rqt(c, q, 0), rqt(c, q, 1)
        hit = (j >= place) & (j < place + NHG_W * n)
        r = hit.select(fx.Int32(q), r)
        k = hit.select(j - place, k)
    n = rqt(c, r, 0)
    return r, k // n, k % n


# The balance model (us, doubled): a block, an item (its partial), a run (its
# pipeline's start). Each CTA works out its load under both assignments; the
# one with the smaller busiest CTA runs (ties: contiguous ranges).
LOAD_BLOCK, LOAD_ITEM, LOAD_RUN = 9, 18, 24


def contig_runs(c, bid):
    """CTA bid's contiguous range: (k0, k1, first group, runs), from the LDS table."""
    k0, k1 = table_start(c, bid), table_start(c, bid + 1)
    g0 = table_group(c, k0)
    nruns = (k1 > k0).select(table_group(c, k1 - 1) - g0 + 1, fx.Int32(0))
    return k0, k1, g0, nruns


def contig_run(c, rng, gi):
    """Run gi of the range: (request, head group, first split, end split)."""
    k0, k1, g0, _ = rng
    grp = g0 + gi
    rq = grp // NHG_W
    hg = grp % NHG_W
    n = rqt(c, rq, 0)
    gfirst = rqt(c, rq, 3) + hg * n
    return rq, hg, fx.max(k0 - gfirst, fx.Int32(0)), fx.min(k1 - gfirst, n)


def dealt_count(tot, bid):
    """How many deal places (bid, 2 B - 1 - bid, 2 B + bid, ...) fall below tot."""
    ev = fx.max(tot - bid, fx.Int32(0))
    od = fx.max(tot - (2 * BLOCKS - 1 - bid), fx.Int32(0))
    return (ev + 2 * BLOCKS - 1) // (2 * BLOCKS) + (od + 2 * BLOCKS - 1) // (2 * BLOCKS)


def dealt_place(bid, i):
    """CTA bid's i-th deal place."""
    if isinstance(i, int):
        return (i // 2) * (2 * BLOCKS) + (bid if i % 2 == 0 else 2 * BLOCKS - 1 - bid)
    return (i // 2) * (2 * BLOCKS) + (i % 2 == 0).select(fx.Int32(0) + bid, 2 * BLOCKS - 1 - bid)


def table_start(c, b):
    """CTA b's first item under contiguous ranges, from the LDS table: item b
    when the items fit the CTAs, else the item boundary nearest b / 256 of all
    the items' cost (an item's blocks + ``ITEM_COST``; items in order)."""
    nreq = c["NREQ"]
    total = fx.ptr_load(c["rqt"] + RQT_TOT)
    ctot = fx.ptr_load(c["rqt"] + (RQT_TOT + 1))
    x = (b * ctot) // BLOCKS
    r = fx.Int32(0)
    for q in range(1, nreq):
        r = r + (rqt(c, q, 5) < x).select(fx.Int32(1), fx.Int32(0))
    n, nbl = rqt(c, r, 0), rqt(c, r, 2)
    w = nbl + ITEM_COST * n
    xr = x - rqt(c, r, 5)
    sp = ((xr % w) * n + w // 2) // w
    k = rqt(c, r, 3) + (xr // w) * n + sp
    k = (x == 0).select(fx.Int32(0), fx.min(k, total))
    k = (b >= BLOCKS).select(total, k)
    return (total <= BLOCKS).select(fx.min(fx.Int32(0) + b, total), k)


def table_group(c, k):
    """Item k's (request, head group) as one index 6 r + hg, from the LDS table."""
    r = fx.Int32(0)
    for q in range(1, c["NREQ"]):
        r = r + (k >= rqt(c, q, 3)).select(fx.Int32(1), fx.Int32(0))
    return NHG_W * r + (k - rqt(c, r, 3)) // rqt(c, r, 0)


def blocks_before(c, k):
    """The blocks of items 0 .. k - 1 (items in (request, head group, split) order)."""
    r = fx.Int32(0)
    for q in range(1, c["NREQ"]):
        r = r + (k >= rqt(c, q, 3)).select(fx.Int32(1), fx.Int32(0))
    n, nbl = rqt(c, r, 0), rqt(c, r, 2)
    e = k - rqt(c, r, 3)
    return rqt(c, r, 4) + (e // n) * nbl + ((e % n) * nbl) // n


def busiest(c):
    """The modelled busiest CTA under contiguous ranges and under the deal (in
    LDS ``balw``): thread t models CTA t % 256 under assignment t // 256."""
    tid = c["tid"]
    b = tid % BLOCKS
    k0, k1 = table_start(c, b), table_start(c, b + 1)
    runs = (k1 > k0).select(table_group(c, k1 - 1) - table_group(c, k0) + 1, fx.Int32(0))
    lc = LOAD_ITEM * (k1 - k0) + LOAD_RUN * runs + LOAD_BLOCK * (blocks_before(c, k1) - blocks_before(c, k0))
    tot = fx.ptr_load(c["rqt"] + RQT_TOT)
    mine = dealt_count(tot, b)
    ld = (LOAD_ITEM + LOAD_RUN) * mine
    for i in range(DEALT_MAX):
        rq, _, sp = dealt_item(c, fx.min(dealt_place(b, i), tot - 1))
        ld = ld + (i < mine).select(LOAD_BLOCK * split_blocks(c, rq, sp, sp + 1), fx.Int32(0))
    # every lane of a wave stores the wave's max (one value, one word)
    fx.ptr_store(wave_max((tid < BLOCKS).select(lc, ld)), c["balw"] + tid // 64)
    gpu.barrier()
    mc = fx.ptr_load(c["balw"] + 0)
    md = fx.ptr_load(c["balw"] + WAVES // 2)
    for w_ in range(1, WAVES // 2):
        mc = fx.max(mc, fx.ptr_load(c["balw"] + w_))
        md = fx.max(md, fx.ptr_load(c["balw"] + WAVES // 2 + w_))
    return uniform(mc), uniform(md)


DEALT_MAX = -(-NHG_W * MAX_SPLITS * MAX_REQS // BLOCKS) + 1  # items a CTA in the deal, at most
DEAL_CTA = KV_CTA - 1  # 253: the choice at launch, then one g_proj group (idle until the q flags)
DEAL_WORDS = RQT_TOT + 3  # the table, the items and the cost in all, the choice
DEAL_LOADS = DEAL_WORDS  # then the two busiest loads (for the checks)


@flyc.jit
def deal_choose(c):
    """The deal's table and the choice of assignment, into DEAL for every CTA."""
    tid = c["tid"]
    deal_table(c)
    mc, md = busiest(c)
    if tid < DEAL_WORDS - 1:
        c["put"](c["deal"], tid, fx.ptr_load(c["rqt"] + tid))
    if tid == DEAL_WORDS - 1:
        c["put"](c["deal"], tid, (md < mc).select(fx.Int32(1), fx.Int32(0)))
    if tid == DEAL_LOADS:
        c["put"](c["deal"], tid, mc)
    if tid == DEAL_LOADS + 1:
        c["put"](c["deal"], tid, md)
    gpu.barrier()


@flyc.jit
def deal_read(c):
    """DEAL's table into this CTA's LDS; whether the items are dealt."""
    tid = c["tid"]
    v = c["poll"]([(c["deal"], fx.min(tid, DEAL_WORDS - 1), 1)])[0][0]
    if tid < DEAL_WORDS - 1:
        fx.ptr_store(v, c["rqt"] + tid)
    if tid == DEAL_WORDS - 1:
        fx.ptr_store(v, c["balw"] + 0)
    gpu.barrier()
    return uniform(fx.ptr_load(c["balw"] + 0)) != 0


def split_blocks(c, rq, sa, sb):
    """Request rq's blocks in splits sa .. sb."""
    n, nbl = rqt(c, rq, 0), rqt(c, rq, 2)
    return (sb * nbl) // n - (sa * nbl) // n


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


def _split_loads(c, t, h, o, n):
    """Row t, head h, dims 8 o ..: every split's (max, sum) and values, all in flight at once."""
    s = c["S"]
    mws, pvs = [], []
    for sp in range_constexpr(MAX_SPLITS):
        spc = fx.min(fx.Int32(sp), n - 1)
        mws.append(
            fx.Vector(
                bo.buffer_load(rsrc(c["ml"]), ((spc * s + t) * NHA + h) * 2, vec_width=2, dtype=T.f32, cache_modifier=CM_DEV)
            )
        )
        pvs.append(_ld4(c["pv"], (((spc * s + t) * NHA + h) * KV + 8 * o) // 2, CM_DEV))
    return mws, pvs


def _split_max(mws, n):
    """The live splits' (max, sum), past n (NEG, 0), and their max."""
    mx = fx.Float32(NEG)
    mls = []
    for sp in range_constexpr(MAX_SPLITS):
        live = sp < n
        mm = live.select(mws[sp][0], fx.Float32(NEG))
        ll = live.select(mws[sp][1], fx.Float32(0.0))
        mls.append((mm, ll))
        mx = fx.max(mx, mm)
    return mx, mls


@flyc.jit
def _merge_task(c, e, h, half, n):
    """Merge item e (row e / 64, 8 dims at 8 (e % 64)) of head h: its n split
    partials in split order -> the bf16 latent output in LDS; the row's LSE
    beside it, pushed with the first value half."""
    s = c["S"]
    a = c["args"]
    t = e // (KV // 8)
    o = e % (KV // 8)
    owner = h // NHL
    hl = h % NHL
    # one request: a split's values loaded as its weight is known (fewer
    # registers live than ``_split_loads``)
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
        tot = [tot[e_] + wgt * vals[e_] for e_ in range(8)]
    inv = (lsum > fx.Float32(0.0)).select(fx.Float32(1.0) / lsum, fx.Float32(0.0))
    ob = fx.Vector.from_elements([tot[e_] * inv for e_ in range(8)], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    fx.ptr_store(ob, c["xl"] + (t * OL_ROW + 8 * o) // 2)
    if (o == 0) & (half == 0):
        lse = (lsum > fx.Float32(0.0)).select(
            (mx + fx.Float32(fmath.log2(lsum))) * fx.Float32(LN2),
            fx.Float32(EMPTY_LSE),
        )
        # in the row's pad: pushed with the values, after the GEMV
        fx.ptr_store(lse.bitcast(fx.Int32), c["xl"] + (t * OL_ROW + KV) // 2)


@flyc.jit
def stage_uv(c, task):
    """Head task // 2's split partials (its group's items, every request's)
    merged in split order a row -> the bf16 latent output in LDS and the head's
    LSE; then W_UV for value columns 64 (task % 2) ..: the values, and with
    the first half the LSE, pushed to the head's owner rank (mla_combine)."""
    HQ, NCT, NHG = c["HQ"], c["NCT"], c["NHG"]
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    nreq = c["NREQ"]
    h = task // 2
    half = task % 2
    hg = h // HQ
    bases, _ = item_bases(c)
    # the tile streams into LDS on waves 6, 7 while waves 0 .. 5 poll and merge
    # (a wave's loads complete in order)
    _uv_dma(c, task)
    # thread j of request r's run: item (r, hg, j)
    k0 = fx.Int32(0)
    for r in range_constexpr(nreq):
        first, n = bases[r]
        j = tid - k0
        if (j >= 0) & (j < n):
            c["poll"]([(c["itrdy"], first + hg * n + j, 1)])
        k0 = k0 + n
    _bar()
    stamp(c["on"], c["tls"], tid, 8)
    xl, wl = c["xl"], c["wl"]
    owner = h // NHL
    hl = h % NHL
    rounds = -(-s * (KV // 8) // MERGE_THREADS)
    if const_expr(rounds == 1):
        if tid < s * (KV // 8):
            _merge_task(c, tid, h, half, bases[0][1])
    else:
        ns_ = [n for _, n in bases]
        if tid < MERGE_THREADS:
            for rd in range(rounds):
                e = tid + MERGE_THREADS * rd
                _merge_task(c, e, h, half, pick(ns_, e // (G_ROWS * (KV // 8))))
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    vq = wave % 2
    kq = wave // 2
    g, r16 = lane // 16, lane % 16
    tiles = gemv.tiles(s)
    for ti in range_constexpr(len(tiles)):
        t0, sj = tiles[ti]
        acc = _tgemv(lane, wl, UV_ROW, 32 * vq, 128 * kq, 4, xl + t0 * (OL_ROW // 2), OL_ROW, sj)
        rb = red + (wave * 64 + lane) * 8
        for dt in range_constexpr(2):
            fx.ptr_store(acc[dt], rb + 4 * dt)
        gpu.barrier()
        if (wave < 2) & (r16 < sj):
            vals = [None] * 8
            for dt in range_constexpr(2):
                for i in range_constexpr(4):
                    tot = fx.Float32(0.0)
                    for q in range_constexpr(4):
                        tot = tot + fx.ptr_load(red + (((2 * q + wave) * 64 + lane) * 8 + 4 * dt + i))
                    vals[2 * i + dt] = tot
            col = half * UV_HALF + 32 * wave + 8 * g
            mc.push_values(c, owner, hl, t0 + r16, col, vals)
            if (wave == 0) & (g == 0) & (half == 0):
                lse = fx.Int32(fx.ptr_load(xl + ((t0 + r16) * OL_ROW + KV) // 2)).bitcast(fx.Float32)
                mc.push_lse(c, owner, hl, t0 + r16, lse)
                if const_expr(c["records"]):
                    bo.buffer_store(lse, rsrc(a["send"]), ((owner * s + t0 + r16) * NHL + hl) * (REC // 2) + VD // 2)
            if const_expr(c["records"]):
                ob = fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
                bo.buffer_store(ob, rsrc(a["send"]), (((owner * s + t0 + r16) * NHL + hl) * REC + col) // 2)
        gpu.barrier()



def merge_units(s):
    """The merge's (head, row) units a CTA (head-major: unit h s + t), every CTA's share."""
    return -(-NHA * s // BLOCKS)


@flyc.jit
def _merge(c, u, o, n):
    """Unit u (head u // S, row u % S), 8 dims at 8 o: its n split partials in
    split order -> the bf16 latent output row (``ol``); the row's LSE into
    ``send`` once (o == 0)."""
    s = c["S"]
    a = c["args"]
    h = u // s
    t = u % s
    owner = h // NHL
    hl = h % NHL
    mws, pvs = _split_loads(c, t, h, o, n)
    mx, mls = _split_max(mws, n)
    lsum = fx.Float32(0.0)
    tot = [fx.Float32(0.0)] * 8
    for sp in range_constexpr(MAX_SPLITS):
        mm, ll = mls[sp]
        wgt = hw_exp2(mm - mx) * ll
        lsum = lsum + wgt
        vals = fx.Vector(pvs[sp]).bitcast(fx.Float16).to(fx.Float32)
        tot = [tot[e_] + wgt * vals[e_] for e_ in range(8)]
    inv = (lsum > fx.Float32(0.0)).select(fx.Float32(1.0) / lsum, fx.Float32(0.0))
    ob = fx.Vector.from_elements([tot[e_] * inv for e_ in range(8)], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    bo.buffer_store(ob, rsrc(c["ol"]), ((t * NHA + h) * KV + 8 * o) // 2, cache_modifier=CM_DEV)
    if o == 0:
        lse = (lsum > fx.Float32(0.0)).select(
            (mx + fx.Float32(fmath.log2(lsum))) * fx.Float32(LN2),
            fx.Float32(EMPTY_LSE),
        )
        # for the W_UV task's push with its first value half
        bo.buffer_store(lse, rsrc(c["olse"]), t * NHA + h, cache_modifier=CM_DEV)


MERGE_THREADS = UV_DMA_W0 * 64  # the merge's threads (64 a unit: its 8-dim pieces)


@flyc.jit
def stage_merge(c, bid):
    """CTA bid's merge units, MERGE_THREADS // 64 at a time, each once its
    request's head group's split items are in; then OLRDY. A W_UV CTA's tile
    streams into LDS meanwhile (waves 6, 7)."""
    HQ = c["HQ"]
    s, tid = c["S"], c["tid"]
    per = merge_units(s)
    nu = MERGE_THREADS // 64
    bases, _ = item_bases(c)
    if bid < UV_CTAS:
        _uv_dma(c, bid)
    for rd in range_constexpr(-(-per // nu)):
        k = tid // 64
        j = tid % 64
        u = bid * per + rd * nu + k
        live = (tid < MERGE_THREADS) & (rd * nu + k < per) & (u < NHA * s)
        uc = fx.min(u, NHA * s - 1)
        r = (uc % s) // G_ROWS
        hg = (uc // s) // HQ
        n = pick([nn for _, nn in bases], r)
        first = pick([f for f, _ in bases], r)
        if live & (j < n):
            c["poll"]([(c["itrdy"], first + hg * n + j, 1)])
        _bar()
        if live:
            _merge(c, uc, j, n)
    stamp(c["on"], c["tls"], tid, 8)
    publish(c["put"], c["olrdy"], bid, fx.Int32(1), tid == 0)


@flyc.jit
def stage_wuv(c, task):
    """W_UV for head task // 2, value columns 64 (task % 2) ..: its merged
    rows (the merge CTAs' OLRDY) into LDS, the tile already there -> ``send``
    (bf16)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    h = task // 2
    half = task % 2
    per = merge_units(s)
    b0 = (h * s) // per
    nb = ((h + 1) * s - 1) // per - b0 + 1
    if tid < nb:
        c["poll"]([(c["olrdy"], b0 + tid, 1)])
    _bar()
    xl, wl = c["xl"], c["wl"]
    words = s * (KV // 8)
    for i in range_constexpr(-(-words // THREADS)):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (KV // 8)
        o = e % (KV // 8)
        v = _ld4(c["ol"], ((t * NHA + h) * KV + 8 * o) // 2, CM_DEV)
        if tid + THREADS * i < words:
            fx.ptr_store(v, xl + (t * OL_ROW + 8 * o) // 2)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    owner = h // NHL
    hl = h % NHL
    vq = wave % 2
    kq = wave // 2
    g, r16 = lane // 16, lane % 16
    tiles = gemv.tiles(s)
    for ti in range_constexpr(len(tiles)):
        t0, sj = tiles[ti]
        acc = _tgemv(lane, wl, UV_ROW, 32 * vq, 128 * kq, 4, xl + t0 * (OL_ROW // 2), OL_ROW, sj)
        rb = red + (wave * 64 + lane) * 8
        for dt in range_constexpr(2):
            fx.ptr_store(acc[dt], rb + 4 * dt)
        gpu.barrier()
        if (wave < 2) & (r16 < sj):
            vals = [None] * 8
            for dt in range_constexpr(2):
                for i in range_constexpr(4):
                    tot = fx.Float32(0.0)
                    for q in range_constexpr(4):
                        tot = tot + fx.ptr_load(red + (((2 * q + wave) * 64 + lane) * 8 + 4 * dt + i))
                    vals[2 * i + dt] = tot
            col = half * UV_HALF + 32 * wave + 8 * g
            mc.push_values(c, owner, hl, t0 + r16, col, vals)
            if (wave == 0) & (g == 0) & (half == 0):
                lse = fx.Int32(_ld(c["olse"], (t0 + r16) * NHA + h, CM_DEV)).bitcast(fx.Float32)
                mc.push_lse(c, owner, hl, t0 + r16, lse)
                if const_expr(c["records"]):
                    bo.buffer_store(lse, rsrc(a["send"]), ((owner * s + t0 + r16) * NHL + hl) * (REC // 2) + VD // 2)
            if const_expr(c["records"]):
                ob = fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
                bo.buffer_store(ob, rsrc(a["send"]), (((owner * s + t0 + r16) * NHL + hl) * REC + col) // 2)
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
    # here, not at the top: the core's module imports mla_tail, which imports this one
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_decode_c64 as mdc

    s = key.tokens
    nreq = reqs_of(s)
    wd = wide(s)
    hq = heads_of(s)
    nct, nhg = hq * G_ROWS // 16, NHA // hq
    ns = key.nblocks + 1
    lay = scratch_layout(key)
    # the GEMV x rows: every row (narrow), or a 16-row tile of a K segment and
    # every row's normed q latent (wide)
    seg = 2 * WAVES * gemv.KCH  # stage 1's two x round buffers (wide)
    xl_words = (
        max(ROWS * (seg + 2 * LDS_PAD), s * (QL + LDS_PAD)) // 2 if wd else s * (HIDDEN + LDS_PAD) // 2
    )

    @fx.struct
    class AttnLds:
        red: fx.Array[fx.Float32, WAVES * 64 * 8, 16]
        # the sources, f32 (narrow) or packed bf16 pairs (wide)
        vals: fx.Array[fx.Float32, s * ns * COLS // (2 if wd else 1), 16]

    @fx.struct
    class GemvLds:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        xl: fx.Array[fx.Int32, xl_words, 16]

    @fx.struct
    class AbsLds:
        red: fx.Array[fx.Float32, WAVES * 64 * 8, 16]
        xl: fx.Array[fx.Int32, s * (QN + LDS_PAD) // 2, 16]
        wl: fx.Array[fx.Int32, QN * ABS_ROW // 2, 16]

    if nreq == 1:

        @fx.struct
        class CoreLds:  # the one-request pipeline's
            kv: fx.Array[fx.Int32, KV_BUFS1 * BK * RWK, 16]
            p: fx.Array[fx.Int32, 2 * NRP1 * 16 * PW, 16]
            alpha: fx.Array[fx.Int32, 2 * NRP1 * 16, 16]
            ml: fx.Array[fx.Int32, 2 * 2 * NRP1 * 16, 16]
            q: fx.Array[fx.Int32, NRP1 * 16 * RWK, 16]

    else:

        @fx.struct
        class CoreLds:
            kv: fx.Array[fx.Int32, KV_BUFS * BK * RWK, 16]
            vt: fx.Array[fx.Int32, KV * VTW, 16]
            p: fx.Array[fx.Int32, nct * 16 * PW, 16]
            alpha: fx.Array[fx.Int32, nct * 16, 16]
            ml: fx.Array[fx.Int32, 2 * nct * 16, 16]

    @fx.struct
    class UvLds:
        red: fx.Array[fx.Float32, WAVES * 64 * 8, 16]
        xl: fx.Array[fx.Int32, s * OL_ROW // 2, 16]
        wl: fx.Array[fx.Int32, KV * UV_ROW // 2, 16]

    if key.core == "c64":

        @fx.union
        class RoleLds:
            attn: AttnLds
            gemv: GemvLds
            absorb: AbsLds
            dec: mdc.Smem  # the decode core's, the whole of it

    else:

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
        balw: fx.Array[fx.Int32, WAVES, 16]
        rqt: fx.Array[fx.Int32, RQT * MAX_REQS + 4, 16]

    keyed = key_tuple(key, SOURCES)
    name = (
        f"k3_mono_mla_front_s{s}_nb{key.nblocks}_w{int(key.write_idx >= 0)}"
        f"_t{int(key.timeline)}_r{int(key.records)}_x{int(key.x_in)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mla_front(
        prefix: Int64, blocks: Int64, blk_sm: Int32, blk_sr: Int32, ares_nw: Int64,
        ares_qk: Int64, in_nw: Int64, w_qkv: Int64, w_g: Int64, q_nw: Int64,
        kv_nw: Int64, w_qb: Int64, w_kc: Int64, w_vc: Int64, kv: Int64, kv_idx: Int64,
        kv_ptr: Int64, g_kv_ptr: Int64, loc: Int64, core: Int64, send: Int64,
        scratch: Int64, peers: Int64, rank: Int32, layer: Int32, epoch: Int64, tl: Int64,
        dec_part: Int64, dec_plse: Int64, dec_wq: Int64, dec_cnt: Int64,
        dec_o: Int64, dec_lse: Int64, dec_half: Int64,
    ):  # fmt: skip
        _ = keyed
        # Every wait of this launch runs under the guard: one past its deadline
        # flags every peer, and the launch stops on all of them.
        with guard.scope(
            guard.Guard(
                epoch, own=load_ptr64(peers, rank), peers=peers, rank=rank, npes=DCP,
                flags_offset=_flags_offset(),
            )
        ):  # fmt: skip
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            alloc = fx.SharedAllocator()
            lds = alloc.allocate(Smem).peek()
            role = alloc.allocate(RoleLds)
            al, gl = role.attn.peek(), role.gemv.peek()
            abl = role.absorb.peek()
            if const_expr(key.core == "c64"):
                dl = role.dec.peek()
            else:
                cl, ul = role.core.peek(), role.uv.peek()
            mb = Mailbox(step_tag(epoch, layer) - 1)
            gate = scratch + fx.Int64(lay["g"][0])
            c = {
                "S": s, "NS": ns, "NB": key.nblocks, "tid": tid, "bid": bid,
                "lane": tid % 64, "wave": tid // 64, "delta": False, "debug": False,
                "eps": key.eps, "out_eps": key.out_eps, "q_eps": key.q_eps,
                "kv_eps": key.kv_eps, "write_idx": key.write_idx, "gate_local": not wd,
                "vals_bf16": wd, "NREQ": nreq, "HQ": hq, "NCT": nct, "NHG": nhg,
                "qlora_out": False, "qn": False, "put": mb.put, "poll": mb.poll,
                # every role but the attention's starts with the same red (offset 0)
                "red": al.red.ptr, "vals": al.vals.ptr, "xl": gl.xl.ptr,
                "scale": key.scale, "rank": rank, "loc_words": 2,
                "qg_off": fx.Int64(qg_offset() + key.qg_par * QG_COPY),
                "xbuf": prefix if key.x_in else scratch + fx.Int64(lay["x"][0]),
                "x_in": key.x_in,
                "lat": scratch + fx.Int64(lay["lat"][0]),
                "qb": scratch + fx.Int64(lay["qb"][0]),
                "q8": scratch + fx.Int64(lay["q8"][0]),
                "newkv": scratch + fx.Int64(lay["newkv"][0]),
                "pv": scratch + fx.Int64(lay["pv"][0]),
                "ml": scratch + fx.Int64(lay["ml"][0]),
                "records": key.records, "gate0": S1_GATE0, "gbuf": gate,
                "a2a_peer": mc.PeerRegions(peers, key.qg_par),
                "args": {
                    "prefix": prefix, "delta": 0, "blocks": blocks, "blk_sm": blk_sm,
                    "blk_sr": blk_sr, "ares_nw": ares_nw, "ares_qk": ares_qk, "in_nw": in_nw,
                    "x_dbg": 0, "w_qkv": w_qkv, "w_g": w_g, "q_nw": q_nw, "kv_nw": kv_nw,
                    "w_qb": w_qb, "w_kc": w_kc, "w_vc": w_vc, "kv": kv, "kv_idx": kv_idx,
                    "kv_ptr": kv_ptr, "g_kv_ptr": g_kv_ptr, "loc": loc, "gate": gate,
                    "send": send, "core": core, "q_lora": 0, "peers": peers,
                },
            }  # fmt: skip
            if const_expr(key.core != "c64"):
                c.update({
                    "kvl": cl.kv.ptr, "pl": cl.p.ptr, "al": cl.alpha.ptr, "mll": cl.ml.ptr,
                    "qll": cl.q.ptr if nreq == 1 else None, "vt": cl.vt.ptr if nreq > 1 else None,
                })  # fmt: skip
            for region in ("part", "wgt", "msq", "xrdy", "ss", "qrdy", "absrdy", "kvrdy", "itrdy", "grdy"):
                c[region] = sreg(scratch, lay[region][0], region)
            if const_expr(wd):
                # the wide merge's latent rows, their LSEs and its CTAs' flags
                c["olrdy"] = sreg(scratch, lay["olrdy"][0], "olrdy")
                c["ol"] = scratch + fx.Int64(lay["ol"][0])
                c["olse"] = scratch + fx.Int64(lay["olse"][0])
                c["balw"] = lds.balw.ptr
                c["deal"] = sreg(scratch, lay["deal"][0], "deal")
                c["rqt"] = lds.rqt.ptr
            on, tls = key.timeline, lds.tls.ptr
            c["on"], c["tls"] = on, tls
            stamp_begin(on, tls, tid, TL_POINTS)
            if const_expr(wd and nreq > 1):
                # the attention's assignment, worked out at launch on a CTA whose
                # only role (a g_proj group) is off the q path
                if bid == DEAL_CTA:
                    deal_choose(c)
            if const_expr(not key.x_in):
                if bid < NA:
                    stage_slice(c, bid)
                    if const_expr(wd):
                        if bid < s:
                            stage_gate(c, bid)
                    stage_mix(c, bid)
            if (bid >= QKV0) & (bid < GP0):
                # fused_qkv_a: the q path's first weights, at launch
                ops = s1_loads(c, bid - QKV0)
                if const_expr(wd):
                    x_seen(c)
                    stage1_wide(c, bid - QKV0, ops)
                else:
                    load_x(c)
                    stage1(c, bid - QKV0, ops)
            if bid >= GP0:
                # g_proj, read only after the launch
                ops = s1_loads(c, S1_GATE0 + bid - GP0)
                if const_expr(wd):
                    x_seen(c)
                    stage1_wide(c, S1_GATE0 + bid - GP0, ops)
                else:
                    load_x(c)
                    stage1(c, S1_GATE0 + bid - GP0, ops)
                mc.gate_ready(c, S1_GATE0 + bid - GP0)
                if bid < GP0 + GP2:
                    ops2 = s1_loads(c, S1_GATE0 + BLOCKS - GP0 + bid - GP0)
                    if const_expr(wd):
                        stage1_wide(c, S1_GATE0 + BLOCKS - GP0 + bid - GP0, ops2)
                    else:
                        stage1(c, S1_GATE0 + BLOCKS - GP0 + bid - GP0, ops2)
                    mc.gate_ready(c, S1_GATE0 + BLOCKS - GP0 + bid - GP0)
            if bid == KV_CTA:
                stage_kv(c)
            stamp(on, tls, tid, 2)
            # stage 2: this rank's 144 q_b groups, one a CTA on CTAs 0 .. 143 (a
            # group index past them loads nothing)
            qbg = QBG_L * rank + bid
            live_qb = bid < QBG_L
            ops = s2_loads(c, live_qb.select(qbg, fx.Int32(QBG)))
            # stage 3's tile (the absorb CTAs) in flight meanwhile
            lo_abs = (bid >= ABS0) & (bid < ABS0 + ABS_N0)
            hi_abs = (bid >= ABS1) & (bid < ABS1 + ABS_L - ABS_N0)
            task = ABS_L * rank + lo_abs.select(bid - ABS0, bid - ABS1 + ABS_N0)
            live_abs = lo_abs | hi_abs
            w3 = s3_loads(c, live_abs.select(task, fx.Int32(0)))
            if live_qb:
                if const_expr(wd):
                    load_qlora_wide(c)
                else:
                    load_qlora(c)
            stamp(on, tls, tid, 3)
            if live_qb:
                stage2(c, qbg, ops)
                stage2_flag(c, qbg, 0)
            _bar()
            stamp(on, tls, tid, 4)
            # stage 3: this rank's 48 absorb tasks, one a CTA; then the q flags
            c["xl"], c["wl"] = abl.xl.ptr, abl.wl.ptr
            if live_abs:
                stage3(c, task, w3)
            if bid == QG_FLAG_CTA:
                qg_flags(c)
            stamp(on, tls, tid, 5)
            if const_expr(key.core == "c64"):
                # the decode core through W_UV, its records into the owners'
                # all-to-all cells; its LDS is the whole union
                gpu.barrier()
                dec_a = {
                    "kv": kv, "kv_idx": kv_idx, "kv_ptr": kv_ptr, "g_ptr": g_kv_ptr, "rank": rank,
                    "o": dec_o, "lse": dec_lse, "part": dec_part, "plse": dec_plse, "wq": dec_wq,
                    "cnt": dec_cnt, "layer": fx.Int32(key.mla_ord), "wuv": w_vc, "half": dec_half,
                    "tl": fx.Int64(0),
                }
                hk = core_hooks(c)
                mdc.decode_stage(
                    mdc.MlaDecodeBuild(scale=key.scale), dl, bid, tid, dec_a, hk["q_load"], hk["ready"],
                    hk["kv_ready"], hk["emit_values"], hk["emit_lse"],
                )  # fmt: skip
                # the combine's LDS (the attention role's) over the core's
                gpu.barrier()
            else:
                if const_expr(nreq == 1):
                    _, _, _, n, _ = request(c)
                    if bid < nhg * n:
                        attention1(c, 0, bid // n, bid % n, bid % n + 1)
                else:
                    # the assignment with the smaller modelled busiest CTA (DEAL_CTA's
                    # choice; a function of the batch alone)
                    assert nhg == NHG_W
                    dealt = deal_read(c)
                    # contiguous ranges a run at a time, or the dealt items one at a
                    # time; the range kept in LDS, so nothing more stays live across
                    # the attention than the loop's count
                    rng = contig_runs(c, bid)
                    tot = fx.ptr_load(c["rqt"] + RQT_TOT)
                    count = dealt.select(dealt_count(tot, bid), rng[3])
                    if tid == 0:
                        for f in range_constexpr(3):
                            fx.ptr_store(rng[f], c["balw"] + (1 + f))
                    gpu.barrier()
                    for i in range(count):
                        dealt = fx.ptr_load(c["balw"] + 0) != 0
                        rng = tuple(fx.ptr_load(c["balw"] + (1 + f)) for f in range(3)) + (count,)
                        rc, hc, sac, sbc = contig_run(c, rng, i)
                        rd, hd, spd = dealt_item(c, dealt_place(bid, i))
                        rq = dealt.select(rd, rc)
                        hg = dealt.select(hd, hc)
                        sa = dealt.select(spd, sac)
                        sb = dealt.select(spd + 1, sbc)
                        if sb > sa:
                            attention(c, rq, hg, sa, sb)
                c["xl"], c["wl"] = ul.xl.ptr, ul.wl.ptr
                if const_expr(wd):
                    # every CTA's share of the (head, row) merges, then W_UV a (head, half)
                    stage_merge(c, bid)
                    if bid < UV_CTAS:
                        stage_wuv(c, bid)
                else:
                    # one request: a (head, half) task merges its head's rows itself
                    if bid < UV_CTAS:
                        stage_uv(c, bid)
            stamp(on, tls, tid, 9)
            # the owner's combine and output gate, a (request, local head) a
            # task on the last CTAs
            comb0 = BLOCKS - mc.combine_tasks(s)
            if bid >= comb0:
                mc.stage_combine(c, bid - comb0)
            stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        prefix: Int64, blocks: Int64, blk_sm: Int32, blk_sr: Int32, ares_nw: Int64,
        ares_qk: Int64, in_nw: Int64, w_qkv: Int64, w_g: Int64, q_nw: Int64,
        kv_nw: Int64, w_qb: Int64, w_kc: Int64, w_vc: Int64, kv: Int64, kv_idx: Int64,
        kv_ptr: Int64, g_kv_ptr: Int64, loc: Int64, core: Int64, send: Int64,
        scratch: Int64, peers: Int64, rank: Int32, layer: Int32, epoch: Int64, tl: Int64,
        dec_part: Int64, dec_plse: Int64, dec_wq: Int64, dec_cnt: Int64,
        dec_o: Int64, dec_lse: Int64, dec_half: Int64,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        mla_front(
            prefix, blocks, blk_sm, blk_sr, ares_nw, ares_qk, in_nw, w_qkv, w_g, q_nw,
            kv_nw, w_qb, w_kc, w_vc, kv, kv_idx, kv_ptr, g_kv_ptr, loc, core, send,
            scratch, peers, rank, layer, epoch, tl, dec_part, dec_plse, dec_wq, dec_cnt,
            dec_o, dec_lse, dec_half,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    if key.core == "c64":
        # the decode core's registers: no AGPRs, its accumulators in VGPRs; no
        # packed fp32 (it serializes with the MFMAs) -- as its own launch
        mdc._agpr_hint_lowering()
        launch = flyc.compile[{"agpr_alloc": 0, "llvm_options": {"slp-threshold": 100000}}](launch)
    _BUILDS[key] = launch
    return launch


def ptr(t):
    return 0 if t is None else t.data_ptr()


# W_UK's local shard as SGLang holds it: [12, 128 n, 512 k], n innermost
W_KC_STRIDES = (QN * KV, 1, QN)

WEIGHT_SHAPES = {
    "w_qkv": (LATW, HIDDEN), "w_g": (GATE, HIDDEN), "w_qb": (NHL * QHD, QL),
    "w_kc": (NHL, QN, KV), "w_vc": (NHA, KV, VD),
}  # fmt: skip


def weight_ok(name, t) -> bool:
    """Whether ``t`` is weight ``name`` as the launch reads it: bf16 of its
    shape, contiguous (W_UK: ``W_KC_STRIDES``)."""
    if not isinstance(t, torch.Tensor) or t.dtype != torch.bfloat16 or tuple(t.shape) != WEIGHT_SHAPES[name]:
        return False
    return tuple(t.stride()) == W_KC_STRIDES if name == "w_kc" else t.is_contiguous()


def mla_front_launch(
    key: MlaFrontBuild, *, prefix, blocks, w, kv, kv_idx, kv_ptr, g_kv_ptr, loc,
    core, scratch, peers, rank, layer, epoch, send=None, tl=None, plan=None,
):  # fmt: skip
    """Host launcher. ``w``: ares_nw, ares_qk, in_nw, q_nw, kv_nw and the
    ``WEIGHT_SHAPES`` weights (bf16, contiguous); ``kv``: this rank's e4m3 KV
    rows [rows, 576] (row-contiguous); ``kv_idx`` / ``kv_ptr``: the requests'
    local rows by position (int32, ragged: SGLang's DCP verify indices);
    ``g_kv_ptr``: their global KV indptr (int32, verify rows included);
    ``loc``: the rows' KV slots (int64);
    ``core`` [S, 1536] bf16 out (the gated core); ``peers``: the DCP ranks'
    peer buffers (int64 [8], ``peer_end`` bytes each: the q all-gather's QG
    copies and the all-to-all's A2A copies, ``key.qg_par`` the one this launch
    uses); ``send`` [8, S, 12, 130] bf16 out, a ``records`` build only;
    ``w_qb`` / ``w_kc`` this rank's 12 heads' shards. An ``x_in`` build takes
    the x rows [S, 7168] bf16 as ``prefix`` and no ``blocks`` (None)."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_decode_c64 as mdc

    if key.x_in:
        assert blocks is None and key.write_idx < 0
        assert prefix.is_contiguous() and tuple(prefix.shape) == (key.tokens, HIDDEN)
        blk = (0, 0, 0)
    else:
        assert key.write_idx < blocks.shape[1]
        blk = (blocks.data_ptr(), blocks.stride(0), blocks.stride(1))
    for n, shp in WEIGHT_SHAPES.items():
        assert weight_ok(n, w[n]), (n, tuple(w[n].shape), w[n].stride(), w[n].dtype)
    assert kv.dtype in (torch.float8_e4m3fn, torch.uint8) and kv.stride(-2) == DQK
    assert kv_idx.dtype == torch.int32 and kv_ptr.dtype == torch.int32
    assert loc.dtype == torch.int64 and g_kv_ptr.dtype == torch.int32
    assert peers.dtype == torch.int64 and peers.numel() == DCP and key.qg_par in (0, 1)
    nreq = reqs_of(key.tokens)
    assert kv_ptr.numel() >= nreq + 1 and g_kv_ptr.numel() >= nreq + 1 and loc.numel() >= key.tokens
    assert (send is not None) == key.records
    if send is not None:
        assert tuple(send.shape) == (DCP, key.tokens, NHL, REC) and send.dtype == torch.bfloat16
    assert tuple(core.shape) == (key.tokens, GATE) and core.dtype == torch.bfloat16 and core.is_contiguous()
    dec = (0,) * 7
    if key.core == "c64":
        bufs = dec_buffers(core.device)
        if key.mla_ord == 0 if plan is None else plan:
            # the step's work list, once a step: its first front launches it
            # (``plan``: a warmup of some of the layers launches it before each)
            mdc.mla_decode_plan_launch(kv_ptr=kv_ptr, g_ptr=g_kv_ptr, nreq=nreq, bufs=bufs)
        dec = (
            bufs.part.data_ptr(), bufs.plse.data_ptr(), bufs.wq.data_ptr(), bufs.cnt.data_ptr(),
            bufs.o.data_ptr(), bufs.lse.data_ptr(), bufs.half.data_ptr(),
        )
    f = build(key)
    f(
        prefix.data_ptr(), *blk,
        w["ares_nw"].data_ptr(), w["ares_qk"].data_ptr(), w["in_nw"].data_ptr(),
        w["w_qkv"].data_ptr(), w["w_g"].data_ptr(), w["q_nw"].data_ptr(),
        w["kv_nw"].data_ptr(), w["w_qb"].data_ptr(), w["w_kc"].data_ptr(),
        w["w_vc"].data_ptr(), kv.data_ptr(), kv_idx.data_ptr(), kv_ptr.data_ptr(),
        g_kv_ptr.data_ptr(), loc.data_ptr(), core.data_ptr(), ptr(send),
        scratch.data_ptr(), peers.data_ptr(), rank, layer, epoch.data_ptr(), ptr(tl), *dec,
        stream=torch.cuda.current_stream(),
    )  # fmt: skip


_DEC = {}


def dec_buffers(device):
    """The decode core's work list, counters and split partials, one set a
    device: the layers run in turn, so they share it."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_decode_c64 as mdc

    if device not in _DEC:
        _DEC[device] = mdc.StageBuffers(device, MAX_REQS * G_ROWS)
    return _DEC[device]


def scratch_view(key: MlaFrontBuild, scratch, name: str, dtype, shape):
    """A scratch region as a tensor, for the checks."""
    off, nbytes = scratch_layout(key)[name]
    return scratch[off : off + nbytes].view(dtype).view(shape)
