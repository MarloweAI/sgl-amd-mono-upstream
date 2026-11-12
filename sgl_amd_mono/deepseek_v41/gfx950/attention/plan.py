# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 and K2 attention geometry and scratch layout."""

import msgspec

from sgl_amd_mono.common.sync import FLAG_STRIDE

HIDDEN = 5120
Q_RANK = 1280
KV_DIM = 512
QKV_ROWS = Q_RANK + KV_DIM  # 1792
HEAD_DIM = 512
NOPE = 448
ROPE = 64
HALF = ROPE // 2
N_HEADS = 64
O_GROUPS = 8
O_RANK = 1024
WINDOW = 128
TOPK = 512
EPS = 1e-20

ROWS = 16  # GEMV rows a task
TILE = 16  # tokens a GEMV pass (an MFMA's N)
HEAD_TILE = 16  # heads an attention MFMA takes
WOA_ROWS = 32  # a wo_a task's rows: one MX group of its output

# KV record (V4 fp8_ds_mla): 576 B rows (448 e4m3, 64 bf16), then 8 B of scales a token
DATA = 576

PAIR = 8  # a mailbox (value, tag) pair
COMB_PARTS = 2  # K2's combine units a (token, 8-head group) by default (``comb_parts``)
KEYS = TOPK + WINDOW  # a token's attention keys at most


def cdiv(a, b):
    return -(-a // b)


def round_up(a, b):
    return cdiv(a, b) * b


class Dims(msgspec.Struct, frozen=True):
    tp: int

    def __post_init__(self):
        assert self.tp in (2, 4, 8), self.tp

    @property
    def heads(self) -> int:
        return N_HEADS // self.tp

    @property
    def tile_heads(self) -> int:
        """``heads`` padded to whole head tiles: TP8's 8 heads fill half of one."""
        return round_up(self.heads, HEAD_TILE)

    @property
    def head_tiles(self) -> int:
        return self.tile_heads // HEAD_TILE

    @property
    def groups(self) -> int:
        return O_GROUPS // self.tp

    @property
    def group_k(self) -> int:
        """wo_a's K a group: its eight heads' outputs."""
        return self.heads // self.groups * HEAD_DIM  # 4096

    @property
    def o_rows(self) -> int:
        return self.groups * O_RANK

    @property
    def q_rows(self) -> int:
        return self.heads * HEAD_DIM

    @property
    def wqb_tasks(self) -> int:
        return self.q_rows // ROWS

    @property
    def woa_tasks(self) -> int:
        return self.o_rows // WOA_ROWS

    @property
    def wob_tasks(self) -> int:
        return HIDDEN // ROWS  # 320


WQKV_TASKS = QKV_ROWS // ROWS  # 112
WQKV_KPARTS = 2  # X8's K halves (a flag each); an LDS window of the wqkv GEMV


def wqkv_order(s):
    """Return SGLang's wqkv_a K order at width s as (chains, halves, sum order)."""
    if s <= 32:
        w = 16 if s <= 4 else 8
        return w, False, tuple(range(w))
    assert s <= 64
    return 5, True, (0, 4, 1, 2, 3)


def wqkv_split(s, own):
    """Whether K1 runs each wqkv row tile's chains on two CTAs: SGLang's order of an even
    chain count, and room past their 224 of 256 CTAs for the norms, gates, qkv and key table."""
    waves, halves, _ = wqkv_order(s)
    return not halves and waves % 2 == 0 and 2 * WQKV_TASKS + 4 * own < 256


def token_tiles(s):
    return [(t0, min(TILE, s - t0)) for t0 in range(0, s, TILE)]


def tile_rows(s):
    return min(s, TILE)


def byte_layout(items, start=0, align=256):
    out, off = {}, start
    for name, nbytes in items:
        off = round_up(off, align)
        out[name] = (off, nbytes)
        off += nbytes
    return out


def front_scratch(s: int, d: Dims) -> dict:
    """Return K1's scratch regions -> (byte offset, bytes)."""
    return byte_layout(
        [
            ("x8", s * HIDDEN),  # e4m3 words (S > 16: quantized once, shared)
            ("x8s", s * HIDDEN // 32 * 4),  # an i32 code a group
            ("xrdy", WQKV_KPARTS * s * PAIR),  # a flag a (token, K half)
            ("qkvp", WQKV_KPARTS * s * QKV_ROWS * PAIR),
            ("qx8", s * Q_RANK),  # e4m3 words
            ("qx8s", s * Q_RANK // 32 * 4),  # an i32 code a group
            ("qrdy", s * PAIR * FLAG_STRIDE),
            # an index source's ``index_query``: bf16 IQ, a flag a (token tile, 16-column task)
            ("iq", s * 4096 * 2),
            ("iqrdy", cdiv(s, 16) * 256 * PAIR),
            # its head weights' partials, a pair a (token, head, 512-K split)
            ("iwp", cdiv(s, 16) * 32 * 16 * (HIDDEN // 512) * PAIR),
            # its compressor projection's partials, a pair a (token, column, K quarter)
            ("cpp", cdiv(s, 16) * 16 * 1024 * 4 * PAIR),
            # a flag a (token tile, 16-column tile) of it, once its sums are stored
            ("cprdy", cdiv(s, 16) * 64 * PAIR),
        ]
    )


def comb_parts(d: Dims, s: int) -> int:
    """K2's combine units a (token, 8-head group), a part of the head dims each: four on TP8 at
    one token tile, where a unit's chunk partials then stream in half the bytes."""
    return 4 if d.tp == 8 and s <= TILE else COMB_PARTS


def split_keys(d: Dims, s: int) -> int:
    """A split unit's keys (K2's flash-decoding over a chunk of a token's keys): 64
    where the units fit, else 128. Two head tiles: both tiles' units of every chunk
    fit the grid. One: every chunk's units and the combine's fit 112 CTAs; past that
    the combine's twice as many partials cost more than the split's halved gather."""
    if d.head_tiles == 2:
        return 64 if s * cdiv(KEYS, 64) <= 256 else 128
    units = s * (cdiv(KEYS, 64) + d.heads // 8 * COMB_PARTS)
    return 64 if units <= 112 else 128


WORD = 16  # a tagged word's bytes: three data words, then the launch tag


def tagged_x8b(d: Dims, s: int) -> bool:
    """Whether wo_a hands X8B to wo_b as tagged words (no store-completion wait, no
    flag): on TP4 and TP8 at one token tile."""
    return d.tp in (4, 8) and s <= TILE


def tagged_q_split(d: Dims, s: int, tok_par: bool = False) -> bool:
    """Whether, in one launch a layer, wq_b hands q to the split as tagged words
    (``q_words``: no q flag), each row's KV record behind a flag of its own: on TP8 at
    one token tile, not token parallel."""
    return d.tp == 8 and s <= TILE and not tok_par


def q_words(d: Dims) -> int:
    """Tagged q's words a token (``tagged_q_split``): three a wq_b task,
    its 16 bf16 dims as 6, 6 and 4."""
    return d.wqb_tasks * 3


def x8b_words(d: Dims, s: int) -> int:
    """Tagged X8B's words: three a (token, wo_a task), its 32 e4m3 and E8M0 code."""
    return s * d.woa_tasks * 3


def back_scratch(s: int, d: Dims, start: int = 0) -> dict:
    """Return K2's scratch regions past ``start`` -> (byte offset, bytes)."""
    nchunk = cdiv(KEYS, split_keys(d, s))
    if tagged_x8b(d, s):
        x8b = [("x8t", x8b_words(d, s) * WORD)]
    else:
        x8b = [
            ("x8b", s * d.o_rows),
            ("x8bs", s * d.o_rows // 32 * 4),
            # a flag a wo_a unit, a line each (``FLAG_STRIDE``)
            ("ardy", d.woa_tasks * cdiv(s, ROWS) * PAIR * FLAG_STRIDE),
        ]
    return byte_layout(
        [
            ("part", s * nchunk * d.heads * HEAD_DIM * 2),  # bf16 partials
            ("pstat", s * nchunk * d.heads * 8),  # (max, sum) f32
            ("srdy", s * nchunk * PAIR * FLAG_STRIDE),
            ("xo", s * d.q_rows * 2),  # bf16: wo_a reads it unquantized
            ("crdy", s * d.heads // 8 * comb_parts(d, s) * PAIR * FLAG_STRIDE),
        ]
        + x8b,
        start=start,
    )
