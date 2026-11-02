# SPDX-License-Identifier: Apache-2.0
"""D-attn: the DFlash2 draft's attention over the block for S = ``tokens`` rows (a
block of 8 consecutive rows a request): the Muse Spark K-attn (``stages/
attention.py``) at the draft's geometry: head_dim 128, one KV head a rank (8 query
heads of it), fp8 page-16 cache [slot, 1, 128], the block's eight queries all see
all eight block keys (an encoder-only layer) and the prefix's last ``window`` keys
(the window from the prefix's end, not the row's), the page ids straight from
``req_to_token`` (the slot of a page's first key / 16). A page is 2 KB, two DMA
instructions; a chunk's K / V are eight; the key-split tree folds in two rounds
(four waves a round); the merge jobs fold 128 dims in two passes of 64.

Everything else (the split-KV tasks, the one-task-a-CTA partition sizing, the
lazy rescale, the transposed V reads, the two-level merge) is K-attn's.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import rocdl as rocdl_raw
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    ballot,
    fp8_to_bf16x8,
    hw_exp2,
    mfma_bf16,
    permlane_swap,
    rsrc,
)
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, pair_layout
from sgl_amd_mono.muse_spark.gfx950.config import LOG2E, MAX_TOKENS
from sgl_amd_mono.muse_spark.gfx950.device import (
    TRACE_LAYERS,
    TRACE_SLOTS,
    lds_dma_16,
    poll_flags,
    release_stores,
    stamp,
)
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_CHUNK as CHUNK
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_KEY_TILES as KEY_TILES
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_KEYS_PER_WAVE as KEYS_PER_WAVE
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_MERGE_GROUPS as GROUPS
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_RING_DEPTH as RING_DEPTH
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_V_STEPS as V_STEPS

# the draft's geometry (a rank of TP 8)
HEAD_DIM = 128
LOCAL_KV_HEADS = 1
GQA = 8  # query heads a KV head on a rank
Q_K = GQA * LOCAL_KV_HEADS * HEAD_DIM  # 1024: the q and attention-output widths
D_STEPS = HEAD_DIM // 32  # 4 MFMA k-steps over the dims
D_TILES = HEAD_DIM // 16  # 8 dim tiles of the PV MFMA
MAX_PARTS = 16  # partitions a request at most: a window of 2048 + 8 keys is 9 chunks
IDS_CHUNKS = 16  # chunks whose page ids a task stages in LDS (the window's 9)
PART_ROW = 4 + HEAD_DIM  # fp32 words a partial row: (m, l, pad, pad, O[128])
Q_PITCH = (
    HEAD_DIM + 4
)  # the Q tile's LDS pitch (bf16 elements, 16 B aligned, bank-spread)


def attn_rows(q_len: int) -> int:
    return q_len * GQA


def dattn_scratch_layout(tokens: int, q_len: int) -> dict:
    """D-attn's scratch (a "pair" is 8 bytes): the tasks' partial rows, a flag a
    task, the level-2 records and flags, the trace."""
    R = tokens // q_len
    rows = attn_rows(q_len)
    tiles = (rows + 15) // 16
    part_words = R * LOCAL_KV_HEADS * MAX_PARTS * rows * PART_ROW
    l2_words = R * LOCAL_KV_HEADS * tiles * GROUPS * 16 * PART_ROW
    return pair_layout(
        [
            ("attn_part", part_words // 2),
            (
                "attn_pflag",
                R * LOCAL_KV_HEADS * tiles * MAX_PARTS,
            ),  # a flag a task (its first row tile)
            ("attn_l2", l2_words // 2),
            ("attn_l2flag", R * LOCAL_KV_HEADS * tiles * GROUPS),
            ("attn_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


TILE = 16  # keys an MFMA tile
NEG = -1.0e30  # a masked score, and "no key yet" of a row's running max
NO_MAX = -1.0e29  # a running max below this is "no key yet": exp2 then takes 0
O_OFF = 4  # a partial row: m, l, pad, pad, O[128] (O 16 B aligned)
COMB_ROW = PART_ROW  # a tile row's (m, l, pad, pad, O[64]) in the LDS tree
MERGE_LANES = 4  # partitions folded side by side a (row, 8 dims): 16 x 8 x 4 = THREADS
PAGE_WORDS = (
    16 * HEAD_DIM // 4
)  # a 16-key fp8 page of one head in LDS: 2 KB, chunk-major (dims 16 c .., key)
PAGE_DMAS = (
    HEAD_DIM // 64
)  # DMA instructions a page: 64 lanes x 16 B cover 16 keys x 64 dims
V_SKEW = 16  # words between the two V pages beyond a page: the transposed reads of both hit distinct banks
SLOT_WORDS = (
    2 * KEY_TILES * PAGE_WORDS + V_SKEW
)  # a ring slot: a wave's chunk, its K pages then its V pages
TAU = 8.0  # a row's reference max moves only when a chunk's max exceeds it by this (log2 units): P <= 256
DIRECT_MAX = (
    16  # partitions a level-2 job folds itself (four a lane); more go through level 1
)


def comb_floats(nt: int) -> int:
    """The key-split merge's LDS floats: four waves' state of ``nt`` tiles (two rounds)."""
    return (WAVES // 2) * nt * 16 * COMB_ROW


def dattn_lds(nt: int):
    """D-attn's LDS as two aliasing views (a union of them, built by the launch): a
    task's Q tiles and the merge jobs' reduction, then either each wave's P tiles
    and V^T (the chunk loop) or the key splits' merge tree (after it)."""

    @fx.struct
    class Loop:
        qs: fx.Array[
            fx.BFloat16, nt * 16 * Q_PITCH, 16
        ]  # the task's Q rows (its row tiles)
        red: fx.Array[fx.Float32, THREADS * 10, 16]  # the merge jobs' lane combine
        ring: fx.Array[
            fx.Int32, WAVES * RING_DEPTH * SLOT_WORDS, 16
        ]  # each wave's K / V chunks (DMA ring)
        ids: fx.Array[
            fx.Int32, IDS_CHUNKS * 16, 16
        ]  # the task's page ids, a block of chunks

    @fx.struct
    class Tree:
        qs: fx.Array[fx.BFloat16, nt * 16 * Q_PITCH, 16]
        red: fx.Array[fx.Float32, THREADS * 10, 16]
        comb: fx.Array[fx.Float32, comb_floats(nt), 16]

    return Loop, Tree


def lds_views(loop, tree):
    """``c`` entries of the two ``dattn_lds`` views (``.peek()`` of union members)."""
    return {"qs": loop.qs.ptr, "red": loop.red.ptr, "ring": loop.ring.ptr, "ids": loop.ids.ptr, "comb": tree.comb.ptr}  # fmt: skip


@flyc.jit
def run_dattn(c, o_flag=None):
    """D-attn's work on this CTA for ``c["S"]`` rows, ``c["QL"]`` a request.
    ``o_flag`` (a mailbox region): each level-2 merge raises its (request, KV
    head, row tile)'s flag once its o rows are out (a launch that goes on to K2)."""
    S, QL = c["S"], c["QL"]
    tid, bid, lane, wave = c["tid"], c["bid"], c["lane"], c["wave"]
    put, poll = c["put"], c["poll"]
    part_w, pflag, l2_w, l2flag = c["part"], c["pflag"], c["l2"], c["l2flag"]
    a = c["args"]
    R = S // QL
    ROWS = QL * GQA  # rows a (request, KV head): 64 or 8
    ROW_TILES = (ROWS + 15) // 16  # 4 or 1
    NT = c["NT"]  # row tiles a task (the build's; 4 spills registers, 2 fits)
    TASK_TILES = ROW_TILES // NT
    l16 = lane % 16
    g4 = lane // 16
    V8B = fx.Vector.make_type(8, fx.BFloat16)
    V2I = fx.Vector.make_type(2, fx.Int32)
    V4F = fx.Vector.make_type(4, fx.Float32)
    qs, ring, ids, comb, red = c["qs"], c["ring"], c["ids"], c["comb"], c["red"]
    wring = ring + wave * (RING_DEPTH * SLOT_WORDS)  # this wave's K / V ring
    table_stride, extra, window, part_in = (
        a["table_stride"],
        a["extra"],
        a["window"],
        a["part_in"],
    )
    lens64, rpi64 = a["lens64"], a["rpi64"]
    qk_scale = a["scale_bits"].bitcast(fx.Float32) * fx.Float32(LOG2E)
    r_k = rsrc(a["k_cache"])
    r_v = rsrc(a["v_cache"])
    r_pt = rsrc(a["page_table"], a["table_bytes"])
    r_q = rsrc(a["q"])
    r_part = rsrc(part_w.value)
    r_l2 = rsrc(l2_w.value)
    stamp(c, 0)

    def req_pool_of(r):
        """Request r's row of ``req_to_token`` (int32, or the low word of its int64)."""
        return fx.Int32(
            bo.buffer_load(
                rsrc(a["req_pool"]),
                (rpi64 != 0).select(2 * r, r),
                vec_width=1,
                dtype=T.i32,
            )
        )

    def seq_len_of(r):
        """Request r's keys (int32, or the low word of its int64), the step's rows included."""
        return (
            fx.Int32(
                bo.buffer_load(
                    rsrc(a["seq_lens"]),
                    (lens64 != 0).select(2 * r, r),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            + extra
        )

    def chunk_range(keys, qpos_lo):
        """The chunks [lo, hi) of a request with ``keys`` keys a window from its
        first row's position leaves unmasked (every chunk of a full one)."""
        hi = (keys + CHUNK - 1) // CHUNK
        ctx = keys - QL
        first = fx.max(
            ctx - window, fx.Int32(0)
        )  # the prefix's last ``window`` keys, then the block
        lo = (window > 0).select(first // CHUNK, fx.Int32(0))
        return lo, hi

    # the partitions: a request's keys from its first unmasked chunk (a sliding
    # window's start; 0 for a full layer), the longest such span over at most
    # MAX_PARTS of them
    def span_of(r):
        lo_r, hi_r = chunk_range(seq_len_of(r), 0)
        return hi_r - lo_r

    max_span = span_of(fx.Int32(0))
    total_span = max_span
    for rr in range_constexpr(1, R):
        sp = span_of(fx.Int32(rr))
        max_span = fx.max(max_span, sp)
        total_span = total_span + sp

    # the partition (chunks a task) from the step's total chunk work, so that the
    # step has one task a CTA whatever the request mix (a task costs ~12 us beside
    # its chunks of ~1.8 us: its Q and page ids, the first chunk's exposed
    # latency, the key-split tree and the publish; eight 95k requests at 128
    # partitions each were 15 tasks a CTA, 212 us), the per-request rounding
    # checked so that no CTA gets a second task, and at least what keeps the
    # longest request within MAX_PARTS
    def tasks_of(pc_):
        n = fx.Int32(0)
        for rr in range_constexpr(R):
            lo_r, hi_r = chunk_range(seq_len_of(fx.Int32(rr)), 0)
            n = n + ((hi_r - lo_r) + pc_ - 1) // pc_
        return n * (LOCAL_KV_HEADS * TASK_TILES)

    per_cta = (total_span * (LOCAL_KV_HEADS * TASK_TILES) + BLOCKS - 1) // BLOCKS
    floor_ = (max_span + MAX_PARTS - 1) // MAX_PARTS
    pc = fx.max(fx.max(per_cta, floor_), fx.Int32(1))
    for _ in range_constexpr(3):
        pc = (tasks_of(pc) > BLOCKS).select(pc + 1, pc)
    part = (part_in > 0).select(part_in, pc * CHUNK)
    n_parts = (
        max_span * CHUNK + part - 1
    ) // part  # partitions a (request, head) at most

    def n_parts_of(r):
        """Request r's partitions (its span's, the rest of the task slots idle)."""
        lo, hi = chunk_range(seq_len_of(r), 0)
        return ((hi - lo) * CHUNK + part - 1) // part

    def pflag_at(r, h, rt, p):
        """Pair index of the flag of the task holding row tile ``rt`` of (request r,
        head h) over partition p: a task's own (its first tile's), not the
        partition's: the tasks of a partition's row tiles finish apart (a merge
        that took the first one's flag read the other tiles' rows of the last layer)."""
        return ((r * LOCAL_KV_HEADS + h) * ROW_TILES + (rt // NT) * NT) * MAX_PARTS + p

    def part_row(r, h, p, row):
        """Float index of (request r, head h, partition p)'s row record."""
        return (((r * LOCAL_KV_HEADS + h) * MAX_PARTS + p) * ROWS + row) * PART_ROW

    def l2_row(r, h, rt, g, row16):
        """Float index of (request r, head h, row tile rt, group g)'s level-2 row record."""
        return (
            (((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * GROUPS + g) * 16 + row16
        ) * PART_ROW

    def fold(m, l, ov, m2, l2, v2, ok):
        """(m, l, O) with a partial folded in (``ok``: it exists)."""
        mn = fx.max(m, m2)
        mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
        a1 = ok.select(hw_exp2(m - mg), fx.Float32(1.0))
        a2 = ok.select(hw_exp2(m2 - mg), fx.Float32(0.0))
        return ok.select(mn, m), l * a1 + l2 * a2, ov * a1 + v2 * a2

    # ============================================ attention tasks
    def attn_task(r, h, rt0, p):
        """Row tiles rt0 .. rt0 + NT of (request r, KV head h) over partition p: one
        K / V load and one V^T staging a chunk serve every tile of the task."""
        keys = seq_len_of(r)
        ctx = keys - QL
        ks = wave  # this wave's key split
        # Q: rows 16 (rt0 + it) + i = 8 qpos + hg of (request r, head h) into LDS, 16 B a thread
        if tid < NT * 16 * HEAD_DIM // 8:
            i_t = tid // (16 * HEAD_DIM // 8)
            q_row = (tid % (16 * HEAD_DIM // 8)) // (HEAD_DIM // 8)
            row = (rt0 + i_t) * 16 + q_row
            d0 = (tid % (HEAD_DIM // 8)) * 8
            t = r * QL + fx.min(row, ROWS - 1) // GQA
            qv = fx.Vector(
                bo.buffer_load(
                    r_q,
                    (t * Q_K + (h * GQA + fx.min(row, ROWS - 1) % GQA) * HEAD_DIM + d0)
                    // 2,
                    vec_width=4,
                    dtype=T.i32,
                )  # fmt: skip
            )
            fx.ptr_store(
                qv.bitcast(fx.BFloat16), qs + ((i_t * 16 + q_row) * Q_PITCH + d0)
            )
        gpu.barrier()
        # each tile's A operands of this lane: row l16, dims 8 g4 + 32 step
        qa = [
            [
                fx.Vector(
                    fx.ptr_load(
                        qs + ((it * 16 + l16) * Q_PITCH + 32 * step + 8 * g4),
                        result_type=V8B,
                    )
                )  # fmt: skip
                for step in range(D_STEPS)
            ]
            for it in range(NT)
        ]
        # S^T tiles: K is the MFMA's A operand (key l16, dims 8 g4 ..), Q^T its B
        # (row l16): a lane holds row l16 of each row tile and keys 4 g4 .. 4 g4 + 3 of
        # each key tile, so a row's running statistics are one lane's (four lanes
        # share a row), and its P values are packed in registers straight into the
        # PV MFMA's A operand (keys of tile 0 then tile 1; the V^T reads use the
        # same key order). The reference max of a row moves only when a chunk's max
        # exceeds it by TAU (lazy rescale, every row of the wave at once).
        r_lane = [(rt0 + it) * 16 + l16 for it in range(NT)]
        qpos = [fx.min(fx.min(r_lane[it], ROWS - 1) // GQA, QL - 1) for it in range(NT)]
        row_live = [r_lane[it] < ROWS for it in range(NT)]
        m_ref = [
            fx.Float32(NEG) for _ in range(NT)
        ]  # the row's reference max (its running max, lazily)
        l_lane = [
            fx.Float32(0.0) for _ in range(NT)
        ]  # this lane's keys' share of the row sum
        accs = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(D_TILES)]
            for _ in range(NT)
        ]
        lo, hi = chunk_range(keys, 0)
        c_lo = fx.max(lo, lo + p * (part // CHUNK))
        c_hi = fx.min(hi, lo + (p + 1) * (part // CHUNK))

        # The chunk loop: a wave's 32 keys of a chunk are two K pages and two V pages
        # (1 KB each, one head), landed in LDS by DMA (16 B a lane; chunk-major: dims
        # 16 c .. of key k at bytes 256 c + 16 k, conflict-free for both the K reads
        # and the transposed V reads) in the wave's ring slot; two chunks are in
        # flight while one computes. (A VGPR-destination software pipeline was
        # drained by the compiler at every loop head: 6 us a chunk at 300k keys.)
        # The page ids of a block of IDS_CHUNKS chunks are staged in LDS first.
        assert KEY_TILES == 2 and V_STEPS == 1
        c_last = fx.max(c_hi - 1, c_lo)
        l16d, c16d = (
            lane % 16,
            lane // 16,
        )  # the DMA lane: key l16d, dims chunk c16d of a page
        # the transposed V read: this lane addresses key 4 g4 + (l16 / 2) % 4 of page (l16 / 2) / 4
        v_page = (l16 // 2) // 4
        v_key = 4 * g4 + (l16 // 2) % 4

        def issue(cc, sl, b_lo):
            """Chunk cc's four pages -> ring slot sl (always eight loads: the counts)."""
            slot = wring + sl * SLOT_WORDS
            for kt in range_constexpr(KEY_TILES):
                pg = fx.ptr_load(ids + ((cc - b_lo) * 16 + 2 * ks + kt))  # uniform
                for hf in range_constexpr(PAGE_DMAS):
                    voff = ((pg * 16 + l16d) * LOCAL_KV_HEADS + h) * HEAD_DIM + (
                        c16d + 4 * hf
                    ) * 16
                    lds_dma_16(r_k, slot + (kt * PAGE_WORDS + hf * 256), voff)
                    lds_dma_16(
                        r_v,
                        slot + ((KEY_TILES + kt) * PAGE_WORDS + kt * V_SKEW + hf * 256),
                        voff,
                    )

        def quad_max(v):
            """The max over the four lanes sharing a row (l16, l16 + 16, + 32, + 48)."""
            for off in (32, 16):
                x, y = permlane_swap(off, v.bitcast(fx.Int32), v.bitcast(fx.Int32))
                v = fx.max(x.bitcast(fx.Float32), y.bitcast(fx.Float32))
            return v

        def quad_sum(v):
            for off in (32, 16):
                x, y = permlane_swap(off, v.bitcast(fx.Int32), v.bitcast(fx.Int32))
                v = x.bitcast(fx.Float32) + y.bitcast(fx.Float32)
            return v

        n_blk = (c_hi - c_lo + IDS_CHUNKS - 1) // IDS_CHUNKS
        stamp(c, 3)  # a task's chunk loop begins (the last task's stamps survive)
        for b in range(0, n_blk):
            b_lo = c_lo + b * IDS_CHUNKS
            b_hi = fx.min(b_lo + IDS_CHUNKS, c_hi)
            b_last = b_hi - 1
            gpu.barrier()  # the previous block's ids are consumed
            rpi = req_pool_of(r)
            if tid < IDS_CHUNKS * 16:
                idx = tid
                cc_i = fx.min(b_lo + idx // 16, c_last)
                slot0 = fx.Int32(bo.buffer_load(r_pt, rpi * table_stride + (cc_i * 16 + idx % 16) * 16, vec_width=1, dtype=T.i32))  # fmt: skip
                fx.ptr_store(slot0 // 16, ids + idx)
            gpu.barrier()
            issue(b_lo, 0, b_lo)
            issue(fx.min(b_lo + 1, b_last), 1, b_lo)
            for cc in range(b_lo, b_hi):
                sl = (cc - b_lo) % 2
                slot = wring + sl * SLOT_WORDS
                key0 = cc * CHUNK + ks * KEYS_PER_WAVE
                rocdl.s_waitcnt(
                    vmcnt=2 * KEY_TILES * PAGE_DMAS
                )  # chunk cc's pages landed (cc + 1's in flight)
                # the chunk's K operands (tile kt, step: key l16, dims 32 step + 8 g4)
                kb = []
                for kt in range_constexpr(KEY_TILES):
                    for step in range_constexpr(D_STEPS):
                        widx = (
                            slot
                            + kt * PAGE_WORDS
                            + ((2 * step + g4 // 2) * 256 + l16 * 16 + (g4 % 2) * 8)
                            // 4
                        )
                        w = fx.Vector(fx.ptr_load(widx, result_type=V2I))
                        kb.append(fp8_to_bf16x8(w[0], w[1]))
                # ---- S^T: tile (it, kt): row l16, keys 4 g4 + e of key tile kt; scaled and masked
                inner = (cc != c_last) & (cc != lo)  # a chunk with no masked key
                sc = []  # [it][kt * 4 + e]
                mx = []
                for it in range_constexpr(NT):
                    vals = []
                    for kt in range_constexpr(KEY_TILES):
                        st = fx.Vector.filled(4, 0.0, fx.Float32)
                        for step in range_constexpr(D_STEPS):
                            st = mfma_bf16(kb[kt * D_STEPS + step], qa[it][step], st)
                        for e in range_constexpr(4):
                            kidx = key0 + kt * TILE + 4 * g4 + e
                            ok = row_live[it] & (inner | ((kidx < keys) & ((window == 0) | ((ctx - kidx) <= window))))  # fmt: skip
                            vals.append(
                                ok.select(fx.Float32(st[e]) * qk_scale, fx.Float32(NEG))
                            )
                    sc.append(vals)
                    m8 = vals[0]
                    for j in range_constexpr(1, 2 * KEY_TILES * 4 // 2):
                        m8 = fx.max(m8, vals[j])
                    mx.append(quad_max(m8))
                # ---- the lazy rescale: when any row's chunk max exceeds its reference by TAU,
                # every row takes its new max and its sum and O are rescaled (selects:
                # the step is wave-uniform, the values are loop-carried)
                need = mx[0] > m_ref[0] + TAU
                for it in range_constexpr(1, NT):
                    need = need | (mx[it] > m_ref[it] + TAU)
                move = ballot(need) != 0
                alphas_acc = []
                for it in range_constexpr(NT):
                    m_new = move.select(fx.max(m_ref[it], mx[it]), m_ref[it])
                    m_g = (m_new > fx.Float32(NO_MAX)).select(m_new, fx.Float32(0.0))
                    alpha = hw_exp2(
                        m_ref[it] - m_g
                    )  # 0 from NEG (a first chunk), 1 when nothing moved
                    l_lane[it] = l_lane[it] * alpha
                    m_ref[it] = m_new
                    # O rows 4 g4 + e take the alpha of row 4 g4 + e: from lane 4 g4 + e (its g4 = 0 copy)
                    a_rows = [
                        fx.Int32(
                            rocdl.ds_bpermute(
                                T.i32, (4 * g4 + e) * 4, alpha.bitcast(fx.Int32)
                            )
                        ).bitcast(fx.Float32)
                        for e in range(4)
                    ]
                    al = fx.Vector.from_elements(a_rows, fx.Float32)
                    accs[it] = [accs[it][dt] * al for dt in range(D_TILES)]
                # ---- P = exp2(S - m) as the PV A operand (eight bf16 a lane), the lane's row sums
                p_ops = []
                for it in range_constexpr(NT):
                    m_g = (m_ref[it] > fx.Float32(NO_MAX)).select(
                        m_ref[it], fx.Float32(0.0)
                    )
                    ps = [
                        hw_exp2(sc[it][j] - m_g) for j in range(2 * KEY_TILES * 4 // 2)
                    ]
                    acc_l = ps[0]
                    for j in range_constexpr(1, len(ps)):
                        acc_l = acc_l + ps[j]
                    l_lane[it] = l_lane[it] + acc_l
                    p_ops.append(
                        fx.Vector.from_elements(ps, fx.Float32).to(fx.BFloat16)
                    )
                # ---- PV: V^T operands by transposed LDS reads (dim tile dt: lane (g4, l16) gets
                # dim 16 dt + l16 at keys 4 g4 .. of page 0 then of page 1)
                b_v = []
                for dt in range_constexpr(D_TILES):
                    widx = slot + ((KEY_TILES + v_page) * PAGE_WORDS + v_page * V_SKEW) + (dt * 256 + v_key * 16 + (l16 % 2) * 8) // 4  # fmt: skip
                    w = fx.Vector(
                        rocdl_raw.ds_read_tr8_b64(
                            T.vec(2, T.i32), widx.llvm_ptr
                        ).result,
                        (2,),
                        fx.Int32,
                    )  # gfx950 ds_read_b64_tr_b8
                    b_v.append(fp8_to_bf16x8(w[0], w[1]))
                accs = [
                    [
                        mfma_bf16(p_ops[it], b_v[dt], accs[it][dt])
                        for dt in range(D_TILES)
                    ]
                    for it in range(NT)
                ]
                rocdl.s_waitcnt(lgkmcnt=0)  # every read of the slot is done
                issue(fx.min(cc + 2, b_last), sl, b_lo)  # chunk cc + 2 into this slot
                if cc == b_lo:
                    stamp(c, 4)  # the first chunk done
            rocdl.s_waitcnt(
                vmcnt=0
            )  # the tail's repeated loads land before the LDS is reused
        stamp(c, 5)  # the chunk loop done
        gpu.barrier()  # every wave is done with the ring: the merge tree reuses the LDS

        # ---- the key splits merge in LDS in one round: every wave writes its
        # (m, l, O) rows (a (split, tile, row) record is COMB_ROW floats, O from
        # O_OFF); then each wave owns 2 NT of the task's rows, folds the eight splits
        # of them (a lane: one row, DPL dims) and publishes them with vector stores
        def comb_at(slot, it, row):
            return comb + ((slot * NT + it) * 16 + row) * COMB_ROW

        RPW = 2 * NT  # rows a wave: NT 16 / 8
        LPR = 64 // RPW  # lanes a row: 16 or 32
        DPL = HEAD_DIM // LPR  # dims a lane: 8 or 4
        VD = fx.Vector.make_type(DPL, fx.Float32)
        trow = wave * RPW + lane // LPR  # the task row (0 .. NT 16)
        it_f, row_f = trow // 16, trow % 16
        dd = (lane % LPR) * DPL
        m_f = fx.Float32(NEG)
        l_f = fx.Float32(0.0)
        o_f = fx.Vector.filled(DPL, 0.0, fx.Float32)
        l_rows = [quad_sum(l_lane[it]) for it in range(NT)]
        for rnd in range_constexpr(2):
            # the round's four waves write their (m, l, O) rows (the tree holds four)
            if wave // (WAVES // 2) == rnd:
                for it in range_constexpr(NT):
                    if g4 == 0:
                        base = comb_at(ks % (WAVES // 2), it, l16)
                        fx.ptr_store(m_ref[it], base)
                        fx.ptr_store(l_rows[it], base + 1)
                    for e in range_constexpr(4):
                        base = comb_at(ks % (WAVES // 2), it, 4 * g4 + e)
                        for dt in range_constexpr(D_TILES):
                            fx.ptr_store(
                                fx.Float32(accs[it][dt][e]),
                                base + (O_OFF + dt * 16 + l16),
                            )
            gpu.barrier()
            for src in range_constexpr(WAVES // 2):
                base = comb_at(src, it_f, row_f)
                m2 = fx.ptr_load(base)
                l2 = fx.ptr_load(base + 1)
                v2 = fx.Vector(fx.ptr_load(base + (O_OFF + dd), result_type=VD))
                mn = fx.max(m_f, m2)
                mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
                a1 = hw_exp2(m_f - mg)
                a2 = hw_exp2(m2 - mg)
                m_f, l_f, o_f = mn, l_f * a1 + l2 * a2, o_f * a1 + v2 * a2
            gpu.barrier()  # the tree is rewritten by the next round / the next task
        # ---- the task's partial rows: plain fp32 words, then the flag
        row = rt0 * 16 + trow
        if row < ROWS:
            base = part_row(r, h, p, row)
            if lane % LPR == 0:
                bo.buffer_store(
                    fx.Vector.from_elements([m_f, l_f], fx.Float32),
                    r_part,
                    base,
                    cache_modifier=CM_DEV,
                )
            for hf in range_constexpr(DPL // 4):
                quad = fx.Vector.from_elements(
                    [fx.Float32(o_f[4 * hf + j]) for j in range(4)], fx.Float32
                )
                bo.buffer_store(
                    quad, r_part, base + O_OFF + dd + 4 * hf, cache_modifier=CM_DEV
                )
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(pflag, pflag_at(r, h, rt0, p), fx.Int32(1))

    # the tasks, request-major over the requests' own partition counts (a short
    # request has few): job -> (request, head, tile pair, partition) through the
    # requests' cumulative task counts, so a CTA iterates real tasks only
    counts = [
        n_parts_of(fx.Int32(rr)) * (LOCAL_KV_HEADS * TASK_TILES) for rr in range(R)
    ]
    starts = [fx.Int32(0)]
    for rr in range_constexpr(1, R):
        starts.append(starts[-1] + counts[rr - 1])
    total = starts[-1] + counts[-1]
    for job in range(bid, total, BLOCKS):
        r = fx.Int32(0)
        for rr in range_constexpr(1, R):
            r = (job >= starts[rr]).select(fx.Int32(rr), r)
        base = starts[0]
        for rr in range_constexpr(1, R):
            base = (r == rr).select(starts[rr], base)
        np_r = n_parts_of(r)
        local = job - base
        p = local % np_r
        rt0 = ((local // np_r) % TASK_TILES) * NT
        h = (local // (np_r * TASK_TILES)) % LOCAL_KV_HEADS
        attn_task(r, h, rt0, p)
    stamp(c, 1)  # the attention tasks done

    # ============================================ merge jobs
    # a thread: row 16 rt + tid / 32, dims 8 ((tid / 4) % 8) + 64 pass .., partition lane tid % 4
    row16 = tid // 32
    dch = (tid // 4) % 8
    pl = tid % 4
    D_PASSES = HEAD_DIM // 64
    mbid = (
        BLOCKS - 1 - bid
    )  # the last CTAs: the task loop's extra tasks sit on the first
    n_l1 = R * LOCAL_KV_HEADS * ROW_TILES * GROUPS
    n_l2 = R * LOCAL_KV_HEADS * ROW_TILES

    def load_rec(r_src, base, d0):
        """A record's (m, l) and this thread's 8 dims of O from ``d0`` (plain words)."""
        ml = fx.Vector(
            bo.buffer_load(r_src, base, vec_width=2, dtype=T.f32, cache_modifier=CM_DEV)
        )
        o0 = fx.Vector(
            bo.buffer_load(
                r_src,
                base + O_OFF + d0,
                vec_width=4,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )
        o1 = fx.Vector(
            bo.buffer_load(
                r_src,
                base + O_OFF + d0 + 4,
                vec_width=4,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )
        return ml, o0, o1

    def fold_rec(m, l, o0, o1, rec, ok):
        ml, p0, p1 = rec
        m2, l2 = fx.Float32(ml[0]), fx.Float32(ml[1])
        mn = fx.max(m, m2)
        mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
        a1 = ok.select(hw_exp2(m - mg), fx.Float32(1.0))
        a2 = ok.select(hw_exp2(m2 - mg), fx.Float32(0.0))
        return ok.select(mn, m), l * a1 + l2 * a2, o0 * a1 + p0 * a2, o1 * a1 + p1 * a2

    def combine_lanes(m, l, o0, o1):
        """The four partition lanes of a (row, 8 dims) fold in LDS: every lane folds
        the four in lane order (the same result on each; lane 0 stores)."""
        slot = red + tid * 10
        fx.ptr_store(m, slot)
        fx.ptr_store(l, slot + 1)
        for j in range_constexpr(4):
            fx.ptr_store(fx.Float32(o0[j]), slot + (2 + j))
            fx.ptr_store(fx.Float32(o1[j]), slot + (6 + j))
        gpu.barrier()
        base = red + (tid - pl) * 10
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)
        for src in range_constexpr(MERGE_LANES):
            s2 = base + src * 10
            m2 = fx.ptr_load(s2)
            l2 = fx.ptr_load(s2 + 1)
            p0 = fx.Vector.from_elements(
                [fx.ptr_load(s2 + (2 + j)) for j in range(4)], fx.Float32
            )
            p1 = fx.Vector.from_elements(
                [fx.ptr_load(s2 + (6 + j)) for j in range(4)], fx.Float32
            )
            mn = fx.max(m, m2)
            mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
            a1 = hw_exp2(m - mg)
            a2 = hw_exp2(m2 - mg)
            m, l, o0, o1 = mn, l * a1 + l2 * a2, o0 * a1 + p0 * a2, o1 * a1 + p1 * a2
        gpu.barrier()  # ``red`` is reused by the CTA's next job
        return m, l, o0, o1

    def merge_l1(r, h, rt, g):
        """Level-1 job: partitions [g PPG, (g + 1) PPG) of (request r, head h) for
        row tile rt, folded four lanes a (row, 8 dims) -> a level-2 record (two
        passes over the dims)."""
        np_r = n_parts_of(r)
        ppg = (np_r + GROUPS - 1) // GROUPS
        p_lo = g * ppg
        cnt = fx.max(fx.min(np_r - p_lo, ppg), fx.Int32(0))
        poll_flags(c, [(pflag, pflag_at(r, h, rt, p_lo), cnt)])
        row = fx.min(rt * 16 + row16, ROWS - 1)
        for dp in range_constexpr(D_PASSES):
            d0 = (dch + 8 * dp) * 8
            m = fx.Float32(NEG)
            l = fx.Float32(0.0)
            o0 = fx.Vector.filled(4, 0.0, fx.Float32)
            o1 = fx.Vector.filled(4, 0.0, fx.Float32)
            # four partitions a lane in flight a round
            for k0 in range(0, (ppg + MERGE_LANES * 4 - 1) // (MERGE_LANES * 4)):
                ps = [p_lo + (k0 * 4 + k) * MERGE_LANES + pl for k in range(4)]
                recs = [
                    load_rec(
                        r_part, part_row(r, h, fx.min(pk, p_lo + cnt - 1), row), d0
                    )
                    for pk in ps
                ]
                for k in range_constexpr(4):
                    m, l, o0, o1 = fold_rec(m, l, o0, o1, recs[k], ps[k] < p_lo + cnt)
            m, l, o0, o1 = combine_lanes(m, l, o0, o1)
            if (pl == 0) & (rt * 16 + row16 < ROWS):
                base = l2_row(r, h, rt, g, row16)
                if dch == 0:
                    if const_expr(dp == 0):
                        bo.buffer_store(
                            fx.Vector.from_elements([m, l], fx.Float32),
                            r_l2,
                            base,
                            cache_modifier=CM_DEV,
                        )
                bo.buffer_store(o0, r_l2, base + O_OFF + d0, cache_modifier=CM_DEV)
                bo.buffer_store(o1, r_l2, base + O_OFF + d0 + 4, cache_modifier=CM_DEV)
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(
                l2flag,
                ((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * GROUPS + g,
                fx.Int32(1),
            )

    def merge_l2(r, h, rt):
        """Level-2 job: the groups' records (or, with at most DIRECT_MAX partitions,
        the partitions themselves) of (request r, head h, row tile rt) -> o rows
        (two passes over the dims)."""
        np_r = n_parts_of(r)
        direct = np_r <= DIRECT_MAX
        row = fx.min(rt * 16 + row16, ROWS - 1)
        # the source records: the partitions themselves (at most GROUPS of them), or
        # the level-1 records; one resource over the selected base address
        if direct:
            poll_flags(c, [(pflag, pflag_at(r, h, rt, 0), np_r)])
        else:
            poll_flags(
                c,
                [
                    (
                        l2flag,
                        ((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * GROUPS,
                        GROUPS,
                    )
                ],
            )
        r_src = rsrc(direct.select(part_w.value, l2_w.value))
        for dp in range_constexpr(D_PASSES):
            d0 = (dch + 8 * dp) * 8
            m = fx.Float32(NEG)
            l = fx.Float32(0.0)
            o0 = fx.Vector.filled(4, 0.0, fx.Float32)
            o1 = fx.Vector.filled(4, 0.0, fx.Float32)
            for k in range_constexpr(DIRECT_MAX // MERGE_LANES):
                g = k * MERGE_LANES + pl  # a partition (direct) or a group (g < GROUPS)
                idx = direct.select(
                    part_row(r, h, fx.min(g, np_r - 1), row),
                    l2_row(r, h, rt, fx.min(g, GROUPS - 1), row16),
                )
                ok = direct.select(g < np_r, g < GROUPS)
                m, l, o0, o1 = fold_rec(m, l, o0, o1, load_rec(r_src, idx, d0), ok)
            m, l, o0, o1 = combine_lanes(m, l, o0, o1)
            if (pl == 0) & (rt * 16 + row16 < ROWS):
                inv = fx.Float32(1.0) / l
                t = r * QL + (rt * 16 + row16) // GQA
                ov = fx.Vector.from_elements([fx.Float32(o0[j]) for j in range(4)] + [fx.Float32(o1[j]) for j in range(4)], fx.Float32)  # fmt: skip
                bo.buffer_store(
                    (ov * inv).to(fx.BFloat16),
                    rsrc(a["o"]),
                    t * Q_K + (h * GQA + (rt * 16 + row16) % GQA) * HEAD_DIM + d0,
                    cache_modifier=CM_DEV,
                )
        if const_expr(o_flag is not None):
            # the (request, head, row tile)'s o rows are out: its flag, for the og tasks
            release_stores()
            gpu.barrier()
            if tid == 0:
                put(o_flag, (r * LOCAL_KV_HEADS + h) * ROW_TILES + rt, fx.Int32(1))

    for j in range(mbid, n_l1, BLOCKS):
        g = j % GROUPS
        rt = (j // GROUPS) % ROW_TILES
        h = (j // (GROUPS * ROW_TILES)) % LOCAL_KV_HEADS
        r = j // (GROUPS * ROW_TILES * LOCAL_KV_HEADS)
        if (
            n_parts_of(r) > DIRECT_MAX
        ):  # else the level-2 job folds the partitions itself
            merge_l1(r, h, rt, g)
    if mbid < n_l2:
        rt = mbid % ROW_TILES
        h = (mbid // ROW_TILES) % LOCAL_KV_HEADS
        r = mbid // (ROW_TILES * LOCAL_KV_HEADS)
        merge_l2(r, h, rt)
    stamp(c, 2)  # the merge done (end)


# ---------------------------------------------------------------- the launch
import msgspec  # noqa: E402
from flydsl.expr.typing import Int32, Int64  # noqa: E402

from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.common.epoch import kernel_symbol  # noqa: E402
from sgl_amd_mono.common.plan import kernel_sources, key_tuple  # noqa: E402
from sgl_amd_mono.common.sync import Mailbox, sreg  # noqa: E402
from sgl_amd_mono.muse_spark.gfx950 import compat  # noqa: E402
from sgl_amd_mono.muse_spark.gfx950.device import step_tag  # noqa: E402

BRIDGE = compat.install()
SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)


class DAttnBuild(msgspec.Struct, frozen=True):
    """A D-attn build: the step's row count and rows a request (the draft block),
    the row tiles a task shares its K / V loads over, the fault injection site."""

    tokens: int
    q_len: int = 8
    nt: int = 2
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False


def _attn_views(lds):
    return lds.loop.peek(), lds.tree.peek()


def dattn_context(
    key: DAttnBuild, lds, args: dict, scratch, tag, layout: dict, layer=None
) -> dict:
    tid = fx.thread_idx.x
    mb = Mailbox(tag)
    c = {
        "S": key.tokens,
        "QL": key.q_len,
        "NT": min(key.nt, (attn_rows(key.q_len) + 15) // 16),
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        **lds_views(*_attn_views(lds)),
        "put": mb.put,
        "poll": mb.poll,
        "args": args,
        "trace_on": key.trace,
        "layer": layer,
    }
    for region in ("attn_part", "attn_pflag", "attn_l2", "attn_l2flag", "attn_trace"):
        c[region[5:]] = sreg(
            scratch, layout[region][0], region
        )  # "attn_part" -> c["part"]
    return c


def build_dattn(key: DAttnBuild):
    """``@flyc.jit`` launcher of D-attn for ``key.tokens`` rows, ``key.q_len`` a request.
    Arguments: ``q`` bf16 [S, Q_K] (normed, rotated), ``o`` bf16 [S, Q_K], the layer's
    fp8 ``k_cache`` / ``v_cache`` [slots, 1, 128], ``req_to_token`` int32 [rows,
    ``table_stride``] (``table_bytes`` its size), ``req_pool`` (int32 / int64 [R]:
    each request's row; ``rpi64``), ``seq_lens`` (the prefix lengths; ``lens64``),
    ``extra`` (QL: the block's keys past the prefix), ``window``, ``part_in`` (0: sized
    here), ``scale_bits``, ``scratch``, ``epoch``, ``layer``."""
    S, QL, NT = key.tokens, key.q_len, key.nt
    assert QL == 8 and S % QL == 0 and 1 <= S <= MAX_TOKENS
    row_tiles = (attn_rows(QL) + 15) // 16
    NT = min(NT, row_tiles)
    assert row_tiles % NT == 0, (row_tiles, NT)
    name = kernel_symbol(
        "muse_spark_dattn", s=S, q=QL, n=NT, f=key.fault_site, t=int(key.trace)
    )
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout = dattn_scratch_layout(S, QL)
    Loop, Tree = dattn_lds(NT)

    @fx.union
    class Lds:
        loop: Loop  # type: ignore[valid-type]
        tree: Tree  # type: ignore[valid-type]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dattn(
        q: Int64, o: Int64, k_cache: Int64, v_cache: Int64, page_table: Int64, table_stride: Int32, table_bytes: Int32,
        req_pool: Int64, rpi64: Int32, seq_lens: Int64, lens64: Int32, extra: Int32, window: Int32, part_in: Int32,
        scale_bits: Int32, scratch: Int64, epoch: Int64, layer: Int32,
    ):  # fmt: skip
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
            lds = fx.SharedAllocator().allocate(Lds)
            args = {
                "q": q, "o": o, "k_cache": k_cache, "v_cache": v_cache, "page_table": page_table,
                "table_stride": table_stride, "table_bytes": table_bytes, "req_pool": req_pool, "rpi64": rpi64,
                "seq_lens": seq_lens, "lens64": lens64, "extra": extra, "window": window, "part_in": part_in,
                "scale_bits": scale_bits,
            }  # fmt: skip
            c = dattn_context(
                key, lds, args, scratch, step_tag(epoch, layer), layout, layer=layer
            )
            run_dattn(c)

    @flyc.jit
    def launch(
        q: Int64, o: Int64, k_cache: Int64, v_cache: Int64, page_table: Int64, table_stride: Int32, table_bytes: Int32,
        req_pool: Int64, rpi64: Int32, seq_lens: Int64, lens64: Int32, extra: Int32, window: Int32, part_in: Int32,
        scale_bits: Int32, scratch: Int64, epoch: Int64, layer: Int32, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        dattn(
            q, o, k_cache, v_cache, page_table, table_stride, table_bytes, req_pool, rpi64, seq_lens, lens64,
            extra, window, part_in, scale_bits, scratch, epoch, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
