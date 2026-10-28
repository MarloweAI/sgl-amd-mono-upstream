# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from ROCm/aiter#6173 at 56d6390def (Apache-2.0):
# aiter/ops/flydsl/kernels/glm5_mono/glm/layout.py

"""Compile-time storage layout and CTA schedule for the GLM-5 MonoKernel."""

from .config import (
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    QKV_A_ROWS,
    TOP_K,
    V_DIM,
)
from .tiles import (
    BLOCKS,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
)

INDEX_HEADS = 32
INDEX_DIM = 128
# LM-head rows a tail task computes
HEAD_TILE = 16
INDEX_Q_ROWS = INDEX_HEADS * INDEX_DIM
INDEX_TILE = 16
INDEX_KEYS_PER_TASK = 64
# a long row's paged top-k runs on up to this many slices (CTAs) per row, each with
# staging room for this many threshold-bin candidates
INDEX_SEL_SLICES = 16
INDEX_CAND_STAGE_CAP = 16384
# keys a select chunk (a CTA's 512 threads x 8), and the chunks past which a row selects
# on up to INDEX_SEL_SLICES CTAs: up to 8 chunks the hand-offs cost what the slices save
INDEX_SEL_CHUNK = 4096
INDEX_SEL_PAR_CHUNKS = 8


def select_slices(bound):
    """(slices, chunks a slice) of a row of ``bound`` > 0 keys whose top-k runs on
    several CTAs: runs of whole chunks in position order, at most INDEX_SEL_SLICES, none
    empty. The kernel calls it on traced Int32 and the CPU test on ints."""
    chunks = (bound + INDEX_SEL_CHUNK - 1) // INDEX_SEL_CHUNK
    per = (chunks + INDEX_SEL_SLICES - 1) // INDEX_SEL_SLICES
    return (chunks + per - 1) // per, per


DCP_SUMMARY_PAIRS = KV_LORA // 2 + 2


def split_acc_head(head_group, lane_group: int, element: int):
    """Map one split-PV MFMA row to its local attention head."""

    return head_group * WAVES + lane_group * 4 + element


def split_score_column(wave: int, lane: int):
    """Locate this wave's head column in a 16x16 MFMA score tile."""

    return wave + 16 * ((lane % 16) // 4)


def fp8_kv_upper_pair_lane(lane):
    return (lane & -4) + 2


def fp8_pe_upper_pair_lane(lane):
    return (lane & -2) + 1


N_QKV_A = QKV_A_ROWS // QKV_A_TILE
N_ROW_TILES = HIDDEN // ROW_TILE
N_ROUTER = N_EXPERTS // ROUTER_TILE
XQ_BLOCKS = HIDDEN // 128
XQ_GROUPS = HIDDEN // 32
XQ_WAVES = (XQ_BLOCKS + N_ROUTER - 1) // N_ROUTER
assert XQ_WAVES * 4 <= WAVES


def dn_tile(samples: int, expert_mxfp4: bool = False) -> int:
    """Return rows per expert-down/FFN-reduce task for the tuned schedule."""

    return 32 if samples == 1 or (samples > 4 and expert_mxfp4) else HIDDEN // BLOCKS


def sparse_keys_per_task(samples: int, heads: int = WAVES) -> int:
    """Use narrower sparse tiles when samples or attention heads fill LDS."""

    return 32 if heads > 2 * WAVES else 64


def sample_wave_batches(samples: int) -> int:
    return (samples + WAVES - 1) // WAVES


def ug_task_rounds(inter: int) -> int:
    return (inter + BLOCKS - 1) // BLOCKS


def down_prefetch_batch(samples: int, native_fp4_mfma: bool = False) -> int:
    if samples <= 4:
        return 9
    if native_fp4_mfma:
        return 4 if samples <= 6 else 3
    return 8


def down_x_words(
    samples: int, inter: int, expert_mxfp4: bool, native_fp4_mfma: bool = False
) -> int:
    return (
        samples * MOE_SLOTS * inter // (4 if native_fp4_mfma or not expert_mxfp4 else 2)
    )


def dcp_summary_index(source, sample, tile, item, samples, n_uv):
    return ((source * samples + sample) * n_uv + tile) * DCP_SUMMARY_PAIRS + item


def dcp_uv_owner(tile, output_heads, rank):
    return tile // (V_DIM // UV_TILE) // output_heads == rank


def dcp_local_uv_tile(tile, output_heads):
    return tile % (output_heads * V_DIM // UV_TILE)


def ug_split(samples: int, inter: int = INTER):
    """Return the balanced up/gate leftover split for batches two and four."""

    full_tiles, remainder = divmod((samples * TOP_K + 1) * (inter // UG_TILE), BLOCKS)
    if samples not in (2, 4) or remainder == 0 or BLOCKS % remainder:
        return None
    segments = BLOCKS // remainder
    if (HIDDEN // 128) % segments or (HIDDEN // 128) // segments > WAVES // 2:
        return None
    return full_tiles, segments


def _align(size: int, alignment: int = 256) -> int:
    return (size + alignment - 1) // alignment * alignment


def layout(
    samples: int,
    heads: int,
    npes: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    inter: int = INTER,
    output_heads: int | None = None,
    dcp_size: int = 1,
    native_fp4_mfma: bool = False,
    tokens: bool = False,
    head_vocab: int = 0,
):
    """Return byte offsets for per-rank scratch and symmetric peer buffers.

    ``tokens``: also the layer input's rows, which the token's owner rank pushes
    (``xin``, two slots by launch parity). ``head_vocab``: the tail launch's buffers
    instead, the gathered bf16 logits of a ``head_vocab`` vocabulary (two slots)."""
    from sgl_amd_mono.common.guard import FLAG_BYTES

    if head_vocab:
        part = samples * head_vocab * 4
        return {"_bytes": 0}, {
            "logits": 0,
            "_logits_stride": part,
            "flags": 2 * part,
            "_bytes": 2 * part + FLAG_BYTES,
        }
    split_count = sparse_attention_topk // sparse_keys_per_task(samples, heads)
    output_heads = heads if output_heads is None else output_heads
    pair_bytes = 8
    items = [
        ("q_a", samples * Q_LORA * pair_bytes),
        ("q_an", samples * Q_LORA // 2 * pair_bytes),
        ("kv_a", samples * (KV_LORA + PE_DIM) * pair_bytes),
        ("kvnew", samples * KV_LORA * pair_bytes),
        ("penew", samples * PE_DIM * pair_bytes),
        ("q_nope", samples * heads * NOPE_DIM * pair_bytes),
        ("q_pe", samples * heads * PE_DIM * pair_bytes),
        ("q_lat", samples * heads * KV_LORA * pair_bytes),
        ("sp_acc", samples * split_count * heads * KV_LORA * pair_bytes),
        ("sp_m", samples * split_count * heads * pair_bytes),
        ("sp_l", samples * split_count * heads * pair_bytes),
        ("o", samples * output_heads * V_DIM * pair_bytes),
        ("a", samples * HIDDEN * pair_bytes),
        ("scores", samples * N_EXPERTS * pair_bytes),
        ("xq", samples * HIDDEN // 4 * pair_bytes),
        (
            "xqs",
            samples * (XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS) * pair_bytes,
        ),
        ("sel", samples * MOE_SLOTS * pair_bytes),
        ("prob", samples * MOE_SLOTS * pair_bytes),
        ("mid", samples * MOE_SLOTS * inter * pair_bytes),
        ("ugp", BLOCKS * samples * 2 * UG_TILE * pair_bytes),
        ("xqd", samples * HIDDEN * 4),
    ]
    if with_indexer:
        items += [
            ("index_k", samples * INDEX_DIM * pair_bytes),
            # each row's new key as cached: its FP8 value words, then its scale
            ("index_k8", samples * (INDEX_DIM // 4 + 1) * pair_bytes),
            ("index_w", samples * INDEX_HEADS * pair_bytes),
            ("index_q", samples * INDEX_Q_ROWS // 2 * pair_bytes),
            ("index_qh", samples * INDEX_Q_ROWS // 2 * pair_bytes),
            ("index_scores", samples * index_max_seq * pair_bytes),
            ("indices", samples * sparse_attention_topk * pair_bytes),
            # parallel top-k: per slice a 4096-bin histogram and its ready pair, per
            # row the threshold (bin prefix, keys still needed, keys in the bin), per
            # slice its winners' positions, its (key, position) candidates and their
            # counts
            ("index_part_hist", samples * INDEX_SEL_SLICES * 4096 * 4),
            ("index_part_ready", samples * INDEX_SEL_SLICES * pair_bytes),
            ("index_thresh", samples * 4 * pair_bytes),
            ("index_win_stage", samples * INDEX_SEL_SLICES * sparse_attention_topk * 4),
            ("index_cand_stage", samples * INDEX_SEL_SLICES * INDEX_CAND_STAGE_CAP * 8),
            ("index_gather_count", samples * INDEX_SEL_SLICES * pair_bytes),
        ]

    offset, scratch = 0, {}
    for name, size in items:
        scratch[name] = offset
        offset += _align(size)
    scratch["_bytes"] = offset

    part = npes * samples * HIDDEN * pair_bytes
    region = 2 * part
    dcp_part = (
        dcp_size * samples * (heads * V_DIM // UV_TILE) * DCP_SUMMARY_PAIRS * pair_bytes
        if dcp_size > 1
        else 0
    )
    symmetric = {
        "attn": 0,
        "ffn": region,
        "dcp": 2 * region,
        # the guard's error flags: rank r writes word r of every rank's flags
        "flags": 2 * region + 2 * dcp_part,
        "_part_stride": part,
        "_dcp_part_stride": dcp_part,
        "_bytes": 2 * region + 2 * dcp_part + FLAG_BYTES,
    }
    if tokens:
        symmetric["xin"] = symmetric["_bytes"]
        symmetric["_xin_stride"] = samples * HIDDEN * 4
        symmetric["_bytes"] += 2 * samples * HIDDEN * 4
    return scratch, symmetric


def stage_tasks(
    samples: int,
    heads: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    expert_mxfp4: bool = False,
    inter: int = INTER,
    tokens: bool = False,
    head_rows: int = 0,
):
    """Return ``(stage name, task count)`` pairs in execution order; with ``tokens``
    led by the embedding rows' push. With ``head_rows`` (this rank's LM-head rows)
    the tail's instead."""

    if head_rows:
        return [("lmhead", head_rows // HEAD_TILE), ("gather", BLOCKS)]
    tasks = [("embed", N_ROW_TILES)] if tokens else []
    tasks += [
        # with the indexer, the index K and weights projections' 16-row tiles follow
        (
            "qkv_a",
            N_QKV_A + ((INDEX_DIM + INDEX_HEADS) // QKV_A_TILE if with_indexer else 0),
        ),
        ("q_norm", samples),
        ("cache", samples),
    ]
    if with_indexer:
        tasks += [("index_key", samples)]
    tasks += [
        ("q_b", heads * (NOPE_DIM + PE_DIM) // Q_B_TILE),
    ]
    if with_indexer:
        # index_qh: each row's query, rotated and Hadamard-transformed, 8 heads a task
        tasks += [
            ("index_q", INDEX_Q_ROWS // INDEX_TILE),
            ("index_qh", samples * INDEX_HEADS // 8),
        ]
    tasks += [("uk", heads * KV_LORA // UK_TILE)]
    if with_indexer:
        tasks += [
            # one task a CTA: its runs of the requests' keys
            ("index_score", BLOCKS),
            ("index_hist", samples * INDEX_SEL_SLICES),
            ("index_thresh", samples),
            ("index_gather", samples * INDEX_SEL_SLICES),
            ("index_select", samples),
        ]
    tasks += [
        (
            "split",
            samples
            * (heads // WAVES)
            * (sparse_attention_topk // sparse_keys_per_task(samples, heads)),
        ),
        ("uv", samples * (heads * V_DIM // UV_TILE)),
        ("o", N_ROW_TILES),
        ("router", samples * N_ROUTER),
        (
            "ug",
            (
                BLOCKS
                if samples == 1
                else samples
                * max(
                    BLOCKS,
                    ((MOE_SLOTS * inter // UG_TILE + BLOCKS - 1) // BLOCKS) * BLOCKS,
                )
            ),
        ),
        ("down", HIDDEN // dn_tile(samples, expert_mxfp4)),
    ]
    return tasks


def stage_bases(
    samples: int,
    heads: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    expert_mxfp4: bool = False,
    inter: int = INTER,
):
    """Return ``{stage name: base CTA}``: task ``t`` of a stage runs on CTA
    ``(base + t) % BLOCKS``."""

    tasks = dict(
        stage_tasks(
            samples,
            heads,
            sparse_attention_topk,
            with_indexer,
            index_max_seq,
            expert_mxfp4,
            inter=inter,
        )
    )
    base, acc = {}, 0
    # q_norm (with each row's cache task) on the first CTAs past qkv_a's, so it polls
    # from the launch start; q_b after it. split from the CTA past q_b's, so its tiles
    # land on CTAs freed by qkv_a, the last few on the q_norm CTAs; the cache keeps one
    # CTA of the sequence after split, so uk (on the q_b CTAs it waits on anyway) and
    # the later stages keep their CTAs.
    for name in (
        "qkv_a",
        "q_norm",
        "q_b",
        "split",
        "cache",
        "uk",
        "uv",
        "o",
        "router",
        "ug",
        "down",
    ):
        base[name] = acc % BLOCKS
        acc += 1 if name == "cache" else tasks[name]
    base["cache"] = base["q_norm"]
    # o from split's first CTA: its tasks fill the split CTAs, free once the
    # attention ends
    base["o"] = base["split"]
    if with_indexer:
        # q_b occupies exactly half the grid.  Put the second half of index-Q on
        # the complementary CTAs while q_b CTAs reuse their normalized q_lora
        # tile for the first half, instead of serializing two index-Q tiles on
        # every q_b CTA.
        base["index_q"] = (base["q_b"] + tasks["q_b"]) % BLOCKS
        # on the first CTAs of index_q's own tiles, after their tile
        base["index_qh"] = base["index_q"]
        # each row's index key on its q_norm CTA, after the CTA's index-query tile
        base["index_key"] = base["q_norm"]
        # one scoring task a CTA, numbered by its CTA
        base["index_score"] = 0
        base["index_select"] = 100
        base["index_thresh"] = 100
        base["index_hist"] = BLOCKS // 2
        base["index_gather"] = BLOCKS // 2
    return base
