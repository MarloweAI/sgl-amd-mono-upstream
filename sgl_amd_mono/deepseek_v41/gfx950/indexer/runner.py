# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang#43984 at 134919476c (kernels/ops/mono/gfx950/
# deepseek_v41/indexer/runner.py)
"""K2i's host side on one rank (``DSV41MonoRunner.enable_k2i``): K2i's logits, then SGLang's top-k.

Every tensor a replayed launch reads or writes lives here, so captured graphs stay valid. A
``full`` source keeps SGLang's top 512 of every visible column; a ``producer`` also publishes
its candidate blocks, SGLang's top-k of the 8-column block maxima K2i keeps as it scores
(``select_candidate_blocks_hip``'s level one); a ``reindex`` source scores only those blocks'
columns and maps its top 512 back to positions and slots (``candidate_table``)."""

from __future__ import annotations

from typing import NamedTuple

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
    CandidateBlocks,
    topk_transform_paged_hip,
)

from . import index_plan as ip
from . import topk_one
from .candidate_table import PLANE_ROWS
from .compress import THREADS, build_kc
from .k2i import MAX_ROWS, build_k2i
from .score_fp4 import LENS_ROWS
from .walk_plan import new_plan

# the SGLang names this module reads, pinned by test/unit/test_contract.py
SGLANG_READS = {
    "kernels/ops/attention/dsv4/candidate_blocks_hip.py": (
        # the producer's candidate blocks, as ``select_candidate_blocks_hip`` publishes them
        "CandidateBlocks",
        "compact_lens",
        "compact_page_table",
        "compact_page_size",
        "block_size",
        "ids",
        "topk_transform_paged_hip",
        "topk_transform_paged_hip(scores)",
        "topk_transform_paged_hip(seq_lens)",
        "topk_transform_paged_hip(page_table)",
        "topk_transform_paged_hip(page_indices)",
        "topk_transform_paged_hip(page_size)",
        "topk_transform_paged_hip(raw_indices)",
    ),
}

COMPACT = ip.TOPK_BLOCKS * ip.BLOCK_ROWS  # a REINDEX row's candidate columns
KINDS = ("full", "producer", "reindex")


def _roundup(x: int, m: int) -> int:
    return -(-x // m) * m


def check_plane(*, payload: torch.Tensor, scale: torch.Tensor) -> None:
    """Raise unless the index plane is SGLang's split FP4 layout of 64-row pages."""
    u8 = payload.view(torch.uint8)
    pages = scale.shape[0]
    if not (
        u8.is_contiguous()
        and scale.is_contiguous()
        and scale.dtype == torch.uint8
        and u8.numel() == pages * 4 * PLANE_ROWS * 16
        and scale.numel() == pages * 4 * PLANE_ROWS
    ):
        raise RuntimeError(
            f"K2i index plane: payload {tuple(payload.shape)} {payload.dtype}, scales "
            f"{tuple(scale.shape)} {scale.dtype}; want [pages, 1, 4, {PLANE_ROWS}, 16] "
            f"and [pages, 1, 4, {PLANE_ROWS}] bytes"
        )


class Selection(NamedTuple):
    """A selection's results past its indices: a producer's candidate blocks, and the glue its
    K2 runs (``index_glue`` kind, its tensors and ints by name)."""

    blocks: CandidateBlocks | None
    glue: tuple | None


class K2iRunner:
    """K2i on one device, for rows up to MAX_ROWS and column bounds up to
    ``bound_max``."""

    def __init__(self, device: torch.device, bound_max: int):
        f32 = dict(dtype=torch.float32, device=device)
        i32 = dict(dtype=torch.int32, device=device)
        self.bound_max = _roundup(max(bound_max, COMPACT), ip.CHUNK)
        self.ilog = torch.empty(MAX_ROWS * self.bound_max, **f32)
        # ``reindex_table``'s outputs: blocks newest last, their key table, each row's columns
        self.cand_ids = torch.empty(MAX_ROWS, ip.TOPK_BLOCKS, **i32)
        self.cand_table = torch.empty(MAX_ROWS, ip.TOPK_BLOCKS, **i32)
        self.cand_bound = torch.empty(MAX_ROWS, **i32)
        self.compact_pos = torch.empty(MAX_ROWS, ip.TOPK, **i32)
        # a producer's level one: its rows' 8-column block maxima (16-byte rows, as SGLang's top-k
        # reads them), their blocks and compact widths (LENS_ROWS apart), and its blocks
        self.bstride = self.bound_max // ip.BLOCK_ROWS
        self.bmax = torch.empty(MAX_ROWS * self.bstride, **f32)
        self.blens = torch.empty(2 * LENS_ROWS, **i32)
        self.block_ids = torch.empty(MAX_ROWS, ip.TOPK_BLOCKS, **i32)
        self.block_pages = torch.zeros(MAX_ROWS, 1, **i32)
        self.plans = {ratio: new_plan(device) for ratio in (1, 2)}
        # owners[p - 1][t]: row t's request's first row, for p rows a request
        t = torch.arange(MAX_ROWS)
        self.owners = torch.stack([t // p * p for p in range(1, MAX_ROWS + 1)]).to(
            **i32
        )
        self.one = topk_one.TopKOne(device)
        self._kernels: dict = {}

    def kernel(self, rows: int, blocks: bool = False):
        from ..runner import no_capture

        if (rows, blocks) not in self._kernels:
            no_capture(f"K2i width {rows}")
            self._kernels[rows, blocks] = build_k2i(rows, blocks)
        return self._kernels[rows, blocks]

    def run_glue(self, rows: int, sel: Selection) -> None:
        """A selection's glue (``Selection.glue``) as its own launch, for the tests: what the
        source's K2 runs."""
        from ..runner import no_capture
        from ..stages.index_glue import ARGS, build_glue

        if sel.glue is None:
            return
        kind, gl = sel.glue
        key = ("glue", kind, rows)
        if key not in self._kernels:
            no_capture(f"glue width {rows}")
            self._kernels[key] = build_glue(kind, rows)
        vals = [gl.get(n, 0) for n in ARGS]
        _run_compiled(
            self._kernels[key],
            *(v.data_ptr() if isinstance(v, torch.Tensor) else int(v) for v in vals),
            torch.cuda.current_stream(),
        )

    def kc_args(
        self, *, live, per_req, state, ring_size, norm, eps, freqs, pos, req, raw_loc,
        index_loc, cache, page_size, wk, k_norm, k_eps, plane,
    ) -> tuple:  # fmt: skip
        """An index KV source's compressor's arguments past its projection
        (``compress.KC_ARGS[1:]``), for K1 (which runs it, ``index_query.run_kc``) or
        ``compress``: the compressed record into ``cache`` (V4 pages of ``page_size`` slots,
        ``[pages, page bytes]`` uint8) at ``raw_loc`` (>> 1 at ratio 2), the index key into
        ``plane`` at ``index_loc``; at ratio 2 each even position parks in ``state`` (ring
        ``ring_size``). ``pos``, ``req``, ``raw_loc``, ``index_loc``: int32 or int64 [live]:
        the first ``live`` of the step's rows, SGLang's live rows. ``wk``: the indexer's, in
        ``compress.wk_order``."""
        if not (
            wk.dtype == torch.bfloat16
            and wk.shape == (16, THREADS, 8)
            and wk.is_contiguous()
        ):
            raise RuntimeError(
                f"KC: wk {tuple(wk.shape)} {wk.dtype}, not in KC's order"
            )
        if page_size & (page_size - 1):
            raise RuntimeError(f"KC: compressed pages of {page_size} slots")
        if not (
            cache.dim() == 2 and cache.dtype == torch.uint8 and cache.stride(1) == 1
        ):
            raise RuntimeError(
                f"KC: compressed cache {tuple(cache.shape)} {cache.dtype}"
            )
        payload, pscale = plane
        arrays = (pos, req, raw_loc, index_loc)  # ``compress.INDEX_ARRAYS``
        wide = sum(
            int(t is not None and t.dtype == torch.int64) << k
            for k, t in enumerate(arrays)
        )
        return (
            per_req, state.data_ptr() if state is not None else 0, ring_size,
            norm.data_ptr(), _f32_bits(eps), freqs.data_ptr(), pos.data_ptr(),
            req.data_ptr() if req is not None else 0, raw_loc.data_ptr(),
            index_loc.data_ptr(), cache.data_ptr(), page_size.bit_length() - 1,
            cache.stride(0), wk.data_ptr(), k_norm.data_ptr(), _f32_bits(k_eps),
            payload.data_ptr(), pscale.data_ptr(), wide, live,
        )  # fmt: skip

    def compress(self, *, ratio, rows, projected, kc) -> None:
        """The compressor of ``kc_args`` alone, in one launch, from ``projected``
        (``project_fused`` [rows, 1024] f32 at ratio 2, ``wkv`` [rows, 512] bf16 at
        ratio 1)."""
        key = ("kc", rows, ratio)
        if key not in self._kernels:
            from ..runner import no_capture

            no_capture(f"KC width {rows}")
            self._kernels[key] = build_kc(rows, ratio)
        _run_compiled(
            self._kernels[key], projected.data_ptr(), *kc, torch.cuda.current_stream()
        )

    def select(
        self,
        *,
        kind: str,
        rows: int,
        per_req: int,
        ratio: int,
        query: tuple,  # [rows, 32, 64] E2M1 pairs, [rows, 1, 4, 16, 4] E8M0, [rows, 32] bf16
        plane: tuple,  # the layer's index plane (``check_plane``)
        page_table: torch.Tensor,  # int32 [rows, pages] of 64-row pages
        bounds: torch.Tensor,  # int32 [rows]: the columns each row sees
        raw_out: torch.Tensor,  # int32 [rows, 512]: sparse_raw_indices
        page_out: torch.Tensor,  # int32 [rows, 512]: sparse_page_indices
        candidates=None,  # a REINDEX layer's: the producer's CandidateBlocks
        fill_plan=True,  # FULL / producer: the step's first of ``ratio`` lays out its walk plan
    ) -> Selection:
        """An index source's top-512 of each row (``kind`` in KINDS).

        FULL and producer layers walk the step's plan of ``ratio``, which the step's first of
        them lays out (``fill_plan``). A producer returns its candidate blocks for the step's
        REINDEX layers and, as its K2's glue, their candidate table (``index_glue.TABLE``); a
        REINDEX layer's top-k writes its page indices through that table, and its K2's glue
        its raw indices (``index_glue.RAWS``)."""
        from ..stages import index_glue

        assert kind in KINDS, kind
        _check(
            rows=rows, per_req=per_req, query=query, page_table=page_table,
            bounds=bounds, raw_out=raw_out, page_out=page_out,
        )  # fmt: skip
        inputs = dict(
            rows=rows, per_req=per_req, query=query, plane=plane,
            page_table=page_table, bounds=bounds,
        )  # fmt: skip
        if kind == "reindex":
            compact, lens, ids = self.score_candidates(candidates=candidates, **inputs)
            # compact column j * 8 + i is row i of the virtual page cand_table[j]: its slot
            pos = self.compact_pos[:rows]
            topk_transform_paged_hip(
                scores=compact,
                seq_lens=lens,
                page_table=self.cand_table[:rows],
                page_indices=page_out,
                page_size=ip.BLOCK_ROWS,
                raw_indices=pos,
            )
            glue = {"gl_cols": pos, "gl_ids_out": ids, "gl_raw": raw_out}
            return Selection(None, (index_glue.RAWS, glue))
        producer = kind == "producer"
        logits = self.score(ratio=ratio, fill_plan=fill_plan, blocks=producer, **inputs)
        published = None
        if producer:
            # SGLang's level one on K2i's block maxima: the top TOPK_BLOCKS of each row's
            # first ceil(bound / 8), as ``select_candidate_blocks_hip`` takes them
            nb = -(-logits.shape[1] // ip.BLOCK_ROWS)
            ids = self.block_ids[:rows]
            topk_transform_paged_hip(
                scores=self.bmax[: rows * self.bstride].view(rows, self.bstride)[
                    :, :nb
                ],
                seq_lens=self.blens[:rows],
                page_table=None,
                page_indices=ids,
                page_size=1,
                raw_indices=None,
            )
            published = CandidateBlocks(
                ids=ids,
                compact_lens=self.blens[LENS_ROWS : LENS_ROWS + rows],
                compact_page_table=self.block_pages[:rows],
                compact_page_size=COMPACT,
                block_size=ip.BLOCK_ROWS,
            )
            glue = {
                "gl_ids": ids, "gl_ids_out": self.cand_ids, "gl_table": self.cand_table,
                "gl_bound": self.cand_bound, "gl_ptab": page_table,
                "gl_ptab_stride": page_table.stride(0), "gl_bounds": bounds,
                "gl_owners": self.owners[per_req - 1],
            }  # fmt: skip
        self._paged_topk(
            scores=logits,
            seq_lens=bounds,
            page_table=page_table,
            page_indices=page_out,
            page_size=PLANE_ROWS,
            raw_indices=raw_out,
        )
        return Selection(published, (index_glue.TABLE, glue) if producer else None)

    def _paged_topk(self, **args) -> None:
        """SGLang's paged top-k with both outputs: one launch up to ``topk_one.MAX_ROWS`` rows
        (``topk_one``), SGLang's own launches past that."""
        if not topk_one.TopKOne.fits(args["scores"].shape[0]):
            topk_transform_paged_hip(**args)
            return
        if not topk_one.module.cache_info().currsize:
            from ..runner import no_capture

            no_capture("the one-launch top-k")
        self.one.transform_paged(**args)

    def score(
        self, *, rows, per_req, ratio, query, plane, page_table, bounds, fill_plan=True,
        blocks=False,
    ) -> torch.Tensor:  # fmt: skip
        """A FULL or producer layer's logits [rows, width] f32, valid below each row's bound
        (``fill_plan``: laying out the step's walk plan of ``ratio`` first; ``blocks``: a
        producer's, also its block maxima and lengths).

        ``width`` spans the page table's columns, as wide as SGLang's logits for its top-k."""
        width = min(page_table.shape[1] * PLANE_ROWS, self.bound_max)
        # a page-table entry is 8 virtual 8-row pages (ishift 3)
        self._score(
            rows=rows, query=query, plane=plane, plan=self.plans[ratio], itab=page_table,
            itab_stride=page_table.stride(0), itab_len=page_table.shape[1],
            ibatch=self.owners[per_req - 1], ishift=3, ibound=bounds, lstride=width,
            reindex=0, fill_plan=int(fill_plan), blocks=blocks,
        )  # fmt: skip
        return self.ilog[: rows * width].view(rows, width)

    def score_candidates(
        self, *, candidates, rows, per_req, query, plane, page_table, bounds
    ):  # fmt: skip
        """A REINDEX layer's logits of its candidate blocks' columns -> (compact, lens, ids).

        ``ids`` is ``candidates.ids`` with the newest block last, as the producer's K2 laid out
        their table (``index_glue.TABLE``); compact column j * 8 + i is row i of block ids[j],
        valid below ``lens``."""
        if tuple(candidates.ids.shape) != (rows, ip.TOPK_BLOCKS):
            raise RuntimeError(
                f"K2i: candidate blocks {tuple(candidates.ids.shape)} for {rows} rows"
            )
        owners = self.owners[per_req - 1]
        ids, lens = self.cand_ids[:rows], self.cand_bound[:rows]
        # a REINDEX launch ignores the plan: its task comes from its CTA index
        self._score(
            rows=rows, query=query, plane=plane, plan=self.plans[1], itab=self.cand_table,
            itab_stride=ip.TOPK_BLOCKS, itab_len=ip.TOPK_BLOCKS, ibatch=owners, ishift=0,
            ibound=lens, lstride=COMPACT, reindex=1, fill_plan=0,
        )  # fmt: skip
        return self.ilog[: rows * COMPACT].view(rows, COMPACT), lens, ids

    def _score(
        self, *, rows, query, plane, plan, itab, itab_stride, itab_len, ibatch, ishift,
        ibound, lstride, reindex, fill_plan, blocks=False,
    ) -> None:  # fmt: skip
        q_fp4, q_scale, weights = query
        payload, scale = plane
        _run_compiled(
            self.kernel(rows, blocks),
            q_fp4.data_ptr(), q_scale.data_ptr(), weights.data_ptr(), payload.data_ptr(), scale.data_ptr(),
            plan.data_ptr(), itab.data_ptr(), itab_stride, itab_len, ibatch.data_ptr(), ishift,
            ibound.data_ptr(), self.ilog.data_ptr(), lstride, reindex, fill_plan,
            self.bmax.data_ptr(), self.bstride, self.blens.data_ptr(),
            torch.cuda.current_stream(),
        )  # fmt: skip


def _f32_bits(x: float) -> int:
    import struct

    return struct.unpack("<i", struct.pack("<f", x))[0]


def _check(*, rows, per_req, query, page_table, bounds, raw_out, page_out) -> None:
    """Raise unless a selection's tensors are the shapes K2i reads and writes."""
    if not (0 < rows <= MAX_ROWS and rows % per_req == 0):
        raise RuntimeError(f"K2i: {rows} rows, {per_req} a request")
    q_fp4, q_scale, weights = query
    if not (
        q_fp4.is_contiguous()
        and q_fp4.numel() == rows * ip.HEADS * ip.DIM // 2
        and q_fp4.element_size() == 1
        and q_scale.is_contiguous()
        and q_scale.numel() == rows * 4 * 16 * 4
        and tuple(weights.shape) == (rows, ip.HEADS)
        and weights.dtype == torch.bfloat16
        and weights.is_contiguous()
    ):
        raise RuntimeError(
            f"K2i query: {tuple(q_fp4.shape)} {q_fp4.dtype}, scales "
            f"{tuple(q_scale.shape)}, weights {tuple(weights.shape)} {weights.dtype}"
        )
    if not (
        page_table.dtype == torch.int32
        and page_table.dim() == 2
        and page_table.stride(1) == 1
        and page_table.shape[0] == rows
        and bounds.dtype == torch.int32
        and bounds.is_contiguous()
        and bounds.shape[0] == rows
    ):
        raise RuntimeError(
            f"K2i page table {tuple(page_table.shape)} {page_table.dtype}, "
            f"bounds {tuple(bounds.shape)} {bounds.dtype}"
        )
    for name, t in (("raw", raw_out), ("page", page_out)):
        if not (
            t.dtype == torch.int32
            and t.stride(1) == 1
            and t.stride(0) == ip.TOPK
            and t.shape[0] == rows
        ):
            raise RuntimeError(
                f"K2i {name} indices: {tuple(t.shape)} {t.stride()} {t.dtype}"
            )
