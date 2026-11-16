# SPDX-License-Identifier: Apache-2.0
"""SGLang's paged top-k (``topk_transform_paged_hip``, both outputs) in one launch for up to 8 rows
(``topk_one.cuh``): its split path's histogram and select meet at an in-launch barrier on one
XCC instead of a kernel boundary. A candidate producer's two selections, its top-512 columns and
its top-2048 blocks, share one launch (``transform_paged_pair``). The selected sets are SGLang's;
built with SGLang's JIT against its own top-k header.

Across a TP group (``*_cp``, ``Exchange``), a long row is context-parallel: each rank selects in
its run of the row's chunks, and every rank takes the row's top k of all the ranks' picks."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# the SGLang names this module reads, pinned by test/unit/test_contract.py; the kernel also
# includes SGLang's kernels/jit/csrc/deepseek_v4/topk_v2.cuh, which the build checks
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
    "kernels/jit/utils/compile/loader.py": (
        "load_jit(cuda_files)",
        "load_jit(cuda_wrappers)",
        "load_jit(extra_include_paths)",
        "load_jit(extra_cuda_cflags)",
    ),
    "srt/environ.py": ("envs",),
}

MAX_ROWS = 8  # a row an XCC
_XCC = 8
_PROBLEMS = 2  # a workspace row a row and selection of the pair
_HIST_BINS, _MAX_TIES, _CTR_WORDS = 4096, 2048, 32
_MAX_K = 2048  # a selection's k at most
CP_RANKS = 8  # a context-parallel launch's ranks at most


def cp_cells_bytes(tp: int) -> int:
    """A rank's context-parallel cells: [parity][selection][source rank][row][k] of 8 bytes."""
    return 2 * _PROBLEMS * tp * MAX_ROWS * _MAX_K * 8


@functools.cache
def module():
    """The JIT module, built at first use."""
    import sglang
    from sglang.kernels.jit.utils import load_jit
    from sglang.srt.environ import envs

    csrc = Path(sglang.__file__).resolve().parent / "kernels" / "jit" / "csrc"
    # since sglang#47960 the top-k header needs the exactness mode SGLang's own top-k uses
    mode = getattr(envs, "SGLANG_TOPK_V2_MODE", None)
    return load_jit(
        "sgl_amd_mono_dsv41_topk_one",
        cuda_files=[str(Path(__file__).with_name("topk_one.cuh"))],
        cuda_wrappers=[
            ("transform_paged", "TopKOne::transform_paged"),
            ("transform_paged_pair", "TopKOne::transform_paged_pair"),
            ("transform_paged_cp", "TopKOne::transform_paged_cp"),
            ("transform_paged_pair_cp", "TopKOne::transform_paged_pair_cp"),
        ],
        extra_cuda_cflags=[] if mode is None else [f"-DSGL_TOPK_MODE={mode.get()}"],
        extra_include_paths=[str(csrc)],
    )


class TopKOne:
    """The launch's workspace on one device, zeroed once; every launch leaves it zeroed again
    (each row's last workgroup clears the row's histogram and counters)."""

    def __init__(self, device: torch.device):
        cus = torch.cuda.get_device_properties(device).multi_processor_count
        if cus != 256:  # 8 XCCs of 32 CUs: a row an XCC needs all 8
            raise RuntimeError(
                f"one-launch top-k: {cus} CUs, built for MI355X's 8 XCCs of 32"
            )
        i32 = dict(dtype=torch.int32, device=device)
        slots = _PROBLEMS * MAX_ROWS
        self.hist = torch.zeros(slots * _HIST_BINS, **i32)
        self.ctr = torch.zeros(
            (slots + _XCC) * _CTR_WORDS, **i32
        )  # + a presence line an XCC
        self.ties = torch.zeros(slots * _MAX_TIES * 2, **i32)
        # a context-parallel launch's own picks of each row and selection, and its exchange parity
        self.picks = torch.zeros(slots * _MAX_K, **i32)
        self.parity = torch.zeros(slots, **i32)

    @staticmethod
    def fits(rows: int) -> bool:
        return 0 < rows <= MAX_ROWS

    def transform_paged(
        self, *, scores, seq_lens, page_table, page_indices, page_size, raw_indices
    ):
        """``topk_transform_paged_hip`` with both outputs, for ``fits(rows)``."""
        module().transform_paged(
            scores, seq_lens, page_table, page_indices, raw_indices, page_size,
            self.hist, self.ctr, self.ties,
        )  # fmt: skip

    def transform_paged_pair(
        self, *, scores, seq_lens, page_table, page_indices, page_size, raw_indices,
        block_scores, block_lens, block_ids,
    ) -> None:  # fmt: skip
        """``transform_paged``, and in the same launch SGLang's unpaged top-k of ``block_scores``
        [rows, blocks] (bounds ``block_lens``) into ``block_ids`` [rows, k], for ``fits(rows)``."""
        module().transform_paged_pair(
            scores, seq_lens, page_table, page_indices, raw_indices, page_size,
            block_scores, block_lens, block_ids, self.hist, self.ctr, self.ties,
        )  # fmt: skip

    def transform_paged_cp(
        self, *, scores, seq_lens, page_table, page_indices, page_size, raw_indices, cp,
    ) -> None:  # fmt: skip
        """``transform_paged`` with the rows seeing a band of columns or more selected
        context-parallel over a TP group (``cp``: ``Exchange.args``)."""
        module().transform_paged_cp(
            scores, seq_lens, page_table, page_indices, raw_indices, page_size,
            self.hist, self.ctr, self.ties, self.picks, self.parity, *cp,
        )  # fmt: skip

    def transform_paged_pair_cp(
        self, *, scores, seq_lens, page_table, page_indices, page_size, raw_indices,
        block_scores, block_lens, block_ids, cp,
    ) -> None:  # fmt: skip
        """``transform_paged_pair``, context-parallel as ``transform_paged_cp``."""
        module().transform_paged_pair_cp(
            scores, seq_lens, page_table, page_indices, raw_indices, page_size,
            block_scores, block_lens, block_ids, self.hist, self.ctr, self.ties, self.picks,
            self.parity, *cp,
        )  # fmt: skip


class Exchange:
    """A TP group's context-parallel top-k exchange on one rank: its cells, an uncached
    ``PeerBuffer`` with every peer's mapped, built collectively over ``group``."""

    def __init__(self, *, rank: int, tp: int, group, device: torch.device):
        from sgl_amd_mono.common.peer import PeerBuffer

        if not 2 <= tp <= CP_RANKS:
            raise RuntimeError(
                f"context-parallel top-k: {tp} ranks, built for 2 to {CP_RANKS}"
            )
        self.rank, self.tp = rank, tp
        self.buf = PeerBuffer(cp_cells_bytes(tp), group, rank, tp, device)
        self.buf.bytes.zero_()

    def args(self, band: int) -> tuple:
        """A launch's exchange arguments, rows seeing ``band`` columns or more context-parallel."""
        return (self.buf.local, self.buf.addresses, self.rank, self.tp, band)
