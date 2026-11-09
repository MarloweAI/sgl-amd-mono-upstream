# SPDX-License-Identifier: Apache-2.0
"""SGLang's paged top-k (``topk_transform_paged_hip``, both outputs) in one launch for up to 8 rows
(``topk_one.cuh``): its split path's histogram and select meet at an in-launch barrier on one
XCC instead of a kernel boundary. A candidate producer's two selections, its top-512 columns and
its top-2048 blocks, share one launch (``transform_paged_pair``). The selected sets are SGLang's;
built with SGLang's JIT against its own top-k header."""

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
    ),
}

MAX_ROWS = 8  # a row an XCC
_XCC = 8
_PROBLEMS = 2  # a workspace row a row and selection of the pair
_HIST_BINS, _MAX_TIES, _CTR_WORDS = 4096, 2048, 32


@functools.cache
def module():
    """The JIT module, built at first use."""
    import sglang
    from sglang.kernels.jit.utils import load_jit

    csrc = Path(sglang.__file__).resolve().parent / "kernels" / "jit" / "csrc"
    return load_jit(
        "sgl_amd_mono_dsv41_topk_one",
        cuda_files=[str(Path(__file__).with_name("topk_one.cuh"))],
        cuda_wrappers=[
            ("transform_paged", "TopKOne::transform_paged"),
            ("transform_paged_pair", "TopKOne::transform_paged_pair"),
        ],
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
