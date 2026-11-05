# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang#43984 at 134919476c (kernels/ops/mono/gfx950/
# deepseek_v41/indexer/k2i.py), whose scorer comes from ROCm/ATOM at 68506823
# (MIT License, Copyright (C) 2026, Advanced Micro Devices, Inc.)
"""K2i: ATOM K2a's FP4 scorer (``score_fp4``) as its own launch between K1 and K2.

Each CTA scores the task the step's walk plan (FULL; ``fill_plan``: worked out here, the step's
first launch of the ratio) or its index (REINDEX) gives it, none waiting on another, into ``ilog`` [S, lstride] f32 for SGLang's top-k. ``itab`` is a FULL layer's page
table (an entry 2 ** ``ishift`` virtual 8-row pages, read at row ``ibatch[t]``, the request's
first) or a REINDEX layer's candidate table (an entry one virtual page, ``ishift`` 0).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.plan import BLOCKS, THREADS, kernel_sources

from .score_fp4 import ROWS_LDS_WORDS, run_scores4

SOURCES = kernel_sources(__name__)

_STREAM = fx.Stream(None)
MAX_ROWS = 48  # a step's rows at most (8 DSpark requests of 6)


def build_k2i(tokens: int):
    """K2i for a step of ``tokens`` rows."""
    assert 0 < tokens <= MAX_ROWS, tokens
    name = kernel_symbol("dsv41_mono_k2i", s=tokens)
    keyed = (("tokens", tokens), ("sources", SOURCES))

    @fx.struct
    class RowsLds:
        rows: fx.Array[fx.Int32, ROWS_LDS_WORDS, 16]

    @fx.union
    class K2iLds:
        scorer: RowsLds

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2i(
        iq: Int64, iqs: Int64, iw: Int64, plane: Int64, pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32, ibound: Int64, ilog: Int64,
        lstride: Int32, reindex: Int32, fill_plan: Int32,
    ):  # fmt: skip
        _ = keyed
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(K2iLds).scorer.peek()
        args = {
            "iq": iq, "iqs": iqs, "iw": iw, "plane": plane, "pscale": pscale, "iplan": iplan,
            "itab": itab, "itab_stride": itab_stride, "itab_len": itab_len, "ibatch": ibatch,
            "ishift": ishift, "ibound": ibound, "ilog": ilog, "lstride": lstride, "reindex": reindex,
            "fill_plan": fill_plan,
        }  # fmt: skip
        c = {
            "tid": tid, "lane": tid % 64, "wave": tid // 64, "args": args,
            "rows_lds": lds.rows.ptr,
        }  # fmt: skip
        run_scores4(c, tokens, fx.block_idx.x)

    @flyc.jit
    def launch(
        iq: Int64, iqs: Int64, iw: Int64, plane: Int64, pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32, ibound: Int64, ilog: Int64,
        lstride: Int32, reindex: Int32, fill_plan: Int32, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        k2i(
            iq, iqs, iw, plane, pscale, iplan, itab, itab_stride, itab_len, ibatch, ishift, ibound, ilog,
            lstride, reindex, fill_plan,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
