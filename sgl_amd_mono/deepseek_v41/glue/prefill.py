# SPDX-License-Identifier: Apache-2.0
"""Prefill indexer kernels compiled once, not per context length: aiter's FP4 MQA logits with
the block-table stride an argument (bitwise aiter's, checked at the first call), the schedule's
Triton kernels unspecialized and prewarmed. ``SGLANG_AMD_MONO_DSV41_PREFILL_JIT=0`` keeps them."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import os
import time

import torch

_LOGGER = logging.getLogger(__name__)

ENV = "SGLANG_AMD_MONO_DSV41_PREFILL_JIT"
AITER_MODULE = "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
SCHEDULE_MODULE = "sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip"
ROW_PLAN_STEPS = 14  # aiter's row plan search: s_max up to 2**13
PREP_INTS = (
    "pt_src_stride",
    "pt_dst_stride",
    "T",
    "P",
    "s_max",
    "w_src",
    "w_dst",
    "pt_tiles_per_row",
)

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/attention/dsv4/fp4_indexer_schedule_hip.py": (
        "_prefill_schedule_prep_kernel",
        "MAX_FUSED_ROWS",
        "build_prefill_schedule",
        "build_prefill_schedule(page_table)",
        "build_prefill_schedule(local_ends)",
        "build_prefill_schedule(cta_info_out)",
        "build_prefill_schedule(parallel_unit_num)",
        "build_prefill_schedule(max_seq_len)",
        "build_prefill_schedule(block_k)",
    ),
    "kernels/ops/attention/dsv4/fp4_indexer_hip.py": ("_guarded_pages",),
}

# (old, new) edits of aiter's module: the stride an argument of the kernel and its launcher
_EDITS = (
    (
        "from .pa_mqa_logits_fp4_common import",
        "from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_common import",
    ),
    (
        "        stride_out_row: Int32,\n        weight_scale: fx.Float32,\n    ):",
        "        stride_out_row: Int32,\n        weight_scale: fx.Float32,\n"
        "        stride_bt_row: Int32,\n    ):",
    ),
    (
        "phys_vec = bt_bt[batch_id * _stride_bt + bi_base]",
        "phys_vec = bt_bt[batch_id * stride_bt_row + bi_base]",
    ),
    (
        "        stride_out: fx.Int32,\n        weight_scale: fx.Float32,\n"
        "        gx: fx.Int32,\n        stream: fx.Stream,\n    ):",
        "        stride_out: fx.Int32,\n        weight_scale: fx.Float32,\n"
        "        gx: fx.Int32,\n        stride_bt: fx.Int32,\n        stream: fx.Stream,\n    ):",
    ),
    (
        "kfn(out, q, qs, kv, kvs, bt, w, cta_info_, stride_out, weight_scale).launch(",
        "kfn(out, q, qs, kv, kvs, bt, w, cta_info_, stride_out, weight_scale, stride_bt).launch(",
    ),
    (
        "    max_blocks_per_seq = block_tables.shape[1]\n\n    if (cta_info is None) != (n_ctas is None):",
        "    max_blocks_per_seq = 0  # the stride is the launcher's argument\n\n"
        "    if (cta_info is None) != (n_ctas is None):",
    ),
    (
        "        float(weight_scale),\n        n_ctas,\n        stream,\n    )\n    return out",
        "        float(weight_scale),\n        n_ctas,\n        block_tables.stride(0),\n"
        "        stream,\n    )\n    return out",
    ),
)


def _strided_module():
    from sgl_amd_mono.common.hip_ext import cache_dir

    spec = importlib.util.find_spec(AITER_MODULE)
    if spec is None or spec.origin is None:
        return None, "aiter has no FP4 prefill MQA logits module"
    src = open(spec.origin).read()
    for old, new in _EDITS:
        if src.count(old) != 1:
            return None, f"aiter's source differs at {old.strip()[:60]!r}"
        src = src.replace(old, new)
    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = cache_dir() / "py" / f"pa_mqa_logits_fp4_prefill_strided_{digest}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src)
        tmp.replace(path)
    mspec = importlib.util.spec_from_file_location(
        f"mono_pa_mqa_logits_fp4_prefill_{digest}", path
    )
    module = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(module)
    return module, None


class _Checked:
    """The strided launch, checked bitwise against aiter's at the first eager call."""

    def __init__(self, strided, original):
        self._strided, self._original = strided, original
        self._checked, self.off = False, False

    def __call__(self, *args, **kwargs):
        if self.off:
            return self._original(*args, **kwargs)
        if self._checked or torch.cuda.is_current_stream_capturing():
            return self._strided(*args, **kwargs)
        self._checked = True
        out = kwargs.pop("out", None)
        want = self._original(*args, **kwargs, out=None if out is None else out.clone())
        try:
            got = self._strided(*args, **kwargs, out=out)
        except Exception as e:  # noqa: BLE001
            _LOGGER.warning(
                "DSV4.1 prefill MQA logits: the strided kernel failed (%s); aiter's runs",
                e,
            )
            got = None
        # rows no CTA writes keep whatever they held, NaN too: compare the bits
        if got is not None and torch.equal(
            got.view(torch.int32), want.view(torch.int32)
        ):
            return got
        self.off = True
        _LOGGER.warning(
            "DSV4.1 prefill MQA logits: the strided kernel differs; aiter's runs"
        )
        if out is not None:
            out.copy_(want)
            return out
        return want


def _unspecialized(kernel):
    import triton

    names = [p.name for p in kernel.params if not p.is_constexpr]
    return triton.jit(kernel.fn, do_not_specialize=names)


class _FixedSteps:
    """aiter's row plan with ROW_PLAN_STEPS search steps (surplus steps change nothing),
    so one variant serves every context length."""

    def __init__(self, kernel):
        self.kernel, self.fn = kernel, kernel.fn

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            if kwargs.get("SEARCH_STEPS", ROW_PLAN_STEPS) <= ROW_PLAN_STEPS:
                kwargs["SEARCH_STEPS"] = ROW_PLAN_STEPS
            return launch(*args, **kwargs)

        return run


def _patch_mqa_logits() -> str:
    import aiter.ops.flydsl as flydsl_ops

    module, why = _strided_module()
    if module is None:
        return why
    flydsl_ops.flydsl_pa_mqa_logits_fp4_prefill = _Checked(
        module.flydsl_pa_mqa_logits_fp4_prefill,
        flydsl_ops.flydsl_pa_mqa_logits_fp4_prefill,
    )
    return "the block-table stride an argument"


def _patch_schedule() -> str:
    mod = importlib.import_module(SCHEDULE_MODULE)
    kernel = getattr(mod, "_prefill_schedule_prep_kernel", None)
    if getattr(kernel, "fn", None) is None:
        return "no _prefill_schedule_prep_kernel"
    if not set(PREP_INTS) <= set(getattr(kernel, "arg_names", PREP_INTS)):
        return "its integer arguments differ"
    mod._prefill_schedule_prep_kernel = _unspecialized(kernel)
    aiter_mod = importlib.import_module(AITER_MODULE)
    cta = getattr(aiter_mod, "_prefill_cta_info_kernel", None)
    if getattr(cta, "fn", None) is not None:
        aiter_mod._prefill_cta_info_kernel = _unspecialized(cta)
    return "arguments unspecialized"


def _patch_row_plan() -> str:
    mod = importlib.import_module(AITER_MODULE)
    kernel = getattr(mod, "_prefill_row_plan_kernel", None)
    if getattr(kernel, "fn", None) is None or not {
        "T",
        "P",
        "block_k",
        "s_max",
        "SEARCH_STEPS",
    } <= set(getattr(kernel, "arg_names", ())):
        return "aiter's row plan kernel differs"
    mod._prefill_row_plan_kernel = _FixedSteps(_unspecialized(kernel))
    return f"{ROW_PLAN_STEPS} search steps, arguments unspecialized"


_ACTIVE = []


def install() -> None:
    from sgl_amd_mono.deepseek_v41.glue import flag

    if not flag(ENV):
        return
    _ACTIVE.append(True)
    for name, patch in (
        ("mqa_logits", _patch_mqa_logits),
        ("schedule", _patch_schedule),
        ("row_plan", _patch_row_plan),
    ):
        try:
            how = patch()
        except Exception as e:  # noqa: BLE001
            how = f"{type(e).__name__}: {e}"
        _LOGGER.info("DSV4.1 prefill %s: %s", name, how)


def prewarm(device) -> None:
    """The prefill schedule's kernel variants (a power of two of the rows) compiled now,
    not at the first prefill of each row count."""
    if not _ACTIVE:
        return
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import _guarded_pages
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip import (
        MAX_FUSED_ROWS,
        build_prefill_schedule,
    )

    t0 = time.perf_counter()
    rows = [16 << i for i in range(16) if 16 << i <= MAX_FUSED_ROWS] + [
        2 * MAX_FUSED_ROWS
    ]
    try:
        for t in rows:
            ctas = max(1024, t)
            build_prefill_schedule(
                page_table=torch.zeros((t, 4), dtype=torch.int32, device=device),
                local_ends=torch.full((t,), 64, dtype=torch.int32, device=device),
                cta_info_out=torch.empty((ctas, 6), dtype=torch.int32, device=device),
                parallel_unit_num=ctas,
                max_seq_len=_guarded_pages(4, 4) * 64,
                block_k=256,
            )
        torch.cuda.synchronize(device)
    except Exception as e:  # noqa: BLE001
        _LOGGER.warning("DSV4.1 prefill schedule prewarm: %s: %s", type(e).__name__, e)
        return
    _LOGGER.info(
        "DSV4.1 prefill schedule: %d row counts compiled in %.1f s",
        len(rows),
        time.perf_counter() - t0,
    )
