# SPDX-License-Identifier: Apache-2.0
"""The DSpark draft's vocab projection in FP8 (SGLang's FP8 lm_head path), its per-row quant
in one launch bitwise SGLang's; only the proposals see it, the target verifies in bf16.
Off by default, as SGLang's ``SGLANG_ROCM_OPT_DSPARK_FP8_LM_HEAD``: it lowers the
acceptance more than it saves. ``SGLANG_AMD_MONO_DSV41_DRAFT_HEAD_FP8=1`` turns it on."""

from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl

from sgl_amd_mono.deepseek_v41.glue import agreed, flag, route

ENV = "SGLANG_AMD_MONO_DSV41_DRAFT_HEAD_FP8"
DSPARK = "sglang.srt.models.dspark"
DRAFT = "sglang.srt.models.deepseek_v4_dspark"

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "srt/models/dspark.py": (
        "_fp8_quant_per_row",
        "_FP8_MAX",
        "_fp8_weight_cache",
        "_fp8_wscale_cache",
        "should_apply_lm_head_quant_method",
        "project_through_lm_head(hidden)",
        "project_through_lm_head(lm_head)",
    ),
    "srt/models/deepseek_v4_dspark.py": ("project_through_lm_head",),
    "srt/utils/*.py": ("is_gfx95_supported",),
}


@triton.jit
def _quant_kernel(X, Q, S, K, sx, BK: tl.constexpr):
    m = tl.program_id(0)
    k = tl.arange(0, BK)
    x = tl.load(X + m * sx + k, mask=k < K, other=0.0)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0).to(tl.float32), 1e-6)
    # torch divides by a scalar as a multiply by its reciprocal
    scale = amax * (1.0 / 448.0)
    q = tl.math.div_rn(x.to(tl.float32), scale)
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    tl.store(Q + m * K + k, q.to(tl.float8e4nv), mask=k < K)
    tl.store(S + m, scale)


def quant_rows(x: torch.Tensor):
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    if m:
        _quant_kernel[(m,)](
            x, q, s, k, x.stride(0), BK=triton.next_power_of_2(k), num_warps=8
        )
    return q, s


def _disagrees(x) -> str | None:
    want_q, want_s = sys.modules[DSPARK]._fp8_quant_per_row(x)
    got_q, got_s = quant_rows(x)
    if not torch.equal(want_s, got_s):
        return "scales differ"
    if not torch.equal(want_q.view(torch.uint8), got_q.view(torch.uint8)):
        return f"{(want_q.view(torch.uint8) != got_q.view(torch.uint8)).sum().item()} codes differ"
    return None


def _project(orig, hidden, lm_head):
    dspark = sys.modules[DSPARK]
    weight = getattr(lm_head, "weight", None)
    if (
        dspark.should_apply_lm_head_quant_method(lm_head, lm_head.quant_method)
        or weight is None
        or weight.dtype != torch.bfloat16
        or hidden.dtype != torch.bfloat16
        or hidden.dim() != 2
        or hidden.stride(1) != 1
        or hidden.shape[0] == 0
        or weight.shape[-1] % 16
        or not agreed("draft_head_fp8", lambda: _disagrees(hidden))
    ):
        return orig(hidden, lm_head)
    w_fp8 = getattr(lm_head, "_fp8_weight_cache", None)
    if w_fp8 is None or w_fp8.shape != weight.shape:
        w_fp8, w_scale = dspark._fp8_quant_per_row(weight)
        lm_head._fp8_weight_cache, lm_head._fp8_wscale_cache = w_fp8, w_scale
    h_fp8, h_scale = quant_rows(hidden)
    return torch._scaled_mm(
        h_fp8,
        w_fp8.t(),
        scale_a=h_scale,
        scale_b=lm_head._fp8_wscale_cache.t(),
        out_dtype=hidden.dtype,
    )


def install() -> None:
    if not flag(ENV, default="0") or torch.version.hip is None:
        return
    import importlib

    from sglang.srt.utils import is_gfx95_supported

    if not is_gfx95_supported():
        return
    importlib.import_module(DSPARK)
    route(importlib.import_module(DRAFT), "project_through_lm_head", _project)
