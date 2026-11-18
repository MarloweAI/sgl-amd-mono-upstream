# SPDX-License-Identifier: Apache-2.0
"""Host-side comparison helpers shared by the Qwen3.5 precision paths."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
        return {}
    diff = a - b
    denom = b.norm().clamp_min(1e-30)
    cos = F.cosine_similarity(a, b, dim=0)
    if not torch.isfinite(cos):
        cos = torch.ones((), device=a.device) if diff.abs().max().item() == 0 else cos
    return {
        "cosine": float(cos.item()),
        "rel": float(diff.norm().item() / denom.item()),
        "max_abs": float(diff.abs().max().item()),
    }
