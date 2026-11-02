# SPDX-License-Identifier: Apache-2.0
"""SGLang's DSpark step and prefill glue around the DeepSeek-V4.1 monokernels, in fewer
launches. Each piece wraps an SGLang function and keeps SGLang's where its first eager call
disagrees or its kernel does not build. ``SGLANG_AMD_MONO_DSV41_GLUE=0`` keeps all of SGLang's."""

from __future__ import annotations

import functools
import logging
import os

import torch

_LOGGER = logging.getLogger(__name__)

ENV = "SGLANG_AMD_MONO_DSV41_GLUE"
_INSTALLED = []
# piece -> whether it agreed with SGLang at its first eager call
_AGREED: dict[str, bool] = {}


def flag(name: str) -> bool:
    return os.environ.get(ENV, "1") != "0" and os.environ.get(name, "1") != "0"


def route(owner, name: str, wrapper) -> None:
    """``owner.name`` becomes ``wrapper(orig, *args, **kwargs)``."""
    orig = getattr(owner, name)

    @functools.wraps(orig)
    def routed(*args, **kwargs):
        return wrapper(orig, *args, **kwargs)

    setattr(owner, name, routed)


def agreed(piece: str, why) -> bool | None:
    """The piece's verdict, made at its first eager call (``why()`` None: it agrees);
    None under a capture before any verdict."""
    if piece not in _AGREED:
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            reason = why()
        except Exception as e:  # noqa: BLE001 (a kernel that does not build)
            reason = f"{type(e).__name__}: {e}"
        _AGREED[piece] = reason is None
        _LOGGER.info("DSV4.1 glue %s: %s", piece, reason or "one launch")
    return _AGREED[piece]


def install() -> None:
    """Once a process, before the DSpark worker and the graphs are built."""
    if _INSTALLED or os.environ.get(ENV, "1") == "0":
        return
    _INSTALLED.append(True)
    from sgl_amd_mono.deepseek_v41.glue import (
        accept,
        draft_head,
        inject,
        metadata,
        prefill,
    )

    for piece in (metadata, accept, inject, draft_head, prefill):
        piece.install()


def prewarm(device) -> None:
    from sgl_amd_mono.deepseek_v41.glue import prefill

    prefill.prewarm(device)
