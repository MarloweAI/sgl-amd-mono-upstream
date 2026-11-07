# SPDX-License-Identifier: Apache-2.0
"""SGLang's DSpark step and prefill glue around the DeepSeek-V4.1 monokernels, in fewer
launches. Each piece wraps an SGLang function and takes its own path wherever the
call's inputs have the layout it reads, else SGLang's. ``SGLANG_AMD_MONO_DSV41_GLUE=0``
keeps all of SGLang's."""

from __future__ import annotations

import functools
import logging
import os

_LOGGER = logging.getLogger(__name__)

ENV = "SGLANG_AMD_MONO_DSV41_GLUE"
_INSTALLED = []
_CHOSEN: set = set()


def flag(name: str, default: str = "1") -> bool:
    return os.environ.get(ENV, "1") != "0" and os.environ.get(name, default) != "0"


def chose(piece: str, ours: bool, why: str | None = None) -> bool:
    """``ours``, logged the first time ``piece`` takes each path."""
    if (piece, ours) not in _CHOSEN:
        _CHOSEN.add((piece, ours))
        how = "one launch" if ours else "SGLang's" + (f" ({why})" if why else "")
        _LOGGER.info("DSV4.1 glue %s: %s", piece, how)
    return ours


def route(owner, name: str, wrapper) -> None:
    """``owner.name`` becomes ``wrapper(orig, *args, **kwargs)``."""
    orig = getattr(owner, name)

    @functools.wraps(orig)
    def routed(*args, **kwargs):
        return wrapper(orig, *args, **kwargs)

    setattr(owner, name, routed)


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
