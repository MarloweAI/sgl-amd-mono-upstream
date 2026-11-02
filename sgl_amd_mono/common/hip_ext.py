# SPDX-License-Identifier: Apache-2.0
"""HIP C++ sources built into a torch extension against this process's torch and loaded
with ``torch.ops.load_library``, cached under ``SGLANG_AMD_MONO_CACHE_DIR`` (default
``~/.cache/sgl_amd_mono``) by sources, toolchain and arch; one rank builds, the rest wait."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import torch

_FLAGS = ("-O3",)
# written once the library loaded, so a build that died is redone
_BUILT = "built"


def cache_dir() -> Path:
    default = Path.home() / ".cache" / "sgl_amd_mono"
    return Path(os.environ.get("SGLANG_AMD_MONO_CACHE_DIR") or default)


@functools.cache
def _toolchain() -> str:
    from torch.utils import cpp_extension

    info = Path(cpp_extension.ROCM_HOME or "", ".info", "version")
    return json.dumps(
        {
            "torch": torch.__version__,
            "rocm": info.read_text().strip() if info.is_file() else None,
            "flags": _FLAGS,
        }
    )


def load(name: str, sources: Mapping[str, str], arch: str) -> object:
    """``torch.ops.<name>`` of the extension built from ``sources`` (relative path ->
    text of each .hip translation unit and .h header) for ``arch``; a build that
    fails, or a library that does not load, raises ``RuntimeError``."""
    import filelock

    key = hashlib.sha256(
        "".join(f"{k}\0{v}\0" for k, v in sorted(sources.items())).encode()
        + f"{_toolchain()} {arch}".encode()
    ).hexdigest()[:16]
    lib = f"{name}_{key}"
    root = cache_dir() / "hip" / lib
    root.mkdir(parents=True, exist_ok=True)
    # beside the build directory: torch's own lock is in it
    with filelock.FileLock(str(root) + ".lock"):
        if not (root / _BUILT).is_file():
            _build(lib, sources, root, arch)
            (root / _BUILT).touch()
        else:
            torch.ops.load_library(root / f"{lib}.so")
    return getattr(torch.ops, lib)


def _build(lib: str, sources: Mapping[str, str], root: Path, arch: str) -> None:
    from torch.utils import cpp_extension

    src = root / "src"
    for path, text in sources.items():
        (src / path).parent.mkdir(parents=True, exist_ok=True)
        (src / path).write_text(text, encoding="utf-8")
    (root / "lock").unlink(missing_ok=True)  # left by a build that died
    previous = os.environ.get("PYTORCH_ROCM_ARCH")
    # an --offload-arch flag would add to every arch torch was built for
    os.environ["PYTORCH_ROCM_ARCH"] = arch
    try:
        cpp_extension.load(
            name=lib,
            sources=[str(src / p) for p in sorted(sources) if p.endswith(".hip")],
            extra_include_paths=[str(src)],
            extra_cuda_cflags=list(_FLAGS),
            build_directory=str(root),
            is_python_module=False,
        )
    except OSError as e:  # dlopen's, such as an undefined symbol: the sources' fault
        raise RuntimeError(f"{lib} does not load: {e.__cause__ or e}") from e
    finally:
        if previous is None:
            del os.environ["PYTORCH_ROCM_ARCH"]
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = previous
