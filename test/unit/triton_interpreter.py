"""Run a function of a test file in a child process under Triton's interpreter, for
the Triton kernels a CPU host can run: the interpreter takes effect only when set
before Triton is imported, and the test process has imported it already."""

import json
import os
import subprocess
import sys

_MARK = "INTERPRETED_RESULT "


def run(path: str, name: str, *args):
    """``name(*args)`` of the test file ``path`` (JSON arguments and result)."""
    code = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('interpreted', {path!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        f"out = getattr(mod, {name!r})(*json.loads(sys.argv[1]))\n"
        f"print({_MARK!r} + json.dumps(out))\n"
    )
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(path)] + [p for p in [env.get("PYTHONPATH")] if p]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, json.dumps(args)],
        env=env,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(_MARK)]
    if proc.returncode or not lines:
        raise AssertionError(
            f"{name} failed:\n{proc.stdout[-4000:]}{proc.stderr[-4000:]}"
        )
    return json.loads(lines[-1][len(_MARK) :])
