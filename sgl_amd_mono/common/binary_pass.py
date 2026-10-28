# SPDX-License-Identifier: Apache-2.0
"""FlyDSL's binary pass, run on a freshly parsed copy of the module.

FlyDSL 0.4.0's LLVM can translate one module into different code objects from
one build to the next: its ReassociatePass orders operands by where malloc put
them. Run on a copy parsed back from the module's text, ``gpu-module-to-binary``
gives the same code object in every build, so a code object checked once is the
one every server builds. Importing this module installs it for every FlyDSL
compile in the process.

Remove it once the image's FlyDSL runs its binary pass on a fresh copy itself.
"""

import flydsl.compiler.jit_function as _jit
from flydsl._mlir import ir
from flydsl.compiler.external_llvm import _replace_gpu_module_with_binary_op

_run_pipeline = _jit._run_pipeline


def _run_binary_pass_on_copy(module, fragments, *, verifier, print_after_all):
    """FlyDSL's pipeline on ``module``, with its last pass, the binary pass, on a
    copy parsed from the text of ``module`` before it; the copy's binary then
    replaces the module's ``gpu.module``."""
    *passes, last = fragments
    if not last.strip().startswith("gpu-module-to-binary"):
        return _run_pipeline(
            module, fragments, verifier=verifier, print_after_all=print_after_all
        )
    _run_pipeline(module, passes, verifier=verifier, print_after_all=print_after_all)
    copy = ir.Module.parse(
        module.operation.get_asm(enable_debug_info=True), context=module.context
    )
    _run_pipeline(copy, [last], verifier=verifier, print_after_all=print_after_all)
    _replace_gpu_module_with_binary_op(module, copy)


_jit._run_pipeline = _run_binary_pass_on_copy
