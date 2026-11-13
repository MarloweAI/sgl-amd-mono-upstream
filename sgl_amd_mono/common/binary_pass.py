# SPDX-License-Identifier: Apache-2.0
"""FlyDSL's binary pass on a freshly parsed copy of the module, for this package's
launches only.

FlyDSL 0.4.0's LLVM can translate one module into different code objects from one
build to the next: its ReassociatePass orders operands by where malloc put them.
Run on a copy parsed back from the module's text, ``gpu-module-to-binary`` gives
our launches the same code object in every build, so a code object checked once
(``test/kernels/test_kernarg_lint.py``) is the one every server builds. This is a
mitigation of the LLVM bug, not a fix, and it does not settle every kernel: on
some of AITER's FlyDSL kernels it makes builds vary that are otherwise stable. So
it applies only while FlyDSL compiles a function defined in this package; every
other compile in the process runs unchanged.

Importing this module installs it. Remove it once FlyDSL's LLVM no longer depends
on allocation order.
"""

import contextvars

import flydsl.compiler.jit_function as _jit
from flydsl._mlir import ir
from flydsl.compiler.external_llvm import _replace_gpu_module_with_binary_op

_PACKAGE = __name__.split(".")[0]
# whether the specialization being compiled is one of this package's
_ours = contextvars.ContextVar("sgl_amd_mono_compile", default=False)
_run_pipeline = _jit._run_pipeline
_trace_and_compile = _jit.JitFunction._trace_and_compile


def _trace_and_compile_marked(self, *args, **kwargs):
    token = _ours.set(self.func.__module__.split(".")[0] == _PACKAGE)
    try:
        return _trace_and_compile(self, *args, **kwargs)
    finally:
        _ours.reset(token)


def _run_pipeline_scoped(module, fragments, *, verifier, print_after_all):
    """FlyDSL's pipeline on ``module``. For this package's launches, its last pass, the
    binary pass, runs on a copy parsed from the text of ``module`` before it, and the
    copy's binary then replaces the module's ``gpu.module``."""
    *passes, last = fragments
    if not (_ours.get() and last.strip().startswith("gpu-module-to-binary")):
        return _run_pipeline(
            module, fragments, verifier=verifier, print_after_all=print_after_all
        )
    _run_pipeline(module, passes, verifier=verifier, print_after_all=print_after_all)
    copy = ir.Module.parse(
        module.operation.get_asm(enable_debug_info=True), context=module.context
    )
    _run_pipeline(copy, [last], verifier=verifier, print_after_all=print_after_all)
    _replace_gpu_module_with_binary_op(module, copy)


_jit.JitFunction._trace_and_compile = _trace_and_compile_marked
_jit._run_pipeline = _run_pipeline_scoped
