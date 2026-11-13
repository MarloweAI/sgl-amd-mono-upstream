// SPDX-License-Identifier: Apache-2.0
// The DSpark draft's main_proj split over the TP ranks by output column, in its own launch (main_proj.h).
#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#include "main_proj.h"

#ifndef C10_HIP_KERNEL_LAUNCH_CHECK
#define C10_HIP_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

namespace {

__global__ __launch_bounds__(main_proj::THREADS) void main_proj_kernel(const main_proj::Params p) {
    main_proj::run(p, blockIdx.x, gridDim.x);
}

// x [rows, 15360] bf16; w [5120 / tp, 15360] E4M3 and ws [5120 / tp / 32, 480] E8M0 (this rank's columns of
// main_proj); out [>= rows, 5120] bf16; state [2048] int32 and work (zeroed once); peers: every rank's exchange
// buffer (int64, host), zeroed once
void main_proj_op(const at::Tensor &x, const at::Tensor &w, const at::Tensor &ws, at::Tensor &out, at::Tensor &state,
                  at::Tensor &work, const at::Tensor &peers, int64_t rank, double deadline_s) {
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(x.device());
    const main_proj::Params p = main_proj::params(x, w, ws, out, state, work, peers, rank, deadline_s);
    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(x.device().index()).stream();
    hipLaunchKernelGGL(main_proj_kernel, dim3(main_proj::grid(p)), dim3(main_proj::THREADS), 0, stream, p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("main_proj(Tensor x, Tensor w, Tensor ws, Tensor(a!) out, Tensor(b!) state, Tensor(c!) work, Tensor peers, "
          "int rank, float deadline_s) -> ()");
    m.impl("main_proj", at::kCUDA, &main_proj_op);
}
