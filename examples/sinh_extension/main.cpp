// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ops/arith.h>
#include <runtime/executor.h>
#include <scheduler/all_schedulers.h>
#include <torch/extension.h>

#include <memory>

using namespace nvfuser;

at::Tensor sinh_nvfuser(const at::Tensor& input) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int dim = input.dim();
  auto dtype = input.scalar_type();
  auto x =
      TensorViewBuilder().ndims(dim).dtype(aten_to_data_type(dtype)).build();
  fusion.addInput(x);

  // Using equation sinh(x) = [ exp(x) - exp(-1) ] / 2
  auto output = div(sub(exp(x), exp(neg(x))), IrBuilder::create<Val>(2.0));
  fusion.addOutput(output);

  std::cout << "Create fusion:" << std::endl;
  fusion.print();

  auto heuristic_params =
      SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, {input});

  KernelExecutor ke;
  ke.compile(&fusion, {input}, heuristic_params->lparams);
  auto outputs = ke.run({input}, {}, heuristic_params->lparams);

  return outputs[0].as<at::Tensor>();
}

TORCH_LIBRARY(myop, m) {
  m.def("sinh_nvfuser", sinh_nvfuser);
}

TORCH_LIBRARY_IMPL(myop, CUDA, m) {
  m.impl("sinh_nvfuser", sinh_nvfuser);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
