// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <device_lower/lower2device.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/utils.h>
#include <tests/cpp/utils.h>

using namespace nvfuser;

//------------------------------------------------------------------------------

static void setupRMSNorm_BWD(Fusion* fusion, DataType dtype) {
  FusionGuard fg(fusion);

  NVF_ERROR(
      dtype == DataType::Float || dtype == DataType::Half ||
      dtype == DataType::BFloat16);

  // setup fusion
  auto grad_out = makeContigTensor(2, dtype);
  auto input = makeContigTensor(2, dtype);
  auto weight = makeContigTensor(1, dtype);
  auto rstd = TensorViewBuilder()
                  .contiguity({false, std::nullopt})
                  .shape({-1, 1})
                  .dtype(dtype)
                  .build();

  fusion->addInput(grad_out);
  fusion->addInput(input);
  fusion->addInput(weight);
  fusion->addInput(rstd);

  if (dtype == DataType::Half) {
    grad_out = castOp(DataType::Float, grad_out);
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
    rstd = castOp(DataType::Float, rstd);
  }

  auto rms_norm_results =
      rms_norm_backward(grad_out, input, {1}, rstd, weight, {true, true, true});

  if (dtype != DataType::Float) {
    rms_norm_results.grad_input = castOp(dtype, rms_norm_results.grad_input);
    rms_norm_results.grad_weight = castOp(dtype, rms_norm_results.grad_weight);
  }

  fusion->addOutput(rms_norm_results.grad_input);
  fusion->addOutput(rms_norm_results.grad_weight);
}

static void NvFuserScheduler_RMSNorm_BWD(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(
      dtype == DataType::Float || dtype == DataType::Half ||
      dtype == DataType::BFloat16);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor grad_out = at::randn(input_shape, options);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[1]}, options);
  at::Tensor rstd = at::randn({input_shape[0], 1}, options);

  KernelArgumentHolder args({grad_out, input, weight, rstd});

  runBenchmarkIterations(benchmark_state, executor_cache, args);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (3 * input.numel() + weight.numel() + rstd.numel()) *
      dataTypeSizeByte(dtype));
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_RMSNorm_BWD_fp16,
    setupRMSNorm_BWD,
    NvFuserScheduler_RMSNorm_BWD,
    DataType::Half);

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_RMSNorm_BWD_fp32,
    setupRMSNorm_BWD,
    NvFuserScheduler_RMSNorm_BWD,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_BWD_fp16)
    ->Apply(addCasesOneWave128To32K)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_BWD_fp16)
    ->Apply(addCases16Wave128To32K)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_BWD_fp32)
    ->Apply(addCasesOneWave128To32K)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_BWD_fp32)
    ->Apply(addCases16Wave128To32K)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// TODO: Automatically disable/enable if bf16 is supported
// NVFUSER_BENCHMARK_DEFINE(
//     NvFuserScheduler_RMSNorm_BWD_bf16,
//     setupRMSNorm_BWD,
//     NvFuserScheduler_RMSNorm_BWD,
//     DataType::BFloat16);

// NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_BWD_bf16)
//     ->RangeMultiplier(2)
//     ->Ranges({{16, 64}})
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();

// NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_BWD_bf16)
//     ->RangeMultiplier(2)
//     ->Ranges({{28, 56}})
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();

// NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_BWD_bf16)
//     ->RangeMultiplier(2)
//     ->Ranges({{24, 48}})
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();
