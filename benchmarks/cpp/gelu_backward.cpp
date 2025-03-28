// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Based on NVFuserTest.FusionBiasGeluBwd_CUDA

#include <device_lower/lower2device.h>
#include <fusion.h>
#include <ir/builder.h>
#include <ops/arith.h>
#include <runtime/executor.h>
#include <scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/utils.h>
#include <tests/cpp/utils.h>

using namespace nvfuser;

static void setupFusion(Fusion* fusion) {
  FusionGuard fg(fusion);

  const float k_079 = 0.79788456;
  const float k_004 = 0.044715;
  const float k_010 = 0.1070322243;
  const int64_t k_one = 1L;

  // gradient tensor
  auto t0 = makeContigTensor(3, DataType::Half);
  fusion->addInput(t0);

  auto t1 = castOp(DataType::Float, t0);

  // bias tensor
  auto t2 = makeContigTensor(1, DataType::Half);
  fusion->addInput(t2);

  auto t3 = castOp(DataType::Float, t2);

  // input tensor
  auto t4 = makeContigTensor(3, DataType::Half);
  fusion->addInput(t4);

  auto t5 = castOp(DataType::Float, t4);
  auto t6 = broadcast(t3, {true, true, false});
  auto t7 = add(t6, t5);
  auto t8 = mul(t7, IrBuilder::create<Val>(k_079));
  auto t9 = mul(t7, IrBuilder::create<Val>(k_004));
  auto t10 = mul(t9, t7);
  auto t11 = add(t10, IrBuilder::create<Val>(k_one));
  auto t12 = mul(t8, t11);
  auto t13 = unaryOp(UnaryOpType::Tanh, t12);
  auto t14 = mul(t7, IrBuilder::create<Val>(0.5));
  auto t15 = mul(t13, t13);
  auto t16 = unaryOp(UnaryOpType::Neg, t15);
  auto t17 = add(t16, IrBuilder::create<Val>(k_one));
  auto t18 = mul(t7, IrBuilder::create<Val>(k_010));
  auto t19 = mul(t18, t7);
  auto t20 = add(t19, IrBuilder::create<Val>(k_079));
  auto t21 = mul(t17, t20);
  auto t22 = mul(t14, t21);
  auto t23 = add(t13, IrBuilder::create<Val>(k_one));
  auto t24 = mul(t23, IrBuilder::create<Val>(0.5));
  auto t25 = add(t22, t24);
  auto t26 = mul(t25, t1);

  // Save float output for validation
  fusion->addOutput(t26);
  auto t27 = castOp(DataType::Half, t26);
  fusion->addOutput(t27);
}

static KernelArgumentHolder setupInputs() {
  at::manual_seed(0);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  std::vector<int64_t> input_shape{6, 512, 4096};
  std::vector<int64_t> bias_shape{4096};
  auto at_input = at::randn(input_shape, options);
  auto at_bias = at::randn(bias_shape, options);
  auto at_grad = at::randn(input_shape, options);

  return {at_grad, at_bias, at_input};
}

//------------------------------------------------------------------------------

static void NvFuserScheduler_GeluBackward_SetupFusion(
    benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    Fusion fusion;
    setupFusion(&fusion);
  }
}

BENCHMARK(NvFuserScheduler_GeluBackward_SetupFusion)
    ->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void NvFuserScheduler_GeluBackward_AutoSchedule(
    benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    // Setup (not included in the measurement)
    benchmark_state.PauseTiming();
    Fusion fusion;
    setupFusion(&fusion);
    KernelArgumentHolder args = setupInputs();
    benchmark_state.ResumeTiming();

    // Auto-schedule
    SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, args);
  }
}

BENCHMARK(NvFuserScheduler_GeluBackward_AutoSchedule)
    ->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void NvFuserScheduler_GeluBackward_Lower(
    benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  KernelArgumentHolder args = setupInputs();

  SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, args);

  for (auto _ : benchmark_state) {
    GpuLower(&fusion).run();
  }
}

BENCHMARK(NvFuserScheduler_GeluBackward_Lower)->Unit(benchmark::kMillisecond);

//------------------------------------------------------------------------------

static void NvFuserScheduler_GeluBackward_Compile(
    benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  KernelArgumentHolder args = setupInputs();

  auto heuristic_params =
      SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, args);

  for (auto _ : benchmark_state) {
    KernelExecutor ke;
    ke.compile(&fusion, args, heuristic_params->lparams);
  }
}

BENCHMARK(NvFuserScheduler_GeluBackward_Compile)->Unit(benchmark::kMillisecond);

//------------------------------------------------------------------------------

static void NvFuserScheduler_GeluBackward_RunFusion(
    benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  KernelArgumentHolder args = setupInputs();

  // outputs
  KernelArgumentHolder outputs;

  auto heuristic_params =
      SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, args);

  KernelExecutor ke;
  ke.compile(&fusion, args, heuristic_params->lparams);

  C10_CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : benchmark_state) {
    outputs = ke.run(args, {}, heuristic_params->lparams);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
  }
}

BENCHMARK(NvFuserScheduler_GeluBackward_RunFusion)
    ->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void NvFuserScheduler_GeluBackward_RunFusion_GpuOnly(
    benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  KernelArgumentHolder args = setupInputs();

  auto heuristic_params =
      SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, args);

  KernelExecutor ke;
  ke.compile(&fusion, args, heuristic_params->lparams);

  runBenchmarkIterations(benchmark_state, &ke, args, heuristic_params->lparams);
}

BENCHMARK(NvFuserScheduler_GeluBackward_RunFusion_GpuOnly)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

static void NvFuserScheduler_GeluBackward_RunFusion_CpuOnly(
    benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  KernelArgumentHolder args = setupInputs();

  // outputs
  KernelArgumentHolder outputs;

  auto heuristic_params =
      SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, args);

  KernelExecutor ke;
  ke.setExecuteKernelFlag(false);
  ke.compile(&fusion, args, heuristic_params->lparams);

  for (auto _ : benchmark_state) {
    outputs = ke.run(args, {}, heuristic_params->lparams);
  }
}

BENCHMARK(NvFuserScheduler_GeluBackward_RunFusion_CpuOnly)
    ->Unit(benchmark::kMicrosecond);
