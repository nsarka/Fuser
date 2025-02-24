// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace hir {

using HostIrIntegrationTest = NVFuserTest;

TEST_F(HostIrIntegrationTest, LaunchKernel) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeSymbolicTensor(2);
  fusion.addInput(in);

  TensorView* out = set(in);
  fusion.addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32, 32}, options);
  auto ke = std::make_unique<KernelExecutor>();
  ke->compile(&fusion, {t0});

  auto hic = std::make_unique<HostIrContainer>(1);
  FusionGuard::setCurFusion(hic.get());

  hic->setKernelExecutor(0, std::move(ke));

  IrCloner ir_cloner(hic.get());
  auto hic_in = ir_cloner.clone(in);
  auto hic_out = ir_cloner.clone(out);

  hic->addInput(hic_in);
  hic->addOutput(hic_out);

  auto launch_kernel = IrBuilder::create<LaunchKernel>(
      0,
      LaunchParams(),
      CompileParams(),
      std::vector<Val*>{hic_in},
      std::vector<Val*>{hic_out});

  hic->pushBackTopLevelExprs(launch_kernel);

  HostIrEvaluator hie(std::move(hic));

  auto outputs = hie.runWithInput({{hic_in, t0}});

  EXPECT_TRUE(outputs[0].as<at::Tensor>().equal(t0));
}

TEST_F(HostIrIntegrationTest, Set) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* out = set(in);
  fusion->addOutput(out);

  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(HostIrIntegrationTest, Sum) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeSymbolicTensor(2);
  fusion->addInput(in);

  TensorView* out = sum(in, {0});
  fusion->addOutput(out);

  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(HostIrIntegrationTest, Deallocate) {
  constexpr int64_t kForLoopStop = 1024;
  const std::vector<int64_t> sizes = {8, 64};
  uint8_t device_index = 0;

  c10::cuda::CUDACachingAllocator::resetPeakStats(device_index);

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  auto* for_loop = IrBuilder::create<ForLoop>(
      /*IterDomain=*/makeContigConcreteTensor({0})->axis(0), // unused
      /*index=*/IrBuilder::create<Val>(DataType::Index),
      /*start=*/hic->zeroVal(),
      /*stop=*/IrBuilder::create<Val>(kForLoopStop, DataType::Index),
      /*step=*/hic->oneVal(),
      /*vectorize=*/false,
      /*vectorize_shift=*/nullptr,
      /*unroll_required=*/false,
      CircularBufferLoopStage::NotApplicable,
      /*circular_buffer_loop_stage_depth=*/0);

  TensorView* tv0 = makeConcreteTensor(sizes);
  tv0->setMemoryType(MemoryType::Global);
  auto* allocate = IrBuilder::create<kir::Allocate>(tv0, MemoryType::Global);
  TensorView* tv1 = abs(tv0);

  for_loop->body().push_back(allocate);
  for_loop->body().push_back(tv1->definition());

  hic->pushBackTopLevelExprs(for_loop);
  hic->addOutput(tv1);

  HostIrEvaluator hie(std::move(hic));

  auto outputs = hie.runWithInput({});

  const c10::CachingDeviceAllocator::DeviceStats stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
  std::cout << "memory peak: " << stats.allocated_bytes[0].peak << std::endl;

  EXPECT_EQ(sizes, outputs.at(0).sizes());
}

} // namespace hir

} // namespace nvfuser
