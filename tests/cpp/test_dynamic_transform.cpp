// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <functional>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <csrc/exceptions.h>
#include <dynamic_transform.h>
#include <expr_evaluator.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using DynamicTransformTest = NVFuserTest;

// Simple test of analyzing dynamic reshape
TEST_F(DynamicTransformTest, DynamicTransform1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto reshape_shape0 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(reshape_shape0);
  auto reshape_shape1 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(reshape_shape1);

  auto tv2 = reshape(tv0, {reshape_shape0, reshape_shape1});
  auto tv3 = add(tv1, tv2);

  fusion.addOutput(tv3);

  // tv2 has symbolic axes as reshape is dynamic
  NVF_CHECK(
      tv2->domain()->hasSymbolicAxis(),
      "Expected to have symbolic axes: ",
      tv2->toString());

  // The symbolic axes of tv2 should not be propagated to tv3 as tv1
  // is fully concrete
  NVF_CHECK(
      !tv3->domain()->hasSymbolicAxis(),
      "Not expected to have symbolic axes: ",
      tv3->toString());

  {
    ExpressionEvaluator expr_eval;

    // input: 4, 3
    // output: 3, 4
    expr_eval.bind(tv0->axis(0)->extent(), 4L);
    expr_eval.bind(tv0->axis(1)->extent(), 3L);
    expr_eval.bind(reshape_shape0, 3L);
    expr_eval.bind(reshape_shape1, 4L);
    // We cannot infer the shape of tv1 from the above bound values, since
    // either axis of tv2 might be broadcast against one from tv1.
    expr_eval.bind(tv1->axis(0)->extent(), 3L);
    expr_eval.bind(tv1->axis(1)->extent(), 4L);

    auto initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);
    NVF_CHECK(
        info.getReshapeTransforms().size() == 1,
        "Expected to have one reshape transform: ",
        info.toString());
  }

  {
    ExpressionEvaluator expr_eval;

    // input: 4, 3
    // output: 3, -1
    expr_eval.bind(tv0->axis(0)->extent(), 4L);
    expr_eval.bind(tv0->axis(1)->extent(), 3L);
    expr_eval.bind(reshape_shape0, 3L);
    expr_eval.bind(reshape_shape1, -1L);

    // This should throw an exception since any reshape size of -1 must be
    // specified as a definition-time constant, as opposed to an input scalar.
    EXPECT_THAT(
        [&]() {
          auto initial_info = DynamicTransform::getInitialInfo(&fusion);
          auto info =
              DynamicTransformConcretizationInfo(&initial_info, &expr_eval);
        },
        ::testing::ThrowsMessage<nvfError>(::testing::HasSubstr(
            "Values of -1 passed to reshape must be constant at definition")));
  }

  {
    ExpressionEvaluator expr_eval;

    // input: 4, 3
    // output: 5, 4
    expr_eval.bind(tv0->axis(0)->extent(), 4L);
    expr_eval.bind(tv0->axis(1)->extent(), 3L);
    expr_eval.bind(reshape_shape0, 5L);
    expr_eval.bind(reshape_shape1, 4L);

    // This should fail as (4 * 3) is not equal to (5 * 4)
    EXPECT_THAT(
        [&]() {
          auto initial_info = DynamicTransform::getInitialInfo(&fusion);
          auto info =
              DynamicTransformConcretizationInfo(&initial_info, &expr_eval);
        },
        ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
            "Total element counts across view operation must match:")));
  }
}

// Reshape a tensor like another tensor
TEST_F(DynamicTransformTest, DynamicTransform2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // All tensors are 2D symbolic tensors. tv1 and tv2 have the same shape
  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  // Reshape to the same shape as tv1
  auto tv3 = reshape(tv0, {tv1->axis(0)->extent(), tv1->axis(1)->extent()});
  auto tv4 = add(tv1, tv2);
  auto tv5 = add(tv3, tv4);
  fusion.addOutput(tv5);

  {
    ExpressionEvaluator expr_eval;

    // input: 4, 3
    // output: 3, 4
    expr_eval.bind(tv0->axis(0)->extent(), 4L);
    expr_eval.bind(tv0->axis(1)->extent(), 3L);
    // Bind only tv2 extents. It should be enough as tv1 has the same
    // shape
    expr_eval.bind(tv2->axis(0)->extent(), 3L);
    expr_eval.bind(tv2->axis(1)->extent(), 4L);

    auto initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

    NVF_CHECK(
        info.getReshapeTransforms().size() == 1,
        "Expected to have one reshape transform: ",
        info.toString());
  }
}

// Analyze dynamic reshape and concretize
TEST_F(DynamicTransformTest, DynamicTransform3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto reshape_shape0 = IrBuilder::create<Val>(DataType::Int);
  auto reshape_shape1 = IrBuilder::create<Val>(DataType::Int);

  auto tv2 = reshape(tv0, {reshape_shape0, reshape_shape1});
  auto tv3 = add(tv1, tv2);

  fusion.addOutput(tv3);

  std::vector<int64_t> shape_before({4, 3});
  std::vector<int64_t> shape_after({3, 4});

  ExpressionEvaluator expr_eval;

  // input: 4, 3
  // output: 3, 4
  expr_eval.bind(tv0->axis(0)->extent(), shape_before.at(0));
  expr_eval.bind(tv0->axis(1)->extent(), shape_before.at(1));
  expr_eval.bind(tv1->axis(0)->extent(), shape_after.at(0));
  expr_eval.bind(tv1->axis(1)->extent(), shape_after.at(1));
  // We cannot infer reshape_shape0 and reshape_shape1 from tv0's and tv1's
  // extents alone, since either of these reshaped extents could either match
  // that of tv1 or be 1, resulting in a broadcast.
  expr_eval.bind(reshape_shape0, shape_after.at(0));
  expr_eval.bind(reshape_shape1, shape_after.at(1));

  auto initial_info = DynamicTransform::getInitialInfo(&fusion);
  auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

  DynamicTransform::concretizeFusion(&fusion, &info);
  NVF_CHECK(
      !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(shape_before, options);
  at::Tensor t1 = at::randn(shape_after, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

// Test multiple patterns of reshape
TEST_F(DynamicTransformTest, DynamicTransform4) {
  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
      before_after_shapes = {
          {{4, 3}, {3, 4}},
          {{4, 3}, {12, 1}},
          {{4, 3}, {4, 3}},
          {{4, 6}, {4, 2, 3}},
      };
  for (const auto& before_after : before_after_shapes) {
    const auto& before_shape = before_after.first;
    const auto& after_shape = before_after.second;

    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(before_shape.size());
    fusion.addInput(tv0);
    auto tv1 = makeSymbolicTensor(after_shape.size());
    fusion.addInput(tv1);

    std::vector<Val*> shape_arg;
    for (const auto i : arange(after_shape.size())) {
      (void)i;
      shape_arg.push_back(IrBuilder::create<Val>(DataType::Int));
    }

    auto tv2 = reshape(tv0, shape_arg);

    // tv3 will also have symbolic axes
    auto tv3 = set(tv2);
    auto tv4 = add(tv1, tv3);

    fusion.addOutput(tv4);

    ExpressionEvaluator expr_eval;

    for (const auto i : arange(before_shape.size())) {
      expr_eval.bind(tv0->axis(i)->extent(), before_shape.at(i));
    }

    for (const auto i : arange(after_shape.size())) {
      expr_eval.bind(tv2->axis(i)->extent(), after_shape.at(i));
      // We must bind tv1's extents, since they cannot be inferred until after
      // concretization. Because tv2 is a dynamic reshape both its IterDomains
      // are Symbolic, which means both of tv3's IterDomains are also Symbolic.
      // tv1 has both IterDomains of type Iteration, but it since we add tv3 to
      // it to get tv4, we do not know whether this will resolve broadcasts from
      // tv3 or not until concretization.
      expr_eval.bind(tv1->axis(i)->extent(), after_shape.at(i));
    }

    auto initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

    DynamicTransform::concretizeFusion(&fusion, &info);

    NVF_CHECK(
        !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");
  }
}

// Dynamic reshape followed by static resize
TEST_F(DynamicTransformTest, DynamicTransform5) {
  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
      before_after_shapes = {
          {{4, 3}, {3, 4}},
          //{{4, 3}, {12, 1}}, not possible to do pad a broadcast domain yet
      };

  for (auto before_after : before_after_shapes) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(2);
    fusion.addInput(tv0);

    auto reshape_shape0 = IrBuilder::create<Val>(DataType::Int);
    fusion.addInput(reshape_shape0);
    auto reshape_shape1 = IrBuilder::create<Val>(DataType::Int);
    fusion.addInput(reshape_shape1);

    auto tv1 = reshape(tv0, {reshape_shape0, reshape_shape1});
    auto tv2 =
        pad(tv1,
            {IrBuilder::create<Val>(1L),
             IrBuilder::create<Val>(1L),
             IrBuilder::create<Val>(1L),
             IrBuilder::create<Val>(1L)});
    auto tv3 = set(tv2);

    fusion.addOutput(tv3);

    ExpressionEvaluator expr_eval;

    expr_eval.bind(tv0->axis(0)->extent(), before_after.first.at(0));
    expr_eval.bind(tv0->axis(1)->extent(), before_after.first.at(1));
    expr_eval.bind(tv1->axis(0)->extent(), before_after.second.at(0));
    expr_eval.bind(tv1->axis(1)->extent(), before_after.second.at(1));

    auto initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

    DynamicTransform::concretizeFusion(&fusion, &info);

    NVF_CHECK(
        !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");
  }
}

// Reshape of reshape
TEST_F(DynamicTransformTest, DynamicTransform6) {
  std::vector<std::vector<std::vector<int64_t>>> reshape_lists = {
      {{4, 3}, {3, 4}},
      {{4, 3}, {3, 4}, {12}},
      {{4, 3}, {3, 1, 4}, {12, 1}},
      {{4, 3}, {12}, {3, 4}},
      {{4, 3}, {1, 2, 1, 3, 2}, {3, 4}},
  };

  for (auto reshape_list : reshape_lists) {
    std::vector<TensorView*> reshape_tvs;

    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(reshape_list.at(0).size());
    fusion.addInput(tv0);

    reshape_tvs.push_back(tv0);

    for (auto it = reshape_list.begin() + 1; it != reshape_list.end(); ++it) {
      auto shape = *it;
      std::vector<Val*> shape_arg;
      for (const auto i : arange(shape.size())) {
        (void)i;
        shape_arg.push_back(IrBuilder::create<Val>(DataType::Int));
      }

      auto tv = reshape(reshape_tvs.back(), shape_arg);
      reshape_tvs.push_back(tv);
    }
    fusion.addOutput(reshape_tvs.back());

    ExpressionEvaluator expr_eval;

    for (const auto i : arange(reshape_list.size())) {
      const auto& shape = reshape_list.at(i);
      for (const auto j : arange(shape.size())) {
        expr_eval.bind(reshape_tvs.at(i)->axis(j)->extent(), shape.at(j));
      }
    }

    auto initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

    DynamicTransform::concretizeFusion(&fusion, &info);

    NVF_CHECK(
        !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");
  }
}

// Test equality of DynamicTransformInfo
TEST_F(DynamicTransformTest, DynamicTransform7) {
  // Represents a series of reshapes
  struct TransformList {
    std::vector<std::vector<int64_t>> shapes;
  };

  struct ShapeInfo {
    TransformList ref_transform;
    std::vector<TransformList> equal_transforms;
    std::vector<TransformList> different_transforms;
  };

  std::vector<ShapeInfo> patterns;

  patterns.push_back(ShapeInfo{
      .ref_transform = {{{3, 4}, {4, 3}}},
      .equal_transforms =
          {{{{3, 4}, {4, 3}}}, {{{2, 8}, {4, 4}}}, {{{3, 8}, {4, 6}}}},
      .different_transforms = {{{{3, 4}, {2, 6}}}}});

  patterns.push_back(ShapeInfo{
      .ref_transform = {{{3, 4}, {12}, {1, 4, 3}}},
      .equal_transforms =
          {
              {{{3, 4}, {12}, {1, 4, 3}}},
              {{{5, 8}, {40}, {1, 4, 10}}},
          },
      .different_transforms = {
          {{{3, 4}, {12}, {4, 1, 3}}},
          {{{3, 4}, {12}, {4, 3, 1}}},
      }});

  for (const auto& pattern : patterns) {
    const auto& ref_transform = pattern.ref_transform;
    std::vector<TensorView*> reshape_tvs;

    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(ref_transform.shapes.at(0).size());
    fusion.addInput(tv0);

    reshape_tvs.push_back(tv0);

    for (auto it = ref_transform.shapes.begin() + 1;
         it != ref_transform.shapes.end();
         ++it) {
      const auto& shape = *it;
      std::vector<Val*> shape_arg;
      for (const auto i : arange(shape.size())) {
        (void)i;
        shape_arg.push_back(IrBuilder::create<Val>(DataType::Int));
      }

      auto tv = reshape(reshape_tvs.back(), shape_arg);
      reshape_tvs.push_back(tv);
    }
    fusion.addOutput(reshape_tvs.back());

    ExpressionEvaluator ref_expr_eval;

    for (const auto i : arange(ref_transform.shapes.size())) {
      const auto& shape = ref_transform.shapes.at(i);
      for (const auto j : arange(shape.size())) {
        ref_expr_eval.bind(reshape_tvs.at(i)->axis(j)->extent(), shape.at(j));
      }
    }

    auto ref_initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto ref_info =
        DynamicTransformConcretizationInfo(&ref_initial_info, &ref_expr_eval);

    for (const auto& transform : pattern.equal_transforms) {
      NVF_CHECK(transform.shapes.size() == ref_transform.shapes.size());
      ExpressionEvaluator expr_eval;
      for (const auto i : arange(transform.shapes.size())) {
        const auto& shape = transform.shapes.at(i);
        for (const auto j : arange(shape.size())) {
          expr_eval.bind(reshape_tvs.at(i)->axis(j)->extent(), shape.at(j));
        }
      }

      auto initial_info = DynamicTransform::getInitialInfo(&fusion);
      auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

      NVF_CHECK(
          ref_info == info,
          "Expected to be equal: ",
          ref_info.toString(),
          "\n",
          info.toString());
    }

    for (const auto& transform : pattern.different_transforms) {
      NVF_CHECK(transform.shapes.size() == ref_transform.shapes.size());
      ExpressionEvaluator expr_eval;
      for (const auto i : arange(transform.shapes.size())) {
        const auto& shape = transform.shapes.at(i);
        for (const auto j : arange(shape.size())) {
          expr_eval.bind(reshape_tvs.at(i)->axis(j)->extent(), shape.at(j));
        }
      }

      auto initial_info = DynamicTransform::getInitialInfo(&fusion);
      auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

      NVF_CHECK(
          ref_info != info,
          "Expected to be different: ",
          ref_info.toString(),
          "\n",
          info.toString());
    }
  }
}

// Make sure non-dynamic reshape op is created when possible
TEST_F(DynamicTransformTest, DynamicTransform8) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({3, 4});
  fusion.addInput(tv0);

  auto tv1 =
      reshape(tv0, {IrBuilder::create<Val>(4L), IrBuilder::create<Val>(3L)});
  fusion.addOutput(tv1);

  // Make sure the reshape is recognized as a static reshape
  NVF_CHECK(
      !tv1->domain()->hasSymbolicAxis(),
      "Not expected to have symbolic axes: ",
      tv1->toString());
}

// Mix of static and dynamic reshape. Make sure only dynamic reshape
// is handled by the dynamic transform concretizer.
TEST_F(DynamicTransformTest, DynamicTransform9) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = reshape(tv0, {3, 4}, {4, 3});

  auto reshape_shape0 = IrBuilder::create<Val>(DataType::Int);

  auto tv2 = reshape(tv1, {reshape_shape0});
  fusion.addOutput(tv2);

  // The first reshape is static
  NVF_CHECK(
      !tv1->domain()->hasSymbolicAxis(),
      "Unexpected to have symblic axes: ",
      tv1->toString());
  // The second reshape is static
  NVF_CHECK(
      tv2->domain()->hasSymbolicAxis(),
      "Expected to have symblic axes: ",
      tv2->toString());

  ExpressionEvaluator expr_eval;

  expr_eval.bind(tv0->axis(0)->extent(), 3L);
  expr_eval.bind(tv0->axis(1)->extent(), 4L);
  expr_eval.bind(reshape_shape0, 12L);

  auto initial_info = DynamicTransform::getInitialInfo(&fusion);
  auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

  // There must be only one dynamic reshape entry, and that must be
  // for tv2.
  NVF_CHECK(
      info.getReshapeTransforms().size() == 1,
      info.getReshapeTransforms().at(0).first == 0, // first and only reshape
      "Unexpected dynamic transform info:",
      info.toString());
}

// Make sure inherited symbolic IDs are concretized through producer projection
TEST_F(DynamicTransformTest, DynamicTransform10) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = reshape(
      tv0,
      {IrBuilder::create<Val>(DataType::Int),
       IrBuilder::create<Val>(DataType::Int)});
  auto tv2 = slice(
      tv1,
      {Slice(),
       {IrBuilder::create<Val>(1L),
        sub(tv1->axis(0)->extent(), IrBuilder::create<Val>(1L))}});
  fusion.addOutput(tv2);

  // tv2 has an producer projection (i.e., resize). The input to the expr is
  // symbolic, so is the output. When concretized, both of the input
  // and output must be concretized.

  ExpressionEvaluator expr_eval;

  expr_eval.bind(tv0->axis(0)->extent(), 3L);
  expr_eval.bind(tv0->axis(1)->extent(), 4L);
  expr_eval.bind(tv1->axis(0)->extent(), 4L);
  expr_eval.bind(tv1->axis(1)->extent(), 3L);

  auto initial_info = DynamicTransform::getInitialInfo(&fusion);
  auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

  DynamicTransform::concretizeFusion(&fusion, &info);

  NVF_CHECK(
      !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");
}

// Simple test of hashing. Create concretization info objects with two
// similar but different reshape sizes and see if their hashes are different.
TEST_F(DynamicTransformTest, DynamicTransform11) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = reshape(
      tv0,
      {IrBuilder::create<Val>(DataType::Int),
       IrBuilder::create<Val>(DataType::Int),
       IrBuilder::create<Val>(DataType::Int)});
  fusion.addOutput(tv1);

  ExpressionEvaluator expr_eval1;
  // input: 4, 3
  // output: 2, 2, 3
  expr_eval1.bind(tv0->axis(0)->extent(), 4L);
  expr_eval1.bind(tv0->axis(1)->extent(), 3L);
  expr_eval1.bind(tv1->axis(0)->extent(), 2L);
  expr_eval1.bind(tv1->axis(1)->extent(), 2L);
  expr_eval1.bind(tv1->axis(2)->extent(), 3L);

  auto initial_info1 = DynamicTransform::getInitialInfo(&fusion);
  auto info1 = DynamicTransformConcretizationInfo(&initial_info1, &expr_eval1);

  ExpressionEvaluator expr_eval2;
  ;
  // input: 4, 3
  // output: 3, 2, 2
  expr_eval2.bind(tv0->axis(0)->extent(), 4L);
  expr_eval2.bind(tv0->axis(1)->extent(), 3L);
  expr_eval2.bind(tv1->axis(0)->extent(), 3L);
  expr_eval2.bind(tv1->axis(1)->extent(), 2L);
  expr_eval2.bind(tv1->axis(2)->extent(), 2L);

  auto initial_info2 = DynamicTransform::getInitialInfo(&fusion);
  auto info2 = DynamicTransformConcretizationInfo(&initial_info2, &expr_eval2);

  // Generally different concretizations doesn't always mean different
  // hashes, but in this case they should be different
  auto hash1 = std::hash<DynamicTransformConcretizationInfo>{}(info1);
  auto hash2 = std::hash<DynamicTransformConcretizationInfo>{}(info2);
  NVF_CHECK(
      hash1 != hash2,
      "Unexpected hash collision: ",
      hash1,
      " for\n",
      info1.toString(),
      "and\n",
      info2.toString());
}

// Test FusionExecutorCache with dynamic reshapes
TEST_F(DynamicTransformTest, DynamicTransformFusionExecutorCache) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion->addInput(tv1);

  auto tv2 = reshape(tv0, {tv1->axis(0)->extent(), tv1->axis(1)->extent()});
  auto tv3 = add(tv1, tv2);

  fusion->addOutput(tv3);

  // tv2 has symbolic axes as reshape is dynamic
  NVF_CHECK(
      tv2->domain()->hasSymbolicAxis(),
      "Expected to have symbolic axes: ",
      tv2->toString());

  // The symbolic axes of tv2 should not be propagated to tv3 as tv1
  // is fully concrete
  NVF_CHECK(
      !tv3->domain()->hasSymbolicAxis(),
      "Not expected to have symbolic axes: ",
      tv3->toString());

  FusionExecutorCache executor_cache(std::move(fusion));

  NVF_CHECK(
      executor_cache.countRuntimes() == 0, "Expect to start with no runtimes");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  { // trivial reshape
    auto t0 = at::randn({3, 4}, options);
    auto t1 = at::randn({3, 4}, options);
    auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
    testValidate(
        executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
    NVF_CHECK(
        executor_cache.countRuntimes() == 1,
        "Expect to create a single runtime");
  }
  { // non-trivial reshape: merge and split
    auto t0 = at::randn({3, 4}, options);
    auto t1 = at::randn({4, 3}, options);
    auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
    testValidate(
        executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
    auto num_rts = executor_cache.countRuntimes();
    auto num_concs = executor_cache.countConcretizations();
    NVF_CHECK(num_rts == 2, "Non-trivial reshape should create new runtime");
    NVF_CHECK(
        num_concs == 2,
        "Non-trivial reshape should create new concretization cache level");
  }
  { // different non-trivial reshape
    auto t0 = at::randn({2, 6}, options);
    auto t1 = at::randn({4, 3}, options);
    auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
    testValidate(
        executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
    auto num_rts = executor_cache.countRuntimes();
    auto num_concs = executor_cache.countConcretizations();
    NVF_CHECK(
        num_rts == 2,
        "Second non-trivial reshape should not create new runtime");
    NVF_CHECK(
        num_concs == 2,
        "Second non-trivial reshape should not create new concretization cache "
        "level");
  }
}

using shape_t = std::vector<int64_t>;
using dynamic_view_invocation = std::tuple<
    shape_t, // input_shape
    shape_t, // output_shape
    bool // expect miss
    >;

//! Given a collection of input/output shapes test that FusionExecutorCache
//! properly caches concretized Fusions. The first argument is a vector of
//! input/output shape pairs. Each of these shape pairs will be run using the
//! same FusionExecutorCache. The argument expect_miss indicates whether we
//! expect a cache hit or miss at the concretization level.
//! reshape_before_reduction has the same meaning as in reductionViewAddFusion
//! in test_gpu_view.cpp.
void reductionDynamicViewAddFusion(
    std::vector<dynamic_view_invocation>& invocations,
    bool reshape_before_reduction) {
  constexpr int kReductionAxis = -1;

  auto input_dims = std::get<0>(invocations[0]).size();
  auto output_dims = std::get<1>(invocations[0]).size();

  auto bias_dims = (reshape_before_reduction) ? input_dims : output_dims;

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(input_dims);
  TensorView* bias = makeSymbolicTensor(bias_dims);
  fusion.addInput(x);
  fusion.addInput(bias);

  auto tv1 =
      (reshape_before_reduction) ? add(x, bias) : sum(x, {kReductionAxis});
  // create vectors of input scalars describing this reshape
  std::vector<Val*> output_shape(output_dims);
  for (size_t i : arange(output_dims)) {
    output_shape[i] = IrBuilder::create<Val>(DataType::Int);
    fusion.addInput(output_shape[i]);
  }
  auto x_reshape = reshape(tv1, output_shape);
  auto y = (reshape_before_reduction) ? sum(x_reshape, {kReductionAxis})
                                      : add(x_reshape, bias);
  fusion.addOutput(y);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  size_t num_concretizations = executor_cache.countConcretizations();
  // Check that concretizations and runtimes are cache misses only when they
  // should be
  auto checkCache = [&](bool expect_miss) {
    auto current = executor_cache.countConcretizations();
    ASSERT_EQ(current, num_concretizations + (size_t)expect_miss);
    num_concretizations = current;
  };

  for (auto& inv : invocations) {
    // Shmoo tests can occupy a lot of memory due to allocating many
    // different tensor sizes. So in order to avoid an OOM during this
    // test, we manually clear the allocator after it's reached a certain
    // threshold.
    maybeClearAllocator();

    auto input_shape = std::get<0>(inv);
    auto output_shape = std::get<1>(inv);
    auto expect_miss = std::get<2>(inv);

    NVF_ERROR(input_shape.size() == input_dims);
    NVF_ERROR(output_shape.size() == output_dims);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    at::Tensor at_x = at::randn(input_shape, options);
    auto bias_shape = (reshape_before_reduction) ? input_shape : output_shape;
    if (!reshape_before_reduction) {
      // When bias_shape = output_shape, it may contain -1s
      // concretize bias_shape so that we can properly initialize at_bias
      size_t other_numel = 1;
      ssize_t negone_dim = -1; // negative if no -1 shape is provided
      for (auto i : arange(bias_shape.size())) {
        if (bias_shape[i] == -1) {
          ASSERT_EQ(negone_dim, -1); // test cases should not have multiple -1s
          negone_dim = -1;
        } else {
          other_numel *= bias_shape[i];
        }
      }
      if (negone_dim >= 0) {
        bias_shape[negone_dim] = (int64_t)at_x.numel() / (int64_t)other_numel;
      }
    }
    at::Tensor at_bias = at::randn(bias_shape, options);
    KernelArgumentHolder args = {at_x, at_bias};
    // Add input scalars describing the reshape size for concretization
    for (size_t i : arange(output_dims)) {
      args.push(output_shape[i]);
    }

    auto outputs = executor_cache.runFusionWithInputs(args);
    checkCache(expect_miss);

    auto at_tv1 = reshape_before_reduction ? at_x + at_bias
                                           : at::sum(at_x, kReductionAxis);
    auto at_x_reshape = at::native::view(at_tv1, output_shape);
    auto at_y = reshape_before_reduction ? at::sum(at_x_reshape, kReductionAxis)
                                         : at_x_reshape + at_bias;

    testValidate(
        executor_cache.fusion(), outputs, args, {at_y}, __LINE__, __FILE__);
  }
}

TEST_F(DynamicTransformTest, FusionDynamicReshapeReductionShmoo) {
  auto invocations = std::vector<dynamic_view_invocation>{
      {{8, 3 * 4, 7, 9}, {8, 3 * 4, 7, 9}, true}, // trivial
      {{8, 3 * 4, 7, 5}, {8, 3 * 4, 7, 5}, false}, // trivial
      {{8, 3 * 4, 7, 9}, {8, 3, 4, 7 * 9}, true}, // merge(2) osplit(1, 3)
      {{8, 3 * 4, 7, 9},
       {8, 3, 4 * 7, 9},
       true}, // merge(1) merge(2) osplit(1, 3)
      {{8, 3 * 4, 7, 5},
       {8, 3, 4 * 7, 5},
       false}, // merge(1) merge(2) osplit(1, 3)
      {{8, 3 * 5, 7, 9}, {8, 3, 5 * 7, 9}, false}, // merge(1) osplit(1, 3)

      // test passing -1 dynamically for dimension size
      // This is unsupported. See https://github.com/NVIDIA/Fuser/issues/249
      // Values of -1 must be passed as constants instead of input-dependent
      // scalars.
      //{{8, 3 * 5, 7, 9}, {8, 3, -1, 9}, false} // merge(1) osplit(1, 3)

      // Empty reshapes should translate to FullOp
      {{8, 0, 7, 9}, {7, 8, 0, 9}, true}, // symbolic_sizes = [ -1, -1, 0, -1 ]
      // In the case below there's now a separate Val introduced for the output
      // extent, which is zero. This is represented in
      // DynamicTransformConcretizationInfo causing cache miss
      {{8, 0, 7, 9}, {7, 8, -1, 9}, true}, // symbolic_sizes = [ -1, -1, 0, -1 ]
      {{8, 0, 7, 9}, {7, 8, 0, 0}, true}, // symbolic_sizes = [ -1, -1, 0, 0 ]
      {{8, 0, 7, 9}, {47, 0, 13, 0}, true}, // symbolic_sizes = [ -1, 0, -1, 0 ]
  };
  reductionDynamicViewAddFusion(
      invocations, true /* reshape_before_reduction */);
}

using dynamic_pad_invocation = std::tuple<
    std::vector<int64_t>, // input_shape
    std::vector<int64_t>, // pad_widths
    bool // expect miss
    >;

void reductionDynamicPadAddFusion(
    std::vector<dynamic_pad_invocation>& invocations) {
  constexpr int kReductionAxis = -1;

  auto input_dims = std::get<0>(invocations[0]).size();
  auto num_pad_widths = std::get<1>(invocations[0]).size();

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(input_dims);
  fusion.addInput(x);

  std::vector<Val*> pad_width_vals(num_pad_widths);
  for (auto i : arange(num_pad_widths)) {
    pad_width_vals[i] = IrBuilder::create<Val>(DataType::Int);
    fusion.addInput(pad_width_vals[i]);
  }
  auto x_pad = pad(x, pad_width_vals);
  auto y = sum(x_pad, {kReductionAxis});
  fusion.addOutput(y);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  // Check that concretizations and runtimes are cache misses only when they
  // should be
  size_t num_concretizations = executor_cache.getKernelRuntimes().size();
#define CHECK_CACHE(expect_miss, ...)                        \
  auto current = executor_cache.getKernelRuntimes().size();  \
  auto expected = num_concretizations + (size_t)expect_miss; \
  NVF_CHECK(                                                 \
      current == expected,                                   \
      "Expected cache size ",                                \
      expected,                                              \
      " but found ",                                         \
      current,                                               \
      ". ",                                                  \
      __VA_ARGS__);                                          \
  num_concretizations = current;

  for (auto& inv : invocations) {
    // Shmoo tests can occupy a lot of memory due to allocating many
    // different tensor sizes. So in order to avoid an OOM during this
    // test, we manually clear the allocator after it's reached a certain
    // threshold.
    maybeClearAllocator();

    auto input_shape = std::get<0>(inv);
    auto pad_widths = std::get<1>(inv);
    auto expect_miss = std::get<2>(inv);

    NVF_ERROR(input_shape.size() == input_dims);
    NVF_ERROR(pad_widths.size() == num_pad_widths);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    at::Tensor at_x = at::randn(input_shape, options);
    KernelArgumentHolder args = {at_x};
    // Add input scalars describing the reshape size for concretization
    for (size_t i : arange(pad_widths.size())) {
      args.push(pad_widths[i]);
    }

    auto outputs = executor_cache.runFusionWithInputs(args);
    CHECK_CACHE(
        expect_miss, "Input shape=", input_shape, " pad_widths=", pad_widths);

    auto at_x_pad = at::pad(at_x, pad_widths);
    auto at_y = at::sum(at_x_pad, kReductionAxis);

    testValidate(&fusion, outputs, args, __LINE__, __FILE__);
  }
}
#undef CHECK_CACHE

// Test dynamic pad for various inputs
TEST_F(DynamicTransformTest, DynamicPadShmoo) {
  // NOLINTBEGIN(bugprone-implicit-widening-of-multiplication-result)
  auto invocations = std::vector<dynamic_pad_invocation>{
      {{3, 5}, {0, 0}, true}, // trivial

      {{3, 5}, {2, 1}, false}, // simple pad of both sides
      {{3, 5}, {-1, 1}, false}, // shift by one
      // The following fails with a SIGFPE in innerReductionHeuristic
      // See https://github.com/NVIDIA/Fuser/issues/264
      //{{3, 5}, {-3, -2}, false}, // output is zero-dimensional

      // Output has size 1 so is set to broadcast.
      // This was previously "working" by concretizing the size-1 pad to
      // Iteration, even though it should be Broadcast. When set properly to
      // Broadcast, it fails with an error in ConcretizedBroadcastDomains.
      //{{3, 5}, {0, -4}, true},

      // Test full negative shifts, so output doesn't overlap input
      {{3, 5}, {-5, 2}, false},
      {{3, 5}, {2, -5}, false}, // full shift the other direction, re-use

      // The following reuses the schedule of {3, 5} inputs, and does not set
      // broadcast on the second input dimension.
      {{3, 1}, {1, 1}, false},

      // Test zero-dimensional input
      //{{3, 0}, {0, 0}, false}, // SIGFPE (see #264 above)
      {{3, 0}, {1, 1}, true}, // zero-dimensional concretizes differently
      //{{3, 0}, {-1, 1}, false}, // SIGFPE (see #264 above)
  };
  // NOLINTEND(bugprone-implicit-widening-of-multiplication-result)
  reductionDynamicPadAddFusion(invocations);
}

// Test that a Symbolic root/Broadcast logical is not concretized to
// Iteration/Iteration
TEST_F(DynamicTransformTest, FusionDynamicSliceToBroadcast) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());
  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  // tv0[:2] introduces symbolic IterDomain
  auto tv1 = slice(
      tv0, {{fusion.zeroVal(), IrBuilder::create<Val>(2L), fusion.oneVal()}});
  // tv1 has Broadcast logical, Iteration root
  auto tv2 = slice(tv1, {{fusion.zeroVal(), fusion.oneVal(), fusion.oneVal()}});
  // tv2 has a Symbolic root related to a Broadcast logical through a Resize op
  fusion.addOutput(tv2);

  // At concretization, tv1's logical will be set to Iteration, which will
  // propagate to tv2s root. This test will test that when tv2 root is
  // concretized to Iteration, it does not wind up overwriting the Broadcast
  // logical.

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({5}, options);
  auto outputs = executor_cache.runFusionWithInputs({at0});
  testValidate(&fusion, outputs, {at0}, __LINE__, __FILE__);
}

// Test that empty input to cat is concretized away
TEST_F(DynamicTransformTest, FusionDynamicEmptyCat1) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);

  auto tv3 = cat({tv0, tv1, tv2}, 0);

  fusion.addOutput(tv3);

  // Check correctness
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({5}, options);
  at::Tensor at1 = at::randn({0}, options);
  at::Tensor at2 = at::randn({3}, options);
  auto outputs = executor_cache.runFusionWithInputs({at0, at1, at2});
  testValidate(&fusion, outputs, {at0, at1, at2}, __LINE__, __FILE__);
}

// Test that empty input to cat is concretized away
TEST_F(DynamicTransformTest, FusionDynamicEmptyCat2) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 0);

  fusion.addOutput(tv2);

  // Check correctness
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({5}, options);
  at::Tensor at1 = at::randn({0}, options);
  auto outputs = executor_cache.runFusionWithInputs({at0, at1});
  testValidate(&fusion, outputs, {at0, at1}, __LINE__, __FILE__);

  // Check that fusion consists only of tv2 = set(tv0)
  auto fkr = executor_cache.getMostRecentKernelRuntime();
  auto seg_fusion = fkr->fusionSegments();
  auto output_def = seg_fusion->outputs()[0]->definition();
  EXPECT_TRUE(output_def->isA<LoadStoreOp>());
  EXPECT_EQ(output_def->as<LoadStoreOp>()->opType(), LoadStoreOpType::Set);
  EXPECT_EQ(output_def->input(0), seg_fusion->inputs()[0]);
}

// Repro of https://github.com/NVIDIA/Fuser/issues/418
TEST_F(DynamicTransformTest, DynamicTransformIssue418) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(4);
  fusion->addInput(tv0);
  auto s0 = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(s0);

  auto sh = shape(tv0);
  auto tv1 = reshape(tv0, {sh[0], div(sh[1], s0), s0, sh[2], sh[3]});
  // Reducing along axis 2 in tv1 is equivalent to a partial reduction across
  // axis 1 of tv0.
  auto vm = variance_mean(tv1, {2, 3, 4}, 0, true);
  fusion->addOutput(vm.mean);
  fusion->addOutput(vm.var);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({256, 128, 28, 28}, options);
  auto outputs = executor_cache.runFusionWithInputs({at0, 32});

  testValidate(executor_cache.fusion(), outputs, {at0, 32}, __LINE__, __FILE__);
}

TEST_F(DynamicTransformTest, Issue249) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, tv0);
  auto tv2 = reshape(
      tv1,
      {tv1->axis(0)->extent(),
       tv1->axis(2)->extent(),
       IrBuilder::create<Val>(-1L)});
  auto tv3 = add(tv2, tv2);
  fusion.addOutput(tv3);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn({2, 3, 4, 5}, options);

  auto outputs = executor_cache.runFusionWithInputs({at_x});

  testValidate(executor_cache.fusion(), outputs, {at_x}, __LINE__, __FILE__);
}

// This is just like the test above, but uses an input scalar with value -1
TEST_F(DynamicTransformTest, Issue249InputNegative1) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  auto s0 = IrBuilder::create<Val>(DataType::Int);
  auto s1 = IrBuilder::create<Val>(DataType::Int);
  auto s2 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(s0);
  fusion.addInput(s1);
  fusion.addInput(s2);

  auto tv1 = add(tv0, tv0);
  auto tv2 = reshape(tv1, {s0, s1, s2});
  auto tv3 = add(tv2, tv2);
  fusion.addOutput(tv3);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn({2, 3, 4, 5}, options);

  // Dynamic reshape sizes that are not constant at definition must be explicit:
  // no -1 allowed
  EXPECT_THROW(
      executor_cache.runFusionWithInputs({at_x, 2, 4, -1}), std::exception);

  // Passing explicit sizes works fine
  auto outputs = executor_cache.runFusionWithInputs({at_x, 2, 4, 15});

  testValidate(
      executor_cache.fusion(), outputs, {at_x, 2, 4, 15}, __LINE__, __FILE__);
}

// Test that we can squeeze Symbolic IterDomains and that we properly detect
// improper concretizations where we have squeezed a dimension with extent
// other than 1.
// See https://github.com/NVIDIA/Fuser/issues/1273
TEST_F(DynamicTransformTest, SymbolicSqueeze) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto s0 = IrBuilder::create<Val>(DataType::Index);
  auto s1 = IrBuilder::create<Val>(DataType::Index);
  fusion->addInput(tv0);
  fusion->addInput(s0);
  fusion->addInput(s1);

  auto tv1 = reshape(tv0, {s0, s1});
  auto tv2 = squeeze(
      tv1, std::vector<bool>({false, true})); // Squeeze second dimension
  fusion->addOutput(tv2);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({3, 2}, options);
  KernelArgumentHolder valid_args = {t0, 6, 1};
  // An invalid input has a second dimension that cannot be squeezed
  KernelArgumentHolder invalid_args = {t0, 2, 3};

  auto outputs = executor_cache.runFusionWithInputs(valid_args);

  testValidate(fusion, outputs, valid_args, __LINE__, __FILE__);

  // An informative error message should be given by
  // SqueezeOp::checkConcretization
  EXPECT_THAT(
      [&]() { executor_cache.runFusionWithInputs(invalid_args); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          " must concretize to IterType::Broadcast but found")));
}

// See https://github.com/NVIDIA/Fuser/issues/1468
TEST_F(DynamicTransformTest, SymbolicExpand) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto s0 = IrBuilder::create<Val>(DataType::Index);
  auto s1 = IrBuilder::create<Val>(DataType::Index);
  auto s2 = IrBuilder::create<Val>(DataType::Index);
  auto s3 = IrBuilder::create<Val>(DataType::Index);
  fusion->addInput(s0);
  fusion->addInput(s1);
  fusion->addInput(s2);
  fusion->addInput(s3);

  auto tv1 = reshape(tv0, {s0, s1});
  auto tv2 = expand(tv1, {s2, s3});
  auto tv3 = add(tv2, tv2);

  fusion->addOutput(tv3);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({3, 2}, options);
  KernelArgumentHolder valid_args = {t0, 6, 1, 6, 5};
  // An invalid input has a second dimension that cannot be expanded
  KernelArgumentHolder invalid_args = {t0, 2, 3, 2, 5};

  auto outputs = executor_cache.runFusionWithInputs(valid_args);

  testValidate(
      executor_cache.fusion(), outputs, valid_args, __LINE__, __FILE__);

  // An informative error message should be given during concretization
  EXPECT_THAT(
      [&]() { executor_cache.runFusionWithInputs(invalid_args); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Mismatch in sizes when concretizing expand.")));
}

// Test that constant zero extents are not overwritten during concretization
// with non-constant extents.
// See https://github.com/NVIDIA/Fuser/issues/1572
TEST_F(DynamicTransformTest, ConcretizeConstantExtents) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  // Explicitly cast Int to Index, so that these extents are not immediate
  // constants
  auto tv1 = reshape(
      tv0,
      {
          castOp(DataType::Index, IrBuilder::create<Val>(4096, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(32, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(3, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(128, DataType::Int)),
      });
  auto tv2 = permute(tv1, {1, 2, 0, 3});
  auto tv3 = slice(tv2, {0, 0, 0, 0}, {32, 1, 4096, 128});
  auto tv4 = reshape(
      tv3,
      {
          castOp(DataType::Index, IrBuilder::create<Val>(32, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(4096, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(128, DataType::Int)),
      });
  // Note this slice has zero extent in last dimension. RemoveEmptyPass should
  // recognize this and replace with full()
  auto tv5 = slice(tv4, {0, 0, 0}, {32, 4096, 0});

  fusion->addOutput(tv5);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4096, 12288}, options);

  auto outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Test that dynamic reductions that should result in squeezes are handled
// properly.
// See https://github.com/NVIDIA/Fuser/issues/1667
TEST_F(DynamicTransformTest, DynamicSqueezeTrivialReduction) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(3);
  fusion->addInput(tv0);

  // Explicitly cast Int to Index, so that these extents are not immediate
  // constants
  auto tv1 = reshape(
      tv0,
      {
          castOp(DataType::Index, IrBuilder::create<Val>(1, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(2, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(2, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(1, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(3, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(3, DataType::Int)),
      });
  auto tv2 = sum(tv1, {0, 2, 3, 4});
  fusion->addOutput(tv2);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 2, 9}, options);

  auto outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Same as above but for Welford ops
// See https://github.com/NVIDIA/Fuser/issues/1667
TEST_F(DynamicTransformTest, DynamicSqueezeTrivialWelford) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(3);
  fusion->addInput(tv0);

  // Explicitly cast Int to Index, so that these extents are not immediate
  // constants
  auto tv1 = reshape(
      tv0,
      {
          castOp(DataType::Index, IrBuilder::create<Val>(1, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(2, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(2, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(1, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(3, DataType::Int)),
          castOp(DataType::Index, IrBuilder::create<Val>(3, DataType::Int)),
      });
  auto res =
      variance_mean(tv1, {0, 2, 3, 4}, /*unbiased=*/true, /*keepdim=*/false);
  fusion->addOutput(res.mean);
  fusion->addOutput(res.var);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 2, 9}, options);

  auto outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(DynamicTransformTest, LoopSplit) {
  const int b = 2, s = 3, h = 96, e = 128;

  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeContigConcreteTensor({-1, -1, 12288});
  TensorView* out = reshape(
      in,
      {shape(in)[0],
       shape(in)[1],
       IrBuilder::create<Val>(96),
       IrBuilder::create<Val>(128)});
  fusion.addInput(in);
  fusion.addOutput(out);

  const int d = 2;
  auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, out}) {
    tv->setDeviceMesh(mesh);
    tv->split(2, d, /*inner_split=*/false);
    tv->axis(2)->parallelize(ParallelType::DIDx);
  }

  at::Tensor in_tensor = at::randn({b, s, h * e / d}, at::Device(at::kCUDA));
  KernelArgumentHolder args({in_tensor});
  DynamicTransform::concretizeFusion(&fusion, args);

  ASSERT_EQ(fusion.outputs().size(), 1);
  auto* concrete_out = fusion.outputs().at(0)->as<TensorView>();
  EXPECT_EQ(getShardedLogicalAxis(concrete_out, ParallelType::DIDx), 2);
}

} // namespace nvfuser
