// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <tests/cpp/utils.h>

#include <expr_evaluator.h>
#include <fusion.h>
#include <ops/all_ops.h>

namespace nvfuser {

class ExprEvalTest : public NVFuserTest {};

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

namespace {

inline void checkIntValue(
    ExpressionEvaluator& evaluator,
    Val* val,
    int64_t expected_value) {
  EXPECT_TRUE(val->isIntegralScalar());
  const auto actual_value = evaluator.evaluate(val);
  EXPECT_TRUE(actual_value.hasValue());
  EXPECT_EQ(actual_value, expected_value);
}

inline void checkConstEvaluate(
    const ExpressionEvaluator& evaluator,
    Val* val,
    at::Tensor expected_value) {
  auto actual_value = evaluator.evaluate(val);
  EXPECT_TRUE(expected_value.equal(actual_value.as<at::Tensor>()));
}

} // namespace

// Evaluate basic scalar operations with constant values
TEST_F(ExprEvalTest, Constants) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  ExpressionEvaluator evaluator;

  auto* a = IrBuilder::create<Val>(7L);
  auto* b = IrBuilder::create<Val>(3L);

  // Avoid div operation because it casts int operands to float
  checkIntValue(evaluator, neg(a), -7);
  checkIntValue(evaluator, add(a, b), 10);
  checkIntValue(evaluator, neg(mul(sub(a, b), add(a, b))), -40);
  checkIntValue(evaluator, mod(a, b), 1);
  checkIntValue(evaluator, ceilDiv(a, b), 3);
}

TEST_F(ExprEvalTest, Double) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto ten = IrBuilder::create<Val>(10.0);
  auto two = IrBuilder::create<Val>(2.0);
  auto three = IrBuilder::create<Val>(3.0);
  auto val = castOp(DataType::Int, ceilDiv(sub(ten, two), three));
  auto reference = static_cast<int64_t>(std::ceil((10.0 - 2.0) / 3.0));
  EXPECT_EQ(reference, val->evaluate());
}

// Evaluate basic scalar operations with bound values
TEST_F(ExprEvalTest, Bindings) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  ExpressionEvaluator evaluator;

  auto* a = IrBuilder::create<Val>(DataType::Int);
  auto* b = IrBuilder::create<Val>(DataType::Int);
  auto* c = add(a, b);
  auto* d = neg(ceilDiv(c, b));
  auto* e = IrBuilder::create<Val>(0L);

  // trying to evaluate before binding should give empty results
  EXPECT_FALSE(evaluator.evaluate(a).hasValue());
  EXPECT_FALSE(evaluator.evaluate(d).hasValue());

  evaluator.bind(a, 7L);
  evaluator.bind(b, 3L);

  // can't bind to concrete values
  ASSERT_ANY_THROW(evaluator.bind(e, 100L));

  checkIntValue(evaluator, c, 10);
  checkIntValue(evaluator, sub(a, b), 4);
  checkIntValue(evaluator, mod(a, b), 1);
  checkIntValue(evaluator, ceilDiv(a, b), 3);
  checkIntValue(evaluator, d, -4);

  // Reset evaluation context
  evaluator = ExpressionEvaluator();

  evaluator.bind(a, 2L);
  evaluator.bind(b, 5L);

  checkIntValue(evaluator, c, 7);
  checkIntValue(evaluator, sub(a, b), -3);
  checkIntValue(evaluator, mod(a, b), 2);
  checkIntValue(evaluator, ceilDiv(a, b), 1);
  checkIntValue(evaluator, d, -2);
}

// Evaluate known values with const expression evaluator reference
TEST_F(ExprEvalTest, ConstReference) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  ExpressionEvaluator evaluator;
  auto tv0 = makeContigTensor(1);
  auto tv1 = makeContigTensor(1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({3}, options);
  auto t1 = at::randn({3}, options);

  evaluator.bind(tv0, t0);
  evaluator.bind(tv1, t1);

  checkConstEvaluate(evaluator, tv0, t0);
  checkConstEvaluate(evaluator, neg(tv0), -t0);
  checkConstEvaluate(evaluator, add(tv0, tv1), t0 + t1);
  checkConstEvaluate(evaluator, add(tv0, neg(tv1)), t0 - t1);
}

// Verify intermediate values are added to the known_values_ map.
TEST_F(ExprEvalTest, KnownValUpdate) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  ExpressionEvaluator evaluator;
  auto tv0 = makeContigTensor(1);
  auto tv1 = makeContigTensor(1);
  auto tv2 = add(tv0, tv1);
  auto tv3 = cat({tv2, tv1}, 0);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({3}, options);
  auto t1 = at::randn({3}, options);

  evaluator.bind(tv0, t0);
  evaluator.bind(tv1, t1);

  evaluator.evaluate(tv3);
  NVF_CHECK(evaluator.isKnown(tv2));
}

// Evaluate expressions in a simple IR
TEST_F(ExprEvalTest, Basic) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a non-trivial IR
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv3->split(0, 4);

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  // 1. Create an evaluator
  ExpressionEvaluator evaluator;

  // 2. Bind values
  //
  // IMPORTANT:
  // a. The bindings are only as stable as the Vals are in the fusion graph
  // b. You must use the original (rootDomain) extents
  //  (ex. `tv0->getLogicalDomain()[0]->extent()`
  //   instead of `tv0->axis(0)->extent()`)
  //
  evaluator.bind(tv0->getLogicalDomain()[0]->extent(), 6L);
  evaluator.bind(tv0->getLogicalDomain()[1]->extent(), 128L);
  evaluator.bind(tv1->getLogicalDomain()[0]->extent(), 6L);
  evaluator.bind(tv1->getLogicalDomain()[1]->extent(), 128L);

  // 3. Evaluate and check result values
  EXPECT_EQ(tv2->domain()->nDims(), 3);
  checkIntValue(evaluator, tv2->axis(0)->extent(), 2);
  checkIntValue(evaluator, tv2->axis(1)->extent(), 4);
  checkIntValue(evaluator, tv2->axis(2)->extent(), 128);

  EXPECT_EQ(tv3->domain()->nDims(), 3);
  checkIntValue(evaluator, tv3->axis(0)->extent(), 2);
  checkIntValue(evaluator, tv3->axis(1)->extent(), 4);
  checkIntValue(evaluator, tv3->axis(2)->extent(), 128);
}

// Evaluate expressions in a more complex IR
TEST_F(ExprEvalTest, Complex) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Val>(-1.0));
  TensorView* tv2 = add(tv0, IrBuilder::create<Val>(3.0));
  TensorView* tv3 = mul(tv0, IrBuilder::create<Val>(2.0));
  TensorView* tv4 = add(tv2, tv1);
  TensorView* tv5 = add(tv4, tv3);
  TensorView* tv6 = add(tv0, tv3);

  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  tv5->reorder({{-1, 0}});

  tv6->split(0, 5);
  tv5->merge(0);

  // 1. Create an evaluator
  ExpressionEvaluator evaluator;

  // 2. Bind values
  evaluator.bind(tv0->getLogicalDomain()[0]->extent(), 129L);
  evaluator.bind(tv0->getLogicalDomain()[1]->extent(), 127L);

  // Evaluate and check extent values
  EXPECT_EQ(tv0->domain()->nDims(), 2);
  checkIntValue(evaluator, tv0->axis(0)->extent(), 129);
  checkIntValue(evaluator, tv0->axis(1)->extent(), 127);

  EXPECT_EQ(tv3->domain()->nDims(), 2);
  checkIntValue(evaluator, tv3->axis(0)->extent(), 129);
  checkIntValue(evaluator, tv3->axis(1)->extent(), 127);

  EXPECT_EQ(tv4->domain()->nDims(), 2);
  checkIntValue(evaluator, tv4->axis(0)->extent(), 129);
  checkIntValue(evaluator, tv4->axis(1)->extent(), 127);

  EXPECT_EQ(tv5->domain()->nDims(), 1);
  checkIntValue(evaluator, tv5->axis(0)->extent(), 16383);

  EXPECT_EQ(tv6->domain()->nDims(), 3);
  checkIntValue(evaluator, tv6->axis(0)->extent(), 26);
  checkIntValue(evaluator, tv6->axis(1)->extent(), 5);
  checkIntValue(evaluator, tv6->axis(2)->extent(), 127);
}

// Evaluate expressions post lowering
TEST_F(ExprEvalTest, PostLower) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a non-trivial IR
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv3->split(0, 4);

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  auto* bid_x = add(tv3->axis(0)->extent(), IrBuilder::create<Val>(0L));
  auto* tid_x = add(tv3->axis(-1)->extent(), IrBuilder::create<Val>(0L));

  // Lower
  GpuLower gpulw(&fusion);
  gpulw.run();

  // 1. Create an evaluation context
  ExpressionEvaluator evaluator;

  // 2. Bind values
  evaluator.bind(tv0->getLogicalDomain()[0]->extent(), 6L);
  evaluator.bind(tv0->getLogicalDomain()[1]->extent(), 128L);
  evaluator.bind(tv1->getLogicalDomain()[0]->extent(), 6L);
  evaluator.bind(tv1->getLogicalDomain()[1]->extent(), 128L);

  // 3. Evaluate and check result values
  EXPECT_EQ(tv2->domain()->nDims(), 3);
  checkIntValue(evaluator, tv2->axis(0)->extent(), 2);
  checkIntValue(evaluator, tv2->axis(1)->extent(), 4);
  checkIntValue(evaluator, tv2->axis(2)->extent(), 128);

  EXPECT_EQ(tv3->domain()->nDims(), 3);
  checkIntValue(evaluator, tv3->axis(0)->extent(), 2);
  checkIntValue(evaluator, tv3->axis(1)->extent(), 4);
  checkIntValue(evaluator, tv3->axis(2)->extent(), 128);

  checkIntValue(evaluator, bid_x, 2);
  checkIntValue(evaluator, tid_x, 128);
}

TEST_F(ExprEvalTest, Array) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto* a = IrBuilder::create<Val>(DataType::Int);
  auto* b = IrBuilder::create<Val>(DataType::Int);

  auto arr = IrBuilder::arrayExpr(std::vector<Val*>{a, b});

  auto aa = IrBuilder::getItemExpr(arr, fusion.zeroVal());
  auto bb = IrBuilder::getItemExpr(arr, fusion.oneVal());

  ExpressionEvaluator evaluator;
  evaluator.bind(a, 2L);
  evaluator.bind(b, 5L);

  auto arr_val = evaluator.evaluate(arr);
  std::vector<PolymorphicValue> arr_vec = {2L, 5L};
  EXPECT_EQ(arr_val, arr_vec);

  checkIntValue(evaluator, aa, 2L);
  checkIntValue(evaluator, bb, 5L);
}

TEST_F(ExprEvalTest, EmptyArray) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  EXPECT_THAT(
      [&]() {
        IrBuilder::create<Val>(
            std::vector<int64_t>{},
            ArrayType{std::make_shared<DataType>(DataType::Int), 2});
      },
      ThrowsMessage<nvfuser::nvfError>(HasSubstr("not compatible")));

  auto* a = IrBuilder::create<Val>(
      std::vector<int64_t>{},
      ArrayType{std::make_shared<DataType>(DataType::Int), 0});

  ExpressionEvaluator evaluator;
  auto arr_val = evaluator.evaluate(a);
  EXPECT_EQ(arr_val, std::vector<PolymorphicValue>{});
}

TEST_F(ExprEvalTest, Struct) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  struct A : public Struct {
    int64_t a;
    int64_t b;

    StructType type() const override {
      std::vector<StructType::FieldInfo> fields(2);
      fields.at(0) = {"a", std::make_shared<DataType>(DataType::Int), true};
      fields.at(1) = {"b", std::make_shared<DataType>(DataType::Int), false};
      return StructType::make<A>(fields, "A");
    }

    std::function<PolymorphicValue()> getter(
        const std::string& key) const override {
      if (key == "a") {
        return [this]() { return PolymorphicValue(a); };
      } else if (key == "b") {
        return [this]() { return PolymorphicValue(b); };
      } else {
        NVF_THROW("Invalid key");
      }
    }

    std::function<void(const PolymorphicValue&)> setter(
        const std::string& key) override {
      if (key == "a") {
        return [this](const PolymorphicValue& value) { a = (int64_t)value; };
      } else if (key == "b") {
        return [this](const PolymorphicValue& value) { b = (int64_t)value; };
      } else {
        NVF_THROW("Invalid key");
      }
    }
  };

  auto* a = IrBuilder::create<Val>(DataType::Int);
  auto* b = IrBuilder::create<Val>(DataType::Int);

  auto struct_ = IrBuilder::structExpr<A>({{"a", a}, {"b", b}}, "test_struct");

  auto aa = IrBuilder::getAttrExpr(struct_, "a");
  auto bb = IrBuilder::getAttrExpr(struct_, "b");

  ExpressionEvaluator evaluator;
  evaluator.bind(a, 2L);
  evaluator.bind(b, 5L);

  auto eval_struct = evaluator.evaluate(struct_);
  EXPECT_EQ((PolymorphicValue)(eval_struct->*"a"), 2L);
  EXPECT_EQ((PolymorphicValue)(eval_struct->*"b"), 5L);
  EXPECT_EQ(evaluator.evaluate(aa), 2L);
  EXPECT_EQ(evaluator.evaluate(bb), 5L);
}

TEST_F(ExprEvalTest, TensorEagerExecution) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);
  auto tv2 = add(tv0, tv1);

  at::Tensor a = at::rand({6, 128}).cuda();
  at::Tensor b = at::rand({6, 128}).cuda();

  ExpressionEvaluator evaluator;
  evaluator.bind(tv0, a);
  evaluator.bind(tv1, b);

  EXPECT_TRUE(at::allclose(evaluator.evaluate(tv2).as<at::Tensor>(), a + b));
}

TEST_F(ExprEvalTest, TensorMetaData) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = makeSymbolicTensor(2);
  auto metadata = IrBuilder::metadataExpr(tv);
  auto data = IrBuilder::getAttrExpr(metadata, "data");
  auto sizes = IrBuilder::getAttrExpr(metadata, "logical_size");
  auto strides = IrBuilder::getAttrExpr(metadata, "alloc_stride");
  auto size0 = IrBuilder::getItemExpr(sizes, fusion.zeroVal());
  auto size1 = IrBuilder::getItemExpr(sizes, fusion.oneVal());
  auto stride0 = IrBuilder::getItemExpr(strides, fusion.zeroVal());
  auto stride1 = IrBuilder::getItemExpr(strides, fusion.oneVal());

  at::Tensor a = at::rand({6, 128}).cuda();

  ExpressionEvaluator evaluator;
  evaluator.bind(tv, a);

  std::vector<int64_t> sizes_vec = {6, 128};
  std::vector<int64_t> strides_vec = {128, 1};

  EXPECT_EQ(evaluator.evaluate(data), Pointer(a.data_ptr(), tv->dtype()));
  EXPECT_EQ((std::vector<int64_t>)evaluator.evaluate(sizes), sizes_vec);
  EXPECT_EQ((std::vector<int64_t>)evaluator.evaluate(strides), strides_vec);

  checkIntValue(evaluator, size0, 6L);
  checkIntValue(evaluator, size1, 128L);
  checkIntValue(evaluator, stride0, 128L);
  checkIntValue(evaluator, stride1, 1L);
}

TEST_F(ExprEvalTest, Validation) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto a = IrBuilder::create<Val>(DataType::Int);
  auto b = IrBuilder::create<Val>(DataType::Int);
  auto one = fusion.oneVal(DataType::Int);
  auto c = add(a, one);
  auto d = add(c, b);

  ExpressionEvaluator evaluator;
  evaluator.bind(a, 299792458L);
  evaluator.bind(b, 1L);

  EXPECT_THAT(
      [&]() { evaluator.bind(c, 4L, true); },
      ThrowsMessage<nvfuser::nvfError>(
          HasSubstr("Tried to bind to a value: ")));
  EXPECT_EQ(evaluator.evaluate(c), 299792459L);
  evaluator.bind(d, 299792460L, true);
}

TEST_F(ExprEvalTest, ReverseArray) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto input = IrBuilder::create<Val>(
      DataType(ArrayType{std::make_shared<DataType>(DataType::Int), 5}));
  auto output = IrBuilder::reverseArrayExpr(input);

  ExpressionEvaluator evaluator;
  evaluator.bind(input, std::vector<int64_t>{1, 2, 3, 4, 5});

  auto expect = std::vector<int64_t>{5, 4, 3, 2, 1};
  EXPECT_EQ((std::vector<int64_t>)evaluator.evaluate(output), expect);
}

//! Test evaluating ternary ops
TEST_F(ExprEvalTest, TernaryOps) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  ExpressionEvaluator evaluator;

  auto* a = IrBuilder::create<Val>(7.0);
  auto* b = IrBuilder::create<Val>(3.8);
  auto* c = IrBuilder::create<Val>(0.8);
  auto* d = IrBuilder::create<Val>(0.2);
  auto* t = IrBuilder::create<Val>(true);
  auto* f = IrBuilder::create<Val>(false);

  // Run once without PrecomputedValues, then once with
  for ([[maybe_unused]] auto i : arange(2)) {
    EXPECT_EQ(evaluator.evaluate(clamp(b, c, a)), b->value());
    EXPECT_EQ(evaluator.evaluate(clamp(a, c, b)), b->value());
    EXPECT_EQ(evaluator.evaluate(clamp(d, c, b)), c->value());

    EXPECT_EQ(
        evaluator.evaluate(lerp(a, b, d)),
        a->value() + d->value() * (b->value() - a->value()));

    EXPECT_EQ(
        evaluator.evaluate(lerp(a, b, c)),
        a->value() + c->value() * (b->value() - a->value()));
    EXPECT_EQ(
        evaluator.evaluate(lerp(a, b, d)),
        a->value() + d->value() * (b->value() - a->value()));

    EXPECT_EQ(evaluator.evaluate(threshold(a, c, b)), a->value());
    EXPECT_EQ(evaluator.evaluate(threshold(d, c, b)), b->value());
    EXPECT_EQ(evaluator.evaluate(threshold(d, d, b)), b->value());

    EXPECT_EQ(evaluator.evaluate(where(t, a, b)), a->value());
    EXPECT_EQ(evaluator.evaluate(where(f, a, b)), b->value());

    // Now bind a PrecomputedValues
    PrecomputedValues pv(&fusion);
    evaluator.bindPrecomputedValues(&pv);
  }
}

TEST_F(ExprEvalTest, Permute) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in =
      TensorViewBuilder().shape({-1, -1, -1, 6}).dtype(DataType::Float).build();
  fusion.addInput(in);
  TensorView* out = permute(in, {0, 3, 1, 2});
  fusion.addOutput(out);

  at::Tensor in_tensor =
      at::rand({256}).cuda().as_strided({2, 3, 4, 6}, {128, 32, 8, 1});

  ExpressionEvaluator evaluator;
  evaluator.bind(in, in_tensor);
  at::Tensor out_tensor = evaluator.evaluate(out).as<at::Tensor>();
  EXPECT_THAT(out_tensor.sizes(), ElementsAre(2, 6, 3, 4));
  EXPECT_THAT(out_tensor.strides(), ElementsAre(128, 1, 32, 8));
}

TEST_F(ExprEvalTest, ReshapePermuteReshape) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in =
      TensorViewBuilder().shape({-1, 6}).dtype(DataType::Float).build();
  fusion.addInput(in);
  TensorView* out = reshape(
      in, {size(in, 0), IrBuilder::create<Val>(2), IrBuilder::create<Val>(3)});
  out = permute(out, {1, 2, 0});
  out = reshape(out, {IrBuilder::create<Val>(6), size(out, 2)});
  fusion.addOutput(out);

  at::Tensor in_tensor = at::rand({72}).cuda().as_strided({9, 6}, {8, 1});

  ExpressionEvaluator evaluator;
  evaluator.bind(in, in_tensor);
  at::Tensor out_tensor = evaluator.evaluate(out).as<at::Tensor>();

  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());
  EXPECT_THAT(out_tensor.sizes(), ElementsAre(6, 9));
  EXPECT_THAT(out_tensor.strides(), ElementsAre(1, 8));
}

TEST_F(ExprEvalTest, Reshape_ForwardBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = TensorViewBuilder()
                       .shape({-1, 6})
                       .dtype(DataType::Float)
                       .expanded({true, false})
                       .build();
  fusion.addInput(in);
  TensorView* out = reshape(
      in, {size(in, 0), IrBuilder::create<Val>(2), IrBuilder::create<Val>(3)});
  fusion.addOutput(out);

  at::Tensor in_tensor = at::rand({6}).cuda().as_strided({9, 6}, {0, 1});

  ExpressionEvaluator evaluator;
  evaluator.bind(in, in_tensor);
  at::Tensor out_tensor = evaluator.evaluate(out).as<at::Tensor>();

  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());
  EXPECT_THAT(out_tensor.sizes(), ElementsAre(9, 2, 3));
  EXPECT_THAT(out_tensor.strides(), ElementsAre(0, 3, 1));
}

TEST_F(ExprEvalTest, Reshape_SplitBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = TensorViewBuilder()
                       .shape({-1, 6})
                       .dtype(DataType::Float)
                       .expanded({false, true})
                       .build();
  fusion.addInput(in);
  TensorView* out = reshape(
      in, {size(in, 0), IrBuilder::create<Val>(2), IrBuilder::create<Val>(3)});
  fusion.addOutput(out);

  at::Tensor in_tensor = at::rand({9}).cuda().as_strided({9, 6}, {1, 0});

  ExpressionEvaluator evaluator;
  evaluator.bind(in, in_tensor);
  at::Tensor out_tensor = evaluator.evaluate(out).as<at::Tensor>();

  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());
  EXPECT_THAT(out_tensor.sizes(), ElementsAre(9, 2, 3));
  EXPECT_THAT(out_tensor.strides(), ElementsAre(1, 0, 0));
}

TEST_F(ExprEvalTest, Reshape_MergeBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = TensorViewBuilder()
                       .shape({-1, 6})
                       .dtype(DataType::Float)
                       .expanded({false, true})
                       .build();
  fusion.addInput(in);
  TensorView* out = flatten(in);
  fusion.addOutput(out);

  at::Tensor in_tensor = at::rand({9}).cuda().as_strided({9, 6}, {1, 0});

  ExpressionEvaluator evaluator;
  evaluator.bind(in, in_tensor);
  at::Tensor out_tensor = evaluator.evaluate(out).as<at::Tensor>();

  EXPECT_THAT(out_tensor.sizes(), ElementsAre(54));
  EXPECT_THAT(out_tensor.strides(), ElementsAre(1));
}

TEST_F(ExprEvalTest, SumDiv) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(2);
  TensorView* s = sum(in, {0});
  TensorView* out = div(in, s);
  fusion.addInput(in);
  fusion.addOutput(out);

  at::Tensor in_tensor = at::randn({2, 3}).cuda();

  ExpressionEvaluator evaluator;
  evaluator.bind(in, in_tensor);
  evaluator.evaluate(out);
}

// Verify that the padded inputs are not evaluated
TEST_F(ExprEvalTest, CatOp) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2);
  auto tv2 = cat({tv0, tv1}, 0);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({3, 2}, options);
  auto t1 = at::randn({3, 2}, options);

  ExpressionEvaluator evaluator;
  evaluator.bind(tv0, t0);
  evaluator.bind(tv1, t1);

  at::Tensor out = evaluator.evaluate(tv2).as<at::Tensor>();

  for (auto padded_in : tv2->definition()->inputs()) {
    EXPECT_FALSE(evaluator.isKnown(padded_in));
  }

  EXPECT_TRUE(at::equal(out, at::cat({t0, t1}, 0)));
}

TEST_F(ExprEvalTest, UnaryOpSignbit) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  ExpressionEvaluator evaluator;

  auto* a = IrBuilder::create<Val>(7.0);
  auto* b = IrBuilder::create<Val>(3.8);
  auto* c = IrBuilder::create<Val>(8);
  auto* d = IrBuilder::create<Val>(7);
  auto* e = IrBuilder::create<Val>(-0.8);

  auto* signbit_a = signbit(a);
  auto* signbit_b = signbit(b);
  auto* signbit_c = signbit(c);
  auto* signbit_d = signbit(d);
  auto* signbit_e = signbit(e);

  EXPECT_EQ(evaluator.evaluate(signbit_a).as<bool>(), false);
  EXPECT_EQ(evaluator.evaluate(signbit_b).as<bool>(), false);
  EXPECT_EQ(evaluator.evaluate(signbit_c).as<bool>(), false);
  EXPECT_EQ(evaluator.evaluate(signbit_d).as<bool>(), false);
  EXPECT_EQ(evaluator.evaluate(signbit_e).as<bool>(), true);
}

TEST_F(ExprEvalTest, BinaryOpFmod) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  ExpressionEvaluator evaluator;

  auto* a = IrBuilder::create<Val>(7.0);
  auto* b = IrBuilder::create<Val>(3.8);
  auto* c = IrBuilder::create<Val>(8);
  auto* d = IrBuilder::create<Val>(3);
  auto* e = IrBuilder::create<Val>(-0.8);

  auto* out0 = fmod(a, b);
  auto* out1 = fmod(a, c);
  auto* out2 = fmod(c, d);
  auto* out3 = fmod(c, b);
  auto* out4 = fmod(a, e);
  auto* out5 = fmod(d, e);

  EXPECT_EQ(evaluator.evaluate(out0).as<double>(), std::fmod(7.0, 3.8));
  EXPECT_EQ(evaluator.evaluate(out1).as<double>(), std::fmod(7.0, 8));
  EXPECT_EQ(evaluator.evaluate(out2).as<double>(), std::fmod(8, 3));
  EXPECT_EQ(evaluator.evaluate(out3).as<double>(), std::fmod(8, 3.8));
  EXPECT_EQ(evaluator.evaluate(out4).as<double>(), std::fmod(7.0, -0.8));
  EXPECT_EQ(evaluator.evaluate(out5).as<double>(), std::fmod(3, -0.8));
}

// Test that we properly bind tensor metadata in PrecomputedValues so that we
// can access it from an ExpressionEvaluator
TEST_F(ExprEvalTest, TensorMetadataPrecomputedValues) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto* tv1 = set(tv0);
  fusion.addOutput(tv1);

  PrecomputedValues pv(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({3, 4}, options);

  // now compute metadata of tv0
  auto metadata = fusion.metadataOf(tv0);
  ASSERT_TRUE(metadata != nullptr);
  EXPECT_EQ(metadata->dtype(), metaDataTypeOf(tv0));
  auto logical_size = IrBuilder::getAttrExpr(metadata, "logical_size");
  auto logical_size_0 = IrBuilder::getItemExpr(logical_size, fusion.zeroVal());
  auto logical_size_1 = IrBuilder::getItemExpr(logical_size, fusion.oneVal());

  pv.bindInputs({t0});
  pv.evaluate();

  ExpressionEvaluator evaluator;
  evaluator.bindPrecomputedValues(&pv);

  EXPECT_TRUE(evaluator.evaluate(metadata).hasValue());

  checkIntValue(evaluator, logical_size_0, 3);
  checkIntValue(evaluator, logical_size_1, 4);
}

TEST_F(ExprEvalTest, NamedScalar) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* cache_id = IrBuilder::create<NamedScalar>("cacheId", DataType::UInt64);

  ExpressionEvaluator evaluator;
  constexpr int64_t kCacheIdValue = 1;
  evaluator.bind("cacheId", kCacheIdValue);
  PolymorphicValue cache_id_pvalue = evaluator.evaluate(cache_id);
  EXPECT_EQ(cache_id_pvalue.as<int64_t>(), kCacheIdValue);
}

// TODO: extend to other TernaryOps
TEST_F(ExprEvalTest, TernaryOpsWhere) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2, DataType::Bool);
  auto tv1 = makeContigTensor(2);
  auto tv2 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  // tensor, tensor, tensor
  auto tv3 = where(tv0, tv1, tv2);
  // tensor, tensor, scalar
  auto tv4 = where(tv0, tv1, IrBuilder::create<Val>(1.0));
  // tensor, scalar, tensor
  auto tv5 = where(tv0, IrBuilder::create<Val>(1.0), tv2);
  // tensor, scalar, scalar
  auto tv6 =
      where(tv0, IrBuilder::create<Val>(2.0), IrBuilder::create<Val>(1.0));
  // scalar, tensor, scalar
  auto tv7 =
      where(IrBuilder::create<Val>(true), tv1, IrBuilder::create<Val>(2.0));
  // scalar, scalar, tensor
  auto tv8 =
      where(IrBuilder::create<Val>(false), IrBuilder::create<Val>(2.0), tv2);
  // scalar, tensor, tensor
  auto tv9 = where(IrBuilder::create<Val>(true), tv1, tv2);
  // scalar, scalar, scalar
  auto tv10 = where(
      IrBuilder::create<Val>(true),
      IrBuilder::create<Val>(2.0),
      IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv3);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);
  fusion.addOutput(tv7);
  fusion.addOutput(tv8);
  fusion.addOutput(tv9);
  // avoid non-tensor output which is not supported yet
  fusion.addOutput(add(tv1, tv10));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({3, 2}, options) > 0.5;
  auto t1 = at::randn({3, 2}, options);
  auto t2 = at::randn({3, 2}, options);

  ExpressionEvaluator evaluator;
  evaluator.bind(tv0, t0);
  evaluator.bind(tv1, t1);
  evaluator.bind(tv2, t2);

  at::Tensor out1 = evaluator.evaluate(tv3).as<at::Tensor>();
  at::Tensor out2 = evaluator.evaluate(tv4).as<at::Tensor>();
  at::Tensor out3 = evaluator.evaluate(tv5).as<at::Tensor>();
  at::Tensor out4 = evaluator.evaluate(tv6).as<at::Tensor>();
  at::Tensor out5 = evaluator.evaluate(tv7).as<at::Tensor>();
  at::Tensor out6 = evaluator.evaluate(tv8).as<at::Tensor>();
  at::Tensor out7 = evaluator.evaluate(tv9).as<at::Tensor>();
  at::Tensor out8 = evaluator.evaluate(add(tv1, tv10)).as<at::Tensor>();

  // verify results
  EXPECT_TRUE(at::allclose(out1, at::where(t0, t1, t2)));
  EXPECT_TRUE(at::allclose(out2, at::where(t0, t1, 1.0)));
  EXPECT_TRUE(at::allclose(out3, at::where(t0, 1.0, t2)));
  EXPECT_TRUE(at::allclose(out4, at::where(t0, 2.0, 1.0)));
  EXPECT_TRUE(at::allclose(out5, t1));
  EXPECT_TRUE(at::allclose(out6, t2));
  EXPECT_TRUE(at::allclose(out7, t1));
  EXPECT_TRUE(at::allclose(out8, t1.add(2.0)));
}

TEST_F(ExprEvalTest, Pow) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Test all combinations of pow operations
  // tensor, scalar (int64_t)
  auto tv2 = pow(tv0, IrBuilder::create<Val>(2L));
  // tensor, scalar (double)
  auto tv3 = pow(tv0, IrBuilder::create<Val>(2.0));
  // tensor, tensor
  auto tv4 = pow(tv0, tv1);
  // scalar, scalar (int64_t, int64_t)
  auto tv5 = pow(IrBuilder::create<Val>(3L), IrBuilder::create<Val>(2L));
  // scalar, scalar (double, double)
  auto tv6 = pow(IrBuilder::create<Val>(3.0), IrBuilder::create<Val>(2.0));
  // scalar, scalar (int64_t, double)
  auto tv7 = pow(IrBuilder::create<Val>(3L), IrBuilder::create<Val>(2.0));
  // scalar, scalar (double, int64_t)
  auto tv8 = pow(IrBuilder::create<Val>(3.0), IrBuilder::create<Val>(2L));
  // avoid non-tensor output which is not supported yet
  auto tv9 = add(tv0, add(add(add(tv5, tv6), tv7), tv8));
  // scalar, tensor
  auto tv10 = pow(IrBuilder::create<Val>(3.0), tv0);
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);
  fusion.addOutput(tv9);
  fusion.addOutput(tv10);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({3, 2}, options);
  auto t1 = at::randn({3, 2}, options);

  ExpressionEvaluator evaluator;
  evaluator.bind(tv0, t0);
  evaluator.bind(tv1, t1);

  // Evaluate all outputs
  at::Tensor out1 = evaluator.evaluate(tv2).as<at::Tensor>();
  at::Tensor out2 = evaluator.evaluate(tv3).as<at::Tensor>();
  at::Tensor out3 = evaluator.evaluate(tv4).as<at::Tensor>();
  at::Tensor out9 = evaluator.evaluate(tv9).as<at::Tensor>();
  at::Tensor out10 = evaluator.evaluate(tv10).as<at::Tensor>();
  EXPECT_TRUE(at::allclose(out1, at::pow(t0, 2)));
  EXPECT_TRUE(at::allclose(out2, at::pow(t0, 2.0)));
  // Needs explicit equal_nan=true
  EXPECT_TRUE(at::allclose(out3, at::pow(t0, t1), 1e-5, 1e-8, true));
  EXPECT_TRUE(at::allclose(
      out9,
      t0.add(std::pow(3L, 2L))
          .add(std::pow(3.0, 2.0))
          .add(std::pow(3L, 2.0))
          .add(std::pow(3.0, 2L))));
  EXPECT_TRUE(at::allclose(out10, at::pow(3.0, t0), 1e-5, 1e-8, true));
}
} // namespace nvfuser
