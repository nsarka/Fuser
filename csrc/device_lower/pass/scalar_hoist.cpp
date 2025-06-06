// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <device_lower/pass/magic_zero.h>
#include <expr_simplifier.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir_dispatch.h>
#include <options.h>

#include <device_lower/pass/scalar_hoist.h>

namespace nvfuser {

namespace {

bool shouldHoistToHost(Val* value) {
  if (value->definition() == nullptr) {
    return false;
  }
  auto def = value->definition();
  return def->isA<kir::EncodeTensorMapTiled>();
}

// Get the position of the innermost non-trivial loop
int64_t getInnermostNonTrivialLoop(const std::vector<ForLoop*>& loops) {
  int64_t position = -1;
  for (auto i : arange(loops.size())) {
    if (!loops.at(i)->isTrivial()) {
      position = (int64_t)i;
    }
  }
  return position;
}

// Find the outer-most loop nest that contains all the dependencies of `value`.
int64_t findOutermostPosWithSatisfiedDependency(
    Val* value,
    const std::vector<ForLoop*>& loops) {
  DEBUG_PRINT_SCOPE(value->toInlineString());
  // We don't recursively look into tensor indexing to find its dependency.
  // Instead, we always assume tensor indices to have dependency on all loop
  // variables and prefer to put it at the innermost loop.
  if (value->isA<kir::TensorIndex>()) {
    RECORD_AND_RETURN(getInnermostNonTrivialLoop(loops));
  }
  // For TensorView, we must find its allocation to determine which loop it
  // belongs to. TensorView is handled differently from TensorIndex because
  // TensorIndex is a tensor data access, but TensorView only contains meta data
  // access like `T1.data`, or `toSmem(T1)`.
  if (TensorView* tv = dynamic_cast<TensorView*>(value)) {
    // Can not use lower_utils::getAllocInformation to get the allocation
    // position of TensorView here, because the TensorView might be a special
    // purposed tensor view, like a mbarrier, whose IterDomains are unrelated to
    // the actual allocation.
    for (int64_t pos = (int64_t)loops.size() - 1; pos >= 0; pos--) {
      auto loop = loops.at(pos);
      for (auto expr : loop->body().exprs()) {
        if (auto alloc = dynamic_cast<kir::Allocate*>(expr)) {
          if (alloc->buffer() == tv) {
            RECORD_AND_RETURN(pos);
          }
        }
      }
    }
    RECORD_AND_RETURN(-1);
  }

  auto def = value->definition();

  // If `value` is not computed from other values, then it must either be a loop
  // variable or something like named scalar or constant
  if (def == nullptr) {
    // Check if `value` is a loop variable
    for (auto i : arange(loops.size())) {
      auto loop = loops.at(i);
      // We skip trivial loop here because it is never materialized, so its loop
      // variable is accessible everywhere. For example, if the trivial loop is
      // a un-concretized broadcast, the its loop variable is a constant 0. If
      // the trivial loop is thread parallelized, then its loop variable will be
      // threadIdx.x, which is also accessible on all scopes.
      if (loop->isTrivial()) {
        continue;
      }
      if (loop->index()->sameAs(value)) {
        RECORD_AND_RETURN((int64_t)i);
      }
    }
    // If no loop found, then `value` would could only depend on constants and
    // trivial loop variables, for example:
    // value = blockIdx.x * 256 + threadIdx.x
    // For this case, return -1, which indicates that the computation of this
    // value should be placed at top-level exprs.
    RECORD_AND_RETURN(-1);
  }

  int64_t pos = -1;

  for (auto v : def->inputs()) {
    pos = std::max(pos, findOutermostPosWithSatisfiedDependency(v, loops));
  }

  RECORD_AND_RETURN(pos);
}

// Get the key for `common_scalar_map_`
ForLoop* getLoopAtPos(const std::vector<ForLoop*>& loops, int64_t position) {
  // position < 0 refers to the top level exprs (no corresponding loop)
  if (position < 0) {
    return nullptr;
  }
  return loops.at(position);
}

// Check if in the definition of from, there is a subexpression equivalent to
// reference. If found, then return this subexpression.
Val* findRefAsSubexprOf(Val* from, Val* reference, bool exact) {
  if (!ir_utils::isFunctional(reference)) {
    return nullptr;
  }
  if (exact) {
    if (from == reference) {
      return from;
    }
  } else {
    if (from->sameAs(reference)) {
      return from;
    }
  }
  if (from->isOneOf<TensorView, kir::TensorIndex>()) {
    return nullptr;
  }
  auto def = from->definition();
  if (def != nullptr) {
    for (auto input : def->inputs()) {
      auto common_subexpr = findRefAsSubexprOf(input, reference, exact);
      if (common_subexpr != nullptr) {
        return common_subexpr;
      }
    }
  }
  return nullptr;
}

// Check if the given value is helpful to reuse it. For example
// in x = (a + b) * (a + b), it is helpful to reuse (a + b) as
// c = a + b; x = c * c, because it can reduce the number of arithmetic
// operations. However, it makes no sense to reuse T0.size[0] + T0.size[0] as
// a = T0.size[0]; x = a + a, because although T0.size[0] is a composition of
// GetItem with GetAttr, in C++, these operations are free because it just get
// erased when lowering C++ into assembly languages.
bool isHelpfulToReuse(Val* value) {
  auto def = value->definition();
  if (def == nullptr) {
    return false;
  }
  if (def->isOneOf<GetMetaData, GetAttr, GetItem>()) {
    return false;
  }
  return true;
}

// Find if the given `value` is already computed on the host. If yes, then
// return the host value, else return nullptr.
Val* reuseValsKnownToKernel(Val* value) {
  for (auto val : GpuLower::current()->allKnownVals()) {
    if (val->sameAs(value)) {
      return val;
    }
  }
  return nullptr;
}

} // namespace

std::pair<Val*, bool> CommonScalarMap::hoistScalarImpl(
    Val* value,
    const std::vector<ForLoop*>& loops,
    std::vector<Val*>& seen_subexprs,
    int64_t parent_pos,
    bool is_given) {
  if (value->isA<kir::TensorIndex>()) {
    // Current implementation of value hoisting does not have advanced data flow
    // analysis to handle tensor index. Unlike scalar, the computation of tensor
    // index might be hidden behind a predicate, making its analysis very
    // complicated. For this case we just return a true for the second return
    // value which will help us to make sure that we don't insert it into
    // `common_scalar_map_` so that we won't consider it as a reusing
    // opportunity.
    // Note that although we are unable to handle TensorIndex, we can handle
    // TensorView correctly. A scalar that depends on TensorView is usually
    // something like below: toSmem(T1), or, (char*)(T1.data). These expressions
    // just access the meta data of the tensor, instead of accessing its
    // elements. So we are fine about hoisting it because we can just find its
    // allocation and make sure that our allocation is after it.
    return {value, true};
  } else if (value->isA<TensorView>()) {
    return {value, false};
  }

  auto def = value->definition();
  if (def == nullptr || value->isConstScalar()) {
    return {value, false};
  }

  auto my_pos = findOutermostPosWithSatisfiedDependency(value, loops);
  auto my_loop = getLoopAtPos(loops, my_pos);

  // There are certain experesions that must be evaluated on the host, such as
  // kir::EncodeTensorMapTiled. For these expressions, we should add them to
  // known values of the kernel. Also, reusing subexpressions of these exprs
  // are disabled, because the subexpressions on the host is not accessible to
  // the device. This must happen before reusing, because we can not reuse
  // values on the host for device.
  if (shouldHoistToHost(value)) {
    if (auto known_val = reuseValsKnownToKernel(value)) {
      return {known_val, false};
    }
    GpuLower::current()->allKnownVals().emplace_back(value);
    return {value, false};
  }

  // Check if `value` is already computed. If yes, just reuse it and return.
  if (auto existing_subexpr = reuseScalarIfAlreadyComputed(value, my_loop)) {
    return {existing_subexpr, false};
  }

  for (auto existing_subexpr : seen_subexprs) {
    if (value->sameAs(existing_subexpr)) {
      common_scalar_map_[my_loop].emplace_back(existing_subexpr);
      hoisted_or_reused_.emplace(existing_subexpr);
      return {existing_subexpr, false};
    }
  }

  // Recursively hoist all the producers of `value`
  bool changed = false; // if any of the inputs is replaced by an existing val
  bool has_tensor_index_dependency = false;
  std::vector<Val*> inputs;
  for (auto input : def->inputs()) {
    auto hoist = hoistScalarImpl(input, loops, seen_subexprs, my_pos);
    inputs.emplace_back(hoist.first);
    if (hoist.second) {
      has_tensor_index_dependency = true;
    }
    if (inputs.back() != input) {
      changed = true;
    }
  }

  // If any of the inputs is replaced, then create a new expression whose inputs
  // are replaced with hoisted input
  if (changed) {
    value = IrBuilder::create<Val>(*value->getDataType());
    NVF_ERROR(def->outputs().size() == 1);
    auto create_fn = def->newObjectFunc();
    create_fn(value->container(), inputs, {value}, def->attributes());
  }

  // hoist subexpression to outer loop. If `value` depends on a tensor, then we
  // should never insert it into `common_scalar_map_`, because we can not
  // allocate it at the beginning of the loop. If `value` is the given value to
  // the public `hoistScalar`, then we should always insert it into
  // `common_scalar_map_` so that future `value` could consider reusing it. If
  // `value` is a subexpression of the given value, then we insert it into
  // `common_scalar_map_` only if it can be hoisted to outer loops.
  if (!has_tensor_index_dependency && (is_given || my_pos < parent_pos)) {
    common_scalar_map_[my_loop].emplace_back(value);
    // We never hoist non-functional values because each call returns a
    // different value, therefore non-hoistable.
    if (my_pos < parent_pos && ir_utils::isFunctional(value)) {
      hoisted_or_reused_.emplace(value);
    }
  }
  seen_subexprs.emplace_back(value);
  return {value, has_tensor_index_dependency};
}

namespace {

std::list<VarInfo> getVariableInfo(
    Val* value,
    const std::vector<ForLoop*>& loops) {
  std::list<VarInfo> variables;
  // Loop indices
  for (auto loop : loops) {
    if (loop->isTrivial()) {
      if (loop->iter_domain()->isThread()) {
        variables.push_front({loop->index()});
      }
    } else {
      variables.push_back({loop->index(), loop->isUnrolled()});
    }
  }
  // Tensor metadata
  std::vector<Val*> to_visit{value};
  while (!to_visit.empty()) {
    auto back = to_visit.back();
    to_visit.pop_back();
    auto def = back->definition();
    if (def == nullptr) {
      continue;
    }
    if (def->isA<GetMetaData>()) {
      variables.push_front({back});
      continue;
    }
    to_visit.insert(to_visit.end(), def->inputs().begin(), def->inputs().end());
  }
  return variables;
}

std::vector<Val*> getAssumptions(const std::vector<ForLoop*>& loops) {
  std::vector<Val*> assumptions;
  // assumptions from parallel dimension
  for (auto [p, extent] :
       GpuLower::current()->parallelDimensionMap().getMap()) {
    auto a = IrBuilder::ltExpr(NamedScalar::getParallelIndex(p), extent);
    assumptions.emplace_back(a);
  }
  // assumptions from loop nesting
  for (auto loop : loops) {
    // Trivial loop is not generated, so there is no `if` or `for` in C++ to
    // guard its scope. So we should not assume index < stop. One real example
    // for this is loop rotation, where we might have trivial loop
    //   FOR [index:0, start:0, stop:size]:
    //     IF index < size:
    //       ... = T0[index]
    // The generated code will be
    //   if (0 < size) {
    //     ... = T0[0]
    //   }
    // We should not assume index smaller than size and simplify the code into
    //   if (true) {
    //     ... = T0[0]
    //   }
    // because this will break empty tensor support.
    if (loop->isTrivial()) {
      continue;
    }
    Val* start = loop->start();
    assumptions.push_back(IrBuilder::geExpr(loop->index(), start));
    Val* stop = loop->simplifiedStop();
    if (stop->sameAs(start)) {
      // If stop = start, then this loop will not be computed, so it's not
      // important to simplify its index. However, it is important that we avoid
      // contradicting assumptions, so we omit the index < stop assumption in
      // these cases.
      TORCH_WARN_ONCE(
          "Encountered loop with no iterations. Stop value ",
          stop->toInlineString(),
          " is same as start value ",
          start->toInlineString(),
          ". This could indicate a suboptimal schedule such as "
          "circular-buffering a ",
          "loop that has only a single iteration.");
    } else {
      assumptions.push_back(IrBuilder::ltExpr(loop->index(), stop));
    }
  }
  return assumptions;
}

} // namespace

Val* CommonScalarMap::hoistScalar(
    Val* value,
    const std::vector<ForLoop*>& loops) {
  value =
      simplifyExpr(value, getVariableInfo(value, loops), getAssumptions(loops));
  std::vector<Val*> seen_subexprs;
  return hoistScalarImpl(
             value,
             loops,
             seen_subexprs,
             getInnermostNonTrivialLoop(loops),
             true)
      .first;
}

Val* CommonScalarMap::reuseScalarIfAlreadyComputed(Val* value, ForLoop* loop) {
  // Find if value is computed on the host.
  if (auto host_val = reuseValsKnownToKernel(value)) {
    return host_val;
  }
  // Find if loop already contain `value`.
  auto it = common_scalar_map_.find(loop);
  if (it != common_scalar_map_.end()) {
    auto& scalars = it->second;
    for (auto it = scalars.begin(); it != scalars.end(); it++) {
      auto scalar = *it;
      auto common_subexpr = findRefAsSubexprOf(scalar, value, false);
      if (common_subexpr != nullptr) {
        if (common_subexpr != scalar) {
          // If the reuse is a subexpression instead of the complete
          // expression, we split this subexpression out and allocate it
          // separately.
          scalars.insert(it, common_subexpr);
        }
        hoisted_or_reused_.emplace(common_subexpr);
        return common_subexpr;
      }
    }
  }
  return nullptr;
}

std::vector<Val*> CommonScalarMap::getHoistedScalars(ForLoop* loop) const {
  // In codegen, parallel type group may not be generated as a for loop, so
  // don't allocate in this loop
  if (loop != nullptr && loop->isGroup()) {
    return {};
  }
  std::vector<Val*> result;
  auto it = common_scalar_map_.find(loop);
  if (it != common_scalar_map_.end()) {
    for (auto v : it->second) {
      if (hoisted_or_reused_.count(v) > 0) {
        result.emplace_back(v);
      }
    }
  }
  return result;
}

void CommonScalarMap::initialize(const std::vector<Expr*> exprs) {
  // We only hoist scalars not depending on tensors. In lowered expressions, all
  // these scalars are computed in top level scope.
  NVF_ERROR(
      common_scalar_map_.empty(),
      "CommonScalarMap used before initialization.");
  for (auto expr : exprs) {
    if (lower_utils::isScalarExpr(expr) && expr->outputs().size() == 1) {
      common_scalar_map_[nullptr].emplace_back(expr->output(0));
    } else if (ir_utils::isTvOp(expr)) {
      // We only try to reuse scalar expressions placed at the beginning of the
      // top level scope. For example, if I have
      // i1 = i0 + 1
      // i2 = T0.size[0] * T0.size[1]
      // i3 = address(T0)
      // i4 = i3 + 1
      // ....
      // Then we only consider `i1 = i0 + 1` and `i2 = T0.size[0] * T0.size[1]`
      // as valid reuse-opportunity. The barrier between valid and invalid is
      // the first tensor operation. The reason why we have this barrier is
      // because the `reorderExprsForComputeAt` always place scalar expressions
      // at the beginning if possible. If a scalar expression is placed after a
      // tensor expression, then this scalar operation must have dependency on
      // some tensor. For this case, we just giveup reusing because this helps
      // us to keep the analysis simple.
      break;
    }
  }
}

namespace {

// Check if the given `value` is already allocated (and computed in full) or
// computed partially. If yes, then return the position of the existing
// computation for the first return value, else return -1 for the first return
// value. The second return value is a boolean indicating if the given `value`
// is already fully computed.
std::pair<int64_t, bool> findAllocPointFromDataDependency(
    const std::vector<Expr*>& exprs, // ordered expressions in the scope where
                                     // the hoisted value should be inserted
    Val* value) {
  int64_t pos = -1;
  for (auto i : arange(exprs.size())) {
    auto expr = exprs[i];
    NVF_ERROR(expr != nullptr);
    if (auto alloc = dynamic_cast<kir::Allocate*>(expr)) {
      // Currently this branch is only to handle shared memory address. For
      // shared memory address, we generate code like `toSmem(T7)`, this does
      // not need the shared memory tensor to be computed, but it does require
      // T7 to be allocated.
      auto buffer = alloc->buffer();
      if (buffer == value ||
          findRefAsSubexprOf(value, buffer, true) != nullptr) {
        pos = (int64_t)i;
      }
    }
    for (auto o : expr->outputs()) {
      if (value == o) {
        return {i, true};
      }
      auto subexpr = findRefAsSubexprOf(value, o, true);
      if (subexpr != nullptr) {
        pos = (int64_t)i;
      }
    }
  }
  return {pos, false};
}

// Inserts allocations of hoisted indices
class CommonScalarInserter : private kir::ExprMutator {
 public:
  static std::vector<Expr*> run(
      const std::vector<Expr*>& exprs,
      const CommonScalarMap& common_indices) {
    CommonScalarInserter inserter(exprs, common_indices);
    return std::move(inserter.exprs_);
  }

 private:
  CommonScalarInserter(
      const std::vector<Expr*>& exprs,
      const CommonScalarMap& common_scalar_map)
      : common_scalar_map_(common_scalar_map) {
    IrVisitor::handle(exprs);
    maybeInsertAllocation(nullptr);
    mutate();
  }

  void maybeInsertAllocation(ForLoop* loop) {
    Scope* scope = nullptr;
    if (loop != nullptr) {
      scope = &loop->body();
    }
    const auto& exprs = (scope == nullptr ? exprs_ : scope->exprs());
    int64_t alloc_point = -1;
    Expr* insert_ref = nullptr;
    for (auto value : common_scalar_map_.getHoistedScalars(loop)) {
      auto existing_alloc_info = findAllocPointFromDataDependency(exprs, value);
      // If this value has already been fully computed, then don't insert
      // duplicate allocation and computation.
      if (existing_alloc_info.second) {
        continue;
      }
      // If a partial computation is found, then move the allocation point to
      // after the partial computation, so that the result of this partial
      // computation can be used.
      if (existing_alloc_info.first > alloc_point) {
        alloc_point = existing_alloc_info.first;
        insert_ref = exprs[alloc_point];
      }

      if (!isHelpfulToReuse(value)) {
        continue;
      }

      auto alloc = IrBuilder::create<kir::Allocate>(
          value, MemoryType::Local, GpuLower::current()->kernel()->oneVal());
      const auto def = value->definition();
      NVF_ERROR(
          def != nullptr,
          "Hoisted value must have a definition. ",
          value->toString());
      if (insert_ref == nullptr) {
        registerInsertBefore(exprs.at(0), alloc, scope);
      } else {
        registerInsertAfter(insert_ref, alloc, scope);
      }
      registerInsertAfter(alloc, def, scope);
      insert_ref = def;
    }
  }

  using kir::ExprMutator::handle;

  void handle(ForLoop* loop) final {
    maybeInsertAllocation(loop);
    kir::ExprMutator::handle(loop);
  }

 private:
  const CommonScalarMap& common_scalar_map_;
};

} // namespace

std::vector<Expr*> allocateCommonScalars(const std::vector<Expr*>& exprs) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    return exprs;
  }
  return CommonScalarInserter::run(
      exprs, GpuLower::current()->commonScalarMap());
}

} // namespace nvfuser
