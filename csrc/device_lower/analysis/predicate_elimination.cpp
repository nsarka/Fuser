// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/predicate_elimination.h>

#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <disjoint_set.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <options.h>
#include <predicate_compute.h>
#include <transform_iter.h>
#include <transform_replay.h>

namespace nvfuser {

namespace {

// Tensor memory is similar to shared memory because they are both
// shared between threads in a block. In that sense, we can consider
// tensor memory as special type of shared memory. In this file, we use
// the term "shared memory", "smem" to refer to both shared and tensor
// memories.
bool isSharedMemory(TensorView* tv) {
  return tv->getMemoryType() == MemoryType::Shared ||
      tv->getMemoryType() == MemoryType::Tensor;
}

// Warp primitives are currently limited to un-predicated usage,
//   predicating these ops will require extra steps to ensure that
//   the whole warp will get the same value.
void assertOnWarpOps(const Expr* expr) {
  // Prohibit predicates for LdMatrix expressions in Mma k main loop;
  // Allow predicates for general LdMatrix usage.
  if (ir_utils::isLdMatrixOp(expr)) {
    const LoadStoreOp* ldst = expr->as<LoadStoreOp>();
    TensorView* in_tv = ir_utils::getTv(ldst->in());
    NVF_ERROR(in_tv != nullptr);

    NVF_ERROR(in_tv->definition() != nullptr);

    // nD TMA load doesn't require predicate
    bool is_nd_tma_load =
        ir_utils::isCpAsyncBulkTensorTileLoad(in_tv->definition());

    TensorView* out_tv = ir_utils::getTv(ldst->out());
    NVF_ERROR(out_tv != nullptr);
    bool any_mma_uses =
        std::any_of(out_tv->uses().begin(), out_tv->uses().end(), [](Expr* e) {
          return e->isA<MmaOp>();
        });

    NVF_ERROR(
        !is_nd_tma_load || !any_mma_uses,
        "Predicate elimination: cannot eliminate pred for ldmatrix, use exact "
        "parallel dims. ",
        expr->toString());
  }

  NVF_ERROR(
      !expr->isA<MmaOp>(),
      "Mma op: cannot eliminate predicate for mma op, tiling not valid. ",
      expr->toString());
}

} // namespace

namespace {

// Check if consumer is in the compute warp of a warp specialized loop,
// and the id_in_consumer is the parallel type of the warp specialization.
bool isComputeWarp(TensorView* consumer, IterDomain* id_in_consumer) {
  // TODO: This function can not find all the expressions in the compute
  // warp. For example, if we have:
  //   if (async warp) {
  //     T1 = T0;
  //   } else {
  //     T2 = T1;
  //     T3 = T2;
  //   }
  // then we will return false for T3, which is a false negative. Having
  // a false negative is fine in the sense that we will still be
  // functionally correct, but we will not be able to remove the predicate
  // around T3, which is a missed optimization opportunity.
  // For now, because warp specialization is only used for matmul, for
  // which the circular buffer loop is a reduction loop, and mma is the
  // only expr in the compute warp, we are fine. In the future, we might
  // want to improve this function to find all the expressions in the
  // compute warp, which will require a more sophisticated analysis.
  auto def = consumer->definition();
  if (def == nullptr) {
    return false;
  }
  auto producer_tvs = ir_utils::filterByType<TensorView>(def->inputs());
  if (producer_tvs.empty()) {
    return false;
  }
  return std::all_of(
      producer_tvs.begin(), producer_tvs.end(), [&](TensorView* producer_tv) {
        if (!producer_tv->isCircularBuffered()) {
          return false;
        }
        const auto& type = producer_tv->circularBufferOptions().type;
        return std::holds_alternative<WarpSpecialized>(type) &&
            std::get<WarpSpecialized>(type).on ==
            id_in_consumer->getParallelType();
      });
}

// Utility to check if the scheduled domain of the given
//   TensorView represent an exact shared mem access, meaning
//   that all the thread parallel dimensions on the loop nodes
//   are exact so that the shared mem read/write would not
//   run out of bound because of thread over-subscription.
bool isExactParallelSharedMemAccess(TensorView* tv) {
  for (auto id : tv->getLoopDomain()) {
    if (id->isThreadDim()) {
      // Need to predicate to avoid out of bound access
      //  because of over-subscribed block size.
      if (!lower_utils::isExtentEqualToMaxParallelTypeExtent(
              id, isComputeWarp(tv, id))) {
        return false;
      }
    }
  }
  return true;
}

// Check for conditions where the predicate cannot be removed
//  when either producer or consumer is in shared memory.
bool needSharedMemPredicate(TensorView* producer, TensorView* consumer) {
  // Indexing is based on consumer loop ids so check the consumer.

  // If consumer schedule contains in-exact thread parallel
  //  dimensions, need to predicate against out of bound
  //  shared memory access by out of bound threads.
  if (!isExactParallelSharedMemAccess(consumer)) {
    return true;
  }

  // TODO: This is directed WAR on FusionPersistentNormLocalShared.
  //  This use case along with other previous issues motivate a
  //   joint optimization of predicate removal and buffer reuse.
  // In this particular case:
  //   __shared__ T0 [10], T1[10]
  //   for i in ...
  //      if(pred)
  //        T1[i] = T0[i] + ...  // exp0
  //      T2 = 0;              // init for exp1
  //      if(pred)
  //        T2 = T1 ...        // exp1
  //  If we remove pred around expr1, as the way the pred removal
  //    pass is set up, the init for expr will be pushed up to
  //    initialize T1 instead.
  //  However if we initialize T1, the code will look like:
  //  for i in ...
  //    T1[i] = 0;
  //  for i in ...
  //    if(pred)
  //      T1[i] = T0[i] + ...
  //  Note that we'd be able to reuse buffer of T0 for T1 but
  //    if we initialze T1 we cannot do that and thus the
  //    kernel would not fit in smaller devices.
  if (producer->getMemoryType() == MemoryType::Shared) {
    if (auto producer_def = producer->definition()) {
      if (std::any_of(
              producer_def->inputs().begin(),
              producer_def->inputs().end(),
              [](Val* val) {
                if (auto tv = ir_utils::getTv(val)) {
                  return tv->getMemoryType() == MemoryType::Shared;
                }
                return false;
              })) {
        // Disable shared memory producers that is a consumer
        //  of another shared memory tensor. The initialization would
        //  break potential opportunity to re-use shared mem buffer.
        return true;
      }
    }
  }

  for (auto id : consumer->getLoopDomain()) {
    // TODO: (Enable in a follow up)
    //  smem predicate removal with init would break unroll and unswitch,
    //  eg. as in issue 1133, so disabling this removal pattern for now.
    if (id->getParallelType() == ParallelType::Unroll ||
        id->getParallelType() == ParallelType::Unswitch) {
      return true;
    }
  }

  // TODO: (Enable in a follow up)
  //  This cannot yet be removed since smem initialization needs to be
  //  handled specially, e.g. as in smem_reduce test. Will be able to
  //  lift this one once the generic pred removal pass with fusion
  //  traversal is ready.
  auto consumer_def = consumer->definition();
  if (ir_utils::isReductionOp(consumer_def)) {
    if (producer->getMemoryType() == MemoryType::Shared) {
      return true;
    }
  }

  return false;
}

bool needsPredicateSharedMemAccess(const Expr* expr) {
  DEBUG_PRINT_SCOPE(expr);
  // This is initial step to gradually remove predicates around
  //  sharedmem access in suitable situations.
  // Using an additional variable to track the predicate-on reasons
  //  when the predicate around shared mem cannot be removed.
  for (auto consumer : ir_utils::filterByType<TensorView>(expr->outputs())) {
    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
      if (isSharedMemory(producer) || isSharedMemory(consumer)) {
        if (needSharedMemPredicate(producer, consumer)) {
          RECORD_AND_RETURN(true);
        }
      }
    }
  }
  RECORD_AND_RETURN(false);
}

class ProducerConsumerPairAnalyzer : public OptOutDispatch {
 public:
  //! Checks if a predicate is needed to avoid out-of-bound accesses.
  //!
  //! Due to the way we allocate local-memory tensors, there should
  //! never be out-of-bound accesses with consumer tensors when allocated on
  //! local memory. However, accessing producer tensors still may
  //! result in out-of-bound as they are replayed as consumers.
  static bool needsPredicate(TensorView* producer, TensorView* consumer) {
    // TMA ops handles out of bound accesses automatically in hardware, there is
    // no need for us to predicate it.
    if (ir_utils::isCpAsyncBulkTensorTile(consumer->definition())) {
      return false;
    }
    // Both tensors must be on local or shared memory. Global tensors must be
    // predicated as allocation is done based on root domains. Smem
    // and local tensors are allocated based on loop domains.
    // However, smem tensors are parallelized, which is highly likely, the size
    // of the parallelized axis is the actual size of the axis, not
    // the number of threads. This is currently actively checked to avoid
    // out of bound shared mem access by out of bound threads.
    if (producer->getMemoryType() == MemoryType::Global ||
        consumer->getMemoryType() == MemoryType::Global) {
      return true;
    }

    auto pairwise_map = PairwiseLogicalDomainMap(producer, consumer);
    auto c2p =
        BestEffortReplay::replayPasC(
            producer, consumer, /*consumer_compute_at_axis=*/-1, pairwise_map)
            .getReplay();

    ProducerConsumerPairAnalyzer analyzer(consumer, c2p);

    for (auto id : consumer->getLoopDomain()) {
      if (analyzer.needsPredicate(id)) {
        return true;
      }
    }

    return false;
  }

 private:
  ProducerConsumerPairAnalyzer(
      TensorView* consumer,
      const std::unordered_map<IterDomain*, IterDomain*>& c2p)
      : consumer_(consumer), c2p_(c2p) {}

  // Returns true if no out-of-bound accesses could occur with a
  // producer
  bool needsPredicate(IterDomain* consumer_id) {
    needs_predicate_ = false;
    handle(consumer_id);
    return needs_predicate_;
  }

  void handle(IterDomain* consumer_id) override {
    // The traversal should have ended if needs_predicate_ was true
    NVF_ERROR(!needs_predicate_);

    // If consumer_id is not going to be materialized as a loop (e.g.,
    // broadcast), no need to predicate
    if (consumer_id->isBroadcast()) {
      return;
    }

    // If the ID is parallelized with a non-unique parallel type, the
    // consumer ID may be oversubscribed, which may cause
    // out-of-bounds accesses in the producer
    const auto maybe_oversubscribed = consumer_id->isThread() &&
        (!lower_utils::isExtentEqualToMaxParallelTypeExtent(
            consumer_id, isComputeWarp(consumer_, consumer_id)));
    if (maybe_oversubscribed) {
      // If oversubscribed, there must be a mapped producer ID that is
      // parallelized in the same way. Otherwise, needs to be
      // predicated.
      auto c2p_it = c2p_.find(consumer_id);
      if (c2p_it == c2p_.end() ||
          c2p_it->second->getParallelType() != consumer_id->getParallelType()) {
        needs_predicate_ = true;
        return;
      }
    }

    // If the producer has a matching domain, it should not cause
    // out-of-bound accesses
    if (c2p_.count(consumer_id)) {
      return;
    }

    // If no definition exists, stop traversing
    if (consumer_id->definition() == nullptr) {
      return;
    }

    OptOutDispatch::dispatch(consumer_id->definition());
  }

  // If it splits the input axis evenly, proceeds to check the input
  // axis. Otherwise, we can't skip predication as it might cause
  // out-bound accesses with the producer tensor
  void handle(Split* split) override {
    auto factor = split->factor()->value();
    if (!factor.is<int64_t>()) {
      needs_predicate_ = true;
      return;
    }

    if (factor == 1) {
      // Trivial splits cannot cause out-of-bounds
      return;
    }

    auto in_extent = split->in()->extent();

    if (!in_extent->isConstInt() ||
        ((in_extent->evaluate().as<int64_t>() % factor.as<int64_t>()) != 0)) {
      needs_predicate_ = true;
      return;
    }

    handle(split->in());
  }

  void handle(Merge* merge) override {
    handle(merge->inner());
    if (needs_predicate_) {
      return;
    }
    handle(merge->outer());
  }

  void handle(Resize* resize) override {
    // resize outputs are guaranteed to match by the check above in
    // handle(IterDomain*).
    handle(resize->in());
  }

 private:
  TensorView* consumer_ = nullptr;
  //! BestEffort map from consumer IDs to producer IDs
  const std::unordered_map<IterDomain*, IterDomain*>& c2p_;
  bool needs_predicate_ = false;
};

class PredicateChcker : public IterVisitor {
 public:
  static bool needsPredicate(
      Expr* expr,
      const PredicateElimination& pred_elimination) {
    if (!ir_utils::isTvOp(expr)) {
      return false;
    }

    if (expr->fusion()->hasManaged("don't predicate")) {
      const auto& dont_predicate =
          expr->fusion()->getManaged<std::unordered_set<Expr*>>(
              "don't predicate");
      if (dont_predicate.count(expr) > 0) {
        return false;
      }
    }

    PredicateChcker checker(pred_elimination);
    checker.dispatch(expr);
    return checker.needs_predicate_;
  }

 private:
  PredicateChcker(const PredicateElimination& pred_elimination)
      : pred_elimination_(pred_elimination),
        non_predicated_exprs_(pred_elimination.getNonPredicatedExprs()) {}

  using IterVisitor::handle;

  void dispatch(Expr* expr) final {
    const bool needs_predicate_smem_access =
        needsPredicateSharedMemAccess(expr);
    needs_predicate_ = predicateIntDiv(expr) || needs_predicate_smem_access ||
        predicateProducerConsumerPair(expr) ||
        predicateNonDivisibleLogicalDomains(expr) ||
        predicateNonDivisibleSplit(expr) || predicateExpandReduce(expr) ||
        predicateRNGOp(expr);

    if (needs_predicate_) {
      return;
    }

    // Check expr type-specific conditions
    IterVisitor::dispatch(expr);
  }

  // All "predicateXYZ" functions return true if an expr needs to be
  // predicated.

  // Always predicate rng ops as they are expensive.
  bool predicateRNGOp(Expr* expr) const {
    DEBUG_PRINT_SCOPE(expr);
    RECORD_AND_RETURN(expr->isA<RNGOp>());
  }

  // Always predicate integer division and related ops as we don't
  // know what values are in the out-of-bound region and they may
  // cause exceptions
  bool predicateIntDiv(Expr* expr) const {
    DEBUG_PRINT_SCOPE(expr);
    auto dt = expr->outputs()[0]->getDataType().value();
    RECORD_AND_RETURN(
        (dt == DataType::Int || dt == DataType::Int32) &&
        expr->isA<BinaryOp>() &&
        (expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::Div ||
         expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::Mod ||
         expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::Remainder ||
         expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::CeilDiv));
  }

  // If we're reducing an expanded domain, we need to be careful to predicate it
  // or we could end up reducing a broadcasted value too many times.
  bool predicateExpandReduce(Expr* expr) const {
    DEBUG_PRINT_SCOPE(expr);
    if (!ir_utils::isReductionOp(expr)) {
      RECORD_AND_RETURN(false);
    }
    auto tv_inputs = ir_utils::getTvs(expr->inputs());
    NVF_ERROR(
        !tv_inputs.empty(),
        "Should never have a reduction op without a tensor view input.");
    bool found_expand = false;
    for (auto tv_input : tv_inputs) {
      found_expand = found_expand ||
          std::any_of(tv_input->getLogicalDomain().begin(),
                      tv_input->getLogicalDomain().end(),
                      [](IterDomain* id) { return id->hasExpandedExtent(); });
    }

    if (!found_expand) {
      RECORD_AND_RETURN(false);
    }

    auto tv_outputs = ir_utils::getTvs(expr->outputs());
    if (expr->isA<WelfordOp>() && tv_inputs.size() != tv_outputs.size()) {
      tv_outputs = std::vector<TensorView*>(tv_inputs.size(), tv_outputs[0]);
    }

    NVF_ERROR(
        tv_outputs.size() == tv_inputs.size(),
        "Was expecting matching number of inputs and outputs for expression: ",
        expr->toString());

    for (auto i : arange(tv_inputs.size())) {
      const auto root_p2c =
          PairwiseLogicalDomainMap(tv_inputs[i], tv_outputs[i])
              .mapProducerToConsumer();
      for (auto entry : root_p2c) {
        auto p_id = entry.first;
        auto c_id = entry.second;
        if (p_id->hasExpandedExtent() && c_id->isReduction()) {
          RECORD_AND_RETURN(true);
        }
      }
    }
    RECORD_AND_RETURN(false);
  }

  // Predicates the expression if any producer-consumer pair of the
  // expression needs to be predicated
  bool predicateProducerConsumerPair(Expr* expr) const {
    DEBUG_PRINT_SCOPE(expr);
    for (auto output : ir_utils::filterByType<TensorView>(expr->outputs())) {
      for (auto input : ir_utils::filterByType<TensorView>(expr->inputs())) {
        if (ProducerConsumerPairAnalyzer::needsPredicate(input, output)) {
          RECORD_AND_RETURN(true);
        }
      }
    }
    RECORD_AND_RETURN(false);
  }

  // Utility to find the loop iterdomains of the given
  //   tensor view that will be treated as "zero loops"
  //   in the indexing pass.
  // For details on zero loops, see indexMapFromTV in
  //  lower index pass.
  std::vector<Val*> getZeroLoopIds(const TensorView* tv) const {
    std::vector<Val*> zero_loop_ids;
    for (const auto i : arange(tv->nDims())) {
      auto loop_id = tv->axis(i);
      if (ir_utils::isMemorySharedAcross(
              tv->getMemoryType(), loop_id->getParallelType())) {
        // Thread parallel axes on shared mem are never
        //  zero loops as each thread owns its share
        //  of the shared mem space.
        continue;
      }
      if (
          // Non-thread parallel dimension on the left
          //  of CA axes are zero loops.
          i < tv->getComputeAtPosition() ||
          // Parallel axes on local mem is zero loop.
          // Grid axes on shared mem is zero loop.
          ir_utils::isMemoryPartitionedAcross(
              tv->getMemoryType(), loop_id->getParallelType()) ||
          // Mma axes, similar to vectorization, are
          //  implicit in hardware intrinsics, and thus
          //  will be treated as a zero loop.
          loop_id->isMma()) {
        zero_loop_ids.push_back(loop_id);
      }
    }

    return zero_loop_ids;
  }

  // An index can exceed the logical extent of the indexed domain if
  // it's split. It can cause a reduction op to reduce the same value
  // multiple times. Even a pointwise op can be a problem if the
  // consumer is an alias of the producer. This check excludes such
  // expressions from predicate elimination.
  //
  // This is not an issue if the index includes a zero domain (as defined in
  // index_compute.cpp), the extent is calculated by multiplying the
  // split output domains, so it never cross the domain boundary.
  // So, if a logical domain is split and none of its descendants is a
  // zero domain, the expr needs to be predicated. See
  // FusionPredicateElimination6 for a concrete example.
  //
  // It would be also possible to avoid register aliasing instead of
  // giving up predicate elimination. Since this condition should be
  // rather uncommon, either would be fine as long as correctness is
  // provided.
  bool predicateNonDivisibleLogicalDomains(Expr* expr) const {
    DEBUG_PRINT_SCOPE(expr);
    // TMA ops handles out of bound accesses automatically in hardware, there is
    // no need for us to predicate it.
    if (ir_utils::isCpAsyncBulkTensorTile(expr)) {
      RECORD_AND_RETURN(false);
    }
    for (auto output : ir_utils::filterByType<TensorView>(expr->outputs())) {
      const auto all_exprs = DependencyCheck::getAllExprsBetween(
          {output->getLogicalDomain().begin(),
           output->getLogicalDomain().end()},
          {output->getLoopDomain().begin(), output->getLoopDomain().end()});
      std::unordered_set<Val*> split_logical;
      std::copy_if(
          output->getLogicalDomain().begin(),
          output->getLogicalDomain().end(),
          std::inserter(split_logical, split_logical.end()),
          [&](auto rf_logical) {
            if (rf_logical->isBroadcast()) {
              return false;
            }
            for (Expr* use : rf_logical->uses()) {
              if (std::find(all_exprs.begin(), all_exprs.end(), use) ==
                  all_exprs.end()) {
                continue;
              }
              return use->isA<Split>();
            }
            return false;
          });
      // If no logical domain is split, no need to predicate
      if (split_logical.empty()) {
        continue;
      }
      const auto zero_loop_ids = getZeroLoopIds(output);
      if (zero_loop_ids.empty()) {
        RECORD_AND_RETURN(true);
      }
      const auto vals =
          DependencyCheck::getAllValsBetween(split_logical, zero_loop_ids);
      if (std::any_of(
              split_logical.begin(),
              split_logical.end(),
              [&vals](auto split_logical_id) {
                return std::find(vals.begin(), vals.end(), split_logical_id) ==
                    vals.end();
              })) {
        RECORD_AND_RETURN(true);
      }
    }
    RECORD_AND_RETURN(false);
  }

  // Always predicate if non-divisible split is found. It may be
  // possible to make it less conservative.
  // See FusionPredicateElimination7 for a concrete example.
  bool predicateNonDivisibleSplit(Expr* expr) const {
    DEBUG_PRINT_SCOPE(expr);
    // TMA ops handles out of bound accesses automatically in hardware, there is
    // no need for us to predicate it.
    if (ir_utils::isCpAsyncBulkTensorTile(expr)) {
      RECORD_AND_RETURN(false);
    }

    for (auto output_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
      if ((GpuLower::current()->isTensorIndexerEnabled() &&
           GpuLower::current()->nonDivisiblePredicateInfo().hasPredicate(
               output_tv)) ||
          (!GpuLower::current()->isTensorIndexerEnabled() &&
           GpuLower::current()->nonDivisibleSplitInfo().hasPredicate(
               output_tv))) {
        RECORD_AND_RETURN(true);
      }
    }
    RECORD_AND_RETURN(false);
  }

  // If this is a reduction, and if we omit the predicate for the
  // input, the input may have a garbabe value, which must not be used
  // for this reduction. However, it is still legal to omit its
  // predicate when: 1) the predicate of the input is not omitted and
  // 2) the input can be initialized to the init value of this
  // reduction. When the input is the output of another reduciton, the
  // input is initialized to the init value of the reduction, so the
  // two reductions must use the same init value.
  // See FusionPredicateElimination3 and FusionPredicateElimination4
  // for concrete examples.
  void handle(ReductionOp* rop) final {
    auto input = rop->inputs()[0]->as<TensorView>();
    auto input_def = input->definition();
    // When input_def is null, input must be an input to the fusion,
    // so that must be allocated on global memory. Since we don't omit
    // predication for expressions involving global memory, this
    // should never occur.
    NVF_ERROR(
        input_def != nullptr, "Inconsistent input found: ", input->toString());

    // The input needs to be initialized to the init value to omit
    // the predicate, so if the input has its own init value, i.e.,
    // produced by another reduction, they must use the same init
    // value.
    Val* input_init = ir_utils::getReductionInitValOf(input);
    if (input_init != nullptr && !rop->init()->sameAs(input_init)) {
      needs_predicate_ = true;
      return;
    }

    // If input is not predicated, out-of-bound value may be
    // overwritten by a garbage value. However, it doesn't matter if
    // the input is also produced by another reduction. If the preceding
    // reduction omits the predicate, it means its input must be
    // initialized to its init value, so no predicate should be
    // needed in both of the two reduction ops if they use the same
    // init value, which is guaranteed by the above check, and the
    // same reduction op.
    if (auto input_def_rop = dynamic_cast<ReductionOp*>(input_def)) {
      if (rop->getReductionOpType() != input_def_rop->getReductionOpType() &&
          non_predicated_exprs_.find(input_def) !=
              non_predicated_exprs_.end()) {
        needs_predicate_ = true;
        return;
      }
    } else if (
        non_predicated_exprs_.find(input_def) != non_predicated_exprs_.end()) {
      needs_predicate_ = true;
      return;
    }
  }

  // Welford. See FusionPredicateElimination5.
  void handle(WelfordOp* wop) final {
    for (const auto i : arange(3)) {
      auto init = wop->getInitVals()[i];

      // Welford input can be a scalar. Predicate is required unless
      // the scalar value is equal to the init value.
      auto input = wop->inputs().at(i);
      if (input->isScalar()) {
        if (!input->sameAs(init)) {
          needs_predicate_ = true;
          return;
        }
        continue;
      }

      auto input_tv = dynamic_cast<TensorView*>(input);
      NVF_ERROR(input_tv != nullptr);

      auto input_def = input->definition();

      // When input_def is null, input must be an input to the fusion,
      // so that must be allocated on global memory. Since we don't omit
      // predication for expressions involving global memory, this
      // should never occur.
      NVF_ERROR(
          input_def != nullptr,
          "Inconsistent input found: ",
          input->toString());

      // The input needs to be initialized to the init value to omit
      // the predicate, so if the input has its own init value, i.e.,
      // produced by another reduction, they must use the same init
      // value.
      Val* input_init = ir_utils::getReductionInitValOf(input_tv);
      if (input_init != nullptr && !init->sameAs(input_init)) {
        needs_predicate_ = true;
        return;
      }

      // If input is not predicated, out-of-bound value may be
      // overwritten by a garbage value. However, it doesn't matter if
      // the input is also produced by another welford.
      if (!input_def->isA<WelfordOp>() && !input_def->isA<GroupedWelfordOp>() &&
          non_predicated_exprs_.find(input_def) !=
              non_predicated_exprs_.end()) {
        needs_predicate_ = true;
        return;
      }
    }
  }

  void handle(GroupedReductionOp* grouped_rop) final {
    for (const auto i : arange(grouped_rop->numHorizontallyGroupedExprs())) {
      auto input = grouped_rop->input(i)->as<TensorView>();
      auto input_def = input->definition();
      // When input_def is null, input must be an input to the fusion,
      // so that must be allocated on global memory. Since we don't omit
      // predication for expressions involving global memory, this
      // should never occur.
      NVF_ERROR(
          input_def != nullptr,
          "Inconsistent input found: ",
          input->toString());

      // The input needs to be initialized to the init value to omit
      // the predicate, so if the input has its own init value, i.e.,
      // produced by another reduction, they must use the same init
      // value.
      Val* input_init = ir_utils::getReductionInitValOf(input);
      if (input_init != nullptr &&
          !grouped_rop->initVal(i)->sameAs(input_init)) {
        needs_predicate_ = true;
        return;
      }

      // If input is not predicated, out-of-bound value may be
      // overwritten by a garbage value. However, it doesn't matter if
      // the input is also produced by another reduction. If the preceding
      // reduction omits the predicate, it means its input must be
      // initialized to its init value, so no predicate should be
      // needed in both of the two reduction ops if they use the same
      // init value, which is guaranteed by the above check, and the
      // same reduction op.
      if (auto input_def_rop = dynamic_cast<ReductionOp*>(input_def)) {
        if (grouped_rop->getReductionOpType(i) !=
                input_def_rop->getReductionOpType() &&
            non_predicated_exprs_.find(input_def) !=
                non_predicated_exprs_.end()) {
          needs_predicate_ = true;
          return;
        }
      } else if (
          auto input_def_grouped_rop =
              dynamic_cast<GroupedReductionOp*>(input_def)) {
        auto input_index_as_output =
            input_def_grouped_rop->getExprIndexOfOutput(input);
        if (grouped_rop->getReductionOpType(i) !=
                input_def_grouped_rop->getReductionOpType(
                    input_index_as_output) &&
            non_predicated_exprs_.find(input_def) !=
                non_predicated_exprs_.end()) {
          needs_predicate_ = true;
          return;
        }
      } else if (
          non_predicated_exprs_.find(input_def) !=
          non_predicated_exprs_.end()) {
        needs_predicate_ = true;
        return;
      }
    }
  }

  void handle(GroupedWelfordOp* grouped_wop) final {
    for (const auto expr_idx :
         arange(grouped_wop->numHorizontallyGroupedExprs())) {
      for (const auto val_idx : arange(3)) {
        auto init = grouped_wop->initVals().at(expr_idx).get(val_idx);

        // Welford input can be a scalar. Predicate is required unless
        // the scalar value is equal to the init value.
        auto input = grouped_wop->inputVals().at(expr_idx).get(val_idx);
        if (input->isScalar()) {
          if (!input->sameAs(init)) {
            needs_predicate_ = true;
            return;
          }
          continue;
        }

        auto input_tv = dynamic_cast<TensorView*>(input);
        NVF_ERROR(input_tv != nullptr);

        auto input_def = input->definition();

        // When input_def is null, input must be an input to the fusion,
        // so that must be allocated on global memory. Since we don't omit
        // predication for expressions involving global memory, this
        // should never occur.
        NVF_ERROR(
            input_def != nullptr,
            "Inconsistent input found: ",
            input->toString());

        // The input needs to be initialized to the init value to omit
        // the predicate, so if the input has its own init value, i.e.,
        // produced by another reduction, they must use the same init
        // value.
        Val* input_init = ir_utils::getReductionInitValOf(input_tv);
        if (input_init != nullptr && !init->sameAs(input_init)) {
          needs_predicate_ = true;
          return;
        }

        // If input is not predicated, out-of-bound value may be
        // overwritten by a garbage value. However, it doesn't matter if
        // the input is also produced by another reduction op as it
        // must be initialized and its initialized value is already
        // found to be equal to the initil value of this op.
        if (!input_def->isA<WelfordOp>() &&
            !input_def->isA<GroupedWelfordOp>() &&
            non_predicated_exprs_.find(input_def) !=
                non_predicated_exprs_.end()) {
          needs_predicate_ = true;
          return;
        }
      }
    }
  }

  // Similar to the above reduction constraint but for MMA
  void handle(MmaOp* mma) final {
    for (auto input : ir_utils::filterByType<TensorView>(mma->inputs())) {
      auto input_def = input->definition();
      NVF_ERROR(
          input_def != nullptr,
          "Inconsistent input found: ",
          input->toString());

      Val* input_init = ir_utils::getReductionInitValOf(input);
      if (input_init != nullptr && !mma->init()->sameAs(input_init)) {
        needs_predicate_ = true;
        return;
      }

      if (non_predicated_exprs_.find(input_def) !=
          non_predicated_exprs_.end()) {
        // If producer of mma is non_predicated and initialized
        //  with the same value. The mma should not need a
        //  predicate. In fact this is the only way we can
        //  use mma at the moment since we could not predicate
        //  mma ops without guaranteeing warp uniform results.
        auto input_init = pred_elimination_.getInitValue(input);

        // TODO:
        //   clean up this to support more generic prolog fusion.
        //   Will need additional analysis passes on initialization
        //    propagation and further predicate placement on top.
        // More TODO:
        //  Even when producer is initialized, it is still generally
        //   not safe to remove predicate around reduction ops if the
        //   producer is not predicated.
        //  On the other side, we do have patterns like ldmatrix->mma where
        //   both producer and consumer cannot be safely predicated without
        //   guaranteeing warp uniform results.
        //  This is currently a WAR and relies on validation pass to exclude
        //   complex prolog patterns in mma based matmul kernels. Will
        //   definitely need to revisit and build out predicate and
        //   initialization analysis pass to better handle this case.
        if (input_init != nullptr && !input_init->sameAs(mma->init())) {
          // This is a WAR at the moment. We would need to propagate
          //  initialization information from PredicateElimination
          //  pass to most accurately detect if the input is
          //  initialized correctly.
          // This could also be fixed when we have the traversal
          //  based predicate elimination and initialization pass
          //  ready. Would be easy to clean up this part at that point.
          needs_predicate_ = true;
          return;
        }
      }
    }
  }

 private:
  const PredicateElimination& pred_elimination_;
  const std::unordered_set<const Expr*>& non_predicated_exprs_;
  bool needs_predicate_ = false;
};

} // namespace

PredicateElimination::PredicateElimination(Fusion* fusion) {
  traverseTo(fusion->outputs());
}

bool PredicateElimination::needsPredicate(Expr* expr) const {
  return PredicateChcker::needsPredicate(expr, *this);
}

void PredicateElimination::dispatch(Expr* expr) {
  if (!ir_utils::isTvOp(expr)) {
    return;
  }

  if (needsPredicate(expr)) {
    assertOnWarpOps(expr);
    return;
  }

  non_predicated_exprs_.insert(expr);

  // Ensure all inputs have some values set at the out-of-bound
  // regions
  for (const auto i : arange(expr->inputs().size())) {
    auto input = dynamic_cast<TensorView*>(expr->inputs()[i]);
    if (input == nullptr) {
      continue;
    }
    auto input_def = input->definition();
    // When input_def is null, input must be an input to the fusion,
    // so that must be allocated on global memory. Since we don't omit
    // predication for expressions involving global memory except when we are
    // accessing the global memory with TMA, the following condition should
    // only occur if expr is a TMA load. For TMA loads, initialization is
    // handled in the TMA load itself, so we don't need to set a init value
    // here.
    if (input_def == nullptr) {
      continue;
    }

    // If input is an output of reduction, it should be fully
    // initialied as it's allocated on local memory.
    if (ir_utils::isReductionOp(input_def)) {
      continue;
    }

    if (expr->isA<ReductionOp>()) {
      setReductionInitValue(input, expr->as<ReductionOp>()->init());
      continue;
    } else if (expr->isA<GroupedReductionOp>()) {
      setReductionInitValue(input, expr->as<GroupedReductionOp>()->initVal(i));
      continue;
    } else if (auto wop = dynamic_cast<WelfordOp*>(expr)) {
      Val* init = wop->getInitVals().at(i);
      setReductionInitValue(input, init);
      continue;
    } else if (expr->isA<MmaOp>()) {
      setReductionInitValue(input, expr->as<MmaOp>()->init());
      continue;
    } else if (
        non_predicated_exprs_.find(input_def) != non_predicated_exprs_.end()) {
      // If an input does not need a predicate either, then it should
      // have some value, so no need to set a default value
      continue;
    } else {
      // Make sure input is initialized
      setDefaultInitValue(input);
    }
  }
}

bool PredicateElimination::setDefaultInitValue(TensorView* tv) {
  auto it = init_value_map_.find(tv);
  // If there's already a mapping for tv, it should be mapped to a
  // zero val or a reduction init. Either case, no need to modify
  // the existing mapping.
  if (it == init_value_map_.end()) {
    init_value_map_.insert({tv, nullptr});
  }
  return true;
}

bool PredicateElimination::setReductionInitValue(
    TensorView* tv,
    Val* reduction_init) {
  NVF_ERROR(tv != nullptr);

  auto it = init_value_map_.find(tv);
  if (it == init_value_map_.end()) {
    init_value_map_.insert({tv, reduction_init});
    return true;
  }

  auto existing_val = it->second;
  if (existing_val == nullptr) {
    // If the existing mapping returns nullptr, it means that a
    // default init was set before. Overwrite with the reduction
    // init val.
    init_value_map_[tv] = reduction_init;
    return true;
  } else if (existing_val->sameAs(reduction_init)) {
    return true;
  } else {
    NVF_THROW(
        "Inconsistent setting of initialization value for t",
        tv->name(),
        ". Prev: ",
        existing_val->toString(),
        ", New: ",
        reduction_init->toString());
    return false;
  }
}

bool PredicateElimination::canOmitPredicate(const Expr* expr) const {
  // Predicate elimination can be disabled with
  // NVFUSER_DISABLE=predicate_elimination
  if (isOptionDisabled(DisableOption::PredicateElimination)) {
    assertOnWarpOps(expr);
    return false;
  }

  NVF_ERROR(expr != nullptr);
  const auto out_tv = ir_utils::getTvOutput(expr);
  NVF_ERROR(out_tv != nullptr, "Not a tensor expression");

  if (ir_utils::isTensorScalarFillOp(expr)) {
    if (out_tv->getMemoryType() == MemoryType::Local) {
      // Filling a local tensor with scalar shouldn't
      //   need any predicate currently.
      return true;
    } else if (out_tv->getMemoryType() == MemoryType::Shared) {
      // A shared memory initialization should be same except
      //  that we'd need a predicate to guard against out of
      //  bound access by out of inexact threads.
      return isExactParallelSharedMemAccess(out_tv);
    }
  }

  if (non_predicated_exprs_.find(expr) != non_predicated_exprs_.end()) {
    return true;
  }

  assertOnWarpOps(expr);
  return false;
}

bool PredicateElimination::needsSharedMemoryPredicate(const Expr* expr) const {
  NVF_ERROR(expr != nullptr);
  const auto out_tv = ir_utils::getTvOutput(expr);
  NVF_ERROR(out_tv != nullptr, "Not a tensor expression");
  return needsPredicateSharedMemAccess(expr);
}

void PredicateElimination::propagateRemovalInfo(
    const Expr* from,
    const Expr* to) {
  if (non_predicated_exprs_.count(from)) {
    non_predicated_exprs_.insert(to);
  }
}

Val* PredicateElimination::getInitValue(TensorView* tv) const {
  auto it = init_value_map_.find(tv);
  if (it == init_value_map_.end()) {
    return nullptr;
  }
  auto init_val = it->second;
  if (init_val == nullptr) {
    // No reduction restriction. Just use zero
    auto dtype = *tv->getDataType();
    if (std::holds_alternative<ArrayType>(dtype.type)) {
      return IrBuilder::create<NamedScalar>("{}", dtype);
    }
    return GpuLower::current()->kernel()->zeroVal();
  } else {
    return init_val;
  }
}

std::string PredicateElimination::toString() const {
  std::stringstream ss;
  VectorOfUniqueEntries<TensorView*> non_predicated_tvs;
  for (auto expr : non_predicated_exprs_) {
    for (auto out : expr->outputs()) {
      if (auto ti = dynamic_cast<kir::TensorIndex*>(out)) {
        non_predicated_tvs.pushBack(ti->view());
      } else if (auto tv = dynamic_cast<TensorView*>(out)) {
        non_predicated_tvs.pushBack(tv);
      } else {
        NVF_THROW("Unexpected output ", out, " in ", expr);
      }
    }
  }
  ss << "Tensors that do not need predication:";
  for (auto tv : non_predicated_tvs) {
    ss << " T" << tv->name();
  }
  ss << "\n";
  ss << "Init values:";
  for (auto kv : init_value_map_) {
    ss << " T" << kv.first->name() << "->";
    if (kv.second == nullptr) {
      ss << "<default(0)>";
    } else {
      ss << kv.second;
    }
  }
  ss << "\n";
  ss << "Non-predicated expressions:";
  for (auto expr : non_predicated_exprs_) {
    ss << " " << expr;
  }
  return ss.str();
}

} // namespace nvfuser
