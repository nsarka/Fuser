// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>

#include <ATen/cuda/CUDAContext.h>
#include <debug.h>
#include <device_lower/analysis/device_version.h>
#include <device_lower/analysis/divisible_split.h>
#include <device_lower/analysis/tensor_producer_aliases.h>
#include <device_lower/pass/alias_memory.h>
#include <device_lower/pass/allocation.h>
#include <device_lower/pass/circular_buffer.h>
#include <device_lower/pass/expr_sort.h>
#include <device_lower/pass/fusion_simplifier.h>
#include <device_lower/pass/grid_serialization.h>
#include <device_lower/pass/index.h>
#include <device_lower/pass/inline_ptx.h>
#include <device_lower/pass/insert_syncs.h>
#include <device_lower/pass/instrument.h>
#include <device_lower/pass/loop_rotation.h>
#include <device_lower/pass/loops.h>
#include <device_lower/pass/magic_zero.h>
#include <device_lower/pass/predicate.h>
#include <device_lower/pass/replace_size.h>
#include <device_lower/pass/rng.h>
#include <device_lower/pass/unroll.h>
#include <device_lower/pass/vectorize_welford.h>
#include <device_lower/pass/warp_reduce.h>
#include <device_lower/utils.h>
#include <device_lower/validation.h>
#include <expr_simplifier.h>
#include <fusion.h>
#include <id_model/id_model.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>

#include <list>
#include <unordered_map>
#include <unordered_set>

namespace nvfuser {

thread_local GpuLower* active_gpu_lower = nullptr; // NOLINT
namespace {

class KIRCleaner : public OptOutDispatch {
 public:
  //! Remove nop IR nodes
  static std::vector<Expr*> cleanUp(const std::vector<Expr*>& loop_nests) {
    KIRCleaner cleaner;
    std::vector<Expr*> out_loop_nests;
    for (auto loop_nest : loop_nests) {
      cleaner.dispatch(loop_nest);
      // No need to keep the loop nest if it's determined to be nop
      if (!cleaner.is_nop_) {
        out_loop_nests.push_back(loop_nest);
      }
    }
    return out_loop_nests;
  }

 private:
  using OptOutDispatch::handle;
  void dispatch(Expr* expr) final {
    if (expr->isA<ForLoop>() || expr->isA<kir::IfThenElse>()) {
      OptOutDispatch::dispatch(expr);
    } else {
      // Any non-scoping expr is not considered nop
      is_nop_ = false;
    }
  }

  void handle(ForLoop* fl) final {
    auto exprs = fl->body().exprs();
    fl->body().clear();
    for (auto expr : exprs) {
      dispatch(expr);
      // Add the expr to the loop body only when the expr is not nop
      if (!is_nop_) {
        fl->body().push_back(expr);
      }
    }
    // The loop is nop when no expr exists in the body
    is_nop_ = fl->body().empty();
  }

  void handle(kir::IfThenElse* ite) final {
    const auto conditional = ite->predicate()->value();

    // Visit the then block
    auto then_exprs = ite->thenBody().exprs();
    ite->thenBody().clear();
    if (!conditional->isConst() || conditional->value().as<bool>()) {
      for (auto expr : then_exprs) {
        dispatch(expr);
        if (!is_nop_) {
          ite->thenBody().push_back(expr);
        }
      }
    }

    const bool then_nop = ite->thenBody().empty();

    // Visit the else block
    auto else_exprs = ite->elseBody().exprs();
    ite->elseBody().clear();
    if (!conditional->isConst() || !conditional->value().as<bool>()) {
      for (auto expr : else_exprs) {
        dispatch(expr);
        if (!is_nop_) {
          ite->elseBody().push_back(expr);
        }
      }
    }

    const bool else_nop = ite->elseBody().empty();

    // If the then block is nop but the else is not, invert the
    // conditional and move the exprs in the else block to the then
    // block.
    if (then_nop && !else_nop) {
      Val* pred = ite->predicate()->value();
      Val* not_pred = SimplifyingIrBuilder::logicalNotExpr(pred);
      ite->predicate()->setValue(not_pred);
      for (auto expr : ite->elseBody().exprs()) {
        ite->thenBody().push_back(expr);
      }
      ite->elseBody().clear();
    }

    // This IfThenElse is nop if both the then and else blocks are nop
    is_nop_ = then_nop && else_nop;
  }

 private:
  //! True if the last visited expr is nop
  bool is_nop_ = false;
};

} // namespace

void GpuLower::collectPaddedParallelDims() {
  bool can_be_single_warp = true;

  auto warp_size = at::cuda::warp_size();

  auto used_vals = fusion_->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (auto id : tv->getLoopDomain()) {
      if (tv->definition()) {
        // TODO: Support GroupedReductionOp
        if (auto reduction = dynamic_cast<ReductionOp*>(tv->definition())) {
          if (ir_utils::getMaybeWarpReductionDim(
                  reduction->out(), reduction->in())
                  .has_value()) {
            warp_pad_info_.has_warp_reduction = true;
          }
        }
      }

      // Check ifi TIDx is padded in this kernel
      if (id->hasPaddingToMultipleOfWarp()) {
        NVF_ERROR(
            id->getParallelType() == ParallelType::TIDx,
            "Padded types supported only on TIDx");
        warp_pad_info_.is_tidx_padded = true;
      }

      // Check all possible bindings of TIDx to see
      //  if TIDx will eventually be bound to a single warp.
      if (id->getParallelType() == ParallelType::TIDx) {
        auto size_after_padding = id->getMaybeSizeAfterPadding();
        bool padding_to_single_warp = size_after_padding.has_value() &&
            size_after_padding.value() == warp_size;

        if (id->extent()->isConstInt() &&
            id->extent()->evaluate().as<int64_t>() > warp_size &&
            !padding_to_single_warp) {
          // If we see any other TIDx binding that's larger than
          //  a warp or unknown, we shouldn't lower warp reduce
          //  to a single warp type.
          can_be_single_warp = false;
          warp_pad_info_.is_tidx_single_warp = false;
        } else if (can_be_single_warp) {
          if (padding_to_single_warp ||
              (id->extent()->isConstInt() &&
               id->extent()->evaluate().as<int64_t>() == warp_size)) {
            warp_pad_info_.is_tidx_single_warp = true;
          }
        }
      }
    }
  }
}

void segmenterHintCleanup(Fusion* fusion) {
  for (auto expr : fusion->exprs()) {
    if (expr->isA<LoadStoreOp>()) {
      auto op = expr->as<LoadStoreOp>();
      if (op->opType() == LoadStoreOpType::SegmenterSet) {
        op->setOpType(LoadStoreOpType::Set);
      }
    }
  }
}

std::tuple<Val*, Val*, kir::GetRNGSeedAndOffsetFromHost*>
getRNGSeedAndOffsetFromHost();

void assignRNGOffset(Fusion* fusion) {
  Val* seed = nullptr;
  Val* first_offset = nullptr;
  kir::GetRNGSeedAndOffsetFromHost* getseed_op = nullptr;
  int64_t counter = 0;
  for (auto expr : fusion->exprs()) {
    if (auto rop = dynamic_cast<RNGOp*>(expr)) {
      if (!rop->isDeterministic()) {
        if (seed == nullptr) {
          std::tie(seed, first_offset, getseed_op) =
              getRNGSeedAndOffsetFromHost();
        }
        Val* offset = SimplifyingIrBuilder::addExpr(first_offset, counter);
        rop->setSeedAndOffset(seed, offset);
        counter++;
      }
    }
  }
  if (getseed_op != nullptr) {
    getseed_op->offsets() = counter;
  }
}

// Dump expr string if enable lower_verbose
void dumpExprsIfEnabled(
    const std::vector<Expr*>& exprs,
    std::string pass_name,
    bool force_enable = false) {
  auto enabled_by_env = [&pass_name]() {
    if (!isDebugDumpEnabled(DebugDumpOption::LowerVerbose)) {
      return false;
    }
    const auto& args = getDebugDumpArguments(DebugDumpOption::LowerVerbose);
    return (
        args.empty() ||
        std::find(args.begin(), args.end(), pass_name) != args.end());
  };
  if (force_enable || enabled_by_env()) {
    debug() << "After " << pass_name << ":" << std::endl;
    for (auto exp : exprs) {
      // `Expr::toString()` already ends with a new line.
      debug() << exp->toString();
    }
  }
}

GpuLower::GpuLower(Fusion* fusion, const CompileParams& cparams)
    : passes_(
          // Passes will be executed in the order they are added here
          // Each pass is a pair of (name, function), where the name will be
          // printed in verbose mode of lowering. The function must take a
          // const std::vector<Expr*>& and return a std::vector<Expr*>.
          {{"removeTensorProducerAliases", removeTensorProducerAliases},
           {"LoopNestGenerator", LoopNestGenerator::loweredExprs},
           {"loadStoreOpInserter", loadStoreOpInserter},
           {"insertGridSerializationSyncs", insertGridSerializationSyncs},
           {"insertAllocations", insertAllocations},
           {"reuseMemoryAllocations", reuseMemoryAllocations},
           {"CircularBufferPass", CircularBufferPass::run},
           {"insertRawThreadSynchronization", insertRawThreadSynchronization},
           {"insertWarThreadSynchronization", insertWarThreadSynchronization},
           {"insertWarAsyncWait", insertWarAsyncWait},
           {"rotateLoops", rotateLoops},
           {"UnrollPass", UnrollPass::runPass},
           {"IndexLowering", IndexLowering::getIndexedExprs},
           {"fuseWarpReduce", fuseWarpReduce},
           {"generateConditionalFromPredicate",
            generateConditionalFromPredicate},
           {"vectorizeWelford", vectorizeWelford},
           {"addRNG", addRNG},
           {"allocateCommonScalars", allocateCommonScalars},
           {"insertMagicZero", insertMagicZero},
           {"KIRCleaner", KIRCleaner::cleanUp},
           {"instrumentKernel", instrumentKernel},
           {"lowerToInlinePtx", lowerToInlinePtx}}),
      cparams_(cparams) {
  if (isDebugDumpEnabled(DebugDumpOption::FusionIrMath)) {
    fusion->printMath();
  }
  if (isDebugDumpEnabled(DebugDumpOption::FusionIr)) {
    fusion->print();
  }

  analysis(fusion);
}

namespace {
struct LowerGuard {
  LowerGuard(GpuLower* gpu_lower) {
    active_gpu_lower = gpu_lower;
  }
  ~LowerGuard() {
    active_gpu_lower = nullptr;
  }
};

} // namespace

kir::Kernel* GpuLower::run() {
  FusionGuard fg(fusion_);
  LowerGuard lower_guard(this);
  // Reorder expressions for loop-nest generation respecting computeAt
  // relationships
  auto exprs_lowered = reorderExprsForComputeAt();
  dumpExprsIfEnabled(exprs_lowered, "reorderExprsForComputeAt");

  commonScalarMap().initialize(exprs_lowered);

  // For RNG ops whose seed and offset are not yet set, grab the seed and offset
  // from the host and assign them to the ops.
  // This must be after expr sort, because we do not want the generated
  // computation of offset and seed to be considered as part of fusion
  // definition
  assignRNGOffset(fusion_);

  for (auto [name, pass] : passes()) {
    exprs_lowered = pass(exprs_lowered);
    dumpExprsIfEnabled(exprs_lowered, name);
  }

  // We now have the lowered expressions, finalize the kernel IR. This function
  // will also copy over some relevant information for code generation from
  // GpuLower.
  kernel_->finalize(exprs_lowered);

  return kernel_.get();
}

namespace {

// Get IdModelOptions set through NVFUSER_ENABLE and overwritten for the
// given Fusion
IdModelOptions getIdModelOptions(Fusion* fusion) {
  IdModelOptions options;

  for (auto expr : fusion->exprs()) {
    if (auto ldst = dynamic_cast<LoadStoreOp*>(expr)) {
      if (ldst->opType() == LoadStoreOpType::CpAsyncBulkTensorTile ||
          ldst->opType() == LoadStoreOpType::CpAsyncBulk) {
        options.setBuildTensorIndexer(true);
        if (ldst->opType() == LoadStoreOpType::CpAsyncBulk) {
          options.setInlinePredicate(true);
        }
        continue;
      }
    } else if (expr->isA<MmaOp>()) {
      options.setBuildTensorIndexer(true);
      continue;
    } else if (expr->isOneOf<SliceOp, PadOp>()) {
      options.setProducerIndex(true);
      options.setConsumerIndex(true);
      options.setInlinePredicate(true);
      options.setUnswitchPredicate(true);
      options.setLoop(true);
      continue;
    } else if (auto reshape = dynamic_cast<ViewOp*>(expr)) {
      // The legacy indexer has an issue when an expand broadcast is
      // involved in reshape transformations. Enable both tensor and
      // predicate indexing if found

      auto producer_tv = reshape->in();
      auto consumer_tv = reshape->out();

      // Find expanded producer IDs. Note that corresponding consumer IDs do
      // not inherit the iteration type and are no longer expanded IDs, so the
      // producer domain needs to be checked to find expanded IDs.
      std::unordered_set<IterDomain*> expanded_ids;
      std::copy_if(
          producer_tv->getLogicalDomain().begin(),
          producer_tv->getLogicalDomain().end(),
          std::inserter(expanded_ids, expanded_ids.end()),
          [](IterDomain* logical_id) {
            return logical_id->isBroadcast() && logical_id->hasExpandedExtent();
          });

      if (expanded_ids.empty()) {
        continue;
      }

      // Find corresponding consumer root IDs
      auto c2p = PairwiseLogicalDomainMap(producer_tv, consumer_tv)
                     .mapConsumerToProducer();
      std::unordered_set<Val*> consumer_expanded_root_ids;
      for (auto consumer_root_id : consumer_tv->getRootDomain()) {
        auto producer_logical_id = c2p.at(consumer_root_id);
        if (expanded_ids.count(producer_logical_id)) {
          consumer_expanded_root_ids.insert(consumer_root_id);
        }
      }

      auto reshape_exprs = DependencyCheck::getAllExprsBetween(
          {consumer_tv->getRootDomain().begin(),
           consumer_tv->getRootDomain().end()},
          {consumer_tv->getLogicalDomain().begin(),
           consumer_tv->getLogicalDomain().end()});

      if (std::any_of(
              reshape_exprs.begin(),
              reshape_exprs.end(),
              [&consumer_expanded_root_ids](Expr* expr) {
                return std::any_of(
                    expr->inputs().begin(),
                    expr->inputs().end(),
                    [&](Val* input) {
                      return consumer_expanded_root_ids.count(input);
                    });
              })) {
        options.setProducerIndex(true);
        options.setConsumerIndex(true);
        options.setInlinePredicate(true);
        options.setUnswitchPredicate(true);
      }
    }
  }

  // If a tensor does not have a nice root->logical/allocation->loop
  // linear transformation history, use TensorIndexer
  for (auto tv : fusion->allTvs()) {
    if (tv->getMemoryType() == MemoryType::Tensor ||
        !ir_utils::hasRootToLoopLinearTransformations(tv)) {
      options.setBuildTensorIndexer(true);
    }
  }

  // If not supported, disable use of TensorIndexer by default. It is
  // still used if explicitly opted-in (see, for example,
  // Index::getConsumerIndex)
  if (!TensorIndexer::isSupported(fusion)) {
    // Do not disable building of TensorIndexer as it may be still used
    options.setIndex(false);
    options.setPredicate(false);
    options.setLoop(false);
  }

  return options;
}

} // namespace

void GpuLower::analysis(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::lower");
  NVF_ERROR(fusion != nullptr);
  NVF_ERROR(
      active_gpu_lower == nullptr, "Nested lowering passes are not supported");

  LowerGuard lower_guard(this);

  // Use int64 by default as the kernel index type
  if (!cparams_.index_type.has_value()) {
    cparams_.index_type = PrimDataType::Int;
  }

  // Copy fusion into a new kernel for processing
  kernel_ = std::make_unique<kir::Kernel>(fusion, indexType());
  // Alias the fusion kernel caries around as a view of itself.
  fusion_ = kernel_.get();

  dumpExprsIfEnabled(fusion_->exprs(), "initialize lowering");

  segmenterHintCleanup(fusion_);
  FusionGuard fg(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "segmenterHintCleanup");

  id_model_options_ = getIdModelOptions(fusion_);

  // Temporarily set allKnownVals to inputs. In the future, we will have a real
  // pass to determine how to set allKnownVals.
  // TODO: revisit all passes on how they handle exprs in the fusion. Should we
  // change their use of fusion_->exprs() to only include exprs that are not
  // between inputs and allKnownVals()?
  allKnownVals() = kernel_->inputs();
  dumpExprsIfEnabled(fusion_->exprs(), "set allKnownVals");

  // prepare for lowering
  validateIr(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateIr");

  // Determines minimum device version necessary to compile and run this fusion.
  std::tie(min_device_version_, min_device_version_reason_) =
      MinimumDeviceVersion::compute(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "MinimumDeviceVersion");

  // Checks if any TIDx dim is marked as padded to a warp. Also checks if we can
  // determine the padding is explicitly a single warp.
  collectPaddedParallelDims();
  dumpExprsIfEnabled(fusion_->exprs(), "collectPaddedParallelDims");

  // Replaces integers that are tensor sizes by named scalars as "T0.size[0]"
  replaceSymbolicSizes(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "replaceSymbolicSizes");

  // New IterDomains may be created, so it is expected that generated
  // code may use diffrent variable names
  if (idModelOptions().buildIdModel()) {
    id_model_ = std::make_unique<IdModel>(
        fusion_,
        /*build_graphs=*/true,
        /*allow_self_mapping=*/false,
        /*validate=*/false);
    id_model_->validateAndPropagatePType();
  }

  // Build what's refered to as the compute at map. This map contains the
  // mappings of all iteration domains across the fusion. There are three types
  // of mappings Permissive, Exact, and Loop, see compute_at_map.h/cpp for more
  // information.
  //
  // Depends on IdModel
  compute_at_map_ = std::make_shared<ComputeAtMap>(fusion_);

  // Requires IdModel as expression sorting is necessary
  resolveComputeWith(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "resolveComputeWith");

  if (isDebugDumpEnabled(DebugDumpOption::ComputeAtMap)) {
    debug() << compute_at_map_->toString() << std::endl;
  }
  compute_at_map_->validateAndPropagatePType();
  dumpExprsIfEnabled(fusion_->exprs(), "validateAndPropagatePType");

  // Uses compute_at_map, find all splits that are enforced to be divisible
  divisible_splits_ = getAllDivisibleSplits(fusion_, compute_at_map_.get());
  dumpExprsIfEnabled(fusion_->exprs(), "getAllDivisibleSplits");

  // Used in parallel dimension map
  concretized_broadcast_domains_ =
      std::make_shared<const ConcretizedBroadcastDomains>(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "build ConcretizedBroadcastDomains");

  parallelDimensionMap().build(fusion_);
  if (isDebugDumpEnabled(DebugDumpOption::ParallelDimensions)) {
    debug() << "Parallel dimension map:" << std::endl;
    debug() << parallel_dimension_map_.toString() << std::endl;
  }
  dumpExprsIfEnabled(fusion_->exprs(), "build parallelDimensionMap");

  validate1dTmaLoad(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validate1dTmaLoad");

  // Validate mma data format and compatibility if any on the fusion.
  validateMma(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateMma");

  // Validate swizzle usage on the fusion schedule.
  validateSwizzle(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateSwizzle");

  validateReductions(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateReductions");

  // Compute thread predicates. Depends on parallel_dimension_map_
  thread_pred_map_.build(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "build thread_pred_map_");

  // Fuse cetain patterns of reductions, such as a grid reduction
  // followed by a grid broadcast. Only depends on parallelization and
  // thread predicate map.
  fuseReductionsAndBroadcasts(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "fuseReductionsAndBroadcasts");

  // Depends on ComputeAtMap
  validateAndConvertIterDomainGrouping(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateAndConvertIterDomainGrouping");

  // Assumes all grouped reductions are convered to
  // GroupedReductionOp, which is done by
  // validateAndConvertIterDomainGrouping
  validateGroupedReductions(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateGroupedReductions");

  // Want to run this after parallel map is created.
  // Needs info about grouped reductions.
  // vectorized_accesses_ and vectorized_set_info_ are filled.
  validateAndCollectVectorizeInfo(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateAndCollectVectorizeInfo");

  // all of the lookup TVs are fusion inputs
  validateLookupTV(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "validateLookupTV");

  // Find trivial global to global broadcast, squeeze, and set operations and
  // mark their outputs as aliases of their inputs.
  findTensorProducerAliases(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "findTensorProducerAliases");

  // Depends on thread_pred_map_, validates parallelization collects which
  // tensor views need WAR or RAW syncs
  sync_map_ = std::make_shared<const SyncMap>(fusion_);
  if (isDebugDumpEnabled(DebugDumpOption::SyncMap)) {
    debug() << sync_map_->toString() << std::endl;
  }
  dumpExprsIfEnabled(fusion_->exprs(), "SyncMap");

  non_divisible_split_info_ = std::make_unique<NonDivisibleSplitInfo>(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "build nonDivisibleSplitInfo");

  circularBufferInfo().build(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "build circularBufferInfo");

  compute_at_map_->allocateIndexVariables();
  dumpExprsIfEnabled(fusion_->exprs(), "allocateIndexVariables");

  if (idModelOptions().loop()) {
    // Depends on CircularBufferInfo and compute_at_map_->allocateIndexVariables
    id_model_->allocateLoopIndexVariables();
  }

  if (idModelOptions().buildTensorIndexer()) {
    tensor_indexer_ = std::make_unique<TensorIndexer>(*id_model_);
    non_divisible_predicate_info_ =
        std::make_unique<NonDivisiblePredicateInfo>(fusion_);
  }

  // Detects all exprssions that don't need predicates. Depends on
  // nonDivisibleSplitInfo.
  pred_elimination_ = std::make_unique<PredicateElimination>(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "build predicateElimination");

  consumerToTMAInfo() = getConsumerToTMAInfoMap(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "getConsumerToTMAInfoMap");

  tmemInfo() = computeTMemInfo(fusion_);
  dumpExprsIfEnabled(fusion_->exprs(), "computeTMemInfo");
}

kir::Kernel* GpuLower::kernel() const {
  NVF_CHECK(kernel_);
  return kernel_.get();
}

GpuLower* GpuLower::current() {
  NVF_ERROR(active_gpu_lower != nullptr, "No active GpuLower available");
  return active_gpu_lower;
}

bool GpuLower::hasCurrent() {
  return active_gpu_lower != nullptr;
}

void GpuLower::propagateExprInfo(const Expr* old_expr, const Expr* new_expr) {
  predicateElimination().propagateRemovalInfo(old_expr, new_expr);
  if (old_expr->isA<kir::Allocate>()) {
    auto alloc_info_it =
        localAllocationInfoMap().find(old_expr->as<kir::Allocate>());
    if (alloc_info_it != localAllocationInfoMap().end()) {
      auto alloc_info =
          std::make_unique<LocalAllocationInfo>(*(alloc_info_it->second));
      localAllocationInfoMap().emplace(
          new_expr->as<kir::Allocate>(), std::move(alloc_info));
    }
  }
}

bool GpuLower::resolveComputeWith(Fusion* fusion) {
  std::vector<Expr*> exprs_sorted;

  bool updated = false;
  for (auto val : fusion->usedMathVals()) {
    auto tv = dynamic_cast<TensorView*>(val);
    if (tv == nullptr) {
      continue;
    }
    if (tv->hasComputeWith()) {
      if (exprs_sorted.empty()) {
        exprs_sorted = reorderExprsForComputeAt();
      }
      if (tv->resolveComputeWith(exprs_sorted)) {
        updated = true;
        compute_at_map_->updateComputeWith(tv);
      }
    }
  }

  // The Loop graph needs to be updated as the compute positions of
  // the updated tensors differ
  if (updated && hasIdModel()) {
    id_model_->removeGraph(IdMappingMode::LOOP);
    id_model_->buildGraph(IdMappingMode::LOOP);
    id_model_->validateAndPropagatePType();
  }

  return updated;
}

Val* GpuLower::getLoopIndexVariable(
    IterDomain* id,
    CircularBufferLoopStage stage) const {
  if (idModelOptions().loop()) {
    return idModel().getLoopIndexVariable(id, stage);
  } else {
    return caMap()->getIndexVariable(id, stage);
  }
}

void GpuLower::aliasTensorProducer(TensorView* consumer, TensorView* producer) {
  if (TensorView* producer_alias = getTensorProducerAlias(producer)) {
    // Chase reference. If producer itself is an alias, then get the tensor it
    // is aliased to.
    NVF_ERROR(
        getTensorProducerAlias(producer_alias) == nullptr,
        "Found unsimplified alias from ",
        producer->toString(),
        " to ",
        producer_alias->toString(),
        " which is then aliased to ",
        getTensorProducerAlias(producer_alias)->toString());
    producer = producer_alias;
  }
  tensor_producer_alias_map_[consumer] = producer;
  for (auto& [c, p] : tensor_producer_alias_map_) {
    // If anything was previously aliased _to_ consumer, update those links to
    // point to producer
    if (p == consumer) {
      p = producer;
    }
  }
}

const AllocationDomainInfo& GpuLower::getAllocationInfo(TensorView* tv) const {
  auto it = allocationInfo().find(tv);
  NVF_ERROR(
      it != allocationInfo().end(),
      "Allocation info not found for ",
      tv->toString());
  return it->second;
}

} // namespace nvfuser
