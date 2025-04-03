// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/cuda/CUDAContext.h>
#include <multidevice/utils.h>
#include <scheduler/ampere_multi_matmul.h>
#include <scheduler/hopper_multi_matmul.h>
#include <scheduler/utils.h>

namespace nvfuser {

void MultipleMatmulScheduler::findPatterns() {
  patterns_ = mma_utils::findMatmulPatterns(fusion_);
  NVF_ERROR(!patterns_.empty(), "No matmul patterns were found");
}

void MultipleMatmulScheduler::translatePatterns() {
  mma_results_.reserve(patterns_.size());
  for (mma_utils::MatmulPattern& pattern : patterns_) {
    // TODO: properly handle all mul+sum patterns for Hopper. For now, these
    // should work fine as long as the inner dimensions are the ones being
    // reduced.
    if (!isAmpere(params_->mma_macro) && !isTuring(params_->mma_macro) &&
        pattern.output->definition()->isA<ReductionOp>()) {
      bool found_reduction = false;
      for (size_t dim : arange((size_t)pattern.output->nDims())) {
        NVF_ERROR(
            !found_reduction ||
                !pattern.output->axis((int64_t)dim)->isReduction(),
            "Mul+Sum patterns can only be translated on Hopper if the reduction dim is innermost");
      }
    }

    mma_utils::MatmulPattern::TranslationResult res =
        pattern.translateToMmaOp();
    mma_results_.push_back(res.mma->out()->as<TensorView>());

    // During MatmulPattern translation, we might replace some tensors in the
    // fusion. If those replaced tensors were themselves the A or B members of
    // another MatmulPattern, we should update the pattern to point to the
    // replacement.
    for (mma_utils::MatmulPattern& other_pattern : patterns_) {
      if (&other_pattern == &pattern) {
        continue;
      }
      if (auto it = res.replacements.find(other_pattern.A);
          it != res.replacements.end()) {
        other_pattern.A = it->second;
      }
      if (auto it = res.replacements.find(other_pattern.B);
          it != res.replacements.end()) {
        other_pattern.B = it->second;
      }
    }
  }
}

// Get tensor roles and id roles
// When there are multiple matmul patterns, we can have conflicting roles.
// For now we throw an error if this is the case.
// TODO: This should be checked in canScheduleCompileTime
void MultipleMatmulScheduler::findRoles() {
  // Build IdModel graphs now since translateToMmaOp creates new TVs. Before
  // this point the graphs are not yet built.
  updateIdModel();

  const auto roles_opt = mma_utils::allPatternRoles(id_model_, patterns_);
  NVF_ERROR(
      roles_opt.has_value(),
      "Incompatible roles found between matmul patterns");
  std::tie(id_roles_, tensor_roles_) = roles_opt.value();

  mma_utils::MatmulOperandInnerDimsOpt inner_dims_opt =
      mma_utils::getOperandInnerDims(id_model_, id_roles_, tensor_roles_);
  NVF_ERROR(inner_dims_opt.isValid(), inner_dims_opt.getErrorMsg());
  inner_dims_ = inner_dims_opt.getData();

  countDims();
}

void MultipleMatmulScheduler::countDims() {
  NVF_ERROR(!patterns_.empty());
  TensorView* mma_result = patterns_.front().output;
  num_device_dims_ = numDeviceDims(mma_result);
  for (const auto& it : id_roles_) {
    if (it.second == MatmulDimRole::Batch &&
        // Skip device dims
        !std::any_of(it.first->begin(), it.first->end(), [](Val* v) {
          return v->as<IterDomain>()->isDeviceDim();
        })) {
      // All batch dims will be merged into one, if any exist
      num_local_batch_dims_ = 1;
    }
  }
  num_splitk_dims_ = params_->splitk_factor > 1 ? 1 : 0;
  // Subtract 6 for the [Mo, No, Ko, Mi, Ni, Ki]
  num_device_and_batch_dims_ = num_device_dims_ + num_local_batch_dims_;
}

//! Rebuilds IdModel, then updates all ValGroups in abstract tensors to refer
//! to the new IdModel. This is necessary whenever we perform an operation
//! that creates a new TensorView, such as caching or rFactor
void MultipleMatmulScheduler::updateIdModel() {
  // Build new IdModel
  IdModel new_id_model(fusion_, /*build_graphs=*/false);
  new_id_model.buildBroadcastGraph();

  // Get new broadcast graph
  ValGraph& new_graph = new_id_model.idGraph(IdMappingMode::BROADCAST);

  if (!id_roles_.empty()) {
    // Update id_roles_ to have keys corresponding to ValGroups in the new
    // IdModel
    std::unordered_map<ValGroup, MatmulDimRole> new_id_roles;
    for (auto& [k, v] : id_roles_) {
      const ValGroup& new_group = new_graph.toGroup(k->front());
      new_id_roles.emplace(new_group, v);
    }
    id_roles_ = new_id_roles;
  }

  graph_ = &new_id_model.idGraph(IdMappingMode::BROADCAST);

  // Set id_model_ after we are done using the old one
  id_model_ = std::move(new_id_model);
}

void scheduleMultipleMatmuls(Fusion* fusion, const MatmulParams* params) {
  FusionGuard fg(fusion);

  // NOTE: In the future we should be able to simply check the generation of
  // the macro instead of looking at the device properties here. However,
  // until we have Hopper mma ready, we will be using Ampere macros on Hopper
  // machines for testing. This means in order to trigger Hopper code, we need
  // to look at the device instead of the macro for now. See commented
  // conditions below.
  const auto device_prop = at::cuda::getCurrentDeviceProperties();
  const int cc = device_prop->major * 10 + device_prop->minor;
  if (cc >= 75 && cc < 90) {
    AmpereMultipleMatmulScheduler(fusion, params).run();
  } else if (cc >= 90 && cc < 100) {
    HopperMultipleMatmulScheduler(fusion, params).run();
  } else {
    NVF_THROW(
        "The matrix multiplication scheduler is unavailable for this device: ",
        device_prop->major,
        ".",
        device_prop->minor);
  }
}

void MultipleMatmulScheduler::cacheInputsAndOutputs(bool skip_intermediates) {
  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion_);

  // Cache operands
  for (auto role : {MatmulTensorRole::OPERAND_A, MatmulTensorRole::OPERAND_B}) {
    VectorOfUniqueEntries<TensorView*> unique_operands;
    for (const mma_utils::MatmulPattern& pattern : patterns_) {
      TensorView* immediate_operand =
          role == MatmulTensorRole::OPERAND_A ? pattern.A : pattern.B;
      for (Val* v : InputsOf::output(immediate_operand)) {
        if (auto* tv = dynamic_cast<TensorView*>(v)) {
          unique_operands.pushBack(tv);
        }
      }
    }
    std::vector<TensorView*>& operands =
        role == MatmulTensorRole::OPERAND_A ? as_ : bs_;
    std::vector<TensorView*>& cw_smems =
        role == MatmulTensorRole::OPERAND_A ? acw_smems_ : bcw_smems_;
    int64_t vec_size = role == MatmulTensorRole::OPERAND_A
        ? params_->supported_vec_size.a
        : params_->supported_vec_size.b;

    NVF_ERROR(operands.empty());

    for (TensorView* tv : unique_operands.vector()) {
      // When translating MatmulOp or LinearOp with avoid_intermediates, we
      // introduce some intermediate tensors which need to be ignored during
      // lowering. We set as_ and bs_ to point at the last of these tensors
      // before their next consumer is in non-global memory. Then we cache it
      // and use that as the smem tensor.
      TensorView* remapped = skip_intermediates
          ? scheduler_utils::scheduleInputToSkipIntermediates(tv)
          : tv;
      TensorView* smem_tv = remapped->cacheAfter();
      operands.push_back(remapped);
      cw_smems.push_back(smem_tv);

      setOperandSmemLoadAndCacheOps(smem_tv, vec_size);
      smem_tv->setMemoryType(MemoryType::Shared);
    }
  }

  // Cache epilogue inputs
  if (auto it = tensor_roles_.find(MatmulTensorRole::EPILOGUE_INPUT);
      it != tensor_roles_.end()) {
    for (TensorView* tv : it->second) {
      TensorView* tv_cache = tv->cacheAfter();
      cached_epilogue_inputs_.emplace_back(tv, tv_cache);
    }
  }

  // Cache and fork outputs
  scheduler_utils::cacheAndForkOutputs(fusion_, /*unroll=*/true);
  // In case a member of mma_results_ is a fusion output, we need to do the
  // caching but we also need to update the input afterward
  for (TensorView*& mma_result : mma_results_) {
    if (mma_result->isFusionOutput()) {
      Expr* def = mma_result->definition();
      NVF_ERROR(def != nullptr && def->isA<LoadStoreOp>());
      mma_result = def->input(0)->as<TensorView>();
    }

    // Now that we are finished possibly redefining the inputs to the MmaOps,
    // we can set the macro for those ops
    auto* mma = dynamic_cast<MmaOp*>(mma_result->definition());
    NVF_ERROR(mma != nullptr);
    mma->setMacro(params_->mma_macro);
  }
}

} // namespace nvfuser
