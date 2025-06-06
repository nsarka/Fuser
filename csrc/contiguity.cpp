// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <ir/utils.h>
#include <iter_visitor.h>

#include <contiguity.h>

namespace nvfuser {

OrderedIdInformation::OrderedIdInformation(
    const std::vector<IterDomain*>& alloc_domain,
    std::shared_ptr<const ConcretizedBroadcastDomains> concrete_info)
    : active_ids_(alloc_domain), concrete_info_(std::move(concrete_info)) {
  if (alloc_domain.empty()) {
    return;
  }

  // Grab allocation ids and initialize them.
  for (const auto alloc_i : arange(alloc_domain.size())) {
    auto alloc_id = alloc_domain[alloc_i]->as<IterDomain>();

    // Initialize id_to_alloc_ids to map allocs to themselves
    id_to_alloc_ids_[alloc_id] = {alloc_id};

    // Initialize allocations as being made up of correctly ordered transforms.
    consistently_ordered_ids_.emplace(alloc_id);

    exclusively_consumes_allocs_.emplace(alloc_id);
  }
}

void OrderedIdInformation::traverseTo(const std::vector<IterDomain*>& ids) {
  // Iterate from the allocation domain to the provided ids and fill
  // consistently_ordered_ids_, id_to_alloc_ids_, and
  // exclusively_consumes_allocs_ for all the IDs
  auto exprs = StmtSort::getExprsBetween(
      {active_ids_.begin(), active_ids_.end()}, {ids.begin(), ids.end()});

  // Only forward traversal is considered. OrderedIdGroupInformation can include
  // Backward traversal.
  setCurrentDirection(Direction::Forward);
  for (auto expr : exprs) {
    OptInDispatch::dispatch(expr);
  }
}

bool OrderedIdInformation::checkExclusivelyConsumesAllocs(IterDomain* id) {
  NVF_ERROR(
      isActiveId(id),
      "Error replaying transforms in contiguous ID checker, expected ",
      id->toString(),
      " to be in the active ID set.");

  auto alloc_id_it = findAllocIDs(id);
  NVF_ERROR(
      alloc_id_it != id_to_alloc_ids_.end(),
      "Error replaying transforms in contiguous ID checker, couldn't find "
      "mapped allocs of ",
      id->toString());

  const auto& alloc_ids = alloc_id_it->second;

  // Check all the allocations of all other ids, to see if any alloc_ids in id
  // are also in them.
  for (auto other_active_id : active_ids_) {
    if (other_active_id == id || other_active_id == nullptr) {
      continue;
    }

    auto alloc_id_it = findAllocIDs(other_active_id);
    NVF_ERROR(
        alloc_id_it != id_to_alloc_ids_.end(),
        "Error replaying transforms in contiguous ID checker, couldn't find "
        "mapped allocs of ",
        other_active_id->toString());

    const auto& other_alloc_ids = alloc_id_it->second;

    for (auto other_alloc_id : other_alloc_ids) {
      if (alloc_ids.has(other_alloc_id)) {
        return false;
      }
    }
  }
  return true;
}

void OrderedIdInformation::handle(Merge* merge) {
  if (current_direction_ == Direction::Forward) {
    // If either aren't an active ID, it means the inputs were detected to not
    // be ordered correctly before hitting this expression.
    if (!isActiveId(merge->inner()) || !isActiveId(merge->outer())) {
      return;
    }

    const auto inner_pos = getActiveIdPos(merge->inner());
    const auto outer_pos = getActiveIdPos(merge->outer());

    const bool inner_ordered = isConsistentlyOrdered(merge->inner());
    const bool outer_ordered = isConsistentlyOrdered(merge->outer());

    // Get allocation ids of the two inputs
    const auto& inner_alloc_ids = findAllocIDs(merge->inner())->second;
    const auto& outer_alloc_ids = findAllocIDs(merge->outer())->second;

    // Update maps
    // Find the position inner would have to be considered ordered
    auto pos_after_outer = outer_pos + 1;
    for (; pos_after_outer < int64_t(active_ids_.size()); pos_after_outer++) {
      if (active_ids_[pos_after_outer] == nullptr) {
        // Can't be considered ordered after a nullptr
        break;
      }
      // When using IdModel, reduction domains are excluded from
      // allocation domains but loop promotion may pick reduction
      // domains, which should just be treated as normal domains.
      if (!using_id_graph_) {
        if (active_ids_[pos_after_outer]->isReduction() ||
            ((active_ids_[pos_after_outer]->isBroadcast() &&
              !isConcretized(active_ids_[pos_after_outer])))) {
          // Skip reduction or broadcast axes that aren't concretized in the
          // fusion
          continue;
        }
      }
      break;
    }

    // The output is ordered as long as the inputs were ordered and outer
    // position is directly left of the inner position.
    bool out_ordered = inner_ordered && outer_ordered;
    out_ordered = out_ordered &&
        // If inner_pos is before outer_pos it's not ordered correctly. If for
        // some reason it's the same, that would be an error.
        inner_pos > outer_pos &&
        // Inner could be a broadcast, so doesn't have to be right on
        // pos_after_outer as that ID (if it exists) should not be a broadcast.
        // However, merging over a broadcast should be fine.
        inner_pos <= pos_after_outer;

    if (!using_id_graph_) {
      // TODO: Concretization may prevent contiguous indexing or vectorization.
      //  It prevents contiguous indexing if the concretization is within the
      //  IDs that are used for indexing. For vectorization it just means we
      //  need to make sure the extents of the axes to the right of the
      //  broadcast allocation domain in the contigous merge is bigger than the
      //  vectorization dimension. And that the tensor buffer supports the
      //  vector word size (always done).
      //
      // This shouldn't matter when using the IdModel-based
      // indexer. When concretized, that should be reflected with the
      // indexing path, and as long as the indexing path has a
      // contiguous merge, its output should be safe to index. See
      // also ContigIndexingTest.ConcretizedBroadcastMerge for a
      // concrete example.
      bool outer_is_concretized_bcast =
          merge->outer()->isBroadcast() && isConcretized(merge->outer());
      bool inner_is_concretized_bcast =
          merge->inner()->isBroadcast() && isConcretized(merge->inner());
      out_ordered = out_ordered && !inner_is_concretized_bcast &&
          !outer_is_concretized_bcast;
    }

    if (out_ordered) {
      consistently_ordered_ids_.emplace(merge->out());
    }

    // Don't just remove active_ids_, as if we have something like:
    //   [i0, i1, i2, i3]
    //   ->merge(0, 2)
    //   ->merge(1)
    // The latter merge looks like it's ordered correctly, if we update the
    // active map as:
    //   [i0, i1, i2, i3] -> [i0*i2, i1, i3]
    // Hoever if we instead mark it as:
    //   [i0, i1, i2, i3] -> [i0*i2, i1, nullptr, i3]
    // Or:
    //   [i0, i1, i2, i3] -> [nullptr, i1, i0*i2, i3]
    // It's clear the second merge is not ordered correctly. Doesn't matter
    // which direction we put the iter domain in, prefer putting it in outer as
    // we often are looking for inner dimensions that are contiguous. We don't
    // want to always do this, as it could make ordered merges look non-ordered.
    // For exmaple: [i0, i1, i2, i3]
    // ->merge(0)
    // ->merge(1)
    // ->merge(0)
    // If it's updated as:
    // [i0, i1, i2, i3]
    // -> [i0*i1, nullptr, i2, i3]
    // -> [i0*i1, nullptr, i2*i3, nullptr]
    // Now the final merge looks non-ordered but it is. So only insert a nullptr
    // entry if the out is not ordered.
    active_ids_[outer_pos] = merge->out();

    if (!out_ordered) {
      active_ids_[inner_pos] = nullptr;
    } else {
      active_ids_.erase(active_ids_.begin() + inner_pos);
      for (auto i = outer_pos + 1; i < inner_pos; i++) {
        // If there's broadcast axes between outer and inner and the merge was
        // contiguous, there may be broadcasts between outer and inner that
        // cannot be ordered merged anywhere else so remove them.
        active_ids_.erase(active_ids_.begin() + outer_pos + 1);
      }
    }

    // Update the alloc_id entry for the output.
    VectorOfUniqueEntries<IterDomain*> alloc_ids = inner_alloc_ids;
    alloc_ids.pushBack(outer_alloc_ids);

    id_to_alloc_ids_[merge->out()] = alloc_ids;

    // Need to check this after updating active_ids_ and id_to_alloc_ids_
    if (checkExclusivelyConsumesAllocs(merge->out())) {
      exclusively_consumes_allocs_.emplace(merge->out());
    }
  } else {
    if (!isActiveId(merge->out())) {
      return;
    }
    // Check if merge->out() has its alloc IDs defined.
    auto it = findAllocIDs(merge->out());
    NVF_ERROR(
        it != id_to_alloc_ids_.end(), "Merge->out() alloc IDs must be known");
    const auto& out_alloc_ids = it->second;

    const auto out_pos = getActiveIdPos(merge->out());
    const bool out_is_ordered = isConsistentlyOrdered(merge->out());

    // Replace out with outer and inner in active_ids_
    active_ids_[out_pos] = merge->outer();
    active_ids_.insert(active_ids_.begin() + out_pos + 1, merge->inner());

    if (out_is_ordered) {
      consistently_ordered_ids_.emplace(merge->outer());
      consistently_ordered_ids_.emplace(merge->inner());
    }

    // Update the alloc_id entry for the outputs.
    id_to_alloc_ids_[merge->inner()] = out_alloc_ids;
    id_to_alloc_ids_[merge->outer()] = out_alloc_ids;
  }
}

void OrderedIdInformation::handle(Split* split) {
  if (current_direction_ == Direction::Forward) {
    if (!isActiveId(split->in())) {
      return;
    }

    const auto in_pos = getActiveIdPos(split->in());

    const bool in_ordered = isConsistentlyOrdered(split->in());

    // Get allocation ids of the input
    const auto& in_alloc_ids = findAllocIDs(split->in())->second;

    // Update map for outputs
    // Remove inputs from the active_ids_ and insert the output ID
    active_ids_[in_pos] = split->outer();
    active_ids_.insert(active_ids_.begin() + in_pos + 1, split->inner());

    // The outputs are ordered as long as the input is ordered.
    if (in_ordered) {
      consistently_ordered_ids_.emplace(split->outer());
      consistently_ordered_ids_.emplace(split->inner());
    }

    // Update the alloc_id entry for the outputs.
    id_to_alloc_ids_[split->outer()] = in_alloc_ids;
    id_to_alloc_ids_[split->inner()] = in_alloc_ids;
  } else { // Direction::Backward
    // Backward analysis assumes use of IdModel
    if (!using_id_graph_) {
      return;
    }

    if (!isActiveId(split->outer()) || !isActiveId(split->inner())) {
      return;
    }

    const auto outer_pos = getActiveIdPos(split->outer());
    const auto inner_pos = getActiveIdPos(split->inner());

    const bool outer_ordered = isConsistentlyOrdered(split->outer());
    const bool inner_ordered = isConsistentlyOrdered(split->inner());

    // Get allocation ids of the two inputs
    const auto& outer_alloc_ids = findAllocIDs(split->outer())->second;
    const auto& inner_alloc_ids = findAllocIDs(split->inner())->second;

    // The output is ordered as long as the inputs are ordered and outer is
    // directly to the left of inner. The forward analysis of Merge considers
    // the case where the inner position is not directly next to the outer
    // position because there may be broadcast. However, broadcast IDs should
    // not be included in analyzed allocation IDs when IdModel is used.
    bool out_ordered =
        outer_ordered && inner_ordered && (inner_pos == outer_pos + 1);

    if (out_ordered) {
      consistently_ordered_ids_.emplace(split->in());
    }

    active_ids_[outer_pos] = split->in();

    if (out_ordered) {
      active_ids_.erase(active_ids_.begin() + inner_pos);
    } else {
      active_ids_[inner_pos] = nullptr;
    }

    VectorOfUniqueEntries<IterDomain*> alloc_ids = inner_alloc_ids;
    alloc_ids.pushBack(outer_alloc_ids);
    id_to_alloc_ids_[split->in()] = alloc_ids;

    if (checkExclusivelyConsumesAllocs(split->in())) {
      exclusively_consumes_allocs_.emplace(split->in());
    }
  }
}

// Swizzle generally can't be contiguous because of the non-affine nature of it,
// but we can still analyze the operation in the same way as merge/split.
void OrderedIdInformation::handle(Swizzle* swizzle) {
  IterDomain* in_x_id = current_direction_ == Direction::Forward
      ? swizzle->inX()
      : swizzle->outX();
  IterDomain* in_y_id = current_direction_ == Direction::Forward
      ? swizzle->inY()
      : swizzle->outY();
  IterDomain* out_x_id = current_direction_ == Direction::Forward
      ? swizzle->outX()
      : swizzle->inX();
  IterDomain* out_y_id = current_direction_ == Direction::Forward
      ? swizzle->outY()
      : swizzle->inY();

  if (!isActiveId(in_x_id) || !isActiveId(in_y_id) || !isActiveId(out_x_id) ||
      !isActiveId(out_y_id)) {
    return;
  }

  const auto in_x_pos = getActiveIdPos(in_x_id);
  const auto in_y_pos = getActiveIdPos(in_y_id);

  const bool in_x_ordered = isConsistentlyOrdered(in_x_id);
  const bool in_y_ordered = isConsistentlyOrdered(in_y_id);

  // Get allocation ids of the two inputs
  const auto& in_x_alloc_ids = findAllocIDs(in_x_id)->second;
  const auto& in_y_alloc_ids = findAllocIDs(in_y_id)->second;

  // Update map for outputs
  // Remove inputs from the active_ids_ and insert the output ID
  active_ids_[in_x_pos] = out_x_id;
  active_ids_[in_y_pos] = out_y_id;

  // In the case of no real swizzle we can forward properties on each domain
  // independently.
  if (swizzle->swizzleType() == SwizzleType::NoSwizzle) {
    if (in_x_ordered) {
      consistently_ordered_ids_.emplace(out_x_id);
    }

    if (exclusivelyConsumesAllocs(in_x_id)) {
      exclusively_consumes_allocs_.emplace(out_x_id);
    }

    if (in_y_ordered) {
      consistently_ordered_ids_.emplace(out_y_id);
    }

    if (exclusivelyConsumesAllocs(in_y_id)) {
      exclusively_consumes_allocs_.emplace(out_y_id);
    }

    id_to_alloc_ids_[out_x_id] = in_x_alloc_ids;
    id_to_alloc_ids_[out_y_id] = in_y_alloc_ids;
  } else {
    VectorOfUniqueEntries<IterDomain*> alloc_ids = in_x_alloc_ids;
    alloc_ids.pushBack(in_y_alloc_ids);
    id_to_alloc_ids_[out_x_id] = alloc_ids;
    id_to_alloc_ids_[out_y_id] = alloc_ids;
  }
}

// Swizzle generally can't be contiguous because of the non-affine nature of it,
// but we can still analyze the operation in the same way as merge/split.
void OrderedIdInformation::handle(Swizzle2D* swizzle) {
  IterDomain* in_x_id = current_direction_ == Direction::Forward
      ? swizzle->inX()
      : swizzle->outX();
  IterDomain* in_y_id = current_direction_ == Direction::Forward
      ? swizzle->inY()
      : swizzle->outY();
  IterDomain* out_x_id = current_direction_ == Direction::Forward
      ? swizzle->outX()
      : swizzle->inX();
  IterDomain* out_y_id = current_direction_ == Direction::Forward
      ? swizzle->outY()
      : swizzle->inY();

  if (!isActiveId(in_x_id) || !isActiveId(in_y_id) || !isActiveId(out_x_id) ||
      !isActiveId(out_y_id)) {
    return;
  }

  const auto in_x_pos = getActiveIdPos(in_x_id);
  const auto in_y_pos = getActiveIdPos(in_y_id);

  const bool in_x_ordered = isConsistentlyOrdered(in_x_id);
  const bool in_y_ordered = isConsistentlyOrdered(in_y_id);

  // Get allocation ids of the two inputs
  const auto& in_x_alloc_ids = findAllocIDs(in_x_id)->second;
  const auto& in_y_alloc_ids = findAllocIDs(in_y_id)->second;

  // Update map for outputs
  // Remove inputs from the active_ids_ and insert the output ID
  active_ids_[in_x_pos] = out_x_id;
  active_ids_[in_y_pos] = out_y_id;

  // In the case of no real swizzle we can forward properties on each domain
  // independently.
  if (swizzle->swizzleType() == Swizzle2DType::NoSwizzle) {
    if (in_x_ordered) {
      consistently_ordered_ids_.emplace(out_x_id);
    }

    if (exclusivelyConsumesAllocs(in_x_id)) {
      exclusively_consumes_allocs_.emplace(out_x_id);
    }

    if (in_y_ordered) {
      consistently_ordered_ids_.emplace(out_y_id);
    }

    if (exclusivelyConsumesAllocs(in_y_id)) {
      exclusively_consumes_allocs_.emplace(out_y_id);
    }

    id_to_alloc_ids_[out_x_id] = in_x_alloc_ids;
    id_to_alloc_ids_[out_y_id] = in_y_alloc_ids;
  } else {
    VectorOfUniqueEntries<IterDomain*> alloc_ids = in_x_alloc_ids;
    alloc_ids.pushBack(in_y_alloc_ids);
    id_to_alloc_ids_[out_x_id] = alloc_ids;
    id_to_alloc_ids_[out_y_id] = alloc_ids;
  }
}

void OrderedIdInformation::handle(Resize* resize) {
  IterDomain* in_id =
      current_direction_ == Direction::Forward ? resize->in() : resize->out();
  IterDomain* out_id =
      current_direction_ == Direction::Forward ? resize->out() : resize->in();

  if (!isActiveId(in_id)) {
    return;
  }

  auto in_pos = getActiveIdPos(in_id);

  const bool in_ordered = isConsistentlyOrdered(in_id);

  // Get allocation ids of the two inputs
  const auto& in_alloc_ids = findAllocIDs(in_id)->second;

  // Update map for outputs
  // Remove inputs from the active_ids_ and insert the output ID
  active_ids_[in_pos] = out_id;

  // Not completely certain, but propagating these properties should e
  // fine
  if (in_ordered) {
    consistently_ordered_ids_.emplace(out_id);
  }

  if (exclusivelyConsumesAllocs(in_id)) {
    exclusively_consumes_allocs_.emplace(out_id);
  }

  id_to_alloc_ids_[out_id] = in_alloc_ids;
}

NonDivisibleSplitDependencies::NonDivisibleSplitDependencies(
    // TODO: Revisit reduction rfactor axes and propagation. Should probably use
    // ca_map to propogate non divisibility dependencies across exact map. Still
    // need to think through divisible split and non divisible dependencies to
    // see if there's conflicts where a split might look non divisible but
    // actually is divisible and one's overruling the other.
    const std::vector<IterDomain*>& ids,
    const std::vector<IterDomain*>& alloc_domain,
    const std::unordered_set<Split*>& divisible_splits) {
  if (ids.empty() || alloc_domain.empty()) {
    return;
  }
  auto transforms = StmtSort::getExprsBetween(
      {alloc_domain.begin(), alloc_domain.end()}, {ids.begin(), ids.end()});
  for (auto transform : transforms) {
    auto inp_ids = ir_utils::filterByType<IterDomain>(transform->inputs());
    for (auto inp_id : inp_ids) {
      if (std::find(alloc_domain.begin(), alloc_domain.end(), inp_id) !=
          alloc_domain.end()) {
        // This generally shouldn't happen as there shouldn't be
        // transformations before the allocation ids, but in case for some
        // reason we eventually do have cases like that, we should reset the
        // alloc_ids if for some reason they've been placed in the non
        // divisible split set.
        depends_on_non_divisible_split.erase(inp_id);
      }
    }

    bool inputs_non_divisible =
        std::any_of(inp_ids.begin(), inp_ids.end(), [this](IterDomain* inp_id) {
          return depends_on_non_divisible_split.find(inp_id) !=
              depends_on_non_divisible_split.end();
        });

    auto out_ids = ir_utils::filterByType<IterDomain>(transform->outputs());

    if (inputs_non_divisible) {
      // If any inputs are known to be dependent on a divisible split
      // Mark outputs as dependent on a non_divisible split
      depends_on_non_divisible_split.insert(out_ids.begin(), out_ids.end());
      continue;
    }

    if (!transform->isA<Split>()) {
      continue;
    }

    auto split = transform->as<Split>();
    // If this transform is a non-divisible split
    if (divisible_splits.find(split) == divisible_splits.end()) {
      // Mark outputs as dependent on a non_divisible split
      auto out_ids = ir_utils::filterByType<IterDomain>(transform->outputs());
      depends_on_non_divisible_split.insert(out_ids.begin(), out_ids.end());
    }
  }
}

ContigIDs::ContigIDs(
    const std::vector<IterDomain*>& ids,
    const std::vector<IterDomain*>& alloc_domain,
    const std::vector<std::optional<bool>>& alloc_contiguity,
    const std::unordered_set<IterDomain*>& final_ids,
    const std::unordered_map<IterDomain*, Val*>& index_map,
    const std::unordered_set<Split*>& divisible_splits,
    std::unordered_map<IterDomain*, IterDomain*> p2c_id_map,
    bool ignore_indexability,
    bool ignore_consistent_ordering)
    : alloc_domain_(alloc_domain),
      alloc_contiguity_(alloc_contiguity),
      final_ids_(final_ids),
      index_map_(index_map),
      divisible_splits_(divisible_splits),
      p2c_id_map_(std::move(p2c_id_map)),
      ignore_indexability_(ignore_indexability),
      ignore_consistent_ordering_(ignore_consistent_ordering),
      non_divisible_id_info_(ids, alloc_domain_, divisible_splits_) {
  if (!ids.empty()) {
    // This constructor doesn't provide the following information so it needs to
    // be built.
    ca_map_ = std::make_shared<ComputeAtMap>(ids[0]->fusion());
    concrete_info_ =
        std::make_shared<ConcretizedBroadcastDomains>(ids[0]->fusion());

    consistent_transform_info_ = std::make_unique<const OrderedIdInformation>(
        OrderedIdInformation::get(ids, alloc_domain, concrete_info_));
  }
  build(ids);
}

ContigIDs::ContigIDs(
    const std::vector<IterDomain*>& ids,
    const std::vector<IterDomain*>& alloc_domain,
    const std::vector<std::optional<bool>>& alloc_contiguity,
    const std::unordered_set<IterDomain*>& final_ids,
    const std::unordered_map<IterDomain*, Val*>& index_map,
    const std::unordered_set<Split*>& divisible_splits,
    std::shared_ptr<const ComputeAtMap> ca_map,
    std::shared_ptr<const ConcretizedBroadcastDomains> concrete_info,
    std::unordered_map<IterDomain*, IterDomain*> p2c_id_map,
    bool ignore_indexability,
    bool ignore_consistent_ordering)
    : alloc_domain_(alloc_domain),
      alloc_contiguity_(alloc_contiguity),
      final_ids_(final_ids),
      index_map_(index_map),
      divisible_splits_(divisible_splits),
      ca_map_(std::move(ca_map)),
      concrete_info_(std::move(concrete_info)),
      p2c_id_map_(std::move(p2c_id_map)),
      ignore_indexability_(ignore_indexability),
      ignore_consistent_ordering_(ignore_consistent_ordering),
      consistent_transform_info_(std::make_unique<const OrderedIdInformation>(
          OrderedIdInformation::get(ids, alloc_domain, concrete_info_))),
      non_divisible_id_info_(ids, alloc_domain, divisible_splits_) {
  build(ids);
}

ContigIDs ContigIDs::getNonContigIDs() {
  return ContigIDs({}, {}, {}, {}, {}, {});
}

void ContigIDs::build(const std::vector<IterDomain*>& ids) {
  if (ids.empty() || alloc_domain_.empty()) {
    return;
  }

  NVF_ERROR(
      alloc_domain_.size() == alloc_contiguity_.size(),
      "Arguments don't match ",
      alloc_domain_.size(),
      " != ",
      alloc_contiguity_.size());

  for (const auto alloc_domain_i : arange(alloc_domain_.size())) {
    auto alloc_domain_id = alloc_domain_.at(alloc_domain_i);
    if (alloc_domain_id->isBroadcast()) {
      NVF_ERROR(!alloc_contiguity_.at(alloc_domain_i).has_value());
      continue;
    }
    alloc_to_indexed_id_[alloc_domain_id] = alloc_domain_id;
    // Initialize to false
    is_contig_alloc_[alloc_domain_id] = false;
    auto alloc_contiguity = alloc_contiguity_.at(alloc_domain_i);
    NVF_ERROR(
        alloc_domain_id->isReduction() != alloc_contiguity.has_value(),
        "Expecting a reduction because contiguity has no value, get ",
        alloc_domain_id->toString());
    // Index of merged reductions can always be coalesced, so considering
    // reduction as true contiguity.
    if (alloc_contiguity.value_or(true) &&
        alloc_domain_id->getIterType() != IterType::GatherScatter) {
      contig_ids_.emplace(alloc_domain_id);
      is_contig_alloc_.at(alloc_domain_id) = true;
      within_contig_ids_[alloc_domain_id] = std::unordered_set<IterDomain*>();
    }
  }

  if (!contig_ids_.empty()) {
    auto exprs = StmtSort::getExprsBetween(
        {alloc_domain_.begin(), alloc_domain_.end()}, {ids.begin(), ids.end()});
    for (auto expr : exprs) {
      if (auto resize = dynamic_cast<Resize*>(expr)) {
        resize_deps_.insert(resize->out());
      } else {
        NVF_ERROR(expr != nullptr);
        if (std::any_of(
                expr->inputs().begin(), expr->inputs().end(), [&](Val* inp) {
                  return inp->isA<IterDomain>() &&
                      resize_deps_.count(inp->as<IterDomain>());
                })) {
          for (auto out : ir_utils::filterByType<IterDomain>(expr->outputs())) {
            resize_deps_.insert(out);
          }
        }
      }
      dispatch(expr);
    }
  }
}

void ContigIDs::handle(Merge* merge) {
  if (isOptionDisabled(DisableOption::ContigIndexing)) {
    return;
  }

  // If output is not consistently ordered or doesn't solely consume all
  // allocation domains in its dependencies, then it can't be a contiguously
  // indexable iterdomain.
  if (!(ignore_consistent_ordering_ ||
        consistent_transform_info_->isConsistentlyOrdered(merge->out()))) {
    return;
  }

  if (!consistent_transform_info_->exclusivelyConsumesAllocs(merge->out())) {
    return;
  }

  // If output is not "directly indexable" then it's definitely not contiguously
  // indexable.
  if (!ignore_indexability_ && !isIndexable(merge->out())) {
    return;
  }

  // If inputs are marked as final, stop
  if (final_ids_.count(merge->inner()) || final_ids_.count(merge->outer())) {
    return;
  }

  // Check allocation domains for contiguity
  auto alloc_ids_it =
      consistent_transform_info_->idToAllocIds().find(merge->out());

  NVF_ERROR(
      alloc_ids_it != consistent_transform_info_->idToAllocIds().end(),
      "\nError in contiguous analysis, merge info doesn't exist for:\n",
      merge->toString(),
      "\nId: ",
      merge->out()->toString());

  VectorOfUniqueEntries<IterDomain*> alloc_ids = alloc_ids_it->second;

  bool is_indexing_pass = !ignore_consistent_ordering_;

  IterDomain* last_alloc = nullptr;
  for (auto alloc_id_i : arange(alloc_domain_.size())) {
    auto alloc_id = alloc_domain_[alloc_id_i];
    if (alloc_id->isBroadcast()) {
      NVF_ERROR(!alloc_contiguity_.at(alloc_id_i).has_value());
      continue;
    }
    if (alloc_ids.has(alloc_id)) {
      // ID found, remove it
      alloc_ids.erase(alloc_id);
      // If we're indexing:
      // we could still potentially consider this ID linearly indexable, as we
      // could multiple the index by the last allocation's stride.
      //
      // If we're computing predicates (ignore_consistent_ordering_==true),
      // then we don't have this same constraint, we can just ignore
      // contiguity of the allocations all together.
      auto alloc_contiguity = alloc_contiguity_.at(alloc_id_i);
      NVF_ERROR(
          alloc_id->isReduction() != alloc_contiguity.has_value(),
          "Expecting a reduction because contiguity has no value, get ",
          alloc_id->toString());
      // Index of merged reductions can always be coalesced, so considering
      // reduction as true contiguity.
      if (!alloc_contiguity.value_or(true) && is_indexing_pass) {
        if (!alloc_ids.empty()) {
          return;
        }
      }
      last_alloc = alloc_id;
    }
  }

  // If there's a non_divisible split in the history of merge->out then it can't
  // be contiguously indexable.
  if (non_divisible_id_info_.dependsOnNonDivisibleSplit(merge->out())) {
    return;
  }

  // Don't allow contig indexing after resize as we need traverse back
  // at least to direct outputs of resize ops
  if (resize_deps_.count(merge->out())) {
    return;
  }

  // All broadcasting
  if (last_alloc == nullptr) {
    return;
  }

  // Now we know merge->out is a contiguously indexable ID

  // Reset alloc_ids
  alloc_ids = alloc_ids_it->second;
  for (auto alloc_id : alloc_ids) {
    alloc_to_indexed_id_[alloc_id] = merge->out();
  }

  auto all_within_vals = DependencyCheck::getAllValsBetween(
      {alloc_domain_.begin(), alloc_domain_.end()}, {merge->out()});
  auto all_within_ids = ir_utils::filterByType<IterDomain>(all_within_vals);

  std::unordered_set<IterDomain*> within_id_set(
      all_within_ids.begin(), all_within_ids.end());

  within_id_set.erase(merge->out());
  within_contig_ids_[merge->out()] = within_id_set;
  for (auto id : all_within_ids) {
    contig_ids_.erase(id);
  }

  contig_ids_.emplace(merge->out());
}

IterDomain* ContigIDs::getMappedId(IterDomain* id) const {
  auto it = p2c_id_map_.find(id);
  if (it != p2c_id_map_.end()) {
    return it->second;
  } else {
    return id;
  }
}

bool ContigIDs::isIndexable(IterDomain* id) const {
  // If ID is mapped to consumer through persmissive map but not exact map it
  // will not be mapped through to the exact map through the p2c map. Therefore
  // reject because it involves broadcast resolution.
  if (!ca_map_->idExistsInMap(getMappedId(id))) {
    return false;
  }
  auto c_id =
      ca_map_->getConcreteMappedID(getMappedId(id), IdMappingMode::EXACT);
  return index_map_.find(c_id) != index_map_.end();
}

} // namespace nvfuser
