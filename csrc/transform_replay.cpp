// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <transform_replay.h>

#include <compute_at_map.h>
#include <debug.h>
#include <disjoint_set.h>
#include <fusion.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <ops/arith.h>
#include <options.h>
#include <scheduler/tools/maxinfo_propagator.h>
#include <transform_iter.h>

#include <deque>

namespace nvfuser {

namespace {

using IterDomainMap = std::unordered_map<IterDomain*, IterDomain*>;

class ReplaySelf : public ReplayTransformations {
 private:
  // Took a good bit of this from ReplayTransformations::handle(Split...)
  void handle(Split* s) override {
    // Grab input to the split operation
    auto id_in = s->in();

    // Grab our mapping of that ID to the one we're replaying
    auto it = id_map_.find(id_in);
    if (it == id_map_.end()) {
      if (!error_on_failure_) {
        return;
      }
      // Make sure it exists in the map
      NVF_THROW("Transform traversal failed, dependencies not met.");
    }
    // Grab the ID we're going to replay on
    auto mapped = it->second;

    // This ID should be a loop ID (meaning it has no uses we generated)
    NVF_ERROR(
        loop_ids_.find(mapped) != loop_ids_.end(),
        "Transform traversal failed, modified a node but it was not a loop "
        "node.");

    NVF_ERROR(s->outer()->isRFactorProduct() == s->inner()->isRFactorProduct());

    // Due to rfactor transformations, the iter types of the outputs
    // may not follow the default rule. For example, even if the input
    // is a reduction iter domain, the outputs may not. To replay the
    // original split expression, the output iter types need to be
    // specified explicitly.
    auto [ido, idi] = IterDomain::split(
        mapped,
        s->factor(),
        s->innerSplit(),
        s->outer()->isRFactorProduct(),
        s->outer()->getIterType(),
        s->inner()->getIterType());

    // Remove mapped id from loop IDs
    loop_ids_.erase(mapped);

    // Add outputs to loop IDs
    loop_ids_[ido] = newCounter();
    loop_ids_[idi] = newCounter();

    // Update our ID map to include these outputs
    id_map_[s->outer()] = ido;
    id_map_[s->inner()] = idi;
  }

  void handle(Merge* m) override {
    auto id_outer = m->outer();
    auto id_inner = m->inner();

    auto it_outer = id_map_.find(id_outer);
    auto it_inner = id_map_.find(id_inner);
    if (it_outer == id_map_.end() || it_inner == id_map_.end()) {
      if (!error_on_failure_) {
        return;
      }
      NVF_THROW("Transform traversal failed, dependencies not met.");
    }
    auto id_outer_mapped = it_outer->second;
    auto id_inner_mapped = it_inner->second;

    NVF_ERROR(
        loop_ids_.find(id_outer_mapped) != loop_ids_.end() &&
            loop_ids_.find(id_inner_mapped) != loop_ids_.end(),
        "Transform traversal failed, modified ",
        id_outer_mapped,
        " and ",
        id_inner_mapped,
        " however one or both are not loop nodes.");

    IterDomain* merged_id = IterDomain::merge(id_outer_mapped, id_inner_mapped);

    // Remove inputs from the loop IDs
    loop_ids_.erase(id_outer_mapped);
    loop_ids_.erase(id_inner_mapped);

    // Add the output to the loop IDs
    loop_ids_[merged_id] = newCounter();

    id_map_[m->out()] = merged_id;
  }

  void handle(Swizzle* swizzle) override {
    NVF_THROW("Unexpected expr to self replay: ", swizzle->toString());
  }

  void handle(Swizzle2D* swizzle) override {
    NVF_THROW("Unexpected expr to self replay: ", swizzle->toString());
  }

  void handle(Resize* resize) override {
    auto id_in = resize->in();

    auto it = id_map_.find(id_in);
    if (it == id_map_.end()) {
      if (!error_on_failure_) {
        return;
      }
      NVF_THROW("Transform traversal failed, dependencies not met.");
    }
    auto mapped = it->second;

    NVF_ERROR(
        loop_ids_.find(mapped) != loop_ids_.end(),
        "Transform traversal failed, modified a node but it was not a loop "
        "node.");

    // When the original output is an rfactor, make the replayed
    // output domain also an rfactor
    const auto resize_out_logical = resize->out()->isRFactorProduct();

    auto replayed_out = IterDomain::resize(
        mapped,
        resize->leftExpand(),
        resize->rightExpand(),
        resize_out_logical);

    loop_ids_.erase(mapped);

    loop_ids_[replayed_out] = newCounter();

    id_map_[resize->out()] = replayed_out;
  }

 public:
  ReplaySelf(
      const std::vector<IterDomain*>& target_domain,
      IterDomainMap id_map)
      : ReplayTransformations(target_domain, std::move(id_map)) {
    setErrorOnFailure(false);
  }
};

} // namespace

// Self replay.
TensorDomain* TransformReplay::fullSelfReplay(
    const TensorDomain* new_self_root,
    const TensorDomain* self) {
  FUSER_PERF_SCOPE("TransformReplay::fullSelfReplay");

  NVF_ERROR(
      new_self_root->maybeRoot().size() == self->maybeRoot().size(),
      "Invalid number of IterDomains provided: ",
      new_self_root->maybeRoot(),
      " vs ",
      self->maybeRoot());

  // Map for replay, should be pretty simple.
  IterDomainMap axis_map;
  {
    int64_t i = 0;
    for (auto id : self->maybeRoot()) {
      NVF_ERROR(
          new_self_root->maybeRoot()[i]->isReduction() == id->isReduction() &&
              new_self_root->maybeRoot()[i]->isRFactorProduct() ==
                  id->isRFactorProduct() &&
              new_self_root->maybeRoot()[i]->isBroadcast() == id->isBroadcast(),
          "Axes ",
          id,
          " and ",
          new_self_root->maybeRoot()[i],
          " do not match for self replay.");
      axis_map[id] = new_self_root->maybeRoot()[i];
      i++;
    }
  }

  // Replay producer dimensions.
  ReplaySelf replay(self->loop(), axis_map);
  std::vector<IterDomain*> new_domain(self->nDims(), nullptr);

  {
    int64_t i = 0;
    for (auto id : self->loop()) {
      auto it = replay.getReplay().find(id);
      NVF_ERROR(
          it != replay.getReplay().end(),
          "Error during replay, didn't replay an axis.");
      new_domain[i++] = it->second;
    }

    if (self->hasRoot()) {
      std::vector<IterDomain*> new_logical_domain(
          self->logical().size(), nullptr);
      int64_t i = 0;
      for (auto id : self->logical()) {
        auto it = replay.getReplay().find(id);
        NVF_ERROR(
            it != replay.getReplay().end(),
            "Error during replay, didn't replay an axis.");
        new_logical_domain[i++] = it->second;
      }
      return IrBuilder::createInContainer<TensorDomain>(
          self->container(),
          new_self_root->root(),
          new_logical_domain,
          new_domain,
          self->contiguity());
    }
  }

  return IrBuilder::createInContainer<TensorDomain>(
      self->container(),
      new_self_root->logical(),
      new_domain,
      new_self_root->contiguity());
}

void TransformReplay::selfReplay(
    const TensorDomain* self,
    TensorDomain* new_self,
    bool ignore_reductions) {
  FUSER_PERF_SCOPE("TransformReplay::selfReplay");

  std::vector<IterDomain*> new_self_logical = new_self->logical();
  std::vector<IterDomain*> self_logical = self->logical();
  if (ignore_reductions) {
    new_self_logical = TensorDomain::noReductions(new_self_logical);
    self_logical = TensorDomain::noReductions(self_logical);
  }

  NVF_ERROR_EQ(new_self_logical.size(), self_logical.size());

  IterDomainMap axis_map;
  for (auto&& [id, new_id] : zip(self_logical, new_self_logical)) {
    // Note: we don't want to check for equal `isRFactorProduct`, since we
    // could replay Allocation of the output of a reduction to a later
    // consumer tensor, which would not have the rfactor flag on.
    //
    // Note: this function can be used prior to concretization, where we might
    // have unresolved symbolic ID, where the broadcast flag might mismatch.
    // We skip the check if either id or new_id is symbolic and expect a
    // correct user program.
    NVF_ERROR(
        new_id->isSymbolic() || id->isSymbolic() ||
            new_id->isBroadcast() == id->isBroadcast(),
        "Axes ",
        id,
        " and ",
        new_id,
        " do not match for self replay.");
    axis_map[id] = new_id;
  }

  // We create one ReplaySelf instance to replay loop and allocation. This way,
  // loop and allocation share the same transforms if they are split the same
  // way.
  //
  // We use `self_loop` as the target domain because loop post-dominates
  // allocation.
  const std::vector<IterDomain*>& self_loop = self->loop();
  ReplaySelf replay(self_loop, axis_map);

  // Replay loop.
  if (self_loop != self->logical()) {
    std::vector<IterDomain*> new_loop;
    if (ignore_reductions) {
      for (auto* id : new_self->logical()) {
        if (id->isReduction()) {
          new_loop.push_back(id);
        }
      }
    }

    for (IterDomain* loop_id : self_loop) {
      if (ignore_reductions && loop_id->isReduction()) {
        continue;
      }

      auto it = replay.getReplay().find(loop_id);
      NVF_ERROR(
          it != replay.getReplay().end(),
          "failed to replay IterDomain: ",
          loop_id);
      it->second->parallelize(loop_id->getParallelType());
      new_loop.push_back(it->second);
    }

    new_self->setLoopDomain(new_loop);
  }

  // Replay allocation.
  if (self->hasAllocation()) {
    const std::vector<IterDomain*>& self_allocation = self->allocation();
    const std::vector<std::optional<bool>>& self_contiguity =
        self->contiguity();
    NVF_ERROR_EQ(self_allocation.size(), self_contiguity.size());

    std::vector<IterDomain*> new_alloc_domain;
    std::vector<std::optional<bool>> new_contiguity;
    new_alloc_domain.reserve(self_allocation.size());
    new_contiguity.reserve(self_contiguity.size());

    // Push back the reduction IDs that are not mapped
    if (ignore_reductions) {
      for (auto* id : new_self->logical()) {
        if (id->isReduction()) {
          new_alloc_domain.push_back(id);
          // NOLINTNEXTLINE(modernize-use-emplace)
          new_contiguity.push_back(std::nullopt);
        }
      }
    }

    // Pushing the mapped IDs and corresponding contiguity flags
    for (auto&& [alloc_id, contiguity] :
         zip(self_allocation, self_contiguity)) {
      if (ignore_reductions && alloc_id->isReduction()) {
        continue;
      }
      auto it = replay.getReplay().find(alloc_id);
      NVF_ERROR(
          it != replay.getReplay().end(),
          "failed to replay IterDomain: ",
          alloc_id);
      NVF_ERROR_EQ(
          it->second->isBroadcast(),
          !contiguity.has_value(),
          "Contiguity should be nullopt iff broadcast.");
      new_contiguity.push_back(contiguity);
      it->second->parallelize(alloc_id->getParallelType());
      new_alloc_domain.push_back(it->second);
    }

    new_self->setAllocationDomain(new_alloc_domain, new_contiguity);
  }
}

namespace {

// Grab all IterDomains of producer or consumer that may not be mapped
// with consumer or producer, respectively, due to missing root
// mappings. No root mapping does not always mean dependent IDs are
// not mapped as there could be broadcast forwarded merges.
std::unordered_set<IterDomain*> getMaybeUnmappedIDs(
    const TensorView* tv,
    bool is_producer,
    const std::unordered_map<IterDomain*, IterDomain*>& root_id_map) {
  std::unordered_set<Val*> unmapped_root_ids;

  const auto& root_domain =
      is_producer ? tv->getLogicalDomain() : tv->getMaybeRootDomain();

  for (auto root_id : root_domain) {
    if (root_id_map.count(root_id) == 0) {
      unmapped_root_ids.emplace(root_id);
    }
  }

  auto all_unmapped_vals = DependencyCheck::getAllValsBetween(
      unmapped_root_ids,
      {tv->getLoopDomain().begin(), tv->getLoopDomain().end()});

  std::unordered_set<IterDomain*> all_unmapped_ids;
  std::transform(
      all_unmapped_vals.begin(),
      all_unmapped_vals.end(),
      std::inserter(all_unmapped_ids, all_unmapped_ids.end()),
      [](Val* val) { return val->as<IterDomain>(); });
  return all_unmapped_ids;
}

} // namespace

// Producer could have rfactor axes which consumer may want replayed. We can
// "replay" them as long as it doesn't modify the root rfactor axes. What we
// really want to do is validate if we replayed these axes to the ones they
// mapped to in the consumer the operations would all be the same. then we want
// to start the replay of the producer from the rfactor root axes, not the root.
std::pair<TensorDomain*, int64_t> TransformReplay::replayPasC(
    const TensorView* producer,
    const TensorView* consumer,
    int64_t consumer_pos,
    const LogicalDomainMap& logical_map,
    TransformReplayOptions opt) {
  FUSER_PERF_SCOPE("TransformReplay::replayPasC");
  if (producer == consumer) {
    return {producer->domain(), producer->nDims()};
  }
  consumer_pos = wrapDim(consumer_pos, consumer->nDims() + 1);

  // consumer ids we need to match in producer
  std::vector<IterDomain*> target_consumer_ids(
      consumer->getLoopDomain().begin(),
      consumer->getLoopDomain().begin() + consumer_pos);

  // Instead of replaying from the root, lets try to play forward the history of
  // producer if they match ops on consumer. Enforce if we modify an rfactor
  // axis that those ops must match.
  //
  // Swizzles should not be skipped in the BestEffortReplay matching in this
  // case. If a swizzle mismatch is found, by default BestEffortReplay forwards
  // the mapping to the swizzle outputs, which would help in the case of CaMap
  // build but in the case of transform replay, would need to do the replay from
  // the inputs of the swizzles instead of the outputs, and therefore should not
  // skip swizzles in here.
  auto forward_replay = BestEffortReplay::replayPasC(
      producer,
      consumer,
      consumer_pos,
      logical_map,
      opt.skip_target_swizzle,
      !opt.replay_swizzle,
      !opt.replay_resize);

  // Make a new map based on all the loop ids resulting from best effort replay
  IterDomainMap forwarded_replay_map;
  auto forwarded_replay_loop = forward_replay.getUnorderedLeafIDs();
  for (auto entry : forward_replay.getReplay()) {
    if (forwarded_replay_loop.find(entry.second) !=
        forwarded_replay_loop.end()) {
      forwarded_replay_map[entry.first] = entry.second;
      forwarded_replay_loop.erase(entry.second);
    }
  }

  // Replay producer dimensions.
  ReplayTransformations replay_PasC(target_consumer_ids, forwarded_replay_map);
  replay_PasC.setErrorOnFailure(false)
      .setReplaySwizzle(opt.replay_swizzle)
      .setReplayResize(opt.replay_resize);

  auto producer_loop_ids(replay_PasC.getUnorderedLeafIDs());

  const auto maybe_unmapped_ids = getMaybeUnmappedIDs(
      consumer,
      false,
      logical_map.mapConsumerToProducer(
          consumer->domain(), producer->domain()));

  // Remove all ids from producer_loop_ids that map within the consumer
  // position, we're going to try to further replay the rest of the producer
  // dimensions based on the producers original transformations. Save all dims
  // that mapped to target_consumer_ids.
  std::vector<IterDomain*> dims_mapped2target;
  for (auto c_id : target_consumer_ids) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it == replay_PasC.getReplay().end()) {
      NVF_ERROR(
          maybe_unmapped_ids.count(c_id),
          "Could not find axis, ",
          c_id,
          ", requested in replay.");
      continue;
    }
    NVF_ERROR(
        producer_loop_ids.find(it->second) != producer_loop_ids.end(),
        "Replayed id to match consumer id ",
        c_id,
        " should be a loop in replay map.");
    producer_loop_ids.erase(it->second);
    dims_mapped2target.push_back(it->second);
  }

  // producer_loop_ids now contains all producer ID products that are not used
  // to satisfy the computeAt. Put them in a replay map so we can play forward
  // these IDs in producer (if possible):
  IterDomainMap producer_self_replay_map;
  for (auto entry : producer_loop_ids) {
    producer_self_replay_map[entry.first] = entry.first;
  }

  for (auto entry : forwarded_replay_loop) {
    producer_self_replay_map[entry.first] = entry.first;
  }

  // Check which root domains were used to produce the producer_loop_ids. We may
  // have picked up extra roots in consumer because of broadcast forwarding.
  std::vector<Val*> unordered_non_root_loop_vals;
  for (auto loop_id : replay_PasC.getUnorderedLeafIDs()) {
    if (loop_id.first->definition() == nullptr) {
      continue;
    } else {
      unordered_non_root_loop_vals.emplace_back(loop_id.first);
    }
  }

  auto producer_logical = producer->getLogicalDomain();

  // Figure out all id's that have been processed to generate the
  // unordered_non_root_loop_vals. This needs to be done because we want to
  // match on producer's logical domain, not root domain.
  std::unordered_set<IterDomain*> all_processed_ids;
  {
    auto all_processed_vals_vec = DependencyCheck::getAllValsBetween(
        {producer_logical.begin(), producer_logical.end()},
        unordered_non_root_loop_vals);
    auto all_processed_ids_vec =
        ir_utils::filterByType<IterDomain>(all_processed_vals_vec);
    all_processed_ids.insert(
        all_processed_ids_vec.begin(), all_processed_ids_vec.end());
  }

  // Any root domain that was not used to generate computeIDs we can also put in
  // the map to forward their transformations.
  for (auto producer_logical_id : producer_logical) {
    if (all_processed_ids.find(producer_logical_id) ==
            all_processed_ids.end() &&
        std::find(
            dims_mapped2target.begin(),
            dims_mapped2target.end(),
            producer_logical_id) == dims_mapped2target.end()) {
      producer_self_replay_map[producer_logical_id] = producer_logical_id;
    }
  }

  // Play forward transformations all producer IDs we can
  auto producer_replayed_loop = BestEffortReplay(
      producer->getLoopDomain(),
      producer->getLoopDomain(),
      producer_self_replay_map);

  /*
   * Accumulate axes in to the new domain in the following order, making sure to
   * avoid any duplicates:
   *
   * (1) replay_PasC.getReplay holds mappings from axes in consumer compute at
   * axes -> corresponding generated axes in producer
   *
   * (2) Any axes that were not added, that can be mapped directly from an ID in
   * consumer->domain(). These are axes that were "fully replayed" relative to
   * the consumer, even though it wasn't in the computeAt range.
   *
   * producer_replayed_loop now contain ids that we tried to forward
   * back to what they were in producer. If they couldn't be forwarded they're
   * left in their "most forwarded" form which may be just a remainder of the
   * transformation required to generate the computeAt axes.
   *
   * (3) Axes in producer->domain() that are in producer_replayed_loop
   *
   * (4) Axes not in producer->domain() that are in producer_replayed_loop
   *
   */

  std::vector<IterDomain*> new_IDs;
  std::unordered_set<IterDomain*> used_IDs;
  // Add axes in (1)
  for (auto c_id : target_consumer_ids) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it == replay_PasC.getReplay().end()) {
      NVF_ERROR(
          maybe_unmapped_ids.count(c_id),
          "Could not find axis, ",
          c_id,
          ", requested in replay.");
      continue;
    }
    new_IDs.push_back(it->second);
    used_IDs.emplace(it->second);
  }

  int64_t producer_pos = (int64_t)new_IDs.size();

  // Add axes in (2)
  for (auto c_id : consumer->getLoopDomain()) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it != replay_PasC.getReplay().end()) {
      auto id = it->second;
      // If the loop id from ReplayTransformations is used to move
      // forward in BestEffortReplay, it is not a final ID.
      if (producer_replayed_loop.getUnorderedLeafIDs().find(id) ==
          producer_replayed_loop.getUnorderedLeafIDs().end()) {
        continue;
      }
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (3)
  for (auto id : producer->getLoopDomain()) {
    if (producer_replayed_loop.getUnorderedLeafIDs().find(id) !=
        producer_replayed_loop.getUnorderedLeafIDs().end()) {
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (4)
  for (auto id : producer_replayed_loop.getLeafIDs()) {
    if (used_IDs.find(id) == used_IDs.end()) {
      new_IDs.push_back(id);
    }
  }

  NVF_ERROR(
      !opt.replay_allocation,
      "replayAllocation is not implemented yet for "
      "TransformReplay::replayPasC");

  TensorDomain* replayed = IrBuilder::createInContainer<TensorDomain>(
      producer->container(),
      producer->getRootDomain(),
      producer->getLogicalDomain(),
      producer->getAllocationDomain(),
      new_IDs,
      producer->domain()->contiguity());
  return {replayed, producer_pos};
}

std::pair<TensorDomain*, int64_t> TransformReplay::replayCasP(
    const TensorView* consumer,
    const TensorView* producer,
    int64_t producer_pos,
    const LogicalDomainMap& logical_map,
    TransformReplayOptions opt) {
  FUSER_PERF_SCOPE("TransformReplay::replayCasP");

  // If this is a reduction operation, we may call transform_replay on the same
  // tensor view. When this happens, just return thet target view.
  if (consumer == producer) {
    return {consumer->domain(), consumer->nDims()};
  }
  producer_pos = wrapDim(producer_pos, producer->nDims() + 1);

  // producer ids we need to match in consumer
  std::vector<IterDomain*> target_producer_ids(
      producer->getLoopDomain().begin(),
      producer->getLoopDomain().begin() + producer_pos);
  target_producer_ids = TensorDomain::noReductions(target_producer_ids);

  // Instead of replaying from the root, lets try to forward the history of
  // consumer if they match ops on producer. Enforce if we modify an rfactor
  // axis that those ops match.
  //
  // Note on skip_swizzles: Similar constraints apply in replayPasC. See the
  // corresponding notes there on not skipping swizzles in the
  // matching here.
  //
  // The consumer may have resize, which replayCasP skips and forwards
  // the mapping to the output domain of the resize.
  BestEffortReplay forward_replay = BestEffortReplay::replayCasP(
      consumer,
      producer,
      producer_pos,
      logical_map,
      opt.skip_target_swizzle,
      !opt.replay_swizzle,
      !opt.replay_resize);

  // Track dangling loop ids which can be produced in
  // BestEffortReplay::replayCasP these don't have any equivalent in producer
  // so they're not in the map. We will simply map them to themselves so we
  // don't lose them.
  IterDomainMap forwarded_replay_map;
  auto forwarded_replay_loop = forward_replay.getUnorderedLeafIDs();
  for (auto entry : forward_replay.getReplay()) {
    if (forwarded_replay_loop.find(entry.second) !=
        forwarded_replay_loop.end()) {
      forwarded_replay_map[entry.first] = entry.second;
      forwarded_replay_loop.erase(entry.second);
    }
  }

  // Replay producer dimensions. Currently, resize isn't replayed.
  ReplayTransformations replay_CasP(target_producer_ids, forwarded_replay_map);
  replay_CasP.setErrorOnFailure(false)
      .setReplaySwizzle(opt.replay_swizzle)
      .setReplayResize(opt.replay_resize);

  auto consumer_loop_ids(replay_CasP.getUnorderedLeafIDs());

  const auto maybe_unmapped_ids = getMaybeUnmappedIDs(
      producer,
      true,
      logical_map.mapProducerToConsumer(
          producer->domain(), consumer->domain()));

  // Remove all ids that map to the compute at axis, we're going to replay the
  // rest, track all dims that are needed to match producer CA dims
  std::vector<IterDomain*> dims_mapped2target;
  for (auto p_id : target_producer_ids) {
    auto it = replay_CasP.getReplay().find(p_id);
    if (it == replay_CasP.getReplay().end()) {
      NVF_ERROR(
          maybe_unmapped_ids.count(p_id),
          "Could not find axis, ",
          p_id,
          ", requested in replaying consumer ",
          consumer,
          " as producer ",
          producer);
      continue;
    }
    NVF_ERROR(
        consumer_loop_ids.find(it->second) != consumer_loop_ids.end(),
        "Replayed id to match producer id ",
        p_id,
        " should be a loop in replay map.");
    consumer_loop_ids.erase(it->second);
    dims_mapped2target.push_back(it->second);
  }

  // consumer_loop_ids now contains all consumer ID products that are not used
  // to satisfy the computeAt. Turn into a  map so we can play forward these IDs
  // in consumer (if possible):
  IterDomainMap consumer_self_replay_map;
  for (auto entry : consumer_loop_ids) {
    consumer_self_replay_map[entry.first] = entry.first;
  }

  for (auto entry : forwarded_replay_loop) {
    consumer_self_replay_map[entry.first] = entry.first;
  }

  // Check which root domains were used to produce the consumer_loop_ids. We may
  // have picked up extra roots in consumer because of broadcast forwarding.
  std::vector<Val*> unordered_non_root_loop_vals;
  for (auto loop_id : replay_CasP.getUnorderedLeafIDs()) {
    if (loop_id.first->definition() == nullptr) {
      continue;
    } else {
      unordered_non_root_loop_vals.emplace_back(loop_id.first);
    }
  }

  auto processed_roots = IterVisitor::getInputsTo(unordered_non_root_loop_vals);

  std::vector<IterDomain*> consumer_root = consumer->getMaybeRootDomain();

  // Any root domain that was not used to generate computeIDs we can also put in
  // the map to forward their transformations.
  for (auto consumer_root_id : consumer_root) {
    if (std::find(
            processed_roots.begin(), processed_roots.end(), consumer_root_id) ==
            processed_roots.end() &&
        // Don't re-add roots that may have directly mapped in the replay
        std::find(
            dims_mapped2target.begin(),
            dims_mapped2target.end(),
            consumer_root_id) == dims_mapped2target.end()) {
      consumer_self_replay_map[consumer_root_id] = consumer_root_id;
    }
  }

  // Play forward transformations all consumer IDs we can
  auto consumer_replayed_loop = BestEffortReplay(
      consumer->getLoopDomain(),
      consumer->getLoopDomain(),
      consumer_self_replay_map);

  /*
   * Accumulate axes in to the new domain in the following order, making sure to
   * avoid any duplicates:
   *
   * (1) replay_PasC.getReplay holds mappings from axes in consumer compute at
   * axes -> corresponding generated axes in producer
   *
   * (2) Any axes that were not added, that can be mapped directly from an ID in
   * producer->domain(). These are axes that were "fully replayed" relative to
   * the producer, even though it wasn't in the computeAt range.
   *
   * producer_replayed_loop now contain ids that we tried to forward
   * back to what they were in producer. If they couldn't be forwarded they're
   * left in their "most forwarded" form which may be just a remainder of the
   * transformation required to generate the computeAt axes.
   *
   * (3) Axes in producer->domain() that are in producer_replayed_loop
   *
   * (4) Axes not in producer->domain() that are in producer_replayed_loop
   *
   * TODO: Should (2) and (3) be swapped?
   */

  std::vector<IterDomain*> new_IDs;
  std::unordered_set<IterDomain*> used_IDs;
  // Add axes in (1)
  for (auto p_id : target_producer_ids) {
    auto it = replay_CasP.getReplay().find(p_id);
    if (it == replay_CasP.getReplay().end()) {
      NVF_ERROR(
          maybe_unmapped_ids.count(p_id),
          "Could not find axis, ",
          p_id,
          ", requested in replay.");
      continue;
    }
    new_IDs.push_back(it->second);
    used_IDs.emplace(it->second);
  }

  // pay attention to consumer pos, it should be the actual replayed axis,
  // otherwise further replay may lead to error where the mapped ID is missing.
  // see https://github.com/NVIDIA/Fuser/issues/2593
  int64_t consumer_pos = (int64_t)new_IDs.size();

  // Add axes in (2)
  for (auto p_id : producer->getLoopDomain()) {
    auto it = replay_CasP.getReplay().find(p_id);
    if (it != replay_CasP.getReplay().end()) {
      auto id = it->second;
      // If the loop id from ReplayTransformations is used to move
      // forward in BestEffortReplay, it is not a final ID.
      if (consumer_replayed_loop.getUnorderedLeafIDs().find(id) ==
          consumer_replayed_loop.getUnorderedLeafIDs().end()) {
        continue;
      }
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (3)
  for (auto id : consumer->getLoopDomain()) {
    if (consumer_replayed_loop.getUnorderedLeafIDs().find(id) !=
        consumer_replayed_loop.getUnorderedLeafIDs().end()) {
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (4)
  for (auto id : consumer_replayed_loop.getLeafIDs()) {
    if (used_IDs.find(id) == used_IDs.end()) {
      new_IDs.push_back(id);
    }
  }

  if (!opt.replay_allocation) {
    TensorDomain* replayed = IrBuilder::createInContainer<TensorDomain>(
        consumer->container(),
        consumer->getRootDomain(),
        consumer->getLogicalDomain(),
        consumer->getAllocationDomain(),
        new_IDs,
        consumer->domain()->contiguity());

    return {replayed, consumer_pos};
  }

  NVF_ERROR(
      consumer->definition()->isA<LoadStoreOp>() && !consumer->hasRoot(),
      "TransformReplay::replayCasP currently replays allocation only for Set. "
      "Other ops (e.g. `consumer = broadcast(producer)`) can break. "
      "See https://github.com/NVIDIA/Fuser/pull/1291#discussion_r1391999007 "
      "for details.");

  TensorDomain* replayed = IrBuilder::createInContainer<TensorDomain>(
      consumer->container(),
      consumer->getRootDomain(),
      consumer->getLogicalDomain(),
      /*allocation=*/std::vector<IterDomain*>{},
      /*loop=*/new_IDs,
      consumer->domain()->contiguity());

  if (producer->hasAllocation()) {
    auto replay_CasP = BestEffortReplay(
        new_IDs,
        producer->getLoopDomain(),
        logical_map.mapProducerToConsumer(producer->domain(), replayed));
    const auto& p2c_map = replay_CasP.getReplay();

    auto producer_rank = producer->getAllocationDomain().size();
    std::vector<IterDomain*> new_allocation_domain;
    new_allocation_domain.reserve(producer_rank);
    std::vector<std::optional<bool>> new_contiguity;
    new_contiguity.reserve(producer_rank);

    for (auto i : arange(producer_rank)) {
      IterDomain* alloc_id = producer->getAllocationDomain()[i];
      // We won't find reduction IterDomains in the map. See
      // AllocationDomainTest.CacheBefore.
      if (auto it = p2c_map.find(alloc_id); it != p2c_map.end()) {
        IterDomain* new_alloc_id = it->second;
        new_alloc_id->parallelize(alloc_id->getParallelType());
        new_allocation_domain.push_back(new_alloc_id);
        new_contiguity.push_back(producer->getContiguity()[i]);
      }
    }
    replayed->setAllocationDomain(
        std::move(new_allocation_domain), std::move(new_contiguity));
  }
  return {replayed, consumer_pos};
}

// replay Producer as Consumer
std::pair<TensorDomain*, int64_t> TransformReplay::replayPasC(
    const TensorView* producer,
    const TensorView* consumer,
    int64_t compute_at_axis,
    TransformReplayOptions opt) {
  // Use the pairwise root map as a default mapper
  PairwiseLogicalDomainMap logical_map(producer, consumer);
  // Allow replay through indexing exprs
  logical_map.mapIndexedDomains(true);
  return replayPasC(producer, consumer, compute_at_axis, logical_map, opt);
}

std::pair<TensorDomain*, int64_t> TransformReplay::replayCasP(
    const TensorView* consumer,
    const TensorView* producer,
    int64_t compute_at_axis,
    TransformReplayOptions opt) {
  // Use the pairwise root map as a default mapper
  PairwiseLogicalDomainMap logical_map(producer, consumer);
  // Allow replay through indexing exprs
  logical_map.mapIndexedDomains(true);
  return replayCasP(consumer, producer, compute_at_axis, logical_map, opt);
}

// In a PasC replay, we want the producer to exactly match the consumer:
// all the beginning axes in the producer should be mapped to the consumer in
// the same order. Reductions in the producer needs to be in the back of the
// producer.
int64_t TransformReplay::getMatchedLeafPosWithoutReplayPasC(
    const TensorView* producer,
    const TensorView* consumer,
    int64_t consumer_pos,
    bool skip_resize) {
  FUSER_PERF_SCOPE("transform_replay.cpp::getMatchedLeafPosWithoutReplayPasC");

  // Allow replay through indexing exprs
  const auto pairwise_map =
      PairwiseLogicalDomainMap(producer, consumer).mapIndexedDomains(true);
  IterDomainMap c2p_logical_map = pairwise_map.mapConsumerToProducer();

  // IterDomains in `consumer` root also in `producer` root
  const auto consumer_domain = consumer->getLoopDomain();

  std::unordered_set<Val*> mapped_consumer_roots;
  for (auto entry : c2p_logical_map) {
    mapped_consumer_roots.emplace(entry.first);
  }

  auto unskippable_consumer_ids_vec = DependencyCheck::getAllValsBetween(
      mapped_consumer_roots, {consumer_domain.begin(), consumer_domain.end()});

  std::unordered_set<Val*> unskippable_consumer_ids(
      unskippable_consumer_ids_vec.begin(), unskippable_consumer_ids_vec.end());

  // IterDomains in `producer` root also in `consumer` root
  const auto producer_domain = producer->getLoopDomain();

  auto it_consumer = consumer_domain.begin();
  auto it_producer = producer_domain.begin();

  auto disjoint_sets =
      BestEffortReplay::replayPasC(
          producer, consumer, -1, pairwise_map, true, true, skip_resize)
          .getIterDomainEquivalence();

  int64_t mismatched_consumer_pos = 0;
  int64_t mismatched_producer_pos = 0;
  while (it_consumer != consumer_domain.end()) {
    if (consumer_pos == mismatched_consumer_pos) {
      return mismatched_producer_pos;
    }

    auto consumer_id = *it_consumer;
    if (unskippable_consumer_ids.count(consumer_id) == 0) {
      ++it_consumer;
      ++mismatched_consumer_pos;
      continue;
    }

    if (it_producer == producer_domain.end()) {
      return -1;
    }

    auto producer_id = *it_producer;
    if (disjoint_sets.permissiveAreMapped(producer_id, consumer_id)) {
      ++mismatched_consumer_pos;
      ++mismatched_producer_pos;
      ++it_consumer;
      ++it_producer;
    } else {
      return -1;
    }
  }
  if (consumer_pos == mismatched_consumer_pos) {
    return mismatched_producer_pos;
  }
  return -1;
}

// We want to ignore reductions in the producer in a CasP replay.
int64_t TransformReplay::getMatchedLeafPosWithoutReplayCasP(
    const TensorView* consumer,
    const TensorView* producer,
    int64_t producer_pos,
    bool skip_resize) {
  FUSER_PERF_SCOPE("transform_replay.cpp::getMatchedLeafPosWithoutReplayCasP");

  // Allow replay through indexing exprs
  const auto pairwise_map =
      PairwiseLogicalDomainMap(producer, consumer).mapIndexedDomains(true);
  IterDomainMap p2c_logical_map = pairwise_map.mapProducerToConsumer();

  // IterDomains in `producer` root that are not reduction
  const auto producer_domain = producer->getLoopDomain();
  auto unskippable_producer_ids_vec =
      TensorDomain::noReductions(producer_domain);
  std::unordered_set<IterDomain*> unskippable_producer_ids(
      unskippable_producer_ids_vec.begin(), unskippable_producer_ids_vec.end());

  // IterDomains in `consumer` root also in `producer` root
  const auto consumer_domain = consumer->getLoopDomain();

  std::unordered_set<Val*> mapped_consumer_roots;
  for (auto entry : p2c_logical_map) {
    mapped_consumer_roots.emplace(entry.second);
  }

  auto unskippable_consumer_ids_vec = DependencyCheck::getAllValsBetween(
      mapped_consumer_roots, {consumer_domain.begin(), consumer_domain.end()});

  std::unordered_set<Val*> unskippable_consumer_ids(
      unskippable_consumer_ids_vec.begin(), unskippable_consumer_ids_vec.end());

  auto it_producer = producer_domain.begin();
  auto it_consumer = consumer_domain.begin();

  auto disjoint_sets =
      BestEffortReplay::replayPasC(
          producer, consumer, -1, pairwise_map, true, true, skip_resize)
          .getIterDomainEquivalence();

  int64_t mismatched_producer_pos = 0;
  int64_t mismatched_consumer_pos = 0;
  while (it_producer != producer_domain.end()) {
    if (producer_pos == mismatched_producer_pos) {
      return mismatched_consumer_pos;
    }

    auto producer_id = *it_producer;
    if (unskippable_producer_ids.count(producer_id) == 0) {
      ++it_producer;
      ++mismatched_producer_pos;
      continue;
    }

    if (it_consumer == consumer_domain.end()) {
      return -1;
    }

    auto consumer_id = *it_consumer;
    if (unskippable_consumer_ids.count(consumer_id) == 0) {
      ++it_consumer;
      ++mismatched_consumer_pos;
      continue;
    }

    if (disjoint_sets.permissiveAreMapped(producer_id, consumer_id)) {
      ++mismatched_producer_pos;
      ++mismatched_consumer_pos;
      ++it_producer;
      ++it_consumer;
    } else {
      return -1;
    }
  }
  if (producer_pos == mismatched_producer_pos) {
    return mismatched_consumer_pos;
  }
  return -1;
}

bool TransformReplay::fullSelfMatching(
    const TensorView* replay,
    const TensorView* target) {
  auto replay_root = replay->getMaybeRootDomain();
  auto replay_dom = replay->getLoopDomain();
  auto target_root = target->getMaybeRootDomain();
  auto target_dom = target->getLoopDomain();
  std::unordered_map<IterDomain*, IterDomain*> target2replay_map;
  if (replay_root.size() != target_root.size()) {
    return false;
  }
  target2replay_map.reserve(replay_root.size());
  std::transform(
      target_root.begin(),
      target_root.end(),
      replay_root.begin(),
      std::inserter(target2replay_map, target2replay_map.begin()),
      [](auto a, auto b) { return std::make_pair(a, b); });
  BestEffortReplay replay_(replay_dom, target_dom, target2replay_map);
  auto r = replay_.getReplay();
  for (int64_t i = 0; i < (int64_t)replay_dom.size(); i++) {
    auto target_id = target_dom[i];
    auto replay_it = r.find(target_id);
    if (replay_it == r.end() || replay_it->second != replay_dom[i]) {
      return false;
    }
  }
  return true;
}

namespace {

// Make sure if tv is set to new_td it doesn't violate set compute at and max
// produce at positions.
bool validateDomain(TensorView* tv, TensorDomain* new_td) {
  auto first_mismatch =
      BestEffortReplay::findFirstMismatchedID(tv->domain(), new_td);
  return first_mismatch >= tv->getMaybeMaxProducerPosition() &&
      first_mismatch >= tv->getMaxComputePosition();
}

} // namespace

void TransformPropagator::propagateC2P(TensorView* from, TensorView* to) {
  int64_t pos = replayed_pos_.at(from);
  // Note: [Using multiple TransformPropagators]
  // There are cases that we use multiple TransformPropagators along different
  // spanning trees with different references in the same fusion. Some of these
  // spanning trees could overlap. In cases when there are overlapping nodes,
  // TransformPropagator needs to respect the replay of others, because the
  // current TransformPropagator might not contain the most amount of
  // information on how to do the correct transformation. The logic below tells
  // TransformPropagator to skip the replay when not necessary.
  //
  // Note on resize: When propagating transformations, resize is just
  // skipped, or forwarded, so the matching here is done by skipping it.
  int64_t new_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(to, from, pos, true);
  bool debug_print = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug_print) {
    debug() << "TransformPropagator::propagateC2P" << std::endl;
    debug() << "  from: " << from << " @ " << pos << std::endl;
    debug() << "  to: " << to << std::endl;
  }
  if (new_pos < 0) {
    auto replay = TransformReplay::replayPasC(
        to, from, pos, TransformReplayOptions().skipTargetSwizzle());
    NVF_ERROR(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max "
        "producer position.");
    to->setDomain(replay.first);
    new_pos = replay.second;
    if (debug_print) {
      debug() << "  replayed: " << to << " @ " << new_pos << std::endl;
    }
  } else if (debug_print) {
    debug() << "  replay skipped. result position: " << new_pos << std::endl;
  }
  replayed_pos_[to] = new_pos;
}

void TransformPropagator::propagateP2C(TensorView* from, TensorView* to) {
  int64_t pos = replayed_pos_.at(from);
  // See note [Using multiple TransformPropagators]
  int64_t new_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(to, from, pos, true);
  bool debug_print = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug_print) {
    debug() << "TransformPropagator::propagateP2C" << std::endl;
    debug() << "  from: " << from << " @ " << pos << std::endl;
    debug() << "  to: " << to << std::endl;
  }
  if (new_pos < 0) {
    auto replay = TransformReplay::replayCasP(
        to, from, pos, TransformReplayOptions().skipTargetSwizzle());
    NVF_ERROR(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max "
        "producer position.");
    to->setDomain(replay.first);
    new_pos = replay.second;
    if (debug_print) {
      debug() << "  replayed: " << to << " @ " << new_pos << std::endl;
    }
  } else if (debug_print) {
    debug() << "  replay skipped. result position: " << new_pos << std::endl;
  }
  replayed_pos_[to] = new_pos;
}

void TransformPropagator::propagateSibling(TensorView* from, TensorView* to) {
  int64_t pos = replayed_pos_.at(from);
  // See note [Using multiple TransformPropagators]
  bool debug_print = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug_print) {
    debug() << "TransformPropagator::propagateSibling" << std::endl;
    debug() << "  from: " << from << " @ " << pos << std::endl;
    debug() << "  to: " << to << std::endl;
  }
  if (!TransformReplay::fullSelfMatching(to, from)) {
    auto replay = TransformReplay::fullSelfReplay(to->domain(), from->domain());
    NVF_ERROR(
        validateDomain(to, replay),
        "Tried to set the domain of ",
        to,
        " to ",
        replay,
        " but that would invalidate previously compute at position or max "
        "producer position.");
    to->setDomain(replay);
    if (debug_print) {
      debug() << "  replayed: " << to << " @ " << pos << std::endl;
    }
  } else if (debug_print) {
    debug() << "  replay skipped. result position: " << pos << std::endl;
  }
  replayed_pos_[to] = pos;
}

TransformPropagator::TransformPropagator(TensorView* from, int64_t pos) {
  replayed_pos_[from] = wrapDim(pos, from->nDims() + 1);
}

void MostInlinedTransformPropagator::propagateC2P(
    TensorView* from,
    TensorView* to) {
  int64_t pos = (int64_t)from->nDims();
  // See note [Using multiple TransformPropagators]
  int64_t new_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(to, from, pos, true);
  bool debug_print = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug_print) {
    debug() << "MostInlinedTransformPropagator::propagateC2P" << std::endl;
    debug() << "  from: " << from << std::endl;
    debug() << "  to: " << to << std::endl;
  }
  if (new_pos < 0) {
    auto replay = TransformReplay::replayPasC(
        to, from, pos, TransformReplayOptions().skipTargetSwizzle());
    NVF_ERROR(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max "
        "producer position.");
    to->setDomain(replay.first);
    if (debug_print) {
      debug() << "  replayed: " << to << std::endl;
    }
  } else if (debug_print) {
    debug() << "  replay skipped" << std::endl;
  }
}

void MostInlinedTransformPropagator::propagateP2C(
    TensorView* from,
    TensorView* to) {
  int64_t pos = (int64_t)from->nDims();
  // See note [Using multiple TransformPropagators]
  int64_t new_pos =
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(to, from, pos, true);
  bool debug_print = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug_print) {
    debug() << "MostInlinedTransformPropagator::propagateP2C" << std::endl;
    debug() << "  from: " << from << std::endl;
    debug() << "  to: " << to << std::endl;
  }
  if (new_pos < 0) {
    auto replay = TransformReplay::replayCasP(
        to, from, pos, TransformReplayOptions().skipTargetSwizzle());
    NVF_ERROR(
        validateDomain(to, replay.first),
        "Tried to set the domain of ",
        to,
        " to ",
        replay.first,
        " but that would invalidate previously compute at position or max "
        "producer position.");
    to->setDomain(replay.first);
    if (debug_print) {
      debug() << "  replayed: " << to << std::endl;
    }
  } else if (debug_print) {
    debug() << "  replay skipped" << std::endl;
  }
}

void MostInlinedTransformPropagator::propagateSibling(
    TensorView* from,
    TensorView* to) {
  // See note [Using multiple TransformPropagators]
  bool debug_print = isDebugDumpEnabled(DebugDumpOption::TransformPropagator);
  if (debug_print) {
    debug() << "MostInlinedTransformPropagator::propagateSibling" << std::endl;
    debug() << "  from: " << from << std::endl;
    debug() << "  to: " << to << std::endl;
  }
  if (!TransformReplay::fullSelfMatching(to, from)) {
    auto replay = TransformReplay::fullSelfReplay(to->domain(), from->domain());
    NVF_ERROR(
        validateDomain(to, replay),
        "Tried to set the domain of ",
        to,
        " to ",
        replay,
        " but that would invalidate previously compute at position or max "
        "producer position.");
    to->setDomain(replay);
    if (debug_print) {
      debug() << "  replayed: " << to << std::endl;
    }
  } else if (debug_print) {
    debug() << "  replay skipped" << std::endl;
  }
}

namespace {

// Replays transformations in `old_domain` on `new_root` and returns the new
// TensorDomain that's rooted at `new_root`. This shares quite some code with
// TransformReplay::fullSelfReplay, which can be cleaned up. The main
// challenge for that is that this function uses `ReplayTransformations` and
// `fullSelfReplay` uses `ReplaySelf`, a simplified version of
// `ReplayTransformations` leveraging the fact that it's a self-replay.
TensorDomain* fullReplay(
    const TensorDomain* old_domain,
    const std::vector<IterDomain*>& new_root) {
  std::unordered_map<IterDomain*, IterDomain*> old_root_to_new;
  NVF_CHECK(
      old_domain->maybeRoot().size() == new_root.size(),
      "Unable to replay transformations on a root domain of different size: ",
      old_domain->maybeRoot().size(),
      " vs ",
      new_root.size());
  for (auto i : arange(new_root.size())) {
    old_root_to_new[old_domain->maybeRoot()[i]] = new_root[i];
  }
  NVF_CHECK(
      !old_domain->hasAllocation(),
      "Due to #986, the allocation domain may or may not be between root and "
      "loop. So, when `old_domain` has allocation, it may be incorrect to "
      "use its loop as the target domain: ",
      old_domain->toString(0, /*loop_only=*/false));
  ReplayTransformations replay(old_domain->loop(), old_root_to_new);
  replay.setReplayRFactor(true);

  std::vector<IterDomain*> new_loop;
  new_loop.reserve(old_domain->nDims());
  std::transform(
      old_domain->loop().begin(),
      old_domain->loop().end(),
      std::back_inserter(new_loop),
      [&](IterDomain* old_loop_id) {
        return replay.getReplay().at(old_loop_id);
      });

  if (!old_domain->hasRoot()) {
    return IrBuilder::createInContainer<TensorDomain>(
        old_domain->container(), new_root, new_loop, old_domain->contiguity());
  }

  std::vector<IterDomain*> new_logical;
  new_logical.reserve(old_domain->logical().size());
  std::transform(
      old_domain->logical().begin(),
      old_domain->logical().end(),
      std::back_inserter(new_logical),
      [&](IterDomain* old_logical_id) {
        return replay.getReplay().at(old_logical_id);
      });

  return IrBuilder::createInContainer<TensorDomain>(
      old_domain->container(),
      new_root,
      new_logical,
      new_loop,
      old_domain->contiguity());
}

} // namespace

Expr* replayExprWithNewInput(Expr* e, Val* new_in) {
  auto* new_in_tv = dynamic_cast<TensorView*>(new_in);
  NVF_CHECK(
      new_in_tv != nullptr,
      "This function doesn't support non-TensorView input yet: ",
      new_in);

  std::vector<Val*> new_outs;
  new_outs.reserve(e->outputs().size());

  for (Val* old_out : e->outputs()) {
    auto* old_out_tv = dynamic_cast<TensorView*>(old_out);
    NVF_CHECK(
        old_out_tv != nullptr,
        "This function doesn't support non-TensorView outputs yet: ",
        old_out);
    TensorDomain* old_domain = old_out_tv->domain();

    std::vector<IterDomain*> new_out_root;
    new_out_root.reserve(old_domain->maybeRoot().size());
    int64_t i = 0;
    for (IterDomain* in_logical_id :
         TensorDomain::noReductions(new_in_tv->getLogicalDomain())) {
      // Copy the `rf` flag from `old_domain` and everything else from
      // `in_logical_id`.
      new_out_root.push_back(
          IterDomainBuilder(in_logical_id)
              .is_rfactor_domain(old_domain->maybeRoot()[i]->isRFactorProduct())
              .build());
      i++;
    }
    TensorDomain* new_domain = fullReplay(old_domain, new_out_root);
    TensorView* new_out_tv =
        IrBuilder::create<TensorView>(new_domain, *old_out->getDataType());
    new_outs.push_back(new_out_tv);
  }

  return e->newObjectFunc()(
      e->container(), {new_in_tv}, new_outs, e->attributes());
}

} // namespace nvfuser
