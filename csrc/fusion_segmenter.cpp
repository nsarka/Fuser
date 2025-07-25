// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <algorithm>
#include <sstream>

#include <debug.h>
#include <device_lower/utils.h>
#include <disjoint_set.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/cloner.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <options.h>
#include <scheduler/debug_utils.h>
#include <scheduler/normalization_utils.h>
#include <transform_iter.h>
#include <transform_replay.h>

namespace nvfuser {

namespace {

using GroupSet = VectorOfUniqueEntries<SegmentedGroup*>;

// This helper function converts selected keys to their corresponding values.
// During serialization, we map pointers to an integer. For deserialization, we
// reverse the mapping from integers to pointers.
template <typename K, typename V, typename ContainerV, typename ContainerK>
std::vector<V> convertContainer(
    const ContainerV& all_values,
    const ContainerK& selected_keys) {
  std::vector<V> result;
  result.reserve(selected_keys.size());
  std::transform(
      selected_keys.begin(),
      selected_keys.end(),
      std::back_inserter(result),
      [&](K key) { return all_values.at(key); });
  return result;
}

} // namespace

flatbuffers::Offset<serde::SegmentedGroup> SegmentedGroup::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const std::unordered_map<Val*, int64_t>& vals_to_id_map,
    const std::unordered_map<Expr*, int64_t>& exprs_to_id_map,
    const std::unordered_map<SegmentedGroup*, int64_t>& groups_map,
    const std::unordered_map<SegmentedEdge*, int64_t>& edges_map) const {
  FUSER_PERF_SCOPE("SegmentedGroup::serialize");
  std::vector<int64_t> producer_edges_fb =
      convertContainer<SegmentedEdge*, int64_t>(edges_map, producer_edges);

  std::vector<int64_t> consumer_edges_fb =
      convertContainer<SegmentedEdge*, int64_t>(edges_map, consumer_edges);

  std::vector<int64_t> input_vals_fb =
      convertContainer<Val*, int64_t>(vals_to_id_map, input_vals_.vector());

  std::vector<int64_t> output_vals_fb =
      convertContainer<Val*, int64_t>(vals_to_id_map, output_vals_.vector());

  std::vector<int64_t> exprs_fb =
      convertContainer<Expr*, int64_t>(exprs_to_id_map, exprs_);

  // -1 corresponds with a nullptr value
  int64_t merge_with_segmented_group = -1;
  if (merge_with_ != nullptr) {
    merge_with_segmented_group = groups_map.at(merge_with_);
  }

  // -1 corresponds with a nullptr value
  int64_t merge_through_segmented_edge = -1;
  if (merge_through_ != nullptr) {
    merge_through_segmented_edge = edges_map.at(merge_through_);
  }

  return serde::CreateSegmentedGroupDirect(
      builder,
      &producer_edges_fb,
      &consumer_edges_fb,
      &input_vals_fb,
      &output_vals_fb,
      group_id_,
      toUnderlying(scheduler_type_),
      &exprs_fb,
      level_,
      merge_with_segmented_group,
      merge_through_segmented_edge,
      merged_);
}

void SegmentedGroup::deserialize(
    const serde::SegmentedGroup* buffer,
    const std::deque<Val*>& vals,
    const std::deque<Expr*>& exprs,
    const std::vector<SegmentedGroup*>& groups,
    const std::vector<SegmentedEdge*>& edges) {
  FUSER_PERF_SCOPE("SegmentedGroup::deserialize");
  NVF_ERROR(buffer != nullptr, "serde::SegmentedGroup is nullptr.");

  producer_edges = convertContainer<int64_t, SegmentedEdge*>(
      edges, *buffer->producer_edges());

  consumer_edges = convertContainer<int64_t, SegmentedEdge*>(
      edges, *buffer->consumer_edges());

  input_vals_ = convertContainer<int64_t, Val*>(vals, *buffer->input_vals());

  output_vals_ = convertContainer<int64_t, Val*>(vals, *buffer->output_vals());

  group_id_ = buffer->group_id();

  scheduler_type_ = static_cast<SchedulerType>(buffer->heuristic());

  exprs_ = convertContainer<int64_t, Expr*>(exprs, *buffer->exprs());

  level_ = buffer->level();

  // -1 corresponds with a nullptr value
  if (buffer->merge_with_segmented_group() != -1) {
    merge_with_ = groups.at(buffer->merge_with_segmented_group());
  }

  // -1 corresponds with a nullptr value
  if (buffer->merge_through_segmented_edge() != -1) {
    merge_through_ = edges.at(buffer->merge_through_segmented_edge());
  }

  merged_ = buffer->merged();
}

void SegmentedGroup::makeClonedFusion() {
  auto&& [ir_cloner, fusion_segment] = segmented_fusion_->makeFusion(this);
  NVF_ERROR(fusion_segment != nullptr, "Failed to create segmented fusion.");

  cloned_fusion_ = std::move(fusion_segment);

  // Map inputs for original fusion to the segmented fusion through IrCloner
  const std::vector<Val*>& complete_inputs =
      segmented_fusion_->completeFusion()->inputs();
  original_inputs_in_cloned_fusion_.reserve(complete_inputs.size());
  std::transform(
      complete_inputs.begin(),
      complete_inputs.end(),
      std::back_inserter(original_inputs_in_cloned_fusion_),
      [&complete_to_segment_map = ir_cloner](Val* v) {
        return complete_to_segment_map.clone(v);
      });
}

std::vector<SegmentedGroup::NeighborGroup> SegmentedGroup::getNeighborGroups() {
  std::vector<NeighborGroup> neighbors;
  for (auto inp : producer_edges) {
    if (inp->val->isFusionOutput() || inp->from->exprs_.empty()) {
      // Don't fuse across output nodes, would need to find another
      // path. See the comment in finalMerge.
      // Also, a fusion input group doesn't have any expr and should
      // not be merged
      continue;
    }
    neighbors.emplace_back(inp->from, inp);
  }
  for (auto out : consumer_edges) {
    if (out->val->isFusionOutput()) {
      // Don't fuse across output nodes, would need to find another
      // path. See the comment in finalMerge
      continue;
    }
    neighbors.emplace_back(out->to, out);
  }
  return neighbors;
}

std::vector<SegmentedGroup*> SegmentedGroup::getNeighbors() {
  std::vector<SegmentedGroup*> neighbors;
  auto neighbors_pair = getNeighborGroups();

  std::transform(
      neighbors_pair.begin(),
      neighbors_pair.end(),
      std::back_inserter(neighbors),
      [](auto& neighbor_group) { return neighbor_group.group; });
  return neighbors;
}

std::vector<SegmentedGroup::NeighborGroup> SegmentedGroup::
    getMergeCandidates() {
  // Don't look for candidates if already merged. Input groups should
  // be also ignored from the merge process
  if (merged_ || exprs_.empty()) {
    return {};
  }

  std::vector<NeighborGroup> neighbors = getNeighborGroups();

  // Can this node be merged with another? Check if neighbors are merged, if
  // so and merged neighbor is within 1 level or node merged with neighbor is
  // within 1 level, can't merge this node with anything else.
  bool can_merge_this = true;
  for (auto& neighbor : neighbors) {
    if (!neighbor.group->merged_) {
      continue;
    }
    if (std::abs(neighbor.group->level_ - level_) <= 1) {
      can_merge_this = false;
    }
    if (std::abs(neighbor.group->merge_with_->level_ - level_) <= 1) {
      can_merge_this = false;
    }
  }
  if (!can_merge_this) {
    return {};
  }

  std::vector<bool> can_merge(neighbors.size(), true);

  // Find neighbors with a level that is only 1 differant than this groups level
  for (const auto i : arange(neighbors.size())) {
    if (std::abs(neighbors[i].group->level_ - level_) > 1) {
      can_merge[i] = false;
    }
  }

  // Check neighbor of neighbors we're considering, if any of them are merged
  // with another node, make sure the resulting edge wouldn't have a level
  // difference of 1
  for (const auto i : arange(neighbors.size())) {
    if (!can_merge[i]) {
      continue;
    }

    for (auto neighbor_neighbor : neighbors[i].group->getNeighbors()) {
      // Don't check self
      if (neighbor_neighbor == neighbors[i].group) {
        continue;
      }
      if (neighbor_neighbor->merged_) {
        // check neighbor_neighbor level
        if (std::abs(neighbor_neighbor->level_ - level_) <= 1) {
          can_merge[i] = false;
        }
        if (std::abs(neighbor_neighbor->level_ - neighbors[i].group->level_) <=
            1) {
          can_merge[i] = false;
        }

        // check neighbor_neighber->merged_->level_
        if (std::abs(neighbor_neighbor->merge_with_->level_ - level_) <= 1) {
          can_merge[i] = false;
        }
        if (std::abs(
                neighbor_neighbor->merge_with_->level_ -
                neighbors[i].group->level_) <= 1) {
          can_merge[i] = false;
        }
      }
    }
  }

  std::vector<NeighborGroup> merge_candidates;
  for (const auto i : arange(neighbors.size())) {
    if (can_merge[i]) {
      merge_candidates.push_back(neighbors[i]);
    }
  }
  return merge_candidates;
}

// TODO: Reevaluate what's being done in finalize
void SegmentedGroup::finalize() {
  // Make sure all inputs and outputs of the group are now in input and output
  // vals respectively as they will be used to figure out ordering of groups for
  // the runtime

  for (auto producer_edge : producer_edges) {
    if (!producer_edge->val->isFusionInput()) {
      input_vals_.pushBack(producer_edge->val);
    }
  }

  std::unordered_set<Val*> input_set(input_vals_.begin(), input_vals_.end());

  for (auto i : input_vals_) {
    if (auto tv = dynamic_cast<TensorView*>(i)) {
      // We do not need to add scalars which are the extents of already-added
      // input TensorViews
      for (auto id : TensorDomain::noReductions(tv->getLogicalDomain())) {
        input_set.insert(id->getMaybeExpandedExtent());
      }
    }
  }

  for (auto expr : exprs_) {
    for (auto i : expr->inputs()) {
      if (i->isIntegralScalar() && i->definition() == nullptr &&
          !i->isConstScalar() && !i->isFusionInput() && !input_set.count(i) &&
          !(i->isA<NamedScalar>() &&
            (i->as<NamedScalar>()->getParallelDim() ||
             i->as<NamedScalar>()->getParallelIndex()))) {
        input_set.insert(i);
        input_vals_.pushBack(i);
      }
    }
  }

  // Outputs
  for (auto consumer_edge : consumer_edges) {
    if (!consumer_edge->val->isFusionOutput()) {
      output_vals_.pushBack(consumer_edge->val);
    }
  }

  // alias aware segmentation. we add inputs that are aliased by output
  // generated in this SegmentedGroup
  for (Val* output : output_vals_) {
    if (Val* aliased_input = segmented_fusion_->completeFusion()
                                 ->getOutputAlias(output)
                                 .aliased_io) {
      // aliasing currently only supported as output to input
      NVF_ERROR(
          aliased_input->isFusionInput(),
          "Aliased input ",
          aliased_input->toString(),
          " is not found in the complete fusion.");
      if (!input_set.count(aliased_input)) {
        input_set.insert(aliased_input);
        input_vals_.pushBack(aliased_input);
      }
    }
  }
}

std::ostream& operator<<(std::ostream& os, const SegmentedGroup* group) {
  os << toString(group->schedulerType()) << "{";
  auto expr_to_print = group->exprs();
  std::sort(
      expr_to_print.begin(),
      expr_to_print.end(),
      [](auto expr_a, auto expr_b) -> bool {
        return expr_a->name() < expr_b->name();
      });
  for (const auto i : arange(expr_to_print.size())) {
    os << expr_to_print[i]->name();
    if (i + 1 != expr_to_print.size()) {
      os << ", ";
    }
  }
  os << "}";
  if (group->isMerged()) {
    os << " (merged)";
  }
  return os;
}

void SegmentedGroup::print() const {
  debug() << this << std::endl;
}

std::string toString(const SegmentedGroup* group) {
  std::stringstream ss;
  ss << group;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const SegmentedEdge* edge) {
  os << "e{ " << edge->from << " -> " << edge->to << "("
     << edge->val->toString() << ") }";
  return os;
}

void SegmentedEdge::print() const {
  debug() << this << std::endl;
}

std::string toString(const SegmentedEdge* edge) {
  std::stringstream ss;
  ss << edge;
  return ss.str();
}

std::unique_ptr<SegmentedFusion> SegmentedFusion::fromCompleteFusion(
    std::unique_ptr<Fusion> fusion_ptr,
    SchedulerType scheduler_type,
    const KernelArgumentHolder& runtime_inputs) {
  auto fusion = fusion_ptr.get();
  NVF_ERROR(
      !SegmentCandidateFinder::hasSegmentHints(fusion),
      "SegmentedFusion::fromCompleteFusion cannot be called on a fusion with "
      "segment hints!");

  // convert Welford to two-pass if option is enabled and the original heuristic
  // is persistent
  auto isPersistentScheduler = [&scheduler_type]() {
    return scheduler_type == SchedulerType::InnerPersistent ||
        scheduler_type == SchedulerType::OuterPersistent ||
        scheduler_type == SchedulerType::InnerOuterPersistent;
  };
  SegmentCandidateFinderOptions scfo;
  if (scfo.run_translate_welford && isPersistentScheduler()) {
    SegmentCandidateFinder::translateWelfordInFusion(fusion, runtime_inputs);
  }

  auto segmented_fusion_ptr =
      std::make_unique<SegmentedFusion>(std::move(fusion_ptr));

  // Make a group for the single fusion
  auto single_group = segmented_fusion_ptr->newGroup();

  // Add input and output vals
  single_group->input_vals_.pushBack(fusion->inputs());
  single_group->output_vals_.pushBack(fusion->outputs());

  // Get ordered expression list
  single_group->resetExprList();

  // Assign heuristics and id for the complete fusion
  //  to share the runtime path of segmented fusion.
  single_group->setSchedulerType(scheduler_type);
  single_group->setID(0);

  // Used to log the number of values and expressions in the fusion for
  // serialization sanity check.
  segmented_fusion_ptr->finalize();
  return segmented_fusion_ptr;
}

SegmentedFusion::SegmentedFusion(std::unique_ptr<Fusion> fusion)
    : segmented_fusion_name_{segmentedFusionName()},
      impl_(this),
      complete_fusion_(std::move(fusion)),
      initial_vals_size_{complete_fusion_->vals().size()},
      initial_exprs_size_{complete_fusion_->unordered_exprs().size()} {
  annotateFP16IntermediateTensors();
}

namespace {

//! A SegmentedGroup is serializable if all its values and expressions are
//! compatible with the statements in the complete fusion provided in the
//! SegmentedFusion constructor.
bool isSerializableSegmentedGroup(
    SegmentedGroup* sg,
    const std::unordered_map<Val*, int64_t>& vals_to_id_map,
    const std::unordered_map<Expr*, int64_t>& exprs_to_id_map,
    int64_t initial_vals_size,
    int64_t initial_exprs_size) {
  auto check_value = [&](Val* v) {
    return vals_to_id_map.at(v) < initial_vals_size;
  };
  auto check_expr = [&](Expr* e) {
    return exprs_to_id_map.at(e) < initial_exprs_size;
  };
  bool all_serializable_inputs =
      std::all_of(sg->inputs().begin(), sg->inputs().end(), check_value);
  bool all_serializable_outputs =
      std::all_of(sg->outputs().begin(), sg->outputs().end(), check_value);
  bool all_serializable_exprs =
      std::all_of(sg->exprs().begin(), sg->exprs().end(), check_expr);
  return (
      all_serializable_inputs && all_serializable_outputs &&
      all_serializable_exprs);
}

} // namespace

flatbuffers::Offset<serde::SegmentedFusion> SegmentedFusion::serialize(
    flatbuffers::FlatBufferBuilder& builder) const {
  FUSER_PERF_SCOPE("SegmentedFusion::serialize");
  const std::unordered_map<Val*, int64_t>& vals_to_id_map =
      completeFusion()->deterministic_vals_map();
  const std::unordered_map<Expr*, int64_t>& exprs_to_id_map =
      completeFusion()->deterministic_exprs_map();
  const std::unordered_map<SegmentedGroup*, int64_t>& groups_map =
      impl_.groups_map();
  const std::unordered_map<SegmentedEdge*, int64_t>& edges_map =
      impl_.edges_map();

  bool all_edges_serializable =
      std::all_of(edges_.begin(), edges_.end(), [&](SegmentedEdge* se) {
        return vals_to_id_map.at(se->val) < (int64_t)initial_vals_size_;
      });

  bool all_groups_serializable =
      std::all_of(groups_.begin(), groups_.end(), [&](SegmentedGroup* sg) {
        return isSerializableSegmentedGroup(
            sg,
            vals_to_id_map,
            exprs_to_id_map,
            (int64_t)initial_vals_size_,
            (int64_t)initial_exprs_size_);
      });

  // SegmentCandidateFinder::findSegments can generate new statements when
  // finding valid sub-fusions, so SegmentedGroup can reference statements that
  // do not exist in the original fusion. If we cannot get all statements from
  // the original fusion, we cannot serialize the segmented fusion.
  if (!all_edges_serializable || !all_groups_serializable) {
    return serde::CreateSegmentedFusionDirect(
        builder,
        /*valid=*/false);
  }

  std::vector<flatbuffers::Offset<serde::SegmentedEdge>> edges_fb;
  edges_fb.reserve(edges_.size());
  for (SegmentedEdge* se : edges_) {
    edges_fb.push_back(serialize(builder, se, vals_to_id_map, groups_map));
  }

  std::vector<flatbuffers::Offset<serde::SegmentedGroup>> groups_fb;
  groups_fb.reserve(groups_.size());
  for (SegmentedGroup* sg : groups_) {
    groups_fb.push_back(sg->serialize(
        builder, vals_to_id_map, exprs_to_id_map, groups_map, edges_map));
  }

  std::vector<int64_t> force_fp16_tv_fb;
  force_fp16_tv_fb.reserve(force_fp16_tv_set_.size());
  for (auto tv : force_fp16_tv_set_) {
    force_fp16_tv_fb.push_back(vals_to_id_map.at(tv));
  }

  return serde::CreateSegmentedFusionDirect(
      builder,
      /*valid=*/true,
      segmented_fusion_name_,
      initial_vals_size_,
      initial_exprs_size_,
      &edges_fb,
      &groups_fb,
      &force_fp16_tv_fb,
      toUnderlying(std::get<PrimDataType>(force_half_precision_type_.type)));
}

void SegmentedFusion::deserialize(const serde::SegmentedFusion* buffer) {
  FUSER_PERF_SCOPE("SegmentedFusion::deserialize");
  NVF_ERROR(buffer != nullptr, "serde::SegmentedFusion is nullptr.");

  // NOTE Schedule::proposeHeuristics can add values and expressions to
  // the fusion. We relax the constraints here because we already know the
  // proposed scheduler for each segmented group.
  NVF_ERROR(
      complete_fusion_->vals().size() <= buffer->num_vals(),
      "The complete fusion has ",
      complete_fusion_->vals().size(),
      " values while serialization expected at least",
      buffer->num_vals(),
      " values.");
  NVF_ERROR(
      complete_fusion_->unordered_exprs().size() <= buffer->num_exprs(),
      "The complete fusion has ",
      complete_fusion_->unordered_exprs().size(),
      " expressions while serialization expected at least",
      buffer->num_exprs(),
      " expressions.");
  const std::deque<Val*>& vals = complete_fusion_->deterministic_vals();
  const std::deque<Expr*>& exprs = complete_fusion_->deterministic_exprs();
  segmented_fusion_name_ = buffer->segmented_fusion_name();

  // Construct segmented groups first because they are necessary for the
  // segmented edge's constructor
  // NOTE: Use regular for-loop to avoid unused variable 'idx' error
  for (size_t idx = 0; idx < buffer->groups()->size(); ++idx) {
    newGroup();
  }

  // Create segmented edges
  for (auto idx : arange(buffer->edges()->size())) {
    auto se_fb = buffer->edges()->Get(idx);
    newEdge(
        groups_.at(se_fb->from_segmented_group()),
        groups_.at(se_fb->to_segmented_group()),
        vals.at(se_fb->val()));
  }

  // Deserialize segmented groups
  for (auto idx : arange(buffer->groups()->size())) {
    auto sg_fb = buffer->groups()->Get(idx);
    groups_.at(idx)->deserialize(sg_fb, vals, exprs, groups_, edges_);
  }

  for (auto idx : *buffer->force_fp16_tv_set()) {
    auto val = vals.at(idx);
    NVF_CHECK(
        val->isA<TensorView>(),
        "Segmented Fusion Deserialization: Expected Val to be a TensorView.");
    force_fp16_tv_set_.emplace(val->as<TensorView>());
  }

  force_half_precision_type_ =
      DataType(static_cast<PrimDataType>(buffer->force_half_precision_type()));

  finalize();
}

flatbuffers::Offset<serde::SegmentedEdge> SegmentedFusion::serialize(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::SegmentedEdge* edge,
    const std::unordered_map<Val*, int64_t>& vals_to_id_map,
    const std::unordered_map<SegmentedGroup*, int64_t>& groups_map) const {
  FUSER_PERF_SCOPE("SegmentedEdge::serialize");
  return serde::CreateSegmentedEdge(
      builder,
      groups_map.at(edge->from),
      groups_map.at(edge->to),
      vals_to_id_map.at(edge->val));
}

nvfuser::SegmentedEdge SegmentedFusion::deserialize(
    const serde::SegmentedEdge* buffer,
    const std::deque<Val*>& vals) {
  FUSER_PERF_SCOPE("SegmentedEdge::deserialize");
  NVF_ERROR(buffer != nullptr, "serde::SegmentedEdge is nullptr.");
  NVF_ERROR(
      !groups_.empty(),
      "Expected SegmentedGroup to be populated before deserializing "
      "SegmentedEdge.");
  return {
      groups_.at(buffer->from_segmented_group()),
      groups_.at(buffer->to_segmented_group()),
      vals.at(buffer->val())};
}

SegmentedGroup* SegmentedFusion::Impl::makeGroup() {
  groups_.emplace_back(std::make_unique<SegmentedGroup>(owning_fusion_));
  return groups_.back().get();
}

SegmentedGroup* SegmentedFusion::Impl::makeGroup(Expr* expr) {
  groups_.emplace_back(std::make_unique<SegmentedGroup>(expr, owning_fusion_));
  return groups_.back().get();
}

SegmentedEdge* SegmentedFusion::Impl::makeEdge(
    SegmentedGroup* from,
    SegmentedGroup* to,
    Val* val) {
  edges_.emplace_back(std::make_unique<SegmentedEdge>(from, to, val));
  return edges_.back().get();
}

void SegmentedFusion::removeEdge(SegmentedEdge* edge) {
  NVF_ERROR(edge != nullptr, "Edge is nullptr");
  // Validate edge exists in all expected locations
  SegmentedGroup* producer = edge->from;
  SegmentedGroup* consumer = edge->to;
  auto& producer_consumer_edges = producer->consumer_edges;
  auto& consumer_producer_edges = consumer->producer_edges;

  // Remove edge from producer's consumer edges
  auto producer_edge_it = std::find(
      producer_consumer_edges.begin(), producer_consumer_edges.end(), edge);
  NVF_ERROR(
      producer_edge_it != producer_consumer_edges.end(),
      "Edge not found in producer's consumer edges");
  producer_consumer_edges.erase(producer_edge_it);

  // Remove edge from consumer's producer edges
  auto consumer_edge_it = std::find(
      consumer_producer_edges.begin(), consumer_producer_edges.end(), edge);
  NVF_ERROR(
      consumer_edge_it != consumer_producer_edges.end(),
      "Edge not found in consumer's producer edges");
  consumer_producer_edges.erase(consumer_edge_it);

  // Remove edge from global edge list
  auto edge_it = std::find(edges_.begin(), edges_.end(), edge);
  NVF_ERROR(edge_it != edges_.end(), "Edge not found in global edge list");
  edges_.erase(edge_it);
}

void SegmentedFusion::Impl::cleanUnused() {
  std::unordered_set<SegmentedGroup*> g_used(
      owning_fusion_->groups().begin(), owning_fusion_->groups().end());
  std::unordered_set<SegmentedEdge*> e_used(
      owning_fusion_->edges().begin(), owning_fusion_->edges().end());

  // Remove any edges that are no longer in use
  edges_.erase(
      std::remove_if(
          edges_.begin(),
          edges_.end(),
          [&e_used](auto& e) { return e_used.count(e.get()) == 0; }),
      edges_.end());

  // Remove any groups that are no longer in use
  groups_.erase(
      std::remove_if(
          groups_.begin(),
          groups_.end(),
          [&g_used](auto& g) { return g_used.count(g.get()) == 0; }),
      groups_.end());
}

//! Return mapping from SegmentedGroup to integer id
std::unordered_map<SegmentedGroup*, int64_t> SegmentedFusion::Impl::groups_map()
    const {
  std::unordered_map<SegmentedGroup*, int64_t> group_map;
  int64_t count = 0;
  std::transform(
      groups_.begin(),
      groups_.end(),
      std::inserter(group_map, group_map.end()),
      [&count](const std::unique_ptr<SegmentedGroup>& group_up) {
        return std::make_pair(group_up.get(), count++);
      });
  return group_map;
}

//! Return mapping from SegmentedEdge to integer id
std::unordered_map<SegmentedEdge*, int64_t> SegmentedFusion::Impl::edges_map()
    const {
  std::unordered_map<SegmentedEdge*, int64_t> edge_map;
  int64_t count = 0;
  std::transform(
      edges_.begin(),
      edges_.end(),
      std::inserter(edge_map, edge_map.end()),
      [&count](const std::unique_ptr<SegmentedEdge>& edge_up) {
        return std::make_pair(edge_up.get(), count++);
      });
  return edge_map;
}

SegmentedGroup* SegmentedFusion::newGroup() {
  SegmentedGroup* g = impl_.makeGroup();
  groups_.push_back(g);
  return g;
}

SegmentedGroup* SegmentedFusion::newGroup(Expr* expr) {
  SegmentedGroup* g = impl_.makeGroup(expr);
  groups_.push_back(g);
  return g;
}

SegmentedEdge* SegmentedFusion::newEdge(
    SegmentedGroup* from,
    SegmentedGroup* to,
    Val* val) {
  SegmentedEdge* e = impl_.makeEdge(from, to, val);
  edges_.push_back(e);
  return e;
}

void SegmentedFusion::draw() {
  size_t group_index = 0;
  std::unordered_map<const Expr*, size_t> expr_color_map;

  for (auto group : groups()) {
    for (auto expr : group->exprs()) {
      if (ir_utils::isTvOp(expr)) {
        expr_color_map[expr] = group_index;
      }
    }
    group_index++;
  }

  std::stringstream sstream;
  sstream << "segmented_fusion" << segmented_fusion_name_ << ".dot";
  auto filename = sstream.str();

  IrGraphGenerator::print(
      completeFusion(),
      filename.c_str(),
      IrGraphGenerator::DetailLevel::ComputeOnly,
      &expr_color_map);
}

namespace {

// Concat's producer edges of sg1 and sg2, but removes any edges
// from/to sg1/sg2. If dedup is true, incoming edges with the same val
// are considered duplicate, and only one of them is returned
std::vector<SegmentedEdge*> getMergedProducerEdges(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2,
    bool dedup = true) {
  // At least either of sg1 or sg2 must not be null
  NVF_ERROR(sg1 != nullptr || sg2 != nullptr);
  // If either is null, just return the edges of the other group
  if (sg1 == nullptr) {
    return sg2->producer_edges;
  } else if (sg2 == nullptr) {
    return sg1->producer_edges;
  }

  auto producer_edges = sg1->producer_edges;

  producer_edges.insert(
      producer_edges.end(),
      sg2->producer_edges.begin(),
      sg2->producer_edges.end());

  // Register producers into sg2
  std::unordered_set<Val*> sg2_vals;
  for (auto se : sg2->producer_edges) {
    sg2_vals.emplace(se->val);
  }

  producer_edges.erase(
      std::remove_if(
          producer_edges.begin(),
          producer_edges.end(),
          [&sg1, &sg2, &sg2_vals, dedup](SegmentedEdge* se) {
            // remove edges in between the groups and common uses
            return (se->to == sg1 && se->from == sg2) ||
                (se->to == sg2 && se->from == sg1) ||
                (dedup && (se->to == sg1 && sg2_vals.count(se->val)));
          }),
      producer_edges.end());

  // Remove Duplicate Edges

  return producer_edges;
}

// Concat's consumer edges of sg1 and sg2, but removes any edges from/to sg1/sg2
std::vector<SegmentedEdge*> getMergedConsumerEdges(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2) {
  // At least either of sg1 or sg2 must not be null
  NVF_ERROR(sg1 != nullptr || sg2 != nullptr);
  // If either is null, just return the edges of the other group
  if (sg1 == nullptr) {
    return sg2->consumer_edges;
  } else if (sg2 == nullptr) {
    return sg1->consumer_edges;
  }

  auto consumer_edges = sg1->consumer_edges;
  consumer_edges.insert(
      consumer_edges.end(),
      sg2->consumer_edges.begin(),
      sg2->consumer_edges.end());

  consumer_edges.erase(
      std::remove_if(
          consumer_edges.begin(),
          consumer_edges.end(),
          [&sg1, &sg2](SegmentedEdge* se) {
            return (se->to == sg1 && se->from == sg2) ||
                (se->to == sg2 && se->from == sg1);
          }),
      consumer_edges.end());

  return consumer_edges;
}

// Returns a determinstic, unique set of inputs of the segment group, sg1, or
// the combined group sg1 + sg2
std::vector<Val*> getAllInputs(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2 = nullptr) {
  std::vector<SegmentedEdge*> merged_producer_edges;

  if (sg1 != nullptr && sg2 != nullptr) {
    merged_producer_edges = getMergedProducerEdges(sg1, sg2);
  } else if (sg1 != nullptr) {
    merged_producer_edges = sg1->producer_edges;
  } else if (sg2 != nullptr) {
    merged_producer_edges = sg2->producer_edges;
  }

  VectorOfUniqueEntries<Val*> producer_edge_vals;

  for (auto edge : merged_producer_edges) {
    producer_edge_vals.pushBack(edge->val);
  }

  VectorOfUniqueEntries<Val*> return_vals;
  if (sg1 != nullptr) {
    return_vals.pushBack(sg1->input_vals_);
  }
  if (sg2 != nullptr) {
    return_vals.pushBack(sg2->input_vals_);
  }
  return_vals.pushBack(producer_edge_vals);
  return return_vals.vector();
}

// Returns a determinstic, unique set of outputs of the segment group, sg1, or
// the combined group sg1 + sg2
std::vector<Val*> getAllOutputs(
    const SegmentedGroup* sg1,
    const SegmentedGroup* sg2 = nullptr) {
  std::vector<SegmentedEdge*> merged_consumer_edges;

  if (sg1 != nullptr && sg2 != nullptr) {
    merged_consumer_edges = getMergedConsumerEdges(sg1, sg2);
  } else if (sg1 != nullptr) {
    merged_consumer_edges = sg1->consumer_edges;
  } else if (sg2 != nullptr) {
    merged_consumer_edges = sg2->consumer_edges;
  }

  VectorOfUniqueEntries<Val*> consumer_edge_vals;
  for (auto edge : merged_consumer_edges) {
    consumer_edge_vals.pushBack(edge->val);
  }

  VectorOfUniqueEntries<Val*> return_vals;
  if (sg1 != nullptr) {
    return_vals.pushBack(sg1->output_vals_);
  }
  if (sg2 != nullptr) {
    return_vals.pushBack(sg2->output_vals_);
  }
  return_vals.pushBack(consumer_edge_vals);
  return return_vals.vector();
}

// Set version of getting merged input or output if segmented_groups were
//  merged
//  outputs respects order in segmented_groups for deterministic
//  merge trace
//  will get input if get_inputs otherwise will get ouputs
//  TODO: merge with the binary counter parts
std::vector<Val*> allInputsIfTrueElseOutputs(
    const std::vector<SegmentedGroup*>& segmented_groups,
    bool get_inputs = true) {
  // Get producer edges to get inputs, consumer edges to get outputs
  auto edges_to_process_from_or_to_group =
      [get_inputs](SegmentedGroup* group) -> std::vector<SegmentedEdge*>& {
    return get_inputs ? group->producer_edges : group->consumer_edges;
  };

  // Get the group that is connected to current group
  auto global_vals_from_or_to_group =
      [get_inputs](SegmentedGroup* group) -> const std::vector<Val*>& {
    return get_inputs ? group->input_vals_.vector()
                      : group->output_vals_.vector();
  };

  // Get the group that is connected to current group by given edge
  auto opposite_end_of_edge = [get_inputs](SegmentedEdge* edge) {
    return get_inputs ? edge->from : edge->to;
  };

  // Keep track of value and order to ensure deterministic result
  VectorOfUniqueEntries<Val*> merged_vals;

  // Put groups in a set for quick look up
  std::unordered_set<SegmentedGroup*> segmented_groups_set(
      segmented_groups.begin(), segmented_groups.end());

  // Collect vals associated with edges
  for (auto group : segmented_groups) {
    for (auto edge : edges_to_process_from_or_to_group(group)) {
      if ( // One side of this edge will be `group`, if the other end is
           //  also in segmented_groups, then this is an internal edge
           //  that we don't want.
          !segmented_groups_set.count(opposite_end_of_edge(edge))) {
        merged_vals.pushBack(edge->val);
      }
    }
  }

  // Collect original fusion's inputs/outputs and append at the end
  for (auto group : segmented_groups) {
    for (auto global_val : global_vals_from_or_to_group(group)) {
      merged_vals.pushBack(global_val);
    }
  }

  return merged_vals.vector();
}

// Grab all producer and consumer edges into and out of a group in a
// given list of groups. Does not include any edges within the given
// groups.
std::vector<SegmentedEdge*> getAllEdges(
    const std::vector<SegmentedGroup*>& segmented_groups) {
  VectorOfUniqueEntries<SegmentedEdge*> all_edges;

  for (auto group : segmented_groups) {
    all_edges.insert(
        group->producer_edges.begin(), group->producer_edges.end());
    all_edges.insert(
        group->consumer_edges.begin(), group->consumer_edges.end());
  }

  const std::unordered_set<SegmentedGroup*> group_set(
      {segmented_groups.begin(), segmented_groups.end()});

  auto unique_edges = all_edges.vector();

  // Remove intra edges
  unique_edges.erase(
      std::remove_if(
          unique_edges.begin(),
          unique_edges.end(),
          [&](auto edge) {
            return group_set.count(edge->from) && group_set.count(edge->to);
          }),
      unique_edges.end());

  return unique_edges;
}

// Utility function to list all expressions in a group
void detailGroupPrint(std::ostream& os, const SegmentedGroup* group) {
  IrPrinter irp(os);

  auto sort_val_by_name = [](std::vector<Val*> vals_to_sort) {
    std::sort(vals_to_sort.begin(), vals_to_sort.end(), [](Val* a, Val* b) {
      return a->name() < b->name();
    });
    return vals_to_sort;
  };

  os << "g{";
  if (group->schedulerType() != SchedulerType::None) {
    os << "(" << toString(group->schedulerType()) << ")";
  }
  os << std::endl;
  os << "group id: " << group->groupId() << std::endl;
  os << "inputs:" << std::endl;
  for (auto input : sort_val_by_name(getAllInputs(group))) {
    indent(os, 1) << input << " " << input->getDataType().value() << std::endl;
  }
  os << "outputs:" << std::endl;
  for (auto output : sort_val_by_name(getAllOutputs(group))) {
    indent(os, 1) << output << " " << output->getDataType().value()
                  << std::endl;
  }

  os << std::endl << std::endl;

  for (Expr* e : group->stablyOrderedExprs()) {
    os << e->toString();
    os << "(" << e->name() << ")" << std::endl;
  }
  os << "}" << std::endl << std::endl;
}

//! Insert casts for an intermediate tensorview, i.e. ones
//!  that are in segmentedEdges. The insertion is done on
//!  the complete fusion, which should be owned by a segmented
//!  fusion so that only one segmented fusion will be affected.
//!  The replacement pattern is:
//!                 TV0
//!     replaced as:
//!       fp16_tv = cast(TV0)
//!       fp32_tv = cast(fp16_tv)
//!
//!  The replacement is done only for uses_to_modify exprs. This
//!  function can be called for the same original FP32 tensor multiple
//!  times but with different use exprs. In the second and later
//!  calls, the half-precision tensor should be passed in as well.
//!
//! Returns nullptr if no replacement is done.
TensorView* castIntermediateValueInCompleteFusion(
    Fusion* fusion,
    TensorView* original_fp32_tv,
    const std::vector<Expr*>& uses_to_modify,
    DataType half_type,
    TensorView* half_tv = nullptr) {
  FusionGuard fg(fusion);

  // A utility lambda that creates consumer tensordomain of
  //  the given tv and create a new tensorview around the
  //  new tensordomain with the given data type.
  auto make_consumer_tv = [&](TensorView* from, DataType data_type) {
    // Keep broadcast axes and remove reduction axes
    size_t i = 0;
    auto no_reduction_logical_domain =
        TensorDomain::noReductions(original_fp32_tv->getLogicalDomain());
    std::vector<IterDomain*> new_logical_domain(
        no_reduction_logical_domain.size());
    for (const auto& dom : no_reduction_logical_domain) {
      new_logical_domain[i++] = dom->cloneWithoutRFactor();
    }

    // Create the actual domain and tv.
    return IrBuilder::create<TensorView>(
        IrBuilder::create<TensorDomain>(
            new_logical_domain,
            TensorDomain::getContiguityFilledWith(new_logical_domain, true)),
        data_type);
  };

  TensorView* reverted_fp32_tv = nullptr;
  bool is_replaced = false;

  // replace uses of original tv with fp32_tv in the complete
  //  fusion
  for (auto expr : uses_to_modify) {
    if (reverted_fp32_tv == nullptr) {
      reverted_fp32_tv = make_consumer_tv(original_fp32_tv, DataType::Float);
    }
    auto replaced = ir_utils::replaceValInExprInputs(
        expr, original_fp32_tv, reverted_fp32_tv);
    NVF_ERROR(replaced != expr);
    is_replaced = true;
  }

  if (!is_replaced) {
    return nullptr;
  }

  // create the tv's to cast
  if (half_tv == nullptr) {
    half_tv = make_consumer_tv(original_fp32_tv, half_type);
    IrBuilder::create<UnaryOp>(UnaryOpType::Cast, half_tv, original_fp32_tv);
  }

  // Insert the cast ops.
  IrBuilder::create<UnaryOp>(UnaryOpType::Cast, reverted_fp32_tv, half_tv);

  return half_tv;
}

} // namespace

void SegmentedFusion::finalize() {
  impl_.cleanUnused();
  castInputOutputToLowerPrecision(edges());
}

//! Lower FP precision of inputs and outputs specified by the given
//! edges.
//!
//! The groups_to_merge vector is an optional parameter, and
//! is only used when testing the validity of merging groups. When
//! given, they are treated as a single group, and each edge is
//! either producer or consumer to the group. Uses of cast tensors
//! outside of the groups are not altered.
//!
//! When this is used for a complete fusion after segmentation is completed,
//! groups_to_merge should be empty.
std::vector<SegmentedEdge*> SegmentedFusion::castInputOutputToLowerPrecision(
    const std::vector<SegmentedEdge*>& edges,
    const std::vector<SegmentedGroup*>& groups_to_merge) {
  if (!isOptionEnabled(EnableOption::IoToLowerPrecision)) {
    return {};
  }

  // A map to keep track of the tv's that have been inserted cast
  //  and its fp16 version. Used to avoid cast insertion multiple
  //  times.
  std::unordered_map<TensorView*, TensorView*> fp32_to_half_cast_map;
  // Edges whose associated tensors are cast to lower precision
  std::vector<SegmentedEdge*> affected_edges;

  auto is_to_merge_group = [&groups_to_merge](SegmentedEdge* edge) {
    return std::find(
               groups_to_merge.begin(), groups_to_merge.end(), edge->to) !=
        groups_to_merge.end();
  };

  // Insertions and replacements have to be done carefully when done
  // for a segmented fusion with groups to merge but not merged, i.e.,
  // in the use case of tryMerge. Since the groups to merge are not
  // really merged, they still have their own edges and both of them
  // must be updated accordingly with cast tensors. Insertion of
  // cast-back exprs should be done once for the (virtual) merged
  // group, otherwise there would be multiple same cast exprs in the
  // virtual merged group, which by itself shouldn't be a problem, but
  // the resulting fusion becomes different from the fusion created
  // by actually merging the groups, since in that case the cast
  // should only be inseretd once. Both variations of the original
  // fusion should be functionally correct, but different expressions
  // may trigger different scheduling heuristics or a scheduling
  // failure only in either case.
  //
  // To avoid this discrepancy, when this is done with virtual merged
  // groups, bundle all edges to the merged groups and process them
  // together. This way, only one instance of the cast-back expr should
  // be inserted.
  //
  // Note that this analysis and replacement would be much simpler if we
  // actually created a merged SegmentedGroup for those groups to merge. The
  // merged SegmentedGroup should be temporary, and should be reverted back
  // after this tryMerge is done. However, that would also mean edges would have
  // to be temporary modified and be reverted back to the original state.

  // Edges to the groups to merge. Grouped into vectors by edge vals.
  std::unordered_map<TensorView*, std::vector<SegmentedEdge*>>
      edges_to_merge_groups;
  // Keep track of the edge vals as a vector to apply insertion in a
  // deterministic order
  std::vector<TensorView*> vals_of_edges_to_merge_groups;

  // Bundle edges to the merged groups
  std::vector<std::vector<SegmentedEdge*>> bundled_edges;
  for (auto edge : edges) {
    if (!edge->val->isA<TensorView>()) {
      continue;
    }
    auto edge_tv = edge->val->as<TensorView>();
    // Only look at ones that need to cast to fp16 or bf16
    if (force_fp16_tv_set_.count(edge_tv) == 0) {
      continue;
    }

    if (is_to_merge_group(edge)) {
      if (edges_to_merge_groups.emplace(edge_tv, std::vector<SegmentedEdge*>{})
              .second) {
        vals_of_edges_to_merge_groups.push_back(edge_tv);
      }
      edges_to_merge_groups[edge_tv].push_back(edge);
    } else {
      bundled_edges.push_back({edge});
    }
  }

  for (const auto val : vals_of_edges_to_merge_groups) {
    bundled_edges.emplace_back(edges_to_merge_groups.at(val));
  }

  // Go through all edges of the segmented fusion.
  for (const auto& edges : bundled_edges) {
    auto edge_tv = edges.at(0)->val->as<TensorView>();

    // Gather exprs that should be modified. Start with all use
    // exprs.
    std::vector<Expr*> uses_to_modify;

    for (auto edge_val_use_expr : edge_tv->uses()) {
      if (std::any_of(edges.begin(), edges.end(), [&](SegmentedEdge* edge) {
            return std::find(
                       edge->to->exprs().begin(),
                       edge->to->exprs().end(),
                       edge_val_use_expr) != edge->to->exprs().end();
          })) {
        uses_to_modify.push_back(edge_val_use_expr);
      }
    }

    // Some of SelectOp-like expressions have the limitation that
    // input tensors must be fusion inputs, so even just cast
    // shouldn't be inserted.
    uses_to_modify.erase(
        std::remove_if(
            uses_to_modify.begin(),
            uses_to_modify.end(),
            [&](Expr* edge_val_use_expr) {
              return edge_val_use_expr->isOneOf<
                         SelectOp,
                         SliceOp,
                         IndexSelectOp,
                         GatherOp>() &&
                  edge_val_use_expr->input(0) == edge_tv;
            }),
        uses_to_modify.end());

    if (uses_to_modify.empty()) {
      continue;
    }

    auto cast_tv_it = fp32_to_half_cast_map.find(edge_tv);
    TensorView* cast_tv = nullptr;

    if (cast_tv_it == fp32_to_half_cast_map.end()) {
      // This is the first call to insert cast for this tensor
      cast_tv = castIntermediateValueInCompleteFusion(
          complete_fusion_.get(),
          edge_tv,
          uses_to_modify,
          force_half_precision_type_);
      NVF_ERROR(cast_tv != nullptr);
      fp32_to_half_cast_map[edge_tv] = cast_tv;
    } else {
      // There was cast insertion for this edge_tv, so we already have
      // its half-type tensor. This happens when a tensor is used by
      // different consumer groups. Reuse it to avoid cast edge_tv
      // redundantly.
      cast_tv = cast_tv_it->second;
      cast_tv = castIntermediateValueInCompleteFusion(
          complete_fusion_.get(),
          edge_tv,
          uses_to_modify,
          force_half_precision_type_,
          cast_tv);
    }

    // Update the edge to use the fp16 version
    for (auto edge_to_update : edges) {
      edge_to_update->val = cast_tv;

      // The expr pointers on the group's expr list might have been freed
      //  by now after `ir_utils::replaceValInExprInputs`.
      // Need a valid expression list to continue. Update from and to group.
      edge_to_update->from->resetExprList();
      edge_to_update->to->resetExprList();
      affected_edges.push_back(edge_to_update);
    }
  }
  return affected_edges;
}

std::vector<SegmentedEdge*> SegmentedFusion::getEdgesByVal(Val* val) const {
  std::vector<SegmentedEdge*> edges_with_val;
  std::copy_if(
      cedges().begin(),
      cedges().end(),
      std::back_inserter(edges_with_val),
      [&](auto edge) { return edge->val == val; });
  return edges_with_val;
}

void SegmentedFusion::revertInputOutputPrecisionChanges(
    const std::vector<SegmentedEdge*>& edges) {
  std::unordered_set<Val*> lowered_tv_to_remove;
  std::unordered_set<Val*> same_precision_tv_to_remove;
  for (auto edge : edges) {
    auto lowered_tv = edge->val;
    auto original_tv = lowered_tv->definition()->inputs().at(0);
    for (auto cast_back_expr : lowered_tv->uses()) {
      NVF_ERROR(
          cast_back_expr->isA<UnaryOp>() &&
          cast_back_expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Cast);
      auto same_precision_tv = cast_back_expr->outputs().at(0);
      for (auto expr : complete_fusion_->unordered_uses(same_precision_tv)) {
        ir_utils::replaceValInExprInputs(expr, same_precision_tv, original_tv);
      }
      same_precision_tv_to_remove.insert(same_precision_tv);
    }
    lowered_tv_to_remove.insert(lowered_tv);
    edge->val = original_tv;
  }

  // Any group with an edge with the original TVs may have its
  // expressions replaced.
  std::unordered_set<SegmentedGroup*> groups_to_reset;
  for (auto lowered_tv : lowered_tv_to_remove) {
    auto original_tv = lowered_tv->definition()->inputs().at(0);
    for (auto e : getEdgesByVal(original_tv)) {
      groups_to_reset.insert(e->from);
      groups_to_reset.insert(e->to);
    }
  }
  for (auto group : groups_to_reset) {
    group->resetExprList();
  }

  // Remove the temporary vals
  for (auto v : same_precision_tv_to_remove) {
    v->fusion()->removeVal(v);
  }
  for (auto v : lowered_tv_to_remove) {
    v->fusion()->removeVal(v);
  }
}

//! An utility class to compute and maintain the "producers of"
//!   relationship in a segmented graph. Space heavy and should
//!   avoid use on very large graphs.
//!
//!  Currently trying to move as far as possible with only a
//!   producer map, without transposing it to make a consumer map.
//!  Making it NonCopyable because we should never need to
//!   copy an instance of this class.
//!  TODO: Space efficiency of this class will be important,
//!        because we need it in the pre-merging of segmentedGroups,
//!        currently O(n^2). O(nlogn) would be a reasonable
//!        goal to achieve.
class GroupDependencyAnalysis : public NonCopyable, public SegmenterAnalysis {
 public:
  //! Populate producers of all groups in segmented fusion
  explicit GroupDependencyAnalysis(const SegmentedFusion* segmented_fusion)
      : segmented_fusion_(segmented_fusion) {
    computeAllProducers();
  }

  //! Checks if group is consumer of any group in groups_to_check
  //!  TODO: refactor this similar to isConsumerOf
  bool isConsumerOfAny(
      SegmentedGroup* group,
      const std::vector<SegmentedGroup*>& groups_to_check) {
    auto& producers_of_group = getAllKnownProducersSet(group);
    for (const auto& potential_producer : groups_to_check) {
      if (producers_of_group->has(potential_producer)) {
        return true;
      }
    }
    return false;
  }

  bool isConsumerOf(SegmentedGroup* a, SegmentedGroup* b) {
    auto it = known_producers_of_.find(a);
    if (it == known_producers_of_.end()) {
      return false;
    }
    return it->second->has(b);
  }

  bool isProducerOf(SegmentedGroup* a, SegmentedGroup* b) {
    return isConsumerOf(b, a);
  }

  //! Finds the common producers of given set of groups
  GroupSet getCommonProducersOf(std::vector<SegmentedGroup*> groups);

  //! Update the map when the given two groups have been merged to create `ab`
  //! this method is for book keeping and query only, doesn't implicitly check
  //!  for DAG
  void mergeGroups(SegmentedGroup* a, SegmentedGroup* b, SegmentedGroup* ab);

  //! Update the map when the given two groups have been merged to create
  //! `merged` this method is for book keeping and query only, doesn't
  //! implicitly check
  //!  for DAG
  void mergeGroups(const GroupSet& groups, SegmentedGroup* merged);

  //! Populate all values that is on a path from producer to consumer
  //!  efficiency can be important here. (TODO)
  GroupSet valuesBetween(SegmentedGroup* producer, SegmentedGroup* consumer) {
    if (producer == consumer) {
      return {};
    }

    GroupSet values_between;
    auto& all_producers_of_consumer = known_producers_of_.at(consumer);
    NVF_ERROR(
        all_producers_of_consumer->has(producer),
        "Fusion segment: Trying to compute path between two nodes that are not "
        "producer-consumer pairs");

    for (auto producer_of_consumer : *all_producers_of_consumer) {
      if (known_producers_of_.at(producer_of_consumer)->has(producer)) {
        values_between.pushBack(producer_of_consumer);
      }
    }

    return values_between;
  }

  //! Checks if the segmented fusion this class tracks is still a DAG
  //!  used for generating assertions after transforms
  bool isproducerMapDAG() const {
    for (auto& it : known_producers_of_) {
      if (it.second->has(it.first)) {
        return false;
      }
    }
    return true;
  }

 private:
  //! Collect initial producer info using
  //!  a work list algorithm through forward traversal
  //!  a backward DFS would do the same
  void computeAllProducers();

  //! Add all consumers of `producer` to `to_visit`
  void addConsumersToWorkList(SegmentedGroup* producer, GroupSet& to_visit) {
    for (auto e : producer->consumer_edges) {
      // A consumer wouldn't have been worked before any of its producer
      to_visit.pushBack(e->to);
    }
  }

  //! Propagate all known producers of `from` into `into`, used to keep track
  //! of:
  //!  1. `from` is a producer of `into`
  //!  2. `from` has been merged with other group to create `into`
  void mergeAllKnownProducersIntoFrom(
      SegmentedGroup* into,
      SegmentedGroup* from) {
    auto& producer_set_to_merge = *getAllKnownProducersSet(from);
    for (auto group : producer_set_to_merge) {
      getAllKnownProducersSet(into)->pushBack(group);
    }
  }

  //! Utility to access known producers of a group so far
  std::unique_ptr<GroupSet>& getAllKnownProducersSet(SegmentedGroup* group) {
    auto& producer_set_ptr = known_producers_of_[group];
    if (!producer_set_ptr) {
      producer_set_ptr = std::make_unique<GroupSet>();
    }
    return producer_set_ptr;
  }

  // utility to compute the set intersection of group sets a,b
  GroupSet groupSetIntersection(const GroupSet& a, const GroupSet& b) {
    bool a_is_smaller = a.size() < b.size();
    const auto& smaller_group_set = a_is_smaller ? a : b;
    const auto& bigger_group_set = a_is_smaller ? b : a;

    GroupSet intersection;
    for (auto group : smaller_group_set) {
      if (bigger_group_set.has(group)) {
        intersection.pushBack(group);
      }
    }
    return intersection;
  }

 private:
  const SegmentedFusion* segmented_fusion_;
  std::unordered_map<SegmentedGroup*, std::unique_ptr<GroupSet>>
      known_producers_of_;
};

//! Finds the common producers of given set of groups
GroupSet GroupDependencyAnalysis::getCommonProducersOf(
    std::vector<SegmentedGroup*> groups) {
  if (groups.empty()) {
    return {};
  }

  // Optimization: start with the smallest producer set
  std::sort(
      groups.begin(),
      groups.end(),
      [this](SegmentedGroup* a, SegmentedGroup* b) {
        return known_producers_of_.at(a)->size() <
            known_producers_of_.at(b)->size();
      });

  // Get intersection of producers
  GroupSet common_producers = *(known_producers_of_.at(groups[0]));
  for (const auto i : arange(1, groups.size())) {
    common_producers = groupSetIntersection(
        common_producers, *(known_producers_of_.at(groups[i])));
  }

  return common_producers;
}

//! Update the map when the given two groups have been merged to create `ab`
//! this method is for book keeping and query only, doesn't implicitly check
//!  for DAG
void GroupDependencyAnalysis::mergeGroups(
    SegmentedGroup* a,
    SegmentedGroup* b,
    SegmentedGroup* ab) {
  // Access/Create the producer set of ab
  auto& ab_set = getAllKnownProducersSet(ab);

  // propagate a's and b's known producers into ab
  mergeAllKnownProducersIntoFrom(ab, a);
  mergeAllKnownProducersIntoFrom(ab, b);

  // a, b are now merged, so no longer exist
  ab_set->erase(a);
  ab_set->erase(b);

  // a, b no longer exist, remove their producer sets
  known_producers_of_.erase(a);
  known_producers_of_.erase(b);

  // update producer maps of other groups
  for (auto& it : known_producers_of_) {
    // for all groups that are produced by either a or b
    if (it.second->has(a) || it.second->has(b)) {
      // insert ab as the new producer
      it.second->pushBack(ab);
      // all producers of both a and b are now producers of `it`
      mergeAllKnownProducersIntoFrom(it.first, ab);
    }
    // a, b no longer exist, remove them from `it`
    it.second->erase(a);
    it.second->erase(b);
  }
}

//! Update the map when the given two groups have been merged to create
//! `merged` this method is for book keeping and query only, doesn't
//! implicitly check
//!  for DAG
void GroupDependencyAnalysis::mergeGroups(
    const GroupSet& groups,
    SegmentedGroup* merged) {
  // Access/Create the producer set of merged
  auto& merged_set = getAllKnownProducersSet(merged);

  // Populate all producers of groups and
  //  write into producer map of merged
  std::for_each(
      groups.begin(), groups.end(), [this, merged](SegmentedGroup* group) {
        mergeAllKnownProducersIntoFrom(merged, group);
      });

  // Erase all groups that was merged from producer map
  std::for_each(
      groups.begin(), groups.end(), [this, &merged_set](SegmentedGroup* group) {
        // erase inter dependencies
        merged_set->erase(group);
        // erase producer map tracking merged entires
        known_producers_of_.erase(group);
      });

  // Update producer relationships with other groups in producer map
  for (auto& it : known_producers_of_) {
    auto producer_intersection = groupSetIntersection(*(it.second), groups);
    // if current node has any producer that was merged
    if (!producer_intersection.empty()) {
      for (auto merged_producer : producer_intersection) {
        // delete all disappearing producers
        it.second->erase(merged_producer);
      }
      // insert the new group as producer
      it.second->pushBack(merged);
      // all producers of merged are now producers of `it`
      mergeAllKnownProducersIntoFrom(it.first, merged);
    }
  }
}

//! Collect initial producer info using
//!  a work list algorithm through forward traversal
//!  a backward DFS would do the same
void GroupDependencyAnalysis::computeAllProducers() {
  GroupSet visited;
  GroupSet to_visit;

  // Collect source nodes, with no producers we are guaranteed
  //  a source node on a DAG
  for (auto group : segmented_fusion_->cgroups()) {
    if (group->producer_edges.empty()) {
      visited.pushBack(group);
    }
  }

  // visited now only contain source nodes
  //  they can go backward to nowhere
  for (auto group : visited) {
    addConsumersToWorkList(group, to_visit);
  }

  while (!to_visit.empty()) {
    SegmentedGroup* to_update = nullptr;
    for (auto visiting_group : to_visit) {
      if (std::all_of(
              visiting_group->producer_edges.begin(),
              visiting_group->producer_edges.end(),
              [&visited](SegmentedEdge* e) { return visited.has(e->from); })) {
        // filter multi-edges
        GroupSet producers_of_visiting_group;
        for (auto edge : visiting_group->producer_edges) {
          producers_of_visiting_group.pushBack(edge->from);
        }

        // populate all possible paths
        // from producer backward, including
        // the producer
        for (auto producer : producers_of_visiting_group) {
          getAllKnownProducersSet(visiting_group)->pushBack(producer);
          mergeAllKnownProducersIntoFrom(visiting_group, producer);
        }
        to_update = visiting_group;
        break;
      }
    }
    if (to_update) {
      addConsumersToWorkList(to_update, to_visit);
      to_visit.erase(to_update);
      visited.pushBack(to_update);
    } else {
      NVF_THROW("unreachable, original graph not a DAG");
    }
  }
}

std::ostream& operator<<(
    std::ostream& os,
    const SegmentedFusion* segmented_fusion) {
  // Topologically sort groups
  GroupDependencyAnalysis dependency(segmented_fusion);
  std::vector<SegmentedGroup*> groups_to_print(
      segmented_fusion->cgroups().begin(), segmented_fusion->cgroups().end());
  std::vector<SegmentedGroup*> sorted_groups_to_print;

  // Sort groups topologically from producer to consumer before printing
  while (!groups_to_print.empty()) {
    auto group_it_to_append = groups_to_print.begin();
    for (auto group_it_to_compare = groups_to_print.begin();
         group_it_to_compare != groups_to_print.end();
         group_it_to_compare++) {
      if (dependency.isProducerOf(*group_it_to_compare, *group_it_to_append)) {
        group_it_to_append = group_it_to_compare;
      }
    }
    sorted_groups_to_print.push_back(*group_it_to_append);
    groups_to_print.erase(group_it_to_append);
  }

  // Do a reverse look up to check the order of sorted groups
  std::unordered_map<SegmentedGroup*, size_t> group_order;
  for (const auto i : arange(sorted_groups_to_print.size())) {
    group_order[sorted_groups_to_print[i]] = i;
  }

  // Sort edges to print
  std::vector<SegmentedEdge*> sorted_edges_to_print(
      segmented_fusion->cedges().begin(), segmented_fusion->cedges().end());

  std::sort(
      sorted_edges_to_print.begin(),
      sorted_edges_to_print.end(),
      [&group_order](SegmentedEdge* edge_a, SegmentedEdge* edge_b) {
        return group_order.at(edge_a->from) < group_order.at(edge_b->from);
      });

  os << "Segmented_Fusion Dump: -- fusion segments:\n";
  os << "Segmented_Fusion{ \n";
  os << "groups: \n";
  for (const auto g : sorted_groups_to_print) {
    os << "  " << g << "\n";
  }
  os << "edges: \n";
  for (const auto e : sorted_edges_to_print) {
    os << "  " << e << "\n";
  }
  os << "\ngroup details:\n";
  for (const auto g : sorted_groups_to_print) {
    detailGroupPrint(os, g);
  }
  os << "} //Segmented_Fusion\n";
  return os;
}

void SegmentedFusion::print() const {
  debug() << "Segmented_Fusion Dump: -- Re-written complete fusion:{\n";
  completeFusion()->printMath();
  debug() << "} // {Re-written complete fusion}\n";
  debug() << this << "\n";
}

std::string toString(const SegmentedFusion* segmented_fusion) {
  std::stringstream ss;
  ss << segmented_fusion;
  return ss.str();
}

//! Sets the root as logical and erases root of all inputs in fusion. Any
//! non-constant expressions in those extents are replaced by new scalars with
//! no definition. These mutations are performed throughout the Fusion so that
//! downstream expressions dependent on the original inputs' logical extents can
//! be computed properly.
void eraseInputDistinctRootDomains(Fusion* fusion) {
  FusionGuard fg(fusion);

  // Holds all Val replacements across all inputs
  std::unordered_map<Val*, Val*> replacement_map;

  for (auto tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    // Create a new logical domain and replacement TensorDomain.
    // Given an logical domain, create a new IterDomain.
    // Otherwise, clone the previous IterDomain
    std::vector<IterDomain*> new_logical_domain;
    auto logical = tv->getLogicalDomain();
    new_logical_domain.reserve(logical.size());

    // Does the logical domain contain all concrete sized extents?
    bool tv_is_concrete = true;
    for (auto id : logical) {
      if (!id->extent()->isConstScalar()) {
        tv_is_concrete = false;
        break;
      }
    }

    for (const auto& id : logical) {
      if (id->isRFactorProduct()) {
        // Create new symbolic extents for logical iterDomains
        auto domain_extent = (!tv_is_concrete)
            ? IrBuilder::create<Val>(DataType::Index)
            : id->extent();
        replacement_map.emplace(id->extent(), domain_extent);
        new_logical_domain.push_back(IterDomainBuilder(id)
                                         .extent(domain_extent)
                                         .resetSchedulingParams()
                                         .build());
      } else {
        new_logical_domain.push_back(id->cloneWithoutRFactor());
      }
    }

    TensorDomain* new_td = nullptr;
    if (tv->domain()->hasAllocation()) {
      // we need to reorder the logical domain into allocation domain
      // consistently with the mapping from the old TensorView logical domain to
      // its allocation domain
      std::unordered_map<IterDomain*, IterDomain*> old_to_new;
      for (const auto i : arange(logical.size())) {
        old_to_new.emplace(logical[i], new_logical_domain[i]);
      }

      ReplayTransformations replay(tv->getAllocationDomain(), old_to_new);
      // Without this,
      // https://github.com/NVIDIA/Fuser/blob/e613929a6c21b3095c8817b01b8f177096a26e60/csrc/transform_iter.cpp#L299
      // tries to look for root IDs in the map, which shouldn't exist because
      // the whole purpose of this function is to remove the root domain.
      replay.setErrorOnFailure(false);
      // We don't need replay.setReplayRFactor(true). The new root is the same
      // as the new logical so there aren't any expressions between them.

      std::vector<IterDomain*> new_alloc;
      new_alloc.reserve(tv->getAllocationDomain().size());
      for (IterDomain* alloc_id : tv->getAllocationDomain()) {
        IterDomain* new_alloc_id = replay.getReplay().at(alloc_id);
        // ReplayTransformations replay transforms but not paralelization, so
        // we have to manually parallelize the new allocation ID. In other
        // places, parallelization is usually done through parallelizeAllLike.
        new_alloc_id->parallelize(alloc_id->getParallelType());
        new_alloc.push_back(new_alloc_id);
      }

      std::vector<IterDomain*> new_loop;
      if (tv->getLoopDomain() == tv->getAllocationDomain()) {
        new_loop = new_alloc;
      } else {
        NVF_ERROR(
            tv->getLoopDomain() == tv->getLogicalDomain(),
            tv,
            " has an unexpected loop domain:\n",
            tv->domain()->toString(0, /*loop_only=*/false));

        new_loop = new_logical_domain;
      }

      new_td = IrBuilder::create<TensorDomain>(
          /*root_domain=*/std::vector<IterDomain*>(),
          new_logical_domain,
          new_alloc,
          new_loop,
          tv->domain()->contiguity());
    } else {
      NVF_ERROR(
          tv->getLoopDomain() == tv->getLogicalDomain(),
          tv,
          " has an unexpected loop domain:\n",
          tv->domain()->toString(0, /*loop_only=*/false));
      new_td = IrBuilder::create<TensorDomain>(
          new_logical_domain, tv->domain()->contiguity());
    }

    // Remove reduction domains from new_td
    if (new_td->hasReduction()) {
      std::vector<std::optional<bool>> no_red_contiguity;
      for (size_t i : arange(new_td->maybeAllocation().size())) {
        if (new_td->maybeAllocation()[i]->isReduction()) {
          continue;
        }
        no_red_contiguity.push_back(new_td->contiguity()[i]);
      }
      if (new_td->hasAllocation()) {
        const std::vector<IterDomain*> new_logical =
            TensorDomain::noReductions(new_td->logical());
        new_td = IrBuilder::create<TensorDomain>(
            /*root_domain=*/std::vector<IterDomain*>{},
            /*logical_domain=*/new_logical,
            /*allocation=*/TensorDomain::noReductions(new_td->allocation()),
            /*loop_domain=*/TensorDomain::noReductions(new_td->loop()),
            /*contiguity=*/no_red_contiguity);
      } else {
        new_td = IrBuilder::create<TensorDomain>(
            /*logical_domain=*/TensorDomain::noReductions(new_td->logical()),
            /*contiguity=*/no_red_contiguity);
      }
    }

    replacement_map.emplace(tv->domain(), new_td);
  }

  // This will replace the values in the mapping replacement_map throughout the
  // Fusion
  ir_utils::replaceValue(fusion, replacement_map);
}

std::pair<IrCloner, std::unique_ptr<Fusion>> SegmentedFusion::makeFusion(
    SegmentedGroup* sg) const {
  // TODO Optimize cloning step by only copying values and expressions between
  // the fusion segment's inputs and outputs.
  auto fusion_segment = std::make_unique<Fusion>();

  IrCloner complete_to_segment_map =
      Fusion::copy(completeFusion(), fusion_segment.get());

  std::vector<Val*> input_list(
      fusion_segment->inputs().begin(), fusion_segment->inputs().end());
  for (auto inp : input_list) {
    fusion_segment->removeInput(inp);
  }

  std::vector<Val*> output_list(
      fusion_segment->outputs().begin(), fusion_segment->outputs().end());
  for (auto out : output_list) {
    fusion_segment->removeOutput(out);
  }

  std::vector<TensorView*> view_tvs;
  for (auto inp : getAllInputs(sg)) {
    auto clone_tv = complete_to_segment_map.clone(inp);
    fusion_segment->addInput(clone_tv);
    if (inp->isDefinitionType<ViewOp>()) {
      NVF_ERROR(clone_tv != nullptr && clone_tv->isA<TensorView>());
      view_tvs.push_back(clone_tv->as<TensorView>());
    }
  }

  // note, we would want to keep output consistent and not artificially drop
  // duplicates.
  for (auto out : sg->output_vals_) {
    fusion_segment->addOutput(complete_to_segment_map.clone(out));
  }

  // Replace all vals that are logical extents in fusion_segment->inputs() with
  // new Vals so that they can be bound to the segment inputs.
  eraseInputDistinctRootDomains(fusion_segment.get());

  return std::make_pair(complete_to_segment_map, std::move(fusion_segment));
}

std::unique_ptr<SegmentedFusion> SegmentCandidateFinder::segment(
    const Fusion* fusion,
    const KernelArgumentHolder& inputs,
    SegmentCandidateFinderOptions options) {
  auto fusion_copy = std::make_unique<Fusion>(*fusion);
  return segment(std::move(fusion_copy), inputs, options);
}

// Perform segmentation on and take ownership of the given fusion
std::unique_ptr<SegmentedFusion> SegmentCandidateFinder::segment(
    std::unique_ptr<Fusion> fusion,
    const KernelArgumentHolder& inputs,
    SegmentCandidateFinderOptions options,
    bool multi_device) {
  if (isDebugDumpEnabled(DebugDumpOption::FusionSegments)) {
    debug() << "Segment the fusion (Original Fusion Un-modified): "
            << std::endl;
    fusion->printMath();
  }
  SegmentCandidateFinder scf(std::move(fusion), inputs, options, multi_device);
  return std::move(scf.segmented_fusion_);
}

std::unique_ptr<SegmentedFusion> SegmentCandidateFinder::segment(
    std::unique_ptr<Fusion> fusion,
    const KernelArgumentHolder& inputs,
    SchedulerRuntimeInfo& runtime_info) {
  if (!hasSegmentHints(fusion.get())) {
    scheduler_debug_utils::canScheduleMessage(
        "***Runtime***: Try to schedule fusion un-segmented:\n");
    const auto fusion_heuristic_type =
        Schedule::proposeHeuristics(fusion.get(), runtime_info);
    if (fusion_heuristic_type != SchedulerType::None) {
      return SegmentedFusion::fromCompleteFusion(
          std::move(fusion), fusion_heuristic_type, inputs);
    }
  } else {
    scheduler_debug_utils::canScheduleMessage(
        "***Runtime***: Has segment hints, skip un-segmented scheduling.\n");
  }
  if (fusion) {
    scheduler_debug_utils::canScheduleMessage(
        "\n***Runtime***: Try to schedule fusion segmented:\n");
    return SegmentCandidateFinder::segment(std::move(fusion), inputs);
  } else {
    NVF_THROW("unreachable!");
  }
}

bool SegmentCandidateFinder::hasSegmentHints(Fusion* fusion) {
  for (const auto& expr : fusion->exprs()) {
    if (expr->isA<LoadStoreOp>()) {
      auto op = expr->as<LoadStoreOp>();
      // SegmenterSet is a segmenter hint that needs explicit segment call
      if (op->opType() == LoadStoreOpType::SegmenterSet) {
        return true;
      }
    }
  }
  return false;
}

void SegmentCandidateFinder::resetLevels() {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::resetLevels");

  std::deque<SegmentedGroup*> to_visit;
  std::unordered_map<SegmentedGroup*, int64_t> num_producer_edges;
  for (SegmentedGroup* group : groups()) {
    group->level_ = 0;
    if ((num_producer_edges[group] = std::ssize(group->producer_edges)) == 0) {
      // Start by visiting groups that have no producer edges.
      to_visit.push_back(group);
    }
  }

  int64_t num_visited = 0;
  while (!to_visit.empty()) {
    SegmentedGroup* visiting = to_visit.front();
    to_visit.pop_front();
    num_visited++;

    for (SegmentedEdge* out : visiting->consumer_edges) {
      SegmentedGroup* consumer = out->to;
      consumer->level_ = std::max(consumer->level_, visiting->level_ + 1);
      // After visiting a group, decrement the number of producer edges of each
      // consumer. When that number reaches 0, add the consumer to the visit
      // list.
      if ((--num_producer_edges.at(consumer)) == 0) {
        to_visit.push_back(consumer);
      }
    }
  }

  NVF_ERROR(
      num_visited == std::ssize(groups()), "Error in graph, is not a DAG.");
}

// Disconect group from neighbors, and return edges that were disconnected
void SegmentCandidateFinder::disconnectGroup(SegmentedGroup* group) {
  // Remove producer edges
  std::vector<SegmentedEdge*> producer_edges(
      group->producer_edges.begin(), group->producer_edges.end());
  for (auto edge : producer_edges) {
    segmented_fusion_->removeEdge(edge);
  }

  // Remove consumer edges
  std::vector<SegmentedEdge*> consumer_edges(
      group->consumer_edges.begin(), group->consumer_edges.end());
  for (auto edge : consumer_edges) {
    segmented_fusion_->removeEdge(edge);
  }
}

void SegmentCandidateFinder::eraseGroups(
    std::unordered_set<SegmentedGroup*>& groups_to_erase) {
  for (auto group : groups_to_erase) {
    disconnectGroup(group);
  }

  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [&groups_to_erase](SegmentedGroup* group) {
            if (groups_to_erase.find(group) != groups_to_erase.end()) {
              return true;
            };
            return false;
          }),
      groups().end());
}

std::vector<SegmentedEdge*> SegmentedFusion::getEdgesBetween(
    const SegmentedGroup* producer,
    const SegmentedGroup* consumer) const {
  std::vector<SegmentedEdge*> edges_between;

  // Look through producer's consumer edges
  for (SegmentedEdge* edge : producer->consumer_edges) {
    if (edge->to == consumer) {
      edges_between.push_back(edge);
    }
  }

  return edges_between;
}

void SegmentedFusion::connectGroups(
    SegmentedGroup* producer,
    SegmentedGroup* consumer,
    Val* val) {
  SegmentedEdge* new_edge = newEdge(producer, consumer, val);
  producer->consumer_edges.push_back(new_edge);
  consumer->producer_edges.push_back(new_edge);
}

SegmentedGroup* SegmentCandidateFinder::mergeNodes() {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::mergeNodes");
  SegmentedGroup* last_merged = nullptr;
  auto it = to_merge_.begin();
  NVF_ERROR(to_merge_.size() % 2 == 0);
  while (it != to_merge_.end()) {
    auto group1 = *it++;
    auto group2 = *it++;

    clean_up_groups_.emplace(group1);
    clean_up_groups_.emplace(group2);

    // Make the new joined node
    auto joined_group = segmented_fusion_->newGroup();

    // Merge input and output vals
    joined_group->input_vals_ =
        group1->input_vals_.computeUnion(group2->input_vals_);
    joined_group->output_vals_ =
        group1->output_vals_.computeUnion(group2->output_vals_);

    // Merge expressions
    joined_group->exprs_ = group1->exprs_;
    joined_group->exprs_.insert(
        joined_group->exprs_.end(),
        group2->exprs_.begin(),
        group2->exprs_.end());

    // Get all edges that will connect to the new joined group
    auto producer_edges = getMergedProducerEdges(group1, group2);
    auto consumer_edges = getMergedConsumerEdges(group1, group2);

    // Connect all producer edges to the new joined group
    for (auto edge : producer_edges) {
      segmented_fusion_->connectGroups(edge->from, joined_group, edge->val);
    }

    // Connect all consumer edges from the new joined group
    for (auto edge : consumer_edges) {
      segmented_fusion_->connectGroups(joined_group, edge->to, edge->val);
    }

    // Now that all new connections are made, disconnect the old groups, this
    // invalidates producer_edges and consumer_edges
    for (auto merged_group : {group1, group2}) {
      disconnectGroup(merged_group);
    }

    // Set scheduler type for the new group
    joined_group->setSchedulerType(deriveSchedulerType(joined_group));

    // Update group dependency data if initialized
    if (group_dependency_) {
      group_dependency_->as<GroupDependencyAnalysis>()->mergeGroups(
          group1, group2, joined_group);
    }

    last_merged = joined_group;
  }

  to_merge_.clear();

  // Clean up merged groups
  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [this](SegmentedGroup* group) {
            return this->clean_up_groups_.find(group) !=
                this->clean_up_groups_.end();
          }),
      groups().end());

  clean_up_groups_.clear();
  return last_merged;
}

// Logic largely parallels mergeNodes, but they are used
//  in different phases of segmentation. Should consider
//  a clean up and share the implementations.
SegmentedGroup* SegmentCandidateFinder::mergeAllGivenGroups(
    const std::vector<SegmentedGroup*>& groups_to_merge) {
  NVF_ERROR(
      !groups_to_merge.empty(),
      "fusion segment :(mergeAllGivenGroups) tried to merge no groups");

  // The fusion input auxiliary groups should never be merged.
  const auto& aux_input_groups = getAuxiliaryInputGroups();
  std::vector<SegmentedGroup*> aux_groups_to_merge;
  std::ranges::copy_if(
      groups_to_merge,
      std::back_inserter(aux_groups_to_merge),
      [&](SegmentedGroup* group) {
        return std::ranges::find(aux_input_groups, group) !=
            aux_input_groups.end();
      });
  NVF_ERROR(
      aux_groups_to_merge.empty(),
      "Trying to merge auxiliary input groups: ",
      toDelimitedString(aux_groups_to_merge));

  // Make a set to detect internal edges
  std::unordered_set<SegmentedGroup*> group_set(
      groups_to_merge.begin(), groups_to_merge.end());

  // Create new group
  auto joined_group = segmented_fusion_->newGroup();

  // Track unique vals and exprs to avoid duplicates
  std::unordered_set<Val*> used_edge_vals_set;
  std::unordered_set<Expr*> exprs_set;

  // Merge inputs and outputs from all groups
  for (auto group : groups_to_merge) {
    joined_group->input_vals_.pushBack(group->input_vals_);
    joined_group->output_vals_.pushBack(group->output_vals_);
  }

  // Get all edges that will connect to the new joined group
  auto all_edges = getAllEdges(groups_to_merge);

  // Connect all external edges to the new joined group
  for (auto edge : all_edges) {
    if (group_set.count(edge->from)) {
      // This is a consumer edge from the merged group
      segmented_fusion_->connectGroups(joined_group, edge->to, edge->val);
    } else {
      // This is a producer edge to the merged group
      segmented_fusion_->connectGroups(edge->from, joined_group, edge->val);
    }
  }

  // Disconnect all original groups before connecting the new one, this
  // invalidates all_edges
  for (auto group : groups_to_merge) {
    disconnectGroup(group);
  }

  // Merge all expressions from the groups
  for (auto group : groups_to_merge) {
    for (auto expr : group->exprs_) {
      if (exprs_set.insert(expr).second) {
        joined_group->exprs_.push_back(expr);
      }
    }
  }

  // Clean up original groups
  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [&group_set](SegmentedGroup* group) -> bool {
            return group_set.count(group);
          }),
      groups().end());

  joined_group->setSchedulerType(deriveSchedulerType(joined_group));
  return joined_group;
}

namespace {

// SegmenterSet hints a kernel break
bool tryingToMergeSegmenterSet(Fusion* fusion) {
  for (auto expr : fusion->exprs()) {
    if (expr->isA<LoadStoreOp>() &&
        expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::SegmenterSet) {
      auto out = expr->output(0);
      // output from SegmenterSet node should be:
      //   1. an output from the given fusion, and
      //   2. not be used by any node within the graph
      // This ensures no segment spans across the data flow from SegmenterSet
      if (!out->isFusionOutput() || !out->uses().empty()) {
        return true;
      }
    }
  }
  return false;
}

// Guard to temporarily change the inputs and outputs of a
// fusion. Cast expressions to fp32 and fp16 are also inserted. On
// destruction will return fusion to original state.
class FusionSegmentGuard : public NonCopyable {
 public:
  FusionSegmentGuard() = delete;

  // Narrow the fusion to a segment defined by inputs and outputs
  FusionSegmentGuard(
      Fusion* fusion,
      std::vector<Val*> inputs,
      std::vector<Val*> outputs)
      : fusion_(fusion) {
    FUSER_PERF_SCOPE("Segmenter::FusionSegmentGuard");
    NVF_ERROR(fusion_ != nullptr);
#ifndef NDEBUG
    num_original_exprs_ = fusion_->exprs().size();
    original_tvs_ = fusion_->allTvs();
#endif // NDEBUG
    narrowToNewSegment(inputs, outputs);
  }

  // Just insert cast without narrowing
  FusionSegmentGuard(SegmentedFusion* segmented_fusion)
      : segmented_fusion_(segmented_fusion),
        fusion_(segmented_fusion->completeFusion()) {
    FUSER_PERF_SCOPE("Segmenter::FusionSegmentGuard");
#ifndef NDEBUG
    num_original_exprs_ = fusion_->exprs().size();
    original_tvs_ = fusion_->allTvs();
#endif // NDEBUG
    lowered_precision_edges_ =
        segmented_fusion_->castInputOutputToLowerPrecision(
            segmented_fusion_->edges());
  }

  // Insert cast and narrow the fusion to a merged group of a and b
  FusionSegmentGuard(
      SegmentedFusion* segmented_fusion,
      SegmentedGroup* a,
      SegmentedGroup* b = nullptr)
      : segmented_fusion_(segmented_fusion),
        fusion_(segmented_fusion->completeFusion()) {
    FUSER_PERF_SCOPE("Segmenter::FusionSegmentGuard");
#ifndef NDEBUG
    num_original_exprs_ = fusion_->exprs().size();
    original_tvs_ = fusion_->allTvs();
#endif // NDEBUG

    // Cast inputs and outputs of a merged group consisting of a and
    // b.
    auto all_edges = getMergedProducerEdges(a, b, false);
    auto consumer_edges = getMergedConsumerEdges(a, b);
    std::copy(
        consumer_edges.begin(),
        consumer_edges.end(),
        std::back_inserter(all_edges));
    lowered_precision_edges_ =
        segmented_fusion_->castInputOutputToLowerPrecision(all_edges, {a, b});

    auto new_inputs = getAllInputs(a, b);
    auto new_outputs = getAllOutputs(a, b);

    narrowToNewSegment(new_inputs, new_outputs);
  }

  // Insert cast and narrow the fusion to a merged group of segmented_groups
  FusionSegmentGuard(
      SegmentedFusion* segmented_fusion,
      const std::vector<SegmentedGroup*>& segmented_groups)
      : segmented_fusion_(segmented_fusion),
        fusion_(segmented_fusion->completeFusion()) {
    FUSER_PERF_SCOPE("Segmenter::FusionSegmentGuard");
#ifndef NDEBUG
    num_original_exprs_ = fusion_->exprs().size();
    original_tvs_ = fusion_->allTvs();
#endif // NDEBUG

    // Cast inputs and outputs of a merged group consisting of
    // segmented_groups.
    auto all_edges = getAllEdges(segmented_groups);
    lowered_precision_edges_ =
        segmented_fusion_->castInputOutputToLowerPrecision(
            all_edges, segmented_groups);

    auto new_inputs = allInputsIfTrueElseOutputs(segmented_groups, true);
    auto new_outputs = allInputsIfTrueElseOutputs(segmented_groups, false);

    narrowToNewSegment(new_inputs, new_outputs);
  }

  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~FusionSegmentGuard() {
    FUSER_PERF_SCOPE("~Segmenter::FusionSegmentGuard");

    if (fusion_ == nullptr) {
      return;
    }

    restoreOriginalSegment();

    // Revert the cast
    if (segmented_fusion_ != nullptr && !lowered_precision_edges_.empty()) {
      segmented_fusion_->revertInputOutputPrecisionChanges(
          lowered_precision_edges_);
    }

#ifndef NDEBUG
    // fusion_ should now be equivalent to the original fusion. We
    // can't just compare Expr pointers as we replace Exprs. For
    // now, just make sure there are the same number of exprs.
    auto num_current_exprs = fusion_->exprs().size();
    NVF_ERROR(
        num_original_exprs_ == num_current_exprs,
        "Failed to revert temporary changes. Expected: ",
        num_original_exprs_,
        ", actual: ",
        num_current_exprs);
    auto current_tvs = fusion_->allTvs();
    NVF_ERROR(
        original_tvs_ == current_tvs, "Failed to revert temporary changes.");
#endif
  }

 private:
  void narrowToNewSegment(
      const std::vector<Val*>& new_inputs,
      const std::vector<Val*>& new_outputs) {
    NVF_ERROR(fusion_ != nullptr);

    old_inputs_ = fusion_->inputs();
    old_outputs_ = fusion_->outputs();

    for (auto old_inp : old_inputs_) {
      fusion_->removeInput(old_inp);
    }

    for (auto old_out : old_outputs_) {
      fusion_->removeOutput(old_out);
    }

    for (auto new_inp : new_inputs) {
      fusion_->addInput(new_inp);
    }

    for (auto new_out : new_outputs) {
      fusion_->addOutputInternal(new_out);
    }
  }

  void restoreOriginalSegment() {
    NVF_ERROR(fusion_ != nullptr);

    // If both old inputs and outpus are empty, narrowing must have
    // not been done
    if (old_inputs_.empty() && old_outputs_.empty()) {
      return;
    }

    auto cur_inputs = fusion_->inputs();
    for (auto new_inp : cur_inputs) {
      fusion_->removeInput(new_inp);
    }

    auto cur_outputs = fusion_->outputs();
    for (auto new_out : cur_outputs) {
      fusion_->removeOutput(new_out);
    }

    for (auto old_inp : old_inputs_) {
      fusion_->addInput(old_inp);
    }

    for (auto old_out : old_outputs_) {
      fusion_->addOutputInternal(old_out);
    }
  }

 private:
  SegmentedFusion* segmented_fusion_ = nullptr;
  Fusion* const fusion_ = nullptr;
  std::vector<Val*> old_inputs_;
  std::vector<Val*> old_outputs_;
  std::vector<SegmentedEdge*> lowered_precision_edges_;
#ifndef NDEBUG
  size_t num_original_exprs_ = 0;
  std::vector<TensorView*> original_tvs_;
#endif
};

SchedulerType tryMerge(
    SegmentedFusion* segmented_fusion,
    SchedulerRuntimeInfo& runtime_info,
    SegmentedGroup* a,
    SegmentedGroup* b = nullptr) {
  FusionSegmentGuard fsg(segmented_fusion, a, b);

  NVF_ERROR(
      !segmented_fusion->completeFusion()->unordered_exprs().empty(),
      "We shouldn't attempt to merge empty fusions. "
      "This might not indicate a bug, "
      "but it's definitely a change of world view that we should be aware of.");

  scheduler_debug_utils::canScheduleMessage(
      "\n**Segmenter** Considering fusion:\n",
      segmented_fusion->completeFusion());
  if (tryingToMergeSegmenterSet(segmented_fusion->completeFusion())) {
    return SchedulerType::None;
  }
  return Schedule::proposeHeuristics(
      segmented_fusion->completeFusion(), runtime_info);
}

SchedulerType tryMerge(
    SegmentedFusion* segmented_fusion,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<SegmentedGroup*>& segmented_groups) {
  FusionSegmentGuard fsg(segmented_fusion, segmented_groups);

  NVF_ERROR(
      !segmented_fusion->completeFusion()->unordered_exprs().empty(),
      "We shouldn't attempt to merge empty fusions. "
      "This might not indicate a bug, "
      "but it's definitely a change of world view that we should be aware of.");

  scheduler_debug_utils::canScheduleMessage(
      "\n**Segmenter** Considering fusion:\n",
      segmented_fusion->completeFusion());
  if (tryingToMergeSegmenterSet(segmented_fusion->completeFusion())) {
    return SchedulerType::None;
  }
  return Schedule::proposeHeuristics(
      segmented_fusion->completeFusion(), runtime_info);
}

// This function is for cleanup and
//  easier debugging. It shouldn't affect functionality
//  since segmented fusions are compiled with fusion
//  guard on the edges instead of actually looking
//  at the exprs.
void deDuplicateScalarExprs(std::vector<Expr*>& exprs) {
  // Exprs in SegmentedGroup are not ordered
  // so it is ok to insert them from unordered
  // set
  std::unordered_set<Expr*> scalar_expr_set;

  std::copy_if(
      exprs.begin(),
      exprs.end(),
      std::inserter(scalar_expr_set, scalar_expr_set.end()),
      [](Expr* expr) { return ir_utils::isScalarOp(expr); });

  if (!scalar_expr_set.empty()) {
    exprs.erase(
        std::remove_if(
            exprs.begin(),
            exprs.end(),
            [&scalar_expr_set](Expr* expr) {
              return scalar_expr_set.count(expr);
            }),
        exprs.end());
    exprs.insert(exprs.end(), scalar_expr_set.begin(), scalar_expr_set.end());
  }
}

} // namespace

std::vector<Expr*> SegmentedGroup::stablyOrderedExprs() const {
  // The time complexity is O((V+E)LogV) where V is the number of nodes and E
  // is the number of edges. LogV is due to the use of std::priority_queue to
  // break ties by the original order.
  std::unordered_map<Expr*, int64_t> original_order;
  for (auto&& [i, e] : enumerate(exprs())) {
    original_order[e] = i;
  }

  const std::unordered_set<Expr*> exprs_to_sort(exprs().begin(), exprs().end());

  std::vector<Expr*> ordered_exprs;
  ordered_exprs.reserve(exprs().size());

  auto compare_by_original_order = [&original_order](Expr* a, Expr* b) {
    // std::priority_queue implements a max heap, so this comparor returns true
    // when RHS is originally ordered before LHS.
    return original_order.at(a) > original_order.at(b);
  };
  std::priority_queue<
      Expr*,
      std::vector<Expr*>,
      decltype(compare_by_original_order)>
      to_visit(compare_by_original_order);

  std::unordered_map<Expr*, int64_t> num_producers;
  for (Expr* e : exprs()) {
    int64_t& n = num_producers[e];
    // Val::uses(), which is used later to decrement num_producers, contains
    // unique `Expr`s. Therefore, it's necessary to also dedup here.
    for (auto* in : VectorOfUniqueEntries<Val*>(e->inputs())) {
      Expr* def = in->definition();
      // Exprs in a SegmentedGroup come from the complete fusion, so the
      // producer/consumer of an Expr may be outside the group. Therefore, we
      // check exprs_to_sort.count.
      if (exprs_to_sort.count(def) > 0) {
        n++;
      }
    }

    if (n == 0) {
      to_visit.push(e);
    }
  }

  while (!to_visit.empty()) {
    Expr* e = to_visit.top();
    to_visit.pop();

    ordered_exprs.push_back(e);

    for (Val* out : e->outputs()) {
      for (Expr* user : out->uses()) {
        if (exprs_to_sort.count(user) > 0 && (--num_producers[user]) == 0) {
          to_visit.push(user);
        }
      }
    }
  }

  NVF_ERROR_EQ(
      ordered_exprs.size(), exprs().size(), "exprs() doesn't form a DAG.");

  return ordered_exprs;
}

std::optional<std::unique_ptr<HeuristicParams>> SegmentedGroup::
    getMaybeHeuristicParams(SchedulerRuntimeInfo& runtime_info) {
  FUSER_PERF_SCOPE("SegmentedFusion::getMaybeHeuristicParams");
  auto heuristic_data_cache =
      segmented_fusion_->getCachedHeuristicDataFor(this);
  if (!Schedule::canSchedule(
          schedulerType(),
          runtime_info.fusion(),
          runtime_info,
          heuristic_data_cache,
          /*skip_compile_time_checks=*/true)) {
    return std::nullopt;
  }
  return SchedulerEntry::makeSchedulerInstance(schedulerType())
      ->computeHeuristics(
          runtime_info.fusion(), runtime_info, heuristic_data_cache);
}

void SegmentedGroup::resetExprList() {
  auto input_group_vec = getAllInputs(this);
  std::unordered_set<Val*> input_group_set(
      input_group_vec.begin(), input_group_vec.end());
  auto expr_set =
      DependencyCheck::getAllExprsBetween(input_group_set, getAllOutputs(this));
  exprs_ = std::vector<Expr*>(expr_set.begin(), expr_set.end());
}

// Custom merge node passes:
//  These passes are added at the beginning or the end of
//  the node merging process to direct the heuristics of
//  node merging process
//
//  Should consider generalization and make a proper interface
//   if we have more merge node heuristics like this

//! Translate Welford
//!
//! This pass can be inserted at any stages of segmentation,
//!  and it tries to replace welford ops with persistent
//!  mean and var ops.
//!
//! The checking of feasibility of persistent kernels
//!  is through normalization schedulers. The general idea
//!  is to first try to translate on a copy, and see if
//!  normalization scheduler is willing to produce a
//!  persistent kernel.
//!
//! For complete fusion this pass checks if all the
//!  welford ops can be translated simultaneously to
//!  produce a persistent normalization kernel and
//!  will perform translation if checks pass.
//!
//! For segmented fusion, same check is performed within
//!  each segmented group to collect applicable welford ops,
//!  and actual translations are performed on the complete
//!  fusion after all the checks are done.
class TranslateApplicableWelford {
 public:
  //! Try translation on each segmented group of
  //!  given segmented fusion
  //!  returns true if any welford has been translated
  static bool run(
      SegmentedFusion* segmented_fusion,
      const KernelArgumentHolder& runtime_inputs) {
    FUSER_PERF_SCOPE("TranslateApplicableWelford::run");
    TranslateApplicableWelford translate_welford(
        segmented_fusion, runtime_inputs);
    return translate_welford.translated_any_welford_;
  }

  //! Try translation on complete fusion,
  //!  returns true if any welford has been translated
  static bool run(Fusion* fusion, const KernelArgumentHolder& runtime_inputs) {
    FUSER_PERF_SCOPE("TranslateApplicableWelford::run");
    TranslateApplicableWelford translate_welford(fusion, runtime_inputs);
    return translate_welford.translated_any_welford_;
  }

 private:
  explicit TranslateApplicableWelford(
      SegmentedFusion* segmented_fusion,
      const KernelArgumentHolder& runtime_inputs);

  explicit TranslateApplicableWelford(
      Fusion* fusion,
      const KernelArgumentHolder& runtime_inputs);

  //! Given vector of welford ops from the same fusion,
  //!  checks if translating all of them result in a
  //!  persistent normalization kernel by try-runs on
  //!  a test copy of the original fusion.
  //!
  //! Supported use cases are either un-segmented fusion,
  //!  or all the given welfords are within the same
  //!  segmented group. In the latter case, the segmented
  //!  group containing all the welford ops needs to be
  //!  provided.
  bool wouldTranslateToPersistent(
      const std::vector<WelfordOp*>& original_welfords,
      SegmentedGroup* group = nullptr);

  //! Translate the given welford op into separate
  //! average and standard deviation calculation.
  void translateSingleWelford(WelfordOp* welford);

  //! Utility to test if a translated fusion
  //!  gives a persistent kernel. Uses normalization
  //!  scheduler to do the test.
  bool isValidPersistentFusion(
      Fusion* translated_fusion,
      SchedulerRuntimeInfo& runtime_info);

 private:
  //! Indicates any translation happened.
  bool translated_any_welford_ = false;

  //! a reference to global fusion runtime inputs
  const KernelArgumentHolder& runtime_inputs_;

  //! For translation within group only,
  //!  group boundary at test copy
  //! (see wouldTranslateToPersistent implementation )
  std::vector<Val*> test_group_inputs_;
  std::vector<Val*> test_group_outputs_;
};

TranslateApplicableWelford::TranslateApplicableWelford(
    Fusion* fusion,
    const KernelArgumentHolder& runtime_inputs)
    : runtime_inputs_(runtime_inputs) {
  auto exprs = fusion->exprs();
  std::vector<WelfordOp*> original_welfords(
      ir_utils::filterByType<WelfordOp>(exprs).begin(),
      ir_utils::filterByType<WelfordOp>(exprs).end());

  if (wouldTranslateToPersistent(original_welfords)) {
    for (auto welford : original_welfords) {
      translateSingleWelford(welford);
    }
    translated_any_welford_ = true;
  }
}

TranslateApplicableWelford::TranslateApplicableWelford(
    SegmentedFusion* segmented_fusion,
    const KernelArgumentHolder& runtime_inputs)
    : runtime_inputs_(runtime_inputs) {
  std::vector<SegmentedGroup*> translated_groups;

  {
    // Cast inputs and outputs to lower precision before
    // trying the welford conversion. This could affect the scheduling
    // decision as some runtime parameters such as the vectorization
    // factor and the persistent buffer size would change.
    //
    // To revert the temporary changes before translating welford,
    // this guard is placed inside its own scope. Reverting the
    // changes before the translation is not necessary for
    // correctness, but the sanity check of the FusionSegmentGuard
    // dtor would complain as the number of expressions would change.
    FusionSegmentGuard cast_guard(segmented_fusion);

    // Find welfords that can be translated in each group
    for (auto group : segmented_fusion->groups()) {
      std::vector<WelfordOp*> welford_in_group(
          ir_utils::filterByType<WelfordOp>(group->exprs()).begin(),
          ir_utils::filterByType<WelfordOp>(group->exprs()).end());

      if (wouldTranslateToPersistent(welford_in_group, group)) {
        translated_groups.push_back(group);
      }
    }
  }

  // Actually translate the welford ops
  // and record all the vals that have been
  // replaced by the translation.
  for (auto group : translated_groups) {
    std::vector<WelfordOp*> welford_in_group(
        ir_utils::filterByType<WelfordOp>(group->exprs()).begin(),
        ir_utils::filterByType<WelfordOp>(group->exprs()).end());
    for (auto welford : welford_in_group) {
      translateSingleWelford(welford);
      translated_any_welford_ = true;
    }
  }
}

bool TranslateApplicableWelford::isValidPersistentFusion(
    Fusion* translated_fusion,
    SchedulerRuntimeInfo& runtime_info) {
  // Check reduciton type and get the appropriate heuristic.
  auto reduction_type =
      reduction_scheduler_utils::getReductionType(translated_fusion);
  if (reduction_type == reduction_scheduler_utils::ReductionType::None) {
    return false;
  }
  auto persistent_sh =
      normalization_scheduler_utils::getPersistentHeuristicFor(reduction_type);

  if (!Schedule::canSchedule(persistent_sh, translated_fusion, runtime_info)) {
    return false;
  }
  auto scheduler = SchedulerEntry::makeSchedulerInstance(persistent_sh);
  auto heuristic_params =
      scheduler->computeHeuristics(translated_fusion, runtime_info);

  // Translate welford to two-pass enhances performance for block
  // reductions by reducing instructions and the impact of an extra block
  // synchronization has negligible overhead.
  // However, when it comes to cross grid reduction, the additional grid
  // synchronization carries substantial overhead and does not yield any
  // performance gains.
  return heuristic_params->as<ReductionParams>()->persistent_kernel &&
      !heuristic_params->as<ReductionParams>()->cross_grid_outer_reduction;
}

// Note that when segmented it is assumed that insertion of lower
// precision cast has already been done
bool TranslateApplicableWelford::wouldTranslateToPersistent(
    const std::vector<WelfordOp*>& original_welfords,
    SegmentedGroup* group) {
  if (original_welfords.empty()) {
    return false;
  }

  // Make sure all welford inputs are not already statistics, e.g.
  // FusionSqueezeOnlyWelford_CUDA
  for (auto welford : original_welfords) {
    if (!welford->inN()->isOneInt()) {
      return false;
    }
  }

  // Make sure all welford ops come from the same complete fusion
  auto fusion = original_welfords[0]->fusion();
  NVF_ERROR(
      std::all_of(
          original_welfords.begin(),
          original_welfords.end(),
          [fusion](WelfordOp* welford) { return welford->fusion() == fusion; }),
      "Welfords in given vector not in the same fusion");

  // Make initial `in-progress copy`
  auto test_copy = std::make_unique<Fusion>();
  auto original_to_test_map = Fusion::copy(fusion, test_copy.get());

  std::vector<WelfordOp*> copied_welfords;
  std::transform(
      original_welfords.begin(),
      original_welfords.end(),
      std::back_inserter(copied_welfords),
      [&original_to_test_map](auto welford) {
        return original_to_test_map.clone(welford);
      });
  // Copied welfords will be invalidated on translation, but Vals will be
  // reused, keep a reference to them.
  std::vector<Val*> welford_avgs;
  std::vector<Val*> welford_vars;
  for (auto welford : copied_welfords) {
    welford_avgs.push_back(welford->outAvg());
    welford_vars.push_back(welford->outVar());
  }

  // Translate the welford ops
  for (auto welford_to_translate : copied_welfords) {
    translateSingleWelford(welford_to_translate);
  }

  SchedulerRuntimeInfo runtime_info(test_copy.get(), runtime_inputs_);
  // If we are looking at a segment of fusion,
  //  we maintain the segmented group boundary,
  //  one set for in_progress copy and one set
  //  for `test copy`
  if (group != nullptr) {
    auto original_inputs = getAllInputs(group);
    auto original_outputs = getAllOutputs(group);
    test_group_inputs_.clear();
    test_group_outputs_.clear();
    std::transform(
        original_inputs.begin(),
        original_inputs.end(),
        std::back_inserter(test_group_inputs_),
        [&original_to_test_map](Val* in) {
          return original_to_test_map.clone(in);
        });
    std::transform(
        original_outputs.begin(),
        original_outputs.end(),
        std::back_inserter(test_group_outputs_),
        [&original_to_test_map](Val* out) {
          return original_to_test_map.clone(out);
        });

    // If only average is used from welford, we should still translate, but we
    // might not detect persistence if variance isn't actually used/marked as an
    // output in the test.
    for (auto outs_i : arange(welford_avgs.size())) {
      auto avg = welford_avgs[outs_i];
      auto var = welford_vars[outs_i];
      if (avg->uses().empty()) {
        test_group_outputs_.push_back(avg);
      }

      if (var->uses().empty()) {
        test_group_outputs_.push_back(var);
      }
    }

    // Temporarily localize test copy around
    //  the group boundary
    FusionSegmentGuard fsg(
        test_copy.get(), test_group_inputs_, test_group_outputs_);

    // Test if the translated copy is persistent
    return isValidPersistentFusion(test_copy.get(), runtime_info);
  }
  // In the case where we work on un-segmented
  //  fusion, no group boundary logic, just
  //  translate and test.
  return isValidPersistentFusion(test_copy.get(), runtime_info);
}

void TranslateApplicableWelford::translateSingleWelford(WelfordOp* welford) {
  auto fusion = welford->fusion();
  FusionGuard fg(fusion);
  // Only support translation of welford ops that
  // doesn't take inputs that are already statistics,
  // i.e. an r-factor product.
  // This translation works on un-scheduled fusions so
  //  shouldn't expect to see this.
  NVF_ERROR(welford->inN()->isOneInt());

  // Grab the inputs and outputs of the welford
  auto in_val = welford->in()->as<TensorView>();
  auto out_avg = welford->outAvg()->as<TensorView>();
  auto out_var = welford->outVar()->as<TensorView>();
  auto out_N = welford->outN()->as<TensorView>();

  fusion->removeExpr(welford);
  // Not safe to use welford anymore
  welford = nullptr;

  // Create normalization based welford graph
  //  largely taken from batchnorm cpp benchmark
  const auto& in_logical =
      TensorDomain::noReductions(in_val->getLogicalDomain());
  const auto& out_logical = out_avg->getLogicalDomain();
  std::vector<int64_t> red_axes;

  NVF_ERROR(
      in_logical.size() == out_logical.size(),
      "Invalid root domains of Welford input and output.",
      " Input: ",
      ir_utils::toString(in_logical),
      ". Output: ",
      ir_utils::toString(out_logical));

  // Create scalar version of the feature element
  //  counting.
  Val* num_features = IrBuilder::create<Val>(1.0);
  std::vector<bool> broadcast_mask(in_logical.size(), false);
  for (const auto i : arange((int64_t)in_logical.size())) {
    if (out_logical.at(i)->isReduction()) {
      red_axes.push_back(i);
      broadcast_mask[i] = true;
      num_features = mul(num_features, out_logical.at(i)->extent());
    }
  }

  // Build a normalization expression group that is
  //  equivalent to a welford operation.
  auto x_sum = sum(in_val, red_axes);
  IrBuilder::create<BinaryOp>(BinaryOpType::Div, out_avg, x_sum, num_features);

  // welford.avg may be broadcast. Reuse it if found.
  TensorView* x_avg_bcast = nullptr;
  for (auto& use_expr : out_avg->uses()) {
    if (auto bcast = dynamic_cast<BroadcastOp*>(use_expr)) {
      if (bcast->getBroadcastDimFlags() == broadcast_mask) {
        // Same broadcast found.
        x_avg_bcast = bcast->out()->as<TensorView>();
        break;
      }
    }
  }

  if (x_avg_bcast == nullptr) {
    x_avg_bcast = broadcast(out_avg, broadcast_mask);
  }
  TensorView* x_mean_sub = sub(in_val, x_avg_bcast);

  auto x_mean_sub_pow = mul(x_mean_sub, x_mean_sub);
  IrBuilder::create<ReductionOp>(
      BinaryOpType::Add, IrBuilder::create<Val>(0.0), out_var, x_mean_sub_pow);
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out_N, num_features);

  // out_avg, out_N are now outputs of a pointwise ops and we
  //  need to clear out its reduction domains.
  out_avg->clearReductionIterDomains();
  out_N->clearReductionIterDomains();
}

bool SegmentCandidateFinder::translateWelfordInFusion(
    Fusion* fusion,
    const KernelArgumentHolder& runtime_inputs) {
  return TranslateApplicableWelford::run(fusion, runtime_inputs);
}

void SegmentCandidateFinder::validateIfDebug(bool require_disjoint) {
#ifndef NDEBUG
  resetLevels();
  if (require_disjoint) {
    segmented_fusion_->validateDisjoint();
  }
#endif // NDEBUG
}

//! CombineReductions:
//!  This pass works before the main merge node process
//!    It identifies reduction operations that can be combined
//!    together to form a normalization kernel.
//!  Two reductions are considered the same type if they have
//!   the same root domain length, and the reduction axis are the same.
//!   This pass tries to merge nodes with the same reduction type based
//!   on the graph structure.
class CombineReductions {
  using GroupVec = std::vector<SegmentedGroup*>;
  class ReductionSignature;

 public:
  static void run(SegmentCandidateFinder* segment_candidate_finder) {
    CombineReductions combine_reductions(segment_candidate_finder);
  }
  static bool shouldRun(SegmentCandidateFinder* segment_candidate_finder);

 private:
  CombineReductions(SegmentCandidateFinder* segment_candidate_finder)
      : segment_candidate_finder_(segment_candidate_finder) {
    // Run pass over the segments
    // Collect segmented groups with reductions in them,
    //  Assuming running before any merge happened, so
    //  should see exactly one reduction in each group
    for (auto group : segment_candidate_finder_->groups()) {
      if (auto rop_signature =
              ReductionSignature::makeReductionSignature(group)) {
        groups_with_reductions_.push_back(group);
        // Check if this reduction signature is one that we have seen before
        auto signature_match_it = std::find_if(
            known_reduction_signatures_.begin(),
            known_reduction_signatures_.end(),
            [&rop_signature](auto& know_signature) {
              return know_signature->sameAs(rop_signature.get());
            });
        // Unmatched: Create a new signature entry if not known
        if (signature_match_it == known_reduction_signatures_.end()) {
          group_reduction_signature_map_[group] = rop_signature.get();
          known_reduction_signatures_.emplace_back(std::move(rop_signature));
        } else {
          // Matched known signature: Mark that this groups belongs to know
          // signature
          group_reduction_signature_map_[group] = signature_match_it->get();
        }
      }
    }

    // Keep trying to merge groups with compatible reductions and compatible
    // paths
    //  until no more merge opportunity can be identified
    bool merged_groups = true;
    while (merged_groups) {
      merged_groups = false;

      // Merge one pair of reduction groups at a time, and need
      //  the pass to update dependency info along the way to avoid cycles
      for (const auto first_group_index :
           arange(groups_with_reductions_.size())) {
        if (merged_groups) {
          // Need to break and re-enter this loop because
          // groups_with_reductions_ will be updated
          break;
        }

        // Select one of the group to merge and get its reduction signature
        auto first_group = groups_with_reductions_[first_group_index];
        auto first_group_signature =
            group_reduction_signature_map_.at(first_group);

        for (const auto second_group_index :
             arange(first_group_index + 1, groups_with_reductions_.size())) {
          if (merged_groups) {
            // Need to break and re-enter this loop because
            // groups_with_reductions_ will be updated
            break;
          }
          auto second_group = groups_with_reductions_[second_group_index];
          auto second_group_signature =
              group_reduction_signature_map_.at(second_group);

          // Cannot merge if their signatures are not the same
          if (!first_group_signature->sameAs(second_group_signature)) {
            continue;
          }

          // first try a vertical merge
          merged_groups =
              verticalReductionMerge(first_group, second_group) != nullptr;
          if (!merged_groups) {
            // vertical merge didn't happen, try a horizontal merge
            merged_groups =
                horizontalReductionMerge(first_group, second_group) != nullptr;
          }
        }
      }
    }
  }

  //! Merge a vertical pair of producers and consumers,
  //!  the resulting group will include all nodes that are
  //!  also consumers of producer and producers of consumer,
  //!  i.e. values between the given producer-consumer pair.
  //!  Can be proven that:
  //!   1. Including all of these nodes will be cycle-free
  //!   2. These nodes are the minimal set of nodes to include if
  //!  for producer-consumer pair to be in the same group cycle-free
  //!
  //!  Returns nullptr if such merge cannot be achieved.
  //!  Reasons for not merging will include:
  //!   1. Given groups do not form producer-consumer pair
  //!   2. Merge will create cycle on the graph
  //!   3. The merged joined group cannot be scheduled
  SegmentedGroup* verticalReductionMerge(
      SegmentedGroup* first_group,
      SegmentedGroup* second_group) {
    // This is part of ReductionCombine pass, and we should only call this
    // function on a pair of reduction/normalization groups
    NVF_ERROR(group_reduction_signature_map_.at(first_group)
                  ->sameAs(group_reduction_signature_map_.at(second_group)));
    NVF_ERROR(first_group != second_group);
    // Get the group dependency data from segment finder
    auto dependency_analysis = segment_candidate_finder_->getGroupDependency();

    // Check producer-consumer relationship
    SegmentedGroup* producer = nullptr;
    SegmentedGroup* consumer = nullptr;
    if (dependency_analysis->isConsumerOf(first_group, second_group)) {
      producer = second_group;
      consumer = first_group;
    } else if (dependency_analysis->isProducerOf(first_group, second_group)) {
      producer = first_group;
      consumer = second_group;
    } else {
      // Given groups aren't producer-consumer pair, won't merge
      return nullptr;
    }

    // Collect all groups that we need to merge along with the producer and
    // consumer
    auto all_groups_to_merge =
        getValidMinVerticalMergedGroupSet(producer, consumer);

    if (all_groups_to_merge.empty()) {
      // The vertical paths from producer to consumer have in-compatible
      // reductions
      //   so this vertical merge cannot be done.
      return nullptr;
    }

    // TODO: this step would not be deterministic, because valuesBetween isn't
    //       could fix this by a topological order
    std::vector<SegmentedGroup*> all_groups_to_merge_vec(
        all_groups_to_merge.begin(), all_groups_to_merge.end());

    // Final sanity check: the merged group can actually be scheduled
    if (tryMerge(
            segment_candidate_finder_->segmented_fusion_.get(),
            segment_candidate_finder_->runtimeInfo(),
            all_groups_to_merge_vec) == SchedulerType::None) {
      return nullptr;
    }

    // Merge this group
    auto joined_group =
        segment_candidate_finder_->mergeAllGivenGroups(all_groups_to_merge_vec);

    // Update dependency analysis
    dependency_analysis->mergeGroups(all_groups_to_merge, joined_group);

    // Update the reduction groups that are merged
    groups_with_reductions_.push_back(joined_group);
    group_reduction_signature_map_[joined_group] =
        group_reduction_signature_map_.at(first_group);
    groups_with_reductions_.erase(
        std::remove_if(
            groups_with_reductions_.begin(),
            groups_with_reductions_.end(),
            [&all_groups_to_merge](SegmentedGroup* group) {
              return all_groups_to_merge.has(group);
            }),
        groups_with_reductions_.end());

    return joined_group;
  }

  //! Horizontal reduction merging:
  //!  merge two horizontal groups with reduction expressions to make a joined
  //!  normalization group. A pair of horizontal groups are ones that are not
  //!  a producer-consumer pair, and share either a common producer or a common
  //!  consumer.
  //!
  //!  TODO: This implementation looks at common producers only, since common
  //!  consumers are not computed easily with current dependency analysis.
  SegmentedGroup* horizontalReductionMerge(
      SegmentedGroup* first_group,
      SegmentedGroup* second_group) {
    // This is part of ReductionCombine pass, and we should only call this
    // function on a pair of
    //  reduction/normalization groups
    NVF_ERROR(group_reduction_signature_map_.at(first_group)
                  ->sameAs(group_reduction_signature_map_.at(second_group)));
    NVF_ERROR(first_group != second_group);

    auto dependency_analysis = segment_candidate_finder_->getGroupDependency();

    // Check that the two groups are not producer-consumer's
    if (dependency_analysis->isConsumerOf(first_group, second_group) ||
        dependency_analysis->isProducerOf(first_group, second_group)) {
      // This merge pass will not handle producer-consumer pairs
      return nullptr;
    }

    // Get common producers of the two group
    auto common_producers_set =
        dependency_analysis->getCommonProducersOf({first_group, second_group});
    if (common_producers_set.empty()) {
      // The given pair doesn't have a common producer.
      //  Either they have a common consumer, which we don't handle for now,
      //  or maybe the two given groups are not connected.
      return nullptr;
    }

    // We are looking for a very specific patterns here. The cases that this
    //  pattern will not capture are ones that reductions of different
    //  signatures are so interleaved that we cannot find a clear cut as
    //  explained below, without graph rewriting. Some graph re-writing on the
    //  segmented groups level could provide extra merging opportunities for
    //  free, which could be part of next step.
    //
    // The specific pattern we look for contains a common producer P with
    // immediate consumers C1, C2 such that all paths from C1 to first_group and
    // all paths from C2 to second_group won't hit a reduction with a different
    // signature.

    // Topologically sort the common producers and start with the topologically
    // minimal,
    //  i.e. one that are closest to the two groups. This will cut the search
    //  space.
    std::vector<SegmentedGroup*> common_producers;
    for (auto producer : common_producers_set) {
      if (!std::any_of(
              common_producers_set.begin(),
              common_producers_set.end(),
              [dependency_analysis, producer](SegmentedGroup* group) {
                return dependency_analysis->isProducerOf(producer, group);
              })) {
        common_producers.push_back(producer);
      }
    }

    // Visit the common producers found, starting from topologically minimum,
    // i.e. the ones closer to the groups
    for (auto common_producer : common_producers) {
      // Visit this common producer
      // Use a double loop in case the schedulers like some patterns
      //  better than the other
      for (auto first_consumer_edge : common_producer->consumer_edges) {
        auto producer_of_first_group = first_consumer_edge->to;
        auto to_merge_with_first_group = getValidMinVerticalMergedGroupSet(
            producer_of_first_group, first_group);
        if (to_merge_with_first_group.empty()) {
          // There's no valid merge path from this consumer of common producer,
          //  either due to a conflicting reduction signature, or simply there's
          //  no path to first group
          continue;
        }
        NVF_ERROR(!dependency_analysis->isProducerOf(
            producer_of_first_group, second_group));
        for (auto second_consumer_edge : common_producer->consumer_edges) {
          auto producer_of_second_group = second_consumer_edge->to;
          auto to_merge_with_second_group = getValidMinVerticalMergedGroupSet(
              producer_of_second_group, second_group);
          if (to_merge_with_second_group.empty()) {
            // There's no valid merge path from this consumer of common
            // producer,
            //  either due to a conflicting reduction signature, or simply
            //  there's no path to second group
            continue;
          }
          NVF_ERROR(!dependency_analysis->isProducerOf(
              producer_of_second_group, first_group));
          // At this point we should have a pair of valid candidates,final check
          // is to see if the combined group
          //  can be scheduled by schedulers
          // merge the two paths and de-duplicate,
          //  re-using container here with to_merge_with_second_group
          auto& groups_to_merge_set = to_merge_with_second_group;
          groups_to_merge_set.insert(
              to_merge_with_first_group.begin(),
              to_merge_with_first_group.end());
          std::vector<SegmentedGroup*> groups_to_merge_vec(
              groups_to_merge_set.begin(), groups_to_merge_set.end());
          if (tryMerge(
                  segment_candidate_finder_->segmented_fusion_.get(),
                  segment_candidate_finder_->runtimeInfo(),
                  groups_to_merge_vec) != SchedulerType::None) {
            // Found a valid horizontal merge, want to proceed with merging here
            auto joined_group = segment_candidate_finder_->mergeAllGivenGroups(
                groups_to_merge_vec);
            dependency_analysis->mergeGroups(groups_to_merge_set, joined_group);

            groups_with_reductions_.push_back(joined_group);
            group_reduction_signature_map_[joined_group] =
                group_reduction_signature_map_.at(first_group);
            groups_with_reductions_.erase(
                std::remove_if(
                    groups_with_reductions_.begin(),
                    groups_with_reductions_.end(),
                    [&groups_to_merge_set](SegmentedGroup* group) {
                      return groups_to_merge_set.has(group);
                    }),
                groups_with_reductions_.end());
            return joined_group;
          }
        }
      }
    }

    // Searched all possibilities and there is no valid horizontal merge pattern
    //  found.
    return nullptr;
  }

  //! This is a utility method that is used in both vertical merging and
  //! horizontal merging.
  //!  It is used to identify the smallest set of groups to merge vertically
  //!  involving the
  //!   two given nodes.
  //!  Given a pair of nodes this utility distinguishes 3 cases:
  //!   1. if maybe_producer is the same as maybe_consumer, then returns
  //!   {maybe_producer}
  //!   2. if maybe_producer is actually a producer of consumer, returns a set
  //!   containing
  //!     the smallest merged group that would contain producer and consumer and
  //!     would not introduce a cycle. Returns empty set if such group has
  //!     a conflicting reduction signature.
  //!   3. returns empty set if neither conditions above apply.
  GroupSet getValidMinVerticalMergedGroupSet(
      SegmentedGroup* maybe_producer,
      SegmentedGroup* maybe_consumer) {
    auto dependency_analysis = segment_candidate_finder_->getGroupDependency();
    if (maybe_consumer == maybe_producer) {
      // maybe producer is the same as maybe_consumer
      return {maybe_consumer};
    } else if (dependency_analysis->isConsumerOf(
                   maybe_consumer, maybe_producer)) {
      auto groups_to_check =
          dependency_analysis->valuesBetween(maybe_producer, maybe_consumer);
      groups_to_check.pushBack(maybe_producer);
      groups_to_check.pushBack(maybe_consumer);

      // Check that either no group has a reduction or all groups have the same
      // reduction signature
      ReductionSignature* reduction_signature = nullptr;

      // Iterate through the minimal group set to see if any conflicts
      for (auto group : groups_to_check) {
        // Check that this group does not involve a output edge contraction
        //  This pass is intended to be a pre-merging pass. Since contracting an
        //   output edge does not generate much saving of global memory access
        //   we want to postpone merging these edges till the very final pass
        for (auto producer_edge_of_group : group->producer_edges) {
          if (groups_to_check.has(producer_edge_of_group->from) &&
              producer_edge_of_group->val->isFusionOutput()) {
            return {};
          }
        }
        for (auto consumer_edge_of_group : group->consumer_edges) {
          if (groups_to_check.has(consumer_edge_of_group->to) &&
              consumer_edge_of_group->val->isFusionOutput()) {
            return {};
          }
        }

        // Check that this group does not have a conflicting reduction signature
        if (group_reduction_signature_map_.count(group)) {
          if (reduction_signature != nullptr) {
            if (!group_reduction_signature_map_.at(group)->sameAs(
                    reduction_signature)) {
              // Found a conflict in reduction signature, cannot do a vertical
              // merge
              return {};
            }
          } else {
            reduction_signature = group_reduction_signature_map_.at(group);
          }
        }
      }
      return groups_to_check;
    }
    // maybe producer is not a producer of maybe consumer
    return {};
  }

 private:
  SegmentCandidateFinder* segment_candidate_finder_;

  // Wrapper class for reduction type
  //  Assuming there wouldn't be too many of them
  //  so won't need to create a hash
  // TODO:
  //   Want to reconsider this for transpose operations,
  //   need refactoring to handle reduction fusions across a transpose operation
  class ReductionSignature {
   public:
    bool sameAs(const ReductionSignature* reduction_signature) {
      if (reduction_signature == this) {
        return true;
      }

      if (root_domain_size_ != reduction_signature->root_domain_size_ ||
          has_reduction_ != reduction_signature->has_reduction_ ||
          reduction_axes_.size() !=
              reduction_signature->reduction_axes_.size()) {
        return false;
      }

      for (const auto i : arange(reduction_axes_.size())) {
        if (reduction_axes_[i] != reduction_signature->reduction_axes_[i]) {
          return false;
        }
      }

      return true;
    }

    bool sameAs(const ReductionSignature& reduction_signature) {
      return sameAs(&reduction_signature);
    }

    bool hasReduction() const {
      return has_reduction_;
    }

    static std::unique_ptr<ReductionSignature> makeReductionSignature(
        SegmentedGroup* group) {
      std::unique_ptr<ReductionSignature> signature = nullptr;

      for (auto expr : group->exprs()) {
        std::unique_ptr<ReductionSignature> new_signature = nullptr;

        if (auto rop = dynamic_cast<ReductionOp*>(expr)) {
          new_signature = std::make_unique<ReductionSignature>(rop);
        }
        if (auto wop = dynamic_cast<WelfordOp*>(expr)) {
          new_signature = std::make_unique<ReductionSignature>(wop);
        }

        if (new_signature != nullptr) {
          NVF_ERROR(
              signature == nullptr || !signature->has_reduction_ ||
                  !new_signature->has_reduction_ ||
                  signature->sameAs(new_signature.get()),
              "Conflicting signature found in this group");
          signature = std::move(new_signature);
        }
      }
      return signature;
    }

    template <typename REDUCTION = ReductionOp>
    ReductionSignature(REDUCTION* rop) {
      auto out_tv = rop->out()->template as<TensorView>();
      NVF_ERROR(out_tv != nullptr);
      has_reduction_ = out_tv->hasReduction();
      auto& root_domain = out_tv->getLogicalDomain();
      root_domain_size_ = root_domain.size();

      for (const auto i : arange(root_domain_size_)) {
        if (root_domain[i]->isReduction()) {
          reduction_axes_.push_back(i);
        }
      }
    }

   private:
    int64_t root_domain_size_ = 0;
    std::vector<int64_t> reduction_axes_;
    bool has_reduction_ = false;
  };

  //! Keeps track of groups with reduction expressions,
  //!  using a vector here to maintain a deterministic ordering
  GroupVec groups_with_reductions_;

  //! Maps groups to their corresponding signature type
  std::unordered_map<SegmentedGroup*, ReductionSignature*>
      group_reduction_signature_map_;

  //! Maintains all reduction signatures seen in the segmented fusion
  std::vector<std::unique_ptr<ReductionSignature>> known_reduction_signatures_;
};

//! This is to be checked
bool CombineReductions::shouldRun(
    SegmentCandidateFinder* segment_candidate_finder) {
  std::vector<std::unique_ptr<ReductionSignature>> known_reductions;
  // Iterate over group segments we have before segment candidate finder
  //  tries to merge any groups
  for (auto group : segment_candidate_finder->groups()) {
    if (auto reduction_signature =
            ReductionSignature::makeReductionSignature(group)) {
      if (reduction_signature->hasReduction() &&
          std::any_of(
              known_reductions.begin(),
              known_reductions.end(),
              [&reduction_signature](auto& know_signature) {
                return know_signature->sameAs(reduction_signature.get());
              })) {
        // Found two reductions with the same signature, run pass
        return true;
      }
      known_reductions.emplace_back(std::move(reduction_signature));
    }
  }
  return false;
}

// This preprocessing attempts to find groups of exprs consist of an
// up-cast, followed by some ops and ended by a downcast. It is highly
// likely that such sequences of ops should never be segmented
// out. This is particularly commonly seen in fusions given by Thunder
// as it inserts fine-grained downcasting and upcasting ops. Without
// this preprocessing, a fusion may be segmented right after an
// up-cast op, for example, and in fact it happened quite frequently
// in some of the RoPE cases. This preprocessing does not completely
// avoid such segmentation boundaries, but it should become less
// likely. See also https://github.com/NVIDIA/Fuser/pull/3699.
class MergeUpAndDownCast {
 public:
  static void run(SegmentCandidateFinder* segment_candidate_finder) {
    MergeUpAndDownCast group_cast(segment_candidate_finder);
  }

 private:
  MergeUpAndDownCast(SegmentCandidateFinder* segment_candidate_finder)
      : segment_candidate_finder_(segment_candidate_finder) {
    merge();
  }

  void merge() {
    bool merged = true;
    while (merged) {
      merged = false;
      std::unordered_set<SegmentedGroup*> considered_groups;

      for (SegmentedGroup* group : segment_candidate_finder_->groups()) {
        // If the group is an up-cast group, see if there's a
        // candidate group starting with the group.
        if (!isUpCast(group) || considered_groups.count(group)) {
          continue;
        }

        auto groups_to_merge = getCandidateCastGroup(group);
        if (groups_to_merge.size() < 2) {
          continue;
        }

        for (auto group : groups_to_merge) {
          considered_groups.insert(group);
        }

        // Try merging the detected group
        if (mergeCastGroup(groups_to_merge)) {
          merged = true;
          break;
        }
      }
    }
  }

  // Try to detect a set of groups that could be merged as a cast
  // group. The analysis starts with an initial group that solely
  // consists of an up-cast expression. From the initial group, it
  // traverses its neighbor groups. If the group is an down-cast group,
  // it only traverses through the consumer edges. If it's an up-cast
  // group, it only traverses through the producer edges.
  //
  // Additionaly, this traversal has several safeguards to keep the
  // DAG property intact:
  //
  // - For a given group, it does not visit its consumers if it has
  //   multiple consumers, even if the group is not a down-cast
  //   group.
  // - Similarly, it does not visit a producer if the producer has
  //   multiple cosumers.
  //
  // The basic form of this set of groups should look like an up-cast
  // group, followed by some op groups and ended by a down-cast
  // group. However, it is not always the case because of the above
  // safeguards. For example, the following groups would be detected
  // as a cast group.
  //
  // t1 = bf16ToFp32(t0)
  // t2 = neg(t1)
  // t3 = sin(t2)
  // t4 = cos(t2)
  // t5 = fp32ToBf16(t3)
  // t6 = fp32ToBf16(t4)
  //
  // In this case, t1 and t2 would be detected as a candidate group,
  // but t3 and t4 would not be included. While we could certainly
  // extend the analysis, it would need to make sure the DAG property
  // is not violated.
  std::vector<SegmentedGroup*> getCandidateCastGroup(
      SegmentedGroup* initial_group) {
    std::vector<SegmentedGroup*> groups_to_merge;
    std::unordered_set<SegmentedGroup*> groups_to_merge_set;

    std::deque<SegmentedGroup*> to_visit;
    to_visit.push_back(initial_group);

    while (!to_visit.empty()) {
      SegmentedGroup* group = to_visit.front();
      to_visit.pop_front();

      if (group->exprs().empty()) {
        continue;
      }

      if (groups_to_merge_set.count(group)) {
        continue;
      }

      // For simplicity, all groups are assumed to be the initial
      // single-expr groups. Skip if not

      groups_to_merge.push_back(group);
      groups_to_merge_set.insert(group);

      // Consumer traversal. Stop if this group is a down cast
      // group. Also stop if there are multiple consumer edges to
      // simplify keeping the DAG property.
      if (!isDownCast(group) && group->consumer_edges.size() == 1) {
        auto consumer_edge = group->consumer_edges.at(0);
        SegmentedGroup* consumer_group = consumer_edge->to;
        if (!groups_to_merge_set.count(consumer_group)) {
          to_visit.push_back(consumer_group);
        }
      }

      if (!isUpCast(group)) {
        for (const auto producer_edge : group->producer_edges) {
          SegmentedGroup* producer_group = producer_edge->from;
          // Don't add producers that have more than multiple consumers
          if (producer_group->consumer_edges.size() > 1) {
            continue;
          }
          if (!groups_to_merge_set.count(producer_group)) {
            to_visit.push_back(producer_group);
          }
        }
      }
    }

    return groups_to_merge;
  }

  // Try merging a candidate cast group. Return true if merged.
  bool mergeCastGroup(const std::vector<SegmentedGroup*>& groups) {
    auto sched_type = tryMerge(
        segment_candidate_finder_->segmented_fusion_.get(),
        segment_candidate_finder_->runtimeInfo(),
        groups);

    if (sched_type == SchedulerType::None) {
      return false;
    }

    segment_candidate_finder_->mergeAllGivenGroups(groups);

    return true;
  }

  bool isUpCast(SegmentedGroup* group) const {
    if (auto precision_bits = getProducerConsumerPrecisionBit(group);
        precision_bits.has_value()) {
      return precision_bits->first < precision_bits->second;
    } else {
      return false;
    }
  }

  bool isDownCast(SegmentedGroup* group) const {
    if (auto precision_bits = getProducerConsumerPrecisionBit(group);
        precision_bits.has_value()) {
      return precision_bits->first > precision_bits->second;
    } else {
      return false;
    }
  }

  std::optional<std::pair<int64_t, int64_t>> getProducerConsumerPrecisionBit(
      SegmentedGroup* group) const {
    if (group->exprs().size() != 1) {
      return std::nullopt;
    }

    auto uop = dynamic_cast<UnaryOp*>(group->exprs().front());
    if (uop == nullptr || uop->getUnaryOpType() != UnaryOpType::Cast) {
      return std::nullopt;
    }

    return ir_utils::getPrecisionOfProducerConsumerTensorsBit(uop);
  }

 private:
  SegmentCandidateFinder* segment_candidate_finder_ = nullptr;
};

// Concat is represented as PadOp nodes of inputs, followed by a
// ConcatOp node. It's unlikely they should be separated.
class MergeCatWithInputPads {
 public:
  static void run(SegmentCandidateFinder* segment_candidate_finder) {
    MergeCatWithInputPads merge_cat_with_input_pads(segment_candidate_finder);
  }

 private:
  MergeCatWithInputPads(SegmentCandidateFinder* segment_candidate_finder)
      : segment_candidate_finder_(segment_candidate_finder) {
    merge();
  }

  void merge() {
    std::unordered_set<SegmentedGroup*> considered_groups;
    std::vector<std::vector<SegmentedGroup*>> candidates;

    for (SegmentedGroup* group : segment_candidate_finder_->groups()) {
      if (considered_groups.contains(group)) {
        continue;
      }

      auto groups_to_merge = getGroupsToMerge(group);
      if (!groups_to_merge.has_value()) {
        continue;
      }

      considered_groups.insert(
          groups_to_merge->begin(), groups_to_merge->end());

      candidates.emplace_back(*groups_to_merge);
    }

    // Try merging the detected group
    for (const auto& groups_to_merge : candidates) {
      mergeGroups(groups_to_merge);
    }
  }

  // Given a segmented group, try to detect a concat pattern, where
  // all of the inputs are produced by pad groups.
  std::optional<std::vector<SegmentedGroup*>> getGroupsToMerge(
      SegmentedGroup* cat_group) {
    // This pass is assumed to be applied before the iterative
    // merge step, so only consider groups that are not yet merged
    // at all.
    if (cat_group->exprs().size() != 1) {
      return std::nullopt;
    }

    // Technically, this can be just an add operation as concat can be
    // represented with pad and add.
    // TODO: Consider extending this pattern matching to support
    // add-based concat
    auto cat = dynamic_cast<CatOp*>(cat_group->exprs().at(0));
    if (cat == nullptr) {
      return std::nullopt;
    }

    // Check if a given pad is for the cat. More strictly, the actual
    // padding widths do matter, but it's unikely to make any
    // difference. Since this is a heuristic, this check should be
    // good enough.
    auto is_matching_pad = [&cat](PadOp* pad) {
      if (!pad->value()->isZero()) {
        return false;
      }
      auto padded_axes = pad->getPaddedAxes();
      return padded_axes.size() == 1 &&
          padded_axes.at(0) == cat->concatenatedDim();
    };

    std::vector<SegmentedGroup*> groups_to_merge;
    groups_to_merge.reserve(cat->inputs().size() + 1);
    for (const auto cat_inp : cat->inputs()) {
      auto pad = dynamic_cast<PadOp*>(cat_inp->definition());
      if (pad == nullptr || !is_matching_pad(pad)) {
        return std::nullopt;
      }

      auto producer_edge_it = std::ranges::find_if(
          cat_group->producer_edges, [&](SegmentedEdge* producer_edge) {
            SegmentedGroup* producer_group = producer_edge->from;
            return producer_group->exprs().size() == 1 &&
                producer_group->exprs().at(0) == pad &&
                producer_group->consumer_edges.size() == 1;
          });
      if (producer_edge_it == cat_group->producer_edges.end()) {
        return std::nullopt;
      }

      groups_to_merge.push_back((*producer_edge_it)->from);
    }

    groups_to_merge.push_back(cat_group);

    return groups_to_merge;
  }

  bool mergeGroups(const std::vector<SegmentedGroup*>& groups) {
    auto sched_type = tryMerge(
        segment_candidate_finder_->segmented_fusion_.get(),
        segment_candidate_finder_->runtimeInfo(),
        groups);

    if (sched_type == SchedulerType::None) {
      return false;
    }

    segment_candidate_finder_->mergeAllGivenGroups(groups);

    return true;
  }

 private:
  SegmentCandidateFinder* segment_candidate_finder_ = nullptr;
};

namespace {

//! Allow the segmentation algorithm to prefer certain exprs to merge
class PreferredMergeCandidatePicker {
 public:
  static std::vector<std::pair<SegmentedGroup*, SegmentedGroup::NeighborGroup>>
  get(const std::vector<SegmentedGroup*>& groups) {
    return PreferredMergeCandidatePicker(groups).candidates_;
  }

 private:
  PreferredMergeCandidatePicker(const std::vector<SegmentedGroup*>& groups)
      : groups_(groups) {
    for (auto& group : groups_) {
      if (auto neighbor_to_merge = mergeSelectLikeOpsWithProducers(group);
          neighbor_to_merge.has_value()) {
        candidates_.emplace_back(group, *neighbor_to_merge);
        continue;
      }
      if (auto neighbor_to_merge = mergePadWithConsumers(group);
          neighbor_to_merge.has_value()) {
        candidates_.emplace_back(group, *neighbor_to_merge);
        continue;
      }
    }
  }

  //! Prefer merging groups with select-like exprs with producer
  //! groups, including indexSelect, torchGather and takeAlongAxis
  //! where only one element is selected/gathered/taken, producing a
  //! broadcast domain. Fusing these exprs with producers is
  //! straightforward, but may not be always possible with consumers as
  //! consumer reference tensors may not know about the gathered
  //! domain, much like reduction domains. Moreover, if segmentation is
  //! necessary, it would be more efficient to segment a kernel after
  //! these exprs as the segment output tensors would become smaller.
  //!
  //! A motivating example is cross-entropy loss, where softmax is
  //! followed by takeAlongAxis, and then is followed by a
  //! reduction. Currently, it's not possible to fuse the softmax and
  //! the reduction, so it must be segmented to two groups, and we
  //! want to segment the fusion between the takeAlongAxis and the
  //! reduction, not between the softmax and takeAlongAxis.
  std::optional<SegmentedGroup::NeighborGroup> mergeSelectLikeOpsWithProducers(
      SegmentedGroup* group) const;

  //! Prefer merging pad exprs with consumer groups. Since pad is
  //! likely expand an iter domain, having a segmentation boundary, if
  //! necessary, is preferred to be before than after pad.
  std::optional<SegmentedGroup::NeighborGroup> mergePadWithConsumers(
      SegmentedGroup* group) const;

 private:
  const std::vector<SegmentedGroup*>& groups_;
  std::vector<std::pair<SegmentedGroup*, SegmentedGroup::NeighborGroup>>
      candidates_;
};

std::optional<SegmentedGroup::NeighborGroup> PreferredMergeCandidatePicker::
    mergeSelectLikeOpsWithProducers(SegmentedGroup* group) const {
  if (group->producer_edges.empty() || group->isMerged()) {
    return std::nullopt;
  }

  const auto& exprs = group->exprs();

  // I *think* it's enough to consider the initial merge of
  // select-like ops.
  if (exprs.size() != 1) {
    return std::nullopt;
  }

  auto expr = exprs.at(0);

  // Select-like exprs have a producer ID that is indirectly
  // accessed with an index input
  if (ir_utils::getIndexedProducerID(expr) == nullptr) {
    return std::nullopt;
  }

  auto lookup_tv = ir_utils::getTvInput(expr);

  // If the lookup tv is a fusion input, there's nothing to do
  if (lookup_tv->isFusionInput()) {
    return std::nullopt;
  }

  auto consumer_of_indexed_id = ir_utils::getConsumerOfIndexedProducerID(expr);

  // There must be a consumer ID unless it's a Select expr
  NVF_ERROR(
      consumer_of_indexed_id != nullptr || expr->isA<SelectOp>(),
      "Consumer of indexed ID not found: ",
      expr->toString());

  // In case of non select expr, make sure the consumer ID is a broadcat
  if (!expr->isA<SelectOp>() && !consumer_of_indexed_id->isBroadcast()) {
    return std::nullopt;
  }

  // Find the producer group that corresponds to the lookup tensor
  // of the expr.
  auto producer_edge_it = std::find_if(
      group->producer_edges.begin(),
      group->producer_edges.end(),
      [&lookup_tv](SegmentedEdge* edge) { return edge->val == lookup_tv; });

  // Not sure this could happen. Just assert for now.
  if (producer_edge_it == group->producer_edges.end()) {
    NVF_THROW("Unexpected");
    return std::nullopt;
  }

  auto producer_group = (*producer_edge_it)->from;
  if (producer_group->isMerged()) {
    return std::nullopt;
  }

  // Don't try to merge if not a candidate
  auto merge_candidates = group->getMergeCandidates();
  if (std::ranges::find_if(
          merge_candidates, [&](const SegmentedGroup::NeighborGroup& neighbor) {
            return neighbor.group == producer_group;
          }) == merge_candidates.end()) {
    return std::nullopt;
  }

  return SegmentedGroup::NeighborGroup(producer_group, *producer_edge_it);
}

std::optional<SegmentedGroup::NeighborGroup> PreferredMergeCandidatePicker::
    mergePadWithConsumers(SegmentedGroup* group) const {
  if (group->consumer_edges.empty() || group->isMerged()) {
    return std::nullopt;
  }

  const auto merge_candidates = group->getMergeCandidates();

  if (merge_candidates.empty()) {
    return std::nullopt;
  }

  for (auto expr : group->exprs()) {
    auto pad = dynamic_cast<PadOp*>(expr);
    if (pad == nullptr) {
      continue;
    }

    // If the input of pad is already in the same segment, don't
    // bother
    auto pad_inp = pad->in();
    if (std::ranges::find_if(group->producer_edges, [&](SegmentedEdge* edge) {
          return edge->val == pad_inp;
        }) == group->producer_edges.end()) {
      continue;
    }

    // Look for a consumer edge that has the pad output as its val,
    // which means the pad output is passed to the consumer group.
    for (const auto& consumer_edge : group->consumer_edges) {
      if (consumer_edge->val != pad->out()) {
        continue;
      }

      auto consumer_group = consumer_edge->to;
      if (consumer_group->isMerged()) {
        continue;
      }

      // Don't try to merge if not a candidate
      if (std::ranges::find_if(
              merge_candidates,
              [&](const SegmentedGroup::NeighborGroup& neighbor) {
                return neighbor.group == consumer_group;
              }) == merge_candidates.end()) {
        continue;
      }

      return SegmentedGroup::NeighborGroup(consumer_group, consumer_edge);
    }
  }

  return std::nullopt;
}

} // namespace

bool SegmentCandidateFinder::codeGenSupportedMerge(
    SegmentedGroup* group1,
    SegmentedGroup* group2) {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::codeGenSupportedMerge");
  NVF_ERROR(
      !segmented_fusion_->getEdgesBetween(group1, group2).empty() ||
          !segmented_fusion_->getEdgesBetween(group2, group1).empty(),
      "only support testing immediate producer-consumer groups");
  // The segmemter should ideally be redesigned to be more flexible and
  // decoupled from the schedulers, but for now, we just return
  // `SchedulerType::None` as it is not relevant when the segmenter is
  // used with a custom should-merge function.
  if (options_.custom_should_merge_groups != nullptr) {
    return (options_.custom_should_merge_groups)(group1, group2);
  }
  return tryMerge(segmented_fusion_.get(), runtimeInfo(), group1, group2) !=
      SchedulerType::None;
}

// TODO: consider caching the heuristics value so tryMerge doesn't have to be
//       called twice
SchedulerType SegmentCandidateFinder::deriveSchedulerType(
    SegmentedGroup* group) {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::deriveSchedulerType");
  if (options_.custom_should_merge_groups != nullptr) {
    // We don't need to generate a SchedulerType for multidevice segments at
    // this moment
    return SchedulerType::None;
  }
  auto scheduler_type = tryMerge(segmented_fusion_.get(), runtimeInfo(), group);
  NVF_ERROR(
      scheduler_type != SchedulerType::None,
      "Can not find a scheduler to schedule fusion segment");
  return scheduler_type;
}

SegmentCandidateFinder::SegmentCandidateFinder(
    std::unique_ptr<Fusion> fusion,
    const KernelArgumentHolder& inputs,
    SegmentCandidateFinderOptions options,
    bool multi_device)
    : options_(options), runtime_inputs_(inputs) {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::SegmentCandidateFinder");
  NVF_ERROR(
      options_.custom_should_merge_groups == nullptr ||
          (!options_.run_translate_welford &&
           !options_.run_combine_reductions && options_.run_herrmann_merge &&
           options_.run_final_merge),
      "Invalid Segmenter options");
  segmented_fusion_ = std::make_unique<SegmentedFusion>(std::move(fusion));

  // Conditionally initialize runtime_info_ based on multi_device
  if (!multi_device) {
    runtime_info_.emplace(segmented_fusion_->completeFusion(), inputs);
  }

  privatizeUpcast();
  findSegments();
}

// Add runtimeInfo() accessor with validation
SchedulerRuntimeInfo& SegmentCandidateFinder::runtimeInfo() {
  NVF_ERROR(
      runtime_info_.has_value(),
      "runtime_info_ is not available. This function should not be called in "
      "multi-device segmentation.");
  return *runtime_info_;
}

void SegmentCandidateFinder::buildInitialSegments() {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::buildInitialSegments");
  groups().clear();
  edges().clear();

  // TODO: Make traversal items local to this function.
  // Need this for initialization of the DAG that is process
  std::unordered_map<Expr*, SegmentedGroup*> expr2group;

  // Initialize DAG, convert each expr to a segment group
  auto exprs = completeFusion()->exprs();
  for (auto expr : exprs) {
    if (!ir_utils::isScalarOp(expr)) {
      auto new_group = segmented_fusion_->newGroup(expr);
      expr2group.insert(std::make_pair(expr, new_group));
    }
  }

  // TODO(wujingyue): remove singleton groups that are forwarded. They are
  // useless and cause duplication.
  forwardInputs();

  // Create edges between the Exprs. Mark inputs and outputs of the fusion.
  for (auto expr : exprs) {
    // No group created for scalar ops
    if (ir_utils::isScalarOp(expr)) {
      continue;
    }

    if (excluded_inp_unary_exprs_.has(expr)) {
      continue;
    }

    SegmentedGroup* expr_group = expr2group.at(expr);
    for (auto inp : expr->inputs()) {
      if (isFusionInput(inp)) {
        expr_group->input_vals_.pushBack(inp);
        auto aux_group = input2group_.at(inp);
        segmented_fusion_->connectGroups(aux_group, expr_group, inp);
        continue;
      }

      // Could be something like a constant scalar, definition is nullptr, but
      // isn't an "input" to the fusion. At least not one provided by an
      // external source.
      if (inp->definition() == nullptr) {
        continue;
      }

      // No group created for scalar ops since they may need to be duplicated
      //  to avoid scalar edges. They are handled in resolveScalarsInGroup
      if (inp->isScalar()) {
        continue;
      }

      auto def_group = expr2group.at(inp->definition());
      segmented_fusion_->connectGroups(def_group, expr_group, inp);
    }
    for (auto out : expr->outputs()) {
      if (out->isFusionOutput()) {
        expr_group->output_vals_.pushBack(out);
      }
    }
  }
}

void SegmentCandidateFinder::trySetUpMerge(
    SegmentedGroup* group,
    std::vector<SegmentedGroup::NeighborGroup> candidates) {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::trySetUpMerge");
  if (group->merged_ || group->exprs_.empty()) {
    return;
  }

  if (candidates.empty()) {
    candidates = group->getMergeCandidates();
  }

  if (candidates.empty()) {
    return;
  }

  // Try to find a non-merged candidate that can be merged with this
  // group
  for (const auto& candidate : candidates) {
    if (candidate.group->isMerged() ||
        !codeGenSupportedMerge(group, candidate.group)) {
      continue;
    }

    to_merge_.emplace_back(group);
    to_merge_.emplace_back(candidate.group);

    group->merged_ = true;
    group->merge_with_ = candidate.group;
    group->merge_through_ = candidate.edge;

    candidate.group->merged_ = true;
    candidate.group->merge_with_ = group;
    candidate.group->merge_through_ = candidate.edge;
    return;
  }
}

void SegmentCandidateFinder::resolveForwardedInputs() {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::resolveForwardedInputs");
  for (Val* forwarded_input : forwarded_fusion_inputs_) {
    if (forwarded_input->isFusionInput()) {
      // Nothing to resolve.
      continue;
    }

    if (forwarded_input->isScalar()) {
      // Scalar forwarded inputs will be resolved after this loop.
      // resolveNonscalarForwardedInput resolves only non-scalar ones because
      // consumer_edges of a scalar input is always empty due to
      // `removeScalarEdges`.
      continue;
    }

    resolveNonscalarForwardedInput(forwarded_input);
    // aux_group will be removed from segmented_fusion_ by
    // cleanupForwardedInputs.
  }

  // Un-forward scalar inputs unconditionally.
  for (SegmentedGroup* group : segmented_fusion_->groups()) {
    std::vector<Val*> forwarded_scalar_inputs;
    for (Val* input_val : group->input_vals_) {
      if (!input_val->isFusionInput() && input_val->isScalar()) {
        forwarded_scalar_inputs.push_back(input_val);
      }
    }

    group->input_vals_ = IterVisitor::getInputsTo(group->inputs());
    auto input_exprs = StmtSort::getExprsTo(forwarded_scalar_inputs);
    // Insert those expressions at the beginning of the group
    group->exprs_.insert(
        group->exprs_.begin(), input_exprs.begin(), input_exprs.end());
  }
}

void SegmentCandidateFinder::findSegments() {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::findSegments");

  buildInitialSegments();

  validateIfDebug();

  auto has_welford_ops =
      ir_utils::hasOpsOfType<WelfordOp>(segmented_fusion_->completeFusion());

  if (options_.run_translate_welford && has_welford_ops) {
    if (TranslateApplicableWelford::run(
            segmented_fusion_.get(), runtime_inputs_)) {
      // If modified, rebuild segments as existing expressions may be
      // pulled into welford groups
      buildInitialSegments();
    }
  }

  validateIfDebug();

  for (auto group : groups()) {
    if (!group->outputs().empty()) {
      // Set SchedulerType in case single reduction kernels were left out
      group->setSchedulerType(deriveSchedulerType(group));
    }
  }

  // Remove all scalar edges since they do not represent actual
  //  dependency among segmented groups.
  removeScalarEdges();

  // Run pre-merge heuristics

  MergeCatWithInputPads::run(this);
  validateIfDebug(true);

  MergeUpAndDownCast::run(this);
  validateIfDebug(true);

  if (options_.run_combine_reductions && CombineReductions::shouldRun(this)) {
    CombineReductions::run(this);
  }

  validateIfDebug();

  if (options_.run_herrmann_merge) {
    bool merged_nodes = true;
    // Initial merge iteration
    while (merged_nodes) {
      resetLevels();

      // Try preferred merge first
      for (auto& [group, neighbor] :
           PreferredMergeCandidatePicker::get(groups())) {
        if (!neighbor.group->isMerged()) {
          trySetUpMerge(group, {neighbor});
        }
      }

      // If there are preferred groups to merge, merge them first
      // without considering the rest of groups
      if (to_merge_.empty()) {
        for (auto& group : groups()) {
          trySetUpMerge(group);
        }
      }

      if (to_merge_.empty()) {
        merged_nodes = false;
      }

      mergeNodes();

      validateIfDebug();
    }
  }

  validateIfDebug();

  if (options_.run_final_merge) {
    // TODO: consider interleaving herrmman merge and bruteforce merge, as
    // bruteforce merge can introduce opportunities for more herrmann merge
    finalMerge();
  }

  validateIfDebug();

  // Resolve all the input expressions needed in each group
  resolveForwardedInputs();

  // Do not require segments to be disjoint because, due to
  // resolveForwardedInputs, the graph may not be disjoint as some unary exprs
  // from fusion inputs may be shared in multiple groups.
  validateIfDebug(/*require_disjoint=*/false);

  // Forwarded input groups are no longer used. Clean them up.
  cleanupForwardedInputs();

  finalize();

  // run reset levels to validate the final graph is a DAG. reset levels will
  // fail if not.
  resetLevels();

  if (isDebugDumpEnabled(DebugDumpOption::FusionSegmentsDrawing)) {
    segmented_fusion_->draw();
  }
}

void SegmentCandidateFinder::privatizeUpcast() {
  // Insert castOp to complete_fusion_
  FusionGuard fg(segmented_fusion_->complete_fusion_.get());

  const auto exprs = segmented_fusion_->complete_fusion_->exprs();

  for (auto expr : exprs) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }

    for (const auto i : arange(expr->inputs().size())) {
      auto maybe_upcast_out_tv = dynamic_cast<TensorView*>(expr->input(i));
      if (maybe_upcast_out_tv == nullptr) {
        continue;
      }

      // Check if the input is an output of an upcast op
      auto maybe_upcast_op =
          dynamic_cast<UnaryOp*>(maybe_upcast_out_tv->definition());
      if (maybe_upcast_op == nullptr ||
          maybe_upcast_op->getUnaryOpType() != UnaryOpType::Cast) {
        continue;
      }

      auto precisions =
          ir_utils::getPrecisionOfProducerConsumerTensorsBit(maybe_upcast_op);
      if (!precisions.has_value() || precisions->first >= precisions->second) {
        continue;
      }

      // Check if there's multiple uses of the upcast output
      auto uses_of_upcast_out_tv = maybe_upcast_out_tv->uses();
      if (uses_of_upcast_out_tv.size() < 2) {
        continue;
      }

      // If this is the first use of the upcast output, keep it as is
      if (expr == uses_of_upcast_out_tv.front()) {
        continue;
      }

      TensorView* upcast_out_tv_clone = castOp(
          maybe_upcast_out_tv->dtype(),
          maybe_upcast_op->input(0)->as<TensorView>());
      TransformReplay::selfReplay(
          maybe_upcast_out_tv->domain(), upcast_out_tv_clone->domain());
      expr = ir_utils::replaceValInExprInputs(
          expr, maybe_upcast_out_tv, upcast_out_tv_clone);

      privatized_upcast_ops_[maybe_upcast_op].insert(
          upcast_out_tv_clone->definition()->as<UnaryOp>());
    }
  }
}

void SegmentCandidateFinder::revertPrivatizedUpcast(SegmentedGroup* group) {
  // If a given consumer edge is a duplicate of another edge of the
  // same producer group, remove the given edge from both the producer
  // and consumer groups.
  auto maybe_deduplicate_edge =
      [](SegmentedEdge* maybe_duplicated_consumer_edge) {
        SegmentedGroup* producer_group = maybe_duplicated_consumer_edge->from;

        auto same_edge_it = std::find_if(
            producer_group->consumer_edges.begin(),
            producer_group->consumer_edges.end(),
            [&](SegmentedEdge* consumer_edge) {
              return consumer_edge != maybe_duplicated_consumer_edge &&
                  *consumer_edge == *maybe_duplicated_consumer_edge;
            });

        if (same_edge_it == producer_group->consumer_edges.end()) {
          return;
        }

        // maybe_duplicated_consumer_edge is redundant. Remove it from the
        // from and the two groups
        auto consumer_edge_to_remove = std::find(
            producer_group->consumer_edges.begin(),
            producer_group->consumer_edges.end(),
            maybe_duplicated_consumer_edge);
        NVF_ERROR(
            consumer_edge_to_remove != producer_group->consumer_edges.end());
        producer_group->consumer_edges.erase(consumer_edge_to_remove);

        SegmentedGroup* consumer_group = maybe_duplicated_consumer_edge->to;
        auto producer_edge_to_remove = std::find(
            consumer_group->producer_edges.begin(),
            consumer_group->producer_edges.end(),
            maybe_duplicated_consumer_edge);
        NVF_ERROR(
            producer_edge_to_remove != consumer_group->producer_edges.end());
        consumer_group->producer_edges.erase(producer_edge_to_remove);
      };

  // Replace old_expr with new_expr if found in a given group. Return
  // true if replaced.
  auto maybe_replace =
      [](SegmentedGroup* group, Expr* old_expr, Expr* new_expr) -> bool {
    auto it = std::find(group->exprs_.begin(), group->exprs_.end(), old_expr);
    if (it != group->exprs_.end()) {
      *it = new_expr;
      return true;
    } else {
      return false;
    }
  };

  for (const auto& [original_upcast, clones] : privatized_upcast_ops_) {
    std::vector<UnaryOp*> upcast_in_group;
    Val* upcast_val_to_keep = nullptr;
    for (auto uop : ir_utils::filterByType<UnaryOp>(group->exprs())) {
      if (uop != original_upcast && !clones.count(uop)) {
        continue;
      }

      upcast_in_group.push_back(uop);

      auto upcast_tv = uop->out();

      // Prefer the original upcast if found
      if (upcast_val_to_keep == nullptr ||
          upcast_tv == original_upcast->out()) {
        upcast_val_to_keep = upcast_tv;
      }
    }

    if (upcast_in_group.size() < 2) {
      continue;
    }

    for (auto uop : upcast_in_group) {
      Val* upcast_val_to_replace = uop->out();
      if (upcast_val_to_replace == upcast_val_to_keep) {
        // Keep this uop as is since its output replaces the other
        // upcast outputs
        continue;
      }

      NVF_ERROR(
          upcast_val_to_replace->uses().size() == 1,
          "Multiple use of replicated upcast tensor found: ",
          toDelimitedString(upcast_val_to_replace->uses()));

      auto use_of_upcast_val_to_replace = upcast_val_to_replace->uses().at(0);

      auto updated_expr = ir_utils::replaceValInExprInputs(
          use_of_upcast_val_to_replace,
          upcast_val_to_replace,
          upcast_val_to_keep);

      // Replace use_of_upcast_val_to_replace with
      // updated_expr. use_of_upcast_val_to_replace must be in the
      // same group of its consumer groups
      if (!maybe_replace(group, use_of_upcast_val_to_replace, updated_expr)) {
        for (auto consumer_edge : group->consumer_edges) {
          if (maybe_replace(
                  consumer_edge->to,
                  use_of_upcast_val_to_replace,
                  updated_expr)) {
            break;
          }
        }
      }

      // Update a consumer edge if its val is
      // upcast_val_to_replace. Again, there must be at most one such
      // edge.
      SegmentedEdge* consumer_edge_to_update = nullptr;
      for (auto consumer_edge : group->consumer_edges) {
        if (consumer_edge->val == upcast_val_to_replace) {
          NVF_ERROR(
              consumer_edge_to_update == nullptr,
              "Multiple consumer edges using ",
              upcast_val_to_replace->toString(),
              " found");
          consumer_edge->val = upcast_val_to_keep;
          consumer_edge_to_update = consumer_edge;
        }
      }

      // Now that the consumer edge is updated, it may be a duplicate
      // of an exising edge. Remove if so.
      if (consumer_edge_to_update != nullptr) {
        maybe_deduplicate_edge(consumer_edge_to_update);
      }

      std::erase(group->exprs_, uop);

      // Note that it should not be necessary to do anything with
      // group->output_vals since the inserted upcast ops should never produce
      // fusion outputs.
    }
  }
}

// Decides whether we should forward an input (or a forwarded input) of a
// fusion. Currently, we forward an input only when its single use is a UnaryOp.
// Therefore, this function returns `v`'s single unary use or nullptr if it
// decides not to forward.
UnaryOp* shouldForward(Val* v) {
  const std::vector<Expr*>& uses = v->uses();
  // Just allow stripping out input with single use.
  // Stripping out multi-used inputs can lead to:
  // (1) Fragmentation of the DAG, increased segments, see test in #1301.
  // (2) Miss detection of persistent buffers, see issue #1607.
  if (uses.size() != 1) {
    return nullptr;
  }

  auto* unary_use = dynamic_cast<UnaryOp*>(uses.front());
  if (unary_use == nullptr) {
    return nullptr;
  }

  // Don't forward an input to an output yet. Doing that would lead to an empty
  // group that ought to work in theory but doesn't work in practice with the
  // downstream logic. See #1813 for an example.
  if (unary_use->out()->isFusionOutput()) {
    return nullptr;
  }

  // prevent forward to a SegmenterSet, which could cause unary op forward to a
  // no-op segment. See issue: https://github.com/NVIDIA/Fuser/issues/2658
  if (std::any_of(
          unary_use->out()->uses().begin(),
          unary_use->out()->uses().end(),
          [](const Expr* next_use) {
            if (const LoadStoreOp* use =
                    dynamic_cast<const LoadStoreOp*>(next_use)) {
              if (use->opType() == LoadStoreOpType::SegmenterSet) {
                return true;
              }
            }
            return false;
          })) {
    return nullptr;
  }

  return unary_use;
}

void SegmentCandidateFinder::forwardInputs() {
  excluded_inp_unary_exprs_ = {};
  input2group_.clear();

  std::vector<Val*> extended_fusion_inputs = completeFusion()->inputs();

  // Grab factory ops that should be forwarded. Add created tensors to
  // the fusion input list to make them handled like fusion inputs
  // TODO: Handle more factory methods such as IotaOp, EyeOp,
  // TensorConstruct. Probably should not include relatively expensive
  // ops like RNGOp.
  for (auto expr : completeFusion()->exprs()) {
    if (expr->isA<FullOp>() &&
        // Don't bother if it's a fusion output
        !expr->output(0)->isFusionOutput()) {
      extended_fusion_inputs.push_back(expr->output(0));
      excluded_inp_unary_exprs_.pushBack(expr);
    }
  }

  // "Terminating" outputs from the excluded input unary exprs, these will be
  // treated as complete fusion inputs.
  VectorOfUniqueEntries<Val*> forwarded_inputs;
  {
    std::deque<UnaryOp*> to_visit;
    for (Val* inp : extended_fusion_inputs) {
      if (UnaryOp* unary_use = shouldForward(inp)) {
        to_visit.push_back(unary_use);
      }
    }

    while (!to_visit.empty()) {
      UnaryOp* uop = to_visit.front();
      to_visit.pop_front();

      if (UnaryOp* unary_use = shouldForward(uop->out())) {
        to_visit.push_back(unary_use);
      } else {
        // We cannot extend the chain of unary ops, so we finalize this chain by
        // saving its output as a forwarded input.
        forwarded_inputs.pushBack(uop->out());
      }
      // Either way, `uop` is excluded from merging until
      // `resolveNonscalarForwardedInput` adds it back to one of the segments.
      excluded_inp_unary_exprs_.pushBack(uop);
    }
  }

  // Stop traversing back at factory vals (and fusion inputs)
  auto excluded_fusion_inputs = InputsOf::getInputsTo(
      {forwarded_inputs.begin(), forwarded_inputs.end()},
      extended_fusion_inputs);

  // List of vals to treat as complete fusion inputs for segmentation
  forwarded_fusion_inputs_ = extended_fusion_inputs;

  forwarded_fusion_inputs_.erase(
      std::remove_if(
          forwarded_fusion_inputs_.begin(),
          forwarded_fusion_inputs_.end(),
          [&excluded_fusion_inputs](Val* inp) {
            return std::find(
                       excluded_fusion_inputs.begin(),
                       excluded_fusion_inputs.end(),
                       inp) != excluded_fusion_inputs.end();
          }),
      forwarded_fusion_inputs_.end());

  forwarded_fusion_inputs_.insert(
      forwarded_fusion_inputs_.end(),
      forwarded_inputs.begin(),
      forwarded_inputs.end());

  // Insert auxiliary groups to use group dependency on inputs as well
  for (auto input : forwarded_fusion_inputs_) {
    auto new_group = segmented_fusion_->newGroup();
    input2group_.insert({input, new_group});
  }
}

void SegmentCandidateFinder::cleanupForwardedInputs() {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::cleanupForwardedInputs");
  std::unordered_set<SegmentedGroup*> input_groups;
  for (auto input : forwarded_fusion_inputs_) {
    input_groups.insert(input2group_.at(input));
  }
  eraseGroups(input_groups);

  excluded_inp_unary_exprs_ = {};
  forwarded_fusion_inputs_.clear();
  input2group_.clear();
}

std::vector<SegmentedGroup*> SegmentCandidateFinder::getAuxiliaryInputGroups()
    const {
  std::vector<SegmentedGroup*> aux_groups;
  aux_groups.reserve(input2group_.size());
  std::ranges::transform(input2group_, aux_groups.begin(), [](const auto& kv) {
    return kv.second;
  });
  return aux_groups;
}

void SegmentCandidateFinder::finalMerge() {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::finalMerge");
  auto producer_check = getGroupDependency();

  bool merged_nodes = true;
  while (merged_nodes) {
    // Iterate all groups and check if a group
    //  can merge with one of its consumers
    for (auto producer_group : groups()) {
      if (producer_group->exprs_.empty()) {
        continue;
      }
      // Populate consumers and their corresponding consumer edges
      std::unordered_map<SegmentedGroup*, SegmentedEdge*> consumer_edge_map;
      std::vector<SegmentedGroup*> all_consumers_of_producer_group;
      for (auto consumer : producer_group->consumer_edges) {
        // Since this is the last fusion pass, we can enable fusion through
        // outputs. Priority of this was decreased because if the only
        // connection between groups is an output node, best case scenario we
        // can save a single pass in memory. Where if it wasn't an output it
        // would be two passes.
        consumer_edge_map.insert({consumer->to, consumer});
      }
      // Populate all consumers from the map to avoid duplicate
      std::transform(
          consumer_edge_map.begin(),
          consumer_edge_map.end(),
          std::back_inserter(all_consumers_of_producer_group),
          [](auto& it) { return it.first; });

      for (auto consumer : all_consumers_of_producer_group) {
        if (!producer_check->isConsumerOfAny(
                consumer, all_consumers_of_producer_group) &&
            codeGenSupportedMerge(producer_group, consumer)) {
          to_merge_.emplace_back(producer_group);
          to_merge_.emplace_back(consumer);
          producer_group->merged_ = true;
          producer_group->merge_with_ = consumer;
          producer_group->merge_through_ = consumer_edge_map.at(consumer);
          consumer->merged_ = true;
          consumer->merge_with_ = producer_group;
          consumer->merge_through_ = producer_group->merge_through_;
          break;
        }
      }

      // Only want to merge one pair at a time so break if found any
      if (!to_merge_.empty()) {
        break;
      }
    }

    if (to_merge_.empty()) {
      merged_nodes = false;
    } else {
      NVF_ERROR(
          to_merge_.size() == 2, "merging more than 2 nodes in final iter");
      mergeNodes();
    }
  }
}

void SegmentCandidateFinder::resolveScalarsInGroup(SegmentedGroup* group) {
  std::vector<Val*> to_visit;
  std::unordered_set<Val*> visited;

  const auto processTV = [&to_visit](TensorView* tv) {
    for (auto id : TensorDomain::noReductions(tv->getMaybeRootDomain())) {
      to_visit.push_back(id->getMaybeExpandedExtent());
    }
    if (tv->domain()->hasRoot()) {
      // traverse from root to logical and inspect all Expr attrs and outputs
      std::vector<Val*> all_vals;
      for (const auto id_expr : StmtSort::getExprsBetween(
               {tv->getRootDomain().begin(), tv->getRootDomain().end()},
               {tv->getLogicalDomain().begin(),
                tv->getLogicalDomain().end()})) {
        all_vals.insert(
            all_vals.end(), id_expr->inputs().begin(), id_expr->inputs().end());
        all_vals.insert(
            all_vals.end(),
            id_expr->outputs().begin(),
            id_expr->outputs().end());
        for (const auto attr : id_expr->attributes()) {
          if (attr && attr->isVal()) {
            all_vals.push_back(attr->asVal());
          }
        }
        for (const auto val : all_vals) {
          if (val->isScalar()) {
            to_visit.push_back(val);
          } else if (const auto id = dynamic_cast<IterDomain*>(val)) {
            to_visit.push_back(id->getMaybeExpandedExtent());
          }
        }
      }
    }
  };

  // Segment TensorView inputs will have their logical extents available, so we
  // avoid adding them as separate scalar inputs.
  for (auto e : group->producer_edges) {
    if (const auto tv = dynamic_cast<TensorView*>(e->val)) {
      for (auto id : TensorDomain::noReductions(tv->getLogicalDomain())) {
        visited.insert(id->getMaybeExpandedExtent());
      }
    }
  }

  // Collect all scalar uses in the group
  for (auto expr : group->exprs()) {
    for (auto input : expr->inputs()) {
      if (input->isScalar()) {
        to_visit.push_back(input);
      } else if (auto tv = dynamic_cast<TensorView*>(input); tv &&
                 std::none_of(group->producer_edges.begin(),
                              group->producer_edges.end(),
                              [&tv](SegmentedEdge* e) {
                                return e->val == tv;
                              })) {
        // Intermediate group inputs (producer edges) will have their logical
        // domain reassigned as the root domain, so there is no need to process
        // them. Tensors computed inside this group will need processing,
        // however, as their root->logical transforms must be computed in this
        // group.
        processTV(tv);
      }
    }
    for (auto attr : expr->attributes()) {
      auto attr_val = dynamic_cast<Val*>(attr);
      if (!attr_val) {
        continue;
      }
      if (attr_val->isScalar()) {
        to_visit.push_back(attr_val);
      } else if (auto tv = dynamic_cast<TensorView*>(attr_val)) {
        processTV(tv);
      }
    }
    for (auto output : expr->outputs()) {
      // We must be able to compute output extents for expression, so here we
      // ensure the scalars involved are all available to this group
      if (auto tv = dynamic_cast<TensorView*>(output)) {
        processTV(tv);
      }
    }
  }

  // Keep track of composite fusion inputs used in this group
  std::unordered_set<Val*> input_set;
  for (auto inp : group->input_vals_) {
    input_set.insert(inp);
    if (auto tv = dynamic_cast<TensorView*>(inp)) {
      for (IterDomain* id :
           TensorDomain::noReductions(tv->getLogicalDomain())) {
        // Extents of inputs will already be bound. This prevents adding them
        // as redundant inputs.
        input_set.insert(id->getMaybeExpandedExtent());
      }
    }
  }

  // Record and append all missing scalar exprs at the end.
  std::vector<Expr*> exprs_to_add;

  // Do a stack based traversal of the scalar ops to avoid
  //  combinatorial duplication of exprs.
  while (!to_visit.empty()) {
    auto stack_top_val = to_visit.back();
    if (visited.count(stack_top_val)) {
      to_visit.pop_back();
    } else if (stack_top_val->definition() == nullptr) {
      // A scalar without def can be a scalar, a tensor dim,
      //  or a composite fusion input
      // The first two cases are handled in finalize(),
      //  the last case needs to add new input_val to this group.
      visited.insert(stack_top_val);
      // If this is a composite fusion scalar input, make sure this group has it
      if (stack_top_val->isFusionInput() && !input_set.count(stack_top_val)) {
        group->input_vals_.pushBack(stack_top_val);
        input_set.insert(stack_top_val);
      }
      to_visit.pop_back();
    } else {
      // A scalar with an actual definition
      auto definition_expr = stack_top_val->definition();
      bool all_inputs_visited = true;
      // If any of the inputs are not visited, visit them first
      for (auto input : definition_expr->inputs()) {
        if (!visited.count(input)) {
          all_inputs_visited = false;
          to_visit.push_back(input);
        }
      }
      // This node is ready to be visited
      if (all_inputs_visited) {
        // Collect the defining expr to insert into group
        exprs_to_add.push_back(definition_expr);
        visited.insert(stack_top_val);
        to_visit.pop_back();
      }
    }
  }

  // Add all the defining expr to the group
  for (auto expr : exprs_to_add) {
    group->exprs_.push_back(expr);
  }
}

SegmentedGroup* SegmentCandidateFinder::createInputGroup(Val* forwarded_input) {
  SegmentedGroup* group = segmented_fusion_->newGroup();
  for (auto inp : IterVisitor::getInputsTo({forwarded_input})) {
    // inp may be a factory-created tensor, which is not an input to
    // the group.
    if (std::ranges::find(completeFusion()->inputs(), inp) !=
        completeFusion()->inputs().end()) {
      group->input_vals_.pushBack(inp);
    }
  }
  group->exprs_ = StmtSort::getExprsTo({forwarded_input});
  return group;
}

void SegmentCandidateFinder::resolveNonscalarForwardedInput(
    Val* forwarded_input) {
  SegmentedGroup* aux_group = input2group_.at(forwarded_input);
  NVF_ERROR(aux_group->producer_edges.empty());

  GroupSet consumers;
  for (SegmentedEdge* edge : aux_group->consumer_edges) {
    consumers.pushBack(edge->to);
  }

  for (SegmentedGroup* consumer : consumers) {
    SegmentedGroup* input_group = createInputGroup(forwarded_input);
    std::vector<SegmentedEdge*> edges_to_remove;
    std::vector<SegmentedEdge*> producer_edge_copy = consumer->producer_edges;
    // Use a copy to iterate over edges as connect group can invalidate the
    // original iterator
    for (SegmentedEdge* edge : producer_edge_copy) {
      if (edge->from == aux_group && edge->val == forwarded_input) {
        // Create new edges before removing old ones
        segmented_fusion_->connectGroups(
            input_group, consumer, forwarded_input);
        // Now safe to remove old edges
        edges_to_remove.push_back(edge);
      }
    }
    for (auto edge_to_remove : edges_to_remove) {
      segmented_fusion_->removeEdge(edge_to_remove);
    }
    consumer->input_vals_.erase(forwarded_input);

    if (codeGenSupportedMerge(input_group, consumer)) {
      NVF_ERROR(to_merge_.empty());
      to_merge_.push_back(input_group);
      to_merge_.push_back(consumer);
      mergeNodes();
    }
  }
}

void SegmentCandidateFinder::removeScalarEdges() {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::removeScalarEdges");
  // Remove all scalar edges between groups
  //  They may have been created by welford
  //   translation.
  //  we will not need them after scalar
  //  resolution

  // Collect all scalar edges first since removeEdge modifies the edge lists
  std::vector<SegmentedEdge*> scalar_edges;
  for (auto edge : edges()) {
    if (edge->val->isScalar()) {
      scalar_edges.push_back(edge);
    }
  }

  // Remove each scalar edge through the proper API
  for (auto edge : scalar_edges) {
    segmented_fusion_->removeEdge(edge);
  }
}

void SegmentCandidateFinder::finalize() {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::finalize");
  // Remove unconnected groups
  groups().erase(
      std::remove_if(
          groups().begin(),
          groups().end(),
          [](SegmentedGroup* sg) {
            return sg->producer_edges.empty() && sg->consumer_edges.empty() &&
                sg->output_vals_.empty();
          }),
      groups().end());

  // Add group labeling
  int i = 0;
  for (auto it = groups().begin(); it != groups().end(); it++, i++) {
    deDuplicateScalarExprs((*it)->exprs_);
    (*it)->setID(i);
  }

  // TODO: too many things are currently abstracted under the term
  //  finalize. Need to re-structure in a follow up.

  // Finalize connections between segmented groups
  segmented_fusion_->finalize();

  // Resolve all the scalar expressions needed in each group
  for (auto group : segmented_fusion_->groups()) {
    resolveScalarsInGroup(group);
  }

  for (auto group : segmented_fusion_->groups()) {
    revertPrivatizedUpcast(group);
  }

  // Finalize each group, fill in the missing inputs, i.e. tensor dims.
  for (auto g : groups()) {
    g->setSchedulerType(deriveSchedulerType(g));
    g->finalize();
  }
}

GroupDependencyAnalysis* SegmentCandidateFinder::getGroupDependency() {
  if (!group_dependency_) {
    group_dependency_ =
        std::make_unique<GroupDependencyAnalysis>(segmented_fusion_.get());
  }
  return group_dependency_->as<GroupDependencyAnalysis>();
}

std::unique_ptr<HeuristicParams> SegmentedFusion::makeInitialHeuristicParams(
    SegmentedGroup* sg,
    SchedulerRuntimeInfo& runtime_info) {
  // This will be the first time each group is scheduled. So we'd want to
  //  construct the cache data here.
  auto heuristic_data_cache_ptr = std::make_unique<HeuristicDataCache>();
  auto heuristic_data_cache = heuristic_data_cache_ptr.get();
  setCachedHeuristicDataFor(sg, std::move(heuristic_data_cache_ptr));
  return SchedulerEntry::makeSchedulerInstance(sg->schedulerType())
      ->computeHeuristics(
          runtime_info.fusion(), runtime_info, heuristic_data_cache);
}

HeuristicDataCache* SegmentedFusion::getCachedHeuristicDataFor(
    SegmentedGroup* group) {
  auto data_it = heuristic_data_cache_.find(group);
  if (data_it == heuristic_data_cache_.end()) {
    return nullptr;
  }
  return data_it->second.get();
}

void SegmentedFusion::setCachedHeuristicDataFor(
    SegmentedGroup* group,
    std::unique_ptr<HeuristicDataCache> data) {
  NVF_ERROR(!heuristic_data_cache_.count(group));
  heuristic_data_cache_[group] = std::move(data);
}

void SegmentedFusion::validateDisjoint() const {
  FUSER_PERF_SCOPE("SegmentCandidateFinder::validateDisjoint");
  // Make sure it's disjoint. This property is not maintained after
  // the finalization as some of UnaryOp exprs using inputs may be
  // shared between multiple groups.
  std::unordered_set<Expr*> exprs;

  for (auto group : groups()) {
    if (group->merged_ || group->exprs().empty()) {
      continue;
    }

    for (auto expr : group->exprs()) {
      // Allow scalar exprs to exist in multiple groups
      if (ir_utils::isScalarOp(expr)) {
        continue;
      }
      NVF_ERROR(
          exprs.insert(expr).second,
          "Duplicate expression detected: ",
          expr->toString());
    }
  }
}

namespace {

//! A thin traversal class that collects all the tensorviews
//!  that could cast to fp16 or bf16 if they were segmented edges.
//!  The selected values are currently defined as all the
//!  tensorviews that
//!     1. are not complete fusion input/output,
//!     2. have a use chain that ends with a fp16
//!         complete fusion output
//!     3. are fp32 datatype
class ForceHalfAnnotation : public IterVisitor {
 public:
  static std::unordered_set<TensorView*> getFP16AnnotatedSet(Fusion* fusion) {
    ForceHalfAnnotation annotation;
    std::vector<Val*> fp16_outputs;
    auto& cast_to_type = annotation.cast_to_type_;
    auto other_half_type =
        cast_to_type == DataType::Half ? DataType::BFloat16 : DataType::Half;
    std::copy_if(
        fusion->outputs().begin(),
        fusion->outputs().end(),
        std::back_inserter(fp16_outputs),
        [&cast_to_type, &other_half_type](auto* val) {
          auto dtype = val->getDataType().value();
          if (cast_to_type) {
            NVF_ERROR(
                other_half_type != dtype,
                "Mix of BFloat16 and Float16 in the same graph is not "
                "supported.");
          }
          return val->template isA<TensorView>() &&
              val->getDataType().has_value() &&
              (val->getDataType().value() == DataType::Half ||
               val->getDataType().value() == DataType::BFloat16);
        });

    annotation.traverseTo(fp16_outputs);
    return annotation.force_fp16_tv_set_;
  }

 private:
  using IterVisitor::handle;

  void handle(TensorView* tv) override {
    auto dtype = tv->getDataType();
    if (dtype.has_value() && dtype.value() == DataType::Float &&
        !tv->isFusionOutput() && !tv->isFusionInput()) {
      force_fp16_tv_set_.insert(tv);
    }
  }

  std::unordered_set<TensorView*> force_fp16_tv_set_;
  std::optional<DataType> cast_to_type_ = std::nullopt;
};

} // namespace

void SegmentedFusion::annotateFP16IntermediateTensors() {
  force_fp16_tv_set_ =
      ForceHalfAnnotation::getFP16AnnotatedSet(complete_fusion_.get());
  for (auto out_tv :
       ir_utils::filterByType<TensorView>(complete_fusion_->outputs())) {
    if (out_tv) {
      auto dtype = out_tv->getDataType().value();
      if (dtype == DataType::Half || dtype == DataType::BFloat16) {
        force_half_precision_type_ = dtype;
      }
    }
  }
}

std::string toString(const SegmentCandidateFinderOptions& segment_options) {
  std::stringstream ss;
  ss << "segmentation phases {\n";
  if (segment_options.run_combine_reductions) {
    ss << "combine reductions\n";
  }
  if (segment_options.run_herrmann_merge) {
    ss << "herrmann merging\n";
  }
  if (segment_options.run_final_merge) {
    ss << "final merging\n";
  }
  ss << "\n}\n";
  return ss.str();
}

} // namespace nvfuser
