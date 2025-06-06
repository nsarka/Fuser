// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/builder.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <transform_iter.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvfuser {

// Transform dispatch
void ReplayTransformations::dispatch(Expr* e) {
  auto is_supported_expr =
      e->isOneOf<Split, Merge, Swizzle, Swizzle2D, Resize>();
  NVF_ERROR(
      is_supported_expr, "Invalid expr type found in transform traversal.");
  IterVisitor::dispatch(e);
}

// We're going to replay this split operation on the corresponding ID
void ReplayTransformations::handle(Split* s) {
  // Grab our input to the split node
  auto id_in = s->in();

  // Make sure we have a corresponding entry in our map pointing to the ID we're
  // going to replay the split on
  auto it = id_map_.find(id_in);
  if (it == id_map_.end()) {
    if (error_on_failure_) {
      NVF_THROW("Transform traversal failed, dependencies not met.");
    } else {
      return;
    }
  }

  auto mapped = it->second;
  // Make sure this ID is a loop ID (meaning it has no uses we generated)
  NVF_ERROR(
      loop_ids_.find(mapped) != loop_ids_.end(),
      "Transform traversal failed, modified a node but it was not a loop "
      "node.");

  // Replay the split onto mapped
  NVF_ERROR(s->outer()->isRFactorProduct() == s->inner()->isRFactorProduct());
  auto outs = IterDomain::split(
      mapped,
      s->factor(),
      s->innerSplit(),
      replay_rfactor_ && s->outer()->isRFactorProduct());
  // Remove mapped from the loop IDs
  loop_ids_.erase(mapped);

  // Add outputs to loop IDs
  loop_ids_[outs.first] = newCounter();
  loop_ids_[outs.second] = newCounter();

  // Update our ID map to include these outputs
  id_map_[s->outer()] = outs.first;
  id_map_[s->inner()] = outs.second;
}

// We're going to replay this merge operation on the corresponding IDs
void ReplayTransformations::handle(Merge* m) {
  // Grab the inputs to the merge node
  auto id_outer = m->outer();
  auto id_inner = m->inner();

  // Make sure we have a corresponding entry in our map pointing to the IDs
  // we're going to replay the merge on
  auto it_outer = id_map_.find(id_outer);
  auto it_inner = id_map_.find(id_inner);

  const bool outer_found = it_outer != id_map_.end();
  const bool outer_bcast = id_outer->isBroadcast();
  const bool inner_found = it_inner != id_map_.end();
  const bool inner_bcast = id_inner->isBroadcast();

  // If either are not found
  if (!outer_found || !inner_found) {
    // If both aren't found, it's a failure
    // If outer is found && inner is bcast it is not a failure
    // If inner is found && outer is bcast it is not a failure
    if (!(outer_found || inner_found) || (outer_found && !inner_bcast) ||
        (inner_found && !outer_bcast)) {
      if (error_on_failure_) {
        NVF_THROW("Transform traversal failed, dependencies not met.");
      } else {
        return;
      }
    }
  }

  // If we merge a broadcast dim with a non-broadcast dim, just remap the output
  // to the non-broadcast dim.
  if (inner_found && !outer_found && outer_bcast) {
    id_map_[m->out()] = it_inner->second;
    return;
  }
  if (outer_found && !inner_found && inner_bcast) {
    id_map_[m->out()] = it_outer->second;
    return;
  }

  // Grab the IDs we're going to replay this merge on
  const auto id_outer_mapped = it_outer->second;
  const auto id_inner_mapped = it_inner->second;

  // Make sure these IDs are loop IDs (meaning they have no uses we generated)
  NVF_ERROR(
      loop_ids_.find(id_outer_mapped) != loop_ids_.end() &&
          loop_ids_.find(id_inner_mapped) != loop_ids_.end(),
      "Transform traversal failed, tried to replay with ",
      id_outer_mapped,
      " and ",
      id_inner_mapped,
      " however one or both are not loop nodes.");

  // Replay the merge operation
  auto out = IterDomain::merge(
      id_outer_mapped,
      id_inner_mapped,
      replay_rfactor_ && m->out()->isRFactorProduct());

  // Remove inputs from the loop IDs
  loop_ids_.erase(id_outer_mapped);
  loop_ids_.erase(id_inner_mapped);

  // Add the output to the loop IDs
  loop_ids_[out] = newCounter();

  // Update our ID map with the replayed output
  id_map_[m->out()] = out;
}

void ReplayTransformations::handle(Swizzle* swizzle) {
  // Grab our input to the split node
  auto id_in_x = swizzle->inX();
  auto id_in_y = swizzle->inY();

  // Make sure we have a corresponding entry in our map pointing to the ID we're
  // going to replay the swizzle on
  auto it_x = id_map_.find(id_in_x);
  auto it_y = id_map_.find(id_in_y);

  if (it_x == id_map_.end() || it_y == id_map_.end()) {
    if (error_on_failure_) {
      NVF_THROW("Transform traversal failed, dependencies not met.");
    } else {
      return;
    }
  }

  auto mapped_x = it_x->second;
  auto mapped_y = it_y->second;

  // Make sure this ID is a loop ID (meaning it has no uses we generated)
  NVF_ERROR(
      loop_ids_.find(mapped_x) != loop_ids_.end() &&
          loop_ids_.find(mapped_y) != loop_ids_.end(),
      "Transform traversal failed, modified a node but it was not a loop "
      "node.");

  auto outs = std::make_pair(mapped_x, mapped_y);

  // Replay the swizzle onto mapped
  outs = IterDomain::swizzle(swizzle->swizzleType(), mapped_x, mapped_y);

  // Remove mapped from the loop IDs
  loop_ids_.erase(mapped_x);
  loop_ids_.erase(mapped_y);

  // Add outputs to loop IDs
  loop_ids_[outs.first] = newCounter();
  loop_ids_[outs.second] = newCounter();

  // Update our ID map to include these outputs
  id_map_[swizzle->outX()] = outs.first;
  id_map_[swizzle->outY()] = outs.second;
}

void ReplayTransformations::handle(Swizzle2D* swizzle_2d) {
  // Grab our input to the split node
  auto id_in_x = swizzle_2d->inX();
  auto id_in_y = swizzle_2d->inY();

  // Make sure we have a corresponding entry in our map pointing to the ID we're
  // going to replay the swizzle on
  auto it_x = id_map_.find(id_in_x);
  auto it_y = id_map_.find(id_in_y);

  if (it_x == id_map_.end() || it_y == id_map_.end()) {
    if (error_on_failure_) {
      NVF_THROW("Transform traversal failed, dependencies not met.");
    } else {
      return;
    }
  }

  auto mapped_x = it_x->second;
  auto mapped_y = it_y->second;

  // Make sure this ID is a loop ID (meaning it has no uses we generated)
  NVF_ERROR(
      loop_ids_.find(mapped_x) != loop_ids_.end() &&
          loop_ids_.find(mapped_y) != loop_ids_.end(),
      "Transform traversal failed, modified a node but it was not a loop "
      "node.");

  auto outs = std::make_pair(mapped_x, mapped_y);

  if (replay_swizzle_) {
    // Replay the swizzle onto mapped
    outs = IterDomain::swizzle(swizzle_2d->swizzleType(), mapped_x, mapped_y);

    // Remove mapped from the loop IDs
    loop_ids_.erase(mapped_x);
    loop_ids_.erase(mapped_y);
  }

  // Add outputs to loop IDs
  loop_ids_[outs.first] = newCounter();
  loop_ids_[outs.second] = newCounter();

  // Update our ID map to include these outputs
  id_map_[swizzle_2d->outX()] = outs.first;
  id_map_[swizzle_2d->outY()] = outs.second;
}

void ReplayTransformations::handle(Resize* exp) {
  auto id_in = exp->in();

  auto it = id_map_.find(id_in);
  if (it == id_map_.end()) {
    if (error_on_failure_) {
      NVF_THROW("Transform traversal failed, dependencies not met.");
    } else {
      return;
    }
  }

  auto mapped = it->second;
  // Make sure this ID is a loop ID (meaning it has no uses we generated)
  NVF_ERROR(
      loop_ids_.find(mapped) != loop_ids_.end(),
      "Transform traversal failed, modified a node but it was not a loop "
      "node.");

  auto out = mapped;

  if (replay_resize_) {
    out = IterDomain::resize(
        mapped,
        exp->leftExpand(),
        exp->rightExpand(),
        replay_rfactor_ && exp->out()->isRFactorProduct());
  }

  loop_ids_.erase(mapped);

  loop_ids_[out] = newCounter();

  id_map_[exp->out()] = out;
}

ReplayTransformations::ReplayTransformations(
    const std::vector<IterDomain*>& target_domain,
    std::unordered_map<IterDomain*, IterDomain*> id_map)
    : target_domain_(target_domain), id_map_(std::move(id_map)) {
  // Set all the loop nodes for tracking, all ids start as a loop and will be
  // updated based on the transformations
  for (auto entry : id_map_) {
    loop_ids_[entry.second] = newCounter();
  }
}

// Replays outputs that were generated from ids.first on ids.second
void ReplayTransformations::runReplay() {
  NVF_ERROR(
      !ran_replay_,
      "Cannot run replay twice without creating a new Replay Class.");

  if (error_on_failure_) {
    // Make sure id_map has all the inputs needed to replay target_domain
    auto inps = IterVisitor::getInputsTo(
        std::vector<Val*>(target_domain_.begin(), target_domain_.end()));
    std::for_each(inps.begin(), inps.end(), [this](Val* val) {
      NVF_ERROR(
          val->getValType().value() == ValType::IterDomain,
          "Expected IterDomain only for Replay Transformations, but found ",
          val);
      IterDomain* id = val->as<IterDomain>();
      NVF_ERROR(
          id_map_.find(id) != id_map_.end(),
          "Could not find required input: ",
          id,
          " in provided id_map.");
    });
  }

  ran_replay_ = true;

  if (target_domain_.empty() || id_map_.empty()) {
    return;
  }

  // Switch outDomain to a vector to start the traversal
  std::vector<Val*> traversal_vals(
      target_domain_.begin(), target_domain_.end());
  traverseTo(traversal_vals);

  if (error_on_failure_) {
    NVF_ERROR(
        loop_ids_.size() >= target_domain_.size(),
        "Transform traversal failed, did not find enough output IterDomains.");
  }

  // Validate replay
  for (auto out : target_domain_) {
    auto it_replayed = id_map_.find(out);
    if (it_replayed == id_map_.end()) {
      if (error_on_failure_) {
        NVF_THROW(
            "Transform traversal failed, could not find expected output.");
      }
      continue;
    }

    auto id_replayed = it_replayed->second;
    auto it_loop = loop_ids_.find(id_replayed);
    NVF_ERROR(
        it_loop != loop_ids_.end(),
        "Transform Traversal failed, expected a replayed dim for ",
        out,
        " but one was not created.");
  }

  // Populate loop_vec_ in a deterministic manner. This is deterministic
  // because size_t in loop_ids is filled based on operation order.
  std::set<std::pair<IterDomain*, size_t>, id_int_lt> ordered_set;
  for (auto entry : loop_ids_) {
    ordered_set.emplace(entry);
  }

  loop_vec_.clear();
  loop_vec_.resize(ordered_set.size());
  std::transform(
      ordered_set.begin(),
      ordered_set.end(),
      loop_vec_.begin(),
      [](std::pair<IterDomain*, size_t> entry) { return entry.first; });
}

#define ERROR_ON_FAILURE(cond)                                                 \
  do {                                                                         \
    if (error_on_failure_) {                                                   \
      NVF_ERROR(                                                               \
          (cond),                                                              \
          "Error during best effort replay, a transformation was called that " \
          "conflicts with an root-to-logical call.");                          \
    }                                                                          \
  } while (false)

BestEffortReplay::BestEffortReplay(
    const std::vector<IterDomain*>& replay_domain,
    const std::vector<IterDomain*>& target_domain,
    std::unordered_map<IterDomain*, IterDomain*> target2replay_map,
    std::unordered_map<IterDomain*, IterDomain*> replay_forward_id_map,
    std::unordered_map<IterDomain*, IterDomain*> target_forward_id_map,
    bool skip_replay_swizzle,
    bool skip_target_swizzle,
    bool skip_resize,
    bool error_on_failure)
    : target2replay_id_map_(std::move(target2replay_map)),
      replay_forward_id_map_(std::move(replay_forward_id_map)),
      target_forward_id_map_(std::move(target_forward_id_map)),
      skip_replay_swizzle_(skip_replay_swizzle),
      skip_target_swizzle_(skip_target_swizzle),
      error_on_failure_(error_on_failure) {
  for (auto entry : target2replay_id_map_) {
    loop_ids_[entry.second] = counter++;
  }

  // Grab expr history of iter domains in target_domain
  std::vector<Expr*> target_exprs =
      StmtSort::getExprsTo({target_domain.begin(), target_domain.end()});

  // If we check how an IterDomain was generated, it should only use an
  // IterDomain in an expression once. We pull a map from the input
  // IterDomains to the expression consuming them to generate the
  // replay_domain domain. This will be used to propagate the target_domain to
  // replay_domain map.

  // Map replay domain's IterDomains to the Exprs they're used in
  std::vector<Expr*> replay_exprs =
      StmtSort::getExprsTo({replay_domain.begin(), replay_domain.end()});

  // Track which id's in replay have to be replayed to guarantee root to logical
  // transformations. The iteration domains in the logical axes don't have
  // to be used in a matching expression in target, so we want to exclude those.
  // Only the iteration domains [root_domains, logical) domains have to be used
  // in matching transformation to guarantee logical domain is consistent.
  // However, if any logical id was used to produce the logical domain, we need
  // transformations on them to match the target exactly.
  std::unordered_set<IterDomain*> replay_logical_ids;

  // Track which expressions iteration domains are used, they should only be
  // used in one expression.
  std::unordered_map<IterDomain*, Expr*> replay_id2expr_map;
  for (auto replay_expr : replay_exprs) {
    for (auto id : ir_utils::filterByType<IterDomain>(replay_expr->inputs())) {
      NVF_ERROR(
          replay_id2expr_map.find(id) == replay_id2expr_map.end(),
          "Error trying to map rfactor root domain during replay.",
          " An IterDomain was found to be used in more than one expression.");

      replay_id2expr_map[id] = replay_expr;
    }

    // Only want to forward rfactor in map
    auto out_ids = ir_utils::filterByType<IterDomain>(replay_expr->outputs());
    if (std::any_of(out_ids.begin(), out_ids.end(), [](IterDomain* id) {
          return id->isRFactorProduct();
        })) {
      auto inp_ids = ir_utils::filterByType<IterDomain>(replay_expr->inputs());
      replay_logical_ids.insert(inp_ids.begin(), inp_ids.end());
    }
  }

  std::unordered_map<IterDomain*, Expr*> target_id2expr_map;
  for (auto target_expr : target_exprs) {
    for (auto id : ir_utils::filterByType<IterDomain>(target_expr->inputs())) {
      NVF_ERROR(
          target_id2expr_map.insert({id, target_expr}).second,
          "BestEffortReplay : Unexpected multi-use of id",
          id);
    }
  }

  if (skip_target_swizzle_ || skip_replay_swizzle_) {
    // Progress through all swizzle ops if we are skipping
    //  swizzles on the mapping.
    skipSwizzles(target_id2expr_map, replay_id2expr_map);
  }

  if (skip_resize) {
    skipResizes(target_exprs, replay_exprs);
  }

  std::string err_str(
      "Error during replay, a transformation was called that conflicts with an "
      "rfactor call.");

  bool any_target_expr_contains_broadcast_id = false;

  // Iterate through target IterDomains' history and compare with what we
  // recorded from replay_domain
  for (auto target_expr : target_exprs) {
    auto target_inps_filtered =
        ir_utils::filterByType<IterDomain>(target_expr->inputs());

    // If any input argument in target expression is in the forward map then
    // forward the mapped IterDomains in replay and continue to the next
    // expression as target_expr cannot match a replay_expr
    if (std::any_of(
            target_inps_filtered.begin(),
            target_inps_filtered.end(),
            [&](IterDomain* target_inp) {
              return this->inTargetForwardMap(target_inp);
            })) {
      for (auto target_inp : target_inps_filtered) {
        if (inTargetForwardMap(target_inp)) {
          auto target2replay_it = target2replay_id_map_.find(target_inp);
          if (target2replay_it != target2replay_id_map_.end()) {
            // Replace target_inp entry in target2replay_id_map_ with forwarded
            // id
            target2replay_id_map_[getTargetForwardedId(target_inp)] =
                target2replay_it->second;
            target2replay_id_map_.erase(target_inp);
          }
        }
      }
      // Continue to next target_expr
      continue;
    }

    std::vector<IterDomain*> target_id_inps(
        target_inps_filtered.begin(), target_inps_filtered.end());

    bool target_expr_contains_broadcast_id = std::any_of(
        target_inps_filtered.begin(),
        target_inps_filtered.end(),
        [](IterDomain* id) { return id->isBroadcast(); });
    any_target_expr_contains_broadcast_id =
        any_target_expr_contains_broadcast_id ||
        target_expr_contains_broadcast_id;

    std::vector<IterDomain*> replay_inps =
        std::vector<IterDomain*>(target_id_inps.size(), nullptr);

    bool missing_replay_input = false;

    // Map target_expr inputs to replay domain directly
    for (const auto t_i : arange(target_id_inps.size())) {
      // There might not be a mapping, that could be okay (depends on rfactor
      // checking).
      auto it = target2replay_id_map_.find(target_id_inps[t_i]);
      if (it != target2replay_id_map_.end()) {
        replay_inps[t_i] = getReplayForwardedId(it->second);
      } else {
        missing_replay_input = true;
      }
    }

    // Check if any of the associated replay id's are part of an logical domain
    bool replay_has_logical_inp = std::any_of(
        replay_inps.begin(),
        replay_inps.end(),
        [&replay_logical_ids](IterDomain* id) {
          return id == nullptr ? false
                               : id->isRFactorProduct() &&
                  (replay_logical_ids.find(id) != replay_logical_ids.end());
        });

    // If some replay id inputs are part of rfactor, make sure all target
    // expression inputs map to a replay input
    if (error_on_failure_ && replay_has_logical_inp) {
      bool no_missing_exprs = std::none_of(
          replay_inps.begin(),
          replay_inps.end(),
          [&replay_id2expr_map](IterDomain* id) {
            if (id == nullptr) {
              return true;
            } else {
              return replay_id2expr_map.find(id) == replay_id2expr_map.end();
            }
          });
      // View operation creates a TensorView with rfactor. After view, broadcast
      // operation adds iterDomains for any size-1 dimensions. Therefore, the
      // target domain (broadcast) may contain broadcast ids that are not
      // present in the replay domain (view). In this case, we skip any target
      // expressions that contain broadcast ids.
      NVF_ERROR(
          no_missing_exprs || any_target_expr_contains_broadcast_id, err_str);
    }

    // If any inputs are missing, continue as this expr doesn't match.
    if (missing_replay_input) {
      ERROR_ON_FAILURE(
          !replay_has_logical_inp || any_target_expr_contains_broadcast_id);
      continue;
    }

    // Find which replay_expr maps to the target_expr
    Expr* replay_expr = nullptr;
    // Check if all inputs have the same expression
    bool mismatched_replay_exprs = false;
    for (auto replay_inp : replay_inps) {
      auto it = replay_id2expr_map.find(replay_inp);
      if (it != replay_id2expr_map.end()) {
        if (replay_expr == nullptr) {
          replay_expr = it->second;
        } else {
          mismatched_replay_exprs =
              mismatched_replay_exprs || replay_expr != it->second;
        }
      } else {
        // If no expr is mapped then set mismatched epxrs to go to continue to
        // the next target expr
        mismatched_replay_exprs = true;
      }
    }

    // If expressions of mapped inputs don't match, then continue to next target
    // expr
    if (mismatched_replay_exprs || replay_expr == nullptr) {
      ERROR_ON_FAILURE(!replay_has_logical_inp);
      continue;
    }

    bool mismatched_inputs = replay_inps.size() != replay_expr->inputs().size();
    for (size_t i = 0; i < replay_inps.size() && !mismatched_inputs; i++) {
      mismatched_inputs =
          mismatched_inputs || replay_expr->inputs()[i] != replay_inps[i];
    }

    // If there isn't an logical id in the replay's inputs and there's a
    // mismatched input, continue
    if (mismatched_inputs) {
      ERROR_ON_FAILURE(!replay_has_logical_inp);
      continue;
    }

    // If there isn't an logical id in the replay's inputs and there's a
    // mismatch in replay_expr's and target_expr's outputs, continue
    if (target_expr->outputs().size() != replay_expr->outputs().size()) {
      ERROR_ON_FAILURE(!replay_has_logical_inp);
      continue;
    }

    // If there isn't an logical id in the replay's inputs and there's a
    // mismatch in replay_expr's and target_expr's expression type, continue
    if (typeid(*replay_expr) != typeid(*target_expr)) {
      ERROR_ON_FAILURE(!replay_has_logical_inp);
      continue;
    }

    // If there isn't an logical id in the replay's inputs and there's a
    // mismatch in replay_expr's and target_expr's split factor (if a split
    // expr), continue
    if (replay_expr->isA<Split>()) {
      auto r_split = replay_expr->as<Split>();
      auto t_split = target_expr->as<Split>();
      if (!r_split->factor()->sameAs(t_split->factor()) ||
          r_split->innerSplit() != t_split->innerSplit()) {
        ERROR_ON_FAILURE(!replay_has_logical_inp);
        continue;
      }
    }

    // Need to match swizzle type and parameters if
    //  not skipping swizzles in this mapping pass.
    if (!(skip_replay_swizzle_ || skip_target_swizzle_) &&
        replay_expr->isA<Swizzle2D>()) {
      auto r_swizzle_2d = replay_expr->as<Swizzle2D>();
      auto t_swizzle_2d = target_expr->as<Swizzle2D>();
      if (!(r_swizzle_2d->swizzleType() == t_swizzle_2d->swizzleType())) {
        ERROR_ON_FAILURE(!replay_has_logical_inp);
        continue;
      }
    }

    if (replay_expr->isA<Resize>()) {
      auto r_resize = replay_expr->as<Resize>();
      auto t_resize = target_expr->as<Resize>();
      if (!r_resize->leftExpand()->sameAs(t_resize->leftExpand()) ||
          !r_resize->rightExpand()->sameAs(t_resize->rightExpand())) {
        ERROR_ON_FAILURE(!replay_has_logical_inp);
        continue;
      }
    }

    // Take replay expr inputs out of map:
    for (const auto t_i : arange(target_id_inps.size())) {
      auto t_inp = target_id_inps[t_i];
      auto r_orig_inp = target2replay_id_map_.at(t_inp);
      auto r_maybe_forwarded_inp = replay_inps[t_i];

      // Remove original target2replay_it->second if it's in loop_ids
      if (loop_ids_.find(r_orig_inp) != loop_ids_.end()) {
        loop_ids_.erase(r_orig_inp);
      }

      // Check if we used a forwarded id, if so add forwarded id's to tracking.
      if (r_orig_inp != r_maybe_forwarded_inp) {
        forwarded_ids_.emplace_back(r_orig_inp);
      }
    }

    // Add outputs to map.
    for (const auto i : arange(target_expr->outputs().size())) {
      auto t_out = target_expr->output(i);
      auto r_out = replay_expr->output(i);
      if (t_out->getValType() == ValType::IterDomain &&
          r_out->getValType() == ValType::IterDomain) {
        target2replay_id_map_[t_out->as<IterDomain>()] =
            r_out->as<IterDomain>();
        loop_ids_[r_out->as<IterDomain>()] = counter++;
      }
    }

    if (skip_target_swizzle_ || skip_replay_swizzle_) {
      // Progress through all swizzle ops if we are skipping
      //  swizzles on the mapping.
      skipSwizzles(target_id2expr_map, replay_id2expr_map);
    }

    if (skip_resize) {
      skipResizes(target_exprs, replay_exprs);
    }
  }
}

#undef ERROR_ON_FAILURE

// Find the first position i where td1[i] is not the same as td2[i].
// "Same" means the DAG to generate td1[i] and td2[i] are the
// equivelent.
int64_t BestEffortReplay::findFirstMismatchedID(
    const TensorDomain* td1,
    const TensorDomain* td2) {
  std::unordered_map<IterDomain*, IterDomain*> id_map;
  auto rd1 = td1->maybeRoot();
  auto rd2 = td2->maybeRoot();
  std::unordered_set<IterDomain*> rd2_set(
      td2->maybeRoot().begin(), td2->maybeRoot().end());

  // Find matching root IterDomains, we could make this O(nlog(n)) if we could
  // sort IterDomains.
  for (auto rd1i : rd1) {
    for (auto rd2i : rd2) {
      if (rd1i->sameAs(rd2i) && rd2_set.find(rd2i) != rd2_set.end()) {
        id_map[rd1i] = rd2i;
        rd2_set.erase(rd2i);
        break;
      }
    }
  }

  BestEffortReplay ber(td2->loop(), td1->loop(), id_map);
  for (const auto i :
       arange((int64_t)std::max(td1->loop().size(), td2->loop().size()))) {
    if (ber.getReplay().find(td1->axis(i)) == ber.getReplay().end()) {
      return i;
    }
    // Order is important.
    auto td2_axis = ber.getReplay().at(td1->axis(i));
    if (td2->axis(i) != td2_axis) {
      return i;
    }
  }
  return std::min(td1->nDims(), td2->nDims());
}

ForwardingInfo::ForwardingInfo(
    const TensorView* producer,
    const TensorView* consumer) {
  // No forwarding unless this is broadcast or squeeze
  if (!dynamic_cast<BroadcastOp*>(consumer->definition()) &&
      !dynamic_cast<SqueezeOp*>(consumer->definition())) {
    return;
  }

  // Active indicates the TV that has axes the other TV does not. For
  // broadcast this is the consumer squeeze the producer.
  //
  // Either producer or consumer maps depending on operation
  std::unordered_map<IterDomain*, IterDomain*>* active_forwarding_map = nullptr;
  std::unordered_map<IterDomain*, std::vector<IterDomain*>>*
      active_compliment_map = nullptr;

  // Either squeeze or broadcast dimension flags depending on operation
  const std::vector<bool>* active_dim_flags = nullptr;

  // Either producer or consumer depending on operation
  std::vector<IterDomain*> active_logical_dom;
  const TensorView* active_tv = nullptr;

  if (auto bop = dynamic_cast<BroadcastOp*>(consumer->definition())) {
    active_forwarding_map = &consumer_forwarding_map;
    active_compliment_map = &consumer_compliment_map;
    active_dim_flags = &bop->getBroadcastDimFlags();
    active_logical_dom = consumer->getLogicalDomain();
    active_tv = consumer;
  } else if (auto sop = dynamic_cast<SqueezeOp*>(consumer->definition())) {
    active_forwarding_map = &producer_forwarding_map;
    active_compliment_map = &producer_compliment_map;
    active_dim_flags = &sop->getSqueezeDimFlags();
    active_logical_dom =
        TensorDomain::noReductions(producer->getLogicalDomain());
    active_tv = producer;
  } else {
    NVF_THROW("Should not be reachable");
  }

  NVF_ERROR(active_logical_dom.size() == active_dim_flags->size());

  // Collect which root ids are only in active_tv but not in the inactive
  // tensor.
  //
  // Initialize which id's should beforwarded.
  std::unordered_set<IterDomain*> forwarded_ids;
  for (auto i : arange(active_dim_flags->size())) {
    if (active_dim_flags->at(i)) {
      forwarded_ids.emplace(active_logical_dom.at(i));
    }
  }

  // We have root axes in active_tv that don't exist in the inactive tensor,
  // now forward those to include all id's in active_tv comprised of only axes
  // not in the inactive tensor.
  auto active_tv_history = StmtSort::getExprsTo(std::vector<Val*>(
      active_tv->domain()->loop().begin(), active_tv->domain()->loop().end()));

  auto isInForwardIdSet = [&forwarded_ids](IterDomain* input_id) {
    return forwarded_ids.count(input_id) > 0;
  };

  for (auto expr : active_tv_history) {
    auto input_ids = ir_utils::filterByType<IterDomain>(expr->inputs());
    // If expr inputs are all in forwarded_ids, then so are all outputs
    if (std::all_of(input_ids.begin(), input_ids.end(), isInForwardIdSet)) {
      for (auto output_ids :
           ir_utils::filterByType<IterDomain>(expr->outputs())) {
        forwarded_ids.emplace(output_ids);
      }
    } else if (
        expr->isA<Merge>() &&
        std::any_of(input_ids.begin(), input_ids.end(), isInForwardIdSet)) {
      auto merge_expr = expr->as<Merge>();
      // If
      // - one of the inputs is made of id's in active_tv that don't map to
      //   the inactive tensor,
      // - && the other input maps to an id in both the active and inactive
      //   tensor
      // - && this is a merge
      //
      // For the sake of BestEffortReplay we can forward the input mapping
      //   to both the active and inactive tensor to the output of the
      //   expression
      IterDomain* forwarded_id = nullptr;
      IterDomain* compliment_id = nullptr;

      for (auto input_id : input_ids) {
        if (!isInForwardIdSet(input_id)) {
          NVF_ERROR(forwarded_id == nullptr);
          forwarded_id = input_id;
          active_forwarding_map->emplace(
              std::make_pair(input_id, merge_expr->out()));
        } else {
          NVF_ERROR(compliment_id == nullptr);
          compliment_id = input_id;
        }
      }

      NVF_ERROR(forwarded_id != nullptr);
      NVF_ERROR(compliment_id != nullptr);

      // Set up compliment map
      active_compliment_map->emplace(
          forwarded_id, std::vector<IterDomain*>{compliment_id});
    }
  }
}

namespace {

// Trace chain of swizzles until reaching
//  an IterDomain that's either a loop or
//  not a producer of any swizzle.
IterDomain* getSwizzleFinalOutput(
    IterDomain* id,
    const std::unordered_map<IterDomain*, Expr*>& id2expr) {
  // Note: currently not supporting swizzling consumer of another
  //  swizzle id, so this should terminate in 1 iter, but eventually
  //  will try to support stacked swizzles so keeping this pass
  //  generic.
  while (true) {
    auto expr_it = id2expr.find(id);

    // This means id is a loop that doesn't
    //  have any consumers. Stop iteration in this case.
    if (expr_it == id2expr.end()) {
      break;
    }

    if (auto expr = dynamic_cast<Swizzle2D*>(expr_it->second)) {
      // In the case of 2D swizzle ops, just forward
      //  inX to outX and inY to outY.
      if (id == expr->inX()) {
        id = expr->outX();
      } else {
        NVF_ERROR(
            id == expr->inY(),
            "unknown input to swizzle op",
            id->toString(),
            expr->toString());
        id = expr->outY();
      }
    } else {
      // Probably unreachable but if the expression
      //  is unknown type assume it is not a swizzle op.
      break;
    }
  }

  return id;
}

bool isSwizzleInput(
    IterDomain* input_id,
    const std::unordered_map<IterDomain*, Expr*>& id2expr) {
  auto user_expr_it = id2expr.find(input_id);

  if (user_expr_it == id2expr.end()) {
    return false;
  }

  return user_expr_it->second->isA<Swizzle2D>();
}

} // namespace

void BestEffortReplay::addComplimentLeafIDs(
    const std::unordered_map<IterDomain*, IterDomain*>& forwarding_map,
    const std::unordered_map<IterDomain*, std::vector<IterDomain*>>&
        compliment_map) {
  // ID's could go through more than one forward iteration in the map before it
  // terminates. Grab every id between the forwarded id, and what it was
  // forwarded to
  std::function<void(IterDomain*, std::vector<IterDomain*>&)>
      collectForwardedIds =
          [&forwarding_map, &collectForwardedIds](
              IterDomain* forward_id,
              std::vector<IterDomain*>& forwarded_ids) -> void {
    if (forwarding_map.find(forward_id) != forwarding_map.end()) {
      forwarded_ids.emplace_back(forward_id);
      collectForwardedIds(forwarding_map.at(forward_id), forwarded_ids);
    }
  };

  std::vector<IterDomain*> expanded_forwarded_ids;
  for (auto forwarded_id : forwarded_ids_) {
    collectForwardedIds(forwarded_id, expanded_forwarded_ids);
  }

  // Grab all compliments of forwarded ids.
  std::vector<IterDomain*> compliments;
  for (auto forwarded_id : expanded_forwarded_ids) {
    auto compliment_map_it = compliment_map.find(forwarded_id);
    NVF_ERROR(
        compliment_map_it != compliment_map.end(),
        "Issue tracking forwarded broadcast merges in best effort replay. ",
        forwarded_id->toString());
    compliments.insert(
        compliments.end(),
        compliment_map_it->second.begin(),
        compliment_map_it->second.end());
  }

  // Grab all exprs used to make the forwarded compliments
  auto compliment_exprs =
      StmtSort::getExprsTo({compliments.begin(), compliments.end()});

  // Figure out if there are any loop id in compliment_exprs that aren't
  // the forwarded id
  std::unordered_map<IterDomain*, size_t> loop_ids;

  for (auto expr : compliment_exprs) {
    for (auto inp : ir_utils::filterByType<IterDomain>(expr->inputs())) {
      loop_ids.erase(inp);
    }
    for (auto out : ir_utils::filterByType<IterDomain>(expr->outputs())) {
      // If we used the comliment for forwarded don't add to loop nodes.
      if (std::find(compliments.begin(), compliments.end(), out) ==
          compliments.end()) {
        loop_ids.emplace(out, counter++);
      }
    }
  }

  loop_ids_.insert(loop_ids.begin(), loop_ids.end());
}

BestEffortReplay BestEffortReplay::replayCasP(
    const TensorView* consumer,
    const TensorView* producer,
    int64_t producer_compute_at_axis,
    const LogicalDomainMap& logical_map,
    bool skip_consumer_swizzle,
    bool skip_producer_swizzle,
    bool skip_resize) {
  if (producer_compute_at_axis < 0) {
    producer_compute_at_axis += producer->nDims() + 1;
  }

  NVF_ERROR(
      producer_compute_at_axis >= 0 &&
          producer_compute_at_axis <= producer->nDims(),
      "Invalid axis provided to BestEffortReplay::replayCasP.");

  // producer ids we need to match in consumer
  std::vector<IterDomain*> producer_CA_ids(
      producer->getLoopDomain().begin(),
      producer->getLoopDomain().begin() + producer_compute_at_axis);
  producer_CA_ids = TensorDomain::noReductions(producer_CA_ids);

  // If producer has an rfactor, that's what will match to the consumer
  std::vector<IterDomain*> producer_logical = producer->getLogicalDomain();

  // Figure out all inputs required to generate the compute_at dimensions. We
  // need all deps because inputs on producer may be in getLogicaoDomain, but
  // we may need in logical domain
  auto all_CA_id_deps = DependencyCheck::getAllValsBetween(
      {producer_logical.begin(), producer_logical.end()},
      {producer_CA_ids.begin(), producer_CA_ids.end()});

  // Figure out minimal set of root IDs needed to produce producer_CA_ids:
  std::unordered_set<IterDomain*> producer_CA_root_ids;
  for (IterDomain* id : producer_logical) {
    if (std::find(all_CA_id_deps.begin(), all_CA_id_deps.end(), id) !=
        all_CA_id_deps.end()) {
      producer_CA_root_ids.emplace(id);
    }
  }

  const auto p2c_logical_map = logical_map.mapProducerToConsumer(
      producer->domain(), consumer->domain(), producer_CA_root_ids);

  // See FusionAdvancedComputeAt7 for an example of the forwarding logic
  ForwardingInfo forwarding_info(producer, consumer);

  auto consumer_replay = BestEffortReplay(
      consumer->getLoopDomain(),
      producer_CA_ids,
      p2c_logical_map,
      forwarding_info.consumer_forwarding_map,
      forwarding_info.producer_forwarding_map,
      skip_consumer_swizzle,
      skip_producer_swizzle,
      skip_resize);

  consumer_replay.addComplimentLeafIDs(
      forwarding_info.consumer_forwarding_map,
      forwarding_info.consumer_compliment_map);

  return consumer_replay;
}

// Runs a best effort replay that ignores broadcast axes that appear in
// consumer that are not mapped to producer in logical_map.
BestEffortReplay BestEffortReplay::replayPasC(
    const TensorView* producer,
    const TensorView* consumer,
    int64_t consumer_compute_at_axis,
    const LogicalDomainMap& logical_map,
    bool skip_producer_swizzle,
    bool skip_consumer_swizzle,
    bool skip_resize) {
  if (consumer_compute_at_axis < 0) {
    consumer_compute_at_axis += consumer->nDims() + 1;
  }
  NVF_ERROR(
      consumer_compute_at_axis >= 0 &&
          consumer_compute_at_axis <= consumer->nDims(),
      "Invalid axis provided to BestEffortReplay::replayPasC.");

  // consumer ids we need to match in producer
  std::vector<IterDomain*> consumer_CA_ids(
      consumer->getLoopDomain().begin(),
      consumer->getLoopDomain().begin() + consumer_compute_at_axis);

  // Figure out all inputs required to generate the compute_at dimensions
  auto consumer_CA_root_vals = IterVisitor::getInputsTo(
      std::vector<Val*>(consumer_CA_ids.begin(), consumer_CA_ids.end()));

  std::unordered_set<IterDomain*> consumer_CA_root_ids;
  for (auto val : consumer_CA_root_vals) {
    if (val->getValType().value() == ValType::IterDomain) {
      consumer_CA_root_ids.emplace(val->as<IterDomain>());
    }
  }

  const auto c2p_logical_map = logical_map.mapConsumerToProducer(
      consumer->domain(), producer->domain(), consumer_CA_root_ids);

  ForwardingInfo forwarding_info(producer, consumer);

  // Instead of replaying from the root, lets try to play forward the history
  // of producer if they match ops on consumer. Enforce if we modify an
  // rfactor axis that those ops must match.
  auto producer_replay = BestEffortReplay(
      producer->getLoopDomain(),
      consumer_CA_ids,
      c2p_logical_map,
      forwarding_info.producer_forwarding_map,
      forwarding_info.consumer_forwarding_map,
      skip_producer_swizzle,
      skip_consumer_swizzle,
      skip_resize);

  producer_replay.addComplimentLeafIDs(
      forwarding_info.producer_forwarding_map,
      forwarding_info.producer_compliment_map);

  return producer_replay;
}

void BestEffortReplay::skipSwizzles(
    const std::unordered_map<IterDomain*, Expr*>& target_id2expr,
    const std::unordered_map<IterDomain*, Expr*>& replay_id2expr) {
  // Update target2replay map
  bool updated = true;

  while (updated) {
    updated = false;
    for (auto it : target2replay_id_map_) {
      if ((isSwizzleInput(it.first, target_id2expr) && skip_target_swizzle_) ||
          (isSwizzleInput(it.second, replay_id2expr) && skip_replay_swizzle_)) {
        updated = true;

        auto new_target = skip_target_swizzle_
            ? getSwizzleFinalOutput(it.first, target_id2expr)
            : it.first;
        auto new_replay = skip_replay_swizzle_
            ? getSwizzleFinalOutput(it.second, replay_id2expr)
            : it.second;

        // new_target and new_replay will now be the final output
        //  skipping all swizzles in between. We'd need to
        //  update the mapping and loop ids to the final outputs.
        target2replay_id_map_.erase(it.first);
        NVF_ERROR(
            target2replay_id_map_.insert(std::make_pair(new_target, new_replay))
                .second,
            "Unexpected replay loop");
        // Progress the loop ids if the replay is updated
        if (it.second != new_replay &&
            loop_ids_.find(it.second) != loop_ids_.end()) {
          loop_ids_.erase(it.second);
          loop_ids_[new_replay] = counter++;
        }
        break;
      }
    }
  }
}

// Same logic as skipSwizzles
void BestEffortReplay::skipResizes(
    const std::vector<Expr*>& target_exprs,
    const std::vector<Expr*>& replay_exprs) {
  auto getResizeUse = [](IterDomain* id,
                         const std::vector<Expr*>& exprs) -> Resize* {
    for (auto id_use : id->uses()) {
      if (std::find(exprs.begin(), exprs.end(), id_use) == exprs.end()) {
        continue;
      }
      return dynamic_cast<Resize*>(id_use);
    }
    return nullptr;
  };

  bool updated = true;

  while (updated) {
    updated = false;
    for (auto it : target2replay_id_map_) {
      auto target_id = it.first;
      auto new_target_id = target_id;
      auto replay_id = it.second;
      auto new_replay_id = replay_id;
      if (auto target_resize = getResizeUse(target_id, target_exprs);
          target_resize != nullptr) {
        new_target_id = target_resize->out();
        skipped_resize_id_map_.emplace(target_id, new_target_id);
      }
      if (auto replay_resize = getResizeUse(replay_id, replay_exprs);
          replay_resize != nullptr) {
        new_replay_id = replay_resize->out();
        skipped_resize_id_map_.emplace(replay_id, new_replay_id);
      }

      if (new_target_id == target_id && new_replay_id == replay_id) {
        continue;
      }

      target2replay_id_map_.erase(target_id);
      NVF_ERROR(
          target2replay_id_map_
              .insert(std::make_pair(new_target_id, new_replay_id))
              .second,
          "Unexpected replay loop");
      // Progress the loop ids if the replay is updated
      if (replay_id != new_replay_id &&
          loop_ids_.find(replay_id) != loop_ids_.end()) {
        loop_ids_.erase(replay_id);
        loop_ids_[new_replay_id] = counter++;
      }
      updated = true;
      break;
    }
  }
}

DisjointSets<IterDomain*> BestEffortReplay::getIterDomainEquivalence() {
  DisjointSets<IterDomain*> result;
  using IterDomainMap = std::unordered_map<IterDomain*, IterDomain*>;
  const std::array<IterDomainMap*, 4> maps = {
      &target2replay_id_map_,
      &replay_forward_id_map_,
      &target_forward_id_map_,
      &skipped_resize_id_map_};
  for (auto map : maps) {
    // Sort the keys so that they appear in a deterministic order
    for (auto key : getSortedKeys(*map, Statement::lessThan)) {
      result.mapEntries(key, map->at(key));
    }
  }
  return result;
}

} // namespace nvfuser
