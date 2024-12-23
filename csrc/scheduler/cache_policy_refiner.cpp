// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/debug_utils.h>

namespace nvfuser {

namespace {

template <typename... Args>
void vlog(const Args&... args) {
  scheduler_debug_utils::log("[cache_policy_refiner] ", args...);
}

// Returns whether a pointwise expression `expr` expands its input operand
// `in_tv`.
bool pointwiseExpands(const Expr* expr, const TensorView* in_tv) {
  NVF_CHECK(
      expr->outputs().size() == 1,
      "A pointwise expression is expected to have one output: ",
      expr->toString());
  const Val* out = expr->output(0);

  if (!out->isA<TensorView>()) {
    return false;
  }
  const auto* out_tv = out->as<TensorView>();

  auto logical_domain_map = PairwiseLogicalDomainMap(in_tv, out_tv)
                                .mapBroadcast(true)
                                .mapProducerToConsumer();
  return std::find_if(
             logical_domain_map.begin(),
             logical_domain_map.end(),
             [](const auto& mapping) {
               return mapping.first->isBroadcast() &&
                   !mapping.second->isBroadcast();
             }) != logical_domain_map.end();
}

bool isLoadGlobalToLocal(const Expr* expr) {
  if (!expr->isA<LoadStoreOp>()) {
    return false;
  }
  const LoadStoreOp* ldst = expr->as<LoadStoreOp>();

  if (ldst->opType() != LoadStoreOpType::Set) {
    return false;
  }
  // It should not be necessary to check the output since it should be
  // always a TensorView as long as the input is a TensorView, but
  // just in case.
  if (!ldst->in()->isA<TensorView>() || !ldst->out()->isA<TensorView>()) {
    return false;
  }
  if (ldst->in()->as<TensorView>()->getMemoryType() != MemoryType::Global) {
    return false;
  }
  if (ldst->out()->as<TensorView>()->getMemoryType() != MemoryType::Local) {
    return false;
  }
  return true;
}

// Finds the first expanding use of `ldst`'s output, bypassing all pointwise
// operations.
const Expr* findExpand(const LoadStoreOp* ldst) {
  std::queue<const Expr*> q;
  std::unordered_set<const Expr*> visited;

  auto enqueueIfNotVisited = [&q, &visited](const Expr* expr) {
    if (visited.insert(expr).second) {
      q.push(expr);
    }
  };

  enqueueIfNotVisited(ldst);
  while (!q.empty()) {
    const Expr* def = q.front();
    q.pop();
    for (const Val* def_out : def->outputs()) {
      if (!def_out->isA<TensorView>()) {
        continue;
      }
      const TensorView* def_out_tv = def_out->as<TensorView>();

      for (const Expr* use : def_out->uses()) {
        if (use->isA<ExpandOp>()) {
          return use;
        }

        // Do not bypass another global-to-local load.  We could have a chain of
        // `ld.global -> st.global -> ld.global -> st.global -> ...`. If the
        // traversal doesn't stop at a ld.global, downstream exprs may be
        // traversed many times.
        if (isLoadGlobalToLocal(use)) {
          continue;
        }

        // Bypass BroadcastOps as well as pointwise ops.
        // ExpandingPointwise(BroadcastInDims(x)) is a common pattern for this
        // pass to recognize.
        if (ir_utils::isPointwiseTvOp(use) || use->isA<BroadcastOp>()) {
          if (pointwiseExpands(use, def_out_tv)) {
            return use;
          }
          enqueueIfNotVisited(use);
        }
      }
    }
  }

  return nullptr;
}

// Returns true if the cache policy is changed.
bool refineCachePolicy(LoadStoreOp* ldst) {
  vlog("Processing ", ldst->toString());

  const Expr* expand = findExpand(ldst);
  if (expand == nullptr) {
    vlog(
        "Skipped ",
        ldst->toString(),
        " because we cannot find the using expand.");
    return false;
  }

  auto target_cache_op = CacheOp::AllLevels;
  vlog(
      "Changed the cache op of ",
      ldst->toString(),
      " from ",
      ldst->cacheOp(),
      " to ",
      target_cache_op,
      " because it is expanded by ",
      expand->toString());
  ldst->setCacheOp(target_cache_op);
  return true;
}

} // namespace

void refineCachePolicy(Fusion* fusion) {
  for (Expr* expr : fusion->exprs()) {
    // Currently, we only change cache policy for global->local loads.
    if (isLoadGlobalToLocal(expr)) {
      refineCachePolicy(expr->as<LoadStoreOp>());
    }
  }
}

} // namespace nvfuser
