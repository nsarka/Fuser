// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <logical_domain_map.h>
#include <ops/utils.h>

#include <sstream>

namespace nvfuser {

std::unordered_map<IterDomain*, IterDomain*> LogicalDomainMap::
    mapProducerToConsumer(
        const TensorDomain* producer,
        const TensorDomain* consumer,
        const std::unordered_set<IterDomain*>& dims_to_map) const {
  return map(producer, consumer, dims_to_map, true);
}

std::unordered_map<IterDomain*, IterDomain*> LogicalDomainMap::
    mapProducerToConsumer(
        const TensorDomain* producer,
        const TensorDomain* consumer) const {
  std::unordered_set<IterDomain*> dims_to_map(
      producer->logical().begin(), producer->logical().end());
  return mapProducerToConsumer(producer, consumer, dims_to_map);
}

std::unordered_map<IterDomain*, IterDomain*> LogicalDomainMap::
    mapConsumerToProducer(
        const TensorDomain* consumer,
        const TensorDomain* producer,
        const std::unordered_set<IterDomain*>& dims_to_map) const {
  return map(producer, consumer, dims_to_map, false);
}

std::unordered_map<IterDomain*, IterDomain*> LogicalDomainMap::
    mapConsumerToProducer(
        const TensorDomain* consumer,
        const TensorDomain* producer) const {
  std::unordered_set<IterDomain*> dims_to_map(
      consumer->maybeRoot().begin(), consumer->maybeRoot().end());
  return mapConsumerToProducer(consumer, producer, dims_to_map);
}

PairwiseLogicalDomainMap::PairwiseLogicalDomainMap(
    const TensorView* producer,
    const TensorView* consumer)
    : producer_tv_(producer), consumer_tv_(consumer) {
  NVF_ERROR(producer != nullptr);
  NVF_ERROR(consumer != nullptr);
  NVF_ERROR(producer->fusion() == consumer->fusion());
  NVF_ERROR(consumer->definition() != nullptr);
  auto producer_tvs_of_consumer = ir_utils::producerTvsOf(consumer);
  // Make sure they are really a producer and its consumer
  NVF_ERROR(
      std::find(
          producer_tvs_of_consumer.begin(),
          producer_tvs_of_consumer.end(),
          producer) != producer_tvs_of_consumer.end(),
      "Expected ",
      producer,
      " is a producer of ",
      consumer,
      " but it is not.");
}

namespace {

// Returns producer IDs that don't map identically to consumer. A bool is
// returned indicating whether corresponding consumer IDs exists. For example,
// select doesn't have a consumer ID, whereas index_select does.
std::pair<std::unordered_set<IterDomain*>, bool> getNonMappingDomainInfo(
    const TensorView* producer_tv,
    const TensorView* consumer_tv,
    bool map_different_extents) {
  std::unordered_set<IterDomain*> non_mapping_ids;
  bool has_consumer_id = false;
  if (auto* sop = dynamic_cast<SelectOp*>(consumer_tv->definition())) {
    // indexed ID is indirectly accessed
    non_mapping_ids.insert(sop->getIndexedID());
    has_consumer_id = false;
  } else if (
      auto* sop = dynamic_cast<IndexSelectOp*>(consumer_tv->definition())) {
    // indexed ID is indirectly accessed
    if (producer_tv == sop->lookupTv()) {
      non_mapping_ids.insert(sop->getIndexedID());
      has_consumer_id = true;
    }
  } else if (auto* gop = dynamic_cast<GatherOp*>(consumer_tv->definition())) {
    // indexed ID is indirectly accessed
    if (producer_tv == gop->lookupTv()) {
      non_mapping_ids.insert(gop->getIndexedID());
      has_consumer_id = true;
    }
  } else if (
      auto* iaop =
          dynamic_cast<IndexPutAccumulateOp*>(consumer_tv->definition())) {
    // see [ Note -- IndexPutAccumulateOp semantics ]
    if (producer_tv == iaop->indexTv()) {
      // Indexing ID of index tv do not map to output.
      non_mapping_ids.insert(iaop->getIndexingID());
      has_consumer_id = true;
    } else if (producer_tv == iaop->valueTv()) {
      // indexing ID of value tv do not map to output.
      non_mapping_ids.insert(iaop->getIndexingIDOfValue());
      has_consumer_id = true;
    }
  } else if (auto* top = dynamic_cast<TopKOp*>(consumer_tv->definition());
             top != nullptr && !map_different_extents) {
    // For TopKOp, the topk dimension should not be mapped between producer and
    // consumer because they have different extents: input[topk_dim] = D,
    // output[topk_dim] = k (where k < D)
    if (producer_tv == top->in()) {
      auto producer_logical =
          TensorDomain::noReductions(producer_tv->getLogicalDomain());
      auto topk_dim = top->dim();
      NVF_ERROR(
          topk_dim >= 0 && (size_t)topk_dim < producer_logical.size(),
          "TopKOp dimension ",
          topk_dim,
          " is out of bounds for producer logical domain size ",
          producer_logical.size());
      non_mapping_ids.insert(producer_logical.at(topk_dim));
      has_consumer_id = true;
    }
  }

  return std::make_pair(non_mapping_ids, has_consumer_id);
}

} // namespace

std::unordered_map<IterDomain*, IterDomain*> PairwiseLogicalDomainMap::map(
    const TensorDomain* producer,
    const TensorDomain* consumer,
    const std::unordered_set<IterDomain*>& dims_to_map,
    bool producer_to_consumer) const {
  std::vector<bool> broadcast_flags;
  if (auto* bop = dynamic_cast<BroadcastOp*>(consumer_tv_->definition())) {
    broadcast_flags = bop->getBroadcastDimFlags();
  }

  std::vector<bool> squeeze_flags;
  if (auto* sop = dynamic_cast<SqueezeOp*>(consumer_tv_->definition())) {
    squeeze_flags = sop->getSqueezeDimFlags();
  }

  auto [non_mapping_producer_id, has_consumer_of_indexed_id] =
      getNonMappingDomainInfo(
          producer_tv_, consumer_tv_, map_different_extents_);

  std::unordered_map<IterDomain*, IterDomain*> dom_map;
  const auto producer_logical = TensorDomain::noReductions(producer->logical());
  const auto& consumer_root = consumer->maybeRoot();

  // Check following conditions and add key-value iterdomain pair to domain map:
  // 1. Do not map broadcast ID to non-broadcast ID unless map_broadcast_ =
  // true.
  // 2. Do not map Symbolic ID if the extents are not identical unless
  // map_symbolic_ = true.
  auto updatePairwiseLogicalDomainMap = [&](IterDomain* producer_id,
                                            IterDomain* consumer_id) {
    if (!map_broadcast_ &&
        producer_id->isBroadcast() != consumer_id->isBroadcast()) {
      return;
    }

    // Condition: At least one ID is symbolic.
    //
    // If map_symbolic_ is true:
    //   Map these IDs regardless of other considerations.
    //
    // If map_symbolic_ is false (default):
    //   Map these only if their extents are identical. IterType::Symbolic
    //   reflects that the extent might evaluate to 1 for some inputs, in which
    //   case it may be valid to use those domains in a broadcast op. If the
    //   extents are exactly the same between two aligned IterDomains, the
    //   Symbolic one will be concretized to the same IterType as the other, so
    //   they should be mapped with one another.
    if (!map_symbolic_ &&
        (producer_id->isSymbolic() || consumer_id->isSymbolic()) &&
        (!producer_id->extent()->sameAs(consumer_id->extent()))) {
      return;
    }

    IterDomain* map_key_id = producer_id;
    IterDomain* map_value_id = consumer_id;

    if (!producer_to_consumer) {
      std::swap(map_key_id, map_value_id);
    }

    if (dims_to_map.find(map_key_id) != dims_to_map.end()) {
      dom_map.insert(std::make_pair(map_key_id, map_value_id));
    }
  };

  // Assumes producer and consumer IDs to be trivially aligned and adds them to
  // domain map.
  auto pairwiseMapAllIds = [&](std::vector<IterDomain*> producer_ids,
                               std::vector<IterDomain*> consumer_ids) {
    NVF_ERROR(producer_ids.size() == consumer_ids.size());
    for (auto idx : arange(consumer_ids.size())) {
      IterDomain* producer_id = producer_ids.at(idx);
      IterDomain* consumer_id = consumer_ids.at(idx);
      if (producer_id == nullptr) {
        continue;
      }
      updatePairwiseLogicalDomainMap(producer_id, consumer_id);
    }
  };

  // For MatmulOp, use the corresponding mapped input iterdomains.
  if (auto* op = dynamic_cast<MatmulOp*>(consumer_tv_->definition())) {
    // Check if the producer is lhs/rhs input
    int64_t input_position = producer_tv_->sameAs(op->inA()) ? 0 : 1;
    auto out_size = consumer_root.size();

    // For MatmulOp, the input iterdomains at a given index do not necessarily
    // map to the output iterdomain at that index `mapMatmulOpIterDomains`
    // outputs a vector of the input iterdomains aligned to the output domain
    // which can be used to create a pairwise map between input-output. For eg:
    // 1. `[M, K] x [K, N] -> [M, N]`: For input A, there is no mapping between
    // input and output for index=2
    // 2. `[B, M, K] x [K, N] -> [B, M, N]`: For input B, the second iterdomain
    // maps to the third output iterdomain.
    const std::vector<IterDomain*>& aligned_producer_ids =
        ops::mapMatmulOpIterDomains(producer_logical, input_position, out_size);
    pairwiseMapAllIds(aligned_producer_ids, consumer_root);
    return dom_map;
  }

  // For ScaledMatmulOp, use the corresponding mapped input iterdomains.
  if (auto* op = dynamic_cast<ScaledMmaOp*>(consumer_tv_->definition())) {
    if (consumer_tv_ == op->out()) {
      if (producer_tv_ == op->matrix1() || producer_tv_ == op->matrix2()) {
        auto out_size = consumer_root.size();
        // Check if the producer is lhs/rhs input
        int64_t input_position = (producer_tv_ == op->matrix1()) ? 0 : 1;
        // For MatmulOp, the input iterdomains at a given index do not
        // necessarily map to the output iterdomain at that index
        // `mapMatmulOpIterDomains` outputs a vector of the input iterdomains
        // aligned to the output domain which can be used to create a pairwise
        // map between input-output. For eg:
        // 1. `[M, K] x [K, N] -> [M, N]`: For input A, there is no mapping
        // between input and output for index=2
        // 2. `[B, M, K] x [K, N] -> [B, M, N]`: For input B, the second
        // iterdomain maps to the third output iterdomain.
        const std::vector<IterDomain*>& aligned_producer_ids =
            ops::mapMatmulOpIterDomains(
                producer_logical, input_position, out_size);
        pairwiseMapAllIds(aligned_producer_ids, consumer_root);
        return dom_map;
      }
      // note op->beta() should map as a pointwise
      if (producer_tv_ != op->beta()) {
        return dom_map;
      }
    } else {
      // TODO: map output block scale as well
      return dom_map;
    }
  }

  if (auto* op = dynamic_cast<LinearOp*>(consumer_tv_->definition())) {
    auto out_size = consumer_root.size();

    // Check if the producer is A, B or bias.
    int64_t input_position = -1;
    if (producer == op->inA()->domain()) {
      input_position = 0;
    } else if (producer == op->inB()->domain()) {
      input_position = 1;
    } else if (producer == op->bias()->domain()) {
      input_position = 2;
    } else {
      NVF_THROW("Producer did not match any LinearOp input.")
    }

    // LinearOp:
    // inputs (0) = {*, in_features}
    // weight (1) = {out_features, in_features} / {in_features}
    // bias (2) = {out_features} / {}
    // output = {*, out_features} / {*}
    const bool k_bcast = op->inA()->getLogicalDomain().back()->isBroadcast();

    const std::vector<IterDomain*>& aligned_producer_ids =
        ops::mapLinearOpIterDomains(
            producer_logical, input_position, out_size, k_bcast);
    pairwiseMapAllIds(aligned_producer_ids, consumer_root);
    return dom_map;
  }

  if (auto* op = dynamic_cast<SdpaFwdOp*>(consumer_tv_->definition())) {
    // Note: Explicit handling of DIDx(D) until
    // https://github.com/NVIDIA/Fuser/issues/2563 is resolved. Producers:
    //   query = [DIDx(D)?, N, H, L, E]
    //   key = [DIDx(D)?, N, H, S, E]
    //   value = [DIDx(D)?, N, H, S, Ev]
    // Consumers:
    //   output = [DIDx(D)?, N, H, L, Ev]
    //   logsumexp = [DIDx(D)?, N, H, L]
    // Note: S, E are not mapped together in the producers and do not have any
    // mapping to the consumer.
    if (consumer_tv_->sameAs(op->philox_seed())) {
      return dom_map;
    }
    size_t num_device_dim = producer_logical.at(0)->isDeviceDim() ? 1 : 0;
    // Map N, H from any input (query/key/value)
    for (auto idx : arange(consumer_root.size())) {
      if (idx < (2 + num_device_dim)) {
        updatePairwiseLogicalDomainMap(
            producer_logical.at(idx), consumer_root.at(idx));
      }
      // Map L, E from query and value respectively
      if (idx == (2 + num_device_dim) && producer_tv_->sameAs(op->query())) {
        updatePairwiseLogicalDomainMap(
            producer_logical.at(idx), consumer_root.at(idx));
      }
      // Map Ev from value to output
      if (idx == (3 + num_device_dim) && producer_tv_->sameAs(op->value())) {
        updatePairwiseLogicalDomainMap(
            producer_logical.at(idx), consumer_root.at(idx));
      }
    }
    return dom_map;
  }

  if (auto* op = dynamic_cast<SdpaBwdOp*>(consumer_tv_->definition())) {
    // Producers:
    //   grad_attn = [DIDx(D)? N, H, L, Ev]
    //   query = [DIDx(D)? N, H, L, E]
    //   key = [DIDx(D)? N, H, S, E]
    //   value = [DIDx(D)? N, H, S, Ev]
    //   attn_out = [DIDx(D)? N, H, L, Ev]
    //   logsumexp = [DIDx(D)? N, H, L]
    // Consumers:
    //   grad_query = [DID(D)? N, H, L, E]
    //   grad_key = [DID(D)? N, H, S, E]
    //   grad_value = [DID(D)? N, H, S, Ev]

    if (producer_tv_->sameAs(op->philox_seed())) {
      return dom_map;
    }

    bool producer_has_s =
        producer_tv_->sameAs(op->key()) || producer_tv_->sameAs(op->value());
    bool consumer_has_s = consumer_tv_->sameAs(op->grad_key()) ||
        consumer_tv_->sameAs(op->grad_value());

    bool producer_has_e =
        producer_tv_->sameAs(op->query()) || producer_tv_->sameAs(op->key());
    bool consumer_has_e = consumer_tv_->sameAs(op->grad_query()) ||
        consumer_tv_->sameAs(op->grad_key());

    size_t num_device_dim =
        !producer_logical.empty() && producer_logical.at(0)->isDeviceDim() ? 1
                                                                           : 0;
    for (auto idx : arange(producer_logical.size())) {
      // Map N, H from all producers to consumers
      // producer/consumer[2] = L/S
      // producer/consumer[3] = E/Ev
      if ((idx < 2 + num_device_dim) ||
          (idx == 2 + num_device_dim && producer_has_s == consumer_has_s) ||
          (idx == 3 + num_device_dim && producer_has_e == consumer_has_e)) {
        updatePairwiseLogicalDomainMap(
            producer_logical.at(idx), consumer_root.at(idx));
      }
    }
    return dom_map;
  }

  if (auto* op = dynamic_cast<EmbeddingFwdOp*>(consumer_tv_->definition())) {
    // Producers:
    //   input = [*]
    //   weight = [V, embedding_dim]
    // Consumers:
    //   output = [*, embedding_dim]
    auto ndims_out = consumer_root.size();
    if (producer_tv_->sameAs(op->in())) {
      for (auto idx : arange(ndims_out - 1)) {
        updatePairwiseLogicalDomainMap(
            producer_logical.at(idx), consumer_root.at(idx));
      }
    }
    if (producer_tv_->sameAs(op->weight())) {
      updatePairwiseLogicalDomainMap(
          producer_logical.back(), consumer_root.back());
    }
    return dom_map;
  }

  // TODO: refactor to use getNonMappingDomainInfo instead.
  if (auto* op = dynamic_cast<GroupedMmaOp*>(consumer_tv_->definition())) {
    bool has_g = TensorDomain::noReductions(consumer_root).size() == 3;
    int64_t ndims_out = std::ssize(consumer_root);
    // [rk] is the reduction axis for the matmul operation, it only exists if k
    // is not broadcast.
    bool has_rk = consumer_root.back()->isReduction();
    int64_t out_non_rk_last_idx = has_rk ? ndims_out - 2 : ndims_out - 1;

    int64_t last_producer_idx = std::ssize(producer_logical) - 1;
    if (producer_tv_ == op->matrix1()) {
      // mapping m dimension;
      updatePairwiseLogicalDomainMap(
          producer_logical.at(last_producer_idx - 1),
          consumer_root.at(out_non_rk_last_idx - 1));
      // mapping rk/k dimension;
      if (has_rk) {
        updatePairwiseLogicalDomainMap(
            producer_logical.at(last_producer_idx), consumer_root.back());
      }
    } else if (producer_tv_ == op->matrix2()) {
      // mapping n dimension;
      updatePairwiseLogicalDomainMap(
          producer_logical.at(last_producer_idx),
          consumer_root.at(out_non_rk_last_idx));
      // mapping rk/k dimension;
      if (has_rk) {
        updatePairwiseLogicalDomainMap(
            producer_logical.at(last_producer_idx - 1), consumer_root.back());
      }
    } else if (producer_tv_ == op->offsets()) {
      // mapping g dimension;
      if (has_g) {
        updatePairwiseLogicalDomainMap(
            producer_logical.at(0), consumer_root.at(0));
      }
    } else if (producer_tv_ == op->scale1()) {
      // mapping m dimension;
      updatePairwiseLogicalDomainMap(
          producer_logical.at(last_producer_idx), consumer_root.at(0));
      // mapping g dimension;
      if (has_g) {
        updatePairwiseLogicalDomainMap(
            producer_logical.at(0), consumer_root.at(0));
      }
      // Note: since scale factor k' doesn't have to have the same extent as k,
      // we only map it to rk/k dimension if map_different_extents_ is true.
      if (map_different_extents_ && has_rk) {
        updatePairwiseLogicalDomainMap(
            producer_logical.at(last_producer_idx), consumer_root.back());
      }
    } else if (producer_tv_ == op->scale2()) {
      // mapping n dimension;
      updatePairwiseLogicalDomainMap(
          producer_logical.at(last_producer_idx),
          consumer_root.at(out_non_rk_last_idx));
      if (has_g) {
        // mapping g dimension;
        updatePairwiseLogicalDomainMap(
            producer_logical.at(0), consumer_root.at(0));
      }
      // Note: since scale factor k' doesn't have to have the same extent as k,
      // we only map it to rk/k dimension if map_different_extents_ is true.
      if (map_different_extents_ && has_rk) {
        updatePairwiseLogicalDomainMap(
            producer_logical.at(last_producer_idx - 1), consumer_root.back());
      }
    } else {
      NVF_THROW("Producer did not match any GroupedMmaOp input.");
    }
    return dom_map;
  }

  size_t itc = 0, itp = 0;
  while (itc < consumer_root.size() && itp < producer_logical.size()) {
    IterDomain* producer_id = producer_logical.at(itp);
    IterDomain* consumer_id = consumer_root.at(itc);

    // Conditions to check:
    // 1. Non mapping IDs (e.g., select)
    // 2. IDs that may have different extents (e.g., non indexed
    //  domains of torchGather)
    // 3. Squeeze and unsqueeze

    // Condition 1: when the producer ID is the dim of a select-like op, or when
    // it doesn't map to the output IDs, like indexing IDs of indexPutAccumulate
    if (non_mapping_producer_id.count(producer_id) != 0) {
      // If there's no corresponding consumer, skip the indexed producer
      if (!has_consumer_of_indexed_id) {
        itp++;
        continue;
      }
      // Skip both producer and consumer if mapping not allowed
      if (!map_indexed_domains_) {
        itp++;
        itc++;
        continue;
      }
    }

    // Condition 2: Different extents
    if (auto gop = dynamic_cast<GatherOp*>(consumer_tv_->definition());
        gop != nullptr && !gop->exactSizes() &&
        producer_tv_ == gop->lookupTv() &&
        non_mapping_producer_id.count(producer_id) == 0 &&
        !map_different_extents_) {
      itp++;
      itc++;
      continue;
    }

    // Condition 3: when the consumer ID is a new broadcast domain, there is no
    // mapping for it.
    if (!broadcast_flags.empty() && broadcast_flags.at(itc)) {
      NVF_ERROR(consumer_id->isBroadcast());
      itc++;
      continue;
    }

    // Condition 3: when the producer ID is a removed broadcast domain, there is
    // no mapping for it.
    if (!squeeze_flags.empty() && squeeze_flags.at(itp)) {
      // Dynamic IterDomains can be squeezed, in which case they must concretize
      // to broadcasts
      NVF_ERROR(producer_id->isBroadcast() || producer_id->isSymbolic());
      itp++;
      continue;
    }

    updatePairwiseLogicalDomainMap(producer_id, consumer_id);

    itc++;
    itp++;
  }
  return dom_map;
}

std::unordered_map<IterDomain*, IterDomain*> PairwiseLogicalDomainMap::
    mapProducerToConsumer(
        const std::unordered_set<IterDomain*>* dims_to_map) const {
  if (dims_to_map == nullptr) {
    return LogicalDomainMap::mapProducerToConsumer(
        producerTv()->domain(), consumerTv()->domain());
  } else {
    return LogicalDomainMap::mapProducerToConsumer(
        producerTv()->domain(), consumerTv()->domain(), *dims_to_map);
  }
}

std::unordered_map<IterDomain*, IterDomain*> PairwiseLogicalDomainMap::
    mapConsumerToProducer(
        const std::unordered_set<IterDomain*>* dims_to_map) const {
  if (dims_to_map == nullptr) {
    return LogicalDomainMap::mapConsumerToProducer(
        consumerTv()->domain(), producerTv()->domain());
  } else {
    return LogicalDomainMap::mapConsumerToProducer(
        consumerTv()->domain(), producerTv()->domain(), *dims_to_map);
  }
}

std::string PairwiseLogicalDomainMap::toString() const {
  std::stringstream ss;
  ss << "{producer: " << producerTv() << ", consumer: " << consumerTv();
  auto p2c = mapProducerToConsumer();
  for (auto pair : p2c) {
    ss << ", " << pair.first->toString() << " -> " << pair.second->toString();
  }
  ss << "}";
  return ss.str();
}

namespace {

template <typename T>
auto ensureMapping(
    T& m,
    const typename T::key_type& key,
    const typename T::mapped_type& init_value) {
  auto it = m.find(key);
  if (it == m.end()) {
    it = m.insert({key, init_value}).first;
  }
  return it;
}

TensorView* lookUpTv(const TensorDomain* td) {
  Fusion* fusion = FusionGuard::getCurFusion();
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->vals())) {
    if (tv->domain() == td) {
      return tv;
    }
  }
  return nullptr;
}

} // namespace

std::string DomainKey::toString() const {
  std::stringstream ss;
  if (id()) {
    ss << id();
  } else {
    ss << "null";
  }
  if (concreteId()) {
    ss << " (concrete: " << concreteId() << ")";
  }
  ss << " in ";
  if (td()) {
    auto tv = lookUpTv(td());
    NVF_ERROR(tv != nullptr, "No TV found for ", td()->toString());
    ss << "T" << tv->name() << "[ " << td()->logical() << " ]";
    if (td()->hasRoot()) {
      ss << " (root: [ " << td()->root() << " ])";
    }
  } else {
    ss << "null";
  }
  return ss.str();
}

UnmappableReductionDomains::UnmappableReductionDomains() {
  Fusion* fusion = FusionGuard::getCurFusion();
  traverse(fusion);
}

namespace {

//! Find all domains that a given domain is dependent on
class FindInputDomains : BackwardVisitor {
 private:
  FindInputDomains(TensorView* tv, const IterDomain* id)
      : BackwardVisitor(false), tv_(tv) {
    input_keys_.insert(DomainKey(tv_->domain(), id));
  }

  DomainKeySet find() {
    traverseTo({tv_});
    return input_keys_;
  }

  void dispatch(Expr* expr) override {
    for (auto output : expr->outputs()) {
      if (!output->isA<TensorView>()) {
        continue;
      }
      for (auto input : expr->inputs()) {
        if (!input->isA<TensorView>()) {
          continue;
        }
        propagate(input->as<TensorView>(), output->as<TensorView>());
      }
    }
  }

  void propagate(TensorView* in_tv, TensorView* out_tv) {
    auto c2p = PairwiseLogicalDomainMap(in_tv, out_tv).mapConsumerToProducer();
    for (auto root_dom : out_tv->getMaybeRootDomain()) {
      DomainKey out_key({out_tv->domain(), root_dom});
      if (input_keys_.find(out_key) == input_keys_.end()) {
        continue;
      }
      auto input_id_it = c2p.find(root_dom);
      if (input_id_it == c2p.end()) {
        continue;
      }
      DomainKey input_key(in_tv->domain(), input_id_it->second);
      input_keys_.insert(input_key);
    }
  }

 private:
  TensorView* tv_ = nullptr;
  DomainKeySet input_keys_;

 public:
  static DomainKeySet find(TensorView* tv, const IterDomain* id) {
    return FindInputDomains(tv, id).find();
  }
};

} // namespace

void UnmappableReductionDomains::handleReductionOutput(TensorView* out_tv) {
  std::vector<DomainKey> reduction_keys;
  for (const auto id : out_tv->getMaybeRootDomain()) {
    if (id->isReduction()) {
      DomainKey key(out_tv->domain(), id);
      reduction_keys.push_back(key);
      reduction_domains_.insert({key, {}});
    }
  }
  auto use_chains = DependencyCheck::getAllUseChains(out_tv);
  for (const auto& chain : use_chains) {
    for (const auto& tv : ir_utils::filterByType<TensorView>(chain)) {
      // Do not include the tensor itself in its consumers
      if (tv == out_tv) {
        continue;
      }
      const auto& root_domain = tv->getMaybeRootDomain();
      for (const auto& id : root_domain) {
        DomainKey consumer_key(tv->domain(), id);
        for (const auto& reduction_key : reduction_keys) {
          reduction_domains_.at(reduction_key).insert(consumer_key);
        }
      }
    }
  }
  for (const auto& reduction_key : reduction_keys) {
    reduction_domain_inputs_.insert(
        {reduction_key, FindInputDomains::find(out_tv, reduction_key.id())});
  }
}

void UnmappableReductionDomains::handle(ReductionOp* op) {
  // Builds a map from reduction domains to consumer domains.
  TensorView* out_tv = op->out()->as<TensorView>();
  handleReductionOutput(out_tv);
}

void UnmappableReductionDomains::handle(GroupedReductionOp* op) {
  // Builds a map from reduction domains to consumer domains.
  for (auto out : op->outputs()) {
    handleReductionOutput(out->as<TensorView>());
  }
}

void UnmappableReductionDomains::handle(MmaOp* mma) {
  // Builds a map from reduction domains to consumer domains.
  TensorView* out_tv = mma->out()->as<TensorView>();
  handleReductionOutput(out_tv);
}

void UnmappableReductionDomains::handle(WelfordOp* op) {
  // Builds a map from reduction domains to consumer domains.
  handleReductionOutput(op->outAvg()->as<TensorView>());
  handleReductionOutput(op->outVar()->as<TensorView>());
  handleReductionOutput(op->outN()->as<TensorView>());
}

bool UnmappableReductionDomains::isReductionOutputMapped(
    const DomainKeySet& consumer_domains,
    const ComputeAtLogicalDomainMap& logical_map) const {
  // Check each reduction domain if any of the consumer domains
  // conflicts with it
  for (const auto& kv : reduction_domains_) {
    const DomainKey& reduction_domain = kv.first;
    // Domains that must not be mapped with the reduction domain
    const DomainKeySet& incompatible_domains = kv.second;
    // Input domains to the reduction domain
    const auto& input_keys = reduction_domain_inputs_.at(reduction_domain);
    // Check if any of the consumer domains is an input to the
    // reduction or is mapped with the reduction input domains
    auto it = std::find_if(
        consumer_domains.begin(),
        consumer_domains.end(),
        [&](const auto& consumer_domain) {
          bool is_reduction_input =
              std::find(
                  input_keys.begin(), input_keys.end(), consumer_domain) !=
              input_keys.end();
          if (is_reduction_input) {
            return true;
          }
          bool is_mapped_with_reduction_input = std::any_of(
              input_keys.begin(), input_keys.end(), [&](const auto& input_key) {
                // canMap check requires concretized IDs for broadcast domain.
                // For example, in softmax there are two consecutive
                // reductions, one of the inputs to the 2nd reduction has a
                // broadcast domain and it is not concretized until its
                // consumer is visited where it will be concretized to its
                // consumer's non-broadcast domain.
                // Note that, the analaysis can only make a decision based on
                // the current visited exprs. If the skipped tensor turns out
                // to be an input to reduction domain and also used by the
                // consumers, it becomes a persistent tensor.
                if (input_key.id()->isBroadcast()) {
                  if (!logical_map.isConcretized(
                          input_key.td(), input_key.id())) {
                    return false;
                  }
                }
                return logical_map.canMap(
                    consumer_domain.td(),
                    consumer_domain.id(),
                    input_key.td(),
                    input_key.id());
              });
          return is_mapped_with_reduction_input;
        });

    // None of the consumer domains is used for the reduction.
    // None of the consumer domains is mapped with the reduction input domains.
    // Safe to map.
    if (it == consumer_domains.end()) {
      continue;
    }
    // A consumer domain that is an input to the reduction domain
    const DomainKey& input_to_reduction = *it;

    // Check if mapping input_to_reduction with the other domains in
    // consumer_domains. If there's a domain that is a consumer of the
    // reduction, they must not be mapped together
    for (const auto& consumer_domain : consumer_domains) {
      if (consumer_domain == input_to_reduction) {
        continue;
      }
      if (std::any_of(
              incompatible_domains.begin(),
              incompatible_domains.end(),
              [&](const DomainKey& incompatible_domain) {
                return logical_map.canMap(
                    consumer_domain.td(),
                    consumer_domain.id(),
                    incompatible_domain.td(),
                    incompatible_domain.id());
              })) {
        return true;
      }
    }
  }
  return false;
}

std::string UnmappableReductionDomains::toString() const {
  std::stringstream ss;
  ss << "Reduction-to-consumer map\n";
  for (const auto& kv : reduction_domains_) {
    ss << "\tReduction: " << kv.first.toString() << "\n";
    for (const auto& mapped_val : kv.second) {
      ss << "\t\tConsumer domain: " << mapped_val.toString() << "\n";
    }
  }

  ss << "Reduction-to-producer map\n";
  for (const auto& kv : reduction_domain_inputs_) {
    ss << "\tReduction: " << kv.first.toString() << "\n";
    for (const auto& mapped_val : kv.second) {
      ss << "\t\tProducer domain: " << mapped_val.toString() << "\n";
    }
  }

  return ss.str();
}

void ComputeAtLogicalDomainMap::build(bool map_through_reduction) {
  // Make sure we start from scratch. Throw away previous results.
  eq_set_.clear();
  bcast_map_.clear();
  new_broadcast_domains_.clear();
  removed_broadcast_domains_.clear();
  ComputeAtLogicalDomainMapBuilder builder(*this, map_through_reduction);
}

bool ComputeAtLogicalDomainMap::canMap(
    const TensorDomain* td_a,
    const IterDomain* id_a,
    const TensorDomain* td_b,
    const IterDomain* id_b) const {
  NVF_ERROR(
      td_a->isLogical(id_a) || td_a->isRoot(id_a),
      "Non-root domain is not supported: ",
      id_a);
  NVF_ERROR(
      td_b->isLogical(id_b) || td_b->isRoot(id_b),
      "Non-root domain is not supported: ",
      id_b);

  // Forward to overloaded functions
  if (!id_a->isBroadcast() && !id_b->isBroadcast()) {
    return canMap(DomainKey(td_a, id_a), DomainKey(td_b, id_b));
  } else if (!id_a->isBroadcast()) {
    return canMap(DomainKey(td_a, id_a), td_b, id_b);
  } else if (!id_b->isBroadcast()) {
    return canMap(DomainKey(td_b, id_b), td_a, id_a);
  }

  // At this point, both are broadcast. Every pair of concrete IDs of
  // both id_a and id_b needs to be looked at. Whether they are
  // mappable depends on whether the concrete IDs are broadcast or
  // not. Note that a broadcast axis is used a concrete ID when it is
  // part of an output tensor domain, i.e., when it never gets
  // concretized with any non-broadcast axis.

  // If there exists a pair of non-broadcast concrete IDs is not
  // mappable, id_a and id_b can't be mapped together. Otherwise, they
  // can be mapped when there is any mappable pair is found.
  bool mappable_pair_found = false;
  for (const auto& key_a : getConcretizedKeys(td_a, id_a)) {
    for (const auto& key_b : getConcretizedKeys(td_b, id_b)) {
      const bool mappable = canMap(key_a, key_b);
      mappable_pair_found = mappable_pair_found || mappable;
      // If both concrete IDs are not broadcast, they must be
      // mappable.
      if (!key_a.concreteId()->isBroadcast() &&
          !key_b.concreteId()->isBroadcast() && !mappable) {
        return false;
      }
    }
  }

  return mappable_pair_found;
}

bool ComputeAtLogicalDomainMap::canMap(
    const DomainKey& key_a,
    const TensorDomain* td_b,
    const IterDomain* id_b) const {
  NVF_ERROR(
      td_b->isLogical(id_b) || td_b->isRoot(id_b),
      "Non-root domain is not supported: ",
      id_b);

  if (!id_b->isBroadcast()) {
    return canMap(key_a, DomainKey(td_b, id_b));
  }

  // If id_b is broadcast, look at all the concrete IDs that id_b may
  // be concretized to. Whether it is mappable with key_a depends on
  // whether key_a's concrete ID is also broadcast.
  // 1) key_a's concrete ID is also broadcast: They are mappable when
  // there is any mappable concrete ID exists in the concrete ID set
  // of id_b.
  // 2) key_a's concrete ID is not broadcast: Since key_a is indeed
  // concrete, it must be mappable with any of concrete ID of id_b,
  // except when a id_b concrete is broadcast.
  const bool key_a_bcast =
      key_a.concreteId() && key_a.concreteId()->isBroadcast();
  bool mappable_pair_found = false;
  for (const auto& key_b : getConcretizedKeys(td_b, id_b)) {
    const bool mappable = canMap(key_a, key_b);
    mappable_pair_found = mappable_pair_found || mappable;
    // If both concrete IDs are not broadcast, they must be mappable.
    if (!key_a_bcast && !key_b.concreteId()->isBroadcast() && !mappable) {
      return false;
    }
  }

  return mappable_pair_found;
}

bool ComputeAtLogicalDomainMap::canMap(
    const DomainKey& key_a,
    const DomainKey& key_b) const {
  return key_a == key_b || eq_set_.permissiveAreMapped(key_a, key_b);
}

void ComputeAtLogicalDomainMap::setAlias(
    const TensorDomain* td,
    const TensorDomain* td_alias) {
  auto tmp_bcast_map = bcast_map_;
  for (const auto& kv : bcast_map_) {
    const auto& bcast_map_key = kv.first;
    const auto& bcast_concrete_id_set = kv.second;
    if (bcast_map_key.td() == td) {
      DomainKey alias_key(td_alias, bcast_map_key.id());
      tmp_bcast_map.insert({alias_key, bcast_concrete_id_set});
    }
  }
  bcast_map_ = tmp_bcast_map;

  auto all_elements = eq_set_.getAllElements();
  for (const auto& key : all_elements.vector()) {
    if (key.td() == td) {
      DomainKey alias_key(td_alias, key.id(), key.concreteId());
      eq_set_.mapEntries(key, alias_key);
    }
  }

  auto tmp_new_broadcast_domains = new_broadcast_domains_;
  for (const auto& key : new_broadcast_domains_) {
    if (key.td() == td) {
      DomainKey alias_key(td_alias, key.id());
      tmp_new_broadcast_domains.insert(alias_key);
    }
  }
  new_broadcast_domains_ = tmp_new_broadcast_domains;

  auto tmp_removed_broadcast_domains = removed_broadcast_domains_;
  for (const auto& key : removed_broadcast_domains_) {
    if (key.td() == td) {
      DomainKey alias_key(td_alias, key.id());
      tmp_removed_broadcast_domains.insert(alias_key);
    }
  }
  removed_broadcast_domains_ = tmp_removed_broadcast_domains;
}

std::vector<DomainKey> ComputeAtLogicalDomainMap::getConcretizedKeys(
    const TensorDomain* td,
    const IterDomain* id) const {
  DomainKey key(td, id);
  auto it = bcast_map_.find(key);
  NVF_ERROR(it != bcast_map_.end(), "Not found: ", key.toString());
  std::vector<DomainKey> domains;
  std::transform(
      it->second.begin(),
      it->second.end(),
      std::back_inserter(domains),
      [&](const IterDomain* concrete_id) {
        return DomainKey(td, id, concrete_id);
      });
  return domains;
}

std::unordered_set<const IterDomain*>& ComputeAtLogicalDomainMap::
    getConcretizedDomains(const TensorDomain* td, const IterDomain* id) {
  DomainKey key(td, id);
  auto it = bcast_map_.find(key);
  NVF_ERROR(it != bcast_map_.end(), "Not found: ", key.toString());
  return it->second;
}

bool ComputeAtLogicalDomainMap::isConcretized(
    const TensorDomain* td,
    const IterDomain* id) const {
  DomainKey key(td, id);
  auto it = bcast_map_.find(key);
  return it != bcast_map_.end();
}

std::unordered_map<IterDomain*, IterDomain*> ComputeAtLogicalDomainMap::
    mapBestEffort(
        const TensorDomain* from_td,
        const std::vector<IterDomain*>& from_dom,
        const TensorDomain* to_td,
        const std::vector<IterDomain*>& to_dom) const {
  std::unordered_map<IterDomain*, IterDomain*> id_map;
  for (auto& from_id : from_dom) {
    for (const auto& to_id : to_dom) {
      if (canMap(from_td, from_id, to_td, to_id)) {
        NVF_ERROR(
            id_map.insert({from_id, to_id}).second,
            "Multiple matching ID detected for ",
            from_id);
      }
    }
  }
  return id_map;
}

std::unordered_map<IterDomain*, IterDomain*> ComputeAtLogicalDomainMap::map(
    const TensorDomain* producer,
    const TensorDomain* consumer,
    const std::unordered_set<IterDomain*>& dims_to_map,
    bool producer_to_consumer) const {
  const auto& producer_logical =
      TensorDomain::noReductions(producer->logical());
  const auto& consumer_root = consumer->maybeRoot();
  const TensorDomain* from_td = producer_to_consumer ? producer : consumer;
  const TensorDomain* to_td = producer_to_consumer ? consumer : producer;
  const auto& from_ids =
      producer_to_consumer ? producer_logical : consumer_root;
  const auto& to_ids = producer_to_consumer ? consumer_root : producer_logical;
  std::unordered_map<IterDomain*, IterDomain*> id_map =
      mapBestEffort(from_td, from_ids, to_td, to_ids);
  for (auto& from_id : from_ids) {
    if (dims_to_map.find(from_id) == dims_to_map.end()) {
      // Remove mapping if exists
      id_map.erase(from_id);
      continue;
    }
    if (id_map.find(from_id) != id_map.end()) {
      continue;
    }
    // Matching ID not found. It's an error unless the following three cases:
    // 1. from_id is a new broadcast of a consumer domain; or
    // 2. from_id is a removed broadcast of a producer domain; or
    // 3. from_id is a window axis of a consumer domain; or
    // 4. from_id is a ViewAsScalar domain
    // Note that reduction domains are removed from the producer logical domain.
    if ((!producer_to_consumer &&
         (new_broadcast_domains_.find(DomainKey(from_td, from_id)) !=
              new_broadcast_domains_.end() ||
          from_id->getIterType() == IterType::VectorComponent ||
          (window_axes_.count(from_id) > 0))) ||
        (producer_to_consumer &&
         removed_broadcast_domains_.find(DomainKey(from_td, from_id)) !=
             removed_broadcast_domains_.end())) {
      continue;
    }
    NVF_THROW(
        "Mapping IterDomain ",
        from_id,
        " of ",
        from_td,
        " not possible as it would require recomputing the source tensor.",
        " Producer logical: ",
        producer_logical,
        ". Consumer root: ",
        consumer_root,
        ". Mapping: ",
        this->toString());
  }
  return id_map;
}

std::unordered_set<IterDomain*> ComputeAtLogicalDomainMap::getMappableDims(
    const TensorDomain* producer,
    const TensorDomain* consumer) const {
  //! This funciton previously used mapBestEffort but it can fail when
  //! a domain is mapped to multitple domains, which can happen with
  //! views. Since we only need to find mappable domains, just
  //! grab any domain that is mapped in a pairwise way.

  const auto& producer_logical = producer->logical();
  const auto& consumer_root = consumer->maybeRoot();

  std::unordered_set<IterDomain*> mappable_ids;

  for (const auto& p_id : producer_logical) {
    for (const auto& c_id : consumer_root) {
      if (canMap(producer, p_id, consumer, c_id)) {
        mappable_ids.emplace(p_id);
        mappable_ids.emplace(c_id);
      }
    }
  }

  return mappable_ids;
}

std::string ComputeAtLogicalDomainMap::toString() const {
  return eq_set_.toString();
}

ComputeAtLogicalDomainMapBuilder::ComputeAtLogicalDomainMapBuilder(
    ComputeAtLogicalDomainMap& logical_map,
    bool map_through_reduction)
    : BackwardVisitor(false),
      logical_map_(logical_map),
      map_through_reduction_(map_through_reduction) {
  Fusion* fusion = FusionGuard::getCurFusion();
  NVF_ERROR(fusion != nullptr);
  traverseTo(fusion->outputs(), false);
  if (!pending_map_.empty()) {
    std::stringstream ss;
    ss << "pending map:\n";
    for (auto& kv : pending_map_) {
      ss << "\t" << kv.first.toString() << "\n";
      for (auto& dk : kv.second) {
        ss << "\t\t" << dk.toString() << "\n";
      }
    }
    debug() << ss.str();
  }
  NVF_ERROR(pending_map_.empty());
}

// Set concrete domains for broadcast domains that never get joined
// with a concrete domain. Just set its own domain as a concrete
// domain, which is not concrete but is sufficient for this analysis.
void ComputeAtLogicalDomainMapBuilder::initializeBcastMap(
    const TensorView* tv,
    const IterDomain* id) {
  NVF_ERROR(id->isBroadcast(), "Not a broadcast axis");
  auto key = DomainKey(tv->domain(), id);
  auto it = logical_map_.bcast_map_.find(key);
  if (it != logical_map_.bcast_map_.end()) {
    // already initialized.
    return;
  }

  // This initialization of the entry for the broadcast ID should only
  // happen when the broadcast has no further consumer ID, including
  // resolved non-broadcast IDs. An equivalent condition is that the
  // pairwise map has no mapping for the broadcast.
  for (auto consumer : ir_utils::consumerTvsOf(tv)) {
    const auto p2c =
        PairwiseLogicalDomainMap(tv, consumer).mapProducerToConsumer();
    // Unfortunately, const_cast is required as our const model is
    // broken.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    NVF_ERROR(p2c.find(const_cast<IterDomain*>(id)) == p2c.end());
  }

  logical_map_.bcast_map_.insert({key, {id}});
}

void ComputeAtLogicalDomainMapBuilder::addToPendingList(
    const DomainKey& producer,
    const DomainKey& consumer) {
  auto it = ensureMapping(pending_map_, producer, {});
  auto& consumer_set = it->second;
  consumer_set.insert(consumer);
}

void ComputeAtLogicalDomainMapBuilder::setMapped(
    const DomainKey& producer,
    const DomainKey& consumer) {
  logical_map_.eq_set_.mapEntries(producer, consumer);
}

void ComputeAtLogicalDomainMapBuilder::setInvalid(
    const DomainKey& key1,
    const DomainKey& key2) {
  invalid_mappings_.emplace_back(key1, key2);
}

bool ComputeAtLogicalDomainMapBuilder::isInvalid(
    const DomainKeySet& domains) const {
  // First, collect all invalid mappings for each of the keys in domains
  DomainKeyMap<DomainKeySet> invalid_key_map;
  for (const auto& key : domains) {
    DomainKeySet invalid_keys;
    for (const auto& invalid_pair : invalid_mappings_) {
      if (logical_map_.canMap(key, invalid_pair.first)) {
        invalid_keys.insert(invalid_pair.second);
      } else if (logical_map_.canMap(key, invalid_pair.second)) {
        invalid_keys.insert(invalid_pair.first);
      }
    }
    invalid_key_map.emplace(key, invalid_keys);
  }

  // Next, check if any pair is invalid to map.
  const auto num_keys = domains.size();
  const std::vector<DomainKey> domains_vec({domains.begin(), domains.end()});
  for (const auto i : arange(num_keys)) {
    const auto& key_i = domains_vec[i];
    // If no invalid keys found for key_i, it can be skipped.
    const auto invalid_key_map_it = invalid_key_map.find(key_i);
    if (invalid_key_map_it == invalid_key_map.end()) {
      continue;
    }

    // Set of keys that are invalid to be mapped with key_i.
    const DomainKeySet& invalid_keys_for_i = invalid_key_map_it->second;

    // If any other key in domains is identified mappable with any of
    // the keys in this set, the mapping with key_i is invalid.
    for (const auto j : arange(i + 1, num_keys)) {
      const auto& key_j = domains_vec[j];
      if (std::any_of(
              invalid_keys_for_i.begin(),
              invalid_keys_for_i.end(),
              [&](const auto& invalid_key_for_i) {
                return logical_map_.canMap(key_j, invalid_key_for_i);
              })) {
        return true;
      }
    }
  }
  return false;
}

void ComputeAtLogicalDomainMapBuilder::setMaybeMapped(
    const TensorDomain* producer_td,
    const IterDomain* producer_id,
    const TensorDomain* consumer_td,
    const IterDomain* consumer_id) {
  const DomainKey producer_key(producer_td, producer_id);
  const DomainKey consumer_key(consumer_td, consumer_id);

  if (producer_id->isBroadcast()) {
    ensureMapping(logical_map_.bcast_map_, producer_key, {});
  }

  if (consumer_id->isBroadcast()) {
    NVF_ERROR(
        producer_id->isBroadcast(),
        "Trying to map a non-broadcast producer with a broadcast consumer. ",
        "Producer: ",
        producer_id->toString(),
        ", consumer: ",
        consumer_id->toString());
    // Get bcast_map_ entry for consumer_id
    const auto consumer_bcast_domains =
        logical_map_.getConcretizedKeys(consumer_td, consumer_id);
    auto& producer_domains =
        logical_map_.getConcretizedDomains(producer_td, producer_id);

    // If consumer id is broadcasted, make sure to propagate its concrete_id(s)
    // to producer
    for (const auto& consumer_bcast_key : consumer_bcast_domains) {
      const auto concrete_id = consumer_bcast_key.concreteId();
      const DomainKey producer_bcast_key(producer_td, producer_id, concrete_id);
      producer_domains.insert(concrete_id);
      addToPendingList(producer_bcast_key, consumer_bcast_key);
    }
  } else {
    auto producer_concrete_key = producer_key;
    if (producer_id->isBroadcast()) {
      const auto concrete_id = consumer_id;
      auto& producer_domains =
          logical_map_.getConcretizedDomains(producer_td, producer_id);
      producer_concrete_key = DomainKey(producer_td, producer_id, concrete_id);
      producer_domains.insert(concrete_id);
    }
    addToPendingList(producer_concrete_key, consumer_key);
  }
}

void ComputeAtLogicalDomainMapBuilder::dispatch(Expr* e) {
  // Avoid visiting expressions multiple times
  if (visited_.find(e) != visited_.end()) {
    return;
  }
  BackwardVisitor::dispatch(e);
  visited_.insert(e);
}

void ComputeAtLogicalDomainMapBuilder::mapPointwiseLikeOp(Expr* expr) {
  if (expr->output(0)->getValType() != ValType::TensorView) {
    return;
  }

  // Broadcast is handled separately, so e should never be BroadcastOp.
  NVF_ERROR(!expr->isA<BroadcastOp>());
  NVF_ERROR(!expr->isA<SqueezeOp>());

  NVF_ERROR(!expr->outputs().empty());

  if (expr->outputs().size() > 1) {
    NVF_ERROR(
        expr->isA<WelfordOp>() || expr->isA<GroupedReductionOp>() ||
            expr->isA<GroupedWelfordOp>(),
        "Unknown multi-output Expr type ",
        expr->getOpString(),
        " is found");
  }

  // Record equalities from output to all the inputs
  // ignores non-concretizable broadcasts
  for (auto producer_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
    for (auto consumer_tv :
         ir_utils::filterByType<TensorView>(expr->outputs())) {
      for (const auto& mapping :
           PairwiseLogicalDomainMap(producer_tv, consumer_tv)
               .mapBroadcast(true)
               .mapProducerToConsumer()) {
        setMaybeMapped(
            producer_tv->domain(),
            mapping.first,
            consumer_tv->domain(),
            mapping.second);
      }
    }
  }
}

void ComputeAtLogicalDomainMapBuilder::handle(BroadcastOp* op) {
  const TensorDomain* in_td = op->in()->as<TensorView>()->domain();
  const TensorDomain* out_td = op->out()->as<TensorView>()->domain();
  const auto in_logical = TensorDomain::noReductions(in_td->logical());
  const auto& out_root = out_td->maybeRoot();
  const auto& bcast_dim_flags = op->getBroadcastDimFlags();
  NVF_ERROR(
      out_root.size() == bcast_dim_flags.size(),
      "dim flags: ",
      bcast_dim_flags,
      ", out root: ",
      out_root);
  auto in_it = in_logical.begin();
  auto out_it = out_root.begin();
  while (in_it != in_logical.end() && out_it != out_root.end()) {
    if (bcast_dim_flags.at(std::distance(out_root.begin(), out_it))) {
      // new broadcast dim. No matching dimension in the input
      // tensor.
      logical_map_.new_broadcast_domains_.insert(DomainKey(out_td, *out_it));
      ++out_it;
      continue;
    }
    setMaybeMapped(in_td, *in_it, out_td, *out_it);
    ++in_it;
    ++out_it;
  }
  // At this point, the input domain should have been scanned
  // entirely.
  NVF_ERROR(
      in_it == in_logical.end(),
      "Unmatched domain detected: ",
      *in_it,
      " of ",
      in_td);
  // On the other hand, the output may still have some domains left,
  // and they must be new broadcast domains.
  for (; out_it != out_root.end(); ++out_it) {
    NVF_ERROR(
        bcast_dim_flags.at(std::distance(out_root.begin(), out_it)),
        "Unmatched domain detected: ",
        *out_it,
        " of ",
        out_td);
    logical_map_.new_broadcast_domains_.insert(DomainKey(out_td, *out_it));
  }
}

void ComputeAtLogicalDomainMapBuilder::handle(SqueezeOp* op) {
  const TensorDomain* in_td = op->in()->as<TensorView>()->domain();
  const TensorDomain* out_td = op->out()->as<TensorView>()->domain();
  const auto in_logical = TensorDomain::noReductions(in_td->logical());
  const auto& out_root = out_td->maybeRoot();
  const auto& squeeze_dim_flags = op->getSqueezeDimFlags();
  NVF_ERROR(
      in_logical.size() == squeeze_dim_flags.size(),
      "dim flags: ",
      squeeze_dim_flags,
      ", in root: ",
      in_logical);
  auto in_it = in_logical.begin();
  auto out_it = out_root.begin();
  while (in_it != in_logical.end() && out_it != out_root.end()) {
    if (squeeze_dim_flags.at(std::distance(in_logical.begin(), in_it))) {
      // new broadcast dim. No matching dimension in the input
      // tensor.
      logical_map_.removed_broadcast_domains_.insert(DomainKey(in_td, *in_it));
      ++in_it;
      continue;
    }
    setMaybeMapped(in_td, *in_it, out_td, *out_it);
    ++in_it;
    ++out_it;
  }
  // At this point, the output domain should have been scanned
  // entirely.
  NVF_ERROR(
      out_it == out_root.end(),
      "Unmatched domain detected: ",
      *out_it,
      " of ",
      out_td);
  // On the other hand, the input may still have some domains left,
  // and they must be removed broadcast domains.
  for (; in_it != in_logical.end(); ++in_it) {
    NVF_ERROR(
        squeeze_dim_flags.at(std::distance(in_logical.begin(), in_it)),
        "Unmatched domain detected: ",
        *in_it,
        " of ",
        in_td);
    logical_map_.removed_broadcast_domains_.insert(DomainKey(in_td, *in_it));
  }
}

void ComputeAtLogicalDomainMapBuilder::handle(ViewAsScalar* op) {
  const TensorView* out_tv = op->output(0)->as<TensorView>();
  const TensorDomain* out_td = out_tv->domain();
  const auto& out_root = out_td->maybeRoot();

  const TensorView* in_tv = op->input(0)->as<TensorView>();
  const TensorDomain* in_td = in_tv->domain();

  std::vector<IterDomain*> in_logical =
      TensorDomain::noReductions(in_tv->getLogicalDomain());
  NVF_ERROR(
      in_logical.size() + 1 == out_root.size(),
      "\nExpression: ",
      op,
      "\nInput logical domain: ",
      in_logical,
      "\nOutput root domain: ",
      out_root);
  auto in_it = in_logical.begin();
  auto out_it = out_root.begin();
  while (in_it != in_logical.end() && out_it != out_root.end()) {
    setMaybeMapped(in_td, *in_it, out_td, *out_it);
    ++in_it;
    ++out_it;
  }
  NVF_ERROR(
      (*out_it)->isVectorComponent(),
      "The last dim of ViewDtypeOp's output must be a ViewAsScalar");
}

void ComputeAtLogicalDomainMapBuilder::mapAllPendingMappings(
    const DomainKey& key) {
  auto it = pending_map_.find(key);
  if (it == pending_map_.end()) {
    return;
  }
  const auto& pending_set = it->second;
  // All entries in key_set must be equivalent with each other.
  NVF_ERROR(!pending_set.empty());
  bool consistent = safeToMap(pending_set);
  for (const auto pending_key : pending_set) {
    if (consistent) {
      setMapped(key, pending_key);
    } else {
      setInvalid(key, pending_key);
    }
  }
  // This entry should never be used again, so remove it.
  pending_map_.erase(it);
}

void ComputeAtLogicalDomainMapBuilder::mapAllPendingMappings(
    const TensorDomain* td,
    IterDomain* id) {
  if (id->isBroadcast()) {
    for (const auto& key : logical_map_.getConcretizedKeys(td, id)) {
      mapAllPendingMappings(key);
    }
  } else {
    mapAllPendingMappings(DomainKey(td, id));
  }
}

void ComputeAtLogicalDomainMapBuilder::handle(RNGOp* rop) {
  handle(rop->output(0)->as<TensorView>());
}

void ComputeAtLogicalDomainMapBuilder::handle(TensorView* tv) {
  const TensorDomain* td = tv->domain();
  const auto logical = TensorDomain::noReductions(td->logical());
  for (auto id : logical) {
    if (id->isBroadcast()) {
      initializeBcastMap(tv, id);
    }
    mapAllPendingMappings(td, id);
  }

  // When tv has an logical domain, propagate the domain mappings from
  // each of the logical axes to the dependent root axes.
  if (td->hasViewLikeRFactor()) {
    std::unordered_set<Val*> root_set({td->root().begin(), td->root().end()});
    for (auto logical_id : logical) {
      if (!logical_id->isRFactorProduct()) {
        continue;
      }
      auto dep = DependencyCheck::getAllValsBetween(root_set, {logical_id});
      for (auto id : ir_utils::filterByType<IterDomain>(dep)) {
        if (root_set.find(id) == root_set.end() || logical_id == id) {
          continue;
        }
        // Usually, the itertypes between IterDomain expression inputs and
        // outputs will match. However, it is possible for a Resize operation to
        // take an Iteration input and reduce it to size 1, after which it
        // becomes Broadcast. This check avoids mapping an Iteration and
        // Broadcast domain in such a case.
        if (id->getIterType() == logical_id->getIterType()) {
          setMaybeMapped(td, id, td, logical_id);
        }
      }
    }
    // Once mappings for logical axes are propagated to root axes,
    // aggregates them at each root axis
    for (auto id : tv->getMaybeRootDomain()) {
      if (id->isBroadcast()) {
        // There can be broadcast domains that appear at root domains but
        // are removed at logical domains as they are merged into
        // non-reduction domains. Initialize the map for those broadcast
        // domains.
        initializeBcastMap(tv, id);
      }
      mapAllPendingMappings(td, id);
    }
  }
}

// Checks whether all consumers of a producer can be joined without
// introducing unsupported mappings, i.e., requiring recomputations.
bool ComputeAtLogicalDomainMapBuilder::safeToMap(const DomainKeySet& domains) {
  if (domains.size() <= 1) {
    return true;
  }

  // Can't map if reduction output domains would be mapped
  if (incompatible_domains_.isReductionOutputMapped(domains, logical_map_) &&
      !map_through_reduction_) {
    return false;
  }
  // Make sure mapping these domains won't cause any invalid mapping
  if (isInvalid(domains)) {
    return false;
  }
  return true;
}

namespace {
class ExactLogicalDomainMapBuilder : private IterVisitor {
 public:
  ExactLogicalDomainMapBuilder(
      Fusion* fusion,
      DisjointSets<const IterDomain*>& eq_sets)
      : eq_sets_(eq_sets) {
    traverseTo(fusion->outputs());
  }

 private:
  using IterVisitor::handle;

  void dispatch(Expr* expr) final {
    for (auto producer : ir_utils::filterByType<TensorView>(expr->inputs())) {
      for (auto consumer :
           ir_utils::filterByType<TensorView>(expr->outputs())) {
        PairwiseLogicalDomainMap pwise_map(producer, consumer);
        pwise_map.mapBroadcast(false);
        const auto mappings = pwise_map.mapProducerToConsumer();
        for (const auto& mapping : mappings) {
          eq_sets_.mapEntries(mapping.first, mapping.second);
        }
      }
    }
  }

 private:
  DisjointSets<const IterDomain*>& eq_sets_;
};

} // namespace

ExactLogicalDomainMap::ExactLogicalDomainMap(Fusion* fusion) {
  ExactLogicalDomainMapBuilder builder(fusion, eq_sets_);
}

bool ExactLogicalDomainMap::areMapped(
    const IterDomain* id_a,
    const IterDomain* id_b) const {
  // With expand going into a view operation there can be an instance where an
  // iteration root domain in the consumer resolves the broadcast from the
  // producer, then immediately rfactors it. In this case the consumer root is
  // not mapped exactly to any other domain, so it might no have an entry in
  // eq_sets_. eq_sets_.strictAreMapped would throw in this case so just return
  // false if a mapping doesn't exist.
  if (!eq_sets_.mappingExists(id_a) || !eq_sets_.mappingExists(id_b)) {
    return false;
  }
  return eq_sets_.strictAreMapped(id_a, id_b);
}

std::unordered_map<IterDomain*, IterDomain*> ExactLogicalDomainMap::map(
    const TensorDomain* producer,
    const TensorDomain* consumer,
    const std::unordered_set<IterDomain*>& dims_to_map,
    bool producer_to_consumer) const {
  const auto& producer_logical =
      TensorDomain::noReductions(producer->logical());
  const auto& consumer_root = consumer->maybeRoot();
  const auto& from_ids =
      producer_to_consumer ? producer_logical : consumer_root;
  const auto& to_ids = producer_to_consumer ? consumer_root : producer_logical;

  std::unordered_map<IterDomain*, IterDomain*> id_map;

  for (auto& from_id : from_ids) {
    if (dims_to_map.find(from_id) == dims_to_map.end()) {
      continue;
    }
    for (const auto& to_id : to_ids) {
      if (areMapped(from_id, to_id)) {
        NVF_ERROR(
            id_map.insert({from_id, to_id}).second,
            "Multiple matching ID detected for ",
            from_id);
      }
    }
  }

  return id_map;
}

std::string ExactLogicalDomainMap::toString() const {
  return eq_sets_.toString();
}

const DisjointSets<const IterDomain*>& ExactLogicalDomainMap::getMappedSets()
    const {
  return eq_sets_;
}

} // namespace nvfuser
