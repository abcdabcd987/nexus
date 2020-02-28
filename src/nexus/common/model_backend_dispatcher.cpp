#include "nexus/common/model_backend_dispatcher.h"

#include <glog/logging.h>

namespace nexus {

ModelBackendDispatcher::ModelBackendDispatcher(
    std::string model_session_id, BackendPool* pool,
    LoadBalancePolicy lb_policy)
    : model_session_id_(std::move(model_session_id)),
      backend_pool_(pool),
      lb_policy_(lb_policy),
      rand_gen_(rd_()) {}

void ModelBackendDispatcher::UpdateRoute(const ModelRouteProto& route) {
  std::lock_guard<std::mutex> lock(route_mu_);
  backends_.clear();
  backend_rates_.clear();
  total_throughput_ = 0.;

  double min_rate = std::numeric_limits<double>::max();
  for (auto itr : route.backend_rate()) {
    min_rate = std::min(min_rate, itr.throughput());
  }
  quantum_to_rate_ratio_ = 1. / min_rate;

  for (auto itr : route.backend_rate()) {
    uint32_t backend_id = itr.info().node_id();
    backends_.push_back(backend_id);
    const auto rate = itr.throughput();
    backend_rates_.emplace(backend_id, rate);
    total_throughput_ += rate;
    LOG(INFO) << "- backend " << backend_id << ": " << rate;
    if (backend_quanta_.count(backend_id) == 0) {
      backend_quanta_.emplace(backend_id, rate * quantum_to_rate_ratio_);
    }
  }
  LOG(INFO) << "Total throughput: " << total_throughput_;
  std::sort(backends_.begin(), backends_.end());
  current_drr_index_ %= backends_.size();
  for (auto iter = backend_quanta_.begin(); iter != backend_quanta_.end();) {
    if (backend_rates_.count(iter->first) == 0) {
      iter = backend_quanta_.erase(iter);
    } else {
      ++iter;
    }
  }
}

std::shared_ptr<BackendSession> ModelBackendDispatcher::GetBackend() {
  std::lock_guard<std::mutex> lock(route_mu_);
  switch (lb_policy_) {
    case LoadBalancePolicy::WeightedRR: {
      return GetBackendWeightedRoundRobin();
    }
    case LoadBalancePolicy::DeficitRR: {
      auto backend = GetBackendDeficitRoundRobin();
      if (backend != nullptr) {
        return backend;
      }
      return GetBackendWeightedRoundRobin();
    }
    case LoadBalancePolicy::Query: {
      auto candidate1 = GetBackendWeightedRoundRobin();
      if (candidate1 == nullptr) {
        return nullptr;
      }
      auto candidate2 = GetBackendWeightedRoundRobin();
      if (candidate1 == candidate2) {
        return candidate1;
      }
      if (candidate1->GetUtilization() <= candidate2->GetUtilization()) {
        return candidate1;
      }
      return candidate2;
    }
    default:
      return nullptr;
  }
}

std::shared_ptr<BackendSession>
ModelBackendDispatcher::GetBackendWeightedRoundRobin() {
  std::uniform_real_distribution<float> dis(0, total_throughput_);
  float select = dis(rand_gen_);
  uint i = 0;
  for (; i < backends_.size(); ++i) {
    uint32_t backend_id = backends_[i];
    float rate = backend_rates_.at(backend_id);
    select -= rate;
    if (select < 0) {
      auto backend_sess = backend_pool_->GetBackend(backend_id);
      if (backend_sess != nullptr) {
        return backend_sess;
      }
      break;
    }
  }
  ++i;
  for (uint j = 0; j < backends_.size(); ++j, ++i) {
    auto backend_sess = backend_pool_->GetBackend(backends_[i]);
    if (backend_sess != nullptr) {
      return backend_sess;
    }
  }
  return nullptr;
}

std::shared_ptr<BackendSession>
ModelBackendDispatcher::GetBackendDeficitRoundRobin() {
  for (size_t i = 0; i < 2 * backends_.size(); ++i) {
    size_t idx = (current_drr_index_ + i) % backends_.size();
    uint32_t backend_id = backends_[idx];
    if (backend_quanta_.at(backend_id) >= 1. - 1e-6) {
      auto backend = backend_pool_->GetBackend(backend_id);
      if (backend != nullptr) {
        backend_quanta_[backend_id] -= 1.;
        return backend;
      } else {
        current_drr_index_ = (current_drr_index_ + 1) % backends_.size();
      }
    } else {
      auto rate = backend_rates_[backend_id];
      backend_quanta_[backend_id] += rate * quantum_to_rate_ratio_;
      current_drr_index_ = (current_drr_index_ + 1) % backends_.size();
    }
  }

  return nullptr;
}

std::vector<uint32_t> ModelBackendDispatcher::BackendList() {
  std::vector<uint32_t> ret;
  std::lock_guard<std::mutex> lock(route_mu_);
  for (auto iter : backend_rates_) {
    ret.push_back(iter.first);
  }
  return ret;
}

}  // namespace nexus