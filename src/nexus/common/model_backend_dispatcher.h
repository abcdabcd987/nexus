#ifndef NEXUS_COMMON_MODEL_BACKEND_DISPATCHER_H_
#define NEXUS_COMMON_MODEL_BACKEND_DISPATCHER_H_

#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "nexus/common/backend_pool.h"
#include "nexus/common/load_balance_policy.h"
#include "nexus/common/thread_safety.h"
#include "nexus/proto/control.pb.h"

namespace nexus {

// Dispatch a request of a specific model session to backends. Used by both
// frontend and dispatcher.
class ModelBackendDispatcher {
 public:
  ModelBackendDispatcher(std::string model_session_id, BackendPool* pool,
                         LoadBalancePolicy lb_policy);

  void UpdateRoute(const ModelRouteProto& route);
  std::shared_ptr<BackendSession> GetBackend();
  std::vector<uint32_t> BackendList();

 private:
  std::shared_ptr<BackendSession> GetBackendWeightedRoundRobin();
  std::shared_ptr<BackendSession> GetBackendDeficitRoundRobin();

  const std::string model_session_id_;
  BackendPool* const backend_pool_;
  const LoadBalancePolicy lb_policy_;
  std::mutex route_mu_;

  // Common for all load balancing policies
  double total_throughput_ = 0;
  std::vector<uint32_t> backends_;
  std::unordered_map<uint32_t, double> backend_rates_;

  // Deficit round robin
  double quantum_to_rate_ratio_ = 0;
  size_t current_drr_index_ = 0;
  std::unordered_map<uint32_t, double> backend_quanta_;

  // Weighted round robin
  std::random_device rd_;
  std::mt19937 rand_gen_;
};

}  // namespace nexus

#endif
