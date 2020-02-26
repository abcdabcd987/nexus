#ifndef NEXUS_COMMON_LOAD_BALANCE_POLICY_H_
#define NEXUS_COMMON_LOAD_BALANCE_POLICY_H_

namespace nexus {

enum class LoadBalancePolicy {
  // Weighted round robin
  WeightedRR = 1,
  // Query 2 backends and pick one with lowest utilization
  Query = 2,
  // Deficit round robin
  DeficitRR = 3,
};

}

#endif