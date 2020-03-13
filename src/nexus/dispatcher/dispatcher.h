#ifndef NEXUS_DISPATCHER_DISPATCHER_H_
#define NEXUS_DISPATCHER_DISPATCHER_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "nexus/common/connection.h"
#include "nexus/common/server_base.h"
#include "nexus/dispatcher/rpc_service.h"
#include "nexus/proto/control.pb.h"

namespace nexus {
namespace dispatcher {

class ModelRoute {
 public:
  void Update(const ModelRouteProto& route);
  BackendInfo GetBackend();

 private:
  // Basic infomation from the proto
  std::string model_session_id_;
  std::vector<ModelRouteProto::BackendRate> backends_;
  double total_throughput_ = 0;

  // Members for deficit round robin
  std::unordered_map<uint32_t, double> backend_quanta_;
  double min_rate_ = 0;
  size_t current_drr_index_ = 0;
};

class Dispatcher {
 public:
  Dispatcher(std::string rpc_port, std::string sch_addr, int udp_port);

  virtual ~Dispatcher();

  void Run();

  void Stop();

  void UpdateModelRoutes(const ModelRouteUpdates& request, RpcReply* reply);

 private:
  void Register();

  void Unregister();

  void UdpServerDoReceive();
  void UdpServerDoSend(boost::asio::ip::udp::endpoint endpoint,
                       std::string msg);

  boost::asio::io_context io_context_;

  /*! \brief Indicator whether the dispatcher is running */
  std::atomic_bool running_;
  /*! \brief Interval to update stats to scheduler in seconds */
  uint32_t beacon_interval_sec_;
  /*! \brief Frontend node ID */
  uint32_t node_id_;
  /*! \brief RPC service */
  RpcService rpc_service_;
  /*! \brief RPC client connected to scheduler */
  std::unique_ptr<SchedulerCtrl::Stub> sch_stub_;

  /*! \brief Random number generator */
  std::random_device rd_;
  std::mt19937 rand_gen_;

  // Big lock for the following members
  std::mutex mutex_;
  // Maps model session ID to backend list of the model
  std::unordered_map<std::string, ModelRoute> models_;

  // UDP RPC Server
  int udp_port_;
  boost::asio::ip::udp::socket udp_socket_;
  uint8_t buf_[1400];
  boost::asio::ip::udp::endpoint remote_endpoint_;
};

}  // namespace dispatcher
}  // namespace nexus

#endif