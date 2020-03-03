#ifndef NEXUS_COMMON_BACKEND_DISPATCHER_H_
#define NEXUS_COMMON_BACKEND_DISPATCHER_H_

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "nexus/common/backend_pool.h"
#include "nexus/common/model_backend_dispatcher.h"
#include "nexus/proto/control.pb.h"

namespace nexus {

class BackendDispatcher {
 public:
  BackendDispatcher(boost::asio::io_context* io_context,
                    MessageHandler* message_handler);

  void UpdateBackendPoolAndModelRoute(
      const ModelRouteProto& route,
      ModelBackendDispatcher* model_backend_dispatcher);

  BackendPool& GetBackendPool();

 private:
  boost::asio::io_context* const io_context_;
  MessageHandler* const message_handler_;

  // Backend pool
  BackendPool backend_pool_;

  // Map from backend ID to model sessions servered at this backend.
  // Guarded by backend_sessions_mu_
  std::unordered_map<uint32_t, std::unordered_set<std::string>>
      backend_sessions_;

  std::mutex backend_sessions_mu_;
};

}  // namespace nexus

#endif
