#include "nexus/common/backend_dispatcher.h"

#include <glog/logging.h>

namespace nexus {

BackendDispatcher::BackendDispatcher(boost::asio::io_context* io_context,
                                     MessageHandler* message_handler)
    : io_context_(io_context), message_handler_(message_handler) {}

BackendPool& BackendDispatcher::GetBackendPool() {
  return backend_pool_;
}

void BackendDispatcher::UpdateBackendPoolAndModelRoute(
    const ModelRouteProto& route,
    ModelBackendDispatcher* model_backend_dispatcher) {
  const auto& model_session_id = route.model_session_id();
  // Update backend pool first
  {
    std::lock_guard<std::mutex> lock(backend_sessions_mu_);
    auto old_backends = model_backend_dispatcher->BackendList();
    std::unordered_set<uint32_t> new_backends;
    // Add new backends
    for (auto backend : route.backend_rate()) {
      uint32_t backend_id = backend.info().node_id();
      if (backend_sessions_.count(backend_id) == 0) {
        backend_sessions_.emplace(
            backend_id, std::unordered_set<std::string>{model_session_id});
        backend_pool_.AddBackend(std::make_shared<BackendSession>(
            backend.info(), *io_context_, message_handler_));
      } else {
        backend_sessions_.at(backend_id).insert(model_session_id);
      }
      new_backends.insert(backend_id);
    }
    // Remove unused backends
    for (auto backend_id : old_backends) {
      if (new_backends.count(backend_id) == 0) {
        backend_sessions_.at(backend_id).erase(model_session_id);
        if (backend_sessions_.at(backend_id).empty()) {
          LOG(INFO) << "Remove backend " << backend_id;
          backend_sessions_.erase(backend_id);
          backend_pool_.RemoveBackend(backend_id);
        }
      }
    }
  }
  // Update route to backends with throughput in model handler
  model_backend_dispatcher->UpdateRoute(route);
}

}  // namespace nexus
