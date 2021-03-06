#include "nexus/dispatcher/dispatcher.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <glog/logging.h>
#include <pthread.h>
#include <sys/socket.h>

#include <algorithm>
#include <boost/asio.hpp>
#include <chrono>
#include <sstream>

#include "nexus/common/config.h"
#include "nexus/common/model_def.h"

using boost::asio::ip::udp;

namespace {
void PinCpu(pthread_t thread, int cpu) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);
  int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  LOG_IF(FATAL, rc != 0) << "Error calling pthread_setaffinity_np: " << rc;
}
}  // namespace

namespace nexus {
namespace dispatcher {

UdpRpcServer::UdpRpcServer(int udp_rpc_port, Dispatcher* dispatcher, int rx_cpu,
                           int worker_cpu)
    : udp_rpc_port_(udp_rpc_port),
      rx_cpu_(rx_cpu),
      worker_cpu_(worker_cpu),
      dispatcher_(dispatcher),
      rx_socket_(io_context_),
      tx_socket_(io_context_) {}

UdpRpcServer::~UdpRpcServer() {
  if (running_) {
    LOG(WARNING) << "Calling Stop() in ~UdpRpcServer()";
    Stop();
  }
}

void UdpRpcServer::Run() {
  rx_socket_.open(udp::v4());
#ifdef SO_REUSEPORT
  typedef boost::asio::detail::socket_option::boolean<SOL_SOCKET, SO_REUSEPORT>
      reuse_port;
  rx_socket_.set_option(reuse_port(true));
#endif
  rx_socket_.bind(udp::endpoint(udp::v4(), udp_rpc_port_));
  tx_socket_.open(udp::v4());
  tx_socket_.bind(udp::endpoint(udp::v4(), 0));

  running_ = true;
  worker_thread_ = std::thread(&UdpRpcServer::WorkerThread, this);
  incoming_request_.reset(new RequestContext);
  AsyncReceive();

  // Pin cpu
  std::stringstream ss;
  ss << "UDP RPC server is listening on " << rx_socket_.local_endpoint();
  if (rx_cpu_ >= 0) {
    PinCpu(pthread_self(), rx_cpu_);
    ss << " (pinned on CPU " << rx_cpu_ << ")";
  }
  ss << " and sending from " << tx_socket_.local_endpoint();
  if (worker_cpu_ >= 0) {
    PinCpu(worker_thread_.native_handle(), worker_cpu_);
    ss << " (pinned on CPU " << worker_cpu_ << ")";
  }
  LOG(INFO) << ss.str();

  // Block until done
  io_context_.run();
}

void UdpRpcServer::Stop() {
  running_ = false;
  io_context_.stop();
  rx_socket_.cancel();
  tx_socket_.cancel();
  worker_thread_.join();
}

void UdpRpcServer::AsyncReceive() {
  rx_socket_.async_receive_from(
      boost::asio::buffer(incoming_request_->buf), incoming_request_->endpoint,
      [this](boost::system::error_code ec, size_t len) {
        if (ec == boost::asio::error::operation_aborted) {
          return;
        }
        if (ec || !len) {
          AsyncReceive();
          return;
        }
        incoming_request_->len = len;
        {
          std::unique_lock<std::mutex> lock(queue_mutex_);
          queue_.emplace_back(std::move(incoming_request_));
          queue_cv_.notify_one();
        }
        incoming_request_.reset(new RequestContext);
        AsyncReceive();
      });
}

void UdpRpcServer::WorkerThread() {
  std::deque<std::unique_ptr<RequestContext>> q;
  while (running_) {
    // Move requests from the global queue to the local queue
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (queue_.empty()) {
        // Wait on the CV only when the queue is empty.
        // Hopefully this could reduce the times of context switching.
        queue_cv_.wait(lock, [this] { return !queue_.empty(); });
      }
      while (!queue_.empty()) {
        auto request = std::move(queue_.front());
        queue_.pop_front();
        q.emplace_back(std::move(request));
      }
    }

    // Handle requests
    while (!q.empty()) {
      HandleRequest(std::move(q.front()));
      q.pop_front();
    }
  }
}

namespace {

int ns(const std::chrono::time_point<std::chrono::high_resolution_clock>& x,
       const std::chrono::time_point<std::chrono::high_resolution_clock>& y) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(y - x).count();
}

}  // namespace

void UdpRpcServer::HandleRequest(std::unique_ptr<RequestContext> ctx) {
  DispatchRequest request;
  // Validate request
  bool ok = request.ParseFromString(
      std::string(ctx->buf.data(), ctx->buf.data() + ctx->len));
  if (!ok) {
    LOG_EVERY_N(ERROR, 128)
        << "Bad request. Failed to ParseFromString. Total length = "
        << ctx->len;
    return;
  }
  auto client_endpoint = boost::asio::ip::udp::endpoint(ctx->endpoint.address(),
                                                        request.udp_rpc_port());

  // Handle request
  DispatchReply reply;
  *reply.mutable_model_session() = request.model_session();
  reply.set_query_id(request.query_id());
  std::string model_sess_id = ModelSessionToString(request.model_session());
  dispatcher_->GetBackend(model_sess_id, &reply);

  // Send reply. I think using blocking APIs should be okay here?
  auto msg = reply.SerializeAsString();
  if (msg.empty()) {
    LOG(ERROR) << "Failed to reply.SerializeAsString()";
    return;
  }

  auto len = tx_socket_.send_to(boost::asio::buffer(msg), client_endpoint);
  if (len != msg.size()) {
    LOG(WARNING) << "UDP RPC server reply sent " << len << " bytes, expecting "
                 << msg.size() << " bytes";
  }
}

Dispatcher::Dispatcher(std::string rpc_port, std::string sch_addr, int udp_port,
                       int num_udp_threads, std::vector<int> pin_cpus)
    : udp_port_(udp_port),
      num_udp_threads_(num_udp_threads),
      pin_cpus_(std::move(pin_cpus))
#ifndef NEXUS_DISPATCHER_DEBUG_NO_SCHEDULER
      ,
      rpc_service_(this, rpc_port, 1)
#endif
{
#ifndef SO_REUSEPORT
  CHECK_EQ(num_udp_threads, 1) << "SO_REUSEPORT is not supported. UDP RPC "
                                  "server must be run in single threaded mode.";
#endif
  if (!pin_cpus_.empty()) {
    CHECK_EQ(num_udp_threads_ * 2, pin_cpus_.size())
        << "UDP RPC thread affinity settings should contain exactly twice the "
           "number of thread.";
  }

  // Init scheduler client
  if (sch_addr.find(':') == std::string::npos) {
    // Add default scheduler port if no port specified
    sch_addr += ":" + std::to_string(SCHEDULER_DEFAULT_PORT);
  }
#ifndef NEXUS_DISPATCHER_DEBUG_NO_SCHEDULER
  auto channel =
      grpc::CreateChannel(sch_addr, grpc::InsecureChannelCredentials());
  sch_stub_ = SchedulerCtrl::NewStub(channel);
#endif
}

Dispatcher::~Dispatcher() {
  if (running_) {
    Stop();
  }
}

void Dispatcher::Run() {
  running_ = true;

#ifndef NEXUS_DISPATCHER_DEBUG_NO_SCHEDULER
  // Start RPC service
  rpc_service_.Start();
  // Init Node ID and register frontend to scheduler
  Register();
#endif

  // Run UDP RPC server
  for (int i = 0; i < num_udp_threads_; ++i) {
    int cpu1 = pin_cpus_.empty() ? -1 : pin_cpus_.at(i * 2);
    int cpu2 = pin_cpus_.empty() ? -1 : pin_cpus_.at(i * 2 + 1);
    udp_rpc_servers_.emplace_back(
        new UdpRpcServer(udp_port_, this, cpu1, cpu2));
    workers_.emplace_back(&UdpRpcServer::Run, udp_rpc_servers_.back().get());
  }

  // Nothing to do here
  for (;;) {
    std::this_thread::sleep_for(std::chrono::hours(24));
  }
}

void Dispatcher::Stop() {
  LOG(INFO) << "Shutting down the dispatcher.";
  running_ = false;

#ifndef NEXUS_DISPATCHER_DEBUG_NO_SCHEDULER
  // Unregister frontend
  Unregister();
  // Stop RPC service
  rpc_service_.Stop();
#endif

  // Stop UDP RPC server
  for (auto& server : udp_rpc_servers_) {
    server->Stop();
  }
  for (auto& thread : workers_) {
    thread.join();
  }
}

void Dispatcher::GetBackend(const std::string& model_sess_id,
                            DispatchReply* reply) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = models_.find(model_sess_id);
  if (iter == models_.end()) {
    reply->set_status(CtrlStatus::MODEL_NOT_FOUND);
  } else {
    *reply->mutable_backend() = iter->second.GetBackend();
    reply->set_status(CtrlStatus::CTRL_OK);
  }
}

#ifndef NEXUS_DISPATCHER_DEBUG_NO_SCHEDULER
void Dispatcher::Register() {
  // Init node id
  std::random_device rd;
  std::mt19937 rand_gen(rd());
  std::uniform_int_distribution<uint32_t> dis(
      1, std::numeric_limits<uint32_t>::max());
  node_id_ = dis(rand_gen);

  // Prepare request
  RegisterRequest request;
  request.set_node_type(NodeType::DISPATCHER_NODE);
  request.set_node_id(node_id_);
  request.set_rpc_port(rpc_service_.port());

  while (true) {
    grpc::ClientContext context;
    RegisterReply reply;
    grpc::Status status = sch_stub_->Register(&context, request, &reply);
    if (!status.ok()) {
      LOG(FATAL) << "Failed to connect to scheduler: " << status.error_message()
                 << "(" << status.error_code() << ")";
    }
    CtrlStatus ret = reply.status();
    if (ret == CTRL_OK) {
      beacon_interval_sec_ = reply.beacon_interval_sec();
      return;
    }
    if (ret != CTRL_FRONTEND_NODE_ID_CONFLICT) {
      LOG(FATAL) << "Failed to register frontend to scheduler: "
                 << CtrlStatus_Name(ret);
    }
    // Frontend ID conflict, need to generate a new one
    node_id_ = dis(rand_gen);
    request.set_node_id(node_id_);
  }
}

void Dispatcher::Unregister() {
  UnregisterRequest request;
  request.set_node_type(NodeType::DISPATCHER_NODE);
  request.set_node_id(node_id_);

  grpc::ClientContext context;
  RpcReply reply;
  grpc::Status status = sch_stub_->Unregister(&context, request, &reply);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to connect to scheduler: " << status.error_message()
               << "(" << status.error_code() << ")";
    return;
  }
  CtrlStatus ret = reply.status();
  if (ret != CTRL_OK) {
    LOG(ERROR) << "Failed to unregister frontend: " << CtrlStatus_Name(ret);
  }
}
#endif

void Dispatcher::UpdateModelRoutes(const ModelRouteUpdates& request,
                                   RpcReply* reply) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& model_route : request.model_route()) {
    auto iter = models_.find(model_route.model_session_id());
    if (iter == models_.end()) {
      auto res = models_.emplace(model_route.model_session_id(), ModelRoute());
      iter = res.first;
    }
    iter->second.Update(model_route);
  }
  reply->set_status(CTRL_OK);
}

void ModelRoute::Update(const ModelRouteProto& route) {
  LOG(INFO) << "Update model route for " << route.model_session_id();

  // Save the current DRR backend
  const auto current_drr_backend_id =
      backends_.empty() ? 0 : backends_[current_drr_index_].info().node_id();

  // Update from the proto
  model_session_id_.assign(route.model_session_id());
  backends_.assign(route.backend_rate().begin(), route.backend_rate().end());
  total_throughput_ = 0.;

  // Calculate quantum:rate ratio
  min_rate_ = std::numeric_limits<double>::max();
  for (const auto& backend : backends_) {
    min_rate_ = std::min(min_rate_, backend.throughput());
  }

  // Give quantum to new backends
  std::unordered_map<uint32_t, size_t> backend_idx;
  for (size_t i = 0; i < backends_.size(); ++i) {
    const auto& backend = backends_[i];
    const auto backend_id = backend.info().node_id();
    const auto rate = backend.throughput();
    total_throughput_ += rate;
    LOG(INFO) << "  backend " << backend_id << ": " << rate << " rps";
    backend_quanta_.emplace(backend_id, rate);
    backend_idx.emplace(backend_id, i);
  }
  LOG(INFO) << "  total throughput: " << total_throughput_ << " rps";

  // Remove quantum of old backends
  for (auto iter = backend_quanta_.begin(); iter != backend_quanta_.end();) {
    if (backend_idx.count(iter->first) == 0) {
      iter = backend_quanta_.erase(iter);
    } else {
      ++iter;
    }
  }

  // Recover the current DRR backend
  auto backend_idx_iter = backend_idx.find(current_drr_backend_id);
  if (backend_idx_iter != backend_idx.end()) {
    current_drr_index_ = backend_idx_iter->second;
  } else {
    if (backends_.empty()) {
      current_drr_index_ = 0;
    } else {
      current_drr_index_ %= backends_.size();
    }
  }
}

BackendInfo ModelRoute::GetBackend() {
  for (size_t i = 0;; ++i) {
    const auto& backend = backends_[current_drr_index_];
    const uint32_t backend_id = backend.info().node_id();
    if (backend_quanta_.at(backend_id) >= min_rate_) {
      backend_quanta_[backend_id] -= min_rate_;
      return backend.info();
    } else {
      const auto rate = backend.throughput();
      backend_quanta_[backend_id] += rate;
      current_drr_index_ = (current_drr_index_ + 1) % backends_.size();
    }

    CHECK_LE(i, backends_.size()) << "DRR could not decide.";
  }
}

}  // namespace dispatcher
}  // namespace nexus
