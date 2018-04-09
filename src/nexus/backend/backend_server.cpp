#include <glog/logging.h>
#include <unordered_set>

#include "nexus/common/model_db.h"
#include "nexus/backend/backend_server.h"

namespace nexus {
namespace backend {

BackendServer::BackendServer(std::string port, std::string rpc_port,
                             std::string sch_addr, size_t num_workers,
                             int gpu_id, std::string model_db_root) :
    ServerBase(port),
    rpc_service_(this, rpc_port),
    sch_client_(this, sch_addr),
    gpu_id_(gpu_id),
    running_(false),
    rand_gen_(rd_()) {
  // Init model information
  ModelDatabase::Singleton().Init(model_db_root);
  // Start RPC service
  rpc_service_.Start();
  // Init node id
  std::uniform_int_distribution<uint32_t> dis(
      1, std::numeric_limits<uint32_t>::max());
  node_id_ = dis(rand_gen_);
  while (true) {
    CtrlStatus ret = sch_client_.Register();
    if (ret == CTRL_OK) {
      break;
    }
    if (ret != CTRL_BACKEND_NODE_ID_CONFLICT) {
      LOG(FATAL) << "Failed to register backend server to scheduler: " <<
          CtrlStatus_Name(ret);
    }
    node_id_ = dis(rand_gen_);
  }
  sch_client_.Start();
  // Start workers
  for (size_t i = 0; i < num_workers; ++i) {
    std::unique_ptr<Worker> worker(new Worker(i, this, task_queue_));
    worker->Start();
    workers_.push_back(std::move(worker));
  }
  // Start GPU executor
  gpu_executor_.reset(new GpuExecutorMultiBatching(gpu_id, this));
  gpu_executor_->Start();
}

BackendServer::~BackendServer() {
  if (running_) {
    Stop();
  }
}

void BackendServer::Run() {
  running_ = true;
  LOG(INFO) << "Backend server (id: " << node_id_ << ") is listening on " <<
      address();
  io_service_.run();
}

void BackendServer::Stop() {
  running_ = false;
  // Unregister backend server, and stop scheduler client
  sch_client_.Unregister();
  sch_client_.Stop();
  // Stop accept new connections
  ServerBase::Stop();
  // Stop all frontend connections
  for (auto conn: frontend_connections_) {
    conn->Stop();
  }
  frontend_connections_.clear();
  // Stop GPU executor
  gpu_executor_->Stop();
  // Stop workers
  for (auto& worker : workers_) {
    worker->Stop();
  }
  workers_.clear();
  // Stop RPC service
  rpc_service_.Stop();
  LOG(INFO) << "Backend server stopped";
}

void BackendServer::HandleAccept() {
  std::lock_guard<std::mutex> lock(frontend_mutex_);
  auto conn = std::make_shared<Connection>(std::move(socket_), this);
  frontend_connections_.insert(conn);
  conn->Start();
}

void BackendServer::HandleMessage(std::shared_ptr<Connection> conn,
                                  std::shared_ptr<Message> message) {
  if (message->type() != kBackendRequest) {
    LOG(INFO) << "Wrong message type: " << message->type();
    return;
  }
  auto task = std::make_shared<Task>(conn);
  //message->DecodeBody(&task->query);
  //task->query_id = message->query_id();
  task->DecodeQuery(message);
  task_queue_.push(std::move(task));
}

void BackendServer::HandleError(std::shared_ptr<Connection> conn,
                                boost::system::error_code ec) {
  if (ec == boost::asio::error::eof ||
      ec == boost::asio::error::connection_reset) {
    // frontend disconnects
  } else {
    LOG(ERROR) << "Frontend connection error (" << ec << "): " << ec.message();
  }
  std::lock_guard<std::mutex> lock(frontend_mutex_);
  frontend_connections_.erase(conn);
  conn->Stop();
}

void BackendServer::UpdateModelTable(const ModelTable& request,
                                     RpcReply* reply) {
  SpinlockGuard lock(model_table_lock_);
  std::unordered_set<std::string> load_models;
  for (auto desc : request.model_instance_desc()) {
    auto model_sess = desc.model_session();
    std::string session_id = ModelSessionToString(model_sess);
    load_models.insert(session_id);
    auto model_iter = model_session_map_.find(session_id);
    if (model_iter == model_session_map_.end()) {
      // Load new model instance
      LOG(INFO) << "Load model instance for " << session_id <<
          ", max batch: " << desc.batch();
      auto model_id = ModelSessionToModelID(model_sess);
      auto info = ModelDatabase::Singleton().GetModelInfo(model_id);
      auto model = CreateModelInstance(gpu_id_, desc, info, task_queue_);
      model_instances_.push_back(model);
      model_session_map_.emplace(session_id, model);
    } else {
      auto model = model_iter->second;
      if (model->batch() != desc.batch()) {
        // TODO: Update batch size
      }
    }
  }
  for (auto iter : model_session_map_) {
    if (load_models.find(iter.first) == load_models.end()) {
      // TODO: unload model
    }
  }
  reply->set_status(CTRL_OK);
}

ModelInstancePtr BackendServer::GetModelInstance(
    const std::string& model_session_id) {
  SpinlockGuard lock(model_table_lock_);
  auto itr = model_session_map_.find(model_session_id);
  if (itr == model_session_map_.end()) {
    LOG(ERROR) << "Model session is not loaded: " << model_session_id;
    return nullptr;
  }
  return itr->second;
}

std::vector<ModelInstancePtr> BackendServer::GetAllModelInstances() {
  SpinlockGuard lock(model_table_lock_);
  return model_instances_;
}


} // namespace backend
} // namespace nexus
