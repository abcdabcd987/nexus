#include <sstream>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "nexus/backend/model_exec.h"
#include "nexus/backend/model_ins.h"
#include "nexus/backend/share_prefix_model.h"
#include "nexus/backend/tf_share_model.h"
#include "nexus/common/model_db.h"

namespace nexus {
namespace backend {

DEFINE_int32(backend_count_interval, 1, "Interval to count number of requests in sec");
DEFINE_int32(backend_avg_interval, 5, "Moving average interval in sec");
DEFINE_int32(backend_batch_policy, 0, "0: Sliding window; 1: Earliest first;");

ModelExecutor::ModelExecutor(int gpu_id, const ModelInstanceConfig& config,
                             BlockPriorityQueue<Task>& task_queue) :
    backup_(config.backup()),
    global_task_queue_(task_queue),
    batch_id_(0),
    open_requests_(0),
    req_rate_(FLAGS_backend_count_interval, FLAGS_backend_avg_interval),
    drop_rate_(FLAGS_backend_count_interval, FLAGS_backend_avg_interval),
    num_workers_(36) {
  // Create ModelInstance
  CreateModelInstance(gpu_id, config, &model_);
#ifdef USE_GPU
  auto gpu_device = DeviceManager::Singleton().GetGPUDevice(gpu_id);
  profile_ = ModelDatabase::Singleton().GetModelProfile(
      gpu_device->device_name(), model_->profile_id());
#endif
  req_counter_ = MetricRegistry::Singleton().CreateIntervalCounter(
      FLAGS_backend_count_interval);
  drop_counter_ = MetricRegistry::Singleton().CreateIntervalCounter(
      FLAGS_backend_count_interval);
  input_array_ = model_->CreateInputGpuArray();
  for (auto const& info : config.backup_backend()) {
    backup_backends_.push_back(info.node_id());
  }

  for (int i = 0; i < num_workers_; ++i) {
    auto worker = std::make_shared<ModelWorker>(
        model_.get(), worker_in_queue_, worker_out_queue_);
    worker->Start();
    workers_.push_back(worker);
  }
}

ModelExecutor::~ModelExecutor() {
  for (auto worker : workers_) {
    worker->Stop();
  }
  MetricRegistry::Singleton().RemoveMetric(req_counter_);
  MetricRegistry::Singleton().RemoveMetric(drop_counter_);
}

double ModelExecutor::GetRequestRate() {
  for (auto nreq : req_counter_->GetHistory()) {
    if (req_rate_.rate() < 0 && nreq == 0) {
      continue;
    }
    req_rate_.AddSample(nreq);
  }
  return req_rate_.rate();
}

double ModelExecutor::GetDropRate() {
  for (auto nreq : drop_counter_->GetHistory()) {
    if (drop_rate_.rate() < 0 && nreq == 0) {
      continue;
    }
    drop_rate_.AddSample(nreq);
  }
  return drop_rate_.rate();
}

bool ModelExecutor::IsSharePrefixModel() const {
  return (dynamic_cast<SharePrefixModel*>(model_.get()) != nullptr);
}

bool ModelExecutor::IsTFShareModel() const {
  return (dynamic_cast<TFShareModel*>(model_.get()) != nullptr);
}

bool ModelExecutor::HasBackup() {
  std::lock_guard<std::mutex> lock(backup_mu_);
  return (backup_backends_.size() > 0);
}

std::vector<uint32_t> ModelExecutor::BackupBackends() {
  std::lock_guard<std::mutex> lock(backup_mu_);
  return backup_backends_;
}

void ModelExecutor::UpdateBackupBackends(const ModelInstanceConfig& config) {
  std::lock_guard<std::mutex> lock(backup_mu_);
  backup_backends_.clear();
  for (auto& info : config.backup_backend()) {
    backup_backends_.push_back(info.node_id());
  }
}

bool ModelExecutor::Preprocess(std::shared_ptr<Task> task, bool force) {
  // int cnt = 1;
  // if (task->query.window_size() > 0) {
  //   cnt = task->query.window_size();
  // }
  // bool limit = !force && HasBackup();
  // if (!IncreaseOpenRequests(cnt, limit)) {
  //   return false;
  // }
  // req_counter_->Increase(cnt);
  // model_->Preprocess(task);
  // if (task->result.status() != CTRL_OK) {
  //   return false;
  // }
  // std::lock_guard<std::mutex> lock(task_mu_);
  // processing_tasks_.emplace(task->task_id, task);
  // for (auto input : task->inputs) {
  //   input_queue_.push(input);
  // }
  return true;
}

void ModelExecutor::Enqueue(std::shared_ptr<Task> task) {
  std::lock_guard<std::mutex> lock(task_mu_);
  processing_tasks_.emplace(task->task_id, task);
  task_queue_.push(task);
}

bool ModelExecutor::AddPreprocessedTask(std::shared_ptr<Task> task,
                                        bool force) {
  // int cnt = task->inputs.size();
  // bool limit = !force && HasBackup();
  // if (!IncreaseOpenRequests(cnt, limit)) {
  //   return false;
  // }
  // req_counter_->Increase(cnt);
  // std::lock_guard<std::mutex> lock(task_mu_);
  // processing_tasks_.emplace(task->task_id, task);
  // for (auto input : task->inputs) {
  //   input_queue_.push(input);
  // }
  return true;
}

void ModelExecutor::Postprocess(std::shared_ptr<Task> task) {
  model_->Postprocess(task);
}

uint64_t ModelExecutor::Execute(uint32_t batch) {
  std::shared_ptr<BatchTask> batch_task;
  int dequeue_cnt;
  if (batch == 0) {
    batch = model_->batch();
  }
  
  auto t1 = std::chrono::high_resolution_clock::now();
  std::tie(batch_task, dequeue_cnt) = GetBatchTask(batch);
  auto t2 = std::chrono::high_resolution_clock::now();
  
  int num_drops = dequeue_cnt - batch_task->batch_size();
  drop_counter_->Increase(num_drops);
  
  if (batch_task->batch_size() == 0) {
    //DecreaseOpenRequests(dequeue_cnt);
    if (num_drops > 0) {
      LOG(INFO) << model_->model_session_id() << " drop " << num_drops << " requests";
    }
    std::lock_guard<std::mutex> lock(time_mu_);
    last_exec_finish_ = t2;
    return std::chrono::duration_cast<std::chrono::microseconds>(
        t2 - t1).count();
  }

  uint64_t batch_id = batch_id_.fetch_add(1, std::memory_order_relaxed);
  batch_task->set_batch_id(batch_id);
  // Each time recompute output sizes because it might change for prefix model
  std::unordered_map<std::string, size_t> output_sizes;
  for (auto iter : model_->OutputShapes()) {
    output_sizes.emplace(iter.first, iter.second.NumElements(1));
  }
  batch_task->CreateOutputArrays(output_sizes,
                                 DeviceManager::Singleton().GetCPUDevice());
  model_->Forward(batch_task);
  auto t3 = std::chrono::high_resolution_clock::now();
  {
    std::lock_guard<std::mutex> lock(time_mu_);
    last_exec_finish_ = t3;
  }
  //DecreaseOpenRequests(dequeue_cnt);
  
  auto memcpy_lat = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1).count();
  auto forward_lat = std::chrono::duration_cast<std::chrono::microseconds>(
      t3 - t2).count();
  LOG(INFO) << model_->model_session_id() << " forwards batch " <<
      batch_task->batch_id() << ", size " << batch_task->batch_size() <<
      ", preprocess lat " << memcpy_lat << " us, forward lat " << forward_lat <<
      " us, drop " << num_drops << " requests";

  auto outputs = batch_task->outputs();
  auto tasks = batch_task->tasks();
  // Add output to corresponding tasks, and remove tasks that get all outputs
  std::lock_guard<std::mutex> lock(task_mu_);
  for (int i = 0; i < outputs.size(); ++i) {
    auto output = outputs[i];
    auto task = tasks[i];
    if (task->AddOutput(output)) {
      RemoveTask(task);
    }
  }
  return memcpy_lat + forward_lat;
}

int ModelExecutor::NumberOfOpenRequests() const {
  return open_requests_.load(std::memory_order_relaxed);
}

TimePoint ModelExecutor::LastExecuteFinishTime() {
  std::lock_guard<std::mutex> lock(time_mu_);
  return last_exec_finish_;
}

bool ModelExecutor::IncreaseOpenRequests(int cnt, bool limit_max_batch) {
  if (!limit_max_batch) {
    open_requests_.fetch_add(cnt, std::memory_order_relaxed);
    return true;
  }
  // opportunistic
  int nreqs = open_requests_.load(std::memory_order_relaxed);
  if (nreqs + cnt > model_->batch()) {
    return false;
  }
  open_requests_.fetch_add(cnt, std::memory_order_relaxed);
  return true;
}

void ModelExecutor::DecreaseOpenRequests(int cnt) {
  int prev = open_requests_.fetch_sub(cnt, std::memory_order_relaxed);
  //CHECK_GE(prev, cnt) << "Negative value in open requests";
}

std::pair<std::shared_ptr<BatchTask>, int> ModelExecutor::GetBatchTaskSlidingWindow(
    uint32_t expect_batch_size) {
  auto batch_task = std::make_shared<BatchTask>(model_->max_batch());
  batch_task->SetInputArray(input_array_);
  if (expect_batch_size > model_->batch()) {
    expect_batch_size = model_->batch();
  }
  if (expect_batch_size == 0) {
    return {batch_task, 0};
  }

  std::lock_guard<std::mutex> lock(task_mu_);
  TimePoint now = Clock::now();
  TimePoint finish;
  CHECK(profile_ != nullptr);
  float latency = profile_->GetPreprocessLatency() * expect_batch_size / num_workers_ +
                  profile_->GetForwardLatency(expect_batch_size) +
                  profile_->GetPostprocessLatency();
  finish = now + std::chrono::microseconds(int(latency));

  int dequeue_cnt = 0;
  int current_batch = 0;
  int num_tasks = 0;
  while (!task_queue_.empty()) {
    std::shared_ptr<Task> task = task_queue_.top();
    int num_inputs = 1;
    if (task->query.window_size() > 0) {
      num_inputs = task->query.window_size();
    }
    if (current_batch + num_inputs > expect_batch_size) break;
    task_queue_.pop();
    ++dequeue_cnt;
    if (task->result.status() != CTRL_OK) {
      task->timer.Record("exec");
      RemoveTask(task);
    } else if (task->deadline() < finish) {
      task->result.set_status(TIMEOUT);
      task->timer.Record("exec");
      RemoveTask(task);
    } else {
      current_batch += num_inputs;
      worker_in_queue_.push(task);
      ++num_tasks;
    }
  }

  if (num_tasks == 0) {
    return {batch_task, dequeue_cnt};
  }

  // wait preprocessing finishes
  std::unordered_map<std::string, std::vector<std::shared_ptr<Input>>> model_inputs;
  int finished = 0;
  while (finished < num_tasks) {
    auto task = worker_out_queue_.pop();
    task->timer.Record("exec");
    auto& model_sess_id = task->query.model_session_id();
    if (model_inputs.find(model_sess_id) == model_inputs.end()) {
      model_inputs.emplace(model_sess_id,
                           std::vector<std::shared_ptr<Input>>{});
    }
    for (auto input : task->inputs) {
      model_inputs.at(model_sess_id).push_back(input);
    }
    ++finished;
  }

  for (auto const& iter : model_inputs) {
    for (auto input : iter.second) {
      auto task = processing_tasks_.at(input->task_id);
      batch_task->AppendInput(input, task);
    }
  }
  return {batch_task, dequeue_cnt};
}

void preprocess(ModelInstance* model, std::shared_ptr<Task> task) {
  model->Preprocess(task);
}

std::pair<std::shared_ptr<BatchTask>, int> ModelExecutor::GetBatchTaskEarliest(
    uint32_t expect_batch_size) {
  auto batch_task = std::make_shared<BatchTask>(model_->max_batch());
  batch_task->SetInputArray(input_array_);

  int dequeue_cnt = 0;
  int num_tasks = 0;
  std::unordered_map<std::string, std::vector<std::shared_ptr<Input>>> model_inputs;
  
  { // lock region by task_mu_
    std::lock_guard<std::mutex> lock(task_mu_);
    if (task_queue_.empty()) {
      return {batch_task, 0};
    }

    CHECK(profile_ != nullptr);

    // find the earliest deadline
    TimePoint now = Clock::now();
    TimePoint finish = now;
    //double preprocess_us = 10 * 1e6;
    finish += std::chrono::microseconds(static_cast<int>(profile_->GetPreprocessLatency()));
    finish += std::chrono::microseconds(static_cast<int>(profile_->GetForwardLatency(1)));
    finish += std::chrono::microseconds(static_cast<int>(profile_->GetPostprocessLatency()));
  
    while (!task_queue_.empty()) {
      auto task = task_queue_.top();
      if (task->result.status() != CTRL_OK || task->deadline() < finish) {
        task->timer.Record("exec");
        VLOG(1) << model_->model_session_id() << " drops task " <<
            task->task_id << ", waiting time " <<
            task->timer.GetLatencyMicros("begin", "exec") << " us";
        task->result.set_status(TIMEOUT);
        task_queue_.pop();
        ++dequeue_cnt;
        RemoveTask(task);
      } else {
        finish = task->deadline();
        break;
      }
    }

    if (task_queue_.empty()) {
      return {batch_task, dequeue_cnt};
    }

    int budget = std::chrono::duration_cast<std::chrono::microseconds>(finish - now).count();
    uint32_t batch_size = 0;
    while (!task_queue_.empty()) {
      std::shared_ptr<Task> task = task_queue_.top();
      int num_inputs = 1;
      if (task->query.window_size() > 0) {
        num_inputs = task->query.window_size();
      }
      if (batch_size + num_inputs > model_->batch()) break;
      batch_size += num_inputs;
      // double latency = profile_->GetPreprocessLatency() * batch_size + profile_->GetForwardLatency(batch_size) +
      //   profile_->GetPostprocessLatency();
      double latency = profile_->GetPreprocessLatency() +
                       profile_->GetForwardLatency(batch_size) +
                       profile_->GetPostprocessLatency();
      if (latency > budget) {
        batch_size -= num_inputs;
        break;
      }
      task_queue_.pop();
      ++dequeue_cnt;
      worker_in_queue_.push(task);
      ++num_tasks;
    }
  }   // end of task_mu_


  // Wait for preprocessing finish
  int finished = 0;
  while (finished < num_tasks) {
    auto task = worker_out_queue_.pop();
    task->timer.Record("exec");
    auto& model_sess_id = task->query.model_session_id();
    if (model_inputs.find(model_sess_id) == model_inputs.end()) {
      model_inputs.emplace(model_sess_id,
                           std::vector<std::shared_ptr<Input>>{});
    }
    for (auto input : task->inputs) {
      model_inputs.at(model_sess_id).push_back(input);
    }
    ++finished;
  }

  // for (auto& it : workers) {
  //   it.first.join();
  //   auto task = it.second;
  //   task->timer.Record("exec");
  //   auto& model_sess_id = task->query.model_session_id();
  //   if (model_inputs.find(model_sess_id) == model_inputs.end()) {
  //     model_inputs.emplace(model_sess_id,
  //                          std::vector<std::shared_ptr<Input>>{});
  //   }
  //   for (auto input : task->inputs) {
  //     model_inputs.at(model_sess_id).push_back(input);
  //   }
  // }

  // lock task_mu_
  std::lock_guard<std::mutex> lock(task_mu_);
  std::stringstream ss;
  for (auto const& iter : model_inputs) {
    for (auto input : iter.second) {
      auto task_iter = processing_tasks_.find(input->task_id);
      if (task_iter == processing_tasks_.end()) {
        LOG(FATAL) << model_->model_session_id() << " cannot find task " << input->task_id;
      }
      auto task = task_iter->second;
      batch_task->AppendInput(input, task);
      ss << task->task_id << " ";
    }
  }
  VLOG(1) << model_->model_session_id() << " batch size " <<
      batch_task->batch_size() << ": " << ss.str();
  return {batch_task, dequeue_cnt};

  /*
  budget -= static_cast<int>(profile_->GetPostprocessLatency());
  budget -= static_cast<int>(profile_->GetPreprocessLatency());
  uint32_t batch_size = 1;
  while (true) {
    if (profile_->GetForwardLatency(batch_size
  }
  while (batch_size <= expect_batch_size && profile_->GetForwardLatency(batch_size) < budget)
    ++batch_size;
  --batch_size;

  // gather inputs
  uint32_t current_batch = 0;
  while (current_batch < batch_size && !input_queue_.empty()) {
    auto input = input_queue_.top();
    input_queue_.pop();
    ++dequeue_cnt;

    auto task = processing_tasks_.at(input->task_id);
    model_->Proprocess(task);
    task->timer.Record("exec");
    auto& model_sess_id = task->query.model_session_id();
    if (model_inputs.find(model_sess_id) == model_inputs.end()) {
      model_inputs.emplace(model_sess_id,
                           std::vector<std::shared_ptr<Input> >{});
    }
    model_inputs.at(model_sess_id).push_back(input);
    ++current_batch;
  }

  std::stringstream ss;
  for (auto const& iter : model_inputs) {
    for (auto input : iter.second) {
      auto task = processing_tasks_.at(input->task_id);
      batch_task->AppendInput(input, task);
      ss << task->task_id << " ";
    }
  }
  VLOG(1) << model_->model_session_id() << " batch size " <<
          batch_task->batch_size() << ": " << ss.str();
  return {batch_task, dequeue_cnt};
  */
}

std::pair<std::shared_ptr<BatchTask>, int> ModelExecutor::GetBatchTask(
    uint32_t expect_batch_size) {
  switch (FLAGS_backend_batch_policy) {
    case 0: return GetBatchTaskSlidingWindow(expect_batch_size);
    case 1: return GetBatchTaskEarliest(expect_batch_size);
    default: LOG(FATAL) << "Unknown FLAGS_backend_batch_policy=" << FLAGS_backend_batch_policy;
  }
}

void ModelExecutor::RemoveTask(std::shared_ptr<Task> task) {
  task->stage = kPostprocess;
  global_task_queue_.push(task);
  processing_tasks_.erase(task->task_id);
}

} // namespace backend
} // namespace nexus

