#include "nexus/app/model_handler.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <typeinfo>

#include "nexus/app/request_context.h"
#include "nexus/common/model_def.h"

DEFINE_int32(count_interval, 1, "Interval to count number of requests in sec");
DEFINE_int32(load_balance,
             static_cast<int>(nexus::LoadBalancePolicy::DeficitRR),
             "Load balance policy (1: random, 2: choice of two, "
             "3: deficit round robin)");

namespace nexus {
namespace app {

QueryResult::QueryResult(uint64_t qid) : qid_(qid), ready_(false) {}

uint32_t QueryResult::status() const {
  CheckReady();
  return status_;
}

std::string QueryResult::error_message() const {
  CheckReady();
  return error_message_;
}

void QueryResult::ToProto(ReplyProto* reply) const {
  CheckReady();
  reply->set_status(status_);
  if (status_ != CTRL_OK) {
    reply->set_error_message(error_message_);
  } else {
    for (auto record : records_) {
      auto rec_p = reply->add_output();
      record.ToProto(rec_p);
    }
  }
}

const Record& QueryResult::operator[](uint32_t idx) const {
  CheckReady();
  return records_.at(idx);
}

uint32_t QueryResult::num_records() const {
  CheckReady();
  return records_.size();
}

void QueryResult::CheckReady() const {
  CHECK(ready_) << "Rpc reply for query " << qid_ << " is not ready yet";
}

void QueryResult::SetResult(const QueryResultProto& result) {
  status_ = result.status();
  if (status_ != CTRL_OK) {
    error_message_ = result.error_message();
  } else {
    for (auto record : result.output()) {
      records_.emplace_back(record);
    }
  }
  ready_ = true;
}

void QueryResult::SetError(uint32_t status, const std::string& error_msg) {
  status_ = status;
  error_message_ = error_msg;
  ready_ = true;
}

std::atomic<uint64_t> ModelHandler::global_query_id_(0);

ModelHandler::ModelHandler(const std::string& model_session_id,
                           BackendPool& pool,
                           LoadBalancePolicy lb_policy)
    : model_session_id_(model_session_id),
      dispatcher_(model_session_id, &pool, lb_policy) {
  ParseModelSession(model_session_id, &model_session_);
  counter_ =
      MetricRegistry::Singleton().CreateIntervalCounter(FLAGS_count_interval);
  LOG(INFO) << model_session_id_
            << " load balance policy: " << static_cast<int>(lb_policy);
}

ModelHandler::~ModelHandler() {
  MetricRegistry::Singleton().RemoveMetric(counter_);
}

std::shared_ptr<QueryResult> ModelHandler::Execute(
    std::shared_ptr<RequestContext> ctx, const ValueProto& input,
    std::vector<std::string> output_fields, uint32_t topk,
    std::vector<RectProto> windows) {
  uint64_t qid = global_query_id_.fetch_add(1, std::memory_order_relaxed);
  counter_->Increase(1);
  auto reply = std::make_shared<QueryResult>(qid);
  auto backend = dispatcher_.GetBackend();
  if (backend == nullptr) {
    ctx->HandleError(SERVICE_UNAVAILABLE, "Service unavailable");
    return reply;
  }
  QueryProto query;
  query.set_query_id(qid);
  query.set_model_session_id(model_session_id_);
  query.mutable_input()->CopyFrom(input);
  for (auto field : output_fields) {
    query.add_output_field(field);
  }
  if (topk > 0) {
    query.set_topk(topk);
  }
  for (auto rect : windows) {
    query.add_window()->CopyFrom(rect);
  }
  if (ctx->slack_ms() > 0) {
    query.set_slack_ms(int(floor(ctx->slack_ms())));
  }
  ctx->RecordQuerySend(qid);
  {
    std::lock_guard<std::mutex> lock(query_ctx_mu_);
    query_ctx_.emplace(qid, ctx);
  }
  auto msg = std::make_shared<Message>(kBackendRequest, query.ByteSizeLong());
  msg->EncodeBody(query);
  backend->Write(std::move(msg));
  return reply;
}

void ModelHandler::HandleReply(const QueryResultProto& result) {
  std::lock_guard<std::mutex> lock(query_ctx_mu_);
  uint64_t qid = result.query_id();
  auto iter = query_ctx_.find(qid);
  if (iter == query_ctx_.end()) {
    // FIXME why this happens? lower from FATAL to ERROR temporarily
    LOG(ERROR) << model_session_id_ << " cannot find query context for query "
               << qid;
    return;
  }
  auto ctx = iter->second;
  ctx->HandleQueryResult(result);
  query_ctx_.erase(qid);
}

void ModelHandler::UpdateRoute(const ModelRouteProto& route) {
  return dispatcher_.UpdateRoute(route);
}

std::vector<uint32_t> ModelHandler::BackendList() {
  return dispatcher_.BackendList();
}

}  // namespace app
}  // namespace nexus
