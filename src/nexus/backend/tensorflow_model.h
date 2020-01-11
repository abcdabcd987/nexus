#ifndef NEXUS_BACKEND_TENSORFLOW_MODEL_H_
#define NEXUS_BACKEND_TENSORFLOW_MODEL_H_

#include "nexus/backend/model_ins.h"
#include "tensorflow/c/c_api.h"

namespace nexus {
namespace backend {

class TFShareModel;

class TensorflowModel : public ModelInstance {
public:
  TensorflowModel(int gpu_id, const ModelInstanceConfig &config);

  ~TensorflowModel();

  Shape InputShape() final;

  std::unordered_map<std::string, Shape> OutputShapes() final;

  ArrayPtr CreateInputGpuArray() final;

  std::unordered_map<std::string, ArrayPtr> GetOutputGpuArrays() final;

  void Preprocess(std::shared_ptr<Task> task) final;

  void Forward(std::shared_ptr<BatchTask> batch_task) final;

  void Postprocess(std::shared_ptr<Task> task) final;

private:
  TF_Tensor *NewInputTensor();

  void MarshalDetectionResult(const QueryProto &query,
                              std::shared_ptr<Output> output, int im_height,
                              int im_width, QueryResultProto *result);

  TF_Session* session_ = nullptr;
  TF_Graph* graph_ = nullptr;
  int image_height_;
  int image_width_;
  std::string input_layer_;
  Shape input_shape_;
  size_t input_size_;
  DataType input_data_type_;
  std::vector<std::string> output_layers_;
  std::unordered_map<std::string, Shape> output_shapes_;
  std::unordered_map<std::string, size_t> output_sizes_;
  std::vector<float> input_mean_;
  std::vector<float> input_std_;
  std::unordered_map<int, std::string> classnames_;
  std::vector<TF_Tensor*> input_tensors_;
  bool first_input_array_;

  // supports for TFShareModel
  friend class TFShareModel;
  size_t num_suffixes_;
  TF_Tensor* slice_beg_tensor_ = nullptr;
  TF_Tensor* slice_len_tensor_ = nullptr;
  void set_slice_tensor(TF_Tensor* dst,
                        const std::vector<int32_t> &src);
};

} // namespace backend
} // namespace nexus

#endif // NEXUS_BACKEND_TENSORFLOW_MODEL_H_
