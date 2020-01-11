#include "nexus/backend/tensorflow_util.h"


namespace nexus {
namespace backend {

void AppendProtoWithArray(std::vector<uint8_t> *out, int field_and_tag, const uint8_t *buf, size_t len) {
  CHECK_LT(len, 128);  // Base 128 Varints
  out->push_back(field_and_tag);
  out->push_back(len);
  out->insert(out->end(), buf, buf + len);
}

// https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/core/protobuf/config.proto
// https://developers.google.com/protocol-buffers/docs/encoding
std::vector<uint8_t> GetConfigProto(int gpu_id, double gpu_fraction) {
  // Manually craft GPUOptions
  std::vector<uint8_t> gpu_options;
  if (gpu_fraction) {
    // Proto: double per_process_gpu_memory_fraction = 1;
    const uint8_t* hex = reinterpret_cast<uint8_t*>(&gpu_fraction);
    AppendProtoWithArray(&gpu_options, 0x09, hex, sizeof(double));

    // Proto: bool allow_growth = 4;
    gpu_options.push_back(0x20);
    gpu_options.push_back(0x00);  // False
  } else {
    // Proto: bool allow_growth = 4;
    gpu_options.push_back(0x20);
    gpu_options.push_back(0x01);  // True
  }

  // Proto: string allocator_type = 2;
  AppendProtoWithArray(&gpu_options, 0x12, reinterpret_cast<const uint8_t*>("BFC"), 3);

  // Proto: string visible_device_list = 5;
  gpu_options.push_back(0x2a);
  gpu_options.push_back(0x01);  // Length 1
  gpu_options.push_back(gpu_id + '0');  // ToChar

  // Manually craft ConfigProto
  std::vector<uint8_t> config_proto;

  // Proto: GPUOptions gpu_options = 6;
  AppendProtoWithArray(&config_proto, 0x32, config_proto.data(), config_proto.size());

  return config_proto;
}

void FreeTFBuffer(void* data, size_t len) {
	free(data);
}

TF_Buffer* ReadFileToTFBuffer(const char* file) {
	FILE *f = fopen(file, "rb");
  CHECK(file != nullptr);
	fseek(f, 0, SEEK_END);
	size_t fsize = ftell(f);
	rewind(f);

	void* data = malloc(fsize);
	fread(data, fsize, 1, f);
	fclose(f);

	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = FreeTFBuffer;
	return buf;
}

void TFTensorSliceDeallocator(void*, size_t, void*) {
  // no-op
}

TF_Tensor* TFTensorSlice(const TF_Tensor* tensor, int64_t dim0_start, int64_t dim0_limit) {
  uint8_t* data = reinterpret_cast<uint8_t*>(TF_TensorData(tensor));
  size_t size = TF_TensorByteSize(tensor);
  int ndims = TF_NumDims(tensor);
  std::vector<int64_t> dims(ndims);
  for (int i = 0; i < ndims; ++i) {
    dims[i] = TF_Dim(tensor, i);
  }
  CHECK_LE(dim0_start, dim0_limit);
  CHECK_GE(dim0_start, 0);
  CHECK_LT(dim0_limit, dims[0]);
  CHECK_EQ(size % dims[0], 0);
  size_t dim_offset = size / dims[0];
  uint8_t* slice_data = data + dim_offset * dim0_start;
  size_t slice_size = (dim0_limit - dim0_start) * dim_offset;
  return TF_NewTensor(TF_TensorType(tensor), dims.data(), ndims, slice_data, slice_size, TFTensorSliceDeallocator, nullptr);
}

TF_Output GetTFOperationAsOutputFromGraph(TF_Graph* graph, const char* op_name) {
  TF_Output out;
  out.oper = TF_GraphOperationByName(graph, op_name);
  out.index = 0;
  CHECK(out.oper != nullptr) << "Cannot find operation: " << op_name;
  return out;
}

}
}
