#include <cstdlib>
#include <vector>

#include <tensorflow/c/c_api.h>
#include <glog/logging.h>

namespace nexus {
namespace backend {

std::vector<uint8_t> GetConfigProto(int gpu_id, double gpu_fraction);

void FreeTFBuffer(void* data, size_t len);

TF_Buffer* ReadFileToTFBuffer(const char* file);

void TFTensorSliceDeallocator(void*, size_t, void*);

TF_Tensor* TFTensorSlice(const TF_Tensor* tensor, int64_t dim0_start, int64_t dim0_limit);

TF_Output GetTFOperationAsOutputFromGraph(TF_Graph* graph, const char* op_name);
    
}
}

