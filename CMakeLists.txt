cmake_minimum_required(VERSION 3.13)
project(nexus LANGUAGES CXX)
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# We don't support caffe2/caffe/darknet any more
#option(USE_GPU        "Use GPU"        ON )
#option(USE_TENSORFLOW "Use TensorFlow" ON )
#option(USE_DARKNET    "Use Darknet"    ON )
option(USE_CAFFE2     "Use Caffe2"     ON )
#option(USE_CAFFE      "Use Caffe"      OFF)
set(USE_GPU ON)
set(USE_TENSORFLOW ON)
set(USE_DARKNET OFF)
#set(USE_CAFFE2 OFF)
set(USE_CAFFE OFF)
option(DEBUG_TF_BUILD "Build TensorFlow in Debug model" OFF)
if(USE_CAFFE2 AND USE_CAFFE)
    message(FATAL_ERROR "`USE_CAFFE2` and `USE_CAFFE` cannot be set at the same time.")
endif()
set(CUDA_PATH "/usr/local/cuda" CACHE STRING "Path to the Cuda toolkit")

set(BOOST_ROOT /usr/local/boost)
find_package(Boost 1.69.0 REQUIRED COMPONENTS system filesystem)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED COMPONENTS shared)
find_package(yaml-cpp 0.6.2 REQUIRED)
find_package(OpenCV REQUIRED)
include(ProcessorCount)
ProcessorCount(NPROC)


###### protobuf & gRPC ######
set(GENERATED_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${GENERATED_SRC_DIR}/src/nexus/proto)
protobuf_generate_cpp(PROTO_NNQUERY_CC PROTO_NNQUERY_H
        ${GENERATED_SRC_DIR}/src/nexus/proto
        src/nexus/proto/nnquery.proto)
protobuf_generate_cpp(PROTO_CONTROL_CC PROTO_CONTROL_H
        ${GENERATED_SRC_DIR}/src/nexus/proto
        src/nexus/proto/control.proto)
grpc_generate_cpp(GRPC_CONTROL_CC GRPC_CONTROL_H
        ${GENERATED_SRC_DIR}/src/nexus/proto
        src/nexus/proto/control.proto)
set(GENERATED_PY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python/nexus/proto)
add_custom_command(
    OUTPUT ${GENERATED_PY_DIR}/nnquery_pb2.py
    COMMAND mkdir -p ${GENERATED_PY_DIR}
    COMMAND touch ${GENERATED_PY_DIR}/__init__.py
    COMMAND protobuf::protoc
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/src/nexus/proto
        --python_out=${GENERATED_PY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nexus/proto/nnquery.proto
    DEPENDS
        protobuf::protoc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nexus/proto/nnquery.proto)
add_custom_target(python ALL DEPENDS ${GENERATED_PY_DIR}/nnquery_pb2.py)



###### common source files, headers, and linking options ######
add_library(common OBJECT
        ${PROTO_NNQUERY_CC}
        ${PROTO_NNQUERY_H}
        ${PROTO_CONTROL_CC}
        ${PROTO_CONTROL_H}
        ${GRPC_CONTROL_CC}
        ${GRPC_CONTROL_H}
        src/nexus/common/backend_pool.cpp
        src/nexus/common/buffer.cpp
        src/nexus/common/connection.cpp
        src/nexus/common/data_type.cpp
        src/nexus/common/device.cpp
        src/nexus/common/image.cpp
        src/nexus/common/message.cpp
        src/nexus/common/metric.cpp
        src/nexus/common/model_db.cpp
        src/nexus/common/server_base.cpp
        src/nexus/common/time_util.cpp
        src/nexus/common/util.cpp)
target_include_directories(common PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GENERATED_SRC_DIR}/src)
target_compile_features(common PUBLIC cxx_std_11)
target_link_libraries(common PUBLIC
        yaml-cpp gflags glog::glog gRPC::grpc++ protobuf::libprotobuf
        ${OpenCV_LIBS} Boost::filesystem Boost::system)
set_target_properties(common PROPERTIES POSITION_INDEPENDENT_CODE ON)



###### scheduler ######
add_executable(scheduler
        src/nexus/scheduler/backend_delegate.cpp
        src/nexus/scheduler/complex_query.cpp
        src/nexus/scheduler/frontend_delegate.cpp
        src/nexus/scheduler/sch_info.cpp
        src/nexus/scheduler/scheduler.cpp
        src/nexus/scheduler/scheduler_main.cpp)
target_include_directories(scheduler PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GENERATED_SRC_DIR}/src)
target_compile_features(scheduler PRIVATE cxx_std_11)
target_link_libraries(scheduler PRIVATE common)



###### frontend libnexus.so ######
add_library(nexus SHARED
        src/nexus/app/app_base.cpp
        src/nexus/app/frontend.cpp
        src/nexus/app/model_handler.cpp
        src/nexus/app/request_context.cpp
        src/nexus/app/rpc_service.cpp
        src/nexus/app/worker.cpp)
target_include_directories(nexus PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GENERATED_SRC_DIR}/src)
target_compile_features(nexus PRIVATE cxx_std_11)
target_link_libraries(nexus PUBLIC common)



###### backend ######
add_library(backend_obj
        src/nexus/backend/backend_server.cpp
        src/nexus/backend/backup_client.cpp
        src/nexus/backend/batch_task.cpp
        src/nexus/backend/gpu_executor.cpp
        src/nexus/backend/model_exec.cpp
        src/nexus/backend/model_ins.cpp
        src/nexus/backend/rpc_service.cpp
        src/nexus/backend/share_prefix_model.cpp
        src/nexus/backend/slice.cpp
        src/nexus/backend/task.cpp
        src/nexus/backend/utils.cpp
        src/nexus/backend/worker.cpp)
target_compile_features(backend_obj PUBLIC cxx_std_11)
target_link_libraries(backend_obj PUBLIC common)

add_executable(backend
        src/nexus/backend/backend_main.cpp)
target_compile_features(backend PRIVATE cxx_std_11)
target_link_libraries(backend PRIVATE backend_obj)

## backend: CUDA support ##
if(USE_GPU)
    target_compile_definitions(common PUBLIC USE_GPU)
    target_include_directories(common PUBLIC ${CUDA_PATH}/include)
    target_link_libraries(common PUBLIC
            ${CUDA_PATH}/lib64/stubs/libcuda.so
            ${CUDA_PATH}/lib64/libcudart.so
            ${CUDA_PATH}/lib64/libcurand.so)
endif()

## backend: TensorFlow support ##
if(USE_TENSORFLOW)
    include(ExternalProject)
    ExternalProject_Add(tensorflow
        URL               https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz
        URL_MD5           164d98d460f8028f5e33f3651d027c77
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
    )

    target_compile_definitions(backend_obj PUBLIC USE_TENSORFLOW)
    target_sources(backend_obj PUBLIC
            src/nexus/backend/tensorflow_model.cpp
            src/nexus/backend/tensorflow_util.cpp
            src/nexus/backend/tf_share_model.cpp)
    ExternalProject_Get_property(tensorflow SOURCE_DIR)
    set(tensorflow_DIR ${SOURCE_DIR})
    target_include_directories(backend_obj PUBLIC ${tensorflow_DIR}/include)
    target_link_libraries(backend_obj PUBLIC ${tensorflow_DIR}/lib/libtensorflow.so)
    target_link_libraries(backend_obj PUBLIC ${tensorflow_DIR}/lib/libtensorflow_framework.so)
    add_dependencies(backend_obj tensorflow)
endif()

## backend: Caffe2 support ##
if(USE_CAFFE2)
    target_compile_definitions(backend_obj PRIVATE USE_CAFFE2)
    target_sources(backend_obj PUBLIC src/nexus/backend/caffe2_model.cpp)
    set(CAFFE2_BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR}/caffe2)
    set(CAFFE2_SO
            ${CAFFE2_BUILD_ROOT}/install/lib/libcaffe2.so
            ${CAFFE2_BUILD_ROOT}/install/lib/libcaffe2_gpu.so)
    file(MAKE_DIRECTORY ${CAFFE2_BUILD_ROOT})
    add_custom_command(
        OUTPUT ${CAFFE2_SO}
        WORKING_DIRECTORY ${CAFFE2_BUILD_ROOT}
        COMMAND ${CMAKE_COMMAND} -DUSE_NNPACK=OFF -DUSE_NCCL=OFF -DUSE_MPI=OFF -DUSE_LMDB=OFF -DUSE_LEVELDB=OFF
            -DUSE_NUMA=OFF -DUSE_MPI=OFF -DBUILD_PYTHON=OFF -DBUILD_BINARY=OFF -DBUILD_TEST=OFF
            -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH}
            -DCMAKE_INSTALL_PREFIX=${CAFFE2_BUILD_ROOT}/install
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_DEBUG_POSTFIX=""
            ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/caffe2
        COMMAND make -j${NPROC} caffe2_gpu
        COMMAND make -j${NPROC} install
    )
    add_custom_target(caffe2 DEPENDS ${CAFFE2_SO})
    target_include_directories(backend_obj PUBLIC ${CAFFE2_BUILD_ROOT}/install/include)
    target_link_libraries(backend_obj PUBLIC ${CAFFE2_SO})
    add_dependencies(backend_obj caffe2)
endif()

## backend: Caffe support ##
if(USE_CAFFE)
    target_compile_definitions(backend PRIVATE USE_CAFFE)
    target_sources(backend_obj PUBLIC
        src/nexus/backend/caffe_densecap_model.cpp
        src/nexus/backend/caffe_model.cpp)
    set(CAFFE_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/caffe)
    set(CAFFE_BUILD_ROOT ${CAFFE_SRC_ROOT}/build)
    set(CAFFE_SO ${CAFFE_BUILD_ROOT}/lib/libcaffe.so)
    add_custom_command(
        OUTPUT ${CAFFE_SO}
        WORKING_DIRECTORY ${CAFFE_SRC_ROOT}
        COMMAND make -j${NPROC} all)
    add_custom_target(caffe DEPENDS ${CAFFE_SO})
    target_include_directories(backend_obj PUBLIC ${CAFFE_SRC_ROOT}/src)
    target_include_directories(backend_obj PUBLIC ${CAFFE_SRC_ROOT}/include)
    target_link_libraries(backend_obj PUBLIC ${CAFFE_SO})
    add_dependencies(backend_obj caffe)
endif()

## backend: Darknet support ##
if(USE_DARKNET)
    target_compile_definitions(backend PRIVATE USE_DARKNET)
    target_sources(backend_obj PRIVATE src/nexus/backend/darknet_model.cpp)
    set(DARKNET_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/darknet)
    set(DARKNET_SO ${DARKNET_SRC_ROOT}/libdarknet.so)
    add_custom_command(
        OUTPUT ${DARKNET_SO}
        WORKING_DIRECTORY ${DARKNET_SRC_ROOT}
        COMMAND make -j${NPROC} all)
    add_custom_target(darknet DEPENDS ${DARKNET_SO})
    target_include_directories(backend_obj PUBLIC ${DARKNET_SRC_ROOT}/src)
    target_include_directories(backend_obj PUBLIC ${DARKNET_SRC_ROOT}/include)
    target_link_libraries(backend_obj PUBLIC ${DARKNET_SO})
    add_dependencies(backend_obj darknet)
endif()



###### tools/profiler ######
add_executable(profiler tools/profiler/profiler.cpp)
target_compile_features(profiler PRIVATE cxx_std_11)
target_link_libraries(profiler PRIVATE common backend_obj)



###### tools/test_pb ######
add_executable(test_pb tools/profiler/test_pb.cpp)
target_compile_features(test_pb PRIVATE cxx_std_11)
target_link_libraries(test_pb PRIVATE common backend_obj)



###### tools/test_complex_query ######
add_executable(test_complex_query
        src/nexus/scheduler/complex_query.cpp
        src/nexus/scheduler/sch_info.cpp
        tools/test_complex_query.cpp)
target_compile_features(test_complex_query PRIVATE cxx_std_11)
target_link_libraries(test_complex_query PRIVATE common)



###### tools/bench_tfshare ######
add_executable(bench_tfshare tools/bench_tfshare.cpp)
target_compile_features(bench_tfshare PRIVATE cxx_std_11)
target_link_libraries(bench_tfshare PRIVATE common backend_obj)



# FIXME ###### tests ######
# add_executable(runtest
#         tests/cpp/scheduler/backend_delegate_test.cpp
#         tests/cpp/scheduler/scheduler_test.cpp
#         tests/cpp/test_main.cpp)
# target_compile_features(runtest PRIVATE cxx_std_11)
# target_link_libraries(runtest PRIVATE common backend_obj)
