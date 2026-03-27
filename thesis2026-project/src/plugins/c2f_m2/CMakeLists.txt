cmake_minimum_required(VERSION 3.18)
project(c2f_m2_plugin LANGUAGES CXX CUDA)

add_library(c2f_m2_plugin SHARED
    c2f_m2_plugin.cpp
    c2f_m2_plugin_runtime.cu
    c2f_m2_kernels.cu
)

target_include_directories(c2f_m2_plugin PRIVATE
    /usr/local/cuda/include
    /usr/include/aarch64-linux-gnu
)

target_link_libraries(c2f_m2_plugin PRIVATE
    nvinfer
    cudart
    cudnn
)

set_target_properties(c2f_m2_plugin PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)