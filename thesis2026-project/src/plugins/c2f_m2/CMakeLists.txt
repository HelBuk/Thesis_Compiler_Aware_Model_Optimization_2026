cmake_minimum_required(VERSION 3.18)
project(c2f_m2_plugin LANGUAGES CXX CUDA)

add_library(c2f_m2_plugin SHARED
    c2f_m2_plugin.cpp
    c2f_m2_plugin_runtime.cu
    c2f_m2_kernels.cu
    c2f_m2_fused.cu
    c2f_m2_winograd.cu
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
    CXX_STANDARD  17
    CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
    # Orin Nano/NX: GA10B = sm87
    CUDA_ARCHITECTURES 87
)

# C++ flags: optimise for Cortex-A78AE (Orin CPU)
target_compile_options(c2f_m2_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=armv8.2-a>
    # CUDA flags: fast-math enables fused multiply-add, rsqrt, etc.
    # --use_fast_math also enables -ftz=true (flush denormals) which is safe
    # for inference workloads.
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)
