#include "c2f_m2_plugin.hpp"

#include <cuda_runtime.h>

using namespace nvinfer1;

#define CUDA_CHECK(call)                  \
    do {                                  \
        cudaError_t err = (call);         \
        if (err != cudaSuccess) return 1; \
    } while (0)

#define CUDA_CHECK_LAST()                 \
    do {                                  \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) return 1; \
    } while (0)

extern "C" __global__ void silu_kernel(float* x, int n);
extern "C" __global__ void add_inplace_kernel(float* x, const float* y, int n);
extern "C" __global__ void concat3_ch_kernel(
    const float* a, const float* b, const float* c,
    float* out,
    int N, int C, int H, int W
);

int YoloC2fM2Plugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {

    if (inputDesc[0].dims.nbDims != 4) return 1;
    if (outputDesc[0].dims.nbDims != 4) return 1;

    int N = inputDesc[0].dims.d[0];
    int C = inputDesc[0].dims.d[1];
    int H = inputDesc[0].dims.d[2];
    int W = inputDesc[0].dims.d[3];

    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return 1;
    if ((C % 2) != 0) return 1;

    const float* x = static_cast<const float*>(inputs[0]);
    float* out = static_cast<float*>(outputs[0]);

    int halfC = C / 2;
    int HW = H * W;
    int total = N * C * HW;
    int halfTotal = N * halfC * HW;
    int concatTotal = N * (3 * halfC) * HW;

    float* ws = static_cast<float*>(workspace);

    float* cv1_out = ws;
    float* m0_out = cv1_out + total;
    float* concat_out = m0_out + halfTotal;

    // views into cv1_out, no copies for split
    float* x1 = cv1_out;
    float* x2 = cv1_out + halfTotal;

    int threads = 256;
    int blocks_total = (total + threads - 1) / threads;
    int blocks_half = (halfTotal + threads - 1) / threads;
    int blocks_concat = (concatTotal + threads - 1) / threads;

    // fake cv1
    CUDA_CHECK(cudaMemcpyAsync(
        cv1_out, x, static_cast<size_t>(total) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    silu_kernel<<<blocks_total, threads, 0, stream>>>(cv1_out, total);
    CUDA_CHECK_LAST();

    // fake m0
    CUDA_CHECK(cudaMemcpyAsync(
        m0_out, x2, static_cast<size_t>(halfTotal) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    silu_kernel<<<blocks_half, threads, 0, stream>>>(m0_out, halfTotal);
    CUDA_CHECK_LAST();

    silu_kernel<<<blocks_half, threads, 0, stream>>>(m0_out, halfTotal);
    CUDA_CHECK_LAST();

    add_inplace_kernel<<<blocks_half, threads, 0, stream>>>(m0_out, x2, halfTotal);
    CUDA_CHECK_LAST();

    concat3_ch_kernel<<<blocks_concat, threads, 0, stream>>>(
        x1, x2, m0_out, concat_out, N, halfC, H, W);
    CUDA_CHECK_LAST();

    // fake cv2
    CUDA_CHECK(cudaMemcpyAsync(
        out, concat_out, static_cast<size_t>(total) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    silu_kernel<<<blocks_total, threads, 0, stream>>>(out, total);
    CUDA_CHECK_LAST();

    return 0;
}