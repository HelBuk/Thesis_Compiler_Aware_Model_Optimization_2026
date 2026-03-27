#include "c2f_m2_plugin.hpp"

#include <cuda_runtime.h>
#include <cudnn.h>

using namespace nvinfer1;

#define CUDA_CHECK(call)                         \
    do {                                         \
        cudaError_t err = (call);                \
        if (err != cudaSuccess) return 1;        \
    } while (0)

#define CUDNN_CHECK(call)                        \
    do {                                         \
        cudnnStatus_t s = (call);                \
        if (s != CUDNN_STATUS_SUCCESS) return 1; \
    } while (0)

#define CUDA_CHECK_LAST()                        \
    do {                                         \
        cudaError_t err = cudaGetLastError();    \
        if (err != cudaSuccess) return 1;        \
    } while (0)

extern "C" __global__ void silu_kernel(float* x, int n);
extern "C" __global__ void add_inplace_kernel(float* x, const float* y, int n);
extern "C" __global__ void slice_ch_kernel(
    const float* x,
    float* out,
    int N, int Cin, int Cslice, int H, int W,
    int c_start
);
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

    if (!mCudnn) return 1;
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

    const int halfC = C / 2;
    const int HW = H * W;
    const int total = N * C * HW;
    const int halfTotal = N * halfC * HW;
    const int concatC = 3 * halfC;
    const int concatTotal = N * concatC * HW;

    int threads = 256;
    int blocks_total = (total + threads - 1) / threads;
    int blocks_half = (halfTotal + threads - 1) / threads;
    int blocks_concat = (concatTotal + threads - 1) / threads;

    // Workspace layout:
    // [cv1_out: N*C*H*W]
    // [m0_tmp:  N*(C/2)*H*W]
    // [concat:  N*(3C/2)*H*W]
    // [conv_ws: cudnn temp scratch]
    char* ws = static_cast<char*>(workspace);

    float* cv1_out = reinterpret_cast<float*>(ws);
    ws += static_cast<size_t>(total) * sizeof(float);

    float* m0_out = reinterpret_cast<float*>(ws);
    ws += static_cast<size_t>(halfTotal) * sizeof(float);

    float* concat_out = reinterpret_cast<float*>(ws);
    ws += static_cast<size_t>(concatTotal) * sizeof(float);

    void* conv_ws = reinterpret_cast<void*>(ws);

    // split targets (actual buffers; no memcpy placeholders)
    // x1 and x2 are extracted from cv1_out by slicing
    // to avoid aliasing problems with cuDNN inputs we keep x2 in m0_out initially only when needed
    // but for clarity here:
    float* x1_buf = m0_out; // temporarily reuse m0_out for x1
    float* x2_buf = concat_out; // temporarily reuse beginning of concat buffer for x2 before concat step

    // -------------------------
    // 1) cv1: real 1x1 conv
    // x[N,C,H,W] -> cv1_out[N,C,H,W]
    // -------------------------
    if (!runConv1x1(
            x, cv1_out,
            N, C, C, H, W,
            d_cv1_w, d_cv1_b,
            conv_ws, getMaxConvWorkspaceBytes(N, C, H, W),
            stream)) {
        return 1;
    }

    silu_kernel<<<blocks_total, threads, 0, stream>>>(cv1_out, total);
    CUDA_CHECK_LAST();

    // -------------------------
    // 2) split by channel slice
    // x1 = cv1_out[:, :C/2]
    // x2 = cv1_out[:, C/2:]
    // -------------------------
    slice_ch_kernel<<<blocks_half, threads, 0, stream>>>(
        cv1_out, x1_buf, N, C, halfC, H, W, 0);
    CUDA_CHECK_LAST();

    slice_ch_kernel<<<blocks_half, threads, 0, stream>>>(
        cv1_out, x2_buf, N, C, halfC, H, W, halfC);
    CUDA_CHECK_LAST();

    // -------------------------
    // 3) m0.cv1: real 3x3 conv
    // x2 -> m0_out
    // -------------------------
    if (!runConv3x3(
            x2_buf, m0_out,
            N, halfC, halfC, H, W,
            d_m0_cv1_w, d_m0_cv1_b,
            conv_ws, getMaxConvWorkspaceBytes(N, halfC, H, W),
            stream)) {
        return 1;
    }

    silu_kernel<<<blocks_half, threads, 0, stream>>>(m0_out, halfTotal);
    CUDA_CHECK_LAST();

    // -------------------------
    // 4) m0.cv2: real 3x3 conv
    // m0_out -> m0_out  (use concat_out as temp to avoid in-place cudnn issues)
    // -------------------------
    float* m0_tmp = reinterpret_cast<float*>(concat_out); // safe before concat is written
    if (!runConv3x3(
            m0_out, m0_tmp,
            N, halfC, halfC, H, W,
            d_m0_cv2_w, d_m0_cv2_b,
            conv_ws, getMaxConvWorkspaceBytes(N, halfC, H, W),
            stream)) {
        return 1;
    }

    silu_kernel<<<blocks_half, threads, 0, stream>>>(m0_tmp, halfTotal);
    CUDA_CHECK_LAST();

    // residual add: m0_tmp += x2
    add_inplace_kernel<<<blocks_half, threads, 0, stream>>>(m0_tmp, x2_buf, halfTotal);
    CUDA_CHECK_LAST();

    // -------------------------
    // 5) concat
    // concat(x1, x2, m0_tmp) -> concat_out
    // -------------------------
    // use m0_out as final concat output now
    float* final_concat = m0_out;  // N * (3*halfC) * H * W does NOT fit here, so keep actual concat buffer
    final_concat = reinterpret_cast<float*>(
        static_cast<char*>(workspace) +
        static_cast<size_t>(total) * sizeof(float) +
        static_cast<size_t>(halfTotal) * sizeof(float)
    );

    concat3_ch_kernel<<<blocks_concat, threads, 0, stream>>>(
        x1_buf, x2_buf, m0_tmp,
        final_concat,
        N, halfC, H, W
    );
    CUDA_CHECK_LAST();

    // -------------------------
    // 6) cv2: real 1x1 conv directly to output
    // [N, 3C/2, H, W] -> [N, C, H, W]
    // -------------------------
    if (!runConv1x1(
            final_concat, out,
            N, concatC, C, H, W,
            d_cv2_w, d_cv2_b,
            conv_ws, getMaxConvWorkspaceBytes(N, concatC, H, W),
            stream)) {
        return 1;
    }

    silu_kernel<<<blocks_total, threads, 0, stream>>>(out, total);
    CUDA_CHECK_LAST();

    return 0;
}