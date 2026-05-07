#include "c2f_m2_plugin.hpp"

#include <cuda_runtime.h>
#include <iostream>

using namespace nvinfer1;

// ---------------------------------------------------------------------------
// Kernel declarations (defined in c2f_m2_kernels.cu)
// ---------------------------------------------------------------------------
extern "C" __global__ void silu_kernel(float* x, int n);
extern "C" __global__ void add_inplace_kernel(float* x, const float* y, int n);
extern "C" __global__ void slice_ch_kernel(
    const float* x, float* out,
    int N, int Cin, int Cslice, int H, int W, int c_start);

// ---------------------------------------------------------------------------
// Fully-fused kernel: all 4 convolutions + activations in shared memory
// (defined in c2f_m2_fused.cu).  Fixed at CIN=32, COUT=32, HALFC=16.
// ---------------------------------------------------------------------------
extern "C" __global__ void fused_c2f_model2_kernel(
    const float* __restrict__ x,
    float*       __restrict__ y,
    const float* __restrict__ cv1_w,
    const float* __restrict__ cv1_b,
    const float* __restrict__ m0cv1_w,
    const float* __restrict__ m0cv1_b,
    const float* __restrict__ m0cv2_w,
    const float* __restrict__ m0cv2_b,
    const float* __restrict__ cv2_w,
    const float* __restrict__ cv2_b,
    int H, int W);

static constexpr int    FUSED_TILE_H = 8;
static constexpr int    FUSED_TILE_W = 8;
static constexpr size_t FUSED_SMEM   = 34048;  // see c2f_m2_fused.cu header comment

static void launchSiLU(float* x, size_t n, cudaStream_t stream) {
    if (n == 0) return;
    int threads   = 256;
    int grid_size = static_cast<int>((n >> 2) + (n & 3));
    int blocks    = (grid_size + threads - 1) / threads;
    silu_kernel<<<blocks, threads, 0, stream>>>(x, static_cast<int>(n));
}

static void launchSlice(
    const float* src, float* dst,
    int N, int Cin, int Cslice, int H, int W, int c_start,
    cudaStream_t stream)
{
    int threads = 256;
    int total   = N * Cslice * H * W;
    int blocks  = (total + threads - 1) / threads;
    slice_ch_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, N, Cin, Cslice, H, W, c_start);
}

static void launchAddInplace(float* x, const float* y, size_t n, cudaStream_t stream) {
    int threads = 256;
    int blocks  = static_cast<int>((n + threads - 1) / threads);
    add_inplace_kernel<<<blocks, threads, 0, stream>>>(x, y, static_cast<int>(n));
}

int YoloC2fM2Plugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    (void)outputDesc;

    if (!mDescsCached || !d_cv1_w) {
        if (initialize() != 0) {
            std::cerr << "[enqueue] initialize failed" << std::endl;
            return 1;
        }
    }

    int N = 1, H = 0, W = 0;
    if (inputDesc[0].dims.nbDims == 3) {
        H = inputDesc[0].dims.d[1];
        W = inputDesc[0].dims.d[2];
    } else if (inputDesc[0].dims.nbDims == 4) {
        N = inputDesc[0].dims.d[0];
        H = inputDesc[0].dims.d[2];
        W = inputDesc[0].dims.d[3];
    } else {
        std::cerr << "[enqueue] unsupported nbDims=" << inputDesc[0].dims.nbDims << std::endl;
        return 1;
    }

    if (N != 1) {
        std::cerr << "[enqueue] batch size N>1 not supported" << std::endl;
        return 1;
    }

    const float* x   = static_cast<const float*>(inputs[0]);
    float*       out = static_cast<float*>(outputs[0]);

    // -----------------------------------------------------------------------
    // Fast path: fully-fused shared-memory kernel.
    // Fuses all 4 convolutions (cv1, m0.cv1, m0.cv2, cv2) plus every bias
    // and SiLU activation into one kernel launch with 33 KB of shared memory.
    // Only available for the hardcoded CIN=32 / COUT=32 / HALFC=16 variant.
    // -----------------------------------------------------------------------
    if (mCin == 32 && mCout == 32 && mHalfC == 16) {
        dim3 block(FUSED_TILE_W, FUSED_TILE_H);
        dim3 grid(
            static_cast<unsigned>((W + FUSED_TILE_W - 1) / FUSED_TILE_W),
            static_cast<unsigned>((H + FUSED_TILE_H - 1) / FUSED_TILE_H));

        fused_c2f_model2_kernel<<<grid, block, FUSED_SMEM, stream>>>(
            x, out,
            d_cv1_w,   d_cv1_b,
            d_m0_cv1_w, d_m0_cv1_b,
            d_m0_cv2_w, d_m0_cv2_b,
            d_cv2_w,   d_cv2_b,
            H, W);

        if (cudaGetLastError() == cudaSuccess) return 0;
        std::cerr << "[enqueue] fused kernel failed, falling back to cuDNN" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Fallback: multi-step cuDNN path (handles arbitrary channel counts).
    // -----------------------------------------------------------------------
    if (!workspace) {
        std::cerr << "[enqueue] workspace is null — getWorkspaceSize() must return >0" << std::endl;
        return 1;
    }

    size_t HW        = static_cast<size_t>(H) * W;
    float* cv1_out   = static_cast<float*>(workspace);
    float* m0cv1_out = cv1_out   + static_cast<size_t>(mCout)      * HW;
    float* concat    = m0cv1_out + static_cast<size_t>(mHalfC)     * HW;
    float* x1_buf    = concat;
    float* x2_buf    = concat    + static_cast<size_t>(mHalfC)     * HW;
    float* m0_buf    = concat    + static_cast<size_t>(2 * mHalfC) * HW;
    float* conv_ws   = concat    + static_cast<size_t>(3 * mHalfC) * HW;

    // Step 1: cv1 (1x1) + bias + SiLU.
    if (!runConv(mCv1Desc, x, cv1_out, d_cv1_w, d_cv1_b, conv_ws, stream, false)) {
        std::cerr << "[enqueue] cv1 failed" << std::endl;
        return 1;
    }
    if (!launchBiasSiLUInplace(cv1_out, d_cv1_b, N, mCout, H, W, false, stream)) {
        std::cerr << "[enqueue] cv1 bias+silu failed" << std::endl;
        return 1;
    }

    // Step 2: split channels for x1/x2.
    launchSlice(cv1_out, x1_buf, N, mCout, mHalfC, H, W, 0,      stream);
    launchSlice(cv1_out, x2_buf, N, mCout, mHalfC, H, W, mHalfC, stream);

    // Step 3: m0.cv1 (3x3) + bias + SiLU.
    if (!runConv(mM0Cv1Desc, x2_buf, m0cv1_out, d_m0_cv1_w, d_m0_cv1_b, conv_ws, stream, false)) {
        std::cerr << "[enqueue] m0.cv1 failed" << std::endl;
        return 1;
    }
    if (!launchBiasSiLUInplace(m0cv1_out, d_m0_cv1_b, N, mHalfC, H, W, false, stream)) {
        std::cerr << "[enqueue] m0.cv1 bias+silu failed" << std::endl;
        return 1;
    }

    // Step 4: m0.cv2 (3x3) + bias + SiLU.
    if (!runConv(mM0Cv2Desc, m0cv1_out, m0_buf, d_m0_cv2_w, d_m0_cv2_b, conv_ws, stream, false)) {
        std::cerr << "[enqueue] m0.cv2 failed" << std::endl;
        return 1;
    }
    if (!launchBiasSiLUInplace(m0_buf, d_m0_cv2_b, N, mHalfC, H, W, false, stream)) {
        std::cerr << "[enqueue] m0.cv2 bias+silu failed" << std::endl;
        return 1;
    }

    // Step 5: residual shortcut.
    if (mShortcut) {
        launchAddInplace(m0_buf, x2_buf, static_cast<size_t>(mHalfC) * HW, stream);
    }

    // Step 6: cv2 (1x1) + bias + SiLU.  Output is always NCHW (kLINEAR).
    if (!runConv(mCv2Desc, concat, out, d_cv2_w, d_cv2_b, conv_ws, stream, false)) {
        std::cerr << "[enqueue] cv2 failed" << std::endl;
        return 1;
    }
    if (!launchBiasSiLUInplace(out, d_cv2_b, N, mCout, H, W, false, stream)) {
        std::cerr << "[enqueue] cv2 bias+silu failed" << std::endl;
        return 1;
    }

    return 0;
}
