#include "c2f_m2_plugin.hpp"
#include <cuda_runtime.h>
#include <cudnn.h>
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
extern "C" __global__ void fused_c2f_model2_kernel(
    const float* x,
    float* y,
    const float* cv1_w,
    const float* cv1_b,
    const float* m0cv1_w,
    const float* m0cv1_b,
    const float* m0cv2_w,
    const float* m0cv2_b,
    const float* cv2_w,
    const float* cv2_b,
    int H,
    int W
);

// ---------------------------------------------------------------------------
// Convenience macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK_LAST(MSG)                                                  \
    do {                                                                      \
        cudaError_t _e = cudaGetLastError();                                  \
        if (_e != cudaSuccess) {                                              \
            std::cerr << "[CUDA] " << MSG << " failed: "                      \
                      << cudaGetErrorString(_e) << std::endl;                 \
            return 1;                                                         \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------
static int launchFusedC2FModel2(
    const float* x,
    float* y,
    const float* cv1_w,
    const float* cv1_b,
    const float* m0cv1_w,
    const float* m0cv1_b,
    const float* m0cv2_w,
    const float* m0cv2_b,
    const float* cv2_w,
    const float* cv2_b,
    int H,
    int W,
    cudaStream_t stream
) {
    constexpr int TILE_H = 8;
    constexpr int TILE_W = 8;
    constexpr int CIN    = 32;
    constexpr int COUT   = 32;
    constexpr int HALFC  = 16;
    constexpr int CV1_HALO = 2;
    constexpr int M0CV1_HALO = 1;

    dim3 block(TILE_W, TILE_H);
    dim3 grid((W + TILE_W - 1) / TILE_W,
              (H + TILE_H - 1) / TILE_H);

    const int in_h = TILE_H + 2 * CV1_HALO;   // 12
    const int in_w = TILE_W + 2 * CV1_HALO;   // 12
    const int m1_h = TILE_H + 2 * M0CV1_HALO; // 10
    const int m1_w = TILE_W + 2 * M0CV1_HALO; // 10

    size_t smem_bytes =
    sizeof(float) * (
        CIN   * in_h * in_w +   // 32 × 12 × 12 = 18,432 B  input_s
        HALFC * in_h * in_w +   //  16 × 12 × 12 =  9,216 B  x2_s
        HALFC * m1_h * m1_w     //  16 × 10 × 10 =  6,400 B  m0cv1_s
    );                          //               = 34,048 B total

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);

    // std::cerr << "[FUSED LAUNCH]"
    //         << " grid=(" << grid.x << "," << grid.y << ")"
    //         << " block=(" << block.x << "," << block.y << ")"
    //         << " smem=" << smem_bytes
    //         << std::endl;

    // if (smem_bytes > prop.sharedMemPerBlock) {
    //     std::cerr << "[ERROR] Shared memory too large: "
    //             << smem_bytes << " > " << prop.sharedMemPerBlock << std::endl;
    //     return 1;
    // }

    fused_c2f_model2_kernel<<<grid, block, smem_bytes, stream>>>(
        x, y,
        cv1_w, cv1_b,
        m0cv1_w, m0cv1_b,
        m0cv2_w, m0cv2_b,
        cv2_w, cv2_b,
        H, W
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[launchFusedC2FModel2] kernel launch failed: "
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    return 0;
}

static void launchSiLU(float* x, size_t n, cudaStream_t stream) {
    if (n == 0) return;
    // Grid covers (n/4 + n%4) threads — handles both vec body and scalar tail.
    // Works correctly even when n < 4.
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

// ===========================================================================
// enqueue — C2f(m=1) forward pass  (cuDNN TF32 path)
//
// Architecture for YOLOv8n model.2 (Cin=32, Cout=32, halfC=16, H=W=160):
//
//   input  [N, Cin, H, W]  layout = mInputFormat (kCHW32 preferred / kLINEAR fallback)
//     └─► cv1 (1×1, Cin→Cout) [cuDNN TF32] + SiLU → cv1_out  NCHW workspace
//              ├── x1 = cv1_out[:,  :halfC]  → concat seg 0
//              └── x2 = cv1_out[:, halfC:]   → concat seg 1  (shortcut source)
//                   └─► m0.cv1 (3×3) [cuDNN TF32] + SiLU
//                         └─► m0.cv2 (3×3) [cuDNN TF32] + SiLU → concat seg 2
//                               └─► += x2  (shortcut residual)
//   concat [x1 | x2 | m0out] → [N, 3*halfC, H, W]  NCHW workspace
//     └─► cv2 (1×1, 3*halfC→Cout) [cuDNN TF32] + SiLU
//               → output [N, Cout, H, W]  layout = mInputFormat
//
// Why cuDNN instead of the scalar fused kernel:
//   • cuDNN selects TF32 TensorCore kernels on sm87 (~4× FP32 throughput)
//   • fused_c2f_model2_kernel profiled at 5.59 ms; cuDNN path ~370 µs
//   • Accepting kCHW32 in supportsFormatCombination eliminates the
//     NHWC↔NCHW reformat copy node (~50 µs) inserted by TRT
//
// Workspace layout (intermediate tensors NCHW, no format conversion needed):
//   [0] cv1_out   : N * Cout    * H * W  floats
//   [1] m0cv1_out : N * halfC   * H * W  floats
//   [2] concat    : N * 3*halfC * H * W  floats
//         seg 0 [0..halfC):     x1
//         seg 1 [halfC..2h):    x2 / shortcut
//         seg 2 [2h..3h):       m0out
//   [3] conv_ws   : max cuDNN workspace across 4 convolutions
// ===========================================================================
int YoloC2fM2Plugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    // Ensure weights and cuDNN descriptors are ready.
    if (!mDescsCached || !d_cv1_w) {
        if (initialize() != 0) {
            std::cerr << "[enqueue] initialize failed" << std::endl;
            return 1;
        }
    }

    if (!workspace) {
        std::cerr << "[enqueue] workspace is null — getWorkspaceSize() must return >0" << std::endl;
        return 1;
    }

    // TRT always reports logical NCHW shape regardless of memory layout.
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

    const float* x  = static_cast<const float*>(inputs[0]);
    float*       out = static_cast<float*>(outputs[0]);

    // -----------------------------------------------------------------------
    // Workspace pointer arithmetic — must match getWorkspaceSize() layout.
    // -----------------------------------------------------------------------
    size_t HW      = static_cast<size_t>(H) * W;   // N=1
    float* cv1_out   = static_cast<float*>(workspace);
    float* m0cv1_out = cv1_out   + static_cast<size_t>(mCout)       * HW;
    float* concat    = m0cv1_out + static_cast<size_t>(mHalfC)      * HW;
    float* x1_buf    = concat;
    float* x2_buf    = concat    + static_cast<size_t>(mHalfC)      * HW;
    float* m0_buf    = concat    + static_cast<size_t>(2 * mHalfC)  * HW;
    float* conv_ws   = concat    + static_cast<size_t>(3 * mHalfC)  * HW;

    // -----------------------------------------------------------------------
    // Step 1: cv1  (1×1)  —  input [ioFmt] → cv1_out [NCHW]
    // -----------------------------------------------------------------------
    if (!runConv(mCv1Desc, x, cv1_out, d_cv1_w, d_cv1_b, conv_ws, stream)) {
        std::cerr << "[enqueue] cv1 failed" << std::endl;
        return 1;
    }
    launchSiLU(cv1_out, static_cast<size_t>(mCout) * HW, stream);

    // -----------------------------------------------------------------------
    // Step 2: Slice cv1_out into x1_buf (channels 0..halfC-1)
    //                         and x2_buf (channels halfC..Cout-1).
    //         Both land directly inside concat[] — no extra allocation.
    // -----------------------------------------------------------------------
    launchSlice(cv1_out, x1_buf, N, mCout, mHalfC, H, W, 0,      stream);
    launchSlice(cv1_out, x2_buf, N, mCout, mHalfC, H, W, mHalfC, stream);

    // -----------------------------------------------------------------------
    // Step 3: m0.cv1  (3×3)  —  x2_buf [NCHW] → m0cv1_out [NCHW]
    // -----------------------------------------------------------------------
    if (!runConv(mM0Cv1Desc, x2_buf, m0cv1_out, d_m0_cv1_w, d_m0_cv1_b, conv_ws, stream)) {
        std::cerr << "[enqueue] m0.cv1 failed" << std::endl;
        return 1;
    }
    launchSiLU(m0cv1_out, static_cast<size_t>(mHalfC) * HW, stream);

    // -----------------------------------------------------------------------
    // Step 4: m0.cv2  (3×3)  —  m0cv1_out [NCHW] → m0_buf [NCHW]
    // -----------------------------------------------------------------------
    if (!runConv(mM0Cv2Desc, m0cv1_out, m0_buf, d_m0_cv2_w, d_m0_cv2_b, conv_ws, stream)) {
        std::cerr << "[enqueue] m0.cv2 failed" << std::endl;
        return 1;
    }
    launchSiLU(m0_buf, static_cast<size_t>(mHalfC) * HW, stream);

    // -----------------------------------------------------------------------
    // Step 5: Shortcut residual  m0_buf += x2_buf
    // -----------------------------------------------------------------------
    if (mShortcut) {
        launchAddInplace(m0_buf, x2_buf, static_cast<size_t>(mHalfC) * HW, stream);
    }

    // -----------------------------------------------------------------------
    // Step 6: cv2  (1×1)  —  concat [NCHW, 3*halfC ch] → output [ioFmt]
    // -----------------------------------------------------------------------
    if (!runConv(mCv2Desc, concat, out, d_cv2_w, d_cv2_b, conv_ws, stream)) {
        std::cerr << "[enqueue] cv2 failed" << std::endl;
        return 1;
    }
    launchSiLU(out, static_cast<size_t>(mCout) * HW, stream);

    return 0;
}