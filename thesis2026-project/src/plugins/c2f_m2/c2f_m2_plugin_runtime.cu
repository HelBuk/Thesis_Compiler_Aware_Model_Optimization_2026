#include "c2f_m2_plugin.hpp"
#include <cuda_runtime.h>
#include <cudnn.h>

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
// Convenience macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK_LAST()                                       \
    do {                                                        \
        cudaError_t _e = cudaGetLastError();                    \
        if (_e != cudaSuccess) return 1;                        \
    } while (0)

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------
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
// enqueue — C2f(m=1) forward pass
//
// Architecture for YOLOv8n model.2 (Cin=32, Cout=64, halfC=32, H=W=160):
//
//   input [N, Cin, H, W]
//     └─► cv1 (1×1, Cin→Cout) + SiLU  → cv1_out [N, Cout, H, W]
//              ├── x1 = cv1_out[:,  :halfC, :, :]  (slice 0)
//              └── x2 = cv1_out[:, halfC:, :, :]  (slice 1)
//                         └─► m0.cv1 (3×3) + SiLU → m0cv1_out [N,halfC,H,W]
//                               └─► m0.cv2 (3×3) + SiLU → m0out
//                                     └─► m0out += x2  (shortcut residual)
//   concat [x1 | x2 | m0out] → [N, 3*halfC, H, W]
//     └─► cv2 (1×1, 3*halfC→Cout) + SiLU → output [N, Cout, H, W]
//
// Workspace layout (no aliasing):
//   [0]  cv1_out    [N * Cout    * H * W]  — cv1 output
//   [1]  m0cv1_out  [N * halfC   * H * W]  — bottleneck cv1 output
//   [2]  concat     [N * 3*halfC * H * W]  — cv2 input; filled in 3 parts:
//          concat[0..halfC]:       x1 (sliced from cv1_out)
//          concat[halfC..2*halfC]: x2 (sliced from cv1_out; also shortcut)
//          concat[2*halfC..:    ]: m0out (m0.cv2 output + x2 residual)
//   [3]  conv_ws    [max workspace of all 4 convolutions]
//
// Key correctness fix vs. original code:
//   — x1 and x2 live in SEPARATE regions of concat[], never aliased with
//     m0cv1_out.  The original code set x1_buf=m0_out which caused m0.cv1
//     to silently overwrite x1 before the concat step.
// ===========================================================================
int YoloC2fM2Plugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const*       outputs,
    void*              workspace,
    cudaStream_t       stream) noexcept
{
    if (!mCudnn || !mDescsCached) return 1;
    if (inputDesc[0].dims.nbDims != 4) return 1;

    int N   = inputDesc[0].dims.d[0];
    int Cin = inputDesc[0].dims.d[1];   // == mCin
    int H   = inputDesc[0].dims.d[2];
    int W   = inputDesc[0].dims.d[3];

    if (N <= 0 || H <= 0 || W <= 0 || Cin <= 0) return 1;

    const float* x   = static_cast<const float*>(inputs[0]);
    float*       out = static_cast<float*>(outputs[0]);

    const size_t HW        = static_cast<size_t>(H) * W;
    const size_t cv1Total  = static_cast<size_t>(N) * mCout  * HW;
    const size_t halfTotal = static_cast<size_t>(N) * mHalfC * HW;

    // -----------------------------------------------------------------------
    // Carve workspace into named regions (no aliasing)
    // -----------------------------------------------------------------------
    char* ws = static_cast<char*>(workspace);

    float* cv1_out   = reinterpret_cast<float*>(ws);
    ws += cv1Total  * sizeof(float);

    float* m0cv1_out = reinterpret_cast<float*>(ws);
    ws += halfTotal * sizeof(float);

    float* concat    = reinterpret_cast<float*>(ws);
    ws += 3 * halfTotal * sizeof(float);

    void*  conv_ws   = ws;   // cuDNN scratch — one region shared across all convs

    // Pointers into the three concat segments
    float* x1    = concat;                       // [0,  halfC)
    float* x2    = concat + halfTotal;            // [halfC, 2*halfC)
    float* m0out = concat + 2 * halfTotal;        // [2*halfC, 3*halfC)

    // Max workspace available for cuDNN (rest of the allocated workspace)
    size_t maxConvWs = std::max({
        mCv1Desc.workspaceBytes,
        mM0Cv1Desc.workspaceBytes,
        mM0Cv2Desc.workspaceBytes,
        mCv2Desc.workspaceBytes});

    // -----------------------------------------------------------------------
    // 1) cv1 : [N, Cin, H, W] → [N, Cout, H, W]  (1×1 conv + bias)
    // -----------------------------------------------------------------------
    if (!runConv(mCv1Desc, x, cv1_out, d_cv1_w, d_cv1_b, conv_ws, stream))
        return 1;
    launchSiLU(cv1_out, cv1Total, stream);
    CUDA_CHECK_LAST();

    // -----------------------------------------------------------------------
    // 2) Split cv1_out into x1 (concat seg 0) and x2 (concat seg 1)
    // -----------------------------------------------------------------------
    launchSlice(cv1_out, x1, N, mCout, mHalfC, H, W, 0,       stream);
    CUDA_CHECK_LAST();
    launchSlice(cv1_out, x2, N, mCout, mHalfC, H, W, mHalfC,  stream);
    CUDA_CHECK_LAST();

    // -----------------------------------------------------------------------
    // 3) m0.cv1 : x2 → m0cv1_out  (3×3 conv + bias)
    // -----------------------------------------------------------------------
    if (!runConv(mM0Cv1Desc, x2, m0cv1_out, d_m0_cv1_w, d_m0_cv1_b, conv_ws, stream))
        return 1;
    launchSiLU(m0cv1_out, halfTotal, stream);
    CUDA_CHECK_LAST();

    // -----------------------------------------------------------------------
    // 4) m0.cv2 : m0cv1_out → m0out  (3×3 conv + bias)
    //    m0out lives in concat seg 2 — written directly, no extra copy
    // -----------------------------------------------------------------------
    if (!runConv(mM0Cv2Desc, m0cv1_out, m0out, d_m0_cv2_w, d_m0_cv2_b, conv_ws, stream))
        return 1;
    launchSiLU(m0out, halfTotal, stream);
    CUDA_CHECK_LAST();

    // -----------------------------------------------------------------------
    // 5) Shortcut residual: m0out += x2
    //    x2 and m0out are in different concat segments — safe, no aliasing
    // -----------------------------------------------------------------------
    if (mShortcut) {
        launchAddInplace(m0out, x2, halfTotal, stream);
        CUDA_CHECK_LAST();
    }

    // -----------------------------------------------------------------------
    // 6) cv2 : concat [N, 3*halfC, H, W] → out [N, Cout, H, W]  (1×1 + bias)
    //    concat is already contiguous [x1 | x2 | m0out] — no explicit concat
    //    kernel needed.
    // -----------------------------------------------------------------------
    if (!runConv(mCv2Desc, concat, out, d_cv2_w, d_cv2_b, conv_ws, stream))
        return 1;
    launchSiLU(out, cv1Total, stream);   // cv1Total == N*Cout*H*W
    CUDA_CHECK_LAST();

    return 0;
}
