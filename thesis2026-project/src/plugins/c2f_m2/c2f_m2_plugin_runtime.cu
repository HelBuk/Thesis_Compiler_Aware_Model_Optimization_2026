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
            CIN   * in_h * in_w +
            COUT  * in_h * in_w +
            HALFC * m1_h * m1_w
        );

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cerr << "[FUSED LAUNCH]"
            << " grid=(" << grid.x << "," << grid.y << ")"
            << " block=(" << block.x << "," << block.y << ")"
            << " smem=" << smem_bytes
            << std::endl;

    if (smem_bytes > prop.sharedMemPerBlock) {
        std::cerr << "[ERROR] Shared memory too large: "
                << smem_bytes << " > " << prop.sharedMemPerBlock << std::endl;
        return 1;
    }

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
// enqueue — C2f(m=1) forward pass
//
// Architecture for YOLOv8n model.2 (Cin=32, Cout=32, halfC=16, H=W=160):
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
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    if (!d_cv1_w || !d_cv1_b || !d_m0_cv1_w || !d_m0_cv1_b ||
        !d_m0_cv2_w || !d_m0_cv2_b || !d_cv2_w || !d_cv2_b) {
        if (initialize() != 0) {
            std::cerr << "[enqueue] initialize failed" << std::endl;
            return 1;
        }
    }

    int N = 1;
    int Cin = 0;
    int H = 0;
    int W = 0;

    if (inputDesc[0].dims.nbDims == 3) {
        Cin = inputDesc[0].dims.d[0];
        H   = inputDesc[0].dims.d[1];
        W   = inputDesc[0].dims.d[2];
        N   = 1;
    } else if (inputDesc[0].dims.nbDims == 4) {
        N   = inputDesc[0].dims.d[0];
        Cin = inputDesc[0].dims.d[1];
        H   = inputDesc[0].dims.d[2];
        W   = inputDesc[0].dims.d[3];
    } else {
        std::cerr << "[enqueue] unsupported nbDims=" << inputDesc[0].dims.nbDims << std::endl;
        return 1;
    }

    if (N != 1) {
        std::cerr << "[enqueue] fused kernel currently supports N=1 only" << std::endl;
        return 1;
    }

    if (Cin != 32 || mCin != 32 || mHalfC != 16 || mCout != 32) {
        std::cerr << "[enqueue] fused kernel expects Cin=32, halfC=16, Cout=32"
                  << " but got Cin=" << Cin
                  << " mCin=" << mCin
                  << " mHalfC=" << mHalfC
                  << " mCout=" << mCout
                  << std::endl;
        return 1;
    }

    const float* x = static_cast<const float*>(inputs[0]);
    float* out     = static_cast<float*>(outputs[0]);

    return launchFusedC2FModel2(
        x, out,
        d_cv1_w, d_cv1_b,
        d_m0_cv1_w, d_m0_cv1_b,
        d_m0_cv2_w, d_m0_cv2_b,
        d_cv2_w, d_cv2_b,
        H, W, stream
    );
}