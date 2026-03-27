#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// SiLU (Swish) activation: x * sigmoid(x)
//
// Optimisation notes for Orin (sm87 Ampere):
//  • float4 vectorised load/store: 4× throughput vs scalar
//  • __expf: hardware fast-math approximation (~12 ulp, fine for inference)
//  • In-place operation: single tensor argument
//
// Grid covers (n/4 + n%4) threads.  Each thread handles either:
//   idx < n/4        → float4 vectorised path (4 elements at once)
//   idx in [n/4, n/4 + n%4) → scalar tail path (0–3 leftover elements)
// This correctly handles n < 4 (no vectorised work, only scalar tail).
// ---------------------------------------------------------------------------
extern "C" __global__
void silu_kernel(float* __restrict__ x, int n) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_n  = n >> 2;          // n / 4
    int tail_n = n & 3;           // n % 4  (0, 1, 2, or 3)

    if (idx < vec_n) {
        // Vectorised path — 4 elements per thread
        float4 v = reinterpret_cast<float4*>(x)[idx];
        v.x = v.x / (1.f + __expf(-v.x));
        v.y = v.y / (1.f + __expf(-v.y));
        v.z = v.z / (1.f + __expf(-v.z));
        v.w = v.w / (1.f + __expf(-v.w));
        reinterpret_cast<float4*>(x)[idx] = v;
    } else if (idx < vec_n + tail_n) {
        // Scalar tail — at most 3 threads hit this branch
        int i    = (vec_n << 2) + (idx - vec_n);
        float a  = x[i];
        x[i]     = a / (1.f + __expf(-a));
    }
}

// ---------------------------------------------------------------------------
// In-place element-wise addition: x[i] += y[i]
// ---------------------------------------------------------------------------
extern "C" __global__
void add_inplace_kernel(float* __restrict__ x,
                        const float* __restrict__ y,
                        int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// ---------------------------------------------------------------------------
// Channel slice — copy channels [c_start, c_start + Cslice) from
// x[N, Cin, H, W] into out[N, Cslice, H, W].  NCHW layout.
// ---------------------------------------------------------------------------
extern "C" __global__
void slice_ch_kernel(
    const float* __restrict__ x,
    float*       __restrict__ out,
    int N, int Cin, int Cslice, int H, int W,
    int c_start)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int hw    = H * W;
    int total = N * Cslice * hw;
    if (idx >= total) return;

    int n   =  idx / (Cslice * hw);
    int rem =  idx % (Cslice * hw);
    int c   =  rem / hw;
    int off =  rem % hw;

    out[idx] = x[(n * Cin + (c + c_start)) * hw + off];
}

// ---------------------------------------------------------------------------
// Three-tensor channel concatenation (already in correct positions — see
// enqueue workspace layout).  This kernel is kept for cases where the three
// source tensors are not already contiguous; for the optimised layout used
// in enqueue() the concat buffer is filled in-place and this kernel is NOT
// called.  Retained here for reference / alternate implementations.
// ---------------------------------------------------------------------------
extern "C" __global__
void concat3_ch_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float*       __restrict__ out,
    int N, int C, int H, int W)
{
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int hw   = H * W;
    int c3   = 3 * C;
    int total = N * c3 * hw;
    if (idx >= total) return;

    int n    =  idx / (c3 * hw);
    int rem  =  idx % (c3 * hw);
    int ch   =  rem / hw;
    int off  =  rem % hw;

    const float* src;
    int src_ch;
    if (ch < C) {
        src = a; src_ch = ch;
    } else if (ch < 2 * C) {
        src = b; src_ch = ch - C;
    } else {
        src = c; src_ch = ch - 2 * C;
    }
    out[idx] = src[(n * C + src_ch) * hw + off];
}
