#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void silu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

extern "C" __global__
void add_inplace_kernel(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// Extract channels [c_start, c_start + Cslice) from x[N,Cin,H,W] into out[N,Cslice,H,W]
extern "C" __global__
void slice_ch_kernel(
    const float* x,
    float* out,
    int N, int Cin, int Cslice, int H, int W,
    int c_start
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hw = H * W;
    int total = N * Cslice * hw;
    if (idx >= total) return;

    int inner = idx % (Cslice * hw);
    int c = inner / hw;
    int off = inner % hw;
    int n = idx / (Cslice * hw);

    out[idx] = x[(n * Cin + (c + c_start)) * hw + off];
}

extern "C" __global__
void concat3_ch_kernel(
    const float* a, const float* b, const float* c,
    float* out,
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hw = H * W;
    int c3 = 3 * C;
    int total = N * c3 * hw;
    if (idx >= total) return;

    int inner = idx % (c3 * hw);
    int ch = inner / hw;
    int off = inner % hw;
    int n = idx / (c3 * hw);

    const float* src;
    int src_ch;
    if (ch < C) {
        src = a;
        src_ch = ch;
    } else if (ch < 2 * C) {
        src = b;
        src_ch = ch - C;
    } else {
        src = c;
        src_ch = ch - 2 * C;
    }

    out[idx] = src[(n * C + src_ch) * hw + off];
}