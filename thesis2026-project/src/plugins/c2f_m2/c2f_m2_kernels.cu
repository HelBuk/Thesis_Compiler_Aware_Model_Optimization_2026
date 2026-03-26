// src/plugins/c2f_m2/c2f_m2_kernels.cu
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

    int inner = idx % (c3 * hw);    // position inside one batch item
    int ch = inner / hw;            // output channel index
    int off = inner % hw;           // flattened (h, w) location
    int n = idx / (c3 * hw);        // the batch number

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