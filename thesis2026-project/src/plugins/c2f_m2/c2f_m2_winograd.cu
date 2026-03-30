#include "c2f_m2_plugin.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace {

__device__ __forceinline__ float silu_fast(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void bias_silu_inplace_kernel(
    float* x,
    const float* bias,
    int N,
    int C,
    int H,
    int W,
    bool isNHWC)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int c = 0;
    if (isNHWC) {
        c = idx % C;
    } else {
        int hw = H * W;
        c = (idx / hw) % C;
    }

    float v = x[idx] + (bias ? bias[c] : 0.0f);
    x[idx] = silu_fast(v);
}

__global__ void winograd_conv3x3_silu_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ winoFilter,
    const float* __restrict__ bias,
    int N,
    int C,
    int K,
    int H,
    int W)
{
    int oc = threadIdx.x;
    if (oc >= K) return;

    int tilesW = (W + 1) >> 1;
    int tileIdx = blockIdx.x;
    int tileY = tileIdx / tilesW;
    int tileX = tileIdx % tilesW;
    int n = blockIdx.y;
    if (n >= N) return;

    int oy = tileY * 2;
    int ox = tileX * 2;

    float M[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) M[i] = 0.0f;

    for (int ic = 0; ic < C; ++ic) {
        float d[4][4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                int iy = oy + r - 1;
                int ix = ox + c - 1;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    int inIdx = ((n * C + ic) * H + iy) * W + ix;
                    d[r][c] = x[inIdx];
                } else {
                    d[r][c] = 0.0f;
                }
            }
        }

        float t[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            t[0][j] = d[0][j] - d[2][j];
            t[1][j] = d[1][j] + d[2][j];
            t[2][j] = -d[1][j] + d[2][j];
            t[3][j] = d[1][j] - d[3][j];
        }

        float v[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i][0] = t[i][0] - t[i][2];
            v[i][1] = t[i][1] + t[i][2];
            v[i][2] = -t[i][1] + t[i][2];
            v[i][3] = t[i][1] - t[i][3];
        }

        const float* u = winoFilter + ((oc * C + ic) * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                M[i * 4 + j] += u[i * 4 + j] * v[i][j];
            }
        }
    }

    float m0[4], m1[4], m2[4], m3[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        m0[j] = M[0 * 4 + j];
        m1[j] = M[1 * 4 + j];
        m2[j] = M[2 * 4 + j];
        m3[j] = M[3 * 4 + j];
    }

    float t0[4], t1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        t0[j] = m0[j] + m1[j] + m2[j];
        t1[j] = m1[j] - m2[j] - m3[j];
    }

    float y00 = t0[0] + t0[1] + t0[2];
    float y01 = t0[1] - t0[2] - t0[3];
    float y10 = t1[0] + t1[1] + t1[2];
    float y11 = t1[1] - t1[2] - t1[3];

    float b = bias ? bias[oc] : 0.0f;
    y00 = silu_fast(y00 + b);
    y01 = silu_fast(y01 + b);
    y10 = silu_fast(y10 + b);
    y11 = silu_fast(y11 + b);

    if (oy < H && ox < W) {
        y[((n * K + oc) * H + oy) * W + ox] = y00;
    }
    if (oy < H && (ox + 1) < W) {
        y[((n * K + oc) * H + oy) * W + (ox + 1)] = y01;
    }
    if ((oy + 1) < H && ox < W) {
        y[((n * K + oc) * H + (oy + 1)) * W + ox] = y10;
    }
    if ((oy + 1) < H && (ox + 1) < W) {
        y[((n * K + oc) * H + (oy + 1)) * W + (ox + 1)] = y11;
    }
}

} // namespace

bool precomputeWinoFilterTransform(
    float const* hFilter, int outC, int inC, float** dTransformed) noexcept
{
    if (!hFilter || !dTransformed || outC <= 0 || inC <= 0) return false;

    // Winograd F(2,3): U = G * g * G^T
    static constexpr float G[4][3] = {
        {1.0f, 0.0f, 0.0f},
        {0.5f, 0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f},
        {0.0f, 0.0f, 1.0f},
    };

    std::vector<float> hU(static_cast<size_t>(outC) * inC * 16, 0.0f);

    for (int oc = 0; oc < outC; ++oc) {
        for (int ic = 0; ic < inC; ++ic) {
            float g[3][3];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    size_t idx = ((static_cast<size_t>(oc) * inC + ic) * 3 + r) * 3 + c;
                    g[r][c] = hFilter[idx];
                }
            }

            float t[4][3];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 3; ++j) {
                    t[i][j] = G[i][0] * g[0][j] + G[i][1] * g[1][j] + G[i][2] * g[2][j];
                }
            }

            float u[4][4];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    u[i][j] = t[i][0] * G[j][0] + t[i][1] * G[j][1] + t[i][2] * G[j][2];
                }
            }

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    size_t outIdx = ((static_cast<size_t>(oc) * inC + ic) * 16) + (i * 4 + j);
                    hU[outIdx] = u[i][j];
                }
            }
        }
    }

    size_t bytes = hU.size() * sizeof(float);
    if (cudaMalloc(reinterpret_cast<void**>(dTransformed), bytes) != cudaSuccess) {
        *dTransformed = nullptr;
        return false;
    }

    if (cudaMemcpy(*dTransformed, hU.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*dTransformed);
        *dTransformed = nullptr;
        return false;
    }

    return true;
}

bool launchBiasSiLUInplace(
    float* x,
    float const* bias,
    int N,
    int C,
    int H,
    int W,
    bool isNHWC,
    cudaStream_t stream) noexcept
{
    if (!x || !bias || N <= 0 || C <= 0 || H <= 0 || W <= 0) return false;

    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    bias_silu_inplace_kernel<<<blocks, threads, 0, stream>>>(x, bias, N, C, H, W, isNHWC);
    return cudaGetLastError() == cudaSuccess;
}

bool launchWinogradConv3x3SiLU(
    float const* x,
    float* y,
    float const* winoFilter,
    float const* bias,
    int N,
    int C,
    int K,
    int H,
    int W,
    cudaStream_t stream) noexcept
{
    if (!x || !y || !winoFilter || !bias) return false;
    if (N != 1 || C != 16 || K != 16 || H <= 0 || W <= 0) return false;

    int tilesH = (H + 1) >> 1;
    int tilesW = (W + 1) >> 1;

    dim3 block(32, 1, 1);
    dim3 grid(static_cast<unsigned>(tilesH * tilesW), static_cast<unsigned>(N), 1);

    winograd_conv3x3_silu_kernel<<<grid, block, 0, stream>>>(
        x, y, winoFilter, bias, N, C, K, H, W);

    return cudaGetLastError() == cudaSuccess;
}
