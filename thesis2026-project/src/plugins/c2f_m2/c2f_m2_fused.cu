#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int CIN    = 32;
constexpr int COUT   = 32;
constexpr int HALFC  = 16;   // COUT / 2

constexpr int TILE_H = 8;
constexpr int TILE_W = 8;

// m0.cv2 (3×3) needs m0cv1 at (TILE+2)×(TILE+2) → M0CV1_HALO = 1
// m0.cv1 (3×3) needs x2   at (TILE+4)×(TILE+4) → CV1_HALO    = 2
// cv1    (1×1) needs input at exactly the same (TILE+4) positions
constexpr int CV1_HALO   = 2;   // x2/input tile: TILE + 2*2 = 12×12
constexpr int M0CV1_HALO = 1;   // m0cv1 tile:    TILE + 2*1 = 10×10

constexpr int IN_H = TILE_H + 2 * CV1_HALO;    // 12
constexpr int IN_W = TILE_W + 2 * CV1_HALO;    // 12
constexpr int M1_H = TILE_H + 2 * M0CV1_HALO;  // 10
constexpr int M1_W = TILE_W + 2 * M0CV1_HALO;  // 10

__device__ __forceinline__ float silu_fast(float x) {
    return x / (1.f + __expf(-x));
}

__device__ __forceinline__ int idx_chw(int c, int h, int w, int H, int W) {
    return (c * H + h) * W + w;
}

} // namespace

// Smem per block:
//   input_s : CIN  × IN_H  × IN_W  = 32 × 12 × 12 = 4,608 f32 = 18,432 B
//   x2_s    : HALFC× IN_H  × IN_W  = 16 × 12 × 12 = 2,304 f32 =  9,216 B
//   m0cv1_s : HALFC× M1_H  × M1_W  = 16 × 10 × 10 = 1,600 f32 =  6,400 B
//   Total   = 34,048 B (~33 KB)  →  164 KB / 33 KB ≈ 4 blocks/SM on sm87
//
// x1 (first HALFC channels of cv1) is NOT stored in smem:
// each thread only reads x1 at its own (cy,cx) pixel, so it is computed
// locally in step 4 from input_s — saving 9 KB vs the previous version.
//
// Launch: dim3 grid((W+TILE_W-1)/TILE_W, (H+TILE_H-1)/TILE_H)
//         dim3 block(TILE_W, TILE_H)
//         smem = 34,048 bytes
extern "C" __global__ __launch_bounds__(TILE_H * TILE_W, 4)
void fused_c2f_model2_kernel(
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
    int H, int W
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ox = blockIdx.x * TILE_W + tx;
    const int oy = blockIdx.y * TILE_H + ty;

    extern __shared__ float smem[];
    float* input_s = smem;
    float* x2_s    = input_s + CIN   * IN_H * IN_W;  // 18,432 B offset
    float* m0cv1_s = x2_s    + HALFC * IN_H * IN_W;  // + 9,216 B offset

    // -----------------------------------------------------------------------
    // Stage 1 — load input halo tile (12×12, all CIN channels) into smem
    // -----------------------------------------------------------------------
    for (int c = 0; c < CIN; ++c) {
        for (int sy = ty; sy < IN_H; sy += blockDim.y) {
            const int gy  = blockIdx.y * TILE_H + sy - CV1_HALO;
            const bool gy_ok = (unsigned)gy < (unsigned)H;
            for (int sx = tx; sx < IN_W; sx += blockDim.x) {
                const int gx = blockIdx.x * TILE_W + sx - CV1_HALO;
                float v = (gy_ok && (unsigned)gx < (unsigned)W)
                        ? x[idx_chw(c, gy, gx, H, W)] : 0.f;
                input_s[(c * IN_H + sy) * IN_W + sx] = v;
            }
        }
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 2 — compute x2 = SiLU(cv1[HALFC:COUT, :]) at 12×12 → x2_s
    //
    // Only the x2 half of cv1's output is needed beyond this thread's pixel,
    // so only those HALFC channels enter shared memory.
    // x1 = cv1[0:HALFC, :] is deferred to stage 4 (per-thread, no sharing).
    // -----------------------------------------------------------------------
    for (int oc = 0; oc < HALFC; ++oc) {
        for (int sy = ty; sy < IN_H; sy += blockDim.y) {
            for (int sx = tx; sx < IN_W; sx += blockDim.x) {
                float acc = cv1_b[HALFC + oc];
#pragma unroll
                for (int ic = 0; ic < CIN; ++ic)
                    acc += cv1_w[(HALFC + oc) * CIN + ic]
                         * input_s[(ic * IN_H + sy) * IN_W + sx];
                x2_s[(oc * IN_H + sy) * IN_W + sx] = silu_fast(acc);
            }
        }
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 3 — compute m0.cv1 = SiLU(conv3×3(x2)) at 10×10 → m0cv1_s
    // -----------------------------------------------------------------------
    for (int oc = 0; oc < HALFC; ++oc) {
        for (int sy = ty; sy < M1_H; sy += blockDim.y) {
            for (int sx = tx; sx < M1_W; sx += blockDim.x) {
                // m0cv1 tile (10×10) is offset by +1 inside x2_s (12×12)
                const int x2y = sy + 1;
                const int x2x = sx + 1;
                float acc = m0cv1_b[oc];
#pragma unroll
                for (int ic = 0; ic < HALFC; ++ic) {
#pragma unroll
                    for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
                        for (int kx = 0; kx < 3; ++kx)
                            acc += m0cv1_w[((oc * HALFC + ic) * 3 + ky) * 3 + kx]
                                 * x2_s[(ic * IN_H + (x2y + ky - 1)) * IN_W + (x2x + kx - 1)];
                    }
                }
                m0cv1_s[(oc * M1_H + sy) * M1_W + sx] = silu_fast(acc);
            }
        }
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage 4 — compute output pixel (oy, ox) for all COUT channels
    //
    // Per-thread, no further sync needed:
    //  a) x1  — compute locally from input_s (1×1 conv, only this pixel)
    //  b) x2  — read from x2_s at center pixel
    //  c) m0out = SiLU(m0.cv2(m0cv1_s)) + x2   (residual shortcut)
    //  d) y   = SiLU(cv2([x1 | x2 | m0out]))
    // -----------------------------------------------------------------------
    if (ox < W && oy < H) {
        const int cy = ty + CV1_HALO;     // center in input_s / x2_s  (row 2..9)
        const int cx = tx + CV1_HALO;
        const int my = ty + M0CV1_HALO;   // center in m0cv1_s          (row 1..8)
        const int mx = tx + M0CV1_HALO;

        // a) x1 = SiLU(cv1[0:HALFC]) at this pixel
        float x1[HALFC];
#pragma unroll
        for (int oc = 0; oc < HALFC; ++oc) {
            float acc = cv1_b[oc];
#pragma unroll
            for (int ic = 0; ic < CIN; ++ic)
                acc += cv1_w[oc * CIN + ic] * input_s[(ic * IN_H + cy) * IN_W + cx];
            x1[oc] = silu_fast(acc);
        }

        // b) x2 at this pixel
        float x2[HALFC];
#pragma unroll
        for (int c = 0; c < HALFC; ++c)
            x2[c] = x2_s[(c * IN_H + cy) * IN_W + cx];

        // c) m0out = SiLU(m0.cv2) + x2
        float m0out[HALFC];
#pragma unroll
        for (int oc = 0; oc < HALFC; ++oc) {
            float acc = m0cv2_b[oc];
#pragma unroll
            for (int ic = 0; ic < HALFC; ++ic) {
#pragma unroll
                for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
                    for (int kx = 0; kx < 3; ++kx)
                        acc += m0cv2_w[((oc * HALFC + ic) * 3 + ky) * 3 + kx]
                             * m0cv1_s[(ic * M1_H + (my + ky - 1)) * M1_W + (mx + kx - 1)];
                }
            }
            m0out[oc] = silu_fast(acc) + x2[oc];
        }

        // d) y = SiLU(cv2([x1 | x2 | m0out]))
#pragma unroll 4
        for (int oc = 0; oc < COUT; ++oc) {
            float acc      = cv2_b[oc];
            const int base = oc * (3 * HALFC);
#pragma unroll
            for (int c = 0; c < HALFC; ++c) acc += cv2_w[base            + c] * x1[c];
#pragma unroll
            for (int c = 0; c < HALFC; ++c) acc += cv2_w[base + HALFC    + c] * x2[c];
#pragma unroll
            for (int c = 0; c < HALFC; ++c) acc += cv2_w[base + 2 * HALFC+ c] * m0out[c];
            y[idx_chw(oc, oy, ox, H, W)] = silu_fast(acc);
        }
    }
}
