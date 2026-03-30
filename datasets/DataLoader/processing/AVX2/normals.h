#ifndef NORMALS_AVX2_H_INCLUDED
#define NORMALS_AVX2_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../codecs/image.h"
#include "../getpixels.h"
#include "../sobel.h"

#define F_EPSILON 1e-8

#if INTEL_OPTIMIZATIONS
#include <immintrin.h>
#include <stdalign.h>

static void computeNormalsOnHeatmaps8Bit_AVX2(
    signed char *heatmapPixels,
    unsigned int heatmapWidth,
    unsigned int heatmapHeight,
    unsigned int heatmapChannels,
    unsigned int depthMapSourceChannel,
    unsigned int normalOutputChannel,
    float *gradientX,
    float *gradientY,
    unsigned int borderX,
    unsigned int borderY)
{
    if (!gradientX || !gradientY)
    {
        fprintf(stderr, "Cannot computeNormalsOnHeatmaps8Bit_AVX2 without gradient buffers\n");
        abort();
    }

    if (!heatmapPixels)
    {
        fprintf(stderr, "Cannot computeNormalsOnHeatmaps8Bit_AVX2 without heatmapPixels\n");
        abort();
    }

    // Run Sobel on the depth channel to fill gradientX/gradientY.
    // sobelXY8Bit dispatches to sobelXY8Bit_AVX2 when INTEL_OPTIMIZATIONS is set.
    struct Image heatmapWrapper = {0};
    heatmapWrapper.pixels   = (unsigned char *)heatmapPixels;
    heatmapWrapper.width    = heatmapWidth;
    heatmapWrapper.height   = heatmapHeight;
    heatmapWrapper.channels = heatmapChannels;
    sobelXY8Bit(&heatmapWrapper, gradientX, gradientY, depthMapSourceChannel);

    // Kill one extra border pixel adjacent to the Sobel edge (mirrors _Simple logic).
    borderX += (borderX != 0);
    borderY += (borderY != 0);

    const unsigned int xStart = borderX;
    const unsigned int xEnd   = heatmapWidth  - borderX;
    const unsigned int yStart = borderY;
    const unsigned int yEnd   = heatmapHeight - borderY;

    if (xEnd <= xStart || yEnd <= yStart) { return; }

    const __m256 scale   = _mm256_set1_ps(120.0f);
    const __m256 half    = _mm256_set1_ps(0.5f);
    const __m256 onehalf = _mm256_set1_ps(1.5f);
    const __m256 one     = _mm256_set1_ps(1.0f);

    for (unsigned int y = yStart; y < yEnd; ++y)
    {
        const float *gxp = gradientX + (unsigned long)y * heatmapWidth + xStart;
        const float *gyp = gradientY + (unsigned long)y * heatmapWidth + xStart;
        signed char *hp  = heatmapPixels
                           + ((unsigned long)y * heatmapWidth + xStart) * heatmapChannels
                           + normalOutputChannel;

        unsigned int x = xStart;

        // SIMD inner loop: 8 pixels per iteration
        for (; x + 8 <= xEnd; x += 8, gxp += 8, gyp += 8, hp += (unsigned long)8 * heatmapChannels)
        {
            __m256 dx = _mm256_loadu_ps(gxp);
            __m256 dy = _mm256_loadu_ps(gyp);

            // value = dx^2 + dy^2 + 1.0  (always >= 1.0, rsqrt is numerically safe)
            __m256 value = _mm256_add_ps(
                               _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
                               one);

            // Fast reciprocal sqrt (~11-bit) + one Newton-Raphson step (~23-bit).
            // Accuracy is sufficient for signed char output (range [-120, 120]).
            // Replaces sqrtf() + three div operations with rsqrt + 4 mul + 1 sub.
            __m256 inv_norm = _mm256_rsqrt_ps(value);
            inv_norm = _mm256_mul_ps(inv_norm,
                           _mm256_sub_ps(onehalf,
                               _mm256_mul_ps(half,
                                   _mm256_mul_ps(value,
                                       _mm256_mul_ps(inv_norm, inv_norm)))));

            // nx = 120*dx*inv_norm,  ny = 120*dy*inv_norm,  nz = 120*inv_norm
            __m256 nx_f = _mm256_mul_ps(_mm256_mul_ps(scale, dx), inv_norm);
            __m256 ny_f = _mm256_mul_ps(_mm256_mul_ps(scale, dy), inv_norm);
            __m256 nz_f = _mm256_mul_ps(scale, inv_norm);

            // Convert float -> int32 (round-to-nearest)
            __m256i nx_i32 = _mm256_cvtps_epi32(nx_f);
            __m256i ny_i32 = _mm256_cvtps_epi32(ny_f);
            __m256i nz_i32 = _mm256_cvtps_epi32(nz_f);

            // Pack int32 -> int16 -> int8 (signed saturation).
            // Values are in [-120, 120] so no saturation occurs.
            // Avoids the original 3x int32_t[8] stack arrays (96 bytes -> 24 bytes).
            __m128i nx16 = _mm_packs_epi32(_mm256_castsi256_si128(nx_i32),
                                           _mm256_extracti128_si256(nx_i32, 1));
            __m128i ny16 = _mm_packs_epi32(_mm256_castsi256_si128(ny_i32),
                                           _mm256_extracti128_si256(ny_i32, 1));
            __m128i nz16 = _mm_packs_epi32(_mm256_castsi256_si128(nz_i32),
                                           _mm256_extracti128_si256(nz_i32, 1));

            signed char nx8[8], ny8[8], nz8[8];
            _mm_storel_epi64((__m128i *)nx8, _mm_packs_epi16(nx16, _mm_setzero_si128()));
            _mm_storel_epi64((__m128i *)ny8, _mm_packs_epi16(ny16, _mm_setzero_si128()));
            _mm_storel_epi64((__m128i *)nz8, _mm_packs_epi16(nz16, _mm_setzero_si128()));

            // Scatter-write 3 bytes per pixel at stride heatmapChannels.
            // AVX2 has no byte-granularity scatter, so the stride loop remains scalar.
            signed char *hpj = hp;
            for (int j = 0; j < 8; ++j, hpj += heatmapChannels)
            {
                hpj[0] = nx8[j];
                hpj[1] = ny8[j];
                hpj[2] = nz8[j];
            }
        }

        // Scalar tail: remaining pixels in this row that didn't fill a full SIMD block
        for (; x < xEnd; ++x, ++gxp, ++gyp, hp += heatmapChannels)
        {
            float dx_s       = *gxp;
            float dy_s       = *gyp;
            float inv_norm_s = 1.0f / sqrtf(dx_s * dx_s + dy_s * dy_s + 1.0f);
            hp[0] = (signed char)(120.0f * dx_s * inv_norm_s);
            hp[1] = (signed char)(120.0f * dy_s * inv_norm_s);
            hp[2] = (signed char)(120.0f * inv_norm_s);
        }
    }
}

#endif // INTEL_OPTIMIZATIONS




#ifdef __cplusplus
}
#endif

#endif // NORMALS_AVX2_H_INCLUDED
