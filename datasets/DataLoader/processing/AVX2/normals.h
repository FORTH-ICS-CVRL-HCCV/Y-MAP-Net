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
    float *gradientY)
{
    if (!gradientX || !gradientY)
    {
        fprintf(stderr, "Cannot computeNormalsOnHeatmaps8Bit without gradient buffers\n");
        abort();
    }

    if (!heatmapPixels)
    {
        fprintf(stderr, "Cannot computeNormalsOnHeatmaps8Bit without heatmapPixels\n");
        abort();
    }

    const int totalPixels = heatmapWidth * heatmapHeight;
    unsigned long gradientIndex = 0;
    unsigned long heatmapIndex  = normalOutputChannel;

    __m256 scaleFactor = _mm256_set1_ps(120.0f);  // Scaling factor for [-1, 1] to [-120, 120]
    __m256 epsilon = _mm256_set1_ps(F_EPSILON);   // Small epsilon to avoid divide-by-zero

    for (int i = 0; i < totalPixels; i += 8)
    {
        // Load 8 float values from gradientX and gradientY
        __m256 dx = _mm256_loadu_ps(&gradientX[gradientIndex]);
        __m256 dy = _mm256_loadu_ps(&gradientY[gradientIndex]);
        gradientIndex += 8;

        // Compute value = (dx * dx) + (dy * dy) + 1.0
        __m256 dx2 = _mm256_mul_ps(dx, dx);
        __m256 dy2 = _mm256_mul_ps(dy, dy);
        __m256 value = _mm256_add_ps(_mm256_add_ps(dx2, dy2), _mm256_set1_ps(1.0f));

        // norm = sqrtf(value) + F_EPSILON
        __m256 norm = _mm256_sqrt_ps(value);
        norm = _mm256_add_ps(norm, epsilon);

        // nx = 120.0 * (dx / norm), ny = 120.0 * (dy / norm), nz = 120.0 / norm
        __m256 nx = _mm256_mul_ps(scaleFactor, _mm256_div_ps(dx, norm));
        __m256 ny = _mm256_mul_ps(scaleFactor, _mm256_div_ps(dy, norm));
        __m256 nz = _mm256_div_ps(scaleFactor, norm);

        // Convert results to 32-bit integers
        __m256i nx_i32 = _mm256_cvtps_epi32(nx);
        __m256i ny_i32 = _mm256_cvtps_epi32(ny);
        __m256i nz_i32 = _mm256_cvtps_epi32(nz);

        // Temporary arrays for storing results
        int32_t nx_results[8], ny_results[8], nz_results[8];
        _mm256_storeu_si256((__m256i *)nx_results, nx_i32);
        _mm256_storeu_si256((__m256i *)ny_results, ny_i32);
        _mm256_storeu_si256((__m256i *)nz_results, nz_i32);

        // Write results to heatmapPixels
        for (int j = 0; j < 8; ++j)
        {
            heatmapPixels[heatmapIndex] = (signed char)nx_results[j];  // Heatmap 18
            heatmapIndex += 1;
            heatmapPixels[heatmapIndex] = (signed char)ny_results[j];  // Heatmap 19
            heatmapIndex += 1;
            heatmapPixels[heatmapIndex] = (signed char)nz_results[j];  // Heatmap 20
            heatmapIndex += heatmapChannels - 2;
        }
    }
}

#endif // INTEL_OPTIMIZATIONS










#ifdef __cplusplus
}
#endif

#endif // NORMALS_AVX2_H_INCLUDED
