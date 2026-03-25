#ifndef SOBEL_AVX2_H_INCLUDED
#define SOBEL_AVX2_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if INTEL_OPTIMIZATIONS

#include <stdalign.h>

#include "../../codecs/image.h"
#include <immintrin.h>
#include <stdlib.h>


//This works 2x slower :( than sobelXY8Bit
static void sobel8Bit2Way_ignore_center_AVX2(struct Image *image, float *gradientX, int sobelKernelX[3][3], float *gradientY, int sobelKernelY[3][3], int channel)
{
  if ( (sobelKernelX[1][1]!=0.0)  || (sobelKernelY[1][1]!=0.0) )
   {
       fprintf(stderr,"sobel8Bit2Way_ignore_center_AVX2 will not work with non null center kernel 3x3 element\n");
       fprintf(stderr,"AVX2 has 8 registers not 9\n");
       abort();
   }

  uint8_t *pixelsInput = image->pixels;
  int width            = image->width;
  int height           = image->height;
  int channels         = image->channels;

   __m256 kX = _mm256_set_ps( sobelKernelX[0][0], sobelKernelX[0][1], sobelKernelX[0][2],
                              sobelKernelX[1][0],      /*empty*/      sobelKernelX[1][2],
                              sobelKernelX[2][0], sobelKernelX[2][1], sobelKernelX[2][2]);

   __m256 kY = _mm256_set_ps( sobelKernelY[0][0], sobelKernelY[0][1], sobelKernelY[0][2],
                              sobelKernelY[1][0],      /*empty*/      sobelKernelY[1][2],
                              sobelKernelY[2][0], sobelKernelY[2][1], sobelKernelY[2][2]);


   int gradientTarget;
   int indexStart,nextLineIndexStart,nextnextLineIndexStart;
   int start_x = 0;
   float gradX = 0.0f, gradY = 0.0f;

   for (int y = 1; y < height - 1; ++y)
   {
    indexStart = (y * width + start_x) * channels + channel;
    for (int x = start_x; x < width - 2; x += 1)
    { // Ensure valid blocks
        nextLineIndexStart     = indexStart + (width*channels);
        nextnextLineIndexStart = nextLineIndexStart + (width*channels);

        // Load pixel values into AVX2 register
        __m256 pixels = _mm256_set_ps((float) pixelsInput[indexStart + 0],              (float) pixelsInput[indexStart + channels],        (float) pixelsInput[indexStart + (2*channels)],
                                      (float) pixelsInput[nextLineIndexStart + 0],      /* empty */                                      (float) pixelsInput[nextLineIndexStart + (2*channels)],
                                      (float) pixelsInput[nextnextLineIndexStart + 0], (float)  pixelsInput[nextnextLineIndexStart + channels],  (float) pixelsInput[nextnextLineIndexStart + (2*channels)]);

        // Multiply pixels with kernels
        __m256 productX = _mm256_mul_ps(kX, pixels);
        __m256 productY = _mm256_mul_ps(kY, pixels);

        // Sum all elements in the AVX2 registers
        //float gradX = 0.0f, gradY = 0.0f;
        gradX = productX[0] + productX[1] + productX[2] + productX[3] + productX[4] + productX[5] + productX[6] + productX[7];
        gradY = productY[0] + productY[1] + productY[2] + productY[3] + productY[4] + productY[5] + productY[6] + productY[7];

        //Maybe also do sum using AVX ?
        //__m256 sumX =  _mm256_setzero_ps();
        //__m256 sumY =  _mm256_setzero_ps();

        // Store the results
        gradientTarget = (y+1) * width + (x+1) + channel;
        gradientX[gradientTarget] = gradX;
        gradientY[gradientTarget] = gradY;

        indexStart+=channels;
     }
   }

}



// Core SIMD Sobel for contiguous single-channel data (channels=1 layout).
// Processes 8 output pixels per inner iteration.
// Reads 16 bytes per row starting at x-1; bytes 0..9 are used (x-1..x+8),
// the trailing 6 bytes are harmless reads within the same allocation.
static void sobelXY8Bit_ch1_core(const unsigned char *pixels,
                                   unsigned int width, unsigned int height,
                                   float *gradientX, float *gradientY)
{
    for (unsigned int y = 1; y < height - 1; ++y)
    {
        const unsigned char *row0 = pixels + (y - 1) * width;
        const unsigned char *row1 = row0 + width;
        const unsigned char *row2 = row1 + width;

        float *gxp = gradientX + y * width + 1;
        float *gyp = gradientY + y * width + 1;

        unsigned int x = 1;

        for (; x + 8 < width; x += 8)
        {
            // One 16-byte load per row starting at x-1; bytes 0..9 supply lo/mid/hi.
            __m128i ld0 = _mm_loadu_si128((const __m128i *)(row0 + x - 1));
            __m128i ld1 = _mm_loadu_si128((const __m128i *)(row1 + x - 1));
            __m128i ld2 = _mm_loadu_si128((const __m128i *)(row2 + x - 1));

            // lo=pixels[x-1..x+6], mid=pixels[x..x+7], hi=pixels[x+1..x+8]
            __m128i r0lo  = _mm_cvtepu8_epi16(ld0);
            __m128i r0mid = _mm_cvtepu8_epi16(_mm_srli_si128(ld0, 1));
            __m128i r0hi  = _mm_cvtepu8_epi16(_mm_srli_si128(ld0, 2));
            __m128i r1lo  = _mm_cvtepu8_epi16(ld1);
            __m128i r1hi  = _mm_cvtepu8_epi16(_mm_srli_si128(ld1, 2));
            __m128i r2lo  = _mm_cvtepu8_epi16(ld2);
            __m128i r2mid = _mm_cvtepu8_epi16(_mm_srli_si128(ld2, 1));
            __m128i r2hi  = _mm_cvtepu8_epi16(_mm_srli_si128(ld2, 2));

            // Separable Sobel X: sumX = col_hi - col_lo, col = r0 + 2*r1 + r2
            __m128i col_lo = _mm_add_epi16(_mm_add_epi16(r0lo, r2lo), _mm_slli_epi16(r1lo, 1));
            __m128i col_hi = _mm_add_epi16(_mm_add_epi16(r0hi, r2hi), _mm_slli_epi16(r1hi, 1));
            __m128i sumX16 = _mm_sub_epi16(col_hi, col_lo);

            // Separable Sobel Y: sumY = smooth2 - smooth0, smooth = r_lo + 2*r_mid + r_hi
            __m128i sm0    = _mm_add_epi16(_mm_add_epi16(r0lo, r0hi), _mm_slli_epi16(r0mid, 1));
            __m128i sm2    = _mm_add_epi16(_mm_add_epi16(r2lo, r2hi), _mm_slli_epi16(r2mid, 1));
            __m128i sumY16 = _mm_sub_epi16(sm2, sm0);

            _mm256_storeu_ps(gxp, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sumX16)));
            _mm256_storeu_ps(gyp, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sumY16)));
            gxp += 8;
            gyp += 8;
        }

        // Scalar tail
        for (; x < width - 1; ++x)
        {
            int sX = -(int)row0[x-1] + (int)row0[x+1]
                     - 2*(int)row1[x-1] + 2*(int)row1[x+1]
                     - (int)row2[x-1] + (int)row2[x+1];
            int sY = -(int)row0[x-1] - 2*(int)row0[x] - (int)row0[x+1]
                     + (int)row2[x-1] + 2*(int)row2[x] + (int)row2[x+1];
            *gxp++ = (float)sX;
            *gyp++ = (float)sY;
        }
    }
}

// AVX2 dispatch for sobelXY8Bit.
// For channels==1 (contiguous): calls ch1_core directly.
// For channels>1 (interleaved, e.g. 73-channel heatmap): extracts the single
// target channel to a contiguous stack buffer, then calls ch1_core.
// Falls back (returns without touching gradients) if the image is too large for
// the stack buffer — the scalar path in sobelXY8Bit handles that case.
static void sobelXY8Bit_AVX2(struct Image *image, float *gradientX, float *gradientY, int channel)
{
    unsigned char *pixels  = (unsigned char *)image->pixels;
    unsigned int width     = image->width;
    unsigned int height    = image->height;
    unsigned int channels  = image->channels;

    if (channels == 1)
    {
        sobelXY8Bit_ch1_core(pixels, width, height, gradientX, gradientY);
        return;
    }

    // channels > 1: extract target channel to a contiguous temporary buffer.
    // 256x256 = 65536 bytes is the largest expected heatmap; guard at 256KB.
    unsigned int total = width * height;
    if (total > 262144) { return; }

    unsigned char *buf = (unsigned char *)alloca(total);

    // Deinterleave: collect every channels-th byte starting at offset `channel`.
    // Simple scalar loop; stride-N loads are unavoidable but the Sobel pass that
    // follows (stride-1) is now 8x faster, dominating the overall saving.
    const unsigned char *src = pixels + channel;
    for (unsigned int i = 0; i < total; ++i)
        buf[i] = src[i * channels];

    sobelXY8Bit_ch1_core(buf, width, height, gradientX, gradientY);
}


#endif // INTEL_OPTIMIZATIONS

#ifdef __cplusplus
}
#endif

#endif // SOBEL_H_INCLUDED
