#ifndef CONVERSIONS_AVX2_H_INCLUDED
#define CONVERSIONS_AVX2_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../codecs/image.h"


#if INTEL_OPTIMIZATIONS
#include <immintrin.h>

static void swap16bitEndianness_AVX2(struct Image *img16)
{
    if (img16->bitsperpixel / img16->channels != 16)
    {
        fprintf(stderr, "Invalid image format: expected 16 bits per pixel.\n");
        return;
    }

    unsigned int width = img16->width;
    unsigned int height = img16->height;
    unsigned int channels = img16->channels;
    unsigned short *pixels16 = (unsigned short *)img16->pixels;

    unsigned int totalPixels = width * height * channels;

    unsigned int i = 0;
   // __m256i mask = _mm256_set1_epi16(0xFF00); // Mask to isolate high and low bytes

    for (; i + 15 < totalPixels; i += 16)
    {
        // Load 16 pixels into an AVX2 register
        __m256i pixelBlock = _mm256_loadu_si256((__m256i *)&pixels16[i]);

        // Swap bytes
        __m256i swapped = _mm256_or_si256(
            _mm256_srli_epi16(pixelBlock, 8),  // Shift high byte to low
            _mm256_slli_epi16(pixelBlock, 8)  // Shift low byte to high
        );

        // Store swapped values back
        _mm256_storeu_si256((__m256i *)&pixels16[i], swapped);
    }

    // Handle any remaining pixels
    for (; i < totalPixels; ++i)
    {
        unsigned short pixel16 = pixels16[i];
        pixels16[i] = (pixel16 >> 8) | (pixel16 << 8);
    }
}
#endif // INTEL_OPTIMIZATIONS





#ifdef __cplusplus
}
#endif

#endif // CONVERSIONS_H_INCLUDED


