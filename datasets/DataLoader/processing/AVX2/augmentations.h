#ifndef AUGMENTATIONS_AVX2_H_INCLUDED
#define AUGMENTATIONS_AVX2_H_INCLUDED

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
#include <stdalign.h>
#include <stdint.h>

//RGB arguments might be flipped!
static void adjustBrightnessContrast_AVX2(struct Image *image,
                                     float brightnessR, float contrastR,
                                     float brightnessG, float contrastG,
                                     float brightnessB, float contrastB,
                                     float borderX, float borderY)
{

    uint8_t *pixelsInput = image->pixels;
    int width = image->width;
    int height = image->height;
    int channels = image->channels;

    // Boundary handling
    signed int start_x = (signed int) borderX;
    signed int start_y = (signed int) borderY;
    signed int end_x = width  - start_x;
    signed int end_y = height - start_y;

    if (start_x<0)    { start_x = 0;    }
    if (start_y<0)    { start_y = 0;    }
    if (end_x>width)  { end_x = width;  }
    if (end_y>height) { end_y = height; }


    __m256 zero     = _mm256_setzero_ps();
    __m256 max_val  = _mm256_set1_ps(255.0f);

    __m256 channel_brightness[3];
    // Depending on offset we will encounter one of these three situations for our 8 values
    channel_brightness[2] = _mm256_set_ps( brightnessR, brightnessG, brightnessB, brightnessR, brightnessG, brightnessB, brightnessR, brightnessG ); //Offset 0
    channel_brightness[1] = _mm256_set_ps( brightnessG, brightnessB, brightnessR, brightnessG, brightnessB, brightnessR, brightnessG, brightnessB ); //Offset 1
    channel_brightness[0] = _mm256_set_ps( brightnessB, brightnessR, brightnessG, brightnessB, brightnessR, brightnessG, brightnessB, brightnessR ); //Offset 2

    __m256 channel_contrast[3];
    // Depending on offset we will encounter one of these three situations for our 8 values
    channel_contrast[2] = _mm256_set_ps(contrastR, contrastG, contrastB, contrastR, contrastG, contrastB, contrastR, contrastG); //Offset 0
    channel_contrast[1] = _mm256_set_ps(contrastG, contrastB, contrastR, contrastG, contrastB, contrastR, contrastG, contrastB); //Offset 1
    channel_contrast[0] = _mm256_set_ps(contrastB, contrastR, contrastG, contrastB, contrastR, contrastG, contrastB, contrastR); //Offset 2


    for (int y = start_y; y < end_y; ++y)
    {
        int offset = 0;
        int indexStart = (y * width + start_x) * channels;
        //for (int x = start_x; x < end_x; x+=3)
        for (int index = indexStart;  index<indexStart+(end_x-start_x)*channels; index+=8)
        {
            // Load 3 pixels (RGB)              //   Offset
            __m256 pixels_f = _mm256_set_ps(    //  0   1    2
                (float)pixelsInput[index + 0],  // R0   G0   B0
                (float)pixelsInput[index + 1],  // G0   B0   R1
                (float)pixelsInput[index + 2],  // B0   R1   G1
                (float)pixelsInput[index + 3],  // R1   G1   B1
                (float)pixelsInput[index + 4],  // G1   B1   R2
                (float)pixelsInput[index + 5],  // B1   R2   G2
                (float)pixelsInput[index + 6],  // R2   G2   B2
                (float)pixelsInput[index + 7]   // G2   B2   R3
            );

            // Apply contrast first, then brightness
            __m256 adjusted_pixels = _mm256_add_ps( _mm256_mul_ps(pixels_f, channel_contrast[offset%3]), channel_brightness[offset%3] );

            // Clamp to [0, 255]
            adjusted_pixels = _mm256_min_ps( _mm256_max_ps(adjusted_pixels, zero), max_val );

            //                                                          Offset
            // Store back adjusted pixel values                    // 0    1    2
            pixelsInput[index]     = (uint8_t)adjusted_pixels[7];  // R    G    B
            pixelsInput[index + 1] = (uint8_t)adjusted_pixels[6];  // G    B    R
            pixelsInput[index + 2] = (uint8_t)adjusted_pixels[5];  // B    R    G
            pixelsInput[index + 3] = (uint8_t)adjusted_pixels[4];  // R    G    B
            pixelsInput[index + 4] = (uint8_t)adjusted_pixels[3];  // G    B    R
            pixelsInput[index + 5] = (uint8_t)adjusted_pixels[2];  // B    R    G
            pixelsInput[index + 6] = (uint8_t)adjusted_pixels[1];  // R    G    B
            pixelsInput[index + 7] = (uint8_t)adjusted_pixels[0];  // G    B    R
            //                                                        Next Group
            //                                     Starts with        B    R    G

            //index+=8;
            offset+=1;
        }
    }
}

#endif // INTEL_OPTIMIZATIONS


#ifdef __cplusplus
}
#endif

#endif // AUGMENTATIONS_H_INCLUDED
