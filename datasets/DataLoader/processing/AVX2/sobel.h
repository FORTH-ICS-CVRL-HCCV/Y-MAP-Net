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



#endif // INTEL_OPTIMIZATIONS

#ifdef __cplusplus
}
#endif

#endif // SOBEL_H_INCLUDED
