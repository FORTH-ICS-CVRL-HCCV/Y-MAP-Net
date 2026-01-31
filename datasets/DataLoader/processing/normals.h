#ifndef NORMALS_H_INCLUDED
#define NORMALS_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../codecs/image.h"
#include "getpixels.h"
#include "sobel.h"


#include "AVX2/normals.h"

#define F_EPSILON 1e-8
#define NORMALS_USE_SQRT_APPROXIMATION 0 //<- this improves batch performance ~0.20% but quantizes
#define NORMALS_SQRT_LOOKUP_SIZE 325 // max value 18^2 + 18^2 + 1^2 = 325


#if NORMALS_USE_SQRT_APPROXIMATION
// Precompute sqrt lookup table
static char  sqrtLookupInitialized = 0;
static float sqrtLookup[NORMALS_SQRT_LOOKUP_SIZE]={0};

static void initializeSqrtLookupTable()
{
   //float epsilon = 1e-8;
   for (int i = 0; i < NORMALS_SQRT_LOOKUP_SIZE; ++i)
    {
        sqrtLookup[i] = sqrtf(i) + F_EPSILON;
    }
    sqrtLookupInitialized=1;
}
#endif // NORMALS_USE_SQRT_APPROXIMATION



// Function to compute normals from grayscale image
static void computeNormals8Bit(struct Image *image, struct Image *normals)
{
    int width  = image->width;
    int height = image->height;

    unsigned char * pixels = (unsigned char*) normals->pixels;

    float *gradientX  = (float *) malloc(width * height * sizeof(float));
    if (gradientX!=0)
    {
     float *gradientY = (float *) malloc(width * height * sizeof(float));
     if (gradientY!=0)
     {
      //sobelX(image, gradientX, 0);
      //sobelY(image, gradientY, 0);
      sobelXY8Bit(image, gradientX, gradientY,0);

      float epsilon = 1e-8;

      for (int y = 0; y < height; ++y)
      {
        for (int x = 0; x < width; ++x)
        {
            int idx  = y * width + x;
            float dx = gradientX[idx];
            float dy = gradientY[idx];
            float dz = 1.0;

            float norm = sqrt(dx * dx + dy * dy + dz * dz);

            float nx = (float) dx / (norm + epsilon);
            float ny = (float) dy / (norm + epsilon);
            float nz = (float) dz / (norm + epsilon);

            pixels[(idx * 3) + 0] = (unsigned char) (((nx + 1.0) / 2.0) * 255);
            pixels[(idx * 3) + 1] = (unsigned char) (((ny + 1.0) / 2.0) * 255);
            pixels[(idx * 3) + 2] = (unsigned char) (((nz + 1.0) / 2.0) * 255);
        }
     }

     free(gradientY);
     }
    free(gradientX);
    }
}

// Function to compute normals from grayscale image
static void computeNormals16Bit(struct Image *image, struct Image *normals)
{
    int width  = image->width;
    int height = image->height;

    unsigned short * pixels = (unsigned short*) normals->pixels;

    float *gradientX = (float *) malloc(width * height * sizeof(float));
    if (gradientX!=0)
    {
     float *gradientY = (float *) malloc(width * height * sizeof(float));
     if (gradientY!=0)
     {
      sobelX(image, gradientX, 0);
      sobelY(image, gradientY, 0);

      float epsilon = 1e-8;

      for (int y = 0; y < height; ++y)
      {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            float dx = gradientX[idx];
            float dy = gradientY[idx];
            float dz = 1.0;

            float norm = sqrt(dx * dx + dy * dy + dz * dz);

            float nx = (float) dx / (norm + epsilon);
            float ny = (float) dy / (norm + epsilon);
            float nz = (float) dz / (norm + epsilon);

            pixels[(idx * 3) + 0] = (unsigned short) (((nx + 1.0) / 2.0) * 65535);
            pixels[(idx * 3) + 1] = (unsigned short) (((ny + 1.0) / 2.0) * 65535);
            pixels[(idx * 3) + 2] = (unsigned short) (((nz + 1.0) / 2.0) * 65535);
        }
     }

     free(gradientY);
     }
    free(gradientX);
    }
}

static void computeNormals(struct Image *image, struct Image *normals)
{
    if ( (image->channels==1) && (image->bitsperpixel==8) )
    {
      return computeNormals8Bit(image, normals);
    } else
    if ( (image->channels==1) && (image->bitsperpixel==16) )
    {
      return computeNormals16Bit(image, normals);
    } else
    {
        fprintf(stderr,"Cannot compute normals for %u channels and %u bitsperpixel\n",image->channels,image->bitsperpixel);
        abort();
    }
}


//This does interpolation, but only uses R channel to do 3 times less calculations
static void computeNormalsOnHeatmaps8Bit_SimpleFull(signed char * heatmapPixels, unsigned int heatmapWidth, unsigned int heatmapHeight,unsigned int heatmapChannels,
                                         unsigned int depthMapSourceChannel,unsigned int normalOutputChannel,
                                         float *gradientX,float *gradientY, unsigned int borderX, unsigned int borderY)
{
   if ( (gradientX==0) || (gradientY==0) )
   {
       fprintf(stderr,"Cannot computeNormalsOnHeatmaps8Bit without gradient buffers\n");
       abort();
   }

   if (heatmapPixels==0)
   {
       fprintf(stderr,"Cannot computeNormalsOnHeatmaps8Bit without heatmapPixels\n");
       abort();
   }

   #if NORMALS_USE_SQRT_APPROXIMATION
   if (!sqrtLookupInitialized)
      { initializeSqrtLookupTable(); }
   #endif // NORMALS_USE_SQRT_APPROXIMATION

   struct Image heatmapWrapper = {0};
   heatmapWrapper.pixels       = (unsigned char*) heatmapPixels;
   heatmapWrapper.width        = heatmapWidth;
   heatmapWrapper.height       = heatmapHeight;
   heatmapWrapper.channels     = heatmapChannels;

   int width  = heatmapWidth;
   int height = heatmapHeight;

   //SobelXY8BitLoop uses AVX2
   //#if INTEL_OPTIMIZATIONS
   //  sobelXY8BitLoop(&heatmapWrapper, gradientX, gradientY, depthMapSourceChannel);
   //#else
     //This is optimized for regular non SIMD operations
     sobelXY8Bit(&heatmapWrapper, gradientX, gradientY, depthMapSourceChannel);
   //#endif // INTEL_OPTIMIZATIONS

   //unsigned long gradientIndex = 0;
   unsigned long heatmapIndex  = normalOutputChannel;
   float dx,dy,value,norm,nx,ny,nz; //dz, not used

   signed char * heatmapPixelsPtr = heatmapPixels + heatmapIndex;
   float * gradientXPtr = gradientX;
   float * gradientYPtr = gradientY;

   float * gradientXPtrLimit = gradientX + (width*height);

   while (gradientXPtr<gradientXPtrLimit)
      {
            //Grab dX and dY from gradient memory
            dx = *gradientXPtr;
            gradientXPtr++;
            dy = *gradientYPtr;
            gradientYPtr++;
            //dz = 1.0;

            value = (dx * dx) + (dy * dy) + 1.0; //(dz * dz);

            #if NORMALS_USE_SQRT_APPROXIMATION
            if (value<NORMALS_SQRT_LOOKUP_SIZE)
            { norm = sqrtLookup[(unsigned int) value]; } else
            { norm = (float) sqrtf(value) + F_EPSILON; }
            #else
            norm = (float) sqrtf(value) + F_EPSILON;
            #endif // NORMALS_USE_SQRT_APPROXIMATION

            nz = (float) 120.0 / norm; // Map from [-1, 1] to [-120, 120]
            nx = (float) nz * dx;      // Map from [-1, 1] to [-120, 120]
            ny = (float) nz * dy;      // Map from [-1, 1] to [-120, 120]
            //nx = (float) 120.0 * ((float) dx / norm); // Map from [-1, 1] to [-120, 120] (expensive but more clear)
            //ny = (float) 120.0 * ((float) dy / norm); // Map from [-1, 1] to [-120, 120] (expensive but more clear)

            //heatmapIndex = (y * width * heatmapChannels) + (x * heatmapChannels) + normalOutputChannel;
            *heatmapPixelsPtr  = (signed char) nx; // Heatmap 18
            ++heatmapPixelsPtr;
            *heatmapPixelsPtr  = (signed char) ny; // Heatmap 19
            ++heatmapPixelsPtr;
            *heatmapPixelsPtr  = (signed char) nz; // Heatmap 20
            heatmapPixelsPtr += heatmapChannels-2;
     }
}

//This does interpolation, but only uses R channel to do 3 times less calculations
static void computeNormalsOnHeatmaps8Bit_Simple(signed char * heatmapPixels, unsigned int heatmapWidth, unsigned int heatmapHeight,unsigned int heatmapChannels,
                                         unsigned int depthMapSourceChannel,unsigned int normalOutputChannel,
                                         float *gradientX,float *gradientY, unsigned int borderX, unsigned int borderY)
{
   if ( (gradientX==0) || (gradientY==0) )
   {
       fprintf(stderr,"Cannot computeNormalsOnHeatmaps8Bit without gradient buffers\n");
       abort();
   }

   if (heatmapPixels==0)
   {
       fprintf(stderr,"Cannot computeNormalsOnHeatmaps8Bit without heatmapPixels\n");
       abort();
   }

   #if NORMALS_USE_SQRT_APPROXIMATION
   if (!sqrtLookupInitialized)
      { initializeSqrtLookupTable(); }
   #endif // NORMALS_USE_SQRT_APPROXIMATION

   struct Image heatmapWrapper = {0};
   heatmapWrapper.pixels       = (unsigned char*) heatmapPixels;
   heatmapWrapper.width        = heatmapWidth;
   heatmapWrapper.height       = heatmapHeight;
   heatmapWrapper.channels     = heatmapChannels;

   int width  = heatmapWidth;
   int height = heatmapHeight;

   //SobelXY8BitLoop uses AVX2
   //#if INTEL_OPTIMIZATIONS
   //  sobelXY8BitLoop(&heatmapWrapper, gradientX, gradientY, depthMapSourceChannel);
   //#else
     //This is optimized for regular non SIMD operations
     sobelXY8Bit(&heatmapWrapper, gradientX, gradientY, depthMapSourceChannel);
   //#endif // INTEL_OPTIMIZATIONS

   //unsigned long gradientIndex = 0;
   unsigned long heatmapIndex  = normalOutputChannel;
   float dx,dy,value,norm,nx,ny,nz; //dz, not used

   unsigned char* heatmapPixelsInitial = (unsigned char*) heatmapPixels;
   //float * gradientXPtrLimit = gradientX + (width*height);

   //signed char * heatmapPixelsPtr = heatmapPixelsInitial + heatmapIndex;

   //Kill one border pixel that is going to be an edge because of the border
   //If borders are 0 then this does nothing
   borderX += (borderX!=0);
   borderY += (borderY!=0);
   //------------------------------------------------------------------------

   for (int y=borderY; y<height-borderY; y++)
   {
    signed char * heatmapPixelsPtr = heatmapPixelsInitial + heatmapIndex + (y * heatmapWidth * heatmapChannels) + (borderX * heatmapChannels);
    float * gradientXPtr = gradientX + (width * y) + borderX;
    float * gradientYPtr = gradientY + (width * y) + borderX;

    for (int x=borderX; x<width-borderX; x++)
    {
    // while (gradientXPtr<gradientXPtrLimit)
      {
            //Grab dX and dY from gradient memory
            dx = *gradientXPtr;
            gradientXPtr++;
            dy = *gradientYPtr;
            gradientYPtr++;
            //dz = 1.0;

            value = (dx * dx) + (dy * dy) + 1.0; //(dz * dz);

            #if NORMALS_USE_SQRT_APPROXIMATION
            if (value<NORMALS_SQRT_LOOKUP_SIZE)
            { norm = sqrtLookup[(unsigned int) value]; } else
            { norm = (float) sqrtf(value) + F_EPSILON; }
            #else
            norm = (float) sqrtf(value) + F_EPSILON;
            #endif // NORMALS_USE_SQRT_APPROXIMATION

            nz = (float) 120.0 / norm; // Map from [-1, 1] to [-120, 120]
            nx = (float) nz * dx;      // Map from [-1, 1] to [-120, 120]
            ny = (float) nz * dy;      // Map from [-1, 1] to [-120, 120]
            //nx = (float) 120.0 * ((float) dx / norm); // Map from [-1, 1] to [-120, 120] (expensive but more clear)
            //ny = (float) 120.0 * ((float) dy / norm); // Map from [-1, 1] to [-120, 120] (expensive but more clear)

            //heatmapIndex = (y * width * heatmapChannels) + (x * heatmapChannels) + normalOutputChannel;
            *heatmapPixelsPtr  = (signed char) nx; // Heatmap 18
            ++heatmapPixelsPtr;
            *heatmapPixelsPtr  = (signed char) ny; // Heatmap 19
            ++heatmapPixelsPtr;
            *heatmapPixelsPtr  = (signed char) nz; // Heatmap 20
            heatmapPixelsPtr += heatmapChannels-2;
      }
     }
   }
}


static void computeNormalsOnHeatmaps8Bit(signed char * heatmapPixels, unsigned int heatmapWidth, unsigned int heatmapHeight,unsigned int heatmapChannels,
                                         unsigned int depthMapSourceChannel,unsigned int normalOutputChannel,
                                         float *gradientX,float *gradientY, unsigned int borderX, unsigned int borderY)
{
 //#if INTEL_OPTIMIZATIONS
 //  computeNormalsOnHeatmaps8Bit_AVX2(heatmapPixels,heatmapWidth,heatmapHeight,heatmapChannels,depthMapSourceChannel,normalOutputChannel,gradientX,gradientY);
 //This doesn't work yet
 //#else
   computeNormalsOnHeatmaps8Bit_Simple(heatmapPixels,heatmapWidth,heatmapHeight,heatmapChannels,depthMapSourceChannel,normalOutputChannel,gradientX,gradientY,borderX,borderY);
 //#endif // INTEL_OPTIMIZATIONS
}


//This does interpolation, but only uses R channel to do 3 times less calculations
static void computeNormalsOnHeatmaps8BitStable(signed char * heatmapPixels, unsigned int heatmapWidth, unsigned int heatmapHeight,unsigned int heatmapChannels,
                                         unsigned int depthMapSourceChannel,unsigned int normalOutputChannel,
                                         float *gradientX,float *gradientY)
{
   if ( (gradientX==0) || (gradientY==0) )
   {
       fprintf(stderr,"Cannot computeNormalsOnHeatmaps8Bit without gradient buffers\n");
       abort();
       return;
   }

   struct Image heatmapWrapper={0};
   heatmapWrapper.pixels   = (unsigned char*) heatmapPixels;
   heatmapWrapper.width    = heatmapWidth;
   heatmapWrapper.height   = heatmapHeight;
   heatmapWrapper.channels = heatmapChannels;

   int width  = heatmapWidth;
   int height = heatmapHeight;


   sobelXY8Bit(&heatmapWrapper, gradientX, gradientY,depthMapSourceChannel);

   int idx ;
   float epsilon = 1e-8;
   float dx,dy,dz,norm,nx,ny,nz;

   for (int y = 0; y < height; ++y)
      {
       for (int x = 0; x < width; ++x)
        {
            idx = (y * width) + x;
            dx = gradientX[idx];
            dy = gradientY[idx];
            dz = 1.0;

            norm = sqrt(dx * dx + dy * dy + dz * dz);

            nx = dx / (norm + epsilon);
            ny = dy / (norm + epsilon);
            nz = dz / (norm + epsilon);

            idx = (y * width * heatmapChannels) + (x * heatmapChannels) + normalOutputChannel;
            heatmapPixels[idx + 0] = (signed char)(nx * 127); // Map from [-1, 1] to [-127, 127]
            heatmapPixels[idx + 1] = (signed char)(ny * 127); // Map from [-1, 1] to [-127, 127]
            heatmapPixels[idx + 2] = (signed char)(nz * 127); // Map from [-1, 1] to [-127, 127]
        }
     }
}



//This does interpolation, but only uses R channel to do 3 times less calculations
static void computeNormalsOnHeatmaps16Bit(unsigned char * heatmapPixels, unsigned int heatmapWidth, unsigned int heatmapHeight,unsigned int heatmapChannels,
                                          unsigned int depthMapSourceChannel, unsigned int normalOutputChannel,
                                          float *gradientX,float *gradientY)
{
   if ( (gradientX==0) || (gradientY==0) )
   {
       fprintf(stderr,"Cannot computeNormalsOnHeatmaps without gradient buffers\n");
       return;
   }

   struct Image heatmapWrapper={0};
   heatmapWrapper.pixels   = heatmapPixels;
   heatmapWrapper.width    = heatmapWidth;
   heatmapWrapper.height   = heatmapHeight;
   heatmapWrapper.channels = heatmapChannels;

   int width  = heatmapWidth;
   int height = heatmapHeight;

  int sobelXKernel[3][3] =
    {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
  int sobelYKernel[3][3] =
    {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

   sobel16BitWithChannelsAddressedAs8Bit(&heatmapWrapper, gradientX, depthMapSourceChannel, sobelXKernel);
   sobel16BitWithChannelsAddressedAs8Bit(&heatmapWrapper, gradientY, depthMapSourceChannel, sobelYKernel);

   float epsilon = 1e-8;

   for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = (y * width) + x;
            float dx = gradientX[idx];
            float dy = gradientY[idx];
            float dz = 1.0;

            float norm = sqrt(dx * dx + dy * dy + dz * dz);

            float nx = dx / (norm + epsilon);
            float ny = dy / (norm + epsilon);
            float nz = dz / (norm + epsilon);

            idx = (y * width * heatmapChannels) + (x * heatmapChannels) + normalOutputChannel;

            // Map from [-1, 1] to [0, 65535]
            unsigned short * nXPtr = (unsigned short *) (heatmapPixels + idx + 0);
            unsigned short * nYPtr = (unsigned short *) (heatmapPixels + idx + 2);
            unsigned short * nZPtr = (unsigned short *) (heatmapPixels + idx + 4);
            *nXPtr = (unsigned short)((nx + 1.0) * 65535); // Map from [-1, 1] to [0, 65535]
            *nYPtr = (unsigned short)((ny + 1.0) * 65535); // Map from [-1, 1] to [0, 65535]
            *nZPtr = (unsigned short)((nz + 1.0) * 65535); // Map from [-1, 1] to [0, 65535]
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif // NORMALS_H_INCLUDED
