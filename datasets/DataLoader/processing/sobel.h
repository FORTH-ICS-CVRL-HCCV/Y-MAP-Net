#ifndef SOBEL_H_INCLUDED
#define SOBEL_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../codecs/image.h"
#include "AVX2/sobel.h"

static void sobel8BitSlower(struct Image *image, float *gradient, int channel, int sobelKernel[3][3])
{
    int width  = image->width;
    int height = image->height;

    for (int y = 1; y < height-1; ++y)
    {
        for (int x = 1; x < width-1; ++x)
        {
            float sum = 0.0;
            for (int ky = -1; ky <= 1; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    int pixelValue = getPixelValue8Bit(image, x + kx, y + ky, channel);
                    sum += (float) ( pixelValue * sobelKernel[ky + 1][kx + 1] );
                }
            }
            gradient[y * width + x] = sum; //  / 9; // We divide by 9 to normalize for 3x3 kernel (this has edge cases on borders)
        }
    }
    //TODO: Normalize here ?
}

// Function to apply Sobel filter in X direction
static void sobel8Bit(struct Image *image, float *gradient, int channel, int sobelKernel[3][3])
{
    unsigned char * pixels = (unsigned char* ) image->pixels;
    int width    = image->width;
    int height   = image->height;
    int channels = image->channels;

    for (int y = 1; y < height-1; ++y)
    {
        for (int x = 1; x < width-1; ++x)
        {
            float sum = 0.0;
            for (int ky = -1; ky <= 1; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    int pixelValue = pixels[((y+ky) * width + (x+kx)) * channels + channel];
                    //int pixelValue = getPixelValue8Bit(image, x + kx, y + ky, channel);
                    sum += (float) ( pixelValue * sobelKernel[ky + 1][kx + 1] );
                }
            }
            gradient[y * width + x] = sum; //  / 9; // We can divide by 9 to normalize for 3x3 kernel (this has edge cases on borders)
        }
    }
    //TODO: Normalize here ?
}

static void sobel8Bit2Way(struct Image *image, float *gradientX, int sobelKernelX[3][3], float *gradientY, int sobelKernelY[3][3], int channel)
{
    unsigned char * pixels = (unsigned char* ) image->pixels;
    int width    = image->width;
    int height   = image->height;
    int channels = image->channels;

    int sobX,sobY,pixelValue,yIndex;

    //#pragma omp parallel for
    for (int y = 1; y < height-1; ++y)
    {
        for (int x = 1; x < width-1; ++x)
        {
            int sumX = 0.0;
            int sumY = 0.0;
            for (int ky = -1; ky <= 1; ++ky)
            {
                yIndex = (y + ky) * width;

                for (int kx = -1; kx <= 1; ++kx)
                {
                    sobX = sobelKernelX[ky + 1][kx + 1];
                    sobY = sobelKernelY[ky + 1][kx + 1];

                    pixelValue = pixels[(yIndex + (x+kx)) * channels + channel];

                    sumX += (pixelValue * sobX);
                    sumY += (pixelValue * sobY);
                }
            }
            gradientX[y * width + x] = (float) sumX; //  / 9; // We can divide by 9 to normalize for 3x3 kernel (this has edge cases on borders)
            gradientY[y * width + x] = (float) sumY; //  / 9; // We can divide by 9 to normalize for 3x3 kernel (this has edge cases on borders)
        }
    }
    //TODO: Normalize here ?
}


static void sobelXY8BitLoop(struct Image *image, float *gradientX, float *gradientY,int channel)
{
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


   #if INTEL_OPTIMIZATIONS
     sobel8Bit2Way_ignore_center_AVX2(image, gradientX, sobelXKernel, gradientY, sobelYKernel, channel);
   #else
     sobel8Bit2Way(image, gradientX, sobelXKernel, gradientY, sobelYKernel, channel);
   #endif // INTEL_OPTIMIZATIONS
}


//This function makes batch preparation work 20% faster..!
//I think it works 5x faster than sobelXY8BitLoop
static void sobelXY8Bit(struct Image *image, float *gradientX, float *gradientY,int channel)
{
    /*
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

     We have the following elements
       M11  M12  M13
       M21  M22  M23
       M31  M32  M33
    */

    unsigned char * pixels = (unsigned char* ) image->pixels;
    unsigned int width     = image->width;
    unsigned int height    = image->height;
    unsigned int channels  = image->channels;

    if ( (pixels==0) || (width==0) || (height==0) || (gradientX==0) || (gradientY==0) )
    {
        fprintf(stderr,"Cannot compute sobel with incorrect input\n");
        return ;
    }

    signed int sumX,sumY;

    unsigned int horizontalLine = (width * channels);
    unsigned char * start, * M11,* M12,* M13,* M21,* M23,* M31,* M32,* M33; //* M22, not used
    unsigned int start_x = 1;

    //fprintf(stderr,"sobelXY8Bit %u x %u \n",width,height);
    //#pragma omp parallel for private(start, M11, M12, M13, M21, M22, M23, M31, M32, M33)
    for (unsigned int y = 1; y < height-1; ++y)
    {
       //Calculate Matrix elements
       start = pixels + ((y-1) * width) * channels + channel;

       M11 = start;
       M12 = M11 + channels;
       M13 = M12 + channels;

       M21 = M11 + horizontalLine;
       //M22 = M21 + channels; //NOOP
       M23 = M21 + channels + channels;

       M31 = M21 + horizontalLine;
       M32 = M31 + channels;
       M33 = M32 + channels;

       unsigned int targetAddress = y * width + start_x;
       float *gradientXPtr = gradientX+targetAddress;
       float *gradientYPtr = gradientY+targetAddress;

       for (unsigned int x = start_x; x < width-1; ++x)
        {
             //Make sure sums are clean
             sumX = 0;
             sumY = 0;

             //M11 is the same calculation for both
             sumX -= *M11;
             sumY -= *M11;

             //M12 only has Y
             sumY -= 2 * *M12;

             //M13 has inverse calculation
             sumX += *M13;
             sumY -= *M13;

             //M21 only has X
             sumX -= 2* *M21;

             //M22 has nothing
             //NOOP

             //M23 only has X
             sumX += 2* *M23;

             //M31 has inverse calculation
             sumX -= *M31;
             sumY += *M31;

             //M32 only has Y
             sumY += 2 * *M32;

             //M33 is the same calculation for both
             sumX += *M33;
             sumY += *M33;

             //Slide convolution
             M11+=channels;
             M12+=channels;
             M13+=channels;
             M21+=channels;
             //M22+=channels;  //NOOP
             M23+=channels;
             M31+=channels;
             M32+=channels;
             M33+=channels;

             //Store output
             *gradientXPtr = (float) sumX;
             ++gradientXPtr;
             *gradientYPtr = (float) sumY;
             ++gradientYPtr;
        }
    }
    //fprintf(stderr,"sobelXY8Bit survived \n",width,height);
}



static void sobelX8Bit(struct Image *image, float *gradientX,int channel)
{
    int sobelXKernel[3][3] =
    {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
   sobel8Bit(image,gradientX,channel,sobelXKernel);
}
static void sobelY8Bit(struct Image *image, float *gradientY,int channel)
{
    int sobelYKernel[3][3] =
    {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
   sobel8Bit(image,gradientY,channel,sobelYKernel);
}



// Function to apply Sobel filter in X direction
static void sobel16Bit(struct Image *image, float *gradient, int channel, int sobelKernel[3][3])
{
    int width  = image->width;
    int height = image->height;

    for (int y = 1; y < height-1; ++y)
    {
        for (int x = 1; x < width-1; ++x)
        {
            float sum = 0.0;
            for (int ky = -1; ky <= 1; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    int pixelValue = getPixelValue16Bit(image, x + kx, y + ky, channel);
                    sum += (float) (pixelValue * sobelKernel[ky + 1][kx + 1]);
                }
            }
            gradient[y * width + x] = sum; //  / 9; // We divide by 9 to normalize for 3x3 kernel (this has edge cases on borders)
        }
    }
    //TODO: Normalize here ?
}
static void sobelX16Bit(struct Image *image, float *gradientX,int channel)
{
    int sobelXKernel[3][3] =
    {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
   sobel16Bit(image,gradientX,channel,sobelXKernel);
}
static void sobelY16Bit(struct Image *image, float *gradientY,int channel)
{
    int sobelYKernel[3][3] =
    {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
   sobel16Bit(image,gradientY,channel,sobelYKernel);
}



static void sobelX(struct Image *image, float *gradientX,int channel)
{
    if ( (image->channels==1) && (image->bitsperpixel==8) )
    {
      fprintf(stderr,"sobelX 8bit for %u channels and %u bitsperpixel\n",image->channels,image->bitsperpixel);
      return sobelX8Bit(image,gradientX,channel);
    } else
    if ( (image->channels==1) && (image->bitsperpixel==16) )
    {
      fprintf(stderr,"sobelX 16bit for %u channels and %u bitsperpixel\n",image->channels,image->bitsperpixel);
      return sobelX16Bit(image,gradientX,channel);
    } else
    {
        fprintf(stderr,"Cannot compute sobelX for %u channels and %u bitsperpixel\n",image->channels,image->bitsperpixel);
    }
}

static void sobelY(struct Image *image, float *gradientY,int channel)
{
    if ( (image->channels==1) && (image->bitsperpixel==8) )
    {
      fprintf(stderr,"sobelY 8bit for %u channels and %u bitsperpixel\n",image->channels,image->bitsperpixel);
      return sobelY8Bit(image,gradientY,channel);
    } else
    if ( (image->channels==1) && (image->bitsperpixel==16) )
    {
      fprintf(stderr,"sobelY 16bit for %u channels and %u bitsperpixel\n",image->channels,image->bitsperpixel);
      return sobelY16Bit(image,gradientY,channel);
    } else
    {
        fprintf(stderr,"Cannot compute sobelY for %u channels and %u bitsperpixel\n",image->channels,image->bitsperpixel);
    }
}



// Function to apply Sobel filter in X direction
static void sobel16BitWithChannelsAddressedAs8Bit(struct Image *image, float *gradient, int channel, int sobelKernel[3][3])
{
    int width  = image->width;
    int height = image->height;

    for (int y = 1; y < height-1; ++y)
    {
        for (int x = 1; x < width-1; ++x)
        {
            float sum = 0.0;
            for (int ky = -1; ky <= 1; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    int pixelValue = (int) getPixelValue16BitWithChannelsAddressedAs8Bit(image, x + kx, y + ky, channel);
                    sum += (float) (pixelValue * sobelKernel[ky + 1][kx + 1]);
                }
            }
            gradient[y * width + x] = sum; // 9; //// We divide by 9 to normalize for 3x3 kernel (this has edge cases on borders)
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif // SOBEL_H_INCLUDED
