#ifndef BILATERAL_H_INCLUDED
#define BILATERAL_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../codecs/image.h"

// Bilateral filter function
static void bilateralFilter8Bit(struct Image *image, int d, float sigmaColor, float sigmaSpace)
{
    int width    = image->width;
    int height   = image->height;
    int channels = image->channels;
    unsigned char *output = (unsigned char *)malloc(width * height * channels* sizeof(unsigned char));

    if (output!=0)
    {
    float color_coeff = -0.5 / (sigmaColor * sigmaColor);
    float space_coeff = -0.5 / (sigmaSpace * sigmaSpace);

    int radius = d / 2;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                float sum = 0.0;
                float norm_factor = 0.0;
                unsigned char center_val = getPixelValue8Bit(image, x, y, c);

                for (int ky = -radius; ky <= radius; ++ky)
                {
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        int neighbor_x = x + kx;
                        int neighbor_y = y + ky;
                        unsigned char neighbor_val = getPixelValue8Bit(image, neighbor_x, neighbor_y, c);

                        float space_weight = exp((kx * kx + ky * ky) * space_coeff);
                        float color_weight = exp((neighbor_val - center_val) * (neighbor_val - center_val) * color_coeff);
                        float weight = space_weight * color_weight;

                        sum += neighbor_val * weight;
                        norm_factor += weight;
                    }
                }

                output[(y * width + x) * channels + c] = (unsigned char)(sum / norm_factor);
            }
        }
    }

    // Copy output back to the original image
    unsigned char * pixels = image->pixels;
    for (int i = 0; i < width * height * channels; ++i)
        {
          pixels[i] = output[i];
        }

    free(output);
    }
}



// Bilateral filter function
static void bilateralFilter16Bit(struct Image *image, int d, float sigmaColor, float sigmaSpace)
{
    int width    = image->width;
    int height   = image->height;
    int channels = image->channels;
    unsigned short *output = (unsigned short *) malloc(width * height * channels * sizeof(unsigned short));

    if (output!=0)
    {
    float color_coeff = -0.5 / (sigmaColor * sigmaColor);
    float space_coeff = -0.5 / (sigmaSpace * sigmaSpace);

    int radius = d / 2;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                float sum = 0.0;
                float norm_factor = 0.0;
                unsigned short center_val = getPixelValue16Bit(image, x, y, c);

                for (int ky = -radius; ky <= radius; ++ky)
                {
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        int neighbor_x = x + kx;
                        int neighbor_y = y + ky;
                        unsigned short neighbor_val = getPixelValue16Bit(image, neighbor_x, neighbor_y, c);

                        float space_weight = exp((kx * kx + ky * ky) * space_coeff);
                        float color_weight = exp((neighbor_val - center_val) * (neighbor_val - center_val) * color_coeff);
                        float weight = space_weight * color_weight;

                        sum += neighbor_val * weight;
                        norm_factor += weight;
                    }
                }

                output[(y * width + x) * channels + c] = (unsigned short)(sum / norm_factor);
            }
        }
    }

    // Copy output back to the original image
    unsigned short * pixels = (unsigned short*) image->pixels;
    for (int i = 0; i < width * height * channels; ++i)
        {
          pixels[i] = output[i];
        }

    free(output);
    }
}

static void bilateralFilter(struct Image *image, int d, float sigmaColor, float sigmaSpace)
{
    if ( (image->channels==3) && (image->bitsperpixel==24) )
    {
      fprintf(stderr,"bilateralFilter 8bit for %u channels and %u bitsperpixel\n",image->channels,image->bitsperpixel);
      bilateralFilter8Bit(image, d, sigmaColor, sigmaSpace);
    } else
    if ( (image->channels==3) && (image->bitsperpixel==48) )
    {
      fprintf(stderr,"bilateralFilter 16bit for %u channels and %u bitsperpixel\n",image->channels,image->bitsperpixel);
      bilateralFilter16Bit(image, d, sigmaColor, sigmaSpace);
    } else
    {
        fprintf(stderr,"Cannot compute bilateralFilter for %u channels and %u bitsperpixel\n",image->channels,image->bitsperpixel);
    }
}

#ifdef __cplusplus
}
#endif

#endif // BILATERAL_H_INCLUDED
