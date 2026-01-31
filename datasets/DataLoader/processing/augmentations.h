#ifndef AUGMENTATIONS_H_INCLUDED
#define AUGMENTATIONS_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>

#include "../codecs/image.h"
#include "AVX2/augmentations.h"

#define PI 3.14159265

/*
TODO :
add here
Horizontal Flip
Vertical Flip
Brightness/Contrast change ✓
Pixel Dropout
Blur
Sharpen
Quantize
Rotate

Add empty image
Add completely noise image
*/

// Function to generate a random number between min and max (inclusive)
/* Simple version
static int getRandomNumber(int minVal, int maxVal)
{
    return rand() % (maxVal - minVal + 1) + minVal;
}*/

static int getRandomNumber(int minVal, int maxVal)
{
    unsigned int range = maxVal - minVal + 1;
    // Use 64-bit multiplication to avoid bias
    unsigned int r = ((uint64_t) rand() * range) >> 31;
    return (int)(r + minVal);
}

/* Old version
static float getRandomFloat(float minVal, float maxVal)
{
    // Generate a random float in the range [0, 1]
    float random_float = (float) rand() / RAND_MAX;

    // Scale and shift the random float to fit within the specified range
    float result = random_float * (maxVal - minVal) + minVal;

    return result;
}*/

static float getRandomFloat(float minVal, float maxVal)
{
    float random_float = (float) rand() / ((float)RAND_MAX + 1.0f);
    return random_float * (maxVal - minVal) + minVal;
}

// Function to determine if an event with a fixed chance occurs
static int eventOccurs(float chance_percentage)
{
    if (chance_percentage<=0.0)
    {
        //0 percent chance for event to happen
        return 0;
    } else
    if (chance_percentage >= 100.0)
    {
        // Event will happen for sure
        return 1;
    }

    // Generate a random number between 0 and 99
    float random_number = getRandomFloat(0.0, 99.99);

    // Check if the random number falls within the chance range
    return (random_number < chance_percentage);
}

/*
static void createRandomImage(struct Image *random_image)
{
    unsigned char * ptr    = random_image->pixels;
    unsigned char * ptrEnd = ptr + random_image->width * random_image->height * random_image->channels;

    while (ptr<ptrEnd)
    {
        *ptr = (unsigned char) rand() % 256; // Random value between 0 and 255
        ptr+=1;
    }
   return;
}*/

static inline int fastRandomByte()
{
    return (int)(((uint64_t) rand() * 256) >> 31); // Value in [0, 255]
}

static void createRandomImage(struct Image *random_image)
{
    unsigned char *ptr    = random_image->pixels;
    unsigned char *ptrEnd = ptr + (random_image->width * random_image->height * random_image->channels);

    while (ptr < ptrEnd)
    {
        *ptr++ = (unsigned char) fastRandomByte();
    }
}


static void createRandomImageRGB(struct Image *random_image)
{
    unsigned char *ptr    = random_image->pixels;
    unsigned char *ptrEnd = ptr + (random_image->width * random_image->height * random_image->channels);

    while (ptr < ptrEnd)
    {
        // Generate a new 24-bit random value
        uint32_t randomValue = ((uint32_t)rand() << 16) | ((uint32_t)rand() & 0xFFFF);

        *ptr++ = (randomValue >> 0)  & 0xFF; // Red
        *ptr++ = (randomValue >> 8)  & 0xFF; // Green
        *ptr++ = (randomValue >> 16) & 0xFF; // Blue
    }
}

static void adjustBrightnessContrastNoBorder(struct Image *image, float brightness, float contrast)
{
    unsigned char * ptr    = image->pixels;
    unsigned char * ptrEnd = ptr + image->width * image->height * image->channels;

    // Adjust brightness
    while(ptr<ptrEnd)
    {
        float pixel_value = (float) *ptr;
        pixel_value = (pixel_value * contrast) + brightness;

        pixel_value = (pixel_value < 0) ? 0 : (pixel_value > 255) ? 255 : pixel_value;

        *ptr = (unsigned char) pixel_value;
        ptr+=1;
    }
   return;
}

static void adjustBrightnessContrastSimple(struct Image *image,
                                     float brightnessR, float contrastR,
                                     float brightnessG, float contrastG,
                                     float brightnessB, float contrastB,
                                     float borderX, float borderY
                                     )
{
    signed int start_x = (signed int) borderX;
    signed int start_y = (signed int) borderY;
    signed int end_x = image->width  - start_x;
    signed int end_y = image->height - start_y;

    if (start_x<0)
        {start_x = 0; }
    if (start_y<0)
        {start_y = 0; }

    if (end_x>image->width)
        {end_x = image->width; }
    if (end_y>image->height)
        {end_y = image->height; }


    float pixel_value;

    // Adjust brightness
    for (unsigned int y = start_y; y < end_y; ++y)
    {
        unsigned int index = (y * image->width * image->channels) + (start_x * image->channels); //Avoid doing an expensive calculation for every pixel
        for (unsigned int x = start_x; x < end_x; ++x)
        {
            //We could do a heavier calculation for each iteration, however lets just perform additions to the index instead
            //unsigned int index = (y * image->width * image->channels) + (x * image->channels);

            //R
            pixel_value = (float) image->pixels[index];                                      // Read previous values (image->pixels is not memory aligned)
            pixel_value *= contrastR;                                                        // Adjust contrast
            pixel_value += brightnessR;                                                      // Adjust brightness
            pixel_value = (pixel_value < 0) ? 0 : (pixel_value > 255) ? 255 : pixel_value;   // Clamp pixel value to [0, 255]
            image->pixels[index] = (unsigned char)pixel_value;                               // Update pixel value
            index+=1;

            //G
            pixel_value = (float) image->pixels[index];                                      // Read previous values (image->pixels is not memory aligned)
            pixel_value *= contrastG;                                                        // Adjust contrast
            pixel_value += brightnessG;                                                      // Adjust brightness
            pixel_value = (pixel_value < 0) ? 0 : (pixel_value > 255) ? 255 : pixel_value;   // Clamp pixel value to [0, 255]
            image->pixels[index] = (unsigned char)pixel_value;                               // Update pixel value
            index+=1;

            //B
            pixel_value = (float) image->pixels[index];                                      // Read previous values (image->pixels is not memory aligned)
            pixel_value *= contrastB;                                                        // Adjust contrast
            pixel_value += brightnessB;                                                      // Adjust brightness
            pixel_value = (pixel_value < 0) ? 0 : (pixel_value > 255) ? 255 : pixel_value;   // Clamp pixel value to [0, 255]
            image->pixels[index] = (unsigned char) pixel_value;                              // Update pixel value
            index+=1;
        }
    }
   return;
}


static void adjustBrightnessContrast(struct Image *image,
                                     float brightnessR, float contrastR,
                                     float brightnessG, float contrastG,
                                     float brightnessB, float contrastB,
                                     float borderX, float borderY
                                     )
{
    #if INTEL_OPTIMIZATIONS
     adjustBrightnessContrast_AVX2(image,brightnessR,contrastR,brightnessG,contrastG,brightnessB,contrastB,borderX,borderY);
    #else
     adjustBrightnessContrastSimple(image,brightnessR,contrastR,brightnessG,contrastG,brightnessB,contrastB,borderX,borderY);
    #endif // INTEL_OPTIMIZATIONS
}

//Randomized few pixel attacks
//https://arxiv.org/abs/1710.08864
static void burnedPixels(struct Image *image,int numberOfBurnedPixels,float borderX, float borderY)
{
    if (numberOfBurnedPixels==0) { return; } //Fast-path
    signed int start_x = (signed int) borderX;
    signed int start_y = (signed int) borderY;
    signed int end_x = image->width  - start_x;
    signed int end_y = image->height - start_y;

    if (start_x<0)
        {start_x = 0; }
    if (start_y<0)
        {start_y = 0; }

    if (end_x>image->width)
        {end_x = image->width; }
    if (end_y>image->height)
        {end_y = image->height; }

    //We randomly set some "burned" pixels
    for (int iteration=0; iteration<numberOfBurnedPixels; iteration++)
    {
      unsigned int x = getRandomNumber(start_x,end_x-1);
      unsigned int y = getRandomNumber(start_y,end_y-1);

      unsigned int index = (y * image->width * image->channels) + (x * image->channels);

      /*
      image->pixels[index+0] = (unsigned char) rand() % 256;
      image->pixels[index+1] = (unsigned char) rand() % 256;
      image->pixels[index+2] = (unsigned char) rand() % 256;*/

      //1/3 of rand() calls
      int a = rand();
      image->pixels[index+0] = (unsigned char) a;
      image->pixels[index+1] = (unsigned char) a >> 8;
      image->pixels[index+2] = (unsigned char) a >> 16;
    }
}

static inline int fastRandomInRange(int min, int max)
{
    unsigned int range = max - min + 1;
    return (int)(((uint64_t) rand() * range) >> 31) + min;
}

static void perturbGaussianNoise(struct Image *image,unsigned char magnitude,float borderX, float borderY)
{
    // Calculate maximum deviation based on magnitude
    unsigned char max_deviation = magnitude / 2;

    signed int start_x = (signed int) borderX;
    signed int start_y = (signed int) borderY;
    signed int end_x = image->width  - start_x;
    signed int end_y = image->height - start_y;

    if (start_x<0)
        { start_x = 0; }
    if (start_y<0)
        { start_y = 0; }

    if (end_x>image->width)
        { end_x = image->width; }
    if (end_y>image->height)
        { end_y = image->height; }

    signed short deviation, new_value, summed;
    // Adjust brightness
    for (unsigned int y = start_y; y < end_y; ++y)
    {
        unsigned int index = (y * image->width * image->channels) + (start_x * image->channels);

        for (unsigned int x = start_x; x < end_x; ++x)
        {

            #if RAND_MAX < 0xFFFFFF
             #error "RAND_MAX is too small: need at least 24 bits for 3 random bytes"
            #endif

            unsigned int randomValue  = rand();
            unsigned char rA = (randomValue >> 0)  & 0xFF; // Red
            unsigned char rG = (randomValue >> 8)  & 0xFF; // Green
            unsigned char rB = (randomValue >> 16) & 0xFF; // Blue

            // Generate a random value in range [-max_deviation, max_deviation]
            deviation = (rA % (max_deviation * 2 + 1)) - max_deviation;
            //deviation = fastRandomInRange(-max_deviation, max_deviation);
            new_value = image->pixels[index];
            summed    = new_value + deviation;
            new_value +=  ( (0<=summed) * (summed<=255)) * deviation;
            image->pixels[index] = (unsigned char) new_value;
            index+=1;

            // Generate a random value in range [-max_deviation, max_deviation]
            deviation = (rG % (max_deviation * 2 + 1)) - max_deviation;
            //deviation = fastRandomInRange(-max_deviation, max_deviation);
            new_value = image->pixels[index];
            summed    = new_value + deviation;
            new_value +=  ( (0<=summed) * (summed<=255)) * deviation;
            image->pixels[index] = (unsigned char) new_value;
            index+=1;

            // Generate a random value in range [-max_deviation, max_deviation]
            deviation = (rB % (max_deviation * 2 + 1)) - max_deviation;
            //deviation = fastRandomInRange(-max_deviation, max_deviation);
            new_value = image->pixels[index];
            summed    = new_value + deviation;
            new_value +=  ( (0<=summed) * (summed<=255)) * deviation;
            image->pixels[index] = (unsigned char) new_value;
            index+=1;
        }
    }
}

/*

static void perturbGaussianNoise(struct Image *image, char magnitude, float borderX, float borderY)
{
    unsigned char max_deviation = magnitude / 2;

    signed int start_x = (signed int) borderX;
    signed int start_y = (signed int) borderY;
    signed int end_x = image->width  - start_x;
    signed int end_y = image->height - start_y;

    if (start_x < 0) start_x = 0;
    if (start_y < 0) start_y = 0;
    if (end_x > image->width) end_x = image->width;
    if (end_y > image->height) end_y = image->height;

    unsigned int pixels_wide  = end_x - start_x;
    unsigned int pixels_high  = end_y - start_y;
    unsigned int total_pixels = pixels_wide * pixels_high;


    unsigned int rand_index = 0;

    #define CHANNELS 3  // always assume RGB to help unrolling

    for (unsigned int y = start_y; y < end_y; ++y)
    {
        unsigned int index = (y * image->width * image->channels) + (start_x * image->channels);

        for (unsigned int x = start_x; x < end_x; ++x)
        {
            for (int c = 0; c < CHANNELS; ++c)
            {
                int deviation = fastRandomInRange(-max_deviation, max_deviation);
                int new_value = image->pixels[index];
                int summed = new_value + deviation;

                new_value += ((0 <= summed) & (summed <= 255)) * deviation;
                image->pixels[index] = (unsigned char) new_value;

                index++;
            }
        }
    }
}*/

static void computeFourierFeatures(struct Image *img, int nmin, int nmax, float *fourier_features)
{
    unsigned char *ptr = img->pixels;
    unsigned char *ptrEnd = img->pixels + (img->width * img->height * img->channels);

    // Iterate over each pixel in the image
    while (ptr < ptrEnd)
    {
        // Extract pixel coordinates
        int x = (ptr - img->pixels) % img->width;
        int y = (ptr - img->pixels) / img->width;

        // Iterate over each channel of the pixel
        for (int c = 0; c < img->channels; ++c)
        {
            // Calculate the index of the current pixel in the image
            int pixel_index = (y * img->width + x) * img->channels + c;

            // Extract the scalar value at the current pixel and channel
            float z_value = ptr[pixel_index];

            // Iterate over the range of integers from nmin to nmax
            for (int n = nmin; n <= nmax; ++n)
            {
                // Calculate Fourier features for sine and cosine
                float sine_value   = sin(z_value * pow(2, n) * PI);
                float cosine_value = cos(z_value * pow(2, n) * PI);

                // Store the computed Fourier features
                *fourier_features++ = sine_value;
                *fourier_features++ = cosine_value;
            }
        }
        ptr++;
    }
}

/* // Example: Compute Fourier features for nmin = 7 and nmax = 8
    int nmin = 7;
    int nmax = 8;
    int num_features = width * height * channels * (nmax - nmin + 1) * 2; // Each feature has a sine and cosine component
    float *fourier_features = (float *)malloc(num_features * sizeof(float));
    computeFourierFeatures(&img, nmin, nmax, fourier_features);*/







// Function to rotate the image around its center
static void rotateImage(struct Image *input_img, struct Image *output_img, float angle_degrees)
{
    // Convert angle from degrees to radians
    float angle_radians = angle_degrees * M_PI / 180.0f;

    // Calculate sine and cosine of the rotation angle
    float cos_theta = cosf(angle_radians);
    float sin_theta = sinf(angle_radians);

    // Calculate the coordinates of the center of the input image
    float cx = input_img->width / 2.0f;
    float cy = input_img->height / 2.0f;

    // Calculate the coordinates of the center of the output image
    float new_cx = output_img->width / 2.0f;
    float new_cy = output_img->height / 2.0f;

    // Loop through each pixel in the output image
    for (unsigned int y = 0; y < output_img->height; y++)
    {
        for (unsigned int x = 0; x < output_img->width; x++)
        {
            // Translate the coordinates so that the center becomes the origin
            float tx = x - new_cx;
            float ty = y - new_cy;

            // Rotate the coordinates around the origin
            float new_x = cos_theta * tx - sin_theta * ty;
            float new_y = sin_theta * tx + cos_theta * ty;

            // Translate the coordinates back to their original position
            new_x += cx;
            new_y += cy;

            // Check if the new coordinates are within bounds of the input image
            if (new_x >= 0 && new_x < input_img->width && new_y >= 0 && new_y < input_img->height)
            {
                // Calculate the indices for input and output pixels
                int input_index = ((int)new_y * input_img->width + (int)new_x) * input_img->channels;
                int output_index = (y * output_img->width + x) * input_img->channels;

                // Copy pixel values from input to output
                memcpy(output_img->pixels + output_index, input_img->pixels + input_index, input_img->channels);
            } else
            {
                // Set background color for out-of-bounds pixels (e.g., black)
                memset(output_img->pixels + (y * output_img->width + x) * input_img->channels, 0, input_img->channels);
            }
        }
    }
}




static float generateGaussianNoise(float mean, float stddev)
{
    // Box-Muller transform to generate Gaussian noise
    float u1 = rand() / (RAND_MAX + 1.0f);
    float u2 = rand() / (RAND_MAX + 1.0f);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mean + stddev * z;
}

static void perturbGaussianNoiseHQ(struct Image *img, char magnitude)
{
    // Calculate standard deviation based on magnitude
    float stddev = (float)magnitude / 2.0f;

    unsigned char * ptr    = img->pixels;
    unsigned char * ptrEnd = img->pixels + (img->width * img->height * img->channels);
    // Perturb each pixel with Gaussian noise
    while (ptr<ptrEnd)
    {
     if (*ptr!=0) //Avoid painting over border
     {
       float noise = generateGaussianNoise(0.0f, stddev);
       int new_value = *ptr + (int)roundf(noise);
       *ptr = (unsigned char)fminf(fmaxf(new_value, 0), 255);
     }
      ++ptr;
    }
}


static void perturbGaussianNoiseNoBorder(struct Image *img, char magnitude,float borderX, float borderY)
{
    // Calculate maximum deviation based on magnitude
    unsigned char max_deviation = magnitude / 2;

    unsigned char *ptr = img->pixels;
    unsigned char *ptrEnd = img->pixels + (img->width * img->height * img->channels);

    // Perturb each pixel with simple random noise
    while (ptr < ptrEnd)
    {
      if (*ptr != 0) //Avoid painting over border
        {
            // Generate a random value in range [-max_deviation, max_deviation]
            int deviation = (rand() % (max_deviation * 2 + 1)) - max_deviation;
            int new_value = *ptr;
            int summed    = new_value + deviation;
            new_value +=  ( (0<=summed) * (summed<=255)) * deviation;
            *ptr = (unsigned char) new_value;
        }
        ++ptr;
    }
}



static void perturbGaussianNoiseSlow(struct Image *img, char magnitude)
{
   return;
    // Calculate standard deviation based on magnitude
    float stddev = (float)magnitude / 2.0f;

    // Perturb each pixel with Gaussian noise
    for (unsigned int i = 0; i < img->height; i++)
    {
        for (unsigned int j = 0; j < img->width; j++)
        {
            unsigned int index = (i * img->width + j) * img->channels;
            for (unsigned int k = 0; k < img->channels; k++)
            {
               if (img->pixels[index + k]!=0)
               {
                float noise = generateGaussianNoise(0.0f, stddev);
                int new_value = img->pixels[index + k] + (int)roundf(noise);
                // Ensure pixel values are within 0-255 range
                img->pixels[index + k] = (unsigned char)fminf(fmaxf(new_value, 0), 255);
               }
            }
        }
    }
}


// Function to perform 2D convolution on an image with a given kernel
static void convolveImage(struct Image *input_img, struct Image *output_img,const float *kernel, int kernel_size) {
    // Calculate the border size for padding
    int border_size = kernel_size / 2;

    // Allocate memory for the padded input image
    unsigned int padded_width    = input_img->width  + 2 * border_size;
    unsigned int padded_height   = input_img->height + 2 * border_size;
    unsigned char *padded_pixels = (unsigned char *)malloc(padded_width * padded_height * input_img->channels * sizeof(unsigned char));
    if (padded_pixels == NULL) {
        fprintf(stderr, "Memory allocation failed for padded image\n");
        exit(1);
    }

    // Initialize the padded image with zeros
    memset(padded_pixels, 0, padded_width * padded_height * input_img->channels * sizeof(unsigned char));

    // Copy the input image pixels to the center of the padded image
    for (unsigned int y = 0; y < input_img->height; y++)
    {
        for (unsigned int x = 0; x < input_img->width; x++)
        {
            unsigned int input_index = (y * input_img->width + x) * input_img->channels;
            unsigned int padded_index = ((y + border_size) * padded_width + (x + border_size)) * input_img->channels;
            memcpy(padded_pixels + padded_index, input_img->pixels + input_index, input_img->channels);
        }
    }

    // Perform 2D convolution
    for (unsigned int y = 0; y < input_img->height; y++)
    {
        for (unsigned int x = 0; x < input_img->width; x++)
        {
            for (unsigned int c = 0; c < input_img->channels; c++)
            {
                float sum = 0.0f;
                for (int ky = -border_size; ky <= border_size; ky++)
                {
                    for (int kx = -border_size; kx <= border_size; kx++)
                    {
                        // Calculate the coordinates in the padded image
                        int padded_x = x + kx + border_size;
                        int padded_y = y + ky + border_size;
                        // Calculate the index for accessing the kernel
                        int kernel_index = (ky + border_size) * kernel_size + (kx + border_size);
                        // Calculate the index for accessing the padded image
                        int padded_index = (padded_y * padded_width + padded_x) * input_img->channels + c;
                        // Accumulate the sum
                        sum += kernel[kernel_index] * padded_pixels[padded_index];
                    }
                }
                // Set the pixel value in the output image
                unsigned int output_index = (y * input_img->width + x) * input_img->channels + c;
                output_img->pixels[output_index] = (unsigned char)sum;
            }
        }
    }

    // Free memory allocated for the padded image
    free(padded_pixels);
}


// Function to randomly pick zoom factor, pan_x, and pan_y
static void randomizeZoomAndPan(struct Image *input_img,float maxZoomLimit,int target_width,int target_height, float *zoom_factor, int *pan_x, int *pan_y)
{
  //By default dont zoom
  *zoom_factor = 1.0;
  *pan_x       = 0;
  *pan_y       = 0;
  //return; // Deactivate pan & zoom <- Safety
  if (input_img!=0)
  {
   if ( (input_img->width!=0) && (input_img->height!=0) )
   {
    // Calculate the maximum zoom factor to fit the entire input image
    float max_zoom_factor_x = (float) input_img->width  / target_width;
    float max_zoom_factor_y = (float) input_img->height / target_height;
    float max_zoom_factor = fmin(max_zoom_factor_x, max_zoom_factor_y);

    if (max_zoom_factor>1.0)
    {
      if ((maxZoomLimit>1.0) && (max_zoom_factor > maxZoomLimit)) //(maxZoomLimit!=0.0) &&
      {
          //Accept maxZoomLimit if it is in range
          max_zoom_factor = maxZoomLimit;
      }

    // Randomly select a zoom factor within a reasonable range
    float new_zoom_factor = 1.0 + ((float)rand() / RAND_MAX) * (max_zoom_factor-1.0);

    // Calculate the maximum pan range based on the selected zoom factor
    int max_pan_x = (int)((input_img->width  / new_zoom_factor) - target_width);
    int max_pan_y = (int)((input_img->height / new_zoom_factor) - target_height);

    //fprintf(stderr,"randomizeZoomAndPan(image %ux%u -> %ux%u / zoom=%0.2fx\n",input_img->width,input_img->height,target_width,target_height,new_zoom_factor);
    //fprintf(stderr,"max_pan_x/y  %d/%d\n",max_pan_x,max_pan_y);

    if ( (max_pan_x!=0) || (max_pan_y!=0) ) //We can pan
    { // Randomly select pan_x and pan_y within the maximum pan range
      *zoom_factor = new_zoom_factor;
      if (max_pan_x!=0)
        { *pan_x = rand() % max_pan_x; } // Add 1 to include the upper bound
      if (max_pan_y!=0)
        { *pan_y = rand() % max_pan_y; } // Add 1 to include the upper bound

      //int newX1 = (int)*pan_x;
      //int newY1 = (int)*pan_y;
      //int newX2 =  newX1 + ((int) target_width *new_zoom_factor);
      //int newY2 =  newY1 + ((int) target_height*new_zoom_factor) ;
      //fprintf(stderr," Pan & zoom region %u,%u -> %u,%u \n", newX1 , newY1 ,newX2, newY2);
    }
    } //We can zoom
   }
  }
}

static void transformCoordinatesPanAndZoom(float *x, float *y, float pan_x, float pan_y, float zoom_factor, int targetWidth, int targetHeight)
{
    if ((zoom_factor==1.0) && (pan_x==0.0) && (pan_y==0.0)) { return; } //Fast-path
    // Apply Pan and Zoom Transformation
    float newX1 = pan_x;
    float newY1 = pan_y;
    float newX2 = newX1 + ((int) targetWidth  * zoom_factor);
    float newY2 = newY1 + ((int) targetHeight * zoom_factor);
    float newWidth  = newX2 - newX1;
    float newHeight = newY2 - newY1;

    float input_x = *x;
    float input_y = *y;

    // Calculate corresponding position in the panned and zoomed image space
    *x = ((input_x - newX1) / newWidth)  * targetWidth;
    *y = ((input_y - newY1) / newHeight) * targetHeight;
}

// Function to pan and zoom the image
// pan_x, pan_y refer to the upper left coordinate of the tile
// zoom factor refers to the zoom of the input_img, for example assuming a 100x100 image, targetWidth/Height 50x50,  with zoom_factor 2.0 and panx/y 0 will get the upper [0..50,0..50] pixels
static int panAndZoom8BitImage(struct Image *input_img, float zoom_factor, int pan_x, int pan_y, int targetWidth, int targetHeight)
{
    //fprintf(stderr,"panAndZoom(zoom=%0.2f/pan=%u,%u)\n",zoom_factor,pan_x,pan_y);
    if ( (zoom_factor==1.0) && (pan_x==0) && (pan_y==0) ) { return 1; } //Fast path
    if (zoom_factor==0.0) { return 0; } //This will result in division by zero

    // Allocate memory for the output image pixels
    unsigned char *output_img_pixels = (unsigned char *) malloc(targetWidth * targetHeight * input_img->channels * sizeof(unsigned char));
    if (output_img_pixels == NULL)
    {
        fprintf(stderr, "Memory allocation failed for output image\n");
        return 0;
    }

    //newX1/Y1 and X2/Y2 are the coordinates on the input_img we want to rescale to 0..targetWidth,0..targetHeight
    int newX1     = (int) pan_x;
    int newY1     = (int) pan_y;
    int newX2     = newX1 + ((int) targetWidth  * zoom_factor);
    int newY2     = newY1 + ((int) targetHeight * zoom_factor);
    int newWidth  = newX2-newX1;
    int newHeight = newY2-newY1;

    // Copy the pixels from the input image to the output image based on pan and zoom
    for (int y = 0; y < targetHeight; y++)
    {
       int input_y = (int)(newY1 + ((y * newHeight) / targetHeight));

       if (input_y >= 0 && input_y < input_img->height)
       {
        for (int x = 0; x < targetWidth; x++)
        {
            // Calculate the corresponding position in the input image
            int input_x = (int)(newX1 + ((x * newWidth)  / targetWidth));

            // Check if the input coordinates are within bounds
            if (input_x >= 0 && input_x < input_img->width)
            {
                // Calculate the indices for input and output pixels
                int input_index  = (input_y * input_img->width + input_x) * input_img->channels;
                int output_index = (y * targetWidth + x) * input_img->channels;

                // Copy pixel values from input to output
                memcpy(output_img_pixels + output_index, input_img->pixels + input_index, input_img->channels);
            }
        }
       }
    }

    free(input_img->pixels);
    input_img->pixels       = 0;
    //------------------------------------------
    input_img->pixels       = output_img_pixels;
    input_img->width        = targetWidth;
    input_img->height       = targetHeight;
    input_img->image_size   = targetWidth * targetHeight * input_img->channels * sizeof(unsigned char);

    return 1;
}


// Function to pan and zoom the image
// pan_x, pan_y refer to the upper left coordinate of the tile
// zoom factor refers to the zoom of the input_img, for example assuming a 100x100 image, targetWidth/Height 50x50,  with zoom_factor 2.0 and panx/y 0 will get the upper [0..50,0..50] pixels
static int panAndZoom16BitImage(struct Image *input_img, float zoom_factor, int pan_x, int pan_y, int targetWidth, int targetHeight)
{
    //fprintf(stderr,"panAndZoom(zoom=%0.2f/pan=%u,%u)\n",zoom_factor,pan_x,pan_y);
    if ( (zoom_factor==1.0) && (pan_x==0) && (pan_y==0) ) { return 1; } // Fast path
    if (zoom_factor==0.0) { return 0; } // This will result in division by zero

    // Allocate memory for the output image pixels
    unsigned short *output_img_pixels = (unsigned short *) malloc(targetWidth * targetHeight * input_img->channels * sizeof(unsigned short));
    if (output_img_pixels == NULL)
    {
        fprintf(stderr, "Memory allocation failed for output image\n");
        return 0;
    }

    //This is not needed
    //memset(output_img_pixels, (signed short) 0, targetWidth * targetHeight * input_img->channels * sizeof(unsigned short));

    // newX1/Y1 and X2/Y2 are the coordinates on the input_img we want to rescale to 0..targetWidth,0..targetHeight
    int newX1     = (int) pan_x;
    int newY1     = (int) pan_y;
    int newX2     = newX1 + ((int) targetWidth  * zoom_factor);
    int newY2     = newY1 + ((int) targetHeight * zoom_factor);
    int newWidth  = newX2 - newX1;
    int newHeight = newY2 - newY1;

    // Copy the pixels from the input image to the output image based on pan and zoom
    for (int y = 0; y < targetHeight; y++)
    {
       int input_y = (int)(newY1 + ((y * newHeight) / targetHeight));

       if (input_y >= 0 && input_y < input_img->height)
       {
        for (int x = 0; x < targetWidth; x++)
         {
            // Calculate the corresponding position in the input image
            int input_x = (int)(newX1 + ((x * newWidth)  / targetWidth));

            // Check if the input coordinates are within bounds
            if (input_x >= 0 && input_x < input_img->width)
            {
                // Calculate the indices for input and output pixels
                int input_index  = (input_y * input_img->width + input_x) * input_img->channels;
                int output_index = (y * targetWidth + x) * input_img->channels;

                // Copy pixel values from input to output
                memcpy(output_img_pixels + output_index, input_img->pixels + input_index * sizeof(unsigned short), input_img->channels * sizeof(unsigned short));
            }
         }
       }
    }

    free(input_img->pixels);
    input_img->pixels       = 0;
    //------------------------------------------
    input_img->pixels       = (unsigned char *) output_img_pixels;
    input_img->width        = targetWidth;
    input_img->height       = targetHeight;
    input_img->image_size   = targetWidth * targetHeight * input_img->channels * sizeof(unsigned short);
    input_img->bitsperpixel = 16;

    return 1;
}



static void copyInputRGBFrameToOutput(
                                       struct Heatmaps * in,
                                       struct Heatmaps * out8bit,
                                       unsigned int borderX,
                                       unsigned int borderY,
                                       unsigned long targetImageNumber,
                                       unsigned int outputHeatmapNumber
                                     )
{
 if ( (in!=0) && (out8bit!=0) )
 {
   unsigned int hmWidth   =in->width;
   unsigned int hmHeight  =in->height;
   unsigned int hmChannels=in->channels;

   unsigned char* pixelDestination            = in->pixels + (in->width * in->height * in->channels * targetImageNumber);
   signed char* heatmapDestination8BitInitial = out8bit->pixels + (out8bit->width * out8bit->height * out8bit->channels * targetImageNumber) + outputHeatmapNumber;

   for (unsigned int y=borderY; y<hmHeight-borderY; y++)
   {
    unsigned char* rgbSource              = pixelDestination               + (y * in->width * in->channels) + (borderX * in->channels);
    unsigned char* heatmapDestination8Bit = heatmapDestination8BitInitial  + (y * out8bit->width * out8bit->channels) + (borderX * out8bit->channels);

    for (unsigned int x=borderX; x<hmWidth-borderX; x++)
    {
     *heatmapDestination8Bit = *rgbSource / 2;
     heatmapDestination8Bit+=1;
     rgbSource+=1;

     *heatmapDestination8Bit = *rgbSource / 2;
     heatmapDestination8Bit+=1;
     rgbSource+=1;

     *heatmapDestination8Bit = *rgbSource / 2;
     rgbSource               += hmChannels-2;
     heatmapDestination8Bit  += out8bit->channels-2;
    } // X
   } // Y
 }
}



static void denoisingDiffInputRGBFrameToOutput(
                                       struct Heatmaps * in,
                                       struct Heatmaps * out8bit,
                                       unsigned int borderX,
                                       unsigned int borderY,
                                       unsigned long targetImageNumber,
                                       unsigned int outputHeatmapNumber
                                     )
{
 if ( (in!=0) && (out8bit!=0) )
 {
   unsigned int hmWidth   =in->width;
   unsigned int hmHeight  =in->height;
   unsigned int hmChannels=in->channels;

   unsigned char* pixelDestination              = in->pixels + (in->width * in->height * in->channels * targetImageNumber);
   signed char* heatmapDestination8BitInitial   = out8bit->pixels + (out8bit->width * out8bit->height * out8bit->channels * targetImageNumber) + outputHeatmapNumber;
   signed diff = 0;

   for (unsigned int y=borderY; y<hmHeight-borderY; y++)
   {
    unsigned char* rgbSource              = pixelDestination               + (y * in->width * in->channels) + (borderX * in->channels);
    unsigned char* heatmapDestination8Bit = heatmapDestination8BitInitial  + (y * out8bit->width * out8bit->channels) + (borderX * out8bit->channels);

    for (unsigned int x=borderX; x<hmWidth-borderX; x++)
    {
     diff = (*rgbSource / 2) - *heatmapDestination8Bit;
     *heatmapDestination8Bit = diff;
     heatmapDestination8Bit+=1;
     rgbSource+=1;

     diff = (*rgbSource / 2) - *heatmapDestination8Bit;
     *heatmapDestination8Bit = diff;
     heatmapDestination8Bit+=1;
     rgbSource+=1;

     diff = (*rgbSource / 2) - *heatmapDestination8Bit;
     *heatmapDestination8Bit = diff;
     rgbSource               += hmChannels-2;
     heatmapDestination8Bit  += out8bit->channels-2;
    } // X
   } // Y
 }
}


#ifdef __cplusplus
}
#endif

#endif // AUGMENTATIONS_H_INCLUDED
