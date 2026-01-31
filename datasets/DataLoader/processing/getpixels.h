#ifndef GETPIXELS_H_INCLUDED
#define GETPIXELS_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../codecs/image.h"

// Helper function to get pixel value with boundary checks
static unsigned char getPixelValue8Bit(struct Image *image, int x, int y, int channel)
{
    if (channel>=image->channels)
    {
      channel = 0;//Protect from channel overflow
    }
    unsigned char * pixels = (unsigned char* ) image->pixels;

    if (x < 0) { x = 0; }
    if (x >= image->width)  { x = image->width - 1; }
    if (y < 0) { y = 0; }
    if (y >= image->height) { y = image->height - 1; }

    return pixels[(y * image->width + x) * image->channels + channel];
}

// Helper function to get pixel value with boundary checks
static unsigned short getPixelValue16BitSimple(struct Image *image, int x, int y, int channel)
{
    if (channel>=image->channels)
    {
      channel = 0;//Protect from channel overflow
    }
    unsigned short * pixels = (unsigned short* ) image->pixels;

    if (x < 0) { x = 0; }
    if (x >= image->width)  { x = image->width - 1; }
    if (y < 0) { y = 0; }
    if (y >= image->height) { y = image->height - 1; }


    unsigned short * pixelsPtr = (unsigned short *) (pixels + (y * image->width + x) * image->channels + channel);
    return *pixelsPtr;
    //return pixels[(y * image->width + x) * image->channels + channel];
}

static unsigned short getPixelValue16Bit(struct Image *image, int x, int y, int channel)
{
    if (channel >= image->channels)
    {
        channel = 0; // Protect from channel overflow
    }

    if (x < 0) { x = 0; }
    if (x >= image->width) { x = image->width - 1; }
    if (y < 0) { y = 0; }
    if (y >= image->height) { y = image->height - 1; }

    unsigned short *pixels = (unsigned short *) image->pixels;
    return pixels[(y * image->width + x) * image->channels + channel];
}


// Helper function to get pixel value with boundary checks
static unsigned short getPixelValue16BitWithChannelsAddressedAs8Bit(struct Image *image, int x, int y, int channel)
{
    if (channel>=image->channels)
    {
      channel = 0;//Protect from channel overflow
    }
    unsigned char * pixels = (unsigned char* ) image->pixels;

    if (x < 0) { x = 0; }
    if (x >= image->width)  { x = image->width - 1; }
    if (y < 0) { y = 0; }
    if (y >= image->height) { y = image->height - 1; }

    unsigned short * pixelsPtr = (unsigned short *) (pixels + (y * image->width + x) * image->channels + channel);
    return *pixelsPtr;
}

static unsigned short getPixelValue16BitWithChannelsAddressedAs8BitSPLIT(struct Image *image, int x, int y, int channel)
{
    // Ensure valid channel index
    if (channel >= image->channels)
    {
        channel = 0; // Protect from channel overflow
    }

    // Boundary checks
    if (x < 0) { x = 0; }
    if (x >= image->width) { x = image->width - 1; }
    if (y < 0) { y = 0; }
    if (y >= image->height) { y = image->height - 1; }

    // Cast the pixels array to 8-bit for byte-level access
    unsigned char *pixels = (unsigned char *)image->pixels;

    // Calculate the index of the first byte (low byte) of the 16-bit pixel
    int index = ((y * image->width + x) * image->channels + channel) * 2;

    // Combine the two 8-bit channels to form a 16-bit value
    unsigned short lowByte  = pixels[index];
    unsigned short highByte = pixels[index + 1];

    // Combine the low and high bytes to form the 16-bit value
    return (highByte << 8) | lowByte;
}

#ifdef __cplusplus
}
#endif

#endif // GETPIXELS_H_INCLUDED
