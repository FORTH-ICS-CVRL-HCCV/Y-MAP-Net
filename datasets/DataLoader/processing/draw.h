#ifndef DRAW_H_INCLUDED
#define DRAW_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <math.h>

static void draw_line(unsigned short x1, unsigned short y1, unsigned short x2, unsigned short y2, signed char* imagePixels, unsigned short imageWidth, unsigned short imageHeight, signed char value)
{
    int dc = abs(x1 - x2);
    int dr = abs(y1 - y2);
    int err = dc - dr;
    int err_prime;

    int c = x1;
    int r = y1;
    int sign_c = (x1 < x2) ? 1 : -1;
    int sign_r = (y1 < y2) ? 1 : -1;

    float ed = (dc + dr == 0) ? 1 : sqrt(dc * dc + dr * dr);

    while (1)
    {
        imagePixels[r * imageWidth + c] = value;

        if (c == x2 && r == y2)
        {
            break;
        }

        err_prime = err;
        int c_prime = c;

        if (2 * err_prime >= -dc)
        {
            if (c == x2) { break; }
            if (err_prime + dr < ed)
            {
                imagePixels[(r + sign_r) * imageWidth + c] = value;
            }
            err -= dr;
            c += sign_c;
        }

        if (2 * err_prime <= dr)
        {
            if (r == y2) { break; }
            if (dc - err_prime < ed)
            {
                imagePixels[r * imageWidth + c_prime + sign_c] = value;
            }
            err += dc;
            r += sign_r;
        }
    } // Loop
}


static void setHeatmapValueForContinuousChannels(
    struct Heatmaps *out8bit,
    unsigned long targetImageNumber,
    int startChannel,
    int endChannel,
    signed char value
)
{
    // Safety checks for image index
    if (!out8bit) return;
    if (targetImageNumber >= out8bit->numberOfImages) return;

    // Heatmap geometry
    const unsigned int width    = out8bit->width;
    const unsigned int height   = out8bit->height;
    const unsigned int channels = out8bit->channels;

    if (channels == 0 || width == 0 || height == 0) return;

    // Clamp channel range
    if (startChannel < 0)            { startChannel = 0; }
    if (endChannel >= (int)channels) { endChannel = channels - 1; }
    if (startChannel > endChannel)   { return; }

    const int numChannelsToSet = endChannel - startChannel + 1;

    // Compute total pixels per image
    unsigned long pixelsPerImage = (unsigned long)width * (unsigned long) height * (unsigned long) channels;

    // Compute pointer to the start of the requested image
    signed char *heatmapPTR      = (signed char*) out8bit->pixels + (pixelsPerImage * targetImageNumber);

    // Limit check
    if ((void*)heatmapPTR >= out8bit->pixelsLimit) { return; }

    // Row size in bytes
    const unsigned long rowBytes = (unsigned long) width * (unsigned long) channels;

    // Fast loop: iterate rows and pixels
    for (unsigned int y = 0; y < height; y++)
    {
        signed char *rowPtr = heatmapPTR + y * rowBytes;

        // Pointer to first pixel in this row at startChannel
        signed char *dst = rowPtr + startChannel;

        for (unsigned int x = 0; x < width; x++)
        {
            memset(dst, value, numChannelsToSet);

            // Move to next pixel (skip full channel block)
            dst += channels;
        }
    }
}


#ifdef __cplusplus
}
#endif

#endif // DRAW_H_INCLUDED
