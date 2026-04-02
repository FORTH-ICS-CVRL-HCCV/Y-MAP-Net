#ifndef FLIP_H_INCLUDED
#define FLIP_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>



static void flipImageHoriz_8bit(
                                unsigned char *data,
                                unsigned int width,
                                unsigned int height,
                                unsigned int startChannel,
                                unsigned int channels,
                                unsigned int offsetX,
                                unsigned int offsetY
                              )
{
  if (!data || channels==0) { return; }

  const unsigned int xStart = offsetX;
  const unsigned int xEnd   = width - 1 - offsetX; // inclusive
  if (xStart >= width || xStart >= xEnd) { return; }

  for (unsigned int y = offsetY; y < height - offsetY; ++y)
  {
    unsigned int left  = xStart;
    unsigned int right = xEnd;

    while (left < right)
    {
      unsigned char *L = data + ((y * width + left ) * channels);
      unsigned char *R = data + ((y * width + right) * channels);

      // swap per-channel
      for (unsigned int c = startChannel; c < channels; ++c)
      {
        unsigned char tmp = L[c];
        L[c] = R[c];
        R[c] = tmp;
      }

      ++left;
      --right;
    }
  }
}

static void flipImageHoriz_16bit(
                                 signed short *data,
                                 unsigned int width,
                                 unsigned int height,
                                 unsigned int channels,
                                 unsigned int offsetX,
                                 unsigned int offsetY
                                )
{
  if (!data || channels==0) { return; }

  const unsigned int xStart = offsetX;
  const unsigned int xEnd   = width - 1 - offsetX; // inclusive
  if (xStart >= width || xStart >= xEnd) { return; }

  for (unsigned int y = offsetY; y < height - offsetY; ++y)
  {
    unsigned int left  = xStart;
    unsigned int right = xEnd;

    while (left < right)
    {
      signed short *L = data + ((y * width + left ) * channels);
      signed short *R = data + ((y * width + right) * channels);

      // swap per-channel
      for (unsigned int c = 0; c < channels; ++c)
      {
        signed short tmp = L[c];
        L[c] = R[c];
        R[c] = tmp;
      }

      ++left;
      --right;
    }
  }
}



// ---------------------------------------------------------------------------
// rotate90_heatmap_8bit / rotate90_heatmap_16bit
//
// Rotate ALL channels of an interleaved heatmap buffer by ±90°, filling the
// entire width×height frame (no black border remains).  The same transform as
// rotate90() in augmentations.h — extract the active inner region, rotate it,
// then nearest-neighbour scale to fill the full W×H output.
//
// data    : base pointer for ONE sample  (e.g. out8bit.pixels + sample_offset)
// width/height/channels : dimensions of that buffer
// clockwise : 1 = CW 90°,  0 = CCW 90°
// offsetX/Y : pixel border widths (same values used for the RGB input)
// ---------------------------------------------------------------------------
static void rotate90_heatmap_8bit(
    signed char  *data,
    unsigned int  width,
    unsigned int  height,
    unsigned int  channels,
    int           clockwise,
    unsigned int  offsetX,
    unsigned int  offsetY)
{
    if (!data || channels == 0) return;

    int sx = (int)offsetX,  sy = (int)offsetY;
    int ex = (int)width - sx, ey = (int)height - sy;
    int iW = ex - sx,  iH = ey - sy;
    if (iW <= 0 || iH <= 0) return;

    unsigned int rW = (unsigned int)iH;   /* rotated buffer width  */
    unsigned int rH = (unsigned int)iW;   /* rotated buffer height */
    signed char *rot = (signed char *)malloc((size_t)rW * rH * channels);
    if (!rot) return;

    /* Step 1: rotate inner region into temp buffer */
    for (int y = 0; y < iH; ++y)
        for (int x = 0; x < iW; ++x)
        {
            const signed char *sp = data + ((unsigned int)(sy + y) * width + (unsigned int)(sx + x)) * channels;
            signed char *dp;
            if (clockwise)
                dp = rot + ((unsigned int)x * rW + (unsigned int)(iH - 1 - y)) * channels;
            else
                dp = rot + ((unsigned int)(iW - 1 - x) * rW + (unsigned int)y) * channels;
            for (unsigned int c = 0; c < channels; ++c) dp[c] = sp[c];
        }

    /* Step 2: scale rot (rW×rH) to fill full width×height */
    for (unsigned int oy = 0; oy < height; ++oy)
        for (unsigned int ox = 0; ox < width; ++ox)
        {
            unsigned int rx = (unsigned int)((float)ox * (float)rW / (float)width);
            unsigned int ry = (unsigned int)((float)oy * (float)rH / (float)height);
            if (rx >= rW) rx = rW - 1;
            if (ry >= rH) ry = rH - 1;
            const signed char *sp = rot + (ry * rW + rx) * channels;
            signed char *dp = data + (oy * width + ox) * channels;
            for (unsigned int c = 0; c < channels; ++c) dp[c] = sp[c];
        }

    free(rot);
}

static void rotate90_heatmap_16bit(
    signed short *data,
    unsigned int  width,
    unsigned int  height,
    unsigned int  channels,
    int           clockwise,
    unsigned int  offsetX,
    unsigned int  offsetY)
{
    if (!data || channels == 0) return;

    int sx = (int)offsetX,  sy = (int)offsetY;
    int ex = (int)width - sx, ey = (int)height - sy;
    int iW = ex - sx,  iH = ey - sy;
    if (iW <= 0 || iH <= 0) return;

    unsigned int rW = (unsigned int)iH;
    unsigned int rH = (unsigned int)iW;
    signed short *rot = (signed short *)malloc((size_t)rW * rH * channels * sizeof(signed short));
    if (!rot) return;

    for (int y = 0; y < iH; ++y)
        for (int x = 0; x < iW; ++x)
        {
            const signed short *sp = data + ((unsigned int)(sy + y) * width + (unsigned int)(sx + x)) * channels;
            signed short *dp;
            if (clockwise)
                dp = rot + ((unsigned int)x * rW + (unsigned int)(iH - 1 - y)) * channels;
            else
                dp = rot + ((unsigned int)(iW - 1 - x) * rW + (unsigned int)y) * channels;
            for (unsigned int c = 0; c < channels; ++c) dp[c] = sp[c];
        }

    for (unsigned int oy = 0; oy < height; ++oy)
        for (unsigned int ox = 0; ox < width; ++ox)
        {
            unsigned int rx = (unsigned int)((float)ox * (float)rW / (float)width);
            unsigned int ry = (unsigned int)((float)oy * (float)rH / (float)height);
            if (rx >= rW) rx = rW - 1;
            if (ry >= rH) ry = rH - 1;
            const signed short *sp = rot + (ry * rW + rx) * channels;
            signed short *dp = data + (oy * width + ox) * channels;
            for (unsigned int c = 0; c < channels; ++c) dp[c] = sp[c];
        }

    free(rot);
}


#ifdef __cplusplus
}
#endif

#endif // FLIP_H_INCLUDED
