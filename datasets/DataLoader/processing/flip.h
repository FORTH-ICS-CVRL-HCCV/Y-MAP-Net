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



#ifdef __cplusplus
}
#endif

#endif // FLIP_H_INCLUDED
