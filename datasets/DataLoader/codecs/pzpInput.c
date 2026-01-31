#define USE_PZP_FILES 1

#if USE_PZP_FILES

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "codecs.h"

#include "pzpInput.h"
#include "pzp.h"

int ReadPZP(const char *filename, struct Image *pic)
{
    //fprintf(stderr, "Decompressing %s\n", filename);
    unsigned int width = 0, height = 0;
    unsigned int bitsperpixelExternal = 0, channelsExternal = 3;
    unsigned int bitsperpixelInternal = 24, channelsInternal = 3;
    unsigned int configuration = 0;

    pic->pixels = pzp_decompress_combined(filename, &width, &height, &bitsperpixelExternal, &channelsExternal, &bitsperpixelInternal, &channelsInternal, &configuration);

    unsigned int restoreRLEChannels = configuration & USE_RLE;

    pic->width = width;
    pic->height = height;
    pic->channels = channelsExternal;
    pic->bitsperpixel = bitsperpixelExternal * channelsExternal;
    pic->image_size = width * height * (bitsperpixelExternal / 8) * channelsExternal;

    return 1;
}


int ReadPZPMemory(const char *mem, unsigned int memSize, struct Image *pic)
{
    //fprintf(stderr, "Decompressing %s\n", filename);
    unsigned int width = 0, height = 0;
    unsigned int bitsperpixelExternal = 0, channelsExternal = 3;
    unsigned int bitsperpixelInternal = 24, channelsInternal = 3;
    unsigned int configuration = 0;

    //pic->pixels = pzp_decompress_combined(filename, &width, &height, &bitsperpixelExternal, &channelsExternal, &bitsperpixelInternal, &channelsInternal, &configuration);
    pic->pixels = pzp_decompress_combined_from_memory((const void *) mem, (size_t) memSize, &width, &height, &bitsperpixelExternal, &channelsExternal, &bitsperpixelInternal, &channelsInternal, &configuration);

    unsigned int restoreRLEChannels = configuration & USE_RLE;

    pic->width = width;
    pic->height = height;
    pic->channels = channelsExternal;
    pic->bitsperpixel = bitsperpixelExternal * channelsExternal;
    pic->image_size = width * height * (bitsperpixelExternal / 8) * channelsExternal;

    return 1;
}




int WritePZP(const char *filename, struct Image *pic)
{
    //fprintf(stderr, "Compressing %s\n", filename);
    unsigned int bitsperpixelInternal = pic->bitsperpixel;
    unsigned int channelsInternal = pic->channels;
    unsigned int configuration = USE_COMPRESSION | USE_RLE;

    if (pic->bitsperpixel == 16)
    {
        bitsperpixelInternal = 8;
        channelsInternal *= 2;
    }

    unsigned char **buffers = malloc(channelsInternal * sizeof(unsigned char *));
    if (buffers == NULL)
    {
        return 0;
    }

    for (unsigned int ch = 0; ch < channelsInternal; ch++)
    {
        buffers[ch] = malloc(pic->width * pic->height * sizeof(unsigned char));
    }


    pzp_split_channels(pic->pixels, buffers, channelsInternal, pic->width, pic->height);

    if (configuration & USE_RLE)
           { pzp_RLE_filter(buffers, channelsInternal, pic->width, pic->height); }


    pzp_compress_combined(buffers, pic->width, pic->height, pic->bitsperpixel, pic->channels, bitsperpixelInternal, channelsInternal, configuration, filename);

    for (unsigned int ch = 0; ch < channelsInternal; ch++)
    {
        free(buffers[ch]);
    }
    free(buffers);
    return 1;
}

#endif // USE_PZP_FILES
