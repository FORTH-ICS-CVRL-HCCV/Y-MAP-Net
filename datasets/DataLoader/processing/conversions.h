#ifndef CONVERSIONS_H_INCLUDED
#define CONVERSIONS_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h> // For SHRT_MAX and SHRT_MIN

#include "../codecs/image.h"

#include "AVX2/conversions.h"

/*
//! Byte swap unsigned short
uint16_t swap_uint16( uint16_t val )
{
    return (val << 8) | (val >> 8 );
}

//! Byte swap short
int16_t swap_int16( int16_t val )
{
    return (val << 8) | ((val >> 8) & 0xFF);
}

//! Byte swap unsigned int
uint32_t swap_uint32( uint32_t val )
{
    val = ((val << 8) & 0xFF00FF00 ) | ((val >> 8) & 0xFF00FF );
    return (val << 16) | (val >> 16);
}

//! Byte swap int
int32_t swap_int32( int32_t val )
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF );
    return (val << 16) | ((val >> 16) & 0xFFFF);
}
*/

static void swap16bitEndiannessSimple(struct Image *img16)
{
    if (img16->bitsperpixel/img16->channels != 16)
    {
        fprintf(stderr, "Invalid image format: expected 16 bits per pixel.\n");
        return;
    }

    unsigned int width       = img16->width;
    unsigned int height      = img16->height;
    unsigned int channels    = img16->channels;
    unsigned short *pixels16 = (unsigned short *)img16->pixels;

    unsigned int index = 0;
    unsigned int totalPixels = width * height * channels;
    for (unsigned int i = 0; i<totalPixels; ++i)
    {
      unsigned short pixel16 = pixels16[index];
      unsigned short swapped = (pixel16 >> 8) | (pixel16 << 8);
      pixels16[index] = swapped;
      ++index;
    }
}


static void swap16bitEndianness(struct Image *img16)
{
 #if INTEL_OPTIMIZATIONS
  swap16bitEndianness_AVX2(img16);
 #else
  swap16bitEndiannessSimple(img16);
 #endif // INTEL_OPTIMIZATIONS
}

static void convert16bitTo8bit(const struct Image *img16, struct Image *img8)
{

    if ( ((img16->bitsperpixel/img16->channels) != 16) || ((img8->bitsperpixel/img8->channels) != 8) || (img16->channels != img8->channels) )
    {
        fprintf(stderr, "Invalid image format.\n");
        fprintf(stderr, "16bit image bitsperpixel %u / channels %u.\n",img16->bitsperpixel,img16->channels);
        fprintf(stderr, "8bit image bitsperpixel %u / channels %u.\n",img8->bitsperpixel,img8->channels);
        return;
    }

    unsigned int width       = img16->width;
    unsigned int height      = img16->height;
    unsigned int channels    = img16->channels;
    unsigned short *pixels16 = (unsigned short *)img16->pixels;
    unsigned char *pixels8   = (unsigned char *)img8->pixels;


    unsigned int index = 0;
    for (unsigned int y = 0; y < height; ++y)
    {
        for (unsigned int x = 0; x < width; ++x)
        {
            for (unsigned int c = 0; c < channels; ++c)
            {
                //unsigned int   index = (y * width + x) * channels + c;
                unsigned short pixel16 = pixels16[index];
                unsigned char  pixel8 = (unsigned char) ( (pixel16 * 255) / 65535); // Scale from 16-bit to 8-bit
                pixels8[index] = pixel8;
                ++index;
            }
        }
    }
}

static struct Image *mix3x16bit1ChannelImages(const struct Image *chA, const struct Image *chB, const struct Image *chC)
{
    if (((chA->bitsperpixel / chA->channels) != 16) || ((chB->bitsperpixel / chB->channels) != 16) || ((chC->bitsperpixel / chC->channels) != 16))
    {
        fprintf(stderr, "mix3x16bit1ChannelImages: Invalid input image format (bitsperpixel).\n");
        return NULL;
    }

    if ((chA->channels != 1) || (chB->channels != 1) || (chC->channels != 1))
    {
        fprintf(stderr, "mix3x16bit1ChannelImages: Invalid input image format (channels).\n");
        return NULL;
    }

    if ((chA->width != chB->width) || (chA->width != chC->width) || (chA->height != chB->height) || (chA->height != chC->height))
    {
        fprintf(stderr, "mix3x16bit1ChannelImages: Input images dimensions do not match.\n");
        return NULL;
    }

    struct Image *mixed = createImage(chA->width, chA->height, 3, 48);
    if (!mixed)
    {
        return NULL;
    }

    unsigned int width = chA->width;
    unsigned int height = chA->height;

    unsigned short * target    = (unsigned short*) mixed->pixels;
    unsigned short * chAPixels = (unsigned short*) chA->pixels;
    unsigned short * chBPixels = (unsigned short*) chB->pixels;
    unsigned short * chCPixels = (unsigned short*) chC->pixels;

    unsigned int index3Ch = 0;
    unsigned int index1Ch = 0;
    for (unsigned int y = 0; y < height; ++y)
    {
        for (unsigned int x = 0; x < width; ++x)
        {
            target[index3Ch] = chAPixels[index1Ch];
            index3Ch+=1;

            target[index3Ch] = chBPixels[index1Ch];
            index3Ch+=1;

            target[index3Ch] = chCPixels[index1Ch];
            index3Ch+=1;

            index1Ch+=1;
        }
    }

    return mixed;
}




static int map_2x8BitHeatmapsTo1x16BitHeatmap(SampleNumber sampleID,struct Heatmaps * heatmaps8Bit,int heatmap8BitIndex,struct Heatmaps * heatmaps16Bit,int heatmap16BitIndex)
{
  if (heatmaps8Bit==0)                                   { fprintf(stderr,RED "map_2x8BitHeatmapsTo1x16BitHeatmap no 8bit heatmap provided\n" NORMAL); return 0; }
  if (heatmaps16Bit==0)                                  { fprintf(stderr,RED "map_2x8BitHeatmapsTo1x16BitHeatmap no 16bit heatmap provided\n" NORMAL); return 0; }
  if (heatmaps8Bit->channels<=heatmap8BitIndex)          { fprintf(stderr,RED "map_2x8BitHeatmapsTo1x16BitHeatmap heatmap8BitIndex out of bounds (%u/%lu) \n" NORMAL,heatmap8BitIndex,heatmaps8Bit->numberOfImages);   return 0; }
  if (heatmaps16Bit->channels<=heatmap16BitIndex)        { fprintf(stderr,RED "map_2x8BitHeatmapsTo1x16BitHeatmap heatmap16BitIndex out of bounds (%u/%lu)\n" NORMAL,heatmap16BitIndex,heatmaps16Bit->numberOfImages);  return 0; }
  if (heatmaps8Bit->width!=heatmaps16Bit->width)         { fprintf(stderr,RED "map_2x8BitHeatmapsTo1x16BitHeatmap inconsistent width\n" NORMAL);  return 0; }
  if (heatmaps8Bit->height!=heatmaps16Bit->height)       { fprintf(stderr,RED "map_2x8BitHeatmapsTo1x16BitHeatmap inconsistent height\n" NORMAL);  return 0; }

  int width  = heatmaps8Bit->width;
  int height = heatmaps8Bit->height;

  signed char  * heatmapSource = (signed char*)   heatmaps8Bit->pixels  + (heatmaps8Bit->width  * heatmaps8Bit->height * heatmaps8Bit->channels * sampleID); //This needs to be the start of the heatmap
  signed short * heatmapTarget = (signed short *) heatmaps16Bit->pixels + (heatmaps16Bit->width * heatmaps16Bit->height * heatmaps16Bit->channels * sampleID);

  for (int y=0; y<height; y++)
  {
   for (int x=0; x<width; x++)
   {
    unsigned short * pixelsPtr = (unsigned short *) (heatmapSource + ( (y * heatmaps8Bit->width * heatmaps8Bit->channels) +  (x * heatmaps8Bit->channels) + heatmap8BitIndex) );

    //Convert to signed short range
    int pixelsInt32 = *pixelsPtr;
    pixelsInt32 -= ABS_MINV_16BIT;

    //Save signed short
    heatmapTarget[ (y * heatmaps16Bit->width * heatmaps16Bit->channels) + (x * heatmaps16Bit->channels) + heatmap16BitIndex] = (signed short) pixelsInt32;
   }
  }

 return 1;
}





static struct Image * db_create_8Bit_Image_from_heatmap(struct ImageDatabase * db,unsigned long sampleNumber,unsigned short heatmapNumber)
{
 if (db!=0)
 {
     int width  = db->out8bit.width;
     int height = db->out8bit.height;

     struct Image * img = createImage(width,height,1,8);
     if (img!=0)
     {
      if (img->pixels!=0)
      {
        //Calculate pointer addresses
        signed char * target     = (signed char*) img->pixels;
        signed char * heatmapPTR = (signed char*) db->out8bit.pixels + (width * height * db->out8bit.channels * sampleNumber);

        int x,y;
        for (y=0; y<height; y++)
        {
         for (x=0; x<width; x++)
         {
          signed char * source = heatmapPTR + ((y * width * db->out8bit.channels) + (x* db->out8bit.channels) + heatmapNumber);
          *target=*source;
          target++;
         }
        }

        return img;
       } else
       {
         free(img);
       }
     }
 }
 fprintf(stderr,"Failed db_create_8Bit_Image_from_heatmap\n");
 return 0;
}



static int convert_sint8ImageTouint8(struct Image * img)
{
 if (img!=0)
     {
      if (img->pixels!=0)
      {
        signed char   * source = (signed char *)   img->pixels;
        unsigned char * target = (unsigned char *) img->pixels;

        int i,tmp;
        int iterations = img->width * img->height * img->channels;

        for (i=0; i<iterations; i++)
        {
          tmp = *source;
          *target = (unsigned char) tmp + ABS_MINV; //120 for this application

          source++;
          target++;
        }

        return 1;
       }
     }
 return 0;
}








static struct Image * db_create_16Bit_Image_from_heatmap_8bit(struct ImageDatabase * db,unsigned long sampleNumber,unsigned short heatmapNumber)
{
 if (db!=0)
 {
     int width  = db->out8bit.width;
     int height = db->out8bit.height;

     struct Image * img = createImage(width,height,1,16);
     if (img!=0)
     {
      if (img->pixels!=0)
      {
        //Calculate pointer addresses
        unsigned short * target    = (unsigned short*) img->pixels;
        unsigned char * heatmapPTR = (unsigned char*) db->out8bit.pixels + (width * height * db->out8bit.channels * sampleNumber);

        int x,y;
        for (y=0; y<height; y++)
        {
          for (x=0; x<width; x++)
          {
           unsigned short * source = (unsigned short *) heatmapPTR + ((y * width * db->out8bit.channels) + (x* db->out8bit.channels) + heatmapNumber);
           *target=*source;
           target++;
          }
        }

        return img;
       } else
       {
         free(img);
       }
     }
 }

 fprintf(stderr,"Failed db_create_16Bit_Image_from_heatmap_8bit\n");
 return 0;
}



static struct Image * db_create_16Bit_Image_from_heatmap_16bit(struct ImageDatabase * db,unsigned long sampleNumber,unsigned short heatmapNumber)
{
 if (db!=0)
 {
     int width  = db->out16bit.width;
     int height = db->out16bit.height;

     struct Image * img = createImage(width,height,1,16);
     if (img!=0)
     {
      if (img->pixels!=0)
      {
        //Calculate pointer addresses
        unsigned short * target    = (unsigned short*) img->pixels;
        signed short * heatmapPTR  = (signed short *) db->out16bit.pixels + (width * height * db->out16bit.channels * sampleNumber);

        int x,y;
        for (y=0; y<height; y++)
        {
          for (x=0; x<width; x++)
          {
           signed short * source = heatmapPTR + ((y * width * db->out16bit.channels) + (x* db->out16bit.channels) + heatmapNumber);

           int castToUnsignedShort = *source + ABS_MINV_16BIT; //+ 32767;

           *target=(unsigned short) castToUnsignedShort;
           target++;
          }
        }

        return img;
       } else
       {
         free(img);
       }
     }
 }

 fprintf(stderr,"Failed db_create_16Bit_Image_from_heatmap_16bit\n");
 return 0;
}





static void copy16BitHeatmapTo8BitHeatmapFull(struct Heatmaps * heatmaps8Bit, //NOTICE Images not Image
                                          Heatmap8BitIndex target8BitChannel,
                                          struct Heatmaps * heatmaps16Bit, //NOTICE Images not Image
                                          Heatmap16BitIndex source16BitChannel,
                                          SampleNumber targetImageNumber)
{
  if ( (heatmaps16Bit->width == heatmaps8Bit->width) && (heatmaps16Bit->height == heatmaps8Bit->height) )
  {
    int width         = heatmaps16Bit->width;
    int height        = heatmaps16Bit->height;
    int channels16Bit = heatmaps16Bit->channels;
    int channels8Bit  = heatmaps8Bit->channels;

    signed char *  pixels8Bit  = (signed char *)  heatmaps8Bit->pixels;
    signed short * pixels16Bit = (signed short *) heatmaps16Bit->pixels;

    int y=0,x=0;
    unsigned int mem16B = (width * height * channels16Bit * targetImageNumber) + (y * width * channels16Bit) + (x * channels16Bit) + source16BitChannel;
    unsigned int mem8B  = (width * height * channels8Bit * targetImageNumber)  + (y * width * channels8Bit)  + (x * channels8Bit)  + target8BitChannel;

    for (int i = 0; i < height*width; i++)
    {
      float inV = pixels16Bit[mem16B] / 273.059;
      mem16B += channels16Bit;
      pixels8Bit[mem8B] = (signed char) inV;
      mem8B += channels8Bit;
    }
  } else
  {
      fprintf(stderr,"Cannot copy16BitHeatmapTo8BitHeatmap( from %u 16bit to %u 8bit )\n",source16BitChannel,target8BitChannel);
      fprintf(stderr,"Incoherrent dimensions ( 16Bit %ux%u / 8bit %ux%u) \n",heatmaps16Bit->width,heatmaps16Bit->height,heatmaps8Bit->width,heatmaps8Bit->height);
      abort();
  }
}


static void copy16BitHeatmapTo8BitHeatmap(struct Heatmaps * heatmaps8Bit, //NOTICE Images not Image
                                          Heatmap8BitIndex target8BitChannel,
                                          struct Heatmaps * heatmaps16Bit, //NOTICE Images not Image
                                          Heatmap16BitIndex source16BitChannel,
                                          SampleNumber targetImageNumber,
                                          unsigned int borderX,
                                          unsigned int borderY)
{
  //Use stable version that ignores borders for now..
  //copy16BitHeatmapTo8BitHeatmapFull(heatmaps8Bit,target8BitChannel,heatmaps16Bit,source16BitChannel,targetImageNumber);
  //return;

  if ( (heatmaps16Bit->width == heatmaps8Bit->width) && (heatmaps16Bit->height == heatmaps8Bit->height) )
  {
    int width         = heatmaps16Bit->width;
    int height        = heatmaps16Bit->height;
    int channels16Bit = heatmaps16Bit->channels;
    int channels8Bit  = heatmaps8Bit->channels;
    const float inv_scale = (float) 1.0f / 273.059f;

    //Source
    signed short * pixels16Bit = (signed short *) heatmaps16Bit->pixels;
    signed short * mem16BBase  =  pixels16Bit + (width * height * channels16Bit * targetImageNumber) + source16BitChannel;

    //Target
    signed char  *  pixels8Bit = (signed char *)  heatmaps8Bit->pixels;
    signed char  *  mem8BBase  =  pixels8Bit  + (width * height * channels8Bit  * targetImageNumber) + target8BitChannel;

    for (int y=borderY; y<height-borderY; y++)
    {
      //Source
      signed short * mem16B  = mem16BBase + (y * width * channels16Bit) + (borderX * channels16Bit);
      //Target
      signed char *  mem8B   = mem8BBase  + (y * width * channels8Bit)  + (borderX * channels8Bit);

      for (int x=borderX; x<width-borderX; x++)
      {
        float inV = (float) *mem16B * inv_scale; // / 273.059;
        mem16B += channels16Bit;
        *mem8B = (signed char) inV;
        mem8B += channels8Bit;
      }
    }

  } else
  {
      fprintf(stderr,"Cannot copy16BitHeatmapTo8BitHeatmap( from %u 16bit to %u 8bit )\n",source16BitChannel,target8BitChannel);
      fprintf(stderr,"Incoherrent dimensions ( 16Bit %ux%u / 8bit %ux%u) \n",heatmaps16Bit->width,heatmaps16Bit->height,heatmaps8Bit->width,heatmaps8Bit->height);
      abort();
  }
}


static void copy16BitHeatmapTo8BitHeatmapWithRemap(struct Heatmaps * heatmaps8Bit,
                                                   Heatmap8BitIndex target8BitChannel,
                                                   struct Heatmaps * heatmaps16Bit,
                                                   Heatmap16BitIndex source16BitChannel,
                                                   SampleNumber targetImageNumber,
                                                   unsigned int borderX,
                                                   unsigned int borderY)
{
  if ( (heatmaps16Bit->width == heatmaps8Bit->width) && (heatmaps16Bit->height == heatmaps8Bit->height) )
  {
    int width         = heatmaps16Bit->width;
    int height        = heatmaps16Bit->height;
    int channels16Bit = heatmaps16Bit->channels;
    int channels8Bit  = heatmaps8Bit->channels;

    // Source
    signed short * pixels16Bit = (signed short *) heatmaps16Bit->pixels;
    signed short * mem16BBase  =  pixels16Bit + (width * height * channels16Bit * targetImageNumber) + source16BitChannel;

    // Target
    signed char  *  pixels8Bit = (signed char *)  heatmaps8Bit->pixels;
    signed char  *  mem8BBase  =  pixels8Bit  + (width * height * channels8Bit  * targetImageNumber) + target8BitChannel;

    // First pass: find min and max depth values in the interior region
    signed short min_val = SHRT_MAX;
    signed short max_val = SHRT_MIN;

    for (int y = borderY; y < height - borderY; y++)
    {
      signed short * mem16B = mem16BBase + (y * width * channels16Bit) + (borderX * channels16Bit);
      for (int x = borderX; x < width - borderX; x++)
      {
        signed short val = *mem16B;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        mem16B += channels16Bit;
      }
    }

    // Handle case where all depth values are the same
    float range = (float)(max_val - min_val);
    if (range == 0.0f) range = 1.0f; // Avoid division by zero

    // Precompute per-pass constants: fused scale+offset so the inner loop
    // reduces to one fmul + one fadd instead of subtract + divide + fma.
    // mapped = (val - min_val) / range * 240 - 120
    //        = val * scale + bias
    float inv_range = 1.0f / range;
    float scale     = 240.0f * inv_range;
    float bias      = -120.0f - (float)min_val * scale;

    // Second pass: map depth values to [-120, 120]
    for (int y = borderY; y < height - borderY; y++)
    {
      signed short * mem16B = mem16BBase + (y * width * channels16Bit) + (borderX * channels16Bit);
      signed char  * mem8B  = mem8BBase  + (y * width * channels8Bit)  + (borderX * channels8Bit);

      for (int x = borderX; x < width - borderX; x++)
      {
        signed short val = *mem16B;
        mem16B += channels16Bit;

        float mapped;
        if (min_val == max_val)
        {
          mapped = 0.0f; // Neutral value for uniform depth
        } else
        {
          // Linear mapping: [min_val, max_val] -> [-120, 120]
          // One fmul + one fadd; no per-pixel division.
          mapped = (float)val * scale + bias;

          // Clamp to ensure values stay within [-120, 120]
          if (mapped < -120.0f) { mapped = -120.0f; } else
          if (mapped > 120.0f)  { mapped = 120.0f;  }
        }

        // Round half-away-from-zero: add ±0.5 then truncate.
        // Written as a ternary so GCC emits blendvps/cvttss2si — no libm call.
        *mem8B = (signed char)(mapped + (mapped >= 0.0f ? 0.5f : -0.5f));
        mem8B += channels8Bit;
      }
    }

  } else
  {
      fprintf(stderr,"Cannot copy16BitHeatmapTo8BitHeatmap( from %u 16bit to %u 8bit )\n",source16BitChannel,target8BitChannel);
      fprintf(stderr,"Incoherrent dimensions ( 16Bit %ux%u / 8bit %ux%u) \n",heatmaps16Bit->width,heatmaps16Bit->height,heatmaps8Bit->width,heatmaps8Bit->height);
      abort();
  }
}





/*
    Allocates a new Image structure with pixel buffer.
    bits = 8 or 16
    channels = 1
*/
static struct Image * allocateSingleChannelImage(unsigned int w, unsigned int h, unsigned int bits)
{
    struct Image * img = malloc(sizeof(struct Image));
    if (!img) return 0;

    memset(img, 0, sizeof(struct Image));

    img->width  = w;
    img->height = h;
    img->channels = 1;
    img->bitsperpixel = bits;
    img->image_size = w * h * (bits/8);

    img->pixels = malloc(img->image_size);
    if (!img->pixels)
    {
        free(img);
        return 0;
    }

    return img;
}


/*
    Main function that:
    - Loads the PNG containing segmentation + depth (RGB)
    - Allocates two new Image objects:
         seg_img  = 8-bit class ID
         depth_img = 16-bit depth image
    - Extracts the data into the new images
*/
static int splitSegmentationAndDepthFromSingleFileGeneric(
                                                          struct Image * img,
                                                          struct Image ** seg_img_out,
                                                          struct Image ** depth_img_out
                                                         )
{
    *seg_img_out   = 0;
    *depth_img_out = 0;

    // -------------------------------------------------------
    // 1. Load the raw PNG using your own provided loader
    // -------------------------------------------------------
    if (!img || !img->pixels)
    {
        fprintf(stderr, "splitSegmentationAndDepthFromSingleFileGeneric failed to load\n");
        return 0;
    }

    if (img->channels < 3 || img->bitsperpixel != 8)
    {
        fprintf(stderr, "ERROR: Expected 8-bit RGB file\n");
        return 0;
    }

    unsigned int W = img->width;
    unsigned int H = img->height;

    const unsigned char * src = img->pixels;

    // -------------------------------------------------------
    // 2. Allocate output images
    // -------------------------------------------------------
    struct Image * seg   = allocateSingleChannelImage(W, H, 8);
    struct Image * depth = allocateSingleChannelImage(W, H, 16);

    if (!seg || !depth)
    {
        free(seg);
        free(depth);
        fprintf(stderr, "Allocation failed.\n");
        return 0;
    }

    unsigned char  * seg_pixels   = seg->pixels;
    unsigned short * depth_pixels = (unsigned short*) depth->pixels;

    // -------------------------------------------------------
    // 3. Extract segmentation + depth values
    // -------------------------------------------------------
    #define CHANNEL_R 0
    #define CHANNEL_G 2
    #define CHANNEL_B 1

    unsigned int n = W * H;

#if INTEL_OPTIMIZATIONS
    // AVX2 / SSE4.1 stride-3 deinterleave — 16 pixels (48 bytes) per iteration.
    //
    // Pixel layout in memory: [R0,B0,G0, R1,B1,G1, ...]
    //   CHANNEL_R=0 (position 0) → seg byte
    //   CHANNEL_B=1 (position 1) → depth low byte
    //   CHANNEL_G=2 (position 2) → depth high byte
    //   depth[i] = (G[i] << 8) | B[i]
    //
    // We load three consecutive 16-byte blocks a/b/c covering pixels i..i+15:
    //   a = src[0..15],  b = src[16..31],  c = src[32..47]
    //
    // Channel R at relative byte positions:  a:{0,3,6,9,12,15}  b:{2,5,8,11,14}  c:{1,4,7,10,13}
    // Channel B at relative byte positions:  a:{1,4,7,10,13}    b:{0,3,6,9,12,15} c:{2,5,8,11,14}
    // Channel G at relative byte positions:  a:{2,5,8,11,14}    b:{1,4,7,10,13}   c:{0,3,6,9,12,15}
    //
    // pshufb compacts each group to the low bytes of the result; two byte-shift-and-OR
    // operations then stitch the three partial results into one contiguous 16-byte vector.

    // pshufb masks stored as byte arrays (portable C99; -1 = 0xFF zeroes that output byte).
    // R masks (position 0 per pixel):
    //   a → 6 bytes at output[0..5];   pshufb indices: 0,3,6,9,12,15
    //   b → 5 bytes at output[0..4];   pshufb indices: 2,5,8,11,14  (shift left 6 before OR)
    //   c → 5 bytes at output[0..4];   pshufb indices: 1,4,7,10,13  (shift left 11 before OR)
    static const signed char _mR_a[16] = { 0, 3, 6, 9,12,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    static const signed char _mR_b[16] = { 2, 5, 8,11,14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    static const signed char _mR_c[16] = { 1, 4, 7,10,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    // B masks (position 1 per pixel):
    //   a → 5 bytes;  b → 6 bytes (shift 5);  c → 5 bytes (shift 11)
    static const signed char _mB_a[16] = { 1, 4, 7,10,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    static const signed char _mB_b[16] = { 0, 3, 6, 9,12,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    static const signed char _mB_c[16] = { 2, 5, 8,11,14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    // G masks (position 2 per pixel):
    //   a → 5 bytes;  b → 5 bytes (shift 5);  c → 6 bytes (shift 10)
    static const signed char _mG_a[16] = { 2, 5, 8,11,14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    static const signed char _mG_b[16] = { 1, 4, 7,10,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    static const signed char _mG_c[16] = { 0, 3, 6, 9,12,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };

    // Load masks into registers once (compiler will keep them in regs across loop iterations)
    __m128i shuf_R_a = _mm_loadu_si128((const __m128i*)_mR_a);
    __m128i shuf_R_b = _mm_loadu_si128((const __m128i*)_mR_b);
    __m128i shuf_R_c = _mm_loadu_si128((const __m128i*)_mR_c);
    __m128i shuf_B_a = _mm_loadu_si128((const __m128i*)_mB_a);
    __m128i shuf_B_b = _mm_loadu_si128((const __m128i*)_mB_b);
    __m128i shuf_B_c = _mm_loadu_si128((const __m128i*)_mB_c);
    __m128i shuf_G_a = _mm_loadu_si128((const __m128i*)_mG_a);
    __m128i shuf_G_b = _mm_loadu_si128((const __m128i*)_mG_b);
    __m128i shuf_G_c = _mm_loadu_si128((const __m128i*)_mG_c);

    unsigned int i = 0;
    for (; i + 16 <= n; i += 16)
    {
        const unsigned char *s = src + i * 3;

        __m128i a = _mm_loadu_si128((const __m128i*)(s +  0));
        __m128i b = _mm_loadu_si128((const __m128i*)(s + 16));
        __m128i c = _mm_loadu_si128((const __m128i*)(s + 32));

        // --- R (seg) channel: 6 from a, 5 from b, 5 from c ---
        __m128i seg_16 =
            _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(a, shuf_R_a),
                             _mm_slli_si128(_mm_shuffle_epi8(b, shuf_R_b), 6)),
                             _mm_slli_si128(_mm_shuffle_epi8(c, shuf_R_c), 11));

        // --- B channel (depth low byte): 5 from a, 6 from b, 5 from c ---
        __m128i B_16 =
            _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(a, shuf_B_a),
                             _mm_slli_si128(_mm_shuffle_epi8(b, shuf_B_b), 5)),
                             _mm_slli_si128(_mm_shuffle_epi8(c, shuf_B_c), 11));

        // --- G channel (depth high byte): 5 from a, 5 from b, 6 from c ---
        __m128i G_16 =
            _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(a, shuf_G_a),
                             _mm_slli_si128(_mm_shuffle_epi8(b, shuf_G_b), 5)),
                             _mm_slli_si128(_mm_shuffle_epi8(c, shuf_G_c), 10));

        // --- Write 16 seg bytes ---
        _mm_storeu_si128((__m128i*)(seg_pixels + i), seg_16);

        // --- Build depth: depth[j] = (G[j] << 8) | B[j] ---
        // Low 8 pixels: zero-extend bytes 0-7 of G/B to uint16, combine
        __m128i depth_lo = _mm_or_si128(
            _mm_slli_epi16(_mm_cvtepu8_epi16(G_16), 8),
                           _mm_cvtepu8_epi16(B_16));
        _mm_storeu_si128((__m128i*)(depth_pixels + i),     depth_lo);

        // High 8 pixels: shift G/B right by 8 bytes to bring bytes 8-15 to low end, then same
        __m128i depth_hi = _mm_or_si128(
            _mm_slli_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(G_16, 8)), 8),
                           _mm_cvtepu8_epi16(_mm_srli_si128(B_16, 8)));
        _mm_storeu_si128((__m128i*)(depth_pixels + i + 8), depth_hi);
    }

    // Scalar tail for the remaining < 16 pixels
    for (; i < n; i++)
    {
        unsigned int idx = i * 3;
        seg_pixels[i]   = src[idx + CHANNEL_R];
        depth_pixels[i] = ((unsigned short)src[idx + CHANNEL_G] << 8)
                        |  (unsigned short)src[idx + CHANNEL_B];
    }
#else
    // Flat scalar loop
    for (unsigned int i = 0; i < n; i++)
    {
        unsigned int idx = i * 3;
        seg_pixels[i]   = src[idx + CHANNEL_R];
        depth_pixels[i] = ((unsigned short)src[idx + CHANNEL_G] << 8)
                        |  (unsigned short)src[idx + CHANNEL_B];
    }
#endif

    // -------------------------------------------------------
    // 4. Return both output images
    // -------------------------------------------------------
    *seg_img_out   = seg;
    *depth_img_out = depth;

    destroyImage(img);
    return 1;
}



static int splitSegmentationAndDepthFromSingleFilePNG(
                                                          const char * filename,
                                                          struct Image ** seg_img_out,
                                                          struct Image ** depth_img_out
                                                        )
{
    *seg_img_out   = 0;
    *depth_img_out = 0;

    // -------------------------------------------------------
    // 1. Load the raw PNG using your own provided loader
    // -------------------------------------------------------
    struct Image * img = readImage(filename, PNG_CODEC, 0);
    if (!img || !img->pixels)
    {
        fprintf(stderr, "Failed to load PNG: %s\n", filename);
        return 0;
    }

    if (img->channels < 3 || img->bitsperpixel != 8)
    {
        fprintf(stderr, "ERROR: Expected 8-bit RGB PNG\n");
        return 0;
    }


   return splitSegmentationAndDepthFromSingleFileGeneric(img,seg_img_out,depth_img_out);

}



static int splitSegmentationAndDepthFromSingleFilePZP(
                                                          const char * filename,
                                                          struct Image ** seg_img_out,
                                                          struct Image ** depth_img_out
                                                        )
{
    *seg_img_out   = 0;
    *depth_img_out = 0;

    // -------------------------------------------------------
    // 1. Load the raw PZP using your own provided loader
    // -------------------------------------------------------
    struct Image * img = readImage(filename, PZP_CODEC, 0);
    if (!img || !img->pixels)
    {
        fprintf(stderr, "Failed to load PZP: %s\n", filename);
        return 0;
    }

    if (img->channels < 3 || img->bitsperpixel != 24)
    {
        fprintf(stderr, "ERROR: Expected 24-bit RGB PZP\n");
        fprintf(stderr, "We have %u channels \n",img->channels);
        fprintf(stderr, "We have %u bitsperpixel \n",img->bitsperpixel);
        return 0;
    }

   img->bitsperpixel = 8; // <- Recode bits per pixel how the next function expects it ..

   return splitSegmentationAndDepthFromSingleFileGeneric(img,seg_img_out,depth_img_out);

}


#ifdef __cplusplus
}
#endif

#endif // CONVERSIONS_H_INCLUDED
