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


//DEPTH STEP 2 of PrepareBatch.c: swap the two bytes of every 16-bit sample in place. PNG (libpng without
//png_set_swap) and PZP decode 16-bit samples big-endian, so depth FILES need this to get native values;
//depth split from a multiplexed "all" image is already native and must not be swapped.
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


//DEPTH STEP 4 of PrepareBatch.c: 16-bit depth heatmap -> 8-bit depth heatmap of the same sample.
//  in : signed short depth (resize16BitImageTo16BitHeatmapWithBorders1Ch output, -32767..32768, larger = nearer)
//  out: signed char  depth in [-120, 120]: the content's min (farthest) -> -120, its max (nearest) -> +120,
//       linear in between, rounded half away from zero. A constant depth map becomes 0 everywhere.
//  Only the content inside the letterbox borders (borderX/borderY) is scanned and written, so the borders keep
//  the cleared -120. The 8-bit depth is the SIGNED input of the normals (STEP 5) and depth levels (STEP 6).
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
    Channel layout of the combined ("all") files, as written by depthanythingv3/append16bitToSAM3.py and
    compressSegmentDepth.py: channel 0 (R) = segmentation label, 1 (G) = depth high byte, 2 (B) = depth low
    byte. The DataLoader's default (db_set_combined_layout(), DataLoader.py combinedChannels=(0,1,2)).
    Until 2026-09-23 this function read channel 2 as the high byte and 1 as the low byte (it assumed an
    [R,B,G] memory order), i.e. depth from combined files was byte-swapped -- and the unconditional
    swap16bitEndianness() in PrepareBatch.c (meant for big-endian PNG/PZP depth files) swapped it back, so
    the depth targets were correct. Commit 8adc694 (2026-09-23) fixed the channel order here but kept that
    swap, which then scrambled every multiplexed depth map into byte-wrapped noise until 2026-10-06, when
    PrepareBatch.c stopped swapping multiplexed depth (DEPTH STEP 2). Keep the two in sync: this function
    returns NATIVE-endian depth.
*/
#define COMBINED_SEG_CHANNEL        0
#define COMBINED_DEPTH_HIGH_CHANNEL 1
#define COMBINED_DEPTH_LOW_CHANNEL  2

#if INTEL_OPTIMIZATIONS
/*
    Byte `pos` (0, 1 or 2) of each of the 16 pixels in a 48-byte stride-3 block a|b|c, in order.
    The shifts of _mm_slli_si128 must be compile-time constants, hence one case per position.
*/
static inline __m128i deinterleave3(__m128i a, __m128i b, __m128i c, unsigned int pos)
{
    // position 0: a{0,3,..,15} b{2,5,..,14} c{1,4,..,13}   (6 + 5 + 5 bytes)
    static const signed char m0a[16] = { 0, 3, 6, 9,12,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    static const signed char m0b[16] = { 2, 5, 8,11,14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    static const signed char m0c[16] = { 1, 4, 7,10,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    // position 1: a{1,4,..,13} b{0,3,..,15} c{2,5,..,14}   (5 + 6 + 5 bytes)
    static const signed char m1a[16] = { 1, 4, 7,10,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    static const signed char m1b[16] = { 0, 3, 6, 9,12,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    static const signed char m1c[16] = { 2, 5, 8,11,14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    // position 2: a{2,5,..,14} b{1,4,..,13} c{0,3,..,15}   (5 + 5 + 6 bytes)
    static const signed char m2a[16] = { 2, 5, 8,11,14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    static const signed char m2b[16] = { 1, 4, 7,10,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    static const signed char m2c[16] = { 0, 3, 6, 9,12,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 };
    switch (pos)
    {
      case 0:
        return _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, _mm_loadu_si128((const __m128i*)m0a)),
                                         _mm_slli_si128(_mm_shuffle_epi8(b, _mm_loadu_si128((const __m128i*)m0b)), 6)),
                                         _mm_slli_si128(_mm_shuffle_epi8(c, _mm_loadu_si128((const __m128i*)m0c)), 11));
      case 1:
        return _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, _mm_loadu_si128((const __m128i*)m1a)),
                                         _mm_slli_si128(_mm_shuffle_epi8(b, _mm_loadu_si128((const __m128i*)m1b)), 5)),
                                         _mm_slli_si128(_mm_shuffle_epi8(c, _mm_loadu_si128((const __m128i*)m1c)), 11));
      default:
        return _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, _mm_loadu_si128((const __m128i*)m2a)),
                                         _mm_slli_si128(_mm_shuffle_epi8(b, _mm_loadu_si128((const __m128i*)m2b)), 5)),
                                         _mm_slli_si128(_mm_shuffle_epi8(c, _mm_loadu_si128((const __m128i*)m2c)), 10));
    }
}
#endif

/*
    Main function that:
    - Takes the image containing segmentation + depth (8-bit, 3 or 4 channels)
    - Allocates two new Image objects:
         seg_img  = 8-bit class ID                       (channel segChannel)
         depth_img = 16-bit depth image                  (channel depthHighChannel * 256 + channel depthLowChannel)
    - Extracts the data into the new images; img is consumed (destroyed)
    DEPTH STEP 0a of PrepareBatch.c. depth_img holds the depth VALUES in NATIVE byte order (unsigned, 0..65535,
    larger = nearer for Depth Anything depth): it must NOT go through swap16bitEndianness() like a depth file.
*/
static int splitSegmentationAndDepthFromSingleFileGeneric(
                                                          struct Image * img,
                                                          struct Image ** seg_img_out,
                                                          struct Image ** depth_img_out,
                                                          unsigned int segChannel,
                                                          unsigned int depthHighChannel,
                                                          unsigned int depthLowChannel
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
    if ( (segChannel >= img->channels) || (depthHighChannel >= img->channels) || (depthLowChannel >= img->channels) )
    {
        fprintf(stderr, "ERROR: combined layout channels %u,%u,%u don't fit a %u channel image\n", segChannel, depthHighChannel, depthLowChannel, img->channels);
        destroyImage(img);   // consumed, as on success
        return 0;
    }

    unsigned int W = img->width;
    unsigned int C = img->channels;   // pixel stride (3 for RGB, 4 for RGBA combined files)
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
    unsigned int n = W * H;
    unsigned int i = 0;

#if INTEL_OPTIMIZATIONS
    // SSSE3 / SSE4.1 stride-3 deinterleave, 16 pixels (48 bytes) per iteration (RGB images only):
    // the three wanted bytes of each pixel are gathered with deinterleave3(), then depth = (high << 8) | low.
    if (C == 3)
    {
        for (; i + 16 <= n; i += 16)
        {
            const unsigned char *s = src + i * 3;
            __m128i a = _mm_loadu_si128((const __m128i*)(s +  0));
            __m128i b = _mm_loadu_si128((const __m128i*)(s + 16));
            __m128i c = _mm_loadu_si128((const __m128i*)(s + 32));

            __m128i seg_16  = deinterleave3(a, b, c, segChannel);
            __m128i high_16 = deinterleave3(a, b, c, depthHighChannel);
            __m128i low_16  = deinterleave3(a, b, c, depthLowChannel);

            _mm_storeu_si128((__m128i*)(seg_pixels + i), seg_16);
            // pixels 0-7, then 8-15 (bytes 8-15 shifted down), zero-extended to 16 bits
            _mm_storeu_si128((__m128i*)(depth_pixels + i),
                             _mm_or_si128(_mm_slli_epi16(_mm_cvtepu8_epi16(high_16), 8), _mm_cvtepu8_epi16(low_16)));
            _mm_storeu_si128((__m128i*)(depth_pixels + i + 8),
                             _mm_or_si128(_mm_slli_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(high_16, 8)), 8),
                                          _mm_cvtepu8_epi16(_mm_srli_si128(low_16, 8))));
        }
    }
#endif

    // Scalar: the remaining pixels (all of them without INTEL_OPTIMIZATIONS or for 4-channel images)
    for (; i < n; i++)
    {
        const unsigned char * px = src + (size_t) i * C;
        seg_pixels[i]   = px[segChannel];
        depth_pixels[i] = (unsigned short) (((unsigned short) px[depthHighChannel] << 8) | px[depthLowChannel]);
    }

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


   return splitSegmentationAndDepthFromSingleFileGeneric(img,seg_img_out,depth_img_out,COMBINED_SEG_CHANNEL,COMBINED_DEPTH_HIGH_CHANNEL,COMBINED_DEPTH_LOW_CHANNEL);

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

   return splitSegmentationAndDepthFromSingleFileGeneric(img,seg_img_out,depth_img_out,COMBINED_SEG_CHANNEL,COMBINED_DEPTH_HIGH_CHANNEL,COMBINED_DEPTH_LOW_CHANNEL);

}


#ifdef __cplusplus
}
#endif

#endif // CONVERSIONS_H_INCLUDED
