#ifndef RESIZE_H_INCLUDED
#define RESIZE_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../codecs/image.h"


static int makeSureImageHas3Channels(struct Image * img)
{
  if (img!=0)
  {
    if (img->pixels!=0)
    {
     if (img->channels==3) { return 1; } //Image had already 3 channels
     else
     if (img->channels==1)
      {
          unsigned char * monoPixels = img->pixels;
          unsigned char * newPixels  = (unsigned char*) malloc(sizeof(unsigned char) * (img->width+1) * (img->height+1) * 3);
          if (newPixels!=0)
          {
           unsigned char * newPixelsPTR = newPixels;
           for (int y=0; y<img->height; y++)
            {
             for (int x=0; x<img->width; x++)
             {
               *newPixelsPTR = *monoPixels; newPixelsPTR++;
               *newPixelsPTR = *monoPixels; newPixelsPTR++;
               *newPixelsPTR = *monoPixels; newPixelsPTR++;
               monoPixels++;
             }
            }

           free(img->pixels);
           img->pixels       = newPixels;
           img->channels     = 3;
           img->bitsperpixel = 24;
           img->image_size   = img->width * img->height * img->channels;

           return 1;
          }//We had enough memory to allocate a 3 channel image
      } //Image has 1 channel and we corrected it!
    } //There is an image that has pixels
  } //There is an image to work with
 return 0;
}

static void resizeImage(struct Image *image, unsigned char *newPixels, unsigned int newWidth, unsigned int newHeight)
{
    float x_ratio = (float)image->width  / newWidth;
    float y_ratio = (float)image->height / newHeight;

    for (int y = 0; y < newHeight; y++)
    {
        // Calculate y-coordinate in the original image
        int nearest_y = (int)(y * y_ratio);

        for (int x = 0; x < newWidth; x++)
        {
            // Calculate x-coordinate in the original image
            int nearest_x = (int)(x * x_ratio);

            for (int c = 0; c < image->channels; c++)
            {
                // Calculate offsets for source and destination pixels
                int src_offset = ((nearest_y * image->width + nearest_x) * image->channels + c) * image->bitsperpixel / 8;
                int dest_offset = ((y * newWidth + x) * image->channels + c) * image->bitsperpixel / 8;

                // Copy pixel data directly
                *(unsigned char *)(newPixels + dest_offset) = *(unsigned char *)(image->pixels + src_offset);
            }
        }
    }
}

static void resizeImageWithBordersFast(struct Image *image, unsigned char *newPixels, unsigned int newWidth, unsigned int newHeight,
                            float*keypointXOffsetF, float *keypointYOffsetF, float *keypointXMultiplierF, float *keypointYMultiplierF)
{
    memset(newPixels,0,newWidth*newHeight*image->channels); //<- add black border

    // Calculate aspect ratios
    float aspect_ratio_original = (float)image->width / image->height;
    float aspect_ratio_target   = (float)newWidth / newHeight;

    // Determine the resizing factor and size for maintaining the aspect ratio
    unsigned int resizedWidth=0, resizedHeight=0;
    *keypointXOffsetF = 0;
    *keypointYOffsetF = 0;

    if (aspect_ratio_original > aspect_ratio_target)
    {
        resizedWidth = newWidth;
        resizedHeight = (unsigned int)((float) newWidth / aspect_ratio_original);
        *keypointYOffsetF = (float) (newHeight - resizedHeight) / 2;

    } else
    {
        resizedWidth = (unsigned int)(newHeight * aspect_ratio_original);
        resizedHeight = newHeight;
        *keypointXOffsetF = (float) (newWidth - resizedWidth) / 2;
    }

    // Calculate keypoint multipliers
    *keypointXMultiplierF = (float) resizedWidth  / image->width;
    *keypointYMultiplierF = (float) resizedHeight / image->height;

    // Resize the image while maintaining the aspect ratio and paste onto newPixels
    for (int y = 0; y < resizedHeight; y++)
    {
        for (int x = 0; x < resizedWidth; x++)
        {
            int dest_x = (int) (*keypointXOffsetF + (float) x);
            int dest_y = (int) (*keypointYOffsetF + (float) y);

            if (dest_x >= 0 && dest_x < newWidth && dest_y >= 0 && dest_y < newHeight)
            {
                for (int c = 0; c < image->channels; c++)
                {
                    int src_x = (int) x / *keypointXMultiplierF;
                    int src_y = (int) y / *keypointYMultiplierF;

                    int src_offset  = ((src_y * image->width + src_x) * image->channels + c) * image->bitsperpixel / 8;
                    int dest_offset = ((dest_y * newWidth + dest_x) * image->channels + c) * image->bitsperpixel / 8;
                    *(unsigned char *)(newPixels + dest_offset) = *(unsigned char *)(image->pixels + src_offset);
                }
            }
        }
    }
}


static void resizeImageWithBorders(struct Image *image, unsigned char *newPixels, unsigned int newWidth, unsigned int newHeight,
                                   float *keypointXOffset, float *keypointYOffset, float *keypointXMultiplier, float *keypointYMultiplier)
{
    if (newPixels == 0)
    {
        fprintf(stderr, "resizeImageWithBorders with no new pixels!\n");
        abort();
    }

    if (image->channels != 3)
    {
        fprintf(stderr, "resizeImageWithBorders assumes RGB images\n");
        abort();
    }

    if ((image->width == newWidth) && (image->height == newHeight))
    {
        // Don't need to do anything
        memcpy(newPixels, image->pixels, newWidth * newHeight * image->channels * sizeof(unsigned char));
        return;
    }

    // Initialize with black border
    memset(newPixels, 0, newWidth * newHeight * image->channels);

    // Calculate aspect ratios
    float aspect_ratio_original = (float)image->width / image->height;
    float aspect_ratio_target   = (float)newWidth / newHeight;

    // Determine the resizing factor and size for maintaining the aspect ratio
    unsigned int resizedWidth, resizedHeight;
    if (aspect_ratio_original > aspect_ratio_target)
    {
        resizedWidth = newWidth;
        resizedHeight = (unsigned int)(newWidth / aspect_ratio_original);
    }
    else
    {
        resizedWidth = (unsigned int)(newHeight * aspect_ratio_original);
        resizedHeight = newHeight;
    }

    // Calculate offsets for pasting resized image onto new image
    *keypointXOffset = (float)(newWidth - resizedWidth) / 2;
    *keypointYOffset = (float)(newHeight - resizedHeight) / 2;

    // Calculate keypoint multipliers
    *keypointXMultiplier = (float)resizedWidth / image->width;
    *keypointYMultiplier = (float)resizedHeight / image->height;

    // Pre-compute the bounds for the new image
    int offset_x = (int)(*keypointXOffset);
    int offset_y = (int)(*keypointYOffset);
    int max_x = offset_x + resizedWidth;
    int max_y = offset_y + resizedHeight;

    // Determine valid loop bounds for source image
    int max_src_x = image->width - 2;  // Maximum value for min_src_x to ensure max_src_x is within bounds
    int max_src_y = image->height - 2; // Maximum value for min_src_y to ensure max_src_y is within bounds

    unsigned long addr;
    unsigned int r_avg, g_avg, b_avg;

    // Resize the image while maintaining the aspect ratio and paste onto newPixels
    for (int y = offset_y; y < max_y; y++)
    {
        // Compute the source y-coordinate
        float src_y = (float)(y - offset_y) / *keypointYMultiplier;
        if (src_y < 0 || src_y > max_src_y) continue;  // Skip out-of-bound src_y values

        int min_src_y = (int)floor(src_y);
        int max_src_y = min_src_y + 1;

        for (int x = offset_x; x < max_x; x++)
        {
            // Compute the source x-coordinate
            float src_x = (float)(x - offset_x) / *keypointXMultiplier;
            if (src_x < 0 || src_x > max_src_x) continue;  // Skip out-of-bound src_x values

            int min_src_x = (int)floor(src_x);
            int max_src_x = min_src_x + 1;

            // Interpolate color values for each channel
            // Get the values for the 4 surrounding pixels
            addr = (min_src_y * image->width + min_src_x) * image->channels;
            r_avg = image->pixels[addr];
            g_avg = image->pixels[addr + 1];
            b_avg = image->pixels[addr + 2];

            addr = (min_src_y * image->width + max_src_x) * image->channels;
            r_avg += image->pixels[addr];
            g_avg += image->pixels[addr + 1];
            b_avg += image->pixels[addr + 2];

            addr = (max_src_y * image->width + min_src_x) * image->channels;
            r_avg += image->pixels[addr];
            g_avg += image->pixels[addr + 1];
            b_avg += image->pixels[addr + 2];

            addr = (max_src_y * image->width + max_src_x) * image->channels;
            r_avg += image->pixels[addr];
            g_avg += image->pixels[addr + 1];
            b_avg += image->pixels[addr + 2];

            // Average the colors
            r_avg /= 4;
            g_avg /= 4;
            b_avg /= 4;

            // Set the averaged color values to the newPixels array
            addr = (y * newWidth * image->channels) + (image->channels * x);
            newPixels[addr + 0] = (unsigned char)r_avg;
            newPixels[addr + 1] = (unsigned char)g_avg;
            newPixels[addr + 2] = (unsigned char)b_avg;
        }
    }
}


static void rearrangeInstanceCount(signed char *hmPixels, unsigned int hmWidth, unsigned int hmHeight, unsigned int hmChannels,unsigned int selectedChannel)
{
   #define ASSOCIATION_COUNT 256
   #define ASSOCIATION_ERASE_VALUE -120

   signed char associationCount = 120; // This can be either -119 or 1
   signed char associationNewValues[ASSOCIATION_COUNT];
   for (int i=0; i<ASSOCIATION_COUNT; i++)
   {
       associationNewValues[i] = ASSOCIATION_ERASE_VALUE;
   }

   //int x=0, y=0;
   unsigned long memlocation = selectedChannel;
   for (int i=0; i<hmWidth*hmHeight; i++)
   {
      int thisValueIndex = ((int) hmPixels[memlocation]) + 120;
      if (thisValueIndex==0)
        {
          //Fast path to continue loop
          memlocation += hmChannels;
          continue;
        }
      else
        {
           //Slow path to recount instances
           if (associationNewValues[thisValueIndex]!=ASSOCIATION_ERASE_VALUE)
           {
              hmPixels[memlocation] = associationNewValues[thisValueIndex];
           } else
           {
              associationNewValues[thisValueIndex] = associationCount;
              hmPixels[memlocation]                = associationCount;
              associationCount--;
           }
          memlocation += hmChannels;
        }

   }

}



static void copyThresholdedDepthHeatmapFull(
                                        signed char *hmPixels,
                                        unsigned int borderX,
                                        unsigned int borderY,
                                        unsigned int hmWidth,
                                        unsigned int hmHeight,
                                        unsigned int hmChannels,
                                        unsigned int targetImageNumber,
                                        unsigned int depthChannel,
                                        unsigned int thresholdChannel,
                                        unsigned int levels
                                       )
{
   //fprintf(stderr,"copyThresholdedDepthHeatmap(%u->%u)..\n",depthChannel,thresholdChannel);
   signed char*  heatmapSource8Bit      = hmPixels  + (hmWidth * hmHeight * hmChannels * targetImageNumber) + depthChannel;
   signed char*  heatmapDestination8Bit = hmPixels  + (hmWidth * hmHeight * hmChannels * targetImageNumber) + thresholdChannel;

   signed char hit[4] ={0};
   const signed char mins[4]={64,  0,  -64, -128};
   const signed char maxs[4]={127, 64,   0, -64};

   //TODO: Also adhere to borders of image!
   //Using borderX and borderY
   for (unsigned int i=0; i<hmWidth*hmHeight; i++)
   {
    hit[0] = (signed char) (mins[0]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[0]);
    hit[1] = (signed char) (mins[1]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[1]);
    hit[2] = (signed char) (mins[2]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[2]);
    hit[3] = (signed char) (mins[3]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[3]);
    heatmapSource8Bit       += hmChannels;

    signed char*  heatmapSource8BitRow = heatmapDestination8Bit;
    *heatmapSource8BitRow = (signed char) (hit[0] * 240) - 120;
    heatmapSource8BitRow  += 1;
    *heatmapSource8BitRow = (signed char) (hit[1] * 240) - 120;
    heatmapSource8BitRow  += 1;
    *heatmapSource8BitRow = (signed char) (hit[2] * 240) - 120;
    heatmapSource8BitRow  += 1;
    *heatmapSource8BitRow = (signed char) (hit[3] * 240) - 120;
    heatmapDestination8Bit += hmChannels;
   }
}

static void copyThresholdedDepthHeatmap_4Levels(
                                        signed char *hmPixels,
                                        unsigned int borderX,
                                        unsigned int borderY,
                                        unsigned int hmWidth,
                                        unsigned int hmHeight,
                                        unsigned int hmChannels,
                                        unsigned int targetImageNumber,
                                        unsigned int depthChannel,
                                        unsigned int thresholdChannel
                                       )
{
   //fprintf(stderr,"copyThresholdedDepthHeatmap(%u->%u)..\n",depthChannel,thresholdChannel);
   signed char*  heatmapSource8BitInitial      = hmPixels  + (hmWidth * hmHeight * hmChannels * targetImageNumber) + depthChannel;
   signed char*  heatmapDestination8BitInitial = hmPixels  + (hmWidth * hmHeight * hmChannels * targetImageNumber) + thresholdChannel;

   signed char hit[4] ={0};
   const signed char mins[4]={64,  0,  -64, -128};
   const signed char maxs[4]={127, 64,   0, -64};

   //fprintf(stderr,"Border x %u / Border y %u\n",borderX,borderY);
   for (unsigned int y=borderY; y<hmHeight-borderY; y++)
   {
    signed char* heatmapSource8Bit      = heatmapSource8BitInitial       + (y * hmWidth * hmChannels) + (borderX * hmChannels);
    signed char* heatmapDestination8Bit = heatmapDestination8BitInitial  + (y * hmWidth * hmChannels) + (borderX * hmChannels);

    for (unsigned int x=borderX; x<hmWidth-borderX; x++)
    {
     hit[0] = (signed char) (mins[0]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[0]);
     hit[1] = (signed char) (mins[1]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[1]);
     hit[2] = (signed char) (mins[2]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[2]);
     hit[3] = (signed char) (mins[3]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[3]);
     heatmapSource8Bit       += hmChannels;

     signed char*  heatmapSource8BitRow = heatmapDestination8Bit;
     *heatmapSource8BitRow = (signed char) (hit[0] * MAXV_MINUS_MMINV) - MAXV;
     heatmapSource8BitRow  += 1;
     *heatmapSource8BitRow = (signed char) (hit[1] * MAXV_MINUS_MMINV) - MAXV;
     heatmapSource8BitRow  += 1;
     *heatmapSource8BitRow = (signed char) (hit[2] * MAXV_MINUS_MMINV) - MAXV;
     heatmapSource8BitRow  += 1;
     *heatmapSource8BitRow = (signed char) (hit[3] * MAXV_MINUS_MMINV) - MAXV;
     heatmapDestination8Bit += hmChannels;
    } // X
   } // Y
}


static void copyThresholdedDepthHeatmap_2Levels(
                                        signed char *hmPixels,
                                        unsigned int borderX,
                                        unsigned int borderY,
                                        unsigned int hmWidth,
                                        unsigned int hmHeight,
                                        unsigned int hmChannels,
                                        unsigned int targetImageNumber,
                                        unsigned int depthChannel,
                                        unsigned int thresholdChannel
                                       )
{
   //fprintf(stderr,"copyThresholdedDepthHeatmap(%u->%u)..\n",depthChannel,thresholdChannel);
   signed char*  heatmapSource8BitInitial      = hmPixels  + (hmWidth * hmHeight * hmChannels * targetImageNumber) + depthChannel;
   signed char*  heatmapDestination8BitInitial = hmPixels  + (hmWidth * hmHeight * hmChannels * targetImageNumber) + thresholdChannel;

   signed char hit[4] ={0};
   const signed char mins[4]={64,  0,  -64, -128};
   const signed char maxs[4]={127, 64,   0, -64};

   //fprintf(stderr,"Border x %u / Border y %u\n",borderX,borderY);
   for (unsigned int y=borderY; y<hmHeight-borderY; y++)
   {
    signed char* heatmapSource8Bit      = heatmapSource8BitInitial       + (y * hmWidth * hmChannels) + (borderX * hmChannels);
    signed char* heatmapDestination8Bit = heatmapDestination8BitInitial  + (y * hmWidth * hmChannels) + (borderX * hmChannels);

    for (unsigned int x=borderX; x<hmWidth-borderX; x++)
    {
     hit[0] = (signed char) (mins[0]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[0]);
     hit[1] = (signed char) (mins[1]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[1]);
     //hit[2] = (signed char) (mins[2]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[2]);
     //hit[3] = (signed char) (mins[3]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[3]);
     heatmapSource8Bit       += hmChannels;

     signed char*  heatmapSource8BitRow = heatmapDestination8Bit;
     *heatmapSource8BitRow = (signed char) (hit[0] * MAXV_MINUS_MMINV) - MAXV;
     heatmapSource8BitRow  += 1;
     *heatmapSource8BitRow = (signed char) (hit[1] * MAXV_MINUS_MMINV) - MAXV;
     //heatmapSource8BitRow  += 1;
     //*heatmapSource8BitRow = (signed char) (hit[2] * MAXV_MINUS_MMINV) - MAXV;
     //heatmapSource8BitRow  += 1;
     //*heatmapSource8BitRow = (signed char) (hit[3] * MAXV_MINUS_MMINV) - MAXV;
     heatmapDestination8Bit += hmChannels;
    } // X
   } // Y
}






static void copyThresholdedDepthHeatmap_1Level(
                                        signed char *hmPixels,
                                        unsigned int borderX,
                                        unsigned int borderY,
                                        unsigned int hmWidth,
                                        unsigned int hmHeight,
                                        unsigned int hmChannels,
                                        unsigned int targetImageNumber,
                                        unsigned int depthChannel,
                                        unsigned int thresholdChannel
                                       )
{
   //fprintf(stderr,"copyThresholdedDepthHeatmap(%u->%u)..\n",depthChannel,thresholdChannel);
   signed char*  heatmapSource8BitInitial      = hmPixels  + (hmWidth * hmHeight * hmChannels * targetImageNumber) + depthChannel;
   signed char*  heatmapDestination8BitInitial = hmPixels  + (hmWidth * hmHeight * hmChannels * targetImageNumber) + thresholdChannel;

   signed char hit[4] ={0};
   const signed char mins[4]={64,  0,  -64, -128};
   const signed char maxs[4]={127, 64,   0, -64};

   //fprintf(stderr,"Border x %u / Border y %u\n",borderX,borderY);
   for (unsigned int y=borderY; y<hmHeight-borderY; y++)
   {
    signed char* heatmapSource8Bit      = heatmapSource8BitInitial       + (y * hmWidth * hmChannels) + (borderX * hmChannels);
    signed char* heatmapDestination8Bit = heatmapDestination8BitInitial  + (y * hmWidth * hmChannels) + (borderX * hmChannels);

    for (unsigned int x=borderX; x<hmWidth-borderX; x++)
    {
     hit[0] = (signed char) (mins[0]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[0]);
     //hit[1] = (signed char) (mins[1]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[1]);
     //hit[2] = (signed char) (mins[2]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[2]);
     //hit[3] = (signed char) (mins[3]<=*heatmapSource8Bit) * (*heatmapSource8Bit<=maxs[3]);
     heatmapSource8Bit       += hmChannels;

     signed char*  heatmapSource8BitRow = heatmapDestination8Bit;
     *heatmapSource8BitRow = (signed char) (hit[0] * MAXV_MINUS_MMINV) - MAXV;
     //heatmapSource8BitRow  += 1;
     //*heatmapSource8BitRow = (signed char) (hit[1] * MAXV_MINUS_MMINV) - MAXV;
     //heatmapSource8BitRow  += 1;
     //*heatmapSource8BitRow = (signed char) (hit[2] * MAXV_MINUS_MMINV) - MAXV;
     //heatmapSource8BitRow  += 1;
     //*heatmapSource8BitRow = (signed char) (hit[3] * MAXV_MINUS_MMINV) - MAXV;
     heatmapDestination8Bit += hmChannels;
    } // X
   } // Y
}

static void copyThresholdedDepthHeatmap(
                                        signed char *hmPixels,
                                        unsigned int borderX,
                                        unsigned int borderY,
                                        unsigned int hmWidth,
                                        unsigned int hmHeight,
                                        unsigned int hmChannels,
                                        unsigned int targetImageNumber,
                                        unsigned int depthChannel,
                                        unsigned int thresholdChannel,
                                        unsigned int levels
                                       )
{
   switch (levels)
       {
        case 1:
         copyThresholdedDepthHeatmap_1Level(hmPixels,borderX,borderY,hmWidth,hmHeight,hmChannels,targetImageNumber,depthChannel,thresholdChannel);
        break;

        case 2:
         copyThresholdedDepthHeatmap_2Levels(hmPixels,borderX,borderY,hmWidth,hmHeight,hmChannels,targetImageNumber,depthChannel,thresholdChannel);
        break;

        case 4:
         copyThresholdedDepthHeatmap_4Levels(hmPixels,borderX,borderY,hmWidth,hmHeight,hmChannels,targetImageNumber,depthChannel,thresholdChannel);
        break;

        default:
            fprintf(stderr,"Please implement copyThresholdedDepthHeatmap for %u levels \n",levels);
        break;
       };
}







static void resizeInstanceImageWithBorders1Channel(
    struct Image *image,
    signed char *newPixels,
    unsigned int newWidth,
    unsigned int newHeight,
    unsigned int newChannels,
    unsigned int selectedChannel,
    unsigned int maxID,
    float *keypointXOffset,
    float *keypointYOffset,
    float *keypointXMultiplier,
    float *keypointYMultiplier
)
{
    if (image->channels != 1) { fprintf(stderr,"Segmentation 1ch: Invalid input image\n"); return; }

    float aspect_ratio_original = (float)image->width / image->height;
    float aspect_ratio_target   = (float)newWidth / newHeight;

    unsigned int resizedWidth, resizedHeight;
    if (aspect_ratio_original > aspect_ratio_target)
    {
        resizedWidth  = newWidth;
        resizedHeight = (unsigned int)(newWidth / aspect_ratio_original);
    }
    else
    {
        resizedWidth  = (unsigned int)(newHeight * aspect_ratio_original);
        resizedHeight = newHeight;
    }

    *keypointXOffset     = (float)(newWidth  - resizedWidth)  / 2;
    *keypointYOffset     = (float)(newHeight - resizedHeight) / 2;
    *keypointXMultiplier = (float)resizedWidth  / image->width;
    *keypointYMultiplier = (float)resizedHeight / image->height;

    int start_x = (int)(*keypointXOffset);
    int start_y = (int)(*keypointYOffset);
    int end_x   = start_x + resizedWidth;
    int end_y   = start_y + resizedHeight;

    if (start_x < 0) start_x = 0;
    if (start_y < 0) start_y = 0;
    if (end_x > newWidth)  end_x = newWidth;
    if (end_y > newHeight) end_y = newHeight;

    int max_src_x = image->width  - 1;
    int max_src_y = image->height - 1;

    #if BRANCHLESS_SEGMENTATION_SEPERATION
      signed char rememberThisBecauseItWillGetOverwritten = newPixels[0];
    #endif // BRANCHLESS_SEGMENTATION_SEPERATION

    for (int y = start_y; y < end_y; y++)
    {
        float src_y = (y - *keypointYOffset) / *keypointYMultiplier;
        int min_src_y = (int)src_y;
        if (min_src_y < 0) min_src_y = 0;
        if (min_src_y > max_src_y) min_src_y = max_src_y;

        for (int x = start_x; x < end_x; x++)
        {
            float src_x = (x - *keypointXOffset) / *keypointXMultiplier;
            int min_src_x = (int)src_x;
            if (min_src_x < 0) min_src_x = 0;
            if (min_src_x > max_src_x) min_src_x = max_src_x;

            unsigned long srclocation = (min_src_y * image->width * image->channels) + (min_src_x * image->channels);

            //This is the classID !
            unsigned char classificationID = image->pixels[srclocation];

            unsigned long memlocation = (y * newWidth * newChannels) + (x * newChannels) + selectedChannel;

            /* -------------------------------------------------------
               SEGMENTATION CLASS HANDLING (unchanged)
               ------------------------------------------------------- */

            #if SEPERATE_SEGMENTATIONS_INTO_DIFFERENT_CHANNELS
            unsigned char offsetFromClassification = classificationID -1 ;
            //First segmentation class (person) has a value of 1 not 0
            //This way we send unsegmented classes ( with classificationID = 0 ) to 255 and they will get filtered

            unsigned char overflow_protector = (offsetFromClassification < VALID_SEGMENTATIONS);
            //---------------------------------------------------------------------------------------------
             #if BRANCHLESS_SEGMENTATION_SEPERATION
              newPixels[(memlocation * overflow_protector) + (unsigned char) (offsetFromClassification * overflow_protector)] = (signed char) MAXV;
             #else
              if (classificationID!=0)
                { newPixels[memlocation + (unsigned char) (offsetFromClassification * overflow_protector)] = (signed char) MAXV; }
             #endif // BRANCHLESS_SEGMENTATION_SEPERATION
            #else
             newPixels[memlocation] = (signed char) MINV + classificationID;
            #endif // SEPERATE_SEGMENTATIONS_INTO_DIFFERENT_CHANNELS

            /* instance data ignored */
        }
    }

    #if BRANCHLESS_SEGMENTATION_SEPERATION
      newPixels[0] = rememberThisBecauseItWillGetOverwritten;
    #endif // BRANCHLESS_SEGMENTATION_SEPERATION
}









static void resizeInstanceImageWithBorders(
    struct Image *image,
    signed char *newPixels,
    unsigned int newWidth,
    unsigned int newHeight,
    unsigned int newChannels,
    unsigned int selectedChannel,
    unsigned int maxID,
    float *keypointXOffset,
    float *keypointYOffset,
    float *keypointXMultiplier,
    float *keypointYMultiplier
)
{
    // Assuming already initialized background
    //const signed char MAXV  =  120;
    //const signed char MINV  = -120;

    // Calculate aspect ratios
    float aspect_ratio_original = (float)image->width / image->height;
    float aspect_ratio_target   = (float)newWidth / newHeight;

    // Determine the resizing factor and size for maintaining the aspect ratio
    unsigned int resizedWidth, resizedHeight;
    if (aspect_ratio_original > aspect_ratio_target)
    {
        resizedWidth  = newWidth;
        resizedHeight = (unsigned int)(newWidth / aspect_ratio_original);
    }
    else
    {
        resizedWidth  = (unsigned int)(newHeight * aspect_ratio_original);
        resizedHeight = newHeight;
    }

    // Calculate offsets for pasting resized image onto new image
    *keypointXOffset = (float)(newWidth - resizedWidth)   / 2;
    *keypointYOffset = (float)(newHeight - resizedHeight) / 2;

    // Calculate keypoint multipliers
    *keypointXMultiplier = (float)resizedWidth  / image->width;
    *keypointYMultiplier = (float)resizedHeight / image->height;

    // Pre-compute loop bounds to avoid checks inside the loop
    int start_x = (int)(*keypointXOffset);
    int start_y = (int)(*keypointYOffset);
    int end_x   = start_x + resizedWidth;
    int end_y   = start_y + resizedHeight;

    // Ensure that the loop bounds are within the valid range of the destination image
    if (start_x < 0) start_x = 0;
    if (start_y < 0) start_y = 0;
    if (end_x > newWidth)  end_x = newWidth;
    if (end_y > newHeight) end_y = newHeight;

    // Pre-compute max source coordinates to avoid boundary checks
    int max_src_x = image->width  - 1;
    int max_src_y = image->height - 1;

    // Resize the image while maintaining the aspect ratio and paste onto newPixels
    for (int y = start_y; y < end_y; y++)
    {
        float src_y = (y - *keypointYOffset) / *keypointYMultiplier;
        int min_src_y = (int) src_y;
        if (min_src_y < 0) min_src_y = 0;
        if (min_src_y > max_src_y) min_src_y = max_src_y;

        for (int x = start_x; x < end_x; x++)
        {
            // Calculate source coordinates
            float src_x = (float) (x - *keypointXOffset) / *keypointXMultiplier;

            // Calculate the closest source pixel
            int min_src_x = (int) src_x;

            // Ensure that min_src_x and min_src_y are within the valid range
            if (min_src_x < 0) min_src_x = 0;
            if (min_src_x > max_src_x) min_src_x = max_src_x;

            // Directly assign the closest pixel value without interpolation
            unsigned long srclocation      = (min_src_y * image->width * image->channels) + (min_src_x * image->channels);
            int textData                   = image->pixels[srclocation + 0];
            //unsigned char instanceData     = image->pixels[srclocation + 1];
            unsigned char classificationID = image->pixels[srclocation + 2];

            // Calculate entry point
            unsigned long memlocation   = (y * newWidth * newChannels) + (x * newChannels) + selectedChannel;

            // Add text entry point
            //newPixels[memlocation]      = MINV;  // Initialize to MINV
            //newPixels[memlocation]     += (signed char)((textData > 0) * 2 * MAXV);  // Adjust based on textData

            //----------------------------------------------------------
            // Add classes
            //----------------------------------------------------------

              // Populate Classification Data
              #if SEPERATE_SEGMENTATIONS_INTO_DIFFERENT_CHANNELS
              memlocation = (y * newWidth * newChannels) + (x * newChannels) + selectedChannel + 1 + (classificationID - 1);
              // Assign classification data if within valid ID range
              if (classificationID > 0 && classificationID <= maxID)
              {
                newPixels[memlocation] = (signed char)MAXV;
              }
              //----------------------------------------------------------
              //Add Text Class on top of unlabeled!
              //----------------------------------------------------------
              newPixels[memlocation]   = (signed char)(MINV + (textData > 0) * 2 * MAXV);  // Adjust based on textData
              #else
               int value;
               //This means populate all segmentation classes in a single heatmap
               value  = MINV + classificationID; //Normal, to recover add 120 (classes, bkg should have value 0 / person should have value 1)
               newPixels[memlocation + 1]  = (signed char) value;
              #endif // VALID_SEGMENTATIONS


            //----------------------------------------------------------
            // Populate Instance Data
            //----------------------------------------------------------
            //DISABLE INSTANCE DATA 1/2
            //#if ENABLE_INSTANCE_DATA == 1
            // int inst_value              = MINV + instanceData;
            // newPixels[memlocation + 2]  = (signed char) inst_value;
            //#endif // ENABLE_INSTANCE_DATA
        }
    }
}







static void resizeImageWithBorders3ChTo1ofXChOnlyRChannel(
                                                          struct Image *image,
                                                          signed char *newPixels,
                                                          unsigned int newWidth,
                                                          unsigned int newHeight,
                                                          unsigned int newChannels,
                                                          unsigned int selectedChannel,
                                                          float *keypointXOffset,
                                                          float *keypointYOffset,
                                                          float *keypointXMultiplier,
                                                          float *keypointYMultiplier
                                                         )
{
    //const int MAXV = 120;
    //const int MINV = -120;
    float scaleDown = (float)(MAXV - MINV) / 255.0;

    // Calculate aspect ratios
    float aspect_ratio_original = (float)image->width / image->height;
    float aspect_ratio_target = (float)newWidth / newHeight;

    // Determine the resizing factor and size for maintaining the aspect ratio
    unsigned int resizedWidth, resizedHeight;
    if (aspect_ratio_original > aspect_ratio_target)
    {
        resizedWidth = newWidth;
        resizedHeight = (unsigned int)(newWidth / aspect_ratio_original);
    } else
    {
        resizedWidth = (unsigned int)(newHeight * aspect_ratio_original);
        resizedHeight = newHeight;
    }

    // Calculate offsets for pasting resized image onto new image
    *keypointXOffset = (float)(newWidth - resizedWidth) / 2;
    *keypointYOffset = (float)(newHeight - resizedHeight) / 2;

    // Calculate keypoint multipliers
    *keypointXMultiplier = (float)resizedWidth / image->width;
    *keypointYMultiplier = (float)resizedHeight / image->height;

    // Resize the image while maintaining the aspect ratio and paste onto newPixels
    for (int y = 0; y < resizedHeight; y++)
     {
        for (int x = 0; x < resizedWidth; x++)
         {
            int dest_x = (int)(*keypointXOffset + (float)x);
            int dest_y = (int)(*keypointYOffset + (float)y);

            if (0 <= dest_x && dest_x < newWidth && 0 <= dest_y && dest_y < newHeight) {
                // Calculate source coordinates
                float src_x = (float)x / *keypointXMultiplier;
                float src_y = (float)y / *keypointYMultiplier;

                // Calculate the surrounding pixels for bilinear interpolation
                int src_x0 = (int)floor(src_x);
                int src_x1 = src_x0 + 1;
                int src_y0 = (int)floor(src_y);
                int src_y1 = src_y0 + 1;

                float x_weight = src_x - src_x0;
                float y_weight = src_y - src_y0;

                // Clamp the coordinates to the image dimensions
                src_x0 = src_x0 < 0 ? 0 : src_x0 >= image->width ? image->width - 1 : src_x0;
                src_x1 = src_x1 < 0 ? 0 : src_x1 >= image->width ? image->width - 1 : src_x1;
                src_y0 = src_y0 < 0 ? 0 : src_y0 >= image->height ? image->height - 1 : src_y0;
                src_y1 = src_y1 < 0 ? 0 : src_y1 >= image->height ? image->height - 1 : src_y1;

                // Get the pixel values
                unsigned char pixel00 = image->pixels[(src_y0 * image->width + src_x0) * image->channels + 0];
                unsigned char pixel01 = image->pixels[(src_y0 * image->width + src_x1) * image->channels + 0];
                unsigned char pixel10 = image->pixels[(src_y1 * image->width + src_x0) * image->channels + 0];
                unsigned char pixel11 = image->pixels[(src_y1 * image->width + src_x1) * image->channels + 0];

                // Perform bilinear interpolation
                float r_avg = (1 - x_weight) * (1 - y_weight) * pixel00 +
                              x_weight * (1 - y_weight) * pixel01 +
                              (1 - x_weight) * y_weight * pixel10 +
                              x_weight * y_weight * pixel11;

                // Adjust the average value and assign it to the new image
                float avg = (r_avg * scaleDown) + MINV;
                newPixels[(dest_y * newWidth * newChannels) + (dest_x * newChannels) + selectedChannel] = (signed char)avg;
            }
        }
    }
}


static void resize16BitImageWithBorders3ChTo1ofXChOnlyRChannel(
                                                               struct Image *image,
                                                               signed char *newPixels,
                                                               unsigned int newWidth,
                                                               unsigned int newHeight,
                                                               unsigned int newChannels,
                                                               unsigned int selectedChannel,
                                                               float *keypointXOffset,
                                                               float *keypointYOffset,
                                                               float *keypointXMultiplier,
                                                               float *keypointYMultiplier
                                                              )
{
    unsigned short *pixels = (unsigned short *)image->pixels;

    // Calculate aspect ratios
    float aspect_ratio_original = (float)image->width / image->height;
    float aspect_ratio_target = (float)newWidth / newHeight;

    // Determine the resizing factor and size for maintaining the aspect ratio
    unsigned int resizedWidth, resizedHeight;
    if (aspect_ratio_original > aspect_ratio_target)
    {
        resizedWidth = newWidth;
        resizedHeight = (unsigned int)(newWidth / aspect_ratio_original);
    } else
    {
        resizedWidth = (unsigned int)(newHeight * aspect_ratio_original);
        resizedHeight = newHeight;
    }

    // Calculate offsets for pasting resized image onto new image
    *keypointXOffset = (float)(newWidth - resizedWidth) / 2;
    *keypointYOffset = (float)(newHeight - resizedHeight) / 2;

    // Calculate keypoint multipliers
    *keypointXMultiplier = (float)resizedWidth / image->width;
    *keypointYMultiplier = (float)resizedHeight / image->height;

    // Resize the image while maintaining the aspect ratio and paste onto newPixels
    for (int y = 0; y < resizedHeight; y++)
    {
        for (int x = 0; x < resizedWidth; x++)
        {
            int dest_x = (int)(*keypointXOffset + (float)x);
            int dest_y = (int)(*keypointYOffset + (float)y);

            if (0 <= dest_x && dest_x < newWidth && 0 <= dest_y && dest_y < newHeight)
            {
                // Calculate source coordinates
                float src_x = (float)x / *keypointXMultiplier;
                float src_y = (float)y / *keypointYMultiplier;

                // Calculate the surrounding pixels for bilinear interpolation
                int src_x0 = (int)floor(src_x);
                int src_x1 = src_x0 + 1;
                int src_y0 = (int)floor(src_y);
                int src_y1 = src_y0 + 1;

                float x_weight = src_x - src_x0;
                float y_weight = src_y - src_y0;

                // Clamp the coordinates to the image dimensions
                src_x0 = src_x0 < 0 ? 0 : src_x0 >= image->width  ? image->width - 1 : src_x0;
                src_x1 = src_x1 < 0 ? 0 : src_x1 >= image->width  ? image->width - 1 : src_x1;
                src_y0 = src_y0 < 0 ? 0 : src_y0 >= image->height ? image->height - 1 : src_y0;
                src_y1 = src_y1 < 0 ? 0 : src_y1 >= image->height ? image->height - 1 : src_y1;

                // Get the pixel values
                unsigned short pixel00 = pixels[(src_y0 * image->width + src_x0) * image->channels + 0];
                unsigned short pixel01 = pixels[(src_y0 * image->width + src_x1) * image->channels + 0];
                unsigned short pixel10 = pixels[(src_y1 * image->width + src_x0) * image->channels + 0];
                unsigned short pixel11 = pixels[(src_y1 * image->width + src_x1) * image->channels + 0];

                // Perform bilinear interpolation
                float avg = (1 - x_weight) * (1 - y_weight) * pixel00 +
                            x_weight * (1 - y_weight) * pixel01 +
                            (1 - x_weight) * y_weight * pixel10 +
                            x_weight * y_weight * pixel11;

                unsigned short *pixelPtr = (unsigned short *)(newPixels + (dest_y * newWidth * newChannels) + (dest_x * newChannels) + selectedChannel);
                *pixelPtr = (unsigned short)avg;
            }
        }
    }
}


/*
static void resize16BitImageTo16BitHeatmapWithBorders1ChOLD(
                                                        struct Image *image,
                                                        signed short *newPixels,
                                                        unsigned int newWidth,
                                                        unsigned int newHeight,
                                                        unsigned int newChannels,
                                                        unsigned int selectedChannel,
                                                        float *keypointXOffset,
                                                        float *keypointYOffset,
                                                        float *keypointXMultiplier,
                                                        float *keypointYMultiplier
                                                        )
{
    unsigned short *pixels = (unsigned short *) image->pixels;

    // Calculate aspect ratios
    float aspect_ratio_original = (float) image->width / image->height;
    float aspect_ratio_target   = (float) newWidth / newHeight;

    // Determine the resizing factor and size for maintaining the aspect ratio
    unsigned int resizedWidth, resizedHeight;
    if (aspect_ratio_original > aspect_ratio_target)
    {
        resizedWidth  = newWidth;
        resizedHeight = (unsigned int)(newWidth / aspect_ratio_original);
    } else
    {
        resizedWidth  = (unsigned int)(newHeight * aspect_ratio_original);
        resizedHeight = newHeight;
    }

    // Calculate offsets for pasting resized image onto new image
    *keypointXOffset = (float) (newWidth - resizedWidth)   / 2;
    *keypointYOffset = (float) (newHeight - resizedHeight) / 2;

    // Calculate keypoint multipliers
    *keypointXMultiplier = (float) resizedWidth  / image->width;
    *keypointYMultiplier = (float) resizedHeight / image->height;

    // Calculate valid ranges for dest_x and dest_y
    int start_x = (int) *keypointXOffset;
    int start_y = (int) *keypointYOffset;
    //int end_x   = start_x + resizedWidth;
    //int end_y   = start_y + resizedHeight;

    // Adjust the loop range to avoid out-of-bound checks
    int max_src_x = image->width  - 2;  // Maximum value for src_x0 to ensure src_x1 is within bounds
    int max_src_y = image->height - 2; // Maximum value for src_y0 to ensure src_y1 is within bounds

    float src_x,  src_y;
    int   dest_x, dest_y;
    float x_weight, one_minus_x_weight, y_weight, one_minus_y_weight;
    int src_x0, src_y0, src_x1, src_y1;
    unsigned short pixel00, pixel01, pixel10, pixel11;
    signed int avg;

    for (int y = 0; y < resizedHeight; y++)
    {
        dest_y = start_y + y;
        if (dest_y < 0 || dest_y >= newHeight) continue;  // Skip out-of-bound y values

        src_y = (float) y / *keypointYMultiplier;
        if (src_y < 0 || src_y > max_src_y)    continue;  // Skip out-of-bound src_y values

        src_y0 = (int) src_y;
        src_y1 = src_y0 + 1;
        y_weight = src_y - src_y0;
        one_minus_y_weight = 1 - y_weight;

        for (int x = 0; x < resizedWidth; x++)
        {
            dest_x = start_x + x;
            if (dest_x < 0 || dest_x >= newWidth) continue;  // Skip out-of-bound x values

            src_x = (float) x / *keypointXMultiplier;
            if (src_x < 0 || src_x > max_src_x)   continue;  // Skip out-of-bound src_x values

            src_x0 = (int) src_x;
            src_x1 = src_x0 + 1;
            x_weight = src_x - src_x0;
            one_minus_x_weight = 1 - x_weight;

            // Get the pixel values
            pixel00 = pixels[(src_y0 * image->width + src_x0) * image->channels + 0];
            pixel01 = pixels[(src_y0 * image->width + src_x1) * image->channels + 0];
            pixel10 = pixels[(src_y1 * image->width + src_x0) * image->channels + 0];
            pixel11 = pixels[(src_y1 * image->width + src_x1) * image->channels + 0];

            // Perform bilinear interpolation
            avg =   one_minus_x_weight * one_minus_y_weight * pixel00 +
                    x_weight           * one_minus_y_weight * pixel01 +
                    one_minus_x_weight * y_weight           * pixel10 +
                    x_weight           * y_weight           * pixel11;

            avg -= ABS_MINV_16BIT; // Make avg signed short

            newPixels[ (dest_y * newWidth * newChannels) + (dest_x * newChannels) + selectedChannel ] = (signed short) avg;
        }
    }
}*/

static void resize16BitImageTo16BitHeatmapWithBorders1Ch(
                                                         struct Image *image,
                                                         signed short *newPixels,
                                                         unsigned int newWidth,
                                                         unsigned int newHeight,
                                                         unsigned int newChannels,
                                                         unsigned int selectedChannel,
                                                         float *keypointXOffset,
                                                         float *keypointYOffset,
                                                         float *keypointXMultiplier,
                                                         float *keypointYMultiplier
                                                        )
{
    unsigned short *pixels = (unsigned short *) image->pixels;

    // Calculate aspect ratios
    float aspect_ratio_original = (float) image->width / image->height;
    float aspect_ratio_target   = (float) newWidth / newHeight;

    // Determine resized dimensions while maintaining aspect ratio
    unsigned int resizedWidth, resizedHeight;
    if (aspect_ratio_original > aspect_ratio_target)
    {
        resizedWidth  = newWidth;
        resizedHeight = (unsigned int)(newWidth / aspect_ratio_original);
    } else
    {
        resizedWidth  = (unsigned int)(newHeight * aspect_ratio_original);
        resizedHeight = newHeight;
    }

    // Calculate integer start offsets for centering the resized image
    unsigned int start_x = (newWidth - resizedWidth)   / 2;
    unsigned int start_y = (newHeight - resizedHeight) / 2;
    *keypointXOffset = (float) start_x;
    *keypointYOffset = (float) start_y;

    // Calculate multipliers for coordinates conversion
    *keypointXMultiplier = (float) resizedWidth  / image->width;
    *keypointYMultiplier = (float) resizedHeight / image->height;

    // Precompute inverse multipliers to replace division with multiplication
    float inv_kx = image->width  / (float)resizedWidth;
    float inv_ky = image->height / (float)resizedHeight;

    // Determine valid source ranges to avoid out-of-bounds access
    int max_src_x = image->width  - 2;
    int max_src_y = image->height - 2;
    int x_max_valid = (int)(max_src_x * (*keypointXMultiplier));
    int y_max_valid = (int)(max_src_y * (*keypointYMultiplier));

    // Clamp the valid ranges to the resized dimensions
    int x_end = (resizedWidth - 1 < x_max_valid)  ? resizedWidth  - 1 : x_max_valid;
    int y_end = (resizedHeight - 1 < y_max_valid) ? resizedHeight - 1 : y_max_valid;
    x_end = x_end < 0 ? 0 : x_end;
    y_end = y_end < 0 ? 0 : y_end;

    // Precompute destination indexing constants
    unsigned int dest_channel_offset = selectedChannel;
    unsigned int dest_row_stride = newWidth * newChannels;

    // Iterate over valid y range
    for (int y = 0; y <= y_end; ++y)
    {
        int dest_y = start_y + y;

        // Compute source y and related weights
        float src_y = y * inv_ky;
        int src_y0 = (int)src_y;
        int src_y1 = src_y0 + 1;
        float y_weight = src_y - src_y0;
        float one_minus_y_weight = 1.0f - y_weight;

        // Precompute source row offsets for y0 and y1
        unsigned int src_y0_row = src_y0 * image->width * image->channels;
        unsigned int src_y1_row = src_y1 * image->width * image->channels;

        // Precompute destination row index
        unsigned int dest_y_index = dest_y * dest_row_stride;

        // Iterate over valid x range
        for (int x = 0; x <= x_end; ++x)
        {
            int dest_x = start_x + x;

            // Compute source x and related weights
            float src_x = x * inv_kx;
            int src_x0 = (int)src_x;
            int src_x1 = src_x0 + 1;
            float x_weight = src_x - src_x0;
            float one_minus_x_weight = 1.0f - x_weight;

            // Calculate source pixel offsets
            unsigned int src_x0_offset = src_x0 * image->channels;
            unsigned int src_x1_offset = src_x1 * image->channels;

            // Fetch pixel values using precomputed offsets
            unsigned short pixel00 = pixels[src_y0_row + src_x0_offset];
            unsigned short pixel01 = pixels[src_y0_row + src_x1_offset];
            unsigned short pixel10 = pixels[src_y1_row + src_x0_offset];
            unsigned short pixel11 = pixels[src_y1_row + src_x1_offset];

            // Perform bilinear interpolation
            signed int avg = (signed int)(
                                          one_minus_x_weight * one_minus_y_weight * pixel00 +
                                                    x_weight * one_minus_y_weight * pixel01 +
                                                    one_minus_x_weight * y_weight * pixel10 +
                                                              x_weight * y_weight * pixel11
                                         );
            avg -= ABS_MINV_16BIT; // Adjust to signed short range

            // Calculate destination index and assign the interpolated value
            unsigned int dest_index = dest_y_index + dest_x * newChannels + dest_channel_offset;
            newPixels[dest_index] = (signed short) avg;
        }
    }
}

//This does interpolation
static void resizeImageWithBorders3ChTo1ofXChFull(struct Image *image, signed char *newPixels, unsigned int newWidth, unsigned int newHeight,unsigned int newChannels,unsigned int selectedChannel,
                                             float *keypointXOffset, float *keypointYOffset, float *keypointXMultiplier, float *keypointYMultiplier)
{
    //Assuming already existing background
    //memset(newPixels, 0, newWidth * newHeight * image->channels); // <- this is wrong Initialize with black border

    //const int MAXV  =  120;
    //const int MINV  = -120;
    float scaleDown = (float) (MAXV-MINV)/255.0;

    // Calculate aspect ratios
    float aspect_ratio_original = (float)image->width / image->height;
    float aspect_ratio_target   = (float)    newWidth / newHeight;

    // Determine the resizing factor and size for maintaining the aspect ratio
    unsigned int resizedWidth, resizedHeight;
    if (aspect_ratio_original > aspect_ratio_target)
    {
        resizedWidth = newWidth;
        resizedHeight = (unsigned int)(newWidth / aspect_ratio_original);
    }
    else
    {
        //This case is problematic
        resizedWidth = (unsigned int)(newHeight * aspect_ratio_original);
        resizedHeight = newHeight;
    }

    // Calculate offsets for pasting resized image onto new image
    *keypointXOffset = (float) (newWidth - resizedWidth)   / 2;
    *keypointYOffset = (float) (newHeight - resizedHeight) / 2;

    // Calculate keypoint multipliers
    *keypointXMultiplier = (float)resizedWidth  / image->width;
    *keypointYMultiplier = (float)resizedHeight / image->height;

    /*
     fprintf(stderr,"Original 1CH image %ux%u\n",image->width,image->height);
     fprintf(stderr,"Target 1CH image %ux%u\n",newWidth,newHeight);
     fprintf(stderr,"1CH Image without borders %ux%u\n",resizedWidth,resizedHeight);
     fprintf(stderr,"1CH Image offset %0.2f,%0.2f\n",*keypointXOffset,*keypointYOffset);
     fprintf(stderr,"1CH Image scale %0.2f,%0.2f\n",*keypointXMultiplier,*keypointYMultiplier);*/

    // Resize the image while maintaining the aspect ratio and paste onto newPixels
    for (int y = 0; y < resizedHeight; y++)
    {
        for (int x = 0; x < resizedWidth; x++)
        {
            int dest_x = (int)(*keypointXOffset + (float) x);
            int dest_y = (int)(*keypointYOffset + (float) y);

            if (0 <= dest_x && dest_x < newWidth && 0 <= dest_y && dest_y < newHeight)
            {
                // Calculate source coordinates
                float src_x = (float) x / *keypointXMultiplier;
                float src_y = (float) y / *keypointYMultiplier;

                // Calculate the surrounding pixels to average
                int min_src_x = (int) floor(src_x);
                int min_src_y = (int) floor(src_y);
                int max_src_x = min_src_x + 1;
                int max_src_y = min_src_y + 1;

                // Interpolate color values for each channel
                int count = 0;
                float r_avg = 0.0, g_avg = 0.0, b_avg = 0.0;
                for (int dy = min_src_y; dy <= max_src_y; dy++)
                {
                    for (int dx = min_src_x; dx <= max_src_x; dx++)
                    {
                        if (0 <= dx && dx < image->width && 0 <= dy && dy < image->height)
                        {
                          r_avg += image->pixels[(dy * image->width * image->channels) + (dx * image->channels) + 0];
                          g_avg += image->pixels[(dy * image->width * image->channels) + (dx * image->channels) + 1];
                          b_avg += image->pixels[(dy * image->width * image->channels) + (dx * image->channels) + 2];
                          count++;
                        }
                    }
                }

                r_avg /= count;
                g_avg /= count;
                b_avg /= count;

                float avg = (r_avg + g_avg + b_avg)/ 3 ;
                avg = (avg * scaleDown) + MINV;//make it ready to become a signed char
                newPixels[(dest_y * newWidth * newChannels) + (dest_x * newChannels) + selectedChannel] = (signed char) avg;
            }
        }
    }
}
#ifdef __cplusplus
}
#endif

#endif // IMAGE_H_INCLUDED

