/**
 * @file HeatmapGenerator.h
 * @brief Header file for creating and populating heatmaps
 */
#ifndef HEATMAPGENERATOR_H_INCLUDED
#define HEATMAPGENERATOR_H_INCLUDED

#include "DataLoader.h"

#include <math.h>

// Function to find a heatmap with a specific gradient size
static signed char * find_heatmap_by_gradient_size(struct HeatmapCollection* collection, int gradientSize)
{
  if (collection!=0)
  {
    if (collection->num_heatmaps!=0)
    {
    gradientSize = gradientSize * 2;
    int target_size = gradientSize * gradientSize;  // Calculate the corresponding heatmap size
    for (int i = 0; i < collection->num_heatmaps; i++)
    {
        if (collection->heatmaps[i].dataSize == target_size)
        {
            return collection->heatmaps[i].data;
        }
    }

    fprintf(stderr,"The gradient size requested does not exist %u..           \n",gradientSize);

    for (int i = 0; i < collection->num_heatmaps; i++)
    {
     fprintf(stderr,"Available gradient size: %0.2f\n", sqrt(collection->heatmaps[i].dataSize));
    }
    abort();
    }
  }
  return NULL;  // Return NULL if no heatmap with the specified gradient size is found
}

// Function to find a heatmap with a specific gradient size
static signed char * find_positive_heatmap_by_gradient_size(struct HeatmapCollection* collection, int gradientSize)
{
  if (collection!=0)
  {
    if (collection->num_heatmaps!=0)
    {
    gradientSize = gradientSize * 2;
    int target_size = gradientSize * gradientSize;  // Calculate the corresponding heatmap size
    for (int i = 0; i < collection->num_heatmaps; i++)
    {
        if (collection->heatmaps_positive[i].dataSize == target_size)
        {
            return collection->heatmaps_positive[i].data;
        }
    }

    fprintf(stderr,"The gradient size requested does not exist %u..           \n",gradientSize);

    for (int i = 0; i < collection->num_heatmaps; i++)
    {
     fprintf(stderr,"Available gradient size: %0.2f\n", sqrt(collection->heatmaps_positive[i].dataSize));
    }
    abort();
    }
  }
  return NULL;  // Return NULL if no heatmap with the specified gradient size is found
}



static signed char* generate_heatmap(int gradientSize, int heatmapSize, signed char minValue, signed char maxValue)
{
    heatmapSize  = heatmapSize * 2;
    gradientSize = gradientSize ;

    // Allocate memory for the 2D array using a single malloc call
    signed char* heatmap = (signed char*) malloc(heatmapSize * heatmapSize * sizeof(signed char));

    if (heatmap!=0)
    {
     // Initialize the entire heatmap with the minimum value
     for (int i = 0; i < heatmapSize * heatmapSize; i++)
     {
        heatmap[i] = minValue;
     }

    // Populate the gradient around the center
     if (gradientSize > 0)
     {
     // Center position
      int centerY = (int) heatmapSize / 2;
      int centerX = (int) heatmapSize / 2;

      float gradientSizeF = gradientSize;
      int valueRange = maxValue - minValue;

      for (int y = 0; y < heatmapSize; y++)
        {
            for (int x = 0; x < heatmapSize; x++)
            {
                float distance = (float) sqrtf((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                if (distance < gradientSizeF)
                {
                    float intensity = (float) maxValue - (float) ((float) valueRange / gradientSize) * distance;
                    heatmap[y * heatmapSize + x] = (signed char) roundf(intensity);
                }
            }
        }

        // Set the center to the maximum value
        heatmap[centerY * heatmapSize + centerX] = maxValue;

     }

    return heatmap;
   }
   return NULL;
}



static struct HeatmapCollection* create_heatmap_collection(int minGradientSize, int maxGradientSize, signed char minValue, signed char maxValue)
{
    fprintf(stderr,"create_heatmap_collection gradientSize=[%u..%u] \n",minGradientSize,maxGradientSize);

    if (minGradientSize>maxGradientSize)
    {
        fprintf(stderr,"minimum gradient can't be larger than maximum gradient \n");
        abort();
    }

    struct HeatmapCollection* collection = (struct HeatmapCollection*) malloc(sizeof(struct HeatmapCollection));
    if (collection==0)
    {
        printf("Memory allocation failed!\n");
        return 0;
    }

    collection->minimumGradientSize = minGradientSize;
    collection->maximumGradientSize = maxGradientSize;
    collection->num_heatmaps        = maxGradientSize - minGradientSize + 1;
    collection->heatmaps            = (struct Heatmap*) malloc( collection->num_heatmaps * sizeof(struct Heatmap));
    collection->heatmaps_positive   = (struct Heatmap*) malloc( collection->num_heatmaps * sizeof(struct Heatmap));

    if ( (collection->heatmaps==0) || (collection->heatmaps_positive==0) )
    {
        printf("Memory allocation failed!\n");
        free(collection);
        return 0;
    }

    int thisSize = minGradientSize;
    for (int i = 0; i <collection->num_heatmaps; i++)
    {
        collection->heatmaps[i].dataSize          = (int) ( (thisSize*2) * (thisSize*2) );
        collection->heatmaps[i].data              = generate_heatmap(thisSize, thisSize, minValue, maxValue);
        collection->heatmaps_positive[i].dataSize = (int) ( (thisSize*2) * (thisSize*2) );
        collection->heatmaps_positive[i].data     = generate_heatmap(thisSize, thisSize, 0, maxValue);
        thisSize += 1;
    }

    return collection;
}


static void free_heatmap_collection(struct HeatmapCollection* collection)
{
   if (collection!=0)
   {
    if (collection->heatmaps!=0)
    {
     for (int i = 0; i < collection->num_heatmaps; i++)
     {
        if (collection->heatmaps[i].data!=0)
         {
            free(collection->heatmaps[i].data);
            collection->heatmaps[i].data     = 0;
            collection->heatmaps[i].dataSize = 0;
            free(collection->heatmaps_positive[i].data);
            collection->heatmaps_positive[i].data     = 0;
            collection->heatmaps_positive[i].dataSize = 0;
         }
     }
     free(collection->heatmaps);
     free(collection->heatmaps_positive);
     collection->heatmaps = 0;
     collection->heatmaps_positive = 0;
    }
    free(collection);
   }
}

int countSkeletonsJointsInHeatmaps(
                            struct ImageDatabase *db,
                            unsigned long sampleID,
                            int originalInputWidth,
                            int originalInputHeight,
                            int padX,
                            int padY,
                            float zoom_factor,
                            int pan_x,
                            int pan_y,
                            float offsetX,
                            float offsetY,
                            float scaleX,
                            float scaleY
                           );


int ensurePercentageOfJointsInHeatmap(
                                       float percentage,
                                       struct ImageDatabase *db,
                                       unsigned long sampleID,
                                       int originalInputWidth,
                                       int originalInputHeight,
                                       int padX,
                                       int padY,
                                       float zoom_factor,
                                       int pan_x,
                                       int pan_y,
                                       float offsetX,
                                       float offsetY,
                                       float scaleX,
                                       float scaleY
                                      );

/**
 * @brief Function to populate image database with heatmaps.
 * @param sourceSampleID The sample we want to create heatmaps for
 * @param targetImageID The memory slot to store the particular heatmap to.
 * @param gradientSize The size of each heatmap positive blob.
 * @param offsetX How to scale joints
 * @param offsetY How to offset joints.
 * @param scaleX How to scale joints
 * @param scaleY How to scale joints
 * @return Returns 1 on success, 0 on failure.
 */
int populateHeatmaps(
                     struct ImageDatabase *db,
                     unsigned long sourceSampleID,
                     unsigned long targetImageID,
                     unsigned int gradientSizeDefaultRAW,
                     unsigned int PAFSize,
                     unsigned int doLRFlip,
                     int originalInputWidth,
                     int originalInputHeight,
                     //---------------
                     float zoom_factor,
                     int pan_x,
                     int pan_y,
                     //---------------
                     float offsetX,
                     float offsetY,
                     float scaleX,
                     float scaleY
                    )
;

#endif // HEATMAPGENERATOR_H_INCLUDED
