#include <time.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include "pthreadWorkerPool.h"

#include "DataLoader.h"

#include "codecs/codecs.h"
#include "codecs/ppmInput.h"

#include "processing/resize.h"
#include "processing/augmentations.h"
#include "processing/normals.h"
#include "processing/conversions.h"
#include "processing/bilateral.h"
#include "processing/labels.h"

#include "DBLoader.h"
#include "HeatmapGenerator.h"

#include "PrepareBatch.h"

#include "tools.h"

//----------------------------------------------------------------------------------------------------
//Global instance counter to debug things that might be going out outside (e.g. from Keras/Python)
//----------------------------------------------------------------------------------------------------
int instanceCounter = 0;
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
void * db_allocate_source_list(unsigned int numberOfSources)
{
   struct DatabaseList * dbl = (struct DatabaseList*) malloc(sizeof(struct DatabaseList));
   if (dbl!=0)
   {
       dbl->numberOfKeypointsPerSample = 0; //To be determined
       dbl->totalNumberOfSamples       = 0; //To be determined
       dbl->numberOfSources            = numberOfSources;
       dbl->source = (struct DatabaseEntry *) malloc(sizeof(struct DatabaseEntry) * numberOfSources);
       if (dbl->source!=0)
       {
           memset(dbl->source,0,sizeof(struct DatabaseEntry) * numberOfSources);
       }
   }
   return (void*) dbl;
}
//----------------------------------------------------------------------------------------------------
void db_destroy_source_list(struct DatabaseList* dbList)
{
  if (dbList!=0)
  {
   struct DatabaseList * dbl = (struct DatabaseList*) dbList;
   if (dbl->source!=0)
   {
       #if USE_DINOV2_FEATURES
         for (int i=0; i<dbList->numberOfSources; i++)
         {
            fprintf(stderr,"Haphazard deallocation of descriptor dataset #%u \n",i);
             if (dbList->source[i].descriptorsAsADataset!=NULL)
             {
               free_descriptor_dataset(dbList->source[i].descriptorsAsADataset);
               free(dbList->source[i].descriptorsAsADataset);
               dbList->source[i].descriptorsAsADataset = NULL;
             }
         }
       #endif // USE_DINOV2_FEATURES


      free(dbl->source);
      dbl->source = 0;
   }
   free(dbl);
  }
  return;
}




//The next function is 100% ChatGPT generated..
void convertDepthPathToAllPath(const char *pathToCOCODepthMaps,char *pathToCombinedMetaData,size_t outSize)
{
    const char *p = strstr(pathToCOCODepthMaps, "depth_");
    size_t prefixLen, suffixLen;

    if (!p) {
        // No "depth_" substring: fallback → safe copy
        strncpy(pathToCombinedMetaData, pathToCOCODepthMaps, outSize - 1);
        pathToCombinedMetaData[outSize - 1] = '\0';
        return;
    }

    prefixLen = (size_t)(p - pathToCOCODepthMaps);

    // Copy prefix
    if (prefixLen >= outSize) { // Not enough space
        pathToCombinedMetaData[0] = '\0';
        return;
    }
    memcpy(pathToCombinedMetaData, pathToCOCODepthMaps, prefixLen);

    // Write "all_"
    const char *replacement = "all_";
    size_t replLen = 4;
    if (prefixLen + replLen >= outSize) {
        pathToCombinedMetaData[0] = '\0';
        return;
    }
    memcpy(pathToCombinedMetaData + prefixLen, replacement, replLen);

    // Copy the remainder *after* "depth_"
    const char *after = p + strlen("depth_");
    suffixLen = strlen(after);

    if (prefixLen + replLen + suffixLen >= outSize) {
        pathToCombinedMetaData[0] = '\0';
        return;
    }
    memcpy(pathToCombinedMetaData + prefixLen + replLen, after, suffixLen + 1);
}



//----------------------------------------------------------------------------------------------------
int db_set_source_entry(struct DatabaseList* dbList,unsigned int sourceIDExt,const char * pathToDBFile,const char * pathToCOCOImages,const char * pathToCOCODepthMaps,const char * pathToCOCOSegmentations,int ignoreNoSkeletonSamples)
{
  if (!fileExists(pathToDBFile))
  {
      fprintf(stderr,RED "Cannot create source without DB file, %s does not exist\n",pathToDBFile);
      return 0;
  }

  if (!directoryExists(pathToCOCOImages))
  {
      fprintf(stderr,RED "Cannot create source without Image files, Directory %s does not exist\n",pathToCOCOImages);
      return 0;
  }

  if (!directoryExists(pathToCOCODepthMaps))
  {
      fprintf(stderr,RED "Cannot create source without Depth files, Directory %s does not exist\n",pathToCOCODepthMaps);
      return 0;
  }

  if (!directoryExists(pathToCOCOSegmentations))
  {
      fprintf(stderr,RED "Cannot create source without Segmentation files, Directory %s does not exist\n",pathToCOCOSegmentations);
      return 0;
  }

  DatasetSourceID sourceID = sourceIDExt;
  if (dbList!=0)
  {
   struct DatabaseList * dbl = (struct DatabaseList*) dbList;
   if (sourceID<dbl->numberOfSources)
    {
     snprintf(dbl->source[sourceID].pathToDBFile,       MAX_PATH,"%s",pathToDBFile);
     snprintf(dbl->source[sourceID].pathToCOCOImages,   MAX_PATH,"%s",pathToCOCOImages);
     snprintf(dbl->source[sourceID].pathToCOCODepthMaps,MAX_PATH,"%s",pathToCOCODepthMaps);
     snprintf(dbl->source[sourceID].pathToCOCOSegmentations,MAX_PATH,"%s",pathToCOCOSegmentations);

     //It is very hard to remake all the .db files, so just patch in the new combined meta data..
     snprintf(dbl->source[sourceID].pathToCombinedMetaData,MAX_PATH,"%s",pathToCOCODepthMaps);
     convertDepthPathToAllPath(dbl->source[sourceID].pathToCOCODepthMaps,dbl->source[sourceID].pathToCombinedMetaData,MAX_PATH);

     dbl->source[sourceID].ignoreNoSkeletonSamples = (char) ignoreNoSkeletonSamples;
     dbl->source[sourceID].numberOfSamples = fastReadDatabaseNumberOfSamples(pathToDBFile);
     fprintf(stderr,"dbl->source[%lu].numberOfSamples = %lu \n",sourceID,dbl->source[sourceID].numberOfSamples);
     dbl->numberOfKeypointsPerSample       = fastReadDatabaseNumberOfKeypointsPerSample(pathToDBFile);
     fprintf(stderr,"dbl->numberOfKeypointsPerSample = %hu \n",dbl->numberOfKeypointsPerSample);
     dbl->totalNumberOfSamples   += dbl->source[sourceID].numberOfSamples;
     return 1;
    }
  }

  return 0;
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
void db_change_joint_difficulty(struct ImageDatabase * db ,unsigned short jID,signed char newDifficulty)
{
  if (db!=0)
  {
   if (jID<db->pdb->keypointsForEachSample)
      {
        fprintf(stderr,GREEN "Update joint difficulty for joint #%hu (%s) to %i\n" NORMAL,jID,db->pdb->joint[jID].name,newDifficulty);
        db->pdb->joint[jID].jointDifficulty = newDifficulty;
      } else
      {
        fprintf(stderr,RED "Could not update joint difficulty for joint #%hu\n" NORMAL,jID);
      }
  }
}
//----------------------------------------------------------------------------------------------------
// Function to shuffle an array using Fisher-Yates shuffle algorithm
//----------------------------------------------------------------------------------------------------
void db_shuffle_indices(struct ImageDatabase * db)
{
  if (db!=0)
  {
      if (db->indices!=0)
      {
       checkCanary("Shuffle check A",db->canaryA.shouldRemainUntouched,CANARY_SIZE);
       checkCanary("Shuffle check B",db->canaryB.shouldRemainUntouched,CANARY_SIZE);
       checkCanary("Shuffle check C",db->canaryB.shouldRemainUntouched,CANARY_SIZE);

       unsigned long i,j;
       for (i=0; i<db->numberOfSamples; i++)
        {
          //Old shuffler j = rand() % i; // Generate a random index from 1 to i
          j = (unsigned long) rand() % db->numberOfSamples; // Generate a random index in all the range of samples
          swapULong(&db->indices[i], &db->indices[j]); // Swap the elements at i and j
        }
      } else
      {
        fprintf(stderr,RED "Cannot shuffle dataset because of missing indices!\n" NORMAL);
      }
  } else
  {
    fprintf(stderr,RED "Cannot shuffle dataset without allocated dataset!\n" NORMAL);
  }

  return;
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
struct lossIndexPair
{
    float loss;
    unsigned long index;
};
//----------------------------------------------------------------------------------------------------
// Comparator function to sort pairs by loss value in descending order
//----------------------------------------------------------------------------------------------------
int comparePairs(const void *a, const void *b)
{
    struct lossIndexPair * aI = (struct lossIndexPair *) a;
    struct lossIndexPair * bI = (struct lossIndexPair *) b;

    // Since we want descending order, we subtract b from a
    float lossA = aI->loss;
    float lossB = bI->loss;

    if (lossA < lossB) return 1;
    if (lossA > lossB) return -1;
    return 0;
}
//----------------------------------------------------------------------------------------------------
void db_shuffle_indices_via_loss(struct ImageDatabase *db)
{
    fprintf(stderr,GREEN "experimental: using db_shuffle_indices_via_loss \n" NORMAL);

    if ( (db!=0) && (db->indices!=0) && (db->losses!=0) )
    {
        unsigned long i;

        // Step 1: Create an array of pairs (loss, index)
        struct lossIndexPair * pairs = (struct lossIndexPair *) malloc( (db->numberOfSamples+1) * sizeof(struct lossIndexPair));
        if (pairs != 0)
        {
         unsigned long shuffleStartTime = GetTickCountMicrosecondsMN();

         for (i = 0; i < db->numberOfSamples; i++)
         {
            pairs[i].loss  = (float) db->losses[i] / (db->trainPasses[i] + 0.0001);
            pairs[i].index = db->indices[i];

            /*
            if (db->trainPasses[i]>10)
            {   //Ensure that loss will not get a giant value and that more recent epochs of the model have
                //higher relative "weight" compared to averaged out previous epochs
                db->losses[i] /= 2.0;
                db->trainPasses[i] /= 2;
            } */
         }

         // Step 2: Sort the array of pairs by loss in descending order
         qsort(pairs, db->numberOfSamples, sizeof(struct lossIndexPair), comparePairs);

         // Step 3: Update the indices array with the sorted indices
         for (i = 0; i < db->numberOfSamples; i++)
         {
             db->indices[i] = pairs[i].index;
         }

         // Step 4: Mix odd elements with elements at the end
         //This way every two elements in the start contain a very hard (high loss) and very easy (low loss) sample
         //Samples near the middle contain two medium (medium loss) samples
         //Samples near the end contain a very easy (low loss) and very hard (high loss) sample
         unsigned long mid = db->numberOfSamples / 2;
         for (i = 1; i < mid; i += 2)
         {
            unsigned long j = db->numberOfSamples - i - 1;
            if ( (i<db->numberOfSamples) && (j<db->numberOfSamples) )
            {
             unsigned long temp = db->indices[i];
             db->indices[i] = db->indices[j];
             db->indices[j] = temp;
            }
         }

         //Sanity check
         //----------------------------------------------------------------------
         int foundErrors = 0;
         for (i = 0; i < db->numberOfSamples; i++)
         {
             if ( db->indices[i]>= db->numberOfSamples )
             {
                 fprintf(stderr,RED "Index %lu is erroneous, its value is %lu \n" NORMAL,i,db->indices[i]);
                 foundErrors = 1;
             }
         }
         if (foundErrors)
         {
             fprintf(stderr,RED "Stopping training to protect consistency\n" NORMAL);
             abort();
         }
         //----------------------------------------------------------------------


         // Free the allocated memory
         free(pairs);
         pairs=0;

         unsigned long shuffleEndTime = GetTickCountMicrosecondsMN();
         fprintf(stderr,"db_shuffle_indices_via_loss took %lu μsec\n\n",shuffleEndTime-shuffleStartTime);
         return;
        }
    }

    //Fall-back to regular shuffling if loss based sorting failed..
    db_shuffle_indices(db);
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
void db_update_sample_loss(struct ImageDatabase *db,unsigned long sample,float loss)
{
    if ( (db!=0) && (db->losses!=0) && (sample<db->numberOfSamples) )
    {
        db->losses[sample] = loss;
    } else
    {
        fprintf(stderr,RED "Unable to update loss for sample %lu, aborting to enforce consistency\n" NORMAL,sample);
        abort();
    }
}
//----------------------------------------------------------------------------------------------------
//This gets called from Dataloader.py->updateEpochResults which gets called from train2DPoseEstimation->DataAugmentation->on_epoch_end
//----------------------------------------------------------------------------------------------------
void db_update_sample_loss_range(struct ImageDatabase *db,unsigned long sampleStart,unsigned long sampleEnd,float loss)
{
    //fprintf(stderr,GREEN "db_update_sample_loss_range %lu -> %lu with loss %0.2f \n\n\n" NORMAL,sampleStart,sampleEnd,loss);
    if ( (db!=0) && (db->losses!=0) && (sampleStart<db->numberOfSamples) && (sampleEnd<db->numberOfSamples) )
    {
        SampleNumber sampleID;
        for  (sampleID = sampleStart; sampleID<sampleEnd; sampleID++)
         {
             SampleNumber sID = db->indices[sampleID];
             db->losses[sID]      += loss;
             db->trainPasses[sID] += 1;

             //Keep loss float from ballooning
             if (db->trainPasses[sID]>10)
             {
                db->trainPasses[sID]/=2;
                db->losses[sID]/=2.0;
             }
         }
    } else
    {
        fprintf(stderr,RED "Unable to update loss for sample %lu -> %lu, aborting to enforce consistency\n" NORMAL,sampleStart,sampleEnd);
        abort();
    }
}
//----------------------------------------------------------------------------------------------------
int db_get_filename_of_sample(struct ImageDatabase *db,unsigned long sampleID,char* buffer, size_t buffer_size)
{
  if ( (db!=0) && (db->losses!=0) && (sampleID<db->numberOfSamples) )
    {
        SampleNumber sID = db->indices[sampleID];

        DatasetSourceID sourceID = db_resolve_sample_sourceID((void *) db,sID);

        const char * filenameString = db->pdb->sample[sID].imagePath;
        const char * pathString     = db->dbSources->source[sourceID].pathToCOCOImages;

        if (strlen(filenameString) + 1 <= buffer_size)
        { // Check if buffer is large enough
          snprintf(buffer,buffer_size,"%s/%s",pathString,filenameString);

          return 1;
        } else
        {
         // If the buffer is too small, write a truncated message
         fprintf(stderr,"db_get_filename_of_sample: called with too small a buffer (%lu)", buffer_size - 1);
         buffer[0] = '\0'; // Ensure null-termination
        }
    }
 return 0;
}
//----------------------------------------------------------------------------------------------------
unsigned int db_resolve_sample_sourceID(struct ImageDatabase * db, SampleNumber sample)
{
  unsigned int result = 0;
  if (db!=0)
  {
      if (db->dbSources!=0)
      {
        unsigned long cumulative = 0;
        DatasetSourceID sID;
        for (sID=0; sID<db->dbSources->numberOfSources; sID++)
        {
          //fprintf(stderr,"source #%u | %s / %s  / %s\n", sID, db->dbSources->source[sID].pathToDBFile,db->dbSources->source[sID].pathToCOCOImages,db->dbSources->source[sID].pathToCOCODepthMaps);
          cumulative += db->dbSources->source[sID].numberOfSamples;
          if (sample<cumulative)
          {
              return sID;
          }
        }
      }
  }

  return result;
}
//----------------------------------------------------------------------------------------------------
void * db_create_without_dataset(
                                 unsigned long numberOfSamples,
                                 unsigned long numberOfImages,
                                 unsigned int widthIn,  unsigned int heightIn,unsigned int channelsIn,
                                 unsigned int widthOut, unsigned int heightOut,unsigned int channelsOut8Bit,unsigned int channelsOut16Bit)
{
    fprintf(stderr,"db_create_without_dataset %lu samples\n",numberOfSamples);
    fprintf(stderr,"RGB In  : %ux%u:%u\n",widthIn,heightIn,channelsIn);
    fprintf(stderr,"8-Bit Out : %ux%u:%u\n",widthOut,heightOut,channelsOut8Bit);
    fprintf(stderr,"16-Bit Out : %ux%u:%u\n",widthOut,heightOut,channelsOut16Bit);
    struct ImageDatabase * db = (struct ImageDatabase*) malloc(sizeof(struct ImageDatabase));
    if (db!=0)
    {
        memset(db,0,sizeof(struct ImageDatabase));
        //------------------------------------------------------------------------------
        db->pdb               = 0; // Make sure it is clean
        //------------------------------------------------------------------------------
        db->numberOfSamples               = numberOfSamples;
        db->numberOfImagesThatCanBeLoaded = numberOfImages;

        //------------------------------------------------------------------------------
        //Rearranged-memory to make sure oveflows on output do not destroy indexing
        //This needs to be the same length as the number of samples
        db->indices = (unsigned long *) malloc(sizeof(unsigned long) * (numberOfSamples+1));
        if (db->indices!=0)
        {
           memset(db->indices,0,sizeof(unsigned long) * (numberOfSamples+1));
           unsigned long sampleID = 0;
           for (sampleID=0; sampleID<numberOfSamples; sampleID++)
           {
              db->indices[sampleID] = sampleID;
           }
        } else
        {
             fprintf(stderr,RED "Failed allocating index array for samples\n"NORMAL );
             abort();
        }


        //Initialize losses
        //db->losses = (float *) malloc(sizeof(float) * (numberOfSamples+1));
        db->losses = (float *) db_malloc(sizeof(float) * (numberOfSamples+1));
        if (db->losses!=0)
        {
           memset(db->losses,0,sizeof(float) * (numberOfSamples+1));
        } else
        {
             fprintf(stderr,RED "Failed allocating loss array for samples\n"NORMAL );
             abort();
        }

        //Initialize traning pass count for each sample
        //db->trainPasses = (unsigned long *) malloc(sizeof(unsigned long) * (numberOfSamples+1));
        db->trainPasses = (unsigned long *) db_malloc(sizeof(unsigned long) * (numberOfSamples+1));
        if (db->trainPasses!=0)
        {
           memset(db->trainPasses,0,sizeof(unsigned long) * (numberOfSamples+1));
        } else
        {
           fprintf(stderr,RED "Failed allocating loss array for every sample\n"NORMAL );
           abort();
        }

        //Allocate rest of the memory..
        if (numberOfImages!=0)
        {
         db->in.numberOfImages  = numberOfImages;
         //db->in.pixels          = (void*) malloc(sizeof(unsigned char) * (numberOfImages+1) *  widthIn  * heightIn  * channelsIn);
         db->in.pixels          = (void*) db_malloc(sizeof(unsigned char) * (numberOfImages+1) *  widthIn  * heightIn  * channelsIn);
         if (db->in.pixels==0)
         {
             fprintf(stderr,RED "Failed allocating memory for RGB input images\n"NORMAL );
             abort();
         }
         #if USE_CANARY
         db->canaryA.shouldRemainUntouched = (unsigned long*) malloc(sizeof (unsigned long) * CANARY_SIZE);
         setupCanary(db->canaryA.shouldRemainUntouched,CANARY_SIZE);
         checkCanary("After initialization",db->canaryA.shouldRemainUntouched,CANARY_SIZE);
         #endif // USE_CANARY
         db->in.width           = widthIn;
         db->in.height          = heightIn;
         db->in.channels        = channelsIn;
         db->in.bitsperpixel    = 24;
         db->in.pixelsLimit     = db->in.pixels + (numberOfImages * db->in.width * db->in.height * db->in.channels);
         //fprintf(stderr," db->in.pixels = %p | limit %p | distance %td\n", db->in.pixels, db->in.pixelsLimit, ptrDistance);
         //------------------------------------------------------------------------------


         //db->out8bit.pixels         = (void *) malloc(sizeof(char) * (numberOfImages+1) *  widthOut * heightOut * channelsOut8Bit);
         db->out8bit.pixels         = (void *) db_malloc(sizeof(char) * (numberOfImages+1) *  widthOut * heightOut * channelsOut8Bit);
         if (db->out8bit.pixels==0)
         {
             fprintf(stderr,RED "Failed allocating memory for 8-bit output images\n"NORMAL );
             abort();
         }
         #if USE_CANARY
         db->canaryB.shouldRemainUntouched = (unsigned long*) malloc(sizeof (unsigned long) * CANARY_SIZE);
         setupCanary(db->canaryB.shouldRemainUntouched,CANARY_SIZE);
         checkCanary("After initialization",db->canaryB.shouldRemainUntouched,CANARY_SIZE);
         #endif // USE_CANARY
         db->out8bit.numberOfImages = numberOfImages;
         db->out8bit.channels       = channelsOut8Bit;
         db->out8bit.width          = widthOut;
         db->out8bit.height         = heightOut;
         db->out8bit.bitsperpixel   = 8;
         db->out8bit.pixelsLimit    = db->out8bit.pixels + (numberOfImages * db->out8bit.width * db->out8bit.height * db->out8bit.channels);
         //fprintf(stderr," db->out8bit.pixels = %p | limit %p | distance %td\n", db->out8bit.pixels, db->out8bit.pixelsLimit, ptrDistance);


         //db->out16bit.pixels         = (void *) malloc(sizeof(signed short) * (numberOfImages+1) *  widthOut * heightOut * channelsOut16Bit);
         db->out16bit.pixels         = (void *) db_malloc(sizeof(signed short) * (numberOfImages+1) *  widthOut * heightOut * channelsOut16Bit);
         if (db->out16bit.pixels==0)
         {
             fprintf(stderr,RED "Failed allocating memory for 8-bit output images\n"NORMAL );
             abort();
         }
         #if USE_CANARY
         db->canaryC.shouldRemainUntouched = (unsigned long*) malloc(sizeof (unsigned long) * CANARY_SIZE);
         setupCanary(db->canaryC.shouldRemainUntouched,CANARY_SIZE);
         checkCanary("After initialization",db->canaryC.shouldRemainUntouched,CANARY_SIZE);
         #endif // USE_CANARY
         db->out16bit.numberOfImages = numberOfImages;
         db->out16bit.channels       = channelsOut16Bit;
         db->out16bit.width          = widthOut;
         db->out16bit.height         = heightOut;
         db->out16bit.bitsperpixel   = 16;
         db->out16bit.pixelsLimit    = db->out16bit.pixels + (sizeof(signed short) * numberOfImages * db->out16bit.width * db->out16bit.height * db->out16bit.channels);
        }
    }

    //fprintf(stderr,"db_create pointer is %p\n",db);
    return db;
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------
unsigned char * db_get_in(struct ImageDatabase *db,unsigned long startSample)
{
    if (db!=0)
      { return (unsigned char *) db->in.pixels + (startSample * db->in.width * db->in.height * db->in.channels); }
    return 0;
}
//----------------------------------------------------------------------------------------------------
char * db_get_out(struct ImageDatabase *db,unsigned long startSample)
{
    if (db!=0)
      { return (char *) db->out8bit.pixels + (startSample * db->out8bit.width * db->out8bit.height * db->out8bit.channels * sizeof(char)); }
    return 0;
}
//----------------------------------------------------------------------------------------------------
int db_get_sampleNumberOfSkeletons(struct ImageDatabase *db, unsigned long sampleID)
{
    if (db!=0)
    {
        if (sampleID < db->numberOfSamples)
        {
         SampleNumber sID = db->indices[sampleID];
         return db->pdb->sample[sID].numberOfSkeletons;
        }
    }
    return 0;
}
//----------------------------------------------------------------------------------------------------
signed short * db_get_out16bit(struct ImageDatabase *db,unsigned long startSample,unsigned long endSample)
{
    if (db!=0)
      {
        return (signed short *) db->out16bit.pixels + (startSample * db->out16bit.width * db->out16bit.height * db->out16bit.channels * sizeof(signed short));
      }

    return 0;
}

int db_get_valid_segmentations(struct ImageDatabase *db)
{
  return VALID_SEGMENTATIONS;
}

int db_get_total_segmentation_classes(struct ImageDatabase *db)
{
  return TOTAL_SEGMENTATION_CLASSES;
}



//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------
int db_set_MAX_sample_description_token_value(struct ImageDatabase *db, int newMaxTokenValue)
{
  if (db!=0)
    {
      db->pdb->maxTokenValue              = newMaxTokenValue;
      db->pdb->maxTokenValueSetExternally = 1;
      return 1;
    }
  return 0;
}
//----------------------------------------------------------------------------------------------------
int db_get_MAX_sample_description_token_value(struct ImageDatabase *db)
{
    int returnValue = 0;

    if (db!=0) { returnValue = db->pdb->maxTokenValue; } else
               { fprintf(stderr,RED "db_get_MAX_sample_description_token_value called before database initialization..!\n" NORMAL); }

    return returnValue;
}
//----------------------------------------------------------------------------------------------------
int db_get_MAX_sample_description_tokens_number(struct ImageDatabase *db)
{
    return MAX_DESCRIPTION_TOKENS;
}
//----------------------------------------------------------------------------------------------------
int db_get_sample_description_tokens_number(struct ImageDatabase *db, unsigned long sampleID)
{
    if (db!=0)
    {
        if (sampleID < db->numberOfSamples)
        {
         SampleNumber sID = db->indices[sampleID];

         //Add erase here?
         //if (db->pdb->sample[sID].eraseEntry) { return 0; }

         int resultingNumberOfTokens = MAX_DESCRIPTION_TOKENS;
         if (resultingNumberOfTokens > (int) db->pdb->sample[sID].numberOfTokens)
             {
              resultingNumberOfTokens = (int) db->pdb->sample[sID].numberOfTokens;
             }

         //return db->pdb->sample[sID].numberOfTokens;
         return resultingNumberOfTokens;
        }
    }
    return 0;
}
//----------------------------------------------------------------------------------------------------
float db_get_sample_total_loss(struct ImageDatabase *db, unsigned long sampleID)
{
    if (db!=0)
    {
        if (sampleID < db->numberOfSamples)
        {
         SampleNumber sID = db->indices[sampleID];
         return db->losses[sID];
        }
    }
    return 0;
}
//----------------------------------------------------------------------------------------------------
unsigned long db_get_sample_train_passes(struct ImageDatabase *db, unsigned long sampleID)
{
    if (db!=0)
    {
        if (sampleID < db->numberOfSamples)
        {
         SampleNumber sID = db->indices[sampleID];
         return db->trainPasses[sID];
        }
    }
    return 0;
}
//----------------------------------------------------------------------------------------------------
unsigned short * db_get_sample_description_tokens(struct ImageDatabase *db, unsigned long sampleID)
{
    if (db!=0)
    {
        if (sampleID < db->numberOfSamples)
        {
         SampleNumber sID = db->indices[sampleID];

         if (db->pdb->sample[sID].eraseEntry)
            { return emptyDescriptionTokens; }

         return db->pdb->sample[sID].descriptionTokens;
        }
    }
    return 0;
}


//----------------------------------------------------------------------------------------------------
int db_get_descriptor_elements_number(struct ImageDatabase *db)
{
    if (db!=0)
    {
        return DINOV2_FEATURES_LENGTH;
        //return db->descriptorElementsNumber;
    }
    return 0;
}
//----------------------------------------------------------------------------------------------------
float * db_get_sample_descriptors(struct ImageDatabase *db, unsigned long sampleID)
{
    if (db!=0)
    {
        DatasetSourceID sourceID = db_resolve_sample_sourceID((void *) db,sampleID);
        if (db->dbSources->source[sourceID].descriptorsAsADataset!=0)
        {
         //Important check to check if descriptors are loaded
         if (sampleID < db->numberOfSamples)
          {
           SampleNumber sID = db->indices[sampleID];

           if (db->pdb->sample[sID].descriptor!=0) { return db->pdb->sample[sID].descriptor; } else
                                                   { fprintf(stderr,"Descriptor (sample number %lu) not allocated :(\n",sID); }
          } else
          {
            fprintf(stderr,"Sample ID is out of range %lu/%lu \n",sampleID,db->numberOfSamples);
          }
        } else
        {
            fprintf(stderr,"Source %lu (%s.dinov2) did not load(?) \n",sourceID,db->dbSources->source[sourceID].pathToDBFile);
        }
    }

    fprintf(stderr,"db_get_sample_descriptors(%lu)=NULL\n",sampleID);
    return emptyDescriptor;
    //return NULL;
}
//----------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------
int db_get_description_embeddings_number(struct ImageDatabase *db, unsigned long sampleID)
{
    if (db!=0)
    {
        return db->pdb->embeddings.D;
    }
    return 0;
}
//----------------------------------------------------------------------------------------------------
float * db_get_sample_description_embeddings(struct ImageDatabase *db, unsigned long sampleID, int tokenID)
{
  if (tokenID<MAX_DESCRIPTION_TOKENS)
    {
     unsigned short * tokens = db_get_sample_description_tokens(db,sampleID);
     if (tokens!=0)
      {
        // Return empty Vector if there is empty Description
        if (tokens==emptyDescriptionTokens)
                { return db->pdb->embeddings.emptyVector; }
                //{ return emptyDescriptionVectors; }

        unsigned int numberOfExistingTokens = db_get_sample_description_tokens_number(db,sampleID);
        if (tokenID<numberOfExistingTokens)
        {
          //If a class ID exists for this token serve it
          unsigned int classID = tokens[tokenID];
          float * returnVectorPointer = db->pdb->embeddings.embeddings[classID].vector;
          //fprintf(stderr,"Sample %lu | token %u => class %u => %p\n",sampleID,tokenID, classID ,returnVectorPointer);
          return returnVectorPointer;
        } else
        {
          //If client is asking a vector past our existing vectors reply with the "empty vector"
          return db->pdb->embeddings.emptyVector;
        }
      }
    }
    //If a client is asking for an out of range vector reply with nothing
    return 0;
}
//----------------------------------------------------------------------------------------------------
int db_free_description_token_count(float * token_weights)
{
    if (token_weights!=0)
    {
         free(token_weights);
         return 1;
    }

    return 0;
}
//----------------------------------------------------------------------------------------------------
int db_allocate_token_blacklist(struct ImageDatabase *db, unsigned int numberOfTokens)
{
  if (db!=0)
  {
      if (db->pdb!=0)
      {
          if (db->pdb->tokenBlackList!=0)
          {
              if (db->pdb->tokenBlackList->blackListedTokens!=0)
              {
                fprintf(stderr,"Found existing black list for tokens, freeing it..\n");
                db->pdb->tokenBlackList->blackListSize = 0;
                db->pdb->tokenBlackList->blackListCurrentTokens = 0;

                free(db->pdb->tokenBlackList->blackListedTokens);
                db->pdb->tokenBlackList->blackListedTokens = 0;
              }

              db->pdb->tokenBlackList->blackListedTokens = (unsigned short *) malloc(numberOfTokens * sizeof(unsigned short));
              if (db->pdb->tokenBlackList->blackListedTokens!=0)
              {
                //Clear black list
                memset(db->pdb->tokenBlackList->blackListedTokens,0,numberOfTokens * sizeof(unsigned short));
                db->pdb->tokenBlackList->blackListSize          = numberOfTokens;
                db->pdb->tokenBlackList->blackListCurrentTokens = 0;
                return 1;
              }
          }
      }
  }
      db->pdb->tokenBlackList = (struct DescriptionTokenBlacklist *) malloc(sizeof(struct DescriptionTokenBlacklist));
      if (db->pdb->tokenBlackList!=0)
      {
          memset(db->pdb->tokenBlackList,0,sizeof(struct DescriptionTokenBlacklist)); // <- clean up
          db->pdb->tokenBlackList->blackListSize = 0;//explicit cleanup
          db->pdb->tokenBlackList->blackListCurrentTokens = 0;//explicit cleanup
      }
  return 0;
}
//----------------------------------------------------------------------------------------------------
int db_add_token_to_blacklist(struct ImageDatabase *db, unsigned int tokenValue)
{
 if(tokenValue>=65535)
 {
   fprintf(stderr,"Token %u cannot be accomodated using ushort!\n",tokenValue);
   return 0;
 }

 if (db!=0)
  {
    if (db->pdb!=0)
      {
          if (db->pdb->tokenBlackList!=0)
          {
              if (db->pdb->tokenBlackList->blackListedTokens!=0)
              {
                  if (db->pdb->tokenBlackList->blackListCurrentTokens<db->pdb->tokenBlackList->blackListSize)
                  {
                    db->pdb->tokenBlackList->blackListedTokens[ db->pdb->tokenBlackList->blackListCurrentTokens ] = (unsigned short) tokenValue;
                    db->pdb->tokenBlackList->blackListCurrentTokens += 1;
                    return 1;
                  }
              }
          }
      }
  }
  return 0;
}
//----------------------------------------------------------------------------------------------------
int db_compile_added_token_blacklist(struct ImageDatabase *db)
{
 if (db!=0)
  {
    if (db->pdb!=0)
      {
          if (db->pdb->tokenBlackList->blackListedTokens!=0)
              {
                if (db->pdb->tokenBlackList->blackListCurrentTokens>0)
                  {
                    for (unsigned long sampleNumber=0; sampleNumber<db->numberOfSamples; sampleNumber++)
                      {
                        //Acknowledge shuffling (?)
                        //SampleNumber sID = db->indices[sampleNumber];

                        //This function does not acknowledge shuffle
                        //but data has better memory locality
                        SampleNumber sID = sampleNumber;

                        unsigned short * tokens         = db->pdb->sample[sID].descriptionTokens;
                        int numberOfTokensForThisSample = db->pdb->sample[sID].numberOfTokens;

                        if ( (tokens!=0) && (numberOfTokensForThisSample>0) )
                        {
                           //For each of existing samples go through all of its tokens
                           for (int tID=0; tID<numberOfTokensForThisSample; tID++)
                           {
                            if (tokens[tID]!=0)
                             {
                              //If a token exists in black list
                              for (int blackID=0; blackID<db->pdb->tokenBlackList->blackListCurrentTokens; blackID++)
                              {
                                  if (tokens[tID] == db->pdb->tokenBlackList->blackListedTokens[blackID] )
                                  {
                                    //Token is black listed
                                    tokens[tID] = 0; // <- permanently erased

                                  }
                              } //End of Loop over black listed tokens
                             } //Only do loop if there is a token
                           } //End of loop over all tokens for a specific sample

                           //Rearrange and sift tokens to remove spaces!
                           int validTokenIndex = 0;  // Tracks the position for the next valid token

                           for (int tID = 0; tID < numberOfTokensForThisSample; tID++)
                           {
                             if (tokens[tID] != 0) // Keep non-zero tokens
                             {
                                tokens[validTokenIndex] = tokens[tID];
                                validTokenIndex++;
                             }
                           }

                           // Reduce the number of tokens after removing zero tokens
                           db->pdb->sample[sID].numberOfTokens = validTokenIndex;

                           // Optionally set the remaining tokens to 0 (after validTokenIndex)
                           for (int tID = validTokenIndex; tID < numberOfTokensForThisSample; tID++)
                           {
                             tokens[tID] = 0; // Clear any leftover positions
                           }

                        } //Only do loop if sample has tokens
                     } //End of loop over all samples

                  return 1;
              }
         }
    }
 }
return 0;
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------
unsigned long * db_count_description_tokens(struct ImageDatabase *db, unsigned int numberOfTokens)
{
    if (numberOfTokens==0)
    {
      fprintf(stderr,RED "db_count_description_tokens called with number of tokens = %u \n" NORMAL,numberOfTokens);
      abort();
    }

    if (db!=0)
    {
        unsigned long * counts = (unsigned long *) malloc((numberOfTokens+1) * sizeof(unsigned long)); //Also add a null-termination class :P
        if (counts!=0)
        {
         //Clear counts before count
         memset(counts,0,(numberOfTokens+1) * sizeof(unsigned long));

         //Retrieve all sample tokens and sum them
         fprintf(stderr,"Will count %lu samples ",db->numberOfSamples);
         for (unsigned long sampleNumber=0; sampleNumber<db->numberOfSamples; sampleNumber++)
          {
           //Acknowledge shuffling (?)
           //SampleNumber sID = db->indices[sampleNumber];

           //This function does not acknowledge shuffle
           //but data has better memory locality
           SampleNumber sID = sampleNumber;

           //Receive tokens for particular sID
           unsigned short * tokens         = db->pdb->sample[sID].descriptionTokens;
           int numberOfTokensForThisSample = db->pdb->sample[sID].numberOfTokens;
           if ( (tokens!=0) && (numberOfTokensForThisSample>0) )
           {
             //For each of existing samples count them
             for (int tID=0; tID<numberOfTokensForThisSample; tID++)
              {
                //Make sure no erroneous values are counted!
                if ( (tokens[tID]<=numberOfTokens) && (tokens[tID]!=0) )
                {
                    if (tokens[tID]!=0)
                    {
                     counts[tokens[tID]]+=1;
                     //BUG: For some reason tokens[tID] is +2?
                    }
                } else
                {
                    fprintf(stderr,RED "\nError counting token %u (value %u) for sample %lu \n" NORMAL,tID,tokens[tID],sID);
                    fprintf(stderr,RED "Number of tokens is %u \n" NORMAL,numberOfTokens);
                    fprintf(stderr,RED "This probably means that there are images that have not been properly captioned!\n" NORMAL);
                    fprintf(stderr,RED "Stopping Data Loader to prevent corrupted training!\n" NORMAL);
                    DatasetSourceID sourceID = db_resolve_sample_sourceID((void *) db,sID);
                    fprintf(stderr,"Source  is %s | ",db->dbSources->source[sourceID].pathToCOCOImages);
                    fprintf(stderr,"Sample %lu is %s \n",sID,db->pdb->sample[sID].imagePath);
                    fprintf(stderr,RED "Consider removing this source as a quick fix!\n" NORMAL);
                    fprintf(stderr,RED "Investigate using :\n" NORMAL);
                    fprintf(stderr,RED "cat %s/*.db | grep  -A 10 %s \n" NORMAL,db->dbSources->source[sourceID].pathToCOCOImages,db->pdb->sample[sID].imagePath);
                    abort();
                }
              }
           }
          }
         fprintf(stderr,GREEN " Done\n" NORMAL);

          return counts;
        } //Managed to allocate counts
    }
    return 0;
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------
void bubbleSortTokensUShort(unsigned short *tokens, int numberOfTokens)
{
    for (int i = 0; i < numberOfTokens - 1; i++)
    {
        for (int j = 0; j < numberOfTokens - i - 1; j++)
        {
            if (tokens[j] > tokens[j + 1])
            {
                // Swap tokens[j] and tokens[j+1]
                unsigned short temp = tokens[j];
                tokens[j] = tokens[j + 1];
                tokens[j + 1] = temp;
            }
        }
    }
}
//---------------------------------------------------------------------
int db_sort_description_tokens(struct ImageDatabase *db)
{
    if (db!=0)
    {
         for (unsigned long sampleNumber=0; sampleNumber<db->numberOfSamples; sampleNumber++)
          {
           //This function does not acknowledge shuffle
           //but data has better memory locality
           SampleNumber sID = sampleNumber;
           unsigned short * tokens         = db->pdb->sample[sID].descriptionTokens;
           int numberOfTokensForThisSample = db->pdb->sample[sID].numberOfTokens;
           if ( (tokens!=0) && (numberOfTokensForThisSample>0) )
           {
              bubbleSortTokensUShort(tokens, numberOfTokensForThisSample);
           }
          }

        return 1;
    }
    return 0;
}
//---------------------------------------------------------------------
void bubbleSortTokensByCount(unsigned short *tokens, int numberOfTokens, unsigned long *counts, int numberOfAllTokens, int ascending)
{
    for (int i = 0; i < numberOfTokens - 1; i++)
    {
        for (int j = 0; j < numberOfTokens - i - 1; j++)
        {
            if ( (tokens[j]>=numberOfAllTokens) || (tokens[j+1]>=numberOfAllTokens) )
            {
                fprintf(stderr,"bubbleSortTokensByCount have to access tokens higher than the allocated %u counts\n",numberOfAllTokens);
                return;
            }

            unsigned long count_j   = counts[tokens[j]];
            unsigned long count_j1  = counts[tokens[j + 1]];

            int condition = ascending ? (count_j > count_j1) : (count_j < count_j1);
            if (condition)
            {
                // Swap tokens[j] and tokens[j+1]
                unsigned short temp = tokens[j];
                tokens[j] = tokens[j + 1];
                tokens[j + 1] = temp;
            }
        }
    }
}
//---------------------------------------------------------------------
int db_sort_description_tokens_based_on_count(struct ImageDatabase *db,unsigned int numberOfAllTokens,int ascending)
{
    if (db!=0)
    {
        fprintf(stderr,"Sorting %u tokens based on their frequencies.. \n",numberOfAllTokens);
        if (ascending) { fprintf(stderr,"Sorting will happen on ascending order\n"); } else
                       { fprintf(stderr,"Sorting will happen on descending order\n"); }
        fprintf(stderr,"This is not optimized since it should only happen 1x during startup, so it will take a while.. \n");

        unsigned long * counts = db_count_description_tokens(db,numberOfAllTokens);
        if (counts!=0)
        {
         for (unsigned long sampleNumber=0; sampleNumber<db->numberOfSamples; sampleNumber++)
          {
           //This function does not acknowledge shuffle
           //but data has better memory locality
           SampleNumber sID = sampleNumber;
           unsigned short * tokens         = db->pdb->sample[sID].descriptionTokens;
           int numberOfTokensForThisSample = db->pdb->sample[sID].numberOfTokens;
           if ( (tokens!=0) && (numberOfTokensForThisSample>0) )
           {
              //counts[tokens[sID]] has the number occurances of this token
              bubbleSortTokensByCount(tokens, numberOfTokensForThisSample, counts, numberOfAllTokens, ascending);
           }
          }
          fprintf(stderr,"Freeing counts.. \n");
          free(counts);
          fprintf(stderr,"Survived.. \n");
          return 1;
        }
    }
    return 0;
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// Function to remove duplicate tokens from a single description
void removeDuplicateTokens(unsigned short *tokens, unsigned char *numberOfTokens)
{
    if (tokens == NULL || numberOfTokens == NULL || *numberOfTokens == 0)
    {
        return;
    }

    // Create a boolean array to track if a token has been seen
    unsigned char seen[65535] = {0}; // USHRT_MAX Assuming tokens are unsigned short ( see  unsigned short descriptionTokens[MAX_DESCRIPTION_TOKENS]; )

    unsigned char initialNumberOfTokens = *numberOfTokens;

    unsigned char writeIndex = 0;
    for (unsigned char i = 0; i < *numberOfTokens; i++)
    {
        if (!seen[tokens[i]])
        {
            seen[tokens[i]] = 1; // Mark token as seen
            tokens[writeIndex++] = tokens[i];
        }
    }

    if (writeIndex<initialNumberOfTokens)
    {
     //Cleanup rest of tokens
     for (unsigned char i = writeIndex; i < initialNumberOfTokens; i++)
     {
        tokens[i] = 0; //Clean remaining values after writeIndex
     }
    }


    // Update the number of tokens to reflect the unique tokens only
    // This writes back to the pointer pointing to the DB for immediate update
    *numberOfTokens = (unsigned char) writeIndex;
}
//----------------------------------------------------------------------------------------------------
// Function to iterate over all descriptions and remove duplicate tokens
//----------------------------------------------------------------------------------------------------
int db_remove_duplicate_description_tokens(struct ImageDatabase *db)
{
    if (db == NULL)
    {
        fprintf(stderr, "Database pointer is NULL.\n");
        return 0;
    }

    fprintf(stderr, "Removing duplicate tokens from all descriptions...\n");

    for (unsigned long sampleNumber = 0; sampleNumber < db->numberOfSamples; sampleNumber++)
    {
        SampleNumber sID = sampleNumber;
        unsigned short *tokens                      = db->pdb->sample[sID].descriptionTokens;
        unsigned char  *numberOfTokensForThisSample = &db->pdb->sample[sID].numberOfTokens;

        if (tokens != NULL && *numberOfTokensForThisSample > 0)
        {
            removeDuplicateTokens(tokens, numberOfTokensForThisSample);
        }

        //TODO: at this point the rest of the elements of the description are not cleared..
        //Maybe set them to 0 ?
    }

    fprintf(stderr, "Finished removing duplicate tokens.\n");
    return 1;
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------
float * db_count_description_token_weight(struct ImageDatabase *db, unsigned int numberOfTokens)
{
    if (db!=0)
    {
      float * weights = (float *) malloc( (numberOfTokens) * sizeof(float)); //Also add a null-termination class :P
      if (weights!=0)
      {
        //Initialize all tokens as 1.0
        for (int i=0; i<numberOfTokens; i++)
        {
           weights[i]=1.0;
        }

        unsigned long * counts = db_count_description_tokens(db,numberOfTokens);
        if (counts!=0)
        {
          //Don't include black listed tokens in frequencies!
          if (db->pdb->tokenBlackList->blackListedTokens!=0)
              {
                  if (db->pdb->tokenBlackList->blackListCurrentTokens>0)
                  {
                    for (int i=0; i<db->pdb->tokenBlackList->blackListCurrentTokens; i++)
                            {
                                unsigned short erasedID = db->pdb->tokenBlackList->blackListedTokens[i];
                                if (i<numberOfTokens)
                                {
                                  counts[erasedID] = 0;
                                }
                            }
                  }
              }

          //All samples are counted lets convert them to weights
          unsigned long totalSum = 0;
          for (int i=0; i<numberOfTokens; i++)
                           { totalSum += counts[i]; }

          float maximumValue = -1.0;
          #define NORMALIZE_WEIGHTS 0

          //All samples are counted lets convert them to weights
          for (int i=0; i<numberOfTokens; i++)
             {
                if (counts[i]==0)
                {
                  weights[i] = 1.0;
                } else
                {
                  //Perform the weighting
                  weights[i] = (float) totalSum / (counts[i] + 1e-6);

                  #if NORMALIZE_WEIGHTS
                  //If compiled in also normalize the weights
                  if (weights[i]>maximumValue)
                        { maximumValue=weights[i]; }
                  #endif // NORMALIZE_WEIGHTS
                }
             }

          #if NORMALIZE_WEIGHTS
          //Also normalize ranges to [0..1]
          for (int i=0; i<numberOfTokens; i++)
          {
             weights[i] = weights[i]  /  (maximumValue+0.0001);
          }
          #endif // NORMALIZE_WEIGHTS


          free(counts);
        } //Managed to allocate counts

         return weights;
        } //Managed to allocate weights
    }
    return 0;
}
//---------------------------------------------------------------------
//---------------------------------------------------------------------
//---------------------------------------------------------------------
int db_map_sample_in_to_image(struct ImageDatabase * db,struct Image * picToMap,unsigned long sampleNumber)
{
  if ((db!=0) && (picToMap!=0))
  {
   if (sampleNumber<db->numberOfSamples)
   {
    picToMap->pixels       = (unsigned char*) db->in.pixels + (db->in.width * db->in.height * db->in.channels * sampleNumber);
    picToMap->width        = db->in.width;
    picToMap->height       = db->in.height;
    picToMap->channels     = db->in.channels;
    picToMap->bitsperpixel = (unsigned int) 24;
    picToMap->image_size   = picToMap->width * picToMap->height * picToMap->channels;
    picToMap->timestamp    = 0;
    return 1;
   }
  }
  return 0;
}
//----------------------------------------------------------------------------------------------------
unsigned long db_get_number_of_samples(struct ImageDatabase * db)
{
 if (db!=0)
 {
   return db->numberOfSamples;
 }
 return 0;
}
//----------------------------------------------------------------------------------------------------
unsigned long db_get_number_of_images(struct ImageDatabase * db)
{
 if (db!=0)
 {
   if ( (db->in.numberOfImages == db->out8bit.numberOfImages) && (db->out8bit.numberOfImages == db->out16bit.numberOfImages) )
        { return db->in.numberOfImages; }
 }

 fprintf(stderr,"Inconsistent number of images in I/O\n");
 abort();
 return 0;
}
//----------------------------------------------------------------------------------------------------
//Worker thread source code is PrepareBatch.c
//----------------------------------------------------------------------------------------------------
int db_start_threads(struct ImageDatabase * db,
                     unsigned long sampleStart,
                     unsigned long sampleEnd,
                     int workerThreads,
                     int gradientSize,
                     int PAFSize)
{
    if (db!=0)
    {
      if (db->threadCtx==0)
      {
       db->numberOfThreads = workerThreads;

       //We also create one context to be supplied for each thread..
       db->threadCtx = (struct workerThreadContext *) malloc(db->numberOfThreads * sizeof(struct workerThreadContext));
       if (db->threadCtx==0)
          {
           fprintf(stderr,"Failed allocating thread context");
           return 0;
          } else
          {
            memset(db->threadCtx,0,db->numberOfThreads * sizeof(struct workerThreadContext));
          }

       for (int tID=0; tID<db->numberOfThreads; tID++)
       {
        db->threadCtx[tID].db                = db;
        db->threadCtx[tID].sampleStart       = sampleStart;
        db->threadCtx[tID].sampleEnd         = sampleEnd;
        db->threadCtx[tID].thisThreadNumber  = tID;
        db->threadCtx[tID].workerThreads     = db->numberOfThreads;
        db->threadCtx[tID].gradientSize      = gradientSize;
        db->threadCtx[tID].PAFSize           = PAFSize;
        db->threadCtx[tID].computationOutput = 0;
        db->threadCtx[tID].gradientX         = 0;
        db->threadCtx[tID].gradientY         = 0;
       }

    if ( threadpoolCreate(&db->threadPool,workerThreads,workerThread,(void *) db->threadCtx) )
      {
        return 1;
      }
      fprintf(stderr,RED "db_start_threads failed threadpoolCreate\n" NORMAL);
      return 0;
     } else
     {
      return 1;
     }

    }

  fprintf(stderr,RED "db_start_threads cannot be done without a database!\n" NORMAL);
  return 0; //No database can't do anything
}
//----------------------------------------------------------------------------------------------------
int db_stop_threads(struct ImageDatabase * db)
{
  if (db!=0)
    {
      if (db->threadCtx!=0)
      {
       for (int i=0; i<db->numberOfThreads; i++)
       {
        #if INTEL_OPTIMIZATIONS
         if (db->threadCtx[i].gradientX!=0)
          { _mm_free(db->threadCtx[i].gradientX); db->threadCtx[i].gradientX=0; }
         if (db->threadCtx[i].gradientY!=0)
          { _mm_free(db->threadCtx[i].gradientY); db->threadCtx[i].gradientY=0; }
        #else
         if (db->threadCtx[i].gradientX!=0)
           { free(db->threadCtx[i].gradientX); db->threadCtx[i].gradientX=0; }
         if (db->threadCtx[i].gradientY!=0)
           { free(db->threadCtx[i].gradientY); db->threadCtx[i].gradientY=0; }
        #endif // INTEL_OPTIMIZATIONS
       }

       db->numberOfThreads = 0;
       threadpoolDestroy(&db->threadPool);
       free(db->threadCtx);
       db->threadCtx = 0;
       return 1; // We stopped threads
      }
    }
  return 0; //Was already stopped
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
int db_update(struct ImageDatabase * db,
              unsigned long sampleStart,
              unsigned long sampleEnd,
              int workerThreads,
              int gradientSize,
              int PAFSize)
{
    if (db==0) { return 0; }
    int success = 0;

    //HEATMAP
    //Make sure Heatmap Pattern
    if ( (db->gradients!=0) && (db->gradients->minimumGradientSize<=gradientSize) && (db->gradients->maximumGradientSize>=gradientSize) )
    {
      //We have an allocated heatmap pattern for our gradient that is ok
    } else
    {
     free_heatmap_collection(db->gradients);
     db->gradients = create_heatmap_collection(MINIMUM_GRADIENT_SIZE,gradientSize+MAXIMUM_GRADIENT_SIZE_DIFFICULTY,MINV,MAXV);
    } //heatmap pattern should be ready

    unsigned long threadPrepareStartTime = GetTickCountMicrosecondsMN();

    if ( db_start_threads(db,sampleStart,sampleEnd,workerThreads,gradientSize,PAFSize) )
    {
     //This is already done in db_start_threads the first time, but we make sure
     //to update it on subsequent loops..
     for (unsigned int tID=0; tID<db->numberOfThreads; tID++)
      {
        db->threadCtx[tID].db=db;
        db->threadCtx[tID].sampleStart      = sampleStart;
        db->threadCtx[tID].sampleEnd        = sampleEnd;
        db->threadCtx[tID].thisThreadNumber = tID;
        db->threadCtx[tID].workerThreads    = db->numberOfThreads;
        db->threadCtx[tID].gradientSize     = gradientSize;
        db->threadCtx[tID].computationOutput = 0;
      }

      db->computationsRunning = 1;
      threadpoolMainThreadPrepareWorkForWorkers(&db->threadPool);
      unsigned long threadPrepareEndTime = GetTickCountMicrosecondsMN();
      db->threadStartLagMicroseconds = threadPrepareEndTime - threadPrepareStartTime;

      //Workers ( PrepareBatch.c ) are now working..
      //usleep(10000); //<- These will crash the worker loops for some reason :S
      // ZzZZzzzZZzzz...
      //usleep(10000); //<- These will crash the worker loops for some reason :S
      //Let's see if they finished (impossible with so little time ;P) ..

      //fprintf(stdout,"Main thread waiting for workers to finish..!\n");
      //threadpoolMainThreadWaitForWorkersToFinish(&db->threadPool);      //this function waits foreverfor the workers to finish, instead we use the next one with a hard timeout
      threadPrepareStartTime = GetTickCountMicrosecondsMN();

      threadpoolMainThreadWaitForWorkersToFinishTimeoutSeconds(&db->threadPool,MAXIMUM_WORKER_THREAD_TIMEOUT_SECONDS); //if it takes us more than one minute to collect work, we are deadlocked and should abort
      db->computationsRunning = 0;

      //fprintf(stdout,"Main thread collecting results..!\n");
      for (unsigned int contextID=0; contextID<workerThreads; contextID++)
            {
             success = success + db->threadCtx[contextID].computationOutput;
             /*fprintf(stderr,"Thread #%u %u/%u | %lu->%lu | : Fullfilled Items %u \n",contextID,
                     db->threadCtx[contextID].thisThreadNumber,
                     db->threadCtx[contextID].workerThreads,
                     db->threadCtx[contextID].sampleStart,
                     db->threadCtx[contextID].sampleEnd,
                     db->threadCtx[contextID].fullfilledWork); */
            }

       threadPrepareEndTime = GetTickCountMicrosecondsMN();
       db->threadCompletionLagMicroseconds = threadPrepareEndTime - threadPrepareStartTime;


      } else
      {
        fprintf(stdout,RED "db_update: Failed starting threads..\n" NORMAL);
      }

     if (success==workerThreads)
            { return 1; }

      //If this is called then threads will be re-created next time
      //db_stop_threads(vdb);

      return 0;
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

//This way of updating is working slower than regular updates
//So commented out to make sure they are not accidentally used..
int db_StartUpdate(struct ImageDatabase * db,
                   unsigned long sampleStart,
                   unsigned long sampleEnd,
                   int workerThreads,
                   int gradientSize,
                   int PAFSize)
{
    if (db==0) { return 0; }

    //HEATMAP
    //Make sure Heatmap Pattern
    if ( (db->gradients!=0) && (db->gradients->minimumGradientSize<=gradientSize) && (db->gradients->maximumGradientSize>=gradientSize) )
    {
      //We have an allocated heatmap pattern for our gradient that is ok
    } else
    {
     free_heatmap_collection(db->gradients);
     db->gradients = create_heatmap_collection(MINIMUM_GRADIENT_SIZE,gradientSize+MAXIMUM_GRADIENT_SIZE_DIFFICULTY,MINV,MAXV);
    } //heatmap pattern should be ready


    if ( db_start_threads(db,sampleStart,sampleEnd,workerThreads,gradientSize,PAFSize) )
    {
     for (int tID=0; tID<db->numberOfThreads; tID++)
      {
        db->threadCtx[tID].db=db;
        db->threadCtx[tID].sampleStart      = sampleStart;
        db->threadCtx[tID].sampleEnd        = sampleEnd;
        db->threadCtx[tID].thisThreadNumber = tID;
        db->threadCtx[tID].workerThreads    = db->numberOfThreads;
        db->threadCtx[tID].gradientSize     = gradientSize;
        db->threadCtx[tID].computationOutput = 0;
      }

      db->computationsRunning = 1;
      threadpoolMainThreadPrepareWorkForWorkers(&db->threadPool);
      return 1;
    } else
    {
     fprintf(stdout,RED "db_update: Failed starting threads..\n" NORMAL);
    }
    return 0;
}


//----------------------------------------------------------------------------------------------------
int db_CollectUpdate(struct ImageDatabase * db,
                     unsigned long sampleStart,
                     unsigned long sampleEnd,
                     int workerThreads,
                     int gradientSize,
                     int PAFSize)
{
    if (db==0) { return 0; }
    int success = 0;

    unsigned long threadPrepareStartTime = GetTickCountMicrosecondsMN();
    if (!db->computationsRunning)
    {
     db_StartUpdate(db,sampleStart,sampleEnd,workerThreads,gradientSize,PAFSize);
    }


    if (db->computationsRunning)
    {
      //fprintf(stdout,"Main thread waiting for workers to finish..!\n");
      threadpoolMainThreadWaitForWorkersToFinish(&db->threadPool);
      db->computationsRunning = 0;

      //fprintf(stdout,"Main thread collecting results..!\n");
      for (unsigned int contextID=0; contextID<workerThreads; contextID++)
            { success = success + db->threadCtx[contextID].computationOutput; }

     unsigned long threadPrepareEndTime = GetTickCountMicrosecondsMN();
     db->threadStartLagMicroseconds = threadPrepareEndTime - threadPrepareStartTime;

     if (success==workerThreads)
            { return 1; }

      //If this is called then threads will be re-created next time
      //db_stop_threads(vdb);
    }
    return 0;
}
//----------------------------------------------------------------------------------------------------


//New db_update based on db_StartUpdate / db_CollectUpdate
int db_update_new(struct ImageDatabase * db,
              unsigned long sampleStart,
              unsigned long sampleEnd,
              int workerThreads,
              int gradientSize,
              int PAFSize)
{
    if ( db_StartUpdate(db,sampleStart,sampleEnd,workerThreads,gradientSize,PAFSize) )
    {
       return db_CollectUpdate(db,sampleStart,sampleEnd,workerThreads,gradientSize,PAFSize);
    }

    fprintf(stdout,RED "db_update: Failed starting update..\n" NORMAL);
    return 0;
}
//----------------------------------------------------------------------------------------------------
void db_print_readSpeed(struct ImageDatabase * db)
{
  if (db!=0)
  {
    if (db->files!=0)
    {
      for (int tID=0; tID<db->numberOfThreads; tID++)
            {
              float speedInMB = cache_readSpeedMBPerSecond(db->files, tID);
              float speedInGB = (float) speedInMB/1024.0f;
              if (speedInGB >1.0)   { fprintf(stderr,GREEN);} else
              if (speedInGB >0.3)   { fprintf(stderr,YELLOW);} else
              if (speedInGB >0.003) { fprintf(stderr,RED);} else
                                    { fprintf(stderr,RED);}

              if (speedInGB>0.3) { fprintf(stderr,"T%02u: %0.2f GB/s ",tID, speedInGB ); } else
              if (speedInMB>0.3) { fprintf(stderr,"T%02u: %0.2f MB/s ",tID, speedInMB ); } else
                                 { fprintf(stderr,"T%02u: %0.2f KB/s ",tID, (float) speedInMB * 1024.0f ); }

              fprintf(stderr,NORMAL);
            }
       fprintf(stderr,"\n");
    }
  }
}
//----------------------------------------------------------------------------------------------------

/**
 * @brief Function to compute statistics about the percentage of samples with and without joints in an average batch.
 * @param db Pointer to the ImageDatabase structure.
 * @param batchSize The fixed size of the batch to analyze.
 */
void db_print_batch_joint_stats(struct ImageDatabase *db, unsigned long batchSize)
{
    fprintf(stderr, RED "db_print_batch_joint_stats is not properly implemented\n" NORMAL);
    exit(0);

    if (db == NULL || db->pdb == NULL || batchSize == 0) {
        fprintf(stderr, RED "Invalid database or batch size\n" NORMAL);
        return;
    }

    unsigned long totalBatches = db->pdb->numberOfSamples / batchSize;
    if (db->pdb->numberOfSamples % batchSize != 0) {
        totalBatches++; // Account for partial batch
    }

    unsigned long totalSamplesWithJoints = 0;
    unsigned long totalSamplesWithoutJoints = 0;
    unsigned long sampleNumber;


    unsigned long batchNumber   = 0;
    //unsigned long totalBatches  = (unsigned long) db->numberOfSamples / batchSize;
    for (batchNumber=0; batchNumber<totalBatches; batchNumber++)
        {
            unsigned long start    = (batchNumber+0)*batchSize;
            unsigned long end      = (batchNumber+1)*batchSize;

    // Iterate through all samples to count those with and without joints
    for (sampleNumber = start; sampleNumber < end; sampleNumber++)
    {
        //TODO: FIX THIS
        if (db->pdb->sample[sampleNumber].numberOfSkeletons > 0)
        {
            totalSamplesWithJoints++;
        } else
        {
            totalSamplesWithoutJoints++;
        }
    }

        }

    // Calculate average per batch
    float avgSamplesWithJoints = (float)totalSamplesWithJoints / totalBatches;
    float avgSamplesWithoutJoints = (float)totalSamplesWithoutJoints / totalBatches;

    // Calculate percentages
    float percentageWithJoints = (avgSamplesWithJoints / batchSize) * 100.0;
    float percentageWithoutJoints = (avgSamplesWithoutJoints / batchSize) * 100.0;

    // Print statistics
    fprintf(stderr, "Batch Joint Statistics (Batch Size: %lu):\n", batchSize);
    fprintf(stderr, "Total number of batches: %lu\n", totalBatches);
    fprintf(stderr, "Average samples with joints per batch: %.2f (%.2f%%)\n",
            avgSamplesWithJoints, percentageWithJoints);
    fprintf(stderr, "Average samples without joints per batch: %.2f (%.2f%%)\n",
            avgSamplesWithoutJoints, percentageWithoutJoints);
}

//----------------------------------------------------------------------------------------------------

void db_print_stats(struct ImageDatabase * db)
{
  if ( (db!=0) && (db->pdb!=0) )
  {
   SampleNumber sampleNumber = 0;

   unsigned long totalSamplesWithBackgrounds = 0;
   unsigned long totalSamplesWithSkeletons   = 0;

   unsigned int  minWidth  = 100000;
   unsigned int  minHeight = 100000;

   unsigned long totalHeightWithBackground   = 0;
   unsigned long totalWidthWithBackground    = 0;
   unsigned long totalHeightWithSkeletons    = 0;
   unsigned long totalWidthWithSkeletons     = 0;

   unsigned long totalSkeletons              = 0;
   unsigned long maxSkeletons                = 0;

   fprintf(stderr,"Counting skeletons ... \n");
   for (sampleNumber=0; sampleNumber<db->pdb->numberOfSamples; sampleNumber++)
    {
       unsigned long thisSampleSkeletons = db->pdb->sample[sampleNumber].numberOfSkeletons;

       if (minWidth > db->pdb->sample[sampleNumber].width)
              { minWidth = db->pdb->sample[sampleNumber].width; }

       if (minHeight > db->pdb->sample[sampleNumber].height)
              { minHeight = db->pdb->sample[sampleNumber].height; }

       if ( thisSampleSkeletons == 0 )
       {
           totalSamplesWithBackgrounds += 1;
           totalWidthWithBackground    += (unsigned long) db->pdb->sample[sampleNumber].width  / 10;
           totalHeightWithBackground   += (unsigned long) db->pdb->sample[sampleNumber].height / 10;
       } else
       {
           totalSamplesWithSkeletons   += 1;
           totalWidthWithSkeletons     += (unsigned long) db->pdb->sample[sampleNumber].width  / 10;
           totalHeightWithSkeletons    += (unsigned long) db->pdb->sample[sampleNumber].height / 10;

           if (maxSkeletons<thisSampleSkeletons)
           {
               maxSkeletons = thisSampleSkeletons;
           }

           totalSkeletons += thisSampleSkeletons;
       }
    }

       unsigned long total = totalSamplesWithBackgrounds+totalSamplesWithSkeletons;
       fprintf(stderr,"Total samples                            : %lu \n",total);
       fprintf(stderr,"Total samples that are just background   : %lu \n",totalSamplesWithBackgrounds);
       fprintf(stderr,"Total samples that have skeletons        : %lu \n",totalSamplesWithSkeletons);
       fprintf(stderr,"Max number of skeletons in an image      : %lu \n",maxSkeletons);
       if (totalSamplesWithSkeletons!=0)
         { fprintf(stderr,"Average number of skeletons in sk. image : %lu \n",(unsigned long)  totalSkeletons / totalSamplesWithSkeletons); }

       if (totalSamplesWithBackgrounds!=0)
         { fprintf(stderr,"Ratio of skeleton vs background images   : %0.2f %% \n",(float)  (100.0*totalSamplesWithSkeletons) / totalSamplesWithBackgrounds); }

       if (total!=0)
         { fprintf(stderr,"Skeleton image Ratio                     : %0.2f %% \n",(float)  (100.0*totalSamplesWithSkeletons) / total);
           fprintf(stderr,"Background image Ratio                   : %0.2f %% \n",(float)  (100.0*totalSamplesWithBackgrounds) / total);
         }

       if (totalSamplesWithBackgrounds==0)
       {
         fprintf(stderr,"No background average dimension\n");
       } else
       {
         unsigned long avgBkgWidth  = (unsigned long) 10 * ((unsigned long) totalWidthWithBackground/totalSamplesWithBackgrounds);
         unsigned long avgBkgHeight = (unsigned long) 10 * ((unsigned long) totalHeightWithSkeletons/totalSamplesWithBackgrounds);
         fprintf(stderr,"Average sample dimension with background   : %lu x %lu \n",avgBkgWidth,avgBkgHeight);
       }

      if (totalSamplesWithSkeletons==0)
       {
         fprintf(stderr,"No skeleton average dimension\n");
       } else
       {
         unsigned long avgSkWidth  = (unsigned long) 10 * ((unsigned long) totalWidthWithSkeletons/totalSamplesWithSkeletons);
         unsigned long avgSkHeight = (unsigned long) 10 * ((unsigned long) totalHeightWithSkeletons/totalSamplesWithSkeletons);
         fprintf(stderr,"Average sample dimension with skeletons    : %lu x %lu \n",avgSkWidth,avgSkHeight);
         fprintf(stderr,"Minimum sample dimensions with skeletons   : %u x %u \n",minWidth,minHeight);
       }


  }
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
void db_disable_heatmap_output(struct ImageDatabase * db)
{
   if (db!=0)
   {
     db->doHeatmapOutput = 0;
     fprintf(stderr,"\n\n\n\n\n DISABLING HEATMAP OUTPUT TO SPEED UP TOKEN TRAINING \n\n\n\n\n");
     usleep(100000);
   }
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

//db->add flags have to have been set!
unsigned int calculateNumberOf8BitHeatmaps(int keypointsForEachSample,
                                           int addPAFs,
                                           int addBackgroundHeatmap,
                                           int addDepthMap,
                                           int addDepthLevelsHeatmaps,
                                           int addNormals,
                                           int addSegmentation)
{
     unsigned int channelsOut8BitUsed = keypointsForEachSample +
                                        addBackgroundHeatmap +
                                        addDepthMap +
                                        addDepthLevelsHeatmaps +
                                        addNormals*3 +
                                        addPAFs*12 +
                                        (addSegmentation * (VALID_SEGMENTATIONS)) +
                                        (3*ENABLE_DENOISING_OUTPUT) +
                                        (2*ENABLE_LEFT_RIGHT_JOINT_DISAMBIGUATION_OUTPUT);
      return channelsOut8BitUsed;
}


void * db_create(struct DatabaseList* dbSources,
                 unsigned long numberOfSamples,
                 int streamData,
                 int batchSize,
                 int workerThreads,
                 int gradientSize,
                 int PAFSize,
                 int doAugmentations,
                 int addPAFs,
                 int addBackgroundHeatmap,
                 int addDepthMap,
                 int addDepthLevelsHeatmaps,
                 int addNormals,
                 int addSegmentation,
                 int bytesPerDepthValue,
                 unsigned int widthIn,  unsigned int heightIn,unsigned int channelsIn,
                 unsigned int widthOut, unsigned int heightOut,unsigned int channelsOut8Bit,unsigned int channelsOut16Bit)
{
  fprintf(stderr,"\n\n%s DataLoader \n",version);
  fprintf(stderr,"PThread Worker Pool v%s\n\n",pthreadWorkerPoolVersion);

  instanceCounter+=1;
  fprintf(stderr,"Instance #%u\n",instanceCounter);

  srand((unsigned int)time(NULL)); // Seed the random number generator

  if (dbSources==NULL) { return 0; }
  struct DatabaseList * dbl = (struct DatabaseList*) dbSources;

  fprintf(stderr,"db_create from %u sources \n",dbl->numberOfSources);
  fprintf(stderr,"Asked for %lu samples\n",numberOfSamples);
  fprintf(stderr,"In  : %ux%u:%u\n",widthIn,heightIn,channelsIn);
  fprintf(stderr,"8-Bit Out  : %ux%u:%u\n",widthOut,heightOut,channelsOut8Bit);
  fprintf(stderr,"16-Bit Out : %ux%u:%u\n",widthOut,heightOut,channelsOut8Bit);


   #if PROFILE_THREAD_ACTIVITY
     fprintf(stderr,RED "PROFILING THREAD ACTIVITY IS ON, YOU DONT WANT THIS WHEN ACTUALLY TRAINING\n" NORMAL);
     int ret = system("rm *.log");
     if (ret!=0) { fprintf(stderr,"Failed cleaning previous run log..\n"); }
   #endif // PROFILE_THREAD_ACTIVITY

  struct PoseDatabase* pdb = createPoseDatabase(dbl->totalNumberOfSamples,dbl->numberOfKeypointsPerSample);
  if (pdb!=0)
  {
     DatasetSourceID sID = 0;
     unsigned long startOffset = 0;
     for (sID=0; sID<dbl->numberOfSources; sID++)
     {
       fprintf(stderr,"Reading source #%lu | %s / %s  / %s / %s\n", sID, dbl->source[sID].pathToDBFile,dbl->source[sID].pathToCOCOImages,dbl->source[sID].pathToCOCODepthMaps,dbl->source[sID].pathToCOCOSegmentations);

       #if USE_DINOV2_FEATURES
       char dinoFilename[2048]={0};
       //snprintf(dinoFilename,2048,"%s.dinov2", dbl->source[sID].pathToDBFile);
       snprintf(dinoFilename, sizeof(dinoFilename), "%.*s.dinov2",(int)(sizeof(dinoFilename) - 8),  dbl->source[sID].pathToDBFile);// leave room for ".dinov2" and null terminator

       dbl->source[sID].descriptorsAsADataset=load_descriptor_bin(dinoFilename); //TODO: Deallocate this..
       if (dbl->source[sID].descriptorsAsADataset!=0) { fprintf(stderr,GREEN "Found a DinoV2 Descriptor file in %s \n" NORMAL,dinoFilename); } else
                                                      { fprintf(stderr,RED "Could not find a DinoV2 Descriptor file in %s \n" NORMAL,dinoFilename); }
       #else
         fprintf(stderr,"Dino V2 loading logic not included in this build\n");
       #endif // USE_DINOV2_FEATURES

       unsigned long loadedSamples = 0;
       if (readPoseDatabase(pdb,dbl->source[sID].descriptorsAsADataset,dbl->source[sID].pathToDBFile,sID,startOffset,&loadedSamples,dbl->source[sID].ignoreNoSkeletonSamples))
       {
         if (dbl->source[sID].ignoreNoSkeletonSamples)
         {
             fprintf(stderr,"Ignoring background samples, so artificially reducing samples\n");
             fprintf(stderr,"from %lu to %lu \n",dbl->source[sID].numberOfSamples,loadedSamples);
             dbl->source[sID].numberOfSamples = loadedSamples; //INTERRUPTED
         }
         startOffset += dbl->source[sID].numberOfSamples;
       } else
       {
         fprintf(stderr,RED "Failed reading from: `%s`\n" NORMAL,dbl->source[sID].pathToDBFile);
         exit(1);
       }
     }

     pdb->numberOfSamples = dbl->totalNumberOfSamples;
     if (numberOfSamples!=0)
     {
         fprintf(stderr,"We will only use %lu/%lu samples \n",numberOfSamples,pdb->numberOfSamples);
         pdb->numberOfSamples = numberOfSamples;
     }

     /*
     unsigned int channelsOut8BitEstimated  = pdb->keypointsForEachSample + addBackgroundHeatmap + addDepthMap +
                                              addDepthLevelsHeatmaps + addNormals*3 + addPAFs*12 +
                                             (addSegmentation * (VALID_SEGMENTATIONS)) + (3*ENABLE_DENOISING_OUTPUT) + (2*ENABLE_LEFT_RIGHT_JOINT_DISAMBIGUATION_OUTPUT); */

     unsigned int channelsOut8BitEstimated  = calculateNumberOf8BitHeatmaps(pdb->keypointsForEachSample,
                                                                            addPAFs,
                                                                            addBackgroundHeatmap,
                                                                            addDepthMap,
                                                                            addDepthLevelsHeatmaps,
                                                                            addNormals,
                                                                            addSegmentation);

     unsigned int channelsOut16BitEstimated = addDepthMap;
     if ( (channelsOut8BitEstimated != channelsOut8Bit) || (channelsOut16BitEstimated != channelsOut16Bit) )
     {
         fprintf(stderr,RED "Given a weird number of 8-Bit/16-Bit heatmap output channels, stopping\n" NORMAL);
         fprintf(stderr,RED "Estimated and expected %u heatmaps but db_create given %u 8-Bit channels\n" NORMAL,channelsOut8BitEstimated,channelsOut8Bit);
         fprintf(stderr,RED "Estimated and expected %u heatmaps but db_create given %u 16-Bit channels\n" NORMAL,channelsOut16BitEstimated,channelsOut16Bit);
         fprintf(stderr,"Keypoints: %u \n",pdb->keypointsForEachSample);
         fprintf(stderr,"addPAFs: %u \n",addPAFs);
         fprintf(stderr,"addDepthMap: %u \n",addDepthMap);
         fprintf(stderr,"addNormals: %u \n",addNormals);
         fprintf(stderr,"addSegmentation: %u \n",addSegmentation);
         fprintf(stderr,"addBackgroundHeatmap: %u \n",addBackgroundHeatmap);
         fprintf(stderr,"bytesPerDepthValue: %u \n",bytesPerDepthValue);
         abort();
     }

     //By default we suppose we use the same number of images as samples
     unsigned long numberOfImages = pdb->numberOfSamples;
     if (streamData)
     {
         numberOfImages = batchSize;
         fprintf(stdout,GREEN "DataLoader setup to stream %lu samples using batches of %lu..!\n" NORMAL,pdb->numberOfSamples,numberOfImages);
     }


     struct ImageDatabase * db = db_create_without_dataset(pdb->numberOfSamples,numberOfImages,widthIn,heightIn,channelsIn,widthOut,heightOut,channelsOut8Bit,channelsOut16Bit);
     if (db!=0)
     {
      db->doAugmentations = (char) doAugmentations;
      db->dbSources = dbl;
      db->pdb       = pdb;

      //Always do heatmap output (it can be hackily disabled later if needed)
      db->doHeatmapOutput         = 1;

      //This is important because it routes heatmaps
      db->addPAFHeatmap           = addPAFs;
      db->addBackgroundHeatmap    = addBackgroundHeatmap;
      db->addDepthHeatmap         = addDepthMap;
      db->addDepthLevelsHeatmaps  = addDepthLevelsHeatmaps;
      db->addNormalHeatmaps       = addNormals;
      db->addSegmentationHeatmaps = addSegmentation;

      //WHERE DOES EACH THING GOES IS CONTROLLED HERE
      int startOfDepth = 17;
      if (db->addPAFHeatmap)
       {
           db->PAFHeatmapIndex = 17;
           startOfDepth        = 29;
       }

      db->segmentationHeatmapIndex = startOfDepth + 3 + 6 + 1; //21
      //fprintf(stderr,"db->segmentationHeatmapIndex = %u\n",db->segmentationHeatmapIndex);
      //exit(0);
      db->backgroundHeatmapIndex   = startOfDepth;
      db->depthmapHeatmapIndex8Bit = startOfDepth;
      if (db->addNormalHeatmaps)
       {
         db->normalsHeatmapIndex = db->depthmapHeatmapIndex8Bit + 1;
       }


      //---------------------------------------------
      db->depthmapHeatmapIndex16Bit = 0;


      if (!streamData)
      {
       db_update(db,
                 0,
                 db->numberOfSamples,
                 workerThreads,
                 gradientSize,
                 PAFSize);
      } else
      {
        fprintf(stderr,"Not loading data since we will be streaming..!\n");
      }

      fprintf(stdout,"\nDone loading full dataset..!\n");
      db_print_stats(db); // Printout stats, takes a little time useful to manually check for configuration nmistakes


      #if USE_RAM_CACHE
        db->files = cache_init(128 + pdb->numberOfSamples * 3);
        preloadAllFiles(db);
      #else
        //Just initialize for commonCache mechanism
        db->files = cache_init(0);
      #endif // USE_RAM_CACHE

      return db;
     }

     fprintf(stderr,"Destroying Pose Database\n");
     //If we reach this part we failed
     freePoseDatabase(pdb);
  }

  return 0;
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
int db_destroy(struct ImageDatabase * db)
{
    if (db!=0)
    {
        //-------------------
        db_stop_threads(db);
        //-------------------
        if (db->in.pixels!=0)
        {
          free(db->in.pixels);
          db->in.pixels=0;
        }

        if (db->out8bit.pixels!=0)
        {
          free(db->out8bit.pixels);
          db->out8bit.pixels=0;
        }

        if (db->out16bit.pixels!=0)
        {
          free(db->out16bit.pixels);
          db->out16bit.pixels=0;
        }

        if (db->indices!=0)
        {
          free(db->indices);
          db->indices=0;
        }

        if (db->losses!=0)
        {
          free(db->losses);
          db->losses=0;
        }

        if (db->trainPasses!=0)
        {
            free(db->trainPasses);
            db->trainPasses=0;
        }

        if (db->pdb!=0)
        {
            freePoseDatabase(db->pdb);
            db->pdb = 0;
        }

        if (db->dbSources!=0)
        {
            db_destroy_source_list((void *) db->dbSources);
            db->dbSources = 0;
        }

        if (db->gradients!=0)
        {
         free_heatmap_collection(db->gradients);
         db->gradients = 0;
        }

        //Destroy cache
        cache_destroy(db->files);

         #if USE_CANARY
         if (db->canaryA.shouldRemainUntouched!=0)
         {
          free(db->canaryA.shouldRemainUntouched);
          db->canaryA.shouldRemainUntouched=0;
         }
         if (db->canaryB.shouldRemainUntouched!=0)
         {
          free(db->canaryB.shouldRemainUntouched);
          db->canaryB.shouldRemainUntouched=0;
         }
         if (db->canaryC.shouldRemainUntouched!=0)
         {
          free(db->canaryC.shouldRemainUntouched);
          db->canaryC.shouldRemainUntouched=0;
         }
         #endif // USE_CANARY

        //-------------------
        free(db);
        return 1;
    }

    return 0;
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
struct Image * db_map_sample_out8bit_to_image(struct ImageDatabase * db,unsigned long sampleNumber, unsigned int keypointNumber)
{
  if (db!=0)
  {
   if (sampleNumber<db->numberOfSamples)
   {
    struct Image * newImg  = db_create_8Bit_Image_from_heatmap(db,sampleNumber,keypointNumber);
    if ( (newImg!=0) && (newImg->pixels!=0) )
    {
      return newImg;
    } else
    {
      fprintf(stderr,"Failed mapping sample %lu heatmap 8-bit %u as an Image\n",sampleNumber,keypointNumber);
      destroyImage(newImg);
    }
   }
  }
  return 0;
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
struct Image * db_map_sample_out16bit_to_image(struct ImageDatabase * db,unsigned long sampleNumber, unsigned int keypointNumber)
{
  if (db!=0)
  {
   if (sampleNumber<db->numberOfSamples)
   {
    struct Image * newImg  = db_create_16Bit_Image_from_heatmap_16bit(db,sampleNumber,keypointNumber);
    if ( (newImg!=0) && (newImg->pixels!=0) )
    {
      return newImg;
    } else
    {
      fprintf(stderr,"Failed mapping sample %lu heatmap 8-bit %u as an Image\n",sampleNumber,keypointNumber);
      destroyImage(newImg);
    }
   }
  }
  return 0;
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
int db_set_priority(int priority)
{
    return elevate_nice_priority(priority);
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
void db_save_image(struct ImageDatabase * db,const char * filename,unsigned long sampleNumber)
{
 struct Image tmp={0};
 if (db_map_sample_in_to_image(db,&tmp,sampleNumber))
              { writeImageFile(&tmp,JPG_CODEC,filename); }
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
void db_save_heatmap8bit_as_jpg(struct ImageDatabase * db,const char * filename,unsigned long sampleNumber, unsigned short heatmapNumber)
{
 struct Image *tmp = db_map_sample_out8bit_to_image(db,sampleNumber,heatmapNumber);
 if (tmp!=0) {
                 convert_sint8ImageTouint8(tmp);
                 writeImageFile(tmp,JPG_CODEC,filename); //
                 destroyImage(tmp);
              }
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
void db_save_heatmap8bit_as_png(struct ImageDatabase * db,const char * filename,unsigned long sampleNumber, unsigned short heatmapNumber)
{
 struct Image *tmp = db_map_sample_out8bit_to_image(db,sampleNumber,heatmapNumber);
 if (tmp!=0) {
                 convert_sint8ImageTouint8(tmp);
                 writeImageFile(tmp,PNG_CODEC,filename); //
                 destroyImage(tmp);
              }
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
int db_check_heatmap8bit_empty(struct ImageDatabase * db,unsigned long sampleNumber,unsigned short heatmapNumber)
{
 struct Image *tmp = db_map_sample_out8bit_to_image(db,sampleNumber,heatmapNumber);
 if (tmp!=0) {
               convert_sint8ImageTouint8(tmp);
               signed char * ptr = (signed char*) tmp->pixels;
               for (int y=0; y<tmp->height; y++)
                 {
                  for (int x=0; x<tmp->width; x++)
                  {
                   //for (int c=0; c<tmp->channels; c++) //This is a single 8bit heatmap
                   {
                     if (*ptr!=0) // && (*ptr!=MINV)  //convert_sint8ImageTouint8 should make it zero always
                     {
                         destroyImage(tmp);
                         return 0;
                     }
                     ptr+=1;
                   }
                  }
                 }
               destroyImage(tmp);
              }
   return 1;
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
void db_save_heatmap16bit_as_pnm(struct ImageDatabase * db,const char * filename,unsigned long sampleNumber, unsigned short heatmapNumber)
{
 struct Image *tmp = db_map_sample_out16bit_to_image(db,sampleNumber,heatmapNumber);
 if (tmp!=0) {
                 swap16bitEndianness(tmp);
                 writeImageFile(tmp,PNM_CODEC,filename);
                 destroyImage(tmp);
              }
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

// Function to write the strings associated with each sample to a file
void db_save_descriptions_to_file(struct ImageDatabase *db, const char *filename, unsigned long sampleNumber, struct StringArray * labels)
{
    if (labels==0) { fprintf(stderr,"db_save_descriptions_to_file with empty labels\n"); return; }

    // Open the output file for writing
    FILE *file = fopen(filename, "w");
    if (!file)
    {
        perror("Error opening file");
        return;
    }

    // Get the number of tokens for the current sample
    int numTokens = db_get_sample_description_tokens_number(db, sampleNumber);
    if (numTokens > 0)
        {
            // Get the tokens (IDs) associated with the current sample
            unsigned short *tokens = db_get_sample_description_tokens(db, sampleNumber);

            // Write the sample ID (optional)
            //fprintf(file, "Sample %lu: ", sampleNumber);

            // Write each token's associated string to the file
            //---------------------------------------------------------
            for (int i = 0; i < numTokens; i++)
            {
                const char *label = labels->strings[tokens[i]]; //numTokens should always be ok right?
                if (label) { fprintf(file, "%s ", label); }
            }
            // End the line for this sample
            fprintf(file, "\n");
            //---------------------------------------------------------


            // Write each token ID straight out for debugging
            //---------------------------------------------------------
            for (int i = 0; i < numTokens; i++)
            {
                fprintf(file, "%u ", tokens[i]);
            }
            // End the line for this sample
            fprintf(file, "\n");
            //---------------------------------------------------------


            //Write the embedding vector
            //---------------------------------------------------------
            //We could loop this just for numTokens but we will do it for MAX_DESCRIPTION_TOKENS
            //to also peak at what the dataloader responds with as an empty vector
            //Keep in mind that returned values are normalized via offset/scaling parameters in start
            //of embedding file!
            for (int i = 0; i < MAX_DESCRIPTION_TOKENS; i++) //numTokens
              {
               float * embeddings = db_get_sample_description_embeddings(db, sampleNumber, i);
               if (embeddings!=0)
               {
                fprintf(file, "Embedding Vector for token %u :\n",i);
                for (int embID = 0; embID < db->pdb->embeddings.D; embID++)
                {
                  fprintf(file, "%f ", embeddings[embID]);
                } //Loop over all embeddings
                 fprintf(file, "\n");
               }
              }// Loop over all tokens
            //---------------------------------------------------------

        }

    // Close the file
    fclose(file);
}
//----------------------------------------------------------------------------------------------------
// Function to write the strings associated with each sample to a file
//----------------------------------------------------------------------------------------------------
void db_dump_all_descriptions_to_file(struct ImageDatabase *db, const char *filename, struct StringArray * labels)
{
    fprintf(stderr,"Dumping all labels to %s file\n",filename);
    // Open the output file for writing
    FILE *file = fopen(filename, "w");
    if (!file)
    {
        perror("Error opening file");
        return;
    }

    for (unsigned long sampleNumber=0; sampleNumber<db->numberOfSamples; sampleNumber++)
    {
    // Get the number of tokens for the current sample
    int numTokens = db_get_sample_description_tokens_number(db, sampleNumber);
    if (numTokens > 0)
        {
            // Get the tokens (IDs) associated with the current sample
            unsigned short *tokens = db_get_sample_description_tokens(db, sampleNumber);

            // Write the sample ID (optional)
            //fprintf(file, "Sample %lu: ", sampleNumber);

            // Write each token's associated string to the file
            for (int i = 0; i < numTokens; i++)
            {
                const char *label = labels->strings[tokens[i]];
                if (label) { fprintf(file, "%s\n", label); }
            }

            // End the line for this sample
            //fprintf(file, "\n");
        }
    }

    // Close the file
    fclose(file);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//             The rest of the DataLoader is a test function that might be
//       called as an internal unit-test/benchmark of the DataLoader Functionality
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
int test16BitDepth()
{
  fprintf(stderr,"Do mini png loader test");
  //struct Image* dpth = readImage("./FORTH_HER_DRONE2-8bit.png",PNG_CODEC,0);
  struct Image* dpth = readImage("../coco/cache/coco/depth_val2017/000000021604.png",PNG_CODEC,0);


  fprintf(stderr,"dpth->channels = %u\n",dpth->channels);
  fprintf(stderr,"dpth->bitsperpixel = %u\n",dpth->bitsperpixel);
  if (dpth!=0)
  {
    writeImageFile(dpth,PNM_CODEC,"TEST.pnm");
    struct Image * normals = createImage(dpth->width, dpth->height, 3, 48);
    if (normals!=0)
    {

    swap16bitEndianness(dpth);
    computeNormals(dpth,normals);

    struct Image * normals8bit = createImage(dpth->width, dpth->height, 3, 24);
    if (normals8bit!=0)
    {
     convert16bitTo8bit(normals,normals8bit);
     writeImageFile(normals8bit,PNM_CODEC,"TEST_NORMALS_16bit_TO_8bit.pnm");
     destroyImage(normals8bit);
    }

    swap16bitEndianness(normals);
    writeImageFile(normals,PNM_CODEC,"TEST_NORMALS.pnm");
    swap16bitEndianness(normals);
    bilateralFilter(normals,9,75.0,75.0);
    swap16bitEndianness(normals);
    writeImageFile(normals,PNM_CODEC,"TEST_NORMALS_BILATERAL.pnm");

     destroyImage(normals);
    }
    destroyImage(dpth);
    return 1;
  }
 return 0;
}

//========================================================================================
//========================================================================================
//========================================================================================
//========================================================================================

int test(int argc, char *argv[])
{
  fprintf(stderr,"Running test code!\n\n\n");

  fprintf(stderr,"Cleaning previous files : ");
  int result = system("rm sample*.jpg sample*.pnm sample*.png sample*.txt heatmap*.png one_hot*.png");
  if (result==0) { fprintf(stderr,"Success\n"); } else { fprintf(stderr,"Failed\n"); }
  //--------------------------------
  int useValidationData         = 0;
  int useRAM                    = 0;
  int doAugmentations           = 1;
  int addBackgroundHeatmap      = 0;
  int addDepthMap               = 1;
  int addDepthLevelsHeatmaps    = DEPTH_LEVELS_HEATMAPS;
  int addNormals                = 1;
  int addSegmentation           = 1;
  int addPAFs                   = 1;
  int ignoreNoSkeletonSamples   = 0;
  //--------------------------------
  unsigned long numberOfSamples = 0; // Set to 300 for fast test We only load a few samples to speed up valgrind etc.<- 0 = auto
  unsigned int batchSize        = 32;
  unsigned int numberOfThreads  = 8;
  unsigned int gradientSize     = 22; //Take into account the db_change_joint_difficulty calls below
  //unsigned int gradientSize     = 8 + 2; //Take into account the db_change_joint_difficulty calls below
  unsigned int PAFSize          = 5;

  //New One gradient/paf for whole training!
  gradientSize = 12;
  PAFSize      = 2;
  //--------------------------------
  unsigned int widthIn     = 256; //256 300 420
  unsigned int heightIn    = 256; //256 300 420
  unsigned int channelsIn  = 3;
  //--------------------------------
  unsigned int widthOut    = 256; //256 384
  unsigned int heightOut   = 256; //256 384
  int depthBytes           = 2;// 2 if 16 bit / 1 if 8 bit
  /*
  unsigned int channelsOut8Bit  = 17 + addBackgroundHeatmap + addDepthMap + addDepthLevelsHeatmaps + (12 * addPAFs) +
                                  (3 * addNormals) + (addSegmentation* (getNumberOfValidInstanceLabels()) ) +
                                  (3*ENABLE_DENOISING_OUTPUT) + (2*ENABLE_LEFT_RIGHT_JOINT_DISAMBIGUATION_OUTPUT); */

  unsigned int channelsOut8Bit  = calculateNumberOf8BitHeatmaps(17,
                                                                addPAFs,
                                                                addBackgroundHeatmap,
                                                                addDepthMap,
                                                                addDepthLevelsHeatmaps,
                                                                addNormals,
                                                                addSegmentation);

  unsigned int channelsOut16Bit = addDepthMap;
  //--------------------------------

  int profilingRun        = 0;
  int saveFilesToSeeThem  = 1;
  int saveAllLabels       = 0;
  int processDescriptions = 0;


  char path[1024];//="./";
  snprintf(path,1024,"./");

  fprintf(stderr,"Sanity check struct sizes : \n");
  fprintf(stderr,"struct Heatmaps: %lu bytes \n",sizeof(struct Heatmaps));
  fprintf(stderr,"struct DatabaseEntry: %lu bytes \n",sizeof(struct DatabaseEntry));
  fprintf(stderr,"struct DatabaseList: %lu bytes \n",sizeof(struct DatabaseList));
  fprintf(stderr,"struct ImageDatabase: %lu bytes \n\n",sizeof(struct ImageDatabase));

  if (argc>0)
  {
      for (int i=0; i<argc; i++)
      {
        if (strcmp(argv[i],"--size")==0)
                {
                    widthIn   = atoi(argv[i+1]);
                    heightIn  = atoi(argv[i+2]);
                    widthOut  = atoi(argv[i+1]);
                    heightOut = atoi(argv[i+2]);
                    fprintf(stderr,GREEN "Size set to %ux%u\n" NORMAL,widthIn,heightIn);
                } else
        if (strcmp(argv[i],"--batch")==0)
                {
                    batchSize = atoi(argv[i+1]);
                    fprintf(stderr,GREEN "Batch set to %u\n" NORMAL,batchSize);
                } else
        if (strcmp(argv[i],"--threads")==0)
                {
                    numberOfThreads = atoi(argv[i+1]);
                    fprintf(stderr,GREEN "Threads set to %u\n" NORMAL,numberOfThreads);
                } else
        if (strcmp(argv[i],"--ram")==0)
                {
                    useRAM = 1;
                } else
        if (strcmp(argv[i],"--validation")==0)
                {
                    useValidationData = 1;
                    doAugmentations   = 0;
                    gradientSize      = 12;
                    PAFSize           = 2;
                } else
        if (strcmp(argv[i],"--profile")==0)
                {
                    saveFilesToSeeThem = 0;
                    profilingRun = 1;
                } else
        if (strcmp(argv[i],"-o")==0)
                {
                    snprintf(path,1024,"%s",argv[i+1]);
                } else
        if (strcmp(argv[i],"--savelabels")==0)
                {
                    saveAllLabels = 1;
                } else
        if (strcmp(argv[i],"--descriptions")==0)
                {
                    processDescriptions = 1;
                } else
        if (strcmp(argv[i],"--rt")==0)
                {
                   if (db_set_priority(-20))
                   {
                       fprintf(stderr,GREEN "Sucessfully changed priority\n" NORMAL);
                       //drop_privileges(0);
                   } else
                   {
                       fprintf(stderr,RED "Failed changed priority\n" NORMAL);
                   }
                }


      }
  }

  //Uncomment to test 16 bit detph code
  //test16BitDepth();
  //exit(0);
  #define ONLYVAL2017 1

  unsigned long startTime = GetTickCountMicrosecondsMN();

  signed int sourcesRemaining=0, sourceCounter=0;
  void * dbl=0;

  if (useRAM)
  {
   sourcesRemaining = 10;
   sourceCounter    = 0;
   dbl = db_allocate_source_list(sourcesRemaining);

   //Comment all datasets underneath for little val test
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/coco14/cocoVal14.db",  "../../../ram/datasets/coco14/val2014", "../../../ram/datasets/coco14/depth_val2014", "../../../ram/datasets/coco14/segment_val2014", ignoreNoSkeletonSamples);

   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/coco/cocoTrain.db",            "../../../ram/datasets/coco/cache/coco/train2017", "../../../ram/datasets/coco/cache/coco/depth_train2017", "../../../ram/datasets/coco/cache/coco/segment_train2017", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/coco14/cocoTrain14.db",        "../../../ram/datasets/coco14/train2014",          "../../../ram/datasets/coco14/depth_train2014",          "../../../ram/datasets/coco14/segment_train2014", ignoreNoSkeletonSamples);

   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/300w/indoor.db",          "../../../ram/datasets/300w/indoor",            "../../../ram/datasets/300w/depth_indoor",            "../../../ram/datasets/300w/segment_indoor", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/300w/outdoor.db",         "../../../ram/datasets/300w/outdoor",           "../../../ram/datasets/300w/depth_outdoor",           "../../../ram/datasets/300w/segment_outdoor", ignoreNoSkeletonSamples);

   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/background/AM-2k.db",          "../../../ram/datasets/background/AM-2k/train",    "../../../ram/datasets/background/AM-2k/depth_train",    "../../../ram/datasets/background/AM-2k/segment_train", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/background/BG-20k.db",         "../../../ram/datasets/background/BG-20k/train",   "../../../ram/datasets/background/BG-20k/depth_train",   "../../../ram/datasets/background/BG-20k/segment_train", ignoreNoSkeletonSamples);

   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/openpose/openposeTrain.db",    "../../../ram/datasets/openpose/data/train",       "../../../ram/datasets/openpose/data/depth_train",       "../../../ram/datasets/openpose/data/segment_train", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/openpose/openposeBKG.db",      "../../../ram/datasets/openpose/data/bkg",         "../../../ram/datasets/openpose/data/depth_bkg" ,        "../../../ram/datasets/openpose/data/segment_bkg", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/openpose/openposeFactory.db",  "../../../ram/datasets/openpose/data/factory",     "../../../ram/datasets/openpose/data/depth_factory" ,    "../../../ram/datasets/openpose/data/segment_factory", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/openpose/openposeFactory2.db", "../../../ram/datasets/openpose/data/factory2",    "../../../ram/datasets/openpose/data/depth_factory2" ,   "../../../ram/datasets/openpose/data/segment_factory2", ignoreNoSkeletonSamples);
  } else
  if (useValidationData)
  {
    sourcesRemaining = 2;
    sourceCounter    = 0;
    dbl = db_allocate_source_list(sourcesRemaining);
    sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco/cocoVal.db",            "../coco/cache/coco/val2017", "../coco/cache/coco/depth_val2017", "../coco/cache/coco/segment_val2017", ignoreNoSkeletonSamples);
    sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco14/cocoVal14.db",        "../coco14/val2014",          "../coco14/depth_val2014",          "../coco14/segment_val2014", ignoreNoSkeletonSamples);
  } else
  #if ONLYVAL2017
  {
   fprintf(stderr,"Using only validation of 2017 COCO\n");
   sourcesRemaining = 1;
   sourceCounter    = 0;
   dbl = db_allocate_source_list(sourcesRemaining);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco/cocoVal.db",          "../coco/cache/coco/val2017",            "../coco/cache/coco/depth_val2017",            "../coco/cache/coco/segment_val2017", ignoreNoSkeletonSamples);
  }
  #else
  {
   //sourcesRemaining = 1;
   sourcesRemaining = 8;//7;//10;
   sourceCounter    = 0;
   dbl = db_allocate_source_list(sourcesRemaining);

   //Uncomment for little val test
   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco/cocoVal.db",          "../coco/cache/coco/val2017",            "/home/ammar/Documents/Programming/PZP/test/depth_val2017PZP",            "/home/ammar/Documents/Programming/PZP/test/segment_val2017PZP", ignoreNoSkeletonSamples);
   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco/cocoVal.db",          "../coco/cache/coco/val2017",            "../coco/cache/coco/depth_val2017",            "../coco/cache/coco/segment_val2017", ignoreNoSkeletonSamples);

   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../300w/indoor_jpg.db",          "../300w/indoor_jpg",            "../300w/depth_indoor",            "../300w/segment_indoor", ignoreNoSkeletonSamples);
   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../300w/outdoor_jpg.db",         "../300w/outdoor_jpg",           "../300w/depth_outdoor",           "../300w/segment_outdoor", ignoreNoSkeletonSamples);

   //Test PZP file encoding
   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../300w/indoor_jpg.db",          "../300w/indoor_jpg",            "../300w/depth_indoor_pzp",            "../300w/segment_indoor_pzp", ignoreNoSkeletonSamples);
   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../300w/outdoor_jpg.db",         "../300w/outdoor_jpg",           "../300w/depth_outdoor_pzp",           "../300w/segment_outdoor_pzp", ignoreNoSkeletonSamples);


   //Comment all datasets underneath for little val test
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco14/cocoVal14.db",          "../coco14/val2014",            "../coco14/depth_val2014",            "../coco14/segment_val2014", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco/cocoTrain.db",            "../coco/cache/coco/train2017", "../coco/cache/coco/depth_train2017", "../coco/cache/coco/segment_train2017", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco14/cocoTrain14.db",        "../coco14/train2014",          "../coco14/depth_train2014",          "../coco14/segment_train2014", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../background/AM-2k.db",          "../background/AM-2k/train",    "../background/AM-2k/depth_train",    "../background/AM-2k/segment_train", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../background/BG-20k.db",         "../background/BG-20k/train",   "../background/BG-20k/depth_train",   "../background/BG-20k/segment_train", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../openpose/openposeTrain.db",    "../openpose/data/train",       "../openpose/data/depth_train",       "../openpose/data/segment_train", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../openpose/openposeBKG.db",      "../openpose/data/bkg",         "../openpose/data/depth_bkg" ,        "../openpose/data/segment_bkg", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../openpose/openposeFactory.db",  "../openpose/data/factory",     "../openpose/data/depth_factory" ,    "../openpose/data/segment_factory", ignoreNoSkeletonSamples);
   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../openpose/openposeFactory2.db", "../openpose/data/factory2",    "../openpose/data/depth_factory2" ,   "../openpose/data/segment_factory2", ignoreNoSkeletonSamples);

   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../generated/generatedTrain.db",  "../generated/data/train",      "../generated/data/depth_train" ,   "../generated/data/segment_train",   ignoreNoSkeletonSamples);
  }
  #endif // ONLYVAL2017

  if (sourcesRemaining!=0)
  {
      fprintf(stderr,RED "Could not load all sources (%d sources could not be loaded), stopping test\n",sourcesRemaining);
      abort(); //<-- this is more dramatic :P
      return 0;
  }


  struct ImageDatabase*  db = db_create(dbl,
                                        numberOfSamples,
                                        1, //<- 0 = load all data at once / 1 = stream data
                                        batchSize,//Batch size (will be ignored since not streaming)
                                        numberOfThreads,
                                        gradientSize,
                                        PAFSize,
                                        doAugmentations,
                                        addPAFs,
                                        addBackgroundHeatmap,
                                        addDepthMap,
                                        addDepthLevelsHeatmaps,
                                        addNormals,
                                        addSegmentation,
                                        depthBytes,
                                        widthIn,  heightIn,  channelsIn,
                                        widthOut, heightOut, channelsOut8Bit, channelsOut16Bit);

  unsigned long endTime = GetTickCountMicrosecondsMN();
  fprintf(stderr,"Loading %lu samples with %u threads took %lu microseconds\n",numberOfSamples,numberOfThreads,endTime-startTime);
  if (numberOfSamples>0)
    { fprintf(stderr,"%lu microseconds/sample \n",(endTime-startTime)/numberOfSamples); }
  //AMD Ryzen 7 3800X 8-Core Processor
  //Loading 10000 samples with 4 threads took 18913736 microseconds
  //Loading 10000 samples with 12 threads took 8552730 microseconds
  //Loading 10000 samples with 15 threads took 7332806 microseconds
  db_change_joint_difficulty(db,0,-5); //"nose"
  db_change_joint_difficulty(db,1,-5); //"left_eye"
  db_change_joint_difficulty(db,2,-5); //"right_eye"
  db_change_joint_difficulty(db,3,-2); //"left_ear"
  db_change_joint_difficulty(db,4,-2); //"right_ear"
  db_change_joint_difficulty(db,5,0); // "left_shoulder"
  db_change_joint_difficulty(db,6,0); //"right_shoulder"
  db_change_joint_difficulty(db,7,2); //"left_elbow"
  db_change_joint_difficulty(db,8,2); //"right_elbow"
  db_change_joint_difficulty(db,9,5); //"left_wrist"
  db_change_joint_difficulty(db,10,5); //"right_wrist"
  db_change_joint_difficulty(db,11,0); //"left_hip"
  db_change_joint_difficulty(db,12,0); //"right_hip"
  db_change_joint_difficulty(db,13,2); //"left_knee"
  db_change_joint_difficulty(db,14,2); //"right_knee"
  db_change_joint_difficulty(db,15,5); //"left_ankle"
  db_change_joint_difficulty(db,16,5); //"right_ankle"

  if (processDescriptions)
   {
     int ascending = 1;
     int numberOfAllTokens = db->pdb->maxTokenValue;
     fprintf(stderr,"Processing descriptions, sorting tokens\n");
     db_sort_description_tokens_based_on_count(db,numberOfAllTokens,ascending);
     fprintf(stderr,"Processing descriptions, removing duplicates\n");
     db_remove_duplicate_description_tokens(db);
   }

  unsigned long totalStartTime = GetTickCountMicrosecondsMN();
  unsigned long totalEndTime   = 0;

  if (db!=0)
  {
   // Populate the struct from the file
   //const char vocabularySource[]="../descriptions/vocabulary.json";
   const char vocabularySource[]="../../2d_pose_estimation/vocabulary.json";

   //populateStructFromFile(vocabularySource, &labelArray);
   struct StringArray * labelArray = loadStringArrayFromFile(vocabularySource);
   if (labelArray==0)
   {
       fprintf(stderr,RED "Could not open label array %s  \n" NORMAL,vocabularySource);
       abort();
   }

   //This is a debug function to study the label distribution
   //using python3 compute_word_frequencies.py
   if (saveAllLabels)
      {
        fprintf(stderr,GREEN "Save Labels\n" NORMAL);
        db_dump_all_descriptions_to_file(db,"all_labels.txt",labelArray);
        int i = system("python3 compute_word_frequencies.py");
        if (i==0) { fprintf(stderr,GREEN "Computed word frequencies\n" NORMAL); }
        exit(0);
      }

   // Print the populated strings
   //printStringArray(labelArray);
   if (labelArray->count < db->pdb->maxTokenValue)
   {
       fprintf(stderr,RED "We have %u labels in (%s) but maximum %u token values, something is wrong! \n",labelArray->count,vocabularySource,db->pdb->maxTokenValue);
       fprintf(stderr,RED "(probably the labels..) \n" NORMAL);
       abort();
   }


   int hmNumber;
   SampleNumber sampleNumber = 0;
   char filename[2500]={0};

   unsigned long start,end;

   if (saveFilesToSeeThem)
      {
         int numberOfBatchesToDump = 5;

         if (useValidationData)
         {
           fprintf(stderr,"Dumping all valdiation data (you probably want to score them!)\n");
           numberOfBatchesToDump = (unsigned long) db->numberOfSamples / batchSize;
         }

         fprintf(stderr,"Will now attempt to dump %u batches (%u samples each) to disk so you can see them\n",numberOfBatchesToDump,batchSize);
         //Fast test saving skeletons..
         for (int rep=0; rep<numberOfBatchesToDump; rep++)
         {

          fprintf(stderr,"Dumping %u/%u batch..                               \r",rep,numberOfBatchesToDump);

          start = rep*batchSize;
          end   = (rep+1)*batchSize;
          if (db_update(db,start,end,numberOfThreads,gradientSize,PAFSize))
          {
           for (sampleNumber=0; sampleNumber<batchSize; sampleNumber++)
           {
             //Please note that sample number refers to the batch ( [0..batchSize), use db->indices[start+sampleNumber] for direct db->pdb->sample[X] access


             logMainThreadProgress(1,"saving_test");

             //Do sanity check on skeleton keypoints
             if (db_get_sampleNumberOfSkeletons(db,start+sampleNumber)==0)
             {
              for (hmNumber=0; hmNumber<17; hmNumber++)
              {
               //Since there is no skeleton the heatmap should be empty, if it is not..
               if (!db_check_heatmap8bit_empty(db,sampleNumber,hmNumber))
               {
                 fprintf(stderr,YELLOW "sample%lu / Heatmap %u should be empty but it isn't (?) Bug? " NORMAL,start+sampleNumber,hmNumber);
                 SampleNumber sID = db->indices[start+sampleNumber];
                 DatasetSourceID sourceID = db_resolve_sample_sourceID((void *) db,sID);
                 fprintf(stderr,"sample%lu -> sourceID %lu / file %s \n",start+sampleNumber,sourceID,db->pdb->sample[sID].imagePath);


                 snprintf(filename,2000,"%s/sample%05lu_BUG8bit_hm%d.png",path,sampleNumber,hmNumber);
                 db_save_heatmap8bit_as_png(db,filename,sampleNumber,hmNumber);
               }
              }
             }

            snprintf(filename,2000,"%s/sample%05lu_label.txt",path,sampleNumber+(rep*batchSize));
            SampleNumber sID = db->indices[start+sampleNumber];
            db_save_descriptions_to_file(db,filename,sID,labelArray);

            //Dump Image&Heatmaps to disk to view them!
            snprintf(filename,2000,"%s/sample%05lu_IC.jpg",path,sampleNumber+(rep*batchSize));
            db_save_image(db,filename,sampleNumber);

            for (hmNumber=0; hmNumber<channelsOut8Bit; hmNumber++)
            {
             //snprintf(filename,1024,"sample%lu_C8bit_hm%d.jpg",sampleNumber+(rep*batchSize),hmNumber);
             //db_save_heatmap8bit_as_jpg(db,filename,sampleNumber,hmNumber);
             snprintf(filename,2000,"%s/sample%05lu_C8bit_hm%d.png",path,sampleNumber+(rep*batchSize),hmNumber);
             db_save_heatmap8bit_as_png(db,filename,sampleNumber,hmNumber);
            }

            logMainThreadProgress(0,"saving_test");


            #if TEST_16BIT_HEATMAPS
            logMainThreadProgress(1,"saving_16b_test");
            //Dont save the 16bit heatmap
            for (hmNumber=0; hmNumber<channelsOut16Bit; hmNumber++)
            {
             snprintf(filename,2000,"%s/sample%05lu_C16bit_hm%d.jpg",path,sampleNumber+(rep*batchSize),hmNumber);
             db_save_heatmap16bit_as_pnm(db,filename,sampleNumber,hmNumber);
            }

             //This is the same as the .jpg above but takes more space so deactivated
             /*
             struct Image * depth16bit = db_create_16Bit_Image_from_heatmap_16bit(db,sampleNumber,db->depthmapHeatmapIndex16Bit);
             if (depth16bit!=0)
             {
               snprintf(filename,2000,"sample%lu_C16_DEPTH_hm.pnm",sampleNumber+(rep*batchSize));
               swap16bitEndianness(depth16bit);
               WritePPM(filename,depth16bit);
               destroyImage(depth16bit);
             } */

             logMainThreadProgress(0,"saving_16b_test");
             #endif // TEST_16BIT_HEATMAPS

           }
          }
         } //END OF DUMP OF SAVED FILES

         fprintf(stderr,"Finished dumping files..!\n");
       }



        endTime   = 1;
        startTime = 0;
        unsigned long batchNumber   = 0;
        unsigned long totalBatches  = (unsigned long) db->numberOfSamples / batchSize;
        unsigned long cumulTimeMSec = 0;
        unsigned int  thisMSec      = 0;

        if (profilingRun)
           {
             totalBatches = 10;
             fprintf(stderr,"Will now only go through %lu samples to profile code in valgrind\n",totalBatches);
           }// <- limit run for profiling
        else
           { fprintf(stderr,"Will now attempt to go through all %lu samples to identify bugs, this will take time in valgrind mode\n",db->numberOfSamples); }

        totalStartTime = GetTickCountMicrosecondsMN();
        for (batchNumber=0; batchNumber<totalBatches; batchNumber++)
        {
            start    = (batchNumber+0)*batchSize;
            end      = (batchNumber+1)*batchSize;

            if (end>db->numberOfSamples)
            {
                end = db->numberOfSamples;
            }

            startTime = GetTickCountMicrosecondsMN();
            //Will now attempt to update our samples for this batchNumber
            if (db_update(db,start,end,numberOfThreads,gradientSize,PAFSize))
            {
              //Simulate Getting I/O despite doing nothing with them
              //void * ptrIn  = db_get_in(db,start);
              //void * ptrOut = db_get_out(db,start);
              //fprintf(stderr,"Ptr In %p / Ptr Out %p\n",ptrIn,ptrOut);

            } else
            {
                  fprintf(stderr,"Failed while updating range %lu -> %lu\n",start,end);
                  abort();
            }
            endTime = GetTickCountMicrosecondsMN();

            thisMSec = (endTime-startTime)/1000;
            cumulTimeMSec += thisMSec;
            fprintf(stderr,"%u Threads | Samples %lu->%lu | Batch %lu/%lu | %u msec | Avg: %lu msec\n",numberOfThreads,start,end,batchNumber,totalBatches,thisMSec,(unsigned long) cumulTimeMSec/(batchNumber+1));
            db_print_readSpeed(db);
            if (db->threadStartLagMicroseconds>100)
             { fprintf(stderr,"Thread Lag Time to Start work %lu μsec / Time to prepare batch %lu μsec\n\n\n",db->threadStartLagMicroseconds,db->threadCompletionLagMicroseconds); } else
             { fprintf(stderr,"\n\n\n"); }
        }

        db_destroy(db);

    totalEndTime = GetTickCountMicrosecondsMN();
    fprintf(stderr,"Total : %lu μsec\n\n",totalEndTime-totalStartTime);
    fprintf(stderr,"Avg: %lu μsec per batch\n\n",(totalEndTime-totalStartTime)/(totalBatches+1));


    // Free the allocated memory
    freeStringArray(labelArray);
    labelArray = 0;
   }


  #if PROFILE_THREAD_ACTIVITY
   int i=system("python3 plotThreadLog.py");
   if (i!=0) {fprintf(stderr,"Failed plotting threading data, use python3 plotThreadLog.py manually..\n");}
  #endif // PROFILE_THREAD_ACTIVITY

  fprintf(stderr,"All done, concluding test                            \n");
  return 1;
}

