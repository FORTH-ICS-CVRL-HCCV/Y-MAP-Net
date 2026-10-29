#include <time.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <strings.h>  // strcasecmp, used by the multi-word vocabulary lookup

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
#include "PZPDLoader.h"

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
       dbl->archive                    = 0; //set by pzpdOpenSources() when the sources are .pzpd archives
       for (int i=0; i<4; i++) { dbl->archiveStream[i] = -1; }
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

       // SuperPoint datasets — freed unconditionally (runtime-gated, not USE_DINOV2_FEATURES).
       for (int i=0; i<dbList->numberOfSources; i++)
       {
          if (dbList->source[i].superpointData != NULL)
          {
            free_superpoint_dataset(dbList->source[i].superpointData);
            dbList->source[i].superpointData = NULL;
          }
       }


      free(dbl->source);
      dbl->source = 0;
   }
   pzpdCloseSources(dbl);
   free(dbl);
  }
  return;
}




//The next function is 100% ChatGPT generated..
//Depth sources: a .db source without an explicit all_* directory gets one DERIVED from its depth directory here
//(".../depth_val2017" -> ".../all_val2017"). If that directory holds <name>.pzp / .png for a sample, the sample is
//"multiplexed": its depth AND segmentation come from that combined file (PrepareBatch.c DEPTH STEP 0a), not from
//the depth_* / segment_* files, even though those exist. To force the separate files, pass an all_* path that
//does not exist.
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

// Geolocation is OFF unless Python/the test harness enables it before db_create.
// Kept as a module global (rather than a new db_create parameter) so the ctypes
// signature stays unchanged and existing behaviour is byte-identical when off.
static int g_addGeolocationHeatmap = 0;
void db_set_geolocation_config(int enable) { g_addGeolocationHeatmap = (enable!=0); }

// Channel layout of the combined depth + segmentation ("all") files, read at db_create time (a module global
// for the same reason as geolocation above). Default: 0 = label, 1 = depth high byte, 2 = depth low byte, as
// append16bitToSAM3.py / compressSegmentDepth.py write them (see processing/conversions.h).
static unsigned int g_combinedSegChannel       = COMBINED_SEG_CHANNEL;
static unsigned int g_combinedDepthHighChannel = COMBINED_DEPTH_HIGH_CHANNEL;
static unsigned int g_combinedDepthLowChannel  = COMBINED_DEPTH_LOW_CHANNEL;
int db_set_combined_layout(unsigned int segChannel,unsigned int depthHighChannel,unsigned int depthLowChannel)
{
  if ( (segChannel>3) || (depthHighChannel>3) || (depthLowChannel>3) || (depthHighChannel==depthLowChannel) )
  {
    fprintf(stderr,RED "db_set_combined_layout(%u,%u,%u): channels must be 0..3 and the depth bytes two different channels\n" NORMAL,segChannel,depthHighChannel,depthLowChannel);
    return 0;
  }
  g_combinedSegChannel       = segChannel;
  g_combinedDepthHighChannel = depthHighChannel;
  g_combinedDepthLowChannel  = depthLowChannel;
  return 1;
}

// Derive the geolocation sidecar dir from the images dir: "<parent>/geo_<basename>"
// — the SAME sibling naming datasets/geolocation/run_plonk_teacher.py writes to.
// e.g. ".../cache/coco/val2017" -> ".../cache/coco/geo_val2017". No config column.
void convertImagePathToGeoPath(const char *pathToCOCOImages,char *pathToCOCOGeolocation,size_t outSize)
{
    if (outSize==0) { return; }
    const char *slash = strrchr(pathToCOCOImages, '/');
    const char *base  = slash ? slash + 1 : pathToCOCOImages;
    size_t dirLen     = (size_t)(base - pathToCOCOImages);   // includes trailing '/', or 0
    int n = snprintf(pathToCOCOGeolocation, outSize, "%.*sgeo_%s", (int)dirLen, pathToCOCOImages, base);
    if (n < 0 || (size_t)n >= outSize) { pathToCOCOGeolocation[0] = '\0'; }
}



//----------------------------------------------------------------------------------------------------
int db_set_source_entry(struct DatabaseList* dbList,unsigned int sourceIDExt,const char * pathToDBFile,const char * pathToCOCOImages,const char * pathToCOCODepthMaps,const char * pathToCOCOSegmentations,const char * pathToCOCOAllDataCombined,int ignoreNoSkeletonSamples)
{
  if (!fileExists(pathToDBFile))
  {
      fprintf(stderr,RED "Cannot create source without DB file, %s does not exist\n",pathToDBFile);
      return 0;
  }

  //A .pzpd archive (recognised by its magic, not its extension) replaces the .db file and every
  //directory: the other paths are ignored. See PZPDLoader.c.
  if (pzpdIsArchive(pathToDBFile))
  {
    if ( (dbList==0) || (sourceIDExt>=dbList->numberOfSources) ) { return 0; }
    struct DatabaseEntry * e = &dbList->source[sourceIDExt];
    unsigned long samples = 0;
    unsigned short keypoints = 0;
    if (!pzpdProbeSource(pathToDBFile,&samples,&keypoints)) { return 0; }
    snprintf(e->pathToDBFile,MAX_PATH,"%s",pathToDBFile);
    e->isArchive               = 1;
    e->ignoreNoSkeletonSamples = (char) ignoreNoSkeletonSamples;
    e->numberOfSamples         = samples;
    fprintf(stderr,"dbl->source[%u].numberOfSamples = %lu (archive %s)\n",sourceIDExt,samples,pathToDBFile);
    dbList->numberOfKeypointsPerSample = keypoints;
    dbList->totalNumberOfSamples      += samples;
    return 1;
  }

  if (!directoryExists(pathToCOCOImages))
  {
      fprintf(stderr,RED "Cannot create source without Image files, Directory %s does not exist\n",pathToCOCOImages);
      return 0;
  }


  if (!directoryExists(pathToCOCOAllDataCombined))
  {
   fprintf(stderr,RED "Combined Directory `%s` does not exist\n",pathToCOCOAllDataCombined);
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
  }


  DatasetSourceID sourceID = sourceIDExt;
  if (dbList!=0)
  {
   struct DatabaseList * dbl = (struct DatabaseList*) dbList;
   if (sourceID<dbl->numberOfSources)
    {
     snprintf(dbl->source[sourceID].pathToDBFile,       MAX_PATH,"%s",pathToDBFile);
     snprintf(dbl->source[sourceID].pathToCOCOImages,   MAX_PATH,"%s",pathToCOCOImages);
     convertImagePathToGeoPath(dbl->source[sourceID].pathToCOCOImages,dbl->source[sourceID].pathToCOCOGeolocation,MAX_PATH);
     snprintf(dbl->source[sourceID].pathToCOCODepthMaps,MAX_PATH,"%s",pathToCOCODepthMaps);
     snprintf(dbl->source[sourceID].pathToCOCOSegmentations,MAX_PATH,"%s",pathToCOCOSegmentations);

     if (pathToCOCOAllDataCombined && pathToCOCOAllDataCombined[0] != '\0') {
       snprintf(dbl->source[sourceID].pathToCombinedMetaData,MAX_PATH,"%s",pathToCOCOAllDataCombined);
     } else {
       //It is very hard to remake all the .db files, so just patch in the new combined meta data..
       //(derived from the depth directory, see convertDepthPathToAllPath: an existing all_* file then wins over
       // the depth_* / segment_* files of the sample)
       snprintf(dbl->source[sourceID].pathToCombinedMetaData,MAX_PATH,"%s",pathToCOCODepthMaps);
       convertDepthPathToAllPath(dbl->source[sourceID].pathToCOCODepthMaps,dbl->source[sourceID].pathToCombinedMetaData,MAX_PATH);
     }

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
//[B1] PIPELINE INVARIANT: in-flight workers READ db->indices (through db_resolve_sample),
//so once the async pipeline exists, shuffling with an uncollected StartUpdate is a data
//race. Both shuffle entry points must first call db_pipeline_drain() (Collect + mark
//pipeline cold). Today this is safe only because every db_update is fully synchronous.
//The same invariant covers db->aug, gradientSize/PAFSize and the gradients collection.
//(DATALOADER.md B1 — "THE invariant")
void db_shuffle_indices(struct ImageDatabase * db)
{
  if (db!=0)
  {
      db_pipeline_drain(db); //[B1] enforce the invariant IN C: never permute indices with work in flight

      if (db->indices!=0)
      {
       checkCanary("Shuffle check A",db->canaryA.shouldRemainUntouched,CANARY_SIZE);
       checkCanary("Shuffle check B",db->canaryB.shouldRemainUntouched,CANARY_SIZE);
       checkCanary("Shuffle check C",db->canaryC.shouldRemainUntouched,CANARY_SIZE);

       unsigned long i,j;
       for (i=db->numberOfSamples-1; i>0; i--)
        {
          j = (unsigned long)(((uint64_t)rng_next32() * (uint64_t)(i + 1)) >> 32); // Fisher-Yates
          swapULong(&db->indices[i], &db->indices[j]); // Swap the elements at i and j
        }
       db->indicesVersion += 1; //the archive prefetch schedule follows the new order
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

    //NaN losses (e.g. a diverged Keras step) sort to the END: comparisons against NaN are
    //all false, which would otherwise make this comparator inconsistent — qsort with an
    //inconsistent comparator is undefined behaviour. (DATALOADER.md A6a)
    int aNaN = (lossA != lossA);
    int bNaN = (lossB != lossB);
    if (aNaN || bNaN) { return aNaN - bNaN; }

    if (lossA < lossB) return 1;
    if (lossA > lossB) return -1;
    return 0;
}
//----------------------------------------------------------------------------------------------------
void db_shuffle_indices_via_loss(struct ImageDatabase *db)
{
    fprintf(stderr,GREEN "experimental: using db_shuffle_indices_via_loss \n" NORMAL);

    db_pipeline_drain(db); //[B1] enforce the invariant IN C: never permute indices with work in flight

    if ( (db!=0) && (db->indices!=0) && (db->losses!=0) )
    {
        unsigned long i;

        // Step 1: Create an array of pairs (loss, index)
        struct lossIndexPair * pairs = (struct lossIndexPair *) malloc( (db->numberOfSamples+1) * sizeof(struct lossIndexPair));
        if (pairs != 0)
        {
         unsigned long shuffleStartTime = GetTickCountMicrosecondsMN();

         //losses/trainPasses are indexed by SAMPLE (db_update_sample_loss_range resolves positions), db->indices
         //maps position -> sample: pair each sample with its own loss (pairing losses[i] with indices[i] mixed
         //the two and made this an arbitrary permutation)
         float maxLoss = 0.0f;
         for (i = 0; i < db->numberOfSamples; i++)
         {
            if (db->trainPasses[i]>0)
            {
              float l = (float) db->losses[i] / db->trainPasses[i];
              if (l > maxLoss) { maxLoss = l; }
            }
         }
         for (i = 0; i < db->numberOfSamples; i++)
         {
            //never trained yet (e.g. dropped as the last partial batch): first in line, not stuck at the easy end
            pairs[i].loss  = (db->trainPasses[i]>0) ? (float) db->losses[i] / db->trainPasses[i] : maxLoss + 1.0f;
            pairs[i].index = i;

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

         // Step 2b: jitter each rank by up to +/-LOSS_SHUFFLE_RANK_JITTER places and sort again, so samples that
         // shared a batch (hence an identical loss) are spread over different batches (see configuration.h)
         for (i = 0; i < db->numberOfSamples; i++)
         {
            const float u = (float) rng_next32() * (1.0f / 4294967296.0f); // [0,1)
            pairs[i].loss = -((float) i + (2.0f * u - 1.0f) * (float) LOSS_SHUFFLE_RANK_JITTER); //descending sort = ascending rank
         }
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
         db->indicesVersion += 1; //the archive prefetch schedule follows the new order

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
        fprintf(stderr,RED "db_update_sample_loss: Unable to update loss for sample %lu, aborting to enforce consistency\n" NORMAL,sample);
        abort();
    }
}
//----------------------------------------------------------------------------------------------------
//This gets called from Dataloader.py->updateEpochResults which gets called from train2DPoseEstimation->DataAugmentation->on_epoch_end
//----------------------------------------------------------------------------------------------------
void db_update_sample_loss_range(struct ImageDatabase *db,unsigned long sampleStart,unsigned long sampleEnd,float loss)
{
    //fprintf(stderr,GREEN "db_update_sample_loss_range %lu -> %lu with loss %0.2f \n\n\n" NORMAL,sampleStart,sampleEnd,loss);

    if ( (db!=0) && (db->losses!=0) && (sampleStart<db->numberOfSamples) && (sampleStart<sampleEnd) && (sampleEnd>=db->numberOfSamples) )
    {
      //Detect special case where the batch size does not perfectly divide the number of samples (?)
      //Keras should probably take care of this however it doesn't
      //Without this the code aborts with message : Unable to update loss for sample 4960 -> 5000, aborting to enforce consistency
      if (db->numberOfSamples>0)
      {
         fprintf(stderr,YELLOW "db_update_sample_loss_range: Adjusting sampleEnd based on numberOfSamples=%lu\n" NORMAL,db->numberOfSamples);
         fprintf(stderr,YELLOW "Was sampleEnd=%lu\n" NORMAL,sampleEnd);
         //Clamp to numberOfSamples (not -1): the loop below is exclusive on sampleEnd,
         //so clamping to numberOfSamples-1 permanently skipped the last sample's loss
         //feedback. (DATALOADER.md A6b)
         sampleEnd=db->numberOfSamples;
         fprintf(stderr,YELLOW "Adjusted to sampleEnd=%lu\n" NORMAL,sampleEnd);
      }
    }

    if ( (db!=0) && (db->losses!=0) && (sampleStart<db->numberOfSamples) && (sampleEnd<=db->numberOfSamples) )
    {
        SampleNumber sampleID;
        for  (sampleID = sampleStart; sampleID<sampleEnd; sampleID++)
         {
             SampleNumber sID = 0;
             if (db_resolve_sample(db,sampleID,&sID))
             {
               db->losses[sID]      += loss;
               db->trainPasses[sID] += 1;

               //Keep loss float from ballooning — explicit reset for predictable decay period
               if (db->trainPasses[sID] > 10)
               {
                db->trainPasses[sID] = 5;
                db->losses[sID]     /= 2.0;
               }
             } else
             {
                 fprintf(stderr,RED "db_update_sample_loss_range: Incorrect SampleID resolution\n" NORMAL);
             }
         }
    } else
    {
        fprintf(stderr,RED "db_update_sample_loss_range: Unable to update loss for sample %lu -> %lu, aborting to enforce consistency\n" NORMAL,sampleStart,sampleEnd);
        fprintf(stderr,RED "sampleStart=%lu sampleEnd=%lu numberOfSamples=%lu\n" NORMAL,sampleStart,sampleEnd,db->numberOfSamples);
        abort();
    }
}
//----------------------------------------------------------------------------------------------------
int db_get_filename_of_sample(struct ImageDatabase *db,unsigned long sampleID,char* buffer, size_t buffer_size)
{
  SampleNumber sID = 0;
  DatasetSourceID sourceID = 0;
  if ( (db!=0) && db_resolve_sample_and_source(db,sampleID,&sID,&sourceID) )
    {
        const char * filenameString = db->pdb->sample[sID].imagePath;
        const char * pathString     = db->dbSources->source[sourceID].isArchive ? db->dbSources->source[sourceID].pathToDBFile : //<archive>/<key>
                                                                                   db->dbSources->source[sourceID].pathToCOCOImages;

        if (strlen(pathString) + 1 + strlen(filenameString) + 1 <= buffer_size)
        { // Check if buffer is large enough for full path + '/' + filename + '\0'
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
// Canonical sample resolution — SINGLE SOURCE OF TRUTH
//----------------------------------------------------------------------------------------------------
// Two index spaces exist in the DataLoader and mixing them up has already caused bugs
// (DATALOADER.md A2 — descriptors resolved their source from the wrong space):
//
//   position     : an external 0..db->numberOfSamples-1 index in the CURRENT (shuffled)
//                  ordering — what Python / batching / worker iterations use.
//   sampleNumber : the underlying PoseDatabase index (into db->pdb->sample[]), obtained
//                  through db->indices[position]. losses[] / trainPasses[] are also
//                  stored per sampleNumber.
//   sourceID     : which registered dataset source a sampleNumber belongs to
//                  (cumulative source sample counts — NOTE: defined over sampleNumber,
//                  NEVER over position).
//
// EVERY position->sampleNumber (and optional ->sourceID) lookup must go through these two
// calls so the bounds and corruption checks live in exactly one place.
//
// Return value: 1 on success; 0 (with a stderr message) for an out-of-range position or a
// missing index table — the caller decides whether that is fatal for it. ABORTS on index /
// source-table corruption (a resolved sampleNumber outside the PoseDatabase, or a source
// outside the source list): that is unrecoverable memory corruption, matching the
// pre-existing worker-thread behaviour.
//----------------------------------------------------------------------------------------------------
int db_resolve_sample(struct ImageDatabase *db, SampleNumber position, SampleNumber *outSampleNumber)
{
    if ( (db==0) || (outSampleNumber==0) )
    {
        fprintf(stderr,RED "db_resolve_sample: called without a database/output pointer\n" NORMAL);
        return 0;
    }
    if (db->indices==0)
    {
        fprintf(stderr,RED "db_resolve_sample: no index table allocated\n" NORMAL);
        return 0;
    }
    if (position >= db->numberOfSamples)
    {
        fprintf(stderr,RED "db_resolve_sample: position %lu out of range (%lu samples)\n" NORMAL,position,db->numberOfSamples);
        return 0;
    }

    SampleNumber sampleNumber = db->indices[position];
    if ( (db->pdb!=0) && (sampleNumber >= db->pdb->numberOfSamples) )
    {
        fprintf(stderr,RED "db_resolve_sample: position %lu resolved to sample %lu / %lu\n" NORMAL,position,sampleNumber,db->pdb->numberOfSamples);
        fprintf(stderr,RED "This is probably index corruption, aborting to protect training consistency\n" NORMAL);
        abort();
    }

    *outSampleNumber = sampleNumber;
    return 1;
}
//----------------------------------------------------------------------------------------------------
int db_resolve_sample_and_source(struct ImageDatabase *db, SampleNumber position, SampleNumber *outSampleNumber, DatasetSourceID *outSourceID)
{
    if (!db_resolve_sample(db,position,outSampleNumber))
        { return 0; }

    DatasetSourceID sourceID = db_resolve_sample_sourceID(db,*outSampleNumber);
    if ( (db->dbSources==0) || (sourceID >= db->dbSources->numberOfSources) )
    {
        fprintf(stderr,RED "db_resolve_sample_and_source: position %lu / sample %lu resolved to source %lu / %u\n" NORMAL,
                position,*outSampleNumber,sourceID,(db->dbSources!=0)?db->dbSources->numberOfSources:0);
        fprintf(stderr,RED "This is probably source-table corruption, aborting to protect training consistency\n" NORMAL);
        abort();
    }

    if (outSourceID!=0) { *outSourceID = sourceID; }
    return 1;
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
        //[B1] Double-buffer hook (DATALOADER.md B1.0): when `dataLoaderDoubleBuffer` is
        //enabled this block will ALSO allocate the shadow set (inShadow/out8bitShadow/
        //out16bitShadow — same dims, +~213 MB at batch 40) so db_swap_buffers() can
        //exchange fill/consume pixel pointers. Do this refactor together with the
        //YMAPNET2 channelsIn=6 change so the allocation plumbing is only reworked once.
        //STREAMING-ONLY GUARD: effective flag = (streamData && doubleBuffer). In full-load
        //mode numberOfImages == numberOfSamples, so a shadow set would double the WHOLE
        //dataset's buffer RAM for zero benefit (single fill at db_create, get_in_array /
        //get_out_array already serve stable zero-copy views) — force it off with a printout.
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
         db->in.bitsperpixel    = 8 * channelsIn; //not hardcoded 24: channelsIn is a runtime parameter (6 for the YMAPNET2 dual-frame input)
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
//[B1] These getters expose the (fill-side) buffers directly — which is why Python must
//.copy() before the next update overwrites them. The consumer-side siblings below
//(db_get_consumer_in/out/out16bit) return the shadow set after a swap; when doubleBuffer
//is off they fall back to these, so the Python side stays uniform. (DATALOADER.md B1/B2)
unsigned char * db_get_in(struct ImageDatabase *db,unsigned long startSample)
{
    if (db!=0)
      { return (unsigned char *) db->in.pixels + (startSample * db->in.width * db->in.height * db->in.channels); }
    return 0;
}
//----------------------------------------------------------------------------------------------------
//[B1] Double-buffer pipeline primitives (DATALOADER.md B1) --------------------------------
//----------------------------------------------------------------------------------------------------
//Wait out any in-flight StartUpdate so worker-read state (indices, aug, gradients, the
//fill buffers themselves) can be safely mutated. Called internally by the shuffles,
//db_set_augmentation_params, db_update (defensively) and db_destroy; Python calls it
//before discarding a stale prefetch.
int db_pipeline_drain(struct ImageDatabase * db)
{
    if (db==0) { return 0; }
    if (db->computationsRunning)
    {
        //Wait-only: the in-flight work was released by StartUpdate's Kick (using the
        //combined wait here would reset activeWorkers mid-run and deadlock/abort).
        threadpoolMainThreadWaitForKickedWorkersToFinishTimeoutSeconds(&db->threadPool,MAXIMUM_WORKER_THREAD_TIMEOUT_SECONDS);
        db->computationsRunning = 0;
    }
    return 1;
}
//----------------------------------------------------------------------------------------------------
static void swapHeatmapPixels(struct Heatmaps * a, struct Heatmaps * b)
{
    void * p       = a->pixels;      a->pixels      = b->pixels;      b->pixels      = p;
    void * l       = a->pixelsLimit; a->pixelsLimit = b->pixelsLimit; b->pixelsLimit = l;
}
//----------------------------------------------------------------------------------------------------
//Exchange the fill/consume pixel pointers so the just-filled batch becomes readable through
//db_get_consumer_* while workers overwrite the other set. ONLY legal while the pipeline is
//idle (between CollectUpdate and the next StartUpdate). No-op when doubleBuffer is off.
int db_swap_buffers(struct ImageDatabase * db)
{
    if (db==0)             { return 0; }
    if (!db->doubleBuffer) { return 0; }
    if (db->computationsRunning)
    {
        fprintf(stderr,RED "db_swap_buffers refused: computations are running (collect first)\n" NORMAL);
        return 0;
    }
    swapHeatmapPixels(&db->in,      &db->inShadow);
    swapHeatmapPixels(&db->out8bit, &db->out8bitShadow);
    if ( (db->out16bit.pixels!=0) && (db->out16bitShadow.pixels!=0) )
       { swapHeatmapPixels(&db->out16bit, &db->out16bitShadow); }
    return 1;
}
//----------------------------------------------------------------------------------------------------
unsigned char * db_get_consumer_in(struct ImageDatabase *db,unsigned long startSample)
{
    if (db==0) { return 0; }
    if (db->doubleBuffer)
      { return (unsigned char *) db->inShadow.pixels + (startSample * db->inShadow.width * db->inShadow.height * db->inShadow.channels); }
    return db_get_in(db,startSample);
}
//----------------------------------------------------------------------------------------------------
char * db_get_consumer_out(struct ImageDatabase *db,unsigned long startSample)
{
    if (db==0) { return 0; }
    if (db->doubleBuffer)
      { return (char *) db->out8bitShadow.pixels + (startSample * db->out8bitShadow.width * db->out8bitShadow.height * db->out8bitShadow.channels); }
    return db_get_out(db,startSample);
}
//----------------------------------------------------------------------------------------------------
signed short * db_get_consumer_out16bit(struct ImageDatabase *db,unsigned long startSample)
{
    if (db==0) { return 0; }
    if (db->doubleBuffer)
      { return (signed short *) db->out16bitShadow.pixels + (startSample * db->out16bitShadow.width * db->out16bitShadow.height * db->out16bitShadow.channels); }
    return db_get_out16bit(db,startSample);
}
//----------------------------------------------------------------------------------------------------
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
    SampleNumber sID = 0;
    if ( (db!=0) && db_resolve_sample(db,sampleID,&sID) )
    {
        return db->pdb->sample[sID].numberOfSkeletons;
    }
    return 0;
}
//----------------------------------------------------------------------------------------------------
//endSample parameter removed (DATALOADER.md A6f): it was accepted and ignored — a pointer
//return cannot honour a range. Python argtypes updated in lockstep (DataLoader.py).
signed short * db_get_out16bit(struct ImageDatabase *db,unsigned long startSample)
{
    if (db!=0)
      {
        return (signed short *) db->out16bit.pixels + (startSample * db->out16bit.width * db->out16bit.height * db->out16bit.channels);
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
// 8-bit heatmap channel layout — SINGLE SOURCE OF TRUTH
//---------------------------------------------------------------------
// db_build_heatmap_layout declares every output modality (in channel order) with
// its include flag, then resolves the per-modality start indices by prefix-sum.
// The legacy db->...HeatmapIndex fields are DERIVED from the resolved table, so
// there is exactly one place that decides "which modality occupies which channels".
//
// Adding a new modality = one extra heatmapLayoutAddGroup() line in the right
// position here, kept in sync with calculateNumberOf8BitHeatmaps() (which must sum
// to the same total). The Python side reads the resolved ranges back over ctypes
// (see db_get_channel_range / db_get_channel_group_*), so it never re-hardcodes
// indices either.
void db_build_heatmap_layout(struct ImageDatabase *db, unsigned int keypoints)
{
  if (db==0) { return; }
  struct HeatmapLayout * L = &db->heatmapLayout;
  heatmapLayoutClear(L);

  //Canonical 8-bit heatmap channel order. Enable flags come from db (runtime) and
  //the ENABLE_*/VALID_SEGMENTATIONS compile-time macros (configuration.h).
  heatmapLayoutAddGroup(L,"keypoints",    keypoints,                                       1);
  heatmapLayoutAddGroup(L,"paf",          (unsigned int)numberOfPAFJoints,                 db->addPAFHeatmap);
  heatmapLayoutAddGroup(L,"depth",        1,                                               db->addDepthHeatmap);
  heatmapLayoutAddGroup(L,"normals",      3,                                               db->addNormalHeatmaps);
  heatmapLayoutAddGroup(L,"depth_levels", (unsigned int)db->addDepthLevelsHeatmaps,        db->addDepthLevelsHeatmaps);
  heatmapLayoutAddGroup(L,"denoise",      3,                                               ENABLE_DENOISING_OUTPUT);
  heatmapLayoutAddGroup(L,"left_right",   2,                                               ENABLE_LEFT_RIGHT_JOINT_DISAMBIGUATION_OUTPUT);
  heatmapLayoutAddGroup(L,"segmentation", (unsigned int)VALID_SEGMENTATIONS,               db->addSegmentationHeatmaps);
  heatmapLayoutAddGroup(L,"instance",     (unsigned int)INSTANCE_CHANNELS,                 db->addInstanceHeatmaps);
  heatmapLayoutAddGroup(L,"superpoint",   (unsigned int)db->superpointChannels,            db->addSuperpointHeatmaps);
  heatmapLayoutAddGroup(L,"geolocation",  1,                                               db->addGeolocationHeatmap);

  heatmapLayoutResolve(L);
  heatmapLayoutPrint(L);

  //Derive the legacy index fields from the resolved layout (single source of truth).
  //heatmapLayoutGetStart returns the start even for a disabled modality (where it
  //would begin), preserving the previous unconditional-assignment behaviour.
  int st;
  st = heatmapLayoutGetStart(L,"paf");          if (st>=0) { db->PAFHeatmapIndex          = (Heatmap8BitIndex)st; }
  st = heatmapLayoutGetStart(L,"depth");        if (st>=0) { db->depthmapHeatmapIndex8Bit = (Heatmap8BitIndex)st;
                                                            db->backgroundHeatmapIndex   = (Heatmap8BitIndex)st; }
  st = heatmapLayoutGetStart(L,"normals");      if (st>=0) { db->normalsHeatmapIndex      = (Heatmap8BitIndex)st; }
  st = heatmapLayoutGetStart(L,"depth_levels"); if (st>=0) { db->depthLevelsHeatmapIndex  = (Heatmap8BitIndex)st; }
  st = heatmapLayoutGetStart(L,"denoise");      if (st>=0) { db->denoiseHeatmapIndex      = (Heatmap8BitIndex)st; }
  st = heatmapLayoutGetStart(L,"left_right");   if (st>=0) { db->leftRightHeatmapIndex    = (Heatmap8BitIndex)st; }
  st = heatmapLayoutGetStart(L,"segmentation"); if (st>=0) { db->segmentationHeatmapIndex = (Heatmap8BitIndex)st; }
  st = heatmapLayoutGetStart(L,"instance");     if (st>=0) { db->instanceHeatmapIndex     = (Heatmap8BitIndex)st; }
  st = heatmapLayoutGetStart(L,"superpoint");   if (st>=0) { db->superpointHeatmapIndex   = (Heatmap8BitIndex)st; }
  st = heatmapLayoutGetStart(L,"geolocation");  if (st>=0) { db->geolocationHeatmapIndex  = (Heatmap8BitIndex)st; }
}

//---------------------------------------------------------------------
// Channel-layout getters exposed to the Python DataLoader.py over ctypes.
//---------------------------------------------------------------------
// Name-based lookup: fills *outStart/*outCount for an ENABLED modality and
// returns 1, or returns 0 if the modality is absent/disabled.
int db_get_channel_range(struct ImageDatabase *db, const char *name, unsigned int *outStart, unsigned int *outCount)
{
  if (db==0) { return 0; }
  return heatmapLayoutGetRange(&db->heatmapLayout,name,outStart,outCount);
}

// Index-based enumeration so Python can discover the whole catalogue without
// hardcoding modality names.
int db_get_number_of_channel_groups(struct ImageDatabase *db)
{
  return (db!=0) ? (int)db->heatmapLayout.numberOfGroups : 0;
}

const char * db_get_channel_group_name(struct ImageDatabase *db, int index)
{
  if ( (db==0) || (index<0) || (index>=(int)db->heatmapLayout.numberOfGroups) ) { return ""; }
  const char * n = db->heatmapLayout.group[index].name;
  return (n!=0) ? n : "";
}

int db_get_channel_group_start(struct ImageDatabase *db, int index)
{
  if ( (db==0) || (index<0) || (index>=(int)db->heatmapLayout.numberOfGroups) ) { return -1; }
  return (int)db->heatmapLayout.group[index].start;
}

int db_get_channel_group_count(struct ImageDatabase *db, int index)
{
  if ( (db==0) || (index<0) || (index>=(int)db->heatmapLayout.numberOfGroups) ) { return -1; }
  return (int)db->heatmapLayout.group[index].count;
}

int db_get_channel_group_enabled(struct ImageDatabase *db, int index)
{
  if ( (db==0) || (index<0) || (index>=(int)db->heatmapLayout.numberOfGroups) ) { return 0; }
  return db->heatmapLayout.group[index].enabled;
}

//---------------------------------------------------------------------
// Joint-name getters (DATALOADER.md C5): the .db already carries every joint's name
// (struct Joint.name, DBLoader.h) — expose it so Python's get_labels() reads the loaded
// database instead of hardcoding the COCO-17 list. Same idiom as the channel-group getters.
//---------------------------------------------------------------------
int db_get_number_of_joints(struct ImageDatabase *db)
{
  if ( (db!=0) && (db->pdb!=0) ) { return (int) db->pdb->keypointsForEachSample; }
  return 0;
}

const char * db_get_joint_name(struct ImageDatabase *db, int jID)
{
  if ( (db!=0) && (db->pdb!=0) && (db->pdb->joint!=0) && (jID>=0) && (jID < (int)db->pdb->keypointsForEachSample) )
  {
    return db->pdb->joint[jID].name;
  }
  return "";
}

void db_get_augmentation_params(struct ImageDatabase *db, struct DataAugmentation *out)
{
  if (db && out) { *out = db->aug; }
}

void db_set_augmentation_params(struct ImageDatabase *db, const struct DataAugmentation *params)
{
  if (db && params)
  {
    db_pipeline_drain(db); //[B1] workers read db->aug — never mutate it with work in flight
    db->aug = *params;
  }
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
    SampleNumber sID = 0;
    if ( (db!=0) && db_resolve_sample(db,sampleID,&sID) )
    {
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
    return 0;
}
//----------------------------------------------------------------------------------------------------
float db_get_sample_total_loss(struct ImageDatabase *db, unsigned long sampleID)
{
    SampleNumber sID = 0;
    if ( (db!=0) && db_resolve_sample(db,sampleID,&sID) )
    {
        return db->losses[sID];
    }
    return 0;
}
//----------------------------------------------------------------------------------------------------
unsigned long db_get_sample_train_passes(struct ImageDatabase *db, unsigned long sampleID)
{
    SampleNumber sID = 0;
    if ( (db!=0) && db_resolve_sample(db,sampleID,&sID) )
    {
        return db->trainPasses[sID];
    }
    return 0;
}
//----------------------------------------------------------------------------------------------------
unsigned short * db_get_sample_description_tokens(struct ImageDatabase *db, unsigned long sampleID)
{
    SampleNumber sID = 0;
    if ( (db!=0) && db_resolve_sample(db,sampleID,&sID) )
    {
         if (db->pdb->sample[sID].eraseEntry)
            { return emptyDescriptionTokens; }

         return db->pdb->sample[sID].descriptionTokens;
    }
    return 0;
}
//----------------------------------------------------------------------------------------------------
// Floats per descriptor of a source: its .dinov3 file, or its archive's descriptor_dinov3 table; 0 if none
static unsigned int db_source_descriptor_length(struct DatabaseList *dbl, unsigned int i)
{
    if (dbl->source[i].isArchive) { return dbl->source[i].descriptorLength; }
#if USE_DINOV2_FEATURES
    if (dbl->source[i].descriptorsAsADataset != 0) { return (unsigned int) dbl->source[i].descriptorsAsADataset->value_count; }
#endif
    return 0;
}
//----------------------------------------------------------------------------------------------------
int db_get_descriptor_elements_number(struct ImageDatabase *db)
{
#if USE_DINOV2_FEATURES
    if (db != 0 && db->dbSources != 0)
    {
        for (unsigned int i = 0; i < db->dbSources->numberOfSources; i++)
        {
            unsigned int length = db_source_descriptor_length(db->dbSources,i);
            if (length != 0) { return (int) length; }
        }
    }
#endif
    return 0;
}
//----------------------------------------------------------------------------------------------------
float * db_get_sample_descriptors(struct ImageDatabase *db, unsigned long sampleID)
{
    if (db==0)
    {
        fprintf(stderr,"db_get_sample_descriptors(%lu): db is NULL\n",sampleID);
        exit(1);
    }

    //Canonical resolution (position -> sampleNumber -> source) — see db_resolve_sample_and_source.
    //This is also the A2 fix: the source MUST derive from the resolved sampleNumber, not the position.
    SampleNumber sID = 0;
    DatasetSourceID sourceID = 0;
    if (!db_resolve_sample_and_source(db,sampleID,&sID,&sourceID))
    {
        fprintf(stderr,"db_get_sample_descriptors: sample ID %lu could not be resolved (max %lu)\n",
                sampleID, db->numberOfSamples);
        exit(1);
    }

    // No .dinov2 file for this source — not an inconsistency, return empty vector
    if (db_source_descriptor_length(db->dbSources,sourceID)==0)
    {
        return emptyDescriptor;
    }
    if (db->pdb->sample[sID].descriptor==0)
    {
        fprintf(stderr,"db_get_sample_descriptors: descriptor for sample %lu not allocated\n",sID);
        exit(1);
    }

    return db->pdb->sample[sID].descriptor;
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
// The offset/scaling pair the loaded embeddings are currently expressed in. Defaults to the
// GloVe_D*.embeddings header; db_set_embeddings_offset_scaling overrides it at runtime so the
// tanh token targets can be rescaled without regenerating the file.
float db_get_embeddings_offset(struct ImageDatabase *db)
{
    if (db!=0) { return db->pdb->embeddings.offset; }
    return 0.0f;
}
//----------------------------------------------------------------------------------------------------
float db_get_embeddings_scaling(struct ImageDatabase *db)
{
    if (db!=0) { return db->pdb->embeddings.scaling; }
    return 0.0f;
}
//----------------------------------------------------------------------------------------------------
int db_set_embeddings_offset_scaling(struct ImageDatabase *db, float offset, float scaling)
{
    if (db==0) { return 0; }
    return rescale_embeddings(&db->pdb->embeddings, offset, scaling);
}
//----------------------------------------------------------------------------------------------------
void db_get_batch_descriptors(struct ImageDatabase *db,
                               unsigned long start,
                               unsigned long end,
                               float *out_buffer)
{
#if USE_DINOV2_FEATURES
    unsigned long n = end - start;

    // Read the actual descriptor dimension from the first loaded dataset.
    unsigned long dim = 0;
    if (db != 0 && db->dbSources != 0)
    {
        for (unsigned int i = 0; i < db->dbSources->numberOfSources; i++)
        {
            dim = (unsigned long) db_source_descriptor_length(db->dbSources,i);
            if (dim != 0) { break; }
        }
    }
    if (dim == 0) { return; }

    for (unsigned long i = 0; i < n; i++)
    {
        float *dst = out_buffer + i * dim;
        SampleNumber sID = 0;
        float *src = NULL;
        if (db_resolve_sample(db,start + i,&sID))
            { src = db->pdb->sample[sID].descriptor; }

        if (src != NULL)
            memcpy(dst, src, dim * sizeof(float));
        else
            memset(dst, 0, dim * sizeof(float));
    }
#endif
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
// Batch clone of db_get_sample_description_tokens (DATALOADER.md B3): fills a caller-allocated
// (end-start) x MAX_DESCRIPTION_TOKENS unsigned short buffer in ONE call instead of one ctypes
// call per sample. Respects the shuffle order (per-sample getter resolves through db->indices).
void db_get_batch_tokens(struct ImageDatabase *db,
                         unsigned long start,
                         unsigned long end,
                         unsigned short *out_buffer)
{
    unsigned long n = end - start;
    for (unsigned long i = 0; i < n; i++)
    {
        unsigned short *dst = out_buffer + i * MAX_DESCRIPTION_TOKENS;
        unsigned short *src = db_get_sample_description_tokens(db, start + i);
        if (src != 0)
            { memcpy(dst, src, MAX_DESCRIPTION_TOKENS * sizeof(unsigned short)); }
        else
            { memset(dst, 0, MAX_DESCRIPTION_TOKENS * sizeof(unsigned short)); }
    }
}
//----------------------------------------------------------------------------------------------------
// Batch clone of db_get_sample_description_embeddings (DATALOADER.md B3): fills a caller-allocated
// (end-start) x MAX_DESCRIPTION_TOKENS x D float buffer in ONE call. D is supplied by the caller
// (the width it allocated for), matching the pre-existing per-token copy width. Tokens past a
// sample's numberOfTokens are left zeroed, mirroring the old np.zeros-then-fill loop.
void db_get_batch_embeddings(struct ImageDatabase *db,
                             unsigned long start,
                             unsigned long end,
                             unsigned long D,
                             float *out_buffer)
{
    unsigned long n = end - start;
    if (D == 0) { return; }

    memset(out_buffer, 0, n * MAX_DESCRIPTION_TOKENS * D * sizeof(float));

    // src vectors are only embeddings.D floats long. Copy the narrower of the two widths
    // so a caller that allocated for a different D cannot over-read the source vectors;
    // any surplus destination columns stay zeroed by the memset above.
    unsigned long realD = (db != 0) ? (unsigned long) db->pdb->embeddings.D : 0;
    unsigned long copyD = (realD < D) ? realD : D;
    if (copyD == 0) { return; }

    for (unsigned long i = 0; i < n; i++)
    {
        unsigned long sampleID = start + i;
        int numTokens = db_get_sample_description_tokens_number(db, sampleID);
        for (int t = 0; t < numTokens; t++)
        {
            float *src = db_get_sample_description_embeddings(db, sampleID, t);
            if (src != 0)
            {
                float *dst = out_buffer + (i * MAX_DESCRIPTION_TOKENS + t) * D;
                memcpy(dst, src, copyD * sizeof(float));
            }
        }
    }
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
          // Allocate the blacklist struct on first use
          if (db->pdb->tokenBlackList==0)
          {
              db->pdb->tokenBlackList = (struct DescriptionTokenBlacklist *) malloc(sizeof(struct DescriptionTokenBlacklist));
              if (db->pdb->tokenBlackList!=0)
              {
                  memset(db->pdb->tokenBlackList,0,sizeof(struct DescriptionTokenBlacklist));
                  db->pdb->tokenBlackList->blackListSize = 0;
                  db->pdb->tokenBlackList->blackListCurrentTokens = 0;
              }
          }

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
          if ( (db->pdb->tokenBlackList!=0) && (db->pdb->tokenBlackList->blackListedTokens!=0) )
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
//----------------------------------------------------------------------------------------------------
int db_allocate_token_synonym_map(struct ImageDatabase *db, unsigned int numberOfPairs)
{
  if (db!=0)
  {
    if (db->pdb!=0)
    {
      if (db->pdb->tokenSynonymMap==0)
      {
        db->pdb->tokenSynonymMap = (struct TokenSynonymMap *) malloc(sizeof(struct TokenSynonymMap));
        if (db->pdb->tokenSynonymMap!=0)
        {
          memset(db->pdb->tokenSynonymMap,0,sizeof(struct TokenSynonymMap));
        }
      }

      if (db->pdb->tokenSynonymMap!=0)
      {
        if (db->pdb->tokenSynonymMap->fromTokens!=0)
        {
          free(db->pdb->tokenSynonymMap->fromTokens);
          db->pdb->tokenSynonymMap->fromTokens = 0;
        }
        if (db->pdb->tokenSynonymMap->toTokens!=0)
        {
          free(db->pdb->tokenSynonymMap->toTokens);
          db->pdb->tokenSynonymMap->toTokens = 0;
        }

        db->pdb->tokenSynonymMap->fromTokens = (unsigned short *) malloc(numberOfPairs * sizeof(unsigned short));
        db->pdb->tokenSynonymMap->toTokens   = (unsigned short *) malloc(numberOfPairs * sizeof(unsigned short));

        if (db->pdb->tokenSynonymMap->fromTokens!=0 && db->pdb->tokenSynonymMap->toTokens!=0)
        {
          memset(db->pdb->tokenSynonymMap->fromTokens,0,numberOfPairs * sizeof(unsigned short));
          memset(db->pdb->tokenSynonymMap->toTokens,  0,numberOfPairs * sizeof(unsigned short));
          db->pdb->tokenSynonymMap->pairCapacity = numberOfPairs;
          db->pdb->tokenSynonymMap->pairCount    = 0;
          return 1;
        }
      }
    }
  }
  return 0;
}
//----------------------------------------------------------------------------------------------------
int db_add_token_synonym_pair(struct ImageDatabase *db, unsigned short fromToken, unsigned short toToken)
{
  if (db!=0)
  {
    if (db->pdb!=0)
    {
      struct TokenSynonymMap *sm = db->pdb->tokenSynonymMap;
      if (sm!=0 && sm->fromTokens!=0 && sm->toTokens!=0)
      {
        if (sm->pairCount < sm->pairCapacity)
        {
          sm->fromTokens[sm->pairCount] = fromToken;
          sm->toTokens[sm->pairCount]   = toToken;
          sm->pairCount += 1;
          return 1;
        }
      }
    }
  }
  return 0;
}
//----------------------------------------------------------------------------------------------------
int db_compile_token_synonym_map(struct ImageDatabase *db)
{
  if (db!=0)
  {
    if (db->pdb!=0)
    {
      struct TokenSynonymMap *sm = db->pdb->tokenSynonymMap;
      if (sm!=0 && sm->fromTokens!=0 && sm->toTokens!=0 && sm->pairCount>0)
      {
        for (unsigned long sampleNumber=0; sampleNumber<db->numberOfSamples; sampleNumber++)
        {
          SampleNumber sID = sampleNumber;
          unsigned short *tokens         = db->pdb->sample[sID].descriptionTokens;
          int numberOfTokensForThisSample = db->pdb->sample[sID].numberOfTokens;

          if (tokens!=0 && numberOfTokensForThisSample>0)
          {
            for (int tID=0; tID<numberOfTokensForThisSample; tID++)
            {
              if (tokens[tID]!=0)
              {
                for (unsigned int synID=0; synID<sm->pairCount; synID++)
                {
                  if (tokens[tID] == sm->fromTokens[synID])
                  {
                    tokens[tID] = sm->toTokens[synID];
                    break; // only first matching synonym per token
                  }
                }
              }
            }
          }
        }
        return 1;
      }
    }
  }
  return 0;
}
//----------------------------------------------------------------------------------------------------
// Some vocabulary entries are whole captions stored as ONE token ("A large number of seagulls
// gathered on the beach", id 17965) — scrape artifacts from a caption source that was never
// tokenized. Their GloVe vector is a mean over every word including the stopwords, so it lands
// nowhere useful, and the token is far too rare to ever be learned. Expanding such an entry into
// its component word tokens recovers real supervision ("large number seagulls gathered beach").
//
// ORDER MATTERS: this has to run BEFORE db_compile_added_token_blacklist(), otherwise the
// stopwords it produces ('the', 'of', 'on', 'a') stay in and eat caption slots.
#define MAX_MULTIWORD_PARTS 16

struct MultiwordExpansion
{
  unsigned short fromToken;
  unsigned char  partCount;
  unsigned short parts[MAX_MULTIWORD_PARTS];
};

// Case-insensitive on purpose: the multi-word entries are capitalised ("East Building", "The
// background is the exhibition tent") while the rest of the vocabulary is lowercase, so an exact
// strcmp resolves almost nothing.
static int _vocabulary_lookup(struct StringArray *vocab, const char *word)
{
  for (int i=0; i<vocab->count; i++)
  {
    if (vocab->strings[i]!=0)
    {
      if (strcasecmp(vocab->strings[i],word)==0) { return i; }
    }
  }
  return -1;
}

int db_expand_multiword_tokens(struct ImageDatabase *db, const char *vocabularyPath)
{
  if (db==0)             { return 0; }
  if (db->pdb==0)        { return 0; }
  if (vocabularyPath==0) { return 0; }

  struct StringArray * vocab = loadStringArrayFromFile(vocabularyPath);
  if (vocab==0)
  {
    fprintf(stderr,RED "db_expand_multiword_tokens: could not open vocabulary %s\n" NORMAL,vocabularyPath);
    return 0;
  }

  // The parser keys entries by order of appearance, so index==tokenID only holds if the file
  // covers every token. Bail out instead of rewriting tokens against a shifted vocabulary.
  if (vocab->count < db->pdb->maxTokenValue)
  {
    fprintf(stderr,RED "db_expand_multiword_tokens: %s holds %u entries but tokens go up to %u — refusing to guess\n" NORMAL,
            vocabularyPath,vocab->count,db->pdb->maxTokenValue);
    freeStringArray(vocab);
    return 0;
  }

  unsigned int multiwordEntries = 0;
  for (int i=0; i<vocab->count; i++)
  {
    if ( (vocab->strings[i]!=0) && (strchr(vocab->strings[i],' ')!=0) ) { multiwordEntries+=1; }
  }

  if (multiwordEntries==0)
  {
    fprintf(stderr,"db_expand_multiword_tokens: no multi-word entries in %s, nothing to do\n",vocabularyPath);
    freeStringArray(vocab);
    return 1;
  }

  struct MultiwordExpansion * expansion = (struct MultiwordExpansion *) malloc(multiwordEntries * sizeof(struct MultiwordExpansion));
  if (expansion==0) { freeStringArray(vocab); return 0; }

  //Resolve each multi-word entry into the token IDs of its component words
  //---------------------------------------------------------------------
  unsigned int expansions = 0;
  unsigned int unresolvedWords = 0;
  for (int i=0; i<vocab->count; i++)
  {
    if (vocab->strings[i]==0)                 { continue; }
    if (strchr(vocab->strings[i],' ')==0)     { continue; }

    char work[MAX_STRING_LENGTH+1] = {0};
    snprintf(work,sizeof(work),"%s",vocab->strings[i]);

    struct MultiwordExpansion e;
    e.fromToken = (unsigned short) i;
    e.partCount = 0;

    char * savePtr = 0;
    for (char * word = strtok_r(work," ",&savePtr); word!=0; word = strtok_r(0," ",&savePtr))
    {
      int partID = _vocabulary_lookup(vocab,word);
      if (partID<0)  { unresolvedWords+=1; continue; }  //e.g. 'off-road', 'Revealed' — no such token
      if (partID==i) { continue; }                      //never expand a token into itself

      //De-duplicated here rather than per sample ("The background is the exhibition tent" repeats
      //'the'), so a repeated word never costs one of the MAX_DESCRIPTION_TOKENS slots.
      int alreadyPresent = 0;
      for (int p=0; p<e.partCount; p++)
      {
        if (e.parts[p]==(unsigned short) partID) { alreadyPresent=1; break; }
      }
      if (alreadyPresent)                    { continue; }
      if (e.partCount>=MAX_MULTIWORD_PARTS)  { break; }

      e.parts[e.partCount] = (unsigned short) partID;
      e.partCount += 1;
    }

    //An entry where nothing resolved stays untouched — expanding it to nothing would silently
    //delete what may be the only label a sample has.
    if (e.partCount>0) { expansion[expansions] = e; expansions+=1; }
  }

  //Rewrite every sample's token list through the table
  //---------------------------------------------------------------------
  unsigned long samplesChanged = 0, tokensExpanded = 0, samplesTruncated = 0;
  for (unsigned long sampleNumber=0; sampleNumber<db->numberOfSamples; sampleNumber++)
  {
    //This function does not acknowledge shuffle but data has better memory locality
    SampleNumber sID = sampleNumber;

    unsigned short * tokens         = db->pdb->sample[sID].descriptionTokens;
    int numberOfTokensForThisSample = db->pdb->sample[sID].numberOfTokens;

    if ( (tokens==0) || (numberOfTokensForThisSample<=0) ) { continue; }

    unsigned short rewritten[MAX_DESCRIPTION_TOKENS];
    int outputTokens = 0;
    int sampleChanged = 0;
    int lostTokens = 0;

    for (int tID=0; tID<numberOfTokensForThisSample; tID++)
    {
      struct MultiwordExpansion * hit = 0;
      if (tokens[tID]!=0)
      {
        for (unsigned int eID=0; eID<expansions; eID++)
        {
          if (tokens[tID]==expansion[eID].fromToken) { hit = &expansion[eID]; break; }
        }
      }

      if (hit==0)
      {
        if (outputTokens<MAX_DESCRIPTION_TOKENS) { rewritten[outputTokens] = tokens[tID]; outputTokens+=1; }
        else                                     { lostTokens+=1; }
      }
      else
      {
        sampleChanged = 1;
        tokensExpanded += 1;
        for (int p=0; p<hit->partCount; p++)
        {
          if (outputTokens<MAX_DESCRIPTION_TOKENS) { rewritten[outputTokens] = hit->parts[p]; outputTokens+=1; }
          else                                     { lostTokens+=1; }
        }
      }
    }

    if (sampleChanged)
    {
      memcpy(tokens,rewritten,outputTokens * sizeof(unsigned short));
      //An expansion normally grows the list, but a partly unresolved entry can shrink it
      for (int tID=outputTokens; tID<numberOfTokensForThisSample; tID++) { tokens[tID] = 0; }
      db->pdb->sample[sID].numberOfTokens = (unsigned char) outputTokens;

      samplesChanged += 1;
      if (lostTokens>0) { samplesTruncated += 1; }
    }
  }

  fprintf(stderr,GREEN "db_expand_multiword_tokens: expanded %u/%u multi-word vocabulary entries, rewrote %lu token occurrences in %lu samples\n" NORMAL,
          expansions,multiwordEntries,tokensExpanded,samplesChanged);
  if (unresolvedWords>0)
  {
    fprintf(stderr,"  %u component word(s) are not in the vocabulary and were dropped\n",unresolvedWords);
  }
  if (samplesTruncated>0)
  {
    fprintf(stderr,YELLOW "  %lu sample(s) hit the %u token ceiling and lost trailing tokens\n" NORMAL,samplesTruncated,MAX_DESCRIPTION_TOKENS);
  }

  free(expansion);
  freeStringArray(vocab);
  return 1;
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
// Companion to db_get_sample_description(): inject caption-derived token IDs for a sample whose
// source did not ship pre-tokenized description lines (.pzpd archives -- PZPDLoader.c sets
// numberOfTokens=0 and stores the caption text only, see convertToPZPD.py). DataLoader.py tokenizes
// the caption in Python with the same word-splitting + vocabulary lookup datasets/useVocabulary.py
// uses to build .db token lines, then pushes the resulting ids here so .pzpd and .db sources train
// on identical token semantics -- everything downstream (db_expand_multiword_tokens,
// db_count_description_tokens, db_get_batch_tokens, ...) just reads descriptionTokens either way.
int db_set_sample_description_tokens(struct ImageDatabase *db, unsigned long sampleID, unsigned short *tokenIDs, int count)
{
    SampleNumber sID = 0;
    if ( (db!=0) && (tokenIDs!=0) && db_resolve_sample(db,sampleID,&sID) )
    {
        if (count > MAX_DESCRIPTION_TOKENS) { count = MAX_DESCRIPTION_TOKENS; }
        if (count < 0)                      { count = 0; }

        memset(db->pdb->sample[sID].descriptionTokens, 0, MAX_DESCRIPTION_TOKENS * sizeof(unsigned short));
        for (int i=0; i<count; i++)
        {
            unsigned short thisToken = tokenIDs[i];
            db->pdb->sample[sID].descriptionTokens[i] = thisToken;
            if ( (!db->pdb->maxTokenValueSetExternally) && (thisToken > db->pdb->maxTokenValue) )
                { db->pdb->maxTokenValue = thisToken; }
        }
        db->pdb->sample[sID].numberOfTokens = (unsigned char) count;
        return 1;
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
    unsigned char * seen = (unsigned char *) calloc(65535, sizeof(unsigned char)); // USHRT_MAX
    if (seen == NULL) { return; }

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

    free(seen);
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
                                if (erasedID<numberOfTokens)
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
    picToMap->bitsperpixel = (unsigned int) (8 * db->in.channels); //match db->in, not hardcoded 24 (DATALOADER.md A6c)
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
      //The pipeline is idle here: (re)submit the archive prefetch schedule for this batch
      pzpdPrefetchBatch(db,sampleStart,sampleEnd);

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
      // threadpoolCreate failed — free the context we allocated above so we do
      // not leak it.  db->numberOfThreads is also reset so a future retry of
      // db_start_threads starts from a clean state.
      fprintf(stderr,RED "db_start_threads failed threadpoolCreate\n" NORMAL);
      free(db->threadCtx);
      db->threadCtx       = 0;
      db->numberOfThreads = 0;
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

    //[B1] Defensive: the synchronous path assumes exclusive buffer access — if a pipelined
    //StartUpdate was left uncollected (mixed sync/async use), drain it instead of racing.
    if (db->computationsRunning)
    {
        fprintf(stderr,YELLOW "db_update called with an uncollected StartUpdate in flight — draining it first\n" NORMAL);
        db_pipeline_drain(db);
    }

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

//[B1] db_StartUpdate/db_CollectUpdate are the async split of db_update and are the core of
//the double-buffer pipeline (DATALOADER.md B1). Historical note: an earlier attempt to use
//them "worked worse" — that was a SAME-BUFFER problem, not a protocol problem: without a
//second buffer set the consumer had to .copy() before the next Start, so nothing actually
//overlapped. With the shadow buffers the intended flow becomes
//  Collect(N) -> db_swap_buffers() -> Start(N+1) -> consume N from the shadow side.
//TODO(B1.0): switch db_CollectUpdate's wait to the timeout variant
//(threadpoolMainThreadWaitForWorkersToFinishTimeoutSeconds, MAXIMUM_WORKER_THREAD_TIMEOUT_SECONDS)
//exactly like db_update, so a hung worker aborts instead of deadlocking the pipeline.
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
      //[B1] Actually RELEASE the workers now (Prepare only arms + locks; the broadcast used
      //to live inside the combined wait call, so historically StartUpdate never really
      //started anything — the root cause of the old "async worked worse" verdict).
      threadpoolMainThreadKickWorkers(&db->threadPool);
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
      //[B1] WAIT-ONLY variant with timeout: workers were already released by StartUpdate's
      //Kick; the combined wait function would re-broadcast and reset activeWorkers mid-run.
      threadpoolMainThreadWaitForKickedWorkersToFinishTimeoutSeconds(&db->threadPool,MAXIMUM_WORKER_THREAD_TIMEOUT_SECONDS);
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
    //exit(0);

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
         unsigned long avgBkgHeight = (unsigned long) 10 * ((unsigned long) totalHeightWithBackground/totalSamplesWithBackgrounds);
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

//----------------------------------------------------------------------------------------------------
//[B1] Allocate the shadow (consume-side) buffer set — same dimensions as the primary
//fill-side set (metadata copied wholesale, only pixels/pixelsLimit differ). Only called
//when doubleBuffer is effectively on (streaming), so numberOfImages == batchSize and the
//cost is one extra batch (~213 MB at 256x256x76 / batch 40). The pixelsLimit formulas
//mirror db_create_without_dataset exactly.
//----------------------------------------------------------------------------------------------------
static int db_allocate_shadow_buffers(struct ImageDatabase * db)
{
    if (db==0) { return 0; }
    unsigned long numberOfImages = db->in.numberOfImages;

    db->inShadow        = db->in;
    db->inShadow.pixels = (void*) db_malloc(sizeof(unsigned char) * (numberOfImages+1) * db->in.width * db->in.height * db->in.channels);
    if (db->inShadow.pixels==0) { return 0; }
    db->inShadow.pixelsLimit = db->inShadow.pixels + (numberOfImages * db->in.width * db->in.height * db->in.channels);

    db->out8bitShadow        = db->out8bit;
    db->out8bitShadow.pixels = (void*) db_malloc(sizeof(char) * (numberOfImages+1) * db->out8bit.width * db->out8bit.height * db->out8bit.channels);
    if (db->out8bitShadow.pixels==0) { return 0; }
    db->out8bitShadow.pixelsLimit = db->out8bitShadow.pixels + (numberOfImages * db->out8bit.width * db->out8bit.height * db->out8bit.channels);

    db->out16bitShadow        = db->out16bit;
    db->out16bitShadow.pixels = (void*) db_malloc(sizeof(signed short) * (numberOfImages+1) * db->out16bit.width * db->out16bit.height * db->out16bit.channels);
    if (db->out16bitShadow.pixels==0) { return 0; }
    db->out16bitShadow.pixelsLimit = db->out16bitShadow.pixels + (sizeof(signed short) * numberOfImages * db->out16bit.width * db->out16bit.height * db->out16bit.channels);

    return 1;
}

//db->add flags have to have been set!
unsigned int calculateNumberOf8BitHeatmaps(int keypointsForEachSample,
                                           int addPAFs,
                                           int addBackgroundHeatmap,
                                           int addDepthMap,
                                           int addDepthLevelsHeatmaps,
                                           int addNormals,
                                           int addSegmentation,
                                           int addInstanceDetection,
                                           int addSuperpoint,
                                           int superpointChannels)
{
     unsigned int channelsOut8BitUsed = keypointsForEachSample +
                                        addBackgroundHeatmap +
                                        addDepthMap +
                                        addDepthLevelsHeatmaps +
                                        addNormals*3 +
                                        addPAFs*12 +
                                        (addSegmentation * (VALID_SEGMENTATIONS)) +
                                        (3*ENABLE_DENOISING_OUTPUT) +
                                        (2*ENABLE_LEFT_RIGHT_JOINT_DISAMBIGUATION_OUTPUT) +
                                        (addInstanceDetection * (INSTANCE_CHANNELS)) +
                                        (addSuperpoint * superpointChannels) +
                                        (g_addGeolocationHeatmap ? 1 : 0);
      return channelsOut8BitUsed;
}


void * db_create(struct DatabaseList* dbSources,
                 unsigned long numberOfSamples,
                 int streamData,
                 int doubleBuffer, //[B1] config `dataLoaderDoubleBuffer`; effective only with streamData
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
                 int addInstanceDetection,
                 int addSuperpoint,
                 int superpointChannels,
                 const char * superpointPcaPath,
                 int bytesPerDepthValue,
                 unsigned int widthIn,  unsigned int heightIn,unsigned int channelsIn,
                 unsigned int widthOut, unsigned int heightOut,unsigned int channelsOut8Bit,unsigned int channelsOut16Bit)
{
  fprintf(stderr,"\n\n%s DataLoader \n",version);
  fprintf(stderr,"PThread Worker Pool v%s\n\n",pthreadWorkerPoolVersion);

  instanceCounter+=1;
  fprintf(stderr,"Instance #%u\n",instanceCounter);

  //Mix pid + instance counter into the seeds: time(NULL) alone has 1-second granularity,
  //so the train and validation DataLoaders (constructed back-to-back) would otherwise get
  //identical seeds and correlated random streams. (DATALOADER.md A5)
  srand((unsigned int)(time(NULL) ^ ((unsigned int)getpid() * 2654435761u) ^ ((unsigned int)instanceCounter * 97u))); // Keep for any legacy rand() calls
  rng_seed((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ ((uint64_t)instanceCounter * 0x9E3779B97F4A7C15ULL) ^ 0xDEADBEEFCAFEBABEULL); // Seed fast per-thread PRNG

  if (dbSources==NULL) { return 0; }
  struct DatabaseList * dbl = (struct DatabaseList*) dbSources;

  fprintf(stderr,"db_create from %u sources \n",dbl->numberOfSources);
  fprintf(stderr,"Asked for %lu samples\n",numberOfSamples);
  fprintf(stderr,"In  : %ux%u:%u\n",widthIn,heightIn,channelsIn);
  fprintf(stderr,"8-Bit Out  : %ux%u:%u\n",widthOut,heightOut,channelsOut8Bit);
  fprintf(stderr,"16-Bit Out : %ux%u:%u\n",widthOut,heightOut,channelsOut16Bit);


   #if PROFILE_THREAD_ACTIVITY
     fprintf(stderr,RED "PROFILING THREAD ACTIVITY IS ON, YOU DONT WANT THIS WHEN ACTUALLY TRAINING\n" NORMAL);
     int ret = system("rm *.log");
     if (ret!=0) { fprintf(stderr,"Failed cleaning previous run log..\n"); }
   #endif // PROFILE_THREAD_ACTIVITY

  //.pzpd sources (PZPDLoader.c): all or none, opened as one handle (member = sourceID)
  unsigned int archiveSources = 0;
  for (unsigned int i=0; i<dbl->numberOfSources; i++) { archiveSources += (dbl->source[i].isArchive!=0); }
  if (archiveSources!=0)
  {
     if (archiveSources!=dbl->numberOfSources)
     {
       fprintf(stderr,RED "%u of %u sources are .pzpd archives: mixing archives and .db sources is not supported, convert every source (datasets/convertToPZPD.py)\n" NORMAL,archiveSources,dbl->numberOfSources);
       exit(1);
     }
     if ( (addSuperpoint) || (g_addGeolocationHeatmap) )
     {
       fprintf(stderr,RED ".pzpd archives carry no SuperPoint or geolocation data, disable addSuperpoint / addGeolocation\n" NORMAL);
       exit(1);
     }
     if (!pzpdOpenSources(dbl)) { exit(1); }
  }

  struct PoseDatabase* pdb = createPoseDatabase(dbl->totalNumberOfSamples,dbl->numberOfKeypointsPerSample);
  if (pdb!=0)
  {
     DatasetSourceID sID = 0;
     unsigned long startOffset = 0;
     for (sID=0; sID<dbl->numberOfSources; sID++)
     {
       fprintf(stderr,"Reading source #%lu | %s / %s  / %s / %s\n", sID, dbl->source[sID].pathToDBFile,dbl->source[sID].pathToCOCOImages,dbl->source[sID].pathToCOCODepthMaps,dbl->source[sID].pathToCOCOSegmentations);

       if (dbl->source[sID].isArchive)
       { //Joints, samples, captions and descriptors all come from the archive
         unsigned long loadedSamples = 0;
         if (!readPoseDatabaseFromArchive(pdb,dbl,sID,startOffset,&loadedSamples,dbl->source[sID].ignoreNoSkeletonSamples))
         {
           fprintf(stderr,RED "Failed reading from: `%s`\n" NORMAL,dbl->source[sID].pathToDBFile);
           exit(1);
         }
         if (dbl->source[sID].ignoreNoSkeletonSamples)
         {
             fprintf(stderr,"Ignoring background samples, so artificially reducing samples\n");
             fprintf(stderr,"from %lu to %lu \n",dbl->source[sID].numberOfSamples,loadedSamples);
             dbl->source[sID].numberOfSamples = loadedSamples; //as for .db sources
         }
         startOffset += dbl->source[sID].numberOfSamples;
         continue;
       }

       #if USE_DINOV2_FEATURES
       char dinoFilename[2048]={0};
       //snprintf(dinoFilename,2048,"%s.dinov3", dbl->source[sID].pathToDBFile);
       snprintf(dinoFilename, sizeof(dinoFilename), "%.*s.dinov3",(int)(sizeof(dinoFilename) - 8),  dbl->source[sID].pathToDBFile);// leave room for ".dinov3" and null terminator

       dbl->source[sID].descriptorsAsADataset=load_descriptor_bin(dinoFilename); //TODO: Deallocate this..
       if (dbl->source[sID].descriptorsAsADataset!=0) { fprintf(stderr,GREEN "Found a DinoV3 Descriptor file in %s \n" NORMAL,dinoFilename); } else
                                                      { fprintf(stderr,RED "Could not find a DinoV3 Descriptor file in %s \n" NORMAL,dinoFilename); }
       #else
         fprintf(stderr,"Dino V2 loading logic not included in this build\n");
       #endif // USE_DINOV2_FEATURES

       // SuperPoint sidecar (optional, runtime-gated — NOT behind USE_DINOV2_FEATURES).
       // Gated on addSuperpoint: a ".superpoint" file is loaded FULLY into memory by
       // load_superpoint_bin() (unlike the descriptor/.dinov3 path above, which is a fixed
       // per-sample vector), and on this project's coco14 sources that file is 54GB/26GB.
       // Previously this ran unconditionally regardless of the config flag, so any source
       // with a sidecar present paid that RAM cost even with SuperPoint fully unused
       // downstream (addSuperpoint=0 only gated the PCA-projection heatmap render, not the
       // load). NULL here matches what load_superpoint_bin() itself returns for a
       // missing/absent file, so every downstream consumer already treats "disabled" and
       // "no file present" identically.
       dbl->source[sID].superpointData = NULL;
       if (addSuperpoint)
       {
         char spFilename[2048]={0};
         snprintf(spFilename, sizeof(spFilename), "%.*s.superpoint",(int)(sizeof(spFilename) - 12), dbl->source[sID].pathToDBFile);
         dbl->source[sID].superpointData = load_superpoint_bin(spFilename);
         if (dbl->source[sID].superpointData!=0) { fprintf(stderr,GREEN "Found a SuperPoint file in %s (%d images)\n" NORMAL,spFilename,dbl->source[sID].superpointData->numImages); }
                                                 else { fprintf(stderr,"No SuperPoint file in %s \n",spFilename); }
       } else
       {
         fprintf(stderr,"SuperPoint disabled (addSuperpoint=0) — skipping .superpoint sidecar load\n");
       }

       unsigned long loadedSamples = 0;
       if (readPoseDatabase(pdb,dbl->source[sID].descriptorsAsADataset,dbl->source[sID].superpointData,dbl->source[sID].pathToDBFile,sID,startOffset,&loadedSamples,dbl->source[sID].ignoreNoSkeletonSamples))
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
         //Clamp to the loaded count: createPoseDatabase only allocated totalNumberOfSamples
         //entries, so a larger request walks uninitialized PoseEntry memory (garbage skeleton
         //counts, corrupt tokens, misleading "not properly captioned" abort). (DATALOADER.md A7)
         if (numberOfSamples > dbl->totalNumberOfSamples)
         {
             fprintf(stderr,YELLOW "Requested %lu samples but only %lu are loaded — clamping to avoid reading uninitialized samples!\n" NORMAL,
                     numberOfSamples, dbl->totalNumberOfSamples);
             numberOfSamples = dbl->totalNumberOfSamples;
         }
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
                                                                            addSegmentation,
                                                                            addInstanceDetection,
                                                                            addSuperpoint,
                                                                            superpointChannels);

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
         fprintf(stderr,"addInstanceDetection: %u \n",addInstanceDetection);
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
      db->addInstanceHeatmaps     = addInstanceDetection;
      db->addSuperpointHeatmaps   = (char) addSuperpoint;
      db->superpointChannels      = superpointChannels;
      db->addGeolocationHeatmap   = (char) g_addGeolocationHeatmap;
      db->combinedSegChannel       = (unsigned char) g_combinedSegChannel;
      db->combinedDepthHighChannel = (unsigned char) g_combinedDepthHighChannel;
      db->combinedDepthLowChannel  = (unsigned char) g_combinedDepthLowChannel;
      db->superpointPCA           = NULL;
      if (addSuperpoint && superpointChannels > 0 && superpointPcaPath != NULL && superpointPcaPath[0] != '\0')
      {
          db->superpointPCA = load_superpoint_pca(superpointPcaPath);
          if (db->superpointPCA == NULL)
          {
              fprintf(stderr,RED "SuperPoint enabled but PCA basis '%s' could not be loaded — superpoint channels will stay at background.\n" NORMAL, superpointPcaPath);
          }
          else if (superpointChannels > db->superpointPCA->maxComponents)
          {
              fprintf(stderr,YELLOW "superpointChannels=%d exceeds PCA maxComponents=%d — clamping at render time.\n" NORMAL, superpointChannels, db->superpointPCA->maxComponents);
          }
      }

      //WHERE DOES EACH THING GOES IS CONTROLLED HERE
      //Resolve the 8-bit heatmap channel layout from the add* flags above — this is
      //the single source of truth and derives db->PAFHeatmapIndex /
      //db->depthmapHeatmapIndex8Bit / db->normalsHeatmapIndex /
      //db->segmentationHeatmapIndex (see db_build_heatmap_layout / HeatmapLayout.h).
      db_build_heatmap_layout(db, pdb->keypointsForEachSample);

      //Seed augmentation parameters from compiled-in defaults (overridable at runtime).
      augmentationParamsSetDefaults(&db->aug);

      //[B1] Double buffering — STREAMING-ONLY (DATALOADER.md B1 mode matrix): in full-load
      //mode numberOfImages == numberOfSamples, a shadow set would double whole-dataset RAM
      //for zero benefit (single fill at db_create; get_in/out_array already serve stable
      //zero-copy views), so the flag is forced off with a printout.
      db->doubleBuffer = 0;
      if (doubleBuffer)
      {
        if (streamData)
        {
          if (db_allocate_shadow_buffers(db))
          {
            db->doubleBuffer = 1;
            fprintf(stdout,GREEN "Double buffering ENABLED: shadow batch buffers allocated, Start/Swap/Collect pipelining available\n" NORMAL);
          } else
          {
            fprintf(stderr,RED "Failed allocating shadow buffers for double buffering\n" NORMAL);
            abort();
          }
        } else
        {
          fprintf(stderr,YELLOW "dataLoaderDoubleBuffer requested but data is fully loaded (streamData=0) — forcing it OFF (nothing to overlap; a shadow set would double whole-dataset RAM)\n" NORMAL);
        }
      }

      if (!pzpdPrefetchCreate(db)) { abort(); }

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
        //[B1] Drain BEFORE stopping threads/freeing buffers — a still-running StartUpdate
        //writing into freed pixels would be a use-after-free.
        db_pipeline_drain(db);
        db_stop_threads(db);
        pzpdPrefetchDestroy(db); //every ticket was released by the stopped workers
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

        //[B1] Free the shadow set — pointers may have been exchanged by db_swap_buffers,
        //but freeing both sets' pixels covers both allocations regardless of swap parity.
        if (db->inShadow.pixels!=0)
        {
          free(db->inShadow.pixels);
          db->inShadow.pixels=0;
        }
        if (db->out8bitShadow.pixels!=0)
        {
          free(db->out8bitShadow.pixels);
          db->out8bitShadow.pixels=0;
        }
        if (db->out16bitShadow.pixels!=0)
        {
          free(db->out16bitShadow.pixels);
          db->out16bitShadow.pixels=0;
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

        if (db->superpointPCA!=0)
        {
         free_superpoint_pca(db->superpointPCA);
         db->superpointPCA = 0;
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
  int addInstanceDetection      = 1;
  int addPAFs                   = 1;
  int addSuperpoint             = 0;  // SuperPoint disabled in this test harness
  int superpointChannels        = 0;
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
                                                                addSegmentation,
                                                                addInstanceDetection,
                                                                addSuperpoint,
                                                                superpointChannels);

  unsigned int channelsOut16Bit = addDepthMap;
  //--------------------------------

  int profilingRun        = 0;
  int saveFilesToSeeThem  = 1;
  int saveAllLabels       = 0;
  int processDescriptions = 0;

  //[B1] Double-buffer benchmark knobs (both default off => canonical run is unchanged).
  //--doublebuffer runs the Start/swap/Collect pipeline instead of synchronous db_update;
  //--consumer <msec> simulates the per-batch consumer (GPU/training) step so the pipeline
  //has something to overlap the next batch's fill with.
  int useDoubleBuffer     = 0;
  unsigned int consumerMSec = 0;


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
        if (strcmp(argv[i],"--geolocation")==0)
                {
                    db_set_geolocation_config(1); //must precede db_create; adds the global density channel
                    fprintf(stderr,GREEN "Geolocation heatmap channel ENABLED\n" NORMAL);
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
        if (strcmp(argv[i],"--doublebuffer")==0)
                {
                    useDoubleBuffer = 1;
                    fprintf(stderr,GREEN "Double-buffer pipeline benchmark enabled\n" NORMAL);
                } else
        if (strcmp(argv[i],"--consumer")==0)
                {
                    consumerMSec = atoi(argv[i+1]);
                    fprintf(stderr,GREEN "Simulated consumer step set to %u msec/batch\n" NORMAL,consumerMSec);
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

  //Uncomment to test 16 bit depth code
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
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/coco14/cocoVal14.db",  "../../../ram/datasets/coco14/val2014", "../../../ram/datasets/coco14/depth_val2014", "../../../ram/datasets/coco14/segment_val2014", "../../../ram/datasets/coco14/all_val2014", ignoreNoSkeletonSamples);

   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/coco/cocoTrain.db",            "../../../ram/datasets/coco/cache/coco/train2017", "../../../ram/datasets/coco/cache/coco/depth_train2017", "../../../ram/datasets/coco/cache/coco/segment_train2017", "../../../ram/datasets/coco/cache/coco/all_train2017", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/coco14/cocoTrain14.db",        "../../../ram/datasets/coco14/train2014",          "../../../ram/datasets/coco14/depth_train2014",          "../../../ram/datasets/coco14/segment_train2014", "../../../ram/datasets/coco14/all_train2014", ignoreNoSkeletonSamples);

   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/300w/indoor.db",          "../../../ram/datasets/300w/indoor",            "../../../ram/datasets/300w/depth_indoor",            "../../../ram/datasets/300w/segment_indoor", "../../../ram/datasets/300w/all_indoor", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/300w/outdoor.db",         "../../../ram/datasets/300w/outdoor",           "../../../ram/datasets/300w/depth_outdoor",           "../../../ram/datasets/300w/segment_outdoor", "../../../ram/datasets/300w/all_outdoor", ignoreNoSkeletonSamples);

   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/background/AM-2k.db",          "../../../ram/datasets/background/AM-2k/train",    "../../../ram/datasets/background/AM-2k/depth_train",    "../../../ram/datasets/background/AM-2k/segment_train", "../../../ram/datasets/background/AM-2k/all_train", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/background/BG-20k.db",         "../../../ram/datasets/background/BG-20k/train",   "../../../ram/datasets/background/BG-20k/depth_train",   "../../../ram/datasets/background/BG-20k/segment_train", "../../../ram/datasets/background/BG-20k/all_train", ignoreNoSkeletonSamples);

   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/openpose/openposeTrain.db",    "../../../ram/datasets/openpose/data/train",       "../../../ram/datasets/openpose/data/depth_train",       "../../../ram/datasets/openpose/data/segment_train", "../../../ram/datasets/openpose/data/all_train", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/openpose/openposeBKG.db",      "../../../ram/datasets/openpose/data/bkg",         "../../../ram/datasets/openpose/data/depth_bkg" ,        "../../../ram/datasets/openpose/data/segment_bkg", "../../../ram/datasets/openpose/data/all_bkg", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/openpose/openposeFactory.db",  "../../../ram/datasets/openpose/data/factory",     "../../../ram/datasets/openpose/data/depth_factory" ,    "../../../ram/datasets/openpose/data/segment_factory", "../../../ram/datasets/openpose/data/all_factory", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../../../ram/datasets/openpose/openposeFactory2.db", "../../../ram/datasets/openpose/data/factory2",    "../../../ram/datasets/openpose/data/depth_factory2" ,   "../../../ram/datasets/openpose/data/segment_factory2", "../../../ram/datasets/openpose/data/all_factory2", ignoreNoSkeletonSamples);
  } else
  if (useValidationData)
  {
    sourcesRemaining = 2;
    sourceCounter    = 0;
    dbl = db_allocate_source_list(sourcesRemaining);
    sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco/cocoVal.db",            "../coco/cache/coco/val2017", "../coco/cache/coco/depth_val2017", "../coco/cache/coco/segment_val2017", "../coco/cache/coco/all_val2017", ignoreNoSkeletonSamples);
    sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco14/cocoVal14.db",        "../coco14/val2014",          "../coco14/depth_val2014",          "../coco14/segment_val2014", "../coco14/all_val2014", ignoreNoSkeletonSamples);
  } else
  #if ONLYVAL2017
  {
   fprintf(stderr,"Using only validation of 2017 COCO\n");
   sourcesRemaining = 1;
   sourceCounter    = 0;
   dbl = db_allocate_source_list(sourcesRemaining);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco/cocoVal.db",          "../coco/cache/coco/val2017",            "../coco/cache/coco/depth_val2017",            "../coco/cache/coco/segment_val2017", "../coco/cache/coco/all_val2017", ignoreNoSkeletonSamples);
  }
  #else
  {
   //sourcesRemaining = 1;
   sourcesRemaining = 8;//7;//10;
   sourceCounter    = 0;
   dbl = db_allocate_source_list(sourcesRemaining);

   //Uncomment for little val test
   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco/cocoVal.db",          "../coco/cache/coco/val2017",            "/home/ammar/Documents/Programming/PZP/test/depth_val2017PZP",            "/home/ammar/Documents/Programming/PZP/test/segment_val2017PZP", "/home/ammar/Documents/Programming/PZP/test/all_val2017PZP", ignoreNoSkeletonSamples);
   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco/cocoVal.db",          "../coco/cache/coco/val2017",            "../coco/cache/coco/depth_val2017",            "../coco/cache/coco/segment_val2017", "../coco/cache/coco/all_val2017", ignoreNoSkeletonSamples);

   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../300w/indoor_jpg.db",          "../300w/indoor_jpg",            "../300w/depth_indoor",            "../300w/segment_indoor", "../300w/all_indoor", ignoreNoSkeletonSamples);
   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../300w/outdoor_jpg.db",         "../300w/outdoor_jpg",           "../300w/depth_outdoor",           "../300w/segment_outdoor", "../300w/all_outdoor", ignoreNoSkeletonSamples);

   //Test PZP file encoding
   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../300w/indoor_jpg.db",          "../300w/indoor_jpg",            "../300w/depth_indoor_pzp",            "../300w/segment_indoor_pzp", "../300w/all_indoor_pzp", ignoreNoSkeletonSamples);
   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../300w/outdoor_jpg.db",         "../300w/outdoor_jpg",           "../300w/depth_outdoor_pzp",           "../300w/segment_outdoor_pzp", "../300w/all_outdoor_pzp", ignoreNoSkeletonSamples);


   //Comment all datasets underneath for little val test
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco14/cocoVal14.db",          "../coco14/val2014",            "../coco14/depth_val2014",            "../coco14/segment_val2014", "../coco14/all_val2014", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco/cocoTrain.db",            "../coco/cache/coco/train2017", "../coco/cache/coco/depth_train2017", "../coco/cache/coco/segment_train2017", "../coco/cache/coco/all_train2017", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../coco14/cocoTrain14.db",        "../coco14/train2014",          "../coco14/depth_train2014",          "../coco14/segment_train2014", "../coco14/all_train2014", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../background/AM-2k.db",          "../background/AM-2k/train",    "../background/AM-2k/depth_train",    "../background/AM-2k/segment_train", "../background/AM-2k/all_train", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../background/BG-20k.db",         "../background/BG-20k/train",   "../background/BG-20k/depth_train",   "../background/BG-20k/segment_train", "../background/BG-20k/all_train", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../openpose/openposeTrain.db",    "../openpose/data/train",       "../openpose/data/depth_train",       "../openpose/data/segment_train", "../openpose/data/all_train", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../openpose/openposeBKG.db",      "../openpose/data/bkg",         "../openpose/data/depth_bkg" ,        "../openpose/data/segment_bkg", "../openpose/data/all_bkg", ignoreNoSkeletonSamples);
   sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../openpose/openposeFactory.db",  "../openpose/data/factory",     "../openpose/data/depth_factory" ,    "../openpose/data/segment_factory", "../openpose/data/all_factory", ignoreNoSkeletonSamples);
   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../openpose/openposeFactory2.db", "../openpose/data/factory2",    "../openpose/data/depth_factory2" ,   "../openpose/data/segment_factory2", "../openpose/data/all_factory2", ignoreNoSkeletonSamples);

   //sourcesRemaining -= db_set_source_entry(dbl,sourceCounter++,"../generated/generatedTrain.db",  "../generated/data/train",      "../generated/data/depth_train" ,   "../generated/data/segment_train", "../generated/data/all_train",   ignoreNoSkeletonSamples);
  }
  #endif // ONLYVAL2017

  if (sourcesRemaining!=0)
  {
      fprintf(stderr,RED "Could not load all sources (%d sources could not be loaded), stopping test\n",sourcesRemaining);
      abort(); //<-- this is more dramatic :P
      return 0;
  }

  //Recompute after CLI flag parsing: --geolocation toggles the global that
  //calculateNumberOf8BitHeatmaps() reads, so the count above (pre-parse) can be stale.
  channelsOut8Bit = calculateNumberOf8BitHeatmaps(17,addPAFs,addBackgroundHeatmap,addDepthMap,
                                                  addDepthLevelsHeatmaps,addNormals,addSegmentation,
                                                  addInstanceDetection,addSuperpoint,superpointChannels);

  struct ImageDatabase*  db = db_create(dbl,
                                        numberOfSamples,
                                        1, //<- 0 = load all data at once / 1 = stream data
                                        useDoubleBuffer, //[B1] doubleBuffer: off by default (canonical sync path); --doublebuffer flips the benchmark to the pipeline

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
                                        addInstanceDetection,
                                        addSuperpoint,
                                        superpointChannels,
                                        NULL,
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
   const char vocabularySource[]="../../ymapnet_model/vocabulary.json";

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
                 SampleNumber sID = 0;
                 DatasetSourceID sourceID = 0;
                 if (db_resolve_sample_and_source(db,start+sampleNumber,&sID,&sourceID))
                    { fprintf(stderr,"sample%lu -> sourceID %lu / file %s \n",start+sampleNumber,sourceID,db->pdb->sample[sID].imagePath); }


                 snprintf(filename,2000,"%s/sample%05lu_BUG8bit_hm%d.png",path,sampleNumber,hmNumber);
                 db_save_heatmap8bit_as_png(db,filename,sampleNumber,hmNumber);
               }
              }
             }

            snprintf(filename,2000,"%s/sample%05lu_label.txt",path,sampleNumber+(rep*batchSize));
            //Pass the POSITION: db_save_descriptions_to_file's callees
            //(db_get_sample_description_tokens*) resolve through db->indices themselves.
            //The previous code resolved sID here first — a double resolution that read the
            //wrong sample's description after any shuffle. (found during the resolver refactor)
            db_save_descriptions_to_file(db,filename,start+sampleNumber,labelArray);

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
             fprintf(stderr,"Will now only go through %lu batches to profile code in valgrind\n",totalBatches);
           }// <- limit run for profiling
        else
           { fprintf(stderr,"Will now attempt to go through all %lu samples to identify bugs, this will take time in valgrind mode\n",db->numberOfSamples); }

        totalStartTime = GetTickCountMicrosecondsMN();

        if (useDoubleBuffer)
        {
          //[B1] Pipelined double-buffer benchmark. The consumer step of batch N overlaps the
          //worker fill of batch N+1:  Collect(N) -> swap -> Start(N+1) -> consume N (from the
          //shadow set).  This is a timing-only path; it does not save/validate batch contents.
          fprintf(stderr,GREEN "Double-buffer pipeline: consumer %u msec/batch, %lu batches\n" NORMAL,consumerMSec,totalBatches);
          if (totalBatches>0)
          {
            end = batchSize; if (end>db->numberOfSamples) { end = db->numberOfSamples; }
            db_StartUpdate(db,0,end,numberOfThreads,gradientSize,PAFSize); //kick batch 0
          }
          for (batchNumber=0; batchNumber<totalBatches; batchNumber++)
          {
            start = (batchNumber+0)*batchSize;
            end   = (batchNumber+1)*batchSize;
            if (end>db->numberOfSamples) { end = db->numberOfSamples; }

            startTime = GetTickCountMicrosecondsMN();

            //Wait for batch N's fill to finish, then publish it to the consumer (shadow) side.
            if (!db_CollectUpdate(db,start,end,numberOfThreads,gradientSize,PAFSize))
            {
                fprintf(stderr,"Failed collecting batch %lu (range %lu -> %lu)\n",batchNumber,start,end);
                abort();
            }
            db_swap_buffers(db); //batch N now readable via db_get_consumer_*; fill set is free

            //Kick batch N+1 into the freed fill set so it overlaps the consumer step below.
            if (batchNumber+1 < totalBatches)
            {
                unsigned long nStart = (batchNumber+1)*batchSize;
                unsigned long nEnd   = (batchNumber+2)*batchSize;
                if (nEnd>db->numberOfSamples) { nEnd = db->numberOfSamples; }
                db_StartUpdate(db,nStart,nEnd,numberOfThreads,gradientSize,PAFSize);
            }

            //Consume batch N (would read db_get_consumer_*); simulated GPU/training step.
            if (consumerMSec>0) { usleep(consumerMSec*1000); }

            endTime = GetTickCountMicrosecondsMN();
            thisMSec = (endTime-startTime)/1000;
            cumulTimeMSec += thisMSec;
            fprintf(stderr,"%u Threads | Samples %lu->%lu | Batch %lu/%lu | %u msec | Avg: %lu msec [PIPELINE]\n",numberOfThreads,start,end,batchNumber,totalBatches,thisMSec,(unsigned long) cumulTimeMSec/(batchNumber+1));
          }
        }
        else
        {
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
            //Consume batch N; on the sync path this is fully serial with the fill above.
            if (consumerMSec>0) { usleep(consumerMSec*1000); }
            endTime = GetTickCountMicrosecondsMN();

            thisMSec = (endTime-startTime)/1000;
            cumulTimeMSec += thisMSec;
            fprintf(stderr,"%u Threads | Samples %lu->%lu | Batch %lu/%lu | %u msec | Avg: %lu msec\n",numberOfThreads,start,end,batchNumber,totalBatches,thisMSec,(unsigned long) cumulTimeMSec/(batchNumber+1));
            db_print_readSpeed(db);
            if (db->threadStartLagMicroseconds>100)
             { fprintf(stderr,"Thread Lag Time to Start work %lu μsec / Time to prepare batch %lu μsec\n\n\n",db->threadStartLagMicroseconds,db->threadCompletionLagMicroseconds); } else
             { fprintf(stderr,"\n\n\n"); }
         }
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

