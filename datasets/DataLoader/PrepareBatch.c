#include <time.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "pthreadWorkerPool.h"
#include "DataLoader.h"

#include "codecs/codecs.h"
#include "codecs/image.h"
#include "codecs/pzp.h"

#include "cache.h"

#include "processing/resize.h"
#include "processing/augmentations.h"
#include "processing/normals.h"
#include "processing/draw.h"
#include "processing/conversions.h"
#include "processing/bilateral.h"
#include "processing/flip.h"
#include "processing/recalibration.h"

#include "DBLoader.h"
#include "HeatmapGenerator.h"

#include "PrepareBatch.h"
#include "PZPDLoader.h"
#include "tools.h"

#include <fcntl.h>


void initializeHeatmap16Bit(signed short* heatmapDestination16Bit, int width, int height, int channels)
{
    #define PATTERN_BUFFER_SIZE 512 // Arbitrary size for efficiency
    size_t totalElements = (size_t)width * height * channels;

    // Create a buffer filled with MINV_16BIT
    signed short patternBuffer[PATTERN_BUFFER_SIZE];
    for (size_t i = 0; i < PATTERN_BUFFER_SIZE; i++)
                       { patternBuffer[i] = MINV_16BIT; }

    // Copy the buffer repeatedly into the destination
    size_t copied = 0;
    while (copied < totalElements)
    {
        size_t chunkSize = (totalElements - copied > PATTERN_BUFFER_SIZE)
                           ? PATTERN_BUFFER_SIZE
                           : totalElements - copied;
        memcpy(&heatmapDestination16Bit[copied], patternBuffer, chunkSize * sizeof(signed short));
        copied += chunkSize;
    }
}

int cleanHeatmapsOfTargetSample(struct ImageDatabase * db ,unsigned long targetImageNumber)
{
  signed char* heatmapDestination8Bit = (signed char*) db->out8bit.pixels + (db->out8bit.width * db->out8bit.height * db->out8bit.channels * targetImageNumber);
  memset(heatmapDestination8Bit, (signed char) MINV, db->out8bit.width * db->out8bit.height * db->out8bit.channels * sizeof(signed char));

  //Nuke all 16bit heatmap data for our sample, memset doesn't work for 16bit data
  if (db->out16bit.pixels != NULL)
  {
    signed short* heatmapDestination16Bit = (signed short*) db->out16bit.pixels + (db->out16bit.width * db->out16bit.height * db->out16bit.channels * targetImageNumber);
    initializeHeatmap16Bit(heatmapDestination16Bit, db->out16bit.width, db->out16bit.height, db->out16bit.channels);
  }

  return 1;
}

int signalPrefetchFile(const char * filename)
{
    // Open the file to get the file descriptor
    //-------------------------------------------------------------------------------------
    int fd = open(filename, O_RDONLY);
    if (fd == -1) { fprintf(stderr,"Error prefetching file %s\n!", filename ); return -1; }
    // Use posix_fadvise to give the kernel advice about prefetching
    if (posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED | POSIX_FADV_SEQUENTIAL | POSIX_FADV_NOREUSE) != 0)   {  close(fd); return -1; }
    //if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL) != 0) {  close(fd); return -1; }
    //if (posix_fadvise(fd, 0, 0, POSIX_FADV_NOREUSE) != 0)    {  close(fd); return -1; }
    //-------------------------------------------------------------------------------------
    return fd;
}

void freeFileDescriptor(int fd)
{
   if (fd==0)
   {
       //Already empty file descriptor..
       //fprintf(stderr,"Bug?: freeFileDescriptor(0);\n");
       return;
   }
   if (fd!=-1)
      {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
      }
}



// Every resolved input-file path + codec flag for one sample, filled by resolveSampleFiles().
// Buffers are sized to match the [2049] path buffers / 2048 limit used by the resolver.
struct ResolvedSampleFiles
{
  char rgb[2049];           int rgbIsPZP;
  char depth[2049];         int depthIsPZP;   int depthFound;
  char segmentation[2049];  int segIsPZP;     int segFound;
  char combined[2049];      int combinedIsPZP;
  char geolocation[2049];   int geoFound;     // global lat/lon density PZP (single format)
  int  multiplexed;         // derived: combined[0] != '\0'
};

// [B5] resolvePathToRequestedFiles() used to fileExists()-probe up to ~6 file variants
// (depth pzp/png/_depth.png, seg pzp/png, combined pzp/png) for EVERY sample EVERY epoch —
// >600K stat() syscalls per epoch, all returning last epoch's answers. The choice is now
// resolved once (first touch) and cached in the sample's PoseEntry (enum DL_FileKind); on
// every later touch the paths are rebuilt from the cached kind with zero stat() calls.
// Output strings/flags are byte-identical to the original probing for all cases (incl. the
// "not found" fall-throughs), so downstream behaviour is unchanged. depthFound/segFound are
// derived from the cached kind so the caller can drop its own fileExists() re-check too.
void resolvePathToRequestedFiles(
                                 const char * pathToCOCOImages,
                                 const char * pathToCOCODepthMaps,
                                 const char * pathToCOCOSegmentations,
                                 const char * pathToCombinedMetaData,
                                 const char * pathToCOCOGeolocation,
                                 const char * thisFilename,
                                 int limit,
                                 char* tmp,
                                 struct PoseEntry * sampleCache,
                                 char * pathToRGBFile,
                                 int *rgbIsPZP,
                                 char * pathToDepthFile,
                                 int * depthIsPZP,
                                 int * depthFound,
                                 char * pathToSegmentationFile,
                                 int * segIsPZP,
                                 int * segFound,
                                 char * pathToCombinedMetaDataFile,
                                 int * combinedIsPZP,
                                 char * pathToGeolocationFile,
                                 int * geoFound
                                )
{
  int alreadyResolved = (sampleCache!=0) && (sampleCache->pathResolved);
  unsigned char depthKind    = alreadyResolved ? sampleCache->depthKind    : DL_FILE_NONE;
  unsigned char segKind      = alreadyResolved ? sampleCache->segKind      : DL_FILE_NONE;
  unsigned char combinedKind = alreadyResolved ? sampleCache->combinedKind : DL_FILE_NONE;
  unsigned char geoKind      = alreadyResolved ? sampleCache->geoKind      : DL_FILE_NONE;

  //Resolve RGB Input image path (never probed — always <images>/<filename>)
  snprintf(pathToRGBFile,limit,"%.1023s/%.1023s",pathToCOCOImages,thisFilename);
  *rgbIsPZP = 0;

  //Resolve Depth Input image path
  *depthFound = 0;
  snprintf(tmp,limit/2,"%s",thisFilename);
  if (strlen(tmp)>4)
            {
               tmp[strlen(tmp)-4]=0; //quick & dirty remove extension

               if (!alreadyResolved)
               {
                 //FIRST TOUCH: probe disk. Depth Anything uses %s_depth.png / V2 uses %s.png
                 depthKind = DL_FILE_NONE;
                 snprintf(pathToDepthFile,limit,"%.1023s/%.1000s.pzp",pathToCOCODepthMaps,tmp);
                 if (fileExists(pathToDepthFile)) { depthKind = DL_FILE_PZP; }
                 else
                 {
                  snprintf(pathToDepthFile,limit,"%.1023s/%.1000s.png",pathToCOCODepthMaps,tmp);
                  if (fileExists(pathToDepthFile)) { depthKind = DL_FILE_PNG; }
                  else
                  {
                   snprintf(pathToDepthFile,limit,"%.1023s/%.1000s_depth.png",pathToCOCODepthMaps,tmp);
                   if (fileExists(pathToDepthFile)) { depthKind = DL_FILE_DEPTHPNG; }
                  }
                 }
               }

               //Rebuild the resolved path + flags from the kind (byte-identical to the probe,
               //including the "not found" case which leaves the _depth.png variant string).
               switch (depthKind)
               {
                 case DL_FILE_PZP:      *depthIsPZP=1; snprintf(pathToDepthFile,limit,"%.1023s/%.1000s.pzp",pathToCOCODepthMaps,tmp);        *depthFound=1; break;
                 case DL_FILE_PNG:      *depthIsPZP=0; snprintf(pathToDepthFile,limit,"%.1023s/%.1000s.png",pathToCOCODepthMaps,tmp);        *depthFound=1; break;
                 case DL_FILE_DEPTHPNG: *depthIsPZP=0; snprintf(pathToDepthFile,limit,"%.1023s/%.1000s_depth.png",pathToCOCODepthMaps,tmp); *depthFound=1; break;
                 default:               *depthIsPZP=0; snprintf(pathToDepthFile,limit,"%.1023s/%.1000s_depth.png",pathToCOCODepthMaps,tmp); *depthFound=0; break;
               }
            }

  //Resolve Segmentation File Input image path
  *segFound = 0;
  snprintf(tmp,limit/2,"%s",thisFilename);
            if (strlen(tmp)>4)
            {
               tmp[strlen(tmp)-4]=0; //quick & dirty remove extension

               if (!alreadyResolved)
               {
                 //FIRST TOUCH: probe disk (PZP takes priority, then PNG)
                 segKind = DL_FILE_NONE;
                 snprintf(pathToSegmentationFile,limit,"%.1023s/%.1000s.pzp",pathToCOCOSegmentations,tmp);
                 if (fileExists(pathToSegmentationFile)) { segKind = DL_FILE_PZP; }
                 else
                 {
                  snprintf(pathToSegmentationFile,limit,"%.1023s/%.1000s.png",pathToCOCOSegmentations,tmp);
                  if (fileExists(pathToSegmentationFile)) { segKind = DL_FILE_PNG; }
                 }
               }

               //Rebuild from kind (the "not found" case leaves the .png variant string, as before)
               switch (segKind)
               {
                 case DL_FILE_PZP: *segIsPZP=1; snprintf(pathToSegmentationFile,limit,"%.1023s/%.1000s.pzp",pathToCOCOSegmentations,tmp); *segFound=1; break;
                 case DL_FILE_PNG: *segIsPZP=0; snprintf(pathToSegmentationFile,limit,"%.1023s/%.1000s.png",pathToCOCOSegmentations,tmp); *segFound=1; break;
                 default:          *segIsPZP=0; snprintf(pathToSegmentationFile,limit,"%.1023s/%.1000s.png",pathToCOCOSegmentations,tmp); *segFound=0; break;
               }
            }

  //Resolve Combined Metadata (depth+segmentation multiplexed) file path.
  //PZP takes priority; fallback to PNG.  If neither exists leave the path empty.
  *combinedIsPZP = 0;
  pathToCombinedMetaDataFile[0] = '\0';
  if (pathToCombinedMetaData && strlen(pathToCombinedMetaData) > 0)
  {
      snprintf(tmp, limit/2, "%s", thisFilename);
      if (strlen(tmp) > 4)
      {
          tmp[strlen(tmp)-4] = '\0'; // strip extension

          if (!alreadyResolved)
          {
            //FIRST TOUCH: probe disk
            combinedKind = DL_FILE_NONE;
            snprintf(pathToCombinedMetaDataFile, limit, "%.1023s/%.1000s.pzp", pathToCombinedMetaData, tmp);
            if (fileExists(pathToCombinedMetaDataFile)) { combinedKind = DL_FILE_PZP; }
            else
            {
                snprintf(pathToCombinedMetaDataFile, limit, "%.1023s/%.1000s.png", pathToCombinedMetaData, tmp);
                if (fileExists(pathToCombinedMetaDataFile)) { combinedKind = DL_FILE_PNG; }
            }
          }

          //Rebuild from kind (NONE leaves the path empty, matching the original convention)
          switch (combinedKind)
          {
            case DL_FILE_PZP: *combinedIsPZP=1; snprintf(pathToCombinedMetaDataFile, limit, "%.1023s/%.1000s.pzp", pathToCombinedMetaData, tmp); break;
            case DL_FILE_PNG: *combinedIsPZP=0; snprintf(pathToCombinedMetaDataFile, limit, "%.1023s/%.1000s.png", pathToCombinedMetaData, tmp); break;
            default:          *combinedIsPZP=0; pathToCombinedMetaDataFile[0] = '\0'; break;
          }
      }
  }

  //Resolve Geolocation density file path. Single format (.pzp) and — unlike depth/seg —
  //named "<imageFilename>.pzp" (extension KEPT), matching run_plonk_teacher.py's output.
  //Un-augmented global heatmap; the caller copies it verbatim after all augmentation.
  *geoFound = 0;
  pathToGeolocationFile[0] = '\0';
  if (pathToCOCOGeolocation && pathToCOCOGeolocation[0] != '\0')
  {
      if (!alreadyResolved)
      {
        geoKind = DL_FILE_NONE;
        snprintf(pathToGeolocationFile, limit, "%.1023s/%.1023s.pzp", pathToCOCOGeolocation, thisFilename);
        if (fileExists(pathToGeolocationFile)) { geoKind = DL_FILE_PZP; }
      }
      switch (geoKind)
      {
        case DL_FILE_PZP: snprintf(pathToGeolocationFile, limit, "%.1023s/%.1023s.pzp", pathToCOCOGeolocation, thisFilename); *geoFound=1; break;
        default:          pathToGeolocationFile[0] = '\0'; break;
      }
  }

  //Publish the resolution to the cache on first touch (kinds written before the flag, which
  //is read by later epochs after a full worker barrier — no concurrent touch of one sample).
  if (!alreadyResolved && sampleCache!=0)
  {
      sampleCache->depthKind    = depthKind;
      sampleCache->segKind      = segKind;
      sampleCache->combinedKind = combinedKind;
      sampleCache->geoKind      = geoKind;
      sampleCache->pathResolved = 1;
  }

  return ;
}


// Bundles the per-sample file resolution that both the worker and the preloader repeated
// verbatim: pull the source's 4 base paths, build every input path/codec via
// resolvePathToRequestedFiles(), and derive 'multiplexed'. Takes an already-resolved
// sampleNumber/sourceID (both callers get these from db_resolve_sample_and_source before
// they need them) so this is a sibling to that call, not a second resolution of it.
// 'tmp' is caller scratch (>= limit); every field of 'out' is fully written.
void resolveSampleFiles(struct ImageDatabase * db,
                        SampleNumber sampleNumber,
                        DatasetSourceID sourceID,
                        char * tmp,
                        struct ResolvedSampleFiles * out)
{
  resolvePathToRequestedFiles(
                              db->dbSources->source[sourceID].pathToCOCOImages,
                              db->dbSources->source[sourceID].pathToCOCODepthMaps,
                              db->dbSources->source[sourceID].pathToCOCOSegmentations,
                              db->dbSources->source[sourceID].pathToCombinedMetaData,
                              db->dbSources->source[sourceID].pathToCOCOGeolocation,
                              db->pdb->sample[sampleNumber].imagePath,
                              2048,tmp,
                              &db->pdb->sample[sampleNumber],
                              out->rgb,          &out->rgbIsPZP,
                              out->depth,        &out->depthIsPZP,  &out->depthFound,
                              out->segmentation, &out->segIsPZP,    &out->segFound,
                              out->combined,     &out->combinedIsPZP,
                              out->geolocation,  &out->geoFound);

  //DEPTH: a combined ("all") file wins over the separate depth/segmentation files (DEPTH STEP 0a)
  out->multiplexed = (out->combined[0] != '\0');
}

// The .pzpd counterpart of resolveSampleFiles(): presence and codecs come from the sample's cache,
// filled from the archive index at load (PZPDLoader.c). The "paths" only name the sample in messages.
void resolveArchiveSampleFiles(struct ImageDatabase * db,
                               SampleNumber sampleNumber,
                               struct ResolvedSampleFiles * out)
{
  const struct PoseEntry * e = &db->pdb->sample[sampleNumber];
  snprintf(out->rgb,         sizeof(out->rgb),         "%s (archive rgb)",  e->imagePath);
  snprintf(out->depth,       sizeof(out->depth),       "%s (archive depth)",e->imagePath);
  snprintf(out->segmentation,sizeof(out->segmentation),"%s (archive seg)",  e->imagePath);
  out->combined[0] = '\0';
  if (e->combinedKind!=DL_FILE_NONE) { snprintf(out->combined,sizeof(out->combined),"%s (archive all)",e->imagePath); }
  out->geolocation[0] = '\0';
  out->rgbIsPZP      = 0;
  out->depthIsPZP    = (e->depthKind==DL_FILE_PZP);
  out->depthFound    = (e->depthKind!=DL_FILE_NONE);
  out->segIsPZP      = (e->segKind==DL_FILE_PZP);
  out->segFound      = (e->segKind!=DL_FILE_NONE);
  out->combinedIsPZP = (e->combinedKind==DL_FILE_PZP);
  out->geoFound      = 0;
  //DEPTH: a record with an "all" blob is multiplexed (DEPTH STEP 0a) and its "depth" / "seg" blobs are not read
  out->multiplexed   = (out->combined[0] != '\0');
}


struct Image * cachedReadImage(struct cache * c,unsigned int threadID,const char *filename,unsigned int type)
{
    #if USE_RAM_CACHE
        unsigned long dataSize=0;
        void * data = cache_open(c,filename,&dataSize);
        if ( (data!=0) && (dataSize!=0) )
        {
           return readImageFromMemory(filename,data,dataSize,type);
        } else
        {
         fprintf(stderr,"cachedReadImage: RAM Cache failed to open %s\n",filename);
         abort();
        }
    #else
        #if READ_WHOLE_IMAGE_FILE_BEFORE_DECODING
        //This seems to work ~10% better than regular reading
        size_t imageSizeInBytes = 0;
        void * imageInMemory = read_file_to_common_memory_of_cache(c,threadID,filename,&imageSizeInBytes);

        if (imageInMemory!=NULL)
        {
            return readImageFromMemory(filename,imageInMemory,imageSizeInBytes,type);
        } else
        {
            fprintf(stderr,RED "cachedReadImage: Error reading file %s into memory!" NORMAL,filename);
            abort();
        }
        #else
         //Regular reading from the provided filename
         return readImage(filename,type,0);
        #endif
    #endif // USE_RAM_CACHE
};


void preloadAllFiles(struct ImageDatabase * db)
{
    unsigned long iteration;
    struct ResolvedSampleFiles files = {0};
    char *rgbpath           = files.rgb;
    char *depthpath         = files.depth;
    char *segmentationspath = files.segmentation;
    char *combinedmdatapath = files.combined;
    char tmp[2049]               = {0};

    fprintf(stderr,"Preloading %lu samples\n ",db->numberOfSamples);
    for (iteration=0; iteration<db->numberOfSamples; iteration++)
        {
          if (db->numberOfSamples>iteration)
           {//Check if iteration is valid..
            //Canonical position -> sampleNumber -> sourceID resolution (single source of
            //truth, see db_resolve_sample_and_source in DataLoader.c).
            unsigned long sampleNumber = 0;
            DatasetSourceID sourceID   = 0;
            if (!db_resolve_sample_and_source(db,iteration,&sampleNumber,&sourceID))
            {
               fprintf(stderr,RED "preloadAllFiles could not resolve sample for iteration %lu, aborting\n" NORMAL,iteration);
               abort();
            }


            //Resolve every input-file path/codec for this sample into 'files' (paths aliased above).
            resolveSampleFiles(db,sampleNumber,sourceID,tmp,&files);


            unsigned long hash = 0;
            unsigned long filenameSize=0;
            //------------------------------------------------
            hash = hash_filename(rgbpath);
            cache_add(db->files,rgbpath,hash,&filenameSize,iteration%9 == 0);
            //------------------------------------------------
            if (combinedmdatapath[0] != '\0')
            {
                hash = hash_filename(combinedmdatapath);
                cache_add(db->files,combinedmdatapath,hash,&filenameSize,0);
            }
            else
            {
                hash = hash_filename(depthpath);
                cache_add(db->files,depthpath,hash,&filenameSize,0);
                //------------------------------------------------
                hash = hash_filename(segmentationspath);
                cache_add(db->files,segmentationspath,hash,&filenameSize,0);
            }
            //------------------------------------------------
            //Geolocation density PZP (only when enabled and present for this sample).
            if (db->addGeolocationHeatmap && files.geoFound && files.geolocation[0] != '\0')
            {
                hash = hash_filename(files.geolocation);
                cache_add(db->files,files.geolocation,hash,&filenameSize,0);
            }
           }
        }

    fprintf(stderr,"\n\nDone preloading..\n ");
    fprintf(stderr,"Indexing..\n ");
    cache_arrange(db->files);
    fprintf(stderr,"Done Indexing..\n ");
}



void logThreadProgress(unsigned int threadID,char start,const char * part)
{
 #if PROFILE_THREAD_ACTIVITY
 char filename[512]={0};
 if (threadID==255) { snprintf(filename,512,"thread_main.log"); } else
                    { snprintf(filename,512,"thread_%02u.log",threadID); }
 FILE *fp = fopen(filename,"a");
 if (fp!=0)
 {
     fprintf(fp,"%lu,%u,%s\n",GetTickCountMicrosecondsMN(),(int) start,part);
     fclose(fp);
 }
 #endif // PROFILE_THREAD_ACTIVITY
}

void logMainThreadProgress(char start,const char * part)
{
 #if PROFILE_THREAD_ACTIVITY
  logThreadProgress(255,start,part);
 #endif // PROFILE_THREAD_ACTIVITY
}

void logAllThreadsProgress(unsigned int numberOfThreads,char start,const char * part)
{
 #if PROFILE_THREAD_ACTIVITY
    for (int threadID=0; threadID<numberOfThreads; threadID++)
                { logThreadProgress(threadID,start,part); }
 #endif // PROFILE_THREAD_ACTIVITY
}


void *workerThread(void * arg)
{
    //We are a worker thread so let's retrieve our variables..
    //--------------------------------------------------------------
    struct threadContext * ptr = (struct threadContext *) arg;
    if (ptr==0) { fprintf(stderr,"Worker thread could not retreive its context, aborting \n"); abort(); }
    fprintf(stdout,"TS-%u ",ptr->threadID);
    struct workerThreadContext * contextArray = (struct workerThreadContext *) ptr->argumentToPass;
    if (contextArray==0) { fprintf(stderr,"Worker thread could not retreive its context array, aborting \n"); abort(); }
    struct workerThreadContext * ctx = &contextArray[ptr->threadID];
    if (contextArray==0) { fprintf(stderr,"Worker thread could not retreive its context array item, aborting \n"); abort(); }
    //--------------------------------------------------------------

    //Declare helper buffer once avoid spamming malloc
    //float *gradientX  = 0;
    //float *gradientY  = 0;
    //We use it in ctx->gradientX and ctx->gradientY to be able to dealloc it
    unsigned long startTime = GetTickCountMicrosecondsMN(); //Start time for ctx->lagToFinishWorkMicroseconds;

    threadpoolWorkerInitialWait(ptr);

    // Seed the per-thread fast PRNG (splitmix64) so each worker has independent
    // randomness without contending on glibc's global rand() lock.
    // The db instance pointer is mixed in so worker k of the TRAIN pool and worker k of
    // the VALIDATION pool (created within the same 1-second time() tick) do not end up
    // with identical seeds / correlated augmentation streams. (DATALOADER.md A5)
    rng_seed((uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)ctx->db ^ ((uint64_t)ptr->threadID * 0x9E3779B97F4A7C15ULL));

    //---------------------------------------------------
    //DEBUG BEHAVIOR OF DIFFERENT FUNCTIONS
    //DISABLE THEM TO CHECK THEM..
    //---------------------------------------------------
    const int MASTER_SWITCH   = ctx->db->doHeatmapOutput;
    const int DO_KEYPOINTS    = MASTER_SWITCH;
    const int DO_DEPTH        = MASTER_SWITCH;
    const int DO_NORMALS      = MASTER_SWITCH;
    const int DO_SEGMENTATION = MASTER_SWITCH;
    //---------------------------------------------------

    //if (stick_this_thread_to_core(ctx->thisThreadNumber)!=0) // <- Stick this thread to a specific core
    //   { fprintf(stderr,"Could not stick thread %u to a core\n",ctx->thisThreadNumber);  }

    while (threadpoolWorkerLoopCondition(ptr))
    {
      ctx->lagToStartWorkMicroseconds = GetTickCountMicrosecondsMN() - startTime;
      //--------------------------------------------------------------
      struct ImageDatabase * db        = ctx->db;
      struct PoseDatabase*  pdb        = db->pdb;
      if ( (db==0) || (pdb==0))   { fprintf(stderr,"Worker thread received empty database data, aborting \n"); abort(); }
      unsigned long sampleStart        = ctx->sampleStart;
      unsigned long sampleEnd          = ctx->sampleEnd;
      if (sampleStart>sampleEnd)  { fprintf(stderr,"Erroneous sample range (%lu->%lu), aborting \n",sampleStart,sampleEnd); abort(); }
      unsigned int thisThreadNumber    = ctx->thisThreadNumber;
      unsigned int workerThreads       = ctx->workerThreads;
      unsigned int gradientSize        = ctx->gradientSize;
      unsigned int PAFSize             = ctx->PAFSize;
      //--------------------------------------------------------------
      struct ResolvedSampleFiles files = {0};
      char *rgbpath           = files.rgb;
      char *depthpath         = files.depth;
      char *segmentationspath = files.segmentation;
      char *combinedmdatapath = files.combined;
      char tmp[2049]                   = {0};
      unsigned long iteration          = 0;
      SampleNumber sampleNumber        = 0;
      unsigned long targetImageNumber  = 0;
      //--------------------------------------------------------------


       if (ctx->gradientX==0)
         { ctx->gradientX = (float *) db_malloc((db->out8bit.width+1) * (db->out8bit.height+1) * sizeof(float));  }
       if (ctx->gradientY==0)
         { ctx->gradientY = (float *) db_malloc((db->out8bit.width+1) * (db->out8bit.height+1) * sizeof(float)); }



      if (ctx->gradientX!=0)
         { memset(ctx->gradientX,0,(db->out8bit.width+1) * (db->out8bit.height+1) * sizeof(float)); }
      if (ctx->gradientY!=0)
         { memset(ctx->gradientY,0,(db->out8bit.width+1) * (db->out8bit.height+1) * sizeof(float)); }

      //if (inputRGBBuffer==0)
      //   { inputRGBBuffer = (signed char *) malloc(db->in.width * db->in.height * db->in.channels * sizeof(unsigned char)); }
      //signed char * outputRGBBuffer = 0;
      //--------------------------------------------------------------

      //fprintf(stderr,"Inputs are  : %hux%hu:%hu\n",db->in.width,db->in.height,db->in.channels);
      //fprintf(stderr,"Outputs are : %hux%hu:%hu\n",db->out.width,db->out.height,db->out.channels);

      struct Image * img  = 0;
      struct Image * dpth = 0;
      struct Image * segment = 0;
      float offX,offY,sclX,sclY; // <- Careful these are a second set of variables..!

      //[B1] Double-buffer design property (DATALOADER.md B1): this whole worker body reads
      //and writes ONLY through db->in / db->out8bit / db->out16bit — which stay the
      //fill-side buffers under the double-buffer plan (db_swap_buffers exchanges pixel
      //pointers between fill and shadow sets while the pipeline is idle). Therefore
      //NOTHING in this hot path changes when double buffering lands. Keep it that way:
      //new code here must keep going through db->in/out*, never a cached pixels pointer
      //taken before the batch started.
      if ( (db->in.numberOfImages!=0) && (db->out8bit.numberOfImages!=0) )
      { //Guard against activation without data..!
       for (iteration=sampleStart; iteration<sampleEnd; iteration++)
        {
           if (iteration%workerThreads == thisThreadNumber)
           {
            //Iteration should be a number e.g. 0-31 (for batch size 32)
            //Based on our shuffled indices we can select a DB sampleNumber

           if (db->numberOfSamples>iteration)
           {//Check if iteration is valid..
            //Canonical position -> sampleNumber -> sourceID resolution: all bounds and
            //corruption checks (which used to be inlined here) live inside
            //db_resolve_sample_and_source — index/source corruption aborts in there.
            DatasetSourceID sourceID = 0;
            if (!db_resolve_sample_and_source(db,iteration,&sampleNumber,&sourceID))
            {
               fprintf(stderr,RED "Worker thread %u could not resolve sample for iteration %lu, aborting\n" NORMAL,thisThreadNumber,iteration);
               abort();
            }

            //The target image should go to the same place as the iteration (to make sure we dont run out of memory)
            targetImageNumber = iteration-sampleStart; //We start at 0 always

            if (targetImageNumber>=db->in.numberOfImages)
            {
               fprintf(stderr,"\n\n\nOut of memory access for sample %lu / iteration %lu / limit %lu \n",sampleNumber,iteration,db->in.numberOfImages);
               fprintf(stderr,"Asked to access IN %lu / %lu \n",targetImageNumber,db->in.numberOfImages);
               fprintf(stderr,"Start %lu / End %lu \n",sampleStart,sampleEnd);
               targetImageNumber = targetImageNumber % db->in.numberOfImages;
               abort();
            }

            if (targetImageNumber>=db->out8bit.numberOfImages)
            {
               fprintf(stderr,"\n\n\nOut of memory access for sample %lu / iteration %lu / limit %lu\n",sampleNumber,iteration,db->out8bit.numberOfImages);
               fprintf(stderr,"Asked to access OUT %lu / %lu \n",targetImageNumber,db->in.numberOfImages);
               fprintf(stderr,"Start %lu / End %lu \n",sampleStart,sampleEnd);
               targetImageNumber = targetImageNumber % db->out8bit.numberOfImages;
               abort();
            }

            //Resolve every input-file path/codec for this sample into 'files' (paths aliased above).
            //Archive sources take the sample's blobs (prefetched, or mmap views) instead of files.
            const int fromArchive = (db->dbSources->archive!=0);
            struct ArchiveSample archiveSample;
            if (fromArchive)
            {
              resolveArchiveSampleFiles(db,sampleNumber,&files);
              pzpdSampleBegin(db,sampleNumber,&archiveSample);
            } else
            {
              resolveSampleFiles(db,sampleNumber,sourceID,tmp,&files);
            }
            int rgbIsPZP      = files.rgbIsPZP;
            int depthIsPZP    = files.depthIsPZP;
            int segIsPZP      = files.segIsPZP;
            int combinedIsPZP = files.combinedIsPZP;
            int depthFound    = files.depthFound;
            int segFound      = files.segFound;
            int depthAndSegAreMultiplexed = files.multiplexed;

            float progress = 0.0;
            if (sampleEnd-sampleStart>0) { progress = 100.0 * ((float) (iteration-sampleStart) / (sampleEnd-sampleStart)); }
            if (iteration%100==thisThreadNumber)
              { fprintf(stderr,"\r  Thread %u / %0.2f%% (%s) / %u skeletons       \r",thisThreadNumber,progress,pdb->sample[sampleNumber].imagePath,pdb->sample[sampleNumber].numberOfSkeletons); }


            //###############################################################
            //         Combined Depth and Segmentation loading
            //###############################################################
            // DEPTH STEP 0a (multiplexed sources only, see the DEPTH PIPELINE overview further down).
            // A multiplexed ("all") file packs segmentation AND 16-bit depth into one 8-bit, 3-channel image:
            //   channel db->combinedSegChannel       (default 0) = segmentation label
            //   channel db->combinedDepthHighChannel (default 1) = depth >> 8   (high byte)
            //   channel db->combinedDepthLowChannel  (default 2) = depth & 0xFF (low byte)
            // (layout of append16bitToSAM3.py / compressSegmentDepth.py; changeable from Python with
            // DataLoader(combinedChannels=...) -> db_set_combined_layout). The file comes from the archive's "all"
            // stream or, for .db sources, from the all_* directory (given, or derived from depth_* by
            // convertDepthPathToAllPath). When it exists it is used INSTEAD of the separate depth/segmentation
            // files. splitSegmentationAndDepthFromSingleFileGeneric rebuilds depth = (high << 8) | low as a
            // NATIVE-endian unsigned short image (dpth); depth STEP 2 below must therefore NOT byte-swap it.
            // The DO_* flags come from db->doHeatmapOutput (the "master switch" that
            // db_disable_heatmap_output() clears for token-only training). Every separate
            // depth/segmentation path already honours it; this multiplexed path did not, so
            // with heatmaps off it still read+decoded the combined file and split it into two
            // images that nothing downstream consumed OR freed (~25MB leaked per batch, and a
            // wasted PZP decode per sample). Gate it the same way.
            if ( (depthAndSegAreMultiplexed) && ((DO_DEPTH) || (DO_SEGMENTATION)) )
            {
                logThreadProgress(thisThreadNumber,1,"combined_depth_segmentation_loading");
                struct Image * combined = fromArchive ? pzpdSampleReadImage(db,sampleNumber,&archiveSample,PZPD_STREAM_ALL) :
                                                        cachedReadImage(db->files, thisThreadNumber, combinedmdatapath, combinedIsPZP ? PZP_CODEC : PNG_CODEC);
                if (combined && combined->pixels)
                {
                    if (combinedIsPZP) { combined->bitsperpixel = 8; } // PZP stores as 24-bit RGB; generic expects 8-bit flag
                    splitSegmentationAndDepthFromSingleFileGeneric(combined, &segment, &dpth, db->combinedSegChannel,
                                                                   db->combinedDepthHighChannel, db->combinedDepthLowChannel);
                    // combined is consumed (destroyed) by splitSegmentationAndDepthFromSingleFileGeneric
                }
                logThreadProgress(thisThreadNumber,0,"combined_depth_segmentation_loading");
            }
            //###############################################################
            //###############################################################

            //First of all we clean target heatmaps..
           //===========================================================================================================
            cleanHeatmapsOfTargetSample(db,targetImageNumber);
           //At this point the heatmap for depth map is clean
           //===========================================================================================================

            int eraseSample = 0;

            //Completely destroy 1% of background images (they dont have skeletons) with total noise
            if ( (pdb->sample[sampleNumber].numberOfSkeletons==0) && (eventOccurs(db->aug.chanceDestroy)) )
            {
                  //This is a seperate data path if the destructive eraseSample mode is enabled
                  //This skips any disk I/O at the cost of making the code a little more complex
                  //Be careful for bugs!
                  eraseSample = 1; //If we choose to go full random we should erase the depth map!
                  pdb->sample[sampleNumber].eraseEntry = eraseSample;
                  struct Image randomImage={0};
                  db_map_sample_in_to_image((void*) db,&randomImage,targetImageNumber);
                  //-------------------------------------------------------------------------------
                  if (randomImage.pixels!=0)
                  {
                   cleanHeatmapsOfTargetSample(db,targetImageNumber);
                   createRandomImageRGB(&randomImage);
                  } else
                  {
                      fprintf(stderr,"Error creating random image (Out of memory?)\n");
                      abort();
                  }
            } else
            { //Normal Sample loading

             //When se to zero it means they are not set up!
             int RGBInputDescriptor = 0, DepthDescriptor = 0, SegmentationDescriptor = 0;

             if (!fromArchive) //archive samples were prefetched as one record
             {
             RGBInputDescriptor     = signalPrefetchFile(rgbpath);

             if ( (depthAndSegAreMultiplexed) && ((DO_DEPTH) || (DO_SEGMENTATION)) )
             {
               DepthDescriptor = signalPrefetchFile(combinedmdatapath);
             }
             else
             {
               if ( (db->addDepthHeatmap) && (!eraseSample) && (DO_DEPTH) )
                 { DepthDescriptor        = signalPrefetchFile(depthpath); }
               if ( (db->addSegmentationHeatmaps) && (DO_SEGMENTATION) )
                 { SegmentationDescriptor = signalPrefetchFile(segmentationspath); }
             }
             }

             //Read RGB input
             logThreadProgress(thisThreadNumber,1,"rgb_loading");
             img = fromArchive ? pzpdSampleReadImage(db,sampleNumber,&archiveSample,PZPD_STREAM_RGB) :
                                 cachedReadImage(db->files,thisThreadNumber,rgbpath,NO_CODEC); //JPG_CODEC
             logThreadProgress(thisThreadNumber,0,"rgb_loading");
             if ( (img!=0) && (img->pixels!=0) )
             {
             if (makeSureImageHas3Channels(img))
             {
              //Explicit reset of offsets/scale to make sure no problems occur
              //with failed augmentations
              float offsetX = 0.0;
              float offsetY = 0.0;
              float scaleX  = 1.0;
              float scaleY  = 1.0;
              int   originalImageWidth  = img->width;
              int   originalImageHeight = img->height;
              float zoom_factor     = 1.0;
              float max_zoom_factor = db->aug.maxZoomFactor;
              int   pan_x = 0, pan_y = 0;

              int   doPanAndZoom = 1; //Debug switch

              logThreadProgress(thisThreadNumber,1,"pan/zoom");
              if (db->doAugmentations)
              {
               //Augmentation transform stack
               if (pdb->sample[sampleNumber].numberOfSkeletons==0)
               { //High chance to pan & zoom empty skeletons
                if ( (doPanAndZoom) && (eventOccurs(db->aug.chancePanAndZoom)) )
                  {//50 percent chance to pan&zoom
                   randomizeZoomAndPan(img, max_zoom_factor, db->in.width, db->in.height, &zoom_factor, &pan_x, &pan_y);
                  }
               } else
               {
                //We have skeletons!
                if ( (doPanAndZoom) && (eventOccurs(db->aug.chancePanAndZoom)) )
                  {//Chance to pan&zoom if we have persons in scene (special handling to keep them visible)
                   const int MAX_TRIES_TO_CROP_WITH_JOINTS = 3;
                   int tries = 0;
                   int skeletonJointsInCrop = 0;
                   do
                   {//We iterate the randomization to try forcing joints on training samples and not converting a skeleton'ed
                    //sample to a background one (keeping training balance), having a fixed number of tries improves randomization
                    //without adding terrible complexity in randomization or making code unbearabily slow
                    randomizeZoomAndPan(img, max_zoom_factor, db->in.width, db->in.height, &zoom_factor, &pan_x, &pan_y);

                    /*
                    skeletonJointsInCrop = countSkeletonsJointsInHeatmaps(
                                                                             db,sampleNumber,
                                                                             originalImageWidth,originalImageHeight,
                                                                             10,10,
                                                                             zoom_factor,
                                                                             pan_x,pan_y,
                                                                             offsetX,offsetY,
                                                                             scaleX, scaleY );*/

                    skeletonJointsInCrop = ensurePercentageOfJointsInHeatmap( 0.3, //30% joints in image..!
                                                                              db,sampleNumber,
                                                                              originalImageWidth,originalImageHeight,
                                                                              10,10,
                                                                              zoom_factor,
                                                                              pan_x,pan_y,
                                                                              offsetX,offsetY,
                                                                              scaleX, scaleY );


                    tries = tries + 1;
                    } while ( (skeletonJointsInCrop==0) && (tries<MAX_TRIES_TO_CROP_WITH_JOINTS) );

                    if (skeletonJointsInCrop==0)
                    {
                      //If no skeleton in crop cancel pan & zoom
                      zoom_factor=1.0;
                      pan_x = 0;
                      pan_y = 0;
                    }
                  }
               }

              //We perform the pan&zoom we randomized (if event happened)
              panAndZoom8BitImage(img, zoom_factor, pan_x, pan_y, db->in.width, db->in.height);
              } // Only do augmentations if enabled
              logThreadProgress(thisThreadNumber,0,"pan/zoom");



              //Resize perserving aspect ratio and adding black borders outside of image
              logThreadProgress(thisThreadNumber,1,"resize");
              unsigned char* pixelDestination = db->in.pixels + (db->in.width * db->in.height * db->in.channels * targetImageNumber);
              resizeImageWithBorders(img, pixelDestination, db->in.width, db->in.height, &offsetX, &offsetY, &scaleX, &scaleY);
              logThreadProgress(thisThreadNumber,0,"resize");

              offX = offsetX;
              offY = offsetY;
              sclX = scaleX;
              sclY = scaleY;

              if (ENABLE_DENOISING_OUTPUT)
              { //At this point copy the un-augmented image to output
                if (!eraseSample)
                {
                 copyInputRGBFrameToOutput(&db->in,
                                           &db->out8bit,
                                           (unsigned int) offsetX,
                                           (unsigned int) offsetY,
                                           targetImageNumber,
                                           db->denoiseHeatmapIndex
                                          );

                }
              }


              char doLRFlip  = 0;
              int  doRotate90 = 0;  /* 0=none, 1=CW, 2=CCW */

              logThreadProgress(thisThreadNumber,1,"augmentations");
              //More augmentations after resizing input (so we need less calculations) and before generating heatmaps
              if (db->doAugmentations)
              {
               struct Image augmentImage={0};
               db_map_sample_in_to_image((void*) db,&augmentImage,targetImageNumber);


                if (eventOccurs(db->aug.chanceHorizontalFlip))
                {
                    doLRFlip=1; //Decide this here so it is known for functions that need it..
                }

                //99% of time we are here..
                if (eventOccurs(db->aug.chanceBrightnessContrast))
                { //Change Brightness/Contrast of input image

                  float brightnessR,brightnessG,brightnessB;

                  //From the 50% of the 99% :P
                  if (eventOccurs(db->aug.chanceBrightnessContrastUniform))
                  {
                    //50% of the time altering brightness/contrast
                    //we do each channel seperately
                    brightnessR = getRandomFloat(db->aug.minBrightnessChange, db->aug.maxBrightnessChange);
                    brightnessG = getRandomFloat(db->aug.minBrightnessChange, db->aug.maxBrightnessChange);
                    brightnessB = getRandomFloat(db->aug.minBrightnessChange, db->aug.maxBrightnessChange);
                  } else
                  {
                    //50% of the time we alter all channels together (but more drastically :) )
                    brightnessR = getRandomFloat(db->aug.minUniformBrightnessChange, db->aug.maxUniformBrightnessChange); //Slightly favor brightening up the image..
                    brightnessG = brightnessR; //Use same change as R
                    brightnessB = brightnessR; //Use same change as R
                  }

                  adjustBrightnessContrast(&augmentImage,
                                           brightnessR, getRandomFloat(db->aug.minRelContrastChange, db->aug.maxRelContrastChange), // R
                                           brightnessG, getRandomFloat(db->aug.minRelContrastChange, db->aug.maxRelContrastChange), // G
                                           brightnessB, getRandomFloat(db->aug.minRelContrastChange, db->aug.maxRelContrastChange), // B
                                           offsetX,offsetY);
                }

                if (eventOccurs(db->aug.chancePerturbed))
                {   //Add noise on top of existing signal
                   perturbGaussianNoise(&augmentImage, db->aug.perturbationMagnitude, offsetX, offsetY);
                } else
                if (eventOccurs(db->aug.chanceBurnedPixels))
                {
                    //Burn up to 9 pixels on the camera sensor, for a 300x300 input frame this is 1/10000 corruption
                    //rng_next32: avoid glibc rand()'s global lock in the worker hot path;
                    //the >0 guard: maxBurnedPixels is runtime-settable, %0 is UB. (DATALOADER.md A3)
                    int maxBurned = (db->aug.maxBurnedPixels > 0) ? db->aug.maxBurnedPixels : 1;
                    burnedPixels(&augmentImage, rng_next32() % maxBurned, offsetX, offsetY);
                }

                if ( (pdb->sample[sampleNumber].numberOfSkeletons > 0) && (eventOccurs(db->aug.chanceRotate90)) )
                {
                    doRotate90 = (rng_next32() & 1) ? 1 : 2; /* 1=CW, 2=CCW */
                    rotate90(&augmentImage, doRotate90 == 1, (unsigned int)offsetX, (unsigned int)offsetY);
                }

                if (eventOccurs(db->aug.chanceCoarseDropout))
                {
                    coarseDropout(&augmentImage,
                                  getRandomNumber(db->aug.coarseDropoutHolesMin, db->aug.coarseDropoutHolesMax),
                                  db->aug.coarseDropoutMinSize, db->aug.coarseDropoutMaxSize,
                                  db->aug.coarseDropoutMinSize, db->aug.coarseDropoutMaxSize,
                                  offsetX, offsetY);
                }

                if (eventOccurs(db->aug.chanceGaussianBlur))
                {
                    gaussianBlur(&augmentImage,
                                 getRandomFloat(db->aug.gaussianBlurSigmaMin, db->aug.gaussianBlurSigmaMax),
                                 offsetX, offsetY);
                } else
                if (eventOccurs(db->aug.chanceDefocusBlur))
                {
                    defocusBlur(&augmentImage,
                                getRandomNumber(db->aug.defocusBlurRadiusMin, db->aug.defocusBlurRadiusMax),
                                offsetX, offsetY);
                } else
                if (eventOccurs(db->aug.chanceMotionBlur))
                {
                    motionBlur(&augmentImage,
                               getRandomNumber(db->aug.motionBlurLengthMin, db->aug.motionBlurLengthMax),
                               offsetX, offsetY);
                }
              } // Only do augmentations if enabled

           logThreadProgress(thisThreadNumber,0,"augmentations");

           if (ENABLE_DENOISING_OUTPUT)
           {
             if (ENABLE_DENOISING_DIFFERENCE_OUTPUT)
              { //At this point copy the un-augmented image to output
                if (!eraseSample)
                {
                 denoisingDiffInputRGBFrameToOutput(&db->in,
                                           &db->out8bit,
                                           (unsigned int) offsetX,
                                           (unsigned int) offsetY,
                                           targetImageNumber,
                                           db->denoiseHeatmapIndex
                                          );

                }
              }
           }

            //We clean target heatmaps..
           //===========================================================================================================
            //cleanHeatmapsOfTargetSample(db,targetImageNumber);
           //At this point the heatmap for depth map is clean
           //===========================================================================================================

           //Store sample erased flag
           pdb->sample[sampleNumber].eraseEntry = eraseSample;

           //Load Keypoint Heatmaps
           //===========================================================================================================
            if ( (!eraseSample) && (DO_KEYPOINTS) )
            {
              logThreadProgress(thisThreadNumber,1,"keypoints");


              //TODO: Overwrite PAF background with 0 to make it score less..
              if (SET_PAF_HEATMAP_BACKGROUND_TO_ZERO)
              {
                setHeatmapValueForContinuousChannels(
                                                     &db->out8bit,
                                                     targetImageNumber,
                                                     db->PAFHeatmapIndex,                              //Start of PAFs
                                                     db->PAFHeatmapIndex + numberOfPAFJoints - 1,      //End of PAFs (inclusive)
                                                     0
                                                    );
              }


             //Populate Skeleton output, Needs to happen after resize to have offsets!
             populateHeatmaps(
                               db,sampleNumber,targetImageNumber,gradientSize,PAFSize,doLRFlip,
                               //----------
                               originalImageWidth,
                               originalImageHeight,
                               //----------
                               zoom_factor,
                               pan_x,
                               pan_y,
                               //----------
                               offsetX,
                               offsetY,
                               scaleX,
                               scaleY
                              );
              logThreadProgress(thisThreadNumber,0,"keypoints");
            }
           //===========================================================================================================




           //At this point the heatmap for depth map is clean
           //If we want now is the time to load it in!
           //===========================================================================================================
           // DEPTH PIPELINE (one sample). Depth exists twice in the outputs: a 16-bit channel (db->out16bit,
           // db->depthmapHeatmapIndex16Bit, the 16-bit head) and an 8-bit channel (db->out8bit,
           // db->depthmapHeatmapIndex8Bit = "depthmap", which also feeds the normals and depth-level channels).
           // Source convention (Depth Anything relative depth): unsigned 16-bit, LARGER = NEARER, normalised per
           // image so the nearest pixel is ~65535 and the farthest ~0.
           //
           //  STEP 0  read: multiplexed "all" image already split above (native byte order), otherwise the depth
           //          file: the archive "depth" stream, or <depth dir>/<name>.pzp (preferred) / .png. PNG and PZP
           //          both decode 16-bit samples BIG-endian (byte-identical; verified on 300W PZP vs PNG).
           //  STEP 1  panAndZoom16BitImage: the same pan & zoom crop as the RGB input (nearest neighbour, 16-bit).
           //  STEP 2  swap16bitEndianness: big-endian -> native, ONLY for depth files (not for multiplexed depth).
           //  STEP 3  resize16BitImageTo16BitHeatmapWithBorders1Ch: bilinear letterbox resize into the 16-bit
           //          heatmap, content pasted at the same whole-pixel offset as the RGB, value - 32767 -> SIGNED
           //          short (0..65535 -> -32767..32768; an exact 65535 would wrap to -32768, but bilinear
           //          interpolation practically never lands on it). Letterbox borders keep the cleared value -32767
           //          ("farthest"); every content pixel is written (the last content row / column was skipped,
           //          left at -32767, for images shrunk by less than 2x until 2026-10-06).
           //  STEP 4  copy16BitHeatmapTo8BitHeatmapWithRemap: per-image min..max of the 16-bit content ->
           //          linear to SIGNED char [-120,120]: -120 = farthest, +120 = nearest, 0 = mid range.
           //          Borders stay -120. Every image therefore spans the full range and crosses 0 mid-scene.
           //  STEP 5  computeNormalsOnHeatmaps8Bit: Sobel of the 8-bit depth (read as SIGNED) ->
           //          normalX/normalY/normalZ = 120 * (dx, dy, 1) / |(dx, dy, 1)|.
           //  STEP 6  copyThresholdedDepthHeatmap: depth-level channel(s) thresholded from the 8-bit depth
           //          (1 level: +120 where depth is in [64,127], the nearest part of the range, else -120).
           //  Later, after all heatmaps exist:
           //  STEP 7  horizontal flip (if any): 8-bit and 16-bit depth mirrored with the image, normals recomputed
           //          from the mirrored depth (normalX changes sign). rotate90 (if any): depth rotated with the
           //          image, normals recomputed.
           //  STEP 8  recalibrate8BitDepth / recalibrate16BitDepth: re-stretch the content of both depths to the full
           //          range after augmentation. 8-bit: [-120,120], a no-op when STEP 4's result already spans it (the
           //          usual case). 16-bit: ALWAYS stretched to [MINV_16BIT,MAXV_16BIT] = [-32767,32767], i.e. this is
           //          where the 16-bit target gets its per-image normalisation. Normals / depth levels are NOT
           //          recomputed here (they come from the pre-recalibration 8-bit depth).
           if ( (db->addDepthHeatmap) && (!eraseSample) && (DO_DEPTH) )
           {
               //If no file is present in file system just throw an early error message
               //([B5] depthFound comes from the cached resolution — no per-epoch re-stat)
               if ( (depthAndSegAreMultiplexed) || (depthFound) )
               { // START OF LOADING DEPTH MAP

                logThreadProgress(thisThreadNumber,1,"depth_loading");
                //DEPTH STEP 0: read the depth file (multiplexed depth is already in dpth, native byte order)
                if (depthAndSegAreMultiplexed) { /*fprintf(stderr,"TODO: Depth And Segmentations are multiplexed\n");*/  } else
                if (fromArchive)               { dpth = pzpdSampleReadImage(db,sampleNumber,&archiveSample,PZPD_STREAM_DEPTH); } else
                if (depthIsPZP)                { dpth = cachedReadImage(db->files,thisThreadNumber,depthpath,PZP_CODEC); } else
                                               { dpth = cachedReadImage(db->files,thisThreadNumber,depthpath,PNG_CODEC); }
                logThreadProgress(thisThreadNumber,0,"depth_loading");

                if (dpth!=0)
                {//Success
                 logThreadProgress(thisThreadNumber,1,"depth_processing");
                 if (dpth->channels!=1)      { fprintf(stderr,RED "PLEASE CONVERT YOUR DEPTH TO 1 CHANNEL, IT IS %u (%s) \n" NORMAL,dpth->channels,depthpath); abort(); }
                 if (dpth->bitsperpixel!=16) { fprintf(stderr,RED "PLEASE CONVERT YOUR DEPTH TO 16 BIT (%s) \n" NORMAL,depthpath);    abort(); }

                 signed char*  heatmapDestination8Bit  = (signed char*)  db->out8bit.pixels  + (db->out8bit.width  * db->out8bit.height  * db->out8bit.channels  * targetImageNumber);
                 signed short* heatmapDestination16Bit = (signed short*) db->out16bit.pixels + (db->out16bit.width * db->out16bit.height * db->out16bit.channels * targetImageNumber); //* sizeof(signed short)This needs to be the start of the heatmap
                 if (dpth->bitsperpixel==16)
                    {
                      //DEPTH STEP 1: the same pan & zoom crop as the RGB input (no-op when the sample is not zoomed)
                      panAndZoom16BitImage(dpth, zoom_factor, pan_x, pan_y, db->in.width, db->in.height);
                      //DEPTH STEP 2: PNG / PZP depth files decode big-endian and need the swap; depth split out of a
                      //multiplexed ("all") image is already native ((high<<8)|low in
                      //splitSegmentationAndDepthFromSingleFileGeneric), and swapping it scrambled every such sample
                      //into byte-wrapped noise
                      if (!depthAndSegAreMultiplexed)
                        { swap16bitEndianness(dpth); } //swap should happen on reduced resolution image to conserve cycles
                      //DEPTH STEP 3: letterbox resize into the 16-bit heatmap; also converts unsigned -> SIGNED short
                      //(value - 32767), borders stay at the cleared -32767
                      resize16BitImageTo16BitHeatmapWithBorders1Ch(dpth, heatmapDestination16Bit, db->out16bit.width, db->out16bit.height, db->out16bit.channels, db->depthmapHeatmapIndex16Bit ,&offX,&offY,&sclX,&sclY);
                      //DEPTH STEP 4: 16-bit content min..max -> 8-bit signed [-120 farthest .. +120 nearest]
                      copy16BitHeatmapTo8BitHeatmapWithRemap(&db->out8bit,db->depthmapHeatmapIndex8Bit,&db->out16bit,db->depthmapHeatmapIndex16Bit,targetImageNumber,
                                                      (unsigned int) offsetX,
                                                      (unsigned int) offsetY);
                      //-----------------------
                      if ( (db->addNormalHeatmaps) && (DO_NORMALS))
                        {
                          //DEPTH STEP 5: normals from the SIGNED 8-bit depth (Sobel inside the letterbox borders)
                          //computeNormalsOnHeatmaps needs to be called after resizing and is computed on 8bit image
                          computeNormalsOnHeatmaps8Bit(heatmapDestination8Bit,
                                                       db->out8bit.width, db->out8bit.height, db->out8bit.channels,
                                                       db->depthmapHeatmapIndex8Bit, db->normalsHeatmapIndex,
                                                       ctx->gradientX,ctx->gradientY,
                                                      (unsigned int) offsetX,
                                                      (unsigned int) offsetY);
                        }
                      //-----------------------
                      if (db->addDepthLevelsHeatmaps>0)
                      {
                          //DEPTH STEP 6: depth-level channel(s), thresholds of the SIGNED 8-bit depth
                          copyThresholdedDepthHeatmap((signed char *) db->out8bit.pixels,
                                                      (unsigned int) offsetX,
                                                      (unsigned int) offsetY,
                                                      db->out8bit.width,
                                                      db->out8bit.height,
                                                      db->out8bit.channels,
                                                      targetImageNumber,
                                                      db->depthmapHeatmapIndex8Bit,
                                                      db->depthLevelsHeatmapIndex,
                                                      db->addDepthLevelsHeatmaps
                                                      );
                      }
                    }
                     else
                    {
                      fprintf(stderr,RED "Cannot pad and zoom with current configuration (file is %s) \n" NORMAL,depthpath);
                      abort();
                    }

                 logThreadProgress(thisThreadNumber,0,"depth_processing");
                //-----------------------
                destroyImage(dpth);
                dpth = 0;
                } else
                {
                 fprintf(stderr,"Failed reading depthmap %s\n",depthpath);
                 abort();
                }
            } else //Only populate depth heatmap if heatmap is loaded (otherwise it should be empty)
            {
              fprintf(stderr,"Depth file (%s) does not exist for sample %lu \n",depthpath,sampleNumber);
              fprintf(stderr,RED "Terminating to avoid training inconsistency\n" NORMAL);
              abort();
            }
           } //Depthmap switched on
           //===========================================================================================================






           //Populate segmentations
           //===========================================================================================================
           if ( (db->addSegmentationHeatmaps) && (!eraseSample) && (DO_SEGMENTATION) )
           {
            //([B5] segFound comes from the cached resolution — no per-epoch re-stat)
            if ( (depthAndSegAreMultiplexed) || (segFound) )
               { // START OF LOADING DEPTH MAP
                //Try to actually read the segmentation data file..

                logThreadProgress(thisThreadNumber,1,"segmentation_loading");
                if (depthAndSegAreMultiplexed) { /*fprintf(stderr,"TODO: Depth And Segmentations are multiplexed\n");*/             } else
                if (fromArchive)               { segment = pzpdSampleReadImage(db,sampleNumber,&archiveSample,PZPD_STREAM_SEG); } else
                if (segIsPZP)                  { segment = cachedReadImage(db->files,thisThreadNumber,segmentationspath,PZP_CODEC); } else
                                               { segment = cachedReadImage(db->files,thisThreadNumber,segmentationspath,PNG_CODEC); }
                logThreadProgress(thisThreadNumber,0,"segmentation_loading");

                if ( (segment!=0) && ( (segment->channels==3) || (segment->channels==1) ) )
                {
                 logThreadProgress(thisThreadNumber,1,"segmentation_processing");

                 //Success
                 signed char* heatmapDestination8Bit = (signed char*) db->out8bit.pixels + (db->out8bit.width * db->out8bit.height * db->out8bit.channels * targetImageNumber);

                 panAndZoom8BitImage(segment, zoom_factor, pan_x, pan_y, db->in.width, db->in.height);

                 if (segment->channels==1)
                 {
                  resizeInstanceImageWithBorders1Channel(segment,
                                                heatmapDestination8Bit,
                                                db->out8bit.width, db->out8bit.height, db->out8bit.channels,
                                                db->segmentationHeatmapIndex, VALID_SEGMENTATIONS,
                                                &offX,&offY,&sclX,&sclY);
                 } else
                 {
                  resizeInstanceImageWithBorders(segment,
                                                heatmapDestination8Bit,
                                                db->out8bit.width, db->out8bit.height, db->out8bit.channels,
                                                db->segmentationHeatmapIndex, VALID_SEGMENTATIONS,
                                                &offX,&offY,&sclX,&sclY);

                 }

                  //DISABLED INSTANCES 2/2
                  //#if ENABLE_INSTANCE_DATA == 1
                  // rearrangeInstanceCount(heatmapDestination8Bit,db->out8bit.width, db->out8bit.height, db->out8bit.channels,db->segmentationHeatmapIndex+2);
                  //#endif

                  logThreadProgress(thisThreadNumber,0,"segmentation_processing");
                  destroyImage(segment);
                  segment = 0;
                }
                 else
                {
                 fprintf(stderr,"Failed reading segmentation %s\n",segmentationspath);
                 if (segment==NULL)        { fprintf(stderr,"%s did not yield a proper image\n",segmentationspath); } else
                 if (segment->channels!=3) { fprintf(stderr,"Was expecting %u segmentation channels but got %u for %s\n",3,segment->channels,segmentationspath); }
                 abort();
                }
               } else
               {
                 fprintf(stderr,"Failed finding segmentation path %s\n",segmentationspath);
                 abort();
               }
           } //End of adding segmentation heatmaps
           //===========================================================================================================


           //AUGMENTATION_CHANCE_PERCENT_HORIZONTAL_FLIP
           if (doLRFlip)
                {
                 //TODO: Implement flipping here
                 const unsigned int offx = (unsigned int)offsetX;
                 const unsigned int offy = (unsigned int)offsetY;

                 // Flip network input (8-bit RGB) in-place for this sample
                 unsigned char *inBase = db->in.pixels + (db->in.width * db->in.height * db->in.channels * targetImageNumber);
                 flipImageHoriz_8bit(inBase, db->in.width, db->in.height, 0, db->in.channels, offx, offy);

                 // Flip 8-bit heatmaps in-place for this sample
                 unsigned char *out8Base = (unsigned char*)db->out8bit.pixels + (db->out8bit.width * db->out8bit.height * db->out8bit.channels * targetImageNumber);

                 //startChannel used to be 0, now flipping is done inside the joint/PAF code so we start at the
                 //first non-joint channel that needs flipping (Depth) — resolved from the heatmap layout.
                 //The left/right, instance and SuperPoint groups are ALSO drawn at mirrored positions (with
                 //swapped labels) by HeatmapGenerator.c, so they are skipped here: flipping them again
                 //un-mirrored them. Everything else from depth on (depth, normals, levels, denoise,
                 //segmentation) is rendered from unflipped sources and is mirrored here.
                 {
                   const char * preMirrored[] = { "left_right", "instance", "superpoint" };
                   unsigned int c = db->depthmapHeatmapIndex8Bit;
                   while (c < db->out8bit.channels)
                   {
                     unsigned int runEnd = db->out8bit.channels, skipTo = db->out8bit.channels;
                     for (unsigned int g=0; g<sizeof(preMirrored)/sizeof(preMirrored[0]); g++)
                     {
                       unsigned int gs=0, gc=0;
                       if ( heatmapLayoutGetRange(&db->heatmapLayout,preMirrored[g],&gs,&gc) && (gc>0) && (gs+gc>c) && (gs<runEnd) )
                          { runEnd = (gs>c) ? gs : c; skipTo = gs+gc; }
                     }
                     flipImageHoriz_8bitChannels(out8Base, db->out8bit.width, db->out8bit.height, c, runEnd, db->out8bit.channels, offx, offy);
                     c = skipTo;
                   }
                 }

                 // Normals were computed from the unflipped depth: mirroring them leaves the x component
                 // with the wrong sign, so recompute them from the flipped depth (as the rotation path does)
                 if (db->addNormalHeatmaps && DO_NORMALS)
                    {
                      computeNormalsOnHeatmaps8Bit((signed char*)out8Base,
                                                   db->out8bit.width, db->out8bit.height, db->out8bit.channels,
                                                   db->depthmapHeatmapIndex8Bit, db->normalsHeatmapIndex,
                                                   ctx->gradientX, ctx->gradientY,
                                                   offx, offy);
                    }

                 // DEPTH STEP 7 (flip): the 8-bit depth / depth levels were mirrored above with the other unflipped
                 // channels, normals were recomputed from that mirrored depth, and the 16-bit depth is mirrored here.
                 // Flip 16-bit heatmaps in-place for this sample (if present)
                 if (db->out16bit.pixels && db->out16bit.channels>0)
                    {
                      signed short *out16Base = (signed short*)db->out16bit.pixels + (db->out16bit.width * db->out16bit.height * db->out16bit.channels * targetImageNumber);
                      flipImageHoriz_16bit(out16Base, db->out16bit.width, db->out16bit.height, db->out16bit.channels, offx, offy);
                    }
                }
           //-----------------------------------------------------------

           // Rotate all heatmaps to match the ±90° rotated input image
           if (doRotate90 != 0)
                {
                 const int    cw   = (doRotate90 == 1);
                 const unsigned int offx = (unsigned int)offsetX;
                 const unsigned int offy = (unsigned int)offsetY;

                 // Rotate 8-bit heatmaps (all channels: joints, PAFs, depth, normals, segmentation, …)
                 unsigned char *out8Base = (unsigned char*)db->out8bit.pixels
                                         + (db->out8bit.width * db->out8bit.height * db->out8bit.channels * targetImageNumber);
                 rotate90_heatmap_8bit((signed char*)out8Base,
                                       db->out8bit.width, db->out8bit.height, db->out8bit.channels,
                                       cw, offx, offy);

                 // Recompute normals from the now-rotated depth channel.
                 // This is cleaner than a vector fixup and uses the same code path.
                 // Border is 0 after rotation (rotate90 fills the entire frame).
                 if (db->addNormalHeatmaps && DO_NORMALS)
                    {
                      computeNormalsOnHeatmaps8Bit((signed char*)out8Base,
                                                   db->out8bit.width, db->out8bit.height, db->out8bit.channels,
                                                   db->depthmapHeatmapIndex8Bit, db->normalsHeatmapIndex,
                                                   ctx->gradientX, ctx->gradientY,
                                                   0, 0);
                    }

                 // A ±90° rotation swaps width<->height: rotate90_heatmap_8bit already
                 // reoriented each size *tent* spatially, but the w/h tents now live in the
                 // wrong channels, so swap the instance width/height channel contents per
                 // pixel to match the new orientation (mirrors the normals recompute above;
                 // the center channel is orientation-agnostic and needs no fixup).
                 if (db->addInstanceHeatmaps)
                    {
                      const unsigned int wCh = db->instanceHeatmapIndex + 1;
                      const unsigned int hCh = db->instanceHeatmapIndex + 2;
                      const unsigned int ch  = db->out8bit.channels;
                      signed char * px = (signed char*)out8Base;
                      unsigned int yy,xx;
                      for (yy=0; yy<db->out8bit.height; yy++)
                        {
                          for (xx=0; xx<db->out8bit.width; xx++)
                            {
                              signed char * base = px + ((yy*db->out8bit.width*ch) + (xx*ch));
                              signed char tmp = base[wCh];
                              base[wCh] = base[hCh];
                              base[hCh] = tmp;
                            }
                        }
                    }

                 // DEPTH STEP 7 (rotate90): the 8-bit depth was rotated above with all 8-bit channels and the normals
                 // recomputed from it; the 16-bit depth is rotated here the same way.
                 // Rotate 16-bit heatmaps (depth) if present
                 if (db->out16bit.pixels && db->out16bit.channels > 0)
                    {
                      signed short *out16Base = (signed short*)db->out16bit.pixels
                                              + (db->out16bit.width * db->out16bit.height * db->out16bit.channels * targetImageNumber);
                      rotate90_heatmap_16bit(out16Base,
                                             db->out16bit.width, db->out16bit.height, db->out16bit.channels,
                                             cw, offx, offy);
                    }

                 // The denoising channels (34-36) were written from db->in before this rotation
                 // ran, so rotate90_heatmap_8bit just double-rotated them (and the diff version
                 // would read the now-corrupt existing values as a baseline, producing garbage).
                 // Fix: overwrite the entire frame (borderX=0, borderY=0 — rotation eliminated
                 // all borders) with a plain copy of the correctly-rotated db->in pixels.
                 if (ENABLE_DENOISING_OUTPUT && !eraseSample)
                    {
                      copyInputRGBFrameToOutput(&db->in, &db->out8bit, 0, 0, targetImageNumber, db->denoiseHeatmapIndex);
                    }
                }
           //-----------------------------------------------------------

           // DEPTH STEP 8: post-augmentation depth recalibration: stretch both heatmaps to their full range
           // (see the DEPTH PIPELINE overview: the 16-bit stretch always runs, the 8-bit one only when needed).
           // After a rotate90 there are no letterbox borders any more (bx = by = 0).
           if ( (db->addDepthHeatmap) && (!eraseSample) && (DO_DEPTH) )
                {
                  const unsigned int bx = (doRotate90 != 0) ? 0u : (unsigned int)offsetX;
                  const unsigned int by = (doRotate90 != 0) ? 0u : (unsigned int)offsetY;
                  recalibrate8BitDepth (&db->out8bit,  db->depthmapHeatmapIndex8Bit,  targetImageNumber, bx, by);
                  recalibrate16BitDepth(&db->out16bit, db->depthmapHeatmapIndex16Bit, targetImageNumber, bx, by);
                }

           //-----------------------------------------------------------
           // Geolocation: a GLOBAL lat/lon density channel. It is world-aligned,
           // NOT image-aligned, so it is written LAST — after flip/rotate90 — and
           // copied verbatim with NO pan/zoom/flip/rotate (a re-framed photo is taken
           // at the same place). The 128x64 density PZP is nearest-upsampled to the
           // 256x256 output channel (x2 in x, x4 in y). See GEOLOCATION.md.
           if ( (db->addGeolocationHeatmap) && (!eraseSample) )
                {
                 signed char* out8Base = (signed char*) db->out8bit.pixels
                                       + (db->out8bit.width * db->out8bit.height * db->out8bit.channels * targetImageNumber);
                 const unsigned int W = db->out8bit.width, H = db->out8bit.height, CH = db->out8bit.channels;
                 const unsigned int geoCh = (unsigned int) db->geolocationHeatmapIndex;
                 unsigned int oy,ox;
                 int wrote = 0;

                 if (files.geoFound)
                    {
                      struct Image * geo = cachedReadImage(db->files,thisThreadNumber,files.geolocation,PZP_CODEC);
                      if (geo!=0)
                         {
                           if ( (geo->channels==1) && (geo->bitsperpixel==16) )
                              {
                                const unsigned short * g = (const unsigned short*) geo->pixels;
                                const unsigned int gW = geo->width, gH = geo->height;
                                swap16bitEndianness(geo); //PZP stores the PGM big-endian; make it native
                                for (oy=0; oy<H; oy++)
                                   {
                                     unsigned int sy = (oy*gH)/H;
                                     for (ox=0; ox<W; ox++)
                                        {
                                          unsigned int sx = (ox*gW)/W;                 //nearest upsample
                                          int u16 = (int) g[sy*gW + sx];
                                          int v   = MINV + (u16*MAXV_MINUS_MMINV)/65535; //[0,65535] -> [MINV,MAXV]
                                          out8Base[(oy*W + ox)*CH + geoCh] = (signed char) v;
                                        }
                                   }
                                wrote = 1;
                              } else
                              {
                                fprintf(stderr,RED "Geolocation PZP %s must be 1ch/16bit (got %uch/%ubit)\n" NORMAL,
                                        files.geolocation, geo->channels, geo->bitsperpixel);
                              }
                           destroyImage(geo);
                         }
                    }

                 if (!wrote)
                    { //Absent/unreadable: fill with background so no rotated/stale bytes remain.
                      for (oy=0; oy<H; oy++)
                        for (ox=0; ox<W; ox++)
                          out8Base[(oy*W + ox)*CH + geoCh] = (signed char) MINV;
                    }
                }
           //-----------------------------------------------------------

            } else // Only populate if sample has 3 channels (RGB)
            {
              fprintf(stderr,"Ommitted sample %lu (%s) with %u channels\n",sampleNumber,rgbpath,img->channels);
              fprintf(stderr,RED "Terminating to avoid training inconsistency\n" NORMAL);
              abort();
            }

             destroyImage(img);
             img = 0;
           } else//Only populate sample if input is populated
           {
              fprintf(stderr,"Ommitted sample %lu (%s) that didn't load\n",sampleNumber,rgbpath);
              fprintf(stderr,RED "Terminating to avoid training inconsistency\n" NORMAL);
              abort();
           }

            //Simpler code with less state checking to free any acquired file descriptors
            freeFileDescriptor(RGBInputDescriptor);
            RGBInputDescriptor = 0;
            freeFileDescriptor(DepthDescriptor);
            DepthDescriptor=0;
            freeFileDescriptor(SegmentationDescriptor);
            SegmentationDescriptor =0;

           } //<-- Not an empty/erased sample so we should actually read images etc.

           // Catch-all: the multiplexed split allocates dpth+segment BEFORE the erase/normal
           // branch, but they are only destroyed inside the depth/segmentation blocks, which
           // are skipped for erased samples (and when their add* flags are off). Both pointers
           // are nulled where they are consumed, so this is a no-op on the normal path.
           if (dpth    != 0) { destroyImage(dpth);    dpth    = 0; }
           if (segment != 0) { destroyImage(segment); segment = 0; }

           //Every decode of this sample is done: give its blobs back to the prefetcher
           if (fromArchive) { pzpdSampleEnd(db,&archiveSample); }

           //checkCanary("End of thread check B",db->canaryB.shouldRemainUntouched,CANARY_SIZE);
           //checkCanary("End of thread check C",db->canaryB.shouldRemainUntouched,CANARY_SIZE);
           ctx->fullfilledWork += 1;

           }//Valid iteration

         }//Our threads job..!

        }

      } //We have allocated memory to do something
       else
      {
          fprintf(stderr,RED "Not activating thread because of unallocated memory\n" NORMAL);
      }
        //--------------------------------
        //fprintf(stderr,GREEN "Thread %u: Finished task..\n" NORMAL,ctx->thisThreadNumber);
        //--------------------------------

        logThreadProgress(thisThreadNumber,1,"thread_sync");
        startTime = GetTickCountMicrosecondsMN();
        ctx->computationOutput = 1;
        threadpoolWorkerLoopEnd(ptr);

        ctx->lagToFinishWorkMicroseconds = GetTickCountMicrosecondsMN() - startTime;
        //fprintf(stderr,GREEN "Thread %u: Looping..\n" NORMAL,ctx->thisThreadNumber);
        startTime = GetTickCountMicrosecondsMN();
        logThreadProgress(thisThreadNumber,0,"thread_sync");

        //fprintf(stderr,GREEN "Thread %u: Lag to start microseconds: %lu  / Lag to end microseconds: %lu..\n" NORMAL,ctx->thisThreadNumber,ctx->lagToStartWorkMicroseconds,ctx->lagToFinishWorkMicroseconds);
    } // end of thread

    fprintf(stderr,GREEN "Thread %u: Deallocating gradients after finishing..\n" NORMAL,ctx->thisThreadNumber);

    //Free up temporary buffers
    #if INTEL_OPTIMIZATIONS
     if (ctx->gradientX!=0)
       { _mm_free(ctx->gradientX); ctx->gradientX=0; }
     if (ctx->gradientY!=0)
       { _mm_free(ctx->gradientY); ctx->gradientY=0; }
    #else
     if (ctx->gradientX!=0)
       { free(ctx->gradientX); ctx->gradientX=0; }
     if (ctx->gradientY!=0)
       { free(ctx->gradientY); ctx->gradientY=0; }
    //if (inputRGBBuffer!=0)
    //   { free(inputRGBBuffer); }
    #endif

    pzp_thread_cleanup();  // release per-thread ZSTD decompression context

    return 0;
}
