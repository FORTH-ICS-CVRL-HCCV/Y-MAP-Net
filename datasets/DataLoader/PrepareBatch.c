#include <time.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "pthreadWorkerPool.h"
#include "DataLoader.h"

#include "codecs/codecs.h"
#include "codecs/image.h"

#include "cache.h"

#include "processing/resize.h"
#include "processing/augmentations.h"
#include "processing/normals.h"
#include "processing/draw.h"
#include "processing/conversions.h"
#include "processing/bilateral.h"
#include "processing/flip.h"

#include "DBLoader.h"
#include "HeatmapGenerator.h"

#include "PrepareBatch.h"
#include "tools.h"

#include <fcntl.h>

/*
void initializeHeatmap16BitSlow(signed short* heatmapDestination16Bit, int width, int height, int channels)
{
    size_t totalElements = (size_t)width * height * channels;
    for (size_t i = 0; i < totalElements; ++i)
    {
        heatmapDestination16Bit[i] = MINV_16BIT;
    }
}
*/

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
  signed char*  heatmapDestination8Bit  = (signed char*)  db->out8bit.pixels  + (db->out8bit.width  * db->out8bit.height  * db->out8bit.channels  * targetImageNumber);
  signed short* heatmapDestination16Bit = (signed short*) db->out16bit.pixels + (db->out16bit.width * db->out16bit.height * db->out16bit.channels * targetImageNumber);

  //Nuke all 16bit heatmap data for our sample, memset doesn't work for 16bit data
  initializeHeatmap16Bit(heatmapDestination16Bit, db->out16bit.width, db->out16bit.height, db->out16bit.channels);

  memset(heatmapDestination8Bit, (signed char) MINV,         db->out8bit.width  * db->out8bit.height  * db->out8bit.channels  * sizeof (signed char));
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
       fprintf(stderr,"Bug?: freeFileDescriptor(0);\n");
       return;
   }
   if (fd!=-1)
      {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
      }
}



void resolvePathToRequestedFiles(const char * pathToCOCOImages, const char * pathToCOCODepthMaps, const char * pathToCOCOSegmentations, const char * thisFilename,
                                 int limit,char* tmp,char * pathToRGBFile,int *rgbIsPZP, char * pathToDepthFile,int * depthIsPZP,char * pathToSegmentationFile, int * segIsPZP)
{
  //Resolve RGB Input image path
  snprintf(pathToRGBFile,limit,"%.1023s/%.1023s",pathToCOCOImages,thisFilename);

  //Resolve Depth Input image path
  snprintf(tmp,limit/2,"%s",thisFilename);
  if (strlen(tmp)>4)
            {
               tmp[strlen(tmp)-4]=0; //quick & dirty remove extension

               //Depth Anything uses %s_depth.png  / Depth Anything V2 uses %s.png
               int depthFilePathFound = 1;

               //Try first to open the Depth Anything V2 file..
               *depthIsPZP = 1;
               snprintf(pathToDepthFile,limit,"%.1023s/%.1000s.pzp",pathToCOCODepthMaps,tmp);
               if (!fileExists(pathToDepthFile))
               {
                *depthIsPZP = 0; //Not a pzp file any more
                //Try first to open the Depth Anything V2 file..
                snprintf(pathToDepthFile,limit,"%.1023s/%.1000s.png",pathToCOCODepthMaps,tmp);
                if (!fileExists(pathToDepthFile))
                {
                 //If we fail, fall back to open the Depth Anything V1 file..
                 snprintf(pathToDepthFile,limit,"%.1023s/%.1000s_depth.png",pathToCOCODepthMaps,tmp);
                 if (!fileExists(pathToDepthFile))
                 {
                   depthFilePathFound = 0; //Could not find depth file..
                 }
                }
               }
            }

  //Resolve Segmentatioin File Input image path
  snprintf(tmp,limit/2,"%s",thisFilename);
            if (strlen(tmp)>4)
            {
               tmp[strlen(tmp)-4]=0; //quick & dirty remove extension
               int segmentFilePathFound = 1;

               *segIsPZP = 1;
               snprintf(pathToSegmentationFile,limit,"%.1023s/%.1000s.pzp",pathToCOCOSegmentations,tmp);
               if (!fileExists(pathToSegmentationFile))
               {
               *segIsPZP = 0;
                //Try first to open the Depth Anything V2 file..
                snprintf(pathToSegmentationFile,limit,"%.1023s/%.1000s.png",pathToCOCOSegmentations,tmp);
                if (!fileExists(pathToSegmentationFile))
                   { segmentFilePathFound = 0; }
               }
            }

  return ;
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
    char rgbpath[2049]           = {0};
    char depthpath[2049]         = {0};
    char segmentationspath[2049] = {0};
    char tmp[2049]               = {0};

    fprintf(stderr,"Preloading %lu samples\n ",db->numberOfSamples);
    for (iteration=0; iteration<db->numberOfSamples; iteration++)
        {
          if (db->numberOfSamples>iteration)
           {//Check if iteration is valid..
            unsigned long sampleNumber = iteration;

            if (db->indices!=0)
                { sampleNumber = db->indices[iteration]; }

           if (sampleNumber >= db->pdb->numberOfSamples)
           {
               fprintf(stderr,RED "Out of memory access for sample %lu / iteration %lu / limit %lu \n" NORMAL,sampleNumber,iteration,db->in.numberOfImages);
               fprintf(stderr,"Asked to access sample %lu / %lu \n",sampleNumber,db->pdb->numberOfSamples);
               fprintf(stderr,"This is probably index corruption\n");
               abort();
            }

            //This is just needed to get the correct paths to the files
            DatasetSourceID sourceID = db_resolve_sample_sourceID((void *) db,sampleNumber);

            if (sourceID>=db->dbSources->numberOfSources)
            {
               fprintf(stderr,"\n\n\nOut of memory access for sample %lu / iteration %lu / limit %lu \n",sampleNumber,iteration,db->in.numberOfImages);
               fprintf(stderr,"Asked to access source %lu / %u \n",sourceID,db->dbSources->numberOfSources);
               abort();
            }


            const char * pathToCOCOImages        = db->dbSources->source[sourceID].pathToCOCOImages;
            const char * pathToCOCODepthMaps     = db->dbSources->source[sourceID].pathToCOCODepthMaps;
            const char * pathToCOCOSegmentations = db->dbSources->source[sourceID].pathToCOCOSegmentations;
            const char * thisFilename            = db->pdb->sample[sampleNumber].imagePath;


            //Use the new loader if needed
            int rgbIsPZP   = 0;
            int depthIsPZP = 0;
            int segIsPZP   = 0;

            //Resolve all paths..
            resolvePathToRequestedFiles(
                                        db->dbSources->source[sourceID].pathToCOCOImages,
                                        db->dbSources->source[sourceID].pathToCOCODepthMaps,
                                        db->dbSources->source[sourceID].pathToCOCOSegmentations,
                                        db->pdb->sample[sampleNumber].imagePath,
                                        2048,tmp,rgbpath,&rgbIsPZP,depthpath,&depthIsPZP,segmentationspath,&segIsPZP);


            unsigned long hash = 0;
            unsigned long filenameSize=0;
            //------------------------------------------------
            hash = hash_filename(rgbpath);
            cache_add(db->files,rgbpath,hash,&filenameSize,iteration%9 == 0);
            //------------------------------------------------
            hash = hash_filename(depthpath);
            cache_add(db->files,depthpath,hash,&filenameSize,0);
            //------------------------------------------------
            hash = hash_filename(segmentationspath);
            cache_add(db->files,segmentationspath,hash,&filenameSize,0);
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
      char rgbpath[2049]               = {0};
      char depthpath[2049]             = {0};
      char segmentationspath[2049]     = {0};
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
            sampleNumber = db->indices[iteration];

           if (sampleNumber >= pdb->numberOfSamples)
           {
               fprintf(stderr,RED "Out of memory access for sample %lu / iteration %lu / limit %lu \n" NORMAL,sampleNumber,iteration,db->in.numberOfImages);
               fprintf(stderr,"Asked to access sample %lu / %lu \n",sampleNumber,pdb->numberOfSamples);
               fprintf(stderr,"This is probably index corruption\n");
               abort();
            }

            //This is just needed to get the correct paths to the files
            DatasetSourceID sourceID = db_resolve_sample_sourceID((void *) db,sampleNumber);

            if (sourceID>=db->dbSources->numberOfSources)
            {
               fprintf(stderr,"\n\n\nOut of memory access for sample %lu / iteration %lu / limit %lu \n",sampleNumber,iteration,db->in.numberOfImages);
               fprintf(stderr,"Asked to access source %lu / %u \n",sourceID,db->dbSources->numberOfSources);
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

            const char * pathToCOCOImages        = db->dbSources->source[sourceID].pathToCOCOImages;
            const char * pathToCOCODepthMaps     = db->dbSources->source[sourceID].pathToCOCODepthMaps;
            const char * pathToCOCOSegmentations = db->dbSources->source[sourceID].pathToCOCOSegmentations;
            const char * thisFilename            = pdb->sample[sampleNumber].imagePath;


            //Use the new loader if needed
            int rgbIsPZP   = 0;
            int depthIsPZP = 0;
            int segIsPZP   = 0;
            int depthAndSegAreMultiplexed = 0;

            //Resolve all paths..
            resolvePathToRequestedFiles(
                                        db->dbSources->source[sourceID].pathToCOCOImages,
                                        db->dbSources->source[sourceID].pathToCOCODepthMaps,
                                        db->dbSources->source[sourceID].pathToCOCOSegmentations,
                                        pdb->sample[sampleNumber].imagePath,
                                        2048,tmp,rgbpath,&rgbIsPZP,depthpath,&depthIsPZP,segmentationspath,&segIsPZP);

            float progress = 0.0;
            if (sampleEnd-sampleStart>0) { progress = 100.0 * ((float) (iteration-sampleStart) / (sampleEnd-sampleStart)); }
            if (iteration%100==thisThreadNumber)
              { fprintf(stderr,"\r  Thread %u / %0.2f%% (%s) / %u skeletons       \r",thisThreadNumber,progress,pdb->sample[sampleNumber].imagePath,pdb->sample[sampleNumber].numberOfSkeletons); }


            //###############################################################
            //         New combined Depth and Segmentation loading...
            //###############################################################
            logThreadProgress(thisThreadNumber,1,"combined_depth_segmentation_loading");
            char allFile[2048]={0};
            char filenameNoExtension[1024]={0};
            snprintf(filenameNoExtension,1024,"%s",db->pdb->sample[sampleNumber].imagePath);
            filenameNoExtension[strlen(filenameNoExtension)-4]=0;
            snprintf(allFile,2048,"%.1023s/%.1000s.png",db->dbSources->source[sourceID].pathToCombinedMetaData,filenameNoExtension);
            splitSegmentationAndDepthFromSingleFile(allFile,&segment,&dpth);
            depthAndSegAreMultiplexed = 1;
            logThreadProgress(thisThreadNumber,0,"combined_depth_segmentation_loading");
            //###############################################################
            //###############################################################

            //First of all we clean target heatmaps..
           //===========================================================================================================
            cleanHeatmapsOfTargetSample(db,targetImageNumber);
           //At this point the heatmap for depth map is clean
           //===========================================================================================================

            int eraseSample = 0;

            //Completely destroy 1% of background images (they dont have skeletons) with total noise
            if ( (pdb->sample[sampleNumber].numberOfSkeletons==0) && (eventOccurs(AUGMENTATION_CHANCE_PERCENT_DESTROY)) )
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
             int RGBInputDescriptor = 0, DepthDescriptor = 0, SegmentationDescriptor = 0;

             RGBInputDescriptor     = signalPrefetchFile(rgbpath);

             if (!depthAndSegAreMultiplexed)
             {
               if ( (db->addDepthHeatmap) && (!eraseSample) && (DO_DEPTH) )
                 { DepthDescriptor        = signalPrefetchFile(depthpath); }
               if ( (db->addSegmentationHeatmaps) && (DO_SEGMENTATION) )
                 { SegmentationDescriptor = signalPrefetchFile(segmentationspath); }
             }

             //Read RGB input
             logThreadProgress(thisThreadNumber,1,"rgb_loading");
             img = cachedReadImage(db->files,thisThreadNumber,rgbpath,NO_CODEC); //JPG_CODEC
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
              float max_zoom_factor = MAXIMUM_REL_ZOOM_FACTOR;
              int   pan_x = 0, pan_y = 0;

              int   doPanAndZoom = 1; //Debug switch

              logThreadProgress(thisThreadNumber,1,"pan/zoom");
              if (db->doAugmentations)
              {
               //Augmentation transform stack
               if (pdb->sample[sampleNumber].numberOfSkeletons==0)
               { //High chance to pan & zoom empty skeletons
                if ( (doPanAndZoom) && (eventOccurs(AUGMENTATION_CHANCE_PERCENT_PAN_AND_ZOOM)) )
                  {//50 percent chance to pan&zoom
                   randomizeZoomAndPan(img, max_zoom_factor, db->in.width, db->in.height, &zoom_factor, &pan_x, &pan_y);
                  }
               } else
               {
                //We have skeletons!
                if ( (doPanAndZoom) && (eventOccurs(AUGMENTATION_CHANCE_PERCENT_PAN_AND_ZOOM)) )
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
                                           DENOISING_OUTPUT_HEATMAP_START
                                          );

                }
              }


              char doLRFlip = 0;

              logThreadProgress(thisThreadNumber,1,"augmentations");
              //More augmentations after resizing input (so we need less calculations) and before generating heatmaps
              if (db->doAugmentations)
              {
               struct Image augmentImage={0};
               db_map_sample_in_to_image((void*) db,&augmentImage,targetImageNumber);


                if (eventOccurs(AUGMENTATION_CHANCE_PERCENT_HORIZONTAL_FLIP))
                {
                    doLRFlip=1; //Decide this here so it is known for functions that need it..
                }

                //99% of time we are here..
                if (eventOccurs(AUGMENTATION_CHANCE_PERCENT_BRIGHTNESS_CONTRAST))
                { //Change Brightness/Contrast of input image

                  float brightnessR,brightnessG,brightnessB;

                  //From the 50% of the 99% :P
                  if (eventOccurs(AUGMENTATION_CHANCE_PERCENT_BRIGHTNESS_CONTRAST_UNIFORM))
                  {
                    //50% of the time altering brightness/contrast
                    //we do each channel seperately
                    brightnessR = getRandomFloat(MINIMUM_BRIGHTNESS_CHANGE,MAXIMUM_BRIGHTNESS_CHANGE);
                    brightnessG = getRandomFloat(MINIMUM_BRIGHTNESS_CHANGE,MAXIMUM_BRIGHTNESS_CHANGE);
                    brightnessB = getRandomFloat(MINIMUM_BRIGHTNESS_CHANGE,MAXIMUM_BRIGHTNESS_CHANGE);
                  } else
                  {
                    //50% of the time we alter all channels together (but more drastically :) )
                    brightnessR = getRandomFloat(MINIMUM_UNIFORM_BRIGHTNESS_CHANGE, MAXIMUM_UNIFORM_BRIGHTNESS_CHANGE); //Slightly favor brightening up the image..
                    brightnessG = brightnessR; //Use same change as R
                    brightnessB = brightnessR; //Use same change as R
                  }

                  adjustBrightnessContrast(&augmentImage,
                                           brightnessR, getRandomFloat(MINIMUM_REL_CONTRAST_CHANGE,MAXIMUM_REL_CONTRAST_CHANGE), // R
                                           brightnessG, getRandomFloat(MINIMUM_REL_CONTRAST_CHANGE,MAXIMUM_REL_CONTRAST_CHANGE), // G
                                           brightnessB, getRandomFloat(MINIMUM_REL_CONTRAST_CHANGE,MAXIMUM_REL_CONTRAST_CHANGE), // B
                                           offsetX,offsetY);
                }

                if (eventOccurs(AUGMENTATION_CHANCE_PERCENT_PERTURBED))
                {   //Add noise on top of existing signal
                   perturbGaussianNoise(&augmentImage, PERTURBATION_MAGNITUDE ,offsetX,offsetY);
                } else
                if (eventOccurs(AUGMENTATION_CHANCE_PERCENT_BURNED_PIXELS))
                {
                    //Burn up to 9 pixels on the camera sensor, for a 300x300 input frame this is 1/10000 corruption
                    burnedPixels(&augmentImage,rand()%MAXIMUM_BURNED_PIXELS, offsetX, offsetY);
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
                                           DENOISING_OUTPUT_HEATMAP_START
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
                                                     START_OF_PAF_HEATMAPS, //Start of PAFs
                                                     END_OF_PAF_HEATMAPS, //End of PAFs
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
           if ( (db->addDepthHeatmap) && (!eraseSample) && (DO_DEPTH) )
           {
               //If no file is present in file system just throw an early error message
               if ( (depthAndSegAreMultiplexed) || (fileExists(depthpath)) )
               { // START OF LOADING DEPTH MAP

                logThreadProgress(thisThreadNumber,1,"depth_loading");
                //Try to actually read the depth file..
                if (depthAndSegAreMultiplexed) { /*fprintf(stderr,"TODO: Depth And Segmentations are multiplexed\n");*/  } else
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
                      panAndZoom16BitImage(dpth, zoom_factor, pan_x, pan_y, db->in.width, db->in.height);
                      swap16bitEndianness(dpth); //swap should happen on reduced resolution image to conserve cycles
                      //resize also converts to SIGNED short
                      resize16BitImageTo16BitHeatmapWithBorders1Ch(dpth, heatmapDestination16Bit, db->out16bit.width, db->out16bit.height, db->out16bit.channels, db->depthmapHeatmapIndex16Bit ,&offX,&offY,&sclX,&sclY);
                      //copy16BitHeatmapTo8BitHeatmap
                      copy16BitHeatmapTo8BitHeatmapWithRemap(&db->out8bit,db->depthmapHeatmapIndex8Bit,&db->out16bit,db->depthmapHeatmapIndex16Bit,targetImageNumber,
                                                      (unsigned int) offsetX,
                                                      (unsigned int) offsetY);
                      //-----------------------
                      if ( (db->addNormalHeatmaps) && (DO_NORMALS))
                        {
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
                          //Depth Levels
                          copyThresholdedDepthHeatmap((signed char *) db->out8bit.pixels,
                                                      (unsigned int) offsetX,
                                                      (unsigned int) offsetY,
                                                      db->out8bit.width,
                                                      db->out8bit.height,
                                                      db->out8bit.channels,
                                                      targetImageNumber,
                                                      db->depthmapHeatmapIndex8Bit,
                                                      DEPTH_LEVELS_HEATMAP_START,
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
            if ( (depthAndSegAreMultiplexed) || (fileExists(segmentationspath)) )
               { // START OF LOADING DEPTH MAP
                //Try to actually read the segmentation data file..

                logThreadProgress(thisThreadNumber,1,"segmentation_loading");
                if (depthAndSegAreMultiplexed) { /*fprintf(stderr,"TODO: Depth And Segmentations are multiplexed\n");*/             } else
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

                 //startChannel used to be 0, now flipping is done inside the joint/PAF code so we start at heatmap #29 (Depth) see DataLoader.h
                 unsigned int startChannel = START_OF_NON_JOINT_HEATMAPS_THAT_NEED_FLIPPING;
                 flipImageHoriz_8bit(out8Base, db->out8bit.width, db->out8bit.height, startChannel, db->out8bit.channels, offx, offy);

                 // Flip 16-bit heatmaps in-place for this sample (if present)
                 if (db->out16bit.pixels && db->out16bit.channels>0)
                    {
                      signed short *out16Base = (signed short*)db->out16bit.pixels + (db->out16bit.width * db->out16bit.height * db->out16bit.channels * targetImageNumber);
                      flipImageHoriz_16bit(out16Base, db->out16bit.width, db->out16bit.height, db->out16bit.channels, offx, offy);
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

            //Free acquired file descriptors
            freeFileDescriptor(RGBInputDescriptor);


            if (!depthAndSegAreMultiplexed)
            {
             if ( (db->addDepthHeatmap) && (!eraseSample) && (DO_DEPTH) )
               { freeFileDescriptor(DepthDescriptor); DepthDescriptor=0;}
             if ( (db->addSegmentationHeatmaps) && (DO_SEGMENTATION) )
               { freeFileDescriptor(SegmentationDescriptor); SegmentationDescriptor =0; }
            }

           } //<-- Not an empty/erased sample so we should actually read images etc.

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

    return 0;
}
