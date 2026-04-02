#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DBLoader.h"
#include "DataLoader.h"

#define MAX_LINE_LENGTH 1000

// Parse one skeleton line from the .db file into pdb->sample[sampleNumber].sk[skID].
// The line format is: SK<id>,<bboxX>,<bboxY>,<bboxW>,<bboxH>,<coord0>,<coord1>,...
// The first 5 comma-separated fields are the bbox; everything after is keypoint coords.
// The file stores skeleton IDs 1-based; skID passed in is 0-based — we adjust for that.
int parseSkeleton(struct PoseDatabase * pdb,unsigned long sampleNumber,int skID,char *line)
{
  if ( (pdb!=0) && (line!=0) && (sampleNumber<pdb->numberOfSamples) && (pdb->sample[sampleNumber].sk!=0) )
  {
    struct Skeleton * sk = &pdb->sample[sampleNumber].sk[skID];

    // Read the 1-based skeleton ID and bounding box from the line prefix.
    unsigned short skIDFromFile;
    sscanf(line, "SK%hu,%hu,%hu,%hu,%hu,", &skIDFromFile, &sk->bboxX, &sk->bboxY, &sk->bboxW, &sk->bboxH);

    // File stores IDs starting at 1; convert to 0-based before comparing.
    skIDFromFile = skIDFromFile - 1;
    if (skID != (int)skIDFromFile)
    {
        fprintf(stderr,"Inconsistency in sample %lu / expected skeleton %d / found %hu \n",sampleNumber,skID,skIDFromFile);
    }

    // Walk the comma-separated fields; skip the first 5 (bbox fields already parsed above).
    // coords[] has room for (1 + MAX_KEYPOINT_NUMBER) * 3 entries; stop writing before
    // the end to avoid a buffer overrun on a corrupt or oversized file line.
    const int BBOX_FIELD_COUNT  = 5;
    const int MAX_COORD_FIELDS  = (1 + MAX_KEYPOINT_NUMBER) * 3;
    int fieldIndex = 0;
    char *rest  = NULL;
    char *token;
    for (
         token = strtok_r(line, ",", &rest);
         token != NULL;
         token = strtok_r(NULL, ",", &rest)
        )
        {
           if (fieldIndex >= BBOX_FIELD_COUNT)
            {
              int coordIndex = fieldIndex - BBOX_FIELD_COUNT;
              // Guard: stop if the file provides more coordinates than the struct can hold.
              if (coordIndex >= MAX_COORD_FIELDS)
              {
                  fprintf(stderr,"parseSkeleton: too many coordinate fields in sample %lu (max %d), truncating\n",
                          sampleNumber, MAX_COORD_FIELDS);
                  break;
              }
              sk->coords[coordIndex] = (unsigned short) atoi(token);
            }
           fieldIndex++;
        }

    return 1;
  }
  return 0;
}


// Quick scan of the .db file header to retrieve the sample count without
// loading the full database.  Returns 0 on any parse or I/O failure.
unsigned long fastReadDatabaseNumberOfSamples(const char* path)
{
    unsigned long numberOfSamples = 0;
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr,"Error opening pose database file, Fast Read Samples (%s).\n",path);
        return 0;
    }

    // 'res' is checked after every fscanf — a truncated or malformed file
    // returns 0 instead of silently leaving numberOfSamples uninitialized.
    int res;
    char HEADER[5]={0};
    res = fscanf(fp,"%4s\n",HEADER);
    if (res != 1) { fclose(fp); return 0; }
    res = fscanf(fp,"%lu\n", &numberOfSamples);
    if (res != 1) { fclose(fp); return 0; }
    fclose(fp);
   return numberOfSamples;
}



// Quick scan of the .db file header to retrieve the keypoint count per skeleton
// without loading the full database.  Returns 0 on any parse or I/O failure.
unsigned short fastReadDatabaseNumberOfKeypointsPerSample(const char* path)
{
    unsigned long  numberOfSamples    = 0;
    unsigned short keypointsPerSample = 0;
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr,"Error opening pose database file, Fast Read Keypoint Number (%s).\n",path);
        return 0;
    }

    // 'res' is checked after every fscanf — returns 0 on truncated/malformed input.
    int res;
    char HEADER[5]={0};
    res = fscanf(fp,"%4s\n",HEADER);
    if (res != 1) { fclose(fp); return 0; }
    res = fscanf(fp,"%lu\n", &numberOfSamples);
    if (res != 1) { fclose(fp); return 0; }
    res = fscanf(fp,"%hu\n", &keypointsPerSample);
    if (res != 1) { fclose(fp); return 0; }
    fclose(fp);
   return keypointsPerSample;
}

struct PoseDatabase* createPoseDatabase(unsigned long numberOfSamples,unsigned short keypointNumberForEachSample)
{
 struct PoseDatabase * pdb = (struct PoseDatabase *) malloc(sizeof(struct PoseDatabase));
 if (pdb!=0)
 {
      memset(pdb,0,sizeof(struct PoseDatabase));

      pdb->maxTokenValue                = 0; //This will be updated as it is parsed..
      pdb->maxTokenValueSetExternally   = 0; //This will be updated as it is parsed..
      pdb->numberOfSamples = numberOfSamples; //This is needed for the reads to know that there is enough samples

      pdb->joint = (struct Joint *) malloc((keypointNumberForEachSample+1) * sizeof(struct Joint));
      if (pdb->joint!=0)
      {
          memset(pdb->joint,0,(keypointNumberForEachSample+1) * sizeof(struct Joint)); // <- clean up
      }

      pdb->sample = (struct PoseEntry *)  malloc((numberOfSamples+1) * sizeof(struct PoseEntry));
      if (pdb->sample!=0)
      {
          memset(pdb->sample,0,(numberOfSamples+1) * sizeof(struct PoseEntry)); // <- clean up
      }

      pdb->tokenBlackList = (struct DescriptionTokenBlacklist *) malloc(sizeof(struct DescriptionTokenBlacklist));
      if (pdb->tokenBlackList!=0)
      {
          memset(pdb->tokenBlackList,0,sizeof(struct DescriptionTokenBlacklist)); // <- clean up
          pdb->tokenBlackList->blackListSize = 0;//explicit cleanup
      }

      if (!load_embeddings("2d_pose_estimation/GloVe_D300.embeddings", &pdb->embeddings))
      {
          fprintf(stderr,"Will not go on without embeddings!\n");
          abort();
      }

      //Populate number of keypoints for each sample
      pdb->keypointsForEachSample = keypointNumberForEachSample;
      //fprintf(stderr," pdb->keypointsForEachSample = %hu\n",pdb->keypointsForEachSample);
 }
 return pdb;
}



//Parse a string to unsigned shorts..
//line : 4,1359,1126,583,4,974,1358,4,364,773
int parseDescriptionTokens(struct PoseDatabase * pdb,unsigned long sampleNumber,char *line)
{
  pdb->sample[sampleNumber].numberOfTokens = 0;

  unsigned int tokenID = 0;

  //Completely flush existing data (they should already be clean but be pedantic
  //in case there are multiple loads (?) in the same pose database.
  memset(pdb->sample[sampleNumber].descriptionTokens, 0,
         MAX_DESCRIPTION_TOKENS * sizeof(unsigned short));

  if (strcmp("000",line) == 0)
  {
    // 000 means no description / completely empty tokens
    return 0;
  }
   else
  {
   //go through tokenization
   char * tokenStart = line;
   char * tokenEnd   = 0;
   tokenID = 0; //Reset count for token counting

   do
   {
       //Seek comma seperation (or 0 marking end of line)
       tokenEnd   = strchr(tokenStart,',');

       //Convert comma to null-termination
       if (tokenEnd!=0)
         { *tokenEnd = 0; }

       //Convert string to integer
       unsigned short thisToken = (unsigned short) atoi(tokenStart);

       //Update max token value if it is not set exxternally
       if (!pdb->maxTokenValueSetExternally)
       {
         if (thisToken>pdb->maxTokenValue)
           { pdb->maxTokenValue = thisToken; }
       }

       //Actually store the token as a part of the description for the sample
       pdb->sample[sampleNumber].descriptionTokens[tokenID] = thisToken;

       //Arrange next start of token
       if (tokenEnd != 0)
        { tokenStart = tokenEnd + 1; }

       tokenID+=1;
   }
   while ((tokenEnd!=0) && (tokenStart!=0) && (tokenID<MAX_DESCRIPTION_TOKENS));

   if (tokenID>MAX_DESCRIPTION_TOKENS)
   { //This should never happen..
     fprintf(stderr,"Fatal Error: This is impossible, we have somehow overflowed description tokens while populating sample %lu \n",sampleNumber);
     abort();
   }

   //Use the counter as a metric of the tokens read.
   pdb->sample[sampleNumber].numberOfTokens = tokenID;
   return 1;
  }
}



struct PoseDatabase* readPoseDatabase(struct PoseDatabase* pdb,DescriptorDataset *ds,const char * path,unsigned int sourceID,unsigned long startOffset,unsigned long * loadedSamples,int ignoreNoSkeletonSamples)
{
    unsigned long poseIndex  = startOffset;
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr,"Error opening pose database file (%s).\n",path);
        return 0;
    }


    int res;

    if (pdb != 0)
    {
      char HEADER[5]={0};
      res = fscanf(fp,"%4s\n",HEADER);
      char VERSION_CHAR = '0' + DB_FILE_VERSION;
      if ((HEADER[0]!='D') || (HEADER[1]!='B') || (HEADER[2]!=VERSION_CHAR))
      {
        fprintf(stderr,"Incompatible version .db file (%s / %s), won't use it\n",path,HEADER);
        fprintf(stderr,"Expected %c, and got %c \n",VERSION_CHAR,HEADER[2]);
        fclose(fp);
        return 0;
      }

      //Check header..
      unsigned long thisNumberOfSamples = 0;
      unsigned short thisNumberOfKeypointsForEachSample = 0;
      res = fscanf(fp,"%lu\n",  &thisNumberOfSamples);
      if (res != 1) { fprintf(stderr,"Failed to read sample count from %s\n",path); fclose(fp); return 0; }
      res = fscanf(fp,"%hu\n" , &thisNumberOfKeypointsForEachSample);
      if (res != 1) { fprintf(stderr,"Failed to read keypoint count from %s\n",path); fclose(fp); return 0; }

      if (thisNumberOfKeypointsForEachSample!=pdb->keypointsForEachSample)
      {
          fprintf(stderr,"Inconsistency, we where expecting %u joints but we encountered %u\n",pdb->keypointsForEachSample,thisNumberOfKeypointsForEachSample);
          fprintf(stderr,"Number of samples was %lu\n",thisNumberOfSamples);
          fclose(fp);
          return 0;
      }

      fprintf(stderr,"File has %lu samples , %u keypoints / skeleton\n",thisNumberOfSamples,thisNumberOfKeypointsForEachSample);

      //These should have already be allocated
      if ( (pdb->sample != 0) && (pdb->joint != 0) )
      {
       // First pass: read joint names.
       // fgets is used instead of fscanf("%s") to respect the buffer size limit.
       // The newline fgets leaves at the end is stripped so names compare cleanly.
       for (int jID=0; jID<thisNumberOfKeypointsForEachSample; jID++)
       {
         if (fgets(pdb->joint[jID].name, MAX_JOINT_NAME, fp) == NULL)
         {
             fprintf(stderr,"Failed to read name for joint %d in %s\n",jID,path);
             fclose(fp);
             return 0;
         }
         size_t len = strlen(pdb->joint[jID].name);
         if (len > 0 && pdb->joint[jID].name[len - 1] == '\n')
             { pdb->joint[jID].name[len - 1] = '\0'; }
       }

      // Second pass: read the parent joint index for each joint.
      // Checked immediately — a bad read here would corrupt the skeleton hierarchy.
      for (int jID=0; jID<thisNumberOfKeypointsForEachSample; jID++)
       {
         unsigned int oldParentID = pdb->joint[jID].parent;
         res = fscanf(fp,"%hu\n" , &pdb->joint[jID].parent);
         if (res != 1) { fprintf(stderr,"Failed to read parent for joint %d in %s\n",jID,path); fclose(fp); return 0; }

         if (oldParentID != pdb->joint[jID].parent)
           {
             fprintf(stderr,YELLOW "Joint relation map update (%s) : Child ID %u -> Parent ID %hu | " NORMAL,path,jID,pdb->joint[jID].parent);
             fprintf(stderr,YELLOW "%s -> %s\n" NORMAL,pdb->joint[jID].name,pdb->joint[pdb->joint[jID].parent].name);
             if (oldParentID!=0)
             {
                fprintf(stderr,RED "Detected overwriting joint parent list, stopping execution to prevent a potential problem \n" NORMAL);
                abort();
             }
           }
       }

       // ── Per-sample parsing loop ──────────────────────────────────────────────
       // 'entriesLoaded' tracks how many file entries we have consumed.
       // 'poseIndex'     tracks which slot in pdb->sample we are writing to.
       // When ignoreNoSkeletonSamples is set, background-only entries are dropped:
       // poseIndex is NOT advanced so the slot is silently overwritten by the next
       // entry that does have skeletons.  This keeps the two counters in sync with
       // the descriptor dataset, which is always indexed by entriesLoaded.
       char line[MAX_LINE_LENGTH]={0};
       unsigned long entriesLoaded = 0;
       for (entriesLoaded=0; entriesLoaded<thisNumberOfSamples; entriesLoaded++)
       {
         // Zero-initialise the slot so partial reads leave known-safe values.
         pdb->sample[poseIndex].width              = 0;
         pdb->sample[poseIndex].height             = 0;
         pdb->sample[poseIndex].imagePath[0]       = 0;
         pdb->sample[poseIndex].numberOfSkeletons  = 0;

         // DB_STR(MAX_IMAGE_PATH) expands to the string literal of the constant,
         // keeping the fscanf width specifier automatically in sync with the buffer.
         res = fscanf(fp, "%" DB_STR(MAX_IMAGE_PATH) "s\n%hu,%hu,%hu\n",
                      pdb->sample[poseIndex].imagePath,
                      &pdb->sample[poseIndex].width,
                      &pdb->sample[poseIndex].height,
                      &pdb->sample[poseIndex].numberOfSkeletons);
         if (res != 4)
         {
             fprintf(stderr,"Failed to parse sample header at entry %lu in %s (fscanf returned %d)\n",
                     entriesLoaded, path, res);
             fclose(fp);
             return 0;
         }

         // ── Descriptor matching ───────────────────────────────────────────────
         pdb->sample[poseIndex].descriptor = NULL;
         #if USE_DINOV2_FEATURES
         if (ds!=0)
         {
             if (entriesLoaded<ds->count)
             {
              // Compare basenames only: strstr on full paths gives false positives when
              // one filename is a suffix of another (e.g. "1.jpg" matches inside "21.jpg").
              // strrchr finds the last '/' so we compare only the filename part.
              const char *img_base  = strrchr(pdb->sample[poseIndex].imagePath, '/');
              img_base  = img_base  ? img_base  + 1 : pdb->sample[poseIndex].imagePath;
              const char *desc_base = strrchr(ds->entries[entriesLoaded].filename, '/');
              desc_base = desc_base ? desc_base + 1 : ds->entries[entriesLoaded].filename;

              if (strcmp(img_base, desc_base) == 0)
                {
                    // Point directly into the descriptor dataset's memory — no copy needed.
                    pdb->sample[poseIndex].descriptor = ds->entries[entriesLoaded].values;
                } else
                {
                  fprintf(stderr,"Descriptor Mismatch @ %lu of %s | ",poseIndex,path);
                  fprintf(stderr,"Image (%s) | ",pdb->sample[poseIndex].imagePath);
                  fprintf(stderr,"Descriptor (%s)\n",ds->entries[entriesLoaded].filename);
                  fclose(fp);
                  return 0;
                }
             } else
             {
               // The descriptor file has fewer entries than the pose database — databases
               // must be regenerated together to stay in sync.
               fprintf(stderr,"Pose Index out of Descriptor Database range (%lu vs %u)?\n",poseIndex,ds->count);
               fclose(fp);
               return 0;
             }
         }
         #endif // USE_DINOV2_FEATURES

         // ── Description tokens ────────────────────────────────────────────────
         // Each sample line is followed by a comma-separated token list,
         // e.g. "4,1359,1126,583" — or "000" when no description is available.
         if (fgets(line, MAX_LINE_LENGTH, fp) != NULL)
            { parseDescriptionTokens(pdb,poseIndex,line); }
         else
            { fprintf(stderr,"Failed retrieving description line for entry %lu in %s\n",entriesLoaded,path); }

         // ── Skeleton count sanity check ───────────────────────────────────────
         // A count above MAX_SKELETONS_PER_IMAGE indicates file corruption;
         // continuing would loop thousands of times consuming wrong file content.
         if (pdb->sample[poseIndex].numberOfSkeletons > MAX_SKELETONS_PER_IMAGE)
         {
             fprintf(stderr,"Implausible skeleton count %hu at entry %lu in %s — aborting load\n",
                     pdb->sample[poseIndex].numberOfSkeletons, entriesLoaded, path);
             fclose(fp);
             return 0;
         }

         // Periodic progress indicator (every 1000 entries).
         if (poseIndex % DB_LOAD_PROGRESS_INTERVAL == 0)
         {
             fprintf(stderr,"\r Sample %lu/%lu | %s | %hux%hu | %hu skeletons        \r",
                     poseIndex, pdb->numberOfSamples,
                     pdb->sample[poseIndex].imagePath,
                     pdb->sample[poseIndex].width,
                     pdb->sample[poseIndex].height,
                     pdb->sample[poseIndex].numberOfSkeletons);
         }

         // ── Skeleton allocation and parsing ───────────────────────────────────
         // skipAdvance: when true the current poseIndex slot is discarded and will
         // be overwritten by the next entry.  This is how background-only samples
         // are silently dropped when ignoreNoSkeletonSamples is active.
         int skipAdvance = 0;

         if (pdb->sample[poseIndex].numberOfSkeletons == 0)
         {
               pdb->sample[poseIndex].sk = 0;
               if (ignoreNoSkeletonSamples)
               {
                   skipAdvance = 1;
               }
         } else
         {
              unsigned int skeletonDataSize = pdb->sample[poseIndex].numberOfSkeletons * sizeof(struct Skeleton);
              pdb->sample[poseIndex].sk = (struct Skeleton*) malloc(skeletonDataSize);

              if (pdb->sample[poseIndex].sk != 0)
              {
               memset(pdb->sample[poseIndex].sk, 0, skeletonDataSize);
               for (int skID = 0; skID < pdb->sample[poseIndex].numberOfSkeletons; skID++)
                {
                   if (fgets(line, MAX_LINE_LENGTH, fp) != NULL)
                        { parseSkeleton(pdb, poseIndex, skID, line); }
                   else { fprintf(stderr,"Failed reading skeleton %d for entry %lu in %s\n",skID,entriesLoaded,path); }
                }
              }
         }

         if (!skipAdvance)
         {
             poseIndex += 1;
         }
       } // Per-sample loop

       } //We managed to allocate sample data
       else
       { //We failed early so roll back allocations..
           if (pdb->sample != 0)          { free(pdb->sample); pdb->sample=0; }
           if (pdb->joint  != 0)          { free(pdb->joint);  pdb->joint=0;  }
           if (pdb->tokenBlackList != 0)  { free(pdb->tokenBlackList);  pdb->tokenBlackList=0;  }
       }
    } // We managed to allocate the pose database

  fclose(fp);

  *loadedSamples = poseIndex;
  return pdb;
}





int freePoseDatabase(struct PoseDatabase* pdb)
{
  if(pdb!=0)
    {
      // Free allocated memory
      if (pdb->sample!=0)
      {
      for (int i = 0; i < pdb->numberOfSamples; i++)
       {
         //for (int j = 0; j < pdb->sample[i].numberOfSkeletons; j++)
         // {
         //   free(pdb->sample[i].sk[j].coords); //<- Coords are now a constant to stop memory fragmentation
         // }

         if (pdb->sample[i].sk!=0)
           {
            free(pdb->sample[i].sk);
            pdb->sample[i].sk = 0;
           }
       }
       free(pdb->sample);
       pdb->sample = 0;
      }

      if (pdb->joint!=0)
      {
       free(pdb->joint);
       pdb->joint = 0;
      }

      if (pdb->tokenBlackList!=0)
      {
        free(pdb->tokenBlackList);
        pdb->tokenBlackList = 0;
      }

     if (pdb->embeddings.embeddings!=0)
      { free_embeddings(&pdb->embeddings); }

     free(pdb);//Free everything..

     return 1;
    }
   return 0;
}

