#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DBLoader.h"
#include "DataLoader.h"

#define MAX_LINE_LENGTH 1000

// Function to parse skeleton data from string
int parseSkeleton(struct PoseDatabase * pdb,unsigned long sampleNumber,int skID,char *line)
{
  if ( (pdb!=0) && (line!=0) && (sampleNumber<pdb->numberOfSamples) && (pdb->sample[sampleNumber].sk!=0) )
  {
    struct Skeleton * sk = &pdb->sample[sampleNumber].sk[skID];
    unsigned short skIDFromFile; //<- not used any more
    sscanf(line, "SK%hu,%hu,%hu,%hu,%hu,", &skIDFromFile, &sk->bboxX, &sk->bboxY, &sk->bboxW, &sk->bboxH);

    skIDFromFile = skIDFromFile - 1; //file always is +1 compared to skID
    if (skID!=(int) skIDFromFile)
    {
        fprintf(stderr,"Inconsistency in sample %lu / expected skeleton %d / found %hu \n",sampleNumber,skID,skIDFromFile);
    }

    // Extract coordinates
    int ignored = 5;
    int i = 0;
    char *rest = NULL;
    char *token;
    for (
         token = strtok_r(line, ",", &rest);
         token != NULL;
         token = strtok_r(NULL, ",", &rest)
        )
        {
           if (i>=ignored) //We ignore first values of the line since they contain the bbox!
            {
              sk->coords[i-ignored] = (unsigned short) atoi(token);
            }
          //printf("token:%s\n", token);
          i=i+1;
        }

    //fprintf(stderr,"Sk %u had %u coords\n",skID,i);
    return 1;
  }
  return 0;
}


unsigned long fastReadDatabaseNumberOfSamples(const char* path)
{
    unsigned long numberOfSamples = 0;
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr,"Error opening pose database file, Fast Read Samples (%s).\n",path);
        return 0;
    }

    int res;
    char HEADER[5]={0};
    res = fscanf(fp,"%4s\n",HEADER);
    res = fscanf(fp,"%lu\n", &numberOfSamples);
    fclose(fp);
   return numberOfSamples;
}



unsigned short fastReadDatabaseNumberOfKeypointsPerSample(const char* path)
{
    unsigned long  numberOfSamples = 0;
    unsigned short keypointsPerSample = 0;
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr,"Error opening pose database file, Fast Read Keypoint Number (%s).\n",path);
        return 0;
    }

    int res;
    char HEADER[5]={0};
    res = fscanf(fp,"%4s\n",HEADER);
    res = fscanf(fp,"%lu\n", &numberOfSamples);
    res = fscanf(fp,"%hu\n", &keypointsPerSample);
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
      if (pdb->joint!=0)
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
  for (unsigned int tokenID = 0; tokenID<MAX_DESCRIPTION_TOKENS; tokenID++)
  {
       pdb->sample[sampleNumber].descriptionTokens[tokenID] = 0;
  }

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
      unsigned long thisNumberOfSamples;
      unsigned short thisNumberOfKeypointsForEachSample;
      res = fscanf(fp,"%lu\n",  &thisNumberOfSamples);
      res = fscanf(fp,"%hu\n" , &thisNumberOfKeypointsForEachSample);

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
       for (int jID=0; jID<thisNumberOfKeypointsForEachSample; jID++)
       {
         //res = fscanf(fp,"%s\n" , pdb->joint[jID].name); //Read without checking size
         char * unusedPtr = fgets(pdb->joint[jID].name, MAX_JOINT_NAME-1, fp); //Dont read past 55
         //Remove newline
         size_t len = strlen(pdb->joint[jID].name);
         if (len > 0 && pdb->joint[jID].name[len - 1] == '\n')
             {
              pdb->joint[jID].name[len - 1] = '\0'; // Replace newline with null terminator
             }
         //fprintf(stderr,"Joint %u -> %s\n",jID,pdb->joint[jID].name);
       }

      for (int jID=0; jID<thisNumberOfKeypointsForEachSample; jID++)
       {
         unsigned int oldParentID = pdb->joint[jID].parent;
         res = fscanf(fp,"%hu\n" , &pdb->joint[jID].parent);

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

       char line[MAX_LINE_LENGTH]={0};
       unsigned long entriesLoaded = 0;
       for (entriesLoaded=0; entriesLoaded<thisNumberOfSamples; entriesLoaded++)
       {

         //Ensure sane values if fscanf fails
         pdb->sample[poseIndex].width = 0;
         pdb->sample[poseIndex].height= 0;
         pdb->sample[poseIndex].imagePath[0] = 0;
         pdb->sample[poseIndex].numberOfSkeletons = 0;
         //TODO: Check if MAX_IMAGE_PATH is 64 and replace %63s if it changes
         res = fscanf(fp, "%63s\n%hu,%hu,%hu\n",pdb->sample[poseIndex].imagePath, &pdb->sample[poseIndex].width, &pdb->sample[poseIndex].height, &pdb->sample[poseIndex].numberOfSkeletons);


         pdb->sample[poseIndex].descriptor = NULL;
         #if USE_DINOV2_FEATURES
         if (ds!=0)
         {
             if (entriesLoaded<ds->count)
             {
              if (strstr(pdb->sample[poseIndex].imagePath , ds->entries[entriesLoaded].filename)!=0)
                {
                    pdb->sample[poseIndex].descriptor =  ds->entries[entriesLoaded].values;
                } else
                {
                  fprintf(stderr,"Descriptor Mismatch @ %lu of %s | ",poseIndex,path);
                  fprintf(stderr,"Image (%s) | ",pdb->sample[poseIndex].imagePath);
                  fprintf(stderr,"Descriptor (%s)\n",ds->entries[entriesLoaded].filename);
                  exit(1); //<- This should never happen!
                }
             } else
             {
               fprintf(stderr,"Pose Index out of Descriptor Database range (%lu vs %u)?\n",poseIndex,ds->count);
               fprintf(stderr,"This should never happen, halting termination!\n");
               exit(1); //<- This should never happen!
             }
         }
         #endif // USE_DINOV2_FEATURES



         //TODO: scan here for
         //Parse Description Tokens, should be maximum MAX_DESCRIPTION_TOKENS
         //4,1359,1126,583,4,974,1358,4,364,773
         char * fgetsResult = fgets(line, MAX_LINE_LENGTH, fp);
         if (fgetsResult!=0)
            { parseDescriptionTokens(pdb,poseIndex,line); } else
            { fprintf(stderr,"Failed retrieving description line from file \n"); }


         if (pdb->sample[poseIndex].numberOfSkeletons>100)
         {
           fprintf(stderr,"Something went wrong @ sample %lu",poseIndex);
           exit(1);
         }
         // Allocate memory for skeletons
         if  (poseIndex%1000==0)
             {
              fprintf(stderr,"\r Sample %lu/%lu | %s | %hux%hu | %hu skeletons        \r",poseIndex,pdb->numberOfSamples, pdb->sample[poseIndex].imagePath, pdb->sample[poseIndex].width, pdb->sample[poseIndex].height, pdb->sample[poseIndex].numberOfSkeletons);
             }

             if (pdb->sample[poseIndex].numberOfSkeletons==0)
             {
               pdb->sample[poseIndex].sk = 0; //No skeletons
               if (ignoreNoSkeletonSamples)
               {
                   if (poseIndex>0)
                   {
                       poseIndex-=1;
                   } else
                   {
                     fprintf(stderr,"Cannot ignore first sample for missing a background\n");
                   }
               }
             } else
             {
              unsigned int skeletonDataSize = pdb->sample[poseIndex].numberOfSkeletons * sizeof(struct Skeleton);
              pdb->sample[poseIndex].sk     = (struct Skeleton*) malloc(skeletonDataSize);

              if (pdb->sample[poseIndex].sk!=0)
              {
               memset(pdb->sample[poseIndex].sk,0,skeletonDataSize);
               // Parse skeleton data
               for (int skID = 0; skID < pdb->sample[poseIndex].numberOfSkeletons; skID++)
                {
                   char * fgetsResult = fgets(line, MAX_LINE_LENGTH, fp);
                   if (fgetsResult!=0) { parseSkeleton(pdb,poseIndex,skID,line); }else
                                       { fprintf(stderr,"Failed reading new skeleton ID %u for index %lu\n",skID,poseIndex); }

                }
              } //Skeleton allocation was successful
             }// We have at least one skeleton

             //Increment pose index
             poseIndex +=1;
           } // Each Sample loop

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

