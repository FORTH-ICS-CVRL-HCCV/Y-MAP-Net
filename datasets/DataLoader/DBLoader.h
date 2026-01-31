/**
 * @file DBLoader.h
 * @brief Header file for loading and manipulating pose databases.
 */

#ifndef _DBLOADER_H_
#define _DBLOADER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "processing/embeddings.h"
#include "descriptorConverter.h"
#include "configuration.h"

#define MAX_IMAGE_PATH 64 //TODO<- if you change this check for %63s and change it
#define MAX_JOINT_NAME 64
#define MAX_KEYPOINT_NUMBER 18

#define MAX_DESCRIPTION_TOKENS 32 //This started as 16..


#define FLIP_ODD_HEATMAPS 0

static unsigned short emptyDescriptionTokens[MAX_DESCRIPTION_TOKENS] = {0};
static float emptyDescriptor[DINOV2_FEATURES_LENGTH] = {0}; //Descriptors have ~768 elements but be safe..

//static float emptyDescriptionVectors[300] = {0};

/**
 * @brief Structure representing a skeleton.
 */
struct Skeleton
{
  unsigned short bboxX,bboxY,bboxW,bboxH;
  unsigned short coords[(1+ MAX_KEYPOINT_NUMBER) * 3 ]; //Keypoints (triplets)
};

/**
 * @brief Structure representing a joint.
 */
struct Joint
{
  char name[MAX_JOINT_NAME+1];
  signed char jointDifficulty; // Typically [-10 .. 10]
  unsigned short parent; //Keypoints (triplets)
};

/**
 * @brief Structure representing a pose entry.
 */
struct PoseEntry
{
  char imagePath[MAX_IMAGE_PATH+1];

  unsigned char  numberOfTokens;
  unsigned short descriptionTokens[MAX_DESCRIPTION_TOKENS]; //Max Value 65535

  unsigned short width;
  unsigned short height;
  unsigned short numberOfSkeletons;

  unsigned char  eraseEntry;

  struct Skeleton * sk;

  float * descriptor;
};

/**
 * @brief Structure representing a pose database.
 */
struct DescriptionTokenBlacklist
{
  unsigned int blackListCurrentTokens;
  unsigned int blackListSize;
  unsigned short * blackListedTokens;
};


/**
 * @brief Structure representing a pose database.
 */
struct PoseDatabase
{
  unsigned long  numberOfSamples;
  unsigned short keypointsForEachSample;

  unsigned short maxTokenValue;
  unsigned char  maxTokenValueSetExternally;

  struct PoseEntry * sample;
  struct Joint     * joint;
  struct Embeddings embeddings;
  struct DescriptionTokenBlacklist * tokenBlackList;
};


/**
 * @brief Function to fast read the number of samples in the database from the database file on disk.
 * @param path Path to the database.
 * @return Number of samples in the database.
 */
unsigned long  fastReadDatabaseNumberOfSamples(const char* path);


/**
 * @brief Function to fast read the number of keypoints per sample in the database from the database file on disk.
 * @param path Path to the database.
 * @return Number of keypoints per sample in the database.
 */
unsigned short fastReadDatabaseNumberOfKeypointsPerSample(const char* path);


/**
 * @brief Function to create a pose database.
 * @param numberOfSamples Number of samples.
 * @param keypointNumberForEachSample Number of keypoints for each sample.
 * @return Pointer to the created pose database.
 */
struct PoseDatabase * createPoseDatabase(unsigned long numberOfSamples,unsigned short keypointNumberForEachSample);


/**
 * @brief Function to read a pose database from disk.
 * @param pdb Pointer to the pose database.
 * @param path Path to the database.
 * @param sourceID Source ID.
 * @param startOffset Start offset.
 * @return Pointer to the read pose database.
 */
//struct PoseDatabase * readPoseDatabase(struct PoseDatabase* pdb,const char * path,unsigned int sourceID,unsigned long startOffset,int ignoreNoSkeletonSamples);
struct PoseDatabase* readPoseDatabase(struct PoseDatabase* pdb,DescriptorDataset *ds,const char * path,unsigned int sourceID,unsigned long startOffset,unsigned long * loadedSamples,int ignoreNoSkeletonSamples);

/**
 * @brief Function to free memory allocated for a pose database.
 * @param pdb Pointer to the pose database.
 * @return 1 on success, 0 on failure.
 */
int freePoseDatabase(struct PoseDatabase* pdb);

#ifdef __cplusplus
}
#endif


#endif
