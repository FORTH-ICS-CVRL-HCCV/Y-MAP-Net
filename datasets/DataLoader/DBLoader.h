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
#include "superpointLoader.h"
#include "configuration.h"

// Stringify helpers — used to build fscanf width specifiers from constants
// so that format strings stay in sync with buffer sizes automatically.
#define _DB_STR(x) #x
#define DB_STR(x)  _DB_STR(x)

#define MAX_IMAGE_PATH        512   // Linux PATH_MAX is 4096; 512 covers all practical cases
#define MAX_JOINT_NAME         64
#define MAX_KEYPOINT_NUMBER    18
#define MAX_SKELETONS_PER_IMAGE      128   // Sanity ceiling — images with more than this are corrupt (raised from 62 2026-10-07: legitimate crowd/aerial (UAVHuman) scenes can exceed 62)
#define DB_LOAD_PROGRESS_INTERVAL  1000   // Print a progress line every N samples during DB load

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
  unsigned short bboxX,bboxY,bboxX2,bboxY2; //Bounding box corners (x1,y1)-(x2,y2), as the .db / .pzpd files store them
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

// [B5] Cached codec/extension choice for each optional per-sample file kind — see
// resolvePathToRequestedFiles(). The disk layout never changes between epochs, so the
// answer is resolved once and cached in PoseEntry instead of re-stat()ing every epoch.
enum DL_FileKind
{
  DL_FILE_NONE     = 0, // no file found on disk
  DL_FILE_PZP      = 1, // <name>.pzp
  DL_FILE_PNG      = 2, // <name>.png
  DL_FILE_DEPTHPNG = 3  // <name>_depth.png (Depth-Anything V1, depth only)
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

  // [B5] File-resolution cache: pathResolved is 0 until the first touch resolves this sample,
  // then the *Kind fields hold the codec/extension picked for each optional file (enum DL_FileKind).
  unsigned char  pathResolved;
  unsigned char  depthKind;    // DL_FILE_NONE / PZP / PNG / DEPTHPNG
  unsigned char  segKind;      // DL_FILE_NONE / PZP / PNG
  unsigned char  combinedKind; // DL_FILE_NONE / PZP / PNG
  unsigned char  geoKind;      // DL_FILE_NONE / PZP  (geolocation density, single format)

  struct Skeleton * sk;

  float * descriptor;

  // Borrowed pointer into the source's SuperpointDataset (NULL when no
  // .superpoint file). Matched positionally by basename, like 'descriptor'.
  SuperpointImageEntry * superpoint;

  // .pzpd samples (PZPDLoader.c): record ordinal in the source archive handle, and the caption
  // string (NOT NUL-terminated, points into the mapped archive, valid until the archive is closed).
  unsigned long  archiveOrdinal;
  const char *   description;
  unsigned int   descriptionLength;
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
 * @brief Maps source token IDs to replacement token IDs (synonym remapping).
 *        fromTokens[i] is replaced by toTokens[i] in every sample's token list.
 */
struct TokenSynonymMap
{
  unsigned int   pairCount;
  unsigned int   pairCapacity;
  unsigned short *fromTokens;
  unsigned short *toTokens;
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
  struct TokenSynonymMap           * tokenSynonymMap;
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
struct PoseDatabase* readPoseDatabase(struct PoseDatabase* pdb,DescriptorDataset *ds,SuperpointDataset *sp,const char * path,unsigned int sourceID,unsigned long startOffset,unsigned long * loadedSamples,int ignoreNoSkeletonSamples);

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
