/** @file DataLoader.h
 *  @brief  This is a fast multithreaded library to supply samples for neural network training
            Please look at its Python binding (DataLoader.py) for its intended use.
            The function test() can be used to test the library assuming you have the datasets available in your local file system
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef DATALOADER_H_INCLUDED
#define DATALOADER_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include "pthreadWorkerPool.h"

#include "codecs/image.h"
#include "cache.h"

#include "descriptorConverter.h"
#include "superpointLoader.h"

//.pzpd archives (vendored from the PZP repository, see PZPDLoader.c)
#include "pzpdir/pzpdir.h"

static const char version[]="Heroic v0.96";

//Make sure the DB files are created by a compatible version
//of the python code..
#define DB_FILE_VERSION 1

//Limit for paths used to access files from disk
#define MAX_PATH 2048

//All static method limits and configurations
//are defined here in configuration.h
//------------------------------------------------------
#include "configuration.h"
//------------------------------------------------------

//Single source of truth for the 8-bit heatmap output channel layout
//(which modality occupies which channel range). See HeatmapLayout.h.
//------------------------------------------------------
#include "HeatmapLayout.h"
//------------------------------------------------------

//Runtime-configurable data-augmentation parameters. See DataAugmentation.h.
//------------------------------------------------------
#include "DataAugmentation.h"
//------------------------------------------------------

//Some Terminal Colors
//------------------------------------------------------
#define NORMAL   "\033[0m"
#define BLACK    "\033[30m"      /* Black */
#define RED      "\033[31m"      /* Red */
#define GREEN    "\033[32m"      /* Green */
#define YELLOW   "\033[33m"      /* Yellow */
#define WHITE    "\033[37m"      /* White */
//------------------------------------------------------


/**
 * @brief Type definition for sample number. Use this to make sure when we talk about a sample number
 */
typedef unsigned long SampleNumber;

/**
 * @brief Type definition for dataset source ID.
 */
typedef unsigned long DatasetSourceID;

/**
 * @brief Type definition for a target 8 Bit heatmap.
 */
typedef unsigned int Heatmap8BitIndex;

/**
 * @brief Type definition for a target 16 Bit heatmap.
 */
typedef unsigned int Heatmap16BitIndex;

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//---------               RUNTIME MEMORY SEGMENTATION CHECK         ----------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

#define CANARY_SIZE 10240
//10101010101010101010101010101010 = 2863311530
#define CANARY_VALUE 2863311530
#define USE_CANARY 1

struct Canary
{
   unsigned long * shouldRemainUntouched;
};

static void setupCanary(unsigned long * m, unsigned long length)
{
 #if USE_CANARY
 if (m!=0)
 {
  for (unsigned long i=0; i<length; i++)
  {
      m[i]=CANARY_VALUE;
  }
 } else
 {
     fprintf(stderr,"Aborting, could not setup canary\n");
     abort();
 }
 #endif // USE_CANARY
}

static void checkCanary(const char * msg,const unsigned long * m,const unsigned long length)
{
 #if USE_CANARY
 if (m!=0)
 {
  for (unsigned long i=0; i<length; i++)
  {
      if (m[i]!=CANARY_VALUE)
      {
        fprintf(stderr,RED "Aborting, internal check failed @ %s..\n" NORMAL,msg);
        abort();
      }
  }
 } else
 {
     fprintf(stderr,"Aborting, could not check canary\n");
     abort();
 }
 #endif // USE_CANARY
}
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

/**
 * @brief Structure for holding image data.
 */
struct Heatmaps
{
  void * pixels;
  unsigned int width;
  unsigned int height;
  unsigned int channels;
  unsigned int bitsperpixel;
  unsigned long numberOfImages;
  void * pixelsLimit;
};

/**
 * @brief Structure for representing database entries from different sources.
 */
struct DatabaseEntry
{
  unsigned long numberOfSamples;
  char ignoreNoSkeletonSamples;
  char pathToDBFile[MAX_PATH+1];
  char pathToCOCOImages[MAX_PATH+1];
  char pathToCOCODepthMaps[MAX_PATH+1];
  char pathToCOCOSegmentations[MAX_PATH+1];
  char pathToCombinedMetaData[MAX_PATH+1];
  char pathToCOCOGeolocation[MAX_PATH+1];  //Derived geo_<split> sibling of the images dir (see GEOLOCATION.md).
  //----------------------------------------
  DescriptorDataset * descriptorsAsADataset;
  //----------------------------------------
  // Raw SuperPoint keypoints+descriptors for this source (NULL when no
  // .superpoint file). Owned here; samples borrow pointers into it.
  SuperpointDataset * superpointData;
  //----------------------------------------
  // .pzpd sources (PZPDLoader.c): pathToDBFile is an archive, the directory paths are unused.
  char isArchive;
  unsigned int descriptorLength; // floats per descriptor_dinov3 row, 0 when the archive has none
};

/**
 * @brief Structure for holding a list of databases.
 */
struct DatabaseList
{
  unsigned int numberOfSources;
  unsigned long totalNumberOfSamples;
  unsigned short numberOfKeypointsPerSample;
  struct DatabaseEntry * source;
  //----------------------------------------
  // All .pzpd sources opened as one handle (member = sourceID), NULL for .db sources.
  pzpd * archive;
  int archiveStream[4]; // stream ids of rgb, all, depth, seg (-1 if absent), see enum PZPDStream
};


struct Heatmap
{
    int dataSize;              // Size of the heatmap (heatmapSize x heatmapSize)
    signed char* data;     // Pointer to the heatmap data
};

struct HeatmapCollection
{
    int minimumGradientSize;
    int maximumGradientSize;
    int num_heatmaps;      // Number of heatmaps
    struct Heatmap* heatmaps;              // Array of heatmaps
    struct Heatmap* heatmaps_positive;     // Array of Positive value only heatmaps
};



/**
 * @brief Structure representing an image database.
 */
struct ImageDatabase
{
  //Each Image Database has a number of samples e.g. COCO17 has 118.287 samples
  //these reside in PoseDatabase (DBLoader.c/h)
  unsigned long numberOfSamples;
  unsigned long * indices;
  unsigned long * trainPasses;
  float * losses;
  struct PoseDatabase * pdb;
  //-----------------------------------
  //Each database has a numberOfImages it can accommodate e.g. COCO17 has 118.287 samples
  //This can either be the same as numberOfSamples (if we have enough RAM) or it can be smaller
  //when streaming batches
  unsigned long numberOfImagesThatCanBeLoaded;
  //[B1] Double buffering (DATALOADER.md B1): in/out8bit/out16bit stay THE FILL-SIDE
  //buffers that every worker/heatmap/flip/rotate/denoise path writes through — the hot
  //path never learns about double buffering. The shadow set below holds the CONSUME side;
  //db_swap_buffers() exchanges just the pixels/pixelsLimit pointers between the two sets
  //while the pipeline is idle, so the consumer (Python) reads the shadow side via
  //db_get_consumer_* while workers fill the next batch here.
  struct Heatmaps in;
  struct Heatmaps out8bit;
  struct Heatmaps out16bit;
  //[B1] Shadow (consume-side) buffers — allocated ONLY when doubleBuffer is on, which in
  //turn requires streamData (full-load mode would double whole-dataset RAM for nothing).
  //When doubleBuffer==0 these stay zeroed and db_get_consumer_* fall back to the fill side.
  char doubleBuffer;
  struct Heatmaps inShadow;
  struct Heatmaps out8bitShadow;
  struct Heatmaps out16bitShadow;
  //-----------------------------------
  char doHeatmapOutput;
  //-----------------------------------
  char doAugmentations;
  char addPAFHeatmap;
  char addBackgroundHeatmap;
  char addDepthHeatmap;
  char addNormalHeatmaps;
  char addSegmentationHeatmaps;
  char addDepthLevelsHeatmaps;
  char addInstanceHeatmaps;
  char addSuperpointHeatmaps;
  int  superpointChannels;            // K = number of PCA components / channels
  SuperpointPCA * superpointPCA;      // frozen global basis, shared across sources
  char addGeolocationHeatmap;         // 1 => emit the global lat/lon density channel (see GEOLOCATION.md)
  unsigned char combinedSegChannel;       // combined ("all") files: channel of the segmentation label (db_set_combined_layout)
  unsigned char combinedDepthHighChannel; // combined files: channel of the depth high byte
  unsigned char combinedDepthLowChannel;  // combined files: channel of the depth low byte
  //-----------------------------------
  Heatmap8BitIndex  PAFHeatmapIndex;
  Heatmap8BitIndex  backgroundHeatmapIndex;
  Heatmap8BitIndex  depthmapHeatmapIndex8Bit;
  Heatmap8BitIndex  normalsHeatmapIndex;
  Heatmap8BitIndex  depthLevelsHeatmapIndex;
  Heatmap8BitIndex  denoiseHeatmapIndex;
  Heatmap8BitIndex  leftRightHeatmapIndex;
  Heatmap8BitIndex  segmentationHeatmapIndex;
  Heatmap8BitIndex  instanceHeatmapIndex;
  Heatmap8BitIndex  superpointHeatmapIndex;
  Heatmap8BitIndex  geolocationHeatmapIndex;

  Heatmap16BitIndex  depthmapHeatmapIndex16Bit;
  //-----------------------------------
  //Resolved 8-bit heatmap channel layout (single source of truth, see HeatmapLayout.h).
  //The Heatmap8BitIndex fields above are DERIVED from this table in db_create.
  struct HeatmapLayout heatmapLayout;
  //-----------------------------------
  //Runtime augmentation parameters (seeded from compile-time defaults by
  //augmentationParamsSetDefaults; overridable at runtime or from Python).
  struct DataAugmentation aug;
  //-----------------------------------
  struct DatabaseList * dbSources;
  //-----------------------------------
  // .pzpd prefetcher (PZPDLoader.c): the epoch order is submitted ahead of the workers.
  pzpd_prefetcher * prefetcher;
  unsigned long indicesVersion;    // bumped by every shuffle
  unsigned long prefetchedVersion; // indicesVersion the schedule was submitted for
  unsigned long prefetchNext;      // position the schedule expects the next batch to start at
  //-----------------------------------
  struct HeatmapCollection * gradients;
  //-----------------------------------
  struct cache * files;
  //-----------------------------------
  //unsigned int descriptorElementsNumber; //<- This refers to DescriptorDataset
  //-----------------------------------
  char computationsRunning;
  unsigned short numberOfThreads;
  struct workerPool threadPool;
  struct workerThreadContext *threadCtx;
  //-----------------------------------
  unsigned long threadStartLagMicroseconds;
  unsigned long threadCompletionLagMicroseconds;
  //-----------------------------------
  struct Canary  canaryA;
  struct Canary  canaryB;
  struct Canary  canaryC;
};


static const int numberOfPAFJoints = 12;
static const int PAFJoints[] = {10,8,6,16,14,12,15,13,11,9,7,5};

/*
enum Multiplexing
{
    Unlabeled=0, //0
    Person,      //1
    Vehicle,     //2
    Animal,      //3
    Object,      //4
    Furniture,   //5
    Appliance,   //6
    Material,    //7
    Obstacle,    //8
    Building,    //9
    Nature,      //10
    MULTIPLEXING_CATEGORY_LIMIT
};


static const char segmentation_value_multiplexing[MAX_SEGMENTATION_VALUE_MULTIPLEXING] =
{
    Unlabeled,  // unlabeled (0) - Unlabeled
    Person,  // person (1) - Person
    Vehicle,  // bicycle (2) - Vehicle
    Vehicle,  // car (3) - Vehicle
    Vehicle,  // motorcycle (4) - Vehicle
    Vehicle,  // airplane (5) - Vehicle
    Vehicle,  // bus (6) - Vehicle
    Vehicle,  // train (7) - Vehicle
    Vehicle,  // truck (8) - Vehicle
    Vehicle,  // boat (9) - Vehicle
    Obstacle,  // traffic light (10) - Obstacle
    Obstacle,  // fire hydrant (11) - Obstacle
    Obstacle,  // street sign (12) - Obstacle
    Obstacle,  // stop sign (13) - Obstacle
    Obstacle,  // parking meter (14) - Obstacle
    Furniture,  // bench (15) - Furniture
    Animal,  // bird (16) - Animal
    Animal,  // cat (17) - Animal
    Animal,  // dog (18) - Animal
    Animal,  // horse (19) - Animal
    Animal,  // sheep (20) - Animal
    Animal,  // cow (21) - Animal
    Animal,  // elephant (22) - Animal
    Animal,  // bear (23) - Animal
    Animal,  // zebra (24) - Animal
    Animal,  // giraffe (25) - Animal
    Object,  // hat (26) - Object
    Object,  // backpack (27) - Object
    Object,  // umbrella (28) - Object
    Object,  // shoe (29) - Object
    Object,  // eye glasses (30) - Object
    Object,  // handbag (31) - Object
    Object,  // tie (32) - Object
    Object,  // suitcase (33) - Object
    Object,  // frisbee (34) - Object
    Object,  // skis (35) - Object
    Object,  // snowboard (36) - Object
    Object,  // sports ball (37) - Object
    Object,  // kite (38) - Object
    Object,  // baseball bat (39) - Object
    Object,  // baseball glove (40) - Object
    Object,  // skateboard (41) - Object
    Object,  // surfboard (42) - Object
    Object,  // tennis racket (43) - Object
    Object,  // bottle (44) - Object
    Object,  // plate (45) - Object
    Object,  // wine glass (46) - Object
    Object,  // cup (47) - Object
    Object,  // fork (48) - Object
    Object,  // knife (49) - Object
    Object,  // spoon (50) - Object
    Object,  // bowl (51) - Object
    Object,  // banana (52) - Object
    Object,  // apple (53) - Object
    Object,  // sandwich (54) - Object
    Object,  // orange (55) - Object
    Object,  // broccoli (56) - Object
    Object,  // carrot (57) - Object
    Object,  // hot dog (58) - Object
    Object,  // pizza (59) - Object
    Object,  // donut (60) - Object
    Object,  // cake (61) - Object
    Furniture,  // chair (62) - Furniture
    Furniture,  // couch (63) - Furniture
    Furniture,  // potted plant (64) - Furniture
    Furniture,  // bed (65) - Furniture
    Object,  // mirror (66) - Object
    Furniture,  // dining table (67) - Furniture
    Furniture,  // window (68) - Furniture
    Furniture,  // desk (69) - Furniture
    Furniture,  // toilet (70) - Furniture
    Object,  // door (71) - Object
    Appliance,  // tv (72) - Object
    Appliance,  // laptop (73) - Appliance
    Appliance,  // mouse (74) - Appliance
    Appliance,  // remote (75) - Appliance
    Appliance,  // keyboard (76) - Appliance
    Appliance,  // cell phone (77) - Appliance
    Appliance,  // microwave (78) - Appliance
    Appliance,  // oven (79) - Appliance
    Appliance,  // toaster (80) - Appliance
    Object,     // sink (81) - Appliance
    Appliance,  // refrigerator (82) - Appliance
    Appliance,  // blender (83) - Appliance
    Object,     // book (84) - Object
    Object,     // clock (85) - Object
    Object,     // vase (86) - Object
    Object,     // scissors (87) - Object
    Object,     // teddy bear (88) - Object
    Appliance,  // hair drier (89) - Appliance
    Appliance,  // toothbrush (90) - Object
    Appliance,  // hair brush (91) - Object
    Object,     // banner (92) - Object
    Object,     // blanket (93) - Object
    Obstacle,   // branch (94) - Object
    Building,   // bridge (95) - Obstacle
    Building,   // building-other (96) - Obstacle
    Obstacle,   // bush (97) - Obstacle
    Furniture,  // cabinet (98) - Furniture
    Obstacle,   // cage (99) - Obstacle
    Obstacle,   // cardboard (100) - Material
    Material,   // carpet (101) - Material
    Material,   // ceiling-other (102) - Material
    Material,   // ceiling-tile (103) - Material
    Material,   // cloth (104) - Material
    Object,     // clothes (105) - Object
    Nature,     // clouds (106) - Nature
    Furniture,  // counter (107) - Furniture
    Furniture,  // cupboard (108) - Furniture
    Material,   // curtain (109) - Material
    Object,     // desk-stuff (110) - Object
    Nature,     // dirt (111) - Nature
    Obstacle,   // door-stuff (112) - Obstacle
    Material,   // fence (113) - Obstacle
    Material,   // floor-marble (114) - Material
    Material,   // floor-other (115) - Material
    Material,   // floor-stone (116) - Material
    Material,   // floor-tile (117) - Material
    Material,   // floor-wood (118) - Material
    Nature,     // flower (119) - Nature
    Nature,     // fog (120) - Nature
    Object,     // food-other (121) - Object
    Object,     // fruit (122) - Object
    Furniture,  // furniture-other (123) - Furniture
    Nature,     // grass (124) - Nature
    Nature,     // gravel (125) - Nature
    Nature,     // ground-other (126) - Nature
    Nature,     // hill (127) - Nature
    Building,   // house (128) - Building
    Nature,     // leaves (129) - Nature
    Object,     // light (130) - Obstacle
    Material,   // mat (131) - Material
    Material,   // metal (132) - Material
    Furniture,  // mirror-stuff (133) - Material
    Nature,     // moss (134) - Nature
    Nature,     // mountain (135) - Nature
    Nature,     // mud (136) - Nature
    Material,   // napkin (137) - Material
    Object,     // net (138) - Obstacle
    Material,   // paper (139) - Material
    Building,   // pavement (140) - Material
    Object,     // pillow (141) - Material
    Nature,     // plant-other (142) - Nature
    Material,   // plastic (143) - Material
    Building,   // platform (144) - Obstacle
    Building,   // playingfield (145) - Obstacle
    Building,   // railing (146) - Obstacle
    Building,   // "railroad": 147,
    Nature,     // "river": 148,
    Building,   //"road": 149,
    Nature,     //"rock": 150,
    Building,   //"roof": 151,
    Material,   //"rug": 152,
    Object,     // "salad": 153,
    Nature,     //"sand": 154,
    Nature,     //"sea": 155,
    Furniture,  // "shelf": 156,
    Nature,     //"sky-other": 157,
    Building,   //"skyscraper": 158,
    Nature,     //"snow": 159,
    Material,   //"solid-other": 160,
    Building,   //"stairs": 161,
    Nature,     //"stone": 162,
    Object,     //"straw": 163,
    Building,   //"structural-other": 164,
    Furniture,  // "table": 165,
    Object,     //"tent": 166,
    Material,   //"textile-other": 167,
    Object,     //"towel": 168,
    Nature,     //"tree": 169,
    Object,     //"vegetable": 170,
    Obstacle,   // "wall-brick": 171,
    Obstacle,   // "wall-concrete": 172,
    Obstacle,   // "wall-other": 173,
    Obstacle,   // "wall-panel": 174,
    Obstacle,   // "wall-stone": 175,
    Obstacle,   // "wall-tile": 176,
    Obstacle,   // "wall-wood": 177,
    Nature,     //"water-other": 178,
    Nature,     //"waterdrops": 179,
    Building,   //"window-blind": 180,
    Building,   //"window-other": 181,
    Material    //"wood": 182
};
*/

static int getNumberOfValidInstanceLabels()
{
  return VALID_SEGMENTATIONS;
}


int db_map_sample_in_to_image(struct ImageDatabase * db,struct Image * picToMap,unsigned long sampleNumber);
unsigned int db_resolve_sample_sourceID(struct ImageDatabase * db, SampleNumber sample);

//Canonical position -> sampleNumber (-> sourceID) resolution — the ONLY sanctioned way to
//dereference db->indices[]; all bounds/corruption checks live inside (see DataLoader.c).
//Return 1 on success, 0 for an out-of-range position; abort on index/source corruption.
int db_resolve_sample(struct ImageDatabase *db, SampleNumber position, SampleNumber *outSampleNumber);
int db_resolve_sample_and_source(struct ImageDatabase *db, SampleNumber position, SampleNumber *outSampleNumber, DatasetSourceID *outSourceID);

//[B1] Double-buffer pipeline primitives (DATALOADER.md B1). db_pipeline_drain waits out any
//in-flight StartUpdate — MUST be called before anything that mutates worker-read state
//(shuffle, aug params, destroy); it is invoked internally by those entry points as a
//safety net. db_swap_buffers exchanges fill/consume pixel pointers (no-op when
//doubleBuffer is off, refuses while computations run). db_get_consumer_* return the
//consume-side base pointers, falling back to the fill side when doubleBuffer is off.
int db_pipeline_drain(struct ImageDatabase * db);
int db_swap_buffers(struct ImageDatabase * db);
unsigned char * db_get_consumer_in(struct ImageDatabase *db,unsigned long startSample);
char *          db_get_consumer_out(struct ImageDatabase *db,unsigned long startSample);
signed short *  db_get_consumer_out16bit(struct ImageDatabase *db,unsigned long startSample);

//Fill-side getters (historical Python API; the consumer getters above fall back to these)
unsigned char * db_get_in(struct ImageDatabase *db,unsigned long startSample);
char *          db_get_out(struct ImageDatabase *db,unsigned long startSample);
signed short *  db_get_out16bit(struct ImageDatabase *db,unsigned long startSample);

/**
 * @brief Fill a caller-allocated buffer with DINOv2 descriptors for a batch of samples.
 *        Respects the current shuffle order (uses db->indices[]).
 *        When USE_DINOV2_FEATURES is 0 this function is a no-op.
 * @param db          ImageDatabase handle.
 * @param start       First batch index (inclusive).
 * @param end         Last  batch index (exclusive).
 * @param out_buffer  Pre-allocated float array of size (end-start) * DINOV2_FEATURES_LENGTH.
 */
void db_get_batch_descriptors(struct ImageDatabase *db,
                               unsigned long start,
                               unsigned long end,
                               float *out_buffer);

/**
 * @brief Fill a caller-allocated buffer with the description tokens for a batch of samples,
 *        one C call instead of one ctypes call per sample (DATALOADER.md B3). Respects shuffle order.
 * @param out_buffer  Pre-allocated ushort array of size (end-start) * MAX_DESCRIPTION_TOKENS.
 */
void db_get_batch_tokens(struct ImageDatabase *db,
                         unsigned long start,
                         unsigned long end,
                         unsigned short *out_buffer);

/**
 * @brief Fill a caller-allocated buffer with the token embedding vectors for a batch of samples,
 *        one C call instead of the per-sample/per-token/per-dim ctypes loops (DATALOADER.md B3).
 * @param D           Embedding dimension the caller allocated for (per-token copy width).
 * @param out_buffer  Pre-allocated float array of size (end-start) * MAX_DESCRIPTION_TOKENS * D.
 */
void db_get_batch_embeddings(struct ImageDatabase *db,
                             unsigned long start,
                             unsigned long end,
                             unsigned long D,
                             float *out_buffer);

/**
 * @brief The offset currently applied to the loaded embeddings (from the file header unless
 *        overridden by db_set_embeddings_offset_scaling).
 */
float db_get_embeddings_offset(struct ImageDatabase *db);

/**
 * @brief The scaling currently applied to the loaded embeddings.
 */
float db_get_embeddings_scaling(struct ImageDatabase *db);

/**
 * @brief Re-express the loaded embeddings under a new (offset, scaling) pair.
 *
 * Vectors are stored as (raw + offset) * scaling, which is what the tanh token head is
 * trained against. This undoes the pair currently in force and applies the new one in
 * place, so the targets can be rescaled without regenerating GloVe_D*.embeddings.
 * @return 1 on success, 0 if there are no embeddings or the current scaling is 0
 *         (not invertible).
 */
int db_set_embeddings_offset_scaling(struct ImageDatabase *db, float offset, float scaling);

/**
 * @brief Function for testing the library (assuming you have the datasets in your local filesystem).
 * @return Returns an integer representing the test result.
 */
int test();

#ifdef __cplusplus
}
#endif

#endif // DATALOADER_H_INCLUDED
