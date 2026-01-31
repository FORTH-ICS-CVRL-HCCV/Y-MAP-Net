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

static const char version[]="Heroic v0.91b";

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
  //----------------------------------------
  DescriptorDataset * descriptorsAsADataset;
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
  struct Heatmaps in;
  struct Heatmaps out8bit;
  struct Heatmaps out16bit;
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
  //-----------------------------------
  Heatmap8BitIndex  PAFHeatmapIndex;
  Heatmap8BitIndex  backgroundHeatmapIndex;
  Heatmap8BitIndex  depthmapHeatmapIndex8Bit;
  Heatmap8BitIndex  normalsHeatmapIndex;
  Heatmap8BitIndex  segmentationHeatmapIndex;

  Heatmap16BitIndex  depthmapHeatmapIndex16Bit;
  //-----------------------------------
  struct DatabaseList * dbSources;
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

/**
 * @brief Function for testing the library (assuming you have the datasets in your local filesystem).
 * @return Returns an integer representing the test result.
 */
int test();

#ifdef __cplusplus
}
#endif

#endif // DATALOADER_H_INCLUDED
