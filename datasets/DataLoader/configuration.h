/** @file configuration.h
 *  @brief  This is the main (static) configuration for the DataLoader
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef CONFIGURATION_H_INCLUDED
#define CONFIGURATION_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/**
-----------------------------
         HEATMAP OUTLINE
-----------------------------
8 BIT  ----------------------
-----------------------------
          Keypoints
-----------------------------
hm  0       head
hm  1       endsite_eye.l
hm  2       endsite_eye.r
hm  3       lear
hm  4       rear
hm  5       lshoulder
hm  6       rshoulder
hm  7       lelbow
hm  8       relbow
hm  9       lhand
hm 10       rhand
hm 11       lhip
hm 12       rhip
hm 13       lknee
hm 14       rknee
hm 15       lfoot
hm 16       rfoot
-----------------------------
           PAFs
-----------------------------
hm 17       10->8         0   #  START_OF_PAF_HEATMAPS
hm 18       8->6          1
hm 19       6->0          2
hm 20       16->14        3
hm 21       14->12        4
hm 22       12->0         5
hm 23       15->13        6
hm 24       13->11        7
hm 25       11->0         8
hm 26       9->7          9
hm 27       7->5          10
hm 28       5->0          11  #  END_OF_PAF_HEATMAPS
-----------------------------
hm 29       Depthmap          <= START_OF_NON_JOINT_HEATMAPS_THAT_NEED_FLIPPING (remember to update in configuration.h if this changes)  db->depthmapHeatmapIndex8Bit
-----------------------------
hm 30       Normal X
hm 31       Normal Y
hm 32       Normal Z
-----------------------------
hm 33       Depthmap > 128      # DEPTH_LEVELS_HEATMAP_START
hm 34       Denosing R          # DENOISING_OUTPUT_HEATMAP_START
hm 35       Denosing G
hm 36       Denosing B
hm 37       Left Disambiguation # LEFT_RIGHT_JOINT_DISAMBIGUATION_HEATMAP_START
hm 38       Right Disambiguation
-----------------------------
hm 39       "Person":                 //1,  // db->segmentationHeatmapIndex = 39
hm 40       "Face":                    2,
hm 41       "Hand":                    3,
hm 42       "Foot":                    4,
hm 43       "Vehicle":                 5,
hm 44       "Animal":                  6,
hm 45       "Robot":                   7,
hm 46       "Label / Text":            8, #Text
hm 47       "Box":                     9, #Object
hm 48       "Tool":                    10,
hm 49       "Instrument":              11,
hm 50       "Appliance / Electronics": 12, #Appliance
hm 51       "Conveyor":                13,
hm 52       "Chair":                   14,
hm 53       "Table":                   15,
hm 54       "Bed":                     16,
hm 55       "Furniture":               17,
hm 56       "Light":                   18, #Lamp
hm 57       "Floor":                   19,
hm 58       "Ceiling":                 20,
hm 59       "Wall":                    21,
hm 60       "Door":                    22,
hm 61       "Window":                  23,
hm 62       "Plant / Vegetation":      24,
hm 63       "Road":                    25,
hm 64       "Dirt":                    26,
hm 65       "Sidewalk":                27,
hm 66       "Building":                28,
hm 67       "Mountain":                29,
hm 68       "Sky":                     30,
hm 69       "Food":                    31,
hm 70       "Fruit":                   32,
hm 71       "Water":                   33,
hm 72       "Cup":                  || 34 ||  = #TOTAL_SEGMENTATION_CLASSES
*/

#define TOTAL_SEGMENTATION_CLASSES 34

/**
-----------------------------
... etc
-----------------------------

         HEATMAP OUTLINE
16 BIT ----------------------
hm  0       depth 16 bit
-----------------------------
*/


#define SEPERATE_SEGMENTATIONS_INTO_DIFFERENT_CHANNELS 1


#define BRANCHLESS_SEGMENTATION_SEPERATION 0 //Test, better to leave it to 0
//It is vulnerable to race condition and will corrupt first heatmap pixel :P

//Multiplexing segmentations is a really good idea that works in practice
// This should reflect the contents of datasets/segmentanything3/run.py
//---------------------------------------------
#if SEPERATE_SEGMENTATIONS_INTO_DIFFERENT_CHANNELS
   #define VALID_SEGMENTATIONS TOTAL_SEGMENTATION_CLASSES
#else
   #define VALID_SEGMENTATIONS 1
#endif // SEPERATE_SEGMENTATIONS_INTO_DIFFERENT_CHANNELS
/*
 CLASSES = {
    "Person":                  1,
    "Face":                    2,
    "Hand":                    3,
    "Foot":                    4,
    #--------------------------------
    "Vehicle":                 5,
    "Animal":                  6,
    "Robot":                   7,
    #--------------------------------
    "Label / Text":            8, #Text
    "Box":                     9, #Object
    "Tool":                    10,
    "Instrument":              11,
    "Appliance / Electronics": 12, #Appliance
    "Conveyor":                13,
    #--------------------------------
    "Chair":                   14,
    "Table":                   15,
    "Bed":                     16,
    "Furniture":               17,
    "Light":                   18, #Lamp
    #--------------------------------
    "Floor":                   19,
    "Ceiling":                 20,
    "Wall":                    21,
    "Door":                    22,
    "Window":                  23,
    #--------------------------------
    "Plant / Vegetation":      24,
    "Road":                    25,
    "Dirt":                    26,
    "Sidewalk":                27,
    "Building":                28,
    "Mountain":                29,
    "Sky":                     30,
    #--------------------------------
    "Food":                    31,
    "Fruit":                   32,
    "Water":                   33,
    "Cup":                     34
    #--------------------------------
}
}*/
//---------------------------------------------


#define STRICT_CHECK_WHEN_OPENING_FILES 1

// ----------------------------------------------------------------------------
//  Heatmap modality TOGGLES / COUNTS  (INPUTS — kept)
// ----------------------------------------------------------------------------
//  These decide *which* modalities exist and *how many* channels each occupies.
//  They feed db_build_heatmap_layout() (DataLoader.c) and calculateNumberOf8BitHeatmaps().
//
//  The per-modality START indices that used to live here
//  (START_OF_PAF_HEATMAPS, END_OF_PAF_HEATMAPS,
//   START_OF_NON_JOINT_HEATMAPS_THAT_NEED_FLIPPING, DEPTH_LEVELS_HEATMAP_START,
//   DENOISING_OUTPUT_HEATMAP_START, LEFT_RIGHT_JOINT_DISAMBIGUATION_HEATMAP_START)
//  have been REMOVED: they were hand-maintained literals that silently drifted
//  when toggles changed. The channel layout is now resolved at runtime from the
//  toggles below into db->...HeatmapIndex fields (single source of truth — see
//  HeatmapLayout.h). For the canonical all-enabled layout see the HEATMAP OUTLINE
//  comment at the top of this file.

//Depth Level Heatmaps (count of depth-threshold channels)
#define DEPTH_LEVELS_HEATMAPS 1

//Enable RGB Denoising Output
#define ENABLE_DENOISING_OUTPUT 1
#define ENABLE_DENOISING_DIFFERENCE_OUTPUT 1

//Left/Right
#define ENABLE_LEFT_RIGHT_JOINT_DISAMBIGUATION_OUTPUT 1

//Per-instance person detection (CenterNet-style center + size head).
//Appended after segmentation; enabled at runtime via db->addInstanceHeatmaps
//(see db_build_heatmap_layout / the "instance" group in HeatmapLayout).
// +0 = person center (Gaussian splat, like a joint)
// +1 = bbox width   (real-valued disk around the center)
// +2 = bbox height  (real-valued disk around the center)
#define INSTANCE_CHANNELS 3
//SEGMENTATION_PLAN §E redesign (2026-06-20): the 3 instance channels are re-tasked for
//bottom-up instance SEGMENTATION (not box regression). +0 = person_blob (large size-adaptive
//Gaussian anchor), +1 = person_lr_bridge (signed left<->right cross-body line), +2 = person_head
//(neck Gaussian anchor). INSTANCE_SIZE_NORM / INSTANCE_SIZE_DISK_RADIUS below are now DEAD (the
//old size-value disk and edge-ridge encodings, Obs 14/17) — kept only so old call sites compile.
#define INSTANCE_SIZE_NORM 100.0f
#define INSTANCE_SIZE_DISK_RADIUS 4
//person_blob Gaussian radius = clamp(INSTANCE_BLOB_RADIUS_FRAC * min(boxW,boxH), MINIMUM_GRADIENT_SIZE, INSTANCE_BLOB_RADIUS_MAX)
#define INSTANCE_BLOB_RADIUS_FRAC 0.25f
#define INSTANCE_BLOB_RADIUS_MAX 40

// Dino descriptor code — dimension is read at runtime from the file header's
// value_count field, so it works for any model (DINOv2=768, DINOv3 variants up
// to 2560+).  MAX_DESCRIPTOR_LENGTH is only used for the static emptyDescriptor
// fallback buffer; set it well above any expected model size.
#define MAX_DESCRIPTOR_LENGTH 4096
#define DINOV2_FEATURES_LENGTH MAX_DESCRIPTOR_LENGTH  // kept for any legacy references
#define USE_DINOV2_FEATURES 1

// L2-normalise every descriptor vector once, at load time (DescriptorLoss assumes unit
// norm; the shipped DINOv3 dumps arrive with ||v|| ~ 14.2, which makes its norm-reg term
// dominate). All-zero vectors (= a source with no .dinov3 sidecar) are left untouched so
// they stay detectable as "missing" instead of turning into NaN.
#define L2_NORMALIZE_DESCRIPTORS 1

// ----------------------------------------------------------------------------
//  Data-augmentation parameters  (MOVED — no longer compile-time defines)
// ----------------------------------------------------------------------------
//  All augmentation chance percentages and magnitude values that used to be
//  #defines here (AUGMENTATION_CHANCE_PERCENT_*, MAXIMUM_REL_ZOOM_FACTOR,
//  PERTURBATION_MAGNITUDE, MAXIMUM_BURNED_PIXELS, MINIMUM/MAXIMUM_BRIGHTNESS_*,
//  COARSE_DROPOUT_*, GAUSSIAN/DEFOCUS/MOTION_BLUR_*) have been moved into
//  struct DataAugmentation (DataAugmentation.h).
//
//  The struct is embedded in struct ImageDatabase as db->aug and is seeded
//  from the same default values by augmentationParamsSetDefaults() in db_create.
//  Fields can be overridden at runtime from Python via
//  db_get/set_augmentation_params() without recompilation.

//Use RAM caching -----
//This can be used instead of the TMPFS mechanism
//For the tmpfs mechanism see useRAMfs in train2DPoseEstimator.py and scripts/prepareRAMDatasets.sh
#define USE_RAM_CACHE 0
//---------------------

//Force data loader to read image files to memory
//Before passing them for decoding, this might improve I/O between
//multiple threads
#define READ_WHOLE_IMAGE_FILE_BEFORE_DECODING 1
//---------------------------------------------

//Dump .log files to check thread concurrency
//This should be off (set to 0) for regular training builds
#define PROFILE_THREAD_ACTIVITY 0
//---------------------------------------------

/** @brief The maximum number of seconds to tolerate waiting for a worker thread, if this passes then DataLoader will raise a SIGABRT */
//------------------------------------------------------
#define MAXIMUM_WORKER_THREAD_TIMEOUT_SECONDS 120 //<- This is set high so that profiling might work without being interrupted

//Common limits for 8/16 bit signed values
//------------------------------------------------------
//Depth (PrepareBatch.c DEPTH PIPELINE): unsigned 16-bit depth - ABS_MINV_16BIT -> signed 16-bit heatmap, later
//re-stretched to [MINV_16BIT, MAXV_16BIT]; the 8-bit depth / normals / depth levels live in [MINV, MAXV].
#define ABS_MINV_16BIT 32767
#define MINV_16BIT -32767
#define MAXV_16BIT  32767
#define MINV -120
#define MAXV 120

#define ABS_MINV 120
#define MAXV_MINUS_MMINV 240
//------------------------------------------------------

//There should be not heatmap below this dimension
#define MINIMUM_GRADIENT_SIZE 4
#define MAXIMUM_GRADIENT_SIZE_DIFFICULTY 6

#define SET_PAF_HEATMAP_BACKGROUND_TO_ZERO 0 //<- This causes training to fail (tried until Epoch 19)
#define FLIP_PAF_GRADIENTS_FOR_LEFT_JOINTS 1 //Legacy PAF encoding only (PAF_ENCODING_CHILD_LIT 0)

//Limb PAF encoding (PLAN.md Observation 29 / Experiment W). 1 = every limb is a ramp from PAF_CHILD_VALUE at
//the child joint to PAF_PARENT_VALUE at the parent, independent of body side, never near the -120 background.
//0 = legacy signed -127..+127 ramp (serials <= 288): right limbs lit only at the parent, dark half == background.
//Changing this needs the matching resolver template (ymapnet/skeletons, ymapnet_cpp/src/skeleton.hpp).
#define PAF_ENCODING_CHILD_LIT 1
#define PAF_CHILD_VALUE  127.0f
#define PAF_PARENT_VALUE  64.0f   //keep > ~40: NNLosses _fn_penalty foreground is y_true > 0.33

//Hip limb PAFs. 1 = left/right hip PAF goes to the same-side shoulder; 0 = to the nose, as in the dataset joint
//tables (serials <= 288). With the nose as the only torso hub, a person without an annotated nose (back views,
//~30% of COCO val persons) has no PAF joining upper and lower body; on perfect GT heatmaps the resolver goes
//0.672 -> 0.803 COCO AP with this on (ymapnet/skeletons). Channel count is unchanged; configuration.json must say
//"lhip->lshoulder" / "rhip->rshoulder" and keypoint_parents left_hip:left_shoulder, right_hip:right_shoulder.
#define PAF_HIPS_TO_SHOULDERS 1

//person_lr_bridge encoding. 0 = signed -127 (left joint) -> +127 (right joint), the left half indistinguishable from
//the -120 background; 1 = lit along the whole line like PAF_ENCODING_CHILD_LIT, PAF_CHILD_VALUE at the left joint ->
//PAF_PARENT_VALUE at the right one. The bridges are all that joins the two sides of a person without an annotated
//nose once hips hang off the shoulders: on DataLoader GT (cocoVal) 1 halves the left/right-swapped skeletons of
//nose-less persons (67 -> 30) and lifts the resolver 0.827 -> 0.843 COCO AP. configuration.json must say
//"heatmapBridgeEncoding": "child_lit" (1) or "signed" (0) for the resolver template (ymapnet/skeletons, skeleton.hpp).
#define BRIDGE_ENCODING_LIT 1

//Limb PAF whose parent joint is not annotated (a wrist without its elbow, a knee without its hip, ...). 0 = no PAF
//(the joint can only be grouped geometrically); 1 = draw it to the grandparent instead (wrist -> shoulder,
//ankle -> hip, knee -> shoulder, hip/elbow -> nose) in the same channel with the same ramp. ~0.8% of COCO val PAF
//joints, 5.6% of persons. Needs the resolver's skip limbs (ymapnet/skeletons skip_limbs).
#define PAF_SKIP_MISSING_PARENT 0

//db_shuffle_indices_via_loss: samples are sorted by their average batch loss, but every sample of a batch gets
//that batch's loss, so batch-mates tie and an exact sort would serve the same batches every epoch. Each sorted
//rank is jittered by up to +/- this many places before the final order (~1-2% of COCO train), which mixes
//equal-loss neighbours into different batches while keeping the order by loss.
#define LOSS_SHUFFLE_RANK_JITTER 2048

//Switch dumping 16 bit heatmaps when using test
#define TEST_16BIT_HEATMAPS 1








/*
OLD Detectron 2 class multiplexing..
*/

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

#define MAX_SEGMENTATION_VALUE_MULTIPLEXING 183
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














#ifdef __cplusplus
}
#endif

#endif // DATALOADER_H_INCLUDED
