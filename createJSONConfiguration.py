
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""
import sys
import os
import json
#============================================================================================
def makeJSONConfiguration():
  cfg = {
           'serial'                       : 'unknown',

           'GPUsUsedForTraining'          :["/gpu:0"], #, "/gpu:3"
           'DatasetLoaderThreads'         :30,

           'TrainingDataset'              : [ 
                                              ["datasets/generated/generatedTrain.db", "datasets/generated/data/train",       "datasets/generated/data/depth_train", "datasets/generated/data/segment_train"],
                                              ["datasets/openpose/openposeTrain.db",   "datasets/openpose/data/train",        "datasets/openpose/data/depth_train", "datasets/openpose/data/segment_train"],        #<- for some reason this needs to be first!                           
                                              ["datasets/background/AM-2k.db",         "datasets/background/AM-2k/train",     "datasets/background/AM-2k/depth_train", "datasets/background/AM-2k/segment_train"],                                   
                                              ["datasets/background/BG-20k.db",        "datasets/background/BG-20k/train",    "datasets/background/BG-20k/depth_train", "datasets/background/BG-20k/segment_train"],
                                              ["datasets/coco/cocoTrain.db",           "datasets/coco/cache/coco/train2017",  "datasets/coco/cache/coco/depth_train2017", "datasets/coco/cache/coco/segment_train2017"]
                                            ],

           'ValidationDataset'            : [ ["datasets/coco/cocoVal.db",             "datasets/coco/cache/coco/val2017",    "datasets/coco/cache/coco/depth_val2017", "datasets/coco/cache/coco/segment_val2017"] ],

           'inputWidth'    :400, #256 #Avg 570
           'inputHeight'   :400, #256 #Avg 480
           'outputWidth'   :384, #256
           'outputHeight'  :384,
           'outputChannels':17, # 17 + 2
           'output16BitChannels':1, # 17 + 2
           'outputTokens'  :False,
           'heatmapLossImportanceRelativeToTokens' : 1.0,
           'model'         :'unet', #unet/conv

           'dropoutRate'           : 0.35,
           'activation'            :'leaky_relu', #relu
           'baseChannels'          : 21,# 64 default <- for unet/mobilenet
           'pixelwiseChannels'     : 1200,# 64 default <- for unet/mobilenet
           'bridgeRatio'           : 0.35,# 1.0 default
           'encoderRepetitions'    : 7, # 4 default <- for unet 
           'decoderRepetitions'    : 7, # 4 default <- for unet
           'midSectionRepetitions' : 3, #<- for mobilenet


           'gloveLayers' : 7,
           'multihotLayers' : 3, 
           'learnableTokenResiduals': True,

           'mixedPrecision' : False, #This leads to nan loss during training
           'quantizeModel'  : False,
           'pruneModel'     : False,
           'clusterModel'   : False,

           'RGBMagnitude'        : 255,
           'RGBgaussianNoiseSTD' : 0.0, 
           'RGBImageEncoding'    : 'rgb24',

           'heatmapActive'             :  120, #Max 127 for np.int8
           'heatmapDeactivated'        : -120, #Min -128 for np.int8
           'heatmapGradientSize'       : 23,
           'heatmapGradientSizeMinimum': 6,
           'heatmapPAFSize'            : 5,
           'heatmapPAFSizeMinimum'     : 2, 
           'heatmapReductionEpochLimit': 10,

           'heatmapRuler'              : True, #Add a ruler in first row/column of heatmaps
           'heatmapAddPAFs'            : True, #Add PAFs 
           'heatmapAddDepthmap'        : True, #Add a depth map 
           'heatmapAddNormals'         : True, #Add normals
           'heatmapAddSegmentation'    : True, #Add segmentation
           'heatmapGenerateSkeletonBkg': False, #BKG channel is not just points but also has skeleton
           'heatmapAlternatePattern'   : False,

           'lossWeightJoints'          :2.0,
           'lossWeightPAFs'            :1.0,
           'lossWeightDepth'           :1.0,
           'lossWeightNormals'         :1.0,
           'lossWeightText'            :1.0,
           'lossWeightSegmentation'    :2.0,
           'lossWeightGloveTokens'     :1.0,
           'lossWeightMultihotTokens'  :0.001,

           'streamDataset'     : True,
           'streamValidation'  : False,
           'streamBufferLength': 2000,

           'earlyStoppingMonitor'      : 'val_loss',
           'earlyStoppingHowToMonitor' : 'min',
           'earlyStoppingPatience'     : 45,
           'earlyStoppingMinDelta'     : 0.0001,
           'earlyStoppingStart'        : 25,

           'dataAugmentation' : True,
           'datasetUsage'     : 1.0,
           'learningRate'     : 0.0004,
           'learningRateStart': 0.001,
           'learningRateEnd'  : 0.00015,

           'batchSize'       : 46,
           'epochs'          : 200,
           'loss'            : 'mse', #mse, combine

           'doPostTrainingTraining'           : True,
           'postTrainingEpochs'               : 3,
           'doValidation'                     : True,
           'ignoreNoSkeletonTrainingSamples'  : False, #True
           'logImagesFromValidationSet'       : True,
           'logImagesAlsoOutsideOfTensorboard': False, #Also save heatmap files in current directory

           'keypoint_names': [ 
                      "nose",
                      "left_eye",
                      "right_eye",
                      "left_ear",
                      "right_ear",
                      "left_shoulder",
                      "right_shoulder",
                      "left_elbow",
                      "right_elbow",
                      "left_wrist",
                      "right_wrist",
                      "left_hip",
                      "right_hip",
                      "left_knee",
                      "right_knee",
                      "left_ankle",
                      "right_ankle"
                     ],

           'keypoint_difficulty': [ 
                      -1,#"nose",
                      -1, #"left_eye",
                      -1, #"right_eye",
                      0, #"left_ear",
                      0, #"right_ear",
                      0, #"left_shoulder",
                      0, #"right_shoulder",
                      2, #"left_elbow",
                      2, #"right_elbow",
                      4, #"left_wrist",
                      4, #"right_wrist",
                      0, #"left_hip",
                      0, #"right_hip",
                      2, #"left_knee",
                      2, #"right_knee",
                      4, #"left_ankle",
                      4, #"right_ankle"
                     ],

           'keypoint_parents': { #THIS NEEDS WORK NOSE SHOULD BE CONNECTED TO HIP
                                  "nose": "nose",
                                  "left_eye": "nose",
                                  "right_eye": "nose",
                                  "left_ear": "left_eye",
                                  "right_ear": "right_eye",
                                  "left_shoulder": "nose",
                                  "right_shoulder": "nose",
                                  "left_elbow": "left_shoulder",
                                  "right_elbow": "right_shoulder",
                                  "left_wrist": "left_elbow",
                                  "right_wrist": "right_elbow",
                                  "left_hip": "nose",
                                  "right_hip": "nose",
                                  "left_knee": "left_hip",
                                  "right_knee": "right_hip",
                                  "left_ankle": "left_knee",
                                  "right_ankle": "right_knee"
                              },

           'keypoint_children': {
                                  "nose":      ["left_eye","right_eye","left_shoulder","right_shoulder","left_hip","right_hip"],
                                  "left_eye":  ["left_ear"],
                                  "right_eye": ["right_ear"],
                                  "left_ear":  [],
                                  "right_ear": [],
                                  "left_shoulder":  ["left_elbow"],
                                  "right_shoulder": ["right_elbow"],
                                  "left_elbow": ["left_wrist"],
                                  "right_elbow":["right_wrist"],
                                  "left_wrist": [],
                                  "right_wrist":[],
                                  "left_hip":   ["left_knee"],
                                  "right_hip":  ["right_knee"],
                                  "left_knee":  ["left_ankle"],
                                  "right_knee": ["right_ankle"],
                                  "left_ankle": [],
                                  "right_ankle":[]
                              },
    'paf_parents': [
     0,
     0,
     0,
     0,
     0,
     28,
     19,
     27,
     18,
     26,
     17,
     25,
     22,
     24,
     21,
     23,
     20
    ] 

         }
  return cfg
#============================================================================================
def saveJSONConfiguration(cfg,json_file_path):
  base_path = os.path.dirname(json_file_path)
  if (not os.path.exists(base_path)):
      print("Path ",base_path," does not exist, creating it")
      os.makedirs(base_path)
  with open(json_file_path, 'w') as json_file:
    json.dump(cfg, json_file, indent=4)
  return cfg
#============================================================================================
#Attempt to change paths to ramfs if this is detected, otherwise on normal systems do nothing
#============================================================================================
def redirect_to_ramfs(cfg, ramfs = "../ram/"):
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    # Iterate through each dataset entry
    if os.path.exists(ramfs):
      print("RAMFS detected, checking if it is populated")
      if not os.path.exists("%s/datasets/" % ramfs):
          print("Did not find datasets, running script to copy them (?)")
          os.system("scripts/prepareRAMDatasets.sh")

      for entryID,entry in enumerate(cfg['TrainingDataset']):
        for i in range(len(entry)):
          if ((entry[i]!="enabled") and (entry[i]!="disabled")): #<- hack filter new enable/disabled tag
            # Create new path by prepending base_path
            new_path = os.path.join(ramfs,entry[i])
            
            # Check if the new path exists
            if os.path.exists(new_path):
                # If exists, replace the original path with the new path
                print("Redirecting ",cfg['TrainingDataset'][entryID][i]," to ",new_path)
                cfg['TrainingDataset'][entryID][i] = new_path
            else:
                print(FAIL,"Cannot redirect ",cfg['TrainingDataset'][entryID][i]," to ",new_path,ENDC)
    else:
      print("Did not find a RAMFS, continuing using regular filesystem for datasets")
    return cfg
#============================================================================================
def loadJSONConfiguration(json_file_path,useRAMfs=True):
  cfg = dict()
  # Read the JSON file and load its contents into a dictionary
  with open(json_file_path, 'r') as json_file:
    cfg = json.load(json_file)

  #Populate as index matrix 
  keypoint_names   = cfg['keypoint_names']
  keypoint_parents = cfg['keypoint_parents']
  keypoint_parent_ids = list()
  for i in range(len(keypoint_names)-1): #ignore bkg 
               parent_name  = keypoint_parents[keypoint_names[i]] 
               parent_index = keypoint_names.index(parent_name)
               keypoint_parent_ids.append(int(parent_index))
  cfg['keypoint_parents_ids'] = keypoint_parent_ids

  if not 'bridgeRatio' in cfg:
       cfg['bridgeRatio'] = 1.0

  if not 'heatmapAddDepthLevels' in cfg:
       cfg['heatmapAddDepthLevels'] = 4
  
  #Force user to pick the correct 
  #if not 'learnableTokenResiduals' in cfg:
  #     cfg['learnableTokenResiduals'] = True

  if (useRAMfs):
     print("Using RAMFS")
     cfg = redirect_to_ramfs(cfg)
  else:
     print("Will not use RAMFS (Probably because data loader will automatically do caching)")

  return cfg
#============================================================================================
def createJSONConfiguration(json_file_path):
  cfg = makeJSONConfiguration()
  saveJSONConfiguration(cfg,json_file_path)
  return cfg
#============================================================================================
if __name__ == '__main__':
    if len(sys.argv) != 3 or ( sys.argv[1] != "--label" and sys.argv[1] != "--serial" ):
        print("\n\nWhat is the serial number of this experiment ?")
        print("Correct usage: python3 createJSONConfiguration.py --label serialnumber")
        sys.exit(1)

    serial = sys.argv[2]

    if not serial:
        print("No serial number provided. Exiting...")
        sys.exit(1)

    jsonPath = '2d_pose_estimation/configuration.json'
    print("Creating a new configuration file ( serial ",serial,") in :",jsonPath)

    cfg = makeJSONConfiguration()
    cfg['serial'] = serial
    saveJSONConfiguration(cfg,jsonPath)
