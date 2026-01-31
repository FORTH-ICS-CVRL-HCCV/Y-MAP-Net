#!/usr/bin/python3
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""
#Dependencies should be :
#tensorflow-2.16.1 needs CUDA 12.3, CUDNN 8.9.6 and is built with Clang 17.0.6 Bazel 6.5.0
#python3 -m pip install tf_keras tensorflow==2.16.1 numpy tensorboard opencv-python wget
#----------------------------------------------
import sys
import os
import time
import gc
import math
import numpy as np
import datetime
#----------------------------------------------
useGPU = True
if (len(sys.argv)>1):
       #print('Argument List:', str(sys.argv))
       for i in range(0, len(sys.argv)):
           if (sys.argv[i]=="--cpu"):
             useGPU = False
# Set CUDA_VISIBLE_DEVICES to an empty string to force TensorFlow to use the CPU
if (not useGPU):
     os.environ['CUDA_VISIBLE_DEVICES'] = '' # <- Force CPU
#----------------------------------------------
try:
 import cv2
 import tensorflow as tf
 import keras
 from keras import callbacks
 from keras.callbacks import TensorBoard
 from keras import layers, models
 from keras.models import Sequential

 #C Dataloader
 sys.path.append('datasets/DataLoader')
 from DataLoader import DataLoader

 from NNConverter import saveNNModel
except Exception as e:
 print("An exception occurred:", str(e))
 print("Issue:\n source venv/bin/activate")
 print("Before running this script")
 sys.exit(1)
#-------------------------------------------------------------------------------
from NNLosses import RSquaredMetric,HeatmapDistanceMetric,HeatmapCoreLoss,VanillaMSELossSimple,AdamWCautious
from tools import bcolors,read_json_file,checkIfPathExists,checkIfFileExists,convert_bytes
#-------------------------------------------------------------------------------
def deriveRGBChannelsFromCFG(cfg):
    numberOfChannels = 3

    if ("inputChannels" in cfg):
        numberOfChannels = cfg["inputChannels"]
        return numberOfChannels

    if (cfg['RGBImageEncoding']=='rgb8'):
        numberOfChannels = 1
    if (cfg['RGBImageEncoding']=='rgb16'):
        numberOfChannels = 2
    return numberOfChannels
#-------------------------------------------------------------------------------
def createSelectedModel(cfg,testModel=True):
   channels = deriveRGBChannelsFromCFG(cfg)
   if (cfg['model']=='unet'):
                     from NNModel import build_unet

                     numTokens = 0
                     if ("outputTokens" in cfg) and (cfg["outputTokens"]):
                          if ("tokensOut" in cfg):
                              numTokens = int(cfg["tokensOut"])

                     model = build_unet(
                                          cfg['inputHeight'],
                                          cfg['inputWidth'],
                                          channels,
                                          cfg['outputWidth'],
                                          cfg['outputHeight'],
                                          cfg['outputChannels'],
                                          numTokens  = numTokens,
                                          numClasses = cfg['tokensClasses'],
                                          minHeatmapValue    = cfg['heatmapDeactivated'],
                                          maxHeatmapValue    = cfg['heatmapActive'],
                                          baseChannels       = cfg['baseChannels'],
                                          pixelwiseChannels  = cfg['pixelwiseChannels'],
                                          encoderRepetitions = cfg['encoderRepetitions'],
                                          decoderRepetitions = cfg['decoderRepetitions'],
                                          midSectionRepetitions = cfg['midSectionRepetitions'],
                                          gloveLayers        = cfg['gloveLayers'],
                                          multihotLayers     = cfg['multihotLayers'],
                                          use_learnable_residuals = cfg['learnableTokenResiduals'],
                                          bridgeRatio        = cfg['bridgeRatio'],
                                          forceBridgeSize    = cfg['forceBridgeSize'],
                                          activation         = cfg['activation'],
                                          gaussianNoiseSTD   = cfg['RGBgaussianNoiseSTD'],
                                          dropoutRate        = cfg['dropoutRate'],
                                          nextTokenStrength  = cfg['nextTokenStrength'],
                                          quantize           = cfg['quantizeModel'],
                                          serial             = cfg['serial'],
                                          useDescriptors     = cfg['outputDescriptors']
                                        )
   else:
        print("ERROR, incorrect model type ",cfg['model'])
        sys.exit(1)

   if (testModel):
     from NNModel import test_model_IO
     test_model_IO(model)
   return model
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#============================================================================================
#============================================================================================
#============================================================================================
from NNTraining import logTrainingHistory,logText,logSomeInputsAndOutputs,printTFVersion, TrainingDataGenerator, getOptimizerFromCFG, custom_lr_scheduler, custom_lr_schedulerWarmup, DataAugmentation, weighted_token_loss, extract_validation_losses
#============================================================================================
#============================================================================================
# Main Function
#============================================================================================
#============================================================================================
if __name__ == '__main__':
   # Test the custom learning rate scheduler
   #for epoch in range(1, 200):
   #  print(f"Epoch {epoch}: Learning Rate = {custom_lr_scheduler(epoch)}")
   #sys.exit(0)
   print("Ensuring the same seed for reproducible results always..\n")
   from numpy.random import seed
   seed(1)
   tf.random.set_seed(2)

   #Have dataset variables on main scope
   dataset_generator = None
   shuffleData       = True
   dbTrain           = None
   dbValidation      = None
   testModel         = True
   useRAMfs          = True

   queueSize         = 32


   jsonPath = '2d_pose_estimation/configuration.json'
   if (checkIfFileExists(jsonPath)):
        print(bcolors.OKGREEN,"Loading configuration from file ",jsonPath,bcolors.ENDC)
        from createJSONConfiguration import loadJSONConfiguration
        cfg = loadJSONConfiguration(jsonPath,useRAMfs=useRAMfs)
   else:
        print(bcolors.FAIL,"CREATING FRESH CONFIGURATION!",bcolors.ENDC)
        from createJSONConfiguration import createJSONConfiguration
        cfg = createJSONConfiguration(jsonPath,useRAMfs=useRAMfs)

   ourDType=tf.float32
   if (cfg['mixedPrecision']):
      print(bcolors.WARNING,"Using mixed precision mode!",bcolors.ENDC)
      ##policy = keras.mixed_precision.Policy('mixed_float16')
      #policy = keras.mixed_precision.Policy('float16')
      #keras.mixed_precision.set_global_policy(policy)
      #print(bcolors.WARNING,'Compute dtype:', policy.compute_dtype,bcolors.ENDC)  # Should print 'float16'
      #print(bcolors.WARNING,'Variable dtype:', policy.variable_dtype,bcolors.ENDC)  # Should print 'float32'
      #ourDType=tf.float16
      from tensorflow.keras import mixed_precision
      # Enables mixed precision (e.g., float16 on GPU)
      mixed_precision.set_global_policy("mixed_float16")



    #Before starting training log TF Versions
   printTFVersion()

   saveRestoredWeights    = False
   restoreBestWeights     = False
   resumePreviousTraining = False
   packageTrainedModel    = True
   elevatePriority        = False
   startEpoch             = 0
   if (len(sys.argv)>1):
       print('Argument List:', str(sys.argv))
       for i in range(0, len(sys.argv)):
           if (sys.argv[i]=="--debug"):
              #Enabling this multiplies training time from 6min to 1hour
              tf.config.run_functions_eagerly(True) # <- DEBUG
              cfg['batchSize']=4#<- Reduce batch size since eager executio requires more memory
           if (sys.argv[i]=="--flush"):
              os.system("rm -rf 2d_pose_estimation/ && mkdir 2d_pose_estimation/")
           if (sys.argv[i]=="--novalidation"):
              cfg['doValidation']=False
           if (sys.argv[i]=="--rt"):
              elevatePriority = True
           if (sys.argv[i]=="--nopackage"):
              print("Disabling packaging of model at the end")
              packageTrainedModel = False
           if (sys.argv[i]=="--start"):
              startEpoch=int(sys.argv[i+1])
           if (sys.argv[i]=="--queue"):
              queueSize=int(sys.argv[i+1])
              print("Set queue size to ",queueSize)
           if (sys.argv[i]=="--mem"):
              cfg['datasetUsage']=float(sys.argv[i+1])
           #if (sys.argv[i]=="--stream"): Model so large that we always need to stream now
           #   cfg['streamDataset']      = True
           #   cfg['streamBufferLength'] = 1000 #int(sys.argv[i+1])
           if (sys.argv[i]=="--clear") or (sys.argv[i]=="--clean"):
              os.system("rm -rf 2d_pose_estimation/tensorboard")
              os.system("rm 2d_pose_estimation.zip")
           if (sys.argv[i]=="--resume") or (sys.argv[i]=="--continue"):
              resumePreviousTraining = True
              cfg['earlyStoppingStart'] = 0
           if (sys.argv[i]=="--restoreBestWeights") or (sys.argv[i]=="--restore"):
              restoreBestWeights = True
              resumePreviousTraining = False
              saveRestoredWeights    = False
              cfg['earlyStoppingStart'] = 0  
           if (sys.argv[i]=="--saveRestoredWeights") or (sys.argv[i]=="--save"):
              restoreBestWeights = True
              resumePreviousTraining = False
              saveRestoredWeights = True
           if (sys.argv[i]=="--skiptest") or (sys.argv[i]=="--notest"):
              print("Will not save model upon creating to check its I/O")
              testModel=False
           if (sys.argv[i]=="--test"):
              model = createSelectedModel(cfg,testModel=True)
              from NNModel import retrieveModelOutputDimensions 
              retrieveModelOutputDimensions(model) 
              sys.exit(0)
              tf.saved_model.save(model,'test_model')
              model.save('test_model/model.keras')
              sys.exit(0)

   #--------------------------------------------------------------------------------------------------------------------------------
   #if ("outputTokens" in cfg) and (cfg["outputTokens"]):
   #           print(bcolors.FAIL,"Disabling validation until it is fixed..",bcolors.ENDC)
   #           cfg['doValidation']=False
   if (cfg['loss']=="combine"):
              print(bcolors.FAIL,"Disabling validation when using combined loss..",bcolors.ENDC)
              cfg['doValidation']=False

   if (cfg['outputChannels']>cfg['baseChannels']): 
              print(bcolors.FAIL,"Base channels should at least be the same as output channels..",bcolors.ENDC)
              print(bcolors.FAIL,"go to 2d_pose_estimator/configuration.json and make \"baseChannels\": ",cfg['outputChannels'],bcolors.ENDC)
              #sys.exit(1)

   #Shorthand for number of GPUs used
   numberOfGPUs = len(cfg['GPUsUsedForTraining'])

   #https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/distributed_training.ipynb#scrollTo=nbGleskCACv_
   #https://stackoverflow.com/questions/75403101/how-do-i-distribute-datasets-between-multiple-gpus-in-tensorflow-2
   #This causes : https://github.com/tensorflow/tensorflow/commit/4924ec6c0b68ba3fb8f73a6383881cd4194ed802

   #Comment out start
   """ #<------------  Comment out if you want to use multiple GPUs
   if (numberOfGPUs>1):
       print(bcolors.OKGREEN,"Using mirrored gpu strategy for ",numberOfGPUs," GPUs (",cfg['GPUsUsedForTraining'],")..",bcolors.ENDC)
       batch_size =int( int(cfg["batchSize"]) * (numberOfGPUs) )#numberOfGPUs
       #batch_size = batch_size - 3 *numberOfGPUs # Remove some things to make sure there is enough space for overheads
       cfg["batchSize"] = batch_size #<= try 3x batch size since batch_size is global

       if batch_size % numberOfGPUs != 0:
         suggested = ((batch_size // numberOfGPUs) + 1)
         raise ValueError(
            f"Batch size {batch_size} is not divisible by {numberOfGPUs} GPUs.\n"
            f"Suggested batch sizes: {numberOfGPUs}, {2*numberOfGPUs}, ..., or try {suggested}.")
       strategy = tf.distribute.MirroredStrategy(devices=cfg['GPUsUsedForTraining'])
   elif (numberOfGPUs==1):
       strategy = tf.distribute.OneDeviceStrategy(device=cfg['GPUsUsedForTraining'][0])
   else:
       strategy = tf.distribute.OneDeviceStrategy()

   with strategy.scope():
        """
   if (numberOfGPUs>0): 
        #""" <------------ Uncomment If you want to use multiple GPUS

        #First of all create the Neural Network model
        #Don't load other data in vain if this step fails due to bad configuration..
        #----------------------------------------------------------------------------------------
        if (resumePreviousTraining):
            print(bcolors.OKGREEN,"Continuing training from pretrained network.. ",bcolors.ENDC)
            model_path = "2d_pose_estimation/model.keras"
            from NNModel import load_keypoints_model
            model,input_size,output_size,numHeatmaps = load_keypoints_model(model_path)
            cfg['inputWidth']   = input_size[0]
            cfg['inputHeight']  = input_size[1]
            cfg['outputWidth']  = output_size[0]
            cfg['outputHeight'] = output_size[1]
        else:
            print(bcolors.OKGREEN,"Creating a new Neural Network.. ",bcolors.ENDC)
            model = createSelectedModel(cfg,testModel=testModel)
            from NNModel import retrieveModelOutputDimensions
            cfg['outputWidth'],cfg['outputHeight'],numHeatmaps = retrieveModelOutputDimensions(model) 
            if (restoreBestWeights):
                print(bcolors.OKGREEN,"Restoring best weights.. ",bcolors.ENDC)
                model.load_weights("best.weights.h5")
                if (saveRestoredWeights):
                    print(bcolors.OKGREEN,"Saving best weights as model.. ",bcolors.ENDC)
                    saveNNModel("2d_pose_estimation",model,formats=["keras"]) #Only use keras format to save space! formats=["keras","tf","tflite","onnx"]
                    if (packageTrainedModel):
                       print(bcolors.OKGREEN,"Packaging model without doing any training and exiting ..",bcolors.ENDC)
                       os.system("rm 2d_pose_estimation.zip") #Make sure there is no zip file
                       os.system("zip -r 2d_pose_estimation.zip 2d_pose_estimation/") #Create zip of models
                       print("scp -P 2222 2d_pose_estimation.zip ammar@ammar.gr:/home/ammar/public_html/poseanddepth/archive/2d_pose_estimation_v%s.zip" % str(cfg['serial'])) 
                    sys.exit(0)
        #----------------------------------------------------------------------------------------


        #Command line parameter modifiers after loading the model
        #----------------------------------------------------------------------------------------
        transplantTokenInitialization = False
        if ('transplantTokenInitialization' in cfg):
            transplantTokenInitialization = cfg['transplantTokenInitialization']
        if (len(sys.argv)>1):
         #print('Argument List:', str(sys.argv))
         for i in range(0, len(sys.argv)):
           if (sys.argv[i]=="--trans"):
               transplantTokenInitialization = True           


        if transplantTokenInitialization: 
              from NNTransplant import transplantLayersFromSourceToTarget,load_any_model
              layerNameList = [ ]

              for i in range(int(65)):
                  layerNameList.append("layer_normalization_%u"%i)

              for i in range(int(cfg["tokensOut"])):
                  #layerNameList.append("Y_l1_t%u"%i) <- This has different dimensions 

                  if (i>0):
                    layerNameList.append("Y_l2X_t%u"%i)
                  else:
                    layerNameList.append("Y_l2F_t%u"%i)

                  layerNameList.append("Y_l3_t%u"%i)
                  #layerNameList.append("Y_rescale_t%u"%i) # <- This doesn't have weights
                  
                  layerNameList.append("Y_l4_t%u"%i)
                  layerNameList.append("Y_l5_t%u"%i)
                  layerNameList.append("Y_l6_t%u"%i)
                  layerNameList.append("Y_l7_t%u"%i)

                  if (i>0):
                    layerNameList.append("Y_pre_final_combined_t%u"%i)
                  else:
                    layerNameList.append("Y_pre_final_t%u"%i)

                  #layerNameList.append("res_factor_prev_t%u"%i)
                  layerNameList.append("t%02u"%i)
              
              for i in range(int(cfg["multihotLayers"])):
                  layerNameList.append(f"tm{i}") #<- may emmit ValueError: Layer tmh1 weight shape (1500, 2048) is not compatible with provided weight shape (2400, 2048).
              layerNameList.append("tokens_multihot")
 
              print(bcolors.OKGREEN,"Loading Token Neural Network.. ",bcolors.ENDC)
              tokenOnlyModel = load_any_model("2d_pose_estimation/tokens/model.keras")
              print(bcolors.OKGREEN,"Transplanting Token Weights to Main Network.. ",bcolors.ENDC)
              model = transplantLayersFromSourceToTarget(model, tokenOnlyModel, layerNameList)
              print(bcolors.OKGREEN,"Removing Token Neural Network from memory.. ",bcolors.ENDC)
              del tokenOnlyModel
        #----------------------------------------------------------------------------------------

        # Set up TensorBoard logging
        #----------------------------------------------------------------------------------------
        log_dir = "2d_pose_estimation/tensorboard/" + cfg["serial"] + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        if (cfg['streamDataset']):
           mem=1.0 #When streaming use everything..

        #Validation data is not so big and is loaded first in memory..
        onlyTrainingData = True

        #Prepare Validation Data
        #----------------------------------------------------------------
        if cfg['doValidation']: #(checkIfFileExists(cfg['COCOValidationJSONPath']) and
             onlyTrainingData = False
             dbValidation = DataLoader(
                                       (cfg['inputHeight'], cfg['inputWidth'],cfg['inputChannels']),
                                       (cfg['outputHeight'],cfg['outputWidth'],cfg['outputChannels']),
                                       output16BitChannels = cfg['output16BitChannels'] ,
                                       numberOfThreads = cfg['DatasetLoaderThreads'],
                                       streamData      = int(cfg['streamValidation']), #0, <- set to 0 to keep it in memory and speed up validation 
                                       batchSize       = cfg['batchSize'],
                                       gradientSize    = cfg['heatmapGradientSizeMinimum'],# <- Use final size! cfg['heatmapGradientSize'],
                                       PAFSize         = cfg['heatmapPAFSizeMinimum'],     # <- Use final size! cfg['heatmapPAFSizeMinimum'],
                                       doAugmentations = 0,# <- Don't do augmentations on test set, keep it clean
                                       addPAFs         = int(cfg['heatmapAddPAFs']),
                                       addBackground   = int(cfg['heatmapGenerateSkeletonBkg']),
                                       addDepthMap     = int(cfg['heatmapAddDepthmap']),
                                       addDepthLevelsHeatmaps  = int(cfg['heatmapAddDepthLevels']),
                                       addNormals      = int(cfg['heatmapAddNormals']),
                                       addSegmentation = int(cfg['heatmapAddSegmentation']),
                                       datasets        = cfg["ValidationDataset"],
                                       elevatePriority = elevatePriority,
                                       libraryPath     = "datasets/DataLoader/libDataLoader.so")
             if (cfg['streamValidation']):
                  # Modify the way you call the dataset
                  validation_generator    = TrainingDataGenerator(cfg=cfg, db=dbValidation, batch_size=cfg['batchSize'], validation_data=True, numberOfTokens=cfg["tokensOut"], numberOfClasses=cfg["tokensClasses"],    
                                                                  workers=1, use_multiprocessing=False, max_queue_size=1) #multiprocessing happens inside the dataloader
                  validationDatasetLength = dbValidation.numberOfSamples
                  outValLabels            = dbValidation.get_labels()
             else:
                  print(bcolors.WARNING,"Only streaming validation data is supported any more to simplify code..!",bcolors.ENDC)
                  sys.exit(1)
        #----------------------------------------------------------------


        #The training set is very large, so depending on the system there are two ways to use it
        #try streaming it which is very slow due to I/O operations but can work on small VRAM systems
        #or load it all in memory and use regular TF mechanisms to train on it
        #----------------------------------------------------------------
        if True: #(checkIfFileExists(cfg['COCOTrainingJSONPath'])):
             dbTrain = DataLoader(
                                       (cfg['inputHeight'],cfg['inputWidth'],cfg['inputChannels']),
                                       (cfg['outputHeight'],cfg['outputWidth'],cfg['outputChannels']),
                                       output16BitChannels = cfg['output16BitChannels'] ,
                                       #numberOfSamples = 10000, <- uncomment to just use a few samples for test
                                       streamData      = int(cfg['streamDataset']),
                                       batchSize       = cfg['batchSize'],
                                       numberOfThreads = cfg['DatasetLoaderThreads'],
                                       gradientSize    = cfg['heatmapGradientSize'],
                                       PAFSize         = cfg['heatmapPAFSize'],
                                       doAugmentations = int(cfg['dataAugmentation']),
                                       addPAFs         = int(cfg['heatmapAddPAFs']),
                                       addBackground   = int(cfg['heatmapGenerateSkeletonBkg']),
                                       addDepthMap     = int(cfg['heatmapAddDepthmap']),
                                       addDepthLevelsHeatmaps  = int(cfg['heatmapAddDepthLevels']),
                                       addNormals      = int(cfg['heatmapAddNormals']),
                                       addSegmentation = int(cfg['heatmapAddSegmentation']),
                                       datasets        = cfg["TrainingDataset"],
                                       elevatePriority = elevatePriority,
                                       libraryPath     = "datasets/DataLoader/libDataLoader.so")

             print("Joint gradient sizes ",cfg['heatmapGradientSize']," -> ",cfg['heatmapGradientSizeMinimum'])
             dbTrain.updateJointDifficulty(cfg['keypoint_difficulty'])
             if (cfg['streamDataset']):
                  # Modify the way you call the dataset
                  outLabels             = dbTrain.get_labels()
                  dataset_generator     = TrainingDataGenerator(cfg=cfg, db=dbTrain, batch_size=cfg['batchSize'], labels=outLabels, numberOfTokens=cfg["tokensOut"], numberOfClasses=cfg["tokensClasses"],
                                                                log_dir=log_dir, workers=1, use_multiprocessing=False, max_queue_size=queueSize) #multiprocessing happens inside the dataloader
                  trainingDatasetLength = dbTrain.numberOfSamples
                  shuffleData           = False #<- shuffling is done inside the generator
             else:
                  shuffleData           = True 
                  print(bcolors.WARNING,"Only streaming training data is supported any more to simplify code..!",bcolors.ENDC)
                  sys.exit(1)
        #----------------------------------------------------------------

        steps_per_epoch = dbTrain.numberOfSamples // int(cfg['batchSize'])  

        # Print the shapes of inputs and outputs
        print("Training Configuration :", cfg)
        logText(cfg,log_dir)

        # Define Learning Rate Scheduler callback
        #lr_callback = keras.optimizers.schedules.CosineDecay(initial_learning_rate=cfg['learningRate'], decay_steps=cfg['epochs'] * steps_per_epoch)
        #lr_callback = tf.keras.callbacks.LearningRateScheduler(custom_lr_scheduler) <- This is not configurable
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: custom_lr_scheduler(epoch,cfg['learningRateStart'],cfg['learningRateEnd']))

        #Early Stopping / Checkpointing
        whatToMonitor = cfg['earlyStoppingMonitor']
        howToMonitor  = cfg['earlyStoppingHowToMonitor']
        
        if (onlyTrainingData) and (cfg['earlyStoppingMonitor']=="val_loss"):
             print("You forgot to change the monitor to loss, fixing this automatically")
             cfg['earlyStoppingMonitor']="loss"

        # Define EarlyStopping/ModelCheckpoint callbacks
        #-------------------------------------------------------------------------
        early_stopping = keras.callbacks.EarlyStopping(
                                                        monitor = whatToMonitor,      mode = howToMonitor,  
                                                        #monitor  = 'loss',           mode = 'min', # Monitor the Training loss metric / Mode should be 'min' because we want to minimize the loss metric
                                                        #monitor = 'val_loss',        mode = 'min', # Monitor the Validation loss metric / Mode should be 'min' because we want to minimize the loss metric
                                                        #monitor  = 'hdm',            mode = 'max', # Monitor the Training HDM metric / Mode should be 'max' because we want to maximize the  metric
                                                        #monitor = 'val_hdm',         mode = 'max', # Monitor the Validation HDM metric / Mode should be 'max' because we want to maximize the  metric
                                                        patience             = cfg['earlyStoppingPatience'],  # Number of epochs with no improvement after which training will be stopped
                                                        min_delta            = cfg['earlyStoppingMinDelta'],  # Minimum change in the monitored quantity to qualify as an improvement
                                                        verbose              = 1,                             # Set to 1 for more verbose output
                                                        restore_best_weights = True,                          # Restore model weights from the epoch with the best value of the monitored quantity
                                                        start_from_epoch     = cfg['earlyStoppingStart']      #After which epoch to start early stopping!
                                                       )
        #-------------------------------------------------------------------------
        from NNLosses import ConditionalModelCheckpoint
        checkpointer = ConditionalModelCheckpoint(
                                                  monitor=whatToMonitor,
                                                  mode=howToMonitor,
                                                  filepath          = "best.weights.h5",
                                                  #filepath="checkpoint_epoch_{epoch:02d}.weights.h5",  # Corrected file extension
                                                  save_best_only=True,
                                                  save_weights_only=True,
                                                  start_from_epoch=cfg['earlyStoppingStart'],
                                                  verbose=1
                                                 )
        #-------------------------------------------------------------------------
        # Create a distributed dataset from the tensorflow datasets
        #--------------------------------------------------------------------------------------------------------------------------------
        #----trainingDataset            = trainingDataset.shuffle(100 * cfg['batchSize']).repeat(cfg['epochs']).batch(numberOfGPUs * cfg['batchSize'], drop_remainder=True)
        #----validationDataset          = validationDataset.repeat(cfg['epochs']).batch(numberOfGPUs * cfg['batchSize'], drop_remainder=True)

        #If strategies are restored this needs to be restored ->
        #distributedTrainingDataset = strategy.experimental_distribute_dataset(trainingDataset)

        # Create extra Metrics to have a better grasp of what is happening with the model
        OUTPUT_MAGNITUDE = float( float(cfg['heatmapActive']) - float(cfg['heatmapDeactivated']) )
        #--
        HDM_THRESHOLD   = 0.01 # 1%
        HDM_THRESHOLD   = OUTPUT_MAGNITUDE * HDM_THRESHOLD # <- translate the percentage to an absolute value, commenting out this line reverts to old counts
        #--
        hdm_metric      = HeatmapDistanceMetric(threshold=HDM_THRESHOLD)
        hdm16bit_metric = HeatmapDistanceMetric(name='hdm16', scale=120.0/32767.0, threshold= HDM_THRESHOLD) #(32767.0/120.0) * HDM_THRESHOLD
        #rsq_metric = RSquaredMetric()

        # Initialize the Adam Optimizer using configuration
        #optimizer = AdamWCautious(learning_rate=float(cfg['learningRate']),clipnorm=None,clipvalue=1.0)
        #optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['learningRate'])
        optimizer = getOptimizerFromCFG(cfg)

        #Decide on heatmap loss based on configuration
        hmloss = None
        if (cfg['loss']=="mse"):
             baseScale8b  = cfg['lossBaseHeatmapScale']/120.0
             baseScale16b = cfg['lossBaseHeatmapScale']/32767.0
             hmloss =  HeatmapCoreLoss(
                                      scale             = baseScale8b,
                                      jointGain         = cfg['lossWeightJoints'],
                                      PAFGain           = cfg['lossWeightPAFs'],
                                      DepthGain         = cfg['lossWeightDepth'],
                                      NormalGain        = cfg['lossWeightNormals'],
                                      TextGain          = cfg['lossWeightText'],
                                      SegmentGain       = cfg['lossWeightSegmentation'],
                                      DistanceLevelGain = cfg['lossWeightDepthLevels'],
                                      DenoisingGain     = cfg['lossWeightDenoising'],
                                      PenaltyGain       = cfg['lossPenaltyGain'],
                                      leftRightGain     = cfg['lossLeftRightGain']
                                     )
             hmloss16b =  VanillaMSELossSimple(scale=baseScale16b)
        elif (cfg['loss']=="combine"):
             print(bcolors.WARNING,"Using experimental combined loss..",bcolors.ENDC)
             hmloss = combined_loss
        elif (cfg['loss']=="dssim"):
             print(bcolors.WARNING,"Using experimental dssim loss..",bcolors.ENDC)
             hmloss=dssim_loss 
        else:
             print(bcolors.WARNING,"Using generic loss, will probably not work well..",bcolors.ENDC)
             hmloss=cfg['loss']
             hmloss16b=cfg['loss']


        #Compile a model with the requested loss
        if ("outputTokens" in cfg) and (cfg["outputTokens"]):
             print("Now retrieving token weights!")
             weight_array = dbTrain.get_token_fequencies()
             class_weight_dict = {i: float(weight) for i, weight in enumerate(weight_array)}
                       
             #-------------------------------------------------------------------------------------
             from NNLosses import GloVeMSELoss, MultiHotLoss, WeightedBinaryCrossEntropy, WeightedFocalLoss
             losses = dict()
             #for i in reversed(range(cfg["tokensOut"])): #<- Try reversing order(?)
             for i in range(cfg["tokensOut"]):
                         #Applying a higher weight to the losses for earlier tokens during training. 
                         #This should encourage the network to focus more on improving the accuracy of the earlier tokens by making their losses more prominent.
                         #losses["t%02u"%i] = GloVeMSELoss(weight=10.0/((i+1) * (i+1)))
                         losses["t%02u"%i] = GloVeMSELoss(weight=cfg["lossWeightGloveTokens"] / cfg["tokensOut"])
             #losses['tokens_multihot']  = WeightedBinaryCrossEntropy(weight_array, weight=cfg["lossWeightMultihotTokens"])  #MultiHotLoss()
             losses['tokens_multihot']  = WeightedFocalLoss(weight_array, weight=cfg["lossWeightMultihotTokens"])  #<- Try focal loss
             losses['hm']     = hmloss
             losses['hm_16b'] = hmloss16b
             #-------------------------------------------------------------------------------------
             #This is no longer used, instead of set to 1.0 we pass a None 
             loss_weights = None
             #loss_weights = dict()
             #for i in range(cfg["tokensOut"]):  
             #            loss_weights["t%02u"%i] = 1.0 #cfg["lossWeightGloveTokens"] / cfg["tokensOut"]
             #loss_weights['tokens_multihot']  = 1.0 #cfg["lossWeightMultihotTokens"]
             #loss_weights['hm'] = 1.0  
             #-------------------------------------------------------------------------------------
             from NNLosses import HeatmapDistanceMetric, HeatmapDistanceMetricPartial, NonZeroCorrectPixelMetric, CustomTopKCategoricalAccuracy
             metrics = dict()

             from NNLosses import TopKAccuracyMetric,CosineSimilarityMetric
             #if (not cfg['mixedPrecision']): #Mixed precision nowadays may also have float32 dtype
             if (ourDType==tf.float32): #More relaxed check
                for i in range(cfg["tokensOut"]):
                         #metrics["t%02u"%i] = keras.metrics.CosineSimilarity(name='cossim', dtype=ourDType, axis=1) #<- TODO: implement my own version at some point to fix float16 compat
                         metrics["t%02u"%i] = CosineSimilarityMetric(name='cossim', dtype=ourDType, axis=1) #<- TODO: implement my own version at some point to fix float16 compat
 
                metrics['tokens_multihot']   = [ 
                                                 'accuracy', 
                                                 TopKAccuracyMetric(name="top3_accuracy",k=3, dtype=ourDType),   #<- Custom topk metric compatible with fp16
                                                 TopKAccuracyMetric(name="top5_accuracy",k=5, dtype=ourDType)    #<- Custom topk metric compatible with fp16
                                                 #tf.keras.metrics.TopKCategoricalAccuracy(name="top3_accuracy",k=3, dtype=ourDType), 
                                                 #tf.keras.metrics.TopKCategoricalAccuracy(name="top5_accuracy",k=5, dtype=ourDType)
                                               ]
             else:
                print(bcolors.WARNING,"Disabling captioning metrics since they are not currently compatible with non float32 training :( ",bcolors.ENDC)
                #metrics['tokens_multihot']   = [ CustomTopKCategoricalAccuracy(name="top3_accuracy",k=3), CustomTopKCategoricalAccuracy(name="top5_accuracy",k=5) ] #<- TODO: implement my own version to fix float16 compat


             metrics['hm_16b'] = [
                                     HeatmapDistanceMetric(name="hdm_16b_depth", threshold=HDM_THRESHOLD, scale=120.0/32767.0)
                                 ]

             metrics['hm'] = [
                              hdm_metric,
                              NonZeroCorrectPixelMetric(name="hdm_not0",         accuracyThreshold=HDM_THRESHOLD,start=0),
                              HeatmapDistanceMetricPartial(name="hdm_joints",    threshold=HDM_THRESHOLD,start=0, end=17),
                              NonZeroCorrectPixelMetric(name="hdm_not0_joints",  accuracyThreshold=HDM_THRESHOLD,start=0,end=17),
                              HeatmapDistanceMetricPartial(name="hdm_PAFs",      threshold=HDM_THRESHOLD,start=17,end=29),
                              HeatmapDistanceMetricPartial(name="hdm_depth",     threshold=HDM_THRESHOLD,start=29,end=30),
                              HeatmapDistanceMetricPartial(name="hdm_normal",    threshold=HDM_THRESHOLD,start=30,end=33),
                              HeatmapDistanceMetricPartial(name="hdm_depthlvls", threshold=HDM_THRESHOLD,start=33,end=34),
                              HeatmapDistanceMetricPartial(name="hdm_denoise",   threshold=HDM_THRESHOLD,start=34,end=37),
                              HeatmapDistanceMetricPartial(name="hdm_leftright", threshold=HDM_THRESHOLD,start=37,end=39),
                              HeatmapDistanceMetricPartial(name="hdm_text",      threshold=HDM_THRESHOLD,start=46,end=47),
                              HeatmapDistanceMetricPartial(name="hdm_person",    threshold=HDM_THRESHOLD,start=39,end=40),
                              HeatmapDistanceMetricPartial(name="hdm_vehicle",   threshold=HDM_THRESHOLD,start=43,end=44),
                              HeatmapDistanceMetricPartial(name="hdm_animal",    threshold=HDM_THRESHOLD,start=44,end=45),
                              HeatmapDistanceMetricPartial(name="hdm_floor",     threshold=HDM_THRESHOLD,start=57,end=58),
                              HeatmapDistanceMetricPartial(name="hdm_segms",     threshold=HDM_THRESHOLD,start=39,end=72)
                             ]
             #-------------------------------------------------------------------------------------
             model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
             #-------------------------------------------------------------------------------------
        else: 
             #model.compile(optimizer=optimizer,
             #              loss         = {'hm': hmloss,             'heatmaps_output_16bit': hmloss},
             #              loss_weights = {'hm': 1.0 ,               'heatmaps_output_16bit': 1.0 },
             #              metrics      = {'hm': hdm_metric,         'heatmaps_output_16bit': hdm16bit_metric})
             #model.compile(optimizer=optimizer, loss=combined_two_loss, metrics=[hdm_metric])
             model.compile(optimizer=optimizer, loss=hmloss, metrics=[hdm_metric])
       #--------------------------------------------------------------------------------------------------------------------------------

        #Printout data/size summaries in screen
        #--------------------------------------------------------------------------------------------------------------------------------
        bytesPerValue                   = 1 # np.int8
        channels                        = deriveRGBChannelsFromCFG(cfg)
        heatmapNumber                   = cfg['outputChannels']
        estimatedInputByteSize          = cfg['inputWidth']  * cfg['inputHeight']  * channels      * trainingDatasetLength * bytesPerValue
        estimatedOutputByteSize         = cfg['outputWidth'] * cfg['outputHeight'] * heatmapNumber * trainingDatasetLength * bytesPerValue
        estimatedOutputTokenByteSize    = cfg['tokensOut']   * dbTrain.D * 4 * trainingDatasetLength  #float32
        estimatedOutputMultihotByteSize = cfg['tokensClasses']  * 4 * trainingDatasetLength #float32
        estimatedTotalSize              = estimatedInputByteSize + estimatedOutputByteSize + estimatedOutputTokenByteSize + estimatedOutputMultihotByteSize
        #---------------------------------------------------
        queueInputSize                  = cfg['inputWidth']  * cfg['inputHeight']  * channels * bytesPerValue * queueSize
        queueOutputHMSize               = cfg['outputWidth'] * cfg['outputHeight'] * heatmapNumber * bytesPerValue * queueSize
        queueOutputTokenSize            = cfg['tokensOut']   * dbTrain.D * 4 * queueSize
        queueOutputMultihotSize         = cfg['tokensClasses']  * 4 * queueSize
        #---------------------------------------------------
        print("Dataset/Dataloader Summary")
        print("--------------------------------------------")
        print("Training Queue Size (batches) : ",queueSize)
        print("Training Queue Size : ", convert_bytes(queueInputSize + queueOutputHMSize + queueOutputTokenSize + queueOutputMultihotSize) )
        print("Input Data Size  : ",    convert_bytes(estimatedInputByteSize)," | ", channels," channels")
        print("Output Data Size : ",    convert_bytes(estimatedOutputByteSize)," | ",heatmapNumber," heatmaps")
        print("Output Token Size : ",   convert_bytes(estimatedOutputTokenByteSize)," | ",cfg['tokensOut']," tokens")
        print("Output Multihot Size : ",convert_bytes(estimatedOutputMultihotByteSize)," | ",cfg['tokensClasses']," multihot classes")
        print("Total Data Size Transfered per Epoch : ",convert_bytes(estimatedTotalSize))
        print("Total Data Size Transfered per Epoch per GPU ( ",len(cfg['GPUsUsedForTraining']),"available ) : ",convert_bytes( (estimatedTotalSize)  / numberOfGPUs )  )
        print("--------------------------------------------")
        print(" ")
        #--------------------------------------------------------------------------------------------------------------------------------

        # Train the model
        #--------------------------------------------------------------------------------------------------------------------------------
        print(bcolors.OKGREEN,"Starting training.. ",bcolors.ENDC)
        #--------------------------------------------------------------------------------------------------------------------------------
        distributedValidationDataset = None
        if (not onlyTrainingData):
         if (cfg['streamValidation']):
          print(bcolors.OKGREEN,"Streaming validation dataset from filesystem..",bcolors.ENDC) 
          distributedValidationDataset = validation_generator
         else:
          print(bcolors.OKGREEN,"Loading whole validation dataset in memory..",bcolors.ENDC) 
          distributedValidationDataset = validationDataset
        else:
         print(bcolors.WARNING,"Not using validation data..",bcolors.ENDC) 
        #--------------------------------------------------------------------------------------------------------------------------------
        if (cfg['streamDataset']):
          print(bcolors.OKGREEN,"Streaming training dataset from filesystem..",bcolors.ENDC) 
          distributedTrainingDataset = dataset_generator
        else:
          print(bcolors.OKGREEN,"Loading whole training dataset in memory..",bcolors.ENDC) 
          distributedTrainingDataset = trainingDataset
        #--------------------------------------------------------------------------------------------------------------------------------

        history   = None
        historyFT = None

        data_augmentation = DataAugmentation(cfg,dbTrain)
        if (startEpoch!=0):
             print(bcolors.OKGREEN,"Readjusting everything to start from epoch ",startEpoch," ..",bcolors.ENDC) 
             data_augmentation.on_epoch_end(startEpoch)

        #If you want to use this also enable LOG_THREADING_INFORMATION=True in NNTraining.py
        #from NNTraining import BatchLoggerCallback
        #batch_logger = BatchLoggerCallback(thread_id="train_batches")

        #Epochs can be set to zero to just do post training training
        #--------------------------------------------------------------------------------------------------------------------------------
        if (cfg['epochs']>0):
          history = model.fit(
                               distributedTrainingDataset,
                               batch_size       = cfg['batchSize'],
                               epochs           = cfg['epochs'],
                               validation_data  = distributedValidationDataset,
                               initial_epoch    = startEpoch,
                               shuffle          = shuffleData,
                               #steps_per_epoch  = steps_per_epoch, #<- Disable if not using Tensorflow Data Generator
                               callbacks        = [tensorboard_callback, early_stopping, checkpointer, lr_callback, data_augmentation] #batch_logger
                             )
        #--------------------------------------------------------------------------------------------------------------------------------
        print(bcolors.WARNING,"Recompiling model using defaults to make it more portable across frameworks..",bcolors.ENDC)
        model.compile(optimizer="adam", loss="mse")

   #Exit the multi GPU strategy scope here!
    
   # Perform any model optimizations requested by configuration and then save and package everything
   #--------------------------------------------------------------------------------------------------------------------------------
   if (cfg['pruneModel']):
        from NNOptimize import pruneModel
        model = pruneModel(model,cfg,trainingDataset)

   if (cfg['clusterModel']):
        from NNOptimize import clusterModel
        model = clusterModel(model,cfg,trainingDataset)

   saveNNModel("2d_pose_estimation",model,formats=["keras"]) #Only use keras format to save space! formats=["keras","tf","tflite","onnx"]

   #Log Per sample training report
   dbTrain.dump_sample_report("2d_pose_estimation/sample_report_training.json")

   if (history):
      logTrainingHistory(cfg, "2d_pose_estimation", "loss_history.txt", history)

   #Log best epoch / loss
   #--------------------------------------------------------------------------------------------------------------------------------
   finishDetails = dict()
   finishDetails["BestEpoch"] = checkpointer.bestEpoch
   finishDetails["Best%s" % cfg['earlyStoppingMonitor']]  = checkpointer.best
   finishDetails["BestLog"]   = checkpointer.bestLog
   print("Checkpointer Accepted Solution :", finishDetails)
   logText(finishDetails,log_dir,subject="CheckpointerAcceptedSolution")
   #--------------------------------------------------------------------------------------------------------------------------------

   #Save vocabulary
   #No longer save vocabulary to not overwrite it by mistake
   #os.system("cp datasets/descriptions/index_to_word.json 2d_pose_estimation/vocabulary.json")

   #Save model as INT8 TF-Lite (Disabled because it takes too much space and not currently needed)
   #if (not onlyTrainingData):
   #   print(bcolors.WARNING,"We have a validation set, so saving TF-Lite INT8 model..",bcolors.ENDC)
   #   from NNConverter import saveNNTFLiteINT8Model, saveNNTFLiteFP16Model
   #   saveNNTFLiteINT8Model(model,dbValidation.get_in_array())
   #   saveNNTFLiteFP16Model(model,dbValidation.get_in_array())

   # Package output
   #--------------------------------------------------------------------------------------------------------------------------------
   print(bcolors.OKGREEN,"Training complete..",bcolors.ENDC)
   os.system("date +\"%y-%m-%d_%H-%M-%S\" > 2d_pose_estimation/date.txt") #Tag date
   if (packageTrainedModel):
     print(bcolors.OKGREEN,"Packaging trained model ..",bcolors.ENDC)
     os.system("rm 2d_pose_estimation.zip") #Make sure there is no zip file
     os.system("zip -r 2d_pose_estimation.zip 2d_pose_estimation/") #Create zip of models
   else:
     print("Not packaging model")
   #--------------------------------------------------------------------------------------------------------------------------------

   #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
   if (not onlyTrainingData):
       weight_val_array = dbValidation.get_token_fequencies()
       extract_validation_losses(model, validation_generator, dbValidation)
       dbValidation.dump_sample_report("2d_pose_estimation/sample_report_validation.json")
       print("Done Dumping Validation Samples ")
   #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


   print('You can see a summary using :\n tensorboard --logdir=2d_pose_estimation/tensorboard --bind_all && firefox http://127.0.0.1:6006')

   # Upload results
   #--------------------------------------------------------------------------------------------------------------------------------
   print('To upload results (if you take too long it will timeout) :')
   os.system("timeout 5 scripts/uploadResults.sh")
   #print("scp -P 2222 2d_pose_estimation.zip ammar@ammar.gr:/home/ammar/public_html/poseanddepth")
   #print(" or ")
   print("scp -P 2222 2d_pose_estimation.zip ammar@ammar.gr:/home/ammar/public_html/poseanddepth/archive/2d_pose_estimation_v%s.zip" % str(cfg['serial']))  

