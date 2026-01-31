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
from NNLosses import RSquaredMetric,HeatmapDistanceMetric,vanilla_mse_loss,AdamWCautious
from tools import bcolors,read_json_file,checkIfPathExists,checkIfFileExists,convert_bytes
#-------------------------------------------------------------------------------
def deriveRGBChannelsFromCFG(cfg):
    numberOfChannels = 3
    if (cfg['RGBImageEncoding']=='rgb8'):
        numberOfChannels = 1
    if (cfg['RGBImageEncoding']=='rgb16'):
        numberOfChannels = 2
    return numberOfChannels
#-------------------------------------------------------------------------------
def deriveHeatmapChannelsFromCFG(cfg):
    extraHeatmaps = len(cfg['keypoint_names']) #names include bkg + 1 depthmap
    if (cfg['heatmapAddDepthmap']):
         extraHeatmaps = extraHeatmaps + 1
    return extraHeatmaps
#------------------------------------------------------------------------------- 

#-------------------------------------------------------------------------------
from NNTraining import logTrainingHistory,logText,logSomeInputsAndOutputs,printTFVersion, getOptimizerFromCFG, getLossFromCFG, TrainingDataGenerator, custom_lr_scheduler, custom_lr_schedulerWarmup, DataAugmentation, weighted_token_loss, extract_validation_losses
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

   jsonPath = '2d_pose_estimation/tokens.json'
   if (checkIfFileExists(jsonPath)):
        print(bcolors.OKGREEN,"Loading configuration from file ",jsonPath,bcolors.ENDC)
        from createJSONConfiguration import loadJSONConfiguration
        cfg = loadJSONConfiguration(jsonPath)
   else:
        print(bcolors.FAIL,"CREATING FRESH CONFIGURATION!",bcolors.ENDC)
        from createJSONConfiguration import createJSONConfiguration
        cfg = createJSONConfiguration(jsonPath)

   if (cfg['mixedPrecision']):
      print(bcolors.WARNING,"Using mixed precision mode!",bcolors.ENDC)
      keras.mixed_precision.set_global_policy("mixed_float16") 

   countResponses         = False
   saveRestoredWeights    = False 
   restoreBestWeights     = False
   resumePreviousTraining = False
   elevatePriority        = False
   startEpoch             = 0
   if (len(sys.argv)>1):
       #print('Argument List:', str(sys.argv))
       for i in range(0, len(sys.argv)):
           if (sys.argv[i]=="--debug"):
              #Enabling this multiplies training time from 6min to 1hour
              tf.config.run_functions_eagerly(True) # <- DEBUG
              cfg['batchSize']=4#<- Reduce batch size since eager executio requires more memory
           if (sys.argv[i]=="--flush"):
              os.system("rm -rf 2d_pose_estimation/ && mkdir 2d_pose_estimation/")
           if (sys.argv[i]=="--novalidation"):
              cfg['doValidation']=False
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
           if (sys.argv[i]=="--rt"):
              elevatePriority = True
           if (sys.argv[i]=="--count"):
              countResponses = True
           if (sys.argv[i]=="--restoreBestWeights") or (sys.argv[i]=="--restore"):
              restoreBestWeights     = True
              resumePreviousTraining = False
              saveRestoredWeights    = False 
           if (sys.argv[i]=="--saveRestoredWeights") or (sys.argv[i]=="--save"):
              restoreBestWeights     = True
              resumePreviousTraining = False
              saveRestoredWeights    = True
           if (sys.argv[i]=="--test"):
              from train2DPoseEstimator import createSelectedModel
              model = createSelectedModel(cfg)
              from NNModel import retrieveModelOutputDimensions 
              retrieveModelOutputDimensions(model) 
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
   """
   if (numberOfGPUs>1):
       strategy = tf.distribute.MirroredStrategy(devices=cfg['GPUsUsedForTraining'])
   elif (numberOfGPUs==1):
       strategy = tf.distribute.OneDeviceStrategy(device=cfg['GPUsUsedForTraining'][0])
   else:
       strategy = tf.distribute.OneDeviceStrategy()

   
   with strategy.scope():  #Comment out this line
   """
   if (numberOfGPUs>0):   #and comment in this line for single GPU usage..
        #First of all create the Neural Network model
        #Don't load other data in vain if this step fails due to bad configuration..
        #----------------------------------------------------------------------------------------
        if (resumePreviousTraining):
            print(bcolors.OKGREEN,"Continuing training from pretrained network.. ",bcolors.ENDC)
            model_path = "2d_pose_estimation/tokens/model.keras"
            from NNModel import load_keypoints_model
            model,input_size,output_size,numHeatmaps = load_keypoints_model(model_path)
            cfg['inputWidth']   = input_size[0]
            cfg['inputHeight']  = input_size[1]
            cfg['outputWidth']  = output_size[0]
            cfg['outputHeight'] = output_size[1]
        else:
            print(bcolors.OKGREEN,"Creating a new Token Only Neural Network.. ",bcolors.ENDC)
            #from NNModel import build_simple_cnn
            #model = build_simple_cnn(( cfg['inputWidth'], cfg['inputHeight'], 3), 2037 )
            #from NNModel import build_vit
            #model = build_vit(( cfg['inputWidth'], cfg['inputHeight'], 3), 2037 )
            #from NNModel import build_resnet_cnn
            #model = build_resnet_cnn(( cfg['inputWidth'], cfg['inputHeight'], 3), 2037 ) 
            from NNModel import build_resnethybrid_cnn
            model = build_resnethybrid_cnn(( cfg['inputWidth'], cfg['inputHeight'], 3), # 2037, 
                                            dropoutRate=cfg['dropoutRate'],
                                            gloveLayers=cfg['gloveLayers'],
                                            bridgeLayerWidthCompatibility=cfg['forceBridgeSize'],
                                            multihotLayers=cfg['multihotLayers'], 
                                            numTokens=cfg['tokensOut'],
                                            numClasses=cfg['tokensClasses'],
                                            nextTokenStrength=cfg['nextTokenStrength'],
                                            useDescriptors=cfg['outputDescriptors']
                    ) #<- Experimental

             
        #----------------------------------------------------------------------------------------

        # Set up TensorBoard logging
        #----------------------------------------------------------------------------------------
        log_dir = "2d_pose_estimation/tensorboard/" + cfg["serial"] + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        if (cfg['streamDataset']):
           mem=1.0 #When streaming use everything..

        #Before starting training log TF Versions
        printTFVersion()

        #Validation data is not so big and is loaded first in memory..
        onlyTrainingData = True

        if cfg['doValidation']: #(checkIfFileExists(cfg['COCOValidationJSONPath']) and
             onlyTrainingData = False
             dbValidation = DataLoader(
                                       (cfg['inputHeight'], cfg['inputWidth'], 3),
                                       (cfg['outputHeight'],cfg['outputWidth'],cfg['outputChannels']),
                                       output16BitChannels = cfg['output16BitChannels'] ,
                                       numberOfThreads = cfg['DatasetLoaderThreads'],
                                       streamData      = int(cfg['streamValidation']), #0, <- set to 0 to keep it in memory and speed up validation 
                                       batchSize       = cfg['batchSize'],
                                       gradientSize    = cfg['heatmapGradientSizeMinimum'],# <- Use final size! cfg['heatmapGradientSize'],
                                       PAFSize         = cfg['heatmapPAFSizeMinimum'],     # <- Use final size! cfg['heatmapPAFSizeMinimum'],
                                       doAugmentations = 0,# <- Don't do augmentations on test set keep it clean
                                       addPAFs         = int(cfg['heatmapAddPAFs']),
                                       addBackground   = int(cfg['heatmapGenerateSkeletonBkg']),
                                       addDepthMap     = int(cfg['heatmapAddDepthmap']),
                                       addNormals      = int(cfg['heatmapAddNormals']),
                                       addSegmentation = int(cfg['heatmapAddSegmentation']),
                                       elevatePriority = elevatePriority,
                                       datasets        = cfg["ValidationDataset"],
                                       libraryPath     = "datasets/DataLoader/libDataLoader.so")
             if (cfg['streamValidation']):
                  # Modify the way you call the dataset
                  validation_generator    = TrainingDataGenerator(cfg=cfg, db=dbValidation, batch_size=cfg['batchSize'], validation_data=True, returnOutputImages=False,   numberOfTokens=cfg["tokensOut"],  
                                                                  workers=1, use_multiprocessing=False, max_queue_size=1) #multiprocessing happens inside the dataloader
                  validationDatasetLength = dbValidation.numberOfSamples
                  outValLabels            = dbValidation.get_labels()



        #The training set is very large, so depending on the system there are two ways to use it
        #try streaming it which is very slow due to I/O operations but can work on small VRAM systems
        #or load it all in memory and use regular TF mechanisms to train on it
        if True: #(checkIfFileExists(cfg['COCOTrainingJSONPath'])):
             dbTrain = DataLoader(
                                       (cfg['inputHeight'],cfg['inputWidth'],3),
                                       (cfg['outputHeight'],cfg['outputWidth'],cfg['outputChannels']),
                                       output16BitChannels = cfg['output16BitChannels'] ,
                                       #numberOfSamples = 10000, <- uncomment to just use a few samples for test
                                       streamData      = int(cfg['streamDataset']),
                                       batchSize       = cfg['batchSize'],
                                       numberOfThreads = cfg['DatasetLoaderThreads'],
                                       gradientSize    = cfg['heatmapGradientSize'],
                                       PAFSize         = cfg['heatmapPAFSize'],
                                       doAugmentations = 0,#int(cfg['dataAugmentation']),#0 = DISABLED AUGMENTATIONS TO HELP THIS int(cfg['dataAugmentation']),
                                       addPAFs         = int(cfg['heatmapAddPAFs']),
                                       addBackground   = int(cfg['heatmapGenerateSkeletonBkg']),
                                       addDepthMap     = int(cfg['heatmapAddDepthmap']),
                                       addNormals      = int(cfg['heatmapAddNormals']),
                                       addSegmentation = int(cfg['heatmapAddSegmentation']),
                                       elevatePriority = elevatePriority,
                                       datasets        = cfg["TrainingDataset"], 
                                       libraryPath     = "datasets/DataLoader/libDataLoader.so")
             dbTrain.updateJointDifficulty(cfg['keypoint_difficulty'])
             if (cfg['streamDataset']):
                  # Modify the way you call the dataset
                  outLabels             = dbTrain.get_labels()
                  dataset_generator     = TrainingDataGenerator(cfg=cfg, db=dbTrain, batch_size=cfg['batchSize'], labels=outLabels, log_dir=log_dir, returnOutputImages=False,  numberOfTokens=cfg["tokensOut"],  
                                                                workers=1, use_multiprocessing=False, max_queue_size=4) #multiprocessing happens inside the dataloader
                  trainingDatasetLength = dbTrain.numberOfSamples
                  shuffleData           = False #<- shuffling is done inside the generator
        #----------------------------------------------------------------
  
        # Print the shapes of inputs and outputs
        print("Training Configuration :", cfg)
        logText(cfg,log_dir)

        print("DISABLE HEATMAP OUTPUT :")
        dbValidation.disableHeatmapOutput()
        dbTrain.disableHeatmapOutput()

        # Define Learning Rate Scheduler callback
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
                                                        restore_best_weights = True                           # Restore model weights from the epoch with the best value of the monitored quantity
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
                                                  start_from_epoch=0,
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
        HDM_THRESHOLD   = 0.1
        hdm_metric      = HeatmapDistanceMetric(threshold=HDM_THRESHOLD)
        hdm16bit_metric = HeatmapDistanceMetric(name='hdm16', threshold= HDM_THRESHOLD) #(32767.0/120.0) * HDM_THRESHOLD
        #rsq_metric = RSquaredMetric()

        # Initialize the Adam Optimizer using configuration
        #optimizer = AdamWCautious(learning_rate=float(cfg['learningRate']),clipnorm=None,clipvalue=1.0)
        #optimizer = tf.keras.optimizers.Adam(learning_rate=float(cfg['learningRate']))
        optimizer = getOptimizerFromCFG(cfg)


        #Decide on heatmap loss based on configuration
        hmloss = None
        if (cfg['loss']=="mse"):
             hmloss = vanilla_mse_loss
        elif (cfg['loss']=="combine"):
             print(bcolors.WARNING,"Using experimental combined loss..",bcolors.ENDC)
             hmloss = combined_loss 
        elif (cfg['loss']=="dssim"):
             print(bcolors.WARNING,"Using experimental dssim loss..",bcolors.ENDC)
             hmloss=dssim_loss 
        else:
             print(bcolors.WARNING,"Using experimental dssim loss..",bcolors.ENDC)
             hmloss=cfg['loss']
        #hmloss = getLossFromCFG(cfg) TODO
   


        #Compile a model with the requested loss
        if ("outputTokens" in cfg) and (cfg["outputTokens"]):                              
             #token_loss_function = token_mse_loss
             #Instead of categorical cross-entropy (used in single-label classification), binary cross-entropy is used as the loss function for multi-label tasks. This allows the model to handle each class independently.
             #token_loss_function = keras.losses.CategoricalCrossentropy(from_logits=False) #<- One Class
             #token_loss_function = keras.losses.BinaryCrossentropy(from_logits=False)       #<- Multiple Classes  
             #model.compile(optimizer=optimizer,  loss=weighted_token_loss, metrics=[ 'accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]) 

             print("Now retrieving token weights!")
             weight_array = dbTrain.get_token_fequencies()
             class_weight_dict = {i: float(weight) for i, weight in enumerate(weight_array)}
                       
             #-------------------------------------------------------------------------------------
             from NNLosses import GloVeMSELoss,GloVeCosineLoss,GloVeHybridLoss, MultiHotLoss, WeightedBinaryCrossEntropy, WeightedFocalLoss
             losses = dict()
             for i in range(cfg["tokensOut"]):
                         #losses["t%02u"%i] = GloVeMSELoss(weight=1.0)
                         losses["t%02u"%i] = GloVeHybridLoss(mse_weight=1.0, cosine_weight=1.0)

             #losses['tokens_multihot']  = WeightedBinaryCrossEntropy(weight_array)  #MultiHotLoss()
             losses['tokens_multihot']  = WeightedFocalLoss(weight_array)  #<- Try focal loss

             if (cfg["outputDescriptors"]):
                losses['descriptors']  = "mse"
             #-------------------------------------------------------------------------------------
             loss_weights = dict()
             for i in range(cfg["tokensOut"]):  
                         loss_weights["t%02u"%i] = cfg["lossWeightGloveTokens"] / cfg["tokensOut"]

             loss_weights['tokens_multihot']  = cfg["lossWeightMultihotTokens"] #0.001 

             if (cfg["outputDescriptors"]):
                loss_weights['descriptors']  = 10.0
             #-------------------------------------------------------------------------------------
             from NNLosses import HeatmapDistanceMetricPartial, CustomTopKCategoricalAccuracy
             metrics = dict() 
             for i in range(cfg["tokensOut"]):
                         metrics["t%02u"%i] = keras.metrics.CosineSimilarity(name='cossim', axis=1)

             metrics['tokens_multihot']   = [ 
                                              'accuracy', 
                                              tf.keras.metrics.TopKCategoricalAccuracy(name="top3_accuracy",k=3),
                                              tf.keras.metrics.TopKCategoricalAccuracy(name="top5_accuracy",k=5)
                                            ]
             #-------------------------------------------------------------------------------------

             model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
       #--------------------------------------------------------------------------------------------------------------------------------

   #Printout data/size summaries in screen
   #--------------------------------------------------------------------------------------------------------------------------------
   bytesPerValue           = 1 # np.int8
   channels                = deriveRGBChannelsFromCFG(cfg)
   heatmapNumber           = cfg['outputChannels']
   estimatedInputByteSize  = cfg['inputWidth']  * cfg['inputHeight']  * channels      * trainingDatasetLength * bytesPerValue
   estimatedOutputByteSize = cfg['outputWidth'] * cfg['outputHeight'] * heatmapNumber * trainingDatasetLength * bytesPerValue
   print("Input Data Size  : ",convert_bytes(estimatedInputByteSize)," ", channels," channels")
   print("Output Data Size : ",convert_bytes(estimatedOutputByteSize)," ",heatmapNumber," heatmaps")
   print("Total Data Size  : ",convert_bytes(estimatedInputByteSize+estimatedOutputByteSize))
   print("Total Data Size per GPU ( ",len(cfg['GPUsUsedForTraining']),"available ) : ",convert_bytes( (estimatedInputByteSize+estimatedOutputByteSize)  / numberOfGPUs )  )
   #--------------------------------------------------------------------------------------------------------------------------------

   # Train the model
   #--------------------------------------------------------------------------------------------------------------------------------
   print(bcolors.OKGREEN,"Starting training.. ",bcolors.ENDC)
   #--------------------------------------------------------------------------------------------------------------------------------
   distributedValidationDataset = None
   if (not onlyTrainingData):
    if (cfg['streamValidation']):
     print(bcolors.OKGREEN,"Streaming validation dataset from filesystem.. ",bcolors.ENDC) 
     distributedValidationDataset = validation_generator
    else:
     print(bcolors.OKGREEN,"Loading whole validation dataset in memory.. ",bcolors.ENDC) 
     distributedValidationDataset = validationDataset
   else:
    print(bcolors.WARNING,"Not using validation data.. ",bcolors.ENDC) 
   #--------------------------------------------------------------------------------------------------------------------------------
   if (cfg['streamDataset']):
     print(bcolors.OKGREEN,"Streaming training dataset from filesystem.. ",bcolors.ENDC) 
     distributedTrainingDataset = dataset_generator
   else:
     print(bcolors.OKGREEN,"Loading whole training dataset in memory.. ",bcolors.ENDC) 
     distributedTrainingDataset = trainingDataset
   #--------------------------------------------------------------------------------------------------------------------------------

   history   = None
   historyFT = None

   if (countResponses):
       import keras
       model = keras.saving.load_model("2d_pose_estimation/model.keras", custom_objects={ 'weighted_token_loss':weighted_token_loss }, compile=True, safe_mode=True)

       # Initialize a list to store hits for each list of tokens
       hits_per_token = [0] * 2048
       for index in range(100):#(dbTrain.get_number_of_samples(dbTrain.db) // dbTrain.batchSize ):
            print("Batch ",index,"/",dbTrain.get_number_of_samples(dbTrain.db) // dbTrain.batchSize)
            rgb,tokens = distributedTrainingDataset.__getitem__(index)
            preds = model(rgb)
            print("Results ",preds)
                # Process each sample in the batch
            for batch_idx in range(preds.shape[0]):
                # Ground truth tokens for the current sample
                gt_tokens = tokens[batch_idx]

            # Process each list of 2048 tokens
            for batch_idx in range(preds.shape[0]):
             # Process each list of 2048 tokens
             for list_idx in range(dbTrain.batchSize):
                 # Get predicted tokens for the current list
                 predicted_tokens = preds[batch_idx, list_idx]

                 # Find the indices of the top 5 active tokens
                 top5_predicted_indices = np.argsort(predicted_tokens)[-5:]

                 # Record the frequency of the top 5 tokens
                 for idx in top5_predicted_indices:
                     hits_per_token[idx] += 1       

       with open('token_accuracy.csv', 'w') as f:
          f.write('id,hits\n')
          for i in range(2048):
           f.write(str(i))
           f.write(',')
           f.write(str(hits_per_token[i]))
           f.write('\n')
           print(i," - ",hits_per_token[i])

       sys.exit(0)


   import json

   #print("Validation data copy train blacklist!")
   #dbValidation.tokenblacklistkeys = dbTrain.tokenblacklistkeys 
   #dbValidation.update_token_blacklist(dbValidation.tokenblacklistkeys,lowThreshold=2)
   print("Total number of train black listed keys : ",len(dbTrain.tokenblacklistkeys))
   print("Total number of validation black listed keys : ",len(dbValidation.tokenblacklistkeys))

   print("Now retrieve token weights!")
   weight_array = dbTrain.get_token_fequencies()

   # Convert the weight_array into a dictionary
   class_weight_dict = {i: float(weight) for i, weight in enumerate(weight_array)}
   with open("class_weight_dict_new.json", 'w') as json_file:
     json.dump(class_weight_dict, json_file, indent=4)
   #sys.exit(0)

   if (cfg["epochsFrozen"]):
     history = model.fit(
                           distributedTrainingDataset,
                           batch_size       = cfg['batchSize'],
                           epochs           = cfg["epochsFrozen"],
                           validation_data  = distributedValidationDataset,
                           initial_epoch    = startEpoch,
                           shuffle          = shuffleData,
                           #class_weight     = class_weight_dict,  # Pass the calculated weights here <- this gets now done via loss_weights
                           callbacks        = [tensorboard_callback,early_stopping,checkpointer,lr_callback,DataAugmentation(cfg,dbTrain)]
                          )
     checkpointer.load_best_model()
     
     #Log best epoch / loss
     #--------------------------------------------------------------------------------------------------------------------------------
     finishDetails = dict()
     finishDetails["BestEpoch"] = checkpointer.bestEpoch
     finishDetails["Best%s" % cfg['earlyStoppingMonitor']]  = checkpointer.best
     finishDetails["BestLog"]   = checkpointer.bestLog
     print("Checkpointer Accepted Freezed Solution :", finishDetails)
     logText(finishDetails,log_dir,subject="CheckpointerAcceptedFreezedSolution")
     #--------------------------------------------------------------------------------------------------------------------------------


   # Step 2: Unfreeze the base model
   print("Unfreeze")
   model.get_layer('resnet50').trainable = True  # 'resnet50' or 'convnext_small' should be the name of the base model

   #------------------------------------------------------------------------------------------
   optimizer = getOptimizerFromCFG(cfg)
   model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)


   #Epochs can be set to zero to just do post training training
   #--------------------------------------------------------------------------------------------------------------------------------
   if (cfg['epochs']>0):
     history = model.fit(
                           distributedTrainingDataset,
                           batch_size       = cfg['batchSize'],
                           epochs           = cfg['epochs'],
                           validation_data  = distributedValidationDataset,
                           shuffle          = shuffleData,
                           #class_weight     = class_weight_dict,  # Pass the calculated weights here  <- this gets now done via loss_weights
                           callbacks        = [tensorboard_callback,early_stopping,checkpointer,lr_callback,DataAugmentation(cfg,dbTrain)]
                         )
     checkpointer.load_best_model()

     #Log best epoch / loss
     #--------------------------------------------------------------------------------------------------------------------------------
     finishDetails = dict()
     finishDetails["BestEpoch"] = checkpointer.bestEpoch
     finishDetails["Best%s" % cfg['earlyStoppingMonitor']]  = checkpointer.best
     finishDetails["BestLog"]   = checkpointer.bestLog
     print("Checkpointer Accepted Freezed Solution :", finishDetails)
     logText(finishDetails,log_dir,subject="CheckpointerAcceptedFinalSolution")
     #--------------------------------------------------------------------------------------------------------------------------------
   #--------------------------------------------------------------------------------------------------------------------------------   
   print(bcolors.WARNING,"Recompiling model using defaults to make it more portable..",bcolors.ENDC)
   model.compile(optimizer="adam", loss="mse")

   # Perform any model optimizations requested by configuration and then save and package everything
   #--------------------------------------------------------------------------------------------------------------------------------
   if (cfg['pruneModel']):
        from NNOptimize import pruneModel
        model = pruneModel(model,cfg,trainingDataset)

   if (cfg['clusterModel']):
        from NNOptimize import clusterModel
        model = clusterModel(model,cfg,trainingDataset)

   saveNNModel("2d_pose_estimation/tokens",model,formats=["keras"]) #Only use keras format to save space! formats=["keras","tf","tflite","onnx"]
   
   #Dump training sample report 
   dbTrain.dump_sample_report("2d_pose_estimation/sample_report_training_tokens.json")


   if (history):
      logTrainingHistory(cfg, "2d_pose_estimation", "loss_history.txt", history)
   if (historyFT):
      logTrainingHistory(cfg, "2d_pose_estimation", "loss_finetune_history.txt", historyFT)

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
   os.system("rm 2d_pose_estimation_tokens.zip") #Make sure there is no zip file
   os.system("zip -r 2d_pose_estimation_tokens.zip 2d_pose_estimation/ -x 2d_pose_estimation/model.keras") #Create zip of models (don't include main model)
   #--------------------------------------------------------------------------------------------------------------------------------



   #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
   if (not onlyTrainingData):
     weight_val_array = dbValidation.get_token_fequencies()
     extract_validation_losses(model, validation_generator, dbValidation)
     dbValidation.dump_sample_report("2d_pose_estimation/sample_report_validation_tokens.json")
     print("Done Dumping Validation Samples ")
   #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


   print('You can see a summary using :\n tensorboard --logdir=2d_pose_estimation/tensorboard --bind_all && firefox http://127.0.0.1:6006')

   # Upload results
   #--------------------------------------------------------------------------------------------------------------------------------
   print('To upload results (if you take too long it will timeout) :')
   os.system("timeout 5 scripts/uploadResults.sh")
   print("scp -P 2222 2d_pose_estimation_tokens.zip ammar@ammar.gr:/home/ammar/public_html/poseanddepth")
   print(" or ")
   print("scp -P 2222 2d_pose_estimation_tokens.zip ammar@ammar.gr:/home/ammar/public_html/poseanddepth/archive/2d_pose_estimation_tokens_v%s.zip" % str(cfg['serial']))  


