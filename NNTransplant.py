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


def load_any_model(model_path):
    from tools import bcolors
    from NNLosses import HeatmapDistanceMetric, GloVeMSELoss, HeatmapCoreLoss, WeightedBinaryCrossEntropy, HeatmapDistanceMetricPartial, CustomTopKCategoricalAccuracy
    print(bcolors.OKGREEN,"Loading %s model.. " % model_path,bcolors.ENDC)
    try:
       #Regular keras loading until V3 that breaks
       import keras
       doModelCompilation = True #<- Does this have any effect on loading speed?

       print("Keras is NOW loading the saved model..")
       start      = time.time()
       model = keras.saving.load_model(model_path, custom_objects={
                                                        'HeatmapDistanceMetric':HeatmapDistanceMetric
                                                       ,'HeatmapDistanceMetricPartial':HeatmapDistanceMetricPartial
                                                       ,'GloVeMSELoss':GloVeMSELoss
                                                       ,'WeightedBinaryCrossEntropy':WeightedBinaryCrossEntropy
                                                       ,'HeatmapCoreLoss':HeatmapCoreLoss
                                                       ,'CustomTopKCategoricalAccuracy':CustomTopKCategoricalAccuracy
                                                     }, compile=doModelCompilation)#, safe_mode=True)
       seconds    = time.time() - start
       print("Loading the model took ",seconds," seconds..")
       return model

    except:
      print("Could not load model",model_path)
      return None

def transplantLayersFromSourceToTarget(modelTarget, modelSource, layerNameList):
    """
    Transplants layers from a source model to a target model based on a list of layer names.

    Args:
        modelTarget (Model): The target model where layers will be transplanted.
        modelSource (Model): The source model from which layers and weights will be copied.
        layerNameList (list): List of layer names to transplant.

    Returns:
        Model: The target model with transplanted layers.
    """

    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC    = '\033[0m'

    # Create a mapping from layer names to layers in the source model
    source_layers = {layer.name: layer for layer in modelSource.layers}
    mismatches = 0    

    for layerName in layerNameList:
        # Check if the layer exists in both models
        if layerName in source_layers and layerName in [layer.name for layer in modelTarget.layers]:
            # Get the corresponding layers
            try:
              source_layer = source_layers[layerName]
              target_layer = modelTarget.get_layer(layerName)
              # Copy weights from the source layer to the target layer
              target_layer.set_weights(source_layer.get_weights())
            except Exception as e:
              print(WARNING,"Could not transplant layer ",layerName,".",ENDC)
              print(WARNING,e,ENDC)

        else:
            print(WARNING,"Layer ",layerName," not found in one of the models.",ENDC)
            mismatches = mismatches + 1

    if (mismatches>0):
            print(WARNING,"There where ",mismatches," mismatches while transplanting layers",ENDC)
            print(WARNING,"Could stop here but continuing..",ENDC)
            #raise ValueError(f"transplantLayersFromSourceToTarget failed transplating models.")


    return modelTarget
