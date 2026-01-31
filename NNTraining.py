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
import threading
import queue

LOG_THREADING_INFORMATION=False
tickBase = 0
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
from NNLosses import RSquaredMetric,HeatmapDistanceMetric,AdamWCautious
from tools import bcolors,read_json_file,checkIfPathExists,checkIfFileExists,convert_bytes
#-------------------------------------------------------------------------------
def logTrainingHistory(cfg, log_dir, log_name, history):
  # Extract training and validation loss history
  training_loss   = history.history['loss']
  validation_loss = None 
  if 'val_loss' in history.history:
           validation_loss = history.history['val_loss']

  # Save loss history to a text file
  with open('%s/%s' % (log_dir,log_name) , 'w') as f:
    f.write('Training Loss:\n')
    f.write('\n'.join(map(str, training_loss)))
    if validation_loss:
        f.write('\nValidation Loss:\n')
        f.write('\n'.join(map(str, validation_loss)))

  print("Loss history saved")
#-------------------------------------------------------------------------------
def logText(cfg, log_dir, subject="TrainingParameters"):
    try:
      # Create a summary writer
      param_log = tf.summary.create_file_writer(log_dir)

      with param_log.as_default():
        # Convert the dictionary to a formatted string
        params_str = "\n".join([f"{key}: {value}" for key, value in cfg.items()])
        # Log the parameters as text
        tf.summary.text(subject, params_str, step=0)
    except Exception as e:
        print(f"Error storing logging parameters in tensorboard: {e}")
#-------------------------------------------------------------------------------
def logSomeInputsAndOutputs(inputs, outputs, outputs16B, labels, log_dir, samples, description="Validation Data", localSave=False):
    try:
      # Create a summary writer
      image_log = tf.summary.create_file_writer(log_dir)

      with image_log.as_default():

        sample_indices = np.random.choice(len(inputs), size=min(samples,len(inputs)), replace=False)

        for logID in sample_indices:
            #Store the image as float32 [0..1] RGB to make sure tensorboard visualizes it correctly
            image_as_float = inputs[logID].astype(np.float32)
            image = image_as_float / 255.0

            # Convert input and output arrays to TensorFlow tensors
            bgr_image_tensor = tf.convert_to_tensor([image], dtype=tf.float32)

            # Swap BGR to RGB not needed 
            #rgb_image_tensor = tf.reverse(bgr_image_tensor, axis=[-1])

            # Write input image summary
            tf.summary.image(f"{description} Image {logID} Input", bgr_image_tensor , step=logID) #rgb_image_tensor

            # Write output image summary
            numberOfHeatmaps = outputs.shape[3] #Should be 18 ?
            for heatmapID in range(0,numberOfHeatmaps):
               heatmap = outputs[logID,:,:,heatmapID]
               #print(f"Heatmap {heatmapID} dimensions: {heatmap.shape}")
               # Add batch and channel dimensions
               heatmapS            = np.squeeze(heatmap)
               heatmapS            = np.expand_dims(heatmapS, axis=-1)
               heatmap_as_float    = (heatmapS.astype(np.float32) + 120.0) / 240.0
               output_image_tensor = tf.convert_to_tensor([heatmap_as_float], dtype=tf.float32)

               thisOutputlabel = "#%u" % heatmapID
               if (heatmapID < len(labels)):
                    thisOutputlabel = labels[heatmapID]

               if (localSave):
                 print("Saving local heatmap for sample ",logID)
                 cv2.imwrite('heatmap_in_%u.png'%(logID),image_as_float)
                 cv2.imwrite('heatmap_%u_%u.png'%(logID,heatmapID),heatmap_as_float)
               tf.summary.image(f"{description} Image {logID} Output / {thisOutputlabel}", output_image_tensor, step=logID)

            numberOfHeatmaps = outputs16B.shape[3] #Should be 18 ?
            for heatmapID in range(0,numberOfHeatmaps):
               heatmap = outputs16B[logID,:,:,heatmapID]
               #print(f"Heatmap {heatmapID} dimensions: {heatmap.shape}")
               # Add batch and channel dimensions
               heatmapS            = np.squeeze(heatmap)
               heatmapS            = np.expand_dims(heatmapS, axis=-1)
               heatmap_as_float    =  ( heatmapS.astype(np.float32) + 120.0) / 240.0
               output_image_tensor = tf.convert_to_tensor([heatmap_as_float], dtype=tf.float32)
               thisOutputlabel = "#%u-16BIT" % heatmapID

               if (localSave):
                 print("Saving local heatmap for sample ",logID)
                 cv2.imwrite('heatmap16B_%u_%u.png'%(logID,heatmapID),heatmap_as_float)
               tf.summary.image(f"{description} Image {logID} Output / {thisOutputlabel}", output_image_tensor, step=logID)



    except Exception as e:
        print(f"Error storing image in tensorboard: {e}")
        sys.exit(1)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def get_tick_count_microseconds_mn():
    """Simulates GetTickCountMicrosecondsMN from C."""
    global tickBase
    rawTicks = int(time.time() * 1_000_000)
    if (tickBase==0):
         tickBase = rawTicks

    return rawTicks - tickBase

def log_thread_progress(thread_label: str, start: int, part: str):
    global LOG_THREADING_INFORMATION
    if (LOG_THREADING_INFORMATION):
      filename = f"thread_{thread_label}.log"
      try:
        with open(filename, "a") as fp:
            timestamp = get_tick_count_microseconds_mn()
            fp.write(f"{timestamp},{start},{part}\n")
      except IOError:
        pass  # You can handle the error if needed
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class BatchLoggerCallback(keras.callbacks.Callback):
    def __init__(self, thread_id="gpu"):
        super().__init__()
        self.thread_id = thread_id
        #Make sure any previous logs are erased
        os.system("rm thread_*.log")

    def on_train_batch_begin(self, batch, logs=None):
        log_thread_progress(self.thread_id, 1, f"update_gpu_batch")

    def on_train_batch_end(self, batch, logs=None):
        log_thread_progress(self.thread_id, 0, f"update_gpu_batch")
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def printTFVersion():
       global useGPU
       print("")
       print("Tensorflow version : ",tf.__version__)
       print("Keras version      : ",keras.__version__) #<- no longer available in TF-2.13
       print("Numpy version      : ",np.__version__)
       #-----------------------------
       from tensorflow.python.platform import build_info as tf_build_info
       print("TF/CUDA version    : ",tf_build_info.build_info['cuda_version'])
       print("TF/CUDNN version   : ",tf_build_info.build_info['cudnn_version'])
       print("Use GPU            : ",useGPU)
       #-----------------------------
       if useGPU:
         physical_devices = tf.config.list_physical_devices('GPU')
         if physical_devices:
            for gpuID,gpu in enumerate(physical_devices):
                print("GPU #",gpuID," Name:", gpu.name)
                try:
                    # Note: The following code may not be available in older versions of TensorFlow
                    memory_info = tf.config.experimental.get_memory_info('GPU:%u'%gpuID)
                    print("GPU #",gpuID," Memory Currently Used (in MB):", memory_info['current'] / (1024**2))
                    print("GPU #",gpuID," Memory Peak Used (in MB):", memory_info['peak'] / (1024**2))
                except Exception as e:
                    print(f"Error getting memory info for GPU #{gpuID}: {e}")
         else:
            print("No GPU available.")
       print("Threads Available:")
       os.system("cat /proc/cpuinfo | grep processor | wc -l")
       os.system("lscpu")
       os.system("numactl --hardware")
       os.system("cat /proc/pressure/memory") #<- Memory Pressure
       os.system("df -h /dev/shm")
       os.system("df -h /")
       os.system("free -h") #<- Memory In General
       print("")
#-------------------------------------------------------------------------------
"""
def getLossFromCFG(cfg):
        hmloss = None
        if (cfg['loss']=="mse"):
             from NNLosses import HeatmapCoreLoss
             hmloss =  HeatmapCoreLoss(
                                       jointGain=cfg['lossWeightJoints'],
                                       PAFGain=cfg['lossWeightPAFs'],
                                       DepthGain=cfg['lossWeightDepth'],
                                       NormalGain=cfg['lossWeightNormals'],
                                       TextGain=cfg['lossWeightText'],
                                       SegmentGain=cfg['lossWeightSegmentation']
                                      )
        elif (cfg['loss']=="combine"):
             print(bcolors.WARNING,"Using experimental combined loss..",bcolors.ENDC)
             hmloss = combined_loss 
        elif (cfg['loss']=="dssim"):
             from NNLosses import DSSIMLoss
             print(bcolors.WARNING,"Using experimental dssim loss..",bcolors.ENDC)
             hmloss=DSSIMLoss 
        else:
             print(bcolors.WARNING,"Using ",cfg['loss']," loss..",bcolors.ENDC)
             hmloss=cfg['loss']
        return hmloss
"""
#-------------------------------------------------------------------------------
def getOptimizerFromCFG(cfg):
   if not 'optimizer' in cfg:
      raise ValueError("Did not find a declaration for optimizer in json configuration")

   if (cfg['optimizer']=='adamwcautious'):
      from NNLosses import AdamWCautious
      optimizer = AdamWCautious(learning_rate=float(cfg['learningRate']),clipnorm=None,clipvalue=1.0)
   elif (cfg['optimizer']=='adam'):
      optimizer = tf.keras.optimizers.Adam(learning_rate=float(cfg['learningRate']),clipnorm=None,clipvalue=1.0,global_clipnorm=None)
   elif (cfg['optimizer']=='adamw'):
      optimizer = tf.keras.optimizers.AdamW(learning_rate=float(cfg['learningRate']),clipnorm=None,clipvalue=1.0,global_clipnorm=None)
   else:
      raise ValueError("Unknown optimizer (",cfg['optimizer'],")")
   return optimizer
#============================================================================================
def check_glove_embeddings_correctly_normalized(array):
   #print(array)
   if np.any((array < -1) | (array > 1)):
        min_val = np.min(array)
        max_val = np.max(array)
        raise ValueError(
            f"Array contains values outside [-1, 1] range. "
            f"Actual range: [{min_val:.4f}, {max_val:.4f}]"
        )
   return True
#============================================================================================
def extract_validation_losses(model, validation_generator, dbValidation):
    """
    Extracts the loss for each validation sample.

    Parameters:
    model (keras.Model): The trained model.
    validation_generator (TrainingDataGenerator): The validation data generator.

    Returns:
    np.ndarray: An array of losses per validation sample.
    """
    sample_losses = []

    batchStart = 0
    batchEnd   = 0
    for batch_idx in range(len(validation_generator)):
        inputs, targets = validation_generator[batch_idx]
        batch_size = inputs.shape[0]
        if (batchEnd==0):
            batchEnd = batch_size

        # Compute per-sample losses
        batch_losses = model.evaluate(inputs, targets, batch_size=batch_size, verbose=0, return_dict=True)
        print("Batch ",batch_idx, " loss ",batch_losses)
        dbValidation.updateEpochResults(batch_losses['loss'], batchStart, batchEnd, 1) 
        
        # If multiple losses are returned, sum them up per sample
        if isinstance(batch_losses, dict):
            total_loss = sum(batch_losses.values())
        else:
            total_loss = batch_losses  # Single loss case

        sample_losses.extend([total_loss / batch_size] * batch_size)  # Distribute equally per sample

        batchStart += batch_size
        batchEnd   += batch_size
    return np.array(sample_losses)
#============================================================================================
#============================================================================================
#============================================================================================
class TrainingDataGeneratorSingleSeq(keras.utils.Sequence):
    def __init__(self, cfg, db, batch_size=32, numberOfTokens=16, numberOfClasses=2037, validation_data=False, labels=None, log_dir=None, returnOutputImages=True, **kwargs):
        super().__init__(**kwargs)  # Ensure Keras properly initializes the dataset class

        self.cfg = cfg
        self.db = db
        self.numberOfSamples = self.db.numberOfSamples
        self.batch_size = batch_size
        self.num_batches = self.numberOfSamples // self.batch_size  # Required for Keras 3.1+
        self.epoch = 1
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.labels = labels
        self.numberOfTokens = numberOfTokens
        self.numberOfClasses    = numberOfClasses
        self.returnOutputImages = returnOutputImages
        self.returnOneHot = True
        self.returnGlove = True
        self.combineData = False
        self.db.shuffle()

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min(start_index + self.batch_size, self.numberOfSamples)

        log_thread_progress("dataloader_python",1,"update_cpu_batch")

        npArrayIn, npArrayOut, npArrayOut16Bit = self.db.get_partial_update_IO_array(start_index, end_index)

        if self.cfg.get("outputTokens", False):
            npArrayOutputList = {}

            if self.returnOneHot:
                npArrayTokensOut = self.db.get_partial_token_array(start_index, end_index, encodeAsSingleMultiLabelToken=True).astype("float32")
                npArrayOutputList["tokens_multihot"] = npArrayTokensOut

            if self.returnGlove:
                npArrayEmbeddingsOut = self.db.get_partial_embedding_array(start_index, end_index).astype("float32")
                if self.combineData:
                    npArrayOutputList["tall"] = npArrayEmbeddingsOut.reshape(self.batch_size, self.db.D * self.numberOfTokens)
                else:
                    for i in range(self.numberOfTokens):
                        npArrayOutputList[f"t{i:02d}"] = npArrayEmbeddingsOut[:, i, :]

            if self.returnOutputImages:
                npArrayOutputList["hm"] = npArrayOut

            log_thread_progress("dataloader_python",0,"update_cpu_batch")
            return npArrayIn, npArrayOutputList
 

        log_thread_progress("dataloader_python",0,"update_cpu_batch")
        return npArrayIn, npArrayOut

    def num_batches(self):
        return self.numberOfSamples // self.batch_size

    def batch_size(self):
        return self.batch_size

    def on_epoch_end(self):
        self.epoch += 1
        if not self.validation_data:
            print(f"\nTrainingDataGenerator on_epoch_end shuffling, next epoch is {self.epoch}")
            if self.epoch < 2:
                self.db.shuffle()
            else:
                self.db.shuffle_based_on_loss()
#============================================================================================
#============================================================================================
#Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly <- Old
#https://www.tensorflow.org/api_docs/python/tf/keras/utils/PyDataset
class TrainingDataGenerator(keras.utils.PyDataset):
    def __init__(self,  cfg, db, batch_size=32, numberOfTokens=16, numberOfClasses=2037, validation_data=False, labels=None,  log_dir=None, returnOutputImages = True, **kwargs):
        super().__init__(**kwargs)
        self.cfg                = cfg 
        self.db                 = db
        self.numberOfSamples    = self.db.numberOfSamples
        self.batch_size         = batch_size
        self.num_batches        = self.numberOfSamples // self.batch_size #<- This is needed for Keras 3.1+ compatibility
        self.epoch              = 1
        self.inputChannels      = db.inChannels
        self.validation_data    = validation_data
        self.log_dir            = log_dir
        self.labels             = labels
        self.numberOfTokens     = numberOfTokens
        self.numberOfClasses    = numberOfClasses
        self.returnOutputImages = returnOutputImages
        self.returnOneHot       = True
        self.returnGlove        = True
        self.returnDescriptor   = cfg["outputDescriptors"]
        self.combineData        = False
        self.db.shuffle() 

    def __len__(self):
        return self.numberOfSamples // self.batch_size

    def __getitem__(self, index):
        self.start_index   = index * self.batch_size
        self.end_index     = min(self.start_index + self.batch_size, self.numberOfSamples)

        log_thread_progress("dataloader_python",1,"update_cpu_batch")
        npArrayIn, npArrayOut, npArrayOut16Bit = self.db.get_partial_update_IO_array(self.start_index, self.end_index)

        
        # ----------------------------------------------------------------------------------------
        # Add a 4th channel: 255 - average of RGB per pixel
        # ----------------------------------------------------------------------------------------
        if (self.inputChannels==4):
          avg_rgb = np.mean(npArrayIn, axis=-1, keepdims=True)  # shape: (batch_size, width, height, 1)
          fourth_channel = 255.0 - avg_rgb                       # shape: (batch_size, width, height, 1)
          npArrayIn = np.concatenate([npArrayIn, fourth_channel], axis=-1)  # shape: (batch_size, width, height, 4)
        # ----------------------------------------------------------------------------------------


        if ('outputTokens' in self.cfg) and (self.cfg['outputTokens']):
          npArrayOutputList = dict() 

          if (self.returnDescriptor):
             npArrayDescriptorsOut = self.db.get_partial_descriptor_array(self.start_index, self.end_index)
             npArrayOutputList["descriptors"] = npArrayDescriptorsOut #DINO Descriptor

          if (self.returnOneHot):
            #----------------------------------------------------------------------------------------------------------------------------
            #Grab one-hot encodings
            #----------------------------------------------------------------------------------------------------------------------------
            if (self.combineData) or (self.returnGlove):
               npArrayTokensOut = self.db.get_partial_token_array(self.start_index, self.end_index, encodeAsSingleMultiLabelToken = True).astype('float32')
               npArrayOutputList["tokens_multihot"] = npArrayTokensOut
               #print("tokens_multihot",npArrayTokensOut.shape)
            else:
               npArrayTokensOut = self.db.get_partial_token_array(self.start_index, self.end_index, encodeAsSingleMultiLabelToken = False).astype('float32')
               npArrayTokensOut = npArrayTokensOut.reshape(self.batch_size, self.numberOfTokens * self.numberOfClasses) 
               npArrayOutputList["tokens_multihot"] = npArrayTokensOut
               #print("tokens_multihot",npArrayTokensOut.shape)
            #----------------------------------------------------------------------------------------------------------------------------

          if (self.returnGlove):
            #----------------------------------------------------------------------------------------------------------------------------
            #Immediately grab GloVe embeddings
            #----------------------------------------------------------------------------------------------------------------------------
            npArrayEmbeddingsOut = self.db.get_partial_embedding_array(self.start_index, self.end_index)
            npArrayEmbeddingsOut = npArrayEmbeddingsOut.astype('float32')

            #Everything seems normalized
            #check_glove_embeddings_correctly_normalized(npArrayEmbeddingsOut)
        
            if (self.combineData):
               npArrayOutputList["tall"] = npArrayEmbeddingsOut.reshape(self.batch_size, self.db.D * self.numberOfTokens)
            else:
               for i in range(self.numberOfTokens):
                 npArrayOutputList["t%02u"%i] = npArrayEmbeddingsOut[:,i,:]
            #----------------------------------------------------------------------------------------------------------------------------
          #print("Heatmap Shape ",npArrayOutputList["hm"].shape)
          if (self.returnOutputImages):
             npArrayOutputList["hm"] = npArrayOut
             if npArrayOut16Bit is not None:
                    npArrayOutputList["hm_16b"] = npArrayOut16Bit


          log_thread_progress("dataloader_python",0,"update_cpu_batch")
          return npArrayIn, npArrayOutputList
        else:
          #Regular just RGB -> heatmap output
          log_thread_progress("dataloader_python",0,"update_cpu_batch")
          return npArrayIn, npArrayOut 

    def num_batches(self):
        return self.numberOfSamples // self.batch_size

    def batch_size(self):
        return self.batch_size

    def on_epoch_end(self):
        self.epoch = self.epoch +1
        if (not self.validation_data):
         print("\nTrainingDataGenerator on_epoch_end shuffling, next epoch is ",self.epoch)
         if (self.epoch < 2):
          #in the beginning everything is kind of random so do normal random shuffle
          self.db.shuffle()
         else:
          #After we have accumulated some losses try to shuffle using the losses in an attempt to make training more interesting
          #self.db.shuffle()
          self.db.shuffle_based_on_loss() #<- TODO: This may need to be deactivated if training is unstable
          pass
#============================================================================================
"""
#Experiment directly using a TF Data Generator (to hopefully improve performance) 
def TrainingDataGeneratorTF(cfg, db, batch_size=32, numberOfTokens=16, numberOfClasses=2037,validation_data=False, labels=None, log_dir=None, returnOutputImages=True, workers=1, use_multiprocessing=False, max_queue_size=1):
    # workers=1, use_multiprocessing=False, max_queue_size=1 are ignored but included to ensure compatibility 
    numberOfSamples = db.numberOfSamples
    steps_per_epoch = numberOfSamples // batch_size
    returnOneHot       = True
    returnGlove        = True
    combineData        = False
    D=300

    def generator():

        for i in range(steps_per_epoch):
            start_index = i * batch_size
            end_index = min(start_index + batch_size, numberOfSamples)
            
            log_thread_progress("dataloader_python", 1, "update_cpu_batch")
            npArrayIn, npArrayOut, _ = db.get_partial_update_IO_array(start_index, end_index)

            output_dict = {}

            if 'outputTokens' in cfg and cfg['outputTokens']:
                if returnOneHot:
                    if combineData or returnGlove:
                        npArrayTokensOut = db.get_partial_token_array(start_index, end_index, encodeAsSingleMultiLabelToken=True).astype('float32')
                        output_dict["tokens_multihot"] = npArrayTokensOut
                    else:
                        npArrayTokensOut = db.get_partial_token_array(start_index, end_index, encodeAsSingleMultiLabelToken=False).astype('float32')
                        npArrayTokensOut = npArrayTokensOut.reshape(batch_size, numberOfTokens * numberOfClasses)
                        output_dict["tokens_multihot"] = npArrayTokensOut

                if returnGlove:
                    npArrayEmbeddingsOut = db.get_partial_embedding_array(start_index, end_index).astype('float32')
                    if combineData:
                        output_dict["tall"] = npArrayEmbeddingsOut.reshape(batch_size, -1)
                    else:
                        for i in range(numberOfTokens):
                            output_dict[f"t{i:02d}"] = npArrayEmbeddingsOut[:, i, :]

                if returnOutputImages:
                    output_dict["hm"] = npArrayOut
                log_thread_progress("dataloader_python", 0, "update_cpu_batch")
                yield npArrayIn, output_dict
            else:
                log_thread_progress("dataloader_python", 0, "update_cpu_batch")
                yield npArrayIn, npArrayOut

    # Define the output_signature with correct types
    sample_input = tf.TensorSpec(shape=(batch_size, db.inHeight, db.inWidth, db.inChannels), dtype=tf.uint8)
    if 'outputTokens' in cfg and cfg['outputTokens']:
        out_sig = {}

        if returnOneHot:
            if combineData or returnGlove:
                out_sig["tokens_multihot"] = tf.TensorSpec(shape=(batch_size, numberOfClasses), dtype=tf.float32)
            else:
                out_sig["tokens_multihot"] = tf.TensorSpec(shape=(batch_size, numberOfTokens * numberOfClasses), dtype=tf.float32)

        if returnGlove:
            if combineData:
                out_sig["tall"] = tf.TensorSpec(shape=(batch_size, numberOfTokens * D), dtype=tf.float32)
            else:
                for i in range(numberOfTokens):
                    out_sig[f"t{i:02d}"] = tf.TensorSpec(shape=(batch_size, D), dtype=tf.float32)

        if returnOutputImages:
            out_sig["hm"] = tf.TensorSpec(shape=(batch_size, db.outHeight, db.outWidth, db.out8BitChannels), dtype=tf.int8)

        output_signature = (sample_input, out_sig)
    else:
        sample_output = tf.TensorSpec(shape=(batch_size, db.outHeight, db.outWidth, db.out8BitChannels), dtype=tf.int8)
        output_signature = (sample_input, sample_output)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
"""
#============================================================================================
"""
class TrainingDataGeneratorSeq(keras.utils.Sequence):
    def __init__(self, cfg, db, batch_size=32, numberOfTokens=16, numberOfClasses=2037, validation_data=False, labels=None, log_dir=None, returnOutputImages=True, max_queue_size=4, **kwargs):
        super().__init__(**kwargs)

        self.cfg = cfg
        self.db = db
        self.numberOfSamples = self.db.numberOfSamples
        self.batch_size = batch_size
        self.num_batches = self.numberOfSamples // self.batch_size
        self.epoch = 1
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.labels = labels
        self.numberOfTokens = numberOfTokens
        self.numberOfClasses    = numberOfClasses
        self.returnOutputImages = returnOutputImages
        self.returnOneHot = True
        self.returnGlove = True
        self.combineData = False
        self.db.shuffle()

        # Queue to store prefetched batches
        self.batch_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._prefetch_batches, daemon=True)
        self.worker_thread.start()

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        #Retrieve preloaded batch from queue.
        return self.batch_queue.get()  # Blocks if queue is empty

    def _prefetch_batches(self):
        #Background thread that preloads batches into queue.
        while not self.stop_event.is_set():
            for index in range(self.num_batches):
                if self.stop_event.is_set():
                    break

                # Wait if the queue is full
                while self.batch_queue.qsize() >= self.batch_queue.maxsize and not self.stop_event.is_set():
                    threading.Event().wait(0.01)  # Small sleep to avoid busy waiting

                batch = self._load_batch(index)
                self.batch_queue.put(batch)

    def _load_batch(self, index):
        #Thread-safe method to load a batch.
        start_index = index * self.batch_size
        end_index = min(start_index + self.batch_size, self.numberOfSamples)

        npArrayIn, npArrayOut, npArrayOut16Bit = self.db.get_partial_update_IO_array(start_index, end_index)

        if self.cfg.get("outputTokens", False):
            npArrayOutputList = {}

            if self.returnOneHot:
                npArrayTokensOut = self.db.get_partial_token_array(start_index, end_index, encodeAsSingleMultiLabelToken=True).astype("float32")
                npArrayOutputList["tokens_multihot"] = npArrayTokensOut

            if self.returnGlove:
                npArrayEmbeddingsOut = self.db.get_partial_embedding_array(start_index, end_index).astype("float32")
                if self.combineData:
                    npArrayOutputList["tall"] = npArrayEmbeddingsOut.reshape(self.batch_size, self.db.D * self.numberOfTokens)
                else:
                    for i in range(self.numberOfTokens):
                        npArrayOutputList[f"t{i:02d}"] = npArrayEmbeddingsOut[:, i, :]

            if self.returnOutputImages:
                npArrayOutputList["hm"] = npArrayOut

            return npArrayIn, npArrayOutputList

        return npArrayIn, npArrayOut

    def on_epoch_end(self):
        self.epoch += 1
        if not self.validation_data:
            print(f"\nTrainingDataGenerator on_epoch_end shuffling, next epoch is {self.epoch}")
            if self.epoch < 2:
                self.db.shuffle()
            else:
                self.db.shuffle_based_on_loss()
        
        # Clear queue and refill it for next epoch
        while not self.batch_queue.empty():
            self.batch_queue.get()

    def num_batches(self):
        return self.numberOfSamples // self.batch_size

    def batch_size(self):
        return self.batch_size

    def stop(self):
        #Stops the worker thread gracefully.
        self.stop_event.set()
        self.worker_thread.join()
"""
#============================================================================================
def custom_lr_scheduler(epoch,startLoss=0.0001,endLoss=0.000015):
    #--------------------------
    finalEpochBeforeFlatLine=100
    decimals=5
    maximum=startLoss
    minimum=endLoss
    #--------------------------
    if epoch == 1:
        return round(maximum,decimals)
    elif epoch < finalEpochBeforeFlatLine:
        return round((maximum - minimum) * ((1 - (epoch - 1) / (finalEpochBeforeFlatLine-1) ) ** 2) + minimum,decimals)
    else:
        return round(minimum,decimals)

#============================================================================================
def custom_lr_schedulerWarmup(epoch, warmup_epochs=10, total_epochs=200, initial_lr=0.00015, target_lr=0.001, minimum=0.00015):
    """
    Custom learning rate scheduler with warmup and cosine decay.
    Inspired from : https://arxiv.org/abs/2406.09405v1
    
    Parameters:
    - epoch (int): The current epoch number.
    - warmup_epochs (int): The number of epochs for linear warmup.
    - total_epochs (int): The total number of epochs for training.
    - initial_lr (float): The initial learning rate at the start of warmup.
    - target_lr (float): The target learning rate after warmup.
    - minimum (float): The minimum constant learning rate at the end of epochs.
    
    Returns:
    - float: The adjusted learning rate for the given epoch.
    """
    if epoch <= warmup_epochs:
        # Linear warmup
        lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
    else:
        # Cosine decay
        decay_epochs = total_epochs - warmup_epochs
        decay_ratio = (epoch - warmup_epochs) / decay_epochs
        lr = target_lr * 0.5 * (1 + math.cos(math.pi * decay_ratio))

    if lr < minimum:
         lr = minimum    

    return round(lr, 6)
#============================================================================================
def jointGradientScheduler(epoch, warmup_epochs=10, mature_epochs=200, total_epochs=250,
                           max_joints_gradient=23, min_joints_gradient=8,
                           max_paf_gradient=6, min_paf_gradient=2):

    joint_gradient = min_joints_gradient
    paf_gradient   = min_paf_gradient

    if epoch < warmup_epochs:
        #Stick gradient to max value
        #joint_gradient = max_joints_gradient
        #paf_gradient   = max_paf_gradient

        # Linearly decay to half of max values during warmup
        t = epoch / warmup_epochs  # normalized [0, 1)
        joint_gradient = max_joints_gradient - t * (max_joints_gradient / 2)
        paf_gradient   = max_paf_gradient    - t * (max_paf_gradient / 2)

    elif epoch < mature_epochs:

        #Oscillate with lower magnitude (2/6/25)
        max_joints_gradient = int(max_joints_gradient/2)
        max_paf_gradient    = int(max_paf_gradient/2)

        # Number of steps per area (area = one direction: up or down)
        steps = max(max_joints_gradient - min_joints_gradient,
                    max_paf_gradient - min_paf_gradient)

        # Total number of areas (each with 'steps' epochs)
        total_osc_epochs = mature_epochs - warmup_epochs
        total_areas = total_osc_epochs // steps

        # Current area (0-based)
        area_index = (epoch - warmup_epochs) // steps
        step_index = (epoch - warmup_epochs) % steps
        t = step_index / steps  # normalized position in current area [0,1)

        # Determine direction: even index = down, odd = up
        if area_index % 2 == 0:  # descending
            joint_gradient = max_joints_gradient - (max_joints_gradient - min_joints_gradient) * t
            paf_gradient   = max_paf_gradient    - (max_paf_gradient   - min_paf_gradient)   * t
        else:  # ascending
            joint_gradient = min_joints_gradient + (max_joints_gradient - min_joints_gradient) * t
            paf_gradient   = min_paf_gradient    + (max_paf_gradient   - min_paf_gradient)   * t
    else:
        joint_gradient = min_joints_gradient
        paf_gradient   = min_paf_gradient

    return int(joint_gradient), int(paf_gradient)
#============================================================================================
def jointGradientSchedulerSimple(epoch, warmup_epochs=10, max_joints_gradient=23, min_joints_gradient=8, max_paf_gradient=6, min_paf_gradient=2):
    joint_gradient = min_joints_gradient
    paf_gradient   = min_paf_gradient
    if (min_joints_gradient!=max_joints_gradient) or (min_paf_gradient!=max_paf_gradient):
      if epoch < warmup_epochs:
        # Linearly decay to target during warmup
        t = epoch / warmup_epochs  # normalized [0, 1)
        joint_gradient = max_joints_gradient - t * (max_joints_gradient - min_joints_gradient)
        paf_gradient   = max_paf_gradient    - t * (max_paf_gradient    - min_paf_gradient)
    return int(joint_gradient), int(paf_gradient)
#============================================================================================
class DataAugmentation(keras.callbacks.Callback):
    def __init__(self, cfg, db=None):
        super().__init__()
        self.cfg   = cfg 
        self.epochLimit = cfg["heatmapReductionEpochLimit"]
        self.db    = db 
        self.epoch = 1
        self.time  = time.time()

    def clear_gpu_memory(self, tensors):
        # Clear GPU memory for a list of tensors
        for tensor in tensors:
            del tensor

    #https://keras.io/guides/writing_your_own_callbacks/
    def on_train_batch_end(self, batch, logs=None):
        if (self.db):
         keys = list(logs.keys())
         #self.db.printReadSpeed()
         if ("loss" in keys):
           self.db.updateEpochResults(logs['loss'], self.db.lastStartSample, self.db.lastEndSample, self.epoch)

    def on_epoch_start(self, epoch, logs=None):
        self.time  = time.time() #Is this not executed ?

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        keys = list(logs.keys())
        #print("End epoch ",epoch+1," of training")
        #print("Got log keys:", keys))

        totalSeconds   = time.time() - self.time 
        cpuTimeSeconds = self.db.cpuTimeSeconds
        gpuTimeSeconds = totalSeconds - cpuTimeSeconds
        print("Time it took for epoch %u | CPU : %0.02f sec | GPU : %0.2f sec | Total : %0.02f sec" % (epoch+1,cpuTimeSeconds,gpuTimeSeconds,totalSeconds))
        self.db.printReadSpeed()
        self.db.cpuTimeSeconds = 0 #Reset counter
        self.time  = time.time()   #Reset GPU time (although on_epoch_start should also reset it)

        #print("Doing garbage collection ..",end="")
        gc.collect()
        #print(" ok")
        if (self.db):
            #logs just logs the overall loss, so it is useless..
            #if ("loss" in keys) and ("learning_rate" in keys):
            """
            if ("loss" in keys) and ("learning_rate" in keys):
                #This should have already happened in on_train_batch_end
                #self.db.updateEpochResults(logs['loss'], self.db.lastStartSample, self.db.lastEndSample, epoch)
 
                #Do not dump loss for each epoch ?
                #self.db.dump_sample_report() <- reduce disk spam
                pass
            """

            oldGS=self.db.gradientSize
            oldPS=self.db.PAFSize
            """
            #Go up and down
            self.db.gradientSize,self.db.PAFSize = jointGradientScheduler(epoch, 
                                                                          warmup_epochs=self.cfg["earlyStoppingStart"],
                                                                          mature_epochs=(self.cfg["epochs"] - self.cfg["epochs"]//5), total_epochs=self.cfg["epochs"],
                                                                          max_joints_gradient=self.cfg["heatmapGradientSize"],
                                                                          min_joints_gradient=self.cfg["heatmapGradientSizeMinimum"],
                                                                          max_paf_gradient=self.cfg["heatmapPAFSize"],
                                                                          min_paf_gradient=self.cfg["heatmapPAFSizeMinimum"])
            """

            #Constant minimum size ( seems to eliminate joints altogether )
            #self.db.gradientSize = self.cfg["heatmapGradientSizeMinimum"]
            #self.db.PAFSize      = self.cfg["heatmapPAFSizeMinimum"]

            #Decay to minimum after warmup epochs
            self.db.gradientSize,self.db.PAFSize = jointGradientSchedulerSimple(epoch, 
                                                                                warmup_epochs=self.cfg["earlyStoppingStart"],
                                                                                max_joints_gradient=self.cfg["heatmapGradientSize"],
                                                                                min_joints_gradient=self.cfg["heatmapGradientSizeMinimum"],
                                                                                max_paf_gradient=self.cfg["heatmapPAFSize"],
                                                                                min_paf_gradient=self.cfg["heatmapPAFSizeMinimum"]) 

            if ( (oldGS!=self.db.gradientSize) or (oldPS!=self.db.PAFSize) ):
               print(bcolors.OKGREEN,"Set gradient size from ",oldGS," to ",self.db.gradientSize,bcolors.ENDC, end=" / ")
               print(bcolors.OKGREEN,"Set PAF size from ",oldPS," to ",self.db.PAFSize,bcolors.ENDC) 
 
#============================================================================================
#============================================================================================
def weighted_token_loss(y_true, y_pred):
    # Use BinaryCrossentropy from Keras
    token_loss_function = keras.losses.BinaryCrossentropy(from_logits=False) 
    #token_loss_function = keras.losses.MeanSquaredError() 
    
    # Compute the original loss
    original_loss = token_loss_function(y_true, y_pred)
    
    # Multiply by the weight
    #weighted_loss = 1.0 *  tf.exp(original_loss) #Try perplexity loss e^loss
    weighted_loss = 10.0 * original_loss 
    
    return weighted_loss
#============================================================================================

if __name__ == '__main__':
       for epoch in range(250): 
         jG,pG = jointGradientScheduler(epoch, warmup_epochs=10, mature_epochs=200, total_epochs=250,
                           max_joints_gradient=23, min_joints_gradient=8,
                           max_paf_gradient=6, min_paf_gradient=2)
         print("Epoch ",epoch," Joints:",jG, " PAF:",pG)

