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

LOG_THREADING_INFORMATION = False
tickBase = 0
#----------------------------------------------
useGPU = True
if (len(sys.argv) > 1):
    #print('Argument List:', str(sys.argv))
    for i in range(0, len(sys.argv)):
        if (sys.argv[i] == "--cpu"):
            useGPU = False
# Set CUDA_VISIBLE_DEVICES to an empty string to force TensorFlow to use the CPU
if (not useGPU):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # <- Force CPU
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

    from ymapnet.core.NNConverter import saveNNModel
except Exception as e:
    print("An exception occurred:", str(e))
    print("Issue:\n source venv/bin/activate")
    print("Before running this script")
    sys.exit(1)
#-------------------------------------------------------------------------------
from ymapnet.core.NNLosses import RSquaredMetric, HeatmapDistanceMetric, AdamWCautious
from ymapnet.utils.tools import bcolors, read_json_file, checkIfPathExists, checkIfFileExists, convert_bytes


#-------------------------------------------------------------------------------
def logTrainingHistory(cfg, log_dir, log_name, history):
    # Extract training and validation loss history
    training_loss = history.history['loss']
    validation_loss = None
    if 'val_loss' in history.history:
        validation_loss = history.history['val_loss']

    # Save loss history to a text file
    with open('%s/%s' % (log_dir, log_name), 'w') as f:
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
            # Accept either a config dict (logged as key: value lines) or a plain
            # string (e.g. a TrainingError message) — callers pass both.
            if isinstance(cfg, dict):
                params_str = "\n".join([f"{key}: {value}" for key, value in cfg.items()])
            else:
                params_str = str(cfg)
            # Log the parameters as text
            tf.summary.text(subject, params_str, step=0)
    except Exception as e:
        print(f"Error storing logging parameters in tensorboard: {e}")


#-------------------------------------------------------------------------------
def logSomeInputsAndOutputs(inputs, outputs, outputs16B, labels, log_dir, samples, description="Validation Data",
                            localSave=False):
    try:
        # Create a summary writer
        image_log = tf.summary.create_file_writer(log_dir)

        with image_log.as_default():

            sample_indices = np.random.choice(len(inputs), size=min(samples, len(inputs)), replace=False)

            for logID in sample_indices:
                #Store the image as float32 [0..1] RGB to make sure tensorboard visualizes it correctly
                image_as_float = inputs[logID].astype(np.float32)
                image = image_as_float / 255.0

                # Convert input and output arrays to TensorFlow tensors
                bgr_image_tensor = tf.convert_to_tensor([image], dtype=tf.float32)

                # Swap BGR to RGB not needed
                #rgb_image_tensor = tf.reverse(bgr_image_tensor, axis=[-1])

                # Write input image summary
                tf.summary.image(f"{description} Image {logID} Input", bgr_image_tensor, step=logID)  #rgb_image_tensor

                # Write output image summary
                numberOfHeatmaps = outputs.shape[3]  #Should be 18 ?
                for heatmapID in range(0, numberOfHeatmaps):
                    heatmap = outputs[logID, :, :, heatmapID]
                    #print(f"Heatmap {heatmapID} dimensions: {heatmap.shape}")
                    # Add batch and channel dimensions
                    heatmapS = np.squeeze(heatmap)
                    heatmapS = np.expand_dims(heatmapS, axis=-1)
                    heatmap_as_float = (heatmapS.astype(np.float32) + 120.0) / 240.0
                    output_image_tensor = tf.convert_to_tensor([heatmap_as_float], dtype=tf.float32)

                    thisOutputlabel = "#%u" % heatmapID
                    if (heatmapID < len(labels)):
                        thisOutputlabel = labels[heatmapID]

                    if (localSave):
                        print("Saving local heatmap for sample ", logID)
                        cv2.imwrite('heatmap_in_%u.png' % (logID), image_as_float)
                        cv2.imwrite('heatmap_%u_%u.png' % (logID, heatmapID), heatmap_as_float)
                    tf.summary.image(f"{description} Image {logID} Output / {thisOutputlabel}", output_image_tensor,
                                     step=logID)

                numberOfHeatmaps = outputs16B.shape[3]  #Should be 18 ?
                for heatmapID in range(0, numberOfHeatmaps):
                    heatmap = outputs16B[logID, :, :, heatmapID]
                    #print(f"Heatmap {heatmapID} dimensions: {heatmap.shape}")
                    # Add batch and channel dimensions
                    heatmapS = np.squeeze(heatmap)
                    heatmapS = np.expand_dims(heatmapS, axis=-1)
                    heatmap_as_float = (heatmapS.astype(np.float32) + 120.0) / 240.0
                    output_image_tensor = tf.convert_to_tensor([heatmap_as_float], dtype=tf.float32)
                    thisOutputlabel = "#%u-16BIT" % heatmapID

                    if (localSave):
                        print("Saving local heatmap for sample ", logID)
                        cv2.imwrite('heatmap16B_%u_%u.png' % (logID, heatmapID), heatmap_as_float)
                    tf.summary.image(f"{description} Image {logID} Output / {thisOutputlabel}", output_image_tensor,
                                     step=logID)

    except Exception as e:
        print(f"Error storing image in tensorboard: {e}")
        sys.exit(1)


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def get_tick_count_microseconds_mn():
    """Simulates GetTickCountMicrosecondsMN from C."""
    global tickBase
    rawTicks = int(time.time() * 1_000_000)
    if (tickBase == 0):
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
    print("Tensorflow version : ", tf.__version__)
    print("Keras version      : ", keras.__version__)  #<- no longer available in TF-2.13
    print("Numpy version      : ", np.__version__)
    #-----------------------------
    from tensorflow.python.platform import build_info as tf_build_info
    print("TF/CUDA version    : ", tf_build_info.build_info['cuda_version'])
    print("TF/CUDNN version   : ", tf_build_info.build_info['cudnn_version'])
    print("Use GPU            : ", useGPU)
    #-----------------------------
    if useGPU:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for gpuID, gpu in enumerate(physical_devices):
                print("GPU #", gpuID, " Name:", gpu.name)
                try:
                    # Note: The following code may not be available in older versions of TensorFlow
                    memory_info = tf.config.experimental.get_memory_info('GPU:%u' % gpuID)
                    print("GPU #", gpuID, " Memory Currently Used (in MB):", memory_info['current'] / (1024**2))
                    print("GPU #", gpuID, " Memory Peak Used (in MB):", memory_info['peak'] / (1024**2))
                except Exception as e:
                    print(f"Error getting memory info for GPU #{gpuID}: {e}")
        else:
            print("No GPU available.")
    print("Threads Available:")
    os.system("cat /proc/cpuinfo | grep processor | wc -l")
    os.system("lscpu")
    os.system("numactl --hardware")
    os.system("cat /proc/pressure/memory")  #<- Memory Pressure
    os.system("df -h /dev/shm")
    os.system("df -h /")
    os.system("free -h")  #<- Memory In General
    print("")


#-------------------------------------------------------------------------------
"""
def getLossFromCFG(cfg):
        hmloss = None
        if (cfg['loss']=="mse"):
             from ymapnet.core.NNLosses import HeatmapCoreLoss
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
             from ymapnet.core.NNLosses import DSSIMLoss
             print(bcolors.WARNING,"Using experimental dssim loss..",bcolors.ENDC)
             hmloss=DSSIMLoss 
        else:
             print(bcolors.WARNING,"Using ",cfg['loss']," loss..",bcolors.ENDC)
             hmloss=cfg['loss']
        return hmloss
"""


#-------------------------------------------------------------------------------
def getOptimizerFromCFG(cfg, globalClipNorm=None):
    if not 'optimizer' in cfg:
        raise ValueError("Did not find a declaration for optimizer in json configuration")

    # When globalClipNorm is set (multi-GPU), use global norm clipping instead of
    # per-element clipvalue. With all-reduce, rare high-weight samples have
    # num_gpus x more gradient impact than single-GPU (diluted by fewer replicas
    # than samples); global_clipnorm bounds the total update size regardless.
    wd = float(cfg.get('weightDecay', 0.0))
    beta_1 = float(cfg.get('optimizerBeta1', 0.9))
    beta_2 = float(cfg.get('optimizerBeta2', 0.999))
    epsilon = float(cfg.get('optimizerEpsilon', 1e-7))
    clip_value = None if globalClipNorm else float(cfg.get('optimizerClipValue', 1.0))
    global_clip = globalClipNorm

    # AdamWCautious was missing global_clipnorm, which meant multi-GPU
    # training fell back to per-element clipvalue only.  With gradient
    # all-reduce, rare high-magnitude samples are amplified across replicas
    # and global_clipnorm is the correct way to bound total update size.
    if (cfg['optimizer'] == 'adamwcautious'):
        from ymapnet.core.NNLosses import AdamWCautious
        optimizer = AdamWCautious(learning_rate=float(cfg['learningRate']), beta_1=beta_1, beta_2=beta_2,
                                  epsilon=epsilon, weight_decay=wd or None, clipnorm=None, clipvalue=clip_value,
                                  global_clipnorm=global_clip)
    elif (cfg['optimizer'] == 'adam'):
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(cfg['learningRate']), beta_1=beta_1, beta_2=beta_2,
                                             epsilon=epsilon, clipnorm=None, clipvalue=clip_value,
                                             global_clipnorm=global_clip)
    elif (cfg['optimizer'] == 'adamw'):
        optimizer = tf.keras.optimizers.AdamW(learning_rate=float(cfg['learningRate']), weight_decay=wd, beta_1=beta_1,
                                              beta_2=beta_2, epsilon=epsilon, clipnorm=None, clipvalue=clip_value,
                                              global_clipnorm=global_clip)
    else:
        raise ValueError("Unknown optimizer (", cfg['optimizer'], ")")

    if cfg.get('mixedPrecision', False):
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.global_policy()
        # LossScaleOptimizer applies dynamic loss scaling to prevent gradient
        # underflow in float16 (which has only 5 exponent bits, max ~65504).
        # bfloat16 shares float32's 8 exponent bits and full dynamic range,
        # so loss scaling is unnecessary and would only add overhead.
        if policy.compute_dtype == 'float16':
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
            print(bcolors.WARNING, "Wrapping optimizer with LossScaleOptimizer (float16 policy)", bcolors.ENDC)
        else:
            print(bcolors.OKGREEN, "Skipping LossScaleOptimizer (bfloat16 has float32 dynamic range)", bcolors.ENDC)

    return optimizer


#============================================================================================
def check_vram_and_suggest_batch_size(cfg, threshold_pct=80.0):
    """
    Queries VRAM usage across all visible GPUs and prints a suggestion to
    increase batchSize if utilization is below threshold_pct.

    Returns a list of dicts with keys: gpu_index, used_mb, total_mb, utilization_pct
    """
    import subprocess
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("check_vram_and_suggest_batch_size: nvidia-smi failed")
            return []
    except FileNotFoundError:
        print("check_vram_and_suggest_batch_size: nvidia-smi not found")
        return []

    stats = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        gpu_idx, used_mb, total_mb = int(parts[0]), int(parts[1]), int(parts[2])
        utilization_pct = 100.0 * used_mb / total_mb if total_mb > 0 else 0.0
        stats.append(dict(gpu_index=gpu_idx, used_mb=used_mb, total_mb=total_mb, utilization_pct=utilization_pct))
        print(bcolors.OKGREEN, f"GPU {gpu_idx}: {used_mb} MiB / {total_mb} MiB  ({utilization_pct:.1f}% VRAM used)",
              bcolors.ENDC)

    low_gpus = [s for s in stats if s['utilization_pct'] < threshold_pct]
    if low_gpus:
        current_batch = cfg.get('batchSize', '?')
        headroom_pcts = [threshold_pct - s['utilization_pct'] for s in low_gpus]
        avg_headroom = sum(headroom_pcts) / len(headroom_pcts)
        # Rough estimate: headroom maps linearly to batch size budget
        if isinstance(current_batch, int) and current_batch > 0:
            suggested = int(current_batch * (1.0 + avg_headroom / 100.0))
            # Round down to nearest multiple of 4 for alignment
            suggested = max(current_batch + 1, (suggested // 4) * 4)
            print(
                bcolors.WARNING, f"VRAM utilization is below {threshold_pct:.0f}% on {len(low_gpus)} GPU(s). "
                f"Consider increasing batchSize from {current_batch} to ~{suggested} "
                f"to improve GPU utilization.", bcolors.ENDC)
        else:
            print(
                bcolors.WARNING, f"VRAM utilization is below {threshold_pct:.0f}% on {len(low_gpus)} GPU(s). "
                f"Consider increasing batchSize.", bcolors.ENDC)

    return stats


#============================================================================================
def check_glove_embeddings_correctly_normalized(array):
    #print(array)
    if np.any((array < -1) | (array > 1)):
        min_val = np.min(array)
        max_val = np.max(array)
        raise ValueError(f"Array contains values outside [-1, 1] range. "
                         f"Actual range: [{min_val:.4f}, {max_val:.4f}]")
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
    batchEnd = 0
    # This runs AFTER trainYMAPNet's portability recompile (loss="mse", a single
    # blanket loss). With a *dict* of targets Keras flattens the keys
    # ALPHABETICALLY and pairs them positionally with the model's structural
    # output order — the two only matched by coincidence (hm, hm_16b, t00..t07,
    # tokens_multihot happens to be alphabetical). hm_nll broke that: it is
    # alphabetically third but structurally the LAST output, mispairing its
    # [B,H,W,4] target with t00 [B,300]. Reorder the dict into an explicit list
    # following model.output_names so the pairing is always correct.
    output_names = list(getattr(model, 'output_names', None) or [])

    for batch_idx in range(len(validation_generator)):
        inputs, targets = validation_generator[batch_idx]
        batch_size = inputs.shape[0]
        if (batchEnd == 0):
            batchEnd = batch_size

        if isinstance(targets, dict) and output_names and all(n in targets for n in output_names):
            targets = [targets[n] for n in output_names]

        # Compute per-sample losses
        batch_losses = model.evaluate(inputs, targets, batch_size=batch_size, verbose=0, return_dict=True)
        print("Batch ", batch_idx, " loss ", batch_losses)
        dbValidation.updateEpochResults(batch_losses['loss'], batchStart, batchEnd, 1)

        # If multiple losses are returned, sum them up per sample
        if isinstance(batch_losses, dict):
            total_loss = sum(batch_losses.values())
        else:
            total_loss = batch_losses  # Single loss case

        sample_losses.extend([total_loss / batch_size] * batch_size)  # Distribute equally per sample

        batchStart += batch_size
        batchEnd += batch_size
    return np.array(sample_losses)


#============================================================================================
#============================================================================================
#============================================================================================
class TrainingDataGeneratorSingleSeq(keras.utils.Sequence):

    def __init__(self, cfg, db, batch_size=32, numberOfTokens=16, numberOfClasses=2037, validation_data=False,
                 labels=None, log_dir=None, returnOutputImages=True, **kwargs):
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
        self.numberOfClasses = numberOfClasses
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

        log_thread_progress("dataloader_python", 1, "update_cpu_batch")

        npArrayIn, npArrayOut, npArrayOut16Bit = self.db.get_partial_update_IO_array(start_index, end_index)

        if self.cfg.get("outputTokens", False):
            npArrayOutputList = {}

            if self.returnOneHot:
                npArrayTokensOut = self.db.get_partial_token_array(start_index, end_index,
                                                                   encodeAsSingleMultiLabelToken=True).astype("float32")
                npArrayOutputList["tokens_multihot"] = npArrayTokensOut

            if self.returnGlove:
                npArrayEmbeddingsOut = self.db.get_partial_embedding_array(start_index, end_index).astype("float32")
                if self.combineData:
                    npArrayOutputList["tall"] = npArrayEmbeddingsOut.reshape(self.batch_size,
                                                                             self.db.D * self.numberOfTokens)
                else:
                    for i in range(self.numberOfTokens):
                        npArrayOutputList[f"t{i:02d}"] = npArrayEmbeddingsOut[:, i, :]

            if self.returnOutputImages:
                npArrayOutputList["hm"] = npArrayOut

            log_thread_progress("dataloader_python", 0, "update_cpu_batch")
            return npArrayIn, npArrayOutputList

        log_thread_progress("dataloader_python", 0, "update_cpu_batch")
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

    def __init__(self, cfg, db, batch_size=32, numberOfTokens=16, numberOfClasses=2037, validation_data=False,
                 labels=None, log_dir=None, returnOutputImages=True, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.db = db
        self.numberOfSamples = self.db.numberOfSamples
        self.batch_size = batch_size
        self.num_batches = self.numberOfSamples // self.batch_size  #<- This is needed for Keras 3.1+ compatibility
        self.epoch = 1
        self.inputChannels = db.inChannels
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.labels = labels
        self.numberOfTokens = numberOfTokens
        self.numberOfClasses = numberOfClasses
        self.returnOutputImages = returnOutputImages
        self.returnOneHot = True
        self.returnGlove = True
        self.returnDescriptor = cfg["outputDescriptors"]
        self.combineData = False
        self.db.shuffle()

    def __len__(self):
        return self.numberOfSamples // self.batch_size

    def __getitem__(self, index):
        self.start_index = index * self.batch_size
        self.end_index = min(self.start_index + self.batch_size, self.numberOfSamples)

        log_thread_progress("dataloader_python", 1, "update_cpu_batch")
        npArrayIn, npArrayOut, npArrayOut16Bit = self.db.get_partial_update_IO_array(self.start_index, self.end_index)

        # ----------------------------------------------------------------------------------------
        # Add a 4th channel: 255 - average of RGB per pixel
        # ----------------------------------------------------------------------------------------
        if (self.inputChannels == 4):
            avg_rgb = np.mean(npArrayIn, axis=-1, keepdims=True)  # shape: (batch_size, width, height, 1)
            fourth_channel = 255.0 - avg_rgb  # shape: (batch_size, width, height, 1)
            npArrayIn = np.concatenate([npArrayIn, fourth_channel], axis=-1)  # shape: (batch_size, width, height, 4)
        # ----------------------------------------------------------------------------------------

        if ('outputTokens' in self.cfg) and (self.cfg['outputTokens']):
            npArrayOutputList = dict()

            if (self.returnDescriptor):
                npArrayDescriptorsOut = self.db.get_partial_descriptor_array(self.start_index, self.end_index)
                npArrayOutputList["descriptors"] = npArrayDescriptorsOut  #DINO Descriptor

            if (self.returnOneHot):
                #----------------------------------------------------------------------------------------------------------------------------
                #Grab one-hot encodings
                #----------------------------------------------------------------------------------------------------------------------------
                if (self.combineData) or (self.returnGlove):
                    npArrayTokensOut = self.db.get_partial_token_array(
                        self.start_index, self.end_index, encodeAsSingleMultiLabelToken=True).astype('float32')
                    npArrayOutputList["tokens_multihot"] = npArrayTokensOut
                    #print("tokens_multihot",npArrayTokensOut.shape)
                else:
                    npArrayTokensOut = self.db.get_partial_token_array(
                        self.start_index, self.end_index, encodeAsSingleMultiLabelToken=False).astype('float32')
                    npArrayTokensOut = npArrayTokensOut.reshape(self.batch_size,
                                                                self.numberOfTokens * self.numberOfClasses)
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
                    npArrayOutputList["tall"] = npArrayEmbeddingsOut.reshape(self.batch_size,
                                                                             self.db.D * self.numberOfTokens)
                else:
                    for i in range(self.numberOfTokens):
                        npArrayOutputList["t%02u" % i] = npArrayEmbeddingsOut[:, i, :]
                #----------------------------------------------------------------------------------------------------------------------------
            #print("Heatmap Shape ",npArrayOutputList["hm"].shape)
            if (self.returnOutputImages):
                npArrayOutputList["hm"] = npArrayOut
                if npArrayOut16Bit is not None:
                    npArrayOutputList["hm_16b"] = npArrayOut16Bit
                if self.cfg.get('useGeolocationHead', False) and ('geolocation' in getattr(self.db, 'channel_ranges', {})):
                    # geo_grid GT (Experiment U): reuse the 8-bit geo channel as a per-sample
                    # world-density distribution (sum=1) for the KL loss. No downsample (Tier 1, 256x256).
                    _gs = self.db.channel_ranges['geolocation'][0]
                    _minv = float(self.cfg.get('heatmapDeactivated', -120))
                    _maxv = float(self.cfg.get('heatmapActive', 120))
                    _geo = npArrayOut[..., _gs].astype('float32')                  # (B,256,256) in [MINV,MAXV]
                    _geo = np.clip((_geo - _minv) / (_maxv - _minv), 0.0, 1.0)     # -> [0,1]
                    # Downsample 256x256 -> teacher NATIVE grid (64x128): the 8-bit channel is a
                    # nearest x4/x2 upsample, so strided slicing recovers the native cells exactly.
                    _gh = int(self.cfg.get('geoGridHeight', 64)); _gw = int(self.cfg.get('geoGridWidth', 128))
                    _sy = max(1, _geo.shape[1] // _gh); _sx = max(1, _geo.shape[2] // _gw)
                    _geo = _geo[:, ::_sy, ::_sx][:, :_gh, :_gw]                     # (B,64,128)
                    _ssum = _geo.sum(axis=(1, 2), keepdims=True)
                    _uniform = 1.0 / float(_geo.shape[1] * _geo.shape[2])
                    _geo = np.where(_ssum > 0.0, _geo / np.maximum(_ssum, 1e-8), _uniform)
                    npArrayOutputList["geo_grid"] = _geo.astype('float32')
                    # Exp 281 concentration weighting is applied INSIDE GeolocationKLLoss (derived
                    # from y_true's per-sample max == concentration), NOT via a Keras sample_weight
                    # here: this model's outputs are a LIST, so a name-keyed sample_weight dict
                    # breaks Keras' resolve_path (KeyError: 0). See NNLosses.GeolocationKLLoss.
                if self.cfg.get('heatmapAddDepthNormalsUncertainty', False) and npArrayOut is not None:
                    # Slice depth+normals channels for the hm_nll NLL head.
                    # DepthNormalsNLLLoss normalises by scale internally, so we pass
                    # the raw values here (same convention as "hm" above).
                    _DN = {'depthmap', 'normalX', 'normalY', 'normalZ'}
                    _heatmaps = self.cfg.get('heatmaps', [])
                    _dn_idx = sorted(i for i, h in enumerate(_heatmaps) if h in _DN)
                    if _dn_idx:
                        npArrayOutputList["hm_nll"] = npArrayOut[..., _dn_idx[0]:_dn_idx[-1] + 1]

            log_thread_progress("dataloader_python", 0, "update_cpu_batch")
            return npArrayIn, npArrayOutputList
        else:
            #Regular just RGB -> heatmap output
            log_thread_progress("dataloader_python", 0, "update_cpu_batch")
            return npArrayIn, npArrayOut

    def num_batches(self):
        return self.numberOfSamples // self.batch_size

    def batch_size(self):
        return self.batch_size

    def on_epoch_end(self):
        self.epoch = self.epoch + 1
        if (not self.validation_data):
            print("\nTrainingDataGenerator on_epoch_end shuffling, next epoch is ", self.epoch)
            if (self.epoch < 2):
                #in the beginning everything is kind of random so do normal random shuffle
                self.db.shuffle()
            else:
                #After we have accumulated some losses try to shuffle using the losses in an attempt to make training more interesting
                #self.db.shuffle()
                self.db.shuffle_based_on_loss()  #<- TODO: This may need to be deactivated if training is unstable
                pass


#============================================================================================
# ─────────────────────────────────────────────────────────────────────────────
# PerfProfiler  — per-batch / per-epoch timing callback, dumps to perf.txt
#
# Measures:
#   • batch_ms      : wall-clock time of model.fit()'s train_step() call
#   • gap_ms        : time between on_train_batch_end and next on_train_batch_begin
#                     (= Python overhead + data pipeline time between steps)
#   • epoch_cb_ms   : time spent in all OTHER on_epoch_end callbacks combined
#                     (= overhead of checkpointing, svg_logger, tensorboard, …)
#
# After every epoch a summary is appended to perf.txt with:
#   epoch, steps, batch stats (min/p50/p95/max), gap stats, cb overhead
# ─────────────────────────────────────────────────────────────────────────────
class PerfProfiler(keras.callbacks.Callback):
    """Per-batch timing profiler.  Appends one report block to perf.txt per epoch."""

    PERF_FILE = "perf.txt"

    def __init__(self):
        super().__init__()
        self._batch_start = None
        self._batch_end_ts = None
        self._batch_times = []  # step latencies in ms
        self._gap_times = []  # inter-step gaps in ms
        self._epoch_start = None

    # ── batch hooks ──────────────────────────────────────────────────────────

    def on_train_batch_begin(self, batch, logs=None):
        now = time.perf_counter()
        if self._batch_end_ts is not None:
            self._gap_times.append((now - self._batch_end_ts) * 1000.0)
        self._batch_start = now

    def on_train_batch_end(self, batch, logs=None):
        now = time.perf_counter()
        if self._batch_start is not None:
            self._batch_times.append((now - self._batch_start) * 1000.0)
        self._batch_end_ts = now

    # ── epoch hooks ──────────────────────────────────────────────────────────

    def on_epoch_begin(self, epoch, logs=None):
        self._batch_times = []
        self._gap_times = []
        self._batch_end_ts = None
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        # Measure how long all the *other* epoch-end callbacks take (this one
        # runs first because it was appended last to the list — or we record
        # the time at which we started on_epoch_end and compare to the next
        # on_epoch_begin).  We record the elapsed *from* epoch_start *to* now
        # as "total epoch wall time" and derive callback overhead later.
        epoch_wall_ms = (time.perf_counter() - self._epoch_start) * 1000.0

        bt = self._batch_times
        gt = self._gap_times

        def _stats(vals):
            if not vals:
                return dict(n=0, min=0, p50=0, p95=0, max=0, mean=0)
            a = sorted(vals)
            n = len(a)
            return dict(
                n=n,
                min=a[0],
                p50=a[n // 2],
                p95=a[min(int(n * 0.95), n - 1)],
                max=a[-1],
                mean=sum(a) / n,
            )

        bs = _stats(bt)
        gs = _stats(gt)

        total_step_ms = sum(bt)
        total_gap_ms = sum(gt)
        # Epoch-level callbacks run after on_epoch_end of PerfProfiler, so we
        # can only bound them: epoch_wall - (step time + gaps)
        overhead_ms = epoch_wall_ms - total_step_ms - total_gap_ms

        lines = [
            "",
            "=" * 72,
            "PerfProfiler  epoch=%d   %s" % (epoch + 1, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "=" * 72,
            "Epoch wall time        : %8.1f s" % (epoch_wall_ms / 1000.0),
            "",
            "Batch (train_step) latency [ms]  — %d steps" % bs['n'],
            "  min  : %8.2f" % bs['min'],
            "  p50  : %8.2f" % bs['p50'],
            "  p95  : %8.2f" % bs['p95'],
            "  max  : %8.2f" % bs['max'],
            "  mean : %8.2f" % bs['mean'],
            "  sum  : %8.1f s" % (total_step_ms / 1000.0),
            "",
            "Inter-step gap (data/Python overhead) [ms]  — %d gaps" % gs['n'],
            "  min  : %8.2f" % gs['min'],
            "  p50  : %8.2f" % gs['p50'],
            "  p95  : %8.2f" % gs['p95'],
            "  max  : %8.2f" % gs['max'],
            "  mean : %8.2f" % gs['mean'],
            "  sum  : %8.1f s" % (total_gap_ms / 1000.0),
            "",
            "Epoch-end callbacks + misc overhead : %8.1f s" % (overhead_ms / 1000.0),
            "  (= epoch_wall - step_sum - gap_sum; includes checkpointing, svg, tensorboard)",
        ]

        # Append individual batch times for detailed offline analysis
        if bt:
            lines += [
                "",
                "Per-step latencies (ms, chronological):",
                "  " + "  ".join("%6.1f" % v for v in bt[:200]),  # cap at 200 to keep file sane
            ]

        report = "\n".join(lines) + "\n"

        try:
            with open(self.PERF_FILE, "a") as fh:
                fh.write(report)
            print("[PerfProfiler] report appended to %s" % self.PERF_FILE)
        except Exception as e:
            print("[PerfProfiler] could not write %s: %s" % (self.PERF_FILE, e))

        # Always print summary to console too
        print("[PerfProfiler] epoch=%d  step: mean=%.1fms p95=%.1fms  gap: mean=%.1fms p95=%.1fms  cb_overhead=%.1fs" %
              (epoch + 1, bs['mean'], bs['p95'], gs['mean'], gs['p95'], overhead_ms / 1000.0))


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
def custom_lr_scheduler(epoch, startLoss=0.0001, endLoss=0.000015, warmup_epochs=0):
    # warmup_epochs: when >0 (typically set to 5 for multi-GPU), linearly ramp
    # the LR from endLoss up to startLoss over the first warmup_epochs.
    # Goyal et al. 2017 (https://arxiv.org/abs/1706.02677) showed this prevents
    # divergence when using the linear scaling rule (LR × num_GPUs) because the
    # initial random weights produce high-variance gradients that a large LR
    # amplifies into NaN.  The warmup lets BatchNorm statistics and early weights
    # stabilise before the full learning rate kicks in.
    #--------------------------
    finalEpochBeforeFlatLine = 100
    decimals = 5
    maximum = startLoss
    minimum = endLoss
    #--------------------------
    if warmup_epochs > 0 and epoch < warmup_epochs:
        # Linear ramp from minimum (endLoss) up to maximum (startLoss)
        return round(minimum + (maximum - minimum) * (epoch / warmup_epochs), decimals)
    elif epoch < finalEpochBeforeFlatLine:
        return round((maximum - minimum) * ((1 - (epoch - 1) / (finalEpochBeforeFlatLine - 1))**2) + minimum, decimals)
    else:
        return round(minimum, decimals)


#============================================================================================
def custom_lr_schedulerWarmup(epoch, warmup_epochs=10, total_epochs=200, initial_lr=0.00015, target_lr=0.001,
                              minimum=0.00015):
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
def jointGradientScheduler(epoch, warmup_epochs=10, mature_epochs=200, total_epochs=250, max_joints_gradient=23,
                           min_joints_gradient=8, max_paf_gradient=6, min_paf_gradient=2):

    joint_gradient = min_joints_gradient
    paf_gradient = min_paf_gradient

    if epoch < warmup_epochs:
        #Stick gradient to max value
        #joint_gradient = max_joints_gradient
        #paf_gradient   = max_paf_gradient

        # Linearly decay to half of max values during warmup
        t = epoch / warmup_epochs  # normalized [0, 1)
        joint_gradient = max_joints_gradient - t * (max_joints_gradient / 2)
        paf_gradient = max_paf_gradient - t * (max_paf_gradient / 2)

    elif epoch < mature_epochs:

        #Oscillate with lower magnitude (2/6/25)
        max_joints_gradient = int(max_joints_gradient / 2)
        max_paf_gradient = int(max_paf_gradient / 2)

        # Number of steps per area (area = one direction: up or down)
        steps = max(max_joints_gradient - min_joints_gradient, max_paf_gradient - min_paf_gradient)

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
            paf_gradient = max_paf_gradient - (max_paf_gradient - min_paf_gradient) * t
        else:  # ascending
            joint_gradient = min_joints_gradient + (max_joints_gradient - min_joints_gradient) * t
            paf_gradient = min_paf_gradient + (max_paf_gradient - min_paf_gradient) * t
    else:
        joint_gradient = min_joints_gradient
        paf_gradient = min_paf_gradient

    return int(joint_gradient), int(paf_gradient)


#============================================================================================
def jointGradientSchedulerSimple(epoch, warmup_epochs=10, max_joints_gradient=23, min_joints_gradient=8,
                                 max_paf_gradient=6, min_paf_gradient=2):
    joint_gradient = min_joints_gradient
    paf_gradient = min_paf_gradient
    if (min_joints_gradient != max_joints_gradient) or (min_paf_gradient != max_paf_gradient):
        if epoch < warmup_epochs:
            # Linearly decay to target during warmup
            t = epoch / warmup_epochs  # normalized [0, 1)
            joint_gradient = max_joints_gradient - t * (max_joints_gradient - min_joints_gradient)
            paf_gradient = max_paf_gradient - t * (max_paf_gradient - min_paf_gradient)
    return int(joint_gradient), int(paf_gradient)


#============================================================================================
class DataAugmentation(keras.callbacks.Callback):

    def __init__(self, cfg, db=None):
        super().__init__()
        self.cfg = cfg
        self.epochLimit = cfg["heatmapReductionEpochLimit"]
        self.db = db
        self.epoch = 1
        self.time = time.time()
        self.runningLoss = 0.0  # logs['loss'] of the previous batch of this epoch

    def clear_gpu_memory(self, tensors):
        # Clear GPU memory for a list of tensors
        for tensor in tensors:
            del tensor

    #https://keras.io/guides/writing_your_own_callbacks/
    def on_train_batch_end(self, batch, logs=None):
        if (self.db) and logs and ("loss" in logs):
            #self.db.printReadSpeed()
            # Keras 3 logs['loss'] is the RUNNING MEAN of the epoch so far, not this batch's loss.
            # Batches are all full (len = samples // batchSize), so the mean is unweighted and this
            # batch's own loss is (b+1)*R_b - b*R_(b-1).
            running = float(logs['loss'])
            if batch == 0:
                self.runningLoss = 0.0
            batchLoss = (batch + 1) * running - batch * self.runningLoss
            self.runningLoss = running
            # The batch's samples from its index: the generator reads [index*B, index*B+B) in order,
            # while db.lastStart/EndSample follow whatever the prefetch queue fetched last.
            start = batch * self.cfg['batchSize']
            end = min(start + self.cfg['batchSize'], self.db.numberOfSamples)
            self.db.updateEpochResults(batchLoss, start, end, self.epoch)

    def on_epoch_start(self, epoch, logs=None):
        self.time = time.time()  #Is this not executed ?

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        # trainYMAPNet.py's --start N handling calls this manually (no logs) to pre-seed
        # augmentation state when resuming mid-run; Keras's own calls always pass real logs.
        keys = list(logs.keys()) if logs else []
        #print("End epoch ",epoch+1," of training")
        #print("Got log keys:", keys))

        totalSeconds = time.time() - self.time
        cpuTimeSeconds = self.db.cpuTimeSeconds
        gpuTimeSeconds = totalSeconds - cpuTimeSeconds
        print("Time it took for epoch %u | CPU : %0.02f sec | GPU : %0.2f sec | Total : %0.02f sec" %
              (epoch + 1, cpuTimeSeconds, gpuTimeSeconds, totalSeconds))
        self.db.printReadSpeed()
        self.db.cpuTimeSeconds = 0  #Reset counter
        self.time = time.time()  #Reset GPU time (although on_epoch_start should also reset it)

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

            oldGS = self.db.gradientSize
            oldPS = self.db.PAFSize
            """
            #Go up and down
            self.db.gradientSize,self.db.PAFSize = jointGradientScheduler(epoch,
                                                                          warmup_epochs=self.cfg.get("heatmapGradientWarmupEpochs", self.cfg["earlyStoppingStart"]),
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
            self.db.gradientSize, self.db.PAFSize = jointGradientSchedulerSimple(
                epoch, warmup_epochs=self.cfg.get("heatmapGradientWarmupEpochs", self.cfg["earlyStoppingStart"]),
                max_joints_gradient=self.cfg["heatmapGradientSize"],
                min_joints_gradient=self.cfg["heatmapGradientSizeMinimum"], max_paf_gradient=self.cfg["heatmapPAFSize"],
                min_paf_gradient=self.cfg["heatmapPAFSizeMinimum"])

            if ((oldGS != self.db.gradientSize) or (oldPS != self.db.PAFSize)):
                print(bcolors.OKGREEN, "Set gradient size from ", oldGS, " to ", self.db.gradientSize, bcolors.ENDC,
                      end=" / ")
                print(bcolors.OKGREEN, "Set PAF size from ", oldPS, " to ", self.db.PAFSize, bcolors.ENDC)


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
        jG, pG = jointGradientScheduler(epoch, warmup_epochs=10, mature_epochs=200, total_epochs=250,
                                        max_joints_gradient=23, min_joints_gradient=8, max_paf_gradient=6,
                                        min_paf_gradient=2)
        print("Epoch ", epoch, " Joints:", jG, " PAF:", pG)


#-------------------------------------------------------------------------------
def assertDescriptorsArePopulated(db, label, cfg):
    """Refuse to train when outputDescriptors is on but the loader hands back zeros.

    A source with no descriptor sidecar is not an error in the C loader: it prints
    "Could not find a DinoV3 Descriptor file" and every sample of that source then
    yields an all-zero vector. With outputDescriptors=True those zeros become
    regression targets, so the descriptor head is trained to predict 0 for part of
    the corpus while val (which may have its sidecar) scores against real vectors.
    That reads as a mysterious quality loss rather than missing data, so fail fast.

    Sweeps EVERY sample, not a sample of them: descriptors are already resident in
    RAM, so a full pass costs ~0.4s for 242k samples (measured) while a windowed
    probe can miss a small source entirely (openposeFactory2 is 0.11% of the corpus).

    Note the sidecar name the loader probes is "<db>.dinov3"; older dumps named
    "<db>.dinov2" are NOT picked up and present exactly as this all-zero case.
    Tolerance is configurable via descriptorZeroTolerance (default 0.0 = strict).
    """
    dim = db.get_descriptor_number_of_elements()
    if dim <= 0:
        print(bcolors.FAIL, "outputDescriptors=True but %s exposes descriptor dim=%d." % (label, dim),
              "\nRebuild libDataLoader.so with -DUSE_DINOV2_FEATURES and provide <db>.dinov3 sidecars.",
              bcolors.ENDC)
        sys.exit(1)

    total = int(db.numberOfSamples)
    if total <= 0:
        return

    started = time.time()
    chunk = int(cfg.get('descriptorScanChunk', 4096))
    zero = 0
    perSource = dict()
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        magnitude = np.abs(db.get_partial_descriptor_array(start, end)).sum(axis=1)
        for j in np.nonzero(magnitude == 0.0)[0]:
            zero += 1
            try:
                where = str(db.get_filename_of_sample(start + int(j)))
                where = where[:where.rfind("/")] if "/" in where else where
            except Exception:
                where = "(unknown source)"
            perSource[where] = perSource.get(where, 0) + 1

    fraction = zero / float(total)
    tolerance = float(cfg.get('descriptorZeroTolerance', 0.0))
    print("Descriptor check on %s: %d/%d samples all-zero (%.3f%%), dim=%d, swept in %.1fs" %
          (label, zero, total, 100.0 * fraction, dim, time.time() - started))
    if fraction > tolerance:
        print(bcolors.FAIL,
              "\nREFUSING TO TRAIN: outputDescriptors=True but %d of %d %s samples (%.3f%%)" % (zero, total, label, 100.0 * fraction),
              "\ncarry all-zero DINO vectors (tolerance descriptorZeroTolerance=%g)." % tolerance,
              "\nThose samples would train the descriptor head to regress zero.", bcolors.ENDC)
        for src in sorted(perSource, key=perSource.get, reverse=True):
            print(bcolors.FAIL, "   %7d samples  %s" % (perSource[src], src), bcolors.ENDC)
        print(bcolors.FAIL,
              "\nFix: generate <db>.dinov3 for those sources (older .dinov2 dumps are NOT read),",
              "\ndisable them in the dataset list, set outputDescriptors=false,",
              "\nor raise descriptorZeroTolerance deliberately.", bcolors.ENDC)
        sys.exit(1)


def resolveModelDir():
    """The model directory used to be 2d_pose_estimation/ everywhere; some machines have
    since been renamed to ymapnet_model/ (the new convention -- 2d_pose_estimation/ is
    legacy, see PLAN.md). Detect which one actually exists on THIS machine rather than
    hardcoding either name, so the same configuration.json/code works whether or not a
    given checkout has been renamed yet."""
    if os.path.isdir('ymapnet_model'):
        return 'ymapnet_model'
    return '2d_pose_estimation'


def rebaseModelPath(path, modelDir):
    """cfg values like embeddingsPath/synonymPath store a literal '2d_pose_estimation/...'
    (or 'ymapnet_model/...') prefix baked in when the config was written. Rewrite that
    prefix to whichever directory actually exists on this machine, so the same
    configuration.json works whether this checkout has been renamed or not."""
    for oldPrefix in ('2d_pose_estimation/', 'ymapnet_model/'):
        if path.startswith(oldPrefix):
            return modelDir + '/' + path[len(oldPrefix):]
    return path
