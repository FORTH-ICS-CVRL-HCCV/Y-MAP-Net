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
import json
import time
import gc
import math
import numpy as np
import datetime
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
from ymapnet.core.NNLosses import RSquaredMetric, HeatmapDistanceMetric, heatmap_distance_loss, weighted_mse_loss, vanilla_mse_loss, vanilla_mse16bit_loss, token_mse_loss, combined_loss, combined_two_loss, dssim_loss
from ymapnet.utils.tools import bcolors, read_json_file, checkIfPathExists, checkIfFileExists, convert_bytes


#-------------------------------------------------------------------------------
def deriveRGBChannelsFromCFG(cfg):
    numberOfChannels = 3
    if (cfg['RGBImageEncoding'] == 'rgb8'):
        numberOfChannels = 1
    if (cfg['RGBImageEncoding'] == 'rgb16'):
        numberOfChannels = 2
    return numberOfChannels


#-------------------------------------------------------------------------------
def deriveHeatmapChannelsFromCFG(cfg):
    extraHeatmaps = len(cfg['keypoint_names'])  #names include bkg + 1 depthmap
    if (cfg['heatmapAddDepthmap']):
        extraHeatmaps = extraHeatmaps + 1
    return extraHeatmaps


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
def logTrainingParameters(cfg, log_dir):
    try:
        # Create a summary writer
        param_log = tf.summary.create_file_writer(log_dir)

        with param_log.as_default():
            # Convert the dictionary to a formatted string
            params_str = "\n".join([f"{key}: {value}" for key, value in cfg.items()])
            # Log the parameters as text
            tf.summary.text("Training Parameters", params_str, step=0)
    except Exception as e:
        print(f"Error storing logging parameters in tensorboard: {e}")


#-------------------------------------------------------------------------------
def calculate_stats_to_csv(data, filename="embedding_stats.csv"):
    import csv
    # Open a CSV file for writing
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(["Key", "Min", "Max", "Mean"])

        # Process each key and its values
        for key, values in data.items():
            np_values = np.array(values)
            min_val = np.min(np_values)
            max_val = np.max(np_values)
            mean_val = np.mean(np_values)

            # Write the stats to the CSV file
            writer.writerow([key, min_val, max_val, mean_val])


#-------------------------------------------------------------------------------
def calculate_stats(data):
    result = {}
    for key, values in data.items():
        np_values = np.array(values)
        result[key] = {'min': np.min(np_values), 'max': np.max(np_values), 'mean': np.mean(np_values)}
    return result


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
    print("")
    #-----------------------------


#-------------------------------------------------------------------------------
def positional_encoding(seq_len, model_dim):
    pos = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, model_dim, 2) * -(np.log(10000.0) / model_dim))
    pos_encoding = np.zeros((seq_len, model_dim))
    pos_encoding[:, 0::2] = np.sin(pos * div_term)
    pos_encoding[:, 1::2] = np.cos(pos * div_term)
    return pos_encoding


#-------------------------------------------------------------------------------
def build_resnet_cnn_glove(input_shape, num_classes, maxtokens=16, D=300, layer_width=1024):
    # Load ResNet50 without the top layers (include_top=False)
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'  # Global Average Pooling
    )

    # Freeze the ResNet50 base model
    base_model.trainable = False

    # Create a new model
    inputs = layers.Input(shape=input_shape)

    # Apply the ResNet base model
    x = base_model(inputs)

    # Reshape only if the input is not already flattened
    if len(x.shape) > 2:  # Check if the output needs reshaping
        x = layers.Reshape((2048, ))(x)

    #pos_encoding = positional_encoding(maxtokens, 512)
    #x_with_pos = x + pos_encoding

    # Initialize a variable to hold the previous output for residual connections
    prev_output = None

    # Define multiple outputs, one for each embedding
    glove_outputs = []
    #for i in reversed(range(maxtokens)):
    for i in range(maxtokens):
        #scaling_factor = layers.Dense(1, activation='sigmoid')(x)  # Learnable scaling factor

        # Sequential dense layers for each embedding
        x_token = layers.Dense(layer_width, activation='leaky_relu')(x)
        x_token = layers.LayerNormalization()(x_token)
        x_token = layers.Dropout(0.2 /
                                 (i + 1))(x_token)  #Staggered dropout that gets lower as we progress to next tokens

        x_token = layers.Dense(layer_width, activation='leaky_relu')(x_token)
        x_token = layers.LayerNormalization()(x_token)
        x_token = layers.Dropout(0.2 /
                                 (i + 1))(x_token)  #Staggered dropout that gets lower as we progress to next tokens

        x_token = layers.Dense(layer_width, activation='leaky_relu')(x_token)
        x_token = layers.LayerNormalization()(x_token)
        x_token = layers.Dropout(0.2 /
                                 (i + 1))(x_token)  #Staggered dropout that gets lower as we progress to next tokens
        connect_to_next = x_token

        # Residual connection: add the previous token's output to the current one
        x_token_residual = x_token
        #Residual connection makes all tokens repeat the same thing
        if prev_output is not None:
            prev_output = keras.layers.Rescaling(0.3,
                                                 offset=0.0)(prev_output)  #<- Scale down importance of previous value

            x_token_residual = layers.Dense(layer_width, activation='leaky_relu')(prev_output)
            x_token_residual = layers.LayerNormalization()(x_token)
            x_token_residual = layers.Dropout(0.3)(x_token)  #<- make it very noisy ?
            x_token_residual = layers.Add()([x_token, x_token_residual])

            x_token_residual = layers.Dense(layer_width, activation='leaky_relu')(x_token_residual)
            x_token_residual = layers.LayerNormalization()(x_token_residual)

            x_token_residual = layers.Dense(layer_width, activation='leaky_relu')(x_token_residual)
            x_token_residual = layers.LayerNormalization()(x_token_residual)

            x_token_residual = layers.Dense(layer_width, activation='leaky_relu')(x_token_residual)
            x_token_residual = layers.LayerNormalization()(x_token_residual)

        x_token = layers.Dense(D, activation='leaky_relu')(x_token_residual)
        x_token = layers.LayerNormalization()(x_token)

        #This should be a tanh activation but in order for the appended network to have tanh try linear
        glove_output = layers.Dense(D, activation='tanh', name="t%02u" % i)(x_token)
        #glove_output = layers.Multiply()([glove_output, scaling_factor])
        glove_outputs.append(glove_output)

        # Store the current token's output to be used as residual in the next iteration
        prev_output = connect_to_next  #glove_output or connect_to_next

    # Build the model with multiple outputs
    model = models.Model(inputs, glove_outputs)

    # Print model summary
    model.summary()

    return model


#============================================================================================
def build_resnet_gru_glove(input_shape, num_classes, maxtokens=16, D=50):
    # Load ResNet50 without the top layers (include_top=False)
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'  # Global Average Pooling
    )

    # Freeze the ResNet50 base model
    base_model.trainable = False

    # Define input layer
    inputs = layers.Input(shape=input_shape)

    # Apply the ResNet base model
    x = base_model(inputs)
    x = layers.Reshape((2048, ))(x)  # Flatten if not already flat

    # Additional dense layers to increase feature capacity
    x = layers.Dense(4096, activation='leaky_relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(2048, activation='leaky_relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Repeat the dense output for each token in the sequence
    x = layers.RepeatVector(maxtokens)(x)

    # GRU for sequential processing
    x = layers.GRU(1024, activation='tanh', return_sequences=True)(x)  # Main GRU layer for sequence handling
    x = layers.LayerNormalization()(x)

    # Time-distributed dense layers to output D-dimensional GloVe embeddings per token
    x = layers.TimeDistributed(layers.Dense(512, activation='leaky_relu'))(x)
    x = layers.LayerNormalization()(x)
    x = layers.TimeDistributed(layers.Dropout(0.2))(x)

    # Final dense layer to produce D-dimensional embeddings for each token
    x = layers.TimeDistributed(layers.Dense(D, activation='linear'))(x)

    # Use tf.unstack to create individual token outputs
    # Use custom UnstackLayer to create individual token outputs
    # Replace UnstackLayer call with a for-loop to set output names explicitly
    glove_outputs = []
    for i in range(maxtokens):
        output = layers.Lambda(lambda x: x[:, i], name=f"t{i:02}")(x)  # Name each output as t00, t01, etc.
        glove_outputs.append(output)

    # Build the model with multiple token outputs
    model = models.Model(inputs, glove_outputs)
    model.summary()

    return model


#============================================================================================
def append_final_tanh_glove_layer(model, maxtokens=16, D=300):
    # Concatenate all token outputs into a single vector
    concatenated_output = layers.Concatenate()(model.outputs)
    concatenated_output = layers.Reshape((D * maxtokens, ))(concatenated_output)

    x_token = concatenated_output
    #x_token = layers.Dense(2048, activation='leaky_relu')(concatenated_output)
    #x_token = layers.BatchNormalization()(x_token)
    #x_token = layers.Dropout(0.1)(x_token)

    # Add a single Dense layer of size D * maxtokens as the final output
    final_output = layers.Dense(D * maxtokens, activation='tanh', name="final_output")(x_token)

    # Create a new model with the same inputs and the new single output
    new_model = models.Model(inputs=model.inputs, outputs=final_output)

    # Copy weights from the original model to the new model (up to the old output layers)
    for layer in model.layers:
        if layer.name in new_model.layers:
            new_model.get_layer(layer.name).set_weights(layer.get_weights())

    # Print summary of the new model
    new_model.summary()

    return new_model


#============================================================================================
def append_final_onehot_layer(model, maxtokens=16, D=300, TokensOut=16, Classes=2037):
    # Concatenate all token outputs into a single vector
    concatenated_output = layers.Concatenate()(model.outputs)
    concatenated_output = layers.Reshape((D * maxtokens, ))(concatenated_output)

    x_token = layers.Dense(2048, activation='leaky_relu')(concatenated_output)
    x_token = layers.LayerNormalization()(x_token)
    #x_token = layers.BatchNormalization()(x_token)
    #x_token = layers.Dropout(0.1)(x_token)

    x_token = layers.Dense(2048, activation='leaky_relu')(x_token)
    x_token = layers.LayerNormalization()(x_token)
    #x_token = layers.BatchNormalization()(x_token)
    #x_token = layers.Dropout(0.1)(x_token)

    #x_token = layers.Dense(2048, activation='leaky_relu')(x_token)
    #x_token = layers.LayerNormalization()(x_token)
    #x_token = layers.BatchNormalization()(x_token)
    #x_token = layers.Dropout(0.1)(x_token)

    # Add a single Dense layer of size D * maxtokens as the final output
    final_output = layers.Dense(Classes * TokensOut, activation='sigmoid', name="final_output")(x_token)

    # Create a new model with the same inputs and the new single output
    new_model = models.Model(inputs=model.inputs, outputs=final_output)

    # Copy weights from the original model to the new model (up to the old output layers)
    for layer in model.layers:
        if layer.name in new_model.layers:
            new_model.get_layer(layer.name).set_weights(layer.get_weights())

    # Print summary of the new model
    new_model.summary()

    return new_model


#============================================================================================
#============================================================================================
#Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly <- Old
#https://www.tensorflow.org/api_docs/python/tf/keras/utils/PyDataset
class TrainingDataGenerator(keras.utils.PyDataset):

    def __init__(self, cfg, db, batch_size=32, numberOfTokens=16, D=50, combineData=False, returnOneHot=False,
                 validation_data=False, class_weights=None, labels=None, log_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.db = db
        self.batch_size = batch_size
        self.numberOfSamples = self.db.numberOfSamples
        self.epoch = 1
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.labels = labels

        self.combineData = combineData
        self.returnOneHot = returnOneHot

        self.numberOfTokens = numberOfTokens
        self.class_weights = class_weights
        self.D = D

        self.db.shuffle()

    def __len__(self):
        return self.numberOfSamples // self.batch_size

    def __getitem__(self, index):
        self.start_index = index * self.batch_size
        self.end_index = min(self.start_index + self.batch_size, self.numberOfSamples)

        npArrayIn, npArrayOut, npArrayOut16Bit = self.db.get_partial_update_IO_array(self.start_index, self.end_index)

        if ('outputTokens' in self.cfg) and (self.cfg['outputTokens']):
            if (self.returnOneHot):
                #Grab one-hot encodings
                if (self.combineData):
                    npArrayTokensOut = self.db.get_partial_token_array(self.start_index, self.end_index,
                                                                       encodeAsSingleMultiLabelToken=True)
                    #npArrayTokensOut = npArrayTokensOut.astype('float32')
                else:
                    npArrayTokensOut = self.db.get_partial_token_array(self.start_index, self.end_index,
                                                                       encodeAsSingleMultiLabelToken=False)
                    npArrayTokensOut = npArrayTokensOut.reshape(self.batch_size, self.numberOfTokens * 2037)

                #print("Token output : ",npArrayTokensOut.shape)
                #print("Ones : ",np.sum(npArrayTokensOut))
                #np.set_printoptions(threshold=sys.maxsize)
                #for i in range(16):
                #  print("Token ",i," : ",npArrayTokensOut[i]," -> ones -> ",np.sum(npArrayTokensOut[i]))

                return npArrayIn, npArrayTokensOut
            else:
                #Immediately grab GloVe embeddings
                npArrayEmbeddingsOut = self.db.get_partial_embedding_array(self.start_index, self.end_index)
                npArrayEmbeddingsOut = npArrayEmbeddingsOut.astype('float32')

                #for batch in range(self.batch_size):
                #  print("Sample ",batch," -> ",end=" ")
                #  for tok in range(15):
                #   diff = np.abs(npArrayEmbeddingsOut[batch,tok,:]) - np.abs(npArrayEmbeddingsOut[batch,tok+1,:])
                #   print("TDist ",tok,"->",tok+1," = %0.5f" % np.sum(np.abs(diff)))
                #   np.savetxt("embeddings_%u.csv" % batch, npArrayEmbeddingsOut[batch,:,:], delimiter=",")

                if (self.combineData):
                    return npArrayIn, npArrayEmbeddingsOut.reshape(self.batch_size, self.D * self.numberOfTokens)
                else:
                    npArrayEmbeddingsList = dict()
                    for i in range(self.numberOfTokens):
                        npArrayEmbeddingsList["t%02u" % i] = npArrayEmbeddingsOut[:, i, :]

                    return npArrayIn, npArrayEmbeddingsList

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
                self.db.shuffle()
                #After we have accumulated some losses try to shuffle using the losses in an attempt to make training more interesting
                #self.db.shuffle_based_on_loss() #<- TODO: This may need to be deactivated if training is unstable
                pass


#============================================================================================
def custom_lr_scheduler(epoch, startLoss=0.0001, endLoss=0.000015):
    #--------------------------
    finalEpochBeforeFlatLine = 100
    decimals = 5
    maximum = startLoss
    minimum = endLoss
    #--------------------------
    if epoch == 1:
        return round(maximum, decimals)
    elif epoch < finalEpochBeforeFlatLine:
        return round((maximum - minimum) * ((1 - (epoch - 1) / (finalEpochBeforeFlatLine - 1))**2) + minimum, decimals)
    else:
        return round(minimum, decimals)


#============================================================================================
class DataAugmentation(keras.callbacks.Callback):

    def __init__(self, cfg, db=None):
        super().__init__()
        self.cfg = cfg
        self.db = db
        self.epoch = 1
        self.time = time.time()

    def clear_gpu_memory(self, tensors):
        # Clear GPU memory for a list of tensors
        for tensor in tensors:
            del tensor

    #https://keras.io/guides/writing_your_own_callbacks/
    def on_train_batch_end(self, batch, logs=None):
        self.db.updateEpochResults(logs['loss'], self.db.lastStartSample, self.db.lastEndSample, self.epoch)

    def on_epoch_start(self, epoch, logs=None):
        self.time = time.time()  #Is this not executed ?

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        keys = list(logs.keys())
        print("End epoch {} of training".format((epoch + 1)))
        #print("End epoch {} of training; got log keys: {}".format((epoch+1), keys))

        totalSeconds = time.time() - self.time
        cpuTimeSeconds = self.db.cpuTimeSeconds
        gpuTimeSeconds = totalSeconds - cpuTimeSeconds
        print("Time it took for epoch | CPU : %0.02f sec | GPU : %0.2f sec | Total : %0.02f sec" %
              (cpuTimeSeconds, gpuTimeSeconds, totalSeconds))
        self.db.cpuTimeSeconds = 0  #Reset counter
        self.time = time.time()  #Reset GPU time (although on_epoch_start should also reset it)

        #print("Doing garbage collection ..",end="")
        gc.collect()
        if (self.db):
            #logs just logs the overall loss, so it is useless..
            #if ("loss" in keys) and ("learning_rate" in keys):
            #    self.db.updateEpochResults(logs["loss"],epoch+1,logs["learning_rate"])
            if (epoch % 20 == 19):
                print(bcolors.OKGREEN, "Set gradient size from ", self.db.gradientSize, end=" ")
                self.db.gradientSize = max(self.cfg['heatmapGradientSizeMinimum'], self.db.gradientSize - 1)
                print(" to ", self.db.gradientSize, bcolors.ENDC)

                print(bcolors.OKGREEN, "Set PAF size from ", self.db.PAFSize, end=" ")
                self.db.PAFSize = max(self.cfg['heatmapPAFSizeMinimum'], self.db.PAFSize - 1)
                print(" to ", self.db.PAFSize, bcolors.ENDC)
        #if (self.db):
        #     self.db.refresh_all_frame_augmentations(epoch)


#============================================================================================
class OneHotLoss(keras.losses.Loss):

    def __init__(self, class_weights, weight=1.0, **kwargs):
        super(OneHotLoss, self).__init__(**kwargs)
        self.class_weights = tf.constant(class_weights, dtype=keras.backend.floatx())[None, :]
        self.weight = weight

    def call(self, y_true, y_pred):
        # Ensure both y_true and y_pred are cast to float32
        float_type = keras.backend.floatx()
        y_true_onehot = tf.cast(y_true, float_type)
        y_pred_onehot = tf.cast(y_pred, float_type)
        """
        # Check shapes for debugging
        tf.debugging.assert_shapes([(y_true_onehot, ('batch_size', 2037)), (y_pred_onehot, ('batch_size', 2037)), (self.class_weights, (1, 2037)),], message="Shape mismatch detected in OneHotLoss.")

        y_true_onehot_weighted = tf.multiply(y_true_onehot, self.class_weights)
        mean_val = tf.reduce_mean(y_true_onehot_weighted)
        range_val = tf.reduce_max(y_true_onehot_weighted) - tf.reduce_min(y_true_onehot_weighted)
        y_true_onehot_weighted = (y_true_onehot_weighted - mean_val) / range_val

        y_pred_onehot_weighted = tf.multiply(y_pred_onehot, self.class_weights)
        mean_val = tf.reduce_mean(y_pred_onehot_weighted)
        range_val = tf.reduce_max(y_pred_onehot_weighted) - tf.reduce_min(y_pred_onehot_weighted)
        y_pred_onehot_weighted = (y_pred_onehot_weighted - mean_val) / range_val

        # Compute binary cross-entropy for each class individually
        #ce = tf.keras.losses.categorical_crossentropy(y_true_onehot_weighted, y_pred_onehot_weighted, from_logits=False)
        ce = keras.losses.binary_crossentropy(y_true_onehot_weighted, y_pred_onehot_weighted, from_logits=False)

        # Apply class weights
        scaled_weighted_loss = self.weight * ce#tf.reduce_sum(ce, axis=-1)

        # Scale by the provided weight
        #scaled_weighted_loss = self.weight *  tf.exp(ce) #Try perplexity loss e^loss 
        """

        token_loss_function = keras.losses.BinaryCrossentropy(from_logits=False)
        #token_loss_function = keras.losses.MeanSquaredError()

        # Compute the original loss
        original_loss = token_loss_function(y_true_onehot, y_pred_onehot)

        # Multiply by the weight
        #weighted_loss = 1.0 *  tf.exp(original_loss) #Try perplexity loss e^loss
        scaled_weighted_loss = self.weight * original_loss

        return scaled_weighted_loss


#============================================================================================
class WeightedBinaryCrossEntropy(keras.losses.Loss):

    def __init__(self, class_weights, weight=1.0, **kwargs):
        super(WeightedBinaryCrossEntropy, self).__init__(**kwargs)
        # Store the class weights
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.weight = 1.0

    def call(self, y_true, y_pred):
        # Ensure both y_true and y_pred are cast to float32
        float_type = keras.backend.floatx()
        y_true = tf.cast(y_true, float_type)
        y_pred = tf.cast(y_pred, float_type)

        # Sigmoid activation for multi-label probabilities
        y_pred = tf.nn.sigmoid(y_pred)

        # Calculate positive and negative weights based on class imbalance
        pos_weight = y_true * self.class_weights
        neg_weight = (1 - y_true) * self.class_weights

        # Compute weighted cross-entropy loss
        loss = -pos_weight * tf.math.log(y_pred + 1e-7) - neg_weight * tf.math.log(1 - y_pred + 1e-7)
        return self.weight * tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


#============================================================================================
class GloVeMSELoss(keras.losses.Loss):

    def __init__(self, weight=1.0, **kwargs):
        super(GloVeMSELoss, self).__init__(**kwargs)
        self.weight = weight

    def call(self, y_true, y_pred):
        # Ensure both y_true and y_pred are cast to float32
        float_type = keras.backend.floatx()

        # Extract the embedding vectors (elements 1 to 51)
        y_true_glove = tf.cast(y_true, float_type)  #If upper is uncommented set to 1:
        y_pred_glove = tf.cast(y_pred, float_type)  #If upper is uncommented set to 1:

        #tf.debugging.check_numerics(y_true_glove, "NaN or Inf in y_true_glove")
        #tf.debugging.check_numerics(y_pred_glove, "NaN or Inf in y_pred_glove")

        # Compute the weighted MSE, tf.abs for always postiive values ?
        mse_glove = tf.reduce_mean(tf.square(y_true_glove - y_pred_glove), axis=-1)

        # Apply the weight to the loss
        total_loss = mse_glove * self.weight

        return total_loss


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
    shuffleData = True
    dbTrain = None
    dbValidation = None

    # Training reads the root (git-tracked) tokens.json as its source of truth.
    # After a successful run it is copied to 2d_pose_estimation/tokens.json (the packaged
    # runtime config) alongside the saved 2d_pose_estimation/tokens model.
    jsonPath = 'tokens.json'
    if (checkIfFileExists(jsonPath)):
        print(bcolors.OKGREEN, "Loading configuration from file ", jsonPath, bcolors.ENDC)
        from ymapnet.utils.createJSONConfiguration import loadJSONConfiguration
        cfg = loadJSONConfiguration(jsonPath)
    else:
        print(bcolors.FAIL, "CREATING FRESH CONFIGURATION!", bcolors.ENDC)
        from ymapnet.utils.createJSONConfiguration import createJSONConfiguration
        cfg = createJSONConfiguration(jsonPath)

    if (cfg['mixedPrecision']):
        print(bcolors.WARNING, "Using mixed precision mode!", bcolors.ENDC)
        keras.mixed_precision.set_global_policy("mixed_float16")

    D = 300
    countResponses = False
    saveRestoredWeights = False
    restoreBestWeights = False
    resumePreviousTraining = False
    if (len(sys.argv) > 1):
        #print('Argument List:', str(sys.argv))
        for i in range(0, len(sys.argv)):
            if (sys.argv[i] == "--flush"):
                os.system("rm -rf 2d_pose_estimation/ && mkdir 2d_pose_estimation/")
            if (sys.argv[i] == "--novalidation"):
                cfg['doValidation'] = False
            if (sys.argv[i] == "--mem"):
                cfg['datasetUsage'] = float(sys.argv[i + 1])
            #if (sys.argv[i]=="--stream"): Model so large that we always need to stream now
            #   cfg['streamDataset']      = True
            #   cfg['streamBufferLength'] = 1000 #int(sys.argv[i+1])
            if (sys.argv[i] == "--clear") or (sys.argv[i] == "--clean"):
                os.system("rm -rf 2d_pose_estimation/tensorboard")
                os.system("rm 2d_pose_estimation.zip")
            if (sys.argv[i] == "--resume") or (sys.argv[i] == "--continue"):
                resumePreviousTraining = True
            if (sys.argv[i] == "--count"):
                countResponses = True
            if (sys.argv[i] == "--restoreBestWeights") or (sys.argv[i] == "--restore"):
                restoreBestWeights = True
                resumePreviousTraining = False
                saveRestoredWeights = False
            if (sys.argv[i] == "--saveRestoredWeights") or (sys.argv[i] == "--save"):
                restoreBestWeights = True
                resumePreviousTraining = False
                saveRestoredWeights = True
    #--------------------------------------------------------------------------------------------------------------------------------
    #if ("outputTokens" in cfg) and (cfg["outputTokens"]):
    #           print(bcolors.FAIL,"Disabling validation until it is fixed..",bcolors.ENDC)
    #           cfg['doValidation']=False

    if (cfg['loss'] == "combine"):
        print(bcolors.FAIL, "Disabling validation when using combined loss..", bcolors.ENDC)
        cfg['doValidation'] = False

    if (cfg['outputChannels'] > cfg['baseChannels']):
        print(bcolors.FAIL, "Base channels should at least be the same as output channels..", bcolors.ENDC)
        print(bcolors.FAIL, "go to ./tokens.json and make \"baseChannels\": ",
              cfg['outputChannels'], bcolors.ENDC)
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
    if (numberOfGPUs > 0):  #and comment in this line for single GPU usage..
        #First of all create the Neural Network model
        #Don't load other data in vain if this step fails due to bad configuration..
        #----------------------------------------------------------------------------------------
        if (resumePreviousTraining):
            print(bcolors.OKGREEN, "Continuing training from pretrained network.. ", bcolors.ENDC)
            model_path = "2d_pose_estimation/model.keras"
            from ymapnet.core.NNModel import load_keypoints_model
            model, input_size, output_size, numHeatmaps = load_keypoints_model(model_path)
            cfg['inputWidth'] = input_size[0]
            cfg['inputHeight'] = input_size[1]
            cfg['outputWidth'] = output_size[0]
            cfg['outputHeight'] = output_size[1]
        else:
            print(bcolors.OKGREEN, "Creating a new Neural SIMPLE Network.. ", bcolors.ENDC)
            #from NNModel import build_simple_cnn
            #model = build_simple_cnn(( cfg['inputWidth'], cfg['inputHeight'], 3), 2037 )
            #from NNModel import build_vit
            #model = build_vit(( cfg['inputWidth'], cfg['inputHeight'], 3), 2037 )
            model = build_resnet_cnn_glove((cfg['inputWidth'], cfg['inputHeight'], 3), 2037, maxtokens=cfg["tokensOut"],
                                           D=D)
            #model = build_resnet_gru_glove(( cfg['inputWidth'], cfg['inputHeight'], 3), 2037, maxtokens=cfg["tokensOut"], D=D )

        #----------------------------------------------------------------------------------------

        # Set up TensorBoard logging
        #----------------------------------------------------------------------------------------
        log_dir = "2d_pose_estimation/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        if (cfg['streamDataset']):
            mem = 1.0  #When streaming use everything..

        #Before starting training log TF Versions
        printTFVersion()

        #Validation data is not so big and is loaded first in memory..
        onlyTrainingData = True

        if cfg['doValidation']:  #(checkIfFileExists(cfg['COCOValidationJSONPath']) and
            onlyTrainingData = False
            dbValidation = DataLoader(
                (cfg['inputHeight'], cfg['inputWidth'], 3),
                (cfg['outputHeight'], cfg['outputWidth'], cfg['outputChannels']),
                output16BitChannels=cfg['output16BitChannels'],
                numberOfThreads=cfg['DatasetLoaderThreads'],
                streamData=int(cfg['streamValidation']),  #0, <- set to 0 to keep it in memory and speed up validation 
                batchSize=cfg['batchSize'],
                gradientSize=cfg['heatmapGradientSizeMinimum'],  # <- Use final size! cfg['heatmapGradientSize'],
                PAFSize=cfg['heatmapPAFSizeMinimum'],  # <- Use final size! cfg['heatmapPAFSizeMinimum'],
                doAugmentations=0,  # <- Don't do augmentations on test set keep it clean
                addPAFs=int(cfg['heatmapAddPAFs']),
                addBackground=int(cfg['heatmapGenerateSkeletonBkg']),
                addDepthMap=int(cfg['heatmapAddDepthmap']),
                addNormals=int(cfg['heatmapAddNormals']),
                addSegmentation=int(cfg['heatmapAddSegmentation']),
                datasets=cfg["ValidationDataset"],
                libraryPath="datasets/DataLoader/libDataLoader.so")
            # Modify the way you call the dataset
            weight_val_array = dbValidation.get_token_frequencies()  #<- Not sure if this is ok to do ?
            validation_generator = TrainingDataGenerator(
                cfg=cfg, db=dbValidation, batch_size=cfg['batchSize'], numberOfTokens=cfg["tokensOut"], D=D,
                validation_data=True, class_weights=weight_val_array, workers=1, use_multiprocessing=False,
                max_queue_size=1)  #multiprocessing happens inside the dataloader
            validationDatasetLength = dbValidation.numberOfSamples
            outValLabels = dbValidation.get_labels()

        #The training set is very large, so depending on the system there are two ways to use it
        #try streaming it which is very slow due to I/O operations but can work on small VRAM systems
        #or load it all in memory and use regular TF mechanisms to train on it
        if True:  #(checkIfFileExists(cfg['COCOTrainingJSONPath'])):
            dbTrain = DataLoader(
                (cfg['inputHeight'], cfg['inputWidth'], 3),
                (cfg['outputHeight'], cfg['outputWidth'], cfg['outputChannels']),
                output16BitChannels=cfg['output16BitChannels'],
                #numberOfSamples = 10000, <- uncomment to just use a few samples for test
                streamData=int(cfg['streamDataset']),
                batchSize=cfg['batchSize'],
                numberOfThreads=cfg['DatasetLoaderThreads'],
                gradientSize=cfg['heatmapGradientSize'],
                PAFSize=cfg['heatmapPAFSize'],
                doAugmentations=
                0,  #int(cfg['dataAugmentation']),#0 = DISABLED AUGMENTATIONS TO HELP THIS int(cfg['dataAugmentation']),
                addPAFs=int(cfg['heatmapAddPAFs']),
                addBackground=int(cfg['heatmapGenerateSkeletonBkg']),
                addDepthMap=int(cfg['heatmapAddDepthmap']),
                addNormals=int(cfg['heatmapAddNormals']),
                addSegmentation=int(cfg['heatmapAddSegmentation']),
                datasets=cfg["TrainingDataset"],
                libraryPath="datasets/DataLoader/libDataLoader.so")
            dbTrain.updateJointDifficulty(cfg['keypoint_difficulty'])

            # Modify the way you call the dataset
            weight_array = dbTrain.get_token_frequencies()
            outLabels = dbTrain.get_labels()
            dataset_generator = TrainingDataGenerator(
                cfg=cfg, db=dbTrain, batch_size=cfg['batchSize'], numberOfTokens=cfg["tokensOut"], D=D,
                labels=outLabels, log_dir=log_dir, class_weights=weight_array, workers=1, use_multiprocessing=False,
                max_queue_size=10)  #multiprocessing happens inside the dataloader
            trainingDatasetLength = dbTrain.numberOfSamples
            shuffleData = False  #<- shuffling is done inside the generator
        #----------------------------------------------------------------

        # Print the shapes of inputs and outputs
        print("Training Configuration :", cfg)
        logTrainingParameters(cfg, log_dir)

        print("DISABLE HEATMAP OUTPUT :")
        dbValidation.disableHeatmapOutput()
        dbTrain.disableHeatmapOutput()

        # Define Learning Rate Scheduler callback
        #lr_callback = tf.keras.callbacks.LearningRateScheduler(custom_lr_scheduler) <- This is not configurable
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: custom_lr_scheduler(
            epoch, cfg['learningRateStart'], cfg['learningRateEnd']))

        #Early Stopping / Checkpointing
        whatToMonitor = cfg['earlyStoppingMonitor']
        howToMonitor = cfg['earlyStoppingHowToMonitor']

        if (onlyTrainingData) and (cfg['earlyStoppingMonitor'] == "val_loss"):
            print("You forgot to change the monitor to loss, fixing this automatically")
            cfg['earlyStoppingMonitor'] = "loss"

        # Define EarlyStopping/ModelCheckpoint callbacks
        #-------------------------------------------------------------------------
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=whatToMonitor,
            mode=howToMonitor,
            #monitor  = 'loss',           mode = 'min', # Monitor the Training loss metric / Mode should be 'min' because we want to minimize the loss metric
            #monitor = 'val_loss',        mode = 'min', # Monitor the Validation loss metric / Mode should be 'min' because we want to minimize the loss metric
            #monitor  = 'hdm',            mode = 'max', # Monitor the Training HDM metric / Mode should be 'max' because we want to maximize the  metric
            #monitor = 'val_hdm',         mode = 'max', # Monitor the Validation HDM metric / Mode should be 'max' because we want to maximize the  metric
            patience=cfg[
                'earlyStoppingPatience'],  # Number of epochs with no improvement after which training will be stopped
            min_delta=cfg[
                'earlyStoppingMinDelta'],  # Minimum change in the monitored quantity to qualify as an improvement
            verbose=1,  # Set to 1 for more verbose output
            restore_best_weights=
            True  # Restore model weights from the epoch with the best value of the monitored quantity
        )
        #-------------------------------------------------------------------------
        checkpointer = keras.callbacks.ModelCheckpoint(filepath="best.weights.h5", monitor=whatToMonitor,
                                                       mode=howToMonitor, verbose=1, save_freq='epoch',
                                                       save_best_only=True, save_weights_only=True)
        #-------------------------------------------------------------------------

        # Create a distributed dataset from the tensorflow datasets
        #--------------------------------------------------------------------------------------------------------------------------------
        #----trainingDataset            = trainingDataset.shuffle(100 * cfg['batchSize']).repeat(cfg['epochs']).batch(numberOfGPUs * cfg['batchSize'], drop_remainder=True)
        #----validationDataset          = validationDataset.repeat(cfg['epochs']).batch(numberOfGPUs * cfg['batchSize'], drop_remainder=True)

        #If strategies are restored this needs to be restored ->
        #distributedTrainingDataset = strategy.experimental_distribute_dataset(trainingDataset)

        # Create extra Metrics to have a better grasp of what is happening with the model
        HDM_THRESHOLD = 0.1
        hdm_metric = HeatmapDistanceMetric(threshold=HDM_THRESHOLD)
        hdm16bit_metric = HeatmapDistanceMetric(name='hdm16', threshold=HDM_THRESHOLD)  #(32767.0/120.0) * HDM_THRESHOLD
        #rsq_metric = RSquaredMetric()

        # Initialize the Adam Optimizer using configuration
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['learningRate'])

        #Enabling this multiplies training time from 6min to 1hour
        #tf.config.run_functions_eagerly(True) #This is needed for multi output

        #Decide on heatmap loss based on configuration
        hmloss = None
        if (cfg['loss'] == "mse"):
            hmloss = vanilla_mse_loss
        elif (cfg['loss'] == "combine"):
            print(bcolors.WARNING, "Using experimental combined loss..", bcolors.ENDC)
            hmloss = combined_loss
        elif (cfg['loss'] == "dssim"):
            print(bcolors.WARNING, "Using experimental dssim loss..", bcolors.ENDC)
            hmloss = dssim_loss
        else:
            print(bcolors.WARNING, "Using experimental dssim loss..", bcolors.ENDC)
            hmloss = cfg['loss']

        # Define custom loss for each output
        #losses  = {"t%02u"%i: GloVeMSELoss() for i in range(cfg["tokensOut"])}
        #metrics = {"t%02u"%i: ['accuracy']   for i in range(cfg["tokensOut"])}

        losses = dict()
        #for i in reversed(range(cfg["tokensOut"])): #<- Try reversing order(?)
        for i in range(cfg["tokensOut"]):  #<- Try reversing order(?)
            #Applying a higher weight to the losses for earlier tokens during training.
            #This should encourage the network to focus more on improving the accuracy of the earlier tokens by making their losses more prominent.
            #losses["t%02u"%i] = GloVeMSELoss(weight=10.0/((i+1) * (i+1)))
            losses["t%02u" % i] = GloVeMSELoss(weight=100.0)

        metrics = dict()
        for i in range(cfg["tokensOut"]):  #<- Try reversing order(?)
            metrics["t%02u" % i] = keras.metrics.CosineSimilarity(name='cos', axis=1)

        #Compile a model with the requested loss
        if ("outputTokens" in cfg) and (cfg["outputTokens"]):
            model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        else:
            #model.compile(optimizer=optimizer,
            #              loss         = {'heatmap_output': hmloss,             'heatmaps_output_16bit': hmloss},
            #              loss_weights = {'heatmap_output': 1.0 ,               'heatmaps_output_16bit': 1.0 },
            #              metrics      = {'heatmap_output': hdm_metric,         'heatmaps_output_16bit': hdm16bit_metric})
            #model.compile(optimizer=optimizer, loss=combined_two_loss, metrics=[hdm_metric])
            model.compile(optimizer=optimizer, loss=hmloss, metrics=[hdm_metric])
    #--------------------------------------------------------------------------------------------------------------------------------

    #Printout data/size summaries in screen
    #--------------------------------------------------------------------------------------------------------------------------------
    bytesPerValue = 1  # np.int8
    channels = deriveRGBChannelsFromCFG(cfg)
    heatmapNumber = deriveHeatmapChannelsFromCFG(cfg)
    estimatedInputByteSize = cfg['inputWidth'] * cfg['inputHeight'] * channels * trainingDatasetLength * bytesPerValue
    estimatedOutputByteSize = cfg['outputWidth'] * cfg[
        'outputHeight'] * heatmapNumber * trainingDatasetLength * bytesPerValue
    print("Input Data Size  : ", convert_bytes(estimatedInputByteSize), " ", channels, " channels")
    print("Output Data Size : ", convert_bytes(estimatedOutputByteSize), " ", heatmapNumber, " heatmaps")
    print("Total Data Size  : ", convert_bytes(estimatedInputByteSize + estimatedOutputByteSize))
    print("Total Data Size per GPU ( ", len(cfg['GPUsUsedForTraining']), "available ) : ",
          convert_bytes((estimatedInputByteSize + estimatedOutputByteSize) / numberOfGPUs))
    #--------------------------------------------------------------------------------------------------------------------------------

    # Train the model
    #--------------------------------------------------------------------------------------------------------------------------------
    print(bcolors.OKGREEN, "Starting training.. ", bcolors.ENDC)
    #--------------------------------------------------------------------------------------------------------------------------------
    distributedValidationDataset = None
    if (not onlyTrainingData):
        if (cfg['streamValidation']):
            print(bcolors.OKGREEN, "Streaming validation dataset from filesystem.. ", bcolors.ENDC)
            distributedValidationDataset = validation_generator
        else:
            print(bcolors.OKGREEN, "Loading whole validation dataset in memory.. ", bcolors.ENDC)
            distributedValidationDataset = validationDataset
    else:
        print(bcolors.WARNING, "Not using validation data.. ", bcolors.ENDC)
    #--------------------------------------------------------------------------------------------------------------------------------
    if (cfg['streamDataset']):
        print(bcolors.OKGREEN, "Streaming training dataset from filesystem.. ", bcolors.ENDC)
        distributedTrainingDataset = dataset_generator
    else:
        print(bcolors.OKGREEN, "Loading whole training dataset in memory.. ", bcolors.ENDC)
        distributedTrainingDataset = trainingDataset
    #--------------------------------------------------------------------------------------------------------------------------------

    history = None
    historyFT = None

    if (countResponses):
        import keras
        model = keras.saving.load_model("2d_pose_estimation/model.keras", custom_objects={
            'weighted_token_loss': weighted_token_loss
        }, compile=True, safe_mode=True)

        # Initialize a list to store hits for each list of tokens
        hits_per_token = [0] * 2048
        for index in range(100):  #(dbTrain.get_number_of_samples(dbTrain.db) // dbTrain.batchSize ):
            print("Batch ", index, "/", dbTrain.get_number_of_samples(dbTrain.db) // dbTrain.batchSize)
            rgb, tokens = distributedTrainingDataset.__getitem__(index)
            preds = model(rgb)
            print("Results ", preds)
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
                print(i, " - ", hits_per_token[i])

        sys.exit(0)

    #print("Validation data copy train blacklist!")
    #dbValidation.tokenblacklistkeys = dbTrain.tokenblacklistkeys
    #dbValidation.update_token_blacklist(dbValidation.tokenblacklistkeys,lowThreshold=2)
    print("Total number of train black listed keys : ", len(dbTrain.tokenblacklistkeys))
    print("Total number of validation black listed keys : ", len(dbValidation.tokenblacklistkeys))

    print("Now retrieve token weights!")
    weight_array = dbTrain.get_token_frequencies()

    # Convert the weight_array into a dictionary
    class_weight_dict = None
    """
   class_weight_dict = {i: float(weight) for i, weight in enumerate(weight_array)}
   with open("class_weight_dict_new.json", 'w') as json_file:
     json.dump(class_weight_dict, json_file, indent=4)
   #sys.exit(0)
   """

    if (cfg['epochsFrozen'] > 0):
        history = model.fit(
            distributedTrainingDataset,
            batch_size=cfg['batchSize'],
            epochs=cfg['epochsFrozen'],
            validation_data=distributedValidationDataset,
            shuffle=shuffleData,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, checkpointer, lr_callback,
                       DataAugmentation(cfg, dbTrain)]  #tensorboard_callback, <- this doesnt work
        )

        if (cfg['epochs'] > 0):
            print("Unfreezing RESNET 50 immediately to make sure it does not interfere with Y-NET loading")
            model = append_final_onehot_layer(model, maxtokens=cfg["tokensOut"], D=D,
                                              TokensOut=1)  #TokensOut=16 if combineData = False
            model.get_layer('resnet50').trainable = True  # 'resnet50' should be the name of the base model

    #Epochs can be set to zero to just do post training training
    #--------------------------------------------------------------------------------------------------------------------------------
    if (cfg['epochs'] > 0):

        # Step 2: Unfreeze the base model
        print("Unfreeze RESNET 50, and carry on training with one hot outputs")
        #Retrain with GloVe output
        distributedValidationDataset.returnOneHot = True
        distributedTrainingDataset.returnOneHot = True
        distributedValidationDataset.combineData = True
        distributedTrainingDataset.combineData = True
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['learningRate'])

        class_weight_dict = {i: float(weight) for i, weight in enumerate(weight_array)}
        #model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=False) ,  metrics=['accuracy'])
        #model.compile(optimizer=optimizer, loss=WeightedBinaryCrossEntropy(weight_array,weight=1.0),  metrics=['accuracy'])
        #weight_array = weight_array * 100.0
        model.compile(optimizer=optimizer, loss=OneHotLoss(weight_array, weight=1.0),
                      metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)])
        """
     print("Unfreeze RESNET 50, and carry on training with GloVe outputs")
     #Retrain with GloVe output
     model = append_final_tanh_glove_layer(model, maxtokens=cfg["tokensOut"], D=D)
     model.get_layer('resnet50').trainable    = True  # 'resnet50' should be the name of the base model
     #Recompile the model with separate losses
     distributedValidationDataset.combineData = True
     distributedTrainingDataset.combineData   = True
     optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['learningRate'])
     #model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
     model.compile(optimizer=optimizer, loss=GloVeMSELoss(weight=100.0),  metrics=[keras.metrics.CosineSimilarity(name='cossim',axis=1)])
     """

        history = model.fit(
            distributedTrainingDataset,
            batch_size=cfg['batchSize'],
            epochs=cfg['epochs'],
            validation_data=distributedValidationDataset,
            shuffle=shuffleData,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, checkpointer, lr_callback,
                       DataAugmentation(cfg, dbTrain)]  #tensorboard_callback
        )
    #--------------------------------------------------------------------------------------------------------------------------------
    print(bcolors.WARNING, "Recompiling model using defaults to make it more portable..", bcolors.ENDC)
    model.compile(optimizer="adam", loss="mse")

    # Perform any model optimizations requested by configuration and then save and package everything
    #--------------------------------------------------------------------------------------------------------------------------------
    if (cfg['pruneModel']):
        from ymapnet.core.NNOptimize import pruneModel
        model = pruneModel(model, cfg, trainingDataset)

    if (cfg['clusterModel']):
        from ymapnet.core.NNOptimize import clusterModel
        model = clusterModel(model, cfg, trainingDataset)

    saveNNModel("2d_pose_estimation/tokens", model,
                formats=["keras"])  #Only use keras format to save space! formats=["keras","tf","tflite","onnx"]

    #Promote the training config (root, git-tracked) into the packaged runtime dir now that
    #the model has been successfully saved. Runtime reads 2d_pose_estimation/tokens.json.
    print(bcolors.OKGREEN, "Promoting ./tokens.json -> 2d_pose_estimation/tokens.json", bcolors.ENDC)
    os.system("cp tokens.json 2d_pose_estimation/tokens.json")

    if (history):
        logTrainingHistory(cfg, "2d_pose_estimation", "loss_history.txt", history)
    if (historyFT):
        logTrainingHistory(cfg, "2d_pose_estimation", "loss_finetune_history.txt", historyFT)

    #Save vocabulary
    os.system("cp datasets/descriptions/index_to_word.json 2d_pose_estimation/vocabulary.json")

    # Package output
    #--------------------------------------------------------------------------------------------------------------------------------
    print(bcolors.OKGREEN, "Training complete..", bcolors.ENDC)
    os.system("date +\"%y-%m-%d_%H-%M-%S\" > 2d_pose_estimation/date.txt")  #Tag date
    os.system("rm 2d_pose_estimation_tokens.zip")  #Make sure there is no zip file
    os.system("zip -r 2d_pose_estimation_tokens.zip 2d_pose_estimation/tokens.json 2d_pose_estimation/tokens/*"
              )  #Create zip of models
    #--------------------------------------------------------------------------------------------------------------------------------

    print(
        'You can see a summary using :\n tensorboard --logdir=2d_pose_estimation/tensorboard --bind_all && firefox http://127.0.0.1:6006'
    )

    # Upload results
    #--------------------------------------------------------------------------------------------------------------------------------
    print('To upload results (if you take too long it will timeout) :')
    os.system("timeout 5 scripts/uploadResults.sh")
    print("scp -P 2222 2d_pose_estimation_tokens.zip ammar@ammar.gr:/home/ammar/public_html/poseanddepth")
    print(" or ")
    print(
        "scp -P 2222 2d_pose_estimation_tokens.zip ammar@ammar.gr:/home/ammar/public_html/poseanddepth/archive/2d_pose_estimation_tokens_v%s.zip"
        % str(cfg['serial']))
