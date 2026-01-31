
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""
import os
import numpy as np
import time
#-----------------------------------------------------------------------------------------------------
class TFLiteExecutor():
  #tflite_convert --saved_model_dir=2d_pose_estimation --output_file=2d_pose_estimation/model.tflite
  def __init__(
               self,
               modelPath:str   = "2d_pose_estimation/model.tflite",
               inputWidth      = 220,
               inputHeight     = 220,
               targetWidth     = 96,
               targetHeight    = 96,
               outputChannels  = 18,
               numberOfThreads = 4,
              ):
               #Tensorflow attempt to be reasonable
               #------------------------------------------
               print("Using TF-Lite Runtime")
               import tensorflow as tf
               # Initialize the TFLite interpreter
               self.interpreter = tf.lite.Interpreter(model_path=modelPath,num_threads=numberOfThreads)
               self.interpreter.allocate_tensors()

               #tensor_details   = self.interpreter.get_tensor_details()
               #for tensor in tensor_details:
               #    print(f"Tensor Name: {tensor['name']}")
               #    print(f"Shape: {tensor['shape']}")
               #    print(f"Quantization Parameters: {tensor['quantization']}")

               self.input_details  = self.interpreter.get_input_details()
               self.output_details = self.interpreter.get_output_details()
               #------------------------------------------
               self.input_size        = (inputWidth,inputHeight)
               self.output_size       = (targetWidth,targetHeight)
               self.numberOfHeatmaps  = outputChannels
               self.description       = None
               self.activity          = None
               #------------------------------------------
#-----------------------------------------------------------------------------------------------------
  def predict(self,image):
        image_batch = np.expand_dims(image, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], image_batch)
        self.interpreter.invoke()
        predictions = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
        #print("Prediction : ",predictions[0].shape)
        return predictions[0][0]
#-----------------------------------------------------------------------------------------------------


# Ensure all outputs are converted to float32
def to_float32(tensor):
        return tensor.astype(np.float32)


#-----------------------------------------------------------------------------------------------------
class TFExecutor():
  def __init__(
               self,
               modelPath:str  = "2d_pose_estimation/model.keras",
               inputWidth     = 220,
               inputHeight    = 220,
               targetWidth    = 96,
               targetHeight   = 96,
               outputChannels = 18,
               VRAMLimit      = None,
               fp16           = False,
               profiling      = False
              ):
               self.input_size        = (inputWidth,inputHeight)
               self.output_size       = (targetWidth,targetHeight)
               self.numberOfHeatmaps  = outputChannels
               self.profiling         = profiling
               self.heatmaps          = None
               self.heatmaps_16b      = None
               self.description       = None
               self.multihot_description = None
               self.activity          = None
               #Tensorflow attempt to be reasonable
               #------------------------------------------
               print("Using Tensorflow Runtime")


               self.limitVRAMUse = VRAMLimit
               #self.limitVRAMUse = 4800 #<- The network should only need 4.1GB
 
               import os
               os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION']='false'
               #Make sure CUDA cache is not disabled!
               os.environ['CUDA_CACHE_DISABLE'] = '0'
               #Try to presist cudnn 
               os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
               #Try to allocate as little memory as possible
               if self.limitVRAMUse is None:
                 os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #<- Incompatible with setting size
               #Use seperate threads so execution is not throttled by CPU
               os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
               #0 = all messages are logged (default behavior)
               #1 = INFO messages are not printed
               #2 = INFO and WARNING messages are not printed
               #3 = INFO, WARNING, and ERROR messages are not printed
               os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
               #improve the stability of the auto-tuning process used to select the fastest convolution algorithms
               os.environ['TF_AUTOTUNE_THRESHOLD'] = '1'
               import keras
               import tensorflow as tf

               if self.limitVRAMUse is not None:
                 gpus = tf.config.list_physical_devices('GPU')
                 if gpus:
                   try:
                       tf.config.set_visible_devices(gpus[0], 'GPU')
                       tf.config.set_logical_device_configuration(
                           gpus[0],
                           [tf.config.LogicalDeviceConfiguration(memory_limit=self.limitVRAMUse)]  # 6GB limit
                       )
                       print("Set GPU memory limit to ",self.limitVRAMUse," MB")
                       logical_gpus = tf.config.list_logical_devices('GPU')
                       print(len(gpus), "Physical GPU(s),", len(logical_gpus), "Logical GPU(s)")

                   except RuntimeError as e:
                       print("Error setting memory limit: ", e)

               print("Tensorflow version : ",tf.__version__)
               print("Keras version      : ",keras.__version__) #<- no longer available in TF-2.13
               print("Numpy version      : ",np.__version__)
               #-----------------------------
               from tensorflow.python.platform import build_info as tf_build_info
               print("TF/CUDA version    : ",tf_build_info.build_info['cuda_version'])
               print("TF/CUDNN version   : ",tf_build_info.build_info['cudnn_version'])


               devices = self.get_available_devices()
               print("Available Tensorflow devices are : ",devices)
               self.device = '/device:CPU:0'
               for device in devices:
                   if (device.find("GPU")!=-1):
                       self.device = device 
               print("Selecting device : ",self.device)

               if (self.profiling):
                  self.tensorboard = keras.callbacks.TensorBoard(log_dir = "profiling",histogram_freq = 1) #tf.
                  self.startProfiling()

               self.inf_dtype = np.float32

               if (fp16):
                 from tensorflow.keras import mixed_precision
                 print("Enabling mixed precision (float16) inference..")
                 mixed_precision.set_global_policy('mixed_float16')
                 self.inf_dtype = np.float16

               from NNModel import load_keypoints_model
               self.model,self.input_size,self.output_size,self.numberOfHeatmaps = load_keypoints_model(modelPath)
               #self.model.export("2d_pose_estimation", "tf_saved_model")    #Debug models

               if (fp16):
                  print("Policy:", mixed_precision.global_policy())
                  print("Compute dtype:", self.model.compute_dtype)
                  self.model = tf.keras.models.clone_model(self.model)


               #------------------------------------------
    # Deleting (Calling destructor)
  def __del__(self):
               if (self.profiling):
                  self.stopProfiling()
#-------------------------------------------------------------
  def startProfiling(self):
       print("Starting Tensorflow Profiling (this run will be slower than usual)..\n")
       os.system("rm -rf profiling")
       import tensorflow as tf
       tf.profiler.experimental.start('profiling')
#-------------------------------------------------------------
  def stopProfiling(self):
       print("Stopping Tensorflow Profiling..\n")
       import tensorflow as tf
       tf.profiler.experimental.stop()
       print("Please run:\n")
       print("   tensorboard --logdir profiling\n")
#-------------------------------------------------------------
  def get_available_devices(self):
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']
#-----------------------------------------------------------------------------------------------------
  def print_all_available_prediction_types(self,predictions):
    print("------ Prediction Types ------")
    if isinstance(predictions, list):
        for i, pred in enumerate(predictions):
            if not hasattr(pred, 'shape'):
                print(f"[{i}] Not a tensor-like object: {type(pred)}")
                continue
            print(f"[{i}] shape: {pred.shape}, dtype: {pred.dtype}", end='')

            if pred.ndim == 4:
                print(" --> Possibly heatmaps (e.g., (1, H, W, C))")
            elif pred.ndim == 3:
                print(" --> Possibly sequence/temporal features")
            elif pred.ndim == 2:
                dim = pred.shape[1]
                if dim == 300:
                    print(" --> Possibly GloVe token embedding output (300D)")
                elif dim > 100 and dim < 300:
                    print(" --> Possibly compact embedding or multi-hot (mid-size vector)")
                else:
                    print(" --> Possibly multi-hot or classification logits")
            elif pred.ndim == 1:
                print(" --> Possibly single vector output")
            else:
                print(" --> Unknown shape")
    else:
        # Handle single prediction array (not a list)
        if not hasattr(predictions, 'shape'):
            print("Prediction is not a tensor-like object.")
            return False
        print(f"Shape: {predictions.shape}, dtype: {predictions.dtype}", end='')

        if predictions.ndim == 4:
            print(" --> Possibly heatmaps (e.g., (1, H, W, C))")
        elif predictions.ndim == 3:
            print(" --> Possibly temporal or sequence data")
        elif predictions.ndim == 2:
            print(" --> Possibly multi-hot or token output")
        elif predictions.ndim == 1:
            print(" --> Possibly single vector output")
        else:
            print(" --> Unknown shape")

    print("------ End of Prediction Types ------")
    return True
#-----------------------------------------------------------------------------------------------------
  def predict(self,image):
        if (image.ndim==3):
           image_batch = np.expand_dims(image.astype(self.inf_dtype), axis=0)
        elif (image.ndim==4):
           image_batch = image
        else:
           print("Unexpected dimensions ",image.ndim)

        if (self.profiling):
            print(type(self.model))
            predictions = self.model.predict(image_batch,callbacks = [self.tensorboard])
        else:
            predictions = self.model.predict(image_batch,verbose=0)
            #predictions = self.model(image_batch, training=False) 



        #Do FP16 -> FP32 conversions if needed
        #if isinstance(predictions, list):
        #   predictions = [to_float32(p) for p in predictions]
        #else:
        #   predictions = to_float32(predictions)


        #print("predictions count = ",len(predictions))
        #Printout for all outputs to make sure they are correctly handled
        #self.print_all_available_prediction_types(predictions)

        if type(predictions) is list:
            pass
        else:
          if (predictions.ndim == 4):
            #predictions.shape =  (1, 256, 256, 18)
            return predictions[0]

        #Else we have a list of outputs so find the correct one! 
        heatmap_16bOutputIndex  =  None 
        heatmapOutputIndex      =  9 
        multiHotOutputIndex     =  8
        tokenOutputIndices      =  list()

        #16B 
        #---------------------------
        heatmap_16bOutputIndex  =  1 
        heatmapOutputIndex      =  0 
        multiHotOutputIndex     =  10

        for i,item in enumerate(predictions):
           #print(i," items : ",predictions[i].shape) 
           #print(i," dims :  ",predictions[i].ndim) 
           if predictions[i].ndim == 4:
               #print("Heatmap %u has %u elements " % (i,predictions[i].shape[3]) )
               if (predictions[i].shape[3] == 1):
                   heatmap_16bOutputIndex = i
               else:
                   heatmapOutputIndex = i


               self.heatmaps = predictions[heatmapOutputIndex][0]
           elif predictions[i].ndim == 3:
               print(" unknown output @ ",i)
           elif predictions[i].ndim == 2:
               #print(" multihot @ ",i) 
               #print("  multihot length = ",len(predictions[i][0]))
               dataLength = len(predictions[i][0])
               if (dataLength == 300): #300 dim is the GloVe vectors
                  tokenOutputIndices.append(i)
               else:  
                  multiHotOutputIndex = i #Other dimensionality should be the multihot output
                  self.multihot_description = predictions[multiHotOutputIndex]
        
        if (len(tokenOutputIndices)>0):
           selected_predictions = [predictions[i][0] for i in tokenOutputIndices] 
           self.description = np.vstack(selected_predictions)
 
        if (heatmap_16bOutputIndex is not None):
           self.heatmaps_16b = predictions[heatmap_16bOutputIndex][0]
 
        return self.heatmaps #predictions[heatmapOutputIndex][0]
#-----------------------------------------------------------------------------------------------------


  def predict_multi(self,image):
        if (image.ndim==3):
           image_batch = np.expand_dims(image.astype(self.inf_dtype), axis=0)
        elif (image.ndim==4):
           image_batch = image
        else:
           print("Unexpected dimensions ",image.ndim)

        if (self.profiling):
            print(type(self.model))
            predictions = self.model.predict(image_batch,callbacks = [self.tensorboard])
        else:
            predictions = self.model.predict(image_batch,verbose=0)
            #predictions = self.model(image_batch, training=False) 

        #print("predictions count = ",len(predictions))
 
        if type(predictions) is list:
            pass
        else:
          if (predictions.ndim == 4):
            #predictions.shape =  (1, 256, 256, 18)
            return predictions

        #Else we have a list of outputs so find the correct one! 
        heatmapOutputIndex  =  9 
        multiHotOutputIndex =  8
        tokenOutputIndices    =  list()

        for i,item in enumerate(predictions):
           #print(i," items : ",predictions[i].shape) 
           #print(i," dims :  ",predictions[i].ndim) 
           if predictions[i].ndim == 4:
               #print(" heatmaps @ ",i) 
               heatmapOutputIndex = i
               self.heatmaps = predictions[heatmapOutputIndex]
           elif predictions[i].ndim == 3:
               print(" unknown output @ ",i)
           elif predictions[i].ndim == 2:
               #print(" multihot @ ",i) 
               #print("  multihot length = ",len(predictions[i][0]))
               dataLength = len(predictions[i])
               if (dataLength == 300): #300 dim is the GloVe vectors
                  tokenOutputIndices.append(i)
               else:  
                  multiHotOutputIndex = i #Other dimensionality should be the multihot output
                  self.multihot_description = predictions[multiHotOutputIndex]
        
        if (len(tokenOutputIndices)>0):
           selected_predictions = [predictions[i] for i in tokenOutputIndices] 
           self.description = np.vstack(selected_predictions)
 

        return self.heatmaps #predictions[heatmapOutputIndex][0]
#-----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
class ONNXExecutor():
  #python3 -m tf2onnx.convert --saved-model 2d_pose_estimation --opset 17 --tag serve --output 2d_pose_estimation/model.onnx
  def __init__(
               self,
               modelPath:str  = "2d_pose_estimation/model.onnx",
               inputWidth     = 220,
               inputHeight    = 220,
               targetWidth    = 96,
               targetHeight   = 96,
               outputChannels = 18
              ): 
               print("Using ONNX Runtime") 
               import onnxruntime as ort
               import onnx
               self.input_size        = (inputWidth,inputHeight)
               self.output_size       = (targetWidth,targetHeight)
               self.numberOfHeatmaps  = outputChannels
               self.heatmaps          = None 
               self.multihot_description = None
               self.description       = None
               self.activity          = None
               #------------------------------------------
               self.sess_options = ort.SessionOptions()
               self.sess_options.log_severity_level = 3 #<- log_level
               self.sess_options.intra_op_num_threads = 4
               #self.sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
               self.sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
               self.sess_options.inter_op_num_threads = 4
               self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
               #------------------------------------------
               onnxModelForCheck  = onnx.load(modelPath)
               onnx.checker.check_model(onnxModelForCheck)
               print("ONNX devices available : ", ort.get_device()) 
               providers               = ['CPUExecutionProvider']
               #providers               = ['CUDAExecutionProvider']
               self.options = ort.SessionOptions()
               self.model              = ort.InferenceSession(modelPath, providers=providers, sess_options=self.options)
               for i in range(0,len(self.model.get_inputs())): 
                  print("ONNX INPUTS ",self.model.get_inputs()[i].name)
                  self.inputName = self.model.get_inputs()[i].name
               self.model_input_name   = self.model.get_inputs()
               #------------------------------------------
#-----------------------------------------------------------------------------------------------------
  def predict(self,image):
        imageONNX = np.expand_dims(image, axis=0)
        #------------------------------------------------------------------- 
        thisInputONNX = { self.inputName : imageONNX.astype('float32')}
        #Run input through MocapNET
        output_names_onnx = [otp.name for otp in self.model.get_outputs()]
        predictions = self.model.run(output_names_onnx,thisInputONNX)[0]
        #print("Prediction : ",predictions[0].shape)
        return predictions[0]
#-----------------------------------------------------------------------------------------------------





#-----------------------------------------------------------------------------------------------------
class NNExecutor():
  def __init__(
               self,
               engine:str="tensorflow",
               modelPath:str  = "2d_pose_estimation/",
               inputWidth     = 220,
               inputHeight    = 220,
               targetWidth    = 96,
               targetHeight   = 96,
               outputChannels = 18,
               profiling      = False
              ):
                defaultModelPath = "2d_pose_estimation/"
                self.hz    = 0.0
                self.model = None
                print ("NNExecutor for ",engine)
                print ("Asked to load from: ",modelPath)

                if (engine=="tensorflow")  or (engine=="tf"):
                        if (modelPath==defaultModelPath):
                            modelPath="%s/model.keras"%modelPath
                        self.model = TFExecutor(
                                                modelPath      = modelPath,
                                                profiling      = profiling,
                                                inputWidth     = inputWidth,
                                                inputHeight    = inputHeight,
                                                targetWidth    = targetWidth,
                                                targetHeight   = targetHeight,
                                                outputChannels = outputChannels
                                                )
                elif (engine=="tf-lite") or (engine=="tflite"):
                        if (modelPath==defaultModelPath):
                            modelPath="%s/model.tflite"%modelPath
                        self.model = TFLiteExecutor(
                                                    modelPath      = modelPath,
                                                    inputWidth     = inputWidth,
                                                    inputHeight    = inputHeight,
                                                    targetWidth    = targetWidth,
                                                    targetHeight   = targetHeight,
                                                    outputChannels = outputChannels
                                                    )
                elif (engine=="onnx"):
                        if (modelPath==defaultModelPath):
                            modelPath="%s/model.onnx"%modelPath
                        self.model = ONNXExecutor(
                                                  modelPath      = modelPath,
                                                  inputWidth     = inputWidth,
                                                  inputHeight    = inputHeight,
                                                  targetWidth    = targetWidth,
                                                  targetHeight   = targetHeight,
                                                  outputChannels = outputChannels
                                                 )
               #------------------------------------------
#-----------------------------------------------------------------------------------------------------
  def predict(self,image):
        #------------------------------------------
        start      = time.time()
        prediction = self.model.predict(image)
        seconds    = time.time() - start
        self.hz    = 1 / (seconds+0.0001)
        #------------------------------------------
        return prediction
#-----------------------------------------------------------------------------------------------------
  def predict_multi(self,image):
        #------------------------------------------
        start      = time.time()
        prediction = self.model.predict_multi(image)
        seconds    = time.time() - start
        self.hz    = 1 / (seconds+0.0001)
        #------------------------------------------
        return prediction
#-----------------------------------------------------------------------------------------------------
  def heatmaps_16b(self):
        #------------------------------------------
        return self.model.heatmaps_16b 
        #------------------------------------------
#-----------------------------------------------------------------------------------------------------
  def multihot(self):
        #------------------------------------------
        return self.model.multihot_description 
        #------------------------------------------
  def description(self):
        #------------------------------------------
        return self.model.description 
        #------------------------------------------
#-----------------------------------------------------------------------------------------------------
  def activity(self):
        #------------------------------------------
        return self.model.activity 
        #------------------------------------------
