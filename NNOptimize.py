#!/usr/bin/python3

""" 
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
"""
#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------
def pruneModel(model,cfg,trainingDataset):
   print("Will now attempt to optimize model..!")
   import tensorflow as tf
   import tensorflow_model_optimization as tfmot
   initial_sparsity = 0.0
   final_sparsity = 0.75
   begin_step = 1000
   end_step = 5000
   pruning_params = {
                     'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                                                                              initial_sparsity=initial_sparsity,
                                                                              final_sparsity=final_sparsity,
                                                                              begin_step=begin_step,
                                                                              end_step=end_step
                                                                             ),
                     'BatchNormalization': []
                    }
   model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
   pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
   
   model.compile(optimizer='rmsprop', loss='mse', metrics=['mae', 'acc']) 

   model.fit(
             trainingDataset, 
             epochs=cfg['epochs'], 
             batch_size=cfg['batchSize'], 
             callbacks= pruning_callback, 
             verbose=1
            )
   return model
#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------
def clusterModel(model,cfg,trainingDataset):
   #https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html
   print("Will now attempt to cluster model..!")
   import tensorflow as tf
   import tensorflow_model_optimization as tfmot

   rmsprop=tf.keras.optimizers.RMSprop(learning_rate=cfg['learningRate'], rho=0.9, epsilon=tf.keras.backend.epsilon())  
   model.compile(
                 optimizer=rmsprop,
                 loss='mse',
                 metrics=['mae', 'acc']
                )

   cluster_weights = tfmot.clustering.keras.cluster_weights 
   clustering_params = {
                        'number_of_clusters': 32,
                        'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.LINEAR
                       }
   clustered_model = cluster_weights(model, **clustering_params)
   clustered_model.compile(
                           optimizer=rmsprop,
                           loss='mse',
                           metrics=['mae', 'acc']
                          )
   clustered_model.fit(
                       trainingDataset, 
                       epochs=cfg['epochs'], 
                       batch_size=cfg['batchSize'],
                       verbose=1
                      )


   # Prepare model for serving by removing training-only variables.
   return tfmot.clustering.keras.strip_clustering(clustered_model)
#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------
def quantizeModel(model,cfg,trainingDataset): 
   print("Will now attempt to quantize model..!")
   import tensorflow as tf
   import tensorflow_model_optimization as tfmot

   quantize_model = tfmot.quantization.keras.quantize_model

   # q_aware stands for for quantization aware.
   q_aware_model = quantize_model(model)

   # `quantize_model` requires a recompile.
   q_aware_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae', 'acc']) 

   q_aware_model.fit(
                     trainingDataset, 
                     epochs=1, 
                     batch_size=cfg['batchSize'], 
                     verbose=1, 
                     validation_split=0.1
                    )

   q_aware_model.summary()
   return q_aware_model
#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------
def convertToTensorRT(model,trainIn=0,precision="fp32"):
 try:
   import os
   import sys
   import tensorflow as tf
   from tensorflow.python.compiler.tensorrt import trt_convert as trt
   print("Convert Model to TensorRT / ",precision)
   #-----------------------------------------------------------------
   os.system("rm -rf tensorRTIntermediateTFModel/")
   os.system("rm -rf tensorRTIntermediateTRTModel/")
   #-----------------------------------------------------------------
   model.save("tensorRTIntermediateTFModel", save_format='tf') #save directory..

   if (precision=="fp32"):
      precision_mode='FP32'
   elif (precision=="fp16"):
      precision_mode='FP16'
   elif (precision=="int8"):
      precision_mode='INT8'
   else:
      print("Unknown precision setting ",precision)
      return model


   # https://www.tensorflow.org/api_docs/python/tf/experimental/tensorrt/Converter
   # https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#usage-example
   print("\nConverting to TensorRT/Tensorflow model")
   converter = trt.TrtGraphConverterV2( input_saved_model_dir="tensorRTIntermediateTFModel", precision_mode=precision_mode )

   #with trt.Builder(TRT_LOGGER) as builder, builder.create_network(network_creation_flag) as network, trt.OnnxParser(network, TRT_LOGGER) as parser, builder.create_builder_config() as config:        
   #    profile = builder.create_optimization_profile()     
   #    profile.set_shape("input_1", (1, 224, 224, 3), (1, 224, 224, 3), (1, 224, 224, 3))
   #    config.add_optimization_profile(profile)
   #   engine = builder.build_engine(network, config)


   print("\nconverter.convert")
   if (trainIn!=0):
     converter.convert(calibration_input_fn=trainIn)
     converter.build(input_fn=trainIn)
   else:
     converter.convert()


   print("\nconverter.save")
   converter.save("tensorRTIntermediateTRTModel")
   model = tf.keras.models.load_model("tensorRTIntermediateTRTModel")

   os.system("rm -rf tensorRTIntermediateTFModel/")
   os.system("rm -rf tensorRTIntermediateTRTModel/")
 except:
   print("Error while performing TensorRT conversion")

 return model
#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------
if __name__ == '__main__':
    jsonPath = '2d_pose_estimation/configuration.json'
    model_path = "2d_pose_estimation/model.keras"
    formats = ["keras","tf","tflite","onnx"]

    print("DNNOptimize.py will now convert the model to all compatible formats (",formats,") ")

    from createJSONConfiguration import loadJSONConfiguration
    cfg = loadJSONConfiguration(jsonPath)

    from NNModel import load_keypoints_model
    from NNConverter import saveNNModel
    model,input_size,output_size,numHeatmaps = load_keypoints_model(model_path)

    if (cfg['pruneModel']): 
        model = pruneModel(model,cfg,trainingDataset)

    if (cfg['clusterModel']): 
        model = clusterModel(model,cfg,trainingDataset)

    saveNNModel("2d_pose_estimation",model,formats=formats) #Only use keras format to save space! 


