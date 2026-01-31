import os
import keras
import numpy as np
from tools import bcolors

# ========================================================================================
# ========================================================================================
# ========================================================================================

def saveNNModel(path,model,formats=["keras","tf","tflite","onnx"]):
   # Save the trained model
   if ("keras" in formats):
     print(bcolors.OKGREEN,"Saving result as keras model..",bcolors.ENDC)
     if not os.path.exists(path):
        os.makedirs(path)
     model.save('%s/model.keras' % path)

   # Save the exported tensorflow model
   if ("tf" in formats) or ("tflite" in formats) or ("onnx" in formats):
      print(bcolors.OKGREEN,"Saving result as tensorflow model..",bcolors.ENDC)
      model.export(path, "tf_saved_model")
      #tf.saved_model.save(model,'2d_pose_estimation') #TF 2.15.0

   # Save the tf-lite model
   if ("tflite" in formats) and ("tf" in formats):
        print(bcolors.OKGREEN, "Saving result as a FP-16 tf-lite model..", bcolors.ENDC)
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model('2d_pose_estimation/')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        with open(f'2d_pose_estimation/model.tflite', 'wb') as f:
            f.write(tflite_model)
   elif ("tflite" in formats):
        print(bcolors.OKGREEN,"Saving result as a Regular tf-lite model using tflite_convert..",bcolors.ENDC)
        os.system("tflite_convert --saved_model_dir=2d_pose_estimation --output_file=2d_pose_estimation/model.tflite")
        #converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir="2d_pose_estimation")
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #tflite_model = converter.convert()
        #with open('2d_pose_estimation/model.tflite', 'wb') as f:
        #        f.write(tflite_model)

   # Save the onnx model
   if ("onnx" in formats):
      print(bcolors.OKGREEN,"Saving result as onnx model..",bcolors.ENDC)
      os.system("python3 -m tf2onnx.convert --saved-model 2d_pose_estimation --opset 17 --tag serve --output 2d_pose_estimation/model.onnx")
   print("Done saving model")

# ========================================================================================
# ========================================================================================
# ========================================================================================

def representative_data_gen_all(representative_dataset):
    for input_value in representative_dataset:
        # Ensure the input_value has the correct shape and type
        input_value = input_value.astype(np.float32)  # Ensure the input is float32
        input_value = np.expand_dims(input_value, axis=0)  # Add batch dimension
        yield [input_value]

def representative_data_gen(representative_dataset, number_of_samples=200):
    num_samples = min(number_of_samples, len(representative_dataset))
    indices = np.random.choice(len(representative_dataset), num_samples, replace=False)
    for idx in indices:
        input_value = representative_dataset[idx].astype(np.float32)
        input_value = np.expand_dims(input_value, axis=0)  # Add batch dimension
        yield [input_value]

def saveNNTFLiteINT8Model(model,representative_dataset, number_of_samples=200): 
        saveNNModel('2d_pose_estimation/',model,["tf"])
        print(bcolors.OKGREEN, "Saving result as INT8 tf-lite model..", bcolors.ENDC)
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model('2d_pose_estimation/')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_data_gen(representative_dataset, number_of_samples=number_of_samples)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.int8   # or tf.int8

        print(bcolors.OKGREEN,"Please wait INT8 conversion will take some time..",bcolors.ENDC)
        tflite_model = converter.convert()
        with open('2d_pose_estimation/model_int8.tflite', 'wb') as f:
            f.write(tflite_model)

def saveNNTFLiteFP16Model(model,representative_dataset, number_of_samples=200): 
        saveNNModel('2d_pose_estimation/',model,["tf"])
        print(bcolors.OKGREEN, "Saving result as FP16 tf-lite model..", bcolors.ENDC)
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model('2d_pose_estimation/')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_data_gen(representative_dataset, number_of_samples=number_of_samples)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        print(bcolors.OKGREEN,"Please wait FP16 conversion will take some time..",bcolors.ENDC)
        tflite_model = converter.convert()
        with open('2d_pose_estimation/model_fp16.tflite', 'wb') as f:
            f.write(tflite_model)

