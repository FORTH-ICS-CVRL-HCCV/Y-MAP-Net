#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""

#Dependencies should be :
#tensorflow-2.15.0 needs CUDA 12.2, CUDNN 8.9.4 and is built with Clang 16.0.0 and Bazel 6.1.0 
#python3 -m pip install tensorflow==2.15.0 numpy tensorboard opencv-python wget

import os
import sys
import time
from tools import bcolors
#----------------------------------------------------------------------------------------
try:
 import json
 import cv2
 import numpy as np
 from numpy.linalg import norm
 #import keras
 #import tensorflow as tf
 from createJSONConfiguration import loadJSONConfiguration
 from imageProcessing import castNormalizedCoordinatesToOriginalImage,castNormalizedBBoxCoordinatesToOriginalImage,resize_image_no_borders,resize_image_with_borders
 from resolveJointHierarchy import resolveJointHierarchy,resolveJointHierarchyNew,drawSkeletons

 sys.path.append('datasets/')
 from calculateNormalsFromDepthmap import compute_normals, rgb_to_grayscale, apply_bilateral_filter, improve_depthmap, integrate_normals
except Exception as e:
 print(bcolors.WARNING,"Could not import libraries!",bcolors.ENDC)
 print("An exception occurred:", str(e))
 print("Issue:\n source venv/bin/activate")
 print("Before running this script")
 sys.exit(1)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#----------------------------------------------------------------------------------------

def loadJSON(json_file_path):
  import json
  data = dict()
  # Read the JSON file and load its contents into a dictionary
  with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)
  return data

def get_key_from_index(d, index):
    """
    Given a dictionary and an index, returns the key name corresponding to the index.
    
    Parameters:
    d (dict): The dictionary with keys as names and values as indices.
    index (int): The index for which the corresponding key name is to be returned.
    
    Returns:
    str: The key name corresponding to the given index. If the index is not found, returns 'Index not found'.
    """
    for key, value in d.items():
        if value == index:
            return key
    return 'Index not found'

#----------------------------------------------------------------------------------------
def preprocess_image(frame, target_size=(128, 128), add_borders=False):
    # Resize the frame to the target size and normalize pixel values
    image = frame.astype(np.uint8)
    if (add_borders):
          #print("Going to ",target_size," also adding borders")
          image, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset  = resize_image_with_borders(image,target_size)
          #print("Recovered image size ",image.shape," ")
    else:
          #print("Going to ",target_size," without borders")
          image, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset  = resize_image_no_borders(image,target_size)
          #print("Recovered image size ",image.shape," ")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    return image_rgb, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset
#----------------------------------------------------------------------------------------
def checkIfFileExists(filename):
    return os.path.isfile(filename)
#----------------------------------------------------------------------------------------
def read_json_files(file_path):
    """
    Read JSON files from given file path.
    """
    import json
    with open(file_path, 'r') as file:
            data = json.load(file) 
            return data
    return None
#----------------------------------------------------------------------------------------
def decodeSingleOneHotDescriptionToString(vocabulary, onehot):
    if (vocabulary is None) or (onehot is None):
          return None
    # Invert the vocabulary to get a mapping from index to word
    index_to_word = {int(k): v for k, v in vocabulary.items()}
    
    # Initialize an empty list to store the words
    description_words = []
    
    # Loop through each vector in the one-hot encoding
    for vector in onehot:
        # Find the index of the maximum value (which indicates the presence of a word)
        max_index = np.argmax(vector)
        
        # Map the index to word and add it to the description words list
        word = index_to_word[max_index]
        description_words.append(word)
    
    # Join all the words to form the final description string
    description = ' '.join(description_words)
    
    return description
#----------------------------------------------------------------------------------------
def decodeOneHotDescriptionToString(vocabulary, onehot, token_threshold=0.24, K=7):
    if (vocabulary is None) or (onehot is None):
        print("Cannot decodeOneHotDescriptionToString")
        return None
    
    # Invert the vocabulary to get a mapping from index to word
    index_to_word = {int(k): v for k, v in vocabulary.items()}
    
    # Initialize an empty list to store the words
    description_words = []
 
    #K=7
    #if (K>len(onehot)):
    #    print("One Hot vector has ",len(onehot)," elements, so setting K to this value")
    #    for i,vector in enumerate(onehot):
    #       print("Vector ",i," -> ",vector.shape)
    #    K=len(onehot)
    
    # Loop through each vector in the one-hot encoding
    for vector in onehot:
        # Find the indices of the top K values
        top_K_indices = np.argsort(vector)[-K:][::-1]
        
        # Map the indices to words and their corresponding values
        top_words = []
        for idx in top_K_indices:
            if (token_threshold<vector[idx]):
               if not idx in index_to_word:
                  print("Failure resolving ",idx)
                  print(index_to_word)
                  raise ValueError('Fix your vocabulary file to reflect what the NN outputs.')

               top_words.append((index_to_word[idx], vector[idx]))          

        #top_words = [(index_to_word[idx], vector[idx]) for idx in top_K_indices]
        
        # Format the top words and values as desired
        for word, value in top_words:
          description_words.append("%s:%0.2f" % (word,value))
        #formatted_words = ','.join([f'{word}:{value:.4f}' for word, value in top_words])
        #description_words.append(f'({formatted_words})')
    
    # Join all the formatted strings to form the final description string
    description = ' '.join(description_words)
    
    return description
#----------------------------------------------------------------------------------------
def decodeClassesDescriptionToString(vocabulary, classscores):
    if (vocabulary is None) or (classscores is None):
        return None
    # Ensure vocabulary and classscores have the same length
    if len(vocabulary) != len(classscores):
        raise ValueError("Length of vocabulary and classscores must be the same")
    
    # Create a list of (score, token) tuples
    activations = [(classscores[i], vocabulary[str(i)]) for i in range(len(classscores))]
    
    # Sort activations based on scores in descending order
    activations.sort(reverse=True, key=lambda x: x[0])
    
    # Take top 10 activations
    top10 = activations[:50]
    
    # Format the top 10 activations into a string
    result = "\n".join(f"{score}: {token}" for score, token in top10)
    
    return result
#----------------------------------------------------------------------------------------
def top_16_probabilities(probabilities, id_to_string_dict):
    # Convert TensorFlow tensor to numpy array (if needed)
    if hasattr(probabilities, 'numpy'):
        probabilities = probabilities.numpy()
    
    # Flatten the probabilities array to ensure it's 1D
    probabilities = probabilities.flatten()

    # Get the indices of the 16 highest probabilities
    top_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:16]

    # Create a list of tuples (probability, string description) for the top 16
    top_16 = [(probabilities[i], id_to_string_dict.get(str(i), "Unknown ID")) for i in top_indices]

    return top_16
#----------------------------------------------------------------------------------------
# Load GloVe embeddings
def load_glove_embeddings(file_path, embedding_dim=50):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index
#----------------------------------------------------------------------------------------
# Compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    # Ensure both vectors are 1D numpy arrays of the same shape
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()
    
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return -1  # Return the lowest similarity for zero vectors
    
    # Compute cosine similarity as a scalar
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
#----------------------------------------------------------------------------------------
def mean_squared_error(vec1, vec2):
    # Ensure both vectors are 1D numpy arrays of the same shape
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()
    
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape.")
    
    # Compute MSE as the average of squared differences
    return np.mean((vec1 - vec2) ** 2)
#----------------------------------------------------------------------------------------
# Convert a vector back to words by finding the closest word vectors
def vector_to_sentence(vector, embeddings_index, embedding_dim=50, top_k=5):
    closest_words = []
    
    # Flatten the input vector to ensure it matches the embedding dimensions
    vector = np.asarray(vector).flatten()
    
    # Iterate over all words in the GloVe embedding index
    for word, embedding_vector in embeddings_index.items():
        if len(embedding_vector) == embedding_dim:
            similarity = cosine_similarity(vector, embedding_vector)
            if similarity > -1:  # Only add non-zero similarity results
                closest_words.append((word, similarity))
    
    # Sort by similarity and return the top_k words
    closest_words = sorted(closest_words, key=lambda x: x[1], reverse=True)
    return [word for word, sim in closest_words[:top_k]]
#----------------------------------------------------------------------------------------
# Convert individual word vectors back to words by finding the closest word vectors
def vectors_to_sentencePerWord(word_vectors, embeddings_index, embedding_dim=50):
    sentence_words = []
    
    for vector in word_vectors:
        # Ensure the vector is flattened and properly sized before comparison
        closest_words = vector_to_sentence(vector, embeddings_index, embedding_dim=embedding_dim, top_k=1)
        sentence_words.append(closest_words[0] if closest_words else "")
    
    return " ".join(sentence_words)
#----------------------------------------------------------------------------------------
def vector_to_sentence_close(vector, glove_embeddings,useMSE=False, D=50):
    closest_words = []
    
    # Flatten the input vector to ensure it matches the embedding dimensions
    vector = np.asarray(vector).flatten()

    empty_vector  = np.full(D,0)
    zero_similarity = mean_squared_error(vector, empty_vector)
    #print("Zero sim ",zero_similarity)
    if (zero_similarity<0.003):
       return [' ']
    
    # Iterate over all words in the GloVe embedding index
    #print("Check vector ",vector)
    #print("Against ",len(glove_embeddings.items())," vectors")


    glove_keys = list(glove_embeddings.keys())

    if (useMSE):
      bestScore = 100000
      bestIndex = 0
      for i,embedding_vector in enumerate(glove_embeddings.values()):
        #print("Cosine similarity of ",vector," against ",embedding_vector)
        similarity = mean_squared_error(vector, embedding_vector)
        if (bestScore>similarity):
            #print("Key ",glove_keys[i]," is better than ",glove_keys[bestIndex])
            #print("Beats it by  ",bestScore-similarity)
            bestScore = similarity
            bestIndex = i
    else:
      bestScore = -1
      bestIndex = 0
      for i, embedding_vector in enumerate(glove_embeddings.values()):
        # Calculate cosine similarity
        similarity = cosine_similarity(vector, embedding_vector)
        
        # Update if this similarity is better than the current best
        if similarity > bestScore:
            bestScore = similarity
            bestIndex = i


    closest_words.append("%s(%0.4f)" % (glove_keys[bestIndex],bestScore) )
    
    return closest_words
#----------------------------------------------------------------------------------------
class TokenEstimator2D:
    def __init__(self, modelPath="2d_pose_estimation", token_threshold=20, engine="tensorflow", profiling = False):
        self.model_path     = '%s/tokens' % modelPath
        self.cfg            = loadJSONConfiguration("%s/../tokens.json" % self.model_path)
        #self.instanceLabels = loadJSON("datasets/detectron2/labels.json")
        self.tokenLabels    = loadJSON("%s/vocabulary.json" % modelPath)
        # Specify the names of keypoints (change accordingly based on your model's keypoint order)
        self.keypoint_names = self.cfg['keypoint_names']
        print("Reported Keypoints : ",self.keypoint_names) 
        modelFile = "%s/model.keras"%self.model_path
        print("Will load token only from : ",modelFile) 
        #---------------------------------------------------------------------
        import keras
        keras.config.enable_unsafe_deserialization()
        self.keypoints_model =  keras.saving.load_model(modelFile, compile=True, safe_mode=True)
        self.keypoints_model.summary()
      
        #---------------------------------------------------------------------
        self.onehot_description = None
        self.token_threshold    = token_threshold
        self.input_size         = (self.cfg["inputWidth"],self.cfg["inputHeight"])
        self.frameNumber        = 0
        self.depthmap           = None
        self.activity           = None
        self.activityThreshold  = -17554720.0
        #---------------------------------------------------------------------
        self.skeletons          = None
        self.vocabulary         = None
        self.description        = None
        #---------------------------------------------------------------------
        if ("outputTokens" in self.cfg) and (self.cfg["outputTokens"]):
             self.vocabulary = read_json_files("%s/vocabulary.json" % modelPath)
             self.vocabulary["0"] = " "
             self.vocabulary["1"] = " " 
        #---------------------------------------------------------------------
        self.D = 300 #50 100 200 300
        self.maxtokens = 16
        self.token_threshold = 0.3
        #Don't search in all embeddings
        #glove_file = "datasets/GloVe/glove.6B/glove.6B.%ud.txt" % D  # 50-dimensional GloVe vectors
        #self.glove_embeddings = load_glove_embeddings(glove_file, embedding_dim=D)

        #Search only in possible embeddings
        self.possible_glove_embeddings = dict()
        with open("%s/embeddings_6B_D%u.json" % (modelPath,self.D), 'r') as json_file:
          self.possible_glove_embeddings = json.load(json_file)

    
    def process(self,frame, token_threshold=0.45, visualization=True):
            #print("Frame : ",frame.shape)
            self.input_image, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset = preprocess_image(frame, target_size=self.input_size)
            #print("input_image : ",self.input_image.shape)

            if (self.cfg['RGBImageEncoding']=='rgb8'):
                from imageProcessing import mix_rgb_to_8bit
                imageGivenToNN =  mix_rgb_to_8bit(self.input_image)
            elif (self.cfg['RGBImageEncoding']=='rgb16'):
                from imageProcessing import mix_rgb_to_16bit
                imageGivenToNN =  mix_rgb_to_16bit(self.input_image)
            else:
                imageGivenToNN = self.input_image
            
            if (visualization):
              #print("Target Size : ",self.input_size)
              image_bgr = cv2.cvtColor(imageGivenToNN, cv2.COLOR_BGR2RGB)
              cv2.imshow('NN Input Image', image_bgr) #OpenCV shows BGR
              cv2.waitKey(1)
            else:
              print("Token estimator received image : ",imageGivenToNN.shape)
              

            self.frameNumber = self.frameNumber + 1
            """
            self.glove = self.keypoints_model(np.expand_dims(imageGivenToNN, axis=0))
            print("Recovered Glove shape ",self.glove.shape)
            for i in range(7):
                word = vector_to_sentence_close(self.glove[0,i,1:], self.possible_glove_embeddings)
                print(i," - ",word)
            """




            #Glove Output
            #Offset and scaling extracted using : calculateGloveOffsets.py 
            #IMPORTANT : Make sure these are the same with debugGloveTokens

            #offset  = -0.1662235361903037
            #scaling =  0.27269877187819924

            #DeepSeek / D300 GLoVE
            offset  = 0.06446125477565857
            scaling =  0.35080948607416973

            """
            #Glove Output
            print("Caption : ")
            self.gloveList = self.keypoints_model(np.expand_dims(imageGivenToNN, axis=0))

            # Assuming the new model output is a single array of shape (batch_size, D * maxtokens)
            glove_output = self.gloveList[0]  # Extract the first batch

            # Reshape the flattened output back into (maxtokens, D)
            glove_output_reshaped = np.reshape(glove_output, (self.maxtokens, self.D))

            # Iterate over each token
            for i in range(self.maxtokens):
                # Extract the vector corresponding to the current token, undo scaling
                thisTokenVector = np.array(glove_output_reshaped[i, :], copy=True)  # Copy the array to make it writable
                thisTokenVector /= scaling
                thisTokenVector -= offset

                # Convert the vector to the closest word
                word = vector_to_sentence_close(thisTokenVector, self.possible_glove_embeddings, D=self.D)
                print(word[0], end=" ")

            print(" \n\n\n\n ")
            """



            start      = time.time()
            #Run actual model here
            allOutputs = self.keypoints_model.predict(np.expand_dims(imageGivenToNN, axis=0),verbose=0)
            seconds    = time.time() - start
            hz    = 1 / (seconds+0.0001)
            print(" %0.2f Hz "%hz,end="")
            

            if ('outputDescriptors' in self.cfg) and (self.cfg['outputDescriptors']):
               #print("All Outputs : ",len(allOutputs))
               #for outputID in range(len(allOutputs)):
               #   print("Output ",outputID," : ",len(allOutputs[outputID]))
               self.onehot    = allOutputs[2]
            else:
               self.onehot    = allOutputs[int(self.cfg['tokensOut'])]

            #print("One hot : ",self.onehot, " \n\n\n")
            #print("One hot length : ",len(self.onehot[0]), " \n\n\n")
            #print("Vocabulary : ",self.vocabulary, " \n\n\n")
  
            #One hot, single class output
            self.description = decodeOneHotDescriptionToString(self.vocabulary, self.onehot, token_threshold=self.token_threshold)
            print("Caption : ",self.description, " \n\n\n")
            return self.description


            """
            #Evaluate different outputs
            print("Caption : ")
            self.gloveList = self.keypoints_model(np.expand_dims(imageGivenToNN, axis=0))
            #print("Recovered Glove tokens ",len(self.gloveList))
            for i in range(len(self.gloveList)):
                #print(" Token ",i," shape ",len(self.gloveList[i].shape))

                #We undo scaling ..
                thisTokenVector = np.array(self.gloveList[i][0][:])
                thisTokenVector /= scaling
                thisTokenVector -= offset


                #np.set_printoptions(threshold=sys.maxsize) 
                #print("Token ",i," : ",thisTokenVector)

                word = vector_to_sentence_close(thisTokenVector, self.possible_glove_embeddings, D=self.D)
                #print(i," - ",word)
                print(word[0],end=" ")
            print(" \n\n\n\n ")
            """



            """
            # Make predictions using the model 
            self.everything = self.keypoints_model(np.expand_dims(imageGivenToNN, axis=0))
            #print("Everything : ",self.everything)
            #print("Everything Len : ",len(self.everything))
            self.glove = self.everything[0]
            self.tokens = self.everything[1]
 
            print("Glove shape ",self.glove.shape)
            for i in range(5):
                word = vector_to_sentence(self.glove[0,i,1:], self.glove_embeddings, embedding_dim=50, top_k=1)
                print(i," - ",word)

            #print("Glove : ",self.glove)
            #print("Tokens : ",self.tokens)
            top_16 = top_16_probabilities(self.tokens,self.tokenLabels)
            #print("Top 16 : ",top_16)
            # Output the top 16 with the correct scalar conversion
            for prob, desc in top_16:
                # Ensure prob is a scalar value
                if isinstance(prob, np.ndarray):
                    prob = prob.item()
                elif hasattr(prob, 'numpy'):
                    prob = prob.numpy().item()
                if (prob>threshold):
                  print(f"{desc}({prob:.2f}) ", end ="")
            print("\n\n\n")
            """


      


def main_token_estimation():
    model_path           = '2d_pose_estimation'
    videoFilePath        = "webcam" 
    videoWidth           = 1280
    videoHeight          = 720
    threshold            = 20
    cropInputFrame       = True
    customCrop           = False
    customCropX          = 0.0
    customCropY          = 0.0
    customCropSize       = 0.0
    scale                = 1.0
    # Run the webcam keypoints detection

    profiling = False
    visualize = True
    save      = False
    engine    = "tensorflow"
    if (len(sys.argv)>1):
       #print('Argument List:', str(sys.argv))
       for i in range(0, len(sys.argv)):

           if (sys.argv[i]=="--crop"):
              cropInputFrame = True
              customCrop = True
              customCropX          = int(sys.argv[i+1])
              customCropY          = int(sys.argv[i+2])
              customCropSize       = int(sys.argv[i+3])
           if (sys.argv[i]=="--nocrop"):
              cropInputFrame=False
              customCrop = False 
           if (sys.argv[i]=="--save"):
              save=True
           if (sys.argv[i]=="--from"):
              videoFilePath=sys.argv[i+1]
           if (sys.argv[i]=="--headless"):
              visualize = False
           if (sys.argv[i]=="--scale"):
              scale = float(sys.argv[i+1])
           if (sys.argv[i]=="--size"):
              videoWidth  = int(sys.argv[i+1])
              videoHeight = int(sys.argv[i+2])
           if (sys.argv[i]=="--profiling") or (sys.argv[i]=="--profile"):
               profiling = True
               print("Profiling Activated")
           if (sys.argv[i]=="--engine"):
               engine=sys.argv[i+1]
               print("Set Engine to ",engine)
    
    from runMAPYNet import getCaptureDeviceFromPath 
    cap = getCaptureDeviceFromPath(videoFilePath,videoWidth,videoHeight)
 
    estimator = TokenEstimator2D(modelPath=model_path, token_threshold=threshold, engine=engine, profiling=profiling)

    if (save):
        disable_screensaver()

    failedFrames = 0
    while True:
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        # Capture a single frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            failedFrames=failedFrames+1
            if (failedFrames>100):
               break
        else:
            failedFrames = 0

            if (scale!=1.0):
               original_height, original_width = frame.shape[:2]
               # Calculate new dimensions based on the scale
               new_width = int(original_width * scale)
               new_height = int(original_height * scale)
               # Resize the image using the new dimensions
               frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)



            # Preprocess the frame for the model if we have selected this behavior and the input image should be a rectangle
            if (cropInputFrame) and (estimator.cfg['inputWidth'] == estimator.cfg['inputHeight']):
              if (customCrop):
                  from runMAPYNet import custom_crop
                  frame = custom_crop(frame,customCropX,customCropY,customCropSize)
              else:
                  from runMAPYNet import extract_centered_rectangle
                  frame = extract_centered_rectangle(frame)

            #Extract results
            estimator.process(frame)

            #Visualize
            if (visualize):
              frameWithVis = frame.copy()
              #estimator.visualize(frameWithVis)
        
              # Check for the escape key or 'q' key press
              key = cv2.waitKey(1) & 0xFF 

              if key == 27 or key == ord('q') or key == ord('Q'):
                print("Terminating after receiving keyboard request")
                break

              if (save):
                screenshot(estimator.frameNumber)



        #-------------------------------------------------------------------------------------------------------------------------------------------------
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    if (save):
        enable_screensaver()
        os.system("ffmpeg -framerate 25 -i colorFrame_0_%%05d.png -vf scale=-1:720 -y -r 25 -pix_fmt yuv420p -threads 8 %s_lastRun3DHiRes.mp4" % videoFilePath)
        os.system("rm colorFrame*.png")
#----------------------------------------------------------------------------------------
if __name__ == '__main__': 
    main_token_estimation()

