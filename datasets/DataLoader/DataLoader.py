#!/usr/bin/python3
#-------------------------------------------------------------------------------
import ctypes
#-------------------------------------------------------------------------------
import os
import sys
import time
import json
import numpy as np
from ctypes import *
from os.path import exists
#-------------------------------------------------------------------------------
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
#-------------------------------------------------------------------------------
# Load C library
def loadLibrary(filename, relativePath="", forceUpdate=False):
    if (relativePath != ""):
        filename = relativePath + "/" + filename

    if (forceUpdate) or (not exists(filename)):
        print(bcolors.FAIL,"Could not find DataLoader Library (", filename, "), compiling a fresh one..!",bcolors.ENDC)
        print("Current directory was (", os.getcwd(), ") ")
        directory = os.path.dirname(os.path.abspath(filename))
        creationScript = directory + "/makeLibrary.sh"
        os.system(creationScript)

    if not exists(filename):
        directory = os.path.dirname(os.path.abspath(filename))
        print(bcolors.FAIL,"Could not make DataLoader Library, terminating",bcolors.ENDC)
        print("Directory we tried was : ", directory)
        sys.exit(0)

    libDataLoader = CDLL(filename, mode=ctypes.RTLD_GLOBAL)
    libDataLoader.connect()

    return libDataLoader
#-------------------------------------------------------------------------------
def checkIfAnyValuesOutsideOfRange(arr, minV, maxV):
    for sample in range(arr.shape[0]):
      for x in range(arr.shape[1]):
        for y in range(arr.shape[2]):
          for hm in range(arr.shape[3]):
               val = arr[sample][x][y][hm]
               if (val < minV) or (val > maxV):
                  print("Out of bounds sample",sample,"x",x,"y",y,"hm",hm)
                  print("Val : ",val)
                  return 1
    return 0
#---------------------------------------------------------------------------------------------
def get_blacklisted_keys(vocabulary, tokenblacklist):
    # Initialize an empty list to store the keys
    blacklisted_keys = []
    
    # Iterate over the items in the vocabulary
    for key, value in vocabulary.items():
        # If the value is in the blacklist, add the key to the list
        if value in tokenblacklist:
            blacklisted_keys.append(key)
    
    return blacklisted_keys
#---------------------------------------------------------------------------------------------
"""
def remove_blacklisted_tokens(vocabulary, tokenblacklist):
    # Create a new dictionary to store the filtered vocabulary
    filtered_vocabulary = {}
    
    # Iterate over the items in the original vocabulary
    for key, value in vocabulary.items():
        # If the value is not in the blacklist, add it to the filtered vocabulary
        if value not in tokenblacklist:
            filtered_vocabulary[key] = value
    
    return filtered_vocabulary
"""
#---------------------------------------------------------------------------------------------
def loadJSON(filename):
      vocabulary = list()
      with open(filename, 'r') as json_file:
        vocabulary = json.load(json_file)
      return vocabulary
#---------------------------------------------------------------------------------------------
def is_verb_simple(word):
    # List of typical verb suffixes (this is a simplification)
    verb_suffixes = ["ing", "ed", "en", "es", "s", "ize"]
    return any(word.endswith(suffix) for suffix in verb_suffixes)

# Function to filter only verbs using spaCy
#python3 -m pip install spacy
#python3 -m spacy download en_core_web_sm
def is_verb(nlp,word):
    print("Parsing : ",word)
    doc = nlp(word)
    print("Selecting verbs ")
    return any(token.pos_ == "VERB" for token in doc)
#---------------------------------------------------------------------------------------------
def tokens_to_one_hot(tokens, max_token_value, blacklist, active=1, inactive=0):
    """
    Converts a 1x16 numpy array of tokens to a one-hot encoded matrix.

    Parameters:
    tokens (numpy array): A 1x16 numpy array of tokens.
    max_token_value (int): The maximum token value.

    Returns:
    numpy array: A 16 x (max_token_value + 1) one-hot encoded matrix.
    """
    numberOfSamples = tokens.shape[0] 
    numberOfTokens  = tokens.shape[1] 
    numberOfValues  = max_token_value 
    # Initialize a matrix of zeros with shape (16, max_token_value + 1) 
    one_hot_encoded = np.full((numberOfSamples, numberOfTokens, max_token_value + 1), inactive, dtype=np.int8)

    # Set the corresponding index for each token to 1
    for sampleID in range(numberOfSamples):
     for tokenNumber, tokenValue in enumerate(tokens[sampleID]):
         #if not (str(token) in blacklist): #Blacklist now enforced in C code
            one_hot_encoded[sampleID, tokenNumber, tokenValue] = active
     #Always force 0 as false
     one_hot_encoded[sampleID,:,0] = inactive #forced
    
    return one_hot_encoded
#---------------------------------------------------------------------------------------------
def tokens_to_classes(tokens, max_token_value, blacklist, active=1, inactive=0):
    """
    Converts a 1x16 numpy array of tokens to a multi-hot encoded matrix.

    Parameters:
    tokens (numpy array): A 1x16 numpy array of tokens.
    max_token_value (int): The maximum token value.
    """

    numberOfSamples = tokens.shape[0]
    numberOfTokens  = tokens.shape[1]
    numberOfValues  = max_token_value 

    classes_encoded = np.full((numberOfSamples, max_token_value + 1), inactive, dtype=np.int8) # inactive , dtype=np.int8

    # Set the corresponding index for each token to 1
    for sampleID in range(numberOfSamples):
     for i, token in enumerate(tokens[sampleID]):
         #if not (str(token) in blacklist):  #Blacklist now enforced in C code
           classes_encoded[sampleID, token] = active #active
     #Always force 0 as false
     classes_encoded[sampleID, 0] = inactive #forced
    return classes_encoded
#---------------------------------------------------------------------------------------------
class DataLoader:
    def __init__(self,
                 inDims,
                 outDims,
                 output16BitChannels:int = 0,
                 numberOfSamples: int = 0, #0 means automatically use whole dataset
                 numberOfThreads: int = 4, #Most CPUs nowadays have 4 cores/threads
                 streamData:int = 0, #By default dont stream data load it all at once
                 batchSize:int = 32, #This is only important when streaming data
                 gradientSize: int = 12,
                 PAFSize:int = 5,
                 doAugmentations: int = 1, #1 means we want to do augmentations
                 addPAFs: int = 1, 
                 addBackground: int = 1, 
                 addDepthMap: int = 1,  
                 addDepthLevelsHeatmaps: int = 0,  
                 addNormals: int = 1,
                 addSegmentation: int = 1,
                 bytesPerDepthValue:int = 2,
                 ignoreNoSkeletonSamples: int = 0, #0 means we want all samples
                 datasets = [["cocoTrain.db","../coco/cache/coco/train2017","../coco/cache/coco/depth_train2017", "../coco/cache/coco/segment_train2017"]],
                 vocabularyPath = "2d_pose_estimation/vocabulary.json",
                 elevatePriority = False,
                 libraryPath: str  = "./libDataLoader.so", 
                 forceLibUpdate=False):
        #Tokens 
        #---------------------------------------
        self.vocabulary         = loadJSON(vocabularyPath)
        self.tokenblacklist     = ["(", ")", ",", ".", "a", "an", 's', 'hu', 'hy',  'w']

        stopwords = [ "of", "on", "and", "I", "in", "the", "is", "it", "at", "to", "with", "for", "from", "near", "while"]
        self.tokenblacklist.extend(stopwords)

        overrepresented = ['top','next','two', 'are', 'it', 'its', 'up', 'down', 'left', 'right', 'in', 'out', 'front', 'to', 'has', 'by']
        self.tokenblacklist.extend(overrepresented)

        difficult = ['nintendo', 'wii',  'umpire']
        self.tokenblacklist.extend(difficult)

        # Load the spaCy English model
        #import spacy
        #print("Loading spaCy")
        #nlp = spacy.load("en_core_web_sm")
        #verbs = [word for word in self.vocabulary.values() if is_verb(nlp,word)]
        
        verbs = ['abandoned', 'adjusting', 'advertising', 'alcoholic', 'allowed', 'arm', 'arranged', 'arrive', 'assorted', 'attached', 'bags', 'baked', 'bakery', 'baking', 'barn', 'bathing', 'batter', 'batting', 'beach', 'beak', 'bear', 'believing', 'benches', 'bending', 'bicycles', 'billowing', 'biplane', 'biscuit', 'biting', 'blanket', 'blender', 'block', 'blooming', 'blow', 'blowing', 'bookcase', 'bookstore', 'booze', 'bottle', 'boulders', 'bow', 'broken', 'brushing', 'bubbles', 'buffalo', 'building', 'bun', 'bundt', 'burning', 'camping', 'capped', 'carpeted', 'carriage', 'carrying', 'catch', 'catching', 'cauliflower', 'cave', 'celebrating', 'cellar', 'chained', 'chasing', 'checkered', 'checking', 'chewing', 'choir', 'chopped', 'clad', 'claim', 'cleaning', 'climbing', 'close', 'closed', 'colored', 'combing', 'coming', 'competing', 'connected', 'consisting', 'containing', 'controls', 'conveyor', 'cooked', 'cooking', 'cooling', 'counter', 'cover', 'covered', 'cross', 'crossed', 'crossing', 'crowd', 'crowded', 'curtains', 'cut', 'cute', 'cutting', 'dance', 'dancing', 'decorated', 'decorating', 'decorative', 'depicting', 'desserts', 'digging', 'dining', 'dipping', 'directing', 'displayed', 'do', 'dock', 'docked', 'doing', 'dolphin', 'double', 'doubles', 'drawing', 'drawn', 'dressed', 'dressing', 'dribbling', 'dried', 'drink', 'drinking', 'driven', 'driving', 'drum', 'drying', 'duckling', 'dump', 'eagle', 'eat', 'eaten', 'eating', 'exhibit', 'expired', 'face', 'fallen', 'falling', 'fashioned', 'fast', 'feeding', 'feeds', 'fenced', 'fighting', 'filled', 'fishing', 'fix', 'fixing', 'flavored', 'floating', 'flooded', 'flooring', 'flowing', 'flown', 'fly', 'flying', 'forested', 'fried', 'frosted', 'frosting', 'fry', 'frying', 'galloping', 'gathered', 'gathering', 'gear', 'get', 'getting', 'glassware', 'glazed', 'glove', 'glowing', 'go', 'going', 'goose', 'grab', 'grasses', 'grazing', 'grinding', 'growing', 'hand', 'handle', 'handwritten', 'hanging', 'has', 'hate', 'having', 'headphones', 'held', 'helmet', 'helping', 'herded', 'herding', 'hiking', 'hit', 'hitting', 'hold', 'holding', 'holds', 'hooded', 'hooked', 'horned', 'horseback', 'hugging', 'icing', 'includes', 'including', 'jeep', 'jump', 'jumping', 'kick', 'kicking', 'kind', 'kissing', 'kiteboarding', 'kites', 'kitten', 'kneeling', 'knick', 'laid', 'landing', 'laying', 'leading', 'leads', 'leaf', 'leafless', 'leaning', 'leash', 'leaves', 'left', 'lettering', 'lettuce', 'lift', 'lighting', 'lined', 'link', 'lit', 'living', 'loaded', 'loading', 'lobby', 'lodge', 'log', 'look', 'looking', 'looks', 'lounging', 'lush', 'lying', 'made', 'make', 'making', 'marching', 'match', 'measuring', 'merry', 'microwave', 'microwaves', 'mid', 'milking', 'miss', 'mixed', 'mixing', 'monitor', 'moped', 'motes', 'motorized', 'mound', 'mounted', 'mouse', 'nightstands', 'note', 'nurses', 'objects', 'open', 'opening', 'ornate', 'outfit', 'outfits', 'outstretched', 'oven', 'overlooking', 'pack', 'packed', 'paddling', 'paintbrush', 'painted', 'painting', 'pair', 'pairs', 'pan', 'paneling', 'parachute', 'parasail', 'parasailing', 'parked', 'passes', 'passing', 'patch', 'patterned', 'pay', 'pears', 'peeled', 'pepperoni', 'perched', 'perform', 'performing', 'petting', 'photograph', 'pick', 'piled', 'pillows', 'pitching', 'play', 'played', 'playground', 'playing', 'plays', 'podium', 'pointing', 'pose', 'poses', 'posing', 'poured', 'pouring', 'practicing', 'prepared', 'preparing', 'produce', 'propped', 'pull', 'pulled', 'pulling', 'pump', 'pushing', 'putting', 'racing', 'rack', 'racquet', 'radishes', 'railing', 'rainbow', 'ram', 'reading', 'reads', 'rear', 'reflected', 'reflecting', 'relish', 'remote', 'remove', 'ridden', 'ride', 'rides', 'riding', 'rink', 'ripe', 'roast', 'rocking', 'roll', 'roofed', 'rose', 'row', 'rowing', 'rug', 'run', 'running', 'saddle', 'sail', 'sailing', 'salad', 'salads', 'sand', 'sandbags', 'sauce', 'says', 'seagulls', 'seen', 'selling', 'served', 'set', 'setting', 'sewing', 'shaking', 'shaped', 'sheared', 'shining', 'shorts', 'shot', 'shoveling', 'show', 'shower', 'showing', 'shown', 'sign', 'singing', 'sink', 'sit', 'sits', 'sitting', 'skate', 'skateboarding', 'skating', 'ski', 'skiing', 'skirt', 'sled', 'sleeping', 'slice', 'sliced', 'sliding', 'smile', 'smiling', 'smoke', 'smoking', 'smoothie', 'sniffing', 'snowboarding', 'snowmobile', 'soaked', 'soda', 'sold', 'spanning', 'speaking', 'spewing', 'split', 'spoon', 'spooning', 'spraying', 'sprinkles', 'squatting', 'stacked', 'stadium', 'stained', 'staircase', 'stand', 'standing', 'stands', 'staring', 'steam', 'steering', 'stem', 'stew', 'stick', 'sticking', 'stir', 'stirred', 'stirring', 'stop', 'stopped', 'store', 'stove', 'strawberry', 'stream', 'stripe', 'striped', 'stuffed', 'surfer', 'surfing', 'surrounded', 'swim', 'swimming', 'swims', 'swimsuit', 'swimsuits', 'swinging', 'syrup', 'tag', 'take', 'taken', 'taking', 'talk', 'talking', 'tasting', 'teaching', 'tending', 'throw', 'throwing', 'tie', 'tied', 'tiled', 'toaster', 'tomatoes', 'tongs', 'tooth', 'toothbrush', 'topped', 'tops', 'tossing', 'touch', 'touching', 'tour', 'tow', 'towel', 'towering', 'track', 'train', 'traveling', 'trekking', 'trick', 'trying', 'tying', 'typing', 'uniformed', 'unmade', 'use', 'used', 'using', 'vandalized', 'vending', 'waiting', 'wake', 'walk', 'walking', 'walled', 'washing', 'watch', 'watched', 'watches', 'watching', 'watering', 'wearing', 'wetsuit', 'wheelbarrow', 'wheeled', 'whipped', 'windsurfing', 'wooded', 'worked', 'working', 'worn', 'writing', 'yard']
        #self.tokenblacklist.extend(verbs)

        #print("TEST: DISABLE BLACKLIST")
        #self.tokenblacklist = list() 

        self.tokenblacklistkeys = get_blacklisted_keys(self.vocabulary, self.tokenblacklist)
        print("Token Black list : ",self.tokenblacklist)
        print("Token Black list keys : ",self.tokenblacklistkeys)
        print("Token Black list number of values : ",len(self.tokenblacklistkeys))
        #This should not be done because it corrupts the order of tokens! 
        #vocabularyWithoutBlacklist         = remove_blacklisted_tokens(self.vocabulary , self.tokenblacklist)
        #print("Token remaining number of values : ",len(vocabularyWithoutBlacklist))
        #---------------------------------------
        self.numberOfThreads         = numberOfThreads
        self.initialGradientSize     = gradientSize
        self.doAugmentations         = doAugmentations
        self.ignoreNoSkeletonSamples = ignoreNoSkeletonSamples
        self.gradientSize    = gradientSize
        self.PAFSize         = PAFSize
        self.streamData      = streamData
        self.batchSize       = batchSize
        #---------------------------------------
        self.inWidth         = inDims[0]
        self.inHeight        = inDims[1]
        self.inChannels      = inDims[2]
        #---------------------------------------
        self.outWidth        = outDims[0]
        self.outHeight       = outDims[1]
        self.out8BitChannels = outDims[2]
        self.output16BitChannels = output16BitChannels
        #---------------------------------------
        self.lastStartSample = 0
        self.lastEndSample   = 0
        self.bytesPerDepthValue = bytesPerDepthValue
        self.cpuTimeSeconds  = 0

        #TODO: Maybe make this dynamically update-able
        self.D = 300 #<- Expected dimensions of each GloVE Vector

        self.db = None
        self.libDataLoader = loadLibrary(libraryPath, forceUpdate=forceLibUpdate)

        if (elevatePriority):
          self.setPriority(-20)



        #This is new handling for datasets sources to make enabling/disabling datasets easier 
        #-----------------------------------------------------------------------------
        if (len(datasets)<1):
               raise ValueError("Please give some datasets to load..  ")

        datasetsEnabled=list() 
        for sourceEntry in datasets:
              if len(sourceEntry)==5:
                  #new entry with use/dontuse first element
                  confStr = sourceEntry[0].lower()
                  if (confStr=="use" or confStr=="enable" or confStr=="enabled"  or confStr=="1" or confStr=="true"):
                     datasetsEnabled.append(sourceEntry[1:])
              elif len(sourceEntry)==4:
                  #Regular old style entry :)
                  datasetsEnabled.append(sourceEntry)
              else:
                  print("Incorrectly formatted dataset entry with ",len(sourceEntry)," elements :")
                  print(sourceEntry)

        if (len(datasetsEnabled)<1):
               raise ValueError("Please give some ENABLED datasets to load..  ")

        self.dbListPtr = self.createSourceList(len(datasetsEnabled))
        sourceID = 0 
        for sourceEntry in datasetsEnabled:
            self.addToSourceList(self.dbListPtr,sourceID,sourceEntry[0],sourceEntry[1],sourceEntry[2],sourceEntry[3],self.ignoreNoSkeletonSamples) 
            sourceID = sourceID + 1
        #-----------------------------------------------------------------------------  


        self.libDataLoader.db_create.argtypes = [ctypes.c_void_p,   #struct DatabaseList* dbSources
                                                 ctypes.c_ulong,    #unsigned long numberOfSamples
                                                 ctypes.c_int,      #int streamData
                                                 ctypes.c_int,      #int batchSize
                                                 ctypes.c_int,      #int workerThreads
                                                 ctypes.c_int,      #int gradientSize
                                                 ctypes.c_int,      #int PAFSize
                                                 ctypes.c_int,      #int doAugmentations
                                                 ctypes.c_int,      #int addPAFs
                                                 ctypes.c_int,      #int addBackground
                                                 ctypes.c_int,      #int addDepthMap
                                                 ctypes.c_int,      #int addDepthLevelsHeatmaps
                                                 ctypes.c_int,      #int addNormals
                                                 ctypes.c_int,      #int addSegmentation
                                                 ctypes.c_int,      #int bytesPerDepthValue
                                                 ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, #unsigned int widthIn,  unsigned int heightIn,unsigned int channelsIn,
                                                 ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint] #unsigned int widthOut, unsigned int heightOut,unsigned int channelsOut8Bit,unsigned int output16BitChannels
        self.libDataLoader.db_create.restype  = ctypes.c_void_p

        self.db = self.libDataLoader.db_create(self.dbListPtr,
                                               numberOfSamples, streamData, batchSize, self.numberOfThreads, self.gradientSize, self.PAFSize, doAugmentations, 
                                               addPAFs,addBackground,addDepthMap,addDepthLevelsHeatmaps,addNormals,addSegmentation,bytesPerDepthValue,
                                               self.inWidth,  self.inHeight, self.inChannels, 
                                               self.outWidth, self.outHeight, self.out8BitChannels, self.output16BitChannels)  # Create database
        if not self.db:
            print(bcolors.FAIL,"Failed to create database",bcolors.ENDC)
            raise ValueError("Failed to create database")
            return

        numberOfTokensFromDictionary = len(list(self.vocabulary.keys()))
        print("Number of tokens from dictionary : ",numberOfTokensFromDictionary)
        self.set_max_token_value(numberOfTokensFromDictionary) #This is important to enforce a standard number of tokens and not be based on auto-detection

        #self.sortTokens()                #<- sort tokens
        self.sortTokensBasedOnFrequency(ascendingOrder=0) #<- Sort tokens based on their frequency

        self.removeDuplicateTokens() # <- Remove duplicate tokens

        self.update_token_blacklist(self.tokenblacklistkeys,lowThreshold=0)


        self.libDataLoader.db_get_number_of_samples.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_get_number_of_samples.restype  = ctypes.c_ulong
        self.numberOfSamples = self.libDataLoader.db_get_number_of_samples(self.db)
        print(bcolors.OKGREEN,"Created a database with ",self.numberOfSamples," samples ",bcolors.ENDC)
        #self.test()
 

    def test(self):
        self.libDataLoader.test.argtypes = [ctypes.c_int,ctypes.c_void_p]
        self.libDataLoader.test.restype  = ctypes.c_int
        res = self.libDataLoader.test(0,0)
        return res


    def get_labels(self):
        #TODO: integrate this in coco.db / C code
        jointNames = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle" ]
        return jointNames

    def disableHeatmapOutput(self):
        self.libDataLoader.db_disable_heatmap_output.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_disable_heatmap_output(self.db)

    def createSourceList(self,numberOfSources):
        self.libDataLoader.db_allocate_source_list.argtypes = [ctypes.c_uint]
        self.libDataLoader.db_allocate_source_list.restype  = ctypes.c_void_p

        print("Initializing ",numberOfSources," sources in C code")
        return self.libDataLoader.db_allocate_source_list(numberOfSources)

    def destroySourceList(self,sourceDB):
        self.libDataLoader.db_destroy_source_list.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_destroy_source_list(sourceDB)

    def addToSourceList(self,sourceDB,sourceID,pathToDB,pathToImages,pathToDepth,pathToSegmentation,ignoreNoSkeletonSamples):
        path1 = pathToDB.encode('utf-8')  
        path2 = pathToImages.encode('utf-8')  
        path3 = pathToDepth.encode('utf-8')  
        path4 = pathToSegmentation.encode('utf-8')  
        self.libDataLoader.db_set_source_entry.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self.libDataLoader.db_set_source_entry.restype  = ctypes.c_void_p
        return self.libDataLoader.db_set_source_entry(sourceDB,sourceID,path1,path2,path3,path4,ignoreNoSkeletonSamples)

    def get_number_of_samples(self,db):
        #print("get_number_of_samples()")
        self.libDataLoader.db_get_number_of_samples.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_get_number_of_samples.restype  = ctypes.c_ulong
        return self.libDataLoader.db_get_number_of_samples(db)

    def get_number_of_images(self,db):
        #print("get_number_of_images()")
        self.libDataLoader.db_get_number_of_images.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_get_number_of_images.restype  = ctypes.c_ulong
        return self.libDataLoader.db_get_number_of_images(db)

    def get_total_loss_of_sample(self,sample):
        self.libDataLoader.db_get_sample_total_loss.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.libDataLoader.db_get_sample_total_loss.restype  = ctypes.c_float
        return self.libDataLoader.db_get_sample_total_loss(self.db,sample)

    def get_train_passes_of_sample(self,sample):
        self.libDataLoader.db_get_sample_train_passes.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.libDataLoader.db_get_sample_train_passes.restype  = ctypes.c_ulong
        return self.libDataLoader.db_get_sample_train_passes(self.db,sample)

    def get_filename_of_sample(self,sample):
        buffer_size = 100
        buffer = ctypes.create_string_buffer(buffer_size)
        self.libDataLoader.db_get_filename_of_sample.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.libDataLoader.db_get_filename_of_sample.restype  = ctypes.c_ulong
        if ( self.libDataLoader.db_get_filename_of_sample(self.db,sample,buffer,buffer_size) ):
           return buffer.value.decode('utf-8')
        return ""

    def dump_sample_report(self,path="sample_report.json"):
        print("Writing ",path)
        results = dict()
        for sID in range(self.numberOfSamples):
              if (self.get_total_loss_of_sample(sID)!=0) and (self.get_train_passes_of_sample(sID)!=0):
                 results[sID] = [ sID, self.get_filename_of_sample(sID) , self.get_total_loss_of_sample(sID)  / ( self.get_train_passes_of_sample(sID) + 0.0001 ) ]

        sorted_results = sorted(results.values(), key=lambda x: x[2]) 

        import json
        with open(path, 'w') as fp:
            json.dump(sorted_results, fp)
        print("Done writing ",path)

    def update(self,startSample,endSample):
        self.lastStartSample = startSample
        self.lastEndSample   = endSample
        #print("update(",startSample," , ",endSample,")")
        self.libDataLoader.db_update.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_int]
        self.libDataLoader.db_update.restype  = ctypes.c_int
        return self.libDataLoader.db_update(self.db, startSample, endSample, self.numberOfThreads, self.gradientSize, self.PAFSize)

    def startUpdate(self,startSample,endSample):
        self.lastStartSample = startSample
        self.lastEndSample   = endSample
        self.libDataLoader.db_StartUpdate.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_int]
        self.libDataLoader.db_StartUpdate.restype  = ctypes.c_int
        return self.libDataLoader.db_StartUpdate(self.db, startSample, endSample, self.numberOfThreads, self.gradientSize, self.PAFSize)

    def collectUpdate(self,startSample,endSample):
        self.lastStartSample = startSample
        self.lastEndSample   = endSample
        self.libDataLoader.db_CollectUpdate.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_int]
        self.libDataLoader.db_CollectUpdate.restype  = ctypes.c_int
        return self.libDataLoader.db_CollectUpdate(self.db, startSample, endSample, self.numberOfThreads, self.gradientSize, self.PAFSize)


    def setPriority(self,newPriority):
        self.libDataLoader.db_set_priority.argtypes = [ctypes.c_int]
        self.libDataLoader.db_set_priority.restype  = ctypes.c_int
        return self.libDataLoader.db_set_priority(newPriority)

    def printReadSpeed(self):
        self.libDataLoader.db_print_readSpeed.argtypes = [ctypes.c_void_p] 
        return self.libDataLoader.db_print_readSpeed(self.db)

    def get_partial_update_IO_array(self, startSample=0, endSample=0, produce16BitData=True):
        if (not self.streamData):
           raise ValueError('Getting streaming output access while not streaming will not work!')

        startTime = time.time()

        #print("get_partial_update_IO_array(",startSample," , ",endSample,")")
        #if True: #<- Uncomment to test what happens assuming perfect zero time for getting a new
        if ( self.update(startSample,endSample) ): #<- regular one stage update..!
        #if ( self.collectUpdate(startSample,endSample) ): #<- multi stage update / needs .copy() / needs startUpdate in outside scope works worse :(
           self.libDataLoader.db_get_in.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
           self.libDataLoader.db_get_in.restype  = POINTER(ctypes.c_ubyte)
           pixelsIn  = self.libDataLoader.db_get_in(self.db,0) #We always want to start at the first element
           npArrayIn = np.ctypeslib.as_array(pixelsIn, shape=(endSample-startSample, self.inHeight, self.inWidth, self.inChannels) ).copy()  #<- Copy should copy ~1MB not worth the thread safe risk to not copy

           self.libDataLoader.db_get_out.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
           self.libDataLoader.db_get_out.restype  = POINTER(ctypes.c_byte)
           pixelsOut  = self.libDataLoader.db_get_out(self.db,0) #We always want to start at the first element
           npArrayOut = np.ctypeslib.as_array(pixelsOut, shape=(endSample-startSample, self.outHeight, self.outWidth, self.out8BitChannels) ).copy()  #<- Copy should copy ~3MB not worth the thread safe risk to not copy
 
           #Uncommenting the following printf statements should return :
           #Heatmaps I/O Types  <class 'numpy.ndarray'> / <class 'numpy.ndarray'>
           #print("Heatmaps I/O Types ",type(npArrayIn),"/",type(npArrayOut))
           # Heatmaps I/O DTypes  uint8 / int8
           #print("Heatmaps I/O DTypes ",npArrayIn.dtype,"/",npArrayOut.dtype)
  
           #16-Bit output DEACTIVATED
           npArray16BitOut = None
 
           if (produce16BitData):
            if (self.output16BitChannels>0):
              numberOfImages = self.get_number_of_images(self.db)
              self.libDataLoader.db_get_out16bit.argtypes = [ctypes.c_void_p,ctypes.c_ulong,ctypes.c_ulong]
              self.libDataLoader.db_get_out16bit.restype  = POINTER(ctypes.c_short)  
              pixels16BitOut  = self.libDataLoader.db_get_out16bit(self.db,0,self.batchSize)

              npArray16BitOut =  np.ctypeslib.as_array(pixels16BitOut, shape=(numberOfImages, self.outHeight, self.outWidth, self.output16BitChannels)).astype(np.int16) #Just The Raw 16-bit value [-32767 .. 32767]
              #npArray16BitOut =  npArray16BitOut.copy().astype(np.float32) * (120.0 / 32767.0) #Convert them to a float with the same range as [ -120.0 ... 120.0 ]
              #npArray16BitOut =  np.round(npArray16BitOut, decimals=1) #try rounding to see if more quantized values are easier

              #print("Reducing 8bit output from ",npArrayOut.shape)
              #npArrayOut      = npArrayOut[:, :, :, : (-2 * self.output16BitChannels)] #Ommit last 2 8bit parts of the 16bit heatmap
              #print("Reducing 16bit output from ",npArray16BitOut.shape) 

           """
           #Uncomment to take a look in the outputs..
           min8bit  = np.min(npArrayOut)
           max8bit  = np.max(npArrayOut)
           mean8bit = np.mean(npArrayOut)
           std8bit  = np.std(npArrayOut)
           var8bit  = np.std(npArrayOut)
           #----------------------------------
           min16bit  = np.min(npArray16BitOut)
           max16bit  = np.max(npArray16BitOut)
           mean16bit = np.mean(npArray16BitOut)
           std16bit  = np.std(npArray16BitOut)
           var16bit  = np.std(npArray16BitOut)
           #----------------------------------
           print("8Bit  - Min %0.2f - Max %0.2f - Mean %0.2f - StD %0.2f - Var %0.2f " % (min8bit,max8bit,mean8bit,std8bit,var8bit))
           print("16Bit - Min %0.2f - Max %0.2f - Mean %0.2f - StD %0.2f - Var %0.2f " % (min16bit,max16bit,mean16bit,std16bit,var16bit))
           """           
           self.cpuTimeSeconds = self.cpuTimeSeconds + (time.time() - startTime)
           return  npArrayIn, npArrayOut, npArray16BitOut#.astype(np.int8)

        print("Could not perform DB In update in range ",startSample,endSample)
        raise ValueError('Could not perform DB update')
        return None, None

    def sortTokens(self):
        self.libDataLoader.db_sort_description_tokens.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_sort_description_tokens(self.db)

    def sortTokensBasedOnFrequency(self,ascendingOrder=1):
        MAX_TOKEN_VALUE = self.get_max_token_value()
        if (MAX_TOKEN_VALUE==0):
            raise ValueError("MAX_TOKEN_VALUE is zero!")

        self.libDataLoader.db_sort_description_tokens_based_on_count.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
        self.libDataLoader.db_sort_description_tokens_based_on_count(self.db,MAX_TOKEN_VALUE,int(ascendingOrder))

    def removeDuplicateTokens(self): 
        self.libDataLoader.db_remove_duplicate_description_tokens.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_remove_duplicate_description_tokens(self.db)
 
    def set_max_token_value(self,newValue):
        self.libDataLoader.db_set_MAX_sample_description_token_value.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.libDataLoader.db_set_MAX_sample_description_token_value.restype  = ctypes.c_int
        return  self.libDataLoader.db_set_MAX_sample_description_token_value(self.db,newValue)


    def get_valid_segmentations(self):
        self.libDataLoader.db_get_valid_segmentations.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_get_valid_segmentations.restype  = ctypes.c_int
        VALID_SEGMENTATIONS = self.libDataLoader.db_get_valid_segmentations(self.db)
        return VALID_SEGMENTATIONS

    def get_total_segmentation_classes(self):
        self.libDataLoader.db_get_total_segmentation_classes.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_get_total_segmentation_classes.restype  = ctypes.c_int
        TOTAL_SEGMENTATION_CLASSES = self.libDataLoader.db_get_total_segmentation_classes(self.db)
        return TOTAL_SEGMENTATION_CLASSES


    def get_max_token_value(self):
        self.libDataLoader.db_get_MAX_sample_description_token_value.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_get_MAX_sample_description_token_value.restype  = ctypes.c_int
        MAX_TOKEN_VALUE = self.libDataLoader.db_get_MAX_sample_description_token_value(self.db)
        return MAX_TOKEN_VALUE

    def get_token_number(self):
        self.libDataLoader.db_get_MAX_sample_description_tokens_number.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_get_MAX_sample_description_tokens_number.restype  = ctypes.c_int
        MAX_TOKEN_NUMBER = self.libDataLoader.db_get_MAX_sample_description_tokens_number(self.db)
        return MAX_TOKEN_NUMBER


    def get_descriptor_number_of_elements(self):
        self.libDataLoader.db_get_descriptor_elements_number.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_get_descriptor_elements_number.restype  = ctypes.c_int
        NUMBER_OF_DESCRIPTOR_ELEMENTS = self.libDataLoader.db_get_descriptor_elements_number(self.db)
        return NUMBER_OF_DESCRIPTOR_ELEMENTS

    def update_token_blacklist(self, blacklisttokenIDs, lowThreshold=0):

        if (lowThreshold>0):
           #We have a lower threshold..
           tokenIDsToAddToBlackList = self.get_low_count_tokens(threshold=lowThreshold)
           blacklisttokenIDs.extend(tokenIDsToAddToBlackList)
           print("Overriding blacklist keys to also add those with a count < ",lowThreshold)
           self.tokenblacklistkeys = blacklisttokenIDs

        self.libDataLoader.db_allocate_token_blacklist.argtypes = [ctypes.c_void_p,ctypes.c_uint]
        self.libDataLoader.db_allocate_token_blacklist.restype  = ctypes.c_int
        response = self.libDataLoader.db_allocate_token_blacklist(self.db,len(blacklisttokenIDs))

        if (response):
          self.libDataLoader.db_add_token_to_blacklist.argtypes = [ctypes.c_void_p,ctypes.c_uint]
          self.libDataLoader.db_add_token_to_blacklist.restype  = ctypes.c_int

          for tID in blacklisttokenIDs:
           response = self.libDataLoader.db_add_token_to_blacklist(self.db,int(tID))

        self.libDataLoader.db_compile_added_token_blacklist.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_compile_added_token_blacklist.restype  = ctypes.c_int
        response = self.libDataLoader.db_compile_added_token_blacklist(self.db)

        return response


    def get_low_count_tokens(self,threshold = 10):
        MAX_TOKEN_VALUE = self.get_max_token_value()
        print("MAX_TOKEN_VALUE ",MAX_TOKEN_VALUE)

        self.libDataLoader.db_count_description_tokens.argtypes = [ctypes.c_void_p,ctypes.c_int]
        self.libDataLoader.db_count_description_tokens.restype  = POINTER(ctypes.c_ulong)
        tokenCount = self.libDataLoader.db_count_description_tokens(self.db,MAX_TOKEN_VALUE)

        #Bring tokens from a C array to a numpy list
        tokenCountNP = []
        for tokenID in range(MAX_TOKEN_VALUE):

             thisVal = int(tokenCount[tokenID])
             if (thisVal<threshold): 
               tokenCountNP.append(tokenID)

        print("Freeing memory in C side.. ")
        self.libDataLoader.db_free_description_token_count.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_free_description_token_count.restype  = ctypes.c_int
        self.libDataLoader.db_free_description_token_count(tokenCount)

        print("Tokens with counts lower than ",threshold,".. ")
        tokenDescriptions = list(self.vocabulary.values())
        for tokenID in tokenCountNP:
          print(tokenDescriptions[tokenID], end=" ")

        return tokenCountNP

    def get_token_fequencies(self):
        MAX_TOKEN_VALUE = self.get_max_token_value()
        print("MAX_TOKEN_VALUE ",MAX_TOKEN_VALUE)
        

        print("Printing token frequency summary.. ")
        tokenDescriptions = list(self.vocabulary.values())

        if (len(tokenDescriptions)!=MAX_TOKEN_VALUE):
          print(bcolors.FAIL,"There are ",len(tokenDescriptions)," token descriptions ")
          print("and MAX_TOKEN_VALUE = ",MAX_TOKEN_VALUE,bcolors.ENDC)
          print(bcolors.FAIL, "Setting MAX_TOKEN_VALUE = ",len(tokenDescriptions)," as a workaround, but need to fix this",MAX_TOKEN_VALUE,bcolors.ENDC)
          self.set_max_token_value(len(tokenDescriptions))
          MAX_TOKEN_VALUE = len(tokenDescriptions)

        self.libDataLoader.db_count_description_token_weight.argtypes = [ctypes.c_void_p,ctypes.c_int]
        self.libDataLoader.db_count_description_token_weight.restype  = POINTER(ctypes.c_float)
        tokenFrequency = self.libDataLoader.db_count_description_token_weight(self.db,MAX_TOKEN_VALUE)

        #Bring tokens from a C array to a numpy list
        tokenFrequencyNP = np.full((MAX_TOKEN_VALUE+1),0.0,dtype=np.float32)
        #tokenFrequencyNP = []
        #tokenFrequencyNP.append(0.0) #<- The first zero
        for tokenID in range(MAX_TOKEN_VALUE):
             tokenFrequencyNP[tokenID+1] = (float(tokenFrequency[tokenID]))

        print("Freeing memory in C side.. ")
        self.libDataLoader.db_free_description_token_count.argtypes = [ctypes.c_void_p]
        self.libDataLoader.db_free_description_token_count.restype  = ctypes.c_int
        self.libDataLoader.db_free_description_token_count(tokenFrequency)
 
        print("Reducing weights on black listed items.. ")
        for tokenID in self.tokenblacklistkeys:
             tokenFrequencyNP[int(tokenID)] = 1.0


        howManyValuesToIterateOver = len(tokenDescriptions)
        for tokenID in range(howManyValuesToIterateOver):
          thisDescription = tokenDescriptions[tokenID] 
          thisFrequency   = tokenFrequencyNP[tokenID]
          
          """
          print(thisDescription,"(%0.2f)" % thisFrequency, end=" ")
          if (tokenID%8==0):
            print("") #Add new line every 8 printed tokens for visualization purposes..
          """

          if (tokenFrequencyNP[tokenID]<0.0):
              print("Negative value for token ",tokenDescriptions[tokenID])
              sys.exit(1)
        print("")

        return tokenFrequencyNP


    def get_partial_descriptor_array(self, startSample=0, endSample=0):
        numberOfSamples  = endSample-startSample
        NUMBER_OF_DESCRIPTOR_ELEMENTS  = self.get_descriptor_number_of_elements()

        #print(" NUMBER_OF_DESCRIPTOR_ELEMENTS ", NUMBER_OF_DESCRIPTOR_ELEMENTS, "    " )
        descriptors = np.zeros((numberOfSamples,NUMBER_OF_DESCRIPTOR_ELEMENTS), dtype=np.float32) 

        self.libDataLoader.db_get_sample_descriptors.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
        self.libDataLoader.db_get_sample_descriptors.restype  = POINTER(ctypes.c_float)        
        for sID in range(numberOfSamples):
            thisDescriptorList = self.libDataLoader.db_get_sample_descriptors(self.db,startSample+sID)
            for descriptorID in range(NUMBER_OF_DESCRIPTOR_ELEMENTS):
                descriptors[sID,descriptorID] = thisDescriptorList[descriptorID]

        return descriptors


    def get_partial_token_array(self, startSample=0, endSample=0, encodeAsSingleMultiLabelToken=True):
        #if (not self.streamData):
        #   raise ValueError('Getting streaming output access while not streaming will not work!')

        numberOfSamples  = endSample-startSample
        MAX_TOKEN_VALUE  = self.get_max_token_value()
        MAX_TOKEN_NUMBER = self.get_token_number()

        #print(" MAX_TOKEN_VALUE ", MAX_TOKEN_VALUE, "    " )
        #print(" MAX_TOKEN_NUMBER ", MAX_TOKEN_NUMBER, "    " )
        tokens = np.zeros((numberOfSamples,MAX_TOKEN_NUMBER), dtype=np.uint16) #Tokens are 0..2048 so are encoded as ushort 16bit
        self.libDataLoader.db_get_sample_description_tokens.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
        self.libDataLoader.db_get_sample_description_tokens.restype  = POINTER(ctypes.c_ushort)
        
        for sID in range(numberOfSamples):
            thisTokenList = self.libDataLoader.db_get_sample_description_tokens(self.db,startSample+sID)
            for tokenID in range(MAX_TOKEN_NUMBER):
                tokens[sID,tokenID] = thisTokenList[tokenID]


        if (encodeAsSingleMultiLabelToken):
          #Encode everything as one 
          tokenOutput = tokens_to_classes(tokens, MAX_TOKEN_VALUE, self.tokenblacklistkeys)
        else:
          #Encode each token seperately
          tokenOutput = tokens_to_one_hot(tokens, MAX_TOKEN_VALUE, self.tokenblacklistkeys)

        return tokenOutput


    def get_partial_embedding_array(self, startSample=0, endSample=0):
        numberOfSamples  = endSample-startSample
        MAX_TOKEN_VALUE  = self.get_max_token_value()
        MAX_TOKEN_NUMBER = self.get_token_number()
        D = 300 #TODO: grab this from db_get_description_embeddings_number

                           
        embeddings = np.zeros((numberOfSamples,MAX_TOKEN_NUMBER,D), dtype=np.float32) #Typically 16 Tokens with 300 dimensions

        #Prepare functions
        self.libDataLoader.db_get_sample_description_tokens_number.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.libDataLoader.db_get_sample_description_tokens_number.restype  = ctypes.c_int

        #Prepare to take embeddings
        self.libDataLoader.db_get_sample_description_embeddings.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int]
        self.libDataLoader.db_get_sample_description_embeddings.restype  = POINTER(ctypes.c_float)
         
        #For each of the samples in batch size
        for sID in range(numberOfSamples): 
           numberOfTokensInThisSample = self.libDataLoader.db_get_sample_description_tokens_number(self.db,startSample+sID)
           for tokenID in range(numberOfTokensInThisSample):
              embeddingList = self.libDataLoader.db_get_sample_description_embeddings(self.db, startSample+sID, int(tokenID))
              for embeddingID in range(D): 
                embeddings[sID,tokenID,embeddingID] = float(embeddingList[embeddingID]) 

        #Check data
        """
        for sID in range(numberOfSamples):
           print("Sample : ",sID,end=" ") 
           for embeddingID in range(D): 
              print("Embedding Dim : ",embeddingID," | ",end=" ") 
              for tokenID in range(numberOfTokensInThisSample):
                 print(embeddings[sID,tokenID,embeddingID]," ",end=" ")
              print("\n")
           print("\n")
        """

        return embeddings


    def shuffle(self):
        #Regular shuffling
        self.libDataLoader.db_shuffle_indices.argtypes = [ctypes.c_void_p] 
        self.libDataLoader.db_shuffle_indices(self.db)

    def shuffle_based_on_loss(self):
        #Attempt at smarter shuffling by taking into account the loss
        self.libDataLoader.db_shuffle_indices_via_loss.argtypes = [ctypes.c_void_p] 
        self.libDataLoader.db_shuffle_indices_via_loss(self.db)

    def updateJointDifficulty(self,listOfDifficulties):
        #print("shuffle()") unsigned short jID,signed char newDifficulty
        self.libDataLoader.db_change_joint_difficulty.argtypes = [ctypes.c_void_p,ctypes.c_ushort,ctypes.c_byte]
        for jID in range(len(listOfDifficulties)):
             self.libDataLoader.db_change_joint_difficulty(self.db,jID,listOfDifficulties[jID])

    def updateEpochResults(self, loss, startSample, endSample, epoch):
        self.libDataLoader.db_update_sample_loss_range.argtypes = [ctypes.c_void_p,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_float]
        self.libDataLoader.db_update_sample_loss_range(self.db, startSample, endSample, loss)

    """
    def updateSampleResults(self, losses, epoch, learningRate):
        #This cannot been done with fit, we get losses every batch not for every sample
        print("Dataloader received losses for epoch ",epoch,"!")
        print("Received ",len(losses)," losses ")
        print("Losses = ",losses)
        self.libDataLoader.db_update_sample_loss.argtypes = [ctypes.c_void_p,ctypes.c_ulong,ctypes.c_float]
        for sampleNumber in range(len(losses)):
           db_update_sample_loss(self.db,sampleNumber,losses[sampleNumber])
    """

    def refresh_all_frame_augmentations(self,epoch):
        self.gradientSize = max(8,self.initialGradientSize-epoch)
        return self.update(0,self.numberOfSamples)

    def get_in_array(self):
        if (self.streamData):
           raise ValueError('Getting direct input access through get_in_array while streaming will not work!')

        startSample = 0
        numberOfImages = self.get_number_of_images(self.db)

        self.libDataLoader.db_get_in.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
        self.libDataLoader.db_get_in.restype  = POINTER(ctypes.c_ubyte)
        pixels = self.libDataLoader.db_get_in(self.db,startSample)
        return np.ctypeslib.as_array(pixels, shape=(numberOfImages, self.inHeight, self.inWidth,  self.inChannels))

    def get_out_array(self):
        if (self.streamData):
           raise ValueError('Getting direct output access through get_out_array while streaming will not work!')

        startSample = 0
        numberOfImages = self.get_number_of_images(self.db)

        self.libDataLoader.db_get_out.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
        self.libDataLoader.db_get_out.restype  = POINTER(ctypes.c_byte) #c_byte ctypes.c_void_p
        pixels = self.libDataLoader.db_get_out(self.db,startSample)
        return np.ctypeslib.as_array(pixels, shape=(numberOfImages, self.outHeight, self.outWidth,  self.out8BitChannels))
  

    def get_out_array_16bit(self):
        if (self.streamData):
           raise ValueError('Getting direct output access through get_out_array_16bit while streaming will not work!')

        #DEACTIVATED
        return None

        #16 bit values are mapped in previously loaded samples so 
        numberOfImages = self.get_number_of_images(self.db)
        startItem = 0
        endItem   = numberOfImages

        self.libDataLoader.db_get_out16bit.argtypes = [ctypes.c_void_p,ctypes.c_ulong,ctypes.c_ulong]
        self.libDataLoader.db_get_out16bit.restype  = POINTER(ctypes.c_short)  
        pixels = self.libDataLoader.db_get_out16bit(self.db,startItem,endItem)


        #IMPORTANT : 
        #Don't forget that whatever changes you do here should also be done in get_partial_update_IO_array
        npArray16BitOut =  np.ctypeslib.as_array(pixels, shape=(numberOfImages, self.outHeight, self.outWidth, self.output16BitChannels)).astype(np.int16) 
        #return npArray16BitOut

        #Convert to [-120..120] float32
        npArray16BitOutFloat =  npArray16BitOut.copy().astype(np.float32) * (120.0 / 32767.0)
        #npArray16BitOutFloat =  np.round(npArray16BitOutFloat, decimals=1)

        return npArray16BitOutFloat.astype(np.int8)
 

    def get_out_descriptions(self,encodeAsSingleMultiLabelToken=True):
        if (self.streamData):
           raise ValueError('Getting direct output access through get_out_descriptions while streaming will not work!')

        startSample = 0
        numberOfImages = self.get_number_of_images(self.db)

        npTokenOut = self.get_partial_token_array(startSample=startSample, endSample=numberOfImages, encodeAsSingleMultiLabelToken=encodeAsSingleMultiLabelToken)
        return npTokenOut 

    #---------------------------------------
    def save_image(self,filename,sampleNumber):
        self.libDataLoader.db_save_heatmap.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ulong] 
        path = filename.encode('utf-8')  
        self.libDataLoader.db_save_image(self.db,path,sampleNumber)

    def save_heatmap(self,filename,sampleNumber,heatmapNumber):
        self.libDataLoader.db_save_heatmap8bit_as_jpg.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_ushort] 
        path = filename.encode('utf-8')  
        self.libDataLoader.db_save_heatmap8bit_as_jpg(self.db,path,sampleNumber,heatmapNumber)
    #---------------------------------------
    def __del__(self):
        if self.db:
            self.libDataLoader.db_destroy.argtypes = [ctypes.c_void_p]
            self.libDataLoader.db_destroy.restype  = ctypes.c_int
            self.libDataLoader.db_destroy(self.db)

def convertTokensToText(vocabulary,tokens):
  text = ""
  for token in tokens:
      if token == 0:
         pass
      elif (str(token) in vocabulary):
         text = text + " " + vocabulary[str(token)]
      else:
         text = text + "("+str(token)+")"
 
  return text

def convertTokensToTextMult(vocabulary,tokens):
  text = ""
  for token in tokens:
      tokValueArr = np.where(token == np.max(token))
      tokValue = tokValueArr[0][0]
      if tokValue == 0:
         pass
      elif (str(tokValue) in vocabulary):
         text = text + " " + vocabulary[str(tokValue)]
      else:
         text = text + "("+str(tokValue)+")"
 
  return text

# Test
if __name__ == "__main__":
    import cv2
    import random
    os.system("rm sample*.jpg sample*.pnm sample*.png sample*.txt")

    stream = True

    depthBytes       = 1
    addDepthLevels   = 4
    addDepthMap      = 1
    addTextMap       = 1
    addPAFs          = 1
    addBackground    = 0
    addNormals       = 1
    addSegmentation  = 1
    numberOfHeatmaps = 17 + addBackground + (12*addPAFs) + (addDepthMap*depthBytes) + (3*addNormals*depthBytes) + addDepthLevels + addTextMap + (10*addSegmentation)
    numberOf16BitHeatmaps = 1

    inWidth  = 256 #220
    inHeight = 256 #220

    outWidth  = 256
    outHeight = 256

    if (stream):
      print("Test streaming")

      batchSize=32
      db = DataLoader((inWidth,inHeight,3),
                      (outWidth,outHeight,numberOfHeatmaps), 
                      output16BitChannels=numberOf16BitHeatmaps,
                      numberOfThreads=8,
                      streamData=1,
                      addBackground=addBackground,
                      addDepthMap=addDepthMap,  
                      addDepthLevelsHeatmaps=addDepthLevels,  
                      addNormals=addNormals,
                      addPAFs = addPAFs, 
                      addSegmentation = addSegmentation,

                      batchSize=batchSize,
                      bytesPerDepthValue=depthBytes,
                      datasets = [
                                    ["../coco/cocoTrain.db",         "../coco/cache/coco/train2017","../coco/cache/coco/depth_train2017","../coco/cache/coco/segment_train2017"]
                                   ,[ "../background/AM-2k.db",      "../background/AM-2k/train",   "../background/AM-2k/depth_train",   "../background/AM-2k/segment_train" ]   # <- Disable this if you don't have the data
                                   ,[ "../background/BG-20k.db",     "../background/BG-20k/train",  "../background/BG-20k/depth_train",  "../background/BG-20k/segment_train" ]  # <- Disable this if you don't have the data
                                   #,["../openpose/openposeTrain.db", "../openpose/data/train",      "../openpose/data/depth_train" ,     "../openpose/data/segment_train"]        # <- Disable this if you don't have the data
                                   #,["../generated/generatedTrain.db", "../generated/data/train", "../generated/data/depth_train" , "../generated/data/segment_train" ]
                                ],
                      vocabularyPath = "../../2d_pose_estimation/vocabulary.json",
                      forceLibUpdate=True) # numberOfSamples=10000 
      print("Number of Samples :", db.get_number_of_samples(db.db))
      print("Number of Images  :", db.get_number_of_images(db.db))

      db.get_token_fequencies()
      #sys.exit(0)

      print("Shuffling")
      db.shuffle()
      for batch in range(5):#Grab 3 batches
        print("Generate Batch ",batch)
        in_array, out_array, out_array16bit  = db.get_partial_update_IO_array(startSample=batch*batchSize, endSample=(batch+1)*batchSize)
        token_array                          = db.get_partial_token_array(startSample=batch*batchSize, endSample=(batch+1)*batchSize, encodeAsSingleMultiLabelToken=False)

        for sample in range(batchSize):
          image   =  in_array[sample]
          image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
          
          if not out_array16bit is None:
            #Assuming depth is float32 with values [-120.0 .. 120.0] 
            depthFloat = out_array16bit[sample]
            depth16 = ( depthFloat * (32767.0/120.0) ) + 32767.0
            cv2.imwrite('sampleB%u_%u_D.png'%(batch,sample),depth16.astype(np.uint16))

            #depth16 = out_array16bit[sample] 
            #cv2.imwrite('sampleB%u_%u_D.png'%(batch,sample),depth16.astype(np.uint16))
            #depthf = depth16.astype(np.float32)
            #depthf = depthf + 32767.0
            #depthUInt16 = depthf.astype(np.uint16) 
            #cv2.imwrite('sampleB%u_%u_DF.png'%(batch,sample),depthUInt16)
          #else:
          #  print("No 16 bit data")

          #tokensToText = str(token_array[sample])
          #tokensToText = convertTokensToText(db.vocabulary,token_array[sample])
          tokensToText  = convertTokensToTextMult(db.vocabulary,token_array[sample])
          cv2.putText(image, tokensToText, (1,15),  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (123, 123, 123), 1)
          cv2.putText(image, tokensToText, (2,15),  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
          cv2.putText(image, tokensToText, (3,15),  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

          cv2.imwrite('sampleB%u_%u_I.jpg'%(batch,sample),image)
          for heatmap in range(out_array.shape[3]):
            hm = out_array[sample, :, :, heatmap] # Access one channel at a time
            cv2.imwrite('sampleB%u_%u_P_%u.jpg'%(batch,sample,heatmap),hm)
            db.save_heatmap("sampleB%u_%u_C_hm%u.jpg"%(batch,sample,heatmap) , sample, heatmap)
    else:
      print("Test static mode")
      db = DataLoader((inWidth,inHeight,3),
                      (outWidth,outHeight,numberOfHeatmaps), 
                      output16BitChannels=numberOf16BitHeatmaps,
                      numberOfThreads=8, 
                      addBackground=addBackground,
                      addDepthMap=addDepthMap,  
                      addNormals=addNormals,   
                      bytesPerDepthValue=depthBytes,
                      numberOfSamples=20000,
                      forceLibUpdate=True) # numberOfSamples=10000 
      in_array  = db.get_in_array()
      out_array = db.get_out_array()
      print("Input array shape:", in_array.shape)
      print("Input array DType:", in_array.dtype)
      print("Output array shape:", out_array.shape)
      print("Output array DType:", out_array.dtype)
      #print("Check for values out of range : ",checkIfAnyValuesOutsideOfRange(out_array,-120,120))

      for sample in [3751, 17897, 18960, 19653]:#range(100):
        #sample = random.randrange(0,db.numberOfSamples )
        image = in_array[sample]
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  
        cv2.imwrite('sample%u_I.jpg'%(sample),image)
        for heatmap in range(db.out8BitChannels):
          hm = out_array[sample, :, :, heatmap] # Access one channel at a time
          cv2.imwrite('sample%u_P_%u.jpg'%(sample,heatmap),hm)
          db.save_heatmap("sample%u_C_hm%u.jpg"%(sample,heatmap) , sample, heatmap)

