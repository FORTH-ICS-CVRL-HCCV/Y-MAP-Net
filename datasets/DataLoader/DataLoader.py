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
        print(bcolors.FAIL, "Could not find DataLoader Library (", filename, "), compiling a fresh one..!",
              bcolors.ENDC)
        print("Current directory was (", os.getcwd(), ") ")
        directory = os.path.dirname(os.path.abspath(filename))
        creationScript = directory + "/makeLibrary.sh"
        os.system(creationScript)

    if not exists(filename):
        directory = os.path.dirname(os.path.abspath(filename))
        print(bcolors.FAIL, "Could not make DataLoader Library, terminating", bcolors.ENDC)
        print("Directory we tried was : ", directory)
        sys.exit(0)

    libDataLoader = CDLL(filename, mode=ctypes.RTLD_GLOBAL)
    libDataLoader.connect()

    return libDataLoader
#-------------------------------------------------------------------------------
class DataAugmentationParams(ctypes.Structure):
    """Mirrors struct DataAugmentation in DataAugmentation.h (field order must match exactly)."""
    _fields_ = [
        ('chanceDestroy',                   ctypes.c_float),
        ('chancePerturbed',                 ctypes.c_float),
        ('chancePanAndZoom',                ctypes.c_float),
        ('chanceBurnedPixels',              ctypes.c_float),
        ('chanceBrightnessContrast',        ctypes.c_float),
        ('chanceBrightnessContrastUniform', ctypes.c_float),
        ('chanceHorizontalFlip',            ctypes.c_float),
        ('chanceRotate90',                  ctypes.c_float),
        ('chanceCoarseDropout',             ctypes.c_float),
        ('chanceGaussianBlur',              ctypes.c_float),
        ('chanceDefocusBlur',               ctypes.c_float),
        ('chanceMotionBlur',                ctypes.c_float),
        ('maxZoomFactor',                   ctypes.c_float),
        ('perturbationMagnitude',           ctypes.c_int),
        ('maxBurnedPixels',                 ctypes.c_int),
        ('minBrightnessChange',             ctypes.c_float),
        ('maxBrightnessChange',             ctypes.c_float),
        ('minRelContrastChange',            ctypes.c_float),
        ('maxRelContrastChange',            ctypes.c_float),
        ('minUniformBrightnessChange',      ctypes.c_float),
        ('maxUniformBrightnessChange',      ctypes.c_float),
        ('coarseDropoutHolesMin',           ctypes.c_int),
        ('coarseDropoutHolesMax',           ctypes.c_int),
        ('coarseDropoutMinSize',            ctypes.c_int),
        ('coarseDropoutMaxSize',            ctypes.c_int),
        ('gaussianBlurSigmaMin',            ctypes.c_float),
        ('gaussianBlurSigmaMax',            ctypes.c_float),
        ('defocusBlurRadiusMin',            ctypes.c_int),
        ('defocusBlurRadiusMax',            ctypes.c_int),
        ('motionBlurLengthMin',             ctypes.c_int),
        ('motionBlurLengthMax',             ctypes.c_int),
    ]
#-------------------------------------------------------------------------------
def _setup_ctypes(lib):
    """Set all argtypes and restypes for libDataLoader once at load time."""
    vp = ctypes.c_void_p
    ul = ctypes.c_ulong
    ui = ctypes.c_uint
    i = ctypes.c_int
    f = ctypes.c_float
    cp = ctypes.c_char_p
    sz = ctypes.c_size_t
    us = ctypes.c_ushort
    b = ctypes.c_byte
    ub = ctypes.c_ubyte
    sh = ctypes.c_short
    P = ctypes.POINTER

    lib.test.argtypes = [i, vp]
    lib.test.restype = i

    lib.db_create.argtypes = [
        vp,  # struct DatabaseList* dbSources
        ul,  # unsigned long numberOfSamples
        i,  # streamData
        i,  # doubleBuffer [B1] (effective only with streamData)
        i,
        i,  # batchSize, workerThreads
        i,
        i,  # gradientSize, PAFSize
        i,
        i,
        i,  # doAugmentations, addPAFs, addBackground
        i,
        i,
        i,
        i,  # addDepthMap, addDepthLevelsHeatmaps, addNormals, addSegmentation
        i,  # addInstanceDetection
        i,  # addSuperpoint
        i,  # superpointChannels
        cp,  # superpointPcaPath (char*)
        i,  # bytesPerDepthValue
        ui,
        ui,
        ui,  # widthIn, heightIn, channelsIn
        ui,
        ui,
        ui,
        ui
    ]  # widthOut, heightOut, channelsOut8Bit, output16BitChannels
    lib.db_create.restype = vp

    # Geolocation is a module-global toggle read at db_create time (keeps db_create's
    # signature unchanged). Call db_set_geolocation_config(enable) BEFORE db_create.
    lib.db_set_geolocation_config.argtypes = [i]
    lib.db_set_geolocation_config.restype = None

    # Channel layout of the combined depth + segmentation ("all") files, a module global read at db_create
    # time like geolocation: db_set_combined_layout(label, depth high byte, depth low byte), returns 0 if invalid.
    lib.db_set_combined_layout.argtypes = [ui, ui, ui]
    lib.db_set_combined_layout.restype = i

    # Embeddings source is a module-global path read at db_create time (same reason as
    # geolocation above). Call db_set_embeddings_path(path) BEFORE db_create to point at
    # e.g. 2d_pose_estimation/conceptnet-numberbatch/GloVe_D300.embeddings instead of GloVe.
    lib.db_set_embeddings_path.argtypes = [cp]
    lib.db_set_embeddings_path.restype = None

    # .pzpd archive prefetcher (PZPDLoader.c), module globals read at db_create time:
    # mode (-1 off, 0 auto, 1 map, 2 pagecache, 3 buffers), I/O threads, budget MB, window (0 = defaults)
    lib.db_set_prefetch_config.argtypes = [i, i, i, i]
    lib.db_set_prefetch_config.restype = None
    lib.db_get_sample_description.argtypes = [vp, ul, cp, ui]
    lib.db_get_sample_description.restype = i

    lib.db_destroy.argtypes = [vp]
    lib.db_destroy.restype = i

    lib.db_allocate_source_list.argtypes = [ui]
    lib.db_allocate_source_list.restype = vp
    lib.db_destroy_source_list.argtypes = [vp]
    lib.db_set_source_entry.argtypes = [vp, ui, cp, cp, cp, cp, cp, i]
    lib.db_set_source_entry.restype = vp

    lib.db_get_number_of_samples.argtypes = [vp]
    lib.db_get_number_of_samples.restype = ul
    lib.db_get_number_of_images.argtypes = [vp]
    lib.db_get_number_of_images.restype = ul

    lib.db_get_sample_total_loss.argtypes = [vp, ul]
    lib.db_get_sample_total_loss.restype = f
    lib.db_get_sample_train_passes.argtypes = [vp, ul]
    lib.db_get_sample_train_passes.restype = ul

    lib.db_get_filename_of_sample.argtypes = [vp, ul, cp, sz]
    lib.db_get_filename_of_sample.restype = i  # C returns int (DATALOADER.md C6)

    lib.db_disable_heatmap_output.argtypes = [vp]

    lib.db_update.argtypes = [vp, ul, ul, i, i, i]
    lib.db_update.restype = i
    lib.db_StartUpdate.argtypes = [vp, ul, ul, i, i, i]
    lib.db_StartUpdate.restype = i
    lib.db_CollectUpdate.argtypes = [vp, ul, ul, i, i, i]
    lib.db_CollectUpdate.restype = i

    lib.db_set_priority.argtypes = [i]
    lib.db_set_priority.restype = i
    lib.db_print_readSpeed.argtypes = [vp]

    lib.db_get_in.argtypes = [vp, ul]
    lib.db_get_in.restype = P(ub)
    lib.db_get_out.argtypes = [vp, ul]
    lib.db_get_out.restype = P(b)
    lib.db_get_out16bit.argtypes = [vp, ul]  # endSample param removed C-side (DATALOADER.md A6f)
    lib.db_get_out16bit.restype = P(sh)

    # [B1] Double-buffer pipeline primitives (consumer getters fall back to the fill side
    # when doubleBuffer is off, so the Python read path is uniform in both modes).
    lib.db_pipeline_drain.argtypes = [vp]
    lib.db_pipeline_drain.restype = i
    lib.db_swap_buffers.argtypes = [vp]
    lib.db_swap_buffers.restype = i
    lib.db_get_consumer_in.argtypes = [vp, ul]
    lib.db_get_consumer_in.restype = P(ub)
    lib.db_get_consumer_out.argtypes = [vp, ul]
    lib.db_get_consumer_out.restype = P(b)
    lib.db_get_consumer_out16bit.argtypes = [vp, ul]
    lib.db_get_consumer_out16bit.restype = P(sh)

    lib.db_save_image.argtypes = [vp, cp, ul]
    lib.db_save_heatmap8bit_as_jpg.argtypes = [vp, cp, ul, us]

    lib.db_shuffle_indices.argtypes = [vp]
    lib.db_shuffle_indices_via_loss.argtypes = [vp]
    lib.db_change_joint_difficulty.argtypes = [vp, us, b]
    lib.db_update_sample_loss_range.argtypes = [vp, ul, ul, f]
    lib.db_update_sample_loss.argtypes = [vp, ul, f]

    lib.db_sort_description_tokens.argtypes = [vp]
    lib.db_sort_description_tokens_based_on_count.argtypes = [vp, i, i]
    lib.db_remove_duplicate_description_tokens.argtypes = [vp]
    lib.db_set_MAX_sample_description_token_value.argtypes = [vp, i]
    lib.db_set_MAX_sample_description_token_value.restype = i

    lib.db_get_valid_segmentations.argtypes = [vp]
    lib.db_get_valid_segmentations.restype = i
    lib.db_get_total_segmentation_classes.argtypes = [vp]
    lib.db_get_total_segmentation_classes.restype = i

    # 8-bit heatmap channel layout (single source of truth, resolved C-side in
    # HeatmapLayout.h). Name lookup + index enumeration so Python never re-hardcodes
    # channel ranges.
    lib.db_get_channel_range.argtypes = [vp, cp, P(ui), P(ui)]
    lib.db_get_channel_range.restype = i
    lib.db_get_number_of_channel_groups.argtypes = [vp]
    lib.db_get_number_of_channel_groups.restype = i
    lib.db_get_channel_group_name.argtypes = [vp, i]
    lib.db_get_channel_group_name.restype = cp
    lib.db_get_channel_group_start.argtypes = [vp, i]
    lib.db_get_channel_group_start.restype = i
    lib.db_get_channel_group_count.argtypes = [vp, i]
    lib.db_get_channel_group_count.restype = i
    lib.db_get_channel_group_enabled.argtypes = [vp, i]
    lib.db_get_channel_group_enabled.restype = i

    # Joint names straight from the loaded .db (DATALOADER.md C5)
    lib.db_get_number_of_joints.argtypes = [vp]
    lib.db_get_number_of_joints.restype = i
    lib.db_get_joint_name.argtypes = [vp, i]
    lib.db_get_joint_name.restype = cp

    # Runtime augmentation parameters (struct DataAugmentation in DataAugmentation.h).
    lib.db_get_augmentation_params.argtypes = [vp, ctypes.POINTER(DataAugmentationParams)]
    lib.db_get_augmentation_params.restype = None
    lib.db_set_augmentation_params.argtypes = [vp, ctypes.POINTER(DataAugmentationParams)]
    lib.db_set_augmentation_params.restype = None

    lib.db_get_MAX_sample_description_token_value.argtypes = [vp]
    lib.db_get_MAX_sample_description_token_value.restype = i
    lib.db_get_MAX_sample_description_tokens_number.argtypes = [vp]
    lib.db_get_MAX_sample_description_tokens_number.restype = i
    lib.db_get_descriptor_elements_number.argtypes = [vp]
    lib.db_get_descriptor_elements_number.restype = i

    lib.db_allocate_token_blacklist.argtypes = [vp, ui]
    lib.db_allocate_token_blacklist.restype = i
    lib.db_add_token_to_blacklist.argtypes = [vp, ui]
    lib.db_add_token_to_blacklist.restype = i
    lib.db_compile_added_token_blacklist.argtypes = [vp]
    lib.db_compile_added_token_blacklist.restype = i

    lib.db_allocate_token_synonym_map.argtypes = [vp, ui]
    lib.db_allocate_token_synonym_map.restype = i
    lib.db_add_token_synonym_pair.argtypes = [vp, us, us]
    lib.db_add_token_synonym_pair.restype = i
    lib.db_compile_token_synonym_map.argtypes = [vp]
    lib.db_compile_token_synonym_map.restype = i

    lib.db_expand_multiword_tokens.argtypes = [vp, cp]
    lib.db_expand_multiword_tokens.restype = i

    lib.db_count_description_tokens.argtypes = [vp, i]
    lib.db_count_description_tokens.restype = P(ul)
    lib.db_free_description_token_count.argtypes = [vp]
    lib.db_free_description_token_count.restype = i
    lib.db_count_description_token_weight.argtypes = [vp, i]
    lib.db_count_description_token_weight.restype = P(f)
    lib.db_set_sample_description_tokens.argtypes = [vp, ul, P(us), i]
    lib.db_set_sample_description_tokens.restype = i

    lib.db_get_sample_descriptors.argtypes = [vp, ul]
    lib.db_get_sample_descriptors.restype = P(f)
    lib.db_get_batch_descriptors.argtypes = [vp, ul, ul, P(f)]
    lib.db_get_batch_descriptors.restype = None
    lib.db_get_sample_description_tokens.argtypes = [vp, ul]
    lib.db_get_sample_description_tokens.restype = P(us)
    lib.db_get_sample_description_tokens_number.argtypes = [vp, ul]
    lib.db_get_sample_description_tokens_number.restype = i
    lib.db_get_sample_description_embeddings.argtypes = [vp, ul, i]
    lib.db_get_sample_description_embeddings.restype = P(f)
    lib.db_get_batch_tokens.argtypes = [vp, ul, ul, P(us)]
    lib.db_get_batch_tokens.restype = None
    lib.db_get_batch_embeddings.argtypes = [vp, ul, ul, ul, P(f)]
    lib.db_get_batch_embeddings.restype = None
    lib.db_get_description_embeddings_number.argtypes = [vp, ul]
    lib.db_get_description_embeddings_number.restype = i
    lib.db_get_embeddings_offset.argtypes = [vp]
    lib.db_get_embeddings_offset.restype = f
    lib.db_get_embeddings_scaling.argtypes = [vp]
    lib.db_get_embeddings_scaling.restype = f
    lib.db_set_embeddings_offset_scaling.argtypes = [vp, f, f]
    lib.db_set_embeddings_offset_scaling.restype = i
#-------------------------------------------------------------------------------
def checkIfAnyValuesOutsideOfRange(arr, minV, maxV):
    for sample in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            for y in range(arr.shape[2]):
                for hm in range(arr.shape[3]):
                    val = arr[sample][x][y][hm]
                    if (val < minV) or (val > maxV):
                        print("Out of bounds sample", sample, "x", x, "y", y, "hm", hm)
                        print("Val : ", val)
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
#---------------------------------------------------------------------------------------------
# Function to filter only verbs using spaCy
#python3 -m pip install spacy
#python3 -m spacy download en_core_web_sm
def is_verb(nlp, word):
    print("Parsing : ", word)
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
    numberOfTokens = tokens.shape[1]
    numberOfValues = max_token_value
    # Initialize a matrix of zeros with shape (16, max_token_value + 1)
    one_hot_encoded = np.full((numberOfSamples, numberOfTokens, max_token_value + 1), inactive, dtype=np.int8)

    # Set the corresponding index for each token to 1 (vectorized, DATALOADER.md B3)
    sampleIdx = np.arange(numberOfSamples)[:, None]
    tokenIdx = np.arange(numberOfTokens)[None, :]
    one_hot_encoded[sampleIdx, tokenIdx, tokens] = active
    #Always force 0 as false
    one_hot_encoded[:, :, 0] = inactive  #forced

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
    numberOfTokens = tokens.shape[1]
    numberOfValues = max_token_value

    classes_encoded = np.full((numberOfSamples, max_token_value + 1), inactive,
                              dtype=np.int8)  # inactive , dtype=np.int8

    # Set the corresponding index for each token to 1 (vectorized, DATALOADER.md B3)
    sampleIdx = np.arange(numberOfSamples)[:, None]
    classes_encoded[sampleIdx, tokens] = active  #active
    #Always force 0 as false
    classes_encoded[:, 0] = inactive  #forced
    return classes_encoded
#---------------------------------------------------------------------------------------------
class DataLoader:
    def __init__(
            self,
            inDims,
            outDims,
            output16BitChannels: int = 0,
            numberOfSamples: int = 0,  #0 means automatically use whole dataset
            numberOfThreads: int = 4,  #Most CPUs nowadays have 4 cores/threads
            streamData: int = 0,  #By default dont stream data load it all at once
            doubleBuffer: int = 0,  #[B1] pipeline batch prep with GPU compute (config dataLoaderDoubleBuffer); streaming-only, forced off otherwise
            # (a `zeroCopy` kwarg used to sit here — removed; see the note in __init__ below)
            batchSize: int = 32,  #This is only important when streaming data
            gradientSize: int = 12,
            PAFSize: int = 5,
            doAugmentations: int = 1,  #1 means we want to do augmentations
            addPAFs: int = 1,
            addBackground: int = 1,
            addDepthMap: int = 1,
            addDepthLevelsHeatmaps: int = 0,
            addNormals: int = 1,
            addSegmentation: int = 1,
            addInstanceDetection: int = 0,
            addSuperpoint: int = 0,
            superpointChannels: int = 0,
            superpointPcaPath: str = "",
            addGeolocation: int = 0,  #Global lat/lon density channel (GEOLOCATION.md). Off by default.
            combinedChannels=(0, 1, 2),  #Combined "all" files: channels of (segmentation label, depth high byte, depth low byte)
            prefetchMode: str = "auto",  #.pzpd sources: "auto", "map", "pagecache", "buffers" or "off" (mmap views on demand)
            prefetchIOThreads: int = 0,  #.pzpd prefetcher I/O threads, budget (MB, buffers mode) and window (records); 0 = defaults
            prefetchBudgetMB: int = 0,
            prefetchWindow: int = 0,
            bytesPerDepthValue: int = 2,
            ignoreNoSkeletonSamples: int = 0,  #0 means we want all samples
            datasets=[[
                "cocoTrain.db", "../coco/cache/coco/train2017", "../coco/cache/coco/depth_train2017",
                "../coco/cache/coco/segment_train2017"
            ]],
            vocabularyPath="2d_pose_estimation/vocabulary.json",
            synonymPath=None,
            embeddingsPath=None,  # e.g. "2d_pose_estimation/conceptnet-numberbatch/GloVe_D300.embeddings";
                                   # None keeps the C-side default (GloVe) unchanged.
            elevatePriority=False,
            libraryPath: str = "./libDataLoader.so",
            forceLibUpdate=False):
        #Tokens
        #---------------------------------------
        self.vocabulary = loadJSON(vocabularyPath)
        self.vocabularyPath = vocabularyPath   # the C side re-reads it for expand_multiword_tokens()
        self.synonymPairs = []
        if synonymPath is not None:
            # synonymPath is either a single path (str) or a list of paths. Multiple
            # concept-grouping maps (plural/synonym folding, gender neutralisation,
            # concept simplification) are merged in order; a later file overrides an
            # earlier remap for the same source word. Chains are then resolved
            # transitively (e.g. men->man plus man->person collapses to men->person),
            # so the result is independent of whether the C map applies transitively.
            synonymPaths = [synonymPath] if isinstance(synonymPath, str) else list(synonymPath)
            vocabByWord = {v: int(k) for k, v in self.vocabulary.items()}
            remap = {}
            for path in synonymPaths:
                loaded = 0
                for fromWord, toWord in loadJSON(path):
                    fromID = vocabByWord.get(fromWord)
                    toID = vocabByWord.get(toWord)
                    if fromID is not None and toID is not None:
                        remap[fromID] = toID
                        loaded += 1
                    else:
                        print("[synonym] skipping '%s'->'%s': not found in vocabulary" % (fromWord, toWord))
                print("[synonym] %s: %u pairs" % (path, loaded))

            def resolveTarget(fromID):
                seen = set()
                cur = fromID
                while cur in remap and cur not in seen:
                    seen.add(cur)
                    cur = remap[cur]
                return cur

            self.synonymPairs = [(fromID, target) for fromID in remap
                                 for target in (resolveTarget(fromID),) if target != fromID]
            print("[synonym] loaded %u remapping pairs from %u file(s)"
                  % (len(self.synonymPairs), len(synonymPaths)))
        # Token blacklist lives in token_blacklist.json next to the vocabulary (moved out of
        # this constructor — DATALOADER.md C4). The JSON's "blacklist" key is the ACTIVE list
        # (punctuation + stopwords + overrepresented + difficult); its "verbs" key preserves
        # the historical verb list which was never applied (the extend was commented out).
        blacklistPath = os.path.join(os.path.dirname(vocabularyPath) or ".", "token_blacklist.json")
        if exists(blacklistPath):
            self.tokenblacklist = list(loadJSON(blacklistPath).get("blacklist", []))
            print("Loaded", len(self.tokenblacklist), "token blacklist entries from", blacklistPath)
        else:
            # Fallback for older packaged model dirs that predate token_blacklist.json —
            # the same active list that used to be hardcoded here.
            print(bcolors.WARNING, "No", blacklistPath, "found — using the built-in default token blacklist", bcolors.ENDC)
            self.tokenblacklist = [
                "(", ")", ",", ".", "a", "an", 's', 'hu', 'hy', 'w',
                #stopwords
                "of", "on", "and", "I", "in", "the", "is", "it", "at", "to", "with", "for", "from", "near", "while",
                #overrepresented
                'top', 'next', 'two', 'are', 'it', 'its', 'up', 'down', 'left', 'right', 'in', 'out', 'front', 'to', 'has', 'by',
                #difficult
                'nintendo', 'wii', 'umpire'
            ]

        # (The historical spaCy verb-extraction experiment and its 500+ word list moved
        # to token_blacklist.json under the "verbs" key — never applied to the blacklist.)

        self.tokenblacklistkeys = get_blacklisted_keys(self.vocabulary, self.tokenblacklist)
        print("Token Black list : ", self.tokenblacklist)
        print("Token Black list keys : ", self.tokenblacklistkeys)
        print("Token Black list number of values : ", len(self.tokenblacklistkeys))
        #This should not be done because it corrupts the order of tokens!
        #vocabularyWithoutBlacklist         = remove_blacklisted_tokens(self.vocabulary , self.tokenblacklist)
        #print("Token remaining number of values : ",len(vocabularyWithoutBlacklist))
        #---------------------------------------
        self.numberOfThreads = numberOfThreads
        self.initialGradientSize = gradientSize
        self.doAugmentations = doAugmentations
        self.ignoreNoSkeletonSamples = ignoreNoSkeletonSamples
        self.gradientSize = gradientSize
        self.PAFSize = PAFSize
        self.streamData = streamData
        self.batchSize = batchSize
        # [B1] Effective double-buffer switch: streaming-only (C enforces the same guard
        # and prints when it forces the flag off). _prefetchRange tracks the batch range a
        # StartUpdate is currently filling, None = pipeline cold.
        self.doubleBuffer = int(bool(streamData) and bool(doubleBuffer))
        self._prefetchRange = None
        # NOTE: a [B2] "zero-copy" feature used to live here (a `zeroCopy` kwarg + read-only
        # numpy views of the consume-side buffer, replacing the per-batch .copy() below). It was
        # REMOVED because it is unfixable in this architecture: the training generator prefetches
        # up to `max_queue_size` (queueSize=32) batches ahead, so up to ~32 batches are alive at
        # once — but a view aliases one of only 2 double-buffer slots, so every queued view but
        # the newest is overwritten by a later fetch before the GPU reads it (corrupts the
        # continuous heads: depth/normals/denoise/depthlvls). Making it safe would need a ring as
        # deep as the queue (~34 buffers, ~7 GB host RAM) — and it would still buy nothing, since
        # the .copy() runs on the prefetch thread concurrently with the GPU step and is already
        # hidden behind it. doubleBuffer + copy is the correct, safe design.
        #---------------------------------------
        self.inWidth = inDims[0]
        self.inHeight = inDims[1]
        self.inChannels = inDims[2]
        #---------------------------------------
        self.outWidth = outDims[0]
        self.outHeight = outDims[1]
        self.out8BitChannels = outDims[2]
        self.output16BitChannels = output16BitChannels
        #---------------------------------------
        self.lastStartSample = 0
        self.lastEndSample = 0
        self.bytesPerDepthValue = bytesPerDepthValue
        self.cpuTimeSeconds = 0

        # Dimensionality of each GloVe/token vector. Discovered from the embeddings file
        # the C side actually loaded (see get_embedding_number_of_elements) as soon as the
        # db exists, so swapping in a wider embeddings file needs no edit here.
        self.D = 0

        self.db = None
        self.channel_ranges = {}   # modality name -> (start, end), filled after db_create
        self.channel_layout = []   # full ordered catalogue [(name, start, count, enabled)]
        self.augmentation_params = {}  # field name -> value, filled after db_create
        self.libDataLoader = loadLibrary(libraryPath, forceUpdate=forceLibUpdate)
        _setup_ctypes(self.libDataLoader)

        if (elevatePriority):
            self.setPriority(-20)

        #This is new handling for datasets sources to make enabling/disabling datasets easier
        #-----------------------------------------------------------------------------
        if (len(datasets) < 1):
            raise ValueError("Please give some datasets to load..  ")

        datasetsEnabled = list()
        for sourceEntry in datasets:
            if len(sourceEntry) == 6:
                #new entry with use/dontuse first element
                confStr = sourceEntry[0].lower()
                if (confStr == "use" or confStr == "enable" or confStr == "enabled" or confStr == "1"
                        or confStr == "true"):
                    datasetsEnabled.append(sourceEntry[1:])
            elif len(sourceEntry) == 4:
                #Regular old style entry :)
                datasetsEnabled.append(sourceEntry)
            elif len(sourceEntry) == 2:
                #.pzpd archive entry: [use/dontuse, archive] (the directories are inside the archive)
                confStr = sourceEntry[0].lower()
                if (confStr == "use" or confStr == "enable" or confStr == "enabled" or confStr == "1"
                        or confStr == "true"):
                    datasetsEnabled.append([sourceEntry[1], "", "", "", ""])
            else:
                print("Incorrectly formatted dataset entry with ", len(sourceEntry), " elements :")
                print(sourceEntry)

        if (len(datasetsEnabled) < 1):
            raise ValueError("Please give some ENABLED datasets to load..  ")

        self.dbListPtr = self.createSourceList(len(datasetsEnabled))
        sourceID = 0
        for sourceEntry in datasetsEnabled:
            self.addToSourceList(self.dbListPtr, sourceID, sourceEntry[0], sourceEntry[1], sourceEntry[2],
                                 sourceEntry[3], sourceEntry[4], self.ignoreNoSkeletonSamples)
            sourceID = sourceID + 1
        #-----------------------------------------------------------------------------

        superpointPcaPathBytes = superpointPcaPath.encode("utf-8") if superpointPcaPath else None
        # Toggle the geolocation channel (module global) BEFORE db_create reads it.
        self.libDataLoader.db_set_geolocation_config(int(addGeolocation))
        # How the combined "all" files store segmentation + 16-bit depth (BEFORE db_create reads it). (0,1,2) is
        # what append16bitToSAM3.py / compressSegmentDepth.py write; libraries before 2026-09-23 read (0,2,1).
        if len(combinedChannels) != 3 or not self.libDataLoader.db_set_combined_layout(*[int(c) for c in combinedChannels]):
            raise ValueError("combinedChannels must be (label, depth high, depth low) channels 0..3, depth bytes different: %r" % (combinedChannels,))
        prefetchModes = {"off": -1, "auto": 0, "map": 1, "pagecache": 2, "buffers": 3}
        if prefetchMode not in prefetchModes:
            raise ValueError("prefetchMode must be one of %s" % list(prefetchModes))
        self.libDataLoader.db_set_prefetch_config(prefetchModes[prefetchMode], int(prefetchIOThreads),
                                                  int(prefetchBudgetMB), int(prefetchWindow))
        # Point the embeddings loader at a non-GloVe trio (module global) BEFORE
        # db_create reads it. Left alone (C-side default) when embeddingsPath is None.
        if embeddingsPath is not None:
            self.libDataLoader.db_set_embeddings_path(embeddingsPath.encode('utf-8'))
        self.db = self.libDataLoader.db_create(
            self.dbListPtr, numberOfSamples, streamData, self.doubleBuffer, batchSize, self.numberOfThreads,
            self.gradientSize, self.PAFSize, doAugmentations, addPAFs, addBackground, addDepthMap,
            addDepthLevelsHeatmaps, addNormals, addSegmentation, addInstanceDetection, addSuperpoint,
            superpointChannels, superpointPcaPathBytes,
            bytesPerDepthValue, self.inWidth, self.inHeight, self.inChannels, self.outWidth,
            self.outHeight, self.out8BitChannels, self.output16BitChannels)  # Create database
        if not self.db:
            print(bcolors.FAIL, "Failed to create database", bcolors.ENDC)
            raise ValueError("Failed to create database")
            return

        # Read back the resolved 8-bit heatmap channel layout from C (single source of
        # truth). self.channel_ranges maps modality name -> (start, end) half-open range
        # for every ENABLED modality; loss slices / metrics / decode read from here
        # instead of hardcoding indices. See HeatmapLayout.h / db_build_heatmap_layout.
        self._build_channel_ranges()
        self._build_augmentation_params()

        numberOfTokensFromDictionary = len(list(self.vocabulary.keys()))
        print("Number of tokens from dictionary : ", numberOfTokensFromDictionary)
        self.set_max_token_value(
            numberOfTokensFromDictionary
        )  #This is important to enforce a standard number of tokens and not be based on auto-detection

        # .pzpd sources carry caption text but no pre-tokenized description line (see
        # tokenize_missing_description_tokens docstring) -- tokenize those BEFORE
        # expand_multiword_tokens() so their multi-word vocabulary entries get split too, same
        # as a .db source's.
        self.tokenize_missing_description_tokens()

        # Split whole-caption vocabulary entries ("A large number of seagulls gathered on the
        # beach") into their component word tokens. Runs FIRST, so the words it produces are then
        # sorted by their real frequency, deduplicated, and — crucially — stripped of stopwords by
        # the blacklist below. Doing it after the blacklist would leave 'the'/'of'/'on' in place.
        self.expand_multiword_tokens()

        #self.sortTokens()                #<- sort tokens
        # Sort tokens in DESCENDING frequency order (ascendingOrder=0), i.e. the most
        # common tokens are assigned the lowest slot indices (t00, t01, ...).
        #
        # Rationale — Textual Frequency Law (TFL):
        #   "Adam's Law: Textual Frequency Law on Large Language Models"
        #   https://arxiv.org/abs/2604.02176
        #   Core finding: neural models learn high-frequency linguistic expressions more
        #   reliably than low-frequency ones.  Sentence-level frequency is estimated as
        #   the geometric mean of its constituent word frequencies (see word_frequencies.json).
        #
        # Why descending order matters for the autoregressive token chain in NNModel.py:
        #   add_token_output() builds tokens t00→t07 sequentially.  Each token's hidden
        #   state is passed as a residual to the next (scaled by nextTokenStrength), so
        #   prediction quality degrades along the chain.  Staggered dropout
        #   (dropoutRate / (i+1)) also means t00 is the most strongly regularised and
        #   therefore the most robustly trained slot.
        #
        #   By placing high-frequency (most learnable) tokens first:
        #     1. t00 — the cleanest, residual-free slot — targets the token the model
        #        can predict most reliably, maximising its training signal quality.
        #     2. t00's reliable GloVe embedding becomes a good anchor residual that
        #        steers t01, t02, ... toward a coherent semantic neighbourhood, rather
        #        than propagating an unreliable prediction forward through the chain.
        #     3. removeDuplicateTokens() (called immediately below) retains the first
        #        occurrence of each token; descending sort ensures the copy kept is the
        #        highest-frequency one — again the most learnable target.
        #
        #   Ascending order (low-frequency first) would invert all three benefits:
        #   rare, hard-to-predict tokens at t00 would corrupt every downstream slot.
        self.sortTokensBasedOnFrequency(ascendingOrder=0)  #<- Sort tokens based on their frequency (descending)

        self.removeDuplicateTokens()  # <- Remove duplicate tokens

        self.update_token_blacklist(self.tokenblacklistkeys, lowThreshold=0)

        if self.synonymPairs:
            self.apply_synonym_map(self.synonymPairs)
            # Remapping merges distinct words onto one ID, so a description that read
            # "man person boat" now reads "person person boat" and would waste one of the
            # 8 caption slots. Dedupe again: removeDuplicateTokens() keeps the first
            # occurrence and compacts the array (numberOfTokens shrinks), which pulls the
            # next real token forward into the freed slot.
            self.removeDuplicateTokens()

        self.numberOfSamples = self.libDataLoader.db_get_number_of_samples(self.db)
        print(bcolors.OKGREEN, "Created a database with ", self.numberOfSamples, " samples ", bcolors.ENDC)
        #self.test()

    def test(self):
        res = self.libDataLoader.test(0, 0)
        return res

    def get_labels(self):
        """Joint names as stored in the loaded .db (single source of truth — DATALOADER.md C5).

        Falls back to the historical hardcoded COCO-17 list only if the db exposes no
        names (older .db / uninitialized database)."""
        names = []
        numberOfJoints = self.libDataLoader.db_get_number_of_joints(self.db)
        for jID in range(numberOfJoints):
            raw = self.libDataLoader.db_get_joint_name(self.db, jID)
            name = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else str(raw)
            if name == "":
                names = []
                break
            names.append(name)
        if names:
            return names

        #Fallback: pre-C5 hardcoded COCO-17 list
        return [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
            "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]

    def disableHeatmapOutput(self):
        self.libDataLoader.db_disable_heatmap_output(self.db)

    def createSourceList(self, numberOfSources):
        print("Initializing ", numberOfSources, " sources in C code")
        return self.libDataLoader.db_allocate_source_list(numberOfSources)

    def destroySourceList(self, sourceDB):
        self.libDataLoader.db_destroy_source_list(sourceDB)

    def addToSourceList(self, sourceDB, sourceID, pathToDB, pathToImages, pathToDepth, pathToSegmentation,
                        pathToAllDataCombined, ignoreNoSkeletonSamples):
        path1 = pathToDB.encode('utf-8')
        path2 = pathToImages.encode('utf-8')
        path3 = pathToDepth.encode('utf-8')
        path4 = pathToSegmentation.encode('utf-8')
        path5 = pathToAllDataCombined.encode('utf-8')
        return self.libDataLoader.db_set_source_entry(sourceDB, sourceID, path1, path2, path3, path4, path5,
                                                      ignoreNoSkeletonSamples)

    def get_number_of_samples(self, db):
        return self.libDataLoader.db_get_number_of_samples(db)

    def get_number_of_images(self, db):
        return self.libDataLoader.db_get_number_of_images(db)

    def get_total_loss_of_sample(self, sample):
        return self.libDataLoader.db_get_sample_total_loss(self.db, sample)

    def get_train_passes_of_sample(self, sample):
        return self.libDataLoader.db_get_sample_train_passes(self.db, sample)

    def get_filename_of_sample(self, sample):
        buffer_size = 1024
        buffer = ctypes.create_string_buffer(buffer_size)
        if (self.libDataLoader.db_get_filename_of_sample(self.db, sample, buffer, buffer_size)):
            return buffer.value.decode('utf-8')
        return ""

    def get_sample_description(self, sample):
        """Caption text of the sample at a position (.pzpd sources; "" for .db sources, which carry token IDs only)."""
        buffer_size = 8192
        buffer = ctypes.create_string_buffer(buffer_size)
        n = self.libDataLoader.db_get_sample_description(self.db, sample, buffer, buffer_size)
        if n > 0:
            return buffer.value.decode('utf-8', 'replace')
        return ""

    def dump_sample_report(self, path="sample_report.json"):
        try:
          print("Writing ", path)
          results = dict()
          for sID in range(self.numberOfSamples):
            if (self.get_total_loss_of_sample(sID) != 0) and (self.get_train_passes_of_sample(sID) != 0):
                results[sID] = [
                    sID,
                    self.get_filename_of_sample(sID),
                    self.get_total_loss_of_sample(sID) / (self.get_train_passes_of_sample(sID) + 0.0001)
                ]

          sorted_results = sorted(results.values(), key=lambda x: x[2])

          import json
          with open(path, 'w') as fp:
            json.dump(sorted_results, fp)
          print("Done writing ", path)

        except Exception as e:
            print("Error occured while dump_sample_report was executing: ",str(e))


    def update(self, startSample, endSample):
        self.lastStartSample = startSample
        self.lastEndSample   = endSample
        #print("update(",startSample," , ",endSample,")")
        return self.libDataLoader.db_update(self.db, startSample, endSample, self.numberOfThreads, self.gradientSize,
                                            self.PAFSize)

    def startUpdate(self, startSample, endSample):
        self.lastStartSample = startSample
        self.lastEndSample   = endSample
        return self.libDataLoader.db_StartUpdate(self.db, startSample, endSample, self.numberOfThreads,
                                                 self.gradientSize, self.PAFSize)

    def collectUpdate(self, startSample, endSample):
        self.lastStartSample = startSample
        self.lastEndSample   = endSample
        return self.libDataLoader.db_CollectUpdate(self.db, startSample, endSample, self.numberOfThreads,
                                                   self.gradientSize, self.PAFSize)

    def setPriority(self, newPriority):
        return self.libDataLoader.db_set_priority(newPriority)

    def printReadSpeed(self):
        return self.libDataLoader.db_print_readSpeed(self.db)

    def get_partial_update_IO_array(self, startSample=0, endSample=0, produce16BitData=True):
        if (not self.streamData):
            raise ValueError('Getting streaming output access while not streaming will not work!')

        startTime = time.time()

        if self.doubleBuffer:
            # [B1.1] Pipelined path (DATALOADER.md B1): overlap C batch preparation with
            # the GPU step. Collect the batch a previous call prefetched, swap it to the
            # consume side, immediately start filling the NEXT batch, and read the consume
            # side. The .copy() below is retained in B1.1 (removal is B1.2/zero-copy).
            requested = (startSample, endSample)
            if self._prefetchRange != requested:
                # Cold start / seek / post-shuffle: discard any stale in-flight prefetch
                # (drain is a no-op when nothing is running).
                self.libDataLoader.db_pipeline_drain(self.db)
                self.startUpdate(startSample, endSample)
            updateOk = self.collectUpdate(startSample, endSample)
            if updateOk:
                self.libDataLoader.db_swap_buffers(self.db)
                # Prefetch the next sequential batch — but never across the epoch
                # boundary, so the on_epoch_end shuffle stays race-free by construction.
                batch = endSample - startSample
                if (batch > 0) and (endSample + batch <= self.numberOfSamples):
                    self.startUpdate(endSample, endSample + batch)
                    self._prefetchRange = (endSample, endSample + batch)
                else:
                    self._prefetchRange = None
                # Loss attribution (updateEpochResults reads lastStart/EndSample) must
                # reference the CONSUMED batch, not the prefetched one the helpers set.
                self.lastStartSample = startSample
                self.lastEndSample = endSample
            getIn   = self.libDataLoader.db_get_consumer_in
            getOut  = self.libDataLoader.db_get_consumer_out
            getOut16 = self.libDataLoader.db_get_consumer_out16bit
        else:
            # Classic synchronous path — bit-identical to the pre-B1 behaviour.
            updateOk = self.update(startSample, endSample)  #<- regular one stage update..!
            getIn   = self.libDataLoader.db_get_in
            getOut  = self.libDataLoader.db_get_out
            getOut16 = self.libDataLoader.db_get_out16bit

        if updateOk:
            pixelsIn = getIn(self.db, 0)  #We always want to start at the first element
            npArrayIn = np.ctypeslib.as_array(
                pixelsIn,
                shape=(endSample - startSample, self.inHeight, self.inWidth, self.inChannels))

            pixelsOut = getOut(self.db, 0)  #We always want to start at the first element
            npArrayOut = np.ctypeslib.as_array(
                pixelsOut, shape=(endSample - startSample, self.outHeight, self.outWidth, self.out8BitChannels))

            # Copies detach from the C buffer the next update overwrites. (A zero-copy view path
            # used to be selectable here — removed; see the note in __init__ for why.)
            npArrayIn = npArrayIn.copy()    #<- batch x H x W x 3 uint8 (~8 MB at batch 40)
            npArrayOut = npArrayOut.copy()  #<- batch x H x W x 76 int8 (~200 MB at batch 40 — NOT small!)

            #Uncommenting the following printf statements should return :
            #Heatmaps I/O Types  <class 'numpy.ndarray'> / <class 'numpy.ndarray'>
            #print("Heatmaps I/O Types ",type(npArrayIn),"/",type(npArrayOut))
            # Heatmaps I/O DTypes  uint8 / int8
            #print("Heatmaps I/O DTypes ",npArrayIn.dtype,"/",npArrayOut.dtype)

            #16-Bit output DEACTIVATED
            npArray16BitOut = None

            if (produce16BitData):
                if (self.output16BitChannels > 0):
                    numberOfImages = self.get_number_of_images(self.db)
                    pixels16BitOut = getOut16(self.db, 0)

                    npArray16BitOut = np.ctypeslib.as_array(
                        pixels16BitOut,
                        shape=(numberOfImages, self.outHeight, self.outWidth,
                               self.output16BitChannels))  #Just The Raw 16-bit value [-32767 .. 32767]
                    npArray16BitOut = npArray16BitOut.astype(np.int16)  #astype copies -> detaches from the C buffer (zero-copy view path removed; see __init__)
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
            return npArrayIn, npArrayOut, npArray16BitOut  #.astype(np.int8)

        print("Could not perform DB In update in range ", startSample, endSample)
        raise ValueError('Could not perform DB update')
        return None, None

    def sortTokens(self):
        self.libDataLoader.db_sort_description_tokens(self.db)

    def sortTokensBasedOnFrequency(self, ascendingOrder=1):
        MAX_TOKEN_VALUE = self.get_max_token_value()
        if (MAX_TOKEN_VALUE == 0):
            raise ValueError("MAX_TOKEN_VALUE is zero!")
        self.libDataLoader.db_sort_description_tokens_based_on_count(self.db, MAX_TOKEN_VALUE, int(ascendingOrder))

    def removeDuplicateTokens(self):
        self.libDataLoader.db_remove_duplicate_description_tokens(self.db)

    def set_max_token_value(self, newValue):
        return self.libDataLoader.db_set_MAX_sample_description_token_value(self.db, newValue)

    def get_valid_segmentations(self):
        return self.libDataLoader.db_get_valid_segmentations(self.db)

    def get_total_segmentation_classes(self):
        return self.libDataLoader.db_get_total_segmentation_classes(self.db)

    #---------------------------------------------------------------------
    # 8-bit heatmap channel layout (resolved C-side, single source of truth)
    #---------------------------------------------------------------------
    def _build_channel_ranges(self):
        """Read the resolved channel layout from C into self.channel_ranges.

        self.channel_ranges: dict modality_name -> (start, end) half-open range, in
        channel order, for every ENABLED modality. self.channel_layout keeps the full
        ordered catalogue (including disabled modalities) for inspection/printing.
        """
        self.channel_ranges = {}
        self.channel_layout = []  # list of (name, start, count, enabled) in order
        nGroups = self.libDataLoader.db_get_number_of_channel_groups(self.db)
        for idx in range(nGroups):
            raw = self.libDataLoader.db_get_channel_group_name(self.db, idx)
            name = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else str(raw)
            start = self.libDataLoader.db_get_channel_group_start(self.db, idx)
            count = self.libDataLoader.db_get_channel_group_count(self.db, idx)
            enabled = bool(self.libDataLoader.db_get_channel_group_enabled(self.db, idx))
            self.channel_layout.append((name, start, count, enabled))
            if enabled and count > 0:
                self.channel_ranges[name] = (start, start + count)
        return self.channel_ranges

    def get_channel_ranges(self):
        """dict modality_name -> (start, end) half-open, for enabled modalities."""
        return self.channel_ranges

    def get_channel_range(self, name):
        """(start, end) half-open range for one modality, or None if absent/disabled.

        Reads directly from C so it is always consistent with the resolved layout."""
        start = ctypes.c_uint(0)
        count = ctypes.c_uint(0)
        ok = self.libDataLoader.db_get_channel_range(
            self.db, name.encode('utf-8'), ctypes.byref(start), ctypes.byref(count))
        if not ok:
            return None
        return (start.value, start.value + count.value)

    def print_channel_layout(self):
        """Print the full ordered channel catalogue (enabled + disabled)."""
        print(bcolors.OKGREEN, "8-bit heatmap channel layout:", bcolors.ENDC)
        for (name, start, count, enabled) in self.channel_layout:
            if enabled:
                print("   [%2d..%2d] %-16s (%d ch)" % (start, start + count - 1, name, count))
            else:
                print("   [  off  ] %-16s" % name)

    # ------------------------------------------------------------------
    # Augmentation parameter access
    # ------------------------------------------------------------------
    def _build_augmentation_params(self):
        """Read struct DataAugmentation from C into self.augmentation_params dict."""
        s = DataAugmentationParams()
        self.libDataLoader.db_get_augmentation_params(self.db, ctypes.byref(s))
        self.augmentation_params = {f: getattr(s, f) for f, _ in DataAugmentationParams._fields_}
        return self.augmentation_params

    def get_augmentation_params(self):
        """Return current augmentation params as a plain dict (snapshot)."""
        return dict(self.augmentation_params)

    def set_augmentation_params(self, **kwargs):
        """Override one or more augmentation params and push to C.

        Example:
            db.set_augmentation_params(chanceHorizontalFlip=50.0, chanceRotate90=0.0)
        """
        s = DataAugmentationParams()
        self.libDataLoader.db_get_augmentation_params(self.db, ctypes.byref(s))
        for key, val in kwargs.items():
            if not hasattr(s, key):
                raise ValueError("Unknown augmentation param: %s" % key)
            setattr(s, key, val)
        self.libDataLoader.db_set_augmentation_params(self.db, ctypes.byref(s))
        self._build_augmentation_params()

    def print_augmentation_params(self):
        """Pretty-print the current augmentation parameter set."""
        print(bcolors.OKGREEN, "Augmentation parameters:", bcolors.ENDC)
        for key, val in self.augmentation_params.items():
            print("   %-38s %s" % (key, val))

    def get_max_token_value(self):
        return self.libDataLoader.db_get_MAX_sample_description_token_value(self.db)

    def get_token_number(self):
        return self.libDataLoader.db_get_MAX_sample_description_tokens_number(self.db)

    def get_descriptor_number_of_elements(self):
        return self.libDataLoader.db_get_descriptor_elements_number(self.db)

    def get_embedding_number_of_elements(self):
        # D of the embeddings file the C side loaded (GloVe_D*.embeddings header).
        if self.D <= 0:
            self.D = int(self.libDataLoader.db_get_description_embeddings_number(self.db, 0))
        return self.D

    def get_embedding_offset_scaling(self):
        """(offset, scaling) the loaded token embeddings are currently expressed in.

        The C loader stores vectors as (raw + offset) * scaling — this is the pair the tanh
        token head is trained against, so anything inverting a prediction back to a raw GloVe
        vector must use exactly these two numbers.
        """
        return (float(self.libDataLoader.db_get_embeddings_offset(self.db)),
                float(self.libDataLoader.db_get_embeddings_scaling(self.db)))

    def set_embedding_offset_scaling(self, offset, scaling):
        """Re-express the loaded token embeddings under a new (offset, scaling) pair.

        Rescales the in-memory vectors in place, so the token targets can be moved (e.g. to
        use more of the tanh range) without regenerating GloVe_D*.embeddings. Affects only
        this process; the file on disk is untouched. Call before streaming starts — batches
        already prefetched under the old pair are stale, so the pipeline is reset here.
        Returns True on success, False if the current scaling is 0 and cannot be inverted.
        """
        if scaling == 0:
            raise ValueError("scaling must be non-zero (it is inverted on the next change)")
        ok = int(self.libDataLoader.db_set_embeddings_offset_scaling(self.db, float(offset),
                                                                     float(scaling)))
        if ok:
            self._pipeline_reset()
        return bool(ok)

    def update_token_blacklist(self, blacklisttokenIDs, lowThreshold=0):

        if (lowThreshold > 0):
            #We have a lower threshold..
            tokenIDsToAddToBlackList = self.get_low_count_tokens(threshold=lowThreshold)
            blacklisttokenIDs.extend(tokenIDsToAddToBlackList)
            print("Overriding blacklist keys to also add those with a count < ", lowThreshold)
            self.tokenblacklistkeys = blacklisttokenIDs

        response = self.libDataLoader.db_allocate_token_blacklist(self.db, len(blacklisttokenIDs))

        if (response):
            for tID in blacklisttokenIDs:
                response = self.libDataLoader.db_add_token_to_blacklist(self.db, int(tID))

        response = self.libDataLoader.db_compile_added_token_blacklist(self.db)

        return response

    def apply_synonym_map(self, synonymPairs):
        self.libDataLoader.db_allocate_token_synonym_map(self.db, len(synonymPairs))
        for fromID, toID in synonymPairs:
            self.libDataLoader.db_add_token_synonym_pair(self.db, fromID, toID)
        return self.libDataLoader.db_compile_token_synonym_map(self.db)

    def tokenize_missing_description_tokens(self):
        """.pzpd sources ship caption text only (PZPDLoader.c leaves numberOfTokens=0 -- see
        convertToPZPD.py); .db sources already carry pre-tokenized description lines and
        get_sample_description() always returns "" for them. So: tokenize every sample whose
        caption text is non-empty, with the exact word-splitting + vocabulary lookup
        datasets/useVocabulary.py uses to build .db token lines, and push the ids into C with
        db_set_sample_description_tokens() -- everything downstream (expand_multiword_tokens,
        token counting/weights, get_partial_token_array, ...) then sees the same descriptionTokens
        array it would for a .db source."""
        import re
        word_to_id = {word: int(k) for k, word in self.vocabulary.items()}
        tokenized = 0
        for sID in range(self.get_number_of_samples(self.db)):
            desc = self.get_sample_description(sID)
            if not desc:
                continue
            words = re.findall(r'\w+|[.,()]', desc.lower())
            # id 0 is reserved as the C side's "empty slot" sentinel everywhere (counting,
            # blacklist indexing, ...) even though vocabulary.json legitimately assigns it to a
            # real symbol ("(") -- so it can never appear as an actual token, same as .db sources.
            ids = [word_to_id[w] for w in words if word_to_id.get(w, 0) != 0]
            if not ids:
                continue
            arr = (ctypes.c_ushort * len(ids))(*ids)
            if self.libDataLoader.db_set_sample_description_tokens(self.db, sID, arr, len(ids)):
                tokenized += 1
        if tokenized:
            print(bcolors.OKGREEN, "Tokenized %u .pzpd caption(s) against the vocabulary (%u words)"
                  % (tokenized, len(word_to_id)), bcolors.ENDC)
        return tokenized

    def expand_multiword_tokens(self):
        """Split whole-caption vocabulary entries into their component word tokens.

        A handful of vocabulary entries are entire captions stored as one token
        ("A large number of seagulls gathered on the beach"); their mean-pooled GloVe
        vector is meaningless and they are far too rare to learn. The C side scans the
        vocabulary for entries containing a space and rewrites every occurrence into the
        token IDs of the words it is made of."""
        return self.libDataLoader.db_expand_multiword_tokens(
            self.db, self.vocabularyPath.encode('utf-8'))

    def get_low_count_tokens(self, threshold=10):
        MAX_TOKEN_VALUE = self.get_max_token_value()
        print("MAX_TOKEN_VALUE ", MAX_TOKEN_VALUE)

        tokenCount = self.libDataLoader.db_count_description_tokens(self.db, MAX_TOKEN_VALUE)

        #Bring tokens from a C array to a numpy list
        tokenCountNP = []
        for tokenID in range(MAX_TOKEN_VALUE):
            thisVal = int(tokenCount[tokenID])
            if (thisVal < threshold):
                tokenCountNP.append(tokenID)

        print("Freeing memory in C side.. ")
        self.libDataLoader.db_free_description_token_count(tokenCount)

        print("Tokens with counts lower than ", threshold, ".. ")
        tokenDescriptions = list(self.vocabulary.values())
        for tokenID in tokenCountNP:
            print(tokenDescriptions[tokenID], end=" ")

        return tokenCountNP

    def get_description_token_counts(self):
        """Per-vocabulary-id occurrence count across every sample currently loaded (any source
        type -- .db or .pzpd -- since it reads the already-tokenized captions the C loader
        built at db_create/db_expand_multiword_tokens time). Index i is how many samples'
        captions contain vocabulary id i; index 0 is unused (0 is not a real token id)."""
        MAX_TOKEN_VALUE = self.get_max_token_value()
        tokenCount = self.libDataLoader.db_count_description_tokens(self.db, MAX_TOKEN_VALUE)
        counts = np.array([int(tokenCount[i]) for i in range(MAX_TOKEN_VALUE + 1)], dtype=np.int64)
        self.libDataLoader.db_free_description_token_count(tokenCount)
        return counts

    def get_token_frequencies(self):
        MAX_TOKEN_VALUE = self.get_max_token_value()
        print("MAX_TOKEN_VALUE ", MAX_TOKEN_VALUE)

        print("Printing token frequency summary.. ")
        tokenDescriptions = list(self.vocabulary.values())

        if (len(tokenDescriptions) != MAX_TOKEN_VALUE):
            print(bcolors.FAIL, "There are ", len(tokenDescriptions), " token descriptions ")
            print("and MAX_TOKEN_VALUE = ", MAX_TOKEN_VALUE, bcolors.ENDC)
            print(bcolors.FAIL, "Setting MAX_TOKEN_VALUE = ", len(tokenDescriptions),
                  " as a workaround, but need to fix this", MAX_TOKEN_VALUE, bcolors.ENDC)
            self.set_max_token_value(len(tokenDescriptions))
            MAX_TOKEN_VALUE = len(tokenDescriptions)

        tokenFrequency = self.libDataLoader.db_count_description_token_weight(self.db, MAX_TOKEN_VALUE)

        #Bring tokens from a C array to a numpy list
        tokenFrequencyNP = np.full((MAX_TOKEN_VALUE + 1), 0.0, dtype=np.float32)
        #tokenFrequencyNP = []
        #tokenFrequencyNP.append(0.0) #<- The first zero
        # NOTE (2026-09-08, serial 287): the "+1" here was a leftover from an intended 1-based
        # class axis. It put vocabulary id i's weight on CLASS i+1, while tokens_to_classes()
        # puts class i at index i and the blacklist knock-down below indexes UNSHIFTED -- so the
        # two halves of this function disagreed and 33.0% of classes trained with their
        # NEIGHBOUR's inverse frequency (measured on cocoTrain: `person`, the intended 1.0
        # anchor of the class-weight conditioning, trained at the 8.0 clip ceiling while `dog`
        # and `car` sat at the 0.05 floor). Verify with ymapnet/tokens/verifyClassWeights.py.
        for tokenID in range(MAX_TOKEN_VALUE):
            tokenFrequencyNP[tokenID] = (float(tokenFrequency[tokenID]))

        print("Freeing memory in C side.. ")
        self.libDataLoader.db_free_description_token_count(tokenFrequency)

        print("Reducing weights on black listed items.. ")
        for tokenID in self.tokenblacklistkeys:
            tokenFrequencyNP[int(tokenID)] = 1.0

        howManyValuesToIterateOver = len(tokenDescriptions)
        for tokenID in range(howManyValuesToIterateOver):
            thisDescription = tokenDescriptions[tokenID]
            thisFrequency = tokenFrequencyNP[tokenID]
            """
          print(thisDescription,"(%0.2f)" % thisFrequency, end=" ")
          if (tokenID%8==0):
            print("") #Add new line every 8 printed tokens for visualization purposes..
          """

            if (tokenFrequencyNP[tokenID] < 0.0):
                print("Negative value for token ", tokenDescriptions[tokenID])
                sys.exit(1)
        print("")

        return tokenFrequencyNP

    def get_partial_descriptor_array(self, startSample=0, endSample=0):
        numberOfSamples = endSample - startSample
        NUMBER_OF_DESCRIPTOR_ELEMENTS = self.get_descriptor_number_of_elements()

        descriptors = np.zeros((numberOfSamples, NUMBER_OF_DESCRIPTOR_ELEMENTS), dtype=np.float32)

        # When USE_DINOV2_FEATURES is disabled db_get_descriptor_elements_number returns 0,
        # so the array stays empty and no C call is made.
        if NUMBER_OF_DESCRIPTOR_ELEMENTS > 0:
            ptr = descriptors.ctypes.data_as(POINTER(ctypes.c_float))
            self.libDataLoader.db_get_batch_descriptors(self.db, startSample, endSample, ptr)

        return descriptors

    def get_partial_token_array(self, startSample=0, endSample=0, encodeAsSingleMultiLabelToken=True):
        #if (not self.streamData):
        #   raise ValueError('Getting streaming output access while not streaming will not work!')

        numberOfSamples = endSample - startSample
        MAX_TOKEN_VALUE = self.get_max_token_value()
        MAX_TOKEN_NUMBER = self.get_token_number()

        #print(" MAX_TOKEN_VALUE ", MAX_TOKEN_VALUE, "    " )
        #print(" MAX_TOKEN_NUMBER ", MAX_TOKEN_NUMBER, "    " )
        tokens = np.zeros((numberOfSamples, MAX_TOKEN_NUMBER),
                          dtype=np.uint16)  #Tokens are 0..2048 so are encoded as ushort 16bit
        #argtypes/restype set centrally in _setup_ctypes (DATALOADER.md C1)
        #One C call fills the whole batch instead of a ctypes call per sample (DATALOADER.md B3)
        if numberOfSamples > 0:
            ptr = tokens.ctypes.data_as(POINTER(ctypes.c_uint16))
            self.libDataLoader.db_get_batch_tokens(self.db, startSample, endSample, ptr)

        if (encodeAsSingleMultiLabelToken):
            #Encode everything as one
            tokenOutput = tokens_to_classes(tokens, MAX_TOKEN_VALUE, self.tokenblacklistkeys)
        else:
            #Encode each token seperately
            tokenOutput = tokens_to_one_hot(tokens, MAX_TOKEN_VALUE, self.tokenblacklistkeys)

        return tokenOutput

    def get_partial_embedding_array(self, startSample=0, endSample=0):
        numberOfSamples  = endSample - startSample
        MAX_TOKEN_NUMBER = self.get_token_number()
        D = self.get_embedding_number_of_elements()

        embeddings = np.zeros((numberOfSamples, MAX_TOKEN_NUMBER, D),
                              dtype=np.float32)  #Typically 16 Tokens with D dimensions

        #argtypes/restype set centrally in _setup_ctypes (DATALOADER.md C1)
        #One C call fills the whole batch instead of per-sample/per-token/per-dim loops (DATALOADER.md B3)
        if numberOfSamples > 0:
            ptr = embeddings.ctypes.data_as(POINTER(ctypes.c_float))
            self.libDataLoader.db_get_batch_embeddings(self.db, startSample, endSample, D, ptr)

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

    def _pipeline_reset(self):
        # [B1] Drain any in-flight prefetch and mark the pipeline cold. The C side also
        # drains inside the shuffles / set_augmentation_params (belt-and-braces), but the
        # Python-side _prefetchRange must be invalidated too: a batch prefetched under the
        # OLD index ordering is stale after a shuffle and must be recomputed.
        if getattr(self, 'doubleBuffer', 0):
            self.libDataLoader.db_pipeline_drain(self.db)
        self._prefetchRange = None

    def shuffle(self):
        #Regular shuffling
        # [B1] invariant: in-flight workers read db->indices — never permute them with an
        # uncollected StartUpdate (DATALOADER.md B1); reset also discards stale prefetches.
        self._pipeline_reset()
        self.libDataLoader.db_shuffle_indices(self.db)

    def shuffle_based_on_loss(self):
        #Attempt at smarter shuffling by taking into account the loss
        # [B1] drain/reset-before-shuffle invariant applies here too — see shuffle().
        self._pipeline_reset()
        self.libDataLoader.db_shuffle_indices_via_loss(self.db)

    def updateJointDifficulty(self, listOfDifficulties):
        for jID in range(len(listOfDifficulties)):
            self.libDataLoader.db_change_joint_difficulty(self.db, jID, listOfDifficulties[jID])

    def updateEpochResults(self, loss, startSample, endSample, epoch):
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

    def refresh_all_frame_augmentations(self, epoch):
        self.gradientSize = max(8, self.initialGradientSize - epoch)
        return self.update(0, self.numberOfSamples)

    def get_in_array(self):
        if (self.streamData):
            raise ValueError('Getting direct input access through get_in_array while streaming will not work!')

        startSample = 0
        numberOfImages = self.get_number_of_images(self.db)

        pixels = self.libDataLoader.db_get_in(self.db, startSample)  # argtypes set centrally in _setup_ctypes
        return np.ctypeslib.as_array(pixels, shape=(numberOfImages, self.inHeight, self.inWidth, self.inChannels))

    def get_out_array(self):
        if (self.streamData):
            raise ValueError('Getting direct output access through get_out_array while streaming will not work!')

        startSample = 0
        numberOfImages = self.get_number_of_images(self.db)

        pixels = self.libDataLoader.db_get_out(self.db, startSample)  # argtypes set centrally in _setup_ctypes
        return np.ctypeslib.as_array(pixels,
                                     shape=(numberOfImages, self.outHeight, self.outWidth, self.out8BitChannels))

    def get_out_array_16bit(self):
        if (self.streamData):
            raise ValueError('Getting direct output access through get_out_array_16bit while streaming will not work!')

        #DEACTIVATED
        return None

        #16 bit values are mapped in previously loaded samples so
        numberOfImages = self.get_number_of_images(self.db)
        startItem = 0
        endItem = numberOfImages

        pixels = self.libDataLoader.db_get_out16bit(self.db, startItem)  # argtypes set centrally in _setup_ctypes

        #IMPORTANT :
        #Don't forget that whatever changes you do here should also be done in get_partial_update_IO_array
        npArray16BitOut = np.ctypeslib.as_array(
            pixels, shape=(numberOfImages, self.outHeight, self.outWidth, self.output16BitChannels)).astype(np.int16)
        #return npArray16BitOut

        #Convert to [-120..120] float32
        npArray16BitOutFloat = npArray16BitOut.copy().astype(np.float32) * (120.0 / 32767.0)
        #npArray16BitOutFloat =  np.round(npArray16BitOutFloat, decimals=1)

        return npArray16BitOutFloat.astype(np.int8)

    def get_out_descriptions(self, encodeAsSingleMultiLabelToken=True):
        if (self.streamData):
            raise ValueError('Getting direct output access through get_out_descriptions while streaming will not work!')

        startSample = 0
        numberOfImages = self.get_number_of_images(self.db)

        npTokenOut = self.get_partial_token_array(startSample=startSample, endSample=numberOfImages,
                                                  encodeAsSingleMultiLabelToken=encodeAsSingleMultiLabelToken)
        return npTokenOut

    #---------------------------------------
    def save_image(self, filename, sampleNumber):
        path = filename.encode('utf-8')
        self.libDataLoader.db_save_image(self.db, path, sampleNumber)

    def save_heatmap(self, filename, sampleNumber, heatmapNumber):
        path = filename.encode('utf-8')
        self.libDataLoader.db_save_heatmap8bit_as_jpg(self.db, path, sampleNumber, heatmapNumber)

    #---------------------------------------
    # Lifecycle (DATALOADER.md C3): close() is the explicit teardown path; the context
    # manager makes `with DataLoader(...) as db:` work; __del__ stays best-effort only
    # because during interpreter shutdown the shared library may already be torn down.
    #---------------------------------------
    def close(self):
        """Explicitly destroy the C-side database. Idempotent — safe to call twice."""
        if getattr(self, 'db', None):
            self.libDataLoader.db_destroy(self.db)
            self.db = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # interpreter shutdown — the .so may already be unloaded


def convertTokensToText(vocabulary, tokens):
    text = ""
    for token in tokens:
        if token == 0:
            pass
        elif (str(token) in vocabulary):
            text = text + " " + vocabulary[str(token)]
        else:
            text = text + "(" + str(token) + ")"

    return text


def convertTokensToTextMult(vocabulary, tokens):
    text = ""
    for token in tokens:
        tokValueArr = np.where(token == np.max(token))
        tokValue = tokValueArr[0][0]
        if tokValue == 0:
            pass
        elif (str(tokValue) in vocabulary):
            text = text + " " + vocabulary[str(tokValue)]
        else:
            text = text + "(" + str(tokValue) + ")"

    return text

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#                                            Test
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import cv2
    import random
    os.system("rm sample*.jpg sample*.pnm sample*.png sample*.txt")

    stream = True

    depthBytes = 1
    addDepthLevels = 4
    addDepthMap = 1
    addTextMap = 1
    addPAFs = 1
    addBackground = 0
    addNormals = 1
    addSegmentation = 1
    numberOfHeatmaps = 17 + addBackground + (12 * addPAFs) + (addDepthMap * depthBytes) + (
        3 * addNormals * depthBytes) + addDepthLevels + addTextMap + (10 * addSegmentation)
    numberOf16BitHeatmaps = 1

    inWidth = 256  #220
    inHeight = 256  #220

    outWidth = 256
    outHeight = 256

    if (stream):
        print("Test streaming")

        batchSize = 32
        db = DataLoader(
            (inWidth, inHeight, 3),
            (outWidth, outHeight, numberOfHeatmaps),
            output16BitChannels=numberOf16BitHeatmaps,
            numberOfThreads=8,
            streamData=1,
            addBackground=addBackground,
            addDepthMap=addDepthMap,
            addDepthLevelsHeatmaps=addDepthLevels,
            addNormals=addNormals,
            addPAFs=addPAFs,
            addSegmentation=addSegmentation,
            batchSize=batchSize,
            bytesPerDepthValue=depthBytes,
            datasets=[
                [
                    "../coco/cocoTrain.db", "../coco/cache/coco/train2017", "../coco/cache/coco/depth_train2017",
                    "../coco/cache/coco/segment_train2017"
                ],
                [
                    "../background/AM-2k.db", "../background/AM-2k/train", "../background/AM-2k/depth_train",
                    "../background/AM-2k/segment_train"
                ]  # <- Disable this if you don't have the data
                ,
                [
                    "../background/BG-20k.db", "../background/BG-20k/train", "../background/BG-20k/depth_train",
                    "../background/BG-20k/segment_train"
                ]  # <- Disable this if you don't have the data
                #,["../openpose/openposeTrain.db", "../openpose/data/train",      "../openpose/data/depth_train" ,     "../openpose/data/segment_train"]        # <- Disable this if you don't have the data
                #,["../generated/generatedTrain.db", "../generated/data/train", "../generated/data/depth_train" , "../generated/data/segment_train" ]
            ],
            vocabularyPath="../../2d_pose_estimation/vocabulary.json",
            forceLibUpdate=True)  # numberOfSamples=10000
        print("Number of Samples :", db.get_number_of_samples(db.db))
        print("Number of Images  :", db.get_number_of_images(db.db))

        db.get_token_frequencies()
        #sys.exit(0)

        print("Shuffling")
        db.shuffle()
        for batch in range(5):  #Grab 3 batches
            print("Generate Batch ", batch)
            in_array, out_array, out_array16bit = db.get_partial_update_IO_array(startSample=batch * batchSize,
                                                                                 endSample=(batch + 1) * batchSize)
            token_array = db.get_partial_token_array(startSample=batch * batchSize, endSample=(batch + 1) * batchSize,
                                                     encodeAsSingleMultiLabelToken=False)

            for sample in range(batchSize):
                image = in_array[sample]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if not out_array16bit is None:
                    #Assuming depth is float32 with values [-120.0 .. 120.0]
                    depthFloat = out_array16bit[sample]
                    depth16 = (depthFloat * (32767.0 / 120.0)) + 32767.0
                    cv2.imwrite('sampleB%u_%u_D.png' % (batch, sample), depth16.astype(np.uint16))

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
                tokensToText = convertTokensToTextMult(db.vocabulary, token_array[sample])
                cv2.putText(image, tokensToText, (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (123, 123, 123), 1)
                cv2.putText(image, tokensToText, (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                cv2.putText(image, tokensToText, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                cv2.imwrite('sampleB%u_%u_I.jpg' % (batch, sample), image)
                for heatmap in range(out_array.shape[3]):
                    hm = out_array[sample, :, :, heatmap]  # Access one channel at a time
                    cv2.imwrite('sampleB%u_%u_P_%u.jpg' % (batch, sample, heatmap), hm)
                    db.save_heatmap("sampleB%u_%u_C_hm%u.jpg" % (batch, sample, heatmap), sample, heatmap)
    else:
        print("Test static mode")
        db = DataLoader((inWidth, inHeight, 3), (outWidth, outHeight, numberOfHeatmaps),
                        output16BitChannels=numberOf16BitHeatmaps, numberOfThreads=8, addBackground=addBackground,
                        addDepthMap=addDepthMap, addNormals=addNormals, bytesPerDepthValue=depthBytes,
                        numberOfSamples=20000, forceLibUpdate=True)  # numberOfSamples=10000
        in_array = db.get_in_array()
        out_array = db.get_out_array()
        print("Input array shape:", in_array.shape)
        print("Input array DType:", in_array.dtype)
        print("Output array shape:", out_array.shape)
        print("Output array DType:", out_array.dtype)
        #print("Check for values out of range : ",checkIfAnyValuesOutsideOfRange(out_array,-120,120))

        for sample in [3751, 17897, 18960, 19653]:  #range(100):
            #sample = random.randrange(0,db.numberOfSamples )
            image = in_array[sample]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite('sample%u_I.jpg' % (sample), image)
            for heatmap in range(db.out8BitChannels):
                hm = out_array[sample, :, :, heatmap]  # Access one channel at a time
                cv2.imwrite('sample%u_P_%u.jpg' % (sample, heatmap), hm)
                db.save_heatmap("sample%u_C_hm%u.jpg" % (sample, heatmap), sample, heatmap)
