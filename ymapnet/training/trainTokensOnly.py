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
#----------------------------------------------
useGPU = True
gpuIndex = None
if (len(sys.argv) > 1):
    #print('Argument List:', str(sys.argv))
    for i in range(0, len(sys.argv)):
        if (sys.argv[i] == "--cpu"):
            useGPU = False
        if (sys.argv[i] == "--gpu"):
            gpuIndex = sys.argv[i + 1]
# Set CUDA_VISIBLE_DEVICES to an empty string to force TensorFlow to use the CPU
if (not useGPU):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # <- Force CPU
elif (gpuIndex is not None):
    # Restrict this process to a single physical GPU *before* TensorFlow ever
    # touches the driver, so it never even initializes a context on the others.
    # This is what makes it safe to run alongside another (e.g. containerized)
    # training job pinned to a different physical GPU on the same box: with
    # CUDA_VISIBLE_DEVICES=<gpuIndex>, that physical GPU becomes this process's
    # "/gpu:0", matching the (unchanged) GPUsUsedForTraining config entries.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuIndex)
#----------------------------------------------
try:
    import cv2
    import tensorflow as tf
    import keras
    from keras import callbacks
    from keras.callbacks import TensorBoard
    from keras import layers, models
    from keras.models import Sequential

    # Never pre-allocate the whole device: growth-only allocation lets this
    # process share a physical GPU with whatever else already has memory
    # resident on it (a concurrent job, ours or another container's) instead
    # of grabbing the entire card up front and starving/crashing it.
    if useGPU:
        for _gpu in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(_gpu, True)
            except RuntimeError as _e:
                print("Could not set memory growth on", _gpu, ":", _e)

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
from ymapnet.core.NNLosses import RSquaredMetric, HeatmapDistanceMetric, VanillaMSELossSimple, AdamWCautious
from ymapnet.utils.tools import bcolors, read_json_file, checkIfPathExists, checkIfFileExists, convert_bytes



#-------------------------------------------------------------------------------
class MemoryReclaim(keras.callbacks.Callback):
    """Return freed heap back to the OS at every epoch/validation boundary.

    The validation pass allocates ~27MB of large numpy buffers per batch (the C
    loader's per-batch output copy). Python frees them, but glibc keeps the arenas,
    so RSS climbs ~4GB per epoch on a 5000-sample val set and the OOM killer takes
    the run at epoch ~5 (measured 2026-09-04). gc.collect() drops any cycles and
    malloc_trim(0) hands the arenas back.
    """

    def __init__(self):
        super().__init__()
        self._trim = None
        try:
            import ctypes
            self._trim = ctypes.CDLL("libc.so.6").malloc_trim
            self._trim.argtypes = [ctypes.c_size_t]
        except Exception as e:
            print("MemoryReclaim: malloc_trim unavailable (%s), gc only" % e)

    def _rss_gb(self):
        try:
            for line in open("/proc/self/status"):
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1048576.0
        except Exception:
            pass
        return float("nan")

    def _reclaim(self, tag):
        before = self._rss_gb()
        gc.collect()
        if self._trim is not None:
            self._trim(0)
        after = self._rss_gb()
        print("MemoryReclaim[%s]: RSS %.2f -> %.2f GB" % (tag, before, after), flush=True)
        if os.environ.get("YMAP_MEM_DIAG"):
            import collections
            import numpy as _np
            tot = collections.Counter()
            cnt = collections.Counter()
            for o in gc.get_objects():
                try:
                    if isinstance(o, _np.ndarray):
                        key = "%s %s" % (o.shape, o.dtype)
                        tot[key] += o.nbytes
                        cnt[key] += 1
                except Exception:
                    pass
            live = sum(tot.values())
            print("  live numpy: %.2f GB across %d arrays" % (live / 1e9, sum(cnt.values())), flush=True)
            for k, v in tot.most_common(6):
                print("    %7.0f MB  x%-6d %s" % (v / 1048576, cnt[k], k), flush=True)

    def on_test_end(self, logs=None):
        self._reclaim("val")

    def on_epoch_end(self, epoch, logs=None):
        self._reclaim("epoch%d" % (epoch + 1))


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

#-------------------------------------------------------------------------------
from ymapnet.core.NNTraining import logTrainingHistory, logText, logSomeInputsAndOutputs, printTFVersion, getOptimizerFromCFG, TrainingDataGenerator, custom_lr_scheduler, custom_lr_schedulerWarmup, DataAugmentation, weighted_token_loss, extract_validation_losses, assertDescriptorsArePopulated, resolveModelDir, rebaseModelPath
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

    # Training reads the root (git-tracked) tokens.json as its source of truth,
    # unless --config points at a different (e.g. experiment-specific) file --
    # letting several named arms (tokens_foldON.json, tokens_foldOFF.json, ...)
    # coexist without repeatedly overwriting the tracked tokens.json.
    # After a successful run it is copied to <OUTPUT_DIR>/tokens.json (the packaged
    # runtime config) alongside the saved <OUTPUT_DIR>/tokens model.
    jsonPath = 'tokens.json'
    useRAMfs = True
    for _i in range(0, len(sys.argv)):
        if (sys.argv[_i] == "--config"):
            jsonPath = sys.argv[_i + 1]
        if (sys.argv[_i] == "--no-ramfs"):
            # loadJSONConfiguration's RAMFS auto-redirect (createJSONConfiguration.py
            # redirect_to_ramfs) rewrites dataset paths into a tmpfs mirror and, if that
            # mirror is not yet populated, shells out to scripts/prepareRAMDatasets.sh to
            # copy the datasets into RAM. Two concurrent arms hitting this at once race on
            # the same half-populated cache (observed: "Failed to open file" mid-batch) and
            # the copy itself is an extra, uncapped RAM sink. Skip it for multi-arm runs.
            useRAMfs = False
    if (checkIfFileExists(jsonPath)):
        print(bcolors.OKGREEN, "Loading configuration from file ", jsonPath, bcolors.ENDC)
        from ymapnet.utils.createJSONConfiguration import loadJSONConfiguration
        cfg = loadJSONConfiguration(jsonPath, useRAMfs=useRAMfs)
    else:
        print(bcolors.FAIL, "CREATING FRESH CONFIGURATION!", bcolors.ENDC)
        from ymapnet.utils.createJSONConfiguration import createJSONConfiguration
        cfg = createJSONConfiguration(jsonPath)

    # The model directory is ymapnet_model/ on renamed machines and 2d_pose_estimation/ on
    # legacy ones; resolve it once instead of hardcoding, and rebase the cfg paths that carry
    # a baked-in prefix. MODEL_DIR is treated as READ-ONLY shared state from here on (it holds
    # the embeddings/vocabulary/synonym assets and the canonical packaged tokens model that a
    # concurrent full-net run may be transplanting from) -- every WRITE below goes to OUTPUT_DIR
    # instead, which defaults to MODEL_DIR (old, single-job-on-this-box behaviour) unless
    # --outdir names a private directory for this run.
    MODEL_DIR = resolveModelDir()
    OUTPUT_DIR = MODEL_DIR
    for _i in range(0, len(sys.argv)):
        if (sys.argv[_i] == "--outdir"):
            OUTPUT_DIR = sys.argv[_i + 1]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(bcolors.OKGREEN, "Model directory (shared, read-only assets): using %s/" % MODEL_DIR, bcolors.ENDC)
    print(bcolors.OKGREEN, "Output directory (this run's writes): using %s/" % OUTPUT_DIR, bcolors.ENDC)
    for _k in ('embeddingsPath', 'vocabularyPath'):
        if cfg.get(_k):
            cfg[_k] = rebaseModelPath(cfg[_k], MODEL_DIR)
    if cfg.get('synonymPath'):
        _sp = cfg['synonymPath']
        cfg['synonymPath'] = rebaseModelPath(_sp, MODEL_DIR) if isinstance(_sp, str) \
            else [rebaseModelPath(p, MODEL_DIR) for p in _sp]

    if (cfg['mixedPrecision']):
        print(bcolors.WARNING, "Using mixed precision mode!", bcolors.ENDC)
        keras.mixed_precision.set_global_policy("mixed_float16")

    countResponses = False
    saveRestoredWeights = False
    restoreBestWeights = False
    resumePreviousTraining = False
    elevatePriority = False
    startEpoch = 0
    if (len(sys.argv) > 1):
        #print('Argument List:', str(sys.argv))
        for i in range(0, len(sys.argv)):
            if (sys.argv[i] == "--debug"):
                #Enabling this multiplies training time from 6min to 1hour
                tf.config.run_functions_eagerly(True)  # <- DEBUG
                cfg['batchSize'] = 4  #<- Reduce batch size since eager executio requires more memory
            if (sys.argv[i] == "--flush"):
                # Only ever wipes OUTPUT_DIR (this run's private directory), never the
                # shared MODEL_DIR -- MODEL_DIR may be read concurrently (embeddings,
                # vocabulary, the canonical tokens/model.keras) by another training job.
                os.system("rm -rf %s/ && mkdir -p %s/" % (OUTPUT_DIR, OUTPUT_DIR))
            if (sys.argv[i] == "--novalidation"):
                cfg['doValidation'] = False
            if (sys.argv[i] == "--mem"):
                cfg['datasetUsage'] = float(sys.argv[i + 1])
            #if (sys.argv[i]=="--stream"): Model so large that we always need to stream now
            #   cfg['streamDataset']      = True
            #   cfg['streamBufferLength'] = 1000 #int(sys.argv[i+1])
            if (sys.argv[i] == "--clear") or (sys.argv[i] == "--clean"):
                os.system("rm -rf %s/tensorboard" % OUTPUT_DIR)
                os.system("rm -f %s/tokens.zip" % OUTPUT_DIR)
            if (sys.argv[i] == "--resume") or (sys.argv[i] == "--continue"):
                resumePreviousTraining = True
            if (sys.argv[i] == "--rt"):
                elevatePriority = True
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
            if (sys.argv[i] == "--test"):
                from ymapnet.core.NNModel import build_resnethybrid_cnn
                model = build_resnethybrid_cnn(
                    (cfg['inputWidth'], cfg['inputHeight'], 3),
                    dropoutRate=cfg['dropoutRate'],
                    gloveLayers=cfg['gloveLayers'],
                    bridgeLayerWidthCompatibility=cfg['forceBridgeSize'],
                    multihotLayers=cfg['multihotLayers'],
                    numTokens=cfg['tokensOut'],
                    numClasses=cfg['tokensClasses'],
                    tokenEmbeddingD=cfg.get('tokenEmbeddingD', 300),
                    nextTokenStrength=cfg['nextTokenStrength'],
                    useDescriptors=cfg.get('outputDescriptors', False))
                from ymapnet.core.NNModel import retrieveModelOutputDimensions
                retrieveModelOutputDimensions(model)
                sys.exit(0)
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
            # Resume from THIS run's own output, not the shared MODEL_DIR/tokens/model.keras
            # (which may be a different, concurrently-running experiment's canonical checkpoint).
            model_path = "%s/tokens/model.keras" % OUTPUT_DIR
            from ymapnet.core.NNModel import load_keypoints_model
            model, input_size, output_size, numHeatmaps = load_keypoints_model(model_path)
            cfg['inputWidth'] = input_size[0]
            cfg['inputHeight'] = input_size[1]
            cfg['outputWidth'] = output_size[0]
            cfg['outputHeight'] = output_size[1]
        else:
            print(bcolors.OKGREEN, "Creating a new Token Only Neural Network.. ", bcolors.ENDC)
            #from NNModel import build_simple_cnn
            #model = build_simple_cnn(( cfg['inputWidth'], cfg['inputHeight'], 3), 2037 )
            #from NNModel import build_vit
            #model = build_vit(( cfg['inputWidth'], cfg['inputHeight'], 3), 2037 )
            #from NNModel import build_resnet_cnn
            #model = build_resnet_cnn(( cfg['inputWidth'], cfg['inputHeight'], 3), 2037 )
            from ymapnet.core.NNModel import build_resnethybrid_cnn
            model = build_resnethybrid_cnn(
                (cfg['inputWidth'], cfg['inputHeight'], 3),  # 2037, 
                dropoutRate=cfg['dropoutRate'],
                gloveLayers=cfg['gloveLayers'],
                bridgeLayerWidthCompatibility=cfg['forceBridgeSize'],
                multihotLayers=cfg['multihotLayers'],
                numTokens=cfg['tokensOut'],
                numClasses=cfg['tokensClasses'],
                tokenEmbeddingD=cfg.get('tokenEmbeddingD',300),
                nextTokenStrength=cfg['nextTokenStrength'],
                useDescriptors=cfg.get('outputDescriptors', False))  #<- Experimental

        #----------------------------------------------------------------------------------------

        # Set up TensorBoard logging
        #----------------------------------------------------------------------------------------
        log_dir = OUTPUT_DIR + "/tensorboard/" + cfg["serial"] + "_" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)  # histograms cost per-epoch RAM; disabled on 31GB boxes

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
                addSuperpoint=int(cfg.get('heatmapAddSuperpoint', False)),
                superpointChannels=int(cfg.get('superpointChannels', 0)),
                superpointPcaPath=cfg.get('superpointPcaFile', ''),
                elevatePriority=elevatePriority,
                datasets=cfg["ValidationDataset"],
                vocabularyPath=cfg.get('vocabularyPath', None),
                synonymPath=cfg.get('synonymPath', None),
                embeddingsPath=cfg.get('embeddingsPath', None),
                libraryPath="datasets/DataLoader/libDataLoader.so")
            if (cfg['streamValidation']):
                # Modify the way you call the dataset
                validation_generator = TrainingDataGenerator(
                    cfg=cfg, db=dbValidation, batch_size=cfg['batchSize'], validation_data=True,
                    returnOutputImages=False, numberOfTokens=cfg["tokensOut"], numberOfClasses=cfg["tokensClasses"],
                    workers=1, use_multiprocessing=False,
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
                addSuperpoint=int(cfg.get('heatmapAddSuperpoint', False)),
                superpointChannels=int(cfg.get('superpointChannels', 0)),
                superpointPcaPath=cfg.get('superpointPcaFile', ''),
                elevatePriority=elevatePriority,
                datasets=cfg["TrainingDataset"],
                vocabularyPath=cfg.get('vocabularyPath', None),
                synonymPath=cfg.get('synonymPath', None),
                embeddingsPath=cfg.get('embeddingsPath', None),
                libraryPath="datasets/DataLoader/libDataLoader.so")
            dbTrain.updateJointDifficulty(cfg['keypoint_difficulty'])
            if (cfg['streamDataset']):
                # Modify the way you call the dataset
                outLabels = dbTrain.get_labels()
                dataset_generator = TrainingDataGenerator(
                    cfg=cfg, db=dbTrain, batch_size=cfg['batchSize'], labels=outLabels, log_dir=log_dir,
                    returnOutputImages=False, numberOfTokens=cfg["tokensOut"], numberOfClasses=cfg["tokensClasses"],
                    workers=1, use_multiprocessing=False,
                    max_queue_size=4)  #multiprocessing happens inside the dataloader
                trainingDatasetLength = dbTrain.numberOfSamples
                shuffleData = False  #<- shuffling is done inside the generator
        #----------------------------------------------------------------

        # Print the shapes of inputs and outputs
        print("Training Configuration :", cfg)
        logText(cfg, log_dir)

        # Descriptors are optional here, but when they ARE on a source without a .dinov3
        # sidecar silently yields all-zero vectors that would train the head to regress 0.
        # Refuse before the first step rather than produce a mysteriously worse model.
        if cfg.get('outputDescriptors', False):
            assertDescriptorsArePopulated(dbTrain, "training set", cfg)
            if dbValidation is not None and dbValidation is not dbTrain:
                assertDescriptorsArePopulated(dbValidation, "validation set", cfg)

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
        from ymapnet.core.NNLosses import ConditionalModelCheckpoint
        checkpointer = ConditionalModelCheckpoint(
            monitor=whatToMonitor,
            mode=howToMonitor,
            # Both the weights file and the status file live under OUTPUT_DIR: the class
            # otherwise hardcodes "status.txt" in the CWD, which would collide with (and
            # get overwritten by) a concurrent trainYMAPNet.py run's own status.txt.
            filepath=OUTPUT_DIR + "/best.weights.h5",
            #filepath="checkpoint_epoch_{epoch:02d}.weights.h5",  # Corrected file extension
            save_best_only=True,
            save_weights_only=True,
            start_from_epoch=0,
            verbose=1,
            total_epochs=cfg['epochs'],
            serial=cfg['serial'],  # avoid falling back to reading the shared root configuration.json
            status_path=OUTPUT_DIR + "/status.txt")
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
        #optimizer = AdamWCautious(learning_rate=float(cfg['learningRate']),clipnorm=None,clipvalue=1.0)
        #optimizer = tf.keras.optimizers.Adam(learning_rate=float(cfg['learningRate']))
        optimizer = getOptimizerFromCFG(cfg)

        #Decide on heatmap loss based on configuration
        hmloss = None
        if (cfg['loss'] == "mse"):
            hmloss = VanillaMSELossSimple()
        else:
            print(bcolors.WARNING, "Using generic loss, will probably not work well..", bcolors.ENDC)
            hmloss = cfg['loss']

        #Compile a model with the requested loss
        if ("outputTokens" in cfg) and (cfg["outputTokens"]):
            #token_loss_function = token_mse_loss
            #Instead of categorical cross-entropy (used in single-label classification), binary cross-entropy is used as the loss function for multi-label tasks. This allows the model to handle each class independently.
            #token_loss_function = keras.losses.CategoricalCrossentropy(from_logits=False) #<- One Class
            #token_loss_function = keras.losses.BinaryCrossentropy(from_logits=False)       #<- Multiple Classes
            #model.compile(optimizer=optimizer,  loss=weighted_token_loss, metrics=[ 'accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)])

            print("Now retrieving token weights!")
            weight_array = dbTrain.get_token_frequencies()
            # Ported from trainYMAPNet.py (serial 286+): the raw inverse-frequency weights
            # (totalSum/count, see db_count_description_token_weight in DataLoader.c) span many
            # orders of magnitude -- a token appearing once in the whole training set gets a
            # weight in the hundreds of thousands. Left unclipped, any batch containing such a
            # rare positive injects an enormous weighted_loss term into WeightedFocalLoss/
            # AsymmetricLoss, which reliably destabilised training a handful of epochs into the
            # unfreeze phase (K4 focal/ASL runs, TOKENS.md) -- this, not the BN/LR issues fixed
            # earlier, was the dominant cause. Anchor on the most-frequent real class (-> 1.0)
            # so relative ratios are preserved, then clip so the tail is bounded.
            if cfg.get('multihotClassWeightNormalize', False):
                wa = np.asarray(weight_array, dtype=np.float32)
                real = wa[wa > 1.0]
                anchor = float(real.min()) if real.size else 1.0
                lo, hi = cfg.get('multihotClassWeightClip', [0.05, 8.0])
                normalized = np.clip(wa / (anchor + 1e-9), float(lo), float(hi))
                normalized[wa <= 1.0] = float(lo)     # count==0 / blacklisted -> floor (unlearned)
                weight_array = normalized
                print("  multihot class weights anchored (most-freq->1) + clipped to [%.3f, %.3f]" % (float(lo), float(hi)))
            class_weight_dict = {i: float(weight) for i, weight in enumerate(weight_array)}

            #-------------------------------------------------------------------------------------
            from ymapnet.core.NNLosses import GloVeMSELoss, GloVeCosineLoss, GloVeHybridLoss, MultiHotLoss, WeightedBinaryCrossEntropy, WeightedFocalLoss, AsymmetricLoss
            losses = dict()
            for i in range(cfg["tokensOut"]):
                #losses["t%02u"%i] = GloVeMSELoss(weight=1.0)
                losses["t%02u" % i] = GloVeHybridLoss(mse_weight=1.0, cosine_weight=1.0)

            #losses['tokens_multihot']  = WeightedBinaryCrossEntropy(weight_array)  #MultiHotLoss()
            # multihotLossFunction: "focal" (default, unchanged historical behaviour) or "asl"
            # (TOKENS.md Experiment K4 -- Asymmetric Loss, Ridnik et al. 2021; targets the
            # Obs-26 dominant-class suppression that WeightedFocalLoss's single gamma does not).
            multihotLossFunction = cfg.get("multihotLossFunction", "focal")
            if multihotLossFunction == "asl":
                print(bcolors.OKGREEN, "Using AsymmetricLoss (K4) for tokens_multihot", bcolors.ENDC)
                losses['tokens_multihot'] = AsymmetricLoss(
                    weight_array,
                    gamma_neg=cfg.get("aslGammaNeg", 4.0),
                    gamma_pos=cfg.get("aslGammaPos", 1.0),
                    clip=cfg.get("aslClip", 0.05))
            else:
                losses['tokens_multihot'] = WeightedFocalLoss(weight_array)  #<- Try focal loss

            if (cfg.get("outputDescriptors", False)):
                # Match trainYMAPNet exactly: DescriptorLoss (MSE + cosine + norm-reg), weight
                # from lossWeightDescriptors. Plain "mse" was the old wiring, but MSE alone
                # drives a regression head toward the CENTROID of the target distribution -
                # the documented failure of the token cascade (TOKENS.md 5.1/10.6) - and the
                # DINO targets are directional, so cosine is what matters.
                from ymapnet.core.NNLosses import DescriptorLoss
                losses['descriptors'] = DescriptorLoss(weight=cfg.get('lossWeightDescriptors', 1.0))
            #-------------------------------------------------------------------------------------
            loss_weights = dict()
            for i in range(cfg["tokensOut"]):
                loss_weights["t%02u" % i] = cfg["lossWeightGloveTokens"] / cfg["tokensOut"]

            loss_weights['tokens_multihot'] = cfg["lossWeightMultihotTokens"]  #0.001

            if (cfg.get("outputDescriptors", False)):
                # The gain lives inside DescriptorLoss (as in trainYMAPNet, which passes
                # loss_weights=None), so this stays 1.0. The previous hardcoded 10.0 gave the
                # auxiliary head ~8x the per-token weight and 10x the multihot weight.
                loss_weights['descriptors'] = 1.0
            #-------------------------------------------------------------------------------------
            from ymapnet.core.NNLosses import HeatmapDistanceMetricPartial, CustomTopKCategoricalAccuracy
            metrics = dict()
            for i in range(cfg["tokensOut"]):
                metrics["t%02u" % i] = keras.metrics.CosineSimilarity(name='cossim', axis=1)

            if (cfg.get("outputDescriptors", False)):
                from ymapnet.core.NNLosses import CosineSimilarityMetric
                metrics['descriptors'] = CosineSimilarityMetric(name='desc_cossim', dtype=tf.float32, axis=1)

            metrics['tokens_multihot'] = [
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(name="top3_accuracy", k=3),
                tf.keras.metrics.TopKCategoricalAccuracy(name="top5_accuracy", k=5)
            ]
            #-------------------------------------------------------------------------------------

            model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics,
                          jit_compile=False)  # XLA autoclustering (TF2.19 default) explodes compile-time RAM ~13GB on 31GB boxes
    #--------------------------------------------------------------------------------------------------------------------------------

    #Printout data/size summaries in screen
    #--------------------------------------------------------------------------------------------------------------------------------
    bytesPerValue = 1  # np.int8
    channels = deriveRGBChannelsFromCFG(cfg)
    heatmapNumber = cfg['outputChannels']
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
        model = keras.saving.load_model("%s/model.keras" % OUTPUT_DIR, custom_objects={
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

    import json

    #print("Validation data copy train blacklist!")
    #dbValidation.tokenblacklistkeys = dbTrain.tokenblacklistkeys
    #dbValidation.update_token_blacklist(dbValidation.tokenblacklistkeys,lowThreshold=2)
    print("Total number of train black listed keys : ", len(dbTrain.tokenblacklistkeys))
    print("Total number of validation black listed keys : ", len(dbValidation.tokenblacklistkeys))

    print("Now retrieve token weights!")
    weight_array = dbTrain.get_token_frequencies()

    # Convert the weight_array into a dictionary
    class_weight_dict = {i: float(weight) for i, weight in enumerate(weight_array)}
    with open("%s/class_weight_dict_new.json" % OUTPUT_DIR, 'w') as json_file:
        json.dump(class_weight_dict, json_file, indent=4)
    #sys.exit(0)

    if (cfg["epochsFrozen"]):
        history = model.fit(
            distributedTrainingDataset,
            batch_size=cfg['batchSize'],
            epochs=cfg["epochsFrozen"],
            validation_data=distributedValidationDataset,
            initial_epoch=startEpoch,
            shuffle=shuffleData,
            verbose=2,  # one line per epoch keeps the log small (progress bars spam MBs)
            #class_weight     = class_weight_dict,  # Pass the calculated weights here <- this gets now done via loss_weights
            callbacks=[tensorboard_callback, early_stopping, checkpointer, lr_callback,
                       DataAugmentation(cfg, dbTrain), MemoryReclaim()])
        checkpointer.load_best_model()

        #Log best epoch / loss
        #--------------------------------------------------------------------------------------------------------------------------------
        finishDetails = dict()
        finishDetails["BestEpoch"] = checkpointer.bestEpoch
        finishDetails["Best%s" % cfg['earlyStoppingMonitor']] = checkpointer.best
        finishDetails["BestLog"] = checkpointer.bestLog
        print("Checkpointer Accepted Freezed Solution :", finishDetails)
        logText(finishDetails, log_dir, subject="CheckpointerAcceptedFreezedSolution")
        #--------------------------------------------------------------------------------------------------------------------------------

    # Step 2: Unfreeze the base model
    print("Unfreeze")
    model.get_layer('resnet50').trainable = True  # 'resnet50' or 'convnext_small' should be the name of the base model

    #------------------------------------------------------------------------------------------
    optimizer = getOptimizerFromCFG(cfg)
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics,
                  jit_compile=False)  # XLA autoclustering (TF2.19 default) explodes compile-time RAM ~13GB on 31GB boxes

    # This fit() call has no initial_epoch, so Keras' own epoch counter (and thus
    # lr_callback, which is a pure function of that counter) restarts at 0 right when
    # the ResNet50 trunk becomes fully trainable -- the LR schedule jumps back up near
    # its starting value instead of continuing its decay from where the frozen phase
    # left off. That combination (settled pretrained BatchNorm stats + newly-trainable
    # conv weights + a re-spiked LR) reliably blew up training a few epochs into the
    # unfreeze phase (see K4 focal/ASL runs, TOKENS.md). Use a separate callback that
    # offsets the epoch by epochsFrozen so the LR curve stays continuous across the
    # unfreeze boundary; the fit() call itself keeps epoch 0-indexed so status.txt /
    # checkpointer epoch numbering (X/cfg['epochs']) is unaffected.
    lr_callback_finetune = tf.keras.callbacks.LearningRateScheduler(lambda epoch: custom_lr_scheduler(
        epoch + cfg.get("epochsFrozen", 0), cfg['learningRateStart'], cfg['learningRateEnd']))

    #Epochs can be set to zero to just do post training training
    #--------------------------------------------------------------------------------------------------------------------------------
    if (cfg['epochs'] > 0):
        history = model.fit(
            distributedTrainingDataset,
            batch_size=cfg['batchSize'],
            epochs=cfg['epochs'],
            validation_data=distributedValidationDataset,
            shuffle=shuffleData,
            verbose=2,  # one line per epoch keeps the log small (progress bars spam MBs)
            #class_weight     = class_weight_dict,  # Pass the calculated weights here  <- this gets now done via loss_weights
            callbacks=[tensorboard_callback, early_stopping, checkpointer, lr_callback_finetune,
                       DataAugmentation(cfg, dbTrain), MemoryReclaim()])
        checkpointer.load_best_model()

        #Log best epoch / loss
        #--------------------------------------------------------------------------------------------------------------------------------
        finishDetails = dict()
        finishDetails["BestEpoch"] = checkpointer.bestEpoch
        finishDetails["Best%s" % cfg['earlyStoppingMonitor']] = checkpointer.best
        finishDetails["BestLog"] = checkpointer.bestLog
        print("Checkpointer Accepted Freezed Solution :", finishDetails)
        logText(finishDetails, log_dir, subject="CheckpointerAcceptedFinalSolution")
        #--------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------
    print(bcolors.WARNING, "Recompiling model using defaults to make it more portable..", bcolors.ENDC)
    # jit_compile=False here too: the portability recompile is followed by
    # extract_validation_losses(), which runs the model again — with XLA autoclustering back
    # on (TF2.19 default) that re-triggers a full fused-GEMM/cuDNN autotune of the whole
    # ResNet+heads graph. On this box that hung for 11h AFTER the model was already saved
    # (arm1, 2026-09-05), blocking the queue behind it.
    model.compile(optimizer="adam", loss="mse", jit_compile=False)

    # Perform any model optimizations requested by configuration and then save and package everything
    #--------------------------------------------------------------------------------------------------------------------------------
    if (cfg['pruneModel']):
        from ymapnet.core.NNOptimize import pruneModel
        model = pruneModel(model, cfg, trainingDataset)

    if (cfg['clusterModel']):
        from ymapnet.core.NNOptimize import clusterModel
        model = clusterModel(model, cfg, trainingDataset)

    saveNNModel("%s/tokens" % OUTPUT_DIR, model,
                formats=["keras"])  #Only use keras format to save space! formats=["keras","tf","tflite","onnx"]

    #Promote the training config into the packaged runtime dir now that the model has been
    #successfully saved. Runtime reads <OUTPUT_DIR>/tokens.json. Copies whatever --config
    #pointed at (jsonPath), not a hardcoded 'tokens.json', so named experiment configs promote
    #their own file rather than the tracked default.
    print(bcolors.OKGREEN, "Promoting %s -> %s/tokens.json" % (jsonPath, OUTPUT_DIR), bcolors.ENDC)
    os.system("cp %s %s/tokens.json" % (jsonPath, OUTPUT_DIR))

    #Dump training sample report
    dbTrain.dump_sample_report("%s/sample_report_training_tokens.json" % OUTPUT_DIR)

    if (history):
        logTrainingHistory(cfg, OUTPUT_DIR, "loss_history.txt", history)
    if (historyFT):
        logTrainingHistory(cfg, OUTPUT_DIR, "loss_finetune_history.txt", historyFT)

    #No longer save vocabulary to not overwrite it by mistake
    #os.system("cp datasets/descriptions/index_to_word.json 2d_pose_estimation/vocabulary.json")

    #Save model as INT8 TF-Lite (Disabled because it takes too much space and not currently needed)
    #if (not onlyTrainingData):
    #   print(bcolors.WARNING,"We have a validation set, so saving TF-Lite INT8 model..",bcolors.ENDC)
    #   from NNConverter import saveNNTFLiteINT8Model, saveNNTFLiteFP16Model
    #   saveNNTFLiteINT8Model(model,dbValidation.get_in_array())
    #   saveNNTFLiteFP16Model(model,dbValidation.get_in_array())

    # Package output
    #--------------------------------------------------------------------------------------------------------------------------------
    print(bcolors.OKGREEN, "Training complete..", bcolors.ENDC)
    # zipName is a SIBLING of OUTPUT_DIR (never nested inside it), same convention as the
    # original MODEL_DIR-based naming, so `zip -r` never tries to include its own output.
    zipName = OUTPUT_DIR.rstrip('/').replace('/', '_') + "_tokens.zip"
    os.system("date +\"%%y-%%m-%%d_%%H-%%M-%%S\" > %s/date.txt" % OUTPUT_DIR)  #Tag date
    os.system("rm -f %s" % zipName)
    os.system("zip -r %s %s/ -x %s/model.keras" % (zipName, OUTPUT_DIR, OUTPUT_DIR)
              )  #Create zip of models (don't include main model)
    #--------------------------------------------------------------------------------------------------------------------------------

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if (not onlyTrainingData):
        weight_val_array = dbValidation.get_token_frequencies()
        extract_validation_losses(model, validation_generator, dbValidation)
        dbValidation.dump_sample_report("%s/sample_report_validation_tokens.json" % OUTPUT_DIR)
        print("Done Dumping Validation Samples ")
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    print(
        'You can see a summary using :\n tensorboard --logdir=%s/tensorboard --bind_all && firefox http://127.0.0.1:6006'
        % OUTPUT_DIR)

    # Upload results
    #--------------------------------------------------------------------------------------------------------------------------------
    print('To upload results (if you take too long it will timeout) :')
    os.system("timeout 5 scripts/uploadResults.sh")
    print("scp -P 2222 %s ammar@ammar.gr:/home/ammar/public_html/poseanddepth" % zipName)
    print(" or ")
    print(
        "scp -P 2222 %s ammar@ammar.gr:/home/ammar/public_html/poseanddepth/archive/%s_v%s.zip"
        % (zipName, OUTPUT_DIR.rstrip('/').replace('/', '_'), str(cfg['serial'])))
