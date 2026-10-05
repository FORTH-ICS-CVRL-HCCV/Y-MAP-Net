"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""
import sys
import os
import json


#============================================================================================
def makeJSONConfiguration():
    cfg = {
        'serial':
        'unknown',
        'GPUsUsedForTraining': ["/gpu:0"],  #, "/gpu:3"
        'DatasetLoaderThreads':
        30,
        'TrainingDataset': [
            [
                "datasets/generated/generatedTrain.db", "datasets/generated/data/train",
                "datasets/generated/data/depth_train", "datasets/generated/data/segment_train"
            ],
            [
                "datasets/openpose/openposeTrain.db", "datasets/openpose/data/train",
                "datasets/openpose/data/depth_train", "datasets/openpose/data/segment_train"
            ],  #<- for some reason this needs to be first!                           
            [
                "datasets/background/AM-2k.db", "datasets/background/AM-2k/train",
                "datasets/background/AM-2k/depth_train", "datasets/background/AM-2k/segment_train"
            ],
            [
                "datasets/background/BG-20k.db", "datasets/background/BG-20k/train",
                "datasets/background/BG-20k/depth_train", "datasets/background/BG-20k/segment_train"
            ],
            [
                "datasets/coco/cocoTrain.db", "datasets/coco/cache/coco/train2017",
                "datasets/coco/cache/coco/depth_train2017", "datasets/coco/cache/coco/segment_train2017"
            ]
        ],
        'ValidationDataset': [[
            "datasets/coco/cocoVal.db", "datasets/coco/cache/coco/val2017", "datasets/coco/cache/coco/depth_val2017",
            "datasets/coco/cache/coco/segment_val2017"
        ]],
        'inputWidth':
        400,  #256 #Avg 570
        'inputHeight':
        400,  #256 #Avg 480
        'outputWidth':
        384,  #256
        'outputHeight':
        384,
        'outputChannels':
        17,  # 17 + 2
        'output16BitChannels':
        1,  # 17 + 2
        'outputTokens':
        False,
        'outputDescriptors':
        False,
        'heatmapLossImportanceRelativeToTokens':
        1.0,
        'model':
        'unet',  #unet/conv
        'dropoutRate':
        0.35,
        'activation':
        'leaky_relu',  #relu
        'baseChannels':
        21,  # 64 default <- for unet/mobilenet
        'pixelwiseChannels':
        1200,  # 64 default <- for unet/mobilenet
        'bridgeRatio':
        0.35,  # 1.0 default
        'encoderRepetitions':
        7,  # 4 default <- for unet 
        'decoderRepetitions':
        7,  # 4 default <- for unet
        'maxDecoderChannels':
        0,  #Cap on deep-decoder width (Experiment O / PLAN.md Obs 20); 0 = unclamped legacy layout
        'midSectionRepetitions':
        3,  #<- for mobilenet
        'gloveLayers':
        7,
        'multihotLayers':
        3,
        'learnableTokenResiduals':
        True,
        'mixedPrecision':
        False,  #This leads to nan loss during training
        'quantizeModel':
        False,
        'pruneModel':
        False,
        'clusterModel':
        False,
        'RGBMagnitude':
        255,
        'RGBgaussianNoiseSTD':
        0.0,
        'RGBImageEncoding':
        'rgb24',
        'heatmapActive':
        120,  #Max 127 for np.int8
        'heatmapDeactivated':
        -120,  #Min -128 for np.int8
        'heatmapTanhHeadroom8bit':
        1.0,  #Multiplier on the 8-bit tanh->raw Rescaling (NNModel 'hm' head: joints/PAFs/segms/etc). >1 lifts the output ceiling above heatmapActive so the GT active peak lands at tanh=1/headroom (linear region) instead of the unreachable tanh=1.0 asymptote — cures joint-peak amplitude undershoot without touching the int8 GT encoding. 1.0 = historical behaviour. (PLAN.md Exp T)
        'heatmapTanhHeadroom16bit':
        1.0,  #Same, for the 16-bit 'hm_16b' head (depth): >1 lifts the 32767 ceiling so far-depth values (GT near ±32767) are reachable off the tanh=1.0 asymptote. 1.0 = historical behaviour. (PLAN.md Exp T)
        'heatmapGradientSize':
        23,
        'heatmapGradientSizeMinimum':
        6,
        'heatmapPAFSize':
        5,
        'heatmapPAFSizeMinimum':
        2,
        'heatmapReductionEpochLimit':
        10,
        'heatmapRuler':
        True,  #Add a ruler in first row/column of heatmaps
        'heatmapAddPAFs':
        True,  #Add PAFs 
        'heatmapAddDepthmap':
        True,  #Add a depth map 
        'heatmapAddNormals':
        True,  #Add normals
        'heatmapAddSegmentation':
        True,  #Add segmentation
        'heatmapAddInstanceDetection':
        True,  #Add per-instance person center+size (CenterNet-style) channels (+3)
        'heatmapAddSuperpoint':
        False,  #Add SuperPoint PCA-descriptor heatmap channels (+superpointChannels). Default off until matured.
        'superpointChannels':
        3,  #K = number of PCA components / heatmap channels rendered from .superpoint descriptors
        'superpointPcaFile':
        "datasets/superpoint/superpoint_pca.bin",  #Frozen global PCA basis (datasets/superpoint/fit_superpoint_pca.py)
        'heatmapAddGeolocation':
        False,  #Add the global lat/lon geolocation density channel (+1 to outputChannels). Default off until matured. (GEOLOCATION.md)
        'heatmapAddDepthNormalsUncertainty':
        True,  #Add the aleatoric uncertainty (hm_nll) head: per-pixel log-variance for depth+normals
        'heatmapGenerateSkeletonBkg':
        False,  #BKG channel is not just points but also has skeleton
        'heatmapAlternatePattern':
        False,
        #NOTE: since the Experiment N1 linearization (PLAN.md Obs 18) the lossWeight* gains
        #are LINEAR multipliers of each group's MSE (they previously multiplied inside the
        #square, i.e. effective weight = gain^2). Old-semantics values must be squared —
        #the fixLossGainLinearization marker + migration guard below handle this.
        'fixLossGainLinearization':
        True,
        'lossPenaltyForegroundNormalized':
        False,  #Experiment N2 (PLAN.md Obs 19): FN penalties normalise over foreground pixels only; retune lossPenaltyGain sharply downward when enabling
        'lossWeightJoints':
        4.0,  #= old 2.0 squared (linearized semantics)
        'lossWeightPAFs':
        1.0,
        'lossWeightDepth':
        1.0,
        'lossWeightNormals':
        1.0,
        'lossWeightText':
        1.0,
        'lossWeightSegmentation':
        4.0,  #= old 2.0 squared (linearized semantics)
        'lossWeightInstance':
        1.0,
        'lossWeightSuperpoint':
        1.0,  #Per-group gain for the SuperPoint heatmap MSE in HeatmapCoreLoss
        'lossWeightGeolocation':
        1.0,  #Per-group gain for the geolocation heatmap MSE in HeatmapCoreLoss
        'useGeolocationHead':
        False,  #Experiment U: dedicated global-pooled geo head off the bridge (softmax world-density + KL loss), replacing the collapsed shared-decoder geo channel. GT reused from the 8-bit geo channel. Default off. (PLAN.md Exp U)
        'geoHeadHidden':
        512,  #Hidden width of the geolocation head's dedicated feature transform
        'lossWeightGeolocationHead':
        1.0,  #Weight on the geo_grid KL loss
        'geoGridHeight':
        64,  #geo_grid head output rows (lat) = teacher NATIVE grid (256->64, /4). Smaller than 256 to cut overfit.
        'geoGridWidth':
        128,  #geo_grid head output cols (lon) = teacher NATIVE grid (256->128, /2)
        'geoConcentrationWeighting':
        False,  #Experiment 281: down-weight ambiguous samples by GT concentration (D5 soft-mask). Applied INSIDE GeolocationKLLoss (derived from y_true; a Keras sample_weight dict breaks list-output resolve_path), BATCH-MEAN-NORMALISED so only relative emphasis shifts (magnitude/val-scale preserved), train+val alike. Default off.
        'geoLocalizableThreshold':
        0.005,  #Experiment 281: GT concentration cutoff for the localizable-subset metric geo_acc2500_loc. Calibrated to the PLONK-on-COCO val distribution (max ~0.03, median ~0.0037, uniform floor 1/(H*W)~1.2e-4): 0.005 keeps the top ~25%. Re-check with diagnose_geolocation.py --histogram if the geo data changes.
        'useDepthwiseSeparable':
        False,  #Experiment R1 (PLAN.md): route the encoder/decoder feature path through depthwise-separable convnext_blocks (DW 7x7 -> PW 1x1 4x -> PW 1x1) instead of the full-conv conv_block. ~2x fewer backbone FLOPs; retrains from scratch (layer names change). Default off = full-conv baseline.
        'useCheckpointAveraging':
        False,  #U1 (PLAN.md): post-training, average the top-N best-by-monitor checkpoints (same run), recalibrate BatchNorm, adopt only if it beats the best single epoch. Default off = ship best epoch. Enable first on serial 283.
        'swaNumCheckpoints':
        5,  #U1/CA1: how many best-by-monitor checkpoints to retain and average.
        'swaBNRecalibBatches':
        200,  #U1/CA3: training-mode forward passes used to re-accumulate BatchNorm running stats after averaging (MANDATORY, BN-heavy model).
        'swaMinEpochFraction':
        0.5,  #U1/CA1: only epochs past this fraction of total epochs are eligible for the averaging pool (skip early, pre-plateau iterates).
        'swaAcceptOnlyIfBetter':
        True,  #U1/CA4: adopt the averaged net only if it beats the best single epoch on the monitored metric; else fall back to best-epoch weights.
        'lossWeightDepthNormalsUncertainty':
        1.0,  #Weight for the DepthNormalsNLLLoss on the hm_nll head
        'lossWeightGloveTokens':
        1.0,
        'lossWeightMultihotTokens':
        0.001,
        'lossWeightDescriptors':
        1.0,
        'streamDataset':
        True,
        'streamValidation':
        False,
        'dataLoaderDoubleBuffer':
        False,  #[B1] overlap C batch prep with the GPU step (DATALOADER.md B1); streaming-only, default off until validated per-machine
        # ('dataLoaderZeroCopy' used to be defined here — removed; the zero-copy view path was
        #  unfixable against the 32-deep prefetch queue and saved nothing, see DATALOADER.md B2)
        'streamBufferLength':
        2000,
        'earlyStoppingMonitor':
        'val_loss',
        'earlyStoppingHowToMonitor':
        'min',
        'earlyStoppingPatience':
        45,
        'earlyStoppingMinDelta':
        0.0001,
        'earlyStoppingStart':
        25,
        'dataAugmentation':
        True,
        'augmentation': {
            'chanceDestroy':                   0.0,
            'chancePerturbed':                 35.0,
            'chancePanAndZoom':                45.0,
            'chanceBurnedPixels':              50.0,
            'chanceBrightnessContrast':        50.0,
            'chanceBrightnessContrastUniform': 50.0,
            'chanceHorizontalFlip':            0.0,
            'chanceRotate90':                  10.0,
            'chanceCoarseDropout':             30.0,
            'chanceGaussianBlur':              20.0,
            'chanceDefocusBlur':               15.0,
            'chanceMotionBlur':                15.0,
            'maxZoomFactor':                   1.1,
            'perturbationMagnitude':           100,
            'maxBurnedPixels':                 10,
            'minBrightnessChange':            -55.0,
            'maxBrightnessChange':             55.0,
            'minRelContrastChange':            0.8,
            'maxRelContrastChange':            1.2,
            'minUniformBrightnessChange':     -100.0,
            'maxUniformBrightnessChange':      120.0,
            'coarseDropoutHolesMin':           1,
            'coarseDropoutHolesMax':           8,
            'coarseDropoutMinSize':            8,
            'coarseDropoutMaxSize':            24,
            'gaussianBlurSigmaMin':            0.5,
            'gaussianBlurSigmaMax':            3.0,
            'defocusBlurRadiusMin':            1,
            'defocusBlurRadiusMax':            5,
            'motionBlurLengthMin':             5,
            'motionBlurLengthMax':             15,
        },
        'datasetUsage':
        1.0,
        'learningRate':
        0.0004,
        'learningRateStart':
        0.001,
        'learningRateEnd':
        0.00015,
        'batchSize':
        46,
        'epochs':
        200,
        'loss':
        'mse',  #mse, combine
        'doPostTrainingTraining':
        True,
        'postTrainingEpochs':
        3,
        'doValidation':
        True,
        'ignoreNoSkeletonTrainingSamples':
        False,  #True
        'logImagesFromValidationSet':
        True,
        'logImagesAlsoOutsideOfTensorboard':
        False,  #Also save heatmap files in current directory
        'keypoint_names': [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
            "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ],
        'keypoint_difficulty': [
            -1,  #"nose",
            -1,  #"left_eye",
            -1,  #"right_eye",
            0,  #"left_ear",
            0,  #"right_ear",
            0,  #"left_shoulder",
            0,  #"right_shoulder",
            2,  #"left_elbow",
            2,  #"right_elbow",
            4,  #"left_wrist",
            4,  #"right_wrist",
            0,  #"left_hip",
            0,  #"right_hip",
            2,  #"left_knee",
            2,  #"right_knee",
            4,  #"left_ankle",
            4,  #"right_ankle"
        ],
        'keypoint_parents': {  #THIS NEEDS WORK NOSE SHOULD BE CONNECTED TO HIP
            "nose": "nose",
            "left_eye": "nose",
            "right_eye": "nose",
            "left_ear": "left_eye",
            "right_ear": "right_eye",
            "left_shoulder": "nose",
            "right_shoulder": "nose",
            "left_elbow": "left_shoulder",
            "right_elbow": "right_shoulder",
            "left_wrist": "left_elbow",
            "right_wrist": "right_elbow",
            "left_hip": "nose",
            "right_hip": "nose",
            "left_knee": "left_hip",
            "right_knee": "right_hip",
            "left_ankle": "left_knee",
            "right_ankle": "right_knee"
        },
        'keypoint_children': {
            "nose": ["left_eye", "right_eye", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
            "left_eye": ["left_ear"],
            "right_eye": ["right_ear"],
            "left_ear": [],
            "right_ear": [],
            "left_shoulder": ["left_elbow"],
            "right_shoulder": ["right_elbow"],
            "left_elbow": ["left_wrist"],
            "right_elbow": ["right_wrist"],
            "left_wrist": [],
            "right_wrist": [],
            "left_hip": ["left_knee"],
            "right_hip": ["right_knee"],
            "left_knee": ["left_ankle"],
            "right_knee": ["right_ankle"],
            "left_ankle": [],
            "right_ankle": []
        },
        'paf_parents': [0, 0, 0, 0, 0, 28, 19, 27, 18, 26, 17, 25, 22, 24, 21, 23, 20]
    }
    return cfg


#============================================================================================
def saveJSONConfiguration(cfg, json_file_path):
    base_path = os.path.dirname(json_file_path)
    if (not os.path.exists(base_path)):
        print("Path ", base_path, " does not exist, creating it")
        os.makedirs(base_path)
    with open(json_file_path, 'w') as json_file:
        json.dump(cfg, json_file, indent=4)
    return cfg


#============================================================================================
# Where the .pzpd archives (convertToPZPD.py) live on this box -- prepareRAMDatasets.sh mirrors
# every archive directory referenced by configuration.json from here into <ramfs>/YMAPNet/<name>/.
#============================================================================================
PZPD_SOURCE_ROOT = "/storage/ammarkov/YMAPNet"


#============================================================================================
#Attempt to change .pzpd archive paths to their RAMFS copy if one is detected and populated,
#otherwise on normal systems (or for non-.pzpd sources, which are read directly from wherever
#they are -- no RAMFS staging for the legacy datasets/*.db + directory sources any more) do
#nothing.
#============================================================================================
def redirect_to_ramfs(cfg, ramfs="../ram/"):
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    ramfs_pzpd_root = os.path.join(ramfs, "YMAPNet")

    if os.path.exists(ramfs):
        print("RAMFS detected, checking if it is populated")
        if not os.path.exists(ramfs_pzpd_root):
            print("Did not find .pzpd archives in RAMFS, running script to copy them (?)")
            os.system("scripts/prepareRAMDatasets.sh")

        for key in ('TrainingDataset', 'ValidationDataset'):
            for entryID, entry in enumerate(cfg.get(key, [])):
                for i in range(len(entry)):
                    path = entry[i]
                    # Only .pzpd archive paths under PZPD_SOURCE_ROOT are RAMFS-backed; every
                    # other entry element (the enabled/disabled flag, a legacy .db/directory
                    # entry, or a .pzpd archive that lives somewhere else) is left untouched.
                    if not (isinstance(path, str) and path.endswith(".pzpd")
                            and path.startswith(PZPD_SOURCE_ROOT + "/")):
                        continue
                    rel_path = os.path.relpath(path, PZPD_SOURCE_ROOT)
                    new_path = os.path.join(ramfs_pzpd_root, rel_path)

                    if os.path.exists(new_path):
                        print("Redirecting ", cfg[key][entryID][i], " to ", new_path)
                        cfg[key][entryID][i] = new_path
                    else:
                        print(FAIL, "Cannot redirect ", cfg[key][entryID][i], " to ", new_path, ENDC)
    else:
        print("Did not find a RAMFS, continuing using regular filesystem for datasets")
    return cfg


#============================================================================================
def loadJSONConfiguration(json_file_path, useRAMfs=True):
    cfg = dict()
    # Read the JSON file and load its contents into a dictionary
    with open(json_file_path, 'r') as json_file:
        cfg = json.load(json_file)

    #Populate as index matrix
    keypoint_names = cfg['keypoint_names']
    keypoint_parents = cfg['keypoint_parents']
    keypoint_parent_ids = list()
    for i in range(len(keypoint_names) - 1):  #ignore bkg
        parent_name = keypoint_parents[keypoint_names[i]]
        parent_index = keypoint_names.index(parent_name)
        keypoint_parent_ids.append(int(parent_index))
    cfg['keypoint_parents_ids'] = keypoint_parent_ids

    if not 'bridgeRatio' in cfg:
        cfg['bridgeRatio'] = 1.0

    if not 'heatmapAddDepthLevels' in cfg:
        cfg['heatmapAddDepthLevels'] = 4

    # Backward-compat: existing configs (and packaged model configs) predate the
    # instance center+size head — default it OFF so their outputChannels/heatmaps
    # stay valid. Fresh configs get it from the defaults block above.
    if 'heatmapAddInstanceDetection' not in cfg:
        cfg['heatmapAddInstanceDetection'] = False
    if 'lossWeightInstance' not in cfg:
        cfg['lossWeightInstance'] = 1.0

    # Backward-compat: SuperPoint PCA-descriptor heatmaps add 'superpointChannels'
    # output channels — default OFF so old configs' outputChannels/heatmaps stay valid.
    if 'heatmapAddSuperpoint' not in cfg:
        cfg['heatmapAddSuperpoint'] = False
    if 'superpointChannels' not in cfg:
        cfg['superpointChannels'] = 3
    if 'superpointPcaFile' not in cfg:
        cfg['superpointPcaFile'] = "datasets/superpoint/superpoint_pca.bin"
    if 'lossWeightSuperpoint' not in cfg:
        cfg['lossWeightSuperpoint'] = 1.0

    # Backward-compat: the geolocation density head adds ONE output channel — default
    # OFF so old configs' outputChannels/heatmaps stay valid (enabling it requires
    # bumping outputChannels by +1, same manual convention as superpoint/instance).
    if 'heatmapAddGeolocation' not in cfg:
        cfg['heatmapAddGeolocation'] = False
    if 'lossWeightGeolocation' not in cfg:
        cfg['lossWeightGeolocation'] = 1.0

    # Backward-compat: tanh-output headroom, 8-bit and 16-bit heads (PLAN.md Exp T). Default
    # 1.0 reproduces the historical Rescaling (8bit scale=heatmapActive, 16bit scale=32767),
    # so old configs are numerically unchanged.
    if 'heatmapTanhHeadroom8bit' not in cfg:
        cfg['heatmapTanhHeadroom8bit'] = 1.0
    if 'heatmapTanhHeadroom16bit' not in cfg:
        cfg['heatmapTanhHeadroom16bit'] = 1.0

    # Backward-compat: dedicated geolocation head (PLAN.md Exp U). Default off, so old
    # configs are unchanged; requires heatmapAddGeolocation=true to supply the geo GT.
    if 'useGeolocationHead' not in cfg:
        cfg['useGeolocationHead'] = False
    if 'geoHeadHidden' not in cfg:
        cfg['geoHeadHidden'] = 512
    if 'lossWeightGeolocationHead' not in cfg:
        cfg['lossWeightGeolocationHead'] = 1.0
    if 'geoGridHeight' not in cfg:
        cfg['geoGridHeight'] = 64
    if 'geoGridWidth' not in cfg:
        cfg['geoGridWidth'] = 128
    if 'geoConcentrationWeighting' not in cfg:
        cfg['geoConcentrationWeighting'] = False
    if 'geoLocalizableThreshold' not in cfg:
        cfg['geoLocalizableThreshold'] = 0.005

    # Backward-compat (PLAN.md Obs 18 / Experiment N1): HeatmapCoreLoss gains used to
    # multiply INSIDE the square (effective weight = gain^2); they are now linear
    # multipliers. Configs predating the change carry old-semantics values — square
    # them once so the effective dense-MSE weighting is preserved exactly. (Caveat:
    # for a group that BOTH has gain != 1.0 AND an FN penalty, exact preservation of
    # both terms is impossible with the shared gain — the penalty scales by the extra
    # gain factor. No historical config has that combination: penalty groups
    # (joints/PAFs/instance) all ran gain 1.0.)
    if not cfg.get('fixLossGainLinearization', False):
        for _k in ('lossWeightJoints', 'lossWeightPAFs', 'lossWeightDepth',
                   'lossWeightNormals', 'lossWeightText', 'lossWeightSegmentation',
                   'lossWeightDepthLevels', 'lossWeightDenoising', 'lossLeftRightGain',
                   'lossWeightInstance', 'lossWeightSuperpoint'):
            if _k in cfg:
                cfg[_k] = float(cfg[_k])**2
        cfg['fixLossGainLinearization'] = True
    if 'lossPenaltyForegroundNormalized' not in cfg:
        cfg['lossPenaltyForegroundNormalized'] = False

    # Backward-compat: maxDecoderChannels (Experiment O) — 0 preserves the unclamped
    # legacy decoder layout, so old configs keep loading old checkpoints.
    if 'maxDecoderChannels' not in cfg:
        cfg['maxDecoderChannels'] = 0

    # Backward-compat: double-buffered batch pipeline (DATALOADER.md B1) — off preserves
    # the classic synchronous loader path exactly.
    if 'dataLoaderDoubleBuffer' not in cfg:
        cfg['dataLoaderDoubleBuffer'] = False
    # ('dataLoaderZeroCopy' migration removed with the feature — see DATALOADER.md B2)

    # Backward-compat: the depth+normals aleatoric uncertainty (hm_nll) head is an
    # additional output derived from existing depth/normals channels — it does NOT
    # change outputChannels/heatmaps, but old checkpoints have no such head, so
    # default it OFF for configs that predate it. Fresh configs get it from above.
    if 'heatmapAddDepthNormalsUncertainty' not in cfg:
        cfg['heatmapAddDepthNormalsUncertainty'] = False
    if 'lossWeightDepthNormalsUncertainty' not in cfg:
        cfg['lossWeightDepthNormalsUncertainty'] = 1.0

    # Backward-compat: Experiment R1 depthwise-separable backbone flag. Old configs /
    # checkpoints used the full-conv path; default OFF preserves that architecture.
    if 'useDepthwiseSeparable' not in cfg:
        cfg['useDepthwiseSeparable'] = False

    # Backward-compat: U1 checkpoint averaging (SWA). Default OFF ships the best single
    # epoch exactly as before; the swa* knobs are only consulted when the flag is on.
    if 'useCheckpointAveraging' not in cfg:
        cfg['useCheckpointAveraging'] = False
    if 'swaNumCheckpoints' not in cfg:
        cfg['swaNumCheckpoints'] = 5
    if 'swaBNRecalibBatches' not in cfg:
        cfg['swaBNRecalibBatches'] = 200
    if 'swaMinEpochFraction' not in cfg:
        cfg['swaMinEpochFraction'] = 0.5
    if 'swaAcceptOnlyIfBetter' not in cfg:
        cfg['swaAcceptOnlyIfBetter'] = True

    if 'augmentation' not in cfg:
        cfg['augmentation'] = {
            'chanceDestroy': 0.0, 'chancePerturbed': 35.0, 'chancePanAndZoom': 45.0,
            'chanceBurnedPixels': 50.0, 'chanceBrightnessContrast': 50.0,
            'chanceBrightnessContrastUniform': 50.0, 'chanceHorizontalFlip': 0.0,
            'chanceRotate90': 10.0, 'chanceCoarseDropout': 30.0,
            'chanceGaussianBlur': 20.0, 'chanceDefocusBlur': 15.0, 'chanceMotionBlur': 15.0,
            'maxZoomFactor': 1.1, 'perturbationMagnitude': 100, 'maxBurnedPixels': 10,
            'minBrightnessChange': -55.0, 'maxBrightnessChange': 55.0,
            'minRelContrastChange': 0.8, 'maxRelContrastChange': 1.2,
            'minUniformBrightnessChange': -100.0, 'maxUniformBrightnessChange': 120.0,
            'coarseDropoutHolesMin': 1, 'coarseDropoutHolesMax': 8,
            'coarseDropoutMinSize': 8, 'coarseDropoutMaxSize': 24,
            'gaussianBlurSigmaMin': 0.5, 'gaussianBlurSigmaMax': 3.0,
            'defocusBlurRadiusMin': 1, 'defocusBlurRadiusMax': 5,
            'motionBlurLengthMin': 5, 'motionBlurLengthMax': 15,
        }

    #Force user to pick the correct
    #if not 'learnableTokenResiduals' in cfg:
    #     cfg['learnableTokenResiduals'] = True

    if (useRAMfs):
        print("Using RAMFS")
        cfg = redirect_to_ramfs(cfg)
    else:
        print("Will not use RAMFS (Probably because data loader will automatically do caching)")

    return cfg


#============================================================================================
def createJSONConfiguration(json_file_path):
    cfg = makeJSONConfiguration()
    saveJSONConfiguration(cfg, json_file_path)
    return cfg


#============================================================================================
if __name__ == '__main__':
    if len(sys.argv) != 3 or (sys.argv[1] != "--label" and sys.argv[1] != "--serial"):
        print("\n\nWhat is the serial number of this experiment ?")
        print("Correct usage: python3 createJSONConfiguration.py --label serialnumber")
        sys.exit(1)

    serial = sys.argv[2]

    if not serial:
        print("No serial number provided. Exiting...")
        sys.exit(1)

    # Fresh configs are created at the repo root (the git-tracked training source of truth).
    # They are promoted to 2d_pose_estimation/configuration.json by trainYMAPNet.py after a
    # successful training run.
    jsonPath = 'configuration.json'
    print("Creating a new configuration file ( serial ", serial, ") in :", jsonPath)

    cfg = makeJSONConfiguration()
    cfg['serial'] = serial
    saveJSONConfiguration(cfg, jsonPath)
