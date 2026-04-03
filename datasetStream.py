#!/usr/bin/env python3
"""
datasetStream.py — streams input images from a training or validation
DataLoader database, one image at a time, using the same
read() / isOpened() / release() interface as FolderStreamer /
cv2.VideoCapture so that runYMAPNet.py can pipe entire datasets through
the inference pipeline without any changes to the main loop.

Usage (via runYMAPNet.py --from):
    python3 runYMAPNet.py --from training_dataset   [--model 2d_pose_estimation]
    python3 runYMAPNet.py --from validation_dataset [--model 2d_pose_estimation]
"""

import sys
import os


class DatasetStreamer:
    """
    Wraps the C DataLoader as a single-frame-at-a-time source.

    The DataLoader is created with batchSize=1 and streamData=1 so the C
    library loads exactly one sample per call to get_partial_update_IO_array.
    Augmentations are disabled so the raw source image is returned.

    Parameters
    ----------
    dataset_cfg_key : str
        Key in configuration.json that holds the dataset list —
        either "TrainingDataset" or "ValidationDataset".
    model_path : str
        Directory that contains configuration.json (default: "2d_pose_estimation").
    """

    def __init__(self, dataset_cfg_key, model_path="2d_pose_estimation"):
        import json
        sys.path.append('datasets/DataLoader')
        from DataLoader import DataLoader

        cfg_path = os.path.join(model_path, 'configuration.json')
        with open(cfg_path) as fh:
            cfg = json.load(fh)

        datasets = cfg[dataset_cfg_key]

        self._db = DataLoader(
            (cfg['inputHeight'],  cfg['inputWidth'],  cfg['inputChannels']),
            (cfg['outputHeight'], cfg['outputWidth'],  cfg['outputChannels']),
            output16BitChannels    = cfg.get('output16BitChannels', 0),
            numberOfThreads        = 1,
            streamData             = 1,
            batchSize              = 1,
            gradientSize           = cfg.get('heatmapGradientSizeMinimum',
                                             cfg.get('heatmapGradientSize', 12)),
            PAFSize                = cfg.get('heatmapPAFSizeMinimum',
                                             cfg.get('heatmapPAFSize', 2)),
            doAugmentations        = 0,   # raw images, no flips / crops / colour jitter
            addPAFs                = int(cfg.get('heatmapAddPAFs', 1)),
            addBackground          = int(cfg.get('heatmapGenerateSkeletonBkg', 1)),
            addDepthMap            = int(cfg.get('heatmapAddDepthmap', 1)),
            addDepthLevelsHeatmaps = int(cfg.get('heatmapAddDepthLevels', 0)),
            addNormals             = int(cfg.get('heatmapAddNormals', 1)),
            addSegmentation        = int(cfg.get('heatmapAddSegmentation', 1)),
            datasets               = datasets,
            synonymPath            = cfg.get('synonymPath', None),
            libraryPath            = "datasets/DataLoader/libDataLoader.so",
        )

        self.frameNumber  = 0
        self.total        = self._db.numberOfSamples
        self._should_stop = False
        self._frame       = None
        print(f"DatasetStreamer: {dataset_cfg_key} — {self.total} samples")

    # ------------------------------------------------------------------

    def isOpened(self):
        return not self._should_stop

    def release(self):
        self._should_stop = True

    def read(self):
        if self._should_stop or self.frameNumber >= self.total:
            self._should_stop = True
            return False, None

        try:
            npIn, _, _ = self._db.get_partial_update_IO_array(
                self.frameNumber, self.frameNumber + 1, produce16BitData=False)
        except Exception as e:
            print(f"DatasetStreamer: error reading sample {self.frameNumber}: {e}")
            self._should_stop = True
            return False, None

        # npIn shape: (1, H, W, C)  dtype uint8
        img = npIn[0]

        # DataLoader returns RGB; convert to BGR for cv2-based downstream code
        if img.shape[2] == 3:
            img = img[:, :, ::-1].copy()

        self._frame       = img
        self.frameNumber += 1
        return True, self._frame
