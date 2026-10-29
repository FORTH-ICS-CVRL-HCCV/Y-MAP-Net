"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Learned person score for resolved skeletons.

COCO keypoint AP ranks people by their score, and the default (mean joint confidence * joints / J)
orders them poorly: on serial 292 (1000 val2017 images) ranking the SAME skeletons by their true
OKS raises AP 0.243 -> 0.287. A linear score over the features below, fitted on half of the images,
gains +0.010 / +0.013 AP on the other half (fitPersonScore.py).

The score is linear so the C++ resolver (ymapnet_cpp/src/skeleton.hpp, person_features) computes the
exact same value; keep FEATURES and personFeatures() in sync with it. Weights are per model:
configuration.json "skeletonScoreWeights" = [bias, w_0 .. w_{N-1}] in FEATURES order (GGUF key
ymapnet.skeleton_score_weights).
"""
import numpy as np

# per-joint aux channels sampled at the joints (resolveSkeletons argument names); missing channel -> 0
AUX = ('face', 'hand', 'foot', 'right', 'left', 'bridge')
FEATURES = (['joints', 'mean_conf', 'min_conf', 'max_conf', 'size', 'base_score', 'persons'] +
            ['conf_%02d' % j for j in range(17)] + ['aux_' + a for a in AUX])


def personFeatures(xy, conf, persons, J, W, H, aux):
    """xy: Jx2 joint positions in heatmap pixels, conf: J confidences in 0..1 (0 = absent),
    persons: number of skeletons the resolver outputs for this image,
    aux: {name in AUX: HxW raw heatmap [-120..120] or None} -> feature vector in FEATURES order."""
    present = conf > 0
    c = conf[present]
    x, y = xy[present, 0], xy[present, 1]
    size = np.sqrt((x.max() - x.min() + 1.0) * (y.max() - y.min() + 1.0))
    xi = np.clip(x.astype(np.int64), 0, W - 1)    # positions are >= 0, so truncation == floor (as in C++)
    yi = np.clip(y.astype(np.int64), 0, H - 1)
    auxv = [float(aux[a][yi, xi].mean()) if aux.get(a) is not None else 0.0 for a in AUX]
    return np.array([len(c), c.mean(), c.min(), c.max(), size, c.sum() / J, persons] + list(conf[:17]) + auxv,
                    np.float64)


def personScore(features, weights):
    """weights = [bias, w_0 .. w_{N-1}]"""
    return float(weights[0] + np.dot(weights[1:], features))
