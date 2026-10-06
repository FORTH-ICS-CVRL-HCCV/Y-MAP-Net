"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Joint candidate extraction from the keypoint heatmaps.

Each joint heatmap is a sum of Gaussians (one per person) on a -120 background.
Candidates are strict local maxima above a threshold (grey-dilation NMS), refined
to sub-pixel accuracy with a per-axis parabola fit.  Unlike thresholded-contour
centroids this does not fuse two people's joints into one blob when their
Gaussians touch.

The predicted blobs are bumpy: without smoothing ~47% of COCO val joints get a
second local maximum within 12px (each one later seeds a phantom person); a
sigma=1.5 blur brings that to ~10%, sigma=2 to ~5%; sigma=2.5 gave the best
keypoint AP (benchmarkSkeletons.py, model 288).
"""
import cv2
import numpy as np


def findPeaks(heatmap, threshold, nms_size=5, smooth_sigma=2.5):
    """
    heatmap      : HxW float32 in the raw [-120..120] network range
    threshold    : raw value a peak must exceed
    smooth_sigma : Gaussian blur applied before peak picking (0 = off)
    returns   : Nx3 float32 array of (x_px, y_px, score) with sub-pixel x/y and
                score = (value+120)/240 in [0..1], strongest first.
    """
    hm = np.ascontiguousarray(heatmap, dtype=np.float32)
    if smooth_sigma > 0:
        # blur against a background border: the default reflected border mirrors a joint near the
        # image edge onto itself and drags its peak up to 2px onto the edge pixel
        hm = cv2.GaussianBlur(hm + 120.0, (0, 0), smooth_sigma, borderType=cv2.BORDER_CONSTANT) - 120.0
    H, W = hm.shape
    dil = cv2.dilate(hm, np.ones((nms_size, nms_size), np.uint8))
    ys, xs = np.nonzero((hm >= dil) & (hm > threshold))
    if len(xs) == 0:
        return np.zeros((0, 3), np.float32)

    # plateaus produce several equal maxima; keep one per plateau
    order = np.argsort(-hm[ys, xs], kind='stable')
    ys, xs = ys[order], xs[order]
    keep = []
    taken = np.zeros((H, W), bool)
    r = nms_size // 2
    for y, x in zip(ys, xs):
        if taken[y, x]:
            continue
        keep.append((y, x))
        taken[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
    ys = np.array([k[0] for k in keep])
    xs = np.array([k[1] for k in keep])

    # sub-pixel refinement: vertex of the parabola through the 3 neighbours on each axis
    xl, xr = np.clip(xs - 1, 0, W - 1), np.clip(xs + 1, 0, W - 1)
    yu, yd = np.clip(ys - 1, 0, H - 1), np.clip(ys + 1, 0, H - 1)
    c = hm[ys, xs]
    dx_den = hm[ys, xl] - 2 * c + hm[ys, xr]
    dy_den = hm[yu, xs] - 2 * c + hm[yd, xs]
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = np.where(dx_den < 0, 0.5 * (hm[ys, xl] - hm[ys, xr]) / dx_den, 0.0)
        dy = np.where(dy_den < 0, 0.5 * (hm[yu, xs] - hm[yd, xs]) / dy_den, 0.0)
    dx = np.clip(dx, -0.5, 0.5)
    dy = np.clip(dy, -0.5, 0.5)

    score = (c + 120.0) / 240.0
    return np.stack([xs + dx, ys + dy, score], axis=1).astype(np.float32)
