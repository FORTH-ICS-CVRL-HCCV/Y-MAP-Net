#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Face-crop helpers extracted from runYMAPNet.py.
"""

import os
import cv2
import numpy as np


# =============================================================================
# Face-crop helpers
# =============================================================================

# Heatmap channel indices for facial keypoints (fixed by model architecture)
_FACE_KP_CHANNELS = [0, 1, 2, 3, 4]   # nose, left_eye, right_eye, left_ear, right_ear


def _find_kp_peaks(heatmap, threshold=40, blur_k=3):
    """
    Return a list of (x, y, value) peaks in a 2-D heatmap.
    Uses dilate-then-compare to find local maxima above *threshold*.
    """
    if blur_k > 1:
        hm = cv2.GaussianBlur(heatmap.astype(np.float32), (blur_k, blur_k), 0)
    else:
        hm = heatmap.astype(np.float32)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(hm, kernel)
    local_max = (hm == dilated) & (hm >= threshold)
    ys, xs = np.where(local_max)
    return [(int(x), int(y), float(hm[y, x])) for x, y in zip(xs, ys)]


def _cluster_face_keypoints(all_peaks, cluster_radius):
    """
    Greedy single-linkage clustering of (x, y, value) peaks.
    Two peaks belong to the same face if their Euclidean distance ≤ *cluster_radius*.
    Returns a list of clusters; each cluster is a list of (x, y, value).
    """
    clusters = []
    for pt in all_peaks:
        px, py = pt[0], pt[1]
        best_dist, best_idx = float('inf'), -1
        for ci, cl in enumerate(clusters):
            # Compare against the cluster centroid (mean of existing points)
            cx = sum(p[0] for p in cl) / len(cl)
            cy = sum(p[1] for p in cl) / len(cl)
            d  = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best_idx = d, ci
        if best_dist <= cluster_radius:
            clusters[best_idx].append(pt)
        else:
            clusters.append([pt])
    return clusters


def _dump_face_crops(estimator, frame, face_crop_counter, output_dir="faces",
                     seg_threshold=64, kp_threshold=40, min_seg_pixels=80,
                     padding=0.5):
    """
    Dump RGB face crops to *output_dir/face_XXXXXX.jpg*.

    Strategy
    --------
    Primary path — keypoint-anchored crops:
      1. Find peaks in the nose / eye / ear heatmaps (channels 0-4).
      2. Cluster nearby peaks across all 5 channels; each cluster = one face.
      3. Derive the crop box from the cluster's point spread, padded by *padding*
         (as a fraction of the box diagonal).  The Face segmentation mask is used
         only to validate that the cluster centre falls in a masked region; it is
         NOT used to define the box shape, so merged blobs (people standing close)
         are split correctly.

    Fallback path — segmentation blobs:
      When no facial keypoints fire at all (e.g. face turned away), fall back to
      connected-component bounding boxes of the Face segmentation mask, as before.
    """
    heatmaps = estimator.heatmapsOut
    n_hm     = len(heatmaps)

    fH, fW = frame.shape[:2]

    # ── Face segmentation mask ───────────────────────────────────────────────
    chan_face = getattr(estimator, 'chanFace', -1)
    face_mask = None
    if 0 <= chan_face < n_hm:
        face_mask = heatmaps[chan_face]          # H_hm × W_hm uint8

    hmH = face_mask.shape[0] if face_mask is not None else fH
    hmW = face_mask.shape[1] if face_mask is not None else fW
    scale_x = fW / hmW
    scale_y = fH / hmH

    def _save_crop(x0_hm, y0_hm, x1_hm, y1_hm):
        """Pad, clip, crop and write one face JPEG."""
        w_hm = max(1, x1_hm - x0_hm)
        h_hm = max(1, y1_hm - y0_hm)
        diag = (w_hm ** 2 + h_hm ** 2) ** 0.5
        pad  = int(diag * padding * 0.5)       # same pad on all sides

        fx0 = max(0,  int((x0_hm - pad) * scale_x))
        fy0 = max(0,  int((y0_hm - pad) * scale_y))
        fx1 = min(fW, int((x1_hm + pad) * scale_x))
        fy1 = min(fH, int((y1_hm + pad) * scale_y))

        crop = frame[fy0:fy1, fx0:fx1]
        if crop.size == 0:
            return
        idx  = face_crop_counter[0]
        face_crop_counter[0] += 1
        cv2.imwrite(os.path.join(output_dir, f"face_{idx:06d}.jpg"),
                    crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # ── 1. Collect peaks from all facial keypoint channels ───────────────────
    all_peaks = []
    for ch in _FACE_KP_CHANNELS:
        if ch < n_hm:
            all_peaks.extend(_find_kp_peaks(heatmaps[ch], threshold=kp_threshold))

    # ── 2. Cluster peaks → one cluster per face ──────────────────────────────
    #    Cluster radius: ~15 % of heatmap width covers typical inter-keypoint spread
    cluster_radius = hmW * 0.15
    clusters = _cluster_face_keypoints(all_peaks, cluster_radius) if all_peaks else []

    # ── 3. Keypoint-anchored crops ───────────────────────────────────────────
    # Precompute a binary face mask once (used to extend Y downward)
    bin_face = None
    if face_mask is not None:
        bin_face = (face_mask >= seg_threshold)

    used_kp_path = False
    for cl in clusters:
        # Skip clusters whose centroid isn't in a face-masked region
        if bin_face is not None:
            cx = int(np.clip(sum(p[0] for p in cl) / len(cl), 0, hmW - 1))
            cy = int(np.clip(sum(p[1] for p in cl) / len(cl), 0, hmH - 1))
            if not bin_face[cy, cx]:
                continue

        xs = [p[0] for p in cl]
        ys = [p[1] for p in cl]
        x0_kp, x1_kp = min(xs), max(xs)
        y0_kp, y1_kp = min(ys), max(ys)

        # Keypoints only cover the upper face (eyes/ears/nose); extend the
        # bottom of the box downward using the Face segmentation mask so the
        # chin and jaw are included.  Restrict the column range to the cluster's
        # X span (±cluster_radius) so we don't bleed into a neighbouring face.
        y1_seg = y1_kp
        if bin_face is not None:
            col_lo = max(0,    x0_kp - int(cluster_radius))
            col_hi = min(hmW,  x1_kp + int(cluster_radius))
            col_slice = bin_face[:, col_lo:col_hi]  # (hmH, col_width)
            rows_with_face = np.where(col_slice.any(axis=1))[0]
            if rows_with_face.size:
                y0_kp = min(y0_kp, int(rows_with_face[0]))
                y1_seg = max(y1_kp, int(rows_with_face[-1]))

        _save_crop(x0_kp, y0_kp, x1_kp, y1_seg)
        used_kp_path = True

    # ── 4. Fallback: segmentation blobs (no keypoints detected) ─────────────
    if not used_kp_path and face_mask is not None:
        bin_mask = (face_mask >= seg_threshold).astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            bin_mask, connectivity=8)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_seg_pixels:
                continue
            x0 = stats[lbl, cv2.CC_STAT_LEFT]
            y0 = stats[lbl, cv2.CC_STAT_TOP]
            x1 = x0 + stats[lbl, cv2.CC_STAT_WIDTH]
            y1 = y0 + stats[lbl, cv2.CC_STAT_HEIGHT]
            _save_crop(x0, y0, x1, y1)
