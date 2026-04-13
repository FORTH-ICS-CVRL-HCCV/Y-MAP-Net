#!/usr/bin/python3
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

evaluateYMAPNet.py — End-to-end evaluation of YMAPNet on the ValidationDataset.

Computes, per heatmap group:
  SSIM, MSE, HDM @ 0.01/0.1/0.3/0.5/0.8, TP/TN/FP/FN

Computes, training-equivalent metrics (matching NNLosses.py):
  hdm, hdm_not0, hdm_not0_joints,
  hdm_joints, hdm_PAFs, hdm_depth, hdm_normal, hdm_depthlvls,
  hdm_denoise, hdm_leftright, hdm_text, hdm_person, hdm_vehicle,
  hdm_animal, hdm_floor, hdm_segms

Computes, for tokens:
  GloVe cosine similarity (overall + per-token t00..t07)
  multihot f1/precision/recall/TP/FP/FN
  top3_accuracy, top5_accuracy
  per-class TP/FP/FN/F1 + sparse confusion pairs → saved to a separate JSON

Results are printed to screen and written to a JSON file.

Computes, for skeleton resolution (resolveJointHierarchyNew):
  PCK@0.05/0.1/0.2 overall + per joint
  OKS (COCO Object Keypoint Similarity)
  Skeleton detection rate

Usage:
  python3 evaluateYMAPNet.py [--cpu] [--output results.json] [--model 2d_pose_estimation] [--no-skeleton] [--samples N] [--visualinspection]
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
from collections import defaultdict

# matplotlib is optional — plots are skipped gracefully if not installed
try:
    import matplotlib
    matplotlib.use('Agg')        # non-interactive (write PNGs, no display needed)
    import matplotlib.pyplot as plt
    _HAVE_MATPLOTLIB = True
except ImportError:
    _HAVE_MATPLOTLIB = False

# GPU / CPU toggle
useGPU = True
for arg in sys.argv[1:]:
    if arg == "--cpu":
        useGPU = False
if not useGPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import cv2
    sys.path.append('datasets/DataLoader')
    from DataLoader import DataLoader
    from tools import bcolors, checkIfFileExists
    from createJSONConfiguration import loadJSONConfiguration
    from YMAPNet import YMAPNet
    from resolveJointHierarchy import resolveJointHierarchyNew
except Exception as e:
    print("An exception occurred:", str(e))
    print("Run:  source venv/bin/activate  before this script")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Training-equivalent threshold constants (match NNLosses.py / trainYMAPNet.py)
# OUTPUT_MAGNITUDE = hm_active - hm_inactive = 120 - (-120) = 240
# HDM_THRESHOLD    = OUTPUT_MAGNITUDE * 0.01 = 2.4
# NONZERO_THRESHOLD: NonZeroCorrectPixelMetric default nonzeroThreshold=-110 (float)
#   → in uint8 display space (+ abs(hm_inactive) = +120): -110 + 120 = 10
# ---------------------------------------------------------------------------
HDM_THRESHOLD          = 2.4
NONZERO_THRESHOLD_UINT8 = 10  # GT uint8 pixel >= 10 is "non-background"

# Partial metric channel ranges (matching trainYMAPNet.py metrics list)
# Each entry: (key, start, end)  — end is *exclusive* (Python slice notation)
HDM_PARTIAL_SPECS = [
    ('hdm_joints',    0,  17),
    ('hdm_PAFs',     17,  29),
    ('hdm_depth',    29,  30),
    ('hdm_normal',   30,  33),
    ('hdm_depthlvls',33,  34),
    ('hdm_denoise',  34,  37),
    ('hdm_leftright',37,  39),
    ('hdm_text',     46,  47),
    ('hdm_person',   39,  40),
    ('hdm_vehicle',  43,  44),
    ('hdm_animal',   44,  45),
    ('hdm_floor',    57,  58),
    ('hdm_segms',    39,  73),
]

# Heatmap channel groups for SSIM/MSE/HDM/confusion (display-level metrics)
RANGE_NAMES    = ["Joints", "PAFs", "Depth", "Normals", "DepthLevels", "Denoising", "LeftRight", "Segmentation"]
HEATMAP_RANGES = [(0, 16), (17, 28), (29, 29), (30, 32), (33, 33), (34, 36), (37, 38), (39, 72)]

C1 = 0.01 ** 2
C2 = 0.03 ** 2

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))

def compute_ssim(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mu1, mu2 = np.mean(a), np.mean(b)
    s1  = np.var(a)
    s2  = np.var(b)
    s12 = np.mean((a - mu1) * (b - mu2))
    return float(((2*mu1*mu2 + C1) * (2*s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)))

def compute_hdm_display(a, b, threshold, output_magnitude=240):
    """Fraction of pixels where |a-b| <= threshold*output_magnitude.
    output_magnitude=240 matches the training range (hm_active - hm_inactive = 120-(-120)=240),
    which gives HDM0.01 threshold = 2.4, identical to trainYMAPNet.py."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return float(np.mean(diff <= threshold * output_magnitude))

def compute_confusion(a, b, threshold=127):
    a_bin = (a > threshold).astype(np.uint8)
    b_bin = (b > threshold).astype(np.uint8)
    TP = int(np.sum((a_bin == 1) & (b_bin == 1)))
    TN = int(np.sum((a_bin == 0) & (b_bin == 0)))
    FP = int(np.sum((a_bin == 0) & (b_bin == 1)))
    FN = int(np.sum((a_bin == 1) & (b_bin == 0)))
    return TP, TN, FP, FN

def cosine_similarity_1d(a, b):
    """Cosine similarity between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def multihot_metrics(gt, pred, threshold=0.5):
    """
    gt   : (N,) binary float (0 or 1)
    pred : (N,) float probabilities / scores
    Returns f1, precision, recall, TP, FP, FN.
    Element-wise accuracy is intentionally omitted: with 17977 classes and
    only ~5-20 active per sample, the TN-dominated accuracy is always >99%
    regardless of model quality. F1 (= 2*TP / (2*TP+FP+FN)) is used instead.
    """
    gt_bin   = (gt   > 0.5).astype(np.uint8)
    pred_bin = (pred > threshold).astype(np.uint8)
    TP = int(np.sum((gt_bin == 1) & (pred_bin == 1)))
    FP = int(np.sum((gt_bin == 0) & (pred_bin == 1)))
    FN = int(np.sum((gt_bin == 1) & (pred_bin == 0)))
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * TP / (2 * TP + FP + FN + 1e-8)
    return float(f1), float(precision), float(recall), TP, FP, FN

def topk_hit(pred_scores, gt_binary, k):
    """Return 1.0 if any GT-positive class appears in top-k predictions, else 0.0."""
    gt_active = np.where(gt_binary > 0.5)[0]
    if len(gt_active) == 0:
        return 1.0
    top_k_idx = set(np.argpartition(pred_scores, -k)[-k:])
    return 1.0 if any(idx in top_k_idx for idx in gt_active) else 0.0

def gt_int8_to_display_uint8(gt_channel, hm_active=120, hm_inactive=-120):
    """
    Convert GT int8 heatmap channel to uint8 using the same formula as convertIO:
      uint8_val = float_val + abs(hm_inactive)
    Matches the display conversion used by the estimator's heatmapsOut.
    """
    f = gt_channel.astype(np.float32)
    f = np.clip(f, hm_inactive, hm_active)
    f = f + abs(hm_inactive)
    return f.astype(np.uint8)

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

# ---------------------------------------------------------------------------
# Skeleton evaluation constants
# ---------------------------------------------------------------------------
PCK_THRESHOLDS = [0.05, 0.10, 0.20]

# COCO per-keypoint sigmas (17 joints, same order as keypoint_names in config)
COCO_KP_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035,   # nose, l/r eye, l/r ear
    0.079, 0.079,                          # l/r shoulder
    0.072, 0.072,                          # l/r elbow
    0.062, 0.062,                          # l/r wrist
    0.107, 0.107,                          # l/r hip
    0.087, 0.087,                          # l/r knee
    0.089, 0.089,                          # l/r ankle
], dtype=np.float32)

# First PAF channel index and count in the heatmap output
PAF_FIRST_CH = 17
PAF_NUM_CH   = 12

# ---------------------------------------------------------------------------
# Skeleton evaluation helpers
# ---------------------------------------------------------------------------
def extract_gt_keypoints(gt_heatmaps, n_joints, act_threshold=0.5):
    """
    Peak-pick each of the first n_joints channels of gt_heatmaps (int8 in [-120..120]).
    Returns a list of n_joints tuples (x_norm, y_norm, confidence_norm).
    Missing joints are (0, 0, 0).  Picks the highest-activation centroid when
    multiple contours exist (multi-person GT shares the same channel).
    """
    H, W = gt_heatmaps.shape[:2]
    result = []
    thr_abs = act_threshold * 240.0  # threshold in shifted [0..240] space
    for j in range(n_joints):
        ch = gt_heatmaps[:, :, j].astype(np.float32) + 120.0  # → [0..240]
        if ch.max() < thr_abs:
            result.append((0.0, 0.0, 0.0))
            continue
        binary = (ch >= thr_abs).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_xy, best_v = None, -1.0
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] <= 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            v  = float(ch[min(int(round(cy)), H-1), min(int(round(cx)), W-1)])
            if v > best_v:
                best_v  = v
                best_xy = (cx / W, cy / H)
        if best_xy is not None:
            result.append((best_xy[0], best_xy[1], best_v / 240.0))
        else:
            result.append((0.0, 0.0, 0.0))
    return result


def match_skeleton_to_gt(pred_skeletons, gt_kps, n_joints):
    """
    Greedy centroid matching: return the predicted skeleton whose centroid
    is closest to the GT centroid.  Returns a zero skeleton if none available.
    """
    if not pred_skeletons:
        return [0.0] * (n_joints * 3)
    vis_gt = [(gx, gy) for (gx, gy, gv) in gt_kps if gv > 0]
    if not vis_gt:
        return pred_skeletons[0]
    gt_cx = float(np.mean([p[0] for p in vis_gt]))
    gt_cy = float(np.mean([p[1] for p in vis_gt]))
    best_skel, best_dist = pred_skeletons[0], float('inf')
    for skel in pred_skeletons:
        xs = [skel[j*3]   for j in range(n_joints) if skel[j*3+2] > 0]
        ys = [skel[j*3+1] for j in range(n_joints) if skel[j*3+2] > 0]
        if not xs:
            continue
        d = ((np.mean(xs) - gt_cx)**2 + (np.mean(ys) - gt_cy)**2)**0.5
        if d < best_dist:
            best_dist, best_skel = d, skel
    return best_skel


def compute_pck_hits(pred_skel, gt_kps, thr, n_joints):
    """
    Returns (hits[n_joints], visible[n_joints]) for PCK at normalized threshold thr.
    A joint counts as a hit if it is predicted, visible in GT, and within thr of GT.
    """
    hits    = np.zeros(n_joints, dtype=np.float64)
    visible = np.zeros(n_joints, dtype=np.float64)
    for j in range(n_joints):
        gx, gy, gv = gt_kps[j]
        if gv <= 0:
            continue
        visible[j] = 1.0
        pv = pred_skel[j*3 + 2]
        if pv <= 0:
            continue
        if ((pred_skel[j*3] - gx)**2 + (pred_skel[j*3+1] - gy)**2)**0.5 <= thr:
            hits[j] = 1.0
    return hits, visible


def compute_oks_score(pred_skel, gt_kps, n_joints, kp_sigmas):
    """
    Object Keypoint Similarity (COCO formula).
    s  = sqrt(bounding-box area of GT visible joints)
    Returns OKS in [0..1], or -1.0 if no visible GT joints.
    """
    vis_pts = [(gx, gy) for (gx, gy, gv) in gt_kps if gv > 0]
    if not vis_pts:
        return -1.0
    xs, ys = [p[0] for p in vis_pts], [p[1] for p in vis_pts]
    area = max(1e-8, (max(xs) - min(xs)) * (max(ys) - min(ys)))
    s    = area ** 0.5
    num, denom = 0.0, 0.0
    for j in range(n_joints):
        gx, gy, gv = gt_kps[j]
        if gv <= 0:
            continue
        denom += 1.0
        pv = pred_skel[j*3 + 2]
        if pv <= 0:
            continue
        d2 = (pred_skel[j*3] - gx)**2 + (pred_skel[j*3+1] - gy)**2
        k  = float(kp_sigmas[j]) if j < len(kp_sigmas) else 0.072
        num += float(np.exp(-d2 / (2.0 * s**2 * k**2 + 1e-10)))
    return float(num / denom) if denom > 0 else -1.0


# ---------------------------------------------------------------------------
# Visual inspection helper
# ---------------------------------------------------------------------------
def dump_visual_inspection(sample_idx, rgb_input,
                           gt_heatmaps, pred_heatmapsOut,
                           gt_kps, pred_skels,
                           kp_names, paf_parents=None,
                           out_root='visual_inspection'):
    """
    For each joint channel, write a side-by-side PNG:
      LEFT  — GT heatmap (green) with GT peak dot (blue)
      RIGHT — Predicted heatmap (green) with every resolved-skeleton joint dot (red)
              and the best-matched skeleton joint (yellow)

    Layout: [RGB | GT heatmap + peak | Pred heatmap + peak]
    Saved to  out_root/sample_NNNNN/<joint_name>.png
    """
    sample_dir = os.path.join(out_root, f'sample_{sample_idx:05d}')
    os.makedirs(sample_dir, exist_ok=True)

    n_joints = len(kp_names)
    H_rgb, W_rgb = rgb_input.shape[:2]

    # Resize RGB to match heatmap display height
    H_hm = gt_heatmaps.shape[0]
    W_hm = gt_heatmaps.shape[1]
    rgb_resized = cv2.resize(rgb_input, (W_hm, H_hm))

    # Best-matched skeleton (greedy centroid)
    best_pred = match_skeleton_to_gt(pred_skels, gt_kps, n_joints)

    for j in range(n_joints):
        jname = kp_names[j]

        # ---- GT panel ----
        gt_ch  = gt_heatmaps[:, :, j].astype(np.float32) + 120.0   # [0..240]
        gt_img = np.clip(gt_ch / 240.0 * 255.0, 0, 255).astype(np.uint8)
        gt_bgr = cv2.applyColorMap(gt_img, cv2.COLORMAP_WINTER)     # cool blue-green

        gx, gy, gv = gt_kps[j]
        if gv > 0:
            px_g = int(round(gx * (W_hm - 1)))
            py_g = int(round(gy * (H_hm - 1)))
            cv2.circle(gt_bgr, (px_g, py_g), 5, (255, 0, 0), -1)   # blue = GT

        # ---- Pred panel ----
        if j < len(pred_heatmapsOut):
            pred_ch = pred_heatmapsOut[j]   # uint8 [0..255] from convertIO
        else:
            pred_ch = np.zeros((H_hm, W_hm), dtype=np.uint8)
        pred_bgr = cv2.applyColorMap(pred_ch, cv2.COLORMAP_HOT)

        # All skeleton joints for this keypoint type (all candidates, all skeletons)
        for skel in pred_skels:
            sx = skel[j*3]
            sy = skel[j*3 + 1]
            sv = skel[j*3 + 2]
            if sv > 0:
                px_s = int(round(sx * (W_hm - 1)))
                py_s = int(round(sy * (H_hm - 1)))
                cv2.circle(pred_bgr, (px_s, py_s), 4, (0, 0, 255), -1)   # red = any skel

        # Best matched skeleton joint
        bx = best_pred[j*3]
        by = best_pred[j*3 + 1]
        bv = best_pred[j*3 + 2]
        if bv > 0:
            px_b = int(round(bx * (W_hm - 1)))
            py_b = int(round(by * (H_hm - 1)))
            cv2.circle(pred_bgr, (px_b, py_b), 6, (0, 255, 255), 2)      # yellow ring = best

        # ---- Labels ----
        cv2.putText(gt_bgr,   f'GT  {jname}',   (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(pred_bgr, f'PRD {jname}',   (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # ---- Combine [RGB | GT | Pred] ----
        row = np.concatenate([rgb_resized, gt_bgr, pred_bgr], axis=1)
        cv2.imwrite(os.path.join(sample_dir, f'{j:02d}_{jname}.png'), row)

    # ---- PAF plots ----
    if paf_parents is not None:
        # Map local PAF channel (0..11) → child joint index + name
        paf_ch_to_joint = {}
        for j, abs_ch in enumerate(paf_parents):
            if isinstance(abs_ch, int) and abs_ch >= PAF_FIRST_CH and j < len(kp_names):
                local = abs_ch - PAF_FIRST_CH
                paf_ch_to_joint[local] = (j, kp_names[j])

        for local_ch in range(PAF_NUM_CH):
            abs_ch = PAF_FIRST_CH + local_ch
            if abs_ch >= gt_heatmaps.shape[2]:
                break

            joint_info = paf_ch_to_joint.get(local_ch)
            label = joint_info[1] if joint_info else f'paf{local_ch}'

            # GT PAF panel — shift int8-range to [0..240] → uint8
            gt_raw = gt_heatmaps[:, :, abs_ch].astype(np.float32) + 120.0
            gt_img = np.clip(gt_raw / 240.0 * 255.0, 0, 255).astype(np.uint8)
            gt_bgr = cv2.applyColorMap(gt_img, cv2.COLORMAP_JET)
            cv2.putText(gt_bgr, f'GT PAF {label}', (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Pred PAF panel — already uint8 [0..255]
            if abs_ch < len(pred_heatmapsOut):
                pred_raw = pred_heatmapsOut[abs_ch]
            else:
                pred_raw = np.zeros((H_hm, W_hm), dtype=np.uint8)
            pred_bgr = cv2.applyColorMap(pred_raw, cv2.COLORMAP_JET)

            # Overlay the child joint position for all skeletons and best match
            if joint_info is not None:
                jidx = joint_info[0]
                for skel in pred_skels:
                    sx, sy, sv = skel[jidx*3], skel[jidx*3+1], skel[jidx*3+2]
                    if sv > 0:
                        cv2.circle(pred_bgr,
                                   (int(round(sx*(W_hm-1))), int(round(sy*(H_hm-1)))),
                                   4, (0, 0, 255), -1)   # red = any skel
                bx, by, bv = best_pred[jidx*3], best_pred[jidx*3+1], best_pred[jidx*3+2]
                if bv > 0:
                    cv2.circle(pred_bgr,
                               (int(round(bx*(W_hm-1))), int(round(by*(H_hm-1)))),
                               6, (0, 255, 255), 2)       # yellow ring = best

            cv2.putText(pred_bgr, f'PRD PAF {label}', (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            row = np.concatenate([rgb_resized, gt_bgr, pred_bgr], axis=1)
            cv2.imwrite(
                os.path.join(sample_dir, f'paf_{local_ch:02d}_{label}.png'),
                row
            )


# ---------------------------------------------------------------------------
# Statistical plots for heatmap metrics
# ---------------------------------------------------------------------------
def plot_heatmap_metrics(hm_acc, train_results, serial, output_prefix,
                        token_data=None):
    """
    Generate and save PNG plots summarising evaluation metrics.

    Saved files:
      <output_prefix>_heatmap_overview.png   — mean SSIM / MSE / HDM@0.1 / HDM@0.5 per group
      <output_prefix>_hdm_thresholds.png     — HDM at all 5 thresholds per group (grouped bars)
      <output_prefix>_training_hdm.png       — training-equivalent HDM partial metric bar chart
      <output_prefix>_distributions.png      — box plots of per-sample SSIM and MSE per group
      <output_prefix>_token_metrics.png      — token cosine / multihot / top-k bar chart (if tokens present)
      <output_prefix>_token_distributions.png— box plots of per-sample cosine sim per token slot (if tokens present)

    token_data (optional dict) keys:
      cosine_sims         list[float]        overall cosine similarity per sample
      cosine_per_token    list[list[float]]  per-slot cosine similarity per sample
      multihot_f1         list[float]
      multihot_prec       list[float]
      multihot_rec        list[float]
      top3_acc            list[float]
      top5_acc            list[float]
    """
    if not _HAVE_MATPLOTLIB:
        print("[warn] matplotlib not available — skipping metric plots")
        return

    names  = RANGE_NAMES                  # list of group labels (8 entries)
    n      = len(names)
    x      = np.arange(n)
    colors = [plt.cm.tab10(i / 10.0) for i in range(n)]

    # -----------------------------------------------------------------------
    # Plot 1 — overview: SSIM, MSE, HDM@0.1, HDM@0.5 in one figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'Heatmap Metrics Overview  (model {serial})', fontsize=13)

    metrics_overview = [
        ('SSIM',    'Mean SSIM',      axes[0, 0], False),
        ('MSE',     'Mean MSE',       axes[0, 1], True),   # higher = worse, highlight in red
        ('HDM0.1',  'HDM @ 0.1',     axes[1, 0], False),
        ('HDM0.5',  'HDM @ 0.5',     axes[1, 1], False),
    ]
    for metric, ylabel, ax, invert_good in metrics_overview:
        vals = [float(np.mean(hm_acc[nm][metric])) if hm_acc[nm][metric] else 0.0
                for nm in names]
        bar_colors = colors if not invert_good else ['#d62728'] * n
        bars = ax.bar(x, vals, color=bar_colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        # Annotate each bar with its value
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    path1 = f'{output_prefix}_heatmap_overview.png'
    fig.savefig(path1, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path1}")

    # -----------------------------------------------------------------------
    # Plot 2 — HDM threshold sweep: 5 thresholds × 8 groups (grouped bars)
    # -----------------------------------------------------------------------
    thresholds  = ['HDM0.01', 'HDM0.1', 'HDM0.3', 'HDM0.5', 'HDM0.8']
    thr_labels  = ['0.01',    '0.1',    '0.3',    '0.5',    '0.8']
    n_thr       = len(thresholds)
    bar_width   = 0.15
    thr_colors  = [plt.cm.viridis(i / (n_thr - 1)) for i in range(n_thr)]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'HDM at Different Thresholds  (model {serial})', fontsize=13)
    for ti, (thr_key, thr_lbl) in enumerate(zip(thresholds, thr_labels)):
        vals = [float(np.mean(hm_acc[nm][thr_key])) if hm_acc[nm][thr_key] else 0.0
                for nm in names]
        offset = (ti - n_thr / 2.0 + 0.5) * bar_width
        ax.bar(x + offset, vals, width=bar_width,
               label=f'HDM@{thr_lbl}', color=thr_colors[ti], edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('HDM (fraction of pixels within threshold)', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=n_thr)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    path2 = f'{output_prefix}_hdm_thresholds.png'
    fig.savefig(path2, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path2}")

    # -----------------------------------------------------------------------
    # Plot 3 — training-equivalent HDM partial metrics bar chart
    # -----------------------------------------------------------------------
    # Keep a sensible order: overall first, then per-group partials
    train_keys_ordered = (
        ['hdm', 'hdm_not0', 'hdm_not0_joints'] +
        [k for k, _, _ in HDM_PARTIAL_SPECS]
    )
    train_keys_ordered = [k for k in train_keys_ordered if k in train_results]
    train_vals = [float(train_results[k]) for k in train_keys_ordered]
    xt = np.arange(len(train_keys_ordered))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'Training-Equivalent HDM Metrics  (model {serial})', fontsize=13)
    bar_colors_t = [plt.cm.Set2(i / max(len(train_keys_ordered) - 1, 1))
                    for i in range(len(train_keys_ordered))]
    bars = ax.bar(xt, train_vals, color=bar_colors_t, edgecolor='white', linewidth=0.5)
    ax.set_xticks(xt)
    ax.set_xticklabels(train_keys_ordered, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('HDM score', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    for bar, v in zip(bars, train_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                f'{v:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
    fig.tight_layout()
    path3 = f'{output_prefix}_training_hdm.png'
    fig.savefig(path3, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path3}")

    # -----------------------------------------------------------------------
    # Plot 4 — distribution: box plots of per-sample SSIM and MSE per group
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Per-Sample/Channel Distributions  (model {serial})', fontsize=13)

    ssim_data = [hm_acc[nm]['SSIM'] for nm in names]
    mse_data  = [hm_acc[nm]['MSE']  for nm in names]

    for ax, data, title, ylabel in [
        (axes[0], ssim_data, 'SSIM Distribution',  'SSIM'),
        (axes[1], mse_data,  'MSE Distribution',   'MSE'),
    ]:
        # Remove empty groups so matplotlib doesn't crash on empty arrays
        non_empty = [(nm, d) for nm, d in zip(names, data) if len(d) > 0]
        if non_empty:
            bp_labels, bp_data = zip(*non_empty)
            bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                            medianprops=dict(color='black', linewidth=1.5))
            for patch, color in zip(bp['boxes'], colors[:len(bp_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    path4 = f'{output_prefix}_distributions.png'
    fig.savefig(path4, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path4}")

    # -----------------------------------------------------------------------
    # Plot 5 — token summary metrics (skipped when no token data was collected)
    # -----------------------------------------------------------------------
    if token_data is None:
        return

    cosine_sims      = token_data.get('cosine_sims',      [])
    cosine_per_token = token_data.get('cosine_per_token', [])
    multihot_f1      = token_data.get('multihot_f1',      [])
    multihot_prec    = token_data.get('multihot_prec',    [])
    multihot_rec     = token_data.get('multihot_rec',     [])
    top3_acc         = token_data.get('top3_acc',         [])
    top5_acc         = token_data.get('top5_acc',         [])

    # Bar chart: one bar per aggregate token metric
    token_bar_labels = []
    token_bar_vals   = []

    if cosine_sims:
        token_bar_labels.append('Cosine\n(overall)')
        token_bar_vals.append(float(np.mean(cosine_sims)))
    if multihot_f1:
        token_bar_labels.append('Multihot\nF1')
        token_bar_vals.append(float(np.mean(multihot_f1)))
    if multihot_prec:
        token_bar_labels.append('Multihot\nPrecision')
        token_bar_vals.append(float(np.mean(multihot_prec)))
    if multihot_rec:
        token_bar_labels.append('Multihot\nRecall')
        token_bar_vals.append(float(np.mean(multihot_rec)))
    if top3_acc:
        token_bar_labels.append('Top-3\nAccuracy')
        token_bar_vals.append(float(np.mean(top3_acc)))
    if top5_acc:
        token_bar_labels.append('Top-5\nAccuracy')
        token_bar_vals.append(float(np.mean(top5_acc)))

    # Add per-token-slot cosine means as individual bars
    for ti, slot_sims in enumerate(cosine_per_token):
        if slot_sims:
            token_bar_labels.append(f't{ti:02d}\nCosine')
            token_bar_vals.append(float(np.mean(slot_sims)))

    if token_bar_labels:
        xt = np.arange(len(token_bar_labels))
        bar_colors_tok = [plt.cm.tab20(i / max(len(token_bar_labels) - 1, 1))
                          for i in range(len(token_bar_labels))]
        fig, ax = plt.subplots(figsize=(max(10, len(token_bar_labels) * 0.9), 5))
        fig.suptitle(f'Token Metrics  (model {serial})', fontsize=13)
        bars = ax.bar(xt, token_bar_vals, color=bar_colors_tok,
                      edgecolor='white', linewidth=0.5)
        ax.set_xticks(xt)
        ax.set_xticklabels(token_bar_labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        for bar, v in zip(bars, token_bar_vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)
        fig.tight_layout()
        path5 = f'{output_prefix}_token_metrics.png'
        fig.savefig(path5, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path5}")

    # -----------------------------------------------------------------------
    # Plot 6 — token cosine distribution: box plot per token slot
    # -----------------------------------------------------------------------
    slot_data = [(f't{ti:02d}', sims)
                 for ti, sims in enumerate(cosine_per_token) if sims]
    if slot_data:
        slot_labels, slot_sims_list = zip(*slot_data)
        fig, ax = plt.subplots(figsize=(max(8, len(slot_data) * 0.8), 5))
        fig.suptitle(f'Per-Token Cosine Similarity Distribution  (model {serial})', fontsize=13)
        bp = ax.boxplot(slot_sims_list, labels=slot_labels, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5))
        slot_colors = [plt.cm.tab10(i / 10.0) for i in range(len(slot_data))]
        for patch, color in zip(bp['boxes'], slot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Cosine Similarity', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        fig.tight_layout()
        path6 = f'{output_prefix}_token_distributions.png'
        fig.savefig(path6, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path6}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate():
    model_path       = '2d_pose_estimation'
    json_path        = '2d_pose_estimation/configuration.json'
    output_json      = None  # auto-named after evaluation
    confusion_json   = None  # auto-named after evaluation

    do_skeleton_eval  = True
    do_visual_inspect = False
    max_samples       = None
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--output' and i + 1 < len(argv):
            output_json = argv[i + 1]; i += 2
        elif argv[i] == '--model' and i + 1 < len(argv):
            model_path = argv[i + 1]; i += 2
        elif argv[i] == '--no-skeleton':
            do_skeleton_eval = False; i += 1
        elif argv[i] == '--samples' and i + 1 < len(argv):
            max_samples = int(argv[i + 1]); i += 2
        elif argv[i] == '--visualinspection':
            do_visual_inspect = True; i += 1
        else:
            i += 1

    if do_visual_inspect and (max_samples is None or max_samples >= 20):
        print(bcolors.FAIL,
              "--visualinspection requires --samples N with N < 20 (would produce too many files otherwise)",
              bcolors.ENDC)
        sys.exit(1)

    if not checkIfFileExists(json_path):
        print(bcolors.FAIL, f"Configuration not found: {json_path}", bcolors.ENDC)
        sys.exit(1)

    cfg    = loadJSONConfiguration(json_path, useRAMfs=False)
    serial = cfg.get('serial', 'unknown')

    if output_json is None:
        output_json = f"evaluation_results_{serial}_full.json"
    if confusion_json is None:
        # Place confusion JSON in the same directory as output_json so that
        # when --output points inside 2d_pose_estimation/, everything lands there
        # and gets picked up by the zip step in trainYMAPNet.py.
        out_dir       = os.path.dirname(output_json) or '.'
        confusion_json = os.path.join(out_dir, f"evaluation_results_{serial}_token_confusion.json")

    # Load vocabulary for class-name labelling in the confusion output
    vocab_path = os.path.join(model_path, 'vocabulary.json')
    vocab = {}
    if os.path.exists(vocab_path):
        try:
            with open(vocab_path) as _vf:
                _raw = json.load(_vf)
            if isinstance(_raw, list):
                vocab = {i: w for i, w in enumerate(_raw)}
            elif isinstance(_raw, dict):
                vocab = {int(k): v for k, v in _raw.items()}
            print(f"Loaded vocabulary: {len(vocab)} classes")
        except Exception as _ve:
            print(f"[warn] Could not load vocabulary: {_ve}")

    # -----------------------------------------------------------------------
    # Build validation DataLoader — no augmentation, configured batch size
    # NOTE: batchSize=1 triggers an OOM guard in HeatmapGenerator.c (the
    # output buffer is sized batchSize*sampleSize bytes, so with batchSize=1
    # the guard "pixels + sampleSize >= pixelsLimit" fires immediately).
    # Use cfg['batchSize'] and iterate in sub-batches, same as training.
    # -----------------------------------------------------------------------
    batch_size = int(cfg['batchSize'])
    print(bcolors.OKGREEN, "Creating validation DataLoader...", bcolors.ENDC)
    db = DataLoader(
        (cfg['inputHeight'],  cfg['inputWidth'],  cfg['inputChannels']),
        (cfg['outputHeight'], cfg['outputWidth'],  cfg['outputChannels']),
        output16BitChannels    = cfg['output16BitChannels'],
        numberOfThreads        = cfg['DatasetLoaderThreads'],
        streamData             = 1,
        batchSize              = batch_size,
        gradientSize           = cfg['heatmapGradientSizeMinimum'],
        PAFSize                = cfg['heatmapPAFSizeMinimum'],
        doAugmentations        = 0,
        addPAFs                = int(cfg['heatmapAddPAFs']),
        addBackground          = int(cfg['heatmapGenerateSkeletonBkg']),
        addDepthMap            = int(cfg['heatmapAddDepthmap']),
        addDepthLevelsHeatmaps = int(cfg['heatmapAddDepthLevels']),
        addNormals             = int(cfg['heatmapAddNormals']),
        addSegmentation        = int(cfg['heatmapAddSegmentation']),
        datasets               = cfg['ValidationDataset'],
        libraryPath            = "datasets/DataLoader/libDataLoader.so"
    )

    n_samples   = db.numberOfSamples
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)
    n_channels  = int(cfg['outputChannels'])
    tokens_out  = int(cfg.get('tokensOut',  8))
    has_tokens  = bool(cfg.get('outputTokens', False))
    hm_active   = int(cfg.get('heatmapActive',      120))
    hm_inactive = int(cfg.get('heatmapDeactivated', -120))

    print(f"Validation samples: {n_samples}  |  batch_size: {batch_size}  |  channels: {n_channels}")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(bcolors.OKGREEN, "Loading YMAPNet model...", bcolors.ENDC)
    estimator = YMAPNet(
        modelPath         = model_path,
        threshold         = 0,
        keypoint_threshold= 50.0,   # match webcam default — controls peak detection threshold
        engine            = 'tensorflow',
        profiling         = False,
        compileModel      = False    # skip optimizer state loading — not needed for evaluation
    )

    # -----------------------------------------------------------------------
    # Metric accumulators — display-level (SSIM, MSE, HDM%, confusion)
    # -----------------------------------------------------------------------
    hm_acc = {}
    for name in RANGE_NAMES:
        hm_acc[name] = {
            'SSIM': [], 'MSE': [],
            'HDM0.01': [], 'HDM0.1': [], 'HDM0.3': [], 'HDM0.5': [], 'HDM0.8': [],
            'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0
        }

    # -----------------------------------------------------------------------
    # Metric accumulators — training-equivalent (match NNLosses.py)
    # -----------------------------------------------------------------------
    # hdm / hdm partial: {correct, total}
    # hdm_not0 / hdm_not0_joints: {correct, nonzero}
    train_metrics = {}
    train_metrics['hdm'] = {'correct': 0, 'total': 0}
    train_metrics['hdm_not0'] = {'correct': 0, 'nonzero': 0}
    train_metrics['hdm_not0_joints'] = {'correct': 0, 'nonzero': 0}
    for key, _s, _e in HDM_PARTIAL_SPECS:
        train_metrics[key] = {'correct': 0, 'total': 0}

    # -----------------------------------------------------------------------
    # Token accumulators
    # -----------------------------------------------------------------------
    token_cosine_sims      = []
    token_cosine_per_token = [[] for _ in range(tokens_out)]  # per-token cosine sims
    token_multihot_f1      = []
    token_multihot_prec    = []
    token_multihot_rec     = []
    token_multihot_TP      = 0
    token_multihot_FP      = 0
    token_multihot_FN      = 0
    token_top3_acc         = []
    token_top5_acc         = []

    # Per-class and confusion-pair accumulators (lazy-init on first multihot sample)
    class_TP       = None   # np.int64 array (n_classes,)
    class_FP       = None
    class_FN       = None
    confusion_ctr  = defaultdict(int)  # (true_idx, pred_idx) -> count

    # -----------------------------------------------------------------------
    # Skeleton resolution accumulators
    # -----------------------------------------------------------------------
    kp_names   = cfg.get('keypoint_names', [])
    n_joints   = len(kp_names)
    has_skel_cfg = (n_joints > 0 and
                    'keypoint_parents'  in cfg and
                    'keypoint_children' in cfg and
                    'paf_parents'       in cfg)
    do_skeleton_eval = do_skeleton_eval and has_skel_cfg

    pck_hits     = {t: np.zeros(n_joints, dtype=np.float64) for t in PCK_THRESHOLDS}
    pck_visible  = {t: np.zeros(n_joints, dtype=np.float64) for t in PCK_THRESHOLDS}
    oks_values   = []
    skel_detected = 0
    skel_total    = 0

    # -----------------------------------------------------------------------
    # Main loop — iterate in batches (same size as training) to satisfy the
    # DataLoader's minimum-batch-size constraint in HeatmapGenerator.c
    # -----------------------------------------------------------------------
    start_time   = time.time()
    samples_done = 0
    cosine_error_reported = False

    for batch_start in range(0, n_samples, batch_size):
        batch_end    = min(batch_start + batch_size, n_samples)
        actual_count = batch_end - batch_start

        # --- Get ground-truth data from DataLoader --------------------------
        try:
            npArrayIn, npArrayOut, _ = db.get_partial_update_IO_array(batch_start, batch_end)
        except Exception as e:
            print(f"\nFailed to load batch {batch_start}-{batch_end}: {e}")
            samples_done += actual_count
            continue

        # npArrayIn  : (actual_count, H, W, 3)   uint8
        # npArrayOut : (actual_count, H, W, 73)  int8  in [hm_inactive, hm_active]

        # Fetch GT tokens / embeddings for the whole batch at once
        gt_embeddings_batch = None
        gt_multihot_batch   = None
        if has_tokens:
            try:
                gt_embeddings_batch = db.get_partial_embedding_array(batch_start, batch_end)  # (B, tokensOut, 300)
            except Exception:
                pass
            try:
                gt_multihot_batch = db.get_partial_token_array(batch_start, batch_end,
                                                               encodeAsSingleMultiLabelToken=True)  # (B, n_classes)
            except Exception:
                pass

        # --- Per-sample inference and metric computation --------------------
        for i in range(actual_count):
            rgb_input   = npArrayIn[i]   # (H, W, 3)
            gt_heatmaps = npArrayOut[i]  # (H, W, outputChannels)

            try:
                estimator.process(rgb_input)
            except Exception as e:
                print(f"\nInference failed on sample {batch_start+i}: {e}")
                continue

            n_pred_hm = len(estimator.heatmapsOut)
            n_ch_eval = min(n_channels, gt_heatmaps.shape[2], n_pred_hm)

            # ----------------------------------------------------------------
            # Skeleton resolution metrics
            # ----------------------------------------------------------------
            if do_skeleton_eval:
                try:
                    pred_kp_hm = np.stack(
                        estimator.heatmapsOut[:n_joints], axis=2
                    ).astype(np.float32) - 120.0   # uint8→[-120..120]

                    paf_end  = min(PAF_FIRST_CH + PAF_NUM_CH, n_pred_hm)
                    pred_paf = [
                        estimator.heatmapsOut[PAF_FIRST_CH + j].astype(np.float32) - 120.0
                        for j in range(paf_end - PAF_FIRST_CH)
                    ]

                    pred_skels = resolveJointHierarchyNew(
                        pred_kp_hm, pred_paf, estimator.depthmap,
                        cfg['keypoint_names'],
                        cfg['keypoint_parents'],
                        cfg['keypoint_children'],
                        cfg['paf_parents'],
                        sanity_check=True,
                        person_label_map=None,
                        threshold=estimator.keypoint_threshold,
                    )
                except Exception:
                    pred_skels = []

                gt_kps    = extract_gt_keypoints(gt_heatmaps, n_joints)
                best_pred = match_skeleton_to_gt(pred_skels, gt_kps, n_joints)

                for thr in PCK_THRESHOLDS:
                    h, v = compute_pck_hits(best_pred, gt_kps, thr, n_joints)
                    pck_hits[thr]    += h
                    pck_visible[thr] += v

                oks = compute_oks_score(best_pred, gt_kps, n_joints, COCO_KP_SIGMAS)
                if oks >= 0:
                    oks_values.append(oks)

                skel_total += 1
                if pred_skels:
                    skel_detected += 1

                # ---- Visual inspection dump --------------------------------
                if do_visual_inspect:
                    dump_visual_inspection(
                        sample_idx       = batch_start + i,
                        rgb_input        = rgb_input,
                        gt_heatmaps      = gt_heatmaps,
                        pred_heatmapsOut = estimator.heatmapsOut,
                        gt_kps           = gt_kps,
                        pred_skels       = pred_skels,
                        kp_names         = kp_names,
                        paf_parents      = cfg['paf_parents'],
                    )

            # ----------------------------------------------------------------
            # Build full uint8 stacks for training-equivalent metrics
            # ----------------------------------------------------------------
            H, W = gt_heatmaps.shape[0], gt_heatmaps.shape[1]
            gt_uint8   = np.empty((H, W, n_ch_eval), dtype=np.uint8)
            pred_uint8 = np.empty((H, W, n_ch_eval), dtype=np.uint8)
            for ch in range(n_ch_eval):
                gt_uint8[:, :, ch]   = gt_int8_to_display_uint8(gt_heatmaps[:, :, ch], hm_active, hm_inactive)
                pred_uint8[:, :, ch] = estimator.heatmapsOut[ch]

            diff         = np.abs(gt_uint8.astype(np.float32) - pred_uint8.astype(np.float32))
            correct_mask = diff <= HDM_THRESHOLD
            nonzero_mask = gt_uint8 >= NONZERO_THRESHOLD_UINT8

            # --- Training-equivalent HDM (all channels) ---------------------
            train_metrics['hdm']['correct'] += int(np.sum(correct_mask))
            train_metrics['hdm']['total']   += correct_mask.size

            # --- hdm_not0 (all channels, non-background GT only) ------------
            nz = nonzero_mask
            train_metrics['hdm_not0']['correct'] += int(np.sum(correct_mask & nz))
            train_metrics['hdm_not0']['nonzero'] += int(np.sum(nz))

            # --- hdm_not0_joints (channels 0-16, non-background GT only) ----
            nz_j  = nonzero_mask[:, :, 0:17]
            cor_j = correct_mask[:, :, 0:17]
            train_metrics['hdm_not0_joints']['correct'] += int(np.sum(cor_j & nz_j))
            train_metrics['hdm_not0_joints']['nonzero'] += int(np.sum(nz_j))

            # --- Partial HDM metrics ----------------------------------------
            for key, s, e in HDM_PARTIAL_SPECS:
                e_clamped = min(e, n_ch_eval)
                if s >= e_clamped:
                    continue
                c = correct_mask[:, :, s:e_clamped]
                train_metrics[key]['correct'] += int(np.sum(c))
                train_metrics[key]['total']   += c.size

            # ----------------------------------------------------------------
            # Display-level metrics per heatmap group (SSIM, MSE, HDM%, confusion)
            # ----------------------------------------------------------------
            for range_idx, (ch_start, ch_stop) in enumerate(HEATMAP_RANGES):
                name = RANGE_NAMES[range_idx]
                for ch in range(ch_start, ch_stop + 1):
                    if ch >= n_ch_eval:
                        break
                    gt_ch   = gt_uint8[:, :, ch]
                    pred_ch = pred_uint8[:, :, ch]

                    hm_acc[name]['SSIM'].append(compute_ssim(gt_ch, pred_ch))
                    hm_acc[name]['MSE'].append(compute_mse(gt_ch, pred_ch))
                    hm_acc[name]['HDM0.01'].append(compute_hdm_display(gt_ch, pred_ch, 0.01))
                    hm_acc[name]['HDM0.1'].append(compute_hdm_display(gt_ch, pred_ch, 0.1))
                    hm_acc[name]['HDM0.3'].append(compute_hdm_display(gt_ch, pred_ch, 0.3))
                    hm_acc[name]['HDM0.5'].append(compute_hdm_display(gt_ch, pred_ch, 0.5))
                    hm_acc[name]['HDM0.8'].append(compute_hdm_display(gt_ch, pred_ch, 0.8))
                    tp, tn, fp, fn = compute_confusion(gt_ch, pred_ch)
                    hm_acc[name]['TP'] += tp
                    hm_acc[name]['TN'] += tn
                    hm_acc[name]['FP'] += fp
                    hm_acc[name]['FN'] += fn

            # ----------------------------------------------------------------
            # Token metrics
            # ----------------------------------------------------------------
            if has_tokens:
                # GloVe cosine similarity (overall + per-token)
                try:
                    pred_emb = estimator.keypoints_model.description()  # (tokensOut, 300) or None
                    if pred_emb is not None and gt_embeddings_batch is not None:
                        gt_emb = gt_embeddings_batch[i]  # (tokensOut, 300)
                        if pred_emb.ndim == 2 and gt_emb.ndim == 2:
                            per_token_cos = []
                            for t in range(min(tokens_out, pred_emb.shape[0], gt_emb.shape[0])):
                                cos = cosine_similarity_1d(pred_emb[t], gt_emb[t])
                                per_token_cos.append(cos)
                                if t < len(token_cosine_per_token):
                                    token_cosine_per_token[t].append(cos)
                            if per_token_cos:
                                token_cosine_sims.append(float(np.mean(per_token_cos)))
                        elif pred_emb.ndim == 1 and gt_emb.ndim == 2 and gt_emb.shape[0] == 1:
                            cos = cosine_similarity_1d(pred_emb, gt_emb[0])
                            token_cosine_sims.append(cos)
                            if token_cosine_per_token:
                                token_cosine_per_token[0].append(cos)
                    elif not cosine_error_reported:
                        cosine_error_reported = True
                        emb_info = type(pred_emb).__name__ if pred_emb is not None else "None"
                        gt_info  = type(gt_embeddings_batch).__name__ if gt_embeddings_batch is not None else "None"
                        print(f"\n[warn] cosine skip: pred_emb={emb_info}, gt_embeddings_batch={gt_info}")
                except Exception as ex:
                    if not cosine_error_reported:
                        cosine_error_reported = True
                        print(f"\n[warn] description() error: {ex}")

                # Multihot accuracy / precision / recall / top-k
                try:
                    pred_mh = estimator.keypoints_model.multihot()  # (1, n_classes) or (n_classes,)
                    if pred_mh is not None and gt_multihot_batch is not None:
                        gt_mh = gt_multihot_batch[i].astype(np.float32)  # (n_classes,)
                        if pred_mh.ndim == 2:
                            pred_mh = pred_mh[0]
                        pred_mh = pred_mh.astype(np.float32)
                        f1, prec, rec, tp, fp, fn = multihot_metrics(gt_mh, pred_mh)
                        token_multihot_f1.append(f1)
                        token_multihot_prec.append(prec)
                        token_multihot_rec.append(rec)
                        token_multihot_TP += tp
                        token_multihot_FP += fp
                        token_multihot_FN += fn
                        # Top-k
                        if len(pred_mh) >= 5:
                            token_top3_acc.append(topk_hit(pred_mh, gt_mh, 3))
                            token_top5_acc.append(topk_hit(pred_mh, gt_mh, 5))

                        # Per-class TP/FP/FN + confusion pairs
                        n_cls = len(gt_mh)
                        if class_TP is None:
                            class_TP = np.zeros(n_cls, dtype=np.int64)
                            class_FP = np.zeros(n_cls, dtype=np.int64)
                            class_FN = np.zeros(n_cls, dtype=np.int64)
                        gt_b   = (gt_mh   > 0.5)
                        pred_b = (pred_mh > 0.5)
                        class_TP += (gt_b  & pred_b)
                        class_FP += (~gt_b & pred_b)
                        class_FN += (gt_b  & ~pred_b)
                        # Confusion pairs: for each (missed GT class, wrongly predicted class)
                        fn_idx = np.where(gt_b  & ~pred_b)[0]
                        fp_idx = np.where(~gt_b & pred_b)[0]
                        for fi in fn_idx:
                            for fj in fp_idx:
                                confusion_ctr[(int(fi), int(fj))] += 1
                except Exception:
                    pass

            samples_done += 1

        # --- Progress -------------------------------------------------------
        elapsed = time.time() - start_time
        fps     = samples_done / max(elapsed, 1e-6)
        eta     = (n_samples - samples_done) / max(fps, 1e-6)
        print(f"\r  [{samples_done}/{n_samples}]  {fps:.1f} fps  ETA {eta:.0f}s      ", end='', flush=True)

    print()

    # -----------------------------------------------------------------------
    # Build and print results
    # -----------------------------------------------------------------------
    results = {
        'serial':        serial,
        'total_samples': n_samples,
        'heatmap_metrics': {},
        'training_equivalent_metrics': {},
        'token_metrics':  {}
    }

    # --- Display-level heatmap metrics -------------------------------------
    print("\n=== Heatmap Metrics (display-level) ===")
    for name in RANGE_NAMES:
        m   = hm_acc[name]
        avg = {}
        for key in ['SSIM', 'MSE', 'HDM0.01', 'HDM0.1', 'HDM0.3', 'HDM0.5', 'HDM0.8']:
            avg[key] = float(np.mean(m[key])) if m[key] else 0.0
        avg['TP'] = m['TP']
        avg['TN'] = m['TN']
        avg['FP'] = m['FP']
        avg['FN'] = m['FN']
        results['heatmap_metrics'][name] = avg

        n_obs = len(m['SSIM'])
        print(f"  {name:15s}  n={n_obs:<6d}  "
              f"SSIM={avg['SSIM']:.4f}  MSE={avg['MSE']:8.2f}  "
              f"HDM@0.1={avg['HDM0.1']:.4f}  HDM@0.5={avg['HDM0.5']:.4f}  "
              f"TP={avg['TP']}  FN={avg['FN']}")

    # --- Training-equivalent HDM metrics -----------------------------------
    print(f"\n=== Training-Equivalent Metrics (HDM threshold={HDM_THRESHOLD}) ===")

    def _hdm_score(key):
        m = train_metrics[key]
        if 'total' in m:
            return m['correct'] / (m['total'] + 1) if m['total'] else 0.0
        else:
            return m['correct'] / (m['nonzero'] + 1e-8) if m['nonzero'] else 0.0

    train_results = {}
    for key in ['hdm', 'hdm_not0', 'hdm_not0_joints'] + [k for k, _, _ in HDM_PARTIAL_SPECS]:
        score = _hdm_score(key)
        train_results[key] = float(score)
        raw = train_metrics[key]
        denom = raw.get('total', raw.get('nonzero', 0))
        print(f"  {key:20s}  {score:.6f}   ({raw['correct']}/{denom})")

    results['training_equivalent_metrics'] = train_results

    # --- Token metrics -----------------------------------------------------
    if has_tokens:
        print("\n=== Token Metrics ===")
        token_res = {
            'glove_cosine_similarity': float(np.mean(token_cosine_sims)) if token_cosine_sims else 0.0,
            'multihot_f1':             float(np.mean(token_multihot_f1))   if token_multihot_f1   else 0.0,
            'multihot_precision':      float(np.mean(token_multihot_prec)) if token_multihot_prec else 0.0,
            'multihot_recall':         float(np.mean(token_multihot_rec))  if token_multihot_rec  else 0.0,
            'multihot_TP': token_multihot_TP,
            'multihot_FP': token_multihot_FP,
            'multihot_FN': token_multihot_FN,
            'top3_accuracy': float(np.mean(token_top3_acc)) if token_top3_acc else 0.0,
            'top5_accuracy': float(np.mean(token_top5_acc)) if token_top5_acc else 0.0,
        }
        # Per-token cosine similarity
        for t in range(tokens_out):
            key = f't{t:02d}'
            sims = token_cosine_per_token[t]
            token_res[f'glove_cosine_{key}'] = float(np.mean(sims)) if sims else 0.0

        results['token_metrics'] = token_res
        for k, v in token_res.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # -----------------------------------------------------------------------
    # Token confusion matrix
    # -----------------------------------------------------------------------
    if has_tokens and class_TP is not None:
        n_cls = len(class_TP)

        def _class_name(idx):
            return vocab.get(idx, str(idx))

        # Per-class stats — only emit classes that appeared in GT or predictions
        per_class = []
        for c in range(n_cls):
            tp = int(class_TP[c])
            fp = int(class_FP[c])
            fn = int(class_FN[c])
            if tp == 0 and fp == 0 and fn == 0:
                continue
            prec  = tp / (tp + fp + 1e-8)
            rec   = tp / (tp + fn + 1e-8)
            f1_c  = 2 * tp / (2 * tp + fp + fn + 1e-8)
            per_class.append({
                'class':     _class_name(c),
                'idx':       c,
                'TP':        tp,
                'FP':        fp,
                'FN':        fn,
                'precision': round(float(prec),  4),
                'recall':    round(float(rec),   4),
                'f1':        round(float(f1_c),  4),
            })
        # Sort by GT occurrence (TP+FN) descending — most frequent classes first
        per_class.sort(key=lambda x: x['TP'] + x['FN'], reverse=True)

        # Confusion pairs — sorted by count descending
        confusion_pairs = [
            {
                'true_label': _class_name(fi),
                'true_idx':   fi,
                'pred_label': _class_name(fj),
                'pred_idx':   fj,
                'count':      cnt,
            }
            for (fi, fj), cnt in sorted(confusion_ctr.items(), key=lambda x: -x[1])
        ]

        confusion_out = {
            'serial':          serial,
            'total_samples':   n_samples,
            'n_classes':       n_cls,
            'n_active_classes': len(per_class),
            'n_confusion_pairs': len(confusion_pairs),
            'per_class':       per_class,
            'confusion_pairs': confusion_pairs,
        }
        with open(confusion_json, 'w') as f:
            json.dump(make_json_serializable(confusion_out), f, indent=2)
        print(f"Token confusion matrix saved to {confusion_json}  "
              f"({len(per_class)} active classes, {len(confusion_pairs)} confusion pairs)")

    # -----------------------------------------------------------------------
    # Skeleton resolution results
    # -----------------------------------------------------------------------
    if do_skeleton_eval and skel_total > 0:
        det_rate = skel_detected / skel_total
        print(f"\n=== Skeleton Resolution Metrics ===")
        print(f"  Detection rate: {skel_detected}/{skel_total}  ({100.0*det_rate:.1f}%)")

        skel_results = {
            'detection_rate': round(det_rate, 6),
            'detected': skel_detected,
            'total':    skel_total,
        }

        for thr in PCK_THRESHOLDS:
            total_h = float(pck_hits[thr].sum())
            total_v = float(pck_visible[thr].sum())
            overall = total_h / max(total_v, 1.0)
            per_joint = {}
            parts = []
            for j in range(n_joints):
                pck_j = float(pck_hits[thr][j]) / max(float(pck_visible[thr][j]), 1.0)
                per_joint[kp_names[j]] = round(pck_j, 4)
                parts.append(f"{kp_names[j][:8]}={pck_j:.3f}")
            print(f"  PCK@{thr:.2f}: {overall:.4f}   " + "  ".join(parts))
            skel_results[f'PCK@{thr}'] = {'overall': round(overall, 6), 'per_joint': per_joint}

        if oks_values:
            mean_oks = float(np.mean(oks_values))
            print(f"  OKS:          {mean_oks:.4f}  (n={len(oks_values)})")
            skel_results['OKS'] = round(mean_oks, 6)

        results['skeleton_metrics'] = skel_results

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    with open(output_json, 'w') as f:
        json.dump(make_json_serializable(results), f, indent=4)
    print(f"\nResults saved to {output_json}")

    # -----------------------------------------------------------------------
    # Generate statistical plots (requires matplotlib; skipped if not installed)
    # -----------------------------------------------------------------------
    # Derive a filename prefix from the JSON output path (strip .json suffix)
    plot_prefix = output_json[:-5] if output_json.endswith('.json') else output_json
    print("\n=== Generating Metric Plots ===")
    # Pass token accumulator lists so token plots are generated when tokens are active
    token_plot_data = None
    if has_tokens:
        token_plot_data = {
            'cosine_sims':      token_cosine_sims,
            'cosine_per_token': token_cosine_per_token,
            'multihot_f1':      token_multihot_f1,
            'multihot_prec':    token_multihot_prec,
            'multihot_rec':     token_multihot_rec,
            'top3_acc':         token_top3_acc,
            'top5_acc':         token_top5_acc,
        }
    plot_heatmap_metrics(hm_acc, train_results, serial, plot_prefix,
                         token_data=token_plot_data)

    # -----------------------------------------------------------------------
    # Auto-run visualizeTokenConfusion.py when a confusion JSON was produced
    # -----------------------------------------------------------------------
    if has_tokens and class_TP is not None and os.path.exists(confusion_json):
        print("\n=== Running visualizeTokenConfusion.py ===")

        # Derive output prefix from confusion_json (strip .json suffix)
        confusion_prefix = confusion_json[:-5] if confusion_json.endswith('.json') else confusion_json

        # GloVe embeddings live inside the model directory
        embeddings_path = os.path.join(model_path, 'GloVe_D300.embeddings')

        cmd = [
            sys.executable, 'visualizeTokenConfusion.py',
            confusion_json,
            '--output',      confusion_prefix,
            '--no-show',                          # non-interactive: save PNGs only
            '--embeddings',  embeddings_path,
        ]
        print(f"  {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"[warn] visualizeTokenConfusion.py exited with code {result.returncode}")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    evaluate()
