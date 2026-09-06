#!/usr/bin/env python3
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

analyzeNormalsErrors.py — Identify *modes* in which YMAPNet's surface-normal prediction
errs (channels 30-32, nx/ny/nz), against the ValidationDataset ground truth. Sibling tool
to analyzeDepthErrors.py, reusing its edge/flat + semantic-class bucketing and its
validity gate: normals are computed on the GPU-side data loader as the Sobel gradient of
the depth channel (datasets/DataLoader/processing/normals.h), so wherever GT depth is 0
("no data") the derived normal is meaningless too -- same `gt_depth != 0` mask as depth.

Unlike depth, a normal is a 3-vector, so the primary error metric is *angular* error
(degrees between predicted and GT unit vectors) rather than per-channel MAE -- this is
the standard surface-normal-estimation protocol (mean angular error + %-of-pixels within
11.25/22.5/30 degrees). Per-channel (nx/ny/nz) MAE/bias/RMSE is still reported as a
secondary summary. Buckets:
  - facing-angle : GT normal's angle from the camera axis (0=facing camera, 90=grazing),
                   the normals analog of depth's near/far range bins
  - edge/flat    : per-image top-decile GT depth-gradient pixels vs the rest (same surface
                   discontinuities that break depth also break the normals derived from it)
  - class        : GT semantic mask (person/vehicle/animal/floor), same as depth
  - saturation   : fraction of valid pixels with a component pinned at +-120 (grazing /
                   axis-aligned normals, most easily saturated)

Prints ranked lists of the buckets with the worst mean angular error and worst tight-
threshold (11.25 deg) accuracy relative to the overall, plus a JSON dump of every number
and (unless --no-plots) a PNG diagnostic figure:
  <output>.png : per-bucket mean-angular-error + accuracy-within-11.25deg bar charts,
                 per-channel MAE bar, and a facing-angle vs angular-error 2-D density
                 heatmap (grazing angles are the classic hard case for normal estimation).

Usage:
  python3 ymapnet/evaluation/analyzeNormalsErrors.py [--cpu] [--model ymapnet_model] [--samples N] [--output out.json] [--no-plots]
"""
import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ymapnet.evaluation.evaluateYMAPNet import (
    DataLoader, YMAPNet, bcolors, checkIfFileExists, loadJSONConfiguration,
    resolveModelDir, rebaseModelPath, gt_int8_to_display_uint8,
    HDM_PARTIAL_SPECS, NONZERO_THRESHOLD_UINT8,
)
from ymapnet.evaluation.analyzeDepthErrors import new_bucket, accumulate, bucket_summary, CLASS_CHANNELS

import cv2

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    _HAVE_MATPLOTLIB = True
except ImportError:
    _HAVE_MATPLOTLIB = False

N_ANGLE_BINS = 5
EDGE_PERCENTILE = 90  # top 10% of per-image GT-depth gradient magnitude counts as "edge"
ANGLE_THRESHOLDS_DEG = [11.25, 22.5, 30.0]  # standard surface-normal-estimation accuracy thresholds
HIST_BINS = 60  # facing-angle x angular-error density heatmap
CHANNEL_NAMES = ['nx', 'ny', 'nz']
VEC_EPS = 1e-6


def new_angle_bucket():
    return {'n': 0, 'sum_deg': 0.0, 'sum_deg_sq': 0.0, 'within': [0] * len(ANGLE_THRESHOLDS_DEG)}


def accumulate_angle(bucket, deg_err):
    if deg_err.size == 0:
        return
    bucket['n'] += int(deg_err.size)
    bucket['sum_deg'] += float(np.sum(deg_err))
    bucket['sum_deg_sq'] += float(np.sum(deg_err * deg_err))
    for i, t in enumerate(ANGLE_THRESHOLDS_DEG):
        bucket['within'][i] += int(np.sum(deg_err <= t))


def angle_bucket_summary(bucket):
    n = bucket['n']
    if n == 0:
        return {'n': 0, 'mean_deg': None, 'rmse_deg': None, 'pct_within': {}}
    return {
        'n': n,
        'mean_deg': bucket['sum_deg'] / n,
        'rmse_deg': float(np.sqrt(bucket['sum_deg_sq'] / n)),
        'pct_within': {str(t): bucket['within'][i] / n * 100.0 for i, t in enumerate(ANGLE_THRESHOLDS_DEG)},
    }


def new_head_state():
    edges = np.linspace(0.0, 90.0, N_ANGLE_BINS + 1)
    return {
        'overall': new_angle_bucket(),
        'angle_edges': edges.tolist(),
        'angle_bins': [new_angle_bucket() for _ in range(N_ANGLE_BINS)],
        'edge': new_angle_bucket(),
        'flat': new_angle_bucket(),
        'classes': {name: new_angle_bucket() for name in CLASS_CHANNELS},
        'classes_other': new_angle_bucket(),
        'channel_err': {name: new_bucket() for name in CHANNEL_NAMES},
        'sat': {name: 0 for name in CHANNEL_NAMES},
        'valid_total': 0,
        # GT facing-angle (x) vs angular-error (y) 2-D density, streamed across images.
        'hist_angle_edges': np.linspace(0.0, 90.0, HIST_BINS + 1),
        'hist_err_edges': np.linspace(0.0, 180.0, HIST_BINS + 1),
        'hist': np.zeros((HIST_BINS, HIST_BINS), dtype=np.float64),
    }


def analyze_head(state, gt_depth_raw, gt_normal_raw, pred_normal_raw, sat_value, sat_tol,
                  gt_heatmaps, n_pred_hm, n_ch_eval):
    """gt_depth_raw: 2-D float array, raw signed depth scale, 0 == no GT data (gates validity
    for the normals derived from it). gt_normal_raw, pred_normal_raw: HxWx3 float arrays,
    raw signed scale (unit vector * 120, per normals.h)."""
    valid = gt_depth_raw != 0
    if not np.any(valid):
        return
    state['valid_total'] += int(np.sum(valid))

    gt_len = np.linalg.norm(gt_normal_raw, axis=-1, keepdims=True)
    pred_len = np.linalg.norm(pred_normal_raw, axis=-1, keepdims=True)
    gt_unit = gt_normal_raw / np.maximum(gt_len, VEC_EPS)
    pred_unit = pred_normal_raw / np.maximum(pred_len, VEC_EPS)

    dot = np.clip(np.sum(gt_unit * pred_unit, axis=-1), -1.0, 1.0)
    ang_err = np.degrees(np.arccos(dot))
    facing_deg = np.degrees(np.arccos(np.clip(gt_unit[..., 2], -1.0, 1.0)))

    accumulate_angle(state['overall'], ang_err[valid])

    h, _, _ = np.histogram2d(
        np.clip(facing_deg[valid], state['hist_angle_edges'][0], state['hist_angle_edges'][-1]),
        np.clip(ang_err[valid], state['hist_err_edges'][0], state['hist_err_edges'][-1]),
        bins=[state['hist_angle_edges'], state['hist_err_edges']])
    state['hist'] += h

    # --- facing-angle bins (normals analog of depth's near/far range bins) --
    edges = state['angle_edges']
    bin_idx = np.clip(np.digitize(facing_deg, edges[1:-1]), 0, N_ANGLE_BINS - 1)
    for b in range(N_ANGLE_BINS):
        m = valid & (bin_idx == b)
        accumulate_angle(state['angle_bins'][b], ang_err[m])

    # --- edge vs flat (per-image gradient magnitude of GT depth -- same surface
    # discontinuities that also break the depth-derived normals) -------------
    gt_for_grad = np.where(valid, gt_depth_raw, 0.0).astype(np.float32)
    gx = cv2.Sobel(gt_for_grad, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gt_for_grad, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    valid_grad_vals = grad_mag[valid]
    if valid_grad_vals.size > 0:
        thresh = np.percentile(valid_grad_vals, EDGE_PERCENTILE)
        edge_mask = valid & (grad_mag >= thresh)
        flat_mask = valid & ~edge_mask
        accumulate_angle(state['edge'], ang_err[edge_mask])
        accumulate_angle(state['flat'], ang_err[flat_mask])

    # --- semantic class (from GT segmentation channels) ----------------------
    covered = np.zeros_like(valid)
    for name, (ch_s, _ch_e) in CLASS_CHANNELS.items():
        if ch_s >= n_ch_eval or ch_s >= n_pred_hm or ch_s >= gt_heatmaps.shape[2]:
            continue
        class_mask = valid & (gt_int8_to_display_uint8(gt_heatmaps[:, :, ch_s]) >= NONZERO_THRESHOLD_UINT8)
        accumulate_angle(state['classes'][name], ang_err[class_mask])
        covered |= class_mask
    accumulate_angle(state['classes_other'], ang_err[valid & ~covered])

    # --- per-component raw error (secondary summary) + saturation ------------
    for i, name in enumerate(CHANNEL_NAMES):
        err = pred_normal_raw[..., i] - gt_normal_raw[..., i]
        accumulate(state['channel_err'][name], err[valid])
        state['sat'][name] += int(np.sum(valid & (np.abs(pred_normal_raw[..., i]) >= sat_value - sat_tol)))


def finalize_head(state):
    valid_total = state['valid_total']
    return {
        'valid_pixels': valid_total,
        'overall': angle_bucket_summary(state['overall']),
        'angle_bins': [dict(angle_bucket_summary(b), lo=state['angle_edges'][i], hi=state['angle_edges'][i + 1])
                       for i, b in enumerate(state['angle_bins'])],
        'edge': angle_bucket_summary(state['edge']),
        'flat': angle_bucket_summary(state['flat']),
        'classes': {name: angle_bucket_summary(b) for name, b in state['classes'].items()},
        'other': angle_bucket_summary(state['classes_other']),
        'channel_mae': {name: bucket_summary(b, half_range=120) for name, b in state['channel_err'].items()},
        'saturation_frac': {name: (state['sat'][name] / valid_total) if valid_total else 0.0
                             for name in CHANNEL_NAMES},
    }


def named_buckets_list(summary):
    """[(short_label, full_label, bucket_dict), ...] for every named bucket with data --
    shared by the console report and the plot so the two never drift apart."""
    named = []
    for b in summary['angle_bins']:
        short = f"facing [{b['lo']:.0f}, {b['hi']:.0f}] deg"
        named.append((short, f"{short} (n={b['n']:,})", b))
    named.append(("edge/boundary", f"edge/boundary pixels (n={summary['edge']['n']:,})", summary['edge']))
    named.append(("flat/interior", f"flat/interior pixels (n={summary['flat']['n']:,})", summary['flat']))
    for name, b in summary['classes'].items():
        named.append((f"class '{name}'", f"class '{name}' (n={b['n']:,})", b))
    named.append(("other/background", f"class 'other/background' (n={summary['other']['n']:,})", summary['other']))
    return [(s, f, b) for s, f, b in named if b['mean_deg'] is not None]


def print_report(summary):
    ov = summary['overall']
    print("\n=== Normals head (nx/ny/nz, tanh output range +-120) ===")
    if ov['mean_deg'] is None:
        print("  no valid GT pixels found")
        return
    print(f"  valid pixels: {summary['valid_pixels']:,}")
    print(f"  overall  mean angular error={ov['mean_deg']:.2f} deg  RMSE={ov['rmse_deg']:.2f} deg")
    within_str = "  ".join(f"<{t} deg: {ov['pct_within'][str(t)]:.2f}%" for t in ANGLE_THRESHOLDS_DEG)
    print(f"  accuracy: {within_str}")
    for name in CHANNEL_NAMES:
        cm = summary['channel_mae'][name]
        print(f"  {name}: MAE={cm['mae']:.3f} ({cm['mae_pct']:.2f}% of range)  bias={cm['bias']:+.3f}  "
              f"RMSE={cm['rmse']:.3f}  saturation={summary['saturation_frac'][name]*100:.2f}%")

    named_buckets = [(full, b) for _short, full, b in named_buckets_list(summary)]

    modes = sorted(named_buckets, key=lambda t: t[1]['mean_deg'], reverse=True)
    print("  worst-to-best buckets by mean angular error (relative to overall):")
    for label, b in modes:
        rel = (b['mean_deg'] / ov['mean_deg'] - 1.0) * 100.0 if ov['mean_deg'] > 0 else 0.0
        print(f"    {b['mean_deg']:8.2f} deg  ({rel:+6.1f}% vs overall)  {label}")

    tight = str(ANGLE_THRESHOLDS_DEG[0])
    acc_buckets = sorted(named_buckets, key=lambda t: t[1]['pct_within'][tight])
    print(f"  worst-to-best buckets by accuracy within {ANGLE_THRESHOLDS_DEG[0]} deg:")
    for label, b in acc_buckets:
        print(f"    {b['pct_within'][tight]:6.2f}%  {label}")


def plot_head_figure(summary, state, out_path):
    """<out_path>: mean-angular-error bar, accuracy-within-11.25deg bar, per-channel MAE bar,
    and a facing-angle vs angular-error density heatmap (grazing-angle failure trend)."""
    if not _HAVE_MATPLOTLIB or summary['overall']['mean_deg'] is None:
        return
    named = named_buckets_list(summary)
    tight = str(ANGLE_THRESHOLDS_DEG[0])

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])
    ax_mae = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_ch = fig.add_subplot(gs[0, 2])
    ax_hist = fig.add_subplot(gs[1, :])

    mae_rows = sorted(named, key=lambda t: t[2]['mean_deg'])
    ax_mae.barh([s for s, _f, _b in mae_rows], [b['mean_deg'] for _s, _f, b in mae_rows], color='#c0392b')
    ax_mae.axvline(summary['overall']['mean_deg'], color='black', linestyle='--', linewidth=1, label='overall')
    ax_mae.set_xlabel('mean angular error (deg)')
    ax_mae.set_title('Normals: angular error by bucket')
    ax_mae.legend(fontsize=8)

    acc_rows = sorted(named, key=lambda t: t[2]['pct_within'][tight])
    ax_acc.barh([s for s, _f, _b in acc_rows], [b['pct_within'][tight] for _s, _f, b in acc_rows], color='#2980b9')
    ax_acc.axvline(summary['overall']['pct_within'][tight], color='black', linestyle='--', linewidth=1,
                    label='overall')
    ax_acc.set_xlabel(f'accuracy within {ANGLE_THRESHOLDS_DEG[0]} deg (%)')
    ax_acc.set_title('Normals: tight-threshold accuracy by bucket')
    ax_acc.legend(fontsize=8)

    ch_mae = [summary['channel_mae'][name]['mae'] for name in CHANNEL_NAMES]
    ax_ch.bar(CHANNEL_NAMES, ch_mae, color='#8e44ad')
    ax_ch.set_ylabel('MAE (raw units, +-120 range)')
    ax_ch.set_title('Normals: per-component MAE')

    hist = state['hist'].T  # rows=angular-error bins, cols=facing-angle bins
    angle_edges, err_edges = state['hist_angle_edges'], state['hist_err_edges']
    im = ax_hist.imshow(hist, origin='lower', aspect='auto', cmap='inferno',
                         norm=LogNorm(vmin=1, vmax=max(hist.max(), 2)),
                         extent=[angle_edges[0], angle_edges[-1], err_edges[0], err_edges[-1]])
    ax_hist.set_xlabel('GT facing angle (deg from camera axis; 0=facing camera, 90=grazing)')
    ax_hist.set_ylabel('angular error (deg)')
    ax_hist.set_title('Normals: error density vs GT facing angle')
    fig.colorbar(im, ax=ax_hist, label='pixel count (log scale)')

    fig.suptitle(f'Normals head diagnostics  ({summary["valid_pixels"]:,} valid pixels)')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def analyze():
    model_path = resolveModelDir()
    output_json = None
    max_samples = None
    make_plots = True
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--model' and i + 1 < len(argv):
            model_path = argv[i + 1]
            i += 2
        elif argv[i] == '--samples' and i + 1 < len(argv):
            max_samples = int(argv[i + 1])
            i += 2
        elif argv[i] == '--output' and i + 1 < len(argv):
            output_json = argv[i + 1]
            i += 2
        elif argv[i] == '--no-plots':
            make_plots = False
            i += 1
        elif argv[i] == '--cpu':
            i += 1  # handled at import time by evaluateYMAPNet
        else:
            i += 1

    if make_plots and not _HAVE_MATPLOTLIB:
        print(bcolors.FAIL, "matplotlib not available -- skipping plots (pip install matplotlib)", bcolors.ENDC)
        make_plots = False

    json_path = os.path.join(model_path, 'configuration.json')
    if not checkIfFileExists(json_path):
        print(bcolors.FAIL, f"Configuration not found: {json_path}", bcolors.ENDC)
        sys.exit(1)

    cfg = loadJSONConfiguration(json_path, useRAMfs=False)
    serial = cfg.get('serial', 'unknown')
    if not cfg.get('heatmapAddNormals'):
        print(bcolors.FAIL, "This model was not configured with heatmapAddNormals -- nothing to analyze",
              bcolors.ENDC)
        sys.exit(1)
    if cfg.get('embeddingsPath'):
        cfg['embeddingsPath'] = rebaseModelPath(cfg['embeddingsPath'], model_path)
    if cfg.get('synonymPath'):
        _sp = cfg['synonymPath']
        cfg['synonymPath'] = rebaseModelPath(_sp, model_path) if isinstance(_sp, str) \
            else [rebaseModelPath(p, model_path) for p in _sp]

    if output_json is None:
        output_json = f"normals_error_analysis_{serial}.json"

    vocab_path = os.path.join(model_path, 'vocabulary.json')

    batch_size = int(cfg['batchSize'])
    print(bcolors.OKGREEN, "Creating validation DataLoader...", bcolors.ENDC)
    db = DataLoader((cfg['inputHeight'], cfg['inputWidth'], cfg['inputChannels']),
                     (cfg['outputHeight'], cfg['outputWidth'], cfg['outputChannels']),
                     output16BitChannels=cfg['output16BitChannels'], numberOfThreads=cfg['DatasetLoaderThreads'],
                     streamData=1, batchSize=batch_size, gradientSize=cfg['heatmapGradientSizeMinimum'],
                     PAFSize=cfg['heatmapPAFSizeMinimum'], doAugmentations=0,
                     addPAFs=int(cfg['heatmapAddPAFs']), addBackground=int(cfg['heatmapGenerateSkeletonBkg']),
                     addDepthMap=int(cfg['heatmapAddDepthmap']), addDepthLevelsHeatmaps=int(cfg['heatmapAddDepthLevels']),
                     addNormals=int(cfg['heatmapAddNormals']), addSegmentation=int(cfg['heatmapAddSegmentation']),
                     addInstanceDetection=int(cfg['heatmapAddInstanceDetection']),
                     vocabularyPath=vocab_path, synonymPath=cfg.get('synonymPath', None),
                     embeddingsPath=cfg.get('embeddingsPath', None),
                     datasets=cfg['ValidationDataset'], libraryPath="datasets/DataLoader/libDataLoader.so")

    n_samples = db.numberOfSamples
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)
    n_channels = int(cfg['outputChannels'])
    hm_active = int(cfg.get('heatmapActive', 120))
    hm_inactive = int(cfg.get('heatmapDeactivated', -120))
    depth_ch = next(s for name, s, e in HDM_PARTIAL_SPECS if name == 'hdm_depth')
    normal_s, normal_e = next((s, e) for name, s, e in HDM_PARTIAL_SPECS if name == 'hdm_normal')
    print(f"Validation samples: {n_samples}  |  batch_size: {batch_size}  |  channels: {n_channels}")

    print(bcolors.OKGREEN, "Loading YMAPNet model...", bcolors.ENDC)
    estimator = YMAPNet(modelPath=model_path, threshold=0, keypoint_threshold=50.0,
                         engine='tensorflow', profiling=False, compileModel=False, resolve_skeleton=False)

    state = new_head_state()

    samples_done = 0
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        actual_count = batch_end - batch_start
        try:
            npArrayIn, npArrayOut, npArrayOut16Bit = db.get_partial_update_IO_array(batch_start, batch_end)
        except Exception as e:
            print(f"\nFailed to load batch {batch_start}-{batch_end}: {e}")
            samples_done += actual_count
            continue

        for i in range(actual_count):
            rgb_input = npArrayIn[i]
            gt_heatmaps = npArrayOut[i]
            try:
                estimator.process(rgb_input)
            except Exception as e:
                print(f"\nInference failed on sample {batch_start+i}: {e}")
                continue

            n_pred_hm = len(estimator.heatmapsOut)
            n_ch_eval = min(n_channels, gt_heatmaps.shape[2], n_pred_hm)
            if normal_e <= n_ch_eval and depth_ch < n_ch_eval:
                gt_depth_raw = gt_heatmaps[:, :, depth_ch].astype(np.float32)
                gt_normal_raw = gt_heatmaps[:, :, normal_s:normal_e].astype(np.float32)
                pred_normal_raw = np.stack(
                    [estimator.heatmapsOut[c].astype(np.float32) - abs(hm_inactive) for c in range(normal_s, normal_e)],
                    axis=-1)
                analyze_head(state, gt_depth_raw, gt_normal_raw, pred_normal_raw,
                             sat_value=hm_active, sat_tol=1.0,
                             gt_heatmaps=gt_heatmaps, n_pred_hm=n_pred_hm, n_ch_eval=n_ch_eval)

            samples_done += 1
            if samples_done % 50 == 0 or samples_done == n_samples:
                print(f"\r  processed {samples_done}/{n_samples}", end="", flush=True)
    print()

    summary = finalize_head(state)
    print_report(summary)
    result = {'serial': serial, 'samples': samples_done, 'normals': summary}

    if make_plots:
        out_base = os.path.splitext(output_json)[0]
        plot_head_figure(summary, state, f"{out_base}.png")

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {output_json}")


if __name__ == '__main__':
    analyze()
