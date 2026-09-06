#!/usr/bin/env python3
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

analyzeDepthErrors.py — Identify *modes* in which YMAPNet's depth prediction errs,
for both the 8-bit depth heatmap channel (idx 29) and the 16-bit metric depth head
(hm_16b), against the ValidationDataset ground truth.

Both heads are a tanh projection (0-centered, [-half_range, +half_range]), so besides
magnitude error this also tracks *sign* agreement (which side of the 0-center a pixel
lands on -- a tanh-specific failure mode invisible to MAE/RMSE) and reports every stat
both in raw units and as %-of-range, so the 8-bit (+-120) and 16-bit (+-32767) heads --
different units, same encoding -- are directly comparable. It also cross-checks the two
heads' predictions against each other (rescaled to their own [-1, 1] tanh output) wherever
both have valid GT, since they redundantly encode the same physical depth.

For each head it buckets per-pixel error by:
  - range      : GT depth split into N equal-width bins (near..far)
  - edges      : per-image top-decile GT depth-gradient pixels vs the rest
  - class      : GT semantic mask (person/vehicle/animal/floor) from the
                 segmentation channels the same model already predicts
  - saturation : fraction of valid pixels pinned at the head's output ceiling/floor

and prints ranked lists of the buckets with the worst MAE and worst sign-agreement relative to
the overall, plus a JSON dump of every number and (unless --no-plots) PNG diagnostic figures:
  <output>_8bit.png / _16bit.png : per-bucket MAE + sign-agreement bar charts, and a GT-depth vs
                                    signed-error 2-D density heatmap (the zero-crossing sign-flip
                                    cloud and the range-dependent saturation trend, in one picture)
  <output>_cross.png             : 8-bit vs 16-bit predicted-depth agreement density

Usage:
  python3 ymapnet/evaluation/analyzeDepthErrors.py [--cpu] [--model ymapnet_model] [--samples N] [--output out.json] [--no-plots]
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
from ymapnet.core.NNLosses import absrel as absrel_metric, RMSE as rmse_metric

import cv2

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    _HAVE_MATPLOTLIB = True
except ImportError:
    _HAVE_MATPLOTLIB = False

N_RANGE_BINS = 5
EDGE_PERCENTILE = 90  # top 10% of per-image gradient magnitude counts as "edge"
CLASS_CHANNELS = {name[4:]: (s, e) for name, s, e in HDM_PARTIAL_SPECS
                  if name in ('hdm_person', 'hdm_vehicle', 'hdm_animal', 'hdm_floor')}
HIST_BINS = 60      # GT-depth x signed-error density heatmap, per head
CROSS_HIST_BINS = 60  # 8-bit vs 16-bit predicted-depth agreement density


def new_bucket():
    return {'n': 0, 'sum_abs': 0.0, 'sum_signed': 0.0, 'sum_sq': 0.0, 'sign_agree': 0, 'sign_n': 0}


def accumulate(bucket, err, sign_ok=None):
    """err: 1-D float array of (pred - gt) for the valid pixels of one image.
    sign_ok: same-shape bool array, True where sign(pred) == sign(gt) -- tanh's output is
    0-centered (near/far or front/back of the reference plane), so a sign flip is a distinct
    failure mode from a magnitude error and would be invisible in MAE/RMSE alone."""
    if err.size == 0:
        return
    bucket['n'] += int(err.size)
    bucket['sum_abs'] += float(np.sum(np.abs(err)))
    bucket['sum_signed'] += float(np.sum(err))
    bucket['sum_sq'] += float(np.sum(err * err))
    if sign_ok is not None and sign_ok.size:
        bucket['sign_agree'] += int(np.sum(sign_ok))
        bucket['sign_n'] += int(sign_ok.size)


def bucket_summary(bucket, half_range=None):
    n = bucket['n']
    if n == 0:
        return {'n': 0, 'mae': None, 'bias': None, 'rmse': None, 'sign_agreement': None}
    out = {
        'n': n,
        'mae': bucket['sum_abs'] / n,
        'bias': bucket['sum_signed'] / n,
        'rmse': float(np.sqrt(bucket['sum_sq'] / n)),
        'sign_agreement': (bucket['sign_agree'] / bucket['sign_n']) if bucket['sign_n'] else None,
    }
    # %-of-full-range versions so the 8-bit (+-120) and 16-bit (+-32767) heads,
    # which live on completely different tanh-rescaled units, can be compared directly.
    if half_range:
        out['mae_pct'] = out['mae'] / half_range * 100.0
        out['bias_pct'] = out['bias'] / half_range * 100.0
        out['rmse_pct'] = out['rmse'] / half_range * 100.0
    return out


def new_head_state(range_lo, range_hi):
    edges = np.linspace(range_lo, range_hi, N_RANGE_BINS + 1)
    half_range = max(abs(range_lo), abs(range_hi))
    return {
        'overall': new_bucket(),
        'range_edges': edges.tolist(),
        'half_range': half_range,
        'range_bins': [new_bucket() for _ in range(N_RANGE_BINS)],
        'edge': new_bucket(),
        'flat': new_bucket(),
        'classes': {name: new_bucket() for name in CLASS_CHANNELS},
        'classes_other': new_bucket(),
        'sat_lo': 0,
        'sat_hi': 0,
        'valid_total': 0,
        'absrel_sum': 0.0,
        'absrel_n': 0,
        # GT-depth (x) vs signed-error (y) 2-D density, streamed across images for the
        # diagnostic heatmap -- cheap to accumulate (fixed-size histogram, no raw pixels kept).
        'hist_gt_edges': np.linspace(range_lo, range_hi, HIST_BINS + 1),
        'hist_err_edges': np.linspace(-half_range, half_range, HIST_BINS + 1),
        'hist': np.zeros((HIST_BINS, HIST_BINS), dtype=np.float64),
    }


def analyze_head(state, gt_raw, pred, sat_lo_value, sat_hi_value, sat_tol, gt_heatmaps, n_pred_hm, n_ch_eval):
    """gt_raw, pred: 2-D float arrays, same units (raw signed depth scale). 0 in gt_raw == no GT data."""
    valid = gt_raw != 0
    if not np.any(valid):
        return
    err_full = pred - gt_raw
    # sign(0) == 0, which would count as "agreeing" with a positive/negative GT -- only
    # score sign agreement where the prediction has actually committed to a side of 0.
    sign_ok_full = (np.sign(pred) == np.sign(gt_raw)) & (pred != 0)
    state['valid_total'] += int(np.sum(valid))

    accumulate(state['overall'], err_full[valid], sign_ok_full[valid])

    h, _, _ = np.histogram2d(
        np.clip(gt_raw[valid], state['hist_gt_edges'][0], state['hist_gt_edges'][-1]),
        np.clip(err_full[valid], state['hist_err_edges'][0], state['hist_err_edges'][-1]),
        bins=[state['hist_gt_edges'], state['hist_err_edges']])
    state['hist'] += h

    # AbsRel (skip near-zero GT to avoid blow-up from the relative-error denominator)
    rel_mask = valid & (np.abs(gt_raw) > 1e-6)
    if np.any(rel_mask):
        rel = np.abs(err_full[rel_mask]) / np.abs(gt_raw[rel_mask])
        state['absrel_sum'] += float(np.sum(rel))
        state['absrel_n'] += int(rel.size)

    # --- range bins ---------------------------------------------------------
    edges = state['range_edges']
    bin_idx = np.clip(np.digitize(gt_raw, edges[1:-1]), 0, N_RANGE_BINS - 1)
    for b in range(N_RANGE_BINS):
        m = valid & (bin_idx == b)
        accumulate(state['range_bins'][b], err_full[m], sign_ok_full[m])

    # --- edge vs flat (per-image gradient magnitude of GT) -------------------
    gt_for_grad = np.where(valid, gt_raw, 0.0).astype(np.float32)
    gx = cv2.Sobel(gt_for_grad, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gt_for_grad, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    valid_grad_vals = grad_mag[valid]
    if valid_grad_vals.size > 0:
        thresh = np.percentile(valid_grad_vals, EDGE_PERCENTILE)
        edge_mask = valid & (grad_mag >= thresh)
        flat_mask = valid & ~edge_mask
        accumulate(state['edge'], err_full[edge_mask], sign_ok_full[edge_mask])
        accumulate(state['flat'], err_full[flat_mask], sign_ok_full[flat_mask])

    # --- semantic class (from GT segmentation channels) ----------------------
    covered = np.zeros_like(valid)
    for name, (ch_s, _ch_e) in CLASS_CHANNELS.items():
        if ch_s >= n_ch_eval or ch_s >= n_pred_hm or ch_s >= gt_heatmaps.shape[2]:
            continue
        class_mask = valid & (gt_int8_to_display_uint8(gt_heatmaps[:, :, ch_s]) >= NONZERO_THRESHOLD_UINT8)
        accumulate(state['classes'][name], err_full[class_mask], sign_ok_full[class_mask])
        covered |= class_mask
    accumulate(state['classes_other'], err_full[valid & ~covered], sign_ok_full[valid & ~covered])

    # --- saturation / clipping ------------------------------------------------
    state['sat_lo'] += int(np.sum(valid & (pred <= sat_lo_value + sat_tol)))
    state['sat_hi'] += int(np.sum(valid & (pred >= sat_hi_value - sat_tol)))


def finalize_head(state):
    hr = state['half_range']
    out = {
        'valid_pixels': state['valid_total'],
        'half_range': hr,
        'overall': bucket_summary(state['overall'], hr),
        'absrel': (state['absrel_sum'] / state['absrel_n']) if state['absrel_n'] else None,
        'range_bins': [dict(bucket_summary(b, hr), lo=state['range_edges'][i], hi=state['range_edges'][i + 1])
                        for i, b in enumerate(state['range_bins'])],
        'edge': bucket_summary(state['edge'], hr),
        'flat': bucket_summary(state['flat'], hr),
        'classes': {name: bucket_summary(b, hr) for name, b in state['classes'].items()},
        'other': bucket_summary(state['classes_other'], hr),
        'saturation_lo_frac': (state['sat_lo'] / state['valid_total']) if state['valid_total'] else 0.0,
        'saturation_hi_frac': (state['sat_hi'] / state['valid_total']) if state['valid_total'] else 0.0,
    }
    return out


def new_cross_state():
    edges = np.linspace(-1.0, 1.0, CROSS_HIST_BINS + 1)
    return {'n': 0, 'sum_x': 0.0, 'sum_y': 0.0, 'sum_xy': 0.0, 'sum_x2': 0.0, 'sum_y2': 0.0,
            'sum_abs_diff': 0.0, 'sign_agree': 0,
            'hist_edges': edges, 'hist': np.zeros((CROSS_HIST_BINS, CROSS_HIST_BINS), dtype=np.float64)}


def accumulate_cross(cross, pred8_norm, pred16_norm, mask):
    """pred8_norm, pred16_norm: both rescaled to their head's [-1, 1] tanh output, so the
    8-bit and 16-bit heads' disagreement can be measured on a common scale."""
    x = pred8_norm[mask]
    y = pred16_norm[mask]
    if x.size == 0:
        return
    cross['n'] += int(x.size)
    cross['sum_x'] += float(np.sum(x))
    cross['sum_y'] += float(np.sum(y))
    cross['sum_xy'] += float(np.sum(x * y))
    cross['sum_x2'] += float(np.sum(x * x))
    cross['sum_y2'] += float(np.sum(y * y))
    cross['sum_abs_diff'] += float(np.sum(np.abs(x - y)))
    cross['sign_agree'] += int(np.sum(np.sign(x) == np.sign(y)))
    edges = cross['hist_edges']
    h, _, _ = np.histogram2d(np.clip(x, edges[0], edges[-1]), np.clip(y, edges[0], edges[-1]),
                              bins=[edges, edges])
    cross['hist'] += h


def finalize_cross(cross):
    n = cross['n']
    if n == 0:
        return None
    mean_x, mean_y = cross['sum_x'] / n, cross['sum_y'] / n
    var_x = cross['sum_x2'] / n - mean_x ** 2
    var_y = cross['sum_y2'] / n - mean_y ** 2
    cov = cross['sum_xy'] / n - mean_x * mean_y
    corr = (cov / np.sqrt(var_x * var_y)) if var_x > 1e-12 and var_y > 1e-12 else None
    return {
        'n': n,
        'correlation': float(corr) if corr is not None else None,
        'mean_abs_diff_normalized': cross['sum_abs_diff'] / n,
        'sign_agreement': cross['sign_agree'] / n,
    }


def print_cross_report(cross_summary):
    print("\n=== 8-bit vs 16-bit head agreement (both rescaled to their own [-1, 1] tanh output) ===")
    if cross_summary is None:
        print("  no pixels with both heads valid")
        return
    print(f"  pixels compared: {cross_summary['n']:,}")
    print(f"  correlation: {cross_summary['correlation']:.4f}" if cross_summary['correlation'] is not None
          else "  correlation: n/a")
    print(f"  mean |normalized diff|: {cross_summary['mean_abs_diff_normalized']:.4f}")
    print(f"  sign agreement (same side of 0-center): {cross_summary['sign_agreement']*100:.2f}%")


def named_buckets_list(summary):
    """[(short_label, full_label, bucket_dict), ...] for every named bucket with data --
    shared by the console report and the bar-chart plots so the two never drift apart."""
    named = []
    for b in summary['range_bins']:
        short = f"range [{b['lo']:.0f}, {b['hi']:.0f}]"
        named.append((short, f"{short} (n={b['n']:,})", b))
    named.append(("edge/boundary", f"edge/boundary pixels (n={summary['edge']['n']:,})", summary['edge']))
    named.append(("flat/interior", f"flat/interior pixels (n={summary['flat']['n']:,})", summary['flat']))
    for name, b in summary['classes'].items():
        named.append((f"class '{name}'", f"class '{name}' (n={b['n']:,})", b))
    named.append(("other/background", f"class 'other/background' (n={summary['other']['n']:,})", summary['other']))
    return [(s, f, b) for s, f, b in named if b['mae'] is not None]


def print_report(head_name, summary):
    overall_mae = summary['overall']['mae']
    print(f"\n=== {head_name} depth head (tanh output range +-{summary['half_range']:.0f}) ===")
    if overall_mae is None:
        print("  no valid GT pixels found")
        return
    ov = summary['overall']
    print(f"  valid pixels: {summary['valid_pixels']:,}")
    print(f"  overall  MAE={ov['mae']:.3f} ({ov['mae_pct']:.2f}% of range)  bias={ov['bias']:+.3f}  "
          f"RMSE={ov['rmse']:.3f}"
          + (f"  AbsRel={summary['absrel']:.3f}" if summary['absrel'] is not None else ""))
    print(f"  sign agreement (pred/GT on the same side of the tanh's 0-center): "
          f"{ov['sign_agreement']*100:.2f}%" if ov['sign_agreement'] is not None else "  sign agreement: n/a")
    print(f"  saturation: {summary['saturation_lo_frac']*100:.2f}% at floor, "
          f"{summary['saturation_hi_frac']*100:.2f}% at ceiling")

    named_buckets = [(full, b) for _short, full, b in named_buckets_list(summary)]

    modes = sorted(named_buckets, key=lambda t: t[1]['mae'], reverse=True)
    print("  worst-to-best buckets by MAE (relative to overall):")
    for label, b in modes:
        rel = (b['mae'] / overall_mae - 1.0) * 100.0 if overall_mae > 0 else 0.0
        print(f"    {b['mae']:8.3f}  ({rel:+6.1f}% vs overall)  {label}")

    sign_buckets = [(label, b) for label, b in named_buckets if b['sign_agreement'] is not None]
    sign_buckets.sort(key=lambda t: t[1]['sign_agreement'])
    print("  worst-to-best buckets by sign-agreement (front/back-of-center confusion):")
    for label, b in sign_buckets:
        print(f"    {b['sign_agreement']*100:6.2f}%  {label}")


def plot_head_figure(head_name, summary, state, out_path):
    """<out_path>: MAE-by-bucket bar, sign-agreement-by-bucket bar, and a GT-depth vs
    signed-error density heatmap (visualizes the zero-crossing sign-flip cloud and the
    range-dependent saturation trend in one picture)."""
    if not _HAVE_MATPLOTLIB or summary['overall']['mae'] is None:
        return
    named = named_buckets_list(summary)
    hr = summary['half_range']

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])
    ax_mae, ax_sign, ax_hist = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    mae_rows = sorted(named, key=lambda t: t[2]['mae'])
    ax_mae.barh([s for s, _f, _b in mae_rows], [b['mae'] for _s, _f, b in mae_rows], color='#c0392b')
    ax_mae.axvline(summary['overall']['mae'], color='black', linestyle='--', linewidth=1, label='overall MAE')
    ax_mae.set_xlabel('MAE (raw units)')
    ax_mae.set_title(f'{head_name}: MAE by bucket')
    ax_mae.legend(fontsize=8)

    sign_rows = sorted([(s, f, b) for s, f, b in named if b['sign_agreement'] is not None],
                        key=lambda t: t[2]['sign_agreement'])
    if sign_rows:
        ax_sign.barh([s for s, _f, _b in sign_rows], [b['sign_agreement'] * 100 for _s, _f, b in sign_rows],
                     color='#2980b9')
        if summary['overall']['sign_agreement'] is not None:
            ax_sign.axvline(summary['overall']['sign_agreement'] * 100, color='black', linestyle='--',
                            linewidth=1, label='overall')
        ax_sign.set_xlabel('sign agreement (%)')
        ax_sign.set_title(f'{head_name}: front/back-of-center agreement by bucket')
        ax_sign.legend(fontsize=8)

    hist = state['hist'].T  # rows=error bins, cols=gt bins, so imshow reads (gt=x, err=y)
    gt_edges, err_edges = state['hist_gt_edges'], state['hist_err_edges']
    im = ax_hist.imshow(hist, origin='lower', aspect='auto', cmap='inferno',
                         norm=LogNorm(vmin=1, vmax=max(hist.max(), 2)),
                         extent=[gt_edges[0], gt_edges[-1], err_edges[0], err_edges[-1]])
    ax_hist.axhline(0, color='cyan', linestyle='--', linewidth=1)
    ax_hist.axvline(0, color='cyan', linestyle='--', linewidth=1)
    ax_hist.set_xlabel('GT depth (raw units)')
    ax_hist.set_ylabel('signed error  (pred - GT)')
    ax_hist.set_title(f'{head_name}: error density vs GT depth  (dashed lines = the tanh 0-center)')
    fig.colorbar(im, ax=ax_hist, label='pixel count (log scale)')

    fig.suptitle(f'{head_name} depth head diagnostics  (+-{hr:.0f} tanh range, '
                 f'{summary["valid_pixels"]:,} valid pixels)')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_cross_figure(cross_summary, cross_state, out_path):
    """8-bit vs 16-bit predicted-depth agreement density, both rescaled to [-1, 1]."""
    if not _HAVE_MATPLOTLIB or cross_summary is None:
        return
    edges = cross_state['hist_edges']
    hist = cross_state['hist'].T  # rows=y(16-bit), cols=x(8-bit)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(hist, origin='lower', aspect='auto', cmap='inferno',
                    norm=LogNorm(vmin=1, vmax=max(hist.max(), 2)),
                    extent=[edges[0], edges[-1], edges[0], edges[-1]])
    ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color='cyan', linestyle='--', linewidth=1,
            label='perfect agreement')
    ax.set_xlabel('8-bit head, normalized to [-1, 1]')
    ax.set_ylabel('16-bit head, normalized to [-1, 1]')
    ax.set_title(f"8-bit vs 16-bit predicted depth  (r={cross_summary['correlation']:.4f}, "
                 f"sign agreement={cross_summary['sign_agreement']*100:.1f}%)")
    ax.legend(fontsize=8, loc='upper left')
    fig.colorbar(im, ax=ax, label='pixel count (log scale)')
    fig.tight_layout()
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
    if cfg.get('embeddingsPath'):
        cfg['embeddingsPath'] = rebaseModelPath(cfg['embeddingsPath'], model_path)
    if cfg.get('synonymPath'):
        _sp = cfg['synonymPath']
        cfg['synonymPath'] = rebaseModelPath(_sp, model_path) if isinstance(_sp, str) \
            else [rebaseModelPath(p, model_path) for p in _sp]

    if output_json is None:
        output_json = f"depth_error_analysis_{serial}.json"

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
    print(f"Validation samples: {n_samples}  |  batch_size: {batch_size}  |  channels: {n_channels}")

    print(bcolors.OKGREEN, "Loading YMAPNet model...", bcolors.ENDC)
    estimator = YMAPNet(modelPath=model_path, threshold=0, keypoint_threshold=50.0,
                         engine='tensorflow', profiling=False, compileModel=False, resolve_skeleton=False)

    state8 = new_head_state(hm_inactive, hm_active)
    have_16b = int(cfg['output16BitChannels']) > 0
    state16 = new_head_state(-32767, 32767) if have_16b else None
    cross = new_cross_state() if have_16b else None

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
            pred8 = pred16 = None
            if depth_ch < n_ch_eval:
                gt8_raw = gt_heatmaps[:, :, depth_ch].astype(np.float32)
                pred8 = estimator.heatmapsOut[depth_ch].astype(np.float32) - abs(hm_inactive)
                analyze_head(state8, gt8_raw, pred8, hm_inactive, hm_active, sat_tol=1.0,
                             gt_heatmaps=gt_heatmaps, n_pred_hm=n_pred_hm, n_ch_eval=n_ch_eval)

            if have_16b and npArrayOut16Bit is not None and estimator.heatmap_16b is not None:
                gt16_raw = npArrayOut16Bit[i][:, :, 0].astype(np.float32)
                pred16 = np.asarray(estimator.heatmap_16b)[:, :, 0].astype(np.float32)
                analyze_head(state16, gt16_raw, pred16, -32767, 32767, sat_tol=32767 * 0.01,
                             gt_heatmaps=gt_heatmaps, n_pred_hm=n_pred_hm, n_ch_eval=n_ch_eval)

            # 8-bit vs 16-bit disparity -- compare where BOTH GTs have real depth data,
            # each head's prediction rescaled back to its own [-1, 1] tanh output.
            if pred8 is not None and pred16 is not None:
                common_valid = (gt8_raw != 0) & (gt16_raw != 0)
                accumulate_cross(cross, pred8 / hm_active, pred16 / 32767.0, common_valid)

            samples_done += 1
            if samples_done % 50 == 0 or samples_done == n_samples:
                print(f"\r  processed {samples_done}/{n_samples}", end="", flush=True)
    print()

    out_base = os.path.splitext(output_json)[0]

    summary8 = finalize_head(state8)
    print_report("8-bit", summary8)
    result = {'serial': serial, 'samples': samples_done, '8bit': summary8}
    if make_plots:
        plot_head_figure("8-bit", summary8, state8, f"{out_base}_8bit.png")

    if have_16b:
        summary16 = finalize_head(state16)
        print_report("16-bit metric", summary16)
        result['16bit'] = summary16
        if make_plots:
            plot_head_figure("16-bit metric", summary16, state16, f"{out_base}_16bit.png")

        cross_summary = finalize_cross(cross)
        print_cross_report(cross_summary)
        result['cross_8bit_16bit'] = cross_summary
        if make_plots:
            plot_cross_figure(cross_summary, cross, f"{out_base}_cross.png")
    else:
        print("\n(no 16-bit depth head in this model configuration)")

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {output_json}")


if __name__ == '__main__':
    analyze()
