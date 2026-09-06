#!/usr/bin/env python3
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

analyzeSegmentationConfusion.py — Pixel-level confusion matrix between YMAPNet's segmentation
categories (Person, Vehicle, Floor, Sky, ... -- whatever `configuration.json['heatmaps']` lists
for the `segmentation` channel group) INCLUDING an explicit "unsegmented" (background) category,
against the ValidationDataset ground truth.

The segmentation output is a stack of independent per-category soft masks (a pixel can legitimately
fire multiple categories at once, e.g. Person + Face), not a softmax -- so to get a classic
pairwise confusion matrix this reduces each pixel to a single *dominant* category: whichever
channel has the highest display value, or "unsegmented" if every channel is below
NONZERO_THRESHOLD_UINT8 (same threshold `evaluateYMAPNet.py` already uses to call a pixel
"non-background"). This is done independently for GT and prediction, then
confusion[gt_label, pred_label] is accumulated over every pixel of every validation image.

Reports, per category (including "unsegmented"): precision, recall, F1, IoU, GT frequency;
overall pixel accuracy and mean IoU; and the top confused (GT, pred) pairs by pixel count.

Outputs:
  <output>.json          : full matrix + labels + per-class metrics + top confused pairs
  <output>_matrix.png     : confusion matrix heatmap (raw counts, log scale + row-normalized %)
  <output>_classes.png    : per-class precision/recall/F1/IoU bar chart, sorted by GT frequency

Usage:
  python3 ymapnet/evaluation/analyzeSegmentationConfusion.py [--cpu] [--model ymapnet_model] [--samples N] [--output out.json] [--no-plots]
"""
import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ymapnet.evaluation.evaluateYMAPNet import (
    DataLoader, YMAPNet, bcolors, checkIfFileExists, loadJSONConfiguration,
    resolveModelDir, rebaseModelPath, gt_int8_to_display_uint8, NONZERO_THRESHOLD_UINT8,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    _HAVE_MATPLOTLIB = True
except ImportError:
    _HAVE_MATPLOTLIB = False

UNSEGMENTED = 'unsegmented'
TOP_CONFUSED_PAIRS = 25


def build_labels(cfg, seg_s, seg_e):
    """[UNSEGMENTED] + one label per segmentation channel, from cfg['heatmaps'] if it has names
    for that range, else a generic 'class_<i>' fallback."""
    names = cfg.get('heatmaps')
    if names and len(names) >= seg_e:
        cat_names = list(names[seg_s:seg_e])
    else:
        cat_names = [f'class_{i}' for i in range(seg_s, seg_e)]
    return [UNSEGMENTED] + cat_names


def dominant_label(display_stack, threshold):
    """display_stack: (H, W, C) uint8-ish. Returns (H, W) int labels in [0, C]:
    0 = UNSEGMENTED (every channel below threshold), else 1 + argmax channel index."""
    amax = np.argmax(display_stack, axis=2)
    vmax = np.max(display_stack, axis=2)
    return np.where(vmax >= threshold, amax + 1, 0).astype(np.int64)


def confusion_metrics(matrix, labels):
    """Per-class precision/recall/F1/IoU + overall pixel accuracy / mean IoU from a dense
    [n_labels, n_labels] matrix, matrix[gt, pred] = pixel count."""
    n = matrix.shape[0]
    row_sum = matrix.sum(axis=1)  # GT frequency per class
    col_sum = matrix.sum(axis=0)  # predicted frequency per class
    diag = np.diag(matrix)
    total = matrix.sum()

    per_class = []
    ious = []
    for i in range(n):
        tp, fn, fp = diag[i], row_sum[i] - diag[i], col_sum[i] - diag[i]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and (precision + recall) > 0 else None)
        union = tp + fp + fn
        iou = tp / union if union > 0 else None
        if iou is not None:
            ious.append(iou)
        per_class.append({
            'class': labels[i], 'gt_pixels': int(row_sum[i]), 'pred_pixels': int(col_sum[i]),
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn),
            'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou,
        })

    pairs = []
    for gi in range(n):
        for pi in range(n):
            if gi != pi and matrix[gi, pi] > 0:
                pairs.append({'gt': labels[gi], 'pred': labels[pi], 'count': int(matrix[gi, pi])})
    pairs.sort(key=lambda p: -p['count'])

    return {
        'labels': labels,
        'total_pixels': int(total),
        'pixel_accuracy': float(diag.sum() / total) if total > 0 else None,
        'mean_iou': float(np.mean(ious)) if ious else None,
        'per_class': per_class,
        'confused_pairs': pairs[:TOP_CONFUSED_PAIRS],
    }


def print_report(result):
    print(f"\n=== Segmentation confusion ({len(result['labels'])} categories incl. '{UNSEGMENTED}') ===")
    print(f"  total pixels: {result['total_pixels']:,}")
    print(f"  pixel accuracy: {result['pixel_accuracy']*100:.2f}%   mean IoU: {result['mean_iou']:.4f}")

    by_freq = sorted(result['per_class'], key=lambda c: -c['gt_pixels'])
    print(f"  {'category':<22}{'gt_px':>14}{'precision':>11}{'recall':>9}{'f1':>8}{'iou':>8}")
    for c in by_freq:
        p = f"{c['precision']:.3f}" if c['precision'] is not None else 'n/a'
        r = f"{c['recall']:.3f}" if c['recall'] is not None else 'n/a'
        f1 = f"{c['f1']:.3f}" if c['f1'] is not None else 'n/a'
        iou = f"{c['iou']:.3f}" if c['iou'] is not None else 'n/a'
        print(f"  {c['class']:<22}{c['gt_pixels']:>14,}{p:>11}{r:>9}{f1:>8}{iou:>8}")

    print(f"\n  top confused (GT -> predicted) pairs:")
    for p in result['confused_pairs']:
        print(f"    {p['count']:>12,}   {p['gt']} -> {p['pred']}")


def plot_matrix_figure(matrix, labels, out_path):
    if not _HAVE_MATPLOTLIB:
        return
    n = len(labels)
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=np.float64), where=row_sum > 0)

    cell = max(0.28, min(0.55, 20.0 / n))
    fig_w = max(12, n * cell * 2 + 3)
    fig_h = max(6, n * cell + 2)
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    im1 = ax_raw.imshow(matrix, cmap='YlOrRd', norm=LogNorm(vmin=1, vmax=max(matrix.max(), 2)))
    ax_raw.set_title('Pixel count (log scale)')
    fig.colorbar(im1, ax=ax_raw, fraction=0.046, pad=0.04)

    im2 = ax_norm.imshow(normalized, cmap='YlOrRd', vmin=0, vmax=1)
    ax_norm.set_title('Row-normalized (recall per GT category)')
    fig.colorbar(im2, ax=ax_norm, fraction=0.046, pad=0.04)

    fsize = max(4, min(8, 140 // n))
    for ax in (ax_raw, ax_norm):
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=75, ha='right', fontsize=fsize)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(labels, fontsize=fsize)
        ax.set_xlabel('predicted')
        ax.set_ylabel('GT')

    fig.suptitle(f'Segmentation pixel confusion matrix  ({n} categories incl. "{UNSEGMENTED}")')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_classes_figure(result, out_path):
    if not _HAVE_MATPLOTLIB:
        return
    rows = sorted(result['per_class'], key=lambda c: c['gt_pixels'])
    names = [c['class'] for c in rows]
    precision = [c['precision'] if c['precision'] is not None else 0.0 for c in rows]
    recall = [c['recall'] if c['recall'] is not None else 0.0 for c in rows]
    f1 = [c['f1'] if c['f1'] is not None else 0.0 for c in rows]
    iou = [c['iou'] if c['iou'] is not None else 0.0 for c in rows]

    y = np.arange(len(names))
    h = 0.2
    fig, ax = plt.subplots(figsize=(11, max(6, len(names) * 0.32)))
    ax.barh(y - 1.5 * h, precision, height=h, label='precision', color='#2980b9')
    ax.barh(y - 0.5 * h, recall, height=h, label='recall', color='#27ae60')
    ax.barh(y + 0.5 * h, f1, height=h, label='F1', color='#c0392b')
    ax.barh(y + 1.5 * h, iou, height=h, label='IoU', color='#8e44ad')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  (n={c['gt_pixels']:,})" for n, c in zip(names, rows)], fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('score')
    ax.set_title(f"Per-category segmentation metrics  (sorted by GT pixel frequency, "
                 f"pixel acc={result['pixel_accuracy']*100:.1f}%, mIoU={result['mean_iou']:.3f})", fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")


def analyze():
    model_path = resolveModelDir()
    output_json = None
    max_samples = None
    make_plots = True
    threshold = NONZERO_THRESHOLD_UINT8
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
        elif argv[i] == '--threshold' and i + 1 < len(argv):
            threshold = int(argv[i + 1])
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
        output_json = f"segmentation_confusion_{serial}.json"
    out_base = os.path.splitext(output_json)[0]

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

    seg_range = db.channel_ranges.get('segmentation') if hasattr(db, 'channel_ranges') else None
    if not seg_range:
        print(bcolors.FAIL, "No 'segmentation' channel group in this model/config -- nothing to analyze.",
              bcolors.ENDC)
        sys.exit(1)
    seg_s, seg_e = seg_range
    labels = build_labels(cfg, seg_s, seg_e)
    n_labels = len(labels)
    print(f"Segmentation channels [{seg_s}..{seg_e-1}] ({seg_e-seg_s} categories) + '{UNSEGMENTED}' "
          f"= {n_labels} confusion-matrix labels")

    n_samples = db.numberOfSamples
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)
    n_channels = int(cfg['outputChannels'])
    print(f"Validation samples: {n_samples}  |  batch_size: {batch_size}  |  channels: {n_channels}")

    print(bcolors.OKGREEN, "Loading YMAPNet model...", bcolors.ENDC)
    estimator = YMAPNet(modelPath=model_path, threshold=0, keypoint_threshold=50.0,
                         engine='tensorflow', profiling=False, compileModel=False, resolve_skeleton=False)

    matrix = np.zeros((n_labels, n_labels), dtype=np.int64)

    samples_done = 0
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        actual_count = batch_end - batch_start
        try:
            npArrayIn, npArrayOut, _npArrayOut16Bit = db.get_partial_update_IO_array(batch_start, batch_end)
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
            if seg_e > n_channels or seg_e > gt_heatmaps.shape[2] or seg_e > n_pred_hm:
                samples_done += 1
                continue

            gt_seg = gt_int8_to_display_uint8(gt_heatmaps[:, :, seg_s:seg_e])
            pred_seg = np.stack(estimator.heatmapsOut[seg_s:seg_e], axis=2)

            gt_label = dominant_label(gt_seg, threshold)
            pred_label = dominant_label(pred_seg, threshold)

            flat = (gt_label * n_labels + pred_label).ravel()
            counts = np.bincount(flat, minlength=n_labels * n_labels)
            matrix += counts.reshape(n_labels, n_labels)

            samples_done += 1
            if samples_done % 50 == 0 or samples_done == n_samples:
                print(f"\r  processed {samples_done}/{n_samples}", end="", flush=True)
    print()

    result = confusion_metrics(matrix, labels)
    result['serial'] = serial
    result['samples'] = samples_done
    result['threshold'] = threshold
    print_report(result)

    if make_plots:
        plot_matrix_figure(matrix, labels, f"{out_base}_matrix.png")
        plot_classes_figure(result, f"{out_base}_classes.png")

    with open(output_json, 'w') as f:
        json.dump(dict(result, matrix=matrix.tolist()), f, indent=2)
    print(f"\nWrote {output_json}")


if __name__ == '__main__':
    analyze()
