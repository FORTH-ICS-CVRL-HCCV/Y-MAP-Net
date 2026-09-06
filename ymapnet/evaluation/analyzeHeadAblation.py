#!/usr/bin/env python3
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

analyzeHeadAblation.py — Causal attention-head ablation for YMAPNet's bottleneck self-attention
block (`bottleneck_attention_block`, NNModel.py, `useBottleneckAttention`), inspired by the
"visual retrieval heads" methodology (arXiv 2608.27417): rather than guessing which heads matter
from attention-weight inspection alone, zero out each head's contribution one at a time and measure
how much the model's actual task performance degrades. A head whose ablation barely moves any task
is dead weight; a head whose ablation tanks one task but not others is that task's dedicated capacity.

Mechanism: `keras.layers.MultiHeadAttention` combines heads via its output-projection EinsumDense,
kernel shape (num_heads, key_dim, out_dim) -- zeroing kernel[h, :, :] removes head h's entire
contribution to the block's output (and, through the residual, to everything downstream) without
touching any other head or requiring a model rebuild. Weights are restored after every config, so
the in-memory model is unmodified once the script exits (nothing is written back to disk).

For each config (baseline = unmodified, then one config per ablated head) this runs a validation
pass and reports three independent task metrics, so a head's importance can be told apart *by task*:
  - depth   : 8-bit depth head MAE / AbsRel / sign-agreement (reuses analyzeDepthErrors.py's
              accumulator -- 16-bit is skipped since Observation 27 already found the two heads
              ~99% redundant, and this script pays for an extra full pass per head)
  - segmentation : pixel accuracy / mean IoU (reuses analyzeSegmentationConfusion.py's dominant-
              label confusion matrix)
  - pose    : mean-squared-error over the 'keypoints' display heatmaps (compute_mse, same metric
              evaluateYMAPNet.py already uses per heatmap group)

Outputs:
  <output>.json        : per-config metrics for all three tasks + %-degradation-vs-baseline
  <output>_bars.png     : per-head %-degradation bar chart, one panel per task (unless --no-plots)

Usage:
  python3 ymapnet/evaluation/analyzeHeadAblation.py [--cpu] [--model ymapnet_model] [--samples N]
      [--layer bottleneck_attn_mha] [--output out.json] [--no-plots]

Default --samples is 500 (not the full validation set) since this runs one pass per head+1 baseline;
pass a larger --samples (with a longer shell timeout) for a full-validation-set run.
"""
import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ymapnet.evaluation.evaluateYMAPNet import (
    DataLoader, YMAPNet, bcolors, checkIfFileExists, loadJSONConfiguration,
    resolveModelDir, rebaseModelPath, compute_mse, HDM_PARTIAL_SPECS, gt_int8_to_display_uint8,
)
from ymapnet.evaluation.analyzeDepthErrors import new_head_state, analyze_head, finalize_head
from ymapnet.evaluation.analyzeSegmentationConfusion import (
    build_labels, dominant_label, confusion_metrics, UNSEGMENTED, NONZERO_THRESHOLD_UINT8,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MATPLOTLIB = True
except ImportError:
    _HAVE_MATPLOTLIB = False

DEFAULT_LAYER = 'bottleneck_attn_mha'
DEFAULT_SAMPLES = 500


def get_output_dense(model, layer_name):
    layer = model.get_layer(layer_name)
    output_dense = getattr(layer, '_output_dense', None)
    if output_dense is None or not hasattr(output_dense, 'kernel'):
        print(bcolors.FAIL, f"Layer '{layer_name}' has no '_output_dense.kernel' -- "
              "not a keras.layers.MultiHeadAttention head-combination layer?", bcolors.ENDC)
        sys.exit(1)
    kernel = output_dense.kernel
    if len(kernel.shape) != 3:
        print(bcolors.FAIL, f"Unexpected output_dense kernel shape {kernel.shape} "
              "(expected (num_heads, key_dim, out_dim))", bcolors.ENDC)
        sys.exit(1)
    return output_dense


def set_ablated_head(output_dense, original_kernel, head_idx):
    """head_idx=None restores the original kernel (baseline / unablated)."""
    if head_idx is None:
        output_dense.kernel.assign(original_kernel)
        return
    kernel = original_kernel.copy()
    kernel[head_idx, :, :] = 0.0
    output_dense.kernel.assign(kernel)


def new_pose_state():
    return {'sum_mse': 0.0, 'n': 0}


def accumulate_pose(state, gt_ch, pred_ch):
    state['sum_mse'] += compute_mse(gt_ch, pred_ch)
    state['n'] += 1


def finalize_pose(state):
    return {'mse': (state['sum_mse'] / state['n']) if state['n'] else None, 'n': state['n']}


def run_config(estimator, db, n_samples, batch_size, depth_ch, hm_active, hm_inactive,
               seg_range, seg_labels, kp_range, n_channels):
    seg_s, seg_e = seg_range
    kp_s, kp_e = kp_range
    n_seg_labels = len(seg_labels)

    depth_state = new_head_state(hm_inactive, hm_active)
    seg_matrix = np.zeros((n_seg_labels, n_seg_labels), dtype=np.int64)
    pose_state = new_pose_state()

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
            n_ch_eval = min(n_channels, gt_heatmaps.shape[2], n_pred_hm)

            if depth_ch < n_ch_eval:
                gt8_raw = gt_heatmaps[:, :, depth_ch].astype(np.float32)
                pred8 = estimator.heatmapsOut[depth_ch].astype(np.float32) - abs(hm_inactive)
                analyze_head(depth_state, gt8_raw, pred8, hm_inactive, hm_active, sat_tol=1.0,
                             gt_heatmaps=gt_heatmaps, n_pred_hm=n_pred_hm, n_ch_eval=n_ch_eval)

            if seg_e <= n_ch_eval:
                gt_seg = gt_int8_to_display_uint8(gt_heatmaps[:, :, seg_s:seg_e])
                pred_seg = np.stack(estimator.heatmapsOut[seg_s:seg_e], axis=2)
                gt_label = dominant_label(gt_seg, NONZERO_THRESHOLD_UINT8)
                pred_label = dominant_label(pred_seg, NONZERO_THRESHOLD_UINT8)
                flat = (gt_label * n_seg_labels + pred_label).ravel()
                counts = np.bincount(flat, minlength=n_seg_labels * n_seg_labels)
                seg_matrix += counts.reshape(n_seg_labels, n_seg_labels)

            if kp_e <= n_ch_eval:
                gt_kp = gt_int8_to_display_uint8(gt_heatmaps[:, :, kp_s:kp_e])
                pred_kp = np.stack(estimator.heatmapsOut[kp_s:kp_e], axis=2)
                accumulate_pose(pose_state, gt_kp, pred_kp)

            samples_done += 1
            if samples_done % 50 == 0 or samples_done == n_samples:
                print(f"\r    processed {samples_done}/{n_samples}", end="", flush=True)
    print()

    return {
        'depth': finalize_head(depth_state),
        'segmentation': confusion_metrics(seg_matrix, seg_labels),
        'pose': finalize_pose(pose_state),
    }


def pct_delta(ablated, baseline, higher_is_better):
    """% change vs baseline, signed so positive always means 'worse'."""
    if ablated is None or baseline is None or baseline == 0:
        return None
    raw_pct = (ablated - baseline) / abs(baseline) * 100.0
    return -raw_pct if higher_is_better else raw_pct


def print_comparison(results, baseline_config):
    base = results[baseline_config]
    print(f"\n=== Head ablation impact vs '{baseline_config}' "
          f"(positive % = worse than baseline) ===")
    header = f"  {'config':<10}{'depth MAE%Δ':>13}{'depth sign%Δ':>14}{'seg mIoU%Δ':>13}{'pose MSE%Δ':>13}"
    print(header)
    for name, r in results.items():
        d_mae = pct_delta(r['depth']['overall']['mae'], base['depth']['overall']['mae'], higher_is_better=False)
        d_sign = pct_delta(r['depth']['overall']['sign_agreement'], base['depth']['overall']['sign_agreement'],
                            higher_is_better=True)
        s_iou = pct_delta(r['segmentation']['mean_iou'], base['segmentation']['mean_iou'], higher_is_better=True)
        p_mse = pct_delta(r['pose']['mse'], base['pose']['mse'], higher_is_better=False)
        fmt = lambda v: f"{v:+.2f}" if v is not None else "n/a"
        print(f"  {name:<10}{fmt(d_mae):>13}{fmt(d_sign):>14}{fmt(s_iou):>13}{fmt(p_mse):>13}")


def plot_bars(results, baseline_config, num_heads, out_path):
    if not _HAVE_MATPLOTLIB:
        return
    base = results[baseline_config]
    heads = list(range(num_heads))

    def series(metric_path, higher_is_better):
        vals = []
        for h in heads:
            r = results[f'head_{h}']
            a = r
            for key in metric_path:
                a = a[key]
            b = base
            for key in metric_path:
                b = b[key]
            vals.append(pct_delta(a, b, higher_is_better) or 0.0)
        return vals

    depth_mae = series(['depth', 'overall', 'mae'], higher_is_better=False)
    seg_iou = series(['segmentation', 'mean_iou'], higher_is_better=True)
    pose_mse = series(['pose', 'mse'], higher_is_better=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    titles = ['Depth MAE  (%Δ vs baseline, worse=positive)',
              'Segmentation mean IoU  (%Δ vs baseline, worse=positive)',
              'Pose (keypoints) MSE  (%Δ vs baseline, worse=positive)']
    colors = ['#c0392b', '#8e44ad', '#2980b9']
    for ax, vals, title, color in zip(axes, [depth_mae, seg_iou, pose_mse], titles, colors):
        bar_colors = [color if v >= 0 else '#95a5a6' for v in vals]
        ax.bar(heads, vals, color=bar_colors)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel('ablated head index')
        ax.set_ylabel('% degradation')
        ax.set_title(title, fontsize=9)
        ax.set_xticks(heads)

    fig.suptitle('Bottleneck attention head-ablation impact by task')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def analyze():
    model_path = resolveModelDir()
    output_json = None
    max_samples = DEFAULT_SAMPLES
    make_plots = True
    layer_name = DEFAULT_LAYER
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
        elif argv[i] == '--layer' and i + 1 < len(argv):
            layer_name = argv[i + 1]
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

    if not cfg.get('useBottleneckAttention', False):
        print(bcolors.FAIL, f"cfg['useBottleneckAttention'] is False for this model -- "
              "there is no attention block to ablate.", bcolors.ENDC)
        sys.exit(1)

    if output_json is None:
        output_json = f"head_ablation_{serial}.json"
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

    seg_range = db.channel_ranges.get('segmentation')
    kp_range = db.channel_ranges.get('keypoints')
    if not seg_range or not kp_range:
        print(bcolors.FAIL, "Need both 'segmentation' and 'keypoints' channel groups enabled.", bcolors.ENDC)
        sys.exit(1)
    seg_labels = build_labels(cfg, *seg_range)

    n_samples = db.numberOfSamples
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)
    n_channels = int(cfg['outputChannels'])
    hm_active = int(cfg.get('heatmapActive', 120))
    hm_inactive = int(cfg.get('heatmapDeactivated', -120))
    depth_ch = next(s for name, s, e in HDM_PARTIAL_SPECS if name == 'hdm_depth')
    print(f"Validation samples/config: {n_samples}  |  batch_size: {batch_size}  |  channels: {n_channels}")

    print(bcolors.OKGREEN, "Loading YMAPNet model...", bcolors.ENDC)
    estimator = YMAPNet(modelPath=model_path, threshold=0, keypoint_threshold=50.0,
                         engine='tensorflow', profiling=False, compileModel=False, resolve_skeleton=False)

    keras_model = estimator.keypoints_model.model.model
    output_dense = get_output_dense(keras_model, layer_name)
    original_kernel = output_dense.kernel.numpy().copy()
    num_heads = original_kernel.shape[0]
    print(f"Ablating layer '{layer_name}': {num_heads} heads, output kernel shape {original_kernel.shape}")

    configs = [('baseline', None)] + [(f'head_{h}', h) for h in range(num_heads)]
    results = {}
    try:
        for name, head_idx in configs:
            print(f"\n--- config '{name}' ---")
            set_ablated_head(output_dense, original_kernel, head_idx)
            results[name] = run_config(estimator, db, n_samples, batch_size, depth_ch, hm_active, hm_inactive,
                                        seg_range, seg_labels, kp_range, n_channels)
            r = results[name]
            print(f"    depth MAE={r['depth']['overall']['mae']:.3f}  "
                  f"sign-agree={r['depth']['overall']['sign_agreement']*100:.2f}%  "
                  f"seg mIoU={r['segmentation']['mean_iou']:.4f}  pose MSE={r['pose']['mse']:.3f}")
    finally:
        set_ablated_head(output_dense, original_kernel, None)  # always leave the model unablated

    print_comparison(results, 'baseline')

    if make_plots:
        plot_bars(results, 'baseline', num_heads, f"{out_base}_bars.png")

    out = {'serial': serial, 'layer': layer_name, 'num_heads': num_heads, 'samples_per_config': n_samples,
           'results': results}
    with open(output_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {output_json}")


if __name__ == '__main__':
    analyze()
