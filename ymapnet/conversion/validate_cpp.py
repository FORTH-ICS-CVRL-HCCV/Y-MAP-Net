#!/usr/bin/env python3
"""
Phase 4 validation: compare C++ ggml heatmap output against the Keras baseline.

Usage:
    python3 validate_cpp.py --image 1730235538988943.jpg \
                            --model 2d_pose_estimation \
                            --cpp-bin ymapnet_cpp/build/ymapnet \
                            --gguf   2d_pose_estimation/model_f32.gguf

Pass / Fail thresholds (on the [-120, 120] scale):
    F32 weights vs Python mixed_bfloat16 : max |diff| < 20.0
    F16 weights vs Python mixed_bfloat16 : max |diff| < 30.0

  Note: the Keras model computes in mixed_bfloat16, so comparing against F32 C++
  expects accumulated precision differences.  To compare C++ F32 against a true
  F32 Python baseline, add --force-fp32 (not yet implemented).
"""

import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


def center_crop(image):
    h, w = image.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def keras_heatmap(model_path: str, image_path: str, target_size: int = 256) -> np.ndarray:
    """Run the Keras model and return the raw 8-bit and 16-bit heatmaps [H, W, C] float32."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from ymapnet.core.NNModel import load_keypoints_model

    model, input_size, output_size, num_heatmaps = load_keypoints_model(model_path, compile=False)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    cropped = center_crop(bgr)
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Model has an internal Rescaling(1/255) layer — pass raw uint8-range floats
    batch = np.expand_dims(rgb.astype(np.float32), axis=0)
    pred = model.predict(batch, verbose=0)
    # Multi-output model: pred is a list where pred[0] is the heatmap [1, H, W, C]
    # Single-output model: pred is [1, H, W, C] directly
    hm = pred[0] if isinstance(pred, (list, tuple)) else pred
    # 16-bit head ("hm_16b") is the second output when present
    hm16 = pred[1][0] if isinstance(pred, (list, tuple)) and len(pred) > 1 and pred[1].ndim == 4 else None
    return hm[0], hm16  # [H, W, C], [H, W, C16] or None


def cpp_heatmap(cpp_bin: str, gguf_path: str, image_path: str, target_size: int = 256) -> np.ndarray:
    """Run the C++ binary with --dump, return raw 8-bit and 16-bit heatmaps [H, W, C] float32."""
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            cpp_bin,
            '--model',
            gguf_path,
            '--from',
            image_path,
            '--dump',
            tmp_path,
            '--headless',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f'C++ binary failed (exit {result.returncode})')

        def load(path):
            # C++ writes ggml [W, H, C] = C-order [C, H, W] → rearrange to [H, W, C]
            arr = np.fromfile(path, dtype=np.float32)
            return arr.reshape(-1, target_size, target_size).transpose(1, 2, 0)

        hm = load(tmp_path)
        hm16 = load(tmp_path + '.16bit') if os.path.exists(tmp_path + '.16bit') else None
        return hm, hm16
    finally:
        for p in (tmp_path, tmp_path + '.16bit'):
            if os.path.exists(p):
                os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='1730235538988943.jpg')
    ap.add_argument('--model', default='ymapnet_model/model.keras')
    ap.add_argument('--cpp-bin', default='ymapnet_cpp/build/ymapnet')
    ap.add_argument('--gguf', default='ymapnet_model/model_f32.gguf')
    ap.add_argument('--dtype', default='f32', choices=['f32', 'f16'])
    args = ap.parse_args()

    print(f'[validate] Keras inference on {args.image} ...')
    py_hm, py_hm16 = keras_heatmap(args.model, args.image)
    print(f'[validate] Keras heatmap: shape={py_hm.shape}  min={py_hm.min():.3f}  max={py_hm.max():.3f}')

    print(f'[validate] C++ inference on {args.image} ...')
    cpp_hm, cpp_hm16 = cpp_heatmap(args.cpp_bin, args.gguf, args.image)
    print(f'[validate] C++ heatmap:   shape={cpp_hm.shape}  min={cpp_hm.min():.3f}  max={cpp_hm.max():.3f}')

    diff = np.abs(py_hm - cpp_hm)
    max_diff = diff.max()
    mean_diff = diff.mean()
    # Keras runs mixed_bfloat16; C++ runs F32.  Accumulated bfloat16 error
    # on a deep network of ~100 ops can reach ~10-15 on the [-120,120] scale.
    threshold = 20.0 if args.dtype == 'f32' else 30.0
    passed = max_diff < threshold

    print()
    print('=== Validation Results ===')
    print(f'  max  |diff| = {max_diff:.4f}  (threshold {threshold:.1f})')
    print(f'  mean |diff| = {mean_diff:.4f}')
    per_ch_max = diff.max(axis=(0, 1))  # [C]
    worst_ch = int(per_ch_max.argmax())
    print(f'  worst channel = {worst_ch}  (max diff {per_ch_max[worst_ch]:.4f})')
    print(f'  {"PASS" if passed else "FAIL"}')

    # Per-channel summary (top 5 worst channels)
    top5 = np.argsort(per_ch_max)[::-1][:5]
    print('\n  Top-5 worst channels:')
    for ch in top5:
        print(f'    ch {ch:3d}  max|diff|={per_ch_max[ch]:.4f}')

    # 16-bit head (depth, [-32767, 32767] scale) — report relative to the 8-bit scale
    if py_hm16 is not None and cpp_hm16 is not None:
        d16 = np.abs(py_hm16 - cpp_hm16)
        print(f'\n  16-bit head: max|diff|={d16.max():.1f}  mean|diff|={d16.mean():.1f}  '
              f'(= {d16.max() * 120.0 / 32767.0:.3f} / {d16.mean() * 120.0 / 32767.0:.4f} on the 120 scale)')

    # Find the Python-active pixel (max of pre branch) and compare
    peak_h, peak_w = np.unravel_index(py_hm[:, :, :17].max(axis=2).argmax(), py_hm.shape[:2])
    peak_ch = int(py_hm[:, :, :17].max(axis=(0, 1)).argmax())
    print(f'\n  Python peak pixel [{peak_h},{peak_w}] ch={peak_ch}:')
    print(f'    py  = {py_hm[peak_h,peak_w,peak_ch]:.3f}')
    print(f'    cpp = {cpp_hm[peak_h,peak_w,peak_ch]:.3f}')
    print(f'\n  All channels at peak pixel [{peak_h},{peak_w}]:')
    for ch in range(min(10, py_hm.shape[2])):
        print(f'    ch{ch:2d}  py={py_hm[peak_h,peak_w,ch]:8.3f}  cpp={cpp_hm[peak_h,peak_w,ch]:8.3f}')

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
