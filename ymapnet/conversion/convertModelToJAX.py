#!/usr/bin/python3

# Force CPU-only before TensorFlow is imported when requested.
import os
import sys

useGPU = True
if len(sys.argv) > 1:
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--cpu':
            useGPU = False

if not useGPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # <- Force CPU
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Convert a trained Keras model to JAX-compatible formats.

Three output formats are supported:

  npz       (default) — flat numpy archive of all layer weights.
                        Load with np.load(); plug into any JAX/Flax model
                        that shares the same architecture.

  orbax     — orbax-checkpoint directory (standard JAX ecosystem format).
               Load with orbax.checkpoint.PyTreeCheckpointer().restore().

  stablehlo — StableHLO MLIR artifact via jax2tf.
               Produces model_jax.mlirbc (bytecode, JAX >= 0.4.14) or
               model_jax.mlir (text fallback).  Runnable anywhere JAX's
               XLA backend is available without TensorFlow.

A model_jax_meta.json sidecar is written for all formats, recording the
input/output shapes and parameter count.

Requirements:
    All formats:   pip install jax jaxlib tensorflow keras
    orbax only:    pip install orbax-checkpoint
    stablehlo:     JAX >= 0.4.14 recommended

Usage:
    python3 convertModelToJAX.py [options]

    --model PATH   Path to .keras model file  (default: 2d_pose_estimation/model.keras)
    --output PATH  Output directory           (default: 2d_pose_estimation)
    --npz          Save as flat numpy archive (default)
    --orbax        Save as orbax checkpoint directory
    --stablehlo    Export StableHLO MLIR artifact via jax2tf
    --verify       Run a test inference through JAX after conversion
    --cpu          Force CPU execution (also forces CPU for StableHLO export if VRAM is limited)
"""

import argparse
import json
import numpy as np

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_arg_parser():
    p = argparse.ArgumentParser(description='Convert Keras model to JAX')
    p.add_argument('--model', default='2d_pose_estimation/model.keras', help='Input .keras model path')
    p.add_argument('--output', default='2d_pose_estimation', help='Output directory')
    p.add_argument('--cpu', dest='cpu', action='store_const', const='cpu', help='Force CPU execution')
    p.add_argument('--verify', action='store_true', help='Run a test inference via JAX after conversion')
    p.add_argument('--fp16', action='store_true',
                   help='Export in float16 (I/O and weights; internal compute cast by XLA)')
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument('--npz', dest='format', action='store_const', const='npz',
                     help='Save weights as flat numpy .npz archive (default)')
    fmt.add_argument('--orbax', dest='format', action='store_const', const='orbax',
                     help='Save weights as orbax checkpoint directory')
    fmt.add_argument('--stablehlo', dest='format', action='store_const', const='stablehlo',
                     help='Export StableHLO MLIR artifact via jax2tf')
    p.set_defaults(format='npz')
    return p


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def extract_params(model):
    """Return an ordered flat dict {variable_path: numpy_array}."""
    params = {}
    for var in model.variables:
        # Keras 3: var.path (e.g. "dense/kernel")
        # Keras 2: var.name (e.g. "dense/kernel:0")
        key = getattr(var, 'path', None) or var.name.rstrip(':0')
        key = key.replace('/', '.')  # make safe as npz / dict key
        params[key] = var.numpy()
    return params


def save_metadata(model, output_path):
    """Write a JSON sidecar with input/output shapes and param count."""
    output_shapes = model.output_shape
    if not isinstance(output_shapes, list):
        output_shapes = [output_shapes]
    meta = {
        'input_shape': list(model.input_shape),
        'output_shapes': [list(s) for s in output_shapes],
        'num_params': int(model.count_params()),
    }
    with open(output_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Metadata saved → {output_path}')
    return meta


# ---------------------------------------------------------------------------
# Format: npz
# ---------------------------------------------------------------------------


def convert_npz(model, output_dir, fp16=False):
    params = extract_params(model)
    suffix = '_fp16' if fp16 else ''
    npz_path = os.path.join(output_dir, f'model_jax{suffix}.npz')
    meta_path = os.path.join(output_dir, 'model_jax_meta.json')

    if fp16:
        params = {k: v.astype(np.float16) for k, v in params.items()}
        print(f'Casting {len(params)} weight tensors to float16 ...')

    print(f'Saving {len(params)} weight tensors → {npz_path} ...')
    np.savez(npz_path, **params)
    save_metadata(model, meta_path)

    size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f'NPZ saved → {npz_path}  ({size_mb:.1f} MB)')
    return npz_path


# ---------------------------------------------------------------------------
# Format: orbax
# ---------------------------------------------------------------------------


def convert_orbax(model, output_dir):
    try:
        import orbax.checkpoint as ocp
    except ImportError:
        print('ERROR: orbax-checkpoint not found.  Install with:  pip install orbax-checkpoint')
        sys.exit(1)

    params = extract_params(model)
    ckpt_dir = os.path.abspath(os.path.join(output_dir, 'model_jax_orbax'))
    meta_path = os.path.join(output_dir, 'model_jax_meta.json')

    print(f'Saving orbax checkpoint → {ckpt_dir} ...')
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(ckpt_dir, params)
    save_metadata(model, meta_path)
    print(f'Orbax checkpoint saved → {ckpt_dir}')
    return ckpt_dir


# ---------------------------------------------------------------------------
# Format: stablehlo
# ---------------------------------------------------------------------------


def convert_stablehlo(model, output_dir, fp16=False):
    try:
        import jax
        import jax.numpy as jnp
        from jax.experimental import jax2tf
        import tensorflow as tf
    except ImportError as e:
        print(f'ERROR: {e}')
        print('       Install with:  pip install jax jaxlib tensorflow')
        sys.exit(1)

    shape = model.input_shape  # (None, H, W, C)
    H, W, C = int(shape[1]), int(shape[2]), int(shape[3])
    jnp_dtype = jnp.float16 if fp16 else jnp.float32
    dummy = jnp.zeros([1, H, W, C], dtype=jnp_dtype)

    dtype_label = 'float16' if fp16 else 'float32'
    print(f'Tracing model via jax2tf.call_tf (input {H}×{W}×{C}, {dtype_label}) ...')

    if fp16:
        # Wrap: cast fp16 input → fp32 for the model, cast fp32 output → fp16.
        # XLA fuses these casts and dispatches mixed-precision ops on Tensor Cores.
        def _model_fp16(x):
            out = model(tf.cast(x, tf.float32), training=False)
            if isinstance(out, (list, tuple)):
                return [tf.cast(o, tf.float16) for o in out]
            return tf.cast(out, tf.float16)

        tf_fn = tf.function(_model_fp16)
    else:
        # Notes:
        #  - jit_compile=True is omitted: it asks TF to emit XLA HLO on GPU, which JAX then
        #    cannot trace from CPU (device mismatch). Plain tf.function is fine here.
        #  - call_tf_graph=True is omitted: that flag only works inside jax2tf.convert(),
        #    not under jax.jit / jax.export. Without it call_tf dispatches TF eagerly from JAX.
        tf_fn = tf.function(lambda x: model(x, training=False))

    jax_fn = jax2tf.call_tf(tf_fn)
    jit_fn = jax.jit(jax_fn)

    # Export to StableHLO bytecode (.mlirbc).
    # Pin to exactly 1 device inside jax.default_device() so the artifact has
    # nr_devices=1.  Without this, older JAX versions mark the artifact as
    # nr_devices=0 (multi-device), which JAX 0.9+ refuses to call eagerly in a
    # single-device context.
    #
    # API evolution:
    #   JAX 0.9+      : jax.export.export(fn)(args)  — public API
    #   JAX 0.4.28-0.8: lowered.export()             — semi-public
    #   older          : fallback to MLIR text
    _device = jax.local_devices()[0]
    suffix = '_fp16' if fp16 else ''
    out_path = os.path.join(output_dir, f'model_jax{suffix}.mlirbc')
    exported = None
    with jax.default_device(_device):
        # Try JAX 0.9+ public API: jax.export.export
        try:
            _export_fn = getattr(jax, 'export', None)
            if _export_fn is not None and hasattr(_export_fn, 'export'):
                exported = _export_fn.export(jax.jit(jax_fn))(dummy)
        except Exception:
            exported = None

        # Fallback: lowered.export() (JAX 0.4.28–0.8)
        if exported is None:
            try:
                lowered = jit_fn.lower(dummy)
                exported = lowered.export()
            except AttributeError:
                exported = None

    if exported is not None:
        serialized = exported.serialize()
        with open(out_path, 'wb') as f:
            f.write(serialized)
        nr = getattr(exported, 'nr_devices', '?')
        print(f'StableHLO bytecode saved → {out_path}  (nr_devices={nr})')
    else:
        lowered = jit_fn.lower(dummy)
        try:
            text = lowered.as_text()
        except Exception:
            text = str(lowered.compiler_ir(dialect='stablehlo'))
        suffix = '_fp16' if fp16 else ''
        out_path = os.path.join(output_dir, f'model_jax{suffix}.mlir')
        with open(out_path, 'w') as f:
            f.write(text)
        print(f'StableHLO MLIR text saved → {out_path}')

    meta_path = os.path.join(output_dir, 'model_jax_meta.json')
    save_metadata(model, meta_path)
    return out_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_jax(model):
    try:
        import jax
        import jax.numpy as jnp
        from jax.experimental import jax2tf
        import tensorflow as tf
    except ImportError as e:
        print(f'WARNING: {e} — skipping JAX verification.')
        return

    shape = model.input_shape
    H, W, C = int(shape[1]), int(shape[2]), int(shape[3])
    dummy = jnp.array(np.random.rand(1, H, W, C).astype(np.float32))

    print(f'Verifying via jax2tf.call_tf (input shape: {dummy.shape}) ...')
    tf_fn = tf.function(lambda x: model(x, training=False))
    jax_fn = jax2tf.call_tf(tf_fn)
    result = jax.jit(jax_fn)(dummy)

    if isinstance(result, (list, tuple)):
        shapes = [tuple(r.shape) for r in result]
    else:
        shapes = [tuple(result.shape)]
    print(f'Verification passed — output shapes: {shapes}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = build_arg_parser().parse_args()

    from ymapnet.core.NNModel import load_keypoints_model
    model, input_size, output_size, num_heatmaps = load_keypoints_model(args.model)

    os.makedirs(args.output, exist_ok=True)

    if args.format == 'npz':
        convert_npz(model, args.output, fp16=args.fp16)
    elif args.format == 'orbax':
        convert_orbax(model, args.output)
    elif args.format == 'stablehlo':
        convert_stablehlo(model, args.output, fp16=args.fp16)

    if args.verify:
        verify_jax(model)

    print('\nDone.')


if __name__ == '__main__':
    main()
