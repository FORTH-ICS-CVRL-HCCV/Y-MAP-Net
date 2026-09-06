"""Checkpoint averaging (SWA-style) for Y-MAP-Net — U1 / CA2–CA4.

Post-training, average the top-N best-by-monitor checkpoints retained by
ConditionalModelCheckpoint (CA1), recalibrate the BatchNormalization running
statistics (CA3, mandatory — this model is BN-heavy), then adopt the averaged
weights ONLY if they beat the best single epoch on the monitored metric (CA4).

The iterates come from a single run, so they are mode-connected and the
"model soup" mixing risk does not apply. On return the passed `model` holds the
adopted weights (averaged-or-best); the caller packages exactly that net (CA5).

Everything here is a no-op unless cfg['useCheckpointAveraging'] is True, and the
whole call is guarded so a failure can never regress the shipped best-epoch model.
"""

import numpy as np
import tensorflow as tf


def _bn_layers(model):
    """All BatchNormalization layers, including any nested sub-models. Functional
    models flatten their graph into .layers, but recurse anyway to be safe."""
    found = []

    def walk(layer):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            found.append(layer)
        for sub in getattr(layer, "layers", []):
            walk(sub)

    for l in model.layers:
        walk(l)
    return found


def _iter_input_batches(ds, n_batches):
    """Yield up to n_batches input tensors from either a tf.data.Dataset or a
    Keras Sequence / indexable generator, dropping the label half of each (x, y)."""
    if isinstance(ds, tf.data.Dataset):
        for i, batch in enumerate(ds):
            if i >= n_batches:
                break
            yield batch[0] if isinstance(batch, (tuple, list)) else batch
        return
    try:
        length = len(ds)
    except Exception:
        length = n_batches
    for i in range(min(n_batches, length)):
        batch = ds[i]
        yield batch[0] if isinstance(batch, (tuple, list)) else batch


def _average_checkpoints(model, paths):
    """CA2: uniform element-wise average of the weight files in `paths`, loaded one
    at a time (accumulate in float64, never hold all N in RAM). Sets the result on
    `model`."""
    model.load_weights(paths[0])
    acc = [w.astype(np.float64) for w in model.get_weights()]
    for p in paths[1:]:
        model.load_weights(p)
        for i, w in enumerate(model.get_weights()):
            acc[i] += w
    n = len(paths)
    averaged = [(a / n).astype(np.float32) for a in acc]
    model.set_weights(averaged)


def _recalibrate_batchnorm(model, train_ds, n_batches, recalib_momentum=0.9):
    """CA3: averaged conv/dense weights invalidate every BN layer's stored
    moving_mean/moving_variance. Reset them and re-accumulate over training-mode
    forward passes across the training set. Skipping this is the #1 failure mode."""
    bns = _bn_layers(model)
    saved_momentum = {}
    for bn in bns:
        saved_momentum[bn.name] = bn.momentum
        bn.momentum = recalib_momentum
        bn.moving_mean.assign(tf.zeros_like(bn.moving_mean))
        bn.moving_variance.assign(tf.ones_like(bn.moving_variance))
    ran = 0
    for x in _iter_input_batches(train_ds, n_batches):
        model(x, training=True)  # forward pass updates BN moving stats only
        ran += 1
    for bn in bns:
        bn.momentum = saved_momentum[bn.name]
    return ran, len(bns)


def _is_better(a, b, mode):
    return a < b if mode == 'min' else a > b


def apply_checkpoint_averaging(model, checkpointer, cfg, train_ds, val_ds):
    """CA2–CA4 orchestration. Returns a summary dict (or None if skipped/failed).
    On return `model` holds the adopted weights.

    Preconditions handled by the caller (CA5): call this while the real composite
    losses are still compiled (before the portability recompile) and while
    train_ds / val_ds are still alive.
    """
    if not cfg.get('useCheckpointAveraging', False):
        return None

    monitor = cfg['earlyStoppingMonitor']
    mode = 'min' if cfg.get('earlyStoppingMode', 'auto') != 'max' else 'max'
    # ConditionalModelCheckpoint stores mode explicitly; trust it if present.
    mode = getattr(checkpointer, 'mode', mode)
    accept_only_if_better = cfg.get('swaAcceptOnlyIfBetter', True)
    n_recalib = int(cfg.get('swaBNRecalibBatches', 200))
    best_weights_path = getattr(checkpointer, 'filepath', 'best.weights.h5')

    paths = checkpointer.get_swa_checkpoint_paths()
    if len(paths) < 2:
        print("[SWA] Only %d checkpoint(s) retained — nothing to average, keeping best epoch." % len(paths))
        return None

    best_single = checkpointer.best
    print("[SWA] Averaging %d checkpoints; best single %s = %.6f" % (len(paths), monitor, best_single))

    # CA2 — uniform average.
    _average_checkpoints(model, paths)

    # CA3 — mandatory BN recalibration.
    ran, n_bn = _recalibrate_batchnorm(model, train_ds, n_recalib)
    print("[SWA] Recalibrated %d BatchNorm layers over %d training-mode batches." % (n_bn, ran))

    # CA4 — validated accept/reject guard.
    results = model.evaluate(val_ds, return_dict=True, verbose=0)
    eval_key = monitor[4:] if monitor.startswith('val_') else monitor
    swa_score = results.get(eval_key)
    if swa_score is None:
        print("[SWA] Monitor '%s' (eval key '%s') absent from evaluation; reverting to best epoch." %
              (monitor, eval_key))
        model.load_weights(best_weights_path)
        return {"adopted": False, "reason": "monitor_missing", "num_averaged": len(paths)}

    better = _is_better(swa_score, best_single, mode)
    adopt = better or (not accept_only_if_better)
    print("[SWA] averaged %s = %.6f vs best-single %.6f -> %s" %
          (monitor, swa_score, best_single, "ADOPT averaged" if adopt else "REJECT, keep best epoch"))

    if not adopt:
        model.load_weights(best_weights_path)

    return {
        "adopted": bool(adopt),
        "num_averaged": len(paths),
        "swa_score": float(swa_score),
        "best_single": float(best_single),
        "monitor": monitor,
        "eval": {k: float(v) for k, v in results.items()},
    }
