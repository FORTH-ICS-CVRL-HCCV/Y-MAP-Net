"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""
import os
import sys

import tensorflow as tf
import keras
from keras.losses import Loss
from keras.metrics import Metric

import keras.backend as K
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer
"""
Transformers without Normalization
Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, Zhuang Liu

    Normalization layers are ubiquitous in modern neural networks and have long been considered essential. This work demonstrates that Transformers without normalization can achieve the same or better performance using a remarkably simple technique. We introduce Dynamic Tanh (DyT), an element-wise operation DyT(x)=tanh(αx), as a drop-in replacement for normalization layers in Transformers. DyT is inspired by the observation that layer normalization in Transformers often produces tanh-like, S-shaped input-output mappings. By incorporating DyT, Transformers without normalization can match or exceed the performance of their normalized counterparts, mostly without hyperparameter tuning. We validate the effectiveness of Transformers with DyT across diverse settings, ranging from recognition to generation, supervised to self-supervised learning, and computer vision to language models. These findings challenge the conventional understanding that normalization layers are indispensable in modern neural networks, and offer new insights into their role in deep networks. 

https://arxiv.org/abs/2503.10622v1
"""


class DyT(tf.keras.layers.Layer):

    def __init__(self, channels, init_alpha=1.0, **kwargs):
        super(DyT, self).__init__(**kwargs)
        self.alpha = self.add_weight(name='alpha', shape=(1, ), initializer=tf.keras.initializers.Constant(init_alpha),
                                     trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(channels, ), initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(channels, ), initializer='zeros', trainable=True)

    def call(self, inputs):
        x = tf.tanh(self.alpha * inputs)
        return self.gamma * x + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha.numpy(), 'gamma': self.gamma.numpy(), 'beta': self.beta.numpy()})
        return config


"""
Cautious Optimizers: Improving Training with One Line of Code
Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu

    AdamW has been the default optimizer for transformer pretraining. For many years, our community searched for faster and more stable optimizers with only constrained positive outcomes. In this work, we propose a single-line modification in Pytorch to any momentum-based optimizer, which we rename cautious optimizer, e.g. C-AdamW and C-Lion. Our theoretical result shows that this modification preserves Adam's Hamiltonian function and it does not break the convergence guarantee under the Lyapunov analysis. In addition, a whole new family of optimizers is revealed by our theoretical insight. Among them, we pick the simplest one for empirical experiments, showing not only speed-up on Llama and MAE pretraining up to 1.47 times, but also better results in LLM post-training tasks. 

https://arxiv.org/abs/2411.16085
"""


@keras_export(["keras.optimizers.AdamWCautious"])
class AdamWCautious(optimizer.Optimizer):
    """Optimizer that implements the AdamW algorithm with cautious behavior.

    This optimizer is based on the AdamW algorithm but includes additional
    cautious updates as per the "Cautious Optimizers" paper (https://arxiv.org/abs/2411.16085).

    Args:
        learning_rate: A float, a `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to use. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable that takes no arguments and returns
            the actual value to use. The exponential decay rate for the 1st moment estimates. Defaults to `0.9`.
        beta_2: A float value or a constant float tensor, or a callable that takes no arguments and returns
            the actual value to use. The exponential decay rate for the 2nd moment estimates. Defaults to `0.999`.
        epsilon: A small constant for numerical stability. Defaults to `1e-7`.
        amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence
            of Adam and beyond". Defaults to `False`.
        weight_decay: Weight decay coefficient. Defaults to `None`.
        caution: Boolean. Whether to apply the cautious behavior to the optimizer. Defaults to `False`.
        {{base_optimizer_keyword_args}}
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        caution=True,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="adamw_cautious",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.caution = caution

    def build(self, var_list):
        """Initialize optimizer variables (momentums, velocities, and optionally velocity_hats)."""
        if self.built:
            return
        super().build(var_list)
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(self.add_variable_from_reference(reference_variable=var, name="momentum"))
            self._velocities.append(self.add_variable_from_reference(reference_variable=var, name="velocity"))
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(self.add_variable_from_reference(reference_variable=var,
                                                                            name="velocity_hat"))

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1_power = ops.power(ops.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = ops.power(ops.cast(self.beta_2, variable.dtype), local_step)

        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]

        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        self.assign_add(m, ops.multiply(ops.subtract(gradient, m), 1 - self.beta_1))
        self.assign_add(v, ops.multiply(ops.subtract(ops.square(gradient), v), 1 - self.beta_2))

        if self.amsgrad:
            v_hat = self._velocity_hats[self._get_variable_index(variable)]
            self.assign(v_hat, ops.maximum(v_hat, v))
            v = v_hat

        if self.caution:
            mask = ops.cast(ops.greater(m * gradient, 0), gradient.dtype)
            mask_mean = ops.mean(mask) + 1e-10
            mask_mean = ops.maximum(mask_mean, 1e-3)
            mask = mask / mask_mean
            m = m * mask

        self.assign_sub(
            variable,
            ops.divide(ops.multiply(m, alpha), ops.add(ops.sqrt(v), self.epsilon)),
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "amsgrad": self.amsgrad,
            "caution": self.caution,
        })
        return config


AdamWCautious.__doc__ = AdamWCautious.__doc__.replace("{{base_optimizer_keyword_args}}",
                                                      optimizer.base_optimizer_keyword_args)


# Define the ConditionalModelCheckpoint class
#-------------------------------------------------------------------------------
class ConditionalModelCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, monitor, mode, filepath, save_best_only, save_weights_only, start_from_epoch, verbose=1,
                 total_epochs=None, swa_num_checkpoints=0, swa_min_epoch_fraction=0.5,
                 swa_filepath="swa.slot{slot}.weights.h5", serial=None, status_path="status.txt"):
        super().__init__()
        #----------------------------------------
        self.OKGREEN = '\033[92m'
        self.WARNING = '\033[93m'
        self.OKBLUE = '\033[94m'
        self.ENDC = '\033[0m'
        self.monitor = monitor
        self.mode = mode
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.start_from_epoch = start_from_epoch
        self.verbose = verbose
        # Defaults to the historical CWD-relative "status.txt" so existing callers
        # (trainYMAPNet.py) are unaffected; a concurrent job should pass its own path.
        self.status_path = status_path
        #----------------------------------------
        self.best = None
        self.bestEpoch = None
        self.bestLog = None
        if self.mode == 'min':
            self.best = float('inf')
        elif self.mode == 'max':
            self.best = -float('inf')
        #----------------------------------------
        # CA1 (checkpoint-averaging / SWA): retain the N best-by-monitor accepted
        # checkpoints as a small ranked pool (swa.slot{k}.weights.h5). Disabled when
        # swa_num_checkpoints <= 0 so existing runs are unaffected.
        self.swaNumCheckpoints = swa_num_checkpoints
        self.swaMinEpochFraction = swa_min_epoch_fraction
        self.swaFilepath = swa_filepath
        self._swaPool = []  # list of dicts {score, epoch, path}
        #----------------------------------------
        # Epoch timing for ETA estimation
        if total_epochs is None or serial is None:
            try:
                import json
                with open("configuration.json", "r") as _f:
                    _cfg = json.load(_f)
                if total_epochs is None:
                    total_epochs = int(_cfg.get("epochs", 0))
                if serial is None:
                    serial = _cfg.get("serial", None)
            except Exception:
                if total_epochs is None:
                    total_epochs = 0
        self.total_epochs = total_epochs
        self.serial = serial
        self._epoch_times = []  # list of (epoch_index, end_timestamp)
        #----------------------------------------

    def reset(self):
        print(self.WARNING, "Resetting Checkpointer", self.ENDC)
        self.best = None
        self.bestEpoch = None
        self.bestLog = None
        self._swaPool = []
        if self.mode == 'min':
            self.best = float('inf')
        elif self.mode == 'max':
            self.best = -float('inf')

    #------------------------------------------------------------------------------
    # CA1: maintain the top-N best-by-monitor checkpoint pool for checkpoint averaging.
    def _swa_is_better(self, a, b):
        """True if score a is strictly better than score b under the active mode."""
        return a < b if self.mode == 'min' else a > b

    def _maybe_update_swa_pool(self, epoch, current):
        if self.swaNumCheckpoints <= 0 or current is None:
            return
        # Gate: only late (post-plateau) epochs enter the pool. If total_epochs is
        # unknown (0) the gate is disabled and every scored epoch is eligible.
        if self.total_epochs > 0 and (epoch + 1) < self.swaMinEpochFraction * self.total_epochs:
            return
        if len(self._swaPool) < self.swaNumCheckpoints:
            slot = len(self._swaPool)
            path = self.swaFilepath.format(slot=slot)
            self.model.save_weights(path)
            self._swaPool.append({"score": current, "epoch": epoch, "path": path})
        else:
            # Pool full: replace the worst entry only if the current epoch beats it.
            worst = max(self._swaPool, key=lambda e: e["score"]) if self.mode == 'min' \
                else min(self._swaPool, key=lambda e: e["score"])
            if self._swa_is_better(current, worst["score"]):
                self.model.save_weights(worst["path"])
                worst["score"] = current
                worst["epoch"] = epoch
        if self.verbose > 0:
            pool = sorted(e["epoch"] + 1 for e in self._swaPool)
            print(self.OKBLUE, "SWA pool (top-%d by %s): epochs %s" %
                  (self.swaNumCheckpoints, self.monitor, pool), self.ENDC)

    def get_swa_checkpoint_paths(self):
        """Return the retained checkpoint file paths (best-scoring first)."""
        ordered = sorted(self._swaPool, key=lambda e: e["score"], reverse=(self.mode == 'max'))
        return [e["path"] for e in ordered]

    @staticmethod
    def _collect_system_stats():
        """Return (cpu_pct, ram_used_gb, ram_total_gb, gpu_rows) where gpu_rows is a list of
        (gpu_id, vram_mib) tuples for processes owned by the current PID (including children)."""
        import subprocess, os, re
        cpu_pct = None
        ram_used_gb = None
        ram_total_gb = None
        gpu_rows = []

        try:
            import psutil
            proc = psutil.Process(os.getpid())
            cpu_pct = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            ram_used_gb = vm.used / (1024**3)
            ram_total_gb = vm.total / (1024**3)
        except Exception:
            pass

        try:
            # Collect current PID and all child PIDs to match against nvidia-smi output
            import os, psutil
            own_pids = {os.getpid()}
            try:
                own_pids.update(c.pid for c in psutil.Process(os.getpid()).children(recursive=True))
            except Exception:
                pass

            smi = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"
            ], stderr=subprocess.DEVNULL, timeout=5).decode()
            # Also get GPU index from uuid
            uuid_map = {}
            try:
                uuid_out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                                   stderr=subprocess.DEVNULL, timeout=5).decode()
                for line in uuid_out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2:
                        uuid_map[parts[1]] = int(parts[0])
            except Exception:
                pass

            gpu_total = {}  # gpu_id -> total MiB used by our processes
            for line in smi.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 3:
                    continue
                try:
                    pid = int(parts[0])
                    uuid = parts[1]
                    mib = int(parts[2])
                except ValueError:
                    continue
                if pid not in own_pids:
                    continue
                gpu_id = uuid_map.get(uuid, uuid)
                gpu_total[gpu_id] = gpu_total.get(gpu_id, 0) + mib

            gpu_rows = sorted(gpu_total.items())
        except Exception:
            pass

        return cpu_pct, ram_used_gb, ram_total_gb, gpu_rows

    def _write_status(self, epoch, logs, preamble=None):
        import datetime
        now = datetime.datetime.now()
        self._epoch_times.append((epoch, now.timestamp()))

        # Compute ETA and time-per-epoch using average seconds-per-epoch over recorded history
        eta_str = "N/A"
        epoch_time_str = "N/A"
        if self.total_epochs > 0 and len(self._epoch_times) >= 2:
            epochs_recorded = self._epoch_times[-1][0] - self._epoch_times[0][0]
            if epochs_recorded > 0:
                elapsed = self._epoch_times[-1][1] - self._epoch_times[0][1]
                secs_per_epoch = elapsed / epochs_recorded
                mins, secs = divmod(int(secs_per_epoch), 60)
                epoch_time_str = "%dm %02ds" % (mins, secs) if mins else "%ds" % secs
                remaining_epochs = self.total_epochs - (epoch + 1)
                if remaining_epochs > 0:
                    eta_ts = now + datetime.timedelta(seconds=secs_per_epoch * remaining_epochs)
                    eta_str = eta_ts.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    eta_str = "Done"

        cpu_pct, ram_used_gb, ram_total_gb, gpu_rows = self._collect_system_stats()

        lines = []
        if preamble is not None:
            lines.append(preamble)
            lines.append("")
        if self.serial is not None:
            lines.append("Serial     : %s" % self.serial)
        lines.append("Updated    : %s" % now.strftime("%Y-%m-%d %H:%M:%S"))
        lines.append("Epoch      : %d / %d" %
                     (epoch + 1, self.total_epochs) if self.total_epochs > 0 else "Epoch      : %d" % (epoch + 1))
        lines.append("Epoch time : %s" % epoch_time_str)
        lines.append("ETA        : %s" % eta_str)
        lines.append("Monitor    : %s" % self.monitor)
        lines.append("")
        if cpu_pct is not None:
            lines.append("CPU        : %.1f%%" % cpu_pct)
        if ram_used_gb is not None:
            lines.append("RAM        : %.1f / %.1f GB" % (ram_used_gb, ram_total_gb))
        for gpu_id, vram_mib in gpu_rows:
            lines.append("GPU %-2s VRAM : %d MiB" % (str(gpu_id), vram_mib))
        if cpu_pct is not None or gpu_rows:
            lines.append("")
        if self.bestEpoch is not None:
            lines.append("Best epoch : %d" % (self.bestEpoch + 1))
            lines.append("Best %-10s: %.6f" % (self.monitor, self.best))
            lines.append("")
            lines.append("Best epoch metrics:")
            for k, v in sorted(self.bestLog.items()):
                try:
                    lines.append("  %-30s %.6f" % (k, float(v)))
                except (TypeError, ValueError):
                    lines.append("  %-30s %s" % (k, v))
        else:
            lines.append("Best epoch : (none yet)")
        lines.append("")
        lines.append("Current epoch metrics:")
        for k, v in sorted(logs.items()):
            try:
                lines.append("  %-30s %.6f" % (k, float(v)))
            except (TypeError, ValueError):
                lines.append("  %-30s %s" % (k, v))
        try:
            from ymapnet.reporting.plotTrainingProgressToSVG import baseline_comparison_lines
            cmp_lines = baseline_comparison_lines(logs)
            if cmp_lines:
                lines.append("")
                lines.extend(cmp_lines)
        except Exception:
            pass
        try:
            with open(self.status_path, "w") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass

    def write_completion_status(self):
        """Write a final status.txt indicating training is complete, including best-epoch results."""
        epoch = self.bestEpoch if self.bestEpoch is not None else 0
        logs = self.bestLog if self.bestLog is not None else {}
        self._write_status(epoch, logs, preamble="*** TRAINING COMPLETE ***")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if epoch + 1 < self.start_from_epoch:
            if self.verbose > 0:
                print(self.OKBLUE, end="")
                print("Skipping checkpointing at epoch ", epoch + 1, ", starting from epoch ", self.start_from_epoch,
                      end="")
                print(self.ENDC)
            self._write_status(epoch, logs)
            return

        if current is None:
            if self.verbose > 0:
                print("Monitor value '", self.monitor, "' not found in logs; skipping checkpointing.")
            return

        if self.save_best_only:
            if (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
                if self.verbose > 0:
                    print(self.OKGREEN, end="")
                    print("\nEpoch ", epoch + 1, ": ", self.monitor, " improved from %0.4f" % self.best, " to ",
                          current, ". Saving model.     ", end="")
                    print(self.ENDC)
                self.bestEpoch = epoch
                self.best = current
                self.bestLog = logs
                self._save_model(epoch)
            else:
                if self.verbose > 0:
                    print(self.WARNING, end="")
                    print("\nEpoch ", epoch + 1, ": ", self.monitor, " is %0.4f, it did not improve from %0.4f" %
                          (current, self.best), " (Best is Epoch ", self.bestEpoch, ").     ", end="")
                    print(self.ENDC)
        else:
            if self.verbose > 0:
                print("Epoch ", epoch + 1, ": Saving model.       ")
            self.bestEpoch = epoch
            self.bestLog = logs
            self._save_model(epoch)
        # CA1: update the checkpoint-averaging pool for every scored epoch, regardless
        # of whether this epoch is the single global best.
        self._maybe_update_swa_pool(epoch, current)
        self._write_status(epoch, logs)

    def _save_model(self, epoch):
        if self.save_weights_only:
            self.model.save_weights(self.filepath.format(epoch=epoch + 1))
        else:
            self.model.save(self.filepath.format(epoch=epoch + 1))

    def load_best_model(self):
        epoch = self.bestEpoch
        print(self.OKGREEN, "\n Loading Best Epoch (", epoch, ") weights  \n", self.ENDC)
        if self.save_weights_only:
            self.model.load_weights(self.filepath.format(epoch=epoch + 1), skip_mismatch=False)
        else:
            print("Load model only works when saving weights only")


#-------------------------------------------------------------------------------
#AbsRel : https://arxiv.org/pdf/2401.10891
def absrel(predicted_depth, ground_truth_depth):
    """
    Calculate the Absolute Relative Error (AbsRel) between predicted and ground truth depth maps.

    Args:
    predicted_depth (np.ndarray): 2D array of predicted depth values.
    ground_truth_depth (np.ndarray): 2D array of ground truth depth values.

    Returns:
    float: The calculated AbsRel value.
    """
    import numpy as np

    # Ensure both arrays are numpy arrays
    predicted_depth = np.array(predicted_depth)
    ground_truth_depth = np.array(ground_truth_depth)

    # Validate the shape of the input arrays
    if predicted_depth.shape != ground_truth_depth.shape:
        raise ValueError("Input arrays must have the same shape ", predicted_depth.shape, " , ",
                         ground_truth_depth.shape)

    # Avoid division by zero by masking zero values in ground truth
    mask = ground_truth_depth != 0
    abs_rel_error = np.abs(predicted_depth[mask] - ground_truth_depth[mask]) / ground_truth_depth[mask]

    # Return the mean AbsRel over all valid elements
    return np.mean(abs_rel_error)


#-------------------------------------------------------------------------------
def RMSE(predicted_depth, ground_truth_depth):
    """
    Calculate the Root Mean Square Error (RMSE) between predicted and ground truth depth maps.

    Args:
    predicted_depth (np.ndarray): 2D array of predicted depth values.
    ground_truth_depth (np.ndarray): 2D array of ground truth depth values.

    Returns:
    float: The calculated RMSE value.
    """
    import numpy as np

    # Ensure both arrays are numpy arrays
    predicted_depth = np.array(predicted_depth)
    ground_truth_depth = np.array(ground_truth_depth)

    # Validate the shape of the input arrays
    if predicted_depth.shape != ground_truth_depth.shape:
        raise ValueError("Input arrays must have the same shape")

    # Calculate the squared differences
    squared_diff = (predicted_depth - ground_truth_depth)**2

    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_diff)

    return rmse


#-------------------------------------------------------------------------------
class CosineSimilarityMetric(tf.keras.metrics.Metric):

    def __init__(self, axis=1, name='cosine_similarity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.axis = axis
        self.sum_similarity = self.add_weight(name='sum_similarity', initializer='zeros')
        self.total_weight = self.add_weight(name='total_weight', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Cast inputs to float32 for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Compute dot product along specified axis
        dot_product = tf.reduce_sum(y_true * y_pred, axis=self.axis)

        # Compute L2 norms
        norm_true = tf.norm(y_true, axis=self.axis)
        norm_pred = tf.norm(y_pred, axis=self.axis)

        # Calculate cosine similarity with epsilon for numerical stability
        epsilon = tf.keras.backend.epsilon()
        cosine_sim = dot_product / (norm_true * norm_pred + epsilon)

        # Handle sample weights
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            cosine_sim *= sample_weight
            batch_weight = tf.reduce_sum(sample_weight)
        else:
            # Count all elements in the similarity tensor
            batch_weight = tf.cast(tf.size(cosine_sim), tf.float32)

        # Accumulate results
        batch_similarity = tf.reduce_sum(cosine_sim)
        self.sum_similarity.assign_add(batch_similarity)
        self.total_weight.assign_add(batch_weight)

    def result(self):
        return self.sum_similarity / self.total_weight

    def reset_state(self):
        self.sum_similarity.assign(0.0)
        self.total_weight.assign(0.0)


#-------------------------------------------------------------------------------
class TopKAccuracyMetric(tf.keras.metrics.Metric):

    def __init__(self, k=5, name='top_k_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.count = self.add_weight(name='count', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Cast predictions to float32 for compatibility
        y_pred = tf.cast(y_pred, tf.float32)

        # Convert one-hot encoded y_true to class indices
        y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)

        # Find top-k predicted classes
        top_k_preds = tf.math.top_k(y_pred, k=self.k, sorted=False).indices
        y_true_broadcasted = tf.broadcast_to(y_true[:, tf.newaxis], tf.shape(top_k_preds))

        # Check if true class exists in top-k predictions
        correct = tf.reduce_any(tf.equal(y_true_broadcasted, top_k_preds), axis=1)
        correct = tf.cast(correct, tf.float32)

        # Handle sample weighting
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            correct *= sample_weight
            sample_weight = tf.reduce_sum(sample_weight)
        else:
            sample_weight = tf.cast(tf.shape(y_true)[0], tf.float32)

        # Update state variables
        self.count.assign_add(tf.reduce_sum(correct))
        self.total.assign_add(sample_weight)

    def result(self):
        return tf.math.divide_no_nan(self.count, self.total)

    def reset_state(self):
        self.count.assign(0.0)
        self.total.assign(0.0)


#-------------------------------------------------------------------------------
class MultiHotF1Metric(tf.keras.metrics.Metric):
    """Per-sample micro-F1 for multi-label (multi-hot) binary outputs.

    Unlike BinaryAccuracy (which reaches 99.9% by predicting all-zeros because
    true-negatives dominate 17977 classes) this metric is honest:

        F1 = 2*TP / (2*TP + FP + FN)   computed per sample, then averaged.

    Two thresholds are tracked simultaneously so you can choose which gives
    better separation:
        threshold      – hard decision boundary (default 0.5).
        top_k          – if provided, treat the top-k scoring classes as
                         positive predictions instead of using threshold.
                         This decouples F1 from sigmoid calibration and is
                         useful early in training when the head is still
                         learning its output scale.

    Pass top_k=None to use pure threshold mode (default).
    Pass threshold=None to use pure top-k mode.
    """

    def __init__(self, threshold=0.5, top_k=None, name='multihot_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        # Exactly one of threshold / top_k should be active.
        if threshold is None and top_k is None:
            raise ValueError('At least one of threshold or top_k must be set.')
        self.threshold = threshold
        self.top_k = top_k
        # Accumulators: sum of per-sample F1, number of samples.
        self.f1_sum = self.add_weight(name='f1_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        if self.top_k is not None:
            # Build a binary mask with 1 at the top-k predicted positions.
            # tf.math.top_k returns sorted indices; scatter back to (B, C).
            _, top_indices = tf.math.top_k(y_pred, k=self.top_k, sorted=False)
            batch_size = tf.shape(y_pred)[0]
            num_classes = tf.shape(y_pred)[1]
            # Build (B*K, 2) gather indices then scatter_nd into (B, C).
            batch_idx = tf.repeat(tf.range(batch_size), self.top_k)
            flat_idx = tf.stack([batch_idx, tf.reshape(top_indices, [-1])], axis=1)
            pred_bin = tf.cast(tf.scatter_nd(flat_idx, tf.ones(batch_size * self.top_k), [batch_size, num_classes]),
                               tf.float32)
        else:
            pred_bin = tf.cast(y_pred >= self.threshold, tf.float32)

        # Per-sample TP, FP, FN.
        tp = tf.reduce_sum(y_true * pred_bin, axis=-1)  # (B,)
        fp = tf.reduce_sum((1 - y_true) * pred_bin, axis=-1)
        fn = tf.reduce_sum(y_true * (1 - pred_bin), axis=-1)

        # F1 per sample; define 0/0 = 0.
        denom = 2.0 * tp + fp + fn
        f1 = tf.math.divide_no_nan(2.0 * tp, denom)  # (B,)

        if sample_weight is not None:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
            f1 = f1 * sample_weight
            n = tf.reduce_sum(sample_weight)
        else:
            n = tf.cast(tf.shape(y_true)[0], tf.float32)

        self.f1_sum.assign_add(tf.reduce_sum(f1))
        self.count.assign_add(n)

    def result(self):
        return tf.math.divide_no_nan(self.f1_sum, self.count)

    def reset_state(self):
        self.f1_sum.assign(0.0)
        self.count.assign(0.0)


#-------------------------------------------------------------------------------
#https://github.com/tensorflow/addons/blob/v0.20.0/tensorflow_addons/metrics/r_square.py
class RSquaredMetric(Metric):

    def __init__(self, name='r_squared', **kwargs):
        super(RSquaredMetric, self).__init__(name=name, **kwargs)
        self.ssr = self.add_weight(name='ssr', initializer='zeros')
        self.sst = self.add_weight(name='sst', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure both y_true and y_pred are cast to float32
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision
        y_true = tf.cast(y_true, float_type)
        y_pred = tf.cast(y_pred, float_type)

        # Calculate SSR (Sum of Squares of Residuals)
        ssr_update = tf.reduce_sum(tf.square(y_true - y_pred))
        self.ssr.assign_add(ssr_update)

        # Calculate SST (Total Sum of Squares)
        mean_y_true = tf.reduce_mean(y_true)
        sst_update = tf.reduce_sum(tf.square(y_true - mean_y_true))
        self.sst.assign_add(sst_update)

    def result(self):
        return 1 - (self.ssr / self.sst) if self.sst > 0 else 0.0

    def reset_state(self):
        self.ssr.assign(0.0)
        self.sst.assign(0.0)


#-------------------------------------------------------------------------------
#0.0 indicates no correct pixels and 1.0 indicates all pixels are correct.
class HeatmapDistanceMetric(Metric):

    def __init__(self, name='hdm', threshold=24, scale=1.0, **kwargs):
        super(HeatmapDistanceMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.scale = scale
        self.total_correct_pixels = self.add_weight(name='total_correct_pixels', initializer='zeros')
        self.total_pixels = self.add_weight(name='total_pixels', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision
        y_true = tf.cast(y_true, float_type) * self.scale
        y_pred = tf.cast(y_pred, float_type) * self.scale

        # Apply threshold to determine if the absolute difference is within the threshold
        y_pred_binarized = tf.cast(tf.less_equal(tf.abs(y_true - y_pred), self.threshold), float_type)

        # Count matching pixels
        correct_pixels = tf.reduce_sum(tf.cast(tf.equal(y_pred_binarized, 1), float_type))

        # Update total correct pixels
        self.total_correct_pixels.assign_add(correct_pixels)

        # Update total pixels processed in this batch
        batch_total_pixels = tf.cast(tf.size(y_true), float_type)
        self.total_pixels.assign_add(batch_total_pixels)

    def result(self):
        return self.total_correct_pixels / (self.total_pixels + 1)

    def reset_state(self):
        self.total_correct_pixels.assign(0.0)
        self.total_pixels.assign(0.0)


#-------------------------------------------------------------------------------
class CustomTopKCategoricalAccuracy(tf.keras.metrics.Metric):

    def __init__(self, k=5, name="top_k_categorical_accuracy", **kwargs):
        super(CustomTopKCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.k = k
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure predictions are cast to float32 to avoid issues with mixed precision (float16)
        y_pred = tf.cast(y_pred, tf.float32)

        # Top K predictions
        top_k_preds = tf.math.top_k(y_pred, k=self.k).indices

        # Check if true labels are within the top k predictions
        matches = tf.reduce_any(tf.equal(top_k_preds, tf.expand_dims(tf.cast(y_true, tf.int32), axis=-1)), axis=-1)

        # Update total and count
        matches = tf.cast(matches, tf.float32)
        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        # Return the accuracy
        return self.total / self.count

    def reset_state(self):
        # Reset states for each epoch
        self.total.assign(0.0)
        self.count.assign(0.0)


#-------------------------------------------------------------------------------
class HeatmapDistanceMetricPartial(Metric):

    def __init__(self, name='hdm', threshold=24, start=0, end=None, **kwargs):
        super(HeatmapDistanceMetricPartial, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.start = start
        self.end = end
        self.total_correct_pixels = self.add_weight(name='total_correct_pixels', initializer='zeros')
        self.total_pixels = self.add_weight(name='total_pixels', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        float_type = tf.float32  # Always accumulate metrics in float32 for numerical stability under mixed precision (float16 gave spurious hm_hdm=1.0 on the geo channel)

        # Apply range selection based on start and end indices
        y_true_selected = tf.cast(y_true[..., self.start:self.end], float_type)
        y_pred_selected = tf.cast(y_pred[..., self.start:self.end], float_type)

        # Apply threshold to determine if the absolute difference is within the threshold
        y_pred_binarized = tf.cast(tf.less_equal(tf.abs(y_true_selected - y_pred_selected), self.threshold), float_type)

        # Count matching pixels
        correct_pixels = tf.reduce_sum(tf.cast(tf.equal(y_pred_binarized, 1), float_type))

        # Update total correct pixels
        self.total_correct_pixels.assign_add(correct_pixels)

        # Update total pixels processed in this batch
        batch_total_pixels = tf.cast(tf.size(y_true_selected), float_type)
        self.total_pixels.assign_add(batch_total_pixels)

    def result(self):
        return self.total_correct_pixels / (self.total_pixels + 1)

    def reset_state(self):
        self.total_correct_pixels.assign(0.0)
        self.total_pixels.assign(0.0)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class NonZeroCorrectPixelMetric(Metric):

    def __init__(self, name='hdmnot0', accuracyThreshold=24, nonzeroThreshold=-110.0, start=0, end=None, **kwargs):
        super(NonZeroCorrectPixelMetric, self).__init__(name=name, **kwargs)
        self.accuracyThreshold = accuracyThreshold
        self.start = start
        self.end = end
        self.nonzeroThreshold = nonzeroThreshold
        self.total_correct = self.add_weight(name='total_correct', initializer='zeros')
        self.total_nonzero = self.add_weight(name='total_nonzero', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        float_type = tf.float32  # Always accumulate metrics in float32 for numerical stability under mixed precision (float16 gave spurious hm_hdm=1.0 on the geo channel)

        # Select channels if needed
        y_true_selected = tf.cast(y_true[..., self.start:self.end], float_type)
        y_pred_selected = tf.cast(y_pred[..., self.start:self.end], float_type)

        # Mask of non-zero ground truth pixels
        #nonzero_mask = tf.not_equal(y_true_selected, 0.0)
        nonzero_mask = tf.greater_equal(y_true_selected,
                                        self.nonzeroThreshold)  #Values are [-120.0 .. 120.0] so use -120 as zero

        # Calculate absolute error and mask with nonzero
        abs_error = tf.abs(y_true_selected - y_pred_selected)
        correct_mask = tf.logical_and(nonzero_mask, abs_error <= self.accuracyThreshold)

        # Count correct predictions among non-zero pixels
        correct_count = tf.reduce_sum(tf.cast(correct_mask, float_type))
        nonzero_count = tf.reduce_sum(tf.cast(nonzero_mask, float_type))

        self.total_correct.assign_add(correct_count)
        self.total_nonzero.assign_add(nonzero_count)

    def result(self):
        return self.total_correct / (self.total_nonzero + 1e-8)  # avoid division by zero

    def reset_state(self):
        self.total_correct.assign(0.0)
        self.total_nonzero.assign(0.0)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class DepthNormalsUncertaintyCalibration(Metric):
    """Calibration diagnostic for the depth+normals aleatoric uncertainty (hm_nll) head.

    Pairs with DepthNormalsNLLLoss. The hm_nll output is [B, H, W, 2*C]:
      - first C channels : mean prediction  mu      (tanh space, [-1, 1])
      - last  C channels : log-variance     log_var (unconstrained)
    y_true is the raw-scale depth+normals GT [B, H, W, C]; we divide by `scale`
    to reach the same tanh space the mean lives in (mirrors the loss).

    Reports the mean normalised squared residual  E[ (gt - mu)^2 / sigma^2 ]
    with sigma^2 = exp(log_var). For a correctly-calibrated Gaussian this equals
    1.0. > 1 => the model is OVER-confident (predicted variance too small for the
    error it actually makes); < 1 => UNDER-confident (variance inflated). This is
    the single number that says whether the uncertainty channel is meaningful
    rather than a constant the NLL drove to a trivial value.
    """

    def __init__(self, name='unc_calib', scale=120.0, **kwargs):
        super(DepthNormalsUncertaintyCalibration, self).__init__(name=name, **kwargs)
        self.scale = float(scale)
        self.total_norm_sq = self.add_weight(name='total_norm_sq', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred  = tf.cast(y_pred, tf.float32)
        gt      = tf.cast(y_true, tf.float32) / self.scale  # → tanh space, matches the mean
        n       = tf.shape(y_pred)[-1] // 2
        mu      = y_pred[..., :n]
        log_var = tf.clip_by_value(y_pred[..., n:], -10.0, 10.0)  # same clamp as the loss

        norm_sq = tf.exp(-log_var) * tf.square(gt - mu)           # (gt - mu)^2 / sigma^2
        self.total_norm_sq.assign_add(tf.reduce_sum(norm_sq))
        self.total_count.assign_add(tf.cast(tf.size(norm_sq), tf.float32))

    def result(self):
        return self.total_norm_sq / (self.total_count + 1e-8)

    def reset_state(self):
        self.total_norm_sq.assign(0.0)
        self.total_count.assign(0.0)


#-------------------------------------------------------------------------------
#https://github.com/NRauschmayr/SSIM_Loss
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class SaveHeatmapsCallback(tf.keras.callbacks.Callback):

    def __init__(self, output_dir, num_classes=182):
        super(SaveHeatmapsCallback, self).__init__()
        self.output_dir = output_dir
        self.num_classes = num_classes
        os.makedirs(output_dir, exist_ok=True)

    def on_batch_end(self, batch, logs=None):
        # Assuming model has access to y_true and y_pred tensors as part of the batch data
        y_true = logs['y_true']  # Pass y_true from model or batch data logs
        y_pred = logs['y_pred']  # Pass y_pred from model prediction logs

        y_true_combined = y_true[..., 35]
        y_pred_combined = y_pred[..., 35]

        y_true_one_hot = tf.one_hot(tf.cast(y_true_combined + 120, tf.int32), depth=self.num_classes)
        y_pred_one_hot = tf.one_hot(tf.cast(y_pred_combined + 120, tf.int32), depth=self.num_classes)

        # Save the heatmaps for the batch
        self.save_heatmap(y_true_combined, f'y_true_heatmap_batch{batch}.png')
        self.save_heatmap(y_pred_combined, f'y_pred_heatmap_batch{batch}.png')

        # Save one-hot encoded channels for the batch
        self.save_one_hot_as_png(y_true_one_hot, f'y_true_one_hot_batch{batch}')
        self.save_one_hot_as_png(y_pred_one_hot, f'y_pred_one_hot_batch{batch}')

    def save_heatmap(self, heatmap, filename):
        # Normalize heatmap to [0, 255] and cast to uint8
        heatmap = tf.cast(255 * (heatmap - tf.reduce_min(heatmap)) / (tf.reduce_max(heatmap) - tf.reduce_min(heatmap)),
                          tf.uint8)
        heatmap = tf.expand_dims(heatmap, axis=-1)  # Add channel dimension

        # Encode as PNG and save to disk
        image_png = tf.image.encode_png(heatmap)
        tf.io.write_file(os.path.join(self.output_dir, filename), image_png)

    def save_one_hot_as_png(self, one_hot_array, base_filename):
        # Loop through each channel (class) in the one-hot array
        for channel in range(one_hot_array.shape[-1]):
            # Extract the specific channel
            heatmap_channel = one_hot_array[..., channel]

            # Normalize heatmap to [0, 255] and cast to uint8
            heatmap_channel = tf.cast(
                255 * (heatmap_channel - tf.reduce_min(heatmap_channel)) /
                (tf.reduce_max(heatmap_channel) - tf.reduce_min(heatmap_channel)), tf.uint8)
            heatmap_channel = tf.expand_dims(heatmap_channel, axis=-1)  # Add channel dimension

            # Construct the filename, e.g., "base_filename_heatmap0.png"
            filename = f"{base_filename}_heatmap{channel}.png"

            # Encode as PNG and save to disk
            image_png = tf.image.encode_png(heatmap_channel)
            tf.io.write_file(os.path.join(self.output_dir, filename), image_png)


#-------------------------------------------------------------------------------
#Use if DataLoader C is configured with VALID_SEGMENTATIONS > 1
class VanillaMSELossSimple(keras.losses.Loss):

    def __init__(self, weight=1.0, scale=1.0, **kwargs):
        super(VanillaMSELossSimple, self).__init__(**kwargs)
        self.weight = weight
        self.scale = scale

    def call(self, y_true, y_pred):
        # Ensure both y_true and y_pred are cast to float32
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision
        y_true = tf.cast(y_true, float_type)
        y_pred = tf.cast(y_pred, float_type)

        # Compute the squared difference
        squared_difference = tf.square((y_true - y_pred) * self.scale)

        # Compute the mean over all elements
        mse_loss = tf.reduce_mean(squared_difference)

        return mse_loss * self.weight


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Use if DataLoader C is configured with VALID_SEGMENTATIONS = 1
class VanillaMSELossFast(keras.losses.Loss):

    def __init__(self, weight=1.0, scale=1.0, num_instances=64, num_classes=182, **kwargs):
        super(VanillaMSELossFast, self).__init__(**kwargs)
        self.weight = weight
        self.scale = scale
        self.num_instances = num_instances  # Number of segmentation categories, we try to make the NN life easier by reducing them
        self.num_classes = num_classes  # Number of segmentation categories, now 182 including background
        self.scaling_factor = 120.0  # Scaling factor for combined heatmaps
        self.segmentation_gain = 2.0
        self.instance_gain = 2.0

    def call(self, y_true, y_pred):
        # Ensure both y_true and y_pred are cast to float32
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision

        #Regular loss
        # Calculate MSE for heatmaps 0-33 (except segmentation masks)
        #----------------------------------------------------------------------------
        y_true_first = tf.cast(y_true[..., 0:33], float_type)
        y_pred_first = tf.cast(y_pred[..., 0:33], float_type)

        mse_loss = tf.reduce_mean(tf.square((y_true_first - y_pred_first) * self.scale))
        #----------------------------------------------------------------------------

        #Segmentation
        #----------------------------------------------------------------------------
        # Extract the combined heatmap (heatmaps 35-52 combined into one)
        y_true_combined = tf.cast(y_true[..., 34], tf.int32)
        y_pred_combined = tf.cast(y_pred[..., 34], tf.int32)

        # Shift the values to make them suitable for one-hot encoding
        y_true_indices = y_true_combined + 120
        y_pred_indices = y_pred_combined + 120

        # One-hot encode the combined heatmaps for all classes (including background)
        y_true_one_hot = tf.one_hot(y_true_indices, depth=self.num_classes, dtype=float_type)
        y_pred_one_hot = tf.one_hot(y_pred_indices, depth=self.num_classes, dtype=float_type)

        # Compute MSE for each class in a vectorized manner
        class_mse_loss = tf.reduce_mean(
            tf.square((y_true_one_hot - y_pred_one_hot) * (self.scaling_factor * self.scale)), axis=[0, 1, 2])

        # Average the loss across all classes
        segmentation_loss = tf.reduce_mean(class_mse_loss)
        #----------------------------------------------------------------------------

        # Sum the losses
        total_loss = mse_loss + (segmentation_loss * self.segmentation_gain)

        return total_loss * self.weight


#----------------------------------------------------------------------------
#Use if DataLoader C is configured with VALID_SEGMENTATIONS = 1
class HeatmapCoreLoss(keras.losses.Loss):
    # "Label / Text" is the 8th segmentation class (index 7 within the seg group).
    _TEXT_SEG_OFFSET = 7

    def __init__(self, channel_ranges, scale=1.0, weight=1.0,
                 jointGain=2.0, PAFGain=1.0, DepthGain=1.0, NormalGain=1.0, TextGain=1.0,
                 SegmentGain=2.0, DistanceLevelGain=1.0, DenoisingGain=1.0,
                 leftRightGain=10.0, PenaltyGain=0.8, InstanceGain=1.0,
                 SuperpointGain=1.0, GeolocationGain=1.0,
                 activeRaw=120.0,
                 PenaltyForegroundNormalized=False,
                 **kwargs):
        super(HeatmapCoreLoss, self).__init__(**kwargs)
        self.channel_ranges = channel_ranges
        self.weight = weight
        self.scale = scale
        # The raw "active" heatmap value (cfg['heatmapActive'], +120 for the standard
        # encoding). After the *scale multiply in call(), values live in [-S, +S] with
        # S = activeRaw*scale (= lossBaseHeatmapScale for the standard encoding).
        self.active_raw = float(activeRaw)
        self.active_scaled = float(activeRaw) * float(scale)
        #----------------------------------------
        self.joint_gain         = jointGain
        self.paf_gain           = PAFGain
        self.depthmap_gain      = DepthGain
        self.normal_gain        = NormalGain
        self.text_gain          = TextGain
        self.segmentation_gain  = SegmentGain
        self.distancelevel_gain = DistanceLevelGain
        self.denoising_gain     = DenoisingGain
        self.penalty_gain       = PenaltyGain
        # Obs 19 / Experiment N2: when True the FN penalties normalise over FOREGROUND
        # pixels only (sum/mask) instead of all pixels — see _fn_penalty(). Default False
        # (legacy behaviour); the term grows ~1000x when enabled, retune lossPenaltyGain.
        self.penalty_foreground_normalized = bool(PenaltyForegroundNormalized)
        self.leftRightGain      = leftRightGain
        self.instance_gain      = InstanceGain
        self.superpoint_gain    = SuperpointGain
        self.geolocation_gain   = GeolocationGain
        #----------------------------------------
        # Extract channel start/end from the C-resolved layout table.
        # Fallbacks match the old hardcoded values so the loss is backward-compatible
        # if an older DataLoader is passed, but the canonical path is channel_ranges
        # from DataLoader.channel_ranges (populated by db_build_heatmap_layout).
        def _rng(name, default):
            r = channel_ranges.get(name, default)
            return int(r[0]), int(r[1])

        self.j_s,   self.j_e   = _rng('keypoints',    (0,  17))
        self.p_s,   self.p_e   = _rng('paf',          (17, 29))
        self.d_s,   self.d_e   = _rng('depth',        (29, 30))
        self.n_s,   self.n_e   = _rng('normals',      (30, 33))
        self.dl_s,  self.dl_e  = _rng('depth_levels', (33, 34))
        self.dn_s,  self.dn_e  = _rng('denoise',      (34, 37))
        self.lr_s,  self.lr_e  = _rng('left_right',   (37, 39))
        self.seg_s, self.seg_e = _rng('segmentation', (39, 73))
        # Text = seg_start + offset-of-"Label/Text"-within-seg (class index 7)
        self.text_idx = self.seg_s + self._TEXT_SEG_OFFSET
        # Instance — optional; absent until the center+size head is added (§1 of plan)
        inst = channel_ranges.get('instance')
        self.inst_s = int(inst[0]) if inst is not None else None
        self.inst_e = int(inst[1]) if inst is not None else None
        # SuperPoint — optional; absent until the .superpoint pipeline is enabled.
        # K dense PCA-descriptor heatmap channels, dense signed MSE like normals.
        sp = channel_ranges.get('superpoint')
        self.sp_s = int(sp[0]) if sp is not None else None
        self.sp_e = int(sp[1]) if sp is not None else None
        # Geolocation — optional single global lat/lon density channel (GEOLOCATION.md);
        # dense signed MSE like superpoint. Absent until heatmapAddGeolocation is enabled.
        geo = channel_ranges.get('geolocation')
        self.geo_s = int(geo[0]) if geo is not None else None
        self.geo_e = int(geo[1]) if geo is not None else None
        #----------------------------------------
        # Gains are LINEAR multipliers of each group's mean-squared error since the
        # Experiment N1 linearization (PLAN.md Obs 18; previously they multiplied inside
        # the square, making the effective weight gain^2 and this dict off by a factor
        # of gain). Effective per-channel weight = gain / group channel count.
        magnitude = dict()
        magnitude["Joint"]       = self.joint_gain  / max(1, self.j_e  - self.j_s)
        magnitude["PAF"]         = self.paf_gain    / max(1, self.p_e  - self.p_s)
        magnitude["Depth"]       = self.depthmap_gain
        magnitude["Normal"]      = self.normal_gain / max(1, self.n_e  - self.n_s)
        magnitude["Text"]        = self.text_gain
        magnitude["Segmentation"]= self.segmentation_gain / max(1, self.seg_e - self.seg_s)
        magnitude["DistanceLvl"] = self.distancelevel_gain
        magnitude["Denoising"]   = self.denoising_gain
        magnitude["PenaltyGain"] = self.penalty_gain
        magnitude["leftRightGain"] = self.leftRightGain
        if self.inst_s is not None:
            magnitude["Instance"] = self.instance_gain / 3
        if self.sp_s is not None:
            magnitude["Superpoint"] = self.superpoint_gain / max(1, self.sp_e - self.sp_s)
        if self.geo_s is not None:
            magnitude["Geolocation"] = self.geolocation_gain / max(1, self.geo_e - self.geo_s)
        print("Loss relative magnitudes (approximation) : ", magnitude)
        #----------------------------------------

    def _fn_penalty(self, y_true_slice, y_pred_slice, gain, float_type):
        """False-negative penalty: foreground-masked MSE toward the GT value (PLAN.md Obs 15).

        Normalisation (PLAN.md Obs 19 / Experiment N2):
        - legacy (PenaltyForegroundNormalized=False): reduce_mean over ALL pixels — the
          foreground error sum is divided by the full H*W*C pixel count, diluting the
          term ~1000x relative to what lossPenaltyGain nominally suggests.
        - foreground-normalized (True): reduce_sum(masked)/(reduce_sum(mask)+1.0) — a true
          per-foreground-pixel mean. The +1.0 (matching the peakiness-mask normalisation)
          keeps the term exactly zero on foreground-free batches. ~1000x larger for the
          same inputs, so lossPenaltyGain must be retuned sharply downward when enabling.
        """
        fg = tf.cast(y_true_slice > 0.33, float_type)
        sq = tf.square(y_true_slice - y_pred_slice) * fg
        if self.penalty_foreground_normalized:
            penalty = tf.reduce_sum(sq) / (tf.reduce_sum(fg) + 1.0)
        else:
            penalty = tf.reduce_mean(sq)
        return penalty * gain * self.penalty_gain

    # 1/2 working
    #@tf.function(reduce_retracing=True) #<-Be careful this might cause performance hit if not enough GPU memory present
    def call(self, y_true, y_pred):
        # Ensure both y_true and y_pred are cast to float32
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision
        y_true_cast = tf.cast(y_true, float_type) * self.scale
        y_pred_cast = tf.cast(y_pred, float_type) * self.scale
        #At this point values are in [-S, +S] with S = self.active_scaled
        #(= lossBaseHeatmapScale for the standard raw [-120,120] encoding).
        #Background = -S, active peak = +S — NOT [0,1].

        penalty_joint_disambiguation = 0
        #----------------------------------------------------------------------------

        #----------------------------------------------------------------------------
        # ---- Peakiness / single-peak encouragement — DISABLED 2026-07-08 ----
        # REGRESSION (serials 275, 276 — see failed_experiments.csv): the [0,1]-shifted,
        # GT-peak-masked reformulation of this term (commit 096ab72, meant to fix the Obs-16
        # signed-value pole) collapses the entire joint/PAF head to the -120 floor.
        # Mechanism: minimising (sum-max)/(sum+eps) over the p=(y+S)/(2S) values drives every
        # NON-winning joint/PAF channel toward p=0 (raw -120) at each GT-peak pixel, weighted
        # penalty_gain*0.5 = 5.0. Any given channel is the "loser" at ~16x more peak pixels
        # than it is the "winner", so the net gradient pushes all channels to the floor while
        # the per-channel MSE (weight 1) is too weak to hold them up. Only a single, flickering
        # "winner" channel survives (left_knee @275, left_eye @276). Confirmed with
        # debug_joint_amplitudes.py: 16/17 joints + 11/12 PAFs sit at raw -120 vs GT +120,
        # val_hm_hdm_not0_joints = 0.002 (vs serial 272 = 0.039). The pre-096ab72 (serial <=272)
        # form accidentally pushed non-max channels UP and kept joints alive (weak ~+28 raw
        # undershoot, Obs 15). Removed rather than re-tuned: single-peak selection is already
        # handled by the per-channel MSE + the FN penalty. See PLAN.md Obs 16.
        #----------------------------------------------------------------------------
        peakiness_penalty_total = 0.0
        #----------------------------------------------------------------------------

        # Calculate MSE for 2D Joint Heatmaps
        #----------------------------------------------------------------------------
        y_true_joint = y_true_cast[..., self.j_s:self.j_e]
        y_pred_joint = y_pred_cast[..., self.j_s:self.j_e]
        # Gain OUTSIDE the square (Experiment N1, PLAN.md Obs 18): effective weight is
        # linear in the config gain. Same pattern for every dense term below.
        mse_joint = self.joint_gain * tf.reduce_mean(tf.square(y_true_joint - y_pred_joint))
        #----------------------------------------------------------------------------
        # Joint False Negative Penalty — foreground-masked MSE toward the GT value.
        # (Was (1.0 - y_pred)^2: with values in [-S,+S], S=10, that targeted raw ~12
        # instead of the raw-120 active value, capping converged peaks at ~28 raw —
        # PLAN.md Obs 15, verified 2026-06-12 with debug_joint_amplitudes.py.)
        penalty_joint = self._fn_penalty(y_true_joint, y_pred_joint, self.joint_gain, float_type)
        #----------------------------------------------------------------------------

        # Calculate MSE for PAFs
        #----------------------------------------------------------------------------
        y_true_PAF = y_true_cast[..., self.p_s:self.p_e]
        y_pred_PAF = y_pred_cast[..., self.p_s:self.p_e]
        mse_PAF = self.paf_gain * tf.reduce_mean(tf.square(y_true_PAF - y_pred_PAF))
        #----------------------------------------------------------------------------
        # PAF False Negative Penalty — foreground-masked MSE toward the GT value
        # (same recalibration as penalty_joint above, PLAN.md Obs 15).
        penalty_PAF = self._fn_penalty(y_true_PAF, y_pred_PAF, self.paf_gain, float_type)
        #----------------------------------------------------------------------------

        # Calculate MSE for DepthMap
        #----------------------------------------------------------------------------
        y_true_depthmap = y_true_cast[..., self.d_s:self.d_e]
        y_pred_depthmap = y_pred_cast[..., self.d_s:self.d_e]
        mse_depthmap = self.depthmap_gain * tf.reduce_mean(tf.square(y_true_depthmap - y_pred_depthmap))
        #----------------------------------------------------------------------------

        # Calculate MSE for Normals
        #----------------------------------------------------------------------------
        y_true_normal = y_true_cast[..., self.n_s:self.n_e]
        y_pred_normal = y_pred_cast[..., self.n_s:self.n_e]
        mse_normal = self.normal_gain * tf.reduce_mean(tf.square(y_true_normal - y_pred_normal))
        #----------------------------------------------------------------------------

        # Distance Levels
        #----------------------------------------------------------------------------
        y_true_distance_level = y_true_cast[..., self.dl_s:self.dl_e]
        y_pred_distance_level = y_pred_cast[..., self.dl_s:self.dl_e]
        mse_distance_level = self.distancelevel_gain * tf.reduce_mean(
            tf.square(y_true_distance_level - y_pred_distance_level))
        #----------------------------------------------------------------------------

        # Denoising Output
        #----------------------------------------------------------------------------
        y_true_denoise = y_true_cast[..., self.dn_s:self.dn_e]
        y_pred_denoise = y_pred_cast[..., self.dn_s:self.dn_e]
        mse_denoising = self.denoising_gain * tf.reduce_mean(tf.square(y_true_denoise - y_pred_denoise))
        #----------------------------------------------------------------------------

        # Left/Right Output
        #----------------------------------------------------------------------------
        y_true_leftright = y_true_cast[..., self.lr_s:self.lr_e]
        y_pred_leftright = y_pred_cast[..., self.lr_s:self.lr_e]
        mse_leftright = self.leftRightGain * tf.reduce_mean(tf.square(y_true_leftright - y_pred_leftright))
        #----------------------------------------------------------------------------

        # Segmentation — bounded slice; does NOT absorb trailing channels (e.g. instance)
        #----------------------------------------------------------------------------
        y_true_segm = y_true_cast[..., self.seg_s:self.seg_e]
        y_pred_segm = y_pred_cast[..., self.seg_s:self.seg_e]
        mse_segmentation = self.segmentation_gain * tf.reduce_mean(tf.square(y_true_segm - y_pred_segm))
        #----------------------------------------------------------------------------

        # Text — "Label / Text" segmentation class (index 7 within seg group)
        #----------------------------------------------------------------------------
        y_true_text = y_true_cast[..., self.text_idx:self.text_idx + 1]
        y_pred_text = y_pred_cast[..., self.text_idx:self.text_idx + 1]
        mse_text = self.text_gain * tf.reduce_mean(tf.square(y_true_text - y_pred_text))
        #----------------------------------------------------------------------------

        # Instance segmentation — optional; activates when 'instance' group is present.
        # SEGMENTATION_PLAN §E channels: +0 person_blob, +1 person_lr_bridge, +2 person_head.
        #----------------------------------------------------------------------------
        if self.inst_s is not None:
            y_true_inst = y_true_cast[..., self.inst_s:self.inst_e]
            y_pred_inst = y_pred_cast[..., self.inst_s:self.inst_e]
            mse_instance = self.instance_gain * tf.reduce_mean(tf.square(y_true_inst - y_pred_inst))
            # False-negative penalty on ALL 3 instance channels — foreground-masked MSE toward
            # the GT value. blob/head are joint-like positive peaks that would collapse to
            # background under dense MSE alone without this term (PLAN.md Obs 14); lr_bridge is
            # signed, so this masks its positive (right) half while the negative (left) half
            # rides on dense MSE — the same treatment the proven signed PAFs get. The target is
            # y_true, not 1.0 (PLAN.md Obs 15 recalibration).
            penalty_instance = self._fn_penalty(y_true_inst, y_pred_inst, self.instance_gain, float_type)
        else:
            mse_instance    = 0.0
            penalty_instance = 0.0
        #----------------------------------------------------------------------------

        # SuperPoint — optional dense PCA-descriptor heatmaps; signed MSE like normals.
        #----------------------------------------------------------------------------
        if self.sp_s is not None:
            y_true_sp = y_true_cast[..., self.sp_s:self.sp_e]
            y_pred_sp = y_pred_cast[..., self.sp_s:self.sp_e]
            mse_superpoint = self.superpoint_gain * tf.reduce_mean(tf.square(y_true_sp - y_pred_sp))
        else:
            mse_superpoint = 0.0
        #----------------------------------------------------------------------------

        # Geolocation — optional single global density channel; signed MSE like superpoint.
        #----------------------------------------------------------------------------
        if self.geo_s is not None:
            y_true_geo = y_true_cast[..., self.geo_s:self.geo_e]
            y_pred_geo = y_pred_cast[..., self.geo_s:self.geo_e]
            mse_geolocation = self.geolocation_gain * tf.reduce_mean(tf.square(y_true_geo - y_pred_geo))
        else:
            mse_geolocation = 0.0
        #----------------------------------------------------------------------------

        # Sum the losses
        #----------------------------------------------------------------------------
        total_loss = (mse_joint + mse_PAF + mse_depthmap + mse_normal + mse_text + mse_segmentation +
                      mse_distance_level + mse_denoising + mse_leftright + mse_instance + mse_superpoint +
                      mse_geolocation +
                      penalty_joint + penalty_instance + penalty_joint_disambiguation +
                      penalty_PAF + peakiness_penalty_total)
        #----------------------------------------------------------------------------

        return total_loss * self.weight

    def get_config(self):
        cfg = super(HeatmapCoreLoss, self).get_config()
        cfg.update({
            'channel_ranges': {k: list(v) for k, v in self.channel_ranges.items()},
            'scale':               self.scale,
            'weight':              self.weight,
            'jointGain':           self.joint_gain,
            'PAFGain':             self.paf_gain,
            'DepthGain':           self.depthmap_gain,
            'NormalGain':          self.normal_gain,
            'TextGain':            self.text_gain,
            'SegmentGain':         self.segmentation_gain,
            'DistanceLevelGain':   self.distancelevel_gain,
            'DenoisingGain':       self.denoising_gain,
            'leftRightGain':       self.leftRightGain,
            'PenaltyGain':         self.penalty_gain,
            'InstanceGain':        self.instance_gain,
            'SuperpointGain':      self.superpoint_gain,
            'GeolocationGain':     self.geolocation_gain,
            'activeRaw':           self.active_raw,
            'PenaltyForegroundNormalized': self.penalty_foreground_normalized,
        })
        return cfg


#-------------------------------------------------------------------------------
class DepthNormalsNLLLoss(keras.losses.Loss):
    """Aleatoric uncertainty loss for the depth+normals hm_nll output.

    The model emits hm_nll with shape [B, H, W, 2*C]:
      - first C channels : mean prediction  (tanh space, same as hm before Rescaling)
      - last  C channels : log-variance     (unconstrained)

    y_true has shape [B, H, W, C]: the raw depth+normals ground-truth in the
    same integer scale as the hm output (divided by `scale` inside call to get
    it to tanh space for a numerically consistent NLL).

    NLL = 0.5 * exp(-log_var) * (gt - mu)^2 + 0.5 * log_var
    """

    def __init__(self, scale=120.0, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)
        self.weight = weight

    def call(self, y_true, y_pred):
        y_pred  = tf.cast(y_pred, tf.float32)
        gt      = tf.cast(y_true, tf.float32) / self.scale  # → tanh space [-1, 1]
        n       = tf.shape(y_pred)[-1] // 2
        mu      = y_pred[..., :n]
        log_var = tf.clip_by_value(y_pred[..., n:], -10.0, 10.0)
        nll     = 0.5 * tf.exp(-log_var) * tf.square(gt - mu) + 0.5 * log_var
        return tf.reduce_mean(nll) * self.weight

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'scale': self.scale, 'weight': self.weight})
        return cfg


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class GloVeMSELoss(keras.losses.Loss):

    def __init__(self, weight=1.0, **kwargs):
        super(GloVeMSELoss, self).__init__(**kwargs)
        self.weight = weight

    #Don't use this
    #@tf.function(reduce_retracing=True) #<-Be careful this might cause performance hit if not enough GPU memory present
    def call(self, y_true, y_pred):

        # Check if the last dimension is 300
        #ass1=tf.debugging.assert_equal(tf.shape(y_true)[-1],300,message="GloVeMSELoss y_true does not have the expected shape [..., 300].")
        #ass2=tf.debugging.assert_equal(tf.shape(y_pred)[-1],300,message="GloVeMSELoss y_pred does not have the expected shape [..., 300].")
        #tf.control_dependencies([ass1,ass2])

        # Ensure both y_true and y_pred are cast to float32
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision

        # Extract the embedding vectors (elements 1 to 51)
        y_true_glove = tf.cast(y_true, float_type)  #If upper is uncommented set to 1:
        y_pred_glove = tf.cast(y_pred, float_type)  #If upper is uncommented set to 1:

        # Compute the weighted MSE, tf.abs for always postiive values ?
        mse_glove = tf.reduce_mean(tf.square(y_true_glove - y_pred_glove), axis=-1)

        # Apply the weight to the loss
        total_loss = mse_glove * self.weight

        return total_loss


#-------------------------------------------------------------------------------
class GloVeCosineLoss(keras.losses.Loss):

    def __init__(self, weight=1.0, **kwargs):
        super(GloVeCosineLoss, self).__init__(**kwargs)
        self.weight = weight

    def call(self, y_true, y_pred):
        # Cast to float32 only when needed: l2_normalize on low-precision vectors risks underflow
        if y_true.dtype != tf.float32:
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

        y_true = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred = tf.nn.l2_normalize(y_pred, axis=-1)

        cosine_similarity = tf.reduce_sum(y_true * y_pred, axis=-1)  # Cosine similarity
        cosine_distance = 1 - cosine_similarity  # Convert to a loss function

        return self.weight * cosine_distance


#-------------------------------------------------------------------------------
class GloVeHybridLoss(keras.losses.Loss):
    """Hybrid GloVe embedding loss combining MSE, cosine distance, and norm regularization.

    Why three terms?

    MSE alone (GloVeMSELoss):
        Minimising L2 distance pushes the prediction toward the centroid of the
        training distribution — a low-magnitude average vector.  The model learns
        to predict a "safe" middle-of-the-road embedding that minimises squared
        error but has poor directional alignment, explaining cosine_sim ≈ 0.22.

    Cosine loss alone (GloVeCosineLoss):
        Optimising only direction creates a degenerate solution: the network can
        maximise cosine similarity by predicting any scaled version of the true
        vector, including a near-zero vector whose direction is numerically
        unstable.  It also gives no gradient when the predicted magnitude is
        large but the direction is already close.

    Hybrid (this class):
        MSE anchors the magnitude, cosine loss anchors the direction.  Together
        they force the prediction to match both the direction and the scale of
        the ground-truth GloVe embedding.

    norm_reg_weight (magnitude regularization):
        New term: penalises |‖ŷ‖ − ‖y‖|² so the predicted embedding has the
        same L2 norm as the ground-truth.  This is important because the
        downstream tokens_multihot head uses the GloVe outputs directly as
        input features; if predicted norms drift (e.g. collapse toward zero as
        cosine loss alone would allow), the multihot head receives degraded
        features.  Default weight 0.01 keeps this term small relative to the
        main losses.
    """

    def __init__(self, mse_weight=1.0, cosine_weight=1.0, norm_reg_weight=0.01, **kwargs):
        super(GloVeHybridLoss, self).__init__(**kwargs)
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        # norm_reg_weight: controls how strongly we penalise predicted-vs-true
        # norm mismatch.  Small by default (0.01) — just enough to prevent
        # magnitude collapse without dominating the direction signal.
        self.norm_reg_weight = norm_reg_weight

    def call(self, y_true, y_pred):
        # Cast to float32 only when needed: l2_normalize and tf.norm on
        # low-precision 300-dim vectors can underflow (norm → 0 → NaN).
        # In float32 mode the inputs are already float32 so no cast is needed
        # and none is inserted into the graph (dtype is static in tf.function).
        if y_true.dtype != tf.float32:
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

        # --- MSE term: penalises element-wise distance (anchors magnitude) ---
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

        # --- Cosine term: penalises directional misalignment ------------------
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
        cosine_loss = 1.0 - tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)

        # --- Norm regularization: penalises predicted-vs-true norm mismatch --
        # Without this term, optimising cosine loss alone can let the predicted
        # vector drift to any scale.  Keeping the norm close to the GT norm
        # ensures the multihot head receives consistently-scaled features.
        norm_pred = tf.norm(y_pred, axis=-1)
        norm_true = tf.norm(y_true, axis=-1)
        norm_reg = tf.square(norm_pred - norm_true)

        return (self.mse_weight * mse_loss + self.cosine_weight * cosine_loss + self.norm_reg_weight * norm_reg)


#-------------------------------------------------------------------------------
class GloVeContrastiveLoss(keras.losses.Loss):
    """Decode-aligned loss for the t00..t07 embedding regression (TOKENS.md 10.9).

    GloVeHybridLoss rewards being CLOSE to the target vector, so under uncertainty the optimum is
    the centroid of the plausible words -- measured on serial 287, predictions sit at cos 0.86-0.91
    to their slot's mean target while real targets sit at 0.45-0.53, and nearest-neighbour decoding
    then snaps to hub words. Decoding is argmax cosine over the vocabulary, so this loss trains that
    decision directly: softmax over cosine(prediction, every candidate row) / temperature,
    cross-entropy against the ground-truth row. Wrong-but-nearby hubs are explicit negatives.

    candidate_table: (V, D) targets in the SAME scaled space the C loader emits (the
        GloVe_D300.embeddings.bin rows). The ground-truth row is recovered in-graph as the candidate
        with cosine ~1 to y_true, so the generator is unchanged.
    Masked to zero loss: empty slots (all-zero target), OOV rows (constant vector after the loader's
        affine transform) and targets that are not in candidate_table.
    norm_reg_weight: same |‖ŷ‖-‖y‖|² term as GloVeHybridLoss -- the softmax is scale-free, and the
        multihot head consumes these vectors as features.
    """

    def __init__(self, candidate_table, temperature=0.05, weight=1.0, norm_reg_weight=0.01, **kwargs):
        super(GloVeContrastiveLoss, self).__init__(**kwargs)
        import numpy as np
        table = np.asarray(candidate_table, dtype=np.float32)
        self.table_n = tf.constant(table / np.maximum(np.linalg.norm(table, axis=1, keepdims=True), 1e-8))
        self.temperature = float(temperature)
        self.weight = float(weight)
        self.norm_reg_weight = float(norm_reg_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        t_sim = tf.matmul(tf.nn.l2_normalize(y_true, axis=-1), self.table_n, transpose_b=True)   # (B, V)
        gt = tf.argmax(t_sim, axis=-1)
        valid = tf.logical_and(tf.reduce_max(t_sim, axis=-1) > 0.9999,
                               tf.math.reduce_std(y_true, axis=-1) > 1e-6)                      # not empty / OOV
        logits = tf.matmul(tf.nn.l2_normalize(y_pred, axis=-1), self.table_n, transpose_b=True) / self.temperature
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=logits)
        norm_reg = tf.square(tf.norm(y_pred, axis=-1) - tf.norm(y_true, axis=-1))
        return tf.where(valid, self.weight * ce + self.norm_reg_weight * norm_reg, 0.0)


#-------------------------------------------------------------------------------
class DescriptorLoss(keras.losses.Loss):
    """MSE + cosine loss for supervising the bridge descriptor head
    against pre-computed DINOv2 embeddings (768-dim, L2-normalised).

    DINOv2 vectors are L2-normalised at source, so directional alignment
    (cosine) is as important as magnitude (MSE).  Equal weights by default.
    norm_reg keeps predicted norms close to 1.0 so downstream layers
    receive consistently-scaled features.
    """

    def __init__(self, weight=1.0, norm_reg_weight=0.01, **kwargs):
        super(DescriptorLoss, self).__init__(**kwargs)
        self.mse_weight = weight * 0.5
        self.cosine_weight = weight * 0.5
        self.norm_reg_weight = norm_reg_weight

    def call(self, y_true, y_pred):
        # Cast to float32 only when needed — same rationale as GloVeHybridLoss.
        if y_true.dtype != tf.float32:
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
        cosine_loss = 1.0 - tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
        norm_pred = tf.norm(y_pred, axis=-1)
        norm_true = tf.norm(y_true, axis=-1)
        norm_reg = tf.square(norm_pred - norm_true)
        return (self.mse_weight * mse_loss + self.cosine_weight * cosine_loss + self.norm_reg_weight * norm_reg)


#-------------------------------------------------------------------------------
class MultiHotLoss(keras.losses.Loss):

    def __init__(self, weight=1.0, **kwargs):
        super(MultiHotLoss, self).__init__(**kwargs)
        #self.class_weights = tf.constant(class_weights, dtype=keras.backend.floatx())[None, :]
        self.weight = weight

    def call(self, y_true, y_pred):

        # Check if the last dimension is 2037
        #ass1=tf.debugging.assert_equal(tf.shape(y_true)[-1],2037,message="MultiHotLoss y_true does not have the expected shape [..., 2037].")
        #ass2=tf.debugging.assert_equal(tf.shape(y_pred)[-1],2037,message="MultiHotLoss y_pred does not have the expected shape [..., 2037].")
        #tf.control_dependencies([ass1,ass2])

        # Ensure both y_true and y_pred are cast to float32
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision
        y_true_onehot = tf.cast(y_true, float_type)  #[0:2037]
        y_pred_onehot = tf.cast(y_pred, float_type)  #[0:2037]

        token_loss_function = keras.losses.BinaryCrossentropy(from_logits=False)

        # Compute the original loss
        original_loss = token_loss_function(y_true_onehot, y_pred_onehot)

        # Multiply by the weight
        #weighted_loss = 1.0 *  tf.exp(original_loss) #Try perplexity loss e^loss
        scaled_weighted_loss = self.weight * original_loss

        return scaled_weighted_loss


#-------------------------------------------------------------------------------
class WeightedBinaryCrossEntropyManual(keras.losses.Loss):

    def __init__(self, class_weights, weight=1.0, name="weighted_binary_crossentropy"):
        super(WeightedBinaryCrossEntropyManual, self).__init__(name=name)
        # Convert class weights to a tensor
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.weight = weight

    #2/2 working
    #@tf.function(reduce_retracing=True) #<-Be careful this might cause performance hit if not enough GPU memory present
    def call(self, y_true, y_pred):

        # Ensure both y_true and y_pred are cast to float32
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision
        y_true = tf.cast(y_true, float_type)
        y_pred = tf.cast(y_pred, float_type)

        # Manually compute binary cross-entropy for each class and sample
        bce_loss = -(y_true * tf.math.log(y_pred + 1e-7) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-7))

        # Reshape class_weights to (1, Number Of Classes) to enable broadcasting across the batch
        class_weights = tf.reshape(self.class_weights, [1, -1])

        # Apply class weights to each class in the loss
        weighted_bce_loss = bce_loss * class_weights

        # Compute the mean loss across all classes and batch samples
        #return tf.reduce_mean(weighted_bce_loss) #<- this is probably incorrect :S
        return self.weight * tf.reduce_mean(weighted_bce_loss, axis=-1)
        # OR
        #return self.weight * tf.reduce_mean(tf.reduce_sum(weighted_bce_loss, axis=1))  # Sum over classes, then mean over batch
        # OR
        #return self.weight * tf.reduce_sum(weighted_bce_loss, axis=1)  # No outer mean, do sum


#-------------------------------------------------------------------------------
class WeightedBinaryCrossEntropy(keras.losses.Loss):

    def __init__(self, class_weights, weight=1.0, name="weighted_binary_crossentropy"):
        super(WeightedBinaryCrossEntropy, self).__init__(name=name)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.weight = weight

    def call(self, y_true, y_pred):
        # Ensure both y_true and y_pred are cast to float32
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision
        y_true = tf.cast(y_true, float_type)
        y_pred = tf.cast(y_pred, float_type)

        # Prevent log(0) issues
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Use Keras' built-in binary cross-entropy function for numerical stability
        bce_loss = tf.keras.backend.binary_crossentropy(y_true, y_pred)

        # Reshape class_weights to (1, 2048) to enable broadcasting across the batch
        class_weights = tf.reshape(self.class_weights, [1, -1])

        # Apply class weights element-wise
        weighted_bce_loss = bce_loss * class_weights

        # Compute mean across classes, keeping batch dimension
        return self.weight * tf.reduce_mean(weighted_bce_loss, axis=-1)  # Keep per-sample loss


#-------------------------------------------------------------------------------
class WeightedFocalLoss(keras.losses.Loss):
    """Focal loss with per-class frequency weights and per-class alpha balancing.

    Two improvements over the original implementation:

    gamma (focusing parameter):
        Raised default from 2.0 → 4.0.  With 17 977 classes and typically only
        5-20 active per sample the positive/negative ratio is ≈1:1000.  At
        gamma=2.0 easy true-negatives (the overwhelming majority) are still
        contributing meaningful gradient that drowns out the rare positives.
        gamma=4.0 suppresses these easy negatives far more aggressively, which
        pushes recall up from the observed 0.047 toward a more balanced regime.

    alpha (per-class positive emphasis):
        Standard focal-loss alpha term (Lin et al. 2017).  Applied as a
        per-element weight:
          • positive examples (y_true == 1) receive weight  alpha
          • negative examples (y_true == 0) receive weight  1 - alpha
        Default 0.5 (neutral): class_weights already encodes inverse-frequency
        per class, so setting alpha > 0.5 double-counts the imbalance.
        Empirically, alpha=0.75 caused the model to predict almost every class
        as positive (tokens_multihot BinaryAccuracy dropped from 0.9996 → 0.028
        because FP dominated), so alpha is reset to 0.5.
        Increase alpha only if recall is still near zero after many epochs and
        class_weights alone are insufficient.
    """

    def __init__(self, class_weights, gamma=4.0, alpha=0.5, weight=1.0, name="weighted_focal_loss"):
        super(WeightedFocalLoss, self).__init__(name=name)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        # gamma: focal focusing exponent.  Higher = stronger suppression of easy
        # negatives.  Default raised from 2.0 to 4.0 to handle the extreme
        # positive/negative imbalance in the 17 977-class multihot setting.
        self.gamma = gamma
        # alpha: positive-class balance weight ∈ [0, 1].
        # Positive examples are weighted by alpha, negatives by (1 - alpha).
        # Default 0.75 means positives get 3× the gradient weight of negatives.
        self.alpha = alpha
        self.weight = weight

    def call(self, y_true, y_pred):

        # Ensure both y_true and y_pred are cast to float32
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision
        y_true = tf.cast(y_true, float_type)
        y_pred = tf.cast(y_pred, float_type)

        # Prevent log(0) instability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Compute BCE loss
        bce_loss = tf.keras.backend.binary_crossentropy(y_true, y_pred)

        # Compute pt (probability of the true class)
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)

        # Stable focal modulation: (1 - pt)^gamma computed via exp(gamma * log(...))
        # to avoid pow() numerical issues near pt=1.
        focal_modulation = tf.exp(self.gamma * tf.math.log(1.0 - pt + 1e-7))

        # Combine BCE loss with focal modulation
        focal_loss = bce_loss * focal_modulation

        # Alpha balancing: up-weight positive-class examples by self.alpha,
        # down-weight negatives by (1 - self.alpha).  This is applied before the
        # per-class frequency weights so the two are multiplicative.
        alpha_weight = tf.where(y_true == 1,
                                tf.ones_like(y_true) * self.alpha,
                                tf.ones_like(y_true) * (1.0 - self.alpha))
        focal_loss = focal_loss * alpha_weight

        # Apply per-class frequency weights (inverse-frequency from the dataset)
        class_weights = tf.reshape(self.class_weights, [1, -1])  # (1, num_classes)
        weighted_focal_loss = focal_loss * class_weights

        # Mean loss per sample (average over classes)
        return self.weight * tf.reduce_mean(weighted_focal_loss, axis=-1)


#-------------------------------------------------------------------------------
class AsymmetricLoss(keras.losses.Loss):
    """Asymmetric Loss for multi-label classification (Ridnik et al. 2021,
    arXiv:2009.14119), TOKENS.md Experiment K4 -- a candidate replacement for
    `WeightedFocalLoss` on the 17977-way `tokens_multihot` head.

    Same symptom as WeightedFocalLoss exists to fix (Obs 26: with ~5-20 positives
    out of 17977 classes, easy true-negatives flood the gradient), but ASL treats
    positives and negatives asymmetrically instead of sharing one gamma:

    gamma_neg (focusing on negatives, default 4.0):
        Same role as WeightedFocalLoss's single gamma -- suppresses easy
        true-negatives. Kept high because the imbalance is extreme.

    gamma_pos (focusing on positives, default 1.0):
        Deliberately much lower than gamma_neg. Positive labels in this task are
        already rare and often noisy/ambiguous (a caption may omit a visible
        concept), so down-weighting *hard* positives as aggressively as hard
        negatives would suppress exactly the gradient recall depends on.

    clip (probability shifting, default 0.05):
        Negatives with predicted probability below `clip` are treated as exact
        (loss driven to ~0) before the focal term is applied, discarding the
        long tail of already-confident true negatives entirely rather than
        merely down-weighting them -- ASL's core mechanism beyond ordinary focal
        loss, and the paper's stated source of its recall gain over focal loss.

    Per-class inverse-frequency weighting is kept identical to
    `WeightedFocalLoss` (multiplicative, after the asymmetric focal term) so the
    two losses are swappable via config without touching the class-weight
    conditioning (§5.3) this project already relies on.
    """

    def __init__(self, class_weights, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, weight=1.0,
                 name="asymmetric_loss"):
        super(AsymmetricLoss, self).__init__(name=name)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.weight = weight

    def call(self, y_true, y_pred):
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision
        y_true = tf.cast(y_true, float_type)
        y_pred = tf.cast(y_pred, float_type)
        eps = 1e-7

        xs_pos = y_pred
        xs_neg = 1.0 - y_pred
        # Probability shifting: gives easy negatives (xs_neg already small, i.e.
        # y_pred already close to 1... wait, xs_neg small means y_pred close to
        # 1 which is a FALSE positive, not an easy negative) a hard floor at 0 --
        # re-read: xs_neg is P(negative) = 1-y_pred, so an "easy" true negative
        # has y_pred near 0, xs_neg near 1; shifting xs_neg UP by `clip` (capped
        # at 1) only changes anything for xs_neg close to 1 already, pushing the
        # loss for confidently-correct negatives to exactly log(1)=0 once
        # xs_neg+clip saturates at 1.0.
        if self.clip is not None and self.clip > 0:
            xs_neg = tf.clip_by_value(xs_neg + self.clip, 0.0, 1.0)

        log_pos = y_true * tf.math.log(tf.clip_by_value(xs_pos, eps, 1.0))
        log_neg = (1.0 - y_true) * tf.math.log(tf.clip_by_value(xs_neg, eps, 1.0))
        loss = log_pos + log_neg

        # Asymmetric focusing: one_sided_gamma picks gamma_pos on true positives,
        # gamma_neg on true negatives, per-element (not per-class/per-sample).
        pt = xs_pos * y_true + xs_neg * (1.0 - y_true)
        one_sided_gamma = self.gamma_pos * y_true + self.gamma_neg * (1.0 - y_true)
        # Stable (1-pt)^gamma via exp(gamma*log(...)), matching WeightedFocalLoss's
        # convention rather than tf.pow (avoids issues right at pt=1).
        one_sided_w = tf.exp(one_sided_gamma * tf.math.log(1.0 - pt + eps))
        loss = loss * one_sided_w

        class_weights = tf.reshape(self.class_weights, [1, -1])  # (1, num_classes)
        weighted_loss = -loss * class_weights

        # Mean loss per sample (average over classes), matching WeightedFocalLoss.
        return self.weight * tf.reduce_mean(weighted_loss, axis=-1)


#-------------------------------------------------------------------------------
#https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/dssim.py
class DSSIMLoss(keras.losses.Loss):
    """Difference of Structural Similarity (DSSIM loss function).
    Clipped between 0 and 0.5

    Note: You should add a regularization term like a l2 loss in addition to this one.
    Note: The `kernel_size` should be appropriate for the output size.

    # Arguments
        k1: Parameter of the SSIM (default 0.01)
        k2: Parameter of the SSIM (default 0.03)
        kernel_size: Size of the sliding window (default 3)
        max_value: Max value of the output (default 1.0)
    """

    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0, scalar=1.0):
        super(DSSIMLoss, self).__init__()
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.scalar = scalar
        self.c1 = (self.k1 * self.max_value)**2
        self.c2 = (self.k2 * self.max_value)**2

    def extract_image_patches(self, x, ksizes, ssizes, padding='SAME', data_format='channels_last'):
        kernel = [1, ksizes[0], ksizes[1], 1]
        strides = [1, ssizes[0], ssizes[1], 1]
        if data_format == 'channels_first':
            x = tf.transpose(x, (0, 2, 3, 1))
        patches = tf.image.extract_patches(images=x, sizes=kernel, strides=strides, rates=[1, 1, 1, 1], padding=padding)
        return patches

    def call(self, y_true, y_pred):
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision
        y_true = tf.cast(y_true, float_type)
        y_pred = tf.cast(y_pred, float_type)
        kernel = [self.kernel_size, self.kernel_size]

        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)

        y_true = tf.reshape(y_true, [-1, y_pred_shape[1], y_pred_shape[2], y_pred_shape[3]])
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[1], y_pred_shape[2], y_pred_shape[3]])

        patches_pred = self.extract_image_patches(y_pred, kernel, kernel, padding='VALID')
        patches_true = self.extract_image_patches(y_true, kernel, kernel, padding='VALID')

        # Reshape to get the var in the cells
        patches_pred_shape = tf.shape(patches_pred)
        bs, w, h, ch = patches_pred_shape[0], patches_pred_shape[1], patches_pred_shape[2], patches_pred_shape[3]
        patches_pred = tf.reshape(patches_pred, [bs, w, h, -1])
        patches_true = tf.reshape(patches_true, [bs, w, h, -1])

        # Get mean
        u_true = tf.reduce_mean(patches_true, axis=-1)
        u_pred = tf.reduce_mean(patches_pred, axis=-1)
        # Get variance
        var_true = tf.math.reduce_variance(patches_true, axis=-1)
        var_pred = tf.math.reduce_variance(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = tf.reduce_mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (2 * covar_true_pred + self.c2)
        denom = (tf.square(u_true) + tf.square(u_pred) + self.c1) * (var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
        dssim = (1.0 - ssim) / 2.0
        return self.scalar * tf.reduce_mean(dssim)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np

    # Number of classes
    num_classes = 2048
    batch_size = 4  # Arbitrary batch size for testing

    # Generate random class weights
    class_weights = np.random.rand(num_classes).astype(np.float32)
    print("Class weights shape:", class_weights.shape)

    # Generate random true labels (binary values 0 or 1)
    y_true = np.random.randint(0, 2, size=(batch_size, num_classes)).astype(np.float32)
    print("Y-True shape:", y_true.shape)

    # Generate random predicted probabilities (values between 0 and 1)
    y_pred = np.random.rand(batch_size, num_classes).astype(np.float32)
    print("Y-Pred shape:", y_pred.shape)

    # Initialize loss functions
    manual_bce_loss = WeightedBinaryCrossEntropyManual(class_weights)
    bce_loss = WeightedBinaryCrossEntropy(class_weights)
    focal_loss = WeightedFocalLoss(class_weights, gamma=4.0)

    # Compute losses
    loss_manual_bce = manual_bce_loss.call(y_true, y_pred)
    loss_bce = bce_loss.call(y_true, y_pred)
    loss_focal = focal_loss.call(y_true, y_pred)

    print("loss_manual_bce:", type(loss_manual_bce))
    print("loss_bce:", type(loss_bce))
    print("loss_focal:", type(loss_focal))
    loss_manual_bce_np = loss_manual_bce.numpy()
    loss_bce_np = loss_bce.numpy()
    loss_focal_np = loss_focal.numpy()

    # Print results
    print("Manual Weighted BCE Loss:", loss_manual_bce_np, " -> ", loss_manual_bce_np.shape)
    print("Keras BCE Loss:", loss_bce_np, " -> ", loss_bce_np.shape)
    print("Weighted Focal Loss:", loss_focal_np, " -> ", loss_focal_np.shape)

    # Check if manual BCE and Keras BCE are close
    diff = np.abs(loss_manual_bce_np - loss_bce)
    print("Difference between manual and Keras BCE Loss:", diff)

    # Verify the loss shape (should be batch_size)
    assert loss_manual_bce_np.shape == (
        batch_size, ), "Manual BCE Loss shape incorrect " + str(loss_manual_bce_np.shape)
    assert loss_bce_np.shape == (batch_size, ), "Keras BCE Loss shape incorrect " + str(loss_bce_np.shape)
    assert loss_focal_np.shape == (batch_size, ), "Focal Loss shape incorrect " + str(loss_focal_np.shape)

    print("All tests passed successfully!")


#-------------------------------------------------------------------------------
class GeolocationKLLoss(keras.losses.Loss):
    """KL( q || p ) for the global-pooled geolocation head (PLAN.md Experiment U).

    q = y_true: per-sample normalised (sum=1) density over the HxW world grid.
    p = y_pred: the geo_grid head's softmax over the same grid. Both (B,H,W).
    Summed over the grid, mean over the batch (Keras default reduction).
    """

    def __init__(self, weight=1.0, concentration_weighting=False, **kwargs):
        super(GeolocationKLLoss, self).__init__(**kwargs)
        self.weight = weight
        # Exp 281: down-weight ambiguous (unlocalizable) samples by their GT concentration —
        # the soft form of D5's mask. Applied HERE (not via Keras sample_weight) because this
        # model's outputs are a LIST, for which a name-keyed sample_weight dict breaks Keras'
        # resolve_path (KeyError: 0). The weight is derived from y_true itself, so it needs no
        # extra plumbing and applies to train and val identically.
        self.concentration_weighting = concentration_weighting

    def call(self, y_true, y_pred):
        q = tf.cast(y_true, tf.float32)
        p = tf.cast(y_pred, tf.float32)
        eps = 1e-8
        kl = tf.reduce_sum(q * (tf.math.log(q + eps) - tf.math.log(p + eps)), axis=[1, 2])  # (B,)
        if self.concentration_weighting:
            # concentration = peak mass fraction (q sums to 1, so per-sample max IS it), rescaled
            # [uniform,1] -> [0,1] as relative localizability. Then BATCH-MEAN-NORMALISED so the
            # overall gradient magnitude (and val-loss scale) is preserved and only the RELATIVE
            # emphasis shifts toward localizable samples — without this, PLONK-on-COCO's near-flat
            # concentrations (max ~0.03) would scale every sample's loss to ~0 and freeze the head.
            hw = tf.cast(tf.shape(q)[1] * tf.shape(q)[2], tf.float32)
            conc = tf.reduce_max(q, axis=[1, 2])                       # (B,)
            uni = 1.0 / hw
            w = tf.clip_by_value((conc - uni) / (1.0 - uni), 0.0, 1.0)  # (B,) in [0,1]
            w = w / (tf.reduce_mean(w) + eps)                          # mean(w) -> 1
            kl = kl * w
        return self.weight * kl

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'weight': self.weight, 'concentration_weighting': self.concentration_weighting})
        return cfg


#-------------------------------------------------------------------------------
class GeolocationKmAccuracy(Metric):
    """Great-circle localisation accuracy for the geo_grid head (PLAN.md Exp U).

    argmax of pred & GT (B,H,W) -> (row,col) -> (lat,lon) on the equirect grid
    (row 0 = +90 lat, col 0 = -180 lon; native 64x128 upsampled x4/x2 to 256x256,
    matching run_plonk_teacher.build_equirect_grid) -> haversine km -> fraction of
    samples within threshold_km. Replaces the background-saturated per-pixel hdm.
    """
    _DEG2RAD = 0.017453292519943295

    def __init__(self, name='geo_acc2500', threshold_km=2500.0,
                 grid_h=256, grid_w=256, native_h=64, native_w=128,
                 min_concentration=0.0, **kwargs):
        super(GeolocationKmAccuracy, self).__init__(name=name, **kwargs)
        self.threshold_km = float(threshold_km)
        self.grid_h = int(grid_h); self.grid_w = int(grid_w)
        self.native_h = int(native_h); self.native_w = int(native_w)
        # Exp 281: localizable-subset eval. min_concentration>0 counts only samples whose
        # GT peak mass fraction (== concentration, since y_true sums to 1) clears the threshold,
        # so the accuracy is read on genuinely localizable images. 0.0 = all samples (default).
        self.min_concentration = float(min_concentration)
        self.hits = self.add_weight(name='hits', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def _latlon_rad(self, flat_idx):
        flat_idx = tf.cast(flat_idx, tf.float32)
        row = tf.floor(flat_idx / self.grid_w)
        col = flat_idx - row * self.grid_w
        nrow = row * (self.native_h / self.grid_h)          # 256 -> 64 rows
        ncol = col * (self.native_w / self.grid_w)          # 256 -> 128 cols
        lat = 90.0 - nrow * (180.0 / (self.native_h - 1))
        lon = -180.0 + ncol * (360.0 / self.native_w)
        return lat * self._DEG2RAD, lon * self._DEG2RAD

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(y_true)[0], -1])
        yp = tf.reshape(tf.cast(y_pred, tf.float32), [tf.shape(y_pred)[0], -1])
        lat1, lon1 = self._latlon_rad(tf.argmax(yt, axis=1))
        lat2, lon2 = self._latlon_rad(tf.argmax(yp, axis=1))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = tf.sin(dlat / 2.0) ** 2 + tf.cos(lat1) * tf.cos(lat2) * tf.sin(dlon / 2.0) ** 2
        km = 6371.0 * 2.0 * tf.asin(tf.minimum(1.0, tf.sqrt(a)))
        # Exp 281: mask to the localizable subset (concentration >= threshold). yt sums to 1,
        # so its per-sample max IS the concentration. min_concentration=0.0 -> mask all-ones.
        mask = tf.cast(tf.reduce_max(yt, axis=1) >= self.min_concentration, tf.float32)
        self.hits.assign_add(tf.reduce_sum(tf.cast(km <= self.threshold_km, tf.float32) * mask))
        self.total.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.hits / (self.total + 1e-8)

    def reset_state(self):
        self.hits.assign(0.0)
        self.total.assign(0.0)
