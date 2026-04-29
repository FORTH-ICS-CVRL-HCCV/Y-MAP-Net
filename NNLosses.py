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
                 total_epochs=None):
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
        #----------------------------------------
        self.best = None
        self.bestEpoch = None
        self.bestLog = None
        if self.mode == 'min':
            self.best = float('inf')
        elif self.mode == 'max':
            self.best = -float('inf')
        #----------------------------------------
        # Epoch timing for ETA estimation
        if total_epochs is None:
            try:
                import json
                with open("configuration.json", "r") as _f:
                    _cfg = json.load(_f)
                total_epochs = int(_cfg.get("epochs", 0))
            except Exception:
                total_epochs = 0
        self.total_epochs = total_epochs
        self._epoch_times = []  # list of (epoch_index, end_timestamp)
        #----------------------------------------

    def reset(self):
        print(self.WARNING, "Resetting Checkpointer", self.ENDC)
        self.best = None
        self.bestEpoch = None
        self.bestLog = None
        if self.mode == 'min':
            self.best = float('inf')
        elif self.mode == 'max':
            self.best = -float('inf')

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
            with open("status.txt", "w") as f:
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
        float_type = tf.keras.backend.floatx()

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
        float_type = tf.keras.backend.floatx()

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

    def __init__(self, scale=1.0, weight=1.0, jointGain=2.0, PAFGain=1.0, DepthGain=1.0, NormalGain=1.0, TextGain=1.0,
                 SegmentGain=2.0, DistanceLevelGain=1.0, DenoisingGain=1.0, leftRightGain=10.0, PenaltyGain=0.8,
                 **kwargs):
        super(HeatmapCoreLoss, self).__init__(**kwargs)
        self.weight = weight
        self.scale = scale
        #----------------------------------------
        self.joint_gain = jointGain
        self.paf_gain = PAFGain
        self.depthmap_gain = DepthGain
        self.normal_gain = NormalGain
        self.text_gain = TextGain
        self.segmentation_gain = SegmentGain
        self.distancelevel_gain = DistanceLevelGain
        self.denoising_gain = DenoisingGain
        self.penalty_gain = PenaltyGain  # Tune this value
        self.leftRightGain = leftRightGain
        #----------------------------------------
        # Vanila MSE loss parity calculation between different modalities
        # It doesn't make sense since each thing has its own MSE
        #----------------------------------------
        magnitude = dict()
        magnitude["Joint"] = self.joint_gain / 17
        magnitude["PAF"] = self.paf_gain / (29 - 17)
        magnitude["Depth"] = self.depthmap_gain
        magnitude["Normal"] = self.normal_gain / 3
        magnitude["Text"] = self.text_gain
        magnitude["Segmentation"] = self.segmentation_gain / (72 - 39)
        magnitude["DistanceLvl"] = self.distancelevel_gain
        magnitude["Denoising"] = self.denoising_gain
        magnitude["PenaltyGain"] = self.penalty_gain
        magnitude["leftRightGain"] = self.leftRightGain
        print("Loss relative magnitudes (approximation) : ", magnitude)
        #----------------------------------------

    # 1/2 working
    #@tf.function(reduce_retracing=True) #<-Be careful this might cause performance hit if not enough GPU memory present
    def call(self, y_true, y_pred):
        # Ensure both y_true and y_pred are cast to float32
        float_type = tf.float32  # Always accumulate losses/metrics in float32 for numerical stability under mixed precision
        y_true_cast = tf.cast(y_true, float_type) * self.scale
        y_pred_cast = tf.cast(y_pred, float_type) * self.scale
        #At this point values are converted 0.0 - 1.0

        # Check if the last dimension is 44
        #ass1=tf.debugging.assert_equal(tf.shape(y_true)[-1],44,message="Heatmap Loss y_true does not have the expected shape [..., 44].")
        #ass2=tf.debugging.assert_equal(tf.shape(y_pred)[-1],44,message="Heatmap Loss y_pred does not have the expected shape [..., 44].")
        #tf.control_dependencies([ass1,ass2])

        penalty_joint_disambiguation = 0
        #----------------------------------------------------------------------------

        #----------------------------------------------------------------------------
        # ---- Peakiness / single-peak encouragement ----
        #----------------------------------------------------------------------------
        eps = 1e-6

        # joints
        sum_joint = tf.reduce_sum(y_pred_cast[..., :17], axis=-1)
        max_joint = tf.reduce_max(y_pred_cast[..., :17], axis=-1)
        peakiness_joint = tf.reduce_mean((sum_joint - max_joint) / (sum_joint + eps))

        # PAFs
        sum_paf = tf.reduce_sum(y_pred_cast[..., 17:29], axis=-1)
        max_paf = tf.reduce_max(y_pred_cast[..., 17:29], axis=-1)
        peakiness_paf = tf.reduce_mean((sum_paf - max_paf) / (sum_paf + eps))

        peakiness_penalty_total = self.penalty_gain * 0.5 * (peakiness_joint + peakiness_paf)
        #----------------------------------------------------------------------------

        # Calculate MSE for heatmaps 0-16 2D Joint Heatmaps
        #----------------------------------------------------------------------------
        y_true_joint = y_true_cast[..., :17]
        y_pred_joint = y_pred_cast[..., :17]
        mse_joint = tf.reduce_mean(tf.square((y_true_joint - y_pred_joint) * self.joint_gain))
        #----------------------------------------------------------------------------
        # Joint False Negative Penalty
        penalty_joint = tf.reduce_mean(tf.square(
            (1.0 - y_pred_joint)) * tf.cast(y_true_joint > 0.33, float_type)) * self.joint_gain * self.penalty_gain
        #----------------------------------------------------------------------------

        # Calculate MSE for PAFs 17-28
        #----------------------------------------------------------------------------
        y_true_PAF = y_true_cast[..., 17:29]
        y_pred_PAF = y_pred_cast[..., 17:29]
        mse_PAF = tf.reduce_mean(tf.square((y_true_PAF - y_pred_PAF) * self.paf_gain))
        #----------------------------------------------------------------------------
        # PAF False Negative Penalty
        penalty_PAF = tf.reduce_mean(tf.square(
            (1.0 - y_pred_PAF)) * tf.cast(y_true_PAF > 0.33, float_type)) * self.paf_gain * self.penalty_gain
        #----------------------------------------------------------------------------

        # Calculate MSE for DepthMap 29
        #----------------------------------------------------------------------------
        y_true_depthmap = y_true_cast[..., 29]
        y_pred_depthmap = y_pred_cast[..., 29]
        mse_depthmap = tf.reduce_mean(tf.square((y_true_depthmap - y_pred_depthmap) * self.depthmap_gain))
        #----------------------------------------------------------------------------

        # Calculate MSE for Normals 30-32
        #----------------------------------------------------------------------------
        y_true_normal = y_true_cast[..., 30:33]
        y_pred_normal = y_pred_cast[..., 30:33]
        mse_normal = tf.reduce_mean(tf.square((y_true_normal - y_pred_normal) * self.normal_gain))
        #----------------------------------------------------------------------------

        #Distance Levels
        #----------------------------------------------------------------------------
        y_true_distance_level = y_true_cast[..., 33]
        y_pred_distance_level = y_pred_cast[..., 33]
        mse_distance_level = tf.reduce_mean(
            tf.square((y_true_distance_level - y_pred_distance_level) * self.distancelevel_gain))
        #----------------------------------------------------------------------------

        #Denoising Output
        #----------------------------------------------------------------------------
        y_true_denoise = y_true_cast[..., 34:37]
        y_pred_denoise = y_pred_cast[..., 34:37]
        mse_denoising = tf.reduce_mean(tf.square((y_true_denoise - y_pred_denoise) * self.denoising_gain))
        #----------------------------------------------------------------------------

        #Left/Right Output
        #----------------------------------------------------------------------------
        y_true_leftright = y_true_cast[..., 37:39]
        y_pred_leftright = y_pred_cast[..., 37:39]
        mse_leftright = tf.reduce_mean(tf.square((y_true_leftright - y_pred_leftright) * self.leftRightGain))
        #----------------------------------------------------------------------------

        #Segmentation
        #----------------------------------------------------------------------------
        #New Multiplexed Loss Should be added here :
        y_true_segm = y_true_cast[..., 39:]
        y_pred_segm = y_pred_cast[..., 39:]
        mse_segmentation = tf.reduce_mean(tf.square((y_true_segm - y_pred_segm) * self.segmentation_gain))
        #----------------------------------------------------------------------------

        # Calculate MSE for Text
        #----------------------------------------------------------------------------
        y_true_text = y_true_cast[..., 46]
        y_pred_text = y_pred_cast[..., 46]
        mse_text = tf.reduce_mean(tf.square((y_true_text - y_pred_text) * self.text_gain))
        #----------------------------------------------------------------------------
        """
        #----------------------------------------------------------------------------
        # Segmentation (GT = single int channel, Pred = class logits)
        #----------------------------------------------------------------------------
        # Heatmaps: 0..38
        # Segmentation GT: channel 39 (integer class id)
        # Segmentation Pred: channels 39 : 39 + N_SEG_CLASSES
        #----------------------------------------------------------------------------

        mse_text = 0.0
        N_SEG_CLASSES = 34  # <-- Careful, this needs to be set correctly
 
        # Read raw int8 labels from channel 39
        y_true_segm_raw = tf.cast(y_true[..., 39], tf.int32)

        # Shift labels: -120 -> 0, -119 -> 1, ..., etc.
        y_true_segm = y_true_segm_raw + 120

        # Prediction: segmentation logits
        y_pred_segm_logits = y_pred_cast[..., 39:39 + N_SEG_CLASSES]

        # Sparse categorical cross entropy
        seg_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_segm,
            y_pred_segm_logits,
            from_logits=True
        )

        mse_segmentation = tf.reduce_mean(seg_loss) * self.segmentation_gain
        #-----------------------------------------------------------------------

        """

        # Sum the losses
        #----------------------------------------------------------------------------
        total_loss = (mse_joint + mse_PAF + mse_depthmap + mse_normal + mse_text + mse_segmentation +
                      mse_distance_level + mse_denoising + mse_leftright + penalty_joint +
                      penalty_joint_disambiguation + penalty_PAF + peakiness_penalty_total)
        #----------------------------------------------------------------------------

        return total_loss * self.weight


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
