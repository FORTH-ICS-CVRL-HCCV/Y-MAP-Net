"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Muon: MomentUm Orthogonalized by Newton-schulz
Keller Jordan et al.  https://kellerjordan.github.io/posts/muon/
Muon is Scalable for LLM Training (Moonlight), Liu et al.  https://arxiv.org/abs/2502.16982

keras.optimizers.Muon (checked up to Keras 3.15.1 / master) only orthogonalizes 2D variables and
silently falls back to AdamW for everything else -- in YMAPNet that is every Conv2D kernel, ~53%
of the parameters. MuonConv also orthogonalizes 4D kernels by flattening (kh,kw,cin,cout) to
(kh*kw*cin, cout) as the reference implementation does, and routes vector-like matrices
(depthwise kernels, Dense(1), single-channel convs) plus the network output heads to AdamW.

Newton-Schulz divides by the gradient norm, so the step size does not depend on gradient magnitude:
a matrix whose gradient has collapsed to rounding noise still takes a full-size step in that noise
direction (cf. softmax collapse, arXiv 2610.31642). Check this first if a Muon run spikes.
"""
import re

import keras
from keras.src import ops

# Output heads / per-pixel output convs of YMAPNet -- standard Muon practice keeps these on AdamW.
YMAPNET_OUTPUT_LAYERS = [r"^hm_", r"^tokens_multihot/", r"^descriptors/", r"^geo_logits/", r"^t\d\d/", r"^d\d\d/"]


@keras.saving.register_keras_serializable(package='NNMuon')
class MuonConv(keras.optimizers.Muon):

    def _should_use_adamw(self, variable):
        shape = variable.shape
        if len(shape) not in (2, 4):
            return True
        # Depthwise (kh,kw,C,1), Dense(1) and 1-channel convs have no matrix structure to orthogonalize
        if min(_prod(shape[:-1]), shape[-1]) == 1:
            return True
        if self.exclude_embeddings and "embedding" in variable.path.lower():
            return True
        for keyword in self.exclude_layers:
            if re.search(keyword, variable.path):
                return True
        return False

    def _muon_update_step(self, gradient, variable, lr, m):
        self.assign_add(m, ops.add(gradient, m * (self.momentum - 1)))
        if self.nesterov:
            g = ops.add(gradient, self.momentum * m)
        else:
            g = m
        shape = variable.shape
        g = ops.reshape(g, (_prod(shape[:-1]), shape[-1]))
        update = self.zeropower_via_newtonschulz5(g, self.ns_steps)
        # lr_adjust's sqrt(max(n,m)) scaling is computed on the flattened 2D matrix
        self.assign_sub(variable, ops.reshape(self.lr_adjust(lr * update), shape))


def _prod(dims):
    p = 1
    for d in dims:
        p *= d
    return p
