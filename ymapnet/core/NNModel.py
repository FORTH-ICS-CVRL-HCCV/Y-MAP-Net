"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""

import inspect
import numpy as np
import keras
import tensorflow as tf
import time
from keras import layers, models
from keras.layers import Input, Layer, DepthwiseConv2D, UpSampling2D, Flatten, Dropout, SpatialDropout2D, Conv2D, ZeroPadding2D, Conv2DTranspose, Cropping2D, Concatenate, AvgPool2D, BatchNormalization, ReLU, Reshape, Dense, Add, UpSampling2D, MaxPooling2D, AveragePooling2D, Activation, MaxPool2D, Concatenate, Permute
from keras.models import Model


#-------------------------------------------------------------------------------
def retrieveModelOutputDimensions(model):
    try:
        output_layer = model.get_layer(name="hm")  #heatmap_output
    except:
        print("Could not find heatmap output layer, assuming that it is the last layer")
        output_layer = model.layers[0]  # Assuming the output layer is the first/ layer
        #output_layer = model.layers[-1]  # Assuming the output layer is the last layer
    output_shape = output_layer.output.shape
    output_size = (output_shape[1], output_shape[2])
    numberOfHeatmaps = output_shape[3]
    print("Number of Heatmaps is ", numberOfHeatmaps)
    print("Output Shape is ", output_size)
    return output_shape[1], output_shape[2], output_shape[3]


#----------------------------------------------------------------------------------------
# Saved .keras checkpoints drift against whatever Keras happens to be installed at
# load time in both directions -- e.g. BatchNormalization configs carrying legacy
# tf.keras 'renorm*' keys removed in Keras 3.9+, or MultiHeadAttention configs
# carrying a 'use_gate' key from a newer Keras than is currently installed. Built-in
# layers deserialize with registered_name=None, so a custom_objects entry keyed by
# class name is never consulted (see _retrieve_class_or_fn in serialization_lib.py) --
# patch the shared base from_config to drop whatever the installed __init__ doesn't
# recognize, so old/new checkpoints keep loading across Keras version bumps.
_layer_from_config_original = Layer.from_config.__func__


@classmethod
def _layer_from_config_compat(cls, config):
    config = dict(config)
    try:
        params = inspect.signature(cls.__init__).parameters.values()
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            accepted = {p.name for p in params}
            config = {k: v for k, v in config.items() if k in accepted}
    except (TypeError, ValueError):
        pass
    return _layer_from_config_original(cls, config)


Layer.from_config = _layer_from_config_compat


#----------------------------------------------------------------------------------------
def load_keypoints_model(model_path, compile=True):
    from ymapnet.utils.tools import bcolors
    from ymapnet.core.NNLosses import HeatmapDistanceMetric, GloVeMSELoss, HeatmapCoreLoss, WeightedBinaryCrossEntropy, HeatmapDistanceMetricPartial, CustomTopKCategoricalAccuracy
    print(bcolors.OKGREEN, "Loading %s model.. " % model_path, bcolors.ENDC)
    print(bcolors.OKGREEN, "This takes a while to compile the model and ensure the best possible performance.. ", bcolors.ENDC)
    try:
        #Regular keras loading until V3 that breaks
        import keras
        doModelCompilation = compile  #<- Does this have any effect on loading speed?

        print("Keras is NOW loading the saved model..")
        start = time.time()
        model = keras.saving.load_model(
            model_path, custom_objects={
                'HeatmapDistanceMetric': HeatmapDistanceMetric,
                'HeatmapDistanceMetricPartial': HeatmapDistanceMetricPartial,
                'GloVeMSELoss': GloVeMSELoss,
                'WeightedBinaryCrossEntropy': WeightedBinaryCrossEntropy,
                'HeatmapCoreLoss': HeatmapCoreLoss,
                'CustomTopKCategoricalAccuracy': CustomTopKCategoricalAccuracy,
                'BilinearUpsampling2D': BilinearUpsampling2D,
                'CoordConv2D': CoordConv2D
            }, compile=doModelCompilation)  #, safe_mode=True)
        seconds = time.time() - start
        print("Loading the model took ", seconds, " seconds..")

        input_layer = model.layers[0]  # Assuming the input layer is the first layer

        #This works when we have one output
        #output_layer = model.layers[-1]  # Assuming the output layer is the last layer
        # Access the layer by name
        try:
            output_layer = model.get_layer(name="hm")  #heatmap_output
        except:
            print("Could not find heatmap output layer, assuming that it is the last layer")
            output_layer = model.layers[0]  # Assuming the output layer is the first/ layer
            #output_layer = model.layers[-1]  # Assuming the output layer is the last layer

        input_shape = input_layer.output.shape
        output_shape = output_layer.output.shape
        # Check the shape of the input layer
        input_size = (input_shape[1], input_shape[2])  #This is a little dodgie
        # Get the output layer of the model

        print("Output Shape ", output_shape)
        output_size = (output_shape[1], output_shape[2])
        numberOfHeatmaps = output_shape[3]
    except Exception as e:
        #Fallback to TF2 saved_model loading
        print(bcolors.FAIL, "An exception occurred trying to load keras model:", str(e), bcolors.ENDC)
        print(bcolors.OKGREEN, "Falling back to TF saved_model loader.. ", bcolors.ENDC)
        #model_path_dir = os.path.dirname(path)
        model = tf.saved_model.load(model_path)
        signatures = model.signatures
        signature_keys = signatures.keys()
        if 'serving_default' in signature_keys:
            signature_key = 'serving_default'
        else:
            signature_key = list(signature_keys)[0]  # Use the first signature if 'serving_default' is not available
        input_info = signatures[signature_key].inputs
        output_info = signatures[signature_key].outputs
        input_shape = input_info[0].shape
        output_shape = output_info[0].shape
        input_size = (input_shape[1], input_shape[2])
        output_size = (output_shape[1], output_shape[2])
        numberOfHeatmaps = output_shape[0]  # Assuming the first dimension represents the number of heatmaps

    print("Input shape is ", input_size)
    print("Number of Heatmaps is ", numberOfHeatmaps)
    print("Output Shape is ", output_size)

    return model, input_size, output_size, numberOfHeatmaps


#-------------------------------------------------------------------------------
def load_pictorial_only_ymapnet(
        model_path,
        *,
        keep_hm16=False,
        keep_descriptors=False,
        use_tf_function=True,
        use_xla=False,
        warmup=True,
        warmup_iters=5,
        force_save_reload=False,  # <---- Save/Reload
):
    """
    Load via load_keypoints_model() and prune token heads.
    Optionally force save+reload to lock in pruned graph.

    Returns:
        pruned_model, fast_call_or_None, input_size, output_size, numberOfHeatmaps
    """
    import os
    import tempfile
    model, input_size, output_size, numberOfHeatmaps = load_keypoints_model(model_path)

    if not isinstance(model, tf.keras.Model):
        print("Warning: not a tf.keras.Model (TF saved_model fallback). Returning as-is.")
        return model, None, input_size, output_size, numberOfHeatmaps

    # --- Keep pictorial outputs only ---
    outputs = []
    outputs.append(model.get_layer("hm").output)

    if keep_hm16:
        try:
            outputs.append(model.get_layer("hm_16b").output)
        except Exception:
            pass

    if keep_descriptors:
        try:
            outputs.append(model.get_layer("descriptors").output)
        except Exception:
            pass

    pruned_outputs = outputs[0] if len(outputs) == 1 else outputs

    pruned = tf.keras.Model(
        inputs=model.inputs,
        outputs=pruned_outputs,
        name=model.name + "_pictorial_only",
    )

    # ----------------------------------------------------------
    # Force save & reload (graph hard-pruning)
    # ----------------------------------------------------------
    if force_save_reload:
        print("Forcing save+reload of pruned model...")
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "pruned.keras")
        print("Result is: ", tmp_path)

        pruned.save(tmp_path)
        pruned = tf.keras.models.load_model(tmp_path, compile=False, safe_mode=False)

        # Optional: clean up directory
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass

    # Compile (minimal; mainly for predict plumbing)
    pruned.compile(optimizer="adam", loss=None, jit_compile=False)

    # ----------------------------------------------------------
    # Fast inference path
    # ----------------------------------------------------------
    fast_call = None
    if use_tf_function:
        h, w = int(input_size[1]), int(input_size[0])
        inp_dtype = pruned.inputs[0].dtype

        signature = [tf.TensorSpec(shape=[None, h, w, 3], dtype=inp_dtype)]

        @tf.function(input_signature=signature, jit_compile=bool(use_xla))
        def fast_call(x):
            return pruned(x, training=False)

        if warmup:
            dummy = tf.zeros([1, h, w, 3], dtype=inp_dtype)
            for _ in range(int(warmup_iters)):
                y = fast_call(dummy)
                if isinstance(y, (list, tuple)):
                    _ = tf.reduce_sum([tf.reduce_sum(t) for t in y])
                else:
                    _ = tf.reduce_sum(y)

    return pruned, fast_call, input_size, output_size, numberOfHeatmaps


#-------------------------------------------------------------------------------
def test_model_IO(model):
    try:
        model.save('test.keras')
        load_keypoints_model('test.keras')
        import os
        os.system("rm test.keras")
    except Exception as e:
        raise ValueError('Failed testing model IO')


#-------------------------------------------------------------------------------
class BilinearUpsampling2D(tf.keras.layers.Layer):
    """UpSampling2D with bilinear interpolation, safe under mixed precision.

    Under float32 this delegates to Keras's native UpSampling2D which uses a
    static-size CUDA kernel (same fast path as using UpSampling2D directly).

    Under bfloat16/float16 Keras's built-in bilinear upsampling calls into
    NumPy which rejects non-float32 dtypes ("data type not inexact").  For
    those dtypes we cast to float32, resize with tf.image.resize, then restore
    the original dtype — keeping the rest of the graph in the compute dtype.
    """

    def __init__(self, size=(2, 2), **kwargs):
        super(BilinearUpsampling2D, self).__init__(**kwargs)
        self.size = size
        # Stored sublayer for float32 inputs: uses Keras's optimised static-size
        # CUDA kernel.  Creating it once here avoids allocating a new layer
        # object on every call() invocation.
        # dtype='float32' pins this sublayer's compute dtype to float32.
        # Keras's autocast machinery will cast any bfloat16/float16 inputs to
        # float32 before UpSampling2D.call() runs — so numpy.finfo(bfloat16)
        # is never reached, even under a global mixed_bfloat16 policy.
        self._upsample_f32 = UpSampling2D(size=size, interpolation='bilinear', dtype='float32')

    def build(self, input_shape):
        # Pre-build the sublayer from the shape alone (dtype-agnostic), so
        # Keras 3 never needs to trace call() with a bfloat16 KerasTensor to
        # auto-build it.
        self._upsample_f32.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        h = None if input_shape[1] is None else input_shape[1] * self.size[0]
        w = None if input_shape[2] is None else input_shape[2] * self.size[1]
        return (input_shape[0], h, w, input_shape[3])

    def call(self, inputs):
        # _upsample_f32 has dtype='float32', so Keras autocasts inputs to fp32
        # before the CUDA kernel runs.  We cast the fp32 output back to the
        # caller's dtype so the rest of the graph stays in bfloat16/float16.
        x = self._upsample_f32(inputs)
        if x.dtype != inputs.dtype:
            x = tf.cast(x, inputs.dtype)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'size': self.size})
        return config


#-------------------------------------------------------------------------------
class ReshapeTiles(tf.keras.layers.Layer):

    def __init__(self, tile_height, tile_width, tile_depth, **kwargs):
        super(ReshapeTiles, self).__init__(**kwargs)
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.tile_depth = tile_depth

    def call(self, inputs):
        # Compute dynamic batch size
        batch_size = tf.shape(inputs)[0]
        num_tiles_h = tf.shape(inputs)[1]  # Height-wise tiles
        num_tiles_w = tf.shape(inputs)[2]  # Width-wise tiles

        # Flatten tile dimensions into a new batch dimension
        return tf.reshape(inputs,
                          (batch_size * num_tiles_h * num_tiles_w, self.tile_height, self.tile_width, self.tile_depth))


#-------------------------------------------------------------------------------
# Corrected ReshapeBack layer
class ReshapeBack(tf.keras.layers.Layer):

    def __init__(self, num_tiles, tile_features, **kwargs):
        super(ReshapeBack, self).__init__(**kwargs)
        self.num_tiles = num_tiles
        self.tile_features = tile_features

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0] // (self.num_tiles * self.num_tiles)
        return tf.reshape(inputs, (batch_size, self.num_tiles, self.num_tiles, self.tile_features))


#-------------------------------------------------------------------------------
# Custom TileExtractor layer
class TileExtractor(tf.keras.layers.Layer):

    def __init__(self, tile_size, **kwargs):
        super(TileExtractor, self).__init__(**kwargs)
        self.tile_size = tile_size

    def call(self, inputs):
        patches = tf.image.extract_patches(images=inputs, sizes=[1, self.tile_size, self.tile_size, 1], strides=[
            1, self.tile_size, self.tile_size, 1
        ], rates=[1, 1, 1, 1], padding='VALID')
        batch_size = tf.shape(inputs)[0]
        num_tiles = inputs.shape[1] // self.tile_size  # Assuming square input
        return tf.reshape(patches, [batch_size, num_tiles, num_tiles, self.tile_size, self.tile_size, 3])


#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def positional_encoding(seq_len, model_dim):
    pos = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, model_dim, 2) * -(np.log(10000.0) / model_dim))
    pos_encoding = np.zeros((seq_len, model_dim))
    pos_encoding[:, 0::2] = np.sin(pos * div_term)
    pos_encoding[:, 1::2] = np.cos(pos * div_term)
    return pos_encoding


#-------------------------------------------------------------------------------
class PositionalEncodingLayer(tf.keras.layers.Layer):

    def __init__(self, max_len, d_model, **kwargs):
        super(PositionalEncodingLayer, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        # Custom positional encoding logic here

    def call(self, inputs):
        # Forward logic for the layer
        return inputs + self.positional_encoding

    def get_config(self):
        # Include the custom arguments in the config
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model,
        })
        return config


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def add_token_output(bridge_output, input_layer, numTokens, activation='leaky_relu', dropoutRate=0.2,
                     nextTokenStrength=0.7, layer_width=1228, D=300, depth=7, dropoutDepth=4,
                     use_learnable_residuals=False, token_layer_width=0):
    # Only force float32 on output layers when the backbone is computing in a
    # lower-precision dtype (e.g. bfloat16 under mixed precision).  When the
    # global policy is already float32, dtype=None lets Keras use its default
    # path — no extra cast nodes are inserted into the graph or gradient tape.
    _out_dtype = 'float32' if keras.mixed_precision.global_policy().compute_dtype != 'float32' else None
    """
    Adds a token output branch to the network based on the bridge output and input layer,
    including convolutional and max-pooling layers before the dense layers.
    
    Args:
    - bridge_output: The output of the U-Net bridge (last layer before decoder).
    - input_layer: The original input layer to add skip connections from.
    - numTokens: The number of tokens to predict.
    - activation: The activation function to use.
    - dropoutRate: Dropout rate for the token output layers (if any).
    
    Returns:
    - token_output: The final token output layer.
    """

    # BUG FIX: the original code used max(1.0, nextTokenStrength), which hard-clamped
    # the value to 1.0 regardless of the configuration (e.g. nextTokenStrength=0.5).
    # With nextTokenStrength=1.0, the rescaling at lines below would set the current
    # token's contribution to 0.0 (scale = 1.0 - 1.0 = 0.0) and fully replace it
    # with the previous token's output — making every token after t00 a pure copy of
    # its predecessor.  This caused the observed cosine-similarity degradation from
    # t00=0.33 down to t07=-0.01 as errors compounded along the autoregressive chain.
    # Fix: clamp to [0.0, 1.0] so the configured blend ratio is actually respected.
    nextTokenStrength = min(max(0.0, nextTokenStrength), 1.0)

    # tokenLayerWidth (token_layer_width): 0 = inherit the bridge's flattened width
    # (dim_product, the legacy behaviour). When >0, the internal token Dense layers run at
    # this width, decoupling the token branch size from the bridge.  IMPORTANT: the Reshape
    # below flattens the spatial bridge and MUST stay at the true flattened bridge dim
    # (==layer_width==dim_product) or Keras raises a build-time shape error — so we keep the
    # reshape target (bridge_flat_dim) separate from the Dense width (dense_width).
    bridge_flat_dim = layer_width
    dense_width = token_layer_width if token_layer_width > 0 else layer_width

    # Reshape only if the input is not already flattened
    if len(bridge_output.shape) > 2:  # Check if the output needs reshaping
        x = layers.Reshape((bridge_flat_dim, ))(bridge_output)
    else:
        x = bridge_output  #Else use it without reshape

    #pos_encoding = positional_encoding(numTokens, 512)
    #x_with_pos = x + pos_encoding

    # Initialize a variable to hold the previous output for residual connections
    prev_output = None
    prev_glove = None

    # Define multiple outputs, one for each embedding
    glove_outputs = []

    for i in range(numTokens):

        thisTokenDropout = dropoutRate / (i + 1)
        #scaling_factor = layers.Dense(1, activation='sigmoid')(x)  # Learnable scaling factor

        # Sequential dense layers for each embedding
        x_token = layers.Dense(dense_width, name="Y_l1_t%u" % i)(x)
        x_token = apply_activation(x_token, activation)
        x_token = layers.LayerNormalization(dtype="float32")(x_token)

        if (thisTokenDropout > 0.01):
            x_token = layers.Dropout(thisTokenDropout)(
                x_token)  #Staggered dropout that gets lower as we progress to next tokens

        # Residual connection: blend current token features with previous token features
        x_token_residual = x_token
        if prev_output is not None:

            # Learnable residual mixing
            if i > 0 and use_learnable_residuals == True:
                residual_factor_x = layers.Dense(1, activation='sigmoid', name="res_factor_x_t%u" % i)(x_token_residual)
                residual_factor_prev = layers.Dense(1, activation='sigmoid',
                                                    name="res_factor_prev_t%u" % i)(prev_output)
                x_token_residual = layers.Multiply()([x_token_residual, residual_factor_x])
                prev_output = layers.Multiply()([prev_output, residual_factor_prev])
            elif i > 0 and use_learnable_residuals == False:
                x_token_residual = keras.layers.Rescaling(1.0 - nextTokenStrength, offset=0.0, name="Y_resrescale_t%u" %
                                                          i)(x_token_residual)  #<- Scale down current token
                prev_output = keras.layers.Rescaling(nextTokenStrength, offset=0.0, name="Y_rescale_t%u" %
                                                     i)(prev_output)  #<- Scale down previous token
            # Transform previous token features then add to the (already scaled) current token.
            # Using a separate variable avoids overwriting x_token_residual before the Add.
            prev_transformed = layers.Dense(dense_width, name="Y_l2X_t%u" % i)(prev_output)
            prev_transformed = apply_activation(prev_transformed, activation)
            prev_transformed = layers.LayerNormalization(dtype="float32")(prev_transformed)
            x_token_residual = layers.Add()([x_token_residual, prev_transformed])
        else:
            #Also add the residual part for the first token(!)
            x_token_residual = layers.Dense(dense_width, name="Y_l2F_t%u" % i)(x_token)
            x_token_residual = apply_activation(x_token_residual, activation)
            x_token_residual = layers.LayerNormalization(dtype="float32")(x_token_residual)
            x_token_residual = layers.Dropout(thisTokenDropout)(x_token_residual)

        for d in range(3, depth + 1):  # Loop from depth 3 to the minimum of (depth, 7)
            x_token_residual = layers.Dense(dense_width, name=f"Y_l{d}_t{i}")(x_token_residual)
            x_token_residual = apply_activation(x_token_residual, activation)
            x_token_residual = layers.LayerNormalization(dtype="float32")(x_token_residual)
            # Apply dropout only for depth 3 and 4
            #if (d <= dropoutDepth) and (thisTokenDropout>0.01):
            x_token_residual = layers.Dropout(thisTokenDropout)(x_token_residual)
        #-------------------------------------------
        connect_to_next = x_token_residual  #This is the next residual

        if prev_glove is not None:
            x_token = layers.Dense(D, name="Y_pre_final_combined_t%u" % i)(x_token_residual)
            x_token = apply_activation(x_token, activation)
            x_token = layers.LayerNormalization(dtype="float32")(x_token)
            x_token = layers.Add()([x_token, prev_glove])
        else:
            x_token = layers.Dense(D, name="Y_pre_final_t%u" % i)(x_token_residual)
            x_token = apply_activation(x_token, activation)
            x_token = layers.LayerNormalization(dtype="float32")(x_token)

        #This should be a tanh activation but in order for the appended network to have tanh try linear
        glove_output = layers.Dense(D, activation='tanh', name="t%02u" % i, dtype=_out_dtype)(x_token)
        prev_glove = glove_output
        #glove_output = layers.Multiply()([glove_output, scaling_factor])
        glove_outputs.append(glove_output)

        # Store the current token's output to be used as residual in the next iteration
        prev_output = connect_to_next  #glove_output or connect_to_next
    """
    #EXPERIMENT
    exp = x
    j=0
    for i in range(numTokens):
    #    for j in range(i + 1, min(i+2,numTokens)):  # Avoid duplicate pairs
                    exp = layers.Dense(layer_width, activation=activation, name="Y_l1_t%02u_t%02u"% (i, j))(exp)
                    exp = layers.LayerNormalization(dtype="float32")(exp)
                    exp = layers.Dropout(0.1)(exp) #Staggered dropout that gets lower as we progress to next tokens
                    exp = layers.Dense(layer_width, activation=activation, name="Y_l2_t%02u_t%02u"% (i, j))(exp)
                    exp = layers.LayerNormalization(dtype="float32")(exp)
                    exp = layers.Dropout(0.1)(exp) #Staggered dropout that gets lower as we progress to next tokens
                    exp = layers.Dense(layer_width, activation=activation, name="Y_l3_t%02u_t%02u"% (i, j))(exp)
                    exp = layers.LayerNormalization(dtype="float32")(exp)
                    exp = layers.Dropout(0.1)(exp) #Staggered dropout that gets lower as we progress to next tokens
                    exp = layers.Dense(layer_width, activation=activation, name="Y_l4_t%02u_t%02u"% (i, j))(exp)
                    exp = layers.LayerNormalization(dtype="float32")(exp)
                    exp = layers.Dropout(0.1)(exp) #Staggered dropout that gets lower as we progress to next tokens
                    #exp_output = layers.Dense(D, activation='tanh', name="t%02u_t%02u" % (i, j))(exp)
                    exp_output = layers.Dense(D, activation='tanh', name="d%02u" % (i))(exp)
                    glove_outputs.append(exp_output)
    """

    return glove_outputs


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def append_final_multihot_layer(glove_outputs, bridge_output=None, D=300, TokensOut=8, Classes=2037, layer_width=2048,
                                multihot_layer_width=0, depth=2, activation='leaky_relu', dropoutRate=0.2,
                                use_attention=False, skip_inputs=None):
    _out_dtype = 'float32' if keras.mixed_precision.global_policy().compute_dtype != 'float32' else None
    # multihot_layer_width=0 means fall back to layer_width (bridge-derived dim_product).
    # Set to a positive value (e.g. 1024) to decouple the classification head width
    # from the bridge bottleneck, which is typically too narrow for 17K-class multi-label.
    if multihot_layer_width > 0:
        layer_width = multihot_layer_width
    # Concatenate all glove outputs into a single vector

    concatenated_output = layers.Concatenate()(glove_outputs)

    if use_attention:
        concatenated_output = layers.Reshape(
            (TokensOut, D))(concatenated_output)  # Ensure (batch, seq_len, feature_dim)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=D // 4)(concatenated_output, concatenated_output)
        concatenated_output = layers.Concatenate()([concatenated_output, attn])
    else:
        concatenated_output = layers.Reshape((D * TokensOut, ))(concatenated_output)

    concatenated_output = layers.Flatten()(concatenated_output)

    # skip_inputs (config multihotInputSkip): extra views of the image CONCATENATED next to the
    # cascade, so the classifier is no longer a pure function of the 8 regressed embeddings
    # (TOKENS.md 4.3 / 10.6). Unlike the Add-merge bridge_output path below, nothing is forced
    # into a shared subspace. StopGradient keeps the multihot loss off the bridge (no new
    # Obs 7 token-vs-heatmap pressure); the per-block LayerNorm puts PReLU bridge activations
    # and near-unit-norm descriptors on the same scale as the tanh cascade before the first Dense.
    if skip_inputs:
        parts = [concatenated_output]
        for j, skip in enumerate(skip_inputs):
            skip = layers.Flatten(name="tmh_skip_flat%u" % j)(skip)
            skip = StopGradient(name="tmh_skip_sg%u" % j)(skip)
            skip = layers.LayerNormalization(dtype="float32", name="tmh_skip_ln%u" % j)(skip)
            parts.append(skip)
        concatenated_output = layers.Concatenate(name="tmh_skip_concat")(parts)

    #This actually works worse..
    if (bridge_output is None):
        print("Not connecting to bridge")
        x = concatenated_output
    else:
        bridge_output_flat = layers.Flatten()(bridge_output)
        intermediateADapter = layers.Dense(layer_width, name="tmh_adapter")(concatenated_output)
        intermediateADapter = apply_activation(intermediateADapter, activation)
        concatenated_output = layers.Add()([bridge_output_flat, intermediateADapter])
        x = concatenated_output

    for d in range(depth):  # Loop for desired depth (starting from 1)
        x = layers.Dense(layer_width, name=f"tm{d}")(x)
        x = apply_activation(x, activation)
        #layer_width = 2048 #<- If uncommented forces all layers after first to have this width
        x = layers.LayerNormalization(dtype="float32")(x)
        x = layers.Dropout(dropoutRate)(x)

    #Add a linear layer
    #x = layers.Dense(layer_width, activation='linear', name="tokens_scale")(x)

    # Add a single Dense layer of size D * maxtokens as the final output #NOT TokensOut
    # dtype=_out_dtype: only force float32 under mixed precision; when the global
    # policy is already float32, dtype=None avoids inserting spurious cast nodes.
    token_output = layers.Dense(Classes, activation='sigmoid', name="tokens_multihot", dtype=_out_dtype)(x)

    return token_output


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def append_descriptor_layer(bridge_layer=None, layer_width=768, activation='leaky_relu', dropoutRate=0.1, depth=1):
    _out_dtype = 'float32' if keras.mixed_precision.global_policy().compute_dtype != 'float32' else None
    # Concatenate all glove outputs into a single vector
    concatenated_output = layers.Flatten()(bridge_layer)
    x = concatenated_output

    for d in range(depth):  # Loop for desired depth (starting from 1)
        x = layers.Dense(layer_width, name=f"desc{d}")(x)
        x = apply_activation(x, activation)
        x = layers.LayerNormalization(dtype="float32")(x)
        x = layers.Dropout(dropoutRate)(x)

    # Add a single Dense layer of size D * maxtokens as the final output #NOT TokensOut
    token_output = layers.Dense(layer_width, activation='linear', name="descriptors", dtype=_out_dtype)(x)

    return token_output


#-------------------------------------------------------------------------------
def append_geolocation_head(bridge_layer, hidden=512, activation='leaky_relu', outHeight=64, outWidth=128, dropoutRate=0.3):
    """Dedicated global-pooled geolocation head (PLAN.md Experiment U).

    Predicts a softmax density over the equirectangular world grid from the
    bridge's GLOBAL features — giving geolocation its OWN gradient route from the
    bottleneck, so back-prop shapes the shared features for "where on Earth is
    this" (unlike the serial-278 shared-decoder channel, which prior-collapsed
    because the signal couldn't survive the input-aligned skip connections).

    Skip-free transpose-conv generator: it can only consume the pooled global
    code — there are no input-aligned skips for it to collapse into. GAP (not
    Flatten) keeps the param count sane. Output is a (outHeight,outWidth) proper
    probability distribution (softmax over the flattened grid).
    """
    _out_dtype = 'float32' if keras.mixed_precision.global_policy().compute_dtype != 'float32' else None
    g = layers.GlobalAveragePooling2D(name="geo_gap")(bridge_layer)         # (B, C) global scene code
    g = layers.Dense(hidden, name="geo_dense0")(g)                          # dedicated geo feature transform
    g = apply_activation(g, activation)
    g = layers.LayerNormalization(dtype="float32")(g)
    if dropoutRate > 0.0:
        g = layers.Dropout(dropoutRate, name="geo_drop")(g)                 # regularize (279 overfit hard)
    seedH, seedW, seedC = outHeight // 8, outWidth // 8, 32                 # 3 x2 upsamples -> native res
    g = layers.Dense(seedH * seedW * seedC, name="geo_seed")(g)
    g = apply_activation(g, activation)
    x = layers.Reshape((seedH, seedW, seedC))(g)                            # seed a small spatial map
    # Skip-free learned upsampling to the teacher's NATIVE grid (default 64x128):
    # 3 x2 steps from the seed. Native res (vs the old 256x256, an 8x-redundant
    # nearest-upsample) shrinks the output ~8x, cutting overfit capacity.
    for i, ch in enumerate([32, 16, 8]):
        x = layers.Conv2DTranspose(ch, 3, strides=2, padding='same', name="geo_up%u" % i)(x)
        x = layers.BatchNormalization(name="geo_bn%u" % i)(x)
        x = apply_activation(x, activation)
    x = layers.Conv2D(1, 1, padding='same', name="geo_logits")(x)          # (B,outHeight,outWidth,1) logits
    x = layers.Reshape((outHeight * outWidth, ))(x)
    x = layers.Softmax(name="geo_softmax", dtype=_out_dtype)(x)            # distribution over the world grid
    geo_output = layers.Reshape((outHeight, outWidth), name="geo_grid")(x)
    return geo_output


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def build_resnethybrid_cnn(
        input_shape,  #num_classes,
        activation='leaky_relu',
        gloveLayers=7,
        multihotLayers=2,
        multihotLayerWidth=0,
        bridgeLayerWidthCompatibility=0,
        dropoutRate=0.3,
        nextTokenStrength=0.3,
        numTokens=8,
        numClasses=2037,
        use_learnable_residuals=True,
        useDescriptors=False,
        tokenEmbeddingD=300):
    from keras import layers, models
    import tensorflow as tf

    # Load ResNet50 without the top layers (include_top=False)

    base_model = tf.keras.applications.ResNet50(
        #name="pretrained",
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'  # Global Average Pooling
    )
    """
    base_model = tf.keras.applications.ConvNeXtSmall( 
    #model_name="pretrained",
    include_top=False,
    include_preprocessing=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax')
    """

    # Freeze the ResNet50 base model
    base_model.trainable = False

    # Create a new model
    inputs = layers.Input(shape=input_shape)

    # Apply the ResNet base model
    # training=False pins BatchNormalization layers to their pretrained (ImageNet)
    # running statistics permanently, independent of base_model.trainable. Without
    # this, trainTokensOnly.py's later `model.get_layer('resnet50').trainable = True`
    # (unfreeze step) flips every BN layer from inference mode (frozen moving stats)
    # to training mode (live per-batch stats + running-average updates) -- a backbone
    # that has never seen gradient-based training since ImageNet pretraining suddenly
    # gets noisy per-batch normalization on top of newly-trainable conv weights, which
    # reliably exploded the loss a few epochs into the unfreeze phase (K4 focal/ASL
    # runs, TOKENS.md). Conv kernels and BN gamma/beta still get gradients once
    # trainable=True; only the normalization statistics stay frozen. See the Keras
    # transfer-learning guide's fine-tuning caveat.
    b = base_model(inputs, training=False)

    # Reshape only if the input is not already flattened
    #if len(b.shape) > 2:  # Check if the output needs reshaping
    #    b = layers.Reshape((2048,))(b) #The output of the resnet is 2048

    if len(b.shape) > 2:  # Check if the output needs reshaping
        b = layers.Reshape((7 * 7 * 768, ))(b)  #The output of the resnet is 2048
        #b = layers.Dense(2048, activation='gelu', name="scaledown")(b)

    if bridgeLayerWidthCompatibility != 0:
        print("Adding a compatibility layer (layer width=", bridgeLayerWidthCompatibility,
              ") after resnet to make it more compatible with Y-MAP ")
        dim_product = bridgeLayerWidthCompatibility
        b = layers.Dense(int(bridgeLayerWidthCompatibility), activation='linear', name="resnet_width_adapter")(b)

    if (useDescriptors):
        descriptor_outputs = append_descriptor_layer(bridge_layer=b, layer_width=768)

    glove_outputs = add_token_output(b, inputs, numTokens, activation=activation, dropoutRate=dropoutRate,
                                     D=tokenEmbeddingD,
                                     nextTokenStrength=nextTokenStrength, layer_width=dim_product, depth=gloveLayers,
                                     use_learnable_residuals=use_learnable_residuals)
    number_of_glove_output_tokens = len(glove_outputs)  #This should be the same as numTokens but make sure

    # bridge_output is set to None here intentionally.
    # Passing the bridge tensor (bridge_output=b) was empirically found to hurt
    # multihot performance: the spatial bridge has a large dim_product footprint
    # that forces a huge adapter Dense layer whose parameters add noise rather than
    # signal.  The GloVe outputs (8 × 300 = 2400-dim concat) already carry all the
    # semantic information needed for the 17 977-class multihot head.
    token_output = append_final_multihot_layer(
        glove_outputs, bridge_output=None, D=tokenEmbeddingD, TokensOut=number_of_glove_output_tokens, Classes=numClasses,
        layer_width=dim_product, multihot_layer_width=multihotLayerWidth, depth=multihotLayers)

    #-------------------------------
    modelOutputs = list()
    if (useDescriptors):
        modelOutputs.append(descriptor_outputs)
    modelOutputs.extend(glove_outputs)
    modelOutputs.append(token_output)
    #-------------------------------

    model = Model(inputs, modelOutputs, name="ResnetBasedModel")

    # Print model summary
    model.summary()

    return model


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package='NNModel')
class CoordConv2D(tf.keras.layers.Layer):
    """Appends normalized (x, y) coordinate channels to the feature map.
    Gives the model explicit spatial position information with zero trainable parameters.
    Liu et al. 2018 - https://arxiv.org/abs/1807.03247
    """

    def call(self, x):
        shape = tf.shape(x)
        h, w = shape[1], shape[2]
        xs = tf.cast(tf.linspace(-1.0, 1.0, w), x.dtype)
        ys = tf.cast(tf.linspace(-1.0, 1.0, h), x.dtype)
        x_grid, y_grid = tf.meshgrid(xs, ys)  # [H, W] each
        ones = tf.ones_like(x[:, :, :, :1])
        x_grid = x_grid[None, :, :, None] * ones  # [B, H, W, 1]
        y_grid = y_grid[None, :, :, None] * ones  # [B, H, W, 1]
        return tf.concat([x, x_grid, y_grid], axis=-1)  # [B, H, W, C+2]

    def get_config(self):
        return super().get_config()


#-------------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package='NNModel')
class StopGradient(tf.keras.layers.Layer):
    """Identity forward, zero gradient backward (used by multihotInputSkip)."""

    def call(self, x):
        return tf.stop_gradient(x)


#-------------------------------------------------------------------------------
class AddGaussianNoise(tf.keras.layers.Layer):

    def __init__(self, stddev, **kwargs):
        super(AddGaussianNoise, self).__init__()
        self.stddev = stddev

    def call(self, inputs, training=False):
        if training:
            noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs + noise
        return inputs


#-------------------------------------------------------------------------------
def apply_activation(x, activation, name=None):
    """Route 'prelu' to a PReLU layer (learnable slope per channel); all other
    activations go through the standard Activation layer via their string name.

    For 4-D feature maps (conv), PReLU shares the slope across spatial dims but
    keeps one slope per channel (shared_axes=[1,2]).  For 2-D tensors (Dense),
    each unit gets its own slope (shared_axes=None).
    """
    if activation == 'prelu':
        shared_axes = [1, 2] if len(x.shape) == 4 else None
        layer = keras.layers.PReLU(shared_axes=shared_axes, name=name) if name else keras.layers.PReLU(shared_axes=shared_axes)
        return layer(x)
    return Activation(activation, name=name)(x) if name else Activation(activation)(x)


#-------------------------------------------------------------------------------
def convnext_block(input, name, num_filters, activation='gelu', dropoutRate=0.0):
    shortcut = input

    # Depthwise convolution (like in ConvNeXt)
    x = DepthwiseConv2D(kernel_size=7, padding='same', name=f"dwconv_{name}_{num_filters}")(input)
    x = layers.LayerNormalization(dtype="float32")(x)

    # Pointwise convolution (1x1) for feature transformation
    x = Conv2D(4 * num_filters, kernel_size=1, padding='same', name=f"pwconv1_{name}_{num_filters}")(x)
    x = apply_activation(x, activation)

    # Another 1x1 convolution to bring it back to original num_filters
    x = Conv2D(num_filters, kernel_size=1, padding='same', name=f"pwconv2_{name}_{num_filters}")(x)

    # Residual connection: project shortcut to num_filters if channel count differs
    if shortcut.shape[-1] != num_filters:
        shortcut = Conv2D(num_filters, kernel_size=1, padding='same',
                          name=f"shortcut_proj_{name}_{num_filters}")(shortcut)
    x = Add(name=f"residual_{name}_{num_filters}")([shortcut, x])

    # Optional dropout
    if dropoutRate > 0.0:
        x = Dropout(dropoutRate, name=f"dropout_{name}_{num_filters}")(x)

    return x


#-------------------------------------------------------------------------------
def conv_block(input, name, num_filters, activation='leaky_relu', dropoutRate=0.0, repetitions=2):
    # Save input for residual connection
    shortcut = input

    x = input
    for r in range(repetitions):
        x = Conv2D(num_filters, 3, padding="same", name="conv2D_%s_%u_%s_r%u" % (name, num_filters, activation, r))(x)
        x = BatchNormalization()(x)
        x = apply_activation(x, activation)

    # Optionally add dropout (applied once after all repetitions, before the residual)
    if dropoutRate > 0.0:
        x = SpatialDropout2D(dropoutRate, name="dropout_%s_%u_%u" % (name, num_filters, int(dropoutRate * 100)))(x)

    # Residual connection: if the input and output dimensions match, add them
    if shortcut.shape[-1] == num_filters:
        x = Add(name="residual_In_%s_%u_%s" % (name, num_filters, activation))([shortcut, x])
    else:
        # If dimensions do not match, apply a 1x1 convolution to match them
        shortcut = Conv2D(num_filters, kernel_size=1, padding='same')(shortcut)
        x = Add(name="residual_In_%s_%u_%s" % (name, num_filters, activation))([shortcut, x])

    return x


#-------------------------------------------------------------------------------
def feature_conv(x, name, num_filters, activation, dropoutRate, repetitions, useDepthwiseSeparable):
    """Encoder/decoder feature path. Default: the full-conv `conv_block` (2×3×3
    Conv2D per repetition). Experiment R1 (useDepthwiseSeparable=True): a stack of
    `repetitions` depthwise-separable `convnext_block`s (DW 7×7 → PW 1×1 4× → PW 1×1),
    ~8C² FLOPs/block vs ~18C² for a 2-conv full block. Dropout mirrors conv_block
    (applied once, after the last block)."""
    if not useDepthwiseSeparable:
        return conv_block(x, name, num_filters, activation=activation,
                          dropoutRate=dropoutRate, repetitions=repetitions)
    for r in range(repetitions):
        x = convnext_block(x, "%s_r%u" % (name, r), num_filters, activation=activation,
                           dropoutRate=dropoutRate if r == repetitions - 1 else 0.0)
    return x


#-------------------------------------------------------------------------------
def encoder_block(input, num_filters, activation, dropoutRate=0.0, depth=None, conv_repetitions=2,
                  useDepthwiseSeparable=False):
    if depth == None:
        name = "encoder"
    else:
        name = "encoder_d%u" % depth

    x = feature_conv(input, name, num_filters, activation, dropoutRate, conv_repetitions,
                     useDepthwiseSeparable)
    p = AveragePooling2D((2, 2))(x)  # Use AveragePooling2D
    return x, p


#-------------------------------------------------------------------------------
def decoder_block(input, skip_features, num_filters, activation, namePrefix="decoder", level=0, conv_repetitions=2,
                  upsampling_mode='deconv', useDepthwiseSeparable=False):
    # Include level in every name so clamped filter counts (numKeypoints floor)
    # on multiple levels don't produce duplicate layer names.
    tag = "%s_L%u_%u" % (namePrefix, level, num_filters)

    # ── Upsampling stage ──────────────────────────────────────────────────────
    # Three modes, selected via the config key "deconCheckerboardSmoothing"
    # (passed through build_unet → upsampling_mode here):
    #
    #   'deconv'   (default / empty string in config)
    #              Conv2DTranspose(2×2, stride=2) — the historical path.
    #              kernel=2 divides stride=2 evenly, placing it in the "safe
    #              zone" of the Odena et al. divisibility rule, but training
    #              still steers learned kernels toward the 2×2 tiling pattern
    #              over time, gradually re-introducing checkerboard artifacts.
    #              All existing checkpoints load under this mode because the
    #              layer name ("conv2DT_*") is unchanged.
    #
    #   'nearest'  UpSampling2D(nearest) → Conv2D(3×3) → BN → Activation.
    #              The paper's primary recommendation: nearest-neighbour
    #              interpolation carries no learnable weights so it cannot
    #              develop tiling bias.  The subsequent 3×3 conv (wider than
    #              the original 2×2 transposed kernel) integrates neighbour-
    #              hood context before skip-feature concatenation.  The paper
    #              found nearest outperformed bilinear on image generation tasks.
    #              NOTE: incompatible with 'deconv' checkpoints — layer names
    #              change from "conv2DT_*" to "upsample_*"/"conv_up_*".
    #
    #   'bilinear' Same pipeline but uses bilinear interpolation.  Produces
    #              smoother pre-conv features at the cost of a small blurring
    #              prior; paper found it marginally weaker than nearest.
    #
    # Reference: Odena et al., "Deconvolution and Checkerboard Artifacts"
    #            https://distill.pub/2016/deconv-checkerboard/
    if upsampling_mode == 'deconv':
        # Original transposed-convolution path — preserved unchanged so that
        # existing best.weights.h5 checkpoints load without layer remapping.
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same", name="conv2DT_%s" % tag)(input)
    else:
        # Resize-then-convolve path ('nearest' or 'bilinear').
        # Step 1: parameter-free spatial doubling — no weights means no learned
        #         tiling pattern can form during backprop.
        x = UpSampling2D(size=(2, 2), interpolation=upsampling_mode, name="upsample_%s" % tag)(input)
        # Step 2: 3×3 conv gives the network a larger receptive field than the
        #         2×2 transposed kernel had, smoothing the interpolated feature
        #         map before it meets the encoder skip connection.
        x = Conv2D(num_filters, 3, padding='same', name="conv_up_%s" % tag)(x)
        # BN + Activation mirror the Conv→BN→Act order used in conv_block and
        # bridge_block throughout this file, keeping gradient statistics
        # consistent across all feature-processing stages.
        x = BatchNormalization(name="bn_up_%s" % tag)(x)
        x = apply_activation(x, activation, name="act_up_%s" % tag)

    # Check if spatial dimensions mismatch
    if (skip_features.shape[1] != x.shape[1]) or (skip_features.shape[2] != x.shape[2]):
        print(f"Incompatible skip {skip_features.shape[1]}, {skip_features.shape[2]} with {x.shape[1]}, {x.shape[2]}")
        crop_height = skip_features.shape[1] - x.shape[1]
        crop_width = skip_features.shape[2] - x.shape[2]

        # Crop skip_features if necessary
        if crop_height != 0 or crop_width != 0:
            skip_features = Cropping2D(cropping=((crop_height // 2, crop_height - (crop_height // 2)),
                                                 (crop_width // 2, crop_width - (crop_width // 2))))(skip_features)

    # Concatenate upsampled input with cropped skip features
    x_concat = Concatenate(name="concat_%s" % tag)([x, skip_features])

    # Pass through the convolution block after concatenation
    x_conv = feature_conv(x_concat, tag, num_filters, activation, 0.0, conv_repetitions,
                          useDepthwiseSeparable)

    # Add residual connection (input `x` before conv_block) back to the output of the conv_block
    x_residual = Add(name="residual_%s" % tag)([x, x_conv])

    return x_residual


#-------------------------------------------------------------------------------
def bottleneck_attention_block(x, num_heads=8, name='bottleneck_attn'):
    """
    Spatial self-attention applied at the UNet bottleneck (after the bridge block).

    At 7 encoder stages on a 256×256 input, the bridge operates at 2×2 spatial
    resolution — only 4 spatial tokens. MHA is essentially free here but gives
    the model global relational reasoning that convolutions cannot: e.g. left
    wrist <-> right shoulder interdependencies for pose estimation.

    Uses pre-norm convention (LayerNorm before attention) for training stability.
    Compatible with ASPP bridge, XLA, and mixed precision.
    """
    static_h = int(x.shape[1])
    static_w = int(x.shape[2])
    static_c = int(x.shape[3])
    seq_len = static_h * static_w
    key_dim = max(8, static_c // num_heads)

    # [B, H, W, C] → [B, H*W, C]
    seq = Reshape((seq_len, static_c), name=f'{name}_flatten')(x)

    # Pre-norm → self-attention → residual
    seq_norm = keras.layers.LayerNormalization(name=f'{name}_ln', dtype="float32")(seq)
    attn_out = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f'{name}_mha')(seq_norm,
                                                                                                         seq_norm)
    seq = Add(name=f'{name}_residual')([seq, attn_out])

    # [B, H*W, C] → [B, H, W, C]
    return Reshape((static_h, static_w, static_c), name=f'{name}_unflatten')(seq)


#-------------------------------------------------------------------------------
#A more powerful block in the bridge, such as dilated convolutions or even an atrous spatial pyramid pooling (ASPP) module, which can capture multi-scale context better.
def bridge_block(input, num_filters, activation, layers=3):
    # Conv → BN → Activation order matches conv_block (activation must come after BN)
    x = Conv2D(num_filters, 3, padding="same", dilation_rate=1, name="BridgeStart")(input)
    x = BatchNormalization()(x)
    x = apply_activation(x, activation)

    if (layers >= 2):
        x = Conv2D(num_filters, 3, padding="same", dilation_rate=2, name="BridgeMid")(x)
        x = BatchNormalization()(x)
        x = apply_activation(x, activation)

    if (layers >= 3):
        x = Conv2D(num_filters, 3, padding="same", dilation_rate=4, name="BridgeFinish")(x)
        x = BatchNormalization()(x)
        x = apply_activation(x, activation)

    return x


#-------------------------------------------------------------------------------
def aspp_bridge_block(input, num_filters, activation, num_layers=3):
    """
    ASPP-style bridge that replaces the simple dilated-conv bridge.

    Parallel branches capture multi-scale context at the bottleneck, plus a
    global average-pooling branch that broadcasts a scene-level context vector
    back to the spatial map.  This is particularly beneficial for monocular
    depth and surface-normal estimation, which require understanding global
    scene layout that the encoder skip-connections alone cannot provide.

    num_layers controls how many dilated branches are included (min 2):
      2 → 1×1 + 3×3 dil=1 + global
      3 → + 3×3 dil=2       (default)
      4 → + 3×3 dil=4
    All branches output num_filters channels, are concatenated, then projected
    back to num_filters with a residual connection from the input.
    """
    spatial_h = int(input.shape[1]) if input.shape[1] is not None else 2
    spatial_w = int(input.shape[2]) if input.shape[2] is not None else 2

    # Branch 1: 1×1 — dense channel projection
    b1 = Conv2D(num_filters, 1, padding='same', name='aspp_b1')(input)
    b1 = BatchNormalization()(b1)
    b1 = apply_activation(b1, activation)

    # Branch 2: 3×3 standard conv
    b2 = Conv2D(num_filters, 3, padding='same', dilation_rate=1, name='aspp_b2')(input)
    b2 = BatchNormalization()(b2)
    b2 = apply_activation(b2, activation)

    branches = [b1, b2]

    # Branch 3: 3×3 dil=2
    if num_layers >= 3:
        b3 = Conv2D(num_filters, 3, padding='same', dilation_rate=2, name='aspp_b3')(input)
        b3 = BatchNormalization()(b3)
        b3 = apply_activation(b3, activation)
        branches.append(b3)

    # Branch 4: 3×3 dil=4
    if num_layers >= 4:
        b4 = Conv2D(num_filters, 3, padding='same', dilation_rate=4, name='aspp_b4')(input)
        b4 = BatchNormalization()(b4)
        b4 = apply_activation(b4, activation)
        branches.append(b4)

    # Global context branch: pool to 1×1 → conv → bilinear upsample back
    # AveragePooling2D with spatial pool_size collapses to 1×1 when pool_size == input spatial dims.
    b_global = AveragePooling2D(pool_size=(spatial_h, spatial_w), name='aspp_gap')(input)
    b_global = Conv2D(num_filters, 1, padding='same', name='aspp_global_conv')(b_global)
    b_global = BatchNormalization()(b_global)
    b_global = apply_activation(b_global, activation)
    b_global = BilinearUpsampling2D(size=(spatial_h, spatial_w), name='aspp_global_up')(b_global)
    branches.append(b_global)

    # Concatenate all branches and project back to num_filters
    x = Concatenate(name='aspp_concat')(branches)
    x = Conv2D(num_filters, 1, padding='same', name='aspp_project')(x)
    x = BatchNormalization()(x)
    x = apply_activation(x, activation)

    # Residual: project input to num_filters if channel count differs
    if input.shape[-1] != num_filters:
        shortcut = Conv2D(num_filters, 1, padding='same', name='aspp_shortcut')(input)
    else:
        shortcut = input
    x = Add(name='aspp_residual')([x, shortcut])

    return x


#-------------------------------------------------------------------------------
def build_unet(inputHeight, inputWidth, inputChannels, outputWidth, outputHeight, numKeypoints, num16BitHeatmaps=1,
               numTokens=0, numClasses=2037, minHeatmapValue=-120, maxHeatmapValue=120, encoderGrowthBase=1.4,
               decoderGrowthBase=2.0, baseChannels=64, pixelwiseChannels=0, encoderRepetitions=7, decoderRepetitions=7,
               midSectionRepetitions=3, gloveLayers=7, multihotLayers=2, multihotLayerWidth=0, tokenLayerWidth=0,
               use_learnable_residuals=True, bridgeRatio=1.0, forceBridgeSize=0, activation='leaky_relu',
               gaussianNoiseSTD=0.0, dropoutRate=0.0, nextTokenStrength=0.5, quantize=False, serial="",
               useDescriptors=False, depth_channel_start=-1, depth_channel_end=-1, useASPPBridge=False,
               useDepthRefinementHead=False, useCoordConv=False, useBottleneckAttention=False, convBlockRepetitions=2,
               deconCheckerboardSmoothing='', useDepthNormalsUncertainty=False, maxDecoderChannels=0,
               tanhHeadroom8bit=1.0, tanhHeadroom16bit=1.0, useGeolocationHead=False, geoHeadHidden=512,
               geoGridHeight=64, geoGridWidth=128, useDepthwiseSeparable=False,
               tokenEmbeddingD=300, multihotInputSkip=""):

    #Uncomment to go "single
    ###if (num16BitHeatmaps>0):
    #      DEACTIVATED 16BIT without changing 8bit
    #      numKeypoints = numKeypoints + num16BitHeatmaps
    ###       num16BitHeatmaps  = 0

    input_shape = (inputWidth, inputHeight, inputChannels)
    output_shape = (outputWidth, outputHeight, numKeypoints)

    inputs = Input(input_shape, name="input")
    after_input = keras.layers.Rescaling(scale=1.0 / 255.0, name="float_scaling")(inputs)

    #Fixed : https://github.com/keras-team/keras/issues/19589 ?
    if (gaussianNoiseSTD > 0.0):
        after_input = AddGaussianNoise(gaussianNoiseSTD, name='training_gaussian_noise')(after_input)

    if useCoordConv:
        after_input = CoordConv2D(name='coord_conv')(after_input)

    # Encoder blocks
    encoder_layers = []
    p = after_input
    for i in range(encoderRepetitions):
        thisLayerFilters = int(baseChannels * (encoderGrowthBase**i))
        if (thisLayerFilters < numKeypoints):
            thisLayerFilters = numKeypoints  #Always have at least the number of outputs to not strangle them
        s, p = encoder_block(p, thisLayerFilters, activation, dropoutRate=dropoutRate, depth=i,
                             conv_repetitions=convBlockRepetitions,
                             useDepthwiseSeparable=useDepthwiseSeparable)
        encoder_layers.append(s)

    # UNET Bridge layer
    if (forceBridgeSize == 0):
        forceBridgeSize = int(bridgeRatio * baseChannels * (encoderGrowthBase**encoderRepetitions))
    if useASPPBridge:
        # ASPP-style bridge: parallel multi-scale branches + global context pooling.
        # Better global scene understanding for depth and normals prediction.
        b = aspp_bridge_block(p, forceBridgeSize, activation=activation, num_layers=midSectionRepetitions)
    else:
        b = bridge_block(p, forceBridgeSize, activation=activation, layers=midSectionRepetitions)

    if useBottleneckAttention:
        b = bottleneck_attention_block(b, num_heads=8)

    if (useDescriptors):
        descriptor_outputs = append_descriptor_layer(bridge_layer=b, layer_width=768)

    # Geolocation head (Experiment U): its own route from the bridge `b` so back-prop
    # shapes the global features for geolocation. Tapped BEFORE the decoder — it never
    # touches the input-aligned decoder/skip path that collapsed the 278 shared channel.
    if useGeolocationHead:
        geolocation_output = append_geolocation_head(b, hidden=geoHeadHidden, activation=activation,
                                                     outHeight=geoGridHeight, outWidth=geoGridWidth)

    # Decoder blocks — intentionally use base-2 growth (wider than encoder) for richer reconstruction.
    #
    # deconCheckerboardSmoothing controls the upsampling strategy inside every
    # decoder_block (see that function's docstring for the full rationale):
    #   ''         → 'deconv' (Conv2DTranspose, default, checkpoint-compatible)
    #   'nearest'  → nearest-neighbour upsample + 3×3 Conv2D (paper recommendation)
    #   'bilinear' → bilinear upsample + 3×3 Conv2D
    # The empty-string-to-'deconv' mapping keeps the config value human-readable
    # ("not set" vs an explicit mode name) while keeping the internal code clean.
    _upsampling_mode = 'deconv'
    if deconCheckerboardSmoothing!="":
        _upsampling_mode = deconCheckerboardSmoothing


    decoder_layers = [b]
    for i in range(decoderRepetitions):
        thisLayerFilters = int(baseChannels * (decoderGrowthBase**(encoderRepetitions - i - 1)))
        # maxDecoderChannels (Experiment O, PLAN.md Obs 20): cap the deep-decoder width.
        # With growthBase 1.8 the unclamped first level is 1632 channels at 4x4 spatial
        # resolution (~53M params); a 512 clamp frees ~55M params with the high-res
        # levels untouched. 0 = unclamped (legacy layout, checkpoint-compatible).
        if maxDecoderChannels > 0 and thisLayerFilters > maxDecoderChannels:
            thisLayerFilters = maxDecoderChannels
        if (thisLayerFilters < numKeypoints):
            thisLayerFilters = numKeypoints  #Always have at least the number of outputs to not strangle them
        d = decoder_block(decoder_layers[-1], encoder_layers[-(i + 1)], thisLayerFilters, activation, level=i,
                          conv_repetitions=convBlockRepetitions, upsampling_mode=_upsampling_mode,
                          useDepthwiseSeparable=useDepthwiseSeparable)
        decoder_layers.append(d)

    #Final layer calculations
    print("Out layer ", decoder_layers[-1].shape)
    UNETDimW = decoder_layers[-1].shape[1]
    UNETDimH = decoder_layers[-1].shape[2]
    print("Out layer dims ", UNETDimW, ",", UNETDimH)
    print("Requested final layer ", outputWidth, ",", outputHeight)
    stride_amount = 2
    if (outputHeight >= UNETDimH) or (outputWidth >= UNETDimW):
        stride_amount = 1
        print("Force stride=(1,1) to keep network size (Rule 1)")
    if (outputHeight - UNETDimH < 0) or (outputWidth - UNETDimW < 0):
        stride_amount = 1
        print("Force stride=(1,1) to keep network size (Rule 2)")

    #Heatmap output size and magnitude control
    #----------------------------------------------------------------------
    #Using tanh to force output in [-1..1] avoid over/underflow of data

    # Adding the pixelwise convolution (1x1 convolution) as the last layer
    if (pixelwiseChannels != 0):
        pixelwise = Conv2D(pixelwiseChannels, kernel_size=(1, 1), name="pixelwise")(decoder_layers[-1])
        pixelwise = apply_activation(pixelwise, activation)
    else:
        pixelwise = decoder_layers[-1]

    # Depth / Normals dedicated refinement head (optional)
    # -------------------------------------------------------
    # Two extra 3×3 dilated convolutions produce refined features specifically
    # for the depth and normals channels before the final 1×1 projection.
    # The output is split into three Conv2D projections (pre-depth, depth,
    # post-depth) which are concatenated back to numKeypoints channels.
    # This avoids tensor slicing/Lambda layers and remains fully serialisable.
    depth_ref_enabled = (useDepthRefinementHead and depth_channel_start >= 0 and depth_channel_end > depth_channel_start
                         and depth_channel_end <= numKeypoints)
    if depth_ref_enabled:
        num_depth_ch = depth_channel_end - depth_channel_start
        num_pre_ch = depth_channel_start
        num_post_ch = numKeypoints - depth_channel_end
        # Keep drh_channels small: the 3x3 dilated convs run at full output
        # resolution (e.g. 256x256), so channel count dominates FLOP cost.
        # A 1x1 bottleneck entry reduces pixelwiseChannels → drh_channels first
        # so the expensive 3x3 dilated kernels never see the fat feature map.
        # pixelwiseChannels=512 → drh_channels=64 cuts DRH cost ~16x vs the old
        # max(128, pixelwiseChannels//2)=256 design without the bottleneck.
        drh_channels = max(64, int(pixelwiseChannels // 8) if pixelwiseChannels > 0 else 64)

        # 1x1 bottleneck: collapse pixelwiseChannels → drh_channels before 3x3 dilated convs
        drh = Conv2D(drh_channels, 1, padding='same', name='drh_entry')(pixelwise)
        drh = BatchNormalization(name='drh_entry_bn')(drh)
        drh = apply_activation(drh, activation, name='drh_entry_act')

        drh = Conv2D(drh_channels, 3, padding='same', dilation_rate=2, name='drh_conv1')(drh)
        drh = BatchNormalization(name='drh_bn1')(drh)
        drh = apply_activation(drh, activation, name='drh_act1')
        drh = Conv2D(drh_channels, 3, padding='same', dilation_rate=4, name='drh_conv2')(drh)
        drh = BatchNormalization(name='drh_bn2')(drh)
        drh = apply_activation(drh, activation, name='drh_act2')

        out_pre = Conv2D(num_pre_ch, 1, padding='same', activation='tanh', name='hm_tanh_pre')(pixelwise)
        out_depth = Conv2D(num_depth_ch, 1, padding='same', activation='tanh', name='hm_tanh_depth')(drh)
        out_post = Conv2D(num_post_ch, 1, padding='same', activation='tanh', name='hm_tanh_post')(pixelwise)
        heatmap_output = Concatenate(axis=-1, name='hm_tanh_8bit')([out_pre, out_depth, out_post])
        print(f"Depth refinement head: bottleneck {pixelwiseChannels if pixelwiseChannels > 0 else '?'}→{drh_channels} "
              f"(3x3 dil=2, 3x3 dil=4) @ full res, depth ch {depth_channel_start}–{depth_channel_end-1}")
    else:
        heatmap_output = Conv2D(filters=numKeypoints, kernel_size=1, padding="same", activation="tanh",
                                name="hm_tanh_8bit")(pixelwise)
    # Maintain the desired output dimensions
    if (outputHeight != heatmap_output.shape[1]) or (outputWidth != heatmap_output.shape[2]):
        print("Incompatible final skip ", heatmap_output.shape[1], ",", heatmap_output.shape[2])
        print(" with ", outputHeight, ",", outputWidth)
        crop_height = heatmap_output.shape[1] - outputHeight
        crop_width = heatmap_output.shape[2] - outputWidth
        if (crop_height < 0) or (crop_width < 0):
            print("Network has smaller output than what was requested..")
        else:
            heatmap_output = Cropping2D(
                cropping=((crop_height // 2, crop_height - (crop_height // 2)),
                          (crop_width // 2, crop_width - (crop_width // 2))), name="final_crop")(heatmap_output)
    # Depth+Normals aleatoric uncertainty head (optional)
    # hm_nll output: [B, H, W, 2*C] where first C channels = mean (tanh space),
    # last C channels = log-variance (unconstrained). Supervised by DepthNormalsNLLLoss.
    hm_nll = None
    _unc_enabled = (useDepthNormalsUncertainty and depth_channel_start >= 0
                    and depth_channel_end > depth_channel_start
                    and depth_channel_end <= numKeypoints)
    if _unc_enabled:
        _num_unc = depth_channel_end - depth_channel_start
        # Tap heatmap_output in tanh space [-1,1] (before Rescaling below)
        hm_dn_tanh   = heatmap_output[..., depth_channel_start:depth_channel_end]
        hm_unc_logvar = Conv2D(_num_unc, 1, padding='same', name='hm_unc')(pixelwise)
        hm_nll = Concatenate(name='hm_nll')([hm_dn_tanh, hm_unc_logvar])
        print(f"Depth+Normals uncertainty head: {_num_unc} channels "
              f"(depth+normals ch {depth_channel_start}–{depth_channel_end - 1})")

    #Scale back to min/max Heatmap values ( [-120..120] )
    # tanhHeadroom8bit>1 lifts the 8-bit output ceiling above the GT active value so the fixed
    # target (heatmapActive) lands at tanh=1/headroom instead of the unreachable tanh=1.0
    # asymptote — the GT/int8 encoding and the loss are unchanged (both operate on the raw
    # output), only the final Rescaling widens. headroom=1.0 == historical behaviour.
    scale_factor = max(abs(minHeatmapValue), abs(maxHeatmapValue)) * float(tanhHeadroom8bit)
    # Only force float32 under mixed precision; when global policy is already
    # float32, dtype=None avoids inserting spurious cast nodes in the graph.
    _out_dtype = 'float32' if keras.mixed_precision.global_policy().compute_dtype != 'float32' else None
    heatmap_output = keras.layers.Rescaling(scale=scale_factor, name="hm", dtype=_out_dtype)(heatmap_output)
    print("Final Scale Factor is ", scale_factor)

    # 16Bit output (if it is enabled..)
    #----------------------------------------------------------------------
    heatmaps_16bit_declared = False
    heatmaps_16bit = None
    if (num16BitHeatmaps > 0):
        heatmaps_16bit_declared = True
        heatmaps_16bit = Conv2D(filters=num16BitHeatmaps, kernel_size=1, padding="same", activation="tanh",
                                name=f"hm_tanh_16bit", strides=(stride_amount, stride_amount))(decoder_layers[-1])
        if (outputHeight != heatmaps_16bit.shape[1]) or (outputWidth != heatmaps_16bit.shape[2]):
            crop_height = heatmaps_16bit.shape[1] - outputHeight
            crop_width = heatmaps_16bit.shape[2] - outputWidth
            if (crop_height >= 0) and (crop_width >= 0):
                heatmaps_16bit = Cropping2D(
                    cropping=((crop_height // 2, crop_height - (crop_height // 2)), (crop_width // 2,
                                                                                     crop_width - (crop_width // 2))),
                    name=f"final_crop_16bit")(heatmaps_16bit)
        # Same tanh-headroom logic as the 8-bit head, but for the 16-bit (depth) output:
        # tanhHeadroom16bit>1 lifts the 32767 ceiling so far-depth values (GT near ±32767)
        # are reachable off the tanh=1.0 asymptote. headroom=1.0 == historical behaviour.
        scale_factor_16b = 32767 * float(tanhHeadroom16bit)  #/ scale_factor # This should be something around ~273
        heatmaps_16bit = keras.layers.Rescaling(scale=scale_factor_16b, name="hm_16b", dtype='float32')(heatmaps_16bit)
        print("Final 16-bit Scale Factor is ", scale_factor_16b)  #32767.0

    # Token output (if it is enabled..)
    #----------------------------------------------------------------------
    modelOutputs = []
    if (numTokens > 0):
        #numTokens = 1 #TEST tokens_to_classes
        #numTokens = 8 #TEST tokens_to_glove

        #This works until Keras 3.4.0 and Tensorflow 2.17.0
        #modelOutputs = add_token_output(b, inputs, numTokens, activation=activation, dropoutRate=dropoutRate)
        #token_output = append_final_multihot_layer(modelOutputs, D=300, TokensOut=numTokens ,Classes=2037)
        #modelOutputs.append(token_output)

        # Get shape of tensor b
        b_shape = b.shape  # Returns a tuple-like object
        # Compute the product of spatial and channel dimensions
        dim_product = 1
        for dim in b_shape[1:]:  # Ignore batch size (b_shape[0] is None)
            if dim is not None:
                dim_product *= dim
        print("Bridge dimension is ", dim_product,
              ", FYI this will be used as the layer width for glove/token outputs!")

        modelOutputs = list()
        glove_outputs = add_token_output(b, inputs, numTokens, activation=activation, dropoutRate=dropoutRate,
                                         D=tokenEmbeddingD,
                                         nextTokenStrength=nextTokenStrength, layer_width=int(dim_product),
                                         depth=gloveLayers, use_learnable_residuals=use_learnable_residuals,
                                         token_layer_width=int(tokenLayerWidth))
        number_of_glove_output_tokens = len(glove_outputs)  #This should be the same as numTokens but make sure
        # bridge_output=None: bridge skip disabled — see comment at the equivalent call
        # site in build_resnethybrid_cnn for the full rationale. That rationale is about the
        # Add-merge path and a 37632-wide ResNet bridge; multihotInputSkip is a concat instead.
        multihot_skip = []
        if multihotInputSkip in ("bridge", "bridge+descriptors"):
            multihot_skip.append(b)
        if multihotInputSkip == "bridge+descriptors":
            if not useDescriptors:
                raise ValueError('multihotInputSkip "bridge+descriptors" requires outputDescriptors')
            multihot_skip.append(descriptor_outputs)
        elif multihotInputSkip not in ("", "bridge"):
            raise ValueError('multihotInputSkip must be "", "bridge" or "bridge+descriptors", got %r' %
                             multihotInputSkip)
        token_output = append_final_multihot_layer(
            glove_outputs, bridge_output=None, D=tokenEmbeddingD, TokensOut=number_of_glove_output_tokens, Classes=numClasses,
            layer_width=int(dim_product), multihot_layer_width=multihotLayerWidth, depth=multihotLayers,
            skip_inputs=multihot_skip)

        if (heatmaps_16bit_declared):
            modelOutputs.append(heatmap_output)
            modelOutputs.append(heatmaps_16bit)
            if (useDescriptors):
                modelOutputs.append(descriptor_outputs)
            if useGeolocationHead:
                modelOutputs.append(geolocation_output)
            modelOutputs.extend(glove_outputs)
            modelOutputs.append(token_output)
            if hm_nll is not None:
                modelOutputs.append(hm_nll)
            model = Model(inputs, modelOutputs, name="Y-Net-v%s" % serial)
        else:
            modelOutputs.append(heatmap_output)
            if (useDescriptors):
                modelOutputs.append(descriptor_outputs)
            if useGeolocationHead:
                modelOutputs.append(geolocation_output)
            modelOutputs.extend(glove_outputs)
            modelOutputs.append(token_output)
            if hm_nll is not None:
                modelOutputs.append(hm_nll)
            model = Model(inputs, modelOutputs, name="Y-Net-v%s" % serial)
    else:
        if (heatmaps_16bit_declared):
            modelOutputs.append(heatmap_output)
            modelOutputs.append(heatmaps_16bit)
            if hm_nll is not None:
                modelOutputs.append(hm_nll)
            model = Model(inputs, modelOutputs, name="U-Net-v%s" % serial)
        else:
            if hm_nll is not None:
                modelOutputs.append(heatmap_output)
                modelOutputs.append(hm_nll)
                model = Model(inputs, modelOutputs, name="U-Net-v%s" % serial)
            else:
                model = Model(inputs, heatmap_output, name="U-Net-v%s" % serial)

    # Quantization aware model ( if it is enabled..)
    #----------------------------------------------------------------------
    if quantize:
        try:
            import tensorflow_model_optimization as tfmot
            model = tfmot.quantization.keras.quantize_model(model)
            print("Using quantized model..!")
        except Exception as e:
            print("An exception occurred:", str(e))
            print("Could not quantize model..!")
            print(
                "https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize_aware_activation.py"
            )

    model.summary()
    return model


#============================================================================================
#============================================================================================
# Main Function
#============================================================================================
#============================================================================================
if __name__ == '__main__':
    inputWidth = 210
    inputHeight = 210
    inputChannels = 3
    print("Image size %ux%u:%u" % (inputWidth, inputHeight, inputChannels))

    outputWidth = 96
    outputHeight = 96
    heatmaps = 19
    print("Heatmap size %ux%u:%u" % (outputWidth, outputHeight, heatmaps))

    #MB     #KB   # Bytes
    GPU = 49140 * 1024 * 1024
    sampleSize = (inputWidth * inputHeight * inputChannels) + (outputWidth * outputHeight * heatmaps)

    COCO17trainval = 123287
    COCO17trainvalOnlyHuman = 65000

    print("GPU size ", GPU, " bytes")
    print("Sample size ", sampleSize, " bytes")
    print("Sample Number that fits : ", int(GPU / sampleSize), " ")
    print("COCO17 train+val takes : %0.2f %% of an A6000" % (100 * float(COCO17trainval * sampleSize / GPU)))
    print("COCO17 train+val (only humans) takes : %0.2f %% an A6000" %
          (100 * float(COCO17trainvalOnlyHuman * sampleSize / GPU)))

    GPU = 16376 * 1024 * 1024
    print("COCO17 train+val takes : %0.2f %% of an RTX4080" % (100 * float(COCO17trainval * sampleSize / GPU)))
    print("COCO17 train+val (only humans) takes : %0.2f %% an RTX4080" %
          (100 * float(COCO17trainvalOnlyHuman * sampleSize / GPU)))

    GPU = 8192 * 1024 * 1024
    print("COCO17 train+val takes : %0.2f %% of an GTX1070" % (100 * float(COCO17trainval * sampleSize / GPU)))
    print("COCO17 train+val (only humans) takes : %0.2f %% an GTX1070" %
          (100 * float(COCO17trainvalOnlyHuman * sampleSize / GPU)))
