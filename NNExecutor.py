"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""
import os
import numpy as np
import time


#-----------------------------------------------------------------------------------------------------
class TFLiteExecutor():
    #tflite_convert --saved_model_dir=2d_pose_estimation --output_file=2d_pose_estimation/model.tflite
    def __init__(
        self,
        modelPath: str = "2d_pose_estimation/model_fp16.tflite",
        inputWidth=220,
        inputHeight=220,
        targetWidth=96,
        targetHeight=96,
        outputChannels=18,
        numberOfThreads=4,
    ):
        print("Using TF-Lite Runtime")
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=modelPath, num_threads=numberOfThreads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        #------------------------------------------
        self.input_size = (inputWidth, inputHeight)
        self.output_size = (targetWidth, targetHeight)
        self.numberOfHeatmaps = outputChannels
        self.heatmaps = None
        self.heatmaps_16b = None
        self.description = None
        self.multihot_description = None
        self.activity = None
        #------------------------------------------
#-----------------------------------------------------------------------------------------------------

    def _run(self, image_batch):
        """Run the TFLite interpreter and return all outputs as a list of numpy arrays."""
        self.interpreter.set_tensor(self.input_details[0]['index'], image_batch)
        self.interpreter.invoke()
        return [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
#-----------------------------------------------------------------------------------------------------

    def predict(self, image):
        if image.ndim == 3:
            image_batch = np.expand_dims(image, axis=0).astype(np.float32)
        elif image.ndim == 4:
            image_batch = image.astype(np.float32)
        else:
            print("Unexpected dimensions", image.ndim)
            return None

        predictions = self._run(image_batch)

        if len(predictions) == 1:
            self.heatmaps = predictions[0]
            return self.heatmaps[0]

        hm, hm16, desc, mh = _parse_predictions(predictions)
        # Strip batch dim from heatmap outputs; description/multihot already match TFExecutor shape
        self.heatmaps = hm[0] if hm is not None else None
        self.heatmaps_16b = hm16[0] if hm16 is not None else None
        self.description = desc
        self.multihot_description = mh

        return self.heatmaps
#-----------------------------------------------------------------------------------------------------

    def predict_multi(self, image):
        if image.ndim == 3:
            image_batch = np.expand_dims(image, axis=0).astype(np.float32)
        elif image.ndim == 4:
            image_batch = image.astype(np.float32)
        else:
            print("Unexpected dimensions", image.ndim)
            return None

        predictions = self._run(image_batch)

        if len(predictions) == 1:
            self.heatmaps = predictions[0]
            return self.heatmaps

        self.heatmaps, self.heatmaps_16b, self.description, self.multihot_description = \
            _parse_predictions(predictions)

        return self.heatmaps


#-----------------------------------------------------------------------------------------------------


# Ensure all outputs are converted to float32
def to_float32(tensor):
    return tensor.astype(np.float32)


#-----------------------------------------------------------------------------------------------------
def _parse_predictions(predictions):
    """Identify and separate named outputs from a list of batched prediction arrays.

    Heuristics (matching TFExecutor behaviour):
      - 4-D, channels == 1  → 16-bit heatmap
      - 4-D, channels  > 1  → main heatmaps
      - 2-D, width == 300   → GloVe token embedding
      - 2-D, other width    → multi-hot classification vector

    Returns (heatmaps, heatmaps_16b, description, multihot_description).
    All arrays retain the batch dimension; callers strip [0] for single-image use.
    """
    heatmap_idx = None
    heatmap_16b_idx = None
    multihot_idx = None
    token_indices = []

    for i, pred in enumerate(predictions):
        if pred.ndim == 4:
            if pred.shape[3] == 1:
                heatmap_16b_idx = i
            else:
                heatmap_idx = i
        elif pred.ndim == 2:
            if pred.shape[1] == 300:
                token_indices.append(i)
            else:
                multihot_idx = i

    # Heatmap outputs keep the batch dimension so callers can choose to strip it.
    heatmaps = predictions[heatmap_idx] if heatmap_idx is not None else None
    heatmaps_16b = predictions[heatmap_16b_idx] if heatmap_16b_idx is not None else None
    # Multihot: keep batch dimension (matches TFExecutor behaviour)
    multihot_description = predictions[multihot_idx] if multihot_idx is not None else None
    # Tokens: strip batch dim per token before stacking → shape (N_tokens, 300)
    description = np.vstack([predictions[i][0] for i in token_indices]) if token_indices else None

    return heatmaps, heatmaps_16b, description, multihot_description


#-----------------------------------------------------------------------------------------------------
class TFExecutor():

    def __init__(self, modelPath: str = "2d_pose_estimation/model.keras", inputWidth=256, inputHeight=256,
                 targetWidth=96, targetHeight=96, outputChannels=18, VRAMLimit=None, fp16=False, profiling=False,
                 pruneTokens=False, compileModel=True):
        self.input_size = (inputWidth, inputHeight)
        self.output_size = (targetWidth, targetHeight)
        self.numberOfHeatmaps = outputChannels
        self.profiling = profiling
        self.heatmaps = None
        self.heatmaps_16b = None
        self.description = None
        self.multihot_description = None
        self.activity = None
        #Tensorflow attempt to be reasonable
        #------------------------------------------
        print("Using Tensorflow Runtime")

        self.limitVRAMUse = VRAMLimit

        import os
        os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
        #Make sure CUDA cache is not disabled!
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        #Try to presist cudnn
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        #Try to allocate as little memory as possible
        if self.limitVRAMUse is None:
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  #<- Incompatible with setting size
        #Use seperate threads so execution is not throttled by CPU
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        #0 = all messages are logged (default behavior)
        #1 = INFO messages are not printed
        #2 = INFO and WARNING messages are not printed
        #3 = INFO, WARNING, and ERROR messages are not printed
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        #improve the stability of the auto-tuning process used to select the fastest convolution algorithms
        os.environ['TF_AUTOTUNE_THRESHOLD'] = '1'
        import keras
        import tensorflow as tf

        if self.limitVRAMUse is not None:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                    tf.config.set_logical_device_configuration(
                        gpus[0],
                        [tf.config.LogicalDeviceConfiguration(memory_limit=self.limitVRAMUse)]  # 6GB limit
                    )
                    print("Set GPU memory limit to ", self.limitVRAMUse, " MB")
                    logical_gpus = tf.config.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPU(s),", len(logical_gpus), "Logical GPU(s)")

                except RuntimeError as e:
                    print("Error setting memory limit: ", e)

        print("Tensorflow version : ", tf.__version__)
        print("Keras version      : ", keras.__version__)  #<- no longer available in TF-2.13
        print("Numpy version      : ", np.__version__)
        #-----------------------------
        from tensorflow.python.platform import build_info as tf_build_info
        print("TF/CUDA version    : ", tf_build_info.build_info['cuda_version'])
        print("TF/CUDNN version   : ", tf_build_info.build_info['cudnn_version'])

        devices = self.get_available_devices()
        print("Available Tensorflow devices are : ", devices)
        self.device = '/device:CPU:0'
        for device in devices:
            if (device.find("GPU") != -1):
                self.device = device
        print("Selecting device : ", self.device)

        if (self.profiling):
            self.tensorboard = keras.callbacks.TensorBoard(log_dir="profiling", histogram_freq=1)  #tf.
            self.startProfiling()

        self.inf_dtype = np.float32

        if (fp16):
            from tensorflow.keras import mixed_precision
            print("Enabling mixed precision (float16) inference..")
            mixed_precision.set_global_policy('mixed_float16')
            self.inf_dtype = np.float16

        if (pruneTokens):
            from NNModel import load_pictorial_only_ymapnet
            print(
                "\n\n\n\n\nLoading pruned YMAPNET version, don't wonder why you wont see any tokens/descriptions..!\n\n\n\n\n"
            )
            self.model, self.fast_call, self.input_size, self.output_size, self.numberOfHeatmaps = load_pictorial_only_ymapnet(
                modelPath)
            #self.model.summary()
        else:
            from NNModel import load_keypoints_model
            self.model, self.input_size, self.output_size, self.numberOfHeatmaps = load_keypoints_model(
                modelPath, compile=compileModel)

        #self.model.export("2d_pose_estimation", "tf_saved_model")    #Debug models

        # ── tf.function graph compilation ────────────────────────────
        # Wraps the model call in a compiled graph, avoiding the per-call
        # Python overhead of model.predict() while keeping TF's kernel
        # fusion. To disable: comment out the next two lines.
        self._predict_fn = tf.function(self.model, reduce_retracing=True)
        print("tf.function graph compilation enabled")
        # ─────────────────────────────────────────────────────────────

        if (fp16):
            print("Policy:", mixed_precision.global_policy())
            print("Compute dtype:", self.model.compute_dtype)
            self.model = tf.keras.models.clone_model(self.model)

        #------------------------------------------

    # Deleting (Calling destructor)
    def __del__(self):
        if (self.profiling):
            self.stopProfiling()
#-------------------------------------------------------------

    def startProfiling(self):
        print("Starting Tensorflow Profiling (this run will be slower than usual)..\n")
        os.system("rm -rf profiling")
        import tensorflow as tf
        tf.profiler.experimental.start('profiling')
#-------------------------------------------------------------

    def stopProfiling(self):
        print("Stopping Tensorflow Profiling..\n")
        import tensorflow as tf
        tf.profiler.experimental.stop()
        print("Please run:\n")
        print("   tensorboard --logdir profiling\n")
#-------------------------------------------------------------

    def get_available_devices(self):
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']
#-----------------------------------------------------------------------------------------------------

    def print_all_available_prediction_types(self, predictions):
        print("------ Prediction Types ------")
        if isinstance(predictions, list):
            for i, pred in enumerate(predictions):
                if not hasattr(pred, 'shape'):
                    print(f"[{i}] Not a tensor-like object: {type(pred)}")
                    continue
                print(f"[{i}] shape: {pred.shape}, dtype: {pred.dtype}", end='')

                if pred.ndim == 4:
                    print(" --> Possibly heatmaps (e.g., (1, H, W, C))")
                elif pred.ndim == 3:
                    print(" --> Possibly sequence/temporal features")
                elif pred.ndim == 2:
                    dim = pred.shape[1]
                    if dim == 300:
                        print(" --> Possibly GloVe token embedding output (300D)")
                    elif dim > 100 and dim < 300:
                        print(" --> Possibly compact embedding or multi-hot (mid-size vector)")
                    else:
                        print(" --> Possibly multi-hot or classification logits")
                elif pred.ndim == 1:
                    print(" --> Possibly single vector output")
                else:
                    print(" --> Unknown shape")
        else:
            # Handle single prediction array (not a list)
            if not hasattr(predictions, 'shape'):
                print("Prediction is not a tensor-like object.")
                return False
            print(f"Shape: {predictions.shape}, dtype: {predictions.dtype}", end='')

            if predictions.ndim == 4:
                print(" --> Possibly heatmaps (e.g., (1, H, W, C))")
            elif predictions.ndim == 3:
                print(" --> Possibly temporal or sequence data")
            elif predictions.ndim == 2:
                print(" --> Possibly multi-hot or token output")
            elif predictions.ndim == 1:
                print(" --> Possibly single vector output")
            else:
                print(" --> Unknown shape")

        print("------ End of Prediction Types ------")
        return True
#-----------------------------------------------------------------------------------------------------

    def predict(self, image):
        if (image.ndim == 3):
            image_batch = np.expand_dims(image.astype(self.inf_dtype), axis=0)
        elif (image.ndim == 4):
            image_batch = image
        else:
            print("Unexpected dimensions ", image.ndim)

        if (self.profiling):
            print(type(self.model))
            predictions = self.model.predict(image_batch, callbacks=[self.tensorboard])
        elif hasattr(self, '_predict_fn'):
            # tf.function compiled path — to disable, remove the _predict_fn
            # assignment in __init__ and this branch falls through to model.predict()
            raw = self._predict_fn(image_batch)
            if isinstance(raw, (list, tuple)):
                predictions = [p.numpy() for p in raw]
            else:
                predictions = raw.numpy()
        else:
            predictions = self.model.predict(image_batch, verbose=0)
            #predictions = self.model(image_batch, training=False)

        if type(predictions) is list:
            pass
        else:
            if (predictions.ndim == 4):
                #predictions.shape =  (1, 256, 256, 18)
                return predictions[0]

        #Else we have a list of outputs so find the correct one!
        heatmap_16bOutputIndex = None
        heatmapOutputIndex = 9
        multiHotOutputIndex = 8
        tokenOutputIndices = list()

        #16B
        #---------------------------
        heatmap_16bOutputIndex = 1
        heatmapOutputIndex = 0
        multiHotOutputIndex = 10

        for i, item in enumerate(predictions):
            #print(i," items : ",predictions[i].shape)
            #print(i," dims :  ",predictions[i].ndim)
            if predictions[i].ndim == 4:
                #print("Heatmap %u has %u elements " % (i,predictions[i].shape[3]) )
                if (predictions[i].shape[3] == 1):
                    heatmap_16bOutputIndex = i
                else:
                    heatmapOutputIndex = i

                self.heatmaps = predictions[heatmapOutputIndex][0]
            elif predictions[i].ndim == 3:
                print(" unknown output @ ", i)
            elif predictions[i].ndim == 2:
                #print(" multihot @ ",i)
                #print("  multihot length = ",len(predictions[i][0]))
                dataLength = len(predictions[i][0])
                if (dataLength == 300):  #300 dim is the GloVe vectors
                    tokenOutputIndices.append(i)
                else:
                    multiHotOutputIndex = i  #Other dimensionality should be the multihot output
                    self.multihot_description = predictions[multiHotOutputIndex]

        if (len(tokenOutputIndices) > 0):
            selected_predictions = [predictions[i][0] for i in tokenOutputIndices]
            self.description = np.vstack(selected_predictions)

        if (heatmap_16bOutputIndex is not None):
            self.heatmaps_16b = predictions[heatmap_16bOutputIndex][0]

        return self.heatmaps  #predictions[heatmapOutputIndex][0]
#-----------------------------------------------------------------------------------------------------

    def predict_multi(self, image):
        if (image.ndim == 3):
            image_batch = np.expand_dims(image.astype(self.inf_dtype), axis=0)
        elif (image.ndim == 4):
            image_batch = image
        else:
            print("Unexpected dimensions ", image.ndim)

        if (self.profiling):
            print(type(self.model))
            predictions = self.model.predict(image_batch, callbacks=[self.tensorboard])
        elif hasattr(self, '_predict_fn'):
            # tf.function compiled path — to disable, remove the _predict_fn
            # assignment in __init__ and this branch falls through to model.predict()
            raw = self._predict_fn(image_batch)
            if isinstance(raw, (list, tuple)):
                predictions = [p.numpy() for p in raw]
            else:
                predictions = raw.numpy()
        else:
            predictions = self.model.predict(image_batch, verbose=0)
            #predictions = self.model(image_batch, training=False)

        if type(predictions) is list:
            pass
        else:
            if (predictions.ndim == 4):
                #predictions.shape =  (1, 256, 256, 18)
                return predictions

        #Else we have a list of outputs so find the correct one!
        heatmapOutputIndex = 9
        multiHotOutputIndex = 8
        tokenOutputIndices = list()

        for i, item in enumerate(predictions):
            #print(i," items : ",predictions[i].shape)
            #print(i," dims :  ",predictions[i].ndim)
            if predictions[i].ndim == 4:
                #print(" heatmaps @ ",i)
                heatmapOutputIndex = i
                self.heatmaps = predictions[heatmapOutputIndex]
            elif predictions[i].ndim == 3:
                print(" unknown output @ ", i)
            elif predictions[i].ndim == 2:
                #print(" multihot @ ",i)
                #print("  multihot length = ",len(predictions[i][0]))
                dataLength = len(predictions[i])
                if (dataLength == 300):  #300 dim is the GloVe vectors
                    tokenOutputIndices.append(i)
                else:
                    multiHotOutputIndex = i  #Other dimensionality should be the multihot output
                    self.multihot_description = predictions[multiHotOutputIndex]

        if (len(tokenOutputIndices) > 0):
            selected_predictions = [predictions[i] for i in tokenOutputIndices]
            self.description = np.vstack(selected_predictions)

        return self.heatmaps  #predictions[heatmapOutputIndex][0]


#-----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
class ONNXExecutor():
    #python3 -m tf2onnx.convert --saved-model 2d_pose_estimation --opset 17 --tag serve --output 2d_pose_estimation/model.onnx
    def __init__(
        self,
        modelPath: str = "2d_pose_estimation/model.onnx",
        inputWidth=220,
        inputHeight=220,
        targetWidth=96,
        targetHeight=96,
        outputChannels=18,
        numberOfThreads=4,
    ):
        print("Using ONNX Runtime")
        import onnxruntime as ort
        import onnx
        self.input_size = (inputWidth, inputHeight)
        self.output_size = (targetWidth, targetHeight)
        self.numberOfHeatmaps = outputChannels
        self.heatmaps = None
        self.heatmaps_16b = None
        self.multihot_description = None
        self.description = None
        self.activity = None
        #------------------------------------------
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        sess_options.intra_op_num_threads = numberOfThreads
        sess_options.inter_op_num_threads = numberOfThreads
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        #------------------------------------------
        onnx.checker.check_model(onnx.load(modelPath))
        print("ONNX devices available :", ort.get_device())
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.model = ort.InferenceSession(modelPath, providers=providers, sess_options=sess_options)
        self.output_names = [o.name for o in self.model.get_outputs()]
        self.input_name = self.model.get_inputs()[0].name
        print("ONNX input  :", self.input_name)
        print("ONNX outputs:", self.output_names)
        #------------------------------------------
#-----------------------------------------------------------------------------------------------------

    def _run(self, image_batch):
        """Run an ONNX inference pass and return all outputs as a list of numpy arrays."""
        return self.model.run(self.output_names, {self.input_name: image_batch.astype(np.float32)})
#-----------------------------------------------------------------------------------------------------

    def predict(self, image):
        if image.ndim == 3:
            image_batch = np.expand_dims(image, axis=0)
        elif image.ndim == 4:
            image_batch = image
        else:
            print("Unexpected dimensions", image.ndim)
            return None

        predictions = self._run(image_batch)

        if len(predictions) == 1:
            self.heatmaps = predictions[0]
            return self.heatmaps[0]

        hm, hm16, desc, mh = _parse_predictions(predictions)
        # Strip batch dim from heatmap outputs; description/multihot already match TFExecutor shape
        self.heatmaps = hm[0] if hm is not None else None
        self.heatmaps_16b = hm16[0] if hm16 is not None else None
        self.description = desc
        self.multihot_description = mh

        return self.heatmaps
#-----------------------------------------------------------------------------------------------------

    def predict_multi(self, image):
        if image.ndim == 3:
            image_batch = np.expand_dims(image, axis=0)
        elif image.ndim == 4:
            image_batch = image
        else:
            print("Unexpected dimensions", image.ndim)
            return None

        predictions = self._run(image_batch)

        if len(predictions) == 1:
            self.heatmaps = predictions[0]
            return self.heatmaps

        self.heatmaps, self.heatmaps_16b, self.description, self.multihot_description = \
            _parse_predictions(predictions)

        return self.heatmaps


#-----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
class JAXExecutor():
    # All jax / jax2tf / tensorflow imports are deferred to __init__ so the
    # rest of the codebase works without JAX installed.
    #
    # Two model formats are supported:
    #   .keras   — Keras model bridged via jax2tf.call_tf + jax.jit
    #              (TensorFlow required at runtime)
    #   .mlirbc  — StableHLO bytecode produced by convertModelToJAX.py --stablehlo
    #              (TF-free; pure JAX/XLA execution)
    #
    # pip install "jax[cuda12]" jaxlib orbax-checkpoint keras nvidia-cusparse-cu12
    # echo 'export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cusparse, os; print(os.path.join(os.path.dirname(nvidia.cusparse.__file__),\"lib\"))"  2>/dev/null):$LD_LIBRARY_PATH' >> venv_jax/bin/activate
    # Usage: --engine jax
    def __init__(
        self,
        modelPath: str = "2d_pose_estimation/model_jax.mlirbc",
        inputWidth=256,
        inputHeight=256,
        targetWidth=96,
        targetHeight=96,
        outputChannels=18,
    ):
        print("Using JAX Runtime")
        # ── deferred core import ──────────────────────────────────────
        import jax
        import jax.numpy as jnp
        # ─────────────────────────────────────────────────────────────
        self.input_size = (inputWidth, inputHeight)
        self.output_size = (targetWidth, targetHeight)
        self.numberOfHeatmaps = outputChannels
        self.heatmaps = None
        self.heatmaps_16b = None
        self.description = None
        self.multihot_description = None
        self.activity = None
        self._jnp = jnp

        ext = os.path.splitext(modelPath)[1].lower()
        if ext == '.mlirbc':
            self._init_stablehlo(jax, jnp, modelPath)
        else:
            # .keras or any Keras-loadable format
            self._init_keras(modelPath)
#-----------------------------------------------------------------------------------------------------

    def _init_keras(self, modelPath):
        try:
            import tensorflow as tf
        except ImportError as e:
            print(f'ERROR: {e}')
            print('       Install with:  pip install jax jaxlib tensorflow')
            raise

        from NNModel import load_keypoints_model
        model, self.input_size, self.output_size, self.numberOfHeatmaps = \
            load_keypoints_model(modelPath)

        # For the .keras path there is no JAX-native JIT benefit: call_tf without
        # jax.jit is pure eager TF dispatch, and call_tf under jax.jit fails when
        # JAX has no GPU backend (DLPack device mismatch) or when model variables
        # live on GPU (XLA traces from CPU).  Skip jax2tf entirely and call the
        # TF model directly — identical execution to TFExecutor.
        # True JAX-native execution requires the .mlirbc StableHLO path.
        self._jit_fn = tf.function(lambda x: model(x, training=False))

        # Warm up: trigger tf.function graph tracing before the first real call.
        W, H = self.input_size
        dummy = tf.zeros([1, H, W, 3], dtype=tf.float32)
        _ = self._jit_fn(dummy)
        print(f'JAX: loaded {modelPath}  (tf.function path, use .mlirbc for JAX-native JIT)')
#-----------------------------------------------------------------------------------------------------

    def _init_stablehlo(self, jax, jnp, modelPath):
        # Try public jax.export API (JAX >= 0.4.28), then the experimental
        # module (JAX 0.4.14–0.4.27) as a fallback.
        _deserialize = None
        for _mod_path in ('jax.export', 'jax.experimental.export'):
            try:
                import importlib
                _mod = importlib.import_module(_mod_path)
                _deserialize = _mod.deserialize
                break
            except (ImportError, AttributeError):
                continue

        if _deserialize is None:
            raise ImportError('jax.export.deserialize not found.  Requires JAX >= 0.4.14.\n'
                              'Install with:  pip install "jax[cuda]>=0.4.14"  or  pip install jax>=0.4.14')

        # The .mlirbc artifact was produced via jax2tf.call_tf, which embeds a
        # TF custom-call effect (CallTfEffect).  That effect type is only
        # registered when jax2tf is imported — do it now so deserialize succeeds.
        try:
            from jax.experimental import jax2tf as _jax2tf_reg  # noqa: registers CallTfEffect
        except ImportError:
            pass

        with open(modelPath, 'rb') as f:
            payload = f.read()
        self._exported = _deserialize(payload)

        # Check if the artifact was exported for a specific number of devices.
        # nr_devices=0 means multi-device export, nr_devices=1 means single-device.
        _nr_devices = getattr(self._exported, 'nr_devices', 1)

        # The artifact is pinned to 1 device at export time (devices=[:1]).
        # Put the input on a device that matches the platform the artifact
        # was compiled for (e.g. 'cpu' when exported with --stablehlo).
        _platforms = getattr(self._exported, 'platforms', None) or ('cpu', )
        _platform = _platforms[0].lower()  # e.g. 'cpu' or 'cuda'
        try:
            _device = jax.devices(_platform)[0]
        except RuntimeError:
            _device = jax.local_devices()[0]

        # Wrap exported.call in jax.jit so JAX traces symbolically through
        # _call_exported_lowering (JIT path) rather than _call_exported_impl
        # (eager path).  The eager path in JAX 0.9+ checks that nr_devices in
        # the artifact matches the current device count and raises ValueError
        # when the artifact has nr_devices=0 (multi-device export) but only 1
        # device is present.  The JIT lowering path has no such guard, so the
        # artifact is re-lowered for the actual device topology at runtime.
        _exported_call = jax.jit(self._exported.call)
        self._jit_fn = lambda x: _exported_call(jax.device_put(jnp.asarray(x, dtype=jnp.float32), _device))

        W, H = self.input_size
        dummy = jnp.zeros([1, H, W, 3], dtype=jnp.float32)
        _ = self._jit_fn(dummy)
        print(
            f'JAX: loaded {modelPath}  (StableHLO + jax2tf, platforms={list(_platforms)}, device={_device}, nr_devices={_nr_devices})'
        )
        print(f'JAX version: {jax.__version__}')
#-----------------------------------------------------------------------------------------------------

    def _run(self, image_batch):
        """Run inference; return all outputs as a list of numpy arrays.
        Works for both the tf.function path (.keras) and the StableHLO path (.mlirbc)."""
        raw = self._jit_fn(image_batch.astype(np.float32))
        if isinstance(raw, (list, tuple)):
            return [np.array(r) for r in raw]
        return [np.array(raw)]
#-----------------------------------------------------------------------------------------------------

    def predict(self, image):
        if image.ndim == 3:
            image_batch = np.expand_dims(image, axis=0)
        elif image.ndim == 4:
            image_batch = image
        else:
            print("Unexpected dimensions", image.ndim)
            return None

        predictions = self._run(image_batch)

        if len(predictions) == 1:
            self.heatmaps = predictions[0][0]
            return self.heatmaps

        hm, hm16, desc, mh = _parse_predictions(predictions)
        self.heatmaps = hm[0] if hm is not None else None
        self.heatmaps_16b = hm16[0] if hm16 is not None else None
        self.description = desc
        self.multihot_description = mh
        return self.heatmaps
#-----------------------------------------------------------------------------------------------------

    def predict_multi(self, image):
        if image.ndim == 3:
            image_batch = np.expand_dims(image, axis=0)
        elif image.ndim == 4:
            image_batch = image
        else:
            print("Unexpected dimensions", image.ndim)
            return None

        predictions = self._run(image_batch)

        if len(predictions) == 1:
            self.heatmaps = predictions[0]
            return self.heatmaps

        self.heatmaps, self.heatmaps_16b, self.description, self.multihot_description = \
            _parse_predictions(predictions)
        return self.heatmaps


#-----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
class HAILOExecutor():
    # All hailo_platform imports are deferred to __init__ so the rest of the
    # codebase works without the Hailo runtime installed.
    #
    # Model file: 2d_pose_estimation/model_<arch>.hef  (produced by convertModelToHAILO.py)
    # Usage:      --engine hailo
    def __init__(
        self,
        modelPath: str = "2d_pose_estimation/model_hailo8.hef",
        inputWidth=256,
        inputHeight=256,
        targetWidth=96,
        targetHeight=96,
        outputChannels=18,
    ):
        print("Using Hailo Runtime")
        # ── deferred imports ──────────────────────────────────────────
        from hailo_platform import (
            HEF,
            VDevice,
            HailoStreamInterface,
            ConfigureParams,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            FormatType,
        )
        # ─────────────────────────────────────────────────────────────
        self.input_size = (inputWidth, inputHeight)
        self.output_size = (targetWidth, targetHeight)
        self.numberOfHeatmaps = outputChannels
        self.heatmaps = None
        self.heatmaps_16b = None
        self.description = None
        self.multihot_description = None
        self.activity = None
        # ── open device and load HEF ──────────────────────────────────
        self._target = VDevice()
        self._hef = HEF(modelPath)
        configure_params = ConfigureParams.create_from_hef(self._hef, interface=HailoStreamInterface.PCIe)
        network_groups = self._target.configure(self._hef, configure_params)
        self._network_group = network_groups[0]
        self._ng_params = self._network_group.create_params()
        # ── stream params (float32 in / float32 out) ──────────────────
        input_vstream_params = InputVStreamParams.make(self._network_group, format_type=FormatType.FLOAT32)
        output_vstream_params = OutputVStreamParams.make(self._network_group, format_type=FormatType.FLOAT32)
        self._input_name = list(input_vstream_params.keys())[0]
        self._output_names = list(output_vstream_params.keys())
        # ── enter persistent inference contexts ───────────────────────
        # Kept open for the lifetime of this object to avoid per-frame
        # context-manager overhead at inference time.
        self._infer_pipeline = InferVStreams(self._network_group, input_vstream_params, output_vstream_params)
        self._infer_pipeline.__enter__()
        self._ng_ctx = self._network_group.activate(self._ng_params)
        self._ng_ctx.__enter__()
        # ─────────────────────────────────────────────────────────────
        print(f"Hailo: loaded {modelPath}")
        print(f"  Input  stream : {self._input_name}")
        print(f"  Output streams: {self._output_names}")
#-----------------------------------------------------------------------------------------------------

    def __del__(self):
        try:
            self._ng_ctx.__exit__(None, None, None)
            self._infer_pipeline.__exit__(None, None, None)
            self._target.release()
        except Exception:
            pass
#-----------------------------------------------------------------------------------------------------

    def _run(self, image_batch):
        """Run one Hailo inference pass; return outputs as a list of numpy arrays."""
        # Hailo expects NHWC float32 normalised to [0, 1]
        data = image_batch.astype(np.float32)
        if data.max() > 1.0:
            data = data / 255.0
        results = self._infer_pipeline.infer({self._input_name: data})
        return [results[name] for name in self._output_names]
#-----------------------------------------------------------------------------------------------------

    def predict(self, image):
        if image.ndim == 3:
            image_batch = np.expand_dims(image, axis=0)
        elif image.ndim == 4:
            image_batch = image
        else:
            print("Unexpected dimensions", image.ndim)
            return None

        predictions = self._run(image_batch)

        if len(predictions) == 1:
            self.heatmaps = predictions[0][0]
            return self.heatmaps

        hm, hm16, desc, mh = _parse_predictions(predictions)
        self.heatmaps = hm[0] if hm is not None else None
        self.heatmaps_16b = hm16[0] if hm16 is not None else None
        self.description = desc
        self.multihot_description = mh
        return self.heatmaps
#-----------------------------------------------------------------------------------------------------

    def predict_multi(self, image):
        if image.ndim == 3:
            image_batch = np.expand_dims(image, axis=0)
        elif image.ndim == 4:
            image_batch = image
        else:
            print("Unexpected dimensions", image.ndim)
            return None

        predictions = self._run(image_batch)

        if len(predictions) == 1:
            self.heatmaps = predictions[0]
            return self.heatmaps

        self.heatmaps, self.heatmaps_16b, self.description, self.multihot_description = \
            _parse_predictions(predictions)
        return self.heatmaps


#-----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
class PyTorchExecutor():
    # TorchScript (.pt) model produced by convertModelToPytorch.py.
    #
    # The .pt file is a traced TorchScript model exported in PyTorch NCHW layout.
    # Input images are converted NHWC→NCHW internally; outputs are returned as
    # numpy arrays in the same shape convention as every other executor.
    #
    # pip install torch onnx2torch
    # Usage: --engine pytorch
    def __init__(
        self,
        modelPath: str = "2d_pose_estimation/model_pytorch.pt",
        inputWidth=256,
        inputHeight=256,
        targetWidth=96,
        targetHeight=96,
        outputChannels=18,
    ):
        print("Using PyTorch Runtime")
        # ── deferred imports ──────────────────────────────────────────
        import torch
        # ─────────────────────────────────────────────────────────────
        self.input_size = (inputWidth, inputHeight)
        self.output_size = (targetWidth, targetHeight)
        self.numberOfHeatmaps = outputChannels
        self.heatmaps = None
        self.heatmaps_16b = None
        self.description = None
        self.multihot_description = None
        self.activity = None
        self._torch = torch

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = torch.jit.load(modelPath, map_location=self._device)
        self._model.eval()

        # Determine weight dtype from the first parameter so we can cast
        # input tensors to match (handles fp16-exported models).
        try:
            self._dtype = next(self._model.parameters()).dtype
        except StopIteration:
            self._dtype = torch.float32

        # Warm-up: trigger JIT compilation before the first real call.
        W, H = self.input_size
        dummy = torch.zeros(1, 3, H, W, dtype=self._dtype, device=self._device)
        with torch.no_grad():
            _ = self._model(dummy)
        print(f'PyTorch: loaded {modelPath}  (device={self._device}, dtype={self._dtype})')
        print(f'PyTorch version: {torch.__version__}')
#-----------------------------------------------------------------------------------------------------

    def _run(self, image_batch):
        """Run inference; return all outputs as a list of numpy arrays.

        Expects image_batch in NHWC float32 numpy format (standard project
        convention); converts to NCHW torch.Tensor before calling the model.
        """
        torch = self._torch
        # NHWC → NCHW
        x = np.transpose(image_batch, (0, 3, 1, 2))
        x = torch.from_numpy(x).to(dtype=self._dtype, device=self._device)
        with torch.no_grad():
            raw = self._model(x)
        # Normalise output to a list of numpy arrays in NHWC order
        if isinstance(raw, (list, tuple)):
            outputs = []
            for t in raw:
                arr = t.cpu().numpy()
                # 4-D NCHW → NHWC for heatmap tensors
                if arr.ndim == 4:
                    arr = np.transpose(arr, (0, 2, 3, 1))
                outputs.append(arr)
            return outputs
        else:
            arr = raw.cpu().numpy()
            if arr.ndim == 4:
                arr = np.transpose(arr, (0, 2, 3, 1))
            return [arr]
#-----------------------------------------------------------------------------------------------------

    def predict(self, image):
        if image.ndim == 3:
            image_batch = np.expand_dims(image, axis=0).astype(np.float32)
        elif image.ndim == 4:
            image_batch = image.astype(np.float32)
        else:
            print("Unexpected dimensions", image.ndim)
            return None

        predictions = self._run(image_batch)

        if len(predictions) == 1:
            self.heatmaps = predictions[0][0]
            return self.heatmaps

        hm, hm16, desc, mh = _parse_predictions(predictions)
        self.heatmaps = hm[0] if hm is not None else None
        self.heatmaps_16b = hm16[0] if hm16 is not None else None
        self.description = desc
        self.multihot_description = mh
        return self.heatmaps
#-----------------------------------------------------------------------------------------------------

    def predict_multi(self, image):
        if image.ndim == 3:
            image_batch = np.expand_dims(image, axis=0).astype(np.float32)
        elif image.ndim == 4:
            image_batch = image.astype(np.float32)
        else:
            print("Unexpected dimensions", image.ndim)
            return None

        predictions = self._run(image_batch)

        if len(predictions) == 1:
            self.heatmaps = predictions[0]
            return self.heatmaps

        self.heatmaps, self.heatmaps_16b, self.description, self.multihot_description = \
            _parse_predictions(predictions)
        return self.heatmaps


#-----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
class NNExecutor():

    def __init__(self, engine: str = "tensorflow", modelPath: str = "2d_pose_estimation/", inputWidth=220,
                 inputHeight=220, targetWidth=96, targetHeight=96, outputChannels=18, numberOfThreads=4,
                 profiling=False, pruneTokens=False, VRAMLimit=None, compileModel=True):
        defaultModelDir = "2d_pose_estimation/"
        self.hz = 0.0
        self.model = None
        print("NNExecutor for ", engine)
        print("Asked to load from: ", modelPath)

        def _resolve(directory, *candidates):
            """Return the first existing candidate file under directory, or the first one (with a warning)."""
            import os
            for name in candidates:
                path = os.path.join(directory, name)
                if os.path.isfile(path):
                    return path
            chosen = os.path.join(directory, candidates[0])
            print(f"WARNING: none of {candidates} found in {directory}, trying {chosen}")
            return chosen

        if (engine == "tensorflow") or (engine == "tf"):
            if (modelPath == defaultModelDir):
                modelPath = _resolve(modelPath, "model.keras")
            self.model = TFExecutor(modelPath=modelPath, profiling=profiling, inputWidth=inputWidth,
                                    inputHeight=inputHeight, targetWidth=targetWidth, targetHeight=targetHeight,
                                    outputChannels=outputChannels, pruneTokens=pruneTokens, VRAMLimit=VRAMLimit,
                                    compileModel=compileModel)
        elif (engine == "tf-lite") or (engine == "tflite"):
            if (modelPath == defaultModelDir):
                modelPath = _resolve(modelPath, "model_fp16.tflite", "model.tflite", "model_int8.tflite")
            self.model = TFLiteExecutor(modelPath=modelPath, inputWidth=inputWidth, inputHeight=inputHeight,
                                        targetWidth=targetWidth, targetHeight=targetHeight,
                                        outputChannels=outputChannels, numberOfThreads=numberOfThreads)
        elif (engine == "onnx"):
            if (modelPath == defaultModelDir):
                modelPath = _resolve(modelPath, "model.onnx", "model_fp16.onnx")
            self.model = ONNXExecutor(modelPath=modelPath, inputWidth=inputWidth, inputHeight=inputHeight,
                                      targetWidth=targetWidth, targetHeight=targetHeight, outputChannels=outputChannels,
                                      numberOfThreads=numberOfThreads)
        elif (engine == "jax"):
            if (modelPath == defaultModelDir):
                modelPath = _resolve(modelPath, "model_jax.mlirbc", "model.keras")
            self.model = JAXExecutor(
                modelPath=modelPath,
                inputWidth=inputWidth,
                inputHeight=inputHeight,
                targetWidth=targetWidth,
                targetHeight=targetHeight,
                outputChannels=outputChannels,
            )
        elif (engine == "hailo"):
            if (modelPath == defaultModelDir):
                modelPath = _resolve(modelPath, "model_hailo8.hef", "model_hailo8l.hef", "model_hailo8r.hef",
                                     "model_hailo15h.hef", "model.hef")
            self.model = HAILOExecutor(
                modelPath=modelPath,
                inputWidth=inputWidth,
                inputHeight=inputHeight,
                targetWidth=targetWidth,
                targetHeight=targetHeight,
                outputChannels=outputChannels,
            )
        elif (engine == "pytorch") or (engine == "torch"):
            if (modelPath == defaultModelDir):
                modelPath = _resolve(modelPath, "model_pytorch.pt", "model_pytorch_fp16.pt")
            self.model = PyTorchExecutor(
                modelPath=modelPath,
                inputWidth=inputWidth,
                inputHeight=inputHeight,
                targetWidth=targetWidth,
                targetHeight=targetHeight,
                outputChannels=outputChannels,
            )

    #------------------------------------------
#-----------------------------------------------------------------------------------------------------

    def predict(self, image):
        #------------------------------------------
        start = time.time()
        prediction = self.model.predict(image)
        seconds = time.time() - start
        self.hz = 1 / (seconds + 0.0001)
        #------------------------------------------
        return prediction
#-----------------------------------------------------------------------------------------------------

    def predict_multi(self, image):
        #------------------------------------------
        start = time.time()
        prediction = self.model.predict_multi(image)
        seconds = time.time() - start
        self.hz = 1 / (seconds + 0.0001)
        #------------------------------------------
        return prediction
#-----------------------------------------------------------------------------------------------------

    def heatmaps_16b(self):
        #------------------------------------------
        return self.model.heatmaps_16b
        #------------------------------------------
#-----------------------------------------------------------------------------------------------------

    def multihot(self):
        #------------------------------------------
        return self.model.multihot_description
        #------------------------------------------
    def description(self):
        #------------------------------------------
        return self.model.description
        #------------------------------------------


#-----------------------------------------------------------------------------------------------------

    def activity(self):
        #------------------------------------------
        return self.model.activity
        #------------------------------------------
