# ymapnet_cpp

C++ inference engine for YMAPNet 2D pose estimation using **ggml** (CUDA + CPU) and **OpenCV**.

## System Requirements

| Dependency | Tested version | Notes |
|---|---|---|
| GCC | ≥ 12 | C++17 required |
| CMake | ≥ 3.18 | |
| OpenCV | ≥ 4.5 | core, imgproc, videoio, highgui |
| CUDA Toolkit | 12.x | Optional — CPU fallback available |
| NVIDIA driver | ≥ 525 | For CUDA |
| Python 3 + gguf | 3.10+ | Only needed for `convertModelToGGUF.py` |

## 1. Install system dependencies

```bash
sudo apt-get install -y \
    build-essential cmake git ninja-build pkg-config \
    libopencv-dev

# CUDA (skip for CPU-only build):
sudo apt-get install -y cuda-toolkit-12-2 libcublas-dev-12-2
```

## 2. Export the Keras model to GGUF

From the project root:

```bash
pip install gguf

# F32 — best numerical accuracy
python3 -m ymapnet.conversion.convertModelToGGUF --model ymapnet_model/model.keras --dtype f32 --verify

# F16 — half the size (conv/dense kernels only; biases, norms, PReLU and depthwise stay F32)
python3 -m ymapnet.conversion.convertModelToGGUF --model ymapnet_model/model.keras --dtype f16
```

Supported architecture (v288+): `useDepthwiseSeparable` ConvNeXt blocks, PReLU, CoordConv,
dilated-conv bridge + bottleneck attention, `nearest` upsampling decoder, 8-bit + 16-bit heads.
Level widths and optional parts are discovered from the GGUF tensor names. Older GGUFs
(ASPP bridge, depth-refinement head, Conv2DTranspose decoder) are not supported.
The token / descriptor heads are exported but not run.

## 3. Build

```bash
cd ymapnet_cpp

# Auto-detects CUDA
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)

# CPU-only
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build --parallel $(nproc)

# Vulkan (GPUs without CUDA; needs a Vulkan 1.2 driver). Build deps: libvulkan-dev glslc spirv-headers
cmake -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DYMAPNET_VULKAN=ON
cmake --build build-vulkan --parallel $(nproc)
# run with:  --vulkan 0   (device index; CUDA/CPU are used otherwise)
```

> **Vulkan note:** current ggml master does not add the SPIRV-Headers include path; if the build
> fails with `'spv' has not been declared`, add `-DCMAKE_CXX_FLAGS="-isystem <spirv-headers prefix>/include"`.
> On the RTX 4080 SUPER Vulkan runs at 35 ms/frame (CUDA: 25 ms), max|diff| vs float32 Keras 6.4 (CUDA: 0.23).

> **CUDA architecture note:** The default target is `sm_86` (Ampere). This runs on Ada Lovelace (RTX 4080/4090) via CUDA backward compatibility. Targeting `sm_89` directly triggers a ptxas FP8 bug in CUDA 12.2 — use `sm_86` or upgrade to CUDA ≥ 12.3.

### Android (arm64, Android 13+ / API 33, Vulkan)

Needs the NDK, the OpenCV Android SDK (4.x), Khronos Vulkan-Headers matching the NDK's
`vulkan_core.h` (NDK r30 → v1.4.335), SPIRV-Headers, and a `glslc` newer than the NDK's bundled
shaderc 2022.3 (which cannot compile ggml's `coopmat` shaders; shaderc 2023.8 works).

```bash
NDK=<sdk>/ndk/30.0.16248370
cmake -B build-android -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33 \
  -DOpenCV_DIR=<OpenCV-android-sdk>/sdk/native/jni \
  -DYMAPNET_VULKAN=ON -DVulkan_GLSLC_EXECUTABLE=<glslc ≥ shaderc 2023> \
  -DVulkan_INCLUDE_DIR=<Vulkan-Headers>/include \
  -DSPIRV-Headers_DIR=<spirv-install>/share/cmake/SPIRV-Headers \
  -DCMAKE_CXX_FLAGS="-isystem <spirv-install>/include" \
  -DBUILD_SHARED_LIBS=OFF        # link ggml statically → one self-contained binary
cmake --build build-android --parallel $(nproc)

# Run on device (headless). llvm-strip cuts the binary from ~74 MB to ~33 MB.
$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip build-android/ymapnet
adb shell mkdir -p /data/local/tmp/ymapnet
adb push build-android/ymapnet ymapnet_model/model_f16.gguf image.jpg /data/local/tmp/ymapnet/
adb shell 'cd /data/local/tmp/ymapnet && ./ymapnet --model model_f16.gguf --from image.jpg --vulkan 0 --headless --dump out.bin'
```

### Android app (camera + skeleton overlay)

`android/` is a Gradle project (Java, CameraX, JNI → `ymapnet_core`). Vulkan device 0 is tried
first, then CPU. Paths to the native deps, the model and an optional out-of-tree build directory go in
`android/local.properties` (not committed):

```properties
sdk.dir=<android sdk>
ymapnet.opencvDir=<OpenCV-android-sdk>/sdk/native/jni
ymapnet.glslc=<glslc ≥ shaderc 2023>
ymapnet.vulkanInclude=<Vulkan-Headers>/include
ymapnet.spirvDir=<spirv-install>
ymapnet.model=<repo>/ymapnet_model/model_f16_heatmaps.gguf
ymapnet.ggmlSource=<optional: existing ggml checkout at the pinned commit>
ymapnet.buildDir=<optional: where APKs / native build trees go>
ymapnet.tenpamyModel=<optional, DEV ONLY: tenpamy_model/tenpamy_f16.gguf>
```

`ymapnet.tenpamyModel` adds the "Reverse Score" view: the TENpamy inverse network (heatmaps → RGB,
`ymapnet/training/trainTENpamy.py`) rebuilds the frame from YMAPNet's heatmaps and shows it with its MAE
against the network input (as `ymapnet/evaluation/tenpamyScore.py`). The model is loaded only while the
view is selected. All of it sits behind `YMAPNET_TENPAMY` (C++) / `BuildConfig.TENPAMY` (Java, plus
`Tenpamy.java` and `src/tenpamy.hpp`); without the property the native code is not compiled, the
Java hooks are dead code (stripped by R8) and no model is bundled.
Export with `python3 -m ymapnet.conversion.convertTENpamyToGGUF` (33 MB F16).

```bash
# Heatmap-only model (drops the unused token/descriptor heads: 143 MB → 78 MB, identical heatmaps)
python3 -m ymapnet.conversion.convertModelToGGUF --model ymapnet_model/model.keras --dtype f16 --heatmaps-only
# ... plus the token head for the "Description" text (141 MB; what the app bundles now)
python3 -m ymapnet.conversion.convertModelToGGUF --model ymapnet_model/model.keras --dtype f16 --heatmaps-only --tokens

cd ymapnet_cpp/android
./gradlew assembleRelease          # use release: the debug variant builds ggml at -O0
adb install -r <buildDir>/app/outputs/apk/release/app-release.apk
```

The app loads the F16 GGUF with `upcast_f16` (weights stored as F32 on the GPU): on the Exynos 2200
(Xclipse 920) F16 matmuls are ~2.3x slower than F32, so this keeps the small file and the fast path.
The CLI equivalent is `--upcast`.

Feature maps are kept channel-first (CWH) end to end: ggml's depthwise conv reads/writes that layout
directly, pooling uses strided views and the 3x3 convs are im2col + weights-as-src0 matmuls, so the
ConvNeXt blocks need no layout copies. Forward pass per 256×256 frame (F16 GGUF loaded as F32):

| Device / backend            | before (WHC↔CWH copies) | now (CWH throughout) |
|-----------------------------|-------------------------|----------------------|
| Galaxy S22 Ultra, Vulkan    | 2.35 s                  | 1.32 s               |
| RTX 4080 SUPER, Vulkan      | 42.4 ms                 | 37.4 ms              |
| RTX 4080 SUPER, CUDA        | 27.9 ms                 | 27.2 ms              |
| Galaxy XCover Pro, CPU (8t) | 12.3 s                  | (not re-measured)    |

Two further Vulkan-only speedups (S22 Ultra, median of warm frames with a cool phone, 1180 → 925 ms):

- **Direct 3x3 convs** (`ggml_conv_2d_direct`) instead of im2col, which at 256×256 writes a ~300 MB
  buffer per conv: 1180 → 1090 ms (RTX 4080 Vulkan 35.4 → 32.6 ms). Not used on CUDA, where it is 2x slower.
- **Fused bias + PReLU shader**, a local ggml patch (`patches/ggml-vulkan-bias-prelu.patch`, applied
  by CMake to the pinned ggml, also to a `FETCHCONTENT_SOURCE_DIR_GGML` checkout). ggml-vulkan
  recognises the graph `ADD(bias) → RELU → SUB → MUL → ADD` that `prelu()` builds and runs it as one
  pass instead of five: 1105 → 925 ms. Output is bit-identical to the unfused ops
  (`GGML_VK_DISABLE_FUSION=1` turns it off for comparison).

Accuracy vs Keras float32 on the test image (8-bit heads, [-120, 120] scale): CPU max|diff| 1.1 /
mean 0.001, CUDA 1.1 / 0.009, Vulkan 3.8 / 0.016.
On the CPU backend the app shows a **Threads** button (slider, 1..cores, default all).

Release APK: 175.2 MB with the token head (model 140.9, native ~33 – mostly ggml's embedded Vulkan
shaders –, dex ~1); 111.8 MB with the heatmap-only model.

The token head is exported as a small text graph plus the vocabulary (`ymapnet.token_graph`,
`ymapnet.vocabulary`) and built generically in C++; the CLI prints `Description: word:0.49 ...` and
`--dump PATH` also writes the raw scores to `PATH.tokens`. Checked against Keras float32 on the test
image: scores within 1.6e-4, top-21 words and the description string identical to
`TokenEstimator.decodeOneHotDescriptionToString(..., K=21)`.
The release build is debug-signed for testing; configure a real keystore before distributing.

## 4. Run

```bash
# Model info
ymapnet_cpp/build/ymapnet --model ymapnet_model/model_f32.gguf --info

# Webcam
ymapnet_cpp/build/ymapnet --model ymapnet_model/model_f32.gguf --from 0

# Video file
ymapnet_cpp/build/ymapnet --model ymapnet_model/model_f32.gguf --from video.mp4

# Single image
ymapnet_cpp/build/ymapnet --model ymapnet_model/model_f32.gguf --from image.jpg

# Force CPU backend
ymapnet_cpp/build/ymapnet --model ymapnet_model/model_f32.gguf --from 0 --cpu

# All options
ymapnet_cpp/build/ymapnet --help
```

Two OpenCV windows are shown: the skeleton overlay and a tiled heatmap grid of all 76 output channels.

## 5. Validate against Python baseline

```bash
python3 -m ymapnet.conversion.validate_cpp \
    --image  1730235538988943.jpg \
    --model  ymapnet_model/model.keras \
    --cpp-bin ymapnet_cpp/build/ymapnet \
    --gguf   ymapnet_model/model_f32.gguf
```

The Keras baseline runs in mixed_bfloat16, so the diff is dominated by bf16 rounding
(v288: max |diff| ≈ 21, mean ≈ 0.21 on the `[-120, 120]` scale). Against the same model
forced to float32 the F32 GGUF gives max |diff| 0.23 / mean 0.001 (F16 GGUF: 4.1 / 0.011).
`--dump PATH` also writes the 16-bit head to `PATH.16bit`.

## 6. Source layout

```
ymapnet_cpp/
├── CMakeLists.txt
└── src/
    ├── main.cpp            CLI, video loop, timing
    ├── model.hpp/cpp       GGUF load, ggml graph, forward pass
    ├── preprocess.hpp      Centre-crop, resize, BGR→RGB, CHW planar layout
    ├── skeleton.hpp        Joint topology, multi-person resolve_skeletons(), drawing
    ├── test_skeleton.cpp   Resolver parity check vs Python (scripts/testSkeletonParity.py)
    └── heatmap_vis.hpp     Tiled heatmap grid visualisation
```
