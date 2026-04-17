# ymapnet_cpp

C++ inference engine for YMAPNet pose estimation using **ggml** (CUDA + CPU) and **OpenCV**.

## Architecture

```
Phase 1  convertModelToGGUF.py     Export Keras weights → GGUF (done)
Phase 2  ymapnet_cpp/              CMake skeleton + GGUF loader (this directory)
Phase 3  src/model.cpp forward()   Full U-Net graph in ggml (todo)
Phase 4  src/preprocess/postprocess.hpp  Already implemented
Phase 5  src/main.cpp              Full CLI (already wired, needs Phase 3)
```

## System Requirements

| Dependency | Tested version | Notes |
|---|---|---|
| Ubuntu / Debian | 22.04 / 24.04 | |
| GCC | ≥ 12 | C++17 required |
| CMake | ≥ 3.18 | |
| CUDA Toolkit | 12.x | Optional – CPU fallback available |
| NVIDIA driver | ≥ 525 | For CUDA |
| OpenCV | ≥ 4.5 | core, imgproc, videoio, highgui |
| Python 3 + gguf pkg | 3.10+ | Only needed to run convertModelToGGUF.py |

## 1. Install system dependencies

```bash
# Build tools
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git ninja-build \
    pkg-config

# OpenCV (runtime + dev headers)
sudo apt-get install -y \
    libopencv-dev \
    libopencv-core-dev \
    libopencv-imgproc-dev \
    libopencv-videoio-dev \
    libopencv-highgui-dev

# CUDA Toolkit (skip if CPU-only build)
# Download from https://developer.nvidia.com/cuda-downloads
# or via apt:
sudo apt-get install -y \
    cuda-toolkit-12-2 \
    libcublas-dev-12-2 \
    libcurand-dev-12-2
# Add to PATH if needed:
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 2. Export the Keras model to GGUF (Phase 1)

From the project root (`RGBToPoseDetect2D/`):

```bash
# Install Python dependency (once)
pip install gguf

# Export to F32 (456 MB, best for numerical validation)
python3 convertModelToGGUF.py \
    --model 2d_pose_estimation/model.keras \
    --dtype f32 \
    --verify

# Export to F16 (228 MB, recommended for inference)
python3 convertModelToGGUF.py \
    --model 2d_pose_estimation/model.keras \
    --dtype f16

# Further quantise to Q8_0 using the llama.cpp quantize tool (optional):
# ./llama.cpp/quantize 2d_pose_estimation/model_f32.gguf model_q8.gguf q8_0
```

## 3. Build ymapnet_cpp

```bash
cd ymapnet_cpp

# Configure (auto-detects CUDA)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build (uses all CPU cores)
cmake --build build --parallel $(nproc)

# Binary: ymapnet_cpp/build/ymapnet
```

### CPU-only build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build --parallel $(nproc)
```

### CUDA architecture note

The default target is **sm_86** (Ampere binaries). These run without modification on
Ada Lovelace (RTX 4080 / 4090, sm_89) via CUDA backward compatibility.

Compiling directly for sm_89 triggers a CUDA 12.2 ptxas bug related to FP8 codegen;
the workaround is sm_86 or upgrading to CUDA ≥ 12.3.  To override:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89-real"
```

(`89-real` emits SASS only, bypassing the PTX assembler step that fails.)

## 4. Run

From the project root:

```bash
# Print model info and exit
ymapnet_cpp/build/ymapnet \
    --model 2d_pose_estimation/model_f32.gguf \
    --info

# Webcam (device 0)
ymapnet_cpp/build/ymapnet \
    --model 2d_pose_estimation/model_f32.gguf \
    --from 0

# Video file
ymapnet_cpp/build/ymapnet \
    --model 2d_pose_estimation/model_f32.gguf \
    --from /path/to/video.mp4

# Single image
ymapnet_cpp/build/ymapnet \
    --model 2d_pose_estimation/model_f32.gguf \
    --from /path/to/image.jpg

# Force CPU backend
ymapnet_cpp/build/ymapnet \
    --model 2d_pose_estimation/model_f32.gguf \
    --from 0 --cpu

# All options
ymapnet_cpp/build/ymapnet --help
```

## 5. Source layout

```
ymapnet_cpp/
├── CMakeLists.txt          Build configuration
└── src/
    ├── main.cpp            CLI, video loop, timing
    ├── model.hpp           YMAPNetModel class declaration
    ├── model.cpp           GGUF loading, backend init, forward stub
    ├── preprocess.hpp      BGR→RGB, resize, normalise [0,1]
    ├── postprocess.hpp     Heatmap dequant, peak detection, keypoint extraction
    └── skeleton.hpp        COCO-17 joint names, edges, draw_skeleton()
```

## 6. Phase 3 — implementing the forward pass

`model.cpp::YMAPNetModel::forward()` currently returns zeros. Phase 3 replaces
the stub with the full ggml U-Net graph:

```
encoder (7 × conv_block + avg_pool)
    ↓
bridge (3 × dilated conv)
    ↓
decoder (7 × Conv2DTranspose + skip concat + conv_block)
    ↓
1×1 conv → tanh → ×120 → 73-channel heatmap
```

All 414 weight tensors are loaded onto the CUDA backend and indexed by
`weights_["layer_name.weight"]`. The full layer topology is in
`meta_.layer_arch_json` (written by `convertModelToGGUF.py`).

Key ggml ops needed:

| Layer | ggml op |
|---|---|
| Conv2D + bias | `ggml_conv_2d` + `ggml_add` |
| Leaky ReLU | `ggml_leaky_relu` |
| AveragePooling 2×2 | `ggml_pool_2d(GGML_OP_POOL_AVG)` |
| Upsample 2× | `ggml_upscale` |
| Skip concat | `ggml_concat` |
| Conv2DTranspose | `ggml_upscale` + `ggml_conv_2d` |
| Tanh + scale | `ggml_tanh` + `ggml_scale` |

## 7. Validation (Phase 3+)

Compare heatmap output against the Python baseline:

```bash
# Target: max |diff| < 1.0 on [-120, 120] scale for F32
#         max |diff| < 2.0 on [-120, 120] scale for F16
```
