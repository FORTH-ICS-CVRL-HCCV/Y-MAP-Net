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
python3 convertModelToGGUF.py --model 2d_pose_estimation/model.keras --dtype f32 --verify

# F16 — half the size, recommended for inference
python3 convertModelToGGUF.py --model 2d_pose_estimation/model.keras --dtype f16
```

## 3. Build

```bash
cd ymapnet_cpp

# Auto-detects CUDA
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)

# CPU-only
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build --parallel $(nproc)
```

> **CUDA architecture note:** The default target is `sm_86` (Ampere). This runs on Ada Lovelace (RTX 4080/4090) via CUDA backward compatibility. Targeting `sm_89` directly triggers a ptxas FP8 bug in CUDA 12.2 — use `sm_86` or upgrade to CUDA ≥ 12.3.

## 4. Run

```bash
# Model info
ymapnet_cpp/build/ymapnet --model 2d_pose_estimation/model_f32.gguf --info

# Webcam
ymapnet_cpp/build/ymapnet --model 2d_pose_estimation/model_f32.gguf --from 0

# Video file
ymapnet_cpp/build/ymapnet --model 2d_pose_estimation/model_f32.gguf --from video.mp4

# Single image
ymapnet_cpp/build/ymapnet --model 2d_pose_estimation/model_f32.gguf --from image.jpg

# Force CPU backend
ymapnet_cpp/build/ymapnet --model 2d_pose_estimation/model_f32.gguf --from 0 --cpu

# All options
ymapnet_cpp/build/ymapnet --help
```

Two OpenCV windows are shown: the skeleton overlay and a tiled heatmap grid of all 73 output channels.

## 5. Validate against Python baseline

```bash
python3 validate_cpp.py \
    --image  1730235538988943.jpg \
    --model  2d_pose_estimation/model.keras \
    --cpp-bin ymapnet_cpp/build/ymapnet \
    --gguf   2d_pose_estimation/model_f32.gguf
```

Expected: `max |diff| < 20.0` on the `[-120, 120]` scale (Keras runs mixed_bfloat16; C++ runs F32).

## 6. Source layout

```
ymapnet_cpp/
├── CMakeLists.txt
└── src/
    ├── main.cpp            CLI, video loop, timing
    ├── model.hpp/cpp       GGUF load, ggml graph, forward pass
    ├── preprocess.hpp      Centre-crop, resize, BGR→RGB, CHW planar layout
    ├── postprocess.hpp     Peak detection, keypoint extraction
    ├── skeleton.hpp        Joint names, edges, draw_skeleton()
    └── heatmap_vis.hpp     Tiled heatmap grid visualisation
```
