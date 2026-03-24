# Y-MAP-Net

A real-time **Depth, Normals, Pose, Segmentation and token captioning** system from monocular RGB webcams, video files, or other video sources. Built on TensorFlow/Keras, it supports multi-task outputs including pose keypoints, depth maps, surface normals, segmentation, and natural language embeddings.

---

## Features

- **Real-time inference** from webcam, video files, image folders, screen capture, or ESP32 camera
- **Multi-task outputs**: 2D pose (17 COCO joints), depth, surface normals, segmentation, and text token embeddings
- **Multiple backends**: TensorFlow, TFLite, and ONNX
- **GPU and CPU support** with optional mixed-precision and model quantization
- **Interactive web UI** via Gradio
- **Full training pipeline** with data augmentation, early stopping, and loss weighting

---

## Quick Start

### 1. Setup

```bash
# Automated setup (creates a virtual environment and installs dependencies)
scripts/setup.sh
source venv/bin/activate
```

Or using Docker:

```bash
docker/build_and_deploy.sh
docker run rgbposedetect2d-container
docker attach rgbposedetect2d-container
cd workspace
scripts/setup.sh
source venv/bin/activate
```

### 2. Download a Pre-trained Model

```bash
scripts/downloadPretrained.sh
```

### 3. Run

```bash
./runYMAPNet.sh
```

This starts real-time pose estimation from your default webcam.

---

## Running Inference

### Input Sources

```bash
# Webcam (default)
./runYMAPNet.sh

# Video file
./runYMAPNet.sh --from /path/to/video.mp4

# Image directory / sequence
./runYMAPNet.sh --from /path/to/images/

# Screen capture
./runYMAPNet.sh --from screen

# Specific video device
./runYMAPNet.sh --from /dev/video0
```

### Common Options

| Flag | Description |
|------|-------------|
| `--size W H` | Set input resolution (e.g. `--size 640 480`) |
| `--cpu` | Force CPU-only inference (slower) |
| `--fast` | Disable depth refinement, person ID, and skeleton resolution for speed |
| `--save` | Save output frames to disk |
| `--headless` | Run without any display window |
| `--illustrate` | Enable enhanced visualization overlay |
| `--collab` | Headless mode with save + illustrate (useful for Colab/remote) |
| `--profiling` | Enable performance profiling |

### Web Interface

```bash
python3 gradioServer.py
# Open http://localhost:7860 in your browser
```

---


## Prerequisites

- Python 3.x
- TensorFlow 2.16.1+ (with CUDA 12.3+ and cuDNN 8.9.6+ for GPU support)
- Keras 3+
- NumPy, OpenCV
- See `requirements.txt` for the full list

Install all dependencies:

```bash
pip install -r requirements.txt
# or run the setup script:
python3 scripts/setup.sh
```


---

## License

FORTH License — see [LICENSE](LICENSE) for details.
