# Y-MAP-Net


We present Y-MAP-Net, a Y-shaped neural network architecture designed for real-time multi-task learning on RGB images. Y-MAP-Net simultaneously predicts depth, surface normals, human pose, semantic segmentation, and generates multi-label captions in a single forward pass. To achieve this, we adopt a multi-teacher, single-student training paradigm, where task-specific foundation models supervise the learning of the network, allowing it to distill their capabilities into a unified real-time inference architecture. Y-MAP-Net exhibits strong generalization, architectural simplicity, and computational efficiency, making it well-suited for resource-constrained robotic platforms. By providing rich 3D, semantic, and contextual scene understanding from low-cost RGB cameras, Y-MAP-Net supports key robotic capabilities such as object manipulation and human–robot interaction.


---

## Features

- **Real-time inference** from webcam, video files, image folders or screen capture
- **Multi-task outputs**: 2D pose (17 COCO joints), depth, surface normals, segmentation, and text token embeddings
- **Multiple backends**: TensorFlow, TFLite, JAX and ONNX
- **Interactive web UI** via Gradio

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
