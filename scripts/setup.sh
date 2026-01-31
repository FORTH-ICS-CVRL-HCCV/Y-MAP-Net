#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..

USE_VENV=1
if [[ " $* " == *" --collab "* ]]; then
  echo "Using non-blocking colab mode (no venv)"
  USE_VENV=0
fi

if [ -d venv/ ]; then
  echo "Found a virtual environment"
  if [ "$USE_VENV" -eq "1" ]; then
    source venv/bin/activate
  else
    echo "Colab mode: ignoring venv"
  fi
else
  echo "No virtual environment found"

  # Only create venv if we're supposed to use it
  if [ "$USE_VENV" -eq "1" ]; then
    echo "Creating a virtual environment"
    # (optional) system deps check here
    python3 -m venv venv
    source venv/bin/activate
  else
    echo "Colab mode: not creating venv"
  fi
fi

python3 -m pip install --upgrade pip

if [ "$USE_VENV" -eq "0" ]; then
  echo "Setting up Colab packages"
  python3 -m pip install tensorflow tensorflow-model-optimization tf_keras numpy numba tensorboard tensorboard-plugin-profile etils importlib_resources tf2onnx onnxruntime onnx opencv-python wget gradio
else
  echo "Setting up regular set of packages"
  #python3 -m pip install nvidia-cudnn-cu12 tensorflow==2.17.0 tensorflow-model-optimization tf_keras numpy numba tensorboard tensorboard-plugin-profile tf2onnx onnxruntime onnx opencv-python wget gradio
  #This will probably install keras 3.5+ which currently has a problam with the dataloader so revert back to a working version
  #python3 -m pip install keras==3.4.0

  #For a 50XX series card
  #wget https://github.com/mypapit/tensorflowRTX50/releases/download/2.20dev-ubuntu-24.04-avx-too/tensorflow-2.20.0dev0+selfbuild-cp312-cp312-linux_x86_64.whl
  #python3 -m pip install numpy numba  etils importlib_resources opencv-python wget gradio 
  #python3 -m pip install tensorflow-2.20.0dev0+selfbuild-cp312-cp312-linux_x86_64.whl

  #This just installs latest version (use it with care)
  python3 -m pip install tensorflow[and-cuda] tensorflow-model-optimization tf_keras numpy numba tensorboard tensorboard-plugin-profile etils importlib_resources tf2onnx onnxruntime onnx opencv-python wget gradio 

  #python3 -m pip install tf-nightly[and-cuda] tensorflow-model-optimization tf_keras numpy numba tensorboard tensorboard-plugin-profile etils importlib_resources tf2onnx onnxruntime onnx opencv-python wget gradio 

  #sudo apt install nvidia-cudnn-cu12 
fi

if [ -d 2d_pose_estimation/ ]; then
  echo "Found model data."
else
  echo "Downloading pretrained model data."
  scripts/downloadPretrained.sh
fi


exit 0
