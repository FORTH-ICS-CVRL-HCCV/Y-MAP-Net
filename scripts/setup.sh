#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..


USE_VENV=1
if [[ $* == *--collab* ]]
then 
 echo "Using non-blocking collab mode"
 USE_VENV=0
fi



if [ -d venv/ ]
then
echo "Found a virtual environment" 

if [ "$USE_VENV" -eq "0" ]; then
   source venv/bin/activate
else
   echo "Did not find an existing venv.."
fi

else 
echo "Creating a virtual environment"
#Simple dependency checker that will apt-get stuff if something is missing
# sudo apt-get install python3-venv python3-pip
SYSTEM_DEPENDENCIES="python3-venv python3-pip zip wget libhdf5-dev libzstd-dev rsync numactl"

for REQUIRED_PKG in $SYSTEM_DEPENDENCIES
do
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo "Checking for $REQUIRED_PKG: $PKG_OK"
if [ "" = "$PKG_OK" ]; then

  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."

  #If this is uncommented then only packages that are missing will get prompted..
  #sudo apt-get --yes install $REQUIRED_PKG

  #if this is uncommented then if one package is missing then all missing packages are immediately installed..
  sudo apt-get install $SYSTEM_DEPENDENCIES  
  break
fi
done
#------------------------------------------------------------------------------

if [ "$USE_VENV" -eq "0" ]; then
   echo "Creating a venv.."
   python3 -m venv venv
   source venv/bin/activate
else
   echo "We assume collab machines so not using venv.."
fi


fi 


#https://www.tensorflow.org/install/source#linux
python3 -m pip install --upgrade pip
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
#tensorflow 


if [ -d 2d_pose_estimation/ ]
then
echo "Found model data." 
else
echo "Downloading pretrained model data." 
scripts/downloadPretrained.sh
fi


exit 0
