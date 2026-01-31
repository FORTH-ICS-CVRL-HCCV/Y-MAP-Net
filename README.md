# Y-MAP-Net
Y-MAP-Net: Real-time depth, normals, segmentation, multi-label captioning and 2D human pose in RGB images , Internatonal Conference on Robotics and Automation (ICRA) 2026


![Illustration](https://github.com/FORTH-ICS-CVRL-HCCV/Y-MAP-Net/blob/main/doc/illu.png?raw=true)

This repository aims to provide a neural network that provides: 2D pose estimation, Depth, Normals, Segmentations and Token descriptions from RGB images in real-time using a webcam and a pre-trained Y-MAP-Net model. 
It only depends on Keras/TensorFlow and OpenCV for model inference.


One click deployment in Google Collab : [![Open Y-MAP-Net In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FORTH-ICS-CVRL-HCCV/Y-MAP-Net/blob/main/scripts/y-map-net.ipynb)

## Prerequisites

- Python 3
- TensorFlow >= 2.16.1
- Keras >= 3 
- NumPy
- OpenCV
- wget (for downloading the model, optional)

These can be installed by running [python3 scripts/setup.sh](https://github.com/FORTH-ICS-CVRL-HCCV/Y-MAP-Net/blob/main/scripts/setup.sh) which will also create a venv that can later be activated using source venv/bin/activate 


## Setup 

Ensure that you have installed the required dependencies. You can install them using :
``` 
scripts/setup.sh 
```

## Running

To run use :
```
./runYMAPNet.sh
```

## Training
 
Since preparing the involved datasets to train the project is very complicated..
We need some time before the full training code is polished enough to be redistributed..
Thank you for your patience!



## More Information


![Illustration](https://github.com/FORTH-ICS-CVRL-HCCV/Y-MAP-Net/blob/main/doc/Y-Net.png?raw=true)


