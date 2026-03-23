# Y-MAP-Net

This repository aims to provide a 2D pose estimator in real-time using a webcam and a pre-trained 2D Pose Estimation model. It utilizes TensorFlow for model inference.

## Prerequisites

- Python 3
- TensorFlow 2.16.1+
- Keras 3+
- NumPy
- OpenCV
- wget (for downloading the model, optional)

These can be installed by running [python3 scripts/setup.sh](https://github.com/FORTH-ICS-CVRL-HCCV/RGBToPoseDetect2D/blob/main/scripts/setup.sh) which will also create a venv that can later be activated using source venv/bin/activate 

To make a docker file and then mount it use :
``` 
docker/build_and_deploy.sh
docker run rgbposedetect2d-container
docker attach rgbposedetect2d-container
cd workspace
python3 scripts/setup.sh
source venv/bin/activate
```


Ensure that you have installed the required dependencies. You can install them using :
``` 
scripts/setup.sh 
```

To download a pre-trained model use :
``` 
scripts/downloadPretrained.sh
```

To run use :
``` 
./runYMAPNet.sh
``` 


To train use :
``` 
python3 createJSONConfiguration.py
#Modify the 2d_pose_estimation/configuration.json
datasets/DataLoader/makeLibrary.sh 
python3 trainYMAPNet.py
```


