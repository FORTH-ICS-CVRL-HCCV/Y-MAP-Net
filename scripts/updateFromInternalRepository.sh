#!/bin/bash


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..

SOURCE="../Y-MAP-Net-Development-Repository/"

mkdir scripts/
cp "$SOURCE/scripts/downloadPretrained.sh" scripts/ 
cp "$SOURCE/scripts/downloadModel.sh" scripts/
cp "$SOURCE/scripts/setup.sh" scripts/ 


mkdir datasets/

#Dataloader
mkdir -p datasets/DataLoader
mkdir -p datasets/DataLoader/codecs
mkdir -p datasets/DataLoader/processing
mkdir -p datasets/DataLoader/processing/AVX2
cp $SOURCE/datasets/DataLoader/*.py datasets/DataLoader/
cp $SOURCE/datasets/DataLoader/*.c datasets/DataLoader/
cp $SOURCE/datasets/DataLoader/*.h datasets/DataLoader/
cp $SOURCE/datasets/DataLoader/*.sh datasets/DataLoader/
cp $SOURCE/datasets/DataLoader/codecs/*.c datasets/DataLoader/codecs/
cp $SOURCE/datasets/DataLoader/codecs/*.h datasets/DataLoader/codecs/
cp $SOURCE/datasets/DataLoader/processing/*.c datasets/DataLoader/processing/
cp $SOURCE/datasets/DataLoader/processing/*.h datasets/DataLoader/processing/
cp $SOURCE/datasets/DataLoader/processing/AVX2/*.c datasets/DataLoader/processing/AVX2/
cp $SOURCE/datasets/DataLoader/processing/AVX2/*.h datasets/DataLoader/processing/AVX2/



#Runtime
cp "$SOURCE/NNConverter.py" ./
cp "$SOURCE/NNExecutor.py" ./
cp "$SOURCE/NNLosses.py" ./
cp "$SOURCE/NNModel.py" ./
cp "$SOURCE/NNOptimize.py" ./
cp "$SOURCE/NNTraining.py" ./
cp "$SOURCE/NNTransplant.py" ./
cp "$SOURCE/PoseEstimator2D.py" ./
cp "$SOURCE/TokenEstimator.py" ./
cp "$SOURCE/createJSONConfiguration.py" ./
cp "$SOURCE/espStream.py" ./
cp "$SOURCE/datasets/calculateNormalsFromDepthmap.py" ./ 
cp "$SOURCE/folderStream.py" ./
cp "$SOURCE/gradioClient.py" ./
cp "$SOURCE/gradioServer.py" ./
cp "$SOURCE/illustrate.py" ./
cp "$SOURCE/imageProcessing.py" ./
cp "$SOURCE/license.txt" ./
cp "$SOURCE/resolveJointHierarchy.py" ./
cp "$SOURCE/runYMAPNet.py" ./
cp "$SOURCE/screenStream.py" ./
cp "$SOURCE/tools.py" ./
	 

#Training code
cp "$SOURCE/trainYMAPNet.py" ./
cp "$SOURCE/trainTokensOnly.py" ./

#Other..
cp "$SOURCE/requirements.txt" ./

cp 

exit 0
