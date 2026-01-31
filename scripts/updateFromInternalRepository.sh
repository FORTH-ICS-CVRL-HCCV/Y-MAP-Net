#!/bin/bash


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..

SOURCE="../RGBToPoseDetect2D/"

mkdir scripts/
cp "$SOURCE/scripts/downloadPretrained.sh" scripts/ 
cp "$SOURCE/scripts/setup.sh" scripts/ 


mkdir datasets/
cp "$SOURCE/datasets/calculateNormalsFromDepthmap.py" datasets/ 





cp 

exit 0
