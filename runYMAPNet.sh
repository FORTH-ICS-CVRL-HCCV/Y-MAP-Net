#!/bin/bash


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [[ $* == *--collab* ]]
then 
 echo "Using collab mode"
else
 source venv/bin/activate
fi

#Use the full real-estate of the screen!
QT_AUTO_SCREEN_SCALE_FACTOR=0 QT_SCALE_FACTOR=1 python3 runYMAPNet.py $@

exit 0
