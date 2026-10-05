#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..

#V180 is the ICRA 26 model! (scripts/downloadModel.sh 180)
#V288 is the current default model
scripts/downloadModel.sh 288

exit 0
