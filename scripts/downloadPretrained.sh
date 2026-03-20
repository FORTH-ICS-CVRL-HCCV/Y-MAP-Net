#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..

#V180 is the ICRA 26 model!
scripts/downloadModel.sh 180

exit 0
