#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..


#This is the latest development model!
scripts/downloadModel.sh 262

exit 0
