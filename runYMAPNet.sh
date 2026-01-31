#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"


if [[ $* == *--collab* ]]
then 
 echo "Using collab mode"
else
 source venv/bin/activate
fi

python3 runYMAPNet.py $@

exit 0
