#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..

DLNAME="2d_pose_estimation_v180.zip"

rm $DLNAME
wget http://ammar.gr/ymapnet/archive/$DLNAME
unzip -o $DLNAME

exit 0
