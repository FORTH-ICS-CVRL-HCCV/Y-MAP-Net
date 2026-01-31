#!/bin/bash

# Ensure version argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

VERSION="$1"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
cd ..

# Activate virtual environment
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Exiting."
    exit 1
fi

ZIPFILE="2d_pose_estimation_v${VERSION}.zip"
URL="http://ammar.gr/ymapnet/archive/${ZIPFILE}"

# Check if .zip file already exists locally
if [ -f "$ZIPFILE" ]; then
    echo "$ZIPFILE already exists. Skipping download."
else
    echo "Downloading $ZIPFILE..."
    wget "$URL" -O "$ZIPFILE"
    if [ $? -ne 0 ]; then
        echo "Download failed. Exiting."
        exit 1
    fi
fi

# Unzip, overwriting if needed
echo "Extracting $ZIPFILE..."
unzip -o "$ZIPFILE"

# Notify user
if command -v notify-send >/dev/null 2>&1; then
    notify-send "Model v$VERSION ready to run"
else
    echo "Model v$VERSION ready to run"
fi

exit 0

