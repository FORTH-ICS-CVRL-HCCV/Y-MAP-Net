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



# Strip any trailing letter suffix (e.g. "283a" -> "283") for numeric comparison
VERSION_NUM="${VERSION%%[a-zA-Z]*}"
if [ "$VERSION_NUM" -lt 285 ]; then
    ZIPFILE="2d_pose_estimation_v${VERSION}.zip"
else
    ZIPFILE="ymapnet_model_v${VERSION}.zip"
fi



# Define URLs to try
URLS=(
    "http://ammar.gr/ymapnet/archive/${ZIPFILE}"
    "http://anoiksi.ammar.gr/ymapnet/archive/${ZIPFILE}"
)
# Check if .zip file already exists locally
if [ -f "$ZIPFILE" ]; then
    echo "$ZIPFILE already exists. Skipping download."
else
    echo "Downloading $ZIPFILE..."
    DOWNLOAD_SUCCESS=0
    
    # Try each URL
    for URL in "${URLS[@]}"; do
        echo "Trying: $URL"
        wget "$URL" -O "$ZIPFILE"
        if [ $? -eq 0 ]; then
            echo "Download successful from $URL"
            DOWNLOAD_SUCCESS=1
            break
        else
            echo "Failed to download from $URL, trying next..."
            rm -f "$ZIPFILE"  # Clean up incomplete download
        fi
    done
    
    if [ $DOWNLOAD_SUCCESS -eq 0 ]; then
        echo "Download failed from all sources. Exiting."
        exit 1
    fi
fi
# Unzip, overwriting if needed
echo "Extracting $ZIPFILE..."
unzip -o "$ZIPFILE"
# Accumulate tensorboard logs before notifying
TENSORBOARD_SRC="2d_pose_estimation/tensorboard"
if [ -d "$TENSORBOARD_SRC" ] && [ -n "$(ls -A "$TENSORBOARD_SRC" 2>/dev/null)" ]; then
    mkdir -p logs
    mv "$TENSORBOARD_SRC"/* logs/
    echo "Tensorboard logs moved to logs/"
fi
# Notify user
if command -v notify-send >/dev/null 2>&1; then
    notify-send "Model v$VERSION ready to run"
else
    echo "Model v$VERSION ready to run"
fi
exit 0
