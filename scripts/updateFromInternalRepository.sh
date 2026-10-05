#!/bin/bash
# Pulls the public subset of the internal development repo (RGBToPoseDetect2D)
# into this release snapshot. The internal repo organizes code as an installable
# "ymapnet" package (ymapnet/core, ymapnet/utils, ...); this release mirrors that
# same package layout 1:1 -- imports are NOT rewritten, files are copied verbatim.
#
# Add a new module to a released feature? Add its path to the matching list below
# (and to any transitive ymapnet.* dependency it pulls in that isn't listed yet).

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/.."

SOURCE="../RGBToPoseDetect2D"

if [ ! -d "$SOURCE" ]; then
    echo "ERROR: expected internal repo checkout at $SOURCE" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# ymapnet/ package -- verbatim copy, one file per line so additions are explicit
# ---------------------------------------------------------------------------
PACKAGE_FILES=(
    apps/appFace.py
    apps/appFallDetection.py
    apps/appPoseMatch.py
    apps/runYMAPNet.py
    conversion/convertModelToJAX.py
    conversion/validate_cpp.py
    core/NNCheckpointAveraging.py
    core/NNConverter.py
    core/NNExecutor.py
    core/NNLosses.py
    core/NNModel.py
    core/NNOptimize.py
    core/NNTraining.py
    core/NNTransplant.py
    core/YMAPNet.py
    evaluation/analyzeDepthErrors.py
    evaluation/analyzeHeadAblation.py
    evaluation/analyzeNormalsErrors.py
    evaluation/analyzeSegmentationConfusion.py
    evaluation/evaluateYMAPNet.py
    reporting/illustrate.py
    reporting/plotTrainingProgressToSVG.py
    reporting/statusServer.py
    skeletons/__init__.py
    skeletons/limbs.py
    skeletons/peaks.py
    skeletons/resolve.py
    streams/datasetStream.py
    streams/espStream.py
    streams/folderStream.py
    streams/screenStream.py
    tokens/TokenEstimator.py
    tokens/visualizeTokenConfusion.py
    training/trainFaceIdentification.py
    training/trainGloVeTokensOnly.py
    training/trainTokensOnly.py
    training/trainYMAPNet.py
    utils/calculateNormalsFromDepthmap.py
    utils/createJSONConfiguration.py
    utils/imageProcessing.py
    utils/resolveJointHierarchy.py
    utils/tools.py
    webui/gradioClient.py
    webui/gradioServer.py
)

mkdir -p ymapnet
touch ymapnet/__init__.py
for sub in apps core utils tokens streams webui reporting skeletons training conversion evaluation; do
    mkdir -p "ymapnet/$sub"
    touch "ymapnet/$sub/__init__.py"
done

for f in "${PACKAGE_FILES[@]}"; do
    mkdir -p "ymapnet/$(dirname "$f")"
    cp "$SOURCE/ymapnet/$f" "ymapnet/$f"
done

# ---------------------------------------------------------------------------
# Native DataLoader (C sources are not tracked by this script -- update them
# by hand if datasets/DataLoader/ has drifted; see the internal repo's copy)
# ---------------------------------------------------------------------------
mkdir -p datasets/DataLoader
cp "$SOURCE/datasets/DataLoader/"*.py datasets/DataLoader/ 2>/dev/null || true

# ---------------------------------------------------------------------------
# Public-facing scripts/
# ---------------------------------------------------------------------------
cp "$SOURCE/scripts/downloadPretrained.sh" scripts/
cp "$SOURCE/scripts/downloadModel.sh" scripts/
cp "$SOURCE/scripts/setup.sh" scripts/
cp "$SOURCE/scripts/run_windows.bat" scripts/

# ---------------------------------------------------------------------------
# Other top-level files
# ---------------------------------------------------------------------------
cp "$SOURCE/license.txt" ./
cp "$SOURCE/requirements.txt" ./

exit 0
