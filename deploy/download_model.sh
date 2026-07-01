#!/usr/bin/env bash
# Download nvidia/LocateAnything-3B to the RunPod network volume.
# Run this once inside a RunPod pod with the locate-models volume mounted at /workspace.
# The model is NOT gated — no HF token needed.
set -euo pipefail

DEST=/workspace/models/locate-anything

if [ -d "$DEST/config.json" ] || [ -f "$DEST/config.json" ]; then
    echo "Model already present at $DEST — skipping download."
    exit 0
fi

mkdir -p "$DEST"

echo "Downloading nvidia/LocateAnything-3B to $DEST ..."
HF_XET_HIGH_PERFORMANCE=1 hf download nvidia/LocateAnything-3B \
    --local-dir "$DEST" \
    --exclude "assets/*" "*.mp4"

echo "Download complete. Verifying config.json ..."
ls -lh "$DEST/config.json"
echo "Done."
