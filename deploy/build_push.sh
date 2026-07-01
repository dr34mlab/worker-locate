#!/usr/bin/env bash
# Build and push worker-locate to GHCR.
# Must be run on a machine with Docker and 'gh auth login' done.
set -euo pipefail

IMAGE="ghcr.io/dr34mlab/worker-locate:latest"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Building $IMAGE ..."
docker buildx build \
    --platform linux/amd64 \
    --push \
    -t "$IMAGE" \
    "$ROOT"

echo "Pushed $IMAGE"
