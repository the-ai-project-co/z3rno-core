#!/usr/bin/env bash
#
# build.sh — local helper for building the z3rno-postgres image.
# Matches what the GitHub Actions workflow runs, but single-arch for speed.
#
# Usage:
#   ./build.sh                  # build for your current arch, tag as :dev
#   ./build.sh --tag 17         # tag as :17 instead of :dev
#   ./build.sh --push           # also push to ghcr.io (requires gh auth)
#   ./build.sh --multi-arch     # build linux/amd64 + linux/arm64 (slower)
#
# Exit codes: 0 = success, non-zero = build or push failure.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE="ghcr.io/the-ai-project-co/z3rno-postgres"
TAG="dev"
PUSH=false
MULTI_ARCH=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --multi-arch)
            MULTI_ARCH=true
            shift
            ;;
        -h|--help)
            sed -n '3,15p' "$0"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

cd "$SCRIPT_DIR"

PLATFORMS="linux/amd64"
if $MULTI_ARCH; then
    PLATFORMS="linux/amd64,linux/arm64"
fi

echo "Building $IMAGE:$TAG for $PLATFORMS ..."

if $PUSH; then
    docker buildx build \
        --platform "$PLATFORMS" \
        --tag "$IMAGE:$TAG" \
        --push \
        .
    echo "Pushed $IMAGE:$TAG to ghcr.io"
else
    docker buildx build \
        --platform "$PLATFORMS" \
        --tag "$IMAGE:$TAG" \
        --load \
        .
    echo "Built $IMAGE:$TAG locally (use --push to upload to ghcr.io)"
fi
