#!/usr/bin/env bash
module load gcc/11.5.0 cudnn/9.2.0.82-12 cuda/12.8.0
set -euo pipefail

# Minimal release preparation + build/install helper.
# Usage: run from repo root or call from a configured build directory.

# Optional: set BUILD_DIR env var or pass via BUILD_DIR env.
BUILD_DIR=${BUILD_DIR:-build}
JOBS=${JOBS:-$(nproc)}

echo "Preparing release: build_dir=$BUILD_DIR, jobs=$JOBS"

if [[ -z "${CUDA_ROOT:-}" ]]; then
	echo "WARNING: CUDA_ROOT is not set. Set CUDA_ROOT to your CUDA toolkit path (e.g. /usr/local/cuda)" >&2
fi
if [[ -z "${CUDNN_ROOT:-}" ]]; then
	echo "WARNING: CUDNN_ROOT is not set. Set CUDNN_ROOT to your cuDNN path" >&2
fi

# Create and configure build directory
mkdir -p "$BUILD_DIR"
pushd "$BUILD_DIR" >/dev/null

echo "Configuring CMake (Release)..."
cmake -DCMAKE_BUILD_TYPE=Release ..

echo "Building and installing..."
cmake --build . --target install -- -j"$JOBS"

popd >/dev/null

echo "Install complete. Shared libraries should be installed to the CMake install prefix (project defaults install to src/jaxmg/bin when configured accordingly)."