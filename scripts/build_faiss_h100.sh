#!/bin/bash
# Build FAISS from source with H100 (Hopper sm_90) support

set -euo pipefail

# Activate conda environment
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

# Set up build directory
BUILD_DIR="$HOME/faiss_build"
mkdir -p $BUILD_DIR

echo "=== Building FAISS with H100 (sm_90) support ==="
echo "Build directory: $BUILD_DIR"
echo "Conda env: rag-py310"
echo ""

# Clone FAISS on login node BEFORE submitting job!
# This script assumes faiss repo is already cloned
if [ ! -d "$BUILD_DIR/faiss" ]; then
    echo "ERROR: FAISS repository not found at $BUILD_DIR/faiss"
    echo "Please run this first on login node:"
    echo "  mkdir -p $HOME/faiss_build"
    echo "  cd $HOME/faiss_build"
    echo "  git clone https://github.com/facebookresearch/faiss.git"
    exit 1
fi

cd $BUILD_DIR/faiss

# Load CUDA module
echo "Loading CUDA module..."
module load cuda/12.1
echo "CUDA loaded: $(which nvcc)"
echo "CUDA version: $(nvcc --version | grep release)"
echo ""

# Get conda environment path
CONDA_PREFIX=$(python -c "import sys; print(sys.prefix)")
echo "Conda prefix: $CONDA_PREFIX"
echo "Python: $(which python)"
echo ""

# Install build dependencies if needed (skip on compute nodes without internet)
# Note: Install these on login node first: micromamba install -y cmake make swig numpy
echo "Checking build dependencies..."
which cmake || echo "WARNING: cmake not found"
which make || echo "WARNING: make not found"
echo ""

# Remove old build directory if it exists
if [ -d "build" ]; then
    echo "Removing old build directory..."
    rm -rf build
fi

# Configure CMake with H100 support
echo "Configuring CMake with H100 (sm_90) support..."
mkdir build && cd build

cmake .. \
    -DFAISS_ENABLE_GPU=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80;90" \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DPython_EXECUTABLE=$(which python)

echo ""
echo "CMake configuration complete!"
echo ""

# Build FAISS
echo "Building FAISS with $(nproc) cores (this will take 15-30 minutes)..."
make -j $(nproc)

echo ""
echo "Build complete! Installing Python bindings..."
echo ""

# Install Python bindings
cd ../faiss/python
python setup.py install

echo ""
echo "=== FAISS build complete! ==="
echo ""
echo "Testing installation..."
python -c "import faiss; print('FAISS module location:', faiss.__file__)"

echo ""
echo "✓ FAISS with H100 support successfully installed!"
echo "Note: GPU detection requires running on a compute node with GPU access"
