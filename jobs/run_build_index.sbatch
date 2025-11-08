#!/bin/bash
#SBATCH --account=ssd
#SBATCH --partition=ssd-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=ssd
#SBATCH --cpus-per-task=12
#SBATCH --mem=128GB
#SBATCH --time=36:00:00
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/build_faiss_index_v2_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/build_faiss_index_v2_%j.err

set -euo pipefail

# Change to project directory
cd /project/jevans/nadav/qaApproach

mkdir -p logs

# Ensure micromamba on PATH (installed in $HOME/bin earlier)
export PATH="$HOME/bin:$PATH"
# Provide a sane default for MAMBA_ROOT_PREFIX (prevents unbound-var with set -u)
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
# Initialize micromamba shell hook so we can use `micromamba activate` in this script
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

echo "Starting FAISS index building (IVF GPU version)..."
echo "GPU info:"
nvidia-smi

python -u scripts/build_faiss_index_ivf.py
