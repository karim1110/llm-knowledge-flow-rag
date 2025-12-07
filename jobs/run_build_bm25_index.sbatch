#!/bin/bash
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans
#SBATCH --cpus-per-task=10
#SBATCH --mem=790GB
#SBATCH --time=30:00:00
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/build_bm25_llamaindex_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/build_bm25_llamaindex_%j.err

set -euo pipefail

# Change to project directory
cd /project/jevans/nadav/qaApproach

mkdir -p logs

# Ensure micromamba on PATH (installed in $HOME/bin earlier)
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

echo "Starting BM25/LlamaIndex index building (CPU version)..."
echo "Node: $(hostname)"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"
echo ""

python -u scripts/bm25_llamaindex.py