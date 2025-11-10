#!/bin/bash
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans
#SBATCH --cpus-per-task=10
#SBATCH --mem=1350GB
#SBATCH --time=6:00:00
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/build_faiss_index_v2_cpu_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/build_faiss_index_v2_cpu_%j.err

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

echo "Starting FAISS index building (CPU version)..."
echo "Node: $(hostname)"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"
echo ""

python -u scripts/build_faiss_index_v2_cpu.py
