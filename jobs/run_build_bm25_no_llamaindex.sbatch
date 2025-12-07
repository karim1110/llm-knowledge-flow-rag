#!/bin/bash
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans
#SBATCH --cpus-per-task=10        
#SBATCH --mem=500GB                 
#SBATCH --time=31:00:00            
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/build_rank_bm25_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/build_rank_bm25_%j.err

set -euo pipefail

# Change to project directory
cd /project/jevans/nadav/qaApproach

mkdir -p logs

# Ensure micromamba on PATH
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

echo "Starting rank_bm25 index building (10-100x faster than LlamaIndex)..."
echo "Node: $(hostname)"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"
echo "Available Memory: $(free -h | grep Mem: | awk '{print $2}')"
echo ""

python -u scripts/build_bm25_no_llamaindex.py