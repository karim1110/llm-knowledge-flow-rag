#!/bin/bash
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans
#SBATCH --cpus-per-task=5
#SBATCH --mem=500GB
#SBATCH --time=10:00:00
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/bm25_stage1_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/bm25_stage1_%j.err
#SBATCH --job-name=bm25_stage1

set -euo pipefail

cd /project/jevans/nadav/qaApproach
mkdir -p logs

export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

echo "Starting BM25 Stage 1: CPU-only (BM25 + Semantic Reranking)"
echo "Node: $(hostname)"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"
echo ""

python -u scripts/bm25_retrieval_stage1.py
