#!/bin/bash
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=128GB
#SBATCH --time=2:00:00
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/bm25_stage2_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/bm25_stage2_%j.err
#SBATCH --job-name=bm25_stage2

set -euo pipefail

cd /project/jevans/nadav/qaApproach
mkdir -p logs

export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

echo "Starting BM25 Stage 2: GPU Reranker"
echo "GPU info:"
nvidia-smi
echo ""

python -u scripts/bm25_retrieval_stage2.py
