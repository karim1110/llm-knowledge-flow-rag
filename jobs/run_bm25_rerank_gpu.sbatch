#!/bin/bash
#SBATCH --account=pi-jevans          # Your account (not ssd)
#SBATCH --partition=jevans-gpu       # Your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=300GB
#SBATCH --time=04:00:00
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/bm25_rerank_gpu_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/bm25_rerank_gpu_%j.err

set -euo pipefail

cd /project/jevans/nadav/qaApproach
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

python -u scripts/bm25_retrieval.py --limit 100
