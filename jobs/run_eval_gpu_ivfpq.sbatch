#!/bin/bash
#SBATCH --account=ssd
#SBATCH --partition=ssd-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=ssd
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/eval_gpu_ivfpq_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/eval_gpu_ivfpq_%j.err

set -euo pipefail

cd /project/jevans/nadav/qaApproach

mkdir -p logs

export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

echo "Starting RAG Evaluation - GPU IVF-PQ Index"
echo "GPU info:"
nvidia-smi

python -u scripts/evaluate_gpu_ivfpq.py
