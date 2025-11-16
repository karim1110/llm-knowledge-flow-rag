#!/bin/bash
#SBATCH --job-name=embed_questions
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/embed_questions_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/embed_questions_%j.err

set -euo pipefail

# Ensure micromamba on PATH
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

echo "Re-embedding questions with 'query' prompt..."
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Run embedding
cd /project/jevans/nadav/qaApproach
python -u scripts/embed_questions.py

echo "Embedding complete!"
