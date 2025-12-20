#!/bin/bash
#SBATCH --job-name=verify_paragraphs
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans
#SBATCH --mem=64GB
#SBATCH --time=31:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/verify_paragraphs_%j.out
#SBATCH --error=logs/verify_paragraphs_%j.err

set -euo pipefail

mkdir -p logs

# Ensure micromamba on PATH
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

# Load .env file
set -a
source .env
set +a

# Run verification
# Using -u for unbuffered output to see progress in logs
python -u scripts/verify_paragraphs.py \
    --input data/question_generation/retrieval_results/retrieval_results_understanding_merged_all_patents.json
