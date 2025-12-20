#!/bin/bash
#SBATCH --job-name=verify_remembering_array
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans
#SBATCH --mem=32GB
#SBATCH --time=31:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --output=logs/verify_remembering_array_%A_%a.out
#SBATCH --error=logs/verify_remembering_array_%A_%a.err

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

# Run verification with sharding
# Using -u for unbuffered output to see progress in logs
python -u scripts/verify_paragraphs.py \
    --input data/question_generation/retrieval_results/retrieval_results_remembering_merged_all_patents.json \
    --shard-index $SLURM_ARRAY_TASK_ID \
    --total-shards 10
