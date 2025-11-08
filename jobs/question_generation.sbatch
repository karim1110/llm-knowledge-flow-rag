#!/bin/bash
#SBATCH --job-name=gen_questions
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/question_generation_%j.out
#SBATCH --error=logs/question_generation_%j.err

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

# Run question generation
python -u scripts/question_generation2.py
