#!/bin/bash
#SBATCH --job-name=patent_sample
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans
#SBATCH --mem=500GB
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/patent_sample_%j.out
#SBATCH --error=logs/patent_sample_%j.err

set -euo pipefail

mkdir -p logs

# Ensure micromamba on PATH
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
# Initialize micromamba shell hook
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

# Run the Python script
python -u scripts/createPatentSample.py
