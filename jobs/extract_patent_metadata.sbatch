#!/bin/bash
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans
#SBATCH --job-name=extract_patent_metadata
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/extract_patent_metadata_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/extract_patent_metadata_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail

# Ensure micromamba on PATH
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

cd /project/jevans/nadav/qaApproach
python -u scripts/extract_patent_metadata.py
