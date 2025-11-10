#!/bin/bash
#SBATCH --job-name=retrieval_only
#SBATCH --account=ssd
#SBATCH --partition=ssd-gpu
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/retrieval_only_%j.out
#SBATCH --error=logs/retrieval_only_%j.err

# Activate conda environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

# Run retrieval
cd /project/jevans/nadav/qaApproach
python scripts/retrieval_only.py
