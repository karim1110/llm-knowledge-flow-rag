#!/bin/bash
#SBATCH --job-name=retrieval_only
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/retrieval_only_%j.out
#SBATCH --error=logs/retrieval_only_%j.err

# Activate conda environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

# Use custom FAISS build for sm90 GPUs (only for jevans-gpu partition)
export PYTHONPATH=/project/jevans/nadav/qaApproach/faiss_build/faiss/build/faiss/python:$PYTHONPATH
echo "Using custom FAISS build for sm90"

# Run retrieval
cd /project/jevans/nadav/qaApproach
python scripts/retrieval_only.py --limit 100
