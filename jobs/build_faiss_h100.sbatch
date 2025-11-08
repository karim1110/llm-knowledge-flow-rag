#!/bin/bash
#SBATCH --account=ssd
#SBATCH --partition=ssd-gpu
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/build_faiss_h100_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/build_faiss_h100_%j.err
#SBATCH --job-name=build_faiss_h100

set -euo pipefail

# Change to project directory
cd /project/jevans/nadav/qaApproach

mkdir -p logs

echo "=== Building FAISS with H100 support ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Show GPU info
nvidia-smi

echo ""
bash scripts/build_faiss_h100.sh
