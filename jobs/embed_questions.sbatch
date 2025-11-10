#!/bin/bash
#SBATCH --job-name=embed_questions
#SBATCH --account=ssd
#SBATCH --partition=ssd-gpu
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/embed_questions_%j.out
#SBATCH --error=logs/embed_questions_%j.err

# Create temp dir for vLLM
JOB_TMP=/scratch/midway3/$USER/vllm_tmp/$SLURM_JOBID
mkdir -p "$JOB_TMP"

# Point vLLM to temp dir
export TMPDIR="$JOB_TMP"
export TEMP="$JOB_TMP"
export TMP="$JOB_TMP"

echo "Using TMPDIR=$TMPDIR"

# Activate conda environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

# Run embedding
cd /project/jevans/nadav/qaApproach
python scripts/embed_questions.py

# Cleanup temp dir
rm -rf "$JOB_TMP"
