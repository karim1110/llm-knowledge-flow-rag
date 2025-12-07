#!/bin/bash
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --time=4:00:00
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/load_paragraph_texts_bm25_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/load_paragraph_texts_bm25_%j.err
#SBATCH --job-name=load_texts_bm25

set -euo pipefail

# Load Spark module
module load spark

# Ensure micromamba on PATH
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

# Add Spark Python libraries to PYTHONPATH
export PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9.5-src.zip:${PYTHONPATH:-}"

# Set up Spark configuration
export MASTER="local[${SLURM_CPUS_PER_TASK}]"
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)

cd /project/jevans/nadav/qaApproach
mkdir -p logs

echo "Starting: Load Paragraph Texts (BM25 retrieval results)"
echo "Input: data/question_generation/retrieval_results/retrieval_results_bm25_hierarchical.json"
echo "Node: $(hostname)"
echo ""

python -u scripts/load_paragraph_texts.py \
  --input data/question_generation/retrieval_results/retrieval_results_bm25_hierarchical.json \
  --output data/question_generation/retrieval_results/retrieval_results_bm25_merged.json \
  --merge-context
