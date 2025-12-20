#!/bin/bash
#SBATCH --account=pi-jevans
#SBATCH --partition=jevans
#SBATCH --job-name=extract_metadata_spark
#SBATCH --output=/project/jevans/nadav/qaApproach/logs/extract_metadata_spark_%j.out
#SBATCH --error=/project/jevans/nadav/qaApproach/logs/extract_metadata_spark_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=16

set -euo pipefail

# Load required modules
module load spark

# Ensure micromamba on PATH
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate rag-py310

# Add Spark Python libraries to PYTHONPATH
export PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9.5-src.zip:${PYTHONPATH:-}"

echo "Starting PySpark metadata extraction..."
echo "Node: $(hostname)"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"
echo "SPARK_HOME: $SPARK_HOME"
echo ""

# Set up Spark configuration for local mode
export MASTER="local[${SLURM_CPUS_PER_TASK}]"
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)

# Run the extraction
cd /project/jevans/nadav/qaApproach
python -u scripts/extract_paper_metadata_spark.py

echo "PySpark metadata extraction complete!"
