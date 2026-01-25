#!/bin/bash
#SBATCH --job-name=bifta
#SBATCH --partition=gpu          # Change to your cluster's GPU partition
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/bifta_%j.out
#SBATCH --error=logs/bifta_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load required modules (adjust according to your cluster)
# module load Anaconda3
# module load CUDA

# Activate conda environment (adjust to your environment name)
# eval "$(conda shell.bash hook)"
# conda activate bifta

# Configuration
MODEL="ViT-B/32"
DEVICE="cuda"
SEED=42
DATASET="imagenet"

# Run BiFTA
python main.py \
    --dataset "$DATASET" \
    --crop_strategy random \
    --model_size "$MODEL" \
    --seed "$SEED" \
    --num_workers 4 \
    --device "$DEVICE"

echo "=========================================="
echo "Experiment finished."
echo "End Time: $(date)"
echo "=========================================="
