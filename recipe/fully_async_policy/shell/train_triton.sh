#!/bin/bash
#SBATCH --job-name=dapo_7b_math
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --gpus=8
#SBATCH --partition=gpu-h200-141g-short
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

bash dapo_7b_math_fsdp2_4_4.sh