#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=50GB
#SBATCH --partition=batch-csl

cd /scratch/cs/adis/yuc10/verl/
python -m recipe.fully_async_policy.unittest.test_message_queue_new