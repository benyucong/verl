#!/bin/bash
#SBATCH --job-name=benchmark_mq
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=24
#SBATCH --ntasks=1
#SBATCH --mem=50GB
#SBATCH --partition=batch-csl

python benchmark_message_queues.py \
    --num-producers 4 \
    --samples-per-producer 2000 \
    --num-shards 4