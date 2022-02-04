#!/bin/bash
#SBATCH --job-name=machine_learning_cmp
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=64
#SBATCH --partition=full

source /home/vitornpad/pascal_analyzer/venv/bin/activate

python main.py
