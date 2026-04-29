#!/bin/bash
#SBATCH --job-name=bc_train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load modules
module load anaconda3/2024.10
module load cudatoolkit/12.6

# Activate virtual environment
source activate cos435

# Set data path
export ALFWORLD_DATA=~/COS435_FinalProject/data

# Move to correct directory
cd ~/COS435_FinalProject/src/alfworld_lfm

# Run training
python bc_train.py