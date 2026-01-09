#!/bin/bash
#SBATCH --job-name=LLR_double_mutation              # Name of the job
#SBATCH --output=outputs/%x.%j_output.txt     # Standard output and error log
#SBATCH --error=outputs/%x.%j.err
#SBATCH --ntasks=1                     # Run on a single task
#SBATCH --mem=80G                      # Total memory
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1                   # replace with number of GPUs required
#SBATCH --time=168:00:00               # Infinite time (0 days, 0 hours, 0 minutes)

# Initialize Conda
eval "$(conda shell.bash hook)"

conda activate /scratch/htc/fsafarov/openmm_ff
python MLM_5.py
