#!/bin/bash
#SBATCH --job-name=my_job_pwds              # Name of the job
#SBATCH --output=pwd_outputs/diff_distances.%j_output_half.txt     # Standard output and error log
#SBATCH --error=pwd_outputs/diff%x.%j.err
#SBATCH --ntasks=1                     # Run on a single task
#SBATCH --mem=40G                      # Total memory
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1                   # replace with number of GPUs required
#SBATCH --time=168:00:00               # Infinite time (0 days, 0 hours, 0 minutes)

# Initialize Conda
eval "$(conda shell.bash hook)"

conda activate /scratch/htc/fsafarov/openmm_ff
python calculate_PWDs_1.py
