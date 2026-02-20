#!/bin/bash
#SBATCH --job-name=Epistasis_EDS
#SBATCH --output=%j_Epistasis_EDS.out
#SBATCH --error=%j_Epistasis_EDS.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1

# Load Environment
module unload python-waterboa/2024.06
module load anaconda/3/2021.11
source activate /ptmp/$USER/ESM_Project/env

# Go to project root (assume submission from project root or use explicit path)
# cd $SLURM_SUBMIT_DIR # Optional, Slurm defaults to submit dir
# Ensure we are in the right place if submitted from elsewhere
if [ -d "/ptmp/$USER/ESM_Project" ]; then
    cd /ptmp/$USER/ESM_Project
fi

# Run Scoring (Includes Doubles)
# Using our Champion Model: ESM-2 650M
python -m src.batch_scoring \
    --input_csv data/A4_HUMAN_Seuma_2021.csv \
    --model facebook/esm2_t33_650M_UR50D \
    --method EDS \
    --output_csv results/Epistasis_EDS_650M.csv

# Run Epistasis Calculation
python -m src.epistasis \
    --input_csv results/Epistasis_EDS_650M.csv \
    --output_csv results/Epistasis_EDS_650M_Analysis.csv
