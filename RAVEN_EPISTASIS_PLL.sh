#!/bin/bash
#SBATCH --job-name=Epistasis_PLL
#SBATCH --output=%j_Epistasis_PLL.out
#SBATCH --error=%j_Epistasis_PLL.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1

# Load Environment
module unload python-waterboa/2024.06
module load anaconda/3/2021.11
source activate /ptmp/$USER/ESM_Project/env

if [ -d "/ptmp/$USER/ESM_Project" ]; then
    cd /ptmp/$USER/ESM_Project
fi

# Run Scoring (Includes Doubles)
# Using the Best Probabilistic Model: ESM-1v
python -m src.batch_scoring \
    --input_csv data/A4_HUMAN_Seuma_2021.csv \
    --model facebook/esm1v_t33_650M_UR90S_1 \
    --method PLL \
    --output_csv results/Epistasis_PLL_ESM1v.csv

# Run Epistasis Calculation
python -m src.epistasis \
    --input_csv results/Epistasis_PLL_ESM1v.csv \
    --output_csv results/Epistasis_PLL_ESM1v_Analysis.csv
