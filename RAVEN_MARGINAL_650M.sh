#!/bin/bash
#SBATCH --job-name=ESM2_650M_Marginal
#SBATCH --output=%j_ESM2_650M_Marginal.out
#SBATCH --error=%j_ESM2_650M_Marginal.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1

# Load Environment
module load anaconda/3/2021.11
source activate /ptmp/$USER/ESM_Project/env

if [ -d "/ptmp/$USER/ESM_Project" ]; then
    cd /ptmp/$USER/ESM_Project
fi

# Run Scoring
python -m src.batch_scoring \
    --input_csv data/A4_HUMAN_Seuma_2021.csv \
    --model facebook/esm2_t33_650M_UR50D \
    --method MutantMarginal \
    --output_csv results/A4_HUMAN_Seuma_2021_ESM2_650M_MutantMarginal_predictions.csv

# Run Analysis immediately after
python -m src.analysis \
    --results_csv results/A4_HUMAN_Seuma_2021_ESM2_650M_MutantMarginal_predictions.csv \
    --truth_csv data/A4_HUMAN_Seuma_2021.csv \
    --output_report results/report_MutantMarginal_650M.png
