#!/bin/bash
#SBATCH --job-name=ESM1v_Entropy
#SBATCH --output=%j_ESM1v_Entropy.out
#SBATCH --error=%j_ESM1v_Entropy.err
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
    --model facebook/esm1v_t33_650M_UR90S_1 \
    --method EntropyMLLR \
    --output_csv results/A4_HUMAN_Seuma_2021_ESM1v_EntropyMLLR_predictions.csv

# Run Analysis immediately after
python -m src.analysis \
    --results_csv results/A4_HUMAN_Seuma_2021_ESM1v_EntropyMLLR_predictions.csv \
    --truth_csv data/A4_HUMAN_Seuma_2021.csv \
    --output_report results/report_EntropyMLLR_ESM1v.png
