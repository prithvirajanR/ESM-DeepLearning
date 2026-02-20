#!/bin/bash
#SBATCH --job-name=ROBUST_ESM1v
#SBATCH --output=Robust_ESM1v_%j.out
#SBATCH --error=Robust_ESM1v_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

# Load modules
module load anaconda/3/2021.11

# Activate environment
source activate /ptmp/$USER/ESM_Project/env

# Set environment variables
export MY_PROJECT_DIR="/raven/ptmp/$USER/ESM_Project"
export ENV_DIR="$MY_PROJECT_DIR/env"
export PYTHONPATH="$MY_PROJECT_DIR:$PYTHONPATH"
export HF_HOME="$MY_PROJECT_DIR/cache/huggingface"
export TORCH_HOME="$MY_PROJECT_DIR/cache/torch"

# Run Batch Scoring using EnsembleMLLR
python -m src.batch_scoring \
    --input_csv data/A4_HUMAN_Seuma_2021.csv \
    --output_csv results/A4_HUMAN_Seuma_2021_ESM1v_EnsembleMLLR_predictions.csv \
    --model facebook/esm1v_t33_650M_UR90S_1 \
    --model_type esm1v \
    --method EnsembleMLLR
