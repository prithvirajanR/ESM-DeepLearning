#!/bin/bash
#SBATCH --job-name=ROBUST_650M
#SBATCH --output=Robust_650M_%j.out
#SBATCH --error=Robust_650M_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
# 650M is slower, so we give it 24h. E-MLLR is 5x slower than MLLR but 100x faster than PLL.
python -m src.batch_scoring \
    --input_csv data/A4_HUMAN_Seuma_2021.csv \
    --output_csv results/A4_HUMAN_Seuma_2021_ESM2_650M_EnsembleMLLR_predictions.csv \
    --model facebook/esm2_t33_650M_UR50D \
    --model_type esm2 \
    --method EnsembleMLLR
