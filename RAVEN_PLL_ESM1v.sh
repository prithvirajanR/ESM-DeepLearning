#!/bin/bash
#SBATCH --job-name=PLL_ESM1v
#SBATCH --output=%j_PLL_ESM1v.out
#SBATCH --error=%j_PLL_ESM1v.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --time=24:00:00

# 1. Setup Environment
module purge
module load anaconda/3/2021.11
export MY_PROJECT_DIR="/ptmp/$USER/ESM_Project"
export ENV_DIR="$MY_PROJECT_DIR/env"
source activate "$ENV_DIR"
export HF_HOME="/ptmp/$USER/hf_cache"

mkdir -p $MY_PROJECT_DIR
mkdir -p $HF_HOME
cd $SLURM_SUBMIT_DIR

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
INPUT_FILE="A4_HUMAN_Seuma_2021.csv"
MODEL_NAME="facebook/esm1v_t33_650M_UR90S_1"
METHOD="PLL"
TAG="_ESM1v_650M_PLL"

OUTPUT_FILE="results/${INPUT_FILE%.*}${TAG}_predictions.csv"
# -----------------------------------------------------------------------------

echo "Starting PLL Run for ${TAG}..."

python -m src.batch_scoring \
    --input_csv "data/$INPUT_FILE" \
    --output_csv "$OUTPUT_FILE" \
    --method $METHOD \
    --model $MODEL_NAME \
    --batch_size 128

echo "Scoring Done. (Analysis will be run manually later)."
