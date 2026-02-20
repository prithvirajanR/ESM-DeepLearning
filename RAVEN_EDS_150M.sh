#!/bin/bash
#SBATCH --job-name=EDS_150M
#SBATCH --output=%j_EDS_150M.out
#SBATCH --error=%j_EDS_150M.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1

# 1. Environment Setup
module load anaconda/3/2021.11

MY_PROJECT_DIR="/ptmp/$USER/ESM_Project"
ENV_DIR="$MY_PROJECT_DIR/env"
export HF_HOME="/ptmp/$USER/hf_cache"

echo "Activating Environment at $ENV_DIR"
source activate "$ENV_DIR"

# 2. Paths
INPUT_FILE="A4_HUMAN_Seuma_2021.csv"
OUTPUT_FILE="results/${INPUT_FILE%.*}_ESM2_150M_EDS_predictions.csv"

# 3. Execution
echo "Running EDS with ESM2 150M..."
python -m src.batch_scoring \
    --input_csv "data/$INPUT_FILE" \
    --output_csv "$OUTPUT_FILE" \
    --method EDS \
    --model facebook/esm2_t30_150M_UR50D \
    --batch_size 1 \
    --reference_csv data/ProteinGym_reference_file_substitutions.csv

echo "Done!"
