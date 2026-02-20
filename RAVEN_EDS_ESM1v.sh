#!/bin/bash
#SBATCH --job-name=EDS_ESM1v
#SBATCH --output=%j_EDS_ESM1v.out
#SBATCH --error=%j_EDS_ESM1v.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
OUTPUT_FILE="results/${INPUT_FILE%.*}_ESM1v_EDS_predictions.csv"

# 3. Execution
echo "Running EDS with ESM-1v..."
python -m src.batch_scoring \
    --input_csv "data/$INPUT_FILE" \
    --output_csv "$OUTPUT_FILE" \
    --method EDS \
    --model facebook/esm1v_t33_650M_UR90S_1 \
    --model_type esm1v \
    --batch_size 1 \
    --reference_csv data/ProteinGym_reference_file_substitutions.csv

echo "Done!"
