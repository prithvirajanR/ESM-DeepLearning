#!/bin/bash
#SBATCH --job-name=ESM_Scoring
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"      # Required for GPU nodes
#SBATCH --gres=gpu:a100:1       # Request 1 A100 GPU
#SBATCH --cpus-per-task=18      # Recommended ratio (72 cores / 4 GPUs = 18)
#SBATCH --mem=125000            # Recommended memory (500GB / 4 GPUs = 125GB)
#SBATCH --time=24:00:00         # Max runtime 24 hours

# -----------------------------------------------------------------------------
# RAVEN HPC SUBMISSION SCRIPT
# -----------------------------------------------------------------------------

# 1. Setup Environment
module purge
module load anaconda/3/2021.11

# Create/Activate Virtual Env in /ptmp
# NOTE: We assume you ran SETUP_ENV.sh first!
export MY_PROJECT_DIR="/ptmp/$USER/ESM_Project"
export ENV_DIR="$MY_PROJECT_DIR/env"

echo "🔌 Activating Environment: $ENV_DIR"
source activate "$ENV_DIR"

export HF_HOME="/ptmp/$USER/hf_cache"

mkdir -p $MY_PROJECT_DIR
mkdir -p $HF_HOME

echo "Running on node: $(hostname)"
echo "Project Directory: $MY_PROJECT_DIR"

# 2. Copy Code to Scratch (Improvement: Sync code to fast storage)
# Try to sync from where you submitted the job (SLURM_SUBMIT_DIR)
# rsync -av --exclude 'model_cache' $SLURM_SUBMIT_DIR/ $MY_PROJECT_DIR/

cd $SLURM_SUBMIT_DIR

# 3. Install Dependencies (First time only)
# Ideally, you create the environment once interactively, but here is a safe check.
# pip install -r requirements.txt

# -----------------------------------------------------------------------------
# ⚙️ USER CONFIGURATION (CHANGE THIS PART ONLY)
# -----------------------------------------------------------------------------
INPUT_FILE="A4_HUMAN_Seuma_2021.csv"  # Only change the filename here!
# -----------------------------------------------------------------------------

# 4. Run the Pipeline
echo "Starting ESM Pipeline on $INPUT_FILE..."

# Construct Output Name
OUTPUT_FILE="results/${INPUT_FILE%.*}_ESM2_150M_MLLR_predictions.csv"
REPORT_FILE="results/${INPUT_FILE%.*}_ESM2_150M_MLLR_report.png"

# Run Scoring
python -m src.batch_scoring \
    --input_csv "data/$INPUT_FILE" \
    --output_csv "$OUTPUT_FILE" \
    --method MLLR \
    --model facebook/esm2_t30_150M_UR50D

# 5. Run Analysis
echo "Generating Statistical Report..."
python -m src.analysis \
    --results_csv "$OUTPUT_FILE" \
    --truth_csv "data/$INPUT_FILE" \
    --output_report "$REPORT_FILE"

echo "Job Finished."
