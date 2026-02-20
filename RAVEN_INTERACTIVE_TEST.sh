#!/bin/bash
# -----------------------------------------------------------------------------
# RAVEN INTERACTIVE TEST SCRIPT (LOGIN NODE SAFE)
# -----------------------------------------------------------------------------
# Run this directly in your terminal to test if code works.
# It uses CPU only and processes just 2 samples.
# -----------------------------------------------------------------------------

echo "🦅 Starting Interactive Test on Raven..."

# 1. Setup Environment
# Ensure you are in /ptmp/YOUR_USER/ESM_Project
export MY_PROJECT_DIR=$(pwd)
export HF_HOME="$MY_PROJECT_DIR/hf_cache"

echo "📂 Working Directory: $MY_PROJECT_DIR"

# 2. Load Modules (Same as job, but for current session)
module load anaconda/3/2021.11
# We do NOT load CUDA here to force CPU usage for safety on login node
# module load cuda/11.8 

# -----------------------------------------------------------------------------
# ⚙️ USER CONFIGURATION
# -----------------------------------------------------------------------------
INPUT_FILE="A4_HUMAN_Seuma_2021.csv"
# -----------------------------------------------------------------------------

# 3. Validation Run (Tiny)
echo "🧪 Running Validation (Pipeline Verifier)..."
python -m src.verify_pipeline

# 4. dry-run of main script
echo "🧪 Running Dry-Run of Batch Scoring (2 samples)..."
python -m src.batch_scoring \
    --input_csv "data/$INPUT_FILE" \
    --output_csv results/interactive_test_mllr.csv \
    --method MLLR \
    --max_samples 2 \
    --model facebook/esm2_t30_150M_UR50D 

echo "✅ Test Complete. If you see 'Done!', your code is ready."
echo "   You can now try submitting the full job via sbatch when allowed."
