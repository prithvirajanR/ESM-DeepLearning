#!/bin/bash
#SBATCH --job-name=Landscape_EDS
#SBATCH --output=%j_Landscape_EDS.out
#SBATCH --error=%j_Landscape_EDS.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1

# Load Environment
module unload python-waterboa/2024.06
module load anaconda/3/2021.11
source activate /ptmp/$USER/ESM_Project/env

# Go to project root (assume submission from project root or use explicit path)
if [ -d "/ptmp/$USER/ESM_Project" ]; then
    cd /ptmp/$USER/ESM_Project
fi

# 0. Clean old outputs to prevent checkpoint mismatch
echo "[Phase 4] Cleaning old outputs..."
rm -f data/Synthetic_Landscape.csv
rm -f results/Synthetic_Landscape_EDS.csv
rm -f results/Synthetic_Landscape_EDS.csv.log

# 1. Generate Synthetic Data
logging_info="[Phase 4] Generating Synthetic Landscape (k=2,5,10,20)..."
echo $logging_info
python -m src.generate_synthetic \
    --distances 2,5,10,20 \
    --count_per_k 1000 \
    --seed 42 \
    --output data/Synthetic_Landscape.csv

# 2. Run Scoring (EDS + ESM-2 650M)
echo "[Phase 4] Scoring 4,000 mutants with EDS..."
# Note: Input CSV has 'mutant' and 'mutated_sequence'. batch_scoring should use 'mutated_sequence'.
python -m src.batch_scoring \
    --input_csv data/Synthetic_Landscape.csv \
    --model facebook/esm2_t33_650M_UR50D \
    --method EDS \
    --output_csv results/Synthetic_Landscape_EDS.csv

# 3. Analyze and Plot
echo "[Phase 4] Plotting Fitness Decay..."
python -m src.analyze_landscape \
    --input_csv results/Synthetic_Landscape_EDS.csv \
    --metadata_csv data/Synthetic_Landscape.csv \
    --output_dir results

echo "[Phase 4] Landscape Exploration Complete!"
