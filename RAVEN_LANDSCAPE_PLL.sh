#!/bin/bash
#SBATCH --job-name=Landscape_PLL
#SBATCH --output=%j_Landscape_PLL.out
#SBATCH --error=%j_Landscape_PLL.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1

# Load Environment
module unload python-waterboa/2024.06
module load anaconda/3/2021.11
source activate /ptmp/$USER/ESM_Project/env

if [ -d "/ptmp/$USER/ESM_Project" ]; then
    cd /ptmp/$USER/ESM_Project
fi

# 0. Clean old outputs to prevent checkpoint mismatch
echo "[Phase 4 PLL] Cleaning old outputs..."
rm -f results/Synthetic_Landscape_PLL.csv
rm -f results/Synthetic_Landscape_PLL.csv.log

# 1. Generate Synthetic Data (Same seed=42 as EDS for fair comparison)
echo "[Phase 4 PLL] Generating Synthetic Landscape (k=2,5,10,20)..."
python -m src.generate_synthetic \
    --distances 2,5,10,20 \
    --count_per_k 1000 \
    --seed 42 \
    --output data/Synthetic_Landscape.csv

# 2. Run Scoring (PLL + ESM-1v)
echo "[Phase 4 PLL] Scoring 4,000 mutants with PLL (this will take ~9 hours)..."
python -m src.batch_scoring \
    --input_csv data/Synthetic_Landscape.csv \
    --model facebook/esm1v_t33_650M_UR90S_1 \
    --method PLL \
    --output_csv results/Synthetic_Landscape_PLL.csv

# 3. Analyze and Plot
echo "[Phase 4 PLL] Plotting Fitness Decay..."
python -m src.analyze_landscape \
    --input_csv results/Synthetic_Landscape_PLL.csv \
    --metadata_csv data/Synthetic_Landscape.csv \
    --output_dir results

echo "[Phase 4 PLL] Landscape Exploration Complete!"
