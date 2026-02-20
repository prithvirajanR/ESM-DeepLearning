#!/bin/bash
#SBATCH --job-name=Analyze_PLL
#SBATCH --output=%j_Analyze_PLL.out
#SBATCH --error=%j_Analyze_PLL.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Load Environment
module unload python-waterboa/2024.06
module load anaconda/3/2021.11
source activate /ptmp/$USER/ESM_Project/env

if [ -d "/ptmp/$USER/ESM_Project" ]; then
    cd /ptmp/$USER/ESM_Project
fi

echo "=== PLL Epistasis Analysis ==="

# Run the epistasis analysis comparing EDS and PLL
python src/analyze_epistasis.py \
    --eds_csv results/Epistasis_EDS_650M.csv \
    --pll_csv results/Epistasis_PLL_ESM1v.csv \
    --output_dir results

echo "=== Quick Stats ==="
python3 -c "
import pandas as pd
import numpy as np

# Load PLL results
df = pd.read_csv('results/Epistasis_PLL_ESM1v.csv')
df['n_mutations'] = df['mutant'].apply(lambda x: len(x.split(':')))

singles = df[df['n_mutations'] == 1]
doubles = df[df['n_mutations'] == 2]

print(f'Total rows: {len(df)}')
print(f'Singles: {len(singles)}')
print(f'Doubles: {len(doubles)}')
print(f'Score range: [{df[\"score\"].min():.2f}, {df[\"score\"].max():.2f}]')
print(f'Mean score: {df[\"score\"].mean():.2f}')

# Compare singles vs doubles performance
print(f'\n--- Singles ---')
print(singles['score'].describe())
print(f'\n--- Doubles ---')
print(doubles['score'].describe())
"

echo "=== Analysis Complete ==="
