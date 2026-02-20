#!/bin/bash

# Define Base Directories
BASE="results"
mkdir -p "$BASE/Summary_Plots"

# --- ESM-2 150M ---
echo "Organizing ESM-2 150M..."
mkdir -p "$BASE/ESM2_150M/PLL"
mkdir -p "$BASE/ESM2_150M/EDS"
mkdir -p "$BASE/ESM2_150M/EnsembleMLLR"
# Move PLL
mv "$BASE"/A4_HUMAN_Seuma_2021_ESM2_150M_PLL_predictions.csv* "$BASE/ESM2_150M/PLL/" 2>/dev/null
mv "$BASE"/final_report_ESM2_150M_PLL.png "$BASE/ESM2_150M/PLL/" 2>/dev/null
# Move EDS
mv "$BASE"/A4_HUMAN_Seuma_2021_ESM2_150M_EDS_predictions.csv* "$BASE/ESM2_150M/EDS/" 2>/dev/null
mv "$BASE"/report_EDS_150M.png "$BASE/ESM2_150M/EDS/" 2>/dev/null
# Move EnsembleMLLR
mv "$BASE"/A4_HUMAN_Seuma_2021_ESM2_150M_EnsembleMLLR_predictions.csv* "$BASE/ESM2_150M/EnsembleMLLR/" 2>/dev/null
mv "$BASE"/report_EnsembleMLLR_150M.png "$BASE/ESM2_150M/EnsembleMLLR/" 2>/dev/null

# --- ESM-2 650M ---
echo "Organizing ESM-2 650M..."
mkdir -p "$BASE/ESM2_650M/PLL"
mkdir -p "$BASE/ESM2_650M/EDS"
mkdir -p "$BASE/ESM2_650M/EnsembleMLLR"
# Move PLL
mv "$BASE"/A4_HUMAN_Seuma_2021_ESM2_650M_PLL_predictions.csv* "$BASE/ESM2_650M/PLL/" 2>/dev/null
mv "$BASE"/report_PLL_650M.png "$BASE/ESM2_650M/PLL/" 2>/dev/null
# Move EDS
mv "$BASE"/A4_HUMAN_Seuma_2021_ESM2_650M_EDS_predictions.csv* "$BASE/ESM2_650M/EDS/" 2>/dev/null
mv "$BASE"/report_EDS_650M.png "$BASE/ESM2_650M/EDS/" 2>/dev/null
# Move EnsembleMLLR
mv "$BASE"/A4_HUMAN_Seuma_2021_ESM2_650M_EnsembleMLLR_predictions.csv* "$BASE/ESM2_650M/EnsembleMLLR/" 2>/dev/null
mv "$BASE"/report_EnsembleMLLR_650M.png "$BASE/ESM2_650M/EnsembleMLLR/" 2>/dev/null

# --- ESM-1v ---
echo "Organizing ESM-1v..."
mkdir -p "$BASE/ESM1v/PLL"
mkdir -p "$BASE/ESM1v/EDS"
mkdir -p "$BASE/ESM1v/EnsembleMLLR"
# Move PLL
mv "$BASE"/A4_HUMAN_Seuma_2021_ESM1v_650M_PLL_predictions.csv* "$BASE/ESM1v/PLL/" 2>/dev/null
mv "$BASE"/A4_HUMAN_Seuma_2021_ESM1v_PLL_predictions.csv* "$BASE/ESM1v/PLL/" 2>/dev/null
mv "$BASE"/report_PLL_ESM1v.png "$BASE/ESM1v/PLL/" 2>/dev/null
# Move EDS
mv "$BASE"/A4_HUMAN_Seuma_2021_ESM1v_EDS_predictions.csv* "$BASE/ESM1v/EDS/" 2>/dev/null
mv "$BASE"/report_EDS_ESM1v.png "$BASE/ESM1v/EDS/" 2>/dev/null
# Move EnsembleMLLR
mv "$BASE"/A4_HUMAN_Seuma_2021_ESM1v_EnsembleMLLR_predictions.csv* "$BASE/ESM1v/EnsembleMLLR/" 2>/dev/null
mv "$BASE"/report_EnsembleMLLR_ESM1v.png "$BASE/ESM1v/EnsembleMLLR/" 2>/dev/null


# Consolidate Reports (Optional copies to Summary_Plots if desired, but for now just cleanup)
# Copy all PNGs to Summary_Plots for easy viewing before moving source
# (Actually, users usually want them with the data. I'll stick to moving them.)

echo "Cleanup Complete. Structure:"
echo "results/"
echo "├── ESM2_150M/{PLL, EDS, MLLR}"
echo "├── ESM2_650M/{PLL, EDS, MLLR}"
echo "└── ESM1v/{PLL, EDS, MLLR}"
