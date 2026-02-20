#!/bin/bash
# Load Environment
module load anaconda/3/2021.11
source activate /ptmp/$USER/ESM_Project/env

TRUTH="data/A4_HUMAN_Seuma_2021.csv"

echo "========================================"
echo "   Analyzing EDS (Euclidean Distance)   "
echo "========================================"

echo ""
echo "--- ESM-2 150M ---"
python -m src.analysis \
    --results_csv results/A4_HUMAN_Seuma_2021_ESM2_150M_EDS_predictions.csv \
    --truth_csv "$TRUTH" \
    --output_report results/report_EDS_150M.png

echo ""
echo "--- ESM-2 650M ---"
python -m src.analysis \
    --results_csv results/A4_HUMAN_Seuma_2021_ESM2_650M_EDS_predictions.csv \
    --truth_csv "$TRUTH" \
    --output_report results/report_EDS_650M.png

echo ""
echo "--- ESM-1v ---"
python -m src.analysis \
    --results_csv results/A4_HUMAN_Seuma_2021_ESM1v_EDS_predictions.csv \
    --truth_csv "$TRUTH" \
    --output_report results/report_EDS_ESM1v.png

echo ""
echo "Done."
