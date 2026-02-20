import pandas as pd
import numpy as np
import argparse
import logging
from scipy.stats import spearmanr, zscore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_merge(file1, file2, truth_file):
    logging.info(f"Loading {file1}...")
    df1 = pd.read_csv(file1)
    logging.info(f"Loading {file2}...")
    df2 = pd.read_csv(file2)
    logging.info(f"Loading Truth {truth_file}...")
    truth = pd.read_csv(truth_file)

    # Merge
    merged = pd.merge(df1, df2, on='mutant', suffixes=('_mllr', '_pll'))
    merged = pd.merge(merged, truth, on='mutant')
    
    # Filter valid
    merged = merged.dropna(subset=['score_mllr', 'score_pll', 'DMS_score'])
    
    return merged

def optimize_hybrid(df):
    # Normalize scores (Robust Z-score to handle different scales)
    df['z_mllr'] = zscore(df['score_mllr'])
    df['z_pll'] = zscore(df['score_pll'])
    
    best_rho = -1
    best_alpha = 0
    best_prec = -1
    best_alpha_prec = 0
    
    results = []

    logging.info("Sweeping alpha (weight for MLLR)...")
    
    # Sweep alpha from 0.0 to 1.0
    for alpha in np.linspace(0, 1, 21):
        # Hybrid Score = alpha * MLLR + (1-alpha) * PLL
        df['hybrid'] = alpha * df['z_mllr'] + (1 - alpha) * df['z_pll']
        
        # 1. Spearman
        rho, _ = spearmanr(df['DMS_score'], df['hybrid'])
        
        # 2. Top-100 Precision
        top_k = 100
        top_preds = df.nlargest(top_k, 'hybrid')
        # Assuming binary label is in 'DMS_score_bin'
        if 'DMS_score_bin' in df.columns:
            prec = top_preds['DMS_score_bin'].sum() / top_k
        else:
            prec = 0.0
            
        results.append((alpha, rho, prec))
        
        if rho > best_rho:
            best_rho = rho
            best_alpha = alpha
            
        if prec > best_prec:
            best_prec = prec
            best_alpha_prec = alpha
            
    print(f"\n{'Alpha (MLLR Weight)':<20} | {'Spearman':<10} | {'Top-100 Prec':<15}")
    print("-" * 55)
    for res in results:
        print(f"{res[0]:<20.2f} | {res[1]:<10.4f} | {res[2]:<15.2f}")
        
    print("-" * 55)
    print(f"Best Spearman:  {best_rho:.4f} at alpha={best_alpha:.2f}")
    print(f"Best Precision: {best_prec:.2f} at alpha={best_alpha_prec:.2f}")

    return best_alpha

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mllr_csv", required=True)
    parser.add_argument("--pll_csv", required=True)
    parser.add_argument("--truth_csv", required=True)
    args = parser.parse_args()
    
    df = load_and_merge(args.mllr_csv, args.pll_csv, args.truth_csv)
    optimize_hybrid(df)

if __name__ == "__main__":
    main()
