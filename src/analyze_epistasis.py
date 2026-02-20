import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.stats import pearsonr, spearmanr
import os

def load_and_process(csv_path, method_name, wt_score=None):
    """
    Loads mutant score CSV and returns a DataFrame with epistasis calculations.
    
    For absolute scoring methods (e.g. PLL = full-sequence log-likelihood),
    the raw scores include a large WT baseline component. Direct addition of
    two singles double-counts this baseline, producing a constant offset.
    
    Correct epistasis formula:
        E = Score(AB) - [Score(A) + Score(B) - Score(WT)]
        
    This is equivalent to:
        E = dScore(AB) - [dScore(A) + dScore(B)]
        
    where dScore(X) = Score(X) - Score(WT).
    
    For methods like EDS that already output relative scores (WT = 0),
    the wt_score should be 0, making the formula identical to the naive one.
    
    If wt_score is not provided, it is auto-estimated from the data by
    computing the median offset: median(Score(AB) - Score(A) - Score(B)).
    """
    print(f"[{method_name}] Loading {csv_path}...", flush=True)
    if not os.path.exists(csv_path):
        print(f"[{method_name}] File not found! Skipping.")
        return None

    df = pd.read_csv(csv_path)
    
    # Identify Singles and Doubles
    df['n_mutations'] = df['mutant'].apply(lambda x: len(x.split(':')))
    
    singles = df[df['n_mutations'] == 1].copy()
    doubles = df[df['n_mutations'] == 2].copy()
    
    print(f"[{method_name}] Found {len(singles)} singles and {len(doubles)} doubles.", flush=True)
    
    # Map single mutant scores for quick lookup
    single_score_map = dict(zip(singles['mutant'], singles['score']))
    
    # First pass: collect raw epistasis data to estimate WT score if needed
    raw_data = []
    
    for idx, row in doubles.iterrows():
        mutant = row['mutant']
        score_ab = row['score']
        
        parts = mutant.split(':')
        if len(parts) != 2: continue
        
        m1, m2 = parts
        
        if m1 in single_score_map and m2 in single_score_map:
            score_a = single_score_map[m1]
            score_b = single_score_map[m2]
            raw_data.append({
                'mutant': mutant,
                'score_ab': score_ab,
                'score_a': score_a,
                'score_b': score_b,
            })
    
    if not raw_data:
        print(f"[{method_name}] No valid double mutants found with matching singles!")
        return None
    
    raw_df = pd.DataFrame(raw_data)
    
    # Estimate WT score if not provided
    # For absolute scoring methods (PLL), Score(AB) ≈ ΔScore(AB) + Score(WT)
    # So naive_offset = Score(AB) - Score(A) - Score(B) ≈ -Score(WT)  (when E ≈ 0)
    # Therefore Score(WT) = -naive_offset_median
    if wt_score is None:
        naive_offsets = raw_df['score_ab'] - raw_df['score_a'] - raw_df['score_b']
        naive_median = naive_offsets.median()
        
        # For relative methods like EDS (where WT = 0), the offset should be small
        # For absolute methods like PLL, the offset will be very large (~=|Score(WT)|)
        mean_abs_score = raw_df['score_ab'].abs().mean()
        if abs(naive_median) > 0.5 * mean_abs_score:
            # Absolute scoring method detected - offset is huge relative to scores
            wt_score = -naive_median  # Score(WT) = -offset
            print(f"[{method_name}] Absolute scoring detected. Auto-estimated WT score: {wt_score:.4f}", flush=True)
            print(f"[{method_name}] (Correcting additive baseline by removing double-counted WT component)", flush=True)
        else:
            # Relative scoring method (EDS-like), no WT correction needed
            wt_score = 0.0
            print(f"[{method_name}] Relative scoring detected (WT baseline ≈ 0). No WT correction applied.", flush=True)
    else:
        print(f"[{method_name}] Using provided WT score: {wt_score:.4f}", flush=True)
    
    # Calculate corrected epistasis
    # Expected = Score(A) + Score(B) - Score(WT)  [removes double-counted baseline]
    # Epistasis = Score(AB) - Expected
    epistasis_data = []
    
    for _, row in raw_df.iterrows():
        expected = row['score_a'] + row['score_b'] - wt_score
        epistasis = row['score_ab'] - expected
        
        epistasis_data.append({
            'mutant': row['mutant'],
            'score_ab': row['score_ab'],
            'score_a': row['score_a'],
            'score_b': row['score_b'],
            'expected': expected,
            'epistasis': epistasis,
            'method': method_name
        })
            
    result_df = pd.DataFrame(epistasis_data)
    
    print(f"[{method_name}] Epistasis stats: mean={result_df['epistasis'].mean():.4f}, "
          f"median={result_df['epistasis'].median():.4f}, std={result_df['epistasis'].std():.4f}", flush=True)
    
    return result_df

def plot_epistasis(df, output_prefix):
    """
    Plots Observed vs Expected scores (Scatter) and epistasis distribution.
    """
    method = df['method'].iloc[0]
    
    # --- Scatter Plot: Observed vs Expected ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='expected', y='score_ab', alpha=0.3, s=10)
    
    # Plot line of additivity (y=x)
    min_val = min(df['expected'].min(), df['score_ab'].min())
    max_val = max(df['expected'].max(), df['score_ab'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Additivity', linewidth=2)
    
    # Add Pearson r to title
    r, p = pearsonr(df['expected'], df['score_ab'])
    plt.title(f"Epistasis: Observed vs Expected ({method})\nPearson r = {r:.3f}, p = {p:.2e}")
    plt.xlabel("Expected Score (Additive Model)")
    plt.ylabel("Observed Score (Double Mutant)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{output_prefix}_scatter.png", dpi=300)
    plt.close()
    print(f"[{method}] Scatter plot saved: {output_prefix}_scatter.png", flush=True)
    
    # --- Distribution of Epistasis scores ---
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='epistasis', bins=50, kde=True)
    
    mean_e = df['epistasis'].mean()
    plt.axvline(0, color='r', linestyle='--', label='Zero (No Epistasis)', linewidth=2)
    plt.axvline(mean_e, color='green', linestyle='-', label=f'Mean = {mean_e:.3f}', linewidth=1.5)
    
    plt.title(f"Distribution of Epistasis Scores ({method})")
    plt.xlabel("Epistasis (Observed - Expected)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{output_prefix}_dist.png", dpi=300)
    plt.close()
    print(f"[{method}] Distribution plot saved: {output_prefix}_dist.png", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Analyze epistasis from EDS/PLL scored mutants")
    parser.add_argument("--eds_csv", help="Path to EDS results CSV")
    parser.add_argument("--pll_csv", help="Path to PLL results CSV")
    parser.add_argument("--wt_score_eds", type=float, default=None,
                        help="WT score for EDS (default: auto-detect, typically 0)")
    parser.add_argument("--wt_score_pll", type=float, default=None,
                        help="WT score for PLL (default: auto-estimated from data)")
    parser.add_argument("--output_dir", default="results", help="Output directory for plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.eds_csv:
        df_eds = load_and_process(args.eds_csv, "EDS", wt_score=args.wt_score_eds)
        if df_eds is not None and not df_eds.empty:
            print(f"[EDS] Calculated {len(df_eds)} epistasis interactions.", flush=True)
            plot_epistasis(df_eds, f"{args.output_dir}/Epistasis_EDS")
            df_eds.to_csv(f"{args.output_dir}/Epistasis_EDS_650M_Analysis.csv", index=False)
            print(f"[EDS] Analysis CSV saved.", flush=True)
            
    if args.pll_csv:
        df_pll = load_and_process(args.pll_csv, "PLL", wt_score=args.wt_score_pll)
        if df_pll is not None and not df_pll.empty:
            print(f"[PLL] Calculated {len(df_pll)} epistasis interactions.", flush=True)
            plot_epistasis(df_pll, f"{args.output_dir}/Epistasis_PLL")
            df_pll.to_csv(f"{args.output_dir}/Epistasis_PLL_ESM1v_Analysis.csv", index=False)
            print(f"[PLL] Analysis CSV saved.", flush=True)

if __name__ == "__main__":
    main()
