import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Essential for headless environments (HPC)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to scored synthetic mutants CSV")
    parser.add_argument("--metadata_csv", default=None, help="Path to original CSV with metadata (e.g. n_mutations)")
    parser.add_argument("--output_dir", default="results", help="Directory to save plots")
    args = parser.parse_args()
    
    print(f"Loading {args.input_csv}...", flush=True)
    df = pd.read_csv(args.input_csv)
    
    # Merge with metadata if provided (to get n_mutations)
    if args.metadata_csv:
        print(f"Loading metadata from {args.metadata_csv}...")
        meta_df = pd.read_csv(args.metadata_csv)
        # Assuming 'mutant' is the key. 
        # EDS output has 'mutant' (e.g. A123B:C456D).
        # Synthetic input has 'mutant' and 'n_mutations'.
        if 'mutant' in df.columns and 'mutant' in meta_df.columns:
            df = df.merge(meta_df[['mutant', 'n_mutations']], on='mutant', how='left')
            print(f"Merged metadata. Columns: {df.columns.tolist()}")
        else:
            print("Warning: Could not merge metadata. 'mutant' column missing in one of the files.")

    # Ensure n_mutations is numeric
    if 'n_mutations' not in df.columns:
        print("Error: 'n_mutations' column missing. Did you forget --metadata_csv?")
        return

    df['n_mutations'] = pd.to_numeric(df['n_mutations'], errors='coerce')
    df = df.dropna(subset=['score', 'n_mutations'])
    
    # Calculate Correlation
    spearman_corr = df['n_mutations'].corr(df['score'], method='spearman')
    print(f"Spearman Correlation (Distance vs Score): {spearman_corr:.3f}")
    
    # Plot Fitness Decay (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='n_mutations', y='score', data=df, palette="viridis")
    plt.title(f"Fitness Decay with Mutational Distance\nA4_HUMAN (Spearman r={spearman_corr:.2f})")
    plt.xlabel("Number of Mutations (k)")
    plt.ylabel("DMS Score (Proxies Fitness)")
    plt.grid(True, alpha=0.3)
    
    output_plot = os.path.join(args.output_dir, "Synthetic_Landscape_Decay.png")
    plt.savefig(output_plot, dpi=300)
    print(f"Saved plot to {output_plot}")
    
    # Plot Violin for density
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='n_mutations', y='score', data=df, palette="magma", cut=0)
    plt.title(f"Fitness Distribution at Different Distances\nA4_HUMAN")
    plt.xlabel("Number of Mutations (k)")
    plt.ylabel("DMS Score")
    plt.grid(True, alpha=0.3)
    
    output_violin = os.path.join(args.output_dir, "Synthetic_Landscape_Violin.png")
    plt.savefig(output_violin, dpi=300)
    print(f"Saved violin plot to {output_violin}")

if __name__ == "__main__":
    main()
