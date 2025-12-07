import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm

def calculate_epistasis(df):
    """
    Calculate epistasis scores for double mutants.
    Epistasis = Fitness(AB) - (Fitness(A) + Fitness(B))
    Assumes Fitness is Log-Likelihood ratio or similar additive metric relative to WT.
    """
    # 1. Map single mutants to their scores
    singles = df[df['mutant'].apply(lambda x: len(x.split(':')) == 1)].copy()
    single_map = dict(zip(singles['mutant'], singles['score']))
    
    # 2. Filter double mutants
    doubles = df[df['mutant'].apply(lambda x: len(x.split(':')) == 2)].copy()
    
    epistasis_scores = []
    
    for idx, row in tqdm(doubles.iterrows(), total=len(doubles), desc="Calculating Epistasis"):
        mutant = row['mutant']
        score_ab = row['score']
        
        # Split into components
        m1, m2 = mutant.split(':')
        
        if m1 in single_map and m2 in single_map:
            score_a = single_map[m1]
            score_b = single_map[m2]
            
            # Additivity hypothesis: expected = sum of individuals
            expected = score_a + score_b
            epistasis = score_ab - expected
            
            epistasis_scores.append({
                'mutant': mutant,
                'score_ab': score_ab,
                'score_a': score_a,
                'score_b': score_b,
                'expected': expected,
                'epistasis': epistasis
            })
            
    return pd.DataFrame(epistasis_scores)

def plot_epistasis_heatmap(df_epistasis, output_path):
    """Create a heatmap of epistasis scores."""
    # This requires processing the mutants to get positions
    # Simplified for now: specific pairs or just distribution
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df_epistasis['epistasis'], bins=50, kde=True)
    plt.xlabel('Epistasis Score (Observed - Expected)')
    plt.title('Distribution of Epistasis Scores')
    plt.axvline(0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="CSV with scores for singles and doubles.")
    parser.add_argument("--output_csv", required=True, help="Output CSV for epistasis scores.")
    args = parser.parse_args()
    
    print(f"Loading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    print("Calculating epistasis...")
    df_epistasis = calculate_epistasis(df)
    
    print(f"Found {len(df_epistasis)} valid double mutants.")
    df_epistasis.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")
    
    plot_epistasis_heatmap(df_epistasis, args.output_csv.replace('.csv', '.png'))

if __name__ == "__main__":
    main()
