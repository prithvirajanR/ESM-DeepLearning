import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.landscape import generate_random_mutants
from src.models import ESM2Model
from src.scoring import calculate_masked_marginal
from src.data_loader import DMSDataset

def explore_landscape(wt_sequence: str, model_wrapper, k_values: list, n_per_k: int = 100, method="MLLR"):
    """
    Explore the fitness landscape by scoring random mutants at different distances.
    """
    from src.scoring import MaskedMarginalScoring, PseudoLogLikelihoodScoring, MaskedLogLikelihoodRatioScoring
    
    # Initialize scorer
    if method == "MM":
        scorer = MaskedMarginalScoring()
    elif method == "PLL":
        scorer = PseudoLogLikelihoodScoring()
    elif method == "MLLR":
        scorer = MaskedLogLikelihoodRatioScoring()
    else:
        raise ValueError(f"Unknown method {method}")
        
    results = []
    
    for k in k_values:
        print(f"\n=== Generating and scoring {n_per_k} mutants at k={k} ===")
        
        # Generate mutants
        mutants = generate_random_mutants(wt_sequence, k, n=n_per_k)
        
        # Score each mutant
        for mutant_code, mutant_seq in tqdm(mutants, desc=f"k={k}"):
            try:
                score = scorer.score(model_wrapper, wt_sequence, mutant_code, mutant_seq)
                
                results.append({
                    'k': k,
                    'mutant_code': mutant_code,
                    'predicted_score': score
                })
                
            except Exception as e:
                print(f"Error scoring {mutant_code}: {e}")
                continue
    
    return pd.DataFrame(results)

def plot_landscape(df: pd.DataFrame, output_path: str):
    """Create visualization of fitness landscape."""
    
    # Calculate statistics per k
    stats = df.groupby('k')['predicted_score'].agg(['mean', 'std', 'count']).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean fitness vs distance
    ax1.errorbar(stats['k'], stats['mean'], yerr=stats['std'], 
                 marker='o', markersize=8, capsize=5, capthick=2, linewidth=2)
    ax1.set_xlabel('Mutational Distance (k)', fontsize=12)
    ax1.set_ylabel('Mean Predicted Fitness Score', fontsize=12)
    ax1.set_title('Fitness Decay with Mutational Distance', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution at each k
    for k in sorted(df['k'].unique()):
        data = df[df['k'] == k]['predicted_score']
        ax2.hist(data, alpha=0.5, bins=20, label=f'k={k}')
    
    ax2.set_xlabel('Predicted Fitness Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Predicted Fitness', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    # Print summary
    print("\n=== Landscape Summary ===")
    print(stats.to_string(index=False))

if __name__ == "__main__":
    # Load APP wild-type sequence
    print("Loading APP dataset to get wild-type sequence...")
    dataset = DMSDataset(
        "f:/ESM/data/A4_HUMAN_Seuma_2021.csv",
        reference_file_path="f:/ESM/data/ProteinGym_reference_file_substitutions.csv"
    )
    wt_seq = dataset.target_seq
    print(f"Wild-type sequence length: {len(wt_seq)} amino acids")
    
    # Load model
    print("\nLoading ESM-2 model...")
    model = ESM2Model("facebook/esm2_t30_150M_UR50D")
    model.load_model()
    
    # Explore landscape
    k_values = [1, 5, 10, 20]
    n_per_k = 50  # Generate 50 mutants per distance (adjust if too slow)
    
    print(f"\nExploring landscape at k={k_values} with {n_per_k} mutants each...")
    df = explore_landscape(wt_seq, model, k_values, n_per_k, method="MLLR")
    
    # Save results
    output_csv = "f:/ESM/results/landscape_exploration_mllr.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # Plot
    plot_landscape(df, "f:/ESM/results/landscape_plot_mllr.png")
