"""
Hypothesis Testing for ESM Protein Fitness Landscape Project.

H4: "Predictive performance is much higher for single mutants than for double mutants."
    Tests whether Spearman correlation between model scores and experimental DMS
    fitness is significantly higher for singles than doubles.

H5: "The best-performing model on epistasis will be the most trustworthy
     for exploring the whole landscape."
    Tests which scoring method (EDS vs PLL) better captures epistasis patterns
    and links that to landscape trustworthiness.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, fisher_exact
from scipy.stats import bootstrap
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(eds_csv, pll_csv, dms_csv):
    """Load and merge model scores with experimental DMS data."""
    eds = pd.read_csv(eds_csv)
    pll = pd.read_csv(pll_csv)
    dms = pd.read_csv(dms_csv)
    
    # Merge all data
    merged = dms.copy()
    merged = merged.merge(
        eds[['mutant', 'score']].rename(columns={'score': 'eds_score'}),
        on='mutant', how='inner'
    )
    merged = merged.merge(
        pll[['mutant', 'score']].rename(columns={'score': 'pll_score'}),
        on='mutant', how='inner'
    )
    
    # Compute delta-PLL (relative to WT) for fairer comparison
    # Auto-estimate WT PLL as the maximum PLL score (closest to 0 = least mutated)
    merged['n_mutations'] = merged['mutant'].apply(lambda x: len(x.split(':')))
    
    return merged


def bootstrap_spearman(x, y, n_boot=10000, seed=42):
    """Bootstrap confidence interval for Spearman rho."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        rho, _ = spearmanr(x.iloc[idx], y.iloc[idx])
        rhos.append(rho)
    rhos = np.array(rhos)
    return np.percentile(rhos, [2.5, 97.5])


def test_h4(merged, output_dir):
    """
    H4: Predictive performance is higher for singles than doubles.
    
    Computes Spearman rho(model_score, DMS_score) separately for 
    single and double mutants, with bootstrap CIs and a permutation
    test for the difference.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 4: Single vs Double Mutant Prediction Accuracy")
    print("="*70)
    
    singles = merged[merged['n_mutations'] == 1].copy()
    doubles = merged[merged['n_mutations'] == 2].copy()
    
    results = {}
    
    for method, score_col in [('EDS', 'eds_score'), ('PLL', 'pll_score')]:
        print(f"\n--- {method} ---")
        
        # Singles correlation
        rho_s, p_s = spearmanr(singles[score_col], singles['DMS_score'])
        ci_s = bootstrap_spearman(singles[score_col], singles['DMS_score'])
        
        # Doubles correlation
        rho_d, p_d = spearmanr(doubles[score_col], doubles['DMS_score'])
        ci_d = bootstrap_spearman(doubles[score_col], doubles['DMS_score'])
        
        # Difference
        delta = rho_s - rho_d
        
        # Permutation test for difference in correlations
        n_perm = 10000
        rng = np.random.RandomState(42)
        all_scores = pd.concat([singles[score_col], doubles[score_col]], ignore_index=True)
        all_dms = pd.concat([singles['DMS_score'], doubles['DMS_score']], ignore_index=True)
        n_singles = len(singles)
        
        perm_deltas = []
        for _ in range(n_perm):
            perm_idx = rng.permutation(len(all_scores))
            perm_s = perm_idx[:n_singles]
            perm_d = perm_idx[n_singles:]
            rho_ps, _ = spearmanr(all_scores.iloc[perm_s], all_dms.iloc[perm_s])
            rho_pd, _ = spearmanr(all_scores.iloc[perm_d], all_dms.iloc[perm_d])
            perm_deltas.append(rho_ps - rho_pd)
        
        perm_deltas = np.array(perm_deltas)
        p_perm = np.mean(np.abs(perm_deltas) >= np.abs(delta))
        
        print(f"  Singles (n={len(singles)}): ρ = {rho_s:.4f}  [{ci_s[0]:.4f}, {ci_s[1]:.4f}]  p = {p_s:.2e}")
        print(f"  Doubles (n={len(doubles)}): ρ = {rho_d:.4f}  [{ci_d[0]:.4f}, {ci_d[1]:.4f}]  p = {p_d:.2e}")
        print(f"  Δρ = {delta:.4f}  (permutation p = {p_perm:.4f})")
        
        if delta > 0 and p_perm < 0.05:
            verdict = "✅ SUPPORTED — singles predicted significantly better"
        elif delta > 0:
            verdict = "⚠️ TREND — singles predicted better but not significant"
        else:
            verdict = "❌ NOT SUPPORTED — doubles predicted as well or better"
        print(f"  H4 verdict ({method}): {verdict}")
        
        results[method] = {
            'rho_singles': rho_s, 'ci_singles': ci_s, 'p_singles': p_s,
            'rho_doubles': rho_d, 'ci_doubles': ci_d, 'p_doubles': p_d,
            'delta_rho': delta, 'p_perm': p_perm
        }
    
    # --- Plot H4 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for col, (method, score_col) in enumerate([('EDS', 'eds_score'), ('PLL', 'pll_score')]):
        r = results[method]
        
        # Singles scatter
        ax = axes[0, col]
        ax.scatter(singles[score_col], singles['DMS_score'], alpha=0.5, s=20, c='#2196F3', label='Singles')
        rho_s = r['rho_singles']
        ax.set_title(f"{method}: Singles (n={len(singles)})\nSpearman ρ = {rho_s:.4f}", fontsize=11)
        ax.set_xlabel(f"{method} Score")
        ax.set_ylabel("Experimental DMS Score")
        ax.grid(True, alpha=0.3)
        
        # Doubles scatter  
        ax = axes[1, col]
        ax.scatter(doubles[score_col], doubles['DMS_score'], alpha=0.1, s=5, c='#FF5722', label='Doubles')
        rho_d = r['rho_doubles']
        ax.set_title(f"{method}: Doubles (n={len(doubles)})\nSpearman ρ = {rho_d:.4f}", fontsize=11)
        ax.set_xlabel(f"{method} Score")
        ax.set_ylabel("Experimental DMS Score")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("H4: Single vs Double Mutant Predictive Power", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/H4_singles_vs_doubles.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[H4] Plot saved: {output_dir}/H4_singles_vs_doubles.png")
    
    # --- Bar chart comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['EDS', 'PLL']
    singles_rhos = [results[m]['rho_singles'] for m in methods]
    doubles_rhos = [results[m]['rho_doubles'] for m in methods]
    singles_cis = [results[m]['ci_singles'] for m in methods]
    doubles_cis = [results[m]['ci_doubles'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    s_err = [(r - ci[0], ci[1] - r) for r, ci in zip(singles_rhos, singles_cis)]
    d_err = [(r - ci[0], ci[1] - r) for r, ci in zip(doubles_rhos, doubles_cis)]
    
    bars1 = ax.bar(x - width/2, singles_rhos, width, label='Singles',
                   color='#2196F3', alpha=0.8,
                   yerr=np.array(s_err).T, capsize=5)
    bars2 = ax.bar(x + width/2, doubles_rhos, width, label='Doubles', 
                   color='#FF5722', alpha=0.8,
                   yerr=np.array(d_err).T, capsize=5)
    
    ax.set_ylabel('Spearman ρ (vs Experimental DMS)', fontsize=12)
    ax.set_title('H4: Predictive Performance — Singles vs Doubles', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(singles_rhos), max(doubles_rhos)) * 1.3)
    
    # Add significance annotations
    for i, method in enumerate(methods):
        p = results[method]['p_perm']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        y_max = max(singles_rhos[i], doubles_rhos[i]) + 0.05
        ax.annotate(sig, xy=(i, y_max), ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/H4_barplot.png", dpi=300)
    plt.close()
    print(f"[H4] Bar plot saved: {output_dir}/H4_barplot.png")
    
    return results


def test_h5(merged, eds_analysis_csv, pll_analysis_csv, output_dir):
    """
    H5: The best-performing model on epistasis is most trustworthy for landscape.
    
    Evaluates which method (EDS vs PLL) better captures epistasis patterns by:
    1. Comparing predicted vs experimental epistasis
    2. Assessing sign agreement (positive/negative epistasis)
    3. Linking epistasis accuracy to overall predictive power
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 5: Epistasis Accuracy → Landscape Trustworthiness")
    print("="*70)
    
    # Load epistasis analysis data
    eds_epi = pd.read_csv(eds_analysis_csv)
    pll_epi = pd.read_csv(pll_analysis_csv)
    
    # We need experimental epistasis from DMS data
    dms = merged.copy()
    singles = dms[dms['n_mutations'] == 1].copy()
    doubles = dms[dms['n_mutations'] == 2].copy()
    
    # Build single DMS score map
    single_dms_map = dict(zip(singles['mutant'], singles['DMS_score']))
    
    # Calculate experimental epistasis for doubles
    exp_epistasis = []
    for _, row in doubles.iterrows():
        parts = row['mutant'].split(':')
        if len(parts) != 2:
            continue
        m1, m2 = parts
        if m1 in single_dms_map and m2 in single_dms_map:
            dms_a = single_dms_map[m1]
            dms_b = single_dms_map[m2]
            dms_ab = row['DMS_score']
            exp_e = dms_ab - (dms_a + dms_b)
            exp_epistasis.append({
                'mutant': row['mutant'],
                'exp_epistasis': exp_e,
                'DMS_ab': dms_ab,
                'DMS_a': dms_a,
                'DMS_b': dms_b,
            })
    
    exp_df = pd.DataFrame(exp_epistasis)
    print(f"\nExperimental epistasis computed for {len(exp_df)} double mutants.")
    print(f"  Mean exp epistasis: {exp_df['exp_epistasis'].mean():.4f}")
    print(f"  Std exp epistasis: {exp_df['exp_epistasis'].std():.4f}")
    
    # Merge with model epistasis
    comparison = exp_df.merge(
        eds_epi[['mutant', 'epistasis']].rename(columns={'epistasis': 'eds_epistasis'}),
        on='mutant', how='inner'
    )
    comparison = comparison.merge(
        pll_epi[['mutant', 'epistasis']].rename(columns={'epistasis': 'pll_epistasis'}),
        on='mutant', how='inner'
    )
    
    print(f"  Matched {len(comparison)} double mutants across all datasets.")
    
    results = {}
    
    for method, col in [('EDS', 'eds_epistasis'), ('PLL', 'pll_epistasis')]:
        print(f"\n--- {method} Epistasis Accuracy ---")
        
        # Correlation between predicted and experimental epistasis
        rho, p_rho = spearmanr(comparison[col], comparison['exp_epistasis'])
        r, p_r = pearsonr(comparison[col], comparison['exp_epistasis'])
        
        print(f"  Spearman ρ (predicted vs experimental epistasis): {rho:.4f} (p={p_rho:.2e})")
        print(f"  Pearson  r (predicted vs experimental epistasis): {r:.4f} (p={p_r:.2e})")
        
        # Sign agreement: does the model predict the correct direction of epistasis?
        pred_sign = np.sign(comparison[col])
        exp_sign = np.sign(comparison['exp_epistasis'])
        sign_agree = (pred_sign == exp_sign).mean()
        print(f"  Sign agreement: {sign_agree:.1%}")
        
        # Classification accuracy: positive vs negative epistasis
        # Confusion matrix
        pred_pos = (comparison[col] > 0).values
        exp_pos = (comparison['exp_epistasis'] > 0).values
        
        tp = (pred_pos & exp_pos).sum()
        tn = (~pred_pos & ~exp_pos).sum()
        fp = (pred_pos & ~exp_pos).sum()
        fn = (~pred_pos & ~exp_pos).sum()
        
        print(f"  Positive epistasis detection: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        results[method] = {
            'spearman_rho': rho, 'spearman_p': p_rho,
            'pearson_r': r, 'pearson_p': p_r,
            'sign_agreement': sign_agree,
        }
    
    # --- Determine winner ---
    eds_rho = results['EDS']['spearman_rho']
    pll_rho = results['PLL']['spearman_rho']
    
    print(f"\n--- H5 Comparison ---")
    print(f"  EDS epistasis Spearman ρ: {eds_rho:.4f}")
    print(f"  PLL epistasis Spearman ρ: {pll_rho:.4f}")
    
    if abs(eds_rho) > abs(pll_rho):
        winner = "EDS"
        print(f"  → EDS captures epistasis better")
    else:
        winner = "PLL"
        print(f"  → PLL captures epistasis better")
    
    # Link to overall landscape trustworthiness
    # Use overall Spearman on ALL mutants as proxy for landscape trust
    for method, score_col in [('EDS', 'eds_score'), ('PLL', 'pll_score')]:
        rho_all, _ = spearmanr(merged[score_col], merged['DMS_score'])
        results[method]['overall_rho'] = rho_all
        print(f"  {method} overall Spearman ρ (all mutants): {rho_all:.4f}")
    
    # Does the better epistasis model also have better overall performance?
    eds_overall = results['EDS']['overall_rho']
    pll_overall = results['PLL']['overall_rho']
    
    if (winner == 'EDS' and abs(eds_overall) > abs(pll_overall)) or \
       (winner == 'PLL' and abs(pll_overall) > abs(eds_overall)):
        print(f"\n  ✅ H5 SUPPORTED: {winner} is best at epistasis AND overall prediction.")
    else:
        overall_winner = 'EDS' if abs(eds_overall) > abs(pll_overall) else 'PLL'
        print(f"\n  ❌ H5 NOT SUPPORTED: Best epistasis model ({winner}) ≠ Best overall model ({overall_winner})")
    
    # --- Plot H5 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: EDS predicted vs experimental epistasis
    ax = axes[0]
    ax.scatter(comparison['eds_epistasis'], comparison['exp_epistasis'], 
               alpha=0.1, s=5, c='#2196F3')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    rho_e = results['EDS']['spearman_rho']
    ax.set_title(f"EDS: Predicted vs Exp Epistasis\nSpearman ρ = {rho_e:.4f}", fontsize=11)
    ax.set_xlabel("EDS Predicted Epistasis")
    ax.set_ylabel("Experimental Epistasis (DMS)")
    ax.grid(True, alpha=0.3)
    
    # Panel 2: PLL predicted vs experimental epistasis
    ax = axes[1]
    ax.scatter(comparison['pll_epistasis'], comparison['exp_epistasis'],
               alpha=0.1, s=5, c='#4CAF50')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    rho_p = results['PLL']['spearman_rho']
    ax.set_title(f"PLL: Predicted vs Exp Epistasis\nSpearman ρ = {rho_p:.4f}", fontsize=11)
    ax.set_xlabel("PLL Predicted Epistasis")
    ax.set_ylabel("Experimental Epistasis (DMS)")
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Summary bar chart
    ax = axes[2]
    methods = ['EDS', 'PLL']
    metrics = {
        'Epistasis ρ': [abs(results[m]['spearman_rho']) for m in methods],
        'Overall ρ': [abs(results[m]['overall_rho']) for m in methods],
        'Sign Agreement': [results[m]['sign_agreement'] for m in methods],
    }
    
    x = np.arange(len(methods))
    width = 0.25
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    for i, (metric_name, vals) in enumerate(metrics.items()):
        ax.bar(x + i*width - width, vals, width, label=metric_name, color=colors[i], alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('H5: Epistasis Accuracy Summary', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    fig.suptitle("H5: Best Epistasis Model = Most Trustworthy for Landscape?", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/H5_epistasis_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[H5] Plot saved: {output_dir}/H5_epistasis_accuracy.png")
    
    return results, comparison


def generate_summary_table(h4_results, h5_results, output_dir):
    """Generate a summary CSV with all hypothesis test results."""
    rows = []
    
    for method in ['EDS', 'PLL']:
        rows.append({
            'Method': method,
            'H4_rho_singles': h4_results[method]['rho_singles'],
            'H4_rho_doubles': h4_results[method]['rho_doubles'],
            'H4_delta_rho': h4_results[method]['delta_rho'],
            'H4_p_perm': h4_results[method]['p_perm'],
            'H5_epistasis_rho': h5_results[method]['spearman_rho'],
            'H5_sign_agreement': h5_results[method]['sign_agreement'],
            'H5_overall_rho': h5_results[method]['overall_rho'],
        })
    
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{output_dir}/Hypothesis_Test_Summary.csv", index=False)
    print(f"\n[Summary] Results saved: {output_dir}/Hypothesis_Test_Summary.csv")
    
    # Pretty print
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(summary.to_string(index=False, float_format='%.4f'))
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="H4/H5 Hypothesis Testing")
    parser.add_argument("--eds_csv", default="results/Epistasis_EDS_650M.csv",
                        help="Path to raw EDS scores CSV")
    parser.add_argument("--pll_csv", default="results/Epistasis_PLL_ESM1v.csv",
                        help="Path to raw PLL scores CSV")
    parser.add_argument("--dms_csv", default="data/A4_HUMAN_Seuma_2021.csv",
                        help="Path to experimental DMS data")
    parser.add_argument("--eds_analysis", default="results/Epistasis_EDS_650M_Analysis.csv",
                        help="Path to EDS epistasis analysis CSV")
    parser.add_argument("--pll_analysis", default="results/Epistasis_PLL_ESM1v_Analysis.csv",
                        help="Path to PLL epistasis analysis CSV")
    parser.add_argument("--output_dir", default="results",
                        help="Output directory for plots and results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    merged = load_data(args.eds_csv, args.pll_csv, args.dms_csv)
    print(f"Loaded {len(merged)} mutants with all scores.")
    
    # H4
    h4_results = test_h4(merged, args.output_dir)
    
    # H5
    h5_results, comparison = test_h5(merged, args.eds_analysis, args.pll_analysis, args.output_dir)
    
    # Summary
    summary = generate_summary_table(h4_results, h5_results, args.output_dir)
    
    print("\n✅ Hypothesis testing complete!")


if __name__ == "__main__":
    main()
