import pandas as pd
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, mean_squared_error

def load_data(results_path, truth_path, truth_col='DMS_score', bin_col='DMS_score_bin'):
    """Loads prediction results and merges with ground truth."""
    pred_df = pd.read_csv(results_path)
    truth_df = pd.read_csv(truth_path)
    
    # Merge on 'mutant'
    # Check if 'mutant' column exists in both
    if 'mutant' not in pred_df.columns or 'mutant' not in truth_df.columns:
        raise ValueError("Both CSVs must have a 'mutant' column for merging.")

    merged = pd.merge(pred_df, truth_df, on='mutant', how='inner', suffixes=('_pred', '_truth'))
    
    # Filter out NaNs or errors
    merged = merged[pd.to_numeric(merged['score'], errors='coerce').notnull()]
    merged['score'] = merged['score'].astype(float)
    
    return merged

def calculate_metrics(df, pred_col='score', truth_col='DMS_score', bin_col='DMS_score_bin'):
    """Calculates comprehensive statistics."""
    stats = {}
    
    # 1. Ranking (Spearman)
    corr, pval = spearmanr(df[truth_col], df[pred_col])
    stats['Spearman_Rho'] = corr
    stats['P_Value'] = pval
    
    # 2. Regression Error (RMSE)
    mse = mean_squared_error(df[truth_col], df[pred_col])
    stats['RMSE'] = np.sqrt(mse)
    
    # 3. Classification (AUC-ROC)
    if bin_col in df.columns:
        # Check if bin_col has at least 2 classes
        if len(df[bin_col].unique()) > 1:
            try:
                # Note: ESM scores are usually higher = better fitness, same as DMS.
                # If correlation is negative, we might need to flip, but normally they align.
                stats['AUC_ROC'] = roc_auc_score(df[bin_col], df[pred_col])
                
                precision, recall, _ = precision_recall_curve(df[bin_col], df[pred_col])
                stats['AUC_PR'] = auc(recall, precision)
            except Exception as e:
                 print(f"Warning: Could not calc classification metrics: {e}")
                 stats['AUC_ROC'] = np.nan

    # 4. Top-K Precision (Enrichment)
    # "If we pick the top 100 predictions, what % are actually positive in the lab?"
    if bin_col in df.columns:
        top_k = 100
        if len(df) > top_k:
            top_preds = df.nlargest(top_k, pred_col)
            true_positives = top_preds[bin_col].sum()
            stats[f'Top_{top_k}_Precision'] = true_positives / top_k
            
    return stats

def plot_report(df, stats, output_path, pred_col='score', truth_col='DMS_score', bin_col='DMS_score_bin'):
    """Generates a multi-panel report."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel A: Scatter Plot (Correlation)
    sns.scatterplot(data=df, x=truth_col, y=pred_col, alpha=0.3, ax=axes[0], hue=bin_col)
    axes[0].set_title(f"Prediction vs Ground Truth\nSpearman={stats['Spearman_Rho']:.3f}, RMSE={stats['RMSE']:.2f}")
    axes[0].set_xlabel("Real Fitness (DMS)")
    axes[0].set_ylabel("Predicted Score (ESM)")
    
    # Panel B: Histogram of Scores
    sns.histplot(data=df, x=pred_col, hue=bin_col, kde=True, ax=axes[1])
    axes[1].set_title("Score Distribution by Class\n(Separability)")
    
    # Panel C: ROC Curve
    if 'AUC_ROC' in stats and not np.isnan(stats['AUC_ROC']):
        fpr, tpr, _ = roc_curve(df[bin_col], df[pred_col])
        axes[2].plot(fpr, tpr, label=f"AUC = {stats['AUC_ROC']:.3f}")
        axes[2].plot([0, 1], [0, 1], 'k--')
        axes[2].set_xlabel("False Positive Rate")
        axes[2].set_ylabel("True Positive Rate")
        axes[2].set_title("ROC Curve (Classification Ability)")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No Binary Labels", ha='center')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Report figure saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate rigorous statistical report for ESM predictions.")
    parser.add_argument("--results_csv", type=str, required=True, help="Output from batch_scoring.")
    parser.add_argument("--truth_csv", type=str, required=True, help="Original Data CSV with DMS_score.")
    parser.add_argument("--output_report", type=str, default="results/analysis_report.png", help="Path for plot.")
    args = parser.parse_args()
    
    args = parser.parse_args()
    
    # Configure Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.results_csv + ".report.log"),
            logging.StreamHandler()
        ]
    )

    logging.info("Loading Data...")
    df = load_data(args.results_csv, args.truth_csv)
    logging.info(f"Matched {len(df)} mutants.")
    
    print("Calculating Metrics...")
    stats = calculate_metrics(df)
    
    print("\n" + "="*40)
    print("       STATISTICAL REPORT       ")
    print("="*40)
    for k, v in stats.items():
        print(f"{k:<20}: {v:.4f}")
    print("="*40)
    
    print("Generating Plots...")
    plot_report(df, stats, args.output_report)
    print("Done.")

if __name__ == "__main__":
    main()
