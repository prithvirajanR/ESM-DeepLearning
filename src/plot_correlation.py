import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

# Load results
df = pd.read_csv('f:/ESM/results/singles_validation_facebook_esm2_t30_150M_UR50D.csv')
df_valid = df.dropna(subset=['mm_score'])

# Calculate correlation
corr, pval = scipy.stats.spearmanr(df_valid['dms_score'], df_valid['mm_score'])

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_valid['dms_score'], df_valid['mm_score'], alpha=0.5, s=20)

# Add trend line
z = np.polyfit(df_valid['dms_score'], df_valid['mm_score'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_valid['dms_score'].min(), df_valid['dms_score'].max(), 100)
plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Trend line')

plt.xlabel('Real Lab Fitness Score (DMS)', fontsize=12)
plt.ylabel('ESM-2 Predicted Score (Masked Marginal)', fontsize=12)
plt.title(f'ESM-2 Predictions vs Real Fitness (APP Single Mutants)\nSpearman ρ = {corr:.4f}, p = {pval:.2e}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Save plot
output_path = 'f:/ESM/results/correlation_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")

# Print statistics
print(f"\n=== Validation Results ===")
print(f"Spearman Correlation: {corr:.4f}")
print(f"P-value: {pval:.4e}")
print(f"Mutants scored: {len(df_valid)}/{len(df)}")
print(f"\nInterpretation:")
if corr > 0.7:
    print("Strong positive correlation - Excellent predictions!")
elif corr > 0.4:
    print("Moderate positive correlation - Good predictions.")
elif corr > 0.2:
    print("Weak positive correlation - Limited predictive power.")
else:
    print("Very weak/no correlation - Poor predictions.")

print(f"\nStatistical significance: {'Yes (p < 0.05)' if pval < 0.05 else 'No'}")
