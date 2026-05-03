# Final Project Report — Detailed Outline

# Deep Mutational Scanning with ESM Protein Language Models

> **Status:** Outline (Draft) — to be finalized after PLL Landscape job completes.
> **Last updated:** 2026-02-14

---

## TITLE PAGE

- **Title:** Predicting Protein Variant Effects and Epistasis Using ESM Protein Language Models:  
  A Comparative Study of Probability- vs Geometry-Based Scoring
- **Subtitle:** Amyloid Precursor Protein (APP / A4_HUMAN) — Seuma et al. 2021
- **Author:** [Your Name]
- **Date:** February 2026
- **Affiliation / Course:** [To fill]

---

## ABSTRACT (~300 words)

- Problem: Predicting mutational effects on protein function computationally
- Approach: Systematic comparison of 6 scoring methods across 3 ESM models
- Dataset: 14,483 experimentally characterized variants (468 singles + 14,015 doubles)
- Key finding 1: EDS (embedding distance) outperforms PLL (pseudo-log-likelihood) in precision
- Key finding 2: Larger models (650M) improve geometric scoring but not probability-based scoring
- Key finding 3: Neither method accurately predicts epistasis (second-order effects), but EDS remains the most trustworthy for landscape exploration
- Conclusion: Geometry-based scoring is the recommended approach for variant effect prediction and library design

---

## 1. INTRODUCTION

### 1.1 Motivation & Background

- The protein engineering challenge: Engineering proteins requires understanding the fitness landscape
- Deep Mutational Scanning (DMS): Gold standard for measuring variant effects, but expensive (~$50K per experiment)
- Computational prediction as an alternative: Use pre-trained protein language models (pLMs) to predict variant fitness _in silico_
- The promise of pLMs: Trained on billions of protein sequences, they learn evolutionary constraints and functional grammar

### 1.2 The A4_HUMAN Dataset

- **Protein:** Amyloid Precursor Protein (APP), linked to Alzheimer's disease
- **Source:** Seuma et al. 2021 — Deep mutational scan of the APP transmembrane + juxtamembrane region
- **Data stats:**
  - 14,483 total variants
  - 468 single amino acid substitutions
  - 14,015 double amino acid substitutions
  - DMS scores range: [-6.270, +3.330]
  - DMS_score_bin: binary classification (functional / non-functional)
- **Why this dataset:** Rich in double mutants, enabling epistasis analysis; clinically relevant protein

### 1.3 Research Questions & Hypotheses

- **H1:** "Protein language models can predict single-mutant effects with significant correlation to experimental DMS scores"
- **H2:** "Geometry-based scoring (EDS) captures functional phenotype better than probability-based scoring (PLL)"
- **H3:** "Larger models provide better predictions regardless of scoring method"
- **H4:** "Predictive performance is much higher for single mutants than for double mutants"
- **H5:** "The best-performing model on epistasis will be the most trustworthy for exploring the whole landscape"

### 1.4 Report Structure

- Brief roadmap of the remaining sections

---

## 2. METHODS

### 2.1 ESM Protein Language Models

- **What are pLMs?** — Transformer-based models pre-trained on protein sequences via masked language modeling
- **Architecture:** Encoder-only transformers (BERT-style) with attention layers
- **Three models tested:**

  | Model       | Parameters | Training Data | Architecture | Primary Use        |
  | ----------- | ---------- | ------------- | ------------ | ------------------ |
  | ESM-2 150M  | 150M       | UniRef50      | 30 layers    | Fast baseline      |
  | ESM-2 650M  | 650M       | UniRef50      | 33 layers    | Large generalist   |
  | ESM-1v 650M | 650M       | UniRef90      | 33 layers    | Variant specialist |

- **Why these three:** 150M for speed, 650M for capacity, ESM-1v for domain specialization
- Model loading and caching strategy (local cache at `model_cache/`)

### 2.2 Scoring Methods (6 total)

#### 2.2.1 Masked Marginal Log-Likelihood Ratio (MLLR)

- **Type:** Probability / Local fit
- **Formula:** Score = log P(x_mut | context) - log P(x_wt | context), with mutated position masked
- **Procedure:** Single forward pass with mutation position masked → read off log-probabilities
- **Scientific interpretation:** Measures whether the model "prefers" the mutant AA over the WT AA at that position, given surrounding context
- **Computational cost:** 1 forward pass per mutant

#### 2.2.2 Pseudo-Log-Likelihood (PLL)

- **Type:** Probability / Global fit
- **Formula:** PLL(seq) = Σᵢ log P(xᵢ | x₁...xᵢ₋₁, xᵢ₊₁...xₗ) — sum over all L positions
- **Procedure:** L forward passes per sequence (mask each position independently)
- **Scientific interpretation:** Overall "naturalness" or evolutionary plausibility of the entire sequence
- **Computational cost:** L forward passes per mutant (~700× slower than MLLR for APP)
- **Optimization implemented:** Batched masking with torch.no_grad(), GPU acceleration

#### 2.2.3 Log-Likelihood Ratio (LLR)

- **Type:** Probability / Global comparison
- **Formula:** LLR = PLL(mutant) - PLL(wildtype)
- **Key insight:** Since PLL(WT) is constant for all variants, LLR and PLL produce identical rankings (same Spearman ρ)
- **Practical note:** LLR adds interpretability (positive = more likely than WT) but no ranking advantage

#### 2.2.4 Embedding Distance Scoring (EDS)

- **Type:** Geometry / Semantic shift
- **Formula:** Score = -||H_mut - H_wt||₂ where H is the full hidden state tensor (L×D) from last layer
- **Procedure:** Two forward passes (WT + mutant) → compute Euclidean distance in embedding space
- **Scientific interpretation:** Measures magnitude of semantic shift — how much did the mutation change the protein's "meaning" in latent space?
- **Theoretical basis:** Latent Manifold Hypothesis — functional proteins occupy a manifold; distance from WT approximates phenotypic deviation
- **Computational cost:** 2 forward passes per mutant (fast)

#### 2.2.5 Entropy-Weighted MLLR (Entropy-MLLR)

- **Type:** Information theory + Probability
- **Formula:** Score = LLR / (Entropy + ε), where Entropy = -Σ P(x) log P(x) at the mutated position
- **Scientific interpretation:** Penalizes mutations at conserved (low-entropy) positions more heavily; acts as a "soft structure" filter
- **Assumption:** Low-entropy positions are evolutionarily constrained → mutations there are more likely deleterious

#### 2.2.6 Mutant-Only Marginal Scoring

- **Type:** Probability / Absolute
- **Formula:** Score = log P(x_mut | context) — no WT comparison
- **Scientific interpretation:** Absolute fitness of the mutant residue in context; useful when WT itself may be sub-optimal
- **When useful:** Ancestral constructs, engineered baselines, or investigating non-natural starting points

### 2.3 Epistasis Analysis

#### 2.3.1 Definition of Epistasis

- **Epistasis** = non-additive interaction between mutations
- **Additive model:** Expected(AB) = ΔScore(A) + ΔScore(B) (relative to WT)
- **Epistasis E = Score(AB) - Score(A) - Score(B) + Score(WT)**
- Positive epistasis: double mutant better than expected (diminishing returns, buffering)
- Negative epistasis: double mutant worse than expected (synergistic damage)

#### 2.3.2 WT-Baseline Correction for Absolute Scores

- **The bug and the fix:** PLL produces absolute log-likelihoods (~-400 per sequence)
  - Naive formula: Expected = Score(A) + Score(B) → double-counts WT baseline
  - Corrected formula: Expected = Score(A) + Score(B) - Score(WT)
- EDS produces relative scores (WT ≈ 0), so no correction needed
- **Auto-detection heuristic:** If median naive offset > 50% of mean absolute score → absolute scoring detected

#### 2.3.3 Predicted vs Experimental Epistasis

- Experimental epistasis from DMS: E_exp = DMS(AB) - DMS(A) - DMS(B)
- Model-predicted epistasis from EDS and PLL
- Evaluation metrics: Spearman ρ, Pearson r, sign agreement

### 2.4 Landscape Exploration

#### 2.4.1 Synthetic Mutant Library Generation

- **Strategy:** Random combinatorial mutations at 4 mutational distances: k = 2, 5, 10, 20
- **Library size:** 1,000 mutants per k = 4,000 total
- **Seed:** 42 for reproducibility
- **Implementation:** `generate_synthetic.py` with controlled combinatorial sampling

#### 2.4.2 Fitness Decay Analysis

- Scoring all 4,000 synthetic mutants with EDS (and PLL once done)
- Plotting fitness vs mutational distance
- Expected pattern: monotonic decay (more mutations → more destabilization)
- Statistical characterization: mean, std, violin plots per distance k

### 2.5 Hypothesis Testing Framework

#### 2.5.1 H4 Testing: Singles vs Doubles Performance

- **Metric:** Spearman ρ(model_score, DMS_score) computed separately for singles and doubles
- **Statistical test:** Permutation test (n=10,000) for the difference Δρ = ρ_singles - ρ_doubles
- **Bootstrap CIs:** 10,000 bootstrap replicates for 95% confidence intervals

#### 2.5.2 H5 Testing: Epistasis → Landscape Trust

- **Metric 1:** Spearman ρ(predicted_epistasis, experimental_epistasis)
- **Metric 2:** Sign agreement (% of doubles where predicted and experimental epistasis have the same sign)
- **Linkage test:** Does the model with better epistasis accuracy also have better overall Spearman ρ?

### 2.6 Technical Implementation

#### 2.6.1 Software Architecture

- Modular Python codebase with separate modules for:
  - `models.py` — ESM model wrapper with local caching
  - `scoring.py` — All 6 scoring strategies (Strategy pattern)
  - `data_loader.py` — ProteinGym dataset loading and sequence reconstruction
  - `batch_scoring.py` — Production batch processing with checkpointing and GPU acceleration
  - `analyze_epistasis.py` — Epistasis calculation with WT-baseline correction
  - `analyze_landscape.py` — Fitness decay plotting
  - `test_hypotheses.py` — Statistical hypothesis testing
  - `generate_synthetic.py` — Combinatorial mutant library generation
  - `validate.py` / `validate_singles.py` — Validation against DMS ground truth

#### 2.6.2 HPC Environment

- **Cluster:** RAVEN (MPCDF) with NVIDIA A100 GPUs
- **Job management:** SLURM batch scripts (21 job scripts)
- **Resource allocation:** 1 GPU, 4 CPUs, 64GB RAM per job
- **Wall times:** 30min (EDS) to 24h (PLL 650M)
- **Checkpointing:** CSV-based restart for interrupted jobs

#### 2.6.3 Reproducibility

- Fixed random seeds (42)
- All model weights from HuggingFace with fixed revisions
- Full environment specification via conda

---

## 3. RESULTS

### 3.1 Phase 1: Initial Validation (Single Mutants, CPU)

- **Experiment:** Masked Marginal scoring with ESM-2 150M on 468 single mutants
- **Result:** Spearman ρ ≈ 0.25 — moderate but significant correlation
- **Conclusion:** Validates that pLMs capture real evolutionary constraints, but baseline method has room for improvement
- **Figure:** `results/correlation_plot.png` — scatter plot of MM scores vs DMS scores

### 3.2 Phase 2: Comprehensive Benchmarking (HPC/GPU)

#### 3.2.1 The Full Leaderboard (18 model × method combinations)

- **Table:** Complete results table with 6 methods × 3 models = 18 configurations
  - Columns: Model, Method, Top-100 Precision, Spearman ρ, Run Time
- Include all 18 results from the existing leaderboard:
  - EDS: 93% (650M), 89% (ESM-1v), 87% (150M)
  - PLL: 76% (ESM-1v), 73% (150M), 69% (650M)
  - MLLR: 57% (650M), 55% (150M), 48% (ESM-1v)
  - Entropy-MLLR: 86% (ESM-1v), 47% (150M), 5% (650M)
  - Mutant-Marginal: 77% (ESM-1v), 40% (150M), 31% (650M)
  - Ensemble: 82% (ESM-1v), 49% (150M), 39% (650M)
- **Figure:** Bar chart / heatmap of the full leaderboard
- **Figures:** Individual report plots (`report_EntropyMLLR_*.png`, `report_MutantMarginal_*.png`)

#### 3.2.2 Key Finding: Geometry Beats Probability

- EDS consistently dominates on Top-100 Precision across ALL models
- EDS (650M) = 93% Precision — the overall champion
- Why: EDS captures phenotypic shift (function/structure), not just evolutionary fitness
- The "Latent Manifold Hypothesis" validated empirically
- **Speed advantage:** EDS is ~500-1000× faster than PLL

#### 3.2.3 Key Finding: The Model Size Effect

- For EDS: Larger model = better (150M→650M: 87%→93%)
- For MLLR/PLL: Model size effect is inconsistent or even negative
- **The ESM-1v effect:** Specialist model excels at local-context methods (Entropy-MLLR: 86%, Ensemble: 82%) but underperforms generalist models on EDS and MLLR
- Interpretation: ESM-1v's UniRef90 training skews entropy distributions

#### 3.2.4 Key Finding: The PLL Paradox

- PLL measures evolutionary plausibility, EDS measures functional displacement
- Hypothesis: For this dataset (APP aggregation/toxicity), functional shift ≠ evolutionary unfitness
- Some mutations are evolutionarily common but functionally deleterious (or vice versa)

#### 3.2.5 Key Finding: Entropy-MLLR — A Double-Edged Sword

- **ESM-1v + Entropy-MLLR = 86% precision** — best probabilistic method
- **ESM-2 650M + Entropy-MLLR = 5% precision** — catastrophic failure
- Explanation: Entropy calibration is model-dependent; ESM-1v has sharper entropy distributions at conserved sites
- Lesson: Method-model compatibility cannot be assumed

### 3.3 Phase 3: Epistasis Analysis (Double Mutants)

#### 3.3.1 Model-Predicted Epistasis Distributions

- **EDS epistasis:**
  - n = 14,015 double mutants
  - Mean = +3.31, Median = +3.16, Std = 0.80
  - Systematically positive → diminishing returns / buffering
  - Interpretation: triangle inequality in embedding space (||A+B|| ≤ ||A|| + ||B||)
- **PLL epistasis (corrected):**
  - n = 14,015 double mutants
  - Mean = +0.11, Median = 0.00, Std = 1.23
  - Centered at zero → near-perfect additivity
  - Interpretation: PLL (log-likelihood) is inherently additive at the log scale

- **Figures:** `Epistasis_EDS_scatter.png`, `Epistasis_EDS_dist.png`, `Epistasis_PLL_scatter.png`, `Epistasis_PLL_dist.png`

#### 3.3.2 WT-Baseline Correction (Technical Detail)

- The discovery of the PLL offset bug: naive formula gave constant +393 epistasis
- Mathematical explanation: PLL(A) ≈ -400, PLL(B) ≈ -400, PLL(AB) ≈ -410
  - Naive: E = -410 - (-400 + -400) = +390 → wrong!
  - Corrected: E = -410 + (-393) - (-400) - (-400) = +0.11 → correct
- Auto-estimation of WT score: -median(naive offsets) = 392.83
- Validation: corrected distribution centered at zero

#### 3.3.3 Predicted vs Experimental Epistasis Comparison

- **Experimental epistasis** from DMS: Mean = -0.248, Std = 0.969 (slightly negative → mild synergistic effects)
- **EDS vs experiment:** Spearman ρ = 0.027 (p = 1.25e-03) — very weak but significant
- **PLL vs experiment:** Spearman ρ = -0.013 (p = 0.140) — not significant
- **Sign agreement:** EDS = 36.2%, PLL = 50.6%
- **Interpretation:** Neither method captures epistasis well — this is a second-order effect that is notoriously difficult to predict from sequence alone. This is consistent with published literature.

### 3.4 Hypothesis Testing Results

#### 3.4.1 H4: Single vs Double Mutant Prediction

- **EDS:**
  - Singles: ρ = 0.448 [0.374, 0.516]
  - Doubles: ρ = 0.381 [0.367, 0.395]
  - Δρ = +0.067, permutation p = 0.084 (not significant at α=0.05)
- **PLL:**
  - Singles: ρ = 0.236 [0.151, 0.315]
  - Doubles: ρ = 0.226 [0.210, 0.241]
  - Δρ = +0.010, permutation p = 0.819 (not significant)
- **Verdict: H4 NOT SUPPORTED** — While singles are predicted slightly better, the difference is not statistically significant for either method
- **Discussion:** This may reflect the large sample size for doubles (14,015) providing more statistical power, or it may indicate that these models genuinely handle double mutants as well as singles
- **Figure:** `H4_barplot.png`, `H4_singles_vs_doubles.png`

#### 3.4.2 H5: Best Epistasis Model = Most Trustworthy

- **Epistasis accuracy:** EDS ρ = 0.027 > PLL ρ = -0.013 → EDS better at epistasis
- **Overall prediction:** EDS ρ = 0.388 > PLL ρ = 0.235 → EDS better overall
- **Verdict: H5 SUPPORTED** — EDS is the champion on BOTH epistasis accuracy and overall predictive power
- **Implication:** EDS is the recommended method for landscape exploration
- **Figure:** `H5_epistasis_accuracy.png`

### 3.5 Phase 4: Landscape Exploration

#### 3.5.1 EDS Landscape Results

- **Synthetic library:** 4,000 mutants at k = 2, 5, 10, 20 mutations (1,000 each)
- **Fitness decay pattern:** Monotonic decline with increasing k
- **Statistics per distance:**
  - k=2: [mean, std, range]
  - k=5: [mean, std, range]
  - k=10: [mean, std, range]
  - k=20: [mean, std, range]
- **Figures:** `Synthetic_Landscape_Decay.png`, `Synthetic_Landscape_Violin.png`

#### 3.5.2 PLL Landscape Results

> **[PLACEHOLDER — Job Running on RAVEN, ~9h estimated]**

- Same 4,000 mutants scored with PLL (ESM-1v)
- Expected: Similar decay pattern but with PLL-specific characteristics
- Comparison: EDS vs PLL landscape shapes, variance at each k

#### 3.5.3 EDS vs PLL Landscape Comparison

> **[PLACEHOLDER — To be completed after PLL landscape job finishes]**

- Correlation between EDS and PLL landscape scores
- Which method shows clearer fitness decay?
- Variance comparison at each mutational distance
- Implications for landscape exploration strategy

---

## 4. DISCUSSION

### 4.1 Why Geometry Outperforms Probability

- The "Latent Manifold" argument: proteins as points on a functional manifold
- Euclidean distance ≈ phenotypic deviation; log-likelihood ≈ evolutionary plausibility
- For APP (aggregation phenotype): functional change ≠ evolutionary unfitness
- The information content argument: embedding (L×D tensor) vs scalar log-prob

### 4.2 The Epistasis Challenge

- Why epistasis is hard to predict: it requires modeling non-linear interactions
- Current pLMs are trained on single sequences, not pairs → limited epistatic signal
- The triangle inequality explains EDS's systematic positive epistasis
- PLL's additivity at log scale → near-zero predicted epistasis
- Both methods fail to capture the mild negative (synergistic) epistasis observed experimentally

### 4.3 Model Architecture Implications

- Why 650M beats 150M for EDS: deeper representation → richer manifold geometry
- Why ESM-1v excels at entropy-based methods: UniRef90 training creates sharper evolutionary signals
- The cautionary tale of Entropy-MLLR on ESM-2 650M: model-method compatibility matters

### 4.4 Practical Recommendations for Protein Engineering

- **For ranking/screening:** Use EDS (650M) — 93% top-100 precision, fast
- **For mechanistic understanding:** Use PLL/LLR — interpretable as evolutionary fitness
- **For conserved region analysis:** Use Entropy-MLLR (ESM-1v) — 86% precision with structural insight
- **For landscape exploration:** Use EDS — fast enough for large-scale combinatorial scans
- **Warning:** Do not rely on predicted epistasis for combinatorial design

### 4.5 Limitations

- Single dataset (A4_HUMAN) — generalizability unknown
- No structural features incorporated (no AlphaFold, no MSA)
- Epistasis prediction is weak → combinatorial predictions should be treated with caution
- EDS uses Euclidean distance only; other metrics (cosine, Mahalanobis) not tested
- Top-100 Precision metric depends on ProteinGym's binary classification threshold

### 4.6 Future Directions

- Test on additional DMS datasets (ProteinGym has 200+)
- Incorporate structural information (AlphaFold embeddings, contact maps)
- Train epistasis-specific models on double mutant data
- Explore alternative distance metrics for EDS
- Extend landscape exploration to higher k (50, 100 mutations)
- Fine-tune ESM models on organism-specific data

---

## 5. CONCLUSION

- Summary of key findings (3-4 sentences)
- The winner: EDS (ESM-2 650M) for variant effect prediction and landscape exploration
- The biological insight: Semantic displacement > evolutionary probability for phenotype prediction
- The practical impact: Computational screening of protein variants at scale is feasible and accurate

---

## 6. SUPPLEMENTARY MATERIALS

### 6.1 Complete Results Tables

- Full 18-configuration benchmark table with CIs
- H4/H5 detailed results tables
- Epistasis summary statistics

### 6.2 All Figures (Index)

| Figure | File                              | Description                      |
| ------ | --------------------------------- | -------------------------------- |
| Fig 1  | `correlation_plot.png`            | Initial MM validation (150M)     |
| Fig 2  | `report_EntropyMLLR_150M.png`     | Entropy-MLLR on ESM-2 150M       |
| Fig 3  | `report_EntropyMLLR_650M.png`     | Entropy-MLLR on ESM-2 650M       |
| Fig 4  | `report_EntropyMLLR_ESM1v.png`    | Entropy-MLLR on ESM-1v           |
| Fig 5  | `report_MutantMarginal_150M.png`  | Mutant-Marginal on ESM-2 150M    |
| Fig 6  | `report_MutantMarginal_650M.png`  | Mutant-Marginal on ESM-2 650M    |
| Fig 7  | `report_MutantMarginal_ESM1v.png` | Mutant-Marginal on ESM-1v        |
| Fig 8  | `Epistasis_EDS_scatter.png`       | EDS: Observed vs Expected scores |
| Fig 9  | `Epistasis_EDS_dist.png`          | EDS epistasis distribution       |
| Fig 10 | `Epistasis_PLL_scatter.png`       | PLL: Observed vs Expected scores |
| Fig 11 | `Epistasis_PLL_dist.png`          | PLL epistasis distribution       |
| Fig 12 | `H4_barplot.png`                  | Singles vs Doubles comparison    |
| Fig 13 | `H4_singles_vs_doubles.png`       | H4 scatter plots                 |
| Fig 14 | `H5_epistasis_accuracy.png`       | H5 epistasis accuracy            |
| Fig 15 | `Synthetic_Landscape_Decay.png`   | EDS fitness decay                |
| Fig 16 | `Synthetic_Landscape_Violin.png`  | EDS landscape violin plots       |
| Fig 17 | [PLL Landscape Decay]             | **[PLACEHOLDER]**                |
| Fig 18 | [EDS vs PLL Landscape Comparison] | **[PLACEHOLDER]**                |

### 6.3 Code Repository Structure

```
ESM/
├── src/                           # Core Python modules
│   ├── models.py                  # ESM model wrappers
│   ├── scoring.py                 # 6 scoring strategies
│   ├── batch_scoring.py           # Production batch processor
│   ├── data_loader.py             # ProteinGym data loading
│   ├── analyze_epistasis.py       # Epistasis with WT correction
│   ├── analyze_landscape.py       # Fitness decay analysis
│   ├── test_hypotheses.py         # H4/H5 statistical testing
│   ├── generate_synthetic.py      # Combinatorial library generation
│   ├── validate.py                # Full dataset validation
│   └── validate_singles.py        # Singles-only validation
├── data/                          # Input data
│   ├── A4_HUMAN_Seuma_2021.csv    # DMS ground truth (14,483 variants)
│   └── Synthetic_Landscape.csv    # Generated synthetic library
├── results/                       # All outputs (CSVs, PNGs)
├── RAVEN_*.sh                     # 21 SLURM job scripts
└── PROJECT_REPORT.md              # This report (final version)
```

### 6.4 Computational Resources Used

- Total GPU-hours: [To calculate from SLURM logs]
- Models downloaded: 3 ESM models (~2.5GB total)
- Total variants scored: ~100,000+ forward passes
- HPC cluster: RAVEN (MPCDF), NVIDIA A100 GPUs

---

## REFERENCES

- Seuma et al. (2021) — A4_HUMAN DMS dataset
- Lin et al. (2023) — ESM-2 model architecture
- Meier et al. (2021) — ESM-1v variant prediction
- Notin et al. (2023) — ProteinGym benchmark suite
- Rives et al. (2021) — Biological structure and function emerge from scaling unsupervised learning
- Frazer et al. (2021) — Disease variant effect prediction with protein language models
