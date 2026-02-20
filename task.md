# Project Roadmap: ESM Protein Fitness Landscape

## ✅ Phase 1: Local Setup & Initial Validation (Completed)

- [x] **Environment Setup**
  - [x] Install dependencies (transformers, torch, etc.)
  - [x] Implement local model caching (`f:/ESM/model_cache`)
- [x] **Data Acquisition**
  - [x] Download ProteinGym dataset (APP - Alzheimer's)
  - [x] Verify sequence reconstruction
- [x] **Pipeline Development**
  - [x] Implement `ESM2` and `ESM1v` model wrappers
  - [x] Implement modular scoring strategies (Masked Marginal, PLL, LLR)
  - [x] Create validation scripts
- [x] **Initial Experiments (CPU)**
  - [x] Validate on Single Mutants (Masked Marginal) -> Result: $\rho \approx 0.25$
  - [x] Explore Landscape (Synthetic Mutants) -> Result: Flat landscape (needs better scoring)

## 🚀 Phase 2: High-Performance Computing (HPC/GPU)

- [x] **HPC Setup & Scripting**
  - [x] Create connection-safe scripts (`RAVEN_INTERACTIVE_TEST.sh`)
  - [x] Create production submission script (`RAVEN_JOB.sh`)
  - [x] Ensure robustness (OOM handling, logging, periodic flushing)
- [ ] **Production Run**
  - [x] **Run MLLR on full dataset** (Completed: Job 23251726 - Spearman 0.43)
  - [x] **Run Analysis** (Completed)
- [ ] **Comparative Analysis**
  - [x] **Run MLLR Benchmarks** (Completed: ESM2-650M=0.43, ESM1v=0.29, ESM2-150M=0.37)
  - [/] **Run PLL Benchmarks** (Running: Optimized FP16+Vectorized)
    - [x] **150M** (Completed: Spearman 0.34, Top-100 Precision 0.73!)
    - [x] **650M** (Completed: Rho=0.43, Prec=0.69)
    - [x] **ESM-1v** (Completed: Spearman 0.24, Top-100 Precision 0.76)
  - [x] **Hybrid Analysis (150M)** (Completed: No synergy)
  - [x] **Run EDS Benchmarks** (Novel "Semantic Shift" method)
    - [x] 150M (Rho=0.27, Prec=0.87)
    - [x] **650M (Rho=0.39, Prec=0.93!)** - **CHAMPION** 🏆
    - [x] ESM-1v (Rho=0.34, Prec=0.89)
  - [ ] **Run "Robust" Ensemble MLLR** (New Probabilistic Method)
    - [ ] 150M (Run RAVEN_ROBUST_150M.sh)
    - [ ] 650M (Run RAVEN_ROBUST_650M.sh)
    - [ ] ESM-1v (Run RAVEN_ROBUST_ESM1v.sh)
  - [ ] **Epistasis Analysis**
    - [ ] Score all ~14,000 double mutants.
  - [ ] Calculate "Epistasis Score" = $Fitness(AB) - (Fitness(A) + Fitness(B))$.
  - [ ] Identify cases of strong positive/negative epistasis.

## 📊 Phase 3: Final Analysis & Reporting

- [ ] **Landscape Visualization**
  - [ ] Generate high-resolution 3D or heatmap plots of the fitness landscape.
- [ ] **Method Comparison**
  - [ ] Compare correlations of MM vs. PLL vs. LLR.
  - [ ] Determine the "Best" method for this protein.
- [ ] **Final Report**
  - [ ] Summarize biological findings (APP stability/function).
