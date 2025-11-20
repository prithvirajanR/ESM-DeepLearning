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
- [ ] **Advanced Scoring (The "Heavy Lifting")**
    - [ ] **Run Pseudo-Log-Likelihood (PLL)** on all 14,483 mutants.
        *   *Why:* PLL captures context better than Masked Marginal but is ~100x slower. Needs GPU.
    - [ ] **Run Log-Likelihood Ratio (LLR)**.
        *   *Why:* Normalizes for the wild-type probability.
- [ ] **Model Benchmarking**
    - [ ] Run **ESM-1v (650M)** on the full dataset.
        *   *Why:* Larger models usually predict fitness better.
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
