# 🦅 Raven Pipeline Cheat Sheet

Here is every way you can run your project, from "Testing" to "Full Science".

## 1. The "Standard Production" Run (Do this 99% of the time)

**Goal:** Get scores for my thousands of APP mutations.
**Where:** On the Compute Nodes (via Slurm).
**How:**

```bash
sbatch RAVEN_JOB.sh
```

**Output:** `results/raven_scores_mllr.csv` AND `results/final_report.png`
_(Note: This script automatically runs the scoring AND the statistical analysis)._

---

## 2. The "Safety Check" Run (Do this first)

**Goal:** Prove code works, check for internet, download models (if not uploaded).
**Where:** On the Login Node (Interactively).
**How:**

```bash
./RAVEN_INTERACTIVE_TEST.sh
```

**Output:** "Success" message on screen. No saved files of importance.

---

## 3. The "Deep Science" Runs (Optional Post-Processing)

### A. Epistasis Analysis

**Goal:** Find pairs of mutations that interact (Synergy/Interference).
**Run after:** Step 1 is finished.
**How:**

```bash
# Need an interactive session or run locally
python -m src.epistasis \
    --input_csv results/raven_scores_mllr.csv \
    --output_csv results/epistasis.csv
```

### B. Landscape Exploration

**Goal:** Simulate evolution. See how fast the protein breaks if you mutate it randomly.
**How:**

```bash
python -m src.explore_landscape
```

**Output:** `results/landscape_plot.png`

---

## 4. The "Manual Override" (For Experts)

**Goal:** I want to change the model or method manually.
**How:** Edit `RAVEN_JOB.sh` and change the arguments:

- **Method Check:** Change `--method MLLR` to `--method PLL` (Slower, more rigorous).
- **Model Check:** Change `--model ...150M...` to `--model ...650M...` (Better accuracy).
