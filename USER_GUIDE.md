# Biologist's Guide to the Protein Fitness Pipeline

## 🧬 Overview

This pipeline allows you to predict the "fitness" (stability/function) of protein mutations using state-of-the-art Artificial Intelligence models (ESM-2 and ESM-1v). Think of it as a virtual lab that can test thousands of mutations in hours instead of months.

### How it works (The Biology behind the AI)

The AI has been trained on millions of natural protein sequences (from bacteria to humans). It has learned the "grammar" of evolution—which amino acids tends to sit next to each other, which hydrophobic cores must be preserved, and which surface loops are flexible.

When we score a mutation, we are effectively asking the AI: **"Does this new sequence look like a valid, functional protein?"**

- **High Probability (`Score > 0` or close to 0)**: The mutation respects the protein's structural and evolutionary constraints. It is likely **functional**.
- **Low Probability (`Score < -10`)**: The mutation breaks a critical rule (e.g., putting a charged Arginine in a hydrophobic core). It is likely **unstable** or **non-functional**.

---

## 📂 Project Structure

When you unzip the project folder on your cluster or laptop, here are the important files:

- **`src/`**: The "Engine Room". Contains all the Python code. You generally don't need to touch this.
- **`data/`**: The "Input Folder". Place your mutation CSV files here.
  - _Example_: `A4_HUMAN_Seuma_2021.csv` (Your Alzheimer's dataset).
- **`results/`**: The "Output Folder". The pipeline will save scores and plots here.
  - _Output Files_: `scores.csv` (Raw numbers), `plots.png` (Graphs).
- **`model_cache/`**: The "Brain". Contains the massive downloaded A.I. weighted files (up to 3GB). **Do not delete this**, or the pipeline will try to download them again (which might fail on a cluster without internet).

---

## 🛠️ Step 1: Getting Ready

### 1. The Input File (.csv)

You need a CSV file in the `data/` folder. It **MUST** have these columns:

1.  **`mutant`**: The code for the mutation (e.g., `A42G` or `A42G:H45W` for multiples).
2.  **`mutated_sequence`**: The full amino acid sequence of the mutant.
    - _Note_: If you only have the mutant codes, the pipeline tries to reconstruct the sequence using the `reference_file` in the data folder, but providing the full sequence is safer.

### 2. The Command Line

You will run this pipeline using a "Terminal" or "Command Prompt".

1.  Navigate to the project folder:
    ```bash
    cd path/to/ESM
    ```

---

## 🧪 Step 2: Running Predictions

This is the main step. We use a script called `batch_scoring`.

### The "Magic Command"

Copy and paste this into your terminal to run the standard analysis:

```bash
python -m src.batch_scoring --input_csv data/YOUR_FILE.csv --output_csv results/my_results.csv --method MLLR
```

**Note**: By default (if you don't mention a model), this uses the **ESM-2 150M** model. It is fast and good for initial testing. If you want higher accuracy, see the "Choosing the Right Model" section below.

### Breaking down the arguments (What they mean)

- `--input_csv`: Where your data is.
  - _Example_: `data/A4_HUMAN_Seuma_2021.csv`
- `--output_csv`: Where you want the answers.
  - _Example_: `results/app_predictions_mllr.csv`
- `--method`: **Crucial Choice**. You have 3 options:
  1.  **`MLLR` (Recommended)**: "Masked Log-Likelihood Ratio". Best balance of speed and accuracy. It asks: _"Is this mutant better or worse than the wild-type at this specific position?"_
  2.  **`MM` (Fastest)**: "Masked Marginal". Very fast but less accurate. It treats the rest of the sequence as "Wild Type", so it misses interactions between multiple mutations.
  3.  **`PLL` (Most Rigorous)**: "Pseudo-Log-Likelihood". Looks at the _entire_ sequence context. Very accurate but slow (~100x slower). Only use this if you have a GPU.

### Choosing the Right Model (`--model`)

The pipeline supports three main "brains". Here is how to pick the right one:

1.  **ESM-2 150M (`default`)**

    - **Best for**: fast debugging, running on laptop CPU, or when you have limited GPU memory (~2GB).
    - **Accuracy**: Good, but not the best.
    - _Usage_: No extra arguments needed.

2.  **ESM-2 650M (`facebook/esm2_t33_650M_UR50D`)**

    - **Best for**: Production runs on HPC.
    - **Accuracy**: Significantly better than 150M.
    - _Trade-off_: Slower, needs ~4GB GPU memory.
    - _Usage_: Add `--model facebook/esm2_t33_650M_UR50D`.

3.  **ESM-1v (`facebook/esm1v_t33_650M_UR90S_1`)**

    - **Best for**: "Zero-shot" variant prediction. This model was trained specifically to predict mutations.
    - **Accuracy**: Often the state-of-the-art for this specific task.
    - _Trade-off_: Same speed as 650M, but requires a special flag.
    - _Usage_: Add `--model facebook/esm1v_t33_650M_UR90S_1 --model_type esm1v`.

    * _Usage_: Add `--model facebook/esm1v_t33_650M_UR90S_1 --model_type esm1v`.

---

## 📈 Step 3: Statistical Reporting (New!)

After your job finishes (Step 2), you can now generate a **"Nature-Paper-Level"** statistical report with one command. This compares your AI predictions to the real lab data.

```bash
python -m src.analysis \
    --results_csv results/raven_scores_mllr.csv \
    --truth_csv data/A4_HUMAN_Seuma_2021.csv \
    --output_report results/final_report.png
```

**What you get:**

1.  **Spearman Rho**: Ranking quality (Higher = Better).
2.  **AUC-ROC**: Ability to distinguish functional vs broken proteins (0.5 = Random, 1.0 = Perfect).
3.  **Top-100 Precision**: If you engineered the top 100 designs, how many would work?
4.  **Plots**: A beautiful image (`results/final_report.png`) with Scatter plots, ROC curves, and score distributions.

---

## 📊 Step 4: Epistasis Analysis (Interactions)

Biology is rarely simple. Sometimes, two mutations that are bad individually become good together (like a lock and key change). This is called **Epistasis**.

Run this command to find these hidden interactions:

```bash
python src/epistasis.py --input_csv results/my_results_from_step_2.csv --output_csv results/epistasis_analysis.csv
```

**How to read the "Epistasis" column:**

- **Zero ($\approx 0$)**: **Additivity**. The mutations don't talk to each other. The double mutant is exactly as bad as the sum of its parts. (Most common).
- **Positive ($> 0$)**: **Synergy / Compensatory**. The mutations help each other. Example: Mutation A breaks a bond, Mutation B restores it. The double mutant is surprisingly stable.
- **Negative ($< 0$)**: **Interference**. The mutations clash. Example: Mutation A makes the protein wobbly, Mutation B makes it wobbly, but together they make it unfold completely.

---

## 🌄 Step 5: Exploring the Fitness Landscape

This tool simulates evolution. It answers: _"How structurally robust is this protein locally around the wild-type?"_

It generates random mutants with increasing "evolutionary distance" (k=1, 5, 10, 20 mutations) and scores them.

```bash
python -m src.explore_landscape
```

**Interpretation:**

- **Steep Drop-off**: The protein is fragile. A few random mutations destroy its function (common in active sites).
- **Shallow Decay**: The protein is robust/tolerant. It resembles a neutral network where function is maintained despite sequence drift (common in intrinsically disordered regions).
- **Plot**: Check `results/landscape_plot_mllr.png` to see this decay curve.

---

## 🛑 Troubleshooting

- **Error: "ModuleNotFoundError: No module named 'src'"**
  - _Fix_: You forgot the `-m`. Run `python -m src.batch_scoring ...` instead of `python src/batch_scoring.py ...`.
- **Error: "CUDA out of memory"**
  - _Fix_: Your batch size is too big for the GPU. Add `--batch_size 16` or `--batch_size 8` to your command.
- **Error: "Reference file failure"**
  - _Fix_: Your input CSV is missing the `mutated_sequence` column, and the pipeline couldn't find the Reference CSV to rebuild it. Ensure `ProteinGym_reference_file_substitutions.csv` is in `data/`.

---

## ✅ Step 6: The "Pre-Flight" Safety Check (Recommended)

Before you upload everything to the expensive Supercomputer or buy GPU credits, run this test locally on your laptop.

It checks if your data is readable, if the code works, and if it can save files—all without using a GPU.

```bash
python -m src.verify_pipeline
```

**If this says "🎉 SUCCESS"**, your code is bulletproof. You can upload it to the HPC/Colab with confidence. If it fails, it will tell you exactly what is missing (e.g., "Input CSV not found").

---

## 🔒 HPC Safety & Etiquette (For New Users)

If you are running this on a university supercomputer (HPC) for the first time, don't worry!

1.  **Isolation**: When you submit a job (using Slurm/PBS), the system gives you a **Private GPU**. No one else can touch it while your job is running.
2.  **Safety**: The command `torch.cuda.empty_cache()` inside the script only clears the memory of **your specific job**. It effectively "takes out the trash" for your process. It does **not** affect other students or the main system.
3.  **Good Manners**:
    - This script automatically releases the GPU when it finishes.
    - It cleans up its own memory.
    - You are being a "Good Citizen" by using this code!

---

## 🦅 Raven HPC Guide (MPCDF Specific)

Since you are using the **Raven** cluster, follow these specific steps:

### 1. Where to put files (Crucial!)

Raven does **not** allow running code from your Home directory (`~`) on compute nodes. You **MUST** use `/ptmp`.

- **Good**: `/ptmp/your_username/ESM_Project`
- **Bad**: `/u/your_username/ESM_Project` (This will crash with "File not found")

### 2. Setup Command

Login to Raven and run this to copy your files and install python:

```bash
# 1. Go to your temporary scratch space
cd /ptmp/$USER
mkdir ESM_Project
cd ESM_Project

# 2. Copy your files from your laptop (using SCP or Drag-and-Drop in MobaXterm)
# Upload to this exact folder.

# 3. Load Modules & Install
module load anaconda/3/2023.03
module load cuda/12.1 # Raven supports CUDA 13, but PyTorch likes 11/12
pip install -r requirements.txt
```

### 3. Submit the Job

I have created a special file for you: `RAVEN_JOB.sh`.

Run this command to send your job to the supercomputer:

```bash
sbatch RAVEN_JOB.sh
```

**Check status**:

```bash
squeue -u $USER
```

### 4. Interactive Testing (No Slurm)

If you are strictly testing and **cannot use sbatch**, run this script instead. It runs on the Login/Interactive node using **CPU only** (safe for admins) and tests just 2 mutations.

```bash
# 1. Make the script executable
chmod +x RAVEN_INTERACTIVE_TEST.sh

# 2. Run it
./RAVEN_INTERACTIVE_TEST.sh
```

**If this passes**: Your code is perfect. You can then ask your admin for permission to run the full `sbatch RAVEN_JOB.sh`.
