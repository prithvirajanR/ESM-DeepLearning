# Dataset Selection & Acquisition: A Detailed Retrospective

This document details the step-by-step process we followed to select and acquire the `A4_HUMAN_Seuma_2021` dataset for the Protein Fitness Landscape project.

## 1. The Source: ProteinGym
We chose **ProteinGym** as our source.
*   **What it is:** A standardized collection of Deep Mutational Scanning (DMS) assays.
*   **Why:** It provides a unified format (CSV with `mutant`, `mutated_sequence`, `DMS_score`) and high-quality, experimentally verified ground truth data. It is the gold standard for benchmarking protein models.

## 2. The Filtering Process (The "Human Part")
Our goal was to find a dataset that was both **biologically relevant** (Human) and **complex enough** (Double Mutants) to test our models.

### Step A: Accessing the Reference File
We downloaded the `ProteinGym_reference_file_substitutions.csv`. This metadata file lists every available dataset with columns like:
*   `DMS_id`: The unique ID of the dataset.
*   `taxon`: The organism (e.g., Human, Virus, Bacteria).
*   `includes_multiple_mutants`: Boolean flag.

### Step B: Filtering for Human Proteins
We ran a script (`src/explore_human_data.py`) to filter this list.
*   **Criteria:** `taxon == "Human"`.
*   **Result:** This narrowed down the hundreds of datasets to a subset of human-specific proteins (e.g., TP53, PTEN, BRCA1).
*   **Why:** Human proteins are more relevant for medical applications and drug discovery.

### Step C: The "Double Mutant" Requirement
This was the critical filter.
*   **The Problem:** Most DMS datasets only contain **single amino acid substitutions** (e.g., "A123C").
*   **Our Need:** We wanted to study **Epistasis** (how mutations interact). To do this, we needed **double mutants** (e.g., "A123C:D145E").
*   **The Search:** We specifically looked for datasets where the number of double mutants was significant.
*   **Findings:** Many famous datasets (like TP53) were mostly single mutants. We needed one with a rich "combinatorial" landscape.

## 3. The Selection: Amyloid Precursor Protein (APP)
Out of the candidates, we selected `A4_HUMAN_Seuma_2021`.

### Why APP?
1.  **Clinical Significance:**
    *   **Protein:** Amyloid Beta A4 Precursor Protein.
    *   **Disease:** Alzheimer's Disease.
    *   **Relevance:** Mutations in this protein are directly linked to early-onset Alzheimer's. Understanding which mutations break it (or make it worse) is a high-value scientific question.

2.  **Data Composition (The "Goldilocks" Mix):**
    *   **Total Mutants:** ~14,483.
    *   **Single Mutants:** ~468. (Enough for a solid baseline validation).
    *   **Double Mutants:** ~14,015. (A massive amount of data for testing epistasis).
    *   **Why this matters:** This ratio is rare. It gives us a perfect training ground: validate on the easy singles, then explore the hard doubles.

3.  **Sequence Length:**
    *   It is a long protein (~770 AA). This makes it computationally challenging (hence the need for efficient code) but biologically realistic.

## 4. The Download Process
1.  **Tool:** We used the `huggingface_hub` library (specifically `hf_hub_download`).
2.  **Repo:** `OATML-Markslab/ProteinGym`.
3.  **File:** `substitutions/A4_HUMAN_Seuma_2021.csv`.
4.  **Verification:** We wrote a script to load the CSV and verify we could reconstruct the full mutant sequences from the wild-type sequence using the mutation codes (e.g., verifying that at position 673, the wild type was indeed Alanine before applying the mutation).

## Summary
We didn't just pick a random file. We systematically filtered for **Human** origin + **Epistatic Potential** (Double Mutants) and selected **APP** for its high medical relevance and perfect data distribution.
