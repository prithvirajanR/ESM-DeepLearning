import os
import pandas as pd
import scipy.stats
from tqdm import tqdm
import time
import torch
from src.data_loader import DMSDataset
from src.models import ESM2Model, ESM1vModel
from src.scoring import calculate_pll, calculate_llr, calculate_masked_marginal

def validate_singles_only(dataset_path, ref_path, model_name, model_type="esm2"):
    print(f"--- Starting Validation (Singles Only) ---")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_name}")
    
    # 1. Load Data
    dataset = DMSDataset(dataset_path, reference_file_path=ref_path)
    all_mutants = dataset.get_mutants()
    
    # Filter for single mutants only
    mutants = [(code, seq, score) for code, seq, score in all_mutants if ':' not in code]
    
    print(f"Total mutants in dataset: {len(all_mutants)}")
    print(f"Single mutants to score: {len(mutants)}")
    
    # 2. Load Model
    if model_type == "esm2":
        model_wrapper = ESM2Model(model_name)
    elif model_type == "esm1v":
        model_wrapper = ESM1vModel(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model_wrapper.load_model()
    
    # 3. Scoring Loop
    results = []
    
    # We need the Wild Type sequence for Masked Marginal
    wt_seq = dataset.target_seq
    if not wt_seq:
        raise ValueError("Target sequence (WT) not found in reference file!")

    print("Scoring single mutants with Masked Marginal...")
    start_time = time.time()
    
    for i, (mut_code, mut_seq, dms_score) in tqdm(enumerate(mutants), total=len(mutants)):
        try:
            # Skip if mutant sequence is invalid or empty
            if not mut_seq:
                continue
                
            # Masked Marginal scoring
            mm_score = None
            try:
                # Parse mutation code (e.g. "D672N")
                wt_aa = mut_code[0]
                pos_str = mut_code[1:-1]
                pos = int(pos_str) - 1  # 1-indexed to 0-indexed
                mut_aa = mut_code[-1]
                
                mm_score = calculate_masked_marginal(model_wrapper, wt_seq, pos, mut_aa)
            except Exception as e_mm:
                print(f"MM Error for {mut_code}: {e_mm}")
                pass
            
            results.append({
                "mutant": mut_code,
                "dms_score": dms_score,
                "mm_score": mm_score
            })
            
        except Exception as e:
            print(f"Error scoring {mut_code}: {e}")
            continue

    end_time = time.time()
    print(f"\nScoring finished in {end_time - start_time:.2f} seconds.")
    
    # 4. Analysis
    df = pd.DataFrame(results)
    
    # Calculate Correlation
    print("\n--- Results (Spearman Correlation) ---")
    
    # Masked Marginal vs DMS
    df_mm = df.dropna(subset=['mm_score'])
    if not df_mm.empty:
        corr_mm, p_value = scipy.stats.spearmanr(df_mm['dms_score'], df_mm['mm_score'])
        print(f"Masked Marginal Correlation: {corr_mm:.4f} (p={p_value:.4e})")
        print(f"Number of mutants scored: {len(df_mm)}")
    else:
        print("Masked Marginal: No valid single mutants scored.")
        
    # Save results
    output_file = f"f:/ESM/results/singles_validation_{model_name.replace('/', '_')}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")
    
    return df, corr_mm if not df_mm.empty else None

if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "f:/ESM/data/A4_HUMAN_Seuma_2021.csv"
    REF_PATH = "f:/ESM/data/ProteinGym_reference_file_substitutions.csv"
    MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
    
    df, corr = validate_singles_only(DATASET_PATH, REF_PATH, MODEL_NAME)
