import os
import pandas as pd
import scipy.stats
from tqdm import tqdm
import time
from src.data_loader import DMSDataset
from src.models import get_model
from src.scoring import MaskedMarginalScoring, PseudoLogLikelihoodScoring, LogLikelihoodRatioScoring

def validate_dataset(dataset_path, ref_path, model_name, model_type="esm2", max_mutants=None, scoring_methods=None):
    print(f"--- Starting Validation (Modular) ---")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_name}")
    
    if scoring_methods is None:
        # Default to Masked Marginal only for speed
        scoring_methods = [MaskedMarginalScoring()]
        print("No scoring methods specified. Defaulting to Masked Marginal.")
    
    print(f"Scoring Methods: {[m.name for m in scoring_methods]}")

    # 1. Load Data
    dataset = DMSDataset(dataset_path, reference_file_path=ref_path)
    mutants = dataset.get_mutants()
    
    if max_mutants:
        print(f"Limiting to first {max_mutants} mutants.")
        mutants = mutants[:max_mutants]
    
    print(f"Total mutants to score: {len(mutants)}")
    
    # 2. Load Model using Factory
    model_wrapper = get_model(model_name, model_type)
    model_wrapper.load_model()
    
    # 3. Scoring Loop
    results = []
    wt_seq = dataset.target_seq
    if not wt_seq:
        raise ValueError("Target sequence (WT) not found!")

    print("Scoring mutants...")
    start_time = time.time()
    
    for i, (mut_code, mut_seq, dms_score) in tqdm(enumerate(mutants), total=len(mutants)):
        try:
            row = {
                "mutant": mut_code,
                "dms_score": dms_score
            }
            
            # Run each scoring method
            for method in scoring_methods:
                try:
                    score = method.score(model_wrapper, wt_seq, mut_code, mut_seq)
                    row[f"score_{method.name}"] = score
                except Exception as e:
                    # print(f"Error in {method.name} for {mut_code}: {e}")
                    row[f"score_{method.name}"] = None
            
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {mut_code}: {e}")
            continue

    end_time = time.time()
    print(f"Scoring finished in {end_time - start_time:.2f} seconds.")
    
    # 4. Analysis
    df = pd.DataFrame(results)
    
    print("\n--- Results (Spearman Correlation) ---")
    for method in scoring_methods:
        col_name = f"score_{method.name}"
        if col_name in df.columns:
            df_valid = df.dropna(subset=[col_name])
            if not df_valid.empty:
                corr, pval = scipy.stats.spearmanr(df_valid['dms_score'], df_valid[col_name])
                print(f"{method.name}: rho={corr:.4f}, p={pval:.4e} (n={len(df_valid)})")
            else:
                print(f"{method.name}: No valid scores.")
        
    # Save results
    output_file = f"f:/ESM/results/validation_modular_{model_name.replace('/', '_')}.csv"
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "f:/ESM/data/A4_HUMAN_Seuma_2021.csv"
    REF_PATH = "f:/ESM/data/ProteinGym_reference_file_substitutions.csv"
    MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
    
    # Define which methods to run
    # We can easily add PLL or LLR here now
    methods = [
        MaskedMarginalScoring(),
        # PseudoLogLikelihoodScoring(), # Uncomment to run PLL
        # LogLikelihoodRatioScoring()   # Uncomment to run LLR
    ]
    
    validate_dataset(DATASET_PATH, REF_PATH, MODEL_NAME, max_mutants=50, scoring_methods=methods)
