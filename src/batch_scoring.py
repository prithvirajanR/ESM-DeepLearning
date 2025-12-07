import argparse
import pandas as pd
import torch
import os
import logging
from tqdm import tqdm
from src.models import get_model
from src.scoring import MaskedMarginalScoring, PseudoLogLikelihoodScoring, LogLikelihoodRatioScoring, MaskedLogLikelihoodRatioScoring
from src.data_loader import DMSDataset

def main():
    parser = argparse.ArgumentParser(description="Batch scoring for protein mutations.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV with mutants.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save results.")
    parser.add_argument("--model", type=str, default="facebook/esm2_t30_150M_UR50D", help="HuggingFace model name.")
    parser.add_argument("--model_type", type=str, default="esm2", help="Model type (esm2/esm1v).")
    parser.add_argument("--method", type=str, required=True, choices=["MM", "PLL", "LLR", "MLLR"], help="Scoring method.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for PLL internal masking.")
    parser.add_argument("--start_index", type=int, default=0, help="Manually set start index (override checkpoint).")
    parser.add_argument("--reference_csv", type=str, default="f:/ESM/data/ProteinGym_reference_file_substitutions.csv", help="Path to reference CSV for WT sequence.")
    parser.add_argument("--max_samples", type=int, default=None, help="Stop after processing N samples (for testing).")
    
    args = parser.parse_args()
    
    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.output_csv + ".log"),
            logging.StreamHandler()
        ]
    )
    
    # 1. Load Data
    logging.info(f"Loading data from {args.input_csv}...")
    
    # Check if we need to use DMSDataset to reconstruct sequences
    try:
        df_peek = pd.read_csv(args.input_csv, nrows=1)
        has_mutated_seq = "mutated_sequence" in df_peek.columns
    except pd.errors.EmptyDataError:
        logging.error("Input file is empty.")
        return

    if has_mutated_seq:
        df = pd.read_csv(args.input_csv)
        wt_seq = None # We might need to look this up if method is like LLR but we'll deal with it later
    else:
        logging.info(f"'mutated_sequence' column missing. Using DMSDataset with reference {args.reference_csv}...")
        try:
             # We use the full dataset loader which handles WT and mutation applying
             dataset = DMSDataset(args.input_csv, args.reference_csv)
             mutants_list = dataset.get_mutants()
             # Reconstruct DataFrame with necessary columns
             # get_mutants returns [(mutant, mutated_sequence, score), ...]
             df = pd.DataFrame(mutants_list, columns=['mutant', 'mutated_sequence', 'DMS_score'])
             wt_seq = dataset.target_seq
             logging.info(f"Loaded {len(df)} sequences. WT length: {len(wt_seq)}")
        except Exception as e:
             raise ValueError(f"Failed to load dataset via DMSDataset: {e}. Ensure reference file exists.")

    if "mutated_sequence" not in df.columns:
         raise ValueError("Processed dataframe still lacks 'mutated_sequence'. DMSDataset failed to generate it.")

    # 2. Checkpointing
    start_idx = args.start_index
    mode = 'w'
    header = True
    
    if os.path.exists(args.output_csv) and start_idx == 0:
        # Check if we can resume
        logging.info(f"Output file {args.output_csv} exists. Checking for resume...")
        try:
            # Robust extraction: just count lines to avoid parsing errors
            with open(args.output_csv, 'r') as f:
                # Subtract 1 for header
                row_count = sum(1 for _ in f) - 1
            
            if row_count > 0:
                logging.info(f"Resuming from index {row_count}...")
                start_idx = row_count
                mode = 'a'
                header = False
            else:
                logging.info("Output file present but empty. Starting from scratch.")
        except Exception as e:
            logging.warning(f"Warning: Could not read existing output file ({e}). Starting from scratch.")
    
    if start_idx >= len(df):
        logging.info("Nothing to process.")
        return

    # Apply max_samples limit
    end_idx = len(df)
    if args.max_samples is not None:
        end_idx = min(start_idx + args.max_samples, len(df))
        logging.info(f"Limiting processing to {args.max_samples} samples (indices {start_idx} to {end_idx}).")

    # 3. Load Model
    logging.info(f"Loading model {args.model} ({args.model_type})...")
    model_wrapper = get_model(args.model, args.model_type)
    model_wrapper.load_model()
    
    # 4. Setup Scorer
    if args.method == "MM":
        scorer = MaskedMarginalScoring()
    elif args.method == "PLL":
        scorer = PseudoLogLikelihoodScoring()
    elif args.method == "LLR":
        scorer = LogLikelihoodRatioScoring()
    elif args.method == "MLLR":
        scorer = MaskedLogLikelihoodRatioScoring()
    else:
        raise ValueError(f"Unknown method {args.method}")

    logging.info(f"Starting scoring with {args.method} from index {start_idx}...")
    
    # WT Fallback if not loaded via DMSDataset above
    if wt_seq is None and args.method == "LLR":
         logging.warning("LLR selected but WT sequence unknown. Trying to load reference anyway...")
         dataset = DMSDataset(args.input_csv, args.reference_csv)
         wt_seq = dataset.target_seq

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

    with open(args.output_csv, mode, buffering=1) as f:
        if header:
            f.write("mutant,score,method\n")
            
        for i in tqdm(range(start_idx, end_idx)):
            row = df.iloc[i]
            mutant_code = row['mutant']
            mut_seq = row['mutated_sequence']
            
            # WT Fallback
            current_wt = wt_seq if wt_seq else mut_seq 

            # 1. Safety: Input Sanitization
            if not set(mut_seq).issubset(valid_aa):
                logging.warning(f"Skipping {mutant_code}: Contains invalid amino acids.")
                f.write(f"{mutant_code},InvalidAA,{args.method}\n")
                continue

            # 2. Safety: Sequence Length
            if len(current_wt) > 1022:
                 if i == start_idx: 
                     logging.warning(f"Sequence length {len(current_wt)} is close to/Exceeds model limit (1024). Truncation may occur.") 
            
            try:
                # 3. Robust Execution with OOM Catching
                try:
                    score_val = scorer.score(model_wrapper, current_wt, mutant_code, mut_seq)
                except torch.cuda.OutOfMemoryError:
                     logging.error(f"OOM Error at {mutant_code}. Clearing Cache and retrying once...")
                     torch.cuda.empty_cache()
                     try:
                         score_val = scorer.score(model_wrapper, current_wt, mutant_code, mut_seq)
                     except Exception as e:
                         logging.error(f"Retry failed for {mutant_code}: {e}")
                         score_val = None

                if score_val is not None:
                    f.write(f"{mutant_code},{score_val},{args.method}\n")
                else:
                    f.write(f"{mutant_code},NaN,{args.method}\n")
                    
            except Exception as e:
                logging.error(f"Error at index {i} ({mutant_code}): {e}")
                f.write(f"{mutant_code},Error,{args.method}\n")
            
            # 4. Periodic Flush & Cleanup
            if i % 10 == 0:
                f.flush()
                
            if i % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logging.info("Done!")

if __name__ == "__main__":
    main()
