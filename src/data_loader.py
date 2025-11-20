import pandas as pd
import os
from typing import List, Tuple
from huggingface_hub import hf_hub_download
import shutil

class DMSDataset:
    def __init__(self, file_path: str, reference_file_path: str = None):
        self.file_path = file_path
        self.reference_file_path = reference_file_path
        self.data = None
        self.target_seq = None
        self.dms_id = None
        
        self.load_data()
    
    def load_data(self):
        """Loads the dataset and reference info."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        self.data = pd.read_csv(self.file_path)
        
        # Extract DMS_id from filename (basename without extension)
        basename = os.path.basename(self.file_path)
        self.dms_id = os.path.splitext(basename)[0]
        
        if self.reference_file_path and os.path.exists(self.reference_file_path):
            ref_df = pd.read_csv(self.reference_file_path)
            # Find row with matching DMS_id
            row = ref_df[ref_df['DMS_id'] == self.dms_id]
            if not row.empty:
                self.target_seq = row.iloc[0]['target_seq']
            else:
                print(f"Warning: DMS_id {self.dms_id} not found in reference file.")
        
    def _apply_mutation(self, seq: str, mutation: str) -> str:
        """Applies a mutation (e.g., 'H24C') to the sequence."""
        if mutation == "WT":
            return seq
        
        # Handle multiple mutations separated by colon or similar if present
        # ProteinGym usually uses colon for multiples? Or just single mutants in this file?
        # The file has 'DMS_number_multiple_mutants: 0', so likely single mutants.
        # But let's be robust.
        
        muts = mutation.split(':')
        s_list = list(seq)
        
        for mut in muts:
            try:
                wt_aa = mut[0]
                new_aa = mut[-1]
                pos = int(mut[1:-1]) - 1 # 1-indexed to 0-indexed
                
                if 0 <= pos < len(s_list):
                    if s_list[pos] != wt_aa:
                        # print(f"Warning: WT AA at {pos+1} is {s_list[pos]}, expected {wt_aa}")
                        pass
                    s_list[pos] = new_aa
                else:
                    # print(f"Warning: Position {pos+1} out of bounds.")
                    pass
            except ValueError:
                pass
                
        return "".join(s_list)

    def get_mutants(self) -> List[Tuple[str, str, float]]:
        """Returns a list of (mutant_code, mutant_sequence, fitness_score)."""
        if self.data is None:
            return []
        
        mutants = []
        for _, row in self.data.iterrows():
            mut_code = row['mutant']
            score = row['DMS_score']
            
            if self.target_seq:
                mut_seq = self._apply_mutation(self.target_seq, mut_code)
            else:
                mut_seq = "" # Cannot reconstruct without target seq
            
            mutants.append((mut_code, mut_seq, score))
            
        return mutants

def download_dataset(dataset_name: str, output_dir: str) -> str:
    """
    Downloads a specific dataset from ProteinGym (Hugging Face) using hf_hub_download.
    Also downloads the reference file if not present.
    Returns the path to the saved CSV file.
    """
    print(f"Downloading {dataset_name} via Hugging Face Hub...")
    
    # Ensure reference file is present
    ref_filename = "ProteinGym_reference_file_substitutions.csv"
    ref_path = os.path.join(output_dir, ref_filename)
    if not os.path.exists(ref_path):
        print("Downloading reference file...")
        try:
            cached_ref = hf_hub_download(
                repo_id="OATML-Markslab/ProteinGym",
                filename=ref_filename,
                repo_type="dataset"
            )
            shutil.copy(cached_ref, ref_path)
        except Exception as e:
            print(f"Failed to download reference file: {e}")

    # The file structure in the repo is:
    # ProteinGym_substitutions/BLAT_ECOLX_Stiffler_2015.csv
    if not dataset_name.endswith(".csv"):
        filename = f"{dataset_name}.csv"
    else:
        filename = dataset_name
        
    subpath = f"ProteinGym_substitutions/{filename}"
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}")
        return output_path
    
    try:
        # Download to cache
        cached_path = hf_hub_download(
            repo_id="OATML-Markslab/ProteinGym",
            filename=subpath,
            repo_type="dataset"
        )
        
        # Copy to output_dir
        shutil.copy(cached_path, output_path)
        print(f"Successfully downloaded to {output_path}")
        return output_path

    except Exception as e:
        print(f"Failed to download via HF Hub: {e}")
        raise
