import random
from typing import List, Tuple

# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def generate_random_mutants(wt_sequence: str, k: int, n: int = 100, seed: int = 42) -> List[Tuple[str, str]]:
    """
    Generate n random mutants at exactly k mutations away from wild-type.
    
    Args:
        wt_sequence: Wild-type protein sequence
        k: Number of mutations (mutational distance)
        n: Number of random mutants to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of (mutant_code, mutant_sequence) tuples
    """
    random.seed(seed)
    mutants = []
    
    seq_len = len(wt_sequence)
    
    for i in range(n):
        # Choose k random positions to mutate
        positions = random.sample(range(seq_len), k)
        
        # Create mutant sequence
        mutant_seq = list(wt_sequence)
        mutant_code_parts = []
        
        for pos in sorted(positions):
            wt_aa = wt_sequence[pos]
            
            # Choose a different amino acid
            possible_aas = [aa for aa in AMINO_ACIDS if aa != wt_aa]
            mut_aa = random.choice(possible_aas)
            
            mutant_seq[pos] = mut_aa
            
            # Build mutation code (1-indexed)
            mutant_code_parts.append(f"{wt_aa}{pos+1}{mut_aa}")
        
        mutant_seq_str = "".join(mutant_seq)
        mutant_code = ":".join(mutant_code_parts)
        
        mutants.append((mutant_code, mutant_seq_str))
    
    return mutants

if __name__ == "__main__":
    # Test with a short sequence
    test_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    print("Testing mutant generation:")
    for k in [1, 3, 5]:
        mutants = generate_random_mutants(test_seq, k, n=3)
        print(f"\nk={k} mutations:")
        for code, seq in mutants:
            print(f"  {code}")
            # Verify the number of differences
            diffs = sum(1 for a, b in zip(test_seq, seq) if a != b)
            print(f"    Verified: {diffs} mutations")
