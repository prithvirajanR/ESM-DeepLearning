import random
from src.handle_mutations import parse_mutation_string



# Generate n random mutants of wt sequence with k mutations.
def generate_random_mutants(wt_seq, k, n=100):

    aminoacids = "ACDEFGHIKLMNPQRSTVWY"

    random.seed(42)

    
    L = len(wt_seq)
    mutants = []

    for i in range(n):
        seq = list(wt_seq)
        

        positions = random.sample(range(L), k)
        mutation_codes = []
    
        for pos in positions:
    
            wt = seq[pos]
            mut = random.choice([AA for AA in aminoacids if AA != wt])
            seq[pos] = mut
    
            mutation_codes.append(f"{wt}{pos+1}{mut}")
    
        mutation_sequence = "".join(seq)
        mutation_code = ":".join(mutation_codes)
    
        mutants.append((mutation_code, mutation_sequence))

    return mutants





def apply_mutations(wt_seq, mutation_string):

    seq = list(wt_seq)

    mutations = parse_mutation_string(mutation_string)


    for wt, pos, mut in mutations:

        idx = pos - 1

        if seq[idx] != wt:
            raise ValueError(
                f"Mismatch in WT residue: expected {wt} at position {pos}, "
                f"but WT sequence has {seq[idx]}"
            )

        seq[idx] = mut

    return "".join(seq)



        