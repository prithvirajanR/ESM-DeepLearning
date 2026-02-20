import pandas as pd
import numpy as np
import random
import argparse

# A4_HUMAN Sequence (from ProteinGym_reference_file_substitutions.csv)
WT_SEQ = "MLPGLALLLLAAWTARALEVPTDGNAGLLAEPQIAMFCGRLNMHMNVQNGKWDSDPSGTKTCIDTKEGILQYCQEVYPELQITNVVEANQPVTIQNWCKRGRKQCKTHPHFVIPYRCLVGEFVSDALLVPDKCKFLHQERMDVCETHLHWHTVAKETCSEKSTNLHDYGMLLPCGIDKFRGVEFVCCPLAEESDNVDSADAEEDDSDVWWGGADTDYADGSEDKVVEVAEEEEVAEVEEEEADDDEDDEDGDEVEEEAEEPYEEATERTTSIATTTTTTTESVEEVVREVCSEQAETGPCRAMISRWYFDVTEGKCAPFFYGGCGGNRNNFDTEEYCMAVCGSAMSQSLLKTTQEPLARDPVKLPTTAASTPDAVDKYLETPGDENEHAHFQKAKERLEAKHRERMSQVMREWEEAERQAKNLPKADKKAVIQHFQEKVESLEQEAANERQQLVETHMARVEAMLNDRRRLALENYITALQAVPPRPRHVFNMLKKYVRAEQKDRQHTLKHFEHVRMVDPKKAAQIRSQVMTHLRVIYERMNQSLSLLYNVPAVAEEIQDEVDELLQKEQNYSDDVLANMISEPRISYGNDALMPSLTETKTTVELLPVNGEFSLDDLQPWHSFGADSVPANTENEVEPVDARPAADRGLTTRPGSGLTNIKTEEISEVKMDAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIATVIVITLVMLKKKQYTSIHHGVVEVDAAVTPEERHLSKMQQNGYENPTYKFFEQMQN"

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def generate_mutant(wt_seq, k):
    """
    Generates a random mutant with k substitutions.
    """
    L = len(wt_seq)
    positions = random.sample(range(L), k)
    positions.sort() # Sort for consistent naming
    
    mutations = []
    mut_seq_list = list(wt_seq)
    
    for pos in positions:
        wt_aa = wt_seq[pos]
        # Pick random AA that is NOT wt_aa
        possible_aas = [aa for aa in AMINO_ACIDS if aa != wt_aa]
        mut_aa = random.choice(possible_aas)
        
        mut_seq_list[pos] = mut_aa
        mutations.append(f"{wt_aa}{pos+1}{mut_aa}")
        
    mutant_code = ":".join(mutations)
    mutant_seq = "".join(mut_seq_list)
    
    return mutant_code, mutant_seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", type=str, default="2,5,10,20", help="Comma-separated list of mutation counts (k)")
    parser.add_argument("--count_per_k", type=int, default=1000, help="Number of mutants to generate per k")
    parser.add_argument("--output", default="data/Synthetic_Landscape.csv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set seed for reproducibility (critical: ensures data/results stay in sync across reruns)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    ks = [int(k) for k in args.distances.split(',')]
    data = []
    
    print(f"Generating synthetic landscape for A4_HUMAN (Len {len(WT_SEQ)})...")
    
    # 1. Add Wildtype for baseline
    data.append({
        "mutant": "WT",
        "mutated_sequence": WT_SEQ,
        "n_mutations": 0
    })
    
    for k in ks:
        print(f"Generating {args.count_per_k} mutants at distance k={k}...")
        for _ in range(args.count_per_k):
            code, seq = generate_mutant(WT_SEQ, k)
            data.append({
                "mutant": code,
                "mutated_sequence": seq,
                "n_mutations": k
            })
            
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} mutants to {args.output}")

if __name__ == "__main__":
    main()
