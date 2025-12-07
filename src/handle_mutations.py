def parse_mutation_string(m_str):
   
    muts = m_str.split(":")
    parsed = []

    for m in muts:
        wt = m[0]            # first letter
        mut = m[-1]          # last letter
        pos = int(m[1:-1])   
        parsed.append((wt, pos, mut))

    return parsed


def get_single_mutants(df):
    """
    Returns only rows where 'mutant' contains exactly 1 mutation.
    """
    singles = []

    for _, row in df.iterrows():
        muts = row["mutant"].split(":")
        if len(muts) == 1:
            singles.append(row)

    return pd.DataFrame(singles)


def get_double_mutants(df):
    """
    Returns only rows where 'mutant' contains exactly 2 mutations.
    """
    doubles = []

    for _, row in df.iterrows():
        muts = row["mutant"].split(":")
        if len(muts) == 2:
            doubles.append(row)

    return pd.DataFrame(doubles)
