import pandas as pd

def load_dms(path):
    
    df_raw = pd.read_csv(path)

    df = df_raw.rename(columns= {
        "DMS_score": "fitness",
        "DMS_score_bin": "fitness_bin"
    })

    keep = ["mutant", "fitness", "fitness_bin"]

    return df[keep].copy()


def get_wt_sequence(path, dms_filename):

    ref = pd.read_csv(path)

    row = ref[ref["DMS_filename"] == dms_filename]

    return row.iloc[0]["target_seq"]


