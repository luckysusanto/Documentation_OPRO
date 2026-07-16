from pathlib import Path
import pandas as pd 

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "Raw"
PROCESSED_DIR = SCRIPT_DIR / "Processed"

PREFIX = {"CSIS": "csis", "IndoDisc": "indoDiscourse"}

csis_df = pd.read_csv(RAW_DIR / "csis.csv", sep=';')
indoDisc_df = pd.read_csv(RAW_DIR / "indoDiscourse.csv", sep=';')

def make_splits(datasets: list, seed: int, exemplar_fractions=None):
    combined = pd.concat(datasets, ignore_index=True).drop_duplicates(subset='id')

    hate    = combined[combined['label'] == 'hate'].sample(frac=1, random_state=seed).reset_index(drop=True)
    nonhate = combined[combined['label'] == 'not-hate'].sample(frac=1, random_state=seed).reset_index(drop=True)

    fixed = [
        ('opt_set',  125, 125),
        ('val_set',  125, 125),
        ('test_set', 250, 250),
    ]
    h_used  = sum(hn  for _, hn, _  in fixed)   # 500
    nh_used = sum(nhn for _, _, nhn in fixed)   # 500

    if len(hate) < h_used or len(nonhate) < nh_used:
        raise ValueError("Not enough data for the fixed splits.")

    # Balanced exemplar size from the remainder
    sample_count = min(len(hate) - h_used, len(nonhate) - nh_used)

    splits = {}
    h_off, nh_off = 0, 0
    for name, hn, nhn in fixed + [
        ('exemplars_hate',    sample_count, 0),
        ('exemplars_nonhate', 0, sample_count),
    ]:
        h_part  = hate.iloc[h_off:h_off+hn]       if hn  > 0 else pd.DataFrame()
        nh_part = nonhate.iloc[nh_off:nh_off+nhn] if nhn > 0 else pd.DataFrame()
        splits[name] = (pd.concat([h_part, nh_part], ignore_index=True)
                          .sample(frac=1, random_state=seed)
                          .reset_index(drop=True))
        h_off  += hn
        nh_off += nhn

    if exemplar_fractions is None:
        exemplar_fractions = [i / 10 for i in range(1, 11)]

    # 10%, 20%, ..., 100%
    for pool in ('exemplars_hate', 'exemplars_nonhate'):
        base = splits[pool]
        for frac in exemplar_fractions:
            n = round(len(base) * frac)
            pct = int(round(frac * 100))
            splits[f'{pool}_{pct:03d}'] = base.iloc[:n].reset_index(drop=True)

    return splits

for data_name, data_df in [("CSIS", csis_df), ("IndoDisc", indoDisc_df)]:
    p = PREFIX[data_name]
    splits = make_splits([data_df], seed=42)

    for name, df in splits.items():
        print(f"Saving: {PROCESSED_DIR}/{p}_{name}.csv")
        df.to_csv(PROCESSED_DIR / f"{p}_{name}.csv", sep=";", index=False)