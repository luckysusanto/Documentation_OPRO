import numpy as np
import pandas as pd
from transformers import AutoTokenizer

PATHS = [
    "./Normalized/csis_exemplars_hate_100_normalized.csv",
    "./Normalized/csis_exemplars_nonhate_100_normalized.csv",
    "./Normalized/csis_test_set_normalized.csv",
    "./Normalized/csis_val_set_normalized.csv",
    "./Normalized/indoDiscourse_exemplars_hate_100_normalized.csv",
    "./Normalized/indoDiscourse_exemplars_nonhate_100_normalized.csv",
    "./Normalized/indoDiscourse_test_set_normalized.csv",
    "./Normalized/indoDiscourse_val_set_normalized.csv"

]

TEXT_COL = "text"
MODEL = "indolem/indobertweet-base-uncased"
MAX_SEQ_LENGTH = 512
LOCAL_FILES_ONLY = True

def analyze(path, tokenizer):
    df = pd.read_csv(path, sep=';', engine="python")
    if TEXT_COL not in df.columns:
        print(f"\n{path}")
        print(f"  SKIPPED — column '{TEXT_COL}' not found. Columns: {list(df.columns)}")
        return

    texts = df[TEXT_COL].dropna().astype(str).tolist()
    if not texts:
        print(f"\n{path}")
        print("  SKIPPED — no non-empty text rows.")
        return

    lengths = np.array([
        len(tokenizer(t, add_special_tokens=True, truncation=False)["input_ids"])
        for t in texts
    ])

    n = len(lengths)
    under = int((lengths <= MAX_SEQ_LENGTH).sum())
    over = n - under

    print(f"\n{path}")
    print(f"Number of data under {MAX_SEQ_LENGTH} token length = {under} "
          f"({100 * under / n:.1f}% of dataset)")
    print(f"Number of data over  {MAX_SEQ_LENGTH} token length = {over} "
          f"({100 * over / n:.1f}% of dataset)")
    print(f"  rows: {n} | mean: {lengths.mean():.1f} | "
          f"p95: {np.percentile(lengths, 95):.0f} | "
          f"p99: {np.percentile(lengths, 99):.0f} | max: {lengths.max()}")

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=LOCAL_FILES_ONLY
    )
    for path in PATHS:
        try:
            analyze(path, tokenizer)
        except FileNotFoundError:
            print(f"\n{path}")
            print("  SKIPPED — file not found.")


if __name__ == "__main__":
    main()