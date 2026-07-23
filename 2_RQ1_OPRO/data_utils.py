from pathlib import Path 

import pandas as pd 
from sklearn.metrics import f1_score

import config as C 

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python")
    if C.TEXT_COL not in df.columns or C.LABEL_COL not in df.columns:
        raise ValueError(
            f"{path} must contain columns '{C.TEXT_COL}' and '{C.LABEL_COL}'. "
            f"Found: {list(df.columns)}"
        )
    df = df[[C.TEXT_COL, C.LABEL_COL]].copy()
    df[C.TEXT_COL] = df[C.TEXT_COL].astype(str)
    df[C.LABEL_COL] = df[C.LABEL_COL].astype(str).str.strip().str.lower()
    df = df.dropna(subset=[C.TEXT_COL]).reset_index(drop=True)
 
    bad_mask = ~df[C.LABEL_COL].isin({C.HATE_LABEL, C.NONHATE_LABEL})
    if bad_mask.any():
        bad_rows = df.loc[bad_mask, [C.LABEL_COL]].head(5)
        examples = "; ".join(
            f"row {idx} -> {val!r}"
            for idx, val in bad_rows[C.LABEL_COL].items()
        )
        raise ValueError(
            f"{path}: {int(bad_mask.sum())} row(s) with a label outside "
            f"{{{C.HATE_LABEL!r}, {C.NONHATE_LABEL!r}}}. First few: {examples}. "
            f"Fix the source file in Data/ (or its normalize step) and re-run."
        )
    return df

def load_split(source: str, which: str) -> pd.DataFrame:
    return _read_csv(C.DATA_DIR / f"{source}_{which}_normalized.csv")

def load_exemplar_pool(source: str, portion: str):
    hate = _read_csv(C.DATA_DIR / f"{source}_exemplars_hate_{portion}_normalized.csv")
    nonhate = _read_csv(C.DATA_DIR / f"{source}_exemplars_nonhate_{portion}_normalized.csv")
    return hate, nonhate 

def clean_scored_frame(df: pd.DataFrame):
    # Returns (clean_df, n_total, n_dropped) | n_dropped = dropping rows where pred is None.
    n_total = len(df)
    clean = df[df["pred"].notna()].copy()
    return clean, n_total, n_total - len(clean)

def macro_f1_raw(clean_df: pd.DataFrame) -> float:
    if len(clean_df) == 0:
        return 0.0
    return f1_score(
        clean_df[C.LABEL_COL], clean_df["pred"],
        labels=[C.HATE_LABEL, C.NONHATE_LABEL], average="macro",
    )

def bin_score(raw_f1: float, w: int = None) -> int: 
    w = C.BIN_WIDTH if w is None else w 
    return (int(raw_f1 * 100) // w) * w 

def pred_class_counts(clean_df: pd.DataFrame) -> dict:
    vc = clean_df["pred"].value_counts().to_dict()
    return {
        C.HATE_LABEL: int(vc.get(C.HATE_LABEL, 0)),
        C.NONHATE_LABEL: int(vc.get(C.NONHATE_LABEL, 0)),
    }

def confusion_counts(clean_df: pd.DataFrame) -> dict:
    label, pred = clean_df[C.LABEL_COL], clean_df["pred"]
    return {
        "tp": int(((label == C.HATE_LABEL)    & (pred == C.HATE_LABEL)).sum()),
        "fn": int(((label == C.HATE_LABEL)    & (pred == C.NONHATE_LABEL)).sum()),
        "fp": int(((label == C.NONHATE_LABEL) & (pred == C.HATE_LABEL)).sum()),
        "tn": int(((label == C.NONHATE_LABEL) & (pred == C.NONHATE_LABEL)).sum()),
        "n_per_class": int((label == C.HATE_LABEL).sum()),
    }

def full_metrics(clean_df: pd.DataFrame) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, matthews_corrcoef,
    )
    y, p = clean_df[C.LABEL_COL], clean_df["pred"]
    labels = [C.HATE_LABEL, C.NONHATE_LABEL]
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y, p, labels=labels, average="macro", zero_division=0
    )
    p_h, r_h, f1_h, _ = precision_recall_fscore_support(
        y, p, labels=labels, average="binary",
        pos_label=C.HATE_LABEL, zero_division=0
    )
    return {
        "accuracy":        accuracy_score(y, p),
        "precision_macro": p_macro,
        "recall_macro":    r_macro,
        "macro_f1":        f1_macro,
        "precision_hate":  p_h,
        "recall_hate":     r_h,
        "f1_hate":         f1_h,
        "mcc":             matthews_corrcoef(y, p),
    }


