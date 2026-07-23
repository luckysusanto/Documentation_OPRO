import numpy as np
import pandas as pd 
from datasets import Dataset 
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
)
from scipy.special import softmax

import config as C

def _read_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', engine='python')
    if C.TEXT_COL not in df.columns or C.LABEL_COL not in df.columns:
        raise ValueError(
            f"{path} must contain columns '{C.TEXT_COL}' and '{C.LABEL_COL}'. "
            f"Found: {list(df.columns)}"
        )
    
    def _to_int_label(v):
        if isinstance(v, (int,)) or (isinstance(v, float) and v == int(v)):
            # if already in integer-ish form
            return int(v)
        key = str(v).strip().lower()
        if key in C.LABEL_MAP:
            return C.LABEL_MAP[key]
        try:
            return int(float(key))
        except ValueError:
            raise ValueError(
                f"{path}: unrecognized label {v!r}. Known string labels: "
                f"{list(C.LABEL_MAP)}. Add it to LABEL_MAP in config.py."
            )
 
    df[C.LABEL_COL] = df[C.LABEL_COL].map(_to_int_label).astype(int)
    df = df[[C.TEXT_COL, C.LABEL_COL]].dropna(subset=[C.TEXT_COL]).reset_index(drop=True)
    return df

def load_train_portion(source: str, portion:str, seed: int) -> pd.DataFrame:
    hate = _read_csv(C.DATA_DIR / f"{source}_exemplars_hate_{portion}_normalized.csv")
    nonhate = _read_csv(C.DATA_DIR / f"{source}_exemplars_nonhate_{portion}_normalized.csv")
    df = pd.concat([hate, nonhate], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

def load_split(source: str, which: str) -> pd.DataFrame:
    # which = ['val_set', 'opt_set', 'test_set'], opt_set is for later (OPRO)
    return _read_csv(C.DATA_DIR / F"{source}_{which}_normalized.csv")

def to_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(df.rename(columns={C.LABEL_COL: "labels"}), preserve_index=False)

    def _tok(batch):
        return tokenizer(
            batch[C.TEXT_COL],
            truncation=True,
            max_length=C.MAX_LENGTH,
        )

    ds = ds.map(_tok, batched=True, remove_columns=[C.TEXT_COL]) # Save space, remove text, just keep tensors
    return ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits)
    preds = logits.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    # Positive class = hate = 1
    p_hate, r_hate, f1_hate, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
 
    out = {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "macro_f1": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_hate": p_hate,
        "recall_hate": r_hate,
        "f1_hate": f1_hate,
        "mcc": matthews_corrcoef(labels, preds),
    }
 
    # ROC-AUC needs probabilities; guard against single-class eval batches.
    try:
        probs = softmax(logits, axis=-1)[:, 1]
        if len(np.unique(labels)) == 2:
            out["roc_auc"] = roc_auc_score(labels, probs)
    except Exception:
        pass
 
    return out

