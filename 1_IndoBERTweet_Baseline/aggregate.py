"""
Usage:
    python aggregate.py (uses config's source if not set)
    python aggregate.py --sources csis

Output:
    runs/<source>/raw_results.csv 
"""
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

SUMMARY_METRICS = [
    "test_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "test_macro_f1",
    "test_f1_hate",
    "test_mcc",
    "test_roc_auc",
]


def plot_learning_curve(summary_df, source, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = summary_df["portion_pct"]
    ax.errorbar(x, summary_df["test_macro_f1_mean"],
                yerr=summary_df["test_macro_f1_std"],
                marker="o", capsize=3, color="#26a", label="macro-F1")
    ax.errorbar(x, summary_df["test_accuracy_mean"],
                yerr=summary_df["test_accuracy_std"],
                marker="s", capsize=3, color="#2a6", label="accuracy")
    ax.set_xlabel("training data portion (%)")
    ax.set_ylabel("score")
    ax.set_title(f"{source}: performance vs data portion")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def aggregate_source(source):
    src_dir = C.OUTPUT_DIR / source
    rows = []
    missing = []

    for portion in C.PORTIONS:
        for seed in C.SEEDS:
            cell = f"{source}_{portion}_{seed}"
            mpath = src_dir / cell / "test_metrics.json"
            if not mpath.exists():
                missing.append(cell)
                continue
            try:
                m = json.loads(mpath.read_text())
            except (json.JSONDecodeError, ValueError):
                missing.append(cell + " (unreadable)")
                continue
            row = {"source": source, "portion": portion,
                   "portion_pct": int(portion), "seed": seed,
                   "n_train": m.get("_n_train")}
            row.update({k: m.get(k) for k in SUMMARY_METRICS})
            rows.append(row)

    if not rows:
        print(f"[{source}] no completed cells found under {src_dir}. Skipping.")
        return
    if missing:
        print(f"[{source}] WARNING: {len(missing)} cell(s) missing/unreadable: "
              f"{missing}")
        print(f"[{source}] aggregating the {len(rows)} available cell(s) anyway.")

    raw = pd.DataFrame(rows).sort_values(["portion_pct", "seed"]).reset_index(drop=True)
    raw.to_csv(src_dir / "raw_results.csv", index=False)

    summary_rows = []
    for (portion, pct), g in raw.groupby(["portion", "portion_pct"]):
        n_train_vals = g["n_train"].dropna()
        n_train = int(n_train_vals.iloc[0]) if len(n_train_vals) else None
        rec = {"portion": portion, "portion_pct": pct, "n_train": n_train,
               "n_seeds": len(g)}
        for m in SUMMARY_METRICS:
            rec[f"{m}_mean"] = g[m].mean()
            rec[f"{m}_std"] = g[m].std(ddof=0)
        summary_rows.append(rec)

    summary = pd.DataFrame(summary_rows).sort_values("portion_pct").reset_index(drop=True)
    summary.to_csv(src_dir / "summary.csv", index=False)
    plot_learning_curve(summary, source, src_dir / "learning_curve.png")

    print(f"[{source}] wrote raw_results.csv, summary.csv, learning_curve.png "
          f"({len(rows)} cells across {summary['portion_pct'].nunique()} portions)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=C.SOURCES)
    args = ap.parse_args()
    for source in args.sources:
        aggregate_source(source)


if __name__ == "__main__":
    main()