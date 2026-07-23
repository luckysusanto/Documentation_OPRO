import argparse
import json

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
]

def plot_learning_curve(summary_df, source, model, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = summary_df["portion_pct"]
    ax.errorbar(x, summary_df["test_macro_f1_mean"],
                yerr=summary_df["test_macro_f1_std"],
                marker="o", capsize=3, color="#26a", label="macro-F1")
    ax.errorbar(x, summary_df["test_accuracy_mean"],
                yerr=summary_df["test_accuracy_std"],
                marker="s", capsize=3, color="#2a6", label="accuracy")
    ax.set_xlabel("exemplar portion (%)")
    ax.set_ylabel("score")
    ax.set_title(f"{source} / {model}: OPRO performance vs data portion")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def aggregate_source(source, model):
    src_dir = C.OUTPUT_DIR / source
    rows, missing = [], []

    for portion in C.PORTIONS:
        for seed in C.SEEDS:
            cell = f"{source}_{portion}_{seed}_{model}"
            mpath = src_dir / cell / "test_metrics.json"
            if not mpath.exists():
                missing.append(cell)
                continue
            try:
                m = json.loads(mpath.read_text())
            except (json.JSONDecodeError, ValueError):
                missing.append(cell + " (unreadable)")
                continue
            row = {"source": source, "model": model, "portion": portion,
                   "portion_pct": int(portion), "seed": seed,
                   "test_n_dropped": m.get("test_n_dropped")}
            row.update({k: m.get(k) for k in SUMMARY_METRICS})
            rows.append(row)

    if not rows:
        print(f"[{source}/{model}] no completed cells under {src_dir}. Skipping.")
        return
    if missing:
        print(f"[{source}/{model}] WARNING: {len(missing)} cell(s) missing/"
              f"unreadable: {missing}")
        print(f"[{source}/{model}] aggregating the {len(rows)} available cell(s).")

    raw = (pd.DataFrame(rows)
           .sort_values(["portion_pct", "seed"]).reset_index(drop=True))
    raw.to_csv(src_dir / f"raw_results_{model}.csv", index=False)

    summary_rows = []
    for (portion, pct), g in raw.groupby(["portion", "portion_pct"]):
        rec = {"portion": portion, "portion_pct": pct, "n_seeds": len(g)}
        for m in SUMMARY_METRICS:
            rec[f"{m}_mean"] = g[m].mean()
            rec[f"{m}_std"] = g[m].std(ddof=0)
        summary_rows.append(rec)

    summary = (pd.DataFrame(summary_rows)
               .sort_values("portion_pct").reset_index(drop=True))
    summary.to_csv(src_dir / f"summary_{model}.csv", index=False)
    plot_learning_curve(summary, source, model,
                        src_dir / f"learning_curve_{model}.png")

    print(f"[{source}/{model}] wrote raw_results_{model}.csv, summary_{model}.csv, "
          f"learning_curve_{model}.png ({len(rows)} cells across "
          f"{summary['portion_pct'].nunique()} portions)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=C.SOURCES)
    ap.add_argument("--model", default="qwen3-32b", choices=list(C.MODEL_REPO),
                    help="Which model's cells to aggregate.")
    args = ap.parse_args()
    for source in args.sources:
        aggregate_source(source, args.model)

if __name__ == "__main__":
    main()
