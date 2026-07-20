import argparse
import json 
import sys 
from pathlib import Path 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C 
from train_core import load_tokenizer, train_once 

def plot_training_curve(history, path):
    """eval macro-F1 + train/eval loss per epoch."""
    epochs_f1, f1s = [], []
    tr_ep, tr_loss = [], []
    ev_ep, ev_loss = [], []
    for rec in history:
        if "eval_macro_f1" in rec:
            epochs_f1.append(rec.get("epoch")); f1s.append(rec["eval_macro_f1"])
        if "loss" in rec:
            tr_ep.append(rec.get("epoch")); tr_loss.append(rec["loss"])
        if "eval_loss" in rec:
            ev_ep.append(rec.get("epoch")); ev_loss.append(rec["eval_loss"])
 
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(tr_ep, tr_loss, "o-", color="#888", label="train loss")
    ax1.plot(ev_ep, ev_loss, "s-", color="#c44", label="eval loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs_f1, f1s, "^-", color="#26a", label="eval macro-F1")
    ax2.set_ylabel("macro-F1")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

def load_hp(source):
    hp_path = C.OUTPUT_DIR / source / "best_hp.json"
    if not hp_path.exists():
        sys.exit(
            f"ERROR: {hp_path} not found. Run the HP search for `{source}` first"
        )
    return json.loads(hp_path.read_text())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--portion", required=True, help="e.g. 010, 020, ... 100")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if test_metrics.json already exists.")
    args = ap.parse_args()
 
    run_dir = C.OUTPUT_DIR / args.source / f"{args.source}_{args.portion}_{args.seed}"
    metrics_path = run_dir / "test_metrics.json"
 
    if metrics_path.exists() and not args.force:
        print(f"[skip] {metrics_path} exists; nothing to do (use --force to redo).")
        return
 
    hp = load_hp(args.source)
    tokenizer = load_tokenizer()
    run_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"=== {args.source} | portion {args.portion} | seed {args.seed} ===")
    test_metrics, history, n_train = train_once(
        hp, args.source, args.portion, args.seed, run_dir, tokenizer
    )
    test_metrics["_n_train"] = n_train
 
    plot_training_curve(history, run_dir / "training_curve.png")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    # Write metrics LAST: its presence is the "cell complete" signal, so it must
    # only appear after everything else succeeded.
    metrics_path.write_text(json.dumps(test_metrics, indent=2))
 
    print(f"[done] wrote {metrics_path}")
 
if __name__ == "__main__":
    main()
