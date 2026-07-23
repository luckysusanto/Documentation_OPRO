# python run_single.py --source indoDiscourse --portion 010 --seed 42
import argparse
import json

import config as C
from train_core import load_tokenizer, train_once, checkpoint_path

def load_hp(source: str) -> dict:
    hp_path = C.OUTPUT_DIR / source / "best_hp.json"
    if not hp_path.exists():
        raise FileNotFoundError(
            f"{hp_path} not found. Run the continued-fine-tuning HP search for "
            f"'{source}' first:  python hyperparameter_searching.py --sources {source}\n"
        )
    return json.loads(hp_path.read_text())

def plot_training_curve(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tr = [(h["epoch"], h["loss"]) for h in history if "loss" in h and "epoch" in h]
        ev = [(h["epoch"], h[C.EVAL_METRIC]) for h in history if C.EVAL_METRIC in h]
        if not tr and not ev:
            return

        fig, ax1 = plt.subplots(figsize=(7, 4))
        if tr:
            ax1.plot(*zip(*tr), marker="o", label="train loss")
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("loss")
        if ev:
            ax2 = ax1.twinx()
            ax2.plot(*zip(*ev), marker="s", color="tab:orange", label=C.EVAL_METRIC)
            ax2.set_ylabel(C.EVAL_METRIC)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not plot training curve: {e}")

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=C.SOURCES,
                    help="TARGET source to fine-tune onto. The checkpoint comes "
                         "from the other source automatically.")
    ap.add_argument("--portion", required=True, choices=C.PORTIONS,
                    help="Portion of the TARGET source's training data.")
    ap.add_argument("--seed", required=True, type=int, choices=C.SEEDS)
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if test_metrics.json already exists.")
    return ap.parse_args()

def main():
    args = parse_args()

    run_dir = C.OUTPUT_DIR / args.source / f"{args.source}_{args.portion}_{args.seed}"
    metrics_path = run_dir / "test_metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"[skip] {metrics_path} exists. Use --force to re-run.")
        return

    transfer_source = C.transfer_source_for(args.source)
    ckpt = checkpoint_path(transfer_source, args.seed)
    print(f"[info] target={args.source} portion={args.portion} seed={args.seed}")
    print(f"[info] initializing from {transfer_source} portion-{C.BASE_PORTION}: {ckpt}")

    hp = load_hp(args.source)
    print(f"[info] hyperparameters: {hp}")

    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer()

    test_metrics, history, n_train = train_once(
        hp, args.source, args.portion, args.seed, run_dir, tokenizer
    )

    plot_training_curve(history, run_dir / "training_curve.png")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "hyperparameters.json").write_text(json.dumps(hp, indent=2))
    metrics_path.write_text(json.dumps(test_metrics, indent=2, default=str))

    key = C.EVAL_METRIC.replace("eval_", "")
    print(f"[done] n_train={n_train} "
          f"target {key}={test_metrics.get('test_' + key)} "
          f"forgetting={test_metrics.get('_forgetting')}")

if __name__ == "__main__":
    main()