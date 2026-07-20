import argparse
import json

import config as C
from train_core import load_tokenizer, run_hp_search

def search_one(source: str, tokenizer, out_root, n_trials=None, portion=None):
    # Allow per-invocation overrides without editing config.py.
    if n_trials is not None:
        C.HP_N_TRIALS = n_trials
    if portion is not None:
        C.HP_SEARCH_PORTION = portion

    print(f"\n{'='*60}")
    print(f"HP SEARCH | source={source} | portion={C.HP_SEARCH_PORTION} "
          f"| trials={C.HP_N_TRIALS} | seed={C.HP_SEARCH_SEED}")
    print(f"{'='*60}")

    src_dir = out_root / source
    src_dir.mkdir(parents=True, exist_ok=True)

    hp = run_hp_search(source, tokenizer, src_dir)

    hp_path = src_dir / "best_hp.json"
    hp_path.write_text(json.dumps(hp, indent=2))
    print(f"\n[{source}] best hp -> {hp_path}")
    print(json.dumps(hp, indent=2))
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=C.SOURCES,
                    help="Which sources to search (default: both).")
    ap.add_argument("--n-trials", type=int, default=None,
                    help="Override config.HP_N_TRIALS.")
    ap.add_argument("--portion", type=str, default=None,
                    help="Override config.HP_SEARCH_PORTION, e.g. 100.")
    args = ap.parse_args()

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer()

    results = {}
    for source in args.sources:
        results[source] = search_one(
            source, tokenizer, C.OUTPUT_DIR,
            n_trials=args.n_trials, portion=args.portion,
        )

    print(f"\n{'='*60}\nDONE. Best configs:")
    for source, hp in results.items():
        print(f"  {source}: {hp}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()