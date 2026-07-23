import argparse
import json

import config as C
from train_core import load_tokenizer, run_hp_search, checkpoint_path

def search_one(source: str, tokenizer, n_trials=None, portion=None):
    transfer_source = C.transfer_source_for(source)

    if n_trials is not None:
        C.HP_N_TRIALS = n_trials
    if portion is not None:
        C.HP_SEARCH_PORTION = portion

    print(f"[{source}] continued fine-tuning HP search")
    print(f"  init checkpoint : {transfer_source} portion-{C.BASE_PORTION} seed {C.HP_SEARCH_SEED}")
    print(f"  anchored at     : portion {C.HP_SEARCH_PORTION}")
    print(f"  trials          : {C.HP_N_TRIALS}")
    print(f"  LR range        : {C.HP_SPACE['learning_rate']} "
          f"(folder 1 searched {C.BASE.HP_SPACE['learning_rate']})")
    print(f"  checkpoint path : {checkpoint_path(transfer_source, C.HP_SEARCH_SEED)}")

    src_dir = C.OUTPUT_DIR / source
    src_dir.mkdir(parents=True, exist_ok=True)

    hp = run_hp_search(source, tokenizer, src_dir / "_hp_search")

    hp_path = src_dir / "best_hp.json"
    hp_path.write_text(json.dumps(hp, indent=2))
    print(f"\n[{source}] best hp -> {hp_path}")
    print(json.dumps(hp, indent=2))
    return hp

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", default=C.SOURCES,
                    help="TARGET source(s) to search for. Each one's initializing "
                         "checkpoint is the OTHER source's portion-100 model.")
    ap.add_argument("--n-trials", type=int, default=None,
                    help="Override config.HP_N_TRIALS (useful for a quick probe).")
    ap.add_argument("--portion", type=str, default=None,
                    help="Override config.HP_SEARCH_PORTION (default 010).")
    ap.add_argument("--force", action="store_true",
                    help="Re-search even if best_hp.json already exists.")
    return ap.parse_args()

def main():
    args = parse_args()
    tokenizer = load_tokenizer()

    for source in args.sources:
        if source not in C.SOURCES:
            raise SystemExit(f"unknown source '{source}'. Known: {C.SOURCES}")

        hp_path = C.OUTPUT_DIR / source / "best_hp.json"
        if hp_path.exists() and not args.force:
            print(f"[skip] {hp_path} exists. Use --force to re-search.")
            continue

        search_one(source, tokenizer, args.n_trials, args.portion)

if __name__ == "__main__":
    main()