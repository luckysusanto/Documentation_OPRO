import argparse 
import json
from datetime import datetime 
from pathlib import Path 

import config as C 
import data_utils as D 
import opro_core as K 
from optimize import optimize, _sort_tracker 

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", required=True, choices=list(C.MODEL_REPO),
        help="Model acting as both optimizer and scorer"
    )
    ap.add_argument(
        "--source", required=True, choices=C.SOURCES
    )
    ap.add_argument(
        "--portion", required=True, help="3-digit string, starting from 010, increment of 10, max of 100"
    )
    ap.add_argument(
        "--seed", required=True, type=int
    )
    ap.add_argument(
        "--server-model-name", default=None,
        help="Name the vLLM server registered. Defaults to the HF repo id for --model."
    )
    ap.add_argument(
        "--base-url", default=C.BASE_URL
    )
    ap.add_argument(
        "--max-workers", type=int, default=C.MAX_WORKERS
    )
    ap.add_argument(
        "--step-count", type=int, default=None,
        help="Override config.STEP_COUNT, change for testing purposes" 
    )
    ap.add_argument(
        "--init-rubric", default=None, type=Path,
        help="Path to a rubric text file to seed step 0 with. Defaults to "
             "opro_core.INITIAL_RUBRIC (translated Waseem-Hovy), which is the "
             "RQ1 setting. RQ2 (folder 4) passes a rubric optimized on the "
             "OTHER source to test whether an old hate-speech definition "
             "adapts to a newer one."
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-run even if the test_metrics.json already exists."
    )
    return ap.parse_args()

def select_on_val(tracker, source, client, seed, max_workers, log):
    val_set = D.load_split(source, C.SELECTION_SPLIT)

    seen, ranked = set(), []
    for e in _sort_tracker(tracker):
        if e["rubric"] not in seen:
            seen.add(e["rubric"])
            ranked.append(e)
    shortlist = ranked[:C.KEEP_TOP_K_CANDIDATE]

    results = []
    for i, e in enumerate(shortlist):
        scored = K.score_dataframe(val_set, client, e["rubric"], seed, max_workers)
        clean, n_total, n_dropped = D.clean_scored_frame(scored)
        val_f1 = D.macro_f1_raw(clean)
        results.append({
            "rank_on_opt": i,
            "opt_score": e["score"],
            "val_macro_f1": val_f1,
            "n_total": n_total,
            "n_dropped": n_dropped,
            "rubric": e["rubric"],
        })
        log(f"select: cand {i} opt={e['score']:.4f} val={val_f1:.4f} "
            f"dropped={n_dropped}")
    
    best = max(range(len(results)), key=lambda x: results[x]["val_macro_f1"])
    for j, r in enumerate(results):
        r["selected"] = (j == best)
    return results[best]["rubric"], results 

def report_on_test(rubric, source, client, seed, max_workers, log):
    test_set = D.load_split(source, C.REPORT_SPLIT)
    scored = K.score_dataframe(test_set, client, rubric, seed, max_workers)
    clean, n_total, n_dropped = D.clean_scored_frame(scored)
    metrics = D.full_metrics(clean)
    metrics = {f"test_{k}": v for k, v in metrics.items()}
    metrics["test_n_total"] = n_total
    metrics["test_n_dropped"] = n_dropped
    log(f"test: macro_f1={metrics['test_macro_f1']:.4f} "
        f"n={n_total} dropped={n_dropped}")
    return metrics, scored

def main():
    args = parse_args()
    if args.step_count is not None:
        C.STEP_COUNT = args.step_count

    init_rubric = None
    if args.init_rubric is not None:
        if not args.init_rubric.is_file():
            raise SystemExit(f"--init-rubric file not found: {args.init_rubric}")
        init_rubric = args.init_rubric.read_text(encoding="utf-8").strip()
        if not init_rubric:
            raise SystemExit(f"--init-rubric file is empty: {args.init_rubric}")

    served = args.server_model_name or C.MODEL_REPO[args.model]
    run_dir = (C.OUTPUT_DIR / args.source /
               f"{args.source}_{args.portion}_{args.seed}_{args.model}")
    metrics_path = run_dir / "test_metrics.json"

    if metrics_path.exists() and not args.force:
        print(f"[skip] {metrics_path} exists; nothing to do (use --force to rerun).")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    log_file = open(log_path, "a", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    try:
        log(f"=== CELL START model={args.model} source={args.source} "
            f"portion={args.portion} seed={args.seed} steps={C.STEP_COUNT} ==="
            f"init_rubric={args.init_rubric or 'DEFAULT'} ===")

        client = K.VLLMClient(base_url=args.base_url, served_model_name=served)

        opt_set = D.load_split(args.source, "opt_set")
        exemplar_hate, exemplar_nonhate = D.load_exemplar_pool(
            args.source, args.portion)
        log(f"loaded opt_set={len(opt_set)} "
            f"exemplars(hate={len(exemplar_hate)}, nonhate={len(exemplar_nonhate)})")

        # Optimize on opt set
        tracker = optimize(
            opt_set=opt_set,
            exemplar_hate=exemplar_hate, exemplar_nonhate=exemplar_nonhate,
            client=client, seed=args.seed, max_workers=args.max_workers,
            log=log, traj_path=run_dir / "trajectory.jsonl",
            init_rubric=init_rubric,
        )

        # Select based on val set
        best_rubric, selection = select_on_val(
            tracker, args.source, client, args.seed, args.max_workers, log)
        (run_dir / "best_rubric.txt").write_text(best_rubric)
        (run_dir / "selection.json").write_text(json.dumps(selection, indent=2))

        # Report on test set
        test_metrics, test_scored = report_on_test(
            best_rubric, args.source, client, args.seed, args.max_workers, log)
        test_scored.to_csv(run_dir / "test_scored.csv", index=False)

        test_metrics["_model"] = args.model
        test_metrics["_source"] = args.source
        test_metrics["_portion"] = args.portion
        test_metrics["_seed"] = args.seed
        test_metrics["_extraction_stats"] = dict(K.EXTRACTION_STATS)

        test_metrics["_init_rubric"] = (
            str(args.init_rubric) if args.init_rubric else None)

        # Write log as text
        metrics_path.write_text(json.dumps(test_metrics, indent=2))
        log(f"[done] wrote {metrics_path}")
        log(f"=== EXTRACTION/ROBUSTNESS STATS: {dict(K.EXTRACTION_STATS)} ===")
    except Exception as e:
        log(f"!!! CELL FAILED: {type(e).__name__}: {e}")
        log(f"extraction stats at failure: {dict(K.EXTRACTION_STATS)}")
        raise
    finally:
        log_file.close()

if __name__ == "__main__":
    main()