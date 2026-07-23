import json
import time 
from datetime import datetime 
from pathlib import Path 

from tqdm import tqdm 

import config as C 
import data_utils as D 
import opro_core as K 

def _sort_tracker(tracker):
    return sorted(
        tracker,
        key= lambda x: (x["score"] if x["score"] is not None else -1),
        reverse=True
    )

def _tracker_to_dict_best_to_worst(tracker, k):
    best_first = _sort_tracker(tracker)[:k]
    d = {
        i + 1: {
            "rubric": e["rubric"], 
            "score": e["reported_score"],
            "tp": e["tp"], "tn": e["tn"], "fp": e["fp"], "fn": e["fn"],
            "n_per_class": e["n_per_class"]
        } for i, e in enumerate(best_first)
    }

    return d, len(best_first)

def optimize(opt_set, exemplar_hate, exemplar_nonhate, client, seed,
             max_workers, log, traj_path: Path):
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_file = open(traj_path, "a", encoding="utf-8")
    tracker = []

    try:
        for step in tqdm(range(C.STEP_COUNT + 1), desc="OPRO"):
            step_start = time.time()
            meta_prompt_used = None
            sampled_seed_steps = []

            # During startup, just evaluate the initial rubric
            if step == 0:
                candidates = [K.INITIAL_RUBRIC]
                log(f"step {step}: init (INITIAL_RUBRIC)")
            else:
                n_use = min(len(tracker), C.KEEP_TOP_K_CANDIDATE)
                tried_dict, n_rubrics = _tracker_to_dict_best_to_worst(tracker, n_use)
                log(f"step {step}: optimizing (top {n_rubrics}/{len(tracker)})")
                candidates = []
                for c in range(C.CANDIDATE_PER_STEP):
                    seed_step = step * C.CANDIDATE_PER_STEP + c
                    sampled_seed_steps.append(seed_step)
                    meta_prompt = K.optimizer_prompter(
                        rubrics_dict=tried_dict,
                        exemplar_hate=exemplar_hate,
                        exemplar_nonhate=exemplar_nonhate,
                        n_rubrics=n_rubrics,
                        n_exemplars=C.HATE_EXEMPLAR_PER_STEP,
                        seed=seed, seed_step=seed_step,
                        criteria_cap=C.CRITERIA_CAP,
                    )
                    if meta_prompt_used is None:
                        meta_prompt_used = meta_prompt
                    new_rubric = client.optimizer_rubric_retried(
                        meta_prompt, seed=seed + seed_step,
                        temperature=C.TEMPERATURE)
                    if new_rubric is None:
                        log(f"step {step}: cand {c} optimizer failed after "
                            f"{C.RETRY_MAX} retries -- skipping candidate")
                        continue
                    candidates.append(new_rubric)

                if not candidates:
                    raise RuntimeError(
                        f"step {step}: all {C.CANDIDATE_PER_STEP} optimizer "
                        f"candidates failed after {C.RETRY_MAX} retries each "
                        f"-- aborting run.")
                log(f"step {step}: generated {len(candidates)} candidate(s)")

            # Score candidate on opt set
            step_results = []
            for i, rubric in enumerate(candidates):
                scored = K.score_dataframe(opt_set, client, rubric, seed, max_workers)
                clean, n_total, n_dropped = D.clean_scored_frame(scored)
                if n_dropped:
                    log(f"step {step}: cand {i} DROPPED {n_dropped}/{n_total} "
                        f"rows -- F1 on {len(clean)} rows")
                raw = D.macro_f1_raw(clean)
                reported = D.bin_score(raw)
                conf = D.confusion_counts(clean)
                counts = D.pred_class_counts(clean)
                step_results.append({
                    "rubric": rubric, "raw": raw, "reported": reported,
                    "conf": conf, "counts": counts,
                    "n_total": n_total, "n_dropped": n_dropped,
                })
                log(f"step {step}: cand {i} raw={raw:.4f} reported={reported} "
                    f"pred(h={counts[C.HATE_LABEL]},nh={counts[C.NONHATE_LABEL]}) "
                    f"TP={conf['tp']} FN={conf['fn']} FP={conf['fp']} TN={conf['tn']} "
                    f"dropped={n_dropped}")

            best_idx = max(range(len(step_results)),
                           key=lambda j: step_results[j]["raw"])
            best = step_results[best_idx]

            # Append to tracker-
            for r in step_results:
                tracker.append({
                    "rubric": r["rubric"], "score": r["raw"],
                    "reported_score": r["reported"],
                    "tp": r["conf"]["tp"], "fn": r["conf"]["fn"],
                    "fp": r["conf"]["fp"], "tn": r["conf"]["tn"],
                    "n_per_class": r["conf"]["n_per_class"],
                })
            tracker = _sort_tracker(tracker)

            elapsed = time.time() - step_start
            traj_file.write(json.dumps({
                "step": step, "timestamp": datetime.now().isoformat(),
                "elapsed_sec": round(elapsed, 2),
                "phase": "init" if step == 0 else "optimize",
                "sampled_seed_steps": sampled_seed_steps,
                "candidates": [{
                    "candidate_index": i, "won": (i == best_idx),
                    "raw_score": r["raw"], "reported_score": r["reported"],
                    "pred_counts": r["counts"], "confusion": r["conf"],
                    "rubric": r["rubric"],
                    "n_total": r["n_total"], "n_dropped": r["n_dropped"],
                } for i, r in enumerate(step_results)],
                "step_best": {"raw_score": best["raw"],
                              "reported_score": best["reported"]},
                "overall_best": {"raw_score": tracker[0]["score"],
                                 "reported_score": tracker[0]["reported_score"]},
                "meta_prompt_sample": meta_prompt_used,
                "extraction_stats_so_far": dict(K.EXTRACTION_STATS),
            }) + "\n")
            traj_file.flush()

            log(f"step {step}: DONE best_raw={best['raw']:.4f} "
                f"overall_best={tracker[0]['score']:.4f} {elapsed:.1f}s")
    finally:
        traj_file.close()

    return tracker
