import argparse
import os 
import subprocess
import sys 

import config as C 

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(C.MODEL_REPO),
                    help="Model acting as optimizer and scorer. Also selects "
                         "WHICH transferred rubric is used -- rubrics are "
                         "model-specific.")
    ap.add_argument("--source", required=True, choices=C.SOURCES,
                    help="TARGET source to optimize on. The transferred rubric "
                         "comes from the other source automatically.")
    ap.add_argument("--portion", required=True, choices=C.PORTIONS,
                    help="Portion of the TARGET source's exemplar reservoir.")
    ap.add_argument("--seed", required=True, type=int, choices=C.SEEDS)
    ap.add_argument("--server-model-name", default=None)
    ap.add_argument("--base-url", default=C.BASE_URL)
    ap.add_argument("--max-workers", type=int, default=C.MAX_WORKERS)
    ap.add_argument("--step-count", type=int, default=None,
                    help="Override STEP_COUNT. Testing only -- a value "
                         "different from folder 2's breaks the comparison.")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()

    transfer_source = C.transfer_source_for(args.source)
    rubric = C.rubric_path(transfer_source, args.seed, args.model)

    if not rubric.is_file():
        raise SystemExit(
            f"No transferred rubric at {rubric}\n"
            f"  Folder 4 initializes from folder 2's portion-{C.BASE_PORTION} "
            f"rubric for source '{transfer_source}', seed {args.seed}, "
            f"model '{args.model}'.\n"
            f"  Run folder 2 for that cell first. Note its runs/ is gitignored, "
            f"so a fresh clone will not have it even if folder 2 was run elsewhere."
        )

    base_runner = C.BASE_CONFIG_DIR / "run_single.py"
    if not base_runner.is_file():
        raise SystemExit(f"Cannot find folder 2's runner at {base_runner}")

    cmd = [
        sys.executable, str(base_runner),
        "--model", args.model,
        "--source", args.source,
        "--portion", args.portion,
        "--seed", str(args.seed),
        "--base-url", args.base_url,
        "--max-workers", str(args.max_workers),
        "--init-rubric", str(rubric.resolve()),
    ]
    if args.server_model_name:
        cmd += ["--server-model-name", args.server_model_name]
    if args.step_count is not None:
        cmd += ["--step-count", str(args.step_count)]
    if args.force:
        cmd += ["--force"]

    print(f"[info] target={args.source} portion={args.portion} "
          f"seed={args.seed} model={args.model}")
    print(f"[info] transferring rubric from {transfer_source} "
          f"portion-{C.BASE_PORTION}:")
    print(f"[info]   {rubric}")
    print(f"[info] delegating to folder 2's runner; outputs -> "
          f"{(C.OUTPUT_DIR / args.source).resolve()}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(C.BASE_CONFIG_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    raise SystemExit(subprocess.call(cmd, env=env))

if __name__ == "__main__":
    main()