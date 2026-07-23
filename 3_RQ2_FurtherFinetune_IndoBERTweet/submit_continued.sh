#!/bin/bash

set -euo pipefail

QSUB="run_continued_cell.qsub"

mapfile -t PORTIONS < <(python -c "import config as C; print('\n'.join(C.PORTIONS))")
mapfile -t SEEDS    < <(python -c "import config as C; print('\n'.join(str(s) for s in C.SEEDS))")
OUTPUT_DIR="$(python -c "import config as C; print(C.OUTPUT_DIR)")"
BASE_RUNS="$(python -c "import config as C; print(C.BASE_RUNS_DIR)")"
BASE_PORTION="$(python -c "import config as C; print(C.BASE_PORTION)")"

mkdir -p logs

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1; shift
fi

if [ "${1:-}" = "--smoke" ]; then
  echo "SMOKE TEST: one cell, indoDiscourse portion 010 seed 0."
  echo "Writes to $OUTPUT_DIR/indoDiscourse/indoDiscourse_010_0/ -- delete to re-run."
  qsub "$QSUB" indoDiscourse 010 0
  echo "Submitted. Check logs/ for output."
  exit 0
fi

if [ "$#" -gt 0 ]; then
  SOURCES=("$@")
else
  mapfile -t SOURCES < <(python -c "import config as C; print('\n'.join(C.SOURCES))")
fi

# --- prereq 1: folder 1 checkpoints -------------------------------------------
# Checked up front rather than per-job: 60 jobs each failing on a missing
# checkpoint is a slow and confusing way to discover folder 1 has not been run.
missing_ckpt=0
for SRC in "${SOURCES[@]}"; do
  TRANSFER="$(python -c "import config as C; print(C.transfer_source_for('$SRC'))")"
  for SEED in "${SEEDS[@]}"; do
    CKPT="$BASE_RUNS/$TRANSFER/${TRANSFER}_${BASE_PORTION}_${SEED}/best_model"
    if [ ! -d "$CKPT" ]; then
      echo "MISSING checkpoint: $CKPT" >&2
      missing_ckpt=$((missing_ckpt+1))
    fi
  done
done
if [ "$missing_ckpt" -gt 0 ]; then
  echo "" >&2
  echo "ERROR: $missing_ckpt folder-1 checkpoint(s) missing." >&2
  echo "       Folder 3 initializes from folder 1's portion-$BASE_PORTION models." >&2
  echo "       Run folder 1 to completion first. These are gitignored, so a fresh" >&2
  echo "       clone will not have them even if folder 1 was run elsewhere." >&2
  exit 1
fi

# --- prereq 2: continued-FT hyperparameters -----------------------------------
for SRC in "${SOURCES[@]}"; do
  if [ ! -f "$OUTPUT_DIR/$SRC/best_hp.json" ]; then
    echo "ERROR: $OUTPUT_DIR/$SRC/best_hp.json missing." >&2
    echo "       Run:  python hyperparameter_searching.py --sources $SRC" >&2
    echo "       Do NOT copy folder 1's best_hp.json -- its LR was tuned for" >&2
    echo "       from-scratch training and is typically too high here, which" >&2
    echo "       would manufacture a false negative-transfer result." >&2
    exit 1
  fi
done

echo "=== continued-FT grid: targets=[${SOURCES[*]}] portions=[${PORTIONS[*]}] seeds=[${SEEDS[*]}] ==="
[ "$DRY_RUN" -eq 1 ] && echo "*** DRY RUN: nothing will be submitted ***"

queued=0
skipped=0
for SRC in "${SOURCES[@]}"; do
  for PORTION in "${PORTIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      # Same cell naming as folder 1, so the two folders' results pair by name.
      CELL="${SRC}_${PORTION}_${SEED}"
      if [ -f "$OUTPUT_DIR/$SRC/$CELL/test_metrics.json" ]; then
        echo "skip   $CELL (already complete)"
        skipped=$((skipped+1))
        continue
      fi
      echo "SUBMIT $CELL"
      if [ "$DRY_RUN" -eq 0 ]; then
        qsub -N "cont_${SRC}_${PORTION}_s${SEED}" "$QSUB" "$SRC" "$PORTION" "$SEED"
      fi
      queued=$((queued+1))
    done
  done
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "=== DRY RUN: would queue $queued cell(s), skipping $skipped already-complete. ==="
else
  echo "=== queued $queued cell(s), skipped $skipped already-complete. ==="
  echo "Track them with:  qstat -u $USER"
fi
