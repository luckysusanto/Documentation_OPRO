#!/bin/bash

set -euo pipefail

QSUB="run_transfer_cell.qsub"

MODELS=("qwen3-32b" "sealion-qwen-32b")

mapfile -t PORTIONS    < <(python -c "import config as C; print('\n'.join(C.PORTIONS))")
mapfile -t SEEDS       < <(python -c "import config as C; print('\n'.join(str(s) for s in C.SEEDS))")
mapfile -t ALL_SOURCES < <(python -c "import config as C; print('\n'.join(C.SOURCES))")
OUTPUT_DIR="$(python -c "import config as C; print(C.OUTPUT_DIR)")"
BASE_PORTION="$(python -c "import config as C; print(C.BASE_PORTION)")"

mkdir -p logs

DRY_RUN=0
PICK_MODEL=""
PICK_SOURCE=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --smoke)
      echo "SMOKE TEST: one cell, 2 optimizer steps, indoDiscourse/010/seed 0 on ${MODELS[0]}."
      echo "Transfers the csis portion-$BASE_PORTION rubric for that seed and model."
      echo "Writes to $OUTPUT_DIR/indoDiscourse/indoDiscourse_010_0_${MODELS[0]}/ -- delete to re-run."
      qsub -N "transfer_smoke" "$QSUB" "${MODELS[0]}" indoDiscourse 010 0 2
      echo "Submitted. Watch with: qstat -u $USER ; then check logs/"
      exit 0
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    --model)   PICK_MODEL="${2:?--model needs a value}"; shift 2 ;;
    --source)  PICK_SOURCE="${2:?--source needs a value}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -n "$PICK_MODEL" ]; then
  found=0
  for m in "${MODELS[@]}"; do [ "$m" = "$PICK_MODEL" ] && found=1; done
  if [ "$found" -eq 0 ]; then
    echo "ERROR: '$PICK_MODEL' is not a runnable model. Known: ${MODELS[*]}" >&2
    exit 1
  fi
  MODELS=("$PICK_MODEL")
fi

if [ -n "$PICK_SOURCE" ]; then
  found=0
  for s in "${ALL_SOURCES[@]}"; do [ "$s" = "$PICK_SOURCE" ] && found=1; done
  if [ "$found" -eq 0 ]; then
    echo "ERROR: '$PICK_SOURCE' is not a known source. Known: ${ALL_SOURCES[*]}" >&2
    exit 1
  fi
  SOURCES=("$PICK_SOURCE")
else
  SOURCES=("${ALL_SOURCES[@]}")
fi

# --- prereq: folder 2's transferred rubrics ------------------------------------
# Checked up front: 120 jobs each failing on a missing rubric is a slow and
# confusing way to discover folder 2 has not finished.
missing=0
for MODEL in "${MODELS[@]}"; do
  for SRC in "${SOURCES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUBRIC="$(python -c "
import config as C
print(C.rubric_path(C.transfer_source_for('$SRC'), $SEED, '$MODEL'))")"
      if [ ! -f "$RUBRIC" ]; then
        echo "MISSING rubric: $RUBRIC" >&2
        missing=$((missing+1))
      fi
    done
  done
done
if [ "$missing" -gt 0 ]; then
  echo "" >&2
  echo "ERROR: $missing folder-2 rubric(s) missing." >&2
  echo "       Folder 4 initializes from folder 2's portion-$BASE_PORTION runs." >&2
  echo "       Run folder 2 to completion first. Its runs/ is gitignored, so a" >&2
  echo "       fresh clone will not have them even if folder 2 was run elsewhere." >&2
  exit 1
fi

TOTAL=$(( ${#MODELS[@]} * ${#SOURCES[@]} * ${#PORTIONS[@]} * ${#SEEDS[@]} ))
echo "=== transfer grid: models=[${MODELS[*]}] targets=[${SOURCES[*]}] \
portions=[${PORTIONS[*]}] seeds=[${SEEDS[*]}] -> $TOTAL cell(s) ==="
[ "$DRY_RUN" -eq 1 ] && echo "*** DRY RUN: nothing will be submitted ***"

queued=0
skipped=0
for MODEL in "${MODELS[@]}"; do
  for SRC in "${SOURCES[@]}"; do
    for PORTION in "${PORTIONS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        # Same cell naming as folder 2, so RQ1 and RQ2 cells pair by name.
        CELL="${SRC}_${PORTION}_${SEED}_${MODEL}"
        if [ -f "$OUTPUT_DIR/$SRC/$CELL/test_metrics.json" ]; then
          echo "skip   $CELL (already complete)"
          skipped=$((skipped+1))
          continue
        fi
        echo "SUBMIT $CELL"
        if [ "$DRY_RUN" -eq 0 ]; then
          qsub -N "xfer_${MODEL}_${SRC}_${PORTION}_s${SEED}" \
               "$QSUB" "$MODEL" "$SRC" "$PORTION" "$SEED"
        fi
        queued=$((queued+1))
      done
    done
  done
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "=== DRY RUN: would queue $queued cell(s), skipping $skipped already-complete. ==="
else
  echo "=== queued $queued cell(s), skipped $skipped already-complete. ==="
  echo "Track them with:  qstat -u $USER"
fi