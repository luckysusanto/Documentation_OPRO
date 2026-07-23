#!/bin/bash

# Usage:
#     ./submit_opro.sh                                    # Queue all missing cells, all models/sources
#     ./submit_opro.sh --model qwen3-32b                  # Restrict to one model
#     ./submit_opro.sh --source csis                      # Restrict to one source
#     ./submit_opro.sh --model qwen3-32b --source csis    # Restrict to one model + one source
#     ./submit_opro.sh --smoke                            # Test run to see if the code work

set -euo pipefail

QSUB="run_opro_cell.qsub"
MODELS=("qwen3-32b" "sealion-qwen-32b")

# Everything else comes from config.py
mapfile -t PORTIONS   < <(python -c "import config as C; print('\n'.join(C.PORTIONS))")
mapfile -t SEEDS      < <(python -c "import config as C; print('\n'.join(str(s) for s in C.SEEDS))")
mapfile -t ALL_SOURCES < <(python -c "import config as C; print('\n'.join(C.SOURCES))")
OUTPUT_DIR="$(python -c "import config as C; print(C.OUTPUT_DIR)")"

mkdir -p logs

DRY_RUN=0
PICK_MODEL=""
PICK_SOURCE=""
 
while [ "$#" -gt 0 ]; do
  case "$1" in
    --smoke)
      echo "SMOKE TEST: one cell, 2 optimizer steps, csis/010/seed 0 on ${MODELS[0]}."
      echo "Writes to $OUTPUT_DIR/csis/csis_010_0_${MODELS[0]}/ -- delete that dir to re-run."
      qsub -N "opro_smoke" "$QSUB" "${MODELS[0]}" csis 010 0 2
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
  # Guard against typos and against models with no qsub case branch.
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
 
TOTAL=$(( ${#MODELS[@]} * ${#SOURCES[@]} * ${#PORTIONS[@]} * ${#SEEDS[@]} ))
echo "=== OPRO grid: models=[${MODELS[*]}] sources=[${SOURCES[*]}] \
portions=[${PORTIONS[*]}] seeds=[${SEEDS[*]}] -> $TOTAL cell(s) ==="
[ "$DRY_RUN" -eq 1 ] && echo "*** DRY RUN: nothing will be submitted ***"
 
queued=0
skipped=0
for MODEL in "${MODELS[@]}"; do
  for SRC in "${SOURCES[@]}"; do
    for PORTION in "${PORTIONS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        # Must match run_single.py's run_dir exactly: <source>_<portion>_<seed>_<model>
        CELL="${SRC}_${PORTION}_${SEED}_${MODEL}"
        if [ -f "$OUTPUT_DIR/$SRC/$CELL/test_metrics.json" ]; then
          echo "skip   $CELL (already complete)"
          skipped=$((skipped+1))
          continue
        fi
        echo "SUBMIT $CELL"
        if [ "$DRY_RUN" -eq 0 ]; then
          # Unique -N keeps qstat readable and log filenames distinct.
          qsub -N "opro_${MODEL}_${SRC}_${PORTION}_s${SEED}" \
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
