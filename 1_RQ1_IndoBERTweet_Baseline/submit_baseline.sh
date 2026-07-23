#!/bin/bash
# =============================================================================
# submit_baseline.sh  --  queues the full IndoBERTweet data-portion grid.
#
#   2 sources x 10 portions x 3 seeds = 60 cells, one qsub job each (gpus=1, l40s).
#   The SGE scheduler runs them in parallel as GPUs free up.
#   Uses run_baseline_cell.qsub.
#
# Resumable: cells whose runs/<source>/<cell>/test_metrics.json already exists are
# NOT re-queued. So after a partial run, just re-run this script and only the
# missing cells get submitted.
#
# Prereq: HP search already done (runs/<source>/best_hp.json for every source).
#
# Usage:
#   ./submit_baseline.sh                 # queue all missing cells (both sources)
#   ./submit_baseline.sh csis            # only this source
#   ./submit_baseline.sh csis indoDiscourse
#   ./submit_baseline.sh --smoke         # SMOKE TEST: queue ONE probe cell only
#                                        #   (csis 010 seed 0) to test the qsub path
# =============================================================================

QSUB="run_baseline_cell.qsub"

# Grid values pulled from config.py so this never drifts from the Python side.
mapfile -t PORTIONS < <(python -c "import config as C; print('\n'.join(C.PORTIONS))")
mapfile -t SEEDS    < <(python -c "import config as C; print('\n'.join(str(s) for s in C.SEEDS))")
OUTPUT_DIR="$(python -c "import config as C; print(C.OUTPUT_DIR)")"

mkdir -p logs

# Smoke test, access by: ./submit_baseline.sh --smoke
if [ "${1:-}" = "--smoke" ]; then
  qsub "$QSUB" csis 010 0
  echo "Submitted. Check logs/ for output."
  exit 0
fi

# Accepts one or multiple ./submit_baseline.sh csis indoDiscourse
if [ "$#" -gt 0 ]; then
  SOURCES=("$@")
else
  mapfile -t SOURCES < <(python -c "import config as C; print('\n'.join(C.SOURCES))")
fi

# Check hyperparameter searching is already done for the source
for SRC in "${SOURCES[@]}"; do
  if [ ! -f "$OUTPUT_DIR/$SRC/best_hp.json" ]; then
    echo "ERROR: $OUTPUT_DIR/$SRC/best_hp.json missing. Run the HP search for '$SRC' first." >&2
    exit 1
  fi
done

echo "=== baseline grid: sources=[${SOURCES[*]}] portions=[${PORTIONS[*]}] seeds=[${SEEDS[*]}] ==="

queued=0
skipped=0
for SRC in "${SOURCES[@]}"; do
  for PORTION in "${PORTIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      CELL="${SRC}_${PORTION}_${SEED}"
      if [ -f "$OUTPUT_DIR/$SRC/$CELL/test_metrics.json" ]; then
        echo "skip   $CELL (already complete)"
        skipped=$((skipped+1))
        continue
      fi
      echo "SUBMIT $CELL"
      qsub "$QSUB" "$SRC" "$PORTION" "$SEED"
      queued=$((queued+1))
    done
  done
done

echo "=== queued $queued cell(s), skipped $skipped already-complete. ==="