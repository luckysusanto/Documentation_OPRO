#!/bin/bash
#$ -P multilm
#$ -N cont_hp_search
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l gpu_memory=48G
#$ -l h_rt=08:00:00
#$ -j y
#$ -o bash_logs/

set -euo pipefail

export HF_HOME=../MODEL_CACHE

if [ ! -d "$HF_HOME/hub" ]; then
    echo "WARNING: $HF_HOME/hub not found. Models will be cached to default path."
fi

SOURCES=("csis" "indoDiscourse")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "=================================================="
echo "Continued-fine-tuning HP search starting"
echo "  HF_HOME : $HF_HOME"
echo "  targets : ${SOURCES[*]}"
echo "  logs    : $LOG_DIR/"
echo "=================================================="

for SRC in "${SOURCES[@]}"; do
  LOG_FILE="$LOG_DIR/cont_hpsearch_${SRC}_${STAMP}.log"
  echo ""
  echo ">>> [$SRC] starting  (log: $LOG_FILE)"

  # tee so you see progress live AND keep a full log per source.
  if python hyperparameter_searching.py --sources "$SRC" 2>&1 | tee "$LOG_FILE"; then
    echo ">>> [$SRC] done"
  else
    echo ">>> [$SRC] FAILED — see $LOG_FILE" >&2
    exit 1
  fi
done

echo ""
echo "=================================================="
echo "DONE"
echo "=================================================="
