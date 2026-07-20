#!/usr/bin/env bash
#
# Batch-normalize every .csv in an input folder by calling normalize_hate.py
# once per file. Does NOT modify the Python script.
#
# Usage:
#   ./run_normalize.sh <input_folder> <output_folder> [extra python flags...]
#
# Examples:
#   ./run_normalize.sh ./raw ./clean
#   ./run_normalize.sh ./raw ./clean --lowercase
#   ./run_normalize.sh ./raw ./clean --keep-hash
#
# Any extra args after the two folders are passed straight through to
# normalize_data.py (e.g. --lowercase, --keep-hash).

set -euo pipefail

# --- resolve the Python script location (same dir as this wrapper) ---------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/normalize_data.py"

# --- args ------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_folder> <output_folder> [extra python flags...]" >&2
  exit 1
fi

input_folder="$1"
output_folder="$2"
shift 2
extra_flags=("$@")   # anything left over -> passed to python

# --- validation ------------------------------------------------------------
if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: normalize_data.py not found next to this script ($PY_SCRIPT)" >&2
  exit 1
fi
if [[ ! -d "$input_folder" ]]; then
  echo "ERROR: input folder not found: $input_folder" >&2
  exit 1
fi

mkdir -p "$output_folder"

# --- collect .csv files (case-insensitive, no crash on empty match) --------
shopt -s nullglob nocaseglob
csv_files=("$input_folder"/*.csv)
shopt -u nullglob nocaseglob

if [[ ${#csv_files[@]} -eq 0 ]]; then
  echo "ERROR: no .csv files found in $input_folder" >&2
  exit 1
fi

echo "Found ${#csv_files[@]} CSV file(s) in $input_folder"
echo

# --- loop: one python call per file ----------------------------------------
n_ok=0
n_fail=0
for f in "${csv_files[@]}"; do
  # Skip files that are already normalized outputs, so re-running on a
  # folder that mixes inputs and outputs doesn't double-process them.
  case "$(basename "$f")" in
    *_normalized.csv)
      echo "[SKIP] $(basename "$f") (already a _normalized output)"
      continue
      ;;
  esac

  echo "[RUN ] $(basename "$f")"
  if python3 "$PY_SCRIPT" "$f" -o "$output_folder" "${extra_flags[@]}"; then
    n_ok=$((n_ok + 1))
  else
    echo "[FAIL] $(basename "$f") (see error above)" >&2
    n_fail=$((n_fail + 1))
  fi
done

echo
echo "Done. ${n_ok} succeeded, ${n_fail} failed -> ${output_folder}"

# Non-zero exit if anything failed, so pipelines notice.
[[ $n_fail -eq 0 ]]