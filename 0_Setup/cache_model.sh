#!/usr/bin/env bash

set -euo pipefail

export HF_HOME="../MODEL_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$HF_HOME"

echo "HF_HOME = $HF_HOME"

pip install -q --upgrade "hugggingface_hub[hf_transfer]" 2>/dev/null || \
    echo "Failed to download hf_transfer, using default huggingface_hub instead."

# Need around 230 GB of space.
# Qwen3-8B ~16GB, Qwen3-32B ~65GB, SEA-LION-32B ~65GB, Gemma 12B ~24GB,
# Gemma 31B ~62GB  => roughly 230GB in bf16. Budget ~300GB to be safe.

AVAIL_GB=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc '0-9')
if [ "${AVAIL_GB:-0}" -lt 300 ]; then
    echo "WARNING: Recommended space needed is 300GB. You only have ${AVAIL_GB}GB free at $HF_HOME."
    echo "Ctrl-C to abort. Continuing in 10 seconds."
    sleep 10
fi 

python cache_model.py