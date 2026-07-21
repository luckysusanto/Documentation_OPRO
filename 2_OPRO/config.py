from pathlib import Path 

DATA_DIR = Path("../Data/Normalized")
OUTPUT_DIR = Path("runs")

SOURCES = ["csis", "indoDiscourse"]
SEED = [0, 42, 2026]
PORTIONS = [
    "010", "020", "030", "040", "050",
    "060", "070", "080", "090", "100"
]

TEXT_COL = "text"
LABEL_COL = "label"

HATE_LABEL = "hate"
NONHATE_LABEL = "not-hate" 

# FULL MODEL
# MODEL_REPO = {
#     "qwen3-8b": "Qwen/Qwen3-8B",
#     "qwen3-32b": "Qwen/Qwen3-32B",
#     "sealion-qwen-32b": "aisingapore/Qwen-SEA-LION-v4-32B-IT",
#     "gemma4-12b": "google/gemma-4-12b-it",
#     "gemma4-31b": "google/gemma-4-31b-it"
# }

# Suggested Model
MODEL_REPO = {
    "qwen3-32b": "Qwen/Qwen3-32B",
    "sealion-qwen-32b": "aisingapore/Qwen-SEA-LION-v4-32B-IT",
    "gemma4-31b": "google/gemma-4-31b-it"
}

# vLLM server setting
BASE_URL: "http://127.0.0.1:8000/v1"
MAX_WORKER = 64 # max network request in flight to vLLM server, not dataset worker
SCORER_MAX_TOKENS = 4096
OPT_MAX_TOKENS = 8192

# OPRO HYPERPARAMETER
STEP_COUNT = 15
CANDIDATE_PER_STEP = 4
HATE_EXEMPLAR_PER_STEP = 2
NONHATE_EXEMPLAR_PER_STEP = 2
TEMPERATURE = 1.0
CRITERIA_CAP = 15
KEEP_TOP_K_CANDIDATE = 5
BIN_WIDTH = 2 # Coarse binning of F1. Needed because an F1 of 0.634 vs 0.637 does not provide a good enough signal.

# Retry Policy
RETRY_MAX = 5
RETRY_BACKOFF = 0.5

# Selection/Reporting Split
SELECTION_SPLIT = "val_set"
REPORT_SPLIT = "test_set"
EVAL_METRIC = "macro_f1"
