"""
All important hyperparameters can be set from this file.
"""

from pathlib import Path 

# Paths
## HF_HOME = Path("../MODEL_CACHE") # Set directly using export
DATA_DIR = Path("../Data/Normalized")
MODEL_NAME_OR_PATH = "indolem/indobertweet-base-uncased"
LOCAL_FILES_ONLY = True
OUTPUT_DIR = Path("runs")

# Data
SOURCES = ["csis", "indoDiscourse"]
SEEDS = [0, 42, 2026]
PORTIONS = [
    "010", "020", "030", "040", "050",
    "060", "070", "080", "090", "100"
]

## CSV Schema
TEXT_COL = "text"
LABEL_COL = "label"
LABEL_MAP = {
    "not-hate": 0,
    "hate": 1
}

# Fixed training config
MAX_LENGTH = 512
NUM_LABELS = 2
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 1e-4
EVAL_METRIC = "eval_macro_f1"
GREATER_IS_BETTER = True 
MAX_EPOCHS = 10
EVAL_STEPS_STRATEGY = "epoch"

# Hyperparameter to search
HP_SEARCH_PORTION = "100" # Search at full data
HP_SEARCH_SEED = 42
HP_N_TRIALS = 15

HP_SPACE = {
    "learning_rate": (1e-5, 5e-5), # log-uniform
    "per_device_train_batch_size": [16, 32],
    "weight_decay": (0.0, 0.1), # uniform
    "warmup_ratio": (0.0, 0.1), # uniform
}

# If no hyperparameter searching, just use default
DEFAULT_HP = {
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
}
