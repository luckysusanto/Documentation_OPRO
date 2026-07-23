import sys 
from pathlib import Path 

BASE_CONFIG_DIR = Path(__file__).resolve().parent.parent/"1_RQ1_IndoBERTweet_Baseline"
if not BASE_CONFIG_DIR.is_dir():
    raise FileNotFoundError("Failed to find Folder 1. Check path.")

# CHANGE FROM FOLDER 1'S CONFIG, DO NOT TOUCH
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_folder1_config", BASE_CONFIG_DIR / "config.py")
BASE = _ilu.module_from_spec(_spec)
sys.modules["_folder1_config"] = BASE
_spec.loader.exec_module(BASE)

DATA_DIR                 = BASE.DATA_DIR
MODEL_NAME_OR_PATH       = BASE.MODEL_NAME_OR_PATH
LOCAL_FILES_ONLY         = BASE.LOCAL_FILES_ONLY
 
SOURCES                  = BASE.SOURCES
SEEDS                    = BASE.SEEDS
PORTIONS                 = BASE.PORTIONS
 
TEXT_COL                 = BASE.TEXT_COL
LABEL_COL                = BASE.LABEL_COL
LABEL_MAP                = BASE.LABEL_MAP
 
MAX_LENGTH               = BASE.MAX_LENGTH
NUM_LABELS               = BASE.NUM_LABELS
EARLY_STOPPING_PATIENCE  = BASE.EARLY_STOPPING_PATIENCE
EARLY_STOPPING_THRESHOLD = BASE.EARLY_STOPPING_THRESHOLD
EVAL_METRIC              = BASE.EVAL_METRIC
GREATER_IS_BETTER        = BASE.GREATER_IS_BETTER
MAX_EPOCHS               = BASE.MAX_EPOCHS
EVAL_STEPS_STRATEGY      = BASE.EVAL_STEPS_STRATEGY

_SHARED_WITH_BASE = [
    "DATA_DIR", "MODEL_NAME_OR_PATH", "LOCAL_FILES_ONLY",
    "SOURCES", "SEEDS", "PORTIONS",
    "TEXT_COL", "LABEL_COL", "LABEL_MAP",
    "MAX_LENGTH", "NUM_LABELS",
    "EARLY_STOPPING_PATIENCE", "EARLY_STOPPING_THRESHOLD",
    "EVAL_METRIC", "GREATER_IS_BETTER", "MAX_EPOCHS", "EVAL_STEPS_STRATEGY",
]
 
_drifted = [
    name for name in _SHARED_WITH_BASE
    if globals()[name] != getattr(BASE, name)
]
if _drifted:
    raise ValueError(
        "Folder 3's config has drifted from folder 1's for: "
        f"{', '.join(_drifted)}.\n"
    )
# DO NOT TOUCH ABOVE

# Edittables
OUTPUT_DIR = Path("runs")

BASE_RUNS_DIR = BASE_CONFIG_DIR / BASE.OUTPUT_DIR
BASE_PORTION = "100"

# Experiment: finetune a model trained on SOURCE onto OTHER
def transfer_source_for(source):
    others = [s for s in SOURCES if s != source]
    if len(others) != 1:
        raise ValueError(f"Check source. Should only be either `csis` or `indoDiscourse`.")
    return others[0]

HP_SEARCH_PORTION = "010"
HP_SEARCH_SEED = 42
HP_N_TRIALS = 15

# Lower than folder 1, since this is continued finetuning.
HP_SPACE = {
    "learning_rate": (5e-6, 3e-5),          # log-uniform
    "per_device_train_batch_size": [16, 32],
    "weight_decay": (0.0, 0.1),             # uniform
    "warmup_ratio": (0.0, 0.1),             # uniform
}

DEFAULT_HP = {
    "learning_rate": 1e-5,
    "per_device_train_batch_size": 16,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
}
