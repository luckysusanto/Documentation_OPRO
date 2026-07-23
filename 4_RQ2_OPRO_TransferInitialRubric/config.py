import sys 
from pathlib import Path 

BASE_CONFIG_DIR = Path(__file__).resolve().parent.parent / "2_RQ1_OPRO"
if not BASE_CONFIG_DIR.is_dir():
    raise FileNotFoundError(
        f"Cannot find folder 2 at {BASE_CONFIG_DIR}. Folder 4 inherits its config, "
        f"reuses its OPRO implementation, and reads its optimized rubrics. "
        f"If folder 2 was renamed, update BASE_CONFIG_DIR."
    )

# NO EDIT ZONE -- EDIT FOLDER 2 INSTEAD
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("_folder2_config", BASE_CONFIG_DIR / "config.py")
BASE = _ilu.module_from_spec(_spec)
sys.modules["_folder2_config"] = BASE
_spec.loader.exec_module(BASE)

DATA_DIR      = BASE.DATA_DIR
SOURCES       = BASE.SOURCES
SEEDS         = BASE.SEEDS
PORTIONS      = BASE.PORTIONS
TEXT_COL      = BASE.TEXT_COL
LABEL_COL     = BASE.LABEL_COL
HATE_LABEL    = BASE.HATE_LABEL
NONHATE_LABEL = BASE.NONHATE_LABEL
MODEL_REPO    = BASE.MODEL_REPO
 
BASE_URL          = BASE.BASE_URL
MAX_WORKERS       = BASE.MAX_WORKERS
SCORER_MAX_TOKENS = BASE.SCORER_MAX_TOKENS
OPT_MAX_TOKENS    = BASE.OPT_MAX_TOKENS
 
STEP_COUNT                = BASE.STEP_COUNT
CANDIDATE_PER_STEP        = BASE.CANDIDATE_PER_STEP
HATE_EXEMPLAR_PER_STEP    = BASE.HATE_EXEMPLAR_PER_STEP
NONHATE_EXEMPLAR_PER_STEP = BASE.NONHATE_EXEMPLAR_PER_STEP
TEMPERATURE               = BASE.TEMPERATURE
CRITERIA_CAP              = BASE.CRITERIA_CAP
KEEP_TOP_K_CANDIDATE      = BASE.KEEP_TOP_K_CANDIDATE
BIN_WIDTH                 = BASE.BIN_WIDTH
 
RETRY_MAX     = BASE.RETRY_MAX
RETRY_BACKOFF = BASE.RETRY_BACKOFF
 
SELECTION_SPLIT = BASE.SELECTION_SPLIT
REPORT_SPLIT    = BASE.REPORT_SPLIT
EVAL_METRIC     = BASE.EVAL_METRIC
 
_SHARED_WITH_BASE = [
    "DATA_DIR", "SOURCES", "SEEDS", "PORTIONS",
    "TEXT_COL", "LABEL_COL", "HATE_LABEL", "NONHATE_LABEL", "MODEL_REPO",
    "BASE_URL", "MAX_WORKERS", "SCORER_MAX_TOKENS", "OPT_MAX_TOKENS",
    "STEP_COUNT", "CANDIDATE_PER_STEP",
    "HATE_EXEMPLAR_PER_STEP", "NONHATE_EXEMPLAR_PER_STEP",
    "TEMPERATURE", "CRITERIA_CAP", "KEEP_TOP_K_CANDIDATE", "BIN_WIDTH",
    "RETRY_MAX", "RETRY_BACKOFF",
    "SELECTION_SPLIT", "REPORT_SPLIT", "EVAL_METRIC",
]
 
_drifted = [
    name for name in _SHARED_WITH_BASE
    if globals()[name] != getattr(BASE, name)
]
if _drifted:
    raise ValueError(
        "Folder 4's config has drifted from folder 2's for: "
        f"{', '.join(_drifted)}.\n"
    )

# Editables
OUTPUT_DIR = Path("runs") # This directs to folder 4's "runs"

BASE_RUNS_DIR = BASE_CONFIG_DIR / BASE.OUTPUT_DIR
BASE_PORTION  = "100"   # "old data is finalized" -> always transfer from the full-data rubric
 
 
def transfer_source_for(target_source: str) -> str:
    others = [s for s in SOURCES if s != target_source]
    if len(others) != 1:
        raise ValueError(f"Check source. Should only be either `csis` or `indoDiscourse`.")
    return others[0]
 
 
def rubric_path(transfer_source: str, seed: int, model: str) -> Path:
    return (
        BASE_RUNS_DIR
        / transfer_source
        / f"{transfer_source}_{BASE_PORTION}_{seed}_{model}"
        / "best_rubric.txt"
    )
