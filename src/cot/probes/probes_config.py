from __future__ import annotations
from pathlib import Path

"""
Configuration for run_geom and test_fixed_dimension

Edit these values to point to your activation NPZ/labels and to control
training/testing behavior. This file intentionally contains only the
settings used by the two scripts for clarity.
"""

# ========== Paths ==========
# Activation features NPZ (relative to this file)
ACTS_NPZ = Path("../outputs/collect/resid_post_qwen3_collect.npz")
# Labels CSV path; if None, inferred from ACTS_NPZ by replacing suffix
LABELS_CSV = None

# ========== Feature keys ==========
# Prefix for NPZ keys, e.g., acts_resid_post_layer12
HOOK_POINT_PREFIX = "acts_resid_post_layer"

# Optionally restrict which layers to train/evaluate (list of ints) or None for all
LAYERS_TO_TRAIN: list[int] | None = None

# ========== Output ==========
# Probe reports and scores are written under this directory
OUT_DIR = Path("../outputs/probes")
# Learned vectors from run_geom (rank1/lowrank) are saved here under a run-named subfolder
VECTORS_DIR = Path("../outputs/vectors")

# Optional override for the auto-generated run/output name. When set, all
# outputs (reports, scores, vectors) will be saved under this name instead of
# the default constructed name. Example: OUTPUT_FILE_NAME = "my_experiment_run".
# Set to None to keep auto-naming behavior.
OUTPUT_FILE_NAME: str | None = "run1-scan_all_layers"


# ========== Labels schema ==========
TARGET_COL = "p_value"       # column with labels (e.g., "True"/"False")
POSITIVE_TOKEN = "true"      # positive class (case-insensitive match in TARGET_COL)
GROUP_COL = "example_id"     # group tokens by example for grouped CV
REGIME_COL = "regime"        # used for regime-based filtering and reporting
OFFSET_COL = "offset_from_split"  # token offset relative to split

# ========== Data selection ==========
# Restrict which regimes to use (list of names) or None for all
# Choose from ["i_initial", "ii_inconsequential", "iii_derived", "iv_indeterminate", "v_output", "vi_single_use", "vii_max_use"]
REGIMES_TO_USE: list[str] | None = ["vii_max_use"]

# Offset filters (applied to OFFSET_COL)
FILTER_OFFSET_EQ = None      # exactly equal to this offset, or None
FILTER_OFFSET_MAX = None     # include offsets <= this value, or None
# Either an inclusive range tuple (lo, hi) with None for open bounds,
# or a list of explicit offsets to whitelist (e.g., [0,2,4,6,8])
FILTER_OFFSET_RANGE: tuple[int | None, int | None] | list[int] | None = (-40, 0)

# Optional random subsample of tokens after filtering; None to use all
N_TOKENS: int | None = 10000

# ========== Model / probe ==========
# One of: 'rank1' (k=1) or 'lowrank' (k>1)
CLASSIFIER = "rank1"

# Logistic regression C for the classifier trained on the learned subspace projections
LOGREG_C = 1.0

# Low-rank subspace options
LOWRANK_K = 4             # number of components when lowrank; rank1 forces k=1
LOWRANK_METHOD = "lda"     # 'lda' or 'pls' 

# ========== Evaluation ==========
N_SPLITS = 0              # GroupKFold splits (by GROUP_COL)
N_JOBS = 8               # parallelism for supported estimators
RANDOM_STATE = 0          # RNG seed for reproducibility

# Optional comparison vs DoM/whitened DoM directions in run_geom report
# 'none' | 'dom' | 'whitened_dom' | 'both'
COMPARE_MODE = "dom"
COMPARE_REG_EPS = 1e-3    # Tikhonov epsilon for whitening (Σ + eps I)^-1

# Explicit train/test split options are handled in fixed_test_config for test-only evaluation.

# ========== Report formatting ==========
COL_WIDTHS = dict(layer=6, n=9, comps=7, acc=10, auroc=10, ap=10, f1=10)

# ========== Acceleration ==========
# DEVICE: 'cpu', 'cuda', or 'auto' (use CUDA if available)
DEVICE = 'cpu'
# Torch logistic regression hyperparameters when using GPU ('cuda')
GPU_LR_EPOCHS = 300
GPU_LR_LR = 0.1
GPU_LR_WEIGHT_DECAY = None  # if None, uses 1/LOGREG_C

# ========== Naming ==========
# Limit for run directory name length to prevent OS path issues
MAX_RUN_NAME_LEN = 120

