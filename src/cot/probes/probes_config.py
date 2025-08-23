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
LAYERS_TO_TRAIN: list[int] | None = [12]

# ========== Output ==========
# Probe reports and scores are written under this directory
OUT_DIR = Path("../outputs/probes")
# Learned vectors from run_geom (rank1/lowrank) are saved here under a run-named subfolder
VECTORS_DIR = Path("../outputs/vectors")

# ========== Labels schema ==========
TARGET_COL = "p_value"       # column with labels (e.g., "True"/"False")
POSITIVE_TOKEN = "true"      # positive class (case-insensitive match in TARGET_COL)
GROUP_COL = "example_id"     # group tokens by example for grouped CV
REGIME_COL = "regime"        # used for regime-based filtering and reporting
OFFSET_COL = "offset_from_split"  # token offset relative to split

# ========== Data selection ==========
# Restrict which regimes to use (list of names) or None for all
REGIMES_TO_USE: list[str] | None = ["v_output"] #["i_initial", "iv_indeterminate", "v_output"]

# Offset filters (applied to OFFSET_COL)
FILTER_OFFSET_EQ = None      # exactly equal to this offset, or None
FILTER_OFFSET_MAX = None     # include offsets <= this value, or None
# Either an inclusive range tuple (lo, hi) with None for open bounds,
# or a list of explicit offsets to whitelist (e.g., [0,2,4,6,8])
FILTER_OFFSET_RANGE: tuple[int | None, int | None] | list[int] | None = (0, 10)

# Optional random subsample of tokens after filtering; None to use all
N_TOKENS: int | None = 20000

# ========== Model / probe ==========
# One of: 'ridge', 'logreg', 'rank1', 'lowrank'
CLASSIFIER = "rank1"

# Ridge/logreg options
RIDGE_ALPHA = 1.0
LOGREG_C = 1.0

# PCA options for ridge/logreg
# PCA_VARIANCE may be a float in (0,1] (keep that fraction of variance) or an int (fixed components)
PCA_VARIANCE = 256
PCA_MAX_COMPONENTS = 512

# Low-rank subspace options (used when CLASSIFIER in {'rank1','lowrank'})
LOWRANK_K = 4             # number of components when lowrank; rank1 forces k=1
LOWRANK_METHOD = "lda"     # 'lda' (binary, rank1 only) or 'pls' (k>=1)

# ========== Evaluation ==========
N_SPLITS = 2              # GroupKFold splits (by GROUP_COL)
N_JOBS = -1               # parallelism for supported estimators
RANDOM_STATE = 0          # RNG seed for reproducibility

# Optional comparison vs DoM/whitened DoM directions in run_geom report
# 'none' | 'dom' | 'whitened_dom' | 'both'
COMPARE_MODE = "both"
COMPARE_REG_EPS = 1e-3    # Tikhonov epsilon for whitening (Σ + eps I)^-1

# ========== Report formatting ==========
COL_WIDTHS = dict(layer=6, n=9, comps=7, acc=10, auroc=10, ap=10, f1=10)

