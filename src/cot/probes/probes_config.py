from __future__ import annotations
from pathlib import Path
import math

# --- Inputs (relative to repo/src) ---
# If LABELS_CSV is None, run_probes will try to infer it from the NPZ path.
ACTS_NPZ = Path("../outputs/collect/resid_post_qwen3_collect.npz")
LABELS_CSV = None  # e.g., Path("../outputs/collect/resid_post_qwen3_collect_batched_bucketed_labels.csv")

# The NPZ is expected to have keys like: acts_resid_post_layer0, acts_resid_post_layer1, ...
HOOK_POINT_PREFIX = "acts_resid_post_layer"  # adapt if you change hook points

# Optional: restrict which layers to train on. Use a list of integers
# (e.g., [0, 5, 12]) to train only on those layers found in the activation file.
# Set to None to train on all layers present.
LAYERS_TO_TRAIN = [21]  # type: list[int] | None

# --- Output location/tag ---
RUN_TAG = "resid_post_qwen3_linear_probes"
OUT_DIR = Path("../outputs/probes")  # results will be written under this directory

# --- Target + groups ---
TARGET_COL = "p_value"          # "True"/"False"
POSITIVE_TOKEN = "true"         # case-insensitive match in labels CSV
GROUP_COL = "example_id"        # to group all tokens from the same example
REGIME_COL = "regime"           # for optional per-regime breakdown in report
OFFSET_COL = "offset_from_split"  # optional slice reporting

# --- Model & evaluation ---
CLASSIFIER = "ridge"  # "ridge" or "logreg"
RIDGE_ALPHA = 1.0     # alpha for RidgeClassifier (L2). Ignored if CLASSIFIER="logreg"
LOGREG_C = 1.0        # C for LogisticRegression (L2). Ignored if CLASSIFIER="ridge"

# PCA: either a float in (0,1] for variance target, or an int for fixed components.
# Using an int enables fast randomized SVD. Example: 256
PCA_VARIANCE = 256  # was 0.99
# Additionally cap components to avoid overfitting / speed issues (used when PCA_VARIANCE is not int)
PCA_MAX_COMPONENTS = 512

# Cross-validation with grouping by example
N_SPLITS = 3
N_JOBS = -1          # parallelism for metrics that support it (not used heavily here)
RANDOM_STATE = 0     # for PCA randomized SVD etc.

# Optional filtering (e.g., only the first token after split)
FILTER_OFFSET_EQ = None    # exactly equal to this offset (e.g., 0), or None
FILTER_OFFSET_MAX = 20  # include offsets <= this value (e.g., 1 includes 0 and 1)
# Inclusive range filter: set to a 2-tuple/list (lo, hi). Use None to leave one side open.
# Examples: (0, 1) keeps 0 and 1; (None, 3) keeps <=3; (2, None) keeps >=2
FILTER_OFFSET_RANGE = None  # type: tuple[int | None, int | None] | None

# Pretty table formatting in TXT
COL_WIDTHS = dict(layer=6, n=9, comps=7, acc=10, auroc=10, ap=10, f1=10)

# Safety checks
MIN_EXAMPLES_PER_FOLD = 5
MIN_CLASS_COUNT = 5

# Optional random token subsample for faster experiments
# Set to an integer to sample that many tokens uniformly at random after filtering.
# Leave as None to use all available tokens.
N_TOKENS = 20000  # e.g., 100_000

def pca_n_components(d_model: int, variance: float = PCA_VARIANCE, cap: int = PCA_MAX_COMPONENTS) -> int:
    # We pass a float to PCA(n_components=variance) to keep explained variance; this cap is informative only.
    return min(cap, d_model)


CLASSIFIER = "rank1"