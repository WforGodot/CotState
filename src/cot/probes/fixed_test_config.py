from __future__ import annotations
from pathlib import Path

"""
Minimal configuration for test_fixed_dimension.

This config is intentionally separate from run_geom/probes_config and only
includes the fields needed to load activations/labels, locate saved vectors,
apply simple filters, and format outputs.
"""

# ========== Paths ==========
# Activation features NPZ (relative to this file)
ACTS_NPZ = Path("../outputs/collect/resid_post_qwen3_collect.npz")

# Labels CSV path; if None, inferred from ACTS_NPZ ("*_labels.csv" alongside)
LABELS_CSV = None

# Saved vectors location base (run subfolders under here)
VECTORS_DIR = Path("../outputs/vectors")

# Vector file to test. Set as either:
# - repo-root-relative path (recommended), e.g.,
#   "src/cot/outputs/vectors/L15__off_in_-30..49_n70_31533c49__regs_vii_max_use__Qwen-Qwen3-0.6B/L15_top1.npy"
# - or relative to VECTORS_DIR as a fallback.
# Set this to choose which fixed direction/subspace to evaluate.
VECTOR_FILE = r"src\cot\outputs\vectors\L15__off_-30_to_30__regs_vii_max_use__Qwen-Qwen3-0.6B\L15_top1.npy"

# Output directory for reports
OUT_DIR = Path("../outputs/probes")

# ========== Feature keys ==========
# Prefix for NPZ keys, e.g., acts_resid_post_layer12
HOOK_POINT_PREFIX = "acts_resid_post_layer"

# ========== Labels schema ==========
TARGET_COL = "p_value"       # e.g., "True"/"False" strings
POSITIVE_TOKEN = "true"      # positive class (case-insensitive match in TARGET_COL)
REGIME_COL = "regime"
OFFSET_COL = "offset_from_split"  # token offset relative to split

# ========== Data selection (optional) ==========
# Restrict which regimes to use (list of names) or None for all
REGIMES_TO_USE: list[str] | None = ["vii_max_use"]

# Offset filters (applied to OFFSET_COL)
FILTER_OFFSET_EQ = None      # exactly equal to this offset, or None
FILTER_OFFSET_MAX = None     # include offsets <= this value, or None
# Either an inclusive range tuple (lo, hi) with None for open bounds,
# or a list of explicit offsets to whitelist (e.g., [0,2,4,6,8])
FILTER_OFFSET_RANGE: tuple[int | None, int | None] | list[int] | None = (30,40)

# Explicit train/test split support (collector adds 'split' column)
TRAIN_SPLIT_NAME = 'train'
TEST_SPLIT_NAME = 'test'

# ========== Classifier hyperparams ==========
LOGREG_C = 1.0
N_JOBS = -1

# ========== Optional vector/layer options ==========
# If the vector file contains stacked components (d, k), optionally choose a
# single component by index (0-based). Leave as None to use all columns.
COMPONENT_INDEX: int | None = None

# If the layer cannot be inferred from VECTOR_FILE (expects 'L{num}' in name),
# set it explicitly here. Leave as None to auto-parse from filename.
LAYER_ID: int | None = None
