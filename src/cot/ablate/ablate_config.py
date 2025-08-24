from __future__ import annotations
from pathlib import Path

"""
Configuration for ablation runs.

Edit these to point to your dataset and control which layer/tokens to
modify and how the report is produced.
"""

# ========== Paths ==========
# Relative to the project root (same convention as collect/probes)
DATAGEN_CSV_REL = Path("cot/outputs/datagen/proplogic_questions.csv")
OUT_DIR_REL = Path("cot/outputs/ablate")

# Vector folder containing per-layer direction files (e.g., L15_top1.npy, L14_top1.npy).
# This path is resolved relative to the project 'src' root (see run_ablate).
# Example (matches src/cot/outputs/vectors structure):
#   "cot/outputs/vectors/L15__off_-30_to_30__regs_vii_max_use__Qwen-Qwen3-0.6B"
VECTORS_DIR_REL = "cot/outputs/vectors/all_layers"

# ========== Model ==========
MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cuda"
DTYPE = "float16"  # one of {"float32","float16","bfloat16"}
TRUST_REMOTE_CODE = True

# ========== Batching / performance ==========
# Process examples in padded batches grouped by similar lengths
BATCH_SIZE = 16
LENGTH_BUCKET_SIZE = 64
EMPTY_CACHE_EVERY_N_BATCHES = 0  # set >0 to periodically free CUDA cache

# ========== Hook / layer ==========
# Hook point name without the "hook_" prefix (e.g., "resid_pre", "resid_post", "mlp_out").
HOOK_POINT = "resid_pre"
# Layers to modify (0-based). If None, falls back to single LAYER.
LAYERS: list[int] | None = [10]
# Backward compat: if LAYERS is None, use this single layer
LAYER = 15

# ========== Selection ==========
# Regimes to evaluate from the datagen CSV (e.g., ["v_output"]).
REGIMES_TO_USE = ["vii_max_use"]

# Number of prior tokens (before the split) to modify along the direction.
PRIOR_TOKENS = 1000

# Ablation mode: 'reflect' (x - 2(x·v)v), 'project_out' (x - (x·v)v), 'push' (x + gamma*v)
ABLATED_MODE = 'reflect'  # 'reflect' | 'project_out' | 'push'
# Step size for 'push' mode (in units of the unit-norm v)
PUSH_GAMMA = 1.0

# Number of examples to sample; None for all available after filtering.
N_SAMPLES: int | None = 300

# Strings used to pick target token ids for measuring logit shifts.
# Provide multiple variants; we use the first token id of each string.
# Duplicates are deduplicated. Backward-compat: TRUE_STR/FALSE_STR also supported.
TRUE_STRINGS = [" True", " true", "True", "true", "1", "True"]
FALSE_STRINGS = [" False", " false", "False", "false", "0"]
# TRUE_STR = "1"
# FALSE_STR = "0"

# Random seed for sampling
RANDOM_STATE = 0
