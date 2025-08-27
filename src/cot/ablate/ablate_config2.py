from __future__ import annotations
from pathlib import Path

"""
Configuration for ablation runs.

Edit these to point to your dataset and control which layer/tokens to
modify and how the report is produced.
"""

# ========== Paths ==========
# Relative to the project root (same convention as collect/probes)
DATAGEN_CSV_REL = Path("cot/outputs/datagen2/proplogic_paragraphs.csv")
OUT_DIR_REL = Path("cot/outputs/ablate2")

# Vector folder containing per-layer direction files (e.g., L15_top1.npy, L14_top1.npy).
# This path is resolved relative to the project 'src' root (see run_ablate).
# Example (matches src/cot/outputs/vectors structure):
#   "cot/outputs/vectors/L15__off_-30_to_30__regs_vii_max_use__Qwen-Qwen3-0.6B"
VECTORS_DIR_REL = "cot/outputs/vectors/flip"

# ========== Model ==========
MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cuda"
DTYPE = "float16"  # one of {"float32","float16","bfloat16"}
TRUST_REMOTE_CODE = True

# ========== Batching / performance ==========
# Process examples in padded batches grouped by similar lengths
BATCH_SIZE = 12
LENGTH_BUCKET_SIZE = 64
EMPTY_CACHE_EVERY_N_BATCHES = 0  # set >0 to periodically free CUDA cache

# ========== Hook / layer ==========
# Hook point name without the "hook_" prefix (e.g., "resid_pre", "resid_post", "mlp_out").
HOOK_POINT = "resid_post"
# Layers to modify (0-based). Supports either:
# - list[int]  (single experiment)
# - list[list[int]] (multiple experiments; one inner list per experiment)
# If None, falls back to single LAYER.
LAYERS = [[x] for x in range(10,20,2)]
# Backward compat: if LAYERS is None, use this single layer
LAYER = 10

# ========== Selection ==========
# Regimes to evaluate from the datagen CSV (e.g., ["v_output"]).
REGIMES_TO_USE = ["track_p"]

# Number of prior tokens (before the split) to modify along the direction.
PRIOR_TOKENS = 30

# Ablation mode: 'reflect' (x - 2(x·v)v), 'project_out' (x - (x·v)v), 'push' (x + gamma*v)
ABLATED_MODE = 'reflect'  # 'reflect' | 'project_out' | 'push'
# Step size for 'push' mode (in units of the unit-norm v)
PUSH_GAMMA = 1.0

# ========== Control ablation (random direction) ==========
# If True, also run a control ablation using a random unit vector per layer
# with the same hook/mask/mode settings. Results are reported alongside.
CONTROL_RANDOM_DIR = True
# Optional seed for reproducible random control vectors; if None, uses RANDOM_STATE
CONTROL_SEED: int | None = None

# ========== Selective masking (Top-K by attn × |h·v|) ==========
# If > 0, select only this many tokens among the prior window to modify,
# per layer and example, based on attention weight (from the same layer,
# query at last prefix position) multiplied by |h·v| at the hook point.
# If 0, disable and use the full prior window.
TOPK_BY_ATTNV = None
# Aggregation across heads for attention weights: 'mean' | 'max' | 'sum'
ATTNV_HEAD_AGG = 'mean'

# Number of examples to sample; None for all available after filtering.
N_SAMPLES: int | None = 300

# Absolute-margin threshold for reporting a separate low-margin subset
# (examples with |base margin| < LOW_MARGIN_ABS_THRESH before ablation)
LOW_MARGIN_ABS_THRESH = 0.4

# Strings used to pick target token ids for measuring logit shifts.
# Provide multiple variants; we use the first token id of each string.
# Duplicates are deduplicated. Backward-compat: TRUE_STR/FALSE_STR also supported.
TRUE_STRINGS = [" True", " true", "True", "true", "1", "True"]
FALSE_STRINGS = [" False", " false", "False", "false", "0"]
# TRUE_STR = "1"
# FALSE_STR = "0"

# Extra logging: number of top non-(True/False) tokens to record per example
# in per_example.csv for baseline/ablated (and control when enabled).
OTHER_TOPK = 1

# Random seed for sampling
RANDOM_STATE = 602

ATTN_BLOCK_PREFIX_TOKENS: int | None = -20
ATTN_BLOCK_QUERY_LAST_M: int | None = 3

TOPK_EARLY = 0  # 0 = block all first N (current behavior). Try 1,3,5.

# Also block within-tail relays: for tail queries q, zero attention to keys k in [q-R, q-1].
# 0 = off. Try 1 or 2.
BLOCK_TAIL_RELAY_HOPS = 1
