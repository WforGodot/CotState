# src/cot/collect/collect_config.py
from __future__ import annotations

# -----------------------------
# Model & runtime configuration
# -----------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B"   # start small; override later if needed
TRUST_REMOTE_CODE = True

# Device & dtype defaults; you can override via CLI if you like
DEVICE = "cuda"
DTYPE = "float16"  # "float32" for exactness; keep fp16 for speed/memory

# -----------------------------
# What to collect
# -----------------------------
# Hook site must be a TransformerLens cache name. Common, stable choices:
#   "resid_pre", "resid_mid", "resid_post", "mlp_post", "attn_out", "attn_pattern"
HOOK_POINT = "resid_post"

# Layers to collect (ints). We’ll clamp to valid range at runtime.
# These are reasonable touchpoints for a 0.6B-ish model.
LAYERS = [0, 1, 4, 8, 12]

# -----------------------------
# Sampling / selection controls
# -----------------------------
# How many rows to use from each regime (we take the first N after filtering).
REGIME_SAMPLE_COUNTS = {
    "i_initial": 5,
    "ii_inconsequential": 0,
    "iii_derived": 5,
    "iv_indeterminate": 0,   # usually you’ll keep this 0; they have p_value="Unknown"
    "v_output": 5,
}

# How many tokens to keep AFTER the split point (per regime).
# -1 means "all remaining tokens after the split".
TOKENS_AFTER_BY_REGIME = {
    "i_initial": 1,
    "ii_inconsequential": 1,
    "iii_derived": 1,
    "iv_indeterminate": 1,
    "v_output": 1,
}

# Randomness (row shuffling if you choose to add it later)
SEED = 123

# -----------------------------
# I/O (relative to repo/src/)
# -----------------------------
# Inputs produced by your datagen step:
DATAGEN_CSV_REL = "cot/outputs/datagen/proplogic_questions.csv"

# Output directory for this collection run:
COLLECT_OUT_REL = "cot/outputs/collect"

# Max debug examples
MAX_DEBUG = 10
