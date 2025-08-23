from __future__ import annotations

# -----------------------------
# Model & runtime configuration
# -----------------------------
MODEL_NAME = "Qwen/Qwen3-4B"
TRUST_REMOTE_CODE = True

DEVICE = "cuda"
DTYPE = "float16"

# Batch + bucketing
BATCH_SIZE = 16
LENGTH_BUCKET_SIZE = 64  # group examples by similar seq len to reduce pad waste

# Periodically release cached CUDA blocks (0 disables)
EMPTY_CACHE_EVERY_N_BATCHES = 8

# -----------------------------
# What to collect
# -----------------------------
HOOK_POINT = "resid_post"       # e.g., resid_pre|resid_mid|resid_post|mlp_post|attn_out|attn_pattern
LAYERS = list(range(9, 27))

# -----------------------------
# Sampling / selection controls
# -----------------------------
REGIME_SAMPLE_COUNTS = {
    "i_initial": 500,
    "ii_inconsequential": 0,
    "iii_derived": 500,
    "iv_indeterminate": 0,
    "v_output": 500,
}

# -1 means "all tokens after the split"
TOKENS_AFTER_BY_REGIME = {
    "i_initial": -1,
    "ii_inconsequential": -1,
    "iii_derived": -1,
    "iv_indeterminate": -1,
    "v_output": -1,
}

SEED = 123

# -----------------------------
# I/O (relative to repo/src/)
# -----------------------------
DATAGEN_CSV_REL = "cot/outputs/datagen/proplogic_questions.csv"
COLLECT_OUT_REL = "cot/outputs/collect"

MAX_DEBUG = 10
