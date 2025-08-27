#collect_config.py

from __future__ import annotations

# -----------------------------
# Model & runtime configuration
# -----------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B"
TRUST_REMOTE_CODE = True

DEVICE = "cuda"
DTYPE = "float16"

# Batch + bucketing
BATCH_SIZE = 32
LENGTH_BUCKET_SIZE = 64  # group examples by similar seq len to reduce pad waste

# Periodically release cached CUDA blocks (0 disables)
EMPTY_CACHE_EVERY_N_BATCHES = 8

# -----------------------------
# What to collect
# -----------------------------
HOOK_POINT = "resid_post"       # e.g., resid_pre|resid_mid|resid_post|mlp_post|attn_out|attn_pattern
LAYERS = list(range(10, 20, 2))  # which layers to collect (list of ints)

# -----------------------------
# Sampling / selection controls
# -----------------------------
REGIME_SAMPLE_COUNTS = {
    "i_initial": 0,
    "ii_inconsequential": 0,
    "iii_derived": 0,
    "iv_indeterminate": 0,
    "v_output": 0,
    "vi_single_use": 5000,
    "vii_max_use": 5000,
}

# For each regime, choose how many tokens to keep before and after the split.
# A value of -1 means "all tokens on that side".
# Example: (4, 5) keeps up to 4 tokens before split and up to 5 after.
#          (1, -1) keeps 1 token before and all tokens after.
#          (-1, 0) keeps all tokens before and none after.
TOKENS_AROUND_BY_REGIME = {
    "i_initial": (-40, 40),
    "ii_inconsequential": (-40, 40),
    "iii_derived": (0, 40),
    "iv_indeterminate": (-40, 40),
    "v_output": (0, 20),
    "vi_single_use": (-20, 40),
    "vii_max_use": (-20, 40),
}

SEED = 124

# Fraction of candidate tokens to sample uniformly at random (per-token)
# Set to 1.0 to keep all; None to disable sampling. Example: 0.5
TOKEN_SAMPLE_FRACTION = 0.1

# Train/test split across examples (entire CoT assigned to one split)
TRAIN_FRACTION = 0.7   # remainder goes to test
SPLIT_SEED = 199

# -----------------------------
# I/O (relative to repo/src/)
# -----------------------------
DATAGEN_CSV_REL = "cot/outputs/datagen/proplogic_questions.csv"
COLLECT_OUT_REL = "cot/outputs/collect"

MAX_DEBUG = 10
