# src/cot/collect/collect_paragraph_config.py

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
LAYERS = list(range(0, 50))     # which layers to collect (list of ints)

# -----------------------------
# Sampling / selection controls
# -----------------------------
# Train/test split across examples (entire example assigned to one split)
TRAIN_FRACTION = 0.7
SPLIT_SEED = 100

# Fraction of candidate tokens to sample uniformly at random (per-token)
# Set to 1.0 to keep all; None or 1.0 to keep all; example: 0.1 keeps ~10%
TOKEN_SAMPLE_FRACTION = 0.1

# Limit how many examples to read per regime (useful if your CSV is huge).
# The datagen2 writer uses a single regime "track_p".
REGIME_SAMPLE_COUNTS = {
    "track_p": 8000,  # or fewer if you want a subset
}

SEED = 124  # for any per-run randomness (not tokenizer-related)

# -----------------------------
# I/O (relative to repo/src/)
# -----------------------------
# Point to the datagen2 CSV you generated
DATAGEN_CSV_REL = "cot/outputs/datagen2/proplogic_paragraphs.csv"
# Where to write features/labels/debug
COLLECT_OUT_REL = "cot/outputs/collect2"

MAX_DEBUG = 10
