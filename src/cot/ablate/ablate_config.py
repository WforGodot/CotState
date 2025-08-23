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

# Path to a learned direction (1D .npy). Relative paths are resolved
# relative to this file; you can also use an absolute path.
VECTOR_PATH = Path("../outputs/vectors/L12__off_-10_to_0__regs_v_output__Qwen-Qwen3-0.6B__2/L12_split1_top1.npy")

# ========== Model ==========
MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cuda"
DTYPE = "float16"  # one of {"float32","float16","bfloat16"}
TRUST_REMOTE_CODE = True

# ========== Hook / layer ==========
# Hook point name without the "hook_" prefix (e.g., "resid_post", "mlp_out").
HOOK_POINT = "resid_post"
# Single layer index to modify (0-based)
LAYER = 12

# ========== Selection ==========
# Regimes to evaluate from the datagen CSV (e.g., ["v_output"]).
REGIMES_TO_USE = ["v_output"]

# Number of prior tokens (before the split) to invert along the direction.
PRIOR_TOKENS = 10

# Number of examples to sample; None for all available after filtering.
N_SAMPLES: int | None = 500

# Strings used to pick target token ids for measuring logit shifts.
# If the tokenizer produces multiple tokens, we will use the first token id
# and note this in the report.
TRUE_STR = " True"
FALSE_STR = " False"

# Random seed for sampling
RANDOM_STATE = 0

