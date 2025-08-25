import os

# Variables to sample from (must include 'P')
# Use plenty of symbols so disjoint buckets don't run out.
PREFERRED_VARS = list("PQRSUVWXYZABCD")

# How many items per regime
QUESTIONS_PER_REGIME = {
    "i_initial": 0,
    "ii_inconsequential": 10000,
    "iii_derived": 10000,
    "iv_indeterminate": 10000,
    "v_output": 10000,
    "vi_single_use": 10000,
    "vii_max_use": 10000,
}

# Counts per bucket (inclusive ranges)
COUNTS = {
    "initial": (2, 2),
    "used": (2, 3),
    "unused": (0, 1),
    "indeterminate": (0, 1),
}

SEED = 4

# Output file names (generator saves under src/cot/outputs/datagen)
JSONL_FILENAME = "proplogic_dataset.jsonl"   # structured instances
CSV_FILENAME   = "proplogic_questions.csv"   # natural-language questions
DEBUG_FILENAME = "proplogic_debug.txt"       # first item per regime

MAX_TRIES_PER_ITEM = 100

VII_MAX_DECLARED_USED = 3
VII_MIN_RULE_USAGE_FRAC = 0.7
VII_USED_ON_PATH_FRAC = 0.5

