import os

# Variables to sample from (must include 'P')
# Use plenty of symbols so disjoint buckets don't run out.
PREFERRED_VARS = list("PQRSTUVWXYZABCDEFGHIJKLMNO")

# How many items per regime
QUESTIONS_PER_REGIME = {
    "i_initial": 0,
    "ii_inconsequential": 0,
    "iii_derived": 0,
    "iv_indeterminate": 0,
    "v_output": 3000,
}

# Counts per bucket (inclusive ranges)
COUNTS = {
    "initial": (1, 3),
    "used": (2, 3),
    "unused": (0, 2),
    "indeterminate": (0, 2),
}

SEED = 4

# Output file names (generator saves under src/cot/outputs/datagen)
JSONL_FILENAME = "proplogic_dataset.jsonl"   # structured instances
CSV_FILENAME   = "proplogic_questions.csv"   # natural-language questions
DEBUG_FILENAME = "proplogic_debug.txt"       # first item per regime

MAX_TRIES_PER_ITEM = 200
