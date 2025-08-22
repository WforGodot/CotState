# Variables to sample from (must include 'P')
# Use plenty of symbols so disjoint buckets don't run out.
PREFERRED_VARS = list("PQRSTUVWXYZABCDEFGHIJKLMNO")

# How many items per regime
QUESTIONS_PER_REGIME = {
    "i_initial": 5,
    "ii_inconsequential": 5,
    "iii_derived": 5,
    "iv_indeterminate": 5,
    "v_output": 5,
}

# Counts per bucket (inclusive ranges)
COUNTS = {
    "initial": (1, 3),
    "used": (2, 3),
    "unused": (0, 2),
    "indeterminate": (0, 2),
}

SEED = 4

# Outputs
OUTPUT_JSONL_PATH = "proplogic_dataset.jsonl"   # structured instances
OUTPUT_CSV_PATH   = "proplogic_questions.csv"   # natural-language questions
OUTPUT_DEBUG_TXT  = "proplogic_debug.txt"       # first item per regime
MAX_TRIES_PER_ITEM = 200
