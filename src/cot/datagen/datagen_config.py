import os

# Variables to sample from (must include 'P')
# Use plenty of symbols so disjoint buckets don't run out.
PREFERRED_VARS = list("PQRSTUVWXYZABCDEFGHIJKLMNO")

# How many items per regime
QUESTIONS_PER_REGIME = {
    "i_initial": 1000,
    "ii_inconsequential": 0,
    "iii_derived": 1000,
    "iv_indeterminate": 0,
    "v_output": 1000,
}

# Counts per bucket (inclusive ranges)
COUNTS = {
    "initial": (1, 3),
    "used": (2, 3),
    "unused": (0, 2),
    "indeterminate": (0, 2),
}

SEED = 4

# Output folder and file names
OUTPUT_FOLDER = os.path.join("..", "outputs", "datagen")
JSONL_FILENAME = "proplogic_dataset.jsonl"   # structured instances
CSV_FILENAME   = "proplogic_questions.csv"   # natural-language questions
DEBUG_FILENAME = "proplogic_debug.txt"       # first item per regime

# Full paths
OUTPUT_JSONL_PATH = os.path.join(OUTPUT_FOLDER, JSONL_FILENAME)
OUTPUT_CSV_PATH   = os.path.join(OUTPUT_FOLDER, CSV_FILENAME)
OUTPUT_DEBUG_TXT  = os.path.join(OUTPUT_FOLDER, DEBUG_FILENAME)

MAX_TRIES_PER_ITEM = 200
