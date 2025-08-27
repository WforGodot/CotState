# src/cot/datagen/datagen2_config.py

# Variables to sample from (must include 'P')
PREFERRED_VARS = list("PQRSUVWXABC")

# Total number of items to generate
N_ITEMS = 5000

# How many non-P variables to mention initially (inclusive range)
NON_P_INITIAL_VARS = (3, 3)

# How many change sentences overall (inclusive range).
# A fraction of these will target P (see P_CHANGE_FRAC).
NUM_CHANGES = (3, 8)

# Fraction of change sentences that affect P (approximate)
P_CHANGE_FRAC = 0.5

# Require at least this many P changes (to ensure we actually exercise flips)
MIN_P_CHANGES = 1

# Output file names (written under src/cot/outputs/datagen2)
JSONL_FILENAME = "proplogic_paragraphs.jsonl"
CSV_FILENAME = "proplogic_paragraphs.csv"
DEBUG_FILENAME = "proplogic_paragraphs_debug.txt"

# Random seed
SEED = 7
