# Variables to sample from (must include 'P')
PREFERRED_VARS = list("PQRSUVWXABC")

# Regimes to generate.
# - "explicit":            track P; P changes are "Change P to True/False."
# - "invert":              track P; P changes are "Change P to not P." / "Invert P."
# - "track_other":         track some other var X≠P; final sentence still says "P is ...".
# - "invert_track_other":  same as above, but P changes use inversion phrasing.
REGIMES = ["explicit", "invert", "track_other", "invert_track_other"]

# How many items PER regime (total = len(REGIMES) * N_ITEMS_PER_REGIME)
N_ITEMS_PER_REGIME = 5000

# (Kept for backward compatibility; ignored if REGIMES is set)
N_ITEMS = 20000

# How many non-P variables to mention initially (inclusive range)
NON_P_INITIAL_VARS = (3, 3)

# How many change sentences overall (inclusive range).
NUM_CHANGES = (3, 4)

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
