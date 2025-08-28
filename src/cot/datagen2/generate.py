from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import datagen2_config as cfg


# --------------------------
# Output pathing (relative to src/)
# --------------------------
def _project_root() -> Path:
    # This file lives at src/cot/datagen/generate.py
    return Path(__file__).resolve().parents[2]

def _out_dir() -> Path:
    # New folder for this variant
    return _project_root() / "cot" / "outputs" / "datagen2"

def _out_paths() -> Tuple[Path, Path, Path]:
    odir = _out_dir()
    jsonl = odir / getattr(cfg, "JSONL_FILENAME", "proplogic_paragraphs.jsonl")
    csvf  = odir / getattr(cfg, "CSV_FILENAME", "proplogic_paragraphs.csv")
    dbg   = odir / getattr(cfg, "DEBUG_FILENAME", "proplogic_paragraphs_debug.txt")
    return jsonl, csvf, dbg


# --------------------------
# Helpers
# --------------------------
def _bool_word(b: bool) -> str:
    # Use capitalized True/False to match your existing datasets
    return "True" if b else "False"

def _sentence(text: str) -> str:
    text = text.strip()
    if not text.endswith("."):
        text += "."
    return text

def _join_initial(assignments: Dict[str, bool]) -> str:
    # "Initially, P = True, Q = False, and R = True."
    parts = [f"{v} = {_bool_word(b)}" for v, b in assignments.items()]
    if len(parts) == 1:
        body = parts[0]
    elif len(parts) == 2:
        body = f"{parts[0]} and {parts[1]}"
    else:
        body = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return _sentence("Initially, " + body)

def _choose_initial_vars(
    rng: random.Random,
    all_vars: List[str],
    required_vars: List[str],
) -> List[str]:
    """
    Ensure that all required_vars (e.g., ["P", track_var]) are present,
    then add a few other non-required vars as per NON_P_INITIAL_VARS.
    """
    k_min, k_max = cfg.NON_P_INITIAL_VARS
    k = rng.randint(k_min, k_max)

    # Dedup while preserving order
    required = list(dict.fromkeys(required_vars))
    pool = [v for v in all_vars if v not in required]

    # sample others (bounded by pool size)
    others = rng.sample(pool, k=min(k, len(pool)))
    initial_vars = required + others

    # Keep 'P' first if present for readability, shuffle the rest
    if "P" in initial_vars:
        p_idx = initial_vars.index("P")
        tail = initial_vars[:p_idx] + initial_vars[p_idx+1:]
        rng.shuffle(tail)
        initial_vars = [initial_vars[p_idx]] + tail
    else:
        rng.shuffle(initial_vars)

    return initial_vars

def _invert_phrase(rng: random.Random) -> str:
    # Clear, LLM-friendly inversion phrasings (biased toward "Change P to not P.")
    candidates = (
        ["Change P to not P."] * 3
        + ["Set P to not P."] * 2
        + ["Invert P.", "Flip P."]
    )
    return candidates[rng.randrange(len(candidates))]

def _pick_track_other(rng: random.Random, vars_pool: List[str]) -> str:
    others = [v for v in vars_pool if v != "P"]
    return rng.choice(others) if others else "Q"


# --------------------------
# Paragraph + labels builder
# --------------------------
def _build_paragraph_and_labels(
    rng: random.Random,
    vars_pool: List[str],
    p_change_style: str = "explicit",     # "explicit" or "invert"
    track_var: str = "P",                 # variable the question asks to track
    regime_name: str = "explicit",        # persisted for output/meta
) -> Dict:
    """
    Build a single paragraph that:
      - Starts with an 'Initially, ...' sentence giving truth values.
      - Contains a series of change sentences affecting variables (some hit P).
      - Ends with a restatement 'P is True/False.' (not a change; just a statement).

    Labels:
      - p_char_labels are per-character labels of P over the paragraph:
          'U' (Undefined during the 'Initially, ...' sentence),
          then 'T' or 'F' corresponding to the value of P BEFORE each change sentence.
      - We always compute 'final_p' (P at the end), and also 'final_tracked'
        for the variable in 'track_var' (which may or may not be P).

    The CSV 'question' and 'answer' use 'track_var'.
    The text always ends with "P is ..." to allow P-specific ablations.
    """

    # 1) Choose initial variables, force-include P and track_var
    initial_vars = _choose_initial_vars(rng, vars_pool, required_vars=["P", track_var])

    # Initial assignments (random)
    init_assign: Dict[str, bool] = {v: bool(rng.getrandbits(1)) for v in initial_vars}

    # Build sentences
    sentences: List[str] = []

    # 1) Initially sentence
    initially = _join_initial(init_assign)
    sentences.append(initially)

    # 2) Change sentences plan
    n_changes = rng.randint(cfg.NUM_CHANGES[0], cfg.NUM_CHANGES[1])

    # Decide roughly how many of these will affect P
    target_p_changes = max(
        cfg.MIN_P_CHANGES,
        min(n_changes, int(round(n_changes * float(cfg.P_CHANGE_FRAC))))
    )

    # Construct a bag of targets for change sentences
    bag: List[str] = []
    bag.extend(["P"] * target_p_changes)  # ensure at least some P changes
    while len(bag) < n_changes:
        bag.append(rng.choice(vars_pool))
    rng.shuffle(bag)

    # Track current values for coherence (not strictly required for correctness)
    cur_val: Dict[str, bool] = dict(init_assign)

    # Explicit ops for robust simulation/labeling
    # Each item: {"var": str, "op": "set"|"invert", "value": Optional[bool]}
    change_ops: List[Dict] = []

    # Sentences 2..K: either explicit set or invert for P (depending on p_change_style)
    for v in bag:
        if v == "P" and p_change_style == "invert":
            sentences.append(_invert_phrase(rng))
            change_ops.append({"var": "P", "op": "invert", "value": None})
            cur_val["P"] = (not cur_val.get("P", False))
        else:
            new_val = bool(rng.getrandbits(1))
            cur_val[v] = new_val
            sentences.append(_sentence(f"Change {v} to {_bool_word(new_val)}"))
            change_ops.append({"var": v, "op": "set", "value": new_val})

    # 3) Final restatement sentence: just restate P's *current* value after all changes
    # Simulate P and tracked var over the change_ops timeline
    p_current = init_assign["P"]
    tracked_current = init_assign.get(track_var, False)

    for op in change_ops:
        if op["var"] == "P":
            if op["op"] == "set":
                p_current = bool(op["value"])
            elif op["op"] == "invert":
                p_current = (not p_current)
            else:
                raise ValueError(f"Unknown op: {op['op']}")
        if op["var"] == track_var:
            if op["op"] == "set":
                tracked_current = bool(op["value"])
            elif op["op"] == "invert":
                tracked_current = (not tracked_current)

    final_sentence = _sentence(f"P is {_bool_word(p_current)}")
    sentences.append(final_sentence)

    # 4) Build the full paragraph and labels (labels are for P)
    paragraph_parts: List[str] = []
    labels: List[str] = []

    def append_labeled(text: str, label_char: str):
        paragraph_parts.append(text)
        labels.extend([label_char] * len(text))

    # Initially: P is undefined (U)
    append_labeled(sentences[0], "U")
    p_state: Optional[bool] = init_assign["P"]  # becomes defined after sentence 0 ends

    # Add a space between sentences (space is also labeled)
    def add_space():
        append_labeled(" ", "U" if p_state is None else ("T" if p_state else "F"))

    # Change sentences (1..n_changes): label with current P BEFORE applying change
    for idx, op in enumerate(change_ops, start=1):
        add_space()
        pre_label = "U" if p_state is None else ("T" if p_state else "F")
        append_labeled(sentences[idx], pre_label)

        # Apply change after the sentence if it targets P
        if op["var"] == "P":
            if op["op"] == "set":
                p_state = bool(op["value"])
            elif op["op"] == "invert":
                p_state = (not p_state)

    # Final restatement sentence: label with current p_state (no change occurs here)
    add_space()
    final_label = "U" if p_state is None else ("T" if p_state else "F")
    append_labeled(sentences[-1], final_label)

    paragraph = "".join(paragraph_parts)
    p_char_labels = "".join(labels)
    assert len(paragraph) == len(p_char_labels), "Label string must align with paragraph length."

    # Compute split point: right before the final 'True/False' in the last sentence 'P is X.'
    p_value_str = _bool_word(p_state)
    needle = f"P is {p_value_str}"
    # Use rfind to ensure we use the final restatement occurrence
    pos = paragraph.rfind(needle)
    if pos >= 0:
        p_char_index = pos + len("P is ")
    else:
        p_char_index = None

    return {
        "paragraph": paragraph,
        "p_char_labels": p_char_labels,      # per-character labels for P only
        "final_p": p_state,                  # bool, P at end
        "p_value": p_value_str,              # 'True'/'False' for P at end
        "p_char_index": p_char_index,        # int or None, split before final True/False
        "initial_assignments": init_assign,
        "bag_of_changes": bag,               # variables per change sentence in order
        "change_ops": change_ops,            # explicit ops used
        "sentences": sentences,              # for debugging/inspection
        "regime": regime_name,               # persisted regime name
        "track_var": track_var,              # which var the question asks you to track
        "final_tracked": tracked_current,    # bool, track_var at end
    }


# --------------------------
# Writers
# --------------------------
def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def write_jsonl(path: str, items: List[Dict]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex) + "\n")

def write_csv(path: str, items: List[Dict]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Columns: id, regime, question (about track_var), answer (track_var final),
        # cot (paragraph), p_char_labels (labels for P), p_char_index (split before final True/False for P), p_value
        writer.writerow(["id", "regime", "question", "answer", "cot", "p_char_labels", "p_char_index", "p_value"])
        for idx, ex in enumerate(items):
            tracked = ex.get("track_var", "P")
            question = f"Track {tracked} through the paragraph. What is {tracked} at the end? Answer 'True' or 'False'."
            answer = "True" if ex.get("final_tracked", False) else "False"
            writer.writerow([
                idx,
                ex.get("regime", "explicit"),
                question,
                answer,
                ex["paragraph"],
                ex["p_char_labels"],
                (ex.get("p_char_index", "") if ex.get("p_char_index", None) is not None else ""),
                ex.get("p_value", ""),
            ])

def write_debug_txt(path: str, items: List[Dict], k: int = 5) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("First few examples (paragraph + aligned labels):\n\n")
        for i, ex in enumerate(items[:k]):
            para = ex["paragraph"]
            labs = ex["p_char_labels"]
            f.write(f"=== example {i} ({ex.get('regime','explicit')}, track_var={ex.get('track_var','P')}) ===\n")
            # Insert a visible split marker before the final True/False for P if available
            pci = ex.get("p_char_index", None)
            if pci is not None and isinstance(pci, int) and 0 <= pci <= len(para):
                marked = para[:pci] + "|| " + para[pci:]
            else:
                marked = para
            f.write(marked + "\n")
            f.write(labs + "\n")
            f.write(f"final P: {'True' if ex['final_p'] else 'False'} | "
                    f"final {ex.get('track_var','P')}: {'True' if ex.get('final_tracked', False) else 'False'}\n")
            f.write("sentences:\n")
            for s in ex["sentences"]:
                f.write(f" - {s}\n")
            f.write("\n")


# --------------------------
# Main generation
# --------------------------
def generate_all() -> List[Dict]:
    rng = random.Random(cfg.SEED)
    vars_pool = list(dict.fromkeys(cfg.PREFERRED_VARS))  # unique and ordered
    if "P" not in vars_pool:
        vars_pool = ["P"] + vars_pool

    regimes = getattr(cfg, "REGIMES", ["explicit"])
    n_per = getattr(cfg, "N_ITEMS_PER_REGIME", getattr(cfg, "N_ITEMS", 1000))

    out: List[Dict] = []
    for style in regimes:
        for _ in range(n_per):
            if style == "explicit":
                ex = _build_paragraph_and_labels(
                    rng, vars_pool,
                    p_change_style="explicit",
                    track_var="P",
                    regime_name=style,
                )
            elif style == "invert":
                ex = _build_paragraph_and_labels(
                    rng, vars_pool,
                    p_change_style="invert",
                    track_var="P",
                    regime_name=style,
                )
            elif style == "track_other":
                ex = _build_paragraph_and_labels(
                    rng, vars_pool,
                    p_change_style="explicit",
                    track_var=_pick_track_other(rng, vars_pool),
                    regime_name=style,
                )
            elif style == "invert_track_other":
                ex = _build_paragraph_and_labels(
                    rng, vars_pool,
                    p_change_style="invert",
                    track_var=_pick_track_other(rng, vars_pool),
                    regime_name=style,
                )
            else:
                raise ValueError(f"Unknown regime '{style}'")
            out.append(ex)
    return out


if __name__ == "__main__":
    dataset = generate_all()
    print(f"Generated {len(dataset)} items.")
    jsonl_path, csv_path, debug_path = _out_paths()
    _ensure_parent(jsonl_path)
    if jsonl_path:
        write_jsonl(str(jsonl_path), dataset)
        print(" -", jsonl_path)
    if csv_path:
        write_csv(str(csv_path), dataset)
        print(" -", csv_path)
    if debug_path:
        write_debug_txt(str(debug_path), dataset)
        print(" -", debug_path)
