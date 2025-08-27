# src/cot/datagen/generate.py
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
    # This file lives at src/cot/datagen/generate_datagen2.py
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
# Paragraph + labels builder
# --------------------------
def _choose_initial_vars(rng: random.Random, all_vars: List[str]) -> List[str]:
    # Ensure P is included, then sample a few others
    pool_wo_p = [v for v in all_vars if v != "P"]
    k_min, k_max = cfg.NON_P_INITIAL_VARS
    k = rng.randint(k_min, k_max)
    others = rng.sample(pool_wo_p, k=min(k, len(pool_wo_p)))
    initial_vars = ["P"] + others
    # Keep order but shuffle non-P for variety
    rng.shuffle(initial_vars[1:])
    return initial_vars

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

def _build_paragraph_and_labels(
    rng: random.Random,
    vars_pool: List[str],
) -> Dict:
    """
    Build a single paragraph that:
      - Starts with an 'Initially, ...' sentence giving truth values.
      - Contains a series of 'Change X to True/False.' sentences (some hit P).
      - Ends with a restatement 'P is True/False.' (not a change; just a statement).

    Labeling rule (simple & deterministic):
      - For any sentence that sets or changes P, the new value takes effect
        AFTER the sentence's final period.
      - Characters of the 'Initially, ...' sentence are labeled U (Undefined for P).
      - All other characters are labeled with the current value of P
        BEFORE the sentence's change is applied (T or F).
    """
    # Choose initial variables
    initial_vars = _choose_initial_vars(rng, vars_pool)

    # Initial assignments (random)
    init_assign: Dict[str, bool] = {v: bool(rng.getrandbits(1)) for v in initial_vars}
    # Make P's initial value explicit (may be True/False)
    p_state: Optional[bool] = None  # undefined during the first sentence

    # Build sentences (strings) and per-sentence metadata of whether it changes P
    sentences: List[str] = []

    # 1) Initially sentence
    initially = _join_initial(init_assign)
    sentences.append(initially)

    # 2) Change sentences
    n_changes = rng.randint(cfg.NUM_CHANGES[0], cfg.NUM_CHANGES[1])

    # Decide roughly how many of these will affect P
    target_p_changes = max(
        cfg.MIN_P_CHANGES,
        min(n_changes, int(round(n_changes * float(cfg.P_CHANGE_FRAC))))
    )

    # Construct a bag of targets for change sentences
    other_vars = [v for v in vars_pool if v != "P"]
    bag: List[str] = []

    # Ensure at least MIN_P_CHANGES on P
    bag.extend(["P"] * target_p_changes)
    # Fill the rest with other variables (including possibly more P)
    while len(bag) < n_changes:
        bag.append(rng.choice(vars_pool))
    rng.shuffle(bag)

    # Track the evolving current values of variables (for coherent changes)
    cur_val: Dict[str, bool] = dict(init_assign)

    # Sentences 2..K: "Change X to True/False."
    for v in bag:
        # Pick a target truth value; allow repeats (sometimes no-op)
        new_val = bool(rng.getrandbits(1))
        if v in cur_val:
            cur_val[v] = new_val
        else:
            # If v wasn't in initially, implicitly define it by first change
            cur_val[v] = new_val
        sentences.append(_sentence(f"Change {v} to {_bool_word(new_val)}"))

    # 3) Final restatement sentence: just restate P's *current* value
    # Apply the initial P at the end of sentence 0,
    # then each change to P at the end of its sentence.
    # To know the final value now, simulate that timeline.
    p_current = init_assign["P"]  # after sentence 0 ends
    for idx, v in enumerate(bag, start=1):
        if v == "P":
            # change takes effect after sentence idx ends
            # The sentence text is already fixed; nothing to do here yet
            pass

    # We need the actual final value to restate. Replay cleanly:
    p_current = init_assign["P"]
    for v in bag:
        if v == "P":
            p_current = cur_val["P"]  # cur_val already equals the latest assigned value

    final_sentence = _sentence(f"P is {_bool_word(p_current)}")
    sentences.append(final_sentence)

    # Build the full paragraph and labels
    # Labels are per-character: 'U' (undefined), 'T', or 'F'
    paragraph_parts: List[str] = []
    labels: List[str] = []

    # Helper to append text with label
    def append_labeled(text: str, label_char: str):
        paragraph_parts.append(text)
        labels.extend([label_char] * len(text))

    # Process sentence 0 (Initially,...): label all chars as Undefined (U)
    append_labeled(sentences[0], "U")

    # After sentence 0's period, P becomes defined
    p_state = init_assign["P"]

    # Add a space between sentences (space is also labeled)
    def add_space():
        append_labeled(" ", "U" if p_state is None else ("T" if p_state else "F"))

    # Process change sentences (1..n_changes).
    # For each sentence, label its characters with the current p_state BEFORE applying any change in that sentence.
    for idx, v in enumerate(bag, start=1):
        add_space()
        label_char = "U" if p_state is None else ("T" if p_state else "F")
        append_labeled(sentences[idx], label_char)
        # Apply change after the sentence if it targets P
        if v == "P":
            # Determine the value specified in this sentence we just wrote:
            # The sentence is exactly "Change P to True/False."
            if sentences[idx].endswith("True."):
                p_state = True
            elif sentences[idx].endswith("False."):
                p_state = False

    # Final restatement sentence: no state change, just text labeled with current p_state
    add_space()
    final_label = "U" if p_state is None else ("T" if p_state else "F")
    append_labeled(sentences[-1], final_label)

    paragraph = "".join(paragraph_parts)
    p_char_labels = "".join(labels)

    assert len(paragraph) == len(p_char_labels), "Label string must align with paragraph length."

    # Return record
    return {
        "paragraph": paragraph,
        "p_char_labels": p_char_labels,
        "final_p": p_state,  # bool
        "initial_assignments": init_assign,
        "bag_of_changes": bag,            # variables per change sentence in order
        "sentences": sentences,           # for debugging/inspection
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
        # Close to your prior format; key differences:
        #  - 'cot' column still exists (holds the paragraph)
        #  - Replace p_char_index / p_value with p_char_labels
        writer.writerow(["id", "regime", "question", "answer", "cot", "p_char_labels"])
        for idx, ex in enumerate(items):
            question = "Track P through the paragraph. What is P at the end? Answer 'True' or 'False'."
            answer = "True" if ex["final_p"] else "False"
            writer.writerow([
                idx,
                "track_p",
                question,
                answer,
                ex["paragraph"],
                ex["p_char_labels"],
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
            f.write(f"=== example {i} ===\n")
            f.write(para + "\n")
            f.write(labs + "\n")
            f.write(f"final P: {'True' if ex['final_p'] else 'False'}\n")
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
    out: List[Dict] = []
    for _ in range(cfg.N_ITEMS):
        ex = _build_paragraph_and_labels(rng, vars_pool)
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
