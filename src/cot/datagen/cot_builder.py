# cot_builder.py

from typing import Dict, List, Tuple, Optional
import random, hashlib, json

# ---------- Helpers for NL rule phrasing ----------

def _fmt_lit_words(sign: str, var: str) -> str:
    """Return a literal in words, e.g., '-' -> 'not X', '+' -> 'X'."""
    return f"not {var}" if sign == "-" else var

# Add (or keep) this helper near the top:
def _lit_truth_phrase(sign: str, var: str, want_literal_truth: bool) -> str:
    """
    Render a literal as 'VAR is True/False', absorbing negation into the truth word.
    want_literal_truth=True  -> phrase the condition under which the literal is True
    want_literal_truth=False -> phrase the condition under which the literal is False
    """
    if sign == "+":
        return f"{var} is {'True' if want_literal_truth else 'False'}"
    else:  # '-'
        return f"{var} is {'False' if want_literal_truth else 'True'}"
    
def _describe_rule_natural(rule: Dict, coin_true: bool, add_only: bool = False) -> str:
    """
    UNARY: keep 'matches not X' (or 'matches X') exactly as before.
    AND/OR: choose true-oriented vs false-oriented templates via coin_true.
            If add_only=True, insert 'only' before 'true/false' (e.g., 'is only true when ...').
    """
    head = rule.get("head")
    op = rule.get("op")
    ins = rule.get("inputs", [])
    only = "only " if add_only else ""

    if op == "UNARY":
        s, v = ins[0]
        lit = _fmt_lit_words(s, v)
        if head is None:
            return f"a variable matches {lit}"
        return f"{head} matches {lit}"

    if op in ("AND", "OR"):
        if len(ins) < 2:
            ins = ins + ins[:1]
        (a_s, a_v), (b_s, b_v) = ins[:2]

        if op == "AND":
            if coin_true:
                a = _lit_truth_phrase(a_s, a_v, True)
                b = _lit_truth_phrase(b_s, b_v, True)
                return f"{head} is {only}true when both {a} and {b}"
            else:
                a = _lit_truth_phrase(a_s, a_v, False)
                b = _lit_truth_phrase(b_s, b_v, False)
                return f"{head} is {only}false if either {a} or {b}"
        else:  # OR
            if coin_true:
                a = _lit_truth_phrase(a_s, a_v, True)
                b = _lit_truth_phrase(b_s, b_v, True)
                return f"{head} is {only}true if either {a} or {b}"
            else:
                a = _lit_truth_phrase(a_s, a_v, False)
                b = _lit_truth_phrase(b_s, b_v, False)
                return f"{head} is {only}false when both {a} and {b}"

    if op == "CLAUSE" and head is None:
        parts = [_lit_truth_phrase(s, v, True) for (s, v) in ins]
        if len(parts) == 1:
            joined = parts[0]
        elif len(parts) == 2:
            joined = f"{parts[0]} or {parts[1]}"
        else:
            joined = ", ".join(parts[:-1]) + f", or {parts[-1]}"
        return f"at least one of {joined} must be true"

    return "we apply a rule relating these variables"
# ---------- Standardized boolean/text helpers ----------

def _TF(b: bool) -> str:
    """Canonical boolean string 'True' or 'False'."""
    return "True" if b else "False"

def _join_known_facts(used_vals: Dict[str, bool]) -> str:
    """
    Standardized facts for embedding in sentences.
    Emit as VAR = True/False (with spaces), comma-separated with Oxford 'and'.
    """
    if not used_vals:
        return ""
    bits = [f"{k} = {'True' if v else 'False'}" for k, v in sorted(used_vals.items())]
    if len(bits) == 1:
        return bits[0]
    return ", ".join(bits[:-1]) + f", and {bits[-1]}"

def _first_set_event(status_timeline: Dict[str, List[Tuple[int, bool]]], var: str) -> Optional[Tuple[int, bool]]:
    ev = status_timeline.get(var, [])
    return ev[0] if ev else None

# ---------- Main builder (tokenizer-friendly) ----------

def build_cot_and_annotation(inst: Dict, p_var: str = "P") -> Dict:
    """
    Build a standardized CoT with:
      - All assignments formatted 'VAR = True/False' (with spaces).
      - REMOVE the 'At the start: ...' line entirely.
      - Split marker at earliest of:
          (a) first step where P appears in 'Given that ...', or
          (b) first step where P is the learned head.
      - For regime iv: keep pseudo-label + mid-step anchor behavior.
      - Always append: 'FINAL ANSWER: OUT = True/False.'
      - Binary rule phrasing uses 50/50 true-/false-oriented templates
        and 'X is False' instead of 'not X'.
    """
    initials = inst["initial_assignments"]
    rules = inst["rules"]
    steps = inst["steps"]
    status = inst["status_timeline"]
    out_var = inst["output_var"]
    out_val = inst["output_value"]
    regime = inst.get("regime", "")

    # Reproducible "random" choices per instance
    seed_material = {
        "initials": initials,
        "rules": rules,
        "steps": steps,
        "out_var": out_var,
        "out_val": out_val,
        "regime": regime,
    }
    seed_hex = hashlib.sha256(json.dumps(seed_material, sort_keys=True).encode()).hexdigest()
    rng = random.Random(int(seed_hex, 16) % (2**32))


    # Determine P timing/value the usual way
    p_event = _first_set_event(status, p_var)
    p_value = "Unknown" if p_event is None else ("True" if p_event[1] else "False")

    # --- Regime IV override: label anyway + optional mid-step anchor
    anchor_step_index: Optional[int] = None
    if regime == "iv_indeterminate":
        if "pseudo_p_label" in inst:
            p_value = "True" if inst["pseudo_p_label"] else "False"
        anchor_step_index = inst.get("pseudo_p_anchor_step_index", None)

    lines: List[str] = []

    # Intro (no 'At the start:' line)
    lines.append("I will reason forward from the initial facts until everything relevant is settled.")

    # Stepwise derivations
    step_line_info: List[Tuple[int, Optional[int], str, str]] = []
    
    for step_idx, s in enumerate(steps):
        ridx = s["reason"]
        r = rules[ridx]
        head = s["learned"]
        used_vals = s.get("values_used", {})
        head_val_bool = status[head][-1][1]
        head_val_txt = _TF(head_val_bool)

        # Random per step (but deterministic for this instance):
        flip_order = rng.choice([True, False])   # True => "Since RULE, and because FACTS, ..."
        coin_true  = rng.choice([True, False])   # True => true-oriented template, False => false-oriented

        # When flipped, add 'only' to the rule description
        rule_desc = _describe_rule_natural(r, coin_true=coin_true, add_only=flip_order)
        facts_txt = _join_known_facts(used_vals)

        if facts_txt:
            if flip_order:
                sentence = (
                    f"Since {rule_desc}, and because {facts_txt}, "
                    f"it follows that {head} = {head_val_txt}."
                )
            else:
                sentence = (
                    f"Given that {facts_txt}, and because {rule_desc}, "
                    f"it follows that {head} = {head_val_txt}."
                )
        else:
            sentence = (
                f"Since {rule_desc}, it follows that {head} = {head_val_txt}."
                if flip_order else
                f"Because {rule_desc}, it follows that {head} = {head_val_txt}."
            )

        local = sentence.find(f"{head} = {head_val_txt}")
        head_bool_col = (local + len(f"{head} = ")) if local >= 0 else None

        lines.append(sentence)
        step_line_info.append((len(lines) - 1, head_bool_col, head, head_val_txt))



    # Final line
    lines.append(f"FINAL ANSWER: {out_var} = {_TF(out_val)}.")

    # Build unmarked COT string
    cot = "\n".join(lines)

    # Helper to compute absolute char index (line-based)
    def abs_char_index(line_idx: int, col_in_line: int) -> int:
        prefix = sum(len(lines[i]) + 1 for i in range(line_idx))
        return prefix + col_in_line

    # Decide p_char_index
    p_char_index: Optional[int] = None

    if regime == "iv_indeterminate":
        # Preserve prior iv behavior (mid-step anchor or intro fallback)
        if anchor_step_index is not None and 0 <= anchor_step_index < len(step_line_info):
            line_idx, col_in_line, _head, _txt = step_line_info[anchor_step_index]
            if col_in_line is not None:
                p_char_index = abs_char_index(line_idx, col_in_line)
            else:
                p_char_index = abs_char_index(line_idx, len(lines[line_idx]))
        else:
            p_char_index = abs_char_index(0, len(lines[0]))
    else:
        # Earliest of first-usage-in-facts and first-derivation-as-head
        first_usage_idx: Optional[int] = None
        first_head_idx: Optional[int] = None

        for idx, s in enumerate(steps):
            if first_usage_idx is None and p_var in s.get("values_used", {}):
                first_usage_idx = idx
            if first_head_idx is None and s.get("learned") == p_var:
                first_head_idx = idx
            if first_usage_idx is not None and first_head_idx is not None:
                break

        chosen_step_idx: Optional[int] = None
        if first_usage_idx is not None and first_head_idx is not None:
            chosen_step_idx = min(first_usage_idx, first_head_idx)
        elif first_usage_idx is not None:
            chosen_step_idx = first_usage_idx
        else:
            chosen_step_idx = first_head_idx  # may be None

        if chosen_step_idx is not None:
            line_idx, head_pos, head_name, head_val_txt = step_line_info[chosen_step_idx]
            s = steps[chosen_step_idx]
            if s.get("learned") == p_var:
                if head_pos is not None:
                    p_char_index = abs_char_index(line_idx, head_pos)
                else:
                    p_char_index = abs_char_index(line_idx, len(lines[line_idx]))
            else:
                p_used_val_bool = s["values_used"][p_var]
                p_used_val_txt = _TF(p_used_val_bool)
                needle = f"{p_var} = {p_used_val_txt}"
                k_local = lines[line_idx].find(needle)
                if k_local != -1:
                    p_char_index = abs_char_index(line_idx, k_local + len(f"{p_var} = "))
                elif head_pos is not None:
                    p_char_index = abs_char_index(line_idx, head_pos)
                else:
                    p_char_index = abs_char_index(line_idx, len(lines[line_idx]))
        else:
            p_char_index = None

    cot_marked = cot if p_char_index is None else (cot[:p_char_index] + "|| " + cot[p_char_index:])

    return {
        "cot": cot,
        "cot_marked": cot_marked,
        "p_char_index": p_char_index,
        "p_value": p_value,
    }
