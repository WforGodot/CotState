from typing import Dict, List, Tuple, Optional

# ---------- Helpers for NL rule phrasing ----------

def _fmt_lit_words(sign: str, var: str) -> str:
    """Return a literal in words, e.g., '-' -> 'not X', '+' -> 'X'."""
    return f"not {var}" if sign == "-" else var

def _describe_rule_natural(rule: Dict) -> str:
    """
    Short clause (no trailing period) usable inside a sentence.
    Mirrors your earlier semantics for UNARY/AND/OR/CLAUSE.
    """
    head = rule.get("head")
    op = rule.get("op")
    ins = rule.get("inputs", [])

    if op == "UNARY":
        s, v = ins[0]
        lit = _fmt_lit_words(s, v)
        if head is None:
            return f"a variable matches {lit}"
        return f"{head} matches {lit}"

    if op == "AND":
        (a_s, a_v), (b_s, b_v) = ins
        a = _fmt_lit_words(a_s, a_v)
        b = _fmt_lit_words(b_s, b_v)
        return f"{head} is true only when both {a} and {b} are true"

    if op == "OR":
        (a_s, a_v), (b_s, b_v) = ins
        a = _fmt_lit_words(a_s, a_v)
        b = _fmt_lit_words(b_s, b_v)
        return f"{head} is true when at least one of {a} or {b} is true"

    if op == "CLAUSE" and head is None:
        parts = [f"{_fmt_lit_words(s, v)}" for (s, v) in ins]
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
    Emit as VAR=True/False (no spaces around '='), comma-separated with Oxford 'and'.
    """
    if not used_vals:
        return ""
    bits = [f"{k}={_TF(v)}" for k, v in sorted(used_vals.items())]
    if len(bits) == 1:
        return bits[0]
    return ", ".join(bits[:-1]) + f", and {bits[-1]}"

def _first_set_event(status_timeline: Dict[str, List[Tuple[int, bool]]], var: str) -> Optional[Tuple[int, bool]]:
    ev = status_timeline.get(var, [])
    return ev[0] if ev else None

# ---------- Main builder (standardized, tokenizer-friendly) ----------

def build_cot_and_annotation(inst: Dict, p_var: str = "P") -> Dict:
    """
    Build a standardized CoT.

    Conventions:
    - All boolean values are capitalized: True/False.
    - Every determination is written as 'VAR = True/False'.
    - 'cot_marked' inserts '|| ' immediately before the boolean token
      of the FIRST time p_var becomes determinate.
    - p_value remains 'True'/'False'/'Unknown'.
    - Append a rigid final line: 'FINAL ANSWER: OUT = True/False.'
    """
    initials = inst["initial_assignments"]
    rules = inst["rules"]
    steps = inst["steps"]
    status = inst["status_timeline"]
    out_var = inst["output_var"]
    out_val = inst["output_value"]

    p_event = _first_set_event(status, p_var)
    p_value = "Unknown" if p_event is None else ("True" if p_event[1] else "False")
    p_step_t = None if p_event is None else p_event[0]

    lines: List[str] = []

    # Initial facts: VAR=True/False
    init_bits = [f"{v}={_TF(b)}" for v, b in sorted(initials.items())]
    if init_bits:
        lines.append("I will reason forward from the initial facts until everything relevant is settled.")
        lines.append("At the start: " + "; ".join(init_bits) + ".")
    else:
        lines.append("I will reason forward from the rules, starting with no initial facts stated.")

    # If P is already known at t=0, state it once.
    if p_step_t == 0:
        lines.append(f"Right away, {p_var} = {p_value}.")

    # Stepwise derivations (always 'HEAD = True/False')
    for s in steps:
        ridx = s["reason"]
        r = rules[ridx]
        head = s["learned"]
        used_vals = s.get("values_used", {})
        head_val_bool = status[head][-1][1]
        head_val_txt = _TF(head_val_bool)

        rule_desc = _describe_rule_natural(r)
        facts_txt = _join_known_facts(used_vals)

        if facts_txt:
            sentence = (
                f"Given that {facts_txt}, and because {rule_desc}, "
                f"it follows that {head} = {head_val_txt}."
            )
        else:
            sentence = (
                f"Because {rule_desc}, it follows that {head} = {head_val_txt}."
            )

        lines.append(sentence)

    # Rigid final line
    lines.append(f"FINAL ANSWER: {out_var} = {_TF(out_val)}.")

    cot = "\n".join(lines)

    # Inline marker '|| ' before the boolean of the first P assignment
    p_char_index: Optional[int] = None
    if p_value != "Unknown":
        needle = f"{p_var} = {p_value}"
        k = cot.find(needle)
        if k != -1:
            p_char_index = k + len(f"{p_var} = ")
    cot_marked = cot if p_char_index is None else (cot[:p_char_index] + "|| " + cot[p_char_index:])

    return {
        "cot": cot,
        "cot_marked": cot_marked,
        "p_char_index": p_char_index,
        "p_value": p_value,
    }
