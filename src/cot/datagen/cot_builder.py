# cot_builder.py
from typing import Dict, List, Tuple, Optional

def _fmt_lit(sign: str, var: str) -> str:
    return f"not {var}" if sign == "-" else var

def _fmt_rule(rule: Dict) -> str:
    head = rule.get("head")
    op = rule.get("op")
    ins = rule.get("inputs", [])
    if op == "UNARY":
        return f"{head} = {_fmt_lit(*ins[0])}."
    if op == "AND":
        a, b = ins
        return f"{head} = ({_fmt_lit(*a)}) AND ({_fmt_lit(*b)})."
    if op == "OR":
        a, b = ins
        return f"{head} = ({_fmt_lit(*a)}) OR ({_fmt_lit(*b)})."
    if op == "CLAUSE" and head is None:
        parts = " OR ".join(f"({_fmt_lit(s, v)})" for (s, v) in ins)
        return f"{parts} is True."
    return f"{head} {op} {ins}"

def _first_set_event(status_timeline: Dict[str, List[Tuple[int,bool]]], var: str) -> Optional[Tuple[int,bool]]:
    ev = status_timeline.get(var, [])
    return ev[0] if ev else None

def build_cot_and_annotation(inst: Dict, p_var: str = "P") -> Dict:
    """
    Build a gold chain-of-thought (CoT) that follows the instance's steps and
    returns:
      - cot: str (plain CoT)
      - cot_marked: str (only for your .txt usage; '||' inserted at the point P becomes determined, if any)
      - p_char_index: Optional[int] (0-based character index in 'cot' where P becomes determined; None if never)
      - p_value: 'True' | 'False' | 'Unknown'
    """
    initials = inst["initial_assignments"]
    rules = inst["rules"]
    steps = inst["steps"]
    status = inst["status_timeline"]
    out_var = inst["output_var"]
    out_val = inst["output_value"]

    # Determine when/if P becomes known
    p_event = _first_set_event(status, p_var)  # (t, bool) or None
    p_value = "Unknown"
    if p_event is not None:
        p_value = "True" if p_event[1] else "False"

    lines: List[str] = []

    # Header + initial facts
    init_bits = [f"{v}={ 'True' if b else 'False'}" for v, b in sorted(initials.items())]
    lines.append("Let’s solve this step by step.")
    lines.append("Initial facts: " + "; ".join(init_bits) + ".")

    # If P is determined at t=0, add an explicit note line so we have a clean anchor.
    if p_event is not None and p_event[0] == 0:
        lines.append(f"From initial facts, {p_var} is {p_value}.")

    # Map: step index in 'steps' to human sentence
    # We'll also record the line index where P gets determined (if t>0)
    p_line_idx: Optional[int] = None
    for s in steps:
        ridx = s["reason"]
        r = rules[ridx]
        head = s["learned"]
        # Build a short, deterministic explanation line
        rule_txt = _fmt_rule(r)
        known_txt = "; ".join(f"{k}={'True' if v else 'False'}" for k, v in sorted(s.get("values_used", {}).items()))
        head_val = "True" if inst["status_timeline"][head][-1][1] else "False"
        if known_txt:
            line = f"Step {s['t']}: Using {rule_txt} With values [{known_txt}], we deduce {head} is {head_val}."
        else:
            line = f"Step {s['t']}: Using {rule_txt} we deduce {head} is {head_val}."
        lines.append(line)
        if head == p_var and (p_event is not None and p_event[0] == s["t"]):
            p_line_idx = len(lines) - 1

    # Final conclusion
    lines.append(f"Conclusion: Therefore, {out_var} is {'True' if out_val else 'False'}.")

    # Assemble plain CoT and compute character index of P-determination
    cot = "\n".join(lines)
    p_char_index: Optional[int] = None

    if p_event is not None:
        # Find the exact line where we assert "P is <val>"
        if p_event[0] == 0:
            # Use the explicit "From initial facts, P is X." line we added after initials.
            anchor_line = f"From initial facts, {p_var} is {p_value}."
            # Find its start offset
            offset_before = cot.find(anchor_line)
            if offset_before != -1:
                # Insert point right before the truth word, i.e., after "P is "
                inner_idx = anchor_line.find(f"{p_var} is ") + len(f"{p_var} is ")
                p_char_index = offset_before + inner_idx
        else:
            if p_line_idx is not None:
                # Rebuild the specific line to search reliably
                r = rules[steps[p_line_idx - (2 if (p_event[0]==0) else 1)]["reason"]] if False else None  # not needed
                target_line = lines[p_line_idx]
                # Find substring "... we deduce P is <val>."
                needle = f"{p_var} is {p_value}"
                line_start = cot.find(target_line)
                inner_idx = target_line.find(needle) + len(f"{p_var} is ")
                if line_start != -1 and inner_idx >= 0:
                    p_char_index = line_start + inner_idx

    # Build marked version for .txt only (do not use this for CSV)
    if p_char_index is not None and p_char_index >= 0:
        cot_marked = cot[:p_char_index] + "|| " + cot[p_char_index:]
    else:
        cot_marked = cot

    return {
        "cot": cot,
        "cot_marked": cot_marked,
        "p_char_index": p_char_index,
        "p_value": p_value,
    }
