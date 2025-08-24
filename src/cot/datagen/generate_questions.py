# generate_questions.py

import csv
import json
import os
import random
import math

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import datagen_config as cfg

from logic_builder import build_forward_logic_instance
from cot_builder import build_cot_and_annotation


# --------------------------
# Output pathing (relative to src/)
# --------------------------
def _project_root() -> Path:
    # This file lives at src/cot/datagen/generate_questions.py
    # parents[2] -> src/
    return Path(__file__).resolve().parents[2]

def _out_dir() -> Path:
    # Mirror collect_activations.py behavior: write under src/cot/outputs/datagen
    return _project_root() / "cot" / "outputs" / "datagen"

def _out_paths() -> Tuple[Path, Path, Path]:
    odir = _out_dir()
    jsonl = odir / getattr(cfg, "JSONL_FILENAME", "proplogic_dataset.jsonl")
    csvf = odir / getattr(cfg, "CSV_FILENAME", "proplogic_questions.csv")
    dbg  = odir / getattr(cfg, "DEBUG_FILENAME", "proplogic_debug.txt")
    return jsonl, csvf, dbg


# --------------------------
# Helpers (sampling + checks)
# --------------------------
def _rand_count(rng: random.Random, lo_hi: Tuple[int, int]) -> int:
    lo, hi = lo_hi
    return rng.randint(lo, hi)

def _status_known_t(status_timeline: Dict[str, List[Tuple[int, bool]]], var: str) -> Optional[Tuple[int,bool]]:
    events = status_timeline.get(var, [])
    return events[0] if events else None

def _sample_disjoint_sets(
    rng: random.Random,
    all_vars: List[str],
    want_initial: int,
    want_used: int,
    want_unused: int,
    want_indet: int,
    p: str,
    regime: str,
):
    pool = [v for v in all_vars]
    if p not in pool:
        pool = [p] + pool

    pool_wo_p = [v for v in pool if v != p]

    init: Set[str] = set()
    used: Set[str] = set()
    unused: Set[str] = set()
    indet: Set[str] = set()

    # Regime-specific placement of P
    if regime == "i_initial":
        init.add(p)
    elif regime == "ii_inconsequential":
        init.add(p)
    elif regime == "iii_derived":
        used.add(p)
    elif regime == "iv_indeterminate":
        indet.add(p)
    elif regime == "v_output":
        pass
    elif regime == "vi_single_use":
        # P is initial and must be used exactly once as an input
        init.add(p)
    elif regime == "vii_max_use":
        # P is initial and should be used as an input in every rule if legal
        init.add(p)
    else:
        raise ValueError(f"Unknown regime {regime}")

    need_initial = max(0, want_initial - len(init))
    need_used = max(0, want_used - len(used))
    need_unused = max(0, want_unused - len(unused))
    need_indet = max(0, want_indet - len(indet))

    rng.shuffle(pool_wo_p)
    cursor = 0

    def take(n: int) -> List[str]:
        nonlocal cursor
        chunk = pool_wo_p[cursor: cursor + n]
        cursor += n
        if len(chunk) < n:
            raise ValueError("Not enough variables to satisfy counts; increase PREFERRED_VARS.")
        return chunk

    init.update(take(need_initial))
    used.update(take(need_used))
    unused.update(take(need_unused))
    indet.update(take(need_indet))

    remaining = [v for v in pool if v not in init | used | unused | indet]
    if regime == "v_output":
        output_var = p
    else:
        if not remaining:
            remaining = [v for v in pool if v not in init | unused | indet]
        output_var = rng.choice(remaining)

    assert len(init & used) == 0 and len(init & unused) == 0 and len(init & indet) == 0
    assert len(used & unused) == 0 and len(used & indet) == 0
    assert len(unused & indet) == 0

    return list(init), list(used), list(unused), list(indet), output_var


# --------------------------
# Minimal support backtrace
# --------------------------
def _final_values(status_timeline: Dict[str, List[Tuple[int, bool]]]) -> Dict[str, Optional[bool]]:
    final = {}
    for v, events in status_timeline.items():
        if not events:
            final[v] = None
        else:
            final[v] = events[-1][1]
    return final

def _first_time(status_timeline: Dict[str, List[Tuple[int, bool]]], var: str) -> int:
    ev = status_timeline.get(var, [])
    return ev[0][0] if ev else 10**9  # unknowns get a very large t

def _rule_map_by_head(inst_rules: List[Dict]) -> Dict[str, int]:
    mp = {}
    for idx, r in enumerate(inst_rules):
        head = r.get("head")
        if head:
            mp[head] = idx
    return mp

def _compute_minimal_support(inst: Dict) -> Set[str]:
    steps = inst["steps"]
    # Map var -> its step dict
    learned = {s["learned"]: s for s in steps}

    support: Set[str] = set()
    def need(var: str):
        if var in support:
            return
        support.add(var)
        s = learned.get(var)
        if not s:
            return  # initial var
        for parent in s.get("values_used", {}).keys():
            need(parent)

    need(inst["output_var"])
    return support


# --------------------------
# Attempt per regime
# --------------------------
def _attempt_instance_for_regime(
    rng: random.Random,
    all_vars: List[str],
    regime: str,
    p: str = "P",
):
    # Tunables (with cfg fallbacks)
    vii_max_decl_used = getattr(cfg, "VII_MAX_DECLARED_USED", 3)         # cap declared used in (vii)
    vii_rule_usage_frac = getattr(cfg, "VII_MIN_RULE_USAGE_FRAC", 0.7)    # P must appear in >= this fraction of rules
    vii_used_on_path_frac = getattr(cfg, "VII_USED_ON_PATH_FRAC", 0.5)    # fraction of declared 'used' that must be on minimal path

    for _ in range(cfg.MAX_TRIES_PER_ITEM):
        n_init = _rand_count(rng, cfg.COUNTS["initial"])
        n_used = _rand_count(rng, cfg.COUNTS["used"])
        n_unused = _rand_count(rng, cfg.COUNTS["unused"])
        n_indet = _rand_count(rng, cfg.COUNTS["indeterminate"])

        # Make (vii) easier to satisfy by limiting declared 'used'
        if regime == "vii_max_use":
            n_used = min(n_used, vii_max_decl_used)

        try:
            init, used, unused, indet, out = _sample_disjoint_sets(
                rng, all_vars, n_init, n_used, n_unused, n_indet, p, regime
            )
        except ValueError:
            continue

        # Regime-specific builder constraints
        forbid_inputs = {p} if regime == "ii_inconsequential" else set()

        prefer_input_var = None
        input_usage_limits = None
        if regime == "vi_single_use":
            input_usage_limits = {p: (1, 1)}           # exactly once across all rule inputs
        elif regime == "vii_max_use":
            prefer_input_var = p                        # bias toward using P everywhere

        # Build instance
        try:
            inst = build_forward_logic_instance(
                initial_vars=init,
                used_intermediate_vars=used,
                unused_intermediate_vars=unused,
                indeterminate_intermediate_vars=indet,
                output_var=out,
                seed=rng.getrandbits(31),
                forbid_as_input=forbid_inputs,
                prefer_input_var=prefer_input_var,
                input_usage_limits=input_usage_limits,
            )
        except Exception:
            continue  # resample

        steps = inst["steps"]
        status = inst["status_timeline"]

        # Regime-specific checks
        if regime == "i_initial":
            if p not in init:
                continue

        elif regime == "ii_inconsequential":
            if p not in init:
                continue
            minimal_support = _compute_minimal_support(inst)
            if p in minimal_support:
                continue
            p_used_anywhere = any(
                any(var == p for (_s, var) in r.get("inputs", []))
                for r in inst["rules"]
            )
            if p_used_anywhere:
                continue

        elif regime == "iii_derived":
            if p in init:
                continue
            if _status_known_t(status, p) is None:
                continue

        elif regime == "iv_indeterminate":
            if _status_known_t(status, p) is not None:
                continue
            inst["pseudo_p_label"] = bool(rng.getrandbits(1))
            inst["pseudo_p_anchor_step_index"] = (max(0, len(steps) // 2) if steps else None)

        elif regime == "v_output":
            if inst["output_var"] != p:
                continue

        elif regime == "vi_single_use":
            # Exactly one usage of P across all rule inputs
            uses = sum(
                1 for r in inst["rules"]
                for (_s, v) in r.get("inputs", [])
                if v == p
            )
            if uses != 1:
                continue

        elif regime == "vii_max_use":
            # P should be used in MOST rules (fractional threshold), not all
            total_rules = len(inst["rules"])
            uses_rules = sum(
                1 for r in inst["rules"]
                if any(v == p for (_s, v) in r.get("inputs", []))
            )
            min_uses = max(1, math.ceil(total_rules * float(vii_rule_usage_frac)))
            if uses_rules < min_uses:
                continue

        # -------- Minimal-support checks (relaxed for vii) --------
        minimal_support = _compute_minimal_support(inst)

        if regime != "vii_max_use":
            # Every declared used var must be on the minimal path
            if not set(used).issubset(minimal_support):
                continue
        else:
            # Only require that a fraction of declared 'used' end up on the minimal path
            if len(used) > 0:
                on_path = sum(1 for u in used if u in minimal_support)
                need_on_path = max(1, math.ceil(vii_used_on_path_frac * len(used)))
                if on_path < need_on_path:
                    continue

        # -------- Unused must be determinate AND off the minimal path --------
        unused_ok = True
        for u in unused:
            if _status_known_t(status, u) is None:
                unused_ok = False
                break
            if u in minimal_support:
                unused_ok = False
                break
        if not unused_ok:
            continue

        # Store declared categories for debugging/inspection
        inst["regime"] = regime
        inst["declared_initial"] = init
        inst["declared_used"] = used
        inst["declared_unused"] = unused
        inst["declared_indeterminate"] = indet
        inst["minimal_support"] = sorted(minimal_support)
        if regime == "ii_inconsequential":
            inst["p_forbidden_as_input"] = True

        return inst

    raise RuntimeError(f"Could not construct an instance for regime {regime} after {cfg.MAX_TRIES_PER_ITEM} tries.")




# --------------------------
# NL formatting & exporters
# --------------------------
def _fmt_lit(sign: str, var: str) -> str:
    return f"not {var}" if sign == "-" else var

def _fmt_rule(rule: Dict) -> str:
    head = rule.get("head")
    op = rule.get("op")
    ins = list(rule.get("inputs", []))

    def lit(sig_var):
        s, v = sig_var
        return f"not {v}" if s == "-" else v

    if op == "UNARY":
        if len(ins) != 1:
            # degrade gracefully: show whatever we have
            return f"{head} = {lit(ins[0])}." if ins else f"{head} = ?."
        return f"{head} = {lit(ins[0])}."

    if op in ("AND", "OR"):
        # Defensive: ensure we have two inputs for printing
        if len(ins) < 2:
            ins = ins + ins[:1]  # duplicate first if only one exists
        a, b = ins[0], ins[1]
        joiner = "AND" if op == "AND" else "OR"
        return f"{head} = ({lit(a)}) {joiner} ({lit(b)})."

    if op == "CLAUSE" and head is None:
        parts = " OR ".join(f"({lit(sig_var)})" for sig_var in ins)
        return f"{parts} is True."

    # Fallback
    return f"{head} {op} {ins}"


def instance_to_nl_question(inst: Dict) -> Tuple[str, str]:
    initials = inst["initial_assignments"]
    rules = inst["rules"]
    out_var = inst["output_var"]
    out_val = inst["output_value"]

    initial_lines = [f"{v} is {'True' if b else 'False'}" for v, b in sorted(initials.items())]
    rules_lines = [_fmt_rule(r) for r in rules]

    header = "You are given a small propositional logic system."
    given = "Initially: " + "; ".join(initial_lines) + "."
    rules_text = "Rules:\n" + "\n".join(f"- {line}" for line in rules_lines)
    ask = f"Question: What is the truth value of {out_var}? Answer 'True' or 'False'."

    question_text = "\n".join([header, given, rules_text, ask])
    answer_text = "True" if out_val else "False"
    return question_text, answer_text


def write_jsonl(path: str, items: List[Dict]) -> None:
    if not path:
        return
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex) + "\n")

def write_csv(path: str, items: List[Dict]) -> None:
    if not path:
        return
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # add cot + annotation columns
        writer.writerow(["id", "regime", "question", "answer", "cot", "p_char_index", "p_value"])
        for idx, inst in enumerate(items):
            q, a = instance_to_nl_question(inst)
            # build CoT + annotation (always track 'P', including new regimes)
            cot_pack = build_cot_and_annotation(inst, p_var="P")
            writer.writerow([
                idx, inst.get("regime", ""), q, a,
                cot_pack["cot"],
                cot_pack["p_char_index"] if cot_pack["p_char_index"] is not None else "",
                cot_pack["p_value"],
            ])

def write_debug_txt(path: str, items: List[Dict]) -> None:
    if not path:
        return
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # first example per regime
    firsts: Dict[str, Dict] = {}
    for inst in items:
        r = inst.get("regime", "")
        if r and r not in firsts:
            firsts[r] = inst

    with open(path, "w", encoding="utf-8") as f:
        for regime in sorted(firsts.keys()):
            inst = firsts[regime]
            q, a = instance_to_nl_question(inst)
            f.write(f"=== {regime} ===\n")
            f.write(q + "\n")
            f.write(f"Gold answer: {a}\n")
            f.write("Declared categories:\n")
            f.write(f"  initial:        {sorted(inst.get('declared_initial', []))}\n")
            f.write(f"  useful(used):   {sorted(inst.get('declared_used', []))}\n")
            f.write(f"  useless(unused):{sorted(inst.get('declared_unused', []))}\n")
            f.write(f"  indeterminate:  {sorted(inst.get('declared_indeterminate', []))}\n")
            f.write(f"  output_var:     {inst.get('output_var')}\n")
            f.write("Observed minimal support path to output:\n")
            f.write(f"  minimal_support:{inst.get('minimal_support', [])}\n")

            # ---- CoT + annotation ----
            cot_pack = build_cot_and_annotation(inst, p_var="P")
            f.write("\nCOT (gold):\n")
            f.write(cot_pack["cot_marked"] + "\n")

            if inst.get("regime") == "iv_indeterminate" and "pseudo_p_label" in inst:
                anchor_idx = inst.get("pseudo_p_anchor_step_index", None)
                if anchor_idx is not None and 0 <= anchor_idx < len(inst["steps"]):
                    anchor_var = inst["steps"][anchor_idx]["learned"]
                    f.write(
                        f"(P is logically indeterminate; assigned pseudo-label {cot_pack['p_value']}. "
                        f"Split marker anchored on step {anchor_idx} ({anchor_var}). "
                        f"char_index={cot_pack['p_char_index']})\n"
                    )
                else:
                    f.write(
                        f"(P is logically indeterminate; assigned pseudo-label {cot_pack['p_value']}. "
                        f"Split marker anchored near the header. "
                        f"char_index={cot_pack['p_char_index']})\n"
                    )
            else:
                if cot_pack["p_char_index"] is not None:
                    f.write(f"(P becomes determinate at char index {cot_pack['p_char_index']}, value={cot_pack['p_value']})\n")
                else:
                    f.write("(P remains indeterminate in this instance)\n")

# -------------
# Script entry
# -------------
def generate_all():
    rng = random.Random(cfg.SEED)
    vars_pool = [v for v in cfg.PREFERRED_VARS]
    if "P" not in vars_pool:
        vars_pool = ["P"] + vars_pool

    out: List[Dict] = []
    for regime, n in cfg.QUESTIONS_PER_REGIME.items():
        for _ in range(n):
            inst = _attempt_instance_for_regime(rng, vars_pool, regime, p="P")
            out.append(inst)
    return out

if __name__ == "__main__":
    dataset = generate_all()
    print(
        f"Generated {len(dataset)} items "
        f"({', '.join(f'{k}:{v}' for k,v in cfg.QUESTIONS_PER_REGIME.items())})."
    )

    jsonl_path, csv_path, debug_path = _out_paths()

    # Ensure directory exists
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Always write under src/cot/outputs/datagen (independent of CWD)
    if jsonl_path:
        write_jsonl(str(jsonl_path), dataset)
        print(" -", jsonl_path)
    if csv_path:
        write_csv(str(csv_path), dataset)
        print(" -", csv_path)
    if debug_path:
        write_debug_txt(str(debug_path), dataset)
        print(" -", debug_path)
