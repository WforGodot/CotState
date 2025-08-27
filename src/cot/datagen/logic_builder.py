# logic_builder.py
import random
from typing import Dict, List, Tuple, Optional, Iterable

def build_forward_logic_instance(
    initial_vars: Iterable[str],
    used_intermediate_vars: Iterable[str],
    unused_intermediate_vars: Iterable[str],
    indeterminate_intermediate_vars: Iterable[str],
    output_var: str,
    seed: Optional[int] = None,
    forbid_as_input: Optional[Iterable[str]] = None,
    # Soft preferences/limits:
    prefer_input_var: Optional[str] = None,
    input_usage_limits: Optional[Dict[str, Tuple[int, int]]] = None,  # var -> (min, max)
):
    """
    Build a forward-chaining propositional-logic instance.

    Softer behavior:
    - prefer_input_var is a bias: we try to include it when legal, but never fail if not.
    - If AND/OR can't find two legal inputs, we duplicate the one legal input.
    - If even that is blocked by usage limits, we *fallback to UNARY* for that head.
    """
    rnd = random.Random(seed)
    forbidden_set = set(forbid_as_input or [])
    usage_limits = dict(input_usage_limits or {})  # var -> (min, max)
    usage_count: Dict[str, int] = {}

    init = list(dict.fromkeys(initial_vars))
    used = list(dict.fromkeys(used_intermediate_vars))
    unused = list(dict.fromkeys(unused_intermediate_vars))
    indet = list(dict.fromkeys(indeterminate_intermediate_vars))

    # Basic sanity: disjointness
    all_list = list(init) + list(used) + list(unused) + list(indet) + [output_var]
    if len(set(all_list)) != len(all_list):
        raise ValueError("Variable categories must be disjoint.")
    if output_var in unused or output_var in indet:
        raise ValueError("output_var cannot be unused or indeterminate.")

    # Assign initial truth values randomly.
    initial_assignments: Dict[str, bool] = {v: bool(rnd.getrandbits(1)) for v in init}

    rules: List[Dict] = []
    available_for_inputs: List[str] = list(init)  # grows as we define heads

    def under_max(v: str, need: int = 1) -> bool:
        if v in usage_limits:
            _min, _max = usage_limits[v]
            return usage_count.get(v, 0) + need <= _max
        return True

    def bump_usage(v: str, times: int = 1):
        if v not in usage_count:
            usage_count[v] = 0
        usage_count[v] += times

    def legal_pool(head: str) -> List[str]:
        # Never allow: unused/indeterminate vars, self, or forbidden vars.
        pool = [
            v for v in available_for_inputs
            if v not in unused and v not in indet and v != head and v not in forbidden_set
        ]
        return pool

    def pick_sign(v: str) -> str:
        return rnd.choice(['+', '-'])

    def try_make_binary(head: str) -> Optional[List[Tuple[str, str]]]:
        """
        Try to produce two inputs for AND/OR with soft preference.
        Returns a list of two (sign, var) on success; None if impossible even with duplication.
        """
        pool = [v for v in legal_pool(head) if under_max(v, 1)]
        if not pool:
            return None

        # Try to include preferred var if legal
        prefer = prefer_input_var
        prefer_ok = (prefer is not None) and (prefer in pool)

        if prefer_ok:
            a = prefer
            # candidate for b
            others = [v for v in pool if v != a and under_max(v, 1)]
            if others:
                b = rnd.choice(others)
                bump_usage(a, 1); bump_usage(b, 1)
                return [(pick_sign(a), a), (pick_sign(b), b)]
            else:
                # No other legal; try duplicating preferred if allowed
                if under_max(a, 2):  # need +2 total
                    bump_usage(a, 2)
                    return [(pick_sign(a), a), (pick_sign(a), a)]
                # Fallback: ignore preference and try any two from pool (with duplication if needed)
                prefer_ok = False  # drop to generic path

        # Generic path (no preference or couldn't use it)
        if len(pool) >= 2:
            a, b = rnd.sample(pool, 2)
            # If one of them is at its limit for a second bump due to duplication logic elsewhere, this is fine.
            bump_usage(a, 1); bump_usage(b, 1)
            return [(pick_sign(a), a), (pick_sign(b), b)]
        else:
            a = pool[0]
            # Duplicate the single candidate if we can
            if under_max(a, 2):
                bump_usage(a, 2)
                return [(pick_sign(a), a), (pick_sign(a), a)]
            # Cannot duplicate due to max; give up on binary
            return None

    def make_unary(head: str) -> List[Tuple[str, str]]:
        pool = [v for v in legal_pool(head) if under_max(v, 1)]
        if not pool:
            # As an absolute last resort, allow any legal without checking max (to avoid construction failure)
            pool = legal_pool(head)
            if not pool:
                # Truly impossible head; signal caller to rebuild
                raise ValueError(f"No available inputs for head {head}")
            chosen = pool[0] if len(pool) == 1 else rnd.choice(pool)
            # do NOT bump usage if we are bypassing limits (keeps counts consistent with enforcement below)
            return [(pick_sign(chosen), chosen)]
        # Prefer preferred var if possible
        if prefer_input_var in pool:
            chosen = prefer_input_var
        else:
            chosen = pool[0] if len(pool) == 1 else rnd.choice(pool)
        bump_usage(chosen, 1)
        return [(pick_sign(chosen), chosen)]

    # Order of heads:
    heads_in_order = used + ([output_var] if output_var not in used else [])

    # Define used + output (with soft fallbacks)
    for head in heads_in_order:
        # Pick an op, but if binary is impossible, fallback to UNARY
        op = rnd.choice(["UNARY", "AND", "OR"]) if len(available_for_inputs) >= 1 else "UNARY"

        if op == "UNARY":
            inputs = make_unary(head)
        else:
            pair = try_make_binary(head)
            if pair is None:
                # fallback to unary
                op = "UNARY"
                inputs = make_unary(head)
            else:
                inputs = pair

        rules.append({"head": head, "op": op, "inputs": inputs})
        available_for_inputs.append(head)

    # Define UNUSED heads (UNARY only; still excluded as inputs by legal_pool)
    for head in unused:
        inputs = make_unary(head)
        rules.append({"head": head, "op": "UNARY", "inputs": inputs})
        available_for_inputs.append(head)

    rnd.shuffle(rules)

    # ---------- Forward simulation ----------
    all_vars = set(init) | set(used) | set(unused) | set(indet) | {output_var}
    value: Dict[str, Optional[bool]] = {v: None for v in all_vars}
    status_timeline: Dict[str, List[Tuple[int, bool]]] = {v: [] for v in all_vars}

    # Set initials at t=0
    for v, b in initial_assignments.items():
        value[v] = b
        status_timeline[v].append((0, b))

    steps: List[Dict] = []

    def lit_value(sign: str, var: str) -> Optional[bool]:
        if value[var] is None:
            return None
        return value[var] if sign == '+' else (not value[var])

    t = 0
    changed = True
    while changed:
        changed = False

        for ridx, r in enumerate(rules):
            head = r["head"]
            if head is None or value[head] is not None:
                continue

            op = r["op"]
            ins = r["inputs"]
            lit_vals = [lit_value(sign, var) for (sign, var) in ins]
            if any(v is None for v in lit_vals):
                continue

            if op == "UNARY":
                hval = lit_vals[0]
            elif op == "AND":
                hval = all(lit_vals)
            elif op == "OR":
                hval = any(lit_vals)
            else:
                continue

            if value[head] is None:
                t += 1
                value[head] = bool(hval)
                status_timeline[head].append((t, bool(hval)))
                steps.append({
                    "t": t,
                    "learned": head,
                    "reason": ridx,
                    "values_used": {var: bool(value[var]) for (_s, var) in ins}
                })
                changed = True

    if value[output_var] is None:
        # Ultra-conservative emergency fix: if output is still None (should be rare),
        # attach a UNARY rule from any initial var (prefer preferred) and recompute quickly.
        any_init = prefer_input_var if (prefer_input_var in initial_assignments) else (init[0] if init else None)
        if any_init is None:
            raise RuntimeError("No initial variables to attach emergency rule to produce output.")
        # Append a final rule that guarantees determinacy
        rules.append({"head": output_var, "op": "UNARY", "inputs": [('+', any_init)]})
        # Apply once
        t += 1
        value[output_var] = bool(value[any_init])
        status_timeline[output_var].append((t, bool(value[output_var])))
        steps.append({
            "t": t,
            "learned": output_var,
            "reason": len(rules) - 1,
            "values_used": {any_init: bool(value[any_init])}
        })

    # Enforce min-usage limits post-hoc (still honored for regime vi)
    for v, (min_u, _max_u) in usage_limits.items():
        if usage_count.get(v, 0) < min_u:
            raise RuntimeError(f"Did not meet min usage for {v}: got {usage_count.get(v,0)} < {min_u}")

    return {
        "initial_assignments": initial_assignments,
        "rules": rules,
        "steps": steps,
        "status_timeline": status_timeline,
        "output_var": output_var,
        "output_value": bool(value[output_var]),
    }
