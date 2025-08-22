import random
from typing import Dict, List, Tuple, Optional, Iterable

def build_forward_logic_instance(
    initial_vars: Iterable[str],
    used_intermediate_vars: Iterable[str],
    unused_intermediate_vars: Iterable[str],
    indeterminate_intermediate_vars: Iterable[str],
    output_var: str,
    seed: Optional[int] = None,
):
    """
    Build a forward-chaining propositional-logic instance.

    Changes from previous version:
      - Unused vars now GET THEIR OWN RULES (UNARY) so they end up DETERMINATE,
        but they are never allowed as inputs to anything else (hence useless).
      - We DISALLOW 'OR' in defining rules to avoid 'used' variables being bypassed
        by a trivially true sibling. (UNARY/AND only.)
    """
    rnd = random.Random(seed)

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

    def sample_inputs(op: str, available: List[str], head: str) -> List[Tuple[str, str]]:
        # Never allow unused/indeterminate vars as inputs; and avoid self.
        pool = [v for v in available if v not in unused and v not in indet and v != head]
        if not pool:
            # Fallback: if nothing else, allow initial again (should be rare)
            pool = [v for v in available if v != head]
        if op == "UNARY" or len(pool) == 1:
            parent = pool[0] if len(pool) == 1 else rnd.choice(pool)
            sign = rnd.choice(['+', '-'])
            return [(sign, parent)]
        else:
            # AND: try to ensure distinct parents if possible
            if len(pool) >= 2:
                a, b = rnd.sample(pool, 2)
            else:
                a = b = pool[0]
            return [(rnd.choice(['+', '-']), a), (rnd.choice(['+', '-']), b)]

    # Order of heads to define:
    #   1) used vars (these must be derivable)
    #   2) output var
    heads_in_order = used + ([output_var] if output_var not in used else [])

    # Define used + output heads with UNARY/AND only (no OR)
    for head in heads_in_order:
        op = "UNARY" if len(available_for_inputs) < 2 else rnd.choice(["UNARY", "AND"])
        inputs = sample_inputs(op, available_for_inputs, head)
        rule = {"head": head, "op": op, "inputs": inputs}
        rules.append(rule)
        # This head becomes available as an input for future heads
        available_for_inputs.append(head)

    # Define UNUSED heads too (so they become determinate), but still never usable as inputs.
    # Place them AFTER everything else; they depend on currently-available inputs but
    # they WON'T be accepted as inputs for other rules (see pool filter above).
    for head in unused:
        # force UNARY so it's trivially determinable
        op = "UNARY"
        inputs = sample_inputs(op, available_for_inputs, head)
        rule = {"head": head, "op": op, "inputs": inputs}
        rules.append(rule)
        # Even though we append to available_for_inputs, they'll be filtered out as inputs later.
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
            else:
                # We don't generate OR/CLAUSE in this builder
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
        raise RuntimeError("Output variable was not determined by forward chaining. "
                           "Try different inputs or add more rules.")

    return {
        "initial_assignments": initial_assignments,
        "rules": rules,
        "steps": steps,
        "status_timeline": status_timeline,
        "output_var": output_var,
        "output_value": bool(value[output_var]),
    }
