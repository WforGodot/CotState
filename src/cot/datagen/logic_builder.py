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
    rnd = random.Random(seed)

    init = list(dict.fromkeys(initial_vars))
    used = list(dict.fromkeys(used_intermediate_vars))
    unused = list(dict.fromkeys(unused_intermediate_vars))
    indet = list(dict.fromkeys(indeterminate_intermediate_vars))

    # Disjointness + basic guards
    all_list = list(init) + list(used) + list(unused) + list(indet) + [output_var]
    if len(set(all_list)) != len(all_list):
        raise ValueError("Variable categories must be disjoint.")
    if output_var in unused or output_var in indet:
        raise ValueError("output_var cannot be unused or indeterminate.")

    # Random initial truth values
    initial_assignments: Dict[str, bool] = {v: bool(rnd.getrandbits(1)) for v in init}

    rules: List[Dict] = []
    available_for_inputs: List[str] = list(init)  # grows as we define heads

    def _input_pool(head: str) -> List[str]:
        # Never allow unused/indeterminate vars as inputs; avoid self
        return [v for v in available_for_inputs if v not in unused and v not in indet and v != head]

    def sample_inputs(op: str, head: str) -> List[Tuple[str, str]]:
        pool = _input_pool(head)
        if not pool:
            # fallback: allow any initial again (rare)
            pool = [v for v in available_for_inputs if v != head]
        if op == "UNARY" or len(pool) == 1:
            parent = pool[0] if len(pool) == 1 else rnd.choice(pool)
            return [(rnd.choice(['+', '-']), parent)]
        # binary
        if len(pool) >= 2:
            a, b = rnd.sample(pool, 2)
        else:
            a = b = pool[0]
        return [(rnd.choice(['+', '-']), a), (rnd.choice(['+', '-']), b)]

    # Define used + output heads (now allow UNARY/AND/OR again)
    heads_in_order = used + ([output_var] if output_var not in used else [])
    for head in heads_in_order:
        op = "UNARY" if len(available_for_inputs) < 2 else rnd.choice(["UNARY", "AND", "OR"])
        inputs = sample_inputs(op, head)
        rules.append({"head": head, "op": op, "inputs": inputs})
        available_for_inputs.append(head)

    # Define UNUSED heads too so they become determinate, but they won’t be used downstream.
    for head in unused:
        op = "UNARY"
        inputs = sample_inputs(op, head)
        rules.append({"head": head, "op": op, "inputs": inputs})
        available_for_inputs.append(head)

    # ---------- Forward simulation ----------
    rnd.shuffle(rules)  # randomize presentation; `reason` indexes reflect this order

    all_vars = set(init) | set(used) | set(unused) | set(indet) | {output_var}
    value: Dict[str, Optional[bool]] = {v: None for v in all_vars}
    status_timeline: Dict[str, List[Tuple[int, bool]]] = {v: [] for v in all_vars}

    # t=0 initials
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

            # Gather literal values (may be None)
            lits = [(s, v, lit_value(s, v)) for (s, v) in ins]
            vals = [lv for (_s, _v, lv) in lits]

            deduce = None           # Optional[bool] head value
            support_vars: List[str] = []  # parents actually necessary at this deduction

            if op == "UNARY":
                if vals[0] is not None:
                    deduce = vals[0]
                    support_vars = [ins[0][1]]

            elif op == "AND":
                # Early False if any known False
                false_parents = [v for (s, v, lv) in lits if lv is False]
                true_parents = [v for (s, v, lv) in lits if lv is True]
                unknown = any(lv is None for lv in vals)
                if false_parents:
                    deduce = False
                    # One False parent is sufficient to deduce False
                    support_vars = [sorted(false_parents)[0]]
                elif not unknown and len(true_parents) == len(ins):
                    deduce = True
                    support_vars = sorted([v for (s, v) in ins])

            elif op == "OR":
                # Early True if any known True
                true_parents = [v for (s, v, lv) in lits if lv is True]
                false_parents = [v for (s, v, lv) in lits if lv is False]
                unknown = any(lv is None for lv in vals)
                if true_parents:
                    deduce = True
                    # One True parent is sufficient
                    support_vars = [sorted(true_parents)[0]]
                elif not unknown and len(false_parents) == len(ins):
                    deduce = False
                    support_vars = sorted([v for (s, v) in ins])

            if deduce is None:
                continue  # not ready

            # Commit deduction
            t += 1
            value[head] = bool(deduce)
            status_timeline[head].append((t, bool(deduce)))
            steps.append({
                "t": t,
                "learned": head,
                "reason": ridx,
                "values_used": {sv: bool(value[sv]) for sv in support_vars},
            })
            changed = True

    if value[output_var] is None:
        raise RuntimeError("Output variable was not determined by forward chaining.")

    return {
        "initial_assignments": initial_assignments,
        "rules": rules,
        "steps": steps,
        "status_timeline": status_timeline,
        "output_var": output_var,
        "output_value": bool(value[output_var]),
    }
