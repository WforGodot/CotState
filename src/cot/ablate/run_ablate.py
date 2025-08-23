from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime
import re

import numpy as np
import pandas as pd
import torch

import ablate_config as cfg

# Make sibling 'collect' package importable when running as a script
import sys as _sys
_sys.path.append(str(Path(__file__).resolve().parent.parent))
from collect.model_utils import (
    load_tlens_model,
    tokenize_with_split,
    ensure_dir,
)


REGIMES = ["i_initial", "ii_inconsequential", "iii_derived", "iv_indeterminate", "v_output"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_paths() -> Tuple[Path, Path, Path]:
    root = _project_root()
    datagen_csv = (root / cfg.DATAGEN_CSV_REL).resolve()
    out_dir = (root / cfg.OUT_DIR_REL).resolve()
    # Vector path may be relative to this file
    vec_path = Path(cfg.VECTOR_PATH)
    if not vec_path.is_absolute():
        vec_path = (Path(__file__).parent / vec_path).resolve()
    return datagen_csv, out_dir, vec_path


def _load_vector(vec_path: Path, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    v = np.load(vec_path)
    if v.ndim != 1:
        raise ValueError(f"Loaded vector has shape {v.shape}; expected 1D (d_model,)")
    if int(v.shape[0]) != int(d_model):
        raise ValueError(f"Vector dim {v.shape[0]} != model d_model {d_model}")
    vt = torch.as_tensor(v, dtype=dtype, device=device)
    # Normalize to unit length (avoid zero divide)
    n = vt.norm().clamp_min(1e-12)
    return vt / n


def _get_token_id(model, s: str) -> Tuple[int, bool]:
    toks = model.to_tokens(s, prepend_bos=False)
    tid = int(toks[0, 0].item())
    multi = toks.shape[1] > 1
    return tid, multi


def _select_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_value"] = df["p_value"].astype(str)
    df = df[(df["p_value"] == "True") | (df["p_value"] == "False")]
    regimes = set(cfg.REGIMES_TO_USE) if cfg.REGIMES_TO_USE is not None else set(REGIMES)
    df = df[df["regime"].isin(regimes)]
    if cfg.N_SAMPLES is not None and len(df) > int(cfg.N_SAMPLES):
        df = df.sample(n=int(cfg.N_SAMPLES), random_state=int(cfg.RANDOM_STATE))
    return df.reset_index(drop=True)


def _invert_along_v(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Reflect x across the hyperplane orthogonal to v: x' = x - 2 * proj_v(x)
    # x: [..., d], v: [d]
    # proj scalar: [..., 1]
    proj = x @ v
    return x - 2.0 * proj.unsqueeze(-1) * v


def _hook_inverter(v: torch.Tensor, start_idx: int, end_idx: int):
    # start_idx inclusive, end_idx exclusive
    def fn(resid: torch.Tensor, hook):
        # resid: [batch, pos, d_model]
        if resid.ndim != 3:
            return resid
        b, L, d = resid.shape
        a = max(0, min(L, start_idx))
        bnd = max(a, min(L, end_idx))
        if bnd <= a:
            return resid
        segment = resid[:, a:bnd, :]
        resid[:, a:bnd, :] = _invert_along_v(segment, v)
        return resid
    return fn


def main():
    ap = argparse.ArgumentParser(description="Ablate a learned direction on prior tokens and measure logit shifts")
    ap.add_argument("--n_samples", type=int, default=None)
    args = ap.parse_args()

    datagen_csv, out_dir, vec_path = _resolve_paths()
    ensure_dir(out_dir)

    if not datagen_csv.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {datagen_csv}")
    df = pd.read_csv(datagen_csv)
    required = ["id", "regime", "question", "cot", "p_char_index", "p_value"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing required columns: {miss}")
    rows = _select_rows(df)
    if args.n_samples is not None and len(rows) > args.n_samples:
        rows = rows.sample(n=int(args.n_samples), random_state=int(cfg.RANDOM_STATE)).reset_index(drop=True)
    if len(rows) == 0:
        print("No rows to process after selection. Exiting.")
        return

    # Load model
    model = load_tlens_model(
        model_name=cfg.MODEL_NAME,
        device=cfg.DEVICE,
        dtype=cfg.DTYPE,
        trust_remote_code=cfg.TRUST_REMOTE_CODE,
    )
    try:
        model.eval()
    except Exception:
        pass

    # Probe tokens for True / False
    true_id, true_multi = _get_token_id(model, cfg.TRUE_STR)
    false_id, false_multi = _get_token_id(model, cfg.FALSE_STR)

    # Direction vector
    d_model = int(model.cfg.d_model)
    v = _load_vector(vec_path, d_model=d_model, device=model.cfg.device, dtype=model.cfg.dtype)

    # Hook name
    layer = int(cfg.LAYER)
    hook_name = f"blocks.{layer}.hook_{cfg.HOOK_POINT}"

    # Build run name
    def _sanitize(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._+-]", "-", s)

    vec_stem = _sanitize(vec_path.stem)
    regs_str = "all" if cfg.REGIMES_TO_USE is None else "+".join(_sanitize(r) for r in cfg.REGIMES_TO_USE)
    base_name = f"L{layer}_{cfg.HOOK_POINT}__vec_{vec_stem}__regs_{regs_str}__prior{int(cfg.PRIOR_TOKENS)}__{_sanitize(cfg.MODEL_NAME)}"
    run_name = base_name
    k = 1
    while (out_dir / run_name).exists():
        k += 1
        run_name = f"{base_name}__{k}"
    run_dir = out_dir / run_name
    ensure_dir(run_dir)
    report_path = run_dir / "report.txt"
    per_ex_csv = run_dir / "per_example.csv"

    # Storage for metrics
    rows_out: List[Dict] = []

    prior = int(cfg.PRIOR_TOKENS)

    for _, r in rows.iterrows():
        try:
            p_idx = int(r["p_char_index"])  # char index into CoT
        except Exception:
            continue
        question = str(r["question"]) ; cot = str(r["cot"]) ; regime = str(r["regime"]) ; label = str(r["p_value"]) ; ex_id = int(r["id"]) if pd.notna(r["id"]) else -1

        try:
            tok_full, split_idx, pre_dec, _, _ = tokenize_with_split(model, question, cot, p_idx)
        except Exception:
            continue

        # Use only the prefix up to split
        tokens = tok_full[:, :split_idx]
        L = int(tokens.shape[1])
        if L == 0:
            continue
        start = max(0, L - prior)
        end = L

        with torch.no_grad():
            base_logits = model(tokens)
            base_logit_true = float(base_logits[0, L - 1, true_id].item())
            base_logit_false = float(base_logits[0, L - 1, false_id].item())

            # Ablated forward with hook
            hook_fn = _hook_inverter(v=v, start_idx=start, end_idx=end)
            abl_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
            abl_logit_true = float(abl_logits[0, L - 1, true_id].item())
            abl_logit_false = float(abl_logits[0, L - 1, false_id].item())

        base_margin = base_logit_true - base_logit_false
        abl_margin = abl_logit_true - abl_logit_false
        delta_margin = abl_margin - base_margin

        correct_is_true = (label == "True")
        # Change in the logit of the gold option
        gold_delta = (abl_logit_true - base_logit_true) if correct_is_true else (abl_logit_false - base_logit_false)
        flipped = int(np.sign(base_margin) != np.sign(abl_margin)) if base_margin != 0 else int(np.sign(abl_margin) != 0)

        rows_out.append(dict(
            id=ex_id,
            regime=regime,
            label=label,
            L=L,
            start_idx=start,
            end_idx=end,
            base_true=base_logit_true,
            base_false=base_logit_false,
            abl_true=abl_logit_true,
            abl_false=abl_logit_false,
            base_margin=base_margin,
            abl_margin=abl_margin,
            delta_margin=delta_margin,
            gold_delta=gold_delta,
            flipped=flipped,
        ))

    if not rows_out:
        print("No results to report.")
        return

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(per_ex_csv, index=False)

    # Aggregate summaries
    def _summ(d: pd.DataFrame) -> Dict[str, float]:
        return dict(
            n=int(len(d)),
            avg_delta_margin=float(d["delta_margin"].mean()),
            med_delta_margin=float(d["delta_margin"].median()),
            frac_flipped=float(d["flipped"].mean()),
            avg_gold_delta=float(d["gold_delta"].mean()),
        )

    all_sum = _summ(df_out)
    true_sum = _summ(df_out[df_out["label"] == "True"]) if (df_out["label"] == "True").any() else None
    false_sum = _summ(df_out[df_out["label"] == "False"]) if (df_out["label"] == "False").any() else None

    # Write report
    lines: List[str] = []
    lines.append(f"Ablation run: {run_name}")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    lines.append(f"Model: {cfg.MODEL_NAME} | Hook: {hook_name}")
    lines.append(f"Vector: {vec_path}")
    lines.append(f"Dataset: {datagen_csv}")
    lines.append(f"Regimes: {', '.join(cfg.REGIMES_TO_USE)} | Prior tokens: {cfg.PRIOR_TOKENS}")
    lines.append(f"Samples: {len(df_out)} (requested={cfg.N_SAMPLES})")
    lines.append("")
    lines.append(f"Tokenization note: '{cfg.TRUE_STR}' -> id {true_id}{' (multi-token)' if true_multi else ''}; "
                 f"'{cfg.FALSE_STR}' -> id {false_id}{' (multi-token)' if false_multi else ''}")
    lines.append("")

    def _fmt_summ(title: str, s: Dict[str, float] | None) -> List[str]:
        if s is None:
            return [f"{title}: n=0"]
        return [
            f"{title}: n={s['n']}",
            f"  avg Δmargin (T-F): {s['avg_delta_margin']:.4f}",
            f"  med Δmargin (T-F): {s['med_delta_margin']:.4f}",
            f"  frac flipped sign : {s['frac_flipped']:.3f}",
            f"  avg Δgold logit  : {s['avg_gold_delta']:.4f}",
        ]

    lines.extend(_fmt_summ("Overall", all_sum))
    lines.append("")
    lines.extend(_fmt_summ("Label=True", true_sum))
    lines.append("")
    lines.extend(_fmt_summ("Label=False", false_sum))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report → {report_path}")
    print(f"Per-example CSV → {per_ex_csv}")


if __name__ == "__main__":
    main()
