from __future__ import annotations

"""
Inspect per-token alignment with a learned direction on one example.

Loads a single random (or specified) example from the datagen CSV, computes the
hidden states at a chosen hook point/layer, and prints each token alongside the
"nudge cos" value:
  cos = (h · v) / ||h||
as well as its absolute value |cos|.

Defaults are taken from ablate_config, but can be overridden via CLI flags.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import random

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
    decode_slice,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_paths() -> Tuple[Path, Path]:
    root = _project_root()
    datagen_csv = (root / cfg.DATAGEN_CSV_REL).resolve()
    # Vector path may be relative to this file
    vec_path = Path(cfg.VECTOR_PATH)
    if not vec_path.is_absolute():
        vec_path = (Path(__file__).parent / vec_path).resolve()
    return datagen_csv, vec_path


def _load_vector(vec_path: Path, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    v = np.load(vec_path)
    if v.ndim != 1:
        raise ValueError(f"Loaded vector has shape {v.shape}; expected 1D (d_model,)")
    if int(v.shape[0]) != int(d_model):
        raise ValueError(f"Vector dim {v.shape[0]} != model d_model {d_model}")
    vt = torch.as_tensor(v, dtype=dtype, device=device)
    n = vt.norm().clamp_min(1e-12)
    return vt / n


def _select_row(df: pd.DataFrame, example_id: int | None, seed: int) -> pd.Series:
    df2 = df.copy()
    df2["p_value"] = df2["p_value"].astype(str)
    df2 = df2[(df2["p_value"] == "True") | (df2["p_value"] == "False")]
    if getattr(cfg, 'REGIMES_TO_USE', None) is not None:
        regs = set(cfg.REGIMES_TO_USE)
        df2 = df2[df2["regime"].isin(regs)]
    if len(df2) == 0:
        raise SystemExit("No rows available after filtering by label/regime.")
    if example_id is not None and "id" in df2.columns:
        sub = df2[df2["id"] == int(example_id)]
        if len(sub) == 0:
            raise SystemExit(f"Example id {example_id} not found after filtering.")
        return sub.iloc[0]
    random.seed(int(seed))
    return df2.sample(n=1, random_state=int(seed)).iloc[0]


def main():
    ap = argparse.ArgumentParser(description="Inspect per-token nudge cos for one example")
    ap.add_argument("--example_id", type=int, default=None, help="Pick specific example id from CSV")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for selecting an example")
    ap.add_argument("--layer", type=int, default=None, help="Layer index to inspect; defaults to cfg.LAYER")
    ap.add_argument("--hook", type=str, default=None, help="Hook point name without 'hook_'; defaults to cfg.HOOK_POINT")
    ap.add_argument("--vector", type=str, default=None, help="Override vector .npy path")
    args = ap.parse_args()

    datagen_csv, cfg_vec_path = _resolve_paths()
    if not datagen_csv.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {datagen_csv}")
    df = pd.read_csv(datagen_csv)

    row = _select_row(df, args.example_id, args.seed)
    question = str(row["question"]) ; cot = str(row["cot"]) ; regime = str(row["regime"]) ; label = str(row["p_value"]) ; ex_id = int(row["id"]) if "id" in row else -1
    try:
        p_idx = int(row["p_char_index"])  # char index into CoT
    except Exception:
        raise SystemExit("CSV missing or invalid p_char_index")

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

    # Vector
    vec_path = Path(args.vector).resolve() if args.vector else cfg_vec_path
    d_model = int(model.cfg.d_model)
    v = _load_vector(vec_path, d_model=d_model, device=model.cfg.device, dtype=model.cfg.dtype)

    # Tokenize and split
    tokens, split_idx, text_pre, text_after, text_full = tokenize_with_split(model, question, cot, p_idx)

    # Hook name
    layer = int(args.layer) if args.layer is not None else int(cfg.LAYER)
    hook_point = args.hook if args.hook is not None else str(cfg.HOOK_POINT)
    hook_name = f"blocks.{layer}.hook_{hook_point}"

    # Run and capture
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    if hook_name not in cache:
        raise SystemExit(f"Hook '{hook_name}' not found in cache. Check layer/hook.")
    h = cache[hook_name]  # [1, L, d]
    h = h[0]              # [L, d]

    # Per-token cos and |cos|
    norms = h.norm(dim=-1).clamp_min(1e-12)
    proj = h @ v  # [L]
    cos = proj / norms
    abs_cos = cos.abs()

    # Pretty print
    print(f"Example id={ex_id} | regime={regime} | label={label}")
    print(f"Vector: {vec_path}")
    print(f"Hook: {hook_name} | SeqLen={h.shape[0]} | Split index={split_idx}")
    print("")
    print("Idx  |    cos     |   |cos|    | token")
    print("-----+------------+-----------+---------------------------------")
    L = tokens.shape[1]
    for i in range(L):
        tok_str = decode_slice(model, tokens[0], i, i+1)
        print(f"{i:>4} | {float(cos[i]):>10.6f} | {float(abs_cos[i]):>9.6f} | {tok_str}")
        if i+1 == split_idx:
            print("-----+------------+-----------+---------------------------------  <-- split")


if __name__ == "__main__":
    main()

