# src/cot/collect/collect_activations.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import collect_config as cfg
from model_utils import (
    load_tlens_model,
    tokenize_with_split,
    clamp_layers,
    ensure_dir,
)

REGIMES = ["i_initial", "ii_inconsequential", "iii_derived", "iv_indeterminate", "v_output"]

def _project_root() -> Path:
    # .../repo/src/cot/collect/collect_activations.py -> repo/src
    return Path(__file__).resolve().parents[2]

def _paths():
    root = _project_root()
    datagen_csv = root / cfg.DATAGEN_CSV_REL
    out_dir = root / cfg.COLLECT_OUT_REL
    return datagen_csv, out_dir

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect activations after P-commitment in CoTs.")
    # Optional per-regime overrides for tokens-after and counts
    for r in REGIMES:
        p.add_argument(f"--{r}_tokens_after", type=int, default=None,
                       help=f"Tokens after split to keep for {r} (-1 for all).")
        p.add_argument(f"--{r}_n", type=int, default=None,
                       help=f"Number of examples from {r} to use.")
    p.add_argument("--device", type=str, default=cfg.DEVICE)
    p.add_argument("--dtype", type=str, default=cfg.DTYPE)
    p.add_argument("--hook_point", type=str, default=cfg.HOOK_POINT)
    p.add_argument("--model_name", type=str, default=cfg.MODEL_NAME)
    p.add_argument("--trust_remote_code", action="store_true" if cfg.TRUST_REMOTE_CODE else "store_false",
                   default=cfg.TRUST_REMOTE_CODE)
    return p.parse_args()

def build_selection_maps(args: argparse.Namespace) -> Tuple[Dict[str, int], Dict[str, int]]:
    # tokens-after map (fallback to config)
    tokens_after = {}
    for r in REGIMES:
        override = getattr(args, f"{r}_tokens_after")
        tokens_after[r] = cfg.TOKENS_AFTER_BY_REGIME.get(r, -1) if override is None else override

    # counts map (fallback to config)
    counts = {}
    for r in REGIMES:
        override = getattr(args, f"{r}_n")
        counts[r] = cfg.REGIME_SAMPLE_COUNTS.get(r, 0) if override is None else override

    return tokens_after, counts

def select_rows_by_regime(df: pd.DataFrame, counts: Dict[str, int]) -> pd.DataFrame:
    # Keep only determined P rows (safety)
    df = df.copy()
    df["p_value"] = df["p_value"].astype(str)
    df = df[(df["p_value"] == "True") | (df["p_value"] == "False")]

    # Enforce regime-wise caps in the order they appear (deterministic)
    selected = []
    for r in REGIMES:
        need = counts.get(r, 0)
        if need <= 0:
            continue
        chunk = df[df["regime"] == r].head(need)
        selected.append(chunk)
    if selected:
        return pd.concat(selected, ignore_index=True)
    return df.iloc[0:0].copy()

def main():
    args = parse_args()
    tokens_after_by_regime, counts_by_regime = build_selection_maps(args)
    datagen_csv, out_dir = _paths()
    ensure_dir(out_dir)

    # 1) Load data
    if not datagen_csv.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {datagen_csv}")
    df = pd.read_csv(datagen_csv)
    # Required columns check
    needed_cols = ["id", "regime", "question", "answer", "cot", "p_char_index", "p_value"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Filter & select rows
    rows = select_rows_by_regime(df, counts_by_regime)
    if len(rows) == 0:
        print("No rows selected after filtering. Exiting.")
        return

    # 2) Load model
    model = load_tlens_model(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    n_layers = model.cfg.n_layers
    layers = clamp_layers(cfg.LAYERS, n_layers)
    hook_point = args.hook_point

    # Prepare accumulators
    feats_by_layer: Dict[int, List[np.ndarray]] = {i: [] for i in layers}
    labels: List[Tuple[int, str, str, int, int]] = []
    # tuple = (example_id, regime, p_value, token_offset_from_split, token_abs_index_in_seq)
    debug_lines: List[str] = []

    # 3) Iterate rows and collect
    for _, row in tqdm(rows.iterrows(), total=len(rows)):
        ex_id = int(row["id"])
        question = str(row["question"])
        cot = str(row["cot"])
        regime = str(row["regime"])
        p_val = str(row["p_value"])
        p_char_index_raw = row["p_char_index"]

        # Skip if p_char_index missing
        try:
            p_idx = int(p_char_index_raw)
        except Exception:
            continue

        # Tokenize with precise split
        try:
            tokens, split_tok_idx, text_pre_dec, text_after_dec, text_full = tokenize_with_split(
                model, question, cot, p_idx
            )
        except Exception:
            # Skip malformed examples
            continue

        seq_len = tokens.shape[1]
        n_after = tokens_after_by_regime.get(regime, -1)
        if n_after is None or n_after < 0:
            sel_end = seq_len
        else:
            sel_end = min(seq_len, split_tok_idx + n_after)

        if sel_end <= split_tok_idx:
            # Nothing to collect
            continue

        chosen_token_indices = list(range(split_tok_idx, sel_end))

        # Run with cache (teacher-forced)
        with torch.no_grad():
            _logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)

        # Extract and store
        for lyr in layers:
            try:
                acts = cache[hook_point, lyr]  # shape [1, seq, d_model]
            except KeyError as e:
                raise KeyError(
                    f"Hook point '{hook_point}' not found at layer {lyr}. "
                    f"Try one of: 'resid_pre', 'resid_mid', 'resid_post', 'mlp_post', 'attn_out', 'attn_pattern'."
                ) from e
            sel = acts[0, chosen_token_indices, :].detach().cpu().numpy()  # [K, d_model]
            feats_by_layer[lyr].append(sel)

        # Labels (one per selected token)
        for k, tok_idx in enumerate(chosen_token_indices):
            labels.append((ex_id, regime, p_val, k, tok_idx))

        # Build a compact debug line (up to cfg.MAX_DEBUG examples)
        if len(debug_lines) < cfg.MAX_DEBUG:
            # Reconstruct text slices matching tokenization:
            pre_text = model.to_string(tokens[0, :split_tok_idx])
            sel_text = model.to_string(tokens[0, split_tok_idx:sel_end])
            post_text = model.to_string(tokens[0, sel_end:])
            dbg = pre_text + "[[" + sel_text + "]]" + post_text
            # Keep it small: elide very long lines
            if len(dbg) > 1200:
                dbg = dbg[:600] + " ... " + dbg[-600:]
            header = f"(id={ex_id}, regime={regime}, p={p_val}, split_tok={split_tok_idx}, kept={sel_end - split_tok_idx})"
            debug_lines.append(header + "\n" + dbg + "\n")

    # 4) Finalize arrays and save
    # Stack per-layer features: [N_total_tokens, d_model]
    stacked: Dict[str, np.ndarray] = {}
    total_tokens = 0
    for lyr in layers:
        if len(feats_by_layer[lyr]) == 0:
            continue
        arr = np.concatenate(feats_by_layer[lyr], axis=0)
        stacked[f"acts_{hook_point}_layer{lyr}"] = arr
        total_tokens = max(total_tokens, arr.shape[0])

    # Write outputs
    run_tag = f"{hook_point}_qwen3_collect"
    out_npz = out_dir / f"{run_tag}.npz"
    np.savez_compressed(out_npz, **stacked)

    # Labels CSV
    labels_cols = ["example_id", "regime", "p_value", "offset_from_split", "token_index_in_seq"]
    labels_df = pd.DataFrame(labels, columns=labels_cols)
    labels_csv = out_dir / f"{run_tag}_labels.csv"
    labels_df.to_csv(labels_csv, index=False)

    # Info JSON
    info = {
        "model_name": args.model_name,
        "device": args.device,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "hook_point": hook_point,
        "layers": layers,
        "total_selected_tokens": int(labels_df.shape[0]),
        "unique_examples": int(labels_df["example_id"].nunique()),
        "counts_by_regime": {r: int((rows["regime"] == r).sum()) for r in REGIMES},
        "tokens_after_by_regime": {r: int(v) for r, v in tokens_after_by_regime.items()},
        "inputs_csv": str(datagen_csv),
        "features_file": str(out_npz),
        "labels_file": str(labels_csv),
    }
    info_json = out_dir / f"{run_tag}_info.json"
    with open(info_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # Debug file
    debug_path = out_dir / f"{run_tag}_debug.txt"
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("\n".join(debug_lines))

    print(f"Saved features: {out_npz}")
    print(f"Saved labels  : {labels_csv}")
    print(f"Saved info    : {info_json}")
    print(f"Saved debug   : {debug_path}")

if __name__ == "__main__":
    main()
