from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, DefaultDict
from collections import defaultdict

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
    get_pad_id,
    pad_and_stack,
    make_names_filter,
    decode_slice,
)

REGIMES = ["i_initial", "ii_inconsequential", "iii_derived", "iv_indeterminate", "v_output"]

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _paths():
    root = _project_root()
    datagen_csv = root / cfg.DATAGEN_CSV_REL
    out_dir = root / cfg.COLLECT_OUT_REL
    return datagen_csv, out_dir

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect activations after P-commitment in CoTs (batched, bucketed).")
    for r in REGIMES:
        p.add_argument(f"--{r}_tokens_after", type=int, default=None,
                       help=f"Tokens after split to keep for {r} (-1 for all).")
        p.add_argument(f"--{r}_n", type=int, default=None,
                       help=f"Number of examples from {r} to use.")
    p.add_argument("--device", type=str, default=cfg.DEVICE)
    p.add_argument("--dtype", type=str, default=cfg.DTYPE)
    p.add_argument("--hook_point", type=str, default=cfg.HOOK_POINT)
    p.add_argument("--model_name", type=str, default=cfg.MODEL_NAME)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--bucket_size", type=int, default=cfg.LENGTH_BUCKET_SIZE)
    p.add_argument("--empty_cache_every",
                   type=int,
                   default=getattr(cfg, "EMPTY_CACHE_EVERY_N_BATCHES", 0),
                   help="Call torch.cuda.empty_cache() every N batches (0 to disable).")
    p.add_argument("--trust_remote_code",
                   action="store_true" if cfg.TRUST_REMOTE_CODE else "store_false",
                   default=cfg.TRUST_REMOTE_CODE)
    return p.parse_args()

def build_selection_maps(args: argparse.Namespace) -> Tuple[Dict[str, int], Dict[str, int]]:
    tokens_after = {}
    for r in REGIMES:
        override = getattr(args, f"{r}_tokens_after")
        tokens_after[r] = cfg.TOKENS_AFTER_BY_REGIME.get(r, -1) if override is None else override

    counts = {}
    for r in REGIMES:
        override = getattr(args, f"{r}_n")
        counts[r] = cfg.REGIME_SAMPLE_COUNTS.get(r, 0) if override is None else override
    return tokens_after, counts

def select_rows_by_regime(df: pd.DataFrame, counts: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["p_value"] = df["p_value"].astype(str)
    df = df[(df["p_value"] == "True") | (df["p_value"] == "False")]

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

    # Load data
    if not datagen_csv.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {datagen_csv}")
    df = pd.read_csv(datagen_csv)
    needed_cols = ["id", "regime", "question", "answer", "cot", "p_char_index", "p_value"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    rows = select_rows_by_regime(df, counts_by_regime)
    if len(rows) == 0:
        print("No rows selected after filtering. Exiting.")
        return

    # Load model
    model = load_tlens_model(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    n_layers = model.cfg.n_layers
    layers = clamp_layers(cfg.LAYERS, n_layers)
    hook_point = args.hook_point
    names_filter = make_names_filter(hook_point, layers)
    pad_id = get_pad_id(model)

    # Pretokenize to CPU & compute keep ranges (so we can bucket by length cleanly)
    pretokenized: List[Dict] = []
    for _, row in rows.iterrows():
        try:
            p_idx = int(row["p_char_index"])
        except Exception:
            continue
        question = str(row["question"])
        cot = str(row["cot"])
        regime = str(row["regime"])
        p_val = str(row["p_value"])
        ex_id = int(row["id"])

        try:
            tokens, split_idx, _pre_dec, _after_dec, _full_dec = tokenize_with_split(model, question, cot, p_idx)
        except Exception:
            continue

        seq_len = int(tokens.shape[1])
        n_after = tokens_after_by_regime.get(regime, -1)
        sel_end = seq_len if (n_after is None or n_after < 0) else min(seq_len, split_idx + n_after)
        if sel_end <= split_idx:
            continue

        pretokenized.append({
            "id": ex_id,
            "regime": regime,
            "p_value": p_val,
            "tokens": tokens.cpu(),   # keep on CPU until batch forward
            "seq_len": seq_len,
            "split": split_idx,
            "sel_end": sel_end,
        })

    if len(pretokenized) == 0:
        print("No usable examples after tokenization/split. Exiting.")
        return

    # Bucket by length to reduce pad waste
    bucket_size = max(1, int(args.bucket_size))
    buckets: DefaultDict[int, List[Dict]] = defaultdict(list)
    for ex in pretokenized:
        key = int((ex["seq_len"] - 1) // bucket_size)
        buckets[key].append(ex)

    feats_by_layer: Dict[int, List[np.ndarray]] = {i: [] for i in layers}
    labels: List[Tuple[int, str, str, int, int]] = []  # (example_id, regime, p_value, offset_from_split, token_abs_index)
    debug_lines: List[str] = []

    batch_idx_global = 0
    for key in sorted(buckets.keys()):
        group = buckets[key]
        # Mini-batch inside each length bucket
        for start in tqdm(range(0, len(group), args.batch_size),
                          total=(len(group) + args.batch_size - 1)//args.batch_size,
                          desc=f"bucket~len<{(key+1)*bucket_size}>"):
            chunk = group[start:start+args.batch_size]

            token_tensors = [ex["tokens"] for ex in chunk]  # each [1, L_i]
            split_tok = [ex["split"] for ex in chunk]
            sel_ranges = [(ex["split"], ex["sel_end"]) for ex in chunk]
            seq_lens = [ex["seq_len"] for ex in chunk]
            regimes = [ex["regime"] for ex in chunk]
            pvals = [ex["p_value"] for ex in chunk]
            ex_ids = [ex["id"] for ex in chunk]

            # Stack to batch and move to device
            batch_tokens = pad_and_stack(token_tensors, pad_id=pad_id).to(model.cfg.device)

            # Forward with restricted cache
            with torch.no_grad():
                _logits, cache = model.run_with_cache(
                    batch_tokens, remove_batch_dim=False, names_filter=names_filter
                )

            # Collect activations per layer & example
            for lyr in layers:
                acts = cache[hook_point, lyr]  # [B, max_L, d_model]
                for i, (a, b) in enumerate(sel_ranges):
                    sel = acts[i, a:b, :].detach().cpu().numpy()
                    feats_by_layer[lyr].append(sel)
                del acts  # shorten GPU lifetime

            # Labels + safe debug decode (no padded tail)
            for i, (ex_id, regime, p_val, (a, b), L) in enumerate(zip(ex_ids, regimes, pvals, sel_ranges, seq_lens)):
                for k, tok_idx in enumerate(range(a, b)):
                    labels.append((ex_id, regime, p_val, k, tok_idx))

                if len(debug_lines) < cfg.MAX_DEBUG:
                    row = batch_tokens[i]
                    pre_text  = decode_slice(model, row, 0, a)
                    mid_text  = decode_slice(model, row, a, b)
                    post_text = decode_slice(model, row, b, L)  # <- stop at true length
                    dbg = pre_text + "[[" + mid_text + "]]" + post_text
                    if len(dbg) > 1200:
                        dbg = dbg[:600] + " ... " + dbg[-600:]
                    header = f"(id={ex_id}, regime={regime}, p={p_val}, split_tok={a}, kept={b-a})"
                    debug_lines.append(header + "\n" + dbg + "\n")

            # ---- end-of-batch cleanup to avoid VRAM growth ----
            try:
                del cache
            except NameError:
                pass
            try:
                del _logits
            except NameError:
                pass
            del batch_tokens
            # Also drop CPU refs we no longer need in this scope
            token_tensors.clear()

            batch_idx_global += 1
            if (args.device.startswith("cuda")
                and args.empty_cache_every > 0
                and (batch_idx_global % args.empty_cache_every == 0)):
                torch.cuda.empty_cache()

    # Finalize & save
    stacked: Dict[str, np.ndarray] = {}
    for lyr, chunks in feats_by_layer.items():
        if len(chunks) == 0:
            continue
        arr = np.concatenate(chunks, axis=0)  # [N_tokens, d_model]
        stacked[f"acts_{hook_point}_layer{lyr}"] = arr

    run_tag = f"{hook_point}_qwen3_collect"
    datagen_csv, out_dir = _paths()
    ensure_dir(out_dir)

    out_npz = out_dir / f"{run_tag}.npz"
    np.savez_compressed(out_npz, **stacked)

    labels_cols = ["example_id", "regime", "p_value", "offset_from_split", "token_index_in_seq"]
    labels_df = pd.DataFrame(labels, columns=labels_cols)
    labels_csv = out_dir / f"{run_tag}_labels.csv"
    labels_df.to_csv(labels_csv, index=False)

    info = {
        "model_name": args.model_name,
        "device": args.device,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "hook_point": hook_point,
        "layers": layers,
        "batch_size": args.batch_size,
        "bucket_size": args.bucket_size,
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

    debug_path = out_dir / f"{run_tag}_debug.txt"
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("\n".join(debug_lines))

    print(f"Saved features: {out_npz}")
    print(f"Saved labels  : {labels_csv}")
    print(f"Saved info    : {info_json}")
    print(f"Saved debug   : {debug_path}")

if __name__ == "__main__":
    main()
