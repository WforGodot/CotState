# src/cot/collect/collect.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# Reuse your existing config style; only the I/O paths differ
import collect2_config as cfg
from model_utils import (
    load_tlens_model,
    ensure_dir,
    get_pad_id,
    pad_and_stack,
    make_names_filter,
    decode_slice,
)

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _paths():
    root = _project_root()
    datagen_csv = root / cfg.DATAGEN_CSV_REL  # e.g., cot/outputs/datagen2/proplogic_paragraphs.csv
    out_dir = root / cfg.COLLECT_OUT_REL      # e.g., cot/outputs/collect2
    return datagen_csv, out_dir

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect activations on paragraph dataset; write labels in legacy format.")
    p.add_argument("--device", type=str, default=cfg.DEVICE)
    p.add_argument("--dtype", type=str, default=cfg.DTYPE)
    p.add_argument("--hook_point", type=str, default=cfg.HOOK_POINT)
    p.add_argument("--model_name", type=str, default=cfg.MODEL_NAME)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--bucket_size", type=int, default=cfg.LENGTH_BUCKET_SIZE)
    p.add_argument("--empty_cache_every", type=int, default=getattr(cfg, "EMPTY_CACHE_EVERY_N_BATCHES", 0))
    p.add_argument("--trust_remote_code",
                   action="store_true" if cfg.TRUST_REMOTE_CODE else "store_false",
                   default=cfg.TRUST_REMOTE_CODE)
    return p.parse_args()

def _select_rows_by_regime(df: pd.DataFrame, counts: Dict[str, int]) -> pd.DataFrame:
    if "regime" not in df.columns:
        df = df.copy(); df["regime"] = "track_p"
    selected = []
    for regime, n in counts.items():
        if n <= 0: continue
        selected.append(df[df["regime"] == regime].head(n))
    return pd.concat(selected, ignore_index=True) if selected else df.iloc[0:0].copy()

def _token_labels_from_chars(model, paragraph: str, p_char_labels: str, tok_cot: torch.Tensor) -> List[str]:
    """
    Map each cot token to 'T'/'F'/'U' via majority over the decoded characters of that token.
    """
    assert len(paragraph) == len(p_char_labels)
    labels: List[str] = []
    pos = 0
    Lc = int(tok_cot.shape[1])
    from collections import Counter as _Counter
    for j in range(Lc):
        piece = model.to_string(tok_cot[0, j:j+1])
        seg_len = len(piece)
        sub = p_char_labels[pos:pos+seg_len] if seg_len > 0 else ""
        if not sub:
            labels.append("U")
        else:
            cnt = _Counter(sub)
            # deterministic tie-break: prefer T, then F, then U
            label_char = max(["T","F","U"], key=lambda c: (cnt[c], c))
            labels.append(label_char)
        pos += seg_len
    return labels

def main():
    args = parse_args()
    datagen_csv, out_dir = _paths()
    ensure_dir(out_dir)

    # Load paragraph dataset
    if not datagen_csv.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {datagen_csv}")
    df = pd.read_csv(datagen_csv)
    for need in ["id","question","answer","cot","p_char_labels"]:
        if need not in df.columns:
            raise ValueError(f"CSV missing required column '{need}'")

    # Optional regime column for consistency
    if "regime" not in df.columns:
        df["regime"] = "track_p"

    rows = _select_rows_by_regime(df, getattr(cfg, "REGIME_SAMPLE_COUNTS", {"track_p": len(df)}))
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
    try: model.eval()
    except Exception: pass

    n_layers = model.cfg.n_layers
    layers = [i for i in cfg.LAYERS if 0 <= i < n_layers] or [max(0, n_layers // 2)]
    names_filter = make_names_filter(args.hook_point, layers)
    pad_id = get_pad_id(model)

    # Pretokenize & sample tokens
    pretokenized: List[Dict] = []
    rng_tok = np.random.RandomState(getattr(cfg, "SEED", 0))
    sample_frac = float(getattr(cfg, "TOKEN_SAMPLE_FRACTION", 1.0) or 1.0)
    sample_frac = max(0.0, min(1.0, sample_frac))
    rng_split = np.random.RandomState(getattr(cfg, "SPLIT_SEED", 0))
    train_frac = float(getattr(cfg, "TRAIN_FRACTION", 0.8))

    for _, row in rows.iterrows():
        q = str(row["question"])
        cot = str(row["cot"])
        labs = str(row["p_char_labels"])
        ex_id = int(row["id"])
        regime = str(row["regime"])

        # Tokenize
        tok_q = model.to_tokens(q, prepend_bos=True)     # [1, Lq]
        tok_c = model.to_tokens(cot, prepend_bos=False)  # [1, Lc]
        tokens = torch.cat([tok_q, tok_c], dim=1)        # [1, L]
        Lq, Lc = int(tok_q.shape[1]), int(tok_c.shape[1])
        L = int(tokens.shape[1])

        # Per-token labels: question tokens are Undefined (droppable), cot via majority
        cot_labels = _token_labels_from_chars(model, cot, labs, tok_c)

        # Build full-seq labels in legacy string form 'True'/'False' or None for U
        seq_labels: List[Optional[str]] = [None]*Lq + [
            ("True" if c == "T" else "False" if c == "F" else None) for c in cot_labels
        ]

        # Train/test split per example
        split_name = "train" if (rng_split.rand() < train_frac) else "test"

        # Random per-token sampling; keep only defined (True/False) tokens
        keep_idxs: List[int] = []
        if sample_frac >= 1.0:
            cand = [i for i in range(L) if seq_labels[i] is not None]
        else:
            m = rng_tok.rand(L) < sample_frac
            cand = [i for i in range(L) if m[i] and (seq_labels[i] is not None)]
        keep_idxs = cand
        if not keep_idxs:
            continue

        pretokenized.append({
            "id": ex_id,
            "regime": regime,
            "tokens": tokens.cpu(),
            "seq_len": L,
            "keep_idxs": keep_idxs,
            "seq_labels": seq_labels,  # length L, 'True'/'False'/None
            "ex_split": split_name,
        })

    if len(pretokenized) == 0:
        print("No usable examples after tokenization/sampling. Exiting.")
        return

    # Bucket by length
    bucket_size = max(1, int(args.bucket_size))
    buckets: DefaultDict[int, List[Dict]] = defaultdict(list)
    for ex in pretokenized:
        key = int((ex["seq_len"] - 1) // bucket_size)
        buckets[key].append(ex)

    total_kept_tokens = sum(len(ex["keep_idxs"]) for ex in pretokenized)
    if total_kept_tokens == 0:
        print("No tokens selected; exiting without saving.")
        return

    feats_mmap: Dict[int, np.memmap] = {}
    write_idx: Dict[int, int] = {}
    # Legacy columns & semantics:
    # p_value: 'True'/'False'; offset_from_split: token index (compat); token_index_in_seq: token index
    labels_rows: List[Tuple[int, str, str, int, int, str]] = []
    debug_lines: List[str] = []

    batch_idx_global = 0
    for key in sorted(buckets.keys()):
        group = buckets[key]
        for start in tqdm(range(0, len(group), args.batch_size),
                          total=(len(group) + args.batch_size - 1)//args.batch_size,
                          desc=f"bucket~len<{(key+1)*bucket_size}>"):
            chunk = group[start:start+args.batch_size]

            token_tensors = [ex["tokens"] for ex in chunk]
            keep_lists    = [ex["keep_idxs"] for ex in chunk]
            seq_lens      = [ex["seq_len"] for ex in chunk]
            regimes       = [ex["regime"] for ex in chunk]
            ex_ids        = [ex["id"] for ex in chunk]
            ex_splits     = [ex["ex_split"] for ex in chunk]
            seq_lbls      = [ex["seq_labels"] for ex in chunk]

            batch_tokens = pad_and_stack(token_tensors, pad_id=pad_id).to(model.cfg.device)

            with torch.no_grad():
                _logits, cache = model.run_with_cache(
                    batch_tokens, remove_batch_dim=False, names_filter=names_filter
                )

            for lyr in [i for i in layers]:
                acts = cache[args.hook_point, lyr]  # [B, max_L, d_model]
                if lyr not in feats_mmap:
                    d_model = int(acts.shape[-1])
                    mmap_path = out_dir / f".__tmp_{args.hook_point}_layer{lyr}.dat"
                    feats_mmap[lyr] = np.memmap(
                        mmap_path, dtype=np.float16, mode='w+', shape=(total_kept_tokens, d_model)
                    )
                    write_idx[lyr] = 0

                segs: List[torch.Tensor] = []
                for i, keep in enumerate(keep_lists):
                    if keep:
                        idx = torch.tensor(keep, device=acts.device, dtype=torch.long)
                        segs.append(acts[i, idx, :])
                if segs:
                    batch_sel = torch.cat(segs, dim=0).to(dtype=torch.float16)
                    cpu_batch = batch_sel.detach().to("cpu", non_blocking=True).contiguous()
                    np_batch = cpu_batch.numpy()
                    n = np_batch.shape[0]
                    s = write_idx[lyr]; e = s + n
                    feats_mmap[lyr][s:e, :] = np_batch
                    write_idx[lyr] = e
                del acts

            # Labels + debug
            for i, (ex_id, regime, keep, L, split_name, lbls) in enumerate(
                zip(ex_ids, regimes, keep_lists, seq_lens, ex_splits, seq_lbls)
            ):
                for tok_idx in keep:
                    pv = lbls[tok_idx]  # 'True' / 'False'
                    if pv is None:      # should already be filtered out
                        continue
                    labels_rows.append(
                        (ex_id, regime, pv, int(tok_idx), int(tok_idx), split_name)
                    )

                if len(debug_lines) < getattr(cfg, "MAX_DEBUG", 10):
                    row = batch_tokens[i]
                    parts = []
                    keep_set = set(keep)
                    pos = 0
                    while pos < L:
                        start = pos
                        while pos < L and pos not in keep_set:
                            pos += 1
                        if pos > start:
                            parts.append(decode_slice(model, row, start, pos))
                        start = pos
                        while pos < L and pos in keep_set:
                            pos += 1
                        if pos > start:
                            parts.append("[[" + decode_slice(model, row, start, pos) + "]]")
                    dbg = "".join(parts)
                    if len(dbg) > 1200:
                        dbg = dbg[:600] + " ... " + dbg[-600:]
                    header = f"(id={ex_id}, regime={regime}, kept={len(keep)}, split={split_name})"
                    debug_lines.append(header + "\n" + dbg + "\n")

            # cleanup
            try: del cache
            except NameError: pass
            try: del _logits
            except NameError: pass
            del batch_tokens
            token_tensors.clear()

            batch_idx_global += 1
            if (args.device.startswith("cuda")
                and args.empty_cache_every > 0
                and (batch_idx_global % args.empty_cache_every == 0)):
                torch.cuda.empty_cache()

    # Finalize features NPZ
    stacked: Dict[str, np.ndarray] = {}
    for lyr, mmap_arr in feats_mmap.items():
        n_written = write_idx.get(lyr, 0)
        mmap_arr = mmap_arr[:n_written, :]
        stacked[f"acts_{args.hook_point}_layer{lyr}"] = mmap_arr

    run_tag = f"{args.hook_point}_qwen3_collect"  # legacy naming for maximal compatibility
    datagen_csv, out_dir = _paths()
    ensure_dir(out_dir)

    out_npz = out_dir / f"{run_tag}.npz"
    np.savez(out_npz, **stacked)

    # Legacy labels CSV
    labels_cols = ["example_id","regime","p_value","offset_from_split","token_index_in_seq","split"]
    labels_df = pd.DataFrame(labels_rows, columns=labels_cols)
    labels_csv = out_dir / f"{run_tag}_labels.csv"
    labels_df.to_csv(labels_csv, index=False)

    # Info + debug
    counts_by_regime = {r: int((rows["regime"] == r).sum()) for r in rows["regime"].unique()}
    info = {
        "model_name": args.model_name,
        "device": args.device,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "hook_point": args.hook_point,
        "layers": layers,
        "batch_size": args.batch_size,
        "bucket_size": args.bucket_size,
        "total_selected_tokens": int(labels_df.shape[0]),
        "train_tokens": int((labels_df["split"] == 'train').sum()),
        "test_tokens": int((labels_df["split"] == 'test').sum()),
        "token_sample_fraction": float(getattr(cfg, 'TOKEN_SAMPLE_FRACTION', 1.0) or 1.0),
        "unique_examples": int(labels_df["example_id"].nunique()),
        "counts_by_regime": counts_by_regime,
        "inputs_csv": str(datagen_csv),
        "features_file": str(out_npz),
        "labels_file": str(labels_csv),
        "note": "Paragraph dataset collected but labels written in legacy format; undefined tokens dropped.",
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
    n_train = int((labels_df["split"] == 'train').sum())
    n_test  = int((labels_df["split"] == 'test').sum())
    print(f"Train/Test token counts: train={n_train}  test={n_test}  total={len(labels_df)}")

if __name__ == "__main__":
    main()
