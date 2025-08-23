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
    p = argparse.ArgumentParser(description="Collect activations around the split in CoTs (batched, bucketed).")
    for r in REGIMES:
        # New: tuple override "before,after" (e.g., "4,5", "1,-1").
        p.add_argument(
            f"--{r}_tokens_around",
            type=str,
            default=None,
            help=f"Tokens to keep for {r} as 'before,after' where each is an int and -1 means all.",
        )
        # Backward-compatibility: allow specifying only the 'after' count.
        p.add_argument(
            f"--{r}_tokens_after",
            type=int,
            default=None,
            help=f"[Deprecated] Only the 'after' count for {r} (-1 for all).",
        )
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

def _parse_tuple_flag(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if s is None:
        return None
    try:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 2:
            raise ValueError
        return int(parts[0]), int(parts[1])
    except Exception:
        raise ValueError(f"Invalid --*_tokens_around flag value '{s}'. Use 'before,after' with integers.")

def build_selection_maps(args: argparse.Namespace) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, int]]:
    """
    Returns:
      tok_around_by_regime: {'i_initial': (n_before, n_after), ...} where -1 means all on that side
      counts_by_regime: {'i_initial': N_examples, ...}
    """
    tok_around_by_regime: Dict[str, Tuple[int, int]] = {}
    for r in REGIMES:
        # CLI overrides first
        override = _parse_tuple_flag(getattr(args, f"{r}_tokens_around"))
        if override is not None:
            tok_around_by_regime[r] = override
            continue
        after_only_cli = getattr(args, f"{r}_tokens_after")
        if after_only_cli is not None:
            tok_around_by_regime[r] = (0, int(after_only_cli))
            continue
        # Then config: new dict if present
        around_cfg = getattr(cfg, "TOKENS_AROUND_BY_REGIME", {})
        if r in around_cfg:
            before, after = around_cfg[r]
            tok_around_by_regime[r] = (int(before), int(after))
            continue
        # Fallbacks: legacy single-side configs
        after_only = getattr(cfg, "TOKENS_AFTER_BY_REGIME", {}).get(r, -1)
        tok_around_by_regime[r] = (0, int(after_only))

    counts: Dict[str, int] = {}
    for r in REGIMES:
        override = getattr(args, f"{r}_n")
        counts[r] = cfg.REGIME_SAMPLE_COUNTS.get(r, 0) if override is None else override
    return tok_around_by_regime, counts

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
    tok_around_by_regime, counts_by_regime = build_selection_maps(args)
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
        n_before, n_after = tok_around_by_regime.get(regime, (0, -1))
        # Before range [a_before, split)
        a_before = 0 if (n_before is None or n_before < 0) else max(0, split_idx - int(n_before))
        b_before = split_idx
        # After range [split, b_after)
        a_after = split_idx
        b_after = seq_len if (n_after is None or n_after < 0) else min(seq_len, split_idx + int(n_after))
        # Skip if both empty
        if not (a_before < b_before or a_after < b_after):
            continue

        pretokenized.append({
            "id": ex_id,
            "regime": regime,
            "p_value": p_val,
            "tokens": tokens.cpu(),   # keep on CPU until batch forward
            "seq_len": seq_len,
            "split": split_idx,
            "a_before": a_before,
            "b_before": b_before,
            "a_after": a_after,
            "b_after": b_after,
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

    # Streaming storage: per-layer memmaps (float16) to avoid end-of-run concatenation
    feats_mmap: Dict[int, np.memmap] = {}
    write_idx: Dict[int, int] = {}
    total_kept_tokens = 0
    labels: List[Tuple[int, str, str, int, int]] = []  # (example_id, regime, p_value, offset_from_split, token_abs_index)
    debug_lines: List[str] = []

    # Precompute total tokens to keep for preallocation
    for ex in pretokenized:
        total_kept_tokens += int(max(0, ex["b_before"] - ex["a_before"]))
        total_kept_tokens += int(max(0, ex["b_after"] - ex["a_after"]))

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
            before_ranges = [(ex["a_before"], ex["b_before"]) for ex in chunk]
            after_ranges = [(ex["a_after"], ex["b_after"]) for ex in chunk]
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
                # Lazy-init memmaps after first forward when d_model is known
                if lyr not in feats_mmap:
                    d_model = int(acts.shape[-1])
                    # Use memmap per layer to stream writes; float16 to reduce size
                    mmap_path = out_dir / f".__tmp_{hook_point}_layer{lyr}.dat"
                    feats_mmap[lyr] = np.memmap(
                        mmap_path, dtype=np.float16, mode='w+', shape=(total_kept_tokens, d_model)
                    )
                    write_idx[lyr] = 0
                for i, ((ab, bb), (aa, ba)) in enumerate(zip(before_ranges, after_ranges)):
                    # before segment
                    if bb > ab:
                        sel_bef = acts[i, ab:bb, :].detach().cpu().numpy().astype(np.float16, copy=False)
                        n = sel_bef.shape[0]
                        s = write_idx[lyr]
                        e = s + n
                        feats_mmap[lyr][s:e, :] = sel_bef
                        write_idx[lyr] = e
                    # after segment
                    if ba > aa:
                        sel_aft = acts[i, aa:ba, :].detach().cpu().numpy().astype(np.float16, copy=False)
                        n = sel_aft.shape[0]
                        s = write_idx[lyr]
                        e = s + n
                        feats_mmap[lyr][s:e, :] = sel_aft
                        write_idx[lyr] = e
                del acts  # shorten GPU lifetime

            # Labels + safe debug decode (no padded tail)
            for i, (ex_id, regime, p_val, (ab, bb), (aa, ba), L, sp) in enumerate(
                zip(ex_ids, regimes, pvals, before_ranges, after_ranges, seq_lens, split_tok)
            ):
                # before tokens
                for tok_idx in range(ab, bb):
                    offset = int(tok_idx - sp)
                    labels.append((ex_id, regime, p_val, offset, tok_idx))
                # after tokens
                for tok_idx in range(aa, ba):
                    offset = int(tok_idx - sp)
                    labels.append((ex_id, regime, p_val, offset, tok_idx))

                if len(debug_lines) < cfg.MAX_DEBUG:
                    row = batch_tokens[i]
                    a = ab
                    b = bb
                    a2 = aa
                    b2 = ba
                    # Build a view that brackets the kept before and after segments.
                    parts = []
                    # prefix before kept-before
                    parts.append(decode_slice(model, row, 0, a))
                    # kept-before
                    if b > a:
                        parts.append("[[" + decode_slice(model, row, a, b) + "]]")
                    # between before and after (usually empty, as b == a2 == split)
                    if a2 > b:
                        parts.append(decode_slice(model, row, b, a2))
                    # kept-after
                    if b2 > a2:
                        parts.append("[[" + decode_slice(model, row, a2, b2) + "]]")
                    # tail
                    parts.append(decode_slice(model, row, b2, L))
                    dbg = "".join(parts)
                    if len(dbg) > 1200:
                        dbg = dbg[:600] + " ... " + dbg[-600:]
                    kept_total = max(0, bb-ab) + max(0, ba-aa)
                    header = f"(id={ex_id}, regime={regime}, p={p_val}, split_tok={split_tok[i]}, kept={kept_total})"
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
    # Finalize memmaps into NPZ dictionary without compression
    for lyr, mmap_arr in feats_mmap.items():
        # Sanity: ensure we've written the expected number of rows
        if write_idx.get(lyr, 0) != total_kept_tokens:
            # Trim or pad with zeros as a defensive fallback
            n_written = write_idx.get(lyr, 0)
            if n_written < total_kept_tokens:
                # pad
                pad_rows = total_kept_tokens - n_written
                d_model = mmap_arr.shape[1]
                pad = np.zeros((pad_rows, d_model), dtype=np.float16)
                mmap_arr[n_written:total_kept_tokens, :] = pad
            else:
                # trim view
                mmap_arr = mmap_arr[:total_kept_tokens, :]
        stacked[f"acts_{hook_point}_layer{lyr}"] = mmap_arr

    run_tag = f"{hook_point}_qwen3_collect"
    datagen_csv, out_dir = _paths()
    ensure_dir(out_dir)

    out_npz = out_dir / f"{run_tag}.npz"
    # Save uncompressed for speed; keeps key structure the same
    np.savez(out_npz, **stacked)

    # Best-effort cleanup of temporary memmap backing files
    for _lyr, mmap_arr in feats_mmap.items():
        try:
            fname = getattr(mmap_arr, 'filename', None)
            if fname:
                Path(str(fname)).unlink(missing_ok=True)
        except Exception:
            pass

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
        "tokens_around_by_regime": {r: [int(x) for x in tok_around_by_regime[r]] for r in REGIMES},
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
