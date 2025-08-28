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

import collect_config_forcep as cfg
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

REGIMES = ["i_initial", "ii_inconsequential", "iii_derived", "iv_indeterminate", "v_output", "vi_single_use", "vii_max_use"]

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _paths():
    root = _project_root()
    datagen_csv = root / cfg.DATAGEN_CSV_REL
    out_dir = root / cfg.COLLECT_OUT_REL
    return datagen_csv, out_dir

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect activations around the split in CoTs and, for each sampled token, also run a short prefix with a forced ' P' continuation."
    )
    for r in REGIMES:
        p.add_argument(
            f"--{r}_tokens_around",
            type=str,
            default=None,
            help=f"Tokens to keep for {r} as 'before,after' where each is an int and -1 means all.",
        )
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
    # Force-P controls (CLI can override config)
    p.add_argument("--force_p", action="store_true" if cfg.FORCE_P_ENABLED else "store_false", default=cfg.FORCE_P_ENABLED)
    p.add_argument("--force_p_text", type=str, default=cfg.FORCE_P_TEXT)
    p.add_argument("--force_p_max_per_example", type=int, default=getattr(cfg, "FORCE_P_MAX_PER_EXAMPLE", -1),
                   help="If >0, limit how many kept tokens per example get a forced-' P' run (uniformly across the keep set).")
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
    tok_around_by_regime: Dict[str, Tuple[int, int]] = {}
    for r in REGIMES:
        override = _parse_tuple_flag(getattr(args, f"{r}_tokens_around"))
        if override is not None:
            tok_around_by_regime[r] = override
            continue
        after_only_cli = getattr(args, f"{r}_tokens_after")
        if after_only_cli is not None:
            tok_around_by_regime[r] = (0, int(after_only_cli))
            continue
        around_cfg = getattr(cfg, "TOKENS_AROUND_BY_REGIME", {})
        if r in around_cfg:
            before, after = around_cfg[r]
            tok_around_by_regime[r] = (int(before), int(after))
            continue
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
    try:
        model.eval()
    except Exception:
        pass
    n_layers = model.cfg.n_layers
    layers = clamp_layers(cfg.LAYERS, n_layers)
    hook_point = args.hook_point
    names_filter = make_names_filter(hook_point, layers)
    pad_id = get_pad_id(model)

    # Prepare forced-' P' token IDs once
    force_p_enabled = bool(args.force_p)
    if force_p_enabled:
        # TransformerLens: to_tokens(text, prepend_bos=False) -> [1, L]
        forced_tail = model.to_tokens(str(args.force_p_text), prepend_bos=False, move_to_device=False)
        # safety: ensure it's at least one token
        if forced_tail is None or forced_tail.ndim != 2 or forced_tail.shape[1] == 0:
            raise ValueError(f"force_p_text '{args.force_p_text}' encodes to empty token sequence.")
        forced_tail = forced_tail.to("cpu")
        forced_tail_len = int(forced_tail.shape[1])
        forced_target_rel_idx = forced_tail_len - 1  # capture hidden state at last token of the forced tail
    else:
        forced_tail = None
        forced_tail_len = 0
        forced_target_rel_idx = 0

    # Pretokenize and compute keep ranges (CPU)
    pretokenized: List[Dict] = []
    rng_tok = np.random.RandomState(getattr(cfg, 'SEED', 0))
    sample_frac = float(getattr(cfg, 'TOKEN_SAMPLE_FRACTION', 1.0) or 1.0)
    sample_frac = max(0.0, min(1.0, sample_frac))
    rng_split = np.random.RandomState(getattr(cfg, 'SPLIT_SEED', 0))
    train_frac = float(getattr(cfg, 'TRAIN_FRACTION', 0.8))
    max_forced_per_ex = int(getattr(cfg, 'FORCE_P_MAX_PER_EXAMPLE', -1))
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
        # Before: [a_before, split)
        a_before = 0 if (n_before is None or n_before < 0) else max(0, split_idx - int(n_before))
        b_before = split_idx
        # After:  [split, b_after)
        a_after = split_idx
        b_after = seq_len if (n_after is None or n_after < 0) else min(seq_len, split_idx + int(n_after))

        if not (a_before < b_before or a_after < b_after):
            continue

        # Assign example to train/test split
        ex_split = 'train' if (rng_split.rand() < train_frac) else 'test'

        # Build keep indices with optional random subsampling per token
        keep_idxs: List[int] = []
        if b_before > a_before:
            if sample_frac >= 1.0:
                keep_idxs.extend(range(a_before, b_before))
            else:
                n = b_before - a_before
                m = rng_tok.rand(n) < sample_frac
                keep_idxs.extend([a_before + i for i in range(n) if m[i]])
        if b_after > a_after:
            if sample_frac >= 1.0:
                keep_idxs.extend(range(a_after, b_after))
            else:
                n = b_after - a_after
                m = rng_tok.rand(n) < sample_frac
                keep_idxs.extend([a_after + i for i in range(n) if m[i]])

        # Optionally cap how many kept tokens get forced-' P' prefixes
        forced_selection: Optional[List[int]] = None
        if force_p_enabled and keep_idxs:
            if max_forced_per_ex and max_forced_per_ex > 0 and len(keep_idxs) > max_forced_per_ex:
                # uniform subsample without replacement
                sel = rng_tok.choice(len(keep_idxs), size=max_forced_per_ex, replace=False)
                sel.sort()
                forced_selection = [keep_idxs[i] for i in sel.tolist()]
            else:
                forced_selection = list(keep_idxs)

        pretokenized.append({
            "id": ex_id,
            "regime": regime,
            "p_value": p_val,
            "tokens": tokens.cpu(),   # keep on CPU until batching
            "seq_len": seq_len,
            "split": split_idx,
            "a_before": a_before,
            "b_before": b_before,
            "a_after": a_after,
            "b_after": b_after,
            "keep_idxs": keep_idxs,
            "forced_keep_idxs": forced_selection if force_p_enabled else [],
            "ex_split": ex_split,
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

    # Precompute total rows for memmap allocation
    total_orig_rows = 0
    total_forced_rows = 0
    for ex in pretokenized:
        k = int(len(ex.get("keep_idxs", [])))
        total_orig_rows += k
        if force_p_enabled:
            total_forced_rows += int(len(ex.get("forced_keep_idxs", [])))
    total_rows = total_orig_rows + total_forced_rows

    if total_rows == 0:
        print("No tokens selected after sampling; exiting without saving.")
        return

    # Streaming storage: per-layer memmaps (float16)
    feats_mmap: Dict[int, np.memmap] = {}
    write_idx: Dict[int, int] = {}
    labels: List[Tuple[int, str, str, int, int, str, int, str]] = []
    # (example_id, regime, p_value, offset_from_split, token_index_in_seq, split, pair_id, kind)
    # kind in {'orig','forceP'}; for forceP, token_index_in_seq is -1 (sequence differs).

    debug_lines: List[str] = []

    pair_id_global = 0  # pairs original token state with its forced-' P' variant

    batch_idx_global = 0
    for key in sorted(buckets.keys()):
        group = buckets[key]
        for start in tqdm(range(0, len(group), args.batch_size),
                          total=(len(group) + args.batch_size - 1)//args.batch_size,
                          desc=f"bucket~len<{(key+1)*bucket_size}>"):
            chunk = group[start:start+args.batch_size]

            token_tensors = [ex["tokens"] for ex in chunk]  # each [1, L_i]
            split_tok = [ex["split"] for ex in chunk]
            ex_split_names = [ex["ex_split"] for ex in chunk]
            keep_lists = [ex.get("keep_idxs", []) for ex in chunk]
            forced_keep_lists = [ex.get("forced_keep_idxs", []) for ex in chunk]
            seq_lens = [ex["seq_len"] for ex in chunk]
            regimes = [ex["regime"] for ex in chunk]
            pvals = [ex["p_value"] for ex in chunk]
            ex_ids = [ex["id"] for ex in chunk]

            # ========= Pass 1: full sequences, collect originals at kept token positions =========
            batch_tokens = pad_and_stack(token_tensors, pad_id=pad_id).to(model.cfg.device)
            with torch.no_grad():
                _logits, cache = model.run_with_cache(
                    batch_tokens, remove_batch_dim=False, names_filter=names_filter
                )

            for lyr in layers:
                acts = cache[hook_point, lyr]  # [B, max_L, d_model]
                if lyr not in feats_mmap:
                    d_model = int(acts.shape[-1])
                    mmap_path = out_dir / f".__tmp_{hook_point}_layer{lyr}.dat"
                    feats_mmap[lyr] = np.memmap(
                        mmap_path, dtype=np.float16, mode='w+', shape=(total_rows, d_model)
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
                    n = int(cpu_batch.shape[0])
                    s = write_idx[lyr]
                    e = s + n
                    feats_mmap[lyr][s:e, :] = cpu_batch.numpy()
                    write_idx[lyr] = e

                del acts  # free VRAM sooner

            # Labels (orig) + debug
            for i, (ex_id, regime, p_val, keep, L, sp, ex_split) in enumerate(
                zip(ex_ids, regimes, pvals, keep_lists, seq_lens, split_tok, ex_split_names)
            ):
                keep_sorted = list(keep)
                for tok_idx in keep_sorted:
                    offset = int(tok_idx - sp)
                    pair_id = pair_id_global
                    pair_id_global += 1
                    labels.append((ex_id, regime, p_val, offset, tok_idx, ex_split, pair_id, "orig"))

                if len(debug_lines) < cfg.MAX_DEBUG and keep_sorted:
                    row = batch_tokens[i]
                    keep_set = set(keep_sorted)
                    parts = []
                    pos = 0
                    while pos < L:
                        start_pos = pos
                        while pos < L and pos not in keep_set:
                            pos += 1
                        if pos > start_pos:
                            parts.append(decode_slice(model, row, start_pos, pos))
                        start_pos = pos
                        while pos < L and pos in keep_set:
                            pos += 1
                        if pos > start_pos:
                            parts.append("[[" + decode_slice(model, row, start_pos, pos) + "]]")
                    dbg = "".join(parts)
                    if len(dbg) > 1200:
                        dbg = dbg[:600] + " ... " + dbg[-600:]
                    header = f"(id={ex_id}, regime={regime}, p={p_val}, ex_split={ex_split}, split_tok={sp}, kept={len(keep_sorted)})"
                    debug_lines.append(header + "\n" + dbg + "\n")

            try:
                del cache
            except NameError:
                pass
            try:
                del _logits
            except NameError:
                pass
            del batch_tokens
            token_tensors.clear()

            # ========= Pass 2: forced-' P' short prefixes (efficient) =========
            if force_p_enabled:
                forced_prefixes: List[torch.Tensor] = []
                forced_targets: List[int] = []
                forced_meta: List[Tuple[int, str, str, int, int, str, int]] = []
                # meta: (ex_id, regime, p_val, offset_from_split, orig_tok_idx, ex_split, pair_id)

                # Build in the SAME order as orig labels (so pair_id aligns)
                # We need to reconstruct the pair_id order we used above.
                # We recompute it by iterating chunk in the same nested order and counting.
                # To do that, we need a local cursor over labels we just appended. But simpler:
                # rebuild pair ids using a second counter that mirrors pair_id_global advance.
                # We kept pair_id_global advancing per kept token; the starting base for this chunk is (pair_id_global - num_kept_in_chunk).
                # Compute num kept in chunk:
                num_kept_in_chunk = sum(len(k) for k in keep_lists)
                base_pair_id = pair_id_global - num_kept_in_chunk
                running = 0

                for i, (ex_id, regime, p_val, forced_keep, sp, L, ex_split, full_tokens) in enumerate(
                    zip(ex_ids, regimes, pvals, forced_keep_lists, split_tok, seq_lens, ex_split_names, [ex["tokens"] for ex in chunk])
                ):
                    if not forced_keep:
                        continue
                    # Ensure CPU
                    full_tokens = full_tokens  # [1, L]
                    for tok_idx in forced_keep:
                        pair_id = base_pair_id + running
                        running += 1
                        prefix_len = int(tok_idx + 1)
                        prefix = full_tokens[:, :prefix_len]  # [1, prefix_len]
                        # concat ' P' tail
                        forced_seq = torch.cat([prefix, forced_tail], dim=1)  # [1, prefix_len + T]
                        forced_prefixes.append(forced_seq)
                        forced_targets.append(prefix_len + forced_target_rel_idx)
                        offset = int(tok_idx - sp)
                        forced_meta.append((ex_id, regime, p_val, offset, tok_idx, ex_split, pair_id))

                if forced_prefixes:
                    forced_batch = pad_and_stack(forced_prefixes, pad_id=pad_id).to(model.cfg.device)
                    with torch.no_grad():
                        _logits2, cache2 = model.run_with_cache(
                            forced_batch, remove_batch_dim=False, names_filter=names_filter
                        )

                    for lyr in layers:
                        acts = cache2[hook_point, lyr]  # [Bf, max_Lf, d_model]
                        # gather the target positions
                        tgt = torch.tensor(forced_targets, device=acts.device, dtype=torch.long)
                        batch_sel = acts[torch.arange(acts.shape[0], device=acts.device), tgt, :]
                        cpu_batch = batch_sel.to(dtype=torch.float16).detach().to("cpu", non_blocking=True).contiguous()
                        n = int(cpu_batch.shape[0])
                        s = write_idx[lyr]
                        e = s + n
                        feats_mmap[lyr][s:e, :] = cpu_batch.numpy()
                        write_idx[lyr] = e
                        del acts

                    # Append labels (forceP)
                    for (ex_id, regime, p_val, offset, orig_tok_idx, ex_split, pair_id) in forced_meta:
                        labels.append((ex_id, regime, p_val, offset, -1, ex_split, pair_id, "forceP"))

                    try:
                        del cache2
                    except NameError:
                        pass
                    try:
                        del _logits2
                    except NameError:
                        pass
                    del forced_batch
                    forced_prefixes.clear()
                    forced_targets.clear()
                    forced_meta.clear()

            batch_idx_global += 1
            if (args.device.startswith("cuda")
                and args.empty_cache_every > 0
                and (batch_idx_global % args.empty_cache_every == 0)):
                torch.cuda.empty_cache()

    # Finalize & save
    stacked: Dict[str, np.ndarray] = {}
    for lyr, mmap_arr in feats_mmap.items():
        # Sanity: ensure we've written the expected number of rows
        expected = total_rows
        written = write_idx.get(lyr, 0)
        if written != expected:
            # Trim or pad
            if written < expected:
                pad_rows = expected - written
                d_model = mmap_arr.shape[1]
                mmap_arr[written:expected, :] = np.zeros((pad_rows, d_model), dtype=np.float16)
            else:
                mmap_arr = mmap_arr[:expected, :]
        stacked[f"acts_{hook_point}_layer{lyr}"] = mmap_arr

    run_tag = f"{hook_point}_qwen3_collect_forcep"
    datagen_csv, out_dir = _paths()
    ensure_dir(out_dir)

    out_npz = out_dir / f"{run_tag}.npz"
    np.savez(out_npz, **stacked)

    # Best-effort cleanup of temporary memmap backing files
    for _lyr, mmap_arr in feats_mmap.items():
        try:
            fname = getattr(mmap_arr, 'filename', None)
            if fname:
                Path(str(fname)).unlink(missing_ok=True)
        except Exception:
            pass

    labels_cols = ["example_id", "regime", "p_value", "offset_from_split", "token_index_in_seq", "split", "pair_id", "kind"]
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
        "total_rows": int(labels_df.shape[0]),
        "total_orig_rows": int(total_orig_rows),
        "total_forced_rows": int(total_forced_rows),
        "token_sample_fraction": float(getattr(cfg, 'TOKEN_SAMPLE_FRACTION', 1.0) or 1.0),
        "unique_examples": int(labels_df["example_id"].nunique()),
        "counts_by_regime": {r: int((rows["regime"] == r).sum()) for r in REGIMES},
        "tokens_around_by_regime": {r: [int(x) for x in tok_around_by_regime[r]] for r in REGIMES},
        "inputs_csv": str(datagen_csv),
        "features_file": str(out_npz),
        "labels_file": str(labels_csv),
        "force_p": force_p_enabled,
        "force_p_text": args.force_p_text,
        "force_p_tail_len": int(forced_tail_len),
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
    n_test = int((labels_df["split"] == 'test').sum())
    print(f"Train/Test token counts: train={n_train}  test={n_test}  total={len(labels_df)}")

if __name__ == "__main__":
    main()
