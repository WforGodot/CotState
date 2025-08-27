from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime
import hashlib
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
    get_pad_id,
    pad_and_stack,
)

# Progress bars
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


REGIMES = ["i_initial", "ii_inconsequential", "iii_derived", "iv_indeterminate", "v_output", "vi_single_use", "vii_max_use"]

# Baseline logits cache keyed by a hash of token ids
_BASELINE_CACHE: Dict[str, np.ndarray] = {}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_paths() -> Tuple[Path, Path]:
    root = _project_root()
    datagen_csv = (root / cfg.DATAGEN_CSV_REL).resolve()
    out_dir = (root / cfg.OUT_DIR_REL).resolve()
    return datagen_csv, out_dir


def _get_token_ids(model, strings: List[str]) -> Tuple[List[int], bool, Dict[str, int]]:
    ids: List[int] = []
    any_multi = False
    mapping: Dict[str, int] = {}
    for s in strings:
        toks = model.to_tokens(s, prepend_bos=False)
        tid = int(toks[0, 0].item())
        any_multi = any_multi or (toks.shape[1] > 1)
        mapping[s] = tid
        ids.append(tid)
    # Dedup preserving order
    seen = set()
    dedup: List[int] = []
    for t in ids:
        if t not in seen:
            dedup.append(t)
            seen.add(t)
    return dedup, any_multi, mapping


def _select_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["p_value"] = df["p_value"].astype(str)
    df = df[(df["p_value"] == "True") | (df["p_value"] == "False")]
    regimes = set(cfg.REGIMES_TO_USE) if cfg.REGIMES_TO_USE is not None else set(REGIMES)
    df = df[df["regime"].isin(regimes)]
    if cfg.N_SAMPLES is not None and len(df) > int(cfg.N_SAMPLES):
        df = df.sample(n=int(cfg.N_SAMPLES), random_state=int(cfg.RANDOM_STATE))
    return df.reset_index(drop=True)


def _attn_mass_tail_to_regions(
    attn_layers: Dict[int, torch.Tensor],
    lens: torch.Tensor,
    block_first_n: int,
    last_m: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns three diagnostics (averaged over selected layers & heads):
      1) mass from last token -> firstN
      2) mass from tail queries (last M) -> firstN
      3) mass from tail queries (last M) -> INTERMEDIATE zone [N .. q_start-1]
    """
    B = int(lens.shape[0])
    last_to_firstN = []
    tail_to_firstN = []
    tail_to_intermediate = []

    for L, pat in attn_layers.items():
        # pat: [B, H, Q, K]   (already causal-masked inside model)
        B_, H, Q, K = pat.shape
        dev = pat.device
        lens_b = lens.to(dev)  # [B]
        # Per-example clamps
        bpt = torch.minimum(torch.full_like(lens_b, block_first_n), lens_b)  # [B]
        q_start = torch.clamp(lens_b - int(last_m), min=0)                   # [B]

        # --- 1) last token -> firstN
        pos_k = torch.arange(K, device=dev).unsqueeze(0).expand(B_, -1)                # [B,K]
        mask_firstN_k = (pos_k < bpt.unsqueeze(1))                                     # [B,K]
        last_rows = pat[torch.arange(B_, device=dev)[:, None],
                        torch.arange(H, device=dev)[None, :],
                        (lens_b - 1).unsqueeze(1), :]                                  # [B,H,K]
        mass_last_firstN = (last_rows * mask_firstN_k.unsqueeze(1)).sum(dim=-1).mean(dim=1)  # [B]
        last_to_firstN.append(mass_last_firstN.detach().to("cpu").numpy())

        # --- 2) tail queries -> firstN (avg over q in [q_start .. L-1])
        pos_q = torch.arange(Q, device=dev).unsqueeze(0).expand(B_, -1)                # [B,Q]
        tail_q_mask = pos_q >= q_start.unsqueeze(1)                                    # [B,Q]
        # [B,H,Q,K] masked to tail queries only
        tail_rows = pat * tail_q_mask.unsqueeze(1).unsqueeze(3)                        # [B,H,Q,K]
        # sum over firstN keys
        mass_tail_firstN = (tail_rows * mask_firstN_k.unsqueeze(1).unsqueeze(2)).sum(dim=-1)  # [B,H,Q]
        # average across heads and only over the tail queries actually counted
        num_tail_q = torch.maximum((lens_b - q_start), torch.tensor(1, device=dev))    # [B]
        mass_tail_firstN = mass_tail_firstN.sum(dim=2) / num_tail_q.unsqueeze(1)       # [B,H]
        mass_tail_firstN = mass_tail_firstN.mean(dim=1)                                 # [B]
        tail_to_firstN.append(mass_tail_firstN.detach().to("cpu").numpy())

        # --- 3) tail queries -> intermediate zone [N .. q_start-1]
        # Build per-example intermediate key mask: keys in [bpt .. q_start-1]
        # Start with [B,K] positions
        k_pos = pos_k
        # lower bound is bpt, upper bound exclusive is q_start
        mask_intermediate = (k_pos >= bpt.unsqueeze(1)) & (k_pos < q_start.unsqueeze(1))  # [B,K]
        mass_tail_intermediate = (tail_rows * mask_intermediate.unsqueeze(1).unsqueeze(2)).sum(dim=-1)  # [B,H,Q]
        mass_tail_intermediate = mass_tail_intermediate.sum(dim=2) / num_tail_q.unsqueeze(1)            # [B,H]
        mass_tail_intermediate = mass_tail_intermediate.mean(dim=1)                                     # [B]
        tail_to_intermediate.append(mass_tail_intermediate.detach().to("cpu").numpy())

    if len(last_to_firstN) == 0:
        zeros = np.zeros(B, dtype=float)
        return zeros, zeros, zeros

    # Average across layers
    last_to_firstN = np.mean(np.stack(last_to_firstN, axis=0), axis=0)
    tail_to_firstN = np.mean(np.stack(tail_to_firstN, axis=0), axis=0)
    tail_to_intermediate = np.mean(np.stack(tail_to_intermediate, axis=0), axis=0)
    return last_to_firstN, tail_to_firstN, tail_to_intermediate


def _run_single():
    ap = argparse.ArgumentParser(description="Block attention ONLY for last M queries to the first N keys; measure logit shifts")
    ap.add_argument("--n_samples", type=int, default=None)
    args = ap.parse_args()

    datagen_csv, out_dir = _resolve_paths()
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

    # Decode tokens for True / False
    true_strings = getattr(cfg, 'TRUE_STRINGS', None) or [getattr(cfg, 'TRUE_STR', ' True')]
    false_strings = getattr(cfg, 'FALSE_STRINGS', None) or [getattr(cfg, 'FALSE_STR', ' False')]
    true_ids, true_multi, true_map = _get_token_ids(model, true_strings)
    false_ids, false_multi, false_map = _get_token_ids(model, false_strings)
    if true_multi or false_multi:
        raise ValueError(
            "The strings ' True'/' False' are multi-token for this tokenizer. "
            "Pick single-token forms that match your CoT text, or implement multi-token scoring."
        )

    # Layers to apply attention blocking
    layers_cfg = getattr(cfg, 'LAYERS', None)
    if layers_cfg is None or len(layers_cfg) == 0:
        layers = list(range(int(model.cfg.n_layers)))
    else:
        layers = [int(x) for x in layers_cfg]
    if len(layers) == 0:
        raise ValueError("No layers selected; set LAYERS (list) in config or leave empty to use all layers.")

    # Config knobs
    block_first_n = int(getattr(cfg, 'ATTN_BLOCK_PREFIX_TOKENS', 30))  # first N keys to block
    last_m = int(getattr(cfg, 'ATTN_BLOCK_QUERY_LAST_M', 1))           # only block queries in the last M positions

    # Build run directory
    def _sanitize(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._+-]", "-", s)
    layers_str = "+".join([f"L{L}" for L in layers])
    base_name = f"attnblock_tail__first{block_first_n}__last{last_m}q__{layers_str}"
    run_name = base_name
    k = 1
    while (out_dir / run_name).exists():
        run_name = f"{base_name}_{k}"
        k += 1
    run_dir = out_dir / run_name
    ensure_dir(run_dir)
    report_path = run_dir / "report.txt"
    per_ex_csv = run_dir / "per_example.csv"

    # Pretokenize: keep only the prefix up to p_char_index (same as ablate script)
    pretokenized: List[Dict] = []
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc="Pretokenize/split", leave=False):
        try:
            p_idx = int(r["p_char_index"])
        except Exception:
            continue
        question = str(r["question"])
        cot = str(r["cot"])
        regime = str(r["regime"])
        label = str(r["p_value"])
        ex_id = int(r["id"]) if pd.notna(r["id"]) else -1
        try:
            tok_full, split_idx, _, _, _ = tokenize_with_split(model, question, cot, p_idx)
        except Exception:
            continue
        tokens = tok_full[:, :split_idx]  # prefix only
        L = int(tokens.shape[1])
        if L == 0:
            continue
        pretokenized.append(dict(
            id=ex_id, regime=regime, label=label, tokens=tokens.cpu(), L=L
        ))

    if len(pretokenized) == 0:
        print("[WARN] No usable examples after tokenization/split.")
        return

    # Bucket by length
    bucket_size = max(1, int(getattr(cfg, 'LENGTH_BUCKET_SIZE', 64)))
    buckets: Dict[int, List[Dict]] = {}
    for ex in pretokenized:
        key = int((ex["L"] - 1) // bucket_size)
        buckets.setdefault(key, []).append(ex)

    pad_id = get_pad_id(model)
    bs = max(1, int(getattr(cfg, 'BATCH_SIZE', 32)))
    empty_every = int(getattr(cfg, 'EMPTY_CACHE_EVERY_N_BATCHES', 0) or 0)
    global_batch_idx = 0

    rows_out: List[Dict] = []

    for key in tqdm(sorted(buckets.keys()), desc="Buckets", leave=True):
        group = buckets[key]
        for i in tqdm(range(0, len(group), bs), total=(len(group) + bs - 1)//bs,
                      desc=f"bucket~len<{(key+1)*bucket_size}>", leave=False):
            chunk = group[i:i+bs]
            lens = torch.tensor([ex["L"] for ex in chunk], dtype=torch.long)
            tok_list = [ex["tokens"] for ex in chunk]
            tokens = pad_and_stack(tok_list, pad_id=pad_id).to(model.cfg.device)  # [B, max_L]
            B = tokens.shape[0]
            max_L = tokens.shape[1]
            ar = torch.arange(B, device=tokens.device)
            idx_last = (lens - 1).to(tokens.device)

            # ===== Baseline forward pass & capture baseline attention =====
            capture_attn_base: Dict[int, torch.Tensor] = {}
            hooks_cap = []
            for L_ in layers:
                def make_cap(Lcur: int):
                    def cap(pat, hook):
                        capture_attn_base[Lcur] = pat.detach()
                        return pat
                    return cap
                hooks_cap.append((f"blocks.{L_}.attn.hook_pattern", make_cap(L_)))

            with torch.no_grad():
                base_logits_full = model.run_with_hooks(tokens, fwd_hooks=hooks_cap)
            base_logits_last = base_logits_full[ar, idx_last, :]

            # Cache baseline logits for this batch
            tok_cpu = tokens.detach().to('cpu').numpy()
            h = hashlib.sha1()
            h.update(str(cfg.MODEL_NAME).encode('utf-8'))
            h.update(tok_cpu.tobytes())
            key_hash = h.hexdigest()
            _BASELINE_CACHE[key_hash] = base_logits_last.detach().to('cpu').numpy()

            # Baseline diagnostics
            base_last_to_firstN, base_tail_to_firstN, base_tail_to_inter = _attn_mass_tail_to_regions(
                capture_attn_base, lens, block_first_n, last_m
            )

            # === Build per-example early-key mask (block_k_mask) using baseline patterns ===
            topk_early = int(getattr(cfg, 'TOPK_EARLY', 0))

            with torch.no_grad():
                tail_avgs = []
                for L_, pat in capture_attn_base.items():  # pat: [B, H, Q, K]
                    dev = pat.device
                    B_, H, Q, K = pat.shape
                    lens_b = lens.to(dev)
                    bpt = torch.minimum(torch.full_like(lens_b, block_first_n), lens_b)    # [B]
                    q_start = torch.clamp(lens_b - int(last_m), min=0)                     # [B]

                    # Tail queries mask: q in [q_start .. L-1]
                    pos_q = torch.arange(Q, device=dev).unsqueeze(0).expand(B_, -1)       # [B,Q]
                    tail_q_mask = pos_q >= q_start.unsqueeze(1)                            # [B,Q]

                    # Mean over heads -> [B,Q,K]
                    mh = pat.mean(dim=1)

                    # Average over tail queries per example -> [B,K]
                    num_tail = torch.clamp(lens_b - q_start, min=1)                        # [B]
                    mh_tail_avg = (mh * tail_q_mask.unsqueeze(-1)).sum(dim=1) / num_tail.unsqueeze(-1)
                    tail_avgs.append(mh_tail_avg)

                # Average across selected layers -> [B,K]
                attn_tail_avg = torch.stack(tail_avgs, dim=0).mean(dim=0)

            # Restrict to first-N keys per-example and pick top-K among them
            B, K = attn_tail_avg.shape
            dev = attn_tail_avg.device
            pos_k = torch.arange(K, device=dev).unsqueeze(0).expand(B, -1)                 # [B,K]
            bpt = torch.minimum(torch.full((B,), block_first_n, device=dev), lens.to(dev)) # [B]
            firstN_mask = pos_k < bpt.unsqueeze(1)                                         # [B,K]

            block_k_mask = torch.zeros_like(firstN_mask, dtype=torch.bool)                 # [B,K]
            if topk_early > 0:
                scores = attn_tail_avg.masked_fill(~firstN_mask, float('-inf'))            # [B,K]
                for j in range(B):
                    k_j = int(min(topk_early, int(firstN_mask[j].sum().item())))
                    if k_j > 0:
                        idx = torch.topk(scores[j], k=k_j).indices
                        block_k_mask[j, idx] = True
            else:
                block_k_mask = firstN_mask  # original behavior: block all first-N keys

            # Keep on device for the ablation hook
            block_k_mask = block_k_mask.to(model.cfg.device)    

            # ===== Ablated pass: block only tail queries to first N keys =====
            capture_attn_abl: Dict[int, torch.Tensor] = {}

            fwd_hooks = []
            for L_ in layers:
                def make_block_hook(Lcur: int):
                    def block_hook(pat, hook):
                        # pat: [B, H, Q, K]
                        B_, H, Q, K = pat.shape
                        dev = pat.device
                        lens_b = lens.to(dev)  # [B]

                        bpt = torch.minimum(torch.full_like(lens_b, block_first_n), lens_b)      # [B]
                        q_start = torch.clamp(lens_b - int(last_m), min=0)                       # [B]
                        q_start_effective = torch.maximum(q_start, bpt)                           # [B]

                        # positions
                        pos = torch.arange(K, device=dev)                                        # [K]
                        q_pos = pos.unsqueeze(0).expand(B_, -1)                                   # [B,Q]
                        k_pos = pos.unsqueeze(0).expand(B_, -1)                                   # [B,K]
                        valid_q = q_pos < lens_b.unsqueeze(1)                                     # [B,Q]
                        valid_k = k_pos < lens_b.unsqueeze(1)                                     # [B,K]

                        # Tail queries only: q >= max(bpt, q_start)
                        block_q = (q_pos >= q_start_effective.unsqueeze(1)) & valid_q             # [B,Q]

                        # Early-key block (first N)
                        # If you already computed block_k_mask earlier (TopK), use it; else fall back to first-N
                        if 'block_k_mask' in locals():
                            early_k = block_k_mask.to(dev) & valid_k                              # [B,K]
                        else:
                            early_k = (k_pos < bpt.unsqueeze(1)) & valid_k                        # [B,K]

                        # OPTIONAL: within-tail relay block (keys immediately before q)
                        relay_hops = int(getattr(cfg, 'BLOCK_TAIL_RELAY_HOPS', 0))
                        if relay_hops > 0:
                            # Build [B,Q,K] mask for k in [q-relay_hops, q-1]
                            lower = (q_pos - relay_hops).unsqueeze(2)  # [B,Q,1]
                            upper = (q_pos - 1).unsqueeze(2)           # [B,Q,1]
                            k_b = k_pos.unsqueeze(1)                   # [B,1,K]
                            relay_block = (k_b >= lower) & (k_b <= upper)  # [B,Q,K]

                            # CHANGED: apply relay-hop blocking for ALL queries after first N (q >= bpt)
                            q_after_firstN = (q_pos >= bpt.unsqueeze(1)) & valid_q          # [B,Q]
                            relay_block = relay_block & q_after_firstN.unsqueeze(2)

                            # Also require valid_k
                            relay_block = relay_block & valid_k.unsqueeze(1)
                        else:
                            relay_block = torch.zeros((B_, Q, K), dtype=torch.bool, device=dev)

                        # Combine: early-key block for tail queries, plus optional relay block
                        mask_qk_early = block_q.unsqueeze(2) & early_k.unsqueeze(1)                # [B,Q,K]
                        mask_qk = mask_qk_early | relay_block                                      # [B,Q,K]

                        # Apply and renormalize
                        pat = pat.masked_fill(mask_qk.unsqueeze(1), 0.0)                           # [B,H,Q,K]
                        row_sum = pat.sum(dim=-1, keepdim=True)
                        pat = pat / (row_sum + 1e-12)

                        return pat

                    return block_hook
                fwd_hooks.append((f"blocks.{L_}.attn.hook_pattern", make_block_hook(L_)))

            with torch.no_grad():
                logits_abl_full = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            abl_logits_last = logits_abl_full[ar, idx_last, :]

            # Ablated diagnostics
            abl_last_to_firstN, abl_tail_to_firstN, abl_tail_to_inter = _attn_mass_tail_to_regions(
                capture_attn_abl, lens, block_first_n, last_m
            )

            # ===== Scoring: margins & probabilities (True/False) =====
            base_logits_last = torch.from_numpy(_BASELINE_CACHE[key_hash]).to(tokens.device)
            base_true_lse = torch.logsumexp(base_logits_last[:, true_ids], dim=1)
            base_false_lse = torch.logsumexp(base_logits_last[:, false_ids], dim=1)
            abl_true_lse = torch.logsumexp(abl_logits_last[:, true_ids], dim=1)
            abl_false_lse = torch.logsumexp(abl_logits_last[:, false_ids], dim=1)

            base_probs_last = torch.softmax(base_logits_last, dim=-1)
            abl_probs_last = torch.softmax(abl_logits_last, dim=-1)
            base_p_true = base_probs_last[:, true_ids].sum(dim=1).cpu().numpy()
            base_p_false = base_probs_last[:, false_ids].sum(dim=1).cpu().numpy()
            abl_p_true = abl_probs_last[:, true_ids].sum(dim=1).cpu().numpy()
            abl_p_false = abl_probs_last[:, false_ids].sum(dim=1).cpu().numpy()

            base_margin = (base_true_lse - base_false_lse).cpu().numpy()
            abl_margin = (abl_true_lse - abl_false_lse).cpu().numpy()
            delta_margin = (abl_margin - base_margin)
            flipped = (np.sign(base_margin) != np.sign(abl_margin)).astype(int)

            gold_is_true = np.array([ex["label"] == "True" for ex in chunk], dtype=bool)
            gold_delta = np.where(
                gold_is_true,
                (abl_true_lse - base_true_lse).cpu().numpy(),
                (abl_false_lse - base_false_lse).cpu().numpy(),
            )

            # Write per-example rows
            for j, ex in enumerate(chunk):
                rows_out.append(dict(
                    id=ex["id"],
                    regime=ex["regime"],
                    label=ex["label"],
                    L=int(ex["L"]),
                    # logits & margins
                    base_true=float(base_true_lse[j].item()),
                    base_false=float(base_false_lse[j].item()),
                    abl_true=float(abl_true_lse[j].item()),
                    abl_false=float(abl_false_lse[j].item()),
                    base_margin=float(base_margin[j]),
                    abl_margin=float(abl_margin[j]),
                    delta_margin=float(delta_margin[j]),
                    gold_delta=float(gold_delta[j]),
                    base_p_true=float(base_p_true[j]),
                    base_p_false=float(base_p_false[j]),
                    abl_p_true=float(abl_p_true[j]),
                    abl_p_false=float(abl_p_false[j]),
                    # diagnostics: baseline vs ablated attention masses
                    base_attn_last_to_firstN=float(base_last_to_firstN[j]),
                    base_attn_tail_to_firstN=float(base_tail_to_firstN[j]),
                    base_attn_tail_to_intermediate=float(base_tail_to_inter[j]),
                    abl_attn_last_to_firstN=float(abl_last_to_firstN[j]),
                    abl_attn_tail_to_firstN=float(abl_tail_to_firstN[j]),
                    abl_attn_tail_to_intermediate=float(abl_tail_to_inter[j]),
                    flipped=int(flipped[j]),
                ))

            global_batch_idx += 1
            if (str(model.cfg.device).startswith("cuda") and empty_every > 0 and (global_batch_idx % empty_every == 0)):
                torch.cuda.empty_cache()

    if not rows_out:
        print("[WARN] No results to report.")
        return

    df_out = pd.DataFrame(rows_out)

    # Low-margin mark (same logic)
    low_thr = float(getattr(cfg, 'LOW_MARGIN_ABS_THRESH', 0.4))
    if 'base_margin' in df_out.columns:
        df_out['is_low_margin'] = (df_out['base_margin'].abs() < low_thr).astype(int)
    df_out.to_csv(per_ex_csv, index=False)

    # Aggregates
    def _summ(d: pd.DataFrame) -> Dict[str, float]:
        return dict(
            n=int(len(d)),
            avg_base_margin=float(d["base_margin"].mean()),
            med_base_margin=float(d["base_margin"].median()),
            avg_delta_margin=float(d["delta_margin"].mean()),
            med_delta_margin=float(d["delta_margin"].median()),
            frac_flipped=float(d["flipped"].mean()),
            avg_gold_delta=float(d["gold_delta"].mean()),
            avg_base_p_true=float(d["base_p_true"].mean()) if "base_p_true" in d else float('nan'),
            avg_base_p_false=float(d["base_p_false"].mean()) if "base_p_false" in d else float('nan'),
            avg_abl_p_true=float(d["abl_p_true"].mean()) if "abl_p_true" in d else float('nan'),
            avg_abl_p_false=float(d["abl_p_false"].mean()) if "abl_p_false" in d else float('nan'),
            # diagnostics
            avg_base_attn_last_to_firstN=float(d["base_attn_last_to_firstN"].mean()) if "base_attn_last_to_firstN" in d else float('nan'),
            avg_base_attn_tail_to_firstN=float(d["base_attn_tail_to_firstN"].mean()) if "base_attn_tail_to_firstN" in d else float('nan'),
            avg_base_attn_tail_to_intermediate=float(d["base_attn_tail_to_intermediate"].mean()) if "base_attn_tail_to_intermediate" in d else float('nan'),
            avg_abl_attn_last_to_firstN=float(d["abl_attn_last_to_firstN"].mean()) if "abl_attn_last_to_firstN" in d else float('nan'),
            avg_abl_attn_tail_to_firstN=float(d["abl_attn_tail_to_firstN"].mean()) if "abl_attn_tail_to_firstN" in d else float('nan'),
            avg_abl_attn_tail_to_intermediate=float(d["abl_attn_tail_to_intermediate"].mean()) if "abl_attn_tail_to_intermediate" in d else float('nan'),
        )

    all_sum = _summ(df_out)
    low_sum = _summ(df_out[df_out['is_low_margin'] == 1]) if 'is_low_margin' in df_out.columns else None
    true_sum = _summ(df_out[df_out["label"] == "True"]) if (df_out["label"] == "True").any() else None
    false_sum = _summ(df_out[df_out["label"] == "False"]) if (df_out["label"] == "False").any() else None

    # Report
    lines: List[str] = []
    lines.append(f"Attention-Blocking (tail-only) run: {run_name}")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    lines.append(f"Model: {cfg.MODEL_NAME} | Hooks: " + ", ".join([f"blocks.{L}.attn.hook_pattern" for L in layers]))
    lines.append(f"Dataset: {datagen_csv}")
    lines.append(f"Regimes: {', '.join(cfg.REGIMES_TO_USE)}")
    lines.append(f"Samples: {len(df_out)} (requested={cfg.N_SAMPLES})")
    lines.append(f"Blocked keys: first N={block_first_n} tokens (per-example clamped).")
    lines.append(f"Blocked queries: last M={last_m} tokens (per-example clamped), but only for q ≥ max(N, L−M) to keep causal support non-empty.")
    lines.append("")
    # Tokenization notes
    true_kv = ", ".join([f"{k}→{v}" for k, v in true_map.items()])
    false_kv = ", ".join([f"{k}→{v}" for k, v in false_map.items()])
    lines.append(f"True variants (first-token ids): [{true_kv}]" + (" (some multi-token)" if true_multi else ""))
    lines.append(f"False variants (first-token ids): [{false_kv}]" + (" (some multi-token)" if false_multi else ""))
    lines.append("Probabilities use full softmax over vocab; for multiple variants, masses are summed per class.")
    lines.append("")

    def _fmt_summ(title: str, s: Dict[str, float] | None) -> List[str]:
        if s is None:
            return [f"{title}: n=0"]
        lines = [
            f"{title}: n={s['n']}",
            f"  base p(T/F): {s['avg_base_p_true']:.4f} / {s['avg_base_p_false']:.4f}",
            f"  ablated p(T/F): {s['avg_abl_p_true']:.4f} / {s['avg_abl_p_false']:.4f}",
            f"  base margin (T-F): {s['avg_base_margin']:.4f}",
            f"  Δmargin (T-F): {s['avg_delta_margin']:.4f} | flipped: {s['frac_flipped']:.3f}",
            f"  baseline attn mass: last→firstN {s['avg_base_attn_last_to_firstN']:.4f} | tail→firstN {s['avg_base_attn_tail_to_firstN']:.4f} | tail→intermediate {s['avg_base_attn_tail_to_intermediate']:.4f}",
            f"  ablated   attn mass: last→firstN {s['avg_abl_attn_last_to_firstN']:.4f} | tail→firstN {s['avg_abl_attn_tail_to_firstN']:.4f} | tail→intermediate {s['avg_abl_attn_tail_to_intermediate']:.4f}",
        ]
        return lines

    lines.extend(_fmt_summ("Overall", all_sum))
    lines.append("")
    lines.extend(_fmt_summ("Low-margin", low_sum))
    lines.append("")
    lines.extend(_fmt_summ("Label=True", true_sum))
    lines.append("")
    lines.extend(_fmt_summ("Label=False", false_sum))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report → {report_path}")
    print(f"Per-example CSV → {per_ex_csv}")


def main():
    # Support cfg.LAYERS as list[list[int]] to run multiple experiments
    layers_cfg = getattr(cfg, 'LAYERS', None)
    if layers_cfg is not None and len(layers_cfg) > 0 and isinstance(layers_cfg[0], (list, tuple)):
        for exp_idx, layer_list in enumerate(layers_cfg, start=1):
            try:
                cfg.LAYERS = list(map(int, layer_list))  # type: ignore[attr-defined]
            except Exception:
                cfg.LAYERS = layer_list  # best effort
            print(f"[INFO] Running tail-only attention-blocking experiment {exp_idx} with layers={cfg.LAYERS}")
            _run_single()
    else:
        _run_single()


if __name__ == "__main__":
    main()
