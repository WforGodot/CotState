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

# Cache for logits
_BASELINE_CACHE: Dict[str, np.ndarray] = {}


# ---------- Paths / config helpers ----------

def _project_root() -> Path:
    # Points to repository 'src' directory
    return Path(__file__).resolve().parents[2]


def _resolve_paths() -> Tuple[Path, Path, Path]:
    root = _project_root()
    datagen_csv = (root / cfg.DATAGEN_CSV_REL).resolve()
    out_dir = (root / cfg.OUT_DIR_REL).resolve()
    vec_dir_rel = Path(getattr(cfg, 'VECTORS_DIR_REL', ''))
    vectors_dir = (root / vec_dir_rel).resolve()
    return datagen_csv, out_dir, vectors_dir


def _normalize_hook_point(hp_raw: str) -> str:
    hp = (hp_raw or "").strip().lower()
    if hp in {"resid_post", "post", "residpost"}:
        return "resid_post"
    if hp in {"resid_pre", "pre", "residpre"}:
        return "resid_pre"
    return hp  # pass-through for expert usage


# ---------- Data selection ----------

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


# ---------- Vectors / masking ----------

def _load_vector(vec_path: Path, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    v = np.load(vec_path)
    if v.ndim != 1:
        raise ValueError(f"Loaded vector has shape {v.shape}; expected 1D (d_model,)")
    if int(v.shape[0]) != int(d_model):
        raise ValueError(f"Vector dim {v.shape[0]} != model d_model {d_model}")
    vt = torch.as_tensor(v, dtype=dtype, device=device)
    n = vt.norm().clamp_min(1e-12)
    return vt / n


def _attn_mass_tail_to_regions(
    attn_layers: Dict[int, torch.Tensor],
    lens: torch.Tensor,
    block_first_n: int,
    last_m: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagnostics averaged over provided layers:
      1) mass from last token -> firstN
      2) mass from tail queries (last M) -> firstN
      3) mass from tail queries (last M) -> INTERMEDIATE [N .. q_start-1]
    Shapes are handled explicitly to avoid head/broadcast mismatches.
    """
    B = int(lens.shape[0])
    last_to_firstN = []
    tail_to_firstN = []
    tail_to_intermediate = []

    for _, pat in attn_layers.items():
        # pat: [B, H, Q, K]
        B_, H, Q, K = pat.shape
        dev = pat.device
        lens_b = lens.to(dev)                       # [B]
        bpt = torch.minimum(torch.full_like(lens_b, block_first_n), lens_b)  # [B]
        q_last = (lens_b - 1).clamp(min=0)          # [B]
        q_start = torch.clamp(lens_b - int(last_m), min=0)                    # [B]

        # --- key positions & masks ---
        pos_k = torch.arange(K, device=dev).unsqueeze(0).expand(B_, -1)      # [B,K]
        firstN_k = (pos_k < bpt.unsqueeze(1))                                 # [B,K]
        inter_k  = (pos_k >= bpt.unsqueeze(1)) & (pos_k < q_start.unsqueeze(1))  # [B,K]

        # --- 1) last token -> firstN ---
        # Gather per-example last query row across the Q dimension
        idx_last = q_last.view(B_, 1, 1, 1).expand(B_, H, 1, K)               # [B,H,1,K]
        last_rows = torch.gather(pat, 2, idx_last)                            # [B,H,1,K]
        mask_bhqk = firstN_k[:, None, None, :].expand(B_, H, 1, K)            # [B,H,1,K]
        mass_last_firstN = (last_rows * mask_bhqk).sum(dim=-1).mean(dim=1).squeeze(-1)  # [B]
        last_to_firstN.append(mass_last_firstN.detach().to("cpu").numpy())

        # --- mean over heads for tail computations ---
        mh = pat.mean(dim=1)                                                  # [B,Q,K]

        # Tail query mask per example
        pos_q = torch.arange(Q, device=dev).unsqueeze(0).expand(B_, -1)       # [B,Q]
        tail_q = (pos_q >= q_start.unsqueeze(1)) & (pos_q < lens_b.unsqueeze(1))
        num_tail = torch.clamp(lens_b - q_start, min=1)                        # [B]

        # Average attention from tail queries -> keys
        mh_tail = (mh * tail_q.unsqueeze(-1)).sum(dim=1) / num_tail.unsqueeze(-1)  # [B,K]

        # --- 2) tail -> firstN ---
        mass_tail_firstN = (mh_tail * firstN_k).sum(dim=-1)                   # [B]
        tail_to_firstN.append(mass_tail_firstN.detach().to("cpu").numpy())

        # --- 3) tail -> intermediate [N .. q_start-1] ---
        mass_tail_inter = (mh_tail * inter_k).sum(dim=-1)                     # [B]
        tail_to_intermediate.append(mass_tail_inter.detach().to("cpu").numpy())

    if not last_to_firstN:
        zeros = np.zeros(B, dtype=float)
        return zeros, zeros, zeros

    last_to_firstN = np.mean(np.stack(last_to_firstN, axis=0), axis=0)
    tail_to_firstN = np.mean(np.stack(tail_to_firstN, axis=0), axis=0)
    tail_to_intermediate = np.mean(np.stack(tail_to_intermediate, axis=0), axis=0)
    return last_to_firstN, tail_to_firstN, tail_to_intermediate



# ---------- Main run (one experiment) ----------

def _run_single():
    ap = argparse.ArgumentParser(description="Layer ablation with tail-only attention blocking (baseline and ablated)")
    ap.add_argument("--n_samples", type=int, default=None)
    args = ap.parse_args()

    datagen_csv, out_dir, vectors_dir = _resolve_paths()
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

    # Layers
    layers_cfg = getattr(cfg, 'LAYERS', None)
    if layers_cfg is None or (isinstance(layers_cfg, list) and len(layers_cfg) and isinstance(layers_cfg[0], (list, tuple))):
        # If the top-level config contains list[list[int]], the main() will iterate experiments.
        layers = list(range(int(model.cfg.n_layers)))
    else:
        layers = [int(x) for x in (layers_cfg if layers_cfg else [getattr(cfg, 'LAYER', 0)])]
    if len(layers) == 0:
        raise ValueError("No layers selected; set LAYERS (list) or LAYER (int) in config.")

    # Attention-blocking knobs
    block_first_n = int(getattr(cfg, 'ATTN_BLOCK_PREFIX_TOKENS', 30))
    last_m = int(getattr(cfg, 'ATTN_BLOCK_QUERY_LAST_M', 1))
    topk_early = int(getattr(cfg, 'TOPK_EARLY', 0))
    relay_hops = int(getattr(cfg, 'BLOCK_TAIL_RELAY_HOPS', 0))

    # Hook point for residual ablation
    hook_point_norm = _normalize_hook_point(str(getattr(cfg, "HOOK_POINT", "resid_post")))
    hook_names = {L: f"blocks.{L}.hook_{hook_point_norm}" for L in layers}

    # Vectors
    if not vectors_dir.exists() or not vectors_dir.is_dir():
        raise FileNotFoundError(f"Vector directory not found: {vectors_dir}")
    d_model = int(model.cfg.d_model)
    vec_map: Dict[int, torch.Tensor] = {}
    missing_layers: List[int] = []
    for L in layers:
        f = vectors_dir / f"L{L}_top1.npy"
        if not f.exists():
            missing_layers.append(L)
            continue
        vec_map[L] = _load_vector(f, d_model=d_model, device=model.cfg.device, dtype=model.cfg.dtype)
    if missing_layers:
        print(f"[WARN] Missing vectors for layers: {missing_layers} — ablation will skip these.")
        layers = [L for L in layers if L in vec_map]
    if len(layers) == 0:
        raise FileNotFoundError("No vectors found for any selected layers.")

    # Ablation mode
    abl_mode = str(getattr(cfg, 'ABLATED_MODE', 'reflect')).lower()
    gamma = float(getattr(cfg, 'PUSH_GAMMA', 1.0))

    # Control
    ctrl_enabled: bool = bool(getattr(cfg, 'CONTROL_RANDOM_DIR', False))
    ctrl_vec_map: Dict[int, torch.Tensor] = {}
    if ctrl_enabled:
        seed = getattr(cfg, 'CONTROL_SEED', None)
        if seed is None:
            seed = int(getattr(cfg, 'RANDOM_STATE', 1234)) + 1001
        g = torch.Generator(device=str(model.cfg.device))
        try:
            g.manual_seed(int(seed))
        except Exception:
            g = None
        for L in layers:
            if g is not None and str(model.cfg.device).startswith("cuda"):
                r = torch.randn(d_model, device=model.cfg.device, dtype=model.cfg.dtype, generator=g)
            else:
                r = torch.randn(d_model, device=model.cfg.device, dtype=model.cfg.dtype)
            r = r / r.norm().clamp_min(1e-12)
            ctrl_vec_map[L] = r

    # Build run directory name
    def _sanitize(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._+-]", "-", s)

    layers_str = "+".join([f"L{L}" for L in layers])
    vec_stem = _sanitize(vectors_dir.name)
    mode_str = f"mode_{abl_mode}" + (f"_g{gamma:.2f}" if abl_mode == 'push' else "")
    ablate_str = f"ablate__{vec_stem}__{layers_str}__{mode_str}"
    block_str = f"attnblock__first{block_first_n}__last{last_m}q" + (f"__topkEarly{topk_early}" if topk_early > 0 else "") + (f"__relay{relay_hops}" if relay_hops > 0 else "")
    base_name = f"{ablate_str}__{block_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = base_name
    k = 1
    while (out_dir / run_name).exists():
        run_name = f"{base_name}_{k}"
        k += 1
    run_dir = out_dir / run_name
    ensure_dir(run_dir)
    report_path = run_dir / "report.txt"
    per_ex_csv = run_dir / "per_example.csv"

    # Pretokenize prefixes
    prior = int(getattr(cfg, 'PRIOR_TOKENS', 2))
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
        Ltok = int(tokens.shape[1])
        if Ltok == 0:
            continue
        start = max(0, Ltok - prior)
        pretokenized.append(dict(
            id=ex_id,
            regime=regime,
            label=label,
            tokens=tokens.cpu(),
            L=Ltok,
            start=start,
            end=Ltok,
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

    head_agg = str(getattr(cfg, 'ATTNV_HEAD_AGG', 'mean')).lower()
    topk_by_attnv = int(getattr(cfg, 'TOPK_BY_ATTNV', 0) or 0)

    rows_out: List[Dict] = []

    for key in tqdm(sorted(buckets.keys()), desc="Buckets", leave=True):
        group = buckets[key]
        for i in tqdm(range(0, len(group), bs), total=(len(group) + bs - 1)//bs,
                      desc=f"bucket~len<{(key+1)*bucket_size}>", leave=False):
            chunk = group[i:i+bs]
            lens = torch.tensor([ex["L"] for ex in chunk], dtype=torch.long)
            starts = torch.tensor([ex["start"] for ex in chunk], dtype=torch.long)

            tok_list = [ex["tokens"] for ex in chunk]
            tokens = pad_and_stack(tok_list, pad_id=pad_id).to(model.cfg.device)  # [B, max_L]
            B = tokens.shape[0]
            max_L = tokens.shape[1]
            ar = torch.arange(B, device=tokens.device)
            idx_last = (lens - 1).to(tokens.device)

            # Build prior-window mask [B,max_L] for residual ablation selection
            pos = torch.arange(max_L, device=tokens.device).unsqueeze(0).expand(B, -1)
            valid = pos < lens.unsqueeze(1).to(tokens.device)
            mask_half_default = (pos >= starts.unsqueeze(1).to(tokens.device)) & valid  # last 'prior' tokens

            # ===== PREPASS: capture baseline patterns & residuals under ATTENTION BLOCK =====
            capture_attn_pre: Dict[int, torch.Tensor] = {}
            capture_resid_base: Dict[int, torch.Tensor] = {}

            # We may also need a per-batch early-key mask (TopK_EARLY) computed from baseline patterns
            def make_cap_pat(Lcur: int):
                def cap(pat, hook):
                    capture_attn_pre[Lcur] = pat.detach()
                    return pat
                return cap

            # Build a plain attention-block hook we can reuse
            def make_block_hook_pre(block_k_mask: torch.Tensor | None):
                def block_hook(pat, hook):
                    # pat: [B, H, Q, K]
                    B_, H, Q, K = pat.shape
                    dev = pat.device
                    lens_b = lens.to(dev)

                    bpt = torch.minimum(torch.full_like(lens_b, block_first_n), lens_b)      # [B]
                    q_start = torch.clamp(lens_b - int(last_m), min=0)                       # [B]
                    q_start_eff = torch.maximum(q_start, bpt)                                 # [B]

                    pos = torch.arange(K, device=dev)
                    q_pos = pos.unsqueeze(0).expand(B_, -1)                                   # [B,Q]
                    k_pos = pos.unsqueeze(0).expand(B_, -1)                                   # [B,K]
                    valid_q = q_pos < lens_b.unsqueeze(1)
                    valid_k = k_pos < lens_b.unsqueeze(1)

                    # Tail queries only
                    block_q = (q_pos >= q_start_eff.unsqueeze(1)) & valid_q                   # [B,Q]

                    # Early keys
                    if block_k_mask is not None:
                        early_k = block_k_mask.to(dev) & valid_k                              # [B,K]
                    else:
                        early_k = (k_pos < bpt.unsqueeze(1)) & valid_k                        # [B,K]

                    # Optional within-tail relay
                    if relay_hops > 0:
                        lower = (q_pos - relay_hops).unsqueeze(2)                             # [B,Q,1]
                        upper = (q_pos - 1).unsqueeze(2)
                        k_b = k_pos.unsqueeze(1)                                              # [B,1,K]
                        relay_block = (k_b >= lower) & (k_b <= upper)                         # [B,Q,K]
                        q_after_firstN = (q_pos >= bpt.unsqueeze(1)) & valid_q                # [B,Q]
                        relay_block = relay_block & q_after_firstN.unsqueeze(2) & valid_k.unsqueeze(1)
                    else:
                        relay_block = torch.zeros((B_, Q, K), dtype=torch.bool, device=dev)

                    mask_qk_early = block_q.unsqueeze(2) & early_k.unsqueeze(1)               # [B,Q,K]
                    mask_qk = mask_qk_early | relay_block

                    pat = pat.masked_fill(mask_qk.unsqueeze(1), 0.0)                          # [B,H,Q,K]
                    row_sum = pat.sum(dim=-1, keepdim=True)
                    pat = pat / (row_sum + 1e-12)
                    return pat
                return block_hook

            # First pass w/o precomputed block_k_mask to build it (if TOPK_EARLY>0)
            fwd_hooks_pre = []
            for L_ in layers:
                # Apply temporary block at each selected layer
                fwd_hooks_pre.append((f"blocks.{L_}.attn.hook_pattern", make_block_hook_pre(block_k_mask=None)))
                # Then capture post-block patterns
                fwd_hooks_pre.append((f"blocks.{L_}.attn.hook_pattern", make_cap_pat(L_)))


            # Also capture baseline residuals at hook point for TOPK_BY_ATTNV scoring
            for L_ in layers:
                def make_resid_cap(Lcur: int, hookname: str):
                    def resid_cap(resid, hook):
                        capture_resid_base[Lcur] = resid.detach()
                        return resid
                    return resid_cap
                fwd_hooks_pre.append((hook_names[L_], make_resid_cap(L_, hook_names[L_])))

            with torch.no_grad():
                _ = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks_pre)

            # Build early-key mask block_k_mask based on tail attention (averaged across selected layers)
            # Average heads -> [B,Q,K], then average tail queries -> [B,K]
            tail_avgs = []
            for L_, pat in capture_attn_pre.items():
                dev = pat.device
                B_, H, Q, K = pat.shape
                lens_b = lens.to(dev)
                q_start = torch.clamp(lens_b - int(last_m), min=0)
                pos_q = torch.arange(Q, device=dev).unsqueeze(0).expand(B_, -1)
                tail_q_mask = (pos_q >= q_start.unsqueeze(1))
                mh = pat.mean(dim=1)  # [B,Q,K]
                num_tail = torch.clamp(lens_b - q_start, min=1)
                mh_tail_avg = (mh * tail_q_mask.unsqueeze(-1)).sum(dim=1) / num_tail.unsqueeze(-1)  # [B,K]
                tail_avgs.append(mh_tail_avg)
            attn_tail_avg = torch.stack(tail_avgs, dim=0).mean(dim=0) if len(tail_avgs) else torch.zeros((B, max_L), device=model.cfg.device)

            # Restrict to first-N keys & pick top-K among them (or block all first-N if topk_early==0)
            pos_k = torch.arange(max_L, device=model.cfg.device).unsqueeze(0).expand(B, -1)
            bpt = torch.minimum(torch.full((B,), block_first_n, device=model.cfg.device), lens.to(model.cfg.device))
            firstN_mask = pos_k < bpt.unsqueeze(1)                                # [B,K]
            block_k_mask = torch.zeros_like(firstN_mask, dtype=torch.bool)        # [B,K]
            if topk_early > 0:
                scores = attn_tail_avg.masked_fill(~firstN_mask, float('-inf'))
                for j in range(B):
                    k_j = int(min(topk_early, int(firstN_mask[j].sum().item())))
                    if k_j > 0:
                        idx_sel = torch.topk(scores[j], k=k_j).indices
                        block_k_mask[j, idx_sel] = True
            else:
                block_k_mask = firstN_mask

            # ===== Build selective prior-window masks per layer (TOPK_BY_ATTNV over prior window) =====
            sel_mask_half_per_layer: Dict[int, torch.Tensor] = {L: mask_half_default for L in layers}
            if topk_by_attnv > 0:
                # Compute |h·v| at hook point from the captured baseline residuals (already under block)
                for L in layers:
                    if (L not in capture_attn_pre) or (L not in capture_resid_base) or (L not in vec_map):
                        sel_mask_half_per_layer[L] = mask_half_default
                        continue
                    pat = capture_attn_pre[L]                       # [B,H,Q,K]
                    resid_b = capture_resid_base[L]                 # [B,max_L,d]
                    # aggregate heads
                    if head_agg == 'max':
                        attn_agg = pat.max(dim=1).values            # [B,Q,K]
                    elif head_agg == 'sum':
                        attn_agg = pat.sum(dim=1)
                    else:
                        attn_agg = pat.mean(dim=1)
                    q_idx = (lens - 1).to(attn_agg.device)
                    ar_b = torch.arange(B, device=attn_agg.device)
                    attn_q = attn_agg[ar_b, q_idx, :]               # [B,K]
                    v_vec = vec_map[L].to(resid_b.device)
                    proj_abs = (resid_b @ v_vec).abs()              # [B,max_L]
                    score = attn_q * proj_abs                       # [B,K]
                    score = score.masked_fill(~mask_half_default, float('-inf'))
                    sel = torch.zeros_like(mask_half_default)
                    for j in range(B):
                        k_j = min(int(topk_by_attnv), int(mask_half_default[j].sum().item()))
                        if k_j <= 0: continue
                        vals_j, idx_j = torch.topk(score[j], k=k_j)
                        sel[j, idx_j] = True
                    sel_mask_half_per_layer[L] = sel

            # Full [B,max_L,1] per layer
            mask_full_per_layer: Dict[int, torch.Tensor] = {L: sel_mask_half_per_layer[L].unsqueeze(-1) for L in layers}

            # ===== BASELINE PASS (attention block only) =====
            capture_attn_base: Dict[int, torch.Tensor] = {}
            def make_cap_base(Lcur: int):
                def cap(pat, hook):
                    capture_attn_base[Lcur] = pat.detach()
                    return pat
                return cap

            fwd_hooks_base = []
            # apply block at each selected layer
            def make_block_hook(block_k_mask_local: torch.Tensor):
                def block_hook(pat, hook):
                    # identical to block above, but with fixed block_k_mask passed in
                    B_, H, Q, K = pat.shape
                    dev = pat.device
                    lens_b = lens.to(dev)
                    bpt = torch.minimum(torch.full_like(lens_b, block_first_n), lens_b)
                    q_start = torch.clamp(lens_b - int(last_m), min=0)
                    q_start_eff = torch.maximum(q_start, bpt)

                    pos = torch.arange(K, device=dev)
                    q_pos = pos.unsqueeze(0).expand(B_, -1)
                    k_pos = pos.unsqueeze(0).expand(B_, -1)
                    valid_q = q_pos < lens_b.unsqueeze(1)
                    valid_k = k_pos < lens_b.unsqueeze(1)

                    block_q = (q_pos >= q_start_eff.unsqueeze(1)) & valid_q
                    early_k = block_k_mask_local.to(dev) & valid_k

                    if relay_hops > 0:
                        lower = (q_pos - relay_hops).unsqueeze(2)
                        upper = (q_pos - 1).unsqueeze(2)
                        k_b = k_pos.unsqueeze(1)
                        relay_block = (k_b >= lower) & (k_b <= upper)
                        q_after_firstN = (q_pos >= bpt.unsqueeze(1)) & valid_q
                        relay_block = relay_block & q_after_firstN.unsqueeze(2) & valid_k.unsqueeze(1)
                    else:
                        relay_block = torch.zeros((B_, Q, K), dtype=torch.bool, device=dev)

                    mask_qk_early = block_q.unsqueeze(2) & early_k.unsqueeze(1)
                    mask_qk = mask_qk_early | relay_block
                    pat = pat.masked_fill(mask_qk.unsqueeze(1), 0.0)
                    row_sum = pat.sum(dim=-1, keepdim=True)
                    pat = pat / (row_sum + 1e-12)
                    return pat
                return block_hook

            for L_ in layers:
                # Order: block, then capture
                fwd_hooks_base.append((f"blocks.{L_}.attn.hook_pattern", make_block_hook(block_k_mask)))
                fwd_hooks_base.append((f"blocks.{L_}.attn.hook_pattern", make_cap_base(L_)))

            with torch.no_grad():
                base_logits_full = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks_base)
            base_logits_last = base_logits_full[ar, idx_last, :]

            # Cache key includes tokens + model + block mask bytes
            tok_cpu = tokens.detach().to('cpu').numpy()
            hsh = hashlib.sha1()
            hsh.update(str(cfg.MODEL_NAME).encode('utf-8'))
            hsh.update(tok_cpu.tobytes())
            hsh.update(block_k_mask.detach().to('cpu').numpy().tobytes())
            key_hash = hsh.hexdigest()
            _BASELINE_CACHE[key_hash] = base_logits_last.detach().to('cpu').numpy()

            # Baseline diagnostics
            base_last_to_firstN, base_tail_to_firstN, base_tail_to_inter = _attn_mass_tail_to_regions(
                capture_attn_base, lens, block_first_n, last_m
            )

            # ===== ABLATED PASS (attention block + residual ablation) =====
            capture_attn_abl: Dict[int, torch.Tensor] = {}
            capture_layer: Dict[int, Dict[str, torch.Tensor]] = {L: {} for L in layers}

            def make_cap_abl(Lcur: int):
                def cap(pat, hook):
                    capture_attn_abl[Lcur] = pat.detach()
                    return pat
                return cap

            fwd_hooks_abl = []
            # attention block (same as baseline), capture after block
            for L_ in layers:
                fwd_hooks_abl.append((f"blocks.{L_}.attn.hook_pattern", make_block_hook(block_k_mask)))
                fwd_hooks_abl.append((f"blocks.{L_}.attn.hook_pattern", make_cap_abl(L_)))

            # residual ablation hooks
            for L in layers:
                v_vec = vec_map[L]
                sel_half = sel_mask_half_per_layer[L]
                mask_full = mask_full_per_layer[L]
                def make_resid_hook(vv: torch.Tensor, Lcur: int, sel_mask_half: torch.Tensor, mask_full_l: torch.Tensor):
                    def hook_fn(resid, hook):
                        # Diagnostics (pre-change) over selected prior tokens
                        proj = resid @ vv                           # [B,max_L]
                        norms = resid.norm(dim=-1)                  # [B,max_L]
                        m = sel_mask_half
                        mf = m.float()
                        counts = mf.sum(dim=1).clamp_min(1.0)
                        mean_proj = (proj * mf).sum(dim=1) / counts
                        mean_abs_proj = (proj.abs() * mf).sum(dim=1) / counts
                        mean_norm = (norms * mf).sum(dim=1) / counts
                        abs_cos = (proj.abs() / norms.clamp_min(1e-12))
                        mean_abs_cos = (abs_cos * mf).sum(dim=1) / counts
                        mean_abs_cos2 = ((abs_cos * abs_cos) * mf).sum(dim=1) / counts
                        capture_layer[Lcur]["mean_proj"] = mean_proj.detach().to("cpu")
                        capture_layer[Lcur]["mean_abs_proj"] = mean_abs_proj.detach().to("cpu")
                        capture_layer[Lcur]["mean_norm"] = mean_norm.detach().to("cpu")
                        capture_layer[Lcur]["mean_abs_cos"] = mean_abs_cos.detach().to("cpu")
                        capture_layer[Lcur]["mean_abs_cos2"] = mean_abs_cos2.detach().to("cpu")

                        # Apply ablation mode on masked positions
                        if abl_mode == 'reflect':
                            new = resid - 2.0 * proj.unsqueeze(-1) * vv
                        elif abl_mode == 'project_out':
                            new = resid - proj.unsqueeze(-1) * vv
                        elif abl_mode == 'push':
                            new = resid + gamma * vv
                        else:
                            new = resid
                        return torch.where(mask_full_l, new, resid)
                    return hook_fn
                fwd_hooks_abl.append((hook_names[L], make_resid_hook(v_vec, L, sel_half, mask_full)))

            with torch.no_grad():
                abl_logits_full = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks_abl)
            abl_logits_last = abl_logits_full[ar, idx_last, :]

            # Ablated diagnostics (after block + ablation)
            abl_last_to_firstN, abl_tail_to_firstN, abl_tail_to_inter = _attn_mass_tail_to_regions(
                capture_attn_abl, lens, block_first_n, last_m
            )

            # ===== CONTROL (random dir) PASS (attention block + control resid change) =====
            ctrl_present = ctrl_enabled
            if ctrl_present:
                fwd_hooks_ctrl = []
                for L_ in layers:
                    fwd_hooks_ctrl.append((f"blocks.{L_}.attn.hook_pattern", make_block_hook(block_k_mask)))

                for L in layers:
                    vv = ctrl_vec_map[L]
                    mask_full_l = mask_full_per_layer[L]
                    def make_ctrl_hook(vvec: torch.Tensor, mask_full_local: torch.Tensor):
                        def hook_fn(resid, hook):
                            if abl_mode == 'reflect':
                                proj = resid @ vvec
                                new = resid - 2.0 * proj.unsqueeze(-1) * vvec
                            elif abl_mode == 'project_out':
                                proj = resid @ vvec
                                new = resid - proj.unsqueeze(-1) * vvec
                            elif abl_mode == 'push':
                                new = resid + gamma * vvec
                            else:
                                new = resid
                            return torch.where(mask_full_local, new, resid)
                        return hook_fn
                    fwd_hooks_ctrl.append((hook_names[L], make_ctrl_hook(vv, mask_full_l)))

                with torch.no_grad():
                    ctrl_logits_full = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks_ctrl)
                ctrl_logits_last = ctrl_logits_full[ar, idx_last, :]
            else:
                ctrl_logits_last = None

            # ===== Scoring: margins / probs (True vs False) =====
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

            # Control metrics
            if ctrl_present and ctrl_logits_last is not None:
                ctrl_true_lse = torch.logsumexp(ctrl_logits_last[:, true_ids], dim=1)
                ctrl_false_lse = torch.logsumexp(ctrl_logits_last[:, false_ids], dim=1)
                ctrl_margin = (ctrl_true_lse - ctrl_false_lse).cpu().numpy()
                ctrl_delta_margin = (ctrl_margin - base_margin)
                ctrl_probs_last = torch.softmax(ctrl_logits_last, dim=-1)
                ctrl_p_true = ctrl_probs_last[:, true_ids].sum(dim=1).cpu().numpy()
                ctrl_p_false = ctrl_probs_last[:, false_ids].sum(dim=1).cpu().numpy()
                ctrl_flipped = (np.sign(base_margin) != np.sign(ctrl_margin)).astype(int)
            else:
                ctrl_margin = np.full(B, np.nan)
                ctrl_delta_margin = np.full(B, np.nan)
                ctrl_p_true = np.full(B, np.nan)
                ctrl_p_false = np.full(B, np.nan)
                ctrl_flipped = np.zeros(B, dtype=int)

            # Aggregate per-layer nudge metrics across layers (means)
            def agg(name: str) -> np.ndarray:
                stacks = [capture_layer[L][name].numpy() for L in layers if name in capture_layer[L]]
                if len(stacks) == 0:
                    return np.zeros(B, dtype=float)
                return np.mean(np.stack(stacks, axis=0), axis=0)

            mean_proj = agg("mean_proj")
            mean_abs_proj = agg("mean_abs_proj")
            mean_norm = agg("mean_norm")
            mean_abs_cos = agg("mean_abs_cos")
            mean_abs_cos2 = agg("mean_abs_cos2")
            var_abs_cos = np.maximum(0.0, (mean_abs_cos2 - (mean_abs_cos ** 2)))

            for j, ex in enumerate(chunk):
                rows_out.append(dict(
                    id=ex["id"],
                    regime=ex["regime"],
                    label=ex["label"],
                    L=int(ex["L"]),
                    start_idx=int(ex["start"]),
                    end_idx=int(ex["end"]),
                    # logits/margins
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
                    # control
                    ctrl_margin=float(ctrl_margin[j]),
                    ctrl_delta_margin=float(ctrl_delta_margin[j]),
                    ctrl_p_true=float(ctrl_p_true[j]),
                    ctrl_p_false=float(ctrl_p_false[j]),
                    ctrl_flipped=int(ctrl_flipped[j]),
                    # attention diagnostics
                    base_attn_last_to_firstN=float(base_last_to_firstN[j]),
                    base_attn_tail_to_firstN=float(base_tail_to_firstN[j]),
                    base_attn_tail_to_intermediate=float(base_tail_to_inter[j]),
                    abl_attn_last_to_firstN=float(abl_last_to_firstN[j]),
                    abl_attn_tail_to_firstN=float(abl_tail_to_firstN[j]),
                    abl_attn_tail_to_intermediate=float(abl_tail_to_inter[j]),
                    # nudge diagnostics (pre-change residual stats on masked prior tokens)
                    mean_hdotv=float(mean_proj[j]),
                    mean_abs_hdotv=float(mean_abs_proj[j]),
                    mean_h_norm=float(mean_norm[j]),
                    mean_abs_cos=float(mean_abs_cos[j]),
                    var_abs_cos=float(var_abs_cos[j]),
                    implied_delta_along_v=float(2.0 * mean_proj[j]),
                    implied_delta_along_v_mag=float(2.0 * mean_abs_proj[j]),
                    implied_frac_change_along_v=float(2.0 * mean_abs_cos[j]),
                    flipped=int(flipped[j]),
                ))

            global_batch_idx += 1
            if (str(model.cfg.device).startswith("cuda") and empty_every > 0 and (global_batch_idx % empty_every == 0)):
                torch.cuda.empty_cache()

    if not rows_out:
        print("[WARN] No results to report.")
        return

    df_out = pd.DataFrame(rows_out)

    # Low-margin mark
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
            # control
            avg_ctrl_delta_margin=float(d["ctrl_delta_margin"].mean()) if "ctrl_delta_margin" in d else float('nan'),
            frac_ctrl_flipped=float(d["ctrl_flipped"].mean()) if "ctrl_flipped" in d else float('nan'),
            avg_ctrl_p_true=float(d["ctrl_p_true"].mean()) if "ctrl_p_true" in d else float('nan'),
            avg_ctrl_p_false=float(d["ctrl_p_false"].mean()) if "ctrl_p_false" in d else float('nan'),
            # attn diagnostics
            avg_base_attn_last_to_firstN=float(d["base_attn_last_to_firstN"].mean()) if "base_attn_last_to_firstN" in d else float('nan'),
            avg_base_attn_tail_to_firstN=float(d["base_attn_tail_to_firstN"].mean()) if "base_attn_tail_to_firstN" in d else float('nan'),
            avg_base_attn_tail_to_intermediate=float(d["base_attn_tail_to_intermediate"].mean()) if "base_attn_tail_to_intermediate" in d else float('nan'),
            avg_abl_attn_last_to_firstN=float(d["abl_attn_last_to_firstN"].mean()) if "abl_attn_last_to_firstN" in d else float('nan'),
            avg_abl_attn_tail_to_firstN=float(d["abl_attn_tail_to_firstN"].mean()) if "abl_attn_tail_to_firstN" in d else float('nan'),
            avg_abl_attn_tail_to_intermediate=float(d["abl_attn_tail_to_intermediate"].mean()) if "abl_attn_tail_to_intermediate" in d else float('nan'),
            # nudge summaries
            avg_mean_abs_hdotv=float(d["mean_abs_hdotv"].mean()) if "mean_abs_hdotv" in d else float('nan'),
            avg_mean_h_norm=float(d["mean_h_norm"].mean()) if "mean_h_norm" in d else float('nan'),
            avg_mean_abs_cos=float(d["mean_abs_cos"].mean()) if "mean_abs_cos" in d else float('nan'),
            avg_var_abs_cos=float(d["var_abs_cos"].mean()) if "var_abs_cos" in d else float('nan'),
            avg_implied_frac_change=float(d["implied_frac_change_along_v"].mean()) if "implied_frac_change_along_v" in d else float('nan'),
        )

    all_sum = _summ(df_out)
    low_sum = _summ(df_out[df_out['is_low_margin'] == 1]) if 'is_low_margin' in df_out.columns else None
    true_sum = _summ(df_out[df_out["label"] == "True"]) if (df_out["label"] == "True").any() else None
    false_sum = _summ(df_out[df_out["label"] == "False"]) if (df_out["label"] == "False").any() else None

    # Report
    lines: List[str] = []
    lines.append(f"Ablation-with-Attention-Block run: {run_name}")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    lines.append(f"Model: {cfg.MODEL_NAME}")
    lines.append(f"Hook point: {hook_point_norm} | Layers: " + ", ".join([str(L) for L in layers]))
    lines.append(f"Vectors dir: {vectors_dir}")
    lines.append(f"Dataset: {datagen_csv}")
    lines.append(f"Regimes: {', '.join(cfg.REGIMES_TO_USE)} | Prior tokens: {prior}")
    lines.append(f"Attention block: first N={block_first_n}, last M queries={last_m}, topkEarly={topk_early}, relay_hops={relay_hops}")
    lines.append(f"Ablation mode: {abl_mode}" + (f" (gamma={gamma})" if abl_mode == 'push' else ""))
    lines.append(f"Samples: {len(df_out)} (requested={cfg.N_SAMPLES})")
    if ctrl_enabled:
        lines.append(f"Control: random direction enabled (seed={getattr(cfg, 'CONTROL_SEED', None) or getattr(cfg, 'RANDOM_STATE', 1234)+1001})")
    low_thr = float(getattr(cfg, 'LOW_MARGIN_ABS_THRESH', 0.4))
    lines.append(f"Low-margin threshold: |base margin| < {low_thr}")
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
        out = [
            f"{title}: n={s['n']}",
            f"  base p(T/F): {s['avg_base_p_true']:.4f} / {s['avg_base_p_false']:.4f}",
            f"  ablated p(T/F): {s['avg_abl_p_true']:.4f} / {s['avg_abl_p_false']:.4f}",
            f"  base margin (T-F): {s['avg_base_margin']:.4f}",
            f"  Δmargin (T-F): {s['avg_delta_margin']:.4f} | flipped: {s['frac_flipped']:.3f}",
            f"  control Δmargin: {s['avg_ctrl_delta_margin']:.4f} | ctrl flipped: {s['frac_ctrl_flipped']:.3f}",
            f"  baseline attn mass: last→firstN {s['avg_base_attn_last_to_firstN']:.4f} | tail→firstN {s['avg_base_attn_tail_to_firstN']:.4f} | tail→intermediate {s['avg_base_attn_tail_to_intermediate']:.4f}",
            f"  ablated   attn mass: last→firstN {s['avg_abl_attn_last_to_firstN']:.4f} | tail→firstN {s['avg_abl_attn_tail_to_firstN']:.4f} | tail→intermediate {s['avg_abl_attn_tail_to_intermediate']:.4f}",
            f"  nudge |cos|: {s['avg_mean_abs_cos']:.4f} | var(|cos|): {s['avg_var_abs_cos']:.4f} | implied frac Δ along v: {s['avg_implied_frac_change']:.4f}",
        ]
        return out

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
    # If cfg.LAYERS is list[list[int]], run one experiment per inner list
    layers_cfg = getattr(cfg, 'LAYERS', None)
    if layers_cfg is not None and len(layers_cfg) > 0 and isinstance(layers_cfg[0], (list, tuple)):
        for exp_idx, layer_list in enumerate(layers_cfg, start=1):
            try:
                cfg.LAYERS = list(map(int, layer_list))  # type: ignore[attr-defined]
            except Exception:
                cfg.LAYERS = layer_list  # best effort
            print(f"[INFO] Running ablation+attnblock experiment {exp_idx} with layers={cfg.LAYERS}")
            _run_single()
    else:
        _run_single()


if __name__ == "__main__":
    main()
