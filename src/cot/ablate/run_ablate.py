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
    """
    Normalize cfg.HOOK_POINT to a canonical TransformerLens hook suffix.

    Supported shorthands:
      - 'resid_post' (default), 'post', 'residpost'
      - 'resid_pre', 'pre', 'residpre'  <-- NEW
    Any other value is passed through verbatim to allow advanced users to
    target other points (e.g., 'mlp_out'), assuming those names exist.
    """
    hp = (hp_raw or "").strip().lower()
    if hp in {"resid_post", "post", "residpost"}:
        return "resid_post"
    if hp in {"resid_pre", "pre", "residpre"}:
        return "resid_pre"
    return hp  # pass-through for expert usage


def _load_vector(vec_path: Path, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    v = np.load(vec_path)
    if v.ndim != 1:
        raise ValueError(f"Loaded vector has shape {v.shape}; expected 1D (d_model,)")
    if int(v.shape[0]) != int(d_model):
        raise ValueError(f"Vector dim {v.shape[0]} != model d_model {d_model}")
    vt = torch.as_tensor(v, dtype=dtype, device=device)
    n = vt.norm().clamp_min(1e-12)
    return vt / n


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
    # Deduplicate preserving order
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


def main():
    ap = argparse.ArgumentParser(description="Ablate learned directions on prior tokens at multiple layers and measure logit shifts")
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

    # Probe tokens for True / False (multiple variants supported)
    true_strings = getattr(cfg, 'TRUE_STRINGS', None) or [getattr(cfg, 'TRUE_STR', ' True')]
    false_strings = getattr(cfg, 'FALSE_STRINGS', None) or [getattr(cfg, 'FALSE_STR', ' False')]
    true_ids, true_multi, true_map = _get_token_ids(model, true_strings)
    false_ids, false_multi, false_map = _get_token_ids(model, false_strings)
    if true_multi or false_multi:
        raise ValueError(
            "The strings ' True'/' False' are multi-token for this tokenizer. "
            "Pick single-token forms that match your CoT text, or implement multi-token scoring."
        )

    # Layers & vectors: load a vector per selected layer
    if not vectors_dir.exists() or not vectors_dir.is_dir():
        raise FileNotFoundError(f"Vector directory not found: {vectors_dir}")
    layers: List[int] = list(getattr(cfg, 'LAYERS', None) or [getattr(cfg, 'LAYER', 0)])
    layers = [int(x) for x in layers]
    if len(layers) == 0:
        raise ValueError("No layers selected; set LAYERS (list) or LAYER (int) in config.")

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
        print(f"[WARN] Missing vectors for layers: {missing_layers} — they will be skipped.")
        layers = [L for L in layers if L in vec_map]
    if len(layers) == 0:
        raise FileNotFoundError("No vectors found for any selected layers.")

    # Hook point (supports 'resid_post' and NEW 'resid_pre')
    hook_point_norm = _normalize_hook_point(str(getattr(cfg, "HOOK_POINT", "resid_post")))

    # Build run name
    def _sanitize(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._+-]", "-", s)
    regs_str = "all" if cfg.REGIMES_TO_USE is None else "+".join(_sanitize(r) for r in cfg.REGIMES_TO_USE)
    mode = str(getattr(cfg, 'ABLATED_MODE', 'reflect')).lower()
    gamma = float(getattr(cfg, 'PUSH_GAMMA', 1.0))
    mode_str = f"mode_{mode}" + (f"_g{gamma:.2f}" if mode == 'push' else "")
    layers_str = "+".join([f"L{L}" for L in layers])
    vec_stem = _sanitize(vectors_dir.name)
    base_name = f"{layers_str}_{hook_point_norm}__vecset_{vec_stem}__regs_{regs_str}__prior{int(cfg.PRIOR_TOKENS)}__{mode_str}__{_sanitize(cfg.MODEL_NAME)}"
    run_name = base_name
    k = 1
    out_dir = out_dir  # rename for clarity
    while (out_dir / run_name).exists():
        k += 1
        run_name = f"{base_name}__{k}"
    run_dir = out_dir / run_name
    ensure_dir(run_dir)
    report_path = run_dir / "report.txt"
    per_ex_csv = run_dir / "per_example.csv"

    # Pretokenize prefixes and compute start/end windows
    prior = int(cfg.PRIOR_TOKENS)
    pretokenized: List[Dict] = []
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc="Pretokenize/split", leave=False):
        try:
            p_idx = int(r["p_char_index"])  # char index into CoT
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
        start = max(0, L - prior)
        pretokenized.append(dict(
            id=ex_id,
            regime=regime,
            label=label,
            tokens=tokens.cpu(),
            L=L,
            start=start,
            end=L,
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
    rows_out: List[Dict] = []
    bs = max(1, int(getattr(cfg, 'BATCH_SIZE', 32)))
    empty_every = int(getattr(cfg, 'EMPTY_CACHE_EVERY_N_BATCHES', 0) or 0)
    global_batch_idx = 0

    # Hook point names for all selected layers (supports resid_pre/resid_post)
    hook_names = {L: f"blocks.{L}.hook_{hook_point_norm}" for L in layers}
    abl_mode = mode
    gamma = float(getattr(cfg, 'PUSH_GAMMA', 1.0))

    for key in tqdm(sorted(buckets.keys()), desc="Buckets", leave=True):
        group = buckets[key]
        for i in tqdm(range(0, len(group), bs), total=(len(group) + bs - 1)//bs,
                      desc=f"bucket~len<{(key+1)*bucket_size}>", leave=False):
            chunk = group[i:i+bs]
            lens = torch.tensor([ex["L"] for ex in chunk], dtype=torch.long)
            starts = torch.tensor([ex["start"] for ex in chunk], dtype=torch.long)
            # Stack tokens to [B, max_L]
            tok_list = [ex["tokens"] for ex in chunk]
            tokens = pad_and_stack(tok_list, pad_id=pad_id).to(model.cfg.device)
            B = tokens.shape[0]
            max_L = tokens.shape[1]

            # Duplicate batch: first half baseline (no change), second half ablated
            tokens2 = torch.cat([tokens, tokens], dim=0)

            # Build mask [2B, max_L, 1] where True marks positions to modify
            pos = torch.arange(max_L, device=tokens2.device).unsqueeze(0).expand(B, -1)
            valid = pos < lens.unsqueeze(1).to(tokens2.device)
            mask_half = (pos >= starts.unsqueeze(1).to(tokens2.device)) & valid
            mask = torch.cat([torch.zeros_like(mask_half), mask_half], dim=0).unsqueeze(-1)

            # Capture dicts per layer
            capture_layer: Dict[int, Dict[str, torch.Tensor]] = {L: {} for L in layers}

            # Build a hook for each selected layer
            fwd_hooks = []
            for L in layers:
                v_vec = vec_map[L]  # [d]

                def make_hook(vv: torch.Tensor, Lcur: int):
                    def hook_fn(resid, hook):
                        # resid: [2B, max_L, d] for both resid_pre and resid_post
                        resid_abl = resid[B:2*B, :, :]
                        proj_abl = resid_abl @ vv                   # [B, max_L]
                        norms_abl = resid_abl.norm(dim=-1)          # [B, max_L]
                        m = mask_half                                # [B, max_L]
                        m_f = m.float()
                        counts = m_f.sum(dim=1).clamp_min(1.0)       # [B]
                        mean_proj = (proj_abl * m_f).sum(dim=1) / counts
                        mean_abs_proj = (proj_abl.abs() * m_f).sum(dim=1) / counts
                        mean_norm = (norms_abl * m_f).sum(dim=1) / counts
                        abs_cos = (proj_abl.abs() / norms_abl.clamp_min(1e-12))
                        mean_abs_cos = (abs_cos * m_f).sum(dim=1) / counts
                        mean_abs_cos2 = ((abs_cos * abs_cos) * m_f).sum(dim=1) / counts

                        # Save per-layer metrics (CPU tensors)
                        capture_layer[Lcur]["mean_proj"] = mean_proj.detach().to("cpu")
                        capture_layer[Lcur]["mean_abs_proj"] = mean_abs_proj.detach().to("cpu")
                        capture_layer[Lcur]["mean_norm"] = mean_norm.detach().to("cpu")
                        capture_layer[Lcur]["mean_abs_cos"] = mean_abs_cos.detach().to("cpu")
                        capture_layer[Lcur]["mean_abs_cos2"] = mean_abs_cos2.detach().to("cpu")

                        # Apply selected ablation mode on masked positions
                        if abl_mode == 'reflect':
                            proj = resid @ vv
                            new = resid - 2.0 * proj.unsqueeze(-1) * vv
                        elif abl_mode == 'project_out':
                            proj = resid @ vv
                            new = resid - proj.unsqueeze(-1) * vv
                        elif abl_mode == 'push':
                            new = resid + gamma * vv
                        else:
                            new = resid
                        return torch.where(mask, new, resid)
                    return hook_fn

                fwd_hooks.append((hook_names[L], make_hook(v_vec, L)))

            with torch.no_grad():
                logits2 = model.run_with_hooks(tokens2, fwd_hooks=fwd_hooks)

            # Gather per-example logits at last real position
            idx = (lens - 1).to(logits2.device)
            ar = torch.arange(B, device=logits2.device)
            base_logits_last = logits2[ar, idx, :]
            abl_logits_last = logits2[ar + B, idx, :]

            # Logits gathered as logsumexp over variant ids for margins
            base_true_lse = torch.logsumexp(base_logits_last[:, true_ids], dim=1)
            base_false_lse = torch.logsumexp(base_logits_last[:, false_ids], dim=1)
            abl_true_lse = torch.logsumexp(abl_logits_last[:, true_ids], dim=1)
            abl_false_lse = torch.logsumexp(abl_logits_last[:, false_ids], dim=1)
            base_true = base_true_lse
            base_false = base_false_lse
            abl_true = abl_true_lse
            abl_false = abl_false_lse

            # Full-vocab probabilities
            base_probs_last = torch.softmax(base_logits_last, dim=-1)
            abl_probs_last = torch.softmax(abl_logits_last, dim=-1)
            base_p_true_t = base_probs_last[:, true_ids].sum(dim=1)
            base_p_false_t = base_probs_last[:, false_ids].sum(dim=1)
            abl_p_true_t = abl_probs_last[:, true_ids].sum(dim=1)
            abl_p_false_t = abl_probs_last[:, false_ids].sum(dim=1)

            base_margin_t = (base_true - base_false)
            abl_margin_t = (abl_true - abl_false)
            base_margin = base_margin_t.cpu().numpy()
            abl_margin = abl_margin_t.cpu().numpy()
            delta_margin = (abl_margin - base_margin)

            base_p_true = base_p_true_t.cpu().numpy()
            base_p_false = base_p_false_t.cpu().numpy()
            abl_p_true = abl_p_true_t.cpu().numpy()
            abl_p_false = abl_p_false_t.cpu().numpy()

            gold_is_true = np.array([ex["label"] == "True" for ex in chunk], dtype=bool)
            gold_delta = np.where(
                gold_is_true,
                (abl_true - base_true).cpu().numpy(),
                (abl_false - base_false).cpu().numpy(),
            )
            flipped = (np.sign(base_margin) != np.sign(abl_margin)).astype(int)

            # Aggregate per-layer "nudge" metrics across layers (simple mean)
            if len(layers) > 0:
                # Stack per-layer metrics to [num_layers, B]
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
                # Variance of |cos| via E[X^2] - (E[X])^2, clipped to 0
                var_abs_cos = np.maximum(0.0, (mean_abs_cos2 - (mean_abs_cos ** 2)))
            else:
                mean_proj = np.zeros(B)
                mean_abs_proj = np.zeros(B)
                mean_norm = np.zeros(B)
                mean_abs_cos = np.zeros(B)
                var_abs_cos = np.zeros(B)

            for j, ex in enumerate(chunk):
                rows_out.append(dict(
                    id=ex["id"],
                    regime=ex["regime"],
                    label=ex["label"],
                    L=int(ex["L"]),
                    start_idx=int(ex["start"]),
                    end_idx=int(ex["end"]),
                    base_true=float(base_true[j].item()),
                    base_false=float(base_false[j].item()),
                    abl_true=float(abl_true[j].item()),
                    abl_false=float(abl_false[j].item()),
                    base_margin=float(base_margin[j]),
                    abl_margin=float(abl_margin[j]),
                    delta_margin=float(delta_margin[j]),
                    gold_delta=float(gold_delta[j]),
                    base_p_true=float(base_p_true[j]),
                    base_p_false=float(base_p_false[j]),
                    abl_p_true=float(abl_p_true[j]),
                    abl_p_false=float(abl_p_false[j]),
                    # Aggregated nudge metrics across layers (pre-ablation at each layer)
                    mean_hdotv=float(mean_proj[j]),
                    mean_abs_hdotv=float(mean_abs_proj[j]),
                    mean_h_norm=float(mean_norm[j]),
                    mean_abs_cos=float(mean_abs_cos[j]),
                    var_abs_cos=float(var_abs_cos[j]),
                    implied_delta_along_v=float(2.0 * mean_proj[j]),         # avg across layers
                    implied_delta_along_v_mag=float(2.0 * mean_abs_proj[j]), # avg across layers
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
    df_out.to_csv(per_ex_csv, index=False)

    # Aggregate summaries
    def _summ(d: pd.DataFrame) -> Dict[str, float]:
        return dict(
            n=int(len(d)),
            avg_base_margin=float(d["base_margin"].mean()),
            med_base_margin=float(d["base_margin"].median()),
            avg_delta_margin=float(d["delta_margin"].mean()),
            med_delta_margin=float(d["delta_margin"].median()),
            frac_flipped=float(d["flipped"].mean()),
            avg_gold_delta=float(d["gold_delta"].mean()),
            avg_base_p_true=float(d["base_p_true"].mean()) if len(d) and "base_p_true" in d else float('nan'),
            avg_base_p_false=float(d["base_p_false"].mean()) if len(d) and "base_p_false" in d else float('nan'),
            avg_abl_p_true=float(d["abl_p_true"].mean()) if len(d) and "abl_p_true" in d else float('nan'),
            avg_abl_p_false=float(d["abl_p_false"].mean()) if len(d) and "abl_p_false" in d else float('nan'),
            # Nudge summaries (aggregated across layers already)
            avg_mean_abs_hdotv=float(d["mean_abs_hdotv"].mean()) if "mean_abs_hdotv" in d else float('nan'),
            avg_mean_h_norm=float(d["mean_h_norm"].mean()) if "mean_h_norm" in d else float('nan'),
            avg_mean_abs_cos=float(d["mean_abs_cos"].mean()) if "mean_abs_cos" in d else float('nan'),
            avg_var_abs_cos=float(d["var_abs_cos"].mean()) if "var_abs_cos" in d else float('nan'),
            avg_implied_frac_change=float(d["implied_frac_change_along_v"].mean()) if "implied_frac_change_along_v" in d else float('nan'),
        )

    all_sum = _summ(df_out)
    true_sum = _summ(df_out[df_out["label"] == "True"]) if (df_out["label"] == "True").any() else None
    false_sum = _summ(df_out[df_out["label"] == "False"]) if (df_out["label"] == "False").any() else None

    # Write report
    lines: List[str] = []
    lines.append(f"Ablation run: {run_name}")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    layers_disp = ", ".join([f"{L}" for L in layers])
    lines.append(f"Model: {cfg.MODEL_NAME} | Hooks: " + ", ".join([f"blocks.{L}.hook_{hook_point_norm}" for L in layers]))
    lines.append(f"Vectors dir: {vectors_dir} (loaded layers: [{layers_disp}])")
    lines.append(f"Dataset: {datagen_csv}")
    lines.append(f"Regimes: {', '.join(cfg.REGIMES_TO_USE)} | Prior tokens: {cfg.PRIOR_TOKENS}")
    lines.append(f"Mode: {mode}" + (f" (gamma={gamma})" if mode == 'push' else ""))
    lines.append(f"Samples: {len(df_out)} (requested={cfg.N_SAMPLES})")
    lines.append("")
    # Tokenization notes (IDs for each provided variant)
    true_kv = ", ".join([f"{k}→{v}" for k, v in true_map.items()])
    false_kv = ", ".join([f"{k}→{v}" for k, v in false_map.items()])
    lines.append(f"True variants (first-token ids): [{true_kv}]" + (" (some multi-token)" if true_multi else ""))
    lines.append(f"False variants (first-token ids): [{false_kv}]" + (" (some multi-token)" if false_multi else ""))
    lines.append("Probabilities use full softmax over vocab; for multiple variants, masses are summed per class.")
    lines.append("")

    def _fmt_summ(title: str, s: Dict[str, float] | None) -> List[str]:
        if s is None:
            return [f"{title}: n=0"]
        return [
            f"{title}: n={s['n']}",
            f"  base p(T/F): {s['avg_base_p_true']:.4f} / {s['avg_base_p_false']:.4f}",
            f"  ablated p(T/F): {s['avg_abl_p_true']:.4f} / {s['avg_abl_p_false']:.4f}",
            f"  base margin (T-F): {s['avg_base_margin']:.4f}",
            f"  Δmargin (T-F): {s['avg_delta_margin']:.4f} | flipped: {s['frac_flipped']:.3f}",
            f"  nudge |cos|: {s['avg_mean_abs_cos']:.4f} | var(nudge |cos|): {s['avg_var_abs_cos']:.4f} | frac change 2|cos|: {s['avg_implied_frac_change']:.4f}",
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
