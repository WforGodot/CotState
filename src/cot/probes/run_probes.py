from __future__ import annotations

import sys
import argparse
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Tuple

# Optional progress bars
try:
    from tqdm.auto import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    _HAS_TQDM = False
    def tqdm(x, *args, **kwargs):  # minimal shim
        return x

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score,
    confusion_matrix
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import probes_config as cfg


def _infer_labels_csv(npz_path: Path) -> Path | None:
    # Try a few common stems
    cand = [
        npz_path.with_suffix("").with_suffix(".csv"),  # replace .npz -> .csv (handles '_labels.npz' → '_labels.csv')
        Path(str(npz_path).replace(".npz", "_labels.csv")),
        npz_path.with_name(npz_path.stem + "_labels.csv"),
    ]
    for p in cand:
        if p.exists():
            return p
    return None


def _read_labels(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    # normalize target
    y_raw = df[cfg.TARGET_COL].astype(str).str.strip().str.lower()
    df["_y"] = (y_raw == cfg.POSITIVE_TOKEN.lower()).astype(int)

    # normalize groups
    if cfg.GROUP_COL not in df.columns:
        raise ValueError(f"Missing '{cfg.GROUP_COL}' in labels CSV")

    # Optional filtering by offset (supports range, max, or equality).
    off_col = cfg.OFFSET_COL
    has_any_filter = (
        cfg.FILTER_OFFSET_EQ is not None
        or getattr(cfg, 'FILTER_OFFSET_MAX', None) is not None
        or getattr(cfg, 'FILTER_OFFSET_RANGE', None) is not None
    )
    if has_any_filter:
        if off_col not in df.columns:
            raise ValueError(
                f"Requested offset filtering but '{off_col}' not present in labels."
            )
        # RANGE (inclusive)
        rng = getattr(cfg, 'FILTER_OFFSET_RANGE', None)
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            lo, hi = rng
            if lo is not None:
                df = df[df[off_col] >= lo]
            if hi is not None:
                df = df[df[off_col] <= hi]
        # MAX (<=)
        maxv = getattr(cfg, 'FILTER_OFFSET_MAX', None)
        if maxv is not None:
            df = df[df[off_col] <= maxv]
        # EQ (==)
        if cfg.FILTER_OFFSET_EQ is not None:
            df = df[df[off_col] == cfg.FILTER_OFFSET_EQ]

    return df


def _layer_keys(npz: np.lib.npyio.NpzFile) -> List[Tuple[int, str]]:
    pat = re.compile(rf"^{re.escape(cfg.HOOK_POINT_PREFIX)}(\d+)$")
    out = []
    for k in npz.files:
        m = pat.match(k)
        if m:
            out.append((int(m.group(1)), k))
    out.sort(key=lambda t: t[0])
    if not out:
        raise ValueError(
            f"No layer keys found with prefix '{cfg.HOOK_POINT_PREFIX}'. Keys present: {list(npz.files)[:5]} ..."
        )
    return out


def _build_pipeline(d_model: int) -> Pipeline:
    # Support either variance target (float in (0,1]) or integer component count
    keep_variance = isinstance(cfg.PCA_VARIANCE, float) and 0 < cfg.PCA_VARIANCE < 1.0
    if keep_variance:
        n_components = cfg.PCA_VARIANCE
    elif isinstance(cfg.PCA_VARIANCE, int):
        n_components = min(int(cfg.PCA_VARIANCE), d_model)
    else:
        n_components = min(cfg.PCA_MAX_COMPONENTS, d_model)
    pca = PCA(
        n_components=n_components,
        # float target variance requires 'full' (or 'auto'); randomized only works with integer n_components
        svd_solver="full" if keep_variance else "randomized",
        random_state=cfg.RANDOM_STATE,
    )

    steps = [
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("pca", pca),
    ]
    if cfg.CLASSIFIER == "ridge":
        clf = RidgeClassifier(alpha=cfg.RIDGE_ALPHA, class_weight="balanced")
    elif cfg.CLASSIFIER == "logreg":
        clf = LogisticRegression(
            max_iter=2000, C=cfg.LOGREG_C, solver="lbfgs", class_weight="balanced", n_jobs=cfg.N_JOBS
        )
    else:
        raise ValueError("cfg.CLASSIFIER must be 'ridge' or 'logreg'")
    steps.append(("clf", clf))
    return Pipeline(steps)

def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> Dict[str, float]:
    out = dict(
        accuracy=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, zero_division=0),
    )
    # Some folds can be single-class after grouping; guard AUROC/AP
    try:
        if y_score is not None and len(np.unique(y_true)) > 1:
            out["auroc"] = roc_auc_score(y_true, y_score)
            out["ap"] = average_precision_score(y_true, y_score)
        else:
            out["auroc"] = float("nan")
            out["ap"] = float("nan")
    except Exception:
        out["auroc"] = float("nan")
        out["ap"] = float("nan")
    return out


def _format_table_row(layer: int, n: int, ncomp: int, m: Dict[str, float]) -> str:
    c = cfg.COL_WIDTHS
    return (
        f"{layer:>{c['layer']}}"
        f"{n:>{c['n']}}"
        f"{ncomp:>{c['comps']}}"
        f"{m['acc']*100:>{c['acc']}.2f}"
        f"{m.get('auroc', float('nan'))*100:>{c['auroc']}.2f}"
        f"{m.get('ap', float('nan'))*100:>{c['ap']}.2f}"
        f"{m['f1']*100:>{c['f1']}.2f}"
    )


def _write_report(
    report_path: Path,
    header_lines: List[str],
    table_lines: List[str],
    per_layer_rows: List[Dict],
    per_regime_summary: List[str],
    n_splits: int,
):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line.rstrip() + "\n")
        f.write("\n")
        f.write(f"Per-layer results (grouped {n_splits}-fold CV, metrics in % unless noted):\n")
        f.write("".join(table_lines))
        f.write("\n")
        if per_regime_summary:
            f.write("Per-regime breakdown for the best layer(s):\n")
            for line in per_regime_summary:
                f.write(line.rstrip() + "\n")


def main():
    # Optional CLI to subsample tokens for faster experimentation
    ap = argparse.ArgumentParser(description="Train layerwise probes on activations")
    ap.add_argument("--n_tokens", type=int, default=None,
                    help="Randomly sample this many tokens from the dataset before training.")
    ap.add_argument("--sample_seed", type=int, default=None,
                    help="Random seed for token sampling (defaults to cfg.RANDOM_STATE).")
    cli_args = ap.parse_args()
    npz_path = (Path(__file__).parent / cfg.ACTS_NPZ).resolve()
    labels_csv = cfg.LABELS_CSV
    if labels_csv is None:
        labels_csv = _infer_labels_csv(npz_path)
        if labels_csv is None:
            raise FileNotFoundError(
                f"Could not infer labels CSV from NPZ path {npz_path}. "
                "Set LABELS_CSV in probes_config.py."
            )
    else:
        labels_csv = (Path(__file__).parent / labels_csv).resolve()

    out_dir = (Path(__file__).parent / cfg.OUT_DIR).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"{cfg.RUN_TAG}__{timestamp}"
    report_txt = run_dir / "report.txt"
    scores_csv = run_dir / "layer_scores.csv"

    df = _read_labels(labels_csv)
    # Optional random token subsample (after any filtering in _read_labels)
    target_n = None
    if getattr(cli_args, 'n_tokens', None) is not None and cli_args.n_tokens > 0:
        target_n = int(cli_args.n_tokens)
    elif getattr(cfg, 'N_TOKENS', None) is not None and int(getattr(cfg, 'N_TOKENS')) > 0:
        target_n = int(getattr(cfg, 'N_TOKENS'))
    if target_n is not None and len(df) > target_n:
        seed = cli_args.sample_seed if getattr(cli_args, 'sample_seed', None) is not None else getattr(cfg, 'RANDOM_STATE', 0)
        df = df.sample(n=target_n, random_state=seed)
    # Keep track of row indices relative to the unfiltered labels CSV, so we can
    # subset activation matrices to match any label filtering (e.g., FILTER_OFFSET_EQ)
    row_idx = df.index.to_numpy()
    y = df["_y"].to_numpy()
    groups = df[cfg.GROUP_COL].to_numpy()

    # Basic sanity
    if df[cfg.GROUP_COL].nunique() < cfg.N_SPLITS:
        raise ValueError(f"Not enough unique {cfg.GROUP_COL} for {cfg.N_SPLITS}-fold GroupKFold.")
    class_counts = np.bincount(y, minlength=2)
    if (class_counts < cfg.MIN_CLASS_COUNT).any():
        print(f"[WARN] Very small class counts: {class_counts.tolist()}")

    gkf = GroupKFold(n_splits=cfg.N_SPLITS)

    npz = np.load(npz_path)
    layers = _layer_keys(npz)
    print(f"[INFO] Loaded acts: {npz_path}")
    print(f"[INFO] Labels: {labels_csv} | N={len(df)} tokens | Positive rate={df['_y'].mean():.3f}")
    present_layer_ids = [idx for idx, _ in layers]
    print(f"[INFO] Detected {len(layers)} layer keys with prefix '{cfg.HOOK_POINT_PREFIX}'.")

    # Optionally restrict to requested layers
    if getattr(cfg, 'LAYERS_TO_TRAIN', None) is not None:
        requested = list(dict.fromkeys(cfg.LAYERS_TO_TRAIN))  # de-dupe, keep order
        present_set = set(present_layer_ids)
        req_set = set(int(x) for x in requested)
        missing = sorted(req_set - present_set)
        if missing:
            print(f"[WARN] Requested layers not found in activations: {missing}. Training on intersection only.")
        keep_set = req_set & present_set
        layers = [t for t in layers if t[0] in keep_set]
        print(f"[INFO] Restricting to {len(layers)} requested layers out of {len(present_layer_ids)} present: {[t[0] for t in layers]}")
        if not layers:
            print("[WARN] No layers selected after filtering; proceeding with no training.")

    # Will accumulate one row per layer
    rows_for_csv: List[Dict] = []
    table_lines: List[str] = []

    # Header for pretty table
    hdr = cfg.COL_WIDTHS
    table_lines.append(
        f"{'layer':>{hdr['layer']}}"
        f"{'N':>{hdr['n']}}"
        f"{'PCs':>{hdr['comps']}}"
        f"{'Acc':>{hdr['acc']}}"
        f"{'AUROC':>{hdr['auroc']}}"
        f"{'AP':>{hdr['ap']}}"
        f"{'F1':>{hdr['f1']}}\n"
    )
    table_lines.append("-" * sum(hdr.values()) + "\n")

    # For per-regime reporting on best layer later
    all_layer_preds: Dict[int, np.ndarray] = {}
    all_layer_scores: Dict[int, np.ndarray] = {}

    layer_iter = tqdm(layers, desc="Layers", leave=True) if _HAS_TQDM else layers
    for layer_idx, key in layer_iter:
        X = npz[key]  # shape [N_tokens, d_model]
        # Align X to any filtering applied to the labels (by row index)
        if len(X) != len(df):
            try:
                X = X[row_idx]
            except Exception as e:
                raise ValueError(
                    f"Length mismatch for layer {layer_idx}: X has {len(X)}, labels have {len(df)}. "
                    f"Also failed to subset activations using filtered label indices."
                ) from e
            if len(X) != len(df):
                raise ValueError(
                    f"After subsetting, length mismatch for layer {layer_idx}: X has {len(X)}, labels have {len(df)}"
                )
        # Upcast once to float32 to speed downstream sklearn ops and reduce implicit upcasts
        X = X.astype(np.float32, copy=False)
        d_model = X.shape[1]
        pipe = _build_pipeline(d_model)

        fold_metrics: List[Dict[str, float]] = []
        # Accumulate out-of-fold predictions for per-regime summary
        oof_pred = np.empty_like(y)
        oof_score = np.full_like(y, fill_value=np.nan, dtype=float)
        ncomp_for_layer: int | None = None

        # Precompute splits to show inner progress
        splits = list(gkf.split(X, y, groups))
        fold_iter = tqdm(splits, desc=f"L{layer_idx} CV", leave=False) if _HAS_TQDM else splits
        for train_idx, test_idx in fold_iter:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            pipe.fit(X_tr, y_tr)
            if ncomp_for_layer is None:
                try:
                    ncomp_for_layer = int(getattr(pipe.named_steps["pca"], "n_components_", 0))
                except Exception:
                    ncomp_for_layer = None
            y_hat = pipe.predict(X_te)

            # decision function / probability if available
            try:
                score = pipe.decision_function(X_te)  # RidgeClassifier exposes this
            except Exception:
                try:
                    proba = pipe.predict_proba(X_te)    # LogisticRegression
                    score = proba[:, 1]
                except Exception:
                    score = None

            m = _safe_metrics(y_te, y_hat, score)
            fold_metrics.append(m)

            oof_pred[test_idx] = y_hat
            if score is not None:
                oof_score[test_idx] = score

        # Aggregate fold metrics
        def _avg(name, default=np.nan):
            vals = [d.get(name, default) for d in fold_metrics]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            return float(np.mean(vals)) if vals else float("nan")

        agg = dict(
            layer=layer_idx,
            n=len(X),
            d_model=d_model,
            pca_var=cfg.PCA_VARIANCE,
            acc=_avg("accuracy"),
            auroc=_avg("auroc"),
            ap=_avg("ap"),
            f1=_avg("f1"),
            classifier=cfg.CLASSIFIER,
        )
        rows_for_csv.append(agg)

        # For report table: use PCA component count from first fold fit
        ncomp = int(ncomp_for_layer) if ncomp_for_layer is not None else int(
            getattr(pipe.named_steps.get("pca", PCA()), "n_components", 0)
        )
        table_lines.append(_format_table_row(layer_idx, len(X), ncomp, agg) + "\n")

        all_layer_preds[layer_idx] = oof_pred
        all_layer_scores[layer_idx] = oof_score

    # Write CSV of raw numbers
    run_dir.mkdir(parents=True, exist_ok=True)
    if rows_for_csv:
        pd.DataFrame(rows_for_csv).sort_values("layer").to_csv(scores_csv, index=False)
    else:
        # Write empty CSV with headers to avoid downstream read errors
        pd.DataFrame(columns=[
            "layer", "n", "d_model", "pca_var", "acc", "auroc", "ap", "f1", "classifier"
        ]).to_csv(scores_csv, index=False)
    print(f"[INFO] Wrote per-layer scores CSV → {scores_csv}")

    # Build header for TXT
    header = []
    header.append(f"Probe run: {cfg.RUN_TAG}")
    header.append(f"Timestamp: {timestamp}")
    header.append(f"NPZ: {npz_path}")
    header.append(f"Labels: {labels_csv}")
    header.append(f"Hook prefix: {cfg.HOOK_POINT_PREFIX}  |  Classifier: {cfg.CLASSIFIER}")
    header.append(f"PCA: keep {cfg.PCA_VARIANCE:.3f} variance (cap {cfg.PCA_MAX_COMPONENTS})")
    header.append(f"Grouped CV: {cfg.N_SPLITS} folds by '{cfg.GROUP_COL}'  |  N={len(df)} tokens, "
                  f"Examples={df[cfg.GROUP_COL].nunique()}, PosRate={df['_y'].mean():.3f}")
    # Record any active filters in the report header
    filters = []
    rng = getattr(cfg, 'FILTER_OFFSET_RANGE', None)
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        lo, hi = rng
        filters.append(f"{cfg.OFFSET_COL} in [{lo if lo is not None else '-inf'}, {hi if hi is not None else '+inf'}]")
    maxv = getattr(cfg, 'FILTER_OFFSET_MAX', None)
    if maxv is not None:
        filters.append(f"{cfg.OFFSET_COL} <= {maxv}")
    if cfg.FILTER_OFFSET_EQ is not None:
        filters.append(f"{cfg.OFFSET_COL} == {cfg.FILTER_OFFSET_EQ}")
    if filters:
        header.append("Filter: " + "; ".join(filters))

    # Per-regime summary (best layer by AUROC then Acc)
    scores_df = pd.read_csv(scores_csv)
    scores_df["auroc"] = scores_df["auroc"].fillna(-1.0)
    best = scores_df.sort_values(["auroc", "acc"], ascending=False).head(3)
    per_regime_lines = []
    if cfg.REGIME_COL in df.columns:
        # Map original label indices -> positional indices within the filtered df
        idx_to_pos = {int(ix): pos for pos, ix in enumerate(df.index.to_numpy())}
        for _, row in best.iterrows():
            lid = int(row["layer"])
            y_hat = all_layer_preds[lid]
            y_score = all_layer_scores[lid]
            per_regime_lines.append(f"\nBest layer candidate: L{lid}  (Acc={row['acc']:.3f}, AUROC={row['auroc']:.3f})")
            for reg, sub in df.groupby(cfg.REGIME_COL):
                orig_idx = sub.index.to_numpy()
                # Convert label indices to positional indices into y_hat/y_score
                pos_idx = np.array([idx_to_pos[i] for i in orig_idx if int(i) in idx_to_pos], dtype=int)
                if pos_idx.size == 0:
                    per_regime_lines.append(f"  - {reg:<18} Acc=nan  AUROC=nan")
                    continue
                acc = accuracy_score(df.iloc[pos_idx]["_y"], y_hat[pos_idx])
                try:
                    y_true_reg = df.iloc[pos_idx]["_y"].to_numpy()
                    y_score_reg = y_score[pos_idx]
                    if np.unique(y_true_reg).size > 1 and np.isfinite(y_score_reg).any():
                        auc = roc_auc_score(y_true_reg, y_score_reg[np.isfinite(y_score_reg)])
                    else:
                        auc = float("nan")
                except Exception:
                    auc = float("nan")
                per_regime_lines.append(f"  - {reg:<18} Acc={acc:.3f}  AUROC={auc:.3f}")

    # Write TXT
    _write_report(report_txt, header, table_lines, rows_for_csv, per_regime_lines, cfg.N_SPLITS)

    print(f"[DONE] Report: {report_txt}")
    print(f"[DONE] Scores: {scores_csv}")


if __name__ == "__main__":
    main()
