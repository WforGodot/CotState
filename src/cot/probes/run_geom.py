from __future__ import annotations

import sys
import argparse
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional

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
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression

import probes_config as cfg

# -------------------------
# I/O helpers (same shapes/cols as your current script)
# -------------------------

def _infer_labels_csv(npz_path: Path) -> Path | None:
    cand = [
        npz_path.with_suffix("").with_suffix(".csv"),
        Path(str(npz_path).replace(".npz", "_labels.csv")),
        npz_path.with_name(npz_path.stem + "_labels.csv"),
    ]
    for p in cand:
        if p.exists():
            return p
    return None


def _read_labels(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    # normalize target → {0,1}
    y_raw = df[cfg.TARGET_COL].astype(str).str.strip().str.lower()
    df["_y"] = (y_raw == cfg.POSITIVE_TOKEN.lower()).astype(int)

    # guard grouping column
    if cfg.GROUP_COL not in df.columns:
        raise ValueError(f"Missing '{cfg.GROUP_COL}' in labels CSV")

    # Optional offset filtering (kept identical to your script semantics)
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
        rng = getattr(cfg, 'FILTER_OFFSET_RANGE', None)
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            lo, hi = rng
            if lo is not None:
                df = df[df[off_col] >= lo]
            if hi is not None:
                df = df[df[off_col] <= hi]
        maxv = getattr(cfg, 'FILTER_OFFSET_MAX', None)
        if maxv is not None:
            df = df[df[off_col] <= maxv]
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

# -------------------------
# Supervised low-rank subspace (rank-1 or rank-k)
# -------------------------
class SupervisedSubspace:
    """
    A sklearn-compatible transformer that learns a *supervised* linear subspace:
      - If k==1 and method='lda': uses Linear Discriminant Analysis to get a single
        discriminant direction (binary classification only).
      - If k>=1 and method='pls': uses Partial Least Squares to get k components
        correlated with the target.

    The learned weights `W_std_` are in *standardized* feature space. Use
    `recover_geometry(scale)` to convert to original feature space.
    """
    def __init__(self, k: int = 1, method: str = "lda"):
        self.k = int(k)
        self.method = method.lower()
        self.fitted_ = False
        self.W_std_: Optional[np.ndarray] = None  # (d, k)
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.k < 1:
            raise ValueError("k must be >= 1 for SupervisedSubspace")
        if self.method == "lda":
            if len(np.unique(y)) != 2:
                raise ValueError("LDA subspace requires binary targets")
            if self.k != 1:
                raise ValueError("For binary LDA, k must be 1")
            lda = LinearDiscriminantAnalysis(solver="svd")
            lda.fit(X, y)
            # For solver='svd', `scalings_` gives projection directions (d, n_comp)
            W = np.asarray(lda.scalings_)
            if W.ndim == 1:
                W = W[:, None]
            self.W_std_ = W[:, :1]
            self._model = lda
        elif self.method == "pls":
            pls = PLSRegression(n_components=self.k, scale=False)
            y_reg = y.reshape(-1, 1).astype(float)
            pls.fit(X, y_reg)
            # x_weights_ are (d, k) loadings that define the subspace
            W = np.asarray(pls.x_weights_)
            self.W_std_ = W[:, : self.k]
            self._model = pls
        else:
            raise ValueError("method must be 'lda' or 'pls'")
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("SupervisedSubspace must be fit before transform")
        return X @ self.W_std_

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def recover_geometry(self, scaler: Optional[StandardScaler]) -> np.ndarray:
        """Return the subspace basis in *original* feature space (d, k),
        column-orthonormal (via QR), adjusted for the StandardScaler.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() first")
        W = self.W_std_.copy()
        if scaler is not None and getattr(scaler, 'scale_', None) is not None:
            # Undo standardization: (x_std = (x - mean)/scale) ⇒ w_orig = w_std / scale
            W = W / scaler.scale_.reshape(-1, 1)
        # Orthonormalize columns for clean geometry
        Q, _ = np.linalg.qr(W)
        return Q[:, : self.k]

# -------------------------
# Pipelines
# -------------------------

def _build_pipeline(d_model: int) -> Pipeline:
    """Build a probe pipeline using cfg.CLASSIFIER.

    Supported cfg.CLASSIFIER values:
      - 'ridge'  : StandardScaler → PCA → RidgeClassifier
      - 'logreg' : StandardScaler → PCA → LogisticRegression
      - 'rank1'  : StandardScaler → SupervisedSubspace(k=1, method=cfg.LOWRANK_METHOD or 'lda') → LogisticRegression
      - 'lowrank': StandardScaler → SupervisedSubspace(k=cfg.LOWRANK_K, method=cfg.LOWRANK_METHOD or 'pls') → LogisticRegression
    """
    # Common scaler
    scaler = StandardScaler(with_mean=True, with_std=True)

    # Choose flow based on classifier setting
    clf_name = str(cfg.CLASSIFIER).lower()

    if clf_name in {"ridge", "logreg"}:
        # PCA config: either variance in (0,1] or integer component count; else cap
        keep_variance = isinstance(cfg.PCA_VARIANCE, float) and 0 < cfg.PCA_VARIANCE <= 1.0
        if keep_variance:
            n_components = cfg.PCA_VARIANCE
        elif isinstance(cfg.PCA_VARIANCE, int):
            n_components = min(int(cfg.PCA_VARIANCE), d_model)
        else:
            n_components = min(cfg.PCA_MAX_COMPONENTS, d_model)
        pca = PCA(
            n_components=n_components,
            svd_solver="full" if keep_variance else "randomized",
            random_state=getattr(cfg, 'RANDOM_STATE', 0),
        )
        steps = [("scale", scaler), ("pca", pca)]
        if clf_name == "ridge":
            clf = RidgeClassifier(alpha=cfg.RIDGE_ALPHA, class_weight="balanced")
        else:
            clf = LogisticRegression(
                max_iter=2000,
                C=cfg.LOGREG_C,
                solver="lbfgs",
                class_weight="balanced",
                n_jobs=getattr(cfg, 'N_JOBS', None),
            )
        steps.append(("clf", clf))
        return Pipeline(steps)

    elif clf_name in {"rank1", "lowrank"}:
        if clf_name == "rank1":
            k = 1
            method = getattr(cfg, 'LOWRANK_METHOD', 'lda')
        else:
            k = int(getattr(cfg, 'LOWRANK_K', 4))
            method = getattr(cfg, 'LOWRANK_METHOD', 'pls')
        sub = SupervisedSubspace(k=k, method=method)
        clf = LogisticRegression(
            max_iter=2000,
            C=cfg.LOGREG_C,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=getattr(cfg, 'N_JOBS', None),
        )
        return Pipeline([("scale", scaler), ("subspace", sub), ("clf", clf)])

    else:
        raise ValueError("cfg.CLASSIFIER must be one of {'ridge','logreg','rank1','lowrank'}")

# -------------------------
# Metrics & reporting (kept same style)
# -------------------------

def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> Dict[str, float]:
    out = dict(
        accuracy=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, zero_division=0),
    )
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
    auroc = m.get('auroc', float('nan'))
    ap = m.get('ap', float('nan'))
    return (
        f"{layer:>{c['layer']}}"
        f"{n:>{c['n']}}"
        f"{ncomp:>{c['comps']}}"
        f"{m['acc']*100:>{c['acc']}.2f}"
        f"{(auroc*100 if np.isfinite(auroc) else float('nan')):>{c['auroc']}.2f}"
        f"{(ap*100 if np.isfinite(ap) else float('nan')):>{c['ap']}.2f}"
        f"{m['f1']*100:>{c['f1']}.2f}"
    )


def _write_report(
    report_path: Path,
    header_lines: List[str],
    table_lines: List[str],
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

# -------------------------
# Geometry export (directions/subspaces for ablations)
# -------------------------

def _maybe_save_geometry(run_dir: Path, layer_idx: int, pipe: Pipeline):
    """If the pipeline contains a 'subspace' step, export:
       - direction.npy (d,) for rank1
       - subspace_U.npy (d,k) for lowrank (orthonormal columns)
       Also save scaler mean/scale for reproducible injections.
    """
    steps = pipe.named_steps
    if "subspace" not in steps:
        return
    scaler: Optional[StandardScaler] = steps.get("scale")  # type: ignore
    sub: SupervisedSubspace = steps["subspace"]  # type: ignore
    U = sub.recover_geometry(scaler)

    out_dir = run_dir / f"layer_{layer_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save U and, if rank-1, a convenience 'direction'
    np.save(out_dir / ("direction.npy" if U.shape[1] == 1 else "subspace_U.npy"), U if U.shape[1] > 1 else U[:, 0])

    # Save scaler stats for later intervention code
    if scaler is not None and getattr(scaler, 'scale_', None) is not None:
        np.save(out_dir / "scaler_mean.npy", getattr(scaler, 'mean_', None))
        np.save(out_dir / "scaler_scale.npy", getattr(scaler, 'scale_', None))

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Train layerwise probes (incl. rank-1 / low-rank)")
    ap.add_argument("--n_tokens", type=int, default=None,
                    help="Randomly sample this many tokens before training.")
    ap.add_argument("--sample_seed", type=int, default=None,
                    help="Random seed for token sampling (defaults to cfg.RANDOM_STATE).")
    ap.add_argument("--save_geometry", action="store_true",
                    help="If set, save learned direction/subspace per layer for interventions.")
    cli = ap.parse_args()

    # Paths
    npz_path = (Path(__file__).parent / cfg.ACTS_NPZ).resolve()
    labels_csv = cfg.LABELS_CSV
    if labels_csv is None:
        labels_csv = _infer_labels_csv(npz_path)
        if labels_csv is None:
            raise FileNotFoundError(
                f"Could not infer labels CSV from NPZ path {npz_path}. Set LABELS_CSV in probes_config.py."
            )
    else:
        labels_csv = (Path(__file__).parent / labels_csv).resolve()

    out_dir = (Path(__file__).parent / cfg.OUT_DIR).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"{cfg.RUN_TAG}__{timestamp}"
    report_txt = run_dir / "report.txt"
    scores_csv = run_dir / "layer_scores.csv"

    # Labels (+ optional subsample)
    df = _read_labels(labels_csv)
    target_n: Optional[int] = None
    if cli.n_tokens is not None and cli.n_tokens > 0:
        target_n = int(cli.n_tokens)
    elif getattr(cfg, 'N_TOKENS', None) is not None and int(getattr(cfg, 'N_TOKENS')) > 0:
        target_n = int(getattr(cfg, 'N_TOKENS'))
    if target_n is not None and len(df) > target_n:
        seed = cli.sample_seed if cli.sample_seed is not None else getattr(cfg, 'RANDOM_STATE', 0)
        df = df.sample(n=target_n, random_state=seed)

    row_idx = df.index.to_numpy()
    y = df["_y"].to_numpy()
    groups = df[cfg.GROUP_COL].to_numpy()

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

    # Optional layer restriction
    if getattr(cfg, 'LAYERS_TO_TRAIN', None) is not None:
        requested = list(dict.fromkeys(cfg.LAYERS_TO_TRAIN))
        present_set = set(present_layer_ids)
        req_set = set(int(x) for x in requested)
        missing = sorted(req_set - present_set)
        if missing:
            print(f"[WARN] Requested layers not found in activations: {missing}. Training on intersection only.")
        keep = req_set & present_set
        layers = [t for t in layers if t[0] in keep]
        print(f"[INFO] Restricting to {len(layers)} requested layers out of {len(present_layer_ids)} present: {[t[0] for t in layers]}")
        if not layers:
            print("[WARN] No layers selected after filtering; proceeding with no training.")

    # Pretty table header
    hdr = cfg.COL_WIDTHS
    table_lines: List[str] = []
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

    rows_for_csv: List[Dict] = []
    all_layer_preds: Dict[int, np.ndarray] = {}
    all_layer_scores: Dict[int, np.ndarray] = {}

    layer_iter = tqdm(layers, desc="Layers", leave=True) if _HAS_TQDM else layers
    for layer_idx, key in layer_iter:
        X = npz[key]  # shape [N_tokens, d_model]
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
                    f"After subsetting, length mismatch for L{layer_idx}: X has {len(X)}, labels have {len(df)}"
                )
        X = X.astype(np.float32, copy=False)
        d_model = X.shape[1]

        pipe = _build_pipeline(d_model)

        fold_metrics: List[Dict[str, float]] = []
        oof_pred = np.empty_like(y)
        oof_score = np.full_like(y, fill_value=np.nan, dtype=float)
        ncomp_for_layer: int | None = None

        splits = list(gkf.split(X, y, groups))
        fold_iter = tqdm(splits, desc=f"L{layer_idx} CV", leave=False) if _HAS_TQDM else splits
        for train_idx, test_idx in fold_iter:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            pipe.fit(X_tr, y_tr)

            # Track effective dimensionality presented to the classifier
            if ncomp_for_layer is None:
                if 'pca' in pipe.named_steps:
                    try:
                        ncomp_for_layer = int(getattr(pipe.named_steps['pca'], 'n_components_', 0))
                    except Exception:
                        ncomp_for_layer = None
                elif 'subspace' in pipe.named_steps:
                    ncomp_for_layer = int(pipe.named_steps['subspace'].k)
                else:
                    ncomp_for_layer = d_model

            y_hat = pipe.predict(X_te)
            try:
                score = pipe.decision_function(X_te)
            except Exception:
                try:
                    proba = pipe.predict_proba(X_te)
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
            return float(np.mean(vals)) if vals else float('nan')

        agg = dict(
            layer=layer_idx,
            n=len(X),
            d_model=d_model,
            mode=str(cfg.CLASSIFIER).lower(),
            acc=_avg('accuracy'),
            auroc=_avg('auroc'),
            ap=_avg('ap'),
            f1=_avg('f1'),
        )
        rows_for_csv.append(agg)

        ncomp = int(ncomp_for_layer) if ncomp_for_layer is not None else 0
        table_lines.append(_format_table_row(layer_idx, len(X), ncomp, agg) + "\n")

        all_layer_preds[layer_idx] = oof_pred
        all_layer_scores[layer_idx] = oof_score

        if cli.save_geometry:
            try:
                _maybe_save_geometry(run_dir, layer_idx, pipe)
            except Exception as e:
                print(f"[WARN] Failed to save geometry for L{layer_idx}: {e}")

    # Write CSV of raw numbers
    run_dir.mkdir(parents=True, exist_ok=True)
    if rows_for_csv:
        pd.DataFrame(rows_for_csv).sort_values("layer").to_csv(scores_csv, index=False)
    else:
        pd.DataFrame(columns=["layer", "n", "d_model", "mode", "acc", "auroc", "ap", "f1"]).to_csv(scores_csv, index=False)
    print(f"[INFO] Wrote per-layer scores CSV → {scores_csv}")

    # TXT header
    header = []
    header.append(f"Probe run: {cfg.RUN_TAG}")
    header.append(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    header.append(f"NPZ: {npz_path}")
    header.append(f"Labels: {labels_csv}")
    header.append(f"Hook prefix: {cfg.HOOK_POINT_PREFIX}  |  Mode: {cfg.CLASSIFIER}")
    if str(cfg.CLASSIFIER).lower() in {"ridge", "logreg"}:
        header.append(f"PCA: keep {cfg.PCA_VARIANCE:.3f} variance (cap {cfg.PCA_MAX_COMPONENTS})")
    else:
        k = 1 if str(cfg.CLASSIFIER).lower()=="rank1" else int(getattr(cfg,'LOWRANK_K',4))
        method = getattr(cfg, 'LOWRANK_METHOD', 'pls' if k>1 else 'lda')
        header.append(f"Subspace: method={method}, k={k}")
    header.append(f"Grouped CV: {cfg.N_SPLITS} folds by '{cfg.GROUP_COL}'  |  N={len(df)} tokens, "
                  f"Examples={df[cfg.GROUP_COL].nunique()}, PosRate={df['_y'].mean():.3f}")

    # Record any active filters
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

    # Per-regime summary (best layers)
    per_regime_lines: List[str] = []
    try:
        scores_df = pd.DataFrame(rows_for_csv)
        scores_df['auroc'] = scores_df['auroc'].fillna(-1.0)
        best = scores_df.sort_values(["auroc", "acc"], ascending=False).head(3)
        if cfg.REGIME_COL in df.columns:
            idx_to_pos = {int(ix): pos for pos, ix in enumerate(df.index.to_numpy())}
            for _, row in best.iterrows():
                lid = int(row["layer"])
                y_hat = all_layer_preds[lid]
                y_score = all_layer_scores[lid]
                per_regime_lines.append(f"\nBest layer candidate: L{lid}  (Acc={row['acc']:.3f}, AUROC={row['auroc']:.3f})")
                for reg, sub in df.groupby(cfg.REGIME_COL):
                    orig_idx = sub.index.to_numpy()
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
    except Exception as e:
        print(f"[WARN] Could not compute per-regime summary: {e}")

    # Write TXT
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_report(report_txt, header, table_lines, per_regime_lines, cfg.N_SPLITS)

    print(f"[DONE] Report: {report_txt}")
    print(f"[DONE] Scores: {scores_csv}")


if __name__ == "__main__":
    main()
