from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional

try:
    from tqdm.auto import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False
    def tqdm(x, *args, **kwargs):
        return x

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression

import probes_config as cfg

# ========================= I/O helpers =========================

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
    y_raw = df[cfg.TARGET_COL].astype(str).str.strip().str.lower()
    df["_y"] = (y_raw == cfg.POSITIVE_TOKEN.lower()).astype(int)

    if cfg.GROUP_COL not in df.columns:
        raise ValueError(f"Missing '{cfg.GROUP_COL}' in labels CSV")

    # Optional regime selection
    regimes_to_use = getattr(cfg, 'REGIMES_TO_USE', None)
    if regimes_to_use is not None:
        if cfg.REGIME_COL not in df.columns:
            raise ValueError(f"Requested regime filtering but '{cfg.REGIME_COL}' not present in labels.")
        keep = set(regimes_to_use)
        df = df[df[cfg.REGIME_COL].isin(keep)]

    # Offset-based filtering (supports negative values correctly by ensuring numeric dtype)
    off_col = cfg.OFFSET_COL
    has_any_filter = (
        cfg.FILTER_OFFSET_EQ is not None
        or getattr(cfg, 'FILTER_OFFSET_MAX', None) is not None
        or getattr(cfg, 'FILTER_OFFSET_RANGE', None) is not None
    )
    if has_any_filter:
        if off_col not in df.columns:
            raise ValueError(f"Requested offset filtering but '{off_col}' not present in labels.")
        # Force numeric for robust comparisons with negatives; drop rows where parsing failed
        df[off_col] = pd.to_numeric(df[off_col], errors='coerce')
        df = df.dropna(subset=[off_col])
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
    out: List[Tuple[int, str]] = []
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

# ========================= Supervised subspace =========================

class SupervisedSubspace:
    def __init__(self, k: int = 1, method: str = "lda"):
        self.k = int(k)
        self.method = method.lower()
        self.fitted_ = False
        self.W_std_: Optional[np.ndarray] = None  # (d, k)
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.method == "lda":
            if len(np.unique(y)) != 2:
                raise ValueError("LDA requires binary targets")
            if self.k != 1:
                raise ValueError("Binary LDA implies k=1")
            lda = LinearDiscriminantAnalysis(solver="svd")
            lda.fit(X, y)
            W = np.asarray(lda.scalings_)
            if W.ndim == 1:
                W = W[:, None]
            self.W_std_ = W[:, :1]
            self._model = lda
        elif self.method == "pls":
            pls = PLSRegression(n_components=self.k, scale=False)
            y_reg = y.reshape(-1, 1).astype(float)
            pls.fit(X, y_reg)
            self.W_std_ = np.asarray(pls.x_weights_)[:, : self.k]
            self._model = pls
        else:
            raise ValueError("method must be 'lda' or 'pls'")
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Call fit() first")
        return X @ self.W_std_

    def recover_geometry(self, scaler: Optional[StandardScaler]) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Call fit() first")
        W = self.W_std_.copy()
        if scaler is not None and getattr(scaler, 'scale_', None) is not None:
            W = W / scaler.scale_.reshape(-1, 1)
        Q, _ = np.linalg.qr(W)
        return Q[:, : self.k]

# ========================= Pipelines =========================

def _build_pipeline(d_model: int) -> Pipeline:
    scaler = StandardScaler(with_mean=True, with_std=True)
    clf_name = str(cfg.CLASSIFIER).lower()

    if clf_name in {"ridge", "logreg"}:
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
            clf = LogisticRegression(max_iter=2000, C=cfg.LOGREG_C, solver="lbfgs", class_weight="balanced", n_jobs=getattr(cfg, 'N_JOBS', None))
        steps.append(("clf", clf))
        return Pipeline(steps)

    elif clf_name in {"rank1", "lowrank"}:
        if clf_name == "rank1":
            k = 1; method = getattr(cfg, 'LOWRANK_METHOD', 'lda')
        else:
            k = int(getattr(cfg, 'LOWRANK_K', 4)); method = getattr(cfg, 'LOWRANK_METHOD', 'pls')
        sub = SupervisedSubspace(k=k, method=method)
        clf = LogisticRegression(max_iter=2000, C=cfg.LOGREG_C, solver="lbfgs", class_weight="balanced", n_jobs=getattr(cfg, 'N_JOBS', None))
        return Pipeline([("scale", scaler), ("subspace", sub), ("clf", clf)])

    else:
        raise ValueError("cfg.CLASSIFIER must be one of {'ridge','logreg','rank1','lowrank'}")

# ========================= Metrics & report =========================

def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray]) -> Dict[str, float]:
    out = dict(accuracy=accuracy_score(y_true, y_pred), f1=f1_score(y_true, y_pred, zero_division=0))
    try:
        if y_score is not None and len(np.unique(y_true)) > 1:
            out["auroc"] = roc_auc_score(y_true, y_score)
            out["ap"] = average_precision_score(y_true, y_score)
        else:
            out["auroc"] = float("nan"); out["ap"] = float("nan")
    except Exception:
        out["auroc"] = float("nan"); out["ap"] = float("nan")
    return out


def _format_table_row(layer: int, n: int, ncomp: int, m: Dict[str, float]) -> str:
    c = cfg.COL_WIDTHS
    auroc = m.get('auroc', float('nan')); ap = m.get('ap', float('nan'))
    return (
        f"{layer:>{c['layer']}}"
        f"{n:>{c['n']}}"
        f"{ncomp:>{c['comps']}}"
        f"{m['accuracy']*100:>{c['acc']}.2f}"
        f"{(auroc*100 if np.isfinite(auroc) else float('nan')):>{c['auroc']}.2f}"
        f"{(ap*100 if np.isfinite(ap) else float('nan')):>{c['ap']}.2f}"
        f"{m['f1']*100:>{c['f1']}.2f}"
    )


def _write_report(report_path: Path, header: List[str], table_lines: List[str], per_regime: List[str], compare_lines: List[str], n_splits: int):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        for line in header: f.write(line.rstrip() + "\n")
        f.write("\nPer-layer results (grouped %d-fold CV, metrics in %% unless noted):\n" % n_splits)
        f.write("".join(table_lines)); f.write("\n")
        if per_regime:
            f.write("Per-regime breakdown for the best layer(s):\n")
            for line in per_regime: f.write(line.rstrip() + "\n")
        if compare_lines:
            f.write("\nDirection comparison vs DoM (averaged across folds):\n")
            for line in compare_lines: f.write(line.rstrip() + "\n")

# ========================= Comparison helpers =========================

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1e-12
    na = np.linalg.norm(a) + eps; nb = np.linalg.norm(b) + eps
    return float(np.clip((a @ b) / (na * nb), -1.0, 1.0))


def _compute_dom_vectors_std(X_std: np.ndarray, y: np.ndarray, reg_eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mu0 = X_std[y == 0].mean(axis=0)
    mu1 = X_std[y == 1].mean(axis=0)
    w_dom_std = (mu1 - mu0)
    C = np.cov(X_std, rowvar=False)
    C_reg = C + reg_eps * np.eye(C.shape[0], dtype=C.dtype)
    w_wdom_std = np.linalg.solve(C_reg, w_dom_std)
    return w_dom_std, w_wdom_std


def _maybe_save_geometry(run_dir: Path, layer_idx: int, pipe: Pipeline, order_idx: Optional[np.ndarray] = None):
    steps = pipe.named_steps
    if "subspace" not in steps: return
    scaler: Optional[StandardScaler] = steps.get("scale")  # type: ignore
    sub: SupervisedSubspace = steps["subspace"]  # type: ignore
    U = sub.recover_geometry(scaler)
    out_dir = run_dir / f"layer_{layer_idx:02d}"; out_dir.mkdir(parents=True, exist_ok=True)
    if U.shape[1] == 1:
        np.save(out_dir / "direction.npy", U[:, 0])
    else:
        np.save(out_dir / "subspace_U.npy", U)
        if order_idx is not None:
            np.save(out_dir / "subspace_order.npy", order_idx.astype(int))
    if scaler is not None and getattr(scaler, 'scale_', None) is not None:
        np.save(out_dir / "scaler_mean.npy", getattr(scaler, 'mean_', None))
        np.save(out_dir / "scaler_scale.npy", getattr(scaler, 'scale_', None))

# ========================= Main =========================

def main():
    ap = argparse.ArgumentParser(description="Train layerwise probes (rank-1/low-rank/PCA) + DoM comparisons")
    ap.add_argument("--n_tokens", type=int, default=None)
    ap.add_argument("--sample_seed", type=int, default=None)
    ap.add_argument("--save_geometry", action="store_true")
    cli = ap.parse_args()

    npz_path = (Path(__file__).parent / cfg.ACTS_NPZ).resolve()
    labels_csv = cfg.LABELS_CSV or _infer_labels_csv(npz_path)
    if labels_csv is None:
        raise FileNotFoundError("Could not infer labels CSV; set LABELS_CSV in probes_config.py")
    labels_csv = (Path(__file__).parent / labels_csv).resolve()

    out_dir = (Path(__file__).parent / cfg.OUT_DIR).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"{cfg.RUN_TAG}__{timestamp}"
    report_txt = run_dir / "report.txt"
    scores_csv = run_dir / "layer_scores.csv"

    df = _read_labels(labels_csv)
    target_n = None
    if cli.n_tokens and cli.n_tokens > 0:
        target_n = int(cli.n_tokens)
    elif getattr(cfg, 'N_TOKENS', None):
        target_n = int(getattr(cfg, 'N_TOKENS'))
    if target_n is not None and len(df) > target_n:
        seed = cli.sample_seed if cli.sample_seed is not None else getattr(cfg, 'RANDOM_STATE', 0)
        df = df.sample(n=target_n, random_state=seed)

    row_idx = df.index.to_numpy()
    y = df["_y"].to_numpy()
    groups = df[cfg.GROUP_COL].to_numpy()

    if df[cfg.GROUP_COL].nunique() < cfg.N_SPLITS:
        raise ValueError(f"Not enough unique {cfg.GROUP_COL} for {cfg.N_SPLITS}-fold GroupKFold.")

    gkf = GroupKFold(n_splits=cfg.N_SPLITS)

    npz = np.load(npz_path)
    layers = _layer_keys(npz)
    print(f"[INFO] Loaded acts: {npz_path}")
    print(f"[INFO] Labels: {labels_csv} | N={len(df)} tokens | Positive rate={df['_y'].mean():.3f}")

    # Optional restriction
    present_layer_ids = [idx for idx, _ in layers]
    if getattr(cfg, 'LAYERS_TO_TRAIN', None):
        requested = list(dict.fromkeys(cfg.LAYERS_TO_TRAIN))
        present_set = set(present_layer_ids)
        req_set = set(int(x) for x in requested)
        keep = req_set & set(present_layer_ids)
        missing = sorted(req_set - present_set)
        if missing:
            print(f"[WARN] Requested layers not found: {missing}")
        layers = [t for t in layers if t[0] in keep]
        print(f"[INFO] Restricting to {len(layers)} requested layers: {[t[0] for t in layers]}")

    # Table header
    hdr = cfg.COL_WIDTHS
    table_lines: List[str] = []
    table_lines.append(
        f"{'layer':>{hdr['layer']}}{'N':>{hdr['n']}}{'PCs':>{hdr['comps']}}{'Acc':>{hdr['acc']}}{'AUROC':>{hdr['auroc']}}{'AP':>{hdr['ap']}}{'F1':>{hdr['f1']}}\n"
    )
    table_lines.append("-" * sum(hdr.values()) + "\n")

    rows_for_csv: List[Dict] = []
    all_layer_preds: Dict[int, np.ndarray] = {}
    all_layer_scores: Dict[int, np.ndarray] = {}

    compare_lines: List[str] = []
    COMPARE_MODE = str(getattr(cfg, 'COMPARE_MODE', 'none')).lower()  # 'none'|'dom'|'whitened_dom'|'both'
    REG_EPS = float(getattr(cfg, 'COMPARE_REG_EPS', 1e-3))

    layer_iter = tqdm(layers, desc="Layers", leave=True) if _HAS_TQDM else layers
    for layer_idx, key in layer_iter:
        X = npz[key]
        if len(X) != len(df):
            X = X[row_idx]
            if len(X) != len(df):
                raise ValueError(f"After subsetting, length mismatch for L{layer_idx}: {len(X)} vs {len(df)}")
        X = X.astype(np.float32, copy=False)
        d_model = X.shape[1]

        pipe = _build_pipeline(d_model)

        fold_metrics: List[Dict[str, float]] = []
        oof_pred = np.empty_like(y)
        oof_score = np.full_like(y, fill_value=np.nan, dtype=float)
        ncomp_for_layer: Optional[int] = None

        # For lowrank ordering across folds
        comp_auc_sums: Optional[np.ndarray] = None
        comp_counts: Optional[np.ndarray] = None

        # Compare accumulators
        cos_dom_list: List[float] = []
        cos_wdom_list: List[float] = []
        auc_topdir_list: List[float] = []
        auc_dom_list: List[float] = []
        auc_wdom_list: List[float] = []

        splits = list(gkf.split(X, y, groups))
        fold_iter = tqdm(splits, desc=f"L{layer_idx} CV", leave=False) if _HAS_TQDM else splits
        for train_idx, test_idx in fold_iter:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            pipe.fit(X_tr, y_tr)

            if ncomp_for_layer is None:
                if 'pca' in pipe.named_steps:
                    ncomp_for_layer = int(getattr(pipe.named_steps['pca'], 'n_components_', 0))
                elif 'subspace' in pipe.named_steps:
                    ncomp_for_layer = int(pipe.named_steps['subspace'].k)
                else:
                    ncomp_for_layer = d_model

            y_hat = pipe.predict(X_te)
            try:
                score = pipe.decision_function(X_te)
            except Exception:
                try:
                    proba = pipe.predict_proba(X_te); score = proba[:, 1]
                except Exception:
                    score = None
            m = _safe_metrics(y_te, y_hat, score)
            fold_metrics.append(m)
            oof_pred[test_idx] = y_hat
            if score is not None:
                oof_score[test_idx] = score

            # ---- Compare mode ----
            steps = pipe.named_steps
            if 'subspace' in steps and COMPARE_MODE != 'none':
                scaler: StandardScaler = steps['scale']  # type: ignore
                sub: SupervisedSubspace = steps['subspace']  # type: ignore
                W_std = sub.W_std_  # (d, k)
                if W_std is None:
                    continue
                Xtr_std = scaler.transform(X_tr)
                Xte_std = scaler.transform(X_te)
                k_sub = W_std.shape[1]

                # Rank components by train AUROC
                comp_aurocs = np.zeros(k_sub, dtype=float)
                for j in range(k_sub):
                    s_tr = Xtr_std @ W_std[:, j]
                    if len(np.unique(y_tr)) > 1:
                        try:
                            comp_aurocs[j] = roc_auc_score(y_tr, s_tr)
                        except Exception:
                            comp_aurocs[j] = np.nan
                    else:
                        comp_aurocs[j] = np.nan
                if comp_auc_sums is None:
                    comp_auc_sums = np.zeros_like(comp_aurocs); comp_counts = np.zeros_like(comp_aurocs)
                finite_mask = np.isfinite(comp_aurocs)
                comp_auc_sums[finite_mask] += comp_aurocs[finite_mask]
                comp_counts[finite_mask] += 1

                best_j = int(np.nanargmax(comp_aurocs)) if np.any(finite_mask) else 0
                w_top_std = W_std[:, best_j]

                want_dom = COMPARE_MODE in {"dom", "both", "all"}
                want_wdom = COMPARE_MODE in {"whitened_dom", "both", "all"}
                w_dom_std, w_wdom_std = _compute_dom_vectors_std(Xtr_std, y_tr, reg_eps=REG_EPS)

                scale = getattr(scaler, 'scale_', None)
            if scale is None:
                scale = np.ones(W_std.shape[0], dtype=W_std.dtype)
            else:
                scale = np.asarray(scale)
                inv_scale = 1.0 / (scale + 1e-12)
                w_top_orig = w_top_std * inv_scale
                w_dom_orig = w_dom_std * inv_scale
                w_wdom_orig = w_wdom_std * inv_scale

                if want_dom:
                    cos_dom_list.append(_cos(w_top_orig, w_dom_orig))
                if want_wdom:
                    cos_wdom_list.append(_cos(w_top_orig, w_wdom_orig))

                def _auc_safe(y_true, s):
                    try:
                        if len(np.unique(y_true)) > 1:
                            return float(roc_auc_score(y_true, s))
                        return float('nan')
                    except Exception:
                        return float('nan')

                auc_topdir_list.append(_auc_safe(y_te, Xte_std @ w_top_std))
                if want_dom:
                    auc_dom_list.append(_auc_safe(y_te, Xte_std @ w_dom_std))
                if want_wdom:
                    auc_wdom_list.append(_auc_safe(y_te, Xte_std @ w_wdom_std))

        # Aggregate fold metrics
        def _avg(name: str, default=np.nan):
            vals = [d.get(name, default) for d in fold_metrics]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            return float(np.mean(vals)) if vals else float('nan')

        agg = dict(layer=layer_idx, n=len(X), d_model=d_model, mode=str(cfg.CLASSIFIER).lower(),
                   accuracy=_avg('accuracy'), auroc=_avg('auroc'), ap=_avg('ap'), f1=_avg('f1'))
        rows_for_csv.append(agg)
        ncomp = int(ncomp_for_layer) if ncomp_for_layer is not None else 0
        table_lines.append(_format_table_row(layer_idx, len(X), ncomp, agg) + "\n")
        all_layer_preds[layer_idx] = oof_pred
        all_layer_scores[layer_idx] = oof_score

        # Order for lowrank + save geometry
        order_idx: Optional[np.ndarray] = None
        if comp_auc_sums is not None and comp_counts is not None and np.any(comp_counts > 0):
            mean_aurocs = comp_auc_sums / np.maximum(comp_counts, 1)
            order_idx = np.argsort(-np.nan_to_num(mean_aurocs, nan=-1.0))
        if cli.save_geometry:
            try:
                _maybe_save_geometry(run_dir, layer_idx, pipe, order_idx)
            except Exception as e:
                print(f"[WARN] Failed to save geometry for L{layer_idx}: {e}")

        if COMPARE_MODE != 'none' and (len(cos_dom_list) or len(cos_wdom_list)):
            parts = [f"L{layer_idx:02d} compare:"]
            if len(cos_dom_list):
                parts.append(f" cos(w,DoM)={np.nanmean(cos_dom_list):.3f}")
            if len(cos_wdom_list):
                parts.append(f" cos(w,whitened)={np.nanmean(cos_wdom_list):.3f}")
            if len(auc_topdir_list):
                parts.append(f" AUROC[top-dir]={np.nanmean(auc_topdir_list):.3f}")
            if len(auc_dom_list):
                parts.append(f" AUROC[DoM]={np.nanmean(auc_dom_list):.3f}")
            if len(auc_wdom_list):
                parts.append(f" AUROC[whitened]={np.nanmean(auc_wdom_list):.3f}")
            compare_lines.append(" ".join(parts))

    # Write CSV
    run_dir.mkdir(parents=True, exist_ok=True)
    if rows_for_csv:
        pd.DataFrame(rows_for_csv).sort_values("layer").to_csv(scores_csv, index=False)
    else:
        pd.DataFrame(columns=["layer","n","d_model","mode","accuracy","auroc","ap","f1"]).to_csv(scores_csv, index=False)
    print(f"[INFO] Wrote per-layer scores CSV → {scores_csv}")

    header: List[str] = []
    header.append(f"Probe run: {cfg.RUN_TAG}")
    header.append(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    header.append(f"NPZ: {npz_path}")
    header.append(f"Labels: {labels_csv}")
    header.append(f"Hook prefix: {cfg.HOOK_POINT_PREFIX}  |  Mode: {cfg.CLASSIFIER}")
    if str(cfg.CLASSIFIER).lower() in {"ridge","logreg"}:
        header.append(f"PCA: keep {cfg.PCA_VARIANCE:.3f} variance (cap {cfg.PCA_MAX_COMPONENTS})")
    else:
        k = 1 if str(cfg.CLASSIFIER).lower()=="rank1" else int(getattr(cfg,'LOWRANK_K',4))
        method = getattr(cfg, 'LOWRANK_METHOD', 'pls' if k>1 else 'lda')
        header.append(f"Subspace: method={method}, k={k}")
    header.append(f"Grouped CV: {cfg.N_SPLITS} folds by '{cfg.GROUP_COL}'  |  N={len(df)} tokens, Examples={df[cfg.GROUP_COL].nunique()}, PosRate={df['_y'].mean():.3f}")

    # Filters
    filters = []
    # Regime filter header note
    regimes_to_use = getattr(cfg, 'REGIMES_TO_USE', None)
    if regimes_to_use is not None:
        filters.append(f"regime in {{{', '.join(map(str, regimes_to_use))}}}")
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

    # Per-regime summary for best 3 layers
    per_regime_lines: List[str] = []
    try:
        scores_df = pd.DataFrame(rows_for_csv)
        scores_df['auroc'] = scores_df['auroc'].fillna(-1.0)
        best = scores_df.sort_values(["auroc","accuracy"], ascending=False).head(3)
        if cfg.REGIME_COL in df.columns:
            idx_to_pos = {int(ix): pos for pos, ix in enumerate(df.index.to_numpy())}
            for _, row in best.iterrows():
                lid = int(row['layer'])
                y_hat = all_layer_preds.get(lid); y_score = all_layer_scores.get(lid)
                if y_hat is None or y_score is None: continue
                per_regime_lines.append(f"\nBest layer candidate: L{lid}  (Acc={row['accuracy']:.3f}, AUROC={row['auroc']:.3f})")
                for reg, sub in df.groupby(cfg.REGIME_COL):
                    orig_idx = sub.index.to_numpy()
                    pos_idx = np.array([idx_to_pos[i] for i in orig_idx if int(i) in idx_to_pos], dtype=int)
                    if pos_idx.size == 0:
                        per_regime_lines.append(f"  - {reg:<18} Acc=nan  AUROC=nan"); continue
                    acc = accuracy_score(df.iloc[pos_idx]['_y'], y_hat[pos_idx])
                    try:
                        y_true_reg = df.iloc[pos_idx]['_y'].to_numpy(); y_score_reg = y_score[pos_idx]
                        mask = np.isfinite(y_score_reg)
                        if np.unique(y_true_reg).size > 1 and mask.any():
                            auc = roc_auc_score(y_true_reg[mask], y_score_reg[mask])
                        else: auc = float('nan')
                    except Exception: auc = float('nan')
                    per_regime_lines.append(f"  - {reg:<18} Acc={acc:.3f}  AUROC={auc:.3f}")
    except Exception as e:
        print(f"[WARN] Could not compute per-regime summary: {e}")

    _write_report(report_txt, header, table_lines, per_regime_lines, compare_lines, cfg.N_SPLITS)
    print(f"[DONE] Report: {report_txt}")
    print(f"[DONE] Scores: {scores_csv}")


if __name__ == "__main__":
    main()
