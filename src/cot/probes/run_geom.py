from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional
import json
import hashlib

try:
    from tqdm.auto import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False
    def tqdm(x, *args, **kwargs):
        return x

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression

import probes_config2 as cfg

# Optional torch for GPU-accelerated logistic regression on projections
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

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
        # Interpret tuple as inclusive range, list as explicit whitelist
        if isinstance(rng, tuple) and len(rng) == 2:
            lo, hi = rng
            if lo is not None:
                df = df[df[off_col] >= lo]
            if hi is not None:
                df = df[df[off_col] <= hi]
        elif isinstance(rng, list):
            # whitelist specific offsets (e.g., [-1, 0, 1])
            try:
                whitelist = set(int(x) for x in rng)
                df = df[df[off_col].isin(whitelist)]
            except Exception:
                # If casting fails, fall back to comparing stringified values
                whitelist = set(str(x) for x in rng)
                df = df[df[off_col].astype(str).isin(whitelist)]
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

# ========================= Contrast helpers (for multi-contrast Fisher) =========================

def _build_contrast_ids(df_sub: pd.DataFrame) -> np.ndarray:
    """
    Returns an int array of length len(df_sub) assigning each row to a contrast bucket.
    Config:
      - LOWRANK_LDA_CONTRAST_BY: one of
          * a column name (e.g., cfg.REGIME_COL or cfg.GROUP_COL),
          * a list/tuple of column names (joint key),
          * 'offset_bins' to bin cfg.OFFSET_COL,
          * None → auto: prefer REGIME_COL, else GROUP_COL, else single bucket.
      - LOWRANK_LDA_OFFSET_BIN_EDGES: explicit bin edges for offsets (optional)
      - LOWRANK_LDA_OFFSET_BIN_WIDTH: integer width for equal-width bins (default 5) if no edges
    """
    by = getattr(cfg, 'LOWRANK_LDA_CONTRAST_BY', None)
    if by is None:
        if hasattr(cfg, 'REGIME_COL') and cfg.REGIME_COL in df_sub.columns:
            by = cfg.REGIME_COL
        elif cfg.GROUP_COL in df_sub.columns:
            by = cfg.GROUP_COL
        else:
            return np.zeros(len(df_sub), dtype=int)

    if by == 'offset_bins':
        off_col = cfg.OFFSET_COL
        if off_col not in df_sub.columns:
            return np.zeros(len(df_sub), dtype=int)
        vals = pd.to_numeric(df_sub[off_col], errors='coerce').fillna(0).to_numpy()
        edges = getattr(cfg, 'LOWRANK_LDA_OFFSET_BIN_EDGES', None)
        if edges is not None:
            bins = np.digitize(vals, np.asarray(edges, dtype=float), right=True)
            return bins.astype(int)
        width = int(getattr(cfg, 'LOWRANK_LDA_OFFSET_BIN_WIDTH', 5))
        return (np.floor(vals / max(width, 1))).astype(int)

    if isinstance(by, (list, tuple)):
        keys = df_sub[list(by)].astype(str).agg('|'.join, axis=1)
        return pd.factorize(keys, sort=True)[0].astype(int)

    # Single column name
    if by in df_sub.columns:
        return pd.factorize(df_sub[by].astype(str), sort=True)[0].astype(int)

    # Fallback: single bucket
    return np.zeros(len(df_sub), dtype=int)

# ========================= Supervised subspace =========================

class SupervisedSubspace:
    def __init__(self, k: int = 1, method: str = "lda"):
        self.k = int(k)
        self.method = method.lower()
        self.fitted_ = False
        self.W_std_: Optional[np.ndarray] = None  # (d, k)
        self._model = None
        self._order: Optional[np.ndarray] = None  # ordering of components used (by train AUROC)

    @staticmethod
    def _fisher_dir(X: np.ndarray, y: np.ndarray, ridge: float) -> Optional[np.ndarray]:
        # Binary Fisher direction with ridge-regularized pooled within-class covariance
        cls = np.unique(y)
        if cls.size != 2:
            return None
        X0 = X[y == cls[0]]; X1 = X[y == cls[1]]
        if len(X0) < 2 or len(X1) < 2:
            return None
        mu0 = X0.mean(0); mu1 = X1.mean(0)
        X0c = X0 - mu0; X1c = X1 - mu1
        d = X.shape[1]
        # pooled within-class covariance
        Sw = (X0c.T @ X0c + X1c.T @ X1c) / max((len(X) - 2), 1)
        Sw = Sw + float(ridge) * np.eye(d, dtype=X.dtype)
        try:
            w = np.linalg.solve(Sw, (mu1 - mu0))
        except np.linalg.LinAlgError:
            return None
        # normalize to avoid extreme scales (final QR will re-orthonormalize anyway)
        n = float(np.linalg.norm(w)) + 1e-12
        return (w / n)

    def fit(self, X: np.ndarray, y: np.ndarray, contrast_ids: Optional[np.ndarray] = None):
        if self.k < 1:
            raise ValueError("k must be >= 1")

        # --- Multi-contrast Fisher when method='lda' and k>1 ---
        if self.method == "lda" and self.k > 1:
            if contrast_ids is None:
                raise ValueError("LOWRANK_METHOD='lda' with k>1 requires contrast_ids (multi-contrast Fisher).")

            min_per_class = int(getattr(cfg, 'LOWRANK_LDA_MIN_SAMPLES_PER_CLASS', 10))
            ridge = float(getattr(cfg, 'LOWRANK_LDA_RIDGE', 1e-2))
            Ws: List[np.ndarray] = []
            scores: List[float] = []

            for gid in np.unique(contrast_ids):
                mask = (contrast_ids == gid)
                Xg, yg = X[mask], y[mask]
                # both classes present and enough samples
                if np.unique(yg).size != 2: 
                    continue
                if (np.sum(yg == 0) < min_per_class) or (np.sum(yg == 1) < min_per_class):
                    continue
                w = self._fisher_dir(Xg, yg, ridge)
                if w is None:
                    continue
                Ws.append(w)
                # rank by within-fold train AUROC for this single direction
                try:
                    s = X @ w
                    auc = roc_auc_score(y, s) if np.unique(y).size == 2 else np.nan
                except Exception:
                    auc = np.nan
                scores.append(float(auc) if np.isfinite(auc) else -np.inf)

            if not Ws:
                # Fallback to classic rank-1 LDA if contrasts failed
                lda = LinearDiscriminantAnalysis(solver="svd")
                lda.fit(X, y)
                W = np.asarray(lda.scalings_)
                if W.ndim == 1: W = W[:, None]
                self.W_std_ = W[:, :1]
                self._model = lda
                self.fitted_ = True
                self._order = np.array([0], dtype=int)
                return self

            W_all = np.stack(Ws, axis=1)  # (d, m)
            order = np.argsort(-np.nan_to_num(np.asarray(scores), nan=-np.inf))
            keep = min(self.k, W_all.shape[1])
            self.W_std_ = W_all[:, order[:keep]]
            self._order = order[:keep]
            self._model = ("multicontrast_lda", {"n_dirs_total": W_all.shape[1]})
            self.fitted_ = True
            return self

        # --- Classic binary LDA (k must be 1) ---
        if self.method == "lda":
            if len(np.unique(y)) != 2:
                raise ValueError("LDA requires binary targets")
            if self.k != 1:
                raise ValueError("Binary LDA implies k=1. Use k>1 only with multi-contrast Fisher.")
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



# ========================= GPU helpers (optional) =========================

def _select_device() -> str:
    want = str(getattr(cfg, 'DEVICE', 'cpu')).lower()
    if want == 'auto':
        return 'cuda' if _HAS_TORCH and torch.cuda.is_available() else 'cpu'  # type: ignore[name-defined]
    if want == 'cuda' and (not _HAS_TORCH or not torch.cuda.is_available()):  # type: ignore[attr-defined]
        return 'cpu'
    return 'cpu' if want not in {'cpu','cuda'} else want


def _fit_logreg_torch(Z_tr: np.ndarray, y_tr: np.ndarray, Z_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not available for GPU path")
    device = _select_device()
    if device != 'cuda':
        raise RuntimeError("CUDA not selected/available for GPU path")

    Xtr = torch.from_numpy(Z_tr).to(device)
    ytr = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xte = torch.from_numpy(Z_te).to(device)

    n_features = Xtr.shape[1]
    model = nn.Linear(n_features, 1, bias=True).to(device)

    # Balanced class weighting similar to sklearn class_weight='balanced'
    pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Weight decay ~ 1/C (approximate)
    C = float(getattr(cfg, 'LOGREG_C', 1.0))
    wd_cfg = getattr(cfg, 'GPU_LR_WEIGHT_DECAY', None)
    weight_decay = float(wd_cfg) if wd_cfg is not None else (1.0 / max(C, 1e-8))
    lr = float(getattr(cfg, 'GPU_LR_LR', 0.1))
    epochs = int(getattr(cfg, 'GPU_LR_EPOCHS', 300))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(Xtr).squeeze(-1)
        loss = criterion(logits, ytr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_te = model(Xte).squeeze(-1)
        probs_te = torch.sigmoid(logits_te)
        y_hat = (probs_te >= 0.5).to(torch.int32).cpu().numpy()
        scores = logits_te.detach().float().cpu().numpy()
    return y_hat, scores

# ========================= Pipelines =========================

def _build_pipeline(d_model: int) -> Pipeline:
    scaler = StandardScaler(with_mean=True, with_std=True)
    clf_name = str(cfg.CLASSIFIER).lower()
    if clf_name in {"rank1", "lowrank"}:
        if clf_name == "rank1":
            k = 1; method = getattr(cfg, 'LOWRANK_METHOD', 'lda')
        else:
            k = int(getattr(cfg, 'LOWRANK_K', 4)); method = getattr(cfg, 'LOWRANK_METHOD', 'pls')
        sub = SupervisedSubspace(k=k, method=method)
        clf = LogisticRegression(max_iter=2000, C=cfg.LOGREG_C, solver="lbfgs", class_weight="balanced", n_jobs=getattr(cfg, 'N_JOBS', None))
        return Pipeline([("scale", scaler), ("subspace", sub), ("clf", clf)])

    else:
        raise ValueError("cfg.CLASSIFIER must be one of {'rank1','lowrank'}")

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


def _write_report(report_path: Path, header: List[str], table_lines: List[str], per_regime: List[str], nudge_lines: List[str], compare_lines: List[str], n_splits: int):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        for line in header: f.write(line.rstrip() + "\n")
        f.write("\nPer-layer results (grouped %d-fold CV, metrics in %% unless noted):\n" % n_splits)
        f.write("".join(table_lines)); f.write("\n")
        if per_regime:
            f.write("Per-regime breakdown for the best layer(s):\n")
            for line in per_regime: f.write(line.rstrip() + "\n")
        if nudge_lines:
            f.write("\nAverage nudge |cos| vs learned dir (across folds):\n")
            for line in nudge_lines: f.write(line.rstrip() + "\n")
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
    ap = argparse.ArgumentParser(description="Train layerwise probes (rank-1/low-rank) + DoM comparisons")
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
    n_splits = int(getattr(cfg, 'N_SPLITS', 2))
    # Precompute splits once
    if n_splits and n_splits > 0:
        if df[cfg.GROUP_COL].nunique() < n_splits:
            raise ValueError(f"Not enough unique {cfg.GROUP_COL} for {n_splits}-fold GroupKFold.")
        gkf = GroupKFold(n_splits=n_splits)
        base_splits = list(gkf.split(np.zeros(len(df)), y, groups))
    else:
        if 'split' not in df.columns:
            raise ValueError("N_SPLITS=0 requires a 'split' column with 'train'/'test'.")
        is_tr = (df['split'].astype(str) == 'train').to_numpy()
        is_te = (df['split'].astype(str) == 'test').to_numpy()
        if not is_tr.any() or not is_te.any():
            raise ValueError("N_SPLITS=0: train or test split is empty after filtering.")
        tr_idx = np.where(is_tr)[0]; te_idx = np.where(is_te)[0]
        base_splits = [(tr_idx, te_idx)]

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

    # --- Build descriptive run name and directories ---
    def _sanitize(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._+-]", "-", s)

    # Layers portion
    layers_used = [t[0] for t in layers]
    layers_str = "L" + ("-".join(str(x) for x in layers_used) if layers_used else "none")

    # Offset range portion (prefer FILTER_OFFSET_RANGE if present)
    rng = getattr(cfg, 'FILTER_OFFSET_RANGE', None)
    if isinstance(rng, tuple) and len(rng) == 2:
        lo, hi = rng
        off_str = f"off_{lo if lo is not None else 'min'}_to_{hi if hi is not None else 'max'}"
    elif isinstance(rng, list):
        vals_list = list(rng)
        vals_joined = "+".join(str(x) for x in vals_list)
        if len(vals_joined) <= 48:
            off_str = f"off_in_{vals_joined if vals_joined else 'none'}"
        else:
            # Compact representation for long explicit lists
            digest = hashlib.sha1(",".join(str(x) for x in vals_list).encode("utf-8")).hexdigest()[:8]
            try:
                vmin, vmax = min(vals_list), max(vals_list)
            except Exception:
                vmin, vmax = 'min', 'max'
            off_str = f"off_in_{vmin}..{vmax}_n{len(vals_list)}_{digest}"
    else:
        off_str = "off_all"

    # Regimes portion
    regs = getattr(cfg, 'REGIMES_TO_USE', None)
    regs_str = "regs_all" if regs is None else ("regs_" + "+".join(_sanitize(str(r)) for r in regs))

    # Model portion: try to infer from sibling info JSON
    model_str = "unknownmodel"
    info_json_cand = Path(str(npz_path).replace(".npz", "_info.json"))
    if info_json_cand.exists():
        try:
            with info_json_cand.open("r", encoding="utf-8") as f:
                info = json.load(f)
            mn = info.get("model_name") or info.get("model")
            if mn:
                model_str = _sanitize(str(mn))
        except Exception:
            pass

    base_name = f"{layers_str}__{off_str}__{regs_str}__{model_str}"
    # Enforce a max length on run directory name to avoid OS limits
    try:
        max_len = int(getattr(cfg, 'MAX_RUN_NAME_LEN', 120))
    except Exception:
        max_len = 120

    # Optional override via config
    override_name = getattr(cfg, 'OUTPUT_FILE_NAME', None)
    if override_name is not None and str(override_name).strip() != "":
        run_name = re.sub(r"[^A-Za-z0-9._+-]", "-", str(override_name).strip())
        if len(run_name) > max_len:
            run_name = run_name[:max_len]
    else:
        if len(base_name) > max_len:
            digest = hashlib.sha1(base_name.encode('utf-8')).hexdigest()[:10]
            keep = max(20, max_len - 12)  # keep a reasonable prefix
            base_name = base_name[:keep] + f"__{digest}"
        run_name = base_name
        # Disambiguate if exists
        k_suffix = 1
        while (out_dir / run_name).exists():
            k_suffix += 1
            candidate = f"{base_name}__{k_suffix}"
            if len(candidate) > max_len:
                # Trim to fit max_len while keeping suffix
                suffix = f"__{k_suffix}"
                run_name = base_name[: max_len - len(suffix)] + suffix
            else:
                run_name = candidate

    run_dir = out_dir / run_name
    report_txt = run_dir / "report.txt"
    scores_csv = run_dir / "layer_scores.csv"
    print(f"[INFO] Run dir: {run_dir}")

    # Prepare vectors output directory
    vectors_base = (Path(__file__).parent / getattr(cfg, 'VECTORS_DIR', Path("../outputs/vectors"))).resolve()
    vectors_run_dir = vectors_base / run_name
    vectors_run_dir.mkdir(parents=True, exist_ok=True)

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
    nudge_lines: List[str] = []
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
        # For N_SPLITS=0, keep the fitted pipeline to avoid re-training later
        pipe_for_vectors: Optional[Pipeline] = None

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
        # Nudge |cos| accumulators (per fold means)
        nudge_tr_means: List[float] = []
        nudge_te_means: List[float] = []

        splits = base_splits
        fold_iter = tqdm(splits, desc=f"L{layer_idx} CV" if (n_splits and n_splits>0) else f"L{layer_idx} train→test", leave=False) if _HAS_TQDM else splits
        use_gpu = (_select_device() == 'cuda') if _HAS_TORCH else False
        for train_idx, test_idx in fold_iter:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            contrast_ids_tr = _build_contrast_ids(df.iloc[train_idx])

            # Pass contrast ids to the subspace step so multi-contrast LDA (k>1) can fit
            pipe.fit(X_tr, y_tr, **{"subspace__contrast_ids": contrast_ids_tr})
            if not (n_splits and n_splits > 0):
                # CV=0 (train→test): reuse this trained pipeline for saving vectors later
                pipe_for_vectors = pipe

            if ncomp_for_layer is None:
                if 'pca' in pipe.named_steps:
                    ncomp_for_layer = int(getattr(pipe.named_steps['pca'], 'n_components_', 0))
                elif 'subspace' in pipe.named_steps:
                    ncomp_for_layer = int(pipe.named_steps['subspace'].k)
                else:
                    ncomp_for_layer = d_model

            # Default: CPU sklearn predictions
            y_hat = None; score = None
            if use_gpu and 'subspace' in pipe.named_steps:
                try:
                    # Manually compute projections and run GPU logistic regression
                    scaler: StandardScaler = pipe.named_steps['scale']  # type: ignore
                    sub: SupervisedSubspace = pipe.named_steps['subspace']  # type: ignore
                    W_std = sub.W_std_
                    if W_std is None:
                        raise RuntimeError('Subspace not fitted')
                    Xtr_std = scaler.transform(X_tr)
                    Xte_std = scaler.transform(X_te)
                    Z_tr = Xtr_std @ W_std
                    Z_te = Xte_std @ W_std
                    y_hat, score = _fit_logreg_torch(Z_tr.astype(np.float32), y_tr, Z_te.astype(np.float32))
                except Exception as e:
                    print(f"[WARN] GPU path failed, falling back to CPU sklearn: {e}")
            if y_hat is None:
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

            # ---- Direction selection (top component) and comparisons ----
            steps = pipe.named_steps
            if 'subspace' in steps:
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
                # Compute average nudge |cos| on train/test using original-space unit vector
                scale_arr = getattr(scaler, 'scale_', None)
                if scale_arr is not None:
                    inv_scale = 1.0 / (np.asarray(scale_arr) + 1e-12)
                else:
                    inv_scale = np.ones(W_std.shape[0], dtype=W_std.dtype)
                w_top_orig = w_top_std * inv_scale
                # unit norm
                w_top_orig = w_top_orig / (np.linalg.norm(w_top_orig) + 1e-12)
                tr_norms = np.linalg.norm(X_tr, axis=1) + 1e-12
                te_norms = np.linalg.norm(X_te, axis=1) + 1e-12
                tr_cos = np.abs((X_tr @ w_top_orig) / tr_norms)
                te_cos = np.abs((X_te @ w_top_orig) / te_norms)
                nudge_tr_means.append(float(np.mean(tr_cos)))
                nudge_te_means.append(float(np.mean(te_cos)))

                if COMPARE_MODE != 'none':
                    want_dom = COMPARE_MODE in {"dom", "both", "all"}
                    want_wdom = COMPARE_MODE in {"whitened_dom", "both", "all"}
                    w_dom_std, w_wdom_std = _compute_dom_vectors_std(Xtr_std, y_tr, reg_eps=REG_EPS)

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

        # Order for lowrank components across folds
        order_idx: Optional[np.ndarray] = None
        if comp_auc_sums is not None and comp_counts is not None and np.any(comp_counts > 0):
            mean_aurocs = comp_auc_sums / np.maximum(comp_counts, 1)
            order_idx = np.argsort(-np.nan_to_num(mean_aurocs, nan=-1.0))
        # Save learned vectors (rank1/lowrank) trained on all data (CV) or train-only (train→test)
        try:
            if str(cfg.CLASSIFIER).lower() in {"rank1", "lowrank"}:
                if n_splits and n_splits > 0:
                    # CV>0: fit once on all data for exporting vectors
                    pipe_full = _build_pipeline(d_model)
                    contrast_ids_all = _build_contrast_ids(df)
                    pipe_full.fit(X, y, subspace__contrast_ids=contrast_ids_all)
                    steps_full = pipe_full.named_steps
                else:
                    # CV=0: reuse the already-fitted pipeline from the single train fold
                    if pipe_for_vectors is None:
                        # Fallback safety: if missing (should not happen), fit once on train split
                        tr0, _te0 = base_splits[0]
                        pipe_for_vectors = _build_pipeline(d_model)
                        contrast_ids_tr0 = _build_contrast_ids(df.iloc[tr0])
                        pipe_for_vectors.fit(X[tr0], y[tr0], subspace__contrast_ids=contrast_ids_tr0)
                    steps_full = pipe_for_vectors.named_steps
                if 'subspace' in steps_full:
                    scaler: StandardScaler = steps_full['scale']  # type: ignore
                    sub: SupervisedSubspace = steps_full['subspace']  # type: ignore
                    U = sub.recover_geometry(scaler)  # (d, k)
                    # Order components by cross-fold AUROC if available
                    if order_idx is not None and U.shape[1] == order_idx.shape[0]:
                        U = U[:, order_idx]
                    # Save each vector separately and also the stack (no split suffix)
                    for j in range(U.shape[1]):
                        np.save(vectors_run_dir / f"L{layer_idx}_top{j+1}.npy", U[:, j])
                    np.save(vectors_run_dir / f"L{layer_idx}_top{U.shape[1]}.stack.npy", U)
        except Exception as e:
            print(f"[WARN] Failed to save vectors for L{layer_idx}: {e}")

        # Add nudge |cos| summary line for this layer
        if nudge_tr_means or nudge_te_means:
            nudge_lines.append(
                f"L{layer_idx:02d} nudge |cos|  train={np.nanmean(nudge_tr_means):.4f}  test={np.nanmean(nudge_te_means):.4f}"
            )

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
    header.append(f"Probe run: {run_name}")
    header.append(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    header.append(f"NPZ: {npz_path}")
    header.append(f"Labels: {labels_csv}")
    device_used = _select_device()
    header.append(f"Hook prefix: {cfg.HOOK_POINT_PREFIX}  |  Mode: {cfg.CLASSIFIER}  |  Device: {device_used}")
    k = 1 if str(cfg.CLASSIFIER).lower()=="rank1" else int(getattr(cfg,'LOWRANK_K',4))
    method = getattr(cfg, 'LOWRANK_METHOD', 'pls' if k>1 else 'lda')
    header.append(f"Subspace: method={method}, k={k}")
    if n_splits and n_splits > 0:
        header.append(f"Grouped CV: {n_splits} folds by '{cfg.GROUP_COL}'  |  N={len(df)} tokens, Examples={df[cfg.GROUP_COL].nunique()}, PosRate={df['_y'].mean():.3f}")
    else:
        n_tr = int((df.get('split','').astype(str) == 'train').sum())
        n_te = int((df.get('split','').astype(str) == 'test').sum())
        header.append(f"Train→Test split by 'split' column  |  N={len(df)} tokens (train={n_tr}, test={n_te}), Examples={df[cfg.GROUP_COL].nunique()}, PosRate={df['_y'].mean():.3f}")

    # Filters
    filters = []
    # Regime filter header note
    regimes_to_use = getattr(cfg, 'REGIMES_TO_USE', None)
    if regimes_to_use is not None:
        filters.append(f"regime in {{{', '.join(map(str, regimes_to_use))}}}")
    rng = getattr(cfg, 'FILTER_OFFSET_RANGE', None)
    if isinstance(rng, tuple) and len(rng) == 2:
        lo, hi = rng
        filters.append(f"{cfg.OFFSET_COL} in [{lo if lo is not None else '-inf'}, {hi if hi is not None else '+inf'}]")
    elif isinstance(rng, list):
        filters.append(f"{cfg.OFFSET_COL} in {{{', '.join(map(str, rng))}}}")
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

                    # Ground-truth and predictions for this regime
                    y_true_reg = df.iloc[pos_idx]['_y'].to_numpy()
                    acc_model = accuracy_score(y_true_reg, y_hat[pos_idx])

                    # Use scores to compute threshold-dependent diagnostics
                    try:
                        y_score_reg = y_score[pos_idx]
                        mask = np.isfinite(y_score_reg)
                        # Default threshold: 0.5 for probabilities, 0.0 for decision_function
                        thr_default = 0.5 if (np.nanmin(y_score_reg) >= 0.0 and np.nanmax(y_score_reg) <= 1.0) else 0.0
                        if np.any(mask):
                            # AUROC (threshold-free)
                            auc = roc_auc_score(y_true_reg[mask], y_score_reg[mask]) if np.unique(y_true_reg).size > 1 else float('nan')
                            # Accuracy at default threshold computed from scores
                            y_pred_def = (y_score_reg >= thr_default).astype(int)
                            acc_default = accuracy_score(y_true_reg[mask], y_pred_def[mask])
                            # Best-threshold accuracy (optimistic upper bound for understanding)
                            uniq = np.unique(y_score_reg[mask])
                            # If too many unique values, subsample thresholds for speed
                            if uniq.size > 512:
                                quantiles = np.linspace(0.0, 1.0, 513)
                                cand_thr = np.quantile(y_score_reg[mask], quantiles)
                            else:
                                cand_thr = uniq
                            best_acc = 0.0
                            for t in cand_thr:
                                pred_t = (y_score_reg[mask] >= t).astype(int)
                                a = accuracy_score(y_true_reg[mask], pred_t)
                                if a > best_acc:
                                    best_acc = a
                            # Balanced accuracy at default threshold (robust to class imbalance)
                            from sklearn.metrics import balanced_accuracy_score
                            bacc = balanced_accuracy_score(y_true_reg[mask], (y_score_reg[mask] >= thr_default).astype(int))
                        else:
                            auc = float('nan'); acc_default = float('nan'); best_acc = float('nan'); bacc = float('nan')
                    except Exception:
                        auc = float('nan'); acc_default = float('nan'); best_acc = float('nan'); bacc = float('nan')

                    prev = float(np.mean(y_true_reg)) if y_true_reg.size else float('nan')
                    per_regime_lines.append(
                        f"  - {reg:<18} Acc={acc_model:.3f}  Acc@thr={acc_default:.3f}  Acc@best={best_acc:.3f}  BAcc={bacc:.3f}  Prev={prev:.3f}  AUROC={auc:.3f}"
                    )
    except Exception as e:
        print(f"[WARN] Could not compute per-regime summary: {e}")

    _write_report(report_txt, header, table_lines, per_regime_lines, nudge_lines, compare_lines, cfg.N_SPLITS)
    print(f"[DONE] Report: {report_txt}")
    print(f"[DONE] Scores: {scores_csv}")


if __name__ == "__main__":
    main()
