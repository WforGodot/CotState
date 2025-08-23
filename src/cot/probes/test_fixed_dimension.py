from __future__ import annotations

"""
Test a fixed direction or subspace (learned elsewhere) on a new setting.

Configuration:
- Set VECTOR_FILE to a .npy under the vectors base dir (see BASE_VECTORS_DIR below).
  The file may contain a 1D array (d,) for a single direction or a 2D array
  (d, k) for a k-dimensional subspace (columns are components).
- Optionally set COMPONENT_INDEX to choose a single column out of a stacked file
  (0-based). Leave as None to use all columns.
- If the layer cannot be inferred from the filename (pattern 'L{num}'), set LAYER_ID.

Data selection and evaluation parameters (offset, regimes, sampling, CV splits, etc.)
are controlled by probes_config.py, same as run_geom.

The script will:
- Load the activations NPZ and labels per probes_config, apply filters.
- Load the fixed direction(s) and project features onto it/them.
- Run grouped K-fold CV, fitting only a simple classifier on the projection(s)
  (direction(s) fixed), and compute accuracy/AUROC/AP/F1.
- Print a report similar to run_geom and save it under cfg.OUT_DIR.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import probes_config as cfg
import run_geom  # reuse helpers for reading labels and layer keys


# ========================= User configuration =========================

# Base vectors directory (same convention as compare_vectors/run_geom)
BASE_VECTORS_DIR = (Path(__file__).resolve().parent / getattr(cfg, 'VECTORS_DIR', Path('../outputs/vectors'))).resolve()

# Relative to BASE_VECTORS_DIR. Example: "L12__off_0_to_10__regs_all__unknownmodel/L12_split1_top4.stack.npy"
VECTOR_FILE = "L12__off_0_to_10__regs_i_initial+iii_derived+v_output__Qwen-Qwen3-0.6B__2/L12_split1_top1.npy"  # set this to your vector file

# If VECTOR_FILE is a stack (d, k), optionally choose a single component by index (0-based).
COMPONENT_INDEX: Optional[int] = None  # e.g., 0 for top1; None to use all

# If layer cannot be parsed from filename as L{num}, set it here
LAYER_ID: Optional[int] = None


# ========================= Helpers =========================

def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]", "-", s)


def _parse_layer_from_name(name: str) -> Optional[int]:
    m = re.search(r"L(\d+)", name)
    return int(m.group(1)) if m else None


def _load_vectors(path: Path, comp_idx: Optional[int]) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D vector file, got shape {arr.shape} at {path}")
    if comp_idx is not None:
        if not (0 <= comp_idx < arr.shape[1]):
            raise ValueError(f"comp_idx {comp_idx} out of range for shape {arr.shape}")
        arr = arr[:, comp_idx:comp_idx+1]
    return arr  # shape (d, k)


def _report_header(npz_path: Path, labels_csv: Path, layer_id: int, vec_path: Path, k: int, n_tokens: int, n_examples: int, pos_rate: float) -> list[str]:
    header: list[str] = []
    header.append(f"Fixed-direction test run")
    header.append(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    header.append(f"NPZ: {npz_path}")
    header.append(f"Labels: {labels_csv}")
    header.append(f"Layer: {layer_id}")
    header.append(f"Direction file: {vec_path}")
    header.append(f"Dimensionality k: {k}")
    header.append(f"Grouped CV: {cfg.N_SPLITS} folds by '{cfg.GROUP_COL}'  |  N={n_tokens} tokens, Examples={n_examples}, PosRate={pos_rate:.3f}")
    # Filters summary
    filters = []
    regs = getattr(cfg, 'REGIMES_TO_USE', None)
    if regs is not None:
        filters.append(f"regime in {{{', '.join(map(str, regs))}}}")
    rng = getattr(cfg, 'FILTER_OFFSET_RANGE', None)
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        lo, hi = rng
        filters.append(f"{cfg.OFFSET_COL} in [{lo if lo is not None else '-inf'}, {hi if hi is not None else '+inf'}]")
    if getattr(cfg, 'FILTER_OFFSET_MAX', None) is not None:
        filters.append(f"{cfg.OFFSET_COL} <= {cfg.FILTER_OFFSET_MAX}")
    if getattr(cfg, 'FILTER_OFFSET_EQ', None) is not None:
        filters.append(f"{cfg.OFFSET_COL} == {cfg.FILTER_OFFSET_EQ}")
    if filters:
        header.append("Filter: " + "; ".join(filters))
    return header


def main():
    if VECTOR_FILE == "REPLACE_ME.npy":
        raise SystemExit("Please edit VECTOR_FILE at the top of this script to point to your saved direction file.")

    vec_path = (BASE_VECTORS_DIR / VECTOR_FILE).resolve()
    if not vec_path.exists():
        raise FileNotFoundError(f"Vector file not found: {vec_path}")

    # Load labels and acts, as in run_geom
    npz_path = (Path(__file__).parent / cfg.ACTS_NPZ).resolve()
    labels_csv = cfg.LABELS_CSV or run_geom._infer_labels_csv(npz_path)
    if labels_csv is None:
        raise FileNotFoundError("Could not infer labels CSV; set LABELS_CSV in probes_config.py")
    labels_csv = (Path(__file__).parent / labels_csv).resolve()

    df = run_geom._read_labels(labels_csv)
    if getattr(cfg, 'N_TOKENS', None) and len(df) > int(getattr(cfg, 'N_TOKENS')):
        df = df.sample(n=int(getattr(cfg, 'N_TOKENS')), random_state=getattr(cfg, 'RANDOM_STATE', 0))

    row_idx = df.index.to_numpy()
    y = df['_y'].to_numpy()
    groups = df[cfg.GROUP_COL].to_numpy()

    npz = np.load(npz_path)
    layer_keys = dict(run_geom._layer_keys(npz))  # map layer->key

    # Determine layer ID
    lid = LAYER_ID if LAYER_ID is not None else _parse_layer_from_name(vec_path.name)
    if lid is None:
        raise ValueError("Could not infer layer id from filename and LAYER_ID not set. Expected 'L{num}' in filename.")
    if lid not in layer_keys:
        raise ValueError(f"Layer {lid} not found in NPZ. Available: {sorted(layer_keys.keys())[:10]} ...")

    # Load features for that layer and subset to df rows
    X = npz[layer_keys[lid]]
    if len(X) != len(df):
        X = X[row_idx]
        if len(X) != len(df):
            raise ValueError(f"After subsetting, length mismatch for L{lid}: {len(X)} vs {len(df)}")
    X = X.astype(np.float32, copy=False)

    # Load fixed direction(s) in input space (same space as X)
    W = _load_vectors(vec_path, COMPONENT_INDEX)  # (d, k)
    if X.shape[1] != W.shape[0]:
        raise ValueError(f"Feature dim mismatch: X has d={X.shape[1]} but vectors have d={W.shape[0]}")

    # Try to locate sidecar scaler stats next to the vector file
    scaler_mean_path = vec_path.with_name("scaler_mean.npy")
    scaler_scale_path = vec_path.with_name("scaler_scale.npy")

    # Heuristic: if both scaler files exist and you suspect W is in standardized space,
    # use standardized projection; otherwise assume original-space vector.
    use_std = scaler_mean_path.exists() and scaler_scale_path.exists() and not vec_path.name.endswith(".direction.npy")

    if use_std:
        mean = np.load(scaler_mean_path)
        scale = np.load(scaler_scale_path)
        X_std = (X - mean.reshape(1, -1)) / (scale.reshape(1, -1) + 1e-12)
        Z = X_std @ W
    else:
        Z = X @ W

    # Grouped CV, fitting a simple classifier on Z with direction(s) fixed
    gkf = GroupKFold(n_splits=cfg.N_SPLITS)
    y_pred = np.empty_like(y)
    y_score = np.full_like(y, fill_value=np.nan, dtype=float)
    metrics_rows: list[dict] = []

    for train_idx, test_idx in gkf.split(Z, y, groups):
        Z_tr, Z_te = Z[train_idx], Z[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler(with_mean=True, with_std=True)
        Z_trs = scaler.fit_transform(Z_tr)
        Z_tes = scaler.transform(Z_te)

        clf = LogisticRegression(max_iter=2000, C=cfg.LOGREG_C, solver="lbfgs", class_weight="balanced", n_jobs=getattr(cfg, 'N_JOBS', None))
        clf.fit(Z_trs, y_tr)

        y_hat = clf.predict(Z_tes)
        try:
            score = clf.decision_function(Z_tes)
        except Exception:
            try:
                proba = clf.predict_proba(Z_tes); score = proba[:, 1]
            except Exception:
                score = None

        # Metrics for this fold
        m = dict(accuracy=accuracy_score(y_te, y_hat), f1=f1_score(y_te, y_hat, zero_division=0))
        if score is not None and len(np.unique(y_te)) > 1:
            try:
                m["auroc"] = roc_auc_score(y_te, score)
                m["ap"] = average_precision_score(y_te, score)
            except Exception:
                m["auroc"] = float("nan"); m["ap"] = float("nan")
        else:
            m["auroc"] = float("nan"); m["ap"] = float("nan")

        metrics_rows.append(m)
        y_pred[test_idx] = y_hat
        if score is not None:
            y_score[test_idx] = score

    # Aggregate
    def _avg(name: str, default=np.nan):
        vals = [d.get(name, default) for d in metrics_rows]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float('nan')

    n_tokens = len(df)
    n_examples = df[cfg.GROUP_COL].nunique()
    pos_rate = df['_y'].mean()
    header = _report_header(npz_path, labels_csv, lid, vec_path, W.shape[1], n_tokens, n_examples, pos_rate)

    # Per-regime breakdown (optional)
    per_regime_lines: list[str] = []
    try:
        if cfg.REGIME_COL in df.columns:
            for reg, sub in df.groupby(cfg.REGIME_COL):
                pos_idx = sub.index.to_numpy()
                acc = accuracy_score(df.loc[pos_idx, '_y'], y_pred[pos_idx])
                try:
                    y_true_reg = df.loc[pos_idx, '_y'].to_numpy(); y_score_reg = y_score[pos_idx]
                    mask = np.isfinite(y_score_reg)
                    if np.unique(y_true_reg).size > 1 and mask.any():
                        auc = roc_auc_score(y_true_reg[mask], y_score_reg[mask])
                    else: auc = float('nan')
                except Exception: auc = float('nan')
                per_regime_lines.append(f"  - {reg:<18} Acc={acc:.3f}  AUROC={auc:.3f}")
    except Exception:
        pass

    # Compose final report text
    lines: list[str] = []
    lines.extend(header)
    lines.append("")
    lines.append("Per-fold metrics:")
    for i, m in enumerate(metrics_rows):
        lines.append(f"  Fold {i+1}: Acc={m['accuracy']:.3f}  AUROC={m['auroc']:.3f}  AP={m['ap']:.3f}  F1={m['f1']:.3f}")
    lines.append("")
    lines.append("Averages across folds:")
    lines.append(f"  Acc={_avg('accuracy'):.3f}  AUROC={_avg('auroc'):.3f}  AP={_avg('ap'):.3f}  F1={_avg('f1'):.3f}")
    if per_regime_lines:
        lines.append("")
        lines.append("Per-regime breakdown:")
        lines.extend(per_regime_lines)

    # Print to stdout
    print("\n".join(lines))

    # Also write to a file under OUT_DIR for convenience
    out_dir = (Path(__file__).parent / cfg.OUT_DIR / "fixed_dir_tests").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = _sanitize(vec_path.stem)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f"test_fixed_L{lid}__{base_name}__{ts}.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[SAVED] Report → {out_path}")


if __name__ == "__main__":
    main()

