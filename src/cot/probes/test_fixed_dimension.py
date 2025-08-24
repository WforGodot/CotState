from __future__ import annotations

"""
Test a fixed direction or subspace (learned elsewhere) on a held-out test set.

Configuration:
- In fixed_test_config.py, set VECTOR_FILE as either:
  - a path relative to the repo root (e.g., "src/cot/outputs/vectors/.../L15_top1.npy"), or
  - a path relative to VECTORS_DIR (fallback).
  The file may contain a 1D array (d,) for a single direction or a 2D array
  (d, k) for a k-dimensional subspace (columns are components).
- Optionally set COMPONENT_INDEX in fixed_test_config to choose a single column out of a stacked file
  (0-based). Leave as None to use all columns.
- If the layer cannot be inferred from the filename (pattern 'L{num}'), set LAYER_ID in fixed_test_config.

Data selection and evaluation (offset, regimes, split names) are controlled by
fixed_test_config.py (separate from run_geom).

The script will:
- Load the activations NPZ and labels per fixed_test_config, apply filters.
- Split by explicit train/test column in labels; fit the classifier on train, evaluate on the full test set only.
- Load the fixed direction(s) and project features onto it/them; train a simple classifier on the projections (direction fixed).
- Report accuracy/AUROC/AP/F1 for the test set and save a small text report under cfg.OUT_DIR.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

import fixed_test_config as cfg


# ========================= User configuration =========================

# Base vectors directory (same convention as compare_vectors/run_geom)
BASE_VECTORS_DIR = (Path(__file__).resolve().parent / getattr(cfg, 'VECTORS_DIR', Path('../outputs/vectors'))).resolve()

# Repo root (three levels up from this file: probes -> cot -> src -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _report_header(npz_path: Path, labels_csv: Path, layer_id: int, vec_path: Path, k: int, n_train: int, n_test: int, pos_rate_test: float) -> list[str]:
    header: list[str] = []
    header.append(f"Fixed-direction test run (train→test)")
    header.append(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    header.append(f"NPZ: {npz_path}")
    header.append(f"Labels: {labels_csv}")
    header.append(f"Layer: {layer_id}")
    header.append(f"Direction file: {vec_path}")
    header.append(f"Dimensionality k: {k}")
    header.append(f"Splits: train={n_train}  test={n_test}  |  Test PosRate={pos_rate_test:.3f}")
    # Filters summary
    filters = []
    regs = getattr(cfg, 'REGIMES_TO_USE', None)
    if regs is not None:
        filters.append(f"regime in {{{', '.join(map(str, regs))}}}")
    rng = getattr(cfg, 'FILTER_OFFSET_RANGE', None)
    if isinstance(rng, tuple) and len(rng) == 2:
        lo, hi = rng
        filters.append(f"{cfg.OFFSET_COL} in [{lo if lo is not None else '-inf'}, {hi if hi is not None else '+inf'}]")
    elif isinstance(rng, list):
        filters.append(f"{cfg.OFFSET_COL} in {{{', '.join(map(str, rng))}}}")
    if getattr(cfg, 'FILTER_OFFSET_MAX', None) is not None:
        filters.append(f"{cfg.OFFSET_COL} <= {cfg.FILTER_OFFSET_MAX}")
    if getattr(cfg, 'FILTER_OFFSET_EQ', None) is not None:
        filters.append(f"{cfg.OFFSET_COL} == {cfg.FILTER_OFFSET_EQ}")
    if filters:
        header.append("Filter: " + "; ".join(filters))
    return header


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

    # Optional regime selection
    regimes_to_use = getattr(cfg, 'REGIMES_TO_USE', None)
    if regimes_to_use is not None:
        if cfg.REGIME_COL not in df.columns:
            raise ValueError(f"Requested regime filtering but '{cfg.REGIME_COL}' not present in labels.")
        keep = set(regimes_to_use)
        df = df[df[cfg.REGIME_COL].isin(keep)]

    # Offset-based filtering
    off_col = cfg.OFFSET_COL
    has_any_filter = (
        getattr(cfg, 'FILTER_OFFSET_EQ', None) is not None
        or getattr(cfg, 'FILTER_OFFSET_MAX', None) is not None
        or getattr(cfg, 'FILTER_OFFSET_RANGE', None) is not None
    )
    if has_any_filter:
        if off_col not in df.columns:
            raise ValueError(f"Requested offset filtering but '{off_col}' not present in labels.")
        df[off_col] = pd.to_numeric(df[off_col], errors='coerce')
        df = df.dropna(subset=[off_col])
        rng = getattr(cfg, 'FILTER_OFFSET_RANGE', None)
        if isinstance(rng, tuple) and len(rng) == 2:
            lo, hi = rng
            if lo is not None:
                df = df[df[off_col] >= lo]
            if hi is not None:
                df = df[df[off_col] <= hi]
        elif isinstance(rng, list):
            try:
                whitelist = set(int(x) for x in rng)
                df = df[df[off_col].isin(whitelist)]
            except Exception:
                whitelist = set(str(x) for x in rng)
                df = df[df[off_col].astype(str).isin(whitelist)]
        maxv = getattr(cfg, 'FILTER_OFFSET_MAX', None)
        if maxv is not None:
            df = df[df[off_col] <= maxv]
        eqv = getattr(cfg, 'FILTER_OFFSET_EQ', None)
        if eqv is not None:
            df = df[df[off_col] == eqv]
    return df


def _layer_keys(npz: np.lib.npyio.NpzFile) -> dict[int, str]:
    pat = re.compile(rf"^{re.escape(cfg.HOOK_POINT_PREFIX)}(\d+)$")
    out: dict[int, str] = {}
    for k in npz.files:
        m = pat.match(k)
        if m:
            out[int(m.group(1))] = k
    if not out:
        raise ValueError(
            f"No layer keys found with prefix '{cfg.HOOK_POINT_PREFIX}'. Keys present: {list(npz.files)[:5]} ..."
        )
    return out


def main():
    vector_file = getattr(cfg, 'VECTOR_FILE', None)
    if vector_file in (None, '', 'REPLACE_ME.npy'):
        raise SystemExit("Please set VECTOR_FILE in fixed_test_config.py to point to your saved direction file (repo-root-relative or relative to VECTORS_DIR).")

    # Resolve VECTOR_FILE as either repo-root-relative or relative to BASE_VECTORS_DIR
    vf_raw = str(vector_file)
    vf_norm = vf_raw.replace("\\", "/")
    vf_path = Path(vf_norm)
    if vf_path.is_absolute():
        vec_path = vf_path
    else:
        candidate_repo = (REPO_ROOT / vf_path).resolve()
        if candidate_repo.exists():
            vec_path = candidate_repo
        else:
            vec_path = (BASE_VECTORS_DIR / vf_path).resolve()
    if not vec_path.exists():
        raise FileNotFoundError(f"Vector file not found: {vec_path}")

    # Load labels and acts
    npz_path = (Path(__file__).parent / cfg.ACTS_NPZ).resolve()
    labels_csv = cfg.LABELS_CSV or _infer_labels_csv(npz_path)
    if labels_csv is None:
        raise FileNotFoundError("Could not infer labels CSV; set LABELS_CSV in fixed_test_config.py")
    labels_csv = (Path(__file__).parent / labels_csv).resolve()

    df = _read_labels(labels_csv)
    if 'split' not in df.columns:
        raise ValueError("Labels CSV missing 'split' column; re-run collect_activations with splitting enabled.")
    train_name = str(getattr(cfg, 'TRAIN_SPLIT_NAME', 'train'))
    test_name = str(getattr(cfg, 'TEST_SPLIT_NAME', 'test'))
    is_tr = (df['split'].astype(str) == train_name).to_numpy()
    is_te = (df['split'].astype(str) == test_name).to_numpy()
    if not is_tr.any() or not is_te.any():
        raise ValueError("Train or test split is empty after filtering; adjust filters or splits.")

    row_idx = df.index.to_numpy()
    y = df['_y'].to_numpy()

    npz = np.load(npz_path)
    layer_keys = _layer_keys(npz)  # map layer->key

    # Determine layer ID
    layer_override = getattr(cfg, 'LAYER_ID', None)
    lid = layer_override if layer_override is not None else _parse_layer_from_name(vec_path.name)
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
    comp_idx = getattr(cfg, 'COMPONENT_INDEX', None)
    W = _load_vectors(vec_path, comp_idx)  # (d, k)
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

    # Train/test split on projections
    Z_tr, Z_te = Z[is_tr], Z[is_te]
    y_tr, y_te = y[is_tr], y[is_te]

    scaler = StandardScaler(with_mean=True, with_std=True)
    Z_trs = scaler.fit_transform(Z_tr)
    Z_tes = scaler.transform(Z_te)

    clf = LogisticRegression(max_iter=2000, C=getattr(cfg, 'LOGREG_C', 1.0), solver="lbfgs", class_weight="balanced", n_jobs=getattr(cfg, 'N_JOBS', None))
    clf.fit(Z_trs, y_tr)

    y_hat = clf.predict(Z_tes)
    try:
        score = clf.decision_function(Z_tes)
    except Exception:
        try:
            proba = clf.predict_proba(Z_tes); score = proba[:, 1]
        except Exception:
            score = None

    # Metrics on full test set
    m_acc = accuracy_score(y_te, y_hat)
    m_f1 = f1_score(y_te, y_hat, zero_division=0)
    if score is not None and len(np.unique(y_te)) > 1:
        try:
            m_auc = roc_auc_score(y_te, score)
            m_ap = average_precision_score(y_te, score)
        except Exception:
            m_auc = float('nan'); m_ap = float('nan')
    else:
        m_auc = float('nan'); m_ap = float('nan')

    header = _report_header(npz_path, labels_csv, lid, vec_path, W.shape[1], int(is_tr.sum()), int(is_te.sum()), float(df[is_te]['_y'].mean()))

    # Per-regime breakdown (optional) computed on test set
    per_regime_lines: list[str] = []
    try:
        if cfg.REGIME_COL in df.columns:
            te_df = df[is_te]
            te_order_idx = te_df.index.to_numpy()  # order aligns with y_hat/score
            for reg, sub in te_df.groupby(cfg.REGIME_COL):
                sub_idx = sub.index.to_numpy()
                mask_rows = np.isin(te_order_idx, sub_idx)
                if not mask_rows.any():
                    acc = float('nan'); auc = float('nan')
                else:
                    y_true_reg = df.loc[te_order_idx[mask_rows], '_y'].to_numpy()
                    y_hat_reg = y_hat[mask_rows]
                    acc = accuracy_score(y_true_reg, y_hat_reg)
                    try:
                        if score is not None:
                            y_score_reg = score[mask_rows]
                            msk = np.isfinite(y_score_reg)
                            if np.unique(y_true_reg).size > 1 and msk.any():
                                auc = roc_auc_score(y_true_reg[msk], y_score_reg[msk])
                            else: auc = float('nan')
                        else:
                            auc = float('nan')
                    except Exception:
                        auc = float('nan')
                per_regime_lines.append(f"  - {reg:<18} Acc={acc:.3f}  AUROC={auc:.3f}")
    except Exception:
        pass

    # Compose final report text
    lines: list[str] = []
    lines.extend(header)
    lines.append("")
    lines.append("Test-set metrics:")
    lines.append(f"  Acc={m_acc:.3f}  AUROC={m_auc:.3f}  AP={m_ap:.3f}  F1={m_f1:.3f}")
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
