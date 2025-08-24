from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import re
import json

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

import probes_config as cfg
import run_geom as rg  # reuse helpers

try:
    from tqdm.auto import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False
    def tqdm(x, *args, **kwargs):
        return x


def _safe_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
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


def main():
    ap = argparse.ArgumentParser(description="Train per-layer probes on explicit train split; eval on test split.")
    ap.add_argument("--save_geometry", action="store_true")
    cli = ap.parse_args()

    npz_path = (Path(__file__).parent / cfg.ACTS_NPZ).resolve()
    labels_csv = cfg.LABELS_CSV or rg._infer_labels_csv(npz_path)
    if labels_csv is None:
        raise FileNotFoundError("Could not infer labels CSV; set LABELS_CSV in probes_config.py")
    labels_csv = (Path(__file__).parent / labels_csv).resolve()

    out_dir = (Path(__file__).parent / cfg.OUT_DIR).resolve()

    df = rg._read_labels(labels_csv)
    if 'split' not in df.columns:
        raise ValueError("Labels CSV missing 'split' column; re-run collect_activations with splitting enabled.")
    train_name = str(getattr(cfg, 'TRAIN_SPLIT_NAME', 'train'))
    test_name = str(getattr(cfg, 'TEST_SPLIT_NAME', 'test'))

    row_idx = df.index.to_numpy()
    y = df["_y"].to_numpy()

    npz = np.load(npz_path)
    layers = rg._layer_keys(npz)

    # Build run name
    def _sanitize(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._+-]", "-", s)

    rng = getattr(cfg, 'FILTER_OFFSET_RANGE', None)
    if isinstance(rng, tuple) and len(rng) == 2:
        lo, hi = rng
        off_str = f"off_{lo if lo is not None else 'min'}_to_{hi if hi is not None else 'max'}"
    elif isinstance(rng, list):
        vals = "+".join(str(x) for x in rng)
        off_str = f"off_in_{vals if vals else 'none'}"
    else:
        off_str = "off_all"

    regs = getattr(cfg, 'REGIMES_TO_USE', None)
    regs_str = "regs_all" if regs is None else ("regs_" + "+".join(_sanitize(str(r)) for r in regs))

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

    layers_used = [t[0] for t in layers]
    layers_str = "L" + ("-".join(str(x) for x in layers_used) if layers_used else "none")
    base_name = f"{layers_str}__{off_str}__{regs_str}__{model_str}__train_test"
    run_name = base_name
    k_suffix = 1
    while (out_dir / run_name).exists():
        k_suffix += 1
        run_name = f"{base_name}__{k_suffix}"
    run_dir = out_dir / run_name
    report_txt = run_dir / "report.txt"
    scores_csv = run_dir / "layer_scores.csv"
    (Path(out_dir) / run_name).mkdir(parents=True, exist_ok=True)

    vectors_base = (Path(__file__).parent / getattr(cfg, 'VECTORS_DIR', Path("../outputs/vectors"))).resolve()
    vectors_run_dir = vectors_base / run_name
    vectors_run_dir.mkdir(parents=True, exist_ok=True)

    hdr = cfg.COL_WIDTHS
    table_lines: List[str] = []
    table_lines.append(f"{'layer':>{hdr['layer']}}{'N':>{hdr['n']}}{'PCs':>{hdr['comps']}}{'Acc':>{hdr['acc']}}{'AUROC':>{hdr['auroc']}}{'AP':>{hdr['ap']}}{'F1':>{hdr['f1']}}\n")
    table_lines.append("-" * sum(hdr.values()) + "\n")

    rows_for_csv: List[Dict] = []

    for layer_idx, key in (tqdm(layers, desc="Layers") if _HAS_TQDM else layers):
        X = npz[key]
        if len(X) != len(df):
            X = X[row_idx]
            if len(X) != len(df):
                raise ValueError(f"After subsetting, length mismatch for L{layer_idx}: {len(X)} vs {len(df)}")
        X = X.astype(np.float32, copy=False)
        d_model = X.shape[1]
        pipe = rg._build_pipeline(d_model)

        is_tr = (df['split'].astype(str) == train_name).to_numpy()
        is_te = (df['split'].astype[str] if False else df['split'].astype(str))  # keep mypy quiet
        is_te = (df['split'].astype(str) == test_name).to_numpy()
        X_tr, y_tr = X[is_tr], y[is_tr]
        X_te, y_te = X[is_te], y[is_te]
        if len(X_tr) == 0 or len(X_te) == 0:
            raise ValueError("Train/test split yielded empty set for a layer.")

        pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_te)
        try:
            score = pipe.decision_function(X_te)
        except Exception:
            try:
                proba = pipe.predict_proba(X_te); score = proba[:, 1]
            except Exception:
                score = None
        m = _safe_metrics(y_te, y_hat, score)

        ncomp_for_layer: Optional[int] = None
        if 'pca' in pipe.named_steps:
            ncomp_for_layer = int(getattr(pipe.named_steps['pca'], 'n_components_', 0))
        elif 'subspace' in pipe.named_steps:
            ncomp_for_layer = int(pipe.named_steps['subspace'].k)
        else:
            ncomp_for_layer = d_model

        table_lines.append(
            f"{layer_idx:>{hdr['layer']}}{len(X_te):>{hdr['n']}}{ncomp_for_layer:>{hdr['comps']}}"
            f"{m['accuracy']*100:>{hdr['acc']}.2f}{(m.get('auroc', float('nan'))*100):>{hdr['auroc']}.2f}"
            f"{(m.get('ap', float('nan'))*100):>{hdr['ap']}.2f}{m['f1']*100:>{hdr['f1']}.2f}\n"
        )
        rows_for_csv.append(dict(layer=layer_idx, n=len(X_te), d_model=d_model, **m))

        # Save geometry (rank1/lowrank) trained on train split
        try:
            if str(cfg.CLASSIFIER).lower() in {"rank1", "lowrank"}:
                steps = pipe.named_steps
                if 'subspace' in steps:
                    scaler: StandardScaler = steps['scale']  # type: ignore
                    sub = steps['subspace']  # type: ignore
                    U = sub.recover_geometry(scaler)
                    for j in range(U.shape[1]):
                        np.save(vectors_run_dir / f"L{layer_idx}_top{j+1}.npy", U[:, j])
                    np.save(vectors_run_dir / f"L{layer_idx}_top{U.shape[1]}.stack.npy", U)
        except Exception as e:
            print(f"[WARN] Failed to save vectors for L{layer_idx}: {e}")

    # Write outputs
    pd.DataFrame(rows_for_csv).sort_values("layer").to_csv(scores_csv, index=False)
    header: List[str] = []
    header.append(f"Probe run (train/test): {run_name}")
    header.append(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    header.append(f"NPZ: {npz_path}")
    header.append(f"Labels: {labels_csv}")
    header.append(f"Hook prefix: {cfg.HOOK_POINT_PREFIX}  |  Mode: {cfg.CLASSIFIER}")
    header.append(f"Train split: '{train_name}', Test split: '{test_name}' | N_train={int(is_tr.sum())}, N_test={int(is_te.sum())}")
    report_txt.write_text("\n".join(header) + "\n\n" + "".join(table_lines), encoding="utf-8")
    print(f"[INFO] Wrote per-layer scores CSV → {scores_csv}")
    print(f"[INFO] Wrote report → {report_txt}")


if __name__ == "__main__":
    main()
