from __future__ import annotations

"""
Compare two vector sets by cosine similarity and write a report.

Configuration is inline below. Set FILE_A and FILE_B to filenames located under
src/cot/outputs/vectors/ (relative to this repo). Files may be either a single
vector saved as 1D (d,) or a matrix (d, k) with k components stacked columnwise.

Outputs a TXT report under src/cot/outputs/vectors/compare_reports/ with a
descriptive filename capturing the two inputs and a timestamp.
"""

import os
import re
import math
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, List

import numpy as np


# ========================= User config =========================

# Base directory where vector files live (do not include trailing slash)
BASE_VECTORS_DIR = Path(__file__).resolve().parents[2] / "cot" / "outputs" / "vectors"

# Example values:
#   FILE_A = "L12_top3.stack.npy"
#   FILE_B = "L12_top3.stack.npy"
FILE_A = "L12__off_0_to_5__regs_v_output__Qwen-Qwen3-0.6B__3\L12_split1_top1.npy"
FILE_B = "L12__off_0_to_5__regs_v_output__Qwen-Qwen3-0.6B__3\L12_split2_top1.npy"

# Where to put the comparison reports (TXT)
REPORTS_DIR = BASE_VECTORS_DIR / "compare_reports"


# ========================= Helpers =========================

def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]", "-", s)


def _load_vectors(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array in {path}, got shape {arr.shape}")
    return arr


def _maybe_align_dims(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure both arrays share the same feature dimension on axis 0 (d).
    If not, try transposing one if that fixes it; otherwise raise.
    """
    if a.shape[0] == b.shape[0]:
        return a, b
    if a.T.shape[0] == b.shape[0]:
        return a.T, b
    if b.T.shape[0] == a.shape[0]:
        return a, b.T
    raise ValueError(
        f"Could not align dims: A {a.shape} vs B {b.shape}. Expect (d,k1) and (d,k2)."
    )


def _normalize_columns(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=0) + eps
    return x / n


def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A_n = _normalize_columns(A)
    B_n = _normalize_columns(B)
    C = A_n.T @ B_n  # shape (k1, k2)
    return C, np.abs(C)


def _best_matches(C: np.ndarray) -> List[Tuple[int, int, float]]:
    # For each column in A (rows in C), get best column in B
    out = []
    for i in range(C.shape[0]):
        j = int(np.argmax(C[i]))
        out.append((i, j, float(C[i, j])))
    return out


def _mutual_best(C: np.ndarray) -> List[Tuple[int, int, float]]:
    row_best = np.argmax(C, axis=1)
    col_best = np.argmax(C, axis=0)
    out = []
    for i, j in enumerate(row_best):
        if col_best[j] == i:
            out.append((i, j, float(C[i, j])))
    # Sort by descending similarity
    out.sort(key=lambda t: -abs(t[2]))
    return out


def _hungarian_max(C_abs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maximize total absolute similarity using Hungarian algorithm when available.
    Falls back to greedy matching if SciPy is not installed.
    Returns (rows_idx, cols_idx) arrays for matched pairs.
    """
    try:
        # Prefer SciPy if present
        from scipy.optimize import linear_sum_assignment  # type: ignore

        # We need a square cost matrix. Pad with zeros (which corresponds to abs cos = 0).
        k1, k2 = C_abs.shape
        n = max(k1, k2)
        pad = np.zeros((n, n), dtype=float)
        pad[:k1, :k2] = C_abs
        # Maximize similarity => minimize (1 - sim)
        cost = 1.0 - pad
        r, c = linear_sum_assignment(cost)
        # Keep only real pairs within original sizes
        mask = (r < k1) & (c < k2)
        return r[mask], c[mask]
    except Exception:
        # Greedy fallback: repeatedly take the current max and invalidate its row/col
        C = C_abs.copy()
        k1, k2 = C.shape
        used_r = np.zeros(k1, dtype=bool)
        used_c = np.zeros(k2, dtype=bool)
        pairs_r: List[int] = []
        pairs_c: List[int] = []
        while True:
            C_masked = C.copy()
            C_masked[used_r, :] = -np.inf
            C_masked[:, used_c] = -np.inf
            i, j = np.unravel_index(np.argmax(C_masked), C_masked.shape)
            if not np.isfinite(C_masked[i, j]):
                break
            pairs_r.append(int(i)); pairs_c.append(int(j))
            used_r[i] = True; used_c[j] = True
            if used_r.all() or used_c.all():
                break
        return np.asarray(pairs_r, dtype=int), np.asarray(pairs_c, dtype=int)


def _summ_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    x = x.astype(float, copy=False)
    return float(np.nanmean(x)), float(np.nanstd(x)), float(np.nanmin(x)), float(np.nanmax(x))


def main():
    base = BASE_VECTORS_DIR
    a_path = base / FILE_A
    b_path = base / FILE_B
    if not a_path.exists():
        raise FileNotFoundError(f"Missing A: {a_path}")
    if not b_path.exists():
        raise FileNotFoundError(f"Missing B: {b_path}")

    A = _load_vectors(a_path)
    B = _load_vectors(b_path)
    A, B = _maybe_align_dims(A, B)
    d, k1 = A.shape; _, k2 = B.shape

    C_signed, C_abs = _cosine_matrix(A, B)
    mean_s, std_s, min_s, max_s = _summ_stats(C_signed)
    mean_a, std_a, min_a, max_a = _summ_stats(C_abs)

    # Best matches (A->B, signed and abs)
    best_A_to_B_signed = _best_matches(C_signed)
    best_A_to_B_abs = _best_matches(C_abs)
    mutual_abs = _mutual_best(C_abs)

    # Hungarian matching on absolute similarities
    r_idx, c_idx = _hungarian_max(C_abs)
    matched = [(int(i), int(j), float(C_signed[i, j]), float(C_abs[i, j])) for i, j in zip(r_idx, c_idx)]
    matched.sort(key=lambda t: -t[3])  # sort by abs cos desc
    avg_abs_matched = float(np.mean([m[3] for m in matched])) if matched else float("nan")

    # Attempt to parse layer ids from filenames
    def _layer_of(name: str) -> Optional[int]:
        m = re.search(r"L(\d+)", name)
        return int(m.group(1)) if m else None

    L_a = _layer_of(FILE_A)
    L_b = _layer_of(FILE_B)

    # Report path
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep_name = f"compare__{_sanitize(FILE_A)}__VS__{_sanitize(FILE_B)}__{ts}.txt"
    rep_path = REPORTS_DIR / rep_name

    # Write report
    with rep_path.open("w", encoding="utf-8") as f:
        f.write(f"Vector compare report\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"A: {a_path}\n")
        f.write(f"B: {b_path}\n")
        if L_a is not None or L_b is not None:
            f.write(f"Layer(A): {L_a if L_a is not None else 'n/a'}  |  Layer(B): {L_b if L_b is not None else 'n/a'}\n")
        f.write(f"Dims: d={d}, kA={k1}, kB={k2}\n")
        f.write("\n")
        f.write("Cosine similarity stats (signed):\n")
        f.write(f"  mean={mean_s:.4f}  std={std_s:.4f}  min={min_s:.4f}  max={max_s:.4f}\n")
        f.write("Cosine similarity stats (abs):\n")
        f.write(f"  mean={mean_a:.4f}  std={std_a:.4f}  min={min_a:.4f}  max={max_a:.4f}\n")
        f.write("\n")

        # Top per-A matches by abs cosine
        f.write("Best matches per A component (by abs cosine):\n")
        for i, j, val in sorted(best_A_to_B_abs, key=lambda t: -abs(t[2]))[: min(k1, 20)]:
            f.write(f"  A[{i}] ↔ B[{j}]  abs_cos={abs(val):.4f}  signed={C_signed[i,j]:.4f}\n")
        f.write("\n")

        # Mutual best matches
        f.write("Mutual best matches (by abs cosine):\n")
        if mutual_abs:
            for i, j, val in mutual_abs[: min(len(mutual_abs), 20)]:
                f.write(f"  A[{i}] ⇔ B[{j}]  abs_cos={abs(val):.4f}  signed={C_signed[i,j]:.4f}\n")
        else:
            f.write("  (none)\n")
        f.write("\n")

        # Hungarian matched pairs (absolute)
        f.write("Hungarian matching on abs cosine (pairs up to min(kA,kB)):\n")
        f.write(f"  Avg(abs cos) over matches: {avg_abs_matched:.4f}\n")
        for i, j, cs, ca in matched[: min(len(matched), 50)]:
            f.write(f"  A[{i}] ↔ B[{j}]  abs_cos={ca:.4f}  signed={cs:.4f}\n")

    print(f"[DONE] Wrote report: {rep_path}")


if __name__ == "__main__":
    main()

