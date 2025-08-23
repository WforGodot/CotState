from __future__ import annotations

"""
Compare two vector sets by cosine similarity **and** subspace overlap (principal angles), and write a report.

Configuration is inline below. Set FILE_A and FILE_B to filenames located under
src/cot/outputs/vectors/ (relative to this repo). Files may be either a single
vector saved as 1D (d,) or a matrix (d, k) with k components stacked columnwise.

Outputs a TXT report under src/cot/outputs/vectors/compare_reports/ with a
descriptive filename capturing the two inputs and a timestamp.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, List

import numpy as np

# ========================= User config =========================

# Base directory where vector files live (do not include trailing slash)
BASE_VECTORS_DIR = Path(__file__).resolve().parents[2] / "cot" / "outputs" / "vectors"

# Example (Windows-style paths OK). Use raw strings r"..." if needed.
FILE_A = r"L12__off_0_to_10__regs_i_initial+iii_derived+v_output__Qwen-Qwen3-0.6B__2\L12_split1_top4.stack.npy"
FILE_B = r"L12__off_10_to_20__regs_i_initial+iii_derived+v_output__Qwen-Qwen3-0.6B__2\L12_split2_top4.stack.npy"

# Where to put the comparison reports (TXT)
REPORTS_DIR = BASE_VECTORS_DIR / "compare_reports"

# Max number of per-component matches to print
MAX_LIST = 20

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
    """Ensure both arrays have the same feature dimension on axis 0 (d).
    If not, try transposing one; otherwise raise.
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
    C = A_n.T @ B_n  # (k1, k2)
    return C, np.abs(C)


def _best_matches(C: np.ndarray) -> List[Tuple[int, int, float]]:
    out = []
    for i in range(C.shape[0]):
        j = int(np.argmax(np.abs(C[i])))
        out.append((i, j, float(C[i, j])))
    return out


def _mutual_best(C_abs: np.ndarray, C_signed: np.ndarray) -> List[Tuple[int, int, float]]:
    row_best = np.argmax(C_abs, axis=1)
    col_best = np.argmax(C_abs, axis=0)
    out = []
    for i, j in enumerate(row_best):
        if col_best[j] == i:
            out.append((i, j, float(C_signed[i, j])))
    out.sort(key=lambda t: -abs(t[2]))
    return out


def _hungarian_max(C_abs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Maximize total absolute similarity via Hungarian algorithm (if SciPy available),
    else greedily. Returns (rows_idx, cols_idx).
    """
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        k1, k2 = C_abs.shape
        n = max(k1, k2)
        pad = np.zeros((n, n), dtype=float)
        pad[:k1, :k2] = C_abs
        cost = 1.0 - pad
        r, c = linear_sum_assignment(cost)
        mask = (r < k1) & (c < k2)
        return r[mask], c[mask]
    except Exception:
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

# ========================= Subspace analysis =========================

def _orthonormal_basis(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return an orthonormal basis for the column space of X (d,k').
    Uses QR with column pivoting for stability; falls back to np.linalg.qr.
    """
    d, k = X.shape
    # Normalize first to avoid tiny columns causing issues
    Xn = _normalize_columns(X)
    try:
        # scipy.linalg.qr with pivoting is ideal, but avoid hard dependency
        import scipy.linalg as sl  # type: ignore
        Q, R, P = sl.qr(Xn, mode='economic', pivoting=True)
        # Determine numerical rank
        diag = np.abs(np.diag(R))
        r = int(np.sum(diag > eps))
        U = Q[:, :r]
        return U
    except Exception:
        Q, R = np.linalg.qr(Xn)
        diag = np.abs(np.diag(R))
        r = int(np.sum(diag > eps)) if diag.size else 0
        U = Q[:, :max(1, r)]
        return U


def _principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Return singular values s = cos(theta_i) between two subspaces spanned by columns of U and V.
    U and V must have orthonormal columns.
    """
    M = U.T @ V
    s = np.linalg.svd(M, compute_uv=False)
    # Clip for numerical safety
    s = np.clip(s, -1.0, 1.0)
    return s


def _subspace_metrics(A: np.ndarray, B: np.ndarray) -> dict:
    """Compute rotation-invariant subspace similarity metrics between columns of A and B.
    Returns a dict with:
      - kA, kB, k_min
      - cos_singular (array of cos(theta_i))
      - avg_cos, min_cos, max_cos
      - avg_cos2 (projection overlap), chordal_dist, proj_F_norm2
      - sin_max (operator distance between projectors)
    """
    UA = _orthonormal_basis(A)
    UB = _orthonormal_basis(B)
    s = _principal_angles(UA, UB)  # length = k_min
    k_min = int(s.size)
    if k_min == 0:
        return dict(kA=A.shape[1], kB=B.shape[1], k_min=0)
    cos_vals = np.sort(np.abs(s))[::-1]  # largest first
    avg_cos = float(np.mean(cos_vals))
    avg_cos2 = float(np.mean(cos_vals ** 2))
    min_cos = float(np.min(cos_vals))
    max_cos = float(np.max(cos_vals))
    # Chordal distance (normalized): sqrt(k - sum cos^2) / sqrt(k)
    chordal = float(np.sqrt(max(0.0, k_min - float(np.sum(cos_vals ** 2)))) / np.sqrt(k_min))
    # Projector Frobenius distance squared: ||P_U - P_V||_F^2 = 2k - 2 sum cos^2
    proj_F2 = float(2 * k_min - 2 * float(np.sum(cos_vals ** 2)))
    # Operator norm distance between projectors = sin(theta_max)
    sin_max = float(np.sqrt(max(0.0, 1.0 - (float(np.max(cos_vals)) ** 2))))
    return dict(
        kA=A.shape[1], kB=B.shape[1], k_min=k_min,
        cos_singular=cos_vals,
        avg_cos=avg_cos, avg_cos2=avg_cos2,
        min_cos=min_cos, max_cos=max_cos,
        chordal=chordal, proj_F_norm2=proj_F2, sin_max=sin_max,
    )

# ========================= Main =========================

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

    # Component-wise cosine comparisons (rotation-dependent)
    C_signed, C_abs = _cosine_matrix(A, B)
    mean_s, std_s, min_s, max_s = _summ_stats(C_signed)
    mean_a, std_a, min_a, max_a = _summ_stats(C_abs)

    best_A_to_B = _best_matches(C_signed)
    mutual_abs = _mutual_best(C_abs, C_signed)
    r_idx, c_idx = _hungarian_max(C_abs)
    matched = [(int(i), int(j), float(C_signed[i, j]), float(C_abs[i, j])) for i, j in zip(r_idx, c_idx)]
    matched.sort(key=lambda t: -t[3])
    avg_abs_matched = float(np.mean([m[3] for m in matched])) if matched else float("nan")

    # Subspace analysis (rotation-invariant)
    subm = _subspace_metrics(A, B)

    # Try to parse layer IDs from filenames
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

    with rep_path.open("w", encoding="utf-8") as f:
        f.write("Vector compare report\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"A: {a_path}\n")
        f.write(f"B: {b_path}\n")
        if L_a is not None or L_b is not None:
            f.write(f"Layer(A): {L_a if L_a is not None else 'n/a'}  |  Layer(B): {L_b if L_b is not None else 'n/a'}\n")
        f.write(f"Dims: d={d}, kA={k1}, kB={k2}\n\n")

        # ---- Component-wise cosines ----
        f.write("Cosine similarity stats (signed):\n")
        f.write(f"  mean={mean_s:.4f}  std={std_s:.4f}  min={min_s:.4f}  max={max_s:.4f}\n")
        f.write("Cosine similarity stats (abs):\n")
        f.write(f"  mean={mean_a:.4f}  std={std_a:.4f}  min={min_a:.4f}  max={max_a:.4f}\n\n")

        f.write("Best matches per A component (by abs cosine):\n")
        for i, j, val in sorted(best_A_to_B, key=lambda t: -abs(t[2]))[: min(k1, MAX_LIST)]:
            f.write(f"  A[{i}] ↔ B[{j}]  abs_cos={abs(val):.4f}  signed={val:.4f}\n")
        f.write("\n")

        f.write("Mutual best matches (by abs cosine):\n")
        if mutual_abs:
            for i, j, val in mutual_abs[: min(len(mutual_abs), MAX_LIST)]:
                f.write(f"  A[{i}] ⇔ B[{j}]  abs_cos={abs(val):.4f}  signed={val:.4f}\n")
        else:
            f.write("  (none)\n")
        f.write("\n")

        f.write("Hungarian matching on abs cosine (pairs up to min(kA,kB)):\n")
        f.write(f"  Avg(abs cos) over matches: {avg_abs_matched:.4f}\n")
        for i, j, cs, ca in matched[: min(len(matched), MAX_LIST)]:
            f.write(f"  A[{i}] ↔ B[{j}]  abs_cos={ca:.4f}  signed={cs:.4f}\n")
        f.write("\n")

        # ---- Subspace overlap (principal angles) ----
        if subm.get('k_min', 0) > 0:
            cos_vals = subm['cos_singular']
            deg = np.degrees(np.arccos(np.clip(cos_vals, -1.0, 1.0)))
            f.write("Subspace overlap (rotation-invariant):\n")
            f.write(f"  kA={subm['kA']}  kB={subm['kB']}  k_min={subm['k_min']}\n")
            f.write(f"  cos(theta) (top→bottom): {np.array2string(cos_vals, precision=4, separator=', ')}\n")
            f.write(f"  angles_deg (top→bottom): {np.array2string(deg, precision=2, separator=', ')}\n")
            f.write(f"  avg cos={subm['avg_cos']:.4f}  avg cos^2={subm['avg_cos2']:.4f}  min cos={subm['min_cos']:.4f}  max cos={subm['max_cos']:.4f}\n")
            f.write(f"  chordal distance={subm['chordal']:.4f}  proj_F_norm^2={subm['proj_F_norm2']:.4f}  sin(theta_max)={subm['sin_max']:.4f}\n")
        else:
            f.write("Subspace overlap: insufficient rank to compute principal angles.\n")

    print(f"[DONE] Wrote report: {rep_path}")


if __name__ == "__main__":
    main()
