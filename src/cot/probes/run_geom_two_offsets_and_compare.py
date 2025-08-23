from __future__ import annotations

"""
Run run_geom twice with two different offset ranges (top-k subspace), then
compare the resulting split-1 vector stacks using compare_vectors.

Usage:
- Edit the configuration block below (OFFSET_RANGE_A/B, TOP_K, LAYER_TO_COMPARE).
- Run this file directly: `python -m cot.probes.run_geom_two_offsets_and_compare`

Notes:
- This script mutates probes_config in-process between runs to avoid editing files.
- It detects the newly created vectors run directory by diffing the vectors base
  directory before/after each run.
"""

import time
from pathlib import Path
from typing import Tuple, List

import probes_config as cfg
import run_geom
import compare_vectors


# ========================= User configuration =========================

# Offsets are inclusive ranges: (lo, hi). Use None for open-ended.
# Examples: (0, 5), (-5, 0), (None, 3), (2, None)
OFFSET_RANGE_A: Tuple[int | None, int | None] = (0, 5)
OFFSET_RANGE_B: Tuple[int | None, int | None] = (6, 15)

# Subspace dimensionality (top-k). Uses cfg.CLASSIFIER = 'lowrank'.
TOP_K: int = 4

# Layer to compare (must be among cfg.LAYERS_TO_TRAIN or present in NPZ)
LAYER_TO_COMPARE: int = 12


# ========================= Helpers =========================

def _vectors_base_dir() -> Path:
    # Mirror run_geom: vectors live under cfg.VECTORS_DIR relative to this file
    base = (Path(__file__).parent / getattr(cfg, 'VECTORS_DIR', Path("../outputs/vectors"))).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _list_run_dirs(base: Path) -> List[Path]:
    return sorted([p for p in base.iterdir() if p.is_dir()])


def _new_run_dir(before: List[Path], after: List[Path]) -> Path:
    before_set = {p.name for p in before}
    new_dirs = [p for p in after if p.name not in before_set]
    if len(new_dirs) == 1:
        return new_dirs[0]
    if len(new_dirs) == 0:
        # Fallback: pick the most recently modified dir from 'after'
        if not after:
            raise RuntimeError("No vectors directories found after running run_geom.")
        return max(after, key=lambda p: p.stat().st_mtime)
    # If multiple appeared (unlikely), pick the most recent
    return max(new_dirs, key=lambda p: p.stat().st_mtime)


def _run_one_offset(offset_rng: Tuple[int | None, int | None]) -> Path:
    base = _vectors_base_dir()
    before = _list_run_dirs(base)

    # Mutate config for this run
    cfg.CLASSIFIER = "lowrank"
    cfg.LOWRANK_K = int(TOP_K)
    cfg.FILTER_OFFSET_EQ = None
    cfg.FILTER_OFFSET_MAX = None
    cfg.FILTER_OFFSET_RANGE = offset_rng
    # Optional: ensure we train the desired layer only (speeds up)
    try:
        # Make a copy to avoid accidental aliasing if it's a list
        cfg.LAYERS_TO_TRAIN = [int(LAYER_TO_COMPARE)]
    except Exception:
        pass

    # Run geometry pipeline
    run_geom.main()

    # Detect new vectors run directory
    after = _list_run_dirs(base)
    run_dir = _new_run_dir(before, after)
    return run_dir


def _rel_to_base(p: Path) -> str:
    base = _vectors_base_dir()
    return str(p.relative_to(base)).replace("\\", "/")


def main():
    base = _vectors_base_dir()
    print(f"[INFO] Vectors base dir: {base}")

    print(f"[INFO] Running run_geom for OFFSET_RANGE_A={OFFSET_RANGE_A} (top-k={TOP_K})")
    dir_a = _run_one_offset(OFFSET_RANGE_A)
    time.sleep(0.1)
    print(f"[DONE] A vectors: {dir_a}")

    print(f"[INFO] Running run_geom for OFFSET_RANGE_B={OFFSET_RANGE_B} (top-k={TOP_K})")
    dir_b = _run_one_offset(OFFSET_RANGE_B)
    print(f"[DONE] B vectors: {dir_b}")

    # Compose vector file paths for split 1, top-k stack
    a_vec = dir_a / f"L{LAYER_TO_COMPARE}_split1_top{TOP_K}.stack.npy"
    b_vec = dir_b / f"L{LAYER_TO_COMPARE}_split1_top{TOP_K}.stack.npy"
    if not a_vec.exists():
        raise FileNotFoundError(f"Missing vectors for A: {a_vec}")
    if not b_vec.exists():
        raise FileNotFoundError(f"Missing vectors for B: {b_vec}")

    # Configure compare_vectors module and run
    compare_vectors.FILE_A = _rel_to_base(a_vec)
    compare_vectors.FILE_B = _rel_to_base(b_vec)
    print(f"[INFO] Comparing split-1 vectors:\n  A = {compare_vectors.FILE_A}\n  B = {compare_vectors.FILE_B}")
    compare_vectors.main()


if __name__ == "__main__":
    main()

